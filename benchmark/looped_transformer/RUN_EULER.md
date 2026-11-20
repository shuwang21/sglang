# Training-free looped transformer: Euler, Qwen3-4B-Instruct, MMLU-Pro

This guide reproduces the arXiv 2605.23872 Table 6 cell for Qwen3-4B-Instruct on
MMLU-Pro (+2.64):

```json
{"start":15,"end":19,"k":3,"mode":"block","strategy":"euler","cache_strategy":"first","decode_mode":"full"}
```

- Window `[15, 19)`: 0-based and half-open, so it covers the paper's layers 15–18.
- Update rule: `x_{k+1} = x_k + (1/K)·(g(x_k) − x_k)` for k = 0..K−1. Here `g` runs
  the 4 window layers in order on the full residual stream, with no anchor mixing.
- `decode_mode`: `full` loops decode tokens too, and `bypass` loops only prefill.
- `strategy: "naive"` (`x_{k+1} = g(x_k)`) is also accepted, for the sanity check below.

The code is in `python/sglang/srt/layers/looped.py`, and `Qwen3Model.forward` calls it.

## 0. Environment

```bash
pip install -e "python[all]"          # SGLang from this branch
pip install "lm-eval[api]==0.4.11"
```

Model ID: the HF repo for "Qwen3-4B-Instruct" is `Qwen/Qwen3-4B-Instruct-2507`.
Substitute a local path if you have one. Use the same ID for the server and for
lm-eval's tokenizer.

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507
export LOOP='{"start":15,"end":19,"k":3,"mode":"block","strategy":"euler","cache_strategy":"first","decode_mode":"full"}'
```

> `--disable-cuda-graph` was removed from SGLang main, and passing it is now a
> startup error. Use `--disable-decode-cuda-graph --disable-prefill-cuda-graph`
> instead. If `--loop-config` is set while either CUDA graph phase is still
> enabled, the server logs a warning. The looped path has only been checked in eager mode.

## 1. Servers

Run the baseline and the loop server with identical flags. The loop server adds
only `--loop-config`.

```bash
# baseline (port 30000)
python -m sglang.launch_server --model-path $MODEL --port 30000 \
  --dtype bfloat16 --tp-size 1 --disable-radix-cache \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph \
  --attention-backend flashinfer
```

```bash
# loop (port 30001)
python -m sglang.launch_server --model-path $MODEL --port 30001 \
  --dtype bfloat16 --tp-size 1 --disable-radix-cache \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph \
  --attention-backend flashinfer \
  --loop-config "$LOOP"
```

At startup the loop server should log `Training-free loop enabled: LoopConfig(...)`.
If it raises `NotImplementedError` from `looped.py`, the configuration hit an
unsupported case (see "Limitations").

Quick smoke test:

```bash
curl -s localhost:30001/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"
```

## 2. lm-eval: MMLU-Pro, 5-shot CoT

lm-eval's `mmlu_pro` task is 5-shot CoT `generate_until` with regex answer
extraction. Use the `local-completions` model with no chat template and pass
`--log_samples`.

```bash
run_eval () {  # $1 = port, $2 = output dir, rest = extra lm_eval args
  port=$1; out=$2; shift 2
  lm_eval --model local-completions \
    --model_args model=$MODEL,base_url=http://127.0.0.1:$port/v1/completions,tokenized_requests=False,tokenizer_backend=huggingface,num_concurrent=8,max_retries=3 \
    --tasks mmlu_pro --num_fewshot 5 --batch_size 1 \
    --log_samples --output_path $out "$@"
}
```

Smoke run first. `--limit` applies per subject, so 200 × 14 subjects:

```bash
run_eval 30000 results/baseline_l200 --limit 200
```

```bash
run_eval 30001 results/loop_l200     --limit 200
```

Then the full runs:

```bash
run_eval 30000 results/baseline_full
```

```bash
run_eval 30001 results/loop_full
```

Target scores (paper Table 3, `exact_match,custom-extract` aggregate):

| run      | MMLU-Pro |
|----------|---------:|
| baseline | 57.14    |
| loop     | 59.79    |

## 3. Sanity checks

1. **k=1 must match baseline question by question.** With `k=1` the helper skips
   the loop and takes the original code path, so answers must be identical.
   Use `num_concurrent=1` for both runs so batching cannot change the numerics.

   ```bash
   # restart the loop server with '{"start":15,"end":19,"k":1}', then:
   run_eval 30000 results/baseline_l200_c1 --limit 200   # with num_concurrent=1
   run_eval 30001 results/k1_l200_c1       --limit 200   # with num_concurrent=1
   ```

   Compare the logged samples:

   ```bash
   python - results/baseline_l200_c1 results/k1_l200_c1 <<'EOF'
   import json, sys, pathlib
   def load(d):
       out = {}
       for f in pathlib.Path(d).rglob("samples_*.jsonl"):
           task = f.name.split("samples_")[1].rsplit("_", 1)[0]
           for line in f.open():
               r = json.loads(line)
               out[(task, r["doc_id"])] = (r["filtered_resps"], r["resps"])
       return out
   a, b = load(sys.argv[1]), load(sys.argv[2])
   diff = [k for k in a if a[k] != b.get(k)]
   print(f"{len(a)} samples, {len(diff)} differ"); print(diff[:10])
   EOF
   ```

2. **Naive looping should lose accuracy clearly.** Restart the loop server with
   `"strategy":"naive"` (same window, k=3). It should score well below baseline. If
   naive scores close to Euler, the loop is probably not running. Check that
   decode throughput dropped, since the window then runs 3× per token.

3. **bypass vs full.** Optional: `"decode_mode":"bypass"` loops only prefill.
   Its score is expected to land between baseline and full.

## 4. Implementation notes (what to look at if numbers are off)

- **Residual stream** (`stream_full_value`, `stream_reset`): at the window entry,
  the complete stream `x_0 = pending + residual` comes from
  `ResidualStream.snapshot`. Only a plain add with no owed reduction or finalize
  is accepted. Before each round the stream is cleared and set to the *written*
  state on a private clone of `x_k` (`residual_batch.set_written`), and that
  tensor is passed as `hidden_states`. The first window layer's prepare then
  takes the written-stream path (`read.read(x)`, no add). The clone is needed
  because the next FFN's fused add+norm updates the written residual in place.
  After the 4 layers, `g(x_k)` is read with `snapshot` again. The update runs
  in fp32 and is cast back to bf16. `x_K` is written back the same way for
  layer 19, and the final norm accepts a written stream.
  Numerical note: the baseline fuses the residual add into the next norm (fp32
  accumulation). In the loop, the adds at the window boundary are done in bf16
  by `snapshot`, so even an exact fixed point would not be bit-identical to the
  baseline. That is why k=1 takes the original path instead.
- **KV** (`WindowKVBackup`): every round writes K/V normally at `out_cache_loc`,
  and later rounds overwrite the same slots. After round 1, each window layer's K
  and V at those slots are copied (`kv_layers.read(id).kv()` + `index_select`).
  After the last round they are written back through
  `kv_layers.layer(id).store(attn, KVWriteLoc.for_layer(batch, attn), k, v)`.
  This runs for prefill (EXTEND) and for decode when `decode_mode=full`. In every
  round, attention sees that round's own current-token K/V plus the untouched
  history, whatever the backend.

## Limitations / unsupported (raise, or take the original path)

- Original path: MIXED batches, IDLE, spec-decoding modes (TARGET_VERIFY /
  DRAFT_EXTEND), empty batches, `k=1`, and decode with `decode_mode=bypass`.
- Raise `NotImplementedError`: TP>1 (any owed all-reduce), a non-plain residual
  update (gated / mHC / ...), DeferredFinalize, TBO, a window not inside this PP
  rank, aux hidden-state capture (EAGLE3) inside the window, any KV pool other
  than a plain `MHATokenToKVPool` (MLA, FP8/FP4/MXFP8, hybrid/SWA, HND/5d layout),
  sliding-window or cross-attention layers.
- Not implemented: `cache_strategy=last`, `mode=layer`, other strategies, MoE/MLA models.
- CUDA graph: not supported. Prefill piecewise graphs (torch.compile) and decode
  graphs were never exercised with the loop.
- Only `Qwen3Model` (dense Qwen3, plus subclasses that inherit its `forward`) is wired.
