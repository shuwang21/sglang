# Training-free loop on gpt-oss-20b: how to run

This tests the following loop configuration on `openai/gpt-oss-20b`. The
configuration comes from hearsay; this run is meant to verify it.

```json
{"start":12,"end":15,"k":2,"mode":"block","strategy":"block_anchored","anchor_beta":0.35,"cache_strategy":"first","decode_mode":"full"}
```

- The window `[12, 15)` is 0-based and half-open, so it covers layers 12, 13
  and 14 of 24.
- The semantics are ch-wan's `block_anchored`, unchanged. The natural pass
  writes the KV cache. The K−1 trial passes run with
  `_loop_disable_kv_cache=True` during prefill. The output is
  `lerp(h_K, natural, anchor_beta)`.
- `decode_mode=full` also loops decode tokens. In decode, the trial passes keep
  writing KV (inherited behaviour), so the slot holds the last pass's K/V. See
  "Known limitations".

Code: `GptOssModel.forward` and `GptOssAttention.forward_prepare/forward_core`
in `python/sglang/srt/models/gpt_oss.py`, and
`plan_kv_write` in `python/sglang/srt/models/looped_kv_write.py`.

## Quick path: one short allocation with `run_session.sh`

Machines are lent for at most 4 hours and their disks are wiped afterwards.
`run_session.sh` runs the whole session unattended:

- **2 or more GPUs:** the baseline and loop arms run at the same time. The
  first half of the GPUs each run a baseline server, the second half a loop
  server (one server per GPU, ports `PORT`, `PORT+1`, ...), and each arm's
  eval spreads its requests over its servers. With 4 GPUs: baseline on GPUs
  0 and 1, loop on GPUs 2 and 3.
- **1 GPU:** baseline first, then loop, on the same GPU.

The servers use exactly the flags of sections 1 and 2.

**Cut-off safe.** The eval runs (question, seed) pairs in a fixed hashed
order, so whatever finished is a uniform random subset of the questions. This
matters because MMLU-Pro is sorted by category: `--limit 500` alone picks 500
business questions. The paired report covers the questions both arms
finished, and says so when the run is incomplete. While the evals run, the
report and the tarball are refreshed every `REPORT_EVERY_MIN` (10) minutes.

**Shared GPU nodes: no credentials on the box.** The nodes are shared
between tenants. Leave no GitHub or Hugging Face token on them: no
`hf auth login`, no git credential helper, no `gh`. Run no coding agents
there. Everything below works without credentials:

- gpt-oss-20b and MMLU-Pro are not gated, so they download anonymously.
- Results are pulled from your laptop over ssh and committed there.

```bash
# on the node, after section 0 (venv active, HF_HOME set); for a 4h allocation
TIME_BUDGET_MIN=200 bash benchmark/looped_transformer/run_session.sh
```

`TIME_BUDGET_MIN` stops the evals that many minutes after the start and
writes the final report, leaving time to pull the results before the
allocation ends. With 1 GPU, the baseline gets half of what is left when it
starts, so the loop arm gets as much time.

Before any server starts, the script loads every task's dataset, so a data
problem fails within seconds. Servers bind to 127.0.0.1 only.

Default tasks:

| GPUs | task |
| --- | --- |
| 2+ | all of MMLU-Pro (12,032 questions), medium effort, seed 0 |
| 1 | `--per-category 36` (504 questions, all 14 categories), medium effort, seeds 0 and 1 |

The full run does not finish in 4h; it is meant to be cut off by the time
budget. Other knobs (see the header of `run_session.sh`): `GPUS` (default:
all GPUs), `CONCURRENCY` (in-flight requests per server, default 64), `PORT`.

To run a different plan, put one task per line in a file,
`<name>|<eval_gptoss.py run args>`, and pass `TASKS_FILE=that_file`.
`--per-category N` takes a fixed pseudo-random N questions from every
category; use it instead of `--limit` on MMLU-Pro.

**Sizing the next session.** The eval prints `req/min` and `out tok/s` per
arm. Use the first 15 minutes to compare concurrencies: run a short task
file at `CONCURRENCY=64` and at `CONCURRENCY=128` (watch the server log's
token usage for KV-cache pressure) and keep the faster one.

**AIME 2024 on one GPU** (`tasks/aime24_low_s5.txt`): 30 problems x 5
seeds at low effort, `max_tokens` 32768, all 150 requests in flight at once
(`--concurrency 150`); baseline first, then loop.

```bash
. ~/env.sh && TASKS_FILE=benchmark/looped_transformer/tasks/aime24_low_s5.txt bash benchmark/looped_transformer/run_session.sh
```

`--dataset aime24` reads `HuggingFaceH4/aime_2024`, `--dataset aime25`
`MathArena/aime_2025`, and `--dataset aime` both (60 problems, the year as
the category, so the report adds a per-year table); none is gated. The prompt
asks for a last line `Answer: N`; `\boxed{N}` is accepted too, and answers
compare as integers ("033" = "33"). With 30 problems the paired tests have
little power: a gain shows up as the 5-seed mean per problem, so read the
per-seed accuracies and the sign test together.

Other AIME task files, all with 64 seeds. Medium effort truncates at 32768
tokens (AIME 2024: 5.3% of baseline vs 8.3% of loop answers, which penalises
the loop), so the medium files for AIME 2025 use 65536; low effort does not
truncate at 32768.

| file | problems | effort | max_tokens |
| --- | --- | --- | ---: |
| `tasks/aime24_low_s64.txt` | AIME 2024 (30) | low | 32768 |
| `tasks/aime24_medium_s64.txt` | AIME 2024 (30) | medium | 32768 |
| `tasks/aime24_medium_s64_64k.txt` | AIME 2024 (30) | medium | 65536 |
| `tasks/aime25_low_s64.txt` | AIME 2025 (30) | low | 32768 |
| `tasks/aime25_medium_s64_64k.txt` | AIME 2025 (30) | medium | 65536 |
| `tasks/aime_low_s64.txt` | AIME 2024 + 2025 (60) | low | 32768 |
| `tasks/aime_medium_s64_64k.txt` | AIME 2024 + 2025 (60) | medium | 65536 |

After a fix to answer extraction (e.g. `Answer: \(25\)` was missed before),
re-score saved runs without a server; the first run keeps
`samples.jsonl.orig`. Work on a copy if `fetch_results.sh --watch` is still
syncing that directory, or rsync will put the old rows back.

```bash
python benchmark/looped_transformer/eval_gptoss.py rescore <baseline_dir> <loop_dir>
```

**GPQA (gated) without a login on the node:**

1. On your laptop, where you are logged in to Hugging Face, export the subset:

   ```bash
   python -c "from datasets import load_dataset as l; l('Idavidrein/gpqa','gpqa_diamond',split='train').to_csv('gpqa_diamond.csv')"
   ```

2. Copy it over: `scp gpqa_diamond.csv <node>:sglang-looped/`.
3. Add a task line to the tasks file:

   ```text
   gpqa_diamond_medium|--dataset gpqa --data-file gpqa_diamond.csv --reasoning-effort medium --seeds 0 1
   ```

GPQA's authors ask that its questions are not published online. The raw
outputs (`samples.jsonl`) often restate the question, so do not push GPQA
sessions to a public repository.

**Getting the results off the node: pull from the laptop.** On your laptop
(macOS Terminal, inside this checkout), run:

```bash
bash benchmark/looped_transformer/fetch_results.sh --watch <you>@<node>
```

**Leave the `--watch` running while a session runs.** It fetches every 10
minutes (`--watch=SEC` to change). Allocations are reclaimed at the end
without warning, the disk is wiped, and anything not fetched is lost.
Without `--watch` it fetches once.

The script does three things:

- It rsyncs `~/sglang-looped/results/gptoss_sessions/` from the node into
  `../looped-sessions/`. Tarballs are skipped.
- It commits every session to the results-only branch `shuwang/looped-results`
  and pushes it with your laptop's own git credentials. The commit goes through
  `push_results.sh`, which uses a temporary index, so this checkout is not
  touched.
- It prints the `SUMMARY.md` paths.

You can run it as often as you like while a session is running, for example
after every task. Pass a second argument if the checkout on the node is not
`~/sglang-looped`.

Read the committed results later:

```bash
git fetch origin shuwang/looped-results
```

```bash
# first time: a separate worktree that holds only results
git worktree add -b shuwang/looped-results ../looped-results origin/shuwang/looped-results
```

```bash
# later
git -C ../looped-results pull
```

If a node can push without a stored token, `PUSH_RESULTS=1` makes
`run_session.sh` push from the node after every report. The default is 0.
The session is also re-packed into `results/gptoss_sessions/session_<time>.tar.gz`
after every report. Each session directory contains:

| path | what it is |
| --- | --- |
| `SUMMARY.md`, `reports/*.md`/`*.json` | baseline-vs-loop reports, the results |
| `runs/<task>/{baseline,loop}/samples.jsonl` | every answer with its raw output, needed to re-analyse later |
| `servers/*.log`, `servers/*.log.cmd` | server logs and the exact launch command (proof the loop config was on) |
| `session_info.txt`, `packages.txt`, `nvidia-smi.txt`, `session.log` | git commit, versions, GPU, timeline |

To have the script upload the tarball itself, set `UPLOAD_CMD`. `{}` is replaced
by the tarball path, for example `UPLOAD_CMD='rclone copy {} myremote:gptoss/'`.

- **Ctrl-C:** the in-flight requests are dropped, the servers are stopped,
  the report is written and the tarball is re-packed.
- **Resume:** rerun with `SESSION_DIR=<that session dir>`. Pairs already
  answered are skipped, and the hashed order continues where it stopped. On a
  new node, first copy the session back from the laptop, then resume it:

  ```bash
  rsync -a ../looped-sessions/session_<time>/ <you>@<node>:sglang-looped/results/gptoss_sessions/session_<time>/
  ```

  ```bash
  # on the node
  SESSION_DIR=results/gptoss_sessions/session_<time> TIME_BUDGET_MIN=200 bash benchmark/looped_transformer/run_session.sh
  ```

`session.log` records how long each run took. Use it to size the next session.

## 0. Environment (GPU node)

A fresh node (empty disk, no system CUDA, no sudo) is set up by one script:

```bash
git clone -b shuwang/looped https://github.com/shuwang21/sglang.git ~/sglang-looped
```

```bash
bash ~/sglang-looped/benchmark/looped_transformer/setup_node.sh
```

It installs uv and Rust, creates the venv with the locked dependencies,
installs sglang in editable mode, and fixes the pip-wheel CUDA. The
fixes are:

- `CUDA_HOME` points at the venv's `nvidia/cu13`, which `deep_ep` needs.
- nvcc/crt/nvvm are pinned to 13.0.x, matching the 13.0 runtime headers.
  Otherwise flashinfer fails with "CUDA compiler and CUDA toolkit headers are
  incompatible".
- `lib64` and `libcudart.so` symlinks are added, without which JIT links fail
  with `cannot find -lcudart`.
- The JIT caches are cleared.

It then writes `~/env.sh`, checks the GPUs and nvcc, and downloads the model
and MMLU-Pro. Both are public, so no login is needed. The script is safe to
rerun. In every new shell:

```bash
. ~/env.sh
```

For the manual commands below, also set:

```bash
export MODEL=openai/gpt-oss-20b
export LOOP='{"start":12,"end":15,"k":2,"mode":"block","strategy":"block_anchored","anchor_beta":0.35,"cache_strategy":"first","decode_mode":"full"}'
```

Never log in to Hugging Face on a shared node. GPQA is gated: see "GPQA
(gated) without a login on the node" above.

## 1. Baseline server

```bash
python -m sglang.launch_server --model-path $MODEL --port 30000 \
  --tp-size 1 --attention-backend fa3 --disable-radix-cache \
  --disable-cuda-graph --log-level info \
  --reasoning-parser gpt-oss
```

`fa3` runs only on SM8x/SM9x (A100, H100, H200). On Blackwell (B200 is
SM100) use `--attention-backend triton` for both servers; `run_session.sh`
picks this on its own (`ATTENTION_BACKEND=auto`).

## 2. Loop server

It is identical to the baseline except for `--loop-config`:

```bash
python -m sglang.launch_server --model-path $MODEL --port 30001 \
  --tp-size 1 --attention-backend fa3 --disable-radix-cache \
  --disable-cuda-graph --log-level info \
  --reasoning-parser gpt-oss \
  --loop-config "$LOOP"
```

Notes:

- `--reasoning-parser gpt-oss` is my addition. It only changes how the
  response is split into `reasoning_content` and `content`; it does not change
  the model computation. It is needed for the "reasoning length" column.
  Without it, answer extraction still works (it takes the last `Answer: X` in
  the content), but reasoning length shows `n/a`. Use the same setting on both
  servers.
- With one GPU, run the two servers one after the other on the same card. Two
  servers can share one H200 only if each has `--mem-fraction-static 0.4` or so.
- Look for `--loop-config` / `loop_config` in the `--log-level info` startup
  dump to confirm that the loop server really received the configuration.

## 3. Sanity checks (before any real numbers)

`eval_gptoss.py run` writes `<output>/samples.jsonl`. The field `output` is the
final content, and `reasoning` is the reasoning text.

### 3a. k=1 must match baseline exactly

With `k <= 1`, `GptOssModel.forward` takes `_forward_standard`, which is the
original code path. Restart the loop server with
`--loop-config '{"start":12,"end":15,"k":1}'` and run both servers greedily and
serially. Use `--concurrency 1` so that batching cannot change the numerics.

```bash
for port in 30000 30001; do
  python benchmark/looped_transformer/eval_gptoss.py run \
    --base-url http://127.0.0.1:$port --dataset mmlu_pro --limit 50 \
    --reasoning-effort low --seeds 0 --temperature 0 --max-tokens 4096 \
    --concurrency 1 --output results/gptoss/k1_check_$port
done
```

```bash
python - results/gptoss/k1_check_30000 results/gptoss/k1_check_30001 <<'EOF'
import json, sys
def load(d):
    return {(r["id"], r["seed"]): r for r in map(json.loads, open(f"{d}/samples.jsonl"))}
a, b = load(sys.argv[1]), load(sys.argv[2])
diff = [k for k in a if (a[k].get("reasoning"), a[k].get("output")) != (b[k].get("reasoning"), b[k].get("output"))]
print(f"{len(a)} samples, {len(diff)} differ:", diff[:10])
EOF
```

Expected result: `0 differ`. If anything differs, stop and investigate. The
fallback is supposed to be byte-identical to the original code.

### 3b. The loop produces sane text

Restart the loop server with the real `$LOOP`, then:

```bash
python benchmark/looped_transformer/eval_gptoss.py run \
  --base-url http://127.0.0.1:30001 --dataset mmlu_pro --limit 3 \
  --reasoning-effort low --seeds 0 --temperature 0 --max-tokens 2048 \
  --concurrency 1 --output results/gptoss/loop_smoke3
```

```bash
python -c "import json; [print(r['id'], r['predicted'], r['finish_reason'], '\n', (r.get('reasoning') or '')[:600], '\n---\n', r['output'][:400], '\n=====') for r in map(json.loads, open('results/gptoss/loop_smoke3/samples.jsonl'))]"
```

The reasoning and the answer should be readable English, and each output
should end in an `Answer: X` line. Garbage or repeated tokens means the loop or
KV gating is broken. In that case, do not run the eval matrix.

## 4. Experiments

There are 2 reasoning efforts (low, medium) × 5 seeds (the default `0 1 2 3 4`)
× {baseline, loop}, with temperature 1.0, top-p 1.0 and max-tokens 16384.

Start with a `--per-category 15` smoke run on every cell, then do the full runs.
Do not use `--limit` alone on MMLU-Pro: the dataset is sorted by category, so
the first N questions are all business.

```bash
run_cell () {  # $1 port, $2 tag (baseline|loop), $3 dataset, $4 effort, rest = extra args
  port=$1; tag=$2; ds=$3; effort=$4; shift 4
  python benchmark/looped_transformer/eval_gptoss.py run \
    --base-url http://127.0.0.1:$port --dataset $ds --reasoning-effort $effort \
    --output results/gptoss/${ds}_${effort}_${tag}${SUFFIX} "$@"
}
```

```bash
# smoke on the baseline server (port 30000)
SUFFIX=_c15; for e in low medium; do run_cell 30000 baseline mmlu_pro $e --per-category 15; done
```

```bash
# smoke on the loop server (port 30001)
SUFFIX=_c15; for e in low medium; do run_cell 30001 loop mmlu_pro $e --per-category 15; done
```

```bash
python benchmark/looped_transformer/eval_gptoss.py summarize \
  results/gptoss/mmlu_pro_low_baseline_c15 results/gptoss/mmlu_pro_low_loop_c15 \
  --output results/gptoss/report_mmlu_pro_low_c15.md
```

Run the full runs the same way, with `SUFFIX=` and no `--per-category`.
Several servers with the same flags can serve one run:
`--base-url http://127.0.0.1:30000 http://127.0.0.1:30002` (`--concurrency` is
per server). For GPQA, use
`--dataset gpqa`, which defaults to `gpqa_main` (448 questions), or add
`--hf-config gpqa_diamond` (198 questions). The diamond subset is the one with
published gpt-oss numbers.

The `run` command resumes. If it is interrupted or some requests error, rerun
the same command with the same `--output`. It refuses to resume into a directory
that was written with different settings.

The report contains:

- Accuracy per seed, with the mean and standard deviation.
- Mean completion tokens and reasoning length.
- Truncation rate (`finish_reason == length`) and extraction-failure rate.
- A sign test and a Wilcoxon test on the per-question mean accuracy over the 5
  seeds.
- An exact McNemar test on seed 0.
- Accuracy per category, with a sign test per category.

A large difference in truncation or extraction failure between baseline and
loop means the accuracy delta is partly a formatting/length effect. Look at
those columns first.

To check whether the gain comes from longer reasoning, run
`analyze_length.py` on the same two directories. It relates each question's
accuracy change to its loop/baseline length ratio, and compares against
seed-vs-seed noise inside each arm when there are two or more seeds:

```bash
python benchmark/looped_transformer/analyze_length.py <baseline_dir> <loop_dir>
```

Seeds: SGLang applies the request `seed` only when the server runs with
`--enable-deterministic-inference`. Without it, the 5 seeds are 5 independent
samples that are labelled 0–4 but are not bit-reproducible. That is fine for
mean/std and for the paired tests. Do not expect reruns to match.

## 5. Check the baseline against public numbers first

Before reading anything into baseline-vs-loop, compare the **baseline** with
OpenAI's published gpt-oss-20b numbers at the same reasoning effort (model card
/ gpt-oss paper). Those numbers are for GPQA Diamond, so use
`--hf-config gpqa_diamond`, or `--data-file gpqa_diamond.csv` on a shared node. MMLU-Pro is not in the card, so use GPQA Diamond
for this check. If the baseline is clearly off, the eval setup is wrong. Fix
that first. Things to check:

- the prompt format
- `reasoning_effort` actually reaching the chat template (completion tokens
  should grow clearly from low to medium)
- the truncation rate at 16384 tokens
- the extraction failure rate
- whether the reasoning parser is on

## Known limitations

- **CUDA graph:** not verified. The commands use `--disable-cuda-graph`.
  `decode_mode=full` under decode CUDA graphs has never been run.
- **Batch decode:** the correctness of the loop with batched decode has not been
  verified. The k=1 check only proves that the fallback is unchanged.
- **Trial-pass KV, prefill:** with FA3, extend attention reads the current
  tokens' K/V from the paged cache instead of using the freshly computed `k`/`v`.
  A trial pass with KV writes disabled therefore attends over the natural pass's
  K/V for the in-batch tokens (only its queries change). This is a property of
  the inherited no-write design and is backend dependent: a backend that
  attends on the fresh `k`/`v` would behave differently.
- **Trial-pass KV, decode=full:** trial passes write KV (inherited from the
  Qwen3 integration), so the cache ends up with the last pass's K/V rather than
  the natural pass's K/V.
- **Unsupported setups:** `--loop-config` with EAGLE3/DFlash aux hidden-state
  capture raises `NotImplementedError`. So does the deferred MLP all-reduce
  path (attention TP > 1). Only TP=1 is intended. Mixed batches and
  `decode_mode=bypass` decode take the original path.
