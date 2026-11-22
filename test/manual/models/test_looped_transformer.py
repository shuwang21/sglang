import types
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.looped_kv_write import KVWritePlan, plan_kv_write
from sglang.srt.models.looped_transformer import (
    LoopConfig,
    LoopModelAdapter,
    LoopState,
    TrainingFreeLoopPlugin,
    blend_states,
    run_block_anchored,
)


def test_loop_config_accepts_json_and_aliases():
    config = LoopConfig.from_json('{"loop_start": 2, "loop_end": 4, "loop_k": 3}')
    assert config.start == 2
    assert config.end == 4
    assert config.k == 3


def test_loop_plugin_runs_model_independent_adapter():
    batch = type("Batch", (), {})()
    plugin = TrainingFreeLoopPlugin()
    state = LoopState(torch.tensor([[1.0]]), torch.zeros(1, 1))
    adapter = LoopModelAdapter(
        num_layers=4,
        start_layer=0,
        end_layer=4,
        is_first_rank=True,
        is_last_rank=True,
        is_prefill=True,
        is_mixed=False,
        initial_state=lambda: state,
        run_layer=lambda _layer_id, current: LoopState(
            current.stream() + 1.0, torch.zeros_like(current.stream())
        ),
        finalize=lambda current: current.stream(),
    )
    result = plugin.run(LoopConfig(start=1, end=3, k=2), adapter, batch)
    torch.testing.assert_close(result, torch.tensor([[3.0]]))


def test_loop_plugin_allows_decode_full_trial_cache_writes():
    batch = type("Batch", (), {})()
    plugin = TrainingFreeLoopPlugin()
    adapter = LoopModelAdapter(
        num_layers=2,
        start_layer=0,
        end_layer=2,
        is_first_rank=True,
        is_last_rank=True,
        is_prefill=False,
        is_mixed=False,
        initial_state=lambda: LoopState(torch.ones(1, 1), torch.zeros(1, 1)),
        run_layer=lambda _layer_id, current: LoopState(
            current.stream() + 1.0, torch.zeros_like(current.stream())
        ),
        finalize=lambda current: current.stream(),
    )
    result = plugin.run(
        LoopConfig(start=0, end=1, k=2, decode_mode="full"), adapter, batch
    )
    torch.testing.assert_close(result, torch.tensor([[2.0]]))
    assert getattr(batch, "_loop_decode_full", False) is False


def test_blend_states_preserves_residual_stream():
    base = LoopState(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
    target = LoopState(torch.tensor([[5.0, 6.0]]), torch.tensor([[7.0, 8.0]]))

    blended = blend_states(base, target, 0.25)

    torch.testing.assert_close(blended.stream(), torch.tensor([[6.0, 8.0]]))
    torch.testing.assert_close(blended.residual, target.residual)


def test_block_anchored_k1_is_natural_pass():
    state = LoopState(torch.tensor([[1.0]]), torch.tensor([[2.0]]))
    calls = []

    def run_block(current):
        calls.append(getattr(batch, "_loop_disable_kv_cache", False))
        return LoopState(current.stream() + 1.0, torch.zeros_like(current.stream()))

    batch = type("Batch", (), {})()
    result = run_block_anchored(state, run_block, 1, 0.5, batch)

    torch.testing.assert_close(result.stream(), torch.tensor([[4.0]]))
    assert calls == [False]


def test_block_anchored_disables_cache_for_trial_passes():
    state = LoopState(torch.tensor([[1.0]]), torch.tensor([[2.0]]))
    batch = type("Batch", (), {})()
    calls = []

    def run_block(current):
        calls.append(getattr(batch, "_loop_disable_kv_cache", False))
        return LoopState(current.stream() + 1.0, torch.zeros_like(current.stream()))

    run_block_anchored(state, run_block, 3, 0.5, batch)

    assert calls == [False, True, True]
    assert getattr(batch, "_loop_disable_kv_cache", False) is False


@pytest.mark.parametrize("beta", [-0.1, 1.1])
def test_block_anchored_rejects_invalid_anchor(beta):
    state = LoopState(torch.ones(1, 1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="anchor beta"):
        run_block_anchored(state, lambda current: current, 2, beta, object())


# ---------------------------------------------------------------------------
# gpt-oss: config, KV write gating, and the GptOssModel loop branch
# ---------------------------------------------------------------------------

GPT_OSS_LOOP_JSON = (
    '{"start":12,"end":15,"k":2,"mode":"block","strategy":"block_anchored",'
    '"anchor_beta":0.35,"cache_strategy":"first","decode_mode":"full"}'
)


def _batch(disable=None):
    batch = type("Batch", (), {})()
    if disable is not None:
        batch._loop_disable_kv_cache = disable
    return batch


def test_gpt_oss_loop_config_parses():
    config = LoopConfig.from_json(GPT_OSS_LOOP_JSON)
    assert config == LoopConfig(
        start=12,
        end=15,
        k=2,
        mode="block",
        strategy="block_anchored",
        anchor_beta=0.35,
        cache_strategy="first",
        decode_mode="full",
    )


@pytest.mark.parametrize(
    "fused_available, disable, expected",
    [
        (True, None, KVWritePlan(fused=True, attention=False)),
        (True, False, KVWritePlan(fused=True, attention=False)),
        (True, True, KVWritePlan(fused=False, attention=False)),
        (False, None, KVWritePlan(fused=False, attention=True)),
        (False, False, KVWritePlan(fused=False, attention=True)),
        (False, True, KVWritePlan(fused=False, attention=False)),
    ],
)
def test_plan_kv_write(fused_available, disable, expected):
    plan = plan_kv_write(fused_available, _batch(disable))
    assert plan == expected
    assert not (plan.fused and plan.attention)  # never written twice


def _gpt_oss():
    return pytest.importorskip("sglang.srt.models.gpt_oss")


class _RecordingAttention:
    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, forward_batch, sinks=None, save_kv_cache=True):
        self.calls.append(save_kv_cache)
        return q


def _fake_gpt_oss_attention(rotary_calls):
    q_size, kv_size = 4, 2

    def rotary_emb(positions, q, k, fused_set_kv_buffer_arg=None):
        rotary_calls.append(fused_set_kv_buffer_arg)
        return q, k

    return SimpleNamespace(
        qkv_proj=lambda h: (h.repeat(1, 2)[:, : q_size + 2 * kv_size], None),
        q_size=q_size,
        kv_size=kv_size,
        attn=_RecordingAttention(),
        rotary_emb=rotary_emb,
        sinks=None,
        o_proj=lambda x: (x, None),
    )


@pytest.mark.parametrize("fused_available", [True, False])
@pytest.mark.parametrize("disable", [False, True])
def test_gpt_oss_attention_kv_write_paths(monkeypatch, fused_available, disable):
    """Real GptOssAttention.forward_prepare/forward_core with stubbed kernels:
    the fused rotary write and the attention write follow one decision."""

    gpt_oss = _gpt_oss()
    sentinel = object()
    monkeypatch.setattr(gpt_oss, "_is_npu", False)
    monkeypatch.setattr(
        gpt_oss, "enable_fused_set_kv_buffer", lambda fb: fused_available
    )
    monkeypatch.setattr(
        gpt_oss, "create_fused_set_kv_buffer_arg", lambda **kwargs: sentinel
    )
    rotary_calls = []
    attn = _fake_gpt_oss_attention(rotary_calls)
    batch = _batch(disable)

    state = gpt_oss.GptOssAttention.forward_prepare(
        attn, torch.zeros(3), torch.ones(3, 4), batch
    )
    gpt_oss.GptOssAttention.forward_core(attn, state)

    fused_written = rotary_calls == [sentinel]
    attention_written = attn.attn.calls == [True]
    assert len(rotary_calls) == 1 and len(attn.attn.calls) == 1
    if disable:
        assert not fused_written and not attention_written
    elif fused_available:
        assert fused_written and not attention_written
    else:
        assert not fused_written and attention_written


def test_gpt_oss_attention_npu_keeps_attention_write(monkeypatch):
    gpt_oss = _gpt_oss()
    monkeypatch.setattr(gpt_oss, "_is_npu", True)
    monkeypatch.setattr(gpt_oss, "enable_fused_set_kv_buffer", lambda fb: True)
    rotary_calls = []
    attn = _fake_gpt_oss_attention(rotary_calls)
    state = gpt_oss.GptOssAttention.forward_prepare(
        attn, torch.zeros(3), torch.ones(3, 4), _batch()
    )
    gpt_oss.GptOssAttention.forward_core(attn, state)
    assert rotary_calls == [None] and attn.attn.calls == [True]


class _Mode:
    def __init__(self, prefill, mixed=False):
        self.prefill, self.mixed = prefill, mixed

    def is_prefill(self):
        return self.prefill

    def is_mixed(self):
        return self.mixed


class _AddLayer:
    """Decoder-layer stand-in: (hidden, residual) -> (delta, stream)."""

    def __init__(self, delta, log, batch):
        self.delta, self.log, self.batch = delta, log, batch

    def __call__(self, positions, hidden_states, forward_batch, residual):
        self.log.append(getattr(self.batch, "_loop_disable_kv_cache", False))
        stream = hidden_states if residual is None else hidden_states + residual
        return torch.full_like(stream, self.delta), stream


def _fake_gpt_oss_model(gpt_oss, batch, num_layers=6, capture=()):
    log = []
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=num_layers),
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        start_layer=0,
        end_layer=num_layers,
        layers=[_AddLayer(float(i + 1), log, batch) for i in range(num_layers)],
        layers_to_capture=list(capture),
        embed_tokens=lambda ids: ids.float().reshape(-1, 1),
        norm=lambda h, r=None: (h if r is None else (h + r, None)),
    )
    model._forward_standard = types.MethodType(
        gpt_oss.GptOssModel._forward_standard, model
    )
    return model, log


def _set_loop(monkeypatch, gpt_oss, raw):
    monkeypatch.setattr(gpt_oss, "get_model", lambda: SimpleNamespace(loop_config=raw))


def _run_model(gpt_oss, model, batch):
    return gpt_oss.GptOssModel.forward(
        model, torch.tensor([1, 2]), torch.zeros(2), batch
    )


def test_gpt_oss_model_loop_matches_block_anchored(monkeypatch):
    gpt_oss = _gpt_oss()
    raw = '{"start":2,"end":4,"k":3,"anchor_beta":0.35}'
    _set_loop(monkeypatch, gpt_oss, raw)
    batch = _batch()
    batch.forward_mode = _Mode(prefill=True)
    model, log = _fake_gpt_oss_model(gpt_oss, batch)
    got = _run_model(gpt_oss, model, batch)

    # Reference on the full stream: layer i adds (i + 1).
    emb = torch.tensor([[1.0], [2.0]])
    x0 = emb + 1 + 2  # layers 0, 1

    def block(s):  # layers 2, 3
        return s + 3 + 4

    natural = block(x0)
    h = torch.lerp(x0, natural, 1 / 3)
    for _ in range(2):
        h = torch.lerp(h, block(h), 1 / 3)
    h = torch.lerp(h, natural, 0.35)
    expected = h + 5 + 6  # layers 4, 5
    torch.testing.assert_close(got, expected)
    # prefix 2 + natural 2 + 2 trial passes x 2 + suffix 2 layer calls; only
    # the trial passes run with KV writes disabled.
    assert log == [False] * 4 + [True] * 4 + [False] * 2
    assert not getattr(batch, "_loop_disable_kv_cache", False)


@pytest.mark.parametrize(
    "raw, mode",
    [
        ("{}", _Mode(prefill=True)),
        ('{"start":2,"end":4,"k":1}', _Mode(prefill=True)),
        ('{"start":2,"end":4,"k":2}', _Mode(prefill=True, mixed=True)),
        ('{"start":2,"end":4,"k":2,"decode_mode":"bypass"}', _Mode(prefill=False)),
    ],
)
def test_gpt_oss_model_falls_back_to_standard_path(monkeypatch, raw, mode):
    gpt_oss = _gpt_oss()
    _set_loop(monkeypatch, gpt_oss, raw)
    batch = _batch()
    batch.forward_mode = mode
    model, log = _fake_gpt_oss_model(gpt_oss, batch)
    got = _run_model(gpt_oss, model, batch)
    model_ref, _ = _fake_gpt_oss_model(gpt_oss, batch)
    torch.testing.assert_close(
        got, model_ref._forward_standard(torch.tensor([1, 2]), torch.zeros(2), batch)
    )
    assert log == [False] * 6  # one pass over every layer, KV written


def test_gpt_oss_model_decode_full_loops(monkeypatch):
    gpt_oss = _gpt_oss()
    _set_loop(monkeypatch, gpt_oss, '{"start":2,"end":4,"k":2,"decode_mode":"full"}')
    batch = _batch()
    batch.forward_mode = _Mode(prefill=False)
    model, log = _fake_gpt_oss_model(gpt_oss, batch)
    _run_model(gpt_oss, model, batch)
    # decode_mode=full keeps the trial-pass KV writes (inherited semantics)
    assert log == [False] * 8


def test_gpt_oss_model_rejects_loop_with_capture(monkeypatch):
    gpt_oss = _gpt_oss()
    _set_loop(monkeypatch, gpt_oss, GPT_OSS_LOOP_JSON)
    batch = _batch()
    batch.forward_mode = _Mode(prefill=True)
    model, _ = _fake_gpt_oss_model(gpt_oss, batch, num_layers=24, capture=[3])
    with pytest.raises(NotImplementedError, match="capture"):
        _run_model(gpt_oss, model, batch)
