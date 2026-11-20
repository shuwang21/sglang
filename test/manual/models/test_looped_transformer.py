"""CPU-only tests for the training-free looped transformer helper.

Run from the repo root:
    python -m pytest -q test/manual/models/test_looped_transformer.py

Covers the parts that do not need a GPU: config parsing, the Euler / naive
update, KV backup -> overwrite -> restore with a fake KV router, and the
residual-stream takeover / write-back against the real ``ResidualStream``.
The real attention backends, the MHA pool's write kernel and Qwen3 itself are
not exercised here (see benchmark/looped_transformer/RUN_EULER.md).
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.batch import start, stream_of
from sglang.srt.layers.looped import (
    LoopConfig,
    WindowKVBackup,
    backup_kv_slots,
    loop_update,
    run_layers_looped,
    run_loop,
    stream_full_value,
    stream_reset,
)

PAPER_JSON = (
    '{"start":15,"end":19,"k":3,"mode":"block","strategy":"euler",'
    '"cache_strategy":"first","decode_mode":"full"}'
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_paper_cell():
    config = LoopConfig.from_json(PAPER_JSON)
    assert config == LoopConfig(start=15, end=19, k=3)
    config.validate(num_layers=36)


@pytest.mark.parametrize("raw", ["{}", "", None])
def test_config_empty_is_none(raw):
    assert LoopConfig.from_json(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"start":4,"end":4,"k":2}',
        '{"start":0,"end":2,"k":0}',
        '{"start":0,"end":2,"k":2,"mode":"layer"}',
        '{"start":0,"end":2,"k":2,"cache_strategy":"last"}',
        '{"start":0,"end":2,"k":2,"strategy":"block_anchored"}',
        '{"start":0,"end":2,"k":2,"decode_mode":"x"}',
        '{"start":0,"end":2,"k":2,"bogus":1}',
        "[1]",
    ],
)
def test_config_rejects(raw):
    with pytest.raises(ValueError):
        LoopConfig.from_json(raw)


def test_config_window_exceeds_model():
    with pytest.raises(ValueError):
        LoopConfig(start=30, end=40, k=2).validate(num_layers=36)


# ---------------------------------------------------------------------------
# Euler update with a fake block g
# ---------------------------------------------------------------------------


def _fake_block(dim=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w1 = torch.randn(dim, 2 * dim, generator=gen) * 0.3
    w2 = torch.randn(2 * dim, dim, generator=gen) * 0.3

    def g(x):
        return x + torch.tanh(x @ w1) @ w2

    return g


def test_euler_matches_formula():
    g = _fake_block()
    x0 = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    K = 3
    expected = x0.clone()
    for _ in range(K):
        expected = expected + (g(expected) - expected) / K
    torch.testing.assert_close(run_loop(x0, g, K, "euler"), expected)


def test_euler_k1_is_g():
    g = _fake_block()
    x0 = torch.randn(5, 8, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(run_loop(x0, g, 1, "euler"), g(x0))


def test_naive_is_composition():
    g = _fake_block()
    x0 = torch.randn(5, 8, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(run_loop(x0, g, 3, "naive"), g(g(g(x0))))


def test_update_keeps_dtype():
    x = torch.randn(3, 4).bfloat16()
    g = torch.randn(3, 4).bfloat16()
    assert loop_update(x, g, 3).dtype == torch.bfloat16


def test_run_block_gets_private_copy():
    x0 = torch.ones(2, 2)

    def g(x):
        x.mul_(100)  # a block that clobbers its input must not touch x_k
        return torch.zeros(2, 2)

    out = run_loop(x0, g, 2, "euler")
    torch.testing.assert_close(x0, torch.ones(2, 2))
    torch.testing.assert_close(out, torch.full((2, 2), 0.25))


# ---------------------------------------------------------------------------
# KV backup -> overwrite -> restore
# ---------------------------------------------------------------------------


class _FakeKVLayer:
    def __init__(self, k_buf, v_buf):
        self.k_buf, self.v_buf, self.plain = k_buf, v_buf, True

    def kv(self):
        return self.k_buf, self.v_buf

    def store(self, layer, loc_info, k, v, *args, **kwargs):
        self.k_buf[loc_info.loc] = k
        self.v_buf[loc_info.loc] = v


class _FakeRouter:
    def __init__(self, layer_ids, slots=16, heads=2, head_dim=4):
        self.layers = {
            i: _FakeKVLayer(
                torch.randn(slots, heads, head_dim), torch.randn(slots, heads, head_dim)
            )
            for i in layer_ids
        }

    def layer(self, i):
        return self.layers[i]

    read = layer


def _fake_batch(loc):
    return SimpleNamespace(
        out_cache_loc=loc,
        out_cache_loc_is_physical=False,
        encoder_out_cache_loc=None,
        residual_stream=None,
    )


def test_backup_kv_slots_copies():
    k = torch.randn(6, 2, 3)
    v = torch.randn(6, 2, 3)
    loc = torch.tensor([4, 1])
    kb, vb = backup_kv_slots(k, v, loc)
    k[4] = 0
    torch.testing.assert_close(kb[0], kb[0].clone())
    assert not torch.equal(kb[0], k[4])
    torch.testing.assert_close(vb, v[loc])


def test_kv_backup_restores_first_round():
    window = (2, 3)
    router = _FakeRouter(window)
    loc = torch.tensor([7, 3, 11])
    fb = _fake_batch(loc)
    attn = {i: SimpleNamespace(is_cross_attention=False, layer_id=i) for i in window}
    initial = {i: (l.k_buf.clone(), l.v_buf.clone()) for i, l in router.layers.items()}

    backup = WindowKVBackup(window, attn)
    first = {}
    for rnd in range(3):
        for i in window:
            k = torch.full((3, 2, 4), float(10 * rnd + i))
            router.layer(i).store(attn[i], SimpleNamespace(loc=loc), k, -k)
            if rnd == 0:
                first[i] = k
        if rnd == 0:
            backup.save(fb, router)
    backup.restore(fb, router)

    other = torch.ones(16, dtype=torch.bool)
    other[loc] = False
    for i in window:
        layer = router.layers[i]
        torch.testing.assert_close(layer.k_buf[loc], first[i])
        torch.testing.assert_close(layer.v_buf[loc], -first[i])
        torch.testing.assert_close(layer.k_buf[other], initial[i][0][other])
        torch.testing.assert_close(layer.v_buf[other], initial[i][1][other])


# ---------------------------------------------------------------------------
# Residual stream takeover / write-back with the real ResidualStream
# ---------------------------------------------------------------------------


def _stage(h, fb, f):
    """One boundary stage the way the real prepare/finish use the stream: an
    in-place add of the pending contribution into the residual (like
    fused_add_rmsnorm), write, compute, record a plain-add contribution."""
    stream = stream_of(fb)
    value, residual = stream.input(h)
    if residual is None:
        res = value  # initial (embedding) or written stream
    else:
        res = residual.add_(value)
    stream.write(res)
    return stream.record(f(res), PLAIN_ADD)


def _make_layers(n, dim, seed=0):
    gen = torch.Generator().manual_seed(seed)
    ws = [
        (
            torch.randn(dim, dim, generator=gen) * 0.2,
            torch.randn(dim, dim, generator=gen) * 0.2,
        )
        for _ in range(n)
    ]
    fns = [
        (lambda r, w=w[0]: torch.tanh(r @ w), lambda r, w=w[1]: torch.sin(r @ w))
        for w in ws
    ]
    return fns


def _layer_runner(fns, fb, kv_hook=None):
    def run_layer(i, h):
        fa, ff = fns[i]
        if kv_hook is not None:
            inner = fa

            def fa(r, inner=inner, i=i):
                return inner(r) + kv_hook(i, r)

        h = _stage(h, fb, fa)
        return _stage(h, fb, ff)

    return run_layer


def _ref_layer(fns, i, x):
    fa, ff = fns[i]
    x = x + fa(x)
    return x + ff(x)


def test_stream_snapshot_and_reset():
    fb = _fake_batch(torch.tensor([0]))
    start(fb)
    emb = torch.randn(3, 4)
    h = _stage(emb, fb, lambda r: r * 2)
    full = stream_full_value(h, fb)
    torch.testing.assert_close(full, emb * 3)

    x = torch.randn(3, 4)
    h = stream_reset(x, fb)
    stream = stream_of(fb)
    assert stream.pending is None and stream.residual is h and h is not x
    torch.testing.assert_close(stream_full_value(h, fb), x)
    # the next stage reads exactly x, and mutating the stream leaves x alone
    seen = {}
    h = _stage(h, fb, lambda r: seen.setdefault("r", r.clone()) * 0 + 1)
    torch.testing.assert_close(seen["r"], x)
    h = _stage(h, fb, lambda r: r * 0)
    torch.testing.assert_close(stream_full_value(h, fb), x + 1)


@pytest.mark.parametrize("strategy", ["euler", "naive"])
def test_looped_stack_matches_reference(strategy):
    n, dim, K = 6, 8, 3
    config = LoopConfig(start=2, end=4, k=K, strategy=strategy)
    fns = _make_layers(n, dim)
    fb = _fake_batch(torch.tensor([1, 5]))
    start(fb)
    emb = torch.randn(2, dim, generator=torch.Generator().manual_seed(9))
    # The stack's first stage adds into the embedding in place, as the real
    # fused add+norm does; keep the reference input separately.
    emb_ref = emb.clone()

    router = _FakeRouter(range(config.start, config.end), heads=1, head_dim=dim)
    attn = {
        i: SimpleNamespace(is_cross_attention=False, layer_id=i)
        for i in range(config.start, config.end)
    }
    kv_log = {}

    def kv_hook(i, r):
        # Write this round's "K/V" at out_cache_loc, then read it back from
        # the cache (FA3-like); history slots must be untouched.
        if i not in router.layers:
            return 0
        k = r.reshape(-1, 1, dim)
        router.layer(i).store(attn[i], SimpleNamespace(loc=fb.out_cache_loc), k, k * 2)
        kv_log.setdefault(i, []).append(k.clone())
        return router.layer(i).k_buf[fb.out_cache_loc].reshape(-1, dim) * 0

    run_layer = _layer_runner(fns, fb, kv_hook)
    out = run_layers_looped(
        config,
        start_layer=0,
        end_layer=n,
        hidden_states=emb,
        forward_batch=fb,
        run_layer=run_layer,
        backup=WindowKVBackup(range(config.start, config.end), attn),
        router=router,
    )
    got = stream_full_value(out, fb)

    x = emb_ref
    for i in range(config.start):
        x = _ref_layer(fns, i, x)

    def g(z):
        for i in range(config.start, config.end):
            z = _ref_layer(fns, i, z)
        return z

    for _ in range(K):
        x = x + (g(x) - x) / K if strategy == "euler" else g(x)
    for i in range(config.end, n):
        x = _ref_layer(fns, i, x)
    torch.testing.assert_close(got, x, rtol=1e-5, atol=1e-5)

    # K rounds happened, and the cache holds round 1's K/V for each window layer
    for i in range(config.start, config.end):
        assert len(kv_log[i]) == K
        assert not torch.equal(kv_log[i][0], kv_log[i][-1])
        torch.testing.assert_close(
            router.layers[i].k_buf[fb.out_cache_loc], kv_log[i][0]
        )
        torch.testing.assert_close(
            router.layers[i].v_buf[fb.out_cache_loc], kv_log[i][0] * 2
        )
