# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training-free looped transformer (arXiv 2605.23872), block mode.

A window of decoder layers ``[start, end)`` is treated as one map ``g`` on the
complete residual stream ``x`` and iterated ``K`` times:

    euler: x_{k+1} = x_k + (1/K) * (g(x_k) - x_k)
    naive: x_{k+1} = g(x_k)

KV semantics (paper appendix C, cache_strategy=first): every round writes the
current tokens' K/V for the window layers at the batch's ``out_cache_loc`` as
usual (so each round's attention sees its own current-token K/V plus the
canonical history, independent of whether the backend reads K/V from the
cache or from its arguments). After round 1 the window layers' slots are
backed up; after the last round the backup is written back, so the cache
holds exactly one K/V per (window layer, token): round 1's.

Only the simplest configuration is supported: TP=1, plain residual add, no
TBO / PP-crossing window, a plain (unquantized, NHD) MHA KV pool. Anything
else raises.

Everything torch-only lives at module level without importing the SGLang
runtime, so the algorithmic pieces are testable on CPU.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, fields
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)

_STRATEGIES = ("euler", "naive")
_DECODE_MODES = ("full", "bypass")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopConfig:
    """``--loop-config`` JSON. ``[start, end)`` are 0-based global layer ids."""

    start: int
    end: int
    k: int = 1
    mode: str = "block"
    strategy: str = "euler"
    cache_strategy: str = "first"
    decode_mode: str = "full"

    @classmethod
    def from_json(cls, raw: Union[str, dict, None]) -> Optional["LoopConfig"]:
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if raw in ("", "{}", "null"):
                return None
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--loop-config must be valid JSON: {exc}") from exc
        else:
            value = raw
        if value in (None, {}):
            return None
        if not isinstance(value, dict):
            raise ValueError("--loop-config must be a JSON object")
        aliases = {"loop_start": "start", "loop_end": "end", "loop_k": "k"}
        value = {aliases.get(key, key): item for key, item in value.items()}
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                "--loop-config has unknown fields: " + ", ".join(sorted(unknown))
            )
        config = cls(**value)
        config.validate()
        return config

    def validate(self, num_layers: Optional[int] = None) -> None:
        for name in ("start", "end", "k"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"--loop-config {name} must be an int")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"loop window must satisfy 0 <= start < end, got "
                f"[{self.start}, {self.end})"
            )
        if num_layers is not None and self.end > num_layers:
            raise ValueError(
                f"loop window [{self.start}, {self.end}) exceeds the model's "
                f"{num_layers} layers"
            )
        if self.k < 1:
            raise ValueError(f"loop k must be >= 1, got {self.k}")
        if self.mode != "block":
            raise ValueError(f"only mode=block is supported, got {self.mode!r}")
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_STRATEGIES}, got {self.strategy!r}"
            )
        if self.cache_strategy != "first":
            raise ValueError(
                f"only cache_strategy=first is supported, got {self.cache_strategy!r}"
            )
        if self.decode_mode not in _DECODE_MODES:
            raise ValueError(
                f"decode_mode must be one of {_DECODE_MODES}, got {self.decode_mode!r}"
            )


@functools.lru_cache(maxsize=8)
def parse_loop_config(raw: str) -> Optional[LoopConfig]:
    return LoopConfig.from_json(raw)


def get_loop_config() -> Optional[LoopConfig]:
    """The published ``model.loop_config`` (None when unset or unpublished)."""
    from sglang.srt.runtime_context import get_context, get_model

    if not get_context().is_config_namespace_published("model"):
        return None
    return parse_loop_config(getattr(get_model(), "loop_config", "{}") or "{}")


@functools.lru_cache(maxsize=1)
def _warn_once(config: LoopConfig) -> None:
    try:
        from sglang.srt.model_executor.cuda_graph_config import (
            cuda_graph_fully_disabled,
        )

        graphs_off = cuda_graph_fully_disabled()
    except Exception:  # best effort
        graphs_off = False
    if not graphs_off:
        logger.warning(
            "--loop-config is set but CUDA graph is not fully disabled; the "
            "looped path is only validated eagerly. Pass "
            "--disable-decode-cuda-graph --disable-prefill-cuda-graph."
        )
    logger.info("Training-free loop enabled: %s", config)


def check_loop_config_at_init(num_layers: int) -> Optional[LoopConfig]:
    """Validate once at model construction and warn about CUDA graph."""
    config = get_loop_config()
    if config is None:
        return None
    config.validate(num_layers)
    _warn_once(config)
    return config


# ---------------------------------------------------------------------------
# Pure-torch algorithm pieces
# ---------------------------------------------------------------------------


def loop_update(
    x: torch.Tensor, g: torch.Tensor, k: int, strategy: str = "euler"
) -> torch.Tensor:
    """One outer step on the full residual stream, computed in fp32."""
    if strategy == "naive":
        return g.clone()
    if strategy != "euler":
        raise ValueError(f"unknown loop strategy {strategy!r}")
    xf = x.float()
    out = xf + (g.float() - xf) * (1.0 / k)
    return out.to(x.dtype)


def run_loop(
    x0: torch.Tensor,
    run_block: Callable[[torch.Tensor], torch.Tensor],
    k: int,
    strategy: str = "euler",
    after_first_round: Optional[Callable[[], None]] = None,
) -> torch.Tensor:
    """Iterate ``run_block`` (g) K times from x0. ``run_block`` receives a
    private copy of x_k and returns the complete g(x_k)."""
    x = x0
    for round_idx in range(k):
        g = run_block(x.clone())
        if round_idx == 0 and after_first_round is not None:
            after_first_round()
        x = loop_update(x, g, k, strategy)
    return x


def backup_kv_slots(
    k_buf: torch.Tensor, v_buf: torch.Tensor, loc: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copies of the rows ``loc`` of a token-major (NHD) K/V buffer pair."""
    loc = loc.to(device=k_buf.device, dtype=torch.long)
    return k_buf.index_select(0, loc).clone(), v_buf.index_select(0, loc).clone()


# ---------------------------------------------------------------------------
# KV pool backup / restore
# ---------------------------------------------------------------------------


class WindowKVBackup:
    """Backs up and restores the window layers' K/V at ``out_cache_loc``.

    Reads go through ``kv_layers.read(id).kv()`` and writes through
    ``kv_layers.layer(id).store(layer, KVWriteLoc.for_layer(...), k, v)``,
    the same interface the attention backends use.
    """

    def __init__(self, layer_ids: Sequence[int], attn_layers: Dict[int, object]):
        self.layer_ids = tuple(layer_ids)
        self.attn_layers = attn_layers
        self.saved: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def check_supported(layer_ids, attn_layers):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor.forward_context import (
            get_kv_layers,
            get_token_to_kv_pool,
        )

        pool = get_token_to_kv_pool()
        if type(pool) is not MHATokenToKVPool:
            raise NotImplementedError(
                f"looped transformer supports only a plain MHATokenToKVPool, "
                f"got {type(pool).__name__}"
            )
        if pool.is_quantized_kv_cache or pool.store_dtype != pool.dtype:
            raise NotImplementedError(
                "looped transformer does not support a quantized KV cache"
            )
        if getattr(pool, "kv_cache_layout", "nhd") != "nhd" or getattr(
            pool, "use_hnd", False
        ):
            raise NotImplementedError("looped transformer requires the NHD KV layout")
        router = get_kv_layers()
        for layer_id in layer_ids:
            if not router.layer(layer_id).plain:
                raise NotImplementedError(
                    "looped transformer requires a plain KV router"
                )
            attn = attn_layers[layer_id]
            if getattr(attn, "sliding_window_size", -1) not in (None, -1):
                raise NotImplementedError(
                    "looped transformer does not support sliding window"
                )
            if getattr(attn, "is_cross_attention", False):
                raise NotImplementedError(
                    "looped transformer does not support cross attention"
                )

    def save(self, forward_batch, router=None) -> None:
        if router is None:
            from sglang.srt.model_executor.forward_context import get_kv_layers

            router = get_kv_layers()
        loc = forward_batch.out_cache_loc
        for layer_id in self.layer_ids:
            k_buf, v_buf = router.read(layer_id).kv()
            if k_buf.dim() != 3:
                raise NotImplementedError(
                    f"unexpected K buffer rank {k_buf.dim()} for layer {layer_id}"
                )
            self.saved[layer_id] = backup_kv_slots(k_buf, v_buf, loc)

    def restore(self, forward_batch, router=None) -> None:
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        if router is None:
            from sglang.srt.model_executor.forward_context import get_kv_layers

            router = get_kv_layers()
        for layer_id in self.layer_ids:
            k, v = self.saved.pop(layer_id)
            attn = self.attn_layers[layer_id]
            router.layer(layer_id).store(
                attn, KVWriteLoc.for_layer(forward_batch, attn), k, v
            )


# ---------------------------------------------------------------------------
# Residual stream adapter
# ---------------------------------------------------------------------------


def stream_full_value(hidden_states, forward_batch) -> torch.Tensor:
    """The complete residual stream x (pending contribution + residual), as a
    fresh tensor. Only plain-add, already-complete contributions."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    stream = residual_batch.stream_of(forward_batch)
    if stream.transport is not None:
        raise NotImplementedError("looped transformer: unconsumed row transport")
    pending = stream.pending
    if pending is not None:
        if pending.owed is not None:
            raise NotImplementedError(
                "looped transformer requires a complete contribution "
                f"(got owed {type(pending.owed).__name__}; TP>1 / fused "
                "all-reduce / deferred finalize are not supported)"
            )
        if not pending.update.is_plain_add:
            raise NotImplementedError(
                "looped transformer requires a plain residual add"
            )
    return stream.snapshot(hidden_states)


def stream_reset(x: torch.Tensor, forward_batch) -> torch.Tensor:
    """Make the stream 'written' with a private copy of x and return the
    tensor the next layer must receive as hidden_states. The copy matters:
    the next FFN's fused add+norm updates the written residual in place."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

    residual_batch.clear(forward_batch)
    return residual_batch.set_written(x.clone(), forward_batch)


# ---------------------------------------------------------------------------
# Layer-stack driver
# ---------------------------------------------------------------------------


def should_loop(config: Optional[LoopConfig], forward_batch, num_tokens: int) -> bool:
    from sglang.srt.model_executor.forward_mode import ForwardMode

    if config is None or config.k <= 1 or num_tokens == 0:
        return False
    mode = forward_batch.forward_mode
    if mode == ForwardMode.EXTEND:
        return True
    if mode == ForwardMode.DECODE:
        return config.decode_mode == "full"
    # MIXED, IDLE, spec-decoding and other modes keep the original path.
    return False


def run_decoder_layers(
    *,
    start_layer: int,
    end_layer: int,
    hidden_states,
    forward_batch,
    num_tokens: int,
    run_layer: Callable[[int, object], object],
    attn_of: Callable[[int], object],
    capture_layers: Sequence[int] = (),
):
    """Run local layers ``[start_layer, end_layer)``, looping the configured
    window. ``run_layer(i, hidden)`` runs decoder layer i on the residual
    stream; ``attn_of(i)`` returns layer i's RadixAttention."""
    config = get_loop_config()
    if not should_loop(config, forward_batch, num_tokens):
        for i in range(start_layer, end_layer):
            hidden_states = run_layer(i, hidden_states)
        return hidden_states

    if not (start_layer <= config.start and config.end <= end_layer):
        raise NotImplementedError(
            f"loop window [{config.start}, {config.end}) must lie in this "
            f"pipeline rank's layers [{start_layer}, {end_layer})"
        )
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size != 1:
        raise NotImplementedError("looped transformer supports only TP=1")
    if getattr(forward_batch, "tbo_split_seq_index", None) is not None:
        raise NotImplementedError("looped transformer does not support TBO")
    window = range(config.start, config.end)
    if any(i in window for i in capture_layers):
        raise NotImplementedError(
            "looped transformer does not support aux hidden-state capture in the window"
        )
    attn_layers = {i: attn_of(i) for i in window}
    WindowKVBackup.check_supported(window, attn_layers)
    return run_layers_looped(
        config,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        run_layer=run_layer,
        backup=WindowKVBackup(window, attn_layers),
    )


def run_layers_looped(
    config: LoopConfig,
    *,
    start_layer: int,
    end_layer: int,
    hidden_states,
    forward_batch,
    run_layer: Callable[[int, object], object],
    backup,
    router=None,
):
    """The looped layer stack, after all support checks. ``backup`` provides
    save/restore of the window layers' K/V (``WindowKVBackup``)."""
    window = range(config.start, config.end)
    for i in range(start_layer, config.start):
        hidden_states = run_layer(i, hidden_states)

    # 1. Take over the stream at the window entry: complete x_0.
    x0 = stream_full_value(hidden_states, forward_batch)

    def run_block(x: torch.Tensor) -> torch.Tensor:
        # 2. Reset the stream to x_k; 3. run g; read back the complete g(x_k).
        h = stream_reset(x, forward_batch)
        for i in window:
            h = run_layer(i, h)
        return stream_full_value(h, forward_batch)

    x_final = run_loop(
        x0,
        run_block,
        config.k,
        config.strategy,
        after_first_round=lambda: backup.save(forward_batch, router),
    )
    # cache_strategy=first: rounds 2..K overwrote the same slots; put back
    # round 1's K/V.
    backup.restore(forward_batch, router)

    # 4. Write x_K back as a written stream for layer `end` (or the final norm).
    hidden_states = stream_reset(x_final, forward_batch)
    for i in range(config.end, end_layer):
        hidden_states = run_layer(i, hidden_states)
    return hidden_states
