# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""KV-write gating for training-free loop trial passes.

Kept separate from ``looped_transformer.py`` so that file stays byte-identical
to the implementation commit that ``benchmark/looped_transformer/reproduce.py``
verifies with ``check_source``.
"""

from __future__ import annotations

from dataclasses import dataclass


def loop_kv_cache_disabled(forward_batch) -> bool:
    """Whether the current pass is a loop trial pass that must not write KV."""

    return bool(getattr(forward_batch, "_loop_disable_kv_cache", False))


@dataclass(frozen=True)
class KVWritePlan:
    """Who writes this layer's KV cache entry in one attention call.

    ``fused``: the rotary kernel writes K/V (``fused_set_kv_buffer_arg``).
    ``attention``: the attention backend writes K/V (``save_kv_cache``).
    At most one of them is set; neither is set during a no-KV loop trial pass.
    """

    fused: bool
    attention: bool


def plan_kv_write(fused_available: bool, forward_batch) -> KVWritePlan:
    """Decide the KV write path once per attention call.

    Models that write KV inside the rotary embedding (gpt-oss on CUDA with a
    bf16 KV pool) cannot be stopped with ``save_kv_cache=False`` alone, so the
    decision must also gate the fused write.
    """

    if loop_kv_cache_disabled(forward_batch):
        return KVWritePlan(fused=False, attention=False)
    if fused_available:
        return KVWritePlan(fused=True, attention=False)
    return KVWritePlan(fused=False, attention=True)
