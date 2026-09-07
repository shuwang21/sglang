"""Search space for the triton fused MoE kernel: one point is one tile config."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from sglang.autotune.registry import register_space
from sglang.autotune.space.base import (
    Categorical,
    FeasibilityRule,
    Knob,
    Space,
)
from sglang.autotune.types import Point
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_tuning import (
    get_configs_compute_bound,
)

__all__ = ["MoeTileSpace", "BlockKDivisible"]


class BlockKDivisible(FeasibilityRule):
    """Block-quantized weights need BLOCK_SIZE_K to divide the quant block.

    A tile that straddles two scale blocks would need two scales for one
    accumulation, which the kernel cannot express.
    """

    name = "block_k_divisible"

    def __init__(self, block_k: int) -> None:
        self.block_k = block_k

    def check(self, point: Point, context: Mapping[str, Any]) -> Optional[str]:
        block_size_k = point.get("BLOCK_SIZE_K")
        if self.block_k % block_size_k == 0:
            return None
        return f"BLOCK_SIZE_K={block_size_k} does not divide block_k={self.block_k}"


@register_space("fused_moe_triton")
class MoeTileSpace(Space):
    """Tile knobs, taken apart from the tuner's own candidate list.

    Deriving the per-knob domains from ``get_configs_compute_bound()`` rather
    than restating them keeps this space and the standalone tuner searching the
    same set as the platform lists change.
    """

    def __init__(self, block_shape: Optional[Sequence[int]] = None) -> None:
        candidates = get_configs_compute_bound()
        rules = []
        if block_shape is not None and all(block_shape):
            rules.append(BlockKDivisible(block_k=block_shape[1]))
        super().__init__(rules=rules)
        self._knobs = tuple(
            Knob(
                name=name,
                domain=Categorical(values=tuple(sorted({c[name] for c in candidates}))),
                group="fused_moe_triton",
            )
            for name in candidates[0]
        )

    def knobs(self) -> Sequence[Knob]:
        return self._knobs
