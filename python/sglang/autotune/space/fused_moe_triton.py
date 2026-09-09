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
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    _get_cuda_shared_memory_per_block_optin,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_tuning import (
    get_configs_compute_bound,
)

__all__ = ["MoeTileSpace", "BlockKDivisible", "SharedMemoryFits"]


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


class SharedMemoryFits(FeasibilityRule):
    """Reject tiles the device cannot stage, instead of paying a compile to learn.

    Triton stages both GEMM operands `num_stages - 1` deep, so a tile needs
    `(num_stages - 1) * (BLOCK_M + BLOCK_N) * BLOCK_K * itemsize` bytes. That
    formula is empirical -- it reproduces the `OutOfResources` figures triton
    reports, exactly, on every tile measured so far -- so it is applied only
    when the device limit is known, and a device with more shared memory than
    any tile needs is unaffected.

    On an L4 (99 KiB) this rejects roughly half the compute-bound candidate
    list, which was written for cards with twice the shared memory.
    """

    name = "shared_memory_fits"

    def __init__(self, limit_bytes: int, itemsize: int) -> None:
        self.limit_bytes = limit_bytes
        self.itemsize = itemsize

    def required_bytes(self, point: Point) -> int:
        stages = point.get("num_stages")
        operands = point.get("BLOCK_SIZE_M") + point.get("BLOCK_SIZE_N")
        return (stages - 1) * operands * point.get("BLOCK_SIZE_K") * self.itemsize

    def check(self, point: Point, context: Mapping[str, Any]) -> Optional[str]:
        needed = self.required_bytes(point)
        if needed <= self.limit_bytes:
            return None
        return f"needs {needed} B of shared memory, device has {self.limit_bytes} B"


@register_space("fused_moe_triton")
class MoeTileSpace(Space):
    """Tile knobs, taken apart from the tuner's own candidate list.

    Deriving the per-knob domains from ``get_configs_compute_bound()`` rather
    than restating them keeps this space and the standalone tuner searching the
    same set as the platform lists change.

    ``itemsize`` is the bytes per element both GEMM operands are staged at.
    ``None`` means the caller could not name one, and the shared-memory rule is
    then left out rather than applied with a guess.
    """

    def __init__(
        self,
        block_shape: Optional[Sequence[int]] = None,
        itemsize: Optional[int] = None,
    ) -> None:
        candidates = get_configs_compute_bound()
        rules: List[FeasibilityRule] = []
        if block_shape is not None and all(block_shape):
            rules.append(BlockKDivisible(block_k=block_shape[1]))
        smem_limit = _get_cuda_shared_memory_per_block_optin()
        # Both conditions are the caller's to establish: the device limit, and
        # an element size the formula can use. Skipping the rule costs a
        # compile per oversized tile, which the driver records as OOM;
        # guessing the element size would reject tiles that do fit, and those
        # are never measured and never appear in the report.
        if smem_limit is not None and itemsize is not None:
            rules.append(SharedMemoryFits(limit_bytes=smem_limit, itemsize=itemsize))
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
