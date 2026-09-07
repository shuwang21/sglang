"""Per-shape MoE config reporter: the artifact the kernel actually reads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from sglang.autotune.driver.fused_moe_triton import FusedMoeTritonDriver
from sglang.autotune.registry import register_reporter
from sglang.autotune.report import Reporter
from sglang.autotune.task import TuneResult
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_moe_configs,
)

__all__ = ["MoeConfigReporter"]


def _version_dir() -> str:
    """The `triton_X_Y_Z` level `get_moe_configs` resolves under `configs/`.

    A config tuned on one Triton can lose performance on another, so the
    version is part of the lookup key, not decoration: writing beside it means
    the runtime never finds the file.
    """
    import triton

    return f"triton_{triton.__version__.replace('.', '_')}"


@register_reporter("moe_config")
class MoeConfigReporter(Reporter):
    """Writes ``configs/E=..,N=..,device_name=...json`` and reads it back.

    The read-back is the point: a tuned config the runtime cannot find is worse
    than none, because the run reports a speedup nobody gets. It goes through
    ``get_moe_configs`` under ``SGLANG_MOE_CONFIG_DIR``, the same path serving
    uses.
    """

    def __init__(self, validate: bool = True) -> None:
        self.validate = validate

    def emit(self, result: TuneResult, output_dir: Path) -> List[Path]:
        driver = result.task.driver
        if not isinstance(driver, FusedMoeTritonDriver):
            raise TypeError(
                f"moe_config reporter needs a MoE driver, got {driver.name}"
            )

        best: Dict[str, Dict] = {}
        for bucket, evaluation in sorted(result.best_by_bucket.items()):
            num_tokens = evaluation.measurement.trial.workload.params["num_tokens"]
            best[str(num_tokens)] = dict(driver.render_config(evaluation.point))
        if not best:
            return []

        config_dir = output_dir / "configs" / _version_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / driver.shape.config_filename
        path.write_text(json.dumps(best, indent=4) + "\n", encoding="utf-8")

        if self.validate:
            _assert_loadable(driver, output_dir, expected=sorted(int(k) for k in best))
        return [path]


def _assert_loadable(
    driver: FusedMoeTritonDriver, output_dir: Path, expected: List[int]
) -> None:
    shape = driver.shape
    dtype_str = _dtype_str(shape)
    block_n, block_k = shape.block_shape or (0, 0)
    previous = os.environ.get("SGLANG_MOE_CONFIG_DIR")
    os.environ["SGLANG_MOE_CONFIG_DIR"] = str(output_dir)
    get_moe_configs.cache_clear()
    try:
        loaded = get_moe_configs(
            shape.num_experts,
            _runtime_n(shape),
            dtype_str,
            block_n=block_n,
            block_k=block_k,
            per_channel_quant=shape.per_channel_quant,
        )
    finally:
        if previous is None:
            os.environ.pop("SGLANG_MOE_CONFIG_DIR", None)
        else:
            os.environ["SGLANG_MOE_CONFIG_DIR"] = previous
        get_moe_configs.cache_clear()

    if loaded is None:
        raise RuntimeError(
            f"wrote {shape.config_filename} but get_moe_configs() did not find it "
            f"under SGLANG_MOE_CONFIG_DIR={output_dir}"
        )
    if sorted(int(k) for k in loaded) != expected:
        raise RuntimeError(
            f"config loaded with batch sizes {sorted(loaded)}, expected {expected}"
        )


def _runtime_n(shape) -> int:
    n = shape.shard_intermediate_size // 2
    return n // 2 if shape.dtype_flags["use_int4_w4a16"] else n


def _dtype_str(shape) -> str:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        get_config_dtype_str,
    )

    return get_config_dtype_str(
        shape.torch_dtype,
        use_int8_w8a16=shape.dtype_flags["use_int8_w8a16"],
        use_fp8_w8a8=shape.dtype_flags["use_fp8_w8a8"],
        use_int8_w8a8=shape.dtype_flags["use_int8_w8a8"],
        use_int4_w4a16=shape.dtype_flags["use_int4_w4a16"],
    )
