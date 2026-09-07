"""Backwards-compatible alias.

The contents moved to ``sglang.srt.layers.moe.moe_runner.triton_utils``. They
had to: ``benchmark/`` is excluded from the wheel (``pyproject.toml``), so
anything living here is unreachable from an installed sglang, and the shape
derivation is needed by ``sglang.autotune`` as well as by these scripts.
"""

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_tuning import (
    BenchmarkConfig,
    benchmark_config,
    calculate_shard_intermediate_size,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    get_rocm_configs_compute_bound,
    save_configs,
    sort_config,
)

__all__ = [
    "BenchmarkConfig",
    "benchmark_config",
    "calculate_shard_intermediate_size",
    "get_config_filename",
    "get_configs_compute_bound",
    "get_default_batch_sizes",
    "get_model_config",
    "get_rocm_configs_compute_bound",
    "save_configs",
    "sort_config",
]
