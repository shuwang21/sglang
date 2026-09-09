"""Driver for the triton fused MoE kernel: one trial is one tile config at one M.

No server and no weights. The GEMM shape comes from the model config, the tensors
are synthetic, and the number rendered is the kernel's own latency, so a trial
costs milliseconds rather than the minutes a serving trial costs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from sglang.autotune.executor.base import TrialSlot
from sglang.autotune.measure import MeasurementDriver
from sglang.autotune.registry import register_driver
from sglang.autotune.types import (
    FailureKind,
    Measurement,
    Point,
    Provenance,
    Trial,
    TrialStatus,
    Workload,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device, get_device_name
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_tuning import (
    benchmark_config,
    get_config_filename,
    get_model_config,
    sort_config,
)

__all__ = ["FusedMoeTritonDriver", "MoeShape", "moe_workloads"]

#: Metric name; also the objective's target. Lower is better.
KERNEL_TIME_US = "kernel_time_us"

_DTYPE_FLAGS = ("use_fp8_w8a8", "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16")


def moe_workloads(batch_sizes: Sequence[int]) -> list[Workload]:
    """One workload per token count; each becomes its own result bucket."""
    return [
        Workload(
            name=f"batch_size={m}", kind="moe_tokens", params={"num_tokens": int(m)}
        )
        for m in batch_sizes
    ]


def moe_bucket(workload: Workload, load: Any) -> str:
    return workload.name


class MoeShape:
    """The GEMM the kernel will run, resolved from the model config.

    Nothing here loads weights: the tuner only needs the shapes, and the
    per-model sharding rules that produce them live with the runtime.
    """

    def __init__(
        self,
        model_path: str,
        *,
        tp_size: int,
        ep_size: int = 1,
        dtype: str = "auto",
        per_channel_quant: bool = False,
        disable_shared_experts_fusion: bool = False,
    ) -> None:
        config = get_model_config(
            model_path, tp_size, ep_size, disable_shared_experts_fusion
        )
        self.model_path = model_path
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.architecture = config["architecture"]
        self.num_experts = config["num_experts"]
        self.topk = config["topk"]
        self.hidden_size = config["hidden_size"]
        self.shard_intermediate_size = config["shard_intermediate_size"]
        self.torch_dtype = config["dtype"]
        self.block_shape = config["block_shape"]
        self.per_channel_quant = per_channel_quant
        self.is_dsv4 = self.architecture == "DeepseekV4ForCausalLM"
        self.dtype_flags: Dict[str, bool] = {
            name: dtype == name.removeprefix("use_") for name in _DTYPE_FLAGS
        }

    @property
    def config_filename(self) -> str:
        """The name the runtime will look for, GPU or not.

        ``get_device_name()`` returns None off-GPU, which the filename builder
        cannot format; a dry run still has to be able to print the name, so
        stand in a placeholder rather than fail.
        """
        if get_device_name() is None:
            return "<device_name unknown off-GPU>"
        return get_config_filename(
            self.num_experts,
            self.shard_intermediate_size,
            self.hidden_size,
            self.topk,
            self.torch_dtype,
            *(self.dtype_flags[name] for name in _DTYPE_FLAGS),
            self.per_channel_quant,
            self.block_shape,
        )

    def weight_bytes(self) -> int:
        """Device memory the synthetic w1 and w2 will take, for a dry run."""
        element = torch.finfo(self.torch_dtype).bits // 8
        w1 = self.num_experts * self.shard_intermediate_size * self.hidden_size
        w2 = self.num_experts * self.hidden_size * self.shard_intermediate_size // 2
        return (w1 + w2) * element

    def describe(self) -> str:
        return (
            f"{self.architecture}: E={self.num_experts}, topk={self.topk}, "
            f"hidden={self.hidden_size}, shard_intermediate="
            f"{self.shard_intermediate_size}, dtype={self.torch_dtype}, "
            f"block_shape={self.block_shape}, "
            f"weights={self.weight_bytes() / 2**30:.2f} GiB"
        )


@register_driver("fused_moe_triton")
class FusedMoeTritonDriver(MeasurementDriver):
    """Times one tile config against the resolved shape.

    ``num_iters`` defaults to the standalone tuner's search setting, not its
    reporting one: each measurement replays a graph of ten calls that many
    times, so 100 costs a thousand kernel launches per trial -- ten seconds at
    a large batch size, for precision a search does not need.
    """

    provides: Tuple[str, ...] = (KERNEL_TIME_US,)

    def __init__(self, shape: MoeShape, *, num_iters: int = 10, seed: int = 0) -> None:
        self.shape = shape
        self.num_iters = num_iters
        self.seed = seed

    def prepare(self, workloads: Sequence[Workload]) -> Sequence[Workload]:
        # benchmark_config allocates without an explicit device, and the moe and
        # topk paths read the published ServerArgs; both are set up per process
        # by the standalone tuner's Ray worker, and this driver is that process.
        torch.set_default_device(get_device())
        torch.get_device_module().manual_seed_all(self.seed)
        set_global_server_args_for_scheduler(
            ServerArgs(
                model_path=self.shape.model_path,
                tp_size=self.shape.tp_size,
                ep_size=self.shape.ep_size,
            )
        )
        return workloads

    def provenance(self) -> Provenance:
        import triton

        return Provenance(
            model=self.shape.model_path,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or "",
            gpu_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
            env={"triton": triton.__version__},
        )

    def measure(
        self,
        trial: Trial,
        slot: TrialSlot,
        timeout_s: Optional[float] = None,
    ) -> Measurement:
        started = time.time()
        shape = self.shape
        try:
            kernel_time = benchmark_config(
                dict(trial.point.values),
                trial.workload.params["num_tokens"],
                shape.num_experts,
                shape.shard_intermediate_size,
                shape.hidden_size,
                shape.topk,
                shape.torch_dtype,
                *(shape.dtype_flags[name] for name in _DTYPE_FLAGS),
                shape.per_channel_quant,
                shape.block_shape,
                shape.is_dsv4,
                num_iters=self.num_iters,
            )
        except Exception as exc:  # noqa: BLE001 - a rejected tile is data
            return Measurement(
                trial=trial,
                status=TrialStatus.FAILED,
                failure=_classify(exc),
                message=f"{type(exc).__name__}: {exc}",
                started_at=started,
                duration_s=time.time() - started,
            )
        return Measurement(
            trial=trial,
            status=TrialStatus.OK,
            metrics={KERNEL_TIME_US: float(kernel_time)},
            started_at=started,
            duration_s=time.time() - started,
        )

    def render_config(self, point: Point) -> Mapping[str, Any]:
        """The entry the kernel reads back, in the key order it writes."""
        return sort_config(dict(point.values))


def _classify(exc: Exception) -> FailureKind:
    import triton

    if isinstance(exc, triton.runtime.autotuner.OutOfResources):
        return FailureKind.OOM
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return FailureKind.OOM
    return FailureKind.BENCH_ERROR
