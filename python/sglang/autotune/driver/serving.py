"""Driver for server-level tuning: one trial is one deployment under load.

Launch a server with the point's flags, drive `sglang.benchmark.serving` at it,
rank on the steady-state window, tear it down. The measurement client is reused
rather than reimplemented: `build_parser()` supplies a Namespace with every
field at its real default, so this driver cannot drift as flags are added.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.autotune.executor.base import PruneCheck, TrialSlot
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

logger = logging.getLogger(__name__)

__all__ = ["ServingDriver", "STEADY_OUTPUT_THROUGHPUT", "render_server_flags"]

#: Default objective: throughput over the window that excludes ramp-up.
STEADY_OUTPUT_THROUGHPUT = "steady_output_throughput"

_FULL_RUN_METRICS = (
    "output_throughput",
    "request_throughput",
    "duration",
    "completed",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
)
_STEADY_METRICS = (
    "output_throughput",
    "average_concurrency",
    "peak_concurrency",
    "duration",
    "completed",
)

# Server log fragments that name a failure better than "it did not come up".
_FAILURE_MARKERS = (
    ("out of memory", FailureKind.OOM),
    ("address already in use", FailureKind.PORT_IN_USE),
    ("unrecognized arguments", FailureKind.UNSUPPORTED_FLAG),
    ("invalid choice", FailureKind.UNSUPPORTED_FLAG),
)


def render_server_flags(point: Point) -> List[str]:
    """Point values as launch_server flags.

    Knob names are ServerArgs destinations (`tp_size`), and turning one into a
    flag is the driver's job because only it knows the CLI's shape: a False
    boolean is an absent flag, not `--flag False`.
    """
    flags: List[str] = []
    for name, value in point.values.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif value is not None:
            flags.extend([flag, str(value)])
    return flags


@register_driver("serving")
class ServingDriver(MeasurementDriver):
    """One trial: launch, load, measure, tear down."""

    provides: Tuple[str, ...] = (
        *_FULL_RUN_METRICS,
        *(f"steady_{name}" for name in _STEADY_METRICS),
        "success_rate",
    )

    def __init__(
        self,
        model_path: str,
        *,
        steady_state_ratio: float = 0.8,
        server_timeout_s: float = 600.0,
        extra_server_args: Sequence[str] = (),
        extra_bench_args: Sequence[str] = (),
        log_dir: Optional[Path] = None,
    ) -> None:
        self.model_path = model_path
        self.steady_state_ratio = steady_state_ratio
        self.server_timeout_s = server_timeout_s
        self.extra_server_args = list(extra_server_args)
        self.extra_bench_args = list(extra_bench_args)
        self.log_dir = log_dir

    def provenance(self) -> Provenance:
        import torch

        import sglang

        return Provenance(
            sglang_version=sglang.__version__,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or "",
            gpu_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
        )

    def measure(
        self,
        trial: Trial,
        slot: TrialSlot,
        timeout_s: Optional[float] = None,
        prune_check: Optional[PruneCheck] = None,
    ) -> Measurement:
        from sglang.benchmark.steady_state_serving import (
            run_steady_state_benchmark,
        )
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import popen_launch_server

        started = time.time()
        base_url = f"http://{slot.host}:{slot.port}"
        launch_timeout = min(self.server_timeout_s, timeout_s or self.server_timeout_s)
        process = None
        try:
            process = popen_launch_server(
                self.model_path,
                base_url,
                timeout=launch_timeout,
                other_args=render_server_flags(trial.point) + self.extra_server_args,
                env=slot.env() or None,
            )
            full, steady = run_steady_state_benchmark(
                args=self._bench_args(trial, slot),
                concurrency_ratio=self.steady_state_ratio,
            )
        except Exception as exc:  # noqa: BLE001 - a config that will not serve is data
            return Measurement(
                trial=trial,
                status=self._status_for(exc),
                failure=_classify(str(exc)),
                message=f"{type(exc).__name__}: {exc}",
                started_at=started,
                duration_s=time.time() - started,
            )
        finally:
            if process is not None and process.poll() is None:
                kill_process_tree(process.pid)

        return Measurement(
            trial=trial,
            status=TrialStatus.OK,
            metrics=self._collect(full, steady, trial.workload),
            started_at=started,
            duration_s=time.time() - started,
        )

    def _bench_args(self, trial: Trial, slot: TrialSlot):
        # Imported here so the module stays importable without the
        # benchmark client's dependencies, for a dry run or a unit test.
        from sglang.benchmark.serving import build_parser

        workload = trial.workload
        argv = [
            "--backend",
            "sglang",
            "--model",
            self.model_path,
            "--host",
            slot.host,
            "--port",
            str(slot.port),
            "--dataset-name",
            workload.kind,
        ]
        for name, value in workload.params.items():
            argv.extend([f"--{name.replace('_', '-')}", str(value)])
        if trial.load.max_concurrency is not None:
            argv.extend(["--max-concurrency", str(trial.load.max_concurrency)])
        if trial.load.request_rate is not None:
            argv.extend(["--request-rate", str(trial.load.request_rate)])
        argv.extend(self.extra_bench_args)
        return build_parser().parse_args(argv)

    def _collect(
        self, full: Mapping[str, Any], steady: Any, workload: Workload
    ) -> Dict[str, float]:
        metrics = {
            name: float(full[name])
            for name in _FULL_RUN_METRICS
            if full.get(name) is not None
        }
        for name in _STEADY_METRICS:
            value = getattr(steady, name, None)
            if value is not None:
                metrics[f"steady_{name}"] = float(value)
        # Not a BenchmarkMetrics field: a run where a third of the requests
        # failed can still post a fine throughput.
        requested = workload.params.get("num_prompts")
        if requested:
            metrics["success_rate"] = float(full["completed"]) / float(requested)
        return metrics

    def _status_for(self, exc: Exception) -> TrialStatus:
        return (
            TrialStatus.TIMEOUT if isinstance(exc, TimeoutError) else TrialStatus.FAILED
        )

    def describe_trial(self, trial: Trial, slot: TrialSlot) -> List[str]:
        """The two commands this trial is, for a plan to show before spending.

        Building the benchmark Namespace here is the point: it goes through the
        real parser, so a flag this driver gets wrong fails on a dry run rather
        than after a server has come up.
        """
        args = self._bench_args(trial, slot)
        bench = [
            "python -m sglang.benchmark.serving",
            f"--backend {args.backend}",
            f"--model {args.model}",
            f"--host {args.host} --port {args.port}",
            f"--dataset-name {args.dataset_name}",
            f"--num-prompts {args.num_prompts}",
            f"--random-input-len {args.random_input_len}",
            f"--random-output-len {args.random_output_len}",
        ]
        if args.max_concurrency is not None:
            bench.append(f"--max-concurrency {args.max_concurrency}")
        return [self.render_launch_command(trial.point), " ".join(bench)]

    def render_launch_command(self, point: Point) -> str:
        flags = render_server_flags(point) + self.extra_server_args
        return " ".join(
            ["python -m sglang.launch_server", "--model-path", self.model_path, *flags]
        )

    def render_config(self, point: Point) -> Mapping[str, Any]:
        return {"model_path": self.model_path, **dict(point.values)}


def _classify(message: str) -> FailureKind:
    lowered = message.lower()
    for marker, kind in _FAILURE_MARKERS:
        if marker in lowered:
            return kind
    return FailureKind.HEALTH_TIMEOUT
