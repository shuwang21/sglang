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
# The launcher reports only the exit code, so the cause is in the log above it.
_FAILURE_MARKERS = (
    ("out of memory", FailureKind.OOM),
    ("address already in use", FailureKind.PORT_IN_USE),
    ("unrecognized arguments", FailureKind.UNSUPPORTED_FLAG),
    ("invalid choice", FailureKind.UNSUPPORTED_FLAG),
    ("gated repo", FailureKind.MODEL_LOAD),
    ("must have access", FailureKind.MODEL_LOAD),
    ("401 client error", FailureKind.MODEL_LOAD),
    ("does not appear to have a file named", FailureKind.MODEL_LOAD),
    ("no such file or directory", FailureKind.MODEL_LOAD),
)


# How a Python process announces why it died, last one wins.
_REASON_PREFIXES = (
    "Exception:",
    "RuntimeError:",
    "ValueError:",
    "AssertionError:",
    "OSError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "torch.OutOfMemoryError:",
    "error:",
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
    """One trial: launch, load, measure, tear down.

    A sweep launches the same model tens of times and pays its weight load
    every time, which for a large model is most of a trial. `--load-format
    presharded` caches the post-process weights under a signature over
    parallelism, quantization and dtype -- none of which the tuned knobs touch
    -- so the whole sweep shares one dump. It is opt-in because that cache is
    written inside the model directory, and because the first launch pays for
    it; measurements are unaffected either way, since throughput is taken
    after the server is healthy.
    """

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
        preshard: bool = False,
    ) -> None:
        self.model_path = model_path
        self.preshard = preshard
        self.steady_state_ratio = steady_state_ratio
        self.server_timeout_s = server_timeout_s
        self.extra_server_args = list(extra_server_args)
        self.extra_bench_args = list(extra_bench_args)
        self.log_dir = log_dir

    def provenance(self) -> Provenance:
        import torch

        import sglang

        return Provenance(
            model=self.model_path,
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
        logs = self._open_logs(trial)
        artifacts = {name: str(handle.name) for name, handle in logs.items()}
        logger.info("launching %s -> %s", trial.point, artifacts.get("server_log", "-"))
        try:
            process = popen_launch_server(
                self.model_path,
                base_url,
                timeout=launch_timeout,
                other_args=self._server_args(trial.point),
                env=slot.env() or None,
                return_stdout_stderr=(
                    (logs["server_log"], logs["server_err"]) if logs else None
                ),
            )
            full, steady = run_steady_state_benchmark(
                args=self._bench_args(trial, slot),
                concurrency_ratio=self.steady_state_ratio,
            )
        except Exception as exc:  # noqa: BLE001 - a config that will not serve is data
            tail = _tail(logs.get("server_log"))
            return Measurement(
                trial=trial,
                status=self._status_for(exc),
                failure=_classify(exc, tail),
                message=_reason(exc, tail),
                artifacts=artifacts,
                started_at=started,
                duration_s=time.time() - started,
            )
        finally:
            if process is not None and process.poll() is None:
                kill_process_tree(process.pid)
            for handle in logs.values():
                handle.close()

        return Measurement(
            trial=trial,
            status=TrialStatus.OK,
            metrics=self._collect(full, steady, trial.workload),
            artifacts=artifacts,
            started_at=started,
            duration_s=time.time() - started,
        )

    def _open_logs(self, trial: Trial) -> Dict[str, Any]:
        """Per-trial server log files, tee'd to the terminal by the launcher.

        A failed candidate is only actionable with the server's own output, and
        with several trials in one terminal the interleaved copy is not enough
        to read afterwards.
        """
        if self.log_dir is None:
            return {}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stem = trial.point.fingerprint
        return {
            "server_log": (self.log_dir / f"{stem}.server.log").open(
                "w", encoding="utf-8"
            ),
            "server_err": (self.log_dir / f"{stem}.server.err").open(
                "w", encoding="utf-8"
            ),
        }

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
            metrics[f"steady_{name}"] = float(getattr(steady, name))
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
        launch = (
            f"{self.render_launch_command(trial.point)} "
            f"--host {slot.host} --port {slot.port}"
        )
        return [launch, " ".join(bench)]

    def _server_args(self, point: Point) -> List[str]:
        """Launch flags for one candidate, including the run-wide additions."""
        flags = render_server_flags(point) + self.extra_server_args
        if self.preshard:
            flags += ["--load-format", "presharded"]
        return flags

    def render_launch_command(self, point: Point) -> str:
        flags = self._server_args(point)
        return " ".join(
            ["python -m sglang.launch_server", "--model-path", self.model_path, *flags]
        )

    def render_config(self, point: Point) -> Mapping[str, Any]:
        return {"model_path": self.model_path, **dict(point.values)}


def _tail(handle: Any, limit: int = 4000) -> str:
    """Last of a server log, for classifying a failure the exception cannot name."""
    if handle is None:
        return ""
    try:
        handle.flush()
        return Path(handle.name).read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _reason(exc: Exception, log_tail: str) -> str:
    """The line worth reading, which is rarely the exception.

    The launcher reports "exited with code 1" whatever went wrong; the server
    said why on its way out, one screen above.
    """
    for line in reversed(log_tail.splitlines()):
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _REASON_PREFIXES):
            return stripped[:400]
    return f"{type(exc).__name__}: {exc}"


def _classify(exc: Exception, log_tail: str) -> FailureKind:
    """Name the cause, preferring the log to the exception that reports it."""
    haystack = f"{exc}\n{log_tail}".lower()
    for marker, kind in _FAILURE_MARKERS:
        if marker in haystack:
            return kind
    if isinstance(exc, TimeoutError):
        return FailureKind.HEALTH_TIMEOUT
    # The launcher raises a plain Exception once the process is gone, and only
    # TimeoutError when it is still running and unhealthy.
    return FailureKind.SERVER_CRASH
