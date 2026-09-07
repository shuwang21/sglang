"""A driver that computes metrics from a formula instead of running anything.

This is what makes the orchestrator testable: budget accounting, dedup, resume,
pruning, bucketed ranking, and reporting all run on CPU in CI, with no server,
no weights, and no GPU. It is test scaffolding, not a tuning target.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

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

__all__ = ["MockDriver", "MetricFn", "sleep_metrics", "spawn_child_metrics"]

#: ``(point, workload) -> metrics``, or ``None`` to fail the trial.
MetricFn = Callable[[Point, Workload], Optional[Mapping[str, float]]]


def spawn_child_metrics(point: Point, workload: Workload) -> Mapping[str, float]:
    """Leave a long-lived child behind, as a driver launching a server does.

    Exists to check that killing a stuck worker takes its server with it: an
    orphan holds a port and the GPU against every later trial in that slot.
    """
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(str(point.get("pidfile"))).write_text(str(child.pid), encoding="utf-8")
    time.sleep(float(point.get("seconds", 0.0)))
    return {"output_throughput": 1.0}


def sleep_metrics(point: Point, workload: Workload) -> Optional[Mapping[str, float]]:
    """Cost the wall clock the point's ``seconds`` knob asks for.

    Defined here rather than in a test because pickle carries a function by
    module and name: a pool worker is a separate process, and a closure written
    beside the test it serves is not importable there.
    """
    if point.get("fail"):
        return None
    seconds = float(point.get("seconds", 0.0))
    time.sleep(seconds)
    return {"output_throughput": 1.0 / seconds if seconds else float("inf")}


@register_driver("mock")
class MockDriver(MeasurementDriver):
    """Metrics from ``metric_fn``; every trial costs ``trial_seconds``."""

    def __init__(
        self,
        metric_fn: MetricFn,
        *,
        provides: Sequence[str] = ("output_throughput", "p99_ttft_ms"),
        trial_seconds: float = 1.0,
        failure: FailureKind = FailureKind.OOM,
        model: str = "mock-model",
    ) -> None:
        self.metric_fn = metric_fn
        self.model = model
        # Instance attribute shadowing the class-level tuple, so two mock
        # drivers in one test can declare different metric sets.
        self.provides: Tuple[str, ...] = tuple(provides)
        self.trial_seconds = trial_seconds
        self.failure = failure
        self.measured: list[Trial] = []

    def provenance(self) -> Provenance:
        return Provenance(model=self.model, gpu_name="mock", gpu_count=1)

    def measure(
        self,
        trial: Trial,
        slot: TrialSlot,
        timeout_s: Optional[float] = None,
        prune_check: Optional[PruneCheck] = None,
    ) -> Measurement:
        started = time.time()
        self.measured.append(trial)
        metrics = self.metric_fn(trial.point, trial.workload)

        if metrics is None:
            return self._result(
                trial,
                TrialStatus.FAILED,
                started,
                failure=self.failure,
                message="metric_fn declined this point",
            )

        metrics = dict(metrics)
        if prune_check is not None:
            reason = prune_check(trial, metrics)
            if reason is not None:
                return self._result(
                    trial, TrialStatus.PRUNED, started, metrics=metrics, message=reason
                )
        return self._result(trial, TrialStatus.OK, started, metrics=metrics)

    def _result(
        self,
        trial: Trial,
        status: TrialStatus,
        started: float,
        *,
        metrics: Optional[Mapping[str, float]] = None,
        failure: Optional[FailureKind] = None,
        message: str = "",
    ) -> Measurement:
        return Measurement(
            trial=trial,
            status=status,
            metrics=dict(metrics or {}),
            failure=failure,
            message=message,
            started_at=started,
            duration_s=self.trial_seconds,
        )
