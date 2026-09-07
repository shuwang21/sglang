"""The tuning task and its result: the framework's public API surface.

``TuneTask`` is what a config file deserializes into, and what
``sglang.autotune.tune(task)`` accepts. Holding the *resolved* components
(space, strategy, executor, objective) rather than their names keeps the
orchestrator free of registry lookups and lets Python callers assemble a task
without going through YAML at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from sglang.autotune.executor.base import Executor
from sglang.autotune.measure import LoadPlan, MeasurementDriver
from sglang.autotune.objective import Constraint, Evaluation, Objective
from sglang.autotune.space.base import Space
from sglang.autotune.store import TrialStore
from sglang.autotune.strategy.base import Strategy
from sglang.autotune.types import (
    Budget,
    LoadPoint,
    Measurement,
    Point,
    Provenance,
    Workload,
)

__all__ = ["ModelSpec", "HardwareSpec", "TuneTask", "TuneResult", "BucketRule"]

#: Names the result bucket a (workload, load) pair competes in; "" = shared.
BucketRule = Callable[[Workload, LoadPoint], str]


def _single_bucket(workload: Workload, load: LoadPoint) -> str:
    return ""


@dataclass
class ModelSpec:
    path: str
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tokenizer:
            self.tokenizer = self.path


@dataclass
class HardwareSpec:
    """The device budget a run may use.

    ``gpus_per_trial`` is what determines concurrency: on an 8-GPU host, a
    1-GPU candidate can be evaluated eight-wide. Making this explicit (rather
    than inferring it from ``CUDA_VISIBLE_DEVICES`` as the original
    implementation did) is what lets the executor partition devices safely.
    """

    gpu_count: int = 1
    gpus_per_trial: Optional[int] = None
    nodes: int = 1
    gpu_name: str = ""

    def __post_init__(self) -> None:
        if self.gpus_per_trial is None:
            self.gpus_per_trial = self.gpu_count

    @property
    def concurrent_trials(self) -> int:
        per = max(1, int(self.gpus_per_trial or 1))
        return max(1, self.gpu_count // per)


@dataclass
class TuneTask:
    """A complete, self-contained description of one tuning run."""

    name: str
    model: ModelSpec
    hardware: HardwareSpec
    workloads: Sequence[Workload]
    load_plan: LoadPlan

    space: Space
    strategy: Strategy
    driver: MeasurementDriver
    executor: Executor
    objective: Objective
    constraints: Sequence[Constraint] = ()
    store: Optional[TrialStore] = None

    budget: Budget = field(default_factory=Budget)
    output_dir: Path = Path("./autotune")
    seed: int = 0
    resume: bool = True
    provenance: Provenance = field(default_factory=Provenance)
    #: One winner per bucket. The default is one shared bucket (a single
    #: best.yaml); kernel tuning buckets by shape or batch size instead.
    bucket_of: BucketRule = _single_bucket

    def validate(self) -> List[str]:
        """Config errors worth failing before any GPU is touched.

        Cheap checks here replace expensive discoveries later: a constraint on
        a metric the driver never emits used to mean every candidate passed.
        """
        errors: List[str] = []
        if not self.workloads:
            errors.append("at least one workload is required")

        emitted = set(self.driver.provides)
        if emitted:
            for metric in self.objective.metric_names:
                if metric not in emitted:
                    errors.append(
                        f"objective reads {metric!r}, which driver "
                        f"{self.driver.name!r} does not report"
                    )
            for constraint in self.constraints:
                metric = getattr(constraint, "metric", None)
                if metric is not None and metric not in emitted:
                    errors.append(
                        f"constraint reads {metric!r}, which driver "
                        f"{self.driver.name!r} does not report"
                    )

        errors.extend(self.space.validate(self.space.baseline()))
        if self.hardware.gpus_per_trial and (
            self.hardware.gpus_per_trial > self.hardware.gpu_count
        ):
            errors.append("hardware.gpus_per_trial exceeds hardware.gpu_count")
        return errors

    def estimated_trials(self) -> int:
        """Upper bound used by ``autotune plan`` to state a run's cost.

        A point costs one trial per (workload, load); the strategy says how
        many points it means to try, falling back to the whole space when it
        is unbounded. Zero means neither bound is known.
        """
        per_point = len(self.workloads) * self.load_plan.estimated_runs
        points = self.strategy.estimated_points()
        if points is None:
            points = self.space.cardinality
        total = per_point * points if points is not None else 0
        if self.budget.max_trials is None:
            return total
        return min(total, self.budget.max_trials) if total else self.budget.max_trials


@dataclass
class TuneResult:
    """Everything a run produced. Reporters consume this; nothing else does."""

    task: TuneTask
    measurements: List[Measurement]
    ranking: List[Evaluation]
    #: Per-bucket rankings, best first. ``best`` and ``pareto`` are set only
    #: when there is exactly one bucket; a multi-bucket run's deliverable is
    #: ``best_by_bucket``, and reporters render it as a table.
    ranking_by_bucket: Dict[str, List[Evaluation]] = field(default_factory=dict)
    best_by_bucket: Dict[str, Evaluation] = field(default_factory=dict)
    best: Optional[Evaluation] = None
    pareto: List[Evaluation] = field(default_factory=list)
    partial_reason: Optional[str] = None
    output_dir: Path = Path(".")
    started_at: float = 0.0
    duration_s: float = 0.0

    @property
    def best_point(self) -> Optional[Point]:
        return self.best.point if self.best else None

    @property
    def buckets(self) -> List[str]:
        return sorted(self.ranking_by_bucket)

    @property
    def complete(self) -> bool:
        return self.partial_reason is None
