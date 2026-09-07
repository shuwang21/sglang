"""Core value types for the auto-tune framework.

Everything in this module is plain data: no GPUs, no subprocesses, no I/O.
The pluggable components (space, strategy, executor, objective, store,
reporter) exchange these types and nothing else, which is what lets the
orchestrator loop be tested without hardware.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "Point",
    "Workload",
    "LoadPoint",
    "Fidelity",
    "Provenance",
    "Trial",
    "TrialStatus",
    "FailureKind",
    "Measurement",
    "Budget",
    "stable_hash",
]


def stable_hash(payload: Any, length: int = 16) -> str:
    """Order-independent, type-stable fingerprint for dicts/tuples/scalars."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Point:
    """One concrete assignment of tunable knobs: a candidate deployment.

    A point holds the *full* set of flags to launch with — the space's fixed
    flags merged with the searched knobs — so a point is self-contained and can
    be replayed without re-reading the config.

    Two points that differ only in key order or in numeric formatting must have
    the same ``fingerprint``; that is the basis of dedup and resume.
    """

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {str(k): _normalize_value(v) for k, v in self.values.items()}
        object.__setattr__(self, "values", dict(sorted(normalized.items())))

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.values)

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def merged(self, **overrides: Any) -> "Point":
        return Point({**self.values, **overrides})

    def subset(self, names: Iterable[str]) -> Dict[str, Any]:
        return {n: self.values[n] for n in names if n in self.values}

    def diff(self, other: "Point") -> Dict[str, Tuple[Any, Any]]:
        """Knobs that differ, as ``name -> (self_value, other_value)``.

        Used by reporters to explain *why* a candidate won, rather than dumping
        the whole flag set.
        """
        keys = set(self.values) | set(other.values)
        return {
            k: (self.values.get(k), other.values.get(k))
            for k in sorted(keys)
            if self.values.get(k) != other.values.get(k)
        }

    def __str__(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.values.items())


def _normalize_value(value: Any) -> Any:
    """Collapse representations that mean the same thing to the runtime."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


@dataclass(frozen=True)
class Workload:
    """A dataset slice to benchmark against.

    ``kind`` and ``params`` are handed to the dataset layer; the framework does
    not interpret them. ``prepared_path`` is filled in once the dataset has
    been materialized, so every trial in a run benchmarks byte-identical input.
    """

    name: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    prepared_path: Optional[str] = None
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_hash({"kind": self.kind, "params": dict(self.params)})


@dataclass(frozen=True)
class LoadPoint:
    """How hard to drive the server for a single measurement.

    ``request_rate`` is requests/second; ``None`` means "as fast as possible".
    ``max_concurrency`` caps in-flight requests; ``None`` means unbounded.
    A capacity search (e.g. bisecting QPS against an SLA) produces a sequence
    of ``LoadPoint``s — the search itself lives in the strategy or the driver,
    not here.
    """

    request_rate: Optional[float] = None
    max_concurrency: Optional[int] = None

    def __str__(self) -> str:
        rate = "inf" if self.request_rate is None else f"{self.request_rate:g}"
        conc = "unbounded" if self.max_concurrency is None else self.max_concurrency
        return f"qps={rate},mc={conc}"


@dataclass(frozen=True)
class Fidelity:
    """A reduced-cost evaluation of a point.

    Multi-fidelity strategies (successive halving) evaluate many points cheaply
    at low rungs and promote survivors. ``scale`` is an advisory multiplier on
    request count that drivers apply. ``level`` 0 is full fidelity, comparable
    to a real benchmark run; higher levels are cheaper rungs. Only full-fidelity
    measurements enter the final ranking.
    """

    level: int = 0
    scale: float = 1.0
    #: Independent measurements of this trial. Each is a separate process, so
    #: the spread across them is the run's own noise floor -- without it a
    #: winner is reported at whatever margin, including one below that floor.
    repeats: int = 1
    label: str = "full"

    @property
    def is_full(self) -> bool:
        return self.scale >= 1.0 and self.repeats >= 1 and self.level == 0


FULL_FIDELITY = Fidelity()


@dataclass(frozen=True)
class Provenance:
    """What must match for a recorded trial to still apply.

    Results are only reused across runs when this matches. Stale reuse across a
    kernel or driver change is the failure mode most likely to produce a
    confidently wrong recommendation, and so is reuse across a change of model:
    the knobs and the workload can be identical while the thing being measured
    is not.
    """

    model: str = ""
    git_commit: str = ""
    sglang_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    driver_version: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    env: Mapping[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "model": self.model,
                "git": self.git_commit,
                "sglang": self.sglang_version,
                "torch": self.torch_version,
                "cuda": self.cuda_version,
                "driver": self.driver_version,
                "gpu": self.gpu_name,
                "gpus": self.gpu_count,
                "env": dict(self.env),
            }
        )


class TrialStatus(str, Enum):
    """Terminal state of a trial. Every state is recorded, including failures."""

    OK = "ok"
    INFEASIBLE = "infeasible"  # rejected by a space rule; never launched
    PRUNED = "pruned"  # aborted early, cannot beat the incumbent
    FAILED = "failed"  # launch or benchmark error
    TIMEOUT = "timeout"
    SKIPPED = "skipped"  # deduped against an existing result

    @property
    def has_metrics(self) -> bool:
        return self in (TrialStatus.OK, TrialStatus.PRUNED)

    @property
    def is_settled(self) -> bool:
        """Whether a resumed run may keep this verdict instead of rerunning.

        A failure is not settled. It is as often the environment or a bug in
        the harness as a property of the config, and a resume that trusts one
        can never recover from a bad first attempt: the point counts as
        proposed, so the search reports itself exhausted without running
        anything.
        """
        return self in (TrialStatus.OK, TrialStatus.PRUNED, TrialStatus.INFEASIBLE)

    @property
    def is_rankable(self) -> bool:
        return self is TrialStatus.OK


class FailureKind(str, Enum):
    """Why a trial failed, for actionable reporting.

    A run that reports "6 of 24 candidates OOMed" is useful; one that reports
    "6 failed" is not.
    """

    OOM = "oom"
    PORT_IN_USE = "port_in_use"
    UNSUPPORTED_FLAG = "unsupported_flag"
    MODEL_LOAD = "model_load"
    SERVER_CRASH = "server_crash"
    HEALTH_TIMEOUT = "health_timeout"
    BENCH_ERROR = "bench_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"  # aborted by cancel_all(); never a result
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Trial:
    """The unit of work handed to an executor: one point, measured once."""

    point: Point
    workload: Workload
    load: LoadPoint = LoadPoint()
    fidelity: Fidelity = FULL_FIDELITY
    provenance_fingerprint: str = ""
    attempt: int = 0
    tag: str = ""  # free-form label from the strategy, e.g. "rung0", "probe"
    #: Result bucket this trial competes in, e.g. "batch_size=64". Kernel and
    #: per-batch-size tuning produce one winner per bucket; "" is the single
    #: shared bucket whose winner becomes best.yaml.
    bucket: str = ""

    @property
    def key(self) -> str:
        """Deterministic identity used for dedup, resume, and cross-run reuse."""
        return stable_hash(
            {
                "point": self.point.values,
                "workload": self.workload.fingerprint,
                "workload_name": self.workload.name,
                "load": [self.load.request_rate, self.load.max_concurrency],
                "fidelity": [
                    self.fidelity.level,
                    self.fidelity.scale,
                    self.fidelity.repeats,
                ],
                "provenance": self.provenance_fingerprint,
                "bucket": self.bucket,
                "attempt": self.attempt,
            }
        )

    @property
    def repeat_group(self) -> str:
        """Identity shared by every repeat of this measurement.

        The key separates attempts so the store keeps them all; ranking has to
        put them back together, and this is what it groups on.
        """
        return stable_hash(
            {
                "point": self.point.values,
                "workload": self.workload.fingerprint,
                "workload_name": self.workload.name,
                "load": [self.load.request_rate, self.load.max_concurrency],
                "bucket": self.bucket,
            }
        )

    def __str__(self) -> str:
        return f"{self.workload.name}/{self.load}/{self.point.fingerprint}"


@dataclass
class Measurement:
    """The result of running one trial.

    ``metrics`` is a flat ``name -> float`` map so objectives and constraints
    can address any metric by name (``p99_ttft_ms``, ``output_throughput``,
    ``success_rate``, ...) without the framework knowing what they mean.
    """

    trial: Trial
    status: TrialStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    failure: Optional[FailureKind] = None
    message: str = ""
    hint: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    #: Measurements of *other* points that the same run produced, for drivers
    #: that sweep a candidate set in one call (cutlass profiler, triton
    #: autotune). Each carries its own Trial; the orchestrator records and
    #: tells them like any other result. Executor accounting is unaffected:
    #: one submission still yields exactly one top-level measurement.
    discovered: Tuple["Measurement", ...] = ()

    @property
    def key(self) -> str:
        return self.trial.key

    @property
    def ok(self) -> bool:
        return self.status is TrialStatus.OK

    def metric(self, name: str, default: float = float("nan")) -> float:
        return self.metrics.get(name, default)

    def to_json(self) -> Dict[str, Any]:
        """Round-trips through :meth:`from_json` with an unchanged trial key.

        The flat fields are what a CSV column or a human reads; ``trial`` holds
        the rest of the identity, which only the key and a resumed run need.
        """
        trial = self.trial
        return {
            "key": self.key,
            "status": self.status.value,
            "point": dict(trial.point.values),
            "workload": trial.workload.name,
            "request_rate": trial.load.request_rate,
            "max_concurrency": trial.load.max_concurrency,
            "fidelity": trial.fidelity.level,
            "tag": trial.tag,
            "bucket": trial.bucket,
            "metrics": dict(self.metrics),
            "failure": self.failure.value if self.failure else None,
            "message": self.message,
            "hint": self.hint,
            "artifacts": dict(self.artifacts),
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "extra": dict(self.extra),
            "trial": {
                "workload_kind": trial.workload.kind,
                "workload_params": dict(trial.workload.params),
                "workload_prepared_path": trial.workload.prepared_path,
                "fidelity_scale": trial.fidelity.scale,
                "fidelity_repeats": trial.fidelity.repeats,
                "fidelity_label": trial.fidelity.label,
                "provenance": trial.provenance_fingerprint,
                "attempt": trial.attempt,
            },
            "discovered": [m.to_json() for m in self.discovered],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Measurement":
        nested = payload["trial"]
        failure = payload["failure"]
        return cls(
            trial=Trial(
                point=Point(payload["point"]),
                workload=Workload(
                    name=payload["workload"],
                    kind=nested["workload_kind"],
                    params=nested["workload_params"],
                    prepared_path=nested["workload_prepared_path"],
                ),
                load=LoadPoint(
                    request_rate=payload["request_rate"],
                    max_concurrency=payload["max_concurrency"],
                ),
                fidelity=Fidelity(
                    level=payload["fidelity"],
                    scale=nested["fidelity_scale"],
                    repeats=nested["fidelity_repeats"],
                    label=nested["fidelity_label"],
                ),
                provenance_fingerprint=nested["provenance"],
                attempt=nested["attempt"],
                tag=payload["tag"],
                bucket=payload["bucket"],
            ),
            status=TrialStatus(payload["status"]),
            metrics=dict(payload["metrics"]),
            failure=FailureKind(failure) if failure else None,
            message=payload["message"],
            hint=payload["hint"],
            artifacts=dict(payload["artifacts"]),
            started_at=payload["started_at"],
            duration_s=payload["duration_s"],
            extra=dict(payload["extra"]),
            discovered=tuple(cls.from_json(m) for m in payload["discovered"]),
        )


class BudgetExhausted(RuntimeError):
    """Raised when the run may not start further work."""


@dataclass
class Budget:
    """Hard bounds on a tuning run.

    The orchestrator checks this before every trial and the executor derives
    per-trial timeouts from ``remaining_seconds``. A run that overruns its
    budget is a bug, not a configuration choice.
    """

    max_trials: Optional[int] = None
    wall_clock_hours: Optional[float] = None
    max_gpu_hours: Optional[float] = None

    started_at: Optional[float] = None
    trials_spent: int = 0
    gpu_seconds_spent: float = 0.0

    def start(self, now: Optional[float] = None) -> None:
        self.started_at = now if now is not None else time.time()

    @property
    def deadline(self) -> Optional[float]:
        if self.started_at is None or self.wall_clock_hours is None:
            return None
        return self.started_at + self.wall_clock_hours * 3600.0

    def remaining_seconds(self, now: Optional[float] = None) -> Optional[float]:
        deadline = self.deadline
        if deadline is None:
            return None
        return max(0.0, deadline - (now if now is not None else time.time()))

    def spend(self, measurement: Measurement, gpus: int = 1) -> None:
        self.trials_spent += 1
        self.gpu_seconds_spent += measurement.duration_s * max(1, gpus)

    def exhausted_reason(self, now: Optional[float] = None) -> Optional[str]:
        if self.max_trials is not None and self.trials_spent >= self.max_trials:
            return f"trial budget reached ({self.max_trials})"
        remaining = self.remaining_seconds(now)
        if remaining is not None and remaining <= 0:
            return f"wall-clock budget reached ({self.wall_clock_hours}h)"
        if (
            self.max_gpu_hours is not None
            and self.gpu_seconds_spent >= self.max_gpu_hours * 3600.0
        ):
            return f"gpu-hour budget reached ({self.max_gpu_hours}h)"
        return None

    def is_exhausted(self, now: Optional[float] = None) -> bool:
        return self.exhausted_reason(now) is not None

    def raise_if_exhausted(self, now: Optional[float] = None) -> None:
        reason = self.exhausted_reason(now)
        if reason is not None:
            raise BudgetExhausted(reason)
