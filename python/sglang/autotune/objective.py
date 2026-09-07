"""What "better" means: objectives, constraints, and the feasible frontier.

Separating *feasibility* from *quality* is the point of this module. A
candidate that violates the SLA is not a low-scoring candidate, it is not a
candidate at all — ranking must never let a fast-but-violating config beat a
slower compliant one. The original implementation conflated the two by folding
an ``sla_passed`` flag into a sort tuple, and silently ignored every percentile
threshold, so every candidate "passed".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.autotune.types import Measurement, Point

__all__ = [
    "Direction",
    "Objective",
    "ScalarObjective",
    "Constraint",
    "MetricThreshold",
    "Violation",
    "Evaluation",
    "evaluate",
    "evaluate_point",
    "rank",
    "rank_by_bucket",
    "margin",
    "pareto_front",
]


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    @property
    def sign(self) -> float:
        return 1.0 if self is Direction.MAXIMIZE else -1.0


class Objective(ABC):
    """Maps a measurement to one or more scores, higher-is-better after signing.

    Vector objectives (throughput *and* p99 latency) return more than one
    component; the framework then reports a Pareto front instead of a single
    winner. Scalarization, if wanted, is an objective's own business.
    """

    name: str = "objective"

    @abstractmethod
    def components(self, measurement: Measurement) -> Optional[Tuple[float, ...]]:
        """Signed score components, or ``None`` if the measurement is unusable.

        Returning ``None`` (rather than ``-inf``) keeps failed trials out of
        rankings entirely while leaving them in the record.
        """

    @property
    def metric_names(self) -> Tuple[str, ...]:
        """Metrics this objective reads. Used to validate a driver's output."""
        return ()

    def aggregate(
        self, measurements: Sequence[Measurement]
    ) -> Optional[Tuple[float, ...]]:
        """Point-level score over every full-fidelity measurement of one point.

        A point is measured once per (workload, load), and ``best.yaml`` needs
        one winner, so ranking happens here rather than per measurement. The
        default handles the single-measurement case only and refuses to guess
        otherwise; multi-workload aggregation (min, weighted, ...) is an
        objective's own policy and overrides this.
        """
        if len(measurements) != 1:
            return None
        return self.components(measurements[0])

    def is_better(self, a: Measurement, b: Optional[Measurement]) -> bool:
        if b is None:
            return self.components(a) is not None
        ca, cb = self.components(a), self.components(b)
        if ca is None:
            return False
        if cb is None:
            return True
        return ca > cb


@dataclass
class ScalarObjective(Objective):
    """Optimize a single metric, with optional tie-breakers.

    ``tie_breakers`` matter more than they look: throughput measurements of
    near-identical configs differ by noise, and a stable secondary key keeps
    reruns from reshuffling the leaderboard.
    """

    metric: str
    direction: Direction = Direction.MAXIMIZE
    tie_breakers: Sequence[Tuple[str, Direction]] = ()
    name: str = "scalar"

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return (self.metric, *(m for m, _ in self.tie_breakers))

    def components(self, measurement: Measurement) -> Optional[Tuple[float, ...]]:
        if not measurement.status.is_rankable:
            return None
        primary = measurement.metrics.get(self.metric)
        if primary is None or math.isnan(primary):
            return None
        out = [self.direction.sign * float(primary)]
        for metric, direction in self.tie_breakers:
            value = measurement.metrics.get(metric)
            out.append(
                direction.sign * float(value)
                if value is not None and not math.isnan(value)
                else -math.inf
            )
        return tuple(out)


@dataclass(frozen=True)
class Violation:
    constraint: str
    detail: str


class Constraint(ABC):
    """A hard feasibility gate evaluated against a measurement."""

    name: str = "constraint"

    @abstractmethod
    def check(self, measurement: Measurement) -> Optional[Violation]:
        """Return a violation, or ``None`` if satisfied."""

    def can_check_early(self) -> bool:
        """Whether a partial measurement is enough to reject.

        Constraints on a running p99 can fail fast, letting the executor abort
        a trial that is already hopeless instead of paying for the full run.
        """
        return False


@dataclass
class MetricThreshold(Constraint):
    """``metric`` must stay within ``[min_value, max_value]``.

    This one concrete constraint is included because it covers every SLA the
    cookbook configs express — including the ``p99_*`` and ``success_rate``
    keys that the previous ``meets_sla`` accepted in YAML and then ignored.
    """

    metric: str
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    early: bool = False

    def __post_init__(self) -> None:
        if self.max_value is None and self.min_value is None:
            raise ValueError(f"constraint on {self.metric!r} needs min or max")
        self.name = f"{self.metric}"

    def check(self, measurement: Measurement) -> Optional[Violation]:
        value = measurement.metrics.get(self.metric)
        if value is None or math.isnan(value):
            return Violation(self.name, f"{self.metric} not reported")
        if self.max_value is not None and value > self.max_value:
            return Violation(self.name, f"{self.metric}={value:g} > {self.max_value:g}")
        if self.min_value is not None and value < self.min_value:
            return Violation(self.name, f"{self.metric}={value:g} < {self.min_value:g}")
        return None

    def can_check_early(self) -> bool:
        return self.early


@dataclass
class Evaluation:
    """An objective + constraint verdict on one point.

    ``measurements`` holds every measurement that contributed (one per
    workload, load, and repeat); ``measurement`` is the first, for callers that
    only need a representative. A point is feasible only if all of them are.
    """

    measurements: Tuple[Measurement, ...]
    feasible: bool
    violations: Tuple[Violation, ...] = ()
    components: Optional[Tuple[float, ...]] = None
    #: Widest relative spread of the primary component across the repeats of
    #: any one group, or None when nothing was measured twice.
    spread: Optional[float] = None

    @property
    def measurement(self) -> Measurement:
        return self.measurements[0]

    @property
    def point(self) -> Point:
        return self.measurements[0].trial.point

    @property
    def rankable(self) -> bool:
        return self.feasible and self.components is not None

    @property
    def sort_key(self) -> Tuple[Any, ...]:
        """Feasibility dominates score; unrankable results sort last."""
        return (
            1 if self.feasible else 0,
            self.components if self.components is not None else (-math.inf,),
        )


def evaluate(
    measurement: Measurement,
    objective: Objective,
    constraints: Sequence[Constraint] = (),
) -> Evaluation:
    violations = tuple(
        v for v in (c.check(measurement) for c in constraints) if v is not None
    )
    return Evaluation(
        measurements=(measurement,),
        feasible=not violations and measurement.status.is_rankable,
        violations=violations,
        components=objective.components(measurement),
    )


def _median_repeat(group: Sequence[Measurement], objective: Objective) -> Measurement:
    """The middle measurement of one group, by the objective's primary score.

    A real measurement rather than an average of several: its metrics stay
    mutually consistent and its artifacts still point at the run that produced
    them. With an even count this takes the lower middle, which is the
    pessimistic half.
    """
    scored = sorted(
        group,
        key=lambda m: (c[0] if (c := objective.components(m)) else -math.inf),
    )
    return scored[(len(scored) - 1) // 2]


def _relative_spread(
    group: Sequence[Measurement], objective: Objective
) -> Optional[float]:
    scores = sorted(c[0] for m in group if (c := objective.components(m)))
    if len(scores) < 2:
        return None
    middle = abs(scores[len(scores) // 2])
    return (scores[-1] - scores[0]) / middle if middle else None


def evaluate_point(
    measurements: Sequence[Measurement],
    objective: Objective,
    constraints: Sequence[Constraint] = (),
) -> Evaluation:
    """Verdict on one point from all of its measurements.

    Repeats of one (workload, load) collapse to their median first. They
    measure the same thing, so combining them is arithmetic; measurements of
    *different* workloads are a policy question left to the objective.
    """
    per_measurement = [evaluate(m, objective, constraints) for m in measurements]
    violations = tuple(v for e in per_measurement for v in e.violations)
    feasible = all(e.feasible for e in per_measurement)

    groups: Dict[str, List[Measurement]] = {}
    for m in measurements:
        groups.setdefault(m.trial.repeat_group, []).append(m)
    representatives = [_median_repeat(g, objective) for g in groups.values()]
    # `is not None`, not truthiness: a spread of exactly 0.0 means the point
    # reproduced perfectly, which is the opposite of "never measured twice".
    spreads = [
        s for g in groups.values() if (s := _relative_spread(g, objective)) is not None
    ]

    return Evaluation(
        measurements=tuple(measurements),
        feasible=feasible,
        violations=violations,
        components=objective.aggregate(representatives) if feasible else None,
        spread=max(spreads) if spreads else None,
    )


def rank(
    measurements: Sequence[Measurement],
    objective: Objective,
    constraints: Sequence[Constraint] = (),
) -> List[Evaluation]:
    """One evaluation per point, best first; infeasible and failed at the bottom.

    Only full-fidelity measurements take part. Reduced-fidelity rungs are a
    strategy's private signal and stay in the store, but a winner declared on
    a short run is not a winner.
    """
    by_point: Dict[Tuple[str, str], List[Measurement]] = {}
    for m in measurements:
        if m.trial.fidelity.is_full:
            key = (m.trial.bucket, m.trial.point.fingerprint)
            by_point.setdefault(key, []).append(m)
    evaluations = [
        evaluate_point(group, objective, constraints) for group in by_point.values()
    ]
    return sorted(evaluations, key=lambda e: e.sort_key, reverse=True)


def rank_by_bucket(
    measurements: Sequence[Measurement],
    objective: Objective,
    constraints: Sequence[Constraint] = (),
) -> Dict[str, List[Evaluation]]:
    """:func:`rank`, split by ``Trial.bucket``; each bucket is best first.

    A single-winner run has one bucket, ``""``. A per-shape or per-batch-size
    run has one bucket per shape, and the deliverable is the table of winners.
    """
    ranked = rank(measurements, objective, constraints)
    out: Dict[str, List[Evaluation]] = {}
    for evaluation in ranked:
        out.setdefault(evaluation.measurement.trial.bucket, []).append(evaluation)
    return out


def margin(ranked: Sequence[Evaluation]) -> Optional[float]:
    """How far the winner leads the runner-up, as a fraction of the runner-up.

    A single-shot run reports its best with no sense of scale: a 30% lead and a
    0.3% one read identically. Components are already sign-normalised so higher
    is better, which makes this work for a minimised metric too.
    """
    top = [e for e in ranked if e.rankable and e.components]
    if len(top) < 2:
        return None
    best, second = top[0].components[0], top[1].components[0]
    if second == 0:
        return None
    return (best - second) / abs(second)


def pareto_front(evaluations: Sequence[Evaluation]) -> List[Evaluation]:
    """Non-dominated feasible evaluations, for vector objectives.

    With a single component this degenerates to "the best one", which is the
    correct behavior and means reporters need no special case.
    """
    feasible = [e for e in evaluations if e.rankable]
    front: List[Evaluation] = []
    for candidate in feasible:
        assert candidate.components is not None
        dominated = any(
            other.components is not None
            and other is not candidate
            and all(o >= c for o, c in zip(other.components, candidate.components))
            and any(o > c for o, c in zip(other.components, candidate.components))
            for other in feasible
        )
        if not dominated:
            front.append(candidate)
    return front
