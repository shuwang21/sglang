"""Search space: what is tunable, and which combinations are even legal.

The space owns three things the strategy must not have to know about:

1. the knobs and their domains,
2. the fixed flags every candidate carries,
3. feasibility — combinations that cannot run on this hardware/model.

Pruning infeasible points *here* is what keeps the strategy honest: a strategy
that spends 4 of its 20 trials discovering that ``fa3`` needs sm80+ is wasting
GPU hours on knowledge the space already had.
"""

from __future__ import annotations

import itertools
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from sglang.autotune.types import Point

__all__ = [
    "Domain",
    "Categorical",
    "IntRange",
    "FloatRange",
    "Knob",
    "FeasibilityRule",
    "Feasibility",
    "Conditional",
    "Space",
]


class Domain(ABC):
    """The set of values a knob may take."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Any: ...

    @abstractmethod
    def contains(self, value: Any) -> bool: ...

    def enumerate(self) -> Optional[List[Any]]:
        """Finite value list, or ``None`` if the domain is continuous.

        Grid strategies require this; sampling strategies do not.
        """
        return None

    @property
    def size(self) -> Optional[int]:
        values = self.enumerate()
        return None if values is None else len(values)


@dataclass(frozen=True)
class Categorical(Domain):
    values: Sequence[Any]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(list(self.values))

    def contains(self, value: Any) -> bool:
        return value in self.values

    def enumerate(self) -> Optional[List[Any]]:
        return list(self.values)


@dataclass(frozen=True)
class IntRange(Domain):
    low: int
    high: int
    step: int = 1
    log2: bool = False  # sample/enumerate powers of two (page_size, chunk sizes)

    def enumerate(self) -> Optional[List[Any]]:
        if self.log2:
            out, v = [], max(1, self.low)
            while v <= self.high:
                out.append(v)
                v *= 2
            return out
        return list(range(self.low, self.high + 1, self.step))

    def sample(self, rng: random.Random) -> Any:
        values = self.enumerate() or []
        return rng.choice(values)

    def contains(self, value: Any) -> bool:
        return isinstance(value, int) and self.low <= value <= self.high


@dataclass(frozen=True)
class FloatRange(Domain):
    low: float
    high: float
    step: Optional[float] = None  # set to make the domain enumerable

    def enumerate(self) -> Optional[List[Any]]:
        if self.step is None:
            return None
        n = int(math.floor((self.high - self.low) / self.step)) + 1
        return [round(self.low + i * self.step, 6) for i in range(max(0, n))]

    def sample(self, rng: random.Random) -> Any:
        values = self.enumerate()
        if values is not None:
            return rng.choice(values)
        return rng.uniform(self.low, self.high)

    def contains(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and self.low <= value <= self.high


@dataclass(frozen=True)
class Knob:
    """A tunable flag.

    ``name`` is the canonical ``ServerArgs`` destination (``tp_size``, not
    ``tp``); rendering to a CLI flag is the driver's job. ``group`` lets
    presets and reporters talk about knob families ("attention", "memory").

    ``cost_hint`` is a rough "how much does changing this cost to evaluate"
    signal — restarting with a different ``tp_size`` reloads weights, while
    changing ``schedule_policy`` does not. Cost-aware strategies use it to
    order their sweeps; others ignore it.
    """

    name: str
    domain: Domain
    group: str = "misc"
    description: str = ""
    cost_hint: float = 1.0


@dataclass(frozen=True)
class Feasibility:
    """Result of checking a point against the space's rules."""

    ok: bool
    reasons: Tuple[str, ...] = ()

    @classmethod
    def good(cls) -> "Feasibility":
        return cls(True, ())

    @classmethod
    def bad(cls, *reasons: str) -> "Feasibility":
        return cls(False, tuple(reasons))


class FeasibilityRule(ABC):
    """A predicate that rejects impossible points before any GPU is allocated.

    Rules should live with the subsystem that owns the constraint (attention
    backends know their compute-capability floor; EP knows its divisibility
    requirement) and be contributed into the space, rather than accumulating in
    the tuner as a table of special cases.
    """

    name: str = "rule"

    @abstractmethod
    def check(self, point: Point, context: Mapping[str, Any]) -> Optional[str]:
        """Return a human-readable reason if infeasible, else ``None``."""


@dataclass(frozen=True)
class Conditional:
    """Knobs that only exist when a predicate over the base point holds.

    Example: the ``speculative_num_steps`` / ``speculative_eagle_topk`` knobs
    are meaningless unless ``speculative_algorithm`` is set. Without this,
    a flat space wastes trials on assignments that collapse to the same
    deployment.
    """

    when: Mapping[str, Any]
    knobs: Sequence[Knob]

    def applies(self, point: Point) -> bool:
        return all(point.get(k) == v for k, v in self.when.items())


class Space(ABC):
    """Base class for search spaces.

    Subclasses supply :meth:`knobs` (and usually :meth:`fixed`); the sampling,
    enumeration, and feasibility plumbing is provided here so that adding a new
    space — PD-disaggregation topologies, HiCache tiers, a model-family preset —
    is a matter of declaring knobs and rules.
    """

    name: str = "space"

    def __init__(
        self,
        fixed: Optional[Mapping[str, Any]] = None,
        rules: Sequence[FeasibilityRule] = (),
        conditionals: Sequence[Conditional] = (),
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._fixed: Dict[str, Any] = dict(fixed or {})
        self._rules: List[FeasibilityRule] = list(rules)
        self._conditionals: List[Conditional] = list(conditionals)
        # Hardware/model facts the rules need: gpu_count, compute capability,
        # model config. Populated by the config loader.
        self.context: Dict[str, Any] = dict(context or {})

    # ---- declaration -----------------------------------------------------

    @abstractmethod
    def knobs(self) -> Sequence[Knob]:
        """Knobs searched unconditionally."""

    def fixed(self) -> Mapping[str, Any]:
        """Flags applied to every point but never searched."""
        return dict(self._fixed)

    def rules(self) -> Sequence[FeasibilityRule]:
        return tuple(self._rules)

    def add_rule(self, rule: FeasibilityRule) -> None:
        self._rules.append(rule)

    def knob(self, name: str) -> Optional[Knob]:
        return next((k for k in self.knobs() if k.name == name), None)

    # ---- construction ----------------------------------------------------

    def materialize(self, assignment: Mapping[str, Any]) -> Point:
        """Turn a partial knob assignment into a launchable point.

        Applies fixed flags, then the assignment, then any conditional knobs
        whose predicate now holds (left at their assigned value if present).
        """
        values: Dict[str, Any] = {**self.fixed(), **dict(assignment)}
        point = Point(values)
        for conditional in self._conditionals:
            if not conditional.applies(point):
                for knob in conditional.knobs:
                    values.pop(knob.name, None)
        return Point(values)

    def baseline(self) -> Point:
        """The reference point: fixed flags plus each knob's first value.

        Coordinate-descent style strategies start here, and reporters use it as
        the "before" side of a speedup claim.
        """
        assignment: Dict[str, Any] = {}
        for knob in self.knobs():
            values = knob.domain.enumerate()
            if values:
                assignment[knob.name] = values[0]
        return self.materialize(assignment)

    # ---- exploration -----------------------------------------------------

    def active_knobs(self, point: Point) -> List[Knob]:
        active = list(self.knobs())
        for conditional in self._conditionals:
            if conditional.applies(point):
                active.extend(conditional.knobs)
        return active

    def sample(self, rng: random.Random) -> Point:
        assignment = {k.name: k.domain.sample(rng) for k in self.knobs()}
        point = self.materialize(assignment)
        for conditional in self._conditionals:
            if conditional.applies(point):
                for knob in conditional.knobs:
                    assignment[knob.name] = knob.domain.sample(rng)
        return self.materialize(assignment)

    def grid(self) -> Iterator[Point]:
        """Full Cartesian product, feasible points only.

        Deliberately a generator: the product is frequently enormous and the
        caller is expected to bound it. Ordering follows knob declaration
        order, so callers must not rely on truncation to be representative —
        that was the flaw in the original tiered implementation.
        """
        knobs = [k for k in self.knobs() if k.domain.enumerate()]
        domains = [k.domain.enumerate() or [] for k in knobs]
        for combo in itertools.product(*domains):
            base = dict(zip((k.name for k in knobs), combo))
            for assignment in self._with_conditionals(base):
                point = self.materialize(assignment)
                if self.feasible(point).ok:
                    yield point

    def _with_conditionals(self, base: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """``base`` crossed with the knobs whose predicate it satisfies.

        Conditional knobs are not in :meth:`knobs`, so the product above cannot
        reach them; without this a grid search silently never tries any of them.
        """
        extra = [
            k
            for conditional in self._conditionals
            if conditional.applies(self.materialize(base))
            for k in conditional.knobs
            if k.domain.enumerate()
        ]
        if not extra:
            yield base
            return
        for combo in itertools.product(*(k.domain.enumerate() for k in extra)):
            yield {**base, **dict(zip((k.name for k in extra), combo))}

    @property
    def cardinality(self) -> Optional[int]:
        """Upper bound on the grid, or ``None`` if any domain is continuous.

        An upper bound rather than the exact count, because a conditional's
        knobs multiply only the base points whose predicate holds, and finding
        which those are means enumerating the base product -- which is the one
        thing this property exists to avoid. Counting them everywhere
        overstates; ignoring them understates, and a plan that understates its
        own cost is the worse failure.
        """
        total = 1
        knobs = list(self.knobs()) + [
            k for conditional in self._conditionals for k in conditional.knobs
        ]
        for knob in knobs:
            size = knob.domain.size
            if size is None:
                return None
            total *= max(1, size)
        return total

    # ---- feasibility -----------------------------------------------------

    def feasible(self, point: Point) -> Feasibility:
        reasons = [
            reason
            for reason in (rule.check(point, self.context) for rule in self._rules)
            if reason
        ]
        return Feasibility(not reasons, tuple(reasons))

    def validate(self, point: Point) -> List[str]:
        """Type/domain errors in a point, independent of feasibility rules.

        Catching a typo'd flag or an out-of-domain value here — at config-load
        time — is far cheaper than discovering it when a server fails to start
        twenty minutes into a run.
        """
        errors: List[str] = []
        by_name = {k.name: k for k in self.active_knobs(point)}
        for name, value in point.values.items():
            knob = by_name.get(name)
            if knob is None:
                continue  # fixed flags are not domain-checked
            if not knob.domain.contains(value):
                errors.append(f"{name}={value!r} is outside {knob.domain}")
        return errors
