"""Search strategy: which points to try next, given what we have learned.

The contract is ask/tell, borrowed from the standard optimizer interface:

    strategy.setup(space, objective, constraints, budget, history)
    while not done:
        points = strategy.ask(n)          # n = free executor slots
        for m in run(points):
            strategy.tell(m)

A strategy never launches anything, never reads a config file, and never
decides feasibility. That restriction is what makes strategies cheap to write
and testable against an analytic objective with no GPU.

One trial costs minutes (weight load + warmup + benchmark), so the interesting
strategies are the ones that spend a small trial budget well: reduce fidelity,
prune hopeless runs early, and exploit the structure of the space.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional, Sequence

from sglang.autotune.objective import Constraint, Objective, evaluate
from sglang.autotune.space.base import Space
from sglang.autotune.types import FULL_FIDELITY, Budget, Fidelity, Measurement, Point

__all__ = ["Strategy", "StrategyState"]


class StrategyState:
    """Bookkeeping every strategy needs, so subclasses do not re-implement it."""

    def __init__(self) -> None:
        self.history: List[Measurement] = []
        self.seen: set[str] = set()  # point fingerprints already proposed
        self.best: Optional[Measurement] = None

    def observe(
        self,
        measurement: Measurement,
        objective: Objective,
        constraints: Sequence[Constraint],
    ) -> bool:
        """Record a measurement; return True if it became the new incumbent."""
        self.history.append(measurement)
        self.seen.add(measurement.trial.point.fingerprint)
        candidate = evaluate(measurement, objective, constraints)
        if not candidate.rankable:
            return False
        if self.best is None:
            self.best = measurement
            return True
        incumbent = evaluate(self.best, objective, constraints)
        if candidate.sort_key > incumbent.sort_key:
            self.best = measurement
            return True
        return False

    def is_new(self, point: Point) -> bool:
        return point.fingerprint not in self.seen

    def mark_proposed(self, point: Point) -> None:
        self.seen.add(point.fingerprint)


class Strategy(ABC):
    """Base class for search strategies.

    Subclasses implement :meth:`ask` and :meth:`tell`. Everything else has a
    default that means "no opinion", so a minimal strategy is ~20 lines.
    """

    name: str = "strategy"

    def __init__(self, seed: int = 0, **options: object) -> None:
        self.rng = random.Random(seed)
        self.options: Dict[str, object] = dict(options)
        self.state = StrategyState()
        self.space: Optional[Space] = None
        self.objective: Optional[Objective] = None
        self.constraints: Sequence[Constraint] = ()
        self.budget: Optional[Budget] = None

    # ---- lifecycle -------------------------------------------------------

    def setup(
        self,
        space: Space,
        objective: Objective,
        constraints: Sequence[Constraint],
        budget: Budget,
        history: Sequence[Measurement] = (),
    ) -> None:
        """Bind the run's components and replay any resumed history.

        Replaying history through :meth:`tell` — rather than handing the
        strategy a list — means a resumed run and a fresh run drive the
        strategy through identical state transitions.
        """
        self.space = space
        self.objective = objective
        self.constraints = tuple(constraints)
        self.budget = budget
        for measurement in history:
            # Mirrors Orchestrator._observe, which also records into the state
            # before telling; without it a resumed run re-proposes every point.
            self.state.observe(measurement, self.objective, self.constraints)
            self.tell(measurement)

    # ---- the contract ----------------------------------------------------

    @abstractmethod
    def ask(self, n: int) -> List[Point]:
        """Propose up to ``n`` points to evaluate next.

        May return fewer than ``n`` (including zero) when the strategy wants to
        wait for outstanding results; the orchestrator will call again after
        the next :meth:`tell`. Returning zero while :meth:`is_exhausted` is
        False and nothing is in flight ends the run.
        """

    @abstractmethod
    def tell(self, measurement: Measurement) -> None:
        """Incorporate one result. Called for failures and prunes too.

        Failure information is signal, not noise: a candidate that OOMed tells
        a memory-aware strategy to back off ``mem_fraction_static`` rather than
        to try the same region again.
        """

    # ---- optional hooks --------------------------------------------------

    def fidelity_for(self, point: Point) -> Fidelity:
        """Fidelity at which to evaluate ``point``. Default: full."""
        return FULL_FIDELITY

    def should_prune(
        self, point: Point, partial_metrics: Mapping[str, float]
    ) -> Optional[str]:
        """Abort a running trial early; return a reason, or ``None`` to continue.

        Called by the executor as partial metrics stream in. The default is to
        never prune, so a strategy opts into this rather than inheriting a
        surprise.
        """
        return None

    def is_exhausted(self) -> bool:
        """True when the strategy has nothing left to propose.

        Independent of :class:`Budget`; a grid strategy exhausts its own space
        long before the wall clock runs out.
        """
        return False

    def estimated_points(self) -> Optional[int]:
        """How many points this strategy intends to evaluate, if it knows.

        ``None`` means unbounded or not yet decided; a planner then has only
        the space's own size to go on. Cost is this times the trials each point
        costs, which is the task's to know, not the strategy's.
        """
        return None

    def describe_plan(self) -> str:
        """One-paragraph human summary, printed by ``autotune plan``.

        A tuning run is a several-hour commitment; the user should be able to
        read what it intends to do before spending that.
        """
        return f"{self.name}: no plan description"
