"""Uniform random search: the reference strategy, and the P1 default."""

from __future__ import annotations

from typing import List, Optional

from sglang.autotune.registry import register_strategy
from sglang.autotune.strategy.base import Strategy
from sglang.autotune.types import Measurement, Point

__all__ = ["RandomStrategy"]

# Consecutive duplicate draws tolerated before declaring the space exhausted.
# Rejection sampling has no completion signal of its own, so a nearly-covered
# space would otherwise spin forever.
_MAX_REJECTS = 50


@register_strategy("random")
class RandomStrategy(Strategy):
    """Sample points uniformly, skipping ones already proposed.

    The baseline goes first so a report always has a "before" side to quote a
    speedup against, even when the budget stops the run early.
    """

    def __init__(
        self, seed: int = 0, max_points: Optional[int] = None, **options: object
    ) -> None:
        super().__init__(seed=seed, **options)
        self.max_points = max_points
        self._proposed = 0
        self._baseline_sent = False
        self._space_exhausted = False

    def ask(self, n: int) -> List[Point]:
        points: List[Point] = []
        while len(points) < n and not self._cap_reached():
            point = self._draw()
            if point is None:
                self._space_exhausted = True
                break
            self.state.mark_proposed(point)
            self._proposed += 1
            points.append(point)
        return points

    def tell(self, measurement: Measurement) -> None:
        # Random search ignores feedback by definition, and the incumbent and
        # seen-set bookkeeping is already done for us in StrategyState.observe.
        pass

    def is_exhausted(self) -> bool:
        return self._space_exhausted or self._cap_reached()

    def estimated_points(self) -> Optional[int]:
        return self.max_points

    def describe_plan(self) -> str:
        cap = "unbounded" if self.max_points is None else self.max_points
        # setup() binds the space; a plan printed before then must not claim the
        # space is continuous just because it cannot see it yet.
        if self.space is None:
            total = "the space"
        elif self.space.cardinality is None:
            total = "a continuous space"
        else:
            total = f"{self.space.cardinality} points"
        return f"random: up to {cap} draws from {total}, baseline first"

    def _cap_reached(self) -> bool:
        return self.max_points is not None and self._proposed >= self.max_points

    def _draw(self) -> Optional[Point]:
        if not self._baseline_sent:
            self._baseline_sent = True
            baseline = self.space.baseline()
            if self.state.is_new(baseline):
                return baseline
        for _ in range(_MAX_REJECTS):
            point = self.space.sample(self.rng)
            if self.state.is_new(point):
                return point
        return None
