"""Exhaustive search: every feasible point, once."""

from __future__ import annotations

from typing import Iterator, List, Optional

from sglang.autotune.registry import register_strategy
from sglang.autotune.strategy.base import Strategy
from sglang.autotune.types import Measurement, Point

__all__ = ["GridStrategy"]


@register_strategy("grid")
class GridStrategy(Strategy):
    """Walk the space's Cartesian product in declaration order.

    The one strategy whose result does not depend on a seed, and the only one
    that can say it covered the space. Ordering matters here where it must not
    elsewhere: a run cut short by its budget keeps the prefix of the grid, so
    the first value of each knob is the one worth making the baseline.
    """

    def __init__(self, seed: int = 0, **options: object) -> None:
        super().__init__(seed=seed, **options)
        self._points: Optional[Iterator[Point]] = None
        self._drained = False

    def ask(self, n: int) -> List[Point]:
        if self._points is None:
            self._points = iter(self.space.grid())
        points: List[Point] = []
        while len(points) < n:
            point = next(self._points, None)
            if point is None:
                self._drained = True
                break
            if not self.state.is_new(point):
                continue
            self.state.mark_proposed(point)
            points.append(point)
        return points

    def tell(self, measurement: Measurement) -> None:
        # Exhaustive search has nothing to learn from a result; the incumbent
        # and seen-set bookkeeping already happened in StrategyState.observe.
        pass

    def is_exhausted(self) -> bool:
        return self._drained

    def estimated_points(self) -> Optional[int]:
        return self.space.cardinality if self.space is not None else None

    def describe_plan(self) -> str:
        size = self.estimated_points()
        total = "an unbounded space" if size is None else f"{size} points"
        return f"grid: every point of {total}, in declaration order"
