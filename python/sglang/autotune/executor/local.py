"""Serial in-process executor, driven by a MeasurementDriver."""

from __future__ import annotations

from typing import Optional

from sglang.autotune.executor.base import PruneCheck, SerialExecutor, TrialSlot
from sglang.autotune.measure import MeasurementDriver
from sglang.autotune.registry import register_executor
from sglang.autotune.types import Measurement, Trial

__all__ = ["LocalExecutor"]


@register_executor("local")
class LocalExecutor(SerialExecutor):
    """One trial at a time, measured by ``driver`` in this process.

    The driver reference lives here rather than on ``SerialExecutor`` so that
    the pool executor can hold several drivers, one per slot, without the
    serial path pretending to own one.
    """

    def __init__(
        self, driver: MeasurementDriver, slot: Optional[TrialSlot] = None
    ) -> None:
        super().__init__(slot=slot)
        self.driver = driver

    def run_one(
        self,
        trial: Trial,
        slot: TrialSlot,
        *,
        timeout_s: Optional[float] = None,
        prune_check: Optional[PruneCheck] = None,
    ) -> Measurement:
        return self.driver.measure(
            trial, slot, timeout_s=timeout_s, prune_check=prune_check
        )
