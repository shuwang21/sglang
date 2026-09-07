"""Measurement drivers: turn a point into numbers.

A driver knows how to evaluate one point and report a flat metric map. For a
serving driver that means launching a server, driving load, and tearing down;
for a kernel driver it is a microbenchmark of one tile configuration with no
server at all. It is the only layer that knows about ``launch_server``,
``bench_serving``, triton autotune, or log files.

Splitting this from :class:`~sglang.autotune.executor.base.Executor` means the
serving driver, an offline-throughput driver, a kernel driver, and an accuracy
gate can all run under the same concurrency machinery, and that a new
measurement modality does not have to reimplement process supervision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from sglang.autotune.executor.base import PruneCheck, TrialSlot
from sglang.autotune.types import (
    LoadPoint,
    Measurement,
    Point,
    Provenance,
    Trial,
    Workload,
)

__all__ = ["MeasurementDriver", "LoadPlan", "CapacitySearch"]


@dataclass
class LoadPlan:
    """How a driver should sweep load for one point.

    Two shapes, both common:

    * a fixed list of load points ("measure at qps 1, 2, 4"), or
    * a capacity search ("find the highest qps that still meets the SLA").

    The capacity search is a per-point inner loop rather than a strategy
    concern, because it is not exploring the *config* space — it is
    characterizing one config.
    """

    fixed: Tuple[LoadPoint, ...] = ()
    search: Optional["CapacitySearch"] = None

    def __post_init__(self) -> None:
        if not self.fixed and self.search is None:
            self.fixed = (LoadPoint(),)

    @property
    def estimated_runs(self) -> int:
        if self.search is not None:
            return self.search.max_rounds + 1
        return max(1, len(self.fixed))


@dataclass
class CapacitySearch:
    """Bisect request rate against the run's constraints.

    ``tolerance`` is relative width of the bracket at which to stop; without a
    round cap a noisy SLA boundary will bisect forever, so ``max_rounds`` is
    mandatory, not advisory.
    """

    lower: float
    upper: float
    tolerance: float = 0.1
    max_rounds: int = 5
    max_concurrency: Optional[int] = None


class MeasurementDriver(ABC):
    """Evaluates one point and returns its measurement.

    Nothing here assumes a server. A driver that sweeps many candidates in a
    single call (cutlass profiler, triton autotune) returns the submitted
    point's measurement and attaches the rest as ``Measurement.discovered``.
    """

    name: str = "driver"

    #: Metrics this driver promises to emit. Validated against the objective
    #: and constraints at config-load time, so a config asking to constrain
    #: ``p99_tpot_ms`` against a driver that cannot report it fails loudly
    #: instead of silently passing every candidate.
    provides: Tuple[str, ...] = ()

    # ---- run-scoped setup ------------------------------------------------

    def prepare(self, workloads: Sequence[Workload]) -> Sequence[Workload]:
        """Materialize datasets once per run; return workloads with paths set.

        Every trial must see byte-identical input. Regenerating a sampled
        dataset per trial silently makes candidates incomparable.
        """
        return workloads

    def provenance(self) -> Provenance:
        """Capture the environment fingerprint recorded with every trial."""
        return Provenance()

    # ---- per-trial -------------------------------------------------------

    @abstractmethod
    def measure(
        self,
        trial: Trial,
        slot: TrialSlot,
        timeout_s: Optional[float] = None,
        prune_check: Optional[PruneCheck] = None,
    ) -> Measurement:
        """Run one trial and return its measurement.

        Contract:

        * never raise for an expected failure (OOM, bad flag, crash) — return a
          ``FAILED``/``TIMEOUT`` measurement with a :class:`FailureKind` and the
          log path in ``artifacts`` so the reporter can explain it;
        * always tear down the server, including on timeout and ``KeyboardInterrupt``;
        * honor ``timeout_s``, which the orchestrator derives from the remaining
          budget, not from a per-trial constant;
        * call ``prune_check`` with partial metrics when it can, and return a
          ``PRUNED`` measurement if it returns a reason.
        """

    def teardown(self) -> None:
        """Release run-scoped resources (prepared datasets, caches)."""

    # ---- reporting helpers ----------------------------------------------

    def describe_trial(self, trial: Trial, slot: TrialSlot) -> Sequence[str]:
        """Commands this trial would run, for a plan to print before spending.

        A driver that builds them through the real parsers turns a mistyped
        flag into a dry-run failure rather than one that waits for a model to
        load. Default is empty: a driver with no commands to show says nothing.
        """
        return ()

    def render_launch_command(self, point: Point) -> Optional[str]:
        """The command a user would run to reproduce this point, if any.

        Part of the driver because only it knows how a point maps to a CLI —
        which flags are booleans, which are repeated, what the prefix is.
        Kernel drivers have no launch command and return ``None``.
        """
        return None

    def render_config(self, point: Point) -> Optional[Mapping[str, Any]]:
        """The point as a config the runtime can load directly, if any.

        For the serving driver this is a ``ServerArgs`` YAML consumable by
        ``--config``; for a kernel driver it is the per-shape entry the kernel
        reads from its config directory. Either way, this is what makes a
        tuning result deployable rather than something a human retypes.
        """
        return None
