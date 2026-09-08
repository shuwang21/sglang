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

__all__ = ["MeasurementDriver", "LoadPlan"]


@dataclass
class LoadPlan:
    """The load points to measure one candidate at, e.g. qps 1, 2 and 4."""

    fixed: Tuple[LoadPoint, ...] = ()

    def __post_init__(self) -> None:
        if not self.fixed:
            self.fixed = (LoadPoint(),)

    @property
    def estimated_runs(self) -> int:
        return max(1, len(self.fixed))


class MeasurementDriver(ABC):
    """Evaluates one point and returns its measurement.

    Nothing here assumes a server: one point in, one measurement out.
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
