"""Executor: where and how many trials run at once.

The executor owns *placement and concurrency*; the driver (see
``sglang.autotune.measure``) owns *launch and measurement*. Keeping them apart
is what lets the same local driver run one trial on 8 GPUs or eight trials on
one GPU each, and lets a future Slurm/k8s executor reuse the driver unchanged.

It is also the seam that makes the orchestrator testable: a mock executor that
returns metrics from an analytic function exercises budget, dedup, resume,
pruning, and reporting with no hardware at all.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

from sglang.autotune.types import (
    FailureKind,
    Measurement,
    Trial,
    TrialStatus,
)

__all__ = ["TrialSlot", "Executor", "SerialExecutor", "Submission"]


@dataclass(frozen=True)
class Submission:
    """A trial plus the per-trial limits the orchestrator attaches to it.

    ``timeout_s`` is derived from the remaining run budget, so it changes from
    one submission to the next. It travels with the trial so an executor that
    hands work to a subprocess or another host has nothing to look up.
    """

    trial: Trial
    timeout_s: Optional[float] = None


@dataclass(frozen=True)
class TrialSlot:
    """An exclusive reservation of hardware and a port for one trial.

    Concurrency is the largest wall-clock win available for small models — a
    1-GPU candidate on an 8-GPU host can run eight-wide — and it is entirely a
    matter of partitioning devices and ports correctly, which is why that
    bookkeeping is a first-class type rather than an implementation detail.
    """

    index: int
    gpu_ids: Tuple[int, ...] = ()
    host: str = "127.0.0.1"
    port: int = 30000
    node: str = "local"

    @property
    def visible_devices(self) -> str:
        return ",".join(str(g) for g in self.gpu_ids)

    def env(self) -> dict:
        return {"CUDA_VISIBLE_DEVICES": self.visible_devices} if self.gpu_ids else {}


class Executor(ABC):
    """Runs trials, possibly several at a time, possibly on other machines."""

    name: str = "executor"

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Total concurrent trials this executor can host."""

    @property
    def free_slots(self) -> int:
        """Slots available right now; the orchestrator asks for this many points."""
        return self.capacity

    @abstractmethod
    def submit(
        self,
        trial: Trial,
        *,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Queue a trial. Must not block on the trial completing.

        ``timeout_s`` is forwarded to the driver's ``measure()``; see
        :class:`Submission`.
        """

    @abstractmethod
    def drain(self, block: bool = True) -> Iterator[Measurement]:
        """Yield measurements for finished trials.

        With ``block=True``, wait for at least one result if any are in flight.
        Results are yielded in completion order, not submission order — a
        strategy that needs ordering must impose it via :meth:`ask`.

        Every submitted trial yields exactly one measurement, including trials
        aborted by :meth:`cancel_all` (``FAILED`` with
        ``FailureKind.CANCELLED``). The orchestrator counts in-flight trials
        by submissions minus yields; an executor that swallows a cancelled
        trial leaves it blocked here forever.
        """

    def map(self, trials: Iterable[Trial]) -> Iterator[Measurement]:
        """Convenience: submit everything, yield results as they land."""
        pending = 0
        for trial in trials:
            self.submit(trial)
            pending += 1
            while self.free_slots == 0:
                for measurement in self.drain(block=True):
                    pending -= 1
                    yield measurement
        while pending > 0:
            for measurement in self.drain(block=True):
                pending -= 1
                yield measurement

    def cancel_all(self) -> None:
        """Best-effort abort of in-flight trials, e.g. on SIGINT.

        Implementations must guarantee that no server process outlives the
        executor — an orphaned server holding a port and 80GB of HBM will
        poison every subsequent trial in the run. Aborted trials still surface
        through :meth:`drain`, as ``CANCELLED`` measurements.
        """

    def close(self) -> None:
        self.cancel_all()

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class SerialExecutor(Executor):
    """One trial at a time, in this process's control.

    The default, and the reference semantics every other executor must match.
    Subclasses supply :meth:`run_one`; the queueing here is trivial on purpose.
    """

    name = "serial"

    def __init__(self, slot: Optional[TrialSlot] = None) -> None:
        self.slot = slot or TrialSlot(index=0)
        self._queue: List[Submission] = []
        self._cancelled = False

    @property
    def capacity(self) -> int:
        return 1

    @property
    def free_slots(self) -> int:
        return 0 if self._queue else 1

    def submit(
        self,
        trial: Trial,
        *,
        timeout_s: Optional[float] = None,
    ) -> None:
        self._queue.append(Submission(trial=trial, timeout_s=timeout_s))

    def cancel_all(self) -> None:
        self._cancelled = True

    def drain(self, block: bool = True) -> Iterator[Measurement]:
        while self._queue:
            submission = self._queue.pop(0)
            trial = submission.trial
            started = time.time()
            if self._cancelled:
                yield Measurement(
                    trial=trial,
                    status=TrialStatus.FAILED,
                    failure=FailureKind.CANCELLED,
                    message="cancelled before start",
                    started_at=started,
                )
                continue
            try:
                measurement = self.run_one(
                    trial,
                    self.slot,
                    timeout_s=submission.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - a failed trial is data
                measurement = Measurement(
                    trial=trial,
                    status=TrialStatus.FAILED,
                    failure=FailureKind.UNKNOWN,
                    message=repr(exc),
                    started_at=started,
                    duration_s=time.time() - started,
                )
            yield measurement
            if not block:
                return

    @abstractmethod
    def run_one(
        self,
        trial: Trial,
        slot: TrialSlot,
        *,
        timeout_s: Optional[float] = None,
    ) -> Measurement:
        """Execute a single trial to completion and return its measurement."""
