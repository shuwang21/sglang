"""Concurrent trials, one worker process per GPU slot."""

from __future__ import annotations

import logging
import os
import select
import subprocess
import sys
import tempfile
from typing import Dict, Iterator, List, Optional, Sequence

from sglang.autotune.executor.base import Executor, PruneCheck, Submission, TrialSlot
from sglang.autotune.executor.worker import read_message, write_message
from sglang.autotune.measure import MeasurementDriver
from sglang.autotune.registry import register_executor
from sglang.autotune.types import (
    FailureKind,
    Measurement,
    Trial,
    TrialStatus,
    Workload,
)

logger = logging.getLogger(__name__)

__all__ = ["PoolExecutor", "build_slots"]

_WORKER_MODULE = "sglang.autotune.executor.worker"


def build_slots(
    gpu_count: int,
    gpus_per_trial: int,
    *,
    host: str = "127.0.0.1",
    base_port: int = 31000,
    port_stride: int = 100,
) -> List[TrialSlot]:
    """Partition the GPUs into reservations, one per concurrent trial.

    Ports are spaced rather than consecutive: a server takes a range above the
    one it is given, so adjacent slots would otherwise collide.
    """
    per = max(1, gpus_per_trial)
    count = max(1, gpu_count // per)
    return [
        TrialSlot(
            index=i,
            gpu_ids=tuple(range(i * per, (i + 1) * per)),
            host=host,
            port=base_port + i * port_stride,
        )
        for i in range(count)
    ]


class _Worker:
    """A slot's subprocess, and the one trial it may be running."""

    def __init__(
        self,
        slot: TrialSlot,
        driver: MeasurementDriver,
        workloads: Sequence[Workload],
    ) -> None:
        self.slot = slot
        self.trial: Optional[Trial] = None
        # Pinning has to be in the child's environment from the start: torch
        # reads CUDA_VISIBLE_DEVICES when it first initialises CUDA, and by
        # then a forked child has already inherited the parent's context.
        env = {**os.environ, **slot.env()}
        self._errors = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._errors,
            env=env,
        )
        write_message(
            self.process.stdin,
            {"driver": driver, "slot": slot, "workloads": list(workloads)},
        )
        if read_message(self.process.stdout) is None:
            raise RuntimeError(
                f"slot {slot.index} worker died during setup:\n{self.stderr_tail()}"
            )

    def stderr_tail(self, limit: int = 2000) -> str:
        """What the child said before it stopped talking."""
        self._errors.flush()
        self._errors.seek(0)
        return self._errors.read()[-limit:]

    @property
    def busy(self) -> bool:
        return self.trial is not None

    def send(self, submission: Submission) -> None:
        self.trial = submission.trial
        write_message(
            self.process.stdin,
            {"trial": submission.trial, "timeout_s": submission.timeout_s},
        )

    def receive(self) -> Optional[Measurement]:
        """Read the finished measurement. Only call once the pipe is readable."""
        if self.trial is None:
            return None
        reply = read_message(self.process.stdout)
        trial, self.trial = self.trial, None
        if reply is None:
            # The pipe closed, so the child is gone: an OOM kill or a segfault
            # in the driver. The trial still owes the orchestrator a result.
            return Measurement(
                trial=trial,
                status=TrialStatus.FAILED,
                failure=FailureKind.SERVER_CRASH,
                message=(
                    f"slot {self.slot.index} worker exited during the trial:\n"
                    f"{self.stderr_tail()}"
                ),
            )
        return reply["measurement"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        for handle in (self.process.stdin, self.process.stdout, self._errors):
            if handle is not None and not handle.closed:
                handle.close()


@register_executor("pool")
class PoolExecutor(Executor):
    """One trial per slot, several slots at once.

    The largest wall-clock win available whenever a candidate needs fewer GPUs
    than the host has: eight one-GPU kernel trials on eight GPUs. It is not a
    win for a candidate that already spans every GPU -- there
    ``concurrent_trials`` is one and the serial executor is simpler.

    Pruning is not supported. The predicate is a callable, which cannot cross
    a process boundary, and no strategy implements one; passing it is an error
    rather than a silent no-op.
    """

    def __init__(
        self,
        driver: MeasurementDriver,
        slots: Sequence[TrialSlot],
        workloads: Sequence[Workload] = (),
    ) -> None:
        if not slots:
            raise ValueError("a pool needs at least one slot")
        self.driver = driver
        self.slots = list(slots)
        self.workloads = list(workloads)
        self._workers: Dict[int, _Worker] = {}
        self._queue: List[Submission] = []
        self._abandoned: List[Trial] = []
        self._cancelled = False

    @property
    def capacity(self) -> int:
        return len(self.slots)

    @property
    def free_slots(self) -> int:
        running = sum(1 for w in self._workers.values() if w.busy)
        return max(0, self.capacity - running - len(self._queue))

    def submit(
        self,
        trial: Trial,
        *,
        timeout_s: Optional[float] = None,
        prune_check: Optional[PruneCheck] = None,
    ) -> None:
        if prune_check is not None:
            raise NotImplementedError(
                "pool executor cannot forward a prune predicate to a subprocess"
            )
        self._queue.append(Submission(trial=trial, timeout_s=timeout_s))
        self._dispatch()

    def _dispatch(self) -> None:
        if self._cancelled:
            return
        for slot in self.slots:
            if not self._queue:
                return
            worker = self._workers.get(slot.index)
            if worker is None:
                worker = _Worker(slot, self.driver, self.workloads)
                self._workers[slot.index] = worker
                logger.info(
                    "slot %d on GPU %s", slot.index, slot.visible_devices or "-"
                )
            if not worker.busy:
                worker.send(self._queue.pop(0))

    def drain(self, block: bool = True) -> Iterator[Measurement]:
        # Trials the run gave up on still owe the orchestrator a result: it
        # counts in flight as submissions minus yields, and would wait forever.
        while self._abandoned:
            yield Measurement(
                trial=self._abandoned.pop(0),
                status=TrialStatus.FAILED,
                failure=FailureKind.CANCELLED,
                message="cancelled before completion",
            )

        while True:
            busy = {w.process.stdout: w for w in self._workers.values() if w.busy}
            if not busy:
                return
            # Waiting on whichever pipe speaks first, rather than on each in
            # turn: a slot blocked behind a slow neighbour is a slot not
            # working, which is the whole point of a pool.
            ready, _, _ = select.select(list(busy), [], [], None if block else 0)
            for stream in ready:
                measurement = busy[stream].receive()
                if measurement is not None:
                    yield measurement
            self._dispatch()
            if not block or not ready:
                return

    def cancel_all(self) -> None:
        self._cancelled = True
        self._abandoned.extend(s.trial for s in self._queue)
        self._queue.clear()
        for worker in self._workers.values():
            if worker.trial is not None:
                self._abandoned.append(worker.trial)
                worker.trial = None
            worker.close()

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()
