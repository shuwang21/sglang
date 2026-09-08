"""The tuning loop.

Everything interesting lives in the pluggable components; this module is
deliberately the least clever code in the framework. Its whole job is:

    ask the strategy for points
      -> expand each point into trials (workload x load)
      -> drop infeasible and already-measured ones
      -> run them
      -> record, tell, repeat until the budget or the strategy is done

Keeping policy out of here is what makes the loop reusable: swapping ASHA for
coordinate descent, or a local executor for a Slurm one, changes nothing below.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Dict, Iterator, Optional, Sequence

from sglang.autotune.objective import pareto_front, rank_by_bucket
from sglang.autotune.report import Reporter
from sglang.autotune.task import TuneResult, TuneTask
from sglang.autotune.types import (
    BudgetExhausted,
    Measurement,
    Point,
    Trial,
    TrialStatus,
)

logger = logging.getLogger(__name__)

__all__ = ["Orchestrator", "tune"]


class Orchestrator:
    """Drives one :class:`TuneTask` to completion."""

    def __init__(self, task: TuneTask, reporters: Sequence[Reporter] = ()) -> None:
        self.task = task
        self.reporters = list(reporters)
        self._interrupted = False
        self._completed = 0
        self._total = 0
        self._partial_reason: Optional[str] = None
        self._previous_handlers: Dict[int, object] = {}

    # ---- entry point -----------------------------------------------------

    def run(self) -> TuneResult:
        task = self.task
        errors = task.validate()
        if errors:
            raise ValueError("invalid tune task:\n  " + "\n  ".join(errors))

        started_at = time.time()
        task.budget.start(started_at)
        self._total = task.estimated_trials()
        task.output_dir.mkdir(parents=True, exist_ok=True)

        # One dataset materialization for the whole run: every candidate must
        # be measured against byte-identical input to be comparable.
        workloads = list(task.driver.prepare(task.workloads))
        task.provenance = task.driver.provenance()

        recorded = list(task.store.history()) if (task.store and task.resume) else []
        # A measurement taken against another model, GPU, or build says nothing
        # about this run. The store still dedups on the full trial key, but the
        # strategy's seen-set is keyed on the point alone, so stale history
        # would stop it proposing anything at all.
        same_env = [
            m
            for m in recorded
            if m.trial.provenance_fingerprint == task.provenance.fingerprint
        ]
        if len(same_env) != len(recorded):
            logger.info(
                "ignoring %d trials recorded in a different environment",
                len(recorded) - len(same_env),
            )
        history = [m for m in same_env if m.status.is_settled]
        if len(history) != len(same_env):
            logger.info("retrying %d trials that failed", len(same_env) - len(history))
        if history:
            logger.info("resuming with %d recorded trials", len(history))
        task.strategy.setup(
            space=task.space,
            objective=task.objective,
            constraints=task.constraints,
            budget=task.budget,
            history=history,
        )

        self._install_signal_handlers()
        try:
            self._loop(workloads)
        except BudgetExhausted as exc:
            self._partial_reason = str(exc)
            logger.info("stopping: %s", exc)
        except KeyboardInterrupt:
            self._partial_reason = "interrupted by user"
        finally:
            self._restore_signal_handlers()
            task.executor.close()
            task.driver.teardown()

        result = self._finalize(started_at)
        # After _finalize, which reads history back out of the store.
        if task.store is not None:
            task.store.close()
        return result

    # ---- the loop --------------------------------------------------------

    def _loop(self, workloads: Sequence) -> None:
        task = self.task
        in_flight = 0

        while not self._interrupted:
            task.budget.raise_if_exhausted()

            free = max(0, task.executor.free_slots)
            points = task.strategy.ask(free) if free else []

            for point in points:
                for trial in self._expand(point, workloads):
                    disposition = self._pre_screen(trial)
                    if disposition is not None:
                        self._observe(disposition)
                        continue
                    task.executor.submit(
                        trial, timeout_s=task.budget.remaining_seconds()
                    )
                    in_flight += 1

            if in_flight == 0:
                if not points and (task.strategy.is_exhausted() or free > 0):
                    logger.info("search space exhausted")
                    break
                continue

            for measurement in task.executor.drain(block=True):
                in_flight -= 1
                self._observe(measurement)
                task.budget.spend(measurement, gpus=task.hardware.gpus_per_trial or 1)
                if task.budget.is_exhausted():
                    self._partial_reason = task.budget.exhausted_reason()
                    task.executor.cancel_all()
                    return

    def _expand(self, point: Point, workloads: Sequence) -> Iterator[Trial]:
        """One point becomes one trial per workload, load setting, and repeat."""
        task = self.task
        loads = task.load_plan.fixed
        fidelity = task.fidelity_for(point)
        for workload in workloads:
            for load in loads:
                for attempt in range(fidelity.repeats):
                    yield Trial(
                        point=point,
                        workload=workload,
                        load=load,
                        fidelity=fidelity,
                        attempt=attempt,
                        provenance_fingerprint=task.provenance.fingerprint,
                        bucket=task.bucket_of(workload, load),
                    )

    def _pre_screen(self, trial: Trial) -> Optional[Measurement]:
        """Reject before spending a GPU. Returns a measurement if rejected.

        Two cheap gates, in cost order: space feasibility (free), then the
        store (one lookup). Both produce a recorded measurement rather than a
        silent skip, so the run's trial count reconciles with its report.
        """
        task = self.task

        feasibility = task.space.feasible(trial.point)
        if not feasibility.ok:
            return Measurement(
                trial=trial,
                status=TrialStatus.INFEASIBLE,
                message="; ".join(feasibility.reasons),
            )

        if task.resume and task.store is not None:
            existing = task.store.lookup(trial.key)
            if existing is not None and existing.status.is_settled:
                logger.debug("reusing recorded trial %s", trial.key)
                return existing

        return None

    def _observe(self, measurement: Measurement) -> None:
        """Record, then inform the strategy. Order matters for crash safety."""
        task = self.task
        if task.store is not None and measurement.status is not TrialStatus.SKIPPED:
            task.store.record(measurement)
        improved = task.strategy.state.observe(
            measurement, task.objective, task.constraints
        )
        task.strategy.tell(measurement)
        self._completed += 1
        logger.info("%s", self._progress_line(measurement, improved=improved))

    def _progress_line(self, measurement: Measurement, *, improved: bool) -> str:
        """One line per trial: a silent run looks the same as a hung one."""
        seen = f"{self._completed}/{self._total}" if self._total else self._completed
        bucket = measurement.trial.bucket or measurement.trial.workload.name
        if measurement.failure is not None:
            body = f"{measurement.failure.value}: {measurement.message}"
        elif measurement.metrics:
            body = ", ".join(
                f"{k}={v:g}" for k, v in sorted(measurement.metrics.items())
            )
        else:
            body = measurement.message or measurement.status.value
        best = " <- best" if improved else ""
        return f"[{seen}] {bucket} {measurement.status.value} {body}{best}"

    # ---- finalization ----------------------------------------------------

    def _finalize(self, started_at: float) -> TuneResult:
        task = self.task
        measurements = list(task.store.history()) if task.store else []
        by_bucket = rank_by_bucket(measurements, task.objective, task.constraints)
        best_by_bucket = {
            bucket: best
            for bucket, ranked in by_bucket.items()
            if (best := next((e for e in ranked if e.rankable), None)) is not None
        }
        ranking = [e for ranked in by_bucket.values() for e in ranked]
        single = len(by_bucket) == 1
        only = next(iter(by_bucket.values())) if single else []

        result = TuneResult(
            task=task,
            measurements=measurements,
            ranking=ranking,
            ranking_by_bucket=by_bucket,
            best_by_bucket=best_by_bucket,
            best=next(iter(best_by_bucket.values()), None) if single else None,
            pareto=pareto_front(only),
            partial_reason=self._partial_reason,
            output_dir=task.output_dir,
            started_at=started_at,
            duration_s=time.time() - started_at,
        )

        for reporter in self.reporters:
            emit = reporter.emit if result.complete else reporter.emit_partial
            try:
                for path in emit(result, task.output_dir):
                    logger.info("wrote %s", path)
            except Exception:  # noqa: BLE001 - one bad reporter must not
                # discard a completed run's results
                logger.exception("reporter %s failed", reporter.name)

        return result

    # ---- signals ---------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """First Ctrl-C finishes gracefully; second one is the escape hatch.

        A tuning run holds GPUs and child server processes. Exiting without
        teardown leaves an orphan holding the port and the HBM, which poisons
        every later run on the box.
        """

        def handler(signum: int, frame: object) -> None:
            if self._interrupted:
                self._restore_signal_handlers()
                raise KeyboardInterrupt
            self._interrupted = True
            self._partial_reason = f"interrupted ({signal.Signals(signum).name})"
            logger.warning("interrupt received; finishing current trials")
            self.task.executor.cancel_all()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[sig] = signal.signal(sig, handler)
            except ValueError:
                pass  # not on the main thread

    def _restore_signal_handlers(self) -> None:
        for sig, previous in self._previous_handlers.items():
            try:
                signal.signal(sig, previous)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
        self._previous_handlers.clear()


def tune(task: TuneTask, reporters: Sequence[Reporter] = ()) -> TuneResult:
    """Run a tuning task. The framework's one-line public API."""
    return Orchestrator(task, reporters).run()
