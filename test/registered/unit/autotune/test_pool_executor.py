"""The pool executor, with real subprocesses and no GPU.

Concurrency is the only reason this class exists, so the assertions are about
overlap and about the contract the orchestrator relies on -- every submission
yields exactly one measurement -- rather than about a speedup figure, which
would depend on the machine.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import tempfile
import time
import unittest
from pathlib import Path

from sglang.autotune.driver.mock import MockDriver, sleep_metrics
from sglang.autotune.executor.pool import PoolExecutor, build_slots
from sglang.autotune.measure import LoadPlan
from sglang.autotune.objective import Direction, ScalarObjective
from sglang.autotune.orchestrator import tune
from sglang.autotune.space.simple import SimpleSpace
from sglang.autotune.store import JsonlStore
from sglang.autotune.strategy.grid import GridStrategy
from sglang.autotune.task import HardwareSpec, ModelSpec, TuneTask
from sglang.autotune.types import Point, Trial, TrialStatus, Workload

WORKLOAD = Workload(name="w", kind="mock")


def _trial(seconds: float) -> Trial:
    return Trial(point=Point({"seconds": seconds}), workload=WORKLOAD)


def _pool(slots: int, driver: MockDriver) -> PoolExecutor:
    return PoolExecutor(
        driver,
        build_slots(gpu_count=slots, gpus_per_trial=1),
        workloads=[WORKLOAD],
    )


class TestPoolExecutor(unittest.TestCase):
    def test_trials_actually_overlap(self):
        """Four one-second trials on four slots take about one second."""
        with _pool(4, MockDriver(sleep_metrics)) as pool:
            started = time.time()
            for _ in range(4):
                pool.submit(_trial(1.0))
            results = list(pool.drain(block=True))
            elapsed = time.time() - started

        self.assertEqual(len(results), 4)
        self.assertTrue(all(m.status is TrialStatus.OK for m in results))
        # Serial would be 4s. Allow generously for process startup; the claim
        # is that they ran together, not that the machine is fast.
        self.assertLess(elapsed, 3.0)

    def test_a_slow_trial_does_not_hold_up_its_neighbours(self):
        """Reading slots in turn would serialise them behind the slowest."""
        with _pool(2, MockDriver(sleep_metrics)) as pool:
            pool.submit(_trial(2.0))
            pool.submit(_trial(0.1))
            first = next(iter(pool.drain(block=True)))
            rest = list(pool.drain(block=True))

        self.assertAlmostEqual(first.trial.point.get("seconds"), 0.1)
        self.assertEqual(len(rest), 1)

    def test_every_submission_yields_exactly_one_measurement(self):
        with _pool(2, MockDriver(sleep_metrics)) as pool:
            for seconds in (0.1, 0.1, 0.1, 0.1, 0.1):
                pool.submit(_trial(seconds))
            results = []
            while len(results) < 5:
                results.extend(pool.drain(block=True))

        self.assertEqual(len(results), 5)

    def test_a_cancelled_run_still_accounts_for_its_trials(self):
        """The orchestrator counts in flight as submissions minus yields.

        A cancelled trial that never surfaces leaves that count above zero and
        the loop waiting on a drain that will not produce anything.
        """
        pool = _pool(2, MockDriver(sleep_metrics))
        for _ in range(4):
            pool.submit(_trial(0.2))
        pool.cancel_all()
        results = list(pool.drain(block=False))
        pool.close()

        self.assertEqual(len(results), 4)
        self.assertTrue(
            all(m.failure.value == "cancelled" for m in results),
            [m.failure for m in results],
        )

    def test_a_failing_point_is_data_not_an_exception(self):
        with _pool(2, MockDriver(sleep_metrics)) as pool:
            pool.submit(_trial(0.1))
            pool.submit(
                Trial(point=Point({"seconds": 0.1, "fail": True}), workload=WORKLOAD)
            )
            results = []
            while len(results) < 2:
                results.extend(pool.drain(block=True))

        statuses = {bool(m.trial.point.get("fail")): m.status for m in results}
        self.assertIs(statuses[False], TrialStatus.OK)
        self.assertIs(statuses[True], TrialStatus.FAILED)

    def test_pruning_is_refused_rather_than_ignored(self):
        with _pool(1, MockDriver(sleep_metrics)) as pool:
            with self.assertRaises(NotImplementedError):
                pool.submit(_trial(0.1), prune_check=lambda trial, metrics: "no")


class TestBuildSlots(unittest.TestCase):
    def test_partitions_gpus_and_spaces_ports(self):
        slots = build_slots(gpu_count=8, gpus_per_trial=2)
        self.assertEqual([s.gpu_ids for s in slots], [(0, 1), (2, 3), (4, 5), (6, 7)])
        # A server claims a range above its port, so adjacent slots must not be
        # adjacent numbers.
        self.assertEqual([s.port for s in slots], [31000, 31100, 31200, 31300])

    def test_a_candidate_spanning_every_gpu_leaves_one_slot(self):
        self.assertEqual(len(build_slots(gpu_count=4, gpus_per_trial=4)), 1)


if __name__ == "__main__":
    unittest.main()


class TestPoolUnderTheOrchestrator(unittest.TestCase):
    """The pool driven by the loop, not by direct submits.

    Every earlier case called `submit` itself and so never carried what the
    orchestrator actually sends. It sends a prune predicate unconditionally,
    which the pool refuses, and a whole run died on the first trial.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _task(self, strategy=None) -> TuneTask:
        driver = MockDriver(sleep_metrics)
        return TuneTask(
            name="pooled",
            model=ModelSpec(path="mock/model"),
            hardware=HardwareSpec(gpu_count=2, gpus_per_trial=1),
            workloads=[WORKLOAD],
            load_plan=LoadPlan(),
            space=SimpleSpace({"seconds": [0.1, 0.2, 0.3, 0.4]}),
            strategy=strategy or GridStrategy(),
            driver=driver,
            executor=PoolExecutor(driver, build_slots(2, 1), workloads=[WORKLOAD]),
            objective=ScalarObjective(
                metric="output_throughput", direction=Direction.MAXIMIZE
            ),
            store=JsonlStore(self.tmp / "trials.jsonl"),
            output_dir=self.tmp,
        )

    def test_a_run_completes_through_the_pool(self):
        result = tune(self._task())

        self.assertEqual(len(result.measurements), 4)
        self.assertTrue(all(m.status is TrialStatus.OK for m in result.measurements))
        # Fastest point wins: throughput is 1/seconds.
        self.assertAlmostEqual(result.best_point.get("seconds"), 0.1)

    def test_a_pruning_strategy_is_still_refused(self):
        class Pruner(GridStrategy):
            prunes = True

            def should_prune(self, point, partial_metrics):
                return "no"

        with self.assertRaises(NotImplementedError):
            tune(self._task(strategy=Pruner()))
