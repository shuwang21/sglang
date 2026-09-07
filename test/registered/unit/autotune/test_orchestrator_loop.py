"""The orchestrator loop, exercised end to end with no server and no GPU.

This is what the mock driver exists for: budget accounting, feasibility
rejection, dedup, resume, pruning, SLA enforcement, bucketed ranking, and
reporting are all decided in the loop, and none of them need hardware to be
wrong. A regression here is a wrong recommendation, not a slow one.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import tempfile
import unittest
from pathlib import Path

from sglang.autotune import (
    Budget,
    HardwareSpec,
    LoadPlan,
    LoadPoint,
    ModelSpec,
    Point,
    TrialStatus,
    TuneTask,
    Workload,
    tune,
)
from sglang.autotune.driver import MockDriver
from sglang.autotune.executor.local import LocalExecutor
from sglang.autotune.objective import Direction, MetricThreshold, ScalarObjective
from sglang.autotune.report import MarkdownReporter
from sglang.autotune.space.base import FeasibilityRule
from sglang.autotune.space.simple import SimpleSpace
from sglang.autotune.store import JsonlStore
from sglang.autotune.strategy.random import RandomStrategy
from sglang.test.test_utils import CustomTestCase

KNOBS = {"tp_size": [1, 2, 4], "backend": ["fa3", "triton"]}


def _throughput(point: Point, workload: Workload) -> dict:
    """Throughput rises with tp_size; fa3 is faster but breaches the SLA at tp=4."""
    base = 100.0 * point.get("tp_size")
    bonus = 20.0 if point.get("backend") == "fa3" else 0.0
    ttft = 100.0 * point.get("tp_size")
    return {"output_throughput": base + bonus, "p99_ttft_ms": ttft}


class _RejectTp2(FeasibilityRule):
    name = "no_tp2"

    def check(self, point, context):
        return "tp_size=2 unsupported" if point.get("tp_size") == 2 else None


def _build_task(
    tmp: Path,
    *,
    metric_fn=_throughput,
    constraints=(),
    rules=(),
    max_points=None,
    budget=None,
    store=None,
    bucket_of=None,
    workloads=None,
    load_plan=None,
    trial_seconds=1.0,
) -> TuneTask:
    driver = MockDriver(metric_fn, trial_seconds=trial_seconds)
    kwargs = {}
    if bucket_of is not None:
        kwargs["bucket_of"] = bucket_of
    return TuneTask(
        name="mock-run",
        model=ModelSpec(path="mock/model"),
        hardware=HardwareSpec(gpu_count=1, gpus_per_trial=1),
        workloads=workloads or [Workload(name="chat", kind="random")],
        load_plan=load_plan or LoadPlan(),
        space=SimpleSpace(KNOBS, rules=rules),
        strategy=RandomStrategy(seed=0, max_points=max_points),
        driver=driver,
        executor=LocalExecutor(driver),
        objective=ScalarObjective(
            metric="output_throughput", direction=Direction.MAXIMIZE
        ),
        constraints=constraints,
        store=store if store is not None else JsonlStore(tmp / "trials.jsonl"),
        budget=budget or Budget(),
        output_dir=tmp,
        **kwargs,
    )


class TestOrchestratorLoop(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exhausting_the_space_ranks_every_point_once(self):
        result = tune(_build_task(self.tmp))

        self.assertTrue(result.complete)
        self.assertEqual(len(result.ranking), 6)  # 3 tp_size x 2 backends
        self.assertEqual(result.best_point.get("tp_size"), 4)
        self.assertEqual(result.best_point.get("backend"), "fa3")

    def test_estimated_trials_counts_points_times_workloads(self):
        """`plan` reported one point's cost as the whole run's.

        The estimate multiplied workloads by load settings and stopped, so a
        4-point search over 2 workloads was announced as 2 trials instead of 8
        -- the one number a user reads before committing GPU hours.
        """
        one = _build_task(self.tmp, max_points=3)
        self.assertEqual(one.estimated_trials(), 3)

        two = _build_task(
            self.tmp,
            max_points=4,
            workloads=[Workload(name="a", kind="k"), Workload(name="b", kind="k")],
        )
        self.assertEqual(two.estimated_trials(), 8)

        # No cap on the strategy falls back to the space itself: 3 x 2 knobs.
        self.assertEqual(_build_task(self.tmp).estimated_trials(), 6)

        # A trial budget only ever lowers the estimate: 4 x 2 = 8, capped at 5.
        capped = _build_task(
            self.tmp,
            max_points=4,
            budget=Budget(max_trials=5),
            workloads=[Workload(name="a", kind="k"), Workload(name="b", kind="k")],
        )
        self.assertEqual(capped.estimated_trials(), 5)

    def test_sla_violation_cannot_outrank_a_compliant_candidate(self):
        result = tune(
            _build_task(
                self.tmp,
                constraints=[MetricThreshold(metric="p99_ttft_ms", max_value=250.0)],
            )
        )

        # tp=4 wins on raw throughput but breaches the 250ms ceiling.
        self.assertEqual(result.best_point.get("tp_size"), 2)
        self.assertTrue(result.ranking[0].feasible)
        self.assertFalse(result.ranking[-1].feasible)

    def test_infeasible_points_are_recorded_and_never_measured(self):
        task = _build_task(self.tmp, rules=[_RejectTp2()])
        result = tune(task)

        infeasible = [
            m for m in result.measurements if m.status is TrialStatus.INFEASIBLE
        ]
        self.assertEqual(len(infeasible), 2)  # tp=2 x 2 backends
        self.assertTrue(all("tp_size=2" in m.message for m in infeasible))
        measured = {t.point.get("tp_size") for t in task.driver.measured}
        self.assertNotIn(2, measured)

    def test_trial_budget_stops_the_run_and_still_reports(self):
        result = tune(_build_task(self.tmp, budget=Budget(max_trials=2)))

        self.assertFalse(result.complete)
        self.assertIn("trial budget", result.partial_reason)
        self.assertEqual(len(result.measurements), 2)
        self.assertIsNotNone(result.best_point)

    def test_gpu_hour_budget_counts_trial_duration(self):
        budget = Budget(max_gpu_hours=2.0 / 3600.0)
        result = tune(_build_task(self.tmp, budget=budget, trial_seconds=1.0))

        self.assertIn("gpu-hour budget", result.partial_reason)
        self.assertEqual(len(result.measurements), 2)

    def test_resume_reuses_recorded_trials_instead_of_remeasuring(self):
        first = _build_task(self.tmp, budget=Budget(max_trials=3))
        tune(first)
        first.store.close()

        second = _build_task(self.tmp)
        result = tune(second)

        self.assertEqual(len(result.ranking), 6)
        # The three trials from the first run come back from the store; only
        # the remaining three reach the driver.
        self.assertEqual(len(second.driver.measured), 3)

    def test_point_cap_holds_across_a_resume(self):
        """A resumed run drew a fresh batch instead of honouring the cap.

        The strategy counted its own proposals in a per-process field, which
        starts at zero, so `--max-configs 4` became "4 more each time": a run
        resumed once measured 8 points and reported the planned 4.
        """
        first = _build_task(self.tmp, max_points=2)
        tune(first)
        first.store.close()
        measured_first = len(first.driver.measured)

        second = _build_task(self.tmp, max_points=2)
        tune(second)

        self.assertEqual(measured_first, 2)
        self.assertEqual(len(second.driver.measured), 0)
        self.assertEqual(len(second.store.history()), 2)

    def test_store_round_trips_the_trial_key(self):
        task = _build_task(self.tmp)
        tune(task)
        task.store.close()

        reloaded = JsonlStore(self.tmp / "trials.jsonl")
        original = {m.key for m in task.store.history()}
        self.assertEqual({m.key for m in reloaded.history()}, original)
        for measurement in reloaded.history():
            self.assertIsNotNone(task.store.lookup(measurement.key))

    def test_failed_trials_are_kept_with_a_diagnosis(self):
        def sometimes_oom(point: Point, workload: Workload):
            return None if point.get("tp_size") == 4 else _throughput(point, workload)

        result = tune(_build_task(self.tmp, metric_fn=sometimes_oom))

        failed = [m for m in result.measurements if m.status is TrialStatus.FAILED]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(m.failure.value == "oom" for m in failed))
        self.assertEqual(result.best_point.get("tp_size"), 2)

    def test_buckets_produce_one_winner_each(self):
        loads = LoadPlan(
            fixed=(LoadPoint(max_concurrency=4), LoadPoint(max_concurrency=64))
        )

        def by_concurrency(point: Point, workload: Workload):
            return _throughput(point, workload)

        result = tune(
            _build_task(
                self.tmp,
                metric_fn=by_concurrency,
                load_plan=loads,
                bucket_of=lambda w, load: f"mc={load.max_concurrency}",
            )
        )

        self.assertEqual(result.buckets, ["mc=4", "mc=64"])
        self.assertEqual(len(result.best_by_bucket), 2)
        # Ambiguous winner across buckets, so the single-best fields stay empty.
        self.assertIsNone(result.best)

    def test_markdown_report_names_the_winner_and_the_failures(self):
        def sometimes_oom(point: Point, workload: Workload):
            return None if point.get("tp_size") == 4 else _throughput(point, workload)

        task = _build_task(self.tmp, metric_fn=sometimes_oom)
        result = tune(task, reporters=[MarkdownReporter()])

        summary = (self.tmp / "summary.md").read_text(encoding="utf-8")
        self.assertIn("mock-run", summary)
        self.assertIn("## Failures", summary)
        self.assertIn("oom", summary)
        self.assertIn(str(result.best_point.get("tp_size")), summary)


if __name__ == "__main__":
    unittest.main()
