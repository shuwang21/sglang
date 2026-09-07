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
from sglang.autotune.objective import (
    Direction,
    MetricThreshold,
    ScalarObjective,
    margin,
)
from sglang.autotune.report import MarkdownReporter
from sglang.autotune.space.base import FeasibilityRule
from sglang.autotune.space.simple import SimpleSpace
from sglang.autotune.store import JsonlStore
from sglang.autotune.strategy.grid import GridStrategy
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
    provenance_model="mock-model",
    strategy=None,
) -> TuneTask:
    driver = MockDriver(metric_fn, trial_seconds=trial_seconds, model=provenance_model)
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
        strategy=strategy or RandomStrategy(seed=0, max_points=max_points),
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

    def test_a_different_model_is_a_different_trial(self):
        """A run reused another model's recorded results.

        Only the driver knows which model a trial measured, and the trial key
        covered the knobs, the workload, and the environment but not that. Two
        runs over the same knob grid with different models therefore collided
        in the store, and the second reported the first's numbers as its own.
        """
        first = _build_task(self.tmp, provenance_model="model-a")
        tune(first)
        first.store.close()

        second = _build_task(self.tmp, provenance_model="model-b")
        tune(second)

        self.assertEqual(len(second.driver.measured), 6)
        self.assertEqual(len(second.store.history()), 12)

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


class TestGridStrategy(CustomTestCase):
    """Exhaustive search, and the margin a single-shot run cannot judge."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_covers_every_feasible_point_exactly_once(self):
        task = _build_task(self.tmp, strategy=GridStrategy(), rules=[_RejectTp2()])
        result = tune(task)

        # 3 tp_size x 2 backends, less the two the rule rejects.
        measured = [t.point.fingerprint for t in task.driver.measured]
        self.assertEqual(len(measured), 4)
        self.assertEqual(len(set(measured)), 4)
        self.assertTrue(task.strategy.is_exhausted())
        self.assertEqual(result.best_point.get("tp_size"), 4)

    def test_repeats_are_kept_apart_and_ranked_on_their_median(self):
        """Three measurements of one point, not one measurement counted thrice.

        Two failure modes at once. The trial key did not mention the attempt,
        so the store deduped every repeat down to the first and a resumed run
        saw one. And ranking took whatever single measurement it had, so an
        erratic point that peaked high beat a steady better one.
        """
        calls: dict = {}

        def flaky(point, workload):
            slot = (point.get("tp_size"), point.get("backend"))
            series = {
                (1, "fa3"): [100.0, 300.0, 200.0],  # erratic, median 200
                (1, "triton"): [210.0, 210.0, 210.0],  # steady, median 210
            }.get(slot, [50.0, 50.0, 50.0])
            n = calls.get(slot, 0)
            calls[slot] = n + 1
            return {"output_throughput": series[n], "p99_ttft_ms": 1.0}

        task = _build_task(self.tmp, metric_fn=flaky, strategy=GridStrategy())
        task.repeats = 3
        result = tune(task)

        self.assertEqual(len(task.driver.measured), 18)
        self.assertEqual(len(task.store.history()), 18)
        self.assertEqual(result.best_point.get("backend"), "triton")
        # (300 - 100) / 200, the erratic point's own range over its median.
        self.assertAlmostEqual(max(e.spread for e in result.ranking), 1.0)

    def test_estimated_trials_counts_every_repeat(self):
        """A plan that understates the cost threefold is worse than none."""
        task = _build_task(self.tmp, strategy=GridStrategy())
        task.repeats = 3
        self.assertEqual(task.estimated_trials(), 18)

    def test_a_resume_retries_what_failed(self):
        """A resume kept a run's failures and then reported nothing to do.

        Every trial of the first run failed. The second run replayed those
        into the strategy, which counted the points as proposed and declared
        the space exhausted, and the store handed the failures straight back
        as the answer -- so a bad first attempt was permanent for that output
        directory.
        """
        first = _build_task(
            self.tmp, strategy=GridStrategy(), metric_fn=lambda point, workload: None
        )
        tune(first)
        first.store.close()
        self.assertEqual(len(first.driver.measured), 6)

        second = _build_task(self.tmp, strategy=GridStrategy())
        result = tune(second)

        self.assertEqual(len(second.driver.measured), 6)
        self.assertEqual(len(second.store.history()), 6)
        self.assertIsNotNone(result.best)

    def test_a_resume_keeps_a_point_the_space_rejected(self):
        """An infeasible verdict comes from the space, so a rerun repeats it."""
        first = _build_task(self.tmp, strategy=GridStrategy(), rules=[_RejectTp2()])
        tune(first)
        first.store.close()

        second = _build_task(self.tmp, strategy=GridStrategy(), rules=[_RejectTp2()])
        tune(second)

        self.assertEqual(len(second.driver.measured), 0)

    def test_a_resumed_grid_runs_only_what_is_left(self):
        first = _build_task(
            self.tmp, strategy=GridStrategy(), budget=Budget(max_trials=2)
        )
        tune(first)
        first.store.close()

        second = _build_task(self.tmp, strategy=GridStrategy())
        tune(second)

        self.assertEqual(len(second.driver.measured), 4)
        self.assertEqual(len(second.store.history()), 6)

    def test_margin_reports_the_lead_over_the_runner_up(self):
        """A 30% lead and a 0.3% one read identically without this.

        Ranking states a winner with no sense of scale, so a run whose top two
        differ by noise looks exactly like one with a real answer.
        """
        result = tune(_build_task(self.tmp, strategy=GridStrategy()))
        ranked = result.ranking_by_bucket[""]

        # fa3 at tp=4 scores 420, triton at tp=4 scores 400: a 5% lead.
        self.assertAlmostEqual(margin(ranked), 0.05, places=6)
        self.assertIsNone(margin(ranked[:1]))

    def test_margin_handles_a_minimised_metric(self):
        task = _build_task(self.tmp, strategy=GridStrategy())
        # Minimise the same metric, so the winner is the worst throughput: the
        # sign lives in the objective's components, not in margin.
        task.objective = ScalarObjective(
            metric="output_throughput", direction=Direction.MINIMIZE
        )
        result = tune(task)
        ranked = result.ranking_by_bucket[""]

        self.assertEqual(ranked[0].measurement.metric("output_throughput"), 100.0)
        # 100 beats 120 by a fifth of 120, whichever way the metric points.
        self.assertAlmostEqual(margin(ranked), 20.0 / 120.0, places=6)
