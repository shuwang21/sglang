"""``python -m sglang.autotune`` -- tune a model's kernels or its deployment.

Two subcommands, one loop: ``moe`` searches triton tile configs against a shape
derived from the model, ``serve`` searches ServerArgs against a real server
under load. What differs is the space, the driver, and the objective; the
orchestration is the same either way, which is the point of the framework.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from sglang.autotune.executor.local import LocalExecutor
from sglang.autotune.executor.pool import PoolExecutor, build_slots
from sglang.autotune.measure import LoadPlan
from sglang.autotune.objective import (
    Direction,
    MetricThreshold,
    ScalarObjective,
    margin,
)
from sglang.autotune.orchestrator import tune
from sglang.autotune.report import BestConfigReporter, MarkdownReporter
from sglang.autotune.space.simple import SimpleSpace
from sglang.autotune.store import JsonlStore
from sglang.autotune.registry import STRATEGIES
from sglang.autotune.strategy import grid, random  # noqa: F401  register both
from sglang.autotune.task import HardwareSpec, ModelSpec, TuneTask
from sglang.autotune.executor.base import TrialSlot
from sglang.autotune.types import Budget, LoadPoint, Trial, Workload

logger = logging.getLogger("sglang.autotune")

DTYPE_CHOICES = ("auto", "fp8_w8a8", "int8_w8a8", "int8_w8a16", "int4_w4a16")
# Smoke defaults. The MoE space alone is ~1900 tile configs; start by proving
# the loop closes, then widen.
DEFAULT_BATCH_SIZES = (64, 1024)
DEFAULT_MAX_CONFIGS = 4


def _parse_knob(text: str) -> tuple:
    """``name=a,b,c`` -> (name, [a, b, c]) with ints and bools recovered."""
    name, _, values = text.partition("=")
    if not name or not values:
        raise argparse.ArgumentTypeError(f"expected name=v1,v2 but got {text!r}")
    return name.strip(), [_coerce(v.strip()) for v in values.split(",")]


def _coerce(text: str) -> Any:
    if text in ("true", "false"):
        return text == "true"
    if text == "none":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sglang.autotune")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("moe", "Tune triton fused MoE tile configs."),
        ("serve", "Tune deployment flags against a running server."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--model-path", required=True)
        sub.add_argument("--output-dir", type=Path, default=Path(f"./autotune-{name}"))
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--budget-hours", type=float, default=None)
        sub.add_argument(
            "--max-configs",
            type=int,
            default=DEFAULT_MAX_CONFIGS,
            help=(
                "Points to propose; ones a feasibility rule rejects are "
                "recorded but not measured, so fewer may be timed. "
                "0 searches the whole space."
            ),
        )
        sub.add_argument("--host", default="127.0.0.1")
        sub.add_argument("--port", type=int, default=31000)
        sub.add_argument(
            "--gpu-count",
            type=int,
            default=1,
            help="GPUs the run may use; with --gpus-per-trial it sets concurrency.",
        )
        sub.add_argument(
            "--gpus-per-trial",
            type=int,
            default=1,
            help="GPUs one candidate needs, e.g. its tp size.",
        )
        sub.add_argument(
            "--executor",
            choices=("auto", "serial", "pool"),
            default="auto",
            help=(
                "`auto` pools only when a candidate leaves room beside it. "
                "Force `pool` on one GPU to exercise the subprocess path."
            ),
        )
        sub.add_argument(
            "--strategy",
            choices=("random", "grid"),
            default="random",
            help="`grid` covers the space in order; `random` samples it.",
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the plan; touch no GPU.",
        )

    moe = subparsers.choices["moe"]
    moe.add_argument("--tp", "--tp-size", dest="tp_size", type=int, default=1)
    moe.add_argument("--ep-size", type=int, default=1)
    moe.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    moe.add_argument("--per-channel-quant", action="store_true")
    moe.add_argument("--disable-shared-experts-fusion", action="store_true")
    moe.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Token counts to tune; each gets its own winner.",
    )
    moe.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Timed replays per trial; each replays 10 kernel calls.",
    )

    serve = subparsers.choices["serve"]
    serve.add_argument(
        "--knob",
        type=_parse_knob,
        action="append",
        required=True,
        metavar="NAME=V1,V2",
        help="A ServerArgs field and its candidate values. Repeatable.",
    )
    serve.add_argument("--dataset-name", default="random")
    serve.add_argument("--num-prompts", type=int, default=200)
    serve.add_argument("--random-input-len", type=int, default=1024)
    serve.add_argument("--random-output-len", type=int, default=256)
    serve.add_argument("--max-concurrency", type=int, default=32)
    serve.add_argument("--server-timeout", type=float, default=600.0)
    serve.add_argument(
        "--max-p99-ttft-ms",
        type=float,
        default=None,
        help="Reject a candidate whose p99 TTFT exceeds this.",
    )
    serve.add_argument(
        "--preshard",
        action="store_true",
        help=(
            "Cache post-process weights so later trials skip the load. Writes "
            "under <model_path>/presharded/; the first launch pays to build it."
        ),
    )
    serve.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        help="Passed to launch_server for every candidate. Repeatable.",
    )
    return parser


def _shared(
    args: argparse.Namespace,
    *,
    name: str,
    space,
    strategy,
    driver,
    objective,
    workloads,
    load_plan,
    constraints=(),
    bucket_of=None,
    slot=None,
) -> TuneTask:
    kwargs: Dict[str, Any] = {} if bucket_of is None else {"bucket_of": bucket_of}
    hardware = HardwareSpec(
        gpu_count=args.gpu_count, gpus_per_trial=args.gpus_per_trial
    )
    executor = _executor(args, driver, hardware, workloads, slot)
    return TuneTask(
        name=name,
        model=ModelSpec(path=args.model_path),
        hardware=hardware,
        workloads=workloads,
        load_plan=load_plan,
        space=space,
        strategy=strategy,
        driver=driver,
        executor=executor,
        objective=objective,
        constraints=constraints,
        store=JsonlStore(args.output_dir / "trials.jsonl"),
        budget=Budget(wall_clock_hours=args.budget_hours),
        output_dir=args.output_dir,
        seed=args.seed,
        **kwargs,
    )


def _executor(args, driver, hardware, workloads, slot):
    """A pool only when a candidate leaves room for another beside it.

    Forcing `pool` at one slot buys no parallelism but does run the trial in a
    worker subprocess, which is the half of the pool a single-GPU host can
    still check: that the driver survives being pickled and set up over there.
    """
    if args.executor == "serial":
        return LocalExecutor(driver, slot=slot)
    if args.executor == "auto" and hardware.concurrent_trials <= 1:
        return LocalExecutor(driver, slot=slot)
    slots = build_slots(
        args.gpu_count,
        args.gpus_per_trial,
        host=args.host,
        base_port=args.port,
    )
    return PoolExecutor(driver, slots, workloads=workloads)


def _strategy(args: argparse.Namespace):
    """`grid` ignores --max-configs: covering the space is the whole point."""
    if args.strategy == "grid":
        return STRATEGIES.create("grid", seed=args.seed)
    return STRATEGIES.create(
        "random", seed=args.seed, max_points=args.max_configs or None
    )


def build_moe_task(args: argparse.Namespace) -> TuneTask:
    # Imported per subcommand: a broken kernel stack must not stop a serving
    # run, and vice versa. They share nothing but the loop.
    from sglang.autotune.driver.fused_moe_triton import (
        KERNEL_TIME_US,
        FusedMoeTritonDriver,
        MoeShape,
        moe_bucket,
        moe_workloads,
    )
    from sglang.autotune.space.fused_moe_triton import MoeTileSpace

    shape = MoeShape(
        args.model_path,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        dtype=args.dtype,
        per_channel_quant=args.per_channel_quant,
        disable_shared_experts_fusion=args.disable_shared_experts_fusion,
    )
    driver = FusedMoeTritonDriver(shape, num_iters=args.num_iters, seed=args.seed)
    return _shared(
        args,
        name=f"fused-moe-{Path(args.model_path).name}-tp{args.tp_size}",
        space=MoeTileSpace(block_shape=shape.block_shape),
        strategy=_strategy(args),
        driver=driver,
        objective=ScalarObjective(metric=KERNEL_TIME_US, direction=Direction.MINIMIZE),
        workloads=moe_workloads(args.batch_size),
        load_plan=LoadPlan(),
        bucket_of=moe_bucket,
    )


def build_serve_task(args: argparse.Namespace) -> TuneTask:
    from sglang.autotune.driver.serving import STEADY_OUTPUT_THROUGHPUT, ServingDriver

    driver = ServingDriver(
        args.model_path,
        server_timeout_s=args.server_timeout,
        extra_server_args=args.extra_server_arg,
        log_dir=args.output_dir / "logs",
        preshard=args.preshard,
        preshard_dir=args.output_dir / "presharded",
    )
    workload = Workload(
        name=args.dataset_name,
        kind=args.dataset_name,
        params={
            "num_prompts": args.num_prompts,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
        },
    )
    constraints = (
        [MetricThreshold(metric="p99_ttft_ms", max_value=args.max_p99_ttft_ms)]
        if args.max_p99_ttft_ms is not None
        else []
    )
    return _shared(
        args,
        name=f"serve-{Path(args.model_path).name}",
        space=SimpleSpace(dict(args.knob), group="server_args"),
        strategy=_strategy(args),
        driver=driver,
        objective=ScalarObjective(
            metric=STEADY_OUTPUT_THROUGHPUT, direction=Direction.MAXIMIZE
        ),
        constraints=constraints,
        workloads=[workload],
        load_plan=LoadPlan(fixed=(LoadPoint(max_concurrency=args.max_concurrency),)),
        slot=TrialSlot(index=0, host=args.host, port=args.port),
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "moe":
        from sglang.autotune.report_moe import MoeConfigReporter

        task = build_moe_task(args)
        reporters = [MarkdownReporter(), MoeConfigReporter()]
        print(task.driver.shape.describe())
        print(f"config file: {task.driver.shape.config_filename}")
    else:
        task = build_serve_task(args)
        reporters = [MarkdownReporter(), BestConfigReporter()]
        print(f"knobs: {dict(args.knob)}")

    print(f"space: {task.space.cardinality} points")
    print(f"trials: {task.estimated_trials()}")
    print(f"output: {args.output_dir}")
    if args.dry_run:
        task.strategy.setup(
            space=task.space,
            objective=task.objective,
            constraints=task.constraints,
            budget=task.budget,
        )
        print(f"search: {task.strategy.describe_plan()}")
        points = _preview(task)
        for point in points:
            print(f"  {point}")
        _print_first_trial(task, points)
        return 0

    if not torch.cuda.is_available():
        print(
            "error: tuning needs a visible GPU (--dry-run works without one)",
            file=sys.stderr,
        )
        return 1

    errors = task.validate()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    result = tune(task, reporters=reporters)
    if not result.best_by_bucket:
        counts = task.store.count_by_status() if task.store else {}
        tally = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        print(f"no valid config from {len(result.measurements)} trials: {tally}")
        print(f"see {args.output_dir}/summary.md and logs/ for why")
        return 1

    metric = task.objective.metric_names[0]
    for bucket in result.buckets:
        best = result.best_by_bucket.get(bucket)
        label = bucket or "best"
        if best is None:
            print(f"{label}: no valid config")
            continue
        print(f"{label}: {best.measurement.metric(metric):.2f}  {best.point}")
    return 0 if result.best_by_bucket else 1


def _print_first_trial(task: TuneTask, points: Sequence) -> None:
    """Show what one point turns into, as the driver would run it."""
    if not points:
        return
    trial = Trial(
        point=points[0],
        workload=task.workloads[0],
        load=task.load_plan.fixed[0] if task.load_plan.fixed else LoadPoint(),
        bucket=task.bucket_of(task.workloads[0], LoadPoint()),
    )
    lines = task.driver.describe_trial(trial, task.executor.slot)
    if not lines:
        return
    print("first trial:")
    for line in lines:
        print(f"  {line}")


def _preview(task: TuneTask, limit: int = 5) -> Sequence:
    """The points the strategy would actually propose first.

    Enumerating the space instead would show declaration order, which only a
    grid search follows; for a sampling strategy that is a different set of
    points than the run will try. Safe to consume the strategy here because a
    dry run stops right after.
    """
    return task.strategy.ask(limit)


if __name__ == "__main__":
    raise SystemExit(main())
