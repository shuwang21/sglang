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

from sglang.autotune.driver.fused_moe_triton import (
    KERNEL_TIME_US,
    FusedMoeTritonDriver,
    MoeShape,
    moe_bucket,
    moe_workloads,
)
from sglang.autotune.driver.serving import STEADY_OUTPUT_THROUGHPUT, ServingDriver
from sglang.autotune.executor.local import LocalExecutor
from sglang.autotune.measure import LoadPlan
from sglang.autotune.objective import (
    Direction,
    MetricThreshold,
    ScalarObjective,
)
from sglang.autotune.orchestrator import tune
from sglang.autotune.report import BestConfigReporter, MarkdownReporter
from sglang.autotune.report_moe import MoeConfigReporter
from sglang.autotune.space.fused_moe_triton import MoeTileSpace
from sglang.autotune.space.simple import SimpleSpace
from sglang.autotune.store import JsonlStore
from sglang.autotune.strategy.random import RandomStrategy
from sglang.autotune.task import HardwareSpec, ModelSpec, TuneTask
from sglang.autotune.types import Budget, LoadPoint, Workload

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
            help="Points to sample. 0 searches the whole space.",
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
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=31000)
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
) -> TuneTask:
    kwargs: Dict[str, Any] = {} if bucket_of is None else {"bucket_of": bucket_of}
    return TuneTask(
        name=name,
        model=ModelSpec(path=args.model_path),
        hardware=HardwareSpec(gpu_count=1, gpus_per_trial=1),
        workloads=workloads,
        load_plan=load_plan,
        space=space,
        strategy=strategy,
        driver=driver,
        executor=LocalExecutor(driver),
        objective=objective,
        constraints=constraints,
        store=JsonlStore(args.output_dir / "trials.jsonl"),
        budget=Budget(wall_clock_hours=args.budget_hours),
        output_dir=args.output_dir,
        seed=args.seed,
        **kwargs,
    )


def build_moe_task(args: argparse.Namespace) -> TuneTask:
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
        strategy=RandomStrategy(seed=args.seed, max_points=args.max_configs or None),
        driver=driver,
        objective=ScalarObjective(metric=KERNEL_TIME_US, direction=Direction.MINIMIZE),
        workloads=moe_workloads(args.batch_size),
        load_plan=LoadPlan(),
        bucket_of=moe_bucket,
    )


def build_serve_task(args: argparse.Namespace) -> TuneTask:
    driver = ServingDriver(
        args.model_path,
        server_timeout_s=args.server_timeout,
        extra_server_args=args.extra_server_arg,
        log_dir=args.output_dir / "logs",
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
        strategy=RandomStrategy(seed=args.seed, max_points=args.max_configs or None),
        driver=driver,
        objective=ScalarObjective(
            metric=STEADY_OUTPUT_THROUGHPUT, direction=Direction.MAXIMIZE
        ),
        constraints=constraints,
        workloads=[workload],
        load_plan=LoadPlan(fixed=(LoadPoint(max_concurrency=args.max_concurrency),)),
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "moe":
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
        for point in _preview(task):
            print(f"  {point}")
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
    metric = task.objective.metric_names[0]
    for bucket in result.buckets:
        best = result.best_by_bucket.get(bucket)
        label = bucket or "best"
        if best is None:
            print(f"{label}: no valid config")
            continue
        print(f"{label}: {best.measurement.metric(metric):.2f}  {best.point}")
    return 0 if result.best_by_bucket else 1


def _preview(task: TuneTask, limit: int = 5) -> Sequence:
    """First few feasible points, so a plan shows what will actually be tried."""
    out = []
    for point in task.space.grid():
        out.append(point)
        if len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    raise SystemExit(main())
