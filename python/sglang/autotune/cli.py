"""``python -m sglang.autotune`` -- tune the triton fused MoE kernel for a model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

from sglang.autotune.driver.fused_moe_triton import (
    KERNEL_TIME_US,
    FusedMoeTritonDriver,
    MoeShape,
    moe_bucket,
    moe_workloads,
)
from sglang.autotune.executor.local import LocalExecutor
from sglang.autotune.measure import LoadPlan
from sglang.autotune.objective import Direction, ScalarObjective
from sglang.autotune.orchestrator import tune
from sglang.autotune.report import MarkdownReporter
from sglang.autotune.report_moe import MoeConfigReporter
from sglang.autotune.space.fused_moe_triton import MoeTileSpace
from sglang.autotune.store import JsonlStore
from sglang.autotune.strategy.random import RandomStrategy
from sglang.autotune.task import HardwareSpec, ModelSpec, TuneTask
from sglang.autotune.types import Budget

logger = logging.getLogger("sglang.autotune")

DTYPE_CHOICES = ("auto", "fp8_w8a8", "int8_w8a8", "int8_w8a16", "int4_w4a16")
# Smoke default. The full space is ~1900 tile configs x 18 batch sizes, which
# is a multi-hour run on one GPU; start by proving the loop closes.
DEFAULT_BATCH_SIZES = (64, 1024)
DEFAULT_MAX_CONFIGS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sglang.autotune")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp", "--tp-size", dest="tp_size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    parser.add_argument("--per-channel-quant", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Token counts to tune; each gets its own winner.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=DEFAULT_MAX_CONFIGS,
        help="Tile configs to sample per batch size. 0 searches the whole space.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Timed replays per trial; each replays 10 kernel calls.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./autotune-moe"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved shape and the plan, touch no GPU.",
    )
    return parser


def build_task(args: argparse.Namespace) -> TuneTask:
    shape = MoeShape(
        args.model_path,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        dtype=args.dtype,
        per_channel_quant=args.per_channel_quant,
        disable_shared_experts_fusion=args.disable_shared_experts_fusion,
    )
    driver = FusedMoeTritonDriver(shape, num_iters=args.num_iters, seed=args.seed)
    return TuneTask(
        name=f"fused-moe-{Path(args.model_path).name}-tp{args.tp_size}",
        model=ModelSpec(path=args.model_path),
        hardware=HardwareSpec(gpu_count=1, gpus_per_trial=1),
        workloads=moe_workloads(args.batch_size),
        load_plan=LoadPlan(),
        space=MoeTileSpace(block_shape=shape.block_shape),
        strategy=RandomStrategy(seed=args.seed, max_points=args.max_configs or None),
        driver=driver,
        executor=LocalExecutor(driver),
        objective=ScalarObjective(metric=KERNEL_TIME_US, direction=Direction.MINIMIZE),
        store=JsonlStore(args.output_dir / "trials.jsonl"),
        budget=Budget(wall_clock_hours=args.budget_hours),
        output_dir=args.output_dir,
        seed=args.seed,
        bucket_of=moe_bucket,
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    task = build_task(args)
    shape = task.driver.shape

    print(shape.describe())
    print(f"batch sizes: {args.batch_size}")
    print(f"space: {task.space.cardinality} tile configs")
    print(f"trials: {task.estimated_trials()}")
    print(f"config file: {shape.config_filename}")
    print(f"output: {args.output_dir}")
    if args.dry_run:
        # Binding the space is what lets the strategy describe its plan; the
        # orchestrator does it on a real run, and does it exactly once.
        task.strategy.setup(
            space=task.space,
            objective=task.objective,
            constraints=task.constraints,
            budget=task.budget,
        )
        print(f"search: {task.strategy.describe_plan()}")
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

    result = tune(task, reporters=[MarkdownReporter(), MoeConfigReporter()])
    for bucket in result.buckets:
        best = result.best_by_bucket.get(bucket)
        if best is None:
            print(f"{bucket}: no valid config")
            continue
        print(
            f"{bucket}: {best.measurement.metric(KERNEL_TIME_US):.2f} us  {best.point}"
        )
    return 0 if result.best_by_bucket else 1


if __name__ == "__main__":
    raise SystemExit(main())
