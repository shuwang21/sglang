# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Tuple

import ray
import torch
import triton
from common_utils import (
    BenchmarkConfig,
    benchmark_config,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from ray.experimental.tqdm_ray import tqdm

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device, is_hip, is_xpu

_is_hip = is_hip()
_is_xpu = is_xpu()


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(
        self, seed: int, server_args: ServerArgs, is_dsv4: bool = False
    ) -> None:
        torch.set_default_device(get_device())
        torch.get_device_module().manual_seed_all(0)
        self.seed = seed
        self.is_dsv4 = is_dsv4
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. Ray isolates each worker to a single visible
        # GPU via CUDA_VISIBLE_DEVICES, so the local ordinal is always 0. On
        # ROCm using the global ray gpu id here raises "invalid device ordinal".
        self.device_id = 0 if is_hip() else int(ray.get_gpu_ids()[0])
        set_global_server_args_for_scheduler(server_args)

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        N = shard_intermediate_size // 2
        if use_int4_w4a16:
            N = N // 2
        op_config = get_moe_configs(
            num_experts,
            N,
            dtype_str,
            block_n,
            block_k,
            per_channel_quant,
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                block_shape,
                self.is_dsv4,
            )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        with (
            torch.get_device_module().device(self.device_id)
            if _is_xpu or _is_hip
            else nullcontext()
        ):
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        self.is_dsv4,
                        num_iters=10,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def main(args: argparse.Namespace):
    server_args = ServerArgs(
        model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size
    )

    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    per_channel_quant = args.per_channel_quant

    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    elif args.batch_size is not None:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = get_default_batch_sizes()

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    is_dsv4 = model_config["architecture"] == "DeepseekV4ForCausalLM"
    workers = [
        BenchmarkWorker.remote(args.seed, server_args, is_dsv4) for _ in range(num_gpus)
    ]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        if args.search_space_file:
            with open(args.search_space_file) as f:
                search_space = json.load(f)
            if not isinstance(search_space, list) or not all(
                isinstance(config, dict) for config in search_space
            ):
                raise ValueError(
                    "--search-space-file must contain a JSON list of configs"
                )
        else:
            search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_k = block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]

        filename = get_config_filename(
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
        )
        print(
            f"Start tuning over {len(search_space)} configurations to create {filename}..."
        )

        start = time.perf_counter()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                    search_space,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            filename,
        )
        end = time.perf_counter()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"],
        default="auto",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Tune or benchmark an explicit set of token counts in parallel.",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--search-space-file",
        type=str,
        help="JSON file containing an explicit list of Triton configs to evaluate with --tune.",
    )
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    main(args)
