import json
from types import SimpleNamespace
from typing import Dict, List, TypedDict

import torch

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_moe,
    get_config_dtype_str,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_file_name,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.hf_transformers_utils import get_config

_is_hip = is_hip()


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def calculate_shard_intermediate_size(
    intermediate_size: int, tp_size: int, ep_size: int = 1
) -> int:
    assert tp_size % ep_size == 0
    moe_tp_size = tp_size // ep_size
    assert intermediate_size % moe_tp_size == 0
    return 2 * intermediate_size // moe_tp_size


def get_model_config(
    model_name: str,
    tp_size: int,
    ep_size: int = 1,
    disable_shared_experts_fusion: bool = False,
    topk_ids_dir: str = None,
) -> Dict:
    config = get_config(model_name, trust_remote_code=True)
    architecture = config.architectures[0]
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    if (
        hasattr(config, "quantization_config")
        and "config_groups" in config.quantization_config
    ):
        config_groups = config.quantization_config["config_groups"]
        # Get group_size from the first group's weights config
        first_group = next(iter(config_groups.values()), {})
        weights_config = first_group.get("weights", {})
        group_size = weights_config.get("group_size")
        block_shape = [0, group_size]
        assert len(block_shape) == 2
    # Replace config with text_config for encoder-decoder models after getting block_shape and architecture
    if hasattr(config, "text_config"):
        text_config = config.get_text_config()
        # Some models (e.g. MiniMax-M3) carry text_config as a plain dict; wrap
        # it so downstream attribute access works uniformly.
        if isinstance(text_config, dict):
            config = SimpleNamespace(**text_config)
        else:
            config = text_config

    hidden_size = config.hidden_size
    if architecture == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts // ep_size
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
    elif architecture == "JambaForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "MellumForCausalLM",
    ]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm4MoeForCausalLM",
        "GlmMoeDsaForCausalLM",
        "KimiVLForConditionalGeneration",
        "MistralLarge3ForCausalLM",
    ]:
        E = (config.n_routed_experts // ep_size) + (
            0
            if disable_shared_experts_fusion
            or architecture
            not in [
                "DeepseekV3ForCausalLM",
                "DeepseekV32ForCausalLM",
                "Glm4MoeForCausalLM",
                "GlmMoeDsaForCausalLM",
                "MistralLarge3ForCausalLM",
            ]
            else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.moe_intermediate_size
    elif architecture == "Llama4ForConditionalGeneration":
        E = config.num_local_experts // ep_size + (
            0 if disable_shared_experts_fusion else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "BailingMoEForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
    ]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "HYV3ForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
    elif architecture == "NemotronHForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = getattr(config, "moe_latent_size", None) or hidden_size
    elif architecture == "Gemma4ForConditionalGeneration":
        E = config.num_experts // ep_size
        topk = config.top_k_experts
        intermediate_size = config.moe_intermediate_size
    elif architecture == "Lfm2MoeForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "MiniMaxM3SparseForConditionalGeneration":
        # Serving fuses the shared expert into the routed-expert tensor by
        # default (E = num_local_experts + 1), so tune for that shape unless
        # fusion is explicitly disabled.
        E = config.num_local_experts // ep_size + (
            0 if disable_shared_experts_fusion else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.intermediate_size
    elif architecture == "UnlimitedOCRForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    else:
        # Default: Mixtral
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size

    shard_intermediate_size = calculate_shard_intermediate_size(
        intermediate_size, tp_size, ep_size
    )

    # gfx942 (MI300X) remaps MXFP8 [1,32]->[128,128] at load; tune must match.
    # sm100/gfx95 run native [1,32] and must not remap (mirrors fp8.py).
    if (
        architecture == "MiniMaxM3SparseForConditionalGeneration"
        and is_hip()
        and not is_gfx95_supported()
        and block_shape == [1, 32]
    ):
        block_shape = [128, 128]

    # text_config may not carry torch_dtype; fall back to bf16.
    torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

    return {
        "num_experts": E,
        "topk": topk,
        "hidden_size": hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": torch_dtype,
        "block_shape": block_shape,
        "architecture": architecture,
    }


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    waves_per_eu_range = 0
    for num_stages in [2]:
        for block_m in [32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [1, 2, 4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu_range,
                                }
                            )
    return configs


def get_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    if is_hip():
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [2, 3, 4, 5]:
            for block_m in [16, 32, 64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                )
    return configs


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
        **({"USE_TMA": config["USE_TMA"]} if "USE_TMA" in config else {}),
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    filename: str,
) -> None:
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def get_config_filename(
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
) -> str:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    N = shard_intermediate_size // 2
    if use_int4_w4a16:
        N = N // 2

    filename = get_config_file_name(
        num_experts,
        N,
        dtype_str,
        block_shape,
        per_channel_quant,
    )

    return filename


def get_default_batch_sizes() -> List[int]:
    return [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]


def benchmark_config(
    config: BenchmarkConfig,
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
    block_shape: List[int] = None,
    is_dsv4: bool = False,
    num_iters: int = 100,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    elif use_int4_w4a16:
        w1 = torch.randint(
            0,
            255,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size // 2,
            ),
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            0,
            255,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 4,
            ),
            dtype=torch.uint8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_int4_w4a16:
        block_n = 1 if (block_shape[0] == 0) else block_shape[0]
        block_k = block_shape[1]
        n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
        n_tiles_w2 = (hidden_size + block_n - 1) // block_n
        k_tiles_w1 = (hidden_size + block_k - 1) // block_k
        k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
        w1_scale = torch.randn(
            (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.bfloat16
        )
        w2_scale = torch.randn(
            (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.bfloat16
        )
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
            swiglu_limit=10.0 if is_dsv4 else None,
        )

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # Flush L2 cache with 256 MB data
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        prepare(i)
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies: List[float] = []
    for i in range(num_iters):
        latencies.append(start_events[i].elapsed_time(end_events[i]))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg
