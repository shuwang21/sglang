from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_tuning
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load_common_utils():
    return fused_moe_triton_tuning


def test_get_model_config_supports_kimi_vl():
    common_utils = _load_common_utils()
    text_config = SimpleNamespace(
        hidden_size=2048,
        n_routed_experts=64,
        num_experts_per_tok=6,
        moe_intermediate_size=1408,
        torch_dtype=torch.bfloat16,
    )
    model_config = SimpleNamespace(
        architectures=["KimiVLForConditionalGeneration"],
        text_config=text_config,
        get_text_config=lambda: text_config,
    )

    with patch.object(common_utils, "get_config", return_value=model_config):
        tuned_config = common_utils.get_model_config(
            "moonshotai/Kimi-VL-A3B-Instruct", tp_size=4, ep_size=4
        )

    assert tuned_config == {
        "num_experts": 16,
        "topk": 6,
        "hidden_size": 2048,
        "shard_intermediate_size": 2816,
        "dtype": torch.bfloat16,
        "block_shape": None,
        "architecture": "KimiVLForConditionalGeneration",
    }


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
