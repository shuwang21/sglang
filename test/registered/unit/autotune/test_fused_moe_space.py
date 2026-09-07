"""Feasibility rules for the fused MoE tile space."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.autotune.space.fused_moe_triton import BlockKDivisible, SharedMemoryFits
from sglang.autotune.types import Point
from sglang.test.test_utils import CustomTestCase

# NVIDIA L4, bf16: the limit triton reported, and the tiles it accepted or
# rejected with it. Each `required` is the exact figure from an OutOfResources
# message, so a rewrite of the formula that no longer reproduces them is wrong
# about real hardware, not just about this test.
L4_SHARED_MEMORY_BYTES = 101376
L4_OBSERVATIONS = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, required_bytes, fits)
    (128, 256, 64, 5, 196608, False),
    (64, 256, 128, 4, 245760, False),
    (64, 256, 128, 5, 327680, False),
    (256, 128, 64, 5, 196608, False),
    (32, 32, 256, 4, 98304, True),  # 97 KiB of a 99 KiB limit
    (16, 256, 128, 2, 69632, True),
    (16, 32, 64, 2, 6144, True),
    (16, 32, 256, 2, 24576, True),
]


def _tile(block_m: int, block_n: int, block_k: int, num_stages: int) -> Point:
    return Point(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": num_stages,
        }
    )


class TestSharedMemoryFits(CustomTestCase):
    def setUp(self):
        self.rule = SharedMemoryFits(limit_bytes=L4_SHARED_MEMORY_BYTES, itemsize=2)

    def test_matches_the_bytes_triton_reported(self):
        for block_m, block_n, block_k, stages, required, _ in L4_OBSERVATIONS:
            with self.subTest(m=block_m, n=block_n, k=block_k, stages=stages):
                point = _tile(block_m, block_n, block_k, stages)
                self.assertEqual(self.rule.required_bytes(point), required)

    def test_accepts_exactly_the_tiles_that_ran(self):
        for block_m, block_n, block_k, stages, _, fits in L4_OBSERVATIONS:
            with self.subTest(m=block_m, n=block_n, k=block_k, stages=stages):
                verdict = self.rule.check(_tile(block_m, block_n, block_k, stages), {})
                self.assertEqual(verdict is None, fits)

    def test_wider_dtype_needs_more(self):
        point = _tile(32, 32, 256, 4)
        fp32 = SharedMemoryFits(limit_bytes=L4_SHARED_MEMORY_BYTES, itemsize=4)
        self.assertIsNone(self.rule.check(point, {}))
        self.assertIsNotNone(fp32.check(point, {}))


class TestBlockKDivisible(CustomTestCase):
    def test_rejects_a_tile_that_straddles_two_quant_blocks(self):
        rule = BlockKDivisible(block_k=128)
        self.assertIsNone(rule.check(_tile(64, 64, 64, 3), {}))
        self.assertIsNone(rule.check(_tile(64, 64, 128, 3), {}))
        self.assertIsNotNone(rule.check(_tile(64, 64, 256, 3), {}))


if __name__ == "__main__":
    unittest.main()
