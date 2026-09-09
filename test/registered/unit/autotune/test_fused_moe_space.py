"""Feasibility rules for the fused MoE tile space, and conditional knobs."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.autotune.space.base import Categorical, Conditional, Knob
from sglang.autotune.space.fused_moe_triton import BlockKDivisible, SharedMemoryFits
from sglang.autotune.space.simple import SimpleSpace
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


class TestConditionalKnobs(CustomTestCase):
    """Knobs that exist only under another knob's value.

    A flat space would spend trials on assignments that collapse to the same
    deployment: with speculative_algorithm unset, every value of
    speculative_num_steps launches an identical server.
    """

    def _space(self) -> SimpleSpace:
        return SimpleSpace(
            {"speculative_algorithm": [None, "EAGLE"]},
            conditionals=[
                Conditional(
                    when={"speculative_algorithm": "EAGLE"},
                    knobs=[
                        Knob(name="speculative_num_steps", domain=Categorical((3, 5)))
                    ],
                )
            ],
        )

    def test_the_dependent_knob_is_absent_when_the_predicate_fails(self):
        space = self._space()
        points = {
            p.get("speculative_algorithm"): p
            for p in space.grid()
            if p.get("speculative_num_steps") is None
        }
        self.assertIn(None, points)
        self.assertNotIn("speculative_num_steps", points[None].values)

    def test_the_grid_expands_the_dependent_knob_only_where_it_applies(self):
        combinations = [
            (p.get("speculative_algorithm"), p.get("speculative_num_steps"))
            for p in self._space().grid()
        ]
        self.assertCountEqual(combinations, [(None, None), ("EAGLE", 3), ("EAGLE", 5)])

    def test_the_plan_does_not_understate_a_conditional_space(self):
        """cardinality feeds estimated_trials; understating it misleads a plan.

        Two base values and one two-valued dependent knob is three points; the
        bound may exceed that but must not fall short of it.
        """
        self.assertGreaterEqual(self._space().cardinality, 3)

    def test_validate_accepts_a_value_only_the_conditional_declares(self):
        """active_knobs, not knobs: domain checking must see the extra knob."""
        space = self._space()
        point = Point({"speculative_algorithm": "EAGLE", "speculative_num_steps": 3})
        self.assertEqual(space.validate(point), [])
        bad = Point({"speculative_algorithm": "EAGLE", "speculative_num_steps": 99})
        self.assertEqual(len(space.validate(bad)), 1)


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
