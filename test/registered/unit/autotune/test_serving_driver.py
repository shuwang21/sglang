"""Rendering a point into launch_server flags."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.autotune.driver.serving import render_server_flags
from sglang.autotune.types import Point
from sglang.test.test_utils import CustomTestCase


class TestRenderServerFlags(CustomTestCase):
    def test_booleans_are_presence_not_value(self):
        """`--flag False` turns a store_true flag on, which is the opposite.

        A point carries every knob it was assigned, including the ones set to
        False, so a renderer that stringifies uniformly enables exactly the
        options the search meant to disable.
        """
        flags = render_server_flags(
            Point({"enable_torch_compile": True, "disable_radix_cache": False})
        )
        self.assertEqual(flags, ["--enable-torch-compile"])

    def test_none_is_omitted_so_the_server_keeps_its_default(self):
        self.assertEqual(render_server_flags(Point({"chunked_prefill_size": None})), [])

    def test_names_are_server_args_destinations(self):
        flags = render_server_flags(Point({"tp_size": 2, "mem_fraction_static": 0.85}))
        self.assertEqual(flags, ["--mem-fraction-static", "0.85", "--tp-size", "2"])


if __name__ == "__main__":
    unittest.main()
