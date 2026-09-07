"""Rendering a point into launch_server flags."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.autotune.driver.serving import _classify, render_server_flags
from sglang.autotune.types import FailureKind, Point
from sglang.test.test_utils import CustomTestCase

# What the launcher actually raises: a plain Exception naming only the exit
# code once the process is gone, TimeoutError while it is up but unhealthy.
_EXITED = Exception("Server process exited with code 1. Check server logs for errors.")


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


class TestClassifyFailure(CustomTestCase):
    """A crashed server was reported as a health timeout.

    The launcher's exception names only the exit code, so classifying on it
    alone made every launch failure look like the server was up and unhealthy
    -- the one diagnosis that rules out reading the log, which is where the
    cause actually is.
    """

    def test_reads_the_cause_out_of_the_log(self):
        gated = (
            "OSError: You are trying to access a gated repo.\n"
            "401 Client Error. Cannot access gated repo for url ..."
        )
        self.assertEqual(_classify(_EXITED, gated), FailureKind.MODEL_LOAD)
        self.assertEqual(
            _classify(_EXITED, "torch.OutOfMemoryError: CUDA out of memory"),
            FailureKind.OOM,
        )
        self.assertEqual(
            _classify(_EXITED, "error: unrecognized arguments: --nope"),
            FailureKind.UNSUPPORTED_FLAG,
        )

    def test_an_exit_is_a_crash_and_only_a_timeout_is_a_timeout(self):
        self.assertEqual(
            _classify(_EXITED, "nothing familiar"), FailureKind.SERVER_CRASH
        )
        self.assertEqual(
            _classify(TimeoutError("Timeout waiting for server"), ""),
            FailureKind.HEALTH_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
