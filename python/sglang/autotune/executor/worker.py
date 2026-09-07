"""One pool slot's worker process: `python -m sglang.autotune.executor.worker`.

A separate process per slot, not a thread and not a fork. Three reasons, each
on its own sufficient: ``CUDA_VISIBLE_DEVICES`` has to be set before the
process touches CUDA, so the pinning must happen at spawn; the serving
benchmark client keeps its arguments in a module global; and a trial that
takes its process down with it must not take the run with it.

The protocol is length-prefixed pickle over stdin/stdout, which keeps stderr
free for the child's own logs and the server output it tees.
"""

from __future__ import annotations

import pickle
import struct
import sys
import time
import traceback
from typing import Any, Optional

from sglang.autotune.types import FailureKind, Measurement, TrialStatus

__all__ = ["read_message", "write_message", "main"]

_HEADER = struct.Struct("!I")


def read_message(stream: Any) -> Optional[Any]:
    """Next object, or None at a clean end of stream."""
    header = stream.read(_HEADER.size)
    if not header or len(header) < _HEADER.size:
        return None
    (size,) = _HEADER.unpack(header)
    payload = stream.read(size)
    if len(payload) < size:
        return None
    return pickle.loads(payload)


def write_message(stream: Any, obj: Any) -> None:
    payload = pickle.dumps(obj)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> int:
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    setup = read_message(stdin)
    if setup is None:
        return 0
    driver, slot = setup["driver"], setup["slot"]
    # Per process, not per run: the kernel driver publishes ServerArgs and sets
    # the default device here, and both are process-global.
    driver.prepare(setup["workloads"])
    write_message(stdout, {"ready": True})

    while True:
        submission = read_message(stdin)
        if submission is None:
            return 0
        trial = submission["trial"]
        started = time.time()
        try:
            measurement = driver.measure(
                trial, slot, timeout_s=submission["timeout_s"], prune_check=None
            )
        except BaseException:  # noqa: BLE001 - the parent cannot see this stack
            measurement = Measurement(
                trial=trial,
                status=TrialStatus.FAILED,
                failure=FailureKind.UNKNOWN,
                message=traceback.format_exc(limit=8),
                started_at=started,
                duration_s=time.time() - started,
            )
        write_message(stdout, {"measurement": measurement})


if __name__ == "__main__":
    raise SystemExit(main())
