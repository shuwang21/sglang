"""Reporters: turn a finished (or interrupted) run into artifacts.

Reporters are pure consumers of :class:`~sglang.autotune.task.TuneResult`, so
they can be re-run offline against a stored run — ``autotune report --dir ...``
regenerates everything without re-measuring anything.

The artifact that matters most is the emitted config: a tuning run whose output
has to be retyped by a human into a launch command has lost most of its value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from sglang.autotune.task import TuneResult

__all__ = ["Reporter"]


class Reporter(ABC):
    """Emits one artifact family from a run."""

    name: str = "reporter"

    @abstractmethod
    def emit(self, result: TuneResult, output_dir: Path) -> List[Path]:
        """Write artifacts; return the paths written, for logging."""

    def emit_partial(self, result: TuneResult, output_dir: Path) -> List[Path]:
        """Emit for an interrupted run.

        Defaults to the same output. An interrupted run still has a best
        candidate and it should still be usable — a run that produces nothing
        because it was stopped at hour 11 of 12 is a bad trade.
        """
        return self.emit(result, output_dir)
