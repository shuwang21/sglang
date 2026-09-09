"""Durable trial log: resume, dedup, and cross-run reuse.

A tuning run is long enough that it *will* be interrupted — a preempted node, a
Ctrl-C, an OOM that takes the box down. The store is what makes that a pause
rather than a loss, and it is append-on-completion for exactly that reason: a
run killed mid-trial loses one trial, not the run.

Reuse across runs is gated on the provenance fingerprint (baked into
``Trial.key``). Reusing a measurement taken before a kernel change is the
easiest way to produce a confidently wrong recommendation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sglang.autotune.objective import Constraint, Evaluation, Objective, rank
from sglang.autotune.registry import register_store
from sglang.autotune.types import Measurement

logger = logging.getLogger(__name__)

__all__ = ["TrialStore", "JsonlStore"]


class TrialStore(ABC):
    """Append-only record of every trial, including failures.

    Failures are kept deliberately. "This candidate OOMs at tp=4" is a result,
    and dropping it means the next run rediscovers it at full price.
    """

    name: str = "store"

    # ---- required --------------------------------------------------------

    @abstractmethod
    def record(self, measurement: Measurement) -> None:
        """Durably append one measurement. Must be crash-safe (flush/fsync)."""

    @abstractmethod
    def history(self) -> List[Measurement]:
        """Every measurement recorded so far, in insertion order."""

    # ---- optional --------------------------------------------------------

    def lookup(self, key: str) -> Optional[Measurement]:
        """Existing measurement for a trial key, if any."""
        return next((m for m in self.history() if m.key == key), None)

    def __contains__(self, key: str) -> bool:
        return self.lookup(key) is not None

    def best(
        self,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
    ) -> Optional[Evaluation]:
        ranked = rank(self.history(), objective, constraints)
        top = ranked[0] if ranked else None
        return top if top is not None and top.rankable else None

    def count_by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for measurement in self.history():
            counts[measurement.status.value] = (
                counts.get(measurement.status.value, 0) + 1
            )
        return counts

    def close(self) -> None:
        """Flush and release handles."""

    def __enter__(self) -> "TrialStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@register_store("jsonl")
class JsonlStore(TrialStore):
    """One JSON object per line, loaded into memory on open.

    Written and flushed per trial: the run dies between trials often enough
    that buffering would cost exactly the results worth keeping.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._by_key: Dict[str, Measurement] = {}
        self._order: List[str] = []
        self._handle = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for lineno, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                self._remember(Measurement.from_json(json.loads(line)))
            except (ValueError, KeyError):
                # A kill mid-write leaves a torn final line. Dropping it costs
                # one trial; refusing to load costs the whole resume.
                logger.warning("skipping unreadable trial at %s:%d", self.path, lineno)

    def _remember(self, measurement: Measurement) -> None:
        if measurement.key not in self._by_key:
            self._order.append(measurement.key)
        self._by_key[measurement.key] = measurement

    def record(self, measurement: Measurement) -> None:
        self._remember(measurement)
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(measurement.to_json()) + "\n")
        self._handle.flush()

    def history(self) -> List[Measurement]:
        return [self._by_key[key] for key in self._order]

    def lookup(self, key: str) -> Optional[Measurement]:
        return self._by_key.get(key)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
