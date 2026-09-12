"""Reporters: turn a finished (or interrupted) run into artifacts.

Reporters are pure consumers of :class:`~sglang.autotune.task.TuneResult`, so
they can be re-run offline against a stored run — ``autotune report --dir ...``
regenerates everything without re-measuring anything.

The artifact that matters most is the emitted config: a tuning run whose output
has to be retyped by a human into a launch command has lost most of its value.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from sglang.autotune.objective import Evaluation, margin
from sglang.autotune.registry import register_reporter
from sglang.autotune.task import TuneResult
from sglang.autotune.types import TrialStatus

__all__ = ["Reporter", "MarkdownReporter"]


def _cell(value: float) -> str:
    # A failed trial reports no metrics; Measurement.metric defaults to nan.
    return "-" if math.isnan(value) else f"{value:g}"


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


def _margin_line(ranked: List[Evaluation]) -> List[str]:
    """State the lead, and whether the run could tell it from noise."""
    lead = margin(ranked)
    if lead is None:
        return []
    line = f"Ahead of the runner-up by {lead * 100:.2f}%."
    spread = max((e.spread for e in ranked if e.spread is not None), default=None)
    if spread is None:
        return [line + " Each point was measured once."]
    verdict = "above" if lead > spread else "within"
    return [
        line,
        f"Repeated measurements of one point spread by up to "
        f"{spread * 100:.2f}%, so the lead is {verdict} this run's own noise.",
    ]


@register_reporter("best_config")
class BestConfigReporter(Reporter):
    """`best.yaml` for `launch_server --config`, and `best.sh` beside it.

    A result a human has to retype into a launch command has lost most of its
    value, so the deliverable is the config file the runtime already loads.
    Only a single-bucket run has one winner to emit; a bucketed run's
    deliverable is the table its own reporter writes.
    """

    def emit(self, result: TuneResult, output_dir: Path) -> List[Path]:
        import yaml

        if result.best is None:
            return []
        driver = result.task.driver
        config = driver.render_config(result.best.point)
        if config is None:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        written = [output_dir / "best.yaml"]
        written[0].write_text(yaml.safe_dump(dict(config), sort_keys=True), "utf-8")

        command = driver.render_launch_command(result.best.point)
        if command is not None:
            path = output_dir / "best.sh"
            path.write_text(f"#!/bin/sh\n{command}\n", encoding="utf-8")
            written.append(path)
        return written


@register_reporter("markdown")
class MarkdownReporter(Reporter):
    """``summary.md``: winners per bucket, leaderboards, and a failure digest."""

    def __init__(self, filename: str = "summary.md", top_n: int = 10) -> None:
        self.filename = filename
        self.top_n = top_n

    def emit(self, result: TuneResult, output_dir: Path) -> List[Path]:
        lines = [f"# {result.task.name}", ""]
        lines += self._overview(result)
        for bucket in result.buckets:
            lines += self._bucket_section(result, bucket)
        lines += self._rejections(result)
        lines += self._failures(result)

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return [path]

    def _overview(self, result: TuneResult) -> List[str]:
        counts: Dict[str, int] = {}
        for measurement in result.measurements:
            counts[measurement.status.value] = (
                counts.get(measurement.status.value, 0) + 1
            )
        tally = " | ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        out = [f"{len(result.measurements)} trials: {tally}", ""]
        if result.partial_reason is not None:
            out += [f"> Interrupted: {result.partial_reason}", ""]
        return out

    def _bucket_section(self, result: TuneResult, bucket: str) -> List[str]:
        ranked = result.ranking_by_bucket[bucket]
        title = "## Best" if bucket == "" else f"## Best for `{bucket}`"
        best = result.best_by_bucket.get(bucket)
        if best is None:
            return [title, "", "No feasible candidate.", ""]

        out = [title, "", *self._diff_table(result, best), ""]
        out += [*_margin_line(ranked), ""]
        metrics = sorted({m for e in ranked for m in e.measurement.metrics})
        out += ["| # | " + " | ".join(["point", *metrics]) + " |"]
        out += ["|---" * (len(metrics) + 2) + "|"]
        for i, evaluation in enumerate(ranked[: self.top_n], start=1):
            cells = [_cell(evaluation.measurement.metric(m)) for m in metrics]
            out.append(f"| {i} | `{evaluation.point}` | " + " | ".join(cells) + " |")
        return out + [""]

    def _diff_table(self, result: TuneResult, best: Evaluation) -> List[str]:
        baseline = result.task.space.baseline()
        changes = baseline.diff(best.point)
        if not changes:
            return ["Baseline is the winner; no knob changed."]
        out = ["| knob | baseline | best |", "|---|---|---|"]
        out += [f"| {name} | {was} | {now} |" for name, (was, now) in changes.items()]
        return out

    def _rejections(self, result: TuneResult) -> List[str]:
        """What each feasibility rule removed, and one example of why.

        A rejected point is measured by nothing and appears nowhere else, so a
        rule that is wrong in the expensive direction -- rejecting candidates
        that would have run -- leaves no trace at all. Counting per rule is what
        makes that visible: a rule that suddenly takes most of the space is a
        rule to go and check.
        """
        rejected = [
            m for m in result.measurements if m.status is TrialStatus.INFEASIBLE
        ]
        if not rejected:
            return []
        total = len({m.trial.point.fingerprint for m in result.measurements})
        by_rule: Dict[str, List[str]] = {}
        for m in rejected:
            for reason in m.message.split("; "):
                name, _, detail = reason.partition(": ")
                by_rule.setdefault(name, []).append(detail)
        out = [
            "## Rejected before measuring",
            "",
            f"{len({m.trial.point.fingerprint for m in rejected})} of {total} "
            "points never reached the GPU.",
            "",
            "| rule | points | example |",
            "|---|---|---|",
        ]
        for name, details in sorted(by_rule.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| `{name}` | {len(details)} | {details[0]} |")
        return out + [""]

    def _failures(self, result: TuneResult) -> List[str]:
        failed = [m for m in result.measurements if m.failure is not None]
        if not failed:
            return []
        out = ["## Failures", "", "| point | kind | detail |", "|---|---|---|"]
        out += [
            f"| `{m.trial.point}` | {m.failure.value} | {m.message} |" for m in failed
        ]
        return out + [""]
