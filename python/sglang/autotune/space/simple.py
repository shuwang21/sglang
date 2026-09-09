"""A space whose knobs are declared inline.

The ``server_args`` space derives its knobs from ``ServerArgs`` metadata. This
one takes them verbatim, which is what a unit test, a kernel tile sweep, or a
hand-written one-off needs.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from sglang.autotune.registry import register_space
from sglang.autotune.space.base import (
    Categorical,
    Conditional,
    Domain,
    FeasibilityRule,
    Knob,
    Space,
)

__all__ = ["SimpleSpace"]


def _as_domain(spec: Any) -> Domain:
    return spec if isinstance(spec, Domain) else Categorical(values=tuple(spec))


@register_space("simple")
class SimpleSpace(Space):
    """Knobs given as ``name -> values`` or ``name -> Domain``."""

    def __init__(
        self,
        knobs: Mapping[str, Any],
        *,
        group: str = "misc",
        fixed: Optional[Mapping[str, Any]] = None,
        rules: Sequence[FeasibilityRule] = (),
        conditionals: Sequence[Conditional] = (),
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            fixed=fixed, rules=rules, conditionals=conditionals, context=context
        )
        self._knobs = tuple(
            Knob(name=name, domain=_as_domain(spec), group=group)
            for name, spec in knobs.items()
        )

    def knobs(self) -> Sequence[Knob]:
        return self._knobs
