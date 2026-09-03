"""Name -> class lookup for every pluggable component.

A config selects components by name (``strategy: asha``), so every extension
point needs a registry. Out-of-tree plugins register by importing their module;
a future entry-point scan can be added here without touching callers.

The registries are separate per kind rather than one global namespace so that a
``grid`` strategy and a ``grid`` reporter can coexist, and so an unknown name
produces a useful error listing only the plausible alternatives.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Type, TypeVar

__all__ = [
    "Registry",
    "SPACES",
    "STRATEGIES",
    "EXECUTORS",
    "DRIVERS",
    "OBJECTIVES",
    "CONSTRAINTS",
    "STORES",
    "REPORTERS",
    "register_space",
    "register_strategy",
    "register_executor",
    "register_driver",
    "register_objective",
    "register_constraint",
    "register_store",
    "register_reporter",
]

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of one component kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._entries and self._entries[name] is not cls:
                raise ValueError(
                    f"{self.kind} {name!r} is already registered to "
                    f"{self._entries[name].__name__}"
                )
            self._entries[name] = cls
            # Keep the declared name on the class so logs and reports can
            # identify a component without a reverse lookup.
            if getattr(cls, "name", None) in (None, "", self.kind):
                cls.name = name  # type: ignore[attr-defined]
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}"
            ) from None

    def create(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> List[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


SPACES: Registry = Registry("space")
STRATEGIES: Registry = Registry("strategy")
EXECUTORS: Registry = Registry("executor")
DRIVERS: Registry = Registry("driver")
OBJECTIVES: Registry = Registry("objective")
CONSTRAINTS: Registry = Registry("constraint")
STORES: Registry = Registry("store")
REPORTERS: Registry = Registry("reporter")

register_space = SPACES.register
register_strategy = STRATEGIES.register
register_executor = EXECUTORS.register
register_driver = DRIVERS.register
register_objective = OBJECTIVES.register
register_constraint = CONSTRAINTS.register
register_store = STORES.register
register_reporter = REPORTERS.register
