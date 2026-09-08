"""SGLang auto-tune framework.

Searches the deployment configuration space for a model + hardware + workload
and emits a launchable config. See ``docs/developer_guide/auto_tune_design.md``.

The abstractions are the product here: a contributor adds a search strategy, a
knob family, an execution backend, or an objective without touching the loop.

    from sglang.autotune import TuneTask, tune
    result = tune(task)
"""

from sglang.autotune.driver import MockDriver
from sglang.autotune.executor.base import Executor, SerialExecutor, TrialSlot
from sglang.autotune.executor.local import LocalExecutor
from sglang.autotune.measure import LoadPlan, MeasurementDriver
from sglang.autotune.objective import (
    Constraint,
    Direction,
    MetricThreshold,
    Objective,
    ScalarObjective,
)
from sglang.autotune.orchestrator import Orchestrator, tune
from sglang.autotune.registry import (
    DRIVERS,
    EXECUTORS,
    REPORTERS,
    SPACES,
    STORES,
    STRATEGIES,
)
from sglang.autotune.report import Reporter
from sglang.autotune.space.base import (
    Categorical,
    FeasibilityRule,
    FloatRange,
    IntRange,
    Knob,
    Space,
)
from sglang.autotune.store import TrialStore
from sglang.autotune.strategy.base import Strategy
from sglang.autotune.task import HardwareSpec, ModelSpec, TuneResult, TuneTask
from sglang.autotune.types import (
    Budget,
    Fidelity,
    LoadPoint,
    Measurement,
    Point,
    Provenance,
    Trial,
    TrialStatus,
    Workload,
)

__all__ = [
    # entry points
    "tune",
    "Orchestrator",
    "TuneTask",
    "TuneResult",
    "ModelSpec",
    "HardwareSpec",
    # values
    "Point",
    "Trial",
    "Measurement",
    "TrialStatus",
    "Workload",
    "LoadPoint",
    "Fidelity",
    "Budget",
    "Provenance",
    # extension points
    "Space",
    "Knob",
    "Categorical",
    "IntRange",
    "FloatRange",
    "FeasibilityRule",
    "Strategy",
    "Executor",
    "SerialExecutor",
    "LocalExecutor",
    "TrialSlot",
    "MeasurementDriver",
    "MockDriver",
    "LoadPlan",
    "Objective",
    "ScalarObjective",
    "Direction",
    "Constraint",
    "MetricThreshold",
    "TrialStore",
    "Reporter",
    # registries
    "SPACES",
    "STRATEGIES",
    "EXECUTORS",
    "DRIVERS",
    "STORES",
    "REPORTERS",
]
