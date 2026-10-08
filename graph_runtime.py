"""Request-owned execution objects that must never enter a checkpoint."""
from dataclasses import dataclass
from typing import Any
from threading import Event
from time import perf_counter
from performance import TurnProfile


@dataclass
class GraphRuntime:
    profile: TurnProfile
    trace: Any
    started: float
    cancelled: Event
    guard: Any = None
    memory_service: Any = None
    context_service: Any = None
    context_save: Any = None
    context_cached: Any = None
    memory_tools: Any = None
    tool_context: Any = None
    decision_cached: Any = None
    decision_save: Any = None

    def enrich(self, state):
        return {**state, '_profile': self.profile, '_trace': self.trace,
                '_started': self.started, '_cancelled': self.cancelled}


def restore_runtime(state, trace, cancelled=None, guard=None):
    profile = TurnProfile()
    profile.stages = {k: dict(v) for k, v in state.get('timings', {}).items()}
    return GraphRuntime(profile, trace, perf_counter(), cancelled or Event(), guard)
