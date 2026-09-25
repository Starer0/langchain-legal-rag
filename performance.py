"""Lightweight, per-turn stage timings; no question or document text is stored."""

from time import perf_counter


class TurnProfile:
    def __init__(self):
        self.stages = {}

    def measure(self, stage, operation):
        started = perf_counter()
        try:
            return operation()
        finally:
            entry = self.stages.setdefault(stage, {"calls": 0, "duration_ms": 0.0})
            entry["calls"] += 1
            entry["duration_ms"] += (perf_counter() - started) * 1000

    def snapshot(self, total_ms=None):
        result = {
            "stages": {
                name: {"calls": item["calls"], "duration_ms": round(item["duration_ms"], 2)}
                for name, item in self.stages.items()
            },
            "model_calls": sum(
                self.stages.get(name, {}).get("calls", 0)
                for name in ("rewrite", "decompose", "answer")
            ),
            "reranker_calls": self.stages.get("rerank", {}).get("calls", 0),
        }
        if total_ms is not None:
            result["total_ms"] = round(total_ms, 2)
        return result


def measure(state, stage, operation):
    profile = state.get("_profile")
    return profile.measure(stage, operation) if profile is not None else operation()
