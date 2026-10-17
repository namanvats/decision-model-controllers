import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentbench.openrouter import CallUsage

STREAMS = ("events", "trigger_states", "decision_events")
USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost_usd",
    "latency_s",
    "retries",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


class Recorder:
    """Append-only, line-flushed trial telemetry (infra_plan.md §12)."""

    def __init__(self, trial_dir: Path):
        self.dir = trial_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {
            name: (self.dir / f"{name}.jsonl").open("a", buffering=1) for name in STREAMS
        }
        self.usage: dict[str, dict[str, dict[str, float]]] = {
            "main": defaultdict(lambda: defaultdict(float)),
            "controller": defaultdict(lambda: defaultdict(float)),
        }

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        self._files[stream].write(json.dumps({"ts": time.time(), **record}, default=str) + "\n")

    def event(self, kind: str, **data: Any) -> None:
        self._write("events", {"kind": kind, **data})

    def trigger_state(self, **data: Any) -> None:
        self._write("trigger_states", data)

    def decision(self, **data: Any) -> None:
        self._write("decision_events", data)

    def add_usage(self, role: str, usage: CallUsage) -> None:
        totals = self.usage[role][usage.model]
        totals["calls"] += 1
        totals["errors"] += usage.error is not None
        for field in USAGE_FIELDS:
            totals[field] += getattr(usage, field)

    def _sum(self, role: str, field: str) -> float:
        return sum(m[field] for m in self.usage[role].values())

    def totals(self) -> dict[str, float]:
        usd_main = self._sum("main", "cost_usd")
        usd_controller = self._sum("controller", "cost_usd")
        return {
            "usd_main": usd_main,
            "usd_controller": usd_controller,
            "usd_total": usd_main + usd_controller,
            "main_calls": int(self._sum("main", "calls")),
            "controller_calls": int(self._sum("controller", "calls")),
            "tokens_main": int(
                self._sum("main", "prompt_tokens") + self._sum("main", "completion_tokens")
            ),
            "tokens_controller": int(
                self._sum("controller", "prompt_tokens")
                + self._sum("controller", "completion_tokens")
            ),
            "latency_main_s": self._sum("main", "latency_s"),
            "latency_controller_s": self._sum("controller", "latency_s"),
        }

    def close(self, result: dict[str, Any]) -> None:
        write_json(
            self.dir / "usage.json",
            {**result, "totals": self.totals(), "by_model": self.usage},
        )
        for f in self._files.values():
            f.close()
