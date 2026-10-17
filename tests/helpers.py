import json
from dataclasses import replace
from pathlib import Path

import yaml

from agentbench.config import Limits, load_arm
from agentbench.telemetry import write_json
from agentbench.tools import ExecOutput


def arm_with(path: str = "agents/_mock.yaml", **limits):
    """Load a real arm with smoke limits, optionally overriding individual limits."""
    arm = load_arm(path, "config/limits.yaml#smoke")
    return replace(arm, limits=Limits(**{**arm.limits.model_dump(), **limits}))


class FakeEnv:
    """Sandbox stand-in: `false` fails, everything else succeeds and echoes the command."""

    def __init__(self):
        self.commands: list[str] = []

    async def __call__(self, command: str, timeout_s: int) -> ExecOutput:
        self.commands.append(command)
        if command.strip() == "false":
            return ExecOutput(1, "")
        return ExecOutput(0, f"ran {command}")


def write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def write_trial(
    root: Path,
    exp_id: str,
    task: str,
    arm: str,
    attempt: int,
    *,
    passed: bool,
    usd: float,
    steps: int = 10,
    termination: str = "finished",
    infra_error: bool = False,
    decisions: list[dict] | None = None,
    triggers: list[dict] | None = None,
) -> Path:
    """Write the files a finished trial leaves behind (used by analysis/tasks/replay tests)."""
    short = task.rsplit("/", 1)[-1]
    trial_id = f"{exp_id}--{short}--{arm}--a{attempt}"
    d = root / exp_id / short / arm / str(attempt)
    d.mkdir(parents=True, exist_ok=True)
    write_json(
        d / "manifest.json",
        {"trial_id": trial_id, "trial_uid": trial_id[-16:], "trial_cap_usd": 1.0},
    )
    write_json(
        d / "usage.json",
        {
            "termination_reason": termination,
            "steps": steps,
            "totals": {
                "usd_total": usd,
                "usd_main": usd * 0.95,
                "usd_controller": usd * 0.05,
                "tokens_main": 1000,
                "tokens_controller": 50,
                "main_calls": steps,
                "controller_calls": len(decisions or []),
                "latency_main_s": 10.0,
                "latency_controller_s": 0.5,
            },
        },
    )
    write_json(
        d / "outcome.json",
        {
            "trial_id": trial_id,
            "experiment_id": exp_id,
            "task": task,
            "arm": arm,
            "attempt": attempt,
            "passed": passed,
            "reward": float(passed),
            "termination_reason": termination,
            "infra_error": infra_error,
            "steps": steps,
            "usd_total": usd,
            "wall_s": {"total": 100.0, "agent": 80.0, "provisioning": 10.0, "verifier": 10.0},
        },
    )
    events = [{"kind": "tool_call", "command": "ls", "exit_code": 0, "timed_out": False}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (d / "decision_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "step": 1,
                    "trigger": "T1",
                    "question": "progress",
                    "applied": None,
                    "guard": None,
                    "fallback": False,
                    "latency_s": 0.1,
                    "cost_usd": 0.0,
                    "confidence": None,
                    **x,
                }
            )
            + "\n"
            for x in decisions or []
        )
    )
    (d / "trigger_states.jsonl").write_text("".join(json.dumps(x) + "\n" for x in triggers or []))
    return d
