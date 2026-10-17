import json
from typing import Any

from agentbench.config import Limits
from agentbench.triggers import TriggerDetector

MAX_INSTRUCTION_CHARS = 3000
MAX_COMMAND_CHARS = 300
MAX_TAIL_CHARS = 500
MAX_SUMMARY_CHARS = 1000
MAX_STATE_CHARS = 24_000  # ~6k tokens


def build_state(
    *,
    instruction: str,
    trigger: str,
    step: int,
    limits: Limits,
    usd_used: float,
    detector: TriggerDetector,
    controller_history: list[dict],
    finish_summary: str | None = None,
) -> dict[str, Any]:
    """Compact state shown to every controller (experiment_plan.md §5.4).

    Built only from the agent's own transcript: never tests, verifier output, or solutions.
    """
    tail = MAX_TAIL_CHARS
    while True:
        state = {
            "task_instruction": instruction[:MAX_INSTRUCTION_CHARS],
            "trigger": trigger,
            "step": step,
            "max_steps": limits.max_agent_steps,
            "usd_used": round(usd_used, 4),
            "max_trial_usd": limits.max_trial_usd,
            "last_5_actions": [
                {
                    "command": a.command[:MAX_COMMAND_CHARS],
                    "exit_code": "timeout" if a.timed_out else a.exit_code,
                    "output_tail": a.output[-tail:] if tail else "",
                }
                for a in detector.actions[-5:]
            ],
            "consecutive_failures": detector.consecutive_failures,
            "repeated_command_flag": detector.repeated_command(),
            "verification_like_cmds_run": detector.verification_cmds,
            "verified_after_last_edit": detector.verified_after_last_edit,
            "controller_history": controller_history[-20:],
        }
        if trigger == "T3":
            state["finish_summary"] = (finish_summary or "")[:MAX_SUMMARY_CHARS]
        if len(json.dumps(state)) <= MAX_STATE_CHARS or tail == 0:
            return state
        tail //= 2
