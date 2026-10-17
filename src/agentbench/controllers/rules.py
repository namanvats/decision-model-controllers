from typing import Any

from agentbench.config import TriggerCfg
from agentbench.controllers.base import Decision


class RulesController:
    """Deterministic policy (experiment_plan.md §5.6)."""

    name = "rules"
    timeout_s = None

    def __init__(self, cfg: TriggerCfg):
        self.cfg = cfg

    async def decide(self, question: str, state: dict[str, Any]) -> Decision:
        if question == "finish":
            choice = "finish" if state["verified_after_last_edit"] else "verify"
        elif (
            state["consecutive_failures"] >= self.cfg.rules_abandon_min_failures
            and state["step"] >= self.cfg.g2_min_step_fraction * state["max_steps"]
        ):
            choice = "abandon"
        elif state["trigger"] in ("T1", "T2"):
            choice = "recover"
        else:
            choice = "continue"
        return Decision(question=question, choice=choice)
