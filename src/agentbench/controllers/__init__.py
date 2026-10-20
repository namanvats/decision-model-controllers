from agentbench.config import ConfigError, ResolvedArm
from agentbench.controllers.base import OPTIONS, Controller, Decision, decide_safely
from agentbench.controllers.jev import JevController
from agentbench.controllers.placebo import PlaceboController
from agentbench.controllers.rules import RulesController
from agentbench.controllers.small_llm import SmallLLMController
from agentbench.openrouter import OpenRouter

__all__ = ["OPTIONS", "Controller", "Decision", "decide_safely", "make_controller"]


def _check_options(arm: ResolvedArm) -> None:
    source = arm.criteria or arm.placebo
    for question, options in OPTIONS.items():
        given = getattr(source, question)
        names = set(given.criteria) if arm.criteria else set(given)
        if names != set(options):
            raise ConfigError(f"{arm.id}: {question} options {sorted(names)} != {sorted(options)}")


def make_controller(
    arm: ResolvedArm, client: OpenRouter | None, trial_id: str
) -> Controller | None:
    ctrl = arm.cfg.controller
    match ctrl.type:
        case "none":
            return None
        case "rules":
            return RulesController(arm.triggers)
        case "placebo":
            _check_options(arm)
            return PlaceboController(arm.placebo, trial_id)
        case "small_llm":
            _check_options(arm)
            return SmallLLMController(
                client, arm.controller_model, arm.criteria, arm.controller_prompt, ctrl.timeout_s
            )
        case "jev":
            _check_options(arm)
            return JevController(
                client, arm.jev, arm.criteria, ctrl.timeout_s, ctrl.finish_threshold
            )
    raise ConfigError(f"unknown controller type {ctrl.type}")
