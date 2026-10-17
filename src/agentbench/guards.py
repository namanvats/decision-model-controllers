from dataclasses import dataclass

from agentbench.config import Limits, TriggerCfg


def limit_breach(
    limits: Limits,
    *,
    usd: float,
    tokens: int,
    elapsed_s: float,
    steps: int | None = None,
) -> str | None:
    """G1: hard limits always terminate, regardless of controller advice."""
    if steps is not None and steps >= limits.max_agent_steps:
        return "limit_steps"
    if usd >= limits.max_trial_usd:
        return "limit_usd"
    if tokens >= limits.max_total_tokens:
        return "limit_tokens"
    if elapsed_s >= limits.max_wall_s:
        return "limit_wall"
    return None


@dataclass
class GuardContext:
    cfg: TriggerCfg
    step: int
    max_steps: int
    consecutive_failures: int
    t2_fires: int
    verify_count: int


def abandon_allowed(ctx: GuardContext) -> bool:
    """G2."""
    late_enough = ctx.step >= ctx.cfg.g2_min_step_fraction * ctx.max_steps
    stuck = (
        ctx.consecutive_failures >= ctx.cfg.g2_min_consecutive_failures
        or ctx.t2_fires >= ctx.cfg.g2_min_t2_fires
    )
    return late_enough and stuck


def apply_guards(question: str, choice: str, ctx: GuardContext) -> tuple[str, str | None]:
    """Return (applied action, guard that overrode the choice or None)."""
    if question == "progress" and choice == "abandon" and not abandon_allowed(ctx):
        return "recover", "G2"
    if question == "finish" and choice == "verify" and ctx.verify_count >= ctx.cfg.g3_max_verify:
        return "finish", "G3"
    return choice, None
