import random
from typing import Any

from agentbench.config import PlaceboDist
from agentbench.controllers.base import Decision


class PlaceboController:
    """Random choices drawn from Jev's dev-phase choice frequencies (§5.8)."""

    name = "placebo"
    timeout_s = None

    def __init__(self, dist: PlaceboDist, trial_id: str):
        self.dist = dist
        self.rng = random.Random(f"{dist.seed}:{trial_id}")

    async def decide(self, question: str, state: dict[str, Any]) -> Decision:
        weights: dict[str, float] = getattr(self.dist, question)
        options = list(weights)
        choice = self.rng.choices(options, weights=[weights[o] for o in options])[0]
        return Decision(question=question, choice=choice)
