from typing import Any

from agentbench.config import Criteria, JevCfg
from agentbench.controllers.base import ControllerOutputError, Decision
from agentbench.openrouter import OpenRouter


class JevController:
    name = "jev"

    def __init__(
        self,
        client: OpenRouter,
        jev: JevCfg,
        criteria: Criteria,
        timeout_s: float,
        finish_threshold: float | None = None,
    ):
        self.client = client
        self.jev = jev
        self.criteria = criteria
        self.timeout_s = timeout_s
        self.finish_threshold = finish_threshold

    async def decide(self, question: str, state: dict[str, Any]) -> Decision:
        spec = getattr(self.criteria, question)
        questions = {
            question: {
                "type": "choice",
                "instructions": spec.instructions,
                "criteria": spec.criteria,
            }
        }
        data, usage = await self.client.decide(self.jev, state, questions, self.timeout_s)
        try:
            answer = data["answers"][question]
            choice = answer["choice"]
        except (KeyError, TypeError) as e:
            raise ControllerOutputError(f"malformed response: {str(data)[:200]}", usage) from e
        probabilities = answer.get("probabilities") or {}
        if question == "finish" and self.finish_threshold is not None:
            if "finish" not in probabilities:
                raise ControllerOutputError("threshold policy needs probabilities", usage)
            choice = "finish" if probabilities["finish"] >= self.finish_threshold else "verify"
        return Decision(
            question=question,
            choice=choice,
            probabilities=answer.get("probabilities"),
            confidence=answer.get("confidence"),
            usage=usage,
            latency_s=usage.latency_s,
        )
