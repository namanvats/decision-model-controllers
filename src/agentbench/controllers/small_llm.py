import json
from typing import Any

from agentbench.config import Criteria, ModelCfg
from agentbench.controllers.base import ControllerOutputError, Decision
from agentbench.openrouter import OpenRouter


class SmallLLMController:
    """Same state and option descriptions as Jev, answered by a small chat model (§5.7)."""

    name = "small_llm"

    def __init__(
        self, client: OpenRouter, cfg: ModelCfg, criteria: Criteria, prompt: str, timeout_s: float
    ):
        self.client = client
        self.cfg = cfg
        self.criteria = criteria
        self.prompt = prompt
        self.timeout_s = timeout_s

    def render(self, question: str, state: dict[str, Any]) -> str:
        spec = getattr(self.criteria, question)
        return self.prompt.format(
            instructions=spec.instructions,
            options="\n".join(f"- {k}: {v}" for k, v in spec.criteria.items()),
            option_names=", ".join(spec.criteria),
            state=json.dumps(state, indent=1),
        )

    async def decide(self, question: str, state: dict[str, Any]) -> Decision:
        options = list(getattr(self.criteria, question).criteria)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": options}},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            },
        }
        message, usage = await self.client.chat(
            self.cfg,
            [{"role": "user", "content": self.render(question, state)}],
            response_format=response_format,
            timeout=self.timeout_s,
            max_retries=0,
        )
        content = message.get("content") or ""
        try:
            choice = json.loads(content)["choice"]
        except (ValueError, KeyError, TypeError) as e:
            raise ControllerOutputError(f"malformed response: {content[:200]!r}", usage) from e
        return Decision(question=question, choice=choice, usage=usage, latency_s=usage.latency_s)
