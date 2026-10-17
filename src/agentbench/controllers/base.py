import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from agentbench.openrouter import CallUsage

OPTIONS = {
    "progress": ("continue", "investigate", "recover", "abandon"),
    "finish": ("finish", "verify"),
}
FALLBACK = {"progress": "continue", "finish": "finish"}


@dataclass
class Decision:
    question: str
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    usage: CallUsage | None = None
    latency_s: float = 0.0
    error: str | None = None
    fallback: bool = False


class Controller(Protocol):
    name: str
    timeout_s: float | None

    async def decide(self, question: str, state: dict[str, Any]) -> Decision: ...


class ControllerOutputError(Exception):
    def __init__(self, message: str, usage: CallUsage | None = None):
        super().__init__(message)
        self.usage = usage


async def decide_safely(controller: Controller, question: str, state: dict[str, Any]) -> Decision:
    """G4: timeout, transport error, or malformed output falls back to the safe default."""
    start = time.monotonic()
    try:
        decision = await asyncio.wait_for(controller.decide(question, state), controller.timeout_s)
        if decision.choice not in OPTIONS[question]:
            raise ControllerOutputError(f"invalid choice {decision.choice!r}", decision.usage)
        return decision
    except Exception as e:
        return Decision(
            question=question,
            choice=FALLBACK[question],
            usage=getattr(e, "usage", None),
            latency_s=time.monotonic() - start,
            error=f"{type(e).__name__}: {e}",
            fallback=True,
        )
