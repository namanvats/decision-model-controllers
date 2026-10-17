"""Live Phase 0 contract tests; spend a fraction of a cent. Run: uv run pytest -m integration"""

import os

import pytest
from dotenv import load_dotenv

from agentbench.config import PROJECT_ROOT
from agentbench.controllers import decide_safely, make_controller
from agentbench.openrouter import OpenRouter
from tests.helpers import arm_with

load_dotenv(PROJECT_ROOT / ".env")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY unset"),
]
STATE = {
    "task_instruction": "Create /app/hello.txt containing 'hi'.",
    "trigger": "T3",
    "last_5_actions": [{"command": "echo hi > /app/hello.txt", "exit_code": 0, "output_tail": ""}],
    "finish_summary": "Created the file.",
}


@pytest.fixture
async def client():
    c = OpenRouter()
    yield c
    await c.aclose()


async def test_reasoning_model_reports_cost_and_provider(client):
    arm = arm_with("agents/sonnet-native.yaml")
    _, usage = await client.chat(arm.model, [{"role": "user", "content": "Reply with: ok"}])
    assert usage.cost_usd > 0 and usage.prompt_tokens > 0
    assert usage.served_model.startswith(arm.model.slug)


@pytest.mark.parametrize("arm_path", ["agents/sonnet-jev.yaml", "agents/sonnet-smallllm.yaml"])
async def test_controllers_answer_without_fallback(client, arm_path):
    arm = arm_with(arm_path)
    decision = await decide_safely(make_controller(arm, client, "probe"), "finish", STATE)
    assert not decision.fallback, decision.error
    assert decision.usage.cost_usd > 0
    if arm.controller_type == "jev":
        assert "jev-1.13" in decision.usage.served_model
        assert set(decision.probabilities) == {"finish", "verify"}
