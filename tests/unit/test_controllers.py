import asyncio
import json

import httpx
import pytest
import respx

from agentbench.config import PlaceboDist
from agentbench.controllers import decide_safely, make_controller
from agentbench.controllers.base import Decision
from agentbench.controllers.placebo import PlaceboController
from agentbench.openrouter import OpenRouter
from tests.helpers import arm_with

JEV_URL = "https://openrouter.ai/api/v1/systemone"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def state(**kw):
    return {
        "trigger": "T1",
        "step": 5,
        "max_steps": 10,
        "consecutive_failures": 0,
        "verified_after_last_edit": False,
        **kw,
    }


@pytest.fixture
async def client():
    c = OpenRouter(api_key="test")
    yield c
    await c.aclose()


def jev_response(question="finish", choice="verify"):
    return {
        "id": "d-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            question: {
                "type": "choice",
                "choice": choice,
                "confidence": 0.8,
                "probabilities": {"finish": 0.2, "verify": 0.8},
            }
        },
        "usage": {"input_tokens": 300, "output_tokens": 0, "cost": 0.0000126},
    }


def chat_response(content):
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "anthropic/claude-haiku-5.5",
        "provider": "Anthropic",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "cost": 0.00002,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


@pytest.mark.parametrize(
    ("question", "st", "expected"),
    [
        ("finish", state(verified_after_last_edit=False), "verify"),
        ("finish", state(verified_after_last_edit=True), "finish"),
        ("progress", state(trigger="T1"), "recover"),
        ("progress", state(trigger="T2"), "recover"),
        ("progress", state(trigger="T4"), "continue"),
        ("progress", state(trigger="T1", consecutive_failures=4, step=3), "recover"),
        ("progress", state(trigger="T1", consecutive_failures=4, step=4), "abandon"),
    ],
)
async def test_rules_policy(question, st, expected):
    rules = make_controller(arm_with("agents/sonnet-rules.yaml"), None, "t")
    assert (await rules.decide(question, st)).choice == expected


@respx.mock
async def test_jev_request_and_parse(client):
    route = respx.post(JEV_URL).respond(json=jev_response())
    jev = make_controller(arm_with("agents/sonnet-jev.yaml"), client, "t")
    d = await decide_safely(jev, "finish", state())
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["questions"]["finish"]["type"] == "choice"
    assert set(body["questions"]["finish"]["criteria"]) == {"finish", "verify"}
    assert route.calls[0].request.headers["authorization"] == "Bearer test"
    assert (d.choice, d.fallback, d.confidence) == ("verify", False, 0.8)
    assert d.probabilities == {"finish": 0.2, "verify": 0.8}
    assert d.usage.cost_usd == pytest.approx(0.0000126)
    assert d.usage.served_model == "typesafe/jev-1.13-20260917"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"answers": {}, "usage": {"cost": 0.001}}),
        httpx.Response(200, json=jev_response(choice="nonsense")),
        httpx.Response(500, json={"error": {"code": 500, "message": "boom"}}),
        httpx.ReadTimeout("slow"),
    ],
)
@respx.mock
async def test_jev_failures_fall_back(client, response):
    respx.post(JEV_URL).mock(side_effect=[response])
    jev = make_controller(arm_with("agents/sonnet-jev.yaml"), client, "t")
    d = await decide_safely(jev, "finish", state())
    assert d.fallback and d.choice == "finish" and d.error


@respx.mock
async def test_malformed_response_still_charges_cost(client):
    respx.post(JEV_URL).respond(json={"answers": {}, "usage": {"cost": 0.001}})
    jev = make_controller(arm_with("agents/sonnet-jev.yaml"), client, "t")
    d = await decide_safely(jev, "progress", state())
    assert d.fallback and d.choice == "continue" and d.usage.cost_usd == 0.001


@respx.mock
async def test_small_llm_parses_structured_output(client):
    route = respx.post(CHAT_URL).respond(json=chat_response('{"choice": "verify"}'))
    small = make_controller(arm_with("agents/sonnet-smallllm.yaml"), client, "t")
    d = await decide_safely(small, "finish", state())
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "anthropic/claude-haiku-5.5" and "temperature" not in body
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"]["json_schema"]["schema"]["properties"]["choice"]["enum"] == [
        "finish",
        "verify",
    ]
    assert '"verified_after_last_edit": false' in body["messages"][0]["content"]
    assert (d.choice, d.fallback, d.usage.provider) == ("verify", False, "Anthropic")


@respx.mock
async def test_small_llm_invalid_json_falls_back(client):
    respx.post(CHAT_URL).respond(json=chat_response("I think you should verify"))
    small = make_controller(arm_with("agents/sonnet-smallllm.yaml"), client, "t")
    d = await decide_safely(small, "finish", state())
    assert d.fallback and d.choice == "finish" and d.usage.cost_usd == 0.00002


async def test_timeout_falls_back():
    class Slow:
        name, timeout_s = "slow", 0.01

        async def decide(self, question, st):
            await asyncio.sleep(1)
            return Decision(question, "abandon")

    d = await decide_safely(Slow(), "progress", state())
    assert d.fallback and d.choice == "continue" and "TimeoutError" in d.error


async def test_placebo_is_deterministic_per_trial():
    dist = PlaceboDist(
        schema_version=1,
        seed=7,
        source="test",
        progress={"continue": 0.7, "investigate": 0.1, "recover": 0.1, "abandon": 0.1},
        finish={"finish": 0.5, "verify": 0.5},
    )

    async def run(trial_id):
        p = PlaceboController(dist, trial_id)
        return [(await p.decide("progress", {})).choice for _ in range(20)]

    assert await run("a") == await run("a")
    assert await run("a") != await run("b")
