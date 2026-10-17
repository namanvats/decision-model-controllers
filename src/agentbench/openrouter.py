import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import httpx
import openai
from openai import AsyncOpenAI

from agentbench.config import JevCfg, ModelCfg, load_models

RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass
class CallUsage:
    model: str
    served_model: str | None = None
    provider: str | None = None
    request_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    retries: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatModel(Protocol):
    slug: str

    async def complete(self, messages: list[dict], tools: list[dict]) -> tuple[dict, CallUsage]: ...


def _chat_usage(cfg: ModelCfg, resp: Any, latency: float, retries: int) -> CallUsage:
    usage = resp.usage.model_dump() if resp.usage else {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return CallUsage(
        model=cfg.slug,
        served_model=resp.model,
        provider=getattr(resp, "provider", None),
        request_id=resp.id,
        prompt_tokens=usage.get("prompt_tokens") or 0,
        completion_tokens=usage.get("completion_tokens") or 0,
        cached_tokens=prompt_details.get("cached_tokens") or 0,
        cache_write_tokens=prompt_details.get("cache_write_tokens") or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
        cost_usd=float(usage.get("cost") or 0.0),
        latency_s=latency,
        retries=retries,
    )


class OpenRouter:
    """Thin client for OpenRouter chat completions and the Jev decision endpoint."""

    def __init__(self, api_key: str | None = None):
        gateway = load_models().gateway
        self.api_key = api_key or os.environ[gateway.api_key_env]
        self.chat_client = AsyncOpenAI(
            base_url=gateway.base_url, api_key=self.api_key, max_retries=0
        )
        self.http = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self.chat_client.close()
        await self.http.aclose()

    def _body(self, cfg: ModelCfg) -> dict[str, Any]:
        # OpenRouter picks the host; only hosts supporting every parameter we send qualify.
        body: dict[str, Any] = {"provider": {"require_parameters": True}}
        if cfg.reasoning is not None:
            body["reasoning"] = cfg.reasoning
        if cfg.prompt_caching:
            body["cache_control"] = {"type": "ephemeral"}
        return body

    async def chat(
        self,
        cfg: ModelCfg,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
    ) -> tuple[dict, CallUsage]:
        kwargs: dict[str, Any] = {
            "model": cfg.slug,
            "messages": messages,
            "max_tokens": cfg.max_output_tokens,
            "extra_body": self._body(cfg),
            "timeout": timeout,
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        retries = 0
        start = time.monotonic()
        while True:
            try:
                resp = await self.chat_client.chat.completions.create(**kwargs)
                break
            except RETRYABLE:
                if retries >= max_retries:
                    raise
                retries += 1
                await asyncio.sleep(2**retries)
        if not resp.choices:
            raise RuntimeError(f"empty response from {cfg.slug}: {resp}")
        message = resp.choices[0].message.model_dump(exclude_none=True)
        return message, _chat_usage(cfg, resp, time.monotonic() - start, retries)

    async def decide(
        self, jev: JevCfg, state: dict, questions: dict, timeout: float
    ) -> tuple[dict, CallUsage]:
        start = time.monotonic()
        resp = await self.http.post(
            jev.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": jev.slug, "state": state, "questions": questions},
            timeout=timeout,
        )
        latency = time.monotonic() - start
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return data, CallUsage(
            model=jev.slug,
            served_model=data.get("model"),
            provider=data.get("provider"),
            request_id=data.get("id"),
            prompt_tokens=usage.get("input_tokens") or 0,
            completion_tokens=usage.get("output_tokens") or 0,
            cost_usd=float(usage.get("cost") or 0.0),
            latency_s=latency,
        )


class OpenRouterModel:
    def __init__(self, client: OpenRouter, cfg: ModelCfg):
        self.client = client
        self.cfg = cfg
        self.slug = cfg.slug

    async def complete(self, messages: list[dict], tools: list[dict]) -> tuple[dict, CallUsage]:
        return await self.client.chat(self.cfg, messages, tools)


MOCK_SCRIPT = [
    {"bash": "pwd && ls -la"},
    {"bash": "false"},
    {"bash": "false"},
    {"bash": "cat /etc/os-release | head -3"},
    {"finish": "Mock run complete."},
]


class ScriptedModel:
    """Deterministic stand-in for the reasoning model (smoke runs and tests).

    Each turn is {"bash": cmd}, {"finish": summary}, {"text": reply}, or a list of those
    (several tool calls in one response). After the script ends it keeps calling finish.
    """

    slug = "mock/scripted"

    def __init__(self, script: list | None = None, cost_per_call: float = 0.0):
        self.script = list(MOCK_SCRIPT if script is None else script)
        self.cost_per_call = cost_per_call
        self.calls = 0

    async def complete(self, messages: list[dict], tools: list[dict]) -> tuple[dict, CallUsage]:
        turn = self.script[self.calls] if self.calls < len(self.script) else {"finish": "done"}
        self.calls += 1
        message: dict[str, Any] = {"role": "assistant", "content": None}
        tool_calls = []
        for i, item in enumerate(turn if isinstance(turn, list) else [turn]):
            if "text" in item:
                message["content"] = item["text"]
                continue
            name, value = next(iter(item.items()))
            args = {"command": value} if name == "bash" else {"summary": value}
            tool_calls.append(
                {
                    "id": f"call_{self.calls}_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )
        if tool_calls:
            message["tool_calls"] = tool_calls
        usage = CallUsage(
            model=self.slug, prompt_tokens=100, completion_tokens=20, cost_usd=self.cost_per_call
        )
        return message, usage
