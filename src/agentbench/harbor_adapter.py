import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agentbench import __version__
from agentbench.config import load_arm
from agentbench.controllers import make_controller
from agentbench.loop import AgentLoop
from agentbench.openrouter import OpenRouter, OpenRouterModel, ScriptedModel
from agentbench.telemetry import Recorder
from agentbench.tools import TIMEOUT_EXIT_CODE, ExecOutput

EXEC_GRACE_S = 60


class ControlledAgentOptions(AgentOptions):
    arm: str
    trial_dir: str
    trial_id: str
    limits_ref: str | None = None


class ControlledAgent(BaseAgent):
    """External Harbor agent: the loop runs host-side, only shell commands reach the sandbox."""

    options_model = ControlledAgentOptions

    @staticmethod
    def name() -> str:
        return "agentbench-controlled"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        opts: ControlledAgentOptions = self.options
        arm = load_arm(opts.arm, opts.limits_ref)
        needs_api = arm.model.kind == "openrouter" or arm.controller_type in ("small_llm", "jev")
        client = OpenRouter() if needs_api else None
        llm = (
            OpenRouterModel(client, arm.model)
            if arm.model.kind == "openrouter"
            else ScriptedModel()
        )
        controller = make_controller(arm, client, opts.trial_id)
        recorder = Recorder(Path(opts.trial_dir))

        async def exec_fn(command: str, timeout_s: int) -> ExecOutput:
            try:
                result = await asyncio.wait_for(
                    environment.exec(command=command, timeout_sec=timeout_s),
                    timeout_s + EXEC_GRACE_S,
                )
            except TimeoutError:
                return ExecOutput(TIMEOUT_EXIT_CODE, "", timed_out=True)
            output = (result.stdout or "") + (
                f"\n[stderr]\n{result.stderr}" if result.stderr else ""
            )
            return ExecOutput(
                result.return_code, output, timed_out=result.return_code == TIMEOUT_EXIT_CODE
            )

        try:
            result = await AgentLoop(arm, llm, exec_fn, controller, recorder).run(instruction)
        finally:
            if client:
                await client.aclose()

        totals = recorder.totals()
        cached = sum(m["cached_tokens"] for m in recorder.usage["main"].values())
        context.n_input_tokens = int(
            sum(m["prompt_tokens"] for m in recorder.usage["main"].values())
        )
        context.n_cache_tokens = int(cached)
        context.n_output_tokens = int(
            sum(m["completion_tokens"] for m in recorder.usage["main"].values())
        )
        context.cost_usd = totals["usd_total"]
        context.metadata = {
            "trial_id": opts.trial_id,
            "arm": arm.id,
            "termination_reason": result.termination_reason,
            "steps": result.steps,
            "controller_calls": result.controller_calls,
            "usd_controller": totals["usd_controller"],
        }
