import time
from dataclasses import asdict, dataclass

from agentbench.config import ResolvedArm
from agentbench.controllers import Controller, decide_safely
from agentbench.guards import GuardContext, apply_guards, limit_breach
from agentbench.openrouter import ChatModel
from agentbench.state import build_state
from agentbench.telemetry import Recorder
from agentbench.tools import (
    TOOLS,
    Action,
    ExecFn,
    clamp_timeout,
    format_action,
    output_hash,
    parse_tool_call,
    truncate,
)
from agentbench.triggers import TriggerDetector

GUIDANCE_PREFIX = "[controller guidance — advisory]"
GUIDANCE_FOR = {"investigate": "I", "recover": "R", "verify": "V"}


@dataclass
class LoopResult:
    termination_reason: str
    steps: int
    controller_calls: int
    finish_summary: str | None = None
    error: str | None = None


class AgentLoop:
    """The single ReAct loop shared by every arm; arms differ only in `controller`."""

    def __init__(
        self,
        arm: ResolvedArm,
        llm: ChatModel,
        exec_fn: ExecFn,
        controller: Controller | None,
        recorder: Recorder,
        clock=time.monotonic,
    ):
        self.arm = arm
        self.limits = arm.limits
        self.llm = llm
        self.exec_fn = exec_fn
        self.controller = controller
        self.rec = recorder
        self.clock = clock
        self.detector = TriggerDetector(arm.triggers)
        self.messages: list[dict] = []
        self.history: list[dict] = []
        self.instruction = ""
        self.step = 0
        self.controller_calls = 0
        self.verify_count = 0
        self.finish_summary: str | None = None
        self.started = 0.0

    async def run(self, instruction: str) -> LoopResult:
        self.instruction = instruction
        self.started = self.clock()
        self.messages = [
            {"role": "system", "content": self.arm.system_prompt},
            {"role": "user", "content": instruction},
        ]
        reason, error = "cancelled", None  # stays "cancelled" if Harbor's agent timeout fires
        try:
            reason = await self._run()
        except Exception as e:
            reason, error = "agent_error", f"{type(e).__name__}: {e}"
        finally:
            result = LoopResult(
                reason, self.step, self.controller_calls, self.finish_summary, error
            )
            self.rec.event("termination", step=self.step, **asdict(result))
            self.rec.close({**asdict(result), "wall_s": self.clock() - self.started})
        return result

    def _breach(self, check_steps: bool) -> str | None:
        totals = self.rec.totals()
        return limit_breach(
            self.limits,
            usd=totals["usd_total"],
            tokens=totals["tokens_main"],
            elapsed_s=self.clock() - self.started,
            steps=self.step if check_steps else None,
        )

    async def _run(self) -> str:
        while True:
            if reason := self._breach(check_steps=True):
                return reason
            self.step += 1
            message, usage = await self.llm.complete(self.messages, TOOLS)
            self.rec.add_usage("main", usage)
            self.rec.event("model_call", step=self.step, usage=usage.to_dict())
            self.messages.append(message)
            if reason := self._breach(check_steps=False):
                return reason
            if usage.finish_reason == "content_filter":
                return "model_refusal"

            tool_calls = message.get("tool_calls") or []
            finish_requested = not tool_calls
            summary = message.get("content") or ""
            for call in tool_calls:
                name, args, error = parse_tool_call(call)
                if error:
                    content = f"Error: {error}"
                elif name == "bash":
                    content = await self._bash(args)
                elif name == "finish":
                    finish_requested = True
                    summary = str(args.get("summary", ""))
                    content = "Finish request received."
                else:
                    content = f"Error: unknown tool '{name}'"
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": content}
                )

            if finish_requested:
                self.finish_summary = summary
                trigger = self.detector.finish_requested()
                action = await self._consult(trigger, "finish", summary)
                if action in (None, "finish"):
                    return "finished"
                self._guide(action)
                continue

            trigger = self.detector.fire(self.step, self.detector.candidates(self.step))
            if trigger:
                action = await self._consult(trigger, "progress")
                if action == "abandon":
                    return "controller_abandon"
                if action in GUIDANCE_FOR:
                    self._guide(action)

    async def _bash(self, args: dict) -> str:
        command = str(args.get("command", ""))
        timeout = clamp_timeout(args.get("timeout_s"))
        result = await self.exec_fn(command, timeout)
        head, tail = (
            self.arm.cfg.tool_output_truncation.head_chars,
            self.arm.cfg.tool_output_truncation.tail_chars,
        )
        action = Action(
            index=len(self.detector.actions),
            step=self.step,
            command=command,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            output=truncate(result.output, head, tail),
            output_hash=output_hash(result.output),
        )
        self.detector.observe(action)
        self.rec.event(
            "tool_call",
            step=self.step,
            command=command,
            timeout_s=timeout,
            exit_code=action.exit_code,
            timed_out=action.timed_out,
            output=action.output,
        )
        return format_action(action, timeout)

    async def _consult(self, trigger: str, question: str, summary: str | None = None) -> str | None:
        """Log the trigger state; ask the controller (if any) and apply guards. None = no advice."""
        state = build_state(
            instruction=self.instruction,
            trigger=trigger,
            step=self.step,
            limits=self.limits,
            usd_used=self.rec.totals()["usd_total"],
            detector=self.detector,
            controller_history=self.history,
            finish_summary=summary,
        )
        self.rec.trigger_state(step=self.step, trigger=trigger, question=question, state=state)

        asked = self.arm.cfg.controller.questions
        if self.controller is None or (asked is not None and question not in asked):
            self.rec.event("trigger", step=self.step, trigger=trigger, consulted=False)
            return None
        if self.controller_calls >= self.limits.max_controller_calls:
            self.rec.event("trigger", step=self.step, trigger=trigger, consulted=False, capped=True)
            return None

        self.controller_calls += 1
        decision = await decide_safely(self.controller, question, state)
        if decision.usage:
            self.rec.add_usage("controller", decision.usage)
        ctx = GuardContext(
            cfg=self.arm.triggers,
            step=self.step,
            max_steps=self.limits.max_agent_steps,
            consecutive_failures=self.detector.consecutive_failures,
            t2_fires=self.detector.fire_counts["T2"],
            verify_count=self.verify_count,
        )
        action, guard = apply_guards(question, decision.choice, ctx)
        if action == "verify":
            self.verify_count += 1
        self.history.append({"step": self.step, "trigger": trigger, "choice": action})
        self.rec.decision(
            step=self.step,
            trigger=trigger,
            question=question,
            controller=self.controller.name,
            choice=decision.choice,
            applied=action,
            guard=guard,
            probabilities=decision.probabilities,
            confidence=decision.confidence,
            latency_s=decision.latency_s,
            cost_usd=decision.usage.cost_usd if decision.usage else 0.0,
            served_model=decision.usage.served_model if decision.usage else None,
            fallback=decision.fallback,
            error=decision.error,
        )
        return action

    def _guide(self, action: str) -> None:
        template = GUIDANCE_FOR[action]
        self.messages.append(
            {"role": "user", "content": f"{GUIDANCE_PREFIX}\n{self.arm.guidance[template]}"}
        )
        self.rec.event("guidance", step=self.step, template=template)
