import json

from agentbench.controllers import make_controller
from agentbench.controllers.base import Decision
from agentbench.loop import GUIDANCE_PREFIX, AgentLoop
from agentbench.openrouter import ScriptedModel
from agentbench.telemetry import read_jsonl
from agentbench.tools import TOOLS
from tests.helpers import arm_with


class Always:
    """Controller stub that always gives the same answers."""

    name, timeout_s = "stub", None

    def __init__(self, progress="continue", finish="finish"):
        self.answers = {"progress": progress, "finish": finish}

    async def decide(self, question, state):
        return Decision(question, self.answers[question])


async def run(script, controller=None, arm=None, env=None, recorder=None, **kw):
    loop = AgentLoop(arm or arm_with(), ScriptedModel(script, **kw), env, controller, recorder)
    return loop, await loop.run("do the task")


def usage(recorder):
    return json.loads((recorder.dir / "usage.json").read_text())


async def test_native_finishes_and_logs_trigger_states(fake_env, recorder):
    script = [{"bash": "false"}, {"bash": "false"}, {"finish": "done"}]
    loop, result = await run(script, env=fake_env, recorder=recorder)
    assert result.termination_reason == "finished" and result.steps == 3
    states = read_jsonl(recorder.dir / "trigger_states.jsonl")
    assert [s["trigger"] for s in states] == ["T1", "T3"]
    assert read_jsonl(recorder.dir / "decision_events.jsonl") == []
    assert not any(GUIDANCE_PREFIX in str(m.get("content")) for m in loop.messages)
    assert usage(recorder)["termination_reason"] == "finished"


async def test_text_only_reply_is_a_finish_request(fake_env, recorder):
    _, result = await run([{"text": "All done."}], env=fake_env, recorder=recorder)
    assert result.termination_reason == "finished" and result.finish_summary == "All done."


async def test_rules_verify_loop_capped_by_g3(fake_env, recorder):
    rules = make_controller(arm_with("agents/sonnet-rules.yaml"), None, "t")
    script = [{"bash": "echo x > f"}, {"finish": "a"}, {"finish": "b"}, {"finish": "c"}]
    loop, result = await run(script, rules, env=fake_env, recorder=recorder)
    assert result.termination_reason == "finished" and result.steps == 4
    decisions = read_jsonl(recorder.dir / "decision_events.jsonl")
    assert [d["applied"] for d in decisions] == ["verify", "verify", "finish"]
    assert decisions[-1]["guard"] == "G3"
    guidance = [m["content"] for m in loop.messages if GUIDANCE_PREFIX in str(m.get("content"))]
    assert len(guidance) == 2 and all(g.endswith(loop.arm.guidance["V"]) for g in guidance)


async def test_rules_finish_after_verification(fake_env, recorder):
    rules = make_controller(arm_with("agents/sonnet-rules.yaml"), None, "t")
    script = [{"bash": "echo x > f"}, {"bash": "pytest -q"}, {"finish": "ok"}]
    _, result = await run(script, rules, env=fake_env, recorder=recorder)
    assert result.termination_reason == "finished" and result.steps == 3


async def test_abandon_blocked_by_g2_then_allowed(fake_env, recorder):
    arm = arm_with(max_agent_steps=10)
    script = [{"bash": "false"}] * 10
    _, result = await run(
        script, Always(progress="abandon"), arm=arm, env=fake_env, recorder=recorder
    )
    decisions = read_jsonl(recorder.dir / "decision_events.jsonl")
    assert decisions[0]["applied"] == "recover" and decisions[0]["guard"] == "G2"
    assert decisions[-1]["applied"] == "abandon" and decisions[-1]["guard"] is None
    assert result.termination_reason == "controller_abandon" and result.steps >= 4


async def test_step_limit(fake_env, recorder):
    arm = arm_with(max_agent_steps=5)
    _, result = await run(
        [{"bash": f"echo {i}"} for i in range(20)], arm=arm, env=fake_env, recorder=recorder
    )
    assert result.termination_reason == "limit_steps" and result.steps == 5


async def test_usd_limit_counts_after_each_call(fake_env, recorder):
    arm = arm_with(max_trial_usd=1.0)
    _, result = await run(
        [{"bash": "ls"}] * 10, arm=arm, env=fake_env, recorder=recorder, cost_per_call=0.4
    )
    assert result.termination_reason == "limit_usd" and result.steps == 3
    assert fake_env.commands == ["ls", "ls"]  # the breaching call's tools never run


async def test_controller_call_cap(fake_env, recorder):
    arm = arm_with(max_controller_calls=1)
    script = [{"bash": "false"}] * 8 + [{"finish": "x"}]
    _, result = await run(script, Always(), arm=arm, env=fake_env, recorder=recorder)
    assert result.controller_calls == 1 and result.termination_reason == "finished"


async def test_exec_failure_is_agent_error(recorder):
    async def broken(command, timeout_s):
        raise RuntimeError("sandbox gone")

    _, result = await run([{"bash": "ls"}], env=broken, recorder=recorder)
    assert result.termination_reason == "agent_error" and "sandbox gone" in result.error
    assert usage(recorder)["termination_reason"] == "agent_error"


async def test_multiple_tool_calls_each_get_a_result(fake_env, recorder):
    script = [[{"bash": "ls"}, {"bash": "pwd"}], {"finish": "x"}]
    loop, _ = await run(script, env=fake_env, recorder=recorder)
    tool_msgs = [m for m in loop.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs][:2] == ["call_1_0", "call_1_1"]
    assert fake_env.commands == ["ls", "pwd"]


async def test_controller_only_appends_fixed_templates(fake_env, recorder):
    loop, _ = await run(
        [{"bash": "false"}] * 4 + [{"finish": "x"}],
        Always(progress="investigate"),
        env=fake_env,
        recorder=recorder,
    )
    templates = {f"{GUIDANCE_PREFIX}\n{t}" for t in loop.arm.guidance.values()}
    user_msgs = [m["content"] for m in loop.messages if m["role"] == "user"][1:]
    assert user_msgs and set(user_msgs) <= templates
    assert loop.messages[0]["content"] == loop.arm.system_prompt
    assert TOOLS[0]["function"]["name"] == "bash"


async def test_content_filter_is_a_refusal(fake_env, recorder):
    class Refuses:
        slug = "x"

        async def complete(self, messages, tools):
            from agentbench.openrouter import CallUsage

            return {"role": "assistant", "content": ""}, CallUsage(
                "x", finish_reason="content_filter"
            )

    loop = AgentLoop(arm_with(), Refuses(), fake_env, None, recorder)
    result = await loop.run("do the task")
    assert result.termination_reason == "model_refusal" and fake_env.commands == []
