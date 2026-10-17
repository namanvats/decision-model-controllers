import json

from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from agentbench.harbor_adapter import ControlledAgent


class Env:
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        code = 1 if command == "false" else 0
        return ExecResult(stdout="" if code else f"ran {command}", stderr="", return_code=code)


async def test_mock_arm_runs_through_harbor_agent(tmp_path):
    trial_dir = tmp_path / "trial"
    agent = ControlledAgent(
        logs_dir=tmp_path / "logs",
        arm="agents/_mock.yaml",
        trial_dir=str(trial_dir),
        trial_id="exp--task--_mock--a1",
    )
    context = AgentContext()
    await agent.run("do it", Env(), context)
    assert context.metadata["termination_reason"] == "finished"
    assert context.n_input_tokens > 0 and context.cost_usd == 0
    usage = json.loads((trial_dir / "usage.json").read_text())
    # 5 scripted steps; the script never verifies, so rules answers `verify` twice before G3.
    assert usage["totals"]["main_calls"] == 7
    decisions = [
        json.loads(x) for x in (trial_dir / "decision_events.jsonl").read_text().splitlines()
    ]
    assert [d["trigger"] for d in decisions] == ["T1", "T2", "T3", "T3", "T3"]
    assert decisions[-1]["guard"] == "G3"
