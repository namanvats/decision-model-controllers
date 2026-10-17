from collections import Counter

from agentbench.config import Budget, ExperimentCfg, TaskEntry, TaskList
from agentbench.ledger import Ledger
from agentbench.runner import (
    Trial,
    archive,
    build_outcome,
    harbor_command,
    plan_trials,
    trial_status,
)
from agentbench.telemetry import write_json
from tests.helpers import arm_with, write_trial

EXP = ExperimentCfg(
    schema_version=2,
    id="exp",
    phase="dev",
    tasks="x",
    arms=[],
    attempts=3,
    randomize_arm_order=True,
    seed=1,
    budget=Budget(hard_cap_usd=2, soft_warning_usd=1),
    require_frozen_tasks=False,
)
ARMS = [
    arm_with(p)
    for p in ("agents/sonnet-native.yaml", "agents/sonnet-rules.yaml", "agents/sonnet-jev.yaml")
]
TASKS = TaskList(tasks=[TaskEntry(name="tb/a"), TaskEntry(name="tb/b")])


def test_plan_is_deterministic_and_balanced():
    trials = plan_trials(EXP, ARMS, TASKS)
    assert trials == plan_trials(EXP, ARMS, TASKS)
    assert len(trials) == 2 * 3 * 3
    for i in range(0, len(trials), 3):
        block = trials[i : i + 3]
        assert len({(t.task, t.attempt) for t in block}) == 1
        assert {t.arm_id for t in block} == {a.id for a in ARMS}
    orders = Counter(tuple(t.arm_id for t in trials[i : i + 3]) for i in range(0, 18, 3))
    assert len(orders) > 1  # arm order actually varies across blocks


def test_trial_status_and_infra_retry_once(results_dir):
    trial = Trial("exp", "tb/a", "sonnet-native", 1, 0)
    assert trial_status(trial) == "pending"
    write_trial(
        results_dir, "exp", "tb/a", "sonnet-native", 1, passed=False, usd=0.1, infra_error=True
    )
    assert trial_status(trial) == "retry"
    archive(trial)
    write_trial(
        results_dir, "exp", "tb/a", "sonnet-native", 1, passed=False, usd=0.1, infra_error=True
    )
    assert trial_status(trial) == "gave_up"
    write_trial(results_dir, "exp", "tb/a", "sonnet-native", 1, passed=True, usd=0.1)
    assert trial_status(trial) == "done"


def test_interrupted_trial_is_rerun(results_dir):
    trial = Trial("exp", "tb/a", "sonnet-native", 1, 0)
    trial.dir().mkdir(parents=True)
    assert trial_status(trial) == "retry"


def test_harbor_command_shape():
    trial = Trial("exp", "terminal-bench/fix-git", "sonnet-native", 1, 0)
    cmd = harbor_command(trial, ARMS[0], EXP)
    assert cmd[1] == "run" and "-i" in cmd and "terminal-bench/fix-git" in cmd
    assert cmd[cmd.index("-a") + 1] == "agentbench.harbor_adapter:ControlledAgent"
    assert any(c.startswith("terminal-bench/terminal-bench-2-1@sha256:") for c in cmd)
    assert f"trial_id={trial.id}" in cmd and any(c.startswith("labels=") for c in cmd)


def harbor(reward=1.0, exc=None):
    t = {"started_at": "2026-10-11T00:00:00+00:00", "finished_at": "2026-10-11T00:01:40+00:00"}
    return {
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": exc,
        **t,
        "agent_execution": t,
    }


USAGE = {"termination_reason": "finished", "steps": 4, "totals": {"usd_total": 0.2}}


def test_outcomes():
    trial = Trial("exp", "tb/a", "sonnet-native", 1, 0)
    ok = build_outcome(trial, harbor(1.0), USAGE, 0)
    assert ok["passed"] and not ok["infra_error"] and ok["wall_s"]["total"] == 100
    fail = build_outcome(trial, harbor(0.0), USAGE, 0)
    assert not fail["passed"] and not fail["infra_error"]
    timeout = build_outcome(
        trial,
        harbor(0.0, {"exception_type": "AgentTimeoutError"}),
        {**USAGE, "termination_reason": "cancelled"},
        0,
    )
    assert timeout["termination_reason"] == "harbor_timeout" and not timeout["infra_error"]
    assert build_outcome(trial, None, None, 1)["infra_error"]
    assert build_outcome(trial, harbor(None), USAGE, 0)["infra_error"]
    assert build_outcome(trial, harbor(0.0), {**USAGE, "termination_reason": "agent_error"}, 0)[
        "infra_error"
    ]
    env_error = build_outcome(trial, harbor(0.0, {"exception_type": "RuntimeError"}), USAGE, 0)
    assert env_error["infra_error"]


def test_ledger_admission(results_dir):
    trial = write_trial(results_dir, "exp", "tb/a", "x", 1, passed=True, usd=0.9)
    harbor_manifest = trial / "harbor" / "job" / "a__xyz" / "artifacts" / "manifest.json"
    harbor_manifest.parent.mkdir(parents=True)
    harbor_manifest.write_text("[]")  # Harbor's own manifest must be ignored
    crashed = results_dir / "exp" / "b" / "x" / "1"
    crashed.mkdir(parents=True)
    write_json(crashed / "manifest.json", {"trial_cap_usd": 0.5})
    ledger = Ledger(results_dir / "exp", EXP.budget)
    assert ledger.spent == 1.4  # crashed trial without usage is charged its cap
    assert ledger.admit(0.5)
    assert not ledger.admit(0.5)  # 1.4 + 0.5 in flight + 0.5 > 2.0
    ledger.settle(0.5, 0.05)
    assert ledger.admit(0.5) and ledger.crossed_soft_warning()


FAKE_HARBOR = """
import json, sys, pathlib
args = sys.argv[1:]
out = pathlib.Path(args[args.index("-o") + 1])
trial_dir = pathlib.Path(next(a.split("=", 1)[1] for a in args if a.startswith("trial_dir=")))
(trial_dir / "usage.json").write_text(json.dumps(
    {"termination_reason": "finished", "steps": 3, "totals": {"usd_total": 0.25}}))
t = {"started_at": "2026-10-11T00:00:00+00:00", "finished_at": "2026-10-11T00:00:10+00:00"}
(out / "job" / "fix-git__abc").mkdir(parents=True)
(out / "job" / "fix-git__abc" / "result.json").write_text(json.dumps(
    {"verifier_result": {"rewards": {"reward": 1.0}}, "exception_info": None, **t}))
"""


async def test_run_experiment_end_to_end(monkeypatch, tmp_path, results_dir):
    import sys

    from agentbench import runner

    script = tmp_path / "fake_harbor.py"
    script.write_text(FAKE_HARBOR)
    real = runner.harbor_command

    async def no_prefetch():
        return None

    monkeypatch.setattr(runner, "prefetch_dataset", no_prefetch)
    monkeypatch.setenv("AGENTBENCH_CONCURRENCY", "2")
    monkeypatch.setattr(
        runner, "harbor_command", lambda *a: [sys.executable, str(script), *real(*a)[2:]]
    )
    exp = EXP.model_copy(
        update={"attempts": 1, "budget": Budget(hard_cap_usd=10, soft_warning_usd=5)}
    )
    arms = [arm_with(), ARMS[0]]
    trials = plan_trials(exp, arms, TaskList(tasks=[TaskEntry(name="terminal-bench/fix-git")]))
    logs = []
    await runner.run_experiment(exp, arms, trials, log=logs.append)
    outcomes = [runner.read_json(t.dir() / "outcome.json") for t in trials]
    assert all(o["passed"] and not o["infra_error"] for o in outcomes)
    manifests = [runner.read_json(t.dir() / "manifest.json") for t in trials]
    assert all(m["concurrency"] == 2 for m in manifests)
    assert not (results_dir / "exp" / ".runner.lock").exists()

    logs.clear()
    await runner.run_experiment(exp, arms, trials, log=logs.append)
    assert any("0 to run" in line for line in logs)  # resume skips completed trials


def test_known_trials_ignores_harbor_manifests(results_dir):
    from agentbench.cleanup import known_trials

    trial = write_trial(results_dir, "exp", "tb/a", "x", 1, passed=True, usd=0.1)
    nested = trial / "harbor" / "job" / "a__xyz" / "artifacts" / "manifest.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("[]")
    assert list(known_trials().values()) == ["exp--a--x--a1"]
