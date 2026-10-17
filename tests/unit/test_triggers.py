from agentbench.config import TriggerCfg, read_yaml
from agentbench.tools import Action, output_hash
from agentbench.triggers import TriggerDetector

CFG = TriggerCfg(**read_yaml("config/triggers.yaml"))


def act(i, command="ls", code=0, output=None):
    output = f"out{i}" if output is None else output
    return Action(i, i, command, code, False, output, output_hash(output))


def test_t1_after_two_failures():
    d = TriggerDetector(CFG)
    d.observe(act(1, "a", 1))
    assert "T1" not in d.candidates(1)
    d.observe(act(2, "b", 1))
    assert "T1" in d.candidates(2)
    d.observe(act(3, "c", 0))
    assert d.consecutive_failures == 0


def test_t2_repeated_command_and_output():
    d = TriggerDetector(CFG)
    d.observe(act(1, "make  build"))
    d.observe(act(2, "make build"))
    assert d.repeated_command() and "T2" in d.candidates(2)

    d = TriggerDetector(CFG)
    d.observe(act(1, "a", output="same"))
    d.observe(act(2, "b", output="same"))
    assert d.repeated_output()


def test_t2_ignores_empty_output():
    d = TriggerDetector(CFG)
    d.observe(act(1, "mkdir a", output=""))
    d.observe(act(2, "mkdir b", output=""))
    assert not d.repeated_output()


def test_t4_checkpoint_and_priority_and_cooldown():
    d = TriggerDetector(CFG)
    assert d.candidates(10) == ["T4"]
    assert d.fire(5, ["T4", "T1"]) == "T1"
    assert d.fire(7, ["T1"]) is None  # cooldown: within 3 steps
    assert d.fire(8, ["T1"]) is None
    assert d.fire(9, ["T1"]) == "T1"
    assert d.fire_counts["T1"] == 2


def test_one_trigger_per_step_falls_through_cooldown():
    d = TriggerDetector(CFG)
    d.fire(1, ["T1"])
    assert d.fire(2, ["T1", "T2"]) == "T2"


def test_verified_after_last_edit():
    d = TriggerDetector(CFG)
    assert not d.verified_after_last_edit
    d.observe(act(0, "pytest"))
    assert d.verified_after_last_edit
    d.observe(act(1, "echo x > f"))
    assert not d.verified_after_last_edit
    d.observe(act(2, "pytest -q"))
    assert d.verified_after_last_edit and d.verification_cmds == 2
