import json

from agentbench.config import Limits
from agentbench.state import MAX_STATE_CHARS, build_state
from tests.unit.test_triggers import CFG, TriggerDetector, act

LIMITS = Limits(
    max_agent_steps=50,
    max_trial_usd=2.0,
    max_total_tokens=10**6,
    max_wall_s=600,
    max_controller_calls=5,
)


def make(trigger="T1", detector=None, **kw):
    return build_state(
        instruction="x" * 5000,
        trigger=trigger,
        step=3,
        limits=LIMITS,
        usd_used=0.123456,
        detector=detector or TriggerDetector(CFG),
        controller_history=[],
        **kw,
    )


def test_caps_and_fields():
    d = TriggerDetector(CFG)
    for i in range(7):
        d.observe(act(i, "c" * 1000, output="o" * 2000))
    state = make(detector=d)
    assert len(state["task_instruction"]) == 3000
    assert len(state["last_5_actions"]) == 5
    assert len(state["last_5_actions"][0]["command"]) == 300
    assert len(state["last_5_actions"][0]["output_tail"]) == 500
    assert "finish_summary" not in state and state["usd_used"] == 0.1235


def test_finish_summary_only_on_t3():
    state = make("T3", finish_summary="s" * 3000)
    assert len(state["finish_summary"]) == 1000


def test_total_size_cap():
    assert len(json.dumps(make())) <= MAX_STATE_CHARS
