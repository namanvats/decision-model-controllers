from agentbench.config import Limits, TriggerCfg, read_yaml
from agentbench.guards import GuardContext, abandon_allowed, apply_guards, limit_breach

CFG = TriggerCfg(**read_yaml("config/triggers.yaml"))
LIMITS = Limits(
    max_agent_steps=10,
    max_trial_usd=1.0,
    max_total_tokens=1000,
    max_wall_s=60,
    max_controller_calls=5,
)


def ctx(step=5, failures=0, t2=0, verify=0):
    return GuardContext(CFG, step, 10, failures, t2, verify)


def test_limit_breach():
    ok = {"usd": 0.1, "tokens": 10, "elapsed_s": 1}
    assert limit_breach(LIMITS, steps=3, **ok) is None
    assert limit_breach(LIMITS, steps=10, **ok) == "limit_steps"
    assert limit_breach(LIMITS, **{**ok, "usd": 1.0}) == "limit_usd"
    assert limit_breach(LIMITS, **{**ok, "tokens": 1000}) == "limit_tokens"
    assert limit_breach(LIMITS, **{**ok, "elapsed_s": 60}) == "limit_wall"


def test_g2_abandon():
    assert not abandon_allowed(ctx(step=3, failures=5))  # too early
    assert not abandon_allowed(ctx(step=5, failures=2))  # not stuck
    assert abandon_allowed(ctx(step=4, failures=3))
    assert abandon_allowed(ctx(step=4, t2=2))
    assert apply_guards("progress", "abandon", ctx(step=1)) == ("recover", "G2")
    assert apply_guards("progress", "abandon", ctx(step=5, failures=3)) == ("abandon", None)


def test_g3_verify_cap():
    assert apply_guards("finish", "verify", ctx(verify=1)) == ("verify", None)
    assert apply_guards("finish", "verify", ctx(verify=2)) == ("finish", "G3")
    assert apply_guards("finish", "finish", ctx()) == ("finish", None)
