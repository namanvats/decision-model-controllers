import numpy as np
import pytest

from agentbench.replay import auroc, decision_quality, ece, load_states, replay
from tests.helpers import write_trial


def test_auroc_and_ece():
    labels = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))
    assert ece(np.array([0.0, 0.0, 1.0, 1.0]), labels.astype(float)) == 0.0
    assert ece(np.array([1.0, 1.0, 1.0, 1.0]), labels.astype(float)) == pytest.approx(0.5)


async def test_rules_replay_end_to_end(results_dir):
    def finish_state(verified):
        return {
            "trigger": "T3",
            "question": "finish",
            "step": 3,
            "state": {
                "trigger": "T3",
                "verified_after_last_edit": verified,
                "step": 3,
                "max_steps": 10,
                "consecutive_failures": 0,
            },
        }

    write_trial(
        results_dir,
        "cal",
        "tb/a",
        "sonnet-native",
        1,
        passed=True,
        usd=0.1,
        triggers=[finish_state(True)],
    )
    write_trial(
        results_dir,
        "cal",
        "tb/b",
        "sonnet-native",
        1,
        passed=False,
        usd=0.1,
        triggers=[finish_state(False)],
    )
    rows = load_states(["cal"], "sonnet-native")
    assert [r["passed"] for r in rows] == [True, False]
    cache = results_dir / "_replay" / "rules.jsonl"
    results = {"rules": await replay(rows, "rules", cache)}
    assert await replay(rows, "rules", cache) == results["rules"]  # served from cache
    table = decision_quality(rows, results).set_index("controller")
    assert table.loc["rules", "auroc"] == 1.0 and table.loc["rules", "accuracy"] == 1.0
    assert table.loc["base_rate", "auroc"] == 0.5
