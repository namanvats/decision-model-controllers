import numpy as np
import pandas as pd
import pytest

from agentbench.analysis import analyze, compare, holm
from tests.helpers import write_trial, write_yaml


def test_compare_recovers_known_effects():
    rows = []
    for i in range(40):
        rows.append({"task": f"t{i}", "arm": "native", "pass_rate": 0.6, "usd_per_attempt": 1.0})
        rows.append({"task": f"t{i}", "arm": "jev", "pass_rate": 0.5, "usd_per_attempt": 0.5})
    result = compare(pd.DataFrame(rows), "jev", "native", 2000, 0, -0.10)
    assert result["pass_diff"] == pytest.approx(-0.1)
    assert result["cost_ratio_gm"] == pytest.approx(0.5)
    assert result["n_tasks"] == 40


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_analyze_end_to_end(tmp_path, results_dir):
    rng = np.random.default_rng(0)
    tasks = [
        {"name": f"tb/t{i}", "category": "c", "stratum": ["easy", "mid", "hard"][i % 3]}
        for i in range(12)
    ]
    tasks_path = write_yaml(tmp_path / "tasks.yaml", {"tasks": tasks})
    exp_path = write_yaml(
        tmp_path / "exp.yaml",
        {
            "schema_version": 2,
            "id": "syn",
            "phase": "eval",
            "tasks": str(tasks_path),
            "arms": [],
            "attempts": 3,
            "randomize_arm_order": True,
            "seed": 1,
            "budget": {"hard_cap_usd": 1, "soft_warning_usd": 1},
            "require_frozen_tasks": False,
        },
    )
    for t in tasks:
        for a in range(1, 4):
            write_trial(
                results_dir,
                "syn",
                t["name"],
                "sonnet-native",
                a,
                passed=bool(rng.random() < 0.6),
                usd=1.0,
            )
            abandon = t["stratum"] == "hard" and a == 1
            write_trial(
                results_dir,
                "syn",
                t["name"],
                "sonnet-jev",
                a,
                passed=bool(rng.random() < 0.6) and not abandon,
                usd=0.5,
                termination="controller_abandon" if abandon else "finished",
                decisions=[{"choice": "abandon", "applied": "abandon"}]
                if abandon
                else [{"choice": "recover", "applied": "recover", "guard": "G2"}],
                triggers=[{"step": 1, "trigger": "T1"}],
            )
    write_trial(
        results_dir,
        "syn",
        "tb/t0",
        "sonnet-jev",
        9,
        passed=False,
        usd=0.0,
        infra_error=True,
        termination="agent_error",
    )

    out = analyze([str(exp_path)], tmp_path / "out", log=lambda *_: None)
    comparisons = pd.read_csv(out / "comparisons.csv").set_index("id")
    assert comparisons.loc["C1", "cost_ratio_gm"] == pytest.approx(0.5)
    summary = pd.read_csv(out / "summary.csv").set_index("arm")
    assert summary.loc["sonnet-jev", "n_attempts"] == 36  # infra trial excluded
    report = (out / "report.md").read_text()
    assert "C1 sonnet-jev vs sonnet-native" in report and "Exploratory" in report
    for name in ("pareto_usd_total", "pareto_tokens_total", "pareto_wall_total_s"):
        assert (out / "figures" / f"{name}.png").exists()
    assert (out / "m3_abandon_precision.csv").exists() and (out / "analysis.duckdb").exists()
