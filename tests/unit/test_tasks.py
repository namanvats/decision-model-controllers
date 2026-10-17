import pytest

from agentbench.config import TaskEntry, TaskList, load_tasks, task_checksum
from agentbench.tasks import (
    allocate,
    freeze,
    placebo_distribution,
    sample,
    split,
    stratum,
    suggest_limits,
    write_tasks,
)
from tests.helpers import write_trial, write_yaml


def test_allocate_is_proportional_and_bounded():
    assert allocate({"a": 10, "b": 10}, 6) == {"a": 3, "b": 3}
    assert allocate({"a": 1, "b": 10}, 6) == {"a": 1, "b": 5}
    assert sum(allocate({"a": 2, "b": 1}, 10).values()) == 3


def test_sample_respects_exclusions_and_seed(tmp_path):
    catalog = write_yaml(
        tmp_path / "c.yaml",
        {
            "tasks": [
                {"name": f"tb/{c}{i}", "category": c, "excluded_reason": "gpu" if i == 0 else None}
                for c in "xyz"
                for i in range(5)
            ]
        },
    )
    a = sample(catalog, 6, seed=1)
    assert a == sample(catalog, 6, seed=1)
    assert len(a.tasks) == 6 and not any(t.name.endswith("0") for t in a.tasks)
    assert {t.category for t in a.tasks} == {"x", "y", "z"}


def test_strata_and_split_cap_hard():
    assert (stratum(0, 2), stratum(1, 2), stratum(2, 2)) == ("hard", "mid", "easy")
    names = [f"tb/t{i}" for i in range(30)]
    counts = {n: ((0, 2) if i < 15 else (1, 2) if i < 22 else (2, 2)) for i, n in enumerate(names)}
    dev, held, missing = split(
        TaskList(tasks=[TaskEntry(name=n) for n in names + ["tb/new"]]),
        counts,
        n_dev=8,
        n_heldout=20,
        hard_cap=0.2,
        seed=1,
    )
    assert missing == ["tb/new"] and len(dev.tasks) == 8
    assert sum(t.stratum == "hard" for t in held.tasks) <= 4
    assert not {t.name for t in dev.tasks} & {t.name for t in held.tasks}


def test_freeze_writes_checksum(tmp_path):
    path = tmp_path / "t.yaml"
    write_tasks(path, TaskList(tasks=[TaskEntry(name="tb/a", stratum="mid")]))
    checksum = freeze(path)
    tasks = load_tasks(path)
    assert tasks.frozen and tasks.checksum == checksum == task_checksum(tasks)


def test_suggest_limits_uses_passing_runs(results_dir):
    for i in range(10):
        write_trial(
            results_dir,
            "cal",
            f"tb/t{i}",
            "sonnet-native",
            1,
            passed=True,
            usd=0.1 * (i + 1),
            steps=5 * (i + 1),
        )
    write_trial(results_dir, "cal", "tb/x", "sonnet-native", 1, passed=False, usd=3.0, steps=100)
    result = suggest_limits("cal", "sonnet-native")
    assert result["n_passing"] == 10 and result["max_agent_steps"] == 50
    assert result["max_trial_usd"] == 1.0 and result["failing_runs_over_caps"] == 1


def test_placebo_distribution(results_dir):
    decisions = [
        {"question": "progress", "choice": "continue"},
        {"question": "progress", "choice": "recover"},
        {"question": "progress", "choice": "abandon", "fallback": True},
        {"question": "finish", "choice": "verify"},
    ]
    write_trial(
        results_dir, "dev", "tb/a", "sonnet-jev", 1, passed=True, usd=0.1, decisions=decisions
    )
    dist = placebo_distribution(["dev"], "sonnet-jev", seed=3)
    assert dist.progress == {"continue": 0.5, "investigate": 0.0, "recover": 0.5, "abandon": 0.0}
    assert dist.finish == {"finish": 0.0, "verify": 1.0}
    with pytest.raises(ValueError):
        placebo_distribution(["dev"], "sonnet-native", seed=3)
