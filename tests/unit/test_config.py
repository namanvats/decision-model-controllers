import pytest

from agentbench.config import (
    PROJECT_ROOT,
    ConfigError,
    load_arm,
    load_tasks,
    task_checksum,
    validate_experiment,
)
from tests.helpers import write_yaml

ARM = (PROJECT_ROOT / "agents/sonnet-native.yaml").read_text()


def experiment(tmp_path, arms, tasks_path, frozen=False, phase="dev"):
    return write_yaml(
        tmp_path / "exp.yaml",
        {
            "schema_version": 2,
            "id": "t",
            "phase": phase,
            "tasks": str(tasks_path),
            "arms": [str(a) for a in arms],
            "attempts": 1,
            "randomize_arm_order": True,
            "seed": 1,
            "limits_ref": "config/limits.yaml#smoke",
            "budget": {"hard_cap_usd": 1, "soft_warning_usd": 1},
            "require_frozen_tasks": frozen,
        },
    )


def tasks_file(tmp_path, **extra):
    return write_yaml(tmp_path / "tasks.yaml", {"tasks": [{"name": "tb/a"}], **extra})


def arm_file(tmp_path, name, text):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "arm", ["sonnet-native", "sonnet-rules", "sonnet-smallllm", "sonnet-jev", "_mock"]
)
def test_real_arms_load(arm):
    resolved = load_arm(f"agents/{arm}.yaml", "config/limits.yaml#smoke")
    assert set(resolved.guidance) == {"I", "R", "V"}


def test_placeholder_rejected():
    with pytest.raises(ConfigError, match="CONFIGURE_ME"):
        load_arm("agents/sonnet-native.yaml")


def test_unknown_field_rejected(tmp_path):
    path = arm_file(tmp_path, "x", ARM + "surprise: 1\n")
    with pytest.raises(ConfigError, match="surprise"):
        load_arm(path, "config/limits.yaml#smoke")


def test_unknown_model_alias(tmp_path):
    path = arm_file(tmp_path, "x", ARM.replace("model_ref: sonnet_main", "model_ref: nope"))
    with pytest.raises(ConfigError, match="unknown model alias"):
        load_arm(path, "config/limits.yaml#smoke")


def test_paired_arms_must_match(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBENCH_PHASE", "dev")
    a = arm_file(tmp_path, "a", ARM)
    b = arm_file(
        tmp_path,
        "b",
        ARM.replace("id: sonnet-native", "id: other").replace("head_chars: 4000", "head_chars: 10"),
    )
    _, _, _, errors = validate_experiment(experiment(tmp_path, [a, b], tasks_file(tmp_path)))
    assert any("differ beyond controller" in e for e in errors)


def test_controller_difference_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBENCH_PHASE", "dev")
    a = arm_file(tmp_path, "a", ARM)
    b = arm_file(
        tmp_path,
        "b",
        ARM.replace("id: sonnet-native", "id: r").replace("type: none", "type: rules"),
    )
    _, arms, _, errors = validate_experiment(experiment(tmp_path, [a, b], tasks_file(tmp_path)))
    assert errors == [] and len(arms) == 2


def test_frozen_tasks_required(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBENCH_PHASE", "dev")
    a = arm_file(tmp_path, "a", ARM)
    tasks = tasks_file(tmp_path)
    _, _, _, errors = validate_experiment(experiment(tmp_path, [a], tasks, frozen=True))
    assert any("not frozen" in e for e in errors)

    write_yaml(tasks, {"frozen": True, "checksum": "bad", "tasks": [{"name": "tb/a"}]})
    _, _, _, errors = validate_experiment(experiment(tmp_path, [a], tasks, frozen=True))
    assert any("checksum mismatch" in e for e in errors)

    good = task_checksum(load_tasks(tasks))
    write_yaml(tasks, {"frozen": True, "checksum": good, "tasks": [{"name": "tb/a"}]})
    _, _, _, errors = validate_experiment(experiment(tmp_path, [a], tasks, frozen=True))
    assert errors == []


def test_phase_must_match_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBENCH_PHASE", "smoke")
    a = arm_file(tmp_path, "a", ARM)
    _, _, _, errors = validate_experiment(experiment(tmp_path, [a], tasks_file(tmp_path)))
    assert any("AGENTBENCH_PHASE" in e for e in errors)


@pytest.mark.parametrize("slug", ["~typesafe/jev-latest", "anthropic/claude-sonnet-latest"])
def test_version_aliases_rejected(slug):
    from agentbench.config import ModelCfg, _pinned

    with pytest.raises(ConfigError, match="version aliases"):
        _pinned(ModelCfg(slug=slug))
