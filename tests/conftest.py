from pathlib import Path

import pytest

from agentbench.telemetry import Recorder
from tests.helpers import FakeEnv


@pytest.fixture(autouse=True)
def results_dir(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "results"
    monkeypatch.setenv("AGENTBENCH_RESULTS_DIR", str(root))
    return root


@pytest.fixture
def fake_env() -> FakeEnv:
    return FakeEnv()


@pytest.fixture
def recorder(tmp_path) -> Recorder:
    return Recorder(tmp_path / "trial")
