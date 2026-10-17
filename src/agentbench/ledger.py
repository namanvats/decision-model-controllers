import json
from pathlib import Path

from agentbench.config import Budget


def trial_manifests(exp_dir: Path) -> list[Path]:
    """Our manifests only: <task>/<arm>/<attempt>/ and archived _infra/<trial>/ (not Harbor's)."""
    return sorted([*exp_dir.glob("*/*/*/manifest.json"), *exp_dir.glob("_infra/*/manifest.json")])


def trial_spend(trial_dir: Path) -> float:
    """Actual spend from usage.json; the trial cap if a started trial left no usage record."""
    usage = trial_dir / "usage.json"
    if usage.exists():
        return json.loads(usage.read_text())["totals"]["usd_total"]
    manifest = trial_dir / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["trial_cap_usd"]
    return 0.0


class Ledger:
    """Admission control: spent + in-flight + next trial cap must stay under the hard cap."""

    def __init__(self, exp_dir: Path, budget: Budget):
        self.budget = budget
        self.spent = sum(trial_spend(m.parent) for m in trial_manifests(exp_dir))
        self.in_flight = 0.0
        self.warned = False

    def admit(self, cap: float) -> bool:
        if self.spent + self.in_flight + cap > self.budget.hard_cap_usd:
            return False
        self.in_flight += cap
        return True

    def settle(self, cap: float, actual: float) -> None:
        self.in_flight -= cap
        self.spent += actual

    def crossed_soft_warning(self) -> bool:
        if not self.warned and self.spent >= self.budget.soft_warning_usd:
            self.warned = True
            return True
        return False
