import json
from collections.abc import Callable
from dataclasses import dataclass

from agentbench.runner import results_root

TRIAL_LABEL = "agentbench.trial"
EXPERIMENT_LABEL = "agentbench.experiment"


@dataclass
class SandboxRow:
    id: str
    state: str
    trial_id: str | None
    experiment: str | None
    labels: dict[str, str]


def known_trials() -> dict[str, str]:
    uids = {}
    for manifest in results_root().rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        uids[data["trial_uid"]] = data["trial_id"]
    return uids


def active_experiments() -> set[str]:
    return {lock.parent.name for lock in results_root().glob("*/.runner.lock")}


def audit(
    delete: bool = False, confirm: Callable[[str], bool] = lambda _: False, log=print
) -> None:
    """List Harbor-managed Daytona sandboxes; delete only those matched to our trials."""
    from daytona import Daytona, ListSandboxesQuery

    daytona = Daytona()
    sandboxes = list(daytona.list(ListSandboxesQuery(labels={"harbor.managed": "true"})))
    trials = known_trials()
    active = active_experiments()
    matched, unmatched = [], []
    for sb in sandboxes:
        labels = sb.labels or {}
        row = SandboxRow(
            id=sb.id,
            state=str(sb.state),
            trial_id=trials.get(labels.get(TRIAL_LABEL, "")),
            experiment=labels.get(EXPERIMENT_LABEL),
            labels=labels,
        )
        (matched if row.trial_id else unmatched).append((sb, row))

    log(
        f"{len(sandboxes)} Harbor-managed sandboxes: {len(matched)} from agentbench trials, "
        f"{len(unmatched)} unmatched (never deleted here)"
    )
    for _, row in matched:
        busy = " [experiment still running]" if row.experiment in active else ""
        log(f"  {row.id}  {row.state:<10} {row.trial_id}{busy}")
    for _, row in unmatched:
        log(f"  {row.id}  {row.state:<10} unmatched labels={row.labels}")

    deletable = [(sb, row) for sb, row in matched if row.experiment not in active]
    if not delete or not deletable:
        if deletable:
            log("Dry run. Re-run with --delete to remove the matched orphans.")
        return
    if not confirm(f"Delete {len(deletable)} orphaned sandboxes?"):
        return
    for sb, row in deletable:
        daytona.delete(sb)
        log(f"  deleted {row.id} ({row.trial_id})")
