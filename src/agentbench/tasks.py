import math
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from agentbench.config import (
    PlaceboDist,
    TaskEntry,
    TaskList,
    load_tasks,
    read_yaml,
    resolve,
    task_checksum,
)
from agentbench.controllers import OPTIONS
from agentbench.runner import completed_trial_dirs, read_json
from agentbench.telemetry import read_jsonl

STRATA = ("easy", "mid", "hard")


def write_tasks(path: str | Path, tasks: TaskList) -> None:
    resolve(path).write_text(yaml.safe_dump(tasks.model_dump(), sort_keys=False))


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n items across groups, bounded by group size."""
    n = min(n, sum(sizes.values()))
    total = sum(sizes.values())
    alloc = dict.fromkeys(sizes, 0)
    while sum(alloc.values()) < n:
        open_groups = [k for k in sizes if alloc[k] < sizes[k]]
        best = max(open_groups, key=lambda k: sizes[k] / total * n - alloc[k])
        alloc[best] += 1
    return alloc


def sample(catalog: str | Path, n: int, seed: int, source: str | Path | None = None) -> TaskList:
    """Seeded sample stratified by category from the non-excluded catalog (or a subset list)."""
    entries = [t for t in read_yaml(catalog)["tasks"] if not t.get("excluded_reason")]
    if source:
        keep = {t.name for t in load_tasks(source).tasks}
        entries = [t for t in entries if t["name"] in keep]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for entry in sorted(entries, key=lambda t: t["name"]):
        by_category[entry.get("category") or "uncategorized"].append(entry)
    rng = random.Random(seed)
    alloc = allocate({k: len(v) for k, v in sorted(by_category.items())}, n)
    chosen = [t for k, v in sorted(by_category.items()) for t in rng.sample(v, alloc[k])]
    return TaskList(tasks=[TaskEntry(name=t["name"], category=t.get("category")) for t in chosen])


def native_pass_counts(experiment_id: str, arm_id: str) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for trial_dir in completed_trial_dirs(experiment_id):
        outcome = read_json(trial_dir / "outcome.json")
        if outcome["arm"] != arm_id or outcome["infra_error"]:
            continue
        counts[outcome["task"]][0] += outcome["passed"]
        counts[outcome["task"]][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def stratum(passes: int, attempts: int) -> str:
    if passes == 0:
        return "hard"
    return "easy" if passes == attempts else "mid"


def split(
    calibration_tasks: TaskList,
    pass_counts: dict[str, tuple[int, int]],
    n_dev: int,
    n_heldout: int,
    hard_cap: float,
    seed: int,
) -> tuple[TaskList, TaskList, list[str]]:
    """Stratified dev / held-out split (experiment_plan.md §7.3–7.4)."""
    rng = random.Random(seed)
    pools: dict[str, list[TaskEntry]] = {s: [] for s in STRATA}
    missing = []
    for task in calibration_tasks.tasks:
        if task.name not in pass_counts:
            missing.append(task.name)
            continue
        s = stratum(*pass_counts[task.name])
        pools[s].append(TaskEntry(name=task.name, category=task.category, stratum=s))
    for pool in pools.values():
        rng.shuffle(pool)

    dev_alloc = allocate({s: len(p) for s, p in pools.items()}, n_dev)
    dev = [t for s in STRATA for t in pools[s][: dev_alloc[s]]]
    rest = {s: pools[s][dev_alloc[s] :] for s in STRATA}
    sizes = {s: len(p) for s, p in rest.items()}
    sizes["hard"] = min(sizes["hard"], math.floor(hard_cap * n_heldout))
    held_alloc = allocate(sizes, n_heldout)
    heldout = [t for s in STRATA for t in rest[s][: held_alloc[s]]]
    return TaskList(tasks=dev), TaskList(tasks=heldout), missing


def freeze(path: str | Path) -> str:
    tasks = load_tasks(path)
    frozen = TaskList(frozen=True, checksum=task_checksum(tasks), tasks=tasks.tasks)
    write_tasks(path, frozen)
    return frozen.checksum


def suggest_limits(experiment_id: str, arm_id: str, percentile: float = 90) -> dict:
    """Caps = p90 of steps and USD among passing runs, rounded up (experiment_plan.md §6)."""
    rows = []
    for trial_dir in completed_trial_dirs(experiment_id):
        outcome = read_json(trial_dir / "outcome.json")
        if outcome["arm"] == arm_id and not outcome["infra_error"]:
            rows.append(outcome)
    passing = [r for r in rows if r["passed"]]
    if not passing:
        raise ValueError(f"no passing {arm_id} runs in {experiment_id}")
    steps = float(np.percentile([r["steps"] for r in passing], percentile))
    usd = float(np.percentile([r["usd_total"] for r in passing], percentile))
    max_steps = int(math.ceil(steps / 5) * 5)
    max_usd = math.ceil(usd * 4) / 4
    return {
        "n_trials": len(rows),
        "n_passing": len(passing),
        f"p{percentile:g}_steps": steps,
        f"p{percentile:g}_usd": usd,
        "max_agent_steps": max_steps,
        "max_trial_usd": max_usd,
        "failing_runs_over_caps": sum(
            r["steps"] > max_steps or r["usd_total"] > max_usd for r in rows if not r["passed"]
        ),
    }


def placebo_distribution(experiment_ids: list[str], arm_id: str, seed: int) -> PlaceboDist:
    """Jev's raw per-question choice frequencies on dev runs (pre-guard, no fallbacks)."""
    counts: dict[str, Counter] = {q: Counter() for q in OPTIONS}
    for experiment_id in experiment_ids:
        for trial_dir in completed_trial_dirs(experiment_id):
            if read_json(trial_dir / "outcome.json")["arm"] != arm_id:
                continue
            for d in read_jsonl(trial_dir / "decision_events.jsonl"):
                if not d["fallback"]:
                    counts[d["question"]][d["choice"]] += 1
    freqs = {}
    for question, options in OPTIONS.items():
        total = sum(counts[question].values())
        if not total:
            raise ValueError(f"no {question} decisions from {arm_id} in {experiment_ids}")
        freqs[question] = {o: round(counts[question][o] / total, 4) for o in options}
    n = {q: sum(c.values()) for q, c in counts.items()}
    return PlaceboDist(
        schema_version=1,
        seed=seed,
        source=f"{arm_id} in {','.join(experiment_ids)} on {date.today()} (n={n})",
        **freqs,
    )
