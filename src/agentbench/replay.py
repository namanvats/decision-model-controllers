"""Offline controller study (Phase 2, metric M1): replay logged native trigger states."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from agentbench.config import load_arm
from agentbench.controllers import decide_safely, make_controller
from agentbench.openrouter import OpenRouter
from agentbench.runner import completed_trial_dirs, read_json
from agentbench.telemetry import read_jsonl

REPLAY_ARMS = {
    "jev": "agents/sonnet-jev.yaml",
    "small_llm": "agents/sonnet-smallllm.yaml",
    "rules": "agents/sonnet-rules.yaml",
}
SCORED_CHOICE = {"finish": "finish", "progress": "abandon"}


def load_states(experiment_ids: list[str], arm_id: str) -> list[dict]:
    """Every logged trigger state from native runs, labelled with the trial's grader outcome."""
    rows = []
    for experiment_id in experiment_ids:
        for trial_dir in completed_trial_dirs(experiment_id):
            outcome = read_json(trial_dir / "outcome.json")
            if outcome["arm"] != arm_id or outcome["infra_error"]:
                continue
            for i, record in enumerate(read_jsonl(trial_dir / "trigger_states.jsonl")):
                rows.append(
                    {
                        "key": f"{outcome['trial_id']}#{i}",
                        "task": outcome["task"],
                        "trigger": record["trigger"],
                        "question": record["question"],
                        "state": record["state"],
                        "passed": outcome["passed"],
                    }
                )
    return rows


async def replay(
    rows: list[dict], controller_name: str, cache_path: Path, concurrency: int = 4, log=print
) -> list[dict]:
    """Ask a controller about every state; results are cached so reruns only fill gaps."""
    cache = {r["key"]: r for r in read_jsonl(cache_path)}
    todo = [r for r in rows if r["key"] not in cache]
    log(f"{controller_name}: {len(rows)} states, {len(todo)} to query")
    if todo:
        arm = load_arm(REPLAY_ARMS[controller_name], "config/limits.yaml#calibration")
        client = OpenRouter() if controller_name != "rules" else None
        controller = make_controller(arm, client, trial_id="replay")
        semaphore = asyncio.Semaphore(concurrency)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        async def ask(row: dict) -> None:
            async with semaphore:
                d = await decide_safely(controller, row["question"], row["state"])
            record = {
                "key": row["key"],
                "controller": controller_name,
                "choice": d.choice,
                "probabilities": d.probabilities,
                "confidence": d.confidence,
                "fallback": d.fallback,
                "error": d.error,
                "latency_s": d.latency_s,
                "cost_usd": d.usage.cost_usd if d.usage else 0.0,
            }
            cache[row["key"]] = record
            with cache_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

        try:
            await asyncio.gather(*(ask(r) for r in todo))
        finally:
            if client:
                await client.aclose()
    return [cache[r["key"]] for r in rows]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((scores * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(scores[mask].mean() - labels[mask].mean())
    return float(total / len(scores))


def score_of(record: dict, question: str) -> float:
    target = SCORED_CHOICE[question]
    if record.get("probabilities"):
        return float(record["probabilities"].get(target, 0.0))
    return 1.0 if record["choice"] == target else 0.0


def decision_quality(rows: list[dict], results: dict[str, list[dict]]) -> pd.DataFrame:
    """Q-finish: P(finish) vs passed. Q-progress: P(abandon) vs failed."""
    table = []
    for question in SCORED_CHOICE:
        idx = [i for i, r in enumerate(rows) if r["question"] == question]
        if not idx:
            continue
        passed = np.array([rows[i]["passed"] for i in idx], dtype=float)
        labels = passed if question == "finish" else 1 - passed
        base = labels.mean()
        table.append(
            {
                "question": question,
                "controller": "base_rate",
                "n": len(idx),
                "positive_rate": base,
                "auroc": 0.5,
                "ece": 0.0,
                "accuracy": max(base, 1 - base),
                "fallback_rate": 0.0,
                "choice_counts": "",
            }
        )
        for name, records in results.items():
            recs = [records[i] for i in idx]
            scores = np.array([score_of(r, question) for r in recs])
            predicted = np.array(
                [r["choice"] == SCORED_CHOICE[question] for r in recs], dtype=float
            )
            counts = pd.Series([r["choice"] for r in recs]).value_counts().to_dict()
            table.append(
                {
                    "question": question,
                    "controller": name,
                    "n": len(idx),
                    "positive_rate": base,
                    "auroc": auroc(scores, labels),
                    "ece": ece(scores, labels),
                    "accuracy": float((predicted == labels).mean()),
                    "fallback_rate": float(np.mean([r["fallback"] for r in recs])),
                    "choice_counts": json.dumps(counts),
                }
            )
    return pd.DataFrame(table)
