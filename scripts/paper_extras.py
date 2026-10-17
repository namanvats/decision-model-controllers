"""Robustness and cost-decomposition tables for the paper (exploratory, from existing eval logs).

    uv run python scripts/paper_extras.py experiments/eval.yaml

Writes to results/_analysis/<experiment-id>/extras/.
"""

import json
import sys

import matplotlib
import numpy as np
import pandas as pd

from agentbench.config import load_experiment
from agentbench.runner import completed_trial_dirs, results_root
from agentbench.telemetry import read_jsonl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TREATMENT, CONTROL = "sonnet-jev", "sonnet-native"
INK, MUTED, GRID, MARK = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def cost_after_verify(experiment_id: str) -> pd.DataFrame:
    """Per arm: main-model USD per attempt, split at the first verify nudge."""
    rows = []
    for trial_dir in completed_trial_dirs(experiment_id):
        outcome = json.loads((trial_dir / "outcome.json").read_text())
        if outcome["infra_error"]:
            continue
        events = read_jsonl(trial_dir / "events.jsonl")
        verify_steps = [
            e["step"] for e in events if e["kind"] == "guidance" and e["template"] == "V"
        ]
        first = min(verify_steps) if verify_steps else None
        calls = [e for e in events if e["kind"] == "model_call"]
        total = sum(c["usage"]["cost_usd"] for c in calls)
        after = sum(
            c["usage"]["cost_usd"] for c in calls if first is not None and c["step"] > first
        )
        rows.append(
            {
                "arm": outcome["arm"],
                "usd_main": total,
                "usd_after_first_verify": after,
                "verify_nudges": len(verify_steps),
                "steps": outcome["steps"],
            }
        )
    df = pd.DataFrame(rows).groupby("arm").mean().reset_index()
    df["usd_before_first_verify"] = df.usd_main - df.usd_after_first_verify
    return df


def forest(lr: pd.Series, path) -> None:
    lr = lr.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(lr) + 1.2), dpi=150)
    ax.scatter(np.exp(lr.values), range(len(lr)), color=MARK, s=22, zorder=3)
    ax.axvline(1.0, color=MUTED, linewidth=1)
    ax.axvline(np.exp(lr.mean()), color=MARK, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(lr)), [t.rsplit("/", 1)[-1] for t in lr.index], fontsize=7, color=INK)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks, [f"×{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(f"USD per attempt, {TREATMENT} / {CONTROL} (log scale)", color=MUTED)
    ax.set_title("Per-task cost ratio (dashed: geometric mean)", fontsize=10, color=INK, loc="left")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(experiment_path: str) -> None:
    exp_id = load_experiment(experiment_path).id
    base = results_root() / "_analysis" / exp_id
    out = base / "extras"
    out.mkdir(parents=True, exist_ok=True)

    per_task = pd.read_csv(base / "per_task.csv")
    pass_rate = per_task.pivot(index="task", columns="arm", values="pass_rate")
    usd = per_task.pivot(index="task", columns="arm", values="usd_per_attempt")
    lr = np.log(usd[TREATMENT]) - np.log(usd[CONTROL])
    diff = pass_rate[TREATMENT] - pass_rate[CONTROL]

    loo = pd.DataFrame(
        {"dropped_task": lr.index, "gm_cost_ratio": [np.exp(lr.drop(t).mean()) for t in lr.index]}
    )
    tasks = pd.DataFrame(
        {
            "cost_ratio": np.exp(lr),
            "pass_diff": diff,
            "stratum": per_task.groupby("task").stratum.first(),
        }
    )
    decomposition = cost_after_verify(exp_id)

    loo.to_csv(out / "leave_one_task_out.csv", index=False)
    tasks.sort_values("cost_ratio").to_csv(out / "per_task_jev_vs_native.csv")
    decomposition.to_csv(out / "cost_after_verify.csv", index=False)
    forest(lr, out / "cost_ratio_forest.png")

    print(
        f"tasks: {len(lr)}; Jev costlier on {(lr > 0).sum()}; "
        f"GM cost ratio x{np.exp(lr.mean()):.2f}; "
        f"leave-one-out range x{loo.gm_cost_ratio.min():.2f}-x{loo.gm_cost_ratio.max():.2f}"
    )
    print(f"tasks where pass rate differs (Jev vs native): {(diff != 0).sum()}/{len(diff)}")
    print(decomposition.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/eval.yaml")
