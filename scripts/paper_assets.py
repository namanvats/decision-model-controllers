"""Figures and LaTeX tables for paper/, generated from the eval analysis outputs.

uv run python scripts/paper_assets.py
"""

import matplotlib
import numpy as np
import pandas as pd

from agentbench.config import PROJECT_ROOT
from agentbench.runner import results_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANALYSIS = results_root() / "_analysis" / "tb21-eval-v1"
REPLAY = results_root() / "_replay" / "tb21-eval-v1"
OUT = PROJECT_ROOT / "paper" / "figures"
TABLES = PROJECT_ROOT / "paper" / "tables"
ARMS = ["sonnet-native", "sonnet-rules", "sonnet-smallllm", "sonnet-jev", "sonnet-placebo"]
INK, MUTED, GRID, MARK = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
LABELS = {
    "sonnet-native": "native",
    "sonnet-rules": "rules",
    "sonnet-smallllm": "small LLM",
    "sonnet-jev": "Jev",
    "sonnet-placebo": "placebo",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def effects(per_task: pd.DataFrame, seed: int = 20261011) -> None:
    """Paired (by task) effect of each controller vs native: cost ratio and pass-rate difference."""
    rng = np.random.default_rng(seed)
    pass_rate = per_task.pivot(index="task", columns="arm", values="pass_rate")
    log_usd = np.log(per_task.pivot(index="task", columns="arm", values="usd_per_attempt"))
    idx = rng.integers(0, len(pass_rate), (10_000, len(pass_rate)))
    offsets = {"rules": (6, 5), "small LLM": (6, 5), "Jev": (6, -11), "placebo": (-30, -12)}
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(1, color=MUTED, linewidth=0.8)
    for arm in ("sonnet-rules", "sonnet-smallllm", "sonnet-jev", "sonnet-placebo"):
        d = (pass_rate[arm] - pass_rate["sonnet-native"]).to_numpy() * 100
        lr = (log_usd[arm] - log_usd["sonnet-native"]).to_numpy()
        bd, blr = d[idx].mean(1), lr[idx].mean(1)
        x, y = np.exp(lr.mean()), d.mean()
        ax.errorbar(
            x,
            y,
            xerr=[[x - np.exp(np.percentile(blr, 2.5))], [np.exp(np.percentile(blr, 97.5)) - x]],
            yerr=[[y - np.percentile(bd, 2.5)], [np.percentile(bd, 97.5) - y]],
            fmt="o",
            color=MARK,
            ecolor=MARK,
            elinewidth=0.8,
            markersize=4,
        )
        label = LABELS[arm]
        ax.annotate(
            label, (x, y), xytext=offsets[label], textcoords="offset points", fontsize=7, color=INK
        )
    ax.set_xlabel("Cost ratio vs native (geometric mean)")
    ax.set_ylabel("Pass-rate difference vs native (pp)")
    style(ax)
    fig.savefig(OUT / "effects.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def forest(per_task: pd.DataFrame) -> None:
    usd = per_task.pivot(index="task", columns="arm", values="usd_per_attempt")
    ratio = np.exp(np.log(usd["sonnet-jev"]) - np.log(usd["sonnet-native"])).sort_values()
    fig, ax = plt.subplots(figsize=(3.1, 3.2))
    ax.scatter(ratio.values, range(len(ratio)), color=MARK, s=10, zorder=3)
    ax.axvline(1.0, color=MUTED, linewidth=0.8)
    ax.axvline(np.exp(np.log(ratio).mean()), color=MARK, linewidth=0.8, linestyle="--")
    ax.set_yticks(
        range(len(ratio)), [t.rsplit("/", 1)[-1] for t in ratio.index], fontsize=6, color=INK
    )
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Cost ratio, Jev / native (log scale)")
    style(ax)
    ax.grid(False, axis="y")
    fig.savefig(OUT / "forest.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def num(value: float) -> str:
    """Signed one-decimal number typeset by siunitx (proper minus sign, explicit plus)."""
    return f"\\num{{{value:+.1f}}}"


def write(name: str, spec: str, header: list[str], rows: list[str]) -> None:
    """A complete booktabs tabular, so main.tex only adds the float, caption and label."""
    body = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        *header,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (TABLES / f"{name}.tex").write_text("\n".join(body) + "\n")


def summary_table() -> None:
    summary = pd.read_csv(ANALYSIS / "summary.csv").set_index("arm")
    triggers = pd.read_csv(ANALYSIS / "m2_triggers.csv").groupby("arm").n_triggers.sum()
    verify = pd.read_csv(ANALYSIS / "extras" / "cost_after_verify.csv").set_index("arm")
    rows = []
    for arm in ARMS:
        s, n = summary.loc[arm], summary.loc[arm, "n_attempts"]
        decisions = "--" if arm == "sonnet-native" else f"{triggers[arm] / n:.2f}"
        nudges = "--" if arm == "sonnet-native" else f"{verify.loc[arm, 'verify_nudges']:.2f}"
        rows.append(
            f"{LABELS[arm]} & {s.pass_rate * 100:.1f} & {s.usd_per_attempt:.3f} & "
            f"{s.usd_per_solved_attempt:.3f} & {s.steps_per_attempt:.1f} & "
            f"{s.wall_s_per_attempt / 60:.1f} & {decisions} & {nudges} \\\\"
        )
    header = [
        "Arm & Pass (\\%) & USD/attempt & USD/solved & Steps & Wall (min) & Decisions & Verify \\\\"
    ]
    write("summary", "lrrrrrrr", header, rows)


def comparison_table() -> None:
    comp = pd.read_csv(ANALYSIS / "comparisons.csv")
    rows = []
    for _, c in comp.iterrows():
        p = c.p_cost if c.confirmatory else c.p_cost_holm
        p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append(
            f"{c.id} & {LABELS[c.treatment]} vs {LABELS[c.control]} & "
            f"{num(c.pass_diff * 100)} & [{num(c.pass_diff_ci_lo * 100)}, "
            f"{num(c.pass_diff_ci_hi * 100)}] & {c.cost_ratio_gm:.2f} & "
            f"[{c.cost_ratio_ci_lo:.2f}, {c.cost_ratio_ci_hi:.2f}] & {p_text} \\\\"
        )
    header = ["& Comparison & $\\Delta$ pass & 95\\% CI & Cost ratio & 95\\% CI & $p$ \\\\"]
    write("comparisons", "llrcrcr", header, rows)


def offline_table() -> None:
    quality = pd.read_csv(REPLAY / "decision_quality.csv")
    names = {"base_rate": "base rate", "rules": "rules", "small_llm": "small LLM", "jev": "Jev"}
    rows = []
    for question, label in (("finish", "Q-finish"), ("progress", "Q-progress")):
        q = quality[quality.question == question]
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}} ($n={q.n.iloc[0]}$ states)}} \\\\")
        for _, r in q.iterrows():
            cells = f"{r.auroc:.2f} & {r.ece:.2f} & {r.accuracy:.2f}"
            rows.append(f"\\quad {names[r.controller]} & {cells} \\\\")
    header = ["Controller & AUROC & ECE & Accuracy \\\\"]
    write("offline", "lrrr", header, rows)


def per_task_table() -> None:
    per_task = pd.read_csv(ANALYSIS / "per_task.csv")
    pass_rate = per_task.pivot(index="task", columns="arm", values="pass_rate")
    usd = per_task.pivot(index="task", columns="arm", values="usd_per_attempt")
    stratum = per_task.groupby("task").stratum.first()
    order = {"easy": 0, "mid": 1, "hard": 2}
    tasks = sorted(pass_rate.index, key=lambda t: (order[stratum[t]], t))
    rows = []
    for t in tasks:
        cells = " & ".join(f"{pass_rate.loc[t, a] * 3:.0f}" for a in ARMS)
        ratio = usd.loc[t, "sonnet-jev"] / usd.loc[t, "sonnet-native"]
        name = t.rsplit("/", 1)[-1]
        rows.append(
            f"\\texttt{{{name}}} & {stratum[t]} & {cells} & "
            f"{usd.loc[t, 'sonnet-native']:.3f} & {ratio:.2f} \\\\"
        )
    header = [
        "& & \\multicolumn{5}{c}{Passes out of 3} & USD/attempt & Cost ratio \\\\",
        "\\cmidrule(lr){3-7}",
        "Task & Stratum & native & rules & small & Jev & placebo & native & Jev/native \\\\",
    ]
    write("per_task", "llrrrrrrr", header, rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    per_task = pd.read_csv(ANALYSIS / "per_task.csv")
    effects(per_task)
    forest(per_task)
    summary_table()
    comparison_table()
    offline_table()
    per_task_table()
    print(f"wrote {OUT} and {TABLES}")


if __name__ == "__main__":
    main()
