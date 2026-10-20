"""Analysis of a completed experiment (experiment_plan.md §9–§10)."""

import hashlib
import json
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from agentbench.config import load_experiment, load_tasks, read_yaml
from agentbench.runner import completed_trial_dirs, read_json, results_root
from agentbench.telemetry import read_jsonl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, MARK = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def config_hash(manifest: dict) -> str:
    """Fingerprint of everything that must stay frozen within an arm."""
    provenance = manifest.get("provenance", {})
    frozen = {
        "arm": manifest.get("resolved_arm"),
        "benchmark": manifest.get("benchmark"),
        "uv_lock": provenance.get("uv_lock_sha256"),
        "harbor": provenance.get("harbor_version"),
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:12]


def load_trials(experiment_paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (trials, decisions, trigger_states) for all non-archived trials."""
    trials, decisions, triggers = [], [], []
    for path in experiment_paths:
        exp = load_experiment(path)
        task_meta = {t.name: t for t in load_tasks(exp.tasks).tasks}
        for trial_dir in completed_trial_dirs(exp.id):
            outcome = read_json(trial_dir / "outcome.json")
            totals = (read_json(trial_dir / "usage.json") or {}).get("totals", {})
            events = read_jsonl(trial_dir / "events.jsonl")
            tools = [e for e in events if e["kind"] == "tool_call"]
            hosts = {e["usage"].get("provider") or "?" for e in events if e["kind"] == "model_call"}
            guidance = [e["template"] for e in events if e["kind"] == "guidance"]
            meta = task_meta.get(outcome["task"])
            manifest = read_json(trial_dir / "manifest.json") or {}
            trials.append(
                {
                    **{
                        k: outcome[k]
                        for k in (
                            "trial_id",
                            "task",
                            "arm",
                            "attempt",
                            "reward",
                            "termination_reason",
                            "infra_error",
                            "steps",
                        )
                    },
                    "experiment": exp.id,
                    "passed": int(outcome["passed"]),
                    "stratum": meta.stratum if meta else None,
                    "category": meta.category if meta else None,
                    "usd_total": totals.get("usd_total", 0.0),
                    "usd_main": totals.get("usd_main", 0.0),
                    "usd_controller": totals.get("usd_controller", 0.0),
                    "tokens_main": totals.get("tokens_main", 0),
                    "tokens_controller": totals.get("tokens_controller", 0),
                    "main_calls": totals.get("main_calls", 0),
                    "controller_calls": totals.get("controller_calls", 0),
                    "latency_main_s": totals.get("latency_main_s", 0.0),
                    "latency_controller_s": totals.get("latency_controller_s", 0.0),
                    "wall_total_s": outcome["wall_s"]["total"],
                    "wall_agent_s": outcome["wall_s"]["agent"],
                    "wall_provisioning_s": outcome["wall_s"]["provisioning"],
                    "wall_verifier_s": outcome["wall_s"]["verifier"],
                    "config_hash": config_hash(manifest),
                    "main_hosts": ",".join(sorted(hosts)),
                    "tool_calls": len(tools),
                    "tool_failures": sum(t["exit_code"] != 0 or t["timed_out"] for t in tools),
                    "repeated_commands": len(tools) - len({t["command"] for t in tools}),
                    "guidance_I": guidance.count("I"),
                    "guidance_R": guidance.count("R"),
                    "guidance_V": guidance.count("V"),
                }
            )
            base = {
                "trial_id": outcome["trial_id"],
                "arm": outcome["arm"],
                "task": outcome["task"],
                "passed": int(outcome["passed"]),
                "infra_error": outcome["infra_error"],
            }
            for d in read_jsonl(trial_dir / "decision_events.jsonl"):
                decisions.append(
                    {
                        **base,
                        **{
                            k: d[k]
                            for k in (
                                "step",
                                "trigger",
                                "question",
                                "choice",
                                "applied",
                                "guard",
                                "fallback",
                                "latency_s",
                                "cost_usd",
                                "confidence",
                            )
                        },
                    }
                )
            for s in read_jsonl(trial_dir / "trigger_states.jsonl"):
                triggers.append({**base, "step": s["step"], "trigger": s["trigger"]})
    columns = ["trial_id", "arm", "task", "passed", "infra_error"]
    return (
        pd.DataFrame(trials),
        pd.DataFrame(
            decisions,
            columns=columns
            + [
                "step",
                "trigger",
                "question",
                "choice",
                "applied",
                "guard",
                "fallback",
                "latency_s",
                "cost_usd",
                "confidence",
            ],
        ),
        pd.DataFrame(triggers, columns=columns + ["step", "trigger"]),
    )


def bootstrap_p(boot: np.ndarray) -> float:
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


def compare(
    per_task: pd.DataFrame, treatment: str, control: str, resamples: int, seed: int, margin: float
) -> dict | None:
    """Paired-by-task bootstrap of pass-rate difference and geometric-mean cost ratio."""
    pass_rate = per_task.pivot(index="task", columns="arm", values="pass_rate")
    usd = per_task.pivot(index="task", columns="arm", values="usd_per_attempt")
    if treatment not in pass_rate or control not in pass_rate:
        return None
    both = pass_rate[[treatment, control]].dropna().index
    if len(both) == 0:
        return None
    d = (pass_rate.loc[both, treatment] - pass_rate.loc[both, control]).to_numpy()
    floor = 1e-6
    lr = (
        np.log(usd.loc[both, treatment].clip(lower=floor))
        - np.log(usd.loc[both, control].clip(lower=floor))
    ).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d), (resamples, len(d)))
    boot_d, boot_lr = d[idx].mean(1), lr[idx].mean(1)
    return {
        "n_tasks": len(d),
        "pass_treatment": pass_rate.loc[both, treatment].mean(),
        "pass_control": pass_rate.loc[both, control].mean(),
        "pass_diff": d.mean(),
        "pass_diff_ci_lo": np.percentile(boot_d, 2.5),
        "pass_diff_ci_hi": np.percentile(boot_d, 97.5),
        "pass_diff_lower_95_one_sided": np.percentile(boot_d, 5),
        "margin": margin,
        "cost_ratio_gm": float(np.exp(lr.mean())),
        "cost_ratio_ci_lo": float(np.exp(np.percentile(boot_lr, 2.5))),
        "cost_ratio_ci_hi": float(np.exp(np.percentile(boot_lr, 97.5))),
        "p_pass": bootstrap_p(boot_d),
        "p_cost": bootstrap_p(boot_lr),
    }


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def c1_verdict(row: pd.Series) -> str:
    ratio = (
        f"×{row.cost_ratio_gm:.2f}, 95% CI {row.cost_ratio_ci_lo:.2f}–{row.cost_ratio_ci_hi:.2f}"
    )
    if row.cost_ratio_ci_hi < 1:
        cost = f"cost lower ({ratio})"
    elif row.cost_ratio_ci_lo > 1:
        cost = f"cost higher ({ratio})"
    else:
        cost = f"no clear cost change ({ratio})"
    lower = row.pass_diff_lower_95_one_sided
    if lower >= row.margin:
        guard = (
            f"pass-rate drop smaller than the {row.margin:+.0%} margin "
            f"(one-sided 95% lower bound {lower:+.1%})"
        )
    elif row.pass_diff_ci_hi < row.margin:
        guard = f"pass rate drops by more than the margin (95% CI upper {row.pass_diff_ci_hi:+.1%})"
    else:
        guard = (
            f"guardrail inconclusive (lower bound {lower:+.1%} is below the margin); "
            "report 'no evidence of a large drop', not noninferiority"
        )
    return f"{cost}; {guard}"


def comparisons(per_task: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(cfg["comparisons"]):
        result = compare(
            per_task,
            c["treatment"],
            c["control"],
            cfg["bootstrap_resamples"],
            cfg["seed"] + i,
            cfg["noninferiority_margin"],
        )
        if result:
            rows.append(
                {
                    "id": c["id"],
                    "treatment": c["treatment"],
                    "control": c["control"],
                    "confirmatory": c["confirmatory"],
                    **result,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    secondary = ~df.confirmatory
    for col in ("p_pass", "p_cost"):
        df[f"{col}_holm"] = df[col]
        if secondary.any():
            df.loc[secondary, f"{col}_holm"] = holm(df.loc[secondary, col].tolist())
    return df


def pareto(trials: pd.DataFrame, x: str, label: str, path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    for arm, group in trials.groupby("arm"):
        per_task = group.groupby("task")[[x, "passed"]].mean()
        idx = rng.integers(0, len(per_task), (2000, len(per_task)))
        xs, ys = per_task[x].to_numpy(), per_task["passed"].to_numpy()
        bx, by = xs[idx].mean(1), ys[idx].mean(1)
        mx, my = xs.mean(), ys.mean()
        ax.errorbar(
            mx,
            my,
            xerr=[[mx - np.percentile(bx, 2.5)], [np.percentile(bx, 97.5) - mx]],
            yerr=[[my - np.percentile(by, 2.5)], [np.percentile(by, 97.5) - my]],
            fmt="o",
            color=MARK,
            ecolor=MARK,
            elinewidth=1,
            capsize=0,
            markersize=6,
        )
        ax.annotate(arm, (mx, my), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel(label, color=MUTED)
    ax.set_ylabel("Pass rate", color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        f"Pass rate vs {label.lower()} (mean ± 95% task-bootstrap CI)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_no data_\n"

    def fmt(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def native_arm(arm: str) -> str:
    return arm.split("-")[0] + "-native"


def mechanism_tables(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    m2 = con.sql("""
        SELECT t.arm, t.trigger, count(*) AS n_triggers, count(DISTINCT t.trial_id) AS n_trials,
               avg(t.passed) AS pass_rate_of_trials
        FROM trigger_states t WHERE NOT t.infra_error GROUP BY ALL ORDER BY ALL""").df()
    m2_choices = con.sql("""
        SELECT arm, trigger, choice, applied, count(*) AS n, avg(passed) AS pass_rate_of_trials
        FROM decisions WHERE NOT infra_error GROUP BY ALL ORDER BY ALL""").df()
    m3 = con.sql("""
        SELECT arm, passed, count(*) AS n_attempts, avg(usd_total) AS usd_per_attempt,
               sum(usd_total) AS usd_sum,
               sum((termination_reason = 'controller_abandon')::INT) AS n_abandon,
               avg(guidance_V) AS verify_per_attempt
        FROM trials WHERE NOT infra_error GROUP BY ALL ORDER BY ALL""").df()
    cells = con.sql("""
        SELECT task, arm, avg(passed) AS pass_rate,
               bool_or(termination_reason = 'controller_abandon') AS abandoned
        FROM trials WHERE NOT infra_error GROUP BY ALL""").df()
    native = cells[cells.arm.str.endswith("-native")].set_index(["task", "arm"]).pass_rate
    abandoned = cells[cells.abandoned].copy()
    abandoned["native_pass_rate"] = [
        native.get((r.task, native_arm(r.arm)), np.nan) for r in abandoned.itertuples()
    ]
    m3_abandon = (
        abandoned.groupby("arm")
        .agg(
            abandoned_cells=("task", "count"),
            precision=(
                "native_pass_rate",
                lambda s: float((s.dropna() == 0).mean()) if s.notna().any() else np.nan,
            ),
        )
        .reset_index()
        if not abandoned.empty
        else pd.DataFrame(columns=["arm", "abandoned_cells", "precision"])
    )
    m4 = con.sql("""
        SELECT arm, coalesce(guard, CASE WHEN fallback THEN 'G4' END) AS guard, count(*) AS n
        FROM decisions WHERE NOT infra_error AND (guard IS NOT NULL OR fallback)
        GROUP BY ALL ORDER BY ALL""").df()
    m5 = con.sql("""
        SELECT arm, avg(controller_calls) AS controller_calls_per_trial,
               sum(usd_controller) / nullif(sum(usd_total), 0) AS controller_usd_share,
               sum(latency_controller_s) / nullif(sum(controller_calls), 0)
                   AS controller_latency_mean_s,
               sum(latency_controller_s) / nullif(sum(wall_agent_s), 0) AS controller_latency_share
        FROM trials WHERE NOT infra_error GROUP BY ALL ORDER BY ALL""").df()
    return {
        "m2_triggers": m2,
        "m2_choices": m2_choices,
        "m3_savings": m3,
        "m3_abandon_precision": m3_abandon,
        "m4_guards": m4,
        "m5_overhead": m5,
    }


def analyze(experiment_paths: list[str], out_dir: Path | None = None, log=print) -> Path:
    cfg = read_yaml("config/analysis.yaml")
    exp_ids = [load_experiment(p).id for p in experiment_paths]
    out = out_dir or results_root() / "_analysis" / "+".join(exp_ids)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    trials, decisions, triggers = load_trials(experiment_paths)
    if trials.empty:
        raise SystemExit(f"no completed trials for {exp_ids}")
    con = duckdb.connect(str(out / "analysis.duckdb"))
    for name, df in (("trials", trials), ("decisions", decisions), ("trigger_states", triggers)):
        con.register(f"{name}_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}_df")

    summary = con.sql("""
        SELECT arm, count(DISTINCT task) AS n_tasks, count(*) AS n_attempts,
               avg(passed) AS pass_rate, avg(usd_total) AS usd_per_attempt,
               sum(usd_total) / nullif(sum(passed), 0) AS usd_per_solved_attempt,
               avg(usd_controller) AS controller_usd_per_attempt,
               avg(tokens_main + tokens_controller) AS tokens_per_attempt,
               avg(steps) AS steps_per_attempt, avg(wall_total_s) AS wall_s_per_attempt,
               avg(tool_failures) AS tool_failures_per_attempt,
               avg(repeated_commands) AS repeated_commands_per_attempt
        FROM trials WHERE NOT infra_error GROUP BY arm ORDER BY arm""").df()
    per_task = con.sql("""
        SELECT task, arm, any_value(stratum) AS stratum, any_value(category) AS category,
               count(*) AS n_attempts, avg(passed) AS pass_rate, avg(usd_total) AS usd_per_attempt
        FROM trials WHERE NOT infra_error GROUP BY task, arm ORDER BY task, arm""").df()
    infra = con.sql("""
        SELECT arm, termination_reason, count(*) AS n FROM trials WHERE infra_error
        GROUP BY ALL ORDER BY ALL""").df()
    drift = con.sql("""
        SELECT experiment || '/' || arm AS arm, count(DISTINCT config_hash) AS n_configs
        FROM trials GROUP BY experiment, arm HAVING count(DISTINCT config_hash) > 1""").df()
    hosts = con.sql("""
        SELECT arm, main_hosts, count(*) AS n_attempts FROM trials
        GROUP BY ALL ORDER BY ALL""").df()
    archived = sum(len(list((results_root() / e / "_infra").glob("*"))) for e in exp_ids)

    comp = comparisons(per_task, cfg)
    strata_rows = []
    for column in ("stratum", "category"):
        for value, group in per_task.dropna(subset=[column]).groupby(column):
            for _, row in comparisons(group, cfg).iterrows():
                strata_rows.append({"split": column, "value": value, **row.to_dict()})
    strata = pd.DataFrame(strata_rows)
    mech = mechanism_tables(con)

    clean = trials[~trials.infra_error].assign(
        tokens_total=lambda d: d.tokens_main + d.tokens_controller
    )
    for x, label in (
        ("usd_total", "USD per attempt"),
        ("tokens_total", "Tokens per attempt"),
        ("wall_total_s", "Wall time per attempt (s)"),
    ):
        pareto(clean.dropna(subset=[x]), x, label, out / "figures" / f"pareto_{x}.png", cfg["seed"])

    summary.to_csv(out / "summary.csv", index=False)
    per_task.to_csv(out / "per_task.csv", index=False)
    comp.to_csv(out / "comparisons.csv", index=False)
    strata.to_csv(out / "comparisons_by_stratum.csv", index=False)
    for name, df in mech.items():
        df.to_csv(out / f"{name}.csv", index=False)

    c1 = comp[comp.id == "C1"] if not comp.empty else comp
    secondary = comp[~comp.confirmatory] if not comp.empty else comp
    cols = [
        "id",
        "treatment",
        "control",
        "n_tasks",
        "pass_diff",
        "pass_diff_ci_lo",
        "pass_diff_ci_hi",
        "p_pass_holm",
        "cost_ratio_gm",
        "cost_ratio_ci_lo",
        "cost_ratio_ci_hi",
        "p_cost_holm",
    ]
    report = [
        f"# Analysis: {', '.join(exp_ids)}\n",
        (
            f"> **Config drift:** arms ran under more than one frozen config: "
            f"{dict(zip(drift.arm, drift.n_configs, strict=True))}. See manifests.\n"
            if not drift.empty
            else "Config check: every arm ran under one frozen config per experiment.\n"
        ),
        "## Confirmatory (C1)\n",
        (
            f"**C1 {c1.iloc[0].treatment} vs {c1.iloc[0].control}:** {c1_verdict(c1.iloc[0])}\n"
            if not c1.empty
            else "_C1 arms not present._\n"
        ),
        "## Secondary (C2–C5, Holm-adjusted bootstrap p-values)\n",
        md_table(secondary[[c for c in cols if c in secondary.columns]]),
        "## Per-arm summary (infra errors excluded)\n",
        md_table(summary),
        f"Infra/API faults (excluded above): {len(trials) - len(clean)} current, "
        f"{archived} archived re-runs.\n",
        md_table(infra),
        "Hosts that served the reasoning model (chosen by OpenRouter):\n",
        md_table(hosts),
        "---\n",
        "# Exploratory (mechanism, not confirmatory)\n",
        *(f"## {name}\n\n{md_table(df)}" for name, df in mech.items()),
        "## Comparisons by stratum / category (M6)\n",
        md_table(strata[[c for c in ["split", "value", *cols] if c in strata.columns]]),
        "## Figures\n",
        *(
            f"![{x}](figures/pareto_{x}.png)\n"
            for x in ("usd_total", "tokens_total", "wall_total_s")
        ),
    ]
    (out / "report.md").write_text("\n".join(report))
    con.close()
    log(f"wrote {out}")
    return out
