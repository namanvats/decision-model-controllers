import asyncio
import json
from collections import Counter
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from agentbench.config import PROJECT_ROOT, load_experiment, load_tasks, read_yaml, resolve

load_dotenv(PROJECT_ROOT / ".env")

app = typer.Typer(no_args_is_help=True, help="Controller study on Terminal-Bench.")
tasks_app = typer.Typer(no_args_is_help=True, help="Build, split, and freeze task lists.")
app.add_typer(tasks_app, name="tasks")


def _validated(experiment: str, check_phase: bool = True):
    from agentbench.config import validate_experiment

    exp, arms, tasks, errors = validate_experiment(experiment, check_phase)
    if errors:
        typer.secho(f"{experiment}: {len(errors)} problem(s)", fg="red")
        for e in errors:
            typer.echo(f"  - {e}")
        raise typer.Exit(1)
    return exp, arms, tasks


@app.command()
def validate(experiment: str, check_phase: bool = True) -> None:
    """Check an experiment and everything it references (zero spend)."""
    exp, arms, tasks = _validated(experiment, check_phase)
    typer.secho(
        f"OK: {exp.id} — {len(arms)} arms × {len(tasks.tasks)} tasks × {exp.attempts}", fg="green"
    )


@app.command()
def plan(experiment: str) -> None:
    """Show the randomized trial grid, progress, and estimated spend."""
    from agentbench.runner import concurrency, plan_trials, trial_status

    exp, arms, tasks = _validated(experiment, check_phase=False)
    trials = plan_trials(exp, arms, tasks)
    status = Counter(trial_status(t) for t in trials)
    pricing = read_yaml("config/pricing.yaml")["expected_trial_usd"]
    by_id = {a.id: a for a in arms}
    todo = [t for t in trials if trial_status(t) in ("pending", "retry")]
    expected = sum(pricing.get(by_id[t.arm_id].cfg.reasoning.model_ref, 0) for t in todo)
    ceiling = sum(by_id[t.arm_id].limits.max_trial_usd for t in todo)
    typer.echo(f"{exp.id}: {len(trials)} trials ({dict(status)}), {concurrency(exp)} in parallel")
    typer.echo(
        f"remaining spend: expected ~${expected:.0f}, ceiling ${ceiling:.0f} "
        f"(hard cap ${exp.budget.hard_cap_usd})"
    )
    for t in trials[:12]:
        typer.echo(f"  {t.order:>4}  {t.task_short:<36} a{t.attempt}  {t.arm_id}")
    if len(trials) > 12:
        typer.echo(f"  ... {len(trials) - 12} more")


@app.command()
def run(experiment: str, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Run (or resume) an experiment: one Harbor invocation per trial."""
    from agentbench.runner import plan_trials, run_experiment

    exp, arms, tasks = _validated(experiment)
    trials = plan_trials(exp, arms, tasks)
    if not yes:
        typer.confirm(
            f"Run {exp.id} ({len(trials)} trials, hard cap ${exp.budget.hard_cap_usd})?", abort=True
        )
    asyncio.run(run_experiment(exp, arms, trials, log=typer.echo))


@app.command()
def probe() -> None:
    """Phase 0 contract check: one tiny call each to the reasoning model, small LLM, and Jev."""
    from agentbench.config import load_arm
    from agentbench.controllers import decide_safely, make_controller
    from agentbench.openrouter import OpenRouter

    async def main() -> None:
        client = OpenRouter()
        try:
            arm = load_arm("agents/sonnet-smallllm.yaml", "config/limits.yaml#smoke")
            _, usage = await client.chat(arm.model, [{"role": "user", "content": "Reply with: ok"}])
            typer.echo(f"reasoning  {json.dumps(usage.to_dict())}")
            state = {
                "task_instruction": "Create /app/hello.txt containing 'hi'.",
                "trigger": "T3",
                "last_5_actions": [
                    {"command": "echo hi > /app/hello.txt", "exit_code": 0, "output_tail": ""}
                ],
                "finish_summary": "Created the file.",
            }
            for path in ("agents/sonnet-smallllm.yaml", "agents/sonnet-jev.yaml"):
                arm = load_arm(path, "config/limits.yaml#smoke")
                d = await decide_safely(make_controller(arm, client, "probe"), "finish", state)
                typer.echo(
                    f"{arm.controller_type:<10} choice={d.choice} fallback={d.fallback} "
                    f"error={d.error} probs={d.probabilities} latency={d.latency_s:.2f}s "
                    f"usage={json.dumps(d.usage.to_dict() if d.usage else None)}"
                )
        finally:
            await client.aclose()

    asyncio.run(main())


@tasks_app.command("sample")
def tasks_sample(
    out: str,
    n: int = 30,
    seed: int = 20261011,
    catalog: str = "tasks/catalog.yaml",
    source: str | None = typer.Option(None, help="Restrict to tasks in this list"),
) -> None:
    """Seeded, category-stratified sample of non-excluded catalog tasks."""
    from agentbench.tasks import sample, write_tasks

    tasks = sample(catalog, n, seed, source)
    write_tasks(out, tasks)
    typer.echo(
        f"wrote {len(tasks.tasks)} tasks to {out}: {dict(Counter(t.category for t in tasks.tasks))}"
    )


@tasks_app.command("split")
def tasks_split(
    calibration: str = typer.Argument("experiments/calibration.yaml"),
    arm: str = "sonnet-native",
    n_dev: int = 8,
    n_heldout: int = 36,
    hard_cap: float = 0.2,
    seed: int = 20261011,
    dev_out: str = "tasks/development.yaml",
    eval_out: str = "tasks/evaluation.yaml",
) -> None:
    """Stratify calibrated tasks (hard/mid/easy) and draw dev + held-out lists."""
    from agentbench.tasks import native_pass_counts, split, write_tasks

    exp = load_experiment(calibration)
    dev, heldout, missing = split(
        load_tasks(exp.tasks), native_pass_counts(exp.id, arm), n_dev, n_heldout, hard_cap, seed
    )
    write_tasks(dev_out, dev)
    write_tasks(eval_out, heldout)
    for name, tl in (("dev", dev), ("held-out", heldout)):
        typer.echo(f"{name}: {len(tl.tasks)} tasks {dict(Counter(t.stratum for t in tl.tasks))}")
    if missing:
        typer.secho(f"no calibration outcome for {len(missing)} tasks: {missing}", fg="yellow")


@tasks_app.command("freeze")
def tasks_freeze(path: str) -> None:
    """Mark a task list frozen and record its SHA-256 checksum."""
    from agentbench.tasks import freeze

    typer.echo(f"frozen {path}: sha256 {freeze(path)}")


@app.command()
def limits(
    calibration: str = typer.Argument("experiments/calibration.yaml"), arm: str = "sonnet-native"
) -> None:
    """Suggest frozen caps: p90 of steps and USD among passing calibration runs."""
    from agentbench.tasks import suggest_limits

    result = suggest_limits(load_experiment(calibration).id, arm)
    typer.echo(yaml.safe_dump(result, sort_keys=False))
    typer.echo(
        "Copy max_agent_steps / max_trial_usd into config/limits.yaml, then log it in "
        "docs/decision_log.md."
    )


@app.command("placebo-dist")
def placebo_dist(
    experiments: list[str] = typer.Argument(None),
    arm: str = "sonnet-jev",
    seed: int = 20261011,
    out: str = "config/placebo_distribution.yaml",
) -> None:
    """Derive the placebo choice distribution from dev-phase Jev decisions."""
    from agentbench.tasks import placebo_distribution

    paths = experiments or ["experiments/dev.yaml"]
    dist = placebo_distribution([load_experiment(p).id for p in paths], arm, seed)
    resolve(out).write_text(yaml.safe_dump(dist.model_dump(), sort_keys=False))
    typer.echo(f"wrote {out}: {dist.model_dump()}")


@app.command()
def replay(
    experiments: list[str] = typer.Argument(None),
    arm: str = "sonnet-native",
    controllers: str = "rules,small_llm,jev",
    concurrency: int = 4,
) -> None:
    """Phase 2: replay logged native trigger states to controllers; score vs grader outcomes."""
    from agentbench.replay import decision_quality, load_states
    from agentbench.replay import replay as run_replay
    from agentbench.runner import results_root

    paths = experiments or ["experiments/calibration.yaml"]
    exp_ids = [load_experiment(p).id for p in paths]
    rows = load_states(exp_ids, arm)
    if not rows:
        raise typer.BadParameter(f"no trigger states from {arm} in {exp_ids}")
    out = results_root() / "_replay" / "+".join(exp_ids)
    results = {
        name: asyncio.run(run_replay(rows, name, out / f"{name}.jsonl", concurrency, typer.echo))
        for name in controllers.split(",")
    }
    table = decision_quality(rows, results)
    table.to_csv(out / "decision_quality.csv", index=False)
    typer.echo(table.to_string(index=False))
    typer.echo(f"wrote {out}")


@app.command()
def analyze(
    experiments: list[str] = typer.Argument(None),
    out: Path | None = typer.Option(None),
) -> None:
    """Load trials into DuckDB; write summary, comparisons, mechanism tables, figures, report."""
    from agentbench.analysis import analyze as run_analysis

    run_analysis(experiments or ["experiments/eval.yaml"], out, log=typer.echo)


@app.command()
def cleanup(
    delete: bool = typer.Option(
        False, "--delete", help="Delete matched orphans (default: dry run)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Audit Daytona for sandboxes left behind by agentbench trials."""
    from agentbench.cleanup import audit

    audit(delete, confirm=lambda msg: yes or typer.confirm(msg), log=typer.echo)
