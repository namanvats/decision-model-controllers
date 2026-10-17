import asyncio
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from agentbench import __version__
from agentbench.config import (
    PROJECT_ROOT,
    ExperimentCfg,
    ResolvedArm,
    TaskList,
    load_benchmark,
    resolve,
)
from agentbench.ledger import Ledger, trial_spend
from agentbench.telemetry import write_json

AGENT_IMPORT_PATH = "agentbench.harbor_adapter:ControlledAgent"
SHUTDOWN_GRACE_S = 180


def results_root() -> Path:
    return resolve(os.environ.get("AGENTBENCH_RESULTS_DIR", "results"))


@dataclass(frozen=True)
class Trial:
    experiment_id: str
    task: str
    arm_id: str
    attempt: int
    order: int

    @property
    def id(self) -> str:
        return f"{self.experiment_id}--{self.task_short}--{self.arm_id}--a{self.attempt}"

    @property
    def uid(self) -> str:
        return uuid.uuid5(uuid.NAMESPACE_URL, self.id).hex[:16]

    @property
    def task_short(self) -> str:
        return self.task.rsplit("/", 1)[-1]

    def dir(self) -> Path:
        return (
            results_root() / self.experiment_id / self.task_short / self.arm_id / str(self.attempt)
        )


def plan_trials(exp: ExperimentCfg, arms: list[ResolvedArm], tasks: TaskList) -> list[Trial]:
    """Task × attempt blocks; arm order shuffled within each block with a seeded RNG."""
    rng = random.Random(exp.seed)
    trials: list[Trial] = []
    for task in tasks.tasks:
        for attempt in range(1, exp.attempts + 1):
            block = [a.id for a in arms]
            if exp.randomize_arm_order:
                rng.shuffle(block)
            for arm_id in block:
                trials.append(Trial(exp.id, task.name, arm_id, attempt, len(trials)))
    return trials


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def completed_trial_dirs(experiment_id: str) -> list[Path]:
    """Trial dirs with an outcome, excluding archived infra failures."""
    exp_dir = results_root() / experiment_id
    return sorted(p.parent for p in exp_dir.glob("*/*/*/outcome.json"))


def infra_dir(trial: Trial) -> Path:
    return results_root() / trial.experiment_id / "_infra"


def trial_status(trial: Trial) -> str:
    """done | pending | retry | gave_up. Infra failures are re-run once (§10.2)."""
    outcome = read_json(trial.dir() / "outcome.json")
    if outcome and not outcome["infra_error"]:
        return "done"
    previous = [
        p
        for p in infra_dir(trial).glob(f"{trial.id}__*")
        if (read_json(p / "outcome.json") or {}).get("infra_error")
    ]
    if outcome:
        return "gave_up" if previous else "retry"
    return "retry" if trial.dir().exists() else "pending"


def archive(trial: Trial) -> None:
    target = infra_dir(trial)
    target.mkdir(parents=True, exist_ok=True)
    n = len(list(target.glob(f"{trial.id}__*")))
    shutil.move(trial.dir(), target / f"{trial.id}__{n}")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def provenance() -> dict[str, Any]:
    lock = PROJECT_ROOT / "uv.lock"
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "harbor_version": version("harbor"),
        "agentbench_version": __version__,
        "python": sys.version.split()[0],
    }


def concurrency(exp: ExperimentCfg) -> int:
    """AGENTBENCH_CONCURRENCY (from .env) overrides the experiment's default."""
    value = os.environ.get("AGENTBENCH_CONCURRENCY")
    return int(value) if value else exp.concurrency


def harbor_bin() -> str:
    return str(Path(sys.executable).parent / "harbor")


def dataset_ref() -> str:
    bench = load_benchmark()
    return f"{bench['harbor_dataset']}@{bench['harbor_dataset_ref']}"


async def prefetch_dataset() -> None:
    """Download the pinned dataset into Harbor's cache once, before parallel trials start."""
    proc = await asyncio.create_subprocess_exec(
        harbor_bin(), "download", dataset_ref(), "--cache",
        cwd=PROJECT_ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )  # fmt: skip
    output, _ = await proc.communicate()
    if proc.returncode:
        raise SystemExit(f"dataset download failed:\n{output.decode()[-2000:]}")


def harbor_command(trial: Trial, arm: ResolvedArm, exp: ExperimentCfg) -> list[str]:
    labels = {"agentbench.trial": trial.uid, "agentbench.experiment": exp.id}
    # fmt: off
    cmd = [
        harbor_bin(), "run",
        "-d", dataset_ref(),
        "-i", trial.task,
        "-a", AGENT_IMPORT_PATH,
        "-e", load_benchmark()["environment"],
        "-n", "1", "-k", "1",
        "-o", str(trial.dir() / "harbor"),
        "--job-name", "job",
        "--ak", f"arm={arm.path}",
        "--ak", f"trial_dir={trial.dir()}",
        "--ak", f"trial_id={trial.id}",
        "--ek", f"labels={json.dumps(labels)}",
        "-q", "-y",
    ]
    # fmt: on
    if exp.limits_ref:
        cmd += ["--ak", f"limits_ref={exp.limits_ref}"]
    return cmd


def _seconds(timing: dict | None) -> float | None:
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    start = datetime.fromisoformat(timing["started_at"])
    return (datetime.fromisoformat(timing["finished_at"]) - start).total_seconds()


def build_outcome(trial: Trial, harbor: dict | None, usage: dict | None, exit_code: int) -> dict:
    rewards = ((harbor or {}).get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    exc = (harbor or {}).get("exception_info")
    harbor_timeout = bool(exc) and exc.get("exception_type") == "AgentTimeoutError"
    termination = "harbor_timeout" if harbor_timeout else (usage or {}).get("termination_reason")
    infra_error = (
        harbor is None
        or reward is None
        or termination in (None, "agent_error", "cancelled")
        or (bool(exc) and not harbor_timeout)
    )
    harbor = harbor or {}
    return {
        "trial_id": trial.id,
        "experiment_id": trial.experiment_id,
        "task": trial.task,
        "arm": trial.arm_id,
        "attempt": trial.attempt,
        "passed": reward is not None and reward >= 1.0,
        "reward": reward,
        "termination_reason": termination,
        "infra_error": infra_error,
        "exception_type": exc and exc.get("exception_type"),
        "exception_message": exc and exc.get("exception_message"),
        "harbor_exit_code": exit_code,
        "steps": (usage or {}).get("steps"),
        "usd_total": ((usage or {}).get("totals") or {}).get("usd_total"),
        "wall_s": {
            "total": _seconds(harbor),
            "provisioning": _seconds(harbor.get("environment_setup")),
            "agent": _seconds(harbor.get("agent_execution")),
            "verifier": _seconds(harbor.get("verifier")),
        },
    }


async def run_trial(trial: Trial, arm: ResolvedArm, exp: ExperimentCfg) -> dict:
    trial_dir = trial.dir()
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = harbor_command(trial, arm, exp)
    write_json(
        trial_dir / "manifest.json",
        {
            "trial_id": trial.id,
            "trial_uid": trial.uid,
            "experiment_id": exp.id,
            "phase": exp.phase,
            "task": trial.task,
            "arm_id": trial.arm_id,
            "attempt": trial.attempt,
            "order": trial.order,
            "seed": exp.seed,
            "concurrency": concurrency(exp),
            "trial_cap_usd": arm.limits.max_trial_usd,
            "benchmark": load_benchmark(),
            "provenance": provenance(),
            "resolved_arm": arm.snapshot(),
            "harbor_command": cmd,
            "started_at": datetime.now(UTC).isoformat(),
        },
    )
    with (trial_dir / "harbor.log").open("w") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=PROJECT_ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT
        )
        try:
            exit_code = await proc.wait()
        except asyncio.CancelledError:
            proc.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(proc.wait(), SHUTDOWN_GRACE_S)
            except TimeoutError:
                proc.kill()
            raise

    results = [p for p in (trial_dir / "harbor" / "job").glob("*/result.json")]
    harbor = read_json(results[0]) if results else None
    if harbor:
        write_json(trial_dir / "harbor_result.json", harbor)
    outcome = build_outcome(trial, harbor, read_json(trial_dir / "usage.json"), exit_code)
    write_json(trial_dir / "outcome.json", outcome)
    return outcome


async def run_experiment(
    exp: ExperimentCfg, arms: list[ResolvedArm], trials: list[Trial], log=print
) -> None:
    exp_dir = results_root() / exp.id
    exp_dir.mkdir(parents=True, exist_ok=True)
    log("prefetching dataset into the Harbor cache ...")
    await prefetch_dataset()
    lock = exp_dir / ".runner.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another run is active (delete it if stale)") from None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    by_id = {a.id: a for a in arms}
    todo = [t for t in trials if trial_status(t) in ("pending", "retry")]
    log(f"{len(trials)} trials, {len(trials) - len(todo)} already done, {len(todo)} to run")
    ledger = Ledger(exp_dir, exp.budget)
    parallel = concurrency(exp)
    log(f"running up to {parallel} trials in parallel")
    semaphore = asyncio.Semaphore(parallel)
    stop = asyncio.Event()
    done = 0

    async def worker(trial: Trial) -> None:
        nonlocal done
        async with semaphore:
            if stop.is_set():
                return
            cap = by_id[trial.arm_id].limits.max_trial_usd
            if not ledger.admit(cap):
                log(
                    f"budget: ${ledger.spent:.2f} spent, next trial could cross the "
                    f"${exp.budget.hard_cap_usd} hard cap — stopping admission"
                )
                stop.set()
                return
            if trial.dir().exists():
                archive(trial)
            try:
                outcome = await run_trial(trial, by_id[trial.arm_id], exp)
            finally:
                ledger.settle(cap, trial_spend(trial.dir()))
            done += 1
            status = (
                "INFRA" if outcome["infra_error"] else ("pass" if outcome["passed"] else "fail")
            )
            log(
                f"[{done}/{len(todo)}] {trial.task_short} {trial.arm_id} a{trial.attempt}: "
                f"{status} ({outcome['termination_reason']}) ${outcome['usd_total'] or 0:.3f} "
                f"| spent ${ledger.spent:.2f}"
            )
            if ledger.crossed_soft_warning():
                log(f"WARNING: spend passed soft limit ${exp.budget.soft_warning_usd}")

    try:
        await asyncio.gather(*(worker(t) for t in todo))
    finally:
        lock.unlink(missing_ok=True)
        log("Run `agentbench cleanup` to audit Daytona for orphaned sandboxes.")
