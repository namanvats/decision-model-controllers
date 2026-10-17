# agentbench — decision-delegated controllers on Terminal-Bench 2.1

One ReAct agent loop. Five interchangeable controllers (`none`, `rules`, `small_llm`, `jev`, `placebo`), consulted only at deterministic triggers. Same tasks for every controller. The study design is in [experiment_plan.md](experiment_plan.md), the infrastructure in [infra_plan.md](infra_plan.md), and frozen choices in [docs/decision_log.md](docs/decision_log.md).

```
agentbench run  ──► harbor run (1 trial)  ──► Daytona sandbox (task image + verifier)
                       └─ ControlledAgent.run()  (host side)
                            ├─ reasoning model ─► OpenRouter (host chosen by OpenRouter, logged per call)
                            ├─ bash            ─► environment.exec in the sandbox
                            └─ controller      ─► none | rules | Haiku | Jev | placebo
results/<experiment>/<task>/<arm>/<attempt>/  manifest · events · trigger_states · decision_events · usage · harbor_result · outcome
```

API keys stay on your machine. Only shell commands reach the sandbox.

## Setup

```bash
uv sync                      # creates .venv from uv.lock (Python 3.12)
cp .env.example .env         # fill OPENROUTER_API_KEY, DAYTONA_API_KEY
uv run pytest                # unit tests, zero spend
```

- **OpenRouter:** create one key per phase (smoke, calibration, dev, eval) and set a **credit limit** on each. This is the only provider-enforced spend cap. Put the current phase's key in `.env`.
- **`AGENTBENCH_PHASE`:** set it in `.env` to the phase you are running. `run` refuses an experiment whose `phase` differs, which stops you spending an eval budget with a dev key by accident.
- **`AGENTBENCH_CONCURRENCY`:** sets how many trials (Daytona sandboxes) run at once; 10–20 for real runs. It overrides the experiments' default of 1 and is recorded in each manifest. Keep one value per phase.
- **Commits:** the repo is git-initialised but has no commits. Commit before Phase 1 so every manifest records a real commit; `git_dirty` is recorded too.

## Commands

| Command | What it does |
|---|---|
| `agentbench validate EXP` | Loads every referenced config. Fails on `CONFIGURE_ME`, unknown aliases, paired arms that differ beyond the controller, an unfrozen or changed eval task list, or a phase mismatch |
| `agentbench plan EXP` | Prints the randomized trial grid, progress, and expected spend / ceiling |
| `agentbench run EXP` | Runs or resumes trials. Completed trials are skipped and infra failures re-run once. A budget ledger guards each admission |
| `agentbench probe` | Phase 0 contract check: one tiny call each to Sonnet, Haiku and Jev |
| `agentbench tasks sample/split/freeze` | Builds task lists (§7) |
| `agentbench limits` | Suggests frozen caps: p90 of steps and USD among passing calibration runs |
| `agentbench placebo-dist` | Writes the placebo distribution from dev-phase Jev decisions |
| `agentbench replay` | Phase 2 offline study (M1): AUROC, ECE and accuracy vs grader outcome |
| `agentbench analyze EXP…` | Writes DuckDB, CSVs, `report.md` and Pareto figures |
| `agentbench cleanup [--delete]` | Daytona orphan audit. Dry run by default; deletes only sandboxes labelled with our trial IDs |

## Runbook

Short version with one-line comments per command: [docs/runbook.md](docs/runbook.md).

Run `agentbench cleanup` after every phase and after any interrupted run.

### Phase 0: smoke (about $3)
```bash
# AGENTBENCH_PHASE=smoke, smoke key in .env
uv run agentbench probe                                    # Sonnet / Haiku / Jev answer; costs reported
uv run pytest -m integration                               # same contract, as assertions
uv run harbor run -d "terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a" \
  -a oracle -i terminal-bench/fix-git -e daytona -n 1      # dataset + sandbox + verifier work
uv run agentbench run experiments/smoke_mock.yaml          # our agent, scripted model, no model spend
uv run agentbench run experiments/smoke.yaml               # real Sonnet: native, small-LLM, Jev arms
uv run agentbench cleanup
```
**Gate:** each `outcome.json` has a reward. `usage.json` costs match the OpenRouter activity page. `cleanup` shows zero orphans.

### Phase 1: calibration
```bash
# Optional: re-run with your Daytona plan limits
uv run python scripts/build_catalog.py vendor/tb21 --max-cpus 4 --max-memory-mb 8192 > tasks/catalog.yaml
uv run agentbench tasks sample tasks/calibration.yaml --n 30
uv run agentbench tasks sample tasks/qwen_check.yaml --n 15 --source tasks/calibration.yaml
# AGENTBENCH_PHASE=calibration
uv run agentbench run experiments/calibration.yaml         # 30 tasks × sonnet-native × 2
uv run agentbench tasks split                              # → tasks/development.yaml (8) + tasks/evaluation.yaml (36)
uv run agentbench limits                                   # copy into config/limits.yaml `sonnet:`
```
The Qwen check needs `qwen_main.reasoning` set in `config/models.yaml` first. Then:
```bash
uv run agentbench run experiments/qwen_check.yaml
uv run agentbench limits experiments/qwen_check.yaml --arm qwen-native   # → limits.yaml `qwen:`
```
**Gate:** at least 25 tasks are in mid or easy. Caps are frozen and logged in `docs/decision_log.md`. Go/no-go on Qwen (≥ ~20% pass).

### Phase 2: offline controller study (controller calls only)
```bash
uv run agentbench replay experiments/calibration.yaml
```
**Gate:** Jev beats base rate and rules on at least one question type.

### Phase 3: dev tuning
```bash
# AGENTBENCH_PHASE=dev
uv run agentbench run experiments/dev.yaml                 # native, rules, small LLM, Jev × 2
# tune config/prompts/jev_criteria.yaml, smallllm_controller.md, config/triggers.yaml with equal effort; log iterations
uv run agentbench placebo-dist experiments/dev.yaml
uv run agentbench run experiments/dev_placebo.yaml
uv run agentbench analyze experiments/dev.yaml experiments/dev_placebo.yaml
uv run agentbench tasks freeze tasks/evaluation.yaml       # frozen + SHA-256
git commit -am "Freeze for Phase 4"                        # §12 freeze checklist
```

### Phase 4: held-out evaluation (the confirmatory run)
```bash
# AGENTBENCH_PHASE=eval, eval key with credit limit
uv run agentbench validate experiments/eval.yaml
uv run agentbench plan experiments/eval.yaml
uv run agentbench run experiments/eval.yaml                # 36 tasks × 5 arms × 3 = 540 trials; resumable
uv run agentbench cleanup
uv run agentbench analyze experiments/eval.yaml            # results/_analysis/tb21-eval-v1/report.md
```
If funds are short, remove `sonnet-rules` and `sonnet-placebo` from `experiments/eval.yaml` before the run (§11, minimum viable version).

### Phase 5: Qwen replication (conditional)
```bash
uv run agentbench run experiments/qwen_eval.yaml
uv run agentbench analyze experiments/eval.yaml experiments/qwen_eval.yaml
```

Daytona compute is billed separately. Reconcile it from the Daytona dashboard after each phase.

## Layout

- `agents/`: one YAML per arm. Arms of the same reasoning model may differ only in `controller` (enforced).
- `experiments/`: the trial grid per phase.
- `tasks/`: the catalog and task lists.
- `config/`: models, limits, triggers/guards, prompts, analysis.
- `src/agentbench/`:
  - loop: `loop.py`, `tools.py`, `triggers.py`, `state.py`, `guards.py`, `controllers/`
  - Harbor: `harbor_adapter.py`, `runner.py`, `ledger.py`, `cleanup.py`
  - study: `tasks.py`, `replay.py`, `analysis.py`
