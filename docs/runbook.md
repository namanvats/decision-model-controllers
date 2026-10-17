# Runbook

Run the phases in order. Each phase sets `AGENTBENCH_PHASE` in `.env`; `run` refuses an experiment whose phase doesn't match.

- **Parallel runs:** keep `AGENTBENCH_CONCURRENCY` the same for the whole phase.
- **Interrupted:** after Ctrl-C or a crash, re-run the same `run` command; finished trials are skipped. Then run `cleanup`.

## 0. One-time setup

```bash
uv sync                                         # create .venv from uv.lock
cp .env.example .env                            # then fill in the keys (see below)
uv run pytest                                   # unit tests, no API spend
git add -A && git commit -m "agentbench setup"  # manifests record the commit
```

`.env`:
- `OPENROUTER_API_KEY`: one key per phase, with a credit limit set on it.
- `DAYTONA_API_KEY`: your Daytona key.
- `DAYTONA_TARGET=us`
- `AGENTBENCH_CONCURRENCY=16`: 10–20, within your Daytona plan's sandbox limits.
- `DAYTONA_API_URL`: leave unset.

## 1. Smoke test (`AGENTBENCH_PHASE=smoke`)

```bash
uv run agentbench probe                                 # one tiny call each to Sonnet, Haiku and Jev
uv run agentbench run experiments/smoke_mock.yaml -y    # our agent in a real sandbox, scripted model, no API spend
uv run agentbench run experiments/smoke.yaml -y         # real Sonnet on 1 task: native, small-LLM, Jev arms
uv run agentbench cleanup                               # confirm no Daytona sandboxes were left running
```

Check: each `results/tb21-smoke-v1/fix-git/*/1/outcome.json` has a reward and `infra_error: false`. On failure, read `harbor.log` in the same folder.

## 2. Calibration (`AGENTBENCH_PHASE=calibration`)

```bash
uv run python scripts/build_catalog.py vendor/tb21 > tasks/catalog.yaml  # task list with exclusions applied
uv run agentbench tasks sample tasks/calibration.yaml --n 86              # all eligible tasks (89 minus 3 excluded)
uv run agentbench plan experiments/calibration.yaml                       # trial count, done vs pending, spend estimate
uv run agentbench run experiments/calibration.yaml -y                     # sonnet-native × 2 attempts; finished trials are skipped
uv run agentbench cleanup                                                 # audit Daytona for leftover sandboxes
uv run agentbench tasks split                                             # stratify by pass rate → development.yaml (8) + evaluation.yaml
uv run agentbench limits                                                  # p90 steps/USD of passing runs, as a sanity check
git add -A && git commit -m "Calibration, caps, task split"               # record the state before dev
```

Caps are in `config/limits.yaml` under `sonnet:`: 60 steps / $1.00, above every calibration run. Keep them unless `limits` shows passing runs above them.

Excluded tasks and the reasons are in `tasks/exclusions.yaml` and the catalog. `break-filter-js-from-html` is out because Sonnet refuses it.

## 3. Offline replay (any phase; controller calls only, under $1)

```bash
uv run agentbench replay    # ask Jev / small LLM / rules about logged native states; score vs real outcomes
```

Gate: Jev beats base rate and rules on at least one question. See `results/_replay/tb21-calibration-v1/decision_quality.csv`.

## 4. Dev tuning (`AGENTBENCH_PHASE=dev`)

```bash
uv run agentbench validate experiments/dev.yaml       # all configs filled in, paired arms consistent
uv run agentbench run experiments/dev.yaml -y         # 8 dev tasks × {native, rules, small LLM, Jev} × 2 = 64 trials
uv run agentbench cleanup                             # audit Daytona for leftover sandboxes
uv run agentbench analyze experiments/dev.yaml        # → results/_analysis/tb21-dev-v1/report.md
```

Tune, giving Jev and the small LLM equal effort, and log each iteration in `docs/decision_log.md`:
- `config/prompts/jev_criteria.yaml`: option wording, shared by Jev and the small LLM
- `config/prompts/smallllm_controller.md`: the small LLM's prompt
- `config/triggers.yaml`: trigger and guard thresholds

After a change, `rm -rf results/tb21-dev-v1` and repeat this step. Once done:

```bash
uv run agentbench placebo-dist                         # placebo choice frequencies from Jev's dev decisions
uv run agentbench run experiments/dev_placebo.yaml -y  # placebo arm on the dev tasks (16 trials)
uv run agentbench tasks freeze tasks/evaluation.yaml   # mark held-out list frozen + SHA-256 checksum
git add -A && git commit -m "Freeze for eval"          # nothing changes after this commit
```

## 5. Held-out eval (`AGENTBENCH_PHASE=eval`, eval OpenRouter key)

```bash
uv run agentbench validate experiments/eval.yaml                       # must print OK (frozen tasks, all configs set)
uv run agentbench plan experiments/eval.yaml                           # held-out tasks × 5 arms × 3 attempts, spend estimate
nohup uv run agentbench run experiments/eval.yaml -y > eval.log 2>&1 & # long run in the background
tail -f eval.log                                                       # follow progress
uv run agentbench cleanup                                              # after it finishes: audit Daytona
uv run agentbench analyze experiments/eval.yaml                        # → results/_analysis/tb21-eval-v1/report.md
```

If you're short on budget, remove `sonnet-rules` and `sonnet-placebo` from `experiments/eval.yaml` before running.

## 6. Qwen (optional)

First set `qwen_main.reasoning` in `config/models.yaml` (e.g. `{effort: high}`).

```bash
uv run agentbench tasks sample tasks/qwen_check.yaml --n 15 --source tasks/calibration.yaml  # 15 calibrated tasks
uv run agentbench run experiments/qwen_check.yaml -y                                         # (calibration phase) can Qwen solve ≥ ~20%?
uv run agentbench limits experiments/qwen_check.yaml --arm qwen-native                       # copy into config/limits.yaml under qwen:
uv run agentbench run experiments/qwen_eval.yaml -y                                          # (eval phase) held-out × {qwen-native, qwen-jev} × 3
uv run agentbench analyze experiments/eval.yaml experiments/qwen_eval.yaml                   # Sonnet + Qwen report, including C5
```

## Useful anytime

```bash
uv run agentbench plan <experiment.yaml>          # progress: done / pending / retry
uv run agentbench cleanup --delete                # delete orphaned sandboxes (asks first)
rm results/<experiment-id>/.runner.lock           # only if a crashed run left a stale lock
```
