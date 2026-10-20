#!/usr/bin/env bash
# Eval-2: calibrate the remaining Terminal-Bench 2.1 tasks, draw a new frozen held-out set,
# run all arms on it, and analyze (eval-2 alone and pooled with eval-1).
#
# Safe to re-run: finished trials are skipped, existing task lists are never overwritten,
# and every output goes to new folders (results/tb21-calibration-v2, results/tb21-eval-v2).
#
#   bash scripts/run_eval2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$*"; }

step "1/6 sample the tasks not calibrated yet"
if [[ -f tasks/calibration_2.yaml ]]; then
  echo "tasks/calibration_2.yaml exists, keeping it"
else
  uv run agentbench tasks sample tasks/calibration_2.yaml --n 999 --exclude tasks/calibration.yaml
fi

step "2/6 calibration (sonnet-native x 2)"
export AGENTBENCH_PHASE=calibration
uv run agentbench run experiments/calibration_2.yaml -y
uv run agentbench plan experiments/calibration_2.yaml --require-complete
uv run agentbench cleanup

step "3/6 new held-out set (frozen)"
if [[ -f tasks/evaluation_2.yaml ]]; then
  echo "tasks/evaluation_2.yaml exists, keeping it"
else
  uv run agentbench tasks split experiments/calibration_2.yaml \
    --n-dev 0 --n-heldout 999 --eval-out tasks/evaluation_2.yaml
  uv run agentbench tasks freeze tasks/evaluation_2.yaml
fi

step "4/6 evaluation (6 arms x 3 attempts)"
export AGENTBENCH_PHASE=eval
uv run agentbench validate experiments/eval_2.yaml
uv run agentbench plan experiments/eval_2.yaml
uv run agentbench run experiments/eval_2.yaml -y
uv run agentbench plan experiments/eval_2.yaml --require-complete

step "5/6 sandbox audit"
uv run agentbench cleanup

step "6/6 analysis"
uv run agentbench analyze experiments/eval_2.yaml
uv run agentbench analyze experiments/eval.yaml experiments/eval_2.yaml
uv run agentbench replay experiments/eval_2.yaml

printf '\nDone. Reports:\n  results/_analysis/tb21-eval-v2/report.md\n  results/_analysis/tb21-eval-v1+tb21-eval-v2/report.md\n'
