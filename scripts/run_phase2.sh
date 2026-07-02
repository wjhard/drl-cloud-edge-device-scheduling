#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

run_step() {
  local name="$1"
  shift
  echo "==== ${name} ===="
  if ! "$@"; then
    echo "ERROR: ${name} failed" >&2
    exit 1
  fi
}

run_step "train MaskablePPO" "$PYTHON_BIN" training/train_ppo.py --config training/configs/ppo_mlp_baseline.yaml
run_step "generate validation scenarios" "$PYTHON_BIN" evaluation/generate_validation_scenarios.py
run_step "evaluate RL vs HEFT" "$PYTHON_BIN" evaluation/evaluate.py \
  --config training/configs/ppo_mlp_baseline.yaml \
  --model-path training/checkpoints/ppo_mlp_baseline

