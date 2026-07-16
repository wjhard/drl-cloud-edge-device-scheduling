#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="training/configs/ppo_mlp_residual.yaml"
MODEL_PATH="training/checkpoints/ppo_mlp_residual"
SCENARIO_DIR="evaluation/scenarios"
RESULTS_PATH="evaluation/results/summary_mlp_residual_bestof64.json"
REQUIRED_IMPORTS="import importlib.util; import gymnasium, matplotlib, networkx, numpy, pulp, pytest, rich, torch, tqdm, yaml; assert all(importlib.util.find_spec(name) is not None for name in ('sb3_contrib', 'stable_baselines3', 'tensorboard'))"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

run_step() {
  local name="$1"
  shift
  echo "==== ${name} ===="
  "$@"
}

cd "$PROJECT_ROOT"
echo "FINAL_PIPELINE_START"
echo "project_root=$PROJECT_ROOT"
echo "python_command=$PYTHON_BIN"

echo "==== check dependencies ===="
pip_check_exit=0
import_check_exit=0
"$PYTHON_BIN" -m pip check || pip_check_exit=$?
"$PYTHON_BIN" -c "$REQUIRED_IMPORTS" || import_check_exit=$?

if (( pip_check_exit != 0 )); then
  echo "WARNING: pip check reported conflicts; project import check will determine readiness" >&2
fi

if (( import_check_exit != 0 )); then
  echo "dependencies_ready=false; installing requirements"
  run_step "upgrade pip" "$PYTHON_BIN" -m pip install --upgrade pip
  if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    run_step "install CPU PyTorch" "$PYTHON_BIN" -m pip install \
      "torch==2.12.1+cpu" --index-url https://download.pytorch.org/whl/cpu
  fi
  run_step "install project requirements" "$PYTHON_BIN" -m pip install -r requirements.txt
  echo "==== verify dependency consistency ===="
  if ! "$PYTHON_BIN" -m pip check; then
    echo "WARNING: pip check still reports conflicts outside the project dependency set" >&2
  fi
  run_step "verify required imports" "$PYTHON_BIN" -c "$REQUIRED_IMPORTS"
else
  echo "dependencies_ready=true; skipping installation"
fi

if [[ -f "$MODEL_PATH" || -f "${MODEL_PATH}.zip" ]]; then
  echo "checkpoint_exists=true; skipping training"
  echo "checkpoint_path=${MODEL_PATH}.zip"
else
  echo "checkpoint_exists=false; starting full 200000-step training"
  run_step "train final residual model" "$PYTHON_BIN" training/train_ppo.py --config "$CONFIG_PATH"
fi

if compgen -G "${SCENARIO_DIR}/scenario_*.json" >/dev/null; then
  scenario_files=("$SCENARIO_DIR"/scenario_*.json)
  scenario_count="${#scenario_files[@]}"
  echo "validation_scenarios_exist=true; count=${scenario_count}; skipping generation"
else
  echo "validation_scenarios_exist=false; generating fixed validation scenarios"
  run_step "generate validation scenarios" "$PYTHON_BIN" \
    evaluation/generate_validation_scenarios.py --config "$CONFIG_PATH"
fi

run_step "evaluate Residual best-of-64" "$PYTHON_BIN" \
  evaluation/evaluate_bestofn.py \
  --config "$CONFIG_PATH" \
  --model-path "$MODEL_PATH" \
  --num-samples 64

if [[ ! -f "$RESULTS_PATH" ]]; then
  echo "ERROR: expected results file was not created: $RESULTS_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - "$RESULTS_PATH" <<'PY'
import json
import sys

results_path = sys.argv[1]
with open(results_path, "r", encoding="utf-8") as file:
    summary = json.load(file)
mean_ratio = float(summary["overall"]["mean_ratio"])
records = summary["scenarios"]
outperform_count = sum(float(record["ratio"]) < 1.0 for record in records)
print("FINAL_PIPELINE_RESULT")
print(f"overall_mean_ratio={mean_ratio:.12f}")
print(f"outperform_heft_scenarios={outperform_count}/{len(records)}")
print(f"results_path={results_path}")
print("FINAL_PIPELINE_COMPLETE")
PY
