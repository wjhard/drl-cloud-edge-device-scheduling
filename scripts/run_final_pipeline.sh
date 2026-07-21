#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="training/configs/ppo_mlp_residual.yaml"
MODEL_PATH="training/checkpoints/ppo_mlp_residual"
SCENARIO_DIR="evaluation/scenarios"
RESULTS_DIR="evaluation/results/final_pipeline_lns_repeats"
SUMMARY_PATH="evaluation/results/final_pipeline_lns_summary.json"
CANONICAL_SEEDS=(1565812275 842234145 386081360 1117038938 1760006972)
REQUIRED_IMPORTS="import importlib.util; import gymnasium, matplotlib, networkx, numpy, pulp, pytest, rich, scipy, torch, tqdm, yaml; assert all(importlib.util.find_spec(name) is not None for name in ('sb3_contrib', 'stable_baselines3', 'tensorboard'))"

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
"$PYTHON_BIN" -c "$REQUIRED_IMPORTS" >/dev/null 2>&1 || import_check_exit=$?

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

echo "==== evaluate final Residual best-of-64 + relocation + LNS ===="
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/direct_repeat_*.json
for index in "${!CANONICAL_SEEDS[@]}"; do
  repeat=$((index + 1))
  seed="${CANONICAL_SEEDS[$index]}"
  run_path="$RESULTS_DIR/direct_repeat_${repeat}.json"
  run_step "final LNS repeat ${repeat}/5 (seed=${seed})" "$PYTHON_BIN" \
    evaluation/evaluate_residual_lns.py \
    --config "$CONFIG_PATH" \
    --model-path "$MODEL_PATH" \
    --results-path "$run_path" \
    --sampling-seed "$seed" \
    --num-samples 64 \
    --local-max-passes 3 \
    --lns-iterations 64
done

run_step "analyze five paired final LNS repeats" "$PYTHON_BIN" \
  evaluation/analyze_residual_lns_direct_repeats.py \
  --input-dir "$RESULTS_DIR" \
  --output-path "$SUMMARY_PATH"

if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "ERROR: expected final summary was not created: $SUMMARY_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - "$SUMMARY_PATH" <<'PY'
import json
import sys

results_path = sys.argv[1]
with open(results_path, "r", encoding="utf-8") as file:
    summary = json.load(file)
statistics = summary["statistics"]
mean_ratio = float(statistics["lns_mean_ratio"]["mean"])
sample_std = float(statistics["lns_mean_ratio"]["sample_std"])
p_value = float(statistics["paired_t_test_two_sided"]["p_value"])
winning_runs = sum(
    int(record["lns_better_than_heft_count"]) == 20
    for record in summary["paired_runs"]
)
print("FINAL_PIPELINE_RESULT")
print("method=Residual Best-of-64 + topological relocation + best-only LNS")
print(f"repeat_count={summary['repeat_count']}")
print(f"overall_mean_ratio={mean_ratio:.12f}")
print(f"overall_mean_ratio_6dp={mean_ratio:.6f}")
print(f"sample_std={sample_std:.12f}")
print(f"sample_std_6dp={sample_std:.6f}")
print(f"outperform_heft_scenarios=20/20 in {winning_runs}/5 repeats")
print(f"paired_p_value={p_value:.12e}")
print(f"results_path={results_path}")
print("FINAL_PIPELINE_COMPLETE")
PY
