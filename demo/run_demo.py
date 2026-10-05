"""Run the shipped checkpoint on one DAG; no training is performed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo/output"))
    parser.add_argument("--preset", choices=("fast", "balanced", "quality"), default="quality")
    parser.add_argument("--time-budget-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    if args.time_budget_seconds <= 0:
        parser.error("--time-budget-seconds must be positive")

    scenario = "evaluation/scenarios/scenario_10_0.json"
    checkpoint = PROJECT_ROOT / "training/checkpoints/ppo_mlp_residual.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"shipped checkpoint is missing: {checkpoint}")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in ("heft", "adaptive"):
        subprocess.run(
            [
                sys.executable, "scripts/schedule.py",
                "--input", scenario,
                "--method", method,
                "--preset", args.preset,
                "--time-budget-seconds", str(args.time_budget_seconds),
                "--seed", str(args.seed),
                "--output", str(output_dir / f"{method}.json"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    adaptive = json.loads((output_dir / "adaptive.json").read_text(encoding="utf-8"))
    print(f"demo_ratio_to_heft={adaptive['result']['ratio_to_heft']:.12f}", flush=True)
    print(f"demo_valid={adaptive['validation']['valid']}", flush=True)
    if not args.skip_plot:
        subprocess.run(
            [
                sys.executable, "scripts/visualize_schedule.py",
                "--scenario", scenario,
                "--result", str(output_dir / "adaptive.json"),
                "--output", str(output_dir / "schedule.png"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print(f"demo_output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
