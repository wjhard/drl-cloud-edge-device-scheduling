from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from evaluation.generate_validation_scenarios import generate_scenarios
from policies.rl_scheduler import RLScheduler


DEFAULT_CONFIG = "training/configs/ppo_mlp_ranked.yaml"


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _ensure_scenarios(config_path: str | Path, scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.exists() or not any(scenarios_dir.glob("*.json")):
        generate_scenarios(config_path)
    return sorted(scenarios_dir.glob("*.json"))


def _print_model_path_diagnostic(model_path: str | Path) -> None:
    model_path_abs = Path(model_path).resolve()
    model_zip_path = model_path_abs if model_path_abs.suffix == ".zip" else Path(f"{model_path_abs}.zip")
    mtime = os.path.getmtime(model_zip_path)
    print(f"model_path_arg_abs={model_path_abs}")
    print(f"model_zip_abs={model_zip_path}")
    print(f"model_zip_mtime_epoch={mtime:.6f}")
    print(f"model_zip_mtime_local={datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of RL schedulers with best-of-N per model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", action="append", required=True, help="Model path. Repeat for each ensemble member.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--results-path", default="evaluation/results/summary_mlp_ranked_ensemble_bestof8.json")
    parser.add_argument(
        "--deterministic-fallback",
        action="store_true",
        help="Also include each model's deterministic schedule candidate.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = _load_config(args.config)
    env_config = config["env"]
    evaluation_config = config["evaluation"]
    scenario_paths = _ensure_scenarios(args.config, Path(evaluation_config["scenarios_dir"]))

    print("ENSEMBLE_BESTOFN_EVALUATION")
    print(f"config={Path(args.config).resolve()}")
    print(f"num_models={len(args.model_path)}")
    print(f"num_samples_per_model={args.num_samples}")
    print(f"deterministic_fallback={args.deterministic_fallback}")
    print(f"seed={args.seed}")
    print(f"scenario_count={len(scenario_paths)}")
    for model_path in args.model_path:
        _print_model_path_diagnostic(model_path)

    schedulers = [
        RLScheduler(
            model_path=model_path,
            max_tasks=int(env_config["max_tasks_padding"]),
            deterministic=True,
            reward_mode=str(env_config.get("reward_mode", "raw")),
            normalize_observations=bool(env_config.get("normalize_observations", False)),
            include_upward_rank_feature=bool(env_config.get("include_upward_rank_feature", False)),
            scheduler_mode=str(env_config.get("scheduler_mode", "joint")),
            num_samples=args.num_samples,
            sampling_deterministic_fallback=args.deterministic_fallback,
        )
        for model_path in args.model_path
    ]

    heft_scheduler = HEFTScheduler()
    records: list[dict] = []
    start_time = time.perf_counter()

    for scenario_index, scenario_path in enumerate(scenario_paths):
        dag = load_dag_from_json(scenario_path)

        heft_resource_config = load_resource_config(env_config["resource_config_path"])
        heft_schedule = heft_scheduler.schedule(dag, heft_resource_config)
        heft_makespan = heft_scheduler.compute_makespan(heft_schedule)

        model_records = []
        best_model_index = -1
        best_makespan = float("inf")
        for model_index, scheduler in enumerate(schedulers):
            model_seed = args.seed + scenario_index * 1000 + model_index
            random.seed(model_seed)
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)

            resource_config = load_resource_config(env_config["resource_config_path"])
            schedule = scheduler.schedule(dag, resource_config)
            makespan = scheduler.compute_makespan(schedule)
            model_records.append(
                {
                    "model_index": model_index,
                    "model_path": args.model_path[model_index],
                    "makespan": makespan,
                    "ratio": makespan / heft_makespan if heft_makespan > 0 else 0.0,
                }
            )
            if makespan < best_makespan:
                best_makespan = makespan
                best_model_index = model_index

        records.append(
            {
                "scenario": scenario_path.name,
                "num_tasks": dag.graph.number_of_nodes(),
                "heft_makespan": heft_makespan,
                "rl_makespan": best_makespan,
                "ratio": best_makespan / heft_makespan if heft_makespan > 0 else 0.0,
                "best_model_index": best_model_index,
                "model_candidates": model_records,
            }
        )

    elapsed_seconds = time.perf_counter() - start_time
    ratios = [float(record["ratio"]) for record in records]
    by_size: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_size[int(record["num_tasks"])].append(float(record["ratio"]))

    summary = {
        "config_path": str(args.config),
        "model_paths": args.model_path,
        "scenario_count": len(records),
        "num_samples_per_model": args.num_samples,
        "deterministic_fallback": args.deterministic_fallback,
        "elapsed_seconds": elapsed_seconds,
        "overall": {
            "mean_ratio": _mean(ratios),
            "std_ratio": _std(ratios),
            "better_than_heft_count": sum(1 for ratio in ratios if ratio < 1.0),
        },
        "by_task_size": {
            str(task_size): {
                "count": len(size_ratios),
                "mean_ratio": _mean(size_ratios),
                "std_ratio": _std(size_ratios),
            }
            for task_size, size_ratios in sorted(by_size.items())
        },
        "scenarios": records,
    }
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("ensemble evaluation summary")
    print(f"results_path: {results_path}")
    print(f"elapsed_seconds={elapsed_seconds:.6f}")
    print(f"overall mean_ratio={summary['overall']['mean_ratio']:.6f}, std={summary['overall']['std_ratio']:.6f}")
    print(f"better_than_heft_count={summary['overall']['better_than_heft_count']}")


if __name__ == "__main__":
    main()
