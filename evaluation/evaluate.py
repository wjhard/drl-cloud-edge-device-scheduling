from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
from baselines.hybrid_scheduler import HybridScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from evaluation.generate_validation_scenarios import generate_scenarios
from policies.rl_scheduler import RLScheduler


DEFAULT_CONFIG = "training/configs/ppo_mlp_baseline.yaml"


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


def _resolve_model_zip_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    zip_path = path if path.suffix == ".zip" else Path(f"{path}.zip")
    return zip_path.resolve()


def _print_model_path_diagnostic(stage: str, model_path: str | Path) -> None:
    model_path_abs = Path(model_path).resolve()
    model_zip_path = _resolve_model_zip_path(model_path)
    mtime = os.path.getmtime(model_zip_path)
    mtime_text = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[evaluate:model:{stage}] model_path_arg_abs={model_path_abs}")
    print(f"[evaluate:model:{stage}] model_zip_abs={model_zip_path}")
    print(f"[evaluate:model:{stage}] model_zip_mtime_epoch={mtime:.6f}")
    print(f"[evaluate:model:{stage}] model_zip_mtime_local={mtime_text}")


def evaluate(
    config_path: str | Path,
    model_path: str | Path,
    num_samples: int = 1,
    sampling_deterministic_fallback: bool = True,
    results_path_override: str | Path | None = None,
    include_hybrid: bool = False,
    hybrid_task_threshold: int = 15,
    hybrid_milp_timeout_seconds: float = 10.0,
    sampling_seed: int | None = None,
) -> dict:
    config = _load_config(config_path)
    env_config = config["env"]
    evaluation_config = config["evaluation"]

    scenarios_dir = Path(evaluation_config["scenarios_dir"])
    scenario_paths = _ensure_scenarios(config_path, scenarios_dir)
    if not scenario_paths:
        raise RuntimeError(f"no validation scenarios found in {scenarios_dir}")

    heft_scheduler = HEFTScheduler()
    _print_model_path_diagnostic("before_load", model_path)
    rl_scheduler = RLScheduler(
        model_path=model_path,
        max_tasks=int(env_config["max_tasks_padding"]),
        deterministic=True,
        reward_mode=str(env_config.get("reward_mode", "raw")),
        normalize_observations=bool(env_config.get("normalize_observations", False)),
        include_upward_rank_feature=bool(env_config.get("include_upward_rank_feature", False)),
        scheduler_mode=str(env_config.get("scheduler_mode", "joint")),
        num_samples=num_samples,
        sampling_deterministic_fallback=sampling_deterministic_fallback,
    )
    _print_model_path_diagnostic("after_load", model_path)
    hybrid_scheduler = (
        HybridScheduler(
            residual_scheduler=rl_scheduler,
            task_threshold=hybrid_task_threshold,
            milp_time_limit_seconds=hybrid_milp_timeout_seconds,
        )
        if include_hybrid
        else None
    )

    records: list[dict] = []
    for scenario_index, scenario_path in enumerate(scenario_paths):
        dag = load_dag_from_json(scenario_path)

        heft_resource_config = load_resource_config(env_config["resource_config_path"])
        heft_schedule = heft_scheduler.schedule(dag, heft_resource_config)
        heft_makespan = heft_scheduler.compute_makespan(heft_schedule)

        scenario_seed = sampling_seed + scenario_index if sampling_seed is not None else None
        if scenario_seed is not None:
            _set_sampling_seed(scenario_seed)
        rl_resource_config = load_resource_config(env_config["resource_config_path"])
        rl_schedule = rl_scheduler.schedule(dag, rl_resource_config)
        rl_makespan = rl_scheduler.compute_makespan(rl_schedule)

        ratio = rl_makespan / heft_makespan if heft_makespan > 0 else 0.0
        record = {
            "scenario": scenario_path.name,
            "num_tasks": dag.graph.number_of_nodes(),
            "heft_makespan": heft_makespan,
            "rl_makespan": rl_makespan,
            "ratio": ratio,
        }
        if hybrid_scheduler is not None:
            if scenario_seed is not None:
                _set_sampling_seed(scenario_seed)
            hybrid_resource_config = load_resource_config(env_config["resource_config_path"])
            hybrid_schedule = hybrid_scheduler.schedule(dag, hybrid_resource_config)
            hybrid_makespan = hybrid_scheduler.compute_makespan(hybrid_schedule)
            decision = hybrid_scheduler.last_decision
            if decision is None:
                raise RuntimeError("HybridScheduler did not record its routing decision")
            record.update(
                {
                    "hybrid_makespan": hybrid_makespan,
                    "hybrid_ratio": hybrid_makespan / heft_makespan if heft_makespan > 0 else 0.0,
                    "hybrid_selected_method": decision.selected_method,
                    "hybrid_reason": decision.reason,
                    "hybrid_milp_status": decision.milp_status,
                    "hybrid_milp_proven_optimal": decision.milp_proven_optimal,
                    "hybrid_milp_makespan": decision.milp_makespan,
                    "hybrid_milp_best_bound": decision.milp_best_bound,
                    "hybrid_milp_solve_time_seconds": decision.milp_solve_time_seconds,
                    "hybrid_milp_error": decision.milp_error,
                }
            )
        records.append(record)

    ratios = [record["ratio"] for record in records]
    by_size: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_size[int(record["num_tasks"])].append(float(record["ratio"]))

    grouped = {
        str(task_size): {
            "count": len(size_ratios),
            "mean_ratio": _mean(size_ratios),
            "std_ratio": _std(size_ratios),
        }
        for task_size, size_ratios in sorted(by_size.items())
    }
    if include_hybrid:
        hybrid_by_size: dict[int, list[float]] = defaultdict(list)
        for record in records:
            hybrid_by_size[int(record["num_tasks"])].append(float(record["hybrid_ratio"]))
        for task_size, size_ratios in hybrid_by_size.items():
            grouped[str(task_size)]["hybrid_mean_ratio"] = _mean(size_ratios)
            grouped[str(task_size)]["hybrid_std_ratio"] = _std(size_ratios)

    summary = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "scenario_count": len(records),
        "num_samples": num_samples,
        "sampling_deterministic_fallback": sampling_deterministic_fallback,
        "sampling_seed": sampling_seed,
        "hybrid": {
            "enabled": include_hybrid,
            "task_threshold": hybrid_task_threshold if include_hybrid else None,
            "milp_timeout_seconds": hybrid_milp_timeout_seconds if include_hybrid else None,
            "requires_proven_optimal": True if include_hybrid else None,
        },
        "overall": {
            "mean_ratio": _mean(ratios),
            "std_ratio": _std(ratios),
        },
        "by_task_size": grouped,
        "scenarios": records,
    }
    if include_hybrid:
        hybrid_ratios = [float(record["hybrid_ratio"]) for record in records]
        summary["overall"]["hybrid_mean_ratio"] = _mean(hybrid_ratios)
        summary["overall"]["hybrid_std_ratio"] = _std(hybrid_ratios)

    results_path = Path(results_path_override) if results_path_override is not None else Path(evaluation_config["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary, results_path)
    return summary


def _print_summary(summary: dict, results_path: Path) -> None:
    print("evaluation summary")
    print(f"results_path: {results_path}")
    print(
        f"overall mean_ratio={summary['overall']['mean_ratio']:.6f}, "
        f"std={summary['overall']['std_ratio']:.6f}"
    )
    if summary.get("hybrid", {}).get("enabled"):
        print(
            f"overall hybrid_mean_ratio={summary['overall']['hybrid_mean_ratio']:.6f}, "
            f"std={summary['overall']['hybrid_std_ratio']:.6f}"
        )
    print("by task size:")
    hybrid_enabled = summary.get("hybrid", {}).get("enabled", False)
    print(
        "task_size | count | RL mean | RL std | Hybrid mean | Hybrid std"
        if hybrid_enabled
        else "task_size | count | mean_ratio | std"
    )
    for task_size, stats in summary["by_task_size"].items():
        if hybrid_enabled:
            print(
                f"{task_size:>9} | {stats['count']:>5} | "
                f"{stats['mean_ratio']:.6f} | {stats['std_ratio']:.6f} | "
                f"{stats['hybrid_mean_ratio']:.6f} | {stats['hybrid_std_ratio']:.6f}"
            )
        else:
            print(
                f"{task_size:>9} | {stats['count']:>5} | "
                f"{stats['mean_ratio']:.6f} | {stats['std_ratio']:.6f}"
            )
    print("scenario details:")
    print(
        "scenario | tasks | HEFT | RL | RL/HEFT | Hybrid | Hybrid/HEFT | method | reason"
        if hybrid_enabled
        else "scenario | tasks | HEFT | RL | ratio"
    )
    for record in summary["scenarios"]:
        if hybrid_enabled:
            print(
                f"{record['scenario']} | {record['num_tasks']} | "
                f"{record['heft_makespan']:.6f} | {record['rl_makespan']:.6f} | "
                f"{record['ratio']:.6f} | {record['hybrid_makespan']:.6f} | "
                f"{record['hybrid_ratio']:.6f} | {record['hybrid_selected_method']} | "
                f"{record['hybrid_reason']}"
            )
        else:
            print(
                f"{record['scenario']} | {record['num_tasks']} | "
                f"{record['heft_makespan']:.6f} | {record['rl_makespan']:.6f} | "
                f"{record['ratio']:.6f}"
            )


def _set_sampling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL scheduler against HEFT.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument("--model-path", required=True, help="Path to a saved MaskablePPO model.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of stochastic schedules to sample.")
    parser.add_argument("--results-path", default=None, help="Optional result JSON override.")
    parser.add_argument("--include-hybrid", action="store_true")
    parser.add_argument("--hybrid-task-threshold", type=int, default=15)
    parser.add_argument("--hybrid-milp-timeout", type=float, default=10.0)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument(
        "--no-deterministic-fallback",
        action="store_true",
        help="Disable deterministic candidate when num_samples > 1.",
    )
    args = parser.parse_args()
    evaluate(
        args.config,
        args.model_path,
        num_samples=args.num_samples,
        sampling_deterministic_fallback=not args.no_deterministic_fallback,
        results_path_override=args.results_path,
        include_hybrid=args.include_hybrid,
        hybrid_task_threshold=args.hybrid_task_threshold,
        hybrid_milp_timeout_seconds=args.hybrid_milp_timeout,
        sampling_seed=args.sampling_seed,
    )


if __name__ == "__main__":
    main()


