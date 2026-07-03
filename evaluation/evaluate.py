from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
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


def evaluate(config_path: str | Path, model_path: str | Path) -> dict:
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
    )
    _print_model_path_diagnostic("after_load", model_path)

    records: list[dict] = []
    for scenario_path in scenario_paths:
        dag = load_dag_from_json(scenario_path)

        heft_resource_config = load_resource_config(env_config["resource_config_path"])
        heft_schedule = heft_scheduler.schedule(dag, heft_resource_config)
        heft_makespan = heft_scheduler.compute_makespan(heft_schedule)

        rl_resource_config = load_resource_config(env_config["resource_config_path"])
        rl_schedule = rl_scheduler.schedule(dag, rl_resource_config)
        rl_makespan = rl_scheduler.compute_makespan(rl_schedule)

        ratio = rl_makespan / heft_makespan if heft_makespan > 0 else 0.0
        records.append(
            {
                "scenario": scenario_path.name,
                "num_tasks": dag.graph.number_of_nodes(),
                "heft_makespan": heft_makespan,
                "rl_makespan": rl_makespan,
                "ratio": ratio,
            }
        )

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

    summary = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "scenario_count": len(records),
        "overall": {
            "mean_ratio": _mean(ratios),
            "std_ratio": _std(ratios),
        },
        "by_task_size": grouped,
        "scenarios": records,
    }

    results_path = Path(evaluation_config["results_path"])
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
    print("by task size:")
    print("task_size | count | mean_ratio | std")
    for task_size, stats in summary["by_task_size"].items():
        print(
            f"{task_size:>9} | {stats['count']:>5} | "
            f"{stats['mean_ratio']:.6f} | {stats['std_ratio']:.6f}"
        )
    print("scenario details:")
    print("scenario | tasks | HEFT | RL | ratio")
    for record in summary["scenarios"]:
        print(
            f"{record['scenario']} | {record['num_tasks']} | "
            f"{record['heft_makespan']:.6f} | {record['rl_makespan']:.6f} | "
            f"{record['ratio']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL scheduler against HEFT.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument("--model-path", required=True, help="Path to a saved MaskablePPO model.")
    args = parser.parse_args()
    evaluate(args.config, args.model_path)


if __name__ == "__main__":
    main()


