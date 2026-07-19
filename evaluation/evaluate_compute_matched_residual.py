from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import secrets
import statistics
import sys
import time

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from policies.rl_scheduler import RLScheduler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    config_path: str | Path,
    model_path: str | Path,
    results_path: str | Path,
    sampling_seed: int,
    num_samples: int,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config = config["env"]
    scenario_paths = sorted(Path(config["evaluation"]["scenarios_dir"]).glob("*.json"))
    if not scenario_paths:
        raise RuntimeError("no validation scenarios found")

    scheduler = RLScheduler(
        model_path=model_path,
        max_tasks=int(env_config["max_tasks_padding"]),
        deterministic=True,
        reward_mode=str(env_config.get("reward_mode", "raw")),
        normalize_observations=bool(env_config.get("normalize_observations", False)),
        include_upward_rank_feature=bool(env_config.get("include_upward_rank_feature", False)),
        scheduler_mode=str(env_config.get("scheduler_mode", "residual")),
        num_samples=num_samples,
        sampling_deterministic_fallback=True,
    )
    heft = HEFTScheduler()
    records: list[dict] = []
    started = time.perf_counter()

    for scenario_index, scenario_path in enumerate(scenario_paths):
        dag = load_dag_from_json(scenario_path)
        resources_path = env_config["resource_config_path"]
        heft_makespan = heft.compute_makespan(
            heft.schedule(dag, load_resource_config(resources_path))
        )
        scenario_seed = sampling_seed + scenario_index
        _set_seed(scenario_seed)
        schedule = scheduler.schedule(dag, load_resource_config(resources_path))
        makespan = scheduler.compute_makespan(schedule)
        records.append(
            {
                "scenario": scenario_path.name,
                "num_tasks": dag.graph.number_of_nodes(),
                "scenario_seed": scenario_seed,
                "heft_makespan": heft_makespan,
                "residual_makespan": makespan,
                "ratio": makespan / heft_makespan,
            }
        )

    elapsed = time.perf_counter() - started
    ratios = [float(row["ratio"]) for row in records]
    summary = {
        "method": f"Residual Best-of-{num_samples}",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "sampling_seed": sampling_seed,
        "num_samples": num_samples,
        "scenario_count": len(records),
        "elapsed_seconds": elapsed,
        "overall": {
            "mean_ratio": float(statistics.fmean(ratios)),
            "better_than_heft_count": sum(value < 1.0 for value in ratios),
        },
        "scenarios": records,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("COMPUTE_MATCHED_RESIDUAL_EVALUATION")
    print(f"sampling_seed={sampling_seed}")
    print(f"num_samples={num_samples}")
    print(f"mean_ratio={summary['overall']['mean_ratio']:.12f}")
    print(f"better_than_heft={summary['overall']['better_than_heft_count']}/{len(records)}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a compute-matched pure Residual sampler.")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, required=True)
    args = parser.parse_args()
    evaluate(
        args.config,
        args.model_path,
        args.results_path,
        sampling_seed=args.sampling_seed if args.sampling_seed is not None else secrets.randbits(31),
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
