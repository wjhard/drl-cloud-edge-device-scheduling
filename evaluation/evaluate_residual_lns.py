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
from policies.residual_local_search_scheduler import ResidualLargeNeighborhoodScheduler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    config_path: str | Path,
    model_path: str | Path,
    results_path: str | Path,
    sampling_seed: int,
    num_samples: int = 64,
    local_max_passes: int = 3,
    lns_iterations: int = 64,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config = config["env"]
    scenario_paths = sorted(Path(config["evaluation"]["scenarios_dir"]).glob("*.json"))
    if not scenario_paths:
        raise RuntimeError("no validation scenarios found")
    scheduler = ResidualLargeNeighborhoodScheduler(
        model_path=model_path,
        max_tasks=int(env_config["max_tasks_padding"]),
        num_samples=num_samples,
        local_max_passes=local_max_passes,
        lns_iterations=lns_iterations,
        normalize_observations=bool(env_config.get("normalize_observations", False)),
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
        lns_schedule = scheduler.schedule(dag, load_resource_config(resources_path))
        residual_schedule = scheduler.local_scheduler.last_initial_schedule
        if (
            residual_schedule is None
            or scheduler.last_local_schedule is None
            or scheduler.last_stats is None
        ):
            raise RuntimeError("LNS scheduler did not expose paired phase results")
        residual_makespan = scheduler.compute_makespan(residual_schedule)
        local_makespan = scheduler.compute_makespan(scheduler.last_local_schedule)
        lns_makespan = scheduler.compute_makespan(lns_schedule)
        records.append(
            {
                "scenario": scenario_path.name,
                "num_tasks": dag.graph.number_of_nodes(),
                "scenario_seed": scenario_seed,
                "heft_makespan": heft_makespan,
                "residual_bestof64_makespan": residual_makespan,
                "local_search_makespan": local_makespan,
                "lns_makespan": lns_makespan,
                "residual_bestof64_ratio": residual_makespan / heft_makespan,
                "local_search_ratio": local_makespan / heft_makespan,
                "lns_ratio": lns_makespan / heft_makespan,
                "accepted_repairs": scheduler.last_stats.accepted_repairs,
                "evaluated_neighbors": scheduler.last_stats.evaluated_neighbors,
            }
        )

    residual_ratios = [float(row["residual_bestof64_ratio"]) for row in records]
    local_ratios = [float(row["local_search_ratio"]) for row in records]
    lns_ratios = [float(row["lns_ratio"]) for row in records]
    elapsed = time.perf_counter() - started
    summary = {
        "method": "Residual Best-of-64 + local search + destroy-repair LNS",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "sampling_seed": sampling_seed,
        "num_samples": num_samples,
        "local_max_passes": local_max_passes,
        "lns_iterations": lns_iterations,
        "scenario_count": len(records),
        "elapsed_seconds": elapsed,
        "overall": {
            "residual_bestof64_mean_ratio": float(statistics.fmean(residual_ratios)),
            "local_search_mean_ratio": float(statistics.fmean(local_ratios)),
            "lns_mean_ratio": float(statistics.fmean(lns_ratios)),
            "paired_mean_difference_lns_minus_residual": float(
                statistics.fmean(new - old for new, old in zip(lns_ratios, residual_ratios))
            ),
            "paired_mean_difference_lns_minus_local": float(
                statistics.fmean(new - old for new, old in zip(lns_ratios, local_ratios))
            ),
            "residual_bestof64_better_than_heft_count": sum(
                value < 1.0 for value in residual_ratios
            ),
            "local_search_better_than_heft_count": sum(value < 1.0 for value in local_ratios),
            "lns_better_than_heft_count": sum(value < 1.0 for value in lns_ratios),
            "improved_scenario_count": sum(new < old - 1e-12 for new, old in zip(lns_ratios, local_ratios)),
        },
        "scenarios": records,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RESIDUAL_LNS_EVALUATION")
    print(f"sampling_seed={sampling_seed}")
    print(f"residual_bestof64_mean_ratio={summary['overall']['residual_bestof64_mean_ratio']:.12f}")
    print(f"local_search_mean_ratio={summary['overall']['local_search_mean_ratio']:.12f}")
    print(f"lns_mean_ratio={summary['overall']['lns_mean_ratio']:.12f}")
    print(
        "paired_difference_lns_minus_residual="
        f"{summary['overall']['paired_mean_difference_lns_minus_residual']:+.12f}"
    )
    print(
        "paired_difference_lns_minus_local="
        f"{summary['overall']['paired_mean_difference_lns_minus_local']:+.12f}"
    )
    print(f"better_than_heft={summary['overall']['lns_better_than_heft_count']}/{len(records)}")
    print(f"improved_scenarios={summary['overall']['improved_scenario_count']}/{len(records)}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual local search plus LNS.")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--local-max-passes", type=int, default=3)
    parser.add_argument("--lns-iterations", type=int, default=64)
    args = parser.parse_args()
    evaluate(
        args.config,
        args.model_path,
        args.results_path,
        sampling_seed=args.sampling_seed if args.sampling_seed is not None else secrets.randbits(31),
        num_samples=args.num_samples,
        local_max_passes=args.local_max_passes,
        lns_iterations=args.lns_iterations,
    )


if __name__ == "__main__":
    main()
