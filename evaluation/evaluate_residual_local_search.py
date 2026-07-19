from __future__ import annotations

import argparse
from collections import defaultdict
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
from policies.residual_local_search_scheduler import ResidualLocalSearchScheduler


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


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
    max_passes: int = 3,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config = config["env"]
    scenarios_dir = Path(config["evaluation"]["scenarios_dir"])
    scenario_paths = sorted(scenarios_dir.glob("*.json"))
    if not scenario_paths:
        raise RuntimeError(f"no scenarios found in {scenarios_dir}")

    scheduler = ResidualLocalSearchScheduler(
        model_path=model_path,
        max_tasks=int(env_config["max_tasks_padding"]),
        num_samples=num_samples,
        max_passes=max_passes,
        normalize_observations=bool(env_config.get("normalize_observations", False)),
    )
    heft = HEFTScheduler()
    records: list[dict] = []
    started = time.perf_counter()

    for scenario_index, scenario_path in enumerate(scenario_paths):
        dag = load_dag_from_json(scenario_path)
        resources_path = env_config["resource_config_path"]
        heft_schedule = heft.schedule(dag, load_resource_config(resources_path))
        heft_makespan = heft.compute_makespan(heft_schedule)

        scenario_seed = sampling_seed + scenario_index
        _set_seed(scenario_seed)
        improved_schedule = scheduler.schedule(dag, load_resource_config(resources_path))
        if scheduler.last_initial_schedule is None or scheduler.last_stats is None:
            raise RuntimeError("local-search scheduler did not expose comparison data")
        initial_makespan = scheduler.compute_makespan(scheduler.last_initial_schedule)
        improved_makespan = scheduler.compute_makespan(improved_schedule)
        if improved_makespan > initial_makespan + 1e-9:
            raise RuntimeError("local search worsened its paired Best-of-N initial solution")
        if abs(initial_makespan - scheduler.last_stats.initial_makespan) > 1e-9:
            raise RuntimeError("task-order replay diverged from residual environment physics")

        records.append(
            {
                "scenario": scenario_path.name,
                "num_tasks": dag.graph.number_of_nodes(),
                "scenario_seed": scenario_seed,
                "heft_makespan": heft_makespan,
                "residual_bestof64_makespan": initial_makespan,
                "local_search_makespan": improved_makespan,
                "residual_bestof64_ratio": initial_makespan / heft_makespan,
                "local_search_ratio": improved_makespan / heft_makespan,
                "improvement_makespan": initial_makespan - improved_makespan,
                "evaluated_neighbors": scheduler.last_stats.evaluated_neighbors,
                "accepted_moves": scheduler.last_stats.accepted_moves,
            }
        )

    elapsed = time.perf_counter() - started
    baseline_ratios = [float(row["residual_bestof64_ratio"]) for row in records]
    improved_ratios = [float(row["local_search_ratio"]) for row in records]
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        by_size[int(row["num_tasks"])].append(row)

    summary = {
        "method": "Residual Best-of-64 + precedence-preserving relocation local search",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "sampling_seed": sampling_seed,
        "num_samples": num_samples,
        "max_passes": max_passes,
        "scenario_count": len(records),
        "elapsed_seconds": elapsed,
        "overall": {
            "residual_bestof64_mean_ratio": _mean(baseline_ratios),
            "local_search_mean_ratio": _mean(improved_ratios),
            "paired_mean_difference_local_minus_base": _mean(
                [new - old for new, old in zip(improved_ratios, baseline_ratios)]
            ),
            "residual_bestof64_better_than_heft_count": sum(value < 1.0 for value in baseline_ratios),
            "local_search_better_than_heft_count": sum(value < 1.0 for value in improved_ratios),
            "improved_scenario_count": sum(new < old - 1e-12 for new, old in zip(improved_ratios, baseline_ratios)),
        },
        "by_task_size": {
            str(size): {
                "count": len(rows),
                "residual_bestof64_mean_ratio": _mean(
                    [float(row["residual_bestof64_ratio"]) for row in rows]
                ),
                "local_search_mean_ratio": _mean([float(row["local_search_ratio"]) for row in rows]),
            }
            for size, rows in sorted(by_size.items())
        },
        "scenarios": records,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("RESIDUAL_LOCAL_SEARCH_EVALUATION")
    print(f"sampling_seed={sampling_seed}")
    print(f"num_samples={num_samples}")
    print(f"max_passes={max_passes}")
    print(f"base_mean_ratio={summary['overall']['residual_bestof64_mean_ratio']:.12f}")
    print(f"local_search_mean_ratio={summary['overall']['local_search_mean_ratio']:.12f}")
    print(
        "paired_difference_local_minus_base="
        f"{summary['overall']['paired_mean_difference_local_minus_base']:+.12f}"
    )
    print(
        "better_than_heft="
        f"{summary['overall']['local_search_better_than_heft_count']}/{len(records)}"
    )
    print(f"improved_scenarios={summary['overall']['improved_scenario_count']}/{len(records)}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual Best-of-N with local search.")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()
    sampling_seed = args.sampling_seed if args.sampling_seed is not None else secrets.randbits(31)
    evaluate(
        args.config,
        args.model_path,
        args.results_path,
        sampling_seed=sampling_seed,
        num_samples=args.num_samples,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
