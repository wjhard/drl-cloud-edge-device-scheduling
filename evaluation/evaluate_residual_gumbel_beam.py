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
from policies.residual_gumbel_beam_scheduler import ResidualGumbelBeamScheduler
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
    beam_width: int = 16,
    lns_iterations: int = 64,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config = config["env"]
    scenario_paths = sorted(Path(config["evaluation"]["scenarios_dir"]).glob("*.json"))
    if not scenario_paths:
        raise RuntimeError("no validation scenarios found")
    common = {
        "model_path": model_path,
        "max_tasks": int(env_config["max_tasks_padding"]),
        "lns_iterations": lns_iterations,
        "normalize_observations": bool(env_config.get("normalize_observations", False)),
    }
    baseline = ResidualLargeNeighborhoodScheduler(
        **common,
        num_samples=64,
        local_max_passes=3,
    )
    beam = ResidualGumbelBeamScheduler(
        **common,
        beam_width=beam_width,
        local_max_passes=3,
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
        baseline_schedule = baseline.schedule(dag, load_resource_config(resources_path))
        baseline_makespan = baseline.compute_makespan(baseline_schedule)

        _set_seed(scenario_seed)
        beam_schedule = beam.schedule(dag, load_resource_config(resources_path))
        beam_makespan = beam.compute_makespan(beam_schedule)
        records.append(
            {
                "scenario": scenario_path.name,
                "num_tasks": dag.graph.number_of_nodes(),
                "scenario_seed": scenario_seed,
                "heft_makespan": heft_makespan,
                "bestof64_lns_makespan": baseline_makespan,
                "gumbel_beam_lns_makespan": beam_makespan,
                "bestof64_lns_ratio": baseline_makespan / heft_makespan,
                "gumbel_beam_lns_ratio": beam_makespan / heft_makespan,
                "beam_unique_complete_orders": beam.last_unique_complete_orders,
            }
        )

    baseline_ratios = [float(row["bestof64_lns_ratio"]) for row in records]
    beam_ratios = [float(row["gumbel_beam_lns_ratio"]) for row in records]
    elapsed = time.perf_counter() - started
    summary = {
        "method": "Residual Gumbel-diverse beam + local search + LNS",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "sampling_seed": sampling_seed,
        "beam_width": beam_width,
        "lns_iterations": lns_iterations,
        "scenario_count": len(records),
        "elapsed_seconds": elapsed,
        "overall": {
            "bestof64_lns_mean_ratio": float(statistics.fmean(baseline_ratios)),
            "gumbel_beam_lns_mean_ratio": float(statistics.fmean(beam_ratios)),
            "paired_mean_difference_beam_minus_bestof64": float(
                statistics.fmean(new - old for new, old in zip(beam_ratios, baseline_ratios))
            ),
            "bestof64_lns_better_than_heft_count": sum(value < 1.0 for value in baseline_ratios),
            "gumbel_beam_lns_better_than_heft_count": sum(value < 1.0 for value in beam_ratios),
            "beam_better_scenario_count": sum(new < old - 1e-12 for new, old in zip(beam_ratios, baseline_ratios)),
            "mean_unique_complete_orders": float(
                statistics.fmean(row["beam_unique_complete_orders"] for row in records)
            ),
        },
        "scenarios": records,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RESIDUAL_GUMBEL_BEAM_EVALUATION")
    print(f"sampling_seed={sampling_seed}")
    print(f"beam_width={beam_width}")
    print(f"bestof64_lns_mean_ratio={summary['overall']['bestof64_lns_mean_ratio']:.12f}")
    print(f"gumbel_beam_lns_mean_ratio={summary['overall']['gumbel_beam_lns_mean_ratio']:.12f}")
    print(
        "paired_difference_beam_minus_bestof64="
        f"{summary['overall']['paired_mean_difference_beam_minus_bestof64']:+.12f}"
    )
    print(f"beam_better_scenarios={summary['overall']['beam_better_scenario_count']}/{len(records)}")
    print(f"mean_unique_complete_orders={summary['overall']['mean_unique_complete_orders']:.3f}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gumbel beam against Best-of-64 LNS.")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--lns-iterations", type=int, default=64)
    args = parser.parse_args()
    evaluate(
        args.config,
        args.model_path,
        args.results_path,
        sampling_seed=args.sampling_seed if args.sampling_seed is not None else secrets.randbits(31),
        beam_width=args.beam_width,
        lns_iterations=args.lns_iterations,
    )


if __name__ == "__main__":
    main()
