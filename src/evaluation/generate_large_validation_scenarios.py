from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import generate_random_dag, save_dag_to_json
from evaluation.generate_validation_scenarios import _scenario_counts


DEFAULT_CONFIG = "training/configs/ppo_mlp_normalized_large.yaml"
DEFAULT_EDGE_DENSITY_MIN = 0.4
DEFAULT_EDGE_DENSITY_MAX = 0.6


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_large_scenarios(
    config_path: str | Path = DEFAULT_CONFIG,
    edge_density_min: float = DEFAULT_EDGE_DENSITY_MIN,
    edge_density_max: float = DEFAULT_EDGE_DENSITY_MAX,
) -> list[Path]:
    if not 0.0 <= edge_density_min <= edge_density_max <= 1.0:
        raise ValueError("edge density bounds must satisfy 0 <= min <= max <= 1")

    config = _load_config(config_path)
    evaluation_config = config["evaluation"]
    scenarios_dir = Path(evaluation_config["scenarios_dir"])
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in scenarios_dir.glob("scenario_*.json"):
        existing_file.unlink()

    sizes = [int(size) for size in evaluation_config["validation_task_sizes"]]
    counts = _scenario_counts(int(evaluation_config["num_validation_scenarios"]), sizes)
    seed = int(evaluation_config["validation_seed_start"])
    generated_paths: list[Path] = []
    density_values: list[float] = []

    for task_size in sizes:
        for index in range(counts[task_size]):
            density_rng = random.Random(seed)
            edge_density = density_rng.uniform(edge_density_min, edge_density_max)
            dag = generate_random_dag(
                num_tasks=task_size,
                edge_density=edge_density,
                seed=seed,
            )
            output_path = scenarios_dir / f"scenario_{task_size}_{index}.json"
            save_dag_to_json(dag, output_path)
            generated_paths.append(output_path)
            density_values.append(edge_density)
            seed += 1

    distribution = Counter(path.stem.split("_")[1] for path in generated_paths)
    print(f"generated {len(generated_paths)} large validation scenarios in {scenarios_dir}")
    print(f"seed_start={evaluation_config['validation_seed_start']}")
    print(f"edge_density_min={edge_density_min:.6f}")
    print(f"edge_density_max={edge_density_max:.6f}")
    if density_values:
        print(f"actual_edge_density_min={min(density_values):.6f}")
        print(f"actual_edge_density_max={max(density_values):.6f}")
    print("task size distribution:")
    for task_size in sorted(distribution, key=lambda item: int(item)):
        print(f"  {task_size}: {distribution[task_size]}")
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed large validation DAG scenarios.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to training/evaluation YAML config.")
    parser.add_argument("--edge-density-min", type=float, default=DEFAULT_EDGE_DENSITY_MIN)
    parser.add_argument("--edge-density-max", type=float, default=DEFAULT_EDGE_DENSITY_MAX)
    args = parser.parse_args()
    generate_large_scenarios(
        config_path=args.config,
        edge_density_min=args.edge_density_min,
        edge_density_max=args.edge_density_max,
    )


if __name__ == "__main__":
    main()
