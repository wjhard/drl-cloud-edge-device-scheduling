from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import generate_random_dag, save_dag_to_json


DEFAULT_CONFIG = "training/configs/ppo_mlp_baseline.yaml"


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _scenario_counts(total: int, sizes: list[int]) -> dict[int, int]:
    if total <= 0:
        raise ValueError("num_validation_scenarios must be positive")
    if not sizes:
        raise ValueError("validation_task_sizes must not be empty")

    base_count = total // len(sizes)
    remainder = total % len(sizes)
    return {
        size: base_count + (1 if index < remainder else 0)
        for index, size in enumerate(sizes)
    }


def generate_scenarios(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
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

    for task_size in sizes:
        for index in range(counts[task_size]):
            dag = generate_random_dag(
                num_tasks=task_size,
                edge_density=0.35,
                seed=seed,
            )
            output_path = scenarios_dir / f"scenario_{task_size}_{index}.json"
            save_dag_to_json(dag, output_path)
            generated_paths.append(output_path)
            seed += 1

    distribution = Counter(path.stem.split("_")[1] for path in generated_paths)
    print(f"generated {len(generated_paths)} validation scenarios in {scenarios_dir}")
    print("task size distribution:")
    for task_size in sorted(distribution, key=lambda item: int(item)):
        print(f"  {task_size}: {distribution[task_size]}")
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed validation DAG scenarios.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to training/evaluation YAML config.")
    args = parser.parse_args()
    generate_scenarios(args.config)


if __name__ == "__main__":
    main()

