from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from policies.rl_scheduler import RLScheduler


DEFAULT_MODEL_PATH = "training/checkpoints/ppo_mlp_baseline"
DEFAULT_MAX_TASKS = 30
TASK_SIZES = [10, 12, 14, 15, 17, 19, 21, 23, 24, 25]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RL scheduler action/resource distribution.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    parser.add_argument("--edge-density", type=float, default=0.35)
    parser.add_argument("--seed-start", type=int, default=424200)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    model_zip = Path(f"{model_path}.zip") if model_path.suffix != ".zip" else model_path
    print("ACTION_DISTRIBUTION_CHECK")
    print(f"model_path_arg_abs={model_path}")
    print(f"model_zip_abs={model_zip.resolve()}")
    print(f"model_zip_size_bytes={model_zip.stat().st_size}")
    print(f"model_zip_mtime_epoch={model_zip.stat().st_mtime:.6f}")

    scheduler = RLScheduler(
        model_path=args.model_path,
        max_tasks=args.max_tasks,
        deterministic=True,
    )

    total_counts: Counter[str] = Counter()
    total_tasks = 0
    print("per_dag_distribution:")
    for index, task_size in enumerate(TASK_SIZES):
        seed = args.seed_start + index
        dag = generate_random_dag(
            num_tasks=task_size,
            edge_density=args.edge_density,
            seed=seed,
        )
        resource_config = load_resource_config(default_resource_config_path())
        schedule = scheduler.schedule(dag, resource_config)
        counts = Counter(resource_id for resource_id, _, _ in schedule.values())
        total_counts.update(counts)
        total_tasks += len(schedule)
        ordered_counts = ", ".join(
            f"{resource.id}={counts.get(resource.id, 0)}"
            for resource in resource_config.resources
        )
        print(
            f"  dag_index={index} seed={seed} task_size={task_size} "
            f"total_tasks={len(schedule)} {ordered_counts}"
        )

    print("aggregate_distribution:")
    resource_config = load_resource_config(default_resource_config_path())
    for resource in resource_config.resources:
        count = total_counts.get(resource.id, 0)
        ratio = count / total_tasks if total_tasks else 0.0
        print(f"  {resource.id}: count={count}, ratio={ratio:.6f}")
    print(f"total_scheduled_tasks={total_tasks}")


if __name__ == "__main__":
    main()