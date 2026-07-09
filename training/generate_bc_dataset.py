from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from gymnasium.wrappers import FlattenObservation

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
from env.resource_config import load_resource_config
from env.scheduling_env import SchedulingEnv
from training.dag_curriculum import make_training_dag_generator


DEFAULT_CONFIG = "training/configs/ppo_mlp_normalized.yaml"
DEFAULT_OUTPUT = "training/bc_datasets/mlp_bc_dataset_normalized.npz"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_bc_dataset(
    config_path: str | Path = DEFAULT_CONFIG,
    output_path: str | Path = DEFAULT_OUTPUT,
    num_dags: int = 500,
    seed_start: int = 2_000_000,
) -> Path:
    if num_dags <= 0:
        raise ValueError("num_dags must be positive")

    config = _load_config(config_path)
    env_config = config["env"]
    train_config = config["training"]
    resource_config_path = _resolve_project_path(env_config["resource_config_path"])
    output = _resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dag_generator = make_training_dag_generator(
        min_tasks=int(env_config["min_tasks"]),
        max_tasks=int(env_config["max_tasks"]),
        edge_density_min=float(env_config["edge_density_min"]),
        edge_density_max=float(env_config["edge_density_max"]),
        base_seed=int(train_config.get("seed", 0)),
    )
    scheduler = HEFTScheduler()

    observations: list[np.ndarray] = []
    actions: list[int] = []
    action_masks: list[np.ndarray] = []
    dag_task_counts: list[int] = []

    for dag_index in range(num_dags):
        seed = seed_start + dag_index
        dag = dag_generator(seed=seed)

        heft_resource_config = load_resource_config(resource_config_path)
        heft_schedule = scheduler.schedule(dag, heft_resource_config)
        task_order = scheduler._task_order(dag, heft_resource_config)
        assignment = {
            task_id: resource_id
            for task_id, (resource_id, _, _) in heft_schedule.items()
        }

        env_resource_config = load_resource_config(resource_config_path)
        raw_env = SchedulingEnv(
            dag_generator_fn=lambda seed=None, fixed_dag=dag: fixed_dag,
            resource_config=env_resource_config,
            max_tasks=int(env_config["max_tasks_padding"]),
            reward_mode="raw",
            normalize_observations=True,
        )
        env = FlattenObservation(raw_env)
        observation, _ = env.reset()

        resource_index = {
            resource.id: index
            for index, resource in enumerate(env_resource_config.resources)
        }
        terminated = False
        for task_id in task_order:
            if task_id not in raw_env.ready_tasks:
                raise RuntimeError(
                    f"HEFT task {task_id} is not ready for DAG seed={seed}; "
                    f"ready_tasks={raw_env.ready_tasks}"
                )
            task_slot = raw_env.ready_tasks.index(task_id)
            action = task_slot * raw_env.num_resources + resource_index[assignment[task_id]]

            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(int(action))
            action_masks.append(raw_env.action_masks().astype(np.bool_))

            observation, _, terminated, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError(f"environment truncated while replaying HEFT for DAG seed={seed}")

        if not terminated:
            raise RuntimeError(f"HEFT replay did not terminate for DAG seed={seed}")
        dag_task_counts.append(dag.graph.number_of_nodes())

    observation_array = np.stack(observations).astype(np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    mask_array = np.stack(action_masks).astype(np.bool_)
    metadata = {
        "config_path": str(config_path),
        "num_dags": num_dags,
        "seed_start": seed_start,
        "seed_end": seed_start + num_dags - 1,
        "samples": int(action_array.shape[0]),
        "dag_task_counts": dag_task_counts,
        "observation_shape": list(observation_array.shape),
        "action_mask_shape": list(mask_array.shape),
        "normalize_observations": True,
    }

    np.savez_compressed(
        output,
        observations=observation_array,
        actions=action_array,
        action_masks=mask_array,
        metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
    )

    print("BC_DATASET_GENERATION")
    print(f"config_path={Path(config_path).resolve()}")
    print(f"output_path={output}")
    print(f"num_dags={num_dags}")
    print(f"seed_start={seed_start}")
    print(f"samples={len(actions)}")
    print(f"observation_shape={observation_array.shape}")
    print(f"action_mask_shape={mask_array.shape}")
    print(f"min_tasks={min(dag_task_counts)}")
    print(f"max_tasks={max(dag_task_counts)}")
    print(f"mean_tasks={float(np.mean(dag_task_counts)):.3f}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HEFT behavior cloning dataset for MLP PPO.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to PPO YAML config.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .npz dataset path.")
    parser.add_argument("--num-dags", type=int, default=500, help="Number of random DAGs to generate.")
    parser.add_argument("--seed-start", type=int, default=2_000_000, help="First deterministic DAG seed.")
    args = parser.parse_args()
    generate_bc_dataset(
        config_path=args.config,
        output_path=args.output,
        num_dags=args.num_dags,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    main()
