from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
from env.resource_config import Resource, load_resource_config
from env.scheduling_env import SchedulingEnv
from env.scheduling_utils import find_earliest_slot
from training.dag_curriculum import make_training_dag_generator


DEFAULT_CONFIG = "training/configs/ppo_mlp_bc_warmstart.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_bc_dataset.npz"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_metadata(dataset_path: str | Path) -> dict:
    data = np.load(dataset_path, allow_pickle=False)
    if "metadata" not in data:
        raise ValueError("dataset must include metadata with seed_start and num_dags")
    return json.loads(str(data["metadata"].item()))


def _hypothetical_finish_time(env: SchedulingEnv, task_id: int, resource: Resource) -> tuple[float, float]:
    assert env.dag is not None
    graph = env.dag.graph
    data_ready_time = 0.0
    for predecessor in graph.predecessors(task_id):
        if predecessor not in env.task_times:
            raise RuntimeError(f"predecessor {predecessor} has not been scheduled")
        _, pred_finish = env.task_times[predecessor]
        pred_resource_id = env.task_assignments[predecessor]
        data_size = float(graph.edges[predecessor, task_id].get("data_size", 0.0))
        communication_time = env.resource_config.get_communication_time(
            data_size,
            pred_resource_id,
            resource,
        )
        data_ready_time = max(data_ready_time, pred_finish + communication_time)

    duration = env.resource_config.get_execution_time(graph.nodes[task_id], resource)
    return find_earliest_slot(env.resource_events[resource.id], data_ready_time, duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how often HEFT resource choices are near-ties.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num-dags", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--seed-start", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    env_config = config["env"]
    train_config = config["training"]
    metadata = _load_metadata(_resolve_project_path(args.dataset))
    num_dags = int(args.num_dags if args.num_dags is not None else metadata["num_dags"])
    seed_start = int(args.seed_start if args.seed_start is not None else metadata["seed_start"])
    resource_config_path = _resolve_project_path(env_config["resource_config_path"])

    dag_generator = make_training_dag_generator(
        min_tasks=int(env_config["min_tasks"]),
        max_tasks=int(env_config["max_tasks"]),
        edge_density_min=float(env_config["edge_density_min"]),
        edge_density_max=float(env_config["edge_density_max"]),
        base_seed=int(train_config.get("seed", 0)),
    )
    scheduler = HEFTScheduler()

    total_decisions = 0
    within_1pct = 0
    within_5pct = 0
    within_10pct = 0
    exact_tie = 0
    nearest_alternative_gaps: list[float] = []
    heft_resource_counts: Counter[str] = Counter()
    near_tie_examples: list[dict] = []

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
        env = SchedulingEnv(
            dag_generator_fn=lambda seed=None, fixed_dag=dag: fixed_dag,
            resource_config=env_resource_config,
            max_tasks=int(env_config["max_tasks_padding"]),
            reward_mode="raw",
        )
        env.reset()
        resource_index = {
            resource.id: index
            for index, resource in enumerate(env_resource_config.resources)
        }

        for task_id in task_order:
            expert_resource_id = assignment[task_id]
            alternatives: dict[str, float] = {}
            for resource in env_resource_config.resources:
                _, finish_time = _hypothetical_finish_time(env, task_id, resource)
                alternatives[resource.id] = finish_time

            expert_finish = alternatives[expert_resource_id]
            other_finishes = [
                finish
                for resource_id, finish in alternatives.items()
                if resource_id != expert_resource_id
            ]
            nearest_gap = min(
                abs(finish - expert_finish) / max(abs(expert_finish), 1e-12)
                for finish in other_finishes
            )
            nearest_alternative_gaps.append(float(nearest_gap))
            total_decisions += 1
            heft_resource_counts[expert_resource_id] += 1

            if nearest_gap <= 1e-12:
                exact_tie += 1
            if nearest_gap <= 0.01:
                within_1pct += 1
            if nearest_gap <= args.tolerance:
                within_5pct += 1
                if len(near_tie_examples) < 10:
                    near_tie_examples.append(
                        {
                            "dag_index": dag_index,
                            "seed": seed,
                            "task_id": task_id,
                            "expert_resource_id": expert_resource_id,
                            "expert_finish": expert_finish,
                            "nearest_gap": nearest_gap,
                            "alternatives": alternatives,
                        }
                    )
            if nearest_gap <= 0.10:
                within_10pct += 1

            if task_id not in env.ready_tasks:
                raise RuntimeError(f"task {task_id} is not ready during HEFT replay")
            task_slot = env.ready_tasks.index(task_id)
            action = task_slot * env.num_resources + resource_index[expert_resource_id]
            _, _, terminated, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError(f"environment truncated while replaying DAG seed={seed}")

        if not terminated:
            raise RuntimeError(f"HEFT replay did not terminate for DAG seed={seed}")

    gap_array = np.asarray(nearest_alternative_gaps, dtype=np.float64)
    print("HEFT_DECISION_AMBIGUITY_CHECK")
    print(f"dataset_path={_resolve_project_path(args.dataset)}")
    print(f"config_path={Path(args.config).resolve()}")
    print(f"num_dags={num_dags}")
    print(f"seed_start={seed_start}")
    print(f"decisions={total_decisions}")
    print(f"tolerance={args.tolerance:.6f}")
    print(f"exact_tie_count={exact_tie}")
    print(f"exact_tie_ratio={exact_tie / total_decisions:.6f}")
    print(f"within_1pct_count={within_1pct}")
    print(f"within_1pct_ratio={within_1pct / total_decisions:.6f}")
    print(f"within_5pct_count={within_5pct}")
    print(f"within_5pct_ratio={within_5pct / total_decisions:.6f}")
    print(f"within_10pct_count={within_10pct}")
    print(f"within_10pct_ratio={within_10pct / total_decisions:.6f}")
    print(f"nearest_gap_mean={float(np.mean(gap_array)):.6f}")
    print(f"nearest_gap_median={float(np.median(gap_array)):.6f}")
    print(f"nearest_gap_p25={float(np.quantile(gap_array, 0.25)):.6f}")
    print(f"nearest_gap_p75={float(np.quantile(gap_array, 0.75)):.6f}")
    print(f"nearest_gap_min={float(np.min(gap_array)):.6f}")
    print(f"nearest_gap_max={float(np.max(gap_array)):.6f}")
    print("heft_resource_counts:")
    for resource_id, count in sorted(heft_resource_counts.items()):
        print(f"  {resource_id}: count={count}, ratio={count / total_decisions:.6f}")
    print("near_tie_examples:")
    for example in near_tie_examples:
        alternatives_text = ", ".join(
            f"{resource_id}:{finish:.6f}"
            for resource_id, finish in example["alternatives"].items()
        )
        print(
            "  "
            f"dag_index={example['dag_index']} seed={example['seed']} task_id={example['task_id']} "
            f"expert={example['expert_resource_id']} expert_finish={example['expert_finish']:.6f} "
            f"nearest_gap={example['nearest_gap']:.6f} alternatives={{ {alternatives_text} }}"
        )


if __name__ == "__main__":
    main()
