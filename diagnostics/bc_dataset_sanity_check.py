from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from gymnasium.spaces.utils import unflatten

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import generate_random_dag
from env.resource_config import load_resource_config
from env.scheduling_env import SchedulingEnv


DEFAULT_CONFIG = "training/configs/ppo_mlp_bc_warmstart.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_bc_dataset.npz"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(dataset_path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    action_masks = np.asarray(data["action_masks"], dtype=np.bool_)
    metadata = {}
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"].item()))
    return observations, actions, action_masks, metadata


def _make_observation_space_env(config: dict) -> tuple[SchedulingEnv, list[str]]:
    env_config = config["env"]
    resource_config = load_resource_config(_resolve_project_path(env_config["resource_config_path"]))
    raw_env = SchedulingEnv(
        dag_generator_fn=lambda seed=None: generate_random_dag(1, seed=0),
        resource_config=resource_config,
        max_tasks=int(env_config["max_tasks_padding"]),
        reward_mode="raw",
    )
    resource_ids = [resource.id for resource in resource_config.resources]
    return raw_env, resource_ids


def _sample_summary(
    sample_index: int,
    flat_observation: np.ndarray,
    expert_action: int,
    action_mask: np.ndarray,
    raw_env: SchedulingEnv,
    resource_ids: list[str],
) -> str:
    obs = unflatten(raw_env.observation_space, flat_observation)
    legal_actions = np.flatnonzero(action_mask)
    num_resources = len(resource_ids)
    expert_slot = int(expert_action) // num_resources
    expert_resource_index = int(expert_action) % num_resources
    expert_resource_id = resource_ids[expert_resource_index]
    ready_slots = sorted({int(action) // num_resources for action in legal_actions})
    ready_task_node_ids = obs["ready_task_node_ids"].astype(np.int64)
    ready_task_ids = [int(ready_task_node_ids[slot]) for slot in ready_slots]
    expert_task_id = int(ready_task_node_ids[expert_slot]) if expert_slot < len(ready_task_node_ids) else -1
    valid_task_count = int(obs["task_valid_mask"].sum())
    completed_fraction = float(obs["global_features"][0])
    current_makespan = float(obs["global_features"][1])
    resource_available_times = {
        resource_ids[index]: float(obs["resource_features"][index, 0])
        for index in range(len(resource_ids))
    }

    lines = [
        f"sample_index={sample_index}",
        f"  expert_action={int(expert_action)} legal={bool(action_mask[int(expert_action)])}",
        f"  legal_action_count={len(legal_actions)} ready_slot_count={len(ready_slots)}",
        (
            f"  expert_decoding=slot:{expert_slot}, task_id:{expert_task_id}, "
            f"resource_index:{expert_resource_index}, resource_id:{expert_resource_id}"
        ),
        f"  ready_task_ids={ready_task_ids}",
        (
            f"  observation_summary=valid_task_count:{valid_task_count}, "
            f"completed_fraction:{completed_fraction:.6f}, current_makespan:{current_makespan:.6f}"
        ),
        f"  resource_available_times={resource_available_times}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check HEFT BC dataset actions and chance level.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--chance-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args()

    config = _load_config(args.config)
    observations, actions, action_masks, metadata = _load_dataset(_resolve_project_path(args.dataset))
    raw_env, resource_ids = _make_observation_space_env(config)
    rng = np.random.default_rng(args.seed)

    print("BC_DATASET_SANITY_CHECK")
    print(f"dataset_path={_resolve_project_path(args.dataset)}")
    print(f"config_path={Path(args.config).resolve()}")
    print(f"samples={len(actions)}")
    print(f"observation_shape={observations.shape}")
    print(f"action_mask_shape={action_masks.shape}")
    if metadata:
        print(f"metadata={json.dumps(metadata, ensure_ascii=False)}")

    all_expert_legal = action_masks[np.arange(len(actions)), actions]
    illegal_indices = np.flatnonzero(~all_expert_legal)
    print(f"full_dataset_illegal_expert_actions={len(illegal_indices)}")
    if len(illegal_indices) > 0:
        print(f"first_illegal_indices={illegal_indices[:20].tolist()}")

    sample_count = min(args.num_samples, len(actions))
    sample_indices = rng.choice(len(actions), size=sample_count, replace=False)
    print(f"sampled_indices={sample_indices.tolist()}")
    print("sample_details:")
    sampled_illegal = 0
    for sample_index in sample_indices:
        if not action_masks[sample_index, actions[sample_index]]:
            sampled_illegal += 1
        print(
            _sample_summary(
                int(sample_index),
                observations[sample_index],
                int(actions[sample_index]),
                action_masks[sample_index],
                raw_env,
                resource_ids,
            )
        )
    print(f"sampled_illegal_expert_actions={sampled_illegal}")

    legal_counts = action_masks.sum(axis=1).astype(np.float64)
    if np.any(legal_counts <= 0):
        zero_indices = np.flatnonzero(legal_counts <= 0)
        raise RuntimeError(f"found samples with no legal actions: {zero_indices[:20].tolist()}")

    chance_sample_count = min(args.chance_samples, len(actions))
    chance_indices = rng.choice(len(actions), size=chance_sample_count, replace=False)
    sampled_legal_counts = legal_counts[chance_indices]
    expected_chance_all = float(np.mean(1.0 / legal_counts))
    reciprocal_of_average_all = float(1.0 / np.mean(legal_counts))
    expected_chance_sampled = float(np.mean(1.0 / sampled_legal_counts))

    monte_carlo_correct = 0
    for sample_index in chance_indices:
        legal_actions = np.flatnonzero(action_masks[sample_index])
        guessed_action = int(rng.choice(legal_actions))
        if guessed_action == int(actions[sample_index]):
            monte_carlo_correct += 1
    monte_carlo_accuracy = monte_carlo_correct / chance_sample_count

    print("BC_CHANCE_LEVEL")
    print(f"all_samples={len(actions)}")
    print(f"average_legal_actions_all={float(np.mean(legal_counts)):.6f}")
    print(f"min_legal_actions_all={int(np.min(legal_counts))}")
    print(f"max_legal_actions_all={int(np.max(legal_counts))}")
    print(f"expected_uniform_random_accuracy_all_mean_inverse={expected_chance_all:.6f}")
    print(f"reciprocal_of_average_legal_actions_all={reciprocal_of_average_all:.6f}")
    print(f"chance_sample_count={chance_sample_count}")
    print(f"average_legal_actions_sampled={float(np.mean(sampled_legal_counts)):.6f}")
    print(f"expected_uniform_random_accuracy_sampled={expected_chance_sampled:.6f}")
    print(f"monte_carlo_uniform_random_accuracy_sampled={monte_carlo_accuracy:.6f}")
    print("reference_bc_accuracy_observed_previous_run=0.166708_to_0.172431_peak_to_0.166957_final")


if __name__ == "__main__":
    main()
