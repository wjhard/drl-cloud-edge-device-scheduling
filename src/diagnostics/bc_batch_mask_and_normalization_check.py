from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch as th
import yaml
from gymnasium.spaces.utils import unflatten

sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import generate_random_dag
from env.resource_config import load_resource_config
from env.scheduling_env import SchedulingEnv
from training.train_ppo import _load_config, _resolve_project_path, build_model


DEFAULT_CONFIG = "training/configs/ppo_mlp_bc_warmstart.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_bc_dataset.npz"


def _load_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(dataset_path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    action_masks = np.asarray(data["action_masks"], dtype=np.bool_)
    metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    return observations, actions, action_masks, metadata


def _make_raw_env(config: dict) -> SchedulingEnv:
    env_config = config["env"]
    resource_config = load_resource_config(_resolve_project_path(env_config["resource_config_path"]))
    return SchedulingEnv(
        dag_generator_fn=lambda seed=None: generate_random_dag(1, seed=0),
        resource_config=resource_config,
        max_tasks=int(env_config["max_tasks_padding"]),
        reward_mode="raw",
    )


def _select_diverse_mask_samples(action_masks: np.ndarray) -> list[int]:
    legal_counts = action_masks.sum(axis=1)
    targets = [int(legal_counts.min()), 20, int(legal_counts.max())]
    selected: list[int] = []
    for target in targets:
        distances = np.abs(legal_counts - target)
        for index in np.argsort(distances):
            candidate = int(index)
            if candidate not in selected:
                selected.append(candidate)
                break
    return selected


def _evaluate_log_probs(policy, observations, actions, masks) -> np.ndarray:
    device = policy.device
    obs_tensor = th.as_tensor(observations, device=device).float()
    action_tensor = th.as_tensor(actions, device=device).long()
    mask_tensor = th.as_tensor(masks, device=device)
    with th.no_grad():
        _, log_prob, _ = policy.evaluate_actions(
            obs_tensor,
            action_tensor,
            action_masks=mask_tensor,
        )
    return log_prob.detach().cpu().numpy()


def _print_batch_mask_check(policy, observations, actions, action_masks) -> None:
    selected = _select_diverse_mask_samples(action_masks)
    batch_observations = observations[selected]
    batch_actions = actions[selected]
    batch_masks = action_masks[selected]
    batch_log_probs = _evaluate_log_probs(policy, batch_observations, batch_actions, batch_masks)

    print("CHECK_A_BATCH_MASK_CROSS_CONTAMINATION")
    print(f"selected_indices={selected}")
    print("batch_results:")
    max_abs_diff = 0.0
    for row, sample_index in enumerate(selected):
        single_log_prob = _evaluate_log_probs(
            policy,
            observations[sample_index : sample_index + 1],
            actions[sample_index : sample_index + 1],
            action_masks[sample_index : sample_index + 1],
        )[0]
        diff = abs(float(batch_log_probs[row]) - float(single_log_prob))
        max_abs_diff = max(max_abs_diff, diff)
        print(
            f"  sample_index={sample_index} legal_action_count={int(action_masks[sample_index].sum())} "
            f"expert_action={int(actions[sample_index])} "
            f"batch_log_prob={float(batch_log_probs[row]):.12f} "
            f"single_log_prob={float(single_log_prob):.12f} "
            f"abs_diff={diff:.12e} "
            f"consistent={diff <= 1e-5}"
        )
    print(f"max_abs_diff={max_abs_diff:.12e}")
    print(f"all_consistent={max_abs_diff <= 1e-5}")


def _stats_line(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return (
        f"{name}: min={float(np.min(flat)):.6f}, max={float(np.max(flat)):.6f}, "
        f"mean={float(np.mean(flat)):.6f}, std={float(np.std(flat)):.6f}"
    )


def _print_observation_field_stats(raw_env: SchedulingEnv, observations: np.ndarray) -> None:
    print("CHECK_B_OBSERVATION_SCALE_STATS")
    unflattened = [unflatten(raw_env.observation_space, observation) for observation in observations]
    task_features = np.stack([obs["task_features"] for obs in unflattened])
    task_valid_mask = np.stack([obs["task_valid_mask"] for obs in unflattened])
    task_adjacency = np.stack([obs["task_adjacency"] for obs in unflattened])
    ready_task_node_ids = np.stack([obs["ready_task_node_ids"] for obs in unflattened])
    resource_features = np.stack([obs["resource_features"] for obs in unflattened])
    global_features = np.stack([obs["global_features"] for obs in unflattened])

    task_feature_names = [
        "task_valid_flag",
        "task_id_normalized",
        "task_computation_cost",
        "task_out_degree",
        "task_successor_cost_sum",
    ]
    resource_feature_names = [
        "resource_available_time",
        "resource_tier_encoding",
        "resource_compute_power",
        "resource_bandwidth",
    ]
    global_feature_names = [
        "global_completed_fraction",
        "global_current_makespan",
    ]

    print(_stats_line("task_valid_mask", task_valid_mask))
    print(_stats_line("task_adjacency", task_adjacency))
    print(_stats_line("ready_task_node_ids", ready_task_node_ids))
    for index, name in enumerate(task_feature_names):
        print(_stats_line(f"task_features.{name}", task_features[:, :, index]))
    for index, name in enumerate(resource_feature_names):
        print(_stats_line(f"resource_features.{name}", resource_features[:, :, index]))
    for index, name in enumerate(global_feature_names):
        print(_stats_line(f"global_features.{name}", global_features[:, index]))

    flat_mean = observations.mean(axis=0)
    flat_std = observations.std(axis=0)
    print(_stats_line("flat_observation_all_values", observations))
    print(_stats_line("flat_feature_means", flat_mean))
    print(_stats_line("flat_feature_stds", flat_std))
    print(f"flat_features_with_std_lt_1e-8={int(np.sum(flat_std < 1e-8))}")


def _make_5dag_subset(metadata: dict, observations, actions, action_masks, num_dags: int = 5):
    task_counts = [int(value) for value in metadata["dag_task_counts"]]
    sample_count = sum(task_counts[:num_dags])
    return (
        observations[:sample_count],
        actions[:sample_count],
        action_masks[:sample_count],
        task_counts[:num_dags],
    )


def _train_normalized_subset(
    config: dict,
    full_observations: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    action_masks: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    feature_mean = full_observations.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = full_observations.std(axis=0, keepdims=True).astype(np.float32)
    normalized_observations = ((observations - feature_mean) / (feature_std + 1e-8)).astype(np.float32)

    model, _ = build_model(config)
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(20260708)
    device = policy.device
    sample_count = int(actions.shape[0])
    history: list[dict[str, float]] = []

    print("CHECK_C_NORMALIZED_93_SAMPLE_OVERFIT")
    print(f"subset_samples={sample_count}")
    print(f"epochs={epochs}")
    print(f"batch_size={batch_size}")
    print(f"learning_rate={learning_rate}")
    print("previous_without_normalization_first_accuracy=0.215054")
    print("previous_without_normalization_last_accuracy=0.247312")
    print("previous_without_normalization_best_accuracy=0.279570")

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(sample_count)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, sample_count, batch_size):
            batch_indices = permutation[start : start + batch_size]
            obs_tensor = th.as_tensor(normalized_observations[batch_indices], device=device).float()
            action_tensor = th.as_tensor(actions[batch_indices], device=device).long()
            mask_tensor = th.as_tensor(action_masks[batch_indices], device=device)
            _, log_prob, _ = policy.evaluate_actions(
                obs_tensor,
                action_tensor,
                action_masks=mask_tensor,
            )
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            batch_size_actual = len(batch_indices)
            total_loss += float(loss.detach().cpu().item()) * batch_size_actual
            with th.no_grad():
                distribution = policy.get_distribution(obs_tensor, action_masks=mask_tensor)
                predicted_actions = distribution.mode()
                total_correct += int((predicted_actions == action_tensor).sum().detach().cpu().item())
            total_seen += batch_size_actual

        average_loss = total_loss / total_seen
        accuracy = total_correct / total_seen
        history.append({"epoch": float(epoch), "loss": float(average_loss), "accuracy": float(accuracy)})
        if epoch <= 10 or epoch % 10 == 0 or epoch > epochs - 10:
            print(f"normalized_bc_epoch={epoch} loss={average_loss:.6f} accuracy={accuracy:.6f}")

    first = history[0]
    best = max(history, key=lambda item: item["accuracy"])
    last = history[-1]
    print("CHECK_C_NORMALIZED_OVERFIT_RESULT")
    print(f"first_loss={first['loss']:.6f}")
    print(f"last_loss={last['loss']:.6f}")
    print(f"best_loss={min(item['loss'] for item in history):.6f}")
    print(f"first_accuracy={first['accuracy']:.6f}")
    print(f"last_accuracy={last['accuracy']:.6f}")
    print(f"best_accuracy={best['accuracy']:.6f}")
    print(f"best_accuracy_epoch={int(best['epoch'])}")
    print(f"reached_90_percent_accuracy={best['accuracy'] >= 0.9}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Check BC batch mask behavior and observation normalization.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_path = _resolve_project_path(args.dataset)
    observations, actions, action_masks, metadata = _load_dataset(dataset_path)
    model, _ = build_model(config)

    print("BC_BATCH_MASK_AND_NORMALIZATION_CHECK")
    print(f"dataset_path={dataset_path}")
    print(f"config_path={Path(args.config).resolve()}")
    print(f"samples={len(actions)}")
    print(f"observation_shape={observations.shape}")
    print(f"action_mask_shape={action_masks.shape}")

    _print_batch_mask_check(model.policy, observations, actions, action_masks)
    raw_env = _make_raw_env(config)
    _print_observation_field_stats(raw_env, observations)
    subset_obs, subset_actions, subset_masks, subset_counts = _make_5dag_subset(
        metadata,
        observations,
        actions,
        action_masks,
    )
    print(f"normalized_overfit_subset_dag_task_counts={subset_counts}")
    _train_normalized_subset(
        config,
        observations,
        subset_obs,
        subset_actions,
        subset_masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
