from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th


def _load_bc_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"BC dataset not found: {path}")

    data = np.load(path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    action_masks = np.asarray(data["action_masks"], dtype=np.bool_)
    metadata: dict[str, Any] = {}
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"].item()))

    if observations.shape[0] != actions.shape[0] or actions.shape[0] != action_masks.shape[0]:
        raise ValueError(
            "dataset arrays must have the same leading dimension: "
            f"observations={observations.shape}, actions={actions.shape}, masks={action_masks.shape}"
        )
    if observations.ndim != 2:
        raise ValueError("MLP BC dataset expects flattened 2D observations")
    if action_masks.ndim != 2:
        raise ValueError("BC dataset expects 2D action masks")
    return observations, actions, action_masks, metadata


def pretrain_policy_with_bc(
    model,
    dataset_path: str | Path,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    observations, actions, action_masks, metadata = _load_bc_dataset(dataset_path)
    sample_count = int(actions.shape[0])
    device = model.policy.device
    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(20260708)
    history: list[dict[str, float]] = []

    print("BC_PRETRAIN_START")
    print(f"dataset_path={Path(dataset_path).resolve()}")
    print(f"samples={sample_count}")
    print(f"observation_shape={observations.shape}")
    print(f"action_mask_shape={action_masks.shape}")
    if metadata:
        print(f"dataset_num_dags={metadata.get('num_dags')}")
        print(f"dataset_seed_start={metadata.get('seed_start')}")
        print(f"dataset_seed_end={metadata.get('seed_end')}")
    print(f"epochs={epochs}")
    print(f"batch_size={batch_size}")
    print(f"learning_rate={learning_rate}")

    model.policy.set_training_mode(True)
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(sample_count)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for start in range(0, sample_count, batch_size):
            batch_indices = permutation[start : start + batch_size]
            obs_tensor = th.as_tensor(observations[batch_indices], device=device).float()
            actions_tensor = th.as_tensor(actions[batch_indices], device=device).long()
            masks_tensor = th.as_tensor(action_masks[batch_indices], device=device)

            _, log_prob, _ = model.policy.evaluate_actions(
                obs_tensor,
                actions_tensor,
                action_masks=masks_tensor,
            )
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()

            batch_size_actual = int(len(batch_indices))
            total_loss += float(loss.detach().cpu().item()) * batch_size_actual

            with th.no_grad():
                distribution = model.policy.get_distribution(obs_tensor, action_masks=masks_tensor)
                predicted_actions = distribution.mode()
                total_correct += int((predicted_actions == actions_tensor).sum().detach().cpu().item())
            total_seen += batch_size_actual

        average_loss = total_loss / max(1, total_seen)
        accuracy = total_correct / max(1, total_seen)
        record = {
            "epoch": float(epoch),
            "loss": float(average_loss),
            "accuracy": float(accuracy),
        }
        history.append(record)
        print(f"bc_epoch={epoch} loss={average_loss:.6f} accuracy={accuracy:.6f}", flush=True)

    print("BC_PRETRAIN_DONE")
    return history
