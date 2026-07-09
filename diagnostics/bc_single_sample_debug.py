from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch as th

sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_ppo import _load_config, _resolve_project_path, build_model


DEFAULT_CONFIG = "training/configs/ppo_mlp_bc_warmstart.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_bc_dataset.npz"


def _load_single_sample(dataset_path: str | Path, sample_index: int):
    data = np.load(dataset_path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    action_masks = np.asarray(data["action_masks"], dtype=np.bool_)
    if sample_index < 0 or sample_index >= len(actions):
        raise IndexError(f"sample_index {sample_index} is outside dataset length {len(actions)}")
    return (
        observations[sample_index : sample_index + 1],
        actions[sample_index : sample_index + 1],
        action_masks[sample_index : sample_index + 1],
        observations.shape,
        action_masks.shape,
    )


def _iter_trainable_params(policy):
    for name, param in policy.named_parameters():
        if param.requires_grad:
            yield name, param


def _find_first_weight_matrix(policy):
    for name, param in _iter_trainable_params(policy):
        if param.ndim >= 2:
            return name, param
    raise RuntimeError("no trainable matrix parameter found in policy")


def _distribution_probs(distribution) -> th.Tensor:
    categorical = getattr(distribution, "distribution", None)
    if categorical is None:
        raise RuntimeError(f"distribution object has no .distribution: {type(distribution)}")
    probs = getattr(categorical, "probs", None)
    if probs is None:
        raise RuntimeError(f"categorical distribution has no .probs: {type(categorical)}")
    return probs


def _expert_probability(policy, obs_tensor: th.Tensor, action_tensor: th.Tensor, mask_tensor: th.Tensor) -> float:
    with th.no_grad():
        _, log_prob, _ = policy.evaluate_actions(obs_tensor, action_tensor, action_masks=mask_tensor)
        return float(th.exp(log_prob)[0].detach().cpu().item())


def _run_log_prob_action_index_check(
    policy,
    obs_tensor: th.Tensor,
    expert_action: int,
    full_mask: np.ndarray,
) -> None:
    legal_actions = np.flatnonzero(full_mask[0])
    alternate_actions = [int(action) for action in legal_actions if int(action) != expert_action]
    if not alternate_actions:
        print("EVALUATE_ACTIONS_INDEX_CHECK skipped: sample has only one legal action")
        return
    alternate_action = alternate_actions[0]
    two_action_mask = np.zeros_like(full_mask, dtype=np.bool_)
    two_action_mask[0, expert_action] = True
    two_action_mask[0, alternate_action] = True
    mask_tensor = th.as_tensor(two_action_mask, device=policy.device)
    expert_tensor = th.as_tensor([expert_action], device=policy.device).long()
    alternate_tensor = th.as_tensor([alternate_action], device=policy.device).long()

    with th.no_grad():
        distribution = policy.get_distribution(obs_tensor, action_masks=mask_tensor)
        probs = _distribution_probs(distribution)[0]
        _, expert_log_prob, _ = policy.evaluate_actions(
            obs_tensor,
            expert_tensor,
            action_masks=mask_tensor,
        )
        _, alternate_log_prob, _ = policy.evaluate_actions(
            obs_tensor,
            alternate_tensor,
            action_masks=mask_tensor,
        )
        expert_prob_from_dist = float(probs[expert_action].detach().cpu().item())
        alternate_prob_from_dist = float(probs[alternate_action].detach().cpu().item())
        expert_prob_from_log_prob = float(th.exp(expert_log_prob)[0].detach().cpu().item())
        alternate_prob_from_log_prob = float(th.exp(alternate_log_prob)[0].detach().cpu().item())

    print("EVALUATE_ACTIONS_INDEX_CHECK")
    print(f"two_legal_actions=[{expert_action}, {alternate_action}]")
    print(f"expert_action={expert_action}")
    print(f"alternate_action={alternate_action}")
    print(f"dist_prob_expert={expert_prob_from_dist:.12f}")
    print(f"exp_evaluate_log_prob_expert={expert_prob_from_log_prob:.12f}")
    print(f"dist_prob_alternate={alternate_prob_from_dist:.12f}")
    print(f"exp_evaluate_log_prob_alternate={alternate_prob_from_log_prob:.12f}")
    print(f"prob_sum_two_actions={(expert_prob_from_dist + alternate_prob_from_dist):.12f}")
    print(f"expert_match_abs_diff={abs(expert_prob_from_dist - expert_prob_from_log_prob):.12e}")
    print(f"alternate_match_abs_diff={abs(alternate_prob_from_dist - alternate_prob_from_log_prob):.12e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug BC single-sample overfitting and gradient flow.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_path = _resolve_project_path(args.dataset)
    obs_np, action_np, mask_np, observation_shape, action_mask_shape = _load_single_sample(
        dataset_path,
        args.sample_index,
    )
    expert_action = int(action_np[0])
    legal_actions = np.flatnonzero(mask_np[0])
    if not mask_np[0, expert_action]:
        raise RuntimeError(f"sample expert action {expert_action} is not legal")

    model, _ = build_model(config)
    policy = model.policy
    policy.set_training_mode(True)
    device = policy.device
    obs_tensor = th.as_tensor(obs_np, device=device).float()
    action_tensor = th.as_tensor(action_np, device=device).long()
    mask_tensor = th.as_tensor(mask_np, device=device)
    optimizer_params = list(_iter_trainable_params(policy))
    optimizer = th.optim.Adam([param for _, param in optimizer_params], lr=args.learning_rate)
    first_weight_name, first_weight = _find_first_weight_matrix(policy)

    print("BC_SINGLE_SAMPLE_DEBUG")
    print(f"dataset_path={dataset_path}")
    print(f"config_path={Path(args.config).resolve()}")
    print(f"dataset_observation_shape={observation_shape}")
    print(f"dataset_action_mask_shape={action_mask_shape}")
    print(f"sample_index={args.sample_index}")
    print(f"expert_action={expert_action}")
    print(f"legal_action_count={len(legal_actions)}")
    print(f"legal_actions={legal_actions.tolist()}")
    print(f"steps={args.steps}")
    print(f"learning_rate={args.learning_rate}")
    print("OPTIMIZER_PARAMETER_LIST")
    for index, (name, param) in enumerate(optimizer_params):
        print(
            f"  param_index={index} name={name} shape={tuple(param.shape)} "
            f"requires_grad={param.requires_grad}"
        )
    print(f"FIRST_WEIGHT_MATRIX name={first_weight_name} shape={tuple(first_weight.shape)}")

    _run_log_prob_action_index_check(policy, obs_tensor, expert_action, mask_np)

    print("SINGLE_SAMPLE_TRAINING_TRACE")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        pre_abs_sum = float(first_weight.detach().abs().sum().cpu().item())
        _, log_prob, _ = policy.evaluate_actions(obs_tensor, action_tensor, action_masks=mask_tensor)
        loss = -log_prob.mean()
        pre_prob = float(th.exp(log_prob)[0].detach().cpu().item())
        loss.backward()

        grad = first_weight.grad
        grad_abs_sum = None if grad is None else float(grad.detach().abs().sum().cpu().item())
        optimizer.step()
        post_abs_sum = float(first_weight.detach().abs().sum().cpu().item())
        post_prob = _expert_probability(policy, obs_tensor, action_tensor, mask_tensor)
        print(
            f"step={step:03d} "
            f"loss={float(loss.detach().cpu().item()):.12f} "
            f"pre_expert_prob={pre_prob:.12f} "
            f"post_expert_prob={post_prob:.12f} "
            f"first_weight_grad_abs_sum={grad_abs_sum} "
            f"first_weight_abs_sum_before={pre_abs_sum:.12f} "
            f"first_weight_abs_sum_after={post_abs_sum:.12f} "
            f"first_weight_abs_sum_delta={(post_abs_sum - pre_abs_sum):.12e}"
        )


if __name__ == "__main__":
    main()
