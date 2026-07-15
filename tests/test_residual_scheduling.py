from __future__ import annotations

import numpy as np
import torch as th

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_random_dag
from env.resource_config import load_resource_config
from env.scheduling_env_residual import SchedulingEnvResidual
from training.train_ppo import _load_config, build_model


def test_residual_policy_delta_head_starts_at_zero() -> None:
    config = _load_config("training/configs/ppo_mlp_residual.yaml")
    config["training"]["total_timesteps"] = 1
    model, _ = build_model(config)

    assert th.allclose(model.policy.action_net.weight, th.zeros_like(model.policy.action_net.weight))
    assert th.allclose(model.policy.action_net.bias, th.zeros_like(model.policy.action_net.bias))


def test_untrained_residual_policy_matches_heft_makespan() -> None:
    config = _load_config("training/configs/ppo_mlp_residual.yaml")
    config["training"]["total_timesteps"] = 1
    model, _ = build_model(config)

    dag = generate_random_dag(num_tasks=18, edge_density=0.35, seed=20260711)
    heft_scheduler = HEFTScheduler()
    heft_resource_config = load_resource_config(config["env"]["resource_config_path"])
    heft_schedule = heft_scheduler.schedule(dag, heft_resource_config)
    heft_makespan = heft_scheduler.compute_makespan(heft_schedule)

    env_resource_config = load_resource_config(config["env"]["resource_config_path"])
    env = SchedulingEnvResidual(
        dag_generator_fn=lambda seed=None: dag,
        resource_config=env_resource_config,
        max_tasks=int(config["env"]["max_tasks_padding"]),
        reward_mode=str(config["env"]["reward_mode"]),
        normalize_observations=bool(config["env"]["normalize_observations"]),
    )
    observation, _ = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(
            observation,
            action_masks=env.action_masks(),
            deterministic=True,
        )
        observation, _, terminated, truncated, _ = env.step(int(action))
        assert not truncated

    assert np.isclose(env.current_makespan, heft_makespan, atol=1e-9)
