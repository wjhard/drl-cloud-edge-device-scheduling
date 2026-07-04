from __future__ import annotations

import numpy as np
import torch

from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from env.scheduling_env import SchedulingEnv
from policies.gat_features_extractor import TaskGraphFeaturesExtractor


def test_gat_features_extractor_batched_forward_shape_and_finiteness():
    env = SchedulingEnv(
        dag_generator_fn=lambda seed=None: generate_random_dag(8, edge_density=0.35, seed=seed),
        resource_config=load_resource_config(default_resource_config_path()),
        max_tasks=12,
        reward_mode="relative_heft",
    )
    observation, _ = env.reset(seed=17)
    batch_size = 4
    batch = {
        key: torch.as_tensor(np.stack([value] * batch_size))
        for key, value in observation.items()
    }

    extractor = TaskGraphFeaturesExtractor(
        env.observation_space,
        hidden_dim=32,
        num_gat_layers=2,
        num_heads=2,
    )
    features = extractor(batch)

    assert features.shape == (batch_size, extractor.features_dim)
    assert torch.isfinite(features).all()
