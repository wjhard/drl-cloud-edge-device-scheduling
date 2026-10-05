from __future__ import annotations

from typing import Callable

import numpy as np
from gymnasium import spaces

from baselines.heft_scheduler import compute_upward_ranks
from env.dag_generator import DAGTask
from env.resource_config import Resource, ResourceConfig
from env.scheduling_env_ranked import SchedulingEnvRanked


class SchedulingEnvResidual(SchedulingEnvRanked):
    """Ranked environment that exposes HEFT ready-task ranks for residual policies."""

    def __init__(
        self,
        dag_generator_fn: Callable[..., DAGTask],
        resource_config: ResourceConfig,
        max_tasks: int = 50,
        reward_mode: str = "raw",
        normalize_observations: bool = False,
        include_upward_rank_feature: bool = False,
    ):
        super().__init__(
            dag_generator_fn=dag_generator_fn,
            resource_config=resource_config,
            max_tasks=max_tasks,
            reward_mode=reward_mode,
            normalize_observations=normalize_observations,
            include_upward_rank_feature=False,
        )
        self.observation_space.spaces["ready_task_upward_ranks"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_ready_tasks,),
            dtype=np.float32,
        )

    def _get_observation(self) -> dict[str, np.ndarray]:
        observation = super()._get_observation()
        ready_ranks = np.zeros((self.max_ready_tasks,), dtype=np.float32)
        for slot, task_id in enumerate(self.ready_tasks[: self.max_ready_tasks]):
            ready_ranks[slot] = self._rank_for_policy_logits(int(task_id))
        observation["ready_task_upward_ranks"] = ready_ranks
        return observation

    def _compute_upward_ranks_for_observation(self) -> dict[int, float]:
        assert self.dag is not None
        reference_config = ResourceConfig(
            [
                Resource(
                    id=resource.id,
                    tier=resource.tier,
                    compute_power=resource.compute_power,
                    bandwidth=resource.bandwidth,
                )
                for resource in self.resource_config.resources
            ]
        )
        return compute_upward_ranks(self.dag, reference_config)

    def _rank_for_policy_logits(self, task_id: int) -> float:
        rank_value = float(self._upward_ranks.get(task_id, 0.0))
        if not self.normalize_observations or self.observation_normalizer is None:
            return rank_value

        rank_stats = self.observation_normalizer.stats.get("task_features", {}).get("5")
        if rank_stats is None:
            return rank_value
        mean = float(rank_stats["mean"])
        std = float(rank_stats["std"])
        return (rank_value - mean) / (std + self.observation_normalizer.epsilon)
