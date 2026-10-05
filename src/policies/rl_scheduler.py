from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

from sb3_contrib import MaskablePPO

from env.dag_generator import DAGTask
from env.resource_config import ResourceConfig
from env.scheduling_env import SchedulingEnv
from env.scheduling_env_ranked import SchedulingEnvRanked
from env.scheduling_env_residual import SchedulingEnvResidual
from scheduler_interface import BaseScheduler, ScheduleResult


class RLScheduler(BaseScheduler):
    def __init__(
        self,
        model_path: str | Path,
        max_tasks: int,
        deterministic: bool = True,
        reward_mode: str = "raw",
        normalize_observations: bool = False,
        include_upward_rank_feature: bool = False,
        scheduler_mode: str = "joint",
        num_samples: int = 1,
        sampling_deterministic_fallback: bool = True,
    ):
        if scheduler_mode not in {"joint", "ranked", "residual"}:
            raise ValueError("scheduler_mode must be one of 'joint', 'ranked', or 'residual'")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.model = MaskablePPO.load(str(model_path))
        self.max_tasks = max_tasks
        self.deterministic = deterministic
        self.reward_mode = reward_mode
        self.normalize_observations = normalize_observations
        self.include_upward_rank_feature = include_upward_rank_feature
        self.scheduler_mode = scheduler_mode
        self.num_samples = num_samples
        self.sampling_deterministic_fallback = sampling_deterministic_fallback
        self._uses_dict_observation = isinstance(self.model.observation_space, spaces.Dict)

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        if self.num_samples == 1:
            return self._schedule_once(dag, resource_config, deterministic=self.deterministic)

        best_schedule: ScheduleResult | None = None
        best_makespan = float("inf")

        for _ in range(self.num_samples):
            schedule_result = self._schedule_once(dag, resource_config, deterministic=False)
            makespan = self.compute_makespan(schedule_result)
            if makespan < best_makespan:
                best_schedule = schedule_result
                best_makespan = makespan

        if self.sampling_deterministic_fallback:
            schedule_result = self._schedule_once(dag, resource_config, deterministic=True)
            makespan = self.compute_makespan(schedule_result)
            if makespan < best_makespan:
                best_schedule = schedule_result

        if best_schedule is None:
            raise RuntimeError("sampling did not produce a schedule")
        return best_schedule

    def _schedule_once(
        self,
        dag: DAGTask,
        resource_config: ResourceConfig,
        deterministic: bool,
    ) -> ScheduleResult:
        schedule_result, _ = self._schedule_once_with_order(
            dag,
            resource_config,
            deterministic=deterministic,
        )
        return schedule_result

    def _schedule_once_with_order(
        self,
        dag: DAGTask,
        resource_config: ResourceConfig,
        deterministic: bool,
    ) -> tuple[ScheduleResult, list[int]]:
        env_class = {
            "joint": SchedulingEnv,
            "ranked": SchedulingEnvRanked,
            "residual": SchedulingEnvResidual,
        }[self.scheduler_mode]
        raw_env = env_class(
            dag_generator_fn=lambda seed=None: dag,
            resource_config=resource_config,
            max_tasks=self.max_tasks,
            reward_mode=self.reward_mode,
            normalize_observations=self.normalize_observations,
            include_upward_rank_feature=self.include_upward_rank_feature,
        )
        env: gym.Env = raw_env if self._uses_dict_observation else gym.wrappers.FlattenObservation(raw_env)
        observation, _ = env.reset()

        schedule_result: ScheduleResult = {}
        task_order: list[int] = []
        terminated = False
        while not terminated:
            mask = raw_env.action_masks()
            action, _ = self.model.predict(
                observation,
                action_masks=mask,
                deterministic=deterministic,
            )
            observation, _, terminated, truncated, info = env.step(int(action))
            if truncated:
                raise RuntimeError("RLScheduler environment truncated before completion")
            schedule_result[int(info["task_id"])] = (
                str(info["resource_id"]),
                float(info["start_time"]),
                float(info["finish_time"]),
            )
            task_order.append(int(info["task_id"]))

        return schedule_result, task_order
