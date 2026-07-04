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
from scheduler_interface import BaseScheduler, ScheduleResult


class RLScheduler(BaseScheduler):
    def __init__(
        self,
        model_path: str | Path,
        max_tasks: int,
        deterministic: bool = True,
        reward_mode: str = "raw",
    ):
        self.model = MaskablePPO.load(str(model_path))
        self.max_tasks = max_tasks
        self.deterministic = deterministic
        self.reward_mode = reward_mode
        self._uses_dict_observation = isinstance(self.model.observation_space, spaces.Dict)

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        raw_env = SchedulingEnv(
            dag_generator_fn=lambda seed=None: dag,
            resource_config=resource_config,
            max_tasks=self.max_tasks,
            reward_mode=self.reward_mode,
        )
        env: gym.Env = raw_env if self._uses_dict_observation else gym.wrappers.FlattenObservation(raw_env)
        observation, _ = env.reset()

        schedule_result: ScheduleResult = {}
        terminated = False
        while not terminated:
            mask = raw_env.action_masks()
            action, _ = self.model.predict(
                observation,
                action_masks=mask,
                deterministic=self.deterministic,
            )
            observation, _, terminated, truncated, info = env.step(int(action))
            if truncated:
                raise RuntimeError("RLScheduler environment truncated before completion")
            schedule_result[int(info["task_id"])] = (
                str(info["resource_id"]),
                float(info["start_time"]),
                float(info["finish_time"]),
            )

        return schedule_result
