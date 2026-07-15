from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from env.dag_generator import DAGTask, get_ready_tasks
from env.resource_config import Resource, ResourceConfig
from env.scheduling_env import SchedulingEnv
from env.scheduling_utils import find_earliest_slot


class SchedulingEnvRanked(SchedulingEnv):
    """Environment where the policy ranks ready tasks and resources are chosen by EFT."""

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
            include_upward_rank_feature=include_upward_rank_feature,
        )
        self.action_space = spaces.Discrete(self.max_ready_tasks)

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        ready_slots = min(len(self.ready_tasks), self.max_ready_tasks)
        mask[:ready_slots] = True
        return mask

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.dag is not None
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"action {action} is outside action space")
        if action >= len(self.ready_tasks):
            raise ValueError(f"action {action} selects a non-ready task slot")

        task_id = self.ready_tasks[int(action)]
        resource = self._select_earliest_finish_resource(task_id)
        start_time, finish_time = self._schedule_task(task_id, resource)

        self.completed_tasks.add(task_id)
        self.task_assignments[task_id] = resource.id
        self.task_times[task_id] = (start_time, finish_time)
        self.current_makespan = max(self.current_makespan, finish_time)

        terminated = len(self.completed_tasks) == self.dag.graph.number_of_nodes()
        self.ready_tasks = get_ready_tasks(self.dag, self.completed_tasks)
        if not terminated and not self.ready_tasks:
            raise RuntimeError("no legal actions remain before all tasks are completed")

        if self.reward_mode == "relative_heft":
            reward = -0.01
            if terminated and self._heft_makespan_ref is not None and self._heft_makespan_ref > 1e-12:
                reward -= (self.current_makespan / self._heft_makespan_ref - 1.0)
        else:
            reward = -1.0
            if terminated:
                reward -= self.current_makespan

        info = {
            "task_id": task_id,
            "resource_id": resource.id,
            "start_time": start_time,
            "finish_time": finish_time,
            "makespan": self.current_makespan,
            "ready_tasks": list(self.ready_tasks),
        }
        return self._get_observation(), reward, terminated, False, info

    def _select_earliest_finish_resource(self, task_id: int) -> Resource:
        assert self.dag is not None
        best_resource: Resource | None = None
        best_finish = float("inf")

        for resource in self.resource_config.resources:
            ready_time = self._dependency_ready_time(task_id, resource)
            duration = self.resource_config.get_execution_time(self.dag.graph.nodes[task_id], resource)
            _, finish_time = find_earliest_slot(
                self.resource_events[resource.id],
                ready_time,
                duration,
            )
            if finish_time < best_finish:
                best_resource = resource
                best_finish = finish_time

        if best_resource is None:
            raise RuntimeError("no resource available for ranked scheduling")
        return best_resource

    def _dependency_ready_time(self, task_id: int, resource: Resource) -> float:
        assert self.dag is not None
        ready_time = 0.0
        for predecessor in self.dag.graph.predecessors(task_id):
            if predecessor not in self.task_times:
                raise RuntimeError(f"predecessor {predecessor} has not been scheduled")
            pred_resource_id = self.task_assignments[predecessor]
            _, pred_finish = self.task_times[predecessor]
            data_size = float(self.dag.graph.edges[predecessor, task_id].get("data_size", 0.0))
            communication_time = self.resource_config.get_communication_time(
                data_size,
                pred_resource_id,
                resource,
            )
            ready_time = max(ready_time, pred_finish + communication_time)
        return ready_time
