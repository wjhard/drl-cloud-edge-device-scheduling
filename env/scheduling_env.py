from __future__ import annotations

import os
import sys
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import DAGTask, generate_random_dag, get_ready_tasks
from env.resource_config import Resource, ResourceConfig, default_resource_config_path, load_resource_config
from env.scheduling_utils import ScheduledEvent, find_earliest_slot


TIER_ENCODING = {
    "cloud": 3.0,
    "edge": 2.0,
    "device": 1.0,
}


class SchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dag_generator_fn: Callable[..., DAGTask],
        resource_config: ResourceConfig,
        max_tasks: int = 50,
    ):
        super().__init__()
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")

        self.dag_generator_fn = dag_generator_fn
        self.resource_config = resource_config
        self.max_tasks = max_tasks
        self.max_ready_tasks = max_tasks
        self.num_resources = len(resource_config.resources)

        self.task_feature_dim = 5
        self.resource_feature_dim = 4
        observation_dim = (
            self.max_ready_tasks * self.task_feature_dim
            + self.num_resources * self.resource_feature_dim
            + 2
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_ready_tasks * self.num_resources)

        self.dag: DAGTask | None = None
        self.completed_tasks: set[int] = set()
        self.task_assignments: dict[int, str] = {}
        self.task_times: dict[int, tuple[float, float]] = {}
        self.resource_events: dict[str, list[ScheduledEvent]] = {
            resource.id: [] for resource in resource_config.resources
        }
        self.ready_tasks: list[int] = []
        self.current_makespan = 0.0

    def _call_dag_generator(self, seed: int | None) -> DAGTask:
        if seed is None:
            return self.dag_generator_fn()
        try:
            return self.dag_generator_fn(seed=seed)
        except TypeError:
            return self.dag_generator_fn()

    def _successor_cost_sum(self, task_id: int) -> float:
        assert self.dag is not None
        graph = self.dag.graph
        return sum(
            float(graph.nodes[succ].get("computation_cost", 0.0))
            for succ in graph.successors(task_id)
        )

    def _get_observation(self) -> np.ndarray:
        assert self.dag is not None
        graph = self.dag.graph
        features: list[float] = []

        for slot in range(self.max_ready_tasks):
            if slot < len(self.ready_tasks):
                task_id = self.ready_tasks[slot]
                node_data = graph.nodes[task_id]
                features.extend(
                    [
                        1.0,
                        float(task_id) / max(1, self.max_tasks - 1),
                        float(node_data["computation_cost"]),
                        float(graph.out_degree(task_id)),
                        self._successor_cost_sum(task_id),
                    ]
                )
            else:
                features.extend([0.0] * self.task_feature_dim)

        for resource in self.resource_config.resources:
            resource_available_time = self._resource_available_time(resource.id)
            features.extend(
                [
                    float(resource_available_time),
                    TIER_ENCODING.get(resource.tier, 0.0),
                    float(resource.compute_power),
                    float(resource.bandwidth),
                ]
            )

        total_tasks = graph.number_of_nodes()
        features.extend(
            [
                len(self.completed_tasks) / max(1, total_tasks),
                float(self.current_makespan),
            ]
        )
        return np.asarray(features, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        ready_slots = min(len(self.ready_tasks), self.max_ready_tasks)
        for task_slot in range(ready_slots):
            start = task_slot * self.num_resources
            mask[start : start + self.num_resources] = True
        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.dag = self._call_dag_generator(seed)
        if self.dag.graph.number_of_nodes() > self.max_tasks:
            raise ValueError("generated DAG has more tasks than max_tasks")

        self.resource_config.reset()
        self.completed_tasks = set()
        self.task_assignments = {}
        self.task_times = {}
        self.resource_events = {resource.id: [] for resource in self.resource_config.resources}
        self.current_makespan = 0.0
        self.ready_tasks = get_ready_tasks(self.dag, self.completed_tasks)
        return self._get_observation(), {"ready_tasks": list(self.ready_tasks), "makespan": 0.0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.dag is not None
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"action {action} is outside action space")

        task_slot = int(action) // self.num_resources
        resource_index = int(action) % self.num_resources
        if task_slot >= len(self.ready_tasks):
            raise ValueError(f"action {action} selects a non-ready task slot")

        task_id = self.ready_tasks[task_slot]
        resource = self.resource_config.resources[resource_index]
        start_time, finish_time = self._schedule_task(task_id, resource)

        self.completed_tasks.add(task_id)
        self.task_assignments[task_id] = resource.id
        self.task_times[task_id] = (start_time, finish_time)
        self.current_makespan = max(self.current_makespan, finish_time)

        terminated = len(self.completed_tasks) == self.dag.graph.number_of_nodes()
        self.ready_tasks = get_ready_tasks(self.dag, self.completed_tasks)
        if not terminated and not self.ready_tasks:
            raise RuntimeError("no legal actions remain before all tasks are completed")

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

    def _schedule_task(self, task_id: int, resource: Resource) -> tuple[float, float]:
        assert self.dag is not None
        graph = self.dag.graph
        data_ready_time = 0.0

        for predecessor in graph.predecessors(task_id):
            if predecessor not in self.task_times:
                raise RuntimeError(f"predecessor {predecessor} has not been scheduled")
            _, pred_finish = self.task_times[predecessor]
            pred_resource_id = self.task_assignments[predecessor]
            data_size = float(graph.edges[predecessor, task_id].get("data_size", 0.0))
            communication_time = self.resource_config.get_communication_time(
                data_size,
                pred_resource_id,
                resource,
            )
            data_ready_time = max(data_ready_time, pred_finish + communication_time)

        start_time = max(resource.available_time, data_ready_time)
        execution_time = self.resource_config.get_execution_time(graph.nodes[task_id], resource)
        start_time, finish_time = find_earliest_slot(
            self.resource_events[resource.id],
            data_ready_time,
            execution_time,
        )
        self.resource_events[resource.id].append(
            ScheduledEvent(task_id=task_id, start_time=start_time, finish_time=finish_time)
        )
        self.resource_events[resource.id].sort(key=lambda event: event.start_time)
        resource.available_time = self._resource_available_time(resource.id)
        return start_time, finish_time

    def _resource_available_time(self, resource_id: str) -> float:
        events = self.resource_events.get(resource_id, [])
        if not events:
            return 0.0
        return max(event.finish_time for event in events)


def _run_random_policy_demo() -> None:
    config = load_resource_config(default_resource_config_path())
    env = SchedulingEnv(
        dag_generator_fn=lambda seed=None: generate_random_dag(10, edge_density=0.35, seed=seed),
        resource_config=config,
        max_tasks=20,
    )

    rng = np.random.default_rng(7)
    for episode in range(10):
        _, _ = env.reset(seed=episode)
        terminated = False
        steps = 0
        info: dict[str, Any] = {"makespan": 0.0}
        while not terminated:
            legal_actions = np.flatnonzero(env.action_masks())
            if legal_actions.size == 0:
                raise RuntimeError("random policy found no legal actions before termination")
            action = int(rng.choice(legal_actions))
            _, _, terminated, _, info = env.step(action)
            steps += 1
        print(f"episode {episode + 1}: steps={steps}, makespan={info['makespan']:.6f}")


if __name__ == "__main__":
    _run_random_policy_demo()
