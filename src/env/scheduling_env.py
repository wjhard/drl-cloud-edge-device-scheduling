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
from env.observation_normalizer import ObservationNormalizer
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
        reward_mode: str = "raw",
        normalize_observations: bool = False,
        include_upward_rank_feature: bool = False,
    ):
        super().__init__()
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        if reward_mode not in {"raw", "relative_heft"}:
            raise ValueError("reward_mode must be either 'raw' or 'relative_heft'")

        self.dag_generator_fn = dag_generator_fn
        self.resource_config = resource_config
        self.max_tasks = max_tasks
        self.max_ready_tasks = max_tasks
        self.num_resources = len(resource_config.resources)
        self.reward_mode = reward_mode
        self.normalize_observations = normalize_observations
        self.include_upward_rank_feature = include_upward_rank_feature
        self.observation_normalizer = ObservationNormalizer() if normalize_observations else None

        self.task_feature_dim = 6 if include_upward_rank_feature else 5
        self.resource_feature_dim = 4
        self.global_feature_dim = 2
        self.observation_space = spaces.Dict(
            {
                "task_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_tasks, self.task_feature_dim),
                    dtype=np.float32,
                ),
                "task_valid_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_tasks,),
                    dtype=np.float32,
                ),
                "task_adjacency": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_tasks, self.max_tasks),
                    dtype=np.float32,
                ),
                "ready_task_node_ids": spaces.Box(
                    low=0,
                    high=max(0, self.max_tasks - 1),
                    shape=(self.max_ready_tasks,),
                    dtype=np.int64,
                ),
                "resource_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_resources, self.resource_feature_dim),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
            }
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
        self._heft_makespan_ref: float | None = None
        self._upward_ranks: dict[int, float] = {}

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

    def _task_features(self, task_id: int) -> list[float]:
        assert self.dag is not None
        graph = self.dag.graph
        node_data = graph.nodes[task_id]
        features = [
            1.0,
            float(task_id) / max(1, self.max_tasks - 1),
            float(node_data["computation_cost"]),
            float(graph.out_degree(task_id)),
            self._successor_cost_sum(task_id),
        ]
        if self.include_upward_rank_feature:
            features.append(float(self._upward_ranks.get(task_id, 0.0)))
        return features

    def _get_observation(self) -> dict[str, np.ndarray]:
        assert self.dag is not None
        graph = self.dag.graph
        task_features = np.zeros((self.max_tasks, self.task_feature_dim), dtype=np.float32)
        task_valid_mask = np.zeros((self.max_tasks,), dtype=np.float32)
        task_adjacency = np.zeros((self.max_tasks, self.max_tasks), dtype=np.float32)
        ready_task_node_ids = np.zeros((self.max_ready_tasks,), dtype=np.int64)
        resource_features = np.zeros((self.num_resources, self.resource_feature_dim), dtype=np.float32)

        for task_id in graph.nodes:
            if 0 <= int(task_id) < self.max_tasks:
                task_features[int(task_id)] = np.asarray(self._task_features(int(task_id)), dtype=np.float32)
                task_valid_mask[int(task_id)] = 1.0
                task_adjacency[int(task_id), int(task_id)] = 1.0

        for src, dst in graph.edges:
            if 0 <= int(src) < self.max_tasks and 0 <= int(dst) < self.max_tasks:
                task_adjacency[int(src), int(dst)] = 1.0
                task_adjacency[int(dst), int(src)] = 1.0

        for slot, task_id in enumerate(self.ready_tasks[: self.max_ready_tasks]):
            ready_task_node_ids[slot] = int(task_id)

        for index, resource in enumerate(self.resource_config.resources):
            resource_available_time = self._resource_available_time(resource.id)
            resource_features[index] = np.asarray(
                [
                    float(resource_available_time),
                    TIER_ENCODING.get(resource.tier, 0.0),
                    float(resource.compute_power),
                    float(resource.bandwidth),
                ],
                dtype=np.float32,
            )

        total_tasks = graph.number_of_nodes()
        global_features = np.asarray(
            [
                len(self.completed_tasks) / max(1, total_tasks),
                float(self.current_makespan),
            ],
            dtype=np.float32,
        )
        observation = {
            "task_features": task_features,
            "task_valid_mask": task_valid_mask,
            "task_adjacency": task_adjacency,
            "ready_task_node_ids": ready_task_node_ids,
            "resource_features": resource_features,
            "global_features": global_features,
        }
        if self.observation_normalizer is not None:
            observation = self.observation_normalizer.normalize(observation)
        return observation

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
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
        self._heft_makespan_ref = self._compute_heft_makespan_reference()
        self._upward_ranks = self._compute_upward_ranks_for_observation()
        self.ready_tasks = get_ready_tasks(self.dag, self.completed_tasks)
        return self._get_observation(), {"ready_tasks": list(self.ready_tasks), "makespan": 0.0}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
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

    def _compute_heft_makespan_reference(self) -> float | None:
        if self.reward_mode != "relative_heft":
            return None
        assert self.dag is not None

        from baselines.heft_scheduler import HEFTScheduler

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
        scheduler = HEFTScheduler()
        schedule_result = scheduler.schedule(self.dag, reference_config)
        return scheduler.compute_makespan(schedule_result)

    def _compute_upward_ranks_for_observation(self) -> dict[int, float]:
        if not self.include_upward_rank_feature:
            return {}
        assert self.dag is not None

        from baselines.heft_scheduler import compute_upward_ranks

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
