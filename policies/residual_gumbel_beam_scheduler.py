from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch

from env.dag_generator import DAGTask
from env.resource_config import ResourceConfig
from env.scheduling_env_residual import SchedulingEnvResidual
from policies.residual_local_search_scheduler import (
    LargeNeighborhoodStats,
    LocalSearchStats,
    improve_task_order,
    large_neighborhood_search,
)
from policies.rl_scheduler import RLScheduler
from scheduler_interface import BaseScheduler, ScheduleResult


@dataclass
class _BeamNode:
    env: SchedulingEnvResidual
    observation: dict[str, np.ndarray]
    task_order: list[int]
    schedule: ScheduleResult
    score: float
    terminated: bool = False


class ResidualGumbelBeamScheduler(BaseScheduler):
    """Gumbel-diversified policy beam followed by local and large-neighborhood search."""

    def __init__(
        self,
        model_path: str | Path,
        max_tasks: int,
        beam_width: int = 16,
        gumbel_scale: float = 1.0,
        local_max_passes: int = 3,
        lns_iterations: int = 64,
        normalize_observations: bool = True,
    ):
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if gumbel_scale < 0:
            raise ValueError("gumbel_scale must be non-negative")
        self.residual = RLScheduler(
            model_path=model_path,
            max_tasks=max_tasks,
            deterministic=True,
            reward_mode="relative_heft",
            normalize_observations=normalize_observations,
            scheduler_mode="residual",
            num_samples=1,
        )
        self.max_tasks = max_tasks
        self.beam_width = beam_width
        self.gumbel_scale = gumbel_scale
        self.local_max_passes = local_max_passes
        self.lns_iterations = lns_iterations
        self.normalize_observations = normalize_observations
        self.last_beam_schedule: ScheduleResult | None = None
        self.last_beam_order: list[int] | None = None
        self.last_local_schedule: ScheduleResult | None = None
        self.last_local_stats: LocalSearchStats | None = None
        self.last_lns_stats: LargeNeighborhoodStats | None = None
        self.last_unique_complete_orders = 0

    @staticmethod
    def _gumbel(rng: np.random.Generator) -> float:
        uniform = float(rng.uniform(np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
        return -math.log(-math.log(uniform))

    def _action_probabilities(self, node: _BeamNode) -> np.ndarray:
        mask = node.env.action_masks()
        observation_tensor, _ = self.residual.model.policy.obs_to_tensor(node.observation)
        with torch.no_grad():
            distribution = self.residual.model.policy.get_distribution(
                observation_tensor,
                action_masks=mask.reshape(1, -1),
            )
        probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]
        return probabilities

    def _beam_candidates(
        self,
        dag: DAGTask,
        resource_config: ResourceConfig,
        rng: np.random.Generator,
    ) -> list[_BeamNode]:
        root_env = SchedulingEnvResidual(
            dag_generator_fn=lambda seed=None: dag,
            resource_config=copy.deepcopy(resource_config),
            max_tasks=self.max_tasks,
            reward_mode="raw",
            normalize_observations=self.normalize_observations,
        )
        root_observation, _ = root_env.reset()
        beam = [_BeamNode(root_env, root_observation, [], {}, 0.0)]

        for _ in range(dag.graph.number_of_nodes()):
            expanded: dict[tuple[int, ...], _BeamNode] = {}
            for node in beam:
                probabilities = self._action_probabilities(node)
                legal_actions = np.flatnonzero(node.env.action_masks())
                for action_value in legal_actions:
                    action = int(action_value)
                    child_env = copy.deepcopy(node.env)
                    child_observation, _, terminated, truncated, info = child_env.step(action)
                    if truncated:
                        raise RuntimeError("beam-search environment truncated")
                    task_id = int(info["task_id"])
                    child_order = [*node.task_order, task_id]
                    child_schedule = dict(node.schedule)
                    child_schedule[task_id] = (
                        str(info["resource_id"]),
                        float(info["start_time"]),
                        float(info["finish_time"]),
                    )
                    probability = max(float(probabilities[action]), 1e-30)
                    score = (
                        node.score
                        + math.log(probability)
                        + self.gumbel_scale * self._gumbel(rng)
                    )
                    child = _BeamNode(
                        child_env,
                        child_observation,
                        child_order,
                        child_schedule,
                        score,
                        terminated,
                    )
                    key = tuple(child_order)
                    incumbent = expanded.get(key)
                    if incumbent is None or child.score > incumbent.score:
                        expanded[key] = child
            beam = sorted(expanded.values(), key=lambda item: item.score, reverse=True)[
                : self.beam_width
            ]
            if not beam:
                raise RuntimeError("beam search exhausted all legal prefixes")

        complete = [node for node in beam if node.terminated]
        if not complete:
            raise RuntimeError("beam search produced no complete task order")
        return complete

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        entropy = random.getrandbits(64)
        beam_rng = np.random.default_rng(entropy)
        complete = self._beam_candidates(dag, resource_config, beam_rng)
        self.last_unique_complete_orders = len({tuple(node.task_order) for node in complete})
        best_node = min(complete, key=lambda node: self.compute_makespan(node.schedule))
        beam_schedule = best_node.schedule
        beam_order = best_node.task_order

        local_schedule, local_order, local_stats = improve_task_order(
            dag,
            resource_config,
            beam_order,
            max_passes=self.local_max_passes,
        )
        lns_schedule, _, lns_stats = large_neighborhood_search(
            dag,
            resource_config,
            local_order,
            random.Random(random.getrandbits(64)),
            iterations=self.lns_iterations,
            local_passes=1,
        )
        self.last_beam_schedule = beam_schedule
        self.last_beam_order = beam_order
        self.last_local_schedule = local_schedule
        self.last_local_stats = local_stats
        self.last_lns_stats = lns_stats
        return lns_schedule
