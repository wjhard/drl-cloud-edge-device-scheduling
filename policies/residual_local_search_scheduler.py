from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
import random

from env.dag_generator import DAGTask
from env.resource_config import ResourceConfig
from env.scheduling_utils import ScheduledEvent, find_earliest_slot
from policies.rl_scheduler import RLScheduler
from scheduler_interface import BaseScheduler, ScheduleResult


@dataclass(frozen=True)
class LocalSearchStats:
    initial_makespan: float
    final_makespan: float
    evaluated_neighbors: int
    accepted_moves: int


@dataclass(frozen=True)
class LargeNeighborhoodStats:
    initial_makespan: float
    final_makespan: float
    iterations: int
    accepted_repairs: int
    evaluated_neighbors: int


def schedule_task_order(
    dag: DAGTask,
    resource_config: ResourceConfig,
    task_order: list[int],
) -> ScheduleResult:
    """Replay a topological order with the ranked environment's EFT rule."""
    nodes = set(int(node) for node in dag.graph.nodes)
    if len(task_order) != len(nodes) or set(task_order) != nodes:
        raise ValueError("task_order must contain every DAG task exactly once")

    position = {task_id: index for index, task_id in enumerate(task_order)}
    if any(position[int(src)] >= position[int(dst)] for src, dst in dag.graph.edges):
        raise ValueError("task_order violates DAG precedence constraints")

    resource_events: dict[str, list[ScheduledEvent]] = {
        resource.id: [] for resource in resource_config.resources
    }
    assignments: dict[int, str] = {}
    task_times: dict[int, tuple[float, float]] = {}
    result: ScheduleResult = {}

    for task_id in task_order:
        best_resource_id: str | None = None
        best_start = 0.0
        best_finish = float("inf")

        for resource in resource_config.resources:
            ready_time = 0.0
            for predecessor in dag.graph.predecessors(task_id):
                predecessor = int(predecessor)
                predecessor_resource = assignments[predecessor]
                predecessor_finish = task_times[predecessor][1]
                data_size = float(dag.graph.edges[predecessor, task_id].get("data_size", 0.0))
                communication = resource_config.get_communication_time(
                    data_size,
                    predecessor_resource,
                    resource,
                )
                ready_time = max(ready_time, predecessor_finish + communication)

            duration = resource_config.get_execution_time(dag.graph.nodes[task_id], resource)
            start_time, finish_time = find_earliest_slot(
                resource_events[resource.id],
                ready_time,
                duration,
            )
            if finish_time < best_finish:
                best_resource_id = resource.id
                best_start = start_time
                best_finish = finish_time

        if best_resource_id is None:
            raise RuntimeError("no resource available while replaying task order")
        assignments[task_id] = best_resource_id
        task_times[task_id] = (best_start, best_finish)
        resource_events[best_resource_id].append(
            ScheduledEvent(task_id=task_id, start_time=best_start, finish_time=best_finish)
        )
        resource_events[best_resource_id].sort(key=lambda event: event.start_time)
        result[task_id] = (best_resource_id, best_start, best_finish)

    return result


def improve_task_order(
    dag: DAGTask,
    resource_config: ResourceConfig,
    initial_order: list[int],
    max_passes: int = 3,
) -> tuple[ScheduleResult, list[int], LocalSearchStats]:
    """Best-improvement search over precedence-feasible single-task relocations."""
    if max_passes < 0:
        raise ValueError("max_passes must be non-negative")

    current_order = list(initial_order)
    current_schedule = schedule_task_order(dag, resource_config, current_order)
    current_makespan = BaseScheduler.compute_makespan(BaseScheduler, current_schedule)
    initial_makespan = current_makespan
    evaluated_neighbors = 0
    accepted_moves = 0

    for _ in range(max_passes):
        best_order = current_order
        best_schedule = current_schedule
        best_makespan = current_makespan
        seen: set[tuple[int, ...]] = set()

        for old_index in range(len(current_order)):
            for new_index in range(len(current_order)):
                if old_index == new_index:
                    continue
                candidate = list(current_order)
                task_id = candidate.pop(old_index)
                candidate.insert(new_index, task_id)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                positions = {task: index for index, task in enumerate(candidate)}
                if any(positions[int(src)] >= positions[int(dst)] for src, dst in dag.graph.edges):
                    continue

                candidate_schedule = schedule_task_order(dag, resource_config, candidate)
                candidate_makespan = BaseScheduler.compute_makespan(BaseScheduler, candidate_schedule)
                evaluated_neighbors += 1
                if candidate_makespan < best_makespan - 1e-12:
                    best_order = candidate
                    best_schedule = candidate_schedule
                    best_makespan = candidate_makespan

        if best_makespan >= current_makespan - 1e-12:
            break
        current_order = best_order
        current_schedule = best_schedule
        current_makespan = best_makespan
        accepted_moves += 1

    return (
        current_schedule,
        current_order,
        LocalSearchStats(
            initial_makespan=initial_makespan,
            final_makespan=current_makespan,
            evaluated_neighbors=evaluated_neighbors,
            accepted_moves=accepted_moves,
        ),
    )


def destroy_and_repair_order(
    dag: DAGTask,
    task_order: list[int],
    rng: random.Random,
    destroy_size: int,
) -> list[int]:
    """Release positional anchors for selected tasks and rebuild a topological order."""
    if destroy_size <= 0 or destroy_size > len(task_order):
        raise ValueError("destroy_size must be between one and the task count")
    destroyed = set(rng.sample(task_order, destroy_size))
    original_position = {task_id: index for index, task_id in enumerate(task_order)}
    priorities = {
        task_id: rng.uniform(-0.5, len(task_order) - 0.5)
        if task_id in destroyed
        else float(original_position[task_id])
        for task_id in task_order
    }

    indegree = {int(task): int(dag.graph.in_degree(task)) for task in dag.graph.nodes}
    ready: list[tuple[float, float, int]] = []
    for task_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (priorities[task_id], rng.random(), task_id))

    repaired: list[int] = []
    while ready:
        _, _, task_id = heapq.heappop(ready)
        repaired.append(task_id)
        for successor in dag.graph.successors(task_id):
            successor = int(successor)
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    ready,
                    (priorities[successor], rng.random(), successor),
                )
    if len(repaired) != len(task_order):
        raise RuntimeError("destroy-repair failed to produce a complete topological order")
    return repaired


def large_neighborhood_search(
    dag: DAGTask,
    resource_config: ResourceConfig,
    initial_order: list[int],
    rng: random.Random,
    iterations: int = 64,
    min_destroy_size: int = 2,
    max_destroy_size: int = 4,
    local_passes: int = 1,
) -> tuple[ScheduleResult, list[int], LargeNeighborhoodStats]:
    """Randomized destroy-repair search with deterministic best-only acceptance."""
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if min_destroy_size <= 0 or max_destroy_size < min_destroy_size:
        raise ValueError("invalid destroy-size range")

    best_order = list(initial_order)
    best_schedule = schedule_task_order(dag, resource_config, best_order)
    best_makespan = BaseScheduler.compute_makespan(BaseScheduler, best_schedule)
    initial_makespan = best_makespan
    accepted_repairs = 0
    evaluated_neighbors = 0

    for _ in range(iterations):
        destroy_size = rng.randint(
            min(min_destroy_size, len(best_order)),
            min(max_destroy_size, len(best_order)),
        )
        repaired_order = destroy_and_repair_order(dag, best_order, rng, destroy_size)
        repaired_schedule, repaired_order, local_stats = improve_task_order(
            dag,
            resource_config,
            repaired_order,
            max_passes=local_passes,
        )
        evaluated_neighbors += local_stats.evaluated_neighbors + 1
        repaired_makespan = BaseScheduler.compute_makespan(BaseScheduler, repaired_schedule)
        if repaired_makespan < best_makespan - 1e-12:
            best_order = repaired_order
            best_schedule = repaired_schedule
            best_makespan = repaired_makespan
            accepted_repairs += 1

    return (
        best_schedule,
        best_order,
        LargeNeighborhoodStats(
            initial_makespan=initial_makespan,
            final_makespan=best_makespan,
            iterations=iterations,
            accepted_repairs=accepted_repairs,
            evaluated_neighbors=evaluated_neighbors,
        ),
    )


class ResidualLocalSearchScheduler(BaseScheduler):
    """Best-of-N residual rollouts followed by topological-order local search."""

    def __init__(
        self,
        model_path: str | Path,
        max_tasks: int,
        num_samples: int = 64,
        max_passes: int = 3,
        normalize_observations: bool = True,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.residual = RLScheduler(
            model_path=model_path,
            max_tasks=max_tasks,
            deterministic=True,
            reward_mode="relative_heft",
            normalize_observations=normalize_observations,
            scheduler_mode="residual",
            num_samples=1,
        )
        self.num_samples = num_samples
        self.max_passes = max_passes
        self.last_initial_schedule: ScheduleResult | None = None
        self.last_initial_order: list[int] | None = None
        self.last_final_order: list[int] | None = None
        self.last_stats: LocalSearchStats | None = None

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        best_schedule: ScheduleResult | None = None
        best_order: list[int] | None = None
        best_makespan = float("inf")

        for _ in range(self.num_samples):
            schedule, order = self.residual._schedule_once_with_order(
                dag,
                resource_config,
                deterministic=False,
            )
            makespan = self.compute_makespan(schedule)
            if makespan < best_makespan:
                best_schedule = schedule
                best_order = order
                best_makespan = makespan

        deterministic_schedule, deterministic_order = self.residual._schedule_once_with_order(
            dag,
            resource_config,
            deterministic=True,
        )
        deterministic_makespan = self.compute_makespan(deterministic_schedule)
        if deterministic_makespan < best_makespan:
            best_schedule = deterministic_schedule
            best_order = deterministic_order

        if best_schedule is None or best_order is None:
            raise RuntimeError("residual sampling did not produce an initial schedule")

        improved_schedule, improved_order, stats = improve_task_order(
            dag,
            resource_config,
            best_order,
            max_passes=self.max_passes,
        )
        self.last_initial_schedule = best_schedule
        self.last_initial_order = best_order
        self.last_final_order = improved_order
        self.last_stats = stats
        return improved_schedule


class ResidualLargeNeighborhoodScheduler(BaseScheduler):
    """Residual Best-of-N, local search, then destroy-repair large-neighborhood search."""

    def __init__(
        self,
        model_path: str | Path,
        max_tasks: int,
        num_samples: int = 64,
        local_max_passes: int = 3,
        lns_iterations: int = 64,
        lns_local_passes: int = 1,
        normalize_observations: bool = True,
    ):
        self.local_scheduler = ResidualLocalSearchScheduler(
            model_path=model_path,
            max_tasks=max_tasks,
            num_samples=num_samples,
            max_passes=local_max_passes,
            normalize_observations=normalize_observations,
        )
        self.lns_iterations = lns_iterations
        self.lns_local_passes = lns_local_passes
        self.last_local_schedule: ScheduleResult | None = None
        self.last_local_order: list[int] | None = None
        self.last_final_order: list[int] | None = None
        self.last_stats: LargeNeighborhoodStats | None = None

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        local_schedule = self.local_scheduler.schedule(dag, resource_config)
        local_order = self.local_scheduler.last_final_order
        if local_order is None:
            raise RuntimeError("local-search phase did not expose its task order")
        rng = random.Random(random.getrandbits(64))
        final_schedule, final_order, stats = large_neighborhood_search(
            dag,
            resource_config,
            local_order,
            rng,
            iterations=self.lns_iterations,
            local_passes=self.lns_local_passes,
        )
        if self.compute_makespan(final_schedule) > self.compute_makespan(local_schedule) + 1e-9:
            raise RuntimeError("large-neighborhood search worsened its local-search input")
        self.last_local_schedule = local_schedule
        self.last_local_order = local_order
        self.last_final_order = final_order
        self.last_stats = stats
        return final_schedule
