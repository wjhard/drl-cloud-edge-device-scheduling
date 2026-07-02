from __future__ import annotations

import os
import sys

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import DAGTask, generate_random_dag
from env.resource_config import Resource, ResourceConfig, default_resource_config_path, load_resource_config
from env.scheduling_utils import ScheduledEvent, find_earliest_slot
from scheduler_interface import BaseScheduler, ScheduleResult


class HEFTScheduler(BaseScheduler):
    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        if not nx.is_directed_acyclic_graph(dag.graph):
            raise ValueError("HEFT requires a DAG")

        resource_config.reset()
        task_order = self._task_order(dag, resource_config)

        schedule_result: ScheduleResult = {}
        resource_events: dict[str, list[ScheduledEvent]] = {
            resource.id: [] for resource in resource_config.resources
        }

        for task_id in task_order:
            best_resource: Resource | None = None
            best_start = 0.0
            best_finish = float("inf")

            for resource in resource_config.resources:
                ready_time = self._dependency_ready_time(task_id, resource, dag, resource_config, schedule_result)
                duration = resource_config.get_execution_time(dag.graph.nodes[task_id], resource)
                start, finish = find_earliest_slot(resource_events[resource.id], ready_time, duration)
                if finish < best_finish:
                    best_resource = resource
                    best_start = start
                    best_finish = finish

            if best_resource is None:
                raise RuntimeError("no resource available for HEFT scheduling")
            schedule_result[task_id] = (best_resource.id, best_start, best_finish)
            resource_events[best_resource.id].append(
                ScheduledEvent(task_id=task_id, start_time=best_start, finish_time=best_finish)
            )
            resource_events[best_resource.id].sort(key=lambda event: event.start_time)

        return schedule_result

    def _task_order(self, dag: DAGTask, resource_config: ResourceConfig) -> list[int]:
        ranks = self._compute_upward_ranks(dag, resource_config)
        topological_index = {task_id: index for index, task_id in enumerate(nx.topological_sort(dag.graph))}
        return sorted(dag.graph.nodes, key=lambda task_id: (-ranks[task_id], topological_index[task_id]))

    def _compute_upward_ranks(self, dag: DAGTask, resource_config: ResourceConfig) -> dict[int, float]:
        ranks: dict[int, float] = {}

        def rank(task_id: int) -> float:
            if task_id in ranks:
                return ranks[task_id]

            avg_compute = self._average_compute_time(task_id, dag, resource_config)
            successor_terms = []
            for successor in dag.graph.successors(task_id):
                data_size = float(dag.graph.edges[task_id, successor].get("data_size", 0.0))
                successor_terms.append(
                    self._average_communication_time(data_size, resource_config) + rank(successor)
                )
            ranks[task_id] = avg_compute + (max(successor_terms) if successor_terms else 0.0)
            return ranks[task_id]

        for task_id in reversed(list(nx.topological_sort(dag.graph))):
            rank(task_id)
        return ranks

    def _average_compute_time(
        self,
        task_id: int,
        dag: DAGTask,
        resource_config: ResourceConfig,
    ) -> float:
        return sum(
            resource_config.get_execution_time(dag.graph.nodes[task_id], resource)
            for resource in resource_config.resources
        ) / len(resource_config.resources)

    def _average_communication_time(self, data_size: float, resource_config: ResourceConfig) -> float:
        resources = resource_config.resources
        if len(resources) <= 1:
            return 0.0
        times = [
            resource_config.get_communication_time(data_size, source, target)
            for source in resources
            for target in resources
        ]
        return sum(times) / len(times)

    def _dependency_ready_time(
        self,
        task_id: int,
        resource: Resource,
        dag: DAGTask,
        resource_config: ResourceConfig,
        schedule_result: ScheduleResult,
    ) -> float:
        ready_time = 0.0
        for predecessor in dag.graph.predecessors(task_id):
            if predecessor not in schedule_result:
                raise RuntimeError(f"predecessor {predecessor} must be scheduled before {task_id}")
            pred_resource_id, _, pred_finish = schedule_result[predecessor]
            data_size = float(dag.graph.edges[predecessor, task_id].get("data_size", 0.0))
            communication_time = resource_config.get_communication_time(data_size, pred_resource_id, resource)
            ready_time = max(ready_time, pred_finish + communication_time)
        return ready_time

def _run_heft_demo() -> None:
    dag = generate_random_dag(num_tasks=10, edge_density=0.35, seed=42)
    resource_config = load_resource_config(default_resource_config_path())
    scheduler = HEFTScheduler()
    result = scheduler.schedule(dag, resource_config)

    print("HEFT schedule:")
    for task_id, (resource_id, start_time, finish_time) in sorted(result.items()):
        print(
            f"task {task_id}: resource={resource_id}, "
            f"start={start_time:.6f}, finish={finish_time:.6f}"
        )
    print(f"makespan: {scheduler.compute_makespan(result):.6f}")


if __name__ == "__main__":
    _run_heft_demo()
