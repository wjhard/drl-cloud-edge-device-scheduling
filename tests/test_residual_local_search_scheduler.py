from __future__ import annotations

import networkx as nx

from env.dag_generator import DAGTask, generate_random_dag
from env.resource_config import Resource, ResourceConfig
import random

from policies.residual_local_search_scheduler import (
    destroy_and_repair_order,
    improve_task_order,
    large_neighborhood_search,
    schedule_task_order,
)


def _resources() -> ResourceConfig:
    return ResourceConfig(
        [
            Resource("fast", "cloud", compute_power=10.0, bandwidth=10.0),
            Resource("slow", "edge", compute_power=4.0, bandwidth=5.0),
        ]
    )


def test_local_search_preserves_precedence_and_never_worsens() -> None:
    dag = generate_random_dag(num_tasks=12, edge_density=0.35, seed=9123)
    initial_order = [int(task) for task in nx.topological_sort(dag.graph)]
    initial_schedule = schedule_task_order(dag, _resources(), initial_order)
    improved_schedule, improved_order, stats = improve_task_order(
        dag,
        _resources(),
        initial_order,
        max_passes=3,
    )

    positions = {task: index for index, task in enumerate(improved_order)}
    assert all(positions[int(src)] < positions[int(dst)] for src, dst in dag.graph.edges)
    assert max(value[2] for value in improved_schedule.values()) <= max(
        value[2] for value in initial_schedule.values()
    ) + 1e-12
    assert stats.final_makespan <= stats.initial_makespan + 1e-12


def test_schedule_task_order_rejects_non_topological_order() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, task_id=0, computation_cost=1.0)
    graph.add_node(1, task_id=1, computation_cost=1.0)
    graph.add_edge(0, 1, data_size=1.0)
    dag = DAGTask(graph=graph, source_tasks=[0], sink_tasks=[1])

    try:
        schedule_task_order(dag, _resources(), [1, 0])
    except ValueError as error:
        assert "precedence" in str(error)
    else:
        raise AssertionError("non-topological order should be rejected")


def test_large_neighborhood_search_preserves_precedence_and_never_worsens() -> None:
    dag = generate_random_dag(num_tasks=14, edge_density=0.3, seed=4321)
    initial_order = [int(task) for task in nx.topological_sort(dag.graph)]
    repaired = destroy_and_repair_order(dag, initial_order, random.Random(17), destroy_size=4)
    repaired_positions = {task: index for index, task in enumerate(repaired)}
    assert all(repaired_positions[int(src)] < repaired_positions[int(dst)] for src, dst in dag.graph.edges)

    initial_schedule = schedule_task_order(dag, _resources(), initial_order)
    final_schedule, final_order, stats = large_neighborhood_search(
        dag,
        _resources(),
        initial_order,
        random.Random(23),
        iterations=12,
    )
    final_positions = {task: index for index, task in enumerate(final_order)}
    assert all(final_positions[int(src)] < final_positions[int(dst)] for src, dst in dag.graph.edges)
    assert max(value[2] for value in final_schedule.values()) <= max(
        value[2] for value in initial_schedule.values()
    ) + 1e-12
    assert stats.final_makespan <= stats.initial_makespan + 1e-12
