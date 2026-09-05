from __future__ import annotations

import networkx as nx
from pathlib import Path
import time

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_wide_parallel_dag
from env.dag_generator import generate_random_dag
from env.resource_config import Resource, ResourceConfig
from policies.residual_local_search_scheduler import (
    describe_dag_shape,
    improve_task_order,
    ResidualLargeNeighborhoodScheduler,
    schedule_task_order,
    width_aware_topological_order,
)


def _resources() -> ResourceConfig:
    return ResourceConfig(
        [
            Resource("cloud", "cloud", compute_power=100.0, bandwidth=1000.0),
            Resource("edge", "edge", compute_power=30.0, bandwidth=200.0),
            Resource("device", "device", compute_power=5.0, bandwidth=20.0),
        ]
    )


def test_wide_parallel_generator_has_a_broad_frontier() -> None:
    dag = generate_wide_parallel_dag(num_tasks=25, edge_density=0.12, seed=20260830)
    shape = describe_dag_shape(dag)

    assert nx.is_directed_acyclic_graph(dag.graph)
    assert shape.level_width >= 8
    assert shape.width_ratio >= 0.30
    assert shape.longest_path_edges <= 2


def test_width_aware_order_is_complete_and_topological() -> None:
    dag = generate_wide_parallel_dag(num_tasks=20, edge_density=0.15, seed=17)
    order = width_aware_topological_order(dag, _resources())
    positions = {task_id: index for index, task_id in enumerate(order)}

    assert len(order) == dag.graph.number_of_nodes()
    assert set(order) == {int(task_id) for task_id in dag.graph.nodes}
    assert all(positions[int(src)] < positions[int(dst)] for src, dst in dag.graph.edges)


def test_heft_anchor_replay_matches_heft_makespan() -> None:
    dag = generate_wide_parallel_dag(num_tasks=18, edge_density=0.10, seed=99)
    resources = _resources()
    heft = HEFTScheduler()
    heft_schedule = heft.schedule(dag, resources)
    heft_makespan = heft.compute_makespan(heft_schedule)

    replayed = schedule_task_order(
        dag,
        _resources(),
        heft._task_order(dag, _resources()),
    )
    assert abs(heft.compute_makespan(replayed) - heft_makespan) <= 1e-12


def test_wide_shape_seeded_generator_is_reproducible() -> None:
    first = generate_wide_parallel_dag(num_tasks=16, edge_density=0.12, seed=123)
    second = generate_wide_parallel_dag(num_tasks=16, edge_density=0.12, seed=123)
    assert list(first.graph.nodes(data=True)) == list(second.graph.nodes(data=True))
    assert list(first.graph.edges(data=True)) == list(second.graph.edges(data=True))


def test_expired_deadline_returns_a_valid_non_worsened_schedule() -> None:
    dag = generate_random_dag(num_tasks=18, edge_density=0.35, seed=1234)
    initial_order = list(nx.topological_sort(dag.graph))
    initial_schedule = schedule_task_order(dag, _resources(), initial_order)
    schedule, order, stats = improve_task_order(
        dag,
        _resources(),
        initial_order,
        max_passes=3,
        deadline=time.perf_counter() - 1.0,
    )

    assert order == initial_order
    assert stats.terminated_by_time_limit is True
    assert HEFTScheduler().compute_makespan(schedule) == HEFTScheduler().compute_makespan(
        initial_schedule
    )


def test_adaptive_scheduler_skips_residual_model_beyond_capacity() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dag = generate_random_dag(num_tasks=40, edge_density=0.25, seed=55)
    scheduler = ResidualLargeNeighborhoodScheduler(
        model_path=project_root / "training/checkpoints/ppo_mlp_residual",
        max_tasks=30,
        num_samples=1,
        local_max_passes=0,
        lns_iterations=0,
        normalize_observations=True,
        random_seed=17,
        portfolio_mode="adaptive",
    )
    resources = _resources()
    schedule = scheduler.schedule(dag, resources)
    heft = HEFTScheduler()
    heft_schedule = heft.schedule(dag, _resources())

    assert scheduler.last_residual_skipped_reason is not None
    assert scheduler.last_selected_source in {"heft_anchor", "heft_lns"}
    assert scheduler.compute_makespan(schedule) <= heft.compute_makespan(heft_schedule) + 1e-12
