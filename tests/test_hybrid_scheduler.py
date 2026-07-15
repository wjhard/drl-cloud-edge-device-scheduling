import networkx as nx

from baselines.hybrid_scheduler import HybridScheduler
from baselines.milp_optimal_scheduler import MILPSolveResult
from env.dag_generator import DAGTask
from env.resource_config import Resource, ResourceConfig
from scheduler_interface import BaseScheduler


class RecordingFallbackScheduler(BaseScheduler):
    def __init__(self) -> None:
        self.calls = 0

    def schedule(self, dag, resource_config):
        self.calls += 1
        resource = resource_config.resources[0]
        current_time = 0.0
        result = {}
        for task_id in nx.topological_sort(dag.graph):
            duration = resource_config.get_execution_time(dag.graph.nodes[task_id], resource)
            result[task_id] = (resource.id, current_time, current_time + duration)
            current_time += duration
        return result


class UnprovenMILPScheduler:
    def solve(self, dag, resource_config):
        return MILPSolveResult(
            schedule={0: ("r0", 0.0, 1.0)},
            status="feasible_not_proven_optimal",
            proven_optimal=False,
            makespan=1.0,
            best_bound=0.5,
            relative_gap=0.5,
            solve_time_seconds=0.01,
            solver_status="Optimal",
            solver_solution_status="Solution Found",
        )


def _dag(num_tasks: int) -> DAGTask:
    graph = nx.DiGraph()
    for task_id in range(num_tasks):
        graph.add_node(task_id, task_id=task_id, computation_cost=1.0)
    return DAGTask(graph=graph, source_tasks=list(graph.nodes), sink_tasks=list(graph.nodes))


def _resources() -> ResourceConfig:
    return ResourceConfig([Resource("r0", "cloud", compute_power=1.0, bandwidth=10.0)])


def test_hybrid_uses_proven_milp_solution_for_small_dag() -> None:
    fallback = RecordingFallbackScheduler()
    scheduler = HybridScheduler(fallback, task_threshold=2, milp_time_limit_seconds=10.0)

    schedule = scheduler.schedule(_dag(2), _resources())

    assert len(schedule) == 2
    assert fallback.calls == 0
    assert scheduler.last_decision is not None
    assert scheduler.last_decision.selected_method == "milp_optimal"


def test_hybrid_falls_back_for_large_dag() -> None:
    fallback = RecordingFallbackScheduler()
    scheduler = HybridScheduler(fallback, task_threshold=1, milp_time_limit_seconds=10.0)

    scheduler.schedule(_dag(2), _resources())

    assert fallback.calls == 1
    assert scheduler.last_decision is not None
    assert scheduler.last_decision.reason == "task_count_above_threshold"


def test_hybrid_falls_back_when_milp_is_not_proven_optimal() -> None:
    fallback = RecordingFallbackScheduler()
    scheduler = HybridScheduler(
        fallback,
        task_threshold=2,
        milp_time_limit_seconds=10.0,
        milp_scheduler=UnprovenMILPScheduler(),
    )

    scheduler.schedule(_dag(1), _resources())

    assert fallback.calls == 1
    assert scheduler.last_decision is not None
    assert scheduler.last_decision.reason == "milp_not_proven_optimal"
