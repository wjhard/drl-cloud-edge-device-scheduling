import networkx as nx

from baselines.milp_optimal_scheduler import MILPOptimalScheduler
from env.dag_generator import DAGTask
from env.resource_config import Resource, ResourceConfig


def test_milp_finds_known_optimal_parallel_schedule() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, task_id=0, computation_cost=2.0)
    graph.add_node(1, task_id=1, computation_cost=1.0)
    dag = DAGTask(graph=graph, source_tasks=[0, 1], sink_tasks=[0, 1])
    resources = ResourceConfig(
        [
            Resource(id="fast", tier="edge", compute_power=2.0, bandwidth=10.0),
            Resource(id="slow", tier="device", compute_power=1.0, bandwidth=10.0),
        ]
    )

    result = MILPOptimalScheduler(time_limit_seconds=10.0).solve(dag, resources)

    assert result.proven_optimal
    assert result.status == "optimal"
    assert result.schedule is not None
    assert abs(result.makespan - 1.0) <= 1e-7
    assert abs(result.best_bound - result.makespan) <= 1e-7


def test_milp_bound_parser_prefers_precise_progress_bound() -> None:
    log_text = """
Cbc0010I best solution 0.56, best possible 0.55087644
Lower bound:                    0.551
"""

    assert MILPOptimalScheduler._parse_best_bound(log_text) == 0.55087644
