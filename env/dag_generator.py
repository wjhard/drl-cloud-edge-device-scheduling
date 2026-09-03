from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
from networkx.readwrite import json_graph


@dataclass
class DAGTask:
    graph: nx.DiGraph
    source_tasks: list[int]
    sink_tasks: list[int]


def _build_dag_task(graph: nx.DiGraph) -> DAGTask:
    source_tasks = sorted(node for node in graph.nodes if graph.in_degree(node) == 0)
    sink_tasks = sorted(node for node in graph.nodes if graph.out_degree(node) == 0)
    return DAGTask(graph=graph, source_tasks=source_tasks, sink_tasks=sink_tasks)


def generate_random_dag(
    num_tasks: int,
    edge_density: float = 0.3,
    min_cost: float = 1.0,
    max_cost: float = 10.0,
    seed: int | None = None,
) -> DAGTask:
    """Generate a random layered DAG backed by networkx.DiGraph."""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if not 0.0 <= edge_density <= 1.0:
        raise ValueError("edge_density must be between 0 and 1")
    if min_cost <= 0 or max_cost < min_cost:
        raise ValueError("cost bounds must satisfy 0 < min_cost <= max_cost")

    rng = random.Random(seed)
    graph = nx.DiGraph()
    num_levels = max(2, min(num_tasks, int(math.sqrt(num_tasks)) + 1))

    levels: dict[int, int] = {}
    for task_id in range(num_tasks):
        level = rng.randrange(num_levels)
        levels[task_id] = level
        graph.add_node(
            task_id,
            task_id=task_id,
            level=level,
            computation_cost=rng.uniform(min_cost, max_cost),
        )

    for src in range(num_tasks):
        for dst in range(num_tasks):
            if levels[src] < levels[dst] and rng.random() < edge_density:
                graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("internal error: generated graph is not a DAG")
    return _build_dag_task(graph)


def generate_wide_parallel_dag(
    num_tasks: int,
    edge_density: float = 0.15,
    min_cost: float = 1.0,
    max_cost: float = 10.0,
    seed: int | None = None,
) -> DAGTask:
    """Generate a connected, low-density layered DAG with a broad frontier.

    The ordinary generator samples levels independently and can accidentally
    produce a narrow graph even when a low edge density is requested.  This
    constructor is used by the robustness curriculum and structural tests to
    guarantee a wide-parallel workload rather than merely hoping for one.
    """

    if num_tasks < 3:
        raise ValueError("wide-parallel DAGs require at least three tasks")
    if not 0.0 <= edge_density <= 1.0:
        raise ValueError("edge_density must be between 0 and 1")
    if min_cost <= 0 or max_cost < min_cost:
        raise ValueError("cost bounds must satisfy 0 < min_cost <= max_cost")

    rng = random.Random(seed)
    source_count = max(1, num_tasks // 5)
    sink_count = max(1, num_tasks // 5)
    middle_count = num_tasks - source_count - sink_count
    if middle_count <= 0:
        source_count = 1
        sink_count = 1
        middle_count = num_tasks - 2
    layers = [
        list(range(source_count)),
        list(range(source_count, source_count + middle_count)),
        list(range(source_count + middle_count, num_tasks)),
    ]

    graph = nx.DiGraph()
    for level, nodes in enumerate(layers):
        for task_id in nodes:
            graph.add_node(
                task_id,
                task_id=task_id,
                level=level,
                computation_cost=rng.uniform(min_cost, max_cost),
            )

    # Ensure every non-source task has a predecessor and every non-sink task
    # has a successor.  Additional edges preserve the three-layer DAG shape.
    for src in layers[0]:
        dst = rng.choice(layers[1])
        graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))
    for dst in layers[1]:
        src = rng.choice(layers[0])
        graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))
    for src in layers[1]:
        dst = rng.choice(layers[2])
        graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))
    for dst in layers[2]:
        src = rng.choice(layers[1])
        graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))

    for left, right in ((layers[0], layers[1]), (layers[1], layers[2])):
        for src in left:
            for dst in right:
                if rng.random() < edge_density:
                    graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))
    return _build_dag_task(graph)


def generate_deep_chain_dag(
    num_tasks: int,
    edge_density: float = 0.65,
    min_cost: float = 1.0,
    max_cost: float = 10.0,
    seed: int | None = None,
) -> DAGTask:
    """Generate a mostly serial layered DAG for robustness training."""

    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if not 0.0 <= edge_density <= 1.0:
        raise ValueError("edge_density must be between 0 and 1")
    rng = random.Random(seed)
    graph = nx.DiGraph()
    for task_id in range(num_tasks):
        graph.add_node(
            task_id,
            task_id=task_id,
            level=task_id,
            computation_cost=rng.uniform(min_cost, max_cost),
        )
        if task_id > 0:
            graph.add_edge(
                task_id - 1,
                task_id,
                data_size=rng.uniform(1.0, 10.0),
            )
    # Sparse skip edges increase structural diversity without removing the
    # guaranteed chain backbone.
    for src in range(num_tasks):
        for dst in range(src + 2, num_tasks):
            if rng.random() < edge_density * 0.15:
                graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))
    return _build_dag_task(graph)


def get_ready_tasks(dag: DAGTask, completed_tasks: set[int] | Iterable[int]) -> list[int]:
    completed = set(completed_tasks)
    ready_tasks: list[int] = []
    for task_id in nx.topological_sort(dag.graph):
        if task_id in completed:
            continue
        if all(predecessor in completed for predecessor in dag.graph.predecessors(task_id)):
            ready_tasks.append(task_id)
    return ready_tasks


def save_dag_to_json(dag: DAGTask, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(dag.graph, edges="links")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_dag_from_json(filepath: str | Path) -> DAGTask:
    data = json.loads(Path(filepath).read_text(encoding="utf-8"))
    graph = json_graph.node_link_graph(data, edges="links", directed=True)
    mapping = {node: int(node) for node in graph.nodes if not isinstance(node, int)}
    if mapping:
        graph = nx.relabel_nodes(graph, mapping)
    return _build_dag_task(graph)


def _print_graph_stats(dag: DAGTask) -> None:
    graph = dag.graph
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    avg_in_degree = sum(dict(graph.in_degree()).values()) / node_count
    avg_out_degree = sum(dict(graph.out_degree()).values()) / node_count
    print(f"nodes: {node_count}")
    print(f"edges: {edge_count}")
    print(f"average in-degree: {avg_in_degree:.3f}")
    print(f"average out-degree: {avg_out_degree:.3f}")
    print(f"source tasks: {dag.source_tasks}")
    print(f"sink tasks: {dag.sink_tasks}")


def _try_draw_dag(dag: DAGTask, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        graph = dag.graph
        levels = nx.get_node_attributes(graph, "level")
        pos = nx.multipartite_layout(graph, subset_key="level") if levels else nx.spring_layout(graph)
        labels = {node: str(node) for node in graph.nodes}
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(
            graph,
            pos=pos,
            labels=labels,
            node_color="#87ceeb",
            node_size=900,
            arrows=True,
            arrowsize=18,
            font_size=9,
        )
        edge_labels = {
            (src, dst): f"{attrs.get('data_size', 0.0):.1f}"
            for src, dst, attrs in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        print(f"saved DAG figure to: {output_path}")
    except Exception as exc:  # pragma: no cover - demo-only resilience
        print(f"warning: failed to draw DAG figure: {exc}")


if __name__ == "__main__":
    demo_dag = generate_random_dag(num_tasks=10, edge_density=0.35, seed=42)
    _print_graph_stats(demo_dag)
    _try_draw_dag(demo_dag, Path(__file__).resolve().with_name("dag_demo.png"))

