from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.dag_generator import DAGTask, generate_random_dag, save_dag_to_json


DEFAULT_OUTPUT_ROOT = "evaluation/scenarios_structural"
DEFAULT_SEED_START = 5_000_000
TASK_SIZES = (8, 12, 16, 20, 25)
DEFAULT_DENSITY = 0.35
WIDE_DENSITY_RANGE = (0.10, 0.15)
DEEP_DENSITY_RANGE = (0.60, 0.70)
GROUPS = (
    "wide_parallel",
    "deep_chain",
    "homogeneous_resources",
    "original_control",
)


def _generate_deep_layered_dag(num_tasks: int, edge_density: float, seed: int) -> DAGTask:
    """Generate a narrow, many-level DAG while retaining limited parallel choices."""
    rng = random.Random(seed)
    graph = nx.DiGraph()
    level_count = max(3, math.ceil(num_tasks / 3))
    levels: list[list[int]] = [[] for _ in range(level_count)]

    for task_id in range(num_tasks):
        level = min(task_id // 3, level_count - 1)
        levels[level].append(task_id)
        graph.add_node(
            task_id,
            task_id=task_id,
            level=level,
            computation_cost=rng.uniform(1.0, 10.0),
        )

    nonempty_levels = [nodes for nodes in levels if nodes]
    for src_level, src_nodes in enumerate(nonempty_levels[:-1]):
        for dst_nodes in nonempty_levels[src_level + 1 :]:
            for src in src_nodes:
                for dst in dst_nodes:
                    if rng.random() < edge_density:
                        graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))

    # Guarantee a path through every level without collapsing each level to one task.
    for current_nodes, next_nodes in zip(nonempty_levels, nonempty_levels[1:]):
        anchor_src = current_nodes[0]
        anchor_dst = next_nodes[0]
        if not graph.has_edge(anchor_src, anchor_dst):
            graph.add_edge(anchor_src, anchor_dst, data_size=rng.uniform(1.0, 10.0))
        for dst in next_nodes:
            if graph.in_degree(dst) == 0:
                src = rng.choice(current_nodes)
                graph.add_edge(src, dst, data_size=rng.uniform(1.0, 10.0))

    source_tasks = sorted(node for node in graph if graph.in_degree(node) == 0)
    sink_tasks = sorted(node for node in graph if graph.out_degree(node) == 0)
    return DAGTask(graph=graph, source_tasks=source_tasks, sink_tasks=sink_tasks)


def _scenario_record(
    group: str,
    index: int,
    seed: int,
    requested_density: float,
    dag: DAGTask,
    path: Path,
) -> dict:
    graph = dag.graph
    return {
        "group": group,
        "index": index,
        "seed": seed,
        "requested_edge_density": requested_density,
        "num_tasks": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "actual_graph_density": nx.density(graph),
        "source_task_count": len(dag.source_tasks),
        "sink_task_count": len(dag.sink_tasks),
        "longest_path_edges": nx.dag_longest_path_length(graph),
        "path": path.as_posix(),
    }


def generate_structural_scenarios(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed_start: int = DEFAULT_SEED_START,
) -> dict:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        group_dir = root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for existing_path in group_dir.glob("scenario_*.json"):
            existing_path.unlink()

    records: list[dict] = []
    for index, task_size in enumerate(TASK_SIZES):
        wide_seed = seed_start + index
        wide_density = random.Random(wide_seed).uniform(*WIDE_DENSITY_RANGE)
        wide_dag = generate_random_dag(task_size, edge_density=wide_density, seed=wide_seed)
        wide_path = root / "wide_parallel" / f"scenario_{task_size}_{index}.json"
        save_dag_to_json(wide_dag, wide_path)
        records.append(
            _scenario_record(
                "wide_parallel", index, wide_seed, wide_density, wide_dag, wide_path
            )
        )

        deep_seed = seed_start + 100 + index
        deep_density = random.Random(deep_seed).uniform(*DEEP_DENSITY_RANGE)
        deep_dag = _generate_deep_layered_dag(task_size, deep_density, deep_seed)
        deep_path = root / "deep_chain" / f"scenario_{task_size}_{index}.json"
        save_dag_to_json(deep_dag, deep_path)
        records.append(
            _scenario_record("deep_chain", index, deep_seed, deep_density, deep_dag, deep_path)
        )

        paired_seed = seed_start + 200 + index
        paired_dag = generate_random_dag(
            task_size,
            edge_density=DEFAULT_DENSITY,
            seed=paired_seed,
        )
        for group in ("homogeneous_resources", "original_control"):
            paired_path = root / group / f"scenario_{task_size}_{index}.json"
            save_dag_to_json(paired_dag, paired_path)
            records.append(
                _scenario_record(
                    group,
                    index,
                    paired_seed,
                    DEFAULT_DENSITY,
                    paired_dag,
                    paired_path,
                )
            )

    manifest = {
        "seed_start": seed_start,
        "task_sizes": list(TASK_SIZES),
        "scenario_file_count": len(records),
        "paired_resource_groups": ["homogeneous_resources", "original_control"],
        "resource_configs": {
            "wide_parallel": "configs/resource_default.yaml",
            "deep_chain": "configs/resource_default.yaml",
            "homogeneous_resources": "configs/resource_structural_homogeneous.yaml",
            "original_control": "configs/resource_default.yaml",
        },
        "scenarios": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _print_summary(manifest, manifest_path)
    return manifest


def _print_summary(manifest: dict, manifest_path: Path) -> None:
    print(f"generated {manifest['scenario_file_count']} structural scenarios")
    print(f"manifest_path={manifest_path}")
    print(f"seed_start={manifest['seed_start']}")
    print("group | count | mean_density | mean_sources | mean_longest_path")
    records = manifest["scenarios"]
    for group in GROUPS:
        group_records = [record for record in records if record["group"] == group]
        count = len(group_records)
        mean_density = sum(record["actual_graph_density"] for record in group_records) / count
        mean_sources = sum(record["source_task_count"] for record in group_records) / count
        mean_longest = sum(record["longest_path_edges"] for record in group_records) / count
        print(
            f"{group} | {count} | {mean_density:.6f} | "
            f"{mean_sources:.3f} | {mean_longest:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structural generalization scenarios.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    args = parser.parse_args()
    generate_structural_scenarios(args.output_root, args.seed_start)


if __name__ == "__main__":
    main()
