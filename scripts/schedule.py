from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import networkx as nx
import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from evaluation.runtime_metrics import process_memory_mib
from evaluation.schedule_validation import validate_schedule
from policies.residual_local_search_scheduler import ResidualLargeNeighborhoodScheduler


PRESETS = {
    "fast": {"num_samples": 8, "local_max_passes": 1, "lns_iterations": 8},
    "balanced": {"num_samples": 32, "local_max_passes": 2, "lns_iterations": 32},
    "quality": {"num_samples": 64, "local_max_passes": 3, "lns_iterations": 64},
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_dag_input(dag) -> None:
    if dag.graph.number_of_nodes() == 0:
        raise ValueError("input DAG contains no tasks")
    if not nx.is_directed_acyclic_graph(dag.graph):
        raise ValueError("input graph must be a directed acyclic graph")
    for task_id, data in dag.graph.nodes(data=True):
        if "computation_cost" not in data:
            raise ValueError(f"task {task_id} is missing computation_cost")
        if float(data["computation_cost"]) <= 0:
            raise ValueError(f"task {task_id} has non-positive computation_cost")
    for source, target, data in dag.graph.edges(data=True):
        if float(data.get("data_size", 0.0)) < 0:
            raise ValueError(f"edge {source}->{target} has negative data_size")


def _schedule_rows(schedule) -> list[dict]:
    return [
        {
            "task_id": int(task_id),
            "resource_id": str(resource_id),
            "start_time": float(start_time),
            "finish_time": float(finish_time),
            "duration": float(finish_time) - float(start_time),
        }
        for task_id, (resource_id, start_time, finish_time) in sorted(schedule.items())
    ]


def run_schedule(
    *,
    input_path: str | Path,
    resource_config_path: str | Path = "configs/resource_default.yaml",
    config_path: str | Path = "training/configs/ppo_mlp_residual.yaml",
    model_path: str | Path = "training/checkpoints/ppo_mlp_residual",
    method: str = "adaptive",
    preset: str = "balanced",
    time_budget_seconds: float | None = None,
    seed: int = 20260905,
    num_samples: int | None = None,
    local_max_passes: int | None = None,
    lns_iterations: int | None = None,
) -> dict:
    if method not in {"adaptive", "heft"}:
        raise ValueError("method must be 'adaptive' or 'heft'")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    if time_budget_seconds is not None and time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive")

    chosen = dict(PRESETS[preset])
    if num_samples is not None:
        chosen["num_samples"] = num_samples
    if local_max_passes is not None:
        chosen["local_max_passes"] = local_max_passes
    if lns_iterations is not None:
        chosen["lns_iterations"] = lns_iterations
    if chosen["num_samples"] <= 0:
        raise ValueError("num_samples must be positive")
    if chosen["local_max_passes"] < 0 or chosen["lns_iterations"] < 0:
        raise ValueError("search pass and iteration counts must be non-negative")

    input_path = Path(input_path)
    dag = load_dag_from_json(input_path)
    _validate_dag_input(dag)
    resources = load_resource_config(resource_config_path)
    _set_seed(seed)

    heft = HEFTScheduler()
    heft_started = time.perf_counter()
    heft_schedule = heft.schedule(dag, load_resource_config(resource_config_path))
    heft_seconds = time.perf_counter() - heft_started
    heft_makespan = heft.compute_makespan(heft_schedule)

    memory_before = process_memory_mib()
    load_started = time.perf_counter()
    if method == "heft":
        scheduler = heft
        schedule = heft_schedule
        load_seconds = 0.0
        schedule_seconds = heft_seconds
        selected_source = "heft_anchor"
        diagnostics = {
            "candidate_makespans": {"heft_anchor": heft_makespan},
            "candidate_elapsed_seconds": {"heft_anchor": heft_seconds},
            "residual_skipped_reason": "method=heft",
            "time_limit_reached": False,
        }
    else:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        env_config = config["env"]
        scheduler = ResidualLargeNeighborhoodScheduler(
            model_path=model_path,
            max_tasks=int(env_config["max_tasks_padding"]),
            num_samples=int(chosen["num_samples"]),
            local_max_passes=int(chosen["local_max_passes"]),
            lns_iterations=int(chosen["lns_iterations"]),
            normalize_observations=bool(env_config.get("normalize_observations", False)),
            random_seed=seed,
            portfolio_mode="adaptive",
            time_limit_seconds=time_budget_seconds,
        )
        load_seconds = time.perf_counter() - load_started
        schedule_started = time.perf_counter()
        schedule = scheduler.schedule(dag, resources)
        schedule_seconds = time.perf_counter() - schedule_started
        selected_source = scheduler.last_selected_source
        diagnostics = {
            "candidate_makespans": dict(scheduler.last_candidate_makespans),
            "candidate_elapsed_seconds": dict(scheduler.last_candidate_elapsed_seconds),
            "residual_skipped_reason": scheduler.last_residual_skipped_reason,
            "time_limit_reached": scheduler.last_timed_out,
            "dag_shape": scheduler.last_shape_stats.as_dict()
            if scheduler.last_shape_stats is not None
            else None,
            "residual_samples_completed": scheduler.local_scheduler.last_samples_completed,
        }

    memory_after = process_memory_mib()
    makespan = scheduler.compute_makespan(schedule)
    validation = validate_schedule(dag, resources, schedule)
    if not validation["valid"]:
        raise RuntimeError(
            "scheduler returned an invalid schedule: " + "; ".join(validation["errors"][:5])
        )

    return {
        "schema_version": 1,
        "method": method,
        "selected_source": selected_source,
        "input": {
            "scenario_path": input_path.as_posix(),
            "resource_config_path": Path(resource_config_path).as_posix(),
            "config_path": Path(config_path).as_posix() if method == "adaptive" else None,
            "model_path": Path(model_path).as_posix() if method == "adaptive" else None,
        },
        "parameters": {
            "preset": preset,
            "seed": seed,
            "time_budget_seconds": time_budget_seconds,
            **chosen,
        },
        "dag": {
            "task_count": dag.graph.number_of_nodes(),
            "edge_count": dag.graph.number_of_edges(),
            "source_count": len(dag.source_tasks),
            "sink_count": len(dag.sink_tasks),
        },
        "result": {
            "makespan": makespan,
            "heft_makespan": heft_makespan,
            "ratio_to_heft": makespan / heft_makespan,
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "schedule_seconds": schedule_seconds,
            "heft_reference_seconds": heft_seconds,
            "slowdown_vs_heft": schedule_seconds / max(heft_seconds, 1e-12),
        },
        "memory": {
            "rss_before_mib": memory_before["rss_mib"],
            "rss_after_mib": memory_after["rss_mib"],
            "process_peak_rss_mib": memory_after["peak_rss_mib"],
        },
        "diagnostics": diagnostics,
        "validation": validation,
        "schedule": _schedule_rows(schedule),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schedule one DAG and emit a validated, machine-readable result."
    )
    parser.add_argument("--input", required=True, help="Input DAG in NetworkX node-link JSON format.")
    parser.add_argument(
        "--output",
        default="evaluation/results/schedule_output.json",
        help="Output JSON path, or '-' to write JSON to stdout.",
    )
    parser.add_argument("--resources", default="configs/resource_default.yaml")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--method", choices=("adaptive", "heft"), default="adaptive")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="balanced")
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--local-max-passes", type=int, default=None)
    parser.add_argument("--lns-iterations", type=int, default=None)
    args = parser.parse_args()

    result = run_schedule(
        input_path=args.input,
        resource_config_path=args.resources,
        config_path=args.config,
        model_path=args.model_path,
        method=args.method,
        preset=args.preset,
        time_budget_seconds=args.time_budget_seconds,
        seed=args.seed,
        num_samples=args.num_samples,
        local_max_passes=args.local_max_passes,
        lns_iterations=args.lns_iterations,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(payload)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print("SCHEDULE_COMPLETE")
    print(f"method={result['method']}")
    print(f"selected_source={result['selected_source']}")
    print(f"tasks={result['dag']['task_count']}")
    print(f"makespan={result['result']['makespan']:.12f}")
    print(f"ratio_to_heft={result['result']['ratio_to_heft']:.12f}")
    print(f"schedule_seconds={result['timing']['schedule_seconds']:.6f}")
    print(f"valid={result['validation']['valid']}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
