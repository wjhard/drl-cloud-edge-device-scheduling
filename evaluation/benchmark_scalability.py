from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
import yaml
from scipy import stats as scipy_stats

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from evaluation.runtime_metrics import percentile, process_memory_mib
from evaluation.schedule_validation import validate_schedule
from policies.residual_local_search_scheduler import ResidualLargeNeighborhoodScheduler


DEFAULT_SCENARIOS_DIR = "evaluation/scenarios_large"
DEFAULT_RESULTS_PATH = "evaluation/results/final_adaptive_large_scale.json"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate(records: list[dict]) -> dict:
    ratios = [float(record["ratio_to_heft"]) for record in records]
    wall_times = [float(record["timing"]["schedule_seconds"]) for record in records]
    cpu_times = [float(record["timing"]["process_cpu_seconds"]) for record in records]
    slowdowns = [float(record["timing"]["slowdown_vs_heft"]) for record in records]
    peak_values = [
        float(record["memory"]["process_peak_rss_mib"])
        for record in records
        if record["memory"]["process_peak_rss_mib"] is not None
    ]
    mean_ratio = statistics.fmean(ratios)
    sample_std_ratio = _sample_std(ratios)
    if len(ratios) > 1 and sample_std_ratio > 0:
        standard_error = sample_std_ratio / math.sqrt(len(ratios))
        critical_value = float(scipy_stats.t.ppf(0.975, df=len(ratios) - 1))
        confidence_interval = [
            mean_ratio - critical_value * standard_error,
            mean_ratio + critical_value * standard_error,
        ]
        test = scipy_stats.ttest_1samp(ratios, popmean=1.0)
        t_statistic = float(test.statistic)
        p_value_two_sided = float(test.pvalue)
    else:
        confidence_interval = [mean_ratio, mean_ratio]
        t_statistic = None
        p_value_two_sided = None
    return {
        "scenario_count": len(records),
        "mean_ratio": mean_ratio,
        "sample_std_ratio": sample_std_ratio,
        "mean_improvement_percent_vs_heft": (1.0 - mean_ratio) * 100.0,
        "mean_ratio_confidence_interval_95": confidence_interval,
        "paired_t_test_ratio_vs_one": {
            "difference_definition": "adaptive ratio minus HEFT ratio (1.0)",
            "t_statistic": t_statistic,
            "p_value_two_sided": p_value_two_sided,
        },
        "strictly_better_than_heft_count": sum(value < 1.0 - 1e-12 for value in ratios),
        "non_regression_vs_heft_count": sum(value <= 1.0 + 1e-12 for value in ratios),
        "mean_schedule_seconds": statistics.fmean(wall_times),
        "median_schedule_seconds": statistics.median(wall_times),
        "p95_schedule_seconds": percentile(wall_times, 0.95),
        "max_schedule_seconds": max(wall_times),
        "mean_process_cpu_seconds": statistics.fmean(cpu_times),
        "mean_slowdown_vs_heft": statistics.fmean(slowdowns),
        "max_process_peak_rss_mib": max(peak_values) if peak_values else None,
        "time_limit_reached_count": sum(
            bool(record["diagnostics"]["time_limit_reached"]) for record in records
        ),
        "residual_skipped_count": sum(
            record["diagnostics"]["residual_skipped_reason"] is not None
            for record in records
        ),
        "selected_source_counts": dict(
            sorted(Counter(record["selected_source"] for record in records).items())
        ),
    }


def benchmark(
    *,
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    resource_config_path: str | Path = "configs/resource_default.yaml",
    config_path: str | Path = "training/configs/ppo_mlp_residual.yaml",
    model_path: str | Path = "training/checkpoints/ppo_mlp_residual",
    seed: int = 20260905,
    num_samples: int = 64,
    local_max_passes: int = 3,
    lns_iterations: int = 64,
    time_budget_seconds: float | None = 10.0,
) -> dict:
    scenario_paths = sorted(Path(scenarios_dir).glob("scenario_*.json"))
    if not scenario_paths:
        raise RuntimeError(f"no scenarios found in {scenarios_dir}")

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config = config["env"]
    model_capacity = int(env_config["max_tasks_padding"])
    memory_before_load = process_memory_mib()
    load_started = time.perf_counter()
    scheduler = ResidualLargeNeighborhoodScheduler(
        model_path=model_path,
        max_tasks=model_capacity,
        num_samples=num_samples,
        local_max_passes=local_max_passes,
        lns_iterations=lns_iterations,
        normalize_observations=bool(env_config.get("normalize_observations", False)),
        random_seed=seed,
        portfolio_mode="adaptive",
        time_limit_seconds=time_budget_seconds,
    )
    model_load_seconds = time.perf_counter() - load_started
    memory_after_load = process_memory_mib()

    heft = HEFTScheduler()
    records: list[dict] = []
    benchmark_started = time.perf_counter()
    for scenario_index, scenario_path in enumerate(scenario_paths):
        dag = load_dag_from_json(scenario_path)
        resources = load_resource_config(resource_config_path)
        scenario_seed = seed + scenario_index
        _set_seed(scenario_seed)
        scheduler.random_seed = scenario_seed

        heft_started = time.perf_counter()
        heft_schedule = heft.schedule(dag, load_resource_config(resource_config_path))
        heft_seconds = time.perf_counter() - heft_started
        heft_makespan = heft.compute_makespan(heft_schedule)

        gc.collect()
        memory_before = process_memory_mib()
        cpu_started = time.process_time()
        schedule_started = time.perf_counter()
        schedule = scheduler.schedule(dag, resources)
        schedule_seconds = time.perf_counter() - schedule_started
        process_cpu_seconds = time.process_time() - cpu_started
        memory_after = process_memory_mib()

        validation = validate_schedule(dag, resources, schedule)
        if not validation["valid"]:
            raise RuntimeError(
                f"invalid schedule for {scenario_path.name}: "
                + "; ".join(validation["errors"][:5])
            )
        makespan = scheduler.compute_makespan(schedule)
        ratio = makespan / heft_makespan
        if ratio > 1.0 + 1e-9:
            raise RuntimeError(
                f"HEFT non-regression guarantee failed for {scenario_path.name}: {ratio}"
            )

        record = {
            "scenario": scenario_path.name,
            "num_tasks": dag.graph.number_of_nodes(),
            "edge_count": dag.graph.number_of_edges(),
            "scenario_seed": scenario_seed,
            "heft_makespan": heft_makespan,
            "adaptive_makespan": makespan,
            "ratio_to_heft": ratio,
            "selected_source": scheduler.last_selected_source,
            "timing": {
                "schedule_seconds": schedule_seconds,
                "process_cpu_seconds": process_cpu_seconds,
                "heft_reference_seconds": heft_seconds,
                "slowdown_vs_heft": schedule_seconds / max(heft_seconds, 1e-12),
                "candidate_elapsed_seconds": dict(
                    scheduler.last_candidate_elapsed_seconds
                ),
            },
            "memory": {
                "rss_before_mib": memory_before["rss_mib"],
                "rss_after_mib": memory_after["rss_mib"],
                "rss_delta_mib": (
                    memory_after["rss_mib"] - memory_before["rss_mib"]
                    if memory_before["rss_mib"] is not None
                    and memory_after["rss_mib"] is not None
                    else None
                ),
                "process_peak_rss_mib": memory_after["peak_rss_mib"],
            },
            "diagnostics": {
                "candidate_makespans": dict(scheduler.last_candidate_makespans),
                "time_limit_reached": scheduler.last_timed_out,
                "residual_skipped_reason": scheduler.last_residual_skipped_reason,
                "residual_samples_completed": scheduler.local_scheduler.last_samples_completed,
                "dag_shape": scheduler.last_shape_stats.as_dict()
                if scheduler.last_shape_stats is not None
                else None,
            },
            "validation": validation,
        }
        records.append(record)
        print(
            f"{scenario_path.name}: tasks={record['num_tasks']} "
            f"ratio={ratio:.6f} source={record['selected_source']} "
            f"seconds={schedule_seconds:.3f} timed_out={scheduler.last_timed_out}",
            flush=True,
        )

    groups: dict[str, dict] = {}
    for task_count in sorted({int(record["num_tasks"]) for record in records}):
        group_records = [
            record for record in records if int(record["num_tasks"]) == task_count
        ]
        groups[str(task_count)] = _aggregate(group_records)

    summary = {
        "schema_version": 1,
        "method": "Adaptive Residual portfolio + topological relocation + best-only LNS",
        "protocol": {
            "scenarios_dir": Path(scenarios_dir).as_posix(),
            "resource_config_path": Path(resource_config_path).as_posix(),
            "config_path": Path(config_path).as_posix(),
            "model_path": Path(model_path).as_posix(),
            "model_capacity_tasks": model_capacity,
            "seed": seed,
            "num_samples": num_samples,
            "local_max_passes": local_max_passes,
            "lns_iterations": lns_iterations,
            "time_budget_seconds_per_scenario": time_budget_seconds,
            "time_budget_semantics": (
                "soft wall-clock limit checked between rollouts and neighborhood evaluations; "
                "HEFT anchor construction is always allowed to finish"
            ),
            "logical_cpu_count": os.cpu_count(),
        },
        "model_load": {
            "seconds": model_load_seconds,
            "rss_before_mib": memory_before_load["rss_mib"],
            "rss_after_mib": memory_after_load["rss_mib"],
            "peak_rss_mib": memory_after_load["peak_rss_mib"],
            "checkpoint_size_mib": (
                Path(str(model_path) + ".zip").stat().st_size / (1024.0 * 1024.0)
                if Path(str(model_path) + ".zip").exists()
                else None
            ),
        },
        "overall": _aggregate(records),
        "by_task_size": groups,
        "total_benchmark_seconds": time.perf_counter() - benchmark_started,
        "scenarios": records,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SCALABILITY_BENCHMARK_COMPLETE")
    print(f"mean_ratio={summary['overall']['mean_ratio']:.12f}")
    print(
        "non_regression_vs_heft="
        f"{summary['overall']['non_regression_vs_heft_count']}/{len(records)}"
    )
    print(f"mean_schedule_seconds={summary['overall']['mean_schedule_seconds']:.6f}")
    print(f"p95_schedule_seconds={summary['overall']['p95_schedule_seconds']:.6f}")
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark final adaptive scheduling on fixed 30-60 task DAGs."
    )
    parser.add_argument("--scenarios-dir", default=DEFAULT_SCENARIOS_DIR)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--resources", default="configs/resource_default.yaml")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--local-max-passes", type=int, default=3)
    parser.add_argument("--lns-iterations", type=int, default=64)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=10.0,
        help="Soft wall-clock budget per scenario; use 0 for no limit.",
    )
    args = parser.parse_args()
    benchmark(
        scenarios_dir=args.scenarios_dir,
        results_path=args.results_path,
        resource_config_path=args.resources,
        config_path=args.config,
        model_path=args.model_path,
        seed=args.seed,
        num_samples=args.num_samples,
        local_max_passes=args.local_max_passes,
        lns_iterations=args.lns_iterations,
        time_budget_seconds=(
            None if args.time_budget_seconds == 0 else args.time_budget_seconds
        ),
    )


if __name__ == "__main__":
    main()
