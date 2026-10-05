from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.heft_scheduler import HEFTScheduler
from baselines.milp_optimal_scheduler import MILPOptimalScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from evaluation.generate_structural_generalization_scenarios import DEFAULT_OUTPUT_ROOT
from policies.residual_local_search_scheduler import ResidualLargeNeighborhoodScheduler


DEFAULT_SCENARIOS_DIR = Path(DEFAULT_OUTPUT_ROOT) / "wide_parallel"


def _clone_config(config_path: str | Path, scenarios_dir: str | Path) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config["evaluation"]["scenarios_dir"] = str(scenarios_dir)
    return config


def evaluate_wide_parallel(
    config_path: str | Path = "training/configs/ppo_mlp_residual.yaml",
    model_path: str | Path = "training/checkpoints/ppo_mlp_residual",
    scenarios_dir: str | Path = DEFAULT_SCENARIOS_DIR,
    resource_config_path: str | Path = "configs/resource_default.yaml",
    results_path: str | Path = "evaluation/results/wide_parallel_adaptive.json",
    num_samples: int = 64,
    local_max_passes: int = 3,
    lns_iterations: int = 64,
    seed: int = 20260830,
    wide_width_ratio_threshold: float = 0.30,
    milp_tie_check_max_tasks: int = 10,
    milp_time_limit_seconds: float = 30.0,
) -> dict:
    scenario_paths = sorted(Path(scenarios_dir).glob("scenario_*.json"))
    if not scenario_paths:
        raise RuntimeError(f"no wide-parallel scenarios found in {scenarios_dir}")
    resource_path = Path(resource_config_path)
    rows: list[dict] = []

    for mode in ("residual_only", "adaptive"):
        scheduler = ResidualLargeNeighborhoodScheduler(
            model_path=model_path,
            max_tasks=int(
                yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["env"][
                    "max_tasks_padding"
                ]
            ),
            num_samples=num_samples,
            local_max_passes=local_max_passes,
            lns_iterations=lns_iterations,
            normalize_observations=True,
            random_seed=seed,
            portfolio_mode=mode,
            wide_width_ratio_threshold=wide_width_ratio_threshold,
        )
        for index, scenario_path in enumerate(scenario_paths):
            dag = load_dag_from_json(scenario_path)
            resources = load_resource_config(resource_path)
            schedule = scheduler.schedule(dag, resources)
            heft_resources = load_resource_config(resource_path)
            heft_schedule = HEFTScheduler().schedule(dag, heft_resources)
            heft_makespan = HEFTScheduler().compute_makespan(heft_schedule)
            makespan = scheduler.compute_makespan(schedule)
            row = {
                "mode": mode,
                "scenario": scenario_path.name,
                "makespan": makespan,
                "heft_makespan": heft_makespan,
                "ratio": makespan / heft_makespan,
                "selected_source": scheduler.last_selected_source,
                "candidate_makespans": scheduler.last_candidate_makespans,
                "dag_shape": scheduler.last_shape_stats.as_dict()
                if scheduler.last_shape_stats is not None
                else None,
                "scenario_seed": seed + index,
            }
            rows.append(row)

    optimality_checks: dict[str, dict] = {}
    if milp_tie_check_max_tasks > 0:
        adaptive_rows = [row for row in rows if row["mode"] == "adaptive"]
        for row in adaptive_rows:
            if float(row["ratio"]) < 1.0 - 1e-12:
                continue
            scenario_path = Path(scenarios_dir) / row["scenario"]
            dag = load_dag_from_json(scenario_path)
            if dag.graph.number_of_nodes() > milp_tie_check_max_tasks:
                continue
            solve_result = MILPOptimalScheduler(
                time_limit_seconds=milp_time_limit_seconds
            ).solve(dag, load_resource_config(resource_path))
            optimality_checks[row["scenario"]] = {
                "status": solve_result.status,
                "proven_optimal": solve_result.proven_optimal,
                "optimal_makespan": solve_result.makespan,
                "best_bound": solve_result.best_bound,
                "relative_gap": solve_result.relative_gap,
                "solve_time_seconds": solve_result.solve_time_seconds,
                "adaptive_matches_optimum": (
                    solve_result.makespan is not None
                    and abs(float(row["makespan"]) - float(solve_result.makespan)) <= 1e-6
                ),
            }

    grouped: dict[str, dict] = {}
    for mode in ("residual_only", "adaptive"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        ratios = [float(row["ratio"]) for row in mode_rows]
        grouped[mode] = {
            "scenario_count": len(mode_rows),
            "mean_ratio": statistics.fmean(ratios),
            "sample_std": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            "outperform_heft_count": sum(value < 1.0 for value in ratios),
            "non_regression_vs_heft_count": sum(value <= 1.0 + 1e-12 for value in ratios),
            "heft_tie_proven_optimal_count": sum(
                optimality_checks.get(row["scenario"], {}).get("adaptive_matches_optimum", False)
                for row in mode_rows
                if abs(float(row["ratio"]) - 1.0) <= 1e-12
            ),
            "selected_source_counts": {
                source: sum(row["selected_source"] == source for row in mode_rows)
                for source in sorted({row["selected_source"] for row in mode_rows})
            },
        }
    before = {row["scenario"]: row for row in rows if row["mode"] == "residual_only"}
    after = {row["scenario"]: row for row in rows if row["mode"] == "adaptive"}
    paired_deltas = [
        float(after[name]["ratio"]) - float(before[name]["ratio"]) for name in before
    ]
    result = {
        "method": "Adaptive Residual portfolio for wide-parallel generalization",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "scenarios_dir": str(scenarios_dir),
        "resource_config_path": str(resource_config_path),
        "num_samples": num_samples,
        "local_max_passes": local_max_passes,
        "lns_iterations": lns_iterations,
        "seed": seed,
        "wide_width_ratio_threshold": wide_width_ratio_threshold,
        "groups": grouped,
        "adaptive_heft_tie_optimality_checks": optimality_checks,
        "paired_adaptive_minus_residual_ratio": {
            "mean": statistics.fmean(paired_deltas),
            "improved_or_equal_count": sum(value <= 1e-12 for value in paired_deltas),
            "improved_count": sum(value < -1e-12 for value in paired_deltas),
        },
        "scenarios": rows,
    }
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WIDE_PARALLEL_ADAPTIVE_EVALUATION")
    for mode, summary in grouped.items():
        print(
            f"{mode}: mean_ratio={summary['mean_ratio']:.12f}, "
            f"outperform_heft={summary['outperform_heft_count']}/{summary['scenario_count']}, "
            f"non_regression_vs_heft={summary['non_regression_vs_heft_count']}/{summary['scenario_count']}, "
            f"heft_tie_optimal={summary['heft_tie_proven_optimal_count']}, "
            f"sources={summary['selected_source_counts']}"
        )
    print(
        "adaptive_minus_residual: "
        f"mean={result['paired_adaptive_minus_residual_ratio']['mean']:+.12f}, "
        f"improved={result['paired_adaptive_minus_residual_ratio']['improved_count']}/"
        f"{len(paired_deltas)}"
    )
    print(f"results_path={output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare residual-only and adaptive portfolio on wide-parallel DAGs."
    )
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument(
        "--scenarios-dir",
        default=str(DEFAULT_SCENARIOS_DIR),
    )
    parser.add_argument("--resource-config", default="configs/resource_default.yaml")
    parser.add_argument(
        "--results-path",
        default="evaluation/results/wide_parallel_adaptive.json",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--local-max-passes", type=int, default=3)
    parser.add_argument("--lns-iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--wide-width-ratio-threshold", type=float, default=0.30)
    parser.add_argument(
        "--milp-tie-check-max-tasks",
        type=int,
        default=10,
        help="Use MILP to prove optimality for non-outperforming wide cases up to this size; 0 disables.",
    )
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    args = parser.parse_args()
    evaluate_wide_parallel(
        config_path=args.config,
        model_path=args.model_path,
        scenarios_dir=args.scenarios_dir,
        resource_config_path=args.resource_config,
        results_path=args.results_path,
        num_samples=args.num_samples,
        local_max_passes=args.local_max_passes,
        lns_iterations=args.lns_iterations,
        seed=args.seed,
        wide_width_ratio_threshold=args.wide_width_ratio_threshold,
        milp_tie_check_max_tasks=args.milp_tie_check_max_tasks,
        milp_time_limit_seconds=args.milp_time_limit_seconds,
    )


if __name__ == "__main__":
    main()
