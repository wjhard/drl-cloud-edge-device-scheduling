from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
from baselines.milp_optimal_scheduler import MILPOptimalScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_rl_records(path: str | Path) -> dict[str, dict]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(summary.get("num_samples", 0)) != 16:
        raise ValueError(f"expected a best-of-16 RL summary, got num_samples={summary.get('num_samples')}")
    return {record["scenario"]: record for record in summary["scenarios"]}


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HEFT and residual best-of-16 with MILP.")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument(
        "--rl-summary", default="evaluation/results/summary_mlp_residual_bestof16.json"
    )
    parser.add_argument(
        "--results-path", default="evaluation/results/milp_optimal_comparison.json"
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CBC worker count. Omit on Windows because bundled CBC 2.10.3 can hang with -threads.",
    )
    parser.add_argument("--task-sizes", type=int, nargs="+", default=[10, 15])
    args = parser.parse_args()

    config = _load_yaml(args.config)
    env_config = config["env"]
    scenarios_dir = Path(config["evaluation"]["scenarios_dir"])
    scenario_paths = sorted(
        path
        for path in scenarios_dir.glob("*.json")
        if load_dag_from_json(path).graph.number_of_nodes() in set(args.task_sizes)
    )
    rl_records = _load_rl_records(args.rl_summary)
    results_path = Path(args.results_path)
    solver_logs_dir = results_path.parent / "milp_solver_logs"
    solver_logs_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config_path": str(args.config),
        "rl_summary_path": str(args.rl_summary),
        "time_limit_seconds_per_scenario": args.time_limit,
        "threads": args.threads,
        "task_sizes": args.task_sizes,
        "scenarios": [],
    }

    for index, scenario_path in enumerate(scenario_paths, start=1):
        print(f"[{index}/{len(scenario_paths)}] solving {scenario_path.name}", flush=True)
        dag = load_dag_from_json(scenario_path)
        resource_path = env_config["resource_config_path"]

        heft_scheduler = HEFTScheduler()
        heft_schedule = heft_scheduler.schedule(dag, load_resource_config(resource_path))
        heft_makespan = heft_scheduler.compute_makespan(heft_schedule)

        solver_log_path = solver_logs_dir / f"{scenario_path.stem}.log"
        milp_result = MILPOptimalScheduler(
            time_limit_seconds=args.time_limit,
            threads=args.threads,
            solver_log_path=solver_log_path,
        ).solve(dag, load_resource_config(resource_path))

        rl_record = rl_records.get(scenario_path.name)
        if rl_record is None:
            raise KeyError(f"missing residual best-of-16 record for {scenario_path.name}")
        rl_makespan = float(rl_record["rl_makespan"])
        optimal_value = milp_result.makespan if milp_result.proven_optimal else None
        optimum_upper_bound = min(
            value
            for value in (milp_result.makespan, heft_makespan, rl_makespan)
            if value is not None
        )
        optimum_lower_bound = milp_result.best_bound

        def ratio_interval(candidate_makespan: float) -> dict[str, float] | None:
            if optimum_lower_bound is None:
                return None
            return {
                "lower": candidate_makespan / optimum_upper_bound,
                "upper": candidate_makespan / optimum_lower_bound,
            }

        record = {
            "scenario": scenario_path.name,
            "num_tasks": dag.graph.number_of_nodes(),
            "milp_status": milp_result.status,
            "milp_proven_optimal": milp_result.proven_optimal,
            "milp_makespan": milp_result.makespan,
            "milp_best_bound": milp_result.best_bound,
            "milp_relative_gap": milp_result.relative_gap,
            "milp_solve_time_seconds": milp_result.solve_time_seconds,
            "heft_makespan": heft_makespan,
            "residual_bestof16_makespan": rl_makespan,
            "heft_over_milp_optimal": (
                heft_makespan / optimal_value if optimal_value is not None else None
            ),
            "residual_bestof16_over_milp_optimal": (
                rl_makespan / optimal_value if optimal_value is not None else None
            ),
            "milp_optimum_interval": {
                "lower": optimum_lower_bound,
                "upper": optimum_upper_bound,
            },
            "heft_over_optimum_interval": ratio_interval(heft_makespan),
            "residual_bestof16_over_optimum_interval": ratio_interval(rl_makespan),
            "solver_log": str(solver_log_path),
            "milp_schedule": (
                {
                    str(task_id): {
                        "resource_id": resource_id,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }
                    for task_id, (resource_id, start_time, finish_time) in sorted(
                        (milp_result.schedule or {}).items()
                    )
                }
                if milp_result.schedule is not None
                else None
            ),
        }
        summary["scenarios"].append(record)
        _write_summary(results_path, summary)

        print(
            f"  status={milp_result.status}, proven_optimal={milp_result.proven_optimal}, "
            f"MILP={milp_result.makespan}, bound={milp_result.best_bound}, "
            f"gap={milp_result.relative_gap}, seconds={milp_result.solve_time_seconds:.3f}",
            flush=True,
        )
        if optimal_value is not None:
            print(
                f"  HEFT={heft_makespan:.9f}, HEFT/OPT={heft_makespan / optimal_value:.9f}, "
                f"Residual16={rl_makespan:.9f}, RL/OPT={rl_makespan / optimal_value:.9f}",
                flush=True,
            )
        else:
            print(
                f"  HEFT={heft_makespan:.9f}, Residual16={rl_makespan:.9f}; "
                "ratios omitted because optimality was not proven",
                flush=True,
            )

    proven = sum(record["milp_proven_optimal"] for record in summary["scenarios"])
    feasible_not_proven = sum(
        record["milp_status"] == "feasible_not_proven_optimal"
        for record in summary["scenarios"]
    )
    failed = len(summary["scenarios"]) - proven - feasible_not_proven
    summary["totals"] = {
        "scenario_count": len(summary["scenarios"]),
        "proven_optimal_count": proven,
        "feasible_not_proven_optimal_count": feasible_not_proven,
        "no_feasible_solution_count": failed,
    }
    _write_summary(results_path, summary)
    print(
        f"completed: proven_optimal={proven}, feasible_not_proven={feasible_not_proven}, "
        f"no_solution={failed}",
        flush=True,
    )
    print(f"results_path: {results_path}", flush=True)


if __name__ == "__main__":
    main()
