from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pulp

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import DAGTask, load_dag_from_json
from env.resource_config import ResourceConfig, default_resource_config_path, load_resource_config
from scheduler_interface import BaseScheduler, ScheduleResult


@dataclass(frozen=True)
class MILPSolveResult:
    schedule: ScheduleResult | None
    status: str
    proven_optimal: bool
    makespan: float | None
    best_bound: float | None
    relative_gap: float | None
    solve_time_seconds: float
    solver_status: str
    solver_solution_status: str


class MILPOptimalScheduler(BaseScheduler):
    """Exact unrelated-machine DAG scheduler with assignment-dependent communication."""

    def __init__(
        self,
        time_limit_seconds: float = 300.0,
        threads: int | None = None,
        solver_log_path: str | Path | None = None,
    ) -> None:
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        self.time_limit_seconds = float(time_limit_seconds)
        self.threads = threads
        self.solver_log_path = Path(solver_log_path) if solver_log_path is not None else None
        self.last_result: MILPSolveResult | None = None

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        result = self.solve(dag, resource_config)
        if result.schedule is None:
            raise RuntimeError(f"MILP did not find a feasible schedule: {result.status}")
        return result.schedule

    def solve(self, dag: DAGTask, resource_config: ResourceConfig) -> MILPSolveResult:
        if not nx.is_directed_acyclic_graph(dag.graph):
            raise ValueError("MILP scheduler requires a DAG")

        tasks = list(nx.topological_sort(dag.graph))
        resources = list(resource_config.resources)
        if not tasks:
            result = MILPSolveResult({}, "optimal", True, 0.0, 0.0, 0.0, 0.0, "Optimal", "Optimal Solution Found")
            self.last_result = result
            return result

        task_index = {task_id: index for index, task_id in enumerate(tasks)}
        resource_index = {resource.id: index for index, resource in enumerate(resources)}
        durations = {
            (task_id, resource.id): resource_config.get_execution_time(dag.graph.nodes[task_id], resource)
            for task_id in tasks
            for resource in resources
        }

        heft_scheduler = HEFTScheduler()
        heft_schedule = heft_scheduler.schedule(dag, resource_config)
        heuristic_upper_bound = heft_scheduler.compute_makespan(heft_schedule)
        # HEFT provides both a valid MIP start and a safe finite horizon. Its task
        # assignments and order remain free decision variables in the MILP.
        makespan_upper_bound = heuristic_upper_bound

        communication = {}
        max_communication = 0.0
        for predecessor, successor, edge_data in dag.graph.edges(data=True):
            data_size = float(edge_data.get("data_size", 0.0))
            for source_resource in resources:
                for target_resource in resources:
                    delay = resource_config.get_communication_time(
                        data_size, source_resource, target_resource
                    )
                    communication[(predecessor, successor, source_resource.id, target_resource.id)] = delay
                    max_communication = max(max_communication, delay)

        big_m = makespan_upper_bound + max_communication + 1e-6
        problem = pulp.LpProblem("cloud_edge_device_dag_scheduling", pulp.LpMinimize)
        starts = {
            task_id: pulp.LpVariable(
                f"start_{task_index[task_id]}", lowBound=0.0, upBound=makespan_upper_bound
            )
            for task_id in tasks
        }
        assignments = {
            (task_id, resource.id): pulp.LpVariable(
                f"assign_{task_index[task_id]}_{resource_index[resource.id]}", cat=pulp.LpBinary
            )
            for task_id in tasks
            for resource in resources
        }
        completion = {
            task_id: starts[task_id]
            + pulp.lpSum(
                durations[task_id, resource.id] * assignments[task_id, resource.id]
                for resource in resources
            )
            for task_id in tasks
        }
        makespan = pulp.LpVariable(
            "makespan", lowBound=0.0, upBound=makespan_upper_bound
        )
        problem += makespan

        for task_id in tasks:
            problem += (
                pulp.lpSum(assignments[task_id, resource.id] for resource in resources) == 1,
                f"one_resource_{task_index[task_id]}",
            )
            problem += makespan >= completion[task_id], f"makespan_{task_index[task_id]}"

        for predecessor, successor in dag.graph.edges:
            pred_index = task_index[predecessor]
            succ_index = task_index[successor]
            for source_resource in resources:
                for target_resource in resources:
                    delay = communication[
                        predecessor, successor, source_resource.id, target_resource.id
                    ]
                    problem += (
                        starts[successor]
                        >= completion[predecessor]
                        + delay
                        - big_m
                        * (
                            2
                            - assignments[predecessor, source_resource.id]
                            - assignments[successor, target_resource.id]
                        ),
                        f"precedence_{pred_index}_{succ_index}_{resource_index[source_resource.id]}_"
                        f"{resource_index[target_resource.id]}",
                    )

        transitive_closure = nx.transitive_closure_dag(dag.graph)
        order_variables: dict[tuple[int, int, str], pulp.LpVariable] = {}
        for first_position, first_task in enumerate(tasks):
            for second_task in tasks[first_position + 1 :]:
                if transitive_closure.has_edge(first_task, second_task) or transitive_closure.has_edge(
                    second_task, first_task
                ):
                    continue
                for resource in resources:
                    order = pulp.LpVariable(
                        f"order_{task_index[first_task]}_{task_index[second_task]}_"
                        f"{resource_index[resource.id]}",
                        cat=pulp.LpBinary,
                    )
                    order_variables[first_task, second_task, resource.id] = order
                    both_on_resource_relaxation = big_m * (
                        2
                        - assignments[first_task, resource.id]
                        - assignments[second_task, resource.id]
                    )
                    problem += (
                        starts[second_task]
                        >= completion[first_task]
                        - big_m * (1 - order)
                        - both_on_resource_relaxation,
                        f"no_overlap_forward_{task_index[first_task]}_{task_index[second_task]}_"
                        f"{resource_index[resource.id]}",
                    )
                    problem += (
                        starts[first_task]
                        >= completion[second_task]
                        - big_m * order
                        - both_on_resource_relaxation,
                        f"no_overlap_reverse_{task_index[first_task]}_{task_index[second_task]}_"
                        f"{resource_index[resource.id]}",
                    )

        self._set_heft_warm_start(
            tasks,
            resources,
            heft_schedule,
            starts,
            assignments,
            order_variables,
            makespan,
            heuristic_upper_bound,
        )

        temporary_log = self.solver_log_path is None
        if temporary_log:
            log_file = tempfile.NamedTemporaryFile(prefix="milp_cbc_", suffix=".log", delete=False)
            log_path = Path(log_file.name)
            log_file.close()
        else:
            log_path = self.solver_log_path
            assert log_path is not None
            log_path.parent.mkdir(parents=True, exist_ok=True)

        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=self.time_limit_seconds,
            threads=self.threads,
            presolve=True,
            cuts=True,
            warmStart=True,
            keepFiles=os.name == "nt",
            logPath=str(log_path),
        )
        started = time.perf_counter()
        try:
            problem.solve(solver)
        finally:
            elapsed = time.perf_counter() - started
            if os.name == "nt":
                for suffix in ("mps", "mst", "sol"):
                    Path(f"{problem.name}-pulp.{suffix}").unlink(missing_ok=True)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if temporary_log:
            log_path.unlink(missing_ok=True)

        solver_status = pulp.LpStatus.get(problem.status, str(problem.status))
        solution_status = pulp.LpSolution.get(problem.sol_status, str(problem.sol_status))
        proven_optimal = problem.sol_status == pulp.LpSolutionOptimal
        has_integer_solution = problem.sol_status in {
            pulp.LpSolutionOptimal,
            pulp.LpSolutionIntegerFeasible,
        }

        schedule: ScheduleResult | None = None
        objective_value: float | None = None
        if has_integer_solution:
            schedule = self._extract_schedule(tasks, resources, starts, assignments, durations)
            self._validate_schedule(dag, resource_config, schedule)
            objective_value = max(finish for _, _, finish in schedule.values())

        best_bound = objective_value if proven_optimal else self._parse_best_bound(log_text)
        relative_gap = None
        if objective_value is not None and best_bound is not None:
            relative_gap = max(0.0, objective_value - best_bound) / max(abs(objective_value), 1e-12)

        if proven_optimal:
            status = "optimal"
        elif has_integer_solution:
            status = "feasible_not_proven_optimal"
        elif problem.status == pulp.LpStatusInfeasible:
            status = "infeasible"
        else:
            status = "no_feasible_solution_within_limit"

        result = MILPSolveResult(
            schedule=schedule,
            status=status,
            proven_optimal=proven_optimal,
            makespan=objective_value,
            best_bound=best_bound,
            relative_gap=relative_gap,
            solve_time_seconds=elapsed,
            solver_status=solver_status,
            solver_solution_status=solution_status,
        )
        self.last_result = result
        return result

    @staticmethod
    def _set_heft_warm_start(
        tasks,
        resources,
        heft_schedule,
        starts,
        assignments,
        order_variables,
        makespan,
        heuristic_upper_bound,
    ) -> None:
        for task_id in tasks:
            assigned_resource, start_time, _ = heft_schedule[task_id]
            starts[task_id].setInitialValue(start_time)
            for resource in resources:
                assignments[task_id, resource.id].setInitialValue(
                    1 if resource.id == assigned_resource else 0
                )
        for (first_task, second_task, _), variable in order_variables.items():
            first_start = heft_schedule[first_task][1]
            second_start = heft_schedule[second_task][1]
            variable.setInitialValue(1 if first_start <= second_start else 0)
        makespan.setInitialValue(heuristic_upper_bound)

    @staticmethod
    def _extract_schedule(tasks, resources, starts, assignments, durations) -> ScheduleResult:
        schedule: ScheduleResult = {}
        for task_id in tasks:
            selected = [
                resource
                for resource in resources
                if (pulp.value(assignments[task_id, resource.id]) or 0.0) > 0.5
            ]
            if len(selected) != 1:
                raise RuntimeError(f"MILP returned an invalid assignment for task {task_id}")
            resource = selected[0]
            start_time = float(pulp.value(starts[task_id]))
            finish_time = start_time + durations[task_id, resource.id]
            schedule[task_id] = (resource.id, start_time, finish_time)
        return schedule

    @staticmethod
    def _validate_schedule(
        dag: DAGTask,
        resource_config: ResourceConfig,
        schedule: ScheduleResult,
        tolerance: float = 1e-5,
    ) -> None:
        for predecessor, successor, edge_data in dag.graph.edges(data=True):
            pred_resource, _, pred_finish = schedule[predecessor]
            succ_resource, succ_start, _ = schedule[successor]
            communication = resource_config.get_communication_time(
                float(edge_data.get("data_size", 0.0)), pred_resource, succ_resource
            )
            if succ_start + tolerance < pred_finish + communication:
                raise RuntimeError(
                    f"MILP schedule violates precedence edge {predecessor}->{successor}"
                )

        by_resource: dict[str, list[tuple[int, float, float]]] = {
            resource.id: [] for resource in resource_config.resources
        }
        for task_id, (resource_id, start, finish) in schedule.items():
            by_resource[resource_id].append((task_id, start, finish))
        for resource_id, events in by_resource.items():
            events.sort(key=lambda event: event[1])
            for previous, current in zip(events, events[1:]):
                if current[1] + tolerance < previous[2]:
                    raise RuntimeError(
                        f"MILP schedule overlaps tasks {previous[0]} and {current[0]} on {resource_id}"
                    )

    @staticmethod
    def _parse_best_bound(log_text: str) -> float | None:
        summary_matches = re.findall(r"Lower bound:\s*([-+0-9.eE]+)", log_text)
        progress_matches = re.findall(r"best possible\s+([-+0-9.eE]+)", log_text)
        # CBC rounds the final "Lower bound" summary to three decimals. Prefer
        # the full-precision branch-and-bound progress value when available.
        source = progress_matches if progress_matches else summary_matches
        candidates = [float(value) for value in source]
        finite_candidates = [value for value in candidates if math.isfinite(value)]
        if not finite_candidates:
            return None
        return max(finite_candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve one DAG scheduling scenario with exact MILP.")
    parser.add_argument("scenario")
    parser.add_argument("--resource-config", default=str(default_resource_config_path()))
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--solver-log", default=None)
    args = parser.parse_args()

    dag = load_dag_from_json(args.scenario)
    resource_config = load_resource_config(args.resource_config)
    result = MILPOptimalScheduler(
        time_limit_seconds=args.time_limit,
        threads=args.threads,
        solver_log_path=args.solver_log,
    ).solve(dag, resource_config)
    print(f"status: {result.status}")
    print(f"proven_optimal: {result.proven_optimal}")
    print(f"makespan: {result.makespan}")
    print(f"best_bound: {result.best_bound}")
    print(f"relative_gap: {result.relative_gap}")
    print(f"solve_time_seconds: {result.solve_time_seconds:.6f}")


if __name__ == "__main__":
    main()
