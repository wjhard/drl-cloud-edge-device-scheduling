from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from baselines.milp_optimal_scheduler import MILPOptimalScheduler, MILPSolveResult
from env.dag_generator import DAGTask
from env.resource_config import ResourceConfig
from scheduler_interface import BaseScheduler, ScheduleResult


@dataclass(frozen=True)
class HybridDecision:
    selected_method: str
    reason: str
    num_tasks: int
    milp_status: str | None = None
    milp_proven_optimal: bool | None = None
    milp_makespan: float | None = None
    milp_best_bound: float | None = None
    milp_solve_time_seconds: float | None = None
    milp_error: str | None = None


class HybridScheduler(BaseScheduler):
    """Use proven MILP optima for small DAGs and an RL scheduler otherwise."""

    def __init__(
        self,
        residual_scheduler: BaseScheduler,
        task_threshold: int = 15,
        milp_time_limit_seconds: float = 10.0,
        milp_scheduler: MILPOptimalScheduler | None = None,
    ) -> None:
        if task_threshold <= 0:
            raise ValueError("task_threshold must be positive")
        if milp_time_limit_seconds <= 0:
            raise ValueError("milp_time_limit_seconds must be positive")
        self.residual_scheduler = residual_scheduler
        self.task_threshold = int(task_threshold)
        self.milp_scheduler = milp_scheduler or MILPOptimalScheduler(
            time_limit_seconds=milp_time_limit_seconds
        )
        self.last_decision: HybridDecision | None = None

    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        num_tasks = dag.graph.number_of_nodes()
        if num_tasks <= self.task_threshold:
            try:
                milp_result = self.milp_scheduler.solve(dag, deepcopy(resource_config))
            except Exception as exc:
                self.last_decision = HybridDecision(
                    selected_method="residual_fallback",
                    reason="milp_error",
                    num_tasks=num_tasks,
                    milp_error=f"{type(exc).__name__}: {exc}",
                )
                return self.residual_scheduler.schedule(dag, deepcopy(resource_config))

            if milp_result.proven_optimal and milp_result.schedule is not None:
                self.last_decision = self._decision_from_milp(
                    "milp_optimal", "milp_proven_optimal", num_tasks, milp_result
                )
                return milp_result.schedule

            reason = (
                "milp_not_proven_optimal"
                if milp_result.schedule is not None
                else "milp_no_feasible_solution"
            )
            self.last_decision = self._decision_from_milp(
                "residual_fallback", reason, num_tasks, milp_result
            )
            return self.residual_scheduler.schedule(dag, deepcopy(resource_config))

        self.last_decision = HybridDecision(
            selected_method="residual_fallback",
            reason="task_count_above_threshold",
            num_tasks=num_tasks,
        )
        return self.residual_scheduler.schedule(dag, deepcopy(resource_config))

    @staticmethod
    def _decision_from_milp(
        selected_method: str,
        reason: str,
        num_tasks: int,
        result: MILPSolveResult,
    ) -> HybridDecision:
        return HybridDecision(
            selected_method=selected_method,
            reason=reason,
            num_tasks=num_tasks,
            milp_status=result.status,
            milp_proven_optimal=result.proven_optimal,
            milp_makespan=result.makespan,
            milp_best_bound=result.best_bound,
            milp_solve_time_seconds=result.solve_time_seconds,
        )
