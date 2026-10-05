from __future__ import annotations

from abc import ABC, abstractmethod

from env.dag_generator import DAGTask
from env.resource_config import ResourceConfig

ScheduleResult = dict[int, tuple[str, float, float]]


class BaseScheduler(ABC):
    @abstractmethod
    def schedule(self, dag: DAGTask, resource_config: ResourceConfig) -> ScheduleResult:
        """Return {task_id: (resource_id, start_time, finish_time)}."""

    def compute_makespan(self, schedule_result: ScheduleResult) -> float:
        if not schedule_result:
            return 0.0
        return max(finish_time for _, _, finish_time in schedule_result.values())

