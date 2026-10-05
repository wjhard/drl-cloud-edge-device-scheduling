from __future__ import annotations

from collections import defaultdict

from env.dag_generator import DAGTask
from env.resource_config import ResourceConfig
from scheduler_interface import ScheduleResult


def validate_schedule(
    dag: DAGTask,
    resource_config: ResourceConfig,
    schedule: ScheduleResult,
    *,
    tolerance: float = 1e-9,
) -> dict:
    """Validate completeness, resource intervals, and DAG precedence.

    The returned dictionary is JSON serializable so both the command-line
    interface and benchmark artifacts can retain the validation evidence.
    """

    errors: list[str] = []
    expected_tasks = {int(task_id) for task_id in dag.graph.nodes}
    scheduled_tasks = {int(task_id) for task_id in schedule}
    missing_tasks = sorted(expected_tasks - scheduled_tasks)
    extra_tasks = sorted(scheduled_tasks - expected_tasks)
    if missing_tasks:
        errors.append(f"missing tasks: {missing_tasks}")
    if extra_tasks:
        errors.append(f"unknown tasks: {extra_tasks}")

    known_resources = {resource.id for resource in resource_config.resources}
    events_by_resource: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for task_id, (resource_id, start_time, finish_time) in schedule.items():
        task_id = int(task_id)
        start_time = float(start_time)
        finish_time = float(finish_time)
        if resource_id not in known_resources:
            errors.append(f"task {task_id} uses unknown resource {resource_id!r}")
            continue
        if start_time < -tolerance:
            errors.append(f"task {task_id} has negative start time {start_time}")
        if finish_time < start_time - tolerance:
            errors.append(
                f"task {task_id} finishes before it starts: {start_time} -> {finish_time}"
            )
        events_by_resource[resource_id].append((start_time, finish_time, task_id))

    for resource_id, events in events_by_resource.items():
        ordered = sorted(events)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1] - tolerance:
                errors.append(
                    f"resource {resource_id} overlap: task {previous[2]} ends at "
                    f"{previous[1]}, task {current[2]} starts at {current[0]}"
                )

    for source, target, edge_data in dag.graph.edges(data=True):
        source = int(source)
        target = int(target)
        if source not in schedule or target not in schedule:
            continue
        source_resource, _, source_finish = schedule[source]
        target_resource, target_start, _ = schedule[target]
        communication_time = resource_config.get_communication_time(
            float(edge_data.get("data_size", 0.0)),
            source_resource,
            target_resource,
        )
        ready_time = float(source_finish) + communication_time
        if float(target_start) < ready_time - tolerance:
            errors.append(
                f"precedence violation {source}->{target}: target starts at "
                f"{target_start}, earliest legal time is {ready_time}"
            )

    makespan = max((float(item[2]) for item in schedule.values()), default=0.0)
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "expected_task_count": len(expected_tasks),
        "scheduled_task_count": len(scheduled_tasks & expected_tasks),
        "makespan": makespan,
    }
