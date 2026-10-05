from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ScheduledEventLike(Protocol):
    task_id: int
    start_time: float
    finish_time: float


@dataclass
class ScheduledEvent:
    task_id: int
    start_time: float
    finish_time: float


def find_earliest_slot(
    events: list[ScheduledEventLike],
    ready_time: float,
    duration: float,
) -> tuple[float, float]:
    """Find the earliest insertion slot on a resource timeline."""
    start_time = ready_time
    for event in sorted(events, key=lambda current: current.start_time):
        if start_time + duration <= event.start_time:
            return start_time, start_time + duration
        start_time = max(start_time, event.finish_time)
    return start_time, start_time + duration

