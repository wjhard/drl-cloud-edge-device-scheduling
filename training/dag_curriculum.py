from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from env.dag_generator import DAGTask, generate_random_dag


@dataclass(frozen=True)
class CurriculumPhase:
    progress_end: float
    min_tasks: int
    max_tasks: int


def make_training_dag_generator(
    min_tasks: int,
    max_tasks: int,
    edge_density_min: float,
    edge_density_max: float,
    base_seed: int | None = None,
) -> Callable[..., DAGTask]:
    if min_tasks <= 0 or max_tasks < min_tasks:
        raise ValueError("task bounds must satisfy 0 < min_tasks <= max_tasks")
    if not 0.0 <= edge_density_min <= edge_density_max <= 1.0:
        raise ValueError("edge density bounds must satisfy 0 <= min <= max <= 1")

    stream_rng = random.Random(base_seed)

    def gen(seed: int | None = None) -> DAGTask:
        rng = random.Random(seed) if seed is not None else stream_rng
        num_tasks = rng.randint(min_tasks, max_tasks)
        edge_density = rng.uniform(edge_density_min, edge_density_max)
        dag_seed = rng.randrange(0, 2**32)
        return generate_random_dag(
            num_tasks=num_tasks,
            edge_density=edge_density,
            seed=dag_seed,
        )

    return gen


class ProgressiveTaskRangeDagGenerator:
    def __init__(
        self,
        phases: list[CurriculumPhase],
        edge_density_min: float,
        edge_density_max: float,
        base_seed: int | None = None,
    ):
        if not phases:
            raise ValueError("curriculum phases must not be empty")
        if not 0.0 <= edge_density_min <= edge_density_max <= 1.0:
            raise ValueError("edge density bounds must satisfy 0 <= min <= max <= 1")
        previous_end = 0.0
        for phase in phases:
            if phase.min_tasks <= 0 or phase.max_tasks < phase.min_tasks:
                raise ValueError("phase task bounds must satisfy 0 < min_tasks <= max_tasks")
            if phase.progress_end <= previous_end or phase.progress_end > 1.0:
                raise ValueError("curriculum phase progress_end values must increase up to 1.0")
            previous_end = phase.progress_end
        if phases[-1].progress_end < 1.0:
            raise ValueError("last curriculum phase must end at progress 1.0")

        self.phases = phases
        self.edge_density_min = edge_density_min
        self.edge_density_max = edge_density_max
        self.stream_rng = random.Random(base_seed)
        self.progress_ratio = 0.0

    def set_progress_ratio(self, progress_ratio: float) -> None:
        self.progress_ratio = max(0.0, min(1.0, float(progress_ratio)))

    def _current_phase(self) -> CurriculumPhase:
        for phase in self.phases:
            if self.progress_ratio <= phase.progress_end:
                return phase
        return self.phases[-1]

    def __call__(self, seed: int | None = None) -> DAGTask:
        rng = random.Random(seed) if seed is not None else self.stream_rng
        phase = self._current_phase()
        num_tasks = rng.randint(phase.min_tasks, phase.max_tasks)
        edge_density = rng.uniform(self.edge_density_min, self.edge_density_max)
        dag_seed = rng.randrange(0, 2**32)
        return generate_random_dag(
            num_tasks=num_tasks,
            edge_density=edge_density,
            seed=dag_seed,
        )


def make_progressive_task_range_generator(
    min_tasks: int,
    max_tasks: int,
    edge_density_min: float,
    edge_density_max: float,
    base_seed: int | None = None,
    phases: list[dict] | None = None,
) -> ProgressiveTaskRangeDagGenerator:
    phase_specs = phases or [
        {"progress_end": 0.3, "min_tasks": min_tasks, "max_tasks": min(15, max_tasks)},
        {"progress_end": 0.6, "min_tasks": min_tasks, "max_tasks": min(20, max_tasks)},
        {"progress_end": 1.0, "min_tasks": min_tasks, "max_tasks": max_tasks},
    ]
    parsed_phases = [
        CurriculumPhase(
            progress_end=float(spec["progress_end"]),
            min_tasks=int(spec.get("min_tasks", min_tasks)),
            max_tasks=int(spec.get("max_tasks", max_tasks)),
        )
        for spec in phase_specs
    ]
    return ProgressiveTaskRangeDagGenerator(
        phases=parsed_phases,
        edge_density_min=edge_density_min,
        edge_density_max=edge_density_max,
        base_seed=base_seed,
    )
