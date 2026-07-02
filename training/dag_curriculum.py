from __future__ import annotations

import random
from collections.abc import Callable

from env.dag_generator import DAGTask, generate_random_dag


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

