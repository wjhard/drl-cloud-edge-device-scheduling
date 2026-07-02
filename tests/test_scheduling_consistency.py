from __future__ import annotations

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from env.scheduling_env import SchedulingEnv


def _resource_config():
    return load_resource_config(default_resource_config_path())


def test_env_replay_matches_heft_makespan_for_same_assignment():
    dag = generate_random_dag(num_tasks=16, edge_density=0.45, seed=11)
    heft_resource_config = _resource_config()
    scheduler = HEFTScheduler()

    heft_schedule = scheduler.schedule(dag, heft_resource_config)
    task_order = scheduler._task_order(dag, heft_resource_config)
    assignment = {
        task_id: resource_id
        for task_id, (resource_id, _, _) in heft_schedule.items()
    }

    env_resource_config = _resource_config()
    env = SchedulingEnv(
        dag_generator_fn=lambda seed=None: dag,
        resource_config=env_resource_config,
        max_tasks=20,
    )
    env.reset()

    resource_index = {
        resource.id: index
        for index, resource in enumerate(env_resource_config.resources)
    }
    terminated = False
    info = {"makespan": 0.0}

    for task_id in task_order:
        assert task_id in env.ready_tasks
        task_slot = env.ready_tasks.index(task_id)
        action = task_slot * env.num_resources + resource_index[assignment[task_id]]
        _, _, terminated, _, info = env.step(action)

    assert terminated
    assert abs(info["makespan"] - scheduler.compute_makespan(heft_schedule)) <= 1e-9

