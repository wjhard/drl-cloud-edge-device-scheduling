from __future__ import annotations

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from env.scheduling_env import SchedulingEnv


def _resource_config():
    return load_resource_config(default_resource_config_path())


def test_relative_heft_reward_is_near_step_penalty_when_replaying_heft():
    dag = generate_random_dag(num_tasks=14, edge_density=0.4, seed=303)
    scheduler = HEFTScheduler()
    heft_resource_config = _resource_config()
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
        reward_mode="relative_heft",
    )
    env.reset()

    resource_index = {
        resource.id: index
        for index, resource in enumerate(env_resource_config.resources)
    }
    terminated = False
    reward = 0.0

    for task_id in task_order:
        assert task_id in env.ready_tasks
        task_slot = env.ready_tasks.index(task_id)
        action = task_slot * env.num_resources + resource_index[assignment[task_id]]
        _, reward, terminated, _, _ = env.step(action)

    assert terminated
    assert abs(reward - (-0.01)) <= 1e-9
