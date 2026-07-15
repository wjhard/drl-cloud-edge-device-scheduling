from __future__ import annotations

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from env.scheduling_env_ranked import SchedulingEnvRanked


def _resource_config():
    return load_resource_config(default_resource_config_path())


def test_ranked_env_reset_and_step():
    dag = generate_random_dag(num_tasks=10, edge_density=0.35, seed=707)
    env = SchedulingEnvRanked(
        dag_generator_fn=lambda seed=None: dag,
        resource_config=_resource_config(),
        max_tasks=20,
        reward_mode="relative_heft",
    )

    observation, info = env.reset()
    assert observation["task_features"].shape == (20, env.task_feature_dim)
    assert info["ready_tasks"]

    legal_actions = env.action_masks().nonzero()[0]
    assert legal_actions.size == len(env.ready_tasks)

    _, _, terminated, truncated, step_info = env.step(int(legal_actions[0]))
    assert not truncated
    assert not terminated
    assert step_info["task_id"] in dag.graph.nodes
    assert step_info["resource_id"] in {resource.id for resource in env.resource_config.resources}


def test_ranked_env_matches_heft_when_replaying_heft_task_order():
    dag = generate_random_dag(num_tasks=18, edge_density=0.45, seed=808)
    scheduler = HEFTScheduler()
    heft_resource_config = _resource_config()
    heft_schedule = scheduler.schedule(dag, heft_resource_config)
    task_order = scheduler._task_order(dag, heft_resource_config)

    env = SchedulingEnvRanked(
        dag_generator_fn=lambda seed=None: dag,
        resource_config=_resource_config(),
        max_tasks=25,
        reward_mode="relative_heft",
    )
    env.reset()

    terminated = False
    info = {"makespan": 0.0}
    for task_id in task_order:
        assert task_id in env.ready_tasks
        action = env.ready_tasks.index(task_id)
        _, _, terminated, _, info = env.step(action)

    assert terminated
    assert abs(info["makespan"] - scheduler.compute_makespan(heft_schedule)) <= 1e-9
