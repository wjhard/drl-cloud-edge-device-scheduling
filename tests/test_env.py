from __future__ import annotations

import networkx as nx
import numpy as np

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from env.scheduling_env import SchedulingEnv


def _resource_config():
    return load_resource_config(default_resource_config_path())


def _assert_observation_matches_space(observation, env):
    assert set(observation) == set(env.observation_space.spaces)
    for key, space in env.observation_space.spaces.items():
        assert observation[key].shape == space.shape
        assert observation[key].dtype == space.dtype


def test_dag_generator_no_cycle():
    dag = generate_random_dag(num_tasks=20, edge_density=0.4, seed=1)
    assert nx.is_directed_acyclic_graph(dag.graph)


def test_env_reset_and_step():
    env = SchedulingEnv(
        dag_generator_fn=lambda seed=None: generate_random_dag(8, edge_density=0.35, seed=seed),
        resource_config=_resource_config(),
        max_tasks=20,
    )
    observation, info = env.reset(seed=2)
    _assert_observation_matches_space(observation, env)
    assert info["ready_tasks"]

    terminated = False
    steps = 0
    while not terminated:
        legal_actions = np.flatnonzero(env.action_masks())
        assert legal_actions.size > 0
        observation, reward, terminated, truncated, info = env.step(int(legal_actions[0]))
        _assert_observation_matches_space(observation, env)
        assert not truncated
        assert reward <= -1.0
        steps += 1

    assert steps == env.dag.graph.number_of_nodes()
    assert info["makespan"] > 0.0


def test_heft_valid_schedule():
    dag = generate_random_dag(num_tasks=12, edge_density=0.45, seed=3)
    resource_config = _resource_config()
    scheduler = HEFTScheduler()
    schedule = scheduler.schedule(dag, resource_config)
    assert set(schedule) == set(dag.graph.nodes)

    for predecessor, successor, edge_data in dag.graph.edges(data=True):
        pred_resource_id, _, pred_finish = schedule[predecessor]
        succ_resource_id, succ_start, _ = schedule[successor]
        communication_time = resource_config.get_communication_time(
            edge_data["data_size"],
            pred_resource_id,
            succ_resource_id,
        )
        assert succ_start >= pred_finish + communication_time - 1e-9

    assert scheduler.compute_makespan(schedule) > 0.0


def test_action_mask_consistency():
    env = SchedulingEnv(
        dag_generator_fn=lambda seed=None: generate_random_dag(10, edge_density=0.3, seed=seed),
        resource_config=_resource_config(),
        max_tasks=20,
    )
    env.reset(seed=4)
    mask = env.action_masks()
    assert mask.dtype == bool
    assert mask.shape == (env.action_space.n,)
    assert int(mask.sum()) == len(env.ready_tasks) * env.num_resources
