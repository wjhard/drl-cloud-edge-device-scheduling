from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from env.scheduling_env import SchedulingEnv
from policies.gat_features_extractor import TaskGraphFeaturesExtractor
from policies.rl_scheduler import RLScheduler
from training.dag_curriculum import make_training_dag_generator


def _mask_fn(env):
    if hasattr(env, "action_masks"):
        return env.action_masks()
    return env.unwrapped.action_masks()


def _train_mini_gat_model(tmp_path: Path) -> Path:
    dag_generator = make_training_dag_generator(
        min_tasks=4,
        max_tasks=6,
        edge_density_min=0.2,
        edge_density_max=0.5,
        base_seed=311,
    )
    env = SchedulingEnv(
        dag_generator_fn=dag_generator,
        resource_config=load_resource_config(default_resource_config_path()),
        max_tasks=10,
        reward_mode="relative_heft",
    )
    masked_env = ActionMasker(env, _mask_fn)
    model = MaskablePPO(
        "MultiInputPolicy",
        masked_env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        learning_rate=0.001,
        gamma=0.95,
        policy_kwargs={
            "net_arch": [32, 32],
            "features_extractor_class": TaskGraphFeaturesExtractor,
            "features_extractor_kwargs": {
                "hidden_dim": 32,
                "num_gat_layers": 1,
                "num_heads": 2,
            },
        },
        seed=19,
        verbose=0,
    )
    model.learn(total_timesteps=512)
    model_path = tmp_path / "mini_gat_maskable_ppo"
    model.save(str(model_path))
    return model_path


def test_gat_rl_scheduler_smoke(tmp_path):
    model_path = _train_mini_gat_model(tmp_path)
    scheduler = RLScheduler(
        model_path=model_path,
        max_tasks=10,
        deterministic=True,
        reward_mode="relative_heft",
    )
    dag = generate_random_dag(num_tasks=10, edge_density=0.35, seed=404)
    resource_config = load_resource_config(default_resource_config_path())

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
