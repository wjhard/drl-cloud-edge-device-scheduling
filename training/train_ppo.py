from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import types
from pathlib import Path

sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.resource_config import load_resource_config
from env.scheduling_env import SchedulingEnv
from training.dag_curriculum import make_training_dag_generator


DEFAULT_CONFIG = "training/configs/ppo_mlp_baseline.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _force_tensorboard_stub() -> None:
    # Avoid importing a globally installed TensorFlow package from TensorBoard.
    sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
    try:
        tensorflow_stub = importlib.import_module("tensorboard.compat.tensorflow_stub")
        tensorboard_compat = importlib.import_module("tensorboard.compat")
        tensorboard_compat.tf = tensorflow_stub
        tensorboard_compat.tf2 = tensorflow_stub
        for module_name in (
            "tensorboard.summary.writer.event_file_writer",
            "tensorboard.summary.writer.record_writer",
        ):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "tf"):
                module.tf = tensorflow_stub
    except Exception as exc:
        print(f"warning: failed to force TensorBoard stub: {exc}", flush=True)


class ProgressPrinterCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.interval = max(1, total_timesteps // 10)
        self.next_report = self.interval

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_report:
            percent = min(100.0, 100.0 * self.num_timesteps / self.total_timesteps)
            print(
                f"training progress: {self.num_timesteps}/{self.total_timesteps} "
                f"({percent:.1f}%)",
                flush=True,
            )
            self.next_report += self.interval
        return True


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _mask_fn(env: SchedulingEnv):
    if hasattr(env, "action_masks"):
        return env.action_masks()
    return env.unwrapped.action_masks()


def build_model(config: dict) -> tuple[MaskablePPO, ActionMasker]:
    env_config = config["env"]
    train_config = config["training"]

    dag_generator = make_training_dag_generator(
        min_tasks=int(env_config["min_tasks"]),
        max_tasks=int(env_config["max_tasks"]),
        edge_density_min=float(env_config["edge_density_min"]),
        edge_density_max=float(env_config["edge_density_max"]),
        base_seed=int(train_config["seed"]),
    )
    resource_config = load_resource_config(_resolve_project_path(env_config["resource_config_path"]))
    raw_env = SchedulingEnv(
        dag_generator_fn=dag_generator,
        resource_config=resource_config,
        max_tasks=int(env_config["max_tasks_padding"]),
    )

    monitor_log_dir = _resolve_project_path(train_config["monitor_log_dir"])
    monitor_log_dir.mkdir(parents=True, exist_ok=True)
    monitored_env = Monitor(raw_env, filename=str(monitor_log_dir / "monitor.csv"))
    masked_env = ActionMasker(monitored_env, _mask_fn)

    tensorboard_log = train_config.get("tensorboard_log")
    if tensorboard_log:
        tensorboard_log = str(_resolve_project_path(tensorboard_log))
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    policy_kwargs = {"net_arch": list(train_config["net_arch"])}
    model = MaskablePPO(
        "MlpPolicy",
        masked_env,
        learning_rate=float(train_config["learning_rate"]),
        n_steps=int(train_config["n_steps"]),
        batch_size=int(train_config["batch_size"]),
        n_epochs=int(train_config["n_epochs"]),
        gamma=float(train_config["gamma"]),
        gae_lambda=float(train_config["gae_lambda"]),
        clip_range=float(train_config["clip_range"]),
        ent_coef=float(train_config["ent_coef"]),
        policy_kwargs=policy_kwargs,
        seed=int(train_config["seed"]),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    return model, masked_env


def train(config_path: str | Path) -> Path:
    config = _load_config(config_path)
    train_config = config["training"]
    save_path = _resolve_project_path(train_config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    _force_tensorboard_stub()
    model, _ = build_model(config)
    total_timesteps = int(train_config["total_timesteps"])
    callback = ProgressPrinterCallback(total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=callback,
        tb_log_name="ppo_mlp_baseline",
    )
    model.save(str(save_path))
    elapsed = time.time() - start

    print(f"saved model to: {save_path}")
    print(f"training elapsed seconds: {elapsed:.2f}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO MLP scheduler.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to training YAML config.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
