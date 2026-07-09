from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import generate_random_dag
from env.resource_config import default_resource_config_path, load_resource_config
from policies.rl_scheduler import RLScheduler
from training.behavior_cloning import pretrain_policy_with_bc
from training.train_ppo import _load_config, _resolve_project_path, build_model


DEFAULT_CONFIG = "training/configs/ppo_mlp_bc_warmstart.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_bc_dataset_quick.npz"
DEFAULT_MODEL_PATH = "training/checkpoints/ppo_mlp_bc_warmstart_quick_bc_only"


def _format_counts(counts: Counter[str], resource_ids: list[str]) -> str:
    total = sum(counts.values())
    parts = []
    for resource_id in resource_ids:
        count = counts.get(resource_id, 0)
        ratio = count / total if total else 0.0
        parts.append(f"{resource_id}=count:{count},ratio:{ratio:.6f}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check BC warm start before full PPO training.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dag-seed", type=int, default=515151)
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--edge-density", type=float, default=0.35)
    args = parser.parse_args()

    config = _load_config(args.config)
    model, _ = build_model(config)
    history = pretrain_policy_with_bc(
        model,
        dataset_path=_resolve_project_path(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    first = history[0]
    last = history[-1]
    print("BC_WARMSTART_CHECK")
    print(f"first_loss={first['loss']:.6f}")
    print(f"last_loss={last['loss']:.6f}")
    print(f"loss_delta={last['loss'] - first['loss']:.6f}")
    print(f"first_accuracy={first['accuracy']:.6f}")
    print(f"last_accuracy={last['accuracy']:.6f}")
    print(f"accuracy_delta={last['accuracy'] - first['accuracy']:.6f}")

    model_path = _resolve_project_path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"saved_bc_only_model={model_path}")

    dag = generate_random_dag(
        num_tasks=args.num_tasks,
        edge_density=args.edge_density,
        seed=args.dag_seed,
    )
    resource_config = load_resource_config(default_resource_config_path())
    resource_ids = [resource.id for resource in resource_config.resources]

    heft_scheduler = HEFTScheduler()
    heft_schedule = heft_scheduler.schedule(dag, load_resource_config(default_resource_config_path()))
    heft_counts = Counter(resource_id for resource_id, _, _ in heft_schedule.values())

    rl_scheduler = RLScheduler(
        model_path=model_path,
        max_tasks=int(config["env"]["max_tasks_padding"]),
        deterministic=True,
        reward_mode=str(config["env"].get("reward_mode", "raw")),
    )
    rl_schedule = rl_scheduler.schedule(dag, load_resource_config(default_resource_config_path()))
    rl_counts = Counter(resource_id for resource_id, _, _ in rl_schedule.values())

    print(f"diagnostic_dag_seed={args.dag_seed}")
    print(f"diagnostic_num_tasks={args.num_tasks}")
    print(f"heft_resource_distribution={_format_counts(heft_counts, resource_ids)}")
    print(f"bc_policy_resource_distribution={_format_counts(rl_counts, resource_ids)}")


if __name__ == "__main__":
    main()
