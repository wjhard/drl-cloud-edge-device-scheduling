from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.behavior_cloning import pretrain_policy_with_bc
from training.train_ppo import _load_config, _resolve_project_path, build_model


DEFAULT_CONFIG = "training/configs/ppo_mlp_ranked.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_ranked_bc_dataset_normalized.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavior cloning pretraining only, without PPO learn().")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to PPO YAML config.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to BC .npz dataset.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-path", default=None, help="Optional policy checkpoint path to save after BC.")
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
    if history:
        first = history[0]
        last = history[-1]
        best = max(history, key=lambda item: item["accuracy"])
        print(
            "BC_PRETRAIN_ONLY_SUMMARY "
            f"first_accuracy={first['accuracy']:.6f} "
            f"last_accuracy={last['accuracy']:.6f} "
            f"best_accuracy={best['accuracy']:.6f} "
            f"best_epoch={int(best['epoch'])}"
        )

    if args.save_path:
        save_path = _resolve_project_path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"saved_bc_model_to={save_path}")


if __name__ == "__main__":
    main()
