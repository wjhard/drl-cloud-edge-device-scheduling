from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.behavior_cloning import pretrain_policy_with_bc
from training.train_ppo import _load_config, _resolve_project_path, build_model


DEFAULT_CONFIG = "training/configs/ppo_mlp_bc_warmstart.yaml"
DEFAULT_DATASET = "training/bc_datasets/mlp_bc_dataset.npz"
DEFAULT_SUBSET = "training/bc_datasets/mlp_bc_dataset_overfit_5dag.npz"


def _make_subset(
    dataset_path: Path,
    subset_path: Path,
    num_dags: int,
) -> tuple[Path, dict]:
    data = np.load(dataset_path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    action_masks = np.asarray(data["action_masks"], dtype=np.bool_)
    metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}

    if "dag_task_counts" not in metadata:
        raise ValueError("dataset metadata must include dag_task_counts for DAG-aligned subset selection")
    dag_task_counts = [int(value) for value in metadata["dag_task_counts"]]
    if num_dags <= 0 or num_dags > len(dag_task_counts):
        raise ValueError(f"num_dags must be in [1, {len(dag_task_counts)}]")

    sample_count = sum(dag_task_counts[:num_dags])
    subset_metadata = {
        **metadata,
        "source_dataset": str(dataset_path),
        "subset_num_dags": num_dags,
        "subset_samples": sample_count,
        "subset_dag_task_counts": dag_task_counts[:num_dags],
    }
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subset_path,
        observations=observations[:sample_count],
        actions=actions[:sample_count],
        action_masks=action_masks[:sample_count],
        metadata=np.asarray(json.dumps(subset_metadata), dtype=np.str_),
    )
    return subset_path, subset_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit BC on a tiny DAG-aligned subset.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset-output", default=DEFAULT_SUBSET)
    parser.add_argument("--num-dags", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    dataset_path = _resolve_project_path(args.dataset)
    subset_path = _resolve_project_path(args.subset_output)
    subset_path, subset_metadata = _make_subset(dataset_path, subset_path, args.num_dags)

    print("BC_OVERFIT_CHECK")
    print(f"source_dataset={dataset_path}")
    print(f"subset_path={subset_path}")
    print(f"subset_num_dags={args.num_dags}")
    print(f"subset_samples={subset_metadata['subset_samples']}")
    print(f"subset_dag_task_counts={subset_metadata['subset_dag_task_counts']}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"learning_rate={args.learning_rate}")

    config = _load_config(args.config)
    model, _ = build_model(config)
    history = pretrain_policy_with_bc(
        model,
        dataset_path=subset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    first = history[0]
    best = max(history, key=lambda item: item["accuracy"])
    last = history[-1]
    print("BC_OVERFIT_RESULT")
    print(f"first_loss={first['loss']:.6f}")
    print(f"last_loss={last['loss']:.6f}")
    print(f"best_loss={min(item['loss'] for item in history):.6f}")
    print(f"first_accuracy={first['accuracy']:.6f}")
    print(f"last_accuracy={last['accuracy']:.6f}")
    print(f"best_accuracy={best['accuracy']:.6f}")
    print(f"best_accuracy_epoch={int(best['epoch'])}")
    print(f"reached_90_percent_accuracy={best['accuracy'] >= 0.9}")


if __name__ == "__main__":
    main()
