from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import evaluate


DEFAULT_CONFIG = "training/configs/ppo_mlp_ranked.yaml"
DEFAULT_MODEL_PATH = "training/checkpoints/ppo_mlp_ranked"


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _derived_results_path(config_path: str | Path, suffix: str) -> Path:
    config = _load_config(config_path)
    base_path = Path(config["evaluation"]["results_path"])
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL scheduler with best-of-N stochastic sampling.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--no-deterministic-fallback",
        action="store_true",
        help="Disable deterministic candidate in best-of-N selection.",
    )
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Only run best-of-N, without the single deterministic comparison.",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    deterministic_summary = None
    if not args.skip_deterministic:
        deterministic_path = _derived_results_path(args.config, "deterministic")
        print("BESTOFN_DETERMINISTIC_BASELINE")
        deterministic_summary = evaluate(
            args.config,
            args.model_path,
            num_samples=1,
            sampling_deterministic_fallback=False,
            results_path_override=deterministic_path,
        )

    bestof_path = _derived_results_path(args.config, f"bestof{args.num_samples}")
    print("BESTOFN_STOCHASTIC_EVALUATION")
    bestof_summary = evaluate(
        args.config,
        args.model_path,
        num_samples=args.num_samples,
        sampling_deterministic_fallback=not args.no_deterministic_fallback,
        results_path_override=bestof_path,
    )

    print("BESTOFN_COMPARISON")
    print(f"num_samples={args.num_samples}")
    print(f"seed={args.seed}")
    print(f"sampling_deterministic_fallback={not args.no_deterministic_fallback}")
    print(f"bestof_results_path={bestof_path}")
    print(f"bestof_mean_ratio={bestof_summary['overall']['mean_ratio']:.12f}")
    if deterministic_summary is not None:
        deterministic_mean = deterministic_summary["overall"]["mean_ratio"]
        bestof_mean = bestof_summary["overall"]["mean_ratio"]
        print(f"deterministic_results_path={_derived_results_path(args.config, 'deterministic')}")
        print(f"deterministic_mean_ratio={deterministic_mean:.12f}")
        print(f"mean_ratio_delta_bestof_minus_deterministic={bestof_mean - deterministic_mean:.12f}")


if __name__ == "__main__":
    main()
