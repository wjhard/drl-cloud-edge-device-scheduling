from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import evaluate
from evaluation.generate_structural_generalization_scenarios import (
    DEFAULT_OUTPUT_ROOT,
    GROUPS,
    generate_structural_scenarios,
)


DEFAULT_CONFIG = "training/configs/ppo_mlp_residual.yaml"
DEFAULT_MODEL_PATH = "training/checkpoints/ppo_mlp_residual"
DEFAULT_RESULTS_ROOT = "evaluation/results/structural_generalization"
RESOURCE_CONFIGS = {
    "wide_parallel": "configs/resource_default.yaml",
    "deep_chain": "configs/resource_default.yaml",
    "homogeneous_resources": "configs/resource_structural_homogeneous.yaml",
    "original_control": "configs/resource_default.yaml",
}


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_structural_groups(
    config_path: str | Path = DEFAULT_CONFIG,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scenarios_root: str | Path = DEFAULT_OUTPUT_ROOT,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    num_samples: int = 64,
    sampling_seed: int = 20_260_709,
) -> dict:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    scenario_root_path = Path(scenarios_root)
    manifest_path = scenario_root_path / "manifest.json"
    if not manifest_path.exists():
        generate_structural_scenarios(scenario_root_path)

    base_config = _load_yaml(config_path)
    output_root = Path(results_root)
    generated_config_dir = output_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    combined_groups: dict[str, dict] = {}
    for group in GROUPS:
        group_config = copy.deepcopy(base_config)
        group_config["env"]["resource_config_path"] = RESOURCE_CONFIGS[group]
        group_config["evaluation"]["scenarios_dir"] = str(scenario_root_path / group)
        group_result_path = output_root / f"summary_{group}_bestof{num_samples}.json"
        group_config["evaluation"]["results_path"] = str(group_result_path)
        generated_config_path = generated_config_dir / f"{group}.yaml"
        generated_config_path.write_text(
            yaml.safe_dump(group_config, sort_keys=False),
            encoding="utf-8",
        )

        print(f"STRUCTURAL_GROUP_START group={group}")
        summary = evaluate(
            generated_config_path,
            model_path,
            num_samples=num_samples,
            sampling_deterministic_fallback=True,
            results_path_override=group_result_path,
            sampling_seed=sampling_seed,
        )
        ratios = [float(record["ratio"]) for record in summary["scenarios"]]
        combined_groups[group] = {
            "scenario_count": len(ratios),
            "mean_ratio": float(summary["overall"]["mean_ratio"]),
            "std_ratio": float(summary["overall"]["std_ratio"]),
            "outperform_heft_count": sum(ratio < 1.0 for ratio in ratios),
            "results_path": str(group_result_path),
        }
        print(f"STRUCTURAL_GROUP_END group={group}")

    control_mean = combined_groups["original_control"]["mean_ratio"]
    for stats in combined_groups.values():
        stats["mean_ratio_delta_vs_control"] = stats["mean_ratio"] - control_mean

    combined = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "num_samples": num_samples,
        "sampling_seed": sampling_seed,
        "manifest_path": str(manifest_path),
        "groups": combined_groups,
    }
    combined_path = output_root / "summary_structural_generalization.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print("STRUCTURAL_GENERALIZATION_COMPARISON")
    print("group | mean_ratio | std | ratio<1 | delta_vs_control")
    for group in GROUPS:
        stats = combined_groups[group]
        print(
            f"{group} | {stats['mean_ratio']:.12f} | {stats['std_ratio']:.12f} | "
            f"{stats['outperform_heft_count']}/{stats['scenario_count']} | "
            f"{stats['mean_ratio_delta_vs_control']:+.12f}"
        )
    print(f"combined_results_path={combined_path}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structural generalization groups.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scenarios-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_260_709)
    args = parser.parse_args()
    evaluate_structural_groups(
        config_path=args.config,
        model_path=args.model_path,
        scenarios_root=args.scenarios_root,
        results_root=args.results_root,
        num_samples=args.num_samples,
        sampling_seed=args.seed,
    )


if __name__ == "__main__":
    main()
