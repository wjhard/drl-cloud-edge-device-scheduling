from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_residual_lns import evaluate
from evaluation.generate_structural_generalization_scenarios import (
    DEFAULT_OUTPUT_ROOT,
    GROUPS,
    RESOURCE_CONFIGS,
    generate_structural_scenarios,
)


def evaluate_adaptive_structural(
    config_path: str | Path = "training/configs/ppo_mlp_residual.yaml",
    model_path: str | Path = "training/checkpoints/ppo_mlp_residual",
    scenarios_root: str | Path = DEFAULT_OUTPUT_ROOT,
    results_root: str | Path = "evaluation/results/structural_generalization_adaptive",
    num_samples: int = 64,
    sampling_seed: int = 20260830,
    local_max_passes: int = 3,
    lns_iterations: int = 64,
) -> dict:
    scenario_root = Path(scenarios_root)
    if not (scenario_root / "manifest.json").exists():
        generate_structural_scenarios(scenario_root)
    base_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_root = Path(results_root)
    output_root.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict] = {}

    for group in GROUPS:
        group_config = copy.deepcopy(base_config)
        group_config["env"]["resource_config_path"] = RESOURCE_CONFIGS[group]
        group_config["evaluation"]["scenarios_dir"] = str(scenario_root / group)
        generated_config = output_root / f"{group}.yaml"
        generated_config.write_text(yaml.safe_dump(group_config, sort_keys=False), encoding="utf-8")
        result_path = output_root / f"summary_{group}_adaptive.json"
        summary = evaluate(
            generated_config,
            model_path,
            results_path=result_path,
            sampling_seed=sampling_seed,
            num_samples=num_samples,
            local_max_passes=local_max_passes,
            lns_iterations=lns_iterations,
            portfolio_mode="adaptive",
        )
        groups[group] = {
            "scenario_count": summary["scenario_count"],
            "mean_ratio": summary["overall"]["lns_mean_ratio"],
            "std_ratio": statistics.pstdev(
                float(record["lns_ratio"]) for record in summary["scenarios"]
            ),
            "outperform_heft_count": summary["overall"]["lns_better_than_heft_count"],
            "selected_source_counts": summary["overall"]["portfolio_selected_source_counts"],
            "results_path": str(result_path),
        }

    result = {
        "method": "Adaptive Residual portfolio",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "scenarios_root": str(scenario_root),
        "results_root": str(output_root),
        "num_samples": num_samples,
        "sampling_seed": sampling_seed,
        "groups": groups,
    }
    output_path = output_root / "summary_structural_generalization_adaptive.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"adaptive_structural_results_path={output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive portfolio on all structural groups.")
    parser.add_argument("--config", default="training/configs/ppo_mlp_residual.yaml")
    parser.add_argument("--model-path", default="training/checkpoints/ppo_mlp_residual")
    parser.add_argument("--scenarios-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-root", default="evaluation/results/structural_generalization_adaptive")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--local-max-passes", type=int, default=3)
    parser.add_argument("--lns-iterations", type=int, default=64)
    args = parser.parse_args()
    evaluate_adaptive_structural(
        config_path=args.config,
        model_path=args.model_path,
        scenarios_root=args.scenarios_root,
        results_root=args.results_root,
        num_samples=args.num_samples,
        sampling_seed=args.seed,
        local_max_passes=args.local_max_passes,
        lns_iterations=args.lns_iterations,
    )


if __name__ == "__main__":
    main()
