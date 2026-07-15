from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

from sb3_contrib import MaskablePPO

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from env.scheduling_env import SchedulingEnv
from evaluation.generate_validation_scenarios import generate_scenarios


DEFAULT_CONFIG = "training/configs/ppo_mlp_normalized.yaml"
DEFAULT_MODEL_PATH = "training/checkpoints/ppo_mlp_normalized"


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _ensure_scenarios(config_path: str | Path, scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.exists() or not any(scenarios_dir.glob("*.json")):
        generate_scenarios(config_path)
    return sorted(scenarios_dir.glob("*.json"))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _entropy(resource_ids: list[str]) -> float:
    counts = Counter(resource_ids)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def _make_env(config: dict, scenario_path: Path) -> tuple[SchedulingEnv, gym.Env]:
    env_config = config["env"]
    dag = load_dag_from_json(scenario_path)
    raw_env = SchedulingEnv(
        dag_generator_fn=lambda seed=None, fixed_dag=dag: fixed_dag,
        resource_config=load_resource_config(env_config["resource_config_path"]),
        max_tasks=int(env_config["max_tasks_padding"]),
        reward_mode=str(env_config.get("reward_mode", "raw")),
        normalize_observations=bool(env_config.get("normalize_observations", False)),
    )
    env: gym.Env = raw_env if isinstance(raw_env.observation_space, gym.spaces.Dict) else raw_env
    env = gym.wrappers.FlattenObservation(raw_env)
    return raw_env, env


def _deterministic_complexity(
    model: MaskablePPO,
    config: dict,
    scenario_paths: list[Path],
) -> dict[str, list[float]]:
    legal_action_counts: list[float] = []
    ready_task_counts: list[float] = []
    resource_choices_per_ready_task: list[float] = []

    print("ACTION_SPACE_COMPLEXITY_BY_SCENARIO")
    for scenario_path in scenario_paths:
        raw_env, env = _make_env(config, scenario_path)
        observation, _ = env.reset()
        terminated = False
        scenario_legal_counts: list[float] = []
        scenario_ready_counts: list[float] = []
        scenario_resource_counts: list[float] = []

        while not terminated:
            mask = raw_env.action_masks()
            legal_count = int(mask.sum())
            ready_count = len(raw_env.ready_tasks)
            legal_action_counts.append(float(legal_count))
            ready_task_counts.append(float(ready_count))
            resource_choices_per_ready_task.append(float(legal_count / max(1, ready_count)))
            scenario_legal_counts.append(float(legal_count))
            scenario_ready_counts.append(float(ready_count))
            scenario_resource_counts.append(float(legal_count / max(1, ready_count)))

            action, _ = model.predict(observation, action_masks=mask, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(int(action))
            if truncated:
                raise RuntimeError(f"environment truncated for {scenario_path.name}")

        print(
            f"  {scenario_path.name}: steps={len(scenario_legal_counts)} "
            f"avg_legal_actions={_mean(scenario_legal_counts):.6f} "
            f"avg_ready_tasks={_mean(scenario_ready_counts):.6f} "
            f"avg_resources_per_ready_task={_mean(scenario_resource_counts):.6f}"
        )

    return {
        "legal_action_counts": legal_action_counts,
        "ready_task_counts": ready_task_counts,
        "resource_choices_per_ready_task": resource_choices_per_ready_task,
    }


def _resource_stability(
    model: MaskablePPO,
    config: dict,
    scenario_paths: list[Path],
    stochastic_rollouts: int,
) -> None:
    resource_choices: dict[tuple[str, int], list[str]] = defaultdict(list)
    decision_steps: dict[tuple[str, int], list[int]] = defaultdict(list)

    for scenario_path in scenario_paths:
        for _ in range(stochastic_rollouts):
            raw_env, env = _make_env(config, scenario_path)
            observation, _ = env.reset()
            terminated = False
            step_index = 0

            while not terminated:
                mask = raw_env.action_masks()
                action, _ = model.predict(observation, action_masks=mask, deterministic=False)
                action_int = int(action)
                task_slot = action_int // raw_env.num_resources
                resource_index = action_int % raw_env.num_resources
                task_id = int(raw_env.ready_tasks[task_slot])
                resource_id = raw_env.resource_config.resources[resource_index].id
                key = (scenario_path.name, task_id)
                resource_choices[key].append(resource_id)
                decision_steps[key].append(step_index)

                observation, _, terminated, truncated, _ = env.step(action_int)
                if truncated:
                    raise RuntimeError(f"environment truncated for {scenario_path.name}")
                step_index += 1

    unique_counts: list[float] = []
    majority_ratios: list[float] = []
    entropies: list[float] = []
    step_spans: list[float] = []
    unstable_rows: list[tuple[int, float, float, str, int, Counter[str], list[int]]] = []

    for key, choices in resource_choices.items():
        counts = Counter(choices)
        unique_count = len(counts)
        majority_ratio = max(counts.values()) / len(choices)
        entropy = _entropy(choices)
        steps = decision_steps[key]
        step_span = max(steps) - min(steps)
        unique_counts.append(float(unique_count))
        majority_ratios.append(float(majority_ratio))
        entropies.append(float(entropy))
        step_spans.append(float(step_span))
        if unique_count > 1:
            scenario_name, task_id = key
            unstable_rows.append((unique_count, entropy, majority_ratio, scenario_name, task_id, counts, steps))

    unstable_rows.sort(key=lambda row: (-row[0], -row[1], row[2], row[3], row[4]))
    print("RESOURCE_SELECTION_STABILITY")
    print(f"stochastic_rollouts_per_scenario={stochastic_rollouts}")
    print(f"tracked_scenario_task_pairs={len(resource_choices)}")
    print(f"unstable_pairs_unique_resource_gt_1={sum(1 for value in unique_counts if value > 1.0)}")
    print(f"unstable_pair_ratio={sum(1 for value in unique_counts if value > 1.0) / max(1, len(unique_counts)):.6f}")
    print(f"avg_unique_resource_count={_mean(unique_counts):.6f}")
    print(f"avg_resource_choice_majority_ratio={_mean(majority_ratios):.6f}")
    print(f"avg_resource_choice_entropy={_mean(entropies):.6f}")
    print(f"avg_decision_step_span={_mean(step_spans):.6f}")
    print("top_unstable_pairs:")
    for unique_count, entropy, majority_ratio, scenario_name, task_id, counts, steps in unstable_rows[:10]:
        step_summary = f"min={min(steps)},max={max(steps)},unique={len(set(steps))}"
        counts_text = ", ".join(f"{resource}={count}" for resource, count in sorted(counts.items()))
        print(
            f"  scenario={scenario_name} task_id={task_id} "
            f"unique_resources={unique_count} majority_ratio={majority_ratio:.6f} "
            f"entropy={entropy:.6f} step_span=({step_summary}) choices=[{counts_text}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose joint action-space complexity.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stochastic-rollouts", type=int, default=20)
    args = parser.parse_args()

    config = _load_config(args.config)
    scenario_paths = _ensure_scenarios(args.config, Path(config["evaluation"]["scenarios_dir"]))
    model_path = Path(args.model_path)
    model_zip = model_path if model_path.suffix == ".zip" else Path(f"{model_path}.zip")

    print("ACTION_SPACE_COMPLEXITY_CHECK")
    print(f"config={Path(args.config).resolve()}")
    print(f"model_path={model_path.resolve()}")
    print(f"model_zip={model_zip.resolve()}")
    print(f"model_zip_size_bytes={model_zip.stat().st_size}")
    print(f"scenario_count={len(scenario_paths)}")
    print(f"scenario_dir={Path(config['evaluation']['scenarios_dir']).resolve()}")

    model = MaskablePPO.load(str(model_path))
    complexity = _deterministic_complexity(model, config, scenario_paths)
    print("ACTION_SPACE_COMPLEXITY_AGGREGATE")
    print(f"total_decision_steps={len(complexity['legal_action_counts'])}")
    print(f"avg_legal_actions_per_step={_mean(complexity['legal_action_counts']):.6f}")
    print(f"std_legal_actions_per_step={_std(complexity['legal_action_counts']):.6f}")
    print(f"max_legal_actions_per_step={max(complexity['legal_action_counts']):.0f}")
    print(f"avg_candidate_tasks_per_step={_mean(complexity['ready_task_counts']):.6f}")
    print(f"max_candidate_tasks_per_step={max(complexity['ready_task_counts']):.0f}")
    print(
        "avg_candidate_resources_per_ready_task="
        f"{_mean(complexity['resource_choices_per_ready_task']):.6f}"
    )
    _resource_stability(model, config, scenario_paths, args.stochastic_rollouts)


if __name__ == "__main__":
    main()
