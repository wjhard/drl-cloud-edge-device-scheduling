from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.heft_scheduler import HEFTScheduler
from env.dag_generator import load_dag_from_json
from env.resource_config import load_resource_config
from evaluation.generate_validation_scenarios import generate_scenarios
from policies.rl_scheduler import RLScheduler


DEFAULT_CONFIG = "training/configs/ppo_mlp_ranked.yaml"
DEFAULT_MODEL_PATH = "training/checkpoints/ppo_mlp_ranked"

_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_MODEL_PATH: str | None = None
_WORKER_NUM_SAMPLES: int | None = None
_WORKER_FALLBACK: bool | None = None
_WORKER_SEED: int | None = None
_WORKER_SCHEDULER: RLScheduler | None = None


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _ensure_scenarios(config_path: str | Path, scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.exists() or not any(scenarios_dir.glob("*.json")):
        generate_scenarios(config_path)
    return sorted(scenarios_dir.glob("*.json"))


def _worker_init(
    config_path: str,
    model_path: str,
    num_samples: int,
    sampling_deterministic_fallback: bool,
    seed: int,
) -> None:
    global _WORKER_CONFIG, _WORKER_MODEL_PATH, _WORKER_NUM_SAMPLES, _WORKER_FALLBACK, _WORKER_SEED, _WORKER_SCHEDULER
    init_start = time.perf_counter()
    _WORKER_CONFIG = _load_config(config_path)
    _WORKER_MODEL_PATH = model_path
    _WORKER_NUM_SAMPLES = num_samples
    _WORKER_FALLBACK = sampling_deterministic_fallback
    _WORKER_SEED = seed
    env_config = _WORKER_CONFIG["env"]
    scheduler_load_start = time.perf_counter()
    _WORKER_SCHEDULER = RLScheduler(
        model_path=model_path,
        max_tasks=int(env_config["max_tasks_padding"]),
        deterministic=True,
        reward_mode=str(env_config.get("reward_mode", "raw")),
        normalize_observations=bool(env_config.get("normalize_observations", False)),
        include_upward_rank_feature=bool(env_config.get("include_upward_rank_feature", False)),
        scheduler_mode=str(env_config.get("scheduler_mode", "joint")),
        num_samples=num_samples,
        sampling_deterministic_fallback=sampling_deterministic_fallback,
    )
    scheduler_load_seconds = time.perf_counter() - scheduler_load_start
    init_seconds = time.perf_counter() - init_start
    print(
        f"worker_init pid={os.getpid()} "
        f"scheduler_load_seconds={scheduler_load_seconds:.6f} "
        f"total_init_seconds={init_seconds:.6f}",
        flush=True,
    )


def _evaluate_one(index_and_path: tuple[int, str]) -> dict:
    index, scenario_path_text = index_and_path
    if _WORKER_CONFIG is None or _WORKER_SCHEDULER is None:
        raise RuntimeError("worker was not initialized")
    assert _WORKER_SEED is not None

    worker_seed = _WORKER_SEED + index
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    scenario_path = Path(scenario_path_text)
    env_config = _WORKER_CONFIG["env"]
    dag = load_dag_from_json(scenario_path)

    heft_start = time.perf_counter()
    heft_scheduler = HEFTScheduler()
    heft_resource_config = load_resource_config(env_config["resource_config_path"])
    heft_schedule = heft_scheduler.schedule(dag, heft_resource_config)
    heft_makespan = heft_scheduler.compute_makespan(heft_schedule)
    heft_elapsed_seconds = time.perf_counter() - heft_start

    rl_start = time.perf_counter()
    rl_resource_config = load_resource_config(env_config["resource_config_path"])
    rl_schedule = _WORKER_SCHEDULER.schedule(dag, rl_resource_config)
    rl_makespan = _WORKER_SCHEDULER.compute_makespan(rl_schedule)
    rl_elapsed_seconds = time.perf_counter() - rl_start
    ratio = rl_makespan / heft_makespan if heft_makespan > 0 else 0.0

    return {
        "scenario": scenario_path.name,
        "num_tasks": dag.graph.number_of_nodes(),
        "heft_makespan": heft_makespan,
        "rl_makespan": rl_makespan,
        "ratio": ratio,
        "heft_elapsed_seconds": heft_elapsed_seconds,
        "rl_elapsed_seconds": rl_elapsed_seconds,
    }


def _print_model_path_diagnostic(model_path: str | Path) -> None:
    model_path_abs = Path(model_path).resolve()
    model_zip_path = model_path_abs if model_path_abs.suffix == ".zip" else Path(f"{model_path_abs}.zip")
    mtime = os.path.getmtime(model_zip_path)
    print(f"model_path_arg_abs={model_path_abs}")
    print(f"model_zip_abs={model_zip_path}")
    print(f"model_zip_mtime_epoch={mtime:.6f}")
    print(f"model_zip_mtime_local={datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel best-of-N evaluation across scenarios.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--no-deterministic-fallback", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    evaluation_config = config["evaluation"]
    scenario_paths = _ensure_scenarios(args.config, Path(evaluation_config["scenarios_dir"]))
    result_path = Path(evaluation_config["results_path"]).with_name(
        f"{Path(evaluation_config['results_path']).stem}_parallel_bestof{args.num_samples}.json"
    )

    print("PARALLEL_BESTOFN_EVALUATION")
    print(f"config={Path(args.config).resolve()}")
    _print_model_path_diagnostic(args.model_path)
    print(f"num_samples={args.num_samples}")
    print(f"max_workers={args.max_workers}")
    print(f"seed={args.seed}")
    print(f"scenario_count={len(scenario_paths)}")

    start_time = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_worker_init,
        initargs=(
            args.config,
            args.model_path,
            args.num_samples,
            not args.no_deterministic_fallback,
            args.seed,
        ),
    ) as executor:
        records = list(executor.map(_evaluate_one, enumerate(str(path) for path in scenario_paths)))
    elapsed_seconds = time.perf_counter() - start_time

    ratios = [float(record["ratio"]) for record in records]
    heft_elapsed_values = [float(record["heft_elapsed_seconds"]) for record in records]
    rl_elapsed_values = [float(record["rl_elapsed_seconds"]) for record in records]
    by_size: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_size[int(record["num_tasks"])].append(float(record["ratio"]))

    summary = {
        "config_path": str(args.config),
        "model_path": str(args.model_path),
        "scenario_count": len(records),
        "num_samples": args.num_samples,
        "sampling_deterministic_fallback": not args.no_deterministic_fallback,
        "max_workers": args.max_workers,
        "elapsed_seconds": elapsed_seconds,
        "overall": {
            "mean_ratio": _mean(ratios),
            "std_ratio": _std(ratios),
            "better_than_heft_count": sum(1 for ratio in ratios if ratio < 1.0),
        },
        "timing_breakdown": {
            "total_heft_seconds": sum(heft_elapsed_values),
            "total_rl_schedule_seconds": sum(rl_elapsed_values),
            "mean_heft_seconds_per_scenario": _mean(heft_elapsed_values),
            "mean_rl_schedule_seconds_per_scenario": _mean(rl_elapsed_values),
        },
        "by_task_size": {
            str(task_size): {
                "count": len(size_ratios),
                "mean_ratio": _mean(size_ratios),
                "std_ratio": _std(size_ratios),
            }
            for task_size, size_ratios in sorted(by_size.items())
        },
        "scenarios": records,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("parallel evaluation summary")
    print(f"results_path: {result_path}")
    print(f"elapsed_seconds={elapsed_seconds:.6f}")
    print(f"overall mean_ratio={summary['overall']['mean_ratio']:.6f}, std={summary['overall']['std_ratio']:.6f}")
    print(f"better_than_heft_count={summary['overall']['better_than_heft_count']}")
    print(
        "timing_breakdown: "
        f"total_heft_seconds={summary['timing_breakdown']['total_heft_seconds']:.6f}, "
        f"total_rl_schedule_seconds={summary['timing_breakdown']['total_rl_schedule_seconds']:.6f}, "
        f"mean_heft_seconds_per_scenario={summary['timing_breakdown']['mean_heft_seconds_per_scenario']:.6f}, "
        f"mean_rl_schedule_seconds_per_scenario={summary['timing_breakdown']['mean_rl_schedule_seconds_per_scenario']:.6f}"
    )


if __name__ == "__main__":
    main()
