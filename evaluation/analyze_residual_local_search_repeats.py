from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _sample_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def analyze(results_dir: str | Path) -> dict:
    directory = Path(results_dir)
    paths = sorted(directory.glob("repeat_*.json"))
    if len(paths) < 5:
        raise RuntimeError(f"expected at least 5 repeat JSON files, found {len(paths)}")

    rows: list[dict] = []
    for index, path in enumerate(paths, start=1):
        data = json.loads(path.read_text(encoding="utf-8"))
        overall = data["overall"]
        base = float(overall["residual_bestof64_mean_ratio"])
        improved = float(overall["local_search_mean_ratio"])
        rows.append(
            {
                "repeat": index,
                "sampling_seed": int(data["sampling_seed"]),
                "residual_bestof64_mean_ratio": base,
                "local_search_mean_ratio": improved,
                "difference_local_minus_base": improved - base,
                "residual_bestof64_better_than_heft_count": int(
                    overall["residual_bestof64_better_than_heft_count"]
                ),
                "local_search_better_than_heft_count": int(
                    overall["local_search_better_than_heft_count"]
                ),
                "elapsed_seconds": float(data["elapsed_seconds"]),
                "source": path.as_posix(),
            }
        )

    base_values = np.asarray([row["residual_bestof64_mean_ratio"] for row in rows])
    improved_values = np.asarray([row["local_search_mean_ratio"] for row in rows])
    differences = improved_values - base_values
    base_wins = np.asarray([row["residual_bestof64_better_than_heft_count"] for row in rows])
    improved_wins = np.asarray([row["local_search_better_than_heft_count"] for row in rows])
    test = stats.ttest_rel(improved_values, base_values)

    summary = {
        "design": "independent system-entropy seeds; paired methods share each repeat's seed and exact Best-of-64 samples",
        "difference_definition": "local-search mean_ratio minus paired Residual Best-of-64 mean_ratio",
        "repeat_count": len(rows),
        "paired_runs": rows,
        "statistics": {
            "residual_bestof64_mean_ratio": _sample_stats(base_values),
            "local_search_mean_ratio": _sample_stats(improved_values),
            "paired_difference": _sample_stats(differences),
            "residual_bestof64_better_than_heft_count": _sample_stats(base_wins),
            "local_search_better_than_heft_count": _sample_stats(improved_wins),
            "paired_t_test_two_sided": {
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": int(test.df),
                "p_value": float(test.pvalue),
                "alpha": 0.05,
                "significant": bool(test.pvalue < 0.05),
                "new_method_better": bool(np.mean(differences) < 0.0),
            },
        },
    }
    output_path = directory / "paired_repeats_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("repeat | seed | residual_best64 | local_search | difference | base wins | new wins")
    for row in rows:
        print(
            f"{row['repeat']:>6} | {row['sampling_seed']:>10} | "
            f"{row['residual_bestof64_mean_ratio']:.12f} | "
            f"{row['local_search_mean_ratio']:.12f} | "
            f"{row['difference_local_minus_base']:+.12f} | "
            f"{row['residual_bestof64_better_than_heft_count']:>2}/20 | "
            f"{row['local_search_better_than_heft_count']:>2}/20"
        )
    print(json.dumps(summary["statistics"], indent=2))
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired local-search repeat evaluations.")
    parser.add_argument(
        "--results-dir",
        default="evaluation/results/autonomous_exploration/direction1_local_search",
    )
    args = parser.parse_args()
    analyze(args.results_dir)


if __name__ == "__main__":
    main()
