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
        local = float(overall["local_search_mean_ratio"])
        lns = float(overall["lns_mean_ratio"])
        rows.append(
            {
                "repeat": index,
                "sampling_seed": int(data["sampling_seed"]),
                "local_search_mean_ratio": local,
                "lns_mean_ratio": lns,
                "difference_lns_minus_local": lns - local,
                "local_search_better_than_heft_count": int(
                    overall["local_search_better_than_heft_count"]
                ),
                "lns_better_than_heft_count": int(overall["lns_better_than_heft_count"]),
                "elapsed_seconds": float(data["elapsed_seconds"]),
                "source": path.as_posix(),
            }
        )
    local_values = np.asarray([row["local_search_mean_ratio"] for row in rows])
    lns_values = np.asarray([row["lns_mean_ratio"] for row in rows])
    differences = lns_values - local_values
    local_wins = np.asarray([row["local_search_better_than_heft_count"] for row in rows])
    lns_wins = np.asarray([row["lns_better_than_heft_count"] for row in rows])
    test = stats.ttest_rel(lns_values, local_values)
    summary = {
        "design": "independent system-entropy seeds; LNS and local-search baseline are paired within each run",
        "difference_definition": "LNS mean_ratio minus local-search mean_ratio",
        "repeat_count": len(rows),
        "paired_runs": rows,
        "statistics": {
            "local_search_mean_ratio": _sample_stats(local_values),
            "lns_mean_ratio": _sample_stats(lns_values),
            "paired_difference": _sample_stats(differences),
            "local_search_better_than_heft_count": _sample_stats(local_wins),
            "lns_better_than_heft_count": _sample_stats(lns_wins),
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
    print("repeat | seed | local search | LNS | difference | local wins | LNS wins")
    for row in rows:
        print(
            f"{row['repeat']:>6} | {row['sampling_seed']:>10} | "
            f"{row['local_search_mean_ratio']:.12f} | {row['lns_mean_ratio']:.12f} | "
            f"{row['difference_lns_minus_local']:+.12f} | "
            f"{row['local_search_better_than_heft_count']:>2}/20 | "
            f"{row['lns_better_than_heft_count']:>2}/20"
        )
    print(json.dumps(summary["statistics"], indent=2))
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired LNS repeat evaluations.")
    parser.add_argument(
        "--results-dir",
        default="evaluation/results/autonomous_exploration/direction2_lns",
    )
    args = parser.parse_args()
    analyze(args.results_dir)


if __name__ == "__main__":
    main()
