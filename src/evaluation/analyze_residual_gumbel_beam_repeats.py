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
        baseline = float(overall["bestof64_lns_mean_ratio"])
        beam = float(overall["gumbel_beam_lns_mean_ratio"])
        rows.append(
            {
                "repeat": index,
                "sampling_seed": int(data["sampling_seed"]),
                "bestof64_lns_mean_ratio": baseline,
                "gumbel_beam_lns_mean_ratio": beam,
                "difference_beam_minus_bestof64": beam - baseline,
                "bestof64_lns_better_than_heft_count": int(
                    overall["bestof64_lns_better_than_heft_count"]
                ),
                "gumbel_beam_lns_better_than_heft_count": int(
                    overall["gumbel_beam_lns_better_than_heft_count"]
                ),
                "elapsed_seconds": float(data["elapsed_seconds"]),
                "source": path.as_posix(),
            }
        )
    baseline_values = np.asarray([row["bestof64_lns_mean_ratio"] for row in rows])
    beam_values = np.asarray([row["gumbel_beam_lns_mean_ratio"] for row in rows])
    differences = beam_values - baseline_values
    baseline_wins = np.asarray([row["bestof64_lns_better_than_heft_count"] for row in rows])
    beam_wins = np.asarray([row["gumbel_beam_lns_better_than_heft_count"] for row in rows])
    elapsed = np.asarray([row["elapsed_seconds"] for row in rows])
    test = stats.ttest_rel(beam_values, baseline_values)
    summary = {
        "design": "independent system-entropy seeds; beam and Best-of-64 LNS share each run seed",
        "difference_definition": "Gumbel-beam LNS mean_ratio minus Best-of-64 LNS mean_ratio",
        "repeat_count": len(rows),
        "paired_runs": rows,
        "statistics": {
            "bestof64_lns_mean_ratio": _sample_stats(baseline_values),
            "gumbel_beam_lns_mean_ratio": _sample_stats(beam_values),
            "paired_difference": _sample_stats(differences),
            "bestof64_lns_better_than_heft_count": _sample_stats(baseline_wins),
            "gumbel_beam_lns_better_than_heft_count": _sample_stats(beam_wins),
            "elapsed_seconds": _sample_stats(elapsed),
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
    print("repeat | seed | Best64+LNS | beam+LNS | difference | base wins | beam wins")
    for row in rows:
        print(
            f"{row['repeat']:>6} | {row['sampling_seed']:>10} | "
            f"{row['bestof64_lns_mean_ratio']:.12f} | "
            f"{row['gumbel_beam_lns_mean_ratio']:.12f} | "
            f"{row['difference_beam_minus_bestof64']:+.12f} | "
            f"{row['bestof64_lns_better_than_heft_count']:>2}/20 | "
            f"{row['gumbel_beam_lns_better_than_heft_count']:>2}/20"
        )
    print(json.dumps(summary["statistics"], indent=2))
    print(f"results_path={output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired Gumbel-beam repeats.")
    parser.add_argument(
        "--results-dir",
        default="evaluation/results/autonomous_exploration/direction3_gumbel_beam",
    )
    args = parser.parse_args()
    analyze(args.results_dir)


if __name__ == "__main__":
    main()
