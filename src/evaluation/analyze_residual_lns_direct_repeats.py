from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scipy.stats import ttest_rel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze direct paired Residual Best-of-64 versus LNS repeats."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("direct_repeat_*.json"))
    if len(paths) < 5:
        raise RuntimeError(f"expected at least five direct repeats, found {len(paths)}")

    runs: list[dict] = []
    for repeat, path in enumerate(paths, start=1):
        result = json.loads(path.read_text(encoding="utf-8"))
        overall = result["overall"]
        baseline = float(overall["residual_bestof64_mean_ratio"])
        candidate = float(overall["lns_mean_ratio"])
        runs.append(
            {
                "repeat": repeat,
                "sampling_seed": int(result["sampling_seed"]),
                "residual_bestof64_mean_ratio": baseline,
                "lns_mean_ratio": candidate,
                "difference_lns_minus_residual": candidate - baseline,
                "residual_bestof64_better_than_heft_count": int(
                    overall["residual_bestof64_better_than_heft_count"]
                ),
                "lns_better_than_heft_count": int(overall["lns_better_than_heft_count"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "source": path.as_posix(),
            }
        )

    baseline_values = [row["residual_bestof64_mean_ratio"] for row in runs]
    candidate_values = [row["lns_mean_ratio"] for row in runs]
    differences = [row["difference_lns_minus_residual"] for row in runs]
    test = ttest_rel(candidate_values, baseline_values)

    def describe(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "design": "five system-entropy seeds; raw Residual Best-of-64 and LNS share every scenario seed",
        "difference_definition": "LNS mean_ratio minus paired raw Residual Best-of-64 mean_ratio",
        "repeat_count": len(runs),
        "paired_runs": runs,
        "statistics": {
            "residual_bestof64_mean_ratio": describe(baseline_values),
            "lns_mean_ratio": describe(candidate_values),
            "paired_difference": describe(differences),
            "paired_t_test_two_sided": {
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": len(runs) - 1,
                "p_value": float(test.pvalue),
                "alpha": 0.05,
                "significant": bool(test.pvalue < 0.05),
                "new_method_better": bool(
                    test.pvalue < 0.05
                    and statistics.fmean(candidate_values) < statistics.fmean(baseline_values)
                ),
            },
        },
    }
    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("repeat | seed | Residual Best64 | LNS | difference | base wins | LNS wins")
    for row in runs:
        print(
            f"{row['repeat']:>6} | {row['sampling_seed']:>10} | "
            f"{row['residual_bestof64_mean_ratio']:.12f} | {row['lns_mean_ratio']:.12f} | "
            f"{row['difference_lns_minus_residual']:+.12f} | "
            f"{row['residual_bestof64_better_than_heft_count']}/20 | "
            f"{row['lns_better_than_heft_count']}/20"
        )
    print(json.dumps(summary["statistics"], indent=2))
    print(f"results_path={output_path}")


if __name__ == "__main__":
    main()
