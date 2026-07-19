from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scipy.stats import ttest_rel


def _describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare compute-matched sampling with LNS.")
    parser.add_argument("--sampling-dir", required=True)
    parser.add_argument("--lns-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    sampling_paths = sorted(Path(args.sampling_dir).glob("repeat_*.json"))
    lns_paths = sorted(Path(args.lns_dir).glob("direct_repeat_*.json"))
    if len(sampling_paths) != 5 or len(lns_paths) != 5:
        raise RuntimeError(
            f"expected five runs per method, found sampling={len(sampling_paths)}, LNS={len(lns_paths)}"
        )

    sampling_by_seed = {
        int(data["sampling_seed"]): (path, data)
        for path in sampling_paths
        for data in [json.loads(path.read_text(encoding="utf-8"))]
    }
    lns_by_seed = {
        int(data["sampling_seed"]): (path, data)
        for path in lns_paths
        for data in [json.loads(path.read_text(encoding="utf-8"))]
    }
    seeds = sorted(set(sampling_by_seed) & set(lns_by_seed))
    if len(seeds) != 5 or set(sampling_by_seed) != set(lns_by_seed):
        raise RuntimeError("sampling and LNS seeds are not exactly paired")

    rows: list[dict] = []
    for repeat, seed in enumerate(seeds, start=1):
        sampling_path, sampling = sampling_by_seed[seed]
        lns_path, lns = lns_by_seed[seed]
        sampling_ratio = float(sampling["overall"]["mean_ratio"])
        lns_ratio = float(lns["overall"]["lns_mean_ratio"])
        rows.append(
            {
                "repeat": repeat,
                "sampling_seed": seed,
                "pure_sampling_mean_ratio": sampling_ratio,
                "lns_mean_ratio": lns_ratio,
                "difference_lns_minus_sampling": lns_ratio - sampling_ratio,
                "pure_sampling_better_than_heft_count": int(
                    sampling["overall"]["better_than_heft_count"]
                ),
                "lns_better_than_heft_count": int(lns["overall"]["lns_better_than_heft_count"]),
                "pure_sampling_elapsed_seconds": float(sampling["elapsed_seconds"]),
                "lns_elapsed_seconds": float(lns["elapsed_seconds"]),
                "sampling_source": sampling_path.as_posix(),
                "lns_source": lns_path.as_posix(),
            }
        )

    sampling_ratios = [row["pure_sampling_mean_ratio"] for row in rows]
    lns_ratios = [row["lns_mean_ratio"] for row in rows]
    differences = [row["difference_lns_minus_sampling"] for row in rows]
    sampling_times = [row["pure_sampling_elapsed_seconds"] for row in rows]
    lns_times = [row["lns_elapsed_seconds"] for row in rows]
    test = ttest_rel(lns_ratios, sampling_ratios)
    summary = {
        "design": "five paired seeds; approximately equal 20-scenario wall-clock budgets",
        "difference_definition": "LNS mean_ratio minus compute-matched pure-sampling mean_ratio",
        "paired_runs": rows,
        "statistics": {
            "pure_sampling_mean_ratio": _describe(sampling_ratios),
            "lns_mean_ratio": _describe(lns_ratios),
            "paired_difference": _describe(differences),
            "pure_sampling_elapsed_seconds": _describe(sampling_times),
            "lns_elapsed_seconds": _describe(lns_times),
            "elapsed_ratio_sampling_over_lns": statistics.fmean(sampling_times)
            / statistics.fmean(lns_times),
            "paired_t_test_two_sided": {
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": 4,
                "p_value": float(test.pvalue),
                "alpha": 0.05,
                "significant": bool(test.pvalue < 0.05),
                "lns_better": bool(test.pvalue < 0.05 and statistics.fmean(lns_ratios) < statistics.fmean(sampling_ratios)),
            },
        },
    }
    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("repeat | seed | pure sampling | LNS | LNS-sampling | sample s | LNS s")
    for row in rows:
        print(
            f"{row['repeat']:>6} | {row['sampling_seed']:>10} | "
            f"{row['pure_sampling_mean_ratio']:.12f} | {row['lns_mean_ratio']:.12f} | "
            f"{row['difference_lns_minus_sampling']:+.12f} | "
            f"{row['pure_sampling_elapsed_seconds']:.3f} | {row['lns_elapsed_seconds']:.3f}"
        )
    print(json.dumps(summary["statistics"], indent=2))
    print(f"results_path={output_path}")


if __name__ == "__main__":
    main()
