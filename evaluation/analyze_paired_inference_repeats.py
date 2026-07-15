from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "evaluation" / "results"


def _load_runs(patterns: list[str], metric: str, ratio_field: str) -> dict[int, dict]:
    runs: dict[int, dict] = {}
    for pattern in patterns:
        for path in sorted(RESULTS_ROOT.glob(pattern)):
            summary = json.loads(path.read_text(encoding="utf-8"))
            seed = int(summary["sampling_seed"])
            record = {
                "mean_ratio": float(summary["overall"][metric]),
                "better_than_heft_count": sum(
                    float(scenario[ratio_field]) < 1.0 for scenario in summary["scenarios"]
                ),
                "source": str(path.relative_to(ROOT)),
            }
            if seed in runs and runs[seed] != record:
                raise RuntimeError(f"conflicting duplicate result for seed {seed}")
            runs[seed] = record
    return runs


def _sample_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    best64 = _load_runs(
        [
            "repeated_bestof64/summary_residual_bestof64_run*.json",
            "paired15/bestof64/summary_seed_*.json",
        ],
        metric="mean_ratio",
        ratio_field="ratio",
    )
    hybrid = _load_runs(
        [
            "repeated/summary_hybrid_residual_bestof16_run*.json",
            "paired15/hybrid/summary_seed_*.json",
        ],
        metric="hybrid_mean_ratio",
        ratio_field="hybrid_ratio",
    )
    seeds = sorted(set(best64) & set(hybrid))
    if len(seeds) != 15 or set(best64) != set(hybrid):
        raise RuntimeError(
            f"expected exactly 15 fully paired seeds, got best64={len(best64)}, "
            f"hybrid={len(hybrid)}, intersection={len(seeds)}"
        )

    best64_values = np.asarray([best64[seed]["mean_ratio"] for seed in seeds])
    hybrid_values = np.asarray([hybrid[seed]["mean_ratio"] for seed in seeds])
    differences = best64_values - hybrid_values
    best64_counts = np.asarray([best64[seed]["better_than_heft_count"] for seed in seeds])
    hybrid_counts = np.asarray([hybrid[seed]["better_than_heft_count"] for seed in seeds])
    test = stats.ttest_rel(best64_values, hybrid_values)

    paired_rows = []
    for index, seed in enumerate(seeds, start=1):
        paired_rows.append(
            {
                "pair": index,
                "seed": seed,
                "bestof64_mean_ratio": float(best64_values[index - 1]),
                "hybrid_mean_ratio": float(hybrid_values[index - 1]),
                "difference_bestof64_minus_hybrid": float(differences[index - 1]),
                "bestof64_better_than_heft_count": int(best64_counts[index - 1]),
                "hybrid_better_than_heft_count": int(hybrid_counts[index - 1]),
                "bestof64_source": best64[seed]["source"],
                "hybrid_source": hybrid[seed]["source"],
            }
        )

    summary = {
        "design": "15 paired runs using identical entropy-derived seed within each pair",
        "difference_definition": "Residual best-of-64 mean_ratio minus Hybrid mean_ratio",
        "standard_deviation": "sample standard deviation (n-1)",
        "paired_runs": paired_rows,
        "statistics": {
            "bestof64_mean_ratio": _sample_stats(best64_values),
            "hybrid_mean_ratio": _sample_stats(hybrid_values),
            "bestof64_better_than_heft_count": _sample_stats(best64_counts),
            "hybrid_better_than_heft_count": _sample_stats(hybrid_counts),
            "paired_difference": _sample_stats(differences),
            "paired_t_test_two_sided": {
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": int(test.df),
                "p_value": float(test.pvalue),
                "alpha": 0.05,
                "significant": bool(test.pvalue < 0.05),
            },
        },
    }
    output_path = RESULTS_ROOT / "paired15" / "paired_comparison_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("pair | seed | bestof64 | hybrid | best64-hybrid | best64 wins | hybrid wins")
    for row in paired_rows:
        print(
            f"{row['pair']:>4} | {row['seed']:>10} | "
            f"{row['bestof64_mean_ratio']:.12f} | {row['hybrid_mean_ratio']:.12f} | "
            f"{row['difference_bestof64_minus_hybrid']:+.12f} | "
            f"{row['bestof64_better_than_heft_count']:>2}/20 | "
            f"{row['hybrid_better_than_heft_count']:>2}/20"
        )
    print(json.dumps(summary["statistics"], indent=2))
    print(f"results_path: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
