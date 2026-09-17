from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg

from scripts.visualize_evaluation_summary import generate_visualizations, load_evaluation_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_evaluation_summary_reads_json_and_renders_full_hd(tmp_path: Path) -> None:
    final_path = _write_json(
        tmp_path / "final.json",
        {
            "repeat_count": 3,
            "paired_runs": [
                {"repeat": 1, "lns_mean_ratio": 0.81, "residual_bestof64_mean_ratio": 0.91, "lns_better_than_heft_count": 7},
                {"repeat": 2, "lns_mean_ratio": 0.82, "residual_bestof64_mean_ratio": 0.92, "lns_better_than_heft_count": 8},
                {"repeat": 3, "lns_mean_ratio": 0.83, "residual_bestof64_mean_ratio": 0.93, "lns_better_than_heft_count": 9},
            ],
            "statistics": {
                "lns_mean_ratio": {"mean": 0.82, "sample_std": 0.01},
                "paired_t_test_two_sided": {"p_value": 0.0042, "significant": True},
            },
        },
    )
    wide_path = _write_json(
        tmp_path / "wide.json",
        {
            "groups": {
                "adaptive": {
                    "scenario_count": 4,
                    "mean_ratio": 0.88,
                    "non_regression_vs_heft_count": 4,
                    "outperform_heft_count": 3,
                    "heft_tie_proven_optimal_count": 1,
                }
            }
        },
    )
    large_path = _write_json(
        tmp_path / "large.json",
        {
            "overall": {
                "scenario_count": 4,
                "mean_ratio": 0.89,
                "sample_std_ratio": 0.02,
                "strictly_better_than_heft_count": 4,
                "residual_skipped_count": 2,
                "selected_source_counts": {"heft_lns": 2, "residual_lns": 2},
            },
            "scenarios": [
                {"num_tasks": 20, "ratio_to_heft": 0.86, "diagnostics": {"residual_skipped_reason": None}},
                {"num_tasks": 20, "ratio_to_heft": 0.88, "diagnostics": {"residual_skipped_reason": None}},
                {"num_tasks": 40, "ratio_to_heft": 0.90, "diagnostics": {"residual_skipped_reason": "task_count=40 exceeds residual_model_capacity=32"}},
                {"num_tasks": 40, "ratio_to_heft": 0.92, "diagnostics": {"residual_skipped_reason": "task_count=40 exceeds residual_model_capacity=32"}},
            ],
        },
    )
    os_path = _write_json(
        tmp_path / "os.json",
        {
            "status": "passed",
            "jobs": [
                {"os": "System-A", "status": "passed", "build_exit_code": 0, "run_exit_code": 0},
                {"os": "System-B", "status": "passed", "build_exit_code": 0, "run_exit_code": 0},
            ],
        },
    )
    output_path = tmp_path / "summary.png"

    data = load_evaluation_summary(final_path, wide_path, large_path, os_path)
    generated = generate_visualizations(final_path, wide_path, large_path, os_path, output_path)

    assert data.repeat_count == 3
    assert data.final_mean == 0.82
    assert data.repeat_win_counts == [7, 8, 9]
    assert data.wide_non_regression == 4
    assert data.large_size_means == {20: 0.87, 40: 0.91}
    assert data.residual_capacity == 32
    assert data.residual_skipped_sizes == [40]
    assert data.os_matrix_status == "passed"
    assert generated == [tmp_path / "summary_final.png", tmp_path / "summary_generalization.png"]
    for generated_path in generated:
        assert generated_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        image = mpimg.imread(generated_path)
        assert image.shape[:2] == (1080, 1920)
