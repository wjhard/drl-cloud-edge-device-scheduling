from __future__ import annotations

from pathlib import Path

from evaluation.evaluate_wide_parallel import DEFAULT_SCENARIOS_DIR
from evaluation.generate_structural_generalization_scenarios import (
    DEFAULT_OUTPUT_ROOT,
    GROUPS,
    generate_structural_scenarios,
)
from scripts.reproduce import REPRODUCE_SCENARIOS_DIR


def test_structural_scenarios_are_grouped_deep_and_resource_paired(tmp_path):
    manifest = generate_structural_scenarios(tmp_path, seed_start=5_000_000)

    assert manifest["scenario_file_count"] == 20
    records = manifest["scenarios"]
    assert {record["group"] for record in records} == set(GROUPS)
    assert all(sum(record["group"] == group for record in records) == 5 for group in GROUPS)

    wide_longest = [
        record["longest_path_edges"] for record in records if record["group"] == "wide_parallel"
    ]
    deep_longest = [
        record["longest_path_edges"] for record in records if record["group"] == "deep_chain"
    ]
    assert sum(deep_longest) / len(deep_longest) > sum(wide_longest) / len(wide_longest)

    for task_size in manifest["task_sizes"]:
        homogeneous_path = tmp_path / "homogeneous_resources" / f"scenario_{task_size}_{manifest['task_sizes'].index(task_size)}.json"
        control_path = tmp_path / "original_control" / f"scenario_{task_size}_{manifest['task_sizes'].index(task_size)}.json"
        assert homogeneous_path.read_text(encoding="utf-8") == control_path.read_text(encoding="utf-8")


def test_structural_default_paths_are_pinned_to_v2() -> None:
    assert Path(DEFAULT_OUTPUT_ROOT).as_posix() == "evaluation/scenarios_structural_v2"
    assert DEFAULT_SCENARIOS_DIR.as_posix() == "evaluation/scenarios_structural_v2/wide_parallel"
    assert REPRODUCE_SCENARIOS_DIR == "evaluation/scenarios_structural_v2_reproduce"
