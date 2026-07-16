from __future__ import annotations

from evaluation.generate_structural_generalization_scenarios import (
    GROUPS,
    generate_structural_scenarios,
)


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
