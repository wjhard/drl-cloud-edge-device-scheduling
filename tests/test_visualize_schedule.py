from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg

from scripts.visualize_schedule import generate_visualization, load_visualization_data


def test_visualize_schedule_reads_metrics_and_renders_full_hd(tmp_path: Path) -> None:
    resource_path = tmp_path / "resources.yaml"
    resource_path.write_text(
        """resources:
  - id: cloud_0
    tier: cloud
  - id: edge_0
    tier: edge
  - id: device_0
    tier: device
""",
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {"id": 0, "task_id": 0, "level": 0},
                    {"id": 1, "task_id": 1, "level": 1},
                    {"id": 2, "task_id": 2, "level": 2},
                ],
                "links": [
                    {"source": 0, "target": 1, "data_size": 1.0},
                    {"source": 1, "target": 2, "data_size": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "method": "adaptive",
                "selected_source": "unit_test_source",
                "input": {"resource_config_path": str(resource_path)},
                "dag": {"task_count": 3, "edge_count": 2},
                "result": {"makespan": 7.5, "heft_makespan": 10.0, "ratio_to_heft": 0.75},
                "timing": {"schedule_seconds": 0.125},
                "validation": {"valid": True},
                "schedule": [
                    {"task_id": 0, "resource_id": "cloud_0", "start_time": 0.0, "finish_time": 2.0},
                    {"task_id": 1, "resource_id": "edge_0", "start_time": 2.5, "finish_time": 5.0},
                    {"task_id": 2, "resource_id": "device_0", "start_time": 5.5, "finish_time": 7.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "visualization.png"

    data = load_visualization_data(scenario_path, result_path)
    generated = generate_visualization(scenario_path, result_path, output_path)

    assert data.selected_source == "unit_test_source"
    assert data.makespan == 7.5
    assert data.heft_makespan == 10.0
    assert data.ratio_to_heft == 0.75
    assert data.resource_tiers == {"cloud_0": "cloud", "edge_0": "edge", "device_0": "device"}
    assert generated == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    image = mpimg.imread(output_path)
    assert image.shape[:2] == (1080, 1920)
