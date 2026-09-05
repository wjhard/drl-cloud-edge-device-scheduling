from __future__ import annotations

import networkx as nx

from env.dag_generator import generate_random_dag, save_dag_to_json
from env.resource_config import load_resource_config
from evaluation.schedule_validation import validate_schedule
from policies.residual_local_search_scheduler import schedule_task_order
from scripts.schedule import run_schedule


def test_schedule_validation_accepts_legal_schedule_and_rejects_overlap() -> None:
    dag = generate_random_dag(num_tasks=10, edge_density=0.30, seed=77)
    resources = load_resource_config("configs/resource_default.yaml")
    order = list(nx.topological_sort(dag.graph))
    schedule = schedule_task_order(dag, resources, order)

    valid = validate_schedule(dag, resources, schedule)
    assert valid["valid"] is True
    assert valid["scheduled_task_count"] == 10

    first_task, second_task = sorted(schedule)[:2]
    first_resource, first_start, first_finish = schedule[first_task]
    schedule[second_task] = (first_resource, first_start, first_finish)
    invalid = validate_schedule(dag, resources, schedule)
    assert invalid["valid"] is False
    assert invalid["error_count"] >= 1


def test_heft_cli_result_is_complete_valid_and_json_serializable(tmp_path) -> None:
    dag_path = tmp_path / "scenario.json"
    save_dag_to_json(
        generate_random_dag(num_tasks=12, edge_density=0.35, seed=91),
        dag_path,
    )
    result = run_schedule(input_path=dag_path, method="heft", preset="fast")

    assert result["schema_version"] == 1
    assert result["selected_source"] == "heft_anchor"
    assert result["dag"]["task_count"] == 12
    assert result["result"]["ratio_to_heft"] == 1.0
    assert result["validation"]["valid"] is True
    assert len(result["schedule"]) == 12
