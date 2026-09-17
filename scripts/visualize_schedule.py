"""Render a schedule result as a presentation-ready DAG and resource timeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import networkx as nx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER_ORDER = {"cloud": 0, "edge": 1, "device": 2, "unknown": 3}
TIER_LABELS = {"cloud": "云", "edge": "边", "device": "端", "unknown": "其他"}
TIER_COLORS = {
    "cloud": "#3578B8",
    "edge": "#E38B2C",
    "device": "#3D9B68",
    "unknown": "#777D87",
}


@dataclass(frozen=True)
class VisualizationData:
    graph: nx.DiGraph
    schedule: list[dict[str, Any]]
    resource_tiers: dict[str, str]
    method: str
    selected_source: str
    task_count: int
    edge_count: int
    makespan: float
    heft_makespan: float
    ratio_to_heft: float
    schedule_seconds: float
    valid: bool
    scenario_name: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _task_id(node: dict[str, Any]) -> Any:
    if "id" in node:
        return node["id"]
    if "task_id" in node:
        return node["task_id"]
    raise ValueError("Every scenario node must contain 'id' or 'task_id'.")


def _load_graph(scenario: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in scenario.get("nodes", []):
        task_id = _task_id(node)
        graph.add_node(task_id, **{key: value for key, value in node.items() if key != "id"})
    for link in scenario.get("links", []):
        if "source" not in link or "target" not in link:
            raise ValueError("Every scenario link must contain 'source' and 'target'.")
        graph.add_edge(
            link["source"],
            link["target"],
            **{key: value for key, value in link.items() if key not in {"source", "target"}},
        )
    if not graph:
        raise ValueError("Scenario contains no tasks.")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Scenario graph must be a DAG.")
    return graph


def _infer_tier(resource_id: str) -> str:
    prefix = resource_id.lower().split("_", 1)[0]
    return prefix if prefix in {"cloud", "edge", "device"} else "unknown"


def _resolve_resource_path(result_path: Path, result: dict[str, Any]) -> Path | None:
    configured = result.get("input", {}).get("resource_config_path")
    if not configured:
        return None
    configured_path = Path(configured)
    if configured_path.is_absolute():
        return configured_path
    project_candidate = PROJECT_ROOT / configured_path
    if project_candidate.exists():
        return project_candidate
    result_candidate = result_path.parent / configured_path
    return result_candidate if result_candidate.exists() else None


def _load_resource_tiers(result_path: Path, result: dict[str, Any], schedule: list[dict[str, Any]]) -> dict[str, str]:
    tiers: dict[str, str] = {}
    resource_path = _resolve_resource_path(result_path, result)
    if resource_path is not None and resource_path.exists():
        with resource_path.open("r", encoding="utf-8") as handle:
            resource_payload = yaml.safe_load(handle) or {}
        for resource in resource_payload.get("resources", []):
            resource_id = str(resource["id"])
            tier = str(resource.get("tier", _infer_tier(resource_id))).lower()
            tiers[resource_id] = tier if tier in TIER_ORDER else "unknown"
    for row in schedule:
        resource_id = str(row["resource_id"])
        tiers.setdefault(resource_id, _infer_tier(resource_id))
    return tiers


def load_visualization_data(scenario_path: Path, result_path: Path) -> VisualizationData:
    scenario = _load_json(scenario_path)
    result = _load_json(result_path)
    graph = _load_graph(scenario)
    schedule = result.get("schedule", result.get("assignments", []))
    if not schedule:
        raise ValueError("Schedule result contains no assignments.")
    required_row_fields = {"task_id", "resource_id", "start_time", "finish_time"}
    for row in schedule:
        missing = required_row_fields.difference(row)
        if missing:
            raise ValueError(f"Schedule row is missing fields: {sorted(missing)}")

    result_metrics = result.get("result", {})
    timing = result.get("timing", {})
    validation = result.get("validation", {})
    dag_metrics = result.get("dag", {})
    makespan = float(result_metrics.get("makespan", max(float(row["finish_time"]) for row in schedule)))
    heft_makespan = float(result_metrics["heft_makespan"])
    ratio = float(result_metrics.get("ratio_to_heft", makespan / heft_makespan))
    return VisualizationData(
        graph=graph,
        schedule=list(schedule),
        resource_tiers=_load_resource_tiers(result_path, result, schedule),
        method=str(result.get("method", "unknown")),
        selected_source=str(result.get("selected_source", "unknown")),
        task_count=int(dag_metrics.get("task_count", graph.number_of_nodes())),
        edge_count=int(dag_metrics.get("edge_count", graph.number_of_edges())),
        makespan=makespan,
        heft_makespan=heft_makespan,
        ratio_to_heft=ratio,
        schedule_seconds=float(timing.get("schedule_seconds", 0.0)),
        valid=bool(validation.get("valid", False)),
        scenario_name=scenario_path.name,
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F4F7FB",
            "axes.facecolor": "#FFFFFF",
            "text.color": "#172033",
            "axes.labelcolor": "#344054",
            "xtick.color": "#475467",
            "ytick.color": "#344054",
        }
    )


def _draw_kpis(ax: plt.Axes, data: VisualizationData) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    improvement = (1.0 - data.ratio_to_heft) * 100.0
    change_label = "提升" if improvement >= 0 else "下降"
    cards = [
        ("执行策略", data.selected_source, "#6A4BBC"),
        ("任务 / 依赖", f"{data.task_count} / {data.edge_count}", "#1778B5"),
        ("本方法 Makespan", f"{data.makespan:.6f}", "#137A55"),
        ("HEFT Makespan", f"{data.heft_makespan:.6f}", "#B26713"),
        ("相对 HEFT", f"{data.ratio_to_heft:.6f}\n{change_label} {abs(improvement):.2f}%", "#A43D54"),
        ("合法性 / 耗时", f"{'通过' if data.valid else '失败'}\n{data.schedule_seconds:.3f}s", "#137A55" if data.valid else "#B42318"),
    ]
    gap = 0.012
    width = (1.0 - gap * (len(cards) - 1)) / len(cards)
    for index, (label, value, color) in enumerate(cards):
        x = index * (width + gap)
        card = FancyBboxPatch(
            (x, 0.08),
            width,
            0.82,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        ax.add_patch(card)
        ax.text(x + 0.018, 0.68, label, fontsize=10.5, color="#667085", va="center")
        value_size = 11.5 if "\n" in value else 14
        ax.text(
            x + 0.018,
            0.34,
            value,
            fontsize=value_size,
            fontweight="bold",
            color=color,
            va="center",
            linespacing=1.12,
        )


def _dag_positions(graph: nx.DiGraph) -> dict[Any, tuple[float, float]]:
    node_levels: dict[Any, int] = {}
    generations = list(nx.topological_generations(graph))
    for level, generation in enumerate(generations):
        for node in generation:
            configured_level = graph.nodes[node].get("level")
            node_levels[node] = int(configured_level) if configured_level is not None else level
    grouped: dict[int, list[Any]] = {}
    for node, level in node_levels.items():
        grouped.setdefault(level, []).append(node)
    positions: dict[Any, tuple[float, float]] = {}
    max_level = max(grouped) if grouped else 0
    for level, nodes in sorted(grouped.items()):
        ordered = sorted(nodes, key=lambda item: str(item))
        for index, node in enumerate(ordered):
            x = level / max(max_level, 1)
            y = 1.0 - (index + 1) / (len(ordered) + 1)
            positions[node] = (x, y)
    return positions


def _draw_dag(ax: plt.Axes, data: VisualizationData) -> None:
    assignment_by_task = {row["task_id"]: str(row["resource_id"]) for row in data.schedule}
    positions = _dag_positions(data.graph)
    node_colors = [
        TIER_COLORS[data.resource_tiers.get(assignment_by_task.get(node, ""), "unknown")]
        for node in data.graph.nodes
    ]
    nx.draw_networkx_edges(
        data.graph,
        positions,
        ax=ax,
        edge_color="#98A2B3",
        width=1.4,
        arrows=True,
        arrowsize=17,
        node_size=720,
        connectionstyle="arc3,rad=0.02",
    )
    nx.draw_networkx_nodes(
        data.graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=720,
        edgecolors="#FFFFFF",
        linewidths=2.0,
    )
    nx.draw_networkx_labels(
        data.graph,
        positions,
        labels={node: f"T{node}" for node in data.graph.nodes},
        ax=ax,
        font_size=10,
        font_color="white",
        font_weight="bold",
    )
    ax.set_title("输入任务 DAG 与最终资源归属", fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.text(
        0.0,
        1.01,
        f"场景：{data.scenario_name}　箭头表示前驱约束；节点颜色表示最终分配层级",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#667085",
    )
    ax.margins(x=0.13, y=0.16)
    ax.axis("off")


def _draw_gantt(ax: plt.Axes, data: VisualizationData) -> None:
    resources = sorted(
        data.resource_tiers,
        key=lambda resource_id: (TIER_ORDER[data.resource_tiers[resource_id]], resource_id),
    )
    y_by_resource = {resource_id: len(resources) - index - 1 for index, resource_id in enumerate(resources)}
    makespan = max(data.makespan, max(float(row["finish_time"]) for row in data.schedule))
    for row in sorted(data.schedule, key=lambda item: (str(item["resource_id"]), float(item["start_time"]))):
        resource_id = str(row["resource_id"])
        tier = data.resource_tiers.get(resource_id, "unknown")
        start = float(row["start_time"])
        finish = float(row["finish_time"])
        width = max(finish - start, 0.0)
        y = y_by_resource[resource_id]
        ax.barh(
            y,
            width,
            left=start,
            height=0.58,
            color=TIER_COLORS[tier],
            edgecolor="#FFFFFF",
            linewidth=1.2,
            alpha=0.94,
        )
        ax.text(
            start + width / 2,
            y,
            f"T{row['task_id']}",
            ha="center",
            va="center",
            color="white",
            fontsize=9.5,
            fontweight="bold",
            clip_on=True,
        )
    ax.axvline(data.makespan, color="#B42318", linestyle="--", linewidth=1.6, label="本方法 Makespan")
    ax.axvline(data.heft_makespan, color="#B26713", linestyle=":", linewidth=2.0, label="HEFT Makespan")
    ax.set_yticks(
        list(y_by_resource.values()),
        [f"{resource_id}  [{TIER_LABELS[data.resource_tiers[resource_id]]}]" for resource_id in resources],
    )
    ax.set_xlim(0, max(makespan, data.heft_makespan) * 1.08)
    ax.set_xlabel("调度时间")
    ax.set_title("云—边—端资源泳道甘特图", fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="x", color="#D0D5DD", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)


def generate_visualization(scenario_path: Path, result_path: Path, output_path: Path) -> Path:
    data = load_visualization_data(scenario_path, result_path)
    _configure_style()
    figure = plt.figure(figsize=(12.8, 7.2), dpi=150)
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[0.20, 0.12, 0.68],
        width_ratios=[0.44, 0.56],
        left=0.045,
        right=0.975,
        top=0.94,
        bottom=0.08,
        hspace=0.18,
        wspace=0.09,
    )
    title_ax = figure.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.72, "云—边—端异构计算资源调度结果", fontsize=23, fontweight="bold", va="center")
    title_ax.text(
        0.0,
        0.25,
        f"由真实场景与调度结果文件即时生成　|　method={data.method}",
        fontsize=11,
        color="#667085",
        va="center",
    )
    kpi_ax = figure.add_subplot(grid[1, :])
    _draw_kpis(kpi_ax, data)
    dag_ax = figure.add_subplot(grid[2, 0])
    _draw_dag(dag_ax, data)
    gantt_ax = figure.add_subplot(grid[2, 1])
    _draw_gantt(gantt_ax, data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a schedule JSON as a 1920x1080 DAG and Gantt dashboard.")
    parser.add_argument("--scenario", required=True, type=Path, help="Input DAG node-link JSON.")
    parser.add_argument("--result", required=True, type=Path, help="Output JSON generated by scripts/schedule.py.")
    parser.add_argument("--output", required=True, type=Path, help="Destination PNG path.")
    args = parser.parse_args()
    output = generate_visualization(args.scenario, args.result, args.output)
    print("SCHEDULE_VISUALIZATION_COMPLETE")
    print(f"output={output}")
    print("resolution=1920x1080")


if __name__ == "__main__":
    main()
