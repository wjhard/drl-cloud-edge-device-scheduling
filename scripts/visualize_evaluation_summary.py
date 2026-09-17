"""Render evaluation JSON evidence as a presentation-ready 1920x1080 summary."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


@dataclass(frozen=True)
class EvaluationSummary:
    repeat_labels: list[str]
    final_ratios: list[float]
    baseline_ratios: list[float]
    repeat_win_counts: list[int]
    repeat_count: int
    final_mean: float
    final_std: float
    final_p_value: float
    final_significant: bool | None
    wide_mean: float
    wide_scenarios: int
    wide_non_regression: int
    wide_strict_wins: int
    wide_optimal_ties: int
    large_mean: float
    large_std: float
    large_scenarios: int
    large_strict_wins: int
    large_size_means: dict[int, float]
    large_skipped: int | None
    residual_capacity: int | None
    residual_skipped_sizes: list[int]
    large_source_counts: dict[str, int]
    os_matrix_status: str
    os_jobs: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _require(mapping: dict[str, Any], key: str, source: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in {source}.")
    return mapping[key]


def _capacity_from_scenarios(scenarios: list[dict[str, Any]]) -> int | None:
    capacities: set[int] = set()
    for scenario in scenarios:
        reason = scenario.get("diagnostics", {}).get("residual_skipped_reason")
        if not reason:
            continue
        match = re.search(r"residual_model_capacity=(\d+)", str(reason))
        if match:
            capacities.add(int(match.group(1)))
    return next(iter(capacities)) if len(capacities) == 1 else None


def load_evaluation_summary(
    final_path: Path,
    wide_path: Path,
    large_path: Path,
    os_matrix_path: Path,
) -> EvaluationSummary:
    final = _load_json(final_path)
    wide = _load_json(wide_path)
    large = _load_json(large_path)
    os_matrix = _load_json(os_matrix_path)

    paired_runs = list(_require(final, "paired_runs", final_path.name))
    statistics = _require(final, "statistics", final_path.name)
    final_stats = _require(statistics, "lns_mean_ratio", final_path.name)
    paired_test = _require(statistics, "paired_t_test_two_sided", final_path.name)
    repeat_labels = [f"R{run.get('repeat', index + 1)}" for index, run in enumerate(paired_runs)]

    wide_group = _require(_require(wide, "groups", wide_path.name), "adaptive", wide_path.name)
    large_overall = _require(large, "overall", large_path.name)
    large_scenarios = list(large.get("scenarios", []))
    ratios_by_size: dict[int, list[float]] = defaultdict(list)
    skipped_sizes: set[int] = set()
    for scenario in large_scenarios:
        if "num_tasks" in scenario and "ratio_to_heft" in scenario:
            ratios_by_size[int(scenario["num_tasks"])].append(float(scenario["ratio_to_heft"]))
        if scenario.get("diagnostics", {}).get("residual_skipped_reason") and "num_tasks" in scenario:
            skipped_sizes.add(int(scenario["num_tasks"]))
    size_means = {
        task_size: sum(values) / len(values)
        for task_size, values in sorted(ratios_by_size.items())
        if values
    }

    jobs = [dict(job) for job in os_matrix.get("jobs", [])]
    return EvaluationSummary(
        repeat_labels=repeat_labels,
        final_ratios=[float(run["lns_mean_ratio"]) for run in paired_runs],
        baseline_ratios=[float(run["residual_bestof64_mean_ratio"]) for run in paired_runs],
        repeat_win_counts=[int(run["lns_better_than_heft_count"]) for run in paired_runs],
        repeat_count=int(final.get("repeat_count", len(paired_runs))),
        final_mean=float(_require(final_stats, "mean", final_path.name)),
        final_std=float(_require(final_stats, "sample_std", final_path.name)),
        final_p_value=float(_require(paired_test, "p_value", final_path.name)),
        final_significant=(
            bool(paired_test["significant"]) if "significant" in paired_test else None
        ),
        wide_mean=float(_require(wide_group, "mean_ratio", wide_path.name)),
        wide_scenarios=int(_require(wide_group, "scenario_count", wide_path.name)),
        wide_non_regression=int(_require(wide_group, "non_regression_vs_heft_count", wide_path.name)),
        wide_strict_wins=int(_require(wide_group, "outperform_heft_count", wide_path.name)),
        wide_optimal_ties=int(wide_group.get("heft_tie_proven_optimal_count", 0)),
        large_mean=float(_require(large_overall, "mean_ratio", large_path.name)),
        large_std=float(_require(large_overall, "sample_std_ratio", large_path.name)),
        large_scenarios=int(_require(large_overall, "scenario_count", large_path.name)),
        large_strict_wins=int(
            _require(large_overall, "strictly_better_than_heft_count", large_path.name)
        ),
        large_size_means=size_means,
        large_skipped=(
            int(large_overall["residual_skipped_count"])
            if "residual_skipped_count" in large_overall
            else None
        ),
        residual_capacity=_capacity_from_scenarios(large_scenarios),
        residual_skipped_sizes=sorted(skipped_sizes),
        large_source_counts={
            str(key): int(value)
            for key, value in large_overall.get("selected_source_counts", {}).items()
        },
        os_matrix_status=str(os_matrix.get("status", "unknown")),
        os_jobs=jobs,
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
            "ytick.color": "#475467",
        }
    )


def _card(ax: plt.Axes, x: float, width: float, label: str, value: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, 0.08),
        width,
        0.84,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, 0.67, label, fontsize=10.5, color="#667085", va="center")
    ax.text(
        x + 0.018,
        0.34,
        value,
        fontsize=14,
        color=color,
        fontweight="bold",
        va="center",
        linespacing=1.12,
    )


def _draw_kpis(ax: plt.Axes, data: EvaluationSummary) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    win_text = " · ".join(str(value) for value in data.repeat_win_counts)
    significance = "显著" if data.final_significant is True else "未标注显著性"
    cards = [
        ("最终 5 次统计", f"{data.final_mean:.6f} ± {data.final_std:.6f}", "#6A4BBC"),
        ("独立重复 / 单轮反超场景数", f"{data.repeat_count} 次　{win_text}", "#1677A8"),
        ("双侧配对检验", f"p = {data.final_p_value:.6e}　{significance}", "#A43D54"),
    ]
    gap = 0.016
    width = (1.0 - gap * (len(cards) - 1)) / len(cards)
    for index, (label, value, color) in enumerate(cards):
        _card(ax, index * (width + gap), width, label, value, color)


def _draw_repeats(ax: plt.Axes, data: EvaluationSummary) -> None:
    x_values = list(range(len(data.final_ratios)))
    ax.bar(
        x_values,
        data.final_ratios,
        width=0.58,
        color="#7353B6",
        alpha=0.90,
        label="最终 Adaptive + LNS",
        zorder=2,
    )
    ax.scatter(
        x_values,
        data.baseline_ratios,
        marker="D",
        s=62,
        color="#E08A2E",
        edgecolor="white",
        linewidth=1.2,
        label="Residual Best-of-64",
        zorder=4,
    )
    ax.axhline(1.0, color="#344054", linestyle="--", linewidth=1.4, label="HEFT = 1.0")
    values = data.final_ratios + data.baseline_ratios
    lower = min(values) - max(0.006, (max(values) - min(values)) * 0.25)
    ax.set_ylim(max(0.0, lower), 1.006)
    ax.set_xticks(x_values, data.repeat_labels)
    ax.set_ylabel("mean_ratio（越低越好）")
    ax.set_title("五次独立重复：每次都由同一配对场景评测", fontsize=15, fontweight="bold", loc="left", pad=13)
    ax.grid(axis="y", color="#D0D5DD", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for x_value, ratio in zip(x_values, data.final_ratios):
        ax.text(x_value, ratio + 0.0012, f"{ratio:.6f}", ha="center", va="bottom", fontsize=8.7, color="#503990")


def _draw_wide(ax: plt.Axes, data: EvaluationSummary) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    patch = FancyBboxPatch(
        (0.01, 0.03),
        0.98,
        0.94,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor="#FFFFFF",
        edgecolor="#1677A8",
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(0.07, 0.82, "宽并行 DAG 专项", fontsize=14, fontweight="bold", color="#145B7D")
    ax.text(0.07, 0.62, f"mean_ratio = {data.wide_mean:.6f}", fontsize=13, fontweight="bold")
    ax.text(
        0.07,
        0.40,
        f"不劣于 HEFT　{data.wide_non_regression}/{data.wide_scenarios}\n"
        f"严格反超　　{data.wide_strict_wins}/{data.wide_scenarios}",
        fontsize=11.5,
        linespacing=1.45,
        color="#344054",
    )
    if data.wide_optimal_ties:
        ax.text(
            0.07,
            0.14,
            f"MILP 证明最优平局：{data.wide_optimal_ties}",
            fontsize=10.5,
            color="#A15C00",
            fontweight="bold",
        )


def _draw_large_scale(ax: plt.Axes, data: EvaluationSummary) -> None:
    sizes = sorted(data.large_size_means)
    ratios = [data.large_size_means[size] for size in sizes]
    if sizes:
        ax.plot(sizes, ratios, color="#21845A", marker="o", markersize=8, linewidth=2.4)
        for size, ratio in zip(sizes, ratios):
            ax.text(size, ratio + 0.004, f"{ratio:.3f}", ha="center", fontsize=9, color="#14633F")
        ax.set_xticks(sizes)
    ax.axhline(1.0, color="#344054", linestyle="--", linewidth=1.3)
    lower = min(ratios, default=data.large_mean) - 0.02
    ax.set_ylim(max(0.0, lower), 1.01)
    ax.set_xlabel("任务规模")
    ax.set_ylabel("mean_ratio")
    ax.set_title(
        f"30–60 任务：{data.large_mean:.6f} ± {data.large_std:.6f}\n"
        f"严格反超 {data.large_strict_wins}/{data.large_scenarios}",
        fontsize=13.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.grid(color="#D0D5DD", linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_os_matrix(ax: plt.Axes, data: EvaluationSummary) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    status_color = "#137A55" if data.os_matrix_status == "passed" else "#B42318"
    ax.text(
        0.0,
        0.92,
        f"国产操作系统 Docker 验证矩阵　overall={data.os_matrix_status}",
        fontsize=15,
        fontweight="bold",
        color=status_color,
    )
    if not data.os_jobs:
        ax.text(0.0, 0.62, "JSON 中没有可展示的系统任务记录", fontsize=11, color="#667085")
        return
    gap = 0.018
    width = (1.0 - gap * (len(data.os_jobs) - 1)) / len(data.os_jobs)
    for index, job in enumerate(data.os_jobs):
        x = index * (width + gap)
        passed = str(job.get("status", "unknown")) == "passed"
        color = "#16805A" if passed else "#B42318"
        box = FancyBboxPatch(
            (x, 0.12),
            width,
            0.65,
            boxstyle="round,pad=0.010,rounding_size=0.025",
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x + 0.025, 0.61, str(job.get("os", "unknown")), fontsize=11.5, fontweight="bold", color="#172033")
        ax.text(x + 0.025, 0.39, str(job.get("status", "unknown")).upper(), fontsize=13, fontweight="bold", color=color)
        ax.text(
            x + 0.025,
            0.20,
            f"build={job.get('build_exit_code', 'N/A')}　run={job.get('run_exit_code', 'N/A')}",
            fontsize=9.5,
            color="#667085",
        )


def _draw_boundary(ax: plt.Axes, data: EvaluationSummary) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.01, 0.05),
        0.98,
        0.90,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor="#FFF9ED",
        edgecolor="#D08A1D",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(0.07, 0.80, "诚实边界（来自大规模 JSON）", fontsize=14, fontweight="bold", color="#955B00")
    lines: list[str] = []
    if data.residual_capacity is not None:
        lines.append(f"Residual 模型容量：{data.residual_capacity} tasks")
    if data.large_skipped is not None:
        lines.append(f"跳过 Residual：{data.large_skipped}/{data.large_scenarios}")
    if data.residual_skipped_sizes:
        lines.append("发生规模：" + " / ".join(str(size) for size in data.residual_skipped_sizes))
    if data.large_source_counts:
        source_text = "，".join(f"{name}={count}" for name, count in sorted(data.large_source_counts.items()))
        lines.append("最终来源：" + source_text)
    if not lines:
        lines.append("输入 JSON 未提供容量或降级字段")
    ax.text(0.07, 0.61, "\n".join(lines), fontsize=10.8, color="#5E4B2C", va="top", linespacing=1.50)


def _draw_repeat_details(ax: plt.Axes, data: EvaluationSummary) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.0, 0.02),
        1.0,
        0.96,
        boxstyle="round,pad=0.010,rounding_size=0.025",
        facecolor="#FFFFFF",
        edgecolor="#D0D5DD",
        linewidth=1.2,
    )
    ax.add_patch(box)
    row_labels = ["重复", "Residual Best-of-64", "最终方案", "反超 HEFT 场景数"]
    row_y = [0.77, 0.57, 0.37, 0.17]
    for label, y in zip(row_labels, row_y):
        ax.text(0.045, y, label, fontsize=10.3, fontweight="bold", color="#475467", va="center")
    column_start = 0.30
    column_step = 0.135
    for index, (repeat_label, baseline, final_ratio, wins) in enumerate(
        zip(data.repeat_labels, data.baseline_ratios, data.final_ratios, data.repeat_win_counts)
    ):
        x = column_start + index * column_step
        values = [repeat_label, f"{baseline:.6f}", f"{final_ratio:.6f}", str(wins)]
        for value, y in zip(values, row_y):
            ax.text(x, y, value, fontsize=10.3, color="#172033", va="center", ha="center")


def _save_figure(figure: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path


def _render_final(data: EvaluationSummary, output_path: Path) -> Path:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=150)
    figure.text(0.05, 0.965, "最终方案：五次独立重复与统计显著性", fontsize=23, fontweight="bold", va="top")
    figure.text(
        0.05,
        0.895,
        "柱形为最终方案，菱形为配对的 Residual Best-of-64；虚线为 HEFT 基准",
        fontsize=11,
        color="#667085",
        va="top",
    )
    kpi_ax = figure.add_axes([0.05, 0.745, 0.91, 0.115])
    repeats_ax = figure.add_axes([0.075, 0.335, 0.85, 0.34])
    details_ax = figure.add_axes([0.075, 0.065, 0.85, 0.205])
    _draw_kpis(kpi_ax, data)
    _draw_repeats(repeats_ax, data)
    _draw_repeat_details(details_ax, data)
    return _save_figure(figure, output_path)


def _render_generalization(data: EvaluationSummary, output_path: Path) -> Path:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=150)
    figure.text(0.05, 0.965, "泛化、国产操作系统与模型容量边界", fontsize=23, fontweight="bold", va="top")
    figure.text(
        0.05,
        0.895,
        "全部数字来自宽并行、大规模及 Docker 矩阵 JSON；缺失字段不会被补写",
        fontsize=11,
        color="#667085",
        va="top",
    )
    wide_ax = figure.add_axes([0.055, 0.49, 0.27, 0.32])
    large_ax = figure.add_axes([0.39, 0.49, 0.57, 0.32])
    os_ax = figure.add_axes([0.06, 0.09, 0.62, 0.30])
    boundary_ax = figure.add_axes([0.72, 0.09, 0.24, 0.30])
    _draw_wide(wide_ax, data)
    _draw_large_scale(large_ax, data)
    _draw_os_matrix(os_ax, data)
    _draw_boundary(boundary_ax, data)
    return _save_figure(figure, output_path)


def _variant_path(output_path: Path, view: str) -> Path:
    if output_path.suffix.lower() == ".png":
        return output_path.with_name(f"{output_path.stem}_{view}.png")
    return output_path / f"evaluation_summary_{view}.png"


def generate_visualizations(
    final_path: Path,
    wide_path: Path,
    large_path: Path,
    os_matrix_path: Path,
    output_path: Path,
    view: str = "all",
) -> list[Path]:
    if view not in {"final", "generalization", "all"}:
        raise ValueError(f"Unsupported view: {view}")
    data = load_evaluation_summary(final_path, wide_path, large_path, os_matrix_path)
    _configure_style()
    generated: list[Path] = []
    if view in {"final", "all"}:
        final_output = output_path if view == "final" else _variant_path(output_path, "final")
        generated.append(_render_final(data, final_output))
    if view in {"generalization", "all"}:
        generalization_output = (
            output_path if view == "generalization" else _variant_path(output_path, "generalization")
        )
        generated.append(_render_generalization(data, generalization_output))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Render four evaluation JSON files as a 1920x1080 evidence summary.")
    parser.add_argument("--final", required=True, type=Path, help="Five-repeat final summary JSON.")
    parser.add_argument("--wide", required=True, type=Path, help="Wide-parallel evaluation JSON.")
    parser.add_argument("--large", required=True, type=Path, help="Large-scale evaluation JSON.")
    parser.add_argument("--os-matrix", required=True, type=Path, help="Domestic-OS Docker matrix JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Destination PNG path or base path for --view all.")
    parser.add_argument("--view", choices=("final", "generalization", "all"), default="all")
    args = parser.parse_args()
    outputs = generate_visualizations(
        args.final,
        args.wide,
        args.large,
        args.os_matrix,
        args.output,
        view=args.view,
    )
    print("EVALUATION_VISUALIZATION_COMPLETE")
    for output in outputs:
        print(f"output={output} resolution=1920x1080")


if __name__ == "__main__":
    main()
