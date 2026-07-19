"""Generate evidence-backed figures for docs/技术报告.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures"


def load(relative_path: str) -> dict:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.titleweight"] = "bold"


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.03, y + h * 0.72, title, fontsize=11, fontweight="bold", color=color, va="center")
    ax.text(x + 0.03, y + h * 0.30, body, fontsize=8.5, color="#333333", va="center", linespacing=1.35)


def system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layers = [
        (0.76, "证据与交付层", "JSON / 日志 / 审计 / Word 报告 / openEuler", "#7A3E9D"),
        (0.58, "评测与基线层", "固定场景 · HEFT · MILP · Hybrid · 配对统计", "#B34A3C"),
        (0.40, "策略与搜索层", "Residual Policy · Best-of-N · 合法重定位 · LNS", "#1F6F8B"),
        (0.22, "训练层", "MaskablePPO · 配置驱动 · 归一化 · checkpoint", "#356A3B"),
        (0.04, "环境与调度物理层", "DAG · 资源 · action mask · EFT 插入式时间线", "#8A6D1D"),
    ]
    for y, title, body, color in layers:
        box(ax, 0.08, y, 0.84, 0.13, title, body, color)
    for y in (0.745, 0.565, 0.385, 0.205):
        ax.annotate("", xy=(0.50, y), xytext=(0.50, y + 0.03), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.set_title("统一调度接口贯穿训练、评测与证据链", fontsize=15, pad=12)
    save(fig, "system_architecture.png")


def method_evolution() -> None:
    sources = [
        ("初版 Joint", "evaluation/results/summary.json"),
        ("相对奖励", "evaluation/results/summary_reward_shaped.json"),
        ("观测归一化", "evaluation/results/summary_mlp_normalized.json"),
        ("Ranked", "evaluation/results/summary_mlp_ranked.json"),
        ("Residual", "evaluation/results/summary_mlp_residual.json"),
        ("Best-of-64", "evaluation/results/summary_mlp_residual_bestof64.json"),
    ]
    labels = [name for name, _ in sources] + ["+重定位+LNS"]
    values = [float(load(path)["overall"]["mean_ratio"]) for _, path in sources]
    lns = load("evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json")
    values.append(float(lns["statistics"]["lns_mean_ratio"]["mean"]))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ["#9A9A9A", "#9A9A9A", "#527C9C", "#2F7D5A", "#76549A", "#D2813D", "#B33A3A"]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.68)
    ax.axhline(1.0, color="#202020", linewidth=1.2, linestyle="--", label="HEFT = 1.0")
    ax.set_ylim(0.82, 1.44)
    ax.set_ylabel("mean_ratio（越低越好）")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8.5)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("从策略塌缩到约束搜索精修：每次改进解决一个可诊断问题")
    save(fig, "method_evolution.png")


def final_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ("HEFT rank", "安全先验"),
        ("Residual", "学习有界 delta"),
        ("Best-of-64", "构造多样初解"),
        ("合法重定位", "小邻域精修"),
        ("Best-only LNS", "破坏—修复"),
        ("EFT 调度", "输出资源时间线"),
    ]
    palette = ["#7A6C3A", "#66518E", "#BD7531", "#347886", "#A63B3B", "#356A3B"]
    width = 0.135
    gap = 0.025
    x = 0.025
    for index, ((title, body), color) in enumerate(zip(items, palette)):
        box(ax, x, 0.36, width, 0.28, title, body, color)
        if index < len(items) - 1:
            ax.annotate("", xy=(x + width + gap * 0.78, 0.50), xytext=(x + width + 0.003, 0.50), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
        x += width + gap
    ax.text(0.5, 0.18, "只接受严格改善候选：同一初解上 makespan 单调不增", ha="center", fontsize=10.5, color="#8A2424", fontweight="bold")
    ax.set_title("最终方案：学习构造器 + 约束搜索精修器", fontsize=15, pad=10)
    save(fig, "final_pipeline.png")


def final_statistics() -> None:
    direct = load("evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json")
    compute = load("evaluation/results/autonomous_exploration/compute_matched_sampling/paired_comparison_summary.json")
    names = ["Residual\nBest-of-64", "Residual\nBest-of-128", "Best-of-64\n+ LNS"]
    means = [
        direct["statistics"]["residual_bestof64_mean_ratio"]["mean"],
        compute["statistics"]["pure_sampling_mean_ratio"]["mean"],
        direct["statistics"]["lns_mean_ratio"]["mean"],
    ]
    stds = [
        direct["statistics"]["residual_bestof64_mean_ratio"]["sample_std"],
        compute["statistics"]["pure_sampling_mean_ratio"]["sample_std"],
        direct["statistics"]["lns_mean_ratio"]["sample_std"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=["#7A6A9B", "#C07A34", "#A93434"], width=0.62)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("5 次 mean_ratio（均值 ± 样本标准差）")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.6f}", ha="center", fontsize=9)
    ax.set_title("最终 LNS 改进不由单纯增加采样次数解释")
    save(fig, "final_statistics.png")


def generalization() -> None:
    data = load("evaluation/results/structural_generalization/summary_structural_generalization.json")["groups"]
    keys = ["wide_parallel", "deep_chain", "homogeneous_resources", "original_control"]
    labels = ["宽并行", "深链条", "同构资源", "原始对照"]
    values = [data[key]["mean_ratio"] for key in keys]
    stds = [data[key]["std_ratio"] for key in keys]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, values, yerr=stds, capsize=5, color=["#B54B4B", "#3E789C", "#3F7C55", "#777777"])
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_ylim(0.75, 1.08)
    ax.set_ylabel("mean_ratio")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_title("结构化泛化：宽并行 DAG 是当前明显短板")
    save(fig, "generalization.png")


def main() -> None:
    configure()
    system_architecture()
    method_evolution()
    final_pipeline()
    final_statistics()
    generalization()
    for path in sorted(OUTPUT.glob("*.png")):
        print(f"generated: {path.relative_to(ROOT).as_posix()} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
