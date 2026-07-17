"""Audit evidence paths and numeric claims in docs/技术报告.md.

The audit is intentionally declarative: every result-bearing table/paragraph has
an anchor and a JSON-backed extractor. Concrete repository-relative paths must
exist and be tracked by Git. Selected log-backed claims are checked against their
original text, and tracked text files are scanned for private user-home paths.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "技术报告.md"
JSON_PATH_RE = re.compile(r"evaluation/results/[A-Za-z0-9_./*+-]+\.json")
REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_:/.-])"
    r"((?:[A-Za-z0-9_.+-]+/)+[A-Za-z0-9_.*+-]+\."
    r"(?:json|log|tsv|yaml|yml|py|sh|ps1|zip|npz|csv|png|docx|pdf|sol|txt))"
)
PRIVATE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]Users[\\/][^\\/\s]+|/" + r"home/[^/\s]+)",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".json", ".log", ".md", ".py", ".ps1", ".sh", ".sol", ".tsv", ".txt", ".yaml", ".yml"}
FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Check:
    label: str
    anchor: str
    pattern: str
    source: str
    actual: Callable[[dict], float]
    group: int = 1
    tolerance: float = TOLERANCE
    section: str | None = None


@dataclass(frozen=True)
class TextEvidenceCheck:
    label: str
    report_pattern: str
    source: str
    source_patterns: tuple[str, ...]


def tracked_files() -> set[str]:
    output = subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
        cwd=ROOT,
    )
    return {item.decode("utf-8").replace("\\", "/") for item in output.split(b"\0") if item}


def read_text_file(path: Path) -> str:
    """Read repository evidence while preserving legacy UTF-16 logs."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-16", errors="replace")


def load_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_mean(data: dict) -> float:
    return float(data["overall"]["mean_ratio"])


def better_count(data: dict) -> float:
    return float(sum(float(row["ratio"]) < 1.0 for row in data["scenarios"]))


def dotted(path: str) -> Callable[[dict], float]:
    keys = path.split(".")

    def extract(data: dict) -> float:
        value = data
        for key in keys:
            value = value[key]
        return float(value)

    return extract


def milp_optimal_rows(data: dict) -> list[dict]:
    return [row for row in data["scenarios"] if row["milp_proven_optimal"]]


def milp_mean(field: str) -> Callable[[dict], float]:
    return lambda data: mean(float(row[field]) for row in milp_optimal_rows(data))


def milp_gap_percent(field: str) -> Callable[[dict], float]:
    return lambda data: (milp_mean(field)(data) - 1.0) * 100.0


def row_check(
    checks: list[Check],
    label: str,
    anchor: str,
    source: str,
    column: int,
    actual: Callable[[dict], float],
    section: str | None = None,
) -> None:
    # Markdown rows start and end with a pipe; column 0 is the first real cell.
    pattern = rf"^(?:[^|]*\|){{{column + 1}}}\s*\**\s*({FLOAT_RE})"
    checks.append(Check(label, anchor, pattern, source, actual, section=section))


def build_checks() -> list[Check]:
    checks: list[Check] = []

    # Result overview.
    paired = "evaluation/results/paired15/paired_comparison_summary.json"
    repeated = "evaluation/results/repeated_bestof64/repeated_bestof64_summary.json"
    fixed = "evaluation/results/summary_mlp_residual_bestof64.json"
    milp = "evaluation/results/milp_optimal_comparison.json"
    row_check(checks, "overview paired mean", "最终配对实验中的 Residual", paired, 2,
              dotted("statistics.bestof64_mean_ratio.mean"))
    row_check(checks, "overview paired std", "最终配对实验中的 Residual", paired, 2,
              dotted("statistics.bestof64_mean_ratio.sample_std"))
    checks[-1] = Check(checks[-1].label, checks[-1].anchor,
                       rf"({FLOAT_RE})\s*±\s*\**({FLOAT_RE})", paired,
                       dotted("statistics.bestof64_mean_ratio.sample_std"), group=2)
    row_check(checks, "overview paired wins", "最终配对实验中的 Residual", paired, 3,
              dotted("statistics.bestof64_better_than_heft_count.mean"))
    row_check(checks, "overview repeated mean", "独立 Best-of-64 稳定性实验", repeated, 2,
              dotted("statistics.mean_ratio.mean"))
    checks[-1] = Check(checks[-1].label, checks[-1].anchor,
                       rf"({FLOAT_RE})\s*±\s*\**({FLOAT_RE})", repeated,
                       dotted("statistics.mean_ratio.mean"), group=1)
    checks.append(Check("overview repeated std", "独立 Best-of-64 稳定性实验",
                        rf"({FLOAT_RE})\s*±\s*\**({FLOAT_RE})", repeated,
                        dotted("statistics.mean_ratio.sample_std"), group=2))
    row_check(checks, "overview repeated wins", "独立 Best-of-64 稳定性实验", repeated, 3,
              dotted("statistics.better_than_heft_count.mean"))
    row_check(checks, "overview fixed mean", "固定采样种子 20260709", fixed, 2, summary_mean)
    row_check(checks, "overview fixed wins", "固定采样种子 20260709", fixed, 3, better_count)

    # Abstract prose repeats several values from the overview tables.
    abstract_pattern = (
        rf"为 \*\*({FLOAT_RE}) ± ({FLOAT_RE})\*\*，平均每轮在 \*\*({FLOAT_RE})/20\*\*"
        rf".*?差值为 ({FLOAT_RE}) ± ({FLOAT_RE}).*?`t=({FLOAT_RE})`、`p=({FLOAT_RE})×10\^(-?\d+)`"
        rf".*?得到 \*\*({FLOAT_RE}) ± ({FLOAT_RE})\*\* 和平均 \*\*({FLOAT_RE})/20\*\*"
        rf".*?在 ({FLOAT_RE}) 个 CBC.*?HEFT 平均高于最优解 ({FLOAT_RE})%，"
        rf"Residual\+Best-of-16 仅高于最优解 ({FLOAT_RE})%"
    )
    abstract_values = [
        ("abstract paired mean", 1, paired, dotted("statistics.bestof64_mean_ratio.mean")),
        ("abstract paired std", 2, paired, dotted("statistics.bestof64_mean_ratio.sample_std")),
        ("abstract paired wins", 3, paired, dotted("statistics.bestof64_better_than_heft_count.mean")),
        ("abstract paired diff mean", 4, paired, dotted("statistics.paired_difference.mean")),
        ("abstract paired diff std", 5, paired, dotted("statistics.paired_difference.sample_std")),
        ("abstract t", 6, paired, dotted("statistics.paired_t_test_two_sided.t_statistic")),
        ("abstract p mantissa", 7, paired,
         lambda data: float(data["statistics"]["paired_t_test_two_sided"]["p_value"]) * 1e6),
        ("abstract repeated mean", 9, repeated, dotted("statistics.mean_ratio.mean")),
        ("abstract repeated std", 10, repeated, dotted("statistics.mean_ratio.sample_std")),
        ("abstract repeated wins", 11, repeated, dotted("statistics.better_than_heft_count.mean")),
        ("abstract MILP optimal count", 12, milp, dotted("totals.proven_optimal_count")),
        ("abstract MILP HEFT gap", 13, milp, milp_gap_percent("heft_over_milp_optimal")),
        ("abstract MILP residual gap", 14, milp,
         milp_gap_percent("residual_bestof16_over_milp_optimal")),
    ]
    for label, group, source, extractor in abstract_values:
        checks.append(Check(label, "严格按项目现存结果文件统计", abstract_pattern,
                            source, extractor, group=group))

    # Narrative claims outside tables.
    checks.extend([
        Check("initial prose mean", "初版正式评测", rf"mean_ratio=({FLOAT_RE})",
              "evaluation/results/summary.json", summary_mean),
        Check("initial prose wins", "初版正式评测", rf"，({FLOAT_RE}) 个场景均未反超",
              "evaluation/results/summary.json", lambda data: float(len(data["scenarios"]))),
        Check("reward prose mean", "实际值仍为", rf"实际值仍为 `({FLOAT_RE})`",
              "evaluation/results/summary_reward_shaped.json", summary_mean),
        Check("normalized prose mean", "正式加入 `ObservationNormalizer`", rf"降至 \*\*({FLOAT_RE})\*\*",
              "evaluation/results/summary_mlp_normalized.json", summary_mean),
        Check("ranked prose mean", "重构后的 `SchedulingEnvRanked`", rf"取得 `mean_ratio=({FLOAT_RE})`",
              "evaluation/results/summary_mlp_ranked.json", summary_mean),
        Check("residual prose mean", "训练 200k 后", rf"达到 `mean_ratio=({FLOAT_RE})`",
              "evaluation/results/summary_mlp_residual.json", summary_mean),
        Check("residual prose wins", "训练 200k 后", rf"已有 ({FLOAT_RE})/20",
              "evaluation/results/summary_mlp_residual.json", better_count),
        Check("repeat prose time mean", "5 次 Best-of-64 独立评测平均耗时",
              rf"为 \*\*({FLOAT_RE}) ± ({FLOAT_RE}) 秒", repeated,
              lambda data: mean(r["elapsed_seconds"] for r in data["runs"]), group=1),
        Check("repeat prose time std", "5 次 Best-of-64 独立评测平均耗时",
              rf"为 \*\*({FLOAT_RE}) ± ({FLOAT_RE}) 秒", repeated,
              lambda data: math.sqrt(sum((r["elapsed_seconds"] - mean(x["elapsed_seconds"] for x in data["runs"])) ** 2 for r in data["runs"]) / (len(data["runs"]) - 1)),
              group=2),
    ])

    # Appendix B repeats both paired and independent-run statistics.
    appendix_pattern = (
        rf"实际 `({FLOAT_RE})±({FLOAT_RE})` 和 ({FLOAT_RE})/20.*?"
        rf"结果是 `({FLOAT_RE})±({FLOAT_RE})`、({FLOAT_RE})/20"
    )
    appendix_values = [
        ("appendix paired mean", 1, paired, dotted("statistics.bestof64_mean_ratio.mean")),
        ("appendix paired std", 2, paired, dotted("statistics.bestof64_mean_ratio.sample_std")),
        ("appendix paired wins", 3, paired, dotted("statistics.bestof64_better_than_heft_count.mean")),
        ("appendix repeated mean", 4, repeated, dotted("statistics.mean_ratio.mean")),
        ("appendix repeated std", 5, repeated, dotted("statistics.mean_ratio.sample_std")),
        ("appendix repeated wins", 6, repeated, dotted("statistics.better_than_heft_count.mean")),
    ]
    for label, group, source, extractor in appendix_values:
        checks.append(Check(label, "最终重复次数口径", appendix_pattern,
                            source, extractor, group=group))

    # Main experiment and ablation tables. Each tuple is anchor, JSON, mean column, count column.
    rows = [
        ("`ppo_gat_reward_shaped.yaml`", "evaluation/results/summary_gat_reward_shaped.json", 2, None),
        ("`ppo_gat_reward_shaped_lowent.yaml`", "evaluation/results/summary_gat_lowent.json", 2, None),
        ("| MLP ranked 基线 |", "evaluation/results/summary_mlp_ranked.json", 2, 3),
        ("| ranked + BC |", "evaluation/results/summary_mlp_ranked_bc.json", 2, 3),
        ("| ranked + upward rank", "evaluation/results/summary_mlp_ranked_with_rank.json", 2, 3),
        ("| ranked + 课程学习", "evaluation/results/summary_mlp_ranked_curriculum.json", 2, 3),
        ("| ranked，lr=0.001", "evaluation/results/summary_mlp_ranked_lr1e-3_128.json", 2, 3),
        ("| 上述最优超参数延长训练", "evaluation/results/summary_mlp_ranked_lr1e-3_128_400k.json", 2, 3),
        ("| 3 种子 × Best-of-8", "evaluation/results/summary_mlp_ranked_ensemble3_bestof8.json", 2, 3),
        ("| BC + rank 特征", "evaluation/results/summary_mlp_ranked_all_combined.json", 2, 3),
        ("| 初版 MLP+raw reward", "evaluation/results/summary.json", 2, 3),
        ("| relative_heft 奖励", "evaluation/results/summary_reward_shaped.json", 2, 3),
        ("| GAT，ent=0.03", "evaluation/results/summary_gat_reward_shaped.json", 2, 3),
        ("| GAT，ent=0.01", "evaluation/results/summary_gat_lowent.json", 2, 3),
        ("| MLP + 观测归一化", "evaluation/results/summary_mlp_normalized.json", 2, 3),
        ("| MLP ranked |", "evaluation/results/summary_mlp_ranked.json", 2, 3),
        ("| MLP ranked，调优 lr", "evaluation/results/summary_mlp_ranked_lr1e-3_128.json", 2, 3),
        ("| GAT ranked |", "evaluation/results/summary_gat_ranked.json", 2, 3),
        ("| 3 种子集成，每模型", "evaluation/results/summary_mlp_ranked_ensemble3_bestof8.json", 2, 3),
        ("| Residual deterministic", "evaluation/results/summary_mlp_residual.json", 2, 3),
        ("| Residual Best-of-16", "evaluation/results/summary_mlp_residual_bestof16.json", 2, 3),
        ("| Residual Best-of-64（固定种子）", fixed, 2, 3),
    ]
    for anchor, source, mean_col, count_col in rows:
        row_check(checks, f"{anchor} mean", anchor, source, mean_col, summary_mean)
        if count_col is not None:
            row_check(checks, f"{anchor} wins", anchor, source, count_col, better_count)

    # Residual Best-of-N scan.
    for n in (1, 4, 8, 16, 32, 64):
        anchor = "| 1（deterministic）" if n == 1 else f"| {n} |"
        source = f"evaluation/results/summary_mlp_residual_bestof{n}.json"
        row_check(checks, f"Residual best-of-{n} mean", anchor, source, 1, summary_mean,
                  section="## 3.8 阶段八：Best-of-N 推理采样")
        row_check(checks, f"Residual best-of-{n} wins", anchor, source, 2, better_count,
                  section="## 3.8 阶段八：Best-of-N 推理采样")

    # Repeated Best-of-64 table.
    for run in range(1, 6):
        anchor = f"| {run} |"
        checks.append(Check(f"repeat {run} mean", anchor,
                            rf"^\|\s*{run}\s*\|\s*({FLOAT_RE})", repeated,
                            lambda data, idx=run - 1: float(data["runs"][idx]["mean_ratio"]),
                            section="## 4.2 统计稳健性"))
        checks.append(Check(f"repeat {run} wins", anchor,
                            rf"^\|\s*{run}\s*\|\s*{FLOAT_RE}\s*\|\s*({FLOAT_RE})/20", repeated,
                            lambda data, idx=run - 1: float(data["runs"][idx]["better_than_heft_count"]),
                            section="## 4.2 统计稳健性"))
        checks.append(Check(f"repeat {run} time", anchor,
                            rf"^\|\s*{run}\s*\|(?:[^|]*\|){{2}}\s*({FLOAT_RE})", repeated,
                            lambda data, idx=run - 1: float(data["runs"][idx]["elapsed_seconds"]),
                            tolerance=5e-4, section="## 4.2 统计稳健性"))
    repeat_stats = [
        ("repeat mean", 1, dotted("statistics.mean_ratio.mean")),
        ("repeat std", 2, dotted("statistics.mean_ratio.sample_std")),
        ("repeat win mean", 3, dotted("statistics.better_than_heft_count.mean")),
        ("repeat win std", 4, dotted("statistics.better_than_heft_count.sample_std")),
    ]
    repeat_pattern = (
        rf"({FLOAT_RE})\s*±\s*({FLOAT_RE}).*?"
        rf"({FLOAT_RE})\s*±\s*({FLOAT_RE}).*?"
        rf"({FLOAT_RE})\s*±\s*({FLOAT_RE})"
    )
    for label, group, extractor in repeat_stats:
        checks.append(Check(label, "| 均值 ± 样本标准差 |", repeat_pattern,
                            repeated, extractor, group=group))
    checks.append(Check("repeat time mean", "| 均值 ± 样本标准差 |", repeat_pattern,
                        repeated, lambda data: mean(r["elapsed_seconds"] for r in data["runs"]),
                        group=5, tolerance=5e-4))
    checks.append(Check("repeat time std", "| 均值 ± 样本标准差 |", repeat_pattern,
                        repeated,
                        lambda data: math.sqrt(sum((r["elapsed_seconds"] - mean(x["elapsed_seconds"] for x in data["runs"])) ** 2 for r in data["runs"]) / (len(data["runs"]) - 1)),
                        group=6, tolerance=5e-4))

    # Paired statistics paragraph.
    paired_specs = [
        ("Best-of-64：", rf"`({FLOAT_RE})\s*±\s*({FLOAT_RE})`", 1, dotted("statistics.bestof64_mean_ratio.mean")),
        ("Best-of-64：", rf"`({FLOAT_RE})\s*±\s*({FLOAT_RE})`", 2, dotted("statistics.bestof64_mean_ratio.sample_std")),
        ("Hybrid：", rf"`({FLOAT_RE})\s*±\s*({FLOAT_RE})`", 1, dotted("statistics.hybrid_mean_ratio.mean")),
        ("Hybrid：", rf"`({FLOAT_RE})\s*±\s*({FLOAT_RE})`", 2, dotted("statistics.hybrid_mean_ratio.sample_std")),
        ("配对差值：", rf"`({FLOAT_RE})\s*±\s*({FLOAT_RE})`", 1, dotted("statistics.paired_difference.mean")),
        ("配对差值：", rf"`({FLOAT_RE})\s*±\s*({FLOAT_RE})`", 2, dotted("statistics.paired_difference.sample_std")),
        ("`t(14)=", rf"`t\(14\)=({FLOAT_RE})`", 1, dotted("statistics.paired_t_test_two_sided.t_statistic")),
        ("双侧 `p=", rf"`p=({FLOAT_RE})×10\^(-?\d+)", 1,
         lambda data: float(data["statistics"]["paired_t_test_two_sided"]["p_value"]) * 1e6),
    ]
    for index, (anchor, pattern, group, extractor) in enumerate(paired_specs, 1):
        checks.append(Check(f"paired statistic {index}", anchor, pattern, paired, extractor,
                            group=group, tolerance=5e-5 if "t(14)" in anchor else TOLERANCE))

    # MILP aggregate claims.
    row_check(checks, "MILP HEFT ratio", "| HEFT | 1.", milp, 1,
              milp_mean("heft_over_milp_optimal"))
    row_check(checks, "MILP HEFT gap", "| HEFT | 1.", milp, 2,
              milp_gap_percent("heft_over_milp_optimal"))
    row_check(checks, "MILP residual ratio", "| Residual+Best-of-16 |", milp, 1,
              milp_mean("residual_bestof16_over_milp_optimal"))
    row_check(checks, "MILP residual gap", "| Residual+Best-of-16 |", milp, 2,
              milp_gap_percent("residual_bestof16_over_milp_optimal"))

    # Fixed-seed task-size table.
    for task_size in (10, 15, 20, 25):
        anchor = f"| {task_size} | 5 |"
        row_check(checks, f"task {task_size} mean", anchor, fixed, 2,
                  lambda data, key=str(task_size): float(data["by_task_size"][key]["mean_ratio"]))
        row_check(checks, f"task {task_size} std", anchor, fixed, 3,
                  lambda data, key=str(task_size): float(data["by_task_size"][key]["std_ratio"]))

    # Structural generalization table.
    structural = "evaluation/results/structural_generalization/summary_structural_generalization.json"
    structural_rows = {
        "| 宽并行 |": "wide_parallel",
        "| 深链条 |": "deep_chain",
        "| 同构资源 |": "homogeneous_resources",
        "| 原始分布对照 |": "original_control",
    }
    for anchor, key in structural_rows.items():
        row_check(checks, f"{key} mean", anchor, structural, 1,
                  lambda data, k=key: float(data["groups"][k]["mean_ratio"]))
        row_check(checks, f"{key} std", anchor, structural, 2,
                  lambda data, k=key: float(data["groups"][k]["std_ratio"]))
        row_check(checks, f"{key} wins", anchor, structural, 3,
                  lambda data, k=key: float(data["groups"][k]["outperform_heft_count"]))
        row_check(checks, f"{key} delta", anchor, structural, 4,
                  lambda data, k=key: float(data["groups"][k]["mean_ratio_delta_vs_control"]))

    # Large-model paragraph has two source files and six values.
    large_small = "evaluation/results/summary_mlp_normalized_large_small_eval.json"
    large = "evaluation/results/summary_mlp_normalized_large.json"
    large_anchor = "该通用模型在原 10～25"
    large_pattern = (
        rf"为 \**({FLOAT_RE})\**，在 30、40、50、60 各 5 个场景的大规模验证集上为 \**({FLOAT_RE})\**；"
        rf"各大规模分组分别为 ({FLOAT_RE})、({FLOAT_RE})、({FLOAT_RE})、({FLOAT_RE})"
    )
    checks.append(Check("large model small-set mean", large_anchor, large_pattern,
                        large_small, summary_mean, group=1))
    checks.append(Check("large model large-set mean", large_anchor, large_pattern,
                        large, summary_mean, group=2))
    for group, task_size in enumerate((30, 40, 50, 60), 3):
        checks.append(Check(f"large task {task_size} mean", large_anchor, large_pattern,
                            large,
                            lambda data, key=str(task_size): float(data["by_task_size"][key]["mean_ratio"]),
                            group=group))

    return checks


def build_text_evidence_checks() -> list[TextEvidenceCheck]:
    diagnostic = "artifacts/diagnostic_logs"
    training = "artifacts/training_logs"
    return [
        TextEvidenceCheck(
            "normalization dataset metadata",
            r"500 个 DAG、8038 个决策样本",
            f"{diagnostic}/bc_dataset_sanity_check_output.log",
            (r"samples=8038", r'"num_dags": 500'),
        ),
        TextEvidenceCheck(
            "initial action collapse",
            r"10 个诊断 DAG 的 180 个任务全部分配给 `cloud_0`",
            f"{diagnostic}/diagnostics_action_distribution_output.log",
            (r"cloud_0: count=180, ratio=1\.000000", r"total_scheduled_tasks=180"),
        ),
        TextEvidenceCheck(
            "reward-shaped action collapse",
            r"`cloud_0=100%`",
            f"{diagnostic}/reward_shaped_action_distribution_output.log",
            (r"cloud_0: count=180, ratio=1\.000000",),
        ),
        TextEvidenceCheck(
            "observation scale diagnostics",
            r"8038 个样本.*9\.998.*82\.285.*1000.*30\.936",
            f"{diagnostic}/bc_batch_mask_and_normalization_check_output.log",
            (
                r"samples=8038",
                r"task_computation_cost:.*max=9\.998241",
                r"task_successor_cost_sum:.*max=82\.284714",
                r"resource_bandwidth:.*max=1000\.000000",
                r"flat_observation_all_values:.*std=30\.936210",
            ),
        ),
        TextEvidenceCheck(
            "batch mask and normalized overfit",
            r"合法动作数分别为 4、20、44.*27\.957%.*90\.323%",
            f"{diagnostic}/bc_batch_mask_and_normalization_check_output.log",
            (
                r"legal_action_count=4.*abs_diff=0\.000000000000e\+00",
                r"legal_action_count=20.*abs_diff=0\.000000000000e\+00",
                r"legal_action_count=44.*abs_diff=0\.000000000000e\+00",
                r"previous_without_normalization_best_accuracy=0\.279570",
                r"best_accuracy=0\.903226",
            ),
        ),
        TextEvidenceCheck(
            "joint BC strength",
            r"80 epochs.*99\.104%",
            f"{diagnostic}/mlp_bc_normalized_strength_test_80e_lr1e3_output.log",
            (r"best_accuracy=0\.991043",),
        ),
        TextEvidenceCheck(
            "normalized action distribution",
            r"`cloud_0` 占比由 100% 降至 \*\*91\.11%\*\*",
            f"{diagnostic}/mlp_normalized_action_distribution_output.log",
            (r"cloud_0: count=164, ratio=0\.911111", r"total_scheduled_tasks=180"),
        ),
        TextEvidenceCheck(
            "joint action-space complexity",
            r"350 个决策步骤.*12\.377.*3\.094.*4 个资源.*32.*310.*88\.5714%.*2\.491",
            f"{diagnostic}/action_space_complexity_check_output.log",
            (
                r"total_decision_steps=350",
                r"avg_legal_actions_per_step=12\.377143",
                r"avg_candidate_tasks_per_step=3\.094286",
                r"avg_candidate_resources_per_ready_task=4\.000000",
                r"max_legal_actions_per_step=32",
                r"unstable_pairs_unique_resource_gt_1=310",
                r"unstable_pair_ratio=0\.885714",
                r"avg_unique_resource_count=2\.491429",
            ),
        ),
        TextEvidenceCheck(
            "BC chance level",
            r"16\.7%～17\.2%.*7\.54%",
            f"{diagnostic}/bc_dataset_sanity_check_output.log",
            (
                r"expected_uniform_random_accuracy_all_mean_inverse=0\.075411",
                r"reference_bc_accuracy_observed_previous_run=0\.166708_to_0\.172431",
            ),
        ),
        TextEvidenceCheck(
            "ranked small-set BC overfit",
            r"ranked 专属小数据集在 200 epochs 达到 100%",
            f"{diagnostic}/scan_step1_ranked_bc_small_overfit.log",
            (r"best_accuracy=1\.000000",),
        ),
        TextEvidenceCheck(
            "ranked full-set BC accuracy",
            r"完整 ranked 数据集在 80 epochs 达到 \*\*98\.967%\*\*",
            f"{diagnostic}/scan_step1_ranked_bc_full_pretrain_only.log",
            (r"best_accuracy=0\.989674",),
        ),
        TextEvidenceCheck(
            "openEuler pytest result",
            r"19 passed, 17 warnings in 5\.33s",
            "evaluation/results/openEuler_pytest_structural_final.log",
            (r"19 passed, 17 warnings in 5\.33s",),
        ),
        TextEvidenceCheck(
            "fresh-clone pipeline",
            r"200704 个 PPO rollout 步.*489\.71 秒.*20 个场景",
            "evaluation/results/run_final_pipeline_fresh_clone_full_output.log",
            (
                r"total_timesteps\s+\| 200704",
                r"training elapsed seconds: 489\.71",
                r"generated 20 validation scenarios",
                r"FINAL_PIPELINE_COMPLETE",
            ),
        ),
        TextEvidenceCheck(
            "GAT historical training time",
            r"GAT joint 的 200k 历史训练耗时为 1941\.22 秒",
            f"{training}/gat_full_train_20260707_133559_output.log",
            (r"total_timesteps\s+\| 200704", r"training elapsed seconds: 1941\.22"),
        ),
        TextEvidenceCheck(
            "ranked historical training time",
            r"MLP ranked 为 413\.70 秒",
            f"{training}/mlp_ranked_train_output.log",
            (r"total_timesteps\s+\| 200704", r"training elapsed seconds: 413\.70"),
        ),
        TextEvidenceCheck(
            "residual historical training time",
            r"记录 925\.23 秒",
            f"{training}/residual_train_ppo_mlp_residual.log",
            (r"total_timesteps\s+\| 200704", r"training elapsed seconds: 925\.23"),
        ),
    ]


def main() -> int:
    text = REPORT.read_text(encoding="utf-8")
    lines = text.splitlines()
    tracked = tracked_files()
    failures: list[str] = []
    passed = 0

    print("TECHNICAL_REPORT_RESULT_AUDIT")
    print(f"report: {REPORT.relative_to(ROOT).as_posix()}")
    print(f"tolerance: {TOLERANCE} (absolute; displayed values may be rounded)")
    print("\n[CONCRETE REPOSITORY PATH SCAN]")
    path_occurrences: list[tuple[int, str]] = []
    wildcard_occurrences: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, 1):
        for match in REPO_PATH_RE.finditer(line):
            path = match.group(1)
            if "*" in path:
                wildcard_occurrences.append((line_number, path))
                print(f"INFO line {line_number}: wildcard pattern (not a concrete evidence path): {path}")
                continue
            path_occurrences.append((line_number, path))
            if not (ROOT / path).is_file():
                message = f"line {line_number}: missing repository path: {path}"
                failures.append(message)
                print(f"FAIL {message}")
            elif path not in tracked:
                message = f"line {line_number}: path exists locally but is not tracked by Git: {path}"
                failures.append(message)
                print(f"FAIL {message}")
            else:
                print(f"PASS line {line_number}: tracked path: {path}")

    print("\n[JSON PATH SCAN]")
    occurrences: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, 1):
        for match in JSON_PATH_RE.finditer(line):
            path = match.group(0)
            occurrences.append((line_number, path))
            if "*" in path:
                message = f"line {line_number}: wildcard result path is not auditable: {path}"
                failures.append(message)
                print(f"FAIL {message}")
            elif not (ROOT / path).is_file():
                message = f"line {line_number}: missing result JSON: {path}"
                failures.append(message)
                print(f"FAIL {message}")
            elif path not in tracked:
                message = f"line {line_number}: result JSON exists locally but is not tracked by Git: {path}"
                failures.append(message)
                print(f"FAIL {message}")
            else:
                print(f"PASS line {line_number}: tracked result JSON: {path}")

    print("\n[NUMERIC CLAIMS]")
    cache: dict[str, dict] = {}
    checks = build_checks()
    for check in checks:
        section_start = 0
        section_end = len(lines)
        if check.section is not None:
            section_matches = [index for index, line in enumerate(lines) if check.section in line]
            if not section_matches:
                message = f"{check.label}: section not found: {check.section!r}"
                failures.append(message)
                print(f"FAIL {message}")
                continue
            section_start = section_matches[0]
            for index in range(section_start + 1, len(lines)):
                if lines[index].startswith("## "):
                    section_end = index
                    break
        matching = [
            (index + 1, line)
            for index, line in enumerate(lines)
            if section_start <= index < section_end and check.anchor in line
        ]
        if not matching:
            message = f"{check.label}: anchor not found: {check.anchor!r}"
            failures.append(message)
            print(f"FAIL {message}")
            continue
        # A repeated anchor is allowed only when all occurrences produce the same expected claim.
        matched_claim = False
        for line_number, line in matching:
            match = re.search(check.pattern, line)
            if not match:
                continue
            matched_claim = True
            reported = float(match.group(check.group))
            data = cache.setdefault(check.source, load_json(check.source))
            actual = float(check.actual(data))
            delta = abs(reported - actual)
            token = match.group(check.group)
            decimals = len(token.lower().split("e", 1)[0].split(".", 1)[1]) if "." in token else 0
            display_tolerance = 0.5 * (10 ** -decimals) + 1e-12
            effective_tolerance = max(check.tolerance, display_tolerance)
            if delta <= effective_tolerance:
                passed += 1
                print(
                    f"PASS line {line_number}: {check.label}: "
                    f"reported={reported:.12g}, actual={actual:.12g}, delta={delta:.3g}, "
                    f"allowed={effective_tolerance:.3g}"
                )
            else:
                message = (
                    f"line {line_number}: {check.label}: reported={reported:.12g}, "
                    f"actual={actual:.12g}, delta={delta:.12g}, source={check.source}"
                )
                failures.append(message)
                print(f"FAIL {message}")
        if not matched_claim:
            message = f"{check.label}: value pattern not found near anchor {check.anchor!r}"
            failures.append(message)
            print(f"FAIL {message}")

    print("\n[LOG-BACKED CLAIMS]")
    text_checks = build_text_evidence_checks()
    text_checks_passed = 0
    for check in text_checks:
        if not re.search(check.report_pattern, text):
            message = f"{check.label}: report claim pattern not found: {check.report_pattern!r}"
            failures.append(message)
            print(f"FAIL {message}")
            continue
        source_path = ROOT / check.source
        if not source_path.is_file():
            message = f"{check.label}: missing evidence file: {check.source}"
            failures.append(message)
            print(f"FAIL {message}")
            continue
        if check.source not in tracked:
            message = f"{check.label}: evidence file is not tracked by Git: {check.source}"
            failures.append(message)
            print(f"FAIL {message}")
            continue
        source_text = read_text_file(source_path)
        missing_patterns = [
            pattern for pattern in check.source_patterns
            if not re.search(pattern, source_text, re.MULTILINE)
        ]
        if missing_patterns:
            message = f"{check.label}: evidence patterns missing from {check.source}: {missing_patterns}"
            failures.append(message)
            print(f"FAIL {message}")
            continue
        text_checks_passed += 1
        print(f"PASS {check.label}: {check.source}")

    print("\n[TRACKED-TEXT PRIVACY SCAN]")
    privacy_hits = 0
    for relative_path in sorted(tracked):
        path = ROOT / relative_path
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        content = read_text_file(path)
        for line_number, line in enumerate(content.splitlines(), 1):
            match = PRIVATE_PATH_RE.search(line)
            if match:
                privacy_hits += 1
                message = (
                    f"private user-home path in tracked text: {relative_path}:{line_number}: "
                    f"{match.group(0)}"
                )
                failures.append(message)
                print(f"FAIL {message}")
    if privacy_hits == 0:
        print("PASS no Windows or Linux user-home paths in tracked text files")

    print("\n[SUMMARY]")
    print(f"concrete_path_occurrences: {len(path_occurrences)}")
    print(f"wildcard_path_occurrences: {len(wildcard_occurrences)}")
    print(f"json_path_occurrences: {len(occurrences)}")
    print(f"numeric_checks_passed: {passed}")
    print(f"log_backed_checks_passed: {text_checks_passed}")
    print(f"privacy_path_hits: {privacy_hits}")
    print(f"failures: {len(failures)}")
    if failures:
        print("\n[MISMATCHES / UNRESOLVED]")
        for item in failures:
            print(f"- {item}")
        return 1
    print("All declared result claims, evidence paths, Git tracking checks, and privacy scans passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
