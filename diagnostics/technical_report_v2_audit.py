"""Evidence and numeric-claim audit for the current docs/技术报告.md."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PATH_RE = re.compile(r"`([^`]+\.(?:json|log|yaml|yml|py|sh|ps1|zip|png|docx|txt))`")
RESULT_JSON_RE = re.compile(r"evaluation/results/[A-Za-z0-9_./+-]+\.json")
PRIVATE_RE = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/][^\\/\s]+|/home/[^/\s]+)", re.IGNORECASE)


def load(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get(data: dict, dotted: str) -> float:
    value = data
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def token(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def require(text: str, label: str, needle: str, failures: list[str]) -> None:
    if needle in text:
        print(f"PASS {label}: {needle}")
    else:
        failures.append(f"{label}: missing {needle}")
        print(f"FAIL {label}: missing {needle}")


def audit_report_v2(report_path: Path) -> int:
    text = report_path.read_text(encoding="utf-8")
    failures: list[str] = []
    print("TECHNICAL_REPORT_CURRENT_AUDIT")
    print(f"report: {report_path.relative_to(ROOT).as_posix()}")

    print("\n[REPOSITORY PATHS]")
    paths = sorted(set(PATH_RE.findall(text)))
    for relative in paths:
        candidate = ROOT / relative
        if candidate.exists():
            print(f"PASS path exists: {relative}")
        else:
            failures.append(f"missing path: {relative}")
            print(f"FAIL missing path: {relative}")
    for relative in sorted(set(RESULT_JSON_RE.findall(text))):
        if not (ROOT / relative).is_file():
            failures.append(f"missing result JSON: {relative}")
            print(f"FAIL missing result JSON: {relative}")

    print("\n[DIRECT FINAL RESULT]")
    direct_path = "evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json"
    direct = load(direct_path)
    s = direct["statistics"]
    direct_tokens = {
        "residual mean": token(s["residual_bestof64_mean_ratio"]["mean"], 6),
        "residual std": token(s["residual_bestof64_mean_ratio"]["sample_std"], 6),
        "LNS mean": token(s["lns_mean_ratio"]["mean"], 6),
        "LNS std": token(s["lns_mean_ratio"]["sample_std"], 6),
        "paired diff": token(s["paired_difference"]["mean"], 6),
        "paired diff std": token(s["paired_difference"]["sample_std"], 6),
        "t statistic": token(s["paired_t_test_two_sided"]["t_statistic"], 6),
        "p mantissa": token(s["paired_t_test_two_sided"]["p_value"] * 1e5, 6),
        "elapsed mean": token(mean(row["elapsed_seconds"] for row in direct["paired_runs"]), 6),
    }
    for label, value in direct_tokens.items():
        require(text, label, value, failures)
    for row in direct["paired_runs"]:
        for label, value in (
            (f"run {row['repeat']} residual", row["residual_bestof64_mean_ratio"]),
            (f"run {row['repeat']} LNS", row["lns_mean_ratio"]),
            (f"run {row['repeat']} diff", row["difference_lns_minus_residual"]),
            (f"run {row['repeat']} elapsed", row["elapsed_seconds"]),
        ):
            require(text, label, token(float(value), 6), failures)

    print("\n[COMPUTE-MATCHED RESULT]")
    compute_path = "evaluation/results/autonomous_exploration/compute_matched_sampling/paired_comparison_summary.json"
    compute = load(compute_path)["statistics"]
    compute_tokens = {
        "Best128 mean": token(compute["pure_sampling_mean_ratio"]["mean"], 6),
        "Best128 std": token(compute["pure_sampling_mean_ratio"]["sample_std"], 6),
        "compute diff": token(compute["paired_difference"]["mean"], 6),
        "compute diff std": token(compute["paired_difference"]["sample_std"], 6),
        "sampling elapsed": token(compute["pure_sampling_elapsed_seconds"]["mean"], 6),
        "LNS elapsed": token(compute["lns_elapsed_seconds"]["mean"], 6),
        "elapsed ratio": token(compute["elapsed_ratio_sampling_over_lns"], 6),
        "compute t": token(compute["paired_t_test_two_sided"]["t_statistic"], 6),
        "compute p mantissa": token(compute["paired_t_test_two_sided"]["p_value"] * 1e6, 6),
    }
    for label, value in compute_tokens.items():
        require(text, label, value, failures)

    print("\n[EVOLUTION AND ABLATIONS]")
    summary_claims = [
        ("initial", "evaluation/results/summary.json", 6),
        ("relative reward", "evaluation/results/summary_reward_shaped.json", 6),
        ("normalized", "evaluation/results/summary_mlp_normalized.json", 6),
        ("ranked", "evaluation/results/summary_mlp_ranked.json", 6),
        ("residual", "evaluation/results/summary_mlp_residual.json", 6),
        ("fixed Best64", "evaluation/results/summary_mlp_residual_bestof64.json", 6),
        ("GAT ranked", "evaluation/results/summary_gat_ranked.json", 6),
        ("ranked BC", "evaluation/results/summary_mlp_ranked_bc.json", 6),
        ("rank feature", "evaluation/results/summary_mlp_ranked_with_rank.json", 6),
        ("curriculum", "evaluation/results/summary_mlp_ranked_curriculum.json", 6),
        ("large", "evaluation/results/summary_mlp_normalized_large.json", 6),
        ("large small eval", "evaluation/results/summary_mlp_normalized_large_small_eval.json", 6),
    ]
    for label, source, digits in summary_claims:
        require(text, label, token(float(load(source)["overall"]["mean_ratio"]), digits), failures)

    print("\n[STRUCTURAL GENERALIZATION]")
    structural = load("evaluation/results/structural_generalization/summary_structural_generalization.json")
    for key, values in structural["groups"].items():
        require(text, f"{key} mean", token(values["mean_ratio"], 6), failures)
        require(text, f"{key} std", token(values["std_ratio"], 6), failures)
        require(text, f"{key} delta", token(values["mean_ratio_delta_vs_control"], 6), failures)

    print("\n[MILP]")
    milp = load("evaluation/results/milp_optimal_comparison.json")
    optimal = [row for row in milp["scenarios"] if row["milp_proven_optimal"]]
    require(text, "MILP proven count", f"{len(optimal)} 个已证明最优", failures)
    require(text, "MILP HEFT ratio", token(mean(row["heft_over_milp_optimal"] for row in optimal), 6), failures)
    require(text, "MILP residual ratio", token(mean(row["residual_bestof16_over_milp_optimal"] for row in optimal), 6), failures)

    print("\n[REFERENCES]")
    verified_urls = [
        "https://doi.org/10.1109/71.993206",
        "https://openreview.net/forum?id=WL8FlAugqQ",
        "https://arxiv.org/abs/2309.15517",
        "https://doi.org/10.1287/trsc.1050.0135",
        "https://openreview.net/forum?id=nO5caZwFwYu",
        "https://www.jmlr.org/papers/v22/20-303.html",
        "https://www.dhs.tsinghua.edu.cn/wp-content/uploads/2023/12/2024031107044595.pdf",
    ]
    for url in verified_urls:
        require(text, "verified reference", url, failures)

    print("\n[PRIVACY]")
    hit = PRIVATE_RE.search(text)
    if hit:
        failures.append(f"private path in report: {hit.group(0)}")
        print(f"FAIL private path: {hit.group(0)}")
    else:
        print("PASS no user-home path in report")

    print("\n[SUMMARY]")
    print(f"paths_checked: {len(paths)}")
    print(f"failures: {len(failures)}")
    if failures:
        print("\n[MISMATCHES / UNRESOLVED]")
        for item in failures:
            print(f"- {item}")
        return 1
    print("All current-report evidence paths, numeric claims, references, and privacy checks passed.")
    return 0
