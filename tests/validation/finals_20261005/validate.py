"""Re-run finals layout and checkpoint checks without touching formal results.

Run from the repository root: python tests/validation/finals_20261005/validate.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
HOME_RE = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/][^\\/\r\n]+|/home/[^/\r\n]+)", re.IGNORECASE)


def sanitized(value: str) -> str:
    replacements = (
        (ROOT, "<PROJECT_ROOT>"),
        (Path(sys.executable).resolve().parents[1], "<VALIDATION_ENV>"),
        (Path(tempfile.gettempdir()), "<TEMP_DIR>"),
    )
    for path, replacement in replacements:
        value = value.replace(str(path), replacement).replace(path.as_posix(), replacement)
    return HOME_RE.sub("<USER_HOME>", value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from env.observation_normalizer import DEFAULT_STATS_PATH
    from env.resource_config import default_resource_config_path

    packages = ("env", "baselines", "policies", "training", "evaluation", "diagnostics", "scheduler_interface")
    imported_paths = {}
    for name in packages:
        path = Path(importlib.import_module(name).__file__).resolve()
        path.relative_to(ROOT / "src")
        imported_paths[name] = path.relative_to(ROOT).as_posix()
    assert default_resource_config_path() == ROOT / "configs/resource_default.yaml"
    assert DEFAULT_STATS_PATH == ROOT / "src/env/normalization_stats.json"
    assert DEFAULT_STATS_PATH.is_file()
    docker_patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "!training/checkpoints/ppo_mlp_residual.zip" in docker_patterns
    assert docker_patterns.index("!training/checkpoints/ppo_mlp_residual.zip") > docker_patterns.index("*.zip")

    formal_files = sorted((ROOT / "evaluation/results").rglob("*.json"))
    formal_hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in formal_files}
    checkpoint = ROOT / "training/checkpoints/ppo_mlp_residual.zip"
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in (
            "torch", "pytest", "gymnasium", "numpy", "networkx", "PyYAML",
            "matplotlib", "sb3-contrib", "stable-baselines3", "pulp", "scipy", "setuptools",
        )},
        "torch_runtime_version": importlib.import_module("torch").__version__,
    }
    results = []

    def run(label: str, arguments: list[str]) -> None:
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-B", *arguments], cwd=ROOT, text=True,
            encoding="utf-8", errors="replace", capture_output=True,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.perf_counter() - started
        log_name = f"{label}.log"
        (EVIDENCE / log_name).write_text(
            sanitized(completed.stdout + completed.stderr), encoding="utf-8",
        )
        results.append({
            "label": label, "command": ["python", "-B", *map(sanitized, arguments)],
            "exit_code": completed.returncode, "elapsed_seconds": elapsed, "log": log_name,
        })
        print(f"{label}: exit={completed.returncode} seconds={elapsed:.3f}", flush=True)
        if completed.returncode:
            raise RuntimeError(f"{label} failed; inspect {log_name}")

    with tempfile.TemporaryDirectory(prefix="drl-finals-pytest-") as temporary:
        run("pytest", ["-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--basetemp", temporary])
    run("compile", ["-m", "compileall", "-q", "src", "scripts", "tests", "demo"])
    run("train_cli_help", ["src/training/train_ppo.py", "--help"])
    run("evaluation_cli_help", ["src/evaluation/evaluate_residual_lns.py", "--help"])
    run("reproduce_smoke", [
        "scripts/reproduce.py", "--profile", "smoke", "--skip-tests", "--write-manifest",
        "--output-dir", "tests/validation/finals_20261005/smoke",
    ])
    run("checkpoint_demo", [
        "demo/run_demo.py", "--skip-plot", "--output-dir", "tests/validation/finals_20261005/demo",
    ])
    run("demo_visualization", [
        "scripts/visualize_schedule.py", "--scenario", "evaluation/scenarios/scenario_10_0.json",
        "--result", "tests/validation/finals_20261005/demo/adaptive.json",
        "--output", "tests/validation/finals_20261005/demo/schedule.png",
    ])
    demo = json.loads((EVIDENCE / "demo/adaptive.json").read_text(encoding="utf-8"))
    assert demo["validation"]["valid"]
    assert demo["result"]["ratio_to_heft"] <= 1.0 + 1e-9
    unchanged = all(digest(ROOT / path) == sha for path, sha in formal_hashes.items())
    assert unchanged, "a formal experiment result changed during validation"
    report = {
        "environment": environment,
        "scope": "Native Windows source-layout regression and one shipped-checkpoint inference; no full training or container OS matrix was rerun.",
        "imported_modules": imported_paths,
        "checkpoint_sha256": digest(checkpoint),
        "formal_json_count_unchanged": len(formal_hashes),
        "formal_results_unchanged": unchanged,
        "docker_checkpoint_included_by_ignore_rules": True,
        "commands": results,
        "demo": {
            "task_count": demo["dag"]["task_count"],
            "valid": demo["validation"]["valid"],
            "makespan": demo["result"]["makespan"],
            "heft_makespan": demo["result"]["heft_makespan"],
            "ratio_to_heft": demo["result"]["ratio_to_heft"],
            "selected_source": demo["selected_source"],
            "model_load_seconds": demo["timing"]["model_load_seconds"],
            "schedule_seconds": demo["timing"]["schedule_seconds"],
        },
    }
    (EVIDENCE / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINALS_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
