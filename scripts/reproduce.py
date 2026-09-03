from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.generate_structural_generalization_scenarios import DEFAULT_OUTPUT_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRODUCE_SCENARIOS_DIR = f"{DEFAULT_OUTPUT_ROOT}_reproduce"
PRIVATE_HOME_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]Users[\\/][^\\/\r\n]+|/home/[^/\r\n]+)",
    re.IGNORECASE,
)


def _sanitize_text(value: str) -> str:
    return PRIVATE_HOME_RE.sub("<USER_HOME>", value)


def _public_command(command: list[str]) -> list[str]:
    public: list[str] = []
    python_path = Path(sys.executable).resolve()
    for index, item in enumerate(command):
        try:
            is_python = index == 0 and Path(item).resolve() == python_path
        except (OSError, RuntimeError):
            is_python = False
        public.append("python" if is_python else _sanitize_text(item))
    return public


def _powershell_executable() -> str:
    for candidate in (
        shutil.which("powershell.exe"),
        shutil.which("powershell"),
        str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("unable to locate a Windows PowerShell executable")


def _run(command: list[str], *, label: str, check: bool = True) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    elapsed = time.perf_counter() - started
    print(f"[{label}] exit_code={completed.returncode} elapsed_seconds={elapsed:.3f}", flush=True)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return {
        "label": label,
        "command": _public_command(command),
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
    }


def _environment_report() -> dict:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "project_root": ".",
    }
    git = shutil.which("git")
    if git:
        completed = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        report["git_revision"] = completed.stdout.strip() if completed.returncode == 0 else None
        dirty = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        report["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    os_release = Path("/etc/os-release")
    if os_release.exists():
        report["os_release"] = os_release.read_text(encoding="utf-8", errors="replace")
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Cross-platform reproducibility entry point for the scheduling project."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "wide", "final"),
        default="smoke",
        help="smoke=compile+tests+scenario generation; wide adds adaptive wide-DAG comparison; final runs the five-repeat pipeline.",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--lns-iterations", type=int, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument(
        "--with-os-matrix",
        action="store_true",
        help="Build/run the openEuler, openKylin, and Anolis Docker containers after the local checks.",
    )
    parser.add_argument(
        "--os-matrix-wait-seconds",
        type=float,
        default=120.0,
        help="When --with-os-matrix is enabled, wait this long for Docker Desktop/daemon to become ready.",
    )
    parser.add_argument(
        "--os-matrix-poll-interval",
        type=float,
        default=5.0,
        help="Retry interval passed to the Docker OS matrix checker.",
    )
    args = parser.parse_args()

    commands: list[dict] = []
    environment = _environment_report()
    print("REPRODUCE_START", flush=True)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)

    commands.append(
        _run(
            [sys.executable, "-m", "compileall", "-q", "env", "baselines", "policies", "training", "evaluation", "tests"],
            label="compileall",
        )
    )
    if not args.skip_tests:
        # A unique base directory avoids failures caused by stale or permission-damaged
        # pytest-of-<user> directories on long-lived Windows hosts.
        with tempfile.TemporaryDirectory(prefix="drl-scheduler-pytest-") as pytest_temp:
            commands.append(
                _run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests",
                        "-q",
                        "--basetemp",
                        pytest_temp,
                    ],
                    label="pytest",
                )
            )

    commands.append(
        _run(
            [
                sys.executable,
                "evaluation/generate_structural_generalization_scenarios.py",
                "--output-root",
                REPRODUCE_SCENARIOS_DIR,
            ],
            label="generate structural scenarios",
        )
    )

    if args.profile in {"wide", "final"}:
        wide_command = [
            sys.executable,
            "evaluation/evaluate_wide_parallel.py",
            "--scenarios-dir",
            f"{REPRODUCE_SCENARIOS_DIR}/wide_parallel",
            "--results-path",
            "evaluation/results/wide_parallel_adaptive_reproduce.json",
            "--num-samples",
            str(args.num_samples if args.num_samples is not None else (16 if args.profile == "wide" else 64)),
            "--lns-iterations",
            str(args.lns_iterations if args.lns_iterations is not None else (16 if args.profile == "wide" else 64)),
        ]
        commands.append(_run(wide_command, label="wide-parallel adaptive evaluation"))

    if args.profile == "final":
        if platform.system().lower().startswith("win"):
            commands.append(
                _run(
                    [
                        _powershell_executable(),
                        "-NoLogo",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        "scripts/run_final_pipeline.ps1",
                    ],
                    label="five-repeat final pipeline",
                )
            )
        else:
            commands.append(
                _run(["bash", "scripts/run_final_pipeline.sh"], label="five-repeat final pipeline")
            )

    if args.with_os_matrix:
        commands.append(
            _run(
                [
                    sys.executable,
                    "scripts/run_os_matrix.py",
                    "--wait-seconds",
                    str(args.os_matrix_wait_seconds),
                    "--poll-interval",
                    str(args.os_matrix_poll_interval),
                ],
                label="domestic operating-system Docker matrix",
                check=False,
            )
        )

    manifest = {"environment": environment, "profile": args.profile, "commands": commands}
    if args.write_manifest:
        output_path = PROJECT_ROOT / "evaluation" / "results" / "reproducibility_manifest.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest_path={output_path}", flush=True)
    print("REPRODUCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
