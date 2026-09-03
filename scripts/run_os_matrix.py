from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_JOBS = [
    ("openEuler-24.03-LTS-SP4", "openeuler", "docker/Dockerfile.openeuler"),
    ("openKylin-2.0-SP1", "openkylin", "docker/Dockerfile.openkylin"),
    ("Anolis-23", "anolis", "docker/Dockerfile.anolis"),
]
PRIVATE_HOME_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]Users[\\/][^\\/\r\n]+|/home/[^/\r\n]+)",
    re.IGNORECASE,
)


def _sanitize_text(value: str) -> str:
    return PRIVATE_HOME_RE.sub("<USER_HOME>", value)


def _configured_jobs() -> list[dict[str, str]]:
    return [
        {"os": os_name, "image_tag": tag, "dockerfile": dockerfile}
        for os_name, tag, dockerfile in MATRIX_JOBS
    ]


def _sha256_file(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (PROJECT_ROOT / relative_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_inputs() -> dict:
    paths = ["requirements.txt", *(dockerfile for _, _, dockerfile in MATRIX_JOBS)]
    return {
        "files": [
            {"path": path, "sha256": _sha256_file(path)}
            for path in paths
        ]
    }


def _docker_available(wait_seconds: float = 0.0, poll_interval: float = 5.0) -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker executable not found"
    deadline = time.monotonic() + max(0.0, wait_seconds)
    last_detail = ""
    attempt = 0
    while True:
        attempt += 1
        completed = subprocess.run(
            [docker, "version", "--format", "{{.Server.Version}}"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0:
            return True, completed.stdout.strip()
        last_detail = _sanitize_text((completed.stderr or completed.stdout).strip())
        if time.monotonic() >= deadline:
            break
        remaining = max(0.0, deadline - time.monotonic())
        sleep_for = min(poll_interval, remaining)
        if sleep_for <= 0:
            break
        print(
            f"docker not ready yet (attempt {attempt}); retrying in {sleep_for:.1f}s: {last_detail}",
            file=sys.stderr,
        )
        time.sleep(sleep_for)
    return False, f"docker daemon unavailable: {last_detail}"


def _diagnostics() -> dict:
    diagnostics: dict[str, str | int] = {}
    commands = {
        "docker_context_ls": ["docker", "context", "ls"],
        "wsl_status": ["wsl", "--status"],
        "wsl_list": ["wsl", "-l", "-v"],
    }
    for name, command in commands.items():
        executable = shutil.which(command[0])
        if executable is None:
            diagnostics[name] = f"{command[0]} executable not found"
            continue
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        diagnostics[f"{name}_exit_code"] = completed.returncode
        diagnostics[name] = _sanitize_text(
            _decode_process_output(completed.stdout + completed.stderr).strip()
        )
    return diagnostics


def _decode_process_output(output: bytes) -> str:
    if not output:
        return ""
    if b"\x00" in output[:200]:
        return output.decode("utf-16le", errors="replace")
    return output.decode(sys.getfilesystemencoding() or "utf-8", errors="replace")


def _run_image(os_name: str, tag: str, dockerfile: str, pull: bool) -> dict:
    docker = shutil.which("docker")
    assert docker is not None
    image = f"drl-scheduler:{tag}"
    build_command = [docker, "build"]
    if pull:
        build_command.append("--pull")
    build_command += ["-f", dockerfile, "-t", image, "."]
    started = time.perf_counter()
    build = subprocess.run(build_command, cwd=PROJECT_ROOT, text=True)
    build_elapsed = time.perf_counter() - started
    if build.returncode != 0:
        return {
            "os": os_name,
            "image": image,
            "build_exit_code": build.returncode,
            "run_exit_code": None,
            "build_elapsed_seconds": build_elapsed,
            "status": "build_failed",
        }
    inspect = subprocess.run(
        [docker, "image", "inspect", image, "--format", "{{.Id}}"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    verification_command = (
        "set -eu; "
        "cat /etc/os-release; "
        "python3 --version; "
        "python3 -c 'import torch; print(\"torch=\" + torch.__version__); "
        "print(\"cuda_available=\" + str(torch.cuda.is_available()))'; "
        "python3 -m compileall -q env baselines policies training evaluation scripts tests; "
        "python3 scripts/reproduce.py --profile smoke --write-manifest"
    )
    started = time.perf_counter()
    run = subprocess.run(
        [docker, "run", "--rm", image, "sh", "-lc", verification_command],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    run_elapsed = time.perf_counter() - started
    if run.stdout:
        print(run.stdout, end="" if run.stdout.endswith("\n") else "\n")
    if run.stderr:
        print(run.stderr, file=sys.stderr, end="" if run.stderr.endswith("\n") else "\n")
    return {
        "os": os_name,
        "image": image,
        "image_id": inspect.stdout.strip() if inspect.returncode == 0 else None,
        "build_exit_code": build.returncode,
        "run_exit_code": run.returncode,
        "build_elapsed_seconds": build_elapsed,
        "run_elapsed_seconds": run_elapsed,
        "verification_command": verification_command,
        "stdout": _sanitize_text(run.stdout),
        "stderr": _sanitize_text(run.stderr),
        "status": "passed" if run.returncode == 0 else "run_failed",
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build and run the domestic-OS Docker matrix.")
    parser.add_argument("--pull", action="store_true", help="Pull the latest base image metadata first.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when Docker is unavailable or a job fails.")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=120.0,
        help="Wait this long for Docker Desktop/daemon to become ready before skipping the matrix.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Retry interval while waiting for Docker to become ready.",
    )
    parser.add_argument(
        "--results-path",
        default="evaluation/results/os_matrix.json",
        help="Write the OS matrix result JSON here.",
    )
    args = parser.parse_args()

    results_path = PROJECT_ROOT / args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)

    available, detail = _docker_available(wait_seconds=args.wait_seconds, poll_interval=args.poll_interval)
    diagnostics = _diagnostics()
    print(f"OS_MATRIX_START docker_available={available} detail={detail}")
    if not available:
        result = {
            "status": "skipped",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": detail,
            "configured_jobs": _configured_jobs(),
            "inputs": _evidence_inputs(),
            "diagnostics": diagnostics,
            "jobs": [],
        }
        results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"os_matrix_results_path={results_path}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.strict:
            raise SystemExit(2)
        return

    results = [_run_image(name, tag, dockerfile, args.pull) for name, tag, dockerfile in MATRIX_JOBS]
    result = {
        "status": "passed" if all(job["status"] == "passed" for job in results) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configured_jobs": _configured_jobs(),
        "inputs": _evidence_inputs(),
        "diagnostics": diagnostics,
        "jobs": results,
    }
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"os_matrix_results_path={results_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
