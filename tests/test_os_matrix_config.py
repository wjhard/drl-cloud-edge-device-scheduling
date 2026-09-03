from __future__ import annotations

from pathlib import Path

from scripts.reproduce import _sanitize_text as sanitize_reproduce_text
from scripts.run_os_matrix import MATRIX_JOBS, _configured_jobs, _sanitize_text


def test_domestic_os_matrix_includes_three_official_images() -> None:
    job_names = [name for name, _, _ in MATRIX_JOBS]
    assert job_names == ["openEuler-24.03-LTS-SP4", "openKylin-2.0-SP1", "Anolis-23"]


def test_domestic_os_matrix_dockerfiles_exist() -> None:
    for _, _, dockerfile in MATRIX_JOBS:
        assert Path(dockerfile).exists(), dockerfile


def test_domestic_os_matrix_result_records_configured_jobs() -> None:
    assert _configured_jobs() == [
        {
            "os": "openEuler-24.03-LTS-SP4",
            "image_tag": "openeuler",
            "dockerfile": "docker/Dockerfile.openeuler",
        },
        {
            "os": "openKylin-2.0-SP1",
            "image_tag": "openkylin",
            "dockerfile": "docker/Dockerfile.openkylin",
        },
        {
            "os": "Anolis-23",
            "image_tag": "anolis",
            "dockerfile": "docker/Dockerfile.anolis",
        },
    ]


def test_dockerfiles_use_official_community_registries() -> None:
    expected_base_images = {
        "docker/Dockerfile.openeuler": (
            "FROM hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp4\n"
        ),
        "docker/Dockerfile.openkylin": (
            "FROM quay.io/openkylin/openkylin:2.0\n"
        ),
        "docker/Dockerfile.anolis": (
            "FROM registry.openanolis.cn/openanolis/anolisos:23\n"
        ),
    }
    for path, expected_prefix in expected_base_images.items():
        dockerfile = Path(path).read_text(encoding="utf-8")
        assert dockerfile.startswith(expected_prefix)


def test_rpm_based_images_do_not_require_full_system_upgrade() -> None:
    for path in ("docker/Dockerfile.openeuler", "docker/Dockerfile.anolis"):
        dockerfile = Path(path).read_text(encoding="utf-8")
        assert "dnf -y update" not in dockerfile


def test_openkylin_disables_the_unused_slow_optional_ppa() -> None:
    dockerfile = Path("docker/Dockerfile.openkylin").read_text(encoding="utf-8")
    assert "rm -f /etc/apt/sources.list.d/openkylin-anything.list" in dockerfile


def test_anolis_bootstraps_pip_without_repository_upgrades() -> None:
    dockerfile = Path("docker/Dockerfile.anolis").read_text(encoding="utf-8")
    assert "python3 -m ensurepip --upgrade" in dockerfile
    assert "dnf " not in dockerfile


def test_os_images_do_not_install_unused_compiler_toolchains() -> None:
    for path in (
        "docker/Dockerfile.openeuler",
        "docker/Dockerfile.openkylin",
        "docker/Dockerfile.anolis",
    ):
        dockerfile = Path(path).read_text(encoding="utf-8")
        for unused_package in ("gcc", "gcc-c++", "build-essential", " git"):
            assert unused_package not in dockerfile


def test_os_images_install_cpu_only_pytorch() -> None:
    for path in (
        "docker/Dockerfile.openeuler",
        "docker/Dockerfile.openkylin",
        "docker/Dockerfile.anolis",
    ):
        dockerfile = Path(path).read_text(encoding="utf-8")
        assert "ARG PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu" in dockerfile
        assert "--index-url ${PYTORCH_INDEX_URL}" in dockerfile
        assert "torch==2.12.1+cpu" in dockerfile
        assert "--no-deps" in dockerfile
        assert "python3 -m pip check" in dockerfile
        assert dockerfile.index("COPY requirements.txt ./") < dockerfile.index("COPY . .")
        assert "ARG PYPI_INDEX_URL=" in dockerfile
        assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile


def test_docker_context_excludes_local_environments_and_submission_media() -> None:
    patterns = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".git", ".venv", ".validation_env", "docs", "evaluation/results"} <= patterns
    assert {"*.docx", "*.pptx", "*.mp4", "*.zip"} <= patterns


def test_public_evidence_sanitizes_user_home_paths() -> None:
    windows = r"open C:\Users\example\.docker\config.json"
    linux = "open /home/example/.docker/config.json"
    for sanitizer in (_sanitize_text, sanitize_reproduce_text):
        assert "example" not in sanitizer(windows)
        assert "example" not in sanitizer(linux)
        assert "<USER_HOME>" in sanitizer(windows)
        assert "<USER_HOME>" in sanitizer(linux)
