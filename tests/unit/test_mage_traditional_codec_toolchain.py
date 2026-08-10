from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN_ROOT = REPOSITORY_ROOT / "docker" / "mage-traditional-codec"
BASE_IMAGE_DIGEST = "sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36"
STATIC_FFMPEG_DIGEST = "sha256:11a44711684c0b9f754c047dcd64235b8b52deab251bd0e0a86f22faa160749c"
EXPECTED_WHEEL_HASHES = {
    "codec-video-prep": (
        "0.2.5",
        "1fdf52a26a3499b915a3921926391ab78afe0bc703697eacf7da187c43bfbab6",
    ),
    "numpy": (
        "1.26.4",
        "675d61ffbfa78604709862923189bad94014bef562cc35cf61d3a07bba02a7ed",
    ),
    "opencv-python-headless": (
        "4.11.0.86",
        "0e0a27c19dd1f40ddff94976cfe43066fbbe9dfbb2ec1907d66c19caef42a57b",
    ),
    "Pillow": (
        "12.3.0",
        "78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91",
    ),
}


def test_dockerfile_pins_both_stages_and_avoids_mutable_apt_resolution() -> None:
    dockerfile = (TOOLCHAIN_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM mwader/static-ffmpeg:7.1.1@" + STATIC_FFMPEG_DIGEST + " AS ffmpeg" in dockerfile
    assert "FROM python:3.12-slim@" + BASE_IMAGE_DIGEST in dockerfile
    assert "COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg" in dockerfile
    assert "COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "apt-get" not in dockerfile


def test_toolchain_manifest_and_lock_bind_exact_external_artifacts() -> None:
    manifest = json.loads((TOOLCHAIN_ROOT / "toolchain.json").read_bytes())

    assert manifest["base_image"]["digest"] == BASE_IMAGE_DIGEST
    assert manifest["base_image"]["python_version"] == "3.12.13"
    assert manifest["static_ffmpeg_image"]["digest"] == STATIC_FFMPEG_DIGEST
    assert manifest["static_ffmpeg_image"]["version"] == "7.1.1"
    observed = {
        item["name"]: (item["version"], item["wheel_sha256"])
        for item in manifest["python_distributions"]
    }
    assert observed == EXPECTED_WHEEL_HASHES

    lock = (TOOLCHAIN_ROOT / "requirements.lock").read_text(encoding="utf-8")
    for name, (version, digest) in EXPECTED_WHEEL_HASHES.items():
        assert f"{name}=={version}" in lock
        assert f"--hash=sha256:{digest}" in lock


def test_toolchain_commits_no_wheels_and_runner_is_syntax_valid() -> None:
    wheel_files = list(TOOLCHAIN_ROOT.rglob("*.whl"))
    assert wheel_files == []

    runner_path = TOOLCHAIN_ROOT / "run_cv_preinfer.py"
    source = runner_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(runner_path))
    for required_switch in (
        "--num_sampled_frames",
        "--grouping_mode",
        "--readiness_sum_threshold",
        "--avoid_keyframes",
        "--canvas_format",
    ):
        assert required_switch in source
    assert "loader_payload_sha256" in source
    assert "normalized_loader_meta_sha256" in source
    assert re.search(r'Path\("/input"\)', source)
    assert re.search(r'Path\("/output"\)', source)
