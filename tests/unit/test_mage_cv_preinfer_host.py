from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mage_cv_preinfer_host.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("robata_test_mage_cv_preinfer_host", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(module: ModuleType, tmp_path: Path) -> object:
    video = tmp_path / "segment.mp4"
    video.write_bytes(b"video-bytes")
    out = tmp_path / "out"
    arguments = module._parser().parse_args(
        [
            "--video",
            str(video),
            "--out_dir",
            str(out),
            "--num_sampled_frames",
            "8",
            "--grouping_mode",
            "readiness",
            "--group_size",
            "8",
            "--images_per_group",
            "1",
            "--patch",
            "16",
            "--max_pixels",
            "65536",
            "--readiness_sum_threshold",
            "0",
            "--min_group_frames",
            "8",
            "--max_group_frames",
            "128",
            "--avoid_keyframes",
            "--canvas_format",
            "jpg",
        ]
    )
    return module._request_from_args(arguments)


def test_request_is_strict_and_builds_mage_policy(tmp_path: Path) -> None:
    module = _load()
    request = _request(module, tmp_path)

    assert request.policy == {
        "engine": "hevc",
        "target_canvas": 1,
        "group_size": 8,
        "images_per_group": 1,
        "patch": 16,
        "max_pixels": 65_536,
        "min_group_frames": 8,
        "max_group_frames": 128,
        "canvas_format": "jpg",
        "readiness_sum_threshold": 0,
        "avoid_keyframes": True,
    }


def test_docker_command_is_networkless_pullless_and_digest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    request = _request(module, tmp_path)
    control = tmp_path / "control"
    control.mkdir()
    image = "registry.example/mage-codec@sha256:" + "a" * 64
    command = module._docker_command(request=request, image=image, control_root=control)

    assert command[0] == "docker"
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--pull=never" in command
    assert image in command
    manifest = json.loads((control / "jobs.json").read_text(encoding="utf-8"))
    assert manifest["jobs"][0]["video"] == "/input/segment.mp4"
    assert manifest["jobs"][0]["out_dir"] == "/output/assets"
    assert manifest["jobs"][0]["source_content_sha256"] == module._sha256_file(request.video)


def test_unpinned_image_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    request = _request(module, tmp_path)
    monkeypatch.setenv("MAGE_CV_PREINFER_IMAGE", "registry.example/mage-codec:latest")
    with pytest.raises(module.MageCvPreinferHostError, match="digest-pinned"):
        module._run_docker(request)


def test_unknown_backend_does_not_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    request = _request(module, tmp_path)
    monkeypatch.setenv("MAGE_CV_PREINFER_BACKEND", "auto")
    with pytest.raises(module.MageCvPreinferHostError, match="unsupported"):
        module.run_host_adapter(request)


def test_binary_backend_requires_explicit_real_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    request = _request(module, tmp_path)
    monkeypatch.setenv("MAGE_CV_PREINFER_BACKEND", "binary")
    monkeypatch.delenv("MAGE_CV_PREINFER_REAL_BIN", raising=False)
    with pytest.raises(module.MageCvPreinferHostError, match="MAGE_CV_PREINFER_REAL_BIN"):
        module.run_host_adapter(request)


def test_binary_backend_verifies_required_assets_and_does_not_hide_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    request = _request(module, tmp_path)
    fake = tmp_path / "cv-preinfer.exe"
    fake.write_bytes(b"placeholder")
    monkeypatch.setenv("MAGE_CV_PREINFER_BACKEND", "binary")
    monkeypatch.setenv("MAGE_CV_PREINFER_REAL_BIN", str(fake))

    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(module.MageCvPreinferHostError, match="required assets"):
        module.run_host_adapter(request)
    assert calls and calls[0][0] == str(fake)


def test_missing_avoid_keyframes_is_rejected(tmp_path: Path) -> None:
    module = _load()
    video = tmp_path / "segment.mp4"
    video.write_bytes(b"video")
    arguments = module._parser().parse_args(
        [
            "--video",
            str(video),
            "--out_dir",
            str(tmp_path / "out"),
            "--num_sampled_frames",
            "8",
            "--grouping_mode",
            "readiness",
            "--group_size",
            "8",
            "--images_per_group",
            "1",
            "--patch",
            "16",
            "--max_pixels",
            "65536",
            "--readiness_sum_threshold",
            "0",
            "--min_group_frames",
            "8",
            "--max_group_frames",
            "128",
            "--canvas_format",
            "jpg",
        ]
    )
    with pytest.raises(module.MageCvPreinferHostError, match="avoid_keyframes"):
        module._request_from_args(arguments)


def test_native_codec_readiness_accepts_only_explicit_digest_pinned_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from robata.inference.mage_native_codec import inspect_mage_codec_dependencies

    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"docker")
    bridge = ROOT / "scripts" / "mage_cv_preinfer_host.cmd"
    monkeypatch.setenv("CV_PREINFER_BIN", str(bridge))
    monkeypatch.setenv("MAGE_CV_PREINFER_BACKEND", "docker")
    monkeypatch.setenv("MAGE_CV_PREINFER_IMAGE", "registry.example/mage@sha256:" + "a" * 64)
    monkeypatch.setenv("MAGE_DOCKER_BIN", str(docker))

    report = inspect_mage_codec_dependencies({"engine": "hevc"})
    assert report.ready is True
    assert report.missing_assets == ()

    monkeypatch.setenv("MAGE_CV_PREINFER_IMAGE", "registry.example/mage:latest")
    blocked = inspect_mage_codec_dependencies({"engine": "hevc"})
    assert blocked.ready is False
    assert blocked.blocker_code == "MAGE_CV_PREINFER_HOST_BRIDGE_NOT_READY"
    assert "digest-pinned MAGE_CV_PREINFER_IMAGE" in blocked.missing_assets


def test_container_publication_marker_is_not_forwarded_to_mage_loader(
    tmp_path: Path,
) -> None:
    module = _load()
    source = tmp_path / "container-assets"
    destination = tmp_path / "mage-assets"
    source.mkdir()
    (source / "meta.json").write_text("{}", encoding="utf-8")
    (source / "src_patch_position.npy").write_bytes(b"npy")
    (source / ".robata-publication.json").write_text("{}", encoding="utf-8")

    module._copy_completed_assets(source=source, destination=destination)

    assert (destination / "meta.json").is_file()
    assert (destination / "src_patch_position.npy").is_file()
    assert not (destination / ".robata-publication.json").exists()
