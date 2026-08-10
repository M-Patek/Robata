from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "docker" / "mage-traditional-codec" / "run_cv_preinfer.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("robata_test_traditional_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy() -> dict[str, object]:
    return {
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


def _write_valid_assets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(21, 42, 63)).save(directory / "canvas_000.jpg")
    np.save(directory / "src_patch_position.npy", np.asarray([[0, 0, 0]], dtype=np.int32))
    np.save(directory / "frame_ids.npy", np.asarray([0], dtype=np.int64))
    (directory / "meta.json").write_text(
        json.dumps({"fps": 30.0, "canvas_files": ["canvas_000.jpg"]}),
        encoding="utf-8",
    )


def _job(video: Path, out_dir: Path, *, job_id: str = "segment-000000") -> dict[str, object]:
    return {
        "job_id": job_id,
        "video": str(video),
        "out_dir": str(out_dir),
        "source_content_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }


def _prepare_runner(monkeypatch: pytest.MonkeyPatch, runner: ModuleType) -> None:
    monkeypatch.setattr(
        runner,
        "_require_under",
        lambda path, root, *, label: Path(path).resolve(strict=False),
    )
    monkeypatch.setattr(runner, "_video_frame_count", lambda path: 8)


def test_run_job_publishes_from_sibling_temp_and_reuses_exact_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    _prepare_runner(monkeypatch, runner)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-content")
    out_dir = tmp_path / "cache" / "segment-000000"
    observed_output_dirs: list[Path] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        temporary = Path(command[command.index("--out_dir") + 1])
        observed_output_dirs.append(temporary)
        assert temporary != out_dir
        assert temporary.parent == out_dir.parent
        assert temporary.name.startswith(f".{out_dir.name}.tmp-")
        _write_valid_assets(temporary)
        return subprocess.CompletedProcess(command, 0, stdout="prepared", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    created = runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)

    assert created["publication"]["state"] == "created"
    assert out_dir.is_dir()
    assert (out_dir / runner.PUBLICATION_MARKER_NAME).is_file()
    assert observed_output_dirs and not observed_output_dirs[0].exists()
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []

    reused = runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)

    assert reused["publication"]["state"] == "reused"
    assert reused["command"] == []
    assert len(observed_output_dirs) == 1
    assert created["output"]["loader_payload_sha256"] == reused["output"]["loader_payload_sha256"]


def test_existing_corrupt_result_fails_closed_without_overwrite_or_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    _prepare_runner(monkeypatch, runner)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-content")
    out_dir = tmp_path / "cache" / "segment-000000"
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        _write_valid_assets(Path(command[command.index("--out_dir") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)
    canvas = out_dir / "canvas_000.jpg"
    canvas.write_bytes(b"corrupt-and-must-not-be-replaced")
    corrupted = canvas.read_bytes()

    with pytest.raises(runner.TraditionalCodecRunError, match="asset validation failed"):
        runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)

    assert canvas.read_bytes() == corrupted
    assert calls == 1
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []


def test_existing_different_identity_fails_closed_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    _prepare_runner(monkeypatch, runner)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-content")
    out_dir = tmp_path / "cache" / "segment-000000"

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _write_valid_assets(Path(command[command.index("--out_dir") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)
    marker = out_dir / runner.PUBLICATION_MARKER_NAME
    before = marker.read_bytes()

    with pytest.raises(runner.TraditionalCodecRunError, match="identity differs"):
        runner._run_job(_job(video, out_dir, job_id="different-job"), policy=_policy(), timeout=30)

    assert marker.read_bytes() == before


def test_failed_preparation_removes_temp_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    _prepare_runner(monkeypatch, runner)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-content")
    out_dir = tmp_path / "cache" / "segment-000000"

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        temporary = Path(command[command.index("--out_dir") + 1])
        temporary.joinpath("partial.bin").write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 17, stdout="", stderr="failed")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(runner.TraditionalCodecRunError, match="cv-preinfer failed"):
        runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)

    assert not out_dir.exists()
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []


def test_publication_race_reuses_valid_winner_and_never_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    _prepare_runner(monkeypatch, runner)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-content")
    out_dir = tmp_path / "cache" / "segment-000000"

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _write_valid_assets(Path(command[command.index("--out_dir") + 1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def race_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_rename_directory_create_if_absent", race_publish)

    result = runner._run_job(_job(video, out_dir), policy=_policy(), timeout=30)

    assert result["publication"]["state"] == "reused_after_race"
    assert out_dir.is_dir()
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []


def test_renameat2_unsupported_uses_guarded_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    calls: list[tuple[Path, Path]] = []

    class _FakeRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            import ctypes

            ctypes.set_errno(runner.errno.EINVAL)
            return -1

    class _FakeLibc:
        renameat2 = _FakeRename()

    monkeypatch.setattr(runner.os, "name", "posix")
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.ctypes, "CDLL", lambda *_args, **_kwargs: _FakeLibc())
    monkeypatch.setattr(
        runner,
        "_flock_guarded_directory_rename",
        lambda *, source, destination: calls.append((source, destination)),
    )

    runner._rename_directory_create_if_absent(source, destination)
    assert calls == [(source, destination)]


def test_guarded_fallback_does_not_replace_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    if runner.fcntl is None:
        pytest.skip("fcntl is unavailable on this host")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "asset.bin").write_bytes(b"new")
    destination.mkdir()
    (destination / "asset.bin").write_bytes(b"old")

    with pytest.raises(FileExistsError):
        runner._flock_guarded_directory_rename(source=source, destination=destination)

    assert (destination / "asset.bin").read_bytes() == b"old"
    assert source.exists()
