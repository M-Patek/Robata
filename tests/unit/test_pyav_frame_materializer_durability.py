from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("av")

import robata.adapters.pyav_frame_materializer as frame_materializer
from robata.ports.frame_materialization import (
    FrameMaterializationError,
    FrameMaterializationErrorCode,
)


def test_normalized_rgb24_png_matches_the_direct_encoder() -> None:
    import av

    frame = av.VideoFrame(width=5, height=3, format="rgb24")
    plane = frame.planes[0]
    contents = bytearray(plane.buffer_size)
    for row in range(frame.height):
        for column in range(frame.width * 3):
            contents[row * plane.line_size + column] = (row * 29 + column * 7) % 256
    plane.update(contents)

    direct_png, width, height = frame_materializer._encode_png(frame, max_width=3)
    normalized = frame_materializer._normalize_rgb24(frame, max_width=3)
    cached_surface_png = frame_materializer._encode_png_rgb24(normalized)

    assert (normalized.width, normalized.height) == (width, height)
    assert cached_surface_png == direct_png


def test_pinned_mjpeg_encoder_is_deterministic_and_emits_a_complete_jpeg() -> None:
    import av

    frame = av.VideoFrame(width=5, height=3, format="rgb24")
    plane = frame.planes[0]
    contents = bytearray(plane.buffer_size)
    for row in range(frame.height):
        for column in range(frame.width * 3):
            contents[row * plane.line_size + column] = (row * 17 + column * 13) % 256
    plane.update(contents)
    normalized = frame_materializer._normalize_rgb24(frame, max_width=3)

    first = frame_materializer._encode_jpeg_rgb24(
        normalized,
        qscale=2,
        chroma_subsampling="yuvj420p",
    )
    second = frame_materializer._encode_jpeg_rgb24(
        normalized,
        qscale=2,
        chroma_subsampling="yuvj420p",
    )

    assert first == second
    assert first.startswith(b"\xff\xd8")
    assert first.endswith(b"\xff\xd9")


def test_write_new_file_fsyncs_and_verifies_exact_staged_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_path = tmp_path / "frame.png"
    fsync_descriptors: list[int] = []

    def record_fsync(descriptor: int) -> None:
        fsync_descriptors.append(descriptor)

    monkeypatch.setattr(frame_materializer.os, "fsync", record_fsync)

    frame_materializer._write_new_file(frame_path, b"durable PNG bytes")

    assert frame_path.read_bytes() == b"durable PNG bytes"
    assert len(fsync_descriptors) == 1


def test_verify_staged_file_rejects_corrupt_bytes(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"corrupt bytes")

    with pytest.raises(FrameMaterializationError) as raised:
        frame_materializer._verify_staged_file(frame_path, b"expected bytes")

    assert raised.value.code is FrameMaterializationErrorCode.OUTPUT_IO_ERROR


def test_publish_staging_directory_syncs_nested_staging_before_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    staging = frames_root / ".materialization.partial-fixture"
    nested_directory = staging / "camera" / "nested"
    nested_directory.mkdir(parents=True)
    (nested_directory / "frame.png").write_bytes(b"verified bytes")
    target = frames_root / "package-id"
    synchronized: list[Path] = []

    monkeypatch.setattr(
        frame_materializer,
        "_sync_directory",
        lambda path: synchronized.append(path),
    )

    frame_materializer._publish_staging_directory(staging, target)

    assert synchronized == [nested_directory, nested_directory.parent, staging, frames_root]
    assert not staging.exists()
    assert (target / "camera" / "nested" / "frame.png").read_bytes() == b"verified bytes"


def test_staging_sync_failure_never_exposes_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    staging = frames_root / ".materialization.partial-fixture"
    staging.mkdir()
    target = frames_root / "package-id"

    def fail_staging_sync(path: Path) -> None:
        assert path == staging
        raise OSError("injected staging sync failure")

    monkeypatch.setattr(frame_materializer, "_sync_directory", fail_staging_sync)

    with pytest.raises(FrameMaterializationError) as raised:
        frame_materializer._publish_staging_directory(staging, target)

    assert raised.value.code is FrameMaterializationErrorCode.OUTPUT_IO_ERROR
    assert staging.is_dir()
    assert not target.exists()


def test_parent_sync_failure_blocks_authority_after_complete_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    staging = frames_root / ".materialization.partial-fixture"
    camera_directory = staging / "camera"
    camera_directory.mkdir(parents=True)
    expected_bytes = b"verified bytes"
    (camera_directory / "frame.png").write_bytes(expected_bytes)
    target = frames_root / "package-id"
    synchronized: list[Path] = []

    def fail_parent_sync(path: Path) -> None:
        synchronized.append(path)
        if path == frames_root:
            raise OSError("injected parent sync failure")

    monkeypatch.setattr(frame_materializer, "_sync_directory", fail_parent_sync)

    with pytest.raises(FrameMaterializationError) as raised:
        frame_materializer._publish_staging_directory(staging, target)

    assert raised.value.code is FrameMaterializationErrorCode.OUTPUT_IO_ERROR
    assert synchronized == [camera_directory, staging, frames_root]
    assert not staging.exists()
    assert (target / "camera" / "frame.png").read_bytes() == expected_bytes
