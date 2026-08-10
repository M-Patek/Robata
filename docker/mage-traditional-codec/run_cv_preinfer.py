"""Run and receipt Mage's traditional H.264/HEVC cv-preinfer preparation.

This image intentionally validates only the asset contract consumed by Mage's
``_load_codec_result`` helper. It does not load the Mage model or claim semantic
quality parity with the DCVC path.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - native Windows uses os.rename directly
    fcntl = None  # type: ignore[assignment]

import cv2
import numpy as np
from PIL import Image

TOOLCHAIN_MANIFEST = Path("/opt/robata/toolchain.json")
RUNNER_PATH = Path(__file__).resolve()
PUBLICATION_MARKER_NAME = ".robata-publication.json"
PUBLICATION_SCHEMA_NAME = "robata-mage-traditional-codec-publication"
PUBLICATION_SCHEMA_VERSION = 1
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class TraditionalCodecRunError(RuntimeError):
    """Raised when an input or generated asset fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraditionalCodecRunError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TraditionalCodecRunError(f"{label} must be a JSON object: {path}")
    return value


def _required_int(mapping: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraditionalCodecRunError(f"policy.{key} must be an integer >= {minimum}")
    return value


def _validate_policy(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TraditionalCodecRunError("policy must be an object")
    expected = {
        "engine",
        "target_canvas",
        "group_size",
        "images_per_group",
        "patch",
        "max_pixels",
        "min_group_frames",
        "max_group_frames",
        "canvas_format",
        "readiness_sum_threshold",
        "avoid_keyframes",
    }
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise TraditionalCodecRunError(
            f"policy fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if raw["engine"] not in {"hevc", "traditional", "cv-preinfer"}:
        raise TraditionalCodecRunError("policy.engine must select the traditional codec path")
    if raw["canvas_format"] not in {"jpg", "png", "npy"}:
        raise TraditionalCodecRunError("policy.canvas_format is unsupported")
    if raw["readiness_sum_threshold"] != 0:
        raise TraditionalCodecRunError("policy.readiness_sum_threshold must equal Mage's value 0")
    if raw["avoid_keyframes"] is not True:
        raise TraditionalCodecRunError("policy.avoid_keyframes must be true")

    target_canvas = _required_int(raw, "target_canvas")
    group_size = _required_int(raw, "group_size")
    images_per_group = _required_int(raw, "images_per_group")
    patch = _required_int(raw, "patch")
    max_pixels = _required_int(raw, "max_pixels")
    min_group_frames = _required_int(raw, "min_group_frames")
    max_group_frames = _required_int(raw, "max_group_frames")
    if target_canvas % images_per_group:
        raise TraditionalCodecRunError("target_canvas must be divisible by images_per_group")
    if group_size % images_per_group:
        raise TraditionalCodecRunError("group_size must be divisible by images_per_group")
    if min_group_frames > max_group_frames:
        raise TraditionalCodecRunError("min_group_frames cannot exceed max_group_frames")
    return {
        "engine": "hevc",
        "target_canvas": target_canvas,
        "group_size": group_size,
        "images_per_group": images_per_group,
        "patch": patch,
        "max_pixels": max_pixels,
        "min_group_frames": min_group_frames,
        "max_group_frames": max_group_frames,
        "canvas_format": str(raw["canvas_format"]),
        "readiness_sum_threshold": 0,
        "avoid_keyframes": True,
    }


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise TraditionalCodecRunError(f"{label} must stay below {resolved_root}: {path}")
    return resolved


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if count <= 0:
        raise TraditionalCodecRunError(f"OpenCV could not determine a frame count: {path}")
    return count


def _asset_entry(path: Path, *, root: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TraditionalCodecRunError(
            f"generated asset must be a regular non-symlink file: {path}"
        )
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_mage_loader_assets(out_dir: Path, *, patch: int) -> dict[str, object]:
    meta_path = out_dir / "meta.json"
    positions_path = out_dir / "src_patch_position.npy"
    if (
        meta_path.is_symlink()
        or positions_path.is_symlink()
        or not meta_path.is_file()
        or not positions_path.is_file()
    ):
        raise TraditionalCodecRunError(
            "cv-preinfer output lacks safe regular meta.json or src_patch_position.npy"
        )
    meta = _read_json_object(meta_path, label="cv-preinfer meta")
    raw_canvas_files = meta.get("canvas_files")
    if raw_canvas_files is None:
        raw_canvas_files = []
        for extension in ("npy", "jpg", "png"):
            names = sorted(path.name for path in out_dir.glob(f"canvas_*.{extension}"))
            if names:
                raw_canvas_files = names
                break
    if not isinstance(raw_canvas_files, list) or not raw_canvas_files:
        raise TraditionalCodecRunError("Mage loader would find no canvas files")
    if not all(isinstance(name, str) and name for name in raw_canvas_files):
        raise TraditionalCodecRunError("meta.canvas_files must contain non-empty strings")

    positions = np.load(positions_path, allow_pickle=False)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise TraditionalCodecRunError(
            f"src_patch_position.npy must have shape [N,3], got {positions.shape!r}"
        )
    if not np.issubdtype(positions.dtype, np.number):
        raise TraditionalCodecRunError("src_patch_position.npy must be numeric")
    if not np.isfinite(positions).all():
        raise TraditionalCodecRunError("src_patch_position.npy contains non-finite values")

    canvas_details: list[dict[str, object]] = []
    expected_patch_rows = 0
    for raw_name in raw_canvas_files:
        name = str(raw_name)
        if Path(name).name != name:
            raise TraditionalCodecRunError(f"canvas file must be a basename: {name!r}")
        path = out_dir / name
        if not path.is_file() or path.is_symlink():
            raise TraditionalCodecRunError(f"canvas asset missing or unsafe: {path}")
        if path.suffix.lower() == ".npy":
            image_array = np.load(path, allow_pickle=False)
            image = Image.fromarray(image_array).convert("RGB")
        else:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise TraditionalCodecRunError(f"canvas has invalid dimensions: {path}")
        if width % patch or height % patch:
            raise TraditionalCodecRunError(
                f"canvas dimensions are not aligned to patch={patch}: {width}x{height}"
            )
        expected_patch_rows += (width // patch) * (height // patch)
        detail = _asset_entry(path, root=out_dir)
        detail.update({"width": width, "height": height, "mode_after_loader": "RGB"})
        canvas_details.append(detail)

    fps = float(meta.get("fps") or 30.0)
    if not math.isfinite(fps) or fps <= 0:
        raise TraditionalCodecRunError(f"Mage loader fps would be invalid: {fps!r}")
    if int(positions.shape[0]) != expected_patch_rows:
        raise TraditionalCodecRunError(
            "Mage loader assets disagree: position rows do not equal canvas patch rows "
            f"({positions.shape[0]} != {expected_patch_rows})"
        )

    meta_asset = _asset_entry(meta_path, root=out_dir)
    positions_asset = _asset_entry(positions_path, root=out_dir)
    assets = [meta_asset, positions_asset, *canvas_details]
    loader_payload_assets = [positions_asset, *canvas_details]
    normalized_meta = json.loads(json.dumps(meta))
    for volatile_key in ("out_dir", "timing_sec", "timings", "group_processing_timing_sec"):
        normalized_meta.pop(volatile_key, None)
    normalized_config = normalized_meta.get("config")
    if isinstance(normalized_config, dict):
        normalized_config.pop("out_dir", None)
    return {
        "loader_contract": "Mage _load_codec_result compatible asset shape",
        "loader_compatible": True,
        "semantic_quality_evaluated": False,
        "meta": meta,
        "normalized_loader_meta_sha256": _canonical_sha256(normalized_meta),
        "fps": fps,
        "canvas_count": len(canvas_details),
        "canvas_files": canvas_details,
        "src_positions": {
            "shape": [int(value) for value in positions.shape],
            "dtype": str(positions.dtype),
            "row_count": int(positions.shape[0]),
            "expected_canvas_patch_rows": expected_patch_rows,
        },
        "assets": assets,
        "exact_asset_set_sha256": _canonical_sha256(assets),
        "loader_payload_assets": loader_payload_assets,
        "loader_payload_sha256": _canonical_sha256(loader_payload_assets),
        "meta_contains_volatile_timing_and_output_path": True,
    }


def _command(video: Path, out_dir: Path, policy: dict[str, object], frames: int) -> list[str]:
    requested_frames = (
        int(policy["target_canvas"]) // int(policy["images_per_group"]) * int(policy["group_size"])
    )
    sampled_frames = min(requested_frames, frames)
    return [
        "cv-preinfer",
        "--video",
        str(video),
        "--out_dir",
        str(out_dir),
        "--num_sampled_frames",
        str(sampled_frames),
        "--grouping_mode",
        "readiness",
        "--group_size",
        str(policy["group_size"]),
        "--images_per_group",
        str(policy["images_per_group"]),
        "--patch",
        str(policy["patch"]),
        "--max_pixels",
        str(policy["max_pixels"]),
        "--readiness_sum_threshold",
        "0",
        "--min_group_frames",
        str(policy["min_group_frames"]),
        "--max_group_frames",
        str(policy["max_group_frames"]),
        "--avoid_keyframes",
        "--canvas_format",
        str(policy["canvas_format"]),
    ]


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _output_asset_inventory(out_dir: Path) -> list[dict[str, object]]:
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise TraditionalCodecRunError(
            f"codec output must be a regular non-symlink directory: {out_dir}"
        )
    assets: list[dict[str, object]] = []
    for path in sorted(out_dir.iterdir(), key=lambda item: item.name):
        if path.name == PUBLICATION_MARKER_NAME:
            continue
        if path.is_symlink() or not path.is_file():
            raise TraditionalCodecRunError(
                f"codec output must contain only flat regular files: {path}"
            )
        assets.append(_asset_entry(path, root=out_dir))
    if not assets:
        raise TraditionalCodecRunError("codec output asset inventory is empty")
    return assets


def _publication_identity(
    *,
    job_id: str,
    source_sha256: str,
    source_byte_count: int,
    frame_count: int,
    policy: dict[str, object],
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "source_content_sha256": source_sha256,
        "source_byte_count": source_byte_count,
        "frame_count": frame_count,
        "config_sha256": _canonical_sha256(policy),
    }


def _publication_marker(
    *,
    identity: dict[str, object],
    validation: dict[str, object],
    assets: list[dict[str, object]],
) -> dict[str, object]:
    marker: dict[str, object] = {
        "schema_name": PUBLICATION_SCHEMA_NAME,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "identity": identity,
        "output": {
            "asset_inventory": assets,
            "asset_inventory_sha256": _canonical_sha256(assets),
            "loader_payload_sha256": validation["loader_payload_sha256"],
            "normalized_loader_meta_sha256": validation["normalized_loader_meta_sha256"],
        },
    }
    marker["publication_content_sha256"] = _canonical_sha256(marker)
    return marker


def _validate_published_output(
    out_dir: Path, *, identity: dict[str, object], patch: int
) -> tuple[dict[str, object], dict[str, object]]:
    if not _path_lexists(out_dir):
        raise TraditionalCodecRunError(f"published output is missing: {out_dir}")
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise TraditionalCodecRunError(
            f"published output must be a regular non-symlink directory: {out_dir}"
        )
    marker_path = out_dir / PUBLICATION_MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise TraditionalCodecRunError(
            f"published output lacks a regular publication marker: {marker_path}"
        )
    marker = _read_json_object(marker_path, label="traditional codec publication marker")
    expected_fields = {
        "schema_name",
        "schema_version",
        "identity",
        "output",
        "publication_content_sha256",
    }
    if set(marker) != expected_fields:
        raise TraditionalCodecRunError("publication marker fields differ")
    if marker.get("schema_name") != PUBLICATION_SCHEMA_NAME:
        raise TraditionalCodecRunError("publication marker schema_name differs")
    if marker.get("schema_version") != PUBLICATION_SCHEMA_VERSION:
        raise TraditionalCodecRunError("publication marker schema_version differs")
    content_sha256 = marker.get("publication_content_sha256")
    unsigned = dict(marker)
    unsigned.pop("publication_content_sha256", None)
    if content_sha256 != _canonical_sha256(unsigned):
        raise TraditionalCodecRunError("publication marker content identity differs")
    if marker.get("identity") != identity:
        raise TraditionalCodecRunError(
            "published output identity differs; refusing to overwrite or reuse it"
        )

    try:
        validation = _validate_mage_loader_assets(out_dir, patch=patch)
        assets = _output_asset_inventory(out_dir)
    except TraditionalCodecRunError:
        raise
    except Exception as exc:
        raise TraditionalCodecRunError(
            f"published output asset validation failed: {out_dir}"
        ) from exc
    expected_output = {
        "asset_inventory": assets,
        "asset_inventory_sha256": _canonical_sha256(assets),
        "loader_payload_sha256": validation["loader_payload_sha256"],
        "normalized_loader_meta_sha256": validation["normalized_loader_meta_sha256"],
    }
    if marker.get("output") != expected_output:
        raise TraditionalCodecRunError(
            "published output differs from its immutable publication marker"
        )
    return validation, marker


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not directory_flag:
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise TraditionalCodecRunError(f"cannot fsync unsafe output tree: {root}")
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise TraditionalCodecRunError(f"cannot fsync non-regular output asset: {path}")
        mode = "rb+" if os.name == "nt" else "rb"
        with path.open(mode) as stream:
            os.fsync(stream.fileno())
    _fsync_directory(root)


def _rename_directory_create_if_absent(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform != "linux":
        raise TraditionalCodecRunError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        if error_number not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.EXDEV}:
            raise OSError(error_number, os.strerror(error_number), destination)

    _flock_guarded_directory_rename(source=source, destination=destination)


def _flock_guarded_directory_rename(*, source: Path, destination: Path) -> None:
    """Compatibility path for filesystems lacking renameat2(RENAME_NOREPLACE)."""

    if fcntl is None:
        raise TraditionalCodecRunError(
            "flock is unavailable for guarded traditional cache publication"
        )
    lock_path = destination.with_name(f".{destination.name}.publish.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise TraditionalCodecRunError(
            "could not open traditional cache publication lock"
        ) from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
        os.rename(source, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            with contextlib.suppress(OSError):
                # Publication has already reached a durable terminal state. A stale
                # sibling lock is safe and will be reused by a later guarded writer.
                lock_path.unlink(missing_ok=True)


def _job_result(
    *,
    job_id: str,
    video: Path,
    actual_sha: str,
    frames: int,
    policy: dict[str, object],
    command: list[str],
    wall_seconds: float,
    return_code: int,
    stdout: str,
    stderr: str,
    validation: dict[str, object],
    marker: dict[str, object],
    publication_state: str,
) -> dict[str, object]:
    requested_frames = (
        int(policy["target_canvas"]) // int(policy["images_per_group"]) * int(policy["group_size"])
    )
    return {
        "job_id": job_id,
        "source": {
            "path": str(video),
            "byte_count": video.stat().st_size,
            "sha256": actual_sha,
            "frame_count": frames,
        },
        "config_sha256": _canonical_sha256(policy),
        "requested_sampled_frames": requested_frames,
        "effective_sampled_frames": min(requested_frames, frames),
        "command": command,
        "wall_seconds": wall_seconds,
        "return_code": return_code,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "publication": {
            "state": publication_state,
            "marker_content_sha256": marker["publication_content_sha256"],
        },
        "output": validation,
    }


def _run_job(raw: object, *, policy: dict[str, object], timeout: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TraditionalCodecRunError("each job must be an object")
    expected = {"job_id", "video", "out_dir", "source_content_sha256"}
    if set(raw) != expected:
        raise TraditionalCodecRunError(
            f"job fields differ; missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}"
        )
    job_id = raw["job_id"]
    if not isinstance(job_id, str) or not job_id:
        raise TraditionalCodecRunError("job_id must be a non-empty string")
    raw_video = Path(str(raw["video"]))
    raw_out_dir = Path(str(raw["out_dir"]))
    expected_sha = str(raw["source_content_sha256"])
    if not raw_video.is_absolute() or not raw_out_dir.is_absolute():
        raise TraditionalCodecRunError("video and out_dir must be absolute container paths")
    if raw_video.is_symlink() or raw_out_dir.is_symlink():
        raise TraditionalCodecRunError("video and out_dir must not be symlinks")
    video = _require_under(raw_video, Path("/input"), label="video")
    out_dir = _require_under(raw_out_dir, Path("/output"), label="out_dir")
    if not video.is_file() or video.is_symlink():
        raise TraditionalCodecRunError(f"source must be a regular non-symlink file: {video}")
    actual_sha = _sha256_file(video)
    if actual_sha != expected_sha:
        raise TraditionalCodecRunError(
            f"source SHA-256 differs for {job_id}: expected {expected_sha}, got {actual_sha}"
        )
    frames = _video_frame_count(video)
    identity = _publication_identity(
        job_id=job_id,
        source_sha256=actual_sha,
        source_byte_count=video.stat().st_size,
        frame_count=frames,
        policy=policy,
    )
    if _path_lexists(out_dir):
        validation, marker = _validate_published_output(
            out_dir, identity=identity, patch=int(policy["patch"])
        )
        return _job_result(
            job_id=job_id,
            video=video,
            actual_sha=actual_sha,
            frames=frames,
            policy=policy,
            command=[],
            wall_seconds=0.0,
            return_code=0,
            stdout="",
            stderr="",
            validation=validation,
            marker=marker,
            publication_state="reused",
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.parent.is_symlink() or not out_dir.parent.is_dir():
        raise TraditionalCodecRunError(f"output parent is unsafe: {out_dir.parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent)))
    command = _command(video, temporary, policy, frames)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        wall_seconds = time.perf_counter() - started
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if result.returncode != 0:
            raise TraditionalCodecRunError(
                f"cv-preinfer failed for {job_id} rc={result.returncode}: "
                f"{(stderr or stdout)[-2000:]}"
            )
        validation = _validate_mage_loader_assets(temporary, patch=int(policy["patch"]))
        assets = _output_asset_inventory(temporary)
        if _path_lexists(temporary / PUBLICATION_MARKER_NAME):
            raise TraditionalCodecRunError(
                "cv-preinfer unexpectedly created the reserved publication marker"
            )
        marker = _publication_marker(identity=identity, validation=validation, assets=assets)
        _write_json_atomic(temporary / PUBLICATION_MARKER_NAME, marker)
        _fsync_tree(temporary)
        publication_state = "created"
        try:
            _rename_directory_create_if_absent(temporary, out_dir)
            temporary = Path()
            _fsync_directory(out_dir.parent)
        except FileExistsError:
            validation, marker = _validate_published_output(
                out_dir, identity=identity, patch=int(policy["patch"])
            )
            publication_state = "reused_after_race"
        else:
            validation, marker = _validate_published_output(
                out_dir, identity=identity, patch=int(policy["patch"])
            )
        return _job_result(
            job_id=job_id,
            video=video,
            actual_sha=actual_sha,
            frames=frames,
            policy=policy,
            command=command,
            wall_seconds=wall_seconds,
            return_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            validation=validation,
            marker=marker,
            publication_state=publication_state,
        )
    finally:
        if temporary != Path():
            shutil.rmtree(temporary, ignore_errors=True)


def _distribution_versions() -> dict[str, str]:
    names = ("codec-video-prep", "numpy", "opencv-python-headless", "Pillow")
    return {name: importlib.metadata.version(name) for name in names}


def _toolchain_evidence() -> dict[str, object]:
    manifest = _read_json_object(TOOLCHAIN_MANIFEST, label="toolchain manifest")
    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"], text=True, capture_output=True, check=True
    ).stdout.splitlines()[0]
    cv_preinfer = shutil.which("cv-preinfer")
    if cv_preinfer is None:
        raise TraditionalCodecRunError("cv-preinfer is absent from PATH")
    cv_path = Path(cv_preinfer)
    return {
        "manifest": manifest,
        "manifest_sha256": _sha256_file(TOOLCHAIN_MANIFEST),
        "runner_sha256": _sha256_file(RUNNER_PATH),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg_version_line": ffmpeg,
        "ffmpeg_binary_sha256": _sha256_file(Path(shutil.which("ffmpeg") or "/usr/bin/ffmpeg")),
        "cv_preinfer_path": cv_preinfer,
        "cv_preinfer_entrypoint_sha256": _sha256_file(cv_path),
        "installed_distributions": _distribution_versions(),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    payload = encoded + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    arguments.job_manifest = _require_under(
        arguments.job_manifest, Path("/output"), label="job manifest"
    )
    arguments.receipt = _require_under(arguments.receipt, Path("/output"), label="receipt")
    manifest = _read_json_object(arguments.job_manifest, label="job manifest")
    if manifest.get("schema_name") != "robata-mage-traditional-codec-jobs":
        raise TraditionalCodecRunError("unexpected job manifest schema_name")
    if manifest.get("schema_version") != 1:
        raise TraditionalCodecRunError("unexpected job manifest schema_version")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TraditionalCodecRunError("run_id must be a non-empty string")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise TraditionalCodecRunError("jobs must be a non-empty array")
    policy = _validate_policy(manifest.get("policy"))
    toolchain = _toolchain_evidence()

    started = time.perf_counter()
    results = [_run_job(raw, policy=policy, timeout=arguments.timeout_seconds) for raw in jobs]
    workload_wall = time.perf_counter() - started
    receipt: dict[str, object] = {
        "schema_name": "robata-mage-traditional-codec-container-receipt",
        "schema_version": 1,
        "run_id": run_id,
        "production_eligible": False,
        "scope": {
            "engine": "traditional-h264-hevc",
            "model_loaded": False,
            "semantic_generation_executed": False,
            "semantic_quality_evaluated": False,
            "claim": "cv-preinfer timing and Mage loader-compatible asset validation only",
        },
        "policy": policy,
        "policy_sha256": _canonical_sha256(policy),
        "job_manifest_sha256": _sha256_file(arguments.job_manifest),
        "toolchain": toolchain,
        "jobs": results,
        "measurement": {
            "job_count": len(results),
            "workload_wall_seconds": workload_wall,
            "per_job_wall_seconds": [float(result["wall_seconds"]) for result in results],
            "sum_job_wall_seconds": sum(float(result["wall_seconds"]) for result in results),
        },
    }
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)
    _write_json_atomic(arguments.receipt, receipt)
    print(
        json.dumps(
            {
                "receipt": str(arguments.receipt),
                "receipt_content_sha256": receipt["receipt_content_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
