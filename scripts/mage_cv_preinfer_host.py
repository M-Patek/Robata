"""Host bridge for Mage's traditional ``cv-preinfer`` codec preparation.

Mage's trusted processor code launches one executable named by ``CV_PREINFER_BIN``.
On Windows and on ordinary developer hosts the pinned ``codec-video-prep`` runtime
is not installed natively; the supported bridge is an explicit Docker invocation.
This module is deliberately *not* a codec fallback: it either invokes the selected
pinned toolchain or fails before producing any output.

The command line mirrors the arguments emitted by Mage's
``codec_video_processing_mage_vl._run_cv_preinfer`` helper.  Set
``CV_PREINFER_BIN`` to ``mage_cv_preinfer_host.cmd`` (Windows) or the executable
wrapper (Linux/macOS), and set ``MAGE_CV_PREINFER_IMAGE`` to the digest-pinned
traditional-codec image.  Runtime network access is disabled and image pulling is
disallowed, so readiness cannot silently use a different toolchain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_DIGEST_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_HEX = frozenset("0123456789abcdef")
_DEFAULT_IMAGE_ENV = "MAGE_CV_PREINFER_IMAGE"
_BACKEND_ENV = "MAGE_CV_PREINFER_BACKEND"
_DOCKER_ENV = "MAGE_DOCKER_BIN"
_REAL_BIN_ENV = "MAGE_CV_PREINFER_REAL_BIN"


class MageCvPreinferHostError(RuntimeError):
    """The explicit host bridge could not execute its selected backend."""


@dataclass(frozen=True, slots=True)
class HostCodecRequest:
    """Validated request forwarded by Mage's remote processor code."""

    video: Path
    out_dir: Path
    num_sampled_frames: int
    grouping_mode: str
    group_size: int
    images_per_group: int
    patch: int
    max_pixels: int
    readiness_sum_threshold: int
    min_group_frames: int
    max_group_frames: int
    canvas_format: str

    @property
    def policy(self) -> dict[str, object]:
        return {
            "engine": "hevc",
            "target_canvas": self.num_sampled_frames // self.group_size * self.images_per_group,
            "group_size": self.group_size,
            "images_per_group": self.images_per_group,
            "patch": self.patch,
            "max_pixels": self.max_pixels,
            "min_group_frames": self.min_group_frames,
            "max_group_frames": self.max_group_frames,
            "canvas_format": self.canvas_format,
            "readiness_sum_threshold": self.readiness_sum_threshold,
            "avoid_keyframes": True,
        }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_sampled_frames", type=_positive_int, required=True)
    parser.add_argument("--grouping_mode", choices=("readiness",), required=True)
    parser.add_argument("--group_size", type=_positive_int, required=True)
    parser.add_argument("--images_per_group", type=_positive_int, required=True)
    parser.add_argument("--patch", type=_positive_int, required=True)
    parser.add_argument("--max_pixels", type=_positive_int, required=True)
    parser.add_argument("--readiness_sum_threshold", type=_nonnegative_int, required=True)
    parser.add_argument("--min_group_frames", type=_positive_int, required=True)
    parser.add_argument("--max_group_frames", type=_positive_int, required=True)
    parser.add_argument("--avoid_keyframes", action="store_true")
    parser.add_argument("--canvas_format", choices=("jpg", "png", "npy"), required=True)
    return parser


def _request_from_args(arguments: argparse.Namespace) -> HostCodecRequest:
    if not arguments.avoid_keyframes:
        raise MageCvPreinferHostError(
            "Mage traditional preparation requires the explicit --avoid_keyframes switch"
        )
    if arguments.group_size % arguments.images_per_group:
        raise MageCvPreinferHostError("group_size must be divisible by images_per_group")
    if arguments.num_sampled_frames < arguments.group_size:
        raise MageCvPreinferHostError(
            "num_sampled_frames must contain at least one complete codec group"
        )
    if arguments.max_group_frames < arguments.min_group_frames:
        raise MageCvPreinferHostError("max_group_frames cannot be less than min_group_frames")
    if arguments.readiness_sum_threshold != 0:
        raise MageCvPreinferHostError("readiness_sum_threshold must be zero for Mage's route")

    video = arguments.video.expanduser().resolve()
    if video.is_symlink() or not video.is_file():
        raise MageCvPreinferHostError(f"video is not a regular file: {video}")
    out_dir = arguments.out_dir.expanduser().resolve()
    if out_dir.exists() and (out_dir.is_symlink() or not out_dir.is_dir()):
        raise MageCvPreinferHostError(f"out_dir is not a regular directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return HostCodecRequest(
        video=video,
        out_dir=out_dir,
        num_sampled_frames=arguments.num_sampled_frames,
        grouping_mode=arguments.grouping_mode,
        group_size=arguments.group_size,
        images_per_group=arguments.images_per_group,
        patch=arguments.patch,
        max_pixels=arguments.max_pixels,
        readiness_sum_threshold=arguments.readiness_sum_threshold,
        min_group_frames=arguments.min_group_frames,
        max_group_frames=arguments.max_group_frames,
        canvas_format=arguments.canvas_format,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pinned_image() -> str:
    image = os.environ.get(_DEFAULT_IMAGE_ENV, "").strip()
    if not image:
        raise MageCvPreinferHostError(
            f"{_DEFAULT_IMAGE_ENV} must name the digest-pinned codec-video-prep image"
        )
    if not _DIGEST_IMAGE.fullmatch(image):
        raise MageCvPreinferHostError(
            f"{_DEFAULT_IMAGE_ENV} must use repository@sha256:<64 lowercase hex>"
        )
    return image


def _docker_command(*, request: HostCodecRequest, image: str, control_root: Path) -> list[str]:
    docker = os.environ.get(_DOCKER_ENV, "docker").strip() or "docker"
    manifest = control_root / "jobs.json"
    manifest_payload = {
        "schema_name": "robata-mage-traditional-codec-jobs",
        "schema_version": 1,
        "run_id": f"host-bridge-{request.out_dir.name}",
        "policy": request.policy,
        "jobs": [
            {
                "job_id": request.out_dir.name,
                "video": f"/input/{request.video.name}",
                "out_dir": "/output/assets",
                "source_content_sha256": _sha256_file(request.video),
            }
        ],
    }
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    # The runner atomically publishes into /output/assets.  The host then copies
    # only that completed tree into Mage's temporary directory.
    return [
        docker,
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--platform",
        "linux/amd64",
        "--mount",
        f"type=bind,source={request.video.parent},target=/input,readonly",
        "--mount",
        f"type=bind,source={control_root},target=/output",
        image,
        "--job-manifest",
        "/output/jobs.json",
        "--receipt",
        "/output/receipt.json",
        "--timeout-seconds",
        str(_timeout_seconds()),
    ]


def _copy_completed_assets(*, source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise MageCvPreinferHostError(
            "codec container completed without a regular output directory"
        )
    required = ("meta.json", "src_patch_position.npy")
    if any(not (source / name).is_file() for name in required):
        raise MageCvPreinferHostError(
            "codec container completed without Mage's required meta.json and "
            "src_patch_position.npy assets"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        # The container receipt runner owns this publication marker; it is
        # container evidence, not a Mage model-loader asset.
        if item.name == ".robata-publication.json":
            continue
        target = destination / item.name
        if item.is_symlink():
            raise MageCvPreinferHostError(f"codec output contains a symlink: {item.name}")
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _timeout_seconds() -> int:
    raw = os.environ.get("MAGE_CV_PREINFER_TIMEOUT_SECONDS", "7200").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise MageCvPreinferHostError(
            "MAGE_CV_PREINFER_TIMEOUT_SECONDS must be a positive integer"
        ) from error
    if value <= 0:
        raise MageCvPreinferHostError(
            "MAGE_CV_PREINFER_TIMEOUT_SECONDS must be a positive integer"
        )
    return value


def _run_docker(request: HostCodecRequest) -> None:
    image = _pinned_image()
    with tempfile.TemporaryDirectory(
        prefix=".mage-cv-preinfer-", dir=request.out_dir.parent
    ) as raw:
        control_root = Path(raw)
        command = _docker_command(request=request, image=image, control_root=control_root)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_timeout_seconds(),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MageCvPreinferHostError(
                f"explicit Docker codec preparation could not start: {error}"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no container output")[-4000:]
            raise MageCvPreinferHostError(
                f"explicit Docker codec preparation failed with rc={completed.returncode}: {detail}"
            )
        _copy_completed_assets(source=control_root / "assets", destination=request.out_dir)


def _run_binary(request: HostCodecRequest) -> None:
    configured = os.environ.get(_REAL_BIN_ENV, "").strip()
    if not configured:
        raise MageCvPreinferHostError(
            f"{_REAL_BIN_ENV} is required when {_BACKEND_ENV}=binary; refusing implicit fallback"
        )
    executable = Path(configured).expanduser()
    if not executable.is_file() and shutil.which(configured) is None:
        raise MageCvPreinferHostError(
            f"configured real cv-preinfer executable is missing: {configured}"
        )
    command = [
        configured,
        "--video",
        str(request.video),
        "--out_dir",
        str(request.out_dir),
        "--num_sampled_frames",
        str(request.num_sampled_frames),
        "--grouping_mode",
        request.grouping_mode,
        "--group_size",
        str(request.group_size),
        "--images_per_group",
        str(request.images_per_group),
        "--patch",
        str(request.patch),
        "--max_pixels",
        str(request.max_pixels),
        "--readiness_sum_threshold",
        str(request.readiness_sum_threshold),
        "--min_group_frames",
        str(request.min_group_frames),
        "--max_group_frames",
        str(request.max_group_frames),
        "--avoid_keyframes",
        "--canvas_format",
        request.canvas_format,
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        raise MageCvPreinferHostError(f"configured cv-preinfer could not start: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no executable output")[-4000:]
        raise MageCvPreinferHostError(
            f"configured cv-preinfer failed with rc={completed.returncode}: {detail}"
        )
    if not (request.out_dir / "meta.json").is_file() or not (
        request.out_dir / "src_patch_position.npy"
    ).is_file():
        raise MageCvPreinferHostError(
            "configured cv-preinfer returned success without Mage's required assets"
        )


def run_host_adapter(request: HostCodecRequest) -> None:
    """Run only the explicitly selected backend; never fall back silently."""

    backend = os.environ.get(_BACKEND_ENV, "docker").strip().lower() or "docker"
    if backend == "docker":
        _run_docker(request)
        return
    if backend == "binary":
        _run_binary(request)
        return
    raise MageCvPreinferHostError(
        f"unsupported {_BACKEND_ENV}={backend!r}; choose exactly 'docker' or 'binary'"
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        request = _request_from_args(arguments)
        run_host_adapter(request)
    except (MageCvPreinferHostError, OSError, ValueError) as error:
        print(f"mage-cv-preinfer-host: {error}", file=sys.stderr)
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
