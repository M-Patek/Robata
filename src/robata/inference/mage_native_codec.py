"""Native Mage codec preparation adapter with fail-closed boundaries.

Mage's codec backend is not a frame sampler.  It expects a complete durable
video path and a provider-produced asset directory (``meta.json``,
``src_patch_position.npy`` and canvas images).  This module owns the host
boundary around the traditional ``cv-preinfer`` provider so endpoint code does
not silently fall back to JPEG frames when codec preparation is unavailable.

The adapter is intentionally provider-neutral at the model boundary: it can
prepare and validate assets, but it does not load Mage weights or make a
semantic quality claim.  A caller must persist the returned observation and
bind the asset directory to its normal inference identity before generation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal

MAGE_CODEC_PREPARATION_VERSION: Final = "mage-native-codec-preparation-v1"
TRADITIONAL_ENGINE_NAMES: Final = frozenset({"hevc", "cv-preinfer"})
_HOST_BRIDGE_NAME: Final = "mage_cv_preinfer_host"
_DIGEST_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class MageNativeCodecError(RuntimeError):
    """A native Mage codec prerequisite or preparation failed closed."""


@dataclass(frozen=True, slots=True)
class MageCodecDependencyReport:
    """Stable, JSON-safe host readiness observation."""

    report_version: str
    engine: str
    ready: bool
    blocker_code: str | None
    configured_executable: str | None
    executable_path: str | None
    command_mode: Literal["executable", "python_module", "missing", "not_applicable"]
    command: tuple[str, ...]
    codec_video_prep_version: str | None
    codec_video_prep_importable: bool
    compressed_video_preinfer_importable: bool
    ffmpeg_path: str | None
    ffprobe_path: str | None
    missing_assets: tuple[str, ...]
    detail: str
    remediation: str
    python: str
    platform: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["missing_assets"] = list(self.missing_assets)
        return payload


@dataclass(frozen=True, slots=True)
class MageCodecAssetReport:
    """Validation facts for one provider output directory."""

    output_directory: str
    valid: bool
    required_files: tuple[str, ...]
    canvas_files: tuple[str, ...]
    metadata_keys: tuple[str, ...]
    position_shape: tuple[int, ...] | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_files"] = list(self.required_files)
        payload["canvas_files"] = list(self.canvas_files)
        payload["metadata_keys"] = list(self.metadata_keys)
        payload["position_shape"] = list(self.position_shape) if self.position_shape else None
        return payload


@dataclass(frozen=True, slots=True)
class MageCodecPreparationObservation:
    """One completed or rejected provider invocation."""

    preparation_version: str
    status: Literal["SUCCEEDED", "FAILED"]
    engine: str
    source_video: str
    output_directory: str
    command: tuple[str, ...]
    elapsed_seconds: float
    return_code: int | None
    stdout_tail: str
    stderr_tail: str
    asset_report: MageCodecAssetReport | None
    error_code: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        if self.asset_report is not None:
            payload["asset_report"] = self.asset_report.as_dict()
        return payload


def inspect_mage_codec_dependencies(
    codec_config: Mapping[str, Any],
    *,
    model_directory: Path | None = None,
) -> MageCodecDependencyReport:
    """Inspect selected native codec prerequisites without loading a model."""

    engine = codec_config.get("engine")
    normalized_engine = str(engine) if isinstance(engine, str) else "<invalid>"
    common = {
        "report_version": MAGE_CODEC_PREPARATION_VERSION,
        "engine": normalized_engine,
        "python": sys.version.split()[0],
        "platform": _platform.platform(aliased=True),
    }
    if engine in TRADITIONAL_ENGINE_NAMES:
        return _inspect_traditional(**common)
    if engine == "dcvc-rt":
        return _inspect_neural(codec_config, model_directory, **common)
    return MageCodecDependencyReport(
        **common,
        ready=False,
        blocker_code="INVALID_CODEC_ENGINE",
        configured_executable=None,
        executable_path=None,
        command_mode="not_applicable",
        command=(),
        codec_video_prep_version=None,
        codec_video_prep_importable=False,
        compressed_video_preinfer_importable=False,
        ffmpeg_path=None,
        ffprobe_path=None,
        missing_assets=("codec_config.engine",),
        detail="codec_config.engine must be 'hevc', 'cv-preinfer', or 'dcvc-rt'",
        remediation="Select a supported native Mage codec engine.",
    )


def require_mage_codec_dependencies(
    codec_config: Mapping[str, Any],
    *,
    model_directory: Path | None = None,
) -> MageCodecDependencyReport:
    """Return readiness or raise a structured fail-closed error."""

    report = inspect_mage_codec_dependencies(codec_config, model_directory=model_directory)
    if report.ready:
        return report
    missing = ", ".join(report.missing_assets) if report.missing_assets else "none"
    raise MageNativeCodecError(
        f"{report.detail} [blocker_code={report.blocker_code}; missing={missing}; "
        f"remediation={report.remediation}]"
    )


def validate_mage_codec_assets(output_directory: Path) -> MageCodecAssetReport:
    """Validate the minimum asset contract consumed by Mage's codec loader.

    Validation is deliberately independent of Pillow.  NumPy is used when
    available to reject a corrupt position array; if it is absent, the provider
    bytes remain admissible and Mage's own loader remains authoritative for
    tensor shape and patch-grid semantics.
    """

    requested_directory = Path(output_directory).expanduser()
    required = ("meta.json", "src_patch_position.npy")
    if requested_directory.is_symlink():
        return MageCodecAssetReport(
            output_directory=str(requested_directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=(),
            position_shape=None,
            detail="output directory must not be a symlink",
        )
    directory = requested_directory.resolve()
    if not directory.is_dir():
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=(),
            position_shape=None,
            detail="output directory does not exist",
        )
    missing = [
        name
        for name in required
        if not (directory / name).is_file() or (directory / name).is_symlink()
    ]
    if missing:
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=(),
            position_shape=None,
            detail=f"missing required provider assets: {', '.join(missing)}",
        )
    marker = directory / ".robata-publication.json"
    if marker.exists():
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=(),
            position_shape=None,
            detail="reserved publication marker is not accepted as a model asset",
        )
    try:
        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=(),
            position_shape=None,
            detail=f"meta.json is not valid UTF-8 JSON: {error}",
        )
    if not isinstance(metadata, dict):
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=(),
            position_shape=None,
            detail="meta.json must contain an object",
        )
    canvas_names = metadata.get("canvas_files")
    if not isinstance(canvas_names, list) or not canvas_names:
        canvas_names = sorted(
            path.name
            for pattern in ("canvas_*.jpg", "canvas_*.png", "canvas_*.npy")
            for path in directory.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    safe_canvas_names: list[str] = []
    for name in canvas_names:
        if not isinstance(name, str) or not name or Path(name).name != name:
            return MageCodecAssetReport(
                output_directory=str(directory),
                valid=False,
                required_files=required,
                canvas_files=tuple(safe_canvas_names),
                metadata_keys=tuple(sorted(str(key) for key in metadata)),
                position_shape=None,
                detail="meta.json contains an unsafe canvas file name",
            )
        candidate = directory / name
        if not candidate.is_file() or candidate.is_symlink():
            return MageCodecAssetReport(
                output_directory=str(directory),
                valid=False,
                required_files=required,
                canvas_files=tuple(safe_canvas_names),
                metadata_keys=tuple(sorted(str(key) for key in metadata)),
                position_shape=None,
                detail=f"canvas asset is missing or symlinked: {name}",
            )
        if candidate.stat().st_size <= 0:
            return MageCodecAssetReport(
                output_directory=str(directory),
                valid=False,
                required_files=required,
                canvas_files=tuple(safe_canvas_names),
                metadata_keys=tuple(sorted(str(key) for key in metadata)),
                position_shape=None,
                detail=f"canvas asset is empty: {name}",
            )
        safe_canvas_names.append(name)
    if not safe_canvas_names:
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=(),
            metadata_keys=tuple(sorted(str(key) for key in metadata)),
            position_shape=None,
            detail="provider output contains no canvas files",
        )
    position_shape: tuple[int, ...] | None = None
    try:
        import numpy as np

        loaded = np.load(directory / "src_patch_position.npy", mmap_mode="r", allow_pickle=False)
        position_shape = tuple(int(value) for value in loaded.shape)
    except ImportError:
        pass
    except (OSError, ValueError) as error:
        return MageCodecAssetReport(
            output_directory=str(directory),
            valid=False,
            required_files=required,
            canvas_files=tuple(safe_canvas_names),
            metadata_keys=tuple(sorted(str(key) for key in metadata)),
            position_shape=None,
            detail=f"src_patch_position.npy is not a readable NumPy array: {error}",
        )
    return MageCodecAssetReport(
        output_directory=str(directory),
        valid=True,
        required_files=required,
        canvas_files=tuple(safe_canvas_names),
        metadata_keys=tuple(sorted(str(key) for key in metadata)),
        position_shape=position_shape,
        detail="minimum Mage codec asset contract is present",
    )


def prepare_traditional_codec(
    *,
    video_path: Path,
    output_directory: Path,
    codec_config: Mapping[str, Any],
    timeout_seconds: int = 7_200,
    replace_existing: bool = False,
) -> MageCodecPreparationObservation:
    """Run ``cv-preinfer`` once and publish only validated assets.

    The provider writes into a private sibling directory.  An output directory
    is atomically renamed into place only after validation succeeds.  Existing
    output is never silently replaced unless ``replace_existing=True``.
    """

    source = Path(video_path).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    started = time.perf_counter()
    report: MageCodecDependencyReport
    try:
        if not source.is_file() or source.is_symlink():
            raise MageNativeCodecError(f"source video is not a regular file: {source}")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0
        ):
            raise MageNativeCodecError("timeout_seconds must be a positive integer")
        report = require_mage_codec_dependencies(codec_config)
        command = build_traditional_codec_command(
            report=report,
            video_path=source,
            output_directory=destination,
            codec_config=codec_config,
        )
        if destination.exists() and not replace_existing:
            raise MageNativeCodecError(f"output directory already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=_codec_environment(report),
            )
            stdout_tail = (completed.stdout or "")[-4_000:]
            stderr_tail = (completed.stderr or "")[-4_000:]
            if completed.returncode != 0:
                raise _PreparationProcessError(
                    f"cv-preinfer exited with return code {completed.returncode}",
                    return_code=completed.returncode,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                )
            assets = validate_mage_codec_assets(temp_directory)
            if not assets.valid:
                raise _PreparationProcessError(
                    f"cv-preinfer produced invalid assets: {assets.detail}",
                    return_code=completed.returncode,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                    asset_report=assets,
                )
            if destination.exists():
                shutil.rmtree(destination)
            temp_directory.replace(destination)
            elapsed = float(time.perf_counter() - started)
            return MageCodecPreparationObservation(
                preparation_version=MAGE_CODEC_PREPARATION_VERSION,
                status="SUCCEEDED",
                engine=str(codec_config.get("engine")),
                source_video=str(source),
                output_directory=str(destination),
                command=tuple(command),
                elapsed_seconds=elapsed,
                return_code=completed.returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                asset_report=validate_mage_codec_assets(destination),
                error_code=None,
            )
        except subprocess.TimeoutExpired as error:
            raise _PreparationProcessError(
                f"cv-preinfer timed out after {timeout_seconds}s",
                return_code=None,
                stdout_tail=_tail(error.stdout),
                stderr_tail=_tail(error.stderr),
            ) from error
        finally:
            if temp_directory.exists():
                shutil.rmtree(temp_directory, ignore_errors=True)
    except Exception as error:
        elapsed = float(time.perf_counter() - started)
        command = tuple(locals().get("command", ()))
        process_error = error if isinstance(error, _PreparationProcessError) else None
        return MageCodecPreparationObservation(
            preparation_version=MAGE_CODEC_PREPARATION_VERSION,
            status="FAILED",
            engine=str(codec_config.get("engine")),
            source_video=str(source),
            output_directory=str(destination),
            command=command,
            elapsed_seconds=elapsed,
            return_code=process_error.return_code if process_error else None,
            stdout_tail=process_error.stdout_tail if process_error else "",
            stderr_tail=process_error.stderr_tail if process_error else "",
            asset_report=process_error.asset_report if process_error else None,
            error_code=_error_code(error),
        )


def build_traditional_codec_command(
    *,
    report: MageCodecDependencyReport,
    video_path: Path,
    output_directory: Path,
    codec_config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Build the pinned cv-preinfer command without executing it."""

    if not report.ready or report.engine not in TRADITIONAL_ENGINE_NAMES:
        raise MageNativeCodecError(
            "traditional codec command requested while dependencies are not ready"
        )

    def positive(name: str, default: int) -> int:
        value = codec_config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MageNativeCodecError(f"codec_config.{name} must be a positive integer")
        return int(value)

    executable = report.command
    return (
        *executable,
        "--video",
        str(video_path),
        "--out_dir",
        str(output_directory),
        "--num_sampled_frames",
        str(
            positive(
                "num_sampled_frames",
                positive("target_canvas", 32)
                * positive("group_size", 32)
                // positive("images_per_group", 4),
            )
        ),
        "--grouping_mode",
        "readiness",
        "--group_size",
        str(positive("group_size", 32)),
        "--images_per_group",
        str(positive("images_per_group", 4)),
        "--patch",
        str(positive("patch", positive("patch_size", 16))),
        "--max_pixels",
        str(positive("max_pixels", 150_000)),
        "--readiness_sum_threshold",
        "0",
        "--min_group_frames",
        str(positive("min_group_frames", 8)),
        "--max_group_frames",
        str(positive("max_group_frames", 64)),
        "--avoid_keyframes",
        "--canvas_format",
        "jpg",
    )


def _inspect_traditional(**common: Any) -> MageCodecDependencyReport:
    configured = os.environ.get("CV_PREINFER_BIN", "cv-preinfer").strip() or "cv-preinfer"
    direct = _resolve_executable(configured)
    host_bridge = _is_docker_host_bridge(configured)
    ffmpeg = None if host_bridge else _resolve_executable("ffmpeg")
    ffprobe = None if host_bridge else _resolve_executable("ffprobe")
    codec_importable = _module_available("codec_video_prep")
    preinfer_importable = _module_available("compressed_video_preinfer")
    package_version = _distribution_version("codec-video-prep")
    # Mage's processor launches ``CV_PREINFER_BIN`` as an executable.  An
    # importable Python package is useful diagnostic evidence, but it is not a
    # valid substitute: reporting a module command as ready would leave the
    # processor's own subprocess lookup pointed at the missing console script.
    # Keep the command empty until an actual executable (or the explicit host
    # bridge wrapper) is discoverable.
    command_mode: Literal["executable", "python_module", "missing"]
    command: tuple[str, ...]
    if direct is not None:
        command = (str(direct),)
        command_mode = "executable"
        executable_path = str(direct)
    else:
        command = ()
        command_mode = "missing"
        executable_path = None
    missing: list[str] = []
    if host_bridge:
        if direct is None:
            missing.append("cv-preinfer host bridge")
        docker = _resolve_executable(os.environ.get("MAGE_DOCKER_BIN", "docker"))
        image = os.environ.get("MAGE_CV_PREINFER_IMAGE", "").strip()
        if docker is None:
            missing.append("docker")
        if not _DIGEST_IMAGE.fullmatch(image):
            missing.append("digest-pinned MAGE_CV_PREINFER_IMAGE")
    elif direct is None:
        missing.append("cv-preinfer")
    else:
        if ffmpeg is None:
            missing.append("ffmpeg")
        if ffprobe is None:
            missing.append("ffprobe")
    ready = not missing
    if ready and host_bridge:
        detail = "digest-pinned Docker cv-preinfer host bridge is ready"
        blocker = None
        remediation = "Run a bounded native smoke before quality qualification."
    elif ready:
        detail = "traditional Mage codec prerequisites are discoverable"
        blocker = None
        remediation = "Run a bounded native smoke before quality qualification."
    elif host_bridge:
        detail = "explicit Docker cv-preinfer host bridge is incomplete"
        blocker = "MAGE_CV_PREINFER_HOST_BRIDGE_NOT_READY"
        remediation = (
            "Set CV_PREINFER_BIN to the host bridge, provide MAGE_CV_PREINFER_IMAGE "
            "as a digest-pinned image, and make Docker available."
        )
    elif not command and (codec_importable or preinfer_importable):
        detail = "codec-video-prep is importable but no usable cv-preinfer entrypoint was found"
        blocker = "CV_PREINFER_ENTRYPOINT_MISSING"
        remediation = (
            "Expose the cv-preinfer console script or install a compatible CPython module."
        )
    elif not command:
        detail = "Mage traditional codec preprocessing requires cv-preinfer"
        blocker = "MISSING_CV_PREINFER"
        remediation = "Install codec-video-prep==0.2.5 in the pinned Linux CPython 3.12 toolchain."
    else:
        detail = "cv-preinfer is present but ffmpeg/ffprobe are incomplete"
        blocker = "MISSING_FFMPEG_RUNTIME"
        remediation = "Put matching ffmpeg and ffprobe binaries on PATH."
    return MageCodecDependencyReport(
        **common,
        ready=ready,
        blocker_code=blocker,
        configured_executable=configured,
        executable_path=executable_path,
        command_mode=command_mode,
        command=command,
        codec_video_prep_version=package_version,
        codec_video_prep_importable=codec_importable,
        compressed_video_preinfer_importable=preinfer_importable,
        ffmpeg_path=str(ffmpeg) if ffmpeg else None,
        ffprobe_path=str(ffprobe) if ffprobe else None,
        missing_assets=tuple(missing),
        detail=detail,
        remediation=remediation,
    )


def _inspect_neural(
    codec_config: Mapping[str, Any], model_directory: Path | None, **common: Any
) -> MageCodecDependencyReport:
    raw = codec_config.get("dcvc")
    if not isinstance(raw, Mapping):
        return MageCodecDependencyReport(
            **common,
            ready=False,
            blocker_code="MISSING_DCVC_CONFIG",
            configured_executable=None,
            executable_path=None,
            command_mode="not_applicable",
            command=(),
            codec_video_prep_version=None,
            codec_video_prep_importable=False,
            compressed_video_preinfer_importable=False,
            ffmpeg_path=None,
            ffprobe_path=None,
            missing_assets=("codec_config.dcvc",),
            detail="neural codec configuration is missing",
            remediation="Provide dcvc package and checkpoint paths.",
        )
    root = Path(
        str(raw.get("pkg_dir") or ((model_directory or Path.cwd()) / "neural_codec"))
    ).expanduser()
    expected = {
        "dcvc_readiness_gen.py": root / "dcvc_readiness_gen.py",
        "DCVC/src": root / "DCVC" / "src",
        "dcvc_rt_intra.tar": Path(str(raw.get("intra_ckpt") or (root / "dcvc_rt_intra.tar"))),
        "dcvc_rt_inter.tar": Path(str(raw.get("inter_ckpt") or (root / "dcvc_rt_inter.tar"))),
    }
    missing = tuple(name for name, path in expected.items() if not path.exists())
    return MageCodecDependencyReport(
        **common,
        ready=not missing,
        blocker_code=None if not missing else "MISSING_DCVC_ASSET",
        configured_executable=None,
        executable_path=None,
        command_mode="not_applicable",
        command=(),
        codec_video_prep_version=None,
        codec_video_prep_importable=False,
        compressed_video_preinfer_importable=False,
        ffmpeg_path=None,
        ffprobe_path=None,
        missing_assets=missing,
        detail="neural Mage codec assets are discoverable"
        if not missing
        else "neural codec assets are incomplete",
        remediation="Run a native neural smoke before qualification"
        if not missing
        else "Restore the complete neural_codec bundle.",
    )


def _is_docker_host_bridge(configured: str) -> bool:
    """Return whether the explicit executable is Robata's Docker bridge.

    The bridge is opt-in through ``MAGE_CV_PREINFER_BACKEND=docker``.  Merely
    naming a similarly named file never relaxes the native ffmpeg/ffprobe
    checks, which keeps arbitrary wrappers from becoming a readiness bypass.
    """

    if os.environ.get("MAGE_CV_PREINFER_BACKEND", "").strip().lower() != "docker":
        return False
    return Path(configured).stem.lower() == _HOST_BRIDGE_NAME


def _resolve_executable(value: str) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.is_file() and not candidate.is_symlink():
        return candidate.resolve()
    found = shutil.which(value)
    return Path(found).resolve() if found else None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except (PackageNotFoundError, ValueError):
        return None


def _codec_environment(report: MageCodecDependencyReport) -> dict[str, str]:
    env = dict(os.environ)
    if report.executable_path:
        env["CV_PREINFER_BIN"] = report.executable_path
    return env


def _tail(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")[-4_000:]
    return value[-4_000:]


class _PreparationProcessError(MageNativeCodecError):
    def __init__(
        self,
        message: str,
        *,
        return_code: int | None,
        stdout_tail: str,
        stderr_tail: str,
        asset_report: MageCodecAssetReport | None = None,
    ) -> None:
        super().__init__(message)
        self.return_code = return_code
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.asset_report = asset_report


def _error_code(error: Exception) -> str:
    if isinstance(error, _PreparationProcessError):
        if error.return_code is None:
            return "CV_PREINFER_TIMEOUT"
        return "CV_PREINFER_FAILED"
    if isinstance(error, MageNativeCodecError):
        message = str(error)
        if "already exists" in message:
            return "OUTPUT_EXISTS"
        if "dependencies" in message or "cv-preinfer" in message:
            return "NATIVE_CODEC_NOT_READY"
        return "NATIVE_CODEC_INVALID_REQUEST"
    return "NATIVE_CODEC_INTERNAL_ERROR"


__all__ = [
    "MAGE_CODEC_PREPARATION_VERSION",
    "MageCodecAssetReport",
    "MageCodecDependencyReport",
    "MageCodecPreparationObservation",
    "MageNativeCodecError",
    "build_traditional_codec_command",
    "inspect_mage_codec_dependencies",
    "prepare_traditional_codec",
    "require_mage_codec_dependencies",
    "validate_mage_codec_assets",
]
