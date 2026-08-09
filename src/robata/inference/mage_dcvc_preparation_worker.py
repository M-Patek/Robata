"""Resident, endpoint-external DCVC preparation worker for Mage native codec assets.

The worker is intentionally single-device and single-job. It keeps one DCVC engine
resident, resets recurrent state for every segment, commits outputs by atomic
staging-directory rename, and never treats prepared assets as authoritative inference
state. JSONL over stdin/stdout is the initial local IPC boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout, suppress
from importlib.machinery import ModuleSpec
from pathlib import Path, PurePosixPath
from threading import Lock
from types import ModuleType
from typing import Any, Final, Literal, Protocol, TextIO

from pydantic import JsonValue, ValidationError

from robata.contracts.common import Sha256Digest
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.inference import device_execution_guard as _device_guard
from robata.inference import mage_dcvc_preparation_protocol as _protocol
from robata.inference.device_execution_guard import (
    DeviceExecutionGuard,
    DeviceExecutionGuardBusy,
    ExclusiveFileDeviceGuard,
)
from robata.inference.mage_dcvc_preparation_protocol import (
    MAGE_DCVC_PREPARATION_SIDECAR_NAME,
    MAGE_DCVC_PROVIDER_VERSION,
    MAGE_DCVC_RECIPE_VERSION,
    MAGE_DCVC_TEMP_MARKER_NAME,
    MageDcvcEffectiveConfig,
    MageDcvcPreparationArtifact,
    MageDcvcPreparationRequest,
    MageDcvcPreparationResponse,
    MageDcvcPreparedAsset,
    MageDcvcTempMarker,
    mage_dcvc_artifact_semantic_sha256,
    mage_dcvc_preparation_identity,
)

_PROVIDER_REQUIRED_RELATIVE_FILES: Final = (
    "neural_codec/codec_dcvc_config.py",
    "neural_codec/dcvc_readiness_gen.py",
    "neural_codec/dcvc_rt_engine.py",
    "neural_codec/codec_tools/pipeline/process_video_bitcost_readiness.py",
    "neural_codec/codec_tools/pipeline/process_video_bitcost_mv_mask_collage.py",
    "neural_codec/codec_tools/pipeline/generate_codec_patch_smart_resize.py",
)
_PROVIDER_RECURSIVE_ROOTS: Final = (
    "neural_codec/DCVC/src",
    "neural_codec/codec_tools/codec_patch_gop",
)
_PROVIDER_SOURCE_SUFFIXES: Final = frozenset({".py", ".cpp", ".cu", ".c", ".h", ".hpp"})
_PROVIDER_META_KEY: Final = "robata_dcvc_provider"
_WINDOWS_LEGACY_MAX_PATH: Final = 260


class MageDcvcPreparationError(RuntimeError):
    """Base fail-closed error for the resident preparation boundary."""

    code = "DCVC_PREPARATION_FAILED"


class MageDcvcPreparationRejected(MageDcvcPreparationError):
    """The request or durable state did not match the worker contract."""

    code = "DCVC_PREPARATION_REJECTED"


class MageDcvcPreparationBusy(MageDcvcPreparationError):
    """A second job or a conflicting device holder prevented admission."""

    code = "DCVC_PREPARATION_BUSY"


class MageDcvcBackendError(MageDcvcPreparationError):
    """The external Mage/DCVC provider failed to produce valid assets."""

    code = "DCVC_PROVIDER_FAILED"


class MageDcvcPreparationBackend(Protocol):
    """One process-resident provider implementation."""

    @property
    def effective_config_sha256(self) -> Sha256Digest:
        """Return the exact configuration loaded by the resident engine."""

    def prepare(
        self,
        *,
        source_path: Path,
        output_directory: Path,
    ) -> Mapping[str, JsonValue]:
        """Populate one empty staging directory and return provider metadata."""

    def close(self) -> None:
        """Release optional provider resources."""


MageDcvcDeviceGuard = DeviceExecutionGuard


class PersistentMageDcvcPreparationBackend:
    """One version-bound readiness provider with a process-resident DCVC engine.

    Production configuration is installed as a generated ``codec_dcvc_config``
    overlay *before* any readiness module is imported. The bundled
    ``dcvc_readiness_gen`` remains responsible for its intended score-source wiring;
    Robata never overwrites configuration globals on an already-imported pipeline.
    The module-global upstream engine is forced to load once at worker startup and
    ``_dcvc_bitmaps`` resets its sequence for every segment.
    """

    def __init__(
        self,
        *,
        effective_config: MageDcvcEffectiveConfig,
        model_directory: Path,
        provider_state_root: Path,
        intra_checkpoint_path: Path | None = None,
        inter_checkpoint_path: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = effective_config
        self._model_directory = Path(model_directory).expanduser().resolve()
        self._neural_root = (self._model_directory / "neural_codec").resolve()
        self._intra_checkpoint = (
            Path(intra_checkpoint_path or self._neural_root / "dcvc_rt_intra.tar")
            .expanduser()
            .resolve()
        )
        self._inter_checkpoint = (
            Path(inter_checkpoint_path or self._neural_root / "dcvc_rt_inter.tar")
            .expanduser()
            .resolve()
        )
        self._verify_installation()
        self._overlay_root = _install_dcvc_config_overlay(
            state_root=provider_state_root,
            config=self._config,
        )
        environment = self._provider_environment()
        with _temporary_environment(environment), redirect_stdout(sys.stderr):
            self._readiness = _load_external_provider(
                neural_root=self._neural_root,
                overlay_root=self._overlay_root,
            )
        self._cv2 = _load_cv2()
        self._job_lock = Lock()
        if not callable(getattr(self._readiness, "_get_engine", None)):
            raise MageDcvcBackendError("readiness provider exposes no _get_engine")
        self._engine: Any | None = None
        self._clock = clock
        self._engine_load_count = 0
        self._engine_load_seconds = 0.0
        self._completed_job_count = 0

    @property
    def effective_config_sha256(self) -> Sha256Digest:
        return self._config.effective_config_sha256

    def prepare(
        self,
        *,
        source_path: Path,
        output_directory: Path,
    ) -> Mapping[str, JsonValue]:
        if not self._job_lock.acquire(blocking=False):
            raise MageDcvcPreparationBusy("resident DCVC backend already has an active job")
        try:
            source = Path(source_path).expanduser().resolve()
            output = Path(output_directory).expanduser().resolve()
            if not source.is_file():
                raise MageDcvcBackendError("source segment does not exist")
            if not output.is_dir():
                raise MageDcvcBackendError("provider staging directory does not exist")
            with redirect_stdout(sys.stderr):
                self._ensure_engine_loaded()
                sampled_frame_count, total_frames = self._effective_sampled_frame_count(source)
                self._run_readiness(
                    source_path=source,
                    output_directory=output,
                    sampled_frame_count=sampled_frame_count,
                )
            return self._annotate_provider_meta(
                output,
                effective_sampled_frame_count=sampled_frame_count,
                max_encoded_frame_id=total_frames - 1,
            )
        except MageDcvcPreparationError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise MageDcvcBackendError("resident DCVC preparation failed") from error
        finally:
            self._job_lock.release()

    def close(self) -> None:
        if self._engine is not None and getattr(self._readiness, "_ENGINE", None) is self._engine:
            self._readiness._ENGINE = None  # type: ignore[attr-defined]
        self._engine = None

    def _ensure_engine_loaded(self) -> None:
        if self._engine is not None:
            return
        get_engine = getattr(self._readiness, "_get_engine", None)
        if not callable(get_engine):
            raise MageDcvcBackendError("readiness provider exposes no _get_engine")
        started = self._clock()
        try:
            with _temporary_environment(self._provider_environment()):
                self._engine = get_engine()
        except Exception as error:
            raise MageDcvcBackendError("could not load the resident DCVC engine") from error
        self._engine_load_seconds = max(0.0, float(self._clock() - started))
        self._engine_load_count += 1
        if self._engine_load_count != 1:
            raise MageDcvcBackendError("resident DCVC engine loaded more than once")

    def _verify_installation(self) -> None:
        observed_implementation = build_mage_dcvc_provider_implementation_sha256(
            self._model_directory
        )
        if observed_implementation != self._config.provider_implementation_sha256:
            raise MageDcvcBackendError("provider implementation digest does not match config")
        intra_sha256, _ = _exact_file_sha256(self._intra_checkpoint)
        inter_sha256, _ = _exact_file_sha256(self._inter_checkpoint)
        if intra_sha256 != self._config.intra_checkpoint_sha256:
            raise MageDcvcBackendError("DCVC intra checkpoint digest does not match config")
        if inter_sha256 != self._config.inter_checkpoint_sha256:
            raise MageDcvcBackendError("DCVC inter checkpoint digest does not match config")

    def _provider_environment(self) -> dict[str, str]:
        return {
            "DCVC_INTRA_TAR": str(self._intra_checkpoint),
            "DCVC_INTER_TAR": str(self._inter_checkpoint),
            "DCVC_DEVICE": self._config.preparation_device,
            "DCVC_ENGINE_DIR": str(self._neural_root),
            "DCVC_REPO_DIR": str(self._neural_root),
        }

    def _effective_sampled_frame_count(self, source_path: Path) -> tuple[int, int]:
        capture = self._cv2.VideoCapture(str(source_path))
        try:
            if not bool(capture.isOpened()):
                raise MageDcvcBackendError("OpenCV could not open the source segment")
            total_frames = int(capture.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
        if total_frames <= 0:
            raise MageDcvcBackendError("source segment reports no video frames")
        return min(self._config.sampled_frame_count, total_frames), total_frames

    def _run_readiness(
        self,
        *,
        source_path: Path,
        output_directory: Path,
        sampled_frame_count: int,
    ) -> None:
        main = getattr(self._readiness, "main", None)
        if not callable(main):
            raise MageDcvcBackendError("readiness provider exposes no callable main")
        argv = self._pipeline_argv(
            source_path=source_path,
            output_directory=output_directory,
            sampled_frame_count=sampled_frame_count,
        )
        with _temporary_argv(argv), _temporary_environment(self._provider_environment()):
            main()

    def _pipeline_argv(
        self,
        *,
        source_path: Path,
        output_directory: Path,
        sampled_frame_count: int,
    ) -> list[str]:
        config = self._config
        return [
            "robata-mage-dcvc-provider-v2",
            "--video",
            str(source_path),
            "--out_dir",
            str(output_directory),
            "--num_sampled_frames",
            str(sampled_frame_count),
            "--grouping_mode",
            config.grouping_mode,
            "--readiness_sum_threshold_mode",
            config.readiness_sum_threshold_mode,
            "--group_size",
            str(config.group_size),
            "--images_per_group",
            str(config.images_per_group),
            "--patch",
            str(config.patch),
            "--max_pixels",
            str(config.max_pixels),
            "--min_group_frames",
            str(config.min_group_frames),
            "--max_group_frames",
            str(config.max_group_frames),
            "--readiness_coverage_bins",
            str(config.readiness_coverage_bins),
            "--readiness_delta_ratio",
            str(config.readiness_delta_ratio),
            "--bitcost_grid",
            config.bitcost_grid,
            "--bitcost_pct",
            str(config.bitcost_percentile),
            "--decode_backsearch_max",
            str(config.decode_backsearch_max),
            "--canvas_format",
            config.canvas_format,
        ]

    def _annotate_provider_meta(
        self,
        output_directory: Path,
        *,
        effective_sampled_frame_count: int,
        max_encoded_frame_id: int,
    ) -> dict[str, JsonValue]:
        meta_path = output_directory / "meta.json"
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise MageDcvcBackendError("readiness pipeline produced no valid meta.json") from error
        if not isinstance(raw, dict):
            raise MageDcvcBackendError("readiness meta.json must be an object")
        self._completed_job_count += 1
        provider_meta: dict[str, JsonValue] = {
            "provider_version": self._config.provider_version,
            "recipe_version": self._config.recipe_version,
            "effective_config_sha256": self._config.effective_config_sha256,
            "provider_implementation_sha256": self._config.provider_implementation_sha256,
            "engine": self._config.engine,
            "preparation_device": self._config.preparation_device,
            "device_concurrency_policy": self._config.device_concurrency_policy,
            "max_side": self._config.max_side,
            "configured_sampled_frame_count": self._config.sampled_frame_count,
            "effective_sampled_frame_count": effective_sampled_frame_count,
            "max_encoded_frame_id": max_encoded_frame_id,
            "engine_load_count": self._engine_load_count,
            "engine_load_seconds": self._engine_load_seconds,
            "worker_completed_job_count": self._completed_job_count,
            "sequence_reset_count_for_job": 1,
            "sequence_length_frames": self._config.sequence_length_frames,
            "canvas_token_side": self._config.canvas_token_side,
            "encoded_frame_extent": self._config.encoded_frame_extent,
            "segment_state_policy": "reset-per-job",
        }
        raw[_PROVIDER_META_KEY] = provider_meta
        _write_canonical_json(meta_path, raw)
        return provider_meta


class MageDcvcPreparationWorker:
    """Single-job durable admission and commit boundary around a resident backend."""

    def __init__(
        self,
        *,
        effective_config: MageDcvcEffectiveConfig,
        backend: MageDcvcPreparationBackend,
        input_roots: Sequence[Path],
        output_root: Path,
        generation_device: str,
        device_guard: MageDcvcDeviceGuard | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if backend.effective_config_sha256 != effective_config.effective_config_sha256:
            raise MageDcvcPreparationRejected("backend and worker effective configs differ")
        roots = tuple(Path(root).expanduser().resolve() for root in input_roots)
        if not roots:
            raise MageDcvcPreparationRejected("at least one input root is required")
        if any(not root.is_dir() for root in roots):
            raise MageDcvcPreparationRejected("every input root must already exist")
        self._config = effective_config
        self._backend = backend
        self._input_roots = roots
        self._output_root = Path(output_root).expanduser().resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._generation_device = _normalise_device(generation_device)
        self._device_guard = device_guard
        self._clock = clock
        self._job_lock = Lock()
        self._shared_device = _same_device(
            self._config.preparation_device,
            self._generation_device,
        )
        if self._shared_device:
            if self._config.device_concurrency_policy != "exclusive-shared-device-v1":
                raise MageDcvcPreparationRejected(
                    "shared preparation/generation device requires exclusive policy"
                )
            if self._device_guard is None:
                raise MageDcvcPreparationRejected(
                    "shared preparation/generation device requires a cooperative guard"
                )
        elif self._config.device_concurrency_policy != "separate-device-v1":
            raise MageDcvcPreparationRejected(
                "separate preparation/generation devices require separate-device policy"
            )

    def prepare(self, request: MageDcvcPreparationRequest) -> MageDcvcPreparationResponse:
        started = self._clock()
        if not self._job_lock.acquire(blocking=False):
            return _failure_response(
                request_id=request.request_id,
                status="BUSY",
                wall_seconds=_elapsed(self._clock, started),
                error_code=MageDcvcPreparationBusy.code,
                error_message="worker already has an active preparation job",
            )
        try:
            artifact, output_directory, admission = self._prepare_locked(request)
            return MageDcvcPreparationResponse(
                request_id=request.request_id,
                status=admission,
                preparation_identity=artifact.preparation_identity,
                artifact_semantic_sha256=artifact.artifact_semantic_sha256,
                output_directory=str(output_directory),
                wall_seconds=_elapsed(self._clock, started),
            )
        except MageDcvcPreparationBusy as error:
            return _failure_response(
                request_id=request.request_id,
                status="BUSY",
                wall_seconds=_elapsed(self._clock, started),
                error_code=error.code,
                error_message=str(error),
            )
        except MageDcvcPreparationRejected as error:
            return _failure_response(
                request_id=request.request_id,
                status="REJECTED",
                wall_seconds=_elapsed(self._clock, started),
                error_code=error.code,
                error_message=str(error),
            )
        except MageDcvcPreparationError as error:
            return _failure_response(
                request_id=request.request_id,
                status="FAILED",
                wall_seconds=_elapsed(self._clock, started),
                error_code=error.code,
                error_message=str(error),
            )
        except Exception:
            return _failure_response(
                request_id=request.request_id,
                status="FAILED",
                wall_seconds=_elapsed(self._clock, started),
                error_code="DCVC_PREPARATION_UNEXPECTED",
                error_message="unexpected preparation failure",
            )
        finally:
            self._job_lock.release()

    def close(self) -> None:
        self._backend.close()

    def _prepare_locked(
        self,
        request: MageDcvcPreparationRequest,
    ) -> tuple[MageDcvcPreparationArtifact, Path, Literal["BUILT", "VERIFIED_HIT"]]:
        if request.effective_config_sha256 != self._config.effective_config_sha256:
            raise MageDcvcPreparationRejected("request effective config does not match worker")
        expected_identity = mage_dcvc_preparation_identity(
            source_content_sha256=request.source_content_sha256,
            source_byte_count=request.source_byte_count,
            effective_config_sha256=request.effective_config_sha256,
        )
        if request.preparation_identity != expected_identity:
            raise MageDcvcPreparationRejected("request preparation identity does not match")
        source = self._resolve_source(request.source_path)
        observed_sha256, observed_bytes = _exact_file_sha256(source)
        if (
            observed_sha256 != request.source_content_sha256
            or observed_bytes != request.source_byte_count
        ):
            raise MageDcvcPreparationRejected("source segment bytes do not match request")
        output = self._resolve_output(request.output_relative_path)
        _validate_required_artifact_path_budget(output=output, request=request)
        if output.exists():
            artifact = self._verify_committed_artifact(output=output, request=request)
            return artifact, output, "VERIFIED_HIT"

        output.parent.mkdir(parents=True, exist_ok=True)
        self._recover_stale_temporaries(parent=output.parent, request=request)
        temporary = output.parent / (
            f".robata-dcvc-{request.preparation_identity[:16]}-{uuid.uuid4().hex}"
        )
        temporary.mkdir(exist_ok=False)
        marker = MageDcvcTempMarker(
            request_id=request.request_id,
            preparation_identity=request.preparation_identity,
            effective_config_sha256=request.effective_config_sha256,
        )
        _write_canonical_json(
            temporary / MAGE_DCVC_TEMP_MARKER_NAME,
            marker.model_dump(mode="json"),
        )
        try:
            if self._shared_device:
                device_guard = self._device_guard
                if device_guard is None:
                    raise MageDcvcPreparationRejected("shared device guard disappeared")
                guard = device_guard.hold()
            else:
                guard = _null_guard()
            try:
                with guard:
                    metadata = self._backend.prepare(
                        source_path=source,
                        output_directory=temporary,
                    )
            except DeviceExecutionGuardBusy as error:
                raise MageDcvcPreparationBusy(str(error)) from error
            self._validate_provider_output(temporary)
            assets = _collect_provider_assets(temporary)
            artifact = _build_artifact(
                request=request,
                config=self._config,
                assets=assets,
                provider_metadata=metadata,
            )
            _write_canonical_json(
                temporary / MAGE_DCVC_PREPARATION_SIDECAR_NAME,
                artifact.model_dump(mode="json"),
            )
            _verify_artifact_directory(
                directory=temporary,
                expected=artifact,
                allow_temp_marker=True,
            )
            (temporary / MAGE_DCVC_TEMP_MARKER_NAME).unlink()
            if output.exists():
                raise MageDcvcPreparationRejected("output appeared during atomic commit")
            temporary.replace(output)
            committed = self._verify_committed_artifact(output=output, request=request)
            return committed, output, "BUILT"
        except Exception:
            with suppress(MageDcvcPreparationError, OSError):
                self._remove_owned_temporary(temporary=temporary, request=request)
            raise

    def _resolve_source(self, source_path: str) -> Path:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise MageDcvcPreparationRejected("source segment is not a file")
        if not any(source.is_relative_to(root) for root in self._input_roots):
            raise MageDcvcPreparationRejected("source segment is outside allowed input roots")
        return source

    def _resolve_output(self, output_relative_path: str) -> Path:
        relative = PurePosixPath(output_relative_path)
        output = self._output_root.joinpath(*relative.parts).resolve()
        if not output.is_relative_to(self._output_root) or output == self._output_root:
            raise MageDcvcPreparationRejected("output path escapes the worker output root")
        return output

    def _validate_provider_output(self, directory: Path) -> None:
        meta_path = directory / "meta.json"
        position_path = directory / "src_patch_position.npy"
        if not meta_path.is_file() or not position_path.is_file():
            raise MageDcvcBackendError(
                "provider output requires meta.json and src_patch_position.npy"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise MageDcvcBackendError("provider meta.json is unreadable") from error
        if not isinstance(meta, dict):
            raise MageDcvcBackendError("provider meta.json must be an object")
        provider_meta = meta.get(_PROVIDER_META_KEY)
        if not isinstance(provider_meta, dict):
            raise MageDcvcBackendError("provider meta.json lacks Robata provider identity")
        if provider_meta.get("effective_config_sha256") != self._config.effective_config_sha256:
            raise MageDcvcBackendError("provider meta.json effective config does not match")
        canvas_files = meta.get("canvas_files", meta.get("jpg_files"))
        if not isinstance(canvas_files, list) or not canvas_files:
            raise MageDcvcBackendError("provider meta.json contains no canvas files")
        for item in canvas_files:
            if not isinstance(item, str):
                raise MageDcvcBackendError("provider canvas file name must be a string")
            path = PurePosixPath(item)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise MageDcvcBackendError("provider canvas file name is unsafe")
            canvas = directory.joinpath(*path.parts).resolve()
            if not canvas.is_relative_to(directory) or not canvas.is_file():
                raise MageDcvcBackendError("provider canvas file is missing or escapes output")

    def _verify_committed_artifact(
        self,
        *,
        output: Path,
        request: MageDcvcPreparationRequest,
    ) -> MageDcvcPreparationArtifact:
        if output.is_symlink() or not output.is_dir():
            raise MageDcvcPreparationRejected("committed output is not a normal directory")
        sidecar = output / MAGE_DCVC_PREPARATION_SIDECAR_NAME
        if sidecar.is_symlink() or not sidecar.is_file():
            raise MageDcvcPreparationRejected("committed output sidecar is missing or inaccessible")
        try:
            sidecar_bytes = sidecar.read_bytes()
        except OSError as error:
            raise MageDcvcPreparationRejected("committed output sidecar is unreadable") from error
        try:
            artifact = MageDcvcPreparationArtifact.model_validate_json(
                sidecar_bytes,
                strict=True,
            )
        except ValidationError as error:
            raise MageDcvcPreparationRejected("committed output sidecar is invalid") from error
        if artifact.preparation_identity != request.preparation_identity:
            raise MageDcvcPreparationRejected("committed output belongs to another preparation")
        if artifact.effective_config_sha256 != self._config.effective_config_sha256:
            raise MageDcvcPreparationRejected("committed output uses another effective config")
        if (
            artifact.source_content_sha256 != request.source_content_sha256
            or artifact.source_byte_count != request.source_byte_count
        ):
            raise MageDcvcPreparationRejected("committed output belongs to different source bytes")
        _verify_artifact_directory(directory=output, expected=artifact, allow_temp_marker=False)
        return artifact

    def _recover_stale_temporaries(
        self,
        *,
        parent: Path,
        request: MageDcvcPreparationRequest,
    ) -> None:
        prefix = f".robata-dcvc-{request.preparation_identity[:16]}-"
        for candidate in sorted(parent.glob(f"{prefix}*"), key=str):
            self._remove_owned_temporary(temporary=candidate, request=request)

    def _remove_owned_temporary(
        self,
        *,
        temporary: Path,
        request: MageDcvcPreparationRequest,
    ) -> None:
        resolved = temporary.resolve()
        if (
            resolved.parent != temporary.parent.resolve()
            or resolved.parent != self._resolve_output(request.output_relative_path).parent
            or not resolved.name.startswith(f".robata-dcvc-{request.preparation_identity[:16]}-")
            or resolved.is_symlink()
            or not resolved.is_dir()
        ):
            raise MageDcvcPreparationRejected("refusing to remove an unowned staging path")
        marker_path = resolved / MAGE_DCVC_TEMP_MARKER_NAME
        try:
            marker = MageDcvcTempMarker.model_validate_json(
                marker_path.read_bytes(),
                strict=True,
            )
        except (OSError, ValidationError) as error:
            raise MageDcvcPreparationRejected(
                "stale staging directory has no valid owner marker"
            ) from error
        if (
            marker.preparation_identity != request.preparation_identity
            or marker.effective_config_sha256 != request.effective_config_sha256
        ):
            raise MageDcvcPreparationRejected("stale staging directory belongs to another job")
        shutil.rmtree(resolved)


def build_mage_dcvc_provider_implementation_sha256(
    model_directory: Path,
) -> Sha256Digest:
    """Hash Robata provider code and every external source file it executes."""

    model_root = Path(model_directory).expanduser().resolve()
    files: dict[str, Path] = {
        "robata/device_execution_guard.py": Path(_device_guard.__file__).resolve(),
        "robata/mage_dcvc_preparation_protocol.py": Path(_protocol.__file__).resolve(),
        "robata/mage_dcvc_preparation_worker.py": Path(__file__).resolve(),
    }
    for relative in _PROVIDER_REQUIRED_RELATIVE_FILES:
        files[f"mage/{relative}"] = model_root / Path(PurePosixPath(relative))
    for relative_root in _PROVIDER_RECURSIVE_ROOTS:
        root = model_root / Path(PurePosixPath(relative_root))
        if not root.is_dir():
            raise MageDcvcBackendError(f"provider source directory is missing: {root}")
        for path in root.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix.lower() in _PROVIDER_SOURCE_SUFFIXES
            ):
                relative = path.relative_to(model_root).as_posix()
                files[f"mage/{relative}"] = path
    projection: list[dict[str, JsonValue]] = []
    for label, path in sorted(files.items()):
        digest, byte_count = _exact_file_sha256(path)
        projection.append({"path": label, "sha256": digest, "byte_count": byte_count})
    return semantic_sha256(
        {
            "provider_version": MAGE_DCVC_PROVIDER_VERSION,
            "recipe_version": MAGE_DCVC_RECIPE_VERSION,
            "files": projection,
        }
    )


def build_mage_dcvc_effective_config(
    *,
    model_directory: Path,
    preparation_device: str,
    device_concurrency_policy: str,
    max_side: int = 0,
    target_canvas: int = 32,
    group_size: int = 32,
    images_per_group: int = 4,
    qp: int = 42,
    reset_interval: int = 64,
    intra_period: int = -1,
    max_pixels: int = 150_000,
    min_group_frames: int = 8,
    max_group_frames: int = 128,
    readiness_coverage_bins: int = 3,
    readiness_delta_ratio: float = 0.05,
    bitcost_percentile: int = 99,
    decode_backsearch_max: int = 16,
    intra_checkpoint_path: Path | None = None,
    inter_checkpoint_path: Path | None = None,
) -> MageDcvcEffectiveConfig:
    """Build the initial fixed provider-v2 config from exact local implementation bytes."""

    if target_canvas <= 0 or group_size <= 0 or images_per_group <= 0:
        raise MageDcvcPreparationRejected("canvas and grouping values must be positive")
    if target_canvas % images_per_group != 0:
        raise MageDcvcPreparationRejected("target_canvas must be divisible by images_per_group")
    sampled_frame_count = (target_canvas // images_per_group) * group_size
    model_root = Path(model_directory).expanduser().resolve()
    neural_root = model_root / "neural_codec"
    intra = Path(intra_checkpoint_path or neural_root / "dcvc_rt_intra.tar").resolve()
    inter = Path(inter_checkpoint_path or neural_root / "dcvc_rt_inter.tar").resolve()
    intra_digest, _ = _exact_file_sha256(intra)
    inter_digest, _ = _exact_file_sha256(inter)
    values: dict[str, Any] = {
        "provider_implementation_sha256": build_mage_dcvc_provider_implementation_sha256(
            model_root
        ),
        "intra_checkpoint_sha256": intra_digest,
        "inter_checkpoint_sha256": inter_digest,
        "preparation_device": preparation_device,
        "device_concurrency_policy": device_concurrency_policy,
        "qp": qp,
        "reset_interval": reset_interval,
        "intra_period": intra_period,
        "max_side": max_side,
        "target_canvas": target_canvas,
        "group_size": group_size,
        "images_per_group": images_per_group,
        "sampled_frame_count": sampled_frame_count,
        "max_pixels": max_pixels,
        "min_group_frames": min_group_frames,
        "max_group_frames": max_group_frames,
        "readiness_coverage_bins": readiness_coverage_bins,
        "readiness_delta_ratio": readiness_delta_ratio,
        "bitcost_percentile": bitcost_percentile,
        "decode_backsearch_max": decode_backsearch_max,
    }
    provisional = MageDcvcEffectiveConfig.model_construct(
        **values,
        effective_config_sha256="0" * 64,
    )
    values["effective_config_sha256"] = semantic_sha256(
        provisional.model_dump(mode="json", exclude={"effective_config_sha256"})
    )
    try:
        return MageDcvcEffectiveConfig.model_validate(values, strict=True)
    except ValidationError as error:
        raise MageDcvcPreparationRejected("effective DCVC configuration is invalid") from error


def build_mage_dcvc_preparation_request(
    *,
    request_id: str,
    source_path: Path,
    output_relative_path: str,
    effective_config: MageDcvcEffectiveConfig,
) -> MageDcvcPreparationRequest:
    """Hash immutable source bytes and construct one exact request."""

    source = Path(source_path).expanduser().resolve()
    source_sha256, source_byte_count = _exact_file_sha256(source)
    identity = mage_dcvc_preparation_identity(
        source_content_sha256=source_sha256,
        source_byte_count=source_byte_count,
        effective_config_sha256=effective_config.effective_config_sha256,
    )
    return MageDcvcPreparationRequest(
        request_id=request_id,
        source_path=str(source),
        source_content_sha256=source_sha256,
        source_byte_count=source_byte_count,
        output_relative_path=output_relative_path,
        effective_config_sha256=effective_config.effective_config_sha256,
        preparation_identity=identity,
    )


def serve_mage_dcvc_preparation_jsonl(
    *,
    worker: MageDcvcPreparationWorker,
    input_stream: TextIO,
    output_stream: TextIO,
) -> None:
    """Serve sequential JSONL requests until EOF; malformed lines fail independently."""

    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = MageDcvcPreparationRequest.model_validate_json(line, strict=True)
        except ValidationError as error:
            response = _failure_response(
                request_id=_best_effort_request_id(line),
                status="REJECTED",
                wall_seconds=0.0,
                error_code="DCVC_PROTOCOL_INVALID",
                error_message=_bounded_message(str(error)),
            )
        else:
            response = worker.prepare(request)
        output_stream.write(canonical_json_bytes(response.model_dump(mode="json")).decode("utf-8"))
        output_stream.write("\n")
        output_stream.flush()


def _validate_required_artifact_path_budget(
    *,
    output: Path,
    request: MageDcvcPreparationRequest,
) -> None:
    """Reject legacy-Windows paths before provider work can publish an unreadable artifact."""

    if not _windows_legacy_max_path_applies():
        return
    output_text = str(output)
    if output_text.startswith("\\\\?\\"):
        return
    staging = output.parent / (f".robata-dcvc-{request.preparation_identity[:16]}-{'0' * 32}")
    required_paths = {
        "committed sidecar": output / MAGE_DCVC_PREPARATION_SIDECAR_NAME,
        "staging ownership marker": staging / MAGE_DCVC_TEMP_MARKER_NAME,
        "staging canonical-JSON temporary": staging / f".tmp-{'0' * 32}",
    }
    label, required = max(required_paths.items(), key=lambda item: len(str(item[1])))
    required_length = len(str(required))
    if required_length >= _WINDOWS_LEGACY_MAX_PATH:
        raise MageDcvcPreparationRejected(
            "output path exceeds the legacy Windows MAX_PATH budget: "
            f"{label} requires {required_length} characters but must be shorter than "
            f"{_WINDOWS_LEGACY_MAX_PATH}; choose a shorter output root or "
            "output_relative_path before running DCVC"
        )


def _windows_legacy_max_path_applies() -> bool:
    return os.name == "nt"


def _build_artifact(
    *,
    request: MageDcvcPreparationRequest,
    config: MageDcvcEffectiveConfig,
    assets: tuple[MageDcvcPreparedAsset, ...],
    provider_metadata: Mapping[str, JsonValue],
) -> MageDcvcPreparationArtifact:
    values: dict[str, Any] = {
        "preparation_identity": request.preparation_identity,
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "source_content_sha256": request.source_content_sha256,
        "source_byte_count": request.source_byte_count,
        "assets": assets,
        "provider_metadata": dict(provider_metadata),
    }
    provisional = MageDcvcPreparationArtifact.model_construct(
        **values,
        artifact_semantic_sha256="0" * 64,
    )
    values["artifact_semantic_sha256"] = mage_dcvc_artifact_semantic_sha256(provisional)
    try:
        return MageDcvcPreparationArtifact.model_validate(values, strict=True)
    except ValidationError as error:
        raise MageDcvcBackendError("provider metadata could not form a valid artifact") from error


def _collect_provider_assets(directory: Path) -> tuple[MageDcvcPreparedAsset, ...]:
    assets: list[MageDcvcPreparedAsset] = []
    for path in sorted(directory.rglob("*"), key=str):
        if path.is_symlink():
            raise MageDcvcBackendError("provider output cannot contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in {MAGE_DCVC_TEMP_MARKER_NAME, MAGE_DCVC_PREPARATION_SIDECAR_NAME}:
            continue
        digest, byte_count = _exact_file_sha256(path)
        assets.append(
            MageDcvcPreparedAsset(
                relative_path=relative,
                byte_count=byte_count,
                sha256=digest,
            )
        )
    if not assets:
        raise MageDcvcBackendError("provider produced no exact assets")
    return tuple(assets)


def _verify_artifact_directory(
    *,
    directory: Path,
    expected: MageDcvcPreparationArtifact,
    allow_temp_marker: bool,
) -> None:
    observed = _collect_provider_assets(directory)
    if observed != expected.assets:
        raise MageDcvcPreparationRejected("provider assets do not match committed artifact")
    marker = directory / MAGE_DCVC_TEMP_MARKER_NAME
    if allow_temp_marker:
        if not marker.is_file():
            raise MageDcvcPreparationRejected("staging artifact lost its ownership marker")
    elif marker.exists():
        raise MageDcvcPreparationRejected("committed artifact contains a staging marker")


def _install_dcvc_config_overlay(
    *,
    state_root: Path,
    config: MageDcvcEffectiveConfig,
) -> Path:
    base = Path(state_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    overlay = (base / config.effective_config_sha256).resolve()
    if not overlay.is_relative_to(base) or overlay == base:
        raise MageDcvcBackendError("DCVC config overlay path escaped its state root")
    overlay.mkdir(parents=True, exist_ok=True)
    values = _external_dcvc_config_values(config)
    encoded = canonical_json_bytes(values).decode("utf-8")
    source = (
        '"""Generated Robata DCVC effective-config overlay v2."""\n'
        "import json\n\n"
        f"_VALUES = json.loads({encoded!r})\n\n"
        "def get(key, cast=None):\n"
        "    if key not in _VALUES:\n"
        "        raise KeyError(f'no effective DCVC value for {key!r}')\n"
        "    value = _VALUES[key]\n"
        "    if cast is bool and isinstance(value, str):\n"
        "        return value.strip().lower() in ('1', 'true', 'yes')\n"
        "    return cast(value) if cast is not None else value\n"
    ).encode()
    module_path = overlay / "codec_dcvc_config.py"
    _write_exact_file_once(module_path, source)
    manifest = {
        "overlay_version": "robata-dcvc-config-overlay-v2",
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "module_sha256": hashlib.sha256(source).hexdigest(),
        "effective_config": config.model_dump(mode="json"),
        "values": values,
    }
    _write_canonical_json(overlay / "overlay-manifest-v2.json", manifest)
    return overlay


def _external_dcvc_config_values(
    config: MageDcvcEffectiveConfig,
) -> dict[str, JsonValue]:
    return {
        "qp": config.qp,
        "reset_interval": config.reset_interval,
        "intra_period": config.intra_period,
        "max_side": config.max_side,
        "num_sampled_frames": config.sampled_frame_count,
        "grouping_mode": config.grouping_mode,
        "readiness_sum_threshold_mode": config.readiness_sum_threshold_mode,
        "group_size": config.group_size,
        "images_per_group": config.images_per_group,
        "patch": config.patch,
        "max_pixels": config.max_pixels,
        "min_group_frames": config.min_group_frames,
        "max_group_frames": config.max_group_frames,
        "readiness_coverage_bins": config.readiness_coverage_bins,
        "readiness_delta_ratio": config.readiness_delta_ratio,
        "bitcost_grid": config.bitcost_grid,
        "bitcost_pct": config.bitcost_percentile,
        "decode_backsearch_max": config.decode_backsearch_max,
        "canvas_format": config.canvas_format,
        "per_frame_cap_ratio": config.per_frame_cap_ratio,
        "bottom_atten": config.bottom_attenuation,
        "bottom_band": config.bottom_band_ratio,
        "threshold_scale": config.threshold_scale,
        "random_select": config.random_select,
        "random_seed": config.random_seed,
    }


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise MageDcvcBackendError("OpenCV is required for DCVC preparation") from error


def _load_external_provider(*, neural_root: Path, overlay_root: Path) -> ModuleType:
    if not neural_root.is_dir():
        raise MageDcvcBackendError("Mage neural_codec directory is missing")
    blocked_prefixes = (
        "codec_dcvc_config",
        "dcvc_readiness_gen",
        "dcvc_rt_engine",
        "codec_tools",
        "pipeline",
        "src",
    )
    already_loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in blocked_prefixes)
        )
    )
    if already_loaded:
        raise MageDcvcBackendError(
            "DCVC readiness modules were imported before effective config installation: "
            + ", ".join(already_loaded)
        )
    _install_dcvc_src_namespace(neural_root=neural_root)
    codec_tools = neural_root / "codec_tools"
    for path in reversed((overlay_root, neural_root, codec_tools)):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    importlib.invalidate_caches()
    try:
        readiness = importlib.import_module("dcvc_readiness_gen")
    except (ImportError, OSError, RuntimeError) as error:
        raise MageDcvcBackendError(
            "could not import bundled Mage DCVC readiness provider"
        ) from error
    _verify_external_provider_binding(
        readiness=readiness,
        neural_root=neural_root,
        overlay_root=overlay_root,
    )
    return readiness


def _install_dcvc_src_namespace(*, neural_root: Path) -> None:
    """Install the bundled DCVC ``src`` namespace before any ambient package wins.

    Microsoft's vendored DCVC tree intentionally has no ``src/__init__.py``.  A regular
    third-party package named ``src`` can therefore override the namespace portion even
    when the DCVC repository is first on ``sys.path``.  This process-local package binds
    submodule search exclusively to the qualified model tree without modifying either
    the upstream or qualified checkpoint bytes.
    """

    dcvc_source = (neural_root / "DCVC" / "src").resolve()
    if not dcvc_source.is_dir() or dcvc_source.is_symlink():
        raise MageDcvcBackendError("bundled DCVC src namespace is missing or unsafe")
    namespace = ModuleType("src")
    namespace.__package__ = "src"
    namespace.__path__ = [str(dcvc_source)]
    spec = ModuleSpec("src", loader=None, is_package=True)
    spec.submodule_search_locations = [str(dcvc_source)]
    namespace.__spec__ = spec
    sys.modules["src"] = namespace


def _verify_external_provider_binding(
    *,
    readiness: ModuleType,
    neural_root: Path,
    overlay_root: Path,
) -> None:
    _require_module_origin(readiness, neural_root / "dcvc_readiness_gen.py")
    config_module = getattr(readiness, "_dc", None)
    if not isinstance(config_module, ModuleType):
        raise MageDcvcBackendError("readiness provider has no config module")
    _require_module_origin(config_module, overlay_root / "codec_dcvc_config.py")
    pipeline = getattr(readiness, "P", None)
    if not isinstance(pipeline, ModuleType):
        raise MageDcvcBackendError("readiness provider has no bundled pipeline module")
    _require_module_origin(
        pipeline,
        neural_root / "codec_tools" / "pipeline" / "process_video_bitcost_readiness.py",
    )
    companion = _companion_module_for(pipeline)
    _require_module_origin(
        companion,
        neural_root / "codec_tools" / "pipeline" / "process_video_bitcost_mv_mask_collage.py",
    )
    if getattr(pipeline, "_dc", None) is not config_module:
        raise MageDcvcBackendError("readiness pipeline did not import the effective config overlay")
    if getattr(companion, "_dc", None) is not config_module:
        raise MageDcvcBackendError(
            "readiness companion did not import the effective config overlay"
        )
    engine_module = sys.modules.get("dcvc_rt_engine")
    if not isinstance(engine_module, ModuleType):
        raise MageDcvcBackendError("readiness provider did not import DCVC engine module")
    _require_module_origin(engine_module, neural_root / "dcvc_rt_engine.py")
    dcvc_source = neural_root / "DCVC" / "src"
    expected_dcvc_modules = {
        "src.utils.common": dcvc_source / "utils" / "common.py",
        "src.models.image_model": dcvc_source / "models" / "image_model.py",
        "src.models.video_model": dcvc_source / "models" / "video_model.py",
        "src.layers.cuda_inference": dcvc_source / "layers" / "cuda_inference.py",
        "src.utils.transforms": dcvc_source / "utils" / "transforms.py",
    }
    for module_name, expected_path in expected_dcvc_modules.items():
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            raise MageDcvcBackendError(f"DCVC engine did not import {module_name}")
        _require_module_origin(module, expected_path)


def _companion_module_for(pipeline: ModuleType) -> ModuleType:
    process_group = getattr(pipeline, "process_group", None)
    companion = inspect.getmodule(process_group)
    if companion is None:
        raise MageDcvcBackendError("could not locate readiness companion module")
    return companion


@contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    try:
        sys.argv = list(argv)
        yield
    finally:
        sys.argv = previous


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_exact_file_once(path: Path, payload: bytes) -> None:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_file() or resolved.read_bytes() != payload:
            raise MageDcvcBackendError("existing DCVC config overlay has different bytes")
        return
    temporary = resolved.with_name(f".tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(resolved)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise MageDcvcBackendError("could not install DCVC config overlay") from error


def _require_module_origin(module: ModuleType, expected: Path) -> None:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or Path(origin).resolve() != expected.resolve():
        raise MageDcvcBackendError(
            f"provider module resolved outside the model tree: {module.__name__}"
        )


def _normalise_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    if device == "cpu" or (
        device.startswith("cuda:") and device[5:].isdigit() and int(device[5:]) >= 0
    ):
        return device
    raise MageDcvcPreparationRejected("device must be cpu, cuda, or cuda:<nonnegative-index>")


def _same_device(first: str, second: str) -> bool:
    return _normalise_device(first) == _normalise_device(second)


@contextmanager
def _null_guard() -> Iterator[None]:
    yield


def _exact_file_sha256(path: Path) -> tuple[Sha256Digest, int]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MageDcvcPreparationRejected(f"required exact file is missing: {resolved}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageDcvcPreparationRejected(f"could not read exact file: {resolved}") from error
    if byte_count <= 0:
        raise MageDcvcPreparationRejected(f"required exact file is empty: {resolved}")
    return digest.hexdigest(), byte_count


def _write_canonical_json(path: Path, payload: object) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        temporary.replace(resolved)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise MageDcvcPreparationError(f"could not write canonical JSON: {resolved}") from error


def _elapsed(clock: Callable[[], float], started: float) -> float:
    return max(0.0, float(clock() - started))


def _bounded_message(message: str) -> str:
    stripped = " ".join(message.split())
    return (stripped or "invalid request")[:16_384]


def _best_effort_request_id(line: str) -> str:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return "invalid-request"
    if isinstance(value, dict):
        request_id = value.get("request_id")
        if (
            isinstance(request_id, str)
            and 1 <= len(request_id) <= 256
            and all(character.isalnum() or character in "._:-" for character in request_id)
        ):
            return request_id
    return "invalid-request"


def _failure_response(
    *,
    request_id: str,
    status: str,
    wall_seconds: float,
    error_code: str,
    error_message: str,
) -> MageDcvcPreparationResponse:
    return MageDcvcPreparationResponse.model_validate(
        {
            "request_id": request_id,
            "status": status,
            "wall_seconds": wall_seconds,
            "error_code": error_code,
            "error_message": _bounded_message(error_message),
        },
        strict=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--effective-config", type=Path, required=True)
    parser.add_argument("--provider-state-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generation-device", required=True)
    parser.add_argument(
        "--shared-device-guard-file",
        type=Path,
        default=None,
        help="required when preparation and generation use the same accelerator",
    )
    parser.add_argument("--intra-checkpoint", type=Path, default=None)
    parser.add_argument("--inter-checkpoint", type=Path, default=None)
    return parser


def _load_effective_config(path: Path) -> MageDcvcEffectiveConfig:
    try:
        return MageDcvcEffectiveConfig.model_validate_json(
            Path(path).expanduser().resolve().read_bytes(),
            strict=True,
        )
    except (OSError, ValidationError) as error:
        raise MageDcvcPreparationRejected("effective config file is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = _load_effective_config(arguments.effective_config)
    guard = (
        ExclusiveFileDeviceGuard(arguments.shared_device_guard_file)
        if arguments.shared_device_guard_file is not None
        else None
    )
    backend = PersistentMageDcvcPreparationBackend(
        effective_config=config,
        model_directory=arguments.model_dir,
        provider_state_root=arguments.provider_state_root,
        intra_checkpoint_path=arguments.intra_checkpoint,
        inter_checkpoint_path=arguments.inter_checkpoint,
    )
    worker = MageDcvcPreparationWorker(
        effective_config=config,
        backend=backend,
        input_roots=arguments.input_root,
        output_root=arguments.output_root,
        generation_device=arguments.generation_device,
        device_guard=guard,
    )
    try:
        serve_mage_dcvc_preparation_jsonl(
            worker=worker,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )
    finally:
        worker.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI.
    raise SystemExit(main())


__all__ = [
    "ExclusiveFileDeviceGuard",
    "MageDcvcBackendError",
    "MageDcvcDeviceGuard",
    "MageDcvcPreparationBackend",
    "MageDcvcPreparationBusy",
    "MageDcvcPreparationError",
    "MageDcvcPreparationRejected",
    "MageDcvcPreparationWorker",
    "PersistentMageDcvcPreparationBackend",
    "build_mage_dcvc_effective_config",
    "build_mage_dcvc_preparation_request",
    "build_mage_dcvc_provider_implementation_sha256",
    "main",
    "serve_mage_dcvc_preparation_jsonl",
]
