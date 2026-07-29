"""Selectable target-SKU H.264 exporter with guarded PyAV fallback."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, cast

from robata.adapters.mcap_single_pass import (
    AppendOnlyH264SpoolBranch,
    H264PacketEnvelope,
    iter_h264_spool,
)
from robata.adapters.nvdec_backend import (
    MediaRuntimeBackend,
    MediaRuntimeProvenance,
    NvdecBackendUnavailableError,
    NvdecFallbackReason,
    media_output_snapshot,
    nvdec_fallback_allowed,
    record_nvdec_fallback,
)
from robata.adapters.pyav_mp4_exporter import (
    EXPORTER_NAME,
    EXPORTER_VERSION,
    PyAvH264Mp4Exporter,
)
from robata.application.canonical.bounded_media import ACCESS_UNIT_FRAMING_VERSION
from robata.contracts.cameras import CameraId
from robata.ports.ingestion import ChannelInspection
from robata.ports.video_export import (
    CameraVideoExporter,
    ExportedCameraVideoFacts,
    VideoExportError,
    VideoExportErrorCode,
)
from robata.runtime.observability import RuntimeObserver


class _NvdecDecodedFrameObserver:
    """Track target callbacks because they may publish dependent frame evidence."""

    def __init__(self, observer: Any) -> None:
        self._observer = observer
        self.called = False

    def __call__(
        self,
        envelope: H264PacketEnvelope,
        decoded_frame: Any,
        exported_index: int,
    ) -> None:
        self.called = True
        self._observer(envelope, decoded_frame, exported_index)


class NvdecRuntimeObservation:
    """Thread-safe fallback facts owned by exactly one canonical media run."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._fallback_reasons: set[NvdecFallbackReason] = set()
        self._closed = False

    @property
    def fallback_reasons(self) -> tuple[NvdecFallbackReason, ...]:
        with self._lock:
            return tuple(sorted(self._fallback_reasons, key=lambda reason: reason.value))

    def _record(self, reason: NvdecFallbackReason) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("media runtime observation is already closed")
            self._fallback_reasons.add(reason)

    def close(self) -> None:
        with self._lock:
            self._closed = True


class _ObservedFallbackIncrementalSession:
    """Commit a fallback fact only after the CPU session seals successfully."""

    def __init__(
        self,
        owner: NvdecH264Mp4Exporter,
        session: Any,
        runtime_observation: NvdecRuntimeObservation | None,
        operation: str,
        reason: NvdecFallbackReason,
    ) -> None:
        self._owner = owner
        self._session = session
        self._runtime_observation = runtime_observation
        self._operation = operation
        self._reason = reason
        self._sealed = False

    def append_access_unit(
        self,
        envelope: H264PacketEnvelope,
        reference: Any,
        *,
        framing_version: str,
    ) -> None:
        self._session.append_access_unit(
            envelope,
            reference,
            framing_version=framing_version,
        )

    def seal(self) -> None:
        if self._sealed:
            return
        self._session.seal()
        self._owner._record_completed_fallback(
            runtime_observation=self._runtime_observation,
            operation=self._operation,
            reason=self._reason,
        )
        self._sealed = True

    def abort(self) -> None:
        abort = getattr(self._session, "abort", None)
        if callable(abort):
            abort()

    @property
    def facts(self) -> ExportedCameraVideoFacts:
        return cast(ExportedCameraVideoFacts, self._session.facts)


class _NvdecIncrementalSession:
    """Retry one target session from a bounded local replay spool when still safe."""

    def __init__(
        self,
        owner: NvdecH264Mp4Exporter,
        target_session: Any,
        fallback_begin: Any,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        decoded_frame_observer: Any | None,
        validate_output: bool,
        output_before: tuple[tuple[str, str], ...],
        observer_state: _NvdecDecodedFrameObserver | None,
        runtime_observation: NvdecRuntimeObservation | None,
    ) -> None:
        self._owner = owner
        self._target_session = target_session
        self._fallback_begin = fallback_begin
        self._camera_id = camera_id
        self._channel = channel
        self._video_path = video_path
        self._sidecar_path = sidecar_path
        self._decoded_frame_observer = decoded_frame_observer
        self._validate_output = validate_output
        self._output_before = output_before
        self._observer_state = observer_state
        self._fallback_session: Any | None = None
        self._runtime_observation = runtime_observation
        self._fallback_reason: NvdecFallbackReason | None = None
        self._fallback_operation: str | None = None
        self._sealed = False
        self._replay_path, self._replay_branch = self._new_replay_branch()

    def append_access_unit(
        self,
        envelope: H264PacketEnvelope,
        reference: Any,
        *,
        framing_version: str,
    ) -> None:
        """Append to the selected session, retaining only a retryable local spool."""

        if self._sealed:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                "incremental NVDEC session is already sealed",
            )
        if self._fallback_session is not None:
            self._fallback_session.append_access_unit(
                envelope,
                reference,
                framing_version=framing_version,
            )
            return
        self._replay_branch.append_access_unit(
            envelope,
            reference,
            framing_version=framing_version,
        )
        try:
            self._target_session.append_access_unit(
                envelope,
                reference,
                framing_version=framing_version,
            )
        except NvdecBackendUnavailableError as error:
            self._activate_cpu_fallback(
                error,
                operation="video_export_incremental.append",
            )

    def seal(self) -> None:
        """Seal the target session or replay its exact pre-publication input on CPU."""

        if self._sealed:
            return
        if self._fallback_session is not None:
            self._seal_cpu_fallback()
            self._sealed = True
            self._cleanup_replay_branch()
            return
        try:
            self._target_session.seal()
        except NvdecBackendUnavailableError as error:
            self._activate_cpu_fallback(
                error,
                operation="video_export_incremental.seal",
            )
            assert self._fallback_session is not None
            self._seal_cpu_fallback()
        self._sealed = True
        self._cleanup_replay_branch()

    def abort(self) -> None:
        """Best-effort cleanup of both target state and the retry spool."""

        if self._fallback_session is not None:
            self._abort_session(self._fallback_session)
        else:
            self._abort_target_session()
        self._cleanup_replay_branch()

    @property
    def facts(self) -> ExportedCameraVideoFacts:
        session = (
            self._fallback_session if self._fallback_session is not None else self._target_session
        )
        return cast(ExportedCameraVideoFacts, session.facts)

    def _activate_cpu_fallback(
        self,
        error: NvdecBackendUnavailableError,
        *,
        operation: str,
    ) -> None:
        output_changed = self._output_before != media_output_snapshot(
            (self._video_path, self._sidecar_path)
        )
        if self._observer_state is not None and self._observer_state.called:
            output_changed = True
        if not nvdec_fallback_allowed(error, output_changed=output_changed):
            self._abort_target_session()
            self._cleanup_replay_branch()
            raise VideoExportError(
                VideoExportErrorCode.ATOMIC_COMMIT_FAILED,
                "NVDEC incremental execution changed dependent output; CPU fallback is unsafe",
            ) from error

        self._abort_target_session()
        try:
            self._replay_branch.seal()
            fallback_session = self._fallback_begin(
                self._camera_id,
                self._channel,
                self._video_path,
                self._sidecar_path,
                decoded_frame_observer=self._decoded_frame_observer,
                validate_output=self._validate_output,
            )
            for replayed in iter_h264_spool(self._replay_path):
                fallback_session.append_access_unit(
                    replayed,
                    replayed.packet.reference(),
                    framing_version=ACCESS_UNIT_FRAMING_VERSION,
                )
        except Exception:
            if "fallback_session" in locals():
                self._abort_session(fallback_session)
            self._cleanup_replay_branch()
            raise
        self._fallback_session = fallback_session

        self._fallback_reason = error.reason
        self._fallback_operation = operation

    def _seal_cpu_fallback(self) -> None:
        if self._fallback_session is None or self._fallback_reason is None:
            raise RuntimeError("incremental CPU fallback session is incomplete")
        if self._fallback_operation is None:
            raise RuntimeError("incremental CPU fallback operation is absent")
        self._fallback_session.seal()
        self._owner._record_completed_fallback(
            runtime_observation=self._runtime_observation,
            operation=self._fallback_operation,
            reason=self._fallback_reason,
        )
        self._fallback_reason = None
        self._fallback_operation = None

    def _new_replay_branch(self) -> tuple[Path, AppendOnlyH264SpoolBranch]:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=".nvdec-replay-",
            suffix=".spool",
            dir=self._video_path.parent,
            delete=False,
        ) as temporary:
            replay_path = Path(temporary.name)
        try:
            replay_path.unlink()
        except OSError as error:
            raise VideoExportError(
                VideoExportErrorCode.SOURCE_IO_ERROR,
                f"cannot prepare NVDEC replay spool: {error}",
            ) from error
        return replay_path, AppendOnlyH264SpoolBranch(self._camera_id, replay_path)

    def _abort_target_session(self) -> None:
        self._abort_session(self._target_session)

    @staticmethod
    def _abort_session(session: Any) -> None:
        abort = getattr(session, "abort", None)
        if callable(abort):
            with suppress(Exception):
                abort()

    def _cleanup_replay_branch(self) -> None:
        with suppress(Exception):
            self._replay_branch.abort()
        with suppress(OSError):
            self._replay_path.unlink(missing_ok=True)


class NvdecH264Mp4Exporter:
    """Select a target exporter while CPU retry is still pre-publication safe.

    A target may additionally implement the PyAV begin_incremental surface. That keeps
    the canonical MCAP spool producer on its existing port without requiring CUDA in
    the local reference install.
    """

    def __init__(
        self,
        backend: CameraVideoExporter | None = None,
        *,
        fallback: CameraVideoExporter | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if backend is not None and not callable(getattr(backend, "export", None)):
            raise TypeError("backend must implement CameraVideoExporter")
        if fallback is None:
            fallback = PyAvH264Mp4Exporter()
        if not callable(getattr(fallback, "export", None)):
            raise TypeError("fallback must implement CameraVideoExporter")
        self._backend = backend
        self._fallback = fallback
        self._runtime_observer = runtime_observer

    @property
    def using_nvdec(self) -> bool:
        return self._backend is not None

    def begin_runtime_observation(self) -> NvdecRuntimeObservation:
        """Create an isolated observation handle for one canonical media run."""

        return NvdecRuntimeObservation()

    def close_runtime_observation(
        self,
        runtime_observation: NvdecRuntimeObservation,
    ) -> None:
        if not isinstance(runtime_observation, NvdecRuntimeObservation):
            raise TypeError("runtime_observation must be an NvdecRuntimeObservation")
        runtime_observation.close()

    def completed_runtime_provenance(
        self,
        declared: MediaRuntimeProvenance,
        *,
        runtime_observation: NvdecRuntimeObservation,
    ) -> MediaRuntimeProvenance:
        """Resolve immutable runtime facts without consuming another run's state."""

        if not isinstance(declared, MediaRuntimeProvenance):
            raise TypeError("declared must be a MediaRuntimeProvenance")
        if not isinstance(runtime_observation, NvdecRuntimeObservation):
            raise TypeError("runtime_observation must be an NvdecRuntimeObservation")
        reasons = runtime_observation.fallback_reasons
        if not reasons:
            return declared
        implementation, implementation_version = self._fallback_runtime_identity()
        return MediaRuntimeProvenance.create(
            backend=MediaRuntimeBackend.CPU_FALLBACK,
            implementation=implementation,
            implementation_version=implementation_version,
            selected_input=declared.selected_input,
            supported_inputs=declared.supported_inputs,
            fallback_reasons=reasons,
        )

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        *,
        runtime_observation: NvdecRuntimeObservation | None = None,
    ) -> ExportedCameraVideoFacts:
        if self._backend is None:
            return self._export_cpu_fallback(
                source,
                camera_id,
                channel,
                video_path,
                sidecar_path,
                runtime_observation=runtime_observation,
                operation="video_export",
                reason=NvdecFallbackReason.DEVICE_UNAVAILABLE,
            )
        before = media_output_snapshot((video_path, sidecar_path))
        try:
            return self._backend.export(source, camera_id, channel, video_path, sidecar_path)
        except NvdecBackendUnavailableError as error:
            output_changed = before != media_output_snapshot((video_path, sidecar_path))
            if not nvdec_fallback_allowed(error, output_changed=output_changed):
                raise VideoExportError(
                    VideoExportErrorCode.ATOMIC_COMMIT_FAILED,
                    "NVDEC became unavailable after video output changed; CPU fallback is unsafe",
                ) from error
            return self._export_cpu_fallback(
                source,
                camera_id,
                channel,
                video_path,
                sidecar_path,
                runtime_observation=runtime_observation,
                operation="video_export",
                reason=error.reason,
            )
        except VideoExportError:
            raise
        except Exception as error:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"NVDEC video export failed: {type(error).__name__}: {error}",
            ) from error

    def begin_incremental(
        self,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        *,
        decoded_frame_observer: Any | None = None,
        validate_output: bool = True,
        runtime_observation: NvdecRuntimeObservation | None = None,
    ) -> Any:
        """Select an incremental target session before final output is committed."""

        fallback_begin = getattr(self._fallback, "begin_incremental", None)
        if not callable(fallback_begin):
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                "configured CPU fallback does not support incremental H.264 export",
            )
        if self._backend is None:
            return self._begin_cpu_incremental(
                fallback_begin,
                camera_id,
                channel,
                video_path,
                sidecar_path,
                decoded_frame_observer=decoded_frame_observer,
                validate_output=validate_output,
                runtime_observation=runtime_observation,
                operation="video_export_incremental",
                reason=NvdecFallbackReason.DEVICE_UNAVAILABLE,
            )
        target_begin = getattr(self._backend, "begin_incremental", None)
        if not callable(target_begin):
            return self._begin_cpu_incremental(
                fallback_begin,
                camera_id,
                channel,
                video_path,
                sidecar_path,
                decoded_frame_observer=decoded_frame_observer,
                validate_output=validate_output,
                runtime_observation=runtime_observation,
                operation="video_export_incremental",
                reason=NvdecFallbackReason.UNSUPPORTED_INPUT,
            )
        before = media_output_snapshot((video_path, sidecar_path))
        observer_state = (
            None
            if decoded_frame_observer is None
            else _NvdecDecodedFrameObserver(decoded_frame_observer)
        )
        try:
            target_session = target_begin(
                camera_id,
                channel,
                video_path,
                sidecar_path,
                decoded_frame_observer=observer_state,
                validate_output=validate_output,
            )
        except NvdecBackendUnavailableError as error:
            output_changed = self._target_output_changed(
                before,
                video_path,
                sidecar_path,
                observer_state,
            )
            if not nvdec_fallback_allowed(error, output_changed=output_changed):
                raise VideoExportError(
                    VideoExportErrorCode.ATOMIC_COMMIT_FAILED,
                    "NVDEC incremental setup changed output; CPU fallback is unsafe",
                ) from error
            return self._begin_cpu_incremental(
                fallback_begin,
                camera_id,
                channel,
                video_path,
                sidecar_path,
                decoded_frame_observer=decoded_frame_observer,
                validate_output=validate_output,
                runtime_observation=runtime_observation,
                operation="video_export_incremental.setup",
                reason=error.reason,
            )
        return _NvdecIncrementalSession(
            self,
            target_session,
            fallback_begin,
            camera_id,
            channel,
            video_path,
            sidecar_path,
            decoded_frame_observer,
            validate_output,
            before,
            observer_state,
            runtime_observation,
        )

    def _record_completed_fallback(
        self,
        *,
        runtime_observation: NvdecRuntimeObservation | None,
        operation: str,
        reason: NvdecFallbackReason,
    ) -> None:
        if runtime_observation is not None:
            runtime_observation._record(reason)
        record_nvdec_fallback(
            self._runtime_observer,
            operation=operation,
            reason=reason,
        )

    def _export_cpu_fallback(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        *,
        runtime_observation: NvdecRuntimeObservation | None,
        operation: str,
        reason: NvdecFallbackReason,
    ) -> ExportedCameraVideoFacts:
        facts = self._fallback.export(source, camera_id, channel, video_path, sidecar_path)
        self._record_completed_fallback(
            runtime_observation=runtime_observation,
            operation=operation,
            reason=reason,
        )
        return facts

    def _begin_cpu_incremental(
        self,
        fallback_begin: Any,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
        *,
        decoded_frame_observer: Any | None,
        validate_output: bool,
        runtime_observation: NvdecRuntimeObservation | None,
        operation: str,
        reason: NvdecFallbackReason,
    ) -> _ObservedFallbackIncrementalSession:
        session = fallback_begin(
            camera_id,
            channel,
            video_path,
            sidecar_path,
            decoded_frame_observer=decoded_frame_observer,
            validate_output=validate_output,
        )
        return _ObservedFallbackIncrementalSession(
            self,
            session,
            runtime_observation,
            operation,
            reason,
        )

    @staticmethod
    def _target_output_changed(
        before: tuple[tuple[str, str], ...],
        video_path: Path,
        sidecar_path: Path,
        observer_state: _NvdecDecodedFrameObserver | None,
    ) -> bool:
        return before != media_output_snapshot((video_path, sidecar_path)) or (
            observer_state is not None and observer_state.called
        )

    def _fallback_runtime_identity(self) -> tuple[str, str]:
        if isinstance(self._fallback, PyAvH264Mp4Exporter):
            return (EXPORTER_NAME, EXPORTER_VERSION)
        fallback_type = type(self._fallback)
        return (
            f"{fallback_type.__module__}.{fallback_type.__qualname__}",
            "unversioned-fallback-v1",
        )


__all__ = ["NvdecH264Mp4Exporter", "NvdecRuntimeObservation"]
