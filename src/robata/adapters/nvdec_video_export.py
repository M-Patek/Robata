"""Optional target-SKU video exporter with explicit PyAV fallback behavior."""

from __future__ import annotations

from pathlib import Path

from robata.adapters.nvdec_backend import (
    NvdecBackendUnavailableError,
    record_nvdec_fallback,
)
from robata.contracts.cameras import CameraId
from robata.ports.ingestion import ChannelInspection
from robata.ports.video_export import (
    CameraVideoExporter,
    ExportedCameraVideoFacts,
    VideoExportError,
    VideoExportErrorCode,
)
from robata.runtime.observability import RuntimeObserver


class NvdecH264Mp4Exporter:
    """Dispatch export to NVDEC/DeepStream when configured, otherwise use PyAV.

    A target backend must obey the existing ``CameraVideoExporter`` port and must only
    raise ``NvdecBackendUnavailableError`` before atomically committing either output.
    All port failures remain visible to callers; they are never turned into a fallback
    because that could conceal a timestamp or artifact-contract violation.
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
            from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter

            fallback = PyAvH264Mp4Exporter()
        if not callable(getattr(fallback, "export", None)):
            raise TypeError("fallback must implement CameraVideoExporter")
        self._backend = backend
        self._fallback = fallback
        self._runtime_observer = runtime_observer

    @property
    def using_nvdec(self) -> bool:
        return self._backend is not None

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> ExportedCameraVideoFacts:
        if self._backend is None:
            return self._fallback.export(source, camera_id, channel, video_path, sidecar_path)
        try:
            return self._backend.export(source, camera_id, channel, video_path, sidecar_path)
        except NvdecBackendUnavailableError as error:
            record_nvdec_fallback(
                self._runtime_observer,
                operation="video_export",
                reason=error.reason,
            )
            return self._fallback.export(source, camera_id, channel, video_path, sidecar_path)
        except VideoExportError:
            raise
        except Exception as error:
            raise VideoExportError(
                VideoExportErrorCode.REMUX_FAILED,
                f"NVDEC video export failed: {type(error).__name__}: {error}",
            ) from error


__all__ = ["NvdecH264Mp4Exporter"]
