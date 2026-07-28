"""Optional NVDEC frame materialization with guarded PyAV fallback."""

from __future__ import annotations

from robata.adapters.nvdec_backend import (
    NvdecBackendUnavailableError,
    media_output_snapshot,
    nvdec_fallback_allowed,
    record_nvdec_fallback,
)
from robata.contracts.pipeline import TemporalVisualPackage
from robata.ports.frame_materialization import (
    FrameMaterializationError,
    FrameMaterializationErrorCode,
    FrameMaterializationRequest,
    FrameMaterializer,
)
from robata.runtime.observability import RuntimeObserver


class NvdecFrameMaterializer:
    """Use a target materializer only while fallback remains pre-publication safe."""

    def __init__(
        self,
        backend: FrameMaterializer | None = None,
        *,
        fallback: FrameMaterializer | None = None,
        max_width: int | None = 320,
        max_parallel_cameras: int = 6,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if backend is not None and not callable(getattr(backend, "materialize", None)):
            raise TypeError("backend must implement FrameMaterializer")
        if fallback is None:
            from robata.adapters.pyav_frame_materializer import PyAvFrameMaterializer

            fallback = PyAvFrameMaterializer(
                max_width=max_width,
                max_parallel_cameras=max_parallel_cameras,
            )
        if not callable(getattr(fallback, "materialize", None)):
            raise TypeError("fallback must implement FrameMaterializer")
        self._backend = backend
        self._fallback = fallback
        self._runtime_observer = runtime_observer

    @property
    def using_nvdec(self) -> bool:
        """Whether this instance has a configured target-SKU backend."""

        return self._backend is not None

    def materialize(self, request: FrameMaterializationRequest) -> TemporalVisualPackage:
        if self._backend is None:
            return self._fallback.materialize(request)
        output_paths = (
            (request.output_directory / "frames",)
            if isinstance(request, FrameMaterializationRequest)
            else ()
        )
        before = media_output_snapshot(output_paths)
        try:
            return self._backend.materialize(request)
        except NvdecBackendUnavailableError as error:
            output_changed = before != media_output_snapshot(output_paths)
            if not nvdec_fallback_allowed(error, output_changed=output_changed):
                raise FrameMaterializationError(
                    FrameMaterializationErrorCode.OUTPUT_IO_ERROR,
                    "NVDEC became unavailable after frame output changed; CPU fallback is unsafe",
                ) from error
            record_nvdec_fallback(
                self._runtime_observer,
                operation="frame_materialization",
                reason=error.reason,
            )
            return self._fallback.materialize(request)
        except FrameMaterializationError:
            raise
        except Exception as error:
            raise FrameMaterializationError(
                FrameMaterializationErrorCode.DECODE_FAILED,
                f"NVDEC frame materialization failed: {type(error).__name__}: {error}",
            ) from error


__all__ = ["NvdecFrameMaterializer"]
