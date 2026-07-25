"""Optional NVDEC frame-materialization adapter with a verified PyAV fallback."""

from __future__ import annotations

from robata.adapters.nvdec_backend import (
    NvdecBackendUnavailableError,
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
    """Use an injected target-SKU materializer, falling back before output is written.

    The injected backend has the same ``FrameMaterializer`` contract as PyAV. This
    keeps sampling, package identity, timestamps, and artifact publication outside the
    acceleration choice. Backends must raise ``NvdecBackendUnavailableError`` before
    publishing any package when a CPU retry is required.
    """

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
        try:
            return self._backend.materialize(request)
        except NvdecBackendUnavailableError as error:
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
