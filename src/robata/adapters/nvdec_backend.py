"""Small provider boundary shared by optional NVDEC media adapters.

The CUDA/DeepStream runtime is intentionally not a core dependency. A target-SKU
deployment supplies a backend that implements one of the existing media ports and
raises these errors only for conditions where the CPU reference adapter is allowed to
take over.
"""

from __future__ import annotations

from enum import StrEnum

from robata.runtime.observability import RuntimeObserver, runtime_increment


class NvdecFallbackReason(StrEnum):
    """GPU conditions for which retrying with the CPU reference is safe."""

    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    DEVICE_FAILED = "DEVICE_FAILED"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"


class NvdecBackendUnavailableError(RuntimeError):
    """Signal that an operation has not produced output and may use PyAV instead."""

    def __init__(self, reason: NvdecFallbackReason, message: str) -> None:
        if not isinstance(reason, NvdecFallbackReason):
            raise TypeError("reason must be an NvdecFallbackReason")
        super().__init__(message)
        self.reason = reason


def record_nvdec_fallback(
    runtime_observer: RuntimeObserver | None,
    *,
    operation: str,
    reason: NvdecFallbackReason,
) -> None:
    """Record a CPU retry without allowing telemetry to affect media correctness."""

    runtime_increment(
        runtime_observer,
        "media.nvdec.fallbacks",
        attributes={"operation": operation, "reason": reason.value},
    )


__all__ = [
    "NvdecBackendUnavailableError",
    "NvdecFallbackReason",
    "record_nvdec_fallback",
]
