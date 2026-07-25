from __future__ import annotations

from pathlib import Path

import pytest

from robata.adapters.nvdec_backend import NvdecBackendUnavailableError, NvdecFallbackReason
from robata.adapters.nvdec_frame_materializer import NvdecFrameMaterializer
from robata.adapters.nvdec_video_export import NvdecH264Mp4Exporter
from robata.ports.frame_materialization import (
    FrameMaterializationError,
    FrameMaterializationErrorCode,
)
from robata.ports.video_export import VideoExportError, VideoExportErrorCode


class _Materializer:
    def __init__(self, result: object | BaseException) -> None:
        self.result = result
        self.requests: list[object] = []

    def materialize(self, request: object) -> object:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Exporter:
    def __init__(self, result: object | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[object, ...]] = []

    def export(self, *args: object) -> object:
        self.calls.append(args)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _CounterObserver:
    def __init__(self) -> None:
        self.counters: list[tuple[str, int, dict[str, object] | None]] = []

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.counters.append((name, value, attributes))


def test_nvdec_materializer_uses_backend_when_available() -> None:
    backend = _Materializer("gpu-package")
    fallback = _Materializer("cpu-package")

    adapter = NvdecFrameMaterializer(backend, fallback=fallback)  # type: ignore[arg-type]

    assert adapter.using_nvdec is True
    assert adapter.materialize(object()) == "gpu-package"  # type: ignore[arg-type]
    assert len(backend.requests) == 1
    assert fallback.requests == []


def test_nvdec_materializer_falls_back_only_for_declared_pre_output_condition() -> None:
    backend = _Materializer(
        NvdecBackendUnavailableError(NvdecFallbackReason.DEVICE_FAILED, "device reset")
    )
    fallback = _Materializer("cpu-package")

    assert NvdecFrameMaterializer(backend, fallback=fallback).materialize(object()) == "cpu-package"  # type: ignore[arg-type]
    assert len(fallback.requests) == 1


def test_nvdec_materializer_records_fallback_reason() -> None:
    observer = _CounterObserver()
    adapter = NvdecFrameMaterializer(
        _Materializer(
            NvdecBackendUnavailableError(
                NvdecFallbackReason.DEVICE_UNAVAILABLE,
                "no device",
            )
        ),
        fallback=_Materializer("cpu-package"),
        runtime_observer=observer,  # type: ignore[arg-type]
    )

    assert adapter.materialize(object()) == "cpu-package"  # type: ignore[arg-type]
    assert observer.counters == [
        (
            "media.nvdec.fallbacks",
            1,
            {
                "operation": "frame_materialization",
                "reason": NvdecFallbackReason.DEVICE_UNAVAILABLE.value,
            },
        )
    ]


def test_nvdec_materializer_preserves_port_error() -> None:
    expected = FrameMaterializationError(FrameMaterializationErrorCode.TIMESTAMP_MISMATCH, "bad")
    with pytest.raises(FrameMaterializationError) as raised:
        NvdecFrameMaterializer(_Materializer(expected), fallback=_Materializer("cpu")).materialize(
            object()  # type: ignore[arg-type]
        )
    assert raised.value is expected


def test_nvdec_exporter_falls_back_for_device_unavailability() -> None:
    backend = _Exporter(
        NvdecBackendUnavailableError(NvdecFallbackReason.UNSUPPORTED_INPUT, "codec unavailable")
    )
    fallback = _Exporter("cpu-facts")
    adapter = NvdecH264Mp4Exporter(backend, fallback=fallback)  # type: ignore[arg-type]

    assert adapter.using_nvdec is True
    result = adapter.export(  # type: ignore[arg-type]
        Path("source"), object(), object(), Path("video"), Path("sidecar")
    )
    assert result == "cpu-facts"
    assert len(fallback.calls) == 1


def test_nvdec_exporter_records_fallback_reason() -> None:
    observer = _CounterObserver()
    adapter = NvdecH264Mp4Exporter(
        _Exporter(
            NvdecBackendUnavailableError(
                NvdecFallbackReason.UNSUPPORTED_INPUT,
                "codec unavailable",
            )
        ),
        fallback=_Exporter("cpu-facts"),
        runtime_observer=observer,  # type: ignore[arg-type]
    )

    result = adapter.export(  # type: ignore[arg-type]
        Path("source"), object(), object(), Path("video"), Path("sidecar")
    )
    assert result == "cpu-facts"
    assert observer.counters == [
        (
            "media.nvdec.fallbacks",
            1,
            {
                "operation": "video_export",
                "reason": NvdecFallbackReason.UNSUPPORTED_INPUT.value,
            },
        )
    ]


def test_nvdec_exporter_preserves_port_error() -> None:
    expected = VideoExportError(VideoExportErrorCode.INVALID_TIMESTAMP_METADATA, "bad")
    adapter = NvdecH264Mp4Exporter(_Exporter(expected), fallback=_Exporter("cpu"))  # type: ignore[arg-type]

    with pytest.raises(VideoExportError) as raised:
        adapter.export(Path("source"), object(), object(), Path("video"), Path("sidecar"))  # type: ignore[arg-type]
    assert raised.value is expected
