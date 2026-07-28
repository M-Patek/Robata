from __future__ import annotations

from pathlib import Path

import pytest

from robata.adapters.mcap_single_pass import H264PacketEnvelope
from robata.adapters.nvdec_backend import (
    MediaRuntimeBackend,
    MediaRuntimeProvenance,
    NvdecBackendUnavailableError,
    NvdecFallbackReason,
    NvdecInputProfile,
    NvdecSupportedInput,
)
from robata.adapters.nvdec_frame_materializer import NvdecFrameMaterializer
from robata.adapters.nvdec_video_export import NvdecH264Mp4Exporter
from robata.application.canonical.bounded_media import (
    ACCESS_UNIT_FRAMING_VERSION,
    EncodedMediaPacket,
)
from robata.contracts.cameras import CameraId
from robata.ports.frame_materialization import (
    FrameMaterializationError,
    FrameMaterializationErrorCode,
)
from robata.ports.ingestion import COMPRESSED_IMAGE_SCHEMA, ChannelInspection
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


class _IncrementalSession:
    def __init__(
        self,
        *,
        failure: NvdecBackendUnavailableError | None = None,
        failure_stage: str | None = None,
        observe_before_failure: bool = False,
        facts: object = "incremental-facts",
    ) -> None:
        self.failure = failure
        self.failure_stage = failure_stage
        self.observe_before_failure = observe_before_failure
        self._facts = facts
        self.appended: list[H264PacketEnvelope] = []
        self.observer: object | None = None
        self.sealed = False
        self.aborted = False

    def append_access_unit(
        self,
        envelope: H264PacketEnvelope,
        reference: object,
        *,
        framing_version: str,
    ) -> None:
        self.appended.append(envelope)
        if self.failure_stage == "append":
            if self.observe_before_failure and callable(self.observer):
                self.observer(envelope, object(), len(self.appended) - 1)
            assert self.failure is not None
            raise self.failure

    def seal(self) -> None:
        if self.failure_stage == "seal":
            assert self.failure is not None
            raise self.failure
        self.sealed = True

    def abort(self) -> None:
        self.aborted = True

    @property
    def facts(self) -> object:
        return self._facts


class _IncrementalExporter:
    def __init__(self, sessions: list[_IncrementalSession]) -> None:
        self._sessions = sessions
        self.begin_calls: list[tuple[object, ...]] = []

    def export(self, *args: object) -> object:
        return "unused-export"

    def begin_incremental(
        self,
        *args: object,
        decoded_frame_observer: object | None = None,
        **kwargs: object,
    ) -> _IncrementalSession:
        self.begin_calls.append(args)
        session = self._sessions.pop(0)
        session.observer = decoded_frame_observer
        return session


def _incremental_channel() -> ChannelInspection:
    return ChannelInspection(
        channel_id=1,
        topic="/camera/1",
        schema_name=COMPRESSED_IMAGE_SCHEMA,
        message_encoding="protobuf",
        message_count=1,
        first_message_time_ns=0,
        last_message_time_ns=0,
        monotonic=True,
        codec="h264",
        frame_id="cam_01",
    )


def _incremental_envelope() -> H264PacketEnvelope:
    packet = EncodedMediaPacket(
        traversal_index=0,
        camera_id=CameraId.CAM_01,
        source_order=0,
        source_sequence=0,
        source_timestamp_ns=0,
        aligned_timestamp_ns=0,
        source_locator="memory://cam01/0",
        payload=b"\x00\x00\x00\x01\x41\x80",
        is_keyframe=False,
    )
    return H264PacketEnvelope(
        packet=packet,
        source_publish_time_ns=0,
        embedded_header_time_ns=0,
        nal_types=(1,),
    )


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


def test_nvdec_materializer_does_not_fallback_after_publication_started() -> None:
    fallback = _Materializer("cpu-package")
    adapter = NvdecFrameMaterializer(
        _Materializer(
            NvdecBackendUnavailableError(
                NvdecFallbackReason.DEVICE_FAILED,
                "target staged output",
                publication_started=True,
            )
        ),
        fallback=fallback,
    )

    with pytest.raises(FrameMaterializationError) as raised:
        adapter.materialize(object())  # type: ignore[arg-type]

    assert raised.value.code is FrameMaterializationErrorCode.OUTPUT_IO_ERROR
    assert fallback.requests == []


def test_nvdec_exporter_does_not_fallback_when_target_changed_output(tmp_path: Path) -> None:
    class _WritingExporter(_Exporter):
        def export(self, *args: object) -> object:
            video_path = Path(args[3])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"partial-target-output")
            raise NvdecBackendUnavailableError(
                NvdecFallbackReason.DEVICE_FAILED,
                "target wrote output before device failure",
            )

    fallback = _Exporter("cpu-facts")
    adapter = NvdecH264Mp4Exporter(_WritingExporter("unused"), fallback=fallback)  # type: ignore[arg-type]

    with pytest.raises(VideoExportError) as raised:
        adapter.export(
            tmp_path / "source.mcap",
            object(),
            object(),
            tmp_path / "video.mp4",
            tmp_path / "video.timestamps.jsonl",
        )  # type: ignore[arg-type]

    assert raised.value.code is VideoExportErrorCode.ATOMIC_COMMIT_FAILED
    assert fallback.calls == []


def test_media_runtime_provenance_requires_declared_target_support() -> None:
    profile = NvdecInputProfile(codec="h264", profile="high", width=1920, height=1080)
    support = NvdecSupportedInput(
        codec="h264",
        profiles=("baseline", "high", "main"),
        max_width=3840,
        max_height=2160,
    )
    provenance = MediaRuntimeProvenance.create(
        backend=MediaRuntimeBackend.NVDEC_TARGET,
        implementation="target-nvdec",
        implementation_version="1.0.0",
        selected_input=profile,
        supported_inputs=(support,),
    )

    assert provenance.provenance_sha256
    assert provenance.selected_input == profile

    with pytest.raises(ValueError, match="outside the declared support matrix"):
        MediaRuntimeProvenance.create(
            backend=MediaRuntimeBackend.NVDEC_TARGET,
            implementation="target-nvdec",
            implementation_version="1.0.0",
            selected_input=profile.model_copy(update={"width": 4096}),
            supported_inputs=(support,),
        )


def test_nvdec_exporter_provenance_records_actual_cpu_fallback() -> None:
    profile = NvdecInputProfile(codec="h264", profile="high", width=1920, height=1080)
    support = NvdecSupportedInput(
        codec="h264",
        profiles=("baseline", "high", "main"),
        max_width=3840,
        max_height=2160,
    )
    declared = MediaRuntimeProvenance.create(
        backend=MediaRuntimeBackend.NVDEC_TARGET,
        implementation="target-nvdec",
        implementation_version="1.0.0",
        selected_input=profile,
        supported_inputs=(support,),
    )
    adapter = NvdecH264Mp4Exporter(
        _Exporter(
            NvdecBackendUnavailableError(
                NvdecFallbackReason.UNSUPPORTED_INPUT,
                "target profile is unavailable",
            )
        ),
        fallback=_Exporter("cpu-facts"),
    )  # type: ignore[arg-type]
    runtime_observation = adapter.begin_runtime_observation()

    assert (
        adapter.export(
            Path("source"),
            object(),
            object(),
            Path("video"),
            Path("sidecar"),
            runtime_observation=runtime_observation,
        )
        == "cpu-facts"
    )  # type: ignore[arg-type]

    completed = adapter.completed_runtime_provenance(
        declared,
        runtime_observation=runtime_observation,
    )

    assert runtime_observation.fallback_reasons == (NvdecFallbackReason.UNSUPPORTED_INPUT,)
    assert completed.backend is MediaRuntimeBackend.CPU_FALLBACK
    assert completed.fallback_reasons == (NvdecFallbackReason.UNSUPPORTED_INPUT,)
    assert completed.selected_input == declared.selected_input
    assert completed.supported_inputs == declared.supported_inputs


def test_nvdec_incremental_append_failure_replays_spooled_input_on_cpu(tmp_path: Path) -> None:
    target_session = _IncrementalSession(
        failure=NvdecBackendUnavailableError(
            NvdecFallbackReason.DEVICE_FAILED,
            "target device reset",
        ),
        failure_stage="append",
    )
    fallback_session = _IncrementalSession(facts="cpu-facts")
    target = _IncrementalExporter([target_session])
    fallback = _IncrementalExporter([fallback_session])
    adapter = NvdecH264Mp4Exporter(target, fallback=fallback)  # type: ignore[arg-type]
    runtime_observation = adapter.begin_runtime_observation()
    envelope = _incremental_envelope()

    session = adapter.begin_incremental(
        CameraId.CAM_01,
        _incremental_channel(),
        tmp_path / "video.mp4",
        tmp_path / "video.timestamps.jsonl",
        runtime_observation=runtime_observation,
    )
    session.append_access_unit(
        envelope,
        envelope.packet.reference(),
        framing_version=ACCESS_UNIT_FRAMING_VERSION,
    )
    session.seal()

    assert target_session.aborted is True
    assert fallback_session.appended == [envelope]
    assert fallback_session.sealed is True
    assert session.facts == "cpu-facts"
    assert runtime_observation.fallback_reasons == (NvdecFallbackReason.DEVICE_FAILED,)
    assert list(tmp_path.glob(".nvdec-replay-*.spool")) == []


def test_nvdec_incremental_seal_failure_replays_spooled_input_on_cpu(tmp_path: Path) -> None:
    target_session = _IncrementalSession(
        failure=NvdecBackendUnavailableError(
            NvdecFallbackReason.UNSUPPORTED_INPUT,
            "target codec support changed",
        ),
        failure_stage="seal",
    )
    fallback_session = _IncrementalSession(facts="cpu-facts")
    target = _IncrementalExporter([target_session])
    fallback = _IncrementalExporter([fallback_session])
    adapter = NvdecH264Mp4Exporter(target, fallback=fallback)  # type: ignore[arg-type]
    runtime_observation = adapter.begin_runtime_observation()
    envelope = _incremental_envelope()

    session = adapter.begin_incremental(
        CameraId.CAM_01,
        _incremental_channel(),
        tmp_path / "video.mp4",
        tmp_path / "video.timestamps.jsonl",
        runtime_observation=runtime_observation,
    )
    session.append_access_unit(
        envelope,
        envelope.packet.reference(),
        framing_version=ACCESS_UNIT_FRAMING_VERSION,
    )
    session.seal()

    assert target_session.appended == [envelope]
    assert target_session.aborted is True
    assert fallback_session.appended == [envelope]
    assert fallback_session.sealed is True
    assert runtime_observation.fallback_reasons == (NvdecFallbackReason.UNSUPPORTED_INPUT,)
    assert list(tmp_path.glob(".nvdec-replay-*.spool")) == []


def test_nvdec_incremental_does_not_retry_after_target_observer_runs(tmp_path: Path) -> None:
    target_session = _IncrementalSession(
        failure=NvdecBackendUnavailableError(
            NvdecFallbackReason.DEVICE_FAILED,
            "target failed after frame publication",
        ),
        failure_stage="append",
        observe_before_failure=True,
    )
    fallback = _IncrementalExporter([_IncrementalSession(facts="cpu-facts")])
    adapter = NvdecH264Mp4Exporter(
        _IncrementalExporter([target_session]),
        fallback=fallback,
    )  # type: ignore[arg-type]
    envelope = _incremental_envelope()

    session = adapter.begin_incremental(
        CameraId.CAM_01,
        _incremental_channel(),
        tmp_path / "video.mp4",
        tmp_path / "video.timestamps.jsonl",
        decoded_frame_observer=lambda *_args: None,
    )
    with pytest.raises(VideoExportError) as raised:
        session.append_access_unit(
            envelope,
            envelope.packet.reference(),
            framing_version=ACCESS_UNIT_FRAMING_VERSION,
        )

    assert raised.value.code is VideoExportErrorCode.ATOMIC_COMMIT_FAILED
    assert target_session.aborted is True
    assert fallback.begin_calls == []
    assert list(tmp_path.glob(".nvdec-replay-*.spool")) == []


def test_nvdec_incremental_setup_does_not_retry_after_target_observer_runs(tmp_path: Path) -> None:
    class _SetupFailureExporter(_IncrementalExporter):
        def begin_incremental(
            self,
            *args: object,
            decoded_frame_observer: object | None = None,
            **kwargs: object,
        ) -> _IncrementalSession:
            assert callable(decoded_frame_observer)
            decoded_frame_observer(_incremental_envelope(), object(), 0)
            raise NvdecBackendUnavailableError(
                NvdecFallbackReason.DEVICE_FAILED,
                "target setup failed after frame publication",
            )

    fallback = _IncrementalExporter([_IncrementalSession(facts="cpu-facts")])
    adapter = NvdecH264Mp4Exporter(
        _SetupFailureExporter([]),
        fallback=fallback,
    )  # type: ignore[arg-type]
    runtime_observation = adapter.begin_runtime_observation()

    with pytest.raises(VideoExportError) as raised:
        adapter.begin_incremental(
            CameraId.CAM_01,
            _incremental_channel(),
            tmp_path / "video.mp4",
            tmp_path / "video.timestamps.jsonl",
            decoded_frame_observer=lambda *_args: None,
            runtime_observation=runtime_observation,
        )

    assert raised.value.code is VideoExportErrorCode.ATOMIC_COMMIT_FAILED
    assert fallback.begin_calls == []
    assert runtime_observation.fallback_reasons == ()


def test_nvdec_incremental_does_not_record_a_failed_cpu_retry(tmp_path: Path) -> None:
    target_session = _IncrementalSession(
        failure=NvdecBackendUnavailableError(
            NvdecFallbackReason.DEVICE_FAILED,
            "target device reset",
        ),
        failure_stage="append",
    )
    fallback_session = _IncrementalSession(
        failure=NvdecBackendUnavailableError(
            NvdecFallbackReason.DEVICE_FAILED,
            "CPU output failed",
        ),
        failure_stage="seal",
    )
    adapter = NvdecH264Mp4Exporter(
        _IncrementalExporter([target_session]),
        fallback=_IncrementalExporter([fallback_session]),
    )  # type: ignore[arg-type]
    runtime_observation = adapter.begin_runtime_observation()
    envelope = _incremental_envelope()
    session = adapter.begin_incremental(
        CameraId.CAM_01,
        _incremental_channel(),
        tmp_path / "video.mp4",
        tmp_path / "video.timestamps.jsonl",
        runtime_observation=runtime_observation,
    )
    session.append_access_unit(
        envelope,
        envelope.packet.reference(),
        framing_version=ACCESS_UNIT_FRAMING_VERSION,
    )

    with pytest.raises(NvdecBackendUnavailableError):
        session.seal()

    assert target_session.aborted is True
    assert runtime_observation.fallback_reasons == ()


def test_nvdec_runtime_observations_do_not_leak_between_runs() -> None:
    class _SequenceExporter:
        def __init__(self) -> None:
            self._errors = [
                NvdecBackendUnavailableError(
                    NvdecFallbackReason.DEVICE_FAILED,
                    "first run target failure",
                ),
                NvdecBackendUnavailableError(
                    NvdecFallbackReason.UNSUPPORTED_INPUT,
                    "second run target failure",
                ),
            ]

        def export(self, *args: object) -> object:
            raise self._errors.pop(0)

    adapter = NvdecH264Mp4Exporter(
        _SequenceExporter(),
        fallback=_Exporter("cpu-facts"),
    )  # type: ignore[arg-type]
    first = adapter.begin_runtime_observation()
    second = adapter.begin_runtime_observation()

    for observation in (first, second):
        assert (
            adapter.export(
                Path("source"),
                object(),
                object(),
                Path("video"),
                Path("sidecar"),
                runtime_observation=observation,
            )
            == "cpu-facts"
        )  # type: ignore[arg-type]

    assert first.fallback_reasons == (NvdecFallbackReason.DEVICE_FAILED,)
    assert second.fallback_reasons == (NvdecFallbackReason.UNSUPPORTED_INPUT,)


def test_nvdec_exporter_provenance_retains_all_run_fallback_reasons() -> None:
    class _SequenceExporter:
        def __init__(self) -> None:
            self._errors = [
                NvdecBackendUnavailableError(
                    NvdecFallbackReason.UNSUPPORTED_INPUT,
                    "unsupported target input",
                ),
                NvdecBackendUnavailableError(
                    NvdecFallbackReason.DEVICE_FAILED,
                    "target device reset",
                ),
            ]

        def export(self, *args: object) -> object:
            raise self._errors.pop(0)

    selected_input = NvdecInputProfile(
        codec="h264",
        profile="high",
        width=1920,
        height=1080,
    )
    declared = MediaRuntimeProvenance.create(
        backend=MediaRuntimeBackend.NVDEC_TARGET,
        implementation="target-nvdec",
        implementation_version="1.0.0",
        selected_input=selected_input,
        supported_inputs=(
            NvdecSupportedInput(
                codec="h264",
                profiles=("high",),
                max_width=1920,
                max_height=1080,
            ),
        ),
    )
    adapter = NvdecH264Mp4Exporter(
        _SequenceExporter(),
        fallback=_Exporter("cpu-facts"),
    )  # type: ignore[arg-type]
    runtime_observation = adapter.begin_runtime_observation()

    for _ in range(2):
        assert (
            adapter.export(
                Path("source"),
                object(),
                object(),
                Path("video"),
                Path("sidecar"),
                runtime_observation=runtime_observation,
            )
            == "cpu-facts"
        )  # type: ignore[arg-type]

    completed = adapter.completed_runtime_provenance(
        declared,
        runtime_observation=runtime_observation,
    )

    assert completed.fallback_reasons == (
        NvdecFallbackReason.DEVICE_FAILED,
        NvdecFallbackReason.UNSUPPORTED_INPUT,
    )
    assert runtime_observation.fallback_reasons == (
        NvdecFallbackReason.DEVICE_FAILED,
        NvdecFallbackReason.UNSUPPORTED_INPUT,
    )
    assert (
        adapter.completed_runtime_provenance(
            declared,
            runtime_observation=runtime_observation,
        )
        == completed
    )
