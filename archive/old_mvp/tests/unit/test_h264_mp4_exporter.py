from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from robata.adapters.pyav_mp4_exporter import (
    EXPORT_CONFIG,
    EXPORT_PROFILE_ID,
    EXPORT_PROFILE_VERSION,
    EXPORTER_NAME,
    EXPORTER_VERSION,
    PyAvH264Mp4Exporter,
    _AccessUnit,
    _annex_b_nal_types,
    _canonical_sidecar_line,
    _is_independent_bootstrap,
)
from robata.contracts import CameraId, canonical_json_bytes
from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    VideoExportError,
    VideoExportErrorCode,
)


def _channel(
    *,
    schema: str = COMPRESSED_IMAGE_SCHEMA,
    codec: str = "h264",
    monotonic: bool = True,
    message_count: int = 1,
) -> ChannelInspection:
    return ChannelInspection(
        channel_id=7,
        topic="/camera/0/compressed",
        schema_name=schema,
        message_encoding="protobuf",
        message_count=message_count,
        first_message_time_ns=10,
        last_message_time_ns=20,
        monotonic=monotonic,
        codec=codec,
        frame_id="camera-0",
    )


def _row(log_time_ns: int, *, channel_id: int = 7) -> tuple[Any, Any, Any, Any]:
    schema = SimpleNamespace(name=COMPRESSED_IMAGE_SCHEMA)
    channel = SimpleNamespace(id=channel_id)
    message = SimpleNamespace(
        log_time=log_time_ns,
        publish_time=log_time_ns + 1,
        sequence=log_time_ns,
    )
    decoded = SimpleNamespace(
        format="h264",
        data=b"\x00\x00\x00\x01\x41\x80",
        header=SimpleNamespace(timestamp=log_time_ns + 2),
    )
    return schema, channel, message, decoded


class _FakeReader:
    def __init__(self, rows: list[tuple[Any, Any, Any, Any]]) -> None:
        self._rows = rows

    def iter_decoded_messages(self, **_: Any) -> Any:
        return iter(self._rows)


def test_annex_b_nal_detection_and_bootstrap_order_are_exact() -> None:
    payload = b"".join(
        (
            b"\x00\x00\x01\x67\xaa",
            b"\x00\x00\x00\x01\x68\xbb",
            b"\x00\x00\x01\x65\xcc",
            b"\x00\x00\x00\x01\x41\xdd",
        )
    )

    assert _annex_b_nal_types(payload) == (7, 8, 5, 1)
    assert _is_independent_bootstrap((7, 8, 5))
    assert not _is_independent_bootstrap((5, 7, 8))
    assert not _is_independent_bootstrap((7, 8, 1))
    assert _annex_b_nal_types(b"not-annex-b") == ()


def test_export_profile_is_fixed_read_only_and_json_primitive_only() -> None:
    assert EXPORTER_NAME == "robata.pyav_h264_mp4_exporter"
    assert EXPORTER_VERSION == "0.1.0"
    assert EXPORT_PROFILE_ID == "direct-h264-remux-no-reordering"
    assert EXPORT_PROFILE_VERSION == "1.0"
    assert all(type(value) in {str, int, bool} for value in EXPORT_CONFIG.values())
    canonical_json_bytes(EXPORT_CONFIG)

    with pytest.raises(TypeError):
        cast(dict[str, object], EXPORT_CONFIG)["codec"] = "changed"


def test_sidecar_line_is_canonical_jsonl_with_string_nanoseconds() -> None:
    unit = _AccessUnit(
        log_time_ns=9_007_199_254_740_993,
        publish_time_ns=9_007_199_254_740_994,
        embedded_header_time_ns=9_007_199_254_740_995,
        source_sequence=42,
        payload=b"\x00\x00\x00\x01\x65\x80",
        nal_types=(5,),
    )

    line = _canonical_sidecar_line(
        CameraId.CAM_01,
        unit,
        relative_pts_ns=33_333_333,
        duration_ns=33_333_334,
        packet_index=2,
        duration_is_estimated=False,
    )

    assert line.endswith(b"\n") and line.count(b"\n") == 1
    payload = json.loads(line)
    assert line[:-1] == canonical_json_bytes(payload)
    assert payload["source_log_time_ns"] == "9007199254740993"
    assert payload["source_publish_time_ns"] == "9007199254740994"
    assert payload["embedded_header_time_ns"] == "9007199254740995"
    assert payload["relative_pts_ns"] == "33333333"
    assert payload["relative_dts_ns"] == "33333333"
    assert payload["duration_ns"] == "33333334"
    assert payload["source_sequence"] == 42
    assert payload["schema_version"] == "1.0"
    assert payload["export_profile_id"] == EXPORT_PROFILE_ID
    assert payload["export_profile_version"] == EXPORT_PROFILE_VERSION
    assert payload["time_base_numerator"] == 1
    assert payload["time_base_denominator"] == 1_000_000_000
    assert not any(isinstance(value, float) for value in payload.values())


def test_only_foxglove_header_timestamp_is_authoritative() -> None:
    decoded = SimpleNamespace(header=SimpleNamespace(timestamp=123))
    assert PyAvH264Mp4Exporter._embedded_header_time_ns(decoded) == 123

    top_level_only = SimpleNamespace(
        timestamp=SimpleNamespace(seconds=1, nanos=2),
    )
    with pytest.raises(VideoExportError) as caught:
        PyAvH264Mp4Exporter._embedded_header_time_ns(top_level_only)

    assert caught.value.code is VideoExportErrorCode.INVALID_TIMESTAMP_METADATA


@pytest.mark.parametrize(
    ("channel", "expected_code"),
    [
        (_channel(schema="foxglove.RawImage"), VideoExportErrorCode.UNSUPPORTED_SCHEMA),
        (_channel(codec="vp9"), VideoExportErrorCode.UNSUPPORTED_CODEC),
        (_channel(monotonic=False), VideoExportErrorCode.NONMONOTONIC_LOG_TIME),
    ],
)
def test_request_validation_has_stable_errors(
    tmp_path: Path,
    channel: ChannelInspection,
    expected_code: VideoExportErrorCode,
) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"source")

    with pytest.raises(VideoExportError) as raised:
        PyAvH264Mp4Exporter().export(
            source,
            CameraId.CAM_01,
            channel,
            tmp_path / "video.mp4",
            tmp_path / "video.timestamps.jsonl",
        )

    assert raised.value.code is expected_code


def test_nonmonotonic_source_times_fail_and_owned_temps_are_cleaned(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"source")
    rows = [_row(20), _row(10)]

    with (
        patch(
            "robata.adapters.pyav_mp4_exporter.make_reader",
            return_value=_FakeReader(rows),
        ),
        pytest.raises(VideoExportError) as raised,
    ):
        PyAvH264Mp4Exporter().export(
            source,
            CameraId.CAM_01,
            _channel(message_count=2),
            tmp_path / "video.mp4",
            tmp_path / "video.timestamps.jsonl",
        )

    assert raised.value.code is VideoExportErrorCode.NONMONOTONIC_LOG_TIME
    assert not (tmp_path / "video.mp4").exists()
    assert not (tmp_path / "video.timestamps.jsonl").exists()
    assert not list(tmp_path.glob(".*.robata-*.tmp"))


def test_missing_bootstrap_is_auditable_failure_without_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"source")

    with (
        patch(
            "robata.adapters.pyav_mp4_exporter.make_reader",
            return_value=_FakeReader([_row(10)]),
        ),
        pytest.raises(VideoExportError) as raised,
    ):
        PyAvH264Mp4Exporter().export(
            source,
            CameraId.CAM_01,
            _channel(),
            tmp_path / "video.mp4",
            tmp_path / "video.timestamps.jsonl",
        )

    assert raised.value.code is VideoExportErrorCode.BOOTSTRAP_NOT_FOUND
    assert not (tmp_path / "video.mp4").exists()
    assert not (tmp_path / "video.timestamps.jsonl").exists()


def test_existing_destination_is_never_mutated(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"source")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"existing-video")
    sidecar = tmp_path / "video.timestamps.jsonl"

    with pytest.raises(VideoExportError) as raised:
        PyAvH264Mp4Exporter().export(
            source,
            CameraId.CAM_01,
            _channel(),
            video,
            sidecar,
        )

    assert raised.value.code is VideoExportErrorCode.DESTINATION_EXISTS
    assert video.read_bytes() == b"existing-video"
    assert not sidecar.exists()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 3], 2),
        ([1, 2], 2),
        ([2, 3], 2),
        ([4, 1, 2], 2),
    ],
)
def test_tail_duration_median_uses_exact_half_even(values: list[int], expected: int) -> None:
    assert PyAvH264Mp4Exporter._median_half_even(values) == expected
