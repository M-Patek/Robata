from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from typing import ClassVar

import pytest

import robata.benchmark.production_wemm_shadow as shadow


class _Image:
    def __init__(self, timestamp_ns: int) -> None:
        self.timestamp_ns = timestamp_ns
        self.width = 10
        self.height = 8
        self.mode = "RGB"
        self.convert_called = False
        self.closed = False

    def convert(self, mode: str):
        assert mode == "RGB"
        self.convert_called = True
        return self

    def close(self) -> None:
        self.closed = True


class _Frame:
    def __init__(self, timestamp_ns: int) -> None:
        self.pts = timestamp_ns
        self.time_base = Fraction(1, 1_000_000_000)

    def to_image(self) -> _Image:
        return _Image(self.pts)


class _Packet:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.pts = 0
        self.dts = 0
        self.time_base = Fraction(1, 1_000_000_000)


class _Decoder:
    instances: ClassVar[list[_Decoder]] = []

    def __init__(self) -> None:
        self.calls = 0
        self.__class__.instances.append(self)

    def decode(self, packet: _Packet):
        self.calls += 1
        return [_Frame(packet.pts)]


class _CodecContext:
    @staticmethod
    def create(codec: str, mode: str) -> _Decoder:
        assert (codec, mode) == ("h264", "r")
        return _Decoder()


class _Reader:
    iterations = 0

    def __init__(self, messages):
        self.messages = messages

    def iter_decoded_messages(self, *, topics, log_time_order):
        assert len(topics) == 6
        assert log_time_order is False
        _Reader.iterations += 1
        yield from self.messages


def _manifest(tmp_path):
    source = tmp_path / "sample.mcap"
    source.write_bytes(b"fixture")
    cameras = [
        {
            "camera_id": f"cam_{index:02d}",
            "topic": f"/cam/{index}",
            "frame_count": 20,
            "duration_seconds": 16.0,
        }
        for index in range(1, 7)
    ]
    windows = [
        {"ordinal": 0, "window_id": "w00", "start_seconds": 0.0, "end_seconds": 8.0},
        {"ordinal": 1, "window_id": "w01", "start_seconds": 8.0, "end_seconds": 16.0},
    ]
    return {
        "format": "robata-production-shaped-cohort-v1",
        "source": {
            "path": str(source),
            "common_start_timestamp_ns": "0",
            "camera_count": 6,
            "cameras": cameras,
        },
        "windows": windows,
    }


def test_stateful_iterator_scans_source_once_and_preserves_window_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _Decoder.instances.clear()
    _Reader.iterations = 0
    manifest = _manifest(tmp_path)
    messages = []
    # Interleave six cameras at each timestamp.  Boundary timestamp 8s is
    # intentionally included to exercise carry-over into the second window.
    for timestamp_seconds in range(17):
        timestamp_ns = timestamp_seconds * 1_000_000_000
        for index in range(1, 7):
            topic = f"/cam/{index}"
            schema = SimpleNamespace(name="foxglove.CompressedImage")
            channel = SimpleNamespace(topic=topic)
            message = SimpleNamespace(log_time=timestamp_ns)
            decoded = SimpleNamespace(data=b"h264")
            messages.append((schema, channel, message, decoded))

    reader = _Reader(messages)
    fake_mcap = SimpleNamespace(make_reader=lambda *args, **kwargs: reader)
    fake_decoder = SimpleNamespace(DecoderFactory=lambda: object())
    fake_av = SimpleNamespace(CodecContext=_CodecContext, Packet=_Packet)
    real_import = shadow.import_module

    def fake_import(name: str):
        if name == "av":
            return fake_av
        if name == "mcap.reader":
            return fake_mcap
        if name == "mcap_protobuf.decoder":
            return fake_decoder
        if name == "PIL.Image":
            return SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr(shadow, "import_module", fake_import)
    chunks = list(
        shadow.iter_decode_production_window_chunks(
            manifest,
            frame_count=2,
            window_chunk_size=1,
        )
    )

    assert _Reader.iterations == 1
    assert len(_Decoder.instances) == 6
    assert all(decoder.calls == 17 for decoder in _Decoder.instances)
    assert len(chunks) == 2
    assert list(chunks[0]["cam_01"]) == ["w00"]
    assert list(chunks[1]["cam_01"]) == ["w01"]
    first = chunks[0]["cam_01"]["w00"]
    second = chunks[1]["cam_01"]["w01"]
    assert first.selected_timestamps_ns == (0, 4_000_000_000)
    assert second.selected_timestamps_ns == (8_000_000_000, 12_000_000_000)
    assert first.messages_examined == 9
    assert second.messages_examined == 17
    assert first.decode_failures == ()
    assert second.decode_failures == ()
    assert all(not image.convert_called for image in (*first.frames, *second.frames))


def test_stateful_iterator_batches_windows_without_rescanning(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _Decoder.instances.clear()
    _Reader.iterations = 0
    manifest = _manifest(tmp_path)
    messages = []
    for timestamp_seconds in range(17):
        timestamp_ns = timestamp_seconds * 1_000_000_000
        for index in range(1, 7):
            messages.append(
                (
                    SimpleNamespace(name="foxglove.CompressedImage"),
                    SimpleNamespace(topic=f"/cam/{index}"),
                    SimpleNamespace(log_time=timestamp_ns),
                    SimpleNamespace(data=b"h264"),
                )
            )
    reader = _Reader(messages)
    fake_mcap = SimpleNamespace(make_reader=lambda *args, **kwargs: reader)
    fake_decoder = SimpleNamespace(DecoderFactory=lambda: object())
    fake_av = SimpleNamespace(CodecContext=_CodecContext, Packet=_Packet)
    real_import = shadow.import_module
    monkeypatch.setattr(
        shadow,
        "import_module",
        lambda name: {
            "av": fake_av,
            "mcap.reader": fake_mcap,
            "mcap_protobuf.decoder": fake_decoder,
            "PIL.Image": SimpleNamespace(),
        }.get(name, real_import(name)),
    )

    chunks = list(
        shadow.iter_decode_production_window_chunks(
            manifest,
            frame_count=2,
            window_chunk_size=2,
        )
    )
    assert _Reader.iterations == 1
    assert len(chunks) == 1
    assert list(chunks[0]["cam_01"]) == ["w00", "w01"]
    assert chunks[0]["cam_01"]["w01"].selected_timestamps_ns == (
        8_000_000_000,
        12_000_000_000,
    )
