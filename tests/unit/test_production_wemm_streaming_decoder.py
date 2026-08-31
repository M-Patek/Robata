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

    def copy(self):
        copied = _Image(self.timestamp_ns)
        copied.convert_called = self.convert_called
        return copied

    def close(self) -> None:
        self.closed = True


def test_short_context_padding_is_explicit_and_source_bound() -> None:
    """Only the opt-in refinement path may fill a missing target slot."""

    selected = {
        0: {"delta": 1, "timestamp_ns": 100, "image": _Image(100)},
        2: {"delta": 2, "timestamp_ns": 200, "image": _Image(200)},
        3: {"delta": 3, "timestamp_ns": 300, "image": _Image(300)},
    }
    with pytest.raises(shadow.ProductionWemmShadowError, match="yielded 3/4"):
        shadow._complete_selected_frames(
            selected,
            frame_count=4,
            allow_frame_padding=False,
            field="camera cam_01 window probe",
        )

    rows, padding_indices, observed = shadow._complete_selected_frames(
        selected,
        frame_count=4,
        allow_frame_padding=True,
        field="camera cam_01 window probe",
    )
    assert observed == 3
    assert padding_indices == (1,)
    assert len(rows) == 4
    assert rows[1]["timestamp_ns"] == 100
    assert rows[1]["image"] is not rows[0]["image"]

    group = shadow.ProductionFrameGroup(
        camera_id="cam_01",
        window_id="probe",
        frames=tuple(row["image"] for row in rows),
        selected_timestamps_ns=tuple(int(row["timestamp_ns"]) for row in rows),
        messages_examined=1,
        decoded_frames=3,
        decode_failures=(),
        width=10,
        height=8,
        fps=30.0,
        start_seconds=0.0,
        end_seconds=0.25,
        frame_count_requested=4,
        frame_count_observed=observed,
        frame_padding_indices=padding_indices,
    )
    metadata = group.metadata()
    assert metadata["frame_count_requested"] == 4
    assert metadata["frame_count_observed"] == 3
    assert metadata["frame_padding_used"] is True
    assert metadata["frame_padding_indices"] == [1]


def test_short_context_padding_requires_two_real_frames() -> None:
    selected = {0: {"delta": 1, "timestamp_ns": 100, "image": _Image(100)}}
    with pytest.raises(shadow.ProductionWemmShadowError, match="yielded 1/4"):
        shadow._complete_selected_frames(
            selected,
            frame_count=4,
            allow_frame_padding=True,
            field="camera cam_01 window probe",
        )


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


def _dense_manifest(tmp_path, *, window_chunk_size: int = 1):
    """Build a tiny 4 s/1 s overlap fixture for carry regression tests."""

    manifest = _manifest(tmp_path)
    manifest["source"]["cameras"] = [
        {
            "camera_id": f"cam_{index:02d}",
            "topic": f"/cam/{index}",
            "frame_count": 20,
            "duration_seconds": 8.0,
        }
        for index in range(1, 7)
    ]
    manifest["windows"] = [
        {
            "ordinal": index,
            "window_id": f"w{index:02d}",
            "start_seconds": float(index),
            "end_seconds": float(index + 4),
        }
        for index in range(4)
    ]
    return manifest


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


def test_short_refinement_context_can_opt_into_edge_frame_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A sub-frame edge probe is padded only on the explicit refinement path."""

    _Decoder.instances.clear()
    _Reader.iterations = 0
    manifest = _manifest(tmp_path)
    manifest["source"]["cameras"] = [
        {
            "camera_id": f"cam_{index:02d}",
            "topic": f"/cam/{index}",
            "frame_count": 20,
            "duration_seconds": 1.0,
        }
        for index in range(1, 7)
    ]
    manifest["windows"] = [
        {
            "ordinal": 0,
            "window_id": "edge",
            "start_seconds": 0.0,
            "end_seconds": 0.25,
            "temporal_refinement": True,
            "context_only": True,
        }
    ]
    messages = []
    # Three real frames fall into the short context; the fourth WeMM slot
    # must be a separately-owned duplicate when padding is enabled.
    for timestamp_ns in (0, 100_000_000, 200_000_000, 300_000_000):
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
    with pytest.raises(shadow.ProductionWemmShadowError, match="3/4"):
        list(
            shadow.iter_decode_production_window_chunks(
                manifest,
                frame_count=4,
                window_chunk_size=1,
                allow_frame_padding=False,
            )
        )

    # Recreate the fake reader/decoder state because the strict attempt has
    # consumed the reader and codec instances.
    _Decoder.instances.clear()
    _Reader.iterations = 0
    reader = _Reader(messages)
    monkeypatch.setattr(
        shadow,
        "import_module",
        lambda name: (
            fake_av
            if name == "av"
            else fake_mcap
            if name == "mcap.reader"
            else fake_decoder
            if name == "mcap_protobuf.decoder"
            else SimpleNamespace()
            if name == "PIL.Image"
            else real_import(name)
        ),
    )
    groups = list(
        shadow.iter_decode_production_window_chunks(
            manifest,
            frame_count=4,
            window_chunk_size=1,
            allow_frame_padding=True,
        )
    )
    group = groups[0]["cam_01"]["edge"]
    assert len(group.frames) == 4
    assert group.frame_count_requested == 4
    assert group.frame_count_observed == 3
    assert group.frame_padding_indices
    assert group.metadata()["frame_padding_used"] is True
    assert group.to_dict()["frame_count_observed"] == 3
    assert group.frames[0] is not group.frames[1]


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

    monkeypatch.setattr(
        shadow,
        "import_module",
        fake_import,
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


@pytest.mark.parametrize("window_chunk_size", [1, 2])
def test_dense_overlap_carries_one_frame_to_all_future_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path, window_chunk_size: int
) -> None:
    """A 4 s/1 s grid must fill every window, even across chunk boundaries."""

    _Decoder.instances.clear()
    _Reader.iterations = 0
    manifest = _dense_manifest(tmp_path, window_chunk_size=window_chunk_size)
    messages = []
    # Integer-second frames land exactly on all requested target timestamps:
    # w00=[0,1,2,3], w01=[1,2,3,4], w02=[2,3,4,5], w03=[3,4,5,6].
    for timestamp_seconds in range(8):
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
            frame_count=4,
            window_chunk_size=window_chunk_size,
        )
    )

    assert _Reader.iterations == 1
    expected_windows = [f"w{index:02d}" for index in range(4)]
    flattened = [window_id for chunk in chunks for window_id in chunk["cam_01"]]
    assert flattened == expected_windows
    assert len(chunks) == (4 + window_chunk_size - 1) // window_chunk_size
    for chunk in chunks:
        for group in chunk["cam_01"].values():
            assert len(group.frames) == 4
            assert group.selected_timestamps_ns == tuple(
                int(value * 1_000_000_000)
                for value in range(int(group.start_seconds), int(group.start_seconds) + 4)
            )


def test_dense_overlap_does_not_share_closeable_images_between_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Each overlapping destination owns an independent image object."""

    _Decoder.instances.clear()
    _Reader.iterations = 0
    manifest = _dense_manifest(tmp_path)
    messages = []
    for timestamp_seconds in range(8):
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
            frame_count=4,
            window_chunk_size=1,
        )
    )
    first = chunks[0]["cam_01"]["w00"].frames[1]
    second = chunks[1]["cam_01"]["w01"].frames[0]
    assert first is not second
    first.close()
    assert second.closed is False


def test_dense_overlap_uses_linear_future_match_for_unsorted_manifest_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An externally composed non-monotonic manifest must not lose carry frames."""

    _Decoder.instances.clear()
    _Reader.iterations = 0
    manifest = _dense_manifest(tmp_path)
    # Keep the first context conventional, then deliberately put a later
    # context before an earlier one.  The future starts and ends are both no
    # longer monotonic; binary-search matching would be unsound.
    manifest["windows"] = [
        manifest["windows"][0],
        manifest["windows"][2],
        manifest["windows"][1],
        manifest["windows"][3],
    ]
    messages = []
    for timestamp_seconds in range(8):
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
            frame_count=4,
            window_chunk_size=1,
        )
    )
    groups = {window_id: group for chunk in chunks for window_id, group in chunk["cam_01"].items()}
    assert groups["w01"].selected_timestamps_ns == (
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
        4_000_000_000,
    )
