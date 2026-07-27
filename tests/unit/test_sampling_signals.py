"""Runtime checks for optional adaptive-signal dependencies and frame representation."""

from __future__ import annotations

import pytest

from robata.frame_cache import FramePayload
from robata.ports.decoded_frame import DecodedFrameView
from robata.sampling import signals


def test_signal_detector_fails_closed_when_numpy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signals, "np", None)

    with pytest.raises(ModuleNotFoundError, match="NumPy is required"):
        signals.MotionEnergyDetector().detect(
            (DecodedFrameView(timestamp_ns=0, width=1, height=1, gray_pixels=b"\x00"),),
            camera_id="cam_01",
        )


def test_signal_detector_rejects_opaque_frame_payload_bytes() -> None:
    encoded_looking_payload = FramePayload(
        timestamp_sec=0.0,
        data=b"\x89PNG\r\n\x1a\nnot-decoded-gray-pixels",
    )

    with pytest.raises(TypeError, match="DecodedFrameView"):
        signals.MotionEnergyDetector().detect(
            (encoded_looking_payload,),  # type: ignore[arg-type]
            camera_id="cam_01",
        )


@pytest.mark.skipif(signals.np is None, reason="NumPy is an optional signal-detector dependency")
def test_signal_detectors_use_explicit_dimensions_and_exact_view_timestamps() -> None:
    checkerboard = DecodedFrameView(
        timestamp_ns=1_000_000_001,
        width=3,
        height=2,
        gray_pixels=bytes((0, 255, 0, 255, 0, 255)),
    )
    flat = DecodedFrameView(
        timestamp_ns=2_000_000_003,
        width=3,
        height=2,
        gray_pixels=bytes((0, 0, 0, 0, 0, 0)),
    )

    motion = signals.MotionEnergyDetector(threshold=1.0).detect(
        (flat, checkerboard),
        camera_id="cam_01",
    )
    blur = signals.BlurDetector(threshold=1.0).detect(
        (checkerboard, flat),
        camera_id="cam_01",
    )

    assert [trigger.timestamp_ns for trigger in motion] == [checkerboard.timestamp_ns]
    assert [trigger.timestamp_ns for trigger in blur] == [flat.timestamp_ns]
    assert [trigger.camera_id for trigger in motion] == ["cam_01"]
    assert [trigger.camera_id for trigger in blur] == ["cam_01"]
