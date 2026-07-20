"""Runtime checks for optional adaptive-signal dependencies."""

from __future__ import annotations

import pytest

from robata.frame_cache import FramePayload
from robata.sampling import signals


def test_signal_detector_fails_closed_when_numpy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signals, "np", None)

    with pytest.raises(ModuleNotFoundError, match="NumPy is required"):
        signals.MotionEnergyDetector().detect(
            (FramePayload(timestamp_sec=0.0, data=b"\x00"),),
            camera_id="cam_01",
        )
