from __future__ import annotations

from dataclasses import dataclass

import pytest

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.frame_cache import FramePayload
from robata.ports.decoded_frame import DecodedFrameView
from robata.sampling.adaptive import (
    AdaptiveSampler,
    AdaptiveSamplingPolicy,
    AdaptiveSignal,
    SignalTrigger,
)


@dataclass(frozen=True)
class _Window:
    window_id: str
    interval: NanosecondInterval


class _Detector:
    def detect(self, frames, *, camera_id):  # type: ignore[no-untyped-def]
        del frames
        if camera_id != CameraId.CAM_01.value:
            return ()
        return (
            SignalTrigger(
                signal_type=AdaptiveSignal.MOTION_ENERGY,
                timestamp_ns=2_000_000_000,
                strength=4.0,
                confidence=0.8,
            ),
            SignalTrigger(
                signal_type=AdaptiveSignal.SCENE_CHANGE,
                timestamp_ns=3_000_000_000,
                strength=1.0,
                confidence=0.5,
            ),
        )


def _frames() -> dict[CameraId, tuple[DecodedFrameView, ...]]:
    view = (DecodedFrameView(timestamp_ns=0, width=1, height=1, gray_pixels=b"\x00"),)
    return {camera_id: view for camera_id in CAMERA_IDS}


def test_sampler_filters_signals_binds_camera_and_applies_hysteresis() -> None:
    sampler = AdaptiveSampler(
        AdaptiveSamplingPolicy(
            version="local-adaptive-runtime-v1",
            min_fps=2.0,
            max_fps=10.0,
            triggers=(AdaptiveSignal.MOTION_ENERGY,),
            hysteresis_sec=2.0,
        ),
        (_Detector(),),
    )

    result = sampler.sample(
        _Window(
            window_id="window-1",
            interval=NanosecondInterval(start_ns=0, end_ns=10_000_000_000),
        ),
        _frames(),
    )

    assert result.trigger_count == 1
    assert result.trigger_features[0].camera_id is CameraId.CAM_01
    assert result.actual_fps == pytest.approx(3.6)


def test_sampler_requires_complete_six_camera_input() -> None:
    sampler = AdaptiveSampler(
        AdaptiveSamplingPolicy(
            version="local-adaptive-runtime-v1",
            min_fps=2.0,
            max_fps=10.0,
            triggers=(AdaptiveSignal.MOTION_ENERGY,),
            hysteresis_sec=0.0,
        ),
        (_Detector(),),
    )
    frames = _frames()
    frames.pop(CameraId.CAM_06)

    with pytest.raises(ValueError, match="every canonical camera"):
        sampler.sample(
            _Window(
                window_id="window-1",
                interval=NanosecondInterval(start_ns=0, end_ns=10_000_000_000),
            ),
            frames,
        )


def test_sampler_rejects_opaque_frame_payloads() -> None:
    sampler = AdaptiveSampler(
        AdaptiveSamplingPolicy(
            version="local-adaptive-runtime-v1",
            min_fps=2.0,
            max_fps=10.0,
            triggers=(AdaptiveSignal.MOTION_ENERGY,),
            hysteresis_sec=0.0,
        ),
        (_Detector(),),
    )
    payload = (FramePayload(timestamp_sec=0.0, data=b"\x89PNG-not-a-view"),)
    frames = {camera_id: payload for camera_id in CAMERA_IDS}

    with pytest.raises(TypeError, match="DecodedFrameView"):
        sampler.sample(
            _Window(
                window_id="window-1",
                interval=NanosecondInterval(start_ns=0, end_ns=10_000_000_000),
            ),
            frames,  # type: ignore[arg-type]
        )
