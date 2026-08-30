from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.benchmark.wemm_video_microbatch_benchmark import (
    WEMM_VIDEO_MICROBATCH_BENCHMARK_FORMAT,
    WemmVideoMicrobatchBenchmarkError,
    build_cohort_microbatch_plan,
    run_decoded_video_microbatch_benchmark,
    run_video_microbatch_benchmark,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []
        self.serial_calls = 0
        self.batch_calls: list[int] = []

    @staticmethod
    def _rows(groups):
        return tuple((float(index + 1), float(index + 2)) for index, _ in enumerate(groups))

    def encode_video_frames(self, groups, *, metadata_groups=None):
        del metadata_groups
        groups = tuple(groups)
        self.serial_calls += 1
        self.observations.append(
            {
                "modality": "video",
                "item_count": 1,
                "frame_count": 2,
                "video_grid_thw": [[2, 14, 16]],
            }
        )
        return self._rows(groups)

    def encode_video_frames_batch(self, groups, *, metadata_groups=None, batch_size):
        del metadata_groups
        groups = tuple(groups)
        self.batch_calls.append(int(batch_size))
        self.observations.append(
            {
                "modality": "video",
                "item_count": len(groups),
                "batch": int(batch_size),
                "batch_size": int(batch_size),
                "video_grid_thw": [[2, 14, 16] for _ in groups],
                "processor_tensor_shapes": {"input_ids": [len(groups), 2]},
                "phase_timings": {
                    "processor": 0.001,
                    "model": 0.002,
                    "postprocess": 0.0001,
                    "total": 0.0031,
                },
            }
        )
        return self._rows(groups)


class _Group:
    def __init__(self, camera_id: str, window_id: str) -> None:
        self.frames = (f"{camera_id}-{window_id}-0", f"{camera_id}-{window_id}-1")
        self.camera_id = camera_id
        self.window_id = window_id

    def metadata(self) -> dict[str, object]:
        return {
            "total_num_frames": 2,
            "fps": 1.0,
            "frames_indices": [0, 1],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "window_id": self.window_id,
            "selected_timestamps_ns": [1, 2],
        }


def _manifest(duration: float = 40.833423) -> dict[str, object]:
    cameras = [
        {
            "camera_id": f"cam_{index:02d}",
            "topic": f"/camera/{index}",
            "frame_count": 1226,
            "duration_seconds": duration,
        }
        for index in range(1, 7)
    ]
    windows = [
        {
            "ordinal": index,
            "window_id": f"sample-medium-w{index:02d}",
            "start_seconds": float(index * 4),
            "end_seconds": float(index * 4 + 4),
            "camera_ids": [f"cam_{camera:02d}" for camera in range(1, 7)],
        }
        for index in range(2)
    ]
    return {
        "format": "robata-production-shaped-cohort-v1",
        "source": {
            "path": "data/source/sample-medium.mcap",
            "camera_count": 6,
            "cameras": cameras,
            "common_duration_seconds": duration,
        },
        "windows": windows,
    }


def test_benchmark_compares_serial_and_batch_arms_without_reordering() -> None:
    backend = _FakeBackend()
    groups = [
        ("cam_01-w00-0", "cam_01-w00-1"),
        ("cam_02-w00-0", "cam_02-w00-1"),
        ("cam_03-w00-0", "cam_03-w00-1"),
    ]
    contexts = [
        {
            "camera_id": f"cam_{index:02d}",
            "window_id": "w00",
            "selected_timestamps_ns": [index, index + 1],
        }
        for index in range(1, 4)
    ]
    report = run_video_microbatch_benchmark(
        backend,
        groups,
        metadata_groups=[{"fps": 1.0}] * 3,
        item_contexts=contexts,
        batch_sizes=(2, 4),
    )

    assert report["format"] == WEMM_VIDEO_MICROBATCH_BENCHMARK_FORMAT
    assert report["status"] == "MEASURED_NONPRODUCTION"
    assert backend.serial_calls == 1
    assert backend.batch_calls == [2, 4]
    assert [arm["arm_id"] for arm in report["arms"]] == ["batch2", "batch4"]  # type: ignore[index]
    assert all(
        arm["parity"]["within_tolerance"] is True  # type: ignore[index]
        for arm in report["arms"]  # type: ignore[index]
    )
    assert report["arms"][0]["ordered_items"] == contexts  # type: ignore[index]
    assert report["arms"][0]["observations"][0]["batch"] == 2  # type: ignore[index]
    assert report["controls"]["hash_or_sha_used"] is False  # type: ignore[index]
    json.dumps(report)


def test_cohort_plan_is_bounded_six_camera_and_ordered(tmp_path: Path) -> None:
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    plan = build_cohort_microbatch_plan(path)

    assert plan["status"] == "PLAN_ONLY"
    assert plan["source"]["camera_count"] == 6  # type: ignore[index]
    assert plan["source"]["camera_window_input_count"] == 12  # type: ignore[index]
    assert plan["config"]["batch_sizes"] == [2, 4]  # type: ignore[index]
    assert [row["arm_id"] for row in plan["matrix"]] == [  # type: ignore[index]
        "f4_current",
        "f8_current",
        "f4_high",
        "f8_high",
    ]
    assert plan["matrix"][0]["expected_video_grid_thw"] == [[2, 14, 16]]  # type: ignore[index]
    assert plan["controls"]["model_invoked"] is False  # type: ignore[index]
    json.dumps(plan)


def test_cohort_plan_rejects_duration_over_limit() -> None:
    with pytest.raises(WemmVideoMicrobatchBenchmarkError, match="exceeds bounded limit"):
        build_cohort_microbatch_plan(_manifest(40.9))


def test_benchmark_rejects_context_and_metadata_mismatch() -> None:
    backend = _FakeBackend()
    with pytest.raises(WemmVideoMicrobatchBenchmarkError, match="metadata_groups"):
        run_video_microbatch_benchmark(
            backend,
            [("f0", "f1")],
            metadata_groups=[],
        )
    with pytest.raises(WemmVideoMicrobatchBenchmarkError, match="item_contexts"):
        run_video_microbatch_benchmark(
            backend,
            [("f0", "f1")],
            item_contexts=[],
        )


def test_decoded_group_helper_keeps_window_major_camera_order() -> None:
    backend = _FakeBackend()
    decoded = {
        "cam_02": {"w01": _Group("cam_02", "w01"), "w00": _Group("cam_02", "w00")},
        "cam_01": {"w01": _Group("cam_01", "w01"), "w00": _Group("cam_01", "w00")},
    }
    report = run_decoded_video_microbatch_benchmark(
        backend,
        decoded,
        camera_order=("cam_01", "cam_02"),
        window_order=("w00", "w01"),
        batch_sizes=(2,),
    )
    assert report["source"]["flatten_order"] == "window_major_camera_minor"  # type: ignore[index]
    contexts = report["arms"][0]["ordered_items"]  # type: ignore[index]
    assert [(row["window_id"], row["camera_id"]) for row in contexts] == [  # type: ignore[index]
        ("w00", "cam_01"),
        ("w00", "cam_02"),
        ("w01", "cam_01"),
        ("w01", "cam_02"),
    ]
    assert all(row["selected_timestamps_ns"] == [1, 2] for row in contexts)  # type: ignore[index]
