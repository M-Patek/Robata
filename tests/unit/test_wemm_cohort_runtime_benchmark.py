from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, ClassVar

import pytest

from robata.benchmark.production_wemm_decode_cache import ProductionWemmDecodeCache
from robata.benchmark.production_wemm_open_runner import OpenPhrase
from robata.benchmark.wemm_cohort_runtime_benchmark import (
    AUTHORITY,
    CAMERA_IDS,
    COHORT_FORMAT,
    FORMAT,
    WemmCohortRuntimeBenchmarkError,
    run_wemm_cohort_runtime_benchmark,
)


@dataclass
class _FakeFrame:
    token: str
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeGroup:
    camera_id: str
    window_id: str
    ordinal: int
    frames: tuple[_FakeFrame, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "total_num_frames": len(self.frames),
            "fps": 1.0,
            "width": 32,
            "height": 32,
            "duration": 4.0,
            "video_backend": "fixture",
            "frames_indices": list(range(len(self.frames))),
            # The fixture metadata intentionally carries a unique slot key so
            # the test can prove that flattening does not detach metadata from
            # its corresponding frame group.
            "camera_id": self.camera_id,
            "window_id": self.window_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "window_id": self.window_id,
            "ordinal": self.ordinal,
            "decoded_frames": len(self.frames),
            "selected_timestamps_ns": [
                self.ordinal * 4_000_000_000 + index for index in range(len(self.frames))
            ],
        }


@dataclass
class _FakeObservation:
    modality: str
    item_count: int
    batch_size: int | None = None
    frame_count: int | None = None
    requested_batch_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modality": self.modality,
            "item_count": self.item_count,
        }
        if self.batch_size is not None:
            payload["batch_size"] = self.batch_size
        if self.frame_count is not None:
            payload["frame_count"] = self.frame_count
        if self.requested_batch_size is not None:
            payload["requested_batch_size"] = self.requested_batch_size
        return payload


class _FakeBackend:
    instances: ClassVar[list[_FakeBackend]] = []
    drift_token: ClassVar[str | None] = None

    def __init__(
        self,
        model_directory: str | Path,
        *,
        device: str,
        dimension: int,
        video_max_pixels: int,
        **kwargs: Any,
    ) -> None:
        self.model_directory = Path(model_directory)
        self.device = device
        self.dimension = dimension
        self.video_max_pixels = video_max_pixels
        self.extra_kwargs = kwargs
        # Lightweight identity attributes mirror the real backend's telemetry
        # surface and make this fake tolerant of optional constructor knobs.
        self.identity = "fixture-wemm"
        self.variant = "fixture"
        self.supported_dimensions = (dimension,)
        self.dtype = "float32"
        self.observations: list[_FakeObservation] = []
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        type(self).instances.append(self)

    def encode_texts(self, texts: list[str], *, batch_size: int) -> tuple[tuple[float, ...], ...]:
        assert batch_size == 32
        assert texts == ["alpha", "beta"]
        self.observations.append(_FakeObservation("text", len(texts)))
        return ((1.0, 0.0), (0.0, 1.0))

    @classmethod
    def _rows(
        cls,
        groups: list[Any],
        *,
        drift: bool = False,
    ) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for group in groups:
            token = str(group[0].token)
            # Encode the camera/window slot into the first coordinate.  This
            # makes output order observable without exposing frame content in
            # the benchmark report.
            camera_id, window_id, _frame_index = token.split("-", 2)
            camera_number = int(camera_id.rsplit("_", 1)[1])
            window_number = int(window_id.removeprefix("w"))
            row = (float(window_number * 10 + camera_number), 1.0)
            if drift and cls.drift_token == token:
                row = (row[0] + 0.01, row[1])
            rows.append(row)
        return tuple(rows)

    def encode_video_frames(
        self,
        frame_groups: Any,
        *,
        metadata_groups: Any = None,
    ) -> tuple[tuple[float, ...], ...]:
        groups = list(frame_groups)
        metadata = list(metadata_groups) if metadata_groups is not None else []
        self.calls.append(
            {
                "kind": "serial",
                "tokens": [str(group[0].token) for group in groups],
                "metadata": metadata,
            }
        )
        self.observations.append(
            _FakeObservation(
                "video",
                len(groups),
                batch_size=1,
                frame_count=sum(len(group) for group in groups),
            )
        )
        rows = self._rows(groups)
        self.calls[-1]["rows"] = rows
        return rows

    def encode_video_frames_batch(
        self,
        frame_groups: Any,
        *,
        metadata_groups: Any = None,
        batch_size: int,
    ) -> tuple[tuple[float, ...], ...]:
        groups = list(frame_groups)
        metadata = list(metadata_groups) if metadata_groups is not None else []
        self.calls.append(
            {
                "kind": "batch",
                "batch_size": batch_size,
                "tokens": [str(group[0].token) for group in groups],
                "metadata": metadata,
            }
        )
        # Match the backend's one-observation-per-microbatch shape closely
        # enough to exercise observation slicing without model dependencies.
        rows = self._rows(groups, drift=True)
        for start in range(0, len(groups), batch_size):
            actual_batch_size = min(batch_size, len(groups) - start)
            self.observations.append(
                _FakeObservation(
                    "video",
                    actual_batch_size,
                    batch_size=actual_batch_size,
                    frame_count=sum(
                        len(group) for group in groups[start : start + actual_batch_size]
                    ),
                    requested_batch_size=batch_size,
                )
            )
        self.calls[-1]["rows"] = rows
        return rows

    def close(self) -> None:
        self.closed = True


class _TelemetryCuda:
    def __init__(self) -> None:
        self.reset_calls = 0

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def memory_allocated(self) -> int:
        return 100

    def memory_reserved(self) -> int:
        return 200

    def max_memory_allocated(self) -> int:
        return 300

    def max_memory_reserved(self) -> int:
        return 400

    def mem_get_info(self) -> tuple[int, int]:
        return (900, 1_000)

    def get_device_name(self, _ordinal: int) -> str:
        return "fixture-gpu"


class _TelemetryTorch:
    def __init__(self) -> None:
        self.cuda = _TelemetryCuda()


def _manifest(window_count: int = 10) -> dict[str, Any]:
    return {
        "format": COHORT_FORMAT,
        "authority": AUTHORITY,
        "source": {
            "path": "fixture.mcap",
            "media_type": "application/x-mcap",
            "camera_count": 6,
            "common_duration_seconds": 40.833423,
            "cameras": [
                {
                    "camera_id": camera_id,
                    "topic": f"/fixture/{camera_id}",
                    "frame_count": 1226,
                    "duration_seconds": 40.833423,
                }
                for camera_id in CAMERA_IDS
            ],
        },
        "windows": [
            {
                "ordinal": index,
                "window_id": f"w{index:02d}",
                "start_seconds": float(index * 4),
                "end_seconds": float(index * 4 + 4),
                "camera_ids": list(CAMERA_IDS),
            }
            for index in range(window_count)
        ],
    }


def _catalog_rows() -> list[OpenPhrase]:
    return [
        OpenPhrase("phrase_alpha", "alpha", (("canonical", "alpha"),), {}),
        OpenPhrase("phrase_beta", "beta", (("canonical", "beta"),), {}),
    ]


def _iterator_factory(calls: list[dict[str, Any]]):
    def iterator_factory(
        manifest: dict[str, Any], *, frame_count: int, window_chunk_size: int
    ) -> Any:
        windows = list(manifest["windows"])
        calls.append(
            {
                "frame_count": frame_count,
                "window_chunk_size": window_chunk_size,
                "window_ids": [str(window["window_id"]) for window in windows],
            }
        )
        # Make the decode timing assertion meaningful without opening media.
        sleep(0.001)
        for chunk_start in range(0, len(windows), window_chunk_size):
            chunk = windows[chunk_start : chunk_start + window_chunk_size]
            yield {
                camera_id: {
                    str(window["window_id"]): _FakeGroup(
                        camera_id,
                        str(window["window_id"]),
                        int(window.get("ordinal", chunk_start)),
                        tuple(
                            _FakeFrame(f"{camera_id}-{window['window_id']}-{frame_index}")
                            for frame_index in range(frame_count)
                        ),
                    )
                    for window in chunk
                }
                for camera_id in CAMERA_IDS
            }

    return iterator_factory


@pytest.fixture
def patched_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    module = importlib.import_module("robata.benchmark.wemm_cohort_runtime_benchmark")
    _FakeBackend.instances.clear()
    monkeypatch.setattr(_FakeBackend, "drift_token", None)
    iterator_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "WemmEmbeddingBackend", _FakeBackend)
    monkeypatch.setattr(
        module,
        "load_open_phrase_catalog",
        lambda _catalog: (tuple(_catalog_rows()), {"status": "FIXTURE"}),
    )
    monkeypatch.setattr(
        module,
        "iter_decode_production_window_chunks",
        _iterator_factory(iterator_calls),
    )
    return {"module": module, "iterator_calls": iterator_calls}


def test_runtime_benchmark_flattens_window_camera_order_and_reports_decode(
    patched_runtime: dict[str, Any],
) -> None:
    report = run_wemm_cohort_runtime_benchmark(
        _manifest(),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=4,
        pixel_budget=1234,
        dimension=2,
        device="cpu",
        window_chunk_size=2,
        batch_sizes=(2, 4),
        max_windows=2,
    )

    assert report["format"] == FORMAT
    assert report["status"] == "MEASURED_NONPRODUCTION"
    assert report["production_eligible"] is False
    assert report["source"]["camera_window_input_count"] == 12
    assert report["source"]["chunks_seen"] == 1
    input_order = report["source"]["input_order"]
    assert [item["row_index"] for item in input_order] == list(range(12))
    assert [(item["window_id"], item["camera_id"]) for item in input_order] == [
        (f"w{window:02d}", camera_id) for window in (0, 1) for camera_id in CAMERA_IDS
    ]
    assert all(item["frame_count"] == 4 for item in input_order)
    assert input_order[0]["selected_timestamps_ns"] == [0, 1, 2, 3]
    assert input_order[6]["selected_timestamps_ns"] == [4_000_000_000 + i for i in range(4)]
    assert report["model"]["video_max_pixels"] == 1234
    assert report["model"]["backend_identity"] == "fixture-wemm"
    assert report["model"]["backend_variant"] == "fixture"
    assert tuple(report["model"]["supported_dimensions"]) == (2,)
    assert report["controls"]["gold_read"] is False

    backend = _FakeBackend.instances[-1]
    expected_tokens = [
        f"{camera_id}-w{window:02d}-0" for window in (0, 1) for camera_id in CAMERA_IDS
    ]
    expected_slots = [
        (camera_id, f"w{window:02d}") for window in (0, 1) for camera_id in CAMERA_IDS
    ]
    expected_rows = tuple(
        (float(window * 10 + camera_index), 1.0)
        for window in (0, 1)
        for camera_index in range(1, 7)
    )
    # Warm-up calls contain one group; retain only the measured calls here so
    # this assertion is about the benchmark's flattening contract, not the
    # optional warm-kernel strategy.
    actual_calls = [call for call in backend.calls if call["tokens"] == expected_tokens]
    assert [call["kind"] for call in actual_calls] == ["serial", "batch", "batch"]
    # Every arm receives the same window-major/camera-minor flattening and the
    # metadata remains paired with that exact slot.
    for call in actual_calls:
        assert call["tokens"] == expected_tokens
        assert call["rows"] == expected_rows
        assert [
            (meta["camera_id"], meta["window_id"]) for meta in call["metadata"]
        ] == expected_slots
        assert all(meta["total_num_frames"] == 4 for meta in call["metadata"])
        assert all(meta["frames_indices"] == [0, 1, 2, 3] for meta in call["metadata"])
        assert all(meta["video_backend"] == "fixture" for meta in call["metadata"])

    serial, batch2, batch4 = report["arms"]
    assert [arm["arm_id"] for arm in report["arms"]] == ["serial", "batch2", "batch4"]
    assert serial["control"] is True
    assert batch2["parity_vs_serial"]["row_count_equal"] is True
    assert batch2["parity_vs_serial"]["max_abs_delta"] == 0.0
    assert batch2["parity_vs_serial"]["row_order_preserved"] is True
    assert batch2["parity_vs_serial"]["order_context_count"] == 12
    assert batch2["parity_vs_serial"]["mismatches"] == []
    assert batch4["parity_vs_serial"]["full_order_equal_fraction"] == 1.0
    assert serial["rank_diagnostic"]["top_label_counts_not_gold"]
    assert serial["observations"] == [
        {
            "modality": "video",
            "item_count": 12,
            "batch_size": 1,
            "frame_count": 48,
        }
    ]
    assert len(batch2["observations"]) == 6
    assert all(item["item_count"] == 2 for item in batch2["observations"])
    assert all(item["batch_size"] == 2 for item in batch2["observations"])
    assert all(item["requested_batch_size"] == 2 for item in batch2["observations"])
    assert all(item["frame_count"] == 8 for item in batch2["observations"])
    assert len(batch4["observations"]) == 3
    assert all(item["item_count"] == 4 for item in batch4["observations"])
    assert all(item["batch_size"] == 4 for item in batch4["observations"])
    assert all(item["requested_batch_size"] == 4 for item in batch4["observations"])
    assert all(item["frame_count"] == 16 for item in batch4["observations"])
    assert serial["decode_seconds_shared"] > 0.0
    assert backend.closed
    assert len(patched_runtime["iterator_calls"]) == 1
    assert patched_runtime["iterator_calls"][0]["frame_count"] == 4
    json.dumps(report)


def test_runtime_benchmark_can_replay_explicit_decode_cache_scope(
    patched_runtime: dict[str, Any],
) -> None:
    """A matrix arm can reuse decoded groups without a second decoder pass."""

    cache = ProductionWemmDecodeCache(max_scopes=2)
    first = run_wemm_cohort_runtime_benchmark(
        _manifest(window_count=1),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=4,
        pixel_budget=1234,
        dimension=2,
        device="cpu",
        window_chunk_size=1,
        batch_sizes=(2,),
        decode_cache=cache,
        decode_scope_key=("fixture-cohort", "f4", "p1234", "chunk1"),
    )
    second = run_wemm_cohort_runtime_benchmark(
        _manifest(window_count=1),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=4,
        pixel_budget=2048,
        dimension=2,
        device="cpu",
        window_chunk_size=1,
        batch_sizes=(2,),
        decode_cache=cache,
        # The scope key is intentionally caller-owned.  Reusing it here is a
        # deliberate benchmark assertion that only the caller may decide two
        # manifests/configurations are equivalent.
        decode_scope_key=("fixture-cohort", "f4", "p1234", "chunk1"),
    )

    assert len(patched_runtime["iterator_calls"]) == 1
    assert first["decode_cache"]["miss_count"] == 1
    assert second["decode_cache"]["hit_count"] == 1
    assert second["source"]["camera_window_input_count"] == 6
    cache.clear()


def test_runtime_benchmark_records_lightweight_cpu_and_cuda_memory_telemetry(
    patched_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allocator peaks are arm-scoped, while CPU RSS remains boundary-only."""

    module = patched_runtime["module"]
    telemetry_torch = _TelemetryTorch()
    monkeypatch.setattr(_FakeBackend, "_torch", telemetry_torch, raising=False)
    monkeypatch.setattr(module, "_process_rss_bytes", lambda: 12_345)
    report = run_wemm_cohort_runtime_benchmark(
        _manifest(window_count=1),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=4,
        dimension=2,
        device="cpu",
        window_chunk_size=1,
        batch_sizes=(4,),
    )

    telemetry = report["arms"][1]["memory_telemetry"]
    assert telemetry["status"] == "AVAILABLE"
    assert telemetry["cpu"] == {
        "status": "AVAILABLE",
        "rss_before_bytes": 12_345,
        "rss_after_bytes": 12_345,
        "rss_delta_bytes": 0,
        "source": "psutil.Process.memory_info.rss",
        "peak_scope": "boundary_samples_only",
    }
    gpu = telemetry["gpu"]
    assert gpu["device_name"] == "fixture-gpu"
    assert gpu["allocated_before_bytes"] == 100
    assert gpu["allocated_after_bytes"] == 100
    assert gpu["peak_allocated_bytes"] == 300
    assert gpu["peak_reserved_bytes"] == 400
    assert gpu["total_bytes"] == 1_000
    # serial + Batch4 each reset once for this one decoded chunk.
    assert telemetry_torch.cuda.reset_calls == 2


def test_runtime_parity_keeps_mismatch_context_and_processor_metadata(
    patched_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberately perturb one batched row.  The comparator must report the
    # numerical mismatch *with* its camera/window context and the metadata
    # paired to that row, rather than just returning an aggregate score.
    monkeypatch.setattr(_FakeBackend, "drift_token", "cam_03-w01-0")
    report = run_wemm_cohort_runtime_benchmark(
        _manifest(),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=4,
        dimension=2,
        device="cpu",
        window_chunk_size=2,
        batch_sizes=(2,),
        max_windows=2,
    )

    parity = report["arms"][1]["parity_vs_serial"]
    assert parity["row_count_equal"] is True
    assert parity["dimension_equal"] is True
    assert parity["serial_row_count"] == parity["batch_row_count"] == 12
    assert parity["order_context_count"] == 12
    assert parity["row_order_preserved"] is True
    assert parity["within_tolerance"] is False
    assert parity["max_abs_delta"] == pytest.approx(0.01)
    mismatch = next(item for item in parity["mismatches"] if item["row_index"] == 8)
    context = mismatch["context"]
    assert context["row_index"] == 8
    assert context["camera_id"] == "cam_03"
    assert context["window_id"] == "w01"
    assert context["decoded_frames"] == 4
    processor_metadata = context["processor_video_metadata"]
    assert processor_metadata["camera_id"] == "cam_03"
    assert processor_metadata["window_id"] == "w01"
    assert processor_metadata["total_num_frames"] == 4
    assert processor_metadata["frames_indices"] == [0, 1, 2, 3]
    json.dumps(report)


def test_max_windows_uses_selected_duration_for_camera_throughput(
    patched_runtime: dict[str, Any],
) -> None:
    report = run_wemm_cohort_runtime_benchmark(
        _manifest(),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=4,
        dimension=2,
        device="cpu",
        window_chunk_size=1,
        batch_sizes=(2, 4),
        max_windows=1,
    )

    assert report["source"]["window_count"] == 1
    assert report["source"]["camera_window_input_count"] == 6
    assert report["source"]["represented_window_seconds"] == pytest.approx(4.0)
    for arm in report["arms"]:
        elapsed = arm["estimated_e2e_seconds"]
        assert elapsed > 0.0
        # Six cameras each contribute the selected four-second window.  Using
        # the full 40.833423-second source duration here would inflate this
        # metric by roughly 10x, which max_windows must not do.
        assert arm["source_camera_normalized_realtime"] == pytest.approx(
            6 * 4.0 / elapsed,
            rel=1e-8,
        )
        assert arm["source_camera_normalized_realtime"] != pytest.approx(
            6 * 40.833423 / elapsed,
            rel=1e-3,
        )
    assert len(patched_runtime["iterator_calls"]) == 1


def test_runtime_pipeline_arm_is_bounded_and_records_phases(
    patched_runtime: dict[str, Any],
) -> None:
    report = run_wemm_cohort_runtime_benchmark(
        _manifest(),
        phrase_catalog={"fixture": True},
        model_directory="fixture-model",
        frame_count=2,
        dimension=2,
        device="cpu",
        window_chunk_size=2,
        batch_sizes=(2, 4),
        max_windows=3,
        include_pipeline=True,
        queue_capacity=2,
    )

    pipeline = report["pipeline"]
    assert pipeline["batch_size"] == 4
    assert pipeline["queue_capacity"] == 2
    assert pipeline["chunk_count"] == 2
    timing = pipeline["timing"]
    assert timing["status"] == "SUCCEEDED"
    assert timing["queue_capacity"] == 2
    assert timing["offered_item_count"] == 2
    assert timing["produced_item_count"] == 2
    assert timing["consumed_item_count"] == 2
    phase_names = {phase["name"] for phase in timing["phase_totals"]}
    assert {"media_decode", "model"} <= phase_names
    assert all(item["succeeded"] for item in timing["items"])
    assert len(patched_runtime["iterator_calls"]) == 2
    assert patched_runtime["iterator_calls"][1]["frame_count"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 1}, "frame_count"),
        ({"frame_count": 65}, "frame_count"),
        ({"pixel_budget": 0}, "pixel_budget"),
        ({"dimension": 0}, "dimension"),
        ({"window_chunk_size": 0}, "window_chunk_size"),
        ({"batch_sizes": (0,)}, r"batch_sizes\[0\]"),
        ({"batch_sizes": (65,)}, r"batch_sizes\[0\]"),
        ({"max_windows": 0}, "max_windows"),
        ({"queue_capacity": 0, "include_pipeline": True}, "queue_capacity"),
    ],
)
def test_runtime_rejects_invalid_bounds_before_loading_backend(
    patched_runtime: dict[str, Any],
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(WemmCohortRuntimeBenchmarkError, match=message):
        run_wemm_cohort_runtime_benchmark(
            _manifest(),
            phrase_catalog={"fixture": True},
            model_directory="fixture-model",
            **kwargs,
        )
    assert not _FakeBackend.instances


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifest: manifest.update({"format": "other"}),
        lambda manifest: manifest.update({"authority": "OTHER"}),
        lambda manifest: manifest["source"].update({"camera_count": 5}),
        lambda manifest: manifest["source"]["cameras"].__setitem__(0, {"camera_id": "cam_06"}),
        lambda manifest: manifest["source"].update({"common_duration_seconds": 40.8346}),
        lambda manifest: manifest["windows"][0].update({"start_seconds": 4, "end_seconds": 4}),
        lambda manifest: manifest["windows"].__setitem__(
            1, {"window_id": "w00", "start_seconds": 4, "end_seconds": 8}
        ),
        lambda manifest: manifest.update({"windows": []}),
    ],
)
def test_runtime_rejects_invalid_cohort_shape(
    patched_runtime: dict[str, Any],
    mutator: Any,
) -> None:
    manifest = _manifest()
    mutator(manifest)
    with pytest.raises(WemmCohortRuntimeBenchmarkError):
        run_wemm_cohort_runtime_benchmark(
            manifest,
            phrase_catalog={"fixture": True},
            model_directory="fixture-model",
        )
    assert not _FakeBackend.instances
