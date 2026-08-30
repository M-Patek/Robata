from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_decode_cache import ProductionWemmDecodeCache
from robata.benchmark.production_wemm_open_runner import (
    PHRASE_CATALOG_FORMAT,
    ProductionWemmOpenRunnerError,
    _expand_model_observations,
    dry_run_open_phrase_plan,
    load_open_phrase_catalog,
    run_production_wemm_open,
)


def test_batch_model_observations_expand_in_input_order() -> None:
    class Observation:
        def __init__(self, item_count: int, batch_index: int) -> None:
            self.item_count = item_count
            self.batch_index = batch_index

        def to_dict(self):
            return {
                "modality": "video",
                "item_count": self.item_count,
                "batch_index": self.batch_index,
            }

    rows = _expand_model_observations((Observation(2, 0), Observation(1, 1)), expected_count=3)
    assert [row["batch_member_index"] for row in rows[:2]] == [0, 1]
    assert rows[2]["batch_index"] == 1
    assert _expand_model_observations((Observation(2, 0),), expected_count=1) == ()


def _catalog() -> dict[str, object]:
    return {
        "format": PHRASE_CATALOG_FORMAT,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROVISIONAL_NON_GOLD",
        "phrases": [
            {
                "provisional_id": "open_cupboard",
                "label_text": "open cupboard",
                "texts": {
                    "canonical": "open cupboard",
                    "verb_noun": "verb: open; noun: cupboard",
                    "natural": "a person is opening a cupboard",
                },
                "structured_labels": {"verb": "open", "noun": "cupboard"},
            },
            {
                "provisional_id": "open_drawer",
                "label_text": "open drawer",
                "structured_labels": {"verb": "open", "noun": "drawer"},
            },
        ],
    }


def _manifest() -> dict[str, object]:
    return {
        "format": "robata-production-shaped-cohort-v1",
        "source": {
            "path": "data/source/sample-medium.mcap",
            "camera_count": 2,
            "cameras": [
                {"camera_id": "cam_01", "topic": "/cam/1"},
                {"camera_id": "cam_02", "topic": "/cam/2"},
            ],
        },
        "windows": [
            {
                "ordinal": 0,
                "window_id": "w00",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            },
            {
                "ordinal": 1,
                "window_id": "w01",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
        ],
    }


def test_catalog_loads_arbitrary_phrases_and_generates_opaque_ids() -> None:
    labels, metadata = load_open_phrase_catalog(
        {"phrases": ["wipe the counter", {"label_text": "move tray"}]}
    )
    assert [label.label_text for label in labels] == ["wipe the counter", "move tray"]
    assert labels[0].provisional_id.startswith("phrase_001_")
    assert metadata["epic_ontology_used"] is False
    assert metadata["production_eligible"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"phrases": [{"label_text": "open drawer", "action_key": [1, 2]}]},
        {"phrases": [{"label_text": "open drawer", "verb_id": 3}]},
        {"phrases": [{"label_text": "open drawer", "ground_truth": "x"}]},
    ],
)
def test_catalog_rejects_epic_or_gold_identity(bad: dict[str, object]) -> None:
    with pytest.raises(ProductionWemmOpenRunnerError, match=r"EPIC/gold|numeric EPIC"):
        load_open_phrase_catalog(bad)


def test_catalog_rejects_duplicate_text_or_id() -> None:
    with pytest.raises(ProductionWemmOpenRunnerError, match="duplicate label text"):
        load_open_phrase_catalog({"phrases": ["open drawer", "OPEN DRAWER"]})
    with pytest.raises(ProductionWemmOpenRunnerError, match="duplicate provisional_id"):
        load_open_phrase_catalog(
            {
                "phrases": [
                    {"provisional_id": "same", "label_text": "open drawer"},
                    {"provisional_id": "same", "label_text": "open cupboard"},
                ]
            }
        )


def test_dry_run_does_not_decode_or_invoke_model() -> None:
    report = dry_run_open_phrase_plan(_manifest(), phrase_catalog=_catalog(), max_windows=1)
    assert report["status"] == "DRY_RUN"
    assert report["source"]["window_count"] == 1  # type: ignore[index]
    assert report["catalog"]["phrase_count"] == 2  # type: ignore[index]
    assert report["controls"]["media_decoded"] is False  # type: ignore[index]
    assert report["controls"]["model_invoked"] is False  # type: ignore[index]
    json.dumps(report)


def test_runner_builds_review_only_envelope_with_camera_top_k(monkeypatch) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    class Group:
        frames = ("frame-0", "frame-1")

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {"camera_id": "cam_01", "window_id": "w00", "frame_count": 2}

    monkeypatch.setattr(
        route,
        "decode_production_windows",
        lambda manifest, **kwargs: {
            "cam_01": {"w00": Group()},
            "cam_02": {"w00": Group()},
        },
    )

    class Observation:
        def to_dict(self):
            return {"modality": "video", "frame_count": 2}

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            self.observations.append(Observation())
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    manifest = _manifest()
    manifest["source"] = {
        "path": "data/source/sample-medium.mcap",
        "camera_count": 2,
        "cameras": [
            {"camera_id": "cam_01", "topic": "/cam/1"},
            {"camera_id": "cam_02", "topic": "/cam/2"},
        ],
    }
    report = run_production_wemm_open(
        manifest,
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        max_windows=1,
    )
    assert report["format"] == "robata-production-wemm-preannotation-v1"
    assert report["label_space"]["kind"] == "OPEN_PROVISIONAL_PHRASES"  # type: ignore[index]
    assert report["controls"]["model_invoked"] is True  # type: ignore[index]
    assert report["production_eligible"] is False
    assert report["model"]["window_chunk_size"] == 1  # type: ignore[index]
    assert "producer_consumer" not in report["model"]  # type: ignore[operator]
    proposal = report["windows"][0]["proposals"][0]  # type: ignore[index]
    assert proposal["label_text"] == "open cupboard"
    assert proposal["top_k"][0]["label_text"] == "open cupboard"
    assert len(proposal["evidence"]) == 2
    assert proposal["proposal_interval"]["status"] == "NOT_MEASURED"
    json.dumps(report)


def test_runner_rejects_invalid_frame_count_before_model(monkeypatch) -> None:
    called = False

    def fail_backend(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    import robata.benchmark.production_wemm_open_runner as route

    monkeypatch.setattr(route, "WemmEmbeddingBackend", fail_backend)
    with pytest.raises(ProductionWemmOpenRunnerError, match="frame_count"):
        run_production_wemm_open(
            _manifest(),
            phrase_catalog=_catalog(),
            model_directory="model",
            frame_count=1,
        )
    assert called is False


def test_runner_rejects_invalid_video_pixel_bounds_before_model(monkeypatch) -> None:
    called = False

    def fail_backend(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    import robata.benchmark.production_wemm_open_runner as route

    monkeypatch.setattr(route, "WemmEmbeddingBackend", fail_backend)
    with pytest.raises(ProductionWemmOpenRunnerError, match="video_min_pixels"):
        run_production_wemm_open(
            _manifest(),
            phrase_catalog=_catalog(),
            model_directory="model",
            video_min_pixels=524288,
            video_max_pixels=262144,
        )
    assert called is False


def test_runner_forwards_video_pixel_bounds_and_records_them(monkeypatch) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    class Group:
        frames = ("frame-0", "frame-1")

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {"camera_id": "cam_01", "window_id": "w00", "frame_count": 2}

    monkeypatch.setattr(
        route,
        "decode_production_windows",
        lambda manifest, **kwargs: {
            "cam_01": {"w00": Group()},
            "cam_02": {"w00": Group()},
        },
    )

    backend_kwargs: list[dict[str, object]] = []

    class Observation:
        def to_dict(self):
            return {"modality": "video", "frame_count": 2}

    class FakeBackend:
        def __init__(self, **kwargs):
            backend_kwargs.append(kwargs)
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            self.observations.append(Observation())
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    report = run_production_wemm_open(
        _manifest(),
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        dimension=2,
        max_windows=1,
        video_min_pixels=4096,
        video_max_pixels=524288,
    )
    assert backend_kwargs == [
        {
            "model_directory": "model",
            "device": "cuda",
            "dimension": 2,
            "video_min_pixels": 4096,
            "video_max_pixels": 524288,
        }
    ]
    assert report["model"]["video_min_pixels"] == 4096  # type: ignore[index]
    assert report["model"]["video_max_pixels"] == 524288  # type: ignore[index]


def test_runner_decodes_windows_in_bounded_chunks_and_merges_in_order(monkeypatch) -> None:
    """Chunking must bound decode retention without changing envelope order."""

    import robata.benchmark.production_wemm_open_runner as route

    decode_calls: list[list[str]] = []

    class Group:
        def __init__(self, window_id: str, camera_id: str) -> None:
            self.frames = (f"{camera_id}-{window_id}-frame-0", f"{camera_id}-{window_id}-frame-1")
            self.window_id = window_id
            self.camera_id = camera_id

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 2,
            }

    def fake_decode(manifest, **kwargs):
        del kwargs
        window_rows = manifest["windows"]
        window_ids = [str(row["window_id"]) for row in window_rows]
        decode_calls.append(window_ids)
        return {
            camera_id: {window_id: Group(window_id, camera_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "decode_production_windows", fake_decode)

    class Observation:
        def to_dict(self):
            return {"modality": "video", "frame_count": 2}

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            self.observations.append(Observation())
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    report = run_production_wemm_open(
        _manifest(),
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        window_chunk_size=1,
    )

    assert decode_calls == [["w00"], ["w01"]]
    assert [row["window_id"] for row in report["windows"]] == ["w00", "w01"]
    assert len(report["raw_model_output"]["windows"]) == 2
    assert len(report["raw_model_output"]["backend_observations"]) == 4
    assert report["model"]["window_chunk_size"] == 1  # type: ignore[index]


def test_runner_uses_opt_in_inference_batch_and_preserves_window_camera_order(monkeypatch) -> None:
    """Batch mode must flatten window-major/camera-minor and rebuild the envelope."""

    import robata.benchmark.production_wemm_open_runner as route

    class Group:
        def __init__(self, window_id: str, camera_id: str) -> None:
            self.frames = (
                f"{camera_id}-{window_id}-frame-0",
                f"{camera_id}-{window_id}-frame-1",
            )
            self.window_id = window_id
            self.camera_id = camera_id

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 2,
            }

    def fake_decode(manifest, **kwargs):
        del kwargs
        window_ids = [str(row["window_id"]) for row in manifest["windows"]]
        return {
            camera_id: {window_id: Group(window_id, camera_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "decode_production_windows", fake_decode)

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []
            self.batch_calls: list[dict[str, object]] = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames_batch(self, groups, *, metadata_groups=None, batch_size=2):
            assert metadata_groups is not None
            group_rows = list(groups)
            metadata_rows = list(metadata_groups)
            self.batch_calls.append(
                {
                    "first_frames": [str(row[0]) for row in group_rows],
                    "metadata_count": len(metadata_rows),
                    "batch_size": batch_size,
                }
            )
            return tuple(
                (1.0, 0.0) if str(row[0]).startswith("cam_01") else (0.0, 1.0) for row in group_rows
            )

        def observation_payload(self):
            return []

        def close(self):
            return None

    backend = FakeBackend()
    report = run_production_wemm_open(
        _manifest(),
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        window_chunk_size=2,
        inference_batch_size=2,
        backend=backend,
    )

    assert len(backend.batch_calls) == 1
    assert backend.batch_calls[0] == {
        "first_frames": [
            "cam_01-w00-frame-0",
            "cam_02-w00-frame-0",
            "cam_01-w01-frame-0",
            "cam_02-w01-frame-0",
        ],
        "metadata_count": 4,
        "batch_size": 2,
    }
    assert [row["window_id"] for row in report["windows"]] == ["w00", "w01"]
    assert report["controls"]["model_invoked"] is True  # type: ignore[index]


def test_runner_reuses_explicit_decode_cache_scope(monkeypatch) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    decode_calls: list[int] = []

    class Group:
        def __init__(self, window_id: str, camera_id: str) -> None:
            self.frames = (
                f"{camera_id}-{window_id}-frame-0",
                f"{camera_id}-{window_id}-frame-1",
            )
            self.window_id = window_id
            self.camera_id = camera_id

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 2,
            }

    def fake_iter(manifest, **kwargs):
        del kwargs
        decode_calls.append(1)
        window_ids = [str(row["window_id"]) for row in manifest["windows"]]
        yield {
            camera_id: {window_id: Group(window_id, camera_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "iter_decode_production_window_chunks", fake_iter)

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            return ((1.0, 0.0),)

        def observation_payload(self):
            return []

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    cache = ProductionWemmDecodeCache()
    kwargs = {
        "phrase_catalog": _catalog(),
        "model_directory": "model",
        "frame_count": 2,
        "top_k": 2,
        "dimension": 2,
        "device": "cpu",
        "max_windows": 1,
        "decode_cache": cache,
        "decode_scope_key": ("fixture", "w00", "f2"),
    }
    first = run_production_wemm_open(_manifest(), **kwargs)
    second = run_production_wemm_open(_manifest(), **kwargs)
    assert decode_calls == [1]
    assert first["model"]["decode_cache"]["miss_count"] == 1  # type: ignore[index]
    assert second["model"]["decode_cache"]["hit_count"] == 1  # type: ignore[index]
    assert second["windows"][0]["proposals"][0]["label_text"] == "open cupboard"  # type: ignore[index]
    cache.clear()


def test_runner_pipeline_mode_preserves_envelope_order_and_contract(monkeypatch) -> None:
    """Producer/consumer scheduling must not reorder review windows."""

    import robata.benchmark.production_wemm_open_runner as route

    decode_calls: list[list[str]] = []

    class Group:
        def __init__(self, window_id: str, camera_id: str) -> None:
            self.frames = (
                f"{camera_id}-{window_id}-frame-0",
                f"{camera_id}-{window_id}-frame-1",
            )
            self.window_id = window_id
            self.camera_id = camera_id

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 2,
            }

    def fake_decode(manifest, **kwargs):
        del kwargs
        window_ids = [str(row["window_id"]) for row in manifest["windows"]]
        decode_calls.append(window_ids)
        return {
            camera_id: {window_id: Group(window_id, camera_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "decode_production_windows", fake_decode)

    class Observation:
        def to_dict(self):
            return {"modality": "video", "frame_count": 2}

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            self.observations.append(Observation())
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    report = run_production_wemm_open(
        _manifest(),
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        window_chunk_size=1,
        include_pipeline=True,
        queue_capacity=1,
    )

    assert decode_calls == [["w00"], ["w01"]]
    assert [row["window_id"] for row in report["windows"]] == ["w00", "w01"]
    assert [row["window_id"] for row in report["raw_model_output"]["windows"]] == ["w00", "w01"]
    assert report["model"]["producer_consumer"] is True  # type: ignore[index]
    assert report["model"]["queue_capacity"] == 1  # type: ignore[index]
    pipeline_timing = report["raw_model_output"]["pipeline_timing"]  # type: ignore[index]
    assert pipeline_timing["status"] == "SUCCEEDED"  # type: ignore[index]
    assert pipeline_timing["offered_item_count"] == 2  # type: ignore[index]
    assert pipeline_timing["consumed_item_count"] == 2  # type: ignore[index]
    assert report["windows"][0]["proposals"][0]["label_text"] == "open cupboard"  # type: ignore[index]
    json.dumps(report)


def test_runner_pipeline_closes_cancelled_decoded_payloads(monkeypatch) -> None:
    """A consumer failure must not leave queued frame objects undisposed."""

    import robata.benchmark.production_wemm_open_runner as route

    class Frame:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    frames: list[Frame] = []

    class Group:
        def __init__(self, window_id: str, camera_id: str) -> None:
            first, second = Frame(), Frame()
            frames.extend((first, second))
            self.frames = (first, second)
            self.window_id = window_id
            self.camera_id = camera_id

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 2,
            }

    def fake_decode(manifest, **kwargs):
        del kwargs
        window_ids = [str(row["window_id"]) for row in manifest["windows"]]
        return {
            camera_id: {window_id: Group(window_id, camera_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "decode_production_windows", fake_decode)

    class FailingBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            raise RuntimeError("synthetic consumer failure")

        def observation_payload(self):
            return []

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FailingBackend)
    with pytest.raises(ProductionWemmOpenRunnerError, match="producer-consumer pipeline failed"):
        run_production_wemm_open(
            _manifest(),
            phrase_catalog=_catalog(),
            model_directory="model",
            frame_count=2,
            top_k=2,
            dimension=2,
            device="cpu",
            window_chunk_size=1,
            include_pipeline=True,
            queue_capacity=2,
        )
    assert frames and all(frame.closed for frame in frames)


@pytest.mark.parametrize("bad", [0, True, 1.5])
def test_runner_rejects_invalid_queue_capacity_before_decode(monkeypatch, bad) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    called = False

    def fail_decode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(route, "decode_production_windows", fail_decode)
    with pytest.raises(ProductionWemmOpenRunnerError, match="queue_capacity"):
        run_production_wemm_open(
            _manifest(),
            phrase_catalog=_catalog(),
            model_directory="model",
            queue_capacity=bad,  # type: ignore[arg-type]
        )
    assert called is False


@pytest.mark.parametrize("bad", [0, 65, True, 1.5])
def test_runner_rejects_invalid_inference_batch_size_before_decode(monkeypatch, bad) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    called = False

    def fail_decode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(route, "decode_production_windows", fail_decode)
    with pytest.raises(ProductionWemmOpenRunnerError, match="inference_batch_size"):
        run_production_wemm_open(
            _manifest(),
            phrase_catalog=_catalog(),
            model_directory="model",
            inference_batch_size=bad,  # type: ignore[arg-type]
        )
    assert called is False


def test_open_runner_cli_exposes_inference_batch_size() -> None:
    script = Path(__file__).parents[2] / "scripts" / "run_production_wemm_open.py"
    namespace = runpy.run_path(str(script), run_name="robata_open_runner_cli")
    parser = namespace["_parser"]()
    base = [
        "manifest.json",
        "--phrase-catalog",
        "phrases.json",
        "--model-dir",
        "model",
        "--output",
        "output.json",
    ]
    assert parser.parse_args(base).inference_batch_size == 1
    assert parser.parse_args([*base, "--inference-batch-size", "4"]).inference_batch_size == 4
    assert parser.parse_args([*base, "--video-max-pixels", "524288"]).video_max_pixels == 524288
    assert parser.parse_args([*base, "--video-min-pixels", "4096"]).video_min_pixels == 4096
    assert parser.parse_args(base).pipeline is False
    assert parser.parse_args([*base, "--pipeline", "--queue-capacity", "2"]).pipeline is True
    assert parser.parse_args([*base, "--pipeline", "--queue-capacity", "2"]).queue_capacity == 2


def test_open_runner_cli_forwards_inference_batch_size(tmp_path) -> None:
    script = Path(__file__).parents[2] / "scripts" / "run_production_wemm_open.py"
    namespace = runpy.run_path(str(script), run_name="robata_open_runner_cli")
    calls: list[dict[str, object]] = []

    def fake_run(manifest, **kwargs):
        calls.append({"manifest": manifest, **kwargs})
        return {
            "status": "REVIEW_REQUIRED",
            "windows": [],
            "production_eligible": False,
            "controls": {"model_invoked": True},
            "label_space": {"kind": "OPEN_PROVISIONAL_PHRASES"},
        }

    # ``runpy.run_path`` returns a snapshot of the execution globals; replace
    # the function in ``main``'s actual globals so the forwarding seam is used.
    namespace["main"].__globals__["run_production_wemm_open"] = fake_run
    output = tmp_path / "output.json"
    argv = [
        "manifest.json",
        "--phrase-catalog",
        "phrases.json",
        "--model-dir",
        "model",
        "--output",
        str(output),
        "--inference-batch-size",
        "4",
        "--video-max-pixels",
        "524288",
    ]
    assert namespace["main"](argv) == 0
    assert calls and calls[0]["inference_batch_size"] == 4
    assert calls[0]["video_max_pixels"] == 524288
    assert "include_pipeline" not in calls[0]
    assert "queue_capacity" not in calls[0]
    assert output.is_file()


def test_open_runner_cli_forwards_pipeline_options(tmp_path) -> None:
    script = Path(__file__).parents[2] / "scripts" / "run_production_wemm_open.py"
    namespace = runpy.run_path(str(script), run_name="robata_open_runner_cli")
    calls: list[dict[str, object]] = []

    def fake_run(manifest, **kwargs):
        calls.append({"manifest": manifest, **kwargs})
        return {
            "status": "REVIEW_REQUIRED",
            "windows": [],
            "production_eligible": False,
            "controls": {"model_invoked": True},
            "label_space": {"kind": "OPEN_PROVISIONAL_PHRASES"},
        }

    namespace["main"].__globals__["run_production_wemm_open"] = fake_run
    output = tmp_path / "output.json"
    argv = [
        "manifest.json",
        "--phrase-catalog",
        "phrases.json",
        "--model-dir",
        "model",
        "--output",
        str(output),
        "--pipeline",
        "--queue-capacity",
        "3",
    ]
    assert namespace["main"](argv) == 0
    assert calls
    assert calls[0]["include_pipeline"] is True
    assert calls[0]["queue_capacity"] == 3
    assert output.is_file()


@pytest.mark.parametrize(
    ("max_windows", "expected_window_ids"),
    [(None, ["w00", "w01"]), (1, ["w00"])],
)
def test_runner_uses_source_bound_iterator_for_normal_media(
    monkeypatch, max_windows, expected_window_ids
) -> None:
    """Normal runs pass only selected windows to one streaming scan."""

    import robata.benchmark.production_wemm_open_runner as route

    calls: list[dict[str, object]] = []

    class Group:
        def __init__(self, window_id: str, camera_id: str) -> None:
            self.frames = (f"{camera_id}-{window_id}-0", f"{camera_id}-{window_id}-1")
            self.window_id = window_id
            self.camera_id = camera_id

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 2,
            }

    def fake_iter(manifest, **kwargs):
        calls.append({"manifest": manifest, **kwargs})
        rows = manifest["windows"]
        return iter(
            [
                {
                    camera_id: {
                        str(row["window_id"]): Group(str(row["window_id"]), camera_id)
                        for row in rows
                    }
                    for camera_id in ("cam_01", "cam_02")
                }
            ]
        )

    monkeypatch.setattr(route, "iter_decode_production_window_chunks", fake_iter)

    class Observation:
        def to_dict(self):
            return {"modality": "video", "frame_count": 2}

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del texts, batch_size
            return ((1.0, 0.0), (0.0, 1.0))

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            self.observations.append(Observation())
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

        def close(self):
            return None

    monkeypatch.setattr(route, "WemmEmbeddingBackend", FakeBackend)
    report = run_production_wemm_open(
        _manifest(),
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        window_chunk_size=2,
        max_windows=max_windows,
    )

    assert len(calls) == 1
    assert [row["window_id"] for row in calls[0]["manifest"]["windows"]] == expected_window_ids
    assert calls[0]["frame_count"] == 2
    assert calls[0]["window_chunk_size"] == 2
    assert [row["window_id"] for row in report["windows"]] == expected_window_ids


def test_runner_rejects_invalid_window_chunk_size_before_decode(monkeypatch) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    called = False

    def fail_decode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(route, "decode_production_windows", fail_decode)
    with pytest.raises(ProductionWemmOpenRunnerError, match="window_chunk_size"):
        run_production_wemm_open(
            _manifest(),
            phrase_catalog=_catalog(),
            model_directory="model",
            window_chunk_size=0,
        )
    assert called is False
