from __future__ import annotations

import json

from robata.benchmark.production_wemm_open_runner import run_production_wemm_open


def _catalog() -> dict[str, object]:
    return {
        "phrases": [
            {"provisional_id": "open_cupboard", "label_text": "open cupboard"},
            {"provisional_id": "open_drawer", "label_text": "open drawer"},
        ]
    }


def _manifest() -> dict[str, object]:
    return {
        "format": "robata-production-shaped-cohort-v1",
        "source": {
            "path": "sample.mcap",
            "cameras": [
                {"camera_id": "cam_01", "topic": "/cam/1"},
                {"camera_id": "cam_02", "topic": "/cam/2"},
            ],
        },
        "windows": [
            {"window_id": "w00", "ordinal": 0, "start_seconds": 0.0, "end_seconds": 1.0},
            {"window_id": "w01", "ordinal": 1, "start_seconds": 0.5, "end_seconds": 1.5},
        ],
    }


def test_adaptive_mode_runs_coarse_and_short_second_pass(monkeypatch) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    decode_calls: list[list[str]] = []

    class Group:
        def __init__(self, camera_id: str, window_id: str) -> None:
            self.camera_id = camera_id
            self.window_id = window_id
            self.frames = (f"{camera_id}-{window_id}-frame-0", f"{camera_id}-{window_id}-frame-1")

        def metadata(self):
            return {
                "total_num_frames": 2,
                "fps": 2.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0, 1],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": len(self.frames),
            }

    def fake_decode(manifest, **kwargs):
        del kwargs
        window_ids = [str(row["window_id"]) for row in manifest["windows"]]
        decode_calls.append(window_ids)
        return {
            camera_id: {window_id: Group(camera_id, window_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "decode_production_windows", fake_decode)

    class Observation:
        def __init__(self, call_index: int) -> None:
            self.call_index = call_index

        def to_dict(self):
            return {
                "modality": "video",
                "item_count": 1,
                "call_index": self.call_index,
            }

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []
            self.video_calls = 0

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            return tuple((1.0, 0.0) if "cupboard" in text else (0.0, 1.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del metadata_groups
            self.video_calls += 1
            del groups
            self.observations.append(Observation(self.video_calls))
            # Both coarse contexts support one action, which creates an edge
            # touching coarse segment; the adaptive route still has to issue
            # two short requests and must keep them review-only.
            return ((1.0, 0.0),)

        def observation_payload(self):
            return [item.to_dict() for item in self.observations]

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
        backend=backend,
        window_chunk_size=2,
        temporal_mode="adaptive_score",
        temporal_boundary_mode="midpoint",
        temporal_refinement_span_seconds=0.5,
    )

    # The fine score grid is clipped at the source edges, yielding six probe
    # contexts (three per role) in addition to four coarse camera inputs.
    assert backend.video_calls == 16
    assert decode_calls[0] == ["w00", "w01"]
    assert len(decode_calls) == 4  # one coarse chunk plus three fine chunks
    fine_window_ids = [item for chunk in decode_calls[1:] for item in chunk]
    assert len(fine_window_ids) == 6
    assert all(item.startswith("temporal-refinement::") for item in fine_window_ids)

    assert report["model"]["temporal_mode"] == "adaptive_score"  # type: ignore[index]
    # The orchestration mode is adaptive while the retained coarse resolver
    # report intentionally remains a dense-score artifact.
    assert report["model"]["temporal_suppress_ranking_switch_boundaries"] is True  # type: ignore[index]
    assert report["model"]["temporal_refinement_model_passes"] == 1  # type: ignore[index]
    # The outer sidecar is aligned to the four coarse camera inputs only;
    # recursive fine-pass observations belong to the nested refinement pass.
    assert len(report["raw_model_output"]["backend_observations"]) == 4
    coarse = report["temporal_resolution"]  # type: ignore[index]
    assert coarse["production_eligible"] is False
    assert coarse["mode"] == "dense_score"
    assert coarse["parameters"]["ranking_switch_suppression_active"] is True
    assert coarse["diagnostics"]["ranking_switch_suppression_active"] is True
    plan = report["temporal_refinement_plan"]  # type: ignore[index]
    assert len(plan["requests"]) == 2
    fine_plan = report["temporal_refinement_fine_plan"]  # type: ignore[index]
    assert len(fine_plan["requests"]) == 6
    refinement = report["temporal_refinement"]  # type: ignore[index]
    assert refinement["production_eligible"] is False
    assert refinement["diagnostics"]["result_count"] == 2
    pass_sidecar = report["raw_model_output"]["temporal_refinement"]["pass"]
    assert pass_sidecar["decode_provenance"]["available"] is True
    assert pass_sidecar["decode_provenance"]["padding_used"] is False
    score_resolution = report["temporal_refinement_score_resolution"]  # type: ignore[index]
    assert score_resolution["diagnostics"]["measured_result_count"] == 0
    refined = report["refined_segments"]  # type: ignore[index]
    assert len(refined) == 1
    assert refined[0]["boundary_status"] == "MODEL_REFINEMENT_PENDING"
    assert refined[0]["start_seconds"] is None
    assert refined[0]["end_seconds"] is None
    assert refined[0]["coarse_interval"]["status"] == "MODEL_PROBE_BOUND"
    json.dumps(report)


def test_adaptive_mode_with_no_coarse_boundary_emits_valid_empty_score_sidecar(monkeypatch) -> None:
    import robata.benchmark.production_wemm_open_runner as route

    class Group:
        def __init__(self, camera_id: str, window_id: str) -> None:
            self.camera_id = camera_id
            self.window_id = window_id
            self.frames = (f"{camera_id}-{window_id}-frame",)

        def metadata(self):
            return {
                "total_num_frames": 1,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0],
                "duration": 1.0,
            }

        def to_dict(self):
            return {"camera_id": self.camera_id, "window_id": self.window_id, "frame_count": 1}

    monkeypatch.setattr(
        route,
        "decode_production_windows",
        lambda manifest, **kwargs: {
            camera_id: {
                str(window["window_id"]): Group(camera_id, str(window["window_id"]))
                for window in manifest["windows"]
            }
            for camera_id in ("cam_01", "cam_02")
        },
    )

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del texts, batch_size
            return ((1.0, 0.0), (0.0, 1.0))

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del groups, metadata_groups
            return ((-1.0, 0.0),)

        def observation_payload(self):
            return []

        def close(self):
            return None

    report = route.run_production_wemm_open(
        _manifest(),
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        backend=FakeBackend(),
        temporal_mode="adaptive_score",
        temporal_boundary_mode="midpoint",
        temporal_refinement_span_seconds=0.5,
    )
    resolution = report["temporal_resolution"]
    assert resolution["segments"] == []
    score_result = report["temporal_refinement_score_resolution"]
    assert score_result["format"] == "robata-production-wemm-temporal-score-result-v1"
    assert score_result["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert score_result["status"] == "FINE_SCORE_BOUNDARIES_REVIEW_ONLY"
    assert score_result["results"] == []
    assert report["refined_segments"] == []
    json.dumps(report)


def test_adaptive_mode_projects_fine_score_crossings(monkeypatch) -> None:
    """A before/after score crossing becomes a review-only interval."""

    import robata.benchmark.production_wemm_open_runner as route

    manifest = {
        **_manifest(),
        "windows": [
            {
                "window_id": f"w{index}",
                "ordinal": index,
                "start_seconds": index * 0.5,
                "end_seconds": index * 0.5 + 1.0,
            }
            # Keep a trailing low-state context after the high interval so the
            # offset refinement has a real post-state probe.  An offset at the
            # recording edge is intentionally unresolved by the contract.
            for index in range(6)
        ],
    }

    class Group:
        def __init__(self, camera_id: str, window_id: str) -> None:
            self.frames = (f"{camera_id}-{window_id}-frame",)
            self.camera_id = camera_id
            self.window_id = window_id

        def metadata(self):
            return {
                "total_num_frames": 1,
                "fps": 1.0,
                "width": 10,
                "height": 10,
                "frames_indices": [0],
                "duration": 1.0,
            }

        def to_dict(self):
            return {
                "camera_id": self.camera_id,
                "window_id": self.window_id,
                "frame_count": 1,
            }

    def fake_decode(source_manifest, **kwargs):
        del kwargs
        window_ids = [str(row["window_id"]) for row in source_manifest["windows"]]
        return {
            camera_id: {window_id: Group(camera_id, window_id) for window_id in window_ids}
            for camera_id in ("cam_01", "cam_02")
        }

    monkeypatch.setattr(route, "decode_production_windows", fake_decode)

    class FakeBackend:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []

        def encode_texts(self, texts, *, batch_size):
            del batch_size
            # Keep the target action's absolute score below both temporal
            # thresholds in the low state; it remains in top-k for the fine
            # absolute-score pass even when a distractor ranks first.
            return tuple((1.0, 0.0) if "cupboard" in text else (-1.0, 0.0) for text in texts)

        def encode_video_frames(self, groups, *, metadata_groups=None):
            del metadata_groups
            frame = str(groups[0][0])
            if "temporal-refinement::" in frame:
                request_id = frame.split("temporal-refinement::", 1)[1]
                if "-onset-" in request_id:
                    high = "::after::" in request_id
                elif "-offset-" in request_id:
                    high = "::before::" in request_id
                else:  # pragma: no cover - malformed synthetic ID
                    high = False
            else:
                window_id = frame.split("-", 2)[1]
                high = window_id in {"w1", "w2", "w3"}
            # A low vector has target cosine ~-0.2 (score ~0.4), while high
            # is exact target alignment (score 1.0).  Both cameras agree.
            return ((1.0, 0.0) if high else (-0.2, 1.0),)

        def encode_video_frames_batch(self, groups, *, metadata_groups=None, batch_size=2):
            del batch_size
            return tuple(
                self.encode_video_frames([group], metadata_groups=[metadata_groups[index]])[0]
                for index, group in enumerate(groups)
            )

        def observation_payload(self):
            return []

        def close(self):
            return None

    report = route.run_production_wemm_open(
        manifest,
        phrase_catalog=_catalog(),
        model_directory="model",
        frame_count=2,
        top_k=2,
        dimension=2,
        device="cpu",
        backend=FakeBackend(),
        window_chunk_size=64,
        temporal_mode="adaptive_score",
        temporal_boundary_mode="midpoint",
        temporal_refinement_span_seconds=0.5,
        inference_batch_size=2,
        include_pipeline=True,
        queue_capacity=1,
    )

    score_resolution = report["temporal_refinement_score_resolution"]  # type: ignore[index]
    assert score_resolution["diagnostics"]["measured_result_count"] == 2
    assert report["model"]["temporal_refinement_boundaries_from_wemm"] is True  # type: ignore[index]
    assert report["model"]["producer_consumer"] is True  # type: ignore[index]
    pass_sidecar = report["raw_model_output"]["temporal_refinement"]["pass"]
    assert "pipeline_timing" in pass_sidecar
    assert pass_sidecar["pipeline_timing"]["queue_capacity"] == 1
    refined = report["refined_segments"]  # type: ignore[index]
    assert len(refined) == 1
    assert refined[0]["refinement_status"] == "REFINED"
    assert refined[0]["boundary_status"] == "MODEL_REFINED"
    assert refined[0]["start_seconds"] is not None
    assert refined[0]["end_seconds"] is not None
    assert report["temporal_resolution"]["segments"] != refined  # type: ignore[index]
    json.dumps(report)
