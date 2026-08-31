from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_production_wemm_qwen_candidate_verifier_native.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_production_wemm_qwen_candidate_verifier_native", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _selector_window(*, proposals: int = 1) -> dict[str, object]:
    diagnostics: list[dict[str, object]] = []
    for index in range(proposals):
        diagnostics.append(
            {
                "proposal_id": f"proposal-{index + 1}",
                "top_k": [
                    {
                        "rank": 1,
                        "label_text": "fold garment",
                        "score": 0.8,
                        "provisional_id": "terra-fold-garment",
                        "structured_labels": {
                            "verb": {"value": "fold", "status": "MEASURED"},
                            "noun": {"value": "garment", "status": "MEASURED"},
                        },
                    },
                    {
                        "rank": 2,
                        "label_text": "flatten garment",
                        "score": 0.79,
                        "structured_labels": {
                            "verb": {"value": "flatten", "status": "MEASURED"},
                            "noun": {"value": "garment", "status": "MEASURED"},
                        },
                    },
                ],
            }
        )
    return {
        "window_id": "selector-w00",
        "recording_id": "recording-01",
        "ordinal": 0,
        "source_interval": {
            "start_seconds": 8.0,
            "end_seconds": 16.0,
            "status": "WINDOW_CONTEXT_ONLY",
            "is_action_boundary": False,
        },
        "declared_camera_ids": ["cam_01", "cam_02"],
        "proposal_diagnostics": diagnostics,
        "source_context_is_action_boundary": False,
    }


def test_selector_adapter_normalizes_mapping_interval_and_compact_labels() -> None:
    window = _selector_window()

    assert runner._interval(window) == (8.0, 16.0)
    top_k, adapter = runner._top_k_with_adapter(window)

    assert adapter == {
        "candidate_format": runner.AMBIGUITY_SELECTION_FORMAT,
        "adapter": "selector_proposal_diagnostics_v1",
        "proposal_index": 0,
        "proposal_id": "proposal-1",
        "proposal_count": 1,
    }
    assert top_k[0]["rank"] == 1
    assert top_k[0]["canonical_label"] == "fold garment"
    assert top_k[0]["verb"] == "fold"
    assert top_k[0]["noun"] == "garment"
    assert top_k[0]["label_id"] == "terra-fold-garment"
    # The legacy accessor remains available to old callers.
    assert runner._top_k(window)[1]["canonical_label"] == "flatten garment"


def test_selector_adapter_requires_explicit_proposal_for_multiple_rows() -> None:
    window = _selector_window(proposals=2)

    with pytest.raises(
        runner.ProductionWemmQwenCandidateVerifierError,
        match="multiple proposals",
    ):
        runner._top_k_with_adapter(window)

    top_k, adapter = runner._top_k_with_adapter(window, proposal_index=1)
    assert top_k[0]["rank"] == 1
    assert adapter["proposal_index"] == 1
    assert adapter["proposal_id"] == "proposal-2"


def test_selector_camera_fallback_is_available_without_legacy_manifest_row() -> None:
    window = _selector_window()

    source = runner._source_window(window, {})
    assert source is window
    assert runner._camera_ids(source) == ["cam_01", "cam_02"]


def test_legacy_candidate_pack_shape_remains_compatible() -> None:
    window = {
        "window_id": "legacy-w00",
        "source_interval": [0.0, 4.0],
        "model_context": {
            "wemm": {
                "top_k": [
                    {
                        "rank": 1,
                        "canonical_label": "pick up garment",
                        "verb": "pick up",
                        "noun": "garment",
                    }
                ]
            }
        },
    }

    assert runner._interval(window) == (0.0, 4.0)
    top_k, adapter = runner._top_k_with_adapter(window)
    assert top_k[0]["canonical_label"] == "pick up garment"
    assert adapter["adapter"] == "legacy_model_context_v1"


def test_production_request_disables_frame_digest_by_default() -> None:
    video = SimpleNamespace(
        frame_payloads=(b"a",),
        frame_indices=(0,),
        frame_timestamps_seconds=(0.0,),
        source_fps=1.0,
        total_num_frames=1,
        width=16,
        height=16,
        duration_seconds=1.0,
        interval_start_seconds=0.0,
        interval_end_seconds=1.0,
    )

    request = runner._request(video, "prompt", 8)
    assert request.compute_frame_sha256 is False

    # Legacy benchmark callers can retain the old observation explicitly,
    # without changing the production verifier's default.
    legacy_request = runner._request(
        video,
        "prompt",
        8,
        compute_frame_sha256=True,
    )
    assert legacy_request.compute_frame_sha256 is True


def test_run_records_disabled_frame_digest_policy_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_window = _selector_window()
    candidates = {
        "format": runner.AMBIGUITY_SELECTION_FORMAT,
        "windows": [candidate_window],
    }
    manifest = {"windows": []}
    monkeypatch.setattr(
        runner,
        "_load",
        lambda path: candidates if str(path).endswith("candidates.json") else manifest,
    )

    video = SimpleNamespace(
        frame_payloads=(b"a", b"b"),
        frame_indices=(0, 1),
        frame_timestamps_seconds=(0.0, 1.0),
        source_fps=1.0,
        total_num_frames=2,
        width=16,
        height=16,
        duration_seconds=2.0,
        interval_start_seconds=8.0,
        interval_end_seconds=16.0,
    )
    observation = SimpleNamespace(
        output_text=json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {
                        "rank": 1,
                        "support": "supported",
                        "evidence": ["hand moves"],
                        "boundary": {
                            "status": "measured",
                            "start_time_sec": 0.2,
                            "end_time_sec": 1.1,
                        },
                    }
                ],
                "decision": "accept",
                "selected_rank": 1,
                "segments": [
                    {
                        "candidate_rank": 1,
                        "boundary": {
                            "status": "measured",
                            "start_time_sec": 0.2,
                            "end_time_sec": 1.1,
                        },
                    }
                ],
            }
        ),
        frame_indices=(0, 1),
        frame_timestamps_seconds=(0.0, 1.0),
        rendered_frame_sizes=((16, 16), (16, 16)),
        prompt_tokens=5,
        output_tokens=10,
        generation_seconds=0.1,
        gpu_peak_allocated_bytes=1,
        visual_input=None,
        frame_sha256=(),
    )
    requests: list[Any] = []

    class _FakeRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def load(self) -> SimpleNamespace:
            return SimpleNamespace(load_seconds=0.1, gpu_name="fake")

        def generate_video(self, *, request: Any) -> SimpleNamespace:
            requests.append(request)
            return observation

        def close(self) -> None:
            pass

    monkeypatch.setattr(runner, "LocalHuggingFaceVisionRuntime", _FakeRuntime)
    monkeypatch.setattr(runner, "sample_qwen_native_video", lambda *_args, **_kwargs: video)
    args = SimpleNamespace(
        candidates=tmp_path / "candidates.json",
        manifest=tmp_path / "manifest.json",
        video_root=tmp_path / "video",
        model_dir=tmp_path / "model",
        offload_dir=tmp_path / "offload",
        limit=None,
        proposal_index=None,
        window_id=None,
        camera_id=None,
        frame_count=2,
        max_image_side=320,
        max_new_tokens=32,
        gpu_weight_memory_gib=5,
        cpu_weight_memory_gib=16,
        jpeg_quality=92,
        verdict_scope="selected_only",
        include_optional_fields=False,
    )

    report = runner.run(args)

    assert report["provenance"] == {
        "frame_digest_policy": runner.FRAME_DIGEST_POLICY,
        "frame_sha256_computed": False,
    }
    assert report["controls"]["hash_or_digest_computed"] is False
    assert report["controls"]["frame_sha256_computed"] is False
    assert requests and requests[0].compute_frame_sha256 is False
    assert report["windows"][0]["provenance"]["frame_digest_policy"] == (runner.FRAME_DIGEST_POLICY)

    # A non-conforming runtime must fail closed rather than publishing an
    # untracked digest under a report that says no digest was computed.
    observation.frame_sha256 = ("unexpected",)
    failed_report = runner.run(args)
    assert failed_report["controls"]["frame_sha256_computed"] is False
    assert {row["status"] for row in failed_report["windows"]} == {"FAILED"}
    assert all("returned frame_sha256" in row["error"] for row in failed_report["windows"])


def test_run_with_runtime_reuses_resident_model_and_does_not_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resident shards preserve the native/no-digest contract without reloads."""

    candidate_window = _selector_window()
    candidates = {
        "format": runner.AMBIGUITY_SELECTION_FORMAT,
        "windows": [candidate_window],
    }
    manifest = {"windows": []}
    monkeypatch.setattr(
        runner,
        "_load",
        lambda path: candidates if str(path).endswith("candidates.json") else manifest,
    )

    video = SimpleNamespace(
        frame_payloads=(b"a", b"b"),
        frame_indices=(0, 1),
        frame_timestamps_seconds=(0.0, 1.0),
        source_fps=1.0,
        total_num_frames=2,
        width=16,
        height=16,
        duration_seconds=2.0,
        interval_start_seconds=8.0,
        interval_end_seconds=16.0,
    )
    observation = SimpleNamespace(
        output_text=json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {
                        "rank": 1,
                        "support": "supported",
                        "evidence": ["hand moves"],
                    }
                ],
                "decision": "accept",
                "selected_rank": 1,
            }
        ),
        frame_indices=(0, 1),
        frame_timestamps_seconds=(0.0, 1.0),
        rendered_frame_sizes=((16, 16), (16, 16)),
        prompt_tokens=5,
        output_tokens=10,
        generation_seconds=0.1,
        gpu_peak_allocated_bytes=1,
        visual_input=None,
        frame_sha256=(),
    )

    class _ResidentRuntime:
        def __init__(self) -> None:
            self.loaded = False
            self.load_calls = 0
            self.generate_calls = 0
            self.close_calls = 0
            self.load_observation = None

        def load(self) -> SimpleNamespace:
            self.load_calls += 1
            self.loaded = True
            self.load_observation = SimpleNamespace(load_seconds=0.1, gpu_name="fake")
            return self.load_observation

        def generate_video(self, *, request: Any) -> SimpleNamespace:
            del request
            self.generate_calls += 1
            return observation

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(runner, "sample_qwen_native_video", lambda *_args, **_kwargs: video)
    runtime = _ResidentRuntime()
    args = SimpleNamespace(
        candidates=tmp_path / "candidates.json",
        manifest=tmp_path / "manifest.json",
        video_root=tmp_path / "video",
        model_dir=tmp_path / "model",
        offload_dir=tmp_path / "offload",
        limit=None,
        proposal_index=None,
        window_id=None,
        camera_id=["cam_01"],
        frame_count=2,
        max_image_side=320,
        max_new_tokens=32,
        gpu_weight_memory_gib=5,
        cpu_weight_memory_gib=16,
        jpeg_quality=92,
        verdict_scope="selected_only",
        include_optional_fields=False,
    )

    first = runner.run_with_runtime(args, runtime)
    second = runner.run_with_runtime(args, runtime)

    assert runtime.load_calls == 1
    assert runtime.generate_calls == 2
    assert runtime.close_calls == 0
    assert first["windows"][0]["status"] == "SUCCEEDED"
    assert second["windows"][0]["status"] == "SUCCEEDED"
    assert first["controls"]["complete_native_video_only"] is True
    assert first["controls"]["hash_or_digest_computed"] is False
    assert first["provenance"]["frame_sha256_computed"] is False


def test_run_with_runtime_accepts_preloaded_observation_without_load_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A batch can pass the one initial load observation to every shard."""

    candidate_window = _selector_window()
    candidates = {
        "format": runner.AMBIGUITY_SELECTION_FORMAT,
        "windows": [candidate_window],
    }
    monkeypatch.setattr(
        runner,
        "_load",
        lambda path: candidates if str(path).endswith("candidates.json") else {"windows": []},
    )
    video = SimpleNamespace(
        frame_payloads=(b"a", b"b"),
        frame_indices=(0, 1),
        frame_timestamps_seconds=(0.0, 1.0),
        source_fps=1.0,
        total_num_frames=2,
        width=16,
        height=16,
        duration_seconds=2.0,
        interval_start_seconds=8.0,
        interval_end_seconds=16.0,
    )
    observation = SimpleNamespace(
        output_text=json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {"rank": 1, "support": "supported", "evidence": ["hand moves"]}
                ],
                "decision": "accept",
                "selected_rank": 1,
            }
        ),
        frame_indices=(0, 1),
        frame_timestamps_seconds=(0.0, 1.0),
        rendered_frame_sizes=((16, 16), (16, 16)),
        prompt_tokens=5,
        output_tokens=10,
        generation_seconds=0.1,
        gpu_peak_allocated_bytes=1,
        visual_input=None,
        frame_sha256=(),
    )

    class _PreloadedRuntime:
        loaded = True
        load_observation = SimpleNamespace(load_seconds=1.25, gpu_name="fake")

        def __init__(self) -> None:
            self.load_calls = 0

        def load(self) -> SimpleNamespace:
            self.load_calls += 1
            return self.load_observation

        def generate_video(self, *, request: Any) -> SimpleNamespace:
            del request
            return observation

    monkeypatch.setattr(runner, "sample_qwen_native_video", lambda *_args, **_kwargs: video)
    runtime = _PreloadedRuntime()
    args = SimpleNamespace(
        candidates=tmp_path / "candidates.json",
        manifest=tmp_path / "manifest.json",
        video_root=tmp_path / "video",
        model_dir=tmp_path / "model",
        offload_dir=tmp_path / "offload",
        limit=None,
        proposal_index=None,
        window_id=None,
        camera_id=["cam_01"],
        frame_count=2,
        max_image_side=320,
        max_new_tokens=32,
        gpu_weight_memory_gib=5,
        cpu_weight_memory_gib=16,
        jpeg_quality=92,
        verdict_scope="selected_only",
        include_optional_fields=False,
    )

    report = runner.run_with_runtime(
        args,
        runtime,
        load_observation=runtime.load_observation,
    )

    assert runtime.load_calls == 0
    assert report["model"]["load_seconds"] == 1.25
    assert report["windows"][0]["raw_text"] == observation.output_text
