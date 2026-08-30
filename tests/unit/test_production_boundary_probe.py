from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robata.benchmark.production_boundary_probe import (
    ProductionBoundaryProbeError,
    evaluate_production_boundary_probe,
    find_identity_context,
    frame_ordinal_prompt,
    index_identity_sidecar,
    parse_qwen_boundary_frame_output,
    parse_qwen_boundary_only_output,
)


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_production_qwen_boundary_only.py"
    spec = importlib.util.spec_from_file_location("run_production_qwen_boundary_only", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _identity_sidecar(*rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "production_eligible": False,
        "model": {"label_profile": "production_identity_only"},
        "controls": {"gold_included": False, "gold_read": False},
        "windows": list(rows),
    }


def _manifest() -> dict[str, Any]:
    return {
        "source": {
            "camera_count": 1,
            "cameras": [{"camera_id": "cam_01"}],
        },
        "windows": [
            {
                "ordinal": 0,
                "window_id": "w00",
                "start_seconds": 10.0,
                "end_seconds": 14.0,
                "camera_ids": ["cam_01"],
            }
        ],
    }


def _args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "manifest": Path("manifest.json"),
        "video_root": Path("video"),
        "model_dir": Path("model"),
        "offload_dir": Path("offload"),
        "identity_sidecar": None,
        "output": Path("output.json"),
        "limit": None,
        "camera_id": None,
        "frame_count": 8,
        "max_image_side": 320,
        "max_new_tokens": 160,
        "gpu_weight_memory_gib": 5,
        "cpu_weight_memory_gib": 16,
        "jpeg_quality": 92,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_boundary_parser_accepts_measured_relative_interval() -> None:
    result = parse_qwen_boundary_only_output(
        json.dumps(
            {
                "timestamp_basis": "window_relative_seconds",
                "start_time_sec": 0.4,
                "end_time_sec": 2.1,
                "confidence": 0.8,
                "evidence": "control handle visibly moves",
                "status": "MEASURED",
            }
        ),
        window_duration_seconds=4.0,
    )
    assert result["parse_status"] == "PARSED"
    assert result["boundary_status"] == "MEASURED"
    assert result["start_time_sec"] == pytest.approx(0.4)
    assert result["timestamp_basis"] == "window_relative_seconds"


def test_frame_boundary_parser_accepts_ordered_ordinals() -> None:
    result = parse_qwen_boundary_frame_output(
        json.dumps(
            {
                "coordinate_mode": "sampled_frame_ordinal",
                "start_frame_ordinal": 1,
                "end_frame_ordinal": 5,
                "confidence": 0.8,
                "evidence": "hands lift the garment",
                "status": "MEASURED",
            }
        ),
        frame_count=8,
    )
    assert result["parse_status"] == "PARSED"
    assert result["boundary_status"] == "MEASURED"
    assert result["start_frame_ordinal"] == 1
    assert result["end_frame_ordinal"] == 5


def test_frame_boundary_parser_rejects_equal_ordinals_without_repair() -> None:
    result = parse_qwen_boundary_frame_output(
        json.dumps(
            {
                "coordinate_mode": "sampled_frame_ordinal",
                "start_frame_ordinal": 2,
                "end_frame_ordinal": 2,
                "status": "MEASURED",
            }
        ),
        frame_count=8,
    )
    assert result["parse_status"] == "INVALID"
    assert "FRAME_ORDINAL_END_NOT_AFTER_START" in result["errors"]


def test_frame_boundary_parser_uses_non_default_count_for_range() -> None:
    result = parse_qwen_boundary_frame_output(
        json.dumps(
            {
                "coordinate_mode": "sampled_frame_ordinal",
                "start_frame_ordinal": 2,
                "end_frame_ordinal": 4,
                "status": "MEASURED",
            }
        ),
        frame_count=4,
    )
    assert result["parse_status"] == "INVALID"
    assert "FRAME_ORDINAL_OUT_OF_RANGE" in result["errors"]


def test_frame_ordinal_prompt_tracks_requested_frame_count() -> None:
    default_prompt = frame_ordinal_prompt(8)
    assert "exactly eight sampled frames" in default_prompt
    assert "numbered 0 through 7" in default_prompt
    prompt = frame_ordinal_prompt(4)
    assert "exactly 4 sampled frames" in prompt
    assert "numbered 0 through 3" in prompt
    assert "0 through 7" not in prompt


@pytest.mark.parametrize("frame_count", [0, 1, 2.0, True])
def test_frame_ordinal_prompt_rejects_invalid_count(frame_count: Any) -> None:
    with pytest.raises(ProductionBoundaryProbeError, match="frame_count"):
        frame_ordinal_prompt(frame_count)


def test_runner_prompt_uses_actual_frame_count() -> None:
    prompt, version, attached = runner._prompt_for_context(
        {"status": "UNAVAILABLE"}, prompt_variant="frame_ordinal", frame_count=4
    )
    assert "numbered 0 through 3" in prompt
    assert "0 through 7" not in prompt
    assert version.endswith("-blind")
    assert attached is False


def test_boundary_parser_accepts_unresolved_null_without_inventing_times() -> None:
    result = parse_qwen_boundary_only_output(
        json.dumps(
            {
                "timestamp_basis": "window_relative_seconds",
                "start_time_sec": None,
                "end_time_sec": None,
                "confidence": 0.2,
                "evidence": "the interaction is partly occluded",
                "status": "UNCERTAIN",
            }
        ),
        window_duration_seconds=4.0,
    )
    assert result["parse_status"] == "PARSED"
    assert result["boundary_status"] == "UNCERTAIN"
    assert result["start_time_sec"] is None
    assert result["end_time_sec"] is None


@pytest.mark.parametrize(
    ("start", "end", "error"),
    [(1.0, 1.0, "BOUNDARY_END_NOT_AFTER_START"), (1.0, 5.0, "BOUNDARY_OUT_OF_RANGE")],
)
def test_boundary_parser_rejects_equal_or_out_of_range_without_clipping(
    start: float,
    end: float,
    error: str,
) -> None:
    result = parse_qwen_boundary_only_output(
        json.dumps(
            {
                "timestamp_basis": "window_relative_seconds",
                "start_time_sec": start,
                "end_time_sec": end,
                "confidence": 0.5,
                "evidence": "visible",
                "status": "MEASURED",
            }
        ),
        window_duration_seconds=4.0,
    )
    assert result["parse_status"] == "INVALID"
    assert error in result["errors"]
    assert result["start_time_sec"] is None
    assert result["end_time_sec"] is None


def test_identity_context_is_model_observation_and_camera_bound() -> None:
    indexed = index_identity_sidecar(
        _identity_sidecar(
            {
                "window_id": "w00",
                "camera_id": "cam_01",
                "parsed_identity": {
                    "parse_status": "PARSED",
                    "action": "close faucet",
                    "confidence": 0.7,
                    "evidence": ["handle turns"],
                },
            }
        )
    )
    context = find_identity_context(indexed, window_id="w00", camera_id="cam_01")
    assert context["status"] == "AVAILABLE"
    assert context["action"] == "close faucet"
    assert context["source"] == "qwen_model_observation"
    assert (
        find_identity_context(indexed, window_id="w01", camera_id="cam_01")["status"]
        == "NOT_SUPPLIED"
    )


def test_identity_context_rejects_gold_controls() -> None:
    document = _identity_sidecar({"window_id": "w00"})
    document["controls"] = {"gold_read": True}
    with pytest.raises(ProductionBoundaryProbeError, match="gold"):
        index_identity_sidecar(document)


def test_runner_dry_run_keeps_complete_native_route_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "load_json", lambda *_args, **_kwargs: _manifest())
    identity = _identity_sidecar(
        {
            "window_id": "w00",
            "camera_id": "cam_01",
            "parsed_identity": {"parse_status": "PARSED", "action": "close faucet"},
        }
    )
    monkeypatch.setattr(
        runner,
        "index_identity_sidecar",
        lambda _path: index_identity_sidecar(identity),
    )

    class UnexpectedRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("dry-run must not load Qwen")

    monkeypatch.setattr(runner, "LocalHuggingFaceVisionRuntime", UnexpectedRuntime)
    report = runner.run(_args(identity_sidecar=Path("identity.json"), dry_run=True))
    assert report["status"] == "NOT_RUN"
    assert report["model"]["native_route"] == "complete_native_video"
    assert report["controls"]["model_invoked"] is False
    assert report["controls"]["source_media_decoded"] is False
    row = report["windows"][0]
    assert row["identity_context"]["action"] == "close faucet"
    assert row["parsed_boundary"]["errors"] == ["MODEL_NOT_RUN"]


def test_boundary_evaluator_remains_non_gold() -> None:
    report = evaluate_production_boundary_probe(
        {
            "format": "robata-production-qwen-boundary-only-shadow-v1",
            "windows": [
                {
                    "window_id": "w00",
                    "camera_id": "cam_01",
                    "native_video_complete": True,
                    "parsed_boundary": {
                        "parse_status": "PARSED",
                        "boundary_status": "MEASURED",
                        "evidence": "visible",
                        "confidence": 0.8,
                    },
                }
            ],
        }
    )
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["quality_claim"] is False
    assert report["metrics"]["boundary"]["measured"] == 1
