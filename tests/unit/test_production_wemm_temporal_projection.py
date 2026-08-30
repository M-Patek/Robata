from __future__ import annotations

import copy

import pytest

from robata.benchmark.production_wemm_temporal_projection import (
    BOUNDARY_STATUS,
    FORMAT,
    ProductionWemmTemporalProjectionError,
    attach_temporal_interval_proposals,
    project_temporal_interval_proposals,
    validate_temporal_interval_projection,
)


def _segment(
    segment_id: str = "seg-01",
    *,
    recording_id: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "segment_id": segment_id,
        "provisional_id": "open faucet",
        "label_text": "open faucet",
        "structured_labels": {
            "verb": {"value": "open", "status": "MEASURED"},
            "noun": {"value": "faucet", "status": "MEASURED"},
            "attributes": {"value": None, "status": "NOT_MEASURED"},
            "location": {"value": "sink", "status": "MEASURED"},
            "hand": {"value": "right", "status": "MEASURED"},
        },
        "start_seconds": 1.25,
        "end_seconds": 2.75,
        "boundary_status": BOUNDARY_STATUS,
        "boundary_source": "wemm_temporal_score",
        "boundary_method": "probe_center_midpoint",
        "boundary_confidence": 0.8,
        "confidence": 0.84,
        "peak_score": 0.91,
        "camera_support": ["cam_01", "cam_02"],
        "camera_support_count": 2,
        "supporting_window_ids": ["w01", "w02"],
        "top_k": [
            {
                "window_id": "w01",
                "candidates": [
                    {"rank": 1, "label_text": "open faucet", "score": 0.84},
                    {"rank": 2, "label_text": "wash dishes", "score": 0.79},
                ],
            }
        ],
        "evidence": [{"camera_id": "cam_01", "text": "handle turns"}],
        "review_required": True,
        "automatic_eligible": False,
    }
    if recording_id is not None:
        result["recording_id"] = recording_id
    return result


def _source(*, aggregate: bool = False) -> dict[str, object]:
    segment = _segment(recording_id="recording-01")
    if aggregate:
        # The aggregate wrapper intentionally uses ``review_only`` rather than
        # the producer's ``status=PROPOSALS_ONLY`` field.
        return {
            "format": "robata-production-wemm-review-pack-aggregate-v1",
            "source": {"kind": "multi_recording", "recording_count": 1},
            "temporal_resolution": {
                "review_only": True,
                "segments": [copy.deepcopy(segment)],
            },
            "temporal_segments": [copy.deepcopy(segment)],
        }
    return {
        "format": "robata-production-wemm-preannotation-v1",
        "source": {"recording_id": "recording-01", "path": "sample.mcap"},
        "temporal_resolution": {
            "status": "PROPOSALS_ONLY",
            "production_eligible": False,
            "segments": [segment],
        },
    }


def test_projects_model_interval_without_attaching_it_to_a_window() -> None:
    source = _source()
    before = copy.deepcopy(source)

    report = project_temporal_interval_proposals(source)

    assert source == before
    assert report["format"] == FORMAT
    assert report["production_eligible"] is False
    assert report["status"] == "PENDING_REVIEW"
    proposal = report["temporal_interval_proposals"][0]  # type: ignore[index]
    assert proposal["segment_id"] == "recording-01::temporal::seg-01"
    assert proposal["source_temporal_segment_id"] == "seg-01"
    assert proposal["start_seconds"] == pytest.approx(1.25)
    assert proposal["end_seconds"] == pytest.approx(2.75)
    assert proposal["boundary_status"] == BOUNDARY_STATUS
    assert proposal["review_required"] is True
    assert proposal["automatic_eligible"] is False
    assert proposal["verb"] == "open"
    assert proposal["noun"] == "faucet"
    assert proposal["top_k"][0]["candidates"][1]["label_text"] == "wash dishes"  # type: ignore[index]
    assert proposal["window_context"] == {  # type: ignore[index]
        "is_action_boundary": False,
        "supporting_window_ids": ["w01", "w02"],
    }
    assert report["metrics"]["automatic_eligible_count"] == 0  # type: ignore[index]


def test_aggregate_temporal_alias_is_supported_and_must_match() -> None:
    report = project_temporal_interval_proposals(_source(aggregate=True))
    assert report["sidecar_format"] is None
    assert len(report["temporal_interval_proposals"]) == 1  # type: ignore[arg-type]

    broken = _source(aggregate=True)
    broken["temporal_segments"] = [_segment("different", recording_id="recording-01")]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="does not match"):
        project_temporal_interval_proposals(broken)


def test_attach_is_additive_and_does_not_mutate_existing_draft() -> None:
    draft: dict[str, object] = {
        "format": "robata-production-wemm-annotation-draft-v1",
        "windows": [{"window_id": "w01", "annotation_draft": {"segments": []}}],
        "metrics": {"window_count": 1, "segment_count": 0},
        "review_contract": {"window_context_only": True},
        "controls": {"gold_written": False},
        "limitations": [],
    }
    before = copy.deepcopy(draft)

    attached = attach_temporal_interval_proposals(draft, _source())

    assert draft == before
    assert attached["windows"] == before["windows"]
    assert attached["temporal_interval_proposals"][0]["start_seconds"] == pytest.approx(1.25)  # type: ignore[index]
    assert attached["metrics"]["temporal_interval_proposal_count"] == 1  # type: ignore[index]
    assert attached["review_contract"]["temporal_interval_proposals_separate"] is True  # type: ignore[index]
    assert attached["controls"]["temporal_proposals_to_gold"] is False  # type: ignore[index]
    assert validate_temporal_interval_projection(attached) == attached


def test_invalid_sidecar_flags_and_bounds_are_rejected() -> None:
    broken = _source()
    broken["temporal_resolution"]["segments"][0]["review_required"] = False  # type: ignore[index]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="review_required"):
        project_temporal_interval_proposals(broken)

    broken = _source()
    broken["temporal_resolution"]["segments"][0]["end_seconds"] = 0.5  # type: ignore[index]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="0 <= start < end"):
        project_temporal_interval_proposals(broken)


def test_validation_rejects_attempted_automatic_temporal_promotion() -> None:
    report = project_temporal_interval_proposals(_source())
    report["temporal_interval_proposals"][0]["automatic_eligible"] = True  # type: ignore[index]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="automatic_eligible"):
        validate_temporal_interval_projection(report)


def test_no_sidecar_is_an_explicit_empty_projection() -> None:
    source = {
        "format": "robata-production-wemm-preannotation-v1",
        "source": {"recording_id": "recording-01"},
    }
    report = project_temporal_interval_proposals(source)
    assert report["status"] == "EMPTY"
    assert report["temporal_interval_proposals"] == []
    assert report["controls"]["temporal_sidecar_read"] is False  # type: ignore[index]
