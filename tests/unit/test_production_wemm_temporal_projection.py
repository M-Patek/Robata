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
    assert report["metrics"]["measured_interval_count"] == 0  # type: ignore[index]


def test_projection_canonicalizes_conflicting_nested_recording_lineage() -> None:
    source = _source()
    source["temporal_resolution"]["segments"][0]["source_ref"] = {  # type: ignore[index]
        "recording_id": "stale-recording",
        "path": "sample.mcap",
        "upstream_lineage": {"recording_id": "older-recording"},
    }

    report = project_temporal_interval_proposals(source)
    proposal = report["temporal_interval_proposals"][0]  # type: ignore[index]

    assert proposal["recording_id"] == "recording-01"
    assert proposal["source_ref"]["recording_id"] == "recording-01"  # type: ignore[index]
    assert proposal["source_ref"]["upstream_lineage"]["recording_id"] == "stale-recording"  # type: ignore[index]


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
    assert attached["coarse_temporal_interval_proposals"][0]["start_seconds"] == pytest.approx(1.25)  # type: ignore[index]
    assert attached["temporal_interval_primary_selection"][0]["selection"] == "COARSE_FALLBACK"  # type: ignore[index]
    assert attached["metrics"]["temporal_interval_proposal_count"] == 1  # type: ignore[index]
    assert attached["metrics"]["measured_interval_count"] == 0  # type: ignore[index]
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


def test_projects_refined_and_pending_rows_in_separate_namespaces() -> None:
    source = _source()
    source["refined_segments"] = [
        {
            "segment_id": "seg-measured",
            "provisional_id": "open faucet",
            "coarse_interval": {"start_seconds": 1.0, "end_seconds": 3.0},
            "start_seconds": 1.4,
            "end_seconds": 2.6,
            "boundary_status": "MODEL_REFINED",
            "review_required": True,
            "automatic_eligible": False,
        },
        {
            "segment_id": "seg-pending",
            "provisional_id": "close faucet",
            "coarse_interval": {"start_seconds": 4.0, "end_seconds": 6.0},
            "start_seconds": None,
            "end_seconds": None,
            "boundary_status": "MODEL_REFINEMENT_PENDING",
            "review_required": True,
            "automatic_eligible": False,
        },
    ]
    report = project_temporal_interval_proposals(source)
    assert len(report["temporal_interval_proposals"]) == 1  # type: ignore[arg-type]
    primary = report["temporal_interval_proposals"][0]  # type: ignore[index]
    assert primary["boundary_status"] == "MODEL_REFINED"
    assert primary["start_seconds"] == pytest.approx(1.4)
    assert primary["end_seconds"] == pytest.approx(2.6)
    coarse = report["coarse_temporal_interval_proposals"][0]  # type: ignore[index]
    assert coarse["boundary_status"] == BOUNDARY_STATUS
    assert coarse["start_seconds"] == pytest.approx(1.25)
    assert coarse["end_seconds"] == pytest.approx(2.75)
    assert report["temporal_interval_primary_selection"][0]["selection"] == "MODEL_REFINED"  # type: ignore[index]
    assert report["metrics"]["temporal_interval_primary_refined_count"] == 1  # type: ignore[index]
    assert report["metrics"]["temporal_interval_coarse_fallback_count"] == 0  # type: ignore[index]
    assert report["metrics"]["measured_interval_count"] == 1  # type: ignore[index]
    assert len(report["temporal_refinement_segments"]) == 2  # type: ignore[arg-type]
    assert len(report["refined_temporal_interval_proposals"]) == 1  # type: ignore[arg-type]
    rows = report["temporal_refinement_segments"]  # type: ignore[index]
    assert rows[0]["segment_id"].startswith("recording-01::temporal-refined::")
    assert rows[1]["boundary_status"] == "MODEL_REFINEMENT_PENDING"
    assert rows[1]["start_seconds"] is None
    primary["start_seconds"] = 1.9
    assert rows[0]["start_seconds"] == pytest.approx(1.4)
    validate_temporal_interval_projection(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_seconds", "not-a-number"),
        ("end_seconds", []),
        ("start_seconds", True),
    ],
)
def test_pending_refined_rows_reject_nonnumeric_boundary_values(field: str, value: object) -> None:
    source = _source()
    pending = {
        "segment_id": "seg-pending-invalid",
        "provisional_id": "close faucet",
        "coarse_interval": {"start_seconds": 4.0, "end_seconds": 6.0},
        "start_seconds": None,
        "end_seconds": None,
        "boundary_status": "MODEL_REFINEMENT_PENDING",
        "review_required": True,
        "automatic_eligible": False,
    }
    pending[field] = value
    source["refined_segments"] = [pending]

    with pytest.raises(ProductionWemmTemporalProjectionError, match=field):
        project_temporal_interval_proposals(source)


def test_projection_validation_rejects_malformed_pending_time_values() -> None:
    report = project_temporal_interval_proposals(_source())
    report["temporal_refinement_segments"] = [
        {
            "segment_id": "recording-01::temporal-refined::bad",
            "boundary_status": "MODEL_REFINEMENT_PENDING",
            "start_seconds": "bad",
            "end_seconds": None,
            "coarse_interval": {"start_seconds": 1.0, "end_seconds": 2.0},
            "review_required": True,
            "automatic_eligible": False,
        }
    ]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="start_seconds"):
        validate_temporal_interval_projection(report)


def test_ambiguous_refined_row_keeps_coarse_primary_and_stays_sidecar_only() -> None:
    first = _segment("seg-a", recording_id="recording-01")
    second = _segment("seg-b", recording_id="recording-01")
    second["start_seconds"] = 3.25
    second["end_seconds"] = 4.75
    source = _source()
    source["temporal_resolution"]["segments"] = [first, second]  # type: ignore[index]
    source["refined_segments"] = [
        {
            "segment_id": "refined-ambiguous",
            "provisional_id": "open faucet",
            "coarse_interval": {"start_seconds": 9.0, "end_seconds": 10.0},
            "start_seconds": 9.2,
            "end_seconds": 9.8,
            "boundary_status": "MODEL_REFINED",
            "review_required": True,
            "automatic_eligible": False,
        }
    ]

    report = project_temporal_interval_proposals(source)

    primary = report["temporal_interval_proposals"]  # type: ignore[index]
    assert [row["boundary_status"] for row in primary] == [BOUNDARY_STATUS, BOUNDARY_STATUS]
    assert report["metrics"]["temporal_interval_primary_refined_count"] == 0  # type: ignore[index]
    assert report["metrics"]["temporal_interval_unmatched_refined_count"] == 1  # type: ignore[index]
    assert report["metrics"]["measured_interval_count"] == 0  # type: ignore[index]
    assert report["temporal_refinement_segments"][0]["boundary_status"] == "MODEL_REFINED"  # type: ignore[index]
    validate_temporal_interval_projection(report)


def test_conflicting_refined_action_does_not_replace_explicit_coarse_match() -> None:
    source = _source()
    source["refined_segments"] = [
        {
            "segment_id": "seg-01",
            "provisional_id": "close faucet",
            "coarse_interval": {"start_seconds": 1.0, "end_seconds": 3.0},
            "start_seconds": 1.4,
            "end_seconds": 2.6,
            "boundary_status": "MODEL_REFINED",
            "review_required": True,
            "automatic_eligible": False,
        }
    ]

    report = project_temporal_interval_proposals(source)

    assert report["temporal_interval_proposals"][0]["boundary_status"] == BOUNDARY_STATUS  # type: ignore[index]
    assert report["metrics"]["temporal_interval_primary_refined_count"] == 0  # type: ignore[index]
    assert report["metrics"]["temporal_interval_unmatched_refined_count"] == 1  # type: ignore[index]
    validate_temporal_interval_projection(report)


def test_aggregate_namespace_lineage_matches_refined_row_to_coarse_row() -> None:
    source = _source(aggregate=True)
    refined = {
        "segment_id": "seg-01",
        "provisional_id": "open faucet",
        "coarse_interval": {"start_seconds": 1.25, "end_seconds": 2.75},
        "start_seconds": 1.4,
        "end_seconds": 2.6,
        "boundary_status": "MODEL_REFINED",
        "review_required": True,
        "automatic_eligible": False,
    }
    source["refined_temporal_segments"] = [copy.deepcopy(refined)]
    source["temporal_refinement_segments"] = [copy.deepcopy(refined)]
    source["temporal_refinement"] = {
        "review_only": True,
        "segments": [copy.deepcopy(refined)],
    }

    report = project_temporal_interval_proposals(source)

    assert report["temporal_interval_proposals"][0]["boundary_status"] == "MODEL_REFINED"  # type: ignore[index]
    assert report["metrics"]["temporal_interval_primary_refined_count"] == 1  # type: ignore[index]
    validate_temporal_interval_projection(report)


def test_refined_projection_rejects_automatic_or_missing_coarse_interval() -> None:
    source = _source()
    source["refined_segments"] = [
        {
            "segment_id": "bad",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "boundary_status": "MODEL_REFINED",
            "review_required": False,
            "automatic_eligible": True,
        }
    ]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="review_required"):
        project_temporal_interval_proposals(source)

    source["refined_segments"][0]["review_required"] = True  # type: ignore[index]
    source["refined_segments"][0]["automatic_eligible"] = False  # type: ignore[index]
    with pytest.raises(ProductionWemmTemporalProjectionError, match="coarse_interval"):
        project_temporal_interval_proposals(source)


def test_validation_rejects_orphan_measured_refined_alias() -> None:
    report = project_temporal_interval_proposals(_source())
    orphan = copy.deepcopy(report["temporal_interval_proposals"][0])  # type: ignore[index]
    orphan["boundary_status"] = "MODEL_REFINED"
    orphan["coarse_interval"] = {"start_seconds": 1.0, "end_seconds": 3.0}
    report["refined_temporal_interval_proposals"] = [orphan]
    report["temporal_refinement_segments"] = []
    report["refined_temporal_segments"] = []
    with pytest.raises(ProductionWemmTemporalProjectionError, match="drawn from"):
        validate_temporal_interval_projection(report)
