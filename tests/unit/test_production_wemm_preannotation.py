from __future__ import annotations

import json

import pytest

from robata.benchmark.production_wemm_preannotation import (
    FORMAT,
    REVIEW_FORMAT,
    ProductionWemmPreannotationError,
    build_preannotation_envelope,
    build_review_pack,
    validate_preannotation_envelope,
)


def _source() -> dict[str, object]:
    return {
        "source_id": "recording-01",
        "path": "data/source/sample-medium.mcap",
        "camera_count": 6,
        "camera_ids": [f"cam_{index:02d}" for index in range(1, 7)],
    }


def _proposal() -> dict[str, object]:
    return {
        "proposal_id": "p-01",
        "label_text": "open cupboard door",
        "structured_labels": {
            "verb": "open",
            "noun": "cupboard door",
            "attributes": None,
            "location": "counter",
            "hand": "right hand",
        },
        "start_seconds": 1.25,
        "end_seconds": 2.75,
        "confidence": 0.82,
        "camera_support": ["cam_01", "cam_02"],
        "evidence": [{"camera_id": "cam_01", "text": "door moves from closed to open"}],
        "top_k": [
            {
                "label_text": "open cupboard door",
                "verb": "open",
                "noun": "cupboard door",
                "score": 0.82,
                "camera_id": "cam_01",
            },
            {
                "label_text": "open drawer",
                "verb": "open",
                "noun": "drawer",
                "score": 0.71,
                "camera_id": "cam_01",
            },
        ],
    }


def _coarse_temporal_segment(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "segment_id": "temporal-segment",
        "provisional_id": "open-cupboard",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "boundary_status": "MODEL_PROBE_BOUND",
        "boundary_source": "wemm_temporal_score",
        "boundary_method": "probe_center_midpoint",
        "context_only": True,
        "window_context_only": True,
        "is_action_boundary": False,
        "action_boundary": False,
        "review_required": True,
        "automatic_eligible": False,
    }
    row.update(overrides)
    return row


def _refined_temporal_segment(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "segment_id": "refined-segment",
        "provisional_id": "open faucet",
        "coarse_interval": {"start_seconds": 1.0, "end_seconds": 3.0},
        "start_seconds": 1.4,
        "end_seconds": 2.6,
        "boundary_status": "MODEL_REFINED",
        "boundary_source": "wemm_short_refinement",
        "boundary_method": "short_probe_model",
        "context_only": True,
        "window_context_only": True,
        "is_action_boundary": False,
        "action_boundary": False,
        "review_required": True,
        "automatic_eligible": False,
    }
    row.update(overrides)
    return row


def test_builds_open_review_only_envelope_without_inventing_boundaries() -> None:
    envelope = build_preannotation_envelope(
        _source(),
        [
            {
                "window_id": "w00",
                "ordinal": 0,
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "camera_ids": ["cam_01", "cam_02"],
                "proposals": [_proposal()],
            },
            {
                "window_id": "w01",
                "ordinal": 1,
                "start_seconds": 4.0,
                "end_seconds": 8.0,
                "proposals": [],
            },
        ],
        raw_model_output={"camera_runs": [{"camera_id": "cam_01", "raw": "kept"}]},
    )

    assert envelope["format"] == FORMAT
    assert envelope["label_space"]["kind"] == "OPEN_PROVISIONAL_PHRASES"  # type: ignore[index]
    assert envelope["label_space"]["epic_ontology_used"] is False  # type: ignore[index]
    assert envelope["production_eligible"] is False
    assert envelope["controls"]["raw_candidates_overwritten"] is False  # type: ignore[index]
    assert envelope["controls"]["model_invoked"] is False  # type: ignore[index]
    first = envelope["windows"][0]["proposals"][0]  # type: ignore[index]
    assert first["proposal_interval"]["status"] == "MEASURED"  # type: ignore[index]
    assert first["margin"] == pytest.approx(0.11)
    assert first["review_required"] is True
    assert first["automatic_eligible"] is False
    second = envelope["windows"][1]  # type: ignore[index]
    assert second["window_status"] == "UNKNOWN"
    # The processing window is explicitly marked as context, not an action span.
    assert second["source_interval"]["status"] == "WINDOW_CONTEXT_ONLY"  # type: ignore[index]


def test_missing_proposal_interval_is_not_replaced_by_window() -> None:
    envelope = build_preannotation_envelope(
        _source(),
        [
            {
                "window_id": "w00",
                "start_seconds": 0,
                "end_seconds": 4,
                "proposals": [{"label_text": "wipe table"}],
            }
        ],
    )
    interval = envelope["windows"][0]["proposals"][0]["proposal_interval"]  # type: ignore[index]
    assert interval == {"start_seconds": None, "end_seconds": None, "status": "NOT_MEASURED"}


def test_open_text_is_preserved_and_review_pack_has_no_gold() -> None:
    envelope = build_preannotation_envelope(
        _source(),
        [{"window_id": "w00", "proposals": [{"label_text": "move blue tray"}]}],
        model={"name": "WeMM-Embedding-2B", "route": "video_embedding"},
        raw_model_output={
            "catalog": {
                "format": "robata-production-open-phrase-catalog-v1",
                "phrase_count": 6,
                "epic_ontology_used": False,
                "mapper_used": False,
                "provisional": True,
            }
        },
    )
    review = build_review_pack(envelope)
    assert review["format"] == REVIEW_FORMAT
    assert review["controls"]["gold_written"] is False  # type: ignore[index]
    assert review["model"]["name"] == "WeMM-Embedding-2B"  # type: ignore[index]
    assert review["model_artifact"]["catalog"]["phrase_count"] == 6  # type: ignore[index]
    assert review["model_artifact"]["catalog"]["epic_ontology_used"] is False  # type: ignore[index]
    assert review["items"][0]["proposals"][0]["label_text"] == "move blue tray"  # type: ignore[index]
    assert review["items"][0]["decision_options"] == [
        "accept",
        "edit",
        "split",
        "reject",
        "abstain",
    ]  # type: ignore[index]
    validate_preannotation_envelope(envelope)
    json.dumps(review)


def test_review_contract_and_pack_preserve_all_review_fields_and_window_states() -> None:
    envelope = build_preannotation_envelope(
        _source(),
        [
            {
                "window_id": "w-unknown",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "camera_ids": ["cam_01"],
                "proposals": [],
            },
            {
                "window_id": "w-abstain",
                "start_seconds": 4.0,
                "end_seconds": 8.0,
                "camera_ids": ["cam_01"],
                "decision": "abstain",
                "proposals": [],
            },
            {
                "window_id": "w-split",
                "start_seconds": 8.0,
                "end_seconds": 12.0,
                "camera_ids": ["cam_01"],
                "window_status": "SPLIT",
                "proposals": [
                    {
                        **_proposal(),
                        "proposal_status": "SPLIT",
                        "split_hint": True,
                    }
                ],
            },
        ],
    )

    required = envelope["review_contract"]["required_fields"]  # type: ignore[index]
    assert set(
        (
            "start_seconds",
            "end_seconds",
            "verb",
            "noun",
            "attributes",
            "location",
            "hand",
            "confidence",
            "evidence",
            "camera_support",
            "top_k",
            "margin",
        )
    ).issubset(required)
    assert envelope["review_contract"]["window_context_only"] is True  # type: ignore[index]

    review = build_review_pack(envelope)
    assert review["review_contract"]["required_fields"] == required  # type: ignore[index]
    items = {item["window_id"]: item for item in review["items"]}  # type: ignore[index]
    assert items["w-unknown"]["window_status"] == "UNKNOWN"
    assert items["w-unknown"]["window_decision"] == "pending"
    assert items["w-abstain"]["window_status"] == "ABSTAIN"
    assert items["w-abstain"]["window_decision"] == "abstain"
    assert items["w-split"]["window_status"] == "SPLIT"
    assert items["w-split"]["proposals"][0]["proposal_status"] == "SPLIT"
    assert items["w-split"]["proposals"][0]["split_hint"] is True
    assert "raw_candidates" in items["w-split"]
    assert items["w-split"]["source_interval"]["status"] == "WINDOW_CONTEXT_ONLY"


def test_review_pack_preserves_temporal_resolution_sidecar_without_relabeling_windows() -> None:
    envelope = build_preannotation_envelope(
        _source(),
        [{"window_id": "w00", "start_seconds": 0.0, "end_seconds": 4.0, "proposals": []}],
    )
    envelope["temporal_resolution"] = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [
            _coarse_temporal_segment(
                segment_id="open-cupboard@1.0-2.0",
                supporting_window_ids=["w00"],
            )
        ],
        "score_trajectories": [],
    }

    review = build_review_pack(envelope)
    item = review["items"][0]  # type: ignore[index]
    assert item["source_interval"]["status"] == "WINDOW_CONTEXT_ONLY"  # type: ignore[index]
    assert review["temporal_resolution"]["segments"][0]["boundary_status"] == "MODEL_PROBE_BOUND"  # type: ignore[index]
    assert review["temporal_segments"] == review["temporal_resolution"]["segments"]  # type: ignore[index]

    # Both representations are detached from the caller's mutable envelope.
    review["temporal_segments"][0]["provisional_id"] = "edited"  # type: ignore[index]
    assert envelope["temporal_resolution"]["segments"][0]["provisional_id"] == "open-cupboard"  # type: ignore[index]
    json.dumps(review)


def test_review_pack_rejects_malformed_temporal_resolution_sidecar() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    envelope["temporal_resolution"] = {
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "segments": {"not": "an array"},
    }
    with pytest.raises(
        ProductionWemmPreannotationError,
        match=r"format must be|segments must be an array",
    ):
        build_review_pack(envelope)


def test_alias_only_temporal_segments_are_validated_and_preserved() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    envelope["temporal_segments"] = [_coarse_temporal_segment(segment_id="alias-only")]

    validate_preannotation_envelope(envelope)
    review = build_review_pack(envelope)
    assert "temporal_resolution" not in review
    assert review["temporal_segments"][0]["segment_id"] == "alias-only"  # type: ignore[index]


def test_temporal_segment_alias_must_match_canonical_resolution() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    canonical = {
        **_coarse_temporal_segment(segment_id="canonical"),
    }
    envelope["temporal_resolution"] = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [canonical],
    }
    envelope["temporal_segments"] = [{**canonical, "segment_id": "different"}]
    with pytest.raises(ProductionWemmPreannotationError, match="does not match"):
        validate_preannotation_envelope(envelope)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "MEASURED", "PROPOSALS_ONLY"),
        ("production_eligible", True, "production_eligible"),
    ],
)
def test_validate_rejects_non_review_temporal_sidecar(
    field: str, value: object, message: str
) -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    envelope["temporal_resolution"] = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [],
        field: value,
    }
    with pytest.raises(ProductionWemmPreannotationError, match=message):
        validate_preannotation_envelope(envelope)


def test_validate_rejects_measured_temporal_segment_boundary() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    envelope["temporal_resolution"] = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [
            _coarse_temporal_segment(
                segment_id="measured",
                start_seconds=0.0,
                end_seconds=1.0,
                boundary_status="MEASURED",
            )
        ],
    }
    with pytest.raises(ProductionWemmPreannotationError, match="MODEL_PROBE_BOUND"):
        validate_preannotation_envelope(envelope)


@pytest.mark.parametrize(
    "bad_key, bad_value",
    [("action_key", [1, 2]), ("verb_id", 1), ("noun_key", [2]), ("epic_ontology", True)],
)
def test_epic_identity_is_rejected(bad_key: str, bad_value: object) -> None:
    candidate = {"label_text": "open drawer", bad_key: bad_value}
    with pytest.raises(ProductionWemmPreannotationError, match="EPIC ontology"):
        build_preannotation_envelope(
            _source(), [{"window_id": "w00", "proposals": [{"top_k": [candidate]}]}]
        )


def test_invalid_interval_and_decision_are_rejected() -> None:
    with pytest.raises(ProductionWemmPreannotationError, match="interval"):
        build_preannotation_envelope(
            _source(),
            [{"window_id": "w00", "proposals": [{"start_seconds": 2, "end_seconds": 1}]}],
        )
    with pytest.raises(ProductionWemmPreannotationError, match="decision"):
        build_preannotation_envelope(
            _source(),
            [{"window_id": "w00", "decision": "auto", "proposals": []}],
        )


def test_validate_rejects_stale_automatic_flag() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    envelope["windows"][0]["proposals"] = [  # type: ignore[index]
        {
            "proposal_id": "p",
            "decision": "pending",
            "review_required": False,
            "automatic_eligible": True,
        }
    ]
    with pytest.raises(ProductionWemmPreannotationError, match="require human review"):
        validate_preannotation_envelope(envelope)


def test_adaptive_refinement_sidecars_are_preserved_and_pending_is_review_only() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    refined = _refined_temporal_segment(segment_id="seg-refined")
    pending = _refined_temporal_segment(
        segment_id="seg-pending",
        provisional_id="close faucet",
        coarse_interval={"start_seconds": 4.0, "end_seconds": 6.0},
        start_seconds=None,
        end_seconds=None,
        boundary_status="MODEL_REFINEMENT_PENDING",
    )
    envelope.update(
        {
            "temporal_refinement_plan": {
                "format": "robata-production-wemm-temporal-refinement-plan-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "requests": [],
            },
            "temporal_refinement_fine_plan": {
                "format": "robata-production-wemm-temporal-score-refinement-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "requests": [],
            },
            "temporal_refinement_score_resolution": {
                "format": "robata-production-wemm-temporal-score-result-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "results": [],
            },
            "temporal_refinement": {
                "format": "robata-production-wemm-temporal-refinement-review-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "refined_segments": [refined, pending],
            },
            "refined_segments": [refined, pending],
        }
    )

    validate_preannotation_envelope(envelope)
    review = build_review_pack(envelope)
    assert review["refined_segments"][0]["boundary_status"] == "MODEL_REFINED"  # type: ignore[index]
    assert review["refined_segments"][1]["start_seconds"] is None  # type: ignore[index]
    assert review["refined_temporal_segments"] == review["refined_segments"]  # type: ignore[index]
    assert "temporal_refinement_fine_plan" in review
    assert review["review_contract"]["refined_segments_review_only"] is True  # type: ignore[index]
    # All sidecars are detached from the caller's mutable envelope.
    review["refined_segments"][0]["provisional_id"] = "edited"  # type: ignore[index]
    assert envelope["refined_segments"][0]["provisional_id"] == "open faucet"  # type: ignore[index]


def test_lowercase_refined_status_is_canonicalized_in_review_snapshot() -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    row = _refined_temporal_segment(segment_id="lowercase", boundary_status="model_refined")
    envelope["refined_segments"] = [row]
    validate_preannotation_envelope(envelope)
    review = build_review_pack(envelope)
    assert review["refined_segments"][0]["boundary_status"] == "MODEL_REFINED"  # type: ignore[index]


@pytest.mark.parametrize(
    "bad_row",
    [
        {
            "segment_id": "bad-status",
            "coarse_interval": {"start_seconds": 0.0, "end_seconds": 1.0},
            "boundary_status": "MODEL_PROBE_BOUND",
            "review_required": True,
            "automatic_eligible": False,
            "start_seconds": 0.1,
            "end_seconds": 0.2,
        },
        {
            "segment_id": "bad-pending",
            "coarse_interval": {"start_seconds": 0.0, "end_seconds": 1.0},
            "boundary_status": "MODEL_REFINEMENT_PENDING",
            "review_required": True,
            "automatic_eligible": False,
            "start_seconds": 0.1,
            "end_seconds": 0.2,
        },
    ],
)
def test_validate_rejects_malformed_adaptive_refined_row(bad_row: dict[str, object]) -> None:
    envelope = build_preannotation_envelope(_source(), [{"window_id": "w00", "proposals": []}])
    envelope["refined_segments"] = [bad_row]
    with pytest.raises(ProductionWemmPreannotationError):
        validate_preannotation_envelope(envelope)
