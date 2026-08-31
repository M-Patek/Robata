from __future__ import annotations

from robata.benchmark.atomic_event_error_taxonomy import (
    ATOMIC_EVENT_ERROR_TAXONOMY_VERSION,
    AtomicEventAllowedNextFactor,
    AtomicEventErrorCode,
    AtomicEventErrorLayer,
    AtomicEventEvidenceStatus,
    classify_atomic_event_error,
    classify_atomic_event_error_records,
    summarize_atomic_event_error_classifications,
)
from robata.benchmark.atomic_event_quality import (
    AtomicEventReference,
    score_atomic_event_text,
)


def _healthy_input_review(**overrides: object) -> dict[str, object]:
    review: dict[str, object] = {
        "decode_ok": True,
        "frame_time_match": True,
        "metadata_grid_match": True,
        "processor_probe_ok": True,
        "pre_state_visible": True,
        "target_hand_control_visible": True,
        "post_state_visible": True,
        "contact_state_change_visible": True,
        "direction_observable": True,
        "selected_pair_hits_transition": True,
        "object_legible_after_rendering": True,
        "object_control_too_small_after_thumbnail": False,
        "roi_excludes_target": False,
        "roi_or_scale_discontinuity": False,
        "ordering_unambiguous": True,
        "block_association_unambiguous": True,
        "state_inversion_observed": False,
        "distractor_present": False,
        "attention_competition_observed": False,
    }
    review.update(overrides)
    return review


def _quality(verb: str, noun: str, text: str) -> dict[str, object]:
    return score_atomic_event_text(
        text,
        reference=AtomicEventReference(verb=verb, noun=noun),
    ).to_dict()


def test_taxonomy_has_all_seven_layers_and_layer_specific_next_factors() -> None:
    assert tuple(layer.value for layer in AtomicEventErrorLayer) == (
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
    )

    classification = classify_atomic_event_error(
        input_review=_healthy_input_review(decode_ok=False)
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L0_MEDIA_PROCESSOR
    assert classification.allowed_next_factor is (
        AtomicEventAllowedNextFactor.REPAIR_MEDIA_RUNTIME_ONLY
    )
    assert classification.evidence_status is AtomicEventEvidenceStatus.CONFIRMED
    assert AtomicEventErrorCode.L0_DECODE_FAILURE in classification.codes


def test_media_precedes_temporal_and_spatial_failures_but_preserves_secondaries() -> None:
    classification = classify_atomic_event_error(
        input_review=_healthy_input_review(
            decode_ok=False,
            pre_state_visible=False,
            object_legible_after_rendering=False,
        )
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L0_MEDIA_PROCESSOR
    assert classification.secondary_layers == (
        AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE,
        AtomicEventErrorLayer.L2_SPATIAL_FIDELITY,
    )
    assert AtomicEventErrorCode.L1_PRE_STATE_NOT_VISIBLE in classification.codes
    assert AtomicEventErrorCode.L2_OBJECT_OR_CONTROL_NOT_LEGIBLE in classification.codes


def test_l0_failure_cannot_claim_sufficient_or_visually_correct_input() -> None:
    classification = classify_atomic_event_error(
        quality=_quality("turn off", "tap", "The person turns the faucet off."),
        input_review=_healthy_input_review(decode_ok=False),
        output_review={
            "target_action_asserted": True,
            "target_object_asserted": True,
            "correct_direction_asserted": True,
        },
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L0_MEDIA_PROCESSOR
    assert classification.input_evidence_sufficient is False
    assert classification.visual_joint_correct is False


def test_missing_atomic_transition_fact_prevents_visual_success() -> None:
    classification = classify_atomic_event_error(
        quality=_quality("turn off", "tap", "The person turns the faucet off."),
        input_review=_healthy_input_review(
            contact_state_change_visible=False,
            selected_pair_hits_transition=False,
        ),
        output_review={
            "target_action_asserted": True,
            "target_object_asserted": True,
            "correct_direction_asserted": True,
        },
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE
    assert classification.input_evidence_sufficient is False
    assert classification.visual_joint_correct is False


def test_temporal_and_spatial_evidence_are_distinguished() -> None:
    temporal = classify_atomic_event_error(
        input_review=_healthy_input_review(selected_pair_hits_transition=False)
    )
    spatial = classify_atomic_event_error(
        input_review=_healthy_input_review(roi_excludes_target=True)
    )

    assert temporal.primary_layer is AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE
    assert temporal.allowed_next_factor is (
        AtomicEventAllowedNextFactor.FOCUS_TEMPORAL_ANCHOR_LAYOUT_ONLY
    )
    assert AtomicEventErrorCode.L1_SELECTED_PAIR_MISSES_TRANSITION in temporal.codes
    assert spatial.primary_layer is AtomicEventErrorLayer.L2_SPATIAL_FIDELITY
    assert spatial.allowed_next_factor is (
        AtomicEventAllowedNextFactor.RENDER_BUDGET_OR_FIXED_ROI_ONLY
    )
    assert AtomicEventErrorCode.L2_ROI_EXCLUDES_TARGET in spatial.codes


def test_visible_opposite_direction_is_temporal_binding_not_a_media_failure() -> None:
    classification = classify_atomic_event_error(
        quality=_quality("turn off", "tap", "The person turns the faucet on."),
        input_review=_healthy_input_review(),
        output_review={"opposite_direction_asserted": True},
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT
    assert classification.allowed_next_factor is (
        AtomicEventAllowedNextFactor.STREAM_LAYOUT_OR_NEUTRAL_CAPTIONS_ONLY
    )
    assert AtomicEventErrorCode.L3_STATE_INVERSION in classification.codes
    assert classification.evidence_status is AtomicEventEvidenceStatus.CONFIRMED


def test_distractor_becomes_attention_competition_only_when_it_wins_the_predicate() -> None:
    classification = classify_atomic_event_error(
        quality=_quality(
            "turn off",
            "tap",
            "The person washes a pot while water runs from the faucet.",
        ),
        input_review=_healthy_input_review(distractor_present=True),
        output_review={"target_is_main_predicate": False},
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L4_ATTENTION_COMPETITION
    assert classification.allowed_next_factor is (
        AtomicEventAllowedNextFactor.CONTEXT_ALLOCATION_OR_LAYOUT_ONLY
    )
    assert classification.evidence_status is AtomicEventEvidenceStatus.CONFIRMED


def test_correct_target_plus_adjacent_narrative_is_l5_not_a_visual_failure() -> None:
    classification = classify_atomic_event_error(
        quality=_quality(
            "turn off",
            "tap",
            "The person turns the faucet off and then washes a pot for cleaning.",
        ),
        input_review=_healthy_input_review(distractor_present=True),
        output_review={
            "target_action_asserted": True,
            "target_object_asserted": True,
            "correct_direction_asserted": True,
            "target_is_main_predicate": True,
            "adjacent_purpose_or_action_present": True,
            "reviewer_rationale": "The target is stated first; washing is adjacent narrative.",
        },
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY
    assert classification.secondary_layers == ()
    assert classification.allowed_next_factor is (
        AtomicEventAllowedNextFactor.FREE_PROSE_INSTRUCTION_ONLY
    )
    assert classification.input_evidence_sufficient is True
    assert classification.target_joint_correct is True
    assert classification.visual_joint_correct is True
    assert classification.mapper_input_sufficient is False
    assert AtomicEventErrorCode.L5_MULTIPLE_ACTION_STORY in classification.codes
    assert "separate from visual failure" in classification.rationale


def test_reviewer_vs_lexical_scorer_disagreement_is_l6_not_l5() -> None:
    classification = classify_atomic_event_error(
        quality={
            "atomic_action_match": False,
            "object_match": False,
            "direction_required": True,
            "direction_match": False,
            "opposite_direction_asserted": False,
            "multiple_action_story": False,
            "hedged": False,
            "mapper_input_sufficient": False,
        },
        input_review=_healthy_input_review(),
        output_review={
            "target_action_asserted": True,
            "target_object_asserted": True,
            "correct_direction_asserted": True,
            "target_is_main_predicate": True,
            "lexical_score_disagreement": True,
        },
    )

    assert classification.primary_layer is AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE
    assert classification.secondary_layers == ()
    assert classification.allowed_next_factor is (
        AtomicEventAllowedNextFactor.EVALUATOR_OR_REPORT_ONLY
    )
    assert AtomicEventErrorCode.L6_LEXICAL_SCORE_DISAGREEMENT in classification.codes


def test_raw_output_miss_without_input_review_is_unknown_not_a_visual_claim() -> None:
    classification = classify_atomic_event_error(
        quality=_quality("turn off", "tap", "The person washes a pot by the faucet."),
    )

    assert classification.primary_layer is None
    assert classification.secondary_layers == ()
    assert classification.evidence_status is AtomicEventEvidenceStatus.UNCLEAR
    assert classification.target_joint_correct is False
    assert classification.visual_joint_correct is None
    assert classification.codes == (AtomicEventErrorCode.OUTPUT_TARGET_NOT_CONFIRMED,)


def test_record_projection_scores_raw_rows_and_summarizes_without_model_execution() -> None:
    rows = [
        {
            "uid": "qwen-row",
            "raw_output_text": "The hand closes the cupboard door.",
            "official_reference": {"verb": "close", "noun": "cupboard"},
            "input_review": _healthy_input_review(),
        },
        {
            "uid": "mage-row",
            "raw_output_text": "The person turns the faucet off and washes a pot.",
            "official_reference": {"verb": "turn off", "noun": "tap"},
            "input_review": _healthy_input_review(),
            "output_review": {
                "target_action_asserted": True,
                "target_object_asserted": True,
                "correct_direction_asserted": True,
                "target_is_main_predicate": True,
                "adjacent_purpose_or_action_present": True,
            },
        },
    ]

    classifications = classify_atomic_event_error_records(rows)
    summary = summarize_atomic_event_error_classifications(classifications)

    assert classifications[0].primary_layer is None
    assert classifications[0].visual_joint_correct is True
    assert classifications[1].primary_layer is AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY
    assert summary == {
        "taxonomy_version": ATOMIC_EVENT_ERROR_TAXONOMY_VERSION,
        "case_count": 2,
        "primary_layer_counts": {"L5": 1, "none": 1},
        "evidence_status_counts": {"confirmed": 2},
    }
