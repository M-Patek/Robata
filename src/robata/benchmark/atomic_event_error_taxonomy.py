"""Benchmark-local layered error attribution for atomic Qwen and Mage outputs.

This module deliberately sits beside, rather than inside, the model runners.  It
combines a post-hoc lexical quality projection with optional human input/output
review facts and attributes the *first evidenced* failure layer.  It neither
changes a model input nor projects an observation into a Mapper or production
event.

The lexical scorer in :mod:`robata.benchmark.atomic_event_quality` is useful
evidence, but it is not visual evidence.  In particular, an output that asserts
the expected action, object, and direction while adding an adjacent purpose or
action remains a L5 prose-granularity issue even when the scorer's stricter
``mapper_input_sufficient`` flag is false.  The classifier never turns that
disagreement into an L1--L4 visual failure without an input review.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from robata.benchmark.atomic_event_quality import (
    AtomicEventQualityScore,
    score_raw_output_record,
)

ATOMIC_EVENT_ERROR_TAXONOMY_VERSION = "atomic-event-error-taxonomy-v1"


class AtomicEventErrorLayer(StrEnum):
    """Ordered, benchmark-only layers used for error attribution."""

    L0_MEDIA_PROCESSOR = "L0"
    L1_TEMPORAL_EVIDENCE = "L1"
    L2_SPATIAL_FIDELITY = "L2"
    L3_TEMPORAL_BINDING_LAYOUT = "L3"
    L4_ATTENTION_COMPETITION = "L4"
    L5_OUTPUT_GRANULARITY = "L5"
    L6_EVALUATOR_REFERENCE = "L6"

    # Numeric aliases keep compact review-ledger values ergonomic without
    # discarding the descriptive names used in code and reports.
    L0 = L0_MEDIA_PROCESSOR
    L1 = L1_TEMPORAL_EVIDENCE
    L2 = L2_SPATIAL_FIDELITY
    L3 = L3_TEMPORAL_BINDING_LAYOUT
    L4 = L4_ATTENTION_COMPETITION
    L5 = L5_OUTPUT_GRANULARITY
    L6 = L6_EVALUATOR_REFERENCE


# Short aliases make the small module convenient in review scripts while keeping
# the public names explicit for callers that import it directly.
ErrorLayer = AtomicEventErrorLayer


class AtomicEventEvidenceStatus(StrEnum):
    """How strongly the selected layer is supported by retained evidence."""

    CONFIRMED = "confirmed"
    UNCLEAR = "unclear"
    NOT_REVIEWABLE = "not-reviewable"


EvidenceStatus = AtomicEventEvidenceStatus


class AtomicEventAllowedNextFactor(StrEnum):
    """The only experiment-factor family justified by each primary layer."""

    REPAIR_MEDIA_RUNTIME_ONLY = "repair-media-runtime-only"
    FOCUS_TEMPORAL_ANCHOR_LAYOUT_ONLY = "focus-temporal-anchor-layout-only"
    RENDER_BUDGET_OR_FIXED_ROI_ONLY = "render-budget-or-fixed-roi-only"
    STREAM_LAYOUT_OR_NEUTRAL_CAPTIONS_ONLY = "stream-layout-or-neutral-block-captions-only"
    CONTEXT_ALLOCATION_OR_LAYOUT_ONLY = "context-allocation-or-layout-only"
    FREE_PROSE_INSTRUCTION_ONLY = "free-prose-instruction-only"
    EVALUATOR_OR_REPORT_ONLY = "improve-posthoc-evaluator-or-report-only"


class AtomicEventErrorCode(StrEnum):
    """Inspectable reasons within the L0--L6 taxonomy."""

    L0_DECODE_FAILURE = "L0_DECODE_FAILURE"
    L0_FRAME_TIME_MISMATCH = "L0_FRAME_TIME_MISMATCH"
    L0_METADATA_GRID_MISMATCH = "L0_METADATA_GRID_MISMATCH"
    L0_PROCESSOR_PROBE_FAILURE = "L0_PROCESSOR_PROBE_FAILURE"

    L1_PRE_STATE_NOT_VISIBLE = "L1_PRE_STATE_NOT_VISIBLE"
    L1_TARGET_CONTROL_NOT_VISIBLE = "L1_TARGET_CONTROL_NOT_VISIBLE"
    L1_POST_STATE_NOT_VISIBLE = "L1_POST_STATE_NOT_VISIBLE"
    L1_CONTACT_OR_STATE_CHANGE_NOT_VISIBLE = "L1_CONTACT_OR_STATE_CHANGE_NOT_VISIBLE"
    L1_DIRECTION_NOT_OBSERVABLE = "L1_DIRECTION_NOT_OBSERVABLE"
    L1_SELECTED_PAIR_MISSES_TRANSITION = "L1_SELECTED_PAIR_MISSES_TRANSITION"
    L1_TEMPORAL_GAP_RELAXED_TOO_FAR = "L1_TEMPORAL_GAP_RELAXED_TOO_FAR"

    L2_OBJECT_OR_CONTROL_NOT_LEGIBLE = "L2_OBJECT_OR_CONTROL_NOT_LEGIBLE"
    L2_OBJECT_OR_CONTROL_TOO_SMALL = "L2_OBJECT_OR_CONTROL_TOO_SMALL"
    L2_ROI_EXCLUDES_TARGET = "L2_ROI_EXCLUDES_TARGET"
    L2_MIXED_SCALE_DISCONTINUITY = "L2_MIXED_SCALE_DISCONTINUITY"

    L3_ORDERING_AMBIGUOUS = "L3_ORDERING_AMBIGUOUS"
    L3_BLOCK_ASSOCIATION_AMBIGUOUS = "L3_BLOCK_ASSOCIATION_AMBIGUOUS"
    L3_STATE_INVERSION = "L3_STATE_INVERSION"

    L4_ATTENTION_COMPETITION = "L4_ATTENTION_COMPETITION"

    L5_ADJACENT_PURPOSE_OR_ACTION = "L5_ADJACENT_PURPOSE_OR_ACTION"
    L5_MULTIPLE_ACTION_STORY = "L5_MULTIPLE_ACTION_STORY"
    L5_HEDGED_OR_INSUFFICIENTLY_ATOMIC = "L5_HEDGED_OR_INSUFFICIENTLY_ATOMIC"
    L5_STRICT_PROSE_CONTRACT_ONLY = "L5_STRICT_PROSE_CONTRACT_ONLY"

    L6_LEXICAL_SCORE_DISAGREEMENT = "L6_LEXICAL_SCORE_DISAGREEMENT"
    L6_REFERENCE_AMBIGUITY = "L6_REFERENCE_AMBIGUITY"

    OUTPUT_TARGET_NOT_CONFIRMED = "OUTPUT_TARGET_NOT_CONFIRMED"


@dataclass(frozen=True, slots=True)
class AtomicEventInputReview:
    """Human/CPU review facts about the actual Qwen or Mage input.

    A value of ``None`` means the fact was not reviewed.  Values are intentionally
    facts rather than inferred probabilities; this keeps missing review evidence
    from being turned into a visual-failure claim.
    """

    decode_succeeded: bool | None = None
    frame_times_aligned: bool | None = None
    metadata_grid_aligned: bool | None = None
    processor_probe_succeeded: bool | None = None

    pre_state_visible: bool | None = None
    target_hand_or_control_visible: bool | None = None
    post_state_visible: bool | None = None
    contact_or_state_change_visible: bool | None = None
    direction_observable: bool | None = None
    selected_pair_hits_transition: bool | None = None
    temporal_gap_relaxed_too_far: bool | None = None

    object_or_control_legible_after_rendering: bool | None = None
    object_or_control_too_small_after_thumbnail: bool | None = None
    roi_excludes_target: bool | None = None
    roi_or_scale_discontinuity: bool | None = None

    ordering_unambiguous: bool | None = None
    block_association_unambiguous: bool | None = None
    state_inversion_observed: bool | None = None
    distractor_present: bool | None = None
    attention_competition_observed: bool | None = None

    reviewer_confidence: str | None = None
    reviewer_rationale: str = ""
    evidence_links: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AtomicEventInputReview:
        """Read the review-pack column names, including concise compatibility aliases."""

        return cls(
            decode_succeeded=_optional_bool(
                _first(value, "decode_succeeded", "decode_ok", "decode_success"),
                "decode_succeeded",
            ),
            frame_times_aligned=_optional_bool(
                _first(value, "frame_times_aligned", "frame_time_aligned", "frame_time_match"),
                "frame_times_aligned",
            ),
            metadata_grid_aligned=_optional_bool(
                _first(value, "metadata_grid_aligned", "metadata_grid_match", "grid_match"),
                "metadata_grid_aligned",
            ),
            processor_probe_succeeded=_optional_bool(
                _first(
                    value,
                    "processor_probe_succeeded",
                    "processor_probe_ok",
                    "processor_probe_success",
                ),
                "processor_probe_succeeded",
            ),
            pre_state_visible=_optional_bool(
                _first(value, "pre_state_visible"), "pre_state_visible"
            ),
            target_hand_or_control_visible=_optional_bool(
                _first(
                    value,
                    "target_hand_or_control_visible",
                    "target_hand_control_visible",
                    "target_control_visible",
                ),
                "target_hand_or_control_visible",
            ),
            post_state_visible=_optional_bool(
                _first(value, "post_state_visible"), "post_state_visible"
            ),
            contact_or_state_change_visible=_optional_bool(
                _first(
                    value,
                    "contact_or_state_change_visible",
                    "contact_state_change_visible",
                ),
                "contact_or_state_change_visible",
            ),
            direction_observable=_optional_bool(
                _first(value, "direction_observable"), "direction_observable"
            ),
            selected_pair_hits_transition=_optional_bool(
                _first(
                    value,
                    "selected_pair_hits_transition",
                    "selected_pair_captures_transition",
                ),
                "selected_pair_hits_transition",
            ),
            temporal_gap_relaxed_too_far=_optional_bool(
                _first(value, "temporal_gap_relaxed_too_far"),
                "temporal_gap_relaxed_too_far",
            ),
            object_or_control_legible_after_rendering=_optional_bool(
                _first(
                    value,
                    "object_or_control_legible_after_rendering",
                    "object_legible_after_rendering",
                    "object_legible_after_render",
                ),
                "object_or_control_legible_after_rendering",
            ),
            object_or_control_too_small_after_thumbnail=_optional_bool(
                _first(
                    value,
                    "object_or_control_too_small_after_thumbnail",
                    "object_control_too_small_after_thumbnail",
                    "target_too_small_after_thumbnail",
                ),
                "object_or_control_too_small_after_thumbnail",
            ),
            roi_excludes_target=_optional_bool(
                _first(value, "roi_excludes_target"), "roi_excludes_target"
            ),
            roi_or_scale_discontinuity=_optional_bool(
                _first(
                    value,
                    "roi_or_scale_discontinuity",
                    "mixed_scale_discontinuity",
                    "roi_scale_discontinuity",
                ),
                "roi_or_scale_discontinuity",
            ),
            ordering_unambiguous=_optional_bool(
                _first(value, "ordering_unambiguous", "temporal_order_unambiguous"),
                "ordering_unambiguous",
            ),
            block_association_unambiguous=_optional_bool(
                _first(value, "block_association_unambiguous"),
                "block_association_unambiguous",
            ),
            state_inversion_observed=_optional_bool(
                _first(value, "state_inversion_observed"), "state_inversion_observed"
            ),
            distractor_present=_optional_bool(
                _first(value, "distractor_present"), "distractor_present"
            ),
            attention_competition_observed=_optional_bool(
                _first(value, "attention_competition_observed"),
                "attention_competition_observed",
            ),
            reviewer_confidence=_optional_text(
                _first(value, "reviewer_confidence"), "reviewer_confidence"
            ),
            reviewer_rationale=_text_or_empty(
                _first(value, "reviewer_rationale", "rationale"), "reviewer_rationale"
            ),
            evidence_links=_text_tuple(_first(value, "evidence_links", "evidence_link")),
        )


@dataclass(frozen=True, slots=True)
class AtomicEventOutputReview:
    """Post-hoc reviewer facts about the retained free-text observation."""

    target_action_asserted: bool | None = None
    target_object_asserted: bool | None = None
    correct_direction_asserted: bool | None = None
    target_is_main_predicate: bool | None = None
    adjacent_purpose_or_action_present: bool | None = None
    opposite_direction_asserted: bool | None = None
    insufficiently_atomic_prose: bool | None = None
    lexical_score_disagreement: bool | None = None
    annotation_ambiguous: bool | None = None
    reviewer_rationale: str = ""
    evidence_links: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AtomicEventOutputReview:
        """Read required review-pack output columns and concise aliases."""

        return cls(
            target_action_asserted=_optional_bool(
                _first(value, "target_action_asserted", "action_asserted"),
                "target_action_asserted",
            ),
            target_object_asserted=_optional_bool(
                _first(value, "target_object_asserted", "object_asserted"),
                "target_object_asserted",
            ),
            correct_direction_asserted=_optional_bool(
                _first(value, "correct_direction_asserted", "direction_asserted"),
                "correct_direction_asserted",
            ),
            target_is_main_predicate=_optional_bool(
                _first(value, "target_is_main_predicate", "target_main_predicate"),
                "target_is_main_predicate",
            ),
            adjacent_purpose_or_action_present=_optional_bool(
                _first(
                    value,
                    "adjacent_purpose_or_action_present",
                    "adjacent_action_present",
                    "adjacent_purpose_present",
                ),
                "adjacent_purpose_or_action_present",
            ),
            opposite_direction_asserted=_optional_bool(
                _first(value, "opposite_direction_asserted"), "opposite_direction_asserted"
            ),
            insufficiently_atomic_prose=_optional_bool(
                _first(value, "insufficiently_atomic_prose"), "insufficiently_atomic_prose"
            ),
            lexical_score_disagreement=_optional_bool(
                _first(
                    value,
                    "lexical_score_disagreement",
                    "scorer_review_disagreement",
                ),
                "lexical_score_disagreement",
            ),
            annotation_ambiguous=_optional_bool(
                _first(value, "annotation_ambiguous", "reference_ambiguous"),
                "annotation_ambiguous",
            ),
            reviewer_rationale=_text_or_empty(
                _first(value, "reviewer_rationale", "rationale"), "reviewer_rationale"
            ),
            evidence_links=_text_tuple(_first(value, "evidence_links", "evidence_link")),
        )


@dataclass(frozen=True, slots=True)
class AtomicEventErrorClassification:
    """A sidecar classification, never a model input or event projection."""

    primary_layer: AtomicEventErrorLayer | None
    secondary_layers: tuple[AtomicEventErrorLayer, ...]
    evidence_status: AtomicEventEvidenceStatus
    codes: tuple[AtomicEventErrorCode, ...]
    allowed_next_factor: AtomicEventAllowedNextFactor | None
    rationale: str
    evidence_links: tuple[str, ...]
    input_evidence_sufficient: bool | None
    target_joint_correct: bool | None
    visual_joint_correct: bool | None
    mapper_input_sufficient: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready benchmark-sidecar fields."""

        return {
            "taxonomy_version": ATOMIC_EVENT_ERROR_TAXONOMY_VERSION,
            "primary_layer": self.primary_layer.value if self.primary_layer is not None else None,
            "secondary_layers": [layer.value for layer in self.secondary_layers],
            "evidence_status": self.evidence_status.value,
            "codes": [code.value for code in self.codes],
            "allowed_next_factor": (
                self.allowed_next_factor.value if self.allowed_next_factor is not None else None
            ),
            "rationale": self.rationale,
            "evidence_links": list(self.evidence_links),
            "input_evidence_sufficient": self.input_evidence_sufficient,
            "target_joint_correct": self.target_joint_correct,
            "visual_joint_correct": self.visual_joint_correct,
            "mapper_input_sufficient": self.mapper_input_sufficient,
        }


_NEXT_FACTOR_BY_LAYER: Mapping[AtomicEventErrorLayer, AtomicEventAllowedNextFactor] = {
    AtomicEventErrorLayer.L0_MEDIA_PROCESSOR: (
        AtomicEventAllowedNextFactor.REPAIR_MEDIA_RUNTIME_ONLY
    ),
    AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE: (
        AtomicEventAllowedNextFactor.FOCUS_TEMPORAL_ANCHOR_LAYOUT_ONLY
    ),
    AtomicEventErrorLayer.L2_SPATIAL_FIDELITY: (
        AtomicEventAllowedNextFactor.RENDER_BUDGET_OR_FIXED_ROI_ONLY
    ),
    AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT: (
        AtomicEventAllowedNextFactor.STREAM_LAYOUT_OR_NEUTRAL_CAPTIONS_ONLY
    ),
    AtomicEventErrorLayer.L4_ATTENTION_COMPETITION: (
        AtomicEventAllowedNextFactor.CONTEXT_ALLOCATION_OR_LAYOUT_ONLY
    ),
    AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY: (
        AtomicEventAllowedNextFactor.FREE_PROSE_INSTRUCTION_ONLY
    ),
    AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE: (
        AtomicEventAllowedNextFactor.EVALUATOR_OR_REPORT_ONLY
    ),
}

_LAYER_ORDER: tuple[AtomicEventErrorLayer, ...] = (
    AtomicEventErrorLayer.L0_MEDIA_PROCESSOR,
    AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE,
    AtomicEventErrorLayer.L2_SPATIAL_FIDELITY,
    AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT,
    AtomicEventErrorLayer.L4_ATTENTION_COMPETITION,
    AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY,
    AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE,
)


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be true, false, or null")


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    raise ValueError(f"{field} must be a string or null")


def _text_or_empty(value: Any, field: str) -> str:
    return _optional_text(value, field) or ""


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ValueError("evidence_links must be a string, sequence of strings, or null")


def _coerce_input_review(
    value: AtomicEventInputReview | Mapping[str, Any] | None,
) -> AtomicEventInputReview | None:
    if value is None:
        return None
    if isinstance(value, AtomicEventInputReview):
        return value
    if isinstance(value, Mapping):
        return AtomicEventInputReview.from_mapping(value)
    raise TypeError("input_review must be an AtomicEventInputReview, mapping, or None")


def _coerce_output_review(
    value: AtomicEventOutputReview | Mapping[str, Any] | None,
) -> AtomicEventOutputReview | None:
    if value is None:
        return None
    if isinstance(value, AtomicEventOutputReview):
        return value
    if isinstance(value, Mapping):
        return AtomicEventOutputReview.from_mapping(value)
    raise TypeError("output_review must be an AtomicEventOutputReview, mapping, or None")


def _coerce_quality(
    value: AtomicEventQualityScore | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, AtomicEventQualityScore):
        return asdict(value)
    if isinstance(value, Mapping):
        return value
    raise TypeError("quality must be an AtomicEventQualityScore, mapping, or None")


def _quality_bool(quality: Mapping[str, Any] | None, name: str) -> bool | None:
    if quality is None:
        return None
    value = quality.get(name)
    return value if isinstance(value, bool) else None


def _coalesce(first: bool | None, second: bool | None) -> bool | None:
    return first if first is not None else second


def _input_evidence_sufficient(review: AtomicEventInputReview | None) -> bool | None:
    """Return whether the reviewed input can support an atomic visual judgment.

    This deliberately answers a narrower and more conservative question than
    whether *some* frames were decoded.  A true value means the review has
    positively established the media/processor chain and the complete local
    transition needed to judge the target event.  In particular, a failed
    processor probe or an omitted contact/selected-transition fact must never
    coexist with a true ``visual_joint_correct`` result.

    ``None`` retains the distinction between an unreviewed fact and an observed
    failure.  Every positive fact must be true, while a positively observed
    issue flag prevents success; either kind of observed failure returns false
    immediately.
    """

    if review is None:
        return None

    required_true = (
        # L0: an apparent visual success is not meaningful unless the actual
        # decode -> timestamp -> processor path was all reviewed as aligned.
        review.decode_succeeded,
        review.frame_times_aligned,
        review.metadata_grid_aligned,
        review.processor_probe_succeeded,
        # L1: an atomic transition needs before, target interaction, after,
        # state change/direction, and a selected pair that actually spans it.
        review.pre_state_visible,
        review.target_hand_or_control_visible,
        review.post_state_visible,
        review.contact_or_state_change_visible,
        review.direction_observable,
        review.selected_pair_hits_transition,
        # L2/L3: the target must remain legible and temporally attributable.
        review.object_or_control_legible_after_rendering,
        review.ordering_unambiguous,
        review.block_association_unambiguous,
    )
    disqualifying_issue_flags = (
        review.temporal_gap_relaxed_too_far,
        review.object_or_control_too_small_after_thumbnail,
        review.roi_excludes_target,
        review.roi_or_scale_discontinuity,
        review.state_inversion_observed,
    )
    if any(value is False for value in required_true) or any(
        value is True for value in disqualifying_issue_flags
    ):
        return False
    if all(value is True for value in required_true):
        return True
    return None


def _target_joint_correct(
    output: AtomicEventOutputReview | None,
    quality: Mapping[str, Any] | None,
) -> bool | None:
    action = _coalesce(
        output.target_action_asserted if output is not None else None,
        _quality_bool(quality, "atomic_action_match"),
    )
    object_match = _coalesce(
        output.target_object_asserted if output is not None else None,
        _quality_bool(quality, "object_match"),
    )
    direction = _coalesce(
        output.correct_direction_asserted if output is not None else None,
        _quality_bool(quality, "direction_match"),
    )
    direction_required = _quality_bool(quality, "direction_required")

    values = [action, object_match]
    if direction_required is True or direction is not None:
        values.append(direction)
    if any(value is False for value in values):
        return False
    if values and all(value is True for value in values):
        return True
    return None


def _review_quality_disagreement(
    output: AtomicEventOutputReview | None,
    quality: Mapping[str, Any] | None,
) -> bool:
    if output is None:
        return False
    if output.lexical_score_disagreement is True or output.annotation_ambiguous is True:
        return True
    compared = (
        (output.target_action_asserted, _quality_bool(quality, "atomic_action_match")),
        (output.target_object_asserted, _quality_bool(quality, "object_match")),
        (output.correct_direction_asserted, _quality_bool(quality, "direction_match")),
        (
            output.opposite_direction_asserted,
            _quality_bool(quality, "opposite_direction_asserted"),
        ),
    )
    return any(
        reviewed is not None and lexical is not None and reviewed is not lexical
        for reviewed, lexical in compared
    )


def _basic_temporal_evidence_visible(review: AtomicEventInputReview | None) -> bool:
    if review is None:
        return False
    return all(
        value is True
        for value in (
            review.pre_state_visible,
            review.post_state_visible,
            review.direction_observable,
        )
    )


def _append_code(
    candidates: dict[AtomicEventErrorLayer, list[AtomicEventErrorCode]],
    layer: AtomicEventErrorLayer,
    code: AtomicEventErrorCode,
) -> None:
    candidates.setdefault(layer, []).append(code)


def _human_rationale(
    input_review: AtomicEventInputReview | None,
    output_review: AtomicEventOutputReview | None,
) -> str:
    values = (
        input_review.reviewer_rationale if input_review is not None else "",
        output_review.reviewer_rationale if output_review is not None else "",
    )
    return " ".join(value.strip() for value in values if value.strip())


def _evidence_links(
    input_review: AtomicEventInputReview | None,
    output_review: AtomicEventOutputReview | None,
) -> tuple[str, ...]:
    values = (
        *(input_review.evidence_links if input_review is not None else ()),
        *(output_review.evidence_links if output_review is not None else ()),
    )
    return tuple(dict.fromkeys(values))


def classify_atomic_event_error(
    record: Mapping[str, Any] | None = None,
    *,
    quality: AtomicEventQualityScore | Mapping[str, Any] | None = None,
    input_review: AtomicEventInputReview | Mapping[str, Any] | None = None,
    output_review: AtomicEventOutputReview | Mapping[str, Any] | None = None,
) -> AtomicEventErrorClassification:
    """Classify one Qwen/Mage benchmark row using only retained sidecar facts.

    ``record`` may contain ``quality``, ``input_review``, and ``output_review``
    mappings.  If it has no retained quality mapping but has a raw text/reference
    shape accepted by :func:`score_raw_output_record`, the existing local lexical
    scorer is used.  Supplying an explicit argument always takes precedence over
    the field in ``record``.

    A missing review deliberately produces ``unclear`` or ``not-reviewable``
    rather than a speculative visual L1--L4 label.
    """

    if record is not None and not isinstance(record, Mapping):
        raise TypeError("record must be a mapping or None")

    raw_input = input_review
    raw_output = output_review
    raw_quality = quality
    if record is not None:
        if raw_input is None:
            candidate_input = record.get("input_review")
            if not isinstance(candidate_input, (AtomicEventInputReview, Mapping)):
                candidate_input = record.get("input_observation")
            if isinstance(candidate_input, (AtomicEventInputReview, Mapping)):
                raw_input = candidate_input
        if raw_output is None:
            candidate_output = record.get("output_review")
            if not isinstance(candidate_output, (AtomicEventOutputReview, Mapping)):
                candidate_output = record.get("output_adjudication")
            if isinstance(candidate_output, (AtomicEventOutputReview, Mapping)):
                raw_output = candidate_output
        if raw_quality is None:
            candidate_quality = record.get("quality")
            if isinstance(candidate_quality, (AtomicEventQualityScore, Mapping)):
                raw_quality = candidate_quality
            if raw_quality is None:
                try:
                    raw_quality = score_raw_output_record(record)["quality"]
                except ValueError:
                    # A partial review-pack row legitimately has no raw output/reference.
                    raw_quality = None

    reviewed_input = _coerce_input_review(raw_input)
    reviewed_output = _coerce_output_review(raw_output)
    quality_facts = _coerce_quality(raw_quality)

    candidates: dict[AtomicEventErrorLayer, list[AtomicEventErrorCode]] = {}
    directly_reviewed_layers: set[AtomicEventErrorLayer] = set()

    if reviewed_input is not None:
        media_checks = (
            (reviewed_input.decode_succeeded, AtomicEventErrorCode.L0_DECODE_FAILURE),
            (reviewed_input.frame_times_aligned, AtomicEventErrorCode.L0_FRAME_TIME_MISMATCH),
            (
                reviewed_input.metadata_grid_aligned,
                AtomicEventErrorCode.L0_METADATA_GRID_MISMATCH,
            ),
            (
                reviewed_input.processor_probe_succeeded,
                AtomicEventErrorCode.L0_PROCESSOR_PROBE_FAILURE,
            ),
        )
        for value, code in media_checks:
            if value is False:
                _append_code(candidates, AtomicEventErrorLayer.L0_MEDIA_PROCESSOR, code)
                directly_reviewed_layers.add(AtomicEventErrorLayer.L0_MEDIA_PROCESSOR)

        temporal_checks = (
            (reviewed_input.pre_state_visible, AtomicEventErrorCode.L1_PRE_STATE_NOT_VISIBLE),
            (
                reviewed_input.target_hand_or_control_visible,
                AtomicEventErrorCode.L1_TARGET_CONTROL_NOT_VISIBLE,
            ),
            (reviewed_input.post_state_visible, AtomicEventErrorCode.L1_POST_STATE_NOT_VISIBLE),
            (
                reviewed_input.contact_or_state_change_visible,
                AtomicEventErrorCode.L1_CONTACT_OR_STATE_CHANGE_NOT_VISIBLE,
            ),
            (
                reviewed_input.direction_observable,
                AtomicEventErrorCode.L1_DIRECTION_NOT_OBSERVABLE,
            ),
            (
                reviewed_input.selected_pair_hits_transition,
                AtomicEventErrorCode.L1_SELECTED_PAIR_MISSES_TRANSITION,
            ),
        )
        for value, code in temporal_checks:
            if value is False:
                _append_code(candidates, AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE, code)
                directly_reviewed_layers.add(AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE)
        if reviewed_input.temporal_gap_relaxed_too_far is True:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE,
                AtomicEventErrorCode.L1_TEMPORAL_GAP_RELAXED_TOO_FAR,
            )
            directly_reviewed_layers.add(AtomicEventErrorLayer.L1_TEMPORAL_EVIDENCE)

        spatial_checks = (
            (
                reviewed_input.object_or_control_legible_after_rendering,
                AtomicEventErrorCode.L2_OBJECT_OR_CONTROL_NOT_LEGIBLE,
            ),
            (
                reviewed_input.object_or_control_too_small_after_thumbnail,
                AtomicEventErrorCode.L2_OBJECT_OR_CONTROL_TOO_SMALL,
            ),
            (reviewed_input.roi_excludes_target, AtomicEventErrorCode.L2_ROI_EXCLUDES_TARGET),
            (
                reviewed_input.roi_or_scale_discontinuity,
                AtomicEventErrorCode.L2_MIXED_SCALE_DISCONTINUITY,
            ),
        )
        for value, code in spatial_checks:
            is_legibility_check = code is AtomicEventErrorCode.L2_OBJECT_OR_CONTROL_NOT_LEGIBLE
            is_issue = value is False if is_legibility_check else value is True
            if is_issue:
                _append_code(candidates, AtomicEventErrorLayer.L2_SPATIAL_FIDELITY, code)
                directly_reviewed_layers.add(AtomicEventErrorLayer.L2_SPATIAL_FIDELITY)

        binding_checks = (
            (reviewed_input.ordering_unambiguous, AtomicEventErrorCode.L3_ORDERING_AMBIGUOUS),
            (
                reviewed_input.block_association_unambiguous,
                AtomicEventErrorCode.L3_BLOCK_ASSOCIATION_AMBIGUOUS,
            ),
        )
        for value, code in binding_checks:
            if value is False:
                _append_code(candidates, AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT, code)
                directly_reviewed_layers.add(AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT)
        if reviewed_input.state_inversion_observed is True:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT,
                AtomicEventErrorCode.L3_STATE_INVERSION,
            )
            directly_reviewed_layers.add(AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT)

    output_opposite_direction = _coalesce(
        reviewed_output.opposite_direction_asserted if reviewed_output is not None else None,
        _quality_bool(quality_facts, "opposite_direction_asserted"),
    )
    if output_opposite_direction is True and _basic_temporal_evidence_visible(reviewed_input):
        _append_code(
            candidates,
            AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT,
            AtomicEventErrorCode.L3_STATE_INVERSION,
        )
        if reviewed_output is not None and reviewed_output.opposite_direction_asserted is True:
            directly_reviewed_layers.add(AtomicEventErrorLayer.L3_TEMPORAL_BINDING_LAYOUT)

    target_joint_correct = _target_joint_correct(reviewed_output, quality_facts)
    output_main_predicate = (
        reviewed_output.target_is_main_predicate if reviewed_output is not None else None
    )
    review_quality_disagreement = _review_quality_disagreement(reviewed_output, quality_facts)
    attention_confirmed = bool(
        reviewed_input is not None and reviewed_input.attention_competition_observed is True
    )
    distractor_competes = bool(
        reviewed_input is not None
        and reviewed_input.distractor_present is True
        and (
            output_main_predicate is False
            or _quality_bool(quality_facts, "adjacent_action_only") is True
        )
    )
    if attention_confirmed or distractor_competes:
        _append_code(
            candidates,
            AtomicEventErrorLayer.L4_ATTENTION_COMPETITION,
            AtomicEventErrorCode.L4_ATTENTION_COMPETITION,
        )
        if attention_confirmed or output_main_predicate is False:
            directly_reviewed_layers.add(AtomicEventErrorLayer.L4_ATTENTION_COMPETITION)

    adjacent_narrative = (
        reviewed_output.adjacent_purpose_or_action_present is True
        if reviewed_output is not None
        else False
    )
    insufficiently_atomic = bool(
        reviewed_output is not None and reviewed_output.insufficiently_atomic_prose is True
    )
    multiple_action_story = _quality_bool(quality_facts, "multiple_action_story") is True
    hedged = _quality_bool(quality_facts, "hedged") is True
    strict_mapper_sufficient = _quality_bool(quality_facts, "mapper_input_sufficient")
    strict_contract_only = (
        target_joint_correct is True
        and strict_mapper_sufficient is False
        and not review_quality_disagreement
    )

    # L5 is intentionally gated on the target semantics being present.  A raw
    # target miss with no input review is an unknown cause, not evidence of a
    # perception/spatial/temporal failure.
    if target_joint_correct is True:
        if adjacent_narrative:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY,
                AtomicEventErrorCode.L5_ADJACENT_PURPOSE_OR_ACTION,
            )
            directly_reviewed_layers.add(AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY)
        if output_main_predicate is False and not distractor_competes:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY,
                AtomicEventErrorCode.L5_ADJACENT_PURPOSE_OR_ACTION,
            )
            if reviewed_output is not None:
                directly_reviewed_layers.add(AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY)
        if multiple_action_story:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY,
                AtomicEventErrorCode.L5_MULTIPLE_ACTION_STORY,
            )
        if hedged or insufficiently_atomic:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY,
                AtomicEventErrorCode.L5_HEDGED_OR_INSUFFICIENTLY_ATOMIC,
            )
            if insufficiently_atomic:
                directly_reviewed_layers.add(AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY)
        if strict_contract_only and not (
            adjacent_narrative or multiple_action_story or hedged or insufficiently_atomic
        ):
            _append_code(
                candidates,
                AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY,
                AtomicEventErrorCode.L5_STRICT_PROSE_CONTRACT_ONLY,
            )

    if reviewed_output is not None:
        if review_quality_disagreement:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE,
                AtomicEventErrorCode.L6_LEXICAL_SCORE_DISAGREEMENT,
            )
            directly_reviewed_layers.add(AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE)
        if reviewed_output.annotation_ambiguous is True:
            _append_code(
                candidates,
                AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE,
                AtomicEventErrorCode.L6_REFERENCE_AMBIGUITY,
            )
            directly_reviewed_layers.add(AtomicEventErrorLayer.L6_EVALUATOR_REFERENCE)

    primary_layer = next((layer for layer in _LAYER_ORDER if layer in candidates), None)
    secondary_layers = tuple(
        layer for layer in _LAYER_ORDER if layer in candidates and layer is not primary_layer
    )
    codes = tuple(code for layer in _LAYER_ORDER for code in candidates.get(layer, []))

    if primary_layer is None and target_joint_correct is False:
        codes = (*codes, AtomicEventErrorCode.OUTPUT_TARGET_NOT_CONFIRMED)

    input_evidence_sufficient = _input_evidence_sufficient(reviewed_input)
    # Do not let a text-only target miss stand in for a visual finding.  The
    # complete input review is what permits this combined visual/output result.
    if input_evidence_sufficient is True:
        visual_joint_correct: bool | None = target_joint_correct
    elif input_evidence_sufficient is False:
        visual_joint_correct = False
    else:
        visual_joint_correct = None

    if primary_layer is None:
        if quality_facts is None and reviewed_input is None and reviewed_output is None:
            evidence_status = AtomicEventEvidenceStatus.NOT_REVIEWABLE
        elif target_joint_correct is True and input_evidence_sufficient is True:
            evidence_status = AtomicEventEvidenceStatus.CONFIRMED
        else:
            evidence_status = AtomicEventEvidenceStatus.UNCLEAR
    elif primary_layer in directly_reviewed_layers:
        evidence_status = AtomicEventEvidenceStatus.CONFIRMED
    else:
        # A lexical score can reveal an output pattern but cannot prove a visual
        # mechanism or replace an output adjudication.
        evidence_status = AtomicEventEvidenceStatus.UNCLEAR

    rationale = _rationale(
        primary_layer=primary_layer,
        codes=codes,
        target_joint_correct=target_joint_correct,
        mapper_input_sufficient=strict_mapper_sufficient,
        reviewer_rationale=_human_rationale(reviewed_input, reviewed_output),
    )
    return AtomicEventErrorClassification(
        primary_layer=primary_layer,
        secondary_layers=secondary_layers,
        evidence_status=evidence_status,
        codes=codes,
        allowed_next_factor=(
            _NEXT_FACTOR_BY_LAYER[primary_layer] if primary_layer is not None else None
        ),
        rationale=rationale,
        evidence_links=_evidence_links(reviewed_input, reviewed_output),
        input_evidence_sufficient=input_evidence_sufficient,
        target_joint_correct=target_joint_correct,
        visual_joint_correct=visual_joint_correct,
        mapper_input_sufficient=strict_mapper_sufficient,
    )


def _rationale(
    *,
    primary_layer: AtomicEventErrorLayer | None,
    codes: tuple[AtomicEventErrorCode, ...],
    target_joint_correct: bool | None,
    mapper_input_sufficient: bool | None,
    reviewer_rationale: str,
) -> str:
    if primary_layer is None:
        if target_joint_correct is False:
            base = (
                "The retained output does not establish the target joint action, "
                "but no reviewed input mechanism is evidenced."
            )
        elif target_joint_correct is True:
            base = "No layered error is evidenced by the retained review facts."
        else:
            base = "No retained review evidence is sufficient to attribute a failure layer."
    else:
        labels = ", ".join(
            code.value for code in codes if code.value.startswith(primary_layer.value)
        )
        base = f"{primary_layer.value} is the first evidenced layer ({labels})."
        if (
            primary_layer is AtomicEventErrorLayer.L5_OUTPUT_GRANULARITY
            and target_joint_correct is True
            and mapper_input_sufficient is False
        ):
            base += (
                " Target action/object/direction remain asserted; strict mapper prose "
                "sufficiency is separate from visual failure."
            )
    return f"{base} {reviewer_rationale}".strip()


def classify_atomic_event_record(
    record: Mapping[str, Any],
) -> AtomicEventErrorClassification:
    """Named record-oriented alias for review-pack and runner callers."""

    return classify_atomic_event_error(record)


def classify_atomic_event_error_records(
    records: Sequence[Mapping[str, Any]],
) -> list[AtomicEventErrorClassification]:
    """Classify a local sequence without reading media or invoking a model."""

    return [classify_atomic_event_error(record) for record in records]


def summarize_atomic_event_error_classifications(
    classifications: Sequence[AtomicEventErrorClassification],
) -> dict[str, Any]:
    """Summarize sidecar classifications while retaining unknown attribution."""

    primary_counts = Counter(
        classification.primary_layer.value if classification.primary_layer is not None else "none"
        for classification in classifications
    )
    evidence_counts = Counter(
        classification.evidence_status.value for classification in classifications
    )
    return {
        "taxonomy_version": ATOMIC_EVENT_ERROR_TAXONOMY_VERSION,
        "case_count": len(classifications),
        "primary_layer_counts": dict(sorted(primary_counts.items())),
        "evidence_status_counts": dict(sorted(evidence_counts.items())),
    }
