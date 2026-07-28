"""Durable, causal adaptive-sampling decisions.

This module deliberately sits between pure adaptive coverage planning and any
runtime that materializes extra media.  A decision captures the immutable
upstream evidence and policy that caused additional coordinates, then exposes
only those coordinates that are incremental to the already-frozen base
coverage.  It never makes an event-presence claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.sampling.adaptive import (
    AdaptiveCoveragePlan,
    AdaptiveCoveragePlanner,
    AdaptiveCoveragePolicy,
    AdaptiveUpgradeProvenance,
    AdaptiveUpgradeReason,
    AdaptiveUpgradeRequest,
    AdaptiveUpgradeTargetRole,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

ADAPTIVE_SAMPLING_DECISION_SCHEMA_VERSION: Literal["1.0"] = "1.0"
ADAPTIVE_SAMPLING_DECISION_PROJECTION_VERSION: Literal["adaptive-sampling-decision-semantic-v1"] = (
    "adaptive-sampling-decision-semantic-v1"
)
ADAPTIVE_SAMPLING_DECISION_SCOPE_PROJECTION_VERSION: Literal[
    "adaptive-sampling-decision-scope-semantic-v1"
] = "adaptive-sampling-decision-scope-semantic-v1"
ADAPTIVE_LATE_FEEDBACK_AUDIT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
ADAPTIVE_LATE_FEEDBACK_AUDIT_PROJECTION_VERSION: Literal[
    "adaptive-sampling-late-feedback-semantic-v1"
] = "adaptive-sampling-late-feedback-semantic-v1"


class AdaptiveTriggerEvidenceKind(StrEnum):
    """The immutable evidence family cited by one adaptive trigger."""

    ACCEPTED_SELECTION = "ACCEPTED_SELECTION"
    SELECTED_OUTPUT = "SELECTED_OUTPUT"
    ENRICHED_OUTPUT = "ENRICHED_OUTPUT"
    SOURCE_QUALITY_ARTIFACT = "SOURCE_QUALITY_ARTIFACT"


class AdaptiveSamplingDecisionOutcome(StrEnum):
    """A decision about extra sampling, never a claim about event presence."""

    ADDITIONAL_TARGETS_SCHEDULED = "ADDITIONAL_TARGETS_SCHEDULED"
    BASE_COVERAGE_ALREADY_CONTAINS_TARGETS = "BASE_COVERAGE_ALREADY_CONTAINS_TARGETS"
    UPSTREAM_ABSTAINED = "UPSTREAM_ABSTAINED"
    INCOMPLETE_ACCEPTED_EVIDENCE = "INCOMPLETE_ACCEPTED_EVIDENCE"
    UPGRADE_BUDGET_EXHAUSTED = "UPGRADE_BUDGET_EXHAUSTED"
    DENSE_QA_ALREADY_COMPLETE = "DENSE_QA_ALREADY_COMPLETE"


class AdaptiveNoAdditionalWorkProofKind(StrEnum):
    """The only currently supported domain proof for dense-QA no-extra-work."""

    DENSE_QA_COARSE_COMPLETE = "DENSE_QA_COARSE_COMPLETE"


class AdaptiveDecisionBaseBinding(StrictModel):
    """Exact identity of coverage frozen before feedback is consumed."""

    sampling_plan_sha256: Sha256Digest
    package_set_id: NonEmptyString
    package_set_member_manifest_sha256: Sha256Digest
    package_set_split_plan_sha256: Sha256Digest


class AcceptedAdaptiveEvidenceBinding(StrictModel):
    """The accepted provider branch allowed to cause an adaptive decision."""

    selection_id: OpaqueUuid
    selection_decision_logical_key: NonEmptyString
    selected_output_sha256: Sha256Digest
    enriched_output_artifact_id: OpaqueUuid
    enriched_output_semantic_sha256: Sha256Digest

    @property
    def selection_evidence_sha256(self) -> Sha256Digest:
        """Return a stable digest for selection-level trigger provenance."""

        return accepted_selection_evidence_sha256(self)


class AdaptiveDecisionSourceBinding(StrictModel):
    """Admitted source and alignment terms shared by every extra target."""

    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest


class AdaptiveTriggerProvenance(StrictModel):
    """One evidence-backed adaptive trigger before coordinate expansion."""

    camera_id: CameraId
    trigger_timestamp_ns: Nanoseconds
    reason: AdaptiveUpgradeReason
    evidence_kind: AdaptiveTriggerEvidenceKind
    evidence_sha256: Sha256Digest
    evidence_locator: NonEmptyString


class AdaptiveTargetTriggerProvenance(StrictModel):
    """One expanded trigger fact retained on an incremental coordinate."""

    trigger: AdaptiveTriggerProvenance
    role: AdaptiveUpgradeTargetRole
    context_offset_ns: Nanoseconds = 0
    context_clipped: bool = False

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if self.role is AdaptiveUpgradeTargetRole.TRIGGER:
            if self.context_offset_ns != 0 or self.context_clipped:
                raise ValueError("trigger provenance cannot carry contextual clipping")
        elif self.role is AdaptiveUpgradeTargetRole.PRE_CONTEXT:
            if self.context_offset_ns >= 0:
                raise ValueError("pre-context provenance requires a negative offset")
        elif self.context_offset_ns <= 0:
            raise ValueError("post-context provenance requires a positive offset")
        return self


class AdaptiveIncrementalTarget(StrictModel):
    """A canonical coordinate absent from the frozen base coverage."""

    ordinal: NonNegativeInt
    camera_id: CameraId
    target_ns: Nanoseconds
    trigger_provenance: tuple[AdaptiveTargetTriggerProvenance, ...]

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if not self.trigger_provenance:
            raise ValueError("incremental targets require trigger provenance")
        canonical = tuple(sorted(self.trigger_provenance, key=_target_provenance_sort_key))
        if self.trigger_provenance != canonical:
            raise ValueError("incremental target provenance must be in canonical order")
        if len({_canonical_key(item) for item in self.trigger_provenance}) != len(
            self.trigger_provenance
        ):
            raise ValueError("incremental target provenance must not contain duplicates")
        return self


class AdaptiveCoverageAccounting(StrictModel):
    """Planner counters retained even when no incremental coordinates remain."""

    base_target_count: NonNegativeInt
    upgrade_coordinate_count: NonNegativeInt
    incremental_target_count: NonNegativeInt
    upgrade_coordinates_deduplicated_into_base: NonNegativeInt
    dropped_by_per_camera_budget: NonNegativeInt
    dropped_by_total_budget: NonNegativeInt

    @model_validator(mode="after")
    def validate_counters(self) -> Self:
        expected = (
            self.incremental_target_count
            + self.upgrade_coordinates_deduplicated_into_base
            + self.dropped_by_per_camera_budget
            + self.dropped_by_total_budget
        )
        if self.upgrade_coordinate_count != expected:
            raise ValueError("coverage accounting must explain every upgrade coordinate")
        return self


class AdaptiveNoAdditionalWorkProof(StrictModel):
    """A domain proof that permits a dense-QA extra-work decision to be empty."""

    proof_kind: AdaptiveNoAdditionalWorkProofKind
    evidence_artifact_id: NonEmptyString
    evidence_sha256: Sha256Digest
    policy_version: SchemaVersion


class AdaptiveSamplingDecision(StrictModel):
    """Sealed, content-addressed decision consumed before extra work executes."""

    schema_version: Literal["1.0"] = ADAPTIVE_SAMPLING_DECISION_SCHEMA_VERSION
    decision_id: NonEmptyString
    decision_scope_sha256: Sha256Digest
    semantic_sha256: Sha256Digest
    base: AdaptiveDecisionBaseBinding
    accepted_evidence: AcceptedAdaptiveEvidenceBinding
    source: AdaptiveDecisionSourceBinding
    effective_interval: NanosecondInterval
    policy: AdaptiveCoveragePolicy
    triggers: tuple[AdaptiveTriggerProvenance, ...]
    incremental_targets: tuple[AdaptiveIncrementalTarget, ...]
    coverage_accounting: AdaptiveCoverageAccounting | None
    outcome: AdaptiveSamplingDecisionOutcome
    outcome_detail: NonEmptyString
    no_additional_work_proof: AdaptiveNoAdditionalWorkProof | None = None
    projection_version: Literal["adaptive-sampling-decision-semantic-v1"] = (
        ADAPTIVE_SAMPLING_DECISION_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_identity_and_shape(self) -> Self:
        expected_scope = semantic_sha256(adaptive_sampling_decision_scope_projection(self))
        if self.decision_scope_sha256 != expected_scope:
            raise ValueError("decision scope digest does not match immutable decision inputs")

        canonical_triggers = _normalize_triggers(self.triggers)
        if self.triggers != canonical_triggers:
            raise ValueError("decision triggers must be deduplicated and in canonical order")
        if any(
            not self.effective_interval.contains(item.trigger_timestamp_ns)
            for item in self.triggers
        ):
            raise ValueError("trigger timestamps must lie inside effective_interval")
        _validate_trigger_evidence(self.triggers, self.accepted_evidence)

        expected_targets = tuple(
            sorted(
                self.incremental_targets,
                key=lambda item: (item.camera_id.value, item.target_ns),
            )
        )
        if self.incremental_targets != expected_targets:
            raise ValueError("incremental targets must use canonical camera/timestamp order")
        if tuple(item.ordinal for item in self.incremental_targets) != tuple(
            range(len(self.incremental_targets))
        ):
            raise ValueError("incremental target ordinals must be contiguous from zero")
        coordinates = tuple((item.camera_id, item.target_ns) for item in self.incremental_targets)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("incremental target coordinates must be unique")
        if any(
            not self.effective_interval.contains(item.target_ns)
            for item in self.incremental_targets
        ):
            raise ValueError("incremental targets must lie inside effective_interval")
        if self.coverage_accounting is not None and (
            self.coverage_accounting.incremental_target_count != len(self.incremental_targets)
        ):
            raise ValueError("coverage accounting incremental count does not match targets")

        _validate_outcome(self)
        _validate_planned_content(self)
        expected_digest = semantic_sha256(adaptive_sampling_decision_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("decision semantic digest does not match its projection")
        if self.decision_id != f"adaptive-sampling-decision:{expected_digest}":
            raise ValueError("decision ID does not match its semantic digest")
        return self


class AdaptiveLateFeedbackAudit(StrictModel):
    """Append-only evidence that arrived after a decision slot was sealed."""

    schema_version: Literal["1.0"] = ADAPTIVE_LATE_FEEDBACK_AUDIT_SCHEMA_VERSION
    audit_id: NonEmptyString
    semantic_sha256: Sha256Digest
    decision_scope_sha256: Sha256Digest
    arrival_id: NonEmptyString
    trigger: AdaptiveTriggerProvenance
    observed_at: Rfc3339Timestamp
    projection_version: Literal["adaptive-sampling-late-feedback-semantic-v1"] = (
        ADAPTIVE_LATE_FEEDBACK_AUDIT_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = semantic_sha256(adaptive_late_feedback_audit_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("late-feedback audit semantic digest does not match its projection")
        if self.audit_id != f"adaptive-late-feedback:{expected}":
            raise ValueError("late-feedback audit ID does not match its semantic digest")
        return self


def accepted_selection_evidence_sha256(
    accepted_evidence: AcceptedAdaptiveEvidenceBinding,
) -> Sha256Digest:
    """Derive the accepted-selection reference allowed in trigger provenance."""

    if not isinstance(accepted_evidence, AcceptedAdaptiveEvidenceBinding):
        raise TypeError("accepted_evidence must be an AcceptedAdaptiveEvidenceBinding")
    return semantic_sha256(
        {
            "selection_id": accepted_evidence.selection_id,
            "selection_decision_logical_key": accepted_evidence.selection_decision_logical_key,
        }
    )


def adaptive_sampling_decision_scope_projection(
    decision: AdaptiveSamplingDecision,
) -> dict[str, object]:
    """Return the trigger-independent, policy-bound logical slot to seal once."""

    if not isinstance(decision, AdaptiveSamplingDecision):
        raise TypeError("decision must be an AdaptiveSamplingDecision")
    return _decision_scope_projection(
        policy=decision.policy,
        base=decision.base,
        accepted_evidence=decision.accepted_evidence,
        source=decision.source,
        effective_interval=decision.effective_interval,
    )


def adaptive_sampling_decision_projection(
    decision: AdaptiveSamplingDecision,
) -> dict[str, object]:
    """Return the complete versioned, content-addressed decision projection."""

    if not isinstance(decision, AdaptiveSamplingDecision):
        raise TypeError("decision must be an AdaptiveSamplingDecision")
    return {
        "projection_version": decision.projection_version,
        "schema_version": decision.schema_version,
        "decision_scope_sha256": decision.decision_scope_sha256,
        "base": decision.base,
        "accepted_evidence": decision.accepted_evidence,
        "source": decision.source,
        "effective_interval": decision.effective_interval,
        "policy": decision.policy,
        "triggers": decision.triggers,
        "incremental_targets": decision.incremental_targets,
        "coverage_accounting": decision.coverage_accounting,
        "outcome": decision.outcome.value,
        "outcome_detail": decision.outcome_detail,
        "no_additional_work_proof": decision.no_additional_work_proof,
    }


def adaptive_late_feedback_audit_projection(
    audit: AdaptiveLateFeedbackAudit,
) -> dict[str, object]:
    """Return the immutable projection for one sealed-decision late arrival."""

    if not isinstance(audit, AdaptiveLateFeedbackAudit):
        raise TypeError("audit must be an AdaptiveLateFeedbackAudit")
    return {
        "projection_version": audit.projection_version,
        "schema_version": audit.schema_version,
        "decision_scope_sha256": audit.decision_scope_sha256,
        "arrival_id": audit.arrival_id,
        "trigger": audit.trigger,
        "observed_at": audit.observed_at,
    }


def build_adaptive_sampling_decision(
    *,
    base: AdaptiveDecisionBaseBinding,
    accepted_evidence: AcceptedAdaptiveEvidenceBinding,
    source: AdaptiveDecisionSourceBinding,
    effective_interval: NanosecondInterval,
    policy: AdaptiveCoveragePolicy,
    triggers: Iterable[AdaptiveTriggerProvenance] = (),
    no_trigger_outcome: AdaptiveSamplingDecisionOutcome | None = None,
    outcome_detail: str | None = None,
    no_additional_work_proof: AdaptiveNoAdditionalWorkProof | None = None,
) -> AdaptiveSamplingDecision:
    """Seal canonical extra coordinates from accepted, evidence-backed triggers.

    Input ordering and exact duplicate delivery cannot alter the result.  The
    pure planner retains base coverage internally; this function persists only
    coordinates that are genuinely incremental to that base coverage.
    """

    checked_base = _require_model(base, AdaptiveDecisionBaseBinding, "base")
    checked_evidence = _require_model(
        accepted_evidence,
        AcceptedAdaptiveEvidenceBinding,
        "accepted_evidence",
    )
    checked_source = _require_model(source, AdaptiveDecisionSourceBinding, "source")
    checked_interval = _require_model(effective_interval, NanosecondInterval, "effective_interval")
    checked_policy = _require_model(policy, AdaptiveCoveragePolicy, "policy")
    normalized = _normalize_triggers(triggers)
    if any(not checked_interval.contains(item.trigger_timestamp_ns) for item in normalized):
        raise ValueError("trigger timestamps must lie inside effective_interval")
    _validate_trigger_evidence(normalized, checked_evidence)

    if normalized:
        if no_trigger_outcome is not None:
            raise ValueError("no_trigger_outcome is only valid when trigger input is empty")
        if no_additional_work_proof is not None:
            raise ValueError("no-additional-work proof is only valid without triggers")
        try:
            coverage = AdaptiveCoveragePlanner(checked_policy).plan(
                checked_interval,
                tuple(
                    AdaptiveUpgradeRequest(
                        camera_id=item.camera_id,
                        trigger_timestamp_ns=item.trigger_timestamp_ns,
                        reason=item.reason,
                    )
                    for item in normalized
                ),
            )
        except ValueError as error:
            if "budget cannot preserve every original upgrade trigger" not in str(error):
                raise
            outcome = AdaptiveSamplingDecisionOutcome.UPGRADE_BUDGET_EXHAUSTED
            targets: tuple[AdaptiveIncrementalTarget, ...] = ()
            accounting = None
            detail = outcome_detail or "policy budget cannot preserve every original trigger"
        else:
            targets = _incremental_targets(coverage, normalized)
            accounting = _coverage_accounting(coverage)
            if targets:
                outcome = AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED
                detail = outcome_detail or "canonical incremental targets were sealed"
            else:
                outcome = AdaptiveSamplingDecisionOutcome.BASE_COVERAGE_ALREADY_CONTAINS_TARGETS
                detail = outcome_detail or "frozen base coverage already contains every target"
    else:
        if no_trigger_outcome is None:
            raise ValueError(
                "an explicit no_trigger_outcome is required when no triggers are present"
            )
        if no_trigger_outcome not in {
            AdaptiveSamplingDecisionOutcome.UPSTREAM_ABSTAINED,
            AdaptiveSamplingDecisionOutcome.INCOMPLETE_ACCEPTED_EVIDENCE,
            AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE,
        }:
            raise ValueError("no_trigger_outcome is not valid without triggers")
        if outcome_detail is None:
            raise ValueError("outcome_detail is required when no triggers are present")
        if (no_trigger_outcome is AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE) != (
            no_additional_work_proof is not None
        ):
            raise ValueError("dense-QA no-extra-work outcome requires its domain proof")
        if (
            no_trigger_outcome is not AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE
            and no_additional_work_proof is not None
        ):
            raise ValueError("a no-additional-work proof is only valid for dense-QA completion")
        outcome = no_trigger_outcome
        targets = ()
        accounting = None
        detail = outcome_detail

    scope = _decision_scope_projection(
        policy=checked_policy,
        base=checked_base,
        accepted_evidence=checked_evidence,
        source=checked_source,
        effective_interval=checked_interval,
    )
    scope_digest = semantic_sha256(scope)
    values: dict[str, Any] = {
        "schema_version": ADAPTIVE_SAMPLING_DECISION_SCHEMA_VERSION,
        "decision_scope_sha256": scope_digest,
        "base": checked_base,
        "accepted_evidence": checked_evidence,
        "source": checked_source,
        "effective_interval": checked_interval,
        "policy": checked_policy,
        "triggers": normalized,
        "incremental_targets": targets,
        "coverage_accounting": accounting,
        "outcome": outcome,
        "outcome_detail": detail,
        "no_additional_work_proof": no_additional_work_proof,
        "projection_version": ADAPTIVE_SAMPLING_DECISION_PROJECTION_VERSION,
    }
    draft = AdaptiveSamplingDecision.model_construct(
        decision_id="pending",
        semantic_sha256="0" * 64,
        **values,
    )
    digest = semantic_sha256(adaptive_sampling_decision_projection(draft))
    return AdaptiveSamplingDecision.model_validate(
        {
            **values,
            "decision_id": f"adaptive-sampling-decision:{digest}",
            "semantic_sha256": digest,
        },
        strict=True,
    )


def build_adaptive_late_feedback_audit(
    *,
    decision_scope_sha256: str,
    arrival_id: str,
    trigger: AdaptiveTriggerProvenance,
    observed_at: str,
) -> AdaptiveLateFeedbackAudit:
    """Build an immutable audit record without reopening the sealed decision."""

    checked_trigger = _require_model(trigger, AdaptiveTriggerProvenance, "trigger")
    values: dict[str, Any] = {
        "schema_version": ADAPTIVE_LATE_FEEDBACK_AUDIT_SCHEMA_VERSION,
        "decision_scope_sha256": decision_scope_sha256,
        "arrival_id": arrival_id,
        "trigger": checked_trigger,
        "observed_at": observed_at,
        "projection_version": ADAPTIVE_LATE_FEEDBACK_AUDIT_PROJECTION_VERSION,
    }
    draft = AdaptiveLateFeedbackAudit.model_construct(
        audit_id="pending",
        semantic_sha256="0" * 64,
        **values,
    )
    digest = semantic_sha256(adaptive_late_feedback_audit_projection(draft))
    return AdaptiveLateFeedbackAudit.model_validate(
        {
            **values,
            "audit_id": f"adaptive-late-feedback:{digest}",
            "semantic_sha256": digest,
        },
        strict=True,
    )


def _decision_scope_projection(
    *,
    policy: AdaptiveCoveragePolicy,
    base: AdaptiveDecisionBaseBinding,
    accepted_evidence: AcceptedAdaptiveEvidenceBinding,
    source: AdaptiveDecisionSourceBinding,
    effective_interval: NanosecondInterval,
) -> dict[str, object]:
    return {
        "projection_version": ADAPTIVE_SAMPLING_DECISION_SCOPE_PROJECTION_VERSION,
        "policy": policy,
        "base": base,
        "accepted_evidence": accepted_evidence,
        "source": source,
        "effective_interval": effective_interval,
    }


def _normalize_triggers(
    triggers: Iterable[AdaptiveTriggerProvenance],
) -> tuple[AdaptiveTriggerProvenance, ...]:
    values = tuple(triggers)
    for item in values:
        _require_model(item, AdaptiveTriggerProvenance, "triggers item")
    unique: dict[bytes, AdaptiveTriggerProvenance] = {}
    for item in values:
        unique.setdefault(_canonical_key(item), item)
    return tuple(sorted(unique.values(), key=_trigger_sort_key))


def _validate_trigger_evidence(
    triggers: Sequence[AdaptiveTriggerProvenance],
    accepted_evidence: AcceptedAdaptiveEvidenceBinding,
) -> None:
    selection_digest = accepted_selection_evidence_sha256(accepted_evidence)
    for trigger in triggers:
        if trigger.evidence_kind is AdaptiveTriggerEvidenceKind.ACCEPTED_SELECTION:
            if trigger.evidence_sha256 != selection_digest:
                raise ValueError("selection trigger must cite the accepted selection digest")
        elif trigger.evidence_kind is AdaptiveTriggerEvidenceKind.SELECTED_OUTPUT:
            if trigger.evidence_sha256 != accepted_evidence.selected_output_sha256:
                raise ValueError("selected-output trigger must cite the accepted selected output")
        elif (
            trigger.evidence_kind is AdaptiveTriggerEvidenceKind.ENRICHED_OUTPUT
            and trigger.evidence_sha256 != accepted_evidence.enriched_output_semantic_sha256
        ):
            raise ValueError("enriched-output trigger must cite the accepted enriched output")


def _incremental_targets(
    coverage: AdaptiveCoveragePlan,
    triggers: tuple[AdaptiveTriggerProvenance, ...],
) -> tuple[AdaptiveIncrementalTarget, ...]:
    targets: list[AdaptiveIncrementalTarget] = []
    for covered in coverage.targets:
        if covered.base_coverage:
            continue
        expanded: list[AdaptiveTargetTriggerProvenance] = []
        for provenance in covered.upgrade_provenance:
            for trigger in triggers:
                if _matches_upgrade_provenance(trigger, covered.camera_id, provenance):
                    expanded.append(
                        AdaptiveTargetTriggerProvenance(
                            trigger=trigger,
                            role=provenance.role,
                            context_offset_ns=provenance.context_offset_ns,
                            context_clipped=provenance.context_clipped,
                        )
                    )
        unique_expanded: dict[bytes, AdaptiveTargetTriggerProvenance] = {}
        for item in expanded:
            unique_expanded.setdefault(_canonical_key(item), item)
        canonical = tuple(sorted(unique_expanded.values(), key=_target_provenance_sort_key))
        targets.append(
            AdaptiveIncrementalTarget(
                ordinal=len(targets),
                camera_id=covered.camera_id,
                target_ns=covered.target_ns,
                trigger_provenance=canonical,
            )
        )
    ordered = tuple(sorted(targets, key=lambda item: (item.camera_id.value, item.target_ns)))
    return tuple(
        AdaptiveIncrementalTarget(
            ordinal=ordinal,
            camera_id=item.camera_id,
            target_ns=item.target_ns,
            trigger_provenance=item.trigger_provenance,
        )
        for ordinal, item in enumerate(ordered)
    )


def _matches_upgrade_provenance(
    trigger: AdaptiveTriggerProvenance,
    camera_id: CameraId,
    provenance: AdaptiveUpgradeProvenance,
) -> bool:
    return (
        trigger.camera_id is camera_id
        and trigger.trigger_timestamp_ns == provenance.trigger_timestamp_ns
        and trigger.reason is provenance.reason
    )


def _coverage_accounting(coverage: AdaptiveCoveragePlan) -> AdaptiveCoverageAccounting:
    return AdaptiveCoverageAccounting(
        base_target_count=coverage.base_target_count,
        upgrade_coordinate_count=coverage.upgrade_coordinate_count,
        incremental_target_count=coverage.upgrade_targets_added,
        upgrade_coordinates_deduplicated_into_base=(
            coverage.upgrade_coordinates_deduplicated_into_base
        ),
        dropped_by_per_camera_budget=coverage.dropped_by_per_camera_budget,
        dropped_by_total_budget=coverage.dropped_by_total_budget,
    )


def _validate_outcome(decision: AdaptiveSamplingDecision) -> None:
    has_targets = bool(decision.incremental_targets)
    has_triggers = bool(decision.triggers)
    if decision.outcome is AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED:
        if not has_targets or not has_triggers or decision.coverage_accounting is None:
            raise ValueError(
                "scheduled outcome requires triggers, targets, and coverage accounting"
            )
    elif decision.outcome is AdaptiveSamplingDecisionOutcome.BASE_COVERAGE_ALREADY_CONTAINS_TARGETS:
        if has_targets or not has_triggers or decision.coverage_accounting is None:
            raise ValueError("base-coverage outcome requires triggers and zero incremental targets")
    elif decision.outcome is AdaptiveSamplingDecisionOutcome.UPGRADE_BUDGET_EXHAUSTED:
        if has_targets or not has_triggers or decision.coverage_accounting is not None:
            raise ValueError(
                "budget exhaustion requires triggers and no unproven target accounting"
            )
    elif decision.outcome in {
        AdaptiveSamplingDecisionOutcome.UPSTREAM_ABSTAINED,
        AdaptiveSamplingDecisionOutcome.INCOMPLETE_ACCEPTED_EVIDENCE,
    }:
        if has_targets or has_triggers or decision.coverage_accounting is not None:
            raise ValueError("abstained or incomplete outcome cannot claim incremental targets")
    elif decision.outcome is AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE:
        if (
            has_targets
            or has_triggers
            or decision.coverage_accounting is not None
            or decision.no_additional_work_proof is None
        ):
            raise ValueError("dense-QA completion requires an explicit domain proof and no targets")
    else:
        raise AssertionError("unhandled adaptive sampling decision outcome")

    if (
        decision.outcome is not AdaptiveSamplingDecisionOutcome.DENSE_QA_ALREADY_COMPLETE
        and decision.no_additional_work_proof is not None
    ):
        raise ValueError("no-additional-work proof is only valid for dense-QA completion")


def _validate_planned_content(decision: AdaptiveSamplingDecision) -> None:
    """Recompute trigger-derived coordinates so a valid hash cannot forge a plan."""

    if not decision.triggers:
        return
    requests = tuple(
        AdaptiveUpgradeRequest(
            camera_id=item.camera_id,
            trigger_timestamp_ns=item.trigger_timestamp_ns,
            reason=item.reason,
        )
        for item in decision.triggers
    )
    try:
        coverage = AdaptiveCoveragePlanner(decision.policy).plan(
            decision.effective_interval,
            requests,
        )
    except ValueError as error:
        if "budget cannot preserve every original upgrade trigger" in str(error):
            if decision.outcome is not AdaptiveSamplingDecisionOutcome.UPGRADE_BUDGET_EXHAUSTED:
                raise ValueError(
                    "trigger plan exhausted its budget but decision says otherwise"
                ) from error
            return
        raise ValueError("decision triggers cannot resolve under the frozen policy") from error

    expected_targets = _incremental_targets(coverage, decision.triggers)
    expected_accounting = _coverage_accounting(coverage)
    expected_outcome = (
        AdaptiveSamplingDecisionOutcome.ADDITIONAL_TARGETS_SCHEDULED
        if expected_targets
        else AdaptiveSamplingDecisionOutcome.BASE_COVERAGE_ALREADY_CONTAINS_TARGETS
    )
    if decision.outcome is AdaptiveSamplingDecisionOutcome.UPGRADE_BUDGET_EXHAUSTED:
        raise ValueError("budget-exhausted decision has a resolvable canonical trigger plan")
    if decision.incremental_targets != expected_targets:
        raise ValueError("incremental targets do not match the frozen policy and triggers")
    if decision.coverage_accounting != expected_accounting:
        raise ValueError("coverage accounting does not match the frozen policy and triggers")
    if decision.outcome is not expected_outcome:
        raise ValueError("decision outcome does not match the frozen policy and triggers")


def _trigger_sort_key(
    trigger: AdaptiveTriggerProvenance,
) -> tuple[str, int, str, str, str, str]:
    return (
        trigger.camera_id.value,
        trigger.trigger_timestamp_ns,
        trigger.reason.value,
        trigger.evidence_kind.value,
        trigger.evidence_sha256,
        trigger.evidence_locator,
    )


def _target_provenance_sort_key(
    provenance: AdaptiveTargetTriggerProvenance,
) -> tuple[str, int, str, str, str, str, str, int, bool]:
    trigger = provenance.trigger
    return (
        trigger.camera_id.value,
        trigger.trigger_timestamp_ns,
        trigger.reason.value,
        trigger.evidence_kind.value,
        trigger.evidence_sha256,
        trigger.evidence_locator,
        provenance.role.value,
        provenance.context_offset_ns,
        provenance.context_clipped,
    )


def _canonical_key(value: StrictModel) -> bytes:
    return canonical_json_bytes(value)


def _require_model[T](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    return value


__all__ = [
    "ADAPTIVE_LATE_FEEDBACK_AUDIT_PROJECTION_VERSION",
    "ADAPTIVE_LATE_FEEDBACK_AUDIT_SCHEMA_VERSION",
    "ADAPTIVE_SAMPLING_DECISION_PROJECTION_VERSION",
    "ADAPTIVE_SAMPLING_DECISION_SCHEMA_VERSION",
    "ADAPTIVE_SAMPLING_DECISION_SCOPE_PROJECTION_VERSION",
    "AcceptedAdaptiveEvidenceBinding",
    "AdaptiveCoverageAccounting",
    "AdaptiveDecisionBaseBinding",
    "AdaptiveDecisionSourceBinding",
    "AdaptiveIncrementalTarget",
    "AdaptiveLateFeedbackAudit",
    "AdaptiveNoAdditionalWorkProof",
    "AdaptiveNoAdditionalWorkProofKind",
    "AdaptiveSamplingDecision",
    "AdaptiveSamplingDecisionOutcome",
    "AdaptiveTargetTriggerProvenance",
    "AdaptiveTriggerEvidenceKind",
    "AdaptiveTriggerProvenance",
    "accepted_selection_evidence_sha256",
    "adaptive_late_feedback_audit_projection",
    "adaptive_sampling_decision_projection",
    "adaptive_sampling_decision_scope_projection",
    "build_adaptive_late_feedback_audit",
    "build_adaptive_sampling_decision",
]
