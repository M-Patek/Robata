"""Canonical bridge for the local 21-class product-QA cascade.

This bridge consumes already-retained source-quality, QA, and event facts.  It
does not create a second inference path or change canonical identities.  The
result is deliberately local-only and is recomputable from the canonical run
closure plus its retained quality context.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import StringConstraints, model_validator

from robata.application.canonical.media_quality import (
    FrameQualityObservation,
    LocalMediaQualityReport,
    LocalQualityFlag,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.pipeline import CameraQAStatus
from robata.contracts.qa import (
    ProductQAConfidenceKind,
    ProductQAEvidenceScope,
    ProductQAIssue,
    ProductQAIssueEvidence,
    ProductQAScopeKind,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryRefinementOutcome,
    BoundaryRefinementResult,
)
from robata.event_pipeline.candidate import CandidateReductionResult
from robata.event_pipeline.evidence import ActionEvidenceResult
from robata.inference.enrichment import ProviderObservation
from robata.qa_pipeline.coarse import CameraCoarseResult, CoarseQAResult
from robata.qa_pipeline.completion import QACompletionResult, QACompletionStatus
from robata.qa_pipeline.dense import CameraDenseResult
from robata.qa_pipeline.product import (
    LOCAL_PRODUCT_QA_CASCADE_POLICY_VERSION,
    ProductQACascadeProjector,
    ProductQACascadeResult,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class CanonicalProductQAContext(StrictModel):
    """Local visual/structural facts supplied to the canonical QA reduction."""

    media_quality_report_semantic_sha256: Sha256Digest | None = None
    observed_evidence: tuple[ProductQAIssueEvidence, ...] = ()
    incomplete_reason_codes: tuple[NonEmptyString, ...] = ()
    abstained_reason_codes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if self.incomplete_reason_codes != tuple(sorted(set(self.incomplete_reason_codes))):
            raise ValueError("product QA incomplete reason codes must be unique and sorted")
        if self.abstained_reason_codes != tuple(sorted(set(self.abstained_reason_codes))):
            raise ValueError("product QA abstained reason codes must be unique and sorted")
        if set(self.incomplete_reason_codes) & set(self.abstained_reason_codes):
            raise ValueError("one product QA reason cannot be both incomplete and abstained")
        return self


def product_qa_context_from_media_quality_report(
    report: LocalMediaQualityReport,
) -> CanonicalProductQAContext:
    """Convert P2 sentinel/timeline facts into retained P6 product-QA inputs.

    The conversion intentionally maps only direct observations.  A proxy
    freeze or low-edge observation remains policy-derived evidence rather than
    a model-quality claim.  Timestamp/sequence/skew gaps make unobserved
    classes explicitly incomplete instead of silently producing a pass.
    """

    if not isinstance(report, LocalMediaQualityReport):
        raise TypeError("report must be a LocalMediaQualityReport")

    evidence: list[ProductQAIssueEvidence] = []
    incomplete: list[str] = []
    for ledger in report.camera_ledgers:
        if ledger.timing_count == 0:
            incomplete.append(f"SOURCE_CAMERA_NO_TIMING:{ledger.camera_id.value}")
        for observation in ledger.decoded_observations:
            for flag in observation.flags:
                mapped = _media_quality_evidence(report, observation, flag)
                if mapped is not None:
                    evidence.append(mapped)
        for cadence_gap in ledger.cadence_gaps:
            incomplete.append(
                f"SOURCE_CADENCE_GAP:{ledger.camera_id.value}:{cadence_gap.timestamp_ns}"
            )
        for sequence_gap in ledger.sequence_gaps:
            incomplete.append(
                f"SOURCE_SEQUENCE_GAP:{ledger.camera_id.value}:{sequence_gap.timestamp_ns}"
            )

    if (
        report.requested_interval.start_ns != 0
        or report.requested_interval.end_ns != report.recording_duration_ns
    ):
        incomplete.append("SOURCE_INTERVAL_NOT_FULL_RECORDING")
    for sample in report.cross_camera_skew.samples:
        if sample.skew_ns > report.cross_camera_skew.threshold_ns:
            incomplete.append(f"SOURCE_CROSS_CAMERA_SKEW:{sample.target_ns}")
    if report.cross_camera_skew.incomplete_target_count:
        incomplete.append(
            "SOURCE_CROSS_CAMERA_ALIGNMENT_INCOMPLETE:"
            f"{report.cross_camera_skew.incomplete_target_count}"
        )

    return CanonicalProductQAContext(
        media_quality_report_semantic_sha256=report.semantic_sha256,
        observed_evidence=tuple(evidence),
        incomplete_reason_codes=tuple(sorted(set(incomplete))),
    )


class CanonicalProductQAProjector:
    """Combine source, QA, and event evidence into one complete local result."""

    def __init__(
        self,
        policy_version: str = LOCAL_PRODUCT_QA_CASCADE_POLICY_VERSION,
    ) -> None:
        self._projector = ProductQACascadeProjector(policy_version)

    @property
    def policy_version(self) -> str:
        return self._projector.policy_version

    def project(
        self,
        *,
        recording_id: str,
        recording_duration_ns: int,
        coarse_result: CoarseQAResult,
        qa_completion_result: QACompletionResult,
        candidate_reduction_result: CandidateReductionResult | None = None,
        action_evidence_results: Sequence[ActionEvidenceResult] = (),
        boundary_results: Sequence[BoundaryRefinementResult] = (),
        context: CanonicalProductQAContext | None = None,
        pipeline_incomplete: bool = False,
        pipeline_abstained: bool = False,
    ) -> ProductQACascadeResult:
        """Reduce all available facts while preserving exact upstream references."""

        # Check argument types before touching any nested fields so callers
        # get a deterministic programming-error response even when another
        # envelope is malformed.  Then re-validate every retained envelope at
        # this reduction boundary.  A typed object can still have been
        # assembled with model_construct or deserialized from an untrusted
        # local cache; consuming its fields directly would let a forged claim
        # or lineage reference reach the complete product projection.
        if not isinstance(coarse_result, CoarseQAResult):
            raise TypeError("coarse_result must be a CoarseQAResult")
        if not isinstance(qa_completion_result, QACompletionResult):
            raise TypeError("qa_completion_result must be a QACompletionResult")
        if context is not None and not isinstance(context, CanonicalProductQAContext):
            raise TypeError("context must be CanonicalProductQAContext or None")
        if candidate_reduction_result is not None and not isinstance(
            candidate_reduction_result,
            CandidateReductionResult,
        ):
            raise TypeError("candidate_reduction_result must be a CandidateReductionResult or None")
        for action_evidence_result in action_evidence_results:
            if not isinstance(action_evidence_result, ActionEvidenceResult):
                raise TypeError("action_evidence_results must contain ActionEvidenceResult")
        for boundary_result in boundary_results:
            if not isinstance(boundary_result, BoundaryRefinementResult):
                raise TypeError("boundary_results must contain BoundaryRefinementResult")
        if not isinstance(pipeline_incomplete, bool) or not isinstance(pipeline_abstained, bool):
            raise TypeError("pipeline terminal flags must be bool")

        coarse_result = _validated_instance(coarse_result, CoarseQAResult, "coarse_result")
        qa_completion_result = _validated_instance(
            qa_completion_result,
            QACompletionResult,
            "qa_completion_result",
        )
        if context is None:
            context = CanonicalProductQAContext()
        context = _validated_instance(context, CanonicalProductQAContext, "context")
        if candidate_reduction_result is not None:
            candidate_reduction_result = _validated_instance(
                candidate_reduction_result,
                CandidateReductionResult,
                "candidate_reduction_result",
            )
        checked_action_results = tuple(
            _validated_instance(item, ActionEvidenceResult, "action_evidence_results")
            for item in action_evidence_results
        )
        checked_boundary_results = tuple(
            _validated_instance(item, BoundaryRefinementResult, "boundary_results")
            for item in boundary_results
        )

        observed = list(context.observed_evidence)
        incomplete_reasons = list(context.incomplete_reason_codes)
        abstained_reasons = list(context.abstained_reason_codes)

        for coarse_camera_result in coarse_result.package_camera_results:
            mapped, reason = _qa_result_evidence(
                coarse_camera_result,
                stage="QA_COARSE",
            )
            if mapped is not None:
                observed.append(mapped)
            if reason is not None:
                incomplete_reasons.append(reason)
        dense_result = qa_completion_result.dense_result
        if dense_result is not None:
            for unit in dense_result.units:
                for dense_camera_result in unit.evidence.package_camera_results:
                    mapped, reason = _qa_result_evidence(
                        dense_camera_result,
                        stage="QA_DENSE",
                    )
                    if mapped is not None:
                        observed.append(mapped)
                    if reason is not None:
                        incomplete_reasons.append(reason)

        if qa_completion_result.status is not QACompletionStatus.QA_COMPLETE:
            incomplete_reasons.append(f"QA_COMPLETION_{qa_completion_result.status.value}")

        if candidate_reduction_result is not None:
            observed.extend(_candidate_evidence(candidate_reduction_result))
        for action_result in checked_action_results:
            observed.extend(_action_evidence(action_result))
        for boundary in checked_boundary_results:
            mapped = _boundary_evidence(boundary)
            if mapped is not None:
                observed.append(mapped)

        if pipeline_incomplete:
            incomplete_reasons.append("CANONICAL_PIPELINE_INCOMPLETE")
        if pipeline_abstained:
            abstained_reasons.append("CANONICAL_PIPELINE_ABSTAINED")

        return self._projector.project(
            recording_id=recording_id,
            recording_duration_ns=recording_duration_ns,
            observed_evidence=observed,
            incomplete_reason_codes=tuple(sorted(set(incomplete_reasons))),
            abstained_reason_codes=tuple(sorted(set(abstained_reasons))),
        )


def _validated_instance[StrictModelT: StrictModel](
    value: StrictModelT,
    expected_type: type[StrictModelT],
    label: str,
) -> StrictModelT:
    """Return a strict, detached copy of one canonical upstream envelope."""

    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    try:
        return expected_type.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} failed immutable contract validation") from exc


def _media_quality_evidence(
    report: LocalMediaQualityReport,
    observation: FrameQualityObservation,
    flag: LocalQualityFlag,
) -> ProductQAIssueEvidence | None:
    """Map a direct sentinel flag without turning a proxy into model truth."""

    camera_id = observation.camera_id
    timestamp_ns = observation.aligned_timestamp_ns
    if not 0 <= timestamp_ns < report.recording_duration_ns:
        raise ValueError("media quality observation lies outside recording bounds")

    issue: ProductQAIssue | None
    confidence: float
    confidence_kind: ProductQAConfidenceKind
    note: str | None = None
    if flag is LocalQualityFlag.OBSERVED_BLACK_LUMA:
        issue = ProductQAIssue.BLACK_SCREEN
        confidence = observation.black_fraction_ppm / 1_000_000
        confidence_kind = ProductQAConfidenceKind.DETECTOR_REPORTED
    elif flag is LocalQualityFlag.OBSERVED_OVEREXPOSED_LUMA:
        issue = ProductQAIssue.TOO_DARK_OR_OVEREXPOSED
        confidence = observation.overexposed_fraction_ppm / 1_000_000
        confidence_kind = ProductQAConfidenceKind.DETECTOR_REPORTED
    elif flag is LocalQualityFlag.PROXY_LOW_EDGE_ENERGY:
        issue = ProductQAIssue.BLURRY_LENS
        confidence = 0.5
        confidence_kind = ProductQAConfidenceKind.POLICY_DERIVED
        note = "low-edge-energy proxy; original sentinel frame retained"
    elif flag is LocalQualityFlag.PROXY_FROZEN_CONTENT:
        issue = ProductQAIssue.CAMERA_STATIONARY_OVER_5S
        confidence = 0.5
        confidence_kind = ProductQAConfidenceKind.POLICY_DERIVED
        note = "frozen-content proxy; original sentinel frame retained"
    else:
        return None

    return ProductQAIssueEvidence(
        issue=issue,
        scope=_camera_interval_scope(camera_id),
        interval=NanosecondInterval(start_ns=timestamp_ns, end_ns=timestamp_ns + 1),
        confidence=confidence,
        confidence_kind=confidence_kind,
        evidence_refs=(
            f"media-quality:{report.semantic_sha256}:{camera_id.value}:{timestamp_ns}:{flag.value}",
        ),
        note=note,
    )


def _qa_result_evidence(
    result: CameraCoarseResult | CameraDenseResult,
    *,
    stage: Literal["QA_COARSE", "QA_DENSE"],
) -> tuple[ProductQAIssueEvidence | None, str | None]:
    """Map a typed QA label and leave unknown camera evidence fail-closed."""

    if not isinstance(result, (CameraCoarseResult, CameraDenseResult)):
        raise TypeError("QA result must retain a provider claim")
    claim = result.claim
    camera_id = result.camera_id
    status = result.local_status
    interval = claim.interval
    if interval is None:
        raise ValueError("canonical QA result has no exact interval")
    if status is CameraQAStatus.UNKNOWN:
        return None, f"{stage}_UNKNOWN:{claim.claim_id}"

    issue = _product_issue(claim.label)
    note: str | None = None
    if issue is None:
        if status in {CameraQAStatus.DEGRADED, CameraQAStatus.UNUSABLE}:
            issue = ProductQAIssue.OTHER
            label = "no provider label" if claim.label is None else claim.label
            note = f"unmapped {stage} observation {status.value}: {label}"
        else:
            return None, None

    confidence = claim.model_reported_confidence
    return (
        ProductQAIssueEvidence(
            issue=issue,
            scope=_camera_interval_scope(camera_id),
            interval=NanosecondInterval(start_ns=interval.start_ns, end_ns=interval.end_ns),
            confidence=0.5 if confidence is None else confidence.value,
            confidence_kind=(
                ProductQAConfidenceKind.POLICY_DERIVED
                if confidence is None
                else ProductQAConfidenceKind.MODEL_REPORTED
            ),
            evidence_refs=(
                f"provider-claim:{claim.claim_id}",
                f"{stage.lower()}-source:{result.source_output.enrichment_logical_key}",
            ),
            note=note,
        ),
        None,
    )


def _candidate_evidence(
    result: CandidateReductionResult,
) -> tuple[ProductQAIssueEvidence, ...]:
    evidence: list[ProductQAIssueEvidence] = []
    for candidate in result.candidates:
        issue = _product_issue(candidate.label)
        if issue is None:
            continue
        evidence.append(
            ProductQAIssueEvidence(
                issue=issue,
                scope=ProductQAEvidenceScope(
                    kind=ProductQAScopeKind.TASK_INTERVAL,
                    subject_refs=(f"candidate:{candidate.candidate_logical_key}",),
                ),
                interval=candidate.effective_interval,
                confidence=0.5,
                confidence_kind=ProductQAConfidenceKind.POLICY_DERIVED,
                evidence_refs=(candidate.candidate_logical_key,),
                note="event candidate retained with dense context",
            )
        )
    return tuple(evidence)


def _action_evidence(result: ActionEvidenceResult) -> tuple[ProductQAIssueEvidence, ...]:
    evidence: list[ProductQAIssueEvidence] = []
    for camera_id in CAMERA_IDS:
        for observation in result.camera_evidence[camera_id].observations:
            if (
                observation.observation
                not in {
                    ProviderObservation.SUPPORTING,
                    ProviderObservation.PARTIAL,
                }
                or observation.interval is None
            ):
                continue
            issue = _product_issue(observation.label)
            if issue is None:
                continue
            score = observation.model_reported_score
            evidence.append(
                ProductQAIssueEvidence(
                    issue=issue,
                    scope=_camera_interval_scope(camera_id),
                    interval=observation.interval,
                    confidence=0.5 if score is None else score,
                    confidence_kind=(
                        ProductQAConfidenceKind.POLICY_DERIVED
                        if score is None
                        else ProductQAConfidenceKind.MODEL_REPORTED
                    ),
                    evidence_refs=(
                        observation.source_action_observation_logical_key,
                        result.logical_key,
                    ),
                    note="candidate-centered action evidence",
                )
            )
    return tuple(evidence)


def _boundary_evidence(
    result: BoundaryRefinementResult,
) -> ProductQAIssueEvidence | None:
    if result.outcome is not BoundaryRefinementOutcome.REFINED:
        return None
    issue = _product_issue(result.action_label)
    interval = result.refined_interval
    if issue is None or interval is None:
        return None
    return ProductQAIssueEvidence(
        issue=issue,
        scope=ProductQAEvidenceScope(
            kind=ProductQAScopeKind.TASK_INTERVAL,
            subject_refs=(f"boundary:{result.logical_key}",),
        ),
        interval=interval,
        confidence=0.5,
        confidence_kind=ProductQAConfidenceKind.POLICY_DERIVED,
        evidence_refs=(result.logical_key,),
        note="onset/offset boundary refinement",
    )


def _camera_interval_scope(camera_id: CameraId) -> ProductQAEvidenceScope:
    return ProductQAEvidenceScope(
        kind=ProductQAScopeKind.CAMERA_INTERVAL,
        subject_refs=(f"camera:{camera_id.value}",),
        camera_id=camera_id,
    )


def _product_issue(value: object) -> ProductQAIssue | None:
    if not isinstance(value, str):
        return None
    try:
        return ProductQAIssue(value)
    except ValueError:
        return None


__all__ = [
    "CanonicalProductQAContext",
    "CanonicalProductQAProjector",
    "product_qa_context_from_media_quality_report",
]
