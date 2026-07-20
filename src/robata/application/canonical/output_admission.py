"""Local output admission and event-hypothesis projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.models import (
    CanonicalOfflineConfigurationError,
    NonEmptyString,
    NonNegativeInt,
    _strict_context,
)
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
    _canonical_output_decision_projection_values,
    _fusion_claim_reduction_digest,
    _stable_uuid,
    canonical_output_decision_projection,
)
from robata.application.canonical.reduction import CanonicalFusionReduction
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    AdmissionProof,
    OutputAdmissionProof,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    ProductionAdmittedHypothesisFact,
    ProductionOutputAdmissionPolicyRef,
    validate_evidence_eligibility,
)
from robata.inference.enrichment import (
    EnrichedProviderClaim,
    OrchestratorEnrichedOutput,
    ProviderClaimKind,
)
from robata.inference.models import VisionTask


class CanonicalOutputAdmissionDecision(StrictModel):
    """Local output-level decision; it is not a registered durable schema yet."""

    schema_version: Literal["2.0"]
    decision_id: OpaqueUuid
    decision: Literal["ADMITTED", "NO_EVENTS", "ABSTAINED"]
    evidence_class: AdmissionEvidenceClass
    production_eligible: bool
    semantic_sha256: Sha256Digest
    recording_identity: Sha256Digest
    source_enrichments: tuple[PlatformEnrichedOutputReference, ...]
    fusion_reduction_logical_key: NodeLogicalKey
    fusion_reduction_semantic_sha256: Sha256Digest
    policy_version: SchemaVersion
    policy_sha256: Sha256Digest
    admitted_claim_ordinals: tuple[NonNegativeInt, ...]
    reason_code: NonEmptyString
    production_output_admission: OutputAdmissionProof | None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        validate_evidence_eligibility(self.evidence_class, self.production_eligible)
        if not self.source_enrichments:
            raise ValueError("output decision requires enriched output lineage")
        keys = tuple(item.enrichment_logical_key for item in self.source_enrichments)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("output decision enrichments must be unique and canonical")
        if any(
            item.recording_identity != self.recording_identity for item in self.source_enrichments
        ):
            raise ValueError("output decision crosses recording scope")
        expected_fusion_key = f"fusion-reduction:{self.fusion_reduction_semantic_sha256}"
        if self.fusion_reduction_logical_key != expected_fusion_key:
            raise ValueError(
                "output decision fusion reduction logical key and semantic digest differ"
            )
        if self.decision in {"NO_EVENTS", "ABSTAINED"}:
            if self.admitted_claim_ordinals or self.production_output_admission is not None:
                raise ValueError(
                    "non-admitted output decisions cannot carry claims or an output proof"
                )
        elif self.production_output_admission is None:
            raise ValueError("ADMITTED requires an output admission proof")
        else:
            proof = self.production_output_admission
            if (
                proof.evidence_class is not self.evidence_class
                or proof.production_eligible is not self.production_eligible
            ):
                raise ValueError("output decision evidence metadata differs from its proof")
            if proof.source_enrichments != self.source_enrichments:
                raise ValueError("output decision proof does not bind all source enrichments")
            if (
                proof.output_admission_policy_version != self.policy_version
                or proof.output_admission_policy_sha256 != self.policy_sha256
            ):
                raise ValueError("output decision proof policy differs from decision policy")
            proof_ordinals = tuple(
                sorted(item.fusion_output_ordinal for item in proof.admitted_hypothesis_facts)
            )
            if proof_ordinals != self.admitted_claim_ordinals:
                raise ValueError("output decision ordinals differ from admitted proof facts")
        expected = semantic_sha256(canonical_output_decision_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("output decision semantic_sha256 is inconsistent")
        if self.decision_id != _stable_uuid(CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE, expected):
            raise ValueError("output decision ID is inconsistent")
        return self

    @property
    def source_enrichment(self) -> PlatformEnrichedOutputReference:
        """Compatibility accessor for callers restricted to one call part."""

        if len(self.source_enrichments) != 1:
            raise ValueError("multi-part decisions have more than one source enrichment")
        return self.source_enrichments[0]


class FusionEventHypothesisProjector:
    """Project one exact all-part fusion reduction into platform hypotheses."""

    def __init__(
        self, *, policy: ProductionOutputAdmissionPolicyRef, projector_version: str
    ) -> None:
        if not isinstance(policy, ProductionOutputAdmissionPolicyRef):
            raise TypeError("policy must be a ProductionOutputAdmissionPolicyRef")
        if not isinstance(projector_version, str) or not projector_version:
            raise ValueError("projector_version must be nonempty")
        self._policy = policy
        self._projector_version = projector_version

    def project(
        self,
        *,
        context: AdmittedRecordingContextV2,
        fusion_reduction: CanonicalFusionReduction,
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
        interval: NanosecondInterval,
    ) -> tuple[CanonicalOutputAdmissionDecision, tuple[PlatformEnrichedEventHypothesis, ...]]:
        context = _strict_context(context)
        try:
            reduction = CanonicalFusionReduction.model_validate(
                fusion_reduction.model_dump(mode="python"), strict=True
            )
            outputs = tuple(
                OrchestratorEnrichedOutput.model_validate(
                    item.model_dump(mode="python"), strict=True
                )
                for item in enriched_outputs
            )
        except ValueError as exc:
            raise CanonicalOfflineConfigurationError(
                "fusion reduction lineage failed strict validation"
            ) from exc
        if not outputs:
            raise ValueError("fusion projection requires enriched part outputs")
        output_by_key = {item.enrichment_logical_key: item for item in outputs}
        if len(output_by_key) != len(outputs):
            raise ValueError("fusion projection received duplicate enrichments")
        expected_refs = tuple(
            sorted(
                (item.enrichment for item in reduction.parts),
                key=lambda item: item.enrichment_logical_key,
            )
        )
        actual_refs = tuple(
            sorted(
                (PlatformEnrichedOutputReference.from_output(item) for item in outputs),
                key=lambda item: item.enrichment_logical_key,
            )
        )
        if actual_refs != expected_refs:
            raise ValueError("fusion reduction does not bind the exact enriched output set")
        for part in reduction.parts:
            output = output_by_key[part.enrichment.enrichment_logical_key]
            if output.task is not VisionTask.FUSION_ADJUDICATION:
                raise ValueError("event projection requires FUSION_ADJUDICATION output")
            authority = output.authority
            if (
                authority.recording_identity != context.recording_identity
                or authority.mcap_id != context.ready_manifest.mcap_id
                or authority.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
                or authority.alignment_id != context.alignment_manifest.alignment_id
                or authority.inference_id != part.inference_id
                or output.selected_attempt.output_sha256 != part.selected_attempt_output_sha256
                or output.abstained != part.abstained
            ):
                raise ValueError("fusion part authority does not match admission lineage")
        for reduced_claim in reduction.claims:
            for source in reduced_claim.sources:
                source_output = output_by_key.get(source.enrichment_logical_key)
                if source_output is None or source.source_claim_ordinal >= len(
                    source_output.claims
                ):
                    raise ValueError("fusion claim source is outside its enriched output")
                source_claim = source_output.claims[source.source_claim_ordinal]
                if (
                    source_claim.claim_id != source.source_claim_id
                    or _fusion_claim_reduction_digest(source_claim)
                    != reduced_claim.claim_semantic_sha256
                ):
                    raise ValueError("fusion claim source does not match the reduced claim")
        if not isinstance(interval, NanosecondInterval):
            raise TypeError("projection interval must be a NanosecondInterval")
        if interval.start_ns < 0 or interval.end_ns > context.ready_manifest.recording.duration_ns:
            raise ValueError("projection interval is outside the admitted recording")
        if reduction.outcome == "ALL_PARTS_ABSTAINED":
            decision = _output_decision(
                recording_identity=context.recording_identity,
                source_refs=actual_refs,
                fusion_reduction=reduction,
                policy=self._policy,
                decision="ABSTAINED",
                admitted_claim_ordinals=(),
                reason_code="ALL_REQUIRED_PROVIDER_PARTS_ABSTAINED",
                production_output_admission=None,
            )
            return decision, ()
        if reduction.outcome == "NO_SURVIVING_EVENTS":
            decision = _output_decision(
                recording_identity=context.recording_identity,
                source_refs=actual_refs,
                fusion_reduction=reduction,
                policy=self._policy,
                decision="NO_EVENTS",
                admitted_claim_ordinals=(),
                reason_code="FUSION_REDUCTION_EMPTY",
                production_output_admission=None,
            )
            return decision, ()

        fingerprints: set[str] = set()
        facts: list[ProductionAdmittedHypothesisFact] = []
        drafts: list[tuple[int, NanosecondInterval, str, str]] = []
        for reduced_claim in reduction.claims:
            claim = reduced_claim.representative
            if claim.kind is not ProviderClaimKind.FUSION_HYPOTHESIS:
                raise ValueError("fusion output contains a non-fusion claim")
            if claim.interval is None or not _contains_interval(interval, claim.interval):
                raise ValueError("fusion hypothesis interval is outside the root window")
            if not claim.evidence:
                raise ValueError("fusion hypothesis requires authoritative evidence")
            fingerprint = _fusion_event_fingerprint(
                recording_identity=context.recording_identity,
                claim=claim,
                projector_version=self._projector_version,
            )
            if fingerprint in fingerprints:
                raise ValueError("fusion reduction contains duplicate semantic fingerprints")
            fingerprints.add(fingerprint)
            fusion_digest = semantic_sha256(
                {
                    "semantic_fingerprint_sha256": fingerprint,
                    "fusion_reduction_semantic_sha256": reduction.semantic_sha256,
                    "projector_version": self._projector_version,
                }
            )
            effective_interval = NanosecondInterval(
                start_ns=claim.interval.start_ns,
                end_ns=claim.interval.end_ns,
            )
            fusion_logical_key = f"fusion:{fusion_digest}"
            facts.append(
                ProductionAdmittedHypothesisFact(
                    fusion_output_ordinal=reduced_claim.fusion_output_ordinal,
                    effective_interval=effective_interval,
                    semantic_fingerprint_sha256=fingerprint,
                    fusion_logical_key=fusion_logical_key,
                )
            )
            drafts.append(
                (
                    reduced_claim.fusion_output_ordinal,
                    effective_interval,
                    fingerprint,
                    fusion_logical_key,
                )
            )
        proof = OutputAdmissionProof.create_local_conformance(
            recording_identity=context.recording_identity,
            source_enrichments=actual_refs,
            admitted_hypothesis_facts=facts,
            policy=self._policy,
        )
        admission = AdmissionProof.from_local_conformance_context(context)
        hypotheses = tuple(
            PlatformEnrichedEventHypothesis.create(
                recording_identity=context.recording_identity,
                effective_interval=effective_interval,
                semantic_fingerprint_sha256=fingerprint,
                fusion_logical_key=fusion_logical_key,
                fusion_output_ordinal=claim_ordinal,
                source_enrichments=actual_refs,
                production_admission=admission,
                production_output_admission=proof,
            )
            for claim_ordinal, effective_interval, fingerprint, fusion_logical_key in drafts
        )
        decision = _output_decision(
            recording_identity=context.recording_identity,
            source_refs=actual_refs,
            fusion_reduction=reduction,
            policy=self._policy,
            decision="ADMITTED",
            admitted_claim_ordinals=tuple(item.fusion_output_ordinal for item in hypotheses),
            reason_code="FUSION_REDUCTION_VALIDATED",
            production_output_admission=proof,
        )
        return decision, hypotheses


def _fusion_event_fingerprint(
    *,
    recording_identity: str,
    claim: EnrichedProviderClaim,
    projector_version: str,
) -> str:
    ordered_evidence = tuple(
        sorted(
            claim.evidence,
            key=lambda item: (
                item.package_ordinal,
                item.camera_ordinal,
                item.frame_ordinal,
                item.aligned_timestamp_ns,
                item.source_artifact_sha256,
            ),
        )
    )
    if claim.interval is None:
        raise CanonicalOfflineConfigurationError(
            "fusion event fingerprint requires a reported interval"
        )
    return semantic_sha256(
        {
            "recording_identity": recording_identity,
            "start_ns": str(claim.interval.start_ns),
            "end_ns": str(claim.interval.end_ns),
            "label": claim.label,
            "observation": claim.observation.value,
            "conflict_codes": sorted(claim.conflict_codes),
            "evidence": [
                {
                    "package_semantic_content_sha256": (item.package_semantic_content_sha256),
                    "package_ordinal": item.package_ordinal,
                    "camera_ordinal": item.camera_ordinal,
                    "frame_ordinal": item.frame_ordinal,
                    "source_artifact_sha256": item.source_artifact_sha256,
                    "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
                }
                for item in ordered_evidence
            ],
            "projector_version": projector_version,
        }
    )


def _output_decision(
    *,
    recording_identity: str,
    source_refs: Sequence[PlatformEnrichedOutputReference],
    fusion_reduction: CanonicalFusionReduction,
    policy: ProductionOutputAdmissionPolicyRef,
    decision: Literal["ADMITTED", "NO_EVENTS", "ABSTAINED"],
    admitted_claim_ordinals: tuple[int, ...],
    reason_code: str,
    production_output_admission: OutputAdmissionProof | None,
) -> CanonicalOutputAdmissionDecision:
    refs = tuple(sorted(source_refs, key=lambda item: item.enrichment_logical_key))
    projection = _canonical_output_decision_projection_values(
        decision=decision,
        evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        production_eligible=False,
        recording_identity=recording_identity,
        source_enrichments=refs,
        fusion_reduction_logical_key=fusion_reduction.reduction_logical_key,
        fusion_reduction_semantic_sha256=fusion_reduction.semantic_sha256,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admitted_claim_ordinals=admitted_claim_ordinals,
        reason_code=reason_code,
        production_output_admission=production_output_admission,
    )
    digest = semantic_sha256(projection)
    return CanonicalOutputAdmissionDecision(
        schema_version="2.0",
        decision_id=_stable_uuid(CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE, digest),
        decision=decision,
        evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        production_eligible=False,
        semantic_sha256=digest,
        recording_identity=recording_identity,
        source_enrichments=refs,
        fusion_reduction_logical_key=fusion_reduction.reduction_logical_key,
        fusion_reduction_semantic_sha256=fusion_reduction.semantic_sha256,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admitted_claim_ordinals=admitted_claim_ordinals,
        reason_code=reason_code,
        production_output_admission=production_output_admission,
    )


def _contains_interval(outer: NanosecondInterval, inner: object) -> bool:
    start = getattr(inner, "start_ns", None)
    end = getattr(inner, "end_ns", None)
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and outer.start_ns <= start < end <= outer.end_ns
    )


__all__ = ["CanonicalOutputAdmissionDecision", "FusionEventHypothesisProjector"]
