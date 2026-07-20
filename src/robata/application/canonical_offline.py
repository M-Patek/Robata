"""Canonical, transport-free vertical slice for the registered V2 path.

The service in this module is deliberately local and non-promotional.  It
connects the registered admission evidence through temporal materialization,
provider-specific planning, a single-part inference barrier, strict raw
response parsing, enrichment, and recording-scoped event identity assignment.
No network-capable adapter is accepted by the coordinator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp
from robata.contracts.mainline import SamplingPurpose
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.contracts.temporal import PackageLineage, TemporalPackageSet
from robata.event_pipeline.identity_registry import (
    EventIdentityBatchResult,
    EventIdentityRegistryError,
    EventIdentityRegistryService,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    ProductionAdmissionProof,
    ProductionAdmittedHypothesisFact,
    ProductionOutputAdmissionPolicyRef,
    ProductionOutputAdmissionProof,
)
from robata.inference.adapter import JsonSchemaRef, PackageInput
from robata.inference.call_barrier import (
    InferenceCallBarrierCoordinator,
    InferenceCallBarrierError,
    InferenceCallPartCompletion,
    InferenceCallReduction,
    InMemoryInferenceCallBarrierStorage,
)
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    PROVIDER_CLAIM_SCHEMA_ID,
    EnrichmentAuthorityContext,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderClaimEnricher,
    ProviderClaimEnrichmentError,
    ProviderClaimKind,
    ProviderReferenceCatalog,
    ProviderReferenceCatalogEntry,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
    enrichment_logical_digest,
)
from robata.inference.input_plan import (
    InferenceInputPlan,
    InputPlanTarget,
    PromptOutputContract,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    InferenceStatus,
    ModelCapabilities,
    ModelInference,
    Retryability,
    VisionTask,
)
from robata.inference.offline_fixture import (
    OfflineFixtureVisionAdapter,
    RawProviderBytesStoreError,
    StrictProviderClaimParseError,
)
from robata.inference.orchestrator import (
    InferenceOrchestrationError,
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from robata.inference.preparation import (
    InputPlanPreparer,
    InputPreparationError,
    RenderedItemFactory,
    applicable_limits_from_capabilities,
)
from robata.queue.barrier import BarrierCoordinator, InMemoryBarrierStorage
from robata.sampling.dense import IntervalPart
from robata.sampling.materializer import (
    CanonicalSixCameraFrameIndex,
    FrameArtifactResolver,
    MaterializedTemporalPackage,
    OfflineTemporalPackageMaterializer,
    PackageMaterializationError,
)
from robata.sampling.package_set import PackageSetBuilder, sampling_plan_digest

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class CanonicalOfflineConfigurationError(ValueError):
    """The configured local vertical slice cannot satisfy its contracts."""


class CanonicalOfflineRunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    NO_EVENTS = "NO_EVENTS"
    MATERIALIZATION_FAILED = "MATERIALIZATION_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    BLOCKED_MULTI_PART = "BLOCKED_MULTI_PART"
    IDENTITY_FAILED = "IDENTITY_FAILED"
    CONFIGURATION_FAILED = "CONFIGURATION_FAILED"


class CanonicalOfflineStage(StrEnum):
    ADMISSION = "ADMISSION"
    WINDOW = "WINDOW"
    MATERIALIZATION = "MATERIALIZATION"
    PREPARATION = "PREPARATION"
    INFERENCE = "INFERENCE"
    PARSING = "PARSING"
    ENRICHMENT = "ENRICHMENT"
    OUTPUT_ADMISSION = "OUTPUT_ADMISSION"
    IDENTITY = "IDENTITY"


class CanonicalOfflineError(StrictModel):
    schema_version: Literal["1.0"]
    stage: CanonicalOfflineStage
    code: NonEmptyString
    detail: NonEmptyString


def canonical_root_window_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    """Project root-window semantics without row IDs, associations, or clocks."""

    requested = values["requested_interval"]
    effective = values["interval"]
    if not isinstance(requested, NanosecondInterval) or not isinstance(
        effective, NanosecondInterval
    ):
        raise TypeError("window intervals must be NanosecondInterval values")
    purpose = values["purpose"]
    if not isinstance(purpose, SamplingPurpose):
        raise TypeError("window purpose must be a SamplingPurpose")
    return {
        "source_content_sha256": values["source_content_sha256"],
        "camera_mapping_semantic_sha256": values["camera_mapping_semantic_sha256"],
        "alignment_semantic_sha256": values["alignment_semantic_sha256"],
        "requested_interval": {
            "start_ns": str(requested.start_ns),
            "end_ns": str(requested.end_ns),
        },
        "interval": {
            "start_ns": str(effective.start_ns),
            "end_ns": str(effective.end_ns),
        },
        "purpose": purpose.value,
        "window_policy_version": values["window_policy_version"],
        "source_subject_type": values["source_subject_type"],
        "source_subject_logical_key": values["source_subject_logical_key"],
        "parent_window_logical_key": values["parent_window_logical_key"],
        "source_lineage_sha256": values["source_lineage_sha256"],
        "refinement_role": values["refinement_role"],
        "generation": values["generation"],
    }


class CanonicalRootWindow(StrictModel):
    """Stable root sampling window derived solely from V2 admission evidence."""

    schema_version: Literal["1.0"]
    window_id: OpaqueUuid
    window_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    recording_duration_ns: PositiveInt
    reference_timebase: Literal["recording_relative_ns"]
    requested_interval: NanosecondInterval
    interval: NanosecondInterval
    purpose: SamplingPurpose
    window_policy_version: SchemaVersion
    source_subject_type: Literal["RECORDING"]
    source_subject_logical_key: NodeLogicalKey
    parent_window_logical_key: None = None
    source_lineage_sha256: Sha256Digest
    refinement_role: Literal["ROOT"]
    generation: Literal[0]
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.interval.start_ns < self.requested_interval.start_ns or self.interval.end_ns > (
            self.requested_interval.end_ns
        ):
            raise ValueError("effective root interval must be contained by requested interval")
        if self.interval.start_ns < 0 or self.interval.end_ns > self.recording_duration_ns:
            raise ValueError("effective root interval must be inside recording duration")
        if self.source_subject_logical_key != f"recording:{self.recording_identity}":
            raise ValueError("root window source subject is inconsistent")
        projection = canonical_root_window_projection_values(
            {
                "source_content_sha256": self.source_content_sha256,
                "camera_mapping_semantic_sha256": self.camera_mapping_semantic_sha256,
                "alignment_semantic_sha256": self.alignment_semantic_sha256,
                "requested_interval": self.requested_interval,
                "interval": self.interval,
                "purpose": self.purpose,
                "window_policy_version": self.window_policy_version,
                "source_subject_type": self.source_subject_type,
                "source_subject_logical_key": self.source_subject_logical_key,
                "parent_window_logical_key": self.parent_window_logical_key,
                "source_lineage_sha256": self.source_lineage_sha256,
                "refinement_role": self.refinement_role,
                "generation": self.generation,
            }
        )
        expected = semantic_sha256(projection)
        if self.semantic_sha256 != expected:
            raise ValueError("root window semantic_sha256 is inconsistent")
        if self.window_logical_key != f"temporal-window:{expected}":
            raise ValueError("root window logical key is inconsistent")
        expected_id = _stable_uuid("canonical-root-window", expected)
        if self.window_id != expected_id:
            raise ValueError("root window ID is inconsistent")
        return self

    @classmethod
    def from_context(
        cls,
        *,
        context: AdmittedRecordingContextV2,
        requested_interval: NanosecondInterval,
        purpose: SamplingPurpose,
        window_policy_version: str,
        created_at: str,
    ) -> Self:
        if not isinstance(context, AdmittedRecordingContextV2):
            raise CanonicalOfflineConfigurationError("root window requires V2 admission context")
        try:
            context = AdmittedRecordingContextV2.model_validate(
                context.model_dump(mode="python"), strict=True
            )
        except ValueError as exc:
            raise CanonicalOfflineConfigurationError(
                "V2 admission context failed validation"
            ) from exc
        if purpose is not SamplingPurpose.ACTION_DENSE:
            raise CanonicalOfflineConfigurationError(
                "canonical offline vertical slice requires ACTION_DENSE root purpose"
            )
        if not isinstance(requested_interval, NanosecondInterval):
            raise TypeError("requested_interval must be a NanosecondInterval")
        duration = context.ready_manifest.recording.duration_ns
        effective_start = max(0, requested_interval.start_ns)
        effective_end = min(duration, requested_interval.end_ns)
        if effective_start >= effective_end:
            raise CanonicalOfflineConfigurationError(
                "requested interval does not overlap the admitted recording"
            )
        effective = NanosecondInterval(start_ns=effective_start, end_ns=effective_end)
        values: dict[str, object] = {
            "source_content_sha256": context.source_content_sha256,
            "camera_mapping_semantic_sha256": context.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": context.alignment_semantic_sha256,
            "recording_duration_ns": duration,
            "requested_interval": requested_interval,
            "interval": effective,
            "purpose": purpose,
            "window_policy_version": window_policy_version,
            "source_subject_type": "RECORDING",
            "source_subject_logical_key": f"recording:{context.recording_identity}",
            "parent_window_logical_key": None,
            "source_lineage_sha256": context.semantic_sha256,
            "refinement_role": "ROOT",
            "generation": 0,
        }
        digest = semantic_sha256(canonical_root_window_projection_values(values))
        return cls(
            schema_version="1.0",
            window_id=_stable_uuid("canonical-root-window", digest),
            window_logical_key=f"temporal-window:{digest}",
            semantic_sha256=digest,
            recording_identity=context.recording_identity,
            source_content_sha256=context.source_content_sha256,
            mcap_id=context.ready_manifest.mcap_id,
            camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
            alignment_id=context.alignment_manifest.alignment_id,
            camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=context.alignment_semantic_sha256,
            recording_duration_ns=duration,
            reference_timebase="recording_relative_ns",
            requested_interval=requested_interval,
            interval=effective,
            purpose=purpose,
            window_policy_version=window_policy_version,
            source_subject_type="RECORDING",
            source_subject_logical_key=f"recording:{context.recording_identity}",
            parent_window_logical_key=None,
            source_lineage_sha256=context.semantic_sha256,
            refinement_role="ROOT",
            generation=0,
            created_at=created_at,
        )


def canonical_lineage(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalRootWindow,
    sampling_plan: SamplingPlan,
) -> PackageLineage:
    """Construct lineage rather than accepting caller-provided digest values."""

    context = _strict_context(context)
    try:
        window = CanonicalRootWindow.model_validate(window.model_dump(mode="python"), strict=True)
        sampling_plan = SamplingPlan.model_validate(
            sampling_plan.model_dump(mode="python"), strict=True
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CanonicalOfflineConfigurationError(
            "window or sampling plan failed strict validation"
        ) from exc
    if (
        window.recording_identity != context.recording_identity
        or window.source_content_sha256 != context.source_content_sha256
        or window.mcap_id != context.ready_manifest.mcap_id
        or window.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
        or window.alignment_id != context.alignment_manifest.alignment_id
        or window.camera_mapping_semantic_sha256 != context.camera_mapping_semantic_sha256
        or window.alignment_semantic_sha256 != context.alignment_semantic_sha256
        or window.source_lineage_sha256 != context.semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "root window does not match its admitted recording context"
        )

    return PackageLineage(
        source_content_sha256=context.source_content_sha256,
        window_semantic_sha256=window.semantic_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        sampling_plan_sha256=sampling_plan_digest(sampling_plan),
    )


class CanonicalOfflineExecutionPolicy(StrictModel):
    """Pinned policy bundle for the local vertical slice."""

    schema_version: Literal["1.0"]
    policy_version: SchemaVersion
    window_policy_version: SchemaVersion
    token_policy_version: SchemaVersion
    parser_version: SchemaVersion
    enrichment_policy_version: SchemaVersion
    projector_policy_version: SchemaVersion
    reduction_policy: NonEmptyString
    reduction_policy_version: SchemaVersion
    max_attempts: PositiveInt
    output_admission_policy: ProductionOutputAdmissionPolicyRef
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        expected = semantic_sha256(canonical_execution_policy_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("canonical offline execution policy semantic_sha256 is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_version: str,
        window_policy_version: str,
        token_policy_version: str,
        parser_version: str,
        enrichment_policy_version: str,
        projector_policy_version: str,
        reduction_policy: str,
        reduction_policy_version: str,
        max_attempts: int,
        output_admission_policy: ProductionOutputAdmissionPolicyRef,
    ) -> Self:
        values: dict[str, object] = {
            "policy_version": policy_version,
            "window_policy_version": window_policy_version,
            "token_policy_version": token_policy_version,
            "parser_version": parser_version,
            "enrichment_policy_version": enrichment_policy_version,
            "projector_policy_version": projector_policy_version,
            "reduction_policy": reduction_policy,
            "reduction_policy_version": reduction_policy_version,
            "max_attempts": max_attempts,
            "output_admission_policy": output_admission_policy,
        }
        digest = semantic_sha256(canonical_execution_policy_projection_values(values))
        return cls(
            schema_version="1.0",
            policy_version=policy_version,
            window_policy_version=window_policy_version,
            token_policy_version=token_policy_version,
            parser_version=parser_version,
            enrichment_policy_version=enrichment_policy_version,
            projector_policy_version=projector_policy_version,
            reduction_policy=reduction_policy,
            reduction_policy_version=reduction_policy_version,
            max_attempts=max_attempts,
            output_admission_policy=output_admission_policy,
            semantic_sha256=digest,
        )


def canonical_execution_policy_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    output_policy = values["output_admission_policy"]
    if not isinstance(output_policy, ProductionOutputAdmissionPolicyRef):
        raise TypeError("output_admission_policy must be a ProductionOutputAdmissionPolicyRef")
    return {
        "policy_version": values["policy_version"],
        "window_policy_version": values["window_policy_version"],
        "token_policy_version": values["token_policy_version"],
        "parser_version": values["parser_version"],
        "enrichment_policy_version": values["enrichment_policy_version"],
        "projector_policy_version": values["projector_policy_version"],
        "reduction_policy": values["reduction_policy"],
        "reduction_policy_version": values["reduction_policy_version"],
        "max_attempts": values["max_attempts"],
        "output_admission_policy": output_policy.model_dump(mode="json"),
    }


def canonical_execution_policy_projection(
    policy: CanonicalOfflineExecutionPolicy,
) -> dict[str, object]:
    return canonical_execution_policy_projection_values(
        {
            "policy_version": policy.policy_version,
            "window_policy_version": policy.window_policy_version,
            "token_policy_version": policy.token_policy_version,
            "parser_version": policy.parser_version,
            "enrichment_policy_version": policy.enrichment_policy_version,
            "projector_policy_version": policy.projector_policy_version,
            "reduction_policy": policy.reduction_policy,
            "reduction_policy_version": policy.reduction_policy_version,
            "max_attempts": policy.max_attempts,
            "output_admission_policy": policy.output_admission_policy,
        }
    )


class CanonicalOutputAdmissionDecision(StrictModel):
    """Local output-level decision; it is not a registered durable schema yet."""

    schema_version: Literal["1.0"]
    decision_id: OpaqueUuid
    decision: Literal["PRODUCTION_ADMITTED", "NO_EVENTS"]
    semantic_sha256: Sha256Digest
    recording_identity: Sha256Digest
    source_enrichment: PlatformEnrichedOutputReference
    policy_version: SchemaVersion
    policy_sha256: Sha256Digest
    admitted_claim_ordinals: tuple[NonNegativeInt, ...]
    reason_code: NonEmptyString
    production_output_admission: ProductionOutputAdmissionProof | None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.source_enrichment.recording_identity != self.recording_identity:
            raise ValueError("output decision crosses recording scope")
        if self.decision == "NO_EVENTS":
            if self.admitted_claim_ordinals or self.production_output_admission is not None:
                raise ValueError("NO_EVENTS cannot carry admitted claims or a production proof")
        elif self.production_output_admission is None:
            raise ValueError("PRODUCTION_ADMITTED requires an output admission proof")
        else:
            proof = self.production_output_admission
            if proof.source_enrichments != (self.source_enrichment,):
                raise ValueError("output decision proof does not bind its source enrichment")
            proof_ordinals = tuple(
                sorted(item.fusion_output_ordinal for item in proof.admitted_hypothesis_facts)
            )
            if proof_ordinals != self.admitted_claim_ordinals:
                raise ValueError("output decision ordinals differ from admitted proof facts")
        expected = semantic_sha256(canonical_output_decision_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("output decision semantic_sha256 is inconsistent")
        if self.decision_id != _stable_uuid("canonical-output-admission", expected):
            raise ValueError("output decision ID is inconsistent")
        return self


def canonical_output_decision_projection(
    decision: CanonicalOutputAdmissionDecision,
) -> dict[str, object]:
    return {
        "decision": decision.decision,
        "recording_identity": decision.recording_identity,
        "source_enrichment": decision.source_enrichment.model_dump(mode="json"),
        "policy_version": decision.policy_version,
        "policy_sha256": decision.policy_sha256,
        "admitted_claim_ordinals": list(decision.admitted_claim_ordinals),
        "reason_code": decision.reason_code,
        "production_output_admission": (
            None
            if decision.production_output_admission is None
            else decision.production_output_admission.model_dump(mode="json")
        ),
    }


class FusionEventHypothesisProjector:
    """Turn one admitted fusion enrichment into deterministic event hypotheses."""

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
        enriched_output: OrchestratorEnrichedOutput,
        interval: NanosecondInterval,
    ) -> tuple[CanonicalOutputAdmissionDecision, tuple[PlatformEnrichedEventHypothesis, ...]]:
        context = _strict_context(context)
        try:
            output = OrchestratorEnrichedOutput.model_validate(
                enriched_output.model_dump(mode="python"), strict=True
            )
        except ValueError as exc:
            raise CanonicalOfflineConfigurationError(
                "enriched output failed strict validation"
            ) from exc
        if output.task is not VisionTask.FUSION_ADJUDICATION:
            raise ValueError("event projection requires FUSION_ADJUDICATION output")
        if output.authority.recording_identity != context.recording_identity:
            raise ValueError("enriched output recording does not match admission context")
        if (
            output.authority.mcap_id != context.ready_manifest.mcap_id
            or output.authority.camera_mapping_run_id
            != context.ready_manifest.camera_mapping_run_id
            or output.authority.alignment_id != context.alignment_manifest.alignment_id
        ):
            raise ValueError("enriched output authority does not match admission lineage")
        if not isinstance(interval, NanosecondInterval):
            raise TypeError("projection interval must be a NanosecondInterval")
        if interval.start_ns < 0 or interval.end_ns > context.ready_manifest.recording.duration_ns:
            raise ValueError("projection interval is outside the admitted recording")
        source_ref = PlatformEnrichedOutputReference.from_output(output)
        if output.abstained:
            decision = _output_decision(
                recording_identity=context.recording_identity,
                source_ref=source_ref,
                policy=self._policy,
                decision="NO_EVENTS",
                admitted_claim_ordinals=(),
                reason_code="PROVIDER_ABSTAINED",
                production_output_admission=None,
            )
            return decision, ()

        if not output.claims:
            raise ValueError("non-abstained fusion output must contain hypotheses")
        fingerprints: set[str] = set()
        facts: list[ProductionAdmittedHypothesisFact] = []
        drafts: list[tuple[int, NanosecondInterval, str, str]] = []
        for claim in output.claims:
            if claim.kind is not ProviderClaimKind.FUSION_HYPOTHESIS:
                raise ValueError("fusion output contains a non-fusion claim")
            if claim.interval is None or not _contains_interval(interval, claim.interval):
                raise ValueError("fusion hypothesis interval is outside the root window")
            if not claim.evidence:
                raise ValueError("fusion hypothesis requires authoritative evidence")
            fingerprint = semantic_sha256(
                {
                    "recording_identity": context.recording_identity,
                    "start_ns": str(claim.interval.start_ns),
                    "end_ns": str(claim.interval.end_ns),
                    "label": claim.label,
                    "observation": claim.observation.value,
                    "conflict_codes": sorted(claim.conflict_codes),
                    "evidence": [
                        {
                            "package_semantic_content_sha256": item.package_semantic_content_sha256,
                            "package_ordinal": item.package_ordinal,
                            "camera_ordinal": item.camera_ordinal,
                            "frame_ordinal": item.frame_ordinal,
                            "source_artifact_sha256": item.source_artifact_sha256,
                            "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
                        }
                        for item in claim.evidence
                    ],
                    "projector_version": self._projector_version,
                }
            )
            if fingerprint in fingerprints:
                raise ValueError("fusion output contains duplicate semantic fingerprints")
            fingerprints.add(fingerprint)
            fusion_digest = semantic_sha256(
                {
                    "semantic_fingerprint_sha256": fingerprint,
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
                    fusion_output_ordinal=claim.claim_ordinal,
                    effective_interval=effective_interval,
                    semantic_fingerprint_sha256=fingerprint,
                    fusion_logical_key=fusion_logical_key,
                )
            )
            drafts.append(
                (claim.claim_ordinal, effective_interval, fingerprint, fusion_logical_key)
            )
        proof = ProductionOutputAdmissionProof.create(
            recording_identity=context.recording_identity,
            source_enrichments=(source_ref,),
            admitted_hypothesis_facts=facts,
            policy=self._policy,
        )
        admission = ProductionAdmissionProof.from_context(context)
        hypotheses = tuple(
            PlatformEnrichedEventHypothesis.create(
                recording_identity=context.recording_identity,
                effective_interval=effective_interval,
                semantic_fingerprint_sha256=fingerprint,
                fusion_logical_key=fusion_logical_key,
                fusion_output_ordinal=claim_ordinal,
                source_enrichments=(source_ref,),
                production_admission=admission,
                production_output_admission=proof,
            )
            for claim_ordinal, effective_interval, fingerprint, fusion_logical_key in drafts
        )
        decision = _output_decision(
            recording_identity=context.recording_identity,
            source_ref=source_ref,
            policy=self._policy,
            decision="PRODUCTION_ADMITTED",
            admitted_claim_ordinals=tuple(item.fusion_output_ordinal for item in hypotheses),
            reason_code="FUSION_HYPOTHESES_VALIDATED",
            production_output_admission=proof,
        )
        return decision, hypotheses


class CanonicalOfflineRunResult(StrictModel):
    """Inspectable local result; this is not a registered persistence schema."""

    schema_version: Literal["1.0"]
    run_id: OpaqueUuid
    recording_identity: Sha256Digest
    status: CanonicalOfflineRunStatus
    window: CanonicalRootWindow | None
    materialized_package_ids: tuple[OpaqueUuid, ...]
    package_set: TemporalPackageSet | None
    input_plan: InferenceInputPlan | None
    reference_catalog: ProviderReferenceCatalog | None
    barrier_reduction: InferenceCallReduction | None
    terminal: ModelInference | None
    selection: InferenceAttemptSelection | None
    raw_response: RawProviderResponseArtifact | None
    parsed_claims: ParsedProviderClaimArtifact | None
    selected_output: SelectedAttemptOutput | None
    enriched_output: OrchestratorEnrichedOutput | None
    output_decision: CanonicalOutputAdmissionDecision | None
    hypotheses: tuple[PlatformEnrichedEventHypothesis, ...]
    identity_result: EventIdentityBatchResult | None
    attempt_count: NonNegativeInt
    adapter_infer_calls: NonNegativeInt
    network_call_count: Literal[0]
    error: CanonicalOfflineError | None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.window is not None and self.window.recording_identity != self.recording_identity:
            raise ValueError("run window crosses recording scope")
        if self.package_set is not None:
            expected_ids = tuple(member.package_id for member in self.package_set.members)
            if self.materialized_package_ids != expected_ids:
                raise ValueError("run package IDs do not match the package set")
        elif self.materialized_package_ids:
            raise ValueError("materialized package IDs require a package set")
        if self.input_plan is not None and self.package_set is not None:
            plan_ids = tuple(item.package_id for item in self.input_plan.subject.packages)
            if plan_ids != self.materialized_package_ids:
                raise ValueError("run input plan does not match materialized packages")
        if self.reference_catalog is not None and (
            self.input_plan is None
            or self.reference_catalog.input_plan_id != self.input_plan.input_plan_id
            or self.reference_catalog.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
        ):
            raise ValueError("run reference catalog does not match its input plan")
        if self.barrier_reduction is not None and (
            self.input_plan is None
            or self.barrier_reduction.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
        ):
            raise ValueError("run reduction does not match its input plan")
        if self.terminal is not None and self.attempt_count == 0:
            raise ValueError("a terminal result requires an orchestration attempt")
        if self.selection is not None and (
            self.terminal is None
            or self.selection.inference_id != self.terminal.inference_id
            or self.selection.logical_invocation_id != self.terminal.logical_invocation_id
        ):
            raise ValueError("run selection does not match the selected terminal")

        artifact_chain = (self.raw_response, self.parsed_claims, self.selected_output)
        if any(item is not None for item in artifact_chain):
            if any(item is None for item in artifact_chain) or self.terminal is None:
                raise ValueError("raw, parsed, and selected artifacts must be present together")
            assert self.raw_response is not None
            assert self.parsed_claims is not None
            assert self.selected_output is not None
            if (
                self.raw_response.inference_id != self.terminal.inference_id
                or self.parsed_claims.raw_response != self.raw_response
                or self.selected_output.inference_id != self.terminal.inference_id
                or self.selected_output.raw_response_artifact_id != self.raw_response.artifact_id
                or self.selected_output.parsed_claim_artifact_id != self.parsed_claims.artifact_id
            ):
                raise ValueError("run artifact lineage is inconsistent")
        if self.enriched_output is not None and (
            self.enriched_output.authority.recording_identity != self.recording_identity
            or self.selected_output is None
            or self.enriched_output.selected_attempt != self.selected_output
        ):
            raise ValueError("run enriched output lineage is inconsistent")
        if self.output_decision is not None and (
            self.output_decision.recording_identity != self.recording_identity
            or self.enriched_output is None
            or self.output_decision.source_enrichment
            != PlatformEnrichedOutputReference.from_output(self.enriched_output)
        ):
            raise ValueError("run output decision lineage is inconsistent")
        if any(item.recording_identity != self.recording_identity for item in self.hypotheses):
            raise ValueError("run hypotheses cross recording scope")
        if self.identity_result is not None and (
            self.identity_result.recording_identity != self.recording_identity
            or len(self.identity_result.assignments) != len(self.hypotheses)
        ):
            raise ValueError("run identity result does not cover its hypotheses")

        completed = self.status in {
            CanonicalOfflineRunStatus.SUCCEEDED,
            CanonicalOfflineRunStatus.NO_EVENTS,
        }
        if completed and self.error is not None:
            raise ValueError("completed run cannot contain an error")
        if not completed and self.error is None:
            raise ValueError("non-completed run requires a structured error")
        if self.status is CanonicalOfflineRunStatus.SUCCEEDED:
            if (
                self.terminal is None
                or self.terminal.status is not InferenceStatus.SUCCEEDED
                or self.selection is None
                or self.barrier_reduction is None
                or self.enriched_output is None
                or self.output_decision is None
                or self.output_decision.decision != "PRODUCTION_ADMITTED"
                or not self.hypotheses
                or self.identity_result is None
            ):
                raise ValueError("successful run is missing admitted identity lineage")
        elif self.status is CanonicalOfflineRunStatus.NO_EVENTS:
            if (
                self.terminal is None
                or self.terminal.status is not InferenceStatus.SUCCEEDED
                or self.selection is None
                or self.barrier_reduction is None
                or self.enriched_output is None
                or not self.enriched_output.abstained
                or self.output_decision is None
                or self.output_decision.decision != "NO_EVENTS"
                or self.hypotheses
                or self.identity_result is not None
            ):
                raise ValueError("NO_EVENTS run has an inconsistent terminal shape")
        elif self.status is CanonicalOfflineRunStatus.BLOCKED_MULTI_PART and (
            self.input_plan is None
            or len(self.input_plan.call_plan.parts) <= 1
            or self.terminal is not None
            or self.attempt_count != 0
        ):
            raise ValueError("multi-part block must happen before dispatch")
        return self


class _SinglePartIdentityReducer:
    def reduce(
        self,
        *,
        input_plan: InferenceInputPlan,
        ordered_completions: tuple[InferenceCallPartCompletion, ...],
    ) -> Mapping[str, object]:
        if len(input_plan.call_plan.parts) != 1 or len(ordered_completions) != 1:
            raise CanonicalOfflineConfigurationError("identity reducer requires one call part")
        completion = ordered_completions[0]
        if (
            completion.status is not InferenceStatus.SUCCEEDED
            or completion.normalized_output is None
        ):
            raise CanonicalOfflineConfigurationError("identity reducer requires one success")
        return dict(completion.normalized_output)


class CanonicalOfflinePipeline:
    """Run the canonical post-admission path without network-capable adapters."""

    def __init__(
        self,
        *,
        package_builder: PackageSetBuilder,
        materializer: OfflineTemporalPackageMaterializer,
        input_preparer: InputPlanPreparer,
        adapter: OfflineFixtureVisionAdapter,
        inference_policy: InferencePolicy,
        schema_registry: SchemaRegistry,
        identity_registry: EventIdentityRegistryService,
        execution_policy: CanonicalOfflineExecutionPolicy,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(package_builder, PackageSetBuilder):
            raise TypeError("package_builder must be a PackageSetBuilder")
        if not isinstance(materializer, OfflineTemporalPackageMaterializer):
            raise TypeError("materializer must be an OfflineTemporalPackageMaterializer")
        if not isinstance(input_preparer, InputPlanPreparer):
            raise TypeError("input_preparer must be an InputPlanPreparer")
        if not isinstance(adapter, OfflineFixtureVisionAdapter):
            raise TypeError("canonical pipeline accepts only OfflineFixtureVisionAdapter")
        if not isinstance(inference_policy, InferencePolicy):
            raise TypeError("inference_policy must be an InferencePolicy")
        if not isinstance(schema_registry, SchemaRegistry):
            raise TypeError("schema_registry must be a SchemaRegistry")
        if not isinstance(identity_registry, EventIdentityRegistryService):
            raise TypeError("identity_registry must be an EventIdentityRegistryService")
        if not isinstance(execution_policy, CanonicalOfflineExecutionPolicy):
            raise TypeError("execution_policy must be a CanonicalOfflineExecutionPolicy")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._package_builder = package_builder
        self._materializer = materializer
        self._input_preparer = input_preparer
        self._adapter = adapter
        self._inference_policy = inference_policy
        self._schema_registry = schema_registry
        self._identity_registry = identity_registry
        self._execution_policy = execution_policy
        self._clock = clock or _utc_now
        self._validate_configuration()

        self._ledger = InMemoryInferenceLedger()
        schema_registry.resolve_exact(_schema_ref(inference_policy.output_schema))
        self._orchestrator = InferenceOrchestrator(
            adapters={adapter.provider: adapter},
            task_policies={inference_policy.task: inference_policy},
            schema_artifacts={
                item.ref.artifact_id: item.document_bytes for item in schema_registry.entries
            },
            ledger=self._ledger,
            clock=self._clock,
        )
        self._call_barrier_storage = InMemoryInferenceCallBarrierStorage()
        reducer_key = (
            execution_policy.reduction_policy,
            execution_policy.reduction_policy_version,
        )
        self._call_barrier = InferenceCallBarrierCoordinator(
            barriers=BarrierCoordinator(InMemoryBarrierStorage()),
            storage=self._call_barrier_storage,
            reducers={reducer_key: _SinglePartIdentityReducer()},
        )
        self._enricher = ProviderClaimEnricher(schema_registry)
        self._projector = FusionEventHypothesisProjector(
            policy=execution_policy.output_admission_policy,
            projector_version=execution_policy.projector_policy_version,
        )

    @property
    def ledger(self) -> InMemoryInferenceLedger:
        return self._ledger

    @property
    def call_barrier_storage(self) -> InMemoryInferenceCallBarrierStorage:
        return self._call_barrier_storage

    @property
    def adapter(self) -> OfflineFixtureVisionAdapter:
        return self._adapter

    async def run(
        self,
        *,
        admitted_context: AdmittedRecordingContextV2,
        requested_interval: NanosecondInterval,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        rendered_item_factory: RenderedItemFactory | None = None,
    ) -> CanonicalOfflineRunResult:
        """Execute one exact local run, retaining every admitted intermediate."""

        context = _strict_context(admitted_context)
        if not isinstance(requested_interval, NanosecondInterval):
            raise TypeError("requested_interval must be a NanosecondInterval")
        if not isinstance(sampling_plan, SamplingPlan):
            raise TypeError("sampling_plan must be a SamplingPlan")
        if not isinstance(frame_index, CanonicalSixCameraFrameIndex):
            raise TypeError("frame_index must be a CanonicalSixCameraFrameIndex")
        if not callable(artifact_resolver):
            raise TypeError("artifact_resolver must be callable")
        if rendered_item_factory is not None and not callable(rendered_item_factory):
            raise TypeError("rendered_item_factory must be callable")

        created_at = _timestamp(self._clock)
        run_id = _stable_uuid(
            "canonical-offline-run",
            context.recording_identity,
            requested_interval.start_ns,
            requested_interval.end_ns,
            sampling_plan_digest(sampling_plan),
            self._execution_policy.semantic_sha256,
        )
        infer_calls_before = self._adapter.infer_calls
        state: dict[str, object] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "recording_identity": context.recording_identity,
            "window": None,
            "materialized_package_ids": (),
            "package_set": None,
            "input_plan": None,
            "reference_catalog": None,
            "barrier_reduction": None,
            "terminal": None,
            "selection": None,
            "raw_response": None,
            "parsed_claims": None,
            "selected_output": None,
            "enriched_output": None,
            "output_decision": None,
            "hypotheses": (),
            "identity_result": None,
            "attempt_count": 0,
        }

        def finish(
            status: CanonicalOfflineRunStatus,
            error: CanonicalOfflineError | None = None,
        ) -> CanonicalOfflineRunResult:
            if self._adapter.network_call_count != 0:
                raise CanonicalOfflineConfigurationError(
                    "offline fixture adapter reported a network call"
                )
            return CanonicalOfflineRunResult.model_validate(
                {
                    **state,
                    "status": status,
                    "adapter_infer_calls": self._adapter.infer_calls - infer_calls_before,
                    "network_call_count": 0,
                    "error": error,
                },
                strict=True,
            )

        try:
            if context.alignment_manifest.reference_timebase != "recording_relative_ns":
                raise CanonicalOfflineConfigurationError(
                    "canonical window requires recording_relative_ns alignment"
                )
            window = CanonicalRootWindow.from_context(
                context=context,
                requested_interval=requested_interval,
                purpose=SamplingPurpose.ACTION_DENSE,
                window_policy_version=self._execution_policy.window_policy_version,
                created_at=created_at,
            )
            state["window"] = window
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(CanonicalOfflineStage.WINDOW, "INVALID_ROOT_WINDOW", exc),
            )

        try:
            lineage = canonical_lineage(
                context=context,
                window=window,
                sampling_plan=sampling_plan,
            )
            planned_parts = self._package_builder.plan_parts(window, sampling_plan)
            if not planned_parts:
                raise CanonicalOfflineConfigurationError("root window produced no parts")
            materialized = tuple(
                self._materializer.materialize_admitted(
                    part=part,
                    sampling_plan=sampling_plan,
                    admitted_context=context,
                    frame_index=frame_index,
                    lineage=lineage,
                    window_id=window.window_id,
                    artifact_resolver=artifact_resolver,
                    created_at=created_at,
                )
                for part in planned_parts
            )
            _validate_materialized_chain(
                context=context,
                window=window,
                lineage=lineage,
                planned_parts=planned_parts,
                materialized=materialized,
            )
            package_set = self._package_builder.build_package_set(
                window,
                sampling_plan,
                context.alignment_manifest.alignment_id,
                lineage=lineage,
                materialized_members=tuple(item.package_ref for item in materialized),
                created_at=created_at,
            )
            _validate_package_set_chain(
                context=context,
                window=window,
                lineage=lineage,
                package_set=package_set,
                materialized=materialized,
                reduction_policy_version=self._package_builder.reduction_policy_version,
            )
            state["materialized_package_ids"] = tuple(
                item.package.package_id for item in materialized
            )
            state["package_set"] = package_set
        except PackageMaterializationError as exc:
            return finish(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(CanonicalOfflineStage.MATERIALIZATION, exc.code.value, exc),
            )
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.MATERIALIZATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.MATERIALIZATION,
                    "PACKAGE_CHAIN_INVALID",
                    exc,
                ),
            )

        try:
            capabilities = await self._adapter.capabilities(
                self._inference_policy.model_name,
                self._inference_policy.model_version,
            )
            capabilities = _validated_capabilities(
                capabilities,
                inference_policy=self._inference_policy,
                input_preparer=self._input_preparer,
            )
            applicable_limits = applicable_limits_from_capabilities(
                max_images_per_request=capabilities.max_images_per_request,
                max_pixels_per_image=capabilities.max_pixels_per_image,
                max_payload_bytes=capabilities.max_payload_bytes,
                max_input_tokens=capabilities.max_input_tokens,
            )
            target = InputPlanTarget(
                provider=capabilities.provider,
                model_name=capabilities.model_name,
                model_version=capabilities.model_version,
                adapter_version=self._inference_policy.adapter_version,
                planner_version=self._input_preparer.planner_version,
                capability_snapshot_id=capabilities.snapshot_id,
                capability_snapshot_sha256=capabilities.snapshot_digest,
            )
            request_catalog_id = _stable_uuid(
                "provider-request-catalog",
                lineage,
                self._execution_policy.semantic_sha256,
                capabilities.snapshot_digest,
            )
            prepared = self._input_preparer.prepare_rendering(
                packages=materialized,
                task=VisionTask.FUSION_ADJUDICATION,
                request_catalog_id=request_catalog_id,
                applicable_limits=applicable_limits,
                created_at=created_at,
                rendered_item_factory=rendered_item_factory,
            )
            prompt_entries = ProviderReferenceCatalog.derive_entries(
                request_catalog_sha256=prepared.request_catalog.semantic_sha256,
                rendered_items=prepared.rendered_items,
                token_policy_version=self._execution_policy.token_policy_version,
            )
            rendered_prompt = _rendered_prompt_bytes(
                inference_policy=self._inference_policy,
                request_catalog_sha256=prepared.request_catalog.semantic_sha256,
                token_policy_version=self._execution_policy.token_policy_version,
                entries=prompt_entries,
            )
            enriched_schema = self._required_enriched_schema()
            prompt_output = PromptOutputContract(
                prompt_version=self._inference_policy.prompt_version,
                prompt_sha256=self._inference_policy.prompt_sha256,
                rendered_message_sha256=exact_bytes_sha256(rendered_prompt),
                provider_response_schema_sha256=self._inference_policy.output_schema.sha256,
                enriched_domain_schema_sha256=enriched_schema.sha256,
                protocol_mode="json-schema",
                tool_mode="none",
            )
            input_plan_id = _stable_uuid(
                "inference-input-plan",
                prepared.request_catalog.semantic_sha256,
                target.capability_snapshot_sha256,
                exact_bytes_sha256(rendered_prompt),
                self._execution_policy.semantic_sha256,
            )
            input_plan = self._input_preparer.finalize(
                prepared=prepared,
                input_plan_id=input_plan_id,
                target=target,
                prompt_output=prompt_output,
                created_at=created_at,
            )
            reference_catalog = ProviderReferenceCatalog.build(
                input_plan=input_plan,
                reference_catalog_id=_stable_uuid(
                    "provider-reference-catalog",
                    input_plan.semantic_sha256,
                    self._execution_policy.token_policy_version,
                ),
                token_policy_version=self._execution_policy.token_policy_version,
                created_at=created_at,
            )
            if reference_catalog.entries != prompt_entries:
                raise CanonicalOfflineConfigurationError(
                    "rendered prompt tokens differ from finalized reference catalog"
                )
            _validate_input_plan_chain(
                package_set=package_set,
                materialized=materialized,
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                inference_policy=self._inference_policy,
                execution_policy=self._execution_policy,
                capabilities=capabilities,
            )
            state["input_plan"] = input_plan
            state["reference_catalog"] = reference_catalog
        except asyncio.CancelledError:
            raise
        except (
            InputPreparationError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.CONFIGURATION_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    "INPUT_PLAN_INVALID",
                    exc,
                ),
            )

        if len(input_plan.call_plan.parts) != 1:
            return finish(
                CanonicalOfflineRunStatus.BLOCKED_MULTI_PART,
                _canonical_error(
                    CanonicalOfflineStage.PREPARATION,
                    "MULTI_PART_EXECUTION_NOT_IMPLEMENTED",
                    CanonicalOfflineConfigurationError(
                        "canonical offline execution admits one provider call part"
                    ),
                ),
            )

        package_inputs = _package_inputs(package_set)
        part = input_plan.call_plan.parts[0]
        try:
            self._call_barrier.declare(input_plan, created_at=created_at)
        except InferenceCallBarrierError as exc:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_DECLARATION_FAILED",
                    exc,
                ),
            )

        selected_terminal: ModelInference | None = None
        selection: InferenceAttemptSelection | None = None
        reduction: InferenceCallReduction | None = None
        for attempt in range(1, self._execution_policy.max_attempts + 1):
            state["attempt_count"] = attempt
            try:
                terminal = await self._orchestrator.orchestrate(
                    task=VisionTask.FUSION_ADJUDICATION,
                    package_set_id=package_set.package_set_id,
                    mcap_id=context.ready_manifest.mcap_id,
                    camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
                    alignment_id=context.alignment_manifest.alignment_id,
                    start_ns=window.interval.start_ns,
                    end_ns=window.interval.end_ns,
                    package_inputs=package_inputs,
                    rendered_input_digest=part.item_manifest_sha256,
                    input_plan=input_plan,
                    input_plan_part_ordinal=0,
                    input_config={
                        "canonical_execution_policy_sha256": (
                            self._execution_policy.semantic_sha256
                        )
                    },
                    sampling_config={
                        "sampling_plan_version": sampling_plan.version,
                        "sampling_plan_sha256": sampling_plan_digest(sampling_plan),
                    },
                    metadata={"canonical_offline_run_id": run_id},
                    attempt=attempt,
                    retry_count=attempt - 1,
                )
            except asyncio.CancelledError:
                raise
            except (InferenceOrchestrationError, TypeError, ValueError) as exc:
                return finish(
                    CanonicalOfflineRunStatus.INFERENCE_FAILED,
                    _canonical_error(
                        CanonicalOfflineStage.INFERENCE,
                        "ORCHESTRATION_FAILED",
                        exc,
                    ),
                )
            state["terminal"] = terminal

            if terminal.status is InferenceStatus.SUCCEEDED:
                selection = self._ledger.get_selection(
                    terminal.logical_invocation_id,
                    self._inference_policy.selection_policy_version,
                )
                if selection is None:
                    return finish(
                        CanonicalOfflineRunStatus.INFERENCE_FAILED,
                        _canonical_error(
                            CanonicalOfflineStage.INFERENCE,
                            "SELECTED_ATTEMPT_MISSING",
                            CanonicalOfflineConfigurationError(
                                "successful invocation has no persisted selection"
                            ),
                        ),
                    )
                selected_terminal = self._ledger.get_terminal(selection.inference_id)
                if (
                    selected_terminal is None
                    or selected_terminal.status is not InferenceStatus.SUCCEEDED
                    or not selected_terminal.output_valid
                ):
                    return finish(
                        CanonicalOfflineRunStatus.INFERENCE_FAILED,
                        _canonical_error(
                            CanonicalOfflineStage.INFERENCE,
                            "SELECTED_TERMINAL_INVALID",
                            CanonicalOfflineConfigurationError(
                                "persisted selection does not reference a valid success"
                            ),
                        ),
                    )
                try:
                    self._call_barrier.submit_part_terminal(
                        input_plan,
                        selected_terminal,
                        selection=selection,
                    )
                    reduction = self._call_barrier.reduce(
                        input_plan,
                        reduced_at=_timestamp(self._clock),
                    )
                except InferenceCallBarrierError as exc:
                    return finish(
                        CanonicalOfflineRunStatus.INFERENCE_FAILED,
                        _canonical_error(
                            CanonicalOfflineStage.INFERENCE,
                            "BARRIER_REDUCTION_FAILED",
                            exc,
                        ),
                    )
                if selected_terminal.normalized_output != reduction.normalized_output:
                    return finish(
                        CanonicalOfflineRunStatus.INVALID_OUTPUT,
                        _canonical_error(
                            CanonicalOfflineStage.INFERENCE,
                            "REDUCTION_OUTPUT_MISMATCH",
                            CanonicalOfflineConfigurationError(
                                "single-part reduction changed selected output"
                            ),
                        ),
                    )
                state["terminal"] = selected_terminal
                state["selection"] = selection
                state["barrier_reduction"] = reduction
                break

            failure = terminal.failure
            retryable = failure is not None and failure.retryability in {
                Retryability.RETRYABLE,
                Retryability.RATE_LIMITED,
            }
            if retryable and attempt < self._execution_policy.max_attempts:
                continue
            try:
                self._call_barrier.submit_part_terminal(
                    input_plan,
                    terminal,
                    failure_is_final=True,
                )
            except InferenceCallBarrierError as exc:
                return finish(
                    CanonicalOfflineRunStatus.INFERENCE_FAILED,
                    _canonical_error(
                        CanonicalOfflineStage.INFERENCE,
                        "FINAL_FAILURE_BARRIER_FAILED",
                        exc,
                    ),
                )
            run_status = (
                CanonicalOfflineRunStatus.INVALID_OUTPUT
                if terminal.status is InferenceStatus.INVALID_OUTPUT
                else CanonicalOfflineRunStatus.INFERENCE_FAILED
            )
            failure_detail: object = (
                failure.detail if failure is not None else terminal.status.value
            )
            return finish(
                run_status,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    failure.code if failure is not None else terminal.status.value,
                    failure_detail,
                ),
            )

        if selected_terminal is None or selection is None or reduction is None:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "NO_SELECTED_TERMINAL",
                    "attempt loop ended without a selected successful terminal",
                ),
            )

        try:
            raw_artifact_id = _terminal_raw_artifact_id(selected_terminal)
            stored_raw = self._adapter.raw_store.get(raw_artifact_id)
            if (
                stored_raw.artifact_id != raw_artifact_id
                or stored_raw.request_id != selected_terminal.request_id
                or stored_raw.provider_request_id != selected_terminal.provider_request_id
            ):
                raise CanonicalOfflineConfigurationError(
                    "stored raw bytes do not match the selected terminal request"
                )
            parsed_claims = self._adapter.parser.parse_artifact(
                stored=stored_raw,
                inference_id=selected_terminal.inference_id,
                provider=selected_terminal.provider,
                model_name=selected_terminal.model_name,
                model_version=selected_terminal.model_version,
                provider_claim_schema=self._inference_policy.output_schema,
                task=VisionTask.FUSION_ADJUDICATION,
                artifact_id=_stable_uuid(
                    "parsed-provider-claim",
                    stored_raw.exact_bytes_sha256,
                    self._inference_policy.output_schema.sha256,
                    self._execution_policy.parser_version,
                ),
                created_at=created_at,
            )
            parsed_payload = parsed_claims.payload.model_dump(mode="json")
            if (
                parsed_payload != selected_terminal.normalized_output
                or parsed_payload != reduction.normalized_output
            ):
                raise CanonicalOfflineConfigurationError(
                    "parsed claims differ from selected and reduced output"
                )
            selected_output = SelectedAttemptOutput.create(parsed_claims)
            state["raw_response"] = parsed_claims.raw_response
            state["parsed_claims"] = parsed_claims
            state["selected_output"] = selected_output
        except (
            RawProviderBytesStoreError,
            StrictProviderClaimParseError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.PARSING,
                    "SELECTED_RAW_OUTPUT_INVALID",
                    exc,
                ),
            )

        try:
            work_digest = semantic_sha256(
                {
                    "recording_identity": context.recording_identity,
                    "input_plan_semantic_sha256": input_plan.semantic_sha256,
                    "selected_attempt_output_sha256": selected_output.output_sha256,
                    "enrichment_policy_version": (self._execution_policy.enrichment_policy_version),
                }
            )
            authority = EnrichmentAuthorityContext(
                recording_identity=context.recording_identity,
                mcap_id=context.ready_manifest.mcap_id,
                camera_mapping_run_id=context.ready_manifest.camera_mapping_run_id,
                alignment_id=context.alignment_manifest.alignment_id,
                inference_id=selected_terminal.inference_id,
                logical_invocation_id=selected_terminal.logical_invocation_id,
                prompt_version=self._inference_policy.prompt_version,
                prompt_artifact_id=self._inference_policy.prompt_artifact_id,
                prompt_sha256=self._inference_policy.prompt_sha256,
                work_node_type="INFERENCE_ENRICHMENT",
                work_node_logical_key=f"inference-work:{work_digest}",
            )
            enriched_schema = self._required_enriched_schema()
            logical_digest = enrichment_logical_digest(
                selected_attempt_output_sha256=selected_output.output_sha256,
                request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
                target_schema_sha256=enriched_schema.sha256,
                enrichment_policy_version=self._execution_policy.enrichment_policy_version,
            )
            enriched_output = self._enricher.enrich(
                input_plan=input_plan,
                reference_catalog=reference_catalog,
                parsed_claims=parsed_claims,
                selected_attempt=selected_output,
                authority=authority,
                enriched_output_schema=enriched_schema,
                enrichment_policy_version=self._execution_policy.enrichment_policy_version,
                artifact_id=_stable_uuid("orchestrator-enrichment", logical_digest),
                created_at=created_at,
            )
            state["enriched_output"] = enriched_output
        except (ProviderClaimEnrichmentError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "ENRICHMENT_REJECTED",
                    exc,
                ),
            )

        try:
            output_decision, hypotheses = self._projector.project(
                context=context,
                enriched_output=enriched_output,
                interval=window.interval,
            )
            state["output_decision"] = output_decision
            state["hypotheses"] = hypotheses
        except (CanonicalOfflineConfigurationError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.OUTPUT_ADMISSION,
                    "OUTPUT_ADMISSION_REJECTED",
                    exc,
                ),
            )

        if not hypotheses:
            return finish(CanonicalOfflineRunStatus.NO_EVENTS)

        try:
            identity_result = self._identity_registry.assign_batch(
                admitted_context=context,
                hypotheses=hypotheses,
                enriched_outputs=(enriched_output,),
                decided_at=_timestamp(self._clock),
            )
            state["identity_result"] = identity_result
        except EventIdentityRegistryError as exc:
            return finish(
                CanonicalOfflineRunStatus.IDENTITY_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.IDENTITY,
                    "EVENT_IDENTITY_ASSIGNMENT_FAILED",
                    exc,
                ),
            )
        return finish(CanonicalOfflineRunStatus.SUCCEEDED)

    def _validate_configuration(self) -> None:
        policy = self._inference_policy
        execution = self._execution_policy
        if policy.task is not VisionTask.FUSION_ADJUDICATION:
            raise CanonicalOfflineConfigurationError(
                "canonical pipeline requires FUSION_ADJUDICATION policy"
            )
        if policy.provider != self._adapter.provider:
            raise CanonicalOfflineConfigurationError(
                "inference policy provider does not match offline adapter"
            )
        if policy.output_schema.schema_id != PROVIDER_CLAIM_SCHEMA_ID:
            raise CanonicalOfflineConfigurationError(
                "inference policy must pin the provider-claim schema"
            )
        enriched_schema = policy.enriched_output_schema
        if enriched_schema is None or enriched_schema.schema_id != ENRICHED_OUTPUT_SCHEMA_ID:
            raise CanonicalOfflineConfigurationError(
                "inference policy must pin the enriched-output schema"
            )
        if policy.output_schema.sha256 == enriched_schema.sha256:
            raise CanonicalOfflineConfigurationError(
                "provider and enriched schemas must remain distinct"
            )
        self._schema_registry.resolve_exact(_schema_ref(policy.output_schema))
        self._schema_registry.resolve_exact(_schema_ref(enriched_schema))
        if self._adapter.parser.schema_registry is not self._schema_registry:
            raise CanonicalOfflineConfigurationError(
                "offline parser and pipeline must share one schema registry"
            )
        if self._adapter.parser.parser_version != execution.parser_version:
            raise CanonicalOfflineConfigurationError(
                "parser version does not match execution policy"
            )
        if self._identity_registry.output_admission_policy != execution.output_admission_policy:
            raise CanonicalOfflineConfigurationError(
                "identity registry output policy does not match execution policy"
            )
        rendering_policy = self._input_preparer.policy
        if (
            rendering_policy.reduction_policy != execution.reduction_policy
            or rendering_policy.reduction_policy_version != execution.reduction_policy_version
            or self._package_builder.reduction_policy_version != execution.reduction_policy_version
        ):
            raise CanonicalOfflineConfigurationError(
                "package, rendering, and execution reduction policies differ"
            )
        _require_canonical_uuid(policy.prompt_artifact_id, "prompt_artifact_id")

    def _required_enriched_schema(self) -> JsonSchemaRef:
        schema = self._inference_policy.enriched_output_schema
        if schema is None:
            raise CanonicalOfflineConfigurationError("enriched output schema is missing")
        return schema


def _validate_materialized_chain(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalRootWindow,
    lineage: PackageLineage,
    planned_parts: tuple[IntervalPart, ...],
    materialized: tuple[MaterializedTemporalPackage, ...],
) -> None:
    if len(materialized) != len(planned_parts) or not materialized:
        raise CanonicalOfflineConfigurationError(
            "materialized package count does not match planned parts"
        )
    expected_lineage = PackageLineage(
        source_content_sha256=context.source_content_sha256,
        window_semantic_sha256=window.semantic_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        sampling_plan_sha256=lineage.sampling_plan_sha256,
    )
    if lineage != expected_lineage:
        raise CanonicalOfflineConfigurationError(
            "materialization lineage does not match context and root window"
        )
    for ordinal, (planned, output) in enumerate(zip(planned_parts, materialized, strict=True)):
        package = output.package
        if (
            package.part.ordinal != ordinal
            or package.part.part_count != len(planned_parts)
            or package.part.requested_interval != planned.requested_interval
            or package.part.effective_interval != planned.effective_interval
            or package.part.overlap_before_ns != planned.overlap_before_ns
            or package.part.overlap_after_ns != planned.overlap_after_ns
        ):
            raise CanonicalOfflineConfigurationError(
                "materialized package coordinates differ from the planned part"
            )
        if (
            package.window_id != window.window_id
            or package.mcap_id != context.ready_manifest.mcap_id
            or package.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
            or package.alignment_id != context.alignment_manifest.alignment_id
            or package.lineage != lineage
            or package.sampling_plan_sha256 != lineage.sampling_plan_sha256
        ):
            raise CanonicalOfflineConfigurationError(
                "materialized package authority binding is inconsistent"
            )
        if (
            output.manifest_bytes != canonical_json_bytes(package)
            or output.package_manifest_sha256 != exact_bytes_sha256(output.manifest_bytes)
            or output.package_ref.package_id != package.package_id
            or output.package_ref.package_semantic_content_sha256 != package.semantic_content_sha256
            or output.package_ref.package_manifest_sha256 != output.package_manifest_sha256
        ):
            raise CanonicalOfflineConfigurationError(
                "materialized package exact-byte identity is inconsistent"
            )


def _validate_package_set_chain(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalRootWindow,
    lineage: PackageLineage,
    package_set: TemporalPackageSet,
    materialized: tuple[MaterializedTemporalPackage, ...],
    reduction_policy_version: str,
) -> None:
    if (
        package_set.mcap_id != context.ready_manifest.mcap_id
        or package_set.window_id != window.window_id
        or package_set.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
        or package_set.alignment_id != context.alignment_manifest.alignment_id
        or package_set.lineage != lineage
        or package_set.requested_start_ns != window.requested_interval.start_ns
        or package_set.requested_end_ns != window.requested_interval.end_ns
        or package_set.start_ns != window.interval.start_ns
        or package_set.end_ns != window.interval.end_ns
        or package_set.reduction_policy_version != reduction_policy_version
    ):
        raise CanonicalOfflineConfigurationError(
            "package set does not match context, window, lineage, and reduction policy"
        )
    expected = tuple(
        (
            item.package.package_id,
            item.package.part.ordinal,
            item.package.semantic_content_sha256,
            item.package_manifest_sha256,
        )
        for item in materialized
    )
    actual = tuple(
        (
            member.package_id,
            member.ordinal,
            member.package_semantic_content_sha256,
            member.package_manifest_sha256,
        )
        for member in package_set.members
    )
    if actual != expected:
        raise CanonicalOfflineConfigurationError(
            "package set members differ from exact materialized packages"
        )


def _validated_capabilities(
    capabilities: ModelCapabilities,
    *,
    inference_policy: InferencePolicy,
    input_preparer: InputPlanPreparer,
) -> ModelCapabilities:
    try:
        result = ModelCapabilities.model_validate(
            capabilities.model_dump(mode="python"), strict=True
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CanonicalOfflineConfigurationError(
            "offline capability snapshot failed strict validation"
        ) from exc
    if (
        result.provider != inference_policy.provider
        or result.model_name != inference_policy.model_name
        or result.model_version != inference_policy.model_version
        or inference_policy.task not in result.supported_tasks
    ):
        raise CanonicalOfflineConfigurationError(
            "capability snapshot does not match the pinned inference policy"
        )
    if not result.supports_json_schema or not result.supports_provider_idempotency:
        raise CanonicalOfflineConfigurationError(
            "canonical retry path requires schema and provider idempotency support"
        )
    required_media = set(inference_policy.required_media_types)
    rendering_media = set(input_preparer.policy.accepted_media_types)
    accepted_media = set(result.accepted_media_types)
    if not required_media.issubset(accepted_media) or not rendering_media.issubset(accepted_media):
        raise CanonicalOfflineConfigurationError(
            "capability media types do not cover policy and rendering requirements"
        )
    return result


def _rendered_prompt_bytes(
    *,
    inference_policy: InferencePolicy,
    request_catalog_sha256: str,
    token_policy_version: str,
    entries: tuple[ProviderReferenceCatalogEntry, ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "protocol": "robata-provider-claim-v1",
            "task": inference_policy.task.value,
            "prompt_artifact": {
                "version": inference_policy.prompt_version,
                "artifact_id": inference_policy.prompt_artifact_id,
                "sha256": inference_policy.prompt_sha256,
            },
            "request_catalog_sha256": request_catalog_sha256,
            "token_policy_version": token_policy_version,
            "evidence_catalog": [entry.model_dump(mode="json") for entry in entries],
            "provider_response_schema": inference_policy.output_schema.model_dump(mode="json"),
        }
    )


def _validate_input_plan_chain(
    *,
    package_set: TemporalPackageSet,
    materialized: tuple[MaterializedTemporalPackage, ...],
    input_plan: InferenceInputPlan,
    reference_catalog: ProviderReferenceCatalog,
    inference_policy: InferencePolicy,
    execution_policy: CanonicalOfflineExecutionPolicy,
    capabilities: ModelCapabilities,
) -> None:
    expected_packages = tuple(
        (
            item.package.package_id,
            item.package.part.ordinal,
            item.package.semantic_content_sha256,
            item.package_manifest_sha256,
        )
        for item in materialized
    )
    subject_packages = tuple(
        (
            item.package_id,
            item.ordinal,
            item.semantic_content_sha256,
            item.manifest_bytes_sha256,
        )
        for item in input_plan.subject.packages
    )
    set_packages = tuple(
        (
            item.package_id,
            item.ordinal,
            item.package_semantic_content_sha256,
            item.package_manifest_sha256,
        )
        for item in package_set.members
    )
    if subject_packages != expected_packages or set_packages != expected_packages:
        raise CanonicalOfflineConfigurationError(
            "input plan subject does not exactly match package-set members"
        )
    target = input_plan.target
    if (
        target.provider != inference_policy.provider
        or target.model_name != inference_policy.model_name
        or target.model_version != inference_policy.model_version
        or target.adapter_version != inference_policy.adapter_version
        or target.capability_snapshot_id != capabilities.snapshot_id
        or target.capability_snapshot_sha256 != capabilities.snapshot_digest
    ):
        raise CanonicalOfflineConfigurationError(
            "input plan target differs from pinned policy or capability snapshot"
        )
    expected_limits = (
        capabilities.max_images_per_request,
        capabilities.max_pixels_per_image,
        capabilities.max_payload_bytes,
        capabilities.max_input_tokens,
    )
    actual_limits = (
        input_plan.applicable_limits.max_images_per_request,
        input_plan.applicable_limits.max_pixels_per_image,
        input_plan.applicable_limits.max_payload_bytes_per_request,
        input_plan.applicable_limits.max_input_tokens_per_request,
    )
    if actual_limits != expected_limits:
        raise CanonicalOfflineConfigurationError(
            "input plan limits differ from the capability snapshot"
        )
    if any(
        item.artifact.media_type not in capabilities.accepted_media_types
        for item in input_plan.rendered_items
    ):
        raise CanonicalOfflineConfigurationError(
            "input plan contains media outside the capability snapshot"
        )
    if (
        input_plan.call_plan.reduction_policy != execution_policy.reduction_policy
        or input_plan.call_plan.reduction_policy_version
        != execution_policy.reduction_policy_version
        or input_plan.prompt_output.prompt_version != inference_policy.prompt_version
        or input_plan.prompt_output.prompt_sha256 != inference_policy.prompt_sha256
        or input_plan.prompt_output.provider_response_schema_sha256
        != inference_policy.output_schema.sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "input plan prompt, schema, or reduction policy binding differs"
        )
    if (
        reference_catalog.input_plan_id != input_plan.input_plan_id
        or reference_catalog.input_plan_semantic_sha256 != input_plan.semantic_sha256
        or reference_catalog.request_catalog_id != input_plan.request_catalog.request_catalog_id
        or reference_catalog.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "reference catalog does not bind the finalized input plan"
        )


def _package_inputs(package_set: TemporalPackageSet) -> tuple[PackageInput, ...]:
    return tuple(
        PackageInput(
            package_id=member.package_id,
            package_semantic_content_sha256=member.package_semantic_content_sha256,
            package_manifest_sha256=member.package_manifest_sha256,
            role="TEMPORAL_EVIDENCE",
            ordinal=member.ordinal,
        )
        for member in package_set.members
    )


def _terminal_raw_artifact_id(terminal: ModelInference) -> str:
    raw = terminal.raw_output
    artifact_id = raw.get("artifact_id") if raw is not None else None
    if not isinstance(artifact_id, str) or not artifact_id:
        raise CanonicalOfflineConfigurationError(
            "selected terminal has no valid raw response artifact reference"
        )
    return artifact_id


def _canonical_error(
    stage: CanonicalOfflineStage,
    code: object,
    detail: object,
) -> CanonicalOfflineError:
    code_text = str(code).strip() or "UNKNOWN"
    detail_text = str(detail).strip() or type(detail).__name__
    return CanonicalOfflineError(
        schema_version="1.0",
        stage=stage,
        code=code_text[:512],
        detail=detail_text[:512],
    )


def _require_canonical_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CanonicalOfflineConfigurationError(f"{label} must be a UUID") from exc
    if str(parsed) != value:
        raise CanonicalOfflineConfigurationError(f"{label} must use canonical lowercase UUID text")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _output_decision(
    *,
    recording_identity: str,
    source_ref: PlatformEnrichedOutputReference,
    policy: ProductionOutputAdmissionPolicyRef,
    decision: Literal["PRODUCTION_ADMITTED", "NO_EVENTS"],
    admitted_claim_ordinals: tuple[int, ...],
    reason_code: str,
    production_output_admission: ProductionOutputAdmissionProof | None,
) -> CanonicalOutputAdmissionDecision:
    values = {
        "decision": decision,
        "recording_identity": recording_identity,
        "source_enrichment": source_ref,
        "policy_version": policy.version,
        "policy_sha256": policy.semantic_sha256,
        "admitted_claim_ordinals": admitted_claim_ordinals,
        "reason_code": reason_code,
        "production_output_admission": production_output_admission,
    }
    digest = semantic_sha256(
        {
            **values,
            "source_enrichment": source_ref.model_dump(mode="json"),
            "admitted_claim_ordinals": list(admitted_claim_ordinals),
            "production_output_admission": (
                None
                if production_output_admission is None
                else production_output_admission.model_dump(mode="json")
            ),
        }
    )
    return CanonicalOutputAdmissionDecision(
        schema_version="1.0",
        decision_id=_stable_uuid("canonical-output-admission", digest),
        decision=decision,
        semantic_sha256=digest,
        recording_identity=recording_identity,
        source_enrichment=source_ref,
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


def _strict_context(context: AdmittedRecordingContextV2) -> AdmittedRecordingContextV2:
    if not isinstance(context, AdmittedRecordingContextV2):
        raise CanonicalOfflineConfigurationError("expected AdmittedRecordingContextV2")
    try:
        return AdmittedRecordingContextV2.model_validate(
            context.model_dump(mode="python"), strict=True
        )
    except ValueError as exc:
        raise CanonicalOfflineConfigurationError("V2 admission context failed validation") from exc


def _schema_ref(ref: JsonSchemaRef) -> SchemaRef:
    return SchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _json_schema_ref(ref: SchemaRef) -> JsonSchemaRef:
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "CanonicalOfflineConfigurationError",
    "CanonicalOfflineError",
    "CanonicalOfflineExecutionPolicy",
    "CanonicalOfflinePipeline",
    "CanonicalOfflineRunResult",
    "CanonicalOfflineRunStatus",
    "CanonicalOfflineStage",
    "CanonicalOutputAdmissionDecision",
    "CanonicalRootWindow",
    "FusionEventHypothesisProjector",
    "canonical_execution_policy_projection",
    "canonical_lineage",
    "canonical_root_window_projection_values",
]
