"""Canonical, transport-free vertical slice for the registered V2 path.

The service in this module is deliberately local and non-promotional.  It
connects the registered admission evidence through temporal materialization,
provider-specific planning, an all-part inference barrier, strict raw response
parsing, enrichment, deterministic fusion reduction, and recording-scoped event
identity assignment.
No network-capable adapter is accepted by the coordinator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical_run_membership import (
    CanonicalProcessingRunContext,
    CanonicalProcessingRunPrimaryStatus,
    CanonicalProcessingRunRecord,
    CanonicalRunMembershipError,
    CanonicalRunMembershipJournal,
    canonical_first_work_item_id,
)
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import (
    LogicalNode,
    NodeLogicalKey,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    Rfc3339Timestamp,
    RunNodeDisposition,
    RunNodeRole,
    logical_node_from_semantic_digest,
)
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
    platform_enriched_output_logical_projection,
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
    EnrichedProviderClaim,
    EnrichmentAuthorityContext,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderClaimEnricher,
    ProviderClaimEnrichmentError,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderReferenceCatalog,
    ProviderReferenceCatalogEntry,
    ProviderTaskClaim,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
    enrichment_logical_digest,
)
from robata.inference.evidence import (
    InferenceEvidenceStore,
    InferenceEvidenceStoreError,
    InMemoryInferenceEvidenceStore,
)
from robata.inference.input_plan import (
    InferenceCallPart,
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
    inference_attempt_selection_digest,
)
from robata.inference.offline_fixture import (
    OfflineFixtureVisionAdapter,
    RawProviderBytesStoreError,
    StrictProviderClaimParseError,
)
from robata.inference.orchestrator import (
    InferenceLedger,
    InferenceLedgerError,
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
from robata.ports.logical_node_registry import LogicalNodeRegistry, LogicalNodeRegistryError
from robata.queue.barrier import BarrierCoordinator, InMemoryBarrierStorage
from robata.queue.stage import StageStatus
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

CANONICAL_OFFLINE_PIPELINE_VERSION = "canonical-offline-v1"


class CanonicalOfflineConfigurationError(ValueError):
    """The configured local vertical slice cannot satisfy its contracts."""


class _CanonicalRunMembershipPublicationError(RuntimeError):
    """A typed node or its immutable run attachment could not be published."""


class CanonicalOfflineRunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    NO_EVENTS = "NO_EVENTS"
    ABSTAINED = "ABSTAINED"
    INCOMPLETE = "INCOMPLETE"
    MATERIALIZATION_FAILED = "MATERIALIZATION_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    IDENTITY_FAILED = "IDENTITY_FAILED"
    RUN_MEMBERSHIP_FAILED = "RUN_MEMBERSHIP_FAILED"
    CONFIGURATION_FAILED = "CONFIGURATION_FAILED"


class CanonicalOfflineStage(StrEnum):
    ADMISSION = "ADMISSION"
    WINDOW = "WINDOW"
    MATERIALIZATION = "MATERIALIZATION"
    PREPARATION = "PREPARATION"
    INFERENCE = "INFERENCE"
    PARSING = "PARSING"
    ENRICHMENT = "ENRICHMENT"
    REDUCTION = "REDUCTION"
    OUTPUT_ADMISSION = "OUTPUT_ADMISSION"
    IDENTITY = "IDENTITY"
    RUN_MEMBERSHIP = "RUN_MEMBERSHIP"


class CanonicalOfflineError(StrictModel):
    schema_version: Literal["1.0"]
    stage: CanonicalOfflineStage
    code: NonEmptyString
    detail: NonEmptyString


class CanonicalOfflinePartStatus(StrEnum):
    ENRICHED = "ENRICHED"
    TERMINAL_FAILED = "TERMINAL_FAILED"
    POST_SELECTION_INVALID = "POST_SELECTION_INVALID"


class CanonicalOfflinePartResult(StrictModel):
    """Exact per-call-part lineage retained by the local conformance runner."""

    schema_version: Literal["1.0"]
    part_ordinal: NonNegativeInt
    part_count: PositiveInt
    part_semantic_sha256: Sha256Digest
    status: CanonicalOfflinePartStatus
    orchestration_attempt_count: PositiveInt
    terminal: ModelInference
    selection: InferenceAttemptSelection | None
    completion: InferenceCallPartCompletion
    raw_response: RawProviderResponseArtifact | None
    parsed_claims: ParsedProviderClaimArtifact | None
    selected_output: SelectedAttemptOutput | None
    enriched_output: OrchestratorEnrichedOutput | None
    error: CanonicalOfflineError | None

    @model_validator(mode="after")
    def validate_part_result(self) -> Self:
        terminal = self.terminal
        if (
            terminal.input_plan_part_ordinal != self.part_ordinal
            or terminal.input_plan_part_count != self.part_count
            or terminal.input_plan_part_semantic_sha256 != self.part_semantic_sha256
            or self.completion.part_ordinal != self.part_ordinal
            or self.completion.part_count != self.part_count
            or self.completion.part_semantic_sha256 != self.part_semantic_sha256
            or self.completion.inference_id != terminal.inference_id
            or self.completion.logical_invocation_id != terminal.logical_invocation_id
            or self.completion.status is not terminal.status
        ):
            raise ValueError("part result does not match its terminal barrier member")

        artifacts = (self.raw_response, self.parsed_claims, self.selected_output)
        if any(item is not None for item in artifacts):
            if any(item is None for item in artifacts):
                raise ValueError("part raw, parsed, and selected artifacts are indivisible")
            assert self.raw_response is not None
            assert self.parsed_claims is not None
            assert self.selected_output is not None
            if (
                self.raw_response.inference_id != terminal.inference_id
                or self.parsed_claims.raw_response != self.raw_response
                or self.selected_output.inference_id != terminal.inference_id
                or self.selected_output.raw_response_artifact_id != self.raw_response.artifact_id
                or self.selected_output.parsed_claim_artifact_id != self.parsed_claims.artifact_id
            ):
                raise ValueError("part artifact lineage is inconsistent")
        if self.enriched_output is not None and (
            self.selected_output is None
            or self.enriched_output.selected_attempt != self.selected_output
            or self.enriched_output.authority.inference_id != terminal.inference_id
            or self.enriched_output.authority.logical_invocation_id
            != terminal.logical_invocation_id
        ):
            raise ValueError("part enrichment lineage is inconsistent")

        succeeded = terminal.status is InferenceStatus.SUCCEEDED
        if succeeded:
            if (
                self.selection is None
                or self.selection.inference_id != terminal.inference_id
                or self.selection.logical_invocation_id != terminal.logical_invocation_id
                or self.completion.selection_id != self.selection.selection_id
            ):
                raise ValueError("successful part requires its exact selection")
        elif (
            self.selection is not None
            or any(item is not None for item in artifacts)
            or (self.enriched_output is not None)
        ):
            raise ValueError("failed part cannot carry selected output lineage")

        if self.status is CanonicalOfflinePartStatus.ENRICHED:
            if not succeeded or self.enriched_output is None or self.error is not None:
                raise ValueError("ENRICHED part requires complete successful lineage")
        elif self.status is CanonicalOfflinePartStatus.TERMINAL_FAILED:
            if succeeded or self.error is None:
                raise ValueError("TERMINAL_FAILED part requires a failed terminal and error")
        elif not succeeded or self.enriched_output is not None or self.error is None:
            raise ValueError("POST_SELECTION_INVALID part has an inconsistent shape")
        return self


class CanonicalFusionPartSource(StrictModel):
    part_ordinal: NonNegativeInt
    part_semantic_sha256: Sha256Digest
    completion_id: OpaqueUuid
    inference_id: OpaqueUuid
    selected_attempt_output_sha256: Sha256Digest
    enrichment: PlatformEnrichedOutputReference
    abstained: bool


class CanonicalFusionClaimSource(StrictModel):
    part_ordinal: NonNegativeInt
    source_claim_ordinal: NonNegativeInt
    source_claim_id: OpaqueUuid
    enrichment_logical_key: NodeLogicalKey


class CanonicalReducedFusionClaim(StrictModel):
    fusion_output_ordinal: NonNegativeInt
    claim_semantic_sha256: Sha256Digest
    representative: EnrichedProviderClaim
    sources: tuple[CanonicalFusionClaimSource, ...]

    @model_validator(mode="after")
    def validate_reduced_claim(self) -> Self:
        if self.representative.kind is not ProviderClaimKind.FUSION_HYPOTHESIS:
            raise ValueError("fusion reduction accepts only fusion hypotheses")
        expected_sources = tuple(
            sorted(
                self.sources,
                key=lambda item: (
                    item.part_ordinal,
                    item.source_claim_ordinal,
                    item.source_claim_id,
                ),
            )
        )
        if not self.sources or self.sources != expected_sources:
            raise ValueError("fusion claim sources must be nonempty and canonical")
        source_keys = tuple(
            (item.part_ordinal, item.source_claim_ordinal, item.source_claim_id)
            for item in self.sources
        )
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("fusion claim sources must be unique")
        if self.claim_semantic_sha256 != _fusion_claim_reduction_digest(self.representative):
            raise ValueError("fusion claim semantic digest is inconsistent")
        return self


class CanonicalFusionReduction(StrictModel):
    """Ephemeral reduced fusion view; it never impersonates a selected attempt."""

    schema_version: Literal["1.0"]
    reduction_id: OpaqueUuid
    reduction_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    input_plan_semantic_sha256: Sha256Digest
    barrier_reduction_id: OpaqueUuid
    barrier_reduction_semantic_sha256: Sha256Digest
    reduction_policy: NonEmptyString
    reduction_policy_version: SchemaVersion
    outcome: Literal["CLAIMS", "NO_SURVIVING_EVENTS", "ALL_PARTS_ABSTAINED"]
    parts: tuple[CanonicalFusionPartSource, ...]
    claims: tuple[CanonicalReducedFusionClaim, ...]
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_reduction(self) -> Self:
        if tuple(item.part_ordinal for item in self.parts) != tuple(range(len(self.parts))):
            raise ValueError("fusion reduction parts must be complete and ordered")
        enrichment_keys = tuple(item.enrichment.enrichment_logical_key for item in self.parts)
        if len(set(enrichment_keys)) != len(enrichment_keys):
            raise ValueError("fusion reduction enrichments must be unique")
        if tuple(item.fusion_output_ordinal for item in self.claims) != tuple(
            range(len(self.claims))
        ):
            raise ValueError("reduced fusion ordinals must be contiguous from zero")
        part_by_ordinal = {item.part_ordinal: item for item in self.parts}
        for claim in self.claims:
            for source in claim.sources:
                part = part_by_ordinal.get(source.part_ordinal)
                if (
                    part is None
                    or source.enrichment_logical_key != part.enrichment.enrichment_logical_key
                ):
                    raise ValueError("reduced claim source is outside the part manifest")
        if self.outcome == "ALL_PARTS_ABSTAINED":
            if self.claims or not self.parts or not all(item.abstained for item in self.parts):
                raise ValueError("all-parts abstention has an inconsistent reduction shape")
        elif self.outcome == "NO_SURVIVING_EVENTS":
            if self.claims or not self.parts or any(item.abstained for item in self.parts):
                raise ValueError("empty fusion reduction has an inconsistent shape")
        elif not self.claims or any(item.abstained for item in self.parts):
            raise ValueError("claim reduction requires every required part to complete")
        expected = semantic_sha256(canonical_fusion_reduction_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("fusion reduction semantic digest is inconsistent")
        if self.reduction_logical_key != f"fusion-reduction:{expected}":
            raise ValueError("fusion reduction logical key is inconsistent")
        if self.reduction_id != _stable_uuid("canonical-fusion-reduction", expected):
            raise ValueError("fusion reduction ID is inconsistent")
        return self


def canonical_fusion_reduction_projection(
    reduction: CanonicalFusionReduction,
) -> dict[str, object]:
    return _canonical_fusion_reduction_projection_values(
        schema_version=reduction.schema_version,
        input_plan_semantic_sha256=reduction.input_plan_semantic_sha256,
        barrier_reduction_semantic_sha256=reduction.barrier_reduction_semantic_sha256,
        reduction_policy=reduction.reduction_policy,
        reduction_policy_version=reduction.reduction_policy_version,
        outcome=reduction.outcome,
        parts=reduction.parts,
        claims=reduction.claims,
    )


def _canonical_fusion_reduction_projection_values(
    *,
    schema_version: str,
    input_plan_semantic_sha256: str,
    barrier_reduction_semantic_sha256: str,
    reduction_policy: str,
    reduction_policy_version: str,
    outcome: str,
    parts: Sequence[CanonicalFusionPartSource],
    claims: Sequence[CanonicalReducedFusionClaim],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "input_plan_semantic_sha256": input_plan_semantic_sha256,
        "barrier_reduction_semantic_sha256": barrier_reduction_semantic_sha256,
        "reduction_policy": reduction_policy,
        "reduction_policy_version": reduction_policy_version,
        "outcome": outcome,
        "parts": [
            {
                "part_ordinal": part.part_ordinal,
                "part_semantic_sha256": part.part_semantic_sha256,
                "selected_attempt_output_sha256": part.selected_attempt_output_sha256,
                "enrichment": platform_enriched_output_logical_projection(part.enrichment),
                "abstained": part.abstained,
            }
            for part in parts
        ],
        "claims": [
            {
                "fusion_output_ordinal": claim.fusion_output_ordinal,
                "claim_semantic_sha256": claim.claim_semantic_sha256,
                "representative": _fusion_claim_reduction_projection(claim.representative),
                "sources": [
                    {
                        "part_ordinal": item.part_ordinal,
                        "source_claim_ordinal": item.source_claim_ordinal,
                        "enrichment_logical_key": item.enrichment_logical_key,
                    }
                    for item in claim.sources
                ],
            }
            for claim in claims
        ],
    }


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
    decision: Literal["PRODUCTION_ADMITTED", "NO_EVENTS", "ABSTAINED"]
    semantic_sha256: Sha256Digest
    recording_identity: Sha256Digest
    source_enrichments: tuple[PlatformEnrichedOutputReference, ...]
    fusion_reduction_logical_key: NodeLogicalKey
    fusion_reduction_semantic_sha256: Sha256Digest
    policy_version: SchemaVersion
    policy_sha256: Sha256Digest
    admitted_claim_ordinals: tuple[NonNegativeInt, ...]
    reason_code: NonEmptyString
    production_output_admission: ProductionOutputAdmissionProof | None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
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
                    "non-production output decisions cannot carry claims or a production proof"
                )
        elif self.production_output_admission is None:
            raise ValueError("PRODUCTION_ADMITTED requires an output admission proof")
        else:
            proof = self.production_output_admission
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
        if self.decision_id != _stable_uuid("canonical-output-admission", expected):
            raise ValueError("output decision ID is inconsistent")
        return self

    @property
    def source_enrichment(self) -> PlatformEnrichedOutputReference:
        """Compatibility accessor for callers restricted to one call part."""

        if len(self.source_enrichments) != 1:
            raise ValueError("multi-part decisions have more than one source enrichment")
        return self.source_enrichments[0]


def canonical_output_decision_projection(
    decision: CanonicalOutputAdmissionDecision,
) -> dict[str, object]:
    return _canonical_output_decision_projection_values(
        decision=decision.decision,
        recording_identity=decision.recording_identity,
        source_enrichments=decision.source_enrichments,
        fusion_reduction_logical_key=decision.fusion_reduction_logical_key,
        fusion_reduction_semantic_sha256=decision.fusion_reduction_semantic_sha256,
        policy_version=decision.policy_version,
        policy_sha256=decision.policy_sha256,
        admitted_claim_ordinals=decision.admitted_claim_ordinals,
        reason_code=decision.reason_code,
        production_output_admission=decision.production_output_admission,
    )


def _canonical_output_decision_projection_values(
    *,
    decision: str,
    recording_identity: str,
    source_enrichments: Sequence[PlatformEnrichedOutputReference],
    fusion_reduction_logical_key: str,
    fusion_reduction_semantic_sha256: str,
    policy_version: str,
    policy_sha256: str,
    admitted_claim_ordinals: Sequence[int],
    reason_code: str,
    production_output_admission: ProductionOutputAdmissionProof | None,
) -> dict[str, object]:
    return {
        "decision": decision,
        "recording_identity": recording_identity,
        "source_enrichments": [
            platform_enriched_output_logical_projection(item) for item in source_enrichments
        ],
        "fusion_reduction_logical_key": fusion_reduction_logical_key,
        "fusion_reduction_semantic_sha256": fusion_reduction_semantic_sha256,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "admitted_claim_ordinals": list(admitted_claim_ordinals),
        "reason_code": reason_code,
        "production_output_admission_semantic_sha256": (
            None
            if production_output_admission is None
            else production_output_admission.semantic_sha256
        ),
    }


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
        proof = ProductionOutputAdmissionProof.create(
            recording_identity=context.recording_identity,
            source_enrichments=actual_refs,
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
            decision="PRODUCTION_ADMITTED",
            admitted_claim_ordinals=tuple(item.fusion_output_ordinal for item in hypotheses),
            reason_code="FUSION_REDUCTION_VALIDATED",
            production_output_admission=proof,
        )
        return decision, hypotheses


class CanonicalOfflineRunResult(StrictModel):
    """Inspectable local result; this is not a registered persistence schema."""

    schema_version: Literal["1.0"]
    run_id: OpaqueUuid
    processing_run: CanonicalProcessingRunRecord
    run_memberships: tuple[ProcessingRunNodeMembership, ...]
    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    execution_policy_sha256: Sha256Digest
    status: CanonicalOfflineRunStatus
    window: CanonicalRootWindow | None
    materialized_package_ids: tuple[OpaqueUuid, ...]
    package_set: TemporalPackageSet | None
    input_plan: InferenceInputPlan | None
    reference_catalog: ProviderReferenceCatalog | None
    part_results: tuple[CanonicalOfflinePartResult, ...]
    barrier_reduction: InferenceCallReduction | None
    fusion_reduction: CanonicalFusionReduction | None
    # Single-part compatibility fields. Multi-part truth lives only in part_results.
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
        expected_run_status = CanonicalProcessingRunPrimaryStatus(self.status.value)
        if (
            self.processing_run.run_id != self.run_id
            or self.processing_run.recording_identity != self.recording_identity
            or self.processing_run.mcap_id != self.mcap_id
            or self.processing_run.pipeline_version != CANONICAL_OFFLINE_PIPELINE_VERSION
            or self.processing_run.config_sha256 != self.execution_policy_sha256
            or self.processing_run.primary_status is not expected_run_status
            or self.processing_run.completed_at is None
        ):
            raise ValueError("run result does not match its terminal processing-run record")
        if self.window is not None and self.mcap_id != self.window.mcap_id:
            raise ValueError("run result does not match the root window MCAP")
        membership_identities = tuple(item.identity for item in self.run_memberships)
        if len(set(membership_identities)) != len(membership_identities):
            raise ValueError("run result contains duplicate node memberships")
        if any(item.run_id != self.run_id for item in self.run_memberships):
            raise ValueError("run result contains a membership from another run")
        completed_at = _rfc3339_datetime(self.processing_run.completed_at)
        started_at = _rfc3339_datetime(self.processing_run.started_at)
        if any(
            not started_at <= _rfc3339_datetime(item.attached_at) <= completed_at
            for item in self.run_memberships
        ):
            raise ValueError("run membership timestamps lie outside the processing run")
        if self.window is not None and self.window.recording_identity != self.recording_identity:
            raise ValueError("run window crosses recording scope")
        if self.package_set is not None:
            if self.window is None or self.package_set.mcap_id != self.mcap_id:
                raise ValueError("run package set does not match its root MCAP lineage")
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
        if self.part_results:
            if self.input_plan is None:
                raise ValueError("part results require an input plan")
            if tuple(item.part_ordinal for item in self.part_results) != tuple(
                range(len(self.part_results))
            ):
                raise ValueError("run part results must be an ordered prefix")
            for item in self.part_results:
                planned = self.input_plan.call_plan.parts[item.part_ordinal]
                if (
                    item.part_count != len(self.input_plan.call_plan.parts)
                    or item.part_semantic_sha256 != planned.part_semantic_sha256
                    or item.terminal.mcap_id != self.mcap_id
                    or item.terminal.input_config.get("canonical_execution_policy_sha256")
                    != self.execution_policy_sha256
                ):
                    raise ValueError("run part result differs from its input plan or run binding")
        if self.attempt_count != sum(
            item.orchestration_attempt_count for item in self.part_results
        ):
            raise ValueError("run attempt count must equal all per-part attempts")

        compatibility = (
            self.terminal,
            self.selection,
            self.raw_response,
            self.parsed_claims,
            self.selected_output,
            self.enriched_output,
        )
        if self.part_results and self.part_results[0].part_count == 1:
            part = self.part_results[0]
            if compatibility != (
                part.terminal,
                part.selection,
                part.raw_response,
                part.parsed_claims,
                part.selected_output,
                part.enriched_output,
            ):
                raise ValueError("single-part compatibility fields differ from part truth")
        elif any(item is not None for item in compatibility):
            raise ValueError("multi-part runs cannot expose ambiguous singular lineage")

        enriched_outputs = tuple(
            item.enriched_output for item in self.part_results if item.enriched_output is not None
        )
        enriched_refs = tuple(
            sorted(
                (PlatformEnrichedOutputReference.from_output(item) for item in enriched_outputs),
                key=lambda item: item.enrichment_logical_key,
            )
        )
        if self.fusion_reduction is not None and (
            self.barrier_reduction is None
            or self.input_plan is None
            or self.fusion_reduction.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
            or self.fusion_reduction.barrier_reduction_id != self.barrier_reduction.reduction_id
            or self.fusion_reduction.barrier_reduction_semantic_sha256
            != self.barrier_reduction.reduction_semantic_sha256
            or tuple(
                sorted(
                    (item.enrichment for item in self.fusion_reduction.parts),
                    key=lambda item: item.enrichment_logical_key,
                )
            )
            != enriched_refs
        ):
            raise ValueError("run fusion reduction lineage is inconsistent")
        if self.fusion_reduction is not None:
            for source, result in zip(
                self.fusion_reduction.parts,
                self.part_results,
                strict=True,
            ):
                enriched = result.enriched_output
                selected = result.selected_output
                if (
                    enriched is None
                    or selected is None
                    or source.part_semantic_sha256 != result.part_semantic_sha256
                    or source.completion_id != result.completion.completion_id
                    or source.inference_id != result.terminal.inference_id
                    or source.selected_attempt_output_sha256 != selected.output_sha256
                    or source.enrichment != PlatformEnrichedOutputReference.from_output(enriched)
                    or source.abstained != enriched.abstained
                ):
                    raise ValueError("run fusion part sources differ from per-part truth")
        if self.output_decision is not None and (
            self.output_decision.recording_identity != self.recording_identity
            or self.fusion_reduction is None
            or self.output_decision.source_enrichments != enriched_refs
            or self.output_decision.fusion_reduction_semantic_sha256
            != self.fusion_reduction.semantic_sha256
            or self.output_decision.fusion_reduction_logical_key
            != self.fusion_reduction.reduction_logical_key
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
            CanonicalOfflineRunStatus.ABSTAINED,
        }
        expected_memberships = _canonical_result_membership_lineage(self)
        actual_membership_lineage = tuple(
            (item.node_type, item.node_logical_key, item.role) for item in self.run_memberships
        )
        expected_membership_lineage = tuple(
            (node.node_type, node.node_logical_key, role) for node, role in expected_memberships
        )
        if (
            actual_membership_lineage
            != expected_membership_lineage[: len(actual_membership_lineage)]
        ):
            raise ValueError("run memberships are not the exact ordered lineage prefix")
        for membership, (node, role) in zip(
            self.run_memberships,
            expected_memberships[: len(self.run_memberships)],
            strict=True,
        ):
            if (
                membership.first_work_item_id
                != canonical_first_work_item_id(
                    run_id=self.run_id,
                    node=node,
                    role=role,
                )
                or membership.attached_at != self.processing_run.started_at
                or membership.disposition
                not in {RunNodeDisposition.CREATED, RunNodeDisposition.REUSED}
            ):
                raise ValueError("run membership facts do not match the canonical attachment")
        if completed and not self.run_memberships:
            raise ValueError("completed run requires its complete nonempty membership lineage")
        if self.status is CanonicalOfflineRunStatus.RUN_MEMBERSHIP_FAILED:
            if self.identity_result is not None:
                raise ValueError("membership-failed run cannot carry a published identity result")
            if len(self.run_memberships) >= len(expected_memberships):
                raise ValueError(
                    "membership-failed run requires a strict unpublished lineage suffix"
                )
        elif len(self.run_memberships) != len(expected_memberships):
            raise ValueError("run requires its complete retained membership lineage")
        if completed and self.error is not None:
            raise ValueError("completed run cannot contain an error")
        if not completed and self.error is None:
            raise ValueError("non-completed run requires a structured error")
        if self.status is CanonicalOfflineRunStatus.SUCCEEDED:
            if (
                self.input_plan is None
                or len(self.part_results) != len(self.input_plan.call_plan.parts)
                or any(
                    item.status is not CanonicalOfflinePartStatus.ENRICHED
                    for item in self.part_results
                )
                or self.barrier_reduction is None
                or self.fusion_reduction is None
                or self.fusion_reduction.outcome != "CLAIMS"
                or self.output_decision is None
                or self.output_decision.decision != "PRODUCTION_ADMITTED"
                or not self.hypotheses
                or self.identity_result is None
            ):
                raise ValueError("successful run is missing admitted identity lineage")
        elif self.status is CanonicalOfflineRunStatus.ABSTAINED:
            if (
                self.input_plan is None
                or len(self.part_results) != len(self.input_plan.call_plan.parts)
                or any(
                    item.status is not CanonicalOfflinePartStatus.ENRICHED
                    or item.enriched_output is None
                    or not item.enriched_output.abstained
                    for item in self.part_results
                )
                or self.barrier_reduction is None
                or self.fusion_reduction is None
                or self.fusion_reduction.outcome != "ALL_PARTS_ABSTAINED"
                or self.output_decision is None
                or self.output_decision.decision != "ABSTAINED"
                or self.hypotheses
                or self.identity_result is not None
            ):
                raise ValueError("ABSTAINED run has an inconsistent terminal shape")
        elif self.status is CanonicalOfflineRunStatus.NO_EVENTS:
            if (
                self.input_plan is None
                or len(self.part_results) != len(self.input_plan.call_plan.parts)
                or any(
                    item.status is not CanonicalOfflinePartStatus.ENRICHED
                    for item in self.part_results
                )
                or self.barrier_reduction is None
                or self.fusion_reduction is None
                or self.fusion_reduction.outcome != "NO_SURVIVING_EVENTS"
                or self.output_decision is None
                or self.output_decision.decision != "NO_EVENTS"
                or self.hypotheses
                or self.identity_result is not None
            ):
                raise ValueError("NO_EVENTS run has an inconsistent terminal shape")
        elif self.status is CanonicalOfflineRunStatus.INCOMPLETE and (
            self.input_plan is None
            or len(self.part_results) != len(self.input_plan.call_plan.parts)
            or self.barrier_reduction is not None
            or self.fusion_reduction is not None
            or self.output_decision is not None
            or self.hypotheses
            or self.identity_result is not None
        ):
            raise ValueError("INCOMPLETE run cannot publish reduced output")
        return self


def _canonical_result_membership_lineage(
    result: CanonicalOfflineRunResult,
) -> tuple[tuple[LogicalNode, RunNodeRole], ...]:
    """Rebuild the canonical attachment order solely from retained result lineage."""

    entries: list[tuple[LogicalNode, RunNodeRole]] = []
    if result.window is not None:
        entries.append((canonical_root_window_logical_node(result.window), "ROOT_WINDOW"))
    if result.package_set is not None:
        entries.append((canonical_package_set_logical_node(result.package_set), "PACKAGE_SET"))
    if result.input_plan is not None:
        entries.append((canonical_input_plan_logical_node(result.input_plan), "INPUT_PLAN"))
        entries.extend(
            (canonical_call_part_logical_node(result.input_plan, part), "CALL_PART")
            for part in result.input_plan.call_plan.parts
        )
        entries.append((canonical_call_barrier_logical_node(result.input_plan), "CALL_BARRIER"))
    for part_result in result.part_results:
        if part_result.selection is not None:
            entries.append(
                (canonical_selection_logical_node(part_result.selection), "ATTEMPT_SELECTION")
            )
        if part_result.parsed_claims is not None:
            entries.append(
                (canonical_parsed_claim_logical_node(part_result.parsed_claims), "PARSED_CLAIM")
            )
        if part_result.selected_output is not None:
            entries.append(
                (
                    canonical_selected_output_logical_node(part_result.selected_output),
                    "SELECTED_OUTPUT",
                )
            )
        if part_result.enriched_output is not None:
            entries.append(
                (
                    canonical_enrichment_logical_node(part_result.enriched_output),
                    "ENRICHED_OUTPUT",
                )
            )
    if result.barrier_reduction is not None:
        entries.append(
            (
                canonical_call_reduction_logical_node(result.barrier_reduction),
                "CALL_REDUCTION",
            )
        )
    if result.fusion_reduction is not None:
        entries.append(
            (
                canonical_fusion_reduction_logical_node(result.fusion_reduction),
                "FUSION_REDUCTION",
            )
        )
    if result.output_decision is not None:
        entries.append(
            (
                canonical_output_decision_logical_node(result.output_decision),
                "OUTPUT_DECISION",
            )
        )
    entries.extend(
        (canonical_event_hypothesis_logical_node(item), "EVENT_HYPOTHESIS")
        for item in result.hypotheses
    )
    unique_entries: list[tuple[LogicalNode, RunNodeRole]] = []
    seen: set[tuple[str, str, str]] = set()
    for node, role in entries:
        identity = (node.node_type, node.node_logical_key, role)
        if identity not in seen:
            seen.add(identity)
            unique_entries.append((node, role))
    return tuple(unique_entries)


class _OrderedProviderClaimReducer:
    """Exact-duplicate reducer for ordered required call-part payloads."""

    def reduce(
        self,
        *,
        input_plan: InferenceInputPlan,
        ordered_completions: tuple[InferenceCallPartCompletion, ...],
    ) -> Mapping[str, object]:
        if len(ordered_completions) != len(input_plan.call_plan.parts):
            raise CanonicalOfflineConfigurationError(
                "provider reducer requires the complete call-part set"
            )
        payloads: list[ProviderClaimPayload] = []
        for part, completion in zip(input_plan.call_plan.parts, ordered_completions, strict=True):
            if (
                completion.part_ordinal != part.ordinal
                or completion.part_semantic_sha256 != part.part_semantic_sha256
                or completion.status is not InferenceStatus.SUCCEEDED
                or completion.normalized_output is None
            ):
                raise CanonicalOfflineConfigurationError(
                    "provider reducer received an invalid ordered completion"
                )
            payloads.append(
                ProviderClaimPayload.model_validate(
                    completion.normalized_output,
                    strict=False,
                )
            )
        return _reduce_provider_claim_payloads(tuple(payloads)).model_dump(mode="json")


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
        logical_node_registry: LogicalNodeRegistry,
        execution_policy: CanonicalOfflineExecutionPolicy,
        inference_ledger: InferenceLedger | None = None,
        evidence_store: InferenceEvidenceStore | None = None,
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
        if not callable(getattr(logical_node_registry, "attach_run_node", None)):
            raise TypeError("logical_node_registry must implement attach_run_node")
        if not isinstance(execution_policy, CanonicalOfflineExecutionPolicy):
            raise TypeError("execution_policy must be a CanonicalOfflineExecutionPolicy")
        if (inference_ledger is None) != (evidence_store is None):
            raise CanonicalOfflineConfigurationError(
                "durable inference evidence requires both ledger and evidence store"
            )
        if evidence_store is not None:
            if not isinstance(evidence_store, InferenceEvidenceStore):
                raise TypeError("evidence_store must implement InferenceEvidenceStore")
            ledger_identity: object = inference_ledger
            raw_store_identity: object = adapter.raw_store
            evidence_store_identity: object = evidence_store
            if (
                ledger_identity is not evidence_store_identity
                or raw_store_identity is not evidence_store_identity
            ):
                raise CanonicalOfflineConfigurationError(
                    "durable inference ledger, raw store, and evidence store must be one object"
                )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._package_builder = package_builder
        self._materializer = materializer
        self._input_preparer = input_preparer
        self._adapter = adapter
        self._inference_policy = inference_policy
        self._schema_registry = schema_registry
        self._identity_registry = identity_registry
        self._logical_node_registry = logical_node_registry
        self._execution_policy = execution_policy
        self._clock = clock or _utc_now
        self._validate_configuration()

        self._ledger = (
            inference_ledger if inference_ledger is not None else InMemoryInferenceLedger()
        )
        self._evidence_store = (
            evidence_store if evidence_store is not None else InMemoryInferenceEvidenceStore()
        )
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
            reducers={reducer_key: _OrderedProviderClaimReducer()},
        )
        self._enricher = ProviderClaimEnricher(schema_registry)
        self._projector = FusionEventHypothesisProjector(
            policy=execution_policy.output_admission_policy,
            projector_version=execution_policy.projector_policy_version,
        )

    @property
    def ledger(self) -> InferenceLedger:
        return self._ledger

    @property
    def evidence_store(self) -> InferenceEvidenceStore:
        return self._evidence_store

    @property
    def call_barrier_storage(self) -> InMemoryInferenceCallBarrierStorage:
        return self._call_barrier_storage

    @property
    def adapter(self) -> OfflineFixtureVisionAdapter:
        return self._adapter

    async def run(
        self,
        *,
        processing_run: CanonicalProcessingRunContext,
        admitted_context: AdmittedRecordingContextV2,
        requested_interval: NanosecondInterval,
        sampling_plan: SamplingPlan,
        frame_index: CanonicalSixCameraFrameIndex,
        artifact_resolver: FrameArtifactResolver,
        rendered_item_factory: RenderedItemFactory | None = None,
    ) -> CanonicalOfflineRunResult:
        """Execute one exact local run, retaining every admitted intermediate."""

        context = _strict_context(admitted_context)
        if not isinstance(processing_run, CanonicalProcessingRunContext):
            raise TypeError("processing_run must be a CanonicalProcessingRunContext")
        run_context = CanonicalProcessingRunContext.model_validate(
            processing_run.model_dump(mode="python"), strict=True
        )
        _validate_processing_run_binding(
            processing_run=run_context,
            admitted_context=context,
            execution_policy=self._execution_policy,
        )
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

        observed_at = _timestamp(self._clock)
        if _rfc3339_datetime(run_context.started_at) > _rfc3339_datetime(observed_at):
            raise CanonicalOfflineConfigurationError(
                "processing run started_at cannot be later than the pipeline clock"
            )
        created_at = run_context.started_at
        run_id = run_context.run_id
        journal = CanonicalRunMembershipJournal(
            context=run_context,
            registry=self._logical_node_registry,
        )
        infer_calls_before = self._adapter.infer_calls
        part_results_accumulator: list[CanonicalOfflinePartResult] = []
        state: dict[str, object] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "processing_run": journal.record,
            "run_memberships": (),
            "recording_identity": context.recording_identity,
            "mcap_id": context.ready_manifest.mcap_id,
            "execution_policy_sha256": self._execution_policy.semantic_sha256,
            "window": None,
            "materialized_package_ids": (),
            "package_set": None,
            "input_plan": None,
            "reference_catalog": None,
            "part_results": (),
            "barrier_reduction": None,
            "fusion_reduction": None,
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
            part_results = tuple(part_results_accumulator)
            state["part_results"] = part_results
            state["attempt_count"] = sum(item.orchestration_attempt_count for item in part_results)
            if len(part_results) == 1 and part_results[0].part_count == 1:
                only = part_results[0]
                state.update(
                    {
                        "terminal": only.terminal,
                        "selection": only.selection,
                        "raw_response": only.raw_response,
                        "parsed_claims": only.parsed_claims,
                        "selected_output": only.selected_output,
                        "enriched_output": only.enriched_output,
                    }
                )
            elif part_results:
                state.update(
                    {
                        "terminal": None,
                        "selection": None,
                        "raw_response": None,
                        "parsed_claims": None,
                        "selected_output": None,
                        "enriched_output": None,
                    }
                )
            completed_run = journal.complete(
                CanonicalProcessingRunPrimaryStatus(status.value),
            )
            return CanonicalOfflineRunResult.model_validate(
                {
                    **state,
                    "processing_run": completed_run,
                    "run_memberships": journal.memberships,
                    "status": status,
                    "adapter_infer_calls": self._adapter.infer_calls - infer_calls_before,
                    "network_call_count": 0,
                    "error": error,
                },
                strict=True,
            )

        def attach_nodes(
            entries: Sequence[tuple[LogicalNode, RunNodeRole]],
        ) -> None:
            try:
                for node, role in entries:
                    journal.attach(node, role, created_at)
            except (CanonicalRunMembershipError, LogicalNodeRegistryError) as exc:
                raise _CanonicalRunMembershipPublicationError(str(exc)) from exc

        def membership_failure(error: object) -> CanonicalOfflineRunResult:
            return finish(
                CanonicalOfflineRunStatus.RUN_MEMBERSHIP_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.RUN_MEMBERSHIP,
                    "RUN_NODE_PUBLICATION_FAILED",
                    error,
                ),
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
            attach_nodes(((canonical_root_window_logical_node(window), "ROOT_WINDOW"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

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
            attach_nodes(((canonical_package_set_logical_node(package_set), "PACKAGE_SET"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

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
        try:
            attach_nodes(((canonical_input_plan_logical_node(input_plan), "INPUT_PLAN"),))
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        package_inputs = _package_inputs(package_set)
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
        try:
            attach_nodes(
                (
                    *(
                        (
                            canonical_call_part_logical_node(input_plan, part),
                            "CALL_PART",
                        )
                        for part in input_plan.call_plan.parts
                    ),
                    (canonical_call_barrier_logical_node(input_plan), "CALL_BARRIER"),
                )
            )
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        return await self._execute_declared_call_plan(
            context=context,
            window=window,
            sampling_plan=sampling_plan,
            package_set=package_set,
            package_inputs=package_inputs,
            input_plan=input_plan,
            reference_catalog=reference_catalog,
            created_at=created_at,
            state=state,
            part_results=part_results_accumulator,
            finish=finish,
            attach_nodes=attach_nodes,
            membership_failure=membership_failure,
        )

    async def _orchestrate_call_part(
        self,
        *,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        part: InferenceCallPart,
    ) -> tuple[ModelInference, InferenceAttemptSelection | None, int]:
        terminal: ModelInference | None = None
        for attempt in range(1, self._execution_policy.max_attempts + 1):
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
                input_plan_part_ordinal=part.ordinal,
                input_config={
                    "canonical_execution_policy_sha256": (self._execution_policy.semantic_sha256)
                },
                sampling_config={
                    "sampling_plan_version": sampling_plan.version,
                    "sampling_plan_sha256": sampling_plan_digest(sampling_plan),
                },
                metadata={},
                attempt=attempt,
                retry_count=attempt - 1,
            )
            if terminal.status is InferenceStatus.SUCCEEDED:
                selection = self._ledger.get_selection(
                    terminal.logical_invocation_id,
                    self._inference_policy.selection_policy_version,
                )
                if selection is None:
                    raise CanonicalOfflineConfigurationError(
                        "successful invocation has no persisted selection"
                    )
                selected = self._ledger.get_terminal(selection.inference_id)
                if (
                    selected is None
                    or selected.status is not InferenceStatus.SUCCEEDED
                    or not selected.output_valid
                    or selected.input_plan_part_ordinal != part.ordinal
                ):
                    raise CanonicalOfflineConfigurationError(
                        "persisted selection does not reference the valid part success"
                    )
                return selected, selection, attempt

            failure = terminal.failure
            retryable = failure is not None and failure.retryability in {
                Retryability.RETRYABLE,
                Retryability.RATE_LIMITED,
            }
            if not retryable or attempt == self._execution_policy.max_attempts:
                return terminal, None, attempt
        assert terminal is not None
        return terminal, None, self._execution_policy.max_attempts

    def _build_selected_part_lineage(
        self,
        *,
        context: AdmittedRecordingContextV2,
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        part: InferenceCallPart,
        selected_terminal: ModelInference,
        selection: InferenceAttemptSelection,
    ) -> tuple[
        RawProviderResponseArtifact | None,
        ParsedProviderClaimArtifact | None,
        SelectedAttemptOutput | None,
        OrchestratorEnrichedOutput | None,
        CanonicalOfflineError | None,
    ]:
        try:
            if (
                selection.inference_id != selected_terminal.inference_id
                or selection.logical_invocation_id != selected_terminal.logical_invocation_id
            ):
                raise CanonicalOfflineConfigurationError(
                    "selection does not reference the selected terminal"
                )
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
            artifact_created_at = selected_terminal.completed_at
            expected_raw = RawProviderResponseArtifact.from_bytes(
                data=stored_raw.data,
                artifact_id=stored_raw.artifact_id,
                media_type=stored_raw.media_type,
                provider_request_id=stored_raw.provider_request_id,
                inference_id=selected_terminal.inference_id,
                provider=selected_terminal.provider,
                model_name=selected_terminal.model_name,
                model_version=selected_terminal.model_version,
                created_at=artifact_created_at,
            )
            parsed_artifact_id = _stable_uuid(
                "parsed-provider-claim",
                selected_terminal.inference_id,
                stored_raw.exact_bytes_sha256,
                self._inference_policy.output_schema.sha256,
                self._execution_policy.parser_version,
            )
            parsed_claims = self._evidence_store.get_parsed_claim(parsed_artifact_id)
            if parsed_claims is None:
                parsed_claims = self._adapter.parser.parse_artifact(
                    stored=stored_raw,
                    inference_id=selected_terminal.inference_id,
                    provider=selected_terminal.provider,
                    model_name=selected_terminal.model_name,
                    model_version=selected_terminal.model_version,
                    provider_claim_schema=self._inference_policy.output_schema,
                    task=VisionTask.FUSION_ADJUDICATION,
                    artifact_id=parsed_artifact_id,
                    created_at=artifact_created_at,
                )
                parsed_claims = self._evidence_store.append_parsed_claim(parsed_claims)
            if (
                parsed_claims.artifact_id != parsed_artifact_id
                or parsed_claims.raw_response != expected_raw
                or parsed_claims.provider_claim_schema != self._inference_policy.output_schema
                or parsed_claims.task is not VisionTask.FUSION_ADJUDICATION
                or parsed_claims.parser_version != self._execution_policy.parser_version
                or parsed_claims.created_at != artifact_created_at
                or parsed_claims.payload.model_dump(mode="json")
                != selected_terminal.normalized_output
            ):
                raise CanonicalOfflineConfigurationError(
                    "persisted parsed claims differ from the selected terminal"
                )
            expected_selected = SelectedAttemptOutput.create(parsed_claims, selection)
            selected_output = self._evidence_store.get_selected_output(selection.selection_id)
            if selected_output is None:
                selected_output = self._evidence_store.append_selected_output(expected_selected)
            elif selected_output != expected_selected:
                raise InferenceEvidenceStoreError(
                    "persisted selected output differs from its selection and parsed claim"
                )
        except (InferenceEvidenceStoreError, InferenceLedgerError) as exc:
            return (
                None,
                None,
                None,
                None,
                _canonical_error(
                    CanonicalOfflineStage.PARSING,
                    "INFERENCE_EVIDENCE_CONFLICT",
                    exc,
                ),
            )
        except (
            RawProviderBytesStoreError,
            StrictProviderClaimParseError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                None,
                None,
                None,
                None,
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
                    "input_plan_part_semantic_sha256": part.part_semantic_sha256,
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
            enriched_artifact_id = _stable_uuid("orchestrator-enrichment", logical_digest)
            enriched_output = self._evidence_store.get_enriched_output(enriched_artifact_id)
            if enriched_output is None:
                enriched_output = self._enricher.enrich(
                    input_plan=input_plan,
                    input_plan_part_ordinal=part.ordinal,
                    reference_catalog=reference_catalog,
                    parsed_claims=parsed_claims,
                    selected_attempt=selected_output,
                    authority=authority,
                    enriched_output_schema=enriched_schema,
                    enrichment_policy_version=self._execution_policy.enrichment_policy_version,
                    artifact_id=enriched_artifact_id,
                    created_at=selected_terminal.completed_at,
                )
                enriched_output = self._evidence_store.append_enriched_output(enriched_output)
            elif (
                enriched_output.artifact_id != enriched_artifact_id
                or enriched_output.enrichment_logical_key
                != f"orchestrator-enrichment:{logical_digest}"
                or enriched_output.task is not VisionTask.FUSION_ADJUDICATION
                or enriched_output.selected_attempt != selected_output
                or enriched_output.request_catalog_id
                != input_plan.request_catalog.request_catalog_id
                or enriched_output.request_catalog_sha256
                != input_plan.request_catalog.semantic_sha256
                or enriched_output.reference_catalog_id != reference_catalog.reference_catalog_id
                or enriched_output.reference_catalog_sha256 != reference_catalog.semantic_sha256
                or enriched_output.input_plan_id != input_plan.input_plan_id
                or enriched_output.input_plan_semantic_sha256 != input_plan.semantic_sha256
                or enriched_output.provider_claim_schema != self._inference_policy.output_schema
                or enriched_output.enriched_output_schema != enriched_schema
                or enriched_output.enrichment_policy_version
                != self._execution_policy.enrichment_policy_version
                or enriched_output.authority != authority
                or enriched_output.created_at != selected_terminal.completed_at
            ):
                raise InferenceEvidenceStoreError(
                    "persisted enriched output differs from the current semantic lineage"
                )
        except (InferenceEvidenceStoreError, InferenceLedgerError) as exc:
            return (
                parsed_claims.raw_response,
                parsed_claims,
                selected_output,
                None,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "INFERENCE_EVIDENCE_CONFLICT",
                    exc,
                ),
            )
        except (
            RawProviderBytesStoreError,
            ProviderClaimEnrichmentError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                parsed_claims.raw_response,
                parsed_claims,
                selected_output,
                None,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "ENRICHMENT_REJECTED",
                    exc,
                ),
            )
        return (
            parsed_claims.raw_response,
            parsed_claims,
            selected_output,
            enriched_output,
            None,
        )

    async def _execute_one_call_part(
        self,
        *,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        part: InferenceCallPart,
    ) -> CanonicalOfflinePartResult:
        terminal, selection, attempts_used = await self._orchestrate_call_part(
            context=context,
            window=window,
            sampling_plan=sampling_plan,
            package_set=package_set,
            package_inputs=package_inputs,
            input_plan=input_plan,
            part=part,
        )
        if terminal.status is not InferenceStatus.SUCCEEDED:
            failure = terminal.failure
            error = _canonical_error(
                CanonicalOfflineStage.INFERENCE,
                failure.code if failure is not None else terminal.status.value,
                failure.detail if failure is not None else terminal.status.value,
            )
            completion = self._call_barrier.submit_part_terminal(
                input_plan,
                terminal,
                failure_is_final=True,
            )
            return CanonicalOfflinePartResult(
                schema_version="1.0",
                part_ordinal=part.ordinal,
                part_count=part.part_count,
                part_semantic_sha256=part.part_semantic_sha256,
                status=CanonicalOfflinePartStatus.TERMINAL_FAILED,
                orchestration_attempt_count=attempts_used,
                terminal=terminal,
                selection=None,
                completion=completion,
                raw_response=None,
                parsed_claims=None,
                selected_output=None,
                enriched_output=None,
                error=error,
            )

        if selection is None:
            raise CanonicalOfflineConfigurationError(
                "selected success is missing its selection decision"
            )
        raw, parsed, selected_output, enriched, lineage_error = self._build_selected_part_lineage(
            context=context,
            input_plan=input_plan,
            reference_catalog=reference_catalog,
            part=part,
            selected_terminal=terminal,
            selection=selection,
        )
        completion = self._call_barrier.submit_part_terminal(
            input_plan,
            terminal,
            selection=selection,
        )
        return CanonicalOfflinePartResult(
            schema_version="1.0",
            part_ordinal=part.ordinal,
            part_count=part.part_count,
            part_semantic_sha256=part.part_semantic_sha256,
            status=(
                CanonicalOfflinePartStatus.ENRICHED
                if lineage_error is None
                else CanonicalOfflinePartStatus.POST_SELECTION_INVALID
            ),
            orchestration_attempt_count=attempts_used,
            terminal=terminal,
            selection=selection,
            completion=completion,
            raw_response=raw,
            parsed_claims=parsed,
            selected_output=selected_output,
            enriched_output=enriched,
            error=lineage_error,
        )

    async def _execute_declared_call_plan(
        self,
        *,
        context: AdmittedRecordingContextV2,
        window: CanonicalRootWindow,
        sampling_plan: SamplingPlan,
        package_set: TemporalPackageSet,
        package_inputs: tuple[PackageInput, ...],
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        created_at: str,
        state: dict[str, object],
        part_results: list[CanonicalOfflinePartResult],
        finish: Callable[
            [CanonicalOfflineRunStatus, CanonicalOfflineError | None],
            CanonicalOfflineRunResult,
        ],
        attach_nodes: Callable[
            [Sequence[tuple[LogicalNode, RunNodeRole]]],
            None,
        ],
        membership_failure: Callable[[object], CanonicalOfflineRunResult],
    ) -> CanonicalOfflineRunResult:
        for part in input_plan.call_plan.parts:
            try:
                result = await self._execute_one_call_part(
                    context=context,
                    window=window,
                    sampling_plan=sampling_plan,
                    package_set=package_set,
                    package_inputs=package_inputs,
                    input_plan=input_plan,
                    reference_catalog=reference_catalog,
                    part=part,
                )
            except asyncio.CancelledError:
                raise
            except (
                InferenceOrchestrationError,
                InferenceCallBarrierError,
                CanonicalOfflineConfigurationError,
                TypeError,
                ValueError,
            ) as exc:
                return finish(
                    CanonicalOfflineRunStatus.INFERENCE_FAILED,
                    _canonical_error(
                        CanonicalOfflineStage.INFERENCE,
                        "CALL_PART_EXECUTION_FAILED",
                        exc,
                    ),
                )
            part_results.append(result)
            try:
                nodes: list[tuple[LogicalNode, RunNodeRole]] = []
                if result.selection is not None:
                    nodes.append(
                        (canonical_selection_logical_node(result.selection), "ATTEMPT_SELECTION")
                    )
                if result.parsed_claims is not None:
                    nodes.append(
                        (canonical_parsed_claim_logical_node(result.parsed_claims), "PARSED_CLAIM")
                    )
                if result.selected_output is not None:
                    nodes.append(
                        (
                            canonical_selected_output_logical_node(result.selected_output),
                            "SELECTED_OUTPUT",
                        )
                    )
                if result.enriched_output is not None:
                    nodes.append(
                        (
                            canonical_enrichment_logical_node(result.enriched_output),
                            "ENRICHED_OUTPUT",
                        )
                    )
                attach_nodes(nodes)
            except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
                return membership_failure(exc)

        try:
            aggregate = self._call_barrier.get_aggregate_status(input_plan)
        except InferenceCallBarrierError as exc:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_STATUS_FAILED",
                    exc,
                ),
            )
        if not aggregate.is_complete:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_NOT_TERMINAL",
                    "declared call barrier remained open after every local part execution",
                ),
            )
        if aggregate.overall_status is StageStatus.INCOMPLETE:
            failed_ordinals = tuple(
                item.part_ordinal
                for item in part_results
                if item.status is CanonicalOfflinePartStatus.TERMINAL_FAILED
            )
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "REQUIRED_CALL_PARTS_INCOMPLETE",
                    f"required terminal failures at part ordinals {failed_ordinals}",
                ),
            )
        if aggregate.overall_status is not StageStatus.SUCCEEDED:
            return finish(
                CanonicalOfflineRunStatus.INFERENCE_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.INFERENCE,
                    "BARRIER_STATUS_INVALID",
                    aggregate.overall_status.value,
                ),
            )

        invalid = next(
            (
                item
                for item in part_results
                if item.status is CanonicalOfflinePartStatus.POST_SELECTION_INVALID
            ),
            None,
        )
        if invalid is not None:
            assert invalid.error is not None
            return finish(CanonicalOfflineRunStatus.INVALID_OUTPUT, invalid.error)

        enriched_outputs = tuple(
            item.enriched_output for item in part_results if item.enriched_output is not None
        )
        if len(enriched_outputs) != len(input_plan.call_plan.parts):
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.ENRICHMENT,
                    "PART_ENRICHMENT_SET_INCOMPLETE",
                    "successful barrier lacks exact enriched output coverage",
                ),
            )
        abstention_flags = tuple(item.abstained for item in enriched_outputs)
        if any(abstention_flags) and not all(abstention_flags):
            return finish(
                CanonicalOfflineRunStatus.INCOMPLETE,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "PARTIAL_REQUIRED_ABSTENTION",
                    "required call parts mixed provider claims and abstentions",
                ),
            )

        try:
            barrier_reduction = self._call_barrier.reduce(
                input_plan,
                reduced_at=created_at,
            )
            parsed_payloads = tuple(
                item.parsed_claims.payload
                for item in part_results
                if item.parsed_claims is not None
            )
            expected_payload = _reduce_provider_claim_payloads(parsed_payloads)
            reduced_payload = ProviderClaimPayload.model_validate(
                barrier_reduction.normalized_output,
                strict=False,
            )
            if reduced_payload != expected_payload:
                raise CanonicalOfflineConfigurationError(
                    "barrier reduction differs from exact parsed part claims"
                )
            fusion_reduction = _build_canonical_fusion_reduction(
                input_plan=input_plan,
                barrier_reduction=barrier_reduction,
                part_results=tuple(part_results),
                created_at=created_at,
            )
            state["barrier_reduction"] = barrier_reduction
            state["fusion_reduction"] = fusion_reduction
        except (
            InferenceCallBarrierError,
            CanonicalOfflineConfigurationError,
            TypeError,
            ValueError,
        ) as exc:
            return finish(
                CanonicalOfflineRunStatus.INVALID_OUTPUT,
                _canonical_error(
                    CanonicalOfflineStage.REDUCTION,
                    "FUSION_REDUCTION_REJECTED",
                    exc,
                ),
            )
        try:
            attach_nodes(
                (
                    (
                        canonical_call_reduction_logical_node(barrier_reduction),
                        "CALL_REDUCTION",
                    ),
                    (
                        canonical_fusion_reduction_logical_node(fusion_reduction),
                        "FUSION_REDUCTION",
                    ),
                )
            )
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        try:
            decision, hypotheses = self._projector.project(
                context=context,
                fusion_reduction=fusion_reduction,
                enriched_outputs=enriched_outputs,
                interval=window.interval,
            )
            state["output_decision"] = decision
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
        try:
            attach_nodes(
                (
                    (canonical_output_decision_logical_node(decision), "OUTPUT_DECISION"),
                    *(
                        (canonical_event_hypothesis_logical_node(item), "EVENT_HYPOTHESIS")
                        for item in hypotheses
                    ),
                )
            )
        except (TypeError, ValueError, _CanonicalRunMembershipPublicationError) as exc:
            return membership_failure(exc)

        if decision.decision == "ABSTAINED":
            return finish(CanonicalOfflineRunStatus.ABSTAINED, None)
        if decision.decision == "NO_EVENTS":
            return finish(CanonicalOfflineRunStatus.NO_EVENTS, None)

        try:
            state["identity_result"] = self._identity_registry.assign_batch(
                admitted_context=context,
                hypotheses=hypotheses,
                enriched_outputs=enriched_outputs,
                decided_at=created_at,
            )
        except (EventIdentityRegistryError, TypeError, ValueError) as exc:
            return finish(
                CanonicalOfflineRunStatus.IDENTITY_FAILED,
                _canonical_error(
                    CanonicalOfflineStage.IDENTITY,
                    "IDENTITY_ASSIGNMENT_FAILED",
                    exc,
                ),
            )
        return finish(CanonicalOfflineRunStatus.SUCCEEDED, None)

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


def _reduce_provider_claim_payloads(
    payloads: Sequence[ProviderClaimPayload],
) -> ProviderClaimPayload:
    """Reduce ordered part payloads using exact equality except local ordinals."""

    ordered = tuple(payloads)
    if not ordered:
        raise CanonicalOfflineConfigurationError(
            "provider claim reduction requires at least one part payload"
        )
    abstention_flags = tuple(item.abstained for item in ordered)
    if all(abstention_flags):
        return ProviderClaimPayload(claims=(), abstained=True)
    if any(abstention_flags):
        raise CanonicalOfflineConfigurationError(
            "required part payloads cannot mix claims and abstentions"
        )

    reduced: list[ProviderTaskClaim] = []
    seen_across_parts: set[str] = set()
    for payload in ordered:
        seen_in_part: set[str] = set()
        for claim in payload.claims:
            digest = semantic_sha256(claim.model_dump(mode="json", exclude={"claim_ordinal"}))
            if digest in seen_in_part:
                raise CanonicalOfflineConfigurationError(
                    "one call part contains duplicate provider claims"
                )
            seen_in_part.add(digest)
            if digest in seen_across_parts:
                continue
            seen_across_parts.add(digest)
            reduced.append(claim.model_copy(update={"claim_ordinal": len(reduced)}))
    return ProviderClaimPayload(claims=tuple(reduced), abstained=False)


def _fusion_claim_reduction_projection(
    claim: EnrichedProviderClaim,
) -> dict[str, object]:
    """Project enriched claim content without row IDs or storage locators."""

    confidence = claim.model_reported_confidence
    return {
        "kind": claim.kind.value,
        "package_ordinal": claim.package_ordinal,
        "camera_id": None if claim.camera_id is None else claim.camera_id.value,
        "interval": None if claim.interval is None else claim.interval.model_dump(mode="json"),
        "label": claim.label,
        "observation": claim.observation.value,
        "evidence": [
            {
                "package_ordinal": item.package_ordinal,
                "package_semantic_content_sha256": item.package_semantic_content_sha256,
                "camera_id": item.camera_id.value,
                "camera_ordinal": item.camera_ordinal,
                "frame_ordinal": item.frame_ordinal,
                "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
                "source_timestamp_ns": str(item.source_timestamp_ns),
                "source_artifact_sha256": item.source_artifact_sha256,
            }
            for item in claim.evidence
        ],
        "model_reported_confidence": (
            None
            if confidence is None
            else {
                "kind": confidence.kind,
                "semantics": confidence.semantics,
                "producer_type": confidence.producer_type,
                "producer_version": confidence.producer_version,
                "value": confidence.value,
            }
        ),
        "conflict_codes": list(claim.conflict_codes),
    }


def _fusion_claim_reduction_digest(claim: EnrichedProviderClaim) -> str:
    return semantic_sha256(_fusion_claim_reduction_projection(claim))


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


def _build_canonical_fusion_reduction(
    *,
    input_plan: InferenceInputPlan,
    barrier_reduction: InferenceCallReduction,
    part_results: Sequence[CanonicalOfflinePartResult],
    created_at: str,
) -> CanonicalFusionReduction:
    ordered_results = tuple(part_results)
    planned_parts = input_plan.call_plan.parts
    if (
        len(ordered_results) != len(planned_parts)
        or tuple(item.part_ordinal for item in ordered_results) != tuple(range(len(planned_parts)))
        or barrier_reduction.input_plan_semantic_sha256 != input_plan.semantic_sha256
        or barrier_reduction.ordered_completion_ids
        != tuple(item.completion.completion_id for item in ordered_results)
        or barrier_reduction.reduction_policy != input_plan.call_plan.reduction_policy
        or barrier_reduction.reduction_policy_version
        != input_plan.call_plan.reduction_policy_version
    ):
        raise CanonicalOfflineConfigurationError(
            "fusion reduction inputs do not match the complete ordered call plan"
        )

    part_sources: list[CanonicalFusionPartSource] = []
    grouped_claims: dict[
        str,
        list[tuple[CanonicalOfflinePartResult, EnrichedProviderClaim]],
    ] = {}
    for planned, result in zip(planned_parts, ordered_results, strict=True):
        enriched = result.enriched_output
        selected = result.selected_output
        if (
            result.status is not CanonicalOfflinePartStatus.ENRICHED
            or result.part_semantic_sha256 != planned.part_semantic_sha256
            or enriched is None
            or selected is None
        ):
            raise CanonicalOfflineConfigurationError(
                "fusion reduction requires exact enriched lineage for every part"
            )
        enrichment_ref = PlatformEnrichedOutputReference.from_output(enriched)
        part_sources.append(
            CanonicalFusionPartSource(
                part_ordinal=result.part_ordinal,
                part_semantic_sha256=result.part_semantic_sha256,
                completion_id=result.completion.completion_id,
                inference_id=result.terminal.inference_id,
                selected_attempt_output_sha256=selected.output_sha256,
                enrichment=enrichment_ref,
                abstained=enriched.abstained,
            )
        )
        seen_in_part: set[str] = set()
        for claim in enriched.claims:
            digest = _fusion_claim_reduction_digest(claim)
            if digest in seen_in_part:
                raise CanonicalOfflineConfigurationError(
                    "one enriched part contains duplicate fusion claim semantics"
                )
            seen_in_part.add(digest)
            grouped_claims.setdefault(digest, []).append((result, claim))

    abstention_flags = tuple(item.abstained for item in part_sources)
    if any(abstention_flags) and not all(abstention_flags):
        raise CanonicalOfflineConfigurationError(
            "fusion reduction cannot mix required abstentions and claims"
        )

    reduced_claims: list[CanonicalReducedFusionClaim] = []
    ordered_groups = sorted(
        grouped_claims.items(),
        key=lambda item: (
            -1 if item[1][0][1].interval is None else item[1][0][1].interval.start_ns,
            -1 if item[1][0][1].interval is None else item[1][0][1].interval.end_ns,
            item[0],
        ),
    )
    for digest, group in ordered_groups:
        ordered_group = tuple(
            sorted(
                group,
                key=lambda item: (
                    item[0].part_ordinal,
                    item[1].claim_ordinal,
                    item[1].claim_id,
                ),
            )
        )
        sources = tuple(
            CanonicalFusionClaimSource(
                part_ordinal=result.part_ordinal,
                source_claim_ordinal=claim.claim_ordinal,
                source_claim_id=claim.claim_id,
                enrichment_logical_key=result.enriched_output.enrichment_logical_key,
            )
            for result, claim in ordered_group
            if result.enriched_output is not None
        )
        reduced_claims.append(
            CanonicalReducedFusionClaim(
                fusion_output_ordinal=len(reduced_claims),
                claim_semantic_sha256=digest,
                representative=ordered_group[0][1],
                sources=sources,
            )
        )

    if all(abstention_flags):
        outcome: Literal["CLAIMS", "NO_SURVIVING_EVENTS", "ALL_PARTS_ABSTAINED"] = (
            "ALL_PARTS_ABSTAINED"
        )
    elif reduced_claims:
        outcome = "CLAIMS"
    else:
        outcome = "NO_SURVIVING_EVENTS"
    projection = _canonical_fusion_reduction_projection_values(
        schema_version="1.0",
        input_plan_semantic_sha256=input_plan.semantic_sha256,
        barrier_reduction_semantic_sha256=barrier_reduction.reduction_semantic_sha256,
        reduction_policy=input_plan.call_plan.reduction_policy,
        reduction_policy_version=input_plan.call_plan.reduction_policy_version,
        outcome=outcome,
        parts=part_sources,
        claims=reduced_claims,
    )
    digest = semantic_sha256(projection)
    return CanonicalFusionReduction(
        schema_version="1.0",
        reduction_id=_stable_uuid("canonical-fusion-reduction", digest),
        reduction_logical_key=f"fusion-reduction:{digest}",
        semantic_sha256=digest,
        input_plan_semantic_sha256=input_plan.semantic_sha256,
        barrier_reduction_id=barrier_reduction.reduction_id,
        barrier_reduction_semantic_sha256=(barrier_reduction.reduction_semantic_sha256),
        reduction_policy=input_plan.call_plan.reduction_policy,
        reduction_policy_version=input_plan.call_plan.reduction_policy_version,
        outcome=outcome,
        parts=tuple(part_sources),
        claims=tuple(reduced_claims),
        created_at=created_at,
    )


def _canonical_logical_node(
    *,
    node_type: str,
    key_namespace: str,
    semantic_digest: str,
    logical_key: str,
    identity_policy_version: str,
) -> LogicalNode:
    expected_key = f"{key_namespace}:{semantic_digest}"
    if logical_key != expected_key:
        raise CanonicalOfflineConfigurationError(
            f"{node_type} logical key does not match its typed semantic digest"
        )
    return logical_node_from_semantic_digest(
        node_type=node_type,
        key_namespace=key_namespace,
        semantic_sha256=semantic_digest,
        identity_policy_version=identity_policy_version,
    )


def canonical_root_window_logical_node(window: CanonicalRootWindow) -> LogicalNode:
    """Admit one validated root-window producer into the generic node registry."""

    checked = CanonicalRootWindow.model_validate(window.model_dump(mode="python"), strict=True)
    return _canonical_logical_node(
        node_type="TEMPORAL_WINDOW",
        key_namespace="temporal-window",
        semantic_digest=checked.semantic_sha256,
        logical_key=checked.window_logical_key,
        identity_policy_version="canonical-root-window-node-v1",
    )


def canonical_package_set_semantic_projection(
    package_set: TemporalPackageSet,
) -> dict[str, object]:
    """Return package-set identity without execution or storage locators."""

    checked = TemporalPackageSet.model_validate(package_set.model_dump(mode="python"), strict=True)
    return {
        "schema_version": checked.schema_version,
        "lineage": checked.lineage.model_dump(mode="json"),
        "requested_interval": {
            "start_ns": str(checked.requested_start_ns),
            "end_ns": str(checked.requested_end_ns),
        },
        "effective_interval": {
            "start_ns": str(checked.start_ns),
            "end_ns": str(checked.end_ns),
        },
        "split_reason": checked.split_reason.value,
        "split_policy_version": checked.split_policy_version,
        "split_plan_digest": checked.split_plan_digest,
        "members": [
            {
                "ordinal": member.ordinal,
                "part_count": member.part_count,
                "requested_start_ns": str(member.requested_start_ns),
                "requested_end_ns": str(member.requested_end_ns),
                "start_ns": str(member.start_ns),
                "end_ns": str(member.end_ns),
                "overlap_before_ns": str(member.overlap_before_ns),
                "overlap_after_ns": str(member.overlap_after_ns),
                "package_semantic_content_sha256": member.package_semantic_content_sha256,
            }
            for member in checked.members
        ],
        "member_manifest_sha256": checked.member_manifest_sha256,
        "reduction_policy_version": checked.reduction_policy_version,
    }


def canonical_package_set_logical_node(package_set: TemporalPackageSet) -> LogicalNode:
    checked = TemporalPackageSet.model_validate(package_set.model_dump(mode="python"), strict=True)
    digest = semantic_sha256(canonical_package_set_semantic_projection(checked))
    return _canonical_logical_node(
        node_type="TEMPORAL_PACKAGE_SET",
        key_namespace="temporal-package-set",
        semantic_digest=digest,
        logical_key=f"temporal-package-set:{digest}",
        identity_policy_version="canonical-package-set-node-v1",
    )


def canonical_input_plan_logical_node(input_plan: InferenceInputPlan) -> LogicalNode:
    checked = InferenceInputPlan.model_validate(input_plan.model_dump(mode="python"), strict=True)
    return _canonical_logical_node(
        node_type="INFERENCE_INPUT_PLAN",
        key_namespace="inference-input-plan",
        semantic_digest=checked.semantic_sha256,
        logical_key=f"inference-input-plan:{checked.semantic_sha256}",
        identity_policy_version="canonical-input-plan-node-v1",
    )


def canonical_call_part_logical_node(
    input_plan: InferenceInputPlan,
    part: InferenceCallPart,
) -> LogicalNode:
    checked_plan = InferenceInputPlan.model_validate(
        input_plan.model_dump(mode="python"), strict=True
    )
    checked = InferenceCallPart.model_validate(part.model_dump(mode="python"), strict=True)
    if (
        checked.ordinal >= len(checked_plan.call_plan.parts)
        or checked_plan.call_plan.parts[checked.ordinal] != checked
    ):
        raise CanonicalOfflineConfigurationError(
            "call part is not an exact member of the validated input plan"
        )
    return _canonical_logical_node(
        node_type="INFERENCE_CALL_PART",
        key_namespace="inference-input-call-part",
        semantic_digest=checked.part_semantic_sha256,
        logical_key=checked.part_logical_key,
        identity_policy_version="canonical-input-call-part-node-v1",
    )


def canonical_call_barrier_logical_node(input_plan: InferenceInputPlan) -> LogicalNode:
    checked = InferenceInputPlan.model_validate(input_plan.model_dump(mode="python"), strict=True)
    barrier = checked.call_plan
    return _canonical_logical_node(
        node_type="INFERENCE_CALL_BARRIER",
        key_namespace="inference-input-barrier",
        semantic_digest=barrier.barrier_semantic_sha256,
        logical_key=barrier.barrier_logical_key,
        identity_policy_version="canonical-input-barrier-node-v1",
    )


def canonical_selection_logical_node(
    selection: InferenceAttemptSelection,
) -> LogicalNode:
    checked = InferenceAttemptSelection.model_validate(
        selection.model_dump(mode="python"), strict=True
    )
    digest = inference_attempt_selection_digest(
        logical_invocation_id=checked.logical_invocation_id,
        policy_version=checked.policy_version,
    )
    return _canonical_logical_node(
        node_type="INFERENCE_ATTEMPT_SELECTION",
        key_namespace="inference-attempt-selection",
        semantic_digest=digest,
        logical_key=checked.selection_decision_logical_key,
        identity_policy_version="canonical-attempt-selection-node-v1",
    )


def canonical_parsed_claim_logical_node(
    parsed: ParsedProviderClaimArtifact,
) -> LogicalNode:
    checked = ParsedProviderClaimArtifact.model_validate(
        parsed.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="PARSED_PROVIDER_CLAIM",
        key_namespace="parsed-provider-claim",
        semantic_digest=checked.semantic_sha256,
        logical_key=f"parsed-provider-claim:{checked.semantic_sha256}",
        identity_policy_version="canonical-parsed-claim-node-v1",
    )


def canonical_selected_output_logical_node(
    selected: SelectedAttemptOutput,
) -> LogicalNode:
    checked = SelectedAttemptOutput.model_validate(selected.model_dump(mode="python"), strict=True)
    return _canonical_logical_node(
        node_type="SELECTED_ATTEMPT_OUTPUT",
        key_namespace="selected-attempt-output",
        semantic_digest=checked.output_sha256,
        logical_key=f"selected-attempt-output:{checked.output_sha256}",
        identity_policy_version="canonical-selected-output-node-v1",
    )


def canonical_enrichment_logical_node(
    output: OrchestratorEnrichedOutput,
) -> LogicalNode:
    checked = OrchestratorEnrichedOutput.model_validate(
        output.model_dump(mode="python"), strict=True
    )
    digest = checked.enrichment_logical_key.rsplit(":", maxsplit=1)[-1]
    return _canonical_logical_node(
        node_type="ORCHESTRATOR_ENRICHMENT",
        key_namespace="orchestrator-enrichment",
        semantic_digest=digest,
        logical_key=checked.enrichment_logical_key,
        identity_policy_version="canonical-enrichment-node-v1",
    )


def canonical_call_reduction_logical_node(
    reduction: InferenceCallReduction,
) -> LogicalNode:
    checked = InferenceCallReduction.model_validate(
        reduction.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="INFERENCE_CALL_REDUCTION",
        key_namespace="inference-call-reduction",
        semantic_digest=checked.reduction_semantic_sha256,
        logical_key=f"inference-call-reduction:{checked.reduction_semantic_sha256}",
        identity_policy_version="canonical-call-reduction-node-v1",
    )


def canonical_fusion_reduction_logical_node(
    reduction: CanonicalFusionReduction,
) -> LogicalNode:
    checked = CanonicalFusionReduction.model_validate(
        reduction.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="FUSION_REDUCTION",
        key_namespace="fusion-reduction",
        semantic_digest=checked.semantic_sha256,
        logical_key=checked.reduction_logical_key,
        identity_policy_version="canonical-fusion-reduction-node-v1",
    )


def canonical_output_decision_logical_node(
    decision: CanonicalOutputAdmissionDecision,
) -> LogicalNode:
    checked = CanonicalOutputAdmissionDecision.model_validate(
        decision.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="OUTPUT_ADMISSION_DECISION",
        key_namespace="output-admission-decision",
        semantic_digest=checked.semantic_sha256,
        logical_key=f"output-admission-decision:{checked.semantic_sha256}",
        identity_policy_version="canonical-output-decision-node-v1",
    )


def canonical_event_hypothesis_logical_node(
    hypothesis: PlatformEnrichedEventHypothesis,
) -> LogicalNode:
    checked = PlatformEnrichedEventHypothesis.model_validate(
        hypothesis.model_dump(mode="python"), strict=True
    )
    return _canonical_logical_node(
        node_type="EVENT_HYPOTHESIS",
        key_namespace="event-hypothesis",
        semantic_digest=checked.semantic_sha256,
        logical_key=checked.event_hypothesis_logical_key,
        identity_policy_version="canonical-event-hypothesis-node-v1",
    )


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
    source_refs: Sequence[PlatformEnrichedOutputReference],
    fusion_reduction: CanonicalFusionReduction,
    policy: ProductionOutputAdmissionPolicyRef,
    decision: Literal["PRODUCTION_ADMITTED", "NO_EVENTS", "ABSTAINED"],
    admitted_claim_ordinals: tuple[int, ...],
    reason_code: str,
    production_output_admission: ProductionOutputAdmissionProof | None,
) -> CanonicalOutputAdmissionDecision:
    refs = tuple(sorted(source_refs, key=lambda item: item.enrichment_logical_key))
    projection = _canonical_output_decision_projection_values(
        decision=decision,
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
        schema_version="1.0",
        decision_id=_stable_uuid("canonical-output-admission", digest),
        decision=decision,
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


def _strict_context(context: AdmittedRecordingContextV2) -> AdmittedRecordingContextV2:
    if not isinstance(context, AdmittedRecordingContextV2):
        raise CanonicalOfflineConfigurationError("expected AdmittedRecordingContextV2")
    try:
        return AdmittedRecordingContextV2.model_validate(
            context.model_dump(mode="python"), strict=True
        )
    except ValueError as exc:
        raise CanonicalOfflineConfigurationError("V2 admission context failed validation") from exc


def _validate_processing_run_binding(
    *,
    processing_run: CanonicalProcessingRunContext,
    admitted_context: AdmittedRecordingContextV2,
    execution_policy: CanonicalOfflineExecutionPolicy,
) -> None:
    if (
        processing_run.recording_identity != admitted_context.recording_identity
        or processing_run.mcap_id != admitted_context.ready_manifest.mcap_id
        or processing_run.pipeline_version != CANONICAL_OFFLINE_PIPELINE_VERSION
        or processing_run.config_sha256 != execution_policy.semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "processing run does not bind the admitted recording and execution policy"
        )


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


def _rfc3339_datetime(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an RFC3339 timezone")
    return parsed


__all__ = [
    "CANONICAL_OFFLINE_PIPELINE_VERSION",
    "CanonicalFusionClaimSource",
    "CanonicalFusionPartSource",
    "CanonicalFusionReduction",
    "CanonicalOfflineConfigurationError",
    "CanonicalOfflineError",
    "CanonicalOfflineExecutionPolicy",
    "CanonicalOfflinePartResult",
    "CanonicalOfflinePartStatus",
    "CanonicalOfflinePipeline",
    "CanonicalOfflineRunResult",
    "CanonicalOfflineRunStatus",
    "CanonicalOfflineStage",
    "CanonicalOutputAdmissionDecision",
    "CanonicalReducedFusionClaim",
    "CanonicalRootWindow",
    "FusionEventHypothesisProjector",
    "canonical_call_barrier_logical_node",
    "canonical_call_part_logical_node",
    "canonical_call_reduction_logical_node",
    "canonical_enrichment_logical_node",
    "canonical_event_hypothesis_logical_node",
    "canonical_execution_policy_projection",
    "canonical_fusion_reduction_logical_node",
    "canonical_fusion_reduction_projection",
    "canonical_input_plan_logical_node",
    "canonical_lineage",
    "canonical_output_decision_logical_node",
    "canonical_output_decision_projection",
    "canonical_package_set_logical_node",
    "canonical_package_set_semantic_projection",
    "canonical_parsed_claim_logical_node",
    "canonical_root_window_logical_node",
    "canonical_root_window_projection_values",
    "canonical_selected_output_logical_node",
    "canonical_selection_logical_node",
]
