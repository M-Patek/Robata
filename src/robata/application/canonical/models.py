"""Canonical offline status, lineage, part, window, and policy models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.projections import (
    _stable_uuid,
    canonical_execution_policy_projection,
    canonical_execution_policy_projection_values,
    canonical_root_window_projection_values,
)
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp
from robata.contracts.pipeline import SamplingPurpose
from robata.contracts.sampling_plan import SamplingPlan
from robata.contracts.temporal import PackageLineage
from robata.event_pipeline.candidate import CanonicalCandidateEvent
from robata.event_pipeline.identity_registry import ProductionOutputAdmissionPolicyRef
from robata.inference.call_barrier import InferenceCallPartCompletion
from robata.inference.enrichment import (
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
)
from robata.inference.models import InferenceAttemptSelection, InferenceStatus, ModelInference
from robata.sampling.package_set import sampling_plan_digest

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

CANONICAL_OFFLINE_PIPELINE_VERSION = "canonical-offline-v5"


class CanonicalOfflineConfigurationError(ValueError):
    """The configured local vertical slice cannot satisfy its contracts."""


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
        if purpose not in {
            SamplingPurpose.QA_COARSE,
            SamplingPurpose.QA_DENSE,
            SamplingPurpose.ACTION_DENSE,
        }:
            raise CanonicalOfflineConfigurationError(
                "canonical offline window supports only QA_COARSE, QA_DENSE, and "
                "ACTION_DENSE purposes"
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


def _canonical_candidate_dense_window_projection_values(
    values: dict[str, object],
) -> dict[str, object]:
    projection = canonical_root_window_projection_values(values)
    candidate_interval = values["candidate_effective_interval"]
    if not isinstance(candidate_interval, NanosecondInterval):
        raise TypeError("candidate effective interval must be a NanosecondInterval")
    return {
        **projection,
        "candidate_effective_interval": candidate_interval.model_dump(mode="json"),
        "context_truncated": values["context_truncated"],
    }


class CanonicalCandidateDenseWindow(StrictModel):
    """Candidate-scoped ACTION_DENSE window with explicit clipping evidence."""

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
    candidate_event_id: OpaqueUuid
    candidate_logical_key: NodeLogicalKey
    candidate_effective_interval: NanosecondInterval
    requested_interval: NanosecondInterval
    interval: NanosecondInterval
    context_truncated: bool
    purpose: Literal[SamplingPurpose.ACTION_DENSE]
    window_policy_version: SchemaVersion
    source_subject_type: Literal["CANDIDATE_EVENT"]
    source_subject_logical_key: NodeLogicalKey
    parent_window_logical_key: NodeLogicalKey
    source_lineage_sha256: Sha256Digest
    refinement_role: Literal["CANDIDATE_DENSE"]
    generation: PositiveInt
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if (
            self.source_subject_logical_key != self.candidate_logical_key
            or self.interval.start_ns < 0
            or self.interval.end_ns > self.recording_duration_ns
            or self.interval.start_ns < self.requested_interval.start_ns
            or self.interval.end_ns > self.requested_interval.end_ns
            or self.candidate_effective_interval.start_ns < self.interval.start_ns
            or self.candidate_effective_interval.end_ns > self.interval.end_ns
            or self.context_truncated != (self.interval != self.requested_interval)
        ):
            raise ValueError("candidate dense window has inconsistent subject or bounds")
        projection = _canonical_candidate_dense_window_projection_values(
            {
                "source_content_sha256": self.source_content_sha256,
                "camera_mapping_semantic_sha256": (self.camera_mapping_semantic_sha256),
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
                "candidate_effective_interval": self.candidate_effective_interval,
                "context_truncated": self.context_truncated,
            }
        )
        digest = semantic_sha256(projection)
        if (
            self.semantic_sha256 != digest
            or self.window_logical_key != f"temporal-window:{digest}"
            or self.window_id != _stable_uuid("canonical-candidate-dense-window", digest)
        ):
            raise ValueError("candidate dense window identity is inconsistent")
        return self

    @classmethod
    def from_context(
        cls,
        *,
        context: AdmittedRecordingContextV2,
        candidate: CanonicalCandidateEvent,
        parent_window: CanonicalRootWindow,
        window_policy_version: str,
        created_at: str,
    ) -> Self:
        context = _strict_context(context)
        candidate = CanonicalCandidateEvent.model_validate(
            candidate.model_dump(mode="python"), strict=True
        )
        parent_window = CanonicalRootWindow.model_validate(
            parent_window.model_dump(mode="python"), strict=True
        )
        duration = context.ready_manifest.recording.duration_ns
        if (
            candidate.mcap_id != context.ready_manifest.mcap_id
            or candidate.source_content_sha256 != context.source_content_sha256
            or candidate.camera_mapping_semantic_sha256 != context.camera_mapping_semantic_sha256
            or candidate.alignment_semantic_sha256 != context.alignment_semantic_sha256
            or parent_window.recording_identity != context.recording_identity
            or parent_window.mcap_id != candidate.mcap_id
        ):
            raise CanonicalOfflineConfigurationError(
                "candidate dense window lineage does not match admission context"
            )
        requested = candidate.requested_dense_interval
        effective_start = max(0, requested.start_ns)
        effective_end = min(duration, requested.end_ns)
        if effective_start >= effective_end:
            raise CanonicalOfflineConfigurationError(
                "candidate dense request does not overlap the admitted recording"
            )
        effective = NanosecondInterval(start_ns=effective_start, end_ns=effective_end)
        if (
            candidate.effective_interval.start_ns < effective.start_ns
            or candidate.effective_interval.end_ns > effective.end_ns
        ):
            raise CanonicalOfflineConfigurationError(
                "candidate dense request does not contain its candidate interval"
            )
        source_lineage = semantic_sha256(
            {
                "semantic_projection_version": "candidate-dense-source-lineage-v1",
                "admission_context_semantic_sha256": context.semantic_sha256,
                "candidate_logical_key": candidate.candidate_logical_key,
                "parent_window_semantic_sha256": parent_window.semantic_sha256,
            }
        )
        values: dict[str, object] = {
            "source_content_sha256": context.source_content_sha256,
            "camera_mapping_semantic_sha256": context.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": context.alignment_semantic_sha256,
            "requested_interval": requested,
            "interval": effective,
            "purpose": SamplingPurpose.ACTION_DENSE,
            "window_policy_version": window_policy_version,
            "source_subject_type": "CANDIDATE_EVENT",
            "source_subject_logical_key": candidate.candidate_logical_key,
            "parent_window_logical_key": parent_window.window_logical_key,
            "source_lineage_sha256": source_lineage,
            "refinement_role": "CANDIDATE_DENSE",
            "generation": candidate.generation + 1,
            "candidate_effective_interval": candidate.effective_interval,
            "context_truncated": effective != requested,
        }
        digest = semantic_sha256(_canonical_candidate_dense_window_projection_values(values))
        return cls(
            schema_version="1.0",
            window_id=_stable_uuid("canonical-candidate-dense-window", digest),
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
            candidate_event_id=candidate.candidate_event_id,
            candidate_logical_key=candidate.candidate_logical_key,
            candidate_effective_interval=candidate.effective_interval,
            requested_interval=requested,
            interval=effective,
            context_truncated=effective != requested,
            purpose=SamplingPurpose.ACTION_DENSE,
            window_policy_version=window_policy_version,
            source_subject_type="CANDIDATE_EVENT",
            source_subject_logical_key=candidate.candidate_logical_key,
            parent_window_logical_key=parent_window.window_logical_key,
            source_lineage_sha256=source_lineage,
            refinement_role="CANDIDATE_DENSE",
            generation=candidate.generation + 1,
            created_at=created_at,
        )


def canonical_candidate_dense_lineage(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalCandidateDenseWindow,
    sampling_plan: SamplingPlan,
) -> PackageLineage:
    """Build provider-neutral package lineage for one candidate-dense window."""

    context = _strict_context(context)
    window = CanonicalCandidateDenseWindow.model_validate(
        window.model_dump(mode="python"), strict=True
    )
    sampling_plan = SamplingPlan.model_validate(
        sampling_plan.model_dump(mode="python"), strict=True
    )
    if (
        window.recording_identity != context.recording_identity
        or window.source_content_sha256 != context.source_content_sha256
        or window.mcap_id != context.ready_manifest.mcap_id
        or window.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
        or window.alignment_id != context.alignment_manifest.alignment_id
        or window.camera_mapping_semantic_sha256 != context.camera_mapping_semantic_sha256
        or window.alignment_semantic_sha256 != context.alignment_semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "candidate dense window does not match its admitted recording context"
        )
    return PackageLineage(
        source_content_sha256=context.source_content_sha256,
        window_semantic_sha256=window.semantic_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        sampling_plan_sha256=sampling_plan_digest(
            sampling_plan, purpose=SamplingPurpose.ACTION_DENSE
        ),
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
        sampling_plan_sha256=sampling_plan_digest(sampling_plan, purpose=window.purpose),
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
    provisional_fusion_policy_version: SchemaVersion
    boundary_refinement_policy_version: SchemaVersion
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
        provisional_fusion_policy_version: str,
        boundary_refinement_policy_version: str,
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
            "provisional_fusion_policy_version": provisional_fusion_policy_version,
            "boundary_refinement_policy_version": boundary_refinement_policy_version,
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
            provisional_fusion_policy_version=provisional_fusion_policy_version,
            boundary_refinement_policy_version=boundary_refinement_policy_version,
            max_attempts=max_attempts,
            output_admission_policy=output_admission_policy,
            semantic_sha256=digest,
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


def _strict_context(context: AdmittedRecordingContextV2) -> AdmittedRecordingContextV2:
    if not isinstance(context, AdmittedRecordingContextV2):
        raise CanonicalOfflineConfigurationError("expected AdmittedRecordingContextV2")
    try:
        return AdmittedRecordingContextV2.model_validate(
            context.model_dump(mode="python"), strict=True
        )
    except ValueError as exc:
        raise CanonicalOfflineConfigurationError("V2 admission context failed validation") from exc


__all__ = [
    "CANONICAL_OFFLINE_PIPELINE_VERSION",
    "CanonicalCandidateDenseWindow",
    "CanonicalOfflineConfigurationError",
    "CanonicalOfflineError",
    "CanonicalOfflineExecutionPolicy",
    "CanonicalOfflinePartResult",
    "CanonicalOfflinePartStatus",
    "CanonicalOfflineRunStatus",
    "CanonicalOfflineStage",
    "CanonicalRootWindow",
    "canonical_candidate_dense_lineage",
    "canonical_lineage",
]
