"""Inference models for the vision model inference layer.

Architecture V1.1 — Sections 9 (VisionModelAdapter), 10 (Qwen primary path),
and 11 (GPT shadow path).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VisionTask(StrEnum):
    """VLM task types."""

    QA_COARSE = "QA_COARSE"
    QA_DENSE = "QA_DENSE"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_EVIDENCE = "ACTION_EVIDENCE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"
    FUSION_ADJUDICATION = "FUSION_ADJUDICATION"


class InferenceStatus(StrEnum):
    """Terminal inference attempt status."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    INVALID_OUTPUT = "INVALID_OUTPUT"


class Retryability(StrEnum):
    """Retry classification for failed inferences."""

    RETRYABLE = "RETRYABLE"
    RATE_LIMITED = "RATE_LIMITED"
    PERMANENT = "PERMANENT"


class InputMode(StrEnum):
    """Supported input modes for a vision model."""

    IMAGE = "IMAGE"
    MULTI_IMAGE = "MULTI_IMAGE"
    VIDEO = "VIDEO"


class ConcurrencyClass(StrEnum):
    """Provider concurrency classification."""

    SERIAL = "SERIAL"
    LIMITED = "LIMITED"
    ELASTIC = "ELASTIC"


class ShadowSelectionReason(StrEnum):
    """Reason for shadow route selection."""

    RANDOM = "RANDOM"
    HARD_CASE = "HARD_CASE"


class ShadowRouteStatus(StrEnum):
    """Shadow route lifecycle status."""

    SELECTED = "SELECTED"
    DEFERRED = "DEFERRED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    SKIPPED_BUDGET = "SKIPPED_BUDGET"


# ---------------------------------------------------------------------------
# Capability models (Section 9.1)
# ---------------------------------------------------------------------------


class ModelCapabilities(StrictModel):
    """Snapshot of a model's supported tasks, limits, and policies.

    Per Architecture V1.1 Section 9.1, this is the normalized capability
    discovery record returned by an adapter and consumed by the orchestrator
    when selecting and validating an inference request.
    """

    schema_version: Literal["1.0"]
    snapshot_id: OpaqueUuid
    snapshot_digest: Sha256Digest
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    supported_tasks: tuple[VisionTask, ...]
    input_modes: tuple[InputMode, ...]
    accepted_media_types: tuple[NonEmptyString, ...]
    max_images_per_request: PositiveInt | None
    max_pixels_per_image: PositiveInt | None
    max_payload_bytes: PositiveInt | None
    max_input_tokens: PositiveInt | None
    supports_json_schema: bool
    supports_provider_idempotency: bool
    concurrency_class: ConcurrencyClass
    data_handling_policy_version: SchemaVersion
    observed_at: Rfc3339Timestamp


class CapabilitySnapshot(StrictModel):
    """Immutable reference to a discovered capability snapshot.

    Links a capability record to its digest for integrity verification
    during inference orchestration.
    """

    schema_version: Literal["1.0"]
    snapshot_id: OpaqueUuid
    snapshot_digest: Sha256Digest
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    observed_at: Rfc3339Timestamp


# ---------------------------------------------------------------------------
# Inference usage and failure
# ---------------------------------------------------------------------------


class ModelInferenceUsage(StrictModel):
    """Resource usage for one inference attempt."""

    input_frames: NonNegativeInt
    input_images: NonNegativeInt
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: NonNegativeFiniteFloat | None = None
    currency: NonEmptyString | None = None


class InferenceFailure(StrictModel):
    """Failure details for a non-success inference."""

    code: NonEmptyString
    detail: NonEmptyString
    retryability: Retryability


# ---------------------------------------------------------------------------
# ModelInference (Section 10.2)
# ---------------------------------------------------------------------------


class ModelInference(StrictModel):
    """One inference attempt, including failures.

    Comprehensive per-attempt record as specified in Architecture V1.1
    Section 10.2. Every field is denormalized for audit and replay.
    """

    schema_version: Literal["1.0"]
    inference_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    request_id: OpaqueUuid
    idempotency_key: NonEmptyString
    mcap_id: OpaqueUuid
    package_set_id: OpaqueUuid | None = None
    package_id: OpaqueUuid | None = None
    package_ids: tuple[NonEmptyString, ...]
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    stage: VisionTask
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    prompt_version: SchemaVersion
    prompt_artifact_id: NonEmptyString
    prompt_sha256: Sha256Digest
    rendered_input_digest: Sha256Digest
    output_schema_id: NonEmptyString
    output_schema_version: SchemaVersion
    output_schema_artifact_id: NonEmptyString
    output_schema_sha256: Sha256Digest
    capability_snapshot_id: OpaqueUuid
    capability_snapshot_digest: Sha256Digest
    input_manifest_set_sha256: Sha256Digest
    input_config: dict[str, object]
    sampling_config: dict[str, object]
    generation_config: dict[str, object]
    provider_request_id: NonEmptyString | None = None
    experiment_id: NonEmptyString | None = None
    shadow_route_id: NonEmptyString | None = None
    primary_inference_id: NonEmptyString | None = None
    shadow: bool = False
    attempt: Annotated[int, Field(strict=True, ge=1)]
    retry_count: NonNegativeInt
    status: InferenceStatus
    queued_at: Rfc3339Timestamp
    started_at: Rfc3339Timestamp
    completed_at: Rfc3339Timestamp
    latency_ms: NonNegativeInt
    raw_output: dict[str, object] | None = None
    normalized_output: dict[str, object] | None = None
    output_valid: bool = False
    reported_confidence: dict[str, object] | None = None
    calibrated_confidence: dict[str, object] | None = None
    usage: ModelInferenceUsage
    failure: InferenceFailure | None = None
    created_at: Rfc3339Timestamp


# ---------------------------------------------------------------------------
# Selection and decision models
# ---------------------------------------------------------------------------


class InferenceAttemptSelection(StrictModel):
    """Orchestrator selection of one successful inference attempt.

    Prevents duplicate downstream records when the same logical invocation
    is retried or redelivered. The unique constraint is on the logical
    invocation and the selection policy version.
    """

    schema_version: Literal["1.0"]
    selection_id: OpaqueUuid
    inference_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    policy_version: SchemaVersion
    selected_at: Rfc3339Timestamp


class ProductionDecision(StrictModel):
    """Typed production decision referencing a selected inference result.

    Records which inference-backed normalized intermediate drives a later
    semantic stage. Per Architecture V1.1, provider name alone never
    implies selection.
    """

    schema_version: Literal["1.0"]
    decision_id: OpaqueUuid
    selection_id: OpaqueUuid
    inference_id: OpaqueUuid
    stage: VisionTask
    mcap_id: OpaqueUuid
    package_set_id: OpaqueUuid | None = None
    package_id: OpaqueUuid | None = None
    decision_reason: NonEmptyString | None = None
    policy_version: SchemaVersion
    decided_at: Rfc3339Timestamp


# ---------------------------------------------------------------------------
# Shadow models (Section 11)
# ---------------------------------------------------------------------------


class ShadowRoute(StrictModel):
    """One GPT shadow route selection.

    Reproducibly selects a subset of packages for GPT shadow inference
    without affecting the primary Qwen path.
    """

    schema_version: Literal["1.0"]
    shadow_route_id: OpaqueUuid
    primary_inference_id: OpaqueUuid | None = None
    package_set_id: OpaqueUuid
    package_set_member_manifest_digest: Sha256Digest
    task: VisionTask
    reasons: tuple[ShadowSelectionReason, ...]
    sample_ratio: UnitInterval
    policy_version: SchemaVersion
    status: ShadowRouteStatus
    created_at: Rfc3339Timestamp


class ModelDisagreementSample(StrictModel):
    """One paired Qwen vs GPT evaluation record.

    Append-only disagreement sample as specified in Architecture V1.1
    Section 11.3. References both inference IDs for replay and audit.
    """

    schema_version: Literal["1.0"]
    disagreement_id: OpaqueUuid
    mcap_id: OpaqueUuid
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    package_set_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    qwen_inference_id: OpaqueUuid
    gpt_inference_id: OpaqueUuid
    comparison_contract_version: SchemaVersion
    comparison_config_digest: Sha256Digest
    shadow_route_id: OpaqueUuid
    shadow_reason: ShadowSelectionReason
    field_deltas: tuple[dict[str, object], ...]
    status: NonEmptyString
    adjudication: dict[str, object] | None = None
    created_at: Rfc3339Timestamp


__all__ = [
    "CapabilitySnapshot",
    "ConcurrencyClass",
    "InferenceAttemptSelection",
    "InferenceFailure",
    "InferenceStatus",
    "InputMode",
    "ModelCapabilities",
    "ModelDisagreementSample",
    "ModelInference",
    "ModelInferenceUsage",
    "ProductionDecision",
    "Retryability",
    "ShadowRoute",
    "ShadowRouteStatus",
    "ShadowSelectionReason",
    "VisionTask",
]
