"""Model inference and shadow route contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$",
    ),
]


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


class VisionTask(StrEnum):
    """VLM task types."""

    QA_COARSE = "QA_COARSE"
    QA_DENSE = "QA_DENSE"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_EVIDENCE = "ACTION_EVIDENCE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"
    FUSION_ADJUDICATION = "FUSION_ADJUDICATION"


class ModelInferenceUsage(StrictModel):
    """Resource usage for one inference attempt."""

    input_frames: NonNegativeInt
    input_images: NonNegativeInt
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None = None
    currency: NonEmptyString | None = None


class InferenceFailure(StrictModel):
    """Failure details for a non-success inference."""

    code: NonEmptyString
    detail: NonEmptyString
    retryability: Retryability


class ModelInference(StrictModel):
    """One inference attempt, including failures."""

    schema_version: Literal["1.0"]
    inference_id: NonEmptyString
    logical_invocation_id: NonEmptyString
    request_id: NonEmptyString
    idempotency_key: NonEmptyString
    mcap_id: NonEmptyString
    package_set_id: NonEmptyString | None = None
    package_id: NonEmptyString | None = None
    package_ids: tuple[NonEmptyString, ...]
    camera_mapping_run_id: NonEmptyString
    alignment_id: NonEmptyString
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    stage: VisionTask
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    prompt_version: SchemaVersion
    prompt_artifact_id: NonEmptyString
    prompt_sha256: NonEmptyString
    rendered_input_digest: NonEmptyString
    output_schema_id: NonEmptyString
    output_schema_version: SchemaVersion
    output_schema_artifact_id: NonEmptyString
    output_schema_sha256: NonEmptyString
    capability_snapshot_id: NonEmptyString
    capability_snapshot_digest: NonEmptyString
    input_manifest_set_sha256: NonEmptyString
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


class ShadowRouteReason(StrEnum):
    """Reason for shadow route selection."""

    RANDOM = "RANDOM"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_DISAGREEMENT = "HIGH_DISAGREEMENT"
    AMBIGUOUS_QA = "AMBIGUOUS_QA"
    UNCERTAIN_BOUNDARY = "UNCERTAIN_BOUNDARY"
    INVALID_OUTPUT_REPAIR = "INVALID_OUTPUT_REPAIR"


class ShadowRoute(StrictModel):
    """One GPT shadow route selection."""

    schema_version: Literal["1.0"]
    shadow_route_id: NonEmptyString
    primary_inference_id: NonEmptyString | None = None
    package_set_id: NonEmptyString
    package_set_member_manifest_digest: NonEmptyString
    task: VisionTask
    reasons: tuple[ShadowRouteReason, ...]
    sample_ratio: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    policy_version: SchemaVersion
    status: ShadowRouteStatus
    created_at: Rfc3339Timestamp


__all__ = [
    "InferenceFailure",
    "InferenceStatus",
    "ModelInference",
    "ModelInferenceUsage",
    "Retryability",
    "ShadowRoute",
    "ShadowRouteReason",
    "ShadowRouteStatus",
    "VisionTask",
]
