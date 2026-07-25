"""Adapter protocol for vision model inference.

Architecture V1.1 — Section 9 (VisionModelAdapter interface).

Defines the normalized contract between the inference orchestrator and
concrete model adapters (e.g. Qwen, GPT). All adapters implement the
VisionModelAdapter protocol; the orchestrator works only through this
interface.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import (
    InferenceFailure,
    InferenceStatus,
    ModelCapabilities,
    VisionTask,
)

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

# ---------------------------------------------------------------------------
# Request and envelope models
# ---------------------------------------------------------------------------


class JsonSchemaRef(StrictModel):
    """Reference to an immutable JSON Schema artifact."""

    schema_id: NonEmptyString
    version: SchemaVersion
    artifact_id: NonEmptyString
    sha256: Sha256Digest


class PackageInput(StrictModel):
    """One package member in a multi-package inference request."""

    package_id: OpaqueUuid
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest
    role: NonEmptyString
    ordinal: NonNegativeInt


class VisionInferenceRequest(StrictModel):
    """Normalized request accepted by any VisionModelAdapter implementation.

    Per Architecture V1.1 Section 9.1, this is the provider-neutral
    invocation intent. The orchestrator constructs this after selecting
    adapter, model, prompt, and capability snapshot.
    """

    schema_version: Literal["1.0"]
    logical_invocation_id: OpaqueUuid
    request_id: OpaqueUuid
    idempotency_key: NonEmptyString
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    package_set_id: OpaqueUuid | None = None
    package_inputs: tuple[PackageInput, ...]
    package_input_set_sha256: Sha256Digest
    task: VisionTask
    prompt_version: SchemaVersion
    prompt_artifact_id: NonEmptyString
    prompt_sha256: Sha256Digest
    rendered_input_digest: Sha256Digest
    input_plan_id: OpaqueUuid | None = None
    input_plan_semantic_sha256: Sha256Digest | None = None
    input_plan_part_ordinal: NonNegativeInt | None = None
    input_plan_part_count: PositiveInt | None = None
    input_plan_part_semantic_sha256: Sha256Digest | None = None
    input_plan: InferenceInputPlan | None = None
    output_schema: JsonSchemaRef
    capability_snapshot_id: OpaqueUuid
    capability_snapshot_digest: Sha256Digest
    model_policy_version: SchemaVersion
    generation_config: dict[str, object]
    provider_idempotency_key: NonEmptyString
    timeout_ms: PositiveInt
    metadata: dict[NonEmptyString, NonEmptyString]

    @model_validator(mode="after")
    def validate_input_plan_binding(self) -> Self:
        plan = self.input_plan
        if plan is None:
            if any(
                value is not None
                for value in (
                    self.input_plan_id,
                    self.input_plan_semantic_sha256,
                    self.input_plan_part_ordinal,
                    self.input_plan_part_count,
                    self.input_plan_part_semantic_sha256,
                )
            ):
                raise ValueError("input plan references require the immutable input plan")
            return self
        if (
            self.input_plan_id != plan.input_plan_id
            or self.input_plan_semantic_sha256 != plan.semantic_sha256
        ):
            raise ValueError("input plan identity binding is inconsistent")
        part_fields = (
            self.input_plan_part_ordinal,
            self.input_plan_part_count,
            self.input_plan_part_semantic_sha256,
        )
        if all(value is None for value in part_fields):
            if self.rendered_input_digest != plan.rendering_sha256:
                raise ValueError("input plan rendering binding is inconsistent")
        else:
            if any(value is None for value in part_fields):
                raise ValueError("input plan call part references must be all present")
            assert self.input_plan_part_ordinal is not None
            if self.input_plan_part_ordinal >= len(plan.call_plan.parts):
                raise ValueError("input plan call part ordinal is out of range")
            part = plan.call_plan.parts[self.input_plan_part_ordinal]
            if (
                self.input_plan_part_count != part.part_count
                or self.input_plan_part_semantic_sha256 != part.part_semantic_sha256
                or self.rendered_input_digest != part.item_manifest_sha256
            ):
                raise ValueError("input plan call part binding is inconsistent")
        if (
            self.task is not plan.subject.task
            or self.provider != plan.target.provider
            or self.model_name != plan.target.model_name
            or self.model_version != plan.target.model_version
            or self.capability_snapshot_id != plan.target.capability_snapshot_id
            or self.capability_snapshot_digest != plan.target.capability_snapshot_sha256
        ):
            raise ValueError("input plan target does not match the inference request")
        if (
            self.prompt_version != plan.prompt_output.prompt_version
            or self.prompt_sha256 != plan.prompt_output.prompt_sha256
            or self.output_schema.sha256 != plan.prompt_output.provider_response_schema_sha256
        ):
            raise ValueError("input plan prompt/output contract does not match the request")
        request_packages = tuple(
            (
                item.package_id,
                item.ordinal,
                item.package_semantic_content_sha256,
                item.package_manifest_sha256,
            )
            for item in self.package_inputs
        )
        plan_packages = tuple(
            (
                item.package_id,
                item.ordinal,
                item.semantic_content_sha256,
                item.manifest_bytes_sha256,
            )
            for item in plan.subject.packages
        )
        if request_packages != plan_packages:
            raise ValueError("input plan packages do not match the inference request")
        return self


class VisionUsage(StrictModel):
    """Normalized usage metrics returned by an adapter."""

    input_frames: NonNegativeInt
    input_images: NonNegativeInt
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cost: NonNegativeFiniteFloat | None = None
    currency: NonEmptyString | None = None


# ---------------------------------------------------------------------------
# Normalized output envelope
# ---------------------------------------------------------------------------


class NormalizedOutputEnvelope(StrictModel):
    """Task-typed normalized output wrapper.

    The payload is task-specific and validated by the orchestrator against
    the task schema referenced in the request. The envelope itself is
    provider-agnostic.
    """

    task: VisionTask
    output_schema: JsonSchemaRef
    package_input_set_sha256: Sha256Digest
    input_plan_semantic_sha256: Sha256Digest | None = None
    input_plan_part_ordinal: NonNegativeInt | None = None
    input_plan_part_semantic_sha256: Sha256Digest | None = None
    payload: dict[str, object]


# ---------------------------------------------------------------------------
# Outcome models
# ---------------------------------------------------------------------------


class VisionInferenceSuccess(StrictModel):
    """Validated success outcome from a provider adapter."""

    status: Literal[InferenceStatus.SUCCEEDED]
    provider_request_id: NonEmptyString
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    normalized_output: NormalizedOutputEnvelope
    raw_output_artifact_id: NonEmptyString
    schema_valid: Literal[True]
    reported_confidence: UnitInterval | None = None
    usage: VisionUsage
    latency_ms: NonNegativeInt


class VisionInferenceFailure(StrictModel):
    """Terminal non-success outcome from a provider adapter."""

    status: Literal[
        InferenceStatus.FAILED,
        InferenceStatus.TIMEOUT,
        InferenceStatus.CANCELLED,
        InferenceStatus.INVALID_OUTPUT,
    ]
    provider_request_id: NonEmptyString | None = None
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    normalized_output: None = None
    raw_output_artifact_id: NonEmptyString | None = None
    schema_valid: Literal[False]
    reported_confidence: None = None
    usage: VisionUsage
    latency_ms: NonNegativeInt
    failure: InferenceFailure


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


class ProviderQualificationRequestContract(StrictModel):
    """One exact P6 prompt and context contract permitted in a qualification run."""

    task: VisionTask
    prompt_artifact_id: NonEmptyString
    prompt_version: SchemaVersion
    prompt_sha256: Sha256Digest
    output_schema_sha256: Sha256Digest
    max_input_tokens: PositiveInt
    timeout_ms: PositiveInt
    generation_config_sha256: Sha256Digest


class ProviderQualificationSession(StrictModel):
    """Immutable scope for one real-provider qualification measurement."""

    session_id: OpaqueUuid
    run_namespace: NonEmptyString
    configuration_digest: Sha256Digest
    workload_manifest_digest: Sha256Digest
    request_contracts: tuple[ProviderQualificationRequestContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_request_contracts(self) -> Self:
        identities = tuple(
            (
                contract.task,
                contract.prompt_artifact_id,
                contract.prompt_version,
                contract.prompt_sha256,
                contract.output_schema_sha256,
                contract.max_input_tokens,
                contract.timeout_ms,
                contract.generation_config_sha256,
            )
            for contract in self.request_contracts
        )
        if len(set(identities)) != len(identities):
            raise ValueError("qualification request contracts must be unique")
        return self


class ProviderQualificationObserver(Protocol):
    """Optional fail-open sink for one scoped real-provider qualification run."""

    def record_provider_timing(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        request_id: str,
        logical_invocation_id: str,
        input_plan_part_ordinal: int,
        provider_image_count: int,
        input_tokens: int | None,
        output_tokens: int | None,
        provider_queue_ms: int | None,
        provider_execution_ms: int | None,
        time_to_first_token_ms: int | None,
        end_to_end_ms: int,
        input_tokens_known: bool,
        output_tokens_known: bool,
        accepted: bool,
    ) -> None:
        """Observe one terminal provider response without changing inference semantics."""
        ...

    def record_provider_http_requests(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        count: int,
    ) -> None:
        """Observe actual HTTP posts made after the adapter obtains endpoint capacity."""
        ...

    def record_provider_retries(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        count: int,
    ) -> None:
        """Observe adapter transport retry attempts represented by one retry delay."""
        ...


class VisionModelAdapter(Protocol):
    """Protocol for provider-neutral vision model adapters.

    Concrete implementations (e.g. QwenAdapter, GPTAdapter) satisfy this
    protocol. The orchestrator selects an adapter by provider name and
    delegates inference through this interface.
    """

    @property
    def provider(self) -> str:
        """Canonical provider identifier (e.g. ``'qwen'``, ``'gpt'``)."""
        ...

    async def capabilities(
        self,
        model_name: str,
        model_version: str,
    ) -> ModelCapabilities:
        """Return capability snapshot for the given model.

        The orchestrator uses this to validate that a request's task,
        media, and schema requirements are satisfied before calling
        :meth:`infer`.
        """
        ...

    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        """Execute one inference attempt.

        The adapter is responsible for:
        - Resolving package frame references into provider-accepted media.
        - Enforcing provider limits before sending.
        - Executing exactly the explicit input-plan call part bound to the request.
        - Translating provider responses into the normalized envelope.
        - Preserving the raw response as an immutable artifact.
        - Returning provider identity and request ID.

        The adapter must never decide whether an output becomes production
        truth; that is the orchestrator's responsibility.
        """
        ...


class BatchVisionModelAdapter(VisionModelAdapter, Protocol):
    """Optional extension for adapters that can dispatch one compatible batch.

    Results must retain request order. Raising from infer_batch rejects the
    entire dispatch; adapters must not expose a successful prefix.
    """

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        """Execute one ordered, purpose-compatible provider batch."""
        ...


__all__ = [
    "BatchVisionModelAdapter",
    "JsonSchemaRef",
    "NormalizedOutputEnvelope",
    "PackageInput",
    "ProviderQualificationObserver",
    "ProviderQualificationRequestContract",
    "ProviderQualificationSession",
    "VisionInferenceFailure",
    "VisionInferenceRequest",
    "VisionInferenceSuccess",
    "VisionModelAdapter",
    "VisionUsage",
]
