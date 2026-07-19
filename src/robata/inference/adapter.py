"""Adapter protocol for vision model inference.

Architecture V1.1 — Section 9 (VisionModelAdapter interface).

Defines the normalized contract between the inference orchestrator and
concrete model adapters (e.g. Qwen, GPT). All adapters implement the
VisionModelAdapter protocol; the orchestrator works only through this
interface.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid

from robata.inference.models import (
    InferenceFailure,
    InferenceStatus,
    ModelCapabilities,
    ModelInferenceUsage,
    Retryability,
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
    output_schema: JsonSchemaRef
    capability_snapshot_id: OpaqueUuid
    capability_snapshot_digest: Sha256Digest
    model_policy_version: SchemaVersion
    generation_config: dict[str, object]
    provider_idempotency_key: NonEmptyString
    timeout_ms: PositiveInt
    metadata: dict[NonEmptyString, NonEmptyString]


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
        - Translating provider responses into the normalized envelope.
        - Preserving the raw response as an immutable artifact.
        - Returning provider identity and request ID.

        The adapter must never decide whether an output becomes production
        truth; that is the orchestrator's responsibility.
        """
        ...


__all__ = [
    "JsonSchemaRef",
    "NormalizedOutputEnvelope",
    "PackageInput",
    "VisionInferenceFailure",
    "VisionInferenceRequest",
    "VisionInferenceSuccess",
    "VisionModelAdapter",
    "VisionUsage",
]
