"""Typed pre-EOS inference evidence and incremental window results.

These contracts are additive to the MCAP-bound inference models. Logical
invocation identity remains stable across retries; dispatch attempts and their
accepted-call evidence are separate immutable facts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Self, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.inference import (
    InferenceFailure,
    InferenceStatus,
    ModelInferenceUsage,
)
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    NonEmptyString,
    NonNegativeInt,
    StreamPurpose,
    StreamSubjectRef,
    StreamSubjectType,
    TerminalOutcome,
    validate_rfc3339,
)
from robata.contracts.stream_window import (
    StreamInferenceAttemptIdentity,
    StreamInferenceLogicalIdentity,
)

STREAM_INFERENCE_EVIDENCE_WIRE_VERSION: Literal["1.0"] = "1.0"
STREAM_INFERENCE_INTENT_SCHEMA_ID = "https://schemas.robata.dev/stream-inference-intent"
STREAM_INFERENCE_INTENT_SCHEMA_VERSION = "1.0.0"
STREAM_ACCEPTED_CALL_SCHEMA_ID = "https://schemas.robata.dev/stream-accepted-call-evidence"
STREAM_ACCEPTED_CALL_SCHEMA_VERSION = "1.0.0"
STREAM_INFERENCE_TERMINAL_SCHEMA_ID = "https://schemas.robata.dev/stream-inference-terminal"
STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION = "1.0.0"
STREAM_WINDOW_RESULT_SCHEMA_ID = "https://schemas.robata.dev/stream-window-result"
STREAM_WINDOW_RESULT_SCHEMA_VERSION = "1.0.0"

INTENT_PROJECTION_VERSION = "stream-inference-intent-semantic-v1"
ACCEPTED_CALL_PROJECTION_VERSION = "stream-accepted-call-semantic-v1"
TERMINAL_PROJECTION_VERSION = "stream-inference-terminal-semantic-v1"
WINDOW_RESULT_PROJECTION_VERSION = "stream-window-result-semantic-v1"
WINDOW_RESULT_IDENTITY_POLICY_VERSION = "stream-window-result-identity-v1"
WINDOW_RESULT_KEY_NAMESPACE = "stream-window-result-v1"


def _namespace(label: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"robata:stream-namespace:{label}")


STREAM_WINDOW_RESULT_V1_NAMESPACE = _namespace(WINDOW_RESULT_KEY_NAMESPACE)


def _timestamp(value: str, field_name: str) -> datetime:
    validate_rfc3339(value, field_name)
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized).astimezone(UTC)


class StreamInputPlanReference(StrictModel):
    """Semantic plan identity plus its exact immutable serialized artifact."""

    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    exact_artifact_ref: ArtifactEvidenceRef

    @model_validator(mode="after")
    def validate_media_type(self) -> Self:
        if self.exact_artifact_ref.media_type != "application/json":
            raise ValueError("stream input plan artifact must use application/json")
        return self


class StreamInferenceIntent(StrictModel):
    """Durable typed-source dispatch intent written before adapter execution."""

    schema_version: Literal["1.0"] = STREAM_INFERENCE_EVIDENCE_WIRE_VERSION
    schema_ref: SchemaRef
    window_subject: StreamSubjectRef
    logical_identity: StreamInferenceLogicalIdentity
    attempt_identity: StreamInferenceAttemptIdentity
    input_plan: StreamInputPlanReference
    provider_idempotency_key: NonEmptyString
    dispatch_policy_version: SchemaVersion
    intent_projection_version: SchemaVersion = INTENT_PROJECTION_VERSION
    intent_semantic_sha256: Sha256Digest
    created_at: Rfc3339Timestamp

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        _timestamp(value, "created_at")
        return value

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.intent_projection_version != INTENT_PROJECTION_VERSION:
            raise ValueError("stream inference intent uses the registered projection version")
        logical = self.logical_identity
        if self.window_subject.subject_type is not StreamSubjectType.INCREMENTAL_WINDOW:
            raise ValueError("stream inference intent requires an incremental-window subject")
        if (
            self.window_subject.subject_key != logical.window_key
            or self.window_subject.subject_semantic_sha256 != logical.window_semantic_sha256
            or self.input_plan.input_plan_semantic_sha256 != logical.input_plan_semantic_sha256
        ):
            raise ValueError("stream inference intent does not match its logical identity")
        if self.attempt_identity.stream_inference_logical_id != logical.stream_inference_logical_id:
            raise ValueError("stream inference attempt does not match its logical identity")
        expected = stream_inference_intent_semantic_sha256(self)
        if self.intent_semantic_sha256 != expected:
            raise ValueError("intent_semantic_sha256 does not match the intent projection")
        return self


def stream_inference_intent_semantic_projection(
    intent: StreamInferenceIntent,
) -> dict[str, object]:
    return {
        "intent_projection_version": intent.intent_projection_version,
        "stream_inference_logical_id": intent.logical_identity.stream_inference_logical_id,
        "inference_semantic_sha256": intent.logical_identity.inference_semantic_sha256,
        "inference_attempt_id": intent.attempt_identity.inference_attempt_id,
        "inference_attempt_key": intent.attempt_identity.inference_attempt_key,
        "input_plan_id": intent.input_plan.input_plan_id,
        "input_plan_semantic_sha256": intent.input_plan.input_plan_semantic_sha256,
        "provider_idempotency_key": intent.provider_idempotency_key,
        "dispatch_policy_version": intent.dispatch_policy_version,
    }


def stream_inference_intent_semantic_sha256(intent: StreamInferenceIntent) -> Sha256Digest:
    return semantic_sha256(stream_inference_intent_semantic_projection(intent))


def create_stream_inference_intent(
    *,
    schema_ref: SchemaRef,
    window_subject: StreamSubjectRef,
    logical_identity: StreamInferenceLogicalIdentity,
    attempt_identity: StreamInferenceAttemptIdentity,
    input_plan: StreamInputPlanReference,
    provider_idempotency_key: str,
    dispatch_policy_version: str,
    created_at: str,
) -> StreamInferenceIntent:
    values = {
        "schema_ref": schema_ref,
        "window_subject": window_subject,
        "logical_identity": logical_identity,
        "attempt_identity": attempt_identity,
        "input_plan": input_plan,
        "provider_idempotency_key": provider_idempotency_key,
        "dispatch_policy_version": dispatch_policy_version,
        "created_at": created_at,
    }
    draft = StreamInferenceIntent.model_construct(
        intent_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    return StreamInferenceIntent(
        intent_semantic_sha256=stream_inference_intent_semantic_sha256(draft),
        **cast(dict[str, Any], values),
    )


class StreamInferenceIntentReference(StrictModel):
    """Exact stored intent reference with the identity fields needed for joins."""

    intent_semantic_sha256: Sha256Digest
    stream_inference_logical_id: OpaqueUuid
    inference_attempt_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    artifact_ref: ArtifactEvidenceRef


def reference_stream_inference_intent(
    intent: StreamInferenceIntent,
    artifact_ref: ArtifactEvidenceRef,
) -> StreamInferenceIntentReference:
    return StreamInferenceIntentReference(
        intent_semantic_sha256=intent.intent_semantic_sha256,
        stream_inference_logical_id=intent.logical_identity.stream_inference_logical_id,
        inference_attempt_id=intent.attempt_identity.inference_attempt_id,
        input_plan_semantic_sha256=intent.input_plan.input_plan_semantic_sha256,
        artifact_ref=artifact_ref,
    )


class StreamAcceptedCallEvidence(StrictModel):
    """Complete accepted provider-call closure for one exact dispatch attempt."""

    schema_version: Literal["1.0"] = STREAM_INFERENCE_EVIDENCE_WIRE_VERSION
    schema_ref: SchemaRef
    intent_ref: StreamInferenceIntentReference
    status: InferenceStatus
    provider_request_id: NonEmptyString | None = None
    provider_exchange_ref: ArtifactEvidenceRef
    output_semantic_sha256: Sha256Digest | None = None
    normalized_output_ref: ArtifactEvidenceRef | None = None
    output_valid: bool
    usage: ModelInferenceUsage
    latency_ms: NonNegativeInt
    failure: InferenceFailure | None = None
    accepted_call_projection_version: SchemaVersion = ACCEPTED_CALL_PROJECTION_VERSION
    accepted_call_semantic_sha256: Sha256Digest
    completed_at: Rfc3339Timestamp

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: str) -> str:
        _timestamp(value, "completed_at")
        return value

    @model_validator(mode="after")
    def validate_call_shape(self) -> Self:
        if self.accepted_call_projection_version != ACCEPTED_CALL_PROJECTION_VERSION:
            raise ValueError("accepted call uses the registered projection version")
        succeeded = self.status is InferenceStatus.SUCCEEDED
        output_complete = (
            self.output_semantic_sha256 is not None and self.normalized_output_ref is not None
        )
        if succeeded and (
            not output_complete
            or not self.output_valid
            or self.failure is not None
            or self.provider_request_id is None
        ):
            raise ValueError("successful accepted call requires valid output and provider evidence")
        if not succeeded and (
            self.output_semantic_sha256 is not None
            or self.normalized_output_ref is not None
            or self.output_valid
            or self.failure is None
        ):
            raise ValueError("non-success accepted call requires failure-only evidence")
        expected = stream_accepted_call_semantic_sha256(self)
        if self.accepted_call_semantic_sha256 != expected:
            raise ValueError(
                "accepted_call_semantic_sha256 does not match the accepted-call projection"
            )
        return self


def stream_accepted_call_semantic_projection(
    call: StreamAcceptedCallEvidence,
) -> dict[str, object]:
    return {
        "accepted_call_projection_version": call.accepted_call_projection_version,
        "intent_semantic_sha256": call.intent_ref.intent_semantic_sha256,
        "stream_inference_logical_id": call.intent_ref.stream_inference_logical_id,
        "inference_attempt_id": call.intent_ref.inference_attempt_id,
        "input_plan_semantic_sha256": call.intent_ref.input_plan_semantic_sha256,
        "status": call.status.value,
        "provider_request_id": call.provider_request_id,
        "provider_exchange_exact_sha256": call.provider_exchange_ref.exact_sha256,
        "output_semantic_sha256": call.output_semantic_sha256,
        "output_valid": call.output_valid,
        "failure": call.failure.model_dump(mode="json") if call.failure else None,
    }


def stream_accepted_call_semantic_sha256(
    call: StreamAcceptedCallEvidence,
) -> Sha256Digest:
    return semantic_sha256(stream_accepted_call_semantic_projection(call))


def create_stream_accepted_call_evidence(
    *,
    schema_ref: SchemaRef,
    intent_ref: StreamInferenceIntentReference,
    status: InferenceStatus,
    provider_exchange_ref: ArtifactEvidenceRef,
    output_valid: bool,
    usage: ModelInferenceUsage,
    latency_ms: int,
    completed_at: str,
    provider_request_id: str | None = None,
    output_semantic_sha256: Sha256Digest | None = None,
    normalized_output_ref: ArtifactEvidenceRef | None = None,
    failure: InferenceFailure | None = None,
) -> StreamAcceptedCallEvidence:
    values = {
        "schema_ref": schema_ref,
        "intent_ref": intent_ref,
        "status": status,
        "provider_request_id": provider_request_id,
        "provider_exchange_ref": provider_exchange_ref,
        "output_semantic_sha256": output_semantic_sha256,
        "normalized_output_ref": normalized_output_ref,
        "output_valid": output_valid,
        "usage": usage,
        "latency_ms": latency_ms,
        "failure": failure,
        "completed_at": completed_at,
    }
    draft = StreamAcceptedCallEvidence.model_construct(
        accepted_call_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    return StreamAcceptedCallEvidence(
        accepted_call_semantic_sha256=stream_accepted_call_semantic_sha256(draft),
        **cast(dict[str, Any], values),
    )


class StreamAcceptedCallReference(StrictModel):
    """Exact accepted-call artifact selected by one terminal record."""

    accepted_call_semantic_sha256: Sha256Digest
    stream_inference_logical_id: OpaqueUuid
    inference_attempt_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    status: InferenceStatus
    artifact_ref: ArtifactEvidenceRef


def reference_stream_accepted_call(
    call: StreamAcceptedCallEvidence,
    artifact_ref: ArtifactEvidenceRef,
) -> StreamAcceptedCallReference:
    return StreamAcceptedCallReference(
        accepted_call_semantic_sha256=call.accepted_call_semantic_sha256,
        stream_inference_logical_id=call.intent_ref.stream_inference_logical_id,
        inference_attempt_id=call.intent_ref.inference_attempt_id,
        input_plan_semantic_sha256=call.intent_ref.input_plan_semantic_sha256,
        status=call.status,
        artifact_ref=artifact_ref,
    )


class StreamInferenceTerminal(StrictModel):
    """Attempt terminal that selects one atomically accepted call closure."""

    schema_version: Literal["1.0"] = STREAM_INFERENCE_EVIDENCE_WIRE_VERSION
    schema_ref: SchemaRef
    logical_identity: StreamInferenceLogicalIdentity
    attempt_identity: StreamInferenceAttemptIdentity
    intent_ref: StreamInferenceIntentReference
    accepted_call_ref: StreamAcceptedCallReference
    status: InferenceStatus
    terminal_policy_version: SchemaVersion
    terminal_projection_version: SchemaVersion = TERMINAL_PROJECTION_VERSION
    terminal_semantic_sha256: Sha256Digest
    completed_at: Rfc3339Timestamp

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: str) -> str:
        _timestamp(value, "completed_at")
        return value

    @model_validator(mode="after")
    def validate_terminal_bindings(self) -> Self:
        if self.terminal_projection_version != TERMINAL_PROJECTION_VERSION:
            raise ValueError("stream inference terminal uses the registered projection version")
        logical_id = self.logical_identity.stream_inference_logical_id
        attempt_id = self.attempt_identity.inference_attempt_id
        plan_digest = self.logical_identity.input_plan_semantic_sha256
        if self.attempt_identity.stream_inference_logical_id != logical_id:
            raise ValueError("terminal attempt does not match its logical identity")
        if any(
            reference.stream_inference_logical_id != logical_id
            or reference.inference_attempt_id != attempt_id
            or reference.input_plan_semantic_sha256 != plan_digest
            for reference in (self.intent_ref, self.accepted_call_ref)
        ):
            raise ValueError("terminal evidence references do not match the inference identity")
        if self.status is not self.accepted_call_ref.status:
            raise ValueError("terminal status does not match accepted-call evidence")
        expected = stream_inference_terminal_semantic_sha256(self)
        if self.terminal_semantic_sha256 != expected:
            raise ValueError("terminal_semantic_sha256 does not match the terminal projection")
        return self


def stream_inference_terminal_semantic_projection(
    terminal: StreamInferenceTerminal,
) -> dict[str, object]:
    return {
        "terminal_projection_version": terminal.terminal_projection_version,
        "inference_semantic_sha256": terminal.logical_identity.inference_semantic_sha256,
        "stream_inference_logical_id": terminal.logical_identity.stream_inference_logical_id,
        "inference_attempt_id": terminal.attempt_identity.inference_attempt_id,
        "intent_semantic_sha256": terminal.intent_ref.intent_semantic_sha256,
        "accepted_call_semantic_sha256": (terminal.accepted_call_ref.accepted_call_semantic_sha256),
        "status": terminal.status.value,
        "terminal_policy_version": terminal.terminal_policy_version,
    }


def stream_inference_terminal_semantic_sha256(
    terminal: StreamInferenceTerminal,
) -> Sha256Digest:
    return semantic_sha256(stream_inference_terminal_semantic_projection(terminal))


def create_stream_inference_terminal(
    *,
    schema_ref: SchemaRef,
    logical_identity: StreamInferenceLogicalIdentity,
    attempt_identity: StreamInferenceAttemptIdentity,
    intent_ref: StreamInferenceIntentReference,
    accepted_call_ref: StreamAcceptedCallReference,
    status: InferenceStatus,
    terminal_policy_version: str,
    completed_at: str,
) -> StreamInferenceTerminal:
    values = {
        "schema_ref": schema_ref,
        "logical_identity": logical_identity,
        "attempt_identity": attempt_identity,
        "intent_ref": intent_ref,
        "accepted_call_ref": accepted_call_ref,
        "status": status,
        "terminal_policy_version": terminal_policy_version,
        "completed_at": completed_at,
    }
    draft = StreamInferenceTerminal.model_construct(
        terminal_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    return StreamInferenceTerminal(
        terminal_semantic_sha256=stream_inference_terminal_semantic_sha256(draft),
        **cast(dict[str, Any], values),
    )


class StreamInferenceTerminalReference(StrictModel):
    """Selected terminal lineage retained by an incremental window result."""

    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    purpose: StreamPurpose
    inference_semantic_sha256: Sha256Digest
    stream_inference_logical_id: OpaqueUuid
    inference_attempt_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    status: InferenceStatus
    terminal_semantic_sha256: Sha256Digest
    artifact_ref: ArtifactEvidenceRef


def reference_stream_inference_terminal(
    terminal: StreamInferenceTerminal,
    artifact_ref: ArtifactEvidenceRef,
) -> StreamInferenceTerminalReference:
    logical = terminal.logical_identity
    return StreamInferenceTerminalReference(
        window_key=logical.window_key,
        window_semantic_sha256=logical.window_semantic_sha256,
        purpose=logical.purpose,
        inference_semantic_sha256=logical.inference_semantic_sha256,
        stream_inference_logical_id=logical.stream_inference_logical_id,
        inference_attempt_id=terminal.attempt_identity.inference_attempt_id,
        input_plan_semantic_sha256=logical.input_plan_semantic_sha256,
        status=terminal.status,
        terminal_semantic_sha256=terminal.terminal_semantic_sha256,
        artifact_ref=artifact_ref,
    )


class StreamWindowResult(StrictModel):
    """One immutable, non-final result for an incremental window subject."""

    schema_version: Literal["1.0"] = STREAM_INFERENCE_EVIDENCE_WIRE_VERSION
    schema_ref: SchemaRef
    subject_type: Literal[StreamSubjectType.WINDOW_RESULT] = StreamSubjectType.WINDOW_RESULT
    window_result_id: OpaqueUuid
    window_result_key: NonEmptyString
    window_result_semantic_sha256: Sha256Digest
    window_subject: StreamSubjectRef
    purpose: StreamPurpose
    terminal_outcome: TerminalOutcome
    accepted_terminals: tuple[StreamInferenceTerminalReference, ...]
    result_semantic_evidence_sha256: Sha256Digest
    result_evidence_ref: ArtifactEvidenceRef
    reduction_policy_version: SchemaVersion
    result_projection_version: SchemaVersion = WINDOW_RESULT_PROJECTION_VERSION
    result_identity_policy_version: SchemaVersion = WINDOW_RESULT_IDENTITY_POLICY_VERSION
    created_at: Rfc3339Timestamp

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        _timestamp(value, "created_at")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.result_projection_version != WINDOW_RESULT_PROJECTION_VERSION:
            raise ValueError("window result uses the registered projection version")
        if self.result_identity_policy_version != WINDOW_RESULT_IDENTITY_POLICY_VERSION:
            raise ValueError("window result uses the registered identity policy version")
        if self.window_subject.subject_type is not StreamSubjectType.INCREMENTAL_WINDOW:
            raise ValueError("window result requires an incremental-window subject")
        terminal_order = tuple(
            (terminal.stream_inference_logical_id, terminal.inference_attempt_id)
            for terminal in self.accepted_terminals
        )
        if terminal_order != tuple(sorted(terminal_order)) or len(terminal_order) != len(
            set(terminal_order)
        ):
            raise ValueError("accepted inference terminals must be unique and canonical")
        logical_ids = tuple(
            terminal.stream_inference_logical_id for terminal in self.accepted_terminals
        )
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("window result may select only one attempt per logical inference")
        if any(
            terminal.window_key != self.window_subject.subject_key
            or terminal.window_semantic_sha256 != self.window_subject.subject_semantic_sha256
            or terminal.purpose is not self.purpose
            for terminal in self.accepted_terminals
        ):
            raise ValueError("accepted inference terminals do not match the window result")
        if self.terminal_outcome in {TerminalOutcome.SUCCEEDED, TerminalOutcome.NO_EVENTS} and (
            not self.accepted_terminals
            or any(
                terminal.status is not InferenceStatus.SUCCEEDED
                for terminal in self.accepted_terminals
            )
        ):
            raise ValueError("successful window result requires successful inference terminals")
        expected = stream_window_result_semantic_sha256(self)
        if self.window_result_semantic_sha256 != expected:
            raise ValueError("window_result_semantic_sha256 does not match the result projection")
        if self.window_result_key != derive_stream_window_result_key(expected):
            raise ValueError("window_result_key does not match its semantic digest")
        if self.window_result_id != derive_stream_window_result_id(expected):
            raise ValueError("window_result_id does not match its semantic digest")
        return self

    def reference(self) -> StreamSubjectRef:
        return StreamSubjectRef(
            subject_type=StreamSubjectType.WINDOW_RESULT,
            subject_key=self.window_result_key,
            subject_semantic_sha256=self.window_result_semantic_sha256,
            capture_scope_digest=self.window_subject.capture_scope_digest,
            identity_policy_version=self.result_identity_policy_version,
            schema_ref=self.schema_ref,
        )


def stream_window_result_semantic_projection(result: StreamWindowResult) -> dict[str, object]:
    return {
        "result_projection_version": result.result_projection_version,
        "result_identity_policy_version": result.result_identity_policy_version,
        "window_key": result.window_subject.subject_key,
        "window_semantic_sha256": result.window_subject.subject_semantic_sha256,
        "capture_scope_digest": result.window_subject.capture_scope_digest,
        "purpose": result.purpose.value,
        "terminal_outcome": result.terminal_outcome.value,
        "ordered_terminal_semantic_sha256_values": [
            terminal.terminal_semantic_sha256 for terminal in result.accepted_terminals
        ],
        "result_semantic_evidence_sha256": result.result_semantic_evidence_sha256,
        "reduction_policy_version": result.reduction_policy_version,
    }


def stream_window_result_semantic_sha256(result: StreamWindowResult) -> Sha256Digest:
    return semantic_sha256(stream_window_result_semantic_projection(result))


def derive_stream_window_result_key(result_digest: Sha256Digest) -> str:
    return f"{WINDOW_RESULT_KEY_NAMESPACE}:{result_digest}"


def derive_stream_window_result_id(result_digest: Sha256Digest) -> OpaqueUuid:
    return str(
        uuid5(STREAM_WINDOW_RESULT_V1_NAMESPACE, derive_stream_window_result_key(result_digest))
    )


def create_stream_window_result(
    *,
    schema_ref: SchemaRef,
    window_subject: StreamSubjectRef,
    purpose: StreamPurpose,
    terminal_outcome: TerminalOutcome,
    accepted_terminals: tuple[StreamInferenceTerminalReference, ...],
    result_semantic_evidence_sha256: Sha256Digest,
    result_evidence_ref: ArtifactEvidenceRef,
    reduction_policy_version: str,
    created_at: str,
) -> StreamWindowResult:
    ordered = tuple(
        sorted(
            accepted_terminals,
            key=lambda item: (item.stream_inference_logical_id, item.inference_attempt_id),
        )
    )
    values = {
        "schema_ref": schema_ref,
        "window_subject": window_subject,
        "purpose": purpose,
        "terminal_outcome": terminal_outcome,
        "accepted_terminals": ordered,
        "result_semantic_evidence_sha256": result_semantic_evidence_sha256,
        "result_evidence_ref": result_evidence_ref,
        "reduction_policy_version": reduction_policy_version,
        "created_at": created_at,
    }
    draft = StreamWindowResult.model_construct(
        window_result_id="00000000-0000-0000-0000-000000000000",
        window_result_key="x",
        window_result_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = stream_window_result_semantic_sha256(draft)
    return StreamWindowResult(
        window_result_id=derive_stream_window_result_id(digest),
        window_result_key=derive_stream_window_result_key(digest),
        window_result_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


__all__ = [
    "ACCEPTED_CALL_PROJECTION_VERSION",
    "INTENT_PROJECTION_VERSION",
    "STREAM_ACCEPTED_CALL_SCHEMA_ID",
    "STREAM_ACCEPTED_CALL_SCHEMA_VERSION",
    "STREAM_INFERENCE_EVIDENCE_WIRE_VERSION",
    "STREAM_INFERENCE_INTENT_SCHEMA_ID",
    "STREAM_INFERENCE_INTENT_SCHEMA_VERSION",
    "STREAM_INFERENCE_TERMINAL_SCHEMA_ID",
    "STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION",
    "STREAM_WINDOW_RESULT_SCHEMA_ID",
    "STREAM_WINDOW_RESULT_SCHEMA_VERSION",
    "STREAM_WINDOW_RESULT_V1_NAMESPACE",
    "TERMINAL_PROJECTION_VERSION",
    "WINDOW_RESULT_IDENTITY_POLICY_VERSION",
    "WINDOW_RESULT_KEY_NAMESPACE",
    "WINDOW_RESULT_PROJECTION_VERSION",
    "StreamAcceptedCallEvidence",
    "StreamAcceptedCallReference",
    "StreamInferenceIntent",
    "StreamInferenceIntentReference",
    "StreamInferenceTerminal",
    "StreamInferenceTerminalReference",
    "StreamInputPlanReference",
    "StreamWindowResult",
    "create_stream_accepted_call_evidence",
    "create_stream_inference_intent",
    "create_stream_inference_terminal",
    "create_stream_window_result",
    "derive_stream_window_result_id",
    "derive_stream_window_result_key",
    "reference_stream_accepted_call",
    "reference_stream_inference_intent",
    "reference_stream_inference_terminal",
    "stream_accepted_call_semantic_projection",
    "stream_accepted_call_semantic_sha256",
    "stream_inference_intent_semantic_projection",
    "stream_inference_intent_semantic_sha256",
    "stream_inference_terminal_semantic_projection",
    "stream_inference_terminal_semantic_sha256",
    "stream_window_result_semantic_projection",
    "stream_window_result_semantic_sha256",
]
