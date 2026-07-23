from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from robata.contracts.inference import (
    InferenceFailure,
    InferenceStatus,
    ModelInferenceUsage,
    Retryability,
)
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    StreamPurpose,
    StreamSubjectRef,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_inference import (
    StreamInputPlanReference,
    create_stream_accepted_call_evidence,
    create_stream_inference_intent,
    create_stream_inference_terminal,
    create_stream_window_result,
    reference_stream_accepted_call,
    reference_stream_inference_intent,
    reference_stream_inference_terminal,
)
from robata.contracts.stream_window import (
    create_stream_inference_attempt_identity,
    create_stream_inference_identity,
)

_CAPTURE_DIGEST = "1" * 64
_WINDOW_DIGEST = "2" * 64
_PLAN_DIGEST = "3" * 64
_NOW = "2026-07-22T12:00:00Z"


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _digest(number: int) -> str:
    return f"{number:064x}"


def _schema(name: str, number: int) -> SchemaRef:
    return SchemaRef(
        schema_id=f"https://schemas.robata.dev/{name}",
        version="1.0.0",
        artifact_id=_uuid(number),
        sha256=_digest(1000 + number),
    )


def _artifact(
    name: str,
    number: int,
    *,
    media_type: str = "application/json",
) -> ArtifactEvidenceRef:
    return ArtifactEvidenceRef(
        artifact_id=_uuid(100 + number),
        exact_sha256=_digest(2000 + number),
        byte_count=100 + number,
        media_type=media_type,
        schema_ref=_schema(name, 100 + number),
    )


def _successful_chain(
    *, plan_digest: str = _PLAN_DIGEST, attempt_number: int = 1
) -> dict[str, object]:
    window = StreamSubjectRef(
        subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
        subject_key=f"incremental-window-v1:{_WINDOW_DIGEST}",
        subject_semantic_sha256=_WINDOW_DIGEST,
        capture_scope_digest=_CAPTURE_DIGEST,
        identity_policy_version="incremental-window-identity-v1",
        schema_ref=_schema("incremental-window", 1),
    )
    logical = create_stream_inference_identity(
        schema_ref=_schema("stream-inference", 2),
        window_key=window.subject_key,
        window_semantic_sha256=window.subject_semantic_sha256,
        purpose=StreamPurpose.QA_COARSE,
        input_plan_semantic_sha256=plan_digest,
    )
    attempt = create_stream_inference_attempt_identity(
        schema_ref=_schema("stream-inference-attempt", 3),
        stream_inference_logical_id=logical.stream_inference_logical_id,
        attempt_number=attempt_number,
    )
    plan = StreamInputPlanReference(
        input_plan_id=_uuid(20),
        input_plan_semantic_sha256=plan_digest,
        exact_artifact_ref=_artifact("inference-input-plan", 20),
    )
    intent = create_stream_inference_intent(
        schema_ref=_schema("stream-inference-intent", 4),
        window_subject=window,
        logical_identity=logical,
        attempt_identity=attempt,
        input_plan=plan,
        provider_idempotency_key=f"provider-key-{attempt_number}",
        dispatch_policy_version="dispatch-v1",
        created_at=_NOW,
    )
    intent_ref = reference_stream_inference_intent(
        intent,
        _artifact("stream-inference-intent", 21 + attempt_number),
    )
    accepted = create_stream_accepted_call_evidence(
        schema_ref=_schema("stream-accepted-call-evidence", 5),
        intent_ref=intent_ref,
        status=InferenceStatus.SUCCEEDED,
        provider_request_id=f"provider-request-{attempt_number}",
        provider_exchange_ref=_artifact("provider-exchange", 30 + attempt_number),
        output_semantic_sha256=_digest(3100 + attempt_number),
        normalized_output_ref=_artifact("normalized-output", 40 + attempt_number),
        output_valid=True,
        usage=ModelInferenceUsage(input_frames=12, input_images=12),
        latency_ms=50,
        completed_at=_NOW,
    )
    accepted_ref = reference_stream_accepted_call(
        accepted,
        _artifact("stream-accepted-call-evidence", 50 + attempt_number),
    )
    terminal = create_stream_inference_terminal(
        schema_ref=_schema("stream-inference-terminal", 6),
        logical_identity=logical,
        attempt_identity=attempt,
        intent_ref=intent_ref,
        accepted_call_ref=accepted_ref,
        status=InferenceStatus.SUCCEEDED,
        terminal_policy_version="terminal-v1",
        completed_at=_NOW,
    )
    terminal_ref = reference_stream_inference_terminal(
        terminal,
        _artifact("stream-inference-terminal", 60 + attempt_number),
    )
    return {
        "window": window,
        "logical": logical,
        "attempt": attempt,
        "plan": plan,
        "intent": intent,
        "intent_ref": intent_ref,
        "accepted": accepted,
        "accepted_ref": accepted_ref,
        "terminal": terminal,
        "terminal_ref": terminal_ref,
    }


def test_stream_inference_chain_preserves_logical_and_attempt_identity() -> None:
    first = _successful_chain(attempt_number=1)
    retry = _successful_chain(attempt_number=2)

    assert first["logical"] == retry["logical"]
    assert first["attempt"] != retry["attempt"]
    assert first["intent"] != retry["intent"]
    assert first["terminal"] != retry["terminal"]

    result = create_stream_window_result(
        schema_ref=_schema("stream-window-result", 7),
        window_subject=first["window"],  # type: ignore[arg-type]
        purpose=StreamPurpose.QA_COARSE,
        terminal_outcome=TerminalOutcome.SUCCEEDED,
        accepted_terminals=(first["terminal_ref"],),  # type: ignore[arg-type]
        result_semantic_evidence_sha256=_digest(5000),
        result_evidence_ref=_artifact("window-result-evidence", 70),
        reduction_policy_version="window-reduction-v1",
        created_at=_NOW,
    )

    assert result.reference().subject_type is StreamSubjectType.WINDOW_RESULT
    assert result.reference().capture_scope_digest == _CAPTURE_DIGEST
    assert result.window_result_key.endswith(result.window_result_semantic_sha256)


def test_stream_intent_rejects_cross_bound_plan_or_attempt() -> None:
    chain = _successful_chain()
    intent = chain["intent"]
    assert hasattr(intent, "model_dump")
    payload = intent.model_dump(mode="json")  # type: ignore[union-attr]
    payload["input_plan"]["input_plan_semantic_sha256"] = "f" * 64

    with pytest.raises(ValidationError, match="logical identity"):
        type(intent).model_validate_json(json.dumps(payload))

    other = _successful_chain(plan_digest="e" * 64)
    payload = intent.model_dump(mode="json")  # type: ignore[union-attr]
    payload["attempt_identity"] = other["attempt"].model_dump(mode="json")  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="attempt does not match"):
        type(intent).model_validate_json(json.dumps(payload))


def test_accepted_call_requires_a_complete_success_or_failure_shape() -> None:
    chain = _successful_chain()
    accepted = chain["accepted"]
    assert hasattr(accepted, "model_dump")
    payload = accepted.model_dump(mode="json")  # type: ignore[union-attr]
    payload["output_valid"] = False
    with pytest.raises(ValidationError, match="successful accepted call"):
        type(accepted).model_validate_json(json.dumps(payload))

    failed = create_stream_accepted_call_evidence(
        schema_ref=_schema("stream-accepted-call-evidence", 5),
        intent_ref=chain["intent_ref"],  # type: ignore[arg-type]
        status=InferenceStatus.TIMEOUT,
        provider_exchange_ref=_artifact("provider-timeout-evidence", 80),
        output_valid=False,
        usage=ModelInferenceUsage(input_frames=12, input_images=12),
        latency_ms=5000,
        failure=InferenceFailure(
            code="PROVIDER_TIMEOUT",
            detail="provider deadline expired",
            retryability=Retryability.RETRYABLE,
        ),
        completed_at=_NOW,
    )
    assert failed.failure is not None
    assert failed.normalized_output_ref is None


def test_terminal_rejects_an_accepted_call_from_another_attempt() -> None:
    first = _successful_chain(attempt_number=1)
    retry = _successful_chain(attempt_number=2)
    terminal = first["terminal"]
    assert hasattr(terminal, "model_dump")
    payload = terminal.model_dump(mode="json")  # type: ignore[union-attr]
    payload["accepted_call_ref"] = retry["accepted_ref"].model_dump(mode="json")  # type: ignore[union-attr]

    with pytest.raises(ValidationError, match="do not match the inference identity"):
        type(terminal).model_validate_json(json.dumps(payload))


def test_window_result_selects_only_one_attempt_per_logical_inference() -> None:
    first = _successful_chain(attempt_number=1)
    retry = _successful_chain(attempt_number=2)

    with pytest.raises(ValidationError, match="only one attempt"):
        create_stream_window_result(
            schema_ref=_schema("stream-window-result", 7),
            window_subject=first["window"],  # type: ignore[arg-type]
            purpose=StreamPurpose.QA_COARSE,
            terminal_outcome=TerminalOutcome.SUCCEEDED,
            accepted_terminals=(  # type: ignore[arg-type]
                first["terminal_ref"],
                retry["terminal_ref"],
            ),
            result_semantic_evidence_sha256=_digest(5000),
            result_evidence_ref=_artifact("window-result-evidence", 70),
            reduction_policy_version="window-reduction-v1",
            created_at=_NOW,
        )


def test_semantic_result_identity_excludes_exact_artifact_serialization() -> None:
    chain = _successful_chain()
    common = {
        "schema_ref": _schema("stream-window-result", 7),
        "window_subject": chain["window"],
        "purpose": StreamPurpose.QA_COARSE,
        "terminal_outcome": TerminalOutcome.SUCCEEDED,
        "accepted_terminals": (chain["terminal_ref"],),
        "result_semantic_evidence_sha256": _digest(5000),
        "reduction_policy_version": "window-reduction-v1",
        "created_at": _NOW,
    }
    first = create_stream_window_result(
        **common,  # type: ignore[arg-type]
        result_evidence_ref=_artifact("window-result-evidence", 70),
    )
    second = create_stream_window_result(
        **common,  # type: ignore[arg-type]
        result_evidence_ref=_artifact("window-result-evidence", 71),
    )

    assert first.window_result_semantic_sha256 == second.window_result_semantic_sha256
    assert first.result_evidence_ref != second.result_evidence_ref
