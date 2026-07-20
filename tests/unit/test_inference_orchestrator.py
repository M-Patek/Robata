from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    PackageInput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceFailure,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    ModelInference,
    Retryability,
    VisionTask,
    inference_attempt_selection_logical_key,
)
from robata.inference.orchestrator import (
    CapabilityValidationError,
    InferenceLedgerError,
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
    OrchestrationConfigurationError,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-19T12:00:00Z"
SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["label"],
    "properties": {"label": {"type": "string"}},
}
SCHEMA_REF = JsonSchemaRef(
    schema_id="action-output",
    version="1.0",
    artifact_id="schema-action-output",
    sha256=semantic_sha256(SCHEMA),
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _capabilities(
    *,
    task: VisionTask = VisionTask.ACTION_EVIDENCE,
    supports_json_schema: bool = True,
) -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(100),
        snapshot_digest=_digest(101),
        provider="fake",
        model_name="local-fake",
        model_version="1.0",
        supported_tasks=(task,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=16,
        max_pixels_per_image=2_000_000,
        max_payload_bytes=10_000_000,
        max_input_tokens=10_000,
        supports_json_schema=supports_json_schema,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="1.0",
        observed_at=NOW_TEXT,
    )


def _policy() -> InferencePolicy:
    return InferencePolicy(
        policy_version="policy-1",
        task=VisionTask.ACTION_EVIDENCE,
        provider="fake",
        model_name="local-fake",
        model_version="1.0",
        adapter_version="1.0",
        prompt_version="1.0",
        prompt_artifact_id="prompt-action-evidence",
        prompt_sha256=_digest(200),
        output_schema=SCHEMA_REF,
        generation_config={"temperature": 0.0},
        timeout_ms=500,
        selection_policy_version="selection-1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="1.0",
    )


def _package_input() -> PackageInput:
    return PackageInput(
        package_id=_uuid(1),
        package_semantic_content_sha256=_digest(2),
        package_manifest_sha256=_digest(3),
        role="primary",
        ordinal=0,
    )


def _request_kwargs() -> dict[str, object]:
    return {
        "task": VisionTask.ACTION_EVIDENCE,
        "package_set_id": _uuid(4),
        "mcap_id": _uuid(5),
        "camera_mapping_run_id": _uuid(6),
        "alignment_id": _uuid(7),
        "start_ns": 100,
        "end_ns": 200,
        "package_inputs": (_package_input(),),
        "rendered_input_digest": _digest(8),
        "input_config": {"input_images": 1, "payload_bytes": 100},
        "sampling_config": {"policy": "dense-v1"},
        "metadata": {"fixture": "yes"},
    }


class FakeAdapter:
    provider = "fake"

    def __init__(
        self,
        ledger: InMemoryInferenceLedger,
        outcome_factory: Callable[
            [VisionInferenceRequest], VisionInferenceSuccess | VisionInferenceFailure
        ],
        *,
        capabilities: ModelCapabilities | None = None,
        raised: BaseException | None = None,
    ) -> None:
        self.ledger = ledger
        self.outcome_factory = outcome_factory
        self._capabilities = capabilities or _capabilities()
        self.raised = raised
        self.capability_calls = 0
        self.infer_calls = 0
        self.intent_seen_before_infer = False

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        self.capability_calls += 1
        assert (model_name, model_version) == ("local-fake", "1.0")
        return self._capabilities

    async def infer(
        self, request: VisionInferenceRequest
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        self.infer_calls += 1
        matching_intents = [
            intent
            for intent in self.ledger.list_intents()
            if intent.request.request_id == request.request_id
        ]
        self.intent_seen_before_infer = bool(matching_intents)
        assert self.intent_seen_before_infer
        assert all(
            terminal.inference_id != matching_intents[0].inference_id
            for terminal in self.ledger.list_terminals()
        )
        if self.raised is not None:
            raise self.raised
        return self.outcome_factory(request)


def _success(
    request: VisionInferenceRequest,
    *,
    payload: dict[str, object] | None = None,
) -> VisionInferenceSuccess:
    return VisionInferenceSuccess(
        status=InferenceStatus.SUCCEEDED,
        provider_request_id="fake-provider-request",
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        normalized_output=NormalizedOutputEnvelope(
            task=request.task,
            output_schema=request.output_schema,
            package_input_set_sha256=request.package_input_set_sha256,
            payload=payload or {"label": "grasp"},
        ),
        raw_output_artifact_id="raw-artifact-1",
        schema_valid=True,
        reported_confidence=0.8,
        usage=VisionUsage(
            input_frames=6,
            input_images=1,
            input_tokens=100,
            output_tokens=20,
            cost=0.0,
            currency="USD",
        ),
        latency_ms=3,
    )


def _failure(
    request: VisionInferenceRequest, status: InferenceStatus = InferenceStatus.TIMEOUT
) -> VisionInferenceFailure:
    return VisionInferenceFailure(
        status=status,
        provider_request_id=None,
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        schema_valid=False,
        usage=VisionUsage(input_frames=0, input_images=0),
        latency_ms=2,
        failure=InferenceFailure(
            code="PROVIDER_TIMEOUT",
            detail="offline fixture timeout",
            retryability=Retryability.RETRYABLE,
        ),
    )


def _orchestrator(
    factory: Callable[[VisionInferenceRequest], VisionInferenceSuccess | VisionInferenceFailure],
    *,
    capabilities: ModelCapabilities | None = None,
    raised: BaseException | None = None,
    ledger: InMemoryInferenceLedger | None = None,
    policy: InferencePolicy | None = None,
) -> tuple[InferenceOrchestrator, FakeAdapter, InMemoryInferenceLedger]:
    store = ledger or InMemoryInferenceLedger()
    adapter = FakeAdapter(
        store,
        factory,
        capabilities=capabilities,
        raised=raised,
    )
    selected_policy = policy or _policy()
    orchestrator = InferenceOrchestrator(
        adapters={"fake": adapter},
        task_policies={VisionTask.ACTION_EVIDENCE: selected_policy},
        schema_documents={selected_policy.output_schema.artifact_id: SCHEMA},
        ledger=store,
        clock=lambda: NOW,
    )
    return orchestrator, adapter, store


def _orchestrator_with_schema_artifact(
    raw_schema: bytes,
    reference: JsonSchemaRef,
    *,
    additional_schema_artifacts: Mapping[str, bytes] | None = None,
) -> tuple[InferenceOrchestrator, FakeAdapter, InMemoryInferenceLedger]:
    store = InMemoryInferenceLedger()
    adapter = FakeAdapter(store, _success)
    orchestrator = InferenceOrchestrator(
        adapters={"fake": adapter},
        task_policies={
            VisionTask.ACTION_EVIDENCE: _policy().model_copy(update={"output_schema": reference})
        },
        schema_artifacts={
            reference.artifact_id: raw_schema,
            **dict(additional_schema_artifacts or {}),
        },
        ledger=store,
        clock=lambda: NOW,
    )
    return orchestrator, adapter, store


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_exact_schema_artifact_is_hashed_before_strict_parse_and_compilation() -> None:
    raw_schema = b"  " + canonical_json_bytes(SCHEMA) + b"\n"
    reference = SCHEMA_REF.model_copy(update={"sha256": exact_bytes_sha256(raw_schema)})
    orchestrator, adapter, store = _orchestrator_with_schema_artifact(raw_schema, reference)

    result = _run(orchestrator.orchestrate(**_request_kwargs()))

    assert result.status is InferenceStatus.SUCCEEDED
    assert adapter.infer_calls == 1
    assert len(store.list_intents()) == 1


def test_exact_schema_artifact_digest_mismatch_is_rejected_before_dispatch() -> None:
    raw_schema = b"\xff"
    orchestrator, adapter, store = _orchestrator_with_schema_artifact(raw_schema, SCHEMA_REF)

    with pytest.raises(OrchestrationConfigurationError, match="digest mismatch"):
        _run(orchestrator.orchestrate(**_request_kwargs()))

    assert adapter.infer_calls == 0
    assert store.list_intents() == ()


def test_exact_schema_artifacts_resolve_relative_refs_from_local_registry() -> None:
    common_id = "https://schemas.example.test/v1/common.schema.json"
    root_schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.example.test/v1/action.schema.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["label"],
        "properties": {"label": {"$ref": "common.schema.json#/$defs/label"}},
    }
    common_schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": common_id,
        "$defs": {"label": {"type": "string", "minLength": 1}},
    }
    root_bytes = canonical_json_bytes(root_schema)
    reference = SCHEMA_REF.model_copy(update={"sha256": exact_bytes_sha256(root_bytes)})
    orchestrator, adapter, store = _orchestrator_with_schema_artifact(
        root_bytes,
        reference,
        additional_schema_artifacts={"schema-common": canonical_json_bytes(common_schema)},
    )

    result = _run(orchestrator.orchestrate(**_request_kwargs()))

    assert result.status is InferenceStatus.SUCCEEDED
    assert adapter.infer_calls == 1
    assert len(store.list_intents()) == 1


def test_missing_exact_schema_dependency_fails_locally_before_dispatch() -> None:
    root_schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.example.test/v1/action.schema.json",
        "type": "object",
        "properties": {"label": {"$ref": "common.schema.json#/$defs/label"}},
    }
    root_bytes = canonical_json_bytes(root_schema)
    reference = SCHEMA_REF.model_copy(update={"sha256": exact_bytes_sha256(root_bytes)})
    orchestrator, adapter, store = _orchestrator_with_schema_artifact(root_bytes, reference)

    with pytest.raises(OrchestrationConfigurationError, match="not locally resolvable"):
        _run(orchestrator.orchestrate(**_request_kwargs()))

    assert adapter.capability_calls == 0
    assert adapter.infer_calls == 0
    assert store.list_intents() == ()


@pytest.mark.parametrize(
    ("raw_schema", "message"),
    [
        (b'{"type":"object","type":"array"}', "duplicate JSON object key"),
        (b"\xef\xbb\xbf" + canonical_json_bytes(SCHEMA), "UTF-8 BOM"),
        (b"[]", "root must be a JSON object"),
        (b'{"type":"object","title":"\xff"}', "not strict UTF-8"),
    ],
)
def test_malformed_exact_schema_artifact_is_rejected_without_dispatch(
    raw_schema: bytes, message: str
) -> None:
    reference = SCHEMA_REF.model_copy(update={"sha256": exact_bytes_sha256(raw_schema)})
    orchestrator, adapter, store = _orchestrator_with_schema_artifact(raw_schema, reference)

    with pytest.raises(OrchestrationConfigurationError, match=message):
        _run(orchestrator.orchestrate(**_request_kwargs()))

    assert adapter.infer_calls == 0
    assert store.list_intents() == ()


def test_intent_is_persisted_before_dispatch_and_success_is_idempotent() -> None:
    orchestrator, adapter, store = _orchestrator(_success)

    first = _run(orchestrator.orchestrate(**_request_kwargs()))
    assert first.status is InferenceStatus.SUCCEEDED
    assert first.output_valid is True
    assert first.normalized_output == {"label": "grasp"}
    assert adapter.intent_seen_before_infer is True
    assert adapter.infer_calls == 1
    assert len(store.list_intents()) == 1
    assert len(store.list_terminals()) == 1
    assert len(store.list_selections()) == 1
    selection = store.list_selections()[0]
    assert selection.inference_id == first.inference_id
    assert selection.selection_reason == "FIRST_SCHEMA_VALID_SUCCESS"
    assert selection.selection_decision_logical_key == (
        inference_attempt_selection_logical_key(
            logical_invocation_id=first.logical_invocation_id,
            policy_version="selection-1",
        )
    )

    second = _run(orchestrator.orchestrate(**_request_kwargs()))
    assert second == first
    assert adapter.infer_calls == 1
    assert len(store.list_terminals()) == 1
    assert len(store.list_selections()) == 1


def test_logical_identity_uses_capability_digest_not_snapshot_row_id() -> None:
    capabilities = _capabilities()
    first_orchestrator, _, _ = _orchestrator(
        _success,
        capabilities=capabilities.model_copy(update={"snapshot_id": _uuid(500)}),
    )
    second_orchestrator, _, _ = _orchestrator(
        _success,
        capabilities=capabilities.model_copy(update={"snapshot_id": _uuid(501)}),
    )

    first = _run(first_orchestrator.orchestrate(**_request_kwargs()))
    second = _run(second_orchestrator.orchestrate(**_request_kwargs()))

    assert first.logical_invocation_id == second.logical_invocation_id
    assert first.inference_id == second.inference_id


def test_logical_identity_excludes_package_prompt_and_schema_artifact_locators() -> None:
    relocated_schema = SCHEMA_REF.model_copy(update={"artifact_id": "schema-relocated"})
    relocated_policy = _policy().model_copy(
        update={
            "prompt_artifact_id": "prompt-relocated",
            "output_schema": relocated_schema,
        }
    )
    first_orchestrator, _, first_store = _orchestrator(_success)
    second_orchestrator, _, second_store = _orchestrator(
        _success,
        policy=relocated_policy,
    )
    second_kwargs = _request_kwargs()
    second_package = _package_input().model_copy(
        update={
            "package_id": _uuid(999),
            "package_manifest_sha256": _digest(998),
        }
    )
    second_kwargs["package_inputs"] = (second_package,)

    first = _run(first_orchestrator.orchestrate(**_request_kwargs()))
    second = _run(second_orchestrator.orchestrate(**second_kwargs))
    first_request = first_store.list_intents()[0].request
    second_request = second_store.list_intents()[0].request

    assert first.package_ids != second.package_ids
    assert first.prompt_artifact_id != second.prompt_artifact_id
    assert first_request.output_schema.artifact_id != second_request.output_schema.artifact_id
    assert first_request.package_inputs[0].package_manifest_sha256 != (
        second_request.package_inputs[0].package_manifest_sha256
    )
    assert first_request.package_input_set_sha256 == second_request.package_input_set_sha256
    assert first.logical_invocation_id == second.logical_invocation_id
    assert first.inference_id == second.inference_id
    assert first_store.list_selections()[0].selection_decision_logical_key == (
        second_store.list_selections()[0].selection_decision_logical_key
    )


def test_logical_identity_separates_changed_package_content() -> None:
    first_orchestrator, _, first_store = _orchestrator(_success)
    second_orchestrator, _, second_store = _orchestrator(_success)
    changed_kwargs = _request_kwargs()
    changed_package = _package_input().model_copy(
        update={"package_semantic_content_sha256": _digest(999)}
    )
    changed_kwargs["package_inputs"] = (changed_package,)

    first = _run(first_orchestrator.orchestrate(**_request_kwargs()))
    changed = _run(second_orchestrator.orchestrate(**changed_kwargs))

    assert first_store.list_intents()[0].request.package_input_set_sha256 != (
        second_store.list_intents()[0].request.package_input_set_sha256
    )
    assert first.logical_invocation_id != changed.logical_invocation_id
    assert first.inference_id != changed.inference_id


def test_logical_identity_separates_changed_package_role() -> None:
    first_orchestrator, _, first_store = _orchestrator(_success)
    changed_orchestrator, _, changed_store = _orchestrator(_success)
    changed_kwargs = _request_kwargs()
    changed_kwargs["package_inputs"] = (_package_input().model_copy(update={"role": "context"}),)

    first = _run(first_orchestrator.orchestrate(**_request_kwargs()))
    changed = _run(changed_orchestrator.orchestrate(**changed_kwargs))

    assert first_store.list_intents()[0].request.package_input_set_sha256 != (
        changed_store.list_intents()[0].request.package_input_set_sha256
    )
    assert first.logical_invocation_id != changed.logical_invocation_id


class _TamperingReadLedger(InMemoryInferenceLedger):
    def __init__(self) -> None:
        super().__init__()
        self.tamper_terminal = False

    def get_terminal(self, inference_id: str) -> ModelInference | None:
        terminal = super().get_terminal(inference_id)
        if terminal is None or not self.tamper_terminal:
            return terminal
        return terminal.model_copy(update={"input_manifest_set_sha256": _digest(997)})


def test_selected_terminal_reuse_rejects_tampered_ledger_semantics() -> None:
    ledger = _TamperingReadLedger()
    orchestrator, adapter, _ = _orchestrator(_success, ledger=ledger)
    first = _run(orchestrator.orchestrate(**_request_kwargs()))
    ledger.tamper_terminal = True
    relocated_kwargs = _request_kwargs()
    relocated_kwargs["package_inputs"] = (
        _package_input().model_copy(
            update={
                "package_id": _uuid(996),
                "package_manifest_sha256": _digest(995),
            }
        ),
    )

    with pytest.raises(InferenceLedgerError, match="semantically inconsistent"):
        _run(orchestrator.orchestrate(**relocated_kwargs))

    assert first.status is InferenceStatus.SUCCEEDED
    assert adapter.infer_calls == 1


def test_package_count_does_not_stand_in_for_image_count() -> None:
    orchestrator, adapter, store = _orchestrator(_success)
    first = _package_input()
    second = first.model_copy(update={"package_id": _uuid(9), "ordinal": 1})
    kwargs = _request_kwargs()
    kwargs["package_inputs"] = (first, second)

    result = _run(orchestrator.orchestrate(**kwargs))

    assert result.status is InferenceStatus.SUCCEEDED
    assert len(result.package_ids) == 2
    assert adapter.infer_calls == 1
    assert len(store.list_terminals()) == 1


def test_schema_invalid_success_is_terminal_but_never_selected() -> None:
    orchestrator, adapter, store = _orchestrator(
        lambda request: _success(request, payload={"label": 42})
    )

    result = _run(orchestrator.orchestrate(**_request_kwargs()))
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.output_valid is False
    assert result.normalized_output == {"label": 42}
    assert result.failure is not None
    assert result.failure.code == "OUTPUT_SCHEMA_INVALID"
    assert adapter.infer_calls == 1
    assert store.list_selections() == ()


def test_provider_failure_is_persisted_without_selection() -> None:
    orchestrator, adapter, store = _orchestrator(_failure)

    result = _run(orchestrator.orchestrate(**_request_kwargs()))
    assert result.status is InferenceStatus.TIMEOUT
    assert result.failure is not None
    assert result.failure.code == "PROVIDER_TIMEOUT"
    assert result.output_valid is False
    assert store.list_selections() == ()
    assert adapter.infer_calls == 1


def test_adapter_exception_becomes_a_terminal_failure() -> None:
    orchestrator, adapter, store = _orchestrator(
        _success,
        raised=RuntimeError("provider is intentionally offline"),
    )

    result = _run(orchestrator.orchestrate(**_request_kwargs()))
    assert result.status is InferenceStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "ADAPTER_EXCEPTION"
    assert "intentionally offline" in result.failure.detail
    assert len(store.list_terminals()) == 1
    assert adapter.infer_calls == 1


def test_capability_mismatch_fails_closed_before_intent() -> None:
    orchestrator, adapter, store = _orchestrator(
        _success,
        capabilities=_capabilities(supports_json_schema=False),
    )

    with pytest.raises(CapabilityValidationError, match="JSON Schema"):
        _run(orchestrator.orchestrate(**_request_kwargs()))
    assert adapter.infer_calls == 0
    assert store.list_intents() == ()
    assert store.list_terminals() == ()


@pytest.mark.parametrize("missing", ["policy", "provider", "schema"])
def test_missing_configuration_is_rejected_without_dispatch(missing: str) -> None:
    store = InMemoryInferenceLedger()
    adapter = FakeAdapter(store, _success)
    policies = {VisionTask.ACTION_EVIDENCE: _policy()}
    adapters = {"fake": adapter}
    schemas = {SCHEMA_REF.artifact_id: SCHEMA}
    if missing == "policy":
        policies = {}
    elif missing == "provider":
        adapters = {}
    else:
        schemas = {}
    orchestrator = InferenceOrchestrator(
        adapters=adapters,
        task_policies=policies,
        schema_documents=schemas,
        ledger=store,
        clock=lambda: NOW,
    )

    with pytest.raises(OrchestrationConfigurationError):
        _run(orchestrator.orchestrate(**_request_kwargs()))
    assert adapter.infer_calls == 0
    assert store.list_intents() == ()


def test_retry_uses_new_attempt_id_but_one_idempotent_selection() -> None:
    calls = 0

    def first_failure_then_success(
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        nonlocal calls
        calls += 1
        return _failure(request) if calls == 1 else _success(request)

    orchestrator, adapter, store = _orchestrator(first_failure_then_success)
    first_kwargs = _request_kwargs()
    first = _run(orchestrator.orchestrate(**first_kwargs, attempt=1, retry_count=0))
    second = _run(orchestrator.orchestrate(**first_kwargs, attempt=2, retry_count=1))

    assert first.status is InferenceStatus.TIMEOUT
    assert second.status is InferenceStatus.SUCCEEDED
    assert first.logical_invocation_id == second.logical_invocation_id
    assert first.inference_id != second.inference_id
    assert len(store.list_terminals()) == 2
    assert len(store.list_selections()) == 1
    assert store.list_selections()[0].inference_id == second.inference_id

    replay = _run(orchestrator.orchestrate(**first_kwargs, attempt=2, retry_count=1))
    assert replay == second
    assert adapter.infer_calls == 2


def test_shadow_success_is_recorded_but_cannot_be_selected() -> None:
    orchestrator, adapter, store = _orchestrator(_success)
    kwargs = {
        **_request_kwargs(),
        "shadow": True,
        "experiment_id": "experiment-1",
        "shadow_route_id": _uuid(300),
        "primary_inference_id": _uuid(301),
    }

    result = _run(orchestrator.orchestrate(**kwargs))
    assert result.status is InferenceStatus.SUCCEEDED
    assert result.shadow is True
    assert store.list_selections() == ()
    assert adapter.infer_calls == 1


class RejectingGate:
    async def acquire(self, *, capabilities: ModelCapabilities, timeout_ms: int) -> None:
        del capabilities, timeout_ms
        raise RuntimeError("local quota exhausted")


def test_execution_gate_failure_is_persisted_after_intent() -> None:
    store = InMemoryInferenceLedger()
    adapter = FakeAdapter(store, _success)
    orchestrator = InferenceOrchestrator(
        adapters={"fake": adapter},
        task_policies={VisionTask.ACTION_EVIDENCE: _policy()},
        schema_documents={SCHEMA_REF.artifact_id: SCHEMA},
        ledger=store,
        execution_gate=RejectingGate(),
        clock=lambda: NOW,
    )

    result = _run(orchestrator.orchestrate(**_request_kwargs()))
    assert result.status is InferenceStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "EXECUTION_GATE_REJECTED"
    assert len(store.list_intents()) == 1
    assert adapter.infer_calls == 0
