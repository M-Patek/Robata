from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from robata.adapters.sqlite_inference_evidence import (
    INFERENCE_INTENT_SCHEMA_ID,
    RAW_PROVIDER_RESPONSE_SCHEMA_ID,
    SQLiteInferenceEvidenceLedger,
    SQLiteInferenceEvidenceLedgerError,
)
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import PackageInput, VisionInferenceRequest
from robata.inference.enrichment import (
    EnrichmentAuthorityContext,
    OrchestratorEnrichedOutput,
    ParsedProviderClaimArtifact,
    ProviderClaimEnricher,
    RawProviderResponseArtifact,
    SelectedAttemptOutput,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    InferenceFailure,
    InferenceStatus,
    ModelInference,
    ModelInferenceUsage,
    Retryability,
    inference_attempt_selection_digest,
    inference_attempt_selection_logical_key,
)
from robata.inference.orchestrator import InferenceIntent
from robata.runtime.observability import RuntimeProfileRecorder
from tests.unit.test_inference_enrichment import _Fixture as _EnrichmentFixture
from tests.unit.test_inference_enrichment import _fixture as _enrichment_fixture

NOW = "2026-07-20T12:00:00Z"
EARLIER = "2026-07-20T11:59:59Z"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


def _database(tmp_path: Path) -> Path:
    return tmp_path / "inference-evidence.sqlite3"


def test_initialization_observes_exact_transaction_boundaries(tmp_path: Path) -> None:
    recorder = RuntimeProfileRecorder()

    SQLiteInferenceEvidenceLedger(
        _database(tmp_path),
        SchemaRegistry(),
        runtime_observer=recorder,
    )

    snapshot = recorder.snapshot()
    transaction_counters = tuple(
        counter
        for counter in snapshot.counters
        if counter.name == "sqlite.inference_evidence.transactions"
    )
    commit_count = sum(
        counter.value
        for counter in snapshot.counters
        if counter.name == "sqlite.inference_evidence.commits"
    )
    rollback_count = sum(
        counter.value
        for counter in snapshot.counters
        if counter.name == "sqlite.inference_evidence.rollbacks"
    )
    operations = {
        attribute.value
        for counter in transaction_counters
        for attribute in counter.attributes
        if attribute.name == "operation"
    }

    assert sum(counter.value for counter in transaction_counters) == 2
    assert commit_count == 2
    assert rollback_count == 0
    assert operations == {"initialize_preflight", "initialize_schema"}
    assert sum(span.name == "sqlite.inference_evidence.transaction" for span in snapshot.spans) == 2


def test_ledger_reuses_one_owned_connection_until_closed(tmp_path: Path) -> None:
    recorder = RuntimeProfileRecorder()
    ledger = SQLiteInferenceEvidenceLedger(
        _database(tmp_path),
        SchemaRegistry(),
        runtime_observer=recorder,
    )
    connection = ledger._connection
    assert connection is not None
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    _fixture, intent, _raw_data = _intent()

    assert ledger.append_intent(intent) == intent
    assert ledger.get_intent(intent.inference_id) == intent
    assert ledger._connection is connection
    snapshot = recorder.snapshot()
    assert sum(
        counter.value
        for counter in snapshot.counters
        if counter.name == "sqlite.inference_evidence.connections"
    ) == 1

    ledger.close()
    ledger.close()
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="ledger is closed"):
        ledger.get_intent(intent.inference_id)


def test_owned_connection_serializes_cross_thread_calls(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    _first_fixture, first, _first_raw_data = _intent(identity_offset=1)
    _second_fixture, second, _second_raw_data = _intent(identity_offset=2)
    barrier = Barrier(2)

    def append(intent: InferenceIntent) -> InferenceIntent:
        barrier.wait(timeout=5)
        return ledger.append_intent(intent)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert tuple(pool.map(append, (first, second))) == (first, second)
        assert ledger.get_intent(first.inference_id) == first
        assert ledger.get_intent(second.inference_id) == second
    finally:
        ledger.close()


def test_write_cache_avoids_dependency_readback_and_hot_getters_invalidate(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    _fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    connection = ledger._connection
    assert connection is not None
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    external: SQLiteInferenceEvidenceLedger | None = None
    try:
        ledger.append(
            request_id=intent.request_id,
            provider_request_id="cache-readback-provider",
            data=raw_data,
        )
        assert not any(
            "SELECT * FROM INFERENCE_INTENTS WHERE INFERENCE_ID =" in statement.upper()
            for statement in statements
        )

        statements.clear()
        assert ledger.get_intent(intent.inference_id) == intent
        assert not any(
            "SELECT * FROM INFERENCE_INTENTS WHERE INFERENCE_ID =" in statement.upper()
            for statement in statements
        )

        _other_fixture, other, _other_raw_data = _intent(identity_offset=1)
        external = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
        assert external.append_intent(other) == other
        statements.clear()
        assert ledger.get_intent(intent.inference_id) == intent
        assert any(
            "SELECT * FROM INFERENCE_INTENTS WHERE INFERENCE_ID =" in statement.upper()
            for statement in statements
        )
    finally:
        connection.set_trace_callback(None)
        if external is not None:
            external.close()
        ledger.close()


def test_cached_getters_return_defensive_copies(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    _fixture, intent, raw_data = _intent()
    try:
        assert ledger.append_intent(intent) == intent
        returned_intent = ledger.get_intent(intent.inference_id)
        assert returned_intent is not None
        assert returned_intent is not ledger.get_intent(intent.inference_id)
        # Strict Pydantic models are frozen at the top level, but their nested
        # JSON-shaped mapping remains mutable. Mutating a caller-owned result
        # must not alter the cache used by future reads or idempotent appends.
        returned_intent.input_config["input_images"] = 999
        assert ledger.get_intent(intent.inference_id) == intent
        assert ledger.append_intent(intent) == intent

        stored = ledger.append(
            request_id=intent.request_id,
            provider_request_id="defensive-copy-provider",
            data=raw_data,
        )
        assert ledger.get(stored.artifact_id) is not ledger.get(stored.artifact_id)
    finally:
        ledger.close()


@dataclass(frozen=True)
class _Evidence:
    intent: InferenceIntent
    raw_data: bytes
    terminal: ModelInference
    selection: InferenceAttemptSelection
    parsed: ParsedProviderClaimArtifact
    selected: SelectedAttemptOutput
    enriched: OrchestratorEnrichedOutput


def _intent(
    *,
    shadow: bool = False,
    identity_offset: int = 0,
    logical_invocation_id: str | None = None,
    attempt: int = 1,
) -> tuple[_EnrichmentFixture, InferenceIntent, bytes]:
    fixture = _enrichment_fixture()
    plan = fixture.plan
    part = plan.call_plan.parts[0]
    offset = identity_offset + (100 if shadow else 0)
    invocation_id = logical_invocation_id or _uuid(10_001 + offset)
    request_id = _uuid(10_002 + offset)
    inference_id = _uuid(10_003 + offset)
    package_inputs = tuple(
        PackageInput(
            package_id=item.package_id,
            package_semantic_content_sha256=item.semantic_content_sha256,
            package_manifest_sha256=item.manifest_bytes_sha256,
            role="primary",
            ordinal=item.ordinal,
        )
        for item in plan.subject.packages
    )
    request = VisionInferenceRequest(
        schema_version="1.0",
        logical_invocation_id=invocation_id,
        request_id=request_id,
        idempotency_key=f"inference-fixture:{inference_id}",
        provider=plan.target.provider,
        model_name=plan.target.model_name,
        model_version=plan.target.model_version,
        package_set_id=_uuid(10_004),
        package_inputs=package_inputs,
        package_input_set_sha256=semantic_sha256(
            [
                {
                    "ordinal": item.ordinal,
                    "role": item.role,
                    "package_semantic_content_sha256": (item.package_semantic_content_sha256),
                }
                for item in package_inputs
            ]
        ),
        task=plan.subject.task,
        prompt_version=plan.prompt_output.prompt_version,
        prompt_artifact_id=_uuid(10_005),
        prompt_sha256=plan.prompt_output.prompt_sha256,
        rendered_input_digest=part.item_manifest_sha256,
        input_plan_id=plan.input_plan_id,
        input_plan_semantic_sha256=plan.semantic_sha256,
        input_plan_part_ordinal=part.ordinal,
        input_plan_part_count=part.part_count,
        input_plan_part_semantic_sha256=part.part_semantic_sha256,
        input_plan=plan,
        output_schema=fixture.provider_schema,
        capability_snapshot_id=plan.target.capability_snapshot_id,
        capability_snapshot_digest=plan.target.capability_snapshot_sha256,
        model_policy_version="model-policy-1",
        generation_config={"temperature": 0},
        provider_idempotency_key=part.idempotency_key,
        timeout_ms=1_000,
        metadata={},
    )
    intent = InferenceIntent(
        schema_version="1.0",
        inference_id=inference_id,
        logical_invocation_id=invocation_id,
        request_id=request_id,
        idempotency_key=request.idempotency_key,
        task=request.task,
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        adapter_version=plan.target.adapter_version,
        mcap_id=_uuid(10_006),
        camera_mapping_run_id=_uuid(10_007),
        alignment_id=_uuid(10_008),
        start_ns=0,
        end_ns=100,
        input_config={"input_images": 6},
        sampling_config={"policy": "dense-v1"},
        input_plan_id=plan.input_plan_id,
        input_plan_semantic_sha256=plan.semantic_sha256,
        input_plan_part_ordinal=part.ordinal,
        input_plan_part_count=part.part_count,
        input_plan_part_semantic_sha256=part.part_semantic_sha256,
        experiment_id="shadow-fixture" if shadow else None,
        shadow_route_id="route-fixture" if shadow else None,
        primary_inference_id=_uuid(9_999) if shadow else None,
        attempt=attempt,
        retry_count=attempt - 1,
        shadow=shadow,
        request=request,
        queued_at=NOW,
        created_at=NOW,
    )
    raw_data = canonical_json_bytes(fixture.payload.model_dump(mode="json"))
    return fixture, intent, raw_data


def _build_after_raw(
    fixture: _EnrichmentFixture,
    intent: InferenceIntent,
    raw_data: bytes,
    raw_artifact_id: str,
    provider_request_id: str,
    *,
    identity_offset: int = 0,
) -> _Evidence:
    request = intent.request
    terminal = ModelInference(
        schema_version="1.0",
        inference_id=intent.inference_id,
        logical_invocation_id=intent.logical_invocation_id,
        request_id=intent.request_id,
        idempotency_key=intent.idempotency_key,
        mcap_id=intent.mcap_id,
        package_set_id=request.package_set_id,
        package_id=request.package_inputs[0].package_id,
        package_ids=tuple(item.package_id for item in request.package_inputs),
        camera_mapping_run_id=intent.camera_mapping_run_id,
        alignment_id=intent.alignment_id,
        start_ns=intent.start_ns,
        end_ns=intent.end_ns,
        stage=intent.task,
        provider=intent.provider,
        model_name=intent.model_name,
        model_version=intent.model_version,
        adapter_version=intent.adapter_version,
        prompt_version=request.prompt_version,
        prompt_artifact_id=request.prompt_artifact_id,
        prompt_sha256=request.prompt_sha256,
        rendered_input_digest=request.rendered_input_digest,
        input_plan_id=intent.input_plan_id,
        input_plan_semantic_sha256=intent.input_plan_semantic_sha256,
        input_plan_part_ordinal=intent.input_plan_part_ordinal,
        input_plan_part_count=intent.input_plan_part_count,
        input_plan_part_semantic_sha256=intent.input_plan_part_semantic_sha256,
        output_schema_id=request.output_schema.schema_id,
        output_schema_version=request.output_schema.version,
        output_schema_artifact_id=request.output_schema.artifact_id,
        output_schema_sha256=request.output_schema.sha256,
        capability_snapshot_id=request.capability_snapshot_id,
        capability_snapshot_digest=request.capability_snapshot_digest,
        input_manifest_set_sha256=request.package_input_set_sha256,
        input_config=intent.input_config,
        sampling_config=intent.sampling_config,
        generation_config=request.generation_config,
        provider_idempotency_key=request.provider_idempotency_key,
        provider_request_id=provider_request_id,
        experiment_id=intent.experiment_id,
        shadow_route_id=intent.shadow_route_id,
        primary_inference_id=intent.primary_inference_id,
        shadow=intent.shadow,
        attempt=intent.attempt,
        retry_count=intent.retry_count,
        status=InferenceStatus.SUCCEEDED,
        queued_at=intent.queued_at,
        started_at=NOW,
        completed_at=NOW,
        latency_ms=0,
        raw_output={"artifact_id": raw_artifact_id},
        normalized_output=fixture.payload.model_dump(mode="json"),
        output_valid=True,
        reported_confidence=None,
        calibrated_confidence=None,
        usage=ModelInferenceUsage(input_frames=6, input_images=6),
        failure=None,
        created_at=intent.created_at,
    )
    selection = InferenceAttemptSelection(
        schema_version="1.0",
        selection_id=_stable_uuid(
            "inference-selection",
            inference_attempt_selection_digest(
                logical_invocation_id=intent.logical_invocation_id,
                policy_version="selection-policy-1",
            ),
        ),
        inference_id=intent.inference_id,
        logical_invocation_id=intent.logical_invocation_id,
        policy_version="selection-policy-1",
        selection_reason="FIRST_SCHEMA_VALID_SUCCESS",
        selection_decision_logical_key=inference_attempt_selection_logical_key(
            logical_invocation_id=intent.logical_invocation_id,
            policy_version="selection-policy-1",
        ),
        selected_at=NOW,
    )
    raw = RawProviderResponseArtifact.from_bytes(
        data=raw_data,
        artifact_id=raw_artifact_id,
        media_type="application/json",
        provider_request_id=provider_request_id,
        inference_id=intent.inference_id,
        provider=intent.provider,
        model_name=intent.model_name,
        model_version=intent.model_version,
        created_at=NOW,
    )
    parsed = ParsedProviderClaimArtifact.create(
        artifact_id=_uuid(10_010 + identity_offset + (100 if intent.shadow else 0)),
        raw_response=raw,
        provider_claim_schema=fixture.provider_schema,
        task=intent.task,
        payload=fixture.payload,
        parser_version="strict-parser-1",
        created_at=NOW,
    )
    selected = SelectedAttemptOutput.create(parsed, selection)
    authority = EnrichmentAuthorityContext(
        recording_identity=_digest(10_011),
        mcap_id=intent.mcap_id,
        camera_mapping_run_id=intent.camera_mapping_run_id,
        alignment_id=intent.alignment_id,
        inference_id=intent.inference_id,
        logical_invocation_id=intent.logical_invocation_id,
        prompt_version=request.prompt_version,
        prompt_artifact_id=request.prompt_artifact_id,
        prompt_sha256=request.prompt_sha256,
        work_node_type="INFERENCE_ENRICHMENT",
        work_node_logical_key=f"inference-work:{_digest(10_012)}",
    )
    enriched = ProviderClaimEnricher(fixture.registry).enrich(
        input_plan=fixture.plan,
        reference_catalog=fixture.reference_catalog,
        parsed_claims=parsed,
        selected_attempt=selected,
        authority=authority,
        enriched_output_schema=fixture.enriched_schema,
        enrichment_policy_version="enrichment-policy-1",
        artifact_id=_uuid(10_013 + identity_offset + (100 if intent.shadow else 0)),
        created_at=NOW,
        input_plan_part_ordinal=0,
    )
    return _Evidence(
        intent=intent,
        raw_data=raw_data,
        terminal=terminal,
        selection=selection,
        parsed=parsed,
        selected=selected,
        enriched=enriched,
    )


def _persist_chain(ledger: SQLiteInferenceEvidenceLedger) -> _Evidence:
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id=f"offline:{intent.request_id}",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    ledger.append_terminal(evidence.terminal)
    ledger.append_selection(evidence.selection)
    ledger.append_parsed_claim(evidence.parsed)
    ledger.append_selected_output(evidence.selected)
    ledger.append_enriched_output(evidence.enriched)
    return evidence


def test_sixteen_batched_calls_stay_within_p3_ledger_budget(tmp_path: Path) -> None:
    """P3 keeps 16 complete evidence calls within the ledger I/O envelope."""

    recorder = RuntimeProfileRecorder()
    ledger = SQLiteInferenceEvidenceLedger(
        _database(tmp_path),
        SchemaRegistry(),
        runtime_observer=recorder,
    )
    try:
        for identity_offset in range(16):
            fixture, intent, raw_data = _intent(identity_offset=identity_offset)
            assert ledger.append_intent(intent) == intent
            stored = ledger.append(
                request_id=intent.request_id,
                provider_request_id=f"offline:{intent.request_id}",
                data=raw_data,
            )
            evidence = _build_after_raw(
                fixture,
                intent,
                raw_data,
                stored.artifact_id,
                stored.provider_request_id,
                identity_offset=identity_offset,
            )
            assert ledger.append_terminal_and_selection(
                evidence.terminal,
                evidence.selection,
            ) == (evidence.terminal, evidence.selection)
            assert ledger.append_accepted_lineage(
                evidence.parsed,
                evidence.selected,
                evidence.enriched,
            ) == (evidence.parsed, evidence.selected, evidence.enriched)

        snapshot = recorder.snapshot()
        connection_count = sum(
            counter.value
            for counter in snapshot.counters
            if counter.name == "sqlite.inference_evidence.connections"
        )
        transaction_count = sum(
            counter.value
            for counter in snapshot.counters
            if counter.name == "sqlite.inference_evidence.transactions"
        )
        # Two initialization scopes plus one durable checkpoint for each phase:
        # intent, raw bytes, terminal/selection, and accepted lineage.
        assert connection_count <= 12
        assert transaction_count <= 80
    finally:
        ledger.close()


def test_restart_preserves_complete_selected_evidence_chain(tmp_path: Path) -> None:
    registry = SchemaRegistry()
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, registry)
    evidence = _persist_chain(ledger)

    restarted = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())

    assert restarted.get_intent(evidence.intent.inference_id) == evidence.intent
    assert restarted.get_terminal(evidence.terminal.inference_id) == evidence.terminal
    assert (
        restarted.get_selection(
            evidence.selection.logical_invocation_id,
            evidence.selection.policy_version,
        )
        == evidence.selection
    )
    assert restarted.get_parsed_claim(evidence.parsed.artifact_id) == evidence.parsed
    assert (
        restarted.get_raw_artifact(evidence.parsed.raw_response.artifact_id)
        == evidence.parsed.raw_response
    )
    assert restarted.get_selected_output(evidence.selected.selection_id) == evidence.selected
    assert restarted.get_enriched_output(evidence.enriched.artifact_id) == evidence.enriched
    assert restarted.get(evidence.parsed.raw_response.artifact_id).data == evidence.raw_data

    connection = sqlite3.connect(database)
    try:
        intent_pin = registry.resolve_version(INFERENCE_INTENT_SCHEMA_ID, "1.0.0").ref
        intent_row = connection.execute(
            """
            SELECT contract_schema_id, contract_version,
                   contract_artifact_id, contract_sha256
            FROM inference_intents
            """
        ).fetchone()
        assert intent_row == (
            intent_pin.schema_id,
            intent_pin.version,
            intent_pin.artifact_id,
            intent_pin.sha256,
        )
        raw_pin = registry.resolve_version(RAW_PROVIDER_RESPONSE_SCHEMA_ID, "1.0.0").ref
        raw_row = connection.execute(
            """
            SELECT contract_schema_id, contract_version,
                   contract_artifact_id, contract_sha256
            FROM raw_provider_artifacts
            """
        ).fetchone()
        assert raw_row == (
            raw_pin.schema_id,
            raw_pin.version,
            raw_pin.artifact_id,
            raw_pin.sha256,
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchone() is None
    finally:
        connection.close()


def test_complete_chain_replay_is_exactly_idempotent(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)

    assert ledger.append_intent(evidence.intent) == evidence.intent
    assert ledger.append(
        request_id=evidence.intent.request_id,
        provider_request_id=evidence.parsed.raw_response.provider_request_id,
        data=evidence.raw_data,
    ) == ledger.get(evidence.parsed.raw_response.artifact_id)
    assert ledger.append_terminal(evidence.terminal) == evidence.terminal
    assert ledger.append_raw_artifact(evidence.parsed.raw_response) == evidence.parsed.raw_response
    assert ledger.append_selection(evidence.selection) == evidence.selection
    assert ledger.append_parsed_claim(evidence.parsed) == evidence.parsed
    assert ledger.append_selected_output(evidence.selected) == evidence.selected
    assert ledger.append_enriched_output(evidence.enriched) == evidence.enriched
    assert len(ledger.list_records()) == 1


def test_raw_bytes_require_intent_and_conflicting_request_bytes_are_rejected(
    tmp_path: Path,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    _fixture, intent, raw_data = _intent()
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="persisted inference intent"):
        ledger.append(
            request_id=intent.request_id,
            provider_request_id="provider-request",
            data=raw_data,
        )

    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request",
        data=raw_data,
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="different raw response"):
        ledger.append(
            request_id=intent.request_id,
            provider_request_id="provider-request-2",
            data=b'{"claims":[],"abstained":false}',
        )
    assert ledger.list_records() == (stored,)


def test_downstream_rows_require_their_exact_predecessors(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="intent, terminal"):
        ledger.append_parsed_claim(evidence.parsed)
    assert ledger.get_raw_artifact(stored.artifact_id) is None
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="successful terminal"):
        ledger.append_selection(evidence.selection)

    ledger.append_terminal(evidence.terminal)
    assert ledger.get_raw_artifact(stored.artifact_id) == evidence.parsed.raw_response
    forged_selection = evidence.selection.model_copy(update={"selection_id": _uuid(91_001)})
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="semantically inconsistent"):
        ledger.append_selection(forged_selection)
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="selection and parsed"):
        ledger.append_selected_output(evidence.selected)
    ledger.append_selection(evidence.selection)
    conflicting_reason = evidence.selection.model_copy(
        update={"selection_reason": "MANUAL_OVERRIDE"}
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="different selected attempt"):
        ledger.append_selection(conflicting_reason)
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="selection and parsed"):
        ledger.append_selected_output(evidence.selected)
    ledger.append_parsed_claim(evidence.parsed)
    ledger.append_selected_output(evidence.selected)
    ledger.append_enriched_output(evidence.enriched)


def test_failure_terminal_may_commit_without_raw_provider_bytes(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    provisional = _build_after_raw(
        fixture,
        intent,
        raw_data,
        _uuid(10_020),
        "unused-provider-request",
    ).terminal
    failed = provisional.model_copy(
        update={
            "status": InferenceStatus.TIMEOUT,
            "provider_request_id": None,
            "raw_output": None,
            "normalized_output": None,
            "output_valid": False,
            "failure": InferenceFailure(
                code="PROVIDER_TIMEOUT",
                detail="fixture timeout",
                retryability=Retryability.RETRYABLE,
            ),
        }
    )

    assert ledger.append_terminal(failed) == failed
    assert ledger.list_records() == ()
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="already-persisted raw"):
        ledger.append(
            request_id=intent.request_id,
            provider_request_id="late-provider-request",
            data=raw_data,
        )
    assert ledger.list_records() == ()


def test_terminal_cannot_orphan_raw_bytes_committed_before_crash(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request-before-crash",
        data=raw_data,
    )
    provisional = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    ).terminal
    failed_retry = provisional.model_copy(
        update={
            "status": InferenceStatus.TIMEOUT,
            "provider_request_id": None,
            "raw_output": None,
            "normalized_output": None,
            "output_valid": False,
            "failure": InferenceFailure(
                code="PROVIDER_TIMEOUT",
                detail="retry after raw-byte commit",
                retryability=Retryability.RETRYABLE,
            ),
        }
    )

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="already-persisted raw"):
        ledger.append_terminal(failed_retry)
    assert ledger.get_terminal(intent.inference_id) is None
    assert ledger.get(stored.artifact_id) == stored
    assert ledger.get_raw_artifact(stored.artifact_id) is None


def test_invalid_output_terminal_commits_complete_raw_artifact_quartet(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="invalid-output-provider-request",
        data=raw_data,
    )
    provisional = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    invalid = provisional.terminal.model_copy(
        update={
            "status": InferenceStatus.INVALID_OUTPUT,
            "normalized_output": None,
            "output_valid": False,
            "failure": InferenceFailure(
                code="PROVIDER_SCHEMA_MISMATCH",
                detail="fixture parse failure",
                retryability=Retryability.PERMANENT,
            ),
        }
    )

    assert ledger.append_terminal(invalid) == invalid
    expected_raw = provisional.parsed.raw_response
    assert ledger.get_raw_artifact(stored.artifact_id) == expected_raw


def test_shadow_success_is_durable_but_cannot_be_selected(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent(shadow=True)
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="shadow-provider-request",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )

    assert ledger.append_terminal(evidence.terminal) == evidence.terminal
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="semantically inconsistent"):
        ledger.append_selection(evidence.selection)


def test_shadow_intent_requires_route_identity(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    _fixture, intent, _raw_data = _intent(shadow=True)
    forged = intent.model_copy(update={"shadow_route_id": None, "primary_inference_id": None})

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="shadow route"):
        ledger.append_intent(forged)


def test_intent_requires_exact_registered_output_schema_pin(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    _fixture, intent, _raw_data = _intent()
    forged_request = intent.request.model_copy(
        update={
            "output_schema": intent.request.output_schema.model_copy(
                update={"sha256": _digest(99_999)}
            ),
            "input_plan": None,
            "input_plan_id": None,
            "input_plan_semantic_sha256": None,
            "input_plan_part_ordinal": None,
            "input_plan_part_count": None,
            "input_plan_part_semantic_sha256": None,
        }
    )
    forged = intent.model_copy(
        update={
            "request": forged_request,
            "input_plan_id": None,
            "input_plan_semantic_sha256": None,
            "input_plan_part_ordinal": None,
            "input_plan_part_count": None,
            "input_plan_part_semantic_sha256": None,
        }
    )

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="exact registered"):
        ledger.append_intent(forged)
    assert ledger.get_intent(intent.inference_id) is None


def test_conflicting_intent_is_rejected_without_mutation(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    _fixture, intent, _raw_data = _intent()
    ledger.append_intent(intent)
    conflicting = intent.model_copy(update={"input_config": {"input_images": 7}})

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="conflicting intent"):
        ledger.append_intent(conflicting)
    assert ledger.get_intent(intent.inference_id) == intent


def test_combined_terminal_selection_keeps_first_winner_and_later_terminal(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    winner_ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    loser_ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    first_fixture, first_intent, first_raw_data = _intent()
    second_fixture, second_intent, second_raw_data = _intent(
        identity_offset=1_000,
        logical_invocation_id=first_intent.logical_invocation_id,
        attempt=2,
    )
    third_fixture, third_intent, third_raw_data = _intent(
        identity_offset=2_000,
        logical_invocation_id=first_intent.logical_invocation_id,
        attempt=3,
    )

    def prepare_attempt(
        ledger: SQLiteInferenceEvidenceLedger,
        fixture: _EnrichmentFixture,
        intent: InferenceIntent,
        raw_data: bytes,
        *,
        identity_offset: int,
    ) -> _Evidence:
        ledger.append_intent(intent)
        stored = ledger.append(
            request_id=intent.request_id,
            provider_request_id=f"winner-race:{intent.request_id}",
            data=raw_data,
        )
        return _build_after_raw(
            fixture,
            intent,
            raw_data,
            stored.artifact_id,
            stored.provider_request_id,
            identity_offset=identity_offset,
        )

    try:
        first = prepare_attempt(
            winner_ledger,
            first_fixture,
            first_intent,
            first_raw_data,
            identity_offset=0,
        )
        second = prepare_attempt(
            loser_ledger,
            second_fixture,
            second_intent,
            second_raw_data,
            identity_offset=1_000,
        )
        assert first.selection != second.selection

        assert winner_ledger.append_terminal_and_selection(
            first.terminal,
            first.selection,
        ) == (first.terminal, first.selection)
        # The loser ledger starts this operation with a stale cache. Its retry must
        # preserve the already-durable winner while committing the other terminal.
        assert loser_ledger.append_terminal_and_selection(
            second.terminal,
            second.selection,
        ) == (second.terminal, first.selection)

        assert winner_ledger.get_terminal(second.terminal.inference_id) == second.terminal
        assert (
            loser_ledger.get_selection(
                first.selection.logical_invocation_id,
                first.selection.policy_version,
            )
            == first.selection
        )
        with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="different selected attempt"):
            loser_ledger.append_selection(second.selection)

        third = prepare_attempt(
            loser_ledger,
            third_fixture,
            third_intent,
            third_raw_data,
            identity_offset=2_000,
        )
        # The candidate selection is valid for the second terminal but not the
        # third. The combined operation must roll back that mismatched terminal.
        with pytest.raises(
            SQLiteInferenceEvidenceLedgerError,
            match="candidate terminal attempt",
        ):
            loser_ledger.append_terminal_and_selection(third.terminal, second.selection)
        assert loser_ledger.get_terminal(third.terminal.inference_id) is None
    finally:
        loser_ledger.close()
        winner_ledger.close()


def test_concurrent_conflicting_raw_appends_have_one_durable_winner(tmp_path: Path) -> None:
    database = _database(tmp_path)
    registry = SchemaRegistry()
    first = SQLiteInferenceEvidenceLedger(database, registry)
    second = SQLiteInferenceEvidenceLedger(database, registry)
    _fixture, intent, raw_data = _intent()
    first.append_intent(intent)
    barrier = Barrier(2)

    def append(ledger: SQLiteInferenceEvidenceLedger, data: bytes) -> object:
        barrier.wait(timeout=5)
        try:
            return ledger.append(
                request_id=intent.request_id,
                provider_request_id="concurrent-provider-request",
                data=data,
            )
        except SQLiteInferenceEvidenceLedgerError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda item: append(*item),
                ((first, raw_data), (second, raw_data + b"\n")),
            )
        )

    assert sum(not isinstance(item, Exception) for item in results) == 1
    assert sum(isinstance(item, SQLiteInferenceEvidenceLedgerError) for item in results) == 1
    assert len(first.list_records()) == 1


def test_concurrent_first_constructors_share_one_canonical_database(tmp_path: Path) -> None:
    registries = (SchemaRegistry(), SchemaRegistry())
    last_ledgers: tuple[SQLiteInferenceEvidenceLedger, ...] = ()
    for round_index in range(50):
        database = tmp_path / f"concurrent-initialization-{round_index}.sqlite3"
        barrier = Barrier(2)

        def construct(
            index: int,
            round_barrier: Barrier = barrier,
            round_database: Path = database,
        ) -> object:
            round_barrier.wait(timeout=5)
            try:
                return SQLiteInferenceEvidenceLedger(round_database, registries[index])
            except SQLiteInferenceEvidenceLedgerError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = tuple(pool.map(construct, range(2)))

        failures = tuple(
            item for item in results if not isinstance(item, SQLiteInferenceEvidenceLedger)
        )
        assert not failures, (round_index, failures)
        last_ledgers = tuple(
            item for item in results if isinstance(item, SQLiteInferenceEvidenceLedger)
        )

    _fixture, intent, _raw_data = _intent()
    assert last_ledgers[0].append_intent(intent) == intent
    assert last_ledgers[1].get_intent(intent.inference_id) == intent


def test_equal_semantic_claims_are_retained_for_distinct_attempts(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, first_intent, raw_data = _intent()
    second_fixture, second_intent, second_raw_data = _intent(
        identity_offset=1_000,
        logical_invocation_id=first_intent.logical_invocation_id,
        attempt=2,
    )

    def persist_attempt(
        local_fixture: _EnrichmentFixture,
        intent: InferenceIntent,
        data: bytes,
        identity_offset: int,
    ) -> _Evidence:
        ledger.append_intent(intent)
        stored = ledger.append(
            request_id=intent.request_id,
            provider_request_id=f"provider-attempt-{intent.attempt}",
            data=data,
        )
        evidence = _build_after_raw(
            local_fixture,
            intent,
            data,
            stored.artifact_id,
            stored.provider_request_id,
            identity_offset=identity_offset,
        )
        ledger.append_terminal(evidence.terminal)
        ledger.append_parsed_claim(evidence.parsed)
        return evidence

    first = persist_attempt(fixture, first_intent, raw_data, 0)
    second = persist_attempt(second_fixture, second_intent, second_raw_data, 1_000)

    assert first.parsed.semantic_sha256 == second.parsed.semantic_sha256
    assert first.parsed.artifact_id != second.parsed.artifact_id
    assert first.parsed.raw_response.artifact_id != second.parsed.raw_response.artifact_id
    assert ledger.get_parsed_claim(first.parsed.artifact_id) == first.parsed
    assert ledger.get_parsed_claim(second.parsed.artifact_id) == second.parsed

    connection = sqlite3.connect(ledger.database_path)
    try:
        assert connection.execute("SELECT count(*) FROM raw_provider_responses").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM raw_provider_artifacts").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM model_inference_terminals").fetchone() == (
            2,
        )
        assert connection.execute("SELECT count(*) FROM parsed_provider_claims").fetchone() == (2,)
    finally:
        connection.close()


def test_parsed_payload_must_equal_terminal_normalized_output(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    ledger.append_terminal(evidence.terminal)
    changed_first_claim = fixture.payload.claims[0].model_copy(update={"model_reported_score": 0.7})
    mismatched_payload = fixture.payload.__class__(
        claims=(changed_first_claim, *fixture.payload.claims[1:]),
        abstained=False,
    )
    mismatched = ParsedProviderClaimArtifact.create(
        artifact_id=_uuid(90_001),
        raw_response=evidence.parsed.raw_response,
        provider_claim_schema=fixture.provider_schema,
        task=intent.task,
        payload=mismatched_payload,
        parser_version=evidence.parsed.parser_version,
        created_at=NOW,
    )

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="lineage is inconsistent"):
        ledger.append_parsed_claim(mismatched)
    assert ledger.get_parsed_claim(mismatched.artifact_id) is None


def test_evidence_timestamps_close_monotonically(tmp_path: Path) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    fixture, intent, raw_data = _intent()
    late_intent = intent.model_copy(update={"created_at": NOW, "queued_at": EARLIER})
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="intent timestamps"):
        ledger.append_intent(late_intent)

    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    ledger.append_terminal(evidence.terminal)

    early_raw = evidence.parsed.raw_response.model_copy(update={"created_at": EARLIER})
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="raw provider artifact lineage"):
        ledger.append_raw_artifact(early_raw)

    early_parsed = ParsedProviderClaimArtifact.create(
        artifact_id=_uuid(90_002),
        raw_response=evidence.parsed.raw_response,
        provider_claim_schema=fixture.provider_schema,
        task=intent.task,
        payload=fixture.payload,
        parser_version=evidence.parsed.parser_version,
        created_at=EARLIER,
    )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="parsed provider claim lineage"):
        ledger.append_parsed_claim(early_parsed)

    ledger.append_selection(evidence.selection)
    ledger.append_parsed_claim(evidence.parsed)
    ledger.append_selected_output(evidence.selected)
    early_enriched = evidence.enriched.model_copy(update={"created_at": EARLIER})
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="schema lineage"):
        ledger.append_enriched_output(early_enriched)


def test_schema_drift_and_payload_column_tampering_fail_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    _fixture, intent, _raw_data = _intent()
    ledger.append_intent(intent)
    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'inference_intents_no_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER inference_intents_no_update")
        connection.execute(
            "UPDATE inference_intents SET logical_invocation_id = ? WHERE inference_id = ?",
            (_uuid(88_888), intent.inference_id),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="indexed column"):
        ledger.get_intent(intent.inference_id)

    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER raw_provider_responses_no_delete")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="canonical schema"):
        SQLiteInferenceEvidenceLedger(database, SchemaRegistry())


def test_raw_blob_derived_artifact_identity_tampering_fails_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    _fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request",
        data=raw_data,
    )

    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'raw_provider_responses_no_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER raw_provider_responses_no_update")
        connection.execute(
            "UPDATE raw_provider_responses SET artifact_id = ?",
            (_uuid(92_001),),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="artifact identity"):
        ledger.list_records()


def test_missing_typed_raw_artifact_fails_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    fixture, intent, raw_data = _intent()
    ledger.append_intent(intent)
    stored = ledger.append(
        request_id=intent.request_id,
        provider_request_id="provider-request",
        data=raw_data,
    )
    evidence = _build_after_raw(
        fixture,
        intent,
        raw_data,
        stored.artifact_id,
        stored.provider_request_id,
    )
    ledger.append_terminal(evidence.terminal)

    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'raw_provider_artifacts_no_delete'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER raw_provider_artifacts_no_delete")
        connection.execute("DELETE FROM raw_provider_artifacts")
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="missing their typed raw"):
        ledger.append(
            request_id=intent.request_id,
            provider_request_id=stored.provider_request_id,
            data=raw_data,
        )
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="missing their typed raw"):
        ledger.append_raw_artifact(evidence.parsed.raw_response)
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="missing their typed raw"):
        ledger.append_parsed_claim(evidence.parsed)
    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="missing their typed raw"):
        ledger.get_terminal(intent.inference_id)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT count(*) FROM raw_provider_artifacts").fetchone() == (0,)
    finally:
        connection.close()


def test_typed_raw_normalized_column_tampering_fails_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    evidence = _persist_chain(ledger)

    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'raw_provider_artifacts_no_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER raw_provider_artifacts_no_update")
        connection.execute(
            "UPDATE raw_provider_artifacts SET provider = ? WHERE artifact_id = ?",
            ("tampered-provider", evidence.parsed.raw_response.artifact_id),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="indexed column provider"):
        ledger.get_raw_artifact(evidence.parsed.raw_response.artifact_id)


def test_selection_reason_column_tampering_fails_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    evidence = _persist_chain(ledger)

    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'inference_attempt_selections_no_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER inference_attempt_selections_no_update")
        connection.execute(
            "UPDATE inference_attempt_selections SET selection_reason = ?",
            ("TAMPERED_REASON",),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="selection_reason"):
        ledger.get_selection(
            evidence.selection.logical_invocation_id,
            evidence.selection.policy_version,
        )


def test_crud_uses_targeted_validation_until_explicit_full_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    original_load_state = ledger._load_state

    def unexpected_full_load(_connection: sqlite3.Connection) -> None:
        raise AssertionError("CRUD must not scan the full evidence graph")

    monkeypatch.setattr(ledger, "_load_state", unexpected_full_load)
    evidence = _persist_chain(ledger)

    assert ledger.append_intent(evidence.intent) == evidence.intent
    assert ledger.append(
        request_id=evidence.intent.request_id,
        provider_request_id=evidence.parsed.raw_response.provider_request_id,
        data=evidence.raw_data,
    ) == ledger.get(evidence.parsed.raw_response.artifact_id)
    assert ledger.append_terminal(evidence.terminal) == evidence.terminal
    assert ledger.append_raw_artifact(evidence.parsed.raw_response) == evidence.parsed.raw_response
    assert ledger.append_selection(evidence.selection) == evidence.selection
    assert ledger.append_parsed_claim(evidence.parsed) == evidence.parsed
    assert ledger.append_selected_output(evidence.selected) == evidence.selected
    assert ledger.append_enriched_output(evidence.enriched) == evidence.enriched
    assert ledger.get_intent(evidence.intent.inference_id) == evidence.intent
    assert ledger.get_terminal(evidence.terminal.inference_id) == evidence.terminal
    assert len(ledger.list_records()) == 1

    monkeypatch.setattr(ledger, "_load_state", original_load_state)
    ledger.verify_integrity()


def test_completion_seal_uses_current_incremental_cache_without_full_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    try:
        _persist_chain(ledger)

        def unexpected_full_load(_connection: sqlite3.Connection) -> None:
            raise AssertionError("completion seal must not scan the full evidence graph")

        monkeypatch.setattr(ledger, "_load_state", unexpected_full_load)

        ledger.verify_completion_seal()
    finally:
        ledger.close()


def test_completion_seal_fails_closed_after_external_database_change(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    try:
        connection = sqlite3.connect(database)
        try:
            # This is an external committed database change. The owned ledger
            # connection must observe it through PRAGMA data_version rather than
            # silently refreshing its incrementally validated cache.
            connection.execute("PRAGMA user_version = 999")
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(SQLiteInferenceEvidenceLedgerError, match="seal is stale"):
            ledger.verify_completion_seal()
    finally:
        ledger.close()


def test_full_integrity_audit_remains_available_after_completion_seal(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    evidence = _persist_chain(ledger)
    try:
        ledger.verify_completion_seal()
        connection = sqlite3.connect(database)
        try:
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name = 'enriched_provider_outputs_no_update'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER enriched_provider_outputs_no_update")
            connection.execute(
                "UPDATE enriched_provider_outputs SET semantic_sha256 = ? WHERE artifact_id = ?",
                (_digest(99_998), evidence.enriched.artifact_id),
            )
            connection.execute(trigger_sql)
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(
            SQLiteInferenceEvidenceLedgerError,
            match="indexed column semantic_sha256",
        ):
            ledger.verify_integrity()
    finally:
        ledger.close()


def test_hot_reads_defer_registered_schema_validation_to_full_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = SQLiteInferenceEvidenceLedger(_database(tmp_path), SchemaRegistry())
    evidence = _persist_chain(ledger)
    validate_pinned = ledger.schema_registry.validate_pinned
    validated_schema_ids: list[str] = []

    def counted_validate_pinned(ref: object, instance: object) -> None:
        schema_id = getattr(ref, "schema_id", None)
        assert isinstance(schema_id, str)
        validated_schema_ids.append(schema_id)
        validate_pinned(ref, instance)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ledger.schema_registry,
        "validate_pinned",
        counted_validate_pinned,
    )

    assert ledger.append_terminal(evidence.terminal) == evidence.terminal
    assert evidence.intent.request.output_schema.schema_id in validated_schema_ids
    validated_schema_ids.clear()

    assert ledger.get_intent(evidence.intent.inference_id) == evidence.intent
    assert ledger.get_terminal(evidence.terminal.inference_id) == evidence.terminal
    assert (
        ledger.get_raw_artifact(evidence.parsed.raw_response.artifact_id)
        == evidence.parsed.raw_response
    )
    assert (
        ledger.get_selection(
            evidence.selection.logical_invocation_id,
            evidence.selection.policy_version,
        )
        == evidence.selection
    )
    assert ledger.get_parsed_claim(evidence.parsed.artifact_id) == evidence.parsed
    assert ledger.get_selected_output(evidence.selected.selection_id) == evidence.selected
    assert ledger.get_enriched_output(evidence.enriched.artifact_id) == evidence.enriched
    assert validated_schema_ids == []

    ledger.verify_integrity()

    assert INFERENCE_INTENT_SCHEMA_ID in validated_schema_ids
    assert RAW_PROVIDER_RESPONSE_SCHEMA_ID in validated_schema_ids
    validated_schema_ids.clear()

    restarted = SQLiteInferenceEvidenceLedger(
        _database(tmp_path),
        ledger.schema_registry,
    )

    assert restarted.get_enriched_output(evidence.enriched.artifact_id) == evidence.enriched
    assert INFERENCE_INTENT_SCHEMA_ID in validated_schema_ids
    assert RAW_PROVIDER_RESPONSE_SCHEMA_ID in validated_schema_ids


def test_explicit_full_audit_and_reopen_detect_unrelated_row_tampering(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ledger = SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
    evidence = _persist_chain(ledger)

    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'enriched_provider_outputs_no_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER enriched_provider_outputs_no_update")
        connection.execute(
            "UPDATE enriched_provider_outputs SET semantic_sha256 = ? WHERE artifact_id = ?",
            (_digest(99_999), evidence.enriched.artifact_id),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    assert ledger.get_intent(evidence.intent.inference_id) == evidence.intent
    with pytest.raises(
        SQLiteInferenceEvidenceLedgerError,
        match="indexed column semantic_sha256",
    ):
        ledger.get_enriched_output(evidence.enriched.artifact_id)
    with pytest.raises(
        SQLiteInferenceEvidenceLedgerError,
        match="indexed column semantic_sha256",
    ):
        ledger.verify_integrity()
    with pytest.raises(
        SQLiteInferenceEvidenceLedgerError,
        match="indexed column semantic_sha256",
    ):
        SQLiteInferenceEvidenceLedger(database, SchemaRegistry())
