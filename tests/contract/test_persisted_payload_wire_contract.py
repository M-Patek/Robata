from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from robata.contracts.schema_registry import SchemaPinMismatchError, SchemaRegistry
from robata.event_pipeline.identity_registry import (
    EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
    EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
    EventIdentityOutboxRecord,
    EventIdentityOutboxWireRecord,
    validate_registered_event_identity_outbox_wire_record,
)
from robata.review.models import (
    REVIEW_ANNOTATION_SCHEMA_ID,
    REVIEW_ANNOTATION_SCHEMA_VERSION,
    REVIEW_REOPEN_COMMAND_SCHEMA_ID,
    REVIEW_REOPEN_COMMAND_SCHEMA_VERSION,
    REVIEW_TASK_SCHEMA_ID,
    REVIEW_TASK_SCHEMA_VERSION,
    ReviewAdjudication,
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_annotation,
    create_review_reopen_command,
    create_review_task,
    validate_registered_review_annotation,
    validate_registered_review_reopen_command,
    validate_registered_review_task,
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _registered_payloads(registry: SchemaRegistry) -> tuple[object, ...]:
    outbox_ref = registry.resolve_version(
        EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
        EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
    ).ref
    assignment_key = f"event-identity-assignment:{_digest(1)}"
    outbox_record = EventIdentityOutboxRecord(
        schema_version="1.0",
        outbox_id=str(uuid5(NAMESPACE_URL, f"robata:event-identity-outbox:{assignment_key}")),
        topic="event.identity.assignment",
        recording_identity=_digest(2),
        key=_digest(2),
        assignment_logical_key=assignment_key,
        payload_reference=assignment_key,
        registry_generation=1,
    )
    outbox = EventIdentityOutboxWireRecord.from_record(
        outbox_record,
        schema_ref=outbox_ref,
    )

    task_ref = registry.resolve_version(REVIEW_TASK_SCHEMA_ID, REVIEW_TASK_SCHEMA_VERSION).ref
    policy = create_nonblocking_review_routing_policy(
        policy_version="review-routing-contract-v1",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=10,
                sla_ns=100,
            ),
        ),
    )
    task = create_review_task(
        ReviewRequest(
            request_id=_uuid(10),
            subject=ReviewSubject(
                subject_type="EVENT_HYPOTHESIS",
                subject_id=f"event-hypothesis:{_digest(3)}",
                recording_identity=_digest(4),
            ),
            trigger=ReviewTrigger.LOW_CONFIDENCE,
            reason_codes=("LOW_CONFIDENCE",),
            requested_at_ns=1_000,
        ),
        policy,
        schema_ref=task_ref,
    )
    assert task is not None

    annotation_ref = registry.resolve_version(
        REVIEW_ANNOTATION_SCHEMA_ID,
        REVIEW_ANNOTATION_SCHEMA_VERSION,
    ).ref
    annotation = create_review_annotation(
        task=task,
        lease_fence=1,
        lease_owner="worker-contract",
        reviewer_id="reviewer-contract",
        adjudication=ReviewAdjudication(
            decision_code="ACCEPT",
            reason_codes=("EVIDENCE_VERIFIED",),
            comment="Exact contract fixture.",
        ),
        authored_at_ns=1_020,
        schema_ref=annotation_ref,
    )
    reopen_ref = registry.resolve_version(
        REVIEW_REOPEN_COMMAND_SCHEMA_ID,
        REVIEW_REOPEN_COMMAND_SCHEMA_VERSION,
    ).ref
    reopen = create_review_reopen_command(
        reopen_id=_uuid(11),
        review_task_id=task.review_task_id,
        expected_annotation_id=annotation.annotation_id,
        reason_code="NEW_EVIDENCE",
        requested_at_ns=1_030,
        schema_ref=reopen_ref,
    )
    return outbox, task, annotation, reopen


def test_persisted_payloads_validate_against_exact_registered_pins() -> None:
    registry = SchemaRegistry()
    outbox, task, annotation, reopen = _registered_payloads(registry)

    assert validate_registered_event_identity_outbox_wire_record(outbox, registry) == outbox
    assert validate_registered_review_task(task, registry) == task
    assert validate_registered_review_annotation(annotation, registry) == annotation
    assert validate_registered_review_reopen_command(reopen, registry) == reopen

    for payload in (outbox, task, annotation, reopen):
        registry.validate_pinned(payload.schema_ref, payload.model_dump(mode="json"))


def test_persisted_payload_validators_reject_forged_exact_pins() -> None:
    registry = SchemaRegistry()
    payloads = _registered_payloads(registry)
    validators = (
        validate_registered_event_identity_outbox_wire_record,
        validate_registered_review_task,
        validate_registered_review_annotation,
        validate_registered_review_reopen_command,
    )

    for payload, validator in zip(payloads, validators, strict=True):
        forged_ref = payload.schema_ref.model_copy(update={"sha256": "0" * 64})
        forged = payload.model_copy(update={"schema_ref": forged_ref})
        with pytest.raises(SchemaPinMismatchError):
            validator(forged, registry)


def test_persisted_payload_schemas_are_closed_and_top_level_exact() -> None:
    registry = SchemaRegistry()
    for schema_id, version in (
        (
            EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
            EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
        ),
        (REVIEW_TASK_SCHEMA_ID, REVIEW_TASK_SCHEMA_VERSION),
        (REVIEW_ANNOTATION_SCHEMA_ID, REVIEW_ANNOTATION_SCHEMA_VERSION),
        (REVIEW_REOPEN_COMMAND_SCHEMA_ID, REVIEW_REOPEN_COMMAND_SCHEMA_VERSION),
    ):
        registered = registry.resolve_version(schema_id, version)
        document = registry.get_schema(registered.ref)
        assert document["additionalProperties"] is False
        assert set(document["required"]) == set(document["properties"])
        assert document["properties"]["schema_version"]["const"] == "1.0"
        assert registered.entry.compatibility_mode.value == "NONE"
        assert registered.entry.supported_predecessors == ()

    task_schema = registry.get_schema(
        registry.resolve_version(REVIEW_TASK_SCHEMA_ID, REVIEW_TASK_SCHEMA_VERSION).ref
    )
    assert task_schema["properties"]["requested_at_ns"]["type"] == "string"
    assert task_schema["properties"]["due_at_ns"]["type"] == "string"
