from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from robata.application.canonical.primary_completion import canonical_collection_digest_root
from robata.contracts.hashing import semantic_sha256
from robata.contracts.primary_completion import (
    PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
    PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION,
    PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION,
    PRIMARY_COMPLETION_RECORD_WIRE_VERSION,
    DetailedResultArtifactReference,
    PrimaryCompletionOutcome,
    PrimaryCompletionRecord,
    PrimaryCompletionTerminalStage,
    create_primary_completion_record,
    primary_completion_record_semantic_projection,
    validate_registered_primary_completion_record,
)
from robata.contracts.schema_registry import (
    SchemaPinMismatchError,
    SchemaRegistry,
    SchemaValidationError,
    deterministic_schema_artifact_id,
)

_COLLECTION_ROOTS_BY_COUNT = {
    "run_membership_count": "run_membership_digest_root",
    "barrier_member_count": "barrier_member_digest_root",
    "hypothesis_count": "hypothesis_digest_root",
    "identity_assignment_count": "identity_assignment_digest_root",
    "new_identity_count": "new_identity_digest_root",
    "identity_relation_count": "identity_relation_digest_root",
    "revision_count": "revision_digest_root",
    "selection_decision_count": "selection_decision_digest_root",
    "current_selection_count": "current_selection_digest_root",
    "successor_outbox_count": "successor_outbox_digest_root",
    "skipped_work_item_count": "skipped_work_item_digest_root",
}

_NO_EVENT_COUNTS = {
    "hypothesis_count": 0,
    "identity_assignment_count": 0,
    "new_identity_count": 0,
    "identity_relation_count": 0,
    "revision_count": 0,
    "selection_decision_count": 0,
    "current_selection_count": 0,
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _uuid(value: int) -> str:
    return f"00000000-0000-5000-8000-{value:012x}"


@pytest.mark.parametrize(
    ("ordered_item_digests", "expected"),
    (
        ((), "556f107d94e7f09418244480eb3baeee36f76bc16058e35636500c212043adf0"),
        (("a" * 64,), "80399f944b13884d75d0d11eb29d95be69bd9fab3318c291d9ec8d95b28235c6"),
        (("a" * 64, "a" * 64), "5e3988ba9542c225a09a20e62c68a76e61ad81527764fc4be724206e03550a35"),
        (("a" * 64, "b" * 64), "4bdb9ec007051dc74a470d4e21fefeee9a97ccc17704f466bddc8b89ec38244b"),
        (("b" * 64, "a" * 64), "f69491e3233b6f2d9ff90042a4653ec434e8d6e59caf706757c6d94eb991fa66"),
        (
            tuple(f"{index:064x}" for index in range(128)),
            "a177f14194db24fbacc5f6013cdcf787eefa48d403c161d69f0744131042531d",
        ),
    ),
)
def test_v3_collection_root_regression_vectors(
    ordered_item_digests: tuple[str, ...],
    expected: str,
) -> None:
    assert canonical_collection_digest_root("x", ordered_item_digests) == expected


def _record(**updates: Any) -> PrimaryCompletionRecord:
    registry = SchemaRegistry()
    fields: dict[str, Any] = {
        "schema_ref": registry.resolve_version(
            PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
            PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION,
        ).ref,
        "run_id": _uuid(2),
        "recording_identity": _digest("recording"),
        "mcap_id": _uuid(3),
        "pipeline_version": "canonical-primary-v1",
        "config_sha256": _digest("configuration"),
        "started_at": "2026-07-20T11:00:00Z",
        "outcome": PrimaryCompletionOutcome.PRIMARY_COMPLETE,
        "terminal_stage": PrimaryCompletionTerminalStage.FINAL_FUSION,
        "terminal_evidence_semantic_sha256": _digest("output-decision"),
        "barrier_definition_semantic_sha256": _digest("barrier-definition"),
        "barrier_reduction_semantic_sha256": _digest("barrier-reduction"),
        "output_decision_semantic_sha256": _digest("output-decision"),
        "output_admission_policy_version": "output-admission-v1",
        "output_admission_policy_sha256": _digest("output-policy"),
        "run_membership_count": 12,
        "run_membership_digest_root": _digest("run-memberships"),
        "barrier_member_count": 6,
        "barrier_member_digest_root": _digest("barrier-members"),
        "hypothesis_count": 2,
        "hypothesis_digest_root": _digest("hypotheses"),
        "identity_assignment_count": 2,
        "identity_assignment_digest_root": _digest("identity-assignments"),
        "new_identity_count": 1,
        "new_identity_digest_root": _digest("new-identities"),
        "identity_relation_count": 1,
        "identity_relation_digest_root": _digest("identity-relations"),
        "revision_count": 2,
        "revision_digest_root": _digest("revisions"),
        "selection_decision_count": 2,
        "selection_decision_digest_root": _digest("selection-decisions"),
        "current_selection_count": 2,
        "current_selection_digest_root": _digest("current-selections"),
        "successor_outbox_count": 3,
        "successor_outbox_digest_root": _digest("successor-outbox"),
        "skipped_work_item_count": 0,
        "skipped_work_item_digest_root": _digest("skipped-work-items"),
        # The common ref exercises exact-ref plumbing only; no governed
        # detailed-result schema or bytes validator is claimed by this fixture.
        "detailed_result": DetailedResultArtifactReference(
            artifact_id=_uuid(4),
            exact_bytes_sha256=_digest("detailed-result-bytes"),
            byte_count=4096,
            media_type="application/json",
            schema_ref=registry.resolve_version("https://schemas.robata.dev/common", "1.0.0").ref,
        ),
        "completed_at": "2026-07-20T12:00:00Z",
    }
    fields.update(updates)
    return create_primary_completion_record(**fields)


def test_registered_schema_round_trip_matches_pydantic_contract() -> None:
    registry = SchemaRegistry()
    record = _record()
    payload = json.loads(record.model_dump_json())

    assert registry.validate_pinned(record.schema_ref, payload) is payload
    assert (
        PrimaryCompletionRecord.model_validate_json(record.model_dump_json(), strict=True) == record
    )
    assert validate_registered_primary_completion_record(record, registry) == record
    assert record.schema_version == PRIMARY_COMPLETION_RECORD_WIRE_VERSION == "3.0"
    assert record.schema_ref.version == PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION == "3.0.0"
    assert (
        record.semantic_projection_version
        == PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION
        == "primary-completion-record-semantic-v3"
    )


def test_frozen_v1_remains_exactly_readable_but_is_not_a_default_model_input() -> None:
    registry = SchemaRegistry()
    frozen_v1 = registry.resolve_version(PRIMARY_COMPLETION_RECORD_SCHEMA_ID, "1.0.0")
    current_v3 = registry.resolve_version(
        PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
        PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION,
    )

    assert frozen_v1.entry.projection_version == "primary-completion-record-semantic-v1"
    assert frozen_v1.entry.wire_version == "1.0"
    assert current_v3.entry.projection_version == (
        PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION
    )
    assert registry.get_schema(frozen_v1.ref)["$id"] == (
        "https://schemas.robata.dev/v1/primary-completion-record.schema.json"
    )
    assert registry.upcasters == ()

    v1_payload = _record().model_dump(mode="json")
    v1_payload.pop("terminal_stage")
    v1_payload.pop("terminal_evidence_semantic_sha256")
    v1_payload.update(
        {
            "schema_version": "1.0",
            "schema_ref": frozen_v1.ref.model_dump(mode="json"),
            "semantic_projection_version": "primary-completion-record-semantic-v1",
        }
    )
    assert registry.validate_pinned(frozen_v1.ref, v1_payload) is v1_payload
    with pytest.raises(ValidationError, match=r"schema_version|semantic_projection_version"):
        PrimaryCompletionRecord.model_validate(v1_payload, strict=True)


@pytest.mark.parametrize(
    ("field_name", "timestamp"),
    [
        ("started_at", "2026-02-30T11:00:00Z"),
        ("completed_at", "2026-13-01T12:00:00Z"),
        ("started_at", "2026-07-20T11:00:00+24:00"),
        ("completed_at", "2026-07-20T12:00:00+00:99"),
    ],
)
def test_model_rejects_calendar_or_offset_invalid_rfc3339_timestamps(
    field_name: str,
    timestamp: str,
) -> None:
    with pytest.raises(ValidationError, match=f"{field_name} must be a valid RFC3339 timestamp"):
        _record(**{field_name: timestamp})


def test_completion_clock_is_ordered_by_instant_across_offsets() -> None:
    with pytest.raises(ValidationError, match="completed_at must be at or after started_at"):
        _record(
            started_at="2026-07-20T12:00:00+02:00",
            completed_at="2026-07-20T09:59:59Z",
        )

    same_instant = _record(
        started_at="2026-07-20T12:00:00+02:00",
        completed_at="2026-07-20T10:00:00Z",
    )
    assert same_instant.completed_at == "2026-07-20T10:00:00Z"


def test_registered_v3_schema_rejects_calendar_invalid_timestamp() -> None:
    registry = SchemaRegistry()
    record = _record()
    payload = record.model_dump(mode="json")
    payload["completed_at"] = "2026-02-30T12:00:00Z"

    with pytest.raises(SchemaValidationError, match="date-time"):
        registry.validate_pinned(record.schema_ref, payload)


def test_v3_policy_namespace_changes_the_semantic_identity() -> None:
    record = _record()
    v3_projection = primary_completion_record_semantic_projection(record)
    frozen_v1 = SchemaRegistry().resolve_version(
        PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
        "1.0.0",
    )
    legacy_projection = {
        **v3_projection,
        "semantic_projection_version": "primary-completion-record-semantic-v1",
        "schema_version": "1.0",
        "schema_ref": frozen_v1.ref.model_dump(mode="json"),
    }

    assert semantic_sha256(v3_projection) == record.semantic_sha256
    assert semantic_sha256(legacy_projection) != record.semantic_sha256


def test_every_collection_requires_a_count_and_digest_root() -> None:
    registry = SchemaRegistry()
    record = _record()
    model_fields = PrimaryCompletionRecord.model_fields

    assert set(_COLLECTION_ROOTS_BY_COUNT).issubset(model_fields)
    assert set(_COLLECTION_ROOTS_BY_COUNT.values()).issubset(model_fields)
    for root_field in _COLLECTION_ROOTS_BY_COUNT.values():
        payload = record.model_dump(mode="json")
        payload.pop(root_field)
        with pytest.raises(SchemaValidationError, match="required"):
            registry.validate_pinned(record.schema_ref, payload)


def test_digest_root_tamper_is_rejected_by_semantic_projection() -> None:
    fields = _record().model_dump(mode="python")
    fields["revision_digest_root"] = _digest("tampered-revisions")

    with pytest.raises(ValidationError, match="semantic_sha256"):
        PrimaryCompletionRecord.model_validate(fields, strict=True)


def test_catalog_exact_pin_matches_published_schema_bytes() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(
        PRIMARY_COMPLETION_RECORD_SCHEMA_ID,
        PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION,
    )
    digest = hashlib.sha256(registered.document_bytes).hexdigest()

    assert registered.ref.sha256 == digest
    assert registered.ref.artifact_id == deterministic_schema_artifact_id(digest)

    forged_ref = registered.ref.model_copy(update={"sha256": "0" * 64})
    forged = _record(schema_ref=forged_ref)
    with pytest.raises(SchemaPinMismatchError):
        validate_registered_primary_completion_record(forged, registry)


def test_detailed_result_schema_ref_is_resolved_as_an_exact_quartet() -> None:
    registry = SchemaRegistry()
    record = _record()
    forged_detail = record.detailed_result.model_copy(
        update={
            "schema_ref": record.detailed_result.schema_ref.model_copy(update={"sha256": "0" * 64})
        }
    )
    forged = _record(detailed_result=forged_detail)

    with pytest.raises(SchemaPinMismatchError):
        validate_registered_primary_completion_record(forged, registry)

    invalid_media = record.detailed_result.model_dump(mode="python")
    invalid_media["media_type"] = "application/octet-stream"
    with pytest.raises(ValidationError, match="media_type"):
        DetailedResultArtifactReference.model_validate(invalid_media, strict=True)


def test_no_events_has_explicit_empty_publication_collections() -> None:
    record = _record(
        outcome=PrimaryCompletionOutcome.PRIMARY_COMPLETE_NO_EVENTS,
        **_NO_EVENT_COUNTS,
    )

    for count_field in _NO_EVENT_COUNTS:
        assert getattr(record, count_field) == 0
        assert getattr(record, _COLLECTION_ROOTS_BY_COUNT[count_field])

    fields = record.model_dump(mode="python")
    fields["identity_assignment_count"] = 1
    with pytest.raises(ValidationError, match="PRIMARY_COMPLETE_NO_EVENTS"):
        PrimaryCompletionRecord.model_validate(fields, strict=True)

    wire_payload = record.model_dump(mode="json")
    wire_payload["identity_assignment_count"] = 1
    with pytest.raises(SchemaValidationError):
        SchemaRegistry().validate_pinned(record.schema_ref, wire_payload)


@pytest.mark.parametrize(
    "terminal_stage",
    [
        PrimaryCompletionTerminalStage.EVENT_PROPOSAL,
        PrimaryCompletionTerminalStage.PROVISIONAL_FUSION,
    ],
)
def test_pre_final_no_events_omit_final_fusion_evidence(
    terminal_stage: PrimaryCompletionTerminalStage,
) -> None:
    record = _record(
        outcome=PrimaryCompletionOutcome.PRIMARY_COMPLETE_NO_EVENTS,
        terminal_stage=terminal_stage,
        terminal_evidence_semantic_sha256=_digest(f"terminal:{terminal_stage.value}"),
        barrier_definition_semantic_sha256=None,
        barrier_reduction_semantic_sha256=None,
        output_decision_semantic_sha256=None,
        output_admission_policy_version=None,
        output_admission_policy_sha256=None,
        barrier_member_count=0,
        **_NO_EVENT_COUNTS,
    )

    assert record.terminal_stage is terminal_stage
    assert record.barrier_definition_semantic_sha256 is None
    SchemaRegistry().validate_pinned(record.schema_ref, record.model_dump(mode="json"))

    with pytest.raises(ValidationError, match="barrier members"):
        _record(
            outcome=PrimaryCompletionOutcome.PRIMARY_COMPLETE_NO_EVENTS,
            terminal_stage=terminal_stage,
            terminal_evidence_semantic_sha256=_digest(f"terminal:{terminal_stage.value}"),
            barrier_definition_semantic_sha256=None,
            barrier_reduction_semantic_sha256=None,
            output_decision_semantic_sha256=None,
            output_admission_policy_version=None,
            output_admission_policy_sha256=None,
            barrier_member_count=1,
            **_NO_EVENT_COUNTS,
        )


def test_skip_outcome_is_explicit_and_mutually_bound_to_skip_evidence() -> None:
    with_skips = _record(
        outcome=PrimaryCompletionOutcome.PRIMARY_COMPLETE_WITH_SKIPS,
        skipped_work_item_count=1,
    )
    assert with_skips.skipped_work_item_count == 1

    with pytest.raises(ValidationError, match="requires skipped_work_item_count"):
        _record(outcome=PrimaryCompletionOutcome.PRIMARY_COMPLETE_WITH_SKIPS)
    with pytest.raises(ValidationError, match="requires PRIMARY_COMPLETE_WITH_SKIPS"):
        _record(skipped_work_item_count=1)


def test_projection_is_versioned_and_excludes_opaque_artifact_id_and_completion_clock() -> None:
    first = _record()
    second = _record(
        completed_at="2026-07-20T13:00:00Z",
    )
    reallocated_detail = first.detailed_result.model_copy(update={"artifact_id": _uuid(900)})
    third = _record(detailed_result=reallocated_detail)

    projection = primary_completion_record_semantic_projection(first)
    assert projection["semantic_projection_version"] == (
        PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION
    )
    assert "completed_at" not in projection
    assert "artifact_id" not in projection["detailed_result"]
    assert projection["detailed_result"]["exact_bytes_sha256"] == (
        first.detailed_result.exact_bytes_sha256
    )
    assert projection["detailed_result"]["schema_ref"] == (
        first.detailed_result.schema_ref.model_dump(mode="json")
    )
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.semantic_sha256 == third.semantic_sha256


def test_primary_completion_outcome_is_closed_to_authoritative_terminal_values() -> None:
    assert {item.value for item in PrimaryCompletionOutcome} == {
        "PRIMARY_COMPLETE",
        "PRIMARY_COMPLETE_NO_EVENTS",
        "PRIMARY_COMPLETE_WITH_SKIPS",
    }
