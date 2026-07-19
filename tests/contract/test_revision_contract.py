from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.contracts.revisions import (
    CurrentSelection,
    ImmutableNodeRevision,
    RevisionEligibility,
    SelectionDecision,
    create_immutable_node_revision,
    create_selection_decision,
)
from robata.contracts.schema_registry import SchemaRegistry, SchemaValidationError

REVISION_SCHEMA = "https://schemas.robata.dev/immutable-node-revision"
DECISION_SCHEMA = "https://schemas.robata.dev/selection-decision"
CURRENT_SCHEMA = "https://schemas.robata.dev/current-selection"
SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "schemas" / "v1"


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _logical_key(namespace: str, seed: str) -> str:
    return f"{namespace}:{semantic_sha256({'seed': seed})}"


def _revision(**overrides: Any) -> ImmutableNodeRevision:
    values: dict[str, Any] = {
        "revision_id": _uuid(10),
        "subject_type": "CAMERA_VIDEO_EXPORT",
        "subject_id": _logical_key("camera-video-export:v1", "subject"),
        "revision_key_namespace": "camera-video-revision:v1",
        "payload_sha256": semantic_sha256({"payload": "one"}),
        "lineage_sha256": semantic_sha256({"lineage": "one"}),
        "status_at_publication": "PUBLISHED",
        "eligibility_at_publication": RevisionEligibility.ELIGIBLE,
        "revision_policy_version": "camera-video-revision-v1",
        "supersedes_revision_id": None,
        "supersedes_revision_logical_key": None,
        "published_at": "2026-07-18T12:00:00.123456Z",
    }
    values.update(overrides)
    return create_immutable_node_revision(**values)


def _decision(**overrides: Any) -> SelectionDecision:
    values: dict[str, Any] = {
        "selection_decision_id": _uuid(20),
        "selection_key_namespace": "camera-video-selection:v1",
        "subject_type": "CAMERA_VIDEO_EXPORT",
        "subject_id": _logical_key("camera-video-export:v1", "subject"),
        "selected_revision_id": _uuid(10),
        "selected_revision_logical_key": _revision().revision_logical_key,
        "previous_selection_decision_id": None,
        "previous_selection_decision_logical_key": None,
        "selection_sequence": 1,
        "selection_policy_version": "camera-video-selection-v1",
        "projection_version": "current-selection-v1",
        "selected_at": "2026-07-18T12:01:00.123456Z",
    }
    values.update(overrides)
    return create_selection_decision(**values)


def _current() -> CurrentSelection:
    decision = _decision()
    return CurrentSelection(
        schema_version="1.0",
        subject_type=decision.subject_type,
        subject_id=decision.subject_id,
        selected_revision_id=decision.selected_revision_id,
        selection_decision_id=decision.selection_decision_id,
        selection_policy_version=decision.selection_policy_version,
        projection_version=decision.projection_version,
        selected_at=decision.selected_at,
    )


@pytest.mark.parametrize(
    ("schema_id", "model"),
    [
        (REVISION_SCHEMA, _revision()),
        (DECISION_SCHEMA, _decision()),
        (CURRENT_SCHEMA, _current()),
    ],
)
def test_revision_contracts_round_trip_through_models_and_pinned_schemas(
    schema_id: str,
    model: ImmutableNodeRevision | SelectionDecision | CurrentSelection,
) -> None:
    registry = SchemaRegistry()
    payload = model.model_dump(mode="json")
    ref = registry.resolve_version(schema_id, "1.0.0").ref

    assert registry.validate_pinned(ref, payload) is payload
    assert type(model).model_validate_json(json.dumps(payload)) == model


def test_contract_field_sets_are_exact_and_revision_has_no_current_state() -> None:
    assert set(ImmutableNodeRevision.model_fields) == {
        "schema_version",
        "revision_id",
        "subject_type",
        "subject_id",
        "revision_key_namespace",
        "revision_logical_key",
        "semantic_sha256",
        "payload_sha256",
        "lineage_sha256",
        "status_at_publication",
        "eligibility_at_publication",
        "revision_policy_version",
        "supersedes_revision_id",
        "supersedes_revision_logical_key",
        "published_at",
    }
    assert set(SelectionDecision.model_fields) == {
        "schema_version",
        "selection_decision_id",
        "selection_key_namespace",
        "selection_decision_logical_key",
        "semantic_sha256",
        "subject_type",
        "subject_id",
        "selected_revision_id",
        "selected_revision_logical_key",
        "previous_selection_decision_id",
        "previous_selection_decision_logical_key",
        "selection_sequence",
        "selection_policy_version",
        "projection_version",
        "selected_at",
    }
    assert set(CurrentSelection.model_fields) == {
        "schema_version",
        "subject_type",
        "subject_id",
        "selected_revision_id",
        "selection_decision_id",
        "selection_policy_version",
        "projection_version",
        "selected_at",
    }
    assert not {
        "is_current",
        "current_revision_pointer",
        "selection_decision_id",
        "selected_at",
        "projection_version",
        "run_id",
        "work_item_id",
    }.intersection(ImmutableNodeRevision.model_fields)


def test_revision_builder_uses_the_exact_semantic_projection() -> None:
    revision = _revision()
    expected = semantic_sha256(
        {
            "semantic_projection_version": "immutable-node-revision-semantic-v1",
            "subject_type": revision.subject_type,
            "subject_id": revision.subject_id,
            "payload_sha256": revision.payload_sha256,
            "lineage_sha256": revision.lineage_sha256,
            "status_at_publication": revision.status_at_publication,
            "eligibility_at_publication": revision.eligibility_at_publication.value,
            "revision_policy_version": revision.revision_policy_version,
            "supersedes_revision_logical_key": revision.supersedes_revision_logical_key,
        }
    )

    assert revision.semantic_sha256 == expected
    assert revision.revision_logical_key == f"{revision.revision_key_namespace}:{expected}"


def test_revision_identity_excludes_uuids_time_and_key_namespace() -> None:
    predecessor_key = _logical_key("camera-video-revision:v1", "predecessor")
    first = _revision(
        supersedes_revision_id=_uuid(1),
        supersedes_revision_logical_key=predecessor_key,
    )
    changed_audit = _revision(
        revision_id=_uuid(11),
        revision_key_namespace="alternate-revision:v2",
        supersedes_revision_id=_uuid(2),
        supersedes_revision_logical_key=predecessor_key,
        published_at="2027-01-01T00:00:00+08:00",
    )

    assert first.semantic_sha256 == changed_audit.semantic_sha256
    assert first.revision_logical_key != changed_audit.revision_logical_key
    assert first.revision_id != changed_audit.revision_id


@pytest.mark.parametrize(
    "change",
    [
        {"subject_type": "OTHER_NODE"},
        {"subject_id": _logical_key("camera-video-export:v1", "other")},
        {"payload_sha256": "1" * 64},
        {"lineage_sha256": "2" * 64},
        {"status_at_publication": "REVIEWED"},
        {"eligibility_at_publication": RevisionEligibility.INELIGIBLE},
        {"revision_policy_version": "camera-video-revision-v2"},
        {
            "supersedes_revision_id": _uuid(1),
            "supersedes_revision_logical_key": _logical_key(
                "camera-video-revision:v1", "predecessor"
            ),
        },
    ],
)
def test_every_revision_semantic_input_changes_the_digest(change: dict[str, Any]) -> None:
    assert _revision(**change).semantic_sha256 != _revision().semantic_sha256


def test_selection_builder_uses_the_exact_semantic_projection() -> None:
    decision = _decision()
    expected = semantic_sha256(
        {
            "semantic_projection_version": "selection-decision-semantic-v1",
            "subject_type": decision.subject_type,
            "subject_id": decision.subject_id,
            "selected_revision_logical_key": decision.selected_revision_logical_key,
            "previous_selection_decision_logical_key": (
                decision.previous_selection_decision_logical_key
            ),
            "selection_policy_version": decision.selection_policy_version,
            "projection_version": decision.projection_version,
        }
    )

    assert decision.semantic_sha256 == expected
    assert decision.selection_decision_logical_key == (
        f"{decision.selection_key_namespace}:{expected}"
    )


def test_selection_identity_excludes_uuids_sequence_time_and_key_namespace() -> None:
    previous_key = _logical_key("camera-video-selection:v1", "previous")
    first = _decision(
        previous_selection_decision_id=_uuid(1),
        previous_selection_decision_logical_key=previous_key,
        selection_sequence=2,
    )
    changed_audit = _decision(
        selection_decision_id=_uuid(21),
        selection_key_namespace="alternate-selection:v2",
        selected_revision_id=_uuid(11),
        previous_selection_decision_id=_uuid(2),
        previous_selection_decision_logical_key=previous_key,
        selection_sequence=99,
        selected_at="2027-01-01T00:00:00+08:00",
    )

    assert first.semantic_sha256 == changed_audit.semantic_sha256
    assert first.selection_decision_logical_key != changed_audit.selection_decision_logical_key


@pytest.mark.parametrize(
    "change",
    [
        {"subject_type": "OTHER_NODE"},
        {"subject_id": _logical_key("camera-video-export:v1", "other")},
        {"selected_revision_logical_key": _logical_key("camera-video-revision:v1", "other")},
        {
            "previous_selection_decision_id": _uuid(1),
            "previous_selection_decision_logical_key": _logical_key(
                "camera-video-selection:v1", "previous"
            ),
            "selection_sequence": 2,
        },
        {"selection_policy_version": "camera-video-selection-v2"},
        {"projection_version": "current-selection-v2"},
    ],
)
def test_every_selection_semantic_input_changes_the_digest(change: dict[str, Any]) -> None:
    assert _decision(**change).semantic_sha256 != _decision().semantic_sha256


@pytest.mark.parametrize(
    ("model", "id_field", "key_field"),
    [
        (_revision(), "supersedes_revision_id", "supersedes_revision_logical_key"),
        (
            _decision(),
            "previous_selection_decision_id",
            "previous_selection_decision_logical_key",
        ),
    ],
)
@pytest.mark.parametrize("missing", ["id", "key"])
def test_nullable_predecessor_fields_require_both_null_or_both_set(
    model: ImmutableNodeRevision | SelectionDecision,
    id_field: str,
    key_field: str,
    missing: str,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload[id_field] = None if missing == "id" else _uuid(90)
        payload[key_field] = None if missing == "key" else _logical_key("predecessor:v1", "x")

    python_payload = model.model_dump(mode="python")
    mutate(python_payload)

    with pytest.raises(ValidationError, match="both null or both non-null"):
        type(model).model_validate(python_payload)

    json_payload = model.model_dump(mode="json")
    mutate(json_payload)
    schema_id = REVISION_SCHEMA if isinstance(model, ImmutableNodeRevision) else DECISION_SCHEMA
    with pytest.raises(SchemaValidationError):
        SchemaRegistry().validate(schema_id, json_payload)


@pytest.mark.parametrize("invalid", [0, -1, 1.0, 2**63, True])
def test_selection_sequence_is_a_strict_positive_int64(invalid: object) -> None:
    decision = _decision()
    payload = decision.model_dump(mode="json")
    payload["selection_sequence"] = invalid

    with pytest.raises(ValidationError):
        SelectionDecision.model_validate(payload)
    with pytest.raises(SchemaValidationError):
        SchemaRegistry().validate(DECISION_SCHEMA, payload)

    assert _decision(selection_sequence=2**63 - 1).selection_sequence == 2**63 - 1


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("selection_sequence", 1.0),
        ("selected_at", "2026-02-30T12:00:00Z"),
    ],
)
def test_frozen_directory_registry_uses_strict_revision_validation(
    field: str,
    invalid: object,
) -> None:
    payload = _decision().model_dump(mode="json")
    payload[field] = invalid

    with pytest.raises(SchemaValidationError):
        SchemaRegistry(SCHEMA_DIRECTORY).validate("selection-decision", payload)


@pytest.mark.parametrize(
    ("model_validate", "schema_id", "payload", "field"),
    [
        (
            ImmutableNodeRevision.model_validate,
            REVISION_SCHEMA,
            _revision().model_dump(mode="python"),
            "published_at",
        ),
        (
            SelectionDecision.model_validate,
            DECISION_SCHEMA,
            _decision().model_dump(mode="python"),
            "selected_at",
        ),
        (
            CurrentSelection.model_validate,
            CURRENT_SCHEMA,
            _current().model_dump(mode="python"),
            "selected_at",
        ),
    ],
)
@pytest.mark.parametrize(
    "invalid_timestamp",
    ["2026-02-30T12:00:00Z", "2026-01-01T12:00:00+24:00"],
)
def test_timestamps_reject_calendar_invalid_rfc3339_values_in_models_and_schemas(
    model_validate: Callable[[Any], Any],
    schema_id: str,
    payload: dict[str, Any],
    field: str,
    invalid_timestamp: str,
) -> None:
    payload[field] = invalid_timestamp

    with pytest.raises(ValidationError, match="valid RFC3339 timestamp"):
        model_validate(payload)
    with pytest.raises(SchemaValidationError):
        SchemaRegistry().validate(schema_id, payload)


@pytest.mark.parametrize(
    ("model_validate", "schema_id", "payload"),
    [
        (
            ImmutableNodeRevision.model_validate,
            REVISION_SCHEMA,
            _revision().model_dump(mode="json"),
        ),
        (
            SelectionDecision.model_validate,
            DECISION_SCHEMA,
            _decision().model_dump(mode="json"),
        ),
        (CurrentSelection.model_validate, CURRENT_SCHEMA, _current().model_dump(mode="json")),
    ],
)
def test_contracts_are_closed_and_do_not_coerce_strings(
    model_validate: Callable[[Any], Any],
    schema_id: str,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_validate({**payload, "unexpected": True})
    with pytest.raises(SchemaValidationError, match="additional property"):
        SchemaRegistry().validate(schema_id, {**payload, "unexpected": True})

    payload["subject_type"] = 7
    with pytest.raises(ValidationError, match="valid string"):
        model_validate(payload)
    with pytest.raises(SchemaValidationError, match="not of type 'string'"):
        SchemaRegistry().validate(schema_id, payload)


def test_semantic_digest_and_logical_key_tampering_is_rejected() -> None:
    revision = _revision()
    revision_payload = revision.model_dump(mode="python")
    revision_payload["semantic_sha256"] = "0" * 64
    revision_payload["revision_logical_key"] = f"{revision.revision_key_namespace}:{'0' * 64}"
    with pytest.raises(ValidationError, match="semantic_sha256"):
        ImmutableNodeRevision.model_validate(revision_payload)

    decision = _decision()
    decision_payload = decision.model_dump(mode="json")
    decision_payload["selection_decision_logical_key"] = (
        f"{decision.selection_key_namespace}:{'0' * 64}"
    )
    with pytest.raises(ValidationError, match="selection_key_namespace:semantic_sha256"):
        SelectionDecision.model_validate(decision_payload)


def test_contract_models_are_frozen() -> None:
    revision = _revision()
    decision = _decision()
    current = _current()

    with pytest.raises(ValidationError, match="Instance is frozen"):
        revision.status_at_publication = "CHANGED"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Instance is frozen"):
        decision.selection_sequence = 2  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Instance is frozen"):
        current.selected_revision_id = _uuid(99)  # type: ignore[misc]
