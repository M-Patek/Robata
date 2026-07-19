from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import (
    LogicalNode,
    ProcessingRunNodeMembership,
    RunNodeDisposition,
    logical_node_from_semantic_digest,
)
from robata.contracts.schema_registry import SchemaRegistry, SchemaValidationError

LOGICAL_NODE_SCHEMA = "https://schemas.robata.dev/logical-node"
MEMBERSHIP_SCHEMA = "https://schemas.robata.dev/processing-run-node-membership"


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _node() -> LogicalNode:
    return logical_node_from_semantic_digest(
        node_type="CAMERA_VIDEO_EXPORT",
        key_namespace="camera-video-export:v1",
        semantic_sha256=semantic_sha256(
            {
                "source_content_sha256": "1" * 64,
                "mapping_profile_sha256": "2" * 64,
                "export_config_sha256": "3" * 64,
            }
        ),
        identity_policy_version="camera-video-export-v1",
    )


def _membership(
    node: LogicalNode,
    *,
    run_id: str = _uuid(1),
    disposition: RunNodeDisposition = RunNodeDisposition.CREATED,
) -> ProcessingRunNodeMembership:
    return ProcessingRunNodeMembership(
        schema_version="1.0",
        run_id=run_id,
        node_type=node.node_type,
        node_logical_key=node.node_logical_key,
        role="OUTPUT",
        disposition=disposition,
        first_work_item_id=_uuid(2),
        attached_at="2026-07-18T12:34:56.123456Z",
    )


def test_logical_node_round_trips_through_pydantic_and_pinned_schema() -> None:
    registry = SchemaRegistry()
    node = _node()
    payload = node.model_dump(mode="json")

    ref = registry.resolve_version(LOGICAL_NODE_SCHEMA, "1.0.0").ref
    assert registry.validate_pinned(ref, payload) is payload
    assert LogicalNode.model_validate_json(json.dumps(payload)) == node
    assert node.identity == (node.node_type, node.node_logical_key)


def test_membership_round_trips_through_pydantic_and_pinned_schema() -> None:
    registry = SchemaRegistry()
    membership = _membership(_node())
    payload = membership.model_dump(mode="json")

    ref = registry.resolve_version(MEMBERSHIP_SCHEMA, "1.0.0").ref
    assert registry.validate_pinned(ref, payload) is payload
    assert ProcessingRunNodeMembership.model_validate_json(json.dumps(payload)) == membership


@pytest.mark.parametrize(
    ("model_validate", "schema_id", "payload"),
    [
        (
            LogicalNode.model_validate,
            LOGICAL_NODE_SCHEMA,
            {**_node().model_dump(mode="json"), "unexpected": True},
        ),
        (
            ProcessingRunNodeMembership.model_validate,
            MEMBERSHIP_SCHEMA,
            {**_membership(_node()).model_dump(mode="json"), "unexpected": True},
        ),
    ],
)
def test_contracts_are_closed_to_extra_fields(
    model_validate: Callable[[Any], Any],
    schema_id: str,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_validate(payload)

    with pytest.raises(SchemaValidationError, match="additional property"):
        SchemaRegistry().validate(schema_id, payload)


@pytest.mark.parametrize(
    ("model_validate", "schema_id", "payload"),
    [
        (
            LogicalNode.model_validate,
            LOGICAL_NODE_SCHEMA,
            {**_node().model_dump(mode="json"), "node_type": 7},
        ),
        (
            ProcessingRunNodeMembership.model_validate,
            MEMBERSHIP_SCHEMA,
            {**_membership(_node()).model_dump(mode="json"), "role": 7},
        ),
    ],
)
def test_contracts_do_not_coerce_non_string_wire_values(
    model_validate: Callable[[Any], Any],
    schema_id: str,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="valid string"):
        model_validate(payload)

    with pytest.raises(SchemaValidationError, match="not of type 'string'"):
        SchemaRegistry().validate(schema_id, payload)


def test_logical_key_is_bound_to_namespace_and_semantic_digest() -> None:
    payload = _node().model_dump(mode="json")
    payload["node_logical_key"] = f"different-namespace:v1:{payload['semantic_sha256']}"

    with pytest.raises(ValidationError, match="key_namespace:semantic_sha256"):
        LogicalNode.model_validate(payload)


def test_invalid_calendar_timestamp_is_rejected_by_pydantic_contract() -> None:
    payload = _membership(_node()).model_dump(mode="json")
    payload["attached_at"] = "2026-02-30T12:00:00Z"

    with pytest.raises(ValidationError, match="valid RFC3339 timestamp"):
        ProcessingRunNodeMembership.model_validate(payload)


@pytest.mark.parametrize("disposition", list(RunNodeDisposition))
def test_membership_accepts_exactly_the_four_normative_dispositions(
    disposition: RunNodeDisposition,
) -> None:
    membership = _membership(_node(), disposition=disposition)
    payload = membership.model_dump(mode="json")

    assert {item.value for item in RunNodeDisposition} == {
        "CREATED",
        "REUSED",
        "INVALIDATED",
        "OBSERVED",
    }
    assert membership.disposition is disposition
    assert SchemaRegistry().validate(MEMBERSHIP_SCHEMA, payload) is payload


def test_contract_models_are_frozen() -> None:
    node = _node()
    membership = _membership(node)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        node.node_type = "DIFFERENT_NODE"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Instance is frozen"):
        membership.role = "INPUT"  # type: ignore[misc]


def test_membership_identity_is_the_exact_architecture_four_tuple() -> None:
    node = _node()
    membership = _membership(node)

    assert membership.identity == (
        membership.run_id,
        node.node_type,
        node.node_logical_key,
        membership.role,
    )


def test_logical_node_fields_exclude_execution_and_storage_identity() -> None:
    assert set(LogicalNode.model_fields) == {
        "schema_version",
        "node_type",
        "key_namespace",
        "node_logical_key",
        "semantic_sha256",
        "identity_policy_version",
    }
    assert not {
        "run_id",
        "qa_run_id",
        "work_item_id",
        "attempt_id",
        "lease_id",
        "path",
        "locator",
    }.intersection(LogicalNode.model_fields)


def test_semantic_projection_reuses_one_node_across_two_processing_runs() -> None:
    producer_projection = {
        "source_content_sha256": "a" * 64,
        "mapping_profile_sha256": "b" * 64,
        "export_config_sha256": "c" * 64,
        "producer_contract_version": "1.0.0",
    }
    assert "run_id" not in producer_projection

    digest = semantic_sha256(producer_projection)
    first_node = logical_node_from_semantic_digest(
        node_type="CAMERA_VIDEO_EXPORT",
        key_namespace="camera-video-export:v1",
        semantic_sha256=digest,
        identity_policy_version="camera-video-export-v1",
    )
    second_node = logical_node_from_semantic_digest(
        node_type="CAMERA_VIDEO_EXPORT",
        key_namespace="camera-video-export:v1",
        semantic_sha256=semantic_sha256(dict(reversed(producer_projection.items()))),
        identity_policy_version="camera-video-export-v1",
    )
    first_membership = _membership(first_node, run_id=_uuid(10))
    second_membership = _membership(
        second_node,
        run_id=_uuid(11),
        disposition=RunNodeDisposition.REUSED,
    )

    assert first_node == second_node
    assert first_node.identity == second_node.identity
    assert first_membership.identity != second_membership.identity
    assert first_membership.node_logical_key == second_membership.node_logical_key
