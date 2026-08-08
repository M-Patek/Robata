from __future__ import annotations

from pathlib import Path

import pytest

from robata.contracts import NanosecondInterval
from robata.contracts.hashing import exact_bytes_sha256
from robata.contracts.perception_stream import (
    RefineReason,
    RefineTargetField,
    create_perception_refine_request,
)
from robata.contracts.schema_registry import (
    SchemaRef,
    SchemaRegistry,
    SchemaValidationError,
)
from scripts.export_persisted_wire_schema import export_schema
from tests.support.perception_stream import digest, make_context, make_observation

_PERCEPTION_SCHEMAS = (
    (
        "https://schemas.robata.dev/perception-context-manifest",
        "perception-context-manifest",
        "7b47ea44-25e1-52bb-a9a2-2c969295a602",
        "9f68542d42ef3383284ce9ef2ebf797adc7d884ee72f7392b20215d632e49c7b",
        "perception-context-semantic-v1",
    ),
    (
        "https://schemas.robata.dev/mage-observation",
        "mage-observation",
        "0a877693-e6da-8163-ae69-f99d4b076ae3",
        "af48ed154084029d59a39d00504624a317bda1e66239559bf91f4ed507e0492a",
        "mage-observation-semantic-v1",
    ),
    (
        "https://schemas.robata.dev/perception-refine-request",
        "perception-refine-request",
        "841007f6-0b7f-2488-6f85-4ade12738549",
        "b4f1a71d5d62c03ac8e8c3697a144ecafc10fed8b2f412cad8e32217f73ee2f0",
        "perception-refine-request-semantic-v1",
    ),
)


def _refine_payload() -> dict[str, object]:
    observation = make_observation()
    hypothesis_digest = digest("event-hypothesis")
    request = create_perception_refine_request(
        source_observation_logical_key=observation.observation_logical_key,
        source_observation_semantic_sha256=observation.observation_semantic_sha256,
        target_hypothesis_logical_key=f"event-hypothesis-vnext:{hypothesis_digest}",
        target_hypothesis_semantic_sha256=hypothesis_digest,
        reason=RefineReason.BOUNDARY,
        target_fields=(RefineTargetField.START_BOUNDARY, RefineTargetField.END_BOUNDARY),
        refine_interval=NanosecondInterval(start_ns=500_000_000, end_ns=2_500_000_000),
        refine_policy_version="bounded-refine-v1",
        prompt_version="boundary-only-v1",
    )
    return request.model_dump(mode="json")


def test_perception_vnext_catalog_pins_exact_immutable_schemas() -> None:
    registry = SchemaRegistry()

    for schema_id, _export_name, artifact_id, sha256, projection_version in _PERCEPTION_SCHEMAS:
        registered = registry.resolve_version(schema_id, "1.0.0")
        assert registered.ref == SchemaRef(
            schema_id=schema_id,
            version="1.0.0",
            artifact_id=artifact_id,
            sha256=sha256,
        )
        assert registered.entry.wire_version == "1.0"
        assert registered.entry.owner == "robata-contracts"
        assert registered.entry.projection_version == projection_version
        assert registered.entry.supported_predecessors == ()


def test_perception_vnext_registered_bytes_match_deterministic_exports(tmp_path: Path) -> None:
    registry = SchemaRegistry()

    for schema_id, export_name, _artifact_id, _sha256, _projection_version in _PERCEPTION_SCHEMAS:
        registered = registry.resolve_version(schema_id, "1.0.0")
        exported = tmp_path / f"{export_name}.schema.json"
        export_schema(export_name, exported)

        assert registered.document_bytes == registered.path.read_bytes()
        assert registered.document_bytes == exported.read_bytes()
        assert registered.ref.sha256 == exact_bytes_sha256(registered.document_bytes)


def test_perception_vnext_registry_enforces_exact_six_camera_wire_shapes() -> None:
    registry = SchemaRegistry()

    context = make_context().model_dump(mode="json")
    observation = make_observation().model_dump(mode="json")

    context_schema = registry.resolve_version(
        "https://schemas.robata.dev/perception-context-manifest",
        "1.0.0",
    )
    observation_schema = registry.resolve_version(
        "https://schemas.robata.dev/mage-observation",
        "1.0.0",
    )
    registry.validate_pinned(context_schema.ref, context)
    registry.validate_pinned(observation_schema.ref, observation)

    missing_context_camera = context.copy()
    missing_context_camera["cameras"] = {
        camera_id: binding
        for camera_id, binding in context["cameras"].items()
        if camera_id != "cam_06"
    }
    extra_context_camera = context.copy()
    extra_context_camera["cameras"] = {
        **context["cameras"],
        "cam_07": context["cameras"]["cam_06"],
    }

    empty_semantic_qa = observation.copy()
    empty_semantic_qa["semantic_qa"] = {}
    garbage_semantic_qa = observation.copy()
    garbage_semantic_qa["semantic_qa"] = {camera_id: {} for camera_id in observation["semantic_qa"]}

    empty_camera_evidence = observation.copy()
    empty_camera_evidence["observations"] = [
        {**observation["observations"][0], "camera_evidence": {}}
    ]
    garbage_camera_evidence = observation.copy()
    garbage_camera_evidence["observations"] = [
        {
            **observation["observations"][0],
            "camera_evidence": {
                camera_id: {} for camera_id in observation["observations"][0]["camera_evidence"]
            },
        }
    ]

    for schema, invalid_payload in (
        (context_schema, missing_context_camera),
        (context_schema, extra_context_camera),
        (observation_schema, empty_semantic_qa),
        (observation_schema, garbage_semantic_qa),
        (observation_schema, empty_camera_evidence),
        (observation_schema, garbage_camera_evidence),
    ):
        with pytest.raises(SchemaValidationError):
            registry.validate_pinned(schema.ref, invalid_payload)


def test_perception_vnext_registry_accepts_core_payloads_and_rejects_unknown_fields() -> None:
    registry = SchemaRegistry()
    payloads = {
        "https://schemas.robata.dev/perception-context-manifest": make_context().model_dump(
            mode="json"
        ),
        "https://schemas.robata.dev/mage-observation": make_observation().model_dump(mode="json"),
        "https://schemas.robata.dev/perception-refine-request": _refine_payload(),
    }

    for schema_id, payload in payloads.items():
        registered = registry.resolve_version(schema_id, "1.0.0")
        registry.validate_pinned(registered.ref, payload)

        tampered = dict(payload)
        tampered["unpublished_extension"] = True
        with pytest.raises(SchemaValidationError):
            registry.validate_pinned(registered.ref, tampered)
