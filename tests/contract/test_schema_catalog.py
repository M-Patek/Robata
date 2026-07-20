from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robata.contracts.schema_registry import (
    SchemaAmbiguityError,
    SchemaDefinitionError,
    SchemaPinMismatchError,
    SchemaRef,
    SchemaRegistry,
    deterministic_schema_artifact_id,
)

SCHEMA_ID = "https://schemas.robata.dev/camera-video-export-manifest"
LOGICAL_NODE_SCHEMA_ID = "https://schemas.robata.dev/logical-node"
RUN_NODE_MEMBERSHIP_SCHEMA_ID = "https://schemas.robata.dev/processing-run-node-membership"
IMMUTABLE_REVISION_SCHEMA_ID = "https://schemas.robata.dev/immutable-node-revision"
SELECTION_DECISION_SCHEMA_ID = "https://schemas.robata.dev/selection-decision"
CURRENT_SELECTION_SCHEMA_ID = "https://schemas.robata.dev/current-selection"
MCAP_VALIDATION_REPORT_SCHEMA_ID = "https://schemas.robata.dev/mcap-validation-report"
MCAP_READY_MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/mcap-manifest"
ALIGNMENT_MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/alignment-manifest"
ENRICHED_OUTPUT_SCHEMA_ID = "https://schemas.robata.dev/orchestrator-enriched-output"
INFERENCE_INTENT_SCHEMA_ID = "https://schemas.robata.dev/inference-intent"
MODEL_INFERENCE_SCHEMA_ID = "https://schemas.robata.dev/model-inference"
INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID = "https://schemas.robata.dev/inference-attempt-selection"
RAW_PROVIDER_RESPONSE_SCHEMA_ID = "https://schemas.robata.dev/raw-provider-response-artifact"
PARSED_PROVIDER_CLAIM_SCHEMA_ID = "https://schemas.robata.dev/parsed-provider-claim-artifact"
SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID = "https://schemas.robata.dev/selected-attempt-output"


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _synthetic_catalog(root: Path) -> tuple[Path, dict[str, Any]]:
    document = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.robata.dev/v1/synthetic.schema.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {"value": {"type": "integer"}},
    }
    raw = _write_json(root / "v1" / "synthetic.schema.json", document)
    digest = hashlib.sha256(raw).hexdigest()
    entry = {
        "ref": {
            "schema_id": "https://schemas.robata.dev/synthetic",
            "version": "1.0.0",
            "artifact_id": deterministic_schema_artifact_id(digest),
            "sha256": digest,
        },
        "wire_version": "1.0",
        "document_id": document["$id"],
        "artifact_path": "v1/synthetic.schema.json",
        "owner": "test",
        "canonicalization_version": "rfc8785-v1",
        "projection_version": "synthetic-v1",
        "compatibility_mode": "NONE",
        "lifecycle": "ACTIVE",
        "supported_software": {
            "min_inclusive": "0.1.0",
            "max_exclusive": "0.2.0",
        },
        "supported_predecessors": [],
    }
    catalog = {"catalog_version": "1.0", "schemas": [entry], "upcasters": []}
    path = root / "schema-catalog.json"
    _write_json(path, catalog)
    return path, catalog


def test_production_catalog_pins_exact_v2_manifest() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(SCHEMA_ID, "2.0.0")

    assert registered.ref == SchemaRef(
        schema_id=SCHEMA_ID,
        version="2.0.0",
        artifact_id="bd1ecbc1-a5c2-86f4-0a7c-ebfbc0ae6524",
        sha256="d3f688cff9e7e27c2be1d6424e03fdb35aabb0f0428c7e310040fbed62844ec2",
    )
    assert registered.entry.wire_version == "2.0"


def test_production_catalog_pins_exact_logical_node_contracts() -> None:
    registry = SchemaRegistry()

    assert registry.resolve_version(LOGICAL_NODE_SCHEMA_ID, "1.0.0").ref == SchemaRef(
        schema_id=LOGICAL_NODE_SCHEMA_ID,
        version="1.0.0",
        artifact_id="b4cbbe14-96cd-5018-5111-32834707feb2",
        sha256="fa5562e212d5834d932c237b84ea1864bb1d222d348ea90c31129bec38076e40",
    )
    assert registry.resolve_version(RUN_NODE_MEMBERSHIP_SCHEMA_ID, "1.0.0").ref == SchemaRef(
        schema_id=RUN_NODE_MEMBERSHIP_SCHEMA_ID,
        version="1.0.0",
        artifact_id="c55ccc4d-89b6-7845-9e31-7d55e0a427db",
        sha256="8a472d73825e581bb2fea948db4f3613272c95717f0a2787d232d0fc5e03033e",
    )


def test_production_catalog_pins_exact_revision_selection_contracts() -> None:
    registry = SchemaRegistry()

    assert registry.resolve_version(IMMUTABLE_REVISION_SCHEMA_ID, "1.0.0").ref == SchemaRef(
        schema_id=IMMUTABLE_REVISION_SCHEMA_ID,
        version="1.0.0",
        artifact_id="b32fe5ed-da66-ec11-9149-c8b5ebc34442",
        sha256="2fe93c147d1290d7ca980bb9cdd0addfdb8dfff5398489cb5a23cfc1d8920dd6",
    )
    assert registry.resolve_version(SELECTION_DECISION_SCHEMA_ID, "1.0.0").ref == SchemaRef(
        schema_id=SELECTION_DECISION_SCHEMA_ID,
        version="1.0.0",
        artifact_id="83f8a99d-0e04-f365-e323-9a04bcb18830",
        sha256="2098d47dfddf58705e47ef9f1a149a1455e0bb5c918879587de6a10b488092b8",
    )
    assert registry.resolve_version(CURRENT_SELECTION_SCHEMA_ID, "1.0.0").ref == SchemaRef(
        schema_id=CURRENT_SELECTION_SCHEMA_ID,
        version="1.0.0",
        artifact_id="c7cb333b-1e64-a38b-3401-87229bf97457",
        sha256="71c819d7f14f3b16f6a5f50c60e15cdfb4e5cab44b716fa1a35d0cb853c1c625",
    )


def test_production_catalog_pins_exact_v2_admission_evidence() -> None:
    registry = SchemaRegistry()

    assert registry.resolve_version(MCAP_VALIDATION_REPORT_SCHEMA_ID, "2.0.0").ref == SchemaRef(
        schema_id=MCAP_VALIDATION_REPORT_SCHEMA_ID,
        version="2.0.0",
        artifact_id="8d5525ee-4a06-2096-6e6b-9d610b106e69",
        sha256="882a4a6544a6c242c6faf2203a7b0c645873ff536356fc8c1f50c15a1acf3b48",
    )
    assert registry.resolve_version(MCAP_READY_MANIFEST_SCHEMA_ID, "2.0.0").ref == SchemaRef(
        schema_id=MCAP_READY_MANIFEST_SCHEMA_ID,
        version="2.0.0",
        artifact_id="9b3329aa-53fe-df00-f02d-15e49d911f38",
        sha256="533e365e99fb9c4c8d919944936ce34e79b8fe761440d8b5be81765cffc84ed4",
    )
    assert registry.resolve_version(ALIGNMENT_MANIFEST_SCHEMA_ID, "2.0.0").ref == SchemaRef(
        schema_id=ALIGNMENT_MANIFEST_SCHEMA_ID,
        version="2.0.0",
        artifact_id="f581927c-b4bf-fb8f-66aa-35cbdd9cc7c1",
        sha256="a953d17bb5846d4d40f0af963a8b4b88504c90fa27a7ecaaf689c3f19a5d8469",
    )


def test_production_catalog_pins_exact_v2_enriched_output() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(ENRICHED_OUTPUT_SCHEMA_ID, "2.0.0")

    assert registered.ref == SchemaRef(
        schema_id=ENRICHED_OUTPUT_SCHEMA_ID,
        version="2.0.0",
        artifact_id="9d36ff6a-d241-afcc-1d70-8b8e8d5e84c7",
        sha256="9bbe1ae372f20a7a15e563873ea83522e0e5b18826d9f184e10d2f38df8c103d",
    )
    assert registered.entry.wire_version == "2.0"
    assert registered.entry.projection_version == "orchestrator-enriched-output-v2"

    document = registry.get_schema(registered.ref)
    assert document["properties"]["schema_version"] == {"const": "2.0"}
    selected_attempt = document["$defs"]["selectedAttempt"]
    assert {
        "selection_id",
        "logical_invocation_id",
        "selection_decision_logical_key",
        "selection_policy_version",
    } <= set(selected_attempt["required"])


def test_production_catalog_pins_exact_inference_evidence_contracts() -> None:
    registry = SchemaRegistry()
    expected = (
        SchemaRef(
            schema_id=INFERENCE_INTENT_SCHEMA_ID,
            version="1.0.0",
            artifact_id="59c9e34b-631f-5795-a115-2751fb7573b7",
            sha256="d20538b2f81a4c4b1f7dcc29448638614e585887a135bfc107d9b4d2f9104f40",
        ),
        SchemaRef(
            schema_id=MODEL_INFERENCE_SCHEMA_ID,
            version="1.0.0",
            artifact_id="2fcf18d2-5b9e-cb8d-fc39-1d16ed4159f3",
            sha256="6eb192a4b840f9b8e79000ca07a8770c87fe7462fc71c296911de6848301c9e8",
        ),
        SchemaRef(
            schema_id=INFERENCE_ATTEMPT_SELECTION_SCHEMA_ID,
            version="1.0.0",
            artifact_id="dd131ca4-9ef8-9278-345b-a1a1832a4472",
            sha256="c162c55442c4fca4ca96d6882e5e074d8a19cf683f5e7788ccc76eecafcf8365",
        ),
        SchemaRef(
            schema_id=RAW_PROVIDER_RESPONSE_SCHEMA_ID,
            version="1.0.0",
            artifact_id="b44d2bc9-7068-63a9-e179-2a4416899358",
            sha256="c71f8ab02fbd3c5116ece866afd15f0527f176ccd23ed209fb3d7c748bfd0bd4",
        ),
        SchemaRef(
            schema_id=PARSED_PROVIDER_CLAIM_SCHEMA_ID,
            version="1.0.0",
            artifact_id="ad3c7c60-8759-606e-4698-9a6edeba8bdd",
            sha256="09ab053605ef1564f62352b64e556be91c93ec40ca2d2579b48f10387d7d705d",
        ),
        SchemaRef(
            schema_id=SELECTED_ATTEMPT_OUTPUT_SCHEMA_ID,
            version="1.0.0",
            artifact_id="a3515f6d-1f9c-2c77-e45a-6546c13210bb",
            sha256="7d20cba98321cf2c360580bca82124be958ba2e840d9483de8324681c8f330ec",
        ),
    )

    for ref in expected:
        registered = registry.resolve_version(ref.schema_id, ref.version)
        assert registered.ref == ref
        assert registered.entry.wire_version == "1.0"
        document = registry.get_schema(ref)
        assert document["additionalProperties"] is False
        assert set(document["required"]) == set(document["properties"])

    intent = registry.get_schema(expected[0])
    selection = registry.get_schema(expected[2])
    assert selection["properties"]["selection_reason"] == {
        "minLength": 1,
        "type": "string",
    }
    for ref in expected[:5]:
        document = registry.get_schema(ref)
        assert document["properties"]["schema_version"] == {"const": "1.0"}

    request = intent["$defs"]["VisionInferenceRequest"]
    assert request["additionalProperties"] is False
    assert set(request["required"]) == set(request["properties"])
    for definition_name in ("InferenceInputPlan", "RequestCatalog", "VisionInferenceRequest"):
        assert intent["$defs"][definition_name]["properties"]["schema_version"] == {"const": "1.0"}

    parsed = registry.get_schema(expected[4])
    assert parsed["properties"]["raw_response"] == {
        "$ref": "https://schemas.robata.dev/v1/raw-provider-response-artifact.schema.json"
    }
    assert parsed["properties"]["payload"] == {
        "$ref": "https://schemas.robata.dev/v1/provider-claim-payload.schema.json"
    }


def test_every_catalog_entry_has_exact_digest_and_deterministic_id() -> None:
    registry = SchemaRegistry()

    assert len(registry.entries) == 28
    assert registry.upcasters == ()
    for registered in registry.entries:
        digest = hashlib.sha256(registered.document_bytes).hexdigest()
        assert digest == registered.ref.sha256
        assert registered.ref.artifact_id == deterministic_schema_artifact_id(digest)
        assert registered.entry.compatibility_mode == "NONE"
        assert registered.entry.supported_predecessors == ()


def test_exact_lookup_rejects_a_mismatched_pin() -> None:
    registry = SchemaRegistry()
    expected = registry.resolve_version(SCHEMA_ID, "2.0.0").ref
    mismatched = expected.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(SchemaPinMismatchError):
        registry.resolve_exact(mismatched)


def test_returned_schema_is_isolated_and_multiversion_alias_is_ambiguous() -> None:
    registry = SchemaRegistry()
    ref = registry.resolve_version(SCHEMA_ID, "2.0.0").ref
    first = registry.get_schema(ref)
    first["title"] = "mutated"

    assert registry.get_schema(ref).get("title") != "mutated"
    with pytest.raises(SchemaAmbiguityError, match="ambiguous schema alias"):
        registry.resolve_alias("camera-video-export-manifest")


def test_catalog_rejects_digest_tampering(tmp_path: Path) -> None:
    catalog_path, catalog = _synthetic_catalog(tmp_path)
    catalog["schemas"][0]["ref"]["sha256"] = "0" * 64
    _write_json(catalog_path, catalog)

    with pytest.raises(SchemaDefinitionError, match="SHA-256 mismatch"):
        SchemaRegistry(catalog_path)


def test_catalog_rejects_path_escape(tmp_path: Path) -> None:
    catalog_path, catalog = _synthetic_catalog(tmp_path)
    catalog["schemas"][0]["artifact_path"] = "../synthetic.schema.json"
    _write_json(catalog_path, catalog)

    with pytest.raises(SchemaDefinitionError, match="unsafe path segment"):
        SchemaRegistry(catalog_path)


def test_catalog_is_closed_to_unknown_fields(tmp_path: Path) -> None:
    catalog_path, catalog = _synthetic_catalog(tmp_path)
    catalog["unknown"] = True
    _write_json(catalog_path, catalog)

    with pytest.raises(SchemaDefinitionError, match="Additional properties"):
        SchemaRegistry(catalog_path)


def test_catalog_rejects_uncataloged_recursive_schema(tmp_path: Path) -> None:
    catalog_path, _ = _synthetic_catalog(tmp_path)
    extra = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.robata.dev/v2/extra.schema.json",
        "type": "null",
    }
    _write_json(tmp_path / "v2" / "nested" / "extra.schema.json", extra)

    with pytest.raises(SchemaDefinitionError, match="uncataloged schema documents"):
        SchemaRegistry(catalog_path)


def test_catalog_rejects_duplicate_logical_version(tmp_path: Path) -> None:
    catalog_path, catalog = _synthetic_catalog(tmp_path)
    catalog["schemas"].append(catalog["schemas"][0])
    _write_json(catalog_path, catalog)

    with pytest.raises(SchemaDefinitionError, match="duplicate schema version"):
        SchemaRegistry(catalog_path)


def test_catalog_requires_strict_semver(tmp_path: Path) -> None:
    catalog_path, catalog = _synthetic_catalog(tmp_path)
    catalog["schemas"][0]["ref"]["version"] = "1.0"
    _write_json(catalog_path, catalog)

    with pytest.raises(SchemaDefinitionError, match="catalog validation failed"):
        SchemaRegistry(catalog_path)
