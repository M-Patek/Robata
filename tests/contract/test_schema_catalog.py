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


def test_every_catalog_entry_has_exact_digest_and_deterministic_id() -> None:
    registry = SchemaRegistry()

    assert len(registry.entries) == 21
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
