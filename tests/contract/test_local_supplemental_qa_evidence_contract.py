from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from robata.application.canonical.supplemental_qa_evidence import (
    load_registered_local_supplemental_qa_evidence_document,
    parse_local_supplemental_qa_evidence_document,
    publish_registered_local_supplemental_qa_evidence_document,
    registered_local_supplemental_qa_evidence_document,
    validate_registered_local_supplemental_qa_evidence_document,
)
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import (
    SchemaPinMismatchError,
    SchemaRegistry,
    SchemaValidationError,
)
from robata.qa_pipeline.supplemental import DeterministicSupplementalQaDenseConsumer
from robata.qa_pipeline.supplemental_wire import (
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
)
from tests.unit.test_supplemental_temporal_package import (
    _artifact_bytes_resolver,
    _materialized,
    _plan,
)


def _document(registry: SchemaRegistry, *, alias: int = 0) -> dict[str, object]:
    plan = _plan()
    materialized = _materialized(plan=plan, alias=alias)
    consumer = DeterministicSupplementalQaDenseConsumer()
    input_plan = consumer.prepare(materialized)
    result = consumer.consume(
        materialized,
        input_plan,
        artifact_bytes_resolver=_artifact_bytes_resolver,
    )
    return registered_local_supplemental_qa_evidence_document(
        plan,
        materialized,
        input_plan,
        result,
        registry,
    )


def test_envelope_validates_exact_pin_and_complete_chain() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
    )
    document = _document(registry)

    assert document["schema_version"] == "2.0"
    assert document["schema_ref"] == registered.ref.model_dump(mode="json")
    assert document["evidence_class"] == "LOCAL_CONFORMANCE"
    assert document["production_eligible"] is False
    assert document["package_manifest_sha256"] == document["input_plan"]["package_manifest_sha256"]
    assert document["package_manifest_sha256"] == document["result"]["package_manifest_sha256"]
    assert (
        validate_registered_local_supplemental_qa_evidence_document(document, registry) is document
    )
    assert registry.validate_pinned(registered.ref, document) is document


def test_envelope_schema_is_closed_and_registered_without_predecessors() -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
    )
    schema = registry.get_schema(registered.ref)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["schema_version"]["const"] == "2.0"
    assert schema["properties"]["evidence_class"]["const"] == "LOCAL_CONFORMANCE"
    assert schema["properties"]["production_eligible"]["const"] is False
    assert registered.entry.owner == "robata-qa"
    assert registered.entry.projection_version == ("local-supplemental-qa-evidence-semantic-v2")
    assert registered.entry.compatibility_mode.value == "NONE"
    assert registered.entry.supported_predecessors == ()


def test_exact_loader_replays_and_rejects_noncanonical_or_duplicate_json(
    tmp_path: Path,
) -> None:
    registry = SchemaRegistry()
    document = _document(registry)
    replay = _document(registry)
    assert replay == document

    path = tmp_path / "local-supplemental-qa-evidence.json"
    exact_bytes = canonical_json_bytes(document)
    path.write_bytes(exact_bytes)
    assert load_registered_local_supplemental_qa_evidence_document(path, registry) == document

    path.write_bytes(exact_bytes + b"\n")
    with pytest.raises(ValueError, match="exact canonical JSON"):
        load_registered_local_supplemental_qa_evidence_document(path, registry)

    path.write_bytes(b'{"schema_version":"2.0",' + exact_bytes[1:])
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_registered_local_supplemental_qa_evidence_document(path, registry)


def test_envelope_rejects_forged_pin_and_structural_drift() -> None:
    registry = SchemaRegistry()
    document = _document(registry)

    forged = deepcopy(document)
    forged["schema_ref"]["sha256"] = "0" * 64
    with pytest.raises(SchemaPinMismatchError):
        validate_registered_local_supplemental_qa_evidence_document(forged, registry)

    unknown = deepcopy(document)
    unknown["unexpected"] = True
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_registered_local_supplemental_qa_evidence_document(unknown, registry)


def test_envelope_rejects_top_level_and_nested_semantic_tampering() -> None:
    registry = SchemaRegistry()
    document = _document(registry)

    top_level = deepcopy(document)
    top_level["semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantic_sha256"):
        validate_registered_local_supplemental_qa_evidence_document(top_level, registry)

    nested = deepcopy(document)
    nested["result"]["consumptions"][0]["effective_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantic_sha256"):
        validate_registered_local_supplemental_qa_evidence_document(nested, registry)

    exact_manifest = deepcopy(document)
    exact_manifest["package_manifest_sha256"] = "0" * 64
    exact_manifest["input_plan"]["package_manifest_sha256"] = "0" * 64
    exact_manifest["result"]["package_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact package bytes"):
        validate_registered_local_supplemental_qa_evidence_document(exact_manifest, registry)


def test_v1_pin_remains_immutable_while_v2_is_default() -> None:
    registered = SchemaRegistry().resolve_version(
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
        "1.0.0",
    )

    assert registered.ref.artifact_id == "665afe04-1011-4989-e1bd-128e1d896e21"
    assert registered.ref.sha256 == (
        "1f54a216b700d13ac20f70007560aa3a1b8facdcbd79b9dc3ba9e2f311b830f3"
    )


def test_v2_uses_decimal_nanoseconds_and_replayable_source_binding() -> None:
    registry = SchemaRegistry()
    document = _document(registry)
    parsed = parse_local_supplemental_qa_evidence_document(document, registry)

    target = document["frozen_plan"]["targets"][0]
    outcome = document["package"]["outcomes"][0]
    consumption = document["result"]["consumptions"][0]
    assert isinstance(target["target_ns"], str)
    assert isinstance(document["frozen_plan"]["selection_tolerance_ns"], str)
    assert isinstance(outcome["source_frame"]["aligned_timestamp_ns"], str)
    assert isinstance(consumption["target_ns"], str)
    assert parsed.frozen_plan.source_binding.semantic_sha256


def test_top_semantic_identity_excludes_exact_package_manifest() -> None:
    registry = SchemaRegistry()
    first = _document(registry, alias=0)
    second = _document(registry, alias=1)

    assert first["package_manifest_sha256"] != second["package_manifest_sha256"]
    assert first["semantic_sha256"] == second["semantic_sha256"]


def test_atomic_publish_is_idempotent_and_rejects_different_bytes(tmp_path: Path) -> None:
    registry = SchemaRegistry()
    first = _document(registry, alias=0)
    second = _document(registry, alias=1)
    path = tmp_path / "published.json"

    published = publish_registered_local_supplemental_qa_evidence_document(
        path,
        first,
        registry,
    )
    replay = publish_registered_local_supplemental_qa_evidence_document(
        path,
        first,
        registry,
    )
    assert replay == published
    assert path.read_bytes() == canonical_json_bytes(first)
    assert not tuple(tmp_path.glob("*.tmp"))

    with pytest.raises(ValueError, match="bytes are inconsistent"):
        publish_registered_local_supplemental_qa_evidence_document(
            path,
            second,
            registry,
        )
