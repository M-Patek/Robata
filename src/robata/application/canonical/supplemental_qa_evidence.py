"""Registered document boundary for persisted local supplemental QA evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.qa_pipeline.supplemental import (
    SupplementalQaDenseInputPlan,
    SupplementalQaDenseResult,
)
from robata.qa_pipeline.supplemental_wire import (
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_PROJECTION_VERSION,
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
    LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
    LocalSupplementalQaEvidence,
    local_supplemental_qa_evidence_projection,
)
from robata.sampling.supplemental import (
    FrozenSupplementalTargetPlan,
    MaterializedSupplementalPackage,
    SupplementalEvidenceClass,
)
from robata.tempfiles import make_temp_file


def registered_local_supplemental_qa_evidence_document(
    frozen_plan: FrozenSupplementalTargetPlan,
    materialized: MaterializedSupplementalPackage,
    input_plan: SupplementalQaDenseInputPlan,
    result: SupplementalQaDenseResult,
    registry: SchemaRegistry,
) -> dict[str, object]:
    """Create and validate one exact-pinned local supplemental QA envelope."""

    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")
    registered = registry.resolve_version(
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID,
        LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION,
    )
    values: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": registered.ref,
        "frozen_plan": frozen_plan,
        "package": materialized.package,
        "package_manifest_sha256": materialized.manifest_sha256,
        "input_plan": input_plan,
        "result": result,
        "projection_version": LOCAL_SUPPLEMENTAL_QA_EVIDENCE_PROJECTION_VERSION,
        "evidence_class": SupplementalEvidenceClass.LOCAL_CONFORMANCE,
        "production_eligible": False,
    }
    draft = LocalSupplementalQaEvidence.model_construct(
        semantic_sha256="0" * 64,
        **values,
    )
    evidence = LocalSupplementalQaEvidence.model_validate(
        {
            **values,
            "semantic_sha256": semantic_sha256(local_supplemental_qa_evidence_projection(draft)),
        },
        strict=True,
    )
    document: dict[str, object] = evidence.model_dump(mode="json")
    validate_registered_local_supplemental_qa_evidence_document(document, registry)
    return document


def validate_registered_local_supplemental_qa_evidence_document(
    document: Mapping[str, object],
    registry: SchemaRegistry,
) -> Mapping[str, object]:
    """Validate the embedded exact pin, closed Wire shape, and full evidence chain."""

    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")
    schema_ref = SchemaRef.model_validate(document.get("schema_ref"), strict=True)
    if (
        schema_ref.schema_id != LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID
        or schema_ref.version != LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("schema_ref must identify local-supplemental-qa-evidence@2.0.0")
    registry.validate_pinned(schema_ref, document)
    evidence = _parse_local_supplemental_qa_evidence_document(document)
    registry.resolve_exact(evidence.frozen_plan.source_report_schema_ref)
    return document


def _parse_local_supplemental_qa_evidence_document(
    document: Mapping[str, object],
) -> LocalSupplementalQaEvidence:
    return LocalSupplementalQaEvidence.model_validate_json(canonical_json_bytes(document))


def parse_local_supplemental_qa_evidence_document(
    document: Mapping[str, object],
    registry: SchemaRegistry | None = None,
) -> LocalSupplementalQaEvidence:
    """Validate a registered document and return its strict typed envelope."""

    active_registry = registry or SchemaRegistry()
    validate_registered_local_supplemental_qa_evidence_document(document, active_registry)
    return _parse_local_supplemental_qa_evidence_document(document)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def load_registered_local_supplemental_qa_evidence_document(
    path: Path,
    registry: SchemaRegistry,
) -> dict[str, object]:
    """Load exact canonical JSON and validate its complete registered evidence chain."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")
    raw = path.read_bytes()
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid supplemental QA evidence JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("supplemental QA evidence root must be an object")
    if canonical_json_bytes(parsed) != raw:
        raise ValueError("supplemental QA evidence bytes are not exact canonical JSON")
    document: dict[str, object] = parsed
    validate_registered_local_supplemental_qa_evidence_document(document, registry)
    return document


def publish_registered_local_supplemental_qa_evidence_document(
    path: Path,
    document: Mapping[str, object],
    registry: SchemaRegistry,
) -> Path:
    """Atomically publish exact bytes, reusing only an identical regular file."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    validate_registered_local_supplemental_qa_evidence_document(document, registry)
    contents = canonical_json_bytes(document)
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("supplemental QA evidence parent must be a regular directory")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise ValueError("existing supplemental QA evidence is not a regular file")
        if target.read_bytes() != contents:
            raise ValueError("existing supplemental QA evidence bytes are inconsistent")
        return target.resolve()

    descriptor, temporary = make_temp_file(
        target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != contents:
                raise ValueError(
                    "concurrent supplemental QA evidence bytes are inconsistent"
                ) from None
        return target.resolve()
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "load_registered_local_supplemental_qa_evidence_document",
    "parse_local_supplemental_qa_evidence_document",
    "publish_registered_local_supplemental_qa_evidence_document",
    "registered_local_supplemental_qa_evidence_document",
    "validate_registered_local_supplemental_qa_evidence_document",
]
