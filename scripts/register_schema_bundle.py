"""Atomically register an independent bundle of immutable JSON Schemas.

The manifest describes only previously independent schema publications. Schema
evolution and upcaster edges remain the responsibility of
``register_schema_evolution.py``. Every candidate is normalized to deterministic
UTF-8/LF JSON, the complete proposed catalog is validated in a temporary schema
tree, and the catalog replacement is the sole publication commit point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.schema_registry import (  # noqa: E402
    DEFAULT_SCHEMA_CATALOG,
    SCHEMA_PUBLICATION_LOCK_FILENAME,
    SCHEMA_PUBLICATION_MARKER_FILENAME,
    SchemaRef,
    SchemaRegistry,
    SchemaRegistryError,
    deterministic_schema_artifact_id,
)
from scripts.register_schema import (  # noqa: E402
    PublicationArtifact,
    PublishedSchemaConflictError,
    SchemaRegistrationError,
    _catalog_bytes,
    _contained_artifact_target,
    _entry,
    _load_candidate,
    _publish_artifacts,
    _recover_publication_marker,
    _registration_lock,
    _validation_snapshot_parent,
)

_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_VERSION_DIRECTORY = re.compile(r"^v[0-9]+$")


class SchemaBundleRegistrationError(SchemaRegistrationError):
    """Raised when a schema bundle is invalid or internally ambiguous."""


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SoftwareRangeInput(_StrictInput):
    min_inclusive: str
    max_exclusive: str


class SchemaCandidateInput(_StrictInput):
    candidate: str
    schema_id: str
    version: str
    wire_version: str
    artifact_path: str
    owner: str
    projection_version: str
    canonicalization_version: str
    software: SoftwareRangeInput


class SchemaRegistrationBundle(_StrictInput):
    format_version: Literal["1.0"]
    schemas: tuple[SchemaCandidateInput, ...]


@dataclass(frozen=True, slots=True)
class SchemaBundleItemResult:
    ref: SchemaRef
    artifact_path: str
    changed: bool


@dataclass(frozen=True, slots=True)
class SchemaBundleRegistrationResult:
    items: tuple[SchemaBundleItemResult, ...]
    changed: bool
    dry_run: bool

    @property
    def refs(self) -> tuple[SchemaRef, ...]:
        return tuple(item.ref for item in self.items)

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return tuple(item.artifact_path for item in self.items)


@dataclass(frozen=True, slots=True)
class _PreparedBundle:
    result: SchemaBundleRegistrationResult
    original_catalog: bytes
    new_catalog: bytes
    artifacts: tuple[PublicationArtifact, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaBundleRegistrationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise SchemaBundleRegistrationError(f"non-JSON numeric constant: {value}")


def _decode_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except SchemaBundleRegistrationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaBundleRegistrationError(f"invalid JSON '{source}': {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaBundleRegistrationError(f"JSON root must be an object: {source}")
    return value


def _load_bundle(path: Path) -> SchemaRegistrationBundle:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SchemaBundleRegistrationError(f"cannot read schema bundle '{path}': {exc}") from exc
    _decode_json_object(raw, source=str(path))
    try:
        bundle = SchemaRegistrationBundle.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        raise SchemaBundleRegistrationError(f"invalid schema bundle '{path}': {exc}") from exc
    if not bundle.schemas:
        raise SchemaBundleRegistrationError("schema bundle must contain at least one schema")
    return bundle


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\\" in value or Path(value).is_absolute():
        raise SchemaBundleRegistrationError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} or _PATH_SEGMENT.fullmatch(part) is None for part in path.parts):
        raise SchemaBundleRegistrationError(f"{label} contains an unsafe path segment: {value}")
    return path


def _candidate_path(bundle_root: Path, value: str, *, label: str) -> Path:
    relative = _safe_relative_path(value, label=label)
    candidate = bundle_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SchemaBundleRegistrationError(f"cannot resolve {label} '{value}': {exc}") from exc
    if not resolved.is_file() or not resolved.is_relative_to(bundle_root):
        raise SchemaBundleRegistrationError(f"{label} must resolve to a file below the bundle root")
    return resolved


def _publication_path(schema_root: Path, value: str, *, label: str) -> str:
    path = _safe_relative_path(value, label=label)
    if len(path.parts) < 2 or _VERSION_DIRECTORY.fullmatch(path.parts[0]) is None:
        raise SchemaBundleRegistrationError(f"{label} must live below a v[0-9]+ directory")
    if not path.name.endswith(".schema.json"):
        raise SchemaBundleRegistrationError(f"{label} must end in .schema.json")
    _contained_artifact_target(schema_root, path.as_posix())
    return path.as_posix()


def _assert_unique_inputs(bundle: SchemaRegistrationBundle) -> None:
    keys = tuple((item.schema_id, item.version) for item in bundle.schemas)
    paths = tuple(item.artifact_path for item in bundle.schemas)
    candidates = tuple(item.candidate for item in bundle.schemas)
    checks = (
        (keys, "logical schema versions"),
        (paths, "artifact paths"),
        (candidates, "candidate paths"),
    )
    for values, label in checks:
        if len(set(values)) != len(values):
            raise SchemaBundleRegistrationError(f"schema bundle {label} must be unique")


def _validated_snapshot(
    catalog_path: Path,
    *,
    catalog_bytes: bytes,
    artifacts: tuple[PublicationArtifact, ...],
) -> SchemaRegistry:
    schema_root = catalog_path.parent
    with tempfile.TemporaryDirectory(
        prefix=".schema-bundle-registration-",
        dir=_validation_snapshot_parent(schema_root),
    ) as temporary:
        snapshot_root = Path(temporary) / "schemas"
        shutil.copytree(
            schema_root,
            snapshot_root,
            ignore=shutil.ignore_patterns(
                SCHEMA_PUBLICATION_LOCK_FILENAME,
                SCHEMA_PUBLICATION_MARKER_FILENAME,
            ),
        )
        for artifact in artifacts:
            destination = _contained_artifact_target(snapshot_root, artifact.artifact_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(artifact.contents)
        snapshot_catalog = snapshot_root / catalog_path.name
        snapshot_catalog.write_bytes(catalog_bytes)
        registry = SchemaRegistry(snapshot_catalog)
        registry.validate_schema_documents()
        return registry


def _prepare_bundle(
    bundle_path: Path,
    *,
    catalog_path: Path,
    dry_run: bool,
) -> _PreparedBundle:
    bundle = _load_bundle(bundle_path)
    _assert_unique_inputs(bundle)
    bundle_root = bundle_path.parent.resolve()
    schema_root = catalog_path.parent
    original_catalog = catalog_path.read_bytes()
    catalog_document = _decode_json_object(original_catalog, source=str(catalog_path))
    schemas = catalog_document.get("schemas")
    if not isinstance(schemas, list):
        raise SchemaBundleRegistrationError("schema catalog schemas must be an array")

    current = SchemaRegistry(catalog_path, _skip_publication_lock=True)
    current_by_key = {registered.ref.key: registered for registered in current.entries}
    current_by_path = {registered.entry.artifact_path: registered for registered in current.entries}
    prepared: list[tuple[SchemaCandidateInput, SchemaRef, str, bytes, dict[str, Any], bool]] = []
    for position, candidate_input in enumerate(bundle.schemas):
        label = f"schemas[{position}]"
        relative_artifact = _publication_path(
            schema_root,
            candidate_input.artifact_path,
            label=f"{label}.artifact_path",
        )
        candidate = _candidate_path(
            bundle_root,
            candidate_input.candidate,
            label=f"{label}.candidate",
        )
        _original, document, normalized = _load_candidate(candidate)
        document_id = document.get("$id")
        expected_document_id = f"https://schemas.robata.dev/{relative_artifact}"
        if document_id != expected_document_id:
            raise SchemaBundleRegistrationError(
                f"{label} schema $id must match its catalog artifact path"
            )
        digest = hashlib.sha256(normalized).hexdigest()
        try:
            ref = SchemaRef(
                schema_id=candidate_input.schema_id,
                version=candidate_input.version,
                artifact_id=deterministic_schema_artifact_id(digest),
                sha256=digest,
            )
        except ValidationError as exc:
            raise SchemaBundleRegistrationError(f"invalid {label} schema reference: {exc}") from exc
        proposed_entry = _entry(
            schema_id=ref.schema_id,
            version=ref.version,
            artifact_id=ref.artifact_id,
            digest=ref.sha256,
            wire_version=candidate_input.wire_version,
            document_id=expected_document_id,
            artifact_path=relative_artifact,
            owner=candidate_input.owner,
            canonicalization_version=candidate_input.canonicalization_version,
            projection_version=candidate_input.projection_version,
            software_min=candidate_input.software.min_inclusive,
            software_max=candidate_input.software.max_exclusive,
        )

        published = current_by_key.get(ref.key)
        if published is not None:
            if published.document_bytes != normalized:
                raise PublishedSchemaConflictError(
                    f"published schema {ref.schema_id}@{ref.version} has different exact bytes"
                )
            if published.entry.model_dump(mode="json") != proposed_entry:
                raise PublishedSchemaConflictError(
                    f"published schema {ref.schema_id}@{ref.version} has different metadata"
                )
            changed = False
        else:
            path_owner = current_by_path.get(relative_artifact)
            if path_owner is not None:
                raise PublishedSchemaConflictError(
                    "artifact path is already published for "
                    f"{path_owner.ref.schema_id}@{path_owner.ref.version}: {relative_artifact}"
                )
            changed = True
        prepared.append(
            (candidate_input, ref, relative_artifact, normalized, proposed_entry, changed)
        )

    prepared.sort(key=lambda item: (item[1].schema_id, item[1].version, item[2]))
    artifacts = tuple(
        PublicationArtifact(artifact_path=item[2], contents=item[3], role="json-schema")
        for item in prepared
    )
    new_entries = [item[4] for item in prepared if item[5]]
    schemas.extend(new_entries)
    new_catalog = _catalog_bytes(catalog_document) if new_entries else original_catalog
    snapshot = _validated_snapshot(
        catalog_path,
        catalog_bytes=new_catalog,
        artifacts=artifacts,
    )

    results: list[SchemaBundleItemResult] = []
    for _candidate_input, ref, artifact_path, _normalized, _entry_value, changed in prepared:
        registered = snapshot.resolve_exact(ref, require_software_support=False)
        results.append(
            SchemaBundleItemResult(
                ref=registered.ref,
                artifact_path=artifact_path,
                changed=changed,
            )
        )
    return _PreparedBundle(
        result=SchemaBundleRegistrationResult(
            items=tuple(results),
            changed=bool(new_entries),
            dry_run=dry_run,
        ),
        original_catalog=original_catalog,
        new_catalog=new_catalog,
        artifacts=artifacts,
    )


def register_schema_bundle(
    bundle_path: str | Path,
    *,
    catalog_path: str | Path = DEFAULT_SCHEMA_CATALOG,
    dry_run: bool = False,
) -> SchemaBundleRegistrationResult:
    """Validate and atomically publish one independent multi-schema bundle."""

    bundle = Path(bundle_path).resolve()
    catalog = Path(catalog_path).resolve()
    with _registration_lock(catalog.parent):
        _recover_publication_marker(catalog)
        prepared = _prepare_bundle(bundle, catalog_path=catalog, dry_run=dry_run)
        if not prepared.result.changed or dry_run:
            return prepared.result
        _publish_artifacts(
            catalog_path=catalog,
            original_catalog=prepared.original_catalog,
            new_catalog=prepared.new_catalog,
            artifacts=prepared.artifacts,
        )
        return prepared.result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_SCHEMA_CATALOG)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = register_schema_bundle(
            args.bundle,
            catalog_path=args.catalog,
            dry_run=args.dry_run,
        )
    except (OSError, SchemaRegistrationError, SchemaRegistryError) as exc:
        print(f"schema bundle registration failed: {exc}", file=sys.stderr)
        return 1
    status = "DRY-RUN" if result.dry_run else ("REGISTERED" if result.changed else "UNCHANGED")
    refs = ",".join(f"{ref.schema_id}@{ref.version}" for ref in result.refs)
    print(f"{status} schemas={refs} artifacts={len(result.artifact_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SchemaBundleItemResult",
    "SchemaBundleRegistrationError",
    "SchemaBundleRegistrationResult",
    "main",
    "register_schema_bundle",
]
