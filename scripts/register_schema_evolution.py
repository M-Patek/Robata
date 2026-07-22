"""Atomically register one new schema target and its incoming upcasters.

The bundle is deliberately narrower than a general graph editor: it publishes one
previously unknown target version and one or more direct incoming edges. Existing
schema and upcaster catalog entries are never rewritten.
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
from robata.contracts.schema_upcasting import (  # noqa: E402
    SchemaUpcasterGraph,
    SchemaUpcastingError,
)
from scripts.register_schema import (  # noqa: E402
    PublicationArtifact,
    PublishedSchemaConflictError,
    SchemaRegistrationError,
    _catalog_bytes,
    _contained_artifact_target,
    _load_candidate,
    _publish_artifacts,
    _recover_publication_marker,
    _registration_lock,
    _validation_snapshot_parent,
)

_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_VERSION_DIRECTORY = re.compile(r"^v[0-9]+$")
_UPCASTER_ARTIFACT_ID_DOMAIN = b"robata-schema-upcaster-artifact-id-v1\x00"
_RUNTIME_DENIAL_FLAGS = ("clock", "database", "network", "randomness")


class SchemaEvolutionRegistrationError(SchemaRegistrationError):
    """Raised when a bundle cannot be published without changing existing facts."""


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SoftwareRangeInput(_StrictInput):
    min_inclusive: str
    max_exclusive: str


class TargetSchemaInput(_StrictInput):
    candidate: str
    schema_id: str
    version: str
    wire_version: str
    artifact_path: str
    owner: str
    projection_version: str
    canonicalization_version: str
    software: SoftwareRangeInput


class ArtifactInput(_StrictInput):
    candidate: str
    artifact_path: str


class GoldenVectorInput(_StrictInput):
    input: ArtifactInput
    output: ArtifactInput


class IncomingUpcasterInput(_StrictInput):
    upcaster_id: str
    source: SchemaRef
    code: ArtifactInput
    runtime: ArtifactInput
    golden_vectors: tuple[GoldenVectorInput, ...]


class SchemaEvolutionBundle(_StrictInput):
    format_version: Literal["1.0"]
    target: TargetSchemaInput
    upcasters: tuple[IncomingUpcasterInput, ...]


@dataclass(frozen=True, slots=True)
class SchemaEvolutionRegistrationResult:
    target_ref: SchemaRef
    upcaster_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    changed: bool
    dry_run: bool


@dataclass(frozen=True, slots=True)
class _PreparedEvolution:
    result: SchemaEvolutionRegistrationResult
    original_catalog: bytes
    new_catalog: bytes
    artifacts: tuple[PublicationArtifact, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaEvolutionRegistrationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise SchemaEvolutionRegistrationError(f"non-JSON numeric constant: {value}")


def _decode_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except SchemaEvolutionRegistrationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaEvolutionRegistrationError(f"invalid JSON '{source}': {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaEvolutionRegistrationError(f"JSON root must be an object: {source}")
    return value


def _deterministic_json_bytes(value: dict[str, Any], *, source: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaEvolutionRegistrationError(
            f"JSON candidate is not deterministic: {source}: {exc}"
        ) from exc


def _load_bundle(path: Path) -> SchemaEvolutionBundle:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SchemaEvolutionRegistrationError(
            f"cannot read evolution bundle '{path}': {exc}"
        ) from exc
    _decode_json_object(raw, source=str(path))
    try:
        return SchemaEvolutionBundle.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        raise SchemaEvolutionRegistrationError(f"invalid evolution bundle '{path}': {exc}") from exc


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\\" in value or Path(value).is_absolute():
        raise SchemaEvolutionRegistrationError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} or _PATH_SEGMENT.fullmatch(part) is None for part in path.parts):
        raise SchemaEvolutionRegistrationError(f"{label} contains an unsafe path segment: {value}")
    return path


def _candidate_path(bundle_root: Path, value: str, *, label: str) -> Path:
    relative = _safe_relative_path(value, label=label)
    candidate = bundle_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SchemaEvolutionRegistrationError(f"cannot resolve {label} '{value}': {exc}") from exc
    if not resolved.is_file() or not resolved.is_relative_to(bundle_root):
        raise SchemaEvolutionRegistrationError(
            f"{label} must resolve to a file below the bundle root"
        )
    return resolved


def _publication_path(
    schema_root: Path,
    value: str,
    *,
    label: str,
    suffix: str,
) -> str:
    path = _safe_relative_path(value, label=label)
    if not path.name.endswith(suffix):
        raise SchemaEvolutionRegistrationError(f"{label} must end in {suffix}")
    _contained_artifact_target(schema_root, path.as_posix())
    return path.as_posix()


def _schema_publication_path(schema_root: Path, value: str) -> str:
    path = _safe_relative_path(value, label="target.artifact_path")
    if len(path.parts) < 2 or _VERSION_DIRECTORY.fullmatch(path.parts[0]) is None:
        raise SchemaEvolutionRegistrationError(
            "target.artifact_path must live below a v[0-9]+ directory"
        )
    if not path.name.endswith(".schema.json"):
        raise SchemaEvolutionRegistrationError("target.artifact_path must end in .schema.json")
    _contained_artifact_target(schema_root, path.as_posix())
    return path.as_posix()


def _normalize_python(path: Path) -> bytes:
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise SchemaEvolutionRegistrationError(f"invalid Python candidate '{path}': {exc}") from exc
    if text.startswith("\ufeff"):
        raise SchemaEvolutionRegistrationError(f"Python candidate must not contain a BOM: {path}")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise SchemaEvolutionRegistrationError(f"Python candidate must not be empty: {path}")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized.encode("utf-8")


def _normalize_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SchemaEvolutionRegistrationError(
            f"cannot read JSON candidate '{path}': {exc}"
        ) from exc
    document = _decode_json_object(raw, source=str(path))
    return document, _deterministic_json_bytes(document, source=str(path))


def deterministic_upcaster_artifact_id(role: Literal["CODE", "RUNTIME"], digest: str) -> str:
    """Derive a stable UUID-shaped identity for immutable upcaster artifacts."""

    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("digest must be a lowercase SHA-256 digest")
    identity = hashlib.sha256(
        _UPCASTER_ARTIFACT_ID_DOMAIN + role.encode("ascii") + b"\x00" + digest.encode("ascii")
    ).hexdigest()[:32]
    return f"{identity[:8]}-{identity[8:12]}-{identity[12:16]}-{identity[16:20]}-{identity[20:]}"


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _version_key(value: str) -> tuple[int, int, int]:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value) is None:
        raise SchemaEvolutionRegistrationError(f"invalid semantic version: {value}")
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def _schema_entry(
    target: TargetSchemaInput,
    *,
    document_id: str,
    artifact_path: str,
    target_ref: SchemaRef,
    predecessors: tuple[SchemaRef, ...],
) -> dict[str, Any]:
    return {
        "ref": target_ref.model_dump(mode="json"),
        "wire_version": target.wire_version,
        "document_id": document_id,
        "artifact_path": artifact_path,
        "owner": target.owner,
        "canonicalization_version": target.canonicalization_version,
        "projection_version": target.projection_version,
        "compatibility_mode": "BACKWARD",
        "lifecycle": "ACTIVE",
        "supported_software": {
            "min_inclusive": target.software.min_inclusive,
            "max_exclusive": target.software.max_exclusive,
        },
        "supported_predecessors": [item.model_dump(mode="json") for item in predecessors],
    }


def _add_artifact(
    artifacts: dict[str, PublicationArtifact],
    *,
    artifact_path: str,
    contents: bytes,
    role: str,
) -> None:
    candidate = PublicationArtifact(artifact_path=artifact_path, contents=contents, role=role)
    existing = artifacts.get(artifact_path)
    if existing is None:
        artifacts[artifact_path] = candidate
        return
    if existing.contents != contents:
        raise SchemaEvolutionRegistrationError(
            f"publication artifact path has conflicting candidates: {artifact_path}"
        )


def _assert_destination_compatible(schema_root: Path, artifact: PublicationArtifact) -> None:
    destination = _contained_artifact_target(schema_root, artifact.artifact_path)
    if not destination.exists() and not destination.is_symlink():
        return
    if not destination.is_file() or destination.read_bytes() != artifact.contents:
        raise PublishedSchemaConflictError(
            f"artifact path has different exact bytes: {artifact.artifact_path}"
        )


def _validate_snapshot(
    catalog_path: Path,
    *,
    catalog_bytes: bytes,
    artifacts: tuple[PublicationArtifact, ...],
) -> None:
    schema_root = catalog_path.parent
    with tempfile.TemporaryDirectory(
        prefix=".schema-evolution-registration-",
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
        SchemaUpcasterGraph(registry)


def _prepare_evolution(
    bundle_path: Path,
    *,
    catalog_path: Path,
    dry_run: bool,
) -> _PreparedEvolution:
    bundle = _load_bundle(bundle_path)
    bundle_root = bundle_path.parent.resolve()
    schema_root = catalog_path.parent
    original_catalog = catalog_path.read_bytes()
    catalog_document = _decode_json_object(original_catalog, source=str(catalog_path))
    schemas = catalog_document.get("schemas")
    upcasters = catalog_document.get("upcasters")
    if not isinstance(schemas, list) or not isinstance(upcasters, list):
        raise SchemaEvolutionRegistrationError("schema catalog must contain schema/upcaster arrays")

    current = SchemaRegistry(catalog_path, _skip_publication_lock=True)
    target_artifact_path = _schema_publication_path(
        schema_root,
        bundle.target.artifact_path,
    )
    target_candidate = _candidate_path(
        bundle_root,
        bundle.target.candidate,
        label="target.candidate",
    )
    _original_schema, target_document, target_bytes = _load_candidate(target_candidate)
    document_id = target_document.get("$id")
    expected_document_id = f"https://schemas.robata.dev/{target_artifact_path}"
    if document_id != expected_document_id:
        raise SchemaEvolutionRegistrationError(
            "target schema $id must match its publication artifact path"
        )
    target_digest = _sha256(target_bytes)
    try:
        target_ref = SchemaRef(
            schema_id=bundle.target.schema_id,
            version=bundle.target.version,
            artifact_id=deterministic_schema_artifact_id(target_digest),
            sha256=target_digest,
        )
    except ValidationError as exc:
        raise SchemaEvolutionRegistrationError(f"invalid target schema reference: {exc}") from exc

    if not bundle.upcasters:
        raise SchemaEvolutionRegistrationError("at least one incoming upcaster is required")
    upcaster_ids = [item.upcaster_id for item in bundle.upcasters]
    if len(set(upcaster_ids)) != len(upcaster_ids):
        raise SchemaEvolutionRegistrationError("incoming upcaster IDs must be unique")
    sources = [item.source for item in bundle.upcasters]
    if len(set(sources)) != len(sources):
        raise SchemaEvolutionRegistrationError("incoming upcaster sources must be unique")
    for source in sources:
        current.resolve_exact(source, require_software_support=False)
        if source.schema_id != target_ref.schema_id:
            raise SchemaEvolutionRegistrationError(
                "upcaster source and target must share schema_id"
            )
        if _version_key(source.version) >= _version_key(target_ref.version):
            raise SchemaEvolutionRegistrationError(
                "upcaster source version must be lower than target version"
            )
    predecessors = tuple(
        sorted(sources, key=lambda item: (item.schema_id, _version_key(item.version), item.sha256))
    )
    target_entry = _schema_entry(
        bundle.target,
        document_id=expected_document_id,
        artifact_path=target_artifact_path,
        target_ref=target_ref,
        predecessors=predecessors,
    )

    publication_artifacts: dict[str, PublicationArtifact] = {}
    _add_artifact(
        publication_artifacts,
        artifact_path=target_artifact_path,
        contents=target_bytes,
        role="SCHEMA",
    )
    proposed_upcasters: list[dict[str, Any]] = []
    for edge in sorted(bundle.upcasters, key=lambda item: item.upcaster_id):
        code_path = _publication_path(
            schema_root,
            edge.code.artifact_path,
            label=f"{edge.upcaster_id}.code.artifact_path",
            suffix=".py",
        )
        code_candidate = _candidate_path(
            bundle_root,
            edge.code.candidate,
            label=f"{edge.upcaster_id}.code.candidate",
        )
        code_bytes = _normalize_python(code_candidate)
        code_digest = _sha256(code_bytes)
        _add_artifact(
            publication_artifacts,
            artifact_path=code_path,
            contents=code_bytes,
            role="UPCASTER_CODE",
        )

        runtime_path = _publication_path(
            schema_root,
            edge.runtime.artifact_path,
            label=f"{edge.upcaster_id}.runtime.artifact_path",
            suffix=".json",
        )
        runtime_candidate = _candidate_path(
            bundle_root,
            edge.runtime.candidate,
            label=f"{edge.upcaster_id}.runtime.candidate",
        )
        runtime_document, runtime_bytes = _normalize_json(runtime_candidate)
        for flag in _RUNTIME_DENIAL_FLAGS:
            if runtime_document.get(flag) is not False:
                raise SchemaEvolutionRegistrationError(
                    f"upcaster runtime must declare {flag}=false: {edge.upcaster_id}"
                )
        runtime_digest = _sha256(runtime_bytes)
        _add_artifact(
            publication_artifacts,
            artifact_path=runtime_path,
            contents=runtime_bytes,
            role="UPCASTER_RUNTIME",
        )

        if not edge.golden_vectors:
            raise SchemaEvolutionRegistrationError(
                f"upcaster requires at least one golden vector: {edge.upcaster_id}"
            )
        golden_entries: list[dict[str, str]] = []
        for position, vector in enumerate(edge.golden_vectors):
            input_path = _publication_path(
                schema_root,
                vector.input.artifact_path,
                label=f"{edge.upcaster_id}.golden[{position}].input.artifact_path",
                suffix=".json",
            )
            input_candidate = _candidate_path(
                bundle_root,
                vector.input.candidate,
                label=f"{edge.upcaster_id}.golden[{position}].input.candidate",
            )
            _input_document, input_bytes = _normalize_json(input_candidate)
            _add_artifact(
                publication_artifacts,
                artifact_path=input_path,
                contents=input_bytes,
                role="UPCASTER_GOLDEN_INPUT",
            )

            output_path = _publication_path(
                schema_root,
                vector.output.artifact_path,
                label=f"{edge.upcaster_id}.golden[{position}].output.artifact_path",
                suffix=".json",
            )
            output_candidate = _candidate_path(
                bundle_root,
                vector.output.candidate,
                label=f"{edge.upcaster_id}.golden[{position}].output.candidate",
            )
            _output_document, output_bytes = _normalize_json(output_candidate)
            _add_artifact(
                publication_artifacts,
                artifact_path=output_path,
                contents=output_bytes,
                role="UPCASTER_GOLDEN_OUTPUT",
            )
            golden_entries.append(
                {
                    "input_artifact_path": input_path,
                    "input_sha256": _sha256(input_bytes),
                    "output_artifact_path": output_path,
                    "output_sha256": _sha256(output_bytes),
                }
            )
        golden_entries.sort(
            key=lambda item: (item["input_artifact_path"], item["output_artifact_path"])
        )
        proposed_upcasters.append(
            {
                "upcaster_id": edge.upcaster_id,
                "source": edge.source.model_dump(mode="json"),
                "target": target_ref.model_dump(mode="json"),
                "code_artifact_id": deterministic_upcaster_artifact_id("CODE", code_digest),
                "code_artifact_path": code_path,
                "code_sha256": code_digest,
                "runtime_artifact_id": deterministic_upcaster_artifact_id(
                    "RUNTIME", runtime_digest
                ),
                "runtime_artifact_path": runtime_path,
                "runtime_sha256": runtime_digest,
                "golden_vectors": golden_entries,
            }
        )

    artifacts = tuple(publication_artifacts[path] for path in sorted(publication_artifacts))
    for artifact in artifacts:
        _assert_destination_compatible(schema_root, artifact)

    existing_target = next(
        (
            item
            for item in schemas
            if isinstance(item, dict)
            and isinstance(item.get("ref"), dict)
            and (item["ref"].get("schema_id"), item["ref"].get("version")) == target_ref.key
        ),
        None,
    )
    expected_by_id = {item["upcaster_id"]: item for item in proposed_upcasters}
    existing_by_id = {
        item.get("upcaster_id"): item
        for item in upcasters
        if isinstance(item, dict) and isinstance(item.get("upcaster_id"), str)
    }
    if existing_target is not None:
        if existing_target != target_entry:
            raise PublishedSchemaConflictError(
                "published schema "
                f"{target_ref.schema_id}@{target_ref.version} has different metadata"
            )
        for upcaster_id, expected in expected_by_id.items():
            if existing_by_id.get(upcaster_id) != expected:
                raise PublishedSchemaConflictError(
                    f"published evolution bundle has different upcaster: {upcaster_id}"
                )
        SchemaUpcasterGraph(current)
        return _PreparedEvolution(
            result=SchemaEvolutionRegistrationResult(
                target_ref=target_ref,
                upcaster_ids=tuple(sorted(expected_by_id)),
                artifact_paths=tuple(item.artifact_path for item in artifacts),
                changed=False,
                dry_run=dry_run,
            ),
            original_catalog=original_catalog,
            new_catalog=original_catalog,
            artifacts=artifacts,
        )

    conflicting_ids = sorted(set(expected_by_id).intersection(existing_by_id))
    if conflicting_ids:
        raise PublishedSchemaConflictError(
            f"upcaster IDs are already published without the target bundle: {conflicting_ids!r}"
        )
    schemas.append(target_entry)
    upcasters.extend(proposed_upcasters)
    new_catalog = _catalog_bytes(catalog_document)
    _validate_snapshot(
        catalog_path,
        catalog_bytes=new_catalog,
        artifacts=artifacts,
    )
    return _PreparedEvolution(
        result=SchemaEvolutionRegistrationResult(
            target_ref=target_ref,
            upcaster_ids=tuple(sorted(expected_by_id)),
            artifact_paths=tuple(item.artifact_path for item in artifacts),
            changed=True,
            dry_run=dry_run,
        ),
        original_catalog=original_catalog,
        new_catalog=new_catalog,
        artifacts=artifacts,
    )


def register_schema_evolution(
    bundle_path: str | Path,
    *,
    catalog_path: str | Path = DEFAULT_SCHEMA_CATALOG,
    dry_run: bool = False,
) -> SchemaEvolutionRegistrationResult:
    """Validate and atomically publish one target evolution bundle."""

    bundle = Path(bundle_path).resolve()
    catalog = Path(catalog_path).resolve()
    with _registration_lock(catalog.parent):
        _recover_publication_marker(catalog)
        prepared = _prepare_evolution(bundle, catalog_path=catalog, dry_run=dry_run)
        if not prepared.result.changed or dry_run:
            return prepared.result
        _publish_artifacts(
            catalog_path=catalog,
            original_catalog=prepared.original_catalog,
            new_catalog=prepared.new_catalog,
            artifacts=prepared.artifacts,
            validate_upcasters=True,
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
        result = register_schema_evolution(
            args.bundle,
            catalog_path=args.catalog,
            dry_run=args.dry_run,
        )
    except (OSError, SchemaRegistrationError, SchemaRegistryError, SchemaUpcastingError) as exc:
        print(f"schema evolution registration failed: {exc}", file=sys.stderr)
        return 1
    status = "DRY-RUN" if result.dry_run else ("REGISTERED" if result.changed else "UNCHANGED")
    print(
        f"{status} {result.target_ref.schema_id}@{result.target_ref.version} "
        f"artifact_id={result.target_ref.artifact_id} sha256={result.target_ref.sha256} "
        f"upcasters={','.join(result.upcaster_ids)} artifacts={len(result.artifact_paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SchemaEvolutionRegistrationError",
    "SchemaEvolutionRegistrationResult",
    "deterministic_upcaster_artifact_id",
    "main",
    "register_schema_evolution",
]
