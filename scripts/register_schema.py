"""Register one immutable JSON Schema through a validated catalog snapshot.

``schema-catalog.json`` is the repository's sole machine-readable golden pin:
its exact ``SchemaRef`` four-tuple is not duplicated in a sidecar file. The
schema artifact is installed first and catalog replacement is the unique
publication commit point. Validation or handled write failures leave no
published catalog entry behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.schema_registry import (  # noqa: E402
    DEFAULT_SCHEMA_CATALOG,
    SCHEMA_PUBLICATION_LOCK_FILENAME,
    SCHEMA_PUBLICATION_MARKER_FILENAME,
    SCHEMA_PUBLICATION_MARKER_FORMAT,
    RegisteredSchema,
    SchemaRef,
    SchemaRegistry,
    SchemaRegistryError,
    deterministic_schema_artifact_id,
)


class SchemaRegistrationError(RuntimeError):
    """Raised when a schema cannot be registered without changing published facts."""


class PublishedSchemaConflictError(SchemaRegistrationError):
    """Raised when a published logical version is presented with different content."""


@dataclass(frozen=True, slots=True)
class SchemaRegistrationResult:
    ref: SchemaRef
    artifact_path: str
    changed: bool
    dry_run: bool


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaRegistrationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise SchemaRegistrationError(f"non-JSON numeric constant: {value}")


def _load_candidate(path: Path) -> tuple[bytes, dict[str, Any], bytes]:
    try:
        original = path.read_bytes()
        document = json.loads(
            original.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except SchemaRegistrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaRegistrationError(f"invalid schema candidate '{path}': {exc}") from exc
    if not isinstance(document, dict):
        raise SchemaRegistrationError("schema candidate root must be a JSON object")
    try:
        normalized = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaRegistrationError(f"schema candidate is not deterministic JSON: {exc}") from exc
    return original, document, normalized


def _catalog_bytes(catalog: dict[str, Any]) -> bytes:
    return (json.dumps(catalog, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _derive_artifact_path(document_id: str, requested: str | None) -> str:
    if requested is None:
        parsed = urlsplit(document_id)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "schemas.robata.dev"
            or parsed.query
            or parsed.fragment
        ):
            raise SchemaRegistrationError(
                "schema $id must be an unqualified https://schemas.robata.dev URL"
            )
        requested = parsed.path.removeprefix("/")

    if "\\" in requested or Path(requested).is_absolute():
        raise SchemaRegistrationError("artifact path must be a relative POSIX path")
    path = PurePosixPath(requested)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SchemaRegistrationError("artifact path contains an unsafe segment")
    if len(path.parts) < 2 or not path.parts[0].startswith("v"):
        raise SchemaRegistrationError("artifact path must live below a version directory")
    if not path.name.endswith(".schema.json"):
        raise SchemaRegistrationError("artifact path must end in .schema.json")
    return path.as_posix()


def _contained_artifact_target(schema_root: Path, artifact_path: str) -> Path:
    try:
        resolved_root = schema_root.resolve(strict=True)
    except OSError as exc:
        raise SchemaRegistrationError(f"cannot resolve schema root: {exc}") from exc
    if not resolved_root.is_dir():
        raise SchemaRegistrationError("schema root must be a directory")
    target = resolved_root.joinpath(*PurePosixPath(artifact_path).parts)
    try:
        resolved_target = target.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SchemaRegistrationError(
            f"cannot resolve artifact path: {artifact_path}: {exc}"
        ) from exc
    if not resolved_target.is_relative_to(resolved_root):
        raise SchemaRegistrationError(f"artifact path escapes schema root: {artifact_path}")

    cursor = resolved_root
    for part in PurePosixPath(artifact_path).parts:
        cursor /= part
        if not cursor.exists() and not cursor.is_symlink():
            break
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise SchemaRegistrationError(
                f"cannot inspect artifact path: {artifact_path}: {exc}"
            ) from exc
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            reparse_attribute and file_attributes & reparse_attribute
        ):
            raise SchemaRegistrationError(
                f"artifact path traverses a symlink or reparse point: {artifact_path}"
            )
    return target


def _derive_schema_id(document_id: str, artifact_path: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    name = PurePosixPath(artifact_path).name.removesuffix(".schema.json")
    return f"https://schemas.robata.dev/{name}"


def _entry(
    *,
    schema_id: str,
    version: str,
    artifact_id: str,
    digest: str,
    wire_version: str,
    document_id: str,
    artifact_path: str,
    owner: str,
    canonicalization_version: str,
    projection_version: str,
    software_min: str,
    software_max: str,
) -> dict[str, Any]:
    return {
        "ref": {
            "schema_id": schema_id,
            "version": version,
            "artifact_id": artifact_id,
            "sha256": digest,
        },
        "wire_version": wire_version,
        "document_id": document_id,
        "artifact_path": artifact_path,
        "owner": owner,
        "canonicalization_version": canonicalization_version,
        "projection_version": projection_version,
        "compatibility_mode": "NONE",
        "lifecycle": "ACTIVE",
        "supported_software": {
            "min_inclusive": software_min,
            "max_exclusive": software_max,
        },
        "supported_predecessors": [],
    }


def _validated_snapshot(
    catalog_path: Path,
    artifact_path: str,
    schema_bytes: bytes,
    catalog_bytes: bytes,
) -> SchemaRegistry:
    schema_root = catalog_path.parent
    with tempfile.TemporaryDirectory(
        prefix=".schema-registration-", dir=schema_root.parent
    ) as temporary:
        snapshot_root = Path(temporary) / "schemas"
        shutil.copytree(
            schema_root,
            snapshot_root,
            ignore=shutil.ignore_patterns(SCHEMA_PUBLICATION_LOCK_FILENAME),
        )
        snapshot_artifact = _contained_artifact_target(snapshot_root, artifact_path)
        snapshot_artifact.parent.mkdir(parents=True, exist_ok=True)
        snapshot_artifact.write_bytes(schema_bytes)
        snapshot_catalog = snapshot_root / catalog_path.name
        snapshot_catalog.write_bytes(catalog_bytes)
        registry = SchemaRegistry(snapshot_catalog)
        registry.validate_schema_documents()
        return registry


def _lock_stream(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(  # type: ignore[attr-defined]
        stream.fileno(),
        fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
    )


def _unlock_stream(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def _open_registration_lock(lock_path: Path) -> BinaryIO:
    try:
        metadata = lock_path.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise SchemaRegistrationError(f"cannot inspect schema publication lock: {exc}") from exc
    if metadata is not None:
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            reparse_attribute and file_attributes & reparse_attribute
        ):
            raise SchemaRegistrationError(
                "schema publication lock must not be a symlink or reparse point"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise SchemaRegistrationError("schema publication lock must be a regular file")

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SchemaRegistrationError(f"cannot open schema publication lock: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SchemaRegistrationError("schema publication lock must be a regular file")
        return os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _registration_lock(schema_root: Path) -> Iterator[None]:
    lock_path = schema_root / SCHEMA_PUBLICATION_LOCK_FILENAME
    stream = _open_registration_lock(lock_path)
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        try:
            _lock_stream(stream)
        except OSError as exc:
            raise SchemaRegistrationError("another schema registration is active") from exc
        locked = True
        yield
    finally:
        if locked:
            _unlock_stream(stream)
        stream.close()


def _registered_version(
    registry: SchemaRegistry,
    schema_id: str,
    version: str,
) -> RegisteredSchema | None:
    key = (schema_id, version)
    return next((registered for registered in registry.entries if registered.ref.key == key), None)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)


def _unlink_durable(path: Path, *, missing_ok: bool = False) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise
        return
    _fsync_directory(path.parent)


def _publication_marker_bytes(
    *,
    artifact_path: str,
    artifact_sha256: str,
    original_catalog: bytes,
    new_catalog: bytes,
) -> bytes:
    marker = {
        "format_version": SCHEMA_PUBLICATION_MARKER_FORMAT,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "original_catalog_sha256": hashlib.sha256(original_catalog).hexdigest(),
        "new_catalog_sha256": hashlib.sha256(new_catalog).hexdigest(),
    }
    return _catalog_bytes(marker)


def _recover_publication_marker(catalog_path: Path) -> None:
    marker_path = catalog_path.parent / SCHEMA_PUBLICATION_MARKER_FILENAME
    try:
        marker_raw = marker_path.read_bytes()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SchemaRegistrationError(f"cannot read schema publication marker: {exc}") from exc

    # Registry validation proves marker shape, path containment, digest binding, and closure.
    SchemaRegistry(catalog_path, _skip_publication_lock=True)
    try:
        marker = json.loads(
            marker_raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except SchemaRegistrationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaRegistrationError(f"invalid schema publication marker: {exc}") from exc
    if not isinstance(marker, dict):
        raise SchemaRegistrationError("schema publication marker must be an object")
    artifact_path = marker.get("artifact_path")
    original_digest = marker.get("original_catalog_sha256")
    new_digest = marker.get("new_catalog_sha256")
    if not all(isinstance(value, str) for value in (artifact_path, original_digest, new_digest)):
        raise SchemaRegistrationError("schema publication marker has invalid fields")
    target = _contained_artifact_target(catalog_path.parent, str(artifact_path))
    catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    if catalog_digest == new_digest or (catalog_digest == original_digest and not target.exists()):
        _unlink_durable(marker_path)


def _install_publication_marker(marker_path: Path, marker_bytes: bytes) -> None:
    try:
        existing = marker_path.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != marker_bytes:
            raise SchemaRegistrationError("a different schema publication transaction is pending")
        return
    staged = _write_staged(marker_path, marker_bytes, mode=0o644)
    try:
        _atomic_replace(staged, marker_path)
        _fsync_replaced_paths(staged, marker_path)
    finally:
        _unlink_durable(staged, missing_ok=True)


def _write_staged(path: Path, contents: bytes, *, mode: int) -> Path:
    _ensure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        if os.name != "nt":
            os.chmod(temporary_path, mode)
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary_path
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        _unlink_durable(temporary_path, missing_ok=True)
        raise


def _atomic_replace(source: str | Path, destination: str | Path) -> None:
    os.replace(source, destination)


def _fsync_replaced_paths(source: str | Path, destination: str | Path) -> None:
    parents = {Path(source).parent, Path(destination).parent}
    for parent in sorted(parents, key=str):
        _fsync_directory(parent)


def _publish(
    *,
    catalog_path: Path,
    original_catalog: bytes,
    new_catalog: bytes,
    artifact_path: str,
    schema_bytes: bytes,
) -> None:
    target = _contained_artifact_target(catalog_path.parent, artifact_path)
    target_preexisted = target.exists() or target.is_symlink()
    if target_preexisted and (not target.is_file() or target.read_bytes() != schema_bytes):
        raise PublishedSchemaConflictError(
            f"artifact path has different exact bytes: {artifact_path}"
        )
    if catalog_path.read_bytes() != original_catalog:
        raise SchemaRegistrationError("schema catalog changed while registration was prepared")

    marker_path = catalog_path.parent / SCHEMA_PUBLICATION_MARKER_FILENAME
    marker_bytes = _publication_marker_bytes(
        artifact_path=artifact_path,
        artifact_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        original_catalog=original_catalog,
        new_catalog=new_catalog,
    )
    catalog_mode = stat.S_IMODE(catalog_path.stat().st_mode)
    staged_schema: Path | None = None
    staged_catalog: Path | None = None
    schema_installed_here = False
    catalog_installed = False
    try:
        if not target_preexisted:
            _contained_artifact_target(catalog_path.parent, artifact_path)
            staged_schema = _write_staged(target, schema_bytes, mode=0o644)
        staged_catalog = _write_staged(catalog_path, new_catalog, mode=catalog_mode)
        _install_publication_marker(marker_path, marker_bytes)
        if staged_schema is not None:
            _contained_artifact_target(catalog_path.parent, artifact_path)
            _atomic_replace(staged_schema, target)
            schema_installed_here = True
            _fsync_replaced_paths(staged_schema, target)
        _atomic_replace(staged_catalog, catalog_path)
        catalog_installed = True
        _fsync_replaced_paths(staged_catalog, catalog_path)
        SchemaRegistry(
            catalog_path,
            _skip_publication_lock=True,
        ).validate_schema_documents()
        _unlink_durable(marker_path)
    except BaseException:
        if catalog_installed:
            rollback_catalog = _write_staged(
                catalog_path,
                original_catalog,
                mode=catalog_mode,
            )
            try:
                _atomic_replace(rollback_catalog, catalog_path)
                _fsync_replaced_paths(rollback_catalog, catalog_path)
            finally:
                _unlink_durable(rollback_catalog, missing_ok=True)
        if schema_installed_here:
            _unlink_durable(target, missing_ok=True)
        if not target.exists() and not target.is_symlink():
            _unlink_durable(marker_path, missing_ok=True)
        raise
    finally:
        if staged_schema is not None:
            _unlink_durable(staged_schema, missing_ok=True)
        if staged_catalog is not None:
            _unlink_durable(staged_catalog, missing_ok=True)


def _register_schema_locked(
    candidate_path: str | Path,
    *,
    version: str,
    wire_version: str,
    owner: str,
    projection_version: str,
    catalog_path: str | Path = DEFAULT_SCHEMA_CATALOG,
    schema_id: str | None = None,
    artifact_path: str | None = None,
    canonicalization_version: str = "rfc8785-v1",
    software_min: str = "0.1.0",
    software_max: str = "0.2.0",
    dry_run: bool = False,
) -> SchemaRegistrationResult:
    """Validate and register one immutable schema, or prove an idempotent replay."""

    candidate = Path(candidate_path).resolve()
    catalog = Path(catalog_path).resolve()
    _recover_publication_marker(catalog)
    original, document, normalized = _load_candidate(candidate)
    document_id = document.get("$id")
    if not isinstance(document_id, str) or not document_id:
        raise SchemaRegistrationError("schema candidate must declare a non-empty $id")
    relative_artifact = _derive_artifact_path(document_id, artifact_path)
    if document_id != f"https://schemas.robata.dev/{relative_artifact}":
        raise SchemaRegistrationError("schema $id must match its catalog artifact path")
    _contained_artifact_target(catalog.parent, relative_artifact)
    logical_schema_id = _derive_schema_id(document_id, relative_artifact, schema_id)
    original_catalog = catalog.read_bytes()
    try:
        catalog_document = json.loads(original_catalog.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaRegistrationError(f"invalid schema catalog: {exc}") from exc
    if not isinstance(catalog_document, dict):
        raise SchemaRegistrationError("schema catalog root must be a JSON object")
    current = SchemaRegistry(catalog, _skip_publication_lock=True)
    digest = hashlib.sha256(normalized).hexdigest()
    artifact_id = deterministic_schema_artifact_id(digest)
    proposed_entry = _entry(
        schema_id=logical_schema_id,
        version=version,
        artifact_id=artifact_id,
        digest=digest,
        wire_version=wire_version,
        document_id=document_id,
        artifact_path=relative_artifact,
        owner=owner,
        canonicalization_version=canonicalization_version,
        projection_version=projection_version,
        software_min=software_min,
        software_max=software_max,
    )

    published = _registered_version(current, logical_schema_id, version)
    if published is not None:
        if published.document_bytes not in {original, normalized}:
            raise PublishedSchemaConflictError(
                f"published schema {logical_schema_id}@{version} has different exact bytes"
            )
        if published.entry.model_dump(mode="json") != proposed_entry:
            raise PublishedSchemaConflictError(
                f"published schema {logical_schema_id}@{version} has different metadata"
            )
        return SchemaRegistrationResult(
            ref=published.ref,
            artifact_path=published.entry.artifact_path,
            changed=False,
            dry_run=dry_run,
        )

    schemas = catalog_document.get("schemas")
    if not isinstance(schemas, list):
        raise SchemaRegistrationError("schema catalog schemas must be an array")
    schemas.append(proposed_entry)
    new_catalog = _catalog_bytes(catalog_document)

    snapshot = _validated_snapshot(catalog, relative_artifact, normalized, new_catalog)
    registered = _registered_version(snapshot, logical_schema_id, version)
    if registered is None:
        raise SchemaRegistrationError(f"validated snapshot omitted {logical_schema_id}@{version}")
    result = SchemaRegistrationResult(
        ref=registered.ref,
        artifact_path=relative_artifact,
        changed=True,
        dry_run=dry_run,
    )
    if dry_run:
        return result

    _publish(
        catalog_path=catalog,
        original_catalog=original_catalog,
        new_catalog=new_catalog,
        artifact_path=relative_artifact,
        schema_bytes=normalized,
    )
    return result


def register_schema(
    candidate_path: str | Path,
    *,
    version: str,
    wire_version: str,
    owner: str,
    projection_version: str,
    catalog_path: str | Path = DEFAULT_SCHEMA_CATALOG,
    schema_id: str | None = None,
    artifact_path: str | None = None,
    canonicalization_version: str = "rfc8785-v1",
    software_min: str = "0.1.0",
    software_max: str = "0.2.0",
    dry_run: bool = False,
) -> SchemaRegistrationResult:
    """Validate and register one immutable schema, or prove an idempotent replay."""

    catalog = Path(catalog_path).resolve()
    with _registration_lock(catalog.parent):
        return _register_schema_locked(
            candidate_path,
            version=version,
            wire_version=wire_version,
            owner=owner,
            projection_version=projection_version,
            catalog_path=catalog,
            schema_id=schema_id,
            artifact_path=artifact_path,
            canonicalization_version=canonicalization_version,
            software_min=software_min,
            software_max=software_max,
            dry_run=dry_run,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="JSON Schema candidate outside the registry")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_SCHEMA_CATALOG)
    parser.add_argument(
        "--schema-id", help="logical schema ID; inferred from the filename by default"
    )
    parser.add_argument(
        "--version", required=True, help="catalog semantic version, for example 2.0.0"
    )
    parser.add_argument("--wire-version", required=True, help="wire version, for example 2.0")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--projection-version", required=True)
    parser.add_argument(
        "--artifact-path", help="catalog-relative path; inferred from $id by default"
    )
    parser.add_argument("--canonicalization-version", default="rfc8785-v1")
    parser.add_argument("--software-min", default="0.1.0")
    parser.add_argument("--software-max", default="0.2.0")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = register_schema(
            args.candidate,
            catalog_path=args.catalog,
            schema_id=args.schema_id,
            version=args.version,
            wire_version=args.wire_version,
            owner=args.owner,
            projection_version=args.projection_version,
            artifact_path=args.artifact_path,
            canonicalization_version=args.canonicalization_version,
            software_min=args.software_min,
            software_max=args.software_max,
            dry_run=args.dry_run,
        )
    except (OSError, SchemaRegistrationError, SchemaRegistryError) as exc:
        print(f"schema registration failed: {exc}", file=sys.stderr)
        return 1

    status = "DRY-RUN" if result.dry_run else ("REGISTERED" if result.changed else "UNCHANGED")
    print(
        f"{status} {result.ref.schema_id}@{result.ref.version} "
        f"artifact_id={result.ref.artifact_id} sha256={result.ref.sha256} "
        f"path={result.artifact_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PublishedSchemaConflictError",
    "SchemaRegistrationError",
    "SchemaRegistrationResult",
    "main",
    "register_schema",
]
