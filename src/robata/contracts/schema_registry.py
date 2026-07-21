"""Offline registry and validation for checked-in Robata wire schemas."""

from __future__ import annotations

# mypy: disable-error-code=import-untyped
import copy
import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from time import monotonic, sleep
from typing import Annotated, Any, BinaryIO, Literal, TypeVar, cast

from jsonschema import Draft202012Validator, FormatChecker, validators
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import StringConstraints, model_validator
from pydantic import ValidationError as PydanticValidationError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable

from robata.contracts.common import StrictModel

DEFAULT_SCHEMA_CATALOG = Path(__file__).resolve().parents[3] / "schemas" / "schema-catalog.json"
CATALOG_SCHEMA_FILENAME = "schema-catalog.schema.json"
DEFAULT_SOFTWARE_VERSION = "0.1.0"
SCHEMA_PUBLICATION_LOCK_FILENAME = ".schema-registration.lock"
SCHEMA_PUBLICATION_MARKER_FILENAME = ".schema-publication-transaction.json"
SCHEMA_PUBLICATION_MARKER_FORMAT = "robata-schema-publication-transaction-v1"
_VERSION_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
_WIRE_VERSION_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
_SCHEMA_ID_PATTERN = r"^https://schemas\.robata\.dev/[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"
_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SCHEMA_ARTIFACT_ID_DOMAIN = b"robata-local-artifact-id-v1\x00JSON_SCHEMA\x00"
_SCHEMA_LOCK_TIMEOUT_SECONDS = 30.0
_SCHEMA_LOCK_RETRY_SECONDS = 0.01
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


_ROBATA_FORMAT_CHECKER = FormatChecker()


def _is_rfc3339_datetime(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    if _RFC3339_PATTERN.fullmatch(instance) is None:
        return False
    normalized = f"{instance[:-1]}+00:00" if instance.endswith("Z") else instance
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


cast(
    Callable[[str], Callable[[Callable[[object], bool]], Callable[[object], bool]]],
    _ROBATA_FORMAT_CHECKER.checks,
)("date-time")(_is_rfc3339_datetime)


def _is_strict_json_integer(_checker: object, instance: object) -> bool:
    return isinstance(instance, int) and not isinstance(instance, bool)


_STRICT_TYPE_CHECKER = Draft202012Validator.TYPE_CHECKER.redefine(
    "integer",
    _is_strict_json_integer,
)
_StrictDraft202012Validator = cast(
    Callable[..., type[Draft202012Validator]],
    validators.extend,
)(
    Draft202012Validator,
    type_checker=_STRICT_TYPE_CHECKER,
)


def deterministic_schema_artifact_id(exact_schema_sha256: str) -> str:
    if re.fullmatch(_SHA256_PATTERN, exact_schema_sha256) is None:
        raise ValueError("exact_schema_sha256 must be a lowercase SHA-256 digest")
    identity = hashlib.sha256(
        _SCHEMA_ARTIFACT_ID_DOMAIN + exact_schema_sha256.encode("ascii")
    ).hexdigest()[:32]
    return f"{identity[:8]}-{identity[8:12]}-{identity[12:16]}-{identity[16:20]}-{identity[20:]}"


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_FILE_SUFFIX = ".schema.json"
DEFAULT_SCHEMA_DIRECTORY = Path(__file__).resolve().parents[3] / "schemas" / "v1"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_Payload = TypeVar("_Payload")


class SchemaRegistryError(RuntimeError):
    """Base class for deterministic schema loading and lookup failures."""


class SchemaDefinitionError(SchemaRegistryError):
    """Raised when a checked-in schema is malformed or cannot resolve locally."""

    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail
        super().__init__(f"invalid schema '{source}': {detail}")


class SchemaNotFoundError(SchemaRegistryError, LookupError):
    """Raised when a caller requests an unregistered schema."""

    def __init__(self, schema: str) -> None:
        self.schema = schema
        super().__init__(f"schema is not registered: {schema}")


class SchemaValidationError(ValueError):
    """A stable payload validation error with a canonical JSON path."""

    def __init__(
        self,
        schema_name: str,
        path: str,
        detail: str,
        validator: str | None,
    ) -> None:
        self.schema_name = schema_name
        self.path = path
        self.json_path = path
        self.detail = detail
        self.validator = validator
        super().__init__(f"schema '{schema_name}' validation failed at {path}: {detail}")


# Kept as an explicit alias because both terms are common at call sites.
PayloadValidationError = SchemaValidationError


_CatalogVersion = Annotated[str, StringConstraints(strict=True, pattern=_VERSION_PATTERN)]
_WireVersion = Annotated[str, StringConstraints(strict=True, pattern=_WIRE_VERSION_PATTERN)]
_CatalogId = Annotated[str, StringConstraints(strict=True, pattern=_SCHEMA_ID_PATTERN)]
_UpcasterId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"),
]
_ArtifactId = Annotated[str, StringConstraints(strict=True, pattern=_UUID_PATTERN)]
_Digest = Annotated[str, StringConstraints(strict=True, pattern=_SHA256_PATTERN)]
_NonEmpty = Annotated[str, StringConstraints(strict=True, min_length=1)]


class CompatibilityMode(StrEnum):
    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"
    FULL = "FULL"
    NONE = "NONE"


class SchemaLifecycle(StrEnum):
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


def _version_key(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(_VERSION_PATTERN, value)
    if match is None:
        raise ValueError(f"invalid semantic version: {value}")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


class SchemaRef(StrictModel):
    """Exact immutable four-tuple required at durable consumer boundaries."""

    schema_id: _CatalogId
    version: _CatalogVersion
    artifact_id: _ArtifactId
    sha256: _Digest

    @property
    def key(self) -> tuple[str, str]:
        return self.schema_id, self.version


class SoftwareRange(StrictModel):
    min_inclusive: _CatalogVersion
    max_exclusive: _CatalogVersion

    @model_validator(mode="after")
    def validate_order(self) -> SoftwareRange:
        if _version_key(self.min_inclusive) >= _version_key(self.max_exclusive):
            raise ValueError("min_inclusive must be lower than max_exclusive")
        return self

    def contains(self, version: str) -> bool:
        key = _version_key(version)
        return _version_key(self.min_inclusive) <= key < _version_key(self.max_exclusive)


class SchemaCatalogEntry(StrictModel):
    ref: SchemaRef
    wire_version: _WireVersion
    document_id: _NonEmpty
    artifact_path: _NonEmpty
    owner: _NonEmpty
    canonicalization_version: _NonEmpty
    projection_version: _NonEmpty
    compatibility_mode: CompatibilityMode
    lifecycle: SchemaLifecycle
    supported_software: SoftwareRange
    supported_predecessors: tuple[SchemaRef, ...] = ()


class GoldenVectorCatalogEntry(StrictModel):
    input_artifact_path: _NonEmpty
    input_sha256: _Digest
    output_artifact_path: _NonEmpty
    output_sha256: _Digest


class UpcasterCatalogEntry(StrictModel):
    upcaster_id: _UpcasterId
    source: SchemaRef
    target: SchemaRef
    code_artifact_id: _ArtifactId
    code_artifact_path: _NonEmpty
    code_sha256: _Digest
    runtime_artifact_id: _ArtifactId
    runtime_artifact_path: _NonEmpty
    runtime_sha256: _Digest
    golden_vectors: tuple[GoldenVectorCatalogEntry, ...]

    @model_validator(mode="after")
    def require_golden_vectors(self) -> UpcasterCatalogEntry:
        if not self.golden_vectors:
            raise ValueError("at least one golden vector pair is required")
        pairs = {
            (
                vector.input_artifact_path,
                vector.input_sha256,
                vector.output_artifact_path,
                vector.output_sha256,
            )
            for vector in self.golden_vectors
        }
        if len(pairs) != len(self.golden_vectors):
            raise ValueError("golden vector pairs must be unique")
        return self


class SchemaCatalog(StrictModel):
    catalog_version: Literal["1.0"]
    schemas: tuple[SchemaCatalogEntry, ...]
    upcasters: tuple[UpcasterCatalogEntry, ...] = ()


class _SchemaPublicationMarker(StrictModel):
    format_version: Literal["robata-schema-publication-transaction-v1"]
    artifact_path: _NonEmpty
    artifact_sha256: _Digest
    original_catalog_sha256: _Digest
    new_catalog_sha256: _Digest


@dataclass(frozen=True)
class RegisteredSchema:
    entry: SchemaCatalogEntry
    path: Path
    document_bytes: bytes

    @property
    def ref(self) -> SchemaRef:
        return self.entry.ref


@dataclass(frozen=True)
class RegisteredGoldenVector:
    entry: GoldenVectorCatalogEntry
    input_path: Path
    input_bytes: bytes
    output_path: Path
    output_bytes: bytes


@dataclass(frozen=True)
class RegisteredUpcaster:
    entry: UpcasterCatalogEntry
    code_path: Path
    code_bytes: bytes
    runtime_path: Path
    runtime_bytes: bytes
    golden_vectors: tuple[RegisteredGoldenVector, ...]


class SchemaPinMismatchError(SchemaRegistryError):
    def __init__(self, schema_ref: SchemaRef, expected: SchemaRef) -> None:
        self.schema_ref = schema_ref
        self.expected = expected
        super().__init__(
            f"schema pin mismatch for {schema_ref.schema_id}@{schema_ref.version}: "
            f"expected artifact {expected.artifact_id} and digest {expected.sha256}"
        )


class SchemaAmbiguityError(SchemaRegistryError, LookupError):
    pass


class SchemaCompatibilityError(SchemaRegistryError):
    pass


class _DirectorySchemaRegistry:
    """Load, check, resolve, and validate versioned schemas without network I/O."""

    def __init__(self, schema_directory: str | Path | None = None) -> None:
        self.schema_directory = Path(
            schema_directory if schema_directory is not None else DEFAULT_SCHEMA_DIRECTORY
        ).resolve()
        if not self.schema_directory.is_dir():
            raise SchemaRegistryError(f"schema directory does not exist: {self.schema_directory}")

        paths = sorted(self.schema_directory.glob(f"*{SCHEMA_FILE_SUFFIX}"))
        if not paths:
            raise SchemaRegistryError(f"no schema documents found in: {self.schema_directory}")

        schemas_by_name: dict[str, dict[str, Any]] = {}
        names_by_id: dict[str, str] = {}
        sources_by_name: dict[str, Path] = {}

        for path in paths:
            name = path.name.removesuffix(SCHEMA_FILE_SUFFIX)
            schema = self._load_document(path)
            schema_id = schema.get("$id")
            if not isinstance(schema_id, str) or not schema_id:
                raise SchemaDefinitionError(path.name, "missing non-empty $id")
            if schema.get("$schema") != JSON_SCHEMA_DIALECT:
                raise SchemaDefinitionError(
                    path.name,
                    f"$schema must be {JSON_SCHEMA_DIALECT}",
                )
            if schema_id in names_by_id:
                previous = sources_by_name[names_by_id[schema_id]].name
                raise SchemaDefinitionError(
                    path.name,
                    f"duplicate $id {schema_id!r}; already declared by {previous}",
                )

            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                path_text = _format_json_path(exc.absolute_path)
                raise SchemaDefinitionError(
                    path.name,
                    f"meta-schema failure at {path_text}: {exc.message}",
                ) from exc

            schemas_by_name[name] = schema
            names_by_id[schema_id] = name
            sources_by_name[name] = path

        resources = [
            (schema_id, Resource.from_contents(schemas_by_name[name]))
            for schema_id, name in sorted(names_by_id.items())
        ]
        self._registry: Registry[Any] = Registry[dict[str, Any]]().with_resources(resources)
        self._schemas_by_name = schemas_by_name
        self._names_by_id = names_by_id
        self._sources_by_name = sources_by_name
        self.validate_schema_documents()

    @staticmethod
    def _load_document(path: Path) -> dict[str, Any]:
        try:
            contents = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SchemaDefinitionError(path.name, str(exc)) from exc
        if not isinstance(contents, dict):
            raise SchemaDefinitionError(path.name, "document root must be an object")
        return contents

    @property
    def schema_names(self) -> tuple[str, ...]:
        """Return canonical short names in deterministic order."""

        return tuple(sorted(self._schemas_by_name))

    @property
    def schema_ids(self) -> tuple[str, ...]:
        """Return all registered identifiers in deterministic order."""

        return tuple(sorted(self._names_by_id))

    def get_schema(self, schema: str) -> dict[str, Any]:
        """Return a schema by short name, filename, or exact ``$id``."""

        name = self._canonical_name(schema)
        return self._schemas_by_name[name]

    # ``get`` is convenient for callers treating this object as a registry.
    get = get_schema

    def validator(self, schema: str) -> Draft202012Validator:
        """Build a validator backed only by resources loaded from this directory."""

        document = self.get_schema(schema)
        return _StrictDraft202012Validator(
            document,
            registry=self._registry,
            format_checker=_ROBATA_FORMAT_CHECKER,
        )

    def validate(self, schema: str, payload: _Payload) -> _Payload:
        """Validate a payload, returning it unchanged when valid."""

        name = self._canonical_name(schema)
        errors = sorted(self.validator(name).iter_errors(payload), key=_error_sort_key)
        if errors:
            error = errors[0]
            path_parts, detail = _stable_error_details(error)
            raise SchemaValidationError(
                name,
                _format_json_path(path_parts),
                detail,
                error.validator if isinstance(error.validator, str) else None,
            ) from error
        return payload

    def is_valid(self, schema: str, payload: Any) -> bool:
        """Return whether a payload conforms to the selected schema."""

        try:
            self.validate(schema, payload)
        except SchemaValidationError:
            return False
        return True

    def validate_schema_documents(self) -> None:
        """Recheck meta-schemas and prove every reference resolves locally."""

        for name in self.schema_names:
            schema = self._schemas_by_name[name]
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise SchemaDefinitionError(name, exc.message) from exc

            schema_id = schema["$id"]
            resolver = self._registry.resolver(schema_id)
            for ref in _iter_references(schema):
                try:
                    resolver.lookup(ref)
                except Unresolvable as exc:
                    raise SchemaDefinitionError(
                        self._sources_by_name[name].name,
                        f"unresolved offline reference {ref!r}",
                    ) from exc

    def _canonical_name(self, schema: str) -> str:
        if schema in self._schemas_by_name:
            return schema
        if schema in self._names_by_id:
            return self._names_by_id[schema]
        if schema.endswith(SCHEMA_FILE_SUFFIX):
            candidate = Path(schema).name.removesuffix(SCHEMA_FILE_SUFFIX)
            if candidate in self._schemas_by_name:
                return candidate
        raise SchemaNotFoundError(schema)


def _iter_references(value: Any) -> tuple[str, ...]:
    refs: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                refs.append(ref)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return tuple(refs)


def _error_sort_key(error: ValidationError) -> tuple[Any, ...]:
    path = tuple(
        (0, part) if isinstance(part, int) else (1, str(part)) for part in error.absolute_path
    )
    schema_path = tuple(str(part) for part in error.absolute_schema_path)
    return path, schema_path, error.message


def _stable_error_details(error: ValidationError) -> tuple[tuple[Any, ...], str]:
    path = tuple(error.absolute_path)

    if error.validator == "required" and isinstance(error.instance, dict):
        required = error.validator_value
        if isinstance(required, list):
            missing = [item for item in required if item not in error.instance]
            if missing:
                field = str(missing[0])
                return (*path, field), f"required property {field!r} is missing"

    if (
        error.validator in {"additionalProperties", "unevaluatedProperties"}
        and isinstance(error.instance, dict)
        and isinstance(error.schema, dict)
    ):
        declared = error.schema.get("properties", {})
        if isinstance(declared, dict):
            extras = sorted(set(error.instance) - set(declared))
            if extras:
                field = extras[0]
                return (*path, field), f"additional property {field!r} is not allowed"

    return path, error.message


def _format_json_path(parts: Any) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        elif _IDENTIFIER.fullmatch(str(part)):
            result += f".{part}"
        else:
            result += f"[{json.dumps(str(part), ensure_ascii=True)}]"
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_json_object(source: str, raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaDefinitionError(source, str(exc)) from exc
    if not isinstance(value, dict):
        raise SchemaDefinitionError(source, "document root must be an object")
    return value


def _load_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SchemaDefinitionError(path.name, str(exc)) from exc
    value = _decode_json_object(path.name, raw)
    return raw, value


def _locate_catalog(source: str | Path | None) -> Path:
    path = Path(source) if source is not None else DEFAULT_SCHEMA_CATALOG
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        candidates = (resolved / "schema-catalog.json", resolved.parent / "schema-catalog.json")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    raise SchemaRegistryError(f"schema catalog does not exist for: {resolved}")


def _is_symlink_or_reparse(metadata: os.stat_result) -> bool:
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        reparse_attribute and file_attributes & reparse_attribute
    )


def _scan_schema_tree(root: Path) -> tuple[Path, ...]:
    pending = [root]
    paths: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                inspected = sorted(
                    (
                        Path(entry.path),
                        entry.stat(follow_symlinks=False),
                    )
                    for entry in entries
                )
        except OSError as exc:
            raise SchemaDefinitionError(directory.as_posix(), str(exc)) from exc
        child_directories: list[Path] = []
        for path, metadata in inspected:
            if _is_symlink_or_reparse(metadata):
                raise SchemaDefinitionError(
                    path.relative_to(root).as_posix(),
                    "schema tree must not contain a symlink or reparse point",
                )
            paths.append(path)
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append(path)
        pending.extend(reversed(child_directories))
    return tuple(paths)


def _resolve_artifact_path(root: Path, relative: str) -> Path:
    if "\\" in relative or Path(relative).is_absolute():
        raise SchemaDefinitionError(relative, "artifact_path must be a relative POSIX path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SchemaDefinitionError(relative, "artifact_path contains an unsafe path segment")
    lexical = root / Path(*parts)
    cursor = root
    for part in parts:
        cursor /= part
        if not cursor.exists() and not cursor.is_symlink():
            break
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise SchemaDefinitionError(relative, f"cannot inspect artifact_path: {exc}") from exc
        if _is_symlink_or_reparse(metadata):
            raise SchemaDefinitionError(
                relative,
                "artifact_path traverses a symlink or reparse point",
            )
    try:
        resolved = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SchemaDefinitionError(relative, f"cannot resolve artifact_path: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise SchemaDefinitionError(relative, "artifact_path escapes the catalog directory")
    return resolved


def _lock_schema_reader(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        deadline = monotonic() + _SCHEMA_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if monotonic() >= deadline:
                    raise
                sleep(_SCHEMA_LOCK_RETRY_SECONDS)
                stream.seek(0)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_SH)  # type: ignore[attr-defined]


def _unlock_schema_reader(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def _validate_existing_schema_lock(lock_path: Path) -> None:
    try:
        metadata = lock_path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SchemaRegistryError(f"cannot inspect schema publication lock: {exc}") from exc
    if _is_symlink_or_reparse(metadata):
        raise SchemaRegistryError("schema publication lock must not be a symlink or reparse point")
    if not stat.S_ISREG(metadata.st_mode):
        raise SchemaRegistryError("schema publication lock must be a regular file")


def _open_schema_lock(lock_path: Path, *, writable: bool) -> BinaryIO:
    flags = (os.O_RDWR | os.O_CREAT) if writable else os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SchemaRegistryError("schema publication lock must be a regular file")
        return os.fdopen(descriptor, "r+b" if writable else "rb")
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _schema_publication_read_lock(schema_root: Path, *, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    lock_path = schema_root / SCHEMA_PUBLICATION_LOCK_FILENAME
    _validate_existing_schema_lock(lock_path)
    stream: BinaryIO
    try:
        stream = _open_schema_lock(lock_path, writable=True)
    except OSError as writable_error:
        _validate_existing_schema_lock(lock_path)
        try:
            stream = _open_schema_lock(lock_path, writable=False)
        except FileNotFoundError:
            # A read-only registry cannot have a concurrent local publisher.
            yield
            return
        except OSError as exc:
            raise SchemaRegistryError(f"cannot open schema publication lock: {exc}") from exc
        if stream.seek(0, os.SEEK_END) == 0:
            stream.close()
            raise SchemaRegistryError(
                f"schema publication lock is empty and not writable: {writable_error}"
            ) from writable_error
    else:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()

    locked = False
    try:
        _lock_schema_reader(stream)
        locked = True
    except OSError as exc:
        stream.close()
        raise SchemaRegistryError(f"cannot acquire schema publication read lock: {exc}") from exc
    try:
        yield
    finally:
        if locked:
            _unlock_schema_reader(stream)
        stream.close()


class _CatalogSchemaRegistry:
    """Catalog-driven, digest-pinned, offline multiversion schema registry."""

    def __init__(
        self,
        catalog: str | Path | None = None,
        *,
        software_version: str = DEFAULT_SOFTWARE_VERSION,
        _skip_publication_lock: bool = False,
    ) -> None:
        _version_key(software_version)
        self.software_version = software_version
        self.catalog_path = _locate_catalog(catalog)
        self.schema_root = self.catalog_path.parent.resolve()
        with _schema_publication_read_lock(
            self.schema_root,
            enabled=not _skip_publication_lock,
        ):
            catalog_raw, catalog_data = _load_json_object(self.catalog_path)
            self._catalog_sha256 = hashlib.sha256(catalog_raw).hexdigest()
            self._validate_catalog_document(catalog_data)
            try:
                self.catalog = SchemaCatalog.model_validate_json(catalog_raw, strict=True)
            except PydanticValidationError as exc:
                raise SchemaDefinitionError(self.catalog_path.name, str(exc)) from exc

            self._registered_by_key: dict[tuple[str, str], RegisteredSchema] = {}
            self._registered_by_artifact_id: dict[str, RegisteredSchema] = {}
            self._registered_by_digest: dict[str, RegisteredSchema] = {}
            self._registered_by_document_id: dict[str, RegisteredSchema] = {}
            self._registered_by_path: dict[Path, RegisteredSchema] = {}
            self._documents_by_key: dict[tuple[str, str], dict[str, Any]] = {}
            self._aliases: dict[str, list[SchemaRef]] = {}
            self._registered_upcasters_by_id: dict[str, RegisteredUpcaster] = {}
            self._load_catalog_entries()
            self._build_reference_registry()
            self._validate_catalog_metadata()
            self._load_upcaster_artifacts()
            self.validate_schema_documents()

    def _validate_catalog_metadata(self) -> None:
        for registered in self.entries:
            entry = registered.entry
            if entry.compatibility_mode is CompatibilityMode.NONE:
                if entry.supported_predecessors:
                    raise SchemaDefinitionError(
                        self.catalog_path.name,
                        "NONE compatibility cannot declare supported_predecessors",
                    )
            elif not entry.supported_predecessors:
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    f"{entry.ref.schema_id}@{entry.ref.version} must declare predecessors",
                )
            if len(set(entry.supported_predecessors)) != len(entry.supported_predecessors):
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    f"duplicate predecessor for {entry.ref.schema_id}@{entry.ref.version}",
                )
            for predecessor in entry.supported_predecessors:
                self.resolve_exact(predecessor, require_software_support=False)
                if predecessor.schema_id != entry.ref.schema_id:
                    raise SchemaDefinitionError(
                        self.catalog_path.name,
                        "predecessor and target must share schema_id",
                    )
                if _version_key(predecessor.version) >= _version_key(entry.ref.version):
                    raise SchemaDefinitionError(
                        self.catalog_path.name,
                        "predecessor version must be lower than target version",
                    )

        seen_upcaster_ids: set[str] = set()
        seen_edges: set[tuple[SchemaRef, SchemaRef]] = set()
        outgoing: dict[SchemaRef, list[SchemaRef]] = {}
        upcaster_nodes: set[SchemaRef] = set()
        for upcaster in self.catalog.upcasters:
            if upcaster.upcaster_id in seen_upcaster_ids:
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    f"duplicate upcaster_id: {upcaster.upcaster_id}",
                )
            edge = (upcaster.source, upcaster.target)
            if edge in seen_edges:
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    f"duplicate upcaster edge: {edge!r}",
                )
            seen_upcaster_ids.add(upcaster.upcaster_id)
            seen_edges.add(edge)
            outgoing.setdefault(upcaster.source, []).append(upcaster.target)
            upcaster_nodes.update(edge)
            self.resolve_exact(upcaster.source, require_software_support=False)
            target_registered = self.resolve_exact(upcaster.target, require_software_support=False)
            if upcaster.source.schema_id != upcaster.target.schema_id:
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    "upcaster source and target must share schema_id",
                )
            if _version_key(upcaster.source.version) >= _version_key(upcaster.target.version):
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    "upcaster source version must be lower than target version",
                )

            if upcaster.source not in target_registered.entry.supported_predecessors:
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    "upcaster source must be a declared target predecessor: "
                    f"{upcaster.upcaster_id}",
                )

        ordered_nodes = sorted(
            upcaster_nodes,
            key=lambda ref: (ref.schema_id, _version_key(ref.version)),
        )
        for source in ordered_nodes:
            path_counts: dict[SchemaRef, int] = {source: 1}
            for node in ordered_nodes:
                count = path_counts.get(node, 0)
                if count == 0:
                    continue
                for target in outgoing.get(node, []):
                    path_counts[target] = min(2, path_counts.get(target, 0) + count)
                    if path_counts[target] > 1:
                        raise SchemaDefinitionError(
                            self.catalog_path.name,
                            "multiple upcaster paths connect "
                            f"{source.schema_id}@{source.version} to "
                            f"{target.schema_id}@{target.version}",
                        )

    def _load_upcaster_artifacts(self) -> None:
        for entry in self.catalog.upcasters:
            code_path, code_bytes = self._load_pinned_artifact(
                entry.code_artifact_path,
                entry.code_sha256,
                kind="upcaster code",
            )
            runtime_path, runtime_bytes = self._load_pinned_artifact(
                entry.runtime_artifact_path,
                entry.runtime_sha256,
                kind="upcaster runtime",
            )
            _decode_json_object(entry.runtime_artifact_path, runtime_bytes)

            golden_vectors: list[RegisteredGoldenVector] = []
            for vector in entry.golden_vectors:
                input_path, input_bytes = self._load_pinned_artifact(
                    vector.input_artifact_path,
                    vector.input_sha256,
                    kind="upcaster golden input",
                )
                output_path, output_bytes = self._load_pinned_artifact(
                    vector.output_artifact_path,
                    vector.output_sha256,
                    kind="upcaster golden output",
                )
                input_payload = _decode_json_object(vector.input_artifact_path, input_bytes)
                output_payload = _decode_json_object(vector.output_artifact_path, output_bytes)
                source = self.resolve_exact(entry.source, require_software_support=False)
                target = self.resolve_exact(entry.target, require_software_support=False)
                self._validate_registered(source, input_payload)
                self._validate_registered(target, output_payload)
                golden_vectors.append(
                    RegisteredGoldenVector(
                        entry=vector,
                        input_path=input_path,
                        input_bytes=input_bytes,
                        output_path=output_path,
                        output_bytes=output_bytes,
                    )
                )

            self._registered_upcasters_by_id[entry.upcaster_id] = RegisteredUpcaster(
                entry=entry,
                code_path=code_path,
                code_bytes=code_bytes,
                runtime_path=runtime_path,
                runtime_bytes=runtime_bytes,
                golden_vectors=tuple(golden_vectors),
            )

    def _load_pinned_artifact(
        self,
        relative_path: str,
        expected_sha256: str,
        *,
        kind: str,
    ) -> tuple[Path, bytes]:
        path = _resolve_artifact_path(self.schema_root, relative_path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SchemaDefinitionError(relative_path, f"cannot read {kind}: {exc}") from exc
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SchemaDefinitionError(
                relative_path,
                f"exact SHA-256 mismatch: catalog={expected_sha256}, actual={actual_sha256}",
            )
        return path, raw

    def _pending_publication_artifact(self) -> Path | None:
        marker_path = self.schema_root / SCHEMA_PUBLICATION_MARKER_FILENAME
        try:
            marker_raw = marker_path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SchemaDefinitionError(marker_path.name, str(exc)) from exc
        marker_data = _decode_json_object(marker_path.name, marker_raw)
        try:
            marker = _SchemaPublicationMarker.model_validate(marker_data, strict=True)
        except PydanticValidationError as exc:
            raise SchemaDefinitionError(marker_path.name, str(exc)) from exc
        if self._catalog_sha256 not in {
            marker.original_catalog_sha256,
            marker.new_catalog_sha256,
        }:
            raise SchemaDefinitionError(
                marker_path.name,
                "publication marker does not match the current catalog digest",
            )

        target = _resolve_artifact_path(self.schema_root, marker.artifact_path)
        target_present = target.exists() or target.is_symlink()
        if not target_present:
            if self._catalog_sha256 == marker.new_catalog_sha256:
                raise SchemaDefinitionError(
                    marker.artifact_path,
                    "published marker artifact is missing",
                )
            return None
        if not target.is_file():
            raise SchemaDefinitionError(marker.artifact_path, "marker artifact is not a file")
        try:
            actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise SchemaDefinitionError(marker.artifact_path, str(exc)) from exc
        if actual_sha256 != marker.artifact_sha256:
            raise SchemaDefinitionError(
                marker.artifact_path,
                "publication marker artifact SHA-256 mismatch",
            )
        if self._catalog_sha256 == marker.original_catalog_sha256:
            return target
        return None

    @property
    def entries(self) -> tuple[RegisteredSchema, ...]:
        return tuple(self._registered_by_key[key] for key in sorted(self._registered_by_key))

    @property
    def schema_names(self) -> tuple[str, ...]:
        return tuple(sorted({entry.ref.schema_id for entry in self.catalog.schemas}))

    @property
    def schema_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registered_by_document_id))

    @property
    def upcasters(self) -> tuple[UpcasterCatalogEntry, ...]:
        return self.catalog.upcasters

    @property
    def registered_upcasters(self) -> tuple[RegisteredUpcaster, ...]:
        return tuple(
            self._registered_upcasters_by_id[key]
            for key in sorted(self._registered_upcasters_by_id)
        )

    def _ensure_supported(self, registered: RegisteredSchema) -> None:
        entry = registered.entry
        if entry.lifecycle is SchemaLifecycle.RETIRED:
            raise SchemaCompatibilityError(
                f"schema is retired: {entry.ref.schema_id}@{entry.ref.version}"
            )
        if not entry.supported_software.contains(self.software_version):
            raise SchemaCompatibilityError(
                f"schema {entry.ref.schema_id}@{entry.ref.version} does not support software "
                f"{self.software_version}"
            )

    def resolve_version(self, schema_id: str, version: str) -> RegisteredSchema:
        try:
            registered = self._registered_by_key[(schema_id, version)]
        except KeyError as exc:
            raise SchemaNotFoundError(f"{schema_id}@{version}") from exc
        self._ensure_supported(registered)
        return registered

    def resolve_exact(
        self,
        ref: SchemaRef,
        *,
        require_software_support: bool = True,
    ) -> RegisteredSchema:
        try:
            registered = self._registered_by_key[ref.key]
        except KeyError as exc:
            raise SchemaNotFoundError(f"{ref.schema_id}@{ref.version}") from exc
        if ref != registered.ref:
            raise SchemaPinMismatchError(ref, registered.ref)
        if require_software_support:
            self._ensure_supported(registered)
        return registered

    def resolve_alias(self, alias: str) -> RegisteredSchema:
        refs = self._aliases.get(alias, [])
        if not refs:
            raise SchemaNotFoundError(alias)
        if len(refs) > 1:
            versions = ", ".join(sorted(f"{ref.schema_id}@{ref.version}" for ref in refs))
            raise SchemaAmbiguityError(f"ambiguous schema alias {alias!r}: {versions}")
        return self.resolve_exact(refs[0])

    def _validate_catalog_document(self, data: dict[str, Any]) -> None:
        schema_path = DEFAULT_SCHEMA_CATALOG.parent / CATALOG_SCHEMA_FILENAME
        _, schema = _load_json_object(schema_path)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise SchemaDefinitionError(schema_path.name, exc.message) from exc
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(data),
            key=_error_sort_key,
        )
        if errors:
            error = errors[0]
            raise SchemaDefinitionError(
                self.catalog_path.name,
                f"catalog validation failed at {_format_json_path(error.absolute_path)}: "
                f"{error.message}",
            ) from error

    def _load_catalog_entries(self) -> None:
        for entry in self.catalog.schemas:
            key = entry.ref.key
            if key in self._registered_by_key:
                raise SchemaDefinitionError(
                    self.catalog_path.name,
                    f"duplicate schema version {entry.ref.schema_id}@{entry.ref.version}",
                )
            path = _resolve_artifact_path(self.schema_root, entry.artifact_path)
            raw, document = _load_json_object(path)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != entry.ref.sha256:
                raise SchemaDefinitionError(
                    entry.artifact_path,
                    f"exact SHA-256 mismatch: catalog={entry.ref.sha256}, actual={digest}",
                )
            expected_artifact_id = deterministic_schema_artifact_id(digest)
            if entry.ref.artifact_id != expected_artifact_id:
                raise SchemaDefinitionError(
                    entry.artifact_path,
                    f"artifact_id must be {expected_artifact_id} for exact schema digest",
                )
            if document.get("$schema") != JSON_SCHEMA_DIALECT:
                raise SchemaDefinitionError(
                    entry.artifact_path,
                    f"$schema must be {JSON_SCHEMA_DIALECT}",
                )
            if document.get("$id") != entry.document_id:
                raise SchemaDefinitionError(
                    entry.artifact_path,
                    f"document $id must equal catalog document_id {entry.document_id!r}",
                )
            try:
                Draft202012Validator.check_schema(document)
            except SchemaError as exc:
                raise SchemaDefinitionError(entry.artifact_path, exc.message) from exc

            registered = RegisteredSchema(entry=entry, path=path, document_bytes=raw)
            unique_indexes: tuple[tuple[dict[Any, RegisteredSchema], Any, str], ...] = (
                (self._registered_by_artifact_id, entry.ref.artifact_id, "artifact_id"),
                (self._registered_by_digest, entry.ref.sha256, "sha256"),
                (self._registered_by_document_id, entry.document_id, "document_id"),
                (self._registered_by_path, path, "artifact_path"),
            )
            for index, value, label in unique_indexes:
                if value in index:
                    raise SchemaDefinitionError(
                        self.catalog_path.name,
                        f"duplicate {label}: {value}",
                    )
                index[value] = registered
            self._registered_by_key[key] = registered
            self._documents_by_key[key] = document

            short_name = path.name.removesuffix(SCHEMA_FILE_SUFFIX)
            for alias in (
                entry.ref.schema_id,
                entry.document_id,
                path.name,
                short_name,
                f"{entry.ref.schema_id}@{entry.ref.version}",
            ):
                self._add_alias(alias, entry.ref)

        cataloged = set(self._registered_by_path)
        discovered: set[Path] = set()
        tree_paths = _scan_schema_tree(self.schema_root)
        for path in (item for item in tree_paths if item.name.endswith(SCHEMA_FILE_SUFFIX)):
            relative = path.relative_to(self.schema_root)
            cursor = self.schema_root
            for part in relative.parts:
                cursor /= part
                try:
                    path_metadata = cursor.lstat()
                except OSError as exc:
                    raise SchemaDefinitionError(path.as_posix(), str(exc)) from exc
                if _is_symlink_or_reparse(path_metadata):
                    raise SchemaDefinitionError(
                        path.as_posix(),
                        "schema path must not traverse a symlink or reparse point",
                    )
            if relative.parts == (CATALOG_SCHEMA_FILENAME,):
                continue
            if len(relative.parts) < 2 or re.fullmatch(r"v[0-9]+", relative.parts[0]) is None:
                raise SchemaDefinitionError(
                    relative.as_posix(),
                    "schema document must live below a v[0-9]+ directory",
                )
            if not path.is_file():
                raise SchemaDefinitionError(relative.as_posix(), "schema document is not a file")
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError) as exc:
                raise SchemaDefinitionError(path.as_posix(), str(exc)) from exc
            if not resolved.is_relative_to(self.schema_root):
                raise SchemaDefinitionError(
                    path.as_posix(),
                    "discovered schema path escapes the catalog directory",
                )
            discovered.add(resolved)
        uncataloged = discovered - cataloged
        pending = self._pending_publication_artifact()
        if pending is not None:
            uncataloged.discard(pending)
        missing = sorted(path.relative_to(self.schema_root).as_posix() for path in uncataloged)
        nonexistent = sorted(
            path.relative_to(self.schema_root).as_posix() for path in cataloged - discovered
        )
        if missing or nonexistent:
            detail = []
            if missing:
                detail.append(f"uncataloged schema documents: {missing!r}")
            if nonexistent:
                detail.append(f"catalog paths outside versioned schema set: {nonexistent!r}")
            raise SchemaDefinitionError(self.catalog_path.name, "; ".join(detail))

    def _add_alias(self, alias: str, ref: SchemaRef) -> None:
        refs = self._aliases.setdefault(alias, [])
        if ref not in refs:
            refs.append(ref)

    def _build_reference_registry(self) -> None:
        resources = [
            (registered.entry.document_id, Resource.from_contents(self._documents_by_key[key]))
            for key, registered in sorted(self._registered_by_key.items())
        ]
        self._registry: Registry[Any] = Registry[dict[str, Any]]().with_resources(resources)

    def get_schema(self, schema: str | SchemaRef) -> dict[str, Any]:
        registered = (
            self.resolve_exact(schema)
            if isinstance(schema, SchemaRef)
            else self.resolve_alias(schema)
        )
        return copy.deepcopy(self._documents_by_key[registered.ref.key])

    get = get_schema

    def validator(self, schema: str | SchemaRef) -> Draft202012Validator:
        registered = (
            self.resolve_exact(schema)
            if isinstance(schema, SchemaRef)
            else self.resolve_alias(schema)
        )
        return _StrictDraft202012Validator(
            self._documents_by_key[registered.ref.key],
            registry=self._registry,
            format_checker=_ROBATA_FORMAT_CHECKER,
        )

    def _validate_registered(self, registered: RegisteredSchema, payload: _Payload) -> _Payload:
        validator = _StrictDraft202012Validator(
            self._documents_by_key[registered.ref.key],
            registry=self._registry,
            format_checker=_ROBATA_FORMAT_CHECKER,
        )
        errors = sorted(validator.iter_errors(payload), key=_error_sort_key)
        if errors:
            error = errors[0]
            path_parts, detail = _stable_error_details(error)
            raise SchemaValidationError(
                f"{registered.ref.schema_id}@{registered.ref.version}",
                _format_json_path(path_parts),
                detail,
                error.validator if isinstance(error.validator, str) else None,
            ) from error
        return payload

    def validate(self, schema: str | SchemaRef, payload: _Payload) -> _Payload:
        registered = (
            self.resolve_exact(schema)
            if isinstance(schema, SchemaRef)
            else self.resolve_alias(schema)
        )
        return self._validate_registered(registered, payload)

    def validate_pinned(self, ref: SchemaRef, payload: _Payload) -> _Payload:
        return self._validate_registered(self.resolve_exact(ref), payload)

    def is_valid(self, schema: str | SchemaRef, payload: Any) -> bool:
        try:
            self.validate(schema, payload)
        except SchemaValidationError:
            return False
        return True

    def validate_schema_documents(self) -> None:
        for registered in self.entries:
            document = self._documents_by_key[registered.ref.key]
            resolver = self._registry.resolver(registered.entry.document_id)
            for ref in _iter_references(document):
                try:
                    resolver.lookup(ref)
                except Unresolvable as exc:
                    raise SchemaDefinitionError(
                        registered.entry.artifact_path,
                        f"unresolved offline reference {ref!r}",
                    ) from exc

    def require_compatible(self, writer: SchemaRef, consumer: SchemaRef) -> None:
        self.resolve_exact(writer, require_software_support=False)
        target = self.resolve_exact(consumer, require_software_support=False)
        if writer == consumer:
            return
        if writer not in target.entry.supported_predecessors:
            raise SchemaCompatibilityError(
                f"{writer.schema_id}@{writer.version} is not compatible with "
                f"{consumer.schema_id}@{consumer.version}"
            )


class SchemaRegistry(_CatalogSchemaRegistry):
    """Use catalog mode by default, with explicit frozen-directory compatibility."""

    def __new__(
        cls,
        catalog: str | Path | None = None,
        *,
        software_version: str = DEFAULT_SOFTWARE_VERSION,
        _skip_publication_lock: bool = False,
    ) -> Any:
        del software_version, _skip_publication_lock
        if catalog is not None:
            source = Path(catalog).resolve()
            if source.is_dir() and not (source / DEFAULT_SCHEMA_CATALOG.name).is_file():
                return _DirectorySchemaRegistry(source)
        return super().__new__(cls)


@lru_cache(maxsize=1)
def default_schema_registry() -> SchemaRegistry:
    """Return the process-wide catalog-driven schema registry."""

    return SchemaRegistry()


def validate_payload[Payload](schema: str, payload: Payload) -> Payload:
    """Validate with the checked-in registry and return the payload unchanged."""

    return default_schema_registry().validate(schema, payload)


def validate_pinned_payload[Payload](ref: SchemaRef, payload: Payload) -> Payload:
    """Validate only after verifying the exact schema four-tuple."""

    return default_schema_registry().validate_pinned(ref, payload)


__all__ = [
    "DEFAULT_SCHEMA_CATALOG",
    "DEFAULT_SCHEMA_DIRECTORY",
    "JSON_SCHEMA_DIALECT",
    "SCHEMA_PUBLICATION_LOCK_FILENAME",
    "SCHEMA_PUBLICATION_MARKER_FILENAME",
    "SCHEMA_PUBLICATION_MARKER_FORMAT",
    "CompatibilityMode",
    "GoldenVectorCatalogEntry",
    "PayloadValidationError",
    "RegisteredGoldenVector",
    "RegisteredSchema",
    "RegisteredUpcaster",
    "SchemaAmbiguityError",
    "SchemaCatalog",
    "SchemaCatalogEntry",
    "SchemaCompatibilityError",
    "SchemaDefinitionError",
    "SchemaLifecycle",
    "SchemaNotFoundError",
    "SchemaPinMismatchError",
    "SchemaRef",
    "SchemaRegistry",
    "SchemaRegistryError",
    "SchemaValidationError",
    "SoftwareRange",
    "UpcasterCatalogEntry",
    "default_schema_registry",
    "deterministic_schema_artifact_id",
    "validate_payload",
    "validate_pinned_payload",
]
