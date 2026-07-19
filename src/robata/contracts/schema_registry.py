"""Offline registry and validation for checked-in Robata wire schemas."""

from __future__ import annotations

# mypy: disable-error-code=import-untyped
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, cast

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
_VERSION_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
_WIRE_VERSION_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
_SCHEMA_ID_PATTERN = r"^https://schemas\.robata\.dev/[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"
_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SCHEMA_ARTIFACT_ID_DOMAIN = b"robata-local-artifact-id-v1\x00JSON_SCHEMA\x00"
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


_ROBATA_FORMAT_CHECKER = FormatChecker()


@_ROBATA_FORMAT_CHECKER.checks("date-time")
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


def _is_strict_json_integer(_checker: object, instance: object) -> bool:
    return isinstance(instance, int) and not isinstance(instance, bool)


_STRICT_TYPE_CHECKER = Draft202012Validator.TYPE_CHECKER.redefine(
    "integer",
    _is_strict_json_integer,
)
_StrictDraft202012Validator = cast(
    type[Draft202012Validator],
    validators.extend(  # type: ignore[no-untyped-call]
        Draft202012Validator,
        type_checker=_STRICT_TYPE_CHECKER,
    ),
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


class UpcasterCatalogEntry(StrictModel):
    upcaster_id: _UpcasterId
    source: SchemaRef
    target: SchemaRef
    code_artifact_id: _ArtifactId
    code_sha256: _Digest
    runtime_artifact_id: _ArtifactId
    runtime_sha256: _Digest
    golden_vector_sha256: tuple[_Digest, ...]

    @model_validator(mode="after")
    def require_golden_vectors(self) -> UpcasterCatalogEntry:
        if not self.golden_vector_sha256:
            raise ValueError("at least one golden vector digest is required")
        if len(set(self.golden_vector_sha256)) != len(self.golden_vector_sha256):
            raise ValueError("golden vector digests must be unique")
        return self


class SchemaCatalog(StrictModel):
    catalog_version: Literal["1.0"]
    schemas: tuple[SchemaCatalogEntry, ...]
    upcasters: tuple[UpcasterCatalogEntry, ...] = ()


@dataclass(frozen=True)
class RegisteredSchema:
    entry: SchemaCatalogEntry
    path: Path
    document_bytes: bytes

    @property
    def ref(self) -> SchemaRef:
        return self.entry.ref


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
        self._registry: Registry[Any] = Registry().with_resources(resources)
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


def _load_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaDefinitionError(path.name, str(exc)) from exc
    if not isinstance(value, dict):
        raise SchemaDefinitionError(path.name, "document root must be an object")
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


def _resolve_artifact_path(root: Path, relative: str) -> Path:
    if "\\" in relative or Path(relative).is_absolute():
        raise SchemaDefinitionError(relative, "artifact_path must be a relative POSIX path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SchemaDefinitionError(relative, "artifact_path contains an unsafe path segment")
    resolved = (root / Path(*parts)).resolve()
    if not resolved.is_relative_to(root):
        raise SchemaDefinitionError(relative, "artifact_path escapes the catalog directory")
    return resolved


class _CatalogSchemaRegistry:
    """Catalog-driven, digest-pinned, offline multiversion schema registry."""

    def __init__(
        self,
        catalog: str | Path | None = None,
        *,
        software_version: str = DEFAULT_SOFTWARE_VERSION,
    ) -> None:
        _version_key(software_version)
        self.software_version = software_version
        self.catalog_path = _locate_catalog(catalog)
        self.schema_root = self.catalog_path.parent.resolve()
        catalog_raw, catalog_data = _load_json_object(self.catalog_path)
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
        self._load_catalog_entries()
        self._build_reference_registry()
        self._validate_catalog_metadata()
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
            self.resolve_exact(upcaster.target, require_software_support=False)
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
        discovered = {
            path.resolve()
            for version_dir in self.schema_root.glob("v*")
            if version_dir.is_dir()
            for path in version_dir.rglob(f"*{SCHEMA_FILE_SUFFIX}")
        }
        missing = sorted(
            path.relative_to(self.schema_root).as_posix() for path in discovered - cataloged
        )
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
        self._registry: Registry[Any] = Registry().with_resources(resources)

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
    ) -> Any:
        del software_version
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
    "CompatibilityMode",
    "PayloadValidationError",
    "RegisteredSchema",
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
