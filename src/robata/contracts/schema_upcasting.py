"""Deterministic, fail-closed schema upcasting over catalog-pinned versions."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, cast

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import (
    RegisteredUpcaster,
    SchemaRef,
    SchemaRegistry,
    UpcasterCatalogEntry,
)

type JsonObject = dict[str, Any]
type UpcasterTransform = Callable[[JsonObject], JsonObject]

UPCASTER_CHAIN_DIGEST_POLICY_VERSION: Literal["upcaster-chain-digest-policy-v1"] = (
    "upcaster-chain-digest-policy-v1"
)


class SchemaUpcastingError(RuntimeError):
    """Base class for upcaster graph and execution failures."""


class UpcasterRegistrationError(SchemaUpcastingError):
    """Raised when the registered graph or implementation set is invalid."""


class UpcastPathNotFoundError(SchemaUpcastingError, LookupError):
    """Raised when no approved directed upcast path exists."""


class UpcastAmbiguityError(UpcasterRegistrationError):
    """Raised when more than one approved path connects a version pair."""


class UpcastPurityError(SchemaUpcastingError):
    """Raised when an upcaster mutates input or produces nondeterministic output."""


@dataclass(frozen=True)
class UpcasterImplementation:
    """A callable bound to the immutable code and runtime pins in the catalog."""

    upcaster_id: str
    code_sha256: str
    runtime_sha256: str
    transform: UpcasterTransform


@dataclass(frozen=True)
class UpcastProvenance:
    original_digest: str
    source: SchemaRef
    target: SchemaRef
    upcaster_ids: tuple[str, ...]
    upcaster_chain_digest_policy_version: Literal["upcaster-chain-digest-policy-v1"]
    upcaster_chain_digests: tuple[str, ...]


@dataclass(frozen=True)
class UpcastProjection:
    payload: JsonObject
    provenance: UpcastProvenance


class _RegistryPort(Protocol):
    def resolve_exact(
        self,
        ref: SchemaRef,
        *,
        require_software_support: bool = True,
    ) -> object: ...

    def validate_pinned(self, ref: SchemaRef, payload: Any) -> Any: ...


def _decode_golden_object(raw: bytes, source: str) -> JsonObject:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpcasterRegistrationError(f"invalid golden JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise UpcasterRegistrationError(f"golden JSON root must be an object: {source}")
    return value


def _load_registered_implementation(registered: RegisteredUpcaster) -> UpcasterImplementation:
    entry = registered.entry
    try:
        source = registered.code_bytes.decode("utf-8")
        code = compile(source, entry.code_artifact_path, "exec")
    except (UnicodeError, SyntaxError) as exc:
        raise UpcasterRegistrationError(
            f"cannot compile pinned upcaster code {entry.upcaster_id}: {exc}"
        ) from exc

    def instantiate_transform() -> UpcasterTransform:
        namespace: dict[str, Any] = {
            "__file__": str(registered.code_path),
            "__name__": f"_robata_upcaster_{entry.upcaster_id.replace('-', '_')}",
        }
        try:
            exec(code, namespace)
        except Exception as exc:
            raise UpcasterRegistrationError(
                f"cannot load pinned upcaster code {entry.upcaster_id}: {exc}"
            ) from exc
        transform = namespace.get("upcast")
        if not callable(transform):
            raise UpcasterRegistrationError(
                f"pinned upcaster code has no callable 'upcast': {entry.upcaster_id}"
            )
        return cast(UpcasterTransform, transform)

    instantiate_transform()

    def isolated_transform(payload: JsonObject) -> JsonObject:
        return instantiate_transform()(payload)

    return UpcasterImplementation(
        upcaster_id=entry.upcaster_id,
        code_sha256=entry.code_sha256,
        runtime_sha256=entry.runtime_sha256,
        transform=isolated_transform,
    )


def upcaster_chain_digest(registration: UpcasterCatalogEntry) -> str:
    """Bind one chain element to its versioned schemas, code, and runtime."""

    return semantic_sha256(
        {
            "digest_policy_version": UPCASTER_CHAIN_DIGEST_POLICY_VERSION,
            "upcaster_id": registration.upcaster_id,
            "source": registration.source.model_dump(mode="json"),
            "target": registration.target.model_dump(mode="json"),
            "code_sha256": registration.code_sha256,
            "runtime_sha256": registration.runtime_sha256,
        }
    )


class SchemaUpcasterGraph:
    """Validate and execute the catalog's single-path deterministic upcaster graph."""

    def __init__(
        self,
        registry: SchemaRegistry,
    ) -> None:
        if not isinstance(registry, SchemaRegistry):
            raise UpcasterRegistrationError(
                "SchemaUpcasterGraph requires a catalog-backed SchemaRegistry"
            )
        implementations = {
            registered.entry.upcaster_id: _load_registered_implementation(registered)
            for registered in registry.registered_upcasters
        }
        self._initialize(registry, registry.upcasters, implementations)
        self._verify_registered_goldens(registry.registered_upcasters)

    @classmethod
    def _from_test_components(
        cls,
        registry: _RegistryPort,
        *,
        registrations: tuple[UpcasterCatalogEntry, ...],
        implementations: Mapping[str, UpcasterImplementation] | None = None,
    ) -> Self:
        """Build arbitrary graph mechanics only for tests using a fake registry."""

        if getattr(registry, "_schema_upcaster_test_double", False) is not True:
            raise UpcasterRegistrationError(
                "test-only graph construction requires an explicit fake registry"
            )
        graph = cls.__new__(cls)
        graph._initialize(registry, registrations, implementations or {})
        return graph

    def _initialize(
        self,
        registry: _RegistryPort,
        registrations: tuple[UpcasterCatalogEntry, ...],
        implementations: Mapping[str, UpcasterImplementation],
    ) -> None:
        self.registry = registry
        self._registrations = registrations
        self._implementations: Mapping[str, UpcasterImplementation] = MappingProxyType(
            dict(implementations)
        )
        self._by_edge: dict[tuple[SchemaRef, SchemaRef], UpcasterCatalogEntry] = {}
        self._outgoing: dict[SchemaRef, tuple[UpcasterCatalogEntry, ...]] = {}
        self._determinism_seals: dict[tuple[str, str], str] = {}
        self._execution_lock = RLock()
        self._validate_and_index()

    def _check_determinism_seal(
        self,
        registration: UpcasterCatalogEntry,
        payload: JsonObject,
        output: JsonObject,
    ) -> None:
        key = (registration.upcaster_id, semantic_sha256(payload))
        output_digest = semantic_sha256(output)
        expected_digest = self._determinism_seals.setdefault(key, output_digest)
        if output_digest != expected_digest:
            raise UpcastPurityError(
                f"upcaster {registration.upcaster_id} changed output across calls "
                "for the same canonical input"
            )

    def _validate_and_index(self) -> None:
        ids: set[str] = set()
        outgoing: dict[SchemaRef, list[UpcasterCatalogEntry]] = defaultdict(list)
        nodes: set[SchemaRef] = set()
        for registration in self._registrations:
            self.registry.resolve_exact(registration.source, require_software_support=False)
            self.registry.resolve_exact(registration.target, require_software_support=False)
            if registration.source.schema_id != registration.target.schema_id:
                raise UpcasterRegistrationError("upcaster source and target must share schema_id")
            if registration.upcaster_id in ids:
                raise UpcasterRegistrationError(
                    f"duplicate upcaster_id: {registration.upcaster_id}"
                )
            edge = (registration.source, registration.target)
            if edge in self._by_edge:
                raise UpcasterRegistrationError(
                    "duplicate upcaster edge: "
                    f"{registration.source.version}->{registration.target.version}"
                )
            ids.add(registration.upcaster_id)
            self._by_edge[edge] = registration
            outgoing[registration.source].append(registration)
            nodes.update(edge)

        self._outgoing = {
            source: tuple(sorted(edges, key=lambda edge: edge.upcaster_id))
            for source, edges in outgoing.items()
        }
        self._reject_cycles(nodes)
        self._reject_ambiguous_paths(nodes)

        unknown = sorted(set(self._implementations) - ids)
        if unknown:
            raise UpcasterRegistrationError(
                f"implementations have no catalog registration: {unknown!r}"
            )
        by_id = {item.upcaster_id: item for item in self._registrations}
        for upcaster_id, implementation in self._implementations.items():
            if implementation.upcaster_id != upcaster_id:
                raise UpcasterRegistrationError(
                    f"implementation key does not match upcaster_id: {upcaster_id}"
                )
            registration = by_id[upcaster_id]
            if implementation.code_sha256 != registration.code_sha256:
                raise UpcasterRegistrationError(f"code digest mismatch for upcaster {upcaster_id}")
            if implementation.runtime_sha256 != registration.runtime_sha256:
                raise UpcasterRegistrationError(
                    f"runtime digest mismatch for upcaster {upcaster_id}"
                )

    def _verify_registered_goldens(
        self,
        registered_upcasters: tuple[RegisteredUpcaster, ...],
    ) -> None:
        for registered in registered_upcasters:
            registration = registered.entry
            for vector in registered.golden_vectors:
                golden_input = _decode_golden_object(
                    vector.input_bytes,
                    vector.entry.input_artifact_path,
                )
                golden_output = _decode_golden_object(
                    vector.output_bytes,
                    vector.entry.output_artifact_path,
                )
                self.registry.validate_pinned(registration.source, golden_input)
                self.registry.validate_pinned(registration.target, golden_output)
                actual = self._execute_registration(registration, golden_input)
                if canonical_json_bytes(actual) != canonical_json_bytes(golden_output):
                    raise UpcasterRegistrationError(
                        "golden output mismatch for upcaster "
                        f"{registration.upcaster_id}: {vector.entry.output_artifact_path}"
                    )

    def _execute_registration(
        self,
        registration: UpcasterCatalogEntry,
        payload: JsonObject,
    ) -> JsonObject:
        try:
            implementation = self._implementations[registration.upcaster_id]
        except KeyError as exc:
            raise UpcasterRegistrationError(
                f"no implementation for upcaster {registration.upcaster_id}"
            ) from exc
        with self._execution_lock:
            first_input = copy.deepcopy(payload)
            second_input = copy.deepcopy(payload)
            first_before = canonical_json_bytes(first_input)
            second_before = canonical_json_bytes(second_input)
            first = implementation.transform(first_input)
            second = implementation.transform(second_input)
            if canonical_json_bytes(first_input) != first_before:
                raise UpcastPurityError(f"upcaster {registration.upcaster_id} mutated its input")
            if canonical_json_bytes(second_input) != second_before:
                raise UpcastPurityError(f"upcaster {registration.upcaster_id} mutated its input")
            if not isinstance(first, dict) or not isinstance(second, dict):
                raise UpcastPurityError(
                    f"upcaster {registration.upcaster_id} must return a JSON object"
                )
            if canonical_json_bytes(first) != canonical_json_bytes(second):
                raise UpcastPurityError(f"upcaster {registration.upcaster_id} is nondeterministic")
            self.registry.validate_pinned(registration.target, first)
            self._check_determinism_seal(registration, payload, first)
            return copy.deepcopy(first)

    def _reject_cycles(self, nodes: set[SchemaRef]) -> None:
        state: dict[SchemaRef, int] = {}

        def visit(node: SchemaRef) -> None:
            marker = state.get(node, 0)
            if marker == 1:
                raise UpcasterRegistrationError(
                    f"upcaster graph contains a cycle at {node.schema_id}@{node.version}"
                )
            if marker == 2:
                return
            state[node] = 1
            for edge in self._outgoing.get(node, ()):
                visit(edge.target)
            state[node] = 2

        for node in sorted(nodes, key=lambda ref: ref.key):
            visit(node)

    def _reject_ambiguous_paths(self, nodes: set[SchemaRef]) -> None:
        for source in sorted(nodes, key=lambda ref: ref.key):
            path_counts: dict[SchemaRef, int] = {source: 1}
            queue: deque[SchemaRef] = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self._outgoing.get(node, ()):
                    previous = path_counts.get(edge.target, 0)
                    path_counts[edge.target] = min(2, previous + path_counts[node])
                    if previous == 0:
                        queue.append(edge.target)
            ambiguous = sorted(
                (target for target, count in path_counts.items() if count > 1),
                key=lambda ref: ref.key,
            )
            if ambiguous:
                target = ambiguous[0]
                raise UpcastAmbiguityError(
                    "multiple upcaster paths connect "
                    f"{source.schema_id}@{source.version} to "
                    f"{target.schema_id}@{target.version}"
                )

    def resolve_path(
        self,
        source: SchemaRef,
        target: SchemaRef,
    ) -> tuple[UpcasterCatalogEntry, ...]:
        self.registry.resolve_exact(source, require_software_support=False)
        self.registry.resolve_exact(target, require_software_support=False)
        if source == target:
            return ()

        queue: deque[tuple[SchemaRef, tuple[UpcasterCatalogEntry, ...]]] = deque([(source, ())])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            for edge in self._outgoing.get(node, ()):
                candidate = (*path, edge)
                if edge.target == target:
                    return candidate
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, candidate))
        raise UpcastPathNotFoundError(
            "no registered upcast path from "
            f"{source.schema_id}@{source.version} to "
            f"{target.schema_id}@{target.version}"
        )

    def upcast(
        self,
        source: SchemaRef,
        target: SchemaRef,
        payload: JsonObject,
    ) -> UpcastProjection:
        path = self.resolve_path(source, target)
        self.registry.validate_pinned(source, payload)
        original_digest = semantic_sha256(payload)
        current = copy.deepcopy(payload)
        chain_ids: list[str] = []
        chain_digests: list[str] = []

        for registration in path:
            current = self._execute_registration(registration, current)
            chain_ids.append(registration.upcaster_id)
            chain_digests.append(upcaster_chain_digest(registration))

        return UpcastProjection(
            payload=current,
            provenance=UpcastProvenance(
                original_digest=original_digest,
                source=source,
                target=target,
                upcaster_ids=tuple(chain_ids),
                upcaster_chain_digest_policy_version=(UPCASTER_CHAIN_DIGEST_POLICY_VERSION),
                upcaster_chain_digests=tuple(chain_digests),
            ),
        )


__all__ = [
    "UPCASTER_CHAIN_DIGEST_POLICY_VERSION",
    "JsonObject",
    "SchemaUpcasterGraph",
    "SchemaUpcastingError",
    "UpcastAmbiguityError",
    "UpcastPathNotFoundError",
    "UpcastProjection",
    "UpcastProvenance",
    "UpcastPurityError",
    "UpcasterImplementation",
    "UpcasterRegistrationError",
    "UpcasterTransform",
    "upcaster_chain_digest",
]
