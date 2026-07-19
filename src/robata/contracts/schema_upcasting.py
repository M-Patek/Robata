"""Deterministic, fail-closed schema upcasting over catalog-pinned versions."""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, UpcasterCatalogEntry

type JsonObject = dict[str, Any]
type UpcasterTransform = Callable[[JsonObject], JsonObject]


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
    upcaster_chain_digests: tuple[str, ...]


@dataclass(frozen=True)
class UpcastProjection:
    payload: JsonObject
    provenance: UpcastProvenance


class SchemaUpcasterGraph:
    """Validate and execute the catalog's single-path deterministic upcaster graph."""

    def __init__(
        self,
        registry: SchemaRegistry,
        *,
        registrations: Iterable[UpcasterCatalogEntry] | None = None,
        implementations: Mapping[str, UpcasterImplementation] | None = None,
    ) -> None:
        self.registry = registry
        self.registrations = tuple(registry.upcasters if registrations is None else registrations)
        self.implementations = dict(implementations or {})
        self._by_edge: dict[tuple[SchemaRef, SchemaRef], UpcasterCatalogEntry] = {}
        self._outgoing: dict[SchemaRef, tuple[UpcasterCatalogEntry, ...]] = {}
        self._validate_and_index()

    def _validate_and_index(self) -> None:
        ids: set[str] = set()
        outgoing: dict[SchemaRef, list[UpcasterCatalogEntry]] = defaultdict(list)
        nodes: set[SchemaRef] = set()
        for registration in self.registrations:
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

        unknown = sorted(set(self.implementations) - ids)
        if unknown:
            raise UpcasterRegistrationError(
                f"implementations have no catalog registration: {unknown!r}"
            )
        by_id = {item.upcaster_id: item for item in self.registrations}
        for upcaster_id, implementation in self.implementations.items():
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
            try:
                implementation = self.implementations[registration.upcaster_id]
            except KeyError as exc:
                raise UpcasterRegistrationError(
                    f"no implementation for upcaster {registration.upcaster_id}"
                ) from exc
            first_input = copy.deepcopy(current)
            second_input = copy.deepcopy(current)
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
            current = copy.deepcopy(first)
            chain_ids.append(registration.upcaster_id)
            chain_digests.append(registration.code_sha256)

        return UpcastProjection(
            payload=current,
            provenance=UpcastProvenance(
                original_digest=original_digest,
                source=source,
                target=target,
                upcaster_ids=tuple(chain_ids),
                upcaster_chain_digests=tuple(chain_digests),
            ),
        )


__all__ = [
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
]
