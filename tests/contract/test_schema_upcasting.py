from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import (
    SchemaRef,
    SchemaRegistry,
    UpcasterCatalogEntry,
)
from robata.contracts.schema_upcasting import (
    SchemaUpcasterGraph,
    UpcastAmbiguityError,
    UpcasterImplementation,
    UpcasterRegistrationError,
    UpcastPathNotFoundError,
    UpcastPurityError,
)

LOGICAL_ID = "https://schemas.robata.dev/synthetic"


def _ref(version: str, number: int) -> SchemaRef:
    return SchemaRef(
        schema_id=LOGICAL_ID,
        version=version,
        artifact_id=f"00000000-0000-4000-8000-{number:012x}",
        sha256=f"{number:064x}",
    )


def _edge(
    upcaster_id: str,
    source: SchemaRef,
    target: SchemaRef,
    number: int,
) -> UpcasterCatalogEntry:
    return UpcasterCatalogEntry(
        upcaster_id=upcaster_id,
        source=source,
        target=target,
        code_artifact_id=f"10000000-0000-4000-8000-{number:012x}",
        code_sha256=f"{100 + number:064x}",
        runtime_artifact_id=f"20000000-0000-4000-8000-{number:012x}",
        runtime_sha256=f"{200 + number:064x}",
        golden_vector_sha256=(f"{300 + number:064x}",),
    )


@dataclass
class _FakeRegistry:
    refs: set[SchemaRef]

    @property
    def upcasters(self) -> tuple[UpcasterCatalogEntry, ...]:
        return ()

    def resolve_exact(
        self,
        ref: SchemaRef,
        *,
        require_software_support: bool = True,
    ) -> object:
        del require_software_support
        if ref not in self.refs:
            raise LookupError(ref)
        return object()

    def validate_pinned(self, ref: SchemaRef, payload: Any) -> Any:
        self.resolve_exact(ref)
        return payload


def _graph(
    refs: tuple[SchemaRef, ...],
    registrations: tuple[UpcasterCatalogEntry, ...],
    implementations: dict[str, UpcasterImplementation] | None = None,
) -> SchemaUpcasterGraph:
    registry: Any = _FakeRegistry(set(refs))
    return SchemaUpcasterGraph(
        registry,
        registrations=registrations,
        implementations=implementations,
    )


def test_production_v1_to_v2_has_stable_no_path() -> None:
    registry = SchemaRegistry()
    logical_id = "https://schemas.robata.dev/camera-video-export-manifest"
    source = registry.resolve_version(logical_id, "1.0.0").ref
    target = registry.resolve_version(logical_id, "2.0.0").ref

    with pytest.raises(
        UpcastPathNotFoundError,
        match=r"no registered upcast path.*1\.0\.0.*2\.0\.0",
    ):
        SchemaUpcasterGraph(registry).resolve_path(source, target)


def test_duplicate_edge_is_rejected() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)

    with pytest.raises(UpcasterRegistrationError, match="duplicate upcaster edge"):
        _graph(
            (one, two),
            (_edge("one-to-two", one, two, 1), _edge("also-one-to-two", one, two, 2)),
        )


def test_cycle_is_rejected() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)

    with pytest.raises(UpcasterRegistrationError, match="contains a cycle"):
        _graph(
            (one, two),
            (_edge("one-to-two", one, two, 1), _edge("two-to-one", two, one, 2)),
        )


def test_ambiguous_path_is_rejected() -> None:
    one = _ref("1.0.0", 1)
    two = _ref("2.0.0", 2)
    three = _ref("3.0.0", 3)
    four = _ref("4.0.0", 4)

    with pytest.raises(UpcastAmbiguityError, match="multiple upcaster paths"):
        _graph(
            (one, two, three, four),
            (
                _edge("one-to-two", one, two, 1),
                _edge("one-to-three", one, three, 2),
                _edge("two-to-four", two, four, 3),
                _edge("three-to-four", three, four, 4),
            ),
        )


def test_upcast_rejects_input_mutation() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    registration = _edge("one-to-two", one, two, 1)

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        payload["version"] = 2
        return payload

    implementation = UpcasterImplementation(
        "one-to-two",
        registration.code_sha256,
        registration.runtime_sha256,
        mutate,
    )
    graph = _graph((one, two), (registration,), {"one-to-two": implementation})

    with pytest.raises(UpcastPurityError, match="mutated its input"):
        graph.upcast(one, two, {"version": 1})


def test_upcast_rejects_nondeterminism() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    registration = _edge("one-to-two", one, two, 1)
    calls = 0

    def varying(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {**payload, "call": calls}

    implementation = UpcasterImplementation(
        "one-to-two",
        registration.code_sha256,
        registration.runtime_sha256,
        varying,
    )
    graph = _graph((one, two), (registration,), {"one-to-two": implementation})

    with pytest.raises(UpcastPurityError, match="nondeterministic"):
        graph.upcast(one, two, {"version": 1})


def test_upcast_returns_pinned_provenance_without_mutating_original() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    registration = _edge("one-to-two", one, two, 1)
    implementation = UpcasterImplementation(
        "one-to-two",
        registration.code_sha256,
        registration.runtime_sha256,
        lambda payload: {**payload, "version": 2},
    )
    graph = _graph((one, two), (registration,), {"one-to-two": implementation})
    original = {"version": 1}

    projection = graph.upcast(one, two, original)

    assert original == {"version": 1}
    assert projection.payload == {"version": 2}
    assert projection.provenance.original_digest == semantic_sha256(original)
    assert projection.provenance.source == one
    assert projection.provenance.target == two
    assert projection.provenance.upcaster_ids == ("one-to-two",)
    assert projection.provenance.upcaster_chain_digests == (registration.code_sha256,)


def test_multihop_path_preserves_ordered_chain_provenance() -> None:
    one = _ref("1.0.0", 1)
    two = _ref("2.0.0", 2)
    three = _ref("3.0.0", 3)
    first = _edge("one-to-two", one, two, 1)
    second = _edge("two-to-three", two, three, 2)
    implementations = {
        "one-to-two": UpcasterImplementation(
            "one-to-two",
            first.code_sha256,
            first.runtime_sha256,
            lambda payload: {**payload, "version": 2},
        ),
        "two-to-three": UpcasterImplementation(
            "two-to-three",
            second.code_sha256,
            second.runtime_sha256,
            lambda payload: {**payload, "version": 3},
        ),
    }
    graph = _graph((one, two, three), (first, second), implementations)

    projection = graph.upcast(one, three, {"version": 1})

    assert projection.payload == {"version": 3}
    assert projection.provenance.upcaster_ids == ("one-to-two", "two-to-three")
    assert projection.provenance.upcaster_chain_digests == (
        first.code_sha256,
        second.code_sha256,
    )


def test_implementation_code_and_runtime_pins_must_match_registration() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    registration = _edge("one-to-two", one, two, 1)
    wrong = UpcasterImplementation(
        "one-to-two",
        "0" * 64,
        registration.runtime_sha256,
        lambda payload: payload,
    )

    with pytest.raises(UpcasterRegistrationError, match="code digest mismatch"):
        _graph((one, two), (registration,), {"one-to-two": wrong})


@pytest.mark.parametrize("golden_digests", [(), ("a" * 64, "a" * 64)])
def test_registration_requires_unique_golden_vector_digests(
    golden_digests: tuple[str, ...],
) -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    valid = _edge("one-to-two", one, two, 1)

    with pytest.raises(ValidationError, match="golden vector"):
        UpcasterCatalogEntry.model_validate(
            {
                **valid.model_dump(mode="python"),
                "golden_vector_sha256": golden_digests,
            },
            strict=True,
        )
