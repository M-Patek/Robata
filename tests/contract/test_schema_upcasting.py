from __future__ import annotations

import builtins
import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import (
    SchemaDefinitionError,
    SchemaRef,
    SchemaRegistry,
    SchemaValidationError,
    UpcasterCatalogEntry,
)
from robata.contracts.schema_upcasting import (
    UPCASTER_CHAIN_DIGEST_POLICY_VERSION,
    SchemaUpcasterGraph,
    UpcastAmbiguityError,
    UpcasterImplementation,
    UpcasterRegistrationError,
    UpcastPathNotFoundError,
    UpcastPurityError,
    upcaster_chain_digest,
)

LOGICAL_ID = "https://schemas.robata.dev/synthetic"

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "schema_upcasting"
FIXTURE_CATALOG = FIXTURE_ROOT / "schema-catalog.json"
FIXTURE_CODE = FIXTURE_ROOT / "upcasters" / "synthetic-v1-to-v2.py"
FIXTURE_RUNTIME = FIXTURE_ROOT / "runtime" / "synthetic-python-runtime.json"
FIXTURE_GOLDENS = (
    FIXTURE_ROOT / "golden" / "synthetic-v1-to-v2.input.json",
    FIXTURE_ROOT / "golden" / "synthetic-v1-to-v2.output.json",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _exact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        code_artifact_path=f"test-only/{upcaster_id}.py",
        code_sha256=f"{100 + number:064x}",
        runtime_artifact_id=f"20000000-0000-4000-8000-{number:012x}",
        runtime_artifact_path=f"test-only/{upcaster_id}.runtime.json",
        runtime_sha256=f"{200 + number:064x}",
        golden_vectors=(
            {
                "input_artifact_path": f"test-only/{upcaster_id}.input.json",
                "input_sha256": f"{300 + number:064x}",
                "output_artifact_path": f"test-only/{upcaster_id}.output.json",
                "output_sha256": f"{400 + number:064x}",
            },
        ),
    )


@dataclass
class _FakeRegistry:
    _schema_upcaster_test_double = True

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
    return SchemaUpcasterGraph._from_test_components(
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


def test_upcast_rejects_state_change_across_calls_even_when_each_pair_matches() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    registration = _edge("one-to-two", one, two, 1)
    calls = 0

    def varying_by_pair(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {**payload, "generation": (calls - 1) // 2}

    implementation = UpcasterImplementation(
        "one-to-two",
        registration.code_sha256,
        registration.runtime_sha256,
        varying_by_pair,
    )
    graph = _graph((one, two), (registration,), {"one-to-two": implementation})

    first = graph.upcast(one, two, {"version": 1})
    assert first.payload["generation"] == 0
    with pytest.raises(UpcastPurityError, match="changed output across calls"):
        graph.upcast(one, two, {"version": 1})


def test_chain_digest_binds_runtime_and_exact_registration() -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    registration = _edge("one-to-two", one, two, 1)
    changed_runtime = registration.model_copy(update={"runtime_sha256": "0" * 64})

    digest = upcaster_chain_digest(registration)

    assert digest != registration.code_sha256
    assert digest != upcaster_chain_digest(changed_runtime)


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
    assert projection.provenance.upcaster_chain_digest_policy_version == (
        UPCASTER_CHAIN_DIGEST_POLICY_VERSION
    )
    assert projection.provenance.upcaster_chain_digests == (upcaster_chain_digest(registration),)


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
        upcaster_chain_digest(first),
        upcaster_chain_digest(second),
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


@pytest.mark.parametrize("mutation", ["empty", "duplicate"])
def test_registration_requires_unique_golden_vector_pairs(mutation: str) -> None:
    one, two = _ref("1.0.0", 1), _ref("2.0.0", 2)
    valid = _edge("one-to-two", one, two, 1)
    document = valid.model_dump(mode="python")
    vectors = document["golden_vectors"]
    document["golden_vectors"] = () if mutation == "empty" else (*vectors, *vectors)

    with pytest.raises(ValidationError, match="golden vector"):
        UpcasterCatalogEntry.model_validate(document, strict=True)


def test_registry_backed_fixture_executes_pinned_golden_vector() -> None:
    registry = SchemaRegistry(FIXTURE_CATALOG)
    (registered,) = registry.registered_upcasters
    registration = registered.entry
    source = registration.source
    target = registration.target
    golden_input = _json_object(FIXTURE_GOLDENS[0])
    golden_output = _json_object(FIXTURE_GOLDENS[1])
    original = copy.deepcopy(golden_input)
    (vector,) = registration.golden_vectors
    assert _exact_sha256(FIXTURE_GOLDENS[0]) == vector.input_sha256
    assert _exact_sha256(FIXTURE_GOLDENS[1]) == vector.output_sha256
    assert _exact_sha256(FIXTURE_CODE) == registration.code_sha256
    assert _exact_sha256(FIXTURE_RUNTIME) == registration.runtime_sha256
    assert registered.code_bytes == FIXTURE_CODE.read_bytes()
    assert registered.runtime_bytes == FIXTURE_RUNTIME.read_bytes()
    registry.require_compatible(source, target)

    graph = SchemaUpcasterGraph(registry)
    first = graph.upcast(source, target, golden_input)
    second = graph.upcast(source, target, golden_input)

    assert golden_input == original
    assert first.payload == golden_output == second.payload
    assert first.provenance.source == source
    assert first.provenance.target == target
    assert first.provenance.upcaster_ids == (registration.upcaster_id,)
    assert first.provenance.upcaster_chain_digest_policy_version == (
        UPCASTER_CHAIN_DIGEST_POLICY_VERSION
    )
    assert first.provenance.upcaster_chain_digests == (upcaster_chain_digest(registration),)

    with_note = {**golden_input, "note": "preserve exactly"}
    with_note_projection = graph.upcast(source, target, with_note)
    assert with_note_projection.payload["record_key"] == with_note["record_key"]
    assert with_note_projection.payload["label"] == with_note["label"]
    assert with_note_projection.payload["note"] == with_note["note"]
    assert "review_annotation" not in with_note_projection.payload


def test_registered_execution_isolates_module_state_between_calls(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    upcaster = catalog["upcasters"][0]
    code_path = fixture_root / upcaster["code_artifact_path"]
    code_path.write_text(
        (
            "_calls = 0\n"
            "\n"
            "def upcast(payload):\n"
            "    global _calls\n"
            "    _calls += 1\n"
            "    generation = (_calls - 1) // 2\n"
            "    suffix = '' if generation == 0 else f'-{generation}'\n"
            "    return {**payload, 'schema_version': '2.0', "
            "'label': payload['label'] + suffix}\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    upcaster["code_sha256"] = _exact_sha256(code_path)
    catalog_path.write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    registry = SchemaRegistry(catalog_path)
    graph = SchemaUpcasterGraph(registry)
    (registration,) = registry.upcasters
    golden_input = _json_object(fixture_root / "golden" / "synthetic-v1-to-v2.input.json")
    golden_output = _json_object(fixture_root / "golden" / "synthetic-v1-to-v2.output.json")

    first = graph.upcast(registration.source, registration.target, golden_input)
    second = graph.upcast(registration.source, registration.target, golden_input)

    assert first.payload == golden_output == second.payload


def test_public_graph_golden_seal_rejects_process_state_across_calls(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    upcaster = catalog["upcasters"][0]
    code_path = fixture_root / upcaster["code_artifact_path"]
    state_name = "_robata_test_upcaster_cross_call_counter"
    code_path.write_text(
        (
            "import builtins\n"
            f"_STATE_NAME = {state_name!r}\n"
            "\n"
            "def upcast(payload):\n"
            "    count = getattr(builtins, _STATE_NAME, 0)\n"
            "    setattr(builtins, _STATE_NAME, count + 1)\n"
            "    generation = count // 2\n"
            "    suffix = '' if generation == 0 else f'-{generation}'\n"
            "    return {**payload, 'schema_version': '2.0', "
            "'label': payload['label'] + suffix}\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    upcaster["code_sha256"] = _exact_sha256(code_path)
    catalog_path.write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    golden_input = _json_object(fixture_root / "golden" / "synthetic-v1-to-v2.input.json")

    try:
        registry = SchemaRegistry(catalog_path)
        graph = SchemaUpcasterGraph(registry)
        (registration,) = registry.upcasters

        with pytest.raises(UpcastPurityError, match="changed output across calls"):
            graph.upcast(registration.source, registration.target, golden_input)
    finally:
        if hasattr(builtins, state_name):
            delattr(builtins, state_name)


def test_registry_backed_fixture_validates_source_and_target_payloads() -> None:
    registry = SchemaRegistry(FIXTURE_CATALOG)
    (registration,) = registry.upcasters
    golden_input = _json_object(FIXTURE_GOLDENS[0])
    graph = SchemaUpcasterGraph(registry)
    invalid_source = {key: value for key, value in golden_input.items() if key != "label"}

    with pytest.raises(SchemaValidationError, match="label"):
        graph.upcast(registration.source, registration.target, invalid_source)


def test_test_only_graph_construction_refuses_real_registry() -> None:
    registry = SchemaRegistry(FIXTURE_CATALOG)
    (registration,) = registry.upcasters

    with pytest.raises(UpcasterRegistrationError, match="requires an explicit fake"):
        SchemaUpcasterGraph._from_test_components(
            registry,
            registrations=(registration,),
        )


@pytest.mark.parametrize("keyword", ["registrations", "implementations"])
def test_public_graph_constructor_rejects_component_injection(keyword: str) -> None:
    registry = SchemaRegistry(FIXTURE_CATALOG)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SchemaUpcasterGraph(registry, **{keyword: ()})  # type: ignore[arg-type]


def test_public_graph_constructor_requires_catalog_backed_registry() -> None:
    fake = _FakeRegistry(set())

    with pytest.raises(UpcasterRegistrationError, match="requires a catalog-backed"):
        SchemaUpcasterGraph(fake)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "artifact_field",
    [
        "code_artifact_path",
        "runtime_artifact_path",
        "golden_input",
        "golden_output",
    ],
)
def test_registry_rejects_tampered_upcaster_artifact_bytes(
    tmp_path: Path,
    artifact_field: str,
) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog = _json_object(fixture_root / "schema-catalog.json")
    upcaster = catalog["upcasters"][0]
    if artifact_field == "golden_input":
        relative = upcaster["golden_vectors"][0]["input_artifact_path"]
    elif artifact_field == "golden_output":
        relative = upcaster["golden_vectors"][0]["output_artifact_path"]
    else:
        relative = upcaster[artifact_field]
    path = fixture_root / relative
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(SchemaDefinitionError, match="exact SHA-256 mismatch"):
        SchemaRegistry(fixture_root / "schema-catalog.json")


@pytest.mark.parametrize(
    "artifact_field",
    ["code_artifact_path", "runtime_artifact_path", "golden_input", "golden_output"],
)
def test_registry_rejects_unsafe_upcaster_artifact_paths(
    tmp_path: Path,
    artifact_field: str,
) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    upcaster = catalog["upcasters"][0]
    if artifact_field == "golden_input":
        upcaster["golden_vectors"][0]["input_artifact_path"] = "../input.json"
    elif artifact_field == "golden_output":
        upcaster["golden_vectors"][0]["output_artifact_path"] = "../output.json"
    else:
        upcaster[artifact_field] = "../artifact"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(SchemaDefinitionError, match="unsafe path segment"):
        SchemaRegistry(catalog_path)


@pytest.mark.parametrize("endpoint", ["input", "output"])
def test_registry_validates_golden_source_and_target_payloads(
    tmp_path: Path,
    endpoint: str,
) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    vector = catalog["upcasters"][0]["golden_vectors"][0]
    artifact_path = fixture_root / vector[f"{endpoint}_artifact_path"]
    payload = _json_object(artifact_path)
    payload.pop("label")
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    vector[f"{endpoint}_sha256"] = _exact_sha256(artifact_path)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="label"):
        SchemaRegistry(catalog_path)


def test_graph_rejects_a_pinned_but_incorrect_golden_output(tmp_path: Path) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    vector = catalog["upcasters"][0]["golden_vectors"][0]
    output_path = fixture_root / vector["output_artifact_path"]
    output = _json_object(output_path)
    output["label"] = "different-but-schema-valid"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    vector["output_sha256"] = _exact_sha256(output_path)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    registry = SchemaRegistry(catalog_path)

    with pytest.raises(UpcasterRegistrationError, match="golden output mismatch"):
        SchemaUpcasterGraph(registry)


def test_graph_executes_verified_code_bytes_without_rereading_the_path(tmp_path: Path) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    registry = SchemaRegistry(catalog_path)
    (registered,) = registry.registered_upcasters
    registered.code_path.write_text(
        "raise RuntimeError('unverified path bytes')\n",
        encoding="utf-8",
    )

    graph = SchemaUpcasterGraph(registry)
    golden_input = _json_object(FIXTURE_GOLDENS[0])
    projection = graph.upcast(registered.entry.source, registered.entry.target, golden_input)

    assert projection.payload["schema_version"] == "2.0"


def test_graph_rejects_pinned_code_without_the_required_callable(tmp_path: Path) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    upcaster = catalog["upcasters"][0]
    code_path = fixture_root / upcaster["code_artifact_path"]
    code_path.write_text("VALUE = 'no callable'\n", encoding="utf-8")
    upcaster["code_sha256"] = _exact_sha256(code_path)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    registry = SchemaRegistry(catalog_path)

    with pytest.raises(UpcasterRegistrationError, match="no callable 'upcast'"):
        SchemaUpcasterGraph(registry)


def test_fixture_catalog_rejects_upcaster_edge_without_declared_predecessor(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "schema_upcasting"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    catalog_path = fixture_root / "schema-catalog.json"
    catalog = _json_object(catalog_path)
    target = catalog["schemas"][1]
    target["compatibility_mode"] = "NONE"
    target["supported_predecessors"] = []
    catalog_path.write_text(
        json.dumps(catalog, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        SchemaDefinitionError,
        match="upcaster source must be a declared target predecessor",
    ):
        SchemaRegistry(catalog_path)
