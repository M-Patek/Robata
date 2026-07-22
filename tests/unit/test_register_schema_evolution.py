from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.register_schema as schema_registration
import scripts.register_schema_evolution as evolution_registration
from robata.contracts.schema_registry import (
    SCHEMA_PUBLICATION_MARKER_FILENAME,
    SchemaPinMismatchError,
    SchemaRef,
    SchemaRegistry,
    deterministic_schema_artifact_id,
)
from robata.contracts.schema_upcasting import SchemaUpcasterGraph, UpcasterRegistrationError
from scripts.register_schema import PublishedSchemaConflictError, register_schema
from scripts.register_schema_evolution import (
    SchemaEvolutionRegistrationError,
    register_schema_evolution,
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: dict[str, Any], *, crlf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    path.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)


def _schema(version_directory: str, wire_version: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (f"https://schemas.robata.dev/{version_directory}/synthetic-evolution.schema.json"),
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": wire_version},
            "value": {"type": "integer"},
        },
        "required": ["schema_version", "value"],
        "title": "Synthetic evolution",
        "type": "object",
    }


def _schema_entry(
    *,
    version: str,
    wire_version: str,
    artifact_path: str,
    raw: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "ref": {
            "schema_id": "https://schemas.robata.dev/synthetic-evolution",
            "version": version,
            "artifact_id": deterministic_schema_artifact_id(digest),
            "sha256": digest,
        },
        "wire_version": wire_version,
        "document_id": f"https://schemas.robata.dev/{artifact_path}",
        "artifact_path": artifact_path,
        "owner": "test",
        "canonicalization_version": "rfc8785-v1",
        "projection_version": f"synthetic-evolution-v{wire_version[0]}",
        "compatibility_mode": "NONE",
        "lifecycle": "ACTIVE",
        "supported_software": {
            "min_inclusive": "0.1.0",
            "max_exclusive": "0.2.0",
        },
        "supported_predecessors": [],
    }


def _bundle_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], SchemaRef]:
    schema_root = tmp_path / "schemas"
    source_document = _schema("v1", "1.0")
    source_bytes = _json_bytes(source_document)
    source_path = schema_root / "v1" / "synthetic-evolution.schema.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_bytes)
    source_entry = _schema_entry(
        version="1.0.0",
        wire_version="1.0",
        artifact_path="v1/synthetic-evolution.schema.json",
        raw=source_bytes,
    )
    catalog_path = schema_root / "schema-catalog.json"
    catalog_path.write_bytes(
        _json_bytes(
            {
                "catalog_version": "1.0",
                "schemas": [source_entry],
                "upcasters": [],
            }
        )
    )
    SchemaRegistry(catalog_path)

    bundle_root = tmp_path / "evolution"
    target_candidate = bundle_root / "candidates" / "target.json"
    code_candidate = bundle_root / "candidates" / "upcast.py"
    runtime_candidate = bundle_root / "candidates" / "runtime.json"
    input_candidate = bundle_root / "candidates" / "golden.input.json"
    output_candidate = bundle_root / "candidates" / "golden.output.json"
    _write_json(target_candidate, _schema("v2", "2.0"), crlf=True)
    code_candidate.write_bytes(
        b"def upcast(payload):\r\n    return {**payload, 'schema_version': '2.0'}\r\n"
    )
    _write_json(
        runtime_candidate,
        {
            "runtime": "python",
            "contract": "robata-schema-upcaster-v1",
            "clock": False,
            "database": False,
            "network": False,
            "randomness": False,
        },
        crlf=True,
    )
    _write_json(input_candidate, {"schema_version": "1.0", "value": 7}, crlf=True)
    _write_json(output_candidate, {"schema_version": "2.0", "value": 7}, crlf=True)

    source_ref = SchemaRef.model_validate(source_entry["ref"], strict=True)
    bundle = {
        "format_version": "1.0",
        "target": {
            "candidate": "candidates/target.json",
            "schema_id": "https://schemas.robata.dev/synthetic-evolution",
            "version": "2.0.0",
            "wire_version": "2.0",
            "artifact_path": "v2/synthetic-evolution.schema.json",
            "owner": "test",
            "projection_version": "synthetic-evolution-v2",
            "canonicalization_version": "rfc8785-v1",
            "software": {
                "min_inclusive": "0.1.0",
                "max_exclusive": "0.2.0",
            },
        },
        "upcasters": [
            {
                "upcaster_id": "synthetic-evolution-v1-to-v2",
                "source": source_ref.model_dump(mode="json"),
                "code": {
                    "candidate": "candidates/upcast.py",
                    "artifact_path": "upcasters/synthetic-evolution-v1-to-v2.py",
                },
                "runtime": {
                    "candidate": "candidates/runtime.json",
                    "artifact_path": "runtime/synthetic-evolution-python.json",
                },
                "golden_vectors": [
                    {
                        "input": {
                            "candidate": "candidates/golden.input.json",
                            "artifact_path": ("golden/synthetic-evolution-v1-to-v2.input.json"),
                        },
                        "output": {
                            "candidate": "candidates/golden.output.json",
                            "artifact_path": ("golden/synthetic-evolution-v1-to-v2.output.json"),
                        },
                    }
                ],
            }
        ],
    }
    bundle_path = bundle_root / "bundle.json"
    _write_json(bundle_path, bundle)
    return catalog_path, bundle_path, bundle, source_ref


def _artifact_paths(bundle: dict[str, Any]) -> tuple[str, ...]:
    target = bundle["target"]
    paths = [target["artifact_path"]]
    for edge in bundle["upcasters"]:
        paths.extend((edge["code"]["artifact_path"], edge["runtime"]["artifact_path"]))
        for vector in edge["golden_vectors"]:
            paths.extend(
                (
                    vector["input"]["artifact_path"],
                    vector["output"]["artifact_path"],
                )
            )
    return tuple(sorted(set(paths)))


def test_dry_run_validates_complete_bundle_without_writing(tmp_path: Path) -> None:
    catalog, bundle_path, bundle, _source_ref = _bundle_fixture(tmp_path)
    catalog_before = catalog.read_bytes()

    result = register_schema_evolution(bundle_path, catalog_path=catalog, dry_run=True)

    assert result.changed is True
    assert result.dry_run is True
    assert catalog.read_bytes() == catalog_before
    assert result.artifact_paths == _artifact_paths(bundle)
    assert all(not (catalog.parent / path).exists() for path in result.artifact_paths)
    assert SchemaRegistry(catalog).upcasters == ()


def test_cli_help_runs_outside_repository_working_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "register_schema_evolution.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "bundle" in completed.stdout


def test_publication_and_exact_replay_are_idempotent(tmp_path: Path) -> None:
    catalog, bundle_path, _bundle, source_ref = _bundle_fixture(tmp_path)

    first = register_schema_evolution(bundle_path, catalog_path=catalog)
    catalog_after_first = catalog.read_bytes()
    registry = SchemaRegistry(catalog)
    (upcaster,) = registry.upcasters
    graph = SchemaUpcasterGraph(registry)
    projection = graph.upcast(source_ref, first.target_ref, {"schema_version": "1.0", "value": 9})

    assert first.changed is True
    assert first.dry_run is False
    assert upcaster.target == first.target_ref
    assert registry.resolve_exact(first.target_ref).entry.compatibility_mode.value == "BACKWARD"
    assert registry.resolve_exact(first.target_ref).entry.supported_predecessors == (source_ref,)
    assert projection.payload == {"schema_version": "2.0", "value": 9}
    assert b"\r\n" not in (catalog.parent / upcaster.code_artifact_path).read_bytes()

    replay = register_schema_evolution(bundle_path, catalog_path=catalog)

    assert replay == first.__class__(
        target_ref=first.target_ref,
        upcaster_ids=first.upcaster_ids,
        artifact_paths=first.artifact_paths,
        changed=False,
        dry_run=False,
    )
    assert catalog.read_bytes() == catalog_after_first


def test_golden_mismatch_fails_before_publication(tmp_path: Path) -> None:
    catalog, bundle_path, bundle, _source_ref = _bundle_fixture(tmp_path)
    catalog_before = catalog.read_bytes()
    output = bundle_path.parent / "candidates" / "golden.output.json"
    _write_json(output, {"schema_version": "2.0", "value": 8})

    with pytest.raises(UpcasterRegistrationError, match="golden output mismatch"):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert all(not (catalog.parent / path).exists() for path in _artifact_paths(bundle))


def test_unknown_source_exact_pin_fails_before_publication(tmp_path: Path) -> None:
    catalog, bundle_path, bundle, _source_ref = _bundle_fixture(tmp_path)
    catalog_before = catalog.read_bytes()
    mutated = copy.deepcopy(bundle)
    mutated["upcasters"][0]["source"]["sha256"] = "0" * 64
    _write_json(bundle_path, mutated)

    with pytest.raises(SchemaPinMismatchError):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert all(not (catalog.parent / path).exists() for path in _artifact_paths(bundle))


def test_existing_none_target_cannot_be_retrofitted(tmp_path: Path) -> None:
    catalog, bundle_path, bundle, _source_ref = _bundle_fixture(tmp_path)
    target_candidate = bundle_path.parent / bundle["target"]["candidate"]
    register_schema(
        target_candidate,
        catalog_path=catalog,
        schema_id=bundle["target"]["schema_id"],
        version=bundle["target"]["version"],
        wire_version=bundle["target"]["wire_version"],
        owner=bundle["target"]["owner"],
        projection_version=bundle["target"]["projection_version"],
        artifact_path=bundle["target"]["artifact_path"],
        canonicalization_version=bundle["target"]["canonicalization_version"],
        software_min=bundle["target"]["software"]["min_inclusive"],
        software_max=bundle["target"]["software"]["max_exclusive"],
    )
    catalog_before = catalog.read_bytes()

    with pytest.raises(PublishedSchemaConflictError, match="different metadata"):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert SchemaRegistry(catalog).upcasters == ()


def test_artifact_replace_failure_rolls_back_entire_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, bundle_path, bundle, _source_ref = _bundle_fixture(tmp_path)
    catalog_before = catalog.read_bytes()
    failed_destination = catalog.parent / bundle["upcasters"][0]["code"]["artifact_path"]
    real_replace = schema_registration._atomic_replace
    failed = False

    def fail_code_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination) == failed_destination and not failed:
            failed = True
            raise OSError("simulated upcaster artifact replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(schema_registration, "_atomic_replace", fail_code_replace)

    with pytest.raises(OSError, match="simulated upcaster artifact replace failure"):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert all(not (catalog.parent / path).exists() for path in _artifact_paths(bundle))
    assert not (catalog.parent / SCHEMA_PUBLICATION_MARKER_FILENAME).exists()
    assert SchemaRegistry(catalog).upcasters == ()


def test_pre_catalog_hard_crash_resumes_exact_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, bundle_path, _bundle, _source_ref = _bundle_fixture(tmp_path)
    catalog_before = catalog.read_bytes()
    marker = catalog.parent / SCHEMA_PUBLICATION_MARKER_FILENAME
    real_publish = evolution_registration._publish_artifacts

    class SimulatedHardCrash(BaseException):
        pass

    def leave_partial_bundle(**arguments: Any) -> None:
        artifacts = arguments["artifacts"]
        marker.write_bytes(
            schema_registration._publication_bundle_marker_bytes(
                artifacts=artifacts,
                original_catalog=arguments["original_catalog"],
                new_catalog=arguments["new_catalog"],
            )
        )
        selected = (next(item for item in artifacts if item.role == "SCHEMA"), artifacts[0])
        for artifact in dict.fromkeys(selected):
            destination = catalog.parent / artifact.artifact_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(artifact.contents)
        raise SimulatedHardCrash

    monkeypatch.setattr(evolution_registration, "_publish_artifacts", leave_partial_bundle)
    with pytest.raises(SimulatedHardCrash):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert marker.is_file()
    assert SchemaRegistry(catalog).upcasters == ()

    monkeypatch.setattr(evolution_registration, "_publish_artifacts", real_publish)
    resumed = register_schema_evolution(bundle_path, catalog_path=catalog)

    assert resumed.changed is True
    assert not marker.exists()
    assert len(SchemaRegistry(catalog).upcasters) == 1


def test_post_catalog_hard_crash_recovers_as_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, bundle_path, _bundle, _source_ref = _bundle_fixture(tmp_path)
    marker = catalog.parent / SCHEMA_PUBLICATION_MARKER_FILENAME
    real_publish = evolution_registration._publish_artifacts

    class SimulatedHardCrash(BaseException):
        pass

    def leave_committed_bundle(**arguments: Any) -> None:
        artifacts = arguments["artifacts"]
        marker.write_bytes(
            schema_registration._publication_bundle_marker_bytes(
                artifacts=artifacts,
                original_catalog=arguments["original_catalog"],
                new_catalog=arguments["new_catalog"],
            )
        )
        for artifact in artifacts:
            destination = catalog.parent / artifact.artifact_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(artifact.contents)
        catalog.write_bytes(arguments["new_catalog"])
        raise SimulatedHardCrash

    monkeypatch.setattr(evolution_registration, "_publish_artifacts", leave_committed_bundle)
    with pytest.raises(SimulatedHardCrash):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert marker.is_file()
    assert len(SchemaRegistry(catalog).upcasters) == 1

    monkeypatch.setattr(evolution_registration, "_publish_artifacts", real_publish)
    replay = register_schema_evolution(bundle_path, catalog_path=catalog)

    assert replay.changed is False
    assert not marker.exists()
    assert len(SchemaRegistry(catalog).upcasters) == 1


def test_multiple_incoming_edges_share_one_runtime_artifact(tmp_path: Path) -> None:
    catalog, bundle_path, bundle, source_v1 = _bundle_fixture(tmp_path)
    source_document = _schema("v0", "0.5")
    source_bytes = _json_bytes(source_document)
    source_path = catalog.parent / "v0" / "synthetic-evolution.schema.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_bytes)
    source_entry = _schema_entry(
        version="0.5.0",
        wire_version="0.5",
        artifact_path="v0/synthetic-evolution.schema.json",
        raw=source_bytes,
    )
    catalog_document = json.loads(catalog.read_text(encoding="utf-8"))
    catalog_document["schemas"].append(source_entry)
    catalog.write_bytes(_json_bytes(catalog_document))
    source_v0 = SchemaRef.model_validate(source_entry["ref"], strict=True)

    _write_json(
        bundle_path.parent / "candidates" / "golden-v0.input.json",
        {"schema_version": "0.5", "value": 11},
    )
    _write_json(
        bundle_path.parent / "candidates" / "golden-v0.output.json",
        {"schema_version": "2.0", "value": 11},
    )
    second_edge = copy.deepcopy(bundle["upcasters"][0])
    second_edge["upcaster_id"] = "synthetic-evolution-v0-to-v2"
    second_edge["source"] = source_v0.model_dump(mode="json")
    second_edge["code"]["artifact_path"] = "upcasters/synthetic-evolution-v0-to-v2.py"
    second_edge["golden_vectors"] = [
        {
            "input": {
                "candidate": "candidates/golden-v0.input.json",
                "artifact_path": "golden/synthetic-evolution-v0-to-v2.input.json",
            },
            "output": {
                "candidate": "candidates/golden-v0.output.json",
                "artifact_path": "golden/synthetic-evolution-v0-to-v2.output.json",
            },
        }
    ]
    bundle["upcasters"].append(second_edge)
    _write_json(bundle_path, bundle)

    result = register_schema_evolution(bundle_path, catalog_path=catalog)

    assert result.upcaster_ids == (
        "synthetic-evolution-v0-to-v2",
        "synthetic-evolution-v1-to-v2",
    )
    assert result.artifact_paths.count("runtime/synthetic-evolution-python.json") == 1
    registry = SchemaRegistry(catalog)
    assert len(registry.upcasters) == 2
    assert registry.resolve_exact(result.target_ref).entry.supported_predecessors == (
        source_v0,
        source_v1,
    )
    projection = SchemaUpcasterGraph(registry).upcast(
        source_v0,
        result.target_ref,
        {"schema_version": "0.5", "value": 13},
    )
    assert projection.payload == {"schema_version": "2.0", "value": 13}


def test_runtime_capability_flags_must_all_be_false(tmp_path: Path) -> None:
    catalog, bundle_path, bundle, _source_ref = _bundle_fixture(tmp_path)
    runtime = bundle_path.parent / "candidates" / "runtime.json"
    _write_json(
        runtime,
        {
            "runtime": "python",
            "contract": "robata-schema-upcaster-v1",
            "clock": False,
            "database": False,
            "network": True,
            "randomness": False,
        },
    )

    with pytest.raises(SchemaEvolutionRegistrationError, match="network=false"):
        register_schema_evolution(bundle_path, catalog_path=catalog)

    assert all(not (catalog.parent / path).exists() for path in _artifact_paths(bundle))
