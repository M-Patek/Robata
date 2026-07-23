from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.register_schema as schema_registration
from robata.contracts.schema_registry import (
    SchemaDefinitionError,
    SchemaRegistry,
    deterministic_schema_artifact_id,
)
from scripts.register_schema import PublishedSchemaConflictError
from scripts.register_schema_bundle import (
    SchemaBundleRegistrationError,
    register_schema_bundle,
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _schema(artifact_path: str, *, title: str, reference: str | None = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://schemas.robata.dev/{artifact_path}",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "title": title,
        "type": "object",
    }
    if reference is not None:
        document["$ref"] = f"https://schemas.robata.dev/{reference}"
    return document


def _catalog_entry(
    *,
    name: str,
    artifact_path: str,
    raw: bytes,
) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "ref": {
            "schema_id": f"https://schemas.robata.dev/{name}",
            "version": "1.0.0",
            "artifact_id": deterministic_schema_artifact_id(digest),
            "sha256": digest,
        },
        "wire_version": "1.0",
        "document_id": f"https://schemas.robata.dev/{artifact_path}",
        "artifact_path": artifact_path,
        "owner": "test",
        "canonicalization_version": "rfc8785-v1",
        "projection_version": f"{name}-v1",
        "compatibility_mode": "NONE",
        "lifecycle": "ACTIVE",
        "supported_software": {"min_inclusive": "0.1.0", "max_exclusive": "0.2.0"},
        "supported_predecessors": [],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    schema_root = tmp_path / "schemas"
    base_path = "v1/base.schema.json"
    base = _schema(base_path, title="Base")
    base_bytes = _json_bytes(base)
    (schema_root / base_path).parent.mkdir(parents=True)
    (schema_root / base_path).write_bytes(base_bytes)
    catalog = schema_root / "schema-catalog.json"
    catalog.write_bytes(
        _json_bytes(
            {
                "catalog_version": "1.0",
                "schemas": [_catalog_entry(name="base", artifact_path=base_path, raw=base_bytes)],
                "upcasters": [],
            }
        )
    )
    SchemaRegistry(catalog)

    bundle_root = tmp_path / "bundle"
    (bundle_root / "candidates").mkdir(parents=True)
    alpha_path = "v1/alpha.schema.json"
    beta_path = "v1/beta.schema.json"
    alpha = _schema(alpha_path, title="Café alpha")
    beta = _schema(beta_path, title="Beta", reference=alpha_path)
    # CRLF candidates prove that publication pins the normalized LF bytes.
    (bundle_root / "candidates/alpha.json").write_bytes(_json_bytes(alpha).replace(b"\n", b"\r\n"))
    (bundle_root / "candidates/beta.json").write_bytes(_json_bytes(beta).replace(b"\n", b"\r\n"))
    manifest: dict[str, Any] = {
        "format_version": "1.0",
        "schemas": [
            {
                "candidate": "candidates/beta.json",
                "schema_id": "https://schemas.robata.dev/beta",
                "version": "1.0.0",
                "wire_version": "1.0",
                "artifact_path": beta_path,
                "owner": "test",
                "projection_version": "beta-v1",
                "canonicalization_version": "rfc8785-v1",
                "software": {"min_inclusive": "0.1.0", "max_exclusive": "0.2.0"},
            },
            {
                "candidate": "candidates/alpha.json",
                "schema_id": "https://schemas.robata.dev/alpha",
                "version": "1.0.0",
                "wire_version": "1.0",
                "artifact_path": alpha_path,
                "owner": "test",
                "projection_version": "alpha-v1",
                "canonicalization_version": "rfc8785-v1",
                "software": {"min_inclusive": "0.1.0", "max_exclusive": "0.2.0"},
            },
        ],
    }
    bundle_path = bundle_root / "bundle.json"
    bundle_path.write_bytes(_json_bytes(manifest))
    return catalog, bundle_path, manifest


def test_bundle_normalizes_pins_and_validates_cross_schema_refs(tmp_path: Path) -> None:
    catalog, bundle, manifest = _fixture(tmp_path)

    result = register_schema_bundle(bundle, catalog_path=catalog)

    assert result.changed is True
    assert result.dry_run is False
    assert result.artifact_paths == (
        "v1/alpha.schema.json",
        "v1/beta.schema.json",
    )
    registry = SchemaRegistry(catalog)
    assert {item.ref.schema_id for item in registry.entries} == {
        "https://schemas.robata.dev/alpha",
        "https://schemas.robata.dev/base",
        "https://schemas.robata.dev/beta",
    }
    alpha_document = _schema("v1/alpha.schema.json", title="Café alpha")
    alpha_bytes = _json_bytes(alpha_document)
    assert (catalog.parent / "v1/alpha.schema.json").read_bytes() == alpha_bytes
    assert b"\r\n" not in (catalog.parent / "v1/beta.schema.json").read_bytes()
    assert result.refs[0].artifact_id == deterministic_schema_artifact_id(
        hashlib.sha256(alpha_bytes).hexdigest()
    )
    assert manifest["schemas"][0]["candidate"] == "candidates/beta.json"


def test_bundle_dry_run_validates_complete_snapshot_without_writing(tmp_path: Path) -> None:
    catalog, bundle, _manifest = _fixture(tmp_path)
    catalog_before = catalog.read_bytes()

    result = register_schema_bundle(bundle, catalog_path=catalog, dry_run=True)

    assert result.changed is True
    assert result.dry_run is True
    assert all(item.changed for item in result.items)
    assert catalog.read_bytes() == catalog_before
    assert all(not (catalog.parent / path).exists() for path in result.artifact_paths)


def test_bundle_replay_is_idempotent_and_deterministic(tmp_path: Path) -> None:
    catalog, bundle, _manifest = _fixture(tmp_path)
    first = register_schema_bundle(bundle, catalog_path=catalog)
    catalog_after_first = catalog.read_bytes()
    artifact_bytes = {path: (catalog.parent / path).read_bytes() for path in first.artifact_paths}

    second = register_schema_bundle(bundle, catalog_path=catalog)

    assert second.changed is False
    assert all(not item.changed for item in second.items)
    assert second.refs == first.refs
    assert catalog.read_bytes() == catalog_after_first
    assert {
        path: (catalog.parent / path).read_bytes() for path in second.artifact_paths
    } == artifact_bytes


def test_bundle_conflict_does_not_publish_other_candidates(tmp_path: Path) -> None:
    catalog, bundle, manifest = _fixture(tmp_path)
    register_schema_bundle(bundle, catalog_path=catalog)
    catalog_before = catalog.read_bytes()

    # Add a new candidate while changing an already published logical version. The
    # conflict must be discovered before either candidate is staged.
    alpha_candidate = bundle.parent / "candidates" / "alpha.json"
    changed_alpha = _schema("v1/alpha.schema.json", title="changed")
    alpha_candidate.write_bytes(_json_bytes(changed_alpha))
    gamma_path = "v1/gamma.schema.json"
    (bundle.parent / "candidates/gamma.json").write_bytes(
        _json_bytes(_schema(gamma_path, title="Gamma"))
    )
    manifest["schemas"].append(
        {
            "candidate": "candidates/gamma.json",
            "schema_id": "https://schemas.robata.dev/gamma",
            "version": "1.0.0",
            "wire_version": "1.0",
            "artifact_path": gamma_path,
            "owner": "test",
            "projection_version": "gamma-v1",
            "canonicalization_version": "rfc8785-v1",
            "software": {"min_inclusive": "0.1.0", "max_exclusive": "0.2.0"},
        }
    )
    bundle.write_bytes(_json_bytes(manifest))

    with pytest.raises(PublishedSchemaConflictError, match="different exact bytes"):
        register_schema_bundle(bundle, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / gamma_path).exists()


def test_bundle_validation_failure_leaves_catalog_and_artifacts_untouched(tmp_path: Path) -> None:
    catalog, bundle, _manifest = _fixture(tmp_path)
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    beta_candidate = bundle.parent / "candidates" / "beta.json"
    beta = _schema("v1/beta.schema.json", title="Beta", reference="v1/missing.schema.json")
    beta_candidate.write_bytes(_json_bytes(beta))
    bundle.write_bytes(_json_bytes(manifest))
    catalog_before = catalog.read_bytes()

    with pytest.raises(SchemaDefinitionError, match="unresolved offline reference"):
        register_schema_bundle(bundle, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1/alpha.schema.json").exists()
    assert not (catalog.parent / "v1/beta.schema.json").exists()


def test_bundle_catalog_replace_failure_rolls_back_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, bundle, _manifest = _fixture(tmp_path)
    catalog_before = catalog.read_bytes()
    real_replace = schema_registration._atomic_replace
    failed = False

    def fail_catalog_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination) == catalog and not failed:
            failed = True
            raise OSError("simulated catalog replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(schema_registration, "_atomic_replace", fail_catalog_replace)

    with pytest.raises(OSError, match="simulated catalog replace failure"):
        register_schema_bundle(bundle, catalog_path=catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1/alpha.schema.json").exists()
    assert not (catalog.parent / "v1/beta.schema.json").exists()
    SchemaRegistry(catalog)


def test_bundle_manifest_is_strict_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    catalog, bundle, _manifest = _fixture(tmp_path)
    bundle.write_bytes(b'{"format_version":"1.0","format_version":"1.0","schemas":[]}')

    with pytest.raises(SchemaBundleRegistrationError, match="duplicate JSON object key"):
        register_schema_bundle(bundle, catalog_path=catalog)


def test_bundle_cli_help_runs_outside_repository_working_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "register_schema_bundle.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "bundle" in completed.stdout
