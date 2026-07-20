from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.check_schema_immutability import (
    SchemaImmutabilityError,
    check_schema_immutability,
    main,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _schema_entry(name: str, version: str, path: str, contents: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(contents).hexdigest()
    return {
        "ref": {
            "schema_id": f"https://schemas.robata.dev/{name}",
            "version": version,
            "artifact_id": "00000000-0000-0000-0000-000000000001",
            "sha256": digest,
        },
        "wire_version": version.rsplit(".", 1)[0],
        "document_id": f"https://schemas.robata.dev/{path}",
        "artifact_path": path,
        "owner": "test",
        "canonicalization_version": "rfc8785-v1",
        "projection_version": f"{name}-v{version}",
        "compatibility_mode": "NONE",
        "lifecycle": "ACTIVE",
        "supported_software": {
            "min_inclusive": "0.1.0",
            "max_exclusive": "0.2.0",
        },
        "supported_predecessors": [],
    }


def _upcaster(repo: Path, upcaster_id: str) -> dict[str, Any]:
    code_path = f"upcasters/{upcaster_id}.py"
    runtime_path = f"runtime/{upcaster_id}.json"
    input_path = f"golden/{upcaster_id}.input.json"
    output_path = f"golden/{upcaster_id}.output.json"
    artifacts = {
        code_path: b"def upcast(payload):\n    return dict(payload)\n",
        runtime_path: b'{"runtime":"python"}\n',
        input_path: b'{"schema_version":"1.0"}\n',
        output_path: b'{"schema_version":"2.0"}\n',
    }
    for path, contents in artifacts.items():
        _write_schema(repo, path, contents)
    return {
        "upcaster_id": upcaster_id,
        "source": {"schema": "alpha", "version": "1.0.0"},
        "target": {"schema": "alpha", "version": "2.0.0"},
        "code_artifact_path": code_path,
        "code_sha256": hashlib.sha256(artifacts[code_path]).hexdigest(),
        "runtime_artifact_path": runtime_path,
        "runtime_sha256": hashlib.sha256(artifacts[runtime_path]).hexdigest(),
        "golden_vectors": [
            {
                "input_artifact_path": input_path,
                "input_sha256": hashlib.sha256(artifacts[input_path]).hexdigest(),
                "output_artifact_path": output_path,
                "output_sha256": hashlib.sha256(artifacts[output_path]).hexdigest(),
            }
        ],
    }


def _write_catalog(
    repo: Path,
    schemas: list[dict[str, Any]],
    upcasters: list[dict[str, Any]],
) -> None:
    catalog = {
        "catalog_version": "1.0",
        "schemas": schemas,
        "upcasters": upcasters,
    }
    path = repo / "schemas" / "schema-catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(catalog, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_schema(repo: Path, path: str, contents: bytes) -> None:
    target = repo / "schemas" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repository(
    tmp_path: Path,
) -> tuple[Path, str, dict[str, Any], dict[str, Any], bytes]:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "user.email", "schema-test@example.invalid")
    _git(repo, "config", "user.name", "Schema Test")
    contents = b'{"type":"object"}\n'
    entry = _schema_entry("alpha", "1.0.0", "v1/alpha.schema.json", contents)
    upcaster = _upcaster(repo, "alpha-v1-to-v2")
    _write_schema(repo, entry["artifact_path"], contents)
    _write_catalog(repo, [entry], [upcaster])
    baseline = _commit(repo, "baseline")
    return repo, baseline, entry, upcaster, contents


def _inconsistent_repository(
    tmp_path: Path,
) -> tuple[Path, str, dict[str, Any], bytes]:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "user.email", "schema-test@example.invalid")
    _git(repo, "config", "user.name", "Schema Test")
    pinned_contents = b'{\r\n  "type": "object"\r\n}\r\n'
    baseline_contents = pinned_contents.replace(b"\r\n", b"\n")
    entry = _schema_entry("alpha", "1.0.0", "v1/alpha.schema.json", pinned_contents)
    _write_schema(repo, entry["artifact_path"], baseline_contents)
    _write_catalog(repo, [entry], [])
    baseline = _commit(repo, "inconsistent baseline")
    return repo, baseline, entry, pinned_contents


def test_allows_new_schema_versions_and_upcasters(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    added_contents = b'{"type":"string"}\n'
    added_entry = _schema_entry("alpha", "2.0.0", "v2/alpha.schema.json", added_contents)
    added_upcaster = _upcaster(repo, "alpha-v2-to-v3")
    _write_schema(repo, added_entry["artifact_path"], added_contents)
    _write_catalog(repo, [entry, added_entry], [upcaster, added_upcaster])
    current = _commit(repo, "add schema version")

    result = check_schema_immutability(repo, baseline_ref=baseline)

    assert result.baseline_commit == baseline
    assert result.current_commit == current
    assert result.preserved_schema_versions == 1
    assert result.added_schema_versions == 1
    assert result.preserved_upcasters == 1
    assert result.added_upcasters == 1
    assert result.reconciled_schema_versions == ()


def test_rejects_changed_published_schema_entry(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    changed = copy.deepcopy(entry)
    changed["owner"] = "changed"
    _write_catalog(repo, [changed], [upcaster])
    _commit(repo, "change entry")

    with pytest.raises(SchemaImmutabilityError, match="published schema entry changed"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_rejects_deleted_published_schema_version(tmp_path: Path) -> None:
    repo, baseline, _entry, upcaster, _contents = _repository(tmp_path)
    _write_catalog(repo, [], [upcaster])
    _commit(repo, "delete entry")

    with pytest.raises(SchemaImmutabilityError, match="published schema version was deleted"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_rejects_changed_exact_schema_bytes(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    _write_schema(repo, entry["artifact_path"], b'{"type":"array"}\n')
    _write_catalog(repo, [entry], [upcaster])
    _commit(repo, "change exact bytes")

    with pytest.raises(SchemaImmutabilityError, match="published schema exact bytes changed"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_allows_only_reconciliation_to_unchanged_catalog_pin(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, baseline, entry, pinned_contents = _inconsistent_repository(tmp_path)
    _write_schema(repo, entry["artifact_path"], pinned_contents)
    current = _commit(repo, "reconcile exact bytes")

    result = check_schema_immutability(repo, baseline_ref=baseline)

    label = "https://schemas.robata.dev/alpha@1.0.0"
    assert result.current_commit == current
    assert result.reconciled_schema_versions == (label,)
    assert main(["--repo", str(repo), "--baseline-ref", baseline]) == 0
    output = capsys.readouterr().out
    assert "reconciled_schemas=1" in output
    assert label in output


def test_rejects_inconsistent_baseline_change_that_does_not_reach_catalog_pin(
    tmp_path: Path,
) -> None:
    repo, baseline, entry, _pinned_contents = _inconsistent_repository(tmp_path)
    _write_schema(repo, entry["artifact_path"], b'{"type":"array"}\n')
    _commit(repo, "replace with other bytes")

    with pytest.raises(SchemaImmutabilityError, match="published schema exact bytes changed"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_rejects_reconciliation_that_changes_catalog_pin(tmp_path: Path) -> None:
    repo, baseline, entry, _pinned_contents = _inconsistent_repository(tmp_path)
    replacement = b'{"type":"array"}\n'
    changed_entry = copy.deepcopy(entry)
    changed_entry["ref"]["sha256"] = hashlib.sha256(replacement).hexdigest()
    _write_schema(repo, entry["artifact_path"], replacement)
    _write_catalog(repo, [changed_entry], [])
    _commit(repo, "change pin and bytes")

    with pytest.raises(SchemaImmutabilityError, match="published schema entry changed"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_reconciles_current_six_legacy_blobs_in_temporary_commits(tmp_path: Path) -> None:
    source_catalog = json.loads(
        (REPOSITORY_ROOT / "schemas" / "schema-catalog.json").read_text(encoding="utf-8")
    )
    reconciled_entries: list[dict[str, Any]] = []
    authoritative_bytes: dict[str, bytes] = {}
    for entry in source_catalog["schemas"]:
        artifact_path = entry["artifact_path"]
        contents = (REPOSITORY_ROOT / "schemas" / artifact_path).read_bytes()
        if b"\r\n" not in contents:
            continue
        assert hashlib.sha256(contents).hexdigest() == entry["ref"]["sha256"]
        reconciled_entries.append(entry)
        authoritative_bytes[artifact_path] = contents

    assert len(reconciled_entries) == 6
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "user.email", "schema-test@example.invalid")
    _git(repo, "config", "user.name", "Schema Test")
    for artifact_path, contents in authoritative_bytes.items():
        normalized = contents.replace(b"\r\n", b"\n")
        assert normalized != contents
        _write_schema(repo, artifact_path, normalized)
    _write_catalog(repo, reconciled_entries, [])
    baseline = _commit(repo, "historical normalized blobs")
    for artifact_path, contents in authoritative_bytes.items():
        _write_schema(repo, artifact_path, contents)
    _commit(repo, "catalog authority reconciliation")

    result = check_schema_immutability(repo, baseline_ref=baseline)

    expected_labels = tuple(
        sorted(
            f"{entry['ref']['schema_id']}@{entry['ref']['version']}" for entry in reconciled_entries
        )
    )
    assert result.reconciled_schema_versions == expected_labels


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("delete", "published upcaster was deleted"),
        ("change", "published upcaster changed"),
    ],
)
def test_rejects_deleted_or_changed_upcaster(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    if mutation == "delete":
        current_upcasters: list[dict[str, Any]] = []
    else:
        changed = copy.deepcopy(upcaster)
        changed["code_sha256"] = "9" * 64
        current_upcasters = [changed]
    _write_catalog(repo, [entry], current_upcasters)
    _commit(repo, f"{mutation} upcaster")

    with pytest.raises(SchemaImmutabilityError, match=message):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_rejects_changed_published_upcaster_artifact_bytes(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    code_path = upcaster["code_artifact_path"]
    _write_schema(repo, code_path, b"def upcast(payload):\n    return {}\n")
    _write_catalog(repo, [entry], [upcaster])
    _commit(repo, "change published upcaster code")

    with pytest.raises(SchemaImmutabilityError, match="published upcaster exact bytes changed"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_rejects_added_upcaster_with_unpinned_artifact_bytes(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    added = _upcaster(repo, "alpha-v2-to-v3")
    _write_schema(repo, added["runtime_artifact_path"], b'{"runtime":"changed"}\n')
    _write_catalog(repo, [entry], [upcaster, added])
    _commit(repo, "add upcaster with bad runtime pin")

    with pytest.raises(SchemaImmutabilityError, match="upcaster artifact SHA-256 mismatch"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_rejects_unsafe_upcaster_artifact_path(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    changed = copy.deepcopy(upcaster)
    changed["code_artifact_path"] = "../outside.py"
    _write_catalog(repo, [entry], [changed])
    _commit(repo, "unsafe upcaster path")

    with pytest.raises(SchemaImmutabilityError, match="unsafe artifact path"):
        check_schema_immutability(repo, baseline_ref=baseline)


def test_current_ref_can_select_a_tree_other_than_head(tmp_path: Path) -> None:
    repo, baseline, entry, upcaster, _contents = _repository(tmp_path)
    changed = copy.deepcopy(entry)
    changed["owner"] = "changed"
    _write_catalog(repo, [changed], [upcaster])
    _commit(repo, "invalid head")

    result = check_schema_immutability(
        repo,
        baseline_ref=baseline,
        current_ref=baseline,
    )

    assert result.baseline_commit == baseline
    assert result.current_commit == baseline


def test_cli_requires_and_accepts_explicit_baseline_ref(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, baseline, _entry, _upcaster, _contents = _repository(tmp_path)

    assert main(["--repo", str(repo), "--baseline-ref", baseline]) == 0
    assert "schema immutability check passed" in capsys.readouterr().out
