from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

import scripts.register_schema as schema_registration
import scripts.verify_schema_registry as schema_verification
from robata.contracts.schema_registry import (
    SchemaDefinitionError,
    SchemaRegistry,
    SchemaRegistryError,
    deterministic_schema_artifact_id,
)
from scripts.register_schema import (
    PublishedSchemaConflictError,
    SchemaRegistrationError,
    SchemaRegistrationResult,
    register_schema,
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _schema(
    name: str,
    *,
    version_directory: str = "v1",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://schemas.robata.dev/{version_directory}/{name}.schema.json",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "title": "UTF-8 caf\u00e9",
        "type": "object",
        **extra,
    }


def _entry(name: str, raw: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "ref": {
            "schema_id": f"https://schemas.robata.dev/{name}",
            "version": "1.0.0",
            "artifact_id": deterministic_schema_artifact_id(digest),
            "sha256": digest,
        },
        "wire_version": "1.0",
        "document_id": f"https://schemas.robata.dev/v1/{name}.schema.json",
        "artifact_path": f"v1/{name}.schema.json",
        "owner": "test",
        "canonicalization_version": "rfc8785-v1",
        "projection_version": f"{name}-v1",
        "compatibility_mode": "NONE",
        "lifecycle": "ACTIVE",
        "supported_software": {
            "min_inclusive": "0.1.0",
            "max_exclusive": "0.2.0",
        },
        "supported_predecessors": [],
    }


def _registry(tmp_path: Path) -> Path:
    root = tmp_path / "schemas"
    root.mkdir()
    base = _schema("base")
    raw = _json_bytes(base)
    target = root / "v1" / "base.schema.json"
    target.parent.mkdir()
    target.write_bytes(raw)
    catalog = {
        "catalog_version": "1.0",
        "schemas": [_entry("base", raw)],
        "upcasters": [],
    }
    catalog_path = root / "schema-catalog.json"
    catalog_path.write_bytes(_json_bytes(catalog))
    SchemaRegistry(catalog_path)
    return catalog_path


def _candidate(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "candidate.json"
    text = json.dumps(document, ensure_ascii=False, indent=4)
    path.write_bytes((text + "\n").replace("\n", "\r\n").encode("utf-8"))
    return path


def _register(
    candidate: Path,
    catalog: Path,
    *,
    dry_run: bool = False,
    software_min: str = "0.1.0",
    software_max: str = "0.2.0",
) -> SchemaRegistrationResult:
    return register_schema(
        candidate,
        catalog_path=catalog,
        version="1.0.0",
        wire_version="1.0",
        owner="test",
        projection_version="candidate-v1",
        software_min=software_min,
        software_max=software_max,
        dry_run=dry_run,
    )


def test_validation_snapshot_parent_honors_explicit_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "schema-snapshots"
    snapshot_root.mkdir()
    monkeypatch.setenv("ROBATA_SCHEMA_VALIDATION_TEMP_ROOT", str(snapshot_root))

    assert schema_registration._validation_snapshot_parent(tmp_path / "schemas") == (
        snapshot_root.resolve()
    )


def test_register_normalizes_utf8_lf_and_pins_exact_bytes(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    document = _schema("candidate")
    candidate = _candidate(tmp_path, document)

    result = _register(candidate, catalog)

    target = catalog.parent / result.artifact_path
    expected = _json_bytes(document)
    assert result.changed is True
    assert result.dry_run is False
    assert target.read_bytes() == expected
    assert b"\r\n" not in target.read_bytes()
    assert target.read_bytes().decode("utf-8").find("caf\u00e9") >= 0
    assert result.ref.sha256 == hashlib.sha256(expected).hexdigest()
    assert result.ref.artifact_id == deterministic_schema_artifact_id(result.ref.sha256)
    assert SchemaRegistry(catalog).resolve_exact(result.ref).document_bytes == expected


def test_dry_run_validates_snapshot_without_writing(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    before = catalog.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate"))

    result = _register(candidate, catalog, dry_run=True)

    assert result.changed is True
    assert result.dry_run is True
    assert catalog.read_bytes() == before
    assert not (catalog.parent / result.artifact_path).exists()


def test_registration_replay_is_idempotent(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))
    first = _register(candidate, catalog)
    catalog_after_first = catalog.read_bytes()

    second = _register(candidate, catalog)

    assert first.ref == second.ref
    assert second.changed is False
    assert catalog.read_bytes() == catalog_after_first


def test_registration_and_replay_do_not_require_current_runtime_software_support(
    tmp_path: Path,
) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))

    first = _register(
        candidate,
        catalog,
        software_min="2.0.0",
        software_max="3.0.0",
    )
    second = _register(
        candidate,
        catalog,
        software_min="2.0.0",
        software_max="3.0.0",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.ref == second.ref
    assert SchemaRegistry(catalog, software_version="2.0.0").resolve_exact(
        first.ref
    ).document_bytes == _json_bytes(_schema("candidate"))


def test_published_version_rejects_different_exact_bytes(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))
    result = _register(candidate, catalog)
    target = catalog.parent / result.artifact_path
    catalog_before = catalog.read_bytes()
    artifact_before = target.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate", description="changed"))

    with pytest.raises(PublishedSchemaConflictError, match="different exact bytes"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert target.read_bytes() == artifact_before


def test_unresolved_reference_fails_before_publication(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    candidate = _candidate(
        tmp_path,
        _schema("candidate", properties={"value": {"$ref": "missing.schema.json"}}),
    )

    with pytest.raises(SchemaDefinitionError, match="unresolved offline reference"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()


def test_catalog_replace_failure_removes_unpublished_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate"))
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
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()
    SchemaRegistry(catalog)


def test_catalog_staging_failure_cleans_schema_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate"))
    real_write_staged = schema_registration._write_staged

    def fail_catalog_stage(path: Path, contents: bytes, *, mode: int) -> Path:
        if path == catalog:
            raise OSError("simulated catalog staging failure")
        return real_write_staged(path, contents, mode=mode)

    monkeypatch.setattr(schema_registration, "_write_staged", fail_catalog_stage)

    with pytest.raises(OSError, match="simulated catalog staging failure"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()
    assert list((catalog.parent / "v1").glob(".candidate.schema.json.*.tmp")) == []


def test_schema_staging_fsync_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate"))
    real_fsync = os.fsync
    failed = False

    def fail_first_regular_file_sync(descriptor: int) -> None:
        nonlocal failed
        if stat.S_ISREG(os.fstat(descriptor).st_mode) and not failed:
            failed = True
            raise OSError("simulated staged file sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        schema_registration.os,
        "fsync",
        fail_first_regular_file_sync,
    )

    with pytest.raises(OSError, match="simulated staged file sync failure"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()
    assert list((catalog.parent / "v1").glob(".candidate.schema.json.*.tmp")) == []


def test_catalog_directory_sync_failure_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate"))
    real_fsync_replaced_paths = schema_registration._fsync_replaced_paths
    failed = False

    def fail_catalog_sync(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        real_fsync_replaced_paths(source, destination)
        if Path(destination) == catalog and not failed:
            failed = True
            raise OSError("simulated catalog directory sync failure")

    monkeypatch.setattr(
        schema_registration,
        "_fsync_replaced_paths",
        fail_catalog_sync,
    )

    with pytest.raises(OSError, match="simulated catalog directory sync failure"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()
    SchemaRegistry(catalog)


def test_staged_regular_files_are_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))
    real_fsync = os.fsync
    regular_file_syncs = 0

    def record_fsync(descriptor: int) -> None:
        nonlocal regular_file_syncs
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            regular_file_syncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(schema_registration.os, "fsync", record_fsync)

    _register(candidate, catalog)

    assert regular_file_syncs >= 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not portable to Windows")
def test_registration_preserves_catalog_mode_and_sets_schema_mode(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    catalog.chmod(0o640)
    candidate = _candidate(tmp_path, _schema("candidate"))

    result = _register(candidate, catalog)

    target = catalog.parent / result.artifact_path
    assert stat.S_IMODE(catalog.stat().st_mode) == 0o640
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_unmarked_exact_orphan_is_rejected(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    document = _schema("candidate")
    candidate = _candidate(tmp_path, document)
    target = catalog.parent / "v1" / "candidate.schema.json"
    target.write_bytes(_json_bytes(document))

    with pytest.raises(SchemaDefinitionError, match="uncataloged schema documents"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert target.read_bytes() == _json_bytes(document)


def test_pre_catalog_crash_marker_preserves_only_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    document = _schema("candidate")
    candidate = _candidate(tmp_path, document)
    marker = catalog.parent / schema_registration.SCHEMA_PUBLICATION_MARKER_FILENAME
    real_publish = schema_registration._publish

    class SimulatedHardCrash(BaseException):
        pass

    def leave_pre_catalog_state(**arguments: Any) -> None:
        target = schema_registration._contained_artifact_target(
            catalog.parent,
            arguments["artifact_path"],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(
            schema_registration._publication_marker_bytes(
                artifact_path=arguments["artifact_path"],
                artifact_sha256=hashlib.sha256(arguments["schema_bytes"]).hexdigest(),
                original_catalog=arguments["original_catalog"],
                new_catalog=arguments["new_catalog"],
            )
        )
        target.write_bytes(arguments["schema_bytes"])
        raise SimulatedHardCrash

    monkeypatch.setattr(schema_registration, "_publish", leave_pre_catalog_state)
    with pytest.raises(SimulatedHardCrash):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert marker.is_file()
    assert tuple(item.ref.schema_id for item in SchemaRegistry(catalog).entries) == (
        "https://schemas.robata.dev/base",
    )

    rogue = catalog.parent / "v1" / "rogue.schema.json"
    rogue.write_bytes(_json_bytes(_schema("rogue")))
    with pytest.raises(SchemaDefinitionError, match="uncataloged schema documents"):
        SchemaRegistry(catalog)
    rogue.unlink()

    with schema_registration._registration_lock(catalog.parent):
        schema_registration._recover_publication_marker(catalog)
    assert marker.is_file()

    monkeypatch.setattr(schema_registration, "_publish", real_publish)
    result = _register(candidate, catalog)
    assert result.changed is True
    assert not marker.exists()


def test_post_catalog_crash_marker_is_recovered_before_replay_and_next_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))
    marker = catalog.parent / schema_registration.SCHEMA_PUBLICATION_MARKER_FILENAME
    real_publish = schema_registration._publish
    marker_bytes: list[bytes] = []

    class SimulatedHardCrash(BaseException):
        pass

    def leave_post_catalog_state(**arguments: Any) -> None:
        target = schema_registration._contained_artifact_target(
            catalog.parent,
            arguments["artifact_path"],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        raw_marker = schema_registration._publication_marker_bytes(
            artifact_path=arguments["artifact_path"],
            artifact_sha256=hashlib.sha256(arguments["schema_bytes"]).hexdigest(),
            original_catalog=arguments["original_catalog"],
            new_catalog=arguments["new_catalog"],
        )
        marker_bytes.append(raw_marker)
        marker.write_bytes(raw_marker)
        target.write_bytes(arguments["schema_bytes"])
        catalog.write_bytes(arguments["new_catalog"])
        raise SimulatedHardCrash

    monkeypatch.setattr(schema_registration, "_publish", leave_post_catalog_state)
    with pytest.raises(SimulatedHardCrash):
        _register(candidate, catalog)

    assert marker.is_file()
    assert len(SchemaRegistry(catalog).entries) == 2

    monkeypatch.setattr(schema_registration, "_publish", real_publish)
    replay = _register(candidate, catalog)
    assert replay.changed is False
    assert not marker.exists()

    marker.write_bytes(marker_bytes[0])
    next_candidate = _candidate(tmp_path, _schema("next"))
    following = _register(next_candidate, catalog)
    assert following.changed is True
    assert not marker.exists()
    assert len(SchemaRegistry(catalog).entries) == 3


def test_unmarked_unknown_schema_orphan_remains_fail_closed(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    rogue = catalog.parent / "v1" / "rogue.schema.json"
    rogue.write_bytes(_json_bytes(_schema("rogue")))

    with pytest.raises(SchemaDefinitionError, match="uncataloged schema documents"):
        SchemaRegistry(catalog)


@pytest.mark.parametrize(
    "relative_path",
    ["rogue.schema.json", "experimental/rogue.schema.json"],
)
def test_schema_orphan_outside_version_directory_is_rejected(
    tmp_path: Path,
    relative_path: str,
) -> None:
    catalog = _registry(tmp_path)
    rogue = catalog.parent / relative_path
    rogue.parent.mkdir(parents=True, exist_ok=True)
    rogue.write_bytes(_json_bytes(_schema("rogue")))

    with pytest.raises(SchemaDefinitionError, match=r"below a v\[0-9\]\+ directory"):
        SchemaRegistry(catalog)


def test_in_root_schema_symlink_alias_cannot_disappear_from_closure(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    target = catalog.parent / "v1" / "base.schema.json"
    alias = catalog.parent / "v1" / "alias.schema.json"
    try:
        alias.symlink_to(target.name)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"schema symlinks are unavailable: {exc}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", str(alias), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"schema symlinks are unavailable: {completed.stderr.strip()}")

    with pytest.raises(SchemaDefinitionError, match=r"symlink or reparse point"):
        SchemaRegistry(catalog)


def test_nested_schema_directory_link_is_rejected_before_document_scan(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    support = catalog.parent / "support"
    support.mkdir()
    alias_directory = catalog.parent / "v1" / "aliasdir"
    try:
        alias_directory.symlink_to(support, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {exc}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias_directory), str(support)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"directory junctions are unavailable: {completed.stderr.strip()}")

    with pytest.raises(SchemaDefinitionError, match=r"schema tree.*symlink|reparse point"):
        SchemaRegistry(catalog)


def test_reader_blocks_until_catalog_commit_after_artifact_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))
    target = catalog.parent / "v1" / "candidate.schema.json"
    artifact_installed = threading.Event()
    release_writer = threading.Event()
    reader_done = threading.Event()
    writer_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    reader_entries: list[int] = []
    real_replace = schema_registration._atomic_replace

    def pause_after_artifact(source: str | Path, destination: str | Path) -> None:
        real_replace(source, destination)
        if Path(destination) == target:
            artifact_installed.set()
            if not release_writer.wait(5):
                raise TimeoutError("reader did not exercise the publication window")

    def publish() -> None:
        try:
            _register(candidate, catalog)
        except BaseException as exc:
            writer_errors.append(exc)

    def read() -> None:
        try:
            reader_entries.append(len(SchemaRegistry(catalog).entries))
        except BaseException as exc:
            reader_errors.append(exc)
        finally:
            reader_done.set()

    monkeypatch.setattr(schema_registration, "_atomic_replace", pause_after_artifact)
    writer = threading.Thread(target=publish)
    writer.start()
    try:
        assert artifact_installed.wait(5)
        reader = threading.Thread(target=read)
        reader.start()
        assert not reader_done.wait(0.2)
    finally:
        release_writer.set()
    writer.join(5)
    reader.join(5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_errors == []
    assert reader_errors == []
    assert reader_entries == [2]


def test_external_symlink_directory_is_rejected_before_artifact_write(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_version = catalog.parent / "v2"
    try:
        linked_version.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {exc}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_version), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"directory junctions are unavailable: {completed.stderr.strip()}")
    candidate = _candidate(
        tmp_path,
        _schema("candidate", version_directory="v2"),
    )

    with pytest.raises(SchemaRegistrationError, match=r"escapes|symlink|reparse"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert list(outside.iterdir()) == []
    assert not (catalog.parent / schema_registration.SCHEMA_PUBLICATION_MARKER_FILENAME).exists()


def test_external_lock_link_is_rejected_without_modifying_target(
    tmp_path: Path,
) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    lock = catalog.parent / schema_registration.SCHEMA_PUBLICATION_LOCK_FILENAME
    lock.unlink()
    external_file = tmp_path / "external-lock-target"
    external_file.write_bytes(b"do-not-change")
    protected = external_file
    try:
        lock.symlink_to(external_file)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"lock symlinks are unavailable: {exc}")
        external_directory = tmp_path / "external-lock-directory"
        external_directory.mkdir()
        protected = external_directory / "sentinel"
        protected.write_bytes(b"do-not-change")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(lock), str(external_directory)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"lock junctions are unavailable: {completed.stderr.strip()}")
    candidate = _candidate(tmp_path, _schema("candidate"))

    with pytest.raises(SchemaRegistrationError, match=r"publication lock.*symlink|reparse"):
        _register(candidate, catalog)
    with pytest.raises(SchemaRegistryError, match=r"publication lock.*symlink|reparse"):
        SchemaRegistry(catalog)

    assert protected.read_bytes() == b"do-not-change"
    assert catalog.read_bytes() == catalog_before
    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()


def test_strict_schema_verifier_rejects_pending_publication_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _registry(tmp_path)
    marker = catalog.parent / schema_registration.SCHEMA_PUBLICATION_MARKER_FILENAME
    marker.write_bytes(b"{}\n")
    monkeypatch.setattr(schema_verification, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="pending schema publication marker"):
        schema_verification.main()


def test_unmarked_different_orphan_bytes_are_not_overwritten(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    catalog_before = catalog.read_bytes()
    candidate = _candidate(tmp_path, _schema("candidate"))
    target = catalog.parent / "v1" / "candidate.schema.json"
    target.write_bytes(b"{}\n")

    with pytest.raises(SchemaDefinitionError, match="uncataloged schema documents"):
        _register(candidate, catalog)

    assert catalog.read_bytes() == catalog_before
    assert target.read_bytes() == b"{}\n"


def test_registration_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    catalog = _registry(tmp_path)
    candidate = _candidate(tmp_path, _schema("candidate"))

    with (
        schema_registration._registration_lock(catalog.parent),
        pytest.raises(SchemaRegistrationError, match="another schema registration"),
    ):
        _register(candidate, catalog)

    assert not (catalog.parent / "v1" / "candidate.schema.json").exists()
