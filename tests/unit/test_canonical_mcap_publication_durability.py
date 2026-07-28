from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.adapters.local_artifact_registry import LocalArtifactRegistry
from robata.application.canonical import mcap_source
from robata.contracts.hashing import exact_bytes_sha256
from robata.ports.artifact_registry import ArtifactRegistryError, ArtifactRegistryErrorCode


def test_publish_png_verifies_bytes_and_syncs_target_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = b"deterministic png bytes"
    digest = exact_bytes_sha256(contents)
    file_syncs: list[Path] = []
    directory_syncs: list[Path] = []
    monkeypatch.setattr(
        mcap_source,
        "_sync_file",
        lambda path: file_syncs.append(path),
    )
    monkeypatch.setattr(
        mcap_source,
        "_sync_directory",
        lambda path: directory_syncs.append(path),
    )

    published = mcap_source._publish_png(tmp_path / "frames", digest, contents)

    assert published.read_bytes() == contents
    assert file_syncs == [published]
    assert directory_syncs == [published.parent, published.parent]
    assert not tuple(published.parent.glob("*.tmp"))


def test_publish_exact_state_file_uses_the_same_file_and_parent_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = b"durable state bytes"
    target = tmp_path / "state" / "report.json"
    file_syncs: list[Path] = []
    directory_syncs: list[Path] = []
    monkeypatch.setattr(
        mcap_source,
        "_sync_file",
        lambda path: file_syncs.append(path),
    )
    monkeypatch.setattr(
        mcap_source,
        "_sync_directory",
        lambda path: directory_syncs.append(path),
    )

    published = mcap_source._publish_exact_state_file(
        target,
        contents,
        label="media quality report",
    )

    assert published.read_bytes() == contents
    assert file_syncs == [published]
    assert directory_syncs == [published.parent, published.parent]


def test_state_file_write_sync_failure_never_exposes_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state" / "report.json"

    def fail_write_sync(_descriptor: int) -> None:
        raise OSError("injected write sync failure")

    monkeypatch.setattr(mcap_source.os, "fsync", fail_write_sync)

    with pytest.raises(
        mcap_source.CanonicalMcapSourceError,
        match="cannot write staged media quality report",
    ):
        mcap_source._publish_exact_state_file(
            target,
            b"state bytes",
            label="media quality report",
        )

    assert not target.exists()
    assert not tuple(target.parent.glob("*.tmp"))


def test_publish_png_rejects_digest_mismatch_before_staging(tmp_path: Path) -> None:
    with pytest.raises(mcap_source.CanonicalMcapSourceError, match="digest"):
        mcap_source._publish_png(tmp_path / "frames", "0" * 64, b"wrong bytes")

    assert not (tmp_path / "frames").exists()


def test_publish_png_parent_sync_failure_leaves_complete_target_without_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = b"durable before parent fault"
    digest = exact_bytes_sha256(contents)
    directory_syncs: list[Path] = []

    def fail_after_link(path: Path) -> None:
        directory_syncs.append(path)
        if len(directory_syncs) == 2:
            raise OSError("injected parent sync failure")

    monkeypatch.setattr(mcap_source, "_sync_directory", fail_after_link)
    monkeypatch.setattr(mcap_source, "_sync_file", lambda path: None)

    with pytest.raises(
        mcap_source.CanonicalMcapSourceError,
        match="cannot synchronize published frame artifact",
    ):
        mcap_source._publish_png(tmp_path / "frames", digest, contents)

    target = tmp_path / "frames" / "sha256" / digest[:2] / f"{digest}.png"
    assert target.read_bytes() == contents
    assert not tuple(target.parent.glob("*.tmp"))


def test_publish_png_file_sync_failure_leaves_complete_target_without_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = b"durable before file sync fault"
    digest = exact_bytes_sha256(contents)
    monkeypatch.setattr(mcap_source, "_sync_directory", lambda path: None)

    def fail_target_sync(_path: Path) -> None:
        raise OSError("injected target sync failure")

    monkeypatch.setattr(mcap_source, "_sync_file", fail_target_sync)

    with pytest.raises(
        mcap_source.CanonicalMcapSourceError,
        match="cannot synchronize published frame artifact",
    ):
        mcap_source._publish_png(tmp_path / "frames", digest, contents)

    target = tmp_path / "frames" / "sha256" / digest[:2] / f"{digest}.png"
    assert target.read_bytes() == contents
    assert not tuple(target.parent.glob("*.tmp"))


def test_publish_exact_state_file_preserves_existing_mismatch_contract(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state" / "report.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")

    with pytest.raises(
        mcap_source.CanonicalMcapSourceError,
        match="existing media quality report bytes are inconsistent",
    ):
        mcap_source._publish_exact_state_file(
            target,
            b"new",
            label="media quality report",
        )


def test_startup_registry_reconciliation_removes_unreferenced_partials_and_orphans(
    tmp_path: Path,
) -> None:
    registry = LocalArtifactRegistry(tmp_path / "registry")
    orphan_bytes = b"orphan exact bytes"
    orphan_digest = exact_bytes_sha256(orphan_bytes)
    orphan = registry.blob_root / orphan_digest[:2] / orphan_digest
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(orphan_bytes)
    partial = orphan.parent / ".put-fixture.tmp"
    partial.write_bytes(b"partial")

    mcap_source._reconcile_artifact_registry_for_startup(registry)

    assert not orphan.exists()
    assert not partial.exists()


@pytest.mark.parametrize("stored_bytes", [None, b"corrupt registered bytes"])
def test_startup_registry_reconciliation_rejects_missing_or_corrupt_registered_blob(
    tmp_path: Path,
    stored_bytes: bytes | None,
) -> None:
    registry = LocalArtifactRegistry(tmp_path / "registry")
    expected_bytes = b"registered exact bytes"
    digest = exact_bytes_sha256(expected_bytes)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            """
            INSERT INTO artifacts (
                artifact_id, artifact_type, semantic_sha256, exact_sha256,
                byte_count, media_type, entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "registered-fixture",
                "RAW_MCAP",
                "0" * 64,
                digest,
                len(expected_bytes),
                "application/x-mcap",
                b"{}",
            ),
        )

    if stored_bytes is not None:
        blob = registry.blob_root / digest[:2] / digest
        blob.parent.mkdir(parents=True)
        blob.write_bytes(stored_bytes)

    with pytest.raises(
        mcap_source.CanonicalMcapSourceError,
        match=(
            "artifact registry startup reconciliation failed: "
            "artifact registry reconciliation found unresolved storage discrepancies"
        ),
    ):
        mcap_source._reconcile_artifact_registry_for_startup(registry)


def test_startup_registry_reconciliation_fails_on_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LocalArtifactRegistry(tmp_path / "registry")

    def fail_reconcile(**kwargs: object) -> object:
        assert kwargs == {
            "remove_orphans": True,
            "remove_partials": True,
            "remove_duplicates": True,
            "strict": True,
        }
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INTEGRITY_ERROR,
            "registered blob is missing",
        )

    monkeypatch.setattr(registry, "reconcile", fail_reconcile)

    with pytest.raises(
        mcap_source.CanonicalMcapSourceError,
        match="artifact registry startup reconciliation failed: registered blob is missing",
    ):
        mcap_source._reconcile_artifact_registry_for_startup(registry)
