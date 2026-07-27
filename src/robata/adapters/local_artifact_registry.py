"""SQLite-backed local artifact registry with content-addressed exact blobs."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from robata.contracts.artifacts import (
    ArtifactParent,
    ArtifactRegistryEntry,
    ArtifactRegistrySnapshot,
    ArtifactType,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.ports.artifact_registry import (
    ArtifactBlobSource,
    ArtifactRegistryError,
    ArtifactRegistryErrorCode,
    ArtifactRegistryReconciliation,
    PublishedArtifactDerivation,
)
from robata.runtime.observability import (
    RuntimeAttributeValue,
    RuntimeObserver,
    runtime_increment,
    runtime_span,
)

_ARTIFACT_ID_PATTERN: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_DOMAIN: Final = b"robata-local-artifact-id-v1\x00"
_READ_CHUNK_BYTES: Final = 1024 * 1024

_SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    semantic_sha256 TEXT NOT NULL,
    exact_sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count > 0),
    media_type TEXT NOT NULL,
    entry_json BLOB NOT NULL,
    UNIQUE (artifact_type, semantic_sha256)
);

CREATE TABLE IF NOT EXISTS artifact_locations (
    artifact_id TEXT PRIMARY KEY
        REFERENCES artifacts (artifact_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    uri TEXT NOT NULL,
    object_version TEXT NOT NULL,
    UNIQUE (uri, object_version)
);

CREATE TABLE IF NOT EXISTS artifact_edges (
    child_artifact_id TEXT NOT NULL
        REFERENCES artifacts (artifact_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    parent_artifact_id TEXT NOT NULL
        REFERENCES artifacts (artifact_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    relation TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (child_artifact_id, parent_artifact_id, relation),
    UNIQUE (child_artifact_id, ordinal)
);

CREATE TABLE IF NOT EXISTS derivations (
    logical_key TEXT PRIMARY KEY,
    manifest_artifact_id TEXT NOT NULL
        REFERENCES artifacts (artifact_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    snapshot_json BLOB NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS derivation_artifacts (
    logical_key TEXT NOT NULL
        REFERENCES derivations (logical_key) ON UPDATE RESTRICT ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    artifact_id TEXT NOT NULL
        REFERENCES artifacts (artifact_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    PRIMARY KEY (logical_key, ordinal),
    UNIQUE (logical_key, artifact_id)
);

CREATE INDEX IF NOT EXISTS artifact_edges_parent_idx
    ON artifact_edges (parent_artifact_id);
PRAGMA user_version = 1;
"""


def deterministic_local_artifact_id(
    artifact_type: ArtifactType,
    semantic_sha256: str,
) -> str:
    """Allocate a stable opaque local ID from type plus semantic identity only."""

    if not isinstance(artifact_type, ArtifactType):
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INVALID_REQUEST,
            "artifact_type must be an ArtifactType",
        )
    if not isinstance(semantic_sha256, str) or _SHA256_PATTERN.fullmatch(semantic_sha256) is None:
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INVALID_REQUEST,
            "semantic_sha256 must be a lowercase SHA-256 digest",
        )
    identity_hex = hashlib.sha256(
        _ARTIFACT_ID_DOMAIN
        + artifact_type.value.encode("ascii")
        + b"\x00"
        + semantic_sha256.encode("ascii")
    ).hexdigest()[:32]
    return (
        f"{identity_hex[:8]}-{identity_hex[8:12]}-{identity_hex[12:16]}-"
        f"{identity_hex[16:20]}-{identity_hex[20:]}"
    )


# Short alias for callers that treat allocation as part of publication planning.
allocate_local_artifact_id = deterministic_local_artifact_id


class LocalArtifactRegistry:
    """Local durable registry whose SQLite commit is the publication boundary."""

    def __init__(
        self,
        root: Path,
        *,
        runtime_observer: RuntimeObserver | None = None,
        hardlink_artifact_types: frozenset[ArtifactType] = frozenset(),
    ) -> None:
        if not isinstance(root, Path):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "registry root must be a pathlib.Path",
            )
        if not isinstance(hardlink_artifact_types, frozenset) or any(
            not isinstance(value, ArtifactType) for value in hardlink_artifact_types
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "hardlink_artifact_types must be a frozenset of ArtifactType values",
            )
        self._runtime_observer = runtime_observer
        self._hardlink_artifact_types = hardlink_artifact_types
        try:
            if root.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_REQUEST,
                    f"registry root must not be a symlink: {root}",
                )
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_REQUEST,
                    f"registry root is not a directory: {root}",
                )
            self._root = root.resolve(strict=True)
            blob_directory = self._root / "blobs"
            if blob_directory.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_REQUEST,
                    f"blob directory must not be a symlink: {blob_directory}",
                )
            blob_directory.mkdir(parents=True, exist_ok=True)
            self._blob_root = blob_directory / "sha256"
            if self._blob_root.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_REQUEST,
                    f"blob root must not be a symlink: {self._blob_root}",
                )
            self._blob_root.mkdir(parents=True, exist_ok=True)
            self._database_path = self._root / "registry.sqlite3"
            if self._database_path.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_REQUEST,
                    f"registry database must not be a symlink: {self._database_path}",
                )
        except ArtifactRegistryError:
            raise
        except OSError as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot initialize registry storage at {root}: {error}",
            ) from error
        with runtime_span(
            self._runtime_observer,
            "sqlite.artifact_registry.initialization",
        ):
            self._initialize_database()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def blob_root(self) -> Path:
        return self._blob_root

    @staticmethod
    def artifact_id_for(artifact_type: ArtifactType, semantic_sha256: str) -> str:
        return deterministic_local_artifact_id(artifact_type, semantic_sha256)

    def allocate_artifact_id(
        self,
        artifact_type: ArtifactType,
        semantic_sha256: str,
    ) -> str:
        """Allocate a deterministic ID before an entry or snapshot is built."""

        return deterministic_local_artifact_id(artifact_type, semantic_sha256)

    def publish_derivation(
        self,
        *,
        snapshot: ArtifactRegistrySnapshot,
        logical_key: str,
        manifest_artifact_id: str,
        blob_sources: Mapping[str, ArtifactBlobSource],
        trusted_artifact_ids: frozenset[str] = frozenset(),
        verify_blobs: bool = True,
    ) -> PublishedArtifactDerivation:
        """Put exact blobs, then publish all metadata in one SQLite transaction.

        ``trusted_artifact_ids`` is deliberately narrow: callers may use it only for
        files whose digest/size were verified earlier in this same private staging
        transaction. It avoids re-reading a just-verified immutable source after it
        has been hard-linked into the registry. Reuse and the default public path
        remain fully content-verified.
        """

        checked_key = self._validate_logical_key(logical_key)
        checked_snapshot = self._validate_snapshot(snapshot, manifest_artifact_id)
        checked_sources = self._validate_blob_sources(checked_snapshot, blob_sources)
        checked_trusted_ids = self._validate_trusted_artifact_ids(
            trusted_artifact_ids,
            snapshot=checked_snapshot,
        )
        if not isinstance(verify_blobs, bool):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "verify_blobs must be a boolean",
            )
        snapshot_bytes = canonical_json_bytes(checked_snapshot)
        snapshot_sha256 = exact_bytes_sha256(snapshot_bytes)

        self._assert_storage_layout()
        for entry in checked_snapshot.entries:
            self._put_blob(
                entry,
                checked_sources[entry.artifact_id],
                trusted_source=entry.artifact_id in checked_trusted_ids,
            )

        reused = self._publish_transaction(
            snapshot=checked_snapshot,
            snapshot_bytes=snapshot_bytes,
            snapshot_sha256=snapshot_sha256,
            logical_key=checked_key,
            manifest_artifact_id=manifest_artifact_id,
        )
        # A competing publisher may have won this logical key; reused derivations
        # always take the full integrity path even when the fresh hand-off was trusted.
        verified_snapshot = self.verify_derivation(
            checked_key,
            verify_blobs=verify_blobs or reused,
        )
        return PublishedArtifactDerivation(
            logical_key=checked_key,
            manifest_artifact_id=manifest_artifact_id,
            snapshot=verified_snapshot,
            reused=reused,
        )

    def lookup_derivation(self, logical_key: str) -> PublishedArtifactDerivation | None:
        """Return only a derivation whose metadata and every blob still verify."""

        checked_key = self._validate_logical_key(logical_key)
        if not self._derivation_exists(checked_key):
            return None
        snapshot = self.verify_derivation(checked_key)
        manifests = tuple(
            entry
            for entry in snapshot.entries
            if entry.artifact_type is ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST
        )
        if len(manifests) != 1:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"derivation {checked_key!r} does not have exactly one manifest",
            )
        return PublishedArtifactDerivation(
            logical_key=checked_key,
            manifest_artifact_id=manifests[0].artifact_id,
            snapshot=snapshot,
            reused=True,
        )

    def lookup_artifact(
        self,
        artifact_type: ArtifactType,
        semantic_sha256: str,
    ) -> ArtifactRegistryEntry | None:
        """Return a verified existing entry by type and semantic digest."""

        expected_id = deterministic_local_artifact_id(artifact_type, semantic_sha256)
        self._assert_storage_layout()
        connection = self._connect()
        try:
            with self._observed_transaction_scope(
                connection,
                operation="lookup_artifact",
                write=False,
            ):
                row = connection.execute(
                    """
                    SELECT artifact_id, entry_json
                    FROM artifacts
                    WHERE artifact_type = ? AND semantic_sha256 = ?
                    """,
                    (artifact_type.value, semantic_sha256),
                ).fetchone()
                if row is None:
                    return None
                artifact_id = _row_text(row, "artifact_id")
                if artifact_id != expected_id:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                        "semantic index points to a non-deterministic artifact ID",
                    )
                entry_bytes = _row_bytes(row, "entry_json")
                try:
                    entry = ArtifactRegistryEntry.model_validate_json(entry_bytes, strict=True)
                except (ValidationError, ValueError) as error:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                        f"stored artifact entry is invalid for {artifact_id}: {error}",
                    ) from error
                if (
                    canonical_json_bytes(entry) != entry_bytes
                    or entry.artifact_id != artifact_id
                    or entry.artifact_type is not artifact_type
                    or entry.semantic_sha256 != semantic_sha256
                ):
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                        f"stored artifact entry disagrees with semantic index: {artifact_id}",
                    )
                self._verify_registered_entry(connection, entry)
                anchor_rows = connection.execute(
                    """
                    SELECT d.logical_key, d.manifest_artifact_id,
                           d.snapshot_json, d.snapshot_sha256
                    FROM derivation_artifacts AS da
                    JOIN derivations AS d ON d.logical_key = da.logical_key
                    WHERE da.artifact_id = ?
                    ORDER BY d.logical_key
                    """,
                    (artifact_id,),
                ).fetchall()
                if not anchor_rows:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                        f"artifact has no committed derivation anchor: {artifact_id}",
                    )
                for anchor_row in anchor_rows:
                    anchor_key = _row_text(anchor_row, "logical_key")
                    anchor_snapshot = self._snapshot_from_derivation_row(anchor_row, anchor_key)
                    self._verify_derivation_membership(connection, anchor_key, anchor_snapshot)
                    anchored_entry = next(
                        (
                            candidate
                            for candidate in anchor_snapshot.entries
                            if candidate.artifact_id == artifact_id
                        ),
                        None,
                    )
                    if anchored_entry != entry:
                        raise ArtifactRegistryError(
                            ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                            f"artifact entry disagrees with derivation anchor: {artifact_id}",
                        )
                self._verify_database_graph(connection)
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up artifact by semantic identity: {error}",
            ) from error
        finally:
            connection.close()

        self._verify_blob_file(
            self._blob_path(entry.sha256),
            expected_sha256=entry.sha256,
            expected_bytes=entry.bytes,
            error_code=ArtifactRegistryErrorCode.INTEGRITY_ERROR,
        )
        return entry

    def load_snapshot(self, logical_key: str) -> ArtifactRegistrySnapshot:
        """Load a canonical snapshot without treating it as verified reusable output."""

        checked_key = self._validate_logical_key(logical_key)
        self._assert_storage_layout()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT manifest_artifact_id, snapshot_json, snapshot_sha256
                FROM derivations
                WHERE logical_key = ?
                """,
                (checked_key,),
            ).fetchone()
            if row is None:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.DERIVATION_NOT_FOUND,
                    f"artifact derivation is not registered: {checked_key!r}",
                )
            return self._snapshot_from_derivation_row(row, checked_key)
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot load derivation {checked_key!r}: {error}",
            ) from error
        finally:
            connection.close()

    def resolve_blob(self, artifact_id: str) -> Path:
        """Resolve and rehash one exact blob from authoritative registry metadata."""

        checked_id = self._validate_artifact_id(artifact_id)
        self._assert_storage_layout()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT exact_sha256, byte_count
                FROM artifacts
                WHERE artifact_id = ?
                """,
                (checked_id,),
            ).fetchone()
            if row is None:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.ARTIFACT_NOT_FOUND,
                    f"artifact is not registered: {checked_id}",
                )
            digest = _row_text(row, "exact_sha256")
            byte_count = _row_int(row, "byte_count")
            if _SHA256_PATTERN.fullmatch(digest) is None or byte_count <= 0:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"artifact metadata is corrupt: {checked_id}",
                )
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot resolve artifact {checked_id}: {error}",
            ) from error
        finally:
            connection.close()

        blob_path = self._blob_path(digest)
        self._verify_blob_file(
            blob_path,
            expected_sha256=digest,
            expected_bytes=byte_count,
            error_code=ArtifactRegistryErrorCode.INTEGRITY_ERROR,
        )
        return blob_path

    def resolve_blob_unverified(self, artifact_id: str) -> Path:
        """Resolve a just-published blob after metadata and file-shape checks only.

        This is intentionally not part of the registry protocol. It is consumed only by
        the same-process fresh-publication hand-off, after ``publish_derivation`` has
        verified the exact source bytes and atomically linked the blob.
        """

        checked_id = self._validate_artifact_id(artifact_id)
        self._assert_storage_layout()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT exact_sha256, byte_count
                FROM artifacts
                WHERE artifact_id = ?
                """,
                (checked_id,),
            ).fetchone()
            if row is None:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.ARTIFACT_NOT_FOUND,
                    f"artifact is not registered: {checked_id}",
                )
            digest = _row_text(row, "exact_sha256")
            byte_count = _row_int(row, "byte_count")
            if _SHA256_PATTERN.fullmatch(digest) is None or byte_count <= 0:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"artifact metadata is corrupt: {checked_id}",
                )
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot resolve artifact {checked_id}: {error}",
            ) from error
        finally:
            connection.close()

        blob_path = self._blob_path(digest)
        try:
            blob_stat = blob_path.lstat()
        except OSError as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot inspect content-addressed blob {blob_path}: {error}",
            ) from error
        if (
            blob_path.is_symlink()
            or not stat.S_ISREG(blob_stat.st_mode)
            or blob_stat.st_size != byte_count
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"content-addressed blob has invalid shape: {blob_path}",
            )
        return blob_path

    def reconcile(
        self,
        *,
        remove_orphans: bool = False,
        remove_partials: bool = False,
        remove_duplicates: bool = False,
        strict: bool = False,
    ) -> ArtifactRegistryReconciliation:
        '''Reconcile registry metadata with exact blobs and crash leftovers.

        SQLite metadata remains authoritative. Missing or corrupt registered blobs are
        reported and never deleted; only unreferenced files can be removed, and cleanup
        is opt-in. This makes a restart after blob-first publication deterministic while
        preserving evidence for an operator or a later object-store repair.
        '''

        for name, value in (
            ('remove_orphans', remove_orphans),
            ('remove_partials', remove_partials),
            ('remove_duplicates', remove_duplicates),
            ('strict', strict),
        ):
            if not isinstance(value, bool):
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_REQUEST,
                    f'{name} must be a boolean',
                )

        self._assert_storage_layout()
        with runtime_span(self._runtime_observer, 'sqlite.artifact_registry.reconcile'):
            metadata = self._registered_blob_metadata()
            expected_digests = {digest for _artifact_id, digest, _bytes in metadata}
            missing: list[str] = []
            corrupt: list[str] = []
            visible = 0
            for artifact_id, digest, byte_count in metadata:
                if _SHA256_PATTERN.fullmatch(digest) is None:
                    # Corrupt registry metadata is an observed issue, not a
                    # reason for reconciliation itself to crash before reporting.
                    corrupt.append(artifact_id)
                    continue
                path = self._blob_path(digest)
                try:
                    self._verify_blob_file(
                        path,
                        expected_sha256=digest,
                        expected_bytes=byte_count,
                        error_code=ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    )
                except ArtifactRegistryError:
                    try:
                        path.lstat()
                    except FileNotFoundError:
                        missing.append(artifact_id)
                    except OSError:
                        corrupt.append(artifact_id)
                    else:
                        corrupt.append(artifact_id)
                else:
                    visible += 1

            orphan, partial, duplicate = self._scan_blob_paths(expected_digests)
            cleanup_candidates: list[Path] = []
            if remove_orphans:
                cleanup_candidates.extend(orphan)
            if remove_partials:
                cleanup_candidates.extend(partial)
            if remove_duplicates:
                cleanup_candidates.extend(duplicate)
            removed, failed_removal = self._remove_reconciliation_paths(cleanup_candidates)
            removed_set = set(removed)
            orphan = [path for path in orphan if path not in removed_set]
            partial = [path for path in partial if path not in removed_set]
            duplicate = [path for path in duplicate if path not in removed_set]
            report = ArtifactRegistryReconciliation(
                registered_artifact_count=len(metadata),
                visible_artifact_count=visible,
                missing_artifact_ids=tuple(sorted(missing)),
                corrupt_artifact_ids=tuple(sorted(corrupt)),
                orphan_blob_paths=tuple(sorted(orphan, key=lambda path: path.as_posix())),
                partial_blob_paths=tuple(sorted(partial, key=lambda path: path.as_posix())),
                duplicate_blob_paths=tuple(sorted(duplicate, key=lambda path: path.as_posix())),
                removed_blob_paths=tuple(sorted(removed, key=lambda path: path.as_posix())),
                failed_removal_paths=tuple(
                    sorted(failed_removal, key=lambda path: path.as_posix())
                ),
            )
            runtime_increment(
                self._runtime_observer,
                'sqlite.artifact_registry.reconciliation_runs',
                attributes={'strict': strict},
            )
            if report.issue_count:
                runtime_increment(
                    self._runtime_observer,
                    'sqlite.artifact_registry.reconciliation_issues',
                    value=report.issue_count,
                    attributes={'strict': strict},
                )
            if report.removed_blob_paths:
                runtime_increment(
                    self._runtime_observer,
                    'sqlite.artifact_registry.reconciliation_removed',
                    value=len(report.removed_blob_paths),
                    attributes={'strict': strict},
                )

        if strict and not report.reconciled:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                'artifact registry reconciliation found unresolved storage discrepancies',
            )
        return report

    def reconcile_storage(self, **kwargs: object) -> ArtifactRegistryReconciliation:
        '''Compatibility alias for callers naming the backing store explicitly.'''

        return self.reconcile(**kwargs)  # type: ignore[arg-type]

    def reconcile_blobs(self, **kwargs: object) -> ArtifactRegistryReconciliation:
        '''Compatibility alias for blob-oriented repair tooling.'''

        return self.reconcile(**kwargs)  # type: ignore[arg-type]

    def _registered_blob_metadata(self) -> tuple[tuple[str, str, int], ...]:
        connection = self._connect()
        try:
            with self._observed_transaction_scope(
                connection,
                operation='reconcile_metadata',
                write=False,
            ):
                rows = connection.execute(
                    '''
                    SELECT artifact_id, exact_sha256, byte_count
                    FROM artifacts
                    ORDER BY artifact_id
                    '''
                ).fetchall()
                return tuple(
                    (
                        _row_text(row, 'artifact_id'),
                        _row_text(row, 'exact_sha256'),
                        _row_int(row, 'byte_count'),
                    )
                    for row in rows
                )
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f'cannot read artifact metadata for reconciliation: {error}',
            ) from error
        finally:
            connection.close()

    def _scan_blob_paths(
        self,
        registered_digests: set[str],
    ) -> tuple[list[Path], list[Path], list[Path]]:
        orphan: list[Path] = []
        partial: list[Path] = []
        duplicate: list[Path] = []
        for path in sorted(self._blob_root.rglob('*'), key=lambda value: value.as_posix()):
            try:
                file_stat = path.lstat()
            except OSError:
                partial.append(path)
                continue
            if stat.S_ISDIR(file_stat.st_mode):
                continue
            if path.is_symlink():
                partial.append(path)
                continue
            relative = path.relative_to(self._blob_root)
            name = path.name
            if name.startswith('.put-'):
                partial.append(path)
                continue
            digest_match = _SHA256_PATTERN.fullmatch(name)
            canonical = len(relative.parts) == 2 and relative.parts[0] == name[:2]
            if digest_match is None:
                orphan.append(path)
                continue
            digest = name
            if digest in registered_digests:
                expected_path = self._blob_path(digest)
                if path != expected_path:
                    duplicate.append(path)
                continue
            if not canonical:
                orphan.append(path)
                continue
            try:
                self._verify_blob_file(
                    path,
                    expected_sha256=digest,
                    expected_bytes=file_stat.st_size,
                    error_code=ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                )
            except ArtifactRegistryError:
                partial.append(path)
            else:
                orphan.append(path)
        return orphan, partial, duplicate

    def _remove_reconciliation_paths(
        self,
        paths: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        removed: list[Path] = []
        failed: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                path.relative_to(self._blob_root)
                path.unlink()
            except (OSError, ValueError):
                failed.append(path)
            else:
                removed.append(path)
        return removed, failed

    def verify_derivation(
        self,
        logical_key: str,
        *,
        verify_blobs: bool = True,
    ) -> ArtifactRegistrySnapshot:
        """Verify the snapshot anchor, normalized rows, typed DAG, and optional blobs."""

        if not isinstance(verify_blobs, bool):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "verify_blobs must be a boolean",
            )
        checked_key = self._validate_logical_key(logical_key)
        self._assert_storage_layout()
        connection = self._connect()
        try:
            with self._observed_transaction_scope(
                connection,
                operation="verify_derivation",
                write=False,
            ):
                row = connection.execute(
                    """
                    SELECT manifest_artifact_id, snapshot_json, snapshot_sha256
                    FROM derivations
                    WHERE logical_key = ?
                    """,
                    (checked_key,),
                ).fetchone()
                if row is None:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.DERIVATION_NOT_FOUND,
                        f"artifact derivation is not registered: {checked_key!r}",
                    )
                snapshot = self._snapshot_from_derivation_row(row, checked_key)
                self._verify_derivation_membership(connection, checked_key, snapshot)
                for entry in snapshot.entries:
                    self._verify_registered_entry(connection, entry)
                self._verify_database_graph(connection)
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot verify derivation {checked_key!r}: {error}",
            ) from error
        finally:
            connection.close()

        if verify_blobs:
            for entry in snapshot.entries:
                self._verify_blob_file(
                    self._blob_path(entry.sha256),
                    expected_sha256=entry.sha256,
                    expected_bytes=entry.bytes,
                    error_code=ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                )
        return snapshot

    def _initialize_database(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            user_version = _pragma_int(connection, "user_version")
            if user_version not in (0, 1):
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"unsupported local registry schema version: {user_version}",
                )
            journal_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            journal_mode: object = None if journal_row is None else journal_row[0]
            if not isinstance(journal_mode, str) or journal_mode.lower() != "wal":
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                    "SQLite WAL mode could not be enabled",
                )
            connection.executescript(_SCHEMA_SQL)
            if _pragma_int(connection, "foreign_keys") != 1:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                    "SQLite foreign-key enforcement could not be enabled",
                )
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.transactions",
                attributes={"operation": "initialize_schema", "write": True},
            )
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.commits",
                attributes={"operation": "initialize_schema", "write": True},
            )
        except ArtifactRegistryError:
            raise
        except sqlite3.Error as error:
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.transaction_outcomes_unknown",
                attributes={"operation": "initialize_schema", "write": True},
            )
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot initialize SQLite artifact registry: {error}",
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        if self._database_path.is_symlink():
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"registry database became a symlink: {self._database_path}",
            )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=30.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA trusted_schema = OFF")
            return connection
        except sqlite3.Error as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot open SQLite artifact registry: {error}",
            ) from error

    @contextmanager
    def _observed_transaction_scope(
        self,
        connection: sqlite3.Connection,
        *,
        operation: str,
        write: bool,
    ) -> Iterator[None]:
        attributes: dict[str, RuntimeAttributeValue] = {
            "operation": operation,
            "write": write,
        }
        with runtime_span(
            self._runtime_observer,
            "sqlite.artifact_registry.transaction",
            attributes,
        ):
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.transactions",
                attributes=attributes,
            )
            try:
                yield
            except BaseException:
                self._rollback_observed(
                    connection,
                    attributes=attributes,
                    outcome_unknown_if_inactive=True,
                    suppress_errors=True,
                )
                raise
            try:
                self._commit(connection)
            except BaseException:
                runtime_increment(
                    self._runtime_observer,
                    "sqlite.artifact_registry.commit_failures",
                    attributes=attributes,
                )
                self._rollback_observed(
                    connection,
                    attributes=attributes,
                    outcome_unknown_if_inactive=True,
                    suppress_errors=True,
                )
                raise
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.commits",
                attributes=attributes,
            )

    def _rollback_observed(
        self,
        connection: sqlite3.Connection,
        *,
        attributes: Mapping[str, RuntimeAttributeValue],
        outcome_unknown_if_inactive: bool,
        suppress_errors: bool,
    ) -> None:
        if not connection.in_transaction:
            if outcome_unknown_if_inactive:
                runtime_increment(
                    self._runtime_observer,
                    "sqlite.artifact_registry.transaction_outcomes_unknown",
                    attributes=attributes,
                )
            return
        try:
            self._rollback(connection)
        except BaseException:
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.rollback_failures",
                attributes=attributes,
            )
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.transaction_outcomes_unknown",
                attributes=attributes,
            )
            if not suppress_errors:
                raise
        else:
            runtime_increment(
                self._runtime_observer,
                "sqlite.artifact_registry.rollbacks",
                attributes=attributes,
            )

    def _assert_storage_layout(self) -> None:
        try:
            if not self._root.is_dir() or self._root.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"registry root is no longer a regular directory: {self._root}",
                )
            blob_directory = self._blob_root.parent
            if not blob_directory.is_dir() or blob_directory.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"blob directory is no longer a regular directory: {blob_directory}",
                )
            if not self._blob_root.is_dir() or self._blob_root.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"blob root is no longer a regular directory: {self._blob_root}",
                )
            if self._database_path.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"registry database became a symlink: {self._database_path}",
                )
        except ArtifactRegistryError:
            raise
        except OSError as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot inspect registry storage: {error}",
            ) from error

    def _validate_logical_key(self, logical_key: str) -> str:
        if (
            not isinstance(logical_key, str)
            or not logical_key
            or logical_key.strip() != logical_key
            or "\x00" in logical_key
            or len(logical_key) > 4096
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "logical_key must be a nonempty, trimmed string of at most 4096 characters",
            )
        return logical_key

    def _validate_artifact_id(self, artifact_id: str) -> str:
        if not isinstance(artifact_id, str) or _ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "artifact_id must be a lowercase opaque UUID-shaped value",
            )
        return artifact_id

    def _validate_snapshot(
        self,
        snapshot: ArtifactRegistrySnapshot,
        manifest_artifact_id: str,
    ) -> ArtifactRegistrySnapshot:
        checked_manifest_id = self._validate_artifact_id(manifest_artifact_id)
        if not isinstance(snapshot, ArtifactRegistrySnapshot):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                "snapshot must be an ArtifactRegistrySnapshot",
            )
        try:
            self._validate_snapshot_graph(snapshot, checked_manifest_id)
        except ArtifactRegistryError:
            raise
        except (AttributeError, TypeError, ValueError) as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                f"artifact registry snapshot is malformed: {error}",
            ) from error
        try:
            validated = ArtifactRegistrySnapshot.model_validate(
                snapshot.model_dump(mode="python"),
                strict=True,
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                f"artifact registry snapshot is invalid: {error}",
            ) from error
        self._validate_snapshot_graph(validated, checked_manifest_id)
        return validated

    def _validate_snapshot_graph(
        self,
        snapshot: ArtifactRegistrySnapshot,
        manifest_artifact_id: str,
    ) -> None:
        entries = snapshot.entries
        if not entries:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                "artifact registry snapshot must not be empty",
            )
        if any(not isinstance(entry, ArtifactRegistryEntry) for entry in entries):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                "snapshot entries must be ArtifactRegistryEntry values",
            )
        entries_by_id = {entry.artifact_id: entry for entry in entries}
        if len(entries_by_id) != len(entries):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                "snapshot artifact IDs must be unique",
            )
        for entry in entries:
            if not isinstance(entry.artifact_type, ArtifactType) or not isinstance(
                entry.semantic_sha256,
                str,
            ):
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                    f"artifact entry has an invalid semantic identity: {entry.artifact_id}",
                )
            expected_id = deterministic_local_artifact_id(
                entry.artifact_type,
                entry.semantic_sha256,
            )
            if entry.artifact_id != expected_id:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.ARTIFACT_ID_MISMATCH,
                    f"artifact ID does not match its local semantic identity: {entry.artifact_id}",
                )
            for parent in entry.parents:
                if not isinstance(parent, ArtifactParent):
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.INVALID_SNAPSHOT,
                        f"artifact {entry.artifact_id} has an invalid parent value",
                    )
                if parent.artifact_id not in entries_by_id:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.MISSING_PARENT,
                        f"parent {parent.artifact_id} is absent for artifact {entry.artifact_id}",
                    )
            schema_reference = entry.payload_schema_ref
            if schema_reference is not None and schema_reference.artifact_id not in entries_by_id:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.MISSING_PARENT,
                    f"schema artifact {schema_reference.artifact_id} is absent",
                )

        manifest = entries_by_id.get(manifest_artifact_id)
        if manifest is None:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.ARTIFACT_NOT_FOUND,
                f"manifest artifact is absent from snapshot: {manifest_artifact_id}",
            )
        if manifest.artifact_type is not ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "manifest_artifact_id must identify a camera-video export manifest",
            )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(artifact_id: str) -> None:
            if artifact_id in visiting:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.GRAPH_CYCLE,
                    f"artifact parent graph contains a cycle at {artifact_id}",
                )
            if artifact_id in visited:
                return
            visiting.add(artifact_id)
            for parent in entries_by_id[artifact_id].parents:
                visit(parent.artifact_id)
            visiting.remove(artifact_id)
            visited.add(artifact_id)

        for artifact_id in entries_by_id:
            visit(artifact_id)

        reachable: set[str] = set()

        def collect_dependencies(artifact_id: str) -> None:
            if artifact_id in reachable:
                return
            reachable.add(artifact_id)
            entry = entries_by_id[artifact_id]
            for parent in entry.parents:
                collect_dependencies(parent.artifact_id)
            if entry.payload_schema_ref is not None:
                collect_dependencies(entry.payload_schema_ref.artifact_id)

        collect_dependencies(manifest_artifact_id)
        disconnected = sorted(set(entries_by_id) - reachable)
        if disconnected:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INCOMPLETE_DAG,
                "snapshot contains artifacts outside the manifest dependency closure: "
                + ", ".join(disconnected),
            )

    def _validate_trusted_artifact_ids(
        self,
        trusted_artifact_ids: frozenset[str],
        *,
        snapshot: ArtifactRegistrySnapshot,
    ) -> frozenset[str]:
        if not isinstance(trusted_artifact_ids, frozenset) or any(
            not isinstance(artifact_id, str) for artifact_id in trusted_artifact_ids
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "trusted_artifact_ids must be a frozen set of artifact ID strings",
            )
        expected_ids = {entry.artifact_id for entry in snapshot.entries}
        unknown = sorted(trusted_artifact_ids - expected_ids)
        if unknown:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "trusted artifact IDs are absent from the snapshot: " + ", ".join(unknown),
            )
        return trusted_artifact_ids

    def _validate_blob_sources(
        self,
        snapshot: ArtifactRegistrySnapshot,
        blob_sources: Mapping[str, ArtifactBlobSource],
    ) -> dict[str, ArtifactBlobSource]:
        if not isinstance(blob_sources, Mapping):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                "blob_sources must be a mapping keyed by artifact ID",
            )
        try:
            copied = dict(blob_sources)
        except (TypeError, ValueError) as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INVALID_REQUEST,
                f"blob_sources cannot be read: {error}",
            ) from error
        if any(not isinstance(artifact_id, str) for artifact_id in copied):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.BLOB_SOURCE_INVALID,
                "blob source mapping keys must be artifact ID strings",
            )
        expected_ids = {entry.artifact_id for entry in snapshot.entries}
        actual_ids = set(copied)
        missing = sorted(expected_ids - actual_ids)
        if missing:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.BLOB_SOURCE_MISSING,
                "blob source is missing for artifact IDs: " + ", ".join(missing),
            )
        unexpected = sorted(actual_ids - expected_ids)
        if unexpected:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.BLOB_SOURCE_UNEXPECTED,
                "unexpected blob source artifact IDs: " + ", ".join(unexpected),
            )
        for artifact_id, source in copied.items():
            if not isinstance(artifact_id, str) or not isinstance(source, (Path, bytes)):
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.BLOB_SOURCE_INVALID,
                    f"blob source for {artifact_id!r} must be pathlib.Path or bytes",
                )
        return copied

    def _put_blob(
        self,
        entry: ArtifactRegistryEntry,
        source: ArtifactBlobSource,
        *,
        trusted_source: bool = False,
    ) -> None:
        destination = self._blob_path(entry.sha256)
        shard = destination.parent
        try:
            if shard.is_symlink():
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"blob shard must not be a symlink: {shard}",
                )
            shard.mkdir(parents=True, exist_ok=True)
            if (
                trusted_source
                and isinstance(source, Path)
                and self._put_trusted_hardlink(entry, source, destination, shard)
            ):
                return
            if (
                isinstance(source, Path)
                and entry.artifact_type in self._hardlink_artifact_types
                and self._put_verified_hardlink(entry, source, destination, shard)
            ):
                return
            descriptor, temp_name = tempfile.mkstemp(prefix=".put-", dir=shard)
            temp_path = Path(temp_name)
            try:
                actual_bytes, actual_sha256 = self._write_blob_temp(
                    descriptor,
                    temp_path,
                    source,
                )
                if actual_bytes != entry.bytes:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.BLOB_SIZE_MISMATCH,
                        f"artifact {entry.artifact_id} expected {entry.bytes} bytes, "
                        f"got {actual_bytes}",
                    )
                if actual_sha256 != entry.sha256:
                    raise ArtifactRegistryError(
                        ArtifactRegistryErrorCode.BLOB_DIGEST_MISMATCH,
                        f"artifact {entry.artifact_id} digest does not match its registry entry",
                    )
                if destination.exists() or destination.is_symlink():
                    self._verify_blob_file(
                        destination,
                        expected_sha256=entry.sha256,
                        expected_bytes=entry.bytes,
                        error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
                    )
                    return
                try:
                    os.link(temp_path, destination)
                except FileExistsError:
                    self._verify_blob_file(
                        destination,
                        expected_sha256=entry.sha256,
                        expected_bytes=entry.bytes,
                        error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
                    )
                self._verify_blob_file(
                    destination,
                    expected_sha256=entry.sha256,
                    expected_bytes=entry.bytes,
                    error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
                )
                _fsync_file(destination)
                _fsync_directory(shard)
            finally:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
        except ArtifactRegistryError:
            raise
        except FileNotFoundError as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.BLOB_SOURCE_MISSING,
                f"blob source is missing for artifact {entry.artifact_id}: {error}",
            ) from error
        except OSError as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
                f"cannot store blob for artifact {entry.artifact_id}: {error}",
            ) from error

    def _put_trusted_hardlink(
        self,
        entry: ArtifactRegistryEntry,
        source: Path,
        destination: Path,
        shard: Path,
    ) -> bool:
        """Link bytes already verified by this private publication transaction."""

        source_stat = source.lstat()
        if source.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.BLOB_SOURCE_INVALID,
                f"blob source must be a regular non-symlink file: {source}",
            )
        if source_stat.st_size != entry.bytes:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.BLOB_SIZE_MISMATCH,
                f"trusted artifact {entry.artifact_id} size changed before registration",
            )
        if destination.exists() or destination.is_symlink():
            self._verify_blob_file(
                destination,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
                error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
            )
            return True
        try:
            os.link(source, destination)
        except FileExistsError:
            self._verify_blob_file(
                destination,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
                error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
            )
        except OSError:
            return False
        _fsync_file(destination)
        _fsync_directory(shard)
        return True

    def _put_verified_hardlink(
        self,
        entry: ArtifactRegistryEntry,
        source: Path,
        destination: Path,
        shard: Path,
    ) -> bool:
        self._verify_blob_file(
            source,
            expected_sha256=entry.sha256,
            expected_bytes=entry.bytes,
            error_code=ArtifactRegistryErrorCode.BLOB_DIGEST_MISMATCH,
        )
        if destination.exists() or destination.is_symlink():
            self._verify_blob_file(
                destination,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
                error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
            )
            return True
        try:
            os.link(source, destination)
        except FileExistsError:
            self._verify_blob_file(
                destination,
                expected_sha256=entry.sha256,
                expected_bytes=entry.bytes,
                error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
            )
        except OSError:
            return False
        self._verify_blob_file(
            destination,
            expected_sha256=entry.sha256,
            expected_bytes=entry.bytes,
            error_code=ArtifactRegistryErrorCode.BLOB_CONFLICT,
        )
        _fsync_file(destination)
        _fsync_directory(shard)
        return True

    def _write_blob_temp(
        self,
        descriptor: int,
        temp_path: Path,
        source: ArtifactBlobSource,
    ) -> tuple[int, str]:
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                if isinstance(source, bytes):
                    output.write(source)
                    digest.update(source)
                    byte_count = len(source)
                else:
                    source_stat = source.lstat()
                    if source.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
                        raise ArtifactRegistryError(
                            ArtifactRegistryErrorCode.BLOB_SOURCE_INVALID,
                            f"blob source must be a regular non-symlink file: {source}",
                        )
                    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    source_descriptor = os.open(source, flags)
                    with os.fdopen(source_descriptor, "rb") as input_stream:
                        if not stat.S_ISREG(os.fstat(input_stream.fileno()).st_mode):
                            raise ArtifactRegistryError(
                                ArtifactRegistryErrorCode.BLOB_SOURCE_INVALID,
                                f"blob source must be a regular file: {source}",
                            )
                        while chunk := input_stream.read(_READ_CHUNK_BYTES):
                            output.write(chunk)
                            digest.update(chunk)
                            byte_count += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise
        return byte_count, digest.hexdigest()

    def _verify_blob_file(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
        error_code: ArtifactRegistryErrorCode,
    ) -> None:
        try:
            file_stat = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                raise ArtifactRegistryError(
                    error_code,
                    f"content-addressed blob is not a regular non-symlink file: {path}",
                )
            digest = hashlib.sha256()
            byte_count = 0
            with path.open("rb") as stream:
                while chunk := stream.read(_READ_CHUNK_BYTES):
                    digest.update(chunk)
                    byte_count += len(chunk)
        except ArtifactRegistryError:
            raise
        except OSError as error:
            raise ArtifactRegistryError(
                error_code,
                f"cannot read content-addressed blob {path}: {error}",
            ) from error
        if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ArtifactRegistryError(
                error_code,
                f"content-addressed blob does not match digest and size metadata: {path}",
            )

    def _blob_path(self, sha256: str) -> Path:
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"invalid content digest in registry metadata: {sha256!r}",
            )
        return self._blob_root / sha256[:2] / sha256

    def _publish_transaction(
        self,
        *,
        snapshot: ArtifactRegistrySnapshot,
        snapshot_bytes: bytes,
        snapshot_sha256: str,
        logical_key: str,
        manifest_artifact_id: str,
    ) -> bool:
        connection = self._connect()
        try:
            with self._observed_transaction_scope(
                connection,
                operation="publish_derivation",
                write=True,
            ):
                reused = self._check_existing_derivation(
                    connection,
                    logical_key=logical_key,
                    manifest_artifact_id=manifest_artifact_id,
                    snapshot_bytes=snapshot_bytes,
                    snapshot_sha256=snapshot_sha256,
                )
                inserted: dict[str, bool] = {}
                for entry in snapshot.entries:
                    inserted[entry.artifact_id] = self._insert_or_verify_entry(connection, entry)
                for entry in snapshot.entries:
                    self._insert_or_verify_edges(
                        connection,
                        entry,
                        newly_inserted=inserted[entry.artifact_id],
                    )
                self._verify_database_graph(connection)
                if reused:
                    self._verify_derivation_membership(connection, logical_key, snapshot)
                else:
                    manifest = next(
                        entry
                        for entry in snapshot.entries
                        if entry.artifact_id == manifest_artifact_id
                    )
                    connection.execute(
                        """
                        INSERT INTO derivations (
                            logical_key, manifest_artifact_id, snapshot_json,
                            snapshot_sha256, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            logical_key,
                            manifest_artifact_id,
                            sqlite3.Binary(snapshot_bytes),
                            snapshot_sha256,
                            manifest.created_at,
                        ),
                    )
                    connection.executemany(
                        """
                        INSERT INTO derivation_artifacts (logical_key, ordinal, artifact_id)
                        VALUES (?, ?, ?)
                        """,
                        (
                            (logical_key, ordinal, entry.artifact_id)
                            for ordinal, entry in enumerate(snapshot.entries)
                        ),
                    )
            return reused
        except ArtifactRegistryError:
            raise
        except (sqlite3.Error, OSError) as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.TRANSACTION_FAILED,
                f"artifact derivation transaction failed for {logical_key!r}: {error}",
            ) from error
        finally:
            connection.close()

    def _commit(self, connection: sqlite3.Connection) -> None:
        """Commit hook kept narrow so commit failures can be fault-injected."""

        connection.commit()

    def _rollback(self, connection: sqlite3.Connection) -> None:
        """Rollback hook kept narrow so rollback failures can be fault-injected."""

        connection.rollback()

    def _check_existing_derivation(
        self,
        connection: sqlite3.Connection,
        *,
        logical_key: str,
        manifest_artifact_id: str,
        snapshot_bytes: bytes,
        snapshot_sha256: str,
    ) -> bool:
        row = connection.execute(
            """
            SELECT manifest_artifact_id, snapshot_json, snapshot_sha256
            FROM derivations
            WHERE logical_key = ?
            """,
            (logical_key,),
        ).fetchone()
        if row is None:
            return False
        stored_bytes = _row_bytes(row, "snapshot_json")
        stored_sha256 = _row_text(row, "snapshot_sha256")
        if exact_bytes_sha256(stored_bytes) != stored_sha256:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"stored snapshot digest is corrupt for {logical_key!r}",
            )
        if (
            _row_text(row, "manifest_artifact_id") != manifest_artifact_id
            or stored_bytes != snapshot_bytes
            or stored_sha256 != snapshot_sha256
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.DERIVATION_CONFLICT,
                f"logical key already identifies different immutable output: {logical_key!r}",
            )
        return True

    def _insert_or_verify_entry(
        self,
        connection: sqlite3.Connection,
        entry: ArtifactRegistryEntry,
    ) -> bool:
        entry_bytes = canonical_json_bytes(entry)
        semantic_row = connection.execute(
            """
            SELECT artifact_id
            FROM artifacts
            WHERE artifact_type = ? AND semantic_sha256 = ?
            """,
            (entry.artifact_type.value, entry.semantic_sha256),
        ).fetchone()
        if semantic_row is not None and _row_text(semantic_row, "artifact_id") != entry.artifact_id:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.ARTIFACT_CONFLICT,
                "semantic artifact identity is already bound to a different artifact ID",
            )

        row = connection.execute(
            """
            SELECT artifact_type, semantic_sha256, exact_sha256,
                   byte_count, media_type, entry_json
            FROM artifacts
            WHERE artifact_id = ?
            """,
            (entry.artifact_id,),
        ).fetchone()
        location = connection.execute(
            """
            SELECT artifact_id, uri, object_version
            FROM artifact_locations
            WHERE uri = ? AND object_version = ?
            """,
            (entry.locator.uri, entry.locator.object_version),
        ).fetchone()
        if location is not None and _row_text(location, "artifact_id") != entry.artifact_id:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.LOCATION_CONFLICT,
                "artifact locator and object version are already bound to another artifact",
            )

        if row is not None:
            if (
                _row_text(row, "artifact_type") != entry.artifact_type.value
                or _row_text(row, "semantic_sha256") != entry.semantic_sha256
                or _row_text(row, "exact_sha256") != entry.sha256
                or _row_int(row, "byte_count") != entry.bytes
                or _row_text(row, "media_type") != entry.media_type
                or _row_bytes(row, "entry_json") != entry_bytes
            ):
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.ARTIFACT_CONFLICT,
                    f"artifact ID already has different immutable metadata: {entry.artifact_id}",
                )
            exact_location = connection.execute(
                """
                SELECT uri, object_version
                FROM artifact_locations
                WHERE artifact_id = ?
                """,
                (entry.artifact_id,),
            ).fetchone()
            if exact_location is None:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                    f"registered artifact has no locator row: {entry.artifact_id}",
                )
            if (
                _row_text(exact_location, "uri") != entry.locator.uri
                or _row_text(exact_location, "object_version") != entry.locator.object_version
            ):
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.ARTIFACT_CONFLICT,
                    f"artifact ID already has a different locator: {entry.artifact_id}",
                )
            return False

        connection.execute(
            """
            INSERT INTO artifacts (
                artifact_id, artifact_type, semantic_sha256, exact_sha256,
                byte_count, media_type, entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.artifact_id,
                entry.artifact_type.value,
                entry.semantic_sha256,
                entry.sha256,
                entry.bytes,
                entry.media_type,
                sqlite3.Binary(entry_bytes),
            ),
        )
        connection.execute(
            """
            INSERT INTO artifact_locations (artifact_id, uri, object_version)
            VALUES (?, ?, ?)
            """,
            (entry.artifact_id, entry.locator.uri, entry.locator.object_version),
        )
        return True

    def _insert_or_verify_edges(
        self,
        connection: sqlite3.Connection,
        entry: ArtifactRegistryEntry,
        *,
        newly_inserted: bool,
    ) -> None:
        expected = tuple(
            (parent.artifact_id, parent.relation.value, ordinal)
            for ordinal, parent in enumerate(entry.parents)
        )
        if newly_inserted:
            connection.executemany(
                """
                INSERT INTO artifact_edges (
                    child_artifact_id, parent_artifact_id, relation, ordinal
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (entry.artifact_id, parent_id, relation, ordinal)
                    for parent_id, relation, ordinal in expected
                ),
            )
            return
        rows = connection.execute(
            """
            SELECT parent_artifact_id, relation, ordinal
            FROM artifact_edges
            WHERE child_artifact_id = ?
            ORDER BY ordinal
            """,
            (entry.artifact_id,),
        ).fetchall()
        actual = tuple(
            (
                _row_text(row, "parent_artifact_id"),
                _row_text(row, "relation"),
                _row_int(row, "ordinal"),
            )
            for row in rows
        )
        if actual != expected:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"registered typed edges are corrupt for artifact {entry.artifact_id}",
            )

    def _verify_registered_entry(
        self,
        connection: sqlite3.Connection,
        entry: ArtifactRegistryEntry,
    ) -> None:
        row = connection.execute(
            """
            SELECT artifact_type, semantic_sha256, exact_sha256,
                   byte_count, media_type, entry_json
            FROM artifacts
            WHERE artifact_id = ?
            """,
            (entry.artifact_id,),
        ).fetchone()
        if row is None:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"derivation references an absent artifact row: {entry.artifact_id}",
            )
        if (
            _row_text(row, "artifact_type") != entry.artifact_type.value
            or _row_text(row, "semantic_sha256") != entry.semantic_sha256
            or _row_text(row, "exact_sha256") != entry.sha256
            or _row_int(row, "byte_count") != entry.bytes
            or _row_text(row, "media_type") != entry.media_type
            or _row_bytes(row, "entry_json") != canonical_json_bytes(entry)
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"normalized artifact metadata disagrees with snapshot: {entry.artifact_id}",
            )
        location = connection.execute(
            """
            SELECT uri, object_version
            FROM artifact_locations
            WHERE artifact_id = ?
            """,
            (entry.artifact_id,),
        ).fetchone()
        if location is None or (
            _row_text(location, "uri") != entry.locator.uri
            or _row_text(location, "object_version") != entry.locator.object_version
        ):
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"artifact locator disagrees with snapshot: {entry.artifact_id}",
            )
        self._insert_or_verify_edges(connection, entry, newly_inserted=False)

    def _verify_derivation_membership(
        self,
        connection: sqlite3.Connection,
        logical_key: str,
        snapshot: ArtifactRegistrySnapshot,
    ) -> None:
        rows = connection.execute(
            """
            SELECT ordinal, artifact_id
            FROM derivation_artifacts
            WHERE logical_key = ?
            ORDER BY ordinal
            """,
            (logical_key,),
        ).fetchall()
        actual = tuple((_row_int(row, "ordinal"), _row_text(row, "artifact_id")) for row in rows)
        expected = tuple(
            (ordinal, entry.artifact_id) for ordinal, entry in enumerate(snapshot.entries)
        )
        if actual != expected:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"derivation membership disagrees with snapshot: {logical_key!r}",
            )

    def _verify_database_graph(self, connection: sqlite3.Connection) -> None:
        artifact_ids = {
            _row_text(row, "artifact_id")
            for row in connection.execute("SELECT artifact_id FROM artifacts").fetchall()
        }
        parents_by_child: dict[str, list[str]] = {artifact_id: [] for artifact_id in artifact_ids}
        for row in connection.execute(
            "SELECT child_artifact_id, parent_artifact_id FROM artifact_edges"
        ).fetchall():
            child = _row_text(row, "child_artifact_id")
            parent = _row_text(row, "parent_artifact_id")
            if child not in artifact_ids or parent not in artifact_ids:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.MISSING_PARENT,
                    "artifact edge references an absent normalized artifact row",
                )
            parents_by_child[child].append(parent)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(artifact_id: str) -> None:
            if artifact_id in visiting:
                raise ArtifactRegistryError(
                    ArtifactRegistryErrorCode.GRAPH_CYCLE,
                    f"normalized artifact graph contains a cycle at {artifact_id}",
                )
            if artifact_id in visited:
                return
            visiting.add(artifact_id)
            for parent_id in parents_by_child[artifact_id]:
                visit(parent_id)
            visiting.remove(artifact_id)
            visited.add(artifact_id)

        for artifact_id in artifact_ids:
            visit(artifact_id)

    def _snapshot_from_derivation_row(
        self,
        row: sqlite3.Row,
        logical_key: str,
    ) -> ArtifactRegistrySnapshot:
        snapshot_bytes = _row_bytes(row, "snapshot_json")
        stored_digest = _row_text(row, "snapshot_sha256")
        if exact_bytes_sha256(snapshot_bytes) != stored_digest:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"stored snapshot digest is corrupt for {logical_key!r}",
            )
        try:
            snapshot = ArtifactRegistrySnapshot.model_validate_json(snapshot_bytes, strict=True)
        except (ValidationError, ValueError) as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"stored snapshot is invalid for {logical_key!r}: {error}",
            ) from error
        if canonical_json_bytes(snapshot) != snapshot_bytes:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"stored snapshot is not canonical for {logical_key!r}",
            )
        manifest_artifact_id = _row_text(row, "manifest_artifact_id")
        return self._validate_snapshot(snapshot, manifest_artifact_id)

    def _derivation_exists(self, logical_key: str) -> bool:
        self._assert_storage_layout()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT 1 FROM derivations WHERE logical_key = ?",
                (logical_key,),
            ).fetchone()
            return row is not None
        except sqlite3.Error as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.INTEGRITY_ERROR,
                f"cannot look up derivation {logical_key!r}: {error}",
            ) from error
        finally:
            connection.close()


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite column {column!r} is not text",
        )
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INTEGRITY_ERROR,
            f"SQLite column {column!r} is not an integer",
        )
    return value


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    raise ArtifactRegistryError(
        ArtifactRegistryErrorCode.INTEGRITY_ERROR,
        f"SQLite column {column!r} is not a blob",
    )


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
            f"SQLite PRAGMA {name} returned no value",
        )
    value: object = row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.STORAGE_IO_ERROR,
            f"SQLite PRAGMA {name} returned a non-integer value",
        )
    return value


def _fsync_file(path: Path) -> None:
    # Windows FlushFileBuffers requires a handle opened with write access.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "LocalArtifactRegistry",
    "allocate_local_artifact_id",
    "deterministic_local_artifact_id",
]
