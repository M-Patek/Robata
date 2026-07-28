"""Independent append-only storage for P13 active-learning decisions.

The existing review queue deliberately has a strict v2 schema and remains the
authority for task lifecycle, leases, and annotations.  This store persists a
separate immutable pool-selection artifact and optional late annotation lineage
without adding a hidden dependency to primary completion or review routing.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.review.active_learning import (
    ActiveLearningSelectionDecision,
    verify_active_learning_selection_decision,
)
from robata.review.models import ReviewAnnotation

_APPLICATION_ID = 0x5256534C  # "RVSL"
_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 30_000

_SCHEMA_SQL = f"""
BEGIN IMMEDIATE;

CREATE TABLE active_learning_decisions (
    decision_semantic_sha256 TEXT PRIMARY KEY,
    decision_logical_key TEXT NOT NULL UNIQUE,
    pool_semantic_sha256 TEXT NOT NULL,
    policy_semantic_sha256 TEXT NOT NULL,
    budget INTEGER NOT NULL CHECK (budget >= 0),
    decision_json BLOB NOT NULL,
    decision_exact_sha256 TEXT NOT NULL,
    UNIQUE (pool_semantic_sha256, policy_semantic_sha256, budget)
) STRICT;

CREATE TABLE active_learning_annotation_lineage (
    decision_semantic_sha256 TEXT NOT NULL,
    review_task_id TEXT NOT NULL,
    review_task_semantic_sha256 TEXT NOT NULL,
    annotation_semantic_sha256 TEXT NOT NULL,
    annotation_exact_sha256 TEXT NOT NULL,
    annotation_json BLOB NOT NULL,
    PRIMARY KEY (decision_semantic_sha256, annotation_semantic_sha256),
    FOREIGN KEY (decision_semantic_sha256)
        REFERENCES active_learning_decisions (decision_semantic_sha256)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT;

CREATE INDEX active_learning_annotation_lineage_task
ON active_learning_annotation_lineage (
    decision_semantic_sha256,
    review_task_id,
    annotation_semantic_sha256
);

CREATE TRIGGER active_learning_decisions_no_update
BEFORE UPDATE ON active_learning_decisions
BEGIN
    SELECT RAISE(ABORT, 'active-learning decisions are append-only');
END;

CREATE TRIGGER active_learning_decisions_no_delete
BEFORE DELETE ON active_learning_decisions
BEGIN
    SELECT RAISE(ABORT, 'active-learning decisions cannot be deleted');
END;

CREATE TRIGGER active_learning_annotation_lineage_no_update
BEFORE UPDATE ON active_learning_annotation_lineage
BEGIN
    SELECT RAISE(ABORT, 'active-learning annotation lineage is append-only');
END;

CREATE TRIGGER active_learning_annotation_lineage_no_delete
BEFORE DELETE ON active_learning_annotation_lineage
BEGIN
    SELECT RAISE(ABORT, 'active-learning annotation lineage cannot be deleted');
END;

PRAGMA application_id = {_APPLICATION_ID};
PRAGMA user_version = {_SCHEMA_VERSION};
COMMIT;
"""


class ReviewSelectionStoreError(RuntimeError):
    """Base error for independent active-learning selection persistence."""


class ReviewSelectionStoreConflict(ReviewSelectionStoreError):
    """An immutable decision or lineage key has different exact bytes."""


class ReviewSelectionStoreIntegrityError(ReviewSelectionStoreError):
    """Stored JSON or relational lineage does not replay from its inputs."""


class ReviewSelectionStore:
    """Exact-byte durable store for nonblocking active-learning decisions."""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a pathlib.Path")
        try:
            database_path.parent.mkdir(parents=True, exist_ok=True)
            self._database_path = database_path.parent.resolve(strict=True) / database_path.name
        except OSError as error:
            raise ReviewSelectionStoreError(
                f"cannot prepare review selection database path: {error}"
            ) from error
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the absolute database path without opening a connection."""

        return self._database_path

    def put_or_get(
        self,
        decision: ActiveLearningSelectionDecision,
    ) -> tuple[ActiveLearningSelectionDecision, bool]:
        """Append an exact decision once, or return its byte-identical replay."""

        checked = _require_decision(decision)
        expected = canonical_json_bytes(checked)
        expected_exact = exact_bytes_sha256(expected)

        with self._transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT decision_json, decision_exact_sha256
                FROM active_learning_decisions
                WHERE pool_semantic_sha256 = ?
                  AND policy_semantic_sha256 = ?
                  AND budget = ?
                """,
                (
                    checked.pool.semantic_sha256,
                    checked.policy.semantic_sha256,
                    checked.budget,
                ),
            ).fetchone()
            if row is not None:
                stored = _read_exact_bytes(row["decision_json"], "stored selection decision")
                if row["decision_exact_sha256"] != exact_bytes_sha256(stored):
                    raise ReviewSelectionStoreIntegrityError(
                        "stored selection decision exact digest is inconsistent"
                    )
                if stored != expected:
                    raise ReviewSelectionStoreConflict(
                        "pool, policy, and budget already bind different immutable decision bytes"
                    )
                return _parse_decision(stored), True
            try:
                connection.execute(
                    """
                    INSERT INTO active_learning_decisions (
                        decision_semantic_sha256,
                        decision_logical_key,
                        pool_semantic_sha256,
                        policy_semantic_sha256,
                        budget,
                        decision_json,
                        decision_exact_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checked.semantic_sha256,
                        checked.logical_key,
                        checked.pool.semantic_sha256,
                        checked.policy.semantic_sha256,
                        checked.budget,
                        expected,
                        expected_exact,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ReviewSelectionStoreConflict(
                    f"selection decision conflicts with existing immutable row: {error}"
                ) from error
        return checked, False

    def get(self, semantic_sha256: str) -> ActiveLearningSelectionDecision | None:
        """Load and replay-verify one persisted decision."""

        digest = _digest(semantic_sha256, "semantic_sha256")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT decision_json, decision_exact_sha256
                FROM active_learning_decisions
                WHERE decision_semantic_sha256 = ?
                """,
                (digest,),
            ).fetchone()
        if row is None:
            return None
        raw = _read_exact_bytes(row["decision_json"], "stored selection decision")
        if row["decision_exact_sha256"] != exact_bytes_sha256(raw):
            raise ReviewSelectionStoreIntegrityError(
                "stored selection decision exact digest is inconsistent"
            )
        result = _parse_decision(raw)
        if result.semantic_sha256 != digest:
            raise ReviewSelectionStoreIntegrityError(
                "selection decision path does not match its semantic digest"
            )
        return result

    def list_decisions(self) -> tuple[ActiveLearningSelectionDecision, ...]:
        """Return replay-verified immutable decisions in deterministic identity order."""

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT decision_semantic_sha256, decision_json, decision_exact_sha256
                FROM active_learning_decisions
                ORDER BY decision_semantic_sha256
                """
            ).fetchall()
        decisions: list[ActiveLearningSelectionDecision] = []
        for row in rows:
            raw = _read_exact_bytes(row["decision_json"], "stored selection decision")
            if row["decision_exact_sha256"] != exact_bytes_sha256(raw):
                raise ReviewSelectionStoreIntegrityError(
                    "stored selection decision exact digest is inconsistent"
                )
            decision = _parse_decision(raw)
            if decision.semantic_sha256 != row["decision_semantic_sha256"]:
                raise ReviewSelectionStoreIntegrityError(
                    "selection decision row does not match its semantic digest"
                )
            decisions.append(decision)
        return tuple(decisions)

    def append_annotation_lineage(
        self,
        *,
        decision: ActiveLearningSelectionDecision,
        annotation: ReviewAnnotation,
    ) -> bool:
        """Append a late annotation reference without modifying historical selection."""

        checked_decision = _require_decision(decision)
        stored = self.get(checked_decision.semantic_sha256)
        if stored is None:
            raise ReviewSelectionStoreIntegrityError(
                "annotation lineage requires a persisted selection decision"
            )
        if stored != checked_decision:
            raise ReviewSelectionStoreConflict(
                "persisted selection decision differs from supplied immutable decision"
            )
        checked_annotation = _require_annotation(annotation)
        selected_by_task_id = {
            item.candidate.review_task_id: item.candidate.review_task_semantic_sha256
            for item in checked_decision.candidate_decisions
            if item.candidate.review_task_id in checked_decision.selected_review_task_ids
        }
        task_digest = selected_by_task_id.get(checked_annotation.review_task_id)
        if task_digest is None:
            raise ReviewSelectionStoreIntegrityError(
                "annotation lineage may only cite a task selected by the frozen decision"
            )
        if task_digest != checked_annotation.review_task_semantic_sha256:
            raise ReviewSelectionStoreIntegrityError(
                "annotation review task digest differs from the selected pool candidate"
            )
        payload = canonical_json_bytes(checked_annotation)
        payload_exact = exact_bytes_sha256(payload)

        with self._transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT annotation_json, annotation_exact_sha256
                FROM active_learning_annotation_lineage
                WHERE decision_semantic_sha256 = ?
                  AND annotation_semantic_sha256 = ?
                """,
                (checked_decision.semantic_sha256, checked_annotation.semantic_sha256),
            ).fetchone()
            if row is not None:
                stored_payload = _read_exact_bytes(
                    row["annotation_json"], "stored annotation lineage"
                )
                if row["annotation_exact_sha256"] != exact_bytes_sha256(stored_payload):
                    raise ReviewSelectionStoreIntegrityError(
                        "stored annotation lineage exact digest is inconsistent"
                    )
                if stored_payload != payload:
                    raise ReviewSelectionStoreConflict(
                        "annotation lineage identity has different immutable bytes"
                    )
                return False
            try:
                connection.execute(
                    """
                    INSERT INTO active_learning_annotation_lineage (
                        decision_semantic_sha256,
                        review_task_id,
                        review_task_semantic_sha256,
                        annotation_semantic_sha256,
                        annotation_exact_sha256,
                        annotation_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checked_decision.semantic_sha256,
                        checked_annotation.review_task_id,
                        checked_annotation.review_task_semantic_sha256,
                        checked_annotation.semantic_sha256,
                        payload_exact,
                        payload,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ReviewSelectionStoreConflict(
                    f"annotation lineage conflicts with an immutable row: {error}"
                ) from error
        return True

    def list_annotations(
        self,
        decision_semantic_sha256: str,
    ) -> tuple[ReviewAnnotation, ...]:
        """Return immutable annotation lineage in deterministic append-independent order."""

        digest = _digest(decision_semantic_sha256, "decision_semantic_sha256")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT annotation_json, annotation_exact_sha256
                FROM active_learning_annotation_lineage
                WHERE decision_semantic_sha256 = ?
                ORDER BY review_task_id, annotation_semantic_sha256
                """,
                (digest,),
            ).fetchall()
        result: list[ReviewAnnotation] = []
        for row in rows:
            raw = _read_exact_bytes(row["annotation_json"], "stored annotation lineage")
            if row["annotation_exact_sha256"] != exact_bytes_sha256(raw):
                raise ReviewSelectionStoreIntegrityError(
                    "stored annotation lineage exact digest is inconsistent"
                )
            result.append(_parse_annotation(raw))
        return tuple(result)

    def _initialize_database(self) -> None:
        try:
            with self._connect() as connection:
                application_id = connection.execute("PRAGMA application_id").fetchone()[0]
                schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                required = {
                    "active_learning_decisions",
                    "active_learning_annotation_lineage",
                }
                if not tables:
                    connection.executescript(_SCHEMA_SQL)
                    return
                if application_id != _APPLICATION_ID or schema_version != _SCHEMA_VERSION:
                    raise ReviewSelectionStoreIntegrityError(
                        "review selection database has an incompatible application or "
                        "schema version"
                    )
                if tables != required:
                    raise ReviewSelectionStoreIntegrityError(
                        "review selection database table set is incompatible"
                    )
        except sqlite3.Error as error:
            if isinstance(error, ReviewSelectionStoreIntegrityError):
                raise
            raise ReviewSelectionStoreError(
                f"cannot initialize review selection database: {error}"
            ) from error

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                try:
                    yield connection
                except BaseException:
                    connection.rollback()
                    raise
                else:
                    connection.commit()
        except sqlite3.Error as error:
            raise ReviewSelectionStoreError(
                f"review selection database operation failed: {error}"
            ) from error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _require_decision(value: object) -> ActiveLearningSelectionDecision:
    if not isinstance(value, ActiveLearningSelectionDecision):
        raise TypeError("decision must be an ActiveLearningSelectionDecision")
    try:
        return verify_active_learning_selection_decision(
            ActiveLearningSelectionDecision.model_validate_json(canonical_json_bytes(value))
        )
    except ValueError as error:
        raise ReviewSelectionStoreIntegrityError(f"invalid selection decision: {error}") from error


def _require_annotation(value: object) -> ReviewAnnotation:
    if not isinstance(value, ReviewAnnotation):
        raise TypeError("annotation must be a ReviewAnnotation")
    try:
        return ReviewAnnotation.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ReviewSelectionStoreIntegrityError(f"invalid review annotation: {error}") from error


def _parse_decision(raw: bytes) -> ActiveLearningSelectionDecision:
    _parse_exact_document(raw, "selection decision")
    try:
        decision = ActiveLearningSelectionDecision.model_validate_json(raw)
        return verify_active_learning_selection_decision(decision)
    except ValueError as error:
        raise ReviewSelectionStoreIntegrityError(
            f"stored selection decision cannot replay: {error}"
        ) from error


def _parse_annotation(raw: bytes) -> ReviewAnnotation:
    _parse_exact_document(raw, "annotation lineage")
    try:
        return ReviewAnnotation.model_validate_json(raw)
    except ValueError as error:
        raise ReviewSelectionStoreIntegrityError(
            f"stored annotation lineage is invalid: {error}"
        ) from error


def _parse_exact_document(raw: bytes, subject: str) -> dict[str, Any]:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ReviewSelectionStoreIntegrityError(f"invalid {subject} JSON: {error}") from error
    if not isinstance(document, dict) or canonical_json_bytes(document) != raw:
        raise ReviewSelectionStoreIntegrityError(f"stored {subject} is not exact canonical JSON")
    return document


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _read_exact_bytes(value: object, subject: str) -> bytes:
    if not isinstance(value, bytes):
        raise ReviewSelectionStoreIntegrityError(f"stored {subject} bytes are invalid")
    return value


def _digest(value: object, subject: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{subject} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "ReviewSelectionStore",
    "ReviewSelectionStoreConflict",
    "ReviewSelectionStoreError",
    "ReviewSelectionStoreIntegrityError",
]
