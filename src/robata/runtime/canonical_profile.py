"""Machine-readable local profiling facts for the canonical MCAP command.

The manifest binds inputs known before execution.  The report carries observations made during
and after execution.  Neither model is promotion evidence, and profiling data never participates
in canonical run identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self, cast

from pydantic import Field, StringConstraints, model_validator

from robata.application.canonical.local_composition import CanonicalLocalRunReceipt
from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.queue.models import TERMINAL_WORK_STATES, WorkItemState
from robata.runtime.observability import RuntimeProfileSnapshot, RuntimeSpanSnapshot

CANONICAL_PROFILE_MANIFEST_VERSION: Final = "canonical-profile-manifest-v1"
CANONICAL_PROFILE_REPORT_VERSION: Final = "canonical-profile-report-v2"
CANONICAL_PROFILE_DURATION_DENOMINATOR_POLICY: Final = "canonical-requested-camera-interval-v1"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
RunKey = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
GitCommit = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
SignedInt = Annotated[int, Field(strict=True)]


class CanonicalProfileError(RuntimeError):
    """A local profile manifest or report could not be produced faithfully."""


class _SQLiteSnapshotUnstableError(RuntimeError):
    """An immutable read would omit committed rows that remain in a WAL."""


class ProfileFileFact(StrictModel):
    """Exact bytes of one manifest input, without a machine-specific path."""

    sha256: Sha256Digest
    byte_count: NonNegativeInt


class ProfileGitFacts(StrictModel):
    """Candidate source-tree identity observed before execution."""

    head_commit: GitCommit
    dirty: bool


class ProfileRuntimeFacts(StrictModel):
    """Portable host/runtime facts available without an optional dependency."""

    python_version: NonEmptyString
    python_implementation: NonEmptyString
    platform: NonEmptyString
    machine: NonEmptyString
    logical_cpu_count: PositiveInt | None


class CanonicalProfilePolicyFacts(StrictModel):
    """Exact policy closure that drives the local canonical composition."""

    composition_version: NonEmptyString
    pipeline_version: NonEmptyString
    execution_policy_semantic_sha256: Sha256Digest
    runtime_policy_semantic_sha256: Sha256Digest
    input_planner_version: NonEmptyString
    parser_version: NonEmptyString
    inference_policy_versions: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_inference_policies(self) -> Self:
        if not self.inference_policy_versions:
            raise ValueError("inference_policy_versions must be nonempty")
        if len(set(self.inference_policy_versions)) != len(self.inference_policy_versions):
            raise ValueError("inference_policy_versions must be unique")
        return self


def _manifest_projection(manifest: CanonicalProfileManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"manifest_sha256"})


class CanonicalProfileManifest(StrictModel):
    """Content-addressed local workload description fixed before timing starts."""

    schema_version: Literal["1.0"]
    model_version: Literal["canonical-profile-manifest-v1"]
    manifest_sha256: Sha256Digest
    source: ProfileFileFact
    mapping_config: ProfileFileFact
    uv_lock: ProfileFileFact
    schema_catalog: ProfileFileFact
    git: ProfileGitFacts
    runtime: ProfileRuntimeFacts
    policies: CanonicalProfilePolicyFacts
    camera_count: Literal[6]
    run_key: RunKey
    max_duration_ns: Nanoseconds | None
    allow_unapproved_profile: bool
    duration_denominator_policy: Literal["canonical-requested-camera-interval-v1"]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    production_eligible: Literal[False]
    measurement_status: Literal["NOT_MEASURED"]
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"]

    @classmethod
    def create(
        cls,
        *,
        source: ProfileFileFact,
        mapping_config: ProfileFileFact,
        uv_lock: ProfileFileFact,
        schema_catalog: ProfileFileFact,
        git: ProfileGitFacts,
        runtime: ProfileRuntimeFacts,
        policies: CanonicalProfilePolicyFacts,
        run_key: str,
        max_duration_ns: int | None,
        allow_unapproved_profile: bool,
    ) -> CanonicalProfileManifest:
        if not isinstance(run_key, str) or not run_key.strip() or len(run_key) > 128:
            raise ValueError("run_key must be a nonblank string of at most 128 characters")
        if max_duration_ns is not None and (
            isinstance(max_duration_ns, bool)
            or not isinstance(max_duration_ns, int)
            or max_duration_ns <= 0
        ):
            raise ValueError("max_duration_ns must be a positive integer or None")
        if not isinstance(allow_unapproved_profile, bool):
            raise TypeError("allow_unapproved_profile must be a boolean")
        draft = cls.model_construct(
            schema_version="1.0",
            model_version=CANONICAL_PROFILE_MANIFEST_VERSION,
            manifest_sha256="0" * 64,
            source=source,
            mapping_config=mapping_config,
            uv_lock=uv_lock,
            schema_catalog=schema_catalog,
            git=git,
            runtime=runtime,
            policies=policies,
            camera_count=6,
            run_key=run_key,
            max_duration_ns=max_duration_ns,
            allow_unapproved_profile=allow_unapproved_profile,
            duration_denominator_policy=CANONICAL_PROFILE_DURATION_DENOMINATOR_POLICY,
            evidence_class="LOCAL_CONFORMANCE",
            production_eligible=False,
            measurement_status="NOT_MEASURED",
            qualification_status="NOT_PRODUCTION_QUALIFIED",
        )
        digest = exact_bytes_sha256(canonical_json_bytes(_manifest_projection(draft)))
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "manifest_sha256": digest},
            strict=True,
        )

    @model_validator(mode="after")
    def validate_manifest_digest(self) -> Self:
        if not self.run_key.strip():
            raise ValueError("run_key must be nonblank")
        expected = exact_bytes_sha256(canonical_json_bytes(_manifest_projection(self)))
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match canonical manifest bytes")
        return self


class StateFileClass(StrEnum):
    """Stable local state-byte accounting classes."""

    FRAME_PNG = "FRAME_PNG"
    EXPORTED_VIDEO = "EXPORTED_VIDEO"
    SQLITE = "SQLITE"
    JSON = "JSON"
    CONTENT_ADDRESSED_BLOB = "CONTENT_ADDRESSED_BLOB"
    OTHER = "OTHER"


_STATE_FILE_CLASS_ORDER: Final = tuple(StateFileClass)


class StateFileClassSnapshot(StrictModel):
    file_class: StateFileClass
    file_count: NonNegativeInt
    byte_count: NonNegativeInt


class SQLiteTableRowCount(StrictModel):
    table_name: NonEmptyString
    row_count: NonNegativeInt


class SQLiteReadError(StrictModel):
    error_type: NonEmptyString
    detail: NonEmptyString


class SQLiteDatabaseSnapshot(StrictModel):
    relative_path: NonEmptyString
    tables: tuple[SQLiteTableRowCount, ...]
    error: SQLiteReadError | None = None

    @model_validator(mode="after")
    def validate_database_snapshot(self) -> Self:
        names = tuple(item.table_name for item in self.tables)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("SQLite application tables must be unique and ordered")
        if self.error is not None and self.tables:
            raise ValueError("a failed SQLite snapshot cannot contain partial row counts")
        return self


class StateTreeSnapshot(StrictModel):
    file_count: NonNegativeInt
    byte_count: NonNegativeInt
    classes: tuple[StateFileClassSnapshot, ...]
    sqlite_databases: tuple[SQLiteDatabaseSnapshot, ...]

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        if tuple(item.file_class for item in self.classes) != _STATE_FILE_CLASS_ORDER:
            raise ValueError("state file classes must appear in canonical order")
        if sum(item.file_count for item in self.classes) != self.file_count:
            raise ValueError("state class file counts do not reconcile")
        if sum(item.byte_count for item in self.classes) != self.byte_count:
            raise ValueError("state class byte counts do not reconcile")
        paths = tuple(item.relative_path for item in self.sqlite_databases)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("SQLite database snapshots must be unique and ordered")
        return self


class WorkQueueObservationStatus(StrEnum):
    """Availability of the read-only durable-work backlog observation."""

    AVAILABLE = "AVAILABLE"
    ABSENT = "ABSENT"
    ERROR = "ERROR"


class WorkQueueStateCount(StrictModel):
    state: WorkItemState
    count: NonNegativeInt


class WorkQueueBacklogObservation(StrictModel):
    """Durable work state after the run, without treating backlog as run success."""

    status: WorkQueueObservationStatus
    database_relative_path: Literal["work-scheduler.sqlite3"]
    state_counts: tuple[WorkQueueStateCount, ...]
    nonterminal_backlog_count: NonNegativeInt
    oldest_nonterminal_age_ns: NonNegativeInt | None
    error: SQLiteReadError | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.status is WorkQueueObservationStatus.AVAILABLE:
            if tuple(item.state for item in self.state_counts) != tuple(WorkItemState):
                raise ValueError("available queue state counts must use canonical state order")
            expected_backlog = sum(
                item.count for item in self.state_counts if item.state not in TERMINAL_WORK_STATES
            )
            if self.nonterminal_backlog_count != expected_backlog:
                raise ValueError("nonterminal queue backlog does not reconcile")
            if (self.oldest_nonterminal_age_ns is None) != (expected_backlog == 0):
                raise ValueError("queue age must be present exactly when backlog is nonzero")
            if self.error is not None:
                raise ValueError("an available queue observation cannot contain an error")
        else:
            if self.state_counts or self.nonterminal_backlog_count != 0:
                raise ValueError("unavailable queue observations cannot contain counts")
            if self.oldest_nonterminal_age_ns is not None:
                raise ValueError("unavailable queue observations cannot contain queue age")
            if (self.error is None) != (self.status is WorkQueueObservationStatus.ABSENT):
                raise ValueError("only an ERROR queue observation requires an error")
        return self


ReconciliationStatus = Literal[
    "RECONCILED",
    "PARTIAL",
    "MISMATCH",
    "NOT_AVAILABLE",
    "STRUCTURAL_ERROR",
]


class SpanReconciliation(StrictModel):
    """Structural and wall-time reconciliation for the frozen span tree."""

    status: ReconciliationStatus
    observer_elapsed_ns: NonNegativeInt
    span_count: NonNegativeInt
    root_span_count: NonNegativeInt
    top_level_sum_ns: NonNegativeInt
    top_level_union_ns: NonNegativeInt
    exclusive_sum_ns: NonNegativeInt
    uncovered_wall_ns: NonNegativeInt
    structural_errors: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_span_reconciliation(self) -> Self:
        if tuple(sorted(self.structural_errors)) != self.structural_errors:
            raise ValueError("span structural errors must be ordered")
        if self.top_level_union_ns > self.observer_elapsed_ns:
            raise ValueError("top-level span union cannot exceed observer wall time")
        expected_uncovered = self.observer_elapsed_ns - self.top_level_union_ns
        if self.uncovered_wall_ns != expected_uncovered:
            raise ValueError("uncovered wall time does not reconcile")
        if self.status == "STRUCTURAL_ERROR" and not self.structural_errors:
            raise ValueError("structural span errors require STRUCTURAL_ERROR status")
        if self.status != "STRUCTURAL_ERROR" and self.structural_errors:
            raise ValueError("structural span errors require STRUCTURAL_ERROR status")
        return self


class LedgerReconciliationFact(StrictModel):
    """One counter-to-authority row-count comparison."""

    name: NonEmptyString
    expected: NonNegativeInt | None
    observed: NonNegativeInt | None
    status: ReconciliationStatus

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        if self.expected is None or self.observed is None:
            if self.status not in {"NOT_AVAILABLE", "PARTIAL"}:
                raise ValueError("missing ledger values require an unavailable status")
        elif self.expected != self.observed and self.status != "MISMATCH":
            raise ValueError("mismatched ledger values require MISMATCH status")
        elif self.expected == self.observed and self.status not in {"RECONCILED", "PARTIAL"}:
            raise ValueError("equal ledger values require a reconciled status")
        return self


class LedgerReconciliation(StrictModel):
    """Authoritative row-count checks for counters emitted by the canonical path."""

    status: ReconciliationStatus
    facts: tuple[LedgerReconciliationFact, ...]

    @model_validator(mode="after")
    def validate_facts(self) -> Self:
        names = tuple(item.name for item in self.facts)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("ledger reconciliation facts must be unique and ordered")
        statuses = {item.status for item in self.facts}
        if "MISMATCH" in statuses and self.status != "MISMATCH":
            raise ValueError("ledger mismatch requires MISMATCH status")
        if "NOT_AVAILABLE" in statuses and self.status == "RECONCILED":
            raise ValueError("unavailable ledger facts cannot be fully reconciled")
        return self


class ArtifactByteReconciliation(StrictModel):
    """State-tree byte classes and the explicit limits of local accounting."""

    status: Literal["RECONCILED", "MISMATCH"]
    raw_input_bytes: NonNegativeInt
    state_bytes: NonNegativeInt
    frame_png_bytes: NonNegativeInt
    exported_video_bytes: NonNegativeInt
    sqlite_bytes: NonNegativeInt
    json_bytes: NonNegativeInt
    content_addressed_blob_bytes: NonNegativeInt
    other_bytes: NonNegativeInt
    physical_duplication_status: Literal["NOT_AVAILABLE"] = "NOT_AVAILABLE"

    @model_validator(mode="after")
    def validate_state_sum(self) -> Self:
        classes_sum = sum(
            (
                self.frame_png_bytes,
                self.exported_video_bytes,
                self.sqlite_bytes,
                self.json_bytes,
                self.content_addressed_blob_bytes,
                self.other_bytes,
            )
        )
        if classes_sum != self.state_bytes:
            raise ValueError("artifact byte classes do not reconcile to state_bytes")
        return self


class CanonicalProfileReconciliation(StrictModel):
    """Combined machine-checkable baseline reconciliation facts."""

    spans: SpanReconciliation
    ledger: LedgerReconciliation
    artifact_bytes: ArtifactByteReconciliation


class CanonicalProfileRunError(StrictModel):
    """Structured canonical-command failure retained in a profile report."""

    code: NonEmptyString
    error_type: NonEmptyString
    detail: NonEmptyString


class CanonicalProfileReport(StrictModel):
    """One local observation bound to an immutable pre-execution manifest."""

    schema_version: Literal["1.0"]
    model_version: Literal["canonical-profile-report-v1", "canonical-profile-report-v2"]
    manifest: CanonicalProfileManifest
    manifest_sha256: Sha256Digest
    observer: RuntimeProfileSnapshot
    state_before: StateTreeSnapshot
    state_after: StateTreeSnapshot
    state_file_count_delta: SignedInt
    state_byte_count_delta: SignedInt
    work_queue_after: WorkQueueBacklogObservation
    receipt: CanonicalLocalRunReceipt | None
    error: CanonicalProfileRunError | None
    execution_mode: Literal["FRESH", "REPLAY", "UNKNOWN"]
    source_span_duration_ns: Nanoseconds | None
    recording_duration_ns: Nanoseconds | None
    requested_duration_ns: Nanoseconds | None
    reconciliation: CanonicalProfileReconciliation | None = None
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    production_eligible: Literal[False] = False
    measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"] = "NOT_PRODUCTION_QUALIFIED"

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("report manifest_sha256 does not match its manifest")
        if (self.receipt is None) == (self.error is None):
            raise ValueError("report must contain exactly one of receipt or error")
        if (
            self.state_file_count_delta
            != self.state_after.file_count - self.state_before.file_count
        ):
            raise ValueError("state_file_count_delta does not reconcile")
        if (
            self.state_byte_count_delta
            != self.state_after.byte_count - self.state_before.byte_count
        ):
            raise ValueError("state_byte_count_delta does not reconcile")
        if self.receipt is not None:
            expected_mode = "REPLAY" if self.receipt.replayed else "FRESH"
            if self.execution_mode != expected_mode:
                raise ValueError("execution_mode does not match the canonical receipt")
        elif self.execution_mode != "UNKNOWN":
            raise ValueError("failed runs must use UNKNOWN execution_mode")
        if self.requested_duration_ns is not None and self.recording_duration_ns is None:
            raise ValueError("requested_duration_ns requires recording_duration_ns")
        if (
            self.requested_duration_ns is not None
            and self.recording_duration_ns is not None
            and self.requested_duration_ns > self.recording_duration_ns
        ):
            raise ValueError("requested_duration_ns cannot exceed recording_duration_ns")
        if (
            self.model_version == "canonical-profile-report-v2"
            and self.receipt is not None
            and self.reconciliation is None
        ):
            raise ValueError("successful profile reports require reconciliation facts")
        return self


def exact_file_fact(path: Path, *, label: str) -> ProfileFileFact:
    """Hash one required regular file without retaining its local path."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be a nonempty string")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise CanonicalProfileError(f"cannot read {label}: {error}") from error
    return ProfileFileFact(sha256=digest.hexdigest(), byte_count=byte_count)


def build_canonical_profile_manifest(
    *,
    repository_root: Path,
    source_path: Path,
    mapping_config: Path,
    run_key: str,
    max_duration_ns: int | None,
    allow_unapproved_profile: bool,
) -> CanonicalProfileManifest:
    """Build the complete pre-execution profile manifest."""

    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be pathlib.Path")
    root = repository_root.resolve()
    return CanonicalProfileManifest.create(
        source=exact_file_fact(source_path, label="MCAP source"),
        mapping_config=exact_file_fact(mapping_config, label="mapping config"),
        uv_lock=exact_file_fact(root / "uv.lock", label="uv.lock"),
        schema_catalog=exact_file_fact(
            root / "schemas" / "schema-catalog.json",
            label="schema catalog",
        ),
        git=_git_facts(root),
        runtime=_runtime_facts(),
        policies=_policy_facts(),
        run_key=run_key,
        max_duration_ns=max_duration_ns,
        allow_unapproved_profile=allow_unapproved_profile,
    )


def snapshot_state_tree(
    state_root: Path,
    *,
    excluded_paths: Iterable[Path] = (),
) -> StateTreeSnapshot:
    """Snapshot local state bytes and application rows without writing SQLite state."""

    if not isinstance(state_root, Path):
        raise TypeError("state_root must be pathlib.Path")
    root = state_root.resolve()
    if root.exists() and not root.is_dir():
        raise CanonicalProfileError("state_root must be a directory when it exists")
    excluded = frozenset(path.resolve() for path in excluded_paths)
    counts = {file_class: [0, 0] for file_class in _STATE_FILE_CLASS_ORDER}
    files: list[tuple[str, Path, StateFileClass, int]] = []
    if root.exists():
        for path in root.rglob("*"):
            resolved = path.resolve()
            if resolved in excluded or (not path.is_symlink() and not path.is_file()):
                continue
            try:
                byte_count = path.lstat().st_size if path.is_symlink() else path.stat().st_size
            except OSError as error:
                relative = path.relative_to(root).as_posix()
                raise CanonicalProfileError(
                    f"cannot inspect state file '{relative}': {error}"
                ) from error
            relative = path.relative_to(root).as_posix()
            file_class = _classify_state_file(path)
            files.append((relative, path, file_class, byte_count))
            counts[file_class][0] += 1
            counts[file_class][1] += byte_count

    files.sort(key=lambda item: item[0])
    database_snapshots = tuple(
        _snapshot_sqlite_database(root, path, relative)
        for relative, path, file_class, _byte_count in files
        if file_class is StateFileClass.SQLITE
        and path.name.lower().endswith((".sqlite3", ".sqlite", ".db"))
        and not path.is_symlink()
    )
    classes = tuple(
        StateFileClassSnapshot(
            file_class=file_class,
            file_count=counts[file_class][0],
            byte_count=counts[file_class][1],
        )
        for file_class in _STATE_FILE_CLASS_ORDER
    )
    return StateTreeSnapshot(
        file_count=sum(item.file_count for item in classes),
        byte_count=sum(item.byte_count for item in classes),
        classes=classes,
        sqlite_databases=database_snapshots,
    )


def snapshot_work_queue(
    state_root: Path,
    *,
    observed_at: datetime | None = None,
) -> WorkQueueBacklogObservation:
    """Observe the durable-work backlog through a read-only SQLite connection."""

    if not isinstance(state_root, Path):
        raise TypeError("state_root must be pathlib.Path")
    now = observed_at or datetime.now(UTC)
    if not isinstance(now, datetime):
        raise TypeError("observed_at must be datetime or None")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
    now = now.astimezone(UTC)
    database_path = state_root.resolve() / "work-scheduler.sqlite3"
    if not database_path.exists():
        return WorkQueueBacklogObservation(
            status=WorkQueueObservationStatus.ABSENT,
            database_relative_path="work-scheduler.sqlite3",
            state_counts=(),
            nonterminal_backlog_count=0,
            oldest_nonterminal_age_ns=None,
        )
    try:
        if database_path.is_symlink() or not database_path.is_file():
            raise OSError("work scheduler database is not a regular file")
        uri = _immutable_sqlite_uri(database_path)
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT state, COUNT(*) FROM work_items GROUP BY state ORDER BY state"
            ).fetchall()
            observed_counts = {
                WorkItemState(str(row[0])): _sqlite_nonnegative_count(row[1]) for row in rows
            }
            state_counts = tuple(
                WorkQueueStateCount(state=state, count=observed_counts.get(state, 0))
                for state in WorkItemState
            )
            nonterminal_states = tuple(
                state for state in WorkItemState if state not in TERMINAL_WORK_STATES
            )
            placeholders = ", ".join("?" for _state in nonterminal_states)
            oldest_row = connection.execute(
                f"SELECT MIN(created_at) FROM work_items WHERE state IN ({placeholders})",
                tuple(state.value for state in nonterminal_states),
            ).fetchone()
        backlog_count = sum(
            item.count for item in state_counts if item.state not in TERMINAL_WORK_STATES
        )
        oldest_age_ns = _oldest_queue_age_ns(oldest_row, now, backlog_count)
        return WorkQueueBacklogObservation(
            status=WorkQueueObservationStatus.AVAILABLE,
            database_relative_path="work-scheduler.sqlite3",
            state_counts=state_counts,
            nonterminal_backlog_count=backlog_count,
            oldest_nonterminal_age_ns=oldest_age_ns,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError, _SQLiteSnapshotUnstableError) as error:
        return WorkQueueBacklogObservation(
            status=WorkQueueObservationStatus.ERROR,
            database_relative_path="work-scheduler.sqlite3",
            state_counts=(),
            nonterminal_backlog_count=0,
            oldest_nonterminal_age_ns=None,
            error=SQLiteReadError(
                error_type=type(error).__name__,
                detail=str(error) or "queue observation failed without a diagnostic",
            ),
        )


def unique_runtime_counter_value(snapshot: RuntimeProfileSnapshot, name: str) -> int | None:
    """Return one counter value by name, or None when absent or ambiguous."""

    if not isinstance(snapshot, RuntimeProfileSnapshot):
        raise TypeError("snapshot must be RuntimeProfileSnapshot")
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a nonempty string")
    matching = tuple(counter for counter in snapshot.counters if counter.name == name)
    if len(matching) != 1:
        return None
    return matching[0].value


def build_profile_reconciliation(
    *,
    observer: RuntimeProfileSnapshot,
    state_after: StateTreeSnapshot,
    manifest: CanonicalProfileManifest,
    execution_mode: Literal["FRESH", "REPLAY", "UNKNOWN"],
) -> CanonicalProfileReconciliation:
    """Derive explicit span, ledger, and state-byte reconciliation facts."""

    spans = reconcile_runtime_spans(observer)
    artifact_bytes = _reconcile_artifact_bytes(state_after, manifest.source.byte_count)
    ledger = _reconcile_ledger_counts(observer, state_after, execution_mode=execution_mode)
    return CanonicalProfileReconciliation(
        spans=spans,
        ledger=ledger,
        artifact_bytes=artifact_bytes,
    )


def reconcile_runtime_spans(snapshot: RuntimeProfileSnapshot) -> SpanReconciliation:
    """Validate nesting and reconcile top-level coverage with recorder wall time."""

    if not isinstance(snapshot, RuntimeProfileSnapshot):
        raise TypeError("snapshot must be RuntimeProfileSnapshot")
    by_sequence = {span.sequence: span for span in snapshot.spans}
    children: dict[int, list[RuntimeSpanSnapshot]] = {sequence: [] for sequence in by_sequence}
    errors: set[str] = set()
    roots: list[RuntimeSpanSnapshot] = []
    for span in snapshot.spans:
        if span.ended_offset_ns > snapshot.elapsed_ns:
            errors.add(f"span {span.sequence} ends after observer elapsed time")
        if span.parent_sequence is None:
            roots.append(span)
            continue
        parent = by_sequence.get(span.parent_sequence)
        if parent is None:
            errors.add(f"span {span.sequence} references missing parent {span.parent_sequence}")
            continue
        children[parent.sequence].append(span)
        if (
            span.started_offset_ns < parent.started_offset_ns
            or span.ended_offset_ns > parent.ended_offset_ns
        ):
            errors.add(f"span {span.sequence} is outside parent {parent.sequence}")

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(sequence: int) -> None:
        if sequence in visiting:
            errors.add(f"span parent cycle at {sequence}")
            return
        if sequence in visited:
            return
        visiting.add(sequence)
        for child in children[sequence]:
            visit(child.sequence)
        visiting.remove(sequence)
        visited.add(sequence)

    for span in snapshot.spans:
        visit(span.sequence)

    top_level_intervals = [(span.started_offset_ns, span.ended_offset_ns) for span in roots]
    top_level_union = _interval_union_ns(top_level_intervals)
    top_level_sum = sum(span.elapsed_ns for span in roots)
    exclusive_sum = 0
    for span in snapshot.spans:
        child_union = _interval_union_ns(
            (child.started_offset_ns, child.ended_offset_ns) for child in children[span.sequence]
        )
        exclusive_sum += max(0, span.elapsed_ns - child_union)
    status: ReconciliationStatus = "STRUCTURAL_ERROR" if errors else "RECONCILED"
    return SpanReconciliation(
        status=status,
        observer_elapsed_ns=snapshot.elapsed_ns,
        span_count=len(snapshot.spans),
        root_span_count=len(roots),
        top_level_sum_ns=top_level_sum,
        top_level_union_ns=top_level_union,
        exclusive_sum_ns=exclusive_sum,
        uncovered_wall_ns=snapshot.elapsed_ns - top_level_union,
        structural_errors=tuple(sorted(errors)),
    )


def _interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _reconcile_ledger_counts(
    observer: RuntimeProfileSnapshot,
    state_after: StateTreeSnapshot,
    *,
    execution_mode: Literal["FRESH", "REPLAY", "UNKNOWN"],
) -> LedgerReconciliation:
    if execution_mode != "FRESH":
        return LedgerReconciliation(
            status="NOT_AVAILABLE",
            facts=tuple(
                LedgerReconciliationFact(
                    name=name,
                    expected=None,
                    observed=None,
                    status="NOT_AVAILABLE",
                )
                for name in (
                    "inference.calls_vs_terminals",
                    "inference.outcomes_vs_terminals",
                    "outbox.rows_vs_primary",
                    "review.routes_vs_tasks",
                )
            ),
        )

    facts_data = (
        (
            "inference.calls_vs_terminals",
            _counter_total(observer, "inference.fixture_calls"),
            _database_row_count(
                state_after,
                "inference-evidence.sqlite3",
                "model_inference_terminals",
            ),
        ),
        (
            "inference.outcomes_vs_terminals",
            _counter_total(observer, "inference.call_outcomes"),
            _database_row_count(
                state_after,
                "inference-evidence.sqlite3",
                "model_inference_terminals",
            ),
        ),
        (
            "outbox.rows_vs_primary",
            _counter_total(observer, "delivery.outbox.committed_rows_observed"),
            _database_row_count(state_after, "primary-completion.sqlite3", "primary_outbox"),
        ),
        (
            "review.routes_vs_tasks",
            _counter_total(observer, "review.routing_outcomes"),
            _database_row_count(state_after, "review-queue.sqlite3", "review_tasks"),
        ),
    )
    facts: list[LedgerReconciliationFact] = []
    for name, expected, observed in facts_data:
        if expected is None or observed is None:
            status: ReconciliationStatus = "NOT_AVAILABLE"
        elif expected == observed:
            status = "RECONCILED"
        else:
            status = "MISMATCH"
        facts.append(
            LedgerReconciliationFact(
                name=name,
                expected=expected,
                observed=observed,
                status=status,
            )
        )
    statuses = {fact.status for fact in facts}
    overall: ReconciliationStatus
    if "MISMATCH" in statuses:
        overall = "MISMATCH"
    elif "NOT_AVAILABLE" in statuses:
        overall = "PARTIAL"
    else:
        overall = "RECONCILED"
    return LedgerReconciliation(status=overall, facts=tuple(facts))


def _counter_total(snapshot: RuntimeProfileSnapshot, name: str) -> int | None:
    matching = tuple(counter for counter in snapshot.counters if counter.name == name)
    return sum(counter.value for counter in matching) if matching else None


def _database_row_count(
    snapshot: StateTreeSnapshot,
    database_name: str,
    table_name: str,
) -> int | None:
    for database in snapshot.sqlite_databases:
        if database.relative_path.endswith(database_name):
            if database.error is not None:
                return None
            for table in database.tables:
                if table.table_name == table_name:
                    return table.row_count
    return None


def _reconcile_artifact_bytes(
    snapshot: StateTreeSnapshot,
    raw_input_bytes: int,
) -> ArtifactByteReconciliation:
    by_class = {item.file_class: item.byte_count for item in snapshot.classes}
    class_values = {
        "frame_png_bytes": by_class[StateFileClass.FRAME_PNG],
        "exported_video_bytes": by_class[StateFileClass.EXPORTED_VIDEO],
        "sqlite_bytes": by_class[StateFileClass.SQLITE],
        "json_bytes": by_class[StateFileClass.JSON],
        "content_addressed_blob_bytes": by_class[StateFileClass.CONTENT_ADDRESSED_BLOB],
        "other_bytes": by_class[StateFileClass.OTHER],
    }
    classes_sum = sum(class_values.values())
    return ArtifactByteReconciliation(
        status="RECONCILED" if classes_sum == snapshot.byte_count else "MISMATCH",
        raw_input_bytes=raw_input_bytes,
        state_bytes=snapshot.byte_count,
        frame_png_bytes=class_values["frame_png_bytes"],
        exported_video_bytes=class_values["exported_video_bytes"],
        sqlite_bytes=class_values["sqlite_bytes"],
        json_bytes=class_values["json_bytes"],
        content_addressed_blob_bytes=class_values["content_addressed_blob_bytes"],
        other_bytes=class_values["other_bytes"],
    )


def discover_canonical_profile_durations(
    state_root: Path,
    *,
    source_sha256: str,
) -> tuple[int | None, int | None]:
    """Read persisted source evidence and return recording/requested durations when available."""

    if not isinstance(state_root, Path):
        raise TypeError("state_root must be pathlib.Path")
    if not isinstance(source_sha256, str):
        raise TypeError("source_sha256 must be a string")
    root = state_root.resolve()
    if not root.exists():
        return None, None
    manifests = sorted(root.glob("mcap/*/video-view/camera-video-export-manifest.json"))
    for manifest_path in manifests:
        document = _read_json_object(manifest_path)
        if document is None or document.get("source_content_sha256") != source_sha256:
            continue
        report_path = manifest_path.parents[1] / "media-quality-report.json"
        report = _read_json_object(report_path)
        if report is None:
            return None, None
        recording_duration = _canonical_nonnegative_integer(report.get("recording_duration_ns"))
        requested = report.get("requested_interval")
        if recording_duration is None or not isinstance(requested, dict):
            return recording_duration, None
        start = _canonical_nonnegative_integer(requested.get("start_ns"))
        end = _canonical_nonnegative_integer(requested.get("end_ns"))
        if start is None or end is None or end <= start:
            return recording_duration, None
        return recording_duration, end - start
    return None, None


def _git_facts(repository_root: Path) -> ProfileGitFacts:
    head = _run_git(repository_root, "rev-parse", "--verify", "HEAD^{commit}")
    status = _run_git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    try:
        commit = head.decode("ascii").strip().lower()
    except UnicodeDecodeError as error:
        raise CanonicalProfileError("git HEAD is not ASCII") from error
    return ProfileGitFacts(head_commit=commit, dirty=bool(status))


def _run_git(repository_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise CanonicalProfileError(f"cannot execute git: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CanonicalProfileError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _runtime_facts() -> ProfileRuntimeFacts:
    return ProfileRuntimeFacts(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine() or "UNKNOWN",
        logical_cpu_count=os.cpu_count(),
    )


def _policy_facts() -> CanonicalProfilePolicyFacts:
    from robata.application.canonical.local_composition import (
        local_canonical_runtime_descriptor,
    )

    descriptor = local_canonical_runtime_descriptor()
    return CanonicalProfilePolicyFacts(
        composition_version=descriptor.composition_version,
        pipeline_version=descriptor.pipeline_version,
        execution_policy_semantic_sha256=descriptor.execution_policy_semantic_sha256,
        runtime_policy_semantic_sha256=descriptor.runtime_policy_semantic_sha256,
        input_planner_version=descriptor.input_planner_version,
        parser_version=descriptor.parser_version,
        inference_policy_versions=descriptor.inference_policy_versions,
    )


def _classify_state_file(path: Path) -> StateFileClass:
    name = path.name.lower()
    lowered_parts = {part.lower() for part in path.parts}
    if name.endswith(".png"):
        return StateFileClass.FRAME_PNG
    if name.endswith(".mp4"):
        return StateFileClass.EXPORTED_VIDEO
    if name.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".sqlite", ".db")):
        return StateFileClass.SQLITE
    if name.endswith((".json", ".jsonl")):
        return StateFileClass.JSON
    if {"artifact-registry", "blobs", "sha256"} <= lowered_parts:
        return StateFileClass.CONTENT_ADDRESSED_BLOB
    return StateFileClass.OTHER


def _snapshot_sqlite_database(
    root: Path,
    path: Path,
    relative_path: str,
) -> SQLiteDatabaseSnapshot:
    del root
    try:
        uri = _immutable_sqlite_uri(path)
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'
                ORDER BY name
                """
            ).fetchall()
            tables = tuple(
                SQLiteTableRowCount(
                    table_name=str(row[0]),
                    row_count=_sqlite_row_count(connection, str(row[0])),
                )
                for row in rows
            )
        return SQLiteDatabaseSnapshot(relative_path=relative_path, tables=tables)
    except (OSError, sqlite3.Error, TypeError, ValueError, _SQLiteSnapshotUnstableError) as error:
        return SQLiteDatabaseSnapshot(
            relative_path=relative_path,
            tables=(),
            error=SQLiteReadError(
                error_type=type(error).__name__,
                detail=str(error) or "SQLite snapshot failed without a diagnostic",
            ),
        )


def _immutable_sqlite_uri(path: Path) -> str:
    resolved = path.resolve()
    wal_path = resolved.with_name(f"{resolved.name}-wal")
    try:
        if wal_path.exists():
            if wal_path.is_symlink() or not wal_path.is_file():
                raise OSError("SQLite WAL is not a regular file")
            if wal_path.stat().st_size:
                raise _SQLiteSnapshotUnstableError(
                    "nonempty SQLite WAL prevents a stable immutable snapshot"
                )
    except OSError:
        raise
    return f"{resolved.as_uri()}?mode=ro&immutable=1"


def _sqlite_row_count(connection: sqlite3.Connection, table_name: str) -> int:
    quoted = '"' + table_name.replace('"', '""') + '"'
    row = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()
    if row is None or isinstance(row[0], bool) or not isinstance(row[0], int) or row[0] < 0:
        raise sqlite3.DatabaseError("SQLite table count is not a nonnegative integer")
    return cast(int, row[0])


def _sqlite_nonnegative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise sqlite3.DatabaseError("SQLite count is not a nonnegative integer")
    return value


def _oldest_queue_age_ns(
    oldest_row: tuple[object, ...] | None,
    observed_at: datetime,
    backlog_count: int,
) -> int | None:
    oldest_value = None if oldest_row is None or not oldest_row else oldest_row[0]
    if backlog_count == 0:
        if oldest_value is not None:
            raise sqlite3.DatabaseError("empty queue backlog has an oldest timestamp")
        return None
    if not isinstance(oldest_value, str) or not oldest_value:
        raise sqlite3.DatabaseError("nonempty queue backlog has no oldest timestamp")
    normalized = f"{oldest_value[:-1]}+00:00" if oldest_value.endswith("Z") else oldest_value
    try:
        oldest = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise sqlite3.DatabaseError("queue created_at is not RFC3339") from error
    if oldest.tzinfo is None or oldest.utcoffset() is None:
        raise sqlite3.DatabaseError("queue created_at does not include a timezone")
    delta = observed_at - oldest.astimezone(UTC)
    age_ns = (
        delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
    )
    if age_ns < 0:
        raise sqlite3.DatabaseError("oldest queue item is later than the observation time")
    return age_ns


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _canonical_nonnegative_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if not isinstance(value, str) or not value or (value != "0" and value.startswith("0")):
        return None
    if not value.isascii() or not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


__all__ = [
    "CANONICAL_PROFILE_DURATION_DENOMINATOR_POLICY",
    "CANONICAL_PROFILE_MANIFEST_VERSION",
    "CANONICAL_PROFILE_REPORT_VERSION",
    "ArtifactByteReconciliation",
    "CanonicalProfileError",
    "CanonicalProfileManifest",
    "CanonicalProfilePolicyFacts",
    "CanonicalProfileReconciliation",
    "CanonicalProfileReport",
    "CanonicalProfileRunError",
    "LedgerReconciliation",
    "LedgerReconciliationFact",
    "ProfileFileFact",
    "ProfileGitFacts",
    "ProfileRuntimeFacts",
    "ReconciliationStatus",
    "SQLiteDatabaseSnapshot",
    "SQLiteReadError",
    "SQLiteTableRowCount",
    "SpanReconciliation",
    "StateFileClass",
    "StateFileClassSnapshot",
    "StateTreeSnapshot",
    "WorkQueueBacklogObservation",
    "WorkQueueObservationStatus",
    "WorkQueueStateCount",
    "build_canonical_profile_manifest",
    "build_profile_reconciliation",
    "discover_canonical_profile_durations",
    "exact_file_fact",
    "reconcile_runtime_spans",
    "snapshot_state_tree",
    "snapshot_work_queue",
    "unique_runtime_counter_value",
]
