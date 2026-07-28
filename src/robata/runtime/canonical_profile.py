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
import stat
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
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityComparison,
    MeasuredCapacityInput,
    MeasuredCapacityReport,
    ProviderMode,
    build_measured_capacity_report,
    compare_measured_capacity_reports,
)
from robata.runtime.observability import (
    RuntimeProfileSnapshot,
    RuntimeResourceMeasurement,
    RuntimeSpanSnapshot,
)

CANONICAL_PROFILE_MANIFEST_VERSION: Final = "canonical-profile-manifest-v1"
CANONICAL_PROFILE_REPORT_VERSION: Final = "canonical-profile-report-v3"
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
    file_identity_status: Literal["AVAILABLE", "NOT_AVAILABLE"] = "NOT_AVAILABLE"
    unique_file_count: NonNegativeInt | None = None
    unique_byte_count: NonNegativeInt | None = None
    hardlink_duplicate_path_count: NonNegativeInt | None = None
    hardlink_duplicate_path_bytes: NonNegativeInt | None = None

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
        identity_values = (
            self.unique_file_count,
            self.unique_byte_count,
            self.hardlink_duplicate_path_count,
            self.hardlink_duplicate_path_bytes,
        )
        if self.file_identity_status == "NOT_AVAILABLE":
            if any(value is not None for value in identity_values):
                raise ValueError("unavailable file identity cannot report unique-byte facts")
            return self
        if any(value is None for value in identity_values):
            raise ValueError("available file identity requires complete unique-byte facts")
        assert self.unique_file_count is not None
        assert self.unique_byte_count is not None
        assert self.hardlink_duplicate_path_count is not None
        assert self.hardlink_duplicate_path_bytes is not None
        if self.unique_file_count + self.hardlink_duplicate_path_count != self.file_count:
            raise ValueError("unique and duplicate file paths do not reconcile")
        if self.unique_byte_count + self.hardlink_duplicate_path_bytes != self.byte_count:
            raise ValueError("unique and duplicate file bytes do not reconcile")
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
    physical_duplication_status: Literal["AVAILABLE", "NOT_AVAILABLE"] = "NOT_AVAILABLE"
    unique_state_bytes: NonNegativeInt | None = None
    hardlink_duplicate_path_bytes: NonNegativeInt | None = None

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
        if self.physical_duplication_status == "NOT_AVAILABLE":
            if (
                self.unique_state_bytes is not None
                or self.hardlink_duplicate_path_bytes is not None
            ):
                raise ValueError("unavailable file identity cannot report physical bytes")
            return self
        if self.unique_state_bytes is None or self.hardlink_duplicate_path_bytes is None:
            raise ValueError("available file identity requires physical byte accounting")
        if self.unique_state_bytes + self.hardlink_duplicate_path_bytes != self.state_bytes:
            raise ValueError("physical and duplicate path bytes do not reconcile")
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


class ProfileMetricAvailability(StrEnum):
    """Whether a metric is complete, instrumented-partial, or deliberately absent."""

    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class SQLiteOperationMeasurements(StrictModel):
    """Operation-attributed SQLite facts from the scheduler authority.

    begin_lock_wait_duration_ns is the elapsed duration of SQLite's BEGIN call,
    including lock wait and local SQLite overhead. SQLite's Python API does not expose
    VFS fsync calls, so their count remains explicitly unavailable instead of being
    inferred from commits.
    """

    operation: NonEmptyString
    connection_count: NonNegativeInt
    connection_setup_duration_ns: NonNegativeInt
    transaction_count: NonNegativeInt
    begin_lock_wait_duration_ns: NonNegativeInt
    transaction_duration_ns: NonNegativeInt
    operation_duration_ns: NonNegativeInt
    commit_duration_ns: NonNegativeInt
    rollback_duration_ns: NonNegativeInt
    rows_committed: NonNegativeInt
    rows_rolled_back: NonNegativeInt
    retry_count: NonNegativeInt
    rollback_count: NonNegativeInt
    busy_or_locked_failure_count: NonNegativeInt
    fsync_count_status: ProfileMetricAvailability
    fsync_count: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_fsync_measurement(self) -> Self:
        if self.fsync_count_status is ProfileMetricAvailability.NOT_AVAILABLE:
            if self.fsync_count is not None:
                raise ValueError("unavailable SQLite fsync count must be null")
        elif self.fsync_count is None:
            raise ValueError("available SQLite fsync count requires a value")
        return self


class SQLiteProfileMeasurements(StrictModel):
    """SQLite work and byte-accounting facts with explicit scope boundaries.

    The runtime observer currently exposes process I/O, not SQLite-device I/O.  The
    SQLite-specific read/write byte fields therefore remain unavailable instead of
    relabeling whole-process bytes as database bytes.  Connection and transaction
    counters are instrumented-subset observations until every profile-path adapter
    emits the corresponding counters, and are marked ``PARTIAL`` accordingly.
    """

    state_bytes_before: NonNegativeInt
    state_bytes_after: NonNegativeInt
    state_byte_delta: SignedInt
    connection_count_status: ProfileMetricAvailability
    connection_count: NonNegativeInt | None
    transaction_count_status: ProfileMetricAvailability
    transaction_count: NonNegativeInt | None
    read_transaction_count_status: ProfileMetricAvailability
    read_transaction_count: NonNegativeInt | None
    write_transaction_count_status: ProfileMetricAvailability
    write_transaction_count: NonNegativeInt | None
    sqlite_read_bytes_status: ProfileMetricAvailability
    sqlite_read_bytes: NonNegativeInt | None = None
    sqlite_write_bytes_status: ProfileMetricAvailability
    sqlite_write_bytes: NonNegativeInt | None = None
    process_read_bytes: RuntimeResourceMeasurement
    process_write_bytes: RuntimeResourceMeasurement
    operations: tuple[SQLiteOperationMeasurements, ...] = ()

    @model_validator(mode="after")
    def validate_sqlite_measurements(self) -> Self:
        if self.state_byte_delta != self.state_bytes_after - self.state_bytes_before:
            raise ValueError("SQLite state byte delta does not reconcile")
        for status, value, name in (
            (self.connection_count_status, self.connection_count, "connection_count"),
            (self.transaction_count_status, self.transaction_count, "transaction_count"),
            (
                self.read_transaction_count_status,
                self.read_transaction_count,
                "read_transaction_count",
            ),
            (
                self.write_transaction_count_status,
                self.write_transaction_count,
                "write_transaction_count",
            ),
            (self.sqlite_read_bytes_status, self.sqlite_read_bytes, "sqlite_read_bytes"),
            (self.sqlite_write_bytes_status, self.sqlite_write_bytes, "sqlite_write_bytes"),
        ):
            if (
                status
                in {
                    ProfileMetricAvailability.AVAILABLE,
                    ProfileMetricAvailability.PARTIAL,
                }
                and value is None
            ):
                raise ValueError(f"{status.value.lower()} {name} requires a value")
            if status is ProfileMetricAvailability.NOT_AVAILABLE and value is not None:
                raise ValueError(f"unavailable {name} must be null")
        if (
            self.transaction_count is not None
            and self.read_transaction_count is not None
            and self.write_transaction_count is not None
            and self.transaction_count != self.read_transaction_count + self.write_transaction_count
        ):
            raise ValueError("SQLite transaction counts do not reconcile")
        operations = tuple(item.operation for item in self.operations)
        if operations != tuple(sorted(operations)) or len(set(operations)) != len(operations):
            raise ValueError("SQLite operation measurements must be unique and ordered")

        return self


class ProfileStageResource(StrictModel):
    """Inclusive wall and process CPU aggregation for one instrumented stage name."""

    stage: NonEmptyString
    span_count: NonNegativeInt
    inclusive_wall_time_ns: NonNegativeInt
    process_cpu_status: ProfileMetricAvailability
    inclusive_process_cpu_ns: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_stage_resource(self) -> Self:
        if self.span_count <= 0:
            raise ValueError("stage span_count must be positive")
        if self.process_cpu_status is ProfileMetricAvailability.AVAILABLE:
            if self.inclusive_process_cpu_ns is None:
                raise ValueError("available stage process CPU requires a value")
        elif self.inclusive_process_cpu_ns is not None:
            raise ValueError("unavailable stage process CPU must be null")
        return self


class CompletionProfileMeasurements(StrictModel):
    """Size facts for completion command construction and ordered roots."""

    detail_bytes_status: ProfileMetricAvailability
    detail_bytes: NonNegativeInt | None = None
    command_bytes_status: ProfileMetricAvailability
    command_bytes: NonNegativeInt | None = None
    processing_run_bytes_status: ProfileMetricAvailability
    processing_run_bytes: NonNegativeInt | None = None
    ordered_root_collection_count_status: ProfileMetricAvailability
    ordered_root_collection_count: NonNegativeInt | None = None
    ordered_root_leaf_count_status: ProfileMetricAvailability
    ordered_root_leaf_count: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_completion_measurements(self) -> Self:
        for status, value, name in (
            (self.detail_bytes_status, self.detail_bytes, "detail_bytes"),
            (self.command_bytes_status, self.command_bytes, "command_bytes"),
            (
                self.processing_run_bytes_status,
                self.processing_run_bytes,
                "processing_run_bytes",
            ),
            (
                self.ordered_root_collection_count_status,
                self.ordered_root_collection_count,
                "ordered_root_collection_count",
            ),
            (
                self.ordered_root_leaf_count_status,
                self.ordered_root_leaf_count,
                "ordered_root_leaf_count",
            ),
        ):
            if status is ProfileMetricAvailability.NOT_AVAILABLE:
                if value is not None:
                    raise ValueError(f"unavailable {name} must be null")
            elif value is None:
                raise ValueError(f"{status.value.lower()} {name} requires a value")
        return self


class CanonicalProfileMeasurements(StrictModel):
    """Profile-only resource projection independent of capacity-rate eligibility."""

    workload_fingerprint: Sha256Digest
    source_bytes: NonNegativeInt
    recording_count: PositiveInt
    completion: CompletionProfileMeasurements | None = None
    recording_worker_count: PositiveInt
    camera_count: Literal[6]
    provider_mode: ProviderMode
    sqlite: SQLiteProfileMeasurements
    stages: tuple[ProfileStageResource, ...]

    @model_validator(mode="after")
    def validate_measurements(self) -> Self:
        stage_names = tuple(item.stage for item in self.stages)
        if stage_names != tuple(sorted(stage_names)) or len(set(stage_names)) != len(stage_names):
            raise ValueError("profile stages must be unique and ordered")
        return self


class ProfileStageComparison(StrictModel):
    """One stage-level row; ratios remain null for incompatible workload lineage."""

    stage: NonEmptyString
    baseline_inclusive_wall_time_ns: NonNegativeInt | None
    candidate_inclusive_wall_time_ns: NonNegativeInt | None
    inclusive_wall_time_ratio: float | None
    baseline_inclusive_process_cpu_ns: NonNegativeInt | None
    candidate_inclusive_process_cpu_ns: NonNegativeInt | None
    inclusive_process_cpu_ratio: float | None


class ProfileResourceComparison(StrictModel):
    """One source, SQLite, or process-resource value from a profile comparison."""

    metric: NonEmptyString
    baseline_availability: ProfileMetricAvailability
    candidate_availability: ProfileMetricAvailability
    baseline_value: SignedInt | None
    candidate_value: SignedInt | None
    candidate_to_baseline_ratio: float | None

    @model_validator(mode="after")
    def validate_resource_comparison(self) -> Self:
        for availability, value, side in (
            (self.baseline_availability, self.baseline_value, "baseline"),
            (self.candidate_availability, self.candidate_value, "candidate"),
        ):
            if availability is ProfileMetricAvailability.NOT_AVAILABLE and value is not None:
                raise ValueError(f"unavailable {side} resource must be null")
            if availability is not ProfileMetricAvailability.NOT_AVAILABLE and value is None:
                raise ValueError(f"available {side} resource requires a value")
        return self


class CanonicalProfileComparison(StrictModel):
    """Machine-readable profile comparison with capacity and stage attribution."""

    schema_version: Literal["1.0"]
    model_version: Literal["canonical-profile-comparison-v1"]
    baseline_manifest_sha256: Sha256Digest
    candidate_manifest_sha256: Sha256Digest
    capacity: MeasuredCapacityComparison
    resources: tuple[ProfileResourceComparison, ...]
    stages: tuple[ProfileStageComparison, ...]
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    production_eligible: Literal[False] = False
    measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"] = "NOT_PRODUCTION_QUALIFIED"

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        names = tuple(item.stage for item in self.stages)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("comparison stages must be unique and ordered")
        resource_names = tuple(item.metric for item in self.resources)
        if resource_names != tuple(sorted(resource_names)) or len(set(resource_names)) != len(
            resource_names
        ):
            raise ValueError("comparison resources must be unique and ordered")
        if not self.capacity.comparable:
            for stage in self.stages:
                if (
                    stage.inclusive_wall_time_ratio is not None
                    or stage.inclusive_process_cpu_ratio is not None
                ):
                    raise ValueError("non-comparable profile stages must not report ratios")
            if any(item.candidate_to_baseline_ratio is not None for item in self.resources):
                raise ValueError("non-comparable profile resources must not report ratios")
        return self


class CanonicalProfileReport(StrictModel):
    """One local observation bound to an immutable pre-execution manifest."""

    schema_version: Literal["1.0"]
    model_version: Literal[
        "canonical-profile-report-v1",
        "canonical-profile-report-v2",
        "canonical-profile-report-v3",
    ]
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
    measurements: CanonicalProfileMeasurements | None = None
    capacity: MeasuredCapacityReport | None = None
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
            self.model_version in {"canonical-profile-report-v2", "canonical-profile-report-v3"}
            and self.receipt is not None
            and self.reconciliation is None
        ):
            raise ValueError("successful profile reports require reconciliation facts")
        if self.model_version == "canonical-profile-report-v3":
            if self.measurements is None or self.capacity is None:
                raise ValueError("v3 profile reports require measurements and capacity facts")
            if self.measurements.workload_fingerprint != self.capacity.workload_fingerprint:
                raise ValueError("measurement and capacity workload fingerprints must match")
            if self.measurements.provider_mode is not self.capacity.provider_mode:
                raise ValueError("measurement and capacity provider modes must match")
            if self.capacity.evidence_class is not CapacityEvidenceClass.LOCAL_CONFORMANCE:
                raise ValueError("local profile capacity evidence must remain LOCAL_CONFORMANCE")
            if self.measurements.recording_count != self.capacity.recording_count:
                raise ValueError("measurement and capacity recording counts must match")
            if self.measurements.recording_worker_count != self.capacity.recording_worker_count:
                raise ValueError("measurement and capacity worker counts must match")
            if self.measurements.source_bytes != self.manifest.source.byte_count:
                raise ValueError("measurement source bytes must match the manifest")
            if self.measurements.camera_count != self.manifest.camera_count:
                raise ValueError("measurement camera count must match the manifest")
        elif self.measurements is not None or self.capacity is not None:
            raise ValueError("only v3 profile reports may carry measurement capacity facts")
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
    externally_owned_paths: Iterable[Path] = (),
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
    identities: dict[tuple[int, int], int] = {}
    state_unique_identities: set[tuple[int, int]] = set()
    file_identity_available = True
    unique_byte_count = 0
    for external_path in externally_owned_paths:
        path = Path(external_path)
        try:
            identity_stat = path.lstat()
        except OSError as error:
            raise CanonicalProfileError(
                f"cannot inspect externally owned file '{path}': {error}"
            ) from error
        if path.is_symlink() or not stat.S_ISREG(identity_stat.st_mode):
            raise CanonicalProfileError(
                f"externally owned path must be a regular non-symlink file: {path}"
            )
        identity = (identity_stat.st_dev, identity_stat.st_ino)
        if identity_stat.st_ino == 0:
            file_identity_available = False
        else:
            identities[identity] = identity_stat.st_size
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
            try:
                identity_stat = path.lstat() if path.is_symlink() else path.stat()
            except OSError:
                file_identity_available = False
            else:
                identity = (identity_stat.st_dev, identity_stat.st_ino)
                if identity_stat.st_ino == 0:
                    file_identity_available = False
                elif identity not in identities:
                    identities[identity] = byte_count
                    state_unique_identities.add(identity)
                    unique_byte_count += byte_count
                elif identities[identity] != byte_count:
                    raise CanonicalProfileError(
                        f"hardlinked state paths report different sizes for '{relative}'"
                    )

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
        file_identity_status=("AVAILABLE" if file_identity_available else "NOT_AVAILABLE"),
        unique_file_count=(len(state_unique_identities) if file_identity_available else None),
        unique_byte_count=(unique_byte_count if file_identity_available else None),
        hardlink_duplicate_path_count=(
            len(files) - len(state_unique_identities) if file_identity_available else None
        ),
        hardlink_duplicate_path_bytes=(
            sum(item[3] for item in files) - unique_byte_count if file_identity_available else None
        ),
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


def canonical_profile_workload_fingerprint(manifest: CanonicalProfileManifest) -> str:
    """Return the stable workload/configuration fingerprint used for local comparisons.

    The fingerprint deliberately excludes run key, host facts, and git dirty state, so a
    fresh/replay pair can compare the same workload.  It is profiling metadata only and
    never contributes to canonical run identity.
    """

    if not isinstance(manifest, CanonicalProfileManifest):
        raise TypeError("manifest must be a CanonicalProfileManifest")
    return exact_bytes_sha256(
        canonical_json_bytes(
            {
                "source": manifest.source.model_dump(mode="json"),
                "mapping_config": manifest.mapping_config.model_dump(mode="json"),
                "uv_lock": manifest.uv_lock.model_dump(mode="json"),
                "schema_catalog": manifest.schema_catalog.model_dump(mode="json"),
                "policies": manifest.policies.model_dump(mode="json"),
                "camera_count": manifest.camera_count,
                "max_duration_ns": (
                    None if manifest.max_duration_ns is None else str(manifest.max_duration_ns)
                ),
                "duration_denominator_policy": manifest.duration_denominator_policy,
            }
        )
    )


def build_canonical_profile_measurements(
    *,
    observer: RuntimeProfileSnapshot,
    state_before: StateTreeSnapshot,
    state_after: StateTreeSnapshot,
    manifest: CanonicalProfileManifest,
    receipt: CanonicalLocalRunReceipt | None,
    recording_count: int = 1,
    recording_worker_count: int = 1,
) -> CanonicalProfileMeasurements:
    """Project raw observer/state facts into explicit source, SQLite, and CPU units."""

    if not isinstance(observer, RuntimeProfileSnapshot):
        raise TypeError("observer must be a RuntimeProfileSnapshot")
    if not isinstance(state_before, StateTreeSnapshot):
        raise TypeError("state_before must be a StateTreeSnapshot")
    if not isinstance(state_after, StateTreeSnapshot):
        raise TypeError("state_after must be a StateTreeSnapshot")
    if not isinstance(manifest, CanonicalProfileManifest):
        raise TypeError("manifest must be a CanonicalProfileManifest")
    if receipt is not None and not isinstance(receipt, CanonicalLocalRunReceipt):
        raise TypeError("receipt must be a CanonicalLocalRunReceipt or None")
    if (
        isinstance(recording_count, bool)
        or not isinstance(recording_count, int)
        or recording_count <= 0
    ):
        raise ValueError("recording_count must be a positive integer")
    if (
        isinstance(recording_worker_count, bool)
        or not isinstance(recording_worker_count, int)
        or recording_worker_count <= 0
    ):
        raise ValueError("recording_worker_count must be a positive integer")
    sqlite_before = _state_class_bytes(state_before, StateFileClass.SQLITE)
    sqlite_after = _state_class_bytes(state_after, StateFileClass.SQLITE)
    connection_count, transaction_count, read_transactions, write_transactions = (
        _sqlite_observer_counts(observer)
    )
    return CanonicalProfileMeasurements(
        workload_fingerprint=canonical_profile_workload_fingerprint(manifest),
        source_bytes=manifest.source.byte_count,
        recording_count=recording_count,
        recording_worker_count=recording_worker_count,
        camera_count=manifest.camera_count,
        provider_mode=profile_provider_mode(receipt),
        sqlite=SQLiteProfileMeasurements(
            state_bytes_before=sqlite_before,
            state_bytes_after=sqlite_after,
            state_byte_delta=sqlite_after - sqlite_before,
            connection_count_status=(
                ProfileMetricAvailability.PARTIAL
                if connection_count is not None
                else ProfileMetricAvailability.NOT_AVAILABLE
            ),
            connection_count=connection_count,
            transaction_count_status=(
                ProfileMetricAvailability.PARTIAL
                if transaction_count is not None
                else ProfileMetricAvailability.NOT_AVAILABLE
            ),
            transaction_count=transaction_count,
            read_transaction_count_status=(
                ProfileMetricAvailability.PARTIAL
                if read_transactions is not None
                else ProfileMetricAvailability.NOT_AVAILABLE
            ),
            read_transaction_count=read_transactions,
            write_transaction_count_status=(
                ProfileMetricAvailability.PARTIAL
                if write_transactions is not None
                else ProfileMetricAvailability.NOT_AVAILABLE
            ),
            write_transaction_count=write_transactions,
            sqlite_read_bytes_status=ProfileMetricAvailability.NOT_AVAILABLE,
            sqlite_read_bytes=None,
            sqlite_write_bytes_status=ProfileMetricAvailability.NOT_AVAILABLE,
            sqlite_write_bytes=None,
            process_read_bytes=observer.resources.read_bytes_delta,
            process_write_bytes=observer.resources.write_bytes_delta,
            operations=_sqlite_operation_measurements(observer),
        ),
        completion=_completion_profile_measurements(observer),
        stages=_profile_stage_resources(observer),
    )


def build_profile_capacity(
    *,
    observer: RuntimeProfileSnapshot,
    manifest: CanonicalProfileManifest,
    receipt: CanonicalLocalRunReceipt | None,
    execution_mode: Literal["FRESH", "REPLAY", "UNKNOWN"],
    recording_duration_ns: int | None,
    requested_duration_ns: int | None,
    measurements: CanonicalProfileMeasurements,
) -> MeasuredCapacityReport:
    """Build a rate projection only from direct counters and known workload duration."""

    if not isinstance(observer, RuntimeProfileSnapshot):
        raise TypeError("observer must be a RuntimeProfileSnapshot")
    if not isinstance(manifest, CanonicalProfileManifest):
        raise TypeError("manifest must be a CanonicalProfileManifest")
    if receipt is not None and not isinstance(receipt, CanonicalLocalRunReceipt):
        raise TypeError("receipt must be a CanonicalLocalRunReceipt or None")
    if execution_mode not in {"FRESH", "REPLAY", "UNKNOWN"}:
        raise ValueError("execution_mode must be FRESH, REPLAY, or UNKNOWN")
    if not isinstance(measurements, CanonicalProfileMeasurements):
        raise TypeError("measurements must be a CanonicalProfileMeasurements")
    expected_fingerprint = canonical_profile_workload_fingerprint(manifest)
    if measurements.workload_fingerprint != expected_fingerprint:
        raise ValueError("measurements do not match the profile manifest")
    if measurements.provider_mode is not profile_provider_mode(receipt):
        raise ValueError("measurements do not match the profile provider mode")
    for field_name, value in (
        ("recording_duration_ns", recording_duration_ns),
        ("requested_duration_ns", requested_duration_ns),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{field_name} must be a nonnegative integer or None")
    # The duration policy is the requested six-camera interval.  An unbounded/full
    # recording may use recording_duration_ns only when no requested interval was found.
    workload_duration_ns = requested_duration_ns if requested_duration_ns else recording_duration_ns
    if workload_duration_ns is not None and workload_duration_ns <= 0:
        workload_duration_ns = None
    no_provider_calls = measurements.provider_mode is ProviderMode.NO_PROVIDER_CALLS
    # The P0 plan counter marks the new counter family.  Its presence lets us turn
    # counters whose producer intentionally skips zero increments into known zeros;
    # older artifacts stay nullable instead of being retroactively guessed.
    direct_logical_calls = _counter_total(observer, "inference.logical_calls")
    plan_counter_family_observed = direct_logical_calls is not None
    logical_calls = (
        direct_logical_calls
        if direct_logical_calls is not None
        else _counter_total_with_fallback(
            observer,
            primary="inference.logical_calls",
            fallback=("inference.call_parts",),
            zero_when_no_provider_calls=no_provider_calls,
        )
    )
    windows = _counter_total_with_fallback(
        observer,
        primary="sampling.windows",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    call_parts = _counter_total_with_fallback(
        observer,
        primary="inference.call_parts",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    call_splits = _counter_total_with_fallback(
        observer,
        primary="inference.call_splits",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    provider_images = _counter_total_with_fallback(
        observer,
        primary="inference.provider_images",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    unique_images = _counter_total_with_fallback(
        observer,
        primary="inference.unique_images",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    coarse_unique_images = _counter_total_with_fallback(
        observer,
        primary="inference.coarse_unique_images",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    dense_unique_images = _counter_total_with_fallback(
        observer,
        primary="inference.dense_unique_images",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    input_tokens = _counter_total_with_fallback(
        observer,
        primary="inference.input_tokens",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    output_tokens = _counter_total_with_fallback(
        observer,
        primary="inference.output_tokens",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    output_token_responses = _counter_total_with_fallback(
        observer,
        primary="inference.output_token_responses",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    if output_tokens is None and output_token_responses is not None:
        # Output-token counters deliberately skip zero values while their paired
        # response counter records each response with a known output-token value.
        output_tokens = 0
    retries = _counter_total_with_fallback(
        observer,
        primary="inference.provider_retries",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    batches = _counter_total_with_fallback(
        observer,
        primary="inference.provider_batches",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    if batches is None:
        # Old recorder output distinguished scalar and batch dispatches.  They are
        # disjoint provider batches, so adding them is safe only as a compatibility path.
        scalar = _counter_total(observer, "inference.provider_dispatches")
        grouped = _counter_total(observer, "inference.provider_batch_dispatches")
        if scalar is not None or grouped is not None:
            batches = (scalar or 0) + (grouped or 0)
        elif no_provider_calls:
            batches = 0
    batch_requests = _counter_total_with_fallback(
        observer,
        primary="inference.provider_batch_requests",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    dense_logical_calls = _counter_total_with_fallback(
        observer,
        primary="inference.dense_logical_calls",
        zero_when_no_provider_calls=(no_provider_calls or plan_counter_family_observed),
    )
    dense_provider_images = _counter_total(observer, "inference.dense_provider_images")
    if dense_provider_images is None and (no_provider_calls or plan_counter_family_observed):
        dense_provider_images = 0
    http_requests = None if receipt is None else receipt.network_call_count
    return build_measured_capacity_report(
        MeasuredCapacityInput(
            workload_fingerprint=measurements.workload_fingerprint,
            evidence_class=CapacityEvidenceClass.LOCAL_CONFORMANCE,
            provider_mode=measurements.provider_mode,
            execution_mode=execution_mode,
            recording_count=measurements.recording_count,
            recording_worker_count=measurements.recording_worker_count,
            camera_count=measurements.camera_count,
            recording_duration_ns=workload_duration_ns,
            wall_time_ns=observer.elapsed_ns,
            windows=windows,
            unique_images=unique_images,
            coarse_unique_images=coarse_unique_images,
            dense_unique_images=dense_unique_images,
            provider_images=provider_images,
            logical_calls=logical_calls,
            call_parts=call_parts,
            call_splits=call_splits,
            http_requests=http_requests,
            retries=retries,
            batches=batches,
            batch_requests=batch_requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_token_responses=output_token_responses,
            dense_logical_calls=dense_logical_calls,
            dense_provider_images=dense_provider_images,
        )
    )


def profile_provider_mode(receipt: CanonicalLocalRunReceipt | None) -> ProviderMode:
    """Classify provider behavior from the authoritative local receipt, if any."""

    if receipt is None:
        return ProviderMode.UNKNOWN
    if receipt.network_call_count > 0:
        return ProviderMode.NETWORK_PROVIDER
    if receipt.fixture_inference_calls > 0:
        return ProviderMode.LOCAL_OFFLINE_FIXTURE
    return ProviderMode.NO_PROVIDER_CALLS


def compare_canonical_profile_reports(
    baseline: CanonicalProfileReport,
    candidate: CanonicalProfileReport,
) -> CanonicalProfileComparison:
    """Compare two v3 profile reports by unit-safe capacity and instrumented stage."""

    if not isinstance(baseline, CanonicalProfileReport):
        raise TypeError("baseline must be a CanonicalProfileReport")
    if not isinstance(candidate, CanonicalProfileReport):
        raise TypeError("candidate must be a CanonicalProfileReport")
    if (
        baseline.capacity is None
        or candidate.capacity is None
        or baseline.measurements is None
        or candidate.measurements is None
    ):
        raise ValueError("profile comparison requires v3 reports with capacity facts")
    additional_reasons: tuple[str, ...] = ()
    if {baseline.execution_mode, candidate.execution_mode} == {
        "FRESH",
        "REPLAY",
    } and baseline.manifest.run_key != candidate.manifest.run_key:
        # Workload fingerprints intentionally omit run_key so independent runs can
        # share a capacity workload identity.  A replay comparison additionally
        # requires the same run lineage rather than merely matching source bytes.
        additional_reasons = ("RUN_KEY_CHANGED",)
    capacity = compare_measured_capacity_reports(
        baseline.capacity,
        candidate.capacity,
        additional_non_comparable_reasons=additional_reasons,
    )
    resources = _profile_resource_comparisons(
        baseline.measurements,
        candidate.measurements,
        comparable=capacity.comparable,
    )
    baseline_stages = _stages_by_name(baseline.measurements)
    candidate_stages = _stages_by_name(candidate.measurements)
    stages: list[ProfileStageComparison] = []
    for name in sorted(set(baseline_stages) | set(candidate_stages)):
        before = baseline_stages.get(name)
        after = candidate_stages.get(name)
        before_wall = None if before is None else before.inclusive_wall_time_ns
        after_wall = None if after is None else after.inclusive_wall_time_ns
        before_cpu = None if before is None else before.inclusive_process_cpu_ns
        after_cpu = None if after is None else after.inclusive_process_cpu_ns
        stages.append(
            ProfileStageComparison(
                stage=name,
                baseline_inclusive_wall_time_ns=before_wall,
                candidate_inclusive_wall_time_ns=after_wall,
                inclusive_wall_time_ratio=(
                    _profile_ratio(after_wall, before_wall) if capacity.comparable else None
                ),
                baseline_inclusive_process_cpu_ns=before_cpu,
                candidate_inclusive_process_cpu_ns=after_cpu,
                inclusive_process_cpu_ratio=(
                    _profile_ratio(after_cpu, before_cpu) if capacity.comparable else None
                ),
            )
        )
    return CanonicalProfileComparison(
        schema_version="1.0",
        model_version="canonical-profile-comparison-v1",
        baseline_manifest_sha256=baseline.manifest_sha256,
        candidate_manifest_sha256=candidate.manifest_sha256,
        capacity=capacity,
        resources=resources,
        stages=tuple(stages),
    )


def _profile_resource_comparisons(
    baseline: CanonicalProfileMeasurements,
    candidate: CanonicalProfileMeasurements,
    *,
    comparable: bool,
) -> tuple[ProfileResourceComparison, ...]:
    """Compare source, SQLite, and process-resource values with availability intact."""

    baseline_sqlite = baseline.sqlite
    candidate_sqlite = candidate.sqlite
    values: dict[
        str,
        tuple[
            ProfileMetricAvailability,
            int | None,
            ProfileMetricAvailability,
            int | None,
        ],
    ] = {
        "process.read_bytes": _resource_comparison_values(
            baseline_sqlite.process_read_bytes,
            candidate_sqlite.process_read_bytes,
        ),
        "process.write_bytes": _resource_comparison_values(
            baseline_sqlite.process_write_bytes,
            candidate_sqlite.process_write_bytes,
        ),
        "source.bytes": (
            ProfileMetricAvailability.AVAILABLE,
            baseline.source_bytes,
            ProfileMetricAvailability.AVAILABLE,
            candidate.source_bytes,
        ),
        "sqlite.connection_count": (
            baseline_sqlite.connection_count_status,
            baseline_sqlite.connection_count,
            candidate_sqlite.connection_count_status,
            candidate_sqlite.connection_count,
        ),
        "sqlite.read_bytes": (
            baseline_sqlite.sqlite_read_bytes_status,
            baseline_sqlite.sqlite_read_bytes,
            candidate_sqlite.sqlite_read_bytes_status,
            candidate_sqlite.sqlite_read_bytes,
        ),
        "sqlite.read_transaction_count": (
            baseline_sqlite.read_transaction_count_status,
            baseline_sqlite.read_transaction_count,
            candidate_sqlite.read_transaction_count_status,
            candidate_sqlite.read_transaction_count,
        ),
        "sqlite.state_bytes_after": (
            ProfileMetricAvailability.AVAILABLE,
            baseline_sqlite.state_bytes_after,
            ProfileMetricAvailability.AVAILABLE,
            candidate_sqlite.state_bytes_after,
        ),
        "sqlite.state_bytes_before": (
            ProfileMetricAvailability.AVAILABLE,
            baseline_sqlite.state_bytes_before,
            ProfileMetricAvailability.AVAILABLE,
            candidate_sqlite.state_bytes_before,
        ),
        "sqlite.state_byte_delta": (
            ProfileMetricAvailability.AVAILABLE,
            baseline_sqlite.state_byte_delta,
            ProfileMetricAvailability.AVAILABLE,
            candidate_sqlite.state_byte_delta,
        ),
        "sqlite.transaction_count": (
            baseline_sqlite.transaction_count_status,
            baseline_sqlite.transaction_count,
            candidate_sqlite.transaction_count_status,
            candidate_sqlite.transaction_count,
        ),
        "sqlite.write_bytes": (
            baseline_sqlite.sqlite_write_bytes_status,
            baseline_sqlite.sqlite_write_bytes,
            candidate_sqlite.sqlite_write_bytes_status,
            candidate_sqlite.sqlite_write_bytes,
        ),
        "sqlite.write_transaction_count": (
            baseline_sqlite.write_transaction_count_status,
            baseline_sqlite.write_transaction_count,
            candidate_sqlite.write_transaction_count_status,
            candidate_sqlite.write_transaction_count,
        ),
    }
    values.update(_sqlite_operation_comparison_values(baseline_sqlite, candidate_sqlite))
    values.update(_completion_comparison_values(baseline.completion, candidate.completion))
    return tuple(
        ProfileResourceComparison(
            metric=name,
            baseline_availability=baseline_availability,
            baseline_value=baseline_value,
            candidate_availability=candidate_availability,
            candidate_value=candidate_value,
            candidate_to_baseline_ratio=(
                _profile_ratio(candidate_value, baseline_value) if comparable else None
            ),
        )
        for name, (
            baseline_availability,
            baseline_value,
            candidate_availability,
            candidate_value,
        ) in sorted(values.items())
    )


def _completion_comparison_values(
    baseline: CompletionProfileMeasurements | None,
    candidate: CompletionProfileMeasurements | None,
) -> dict[
    str,
    tuple[
        ProfileMetricAvailability,
        int | None,
        ProfileMetricAvailability,
        int | None,
    ],
]:
    """Project completion payload and ordered-root sizes into profile comparisons."""

    def metric_value(
        measurement: CompletionProfileMeasurements | None,
        name: str,
    ) -> tuple[ProfileMetricAvailability, int | None]:
        if measurement is None:
            return (ProfileMetricAvailability.NOT_AVAILABLE, None)
        status = cast(ProfileMetricAvailability, getattr(measurement, f"{name}_status"))
        value = cast(int | None, getattr(measurement, name))
        return status, value

    names = (
        "command_bytes",
        "detail_bytes",
        "ordered_root_collection_count",
        "ordered_root_leaf_count",
        "processing_run_bytes",
    )
    return {
        f"completion.{name}": (*metric_value(baseline, name), *metric_value(candidate, name))
        for name in names
    }


def _sqlite_operation_comparison_values(
    baseline: SQLiteProfileMeasurements,
    candidate: SQLiteProfileMeasurements,
) -> dict[
    str,
    tuple[
        ProfileMetricAvailability,
        int | None,
        ProfileMetricAvailability,
        int | None,
    ],
]:
    """Project scheduler operations into the existing comparable resource surface."""

    baseline_by_operation = {item.operation: item for item in baseline.operations}
    candidate_by_operation = {item.operation: item for item in candidate.operations}
    metric_names = (
        "begin_lock_wait_duration_ns",
        "busy_or_locked_failure_count",
        "commit_duration_ns",
        "connection_count",
        "connection_setup_duration_ns",
        "fsync_count",
        "operation_duration_ns",
        "rollback_count",
        "rollback_duration_ns",
        "rows_committed",
        "rows_rolled_back",
        "retry_count",
        "transaction_count",
        "transaction_duration_ns",
    )
    values: dict[
        str,
        tuple[
            ProfileMetricAvailability,
            int | None,
            ProfileMetricAvailability,
            int | None,
        ],
    ] = {}
    for operation in sorted(set(baseline_by_operation) | set(candidate_by_operation)):
        baseline_operation = baseline_by_operation.get(operation)
        candidate_operation = candidate_by_operation.get(operation)
        for metric_name in metric_names:
            baseline_status, baseline_value = _sqlite_operation_metric_value(
                baseline_operation,
                metric_name,
            )
            candidate_status, candidate_value = _sqlite_operation_metric_value(
                candidate_operation,
                metric_name,
            )
            values[f"sqlite.operation.{operation}.{metric_name}"] = (
                baseline_status,
                baseline_value,
                candidate_status,
                candidate_value,
            )
    return values


def _sqlite_operation_metric_value(
    operation: SQLiteOperationMeasurements | None,
    metric_name: str,
) -> tuple[ProfileMetricAvailability, int | None]:
    if operation is None:
        return (ProfileMetricAvailability.NOT_AVAILABLE, None)
    if metric_name == "fsync_count":
        return (operation.fsync_count_status, operation.fsync_count)
    return (ProfileMetricAvailability.AVAILABLE, cast(int, getattr(operation, metric_name)))


def _resource_comparison_values(
    baseline: RuntimeResourceMeasurement,
    candidate: RuntimeResourceMeasurement,
) -> tuple[ProfileMetricAvailability, int | None, ProfileMetricAvailability, int | None]:
    return (
        _resource_metric_availability(baseline),
        baseline.value,
        _resource_metric_availability(candidate),
        candidate.value,
    )


def _resource_metric_availability(
    measurement: RuntimeResourceMeasurement,
) -> ProfileMetricAvailability:
    return (
        ProfileMetricAvailability.AVAILABLE
        if measurement.value is not None
        else ProfileMetricAvailability.NOT_AVAILABLE
    )


def _state_class_bytes(snapshot: StateTreeSnapshot, file_class: StateFileClass) -> int:
    for entry in snapshot.classes:
        if entry.file_class is file_class:
            return entry.byte_count
    raise AssertionError("state snapshot omitted a required file class")


def _sqlite_observer_counts(
    observer: RuntimeProfileSnapshot,
) -> tuple[int | None, int | None, int | None, int | None]:
    connections = [
        counter.value
        for counter in observer.counters
        if counter.name.startswith("sqlite.") and counter.name.endswith(".connections")
    ]
    transactions = [
        counter
        for counter in observer.counters
        if counter.name.startswith("sqlite.") and counter.name.endswith(".transactions")
    ]
    if not transactions:
        return (sum(connections) if connections else None, None, None, None)
    total = sum(counter.value for counter in transactions)
    write_count = 0
    for counter in transactions:
        attributes = {attribute.name: attribute.value for attribute in counter.attributes}
        write = attributes.get("write")
        if not isinstance(write, bool):
            return (sum(connections) if connections else None, total, None, None)
        if write:
            write_count += counter.value
    return (sum(connections) if connections else None, total, total - write_count, write_count)


def _sqlite_operation_measurements(
    observer: RuntimeProfileSnapshot,
) -> tuple[SQLiteOperationMeasurements, ...]:
    """Aggregate scheduler facts without turning generic process I/O into SQLite facts."""

    fields = (
        "connection_count",
        "connection_setup_duration_ns",
        "transaction_count",
        "begin_lock_wait_duration_ns",
        "transaction_duration_ns",
        "operation_duration_ns",
        "commit_duration_ns",
        "rollback_duration_ns",
        "rows_committed",
        "rows_rolled_back",
        "retry_count",
        "rollback_count",
        "busy_or_locked_failure_count",
    )
    values_by_operation: dict[str, dict[str, int]] = {}

    def values_for(operation: str) -> dict[str, int]:
        return values_by_operation.setdefault(operation, dict.fromkeys(fields, 0))

    span_fields = {
        "sqlite.work_scheduler.connection_setup": "connection_setup_duration_ns",
        "sqlite.work_scheduler.begin": "begin_lock_wait_duration_ns",
        "sqlite.work_scheduler.transaction": "transaction_duration_ns",
        "sqlite.work_scheduler.operation": "operation_duration_ns",
        "sqlite.work_scheduler.commit": "commit_duration_ns",
        "sqlite.work_scheduler.rollback": "rollback_duration_ns",
    }
    for span in observer.spans:
        field = span_fields.get(span.name)
        if field is None:
            continue
        attributes = {attribute.name: attribute.value for attribute in span.attributes}
        operation = attributes.get("operation")
        if isinstance(operation, str) and operation:
            values_for(operation)[field] += span.elapsed_ns

    counter_fields = {
        "sqlite.work_scheduler.connections": "connection_count",
        "sqlite.work_scheduler.transactions": "transaction_count",
        "sqlite.work_scheduler.rows_committed": "rows_committed",
        "sqlite.work_scheduler.rows_rolled_back": "rows_rolled_back",
        "sqlite.work_scheduler.work_retries": "retry_count",
        "sqlite.work_scheduler.rollbacks": "rollback_count",
        "sqlite.work_scheduler.busy_or_locked_failures": "busy_or_locked_failure_count",
    }
    for counter in observer.counters:
        field = counter_fields.get(counter.name)
        if field is None:
            continue
        attributes = {attribute.name: attribute.value for attribute in counter.attributes}
        operation = attributes.get("operation")
        if isinstance(operation, str) and operation:
            values_for(operation)[field] += counter.value

    return tuple(
        SQLiteOperationMeasurements(
            operation=operation,
            **values,
            fsync_count_status=ProfileMetricAvailability.NOT_AVAILABLE,
            fsync_count=None,
        )
        for operation, values in sorted(values_by_operation.items())
    )


def _completion_profile_measurements(
    observer: RuntimeProfileSnapshot,
) -> CompletionProfileMeasurements:
    """Expose exact P3 counters and preserve absence separately from known zero leaves."""

    root_collections = _counter_total(observer, "completion.command.root_collections")
    detail_bytes = _counter_total(observer, "completion.command.detail_bytes")
    command_bytes = _counter_total(observer, "completion.command.command_bytes")
    processing_run_bytes = _counter_total(
        observer,
        "completion.command.processing_run_bytes",
    )
    root_leaves = _counter_total(observer, "completion.command.root_leaves")

    def observed(value: int | None) -> tuple[ProfileMetricAvailability, int | None]:
        return (
            (ProfileMetricAvailability.AVAILABLE, value)
            if value is not None
            else (ProfileMetricAvailability.NOT_AVAILABLE, None)
        )

    detail_status, detail_value = observed(detail_bytes)
    command_status, command_value = observed(command_bytes)
    processing_status, processing_value = observed(processing_run_bytes)
    collections_status, collections_value = observed(root_collections)
    leaves_status, leaves_value = observed(root_leaves)
    if root_collections is not None and root_leaves is None:
        leaves_status = ProfileMetricAvailability.AVAILABLE
        leaves_value = 0
    return CompletionProfileMeasurements(
        detail_bytes_status=detail_status,
        detail_bytes=detail_value,
        command_bytes_status=command_status,
        command_bytes=command_value,
        processing_run_bytes_status=processing_status,
        processing_run_bytes=processing_value,
        ordered_root_collection_count_status=collections_status,
        ordered_root_collection_count=collections_value,
        ordered_root_leaf_count_status=leaves_status,
        ordered_root_leaf_count=leaves_value,
    )


def _profile_stage_resources(observer: RuntimeProfileSnapshot) -> tuple[ProfileStageResource, ...]:
    grouped: dict[str, list[RuntimeSpanSnapshot]] = {}
    for span in observer.spans:
        grouped.setdefault(span.name, []).append(span)
    rows: list[ProfileStageResource] = []
    for name, spans in sorted(grouped.items()):
        cpu_values = tuple(span.process_cpu_ns for span in spans)
        cpu_available = all(value is not None for value in cpu_values)
        rows.append(
            ProfileStageResource(
                stage=name,
                span_count=len(spans),
                inclusive_wall_time_ns=sum(span.elapsed_ns for span in spans),
                process_cpu_status=(
                    ProfileMetricAvailability.AVAILABLE
                    if cpu_available
                    else ProfileMetricAvailability.NOT_AVAILABLE
                ),
                inclusive_process_cpu_ns=(
                    sum(cast(int, value) for value in cpu_values) if cpu_available else None
                ),
            )
        )
    return tuple(rows)


def _counter_total_with_fallback(
    observer: RuntimeProfileSnapshot,
    *,
    primary: str,
    fallback: tuple[str, ...] = (),
    zero_when_no_provider_calls: bool,
) -> int | None:
    value = _counter_total(observer, primary)
    if value is not None:
        return value
    for name in fallback:
        value = _counter_total(observer, name)
        if value is not None:
            return value
    return 0 if zero_when_no_provider_calls else None


def _stages_by_name(
    measurements: CanonicalProfileMeasurements | None,
) -> dict[str, ProfileStageResource]:
    return {} if measurements is None else {item.stage: item for item in measurements.stages}


def _profile_ratio(candidate: int | None, baseline: int | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


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
        physical_duplication_status=snapshot.file_identity_status,
        unique_state_bytes=snapshot.unique_byte_count,
        hardlink_duplicate_path_bytes=snapshot.hardlink_duplicate_path_bytes,
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
    "CanonicalProfileComparison",
    "CanonicalProfileError",
    "CanonicalProfileManifest",
    "CanonicalProfileMeasurements",
    "CanonicalProfilePolicyFacts",
    "CanonicalProfileReconciliation",
    "CanonicalProfileReport",
    "CanonicalProfileRunError",
    "LedgerReconciliation",
    "LedgerReconciliationFact",
    "ProfileFileFact",
    "ProfileGitFacts",
    "ProfileMetricAvailability",
    "ProfileResourceComparison",
    "ProfileRuntimeFacts",
    "ProfileStageComparison",
    "ProfileStageResource",
    "ReconciliationStatus",
    "SQLiteDatabaseSnapshot",
    "SQLiteOperationMeasurements",
    "SQLiteProfileMeasurements",
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
    "build_canonical_profile_measurements",
    "build_profile_capacity",
    "build_profile_reconciliation",
    "canonical_profile_workload_fingerprint",
    "compare_canonical_profile_reports",
    "discover_canonical_profile_durations",
    "exact_file_fact",
    "profile_provider_mode",
    "reconcile_runtime_spans",
    "snapshot_state_tree",
    "snapshot_work_queue",
    "unique_runtime_counter_value",
]
