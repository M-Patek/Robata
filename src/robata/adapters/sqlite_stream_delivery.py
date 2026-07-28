"""Atomic stream-result publication and durable outbox delivery for local runs."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel

from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler, WorkSchedulerError
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.local_stream_causal import (
    LocalStreamWindowInferencePlan,
    LocalStreamWindowSemanticEvidenceV2,
)
from robata.contracts.stream_common import StreamStage, TerminalOutcome
from robata.contracts.stream_finalization import (
    RecordingFinalizationMap,
    WindowTerminalClosure,
    WindowTerminalMember,
)
from robata.contracts.stream_inference import (
    StreamAcceptedCallEvidence,
    StreamInferenceIntent,
    StreamInferenceTerminal,
    StreamWindowResult,
)
from robata.contracts.stream_planning import (
    ExpectedWindowDeclaration,
    ExpectedWindowPlanSeal,
)
from robata.queue.outbox import (
    Clock,
    OutboxDeliveryClaim,
    OutboxDeliveryError,
    OutboxDeliverySnapshot,
    OutboxDeliveryStatus,
    OutboxFenceError,
    OutboxMessage,
    OutboxRetryPolicy,
)
from robata.queue.stream_models import StreamTerminalEvidence, StreamWorkLease

_EXTENSION_NAME: Final = "stream-delivery-authority"
_EXTENSION_SCHEMA_VERSION: Final = 2
_NONCOMMITTABLE_WINDOW_OUTCOMES: Final = frozenset(
    {
        TerminalOutcome.FAILED,
        TerminalOutcome.CANCELLED,
        TerminalOutcome.EXPIRED,
        TerminalOutcome.QUARANTINED,
        TerminalOutcome.LATE_INPUT,
        TerminalOutcome.INCOMPLETE,
        TerminalOutcome.INVALIDATED,
    }
)
_V1_EXTENSION_OBJECT_NAMES: Final = frozenset(
    {
        "stream_window_results",
        "stream_delivery_outbox",
        "stream_outbox_deliveries",
        "recording_finalizations",
        "stream_delivery_outbox_order",
    }
)
_V1_EXTENSION_SCHEMA_STATEMENTS: Final = (
    """
    CREATE TABLE stream_window_results (
        window_result_id TEXT PRIMARY KEY,
        window_result_key TEXT NOT NULL UNIQUE,
        plan_key TEXT NOT NULL,
        expected_ordinal INTEGER NOT NULL CHECK (expected_ordinal >= 0),
        work_item_id TEXT NOT NULL UNIQUE,
        semantic_sha256 TEXT NOT NULL,
        exact_sha256 TEXT NOT NULL,
        result_json BLOB NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (plan_key, expected_ordinal)
    )
    """,
    """
    CREATE TABLE stream_delivery_outbox (
        outbox_id TEXT PRIMARY KEY,
        plan_key TEXT NOT NULL,
        outbox_ordinal INTEGER NOT NULL CHECK (outbox_ordinal >= 0),
        message_kind TEXT NOT NULL
            CHECK (message_kind IN ('WINDOW_RESULT', 'RECORDING_FINALIZATION')),
        topic TEXT NOT NULL,
        message_key TEXT NOT NULL,
        payload BLOB NOT NULL,
        payload_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        delivered_at TEXT,
        UNIQUE (plan_key, outbox_ordinal)
    )
    """,
    """
    CREATE INDEX stream_delivery_outbox_order
    ON stream_delivery_outbox(plan_key, outbox_ordinal, outbox_id)
    """,
    """
    CREATE TABLE stream_outbox_deliveries (
        outbox_id TEXT PRIMARY KEY,
        status TEXT NOT NULL
            CHECK (status IN ('PENDING', 'LEASED', 'RETRY_WAIT', 'DELIVERED', 'DEAD_LETTER')),
        attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
        lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 0),
        fencing_token TEXT,
        claimed_by TEXT,
        lease_expires_at TEXT,
        next_attempt_at TEXT NOT NULL,
        retry_policy_version TEXT NOT NULL,
        max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
        base_delay_seconds REAL NOT NULL CHECK (base_delay_seconds >= 0),
        max_delay_seconds REAL NOT NULL CHECK (max_delay_seconds >= base_delay_seconds),
        last_error TEXT,
        delivered_at TEXT,
        dead_lettered_at TEXT
    )
    """,
    """
    CREATE TABLE recording_finalizations (
        finalization_key TEXT PRIMARY KEY,
        plan_key TEXT NOT NULL UNIQUE,
        semantic_sha256 TEXT NOT NULL UNIQUE,
        exact_sha256 TEXT NOT NULL,
        finalization_json BLOB NOT NULL,
        outbox_id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    )
    """,
)


_WINDOW_EVIDENCE_COMMIT_STATEMENT: Final = """
    CREATE TABLE stream_window_evidence_commits (
        plan_key TEXT NOT NULL,
        expected_ordinal INTEGER NOT NULL CHECK (expected_ordinal >= 0),
        window_result_id TEXT NOT NULL UNIQUE,
        inference_plan_json BLOB NOT NULL,
        inference_plan_exact_sha256 TEXT NOT NULL,
        semantic_evidence_json BLOB NOT NULL,
        semantic_evidence_exact_sha256 TEXT NOT NULL,
        intent_json BLOB NOT NULL,
        intent_exact_sha256 TEXT NOT NULL,
        accepted_call_json BLOB NOT NULL,
        accepted_call_exact_sha256 TEXT NOT NULL,
        inference_terminal_json BLOB NOT NULL,
        inference_terminal_exact_sha256 TEXT NOT NULL,
        PRIMARY KEY (plan_key, expected_ordinal),
        FOREIGN KEY (window_result_id) REFERENCES stream_window_results (window_result_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """
_EXTENSION_SCHEMA_STATEMENTS: Final = (
    *_V1_EXTENSION_SCHEMA_STATEMENTS,
    _WINDOW_EVIDENCE_COMMIT_STATEMENT,
)
_EXTENSION_OBJECT_NAMES: Final = frozenset(
    {*_V1_EXTENSION_OBJECT_NAMES, "stream_window_evidence_commits"}
)


class SQLiteStreamDeliveryError(OutboxDeliveryError):
    """The stream publication authority cannot preserve its contract."""


class SQLiteStreamDeliveryConflict(SQLiteStreamDeliveryError):
    """An exact replay differs from persisted stream publication state."""


@dataclass(frozen=True, slots=True)
class PreparedWindowReductionEvidence:
    """Exact causal lineage that must be committed with a window result."""

    inference_plan: LocalStreamWindowInferencePlan
    semantic_evidence: LocalStreamWindowSemanticEvidenceV2
    intent: StreamInferenceIntent
    accepted_call: StreamAcceptedCallEvidence
    inference_terminal: StreamInferenceTerminal


@dataclass(frozen=True, slots=True)
class _ValidatedWindowReductionEvidence:
    """Canonical causal records and their exact bytes, ready for one transaction."""

    prepared: PreparedWindowReductionEvidence
    inference_plan_raw: bytes
    semantic_evidence_raw: bytes
    intent_raw: bytes
    accepted_call_raw: bytes
    inference_terminal_raw: bytes


@dataclass(frozen=True, slots=True)
class StreamPublicationCommit:
    """Stable identity returned by an initial commit or exact replay."""

    subject_key: str
    semantic_sha256: str
    exact_sha256: str
    outbox_id: str
    inserted: bool


class SQLiteStreamDeliveryAuthority:
    """One SQLite authority for accepted results, finalization, and relay state."""

    def __init__(
        self,
        authority: SQLiteWorkScheduler,
        *,
        retry_policy: OutboxRetryPolicy,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(authority, SQLiteWorkScheduler):
            raise TypeError("authority must be SQLiteWorkScheduler")
        if not isinstance(retry_policy, OutboxRetryPolicy):
            raise TypeError("retry_policy must be OutboxRetryPolicy")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._authority = authority
        self._retry_policy = retry_policy
        self._clock = clock or _default_clock
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._authority.database_path

    @property
    def authority(self) -> SQLiteWorkScheduler:
        """Return the scheduler that owns this delivery extension's transactions."""

        return self._authority

    def commit_window_reduction(
        self,
        *,
        lease: StreamWorkLease,
        terminal_evidence: StreamTerminalEvidence,
        terminal_member: WindowTerminalMember,
        result: StreamWindowResult,
        causal_evidence: PreparedWindowReductionEvidence | None = None,
        topic: str,
        message_key: str,
        now: datetime | None = None,
    ) -> StreamPublicationCommit:
        """Atomically terminalize WINDOW_REDUCTION and publish its immutable result."""

        checked_lease = _strict(lease, StreamWorkLease, "lease")
        evidence = _strict(terminal_evidence, StreamTerminalEvidence, "terminal_evidence")
        member = _strict(terminal_member, WindowTerminalMember, "terminal_member")
        checked_result = _strict(result, StreamWindowResult, "result")
        checked_causal = _validate_window_reduction_evidence(
            causal_evidence,
            result=checked_result,
        )
        if (
            evidence.outcome in _NONCOMMITTABLE_WINDOW_OUTCOMES
            or member.terminal_outcome in _NONCOMMITTABLE_WINDOW_OUTCOMES
            or checked_result.terminal_outcome in _NONCOMMITTABLE_WINDOW_OUTCOMES
        ):
            raise SQLiteStreamDeliveryConflict(
                "failed or incomplete required window cannot be committed"
            )
        checked_topic = _nonempty(topic, "topic")
        checked_key = _nonempty(message_key, "message_key")
        checked_now = _now(self._clock) if now is None else _checked_datetime(now)
        result_raw = canonical_json_bytes(checked_result)
        result_exact = exact_bytes_sha256(result_raw)
        if _parse_timestamp(evidence.completed_at) > checked_now:
            raise SQLiteStreamDeliveryConflict("terminal evidence completes after authority time")
        if (
            checked_result.terminal_outcome is not evidence.outcome
            or evidence.evidence_ref.exact_sha256 != result_exact
            or evidence.evidence_ref.byte_count != len(result_raw)
            or evidence.evidence_ref.schema_ref != checked_result.schema_ref
            or member.terminal_outcome is not evidence.outcome
            or member.terminal_evidence_ref != evidence.evidence_ref
            or member.terminal_work_item_id != checked_lease.work_item_id
        ):
            raise SQLiteStreamDeliveryConflict(
                "window result, terminal member, and terminal evidence are not one fact"
            )
        evidence_raw = canonical_json_bytes(evidence)
        member_raw = canonical_json_bytes(member)
        outbox_id = _outbox_id("WINDOW_RESULT", checked_result.window_result_key)

        def operation(connection: sqlite3.Connection) -> StreamPublicationCommit:
            row = connection.execute(
                """
                SELECT sw.plan_key, sw.expected_ordinal, sw.stage, sw.work_logical_key,
                       sw.terminal_evidence_json, sw.pending_terminal_json,
                       wi.state, wi.lease_epoch, wi.fencing_token, wi.leased_by,
                       wi.lease_expires_at, wi.attempt, wi.result_reference,
                       wi.result_sha256
                FROM stream_work_plans AS sw
                JOIN work_items AS wi ON wi.work_item_id = sw.work_item_id
                WHERE sw.work_item_id = ?
                """,
                (checked_lease.work_item_id,),
            ).fetchone()
            if row is None:
                raise SQLiteStreamDeliveryError("window reduction work is not durable")
            plan_key = _text(row, "plan_key")
            ordinal = _required_int(row, "expected_ordinal")
            if _text(row, "stage") != StreamStage.WINDOW_REDUCTION.value:
                raise SQLiteStreamDeliveryConflict("only WINDOW_REDUCTION may commit a result")
            declaration_row = connection.execute(
                """
                SELECT declaration_json, terminal_member_json
                FROM expected_windows WHERE plan_key = ? AND ordinal = ?
                """,
                (plan_key, ordinal),
            ).fetchone()
            if declaration_row is None:
                raise SQLiteStreamDeliveryError("window reduction declaration is missing")
            declaration = _parse_exact(
                _bytes(declaration_row, "declaration_json"),
                ExpectedWindowDeclaration,
                "expected declaration",
            )
            if (
                member.plan_key != plan_key
                or member.expected_ordinal != ordinal
                or member.window_key != declaration.window_key
                or member.window_semantic_sha256 != declaration.window_semantic_sha256
                or member.terminal_work_logical_key != _text(row, "work_logical_key")
                or checked_result.window_subject.subject_key != declaration.window_key
                or checked_result.window_subject.subject_semantic_sha256
                != declaration.window_semantic_sha256
                or (
                    checked_causal is not None
                    and (
                        checked_causal.prepared.inference_plan.plan_key != plan_key
                        or checked_causal.prepared.inference_plan.expected_ordinal != ordinal
                        or checked_causal.prepared.inference_plan.window_key
                        != declaration.window_key
                        or checked_causal.prepared.inference_plan.window_semantic_sha256
                        != declaration.window_semantic_sha256
                        or checked_causal.prepared.inference_plan.effective_interval
                        != declaration.effective_interval
                    )
                )
            ):
                raise SQLiteStreamDeliveryConflict(
                    "window publication does not match its declared ordinal"
                )
            existing = connection.execute(
                """
                SELECT * FROM stream_window_results
                WHERE window_result_id = ? OR window_result_key = ?
                   OR (plan_key = ? AND expected_ordinal = ?) OR work_item_id = ?
                """,
                (
                    checked_result.window_result_id,
                    checked_result.window_result_key,
                    plan_key,
                    ordinal,
                    checked_lease.work_item_id,
                ),
            ).fetchall()
            if existing:
                if len(existing) != 1:
                    raise SQLiteStreamDeliveryConflict(
                        "window result identities resolve to different rows"
                    )
                _verify_window_replay(
                    existing[0],
                    result=checked_result,
                    result_raw=result_raw,
                    result_exact=result_exact,
                    plan_key=plan_key,
                    ordinal=ordinal,
                    work_item_id=checked_lease.work_item_id,
                )
                _verify_terminal_replay(
                    row,
                    declaration_row,
                    evidence_raw=evidence_raw,
                    member_raw=member_raw,
                    evidence=evidence,
                )
                _verify_causal_replay(
                    connection,
                    plan_key=plan_key,
                    ordinal=ordinal,
                    window_result_id=checked_result.window_result_id,
                    causal_evidence=checked_causal,
                )
                _verify_outbox(
                    connection,
                    outbox_id=outbox_id,
                    plan_key=plan_key,
                    ordinal=ordinal,
                    kind="WINDOW_RESULT",
                    topic=checked_topic,
                    message_key=checked_key,
                    payload=result_raw,
                )
                return StreamPublicationCommit(
                    subject_key=checked_result.window_result_key,
                    semantic_sha256=checked_result.window_result_semantic_sha256,
                    exact_sha256=result_exact,
                    outbox_id=outbox_id,
                    inserted=False,
                )
            _require_live_running_lease(row, checked_lease, checked_now)
            pending = _optional_bytes(row, "pending_terminal_json")
            if pending is not None and pending != evidence_raw:
                raise SQLiteStreamDeliveryConflict("pending terminal evidence changed")
            existing_member = _optional_bytes(declaration_row, "terminal_member_json")
            if existing_member is not None and existing_member != member_raw:
                raise SQLiteStreamDeliveryConflict("terminal member already differs")
            timestamp = _format_timestamp(checked_now)
            attempt_cursor = connection.execute(
                """
                UPDATE work_attempts
                SET completed_at = ?, outcome = 'SUCCEEDED', error_code = NULL,
                    error_detail = NULL
                WHERE work_item_id = ? AND attempt_number = ? AND lease_epoch = ?
                  AND fencing_token = ? AND outcome = 'ACTIVE'
                """,
                (
                    timestamp,
                    checked_lease.work_item_id,
                    _required_int(row, "attempt"),
                    checked_lease.lease_epoch,
                    checked_lease.fencing_token,
                ),
            )
            if attempt_cursor.rowcount != 1:
                raise SQLiteStreamDeliveryError("active work attempt is missing")
            result_reference = f"stream-terminal-evidence:{evidence.evidence_ref.artifact_id}"
            work_cursor = connection.execute(
                """
                UPDATE work_items
                SET state = 'SUCCEEDED', fencing_token = NULL, leased_by = NULL,
                    lease_expires_at = NULL, retry_not_before_at = NULL,
                    terminal_reason_code = NULL, terminal_reason_detail = NULL,
                    result_reference = ?, result_sha256 = ?, completed_at = ?,
                    updated_at = ?, row_version = row_version + 1
                WHERE work_item_id = ? AND state = 'RUNNING' AND lease_epoch = ?
                  AND fencing_token = ? AND completed_at IS NULL
                """,
                (
                    result_reference,
                    evidence.evidence_ref.exact_sha256,
                    timestamp,
                    timestamp,
                    checked_lease.work_item_id,
                    checked_lease.lease_epoch,
                    checked_lease.fencing_token,
                ),
            )
            if work_cursor.rowcount != 1:
                raise OutboxFenceError("window result work fence is stale")
            connection.execute(
                """
                UPDATE stream_work_plans
                SET terminal_evidence_json = ?, pending_terminal_json = NULL,
                    pending_lease_epoch = NULL, pending_fencing_token = NULL
                WHERE work_item_id = ?
                """,
                (sqlite3.Binary(evidence_raw), checked_lease.work_item_id),
            )
            connection.execute(
                """
                UPDATE expected_windows SET terminal_member_json = ?
                WHERE plan_key = ? AND ordinal = ?
                """,
                (sqlite3.Binary(member_raw), plan_key, ordinal),
            )
            connection.execute(
                """
                INSERT INTO stream_window_results (
                    window_result_id, window_result_key, plan_key, expected_ordinal,
                    work_item_id, semantic_sha256, exact_sha256, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked_result.window_result_id,
                    checked_result.window_result_key,
                    plan_key,
                    ordinal,
                    checked_lease.work_item_id,
                    checked_result.window_result_semantic_sha256,
                    result_exact,
                    sqlite3.Binary(result_raw),
                    checked_result.created_at,
                ),
            )
            if checked_causal is not None:
                _insert_causal_evidence_commit(
                    connection,
                    plan_key=plan_key,
                    ordinal=ordinal,
                    window_result_id=checked_result.window_result_id,
                    causal_evidence=checked_causal,
                )
            self._insert_outbox(
                connection,
                outbox_id=outbox_id,
                plan_key=plan_key,
                ordinal=ordinal,
                kind="WINDOW_RESULT",
                topic=checked_topic,
                message_key=checked_key,
                payload=result_raw,
                created_at=timestamp,
            )
            return StreamPublicationCommit(
                subject_key=checked_result.window_result_key,
                semantic_sha256=checked_result.window_result_semantic_sha256,
                exact_sha256=result_exact,
                outbox_id=outbox_id,
                inserted=True,
            )

        return self._run(
            write=True,
            operation_name="commit_window_reduction",
            operation=operation,
        )

    def publish_recording_finalization(
        self,
        *,
        plan_key: str,
        finalization: RecordingFinalizationMap,
        topic: str,
        message_key: str,
        created_at: datetime | None = None,
    ) -> StreamPublicationCommit:
        """Persist and enqueue finalization only after every durable gate closes."""

        checked_plan_key = _nonempty(plan_key, "plan_key")
        checked = _strict(finalization, RecordingFinalizationMap, "finalization")
        checked_topic = _nonempty(topic, "topic")
        checked_key = _nonempty(message_key, "message_key")
        now = _now(self._clock) if created_at is None else _checked_datetime(created_at)
        timestamp = _format_timestamp(now)
        payload = canonical_json_bytes(checked)
        exact = exact_bytes_sha256(payload)
        outbox_id = _outbox_id("RECORDING_FINALIZATION", checked.finalization_key)

        def operation(connection: sqlite3.Connection) -> StreamPublicationCommit:
            plan = connection.execute(
                """
                SELECT seal_json, terminal_closure_json, export_manifest_sha256
                FROM stream_plans WHERE plan_key = ?
                """,
                (checked_plan_key,),
            ).fetchone()
            if plan is None:
                raise SQLiteStreamDeliveryError("stream plan is not durable")
            seal_raw = _optional_bytes(plan, "seal_json")
            closure_raw = _optional_bytes(plan, "terminal_closure_json")
            export_digest = _optional_text(plan, "export_manifest_sha256")
            if seal_raw is None or closure_raw is None or export_digest is None:
                raise SQLiteStreamDeliveryConflict(
                    "recording finalization requires seal, terminal closure, and export barrier"
                )
            seal = _parse_exact(seal_raw, ExpectedWindowPlanSeal, "expected plan seal")
            closure = _parse_exact(closure_raw, WindowTerminalClosure, "window terminal closure")
            if (
                seal.plan_key != checked_plan_key
                or closure.plan_key != checked_plan_key
                or checked.expected_plan_seal_semantic_sha256 != seal.seal_semantic_sha256
                or checked.window_terminal_closure_semantic_sha256
                != closure.terminal_closure_digest
                or checked.export_manifest_semantic_sha256 != export_digest
            ):
                raise SQLiteStreamDeliveryConflict(
                    "recording finalization does not bind to the closed stream plan"
                )
            windows = connection.execute(
                """
                SELECT ordinal, terminal_member_json FROM expected_windows
                WHERE plan_key = ? ORDER BY ordinal
                """,
                (checked_plan_key,),
            ).fetchall()
            ordinals = tuple(_required_int(row, "ordinal") for row in windows)
            if ordinals != tuple(range(len(windows))) or any(
                _optional_bytes(row, "terminal_member_json") is None for row in windows
            ):
                raise SQLiteStreamDeliveryConflict(
                    "recording finalization requires every declared terminal member"
                )
            result_ordinals = tuple(
                _required_int(row, "expected_ordinal")
                for row in connection.execute(
                    """
                    SELECT expected_ordinal FROM stream_window_results
                    WHERE plan_key = ? ORDER BY expected_ordinal
                    """,
                    (checked_plan_key,),
                ).fetchall()
            )
            if result_ordinals != ordinals:
                raise SQLiteStreamDeliveryConflict(
                    "recording finalization requires one terminal result per declared window"
                )
            causal_ordinals = tuple(
                _required_int(row, "expected_ordinal")
                for row in connection.execute(
                    """
                    SELECT expected_ordinal FROM stream_window_evidence_commits
                    WHERE plan_key = ? ORDER BY expected_ordinal
                    """,
                    (checked_plan_key,),
                ).fetchall()
            )
            if causal_ordinals != ordinals:
                raise SQLiteStreamDeliveryConflict(
                    "recording finalization requires causal evidence for every declared window"
                )
            existing = connection.execute(
                """
                SELECT * FROM recording_finalizations
                WHERE plan_key = ? OR finalization_key = ? OR semantic_sha256 = ?
                """,
                (
                    checked_plan_key,
                    checked.finalization_key,
                    checked.finalization_semantic_sha256,
                ),
            ).fetchall()
            ordinal = len(windows)
            if existing:
                if len(existing) != 1 or not (
                    _text(existing[0], "plan_key") == checked_plan_key
                    and _text(existing[0], "finalization_key") == checked.finalization_key
                    and _text(existing[0], "semantic_sha256")
                    == checked.finalization_semantic_sha256
                    and _text(existing[0], "exact_sha256") == exact
                    and _bytes(existing[0], "finalization_json") == payload
                    and _text(existing[0], "outbox_id") == outbox_id
                ):
                    raise SQLiteStreamDeliveryConflict("finalization replay changed exact facts")
                _verify_outbox(
                    connection,
                    outbox_id=outbox_id,
                    plan_key=checked_plan_key,
                    ordinal=ordinal,
                    kind="RECORDING_FINALIZATION",
                    topic=checked_topic,
                    message_key=checked_key,
                    payload=payload,
                )
                return StreamPublicationCommit(
                    subject_key=checked.finalization_key,
                    semantic_sha256=checked.finalization_semantic_sha256,
                    exact_sha256=exact,
                    outbox_id=outbox_id,
                    inserted=False,
                )
            connection.execute(
                """
                INSERT INTO recording_finalizations (
                    finalization_key, plan_key, semantic_sha256, exact_sha256,
                    finalization_json, outbox_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.finalization_key,
                    checked_plan_key,
                    checked.finalization_semantic_sha256,
                    exact,
                    sqlite3.Binary(payload),
                    outbox_id,
                    timestamp,
                ),
            )
            self._insert_outbox(
                connection,
                outbox_id=outbox_id,
                plan_key=checked_plan_key,
                ordinal=ordinal,
                kind="RECORDING_FINALIZATION",
                topic=checked_topic,
                message_key=checked_key,
                payload=payload,
                created_at=timestamp,
            )
            return StreamPublicationCommit(
                subject_key=checked.finalization_key,
                semantic_sha256=checked.finalization_semantic_sha256,
                exact_sha256=exact,
                outbox_id=outbox_id,
                inserted=True,
            )

        return self._run(
            write=True,
            operation_name="publish_recording_finalization",
            operation=operation,
        )

    def claim(
        self,
        *,
        worker_id: str,
        lease_duration: timedelta,
    ) -> OutboxDeliveryClaim | None:
        checked_worker = _nonempty(worker_id, "worker_id")
        if not isinstance(lease_duration, timedelta) or lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        now = _now(self._clock)
        now_text = _format_timestamp(now)

        def operation(connection: sqlite3.Connection) -> OutboxDeliveryClaim | None:
            self._reconcile(connection, now)
            row = connection.execute(
                """
                SELECT d.outbox_id, d.lease_epoch, d.attempt_count
                FROM stream_outbox_deliveries AS d
                JOIN stream_delivery_outbox AS o ON o.outbox_id = d.outbox_id
                WHERE o.delivered_at IS NULL
                  AND d.status IN ('PENDING', 'RETRY_WAIT')
                  AND d.next_attempt_at <= ?
                ORDER BY o.plan_key, o.outbox_ordinal, o.outbox_id LIMIT 1
                """,
                (now_text,),
            ).fetchone()
            if row is None:
                return None
            outbox_id = _text(row, "outbox_id")
            old_epoch = _required_int(row, "lease_epoch")
            epoch = old_epoch + 1
            attempts = _required_int(row, "attempt_count") + 1
            token = _fencing_token(outbox_id, epoch)
            expires = _format_timestamp(now + lease_duration)
            cursor = connection.execute(
                """
                UPDATE stream_outbox_deliveries
                SET status = 'LEASED', attempt_count = ?, lease_epoch = ?,
                    fencing_token = ?, claimed_by = ?, lease_expires_at = ?,
                    last_error = NULL
                WHERE outbox_id = ? AND status IN ('PENDING', 'RETRY_WAIT')
                  AND lease_epoch = ?
                """,
                (attempts, epoch, token, checked_worker, expires, outbox_id, old_epoch),
            )
            if cursor.rowcount != 1:
                raise SQLiteStreamDeliveryError("stream outbox claim lost its fence")
            return self._claim_from_row(self._joined_row(connection, outbox_id))

        return self._run(write=True, operation_name="claim", operation=operation)

    def acknowledge(self, claim: OutboxDeliveryClaim) -> OutboxDeliverySnapshot:
        checked = _require_claim(claim)
        now = _now(self._clock)
        now_text = _format_timestamp(now)

        def operation(connection: sqlite3.Connection) -> OutboxDeliverySnapshot:
            self._expire_leases(connection, now)
            row = self._joined_row(connection, checked.message.outbox_id)
            if (
                _text(row, "delivery_status") == OutboxDeliveryStatus.DELIVERED.value
                and _optional_text(row, "fencing_token") == checked.delivery.fencing_token
            ):
                return self._snapshot_from_row(row)
            _require_delivery_fence(row, checked)
            cursor = connection.execute(
                """
                UPDATE stream_outbox_deliveries
                SET status = 'DELIVERED', lease_expires_at = NULL,
                    next_attempt_at = ?, delivered_at = ?, dead_lettered_at = NULL,
                    last_error = NULL
                WHERE outbox_id = ? AND status = 'LEASED' AND lease_epoch = ?
                  AND fencing_token = ? AND claimed_by = ?
                """,
                (
                    now_text,
                    now_text,
                    checked.message.outbox_id,
                    checked.delivery.lease_epoch,
                    checked.delivery.fencing_token,
                    checked.delivery.claimed_by,
                ),
            )
            if cursor.rowcount != 1:
                raise OutboxFenceError("stream delivery acknowledgement fence is stale")
            outbox_cursor = connection.execute(
                """
                UPDATE stream_delivery_outbox SET delivered_at = ?
                WHERE outbox_id = ? AND delivered_at IS NULL
                """,
                (now_text, checked.message.outbox_id),
            )
            if outbox_cursor.rowcount != 1:
                raise SQLiteStreamDeliveryError("stream outbox acknowledgement is inconsistent")
            return self._snapshot_from_row(self._joined_row(connection, checked.message.outbox_id))

        return self._run(write=True, operation_name="acknowledge", operation=operation)

    def record_failure(
        self,
        claim: OutboxDeliveryClaim,
        error: str,
    ) -> OutboxDeliverySnapshot:
        checked = _require_claim(claim)
        checked_error = _nonempty(error, "error")[:1000]
        now = _now(self._clock)

        def operation(connection: sqlite3.Connection) -> OutboxDeliverySnapshot:
            self._expire_leases(connection, now)
            row = self._joined_row(connection, checked.message.outbox_id)
            status = OutboxDeliveryStatus(_text(row, "delivery_status"))
            if (
                status in {OutboxDeliveryStatus.RETRY_WAIT, OutboxDeliveryStatus.DEAD_LETTER}
                and _optional_text(row, "fencing_token") == checked.delivery.fencing_token
                and _optional_text(row, "last_error") == checked_error
            ):
                return self._snapshot_from_row(row)
            _require_delivery_fence(row, checked)
            attempt_count = _required_int(row, "attempt_count")
            if attempt_count >= _required_int(row, "max_attempts"):
                next_status = OutboxDeliveryStatus.DEAD_LETTER
                next_at = now
                dead_at: str | None = _format_timestamp(now)
            else:
                policy = _policy_from_row(row)
                next_status = OutboxDeliveryStatus.RETRY_WAIT
                next_at = now + policy.delay_after(attempt_count)
                dead_at = None
            cursor = connection.execute(
                """
                UPDATE stream_outbox_deliveries
                SET status = ?, lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = ?, dead_lettered_at = ?
                WHERE outbox_id = ? AND status = 'LEASED' AND lease_epoch = ?
                  AND fencing_token = ? AND claimed_by = ?
                """,
                (
                    next_status.value,
                    _format_timestamp(next_at),
                    checked_error,
                    dead_at,
                    checked.message.outbox_id,
                    checked.delivery.lease_epoch,
                    checked.delivery.fencing_token,
                    checked.delivery.claimed_by,
                ),
            )
            if cursor.rowcount != 1:
                raise OutboxFenceError("stream delivery failure fence is stale")
            return self._snapshot_from_row(self._joined_row(connection, checked.message.outbox_id))

        return self._run(write=True, operation_name="record_failure", operation=operation)

    def reconcile(self) -> int:
        now = _now(self._clock)
        return self._run(
            write=True,
            operation_name="reconcile",
            operation=lambda connection: self._reconcile(connection, now),
        )

    def get(self, outbox_id: str) -> OutboxDeliverySnapshot | None:
        checked = _nonempty(outbox_id, "outbox_id")

        def operation(connection: sqlite3.Connection) -> OutboxDeliverySnapshot | None:
            row = connection.execute(
                """
                SELECT d.*, d.status AS delivery_status
                FROM stream_outbox_deliveries AS d WHERE d.outbox_id = ?
                """,
                (checked,),
            ).fetchone()
            return None if row is None else self._snapshot_from_row(row)

        return self._run(write=False, operation_name="get", operation=operation)

    def list_dead_letters(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OutboxDeliverySnapshot, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        def operation(connection: sqlite3.Connection) -> tuple[OutboxDeliverySnapshot, ...]:
            rows = connection.execute(
                """
                SELECT d.*, d.status AS delivery_status
                FROM stream_outbox_deliveries AS d
                JOIN stream_delivery_outbox AS o ON o.outbox_id = d.outbox_id
                WHERE d.status = 'DEAD_LETTER'
                ORDER BY o.plan_key, o.outbox_ordinal, o.outbox_id LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(self._snapshot_from_row(row) for row in rows)

        return self._run(write=False, operation_name="list_dead_letters", operation=operation)

    def _insert_outbox(
        self,
        connection: sqlite3.Connection,
        *,
        outbox_id: str,
        plan_key: str,
        ordinal: int,
        kind: str,
        topic: str,
        message_key: str,
        payload: bytes,
        created_at: str,
    ) -> None:
        payload_sha = exact_bytes_sha256(payload)
        connection.execute(
            """
            INSERT INTO stream_delivery_outbox (
                outbox_id, plan_key, outbox_ordinal, message_kind, topic,
                message_key, payload, payload_sha256, created_at, delivered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                outbox_id,
                plan_key,
                ordinal,
                kind,
                topic,
                message_key,
                sqlite3.Binary(payload),
                payload_sha,
                created_at,
            ),
        )
        policy = self._retry_policy
        connection.execute(
            """
            INSERT INTO stream_outbox_deliveries (
                outbox_id, status, attempt_count, lease_epoch, fencing_token,
                claimed_by, lease_expires_at, next_attempt_at,
                retry_policy_version, max_attempts, base_delay_seconds,
                max_delay_seconds, last_error, delivered_at, dead_lettered_at
            ) VALUES (?, 'PENDING', 0, 0, NULL, NULL, NULL, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (
                outbox_id,
                created_at,
                policy.version,
                policy.max_attempts,
                float(policy.base_delay_seconds),
                float(policy.max_delay_seconds),
            ),
        )

    def _reconcile(self, connection: sqlite3.Connection, now: datetime) -> int:
        now_text = _format_timestamp(now)
        before = connection.total_changes
        policy = self._retry_policy
        connection.execute(
            """
            INSERT INTO stream_outbox_deliveries (
                outbox_id, status, attempt_count, lease_epoch, fencing_token,
                claimed_by, lease_expires_at, next_attempt_at,
                retry_policy_version, max_attempts, base_delay_seconds,
                max_delay_seconds, last_error, delivered_at, dead_lettered_at
            )
            SELECT o.outbox_id,
                   CASE WHEN o.delivered_at IS NULL THEN 'PENDING' ELSE 'DELIVERED' END,
                   0, 0, NULL, NULL, NULL, COALESCE(o.delivered_at, ?),
                   ?, ?, ?, ?, NULL, o.delivered_at, NULL
            FROM stream_delivery_outbox AS o
            WHERE NOT EXISTS (
                SELECT 1 FROM stream_outbox_deliveries AS d
                WHERE d.outbox_id = o.outbox_id
            )
            """,
            (
                now_text,
                policy.version,
                policy.max_attempts,
                float(policy.base_delay_seconds),
                float(policy.max_delay_seconds),
            ),
        )
        self._expire_leases(connection, now)
        return connection.total_changes - before

    def _expire_leases(self, connection: sqlite3.Connection, now: datetime) -> None:
        now_text = _format_timestamp(now)
        connection.execute(
            """
            UPDATE stream_outbox_deliveries
            SET status = 'DEAD_LETTER', lease_expires_at = NULL,
                next_attempt_at = ?, last_error = 'delivery lease expired after final attempt',
                dead_lettered_at = ?
            WHERE status = 'LEASED' AND lease_expires_at <= ?
              AND attempt_count >= max_attempts
            """,
            (now_text, now_text, now_text),
        )
        connection.execute(
            """
            UPDATE stream_outbox_deliveries
            SET status = 'RETRY_WAIT', lease_expires_at = NULL,
                next_attempt_at = ?,
                last_error = 'delivery lease expired before acknowledgement'
            WHERE status = 'LEASED' AND lease_expires_at <= ?
              AND attempt_count < max_attempts
            """,
            (now_text, now_text),
        )

    def _joined_row(self, connection: sqlite3.Connection, outbox_id: str) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT o.*, d.status AS delivery_status, d.attempt_count, d.lease_epoch,
                   d.fencing_token, d.claimed_by, d.lease_expires_at,
                   d.next_attempt_at, d.retry_policy_version, d.max_attempts,
                   d.base_delay_seconds, d.max_delay_seconds, d.last_error,
                   d.delivered_at AS delivery_delivered_at, d.dead_lettered_at
            FROM stream_delivery_outbox AS o
            JOIN stream_outbox_deliveries AS d ON d.outbox_id = o.outbox_id
            WHERE o.outbox_id = ?
            """,
            (outbox_id,),
        ).fetchone()
        if row is None:
            raise SQLiteStreamDeliveryError("stream outbox row is missing")
        return cast(sqlite3.Row, row)

    def _claim_from_row(self, row: sqlite3.Row) -> OutboxDeliveryClaim:
        return OutboxDeliveryClaim(
            message=OutboxMessage(
                outbox_id=_text(row, "outbox_id"),
                completion_run_id=_text(row, "plan_key"),
                recording_identity=_text(row, "plan_key"),
                outbox_ordinal=_required_int(row, "outbox_ordinal"),
                topic=_text(row, "topic"),
                key=_text(row, "message_key"),
                payload=_bytes(row, "payload"),
                payload_sha256=_text(row, "payload_sha256"),
            ),
            delivery=self._snapshot_from_row(row),
        )

    def _snapshot_from_row(self, row: sqlite3.Row) -> OutboxDeliverySnapshot:
        delivered_column = (
            "delivery_delivered_at" if "delivery_delivered_at" in row else "delivered_at"
        )
        return OutboxDeliverySnapshot(
            outbox_id=_text(row, "outbox_id"),
            status=OutboxDeliveryStatus(_text(row, "delivery_status")),
            attempt_count=_required_int(row, "attempt_count"),
            lease_epoch=_required_int(row, "lease_epoch"),
            fencing_token=_optional_text(row, "fencing_token"),
            claimed_by=_optional_text(row, "claimed_by"),
            lease_expires_at=_optional_text(row, "lease_expires_at"),
            next_attempt_at=_text(row, "next_attempt_at"),
            retry_policy=_policy_from_row(row),
            last_error=_optional_text(row, "last_error"),
            delivered_at=_optional_text(row, delivered_column),
            dead_lettered_at=_optional_text(row, "dead_lettered_at"),
        )

    def _initialize_database(self) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_extension_metadata (
                    extension_name TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
                )
                """
            )
            metadata = connection.execute(
                """
                SELECT schema_version FROM stream_extension_metadata
                WHERE extension_name = ?
                """,
                (_EXTENSION_NAME,),
            ).fetchone()
            placeholders = ", ".join("?" for _ in _EXTENSION_OBJECT_NAMES)
            rows = connection.execute(
                f"SELECT name FROM sqlite_schema WHERE name IN ({placeholders})",
                tuple(sorted(_EXTENSION_OBJECT_NAMES)),
            ).fetchall()
            existing = {_text(row, "name") for row in rows}
            if metadata is None:
                if existing:
                    raise SQLiteStreamDeliveryError(
                        "refusing to adopt unversioned stream delivery tables"
                    )
                for statement in _EXTENSION_SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO stream_extension_metadata (extension_name, schema_version)
                    VALUES (?, ?)
                    """,
                    (_EXTENSION_NAME, _EXTENSION_SCHEMA_VERSION),
                )
                return
            version = _required_int(metadata, "schema_version")
            if version == 1:
                if existing != _V1_EXTENSION_OBJECT_NAMES:
                    raise SQLiteStreamDeliveryError("stream delivery v1 schema inventory changed")
                connection.execute(_WINDOW_EVIDENCE_COMMIT_STATEMENT)
                cursor = connection.execute(
                    """
                    UPDATE stream_extension_metadata
                    SET schema_version = ?
                    WHERE extension_name = ? AND schema_version = 1
                    """,
                    (_EXTENSION_SCHEMA_VERSION, _EXTENSION_NAME),
                )
                if cursor.rowcount != 1:
                    raise SQLiteStreamDeliveryError(
                        "stream delivery extension migration lost its metadata row"
                    )
                return
            if _required_int(metadata, "schema_version") != _EXTENSION_SCHEMA_VERSION:
                raise SQLiteStreamDeliveryError(
                    "stream delivery extension belongs to another schema version"
                )
            if existing != _EXTENSION_OBJECT_NAMES:
                raise SQLiteStreamDeliveryError("stream delivery schema inventory changed")

        self._run(write=True, operation_name="initialize", operation=operation)

    def _run[T](
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], T],
    ) -> T:
        try:
            return self._authority.run_authority_transaction(
                write=write,
                operation_name=f"stream_delivery.{operation_name}",
                operation=operation,
            )
        except (SQLiteStreamDeliveryError, OutboxFenceError):
            raise
        except (WorkSchedulerError, sqlite3.Error, TypeError, ValueError) as error:
            raise SQLiteStreamDeliveryError(
                f"stream delivery {operation_name} failed: {error}"
            ) from error


def _validate_window_reduction_evidence(
    value: PreparedWindowReductionEvidence | None,
    *,
    result: StreamWindowResult,
) -> _ValidatedWindowReductionEvidence | None:
    """Validate one complete canonical causal chain before opening its transaction."""

    if value is None:
        return None
    if not isinstance(value, PreparedWindowReductionEvidence):
        raise TypeError("causal_evidence must be PreparedWindowReductionEvidence or None")
    prepared = PreparedWindowReductionEvidence(
        inference_plan=_strict(
            value.inference_plan,
            LocalStreamWindowInferencePlan,
            "causal_evidence.inference_plan",
        ),
        semantic_evidence=_strict(
            value.semantic_evidence,
            LocalStreamWindowSemanticEvidenceV2,
            "causal_evidence.semantic_evidence",
        ),
        intent=_strict(value.intent, StreamInferenceIntent, "causal_evidence.intent"),
        accepted_call=_strict(
            value.accepted_call,
            StreamAcceptedCallEvidence,
            "causal_evidence.accepted_call",
        ),
        inference_terminal=_strict(
            value.inference_terminal,
            StreamInferenceTerminal,
            "causal_evidence.inference_terminal",
        ),
    )
    plan_raw = canonical_json_bytes(prepared.inference_plan)
    semantic_raw = canonical_json_bytes(prepared.semantic_evidence)
    intent_raw = canonical_json_bytes(prepared.intent)
    call_raw = canonical_json_bytes(prepared.accepted_call)
    terminal_raw = canonical_json_bytes(prepared.inference_terminal)

    plan = prepared.inference_plan
    semantic = prepared.semantic_evidence
    intent = prepared.intent
    call = prepared.accepted_call
    terminal = prepared.inference_terminal
    _require_exact_canonical_ref(
        semantic.window_inference_plan_ref,
        plan_raw,
        plan.schema_ref,
        "semantic evidence inference plan",
    )
    if (
        semantic.plan_key != plan.plan_key
        or semantic.plan_semantic_sha256 != plan.plan_semantic_sha256
        or semantic.expected_ordinal != plan.expected_ordinal
        or semantic.window_key != plan.window_key
        or semantic.window_semantic_sha256 != plan.window_semantic_sha256
        or semantic.effective_interval != plan.effective_interval
        or semantic.input_plan_semantic_sha256 != plan.input_plan_semantic_sha256
        or semantic.six_camera_slot_closure_semantic_sha256
        != plan.six_camera_slot_closure_semantic_sha256
    ):
        raise SQLiteStreamDeliveryConflict("semantic evidence does not bind to its inference plan")
    _require_exact_canonical_ref(
        intent.input_plan.exact_artifact_ref,
        plan_raw,
        plan.schema_ref,
        "inference intent input plan",
    )
    if (
        intent.input_plan.input_plan_id != plan.input_plan_id
        or intent.input_plan.input_plan_semantic_sha256 != plan.input_plan_semantic_sha256
        or intent.window_subject.subject_key != plan.window_key
        or intent.window_subject.subject_semantic_sha256 != plan.window_semantic_sha256
        or intent.logical_identity.input_plan_semantic_sha256 != plan.input_plan_semantic_sha256
    ):
        raise SQLiteStreamDeliveryConflict(
            "inference intent does not bind to its causal inference plan"
        )
    _require_exact_canonical_ref(
        call.intent_ref.artifact_ref,
        intent_raw,
        intent.schema_ref,
        "accepted call intent",
    )
    if (
        call.intent_ref.intent_semantic_sha256 != intent.intent_semantic_sha256
        or call.intent_ref.stream_inference_logical_id
        != intent.logical_identity.stream_inference_logical_id
        or call.intent_ref.inference_attempt_id != intent.attempt_identity.inference_attempt_id
        or call.intent_ref.input_plan_semantic_sha256 != plan.input_plan_semantic_sha256
    ):
        raise SQLiteStreamDeliveryConflict("accepted call does not bind to its inference intent")
    _require_exact_canonical_ref(
        terminal.intent_ref.artifact_ref,
        intent_raw,
        intent.schema_ref,
        "inference terminal intent",
    )
    _require_exact_canonical_ref(
        terminal.accepted_call_ref.artifact_ref,
        call_raw,
        call.schema_ref,
        "inference terminal accepted call",
    )
    if (
        terminal.logical_identity != intent.logical_identity
        or terminal.attempt_identity != intent.attempt_identity
        or terminal.intent_ref != call.intent_ref
        or terminal.intent_ref.intent_semantic_sha256 != intent.intent_semantic_sha256
        or terminal.accepted_call_ref.accepted_call_semantic_sha256
        != call.accepted_call_semantic_sha256
        or terminal.accepted_call_ref.status is not call.status
        or terminal.status is not call.status
    ):
        raise SQLiteStreamDeliveryConflict(
            "inference terminal does not bind to its intent and accepted call"
        )
    _require_exact_canonical_ref(
        result.result_evidence_ref,
        semantic_raw,
        semantic.schema_ref,
        "window result semantic evidence",
    )
    if (
        result.result_semantic_evidence_sha256 != semantic.semantic_sha256
        or result.window_subject.subject_key != plan.window_key
        or result.window_subject.subject_semantic_sha256 != plan.window_semantic_sha256
        or len(result.accepted_terminals) != 1
    ):
        raise SQLiteStreamDeliveryConflict(
            "window result does not bind to one complete causal evidence chain"
        )
    selected_terminal = result.accepted_terminals[0]
    _require_exact_canonical_ref(
        selected_terminal.artifact_ref,
        terminal_raw,
        terminal.schema_ref,
        "window result inference terminal",
    )
    if (
        selected_terminal.window_key != plan.window_key
        or selected_terminal.window_semantic_sha256 != plan.window_semantic_sha256
        or selected_terminal.purpose is not result.purpose
        or selected_terminal.purpose is not terminal.logical_identity.purpose
        or selected_terminal.inference_semantic_sha256
        != terminal.logical_identity.inference_semantic_sha256
        or selected_terminal.stream_inference_logical_id
        != terminal.logical_identity.stream_inference_logical_id
        or selected_terminal.inference_attempt_id != terminal.attempt_identity.inference_attempt_id
        or selected_terminal.input_plan_semantic_sha256 != plan.input_plan_semantic_sha256
        or selected_terminal.status is not terminal.status
        or selected_terminal.terminal_semantic_sha256 != terminal.terminal_semantic_sha256
    ):
        raise SQLiteStreamDeliveryConflict("window result does not bind to its inference terminal")
    return _ValidatedWindowReductionEvidence(
        prepared=prepared,
        inference_plan_raw=plan_raw,
        semantic_evidence_raw=semantic_raw,
        intent_raw=intent_raw,
        accepted_call_raw=call_raw,
        inference_terminal_raw=terminal_raw,
    )


def _require_exact_canonical_ref(
    reference: object,
    raw: bytes,
    schema_ref: object,
    label: str,
) -> None:
    """Require an artifact reference to name these exact canonical source bytes."""

    exact_sha256 = getattr(reference, "exact_sha256", None)
    byte_count = getattr(reference, "byte_count", None)
    reference_schema = getattr(reference, "schema_ref", None)
    media_type = getattr(reference, "media_type", None)
    if (
        exact_sha256 != exact_bytes_sha256(raw)
        or byte_count != len(raw)
        or reference_schema != schema_ref
        or media_type != "application/json"
    ):
        raise SQLiteStreamDeliveryConflict(f"{label} does not reference exact canonical bytes")


def _insert_causal_evidence_commit(
    connection: sqlite3.Connection,
    *,
    plan_key: str,
    ordinal: int,
    window_result_id: str,
    causal_evidence: _ValidatedWindowReductionEvidence,
) -> None:
    connection.execute(
        """
        INSERT INTO stream_window_evidence_commits (
            plan_key, expected_ordinal, window_result_id,
            inference_plan_json, inference_plan_exact_sha256,
            semantic_evidence_json, semantic_evidence_exact_sha256,
            intent_json, intent_exact_sha256,
            accepted_call_json, accepted_call_exact_sha256,
            inference_terminal_json, inference_terminal_exact_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plan_key,
            ordinal,
            window_result_id,
            sqlite3.Binary(causal_evidence.inference_plan_raw),
            exact_bytes_sha256(causal_evidence.inference_plan_raw),
            sqlite3.Binary(causal_evidence.semantic_evidence_raw),
            exact_bytes_sha256(causal_evidence.semantic_evidence_raw),
            sqlite3.Binary(causal_evidence.intent_raw),
            exact_bytes_sha256(causal_evidence.intent_raw),
            sqlite3.Binary(causal_evidence.accepted_call_raw),
            exact_bytes_sha256(causal_evidence.accepted_call_raw),
            sqlite3.Binary(causal_evidence.inference_terminal_raw),
            exact_bytes_sha256(causal_evidence.inference_terminal_raw),
        ),
    )


def _verify_causal_replay(
    connection: sqlite3.Connection,
    *,
    plan_key: str,
    ordinal: int,
    window_result_id: str,
    causal_evidence: _ValidatedWindowReductionEvidence | None,
) -> None:
    row = connection.execute(
        """
        SELECT * FROM stream_window_evidence_commits
        WHERE plan_key = ? AND expected_ordinal = ?
        """,
        (plan_key, ordinal),
    ).fetchone()
    if causal_evidence is None:
        return
    if row is None:
        raise SQLiteStreamDeliveryConflict(
            "cannot add causal evidence after a window result is already committed"
        )
    if not (
        _text(row, "window_result_id") == window_result_id
        and _bytes(row, "inference_plan_json") == causal_evidence.inference_plan_raw
        and _text(row, "inference_plan_exact_sha256")
        == exact_bytes_sha256(causal_evidence.inference_plan_raw)
        and _bytes(row, "semantic_evidence_json") == causal_evidence.semantic_evidence_raw
        and _text(row, "semantic_evidence_exact_sha256")
        == exact_bytes_sha256(causal_evidence.semantic_evidence_raw)
        and _bytes(row, "intent_json") == causal_evidence.intent_raw
        and _text(row, "intent_exact_sha256") == exact_bytes_sha256(causal_evidence.intent_raw)
        and _bytes(row, "accepted_call_json") == causal_evidence.accepted_call_raw
        and _text(row, "accepted_call_exact_sha256")
        == exact_bytes_sha256(causal_evidence.accepted_call_raw)
        and _bytes(row, "inference_terminal_json") == causal_evidence.inference_terminal_raw
        and _text(row, "inference_terminal_exact_sha256")
        == exact_bytes_sha256(causal_evidence.inference_terminal_raw)
    ):
        raise SQLiteStreamDeliveryConflict("causal evidence replay changed exact facts")


def _verify_window_replay(
    row: sqlite3.Row,
    *,
    result: StreamWindowResult,
    result_raw: bytes,
    result_exact: str,
    plan_key: str,
    ordinal: int,
    work_item_id: str,
) -> None:
    if not (
        _text(row, "window_result_id") == result.window_result_id
        and _text(row, "window_result_key") == result.window_result_key
        and _text(row, "plan_key") == plan_key
        and _required_int(row, "expected_ordinal") == ordinal
        and _text(row, "work_item_id") == work_item_id
        and _text(row, "semantic_sha256") == result.window_result_semantic_sha256
        and _text(row, "exact_sha256") == result_exact
        and _bytes(row, "result_json") == result_raw
        and _text(row, "created_at") == result.created_at
    ):
        raise SQLiteStreamDeliveryConflict("window result replay changed exact facts")


def _verify_terminal_replay(
    work_row: sqlite3.Row,
    declaration_row: sqlite3.Row,
    *,
    evidence_raw: bytes,
    member_raw: bytes,
    evidence: StreamTerminalEvidence,
) -> None:
    if (
        _text(work_row, "state") != "SUCCEEDED"
        or _optional_bytes(work_row, "terminal_evidence_json") != evidence_raw
        or _optional_bytes(declaration_row, "terminal_member_json") != member_raw
        or _optional_text(work_row, "result_reference")
        != f"stream-terminal-evidence:{evidence.evidence_ref.artifact_id}"
        or _optional_text(work_row, "result_sha256") != evidence.evidence_ref.exact_sha256
    ):
        raise SQLiteStreamDeliveryConflict("window terminal replay changed authority facts")


def _verify_outbox(
    connection: sqlite3.Connection,
    *,
    outbox_id: str,
    plan_key: str,
    ordinal: int,
    kind: str,
    topic: str,
    message_key: str,
    payload: bytes,
) -> None:
    row = connection.execute(
        "SELECT * FROM stream_delivery_outbox WHERE outbox_id = ?", (outbox_id,)
    ).fetchone()
    if row is None or not (
        _text(row, "plan_key") == plan_key
        and _required_int(row, "outbox_ordinal") == ordinal
        and _text(row, "message_kind") == kind
        and _text(row, "topic") == topic
        and _text(row, "message_key") == message_key
        and _bytes(row, "payload") == payload
        and _text(row, "payload_sha256") == exact_bytes_sha256(payload)
    ):
        raise SQLiteStreamDeliveryConflict("stream outbox replay changed exact facts")


def _require_live_running_lease(
    row: sqlite3.Row,
    lease: StreamWorkLease,
    now: datetime,
) -> None:
    expires = _optional_text(row, "lease_expires_at")
    if (
        _text(row, "state") != "RUNNING"
        or _required_int(row, "lease_epoch") != lease.lease_epoch
        or _optional_text(row, "fencing_token") != lease.fencing_token
        or _optional_text(row, "leased_by") != lease.worker_id
        or expires is None
        or _parse_timestamp(expires) <= now
    ):
        raise OutboxFenceError("window result work lease is stale, expired, or inactive")


def _require_delivery_fence(row: sqlite3.Row, claim: OutboxDeliveryClaim) -> None:
    if (
        _text(row, "delivery_status") != OutboxDeliveryStatus.LEASED.value
        or _required_int(row, "lease_epoch") != claim.delivery.lease_epoch
        or _optional_text(row, "fencing_token") != claim.delivery.fencing_token
        or _optional_text(row, "claimed_by") != claim.delivery.claimed_by
    ):
        raise OutboxFenceError("stream delivery fence is stale")


def _policy_from_row(row: sqlite3.Row) -> OutboxRetryPolicy:
    return OutboxRetryPolicy(
        version=_text(row, "retry_policy_version"),
        max_attempts=_required_int(row, "max_attempts"),
        base_delay_seconds=float(row["base_delay_seconds"]),
        max_delay_seconds=float(row["max_delay_seconds"]),
    )


def _outbox_id(kind: str, subject_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:stream-delivery-v1:{kind}:{subject_key}"))


def _fencing_token(outbox_id: str, lease_epoch: int) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"robata:stream-outbox-lease-v1:{outbox_id}:{lease_epoch}",
        )
    )


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _now(clock: Clock) -> datetime:
    return _checked_datetime(clock())


def _checked_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return _checked_datetime(datetime.fromisoformat(normalized))


def _strict[T: BaseModel](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    return model_type.model_validate(value.model_dump(mode="python"), strict=True)


def _parse_exact[T: BaseModel](raw: bytes, model_type: type[T], label: str) -> T:
    try:
        parsed = model_type.model_validate_json(raw, strict=True)
    except (TypeError, ValueError) as error:
        raise SQLiteStreamDeliveryError(f"{label} is invalid: {error}") from error
    if canonical_json_bytes(parsed) != raw:
        raise SQLiteStreamDeliveryConflict(f"{label} bytes are not canonical")
    return parsed


def _require_claim(value: object) -> OutboxDeliveryClaim:
    if not isinstance(value, OutboxDeliveryClaim):
        raise TypeError("claim must be OutboxDeliveryClaim")
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _text(row: sqlite3.Row, key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value:
        raise SQLiteStreamDeliveryError(f"persisted {key} must be nonempty text")
    return value


def _optional_text(row: sqlite3.Row, key: str) -> str | None:
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SQLiteStreamDeliveryError(f"persisted {key} must be optional text")
    return value


def _required_int(row: sqlite3.Row, key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SQLiteStreamDeliveryError(f"persisted {key} must be an integer")
    return cast(int, value)


def _bytes(row: sqlite3.Row, key: str) -> bytes:
    value = row[key]
    if not isinstance(value, bytes):
        raise SQLiteStreamDeliveryError(f"persisted {key} must be bytes")
    return value


def _optional_bytes(row: sqlite3.Row, key: str) -> bytes | None:
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, bytes):
        raise SQLiteStreamDeliveryError(f"persisted {key} must be optional bytes")
    return value


__all__ = [
    "SQLiteStreamDeliveryAuthority",
    "SQLiteStreamDeliveryConflict",
    "SQLiteStreamDeliveryError",
    "StreamPublicationCommit",
]
