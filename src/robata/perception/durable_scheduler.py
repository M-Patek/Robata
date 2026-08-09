# ruff: noqa: E501
"""SQLite durable authority for provider-neutral Mage stream perception vNext.

This additive local ledger intentionally has no dependency on the historical
window scheduler. It stores non-overlapping storage-segment/context identities,
uses provider-neutral ``PerceptionStage`` values, and keeps leases/fences outside
of model providers. It is an internal execution projection, not a published wire
contract.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final
from uuid import NAMESPACE_URL, uuid4, uuid5

from robata.application.canonical.mage_stream import MageReasoningContext, MageStreamPlan
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.perception.pipeline import PerceptionStage

DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION: Final = "mage-stream-vnext-durable-scheduler-v2"
DURABLE_PERCEPTION_RUN_KEY_NAMESPACE: Final = "perception-run-vnext-v1"
DURABLE_PERCEPTION_WORK_KEY_NAMESPACE: Final = "perception-work-vnext-v1"
_WORK_UUID_NAMESPACE: Final = "robata:perception-work-vnext-v1"

_NORMAL_STAGES: Final = (
    PerceptionStage.MEDIA_SCAN,
    PerceptionStage.PERCEPTION_OBSERVE,
    PerceptionStage.OBSERVATION_PROJECT,
    PerceptionStage.TEMPORAL_RECONCILE,
)
_EXCEPTION_STAGES: Final = (PerceptionStage.FUSION, PerceptionStage.PERCEPTION_REFINE)
_STAGE_ORDER: Final = {
    PerceptionStage.MEDIA_SCAN: 10,
    PerceptionStage.PERCEPTION_OBSERVE: 20,
    PerceptionStage.OBSERVATION_PROJECT: 30,
    PerceptionStage.TEMPORAL_RECONCILE: 40,
    PerceptionStage.FUSION: 50,
    PerceptionStage.PERCEPTION_REFINE: 60,
    PerceptionStage.FINALIZE: 70,
}


class DurablePerceptionSchedulerError(RuntimeError):
    """The vNext SQLite execution projection cannot preserve its invariants."""


class DurablePerceptionWorkConflict(DurablePerceptionSchedulerError):
    """A semantic identity was replayed with incompatible durable state."""


class DurablePerceptionWorkStateError(DurablePerceptionSchedulerError):
    """A caller attempted an invalid lifecycle transition."""


class DurablePerceptionWorkFenceError(DurablePerceptionSchedulerError):
    """A stale worker attempted to mutate work after its fence changed."""


class DurablePerceptionWorkState(StrEnum):
    PLANNED = "PLANNED"
    READY = "READY"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED_PERMANENT = "FAILED_PERMANENT"


@dataclass(frozen=True, slots=True)
class DurablePerceptionRun:
    run_key: str
    plan_key: str
    plan_semantic_sha256: str
    codec_policy_version: str
    scheduler_policy_version: str
    config_sha256: str
    created_at: str
    derived_work_sealed: bool


@dataclass(frozen=True, slots=True)
class DurablePerceptionContext:
    run_key: str
    focus_segment_ordinal: int
    segment_key: str
    segment_semantic_sha256: str
    context_key: str
    context_semantic_sha256: str
    interval_start_ns: int
    interval_end_ns: int


@dataclass(frozen=True, slots=True)
class DurablePerceptionWorkItem:
    work_item_id: str
    work_logical_key: str
    run_key: str
    context_key: str | None
    focus_segment_ordinal: int | None
    derived_from_work_item_id: str | None
    stage: PerceptionStage
    input_sha256: str
    config_sha256: str
    max_attempts: int
    state: DurablePerceptionWorkState
    lease_epoch: int
    fencing_token: str | None
    leased_by: str | None
    lease_expires_at: str | None
    attempt: int
    retry_not_before_at: str | None
    result_reference: str | None
    result_sha256: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class DurablePerceptionWorkLease:
    work_item_id: str
    worker_id: str
    lease_epoch: int
    fencing_token: str
    lease_expires_at: str


@dataclass(frozen=True, slots=True)
class DurablePerceptionWorkClaim:
    item: DurablePerceptionWorkItem
    lease: DurablePerceptionWorkLease


@dataclass(frozen=True, slots=True)
class DurablePerceptionStageCount:
    stage: PerceptionStage
    planned: int
    ready: int
    leased: int
    running: int
    retry_wait: int
    succeeded: int
    failed_permanent: int


@dataclass(frozen=True, slots=True)
class DurablePerceptionRunSnapshot:
    run: DurablePerceptionRun
    contexts: tuple[DurablePerceptionContext, ...]
    stage_counts: tuple[DurablePerceptionStageCount, ...]

    @property
    def normal_observation_work_count(self) -> int:
        return next(
            item.planned
            for item in self.stage_counts
            if item.stage is PerceptionStage.PERCEPTION_OBSERVE
        )

    @property
    def refinement_work_count(self) -> int:
        return next(
            item.planned
            for item in self.stage_counts
            if item.stage is PerceptionStage.PERCEPTION_REFINE
        )


_SCHEMA: Final = (
    """CREATE TABLE IF NOT EXISTS perception_vnext_runs (
        run_key TEXT PRIMARY KEY, plan_key TEXT NOT NULL,
        plan_semantic_sha256 TEXT NOT NULL, codec_policy_version TEXT NOT NULL,
        scheduler_policy_version TEXT NOT NULL, config_sha256 TEXT NOT NULL,
        plan_json BLOB NOT NULL, created_at TEXT NOT NULL,
        derived_work_sealed INTEGER NOT NULL DEFAULT 0 CHECK(derived_work_sealed IN (0,1)))""",
    """CREATE TABLE IF NOT EXISTS perception_vnext_contexts (
        run_key TEXT NOT NULL REFERENCES perception_vnext_runs(run_key),
        focus_segment_ordinal INTEGER NOT NULL, segment_key TEXT NOT NULL,
        segment_semantic_sha256 TEXT NOT NULL, context_key TEXT NOT NULL,
        context_semantic_sha256 TEXT NOT NULL, interval_start_ns INTEGER NOT NULL,
        interval_end_ns INTEGER NOT NULL, PRIMARY KEY(run_key, focus_segment_ordinal),
        UNIQUE(run_key, context_key))""",
    """CREATE TABLE IF NOT EXISTS perception_vnext_work_items (
        work_item_id TEXT PRIMARY KEY, work_logical_key TEXT NOT NULL UNIQUE,
        run_key TEXT NOT NULL REFERENCES perception_vnext_runs(run_key), context_key TEXT,
        focus_segment_ordinal INTEGER, derived_from_work_item_id TEXT,
        stage TEXT NOT NULL, stage_order INTEGER NOT NULL,
        input_sha256 TEXT NOT NULL, config_sha256 TEXT NOT NULL, max_attempts INTEGER NOT NULL,
        state TEXT NOT NULL, lease_epoch INTEGER NOT NULL, fencing_token TEXT, leased_by TEXT,
        lease_expires_at TEXT, attempt INTEGER NOT NULL, retry_not_before_at TEXT,
        result_reference TEXT, result_sha256 TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        CHECK((context_key IS NULL) = (focus_segment_ordinal IS NULL)),
        CHECK((result_reference IS NULL) = (result_sha256 IS NULL)))""",
    """CREATE TABLE IF NOT EXISTS perception_vnext_work_dependencies (
        downstream_work_item_id TEXT NOT NULL REFERENCES perception_vnext_work_items(work_item_id),
        upstream_work_item_id TEXT NOT NULL REFERENCES perception_vnext_work_items(work_item_id),
        PRIMARY KEY(downstream_work_item_id, upstream_work_item_id))""",
    """CREATE TABLE IF NOT EXISTS perception_vnext_work_attempts (
        work_item_id TEXT NOT NULL REFERENCES perception_vnext_work_items(work_item_id),
        attempt_number INTEGER NOT NULL, lease_epoch INTEGER NOT NULL, fencing_token TEXT NOT NULL UNIQUE,
        worker_id TEXT NOT NULL, claimed_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
        outcome TEXT NOT NULL, error_code TEXT, error_detail TEXT,
        PRIMARY KEY(work_item_id, attempt_number), UNIQUE(work_item_id, lease_epoch))""",
    "CREATE INDEX IF NOT EXISTS perception_vnext_plan ON perception_vnext_runs(plan_key,run_key)",
    "CREATE INDEX IF NOT EXISTS perception_vnext_ready ON perception_vnext_work_items(run_key,state,stage_order,focus_segment_ordinal,work_item_id)",
    "CREATE INDEX IF NOT EXISTS perception_vnext_lease ON perception_vnext_work_items(lease_expires_at) WHERE state IN ('LEASED','RUNNING')",
)


class SQLitePerceptionWorkScheduler:
    """Authoritative SQLite scheduler for the additive Mage stream work graph.

    The class deliberately contains no provider code. A worker claims a durable
    item, runs external work outside the transaction, then accepts a terminal
    result only with the current lease epoch/fencing token.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        scheduler_policy_version: str = DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
    ) -> None:
        _nonempty(scheduler_policy_version, "scheduler_policy_version")
        self._path = Path(database_path).expanduser().resolve()
        if self._path.exists() and self._path.is_dir():
            raise DurablePerceptionSchedulerError("database path must identify a file")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._policy = scheduler_policy_version
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def scheduler_policy_version(self) -> str:
        return self._policy

    def register_plan(
        self,
        plan: MageStreamPlan,
        *,
        codec_policy_version: str,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> DurablePerceptionRun:
        """Persist exact segment/context work once, with no legacy-window nodes."""
        if not isinstance(plan, MageStreamPlan):
            raise TypeError("plan must be MageStreamPlan")
        _nonempty(codec_policy_version, "codec_policy_version")
        _positive_int(max_attempts, "max_attempts")
        current = _now(now)
        config_sha256 = semantic_sha256(
            {
                "scheduler_policy_version": self._policy,
                "codec_policy_version": codec_policy_version,
                "plan_policy_version": plan.policy.policy_version,
            }
        )
        identity = semantic_sha256(
            {
                "plan_key": plan.plan_key,
                "plan_semantic_sha256": plan.plan_semantic_sha256,
                "config_sha256": config_sha256,
            }
        )
        run = DurablePerceptionRun(
            run_key=f"{DURABLE_PERCEPTION_RUN_KEY_NAMESPACE}:{identity}",
            plan_key=plan.plan_key,
            plan_semantic_sha256=plan.plan_semantic_sha256,
            codec_policy_version=codec_policy_version,
            scheduler_policy_version=self._policy,
            config_sha256=config_sha256,
            created_at=_timestamp(current),
            derived_work_sealed=False,
        )
        plan_json = canonical_json_bytes(plan.semantic_projection())
        contexts = tuple(self._context(run.run_key, item) for item in plan.reasoning_contexts)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM perception_vnext_runs WHERE run_key=?",
                (run.run_key,),
            ).fetchone()
            persisted_run = run
            if row is None:
                connection.execute(
                    """INSERT INTO perception_vnext_runs(
                         run_key,plan_key,plan_semantic_sha256,codec_policy_version,
                         scheduler_policy_version,config_sha256,plan_json,created_at,derived_work_sealed)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        run.run_key,
                        run.plan_key,
                        run.plan_semantic_sha256,
                        run.codec_policy_version,
                        run.scheduler_policy_version,
                        run.config_sha256,
                        plan_json,
                        run.created_at,
                        int(run.derived_work_sealed),
                    ),
                )
            else:
                persisted_run = self._run_from_row(row)
                if (
                    not self._same_run_configuration(persisted_run, run)
                    or bytes(row["plan_json"]) != plan_json
                ):
                    raise DurablePerceptionWorkConflict(
                        "existing vNext run identity has a different durable configuration"
                    )
            for context in contexts:
                existing = connection.execute(
                    "SELECT * FROM perception_vnext_contexts WHERE run_key=? AND focus_segment_ordinal=?",
                    (context.run_key, context.focus_segment_ordinal),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO perception_vnext_contexts(
                             run_key,focus_segment_ordinal,segment_key,segment_semantic_sha256,
                             context_key,context_semantic_sha256,interval_start_ns,interval_end_ns)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (
                            context.run_key,
                            context.focus_segment_ordinal,
                            context.segment_key,
                            context.segment_semantic_sha256,
                            context.context_key,
                            context.context_semantic_sha256,
                            context.interval_start_ns,
                            context.interval_end_ns,
                        ),
                    )
                elif self._context_from_row(existing) != context:
                    raise DurablePerceptionWorkConflict(
                        "existing context ordinal has a different causal segment/context identity"
                    )
            work = self._normal_work(persisted_run, contexts, max_attempts, _timestamp(current))
            for item in work:
                self._insert_or_verify_work(connection, item)
            by_context_stage = {(item.context_key, item.stage): item for item in work}
            for downstream, upstream in self._normal_dependencies(by_context_stage):
                self._dependency(connection, downstream, upstream)
            self._refresh_ready(connection, run.run_key, current)
        return persisted_run

    def seal_derived_work(
        self, run_key: str, *, now: datetime | None = None
    ) -> DurablePerceptionRun:
        """Freeze the post-context work set before allowing FINALIZE to run.

        Callers must enqueue and complete every derived FUSION/REFINE item before
        sealing.  Once sealed, late derived scheduling is rejected, making the
        finalization boundary explicit and race-free.
        """
        _nonempty(run_key, "run_key")
        current = _now(now)
        with self._transaction(write=True) as connection:
            run = self._load_run(connection, run_key)
            if run.derived_work_sealed:
                return run
            active = connection.execute(
                """SELECT COUNT(*) AS count FROM perception_vnext_work_items
                   WHERE run_key=? AND stage IN (?,?)
                     AND state != ?""",
                (
                    run_key,
                    PerceptionStage.FUSION.value,
                    PerceptionStage.PERCEPTION_REFINE.value,
                    DurablePerceptionWorkState.SUCCEEDED.value,
                ),
            ).fetchone()
            if active is not None and int(active["count"]) != 0:
                raise DurablePerceptionWorkStateError(
                    "all declared derived work must be SUCCEEDED before the run can be sealed"
                )
            connection.execute(
                "UPDATE perception_vnext_runs SET derived_work_sealed=1 WHERE run_key=?",
                (run_key,),
            )
            self._refresh_ready(connection, run_key, current)
            return self._load_run(connection, run_key)

    def snapshot(self, run_key: str) -> DurablePerceptionRunSnapshot:
        _nonempty(run_key, "run_key")
        with self._transaction(write=False) as connection:
            run_row = connection.execute(
                "SELECT * FROM perception_vnext_runs WHERE run_key=?", (run_key,)
            ).fetchone()
            if run_row is None:
                raise KeyError(f"unknown durable perception run: {run_key}")
            contexts = tuple(
                self._context_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM perception_vnext_contexts WHERE run_key=? ORDER BY focus_segment_ordinal",
                    (run_key,),
                ).fetchall()
            )
            counts: list[DurablePerceptionStageCount] = []
            for stage in PerceptionStage:
                grouped = {
                    str(row["state"]): int(row["count"])
                    for row in connection.execute(
                        """SELECT state,COUNT(*) AS count FROM perception_vnext_work_items
                           WHERE run_key=? AND stage=? GROUP BY state""",
                        (run_key, stage.value),
                    ).fetchall()
                }
                counts.append(
                    DurablePerceptionStageCount(
                        stage=stage,
                        planned=sum(grouped.values()),
                        ready=grouped.get(DurablePerceptionWorkState.READY.value, 0),
                        leased=grouped.get(DurablePerceptionWorkState.LEASED.value, 0),
                        running=grouped.get(DurablePerceptionWorkState.RUNNING.value, 0),
                        retry_wait=grouped.get(DurablePerceptionWorkState.RETRY_WAIT.value, 0),
                        succeeded=grouped.get(DurablePerceptionWorkState.SUCCEEDED.value, 0),
                        failed_permanent=grouped.get(
                            DurablePerceptionWorkState.FAILED_PERMANENT.value, 0
                        ),
                    )
                )
            return DurablePerceptionRunSnapshot(
                run=self._run_from_row(run_row), contexts=contexts, stage_counts=tuple(counts)
            )

    def items_for_run(self, run_key: str) -> tuple[DurablePerceptionWorkItem, ...]:
        _nonempty(run_key, "run_key")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT * FROM perception_vnext_work_items WHERE run_key=?
                   ORDER BY stage_order,focus_segment_ordinal,work_item_id""",
                (run_key,),
            ).fetchall()
            return tuple(self._item_from_row(row) for row in rows)

    def context_work(
        self, run_key: str, focus_segment_ordinal: int
    ) -> tuple[DurablePerceptionWorkItem, ...]:
        _nonempty(run_key, "run_key")
        _nonnegative_int(focus_segment_ordinal, "focus_segment_ordinal")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT * FROM perception_vnext_work_items WHERE run_key=? AND focus_segment_ordinal=?
                   ORDER BY stage_order,work_item_id""",
                (run_key, focus_segment_ordinal),
            ).fetchall()
            return tuple(self._item_from_row(row) for row in rows)

    def dependencies(self, work_item_id: str) -> tuple[str, ...]:
        _nonempty(work_item_id, "work_item_id")
        with self._transaction(write=False) as connection:
            return tuple(
                str(row["upstream_work_item_id"])
                for row in connection.execute(
                    """SELECT upstream_work_item_id FROM perception_vnext_work_dependencies
                       WHERE downstream_work_item_id=? ORDER BY upstream_work_item_id""",
                    (work_item_id,),
                ).fetchall()
            )

    def get(self, work_item_id: str) -> DurablePerceptionWorkItem:
        _nonempty(work_item_id, "work_item_id")
        with self._transaction(write=False) as connection:
            return self._load_item(connection, work_item_id)

    def reconcile(self, *, now: datetime | None = None) -> int:
        """Return expired leases/retry waits to the durable ready graph."""
        current = _now(now)
        with self._transaction(write=True) as connection:
            changed = self._recover_expired(connection, current)
            changed += self._promote_retry_wait(connection, current)
            changed += self._refresh_ready(connection, None, current)
            return changed

    def claim(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        run_key: str | None = None,
        work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkClaim | None:
        """Atomically lease one ready stage without invoking its provider."""
        _nonempty(worker_id, "worker_id")
        _positive_int(lease_duration_seconds, "lease_duration_seconds")
        if run_key is not None:
            _nonempty(run_key, "run_key")
        if work_item_id is not None:
            _nonempty(work_item_id, "work_item_id")
        current = _now(now)
        with self._transaction(write=True) as connection:
            self._recover_expired(connection, current)
            self._promote_retry_wait(connection, current)
            self._refresh_ready(connection, run_key, current)
            clauses = ["state=?"]
            parameters: list[object] = [DurablePerceptionWorkState.READY.value]
            if run_key is not None:
                clauses.append("run_key=?")
                parameters.append(run_key)
            if work_item_id is not None:
                clauses.append("work_item_id=?")
                parameters.append(work_item_id)
            clauses.append(
                "(stage != ? OR EXISTS (SELECT 1 FROM perception_vnext_runs "
                "AS run WHERE run.run_key=perception_vnext_work_items.run_key "
                "AND run.derived_work_sealed=1))"
            )
            parameters.append(PerceptionStage.FINALIZE.value)
            row = connection.execute(
                f"""SELECT * FROM perception_vnext_work_items WHERE {" AND ".join(clauses)}
                    ORDER BY stage_order,focus_segment_ordinal,work_item_id LIMIT 1""",
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            item = self._item_from_row(row)
            epoch = item.lease_epoch + 1
            token = str(uuid4())
            expires = current + timedelta(seconds=lease_duration_seconds)
            cursor = connection.execute(
                """UPDATE perception_vnext_work_items
                   SET state=?,lease_epoch=?,fencing_token=?,leased_by=?,lease_expires_at=?,
                       attempt=attempt+1,retry_not_before_at=NULL,updated_at=?
                   WHERE work_item_id=? AND state=? AND lease_epoch=?""",
                (
                    DurablePerceptionWorkState.LEASED.value,
                    epoch,
                    token,
                    worker_id,
                    _timestamp(expires),
                    _timestamp(current),
                    item.work_item_id,
                    DurablePerceptionWorkState.READY.value,
                    item.lease_epoch,
                ),
            )
            if cursor.rowcount != 1:
                raise DurablePerceptionWorkFenceError("ready work changed while being claimed")
            connection.execute(
                """INSERT INTO perception_vnext_work_attempts(
                       work_item_id,attempt_number,lease_epoch,fencing_token,worker_id,
                       claimed_at,started_at,completed_at,outcome,error_code,error_detail)
                   VALUES(?,?,?,?,?,?,NULL,NULL,'ACTIVE',NULL,NULL)""",
                (item.work_item_id, item.attempt + 1, epoch, token, worker_id, _timestamp(current)),
            )
            leased = self._load_item(connection, item.work_item_id)
            return DurablePerceptionWorkClaim(
                item=leased,
                lease=DurablePerceptionWorkLease(
                    work_item_id=leased.work_item_id,
                    worker_id=worker_id,
                    lease_epoch=epoch,
                    fencing_token=token,
                    lease_expires_at=_timestamp(expires),
                ),
            )

    def claim_and_start(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        run_key: str | None = None,
        work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkClaim | None:
        claim = self.claim(
            worker_id,
            lease_duration_seconds,
            run_key=run_key,
            work_item_id=work_item_id,
            now=now,
        )
        if claim is None:
            return None
        return DurablePerceptionWorkClaim(item=self.start(claim.lease, now=now), lease=claim.lease)

    def start(
        self,
        lease: DurablePerceptionWorkLease,
        *,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkItem:
        current = _now(now)
        with self._transaction(write=True) as connection:
            item = self._live_lease(connection, lease, current)
            if item.stage is PerceptionStage.FINALIZE and not self._run_is_sealed(
                connection, item.run_key
            ):
                raise DurablePerceptionWorkStateError(
                    "FINALIZE cannot start before derived work is sealed"
                )
            if item.state is DurablePerceptionWorkState.RUNNING:
                return item
            if item.state is not DurablePerceptionWorkState.LEASED:
                raise DurablePerceptionWorkStateError("only leased work can start")
            cursor = connection.execute(
                """UPDATE perception_vnext_work_items SET state=?,updated_at=?
                   WHERE work_item_id=? AND state=? AND lease_epoch=? AND fencing_token=?""",
                (
                    DurablePerceptionWorkState.RUNNING.value,
                    _timestamp(current),
                    item.work_item_id,
                    DurablePerceptionWorkState.LEASED.value,
                    lease.lease_epoch,
                    lease.fencing_token,
                ),
            )
            if cursor.rowcount != 1:
                raise DurablePerceptionWorkFenceError("lease changed while starting")
            connection.execute(
                """UPDATE perception_vnext_work_attempts SET started_at=?
                   WHERE work_item_id=? AND lease_epoch=? AND fencing_token=?""",
                (_timestamp(current), item.work_item_id, lease.lease_epoch, lease.fencing_token),
            )
            return self._load_item(connection, item.work_item_id)

    def heartbeat(
        self,
        lease: DurablePerceptionWorkLease,
        lease_duration_seconds: int,
        *,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkLease:
        _positive_int(lease_duration_seconds, "lease_duration_seconds")
        current = _now(now)
        with self._transaction(write=True) as connection:
            item = self._live_lease(connection, lease, current)
            if item.state not in {
                DurablePerceptionWorkState.LEASED,
                DurablePerceptionWorkState.RUNNING,
            }:
                raise DurablePerceptionWorkStateError("only active work can heartbeat")
            expires = current + timedelta(seconds=lease_duration_seconds)
            cursor = connection.execute(
                """UPDATE perception_vnext_work_items SET lease_expires_at=?,updated_at=?
                   WHERE work_item_id=? AND lease_epoch=? AND fencing_token=?""",
                (
                    _timestamp(expires),
                    _timestamp(current),
                    item.work_item_id,
                    lease.lease_epoch,
                    lease.fencing_token,
                ),
            )
            if cursor.rowcount != 1:
                raise DurablePerceptionWorkFenceError("lease changed while heartbeating")
            return DurablePerceptionWorkLease(
                work_item_id=lease.work_item_id,
                worker_id=lease.worker_id,
                lease_epoch=lease.lease_epoch,
                fencing_token=lease.fencing_token,
                lease_expires_at=_timestamp(expires),
            )

    def succeed(
        self,
        lease: DurablePerceptionWorkLease,
        *,
        result_reference: str,
        result_sha256: str,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkItem:
        """Accept one result only under the current worker fence.

        Repeating the exact successful acceptance is safe; a different artifact
        for the same logical work is rejected.
        """
        _nonempty(result_reference, "result_reference")
        _sha256(result_sha256, "result_sha256")
        current = _now(now)
        with self._transaction(write=True) as connection:
            item = self._load_item(connection, lease.work_item_id)
            if item.stage is PerceptionStage.FINALIZE and not self._run_is_sealed(
                connection, item.run_key
            ):
                raise DurablePerceptionWorkStateError(
                    "FINALIZE cannot succeed before derived work is sealed"
                )
            if item.state is DurablePerceptionWorkState.SUCCEEDED:
                if (
                    item.result_reference == result_reference
                    and item.result_sha256 == result_sha256
                ):
                    return item
                raise DurablePerceptionWorkConflict("successful work received a different artifact")
            item = self._live_lease(connection, lease, current)
            if item.state is not DurablePerceptionWorkState.RUNNING:
                raise DurablePerceptionWorkStateError("only running work can succeed")
            cursor = connection.execute(
                """UPDATE perception_vnext_work_items
                   SET state=?,fencing_token=NULL,leased_by=NULL,lease_expires_at=NULL,
                       result_reference=?,result_sha256=?,updated_at=?
                   WHERE work_item_id=? AND state=? AND lease_epoch=? AND fencing_token=?""",
                (
                    DurablePerceptionWorkState.SUCCEEDED.value,
                    result_reference,
                    result_sha256,
                    _timestamp(current),
                    item.work_item_id,
                    DurablePerceptionWorkState.RUNNING.value,
                    lease.lease_epoch,
                    lease.fencing_token,
                ),
            )
            if cursor.rowcount != 1:
                raise DurablePerceptionWorkFenceError("lease changed while succeeding")
            self._finish_attempt(connection, lease, current, "SUCCEEDED", None, None)
            self._refresh_ready(connection, item.run_key, current)
            return self._load_item(connection, item.work_item_id)

    def fail(
        self,
        lease: DurablePerceptionWorkLease,
        *,
        error_code: str,
        retryable: bool,
        error_detail: str | None = None,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkItem:
        """Accept a retryable/nonretryable failure under the current fence."""
        _nonempty(error_code, "error_code")
        if error_detail is not None:
            _nonempty(error_detail, "error_detail")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean")
        _nonnegative_int(retry_delay_seconds, "retry_delay_seconds")
        current = _now(now)
        with self._transaction(write=True) as connection:
            item = self._live_lease(connection, lease, current)
            if item.state not in {
                DurablePerceptionWorkState.LEASED,
                DurablePerceptionWorkState.RUNNING,
            }:
                raise DurablePerceptionWorkStateError("only active work can fail")
            retry = retryable and item.attempt < item.max_attempts
            state = (
                DurablePerceptionWorkState.RETRY_WAIT
                if retry
                else DurablePerceptionWorkState.FAILED_PERMANENT
            )
            retry_at = (
                _timestamp(current + timedelta(seconds=retry_delay_seconds)) if retry else None
            )
            cursor = connection.execute(
                """UPDATE perception_vnext_work_items
                   SET state=?,fencing_token=NULL,leased_by=NULL,lease_expires_at=NULL,
                       retry_not_before_at=?,updated_at=?
                   WHERE work_item_id=? AND lease_epoch=? AND fencing_token=?
                         AND state IN (?,?)""",
                (
                    state.value,
                    retry_at,
                    _timestamp(current),
                    item.work_item_id,
                    lease.lease_epoch,
                    lease.fencing_token,
                    DurablePerceptionWorkState.LEASED.value,
                    DurablePerceptionWorkState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise DurablePerceptionWorkFenceError("lease changed while failing")
            self._finish_attempt(
                connection,
                lease,
                current,
                "FAILED_RETRYABLE" if retry else "FAILED_PERMANENT",
                error_code,
                error_detail,
            )
            if not retry:
                self._refresh_ready(connection, item.run_key, current)
            return self._load_item(connection, item.work_item_id)

    def schedule_derived(
        self,
        *,
        run_key: str,
        focus_segment_ordinal: int,
        stage: PerceptionStage,
        input_sha256: str,
        config_sha256: str,
        max_attempts: int = 3,
        upstream_work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> DurablePerceptionWorkItem:
        """Add post-context FUSION/REFINE work only after its exact input exists.

        Fusion is scheduled when a closed EventTrack makes an exact fusion input
        available. Refinement is scheduled only from a typed ambiguity request;
        normal vNext runs therefore retain a zero ``PERCEPTION_REFINE`` count.
        """
        if stage not in _EXCEPTION_STAGES:
            raise ValueError("only FUSION or PERCEPTION_REFINE is post-context derived work")
        _nonempty(run_key, "run_key")
        _nonnegative_int(focus_segment_ordinal, "focus_segment_ordinal")
        _sha256(input_sha256, "input_sha256")
        _sha256(config_sha256, "config_sha256")
        _positive_int(max_attempts, "max_attempts")
        if upstream_work_item_id is not None:
            _nonempty(upstream_work_item_id, "upstream_work_item_id")
        if stage is PerceptionStage.PERCEPTION_REFINE and upstream_work_item_id is None:
            raise ValueError("PERCEPTION_REFINE requires an upstream FUSION work item")
        if stage is PerceptionStage.FUSION and upstream_work_item_id is not None:
            raise ValueError("FUSION must not declare an upstream derived work item")
        current = _now(now)
        with self._transaction(write=True) as connection:
            run = self._load_run(connection, run_key)
            if run.derived_work_sealed:
                raise DurablePerceptionWorkStateError(
                    "derived work is sealed; no late FUSION or PERCEPTION_REFINE may be scheduled"
                )
            context_row = connection.execute(
                """SELECT * FROM perception_vnext_contexts
                   WHERE run_key=? AND focus_segment_ordinal=?""",
                (run_key, focus_segment_ordinal),
            ).fetchone()
            if context_row is None:
                raise KeyError("unknown context for exceptional perception work")
            context = self._context_from_row(context_row)
            item = self._work_item(
                run,
                context,
                stage,
                input_sha256,
                config_sha256,
                max_attempts,
                _timestamp(current),
                derived_from_work_item_id=upstream_work_item_id,
            )
            self._insert_or_verify_work(connection, item)
            temporal = self._find_stage(
                connection, run_key, context.context_key, PerceptionStage.TEMPORAL_RECONCILE
            )
            upstream = temporal
            if stage is PerceptionStage.PERCEPTION_REFINE:
                assert upstream_work_item_id is not None
                upstream = self._load_item(connection, upstream_work_item_id)
                if (
                    upstream.run_key != run_key
                    or upstream.context_key != context.context_key
                    or upstream.stage is not PerceptionStage.FUSION
                ):
                    raise DurablePerceptionWorkStateError(
                        "PERCEPTION_REFINE upstream must be FUSION for the same context"
                    )
            self._dependency(connection, item.work_item_id, upstream.work_item_id)
            finalize = self._find_stage(connection, run_key, None, PerceptionStage.FINALIZE)
            self._dependency(connection, finalize.work_item_id, item.work_item_id)
            self._refresh_ready(connection, run_key, current)
            return self._load_item(connection, item.work_item_id)

    def _normal_work(
        self,
        run: DurablePerceptionRun,
        contexts: Sequence[DurablePerceptionContext],
        max_attempts: int,
        created_at: str,
    ) -> tuple[DurablePerceptionWorkItem, ...]:
        values: list[DurablePerceptionWorkItem] = []
        for context in contexts:
            for stage in _NORMAL_STAGES:
                source = semantic_sha256(
                    {
                        "context_semantic_sha256": context.context_semantic_sha256,
                        "segment_semantic_sha256": context.segment_semantic_sha256,
                        "stage": stage.value,
                    }
                )
                values.append(
                    self._work_item(
                        run, context, stage, source, run.config_sha256, max_attempts, created_at
                    )
                )
        final_source = semantic_sha256(
            {
                "run_key": run.run_key,
                "plan_semantic_sha256": run.plan_semantic_sha256,
                "stage": PerceptionStage.FINALIZE.value,
            }
        )
        values.append(
            self._work_item(
                run,
                None,
                PerceptionStage.FINALIZE,
                final_source,
                run.config_sha256,
                max_attempts,
                created_at,
            )
        )
        return tuple(values)

    def _normal_dependencies(
        self,
        work: dict[tuple[str | None, PerceptionStage], DurablePerceptionWorkItem],
    ) -> tuple[tuple[str, str], ...]:
        values: list[tuple[str, str]] = []
        context_keys = sorted({key for key, _stage in work if key is not None})
        reconciles: list[DurablePerceptionWorkItem] = []
        for key in context_keys:
            media = work[(key, PerceptionStage.MEDIA_SCAN)]
            observe = work[(key, PerceptionStage.PERCEPTION_OBSERVE)]
            project = work[(key, PerceptionStage.OBSERVATION_PROJECT)]
            reconcile = work[(key, PerceptionStage.TEMPORAL_RECONCILE)]
            values += (
                (observe.work_item_id, media.work_item_id),
                (project.work_item_id, observe.work_item_id),
                (reconcile.work_item_id, project.work_item_id),
            )
            reconciles.append(reconcile)
        reconciles.sort(
            key=lambda item: (
                item.focus_segment_ordinal if item.focus_segment_ordinal is not None else -1,
                item.work_item_id,
            )
        )
        for prior, current in pairwise(reconciles):
            # Causal EventTrack state, not a revived window-N/window-N+1 VLM DAG.
            values.append((current.work_item_id, prior.work_item_id))
        finalize = work[(None, PerceptionStage.FINALIZE)]
        values.extend((finalize.work_item_id, item.work_item_id) for item in reconciles)
        return tuple(values)

    def _work_item(
        self,
        run: DurablePerceptionRun,
        context: DurablePerceptionContext | None,
        stage: PerceptionStage,
        input_sha256: str,
        config_sha256: str,
        max_attempts: int,
        created_at: str,
        derived_from_work_item_id: str | None = None,
    ) -> DurablePerceptionWorkItem:
        identity = semantic_sha256(
            {
                "run_key": run.run_key,
                "context_key": None if context is None else context.context_key,
                "focus_segment_ordinal": None if context is None else context.focus_segment_ordinal,
                "derived_from_work_item_id": derived_from_work_item_id,
                "stage": stage.value,
                "input_sha256": input_sha256,
                "config_sha256": config_sha256,
                "scheduler_policy_version": self._policy,
            }
        )
        return DurablePerceptionWorkItem(
            work_item_id=str(uuid5(NAMESPACE_URL, f"{_WORK_UUID_NAMESPACE}:{identity}")),
            work_logical_key=f"{DURABLE_PERCEPTION_WORK_KEY_NAMESPACE}:{identity}",
            run_key=run.run_key,
            context_key=None if context is None else context.context_key,
            focus_segment_ordinal=None if context is None else context.focus_segment_ordinal,
            derived_from_work_item_id=derived_from_work_item_id,
            stage=stage,
            input_sha256=input_sha256,
            config_sha256=config_sha256,
            max_attempts=max_attempts,
            state=DurablePerceptionWorkState.PLANNED,
            lease_epoch=0,
            fencing_token=None,
            leased_by=None,
            lease_expires_at=None,
            attempt=0,
            retry_not_before_at=None,
            result_reference=None,
            result_sha256=None,
            created_at=created_at,
            updated_at=created_at,
        )

    @staticmethod
    def _context(run_key: str, item: MageReasoningContext) -> DurablePerceptionContext:
        focus = item.ordered_segments[-1]
        return DurablePerceptionContext(
            run_key=run_key,
            focus_segment_ordinal=item.focus_segment_ordinal,
            segment_key=focus.segment_key,
            segment_semantic_sha256=focus.segment_semantic_sha256,
            context_key=item.context_key,
            context_semantic_sha256=item.context_semantic_sha256,
            interval_start_ns=item.context_interval.start_ns,
            interval_end_ns=item.context_interval.end_ns,
        )

    def _insert_or_verify_work(
        self, connection: sqlite3.Connection, item: DurablePerceptionWorkItem
    ) -> None:
        row = connection.execute(
            "SELECT * FROM perception_vnext_work_items WHERE work_item_id=? OR work_logical_key=?",
            (item.work_item_id, item.work_logical_key),
        ).fetchone()
        if row is not None:
            existing = self._item_from_row(row)
            if not self._same_plan(existing, item):
                raise DurablePerceptionWorkConflict(
                    "work identity exists with a different immutable plan"
                )
            return
        connection.execute(
            """INSERT INTO perception_vnext_work_items(
                   work_item_id,work_logical_key,run_key,context_key,focus_segment_ordinal,
                   derived_from_work_item_id,stage,stage_order,input_sha256,config_sha256,max_attempts,state,lease_epoch,
                   fencing_token,leased_by,lease_expires_at,attempt,retry_not_before_at,
                   result_reference,result_sha256,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.work_item_id,
                item.work_logical_key,
                item.run_key,
                item.context_key,
                item.focus_segment_ordinal,
                item.derived_from_work_item_id,
                item.stage.value,
                _STAGE_ORDER[item.stage],
                item.input_sha256,
                item.config_sha256,
                item.max_attempts,
                item.state.value,
                item.lease_epoch,
                item.fencing_token,
                item.leased_by,
                item.lease_expires_at,
                item.attempt,
                item.retry_not_before_at,
                item.result_reference,
                item.result_sha256,
                item.created_at,
                item.updated_at,
            ),
        )

    @staticmethod
    def _same_plan(
        existing: DurablePerceptionWorkItem, expected: DurablePerceptionWorkItem
    ) -> bool:
        return (
            existing.work_item_id == expected.work_item_id
            and existing.work_logical_key == expected.work_logical_key
            and existing.run_key == expected.run_key
            and existing.context_key == expected.context_key
            and existing.focus_segment_ordinal == expected.focus_segment_ordinal
            and existing.derived_from_work_item_id == expected.derived_from_work_item_id
            and existing.stage is expected.stage
            and existing.input_sha256 == expected.input_sha256
            and existing.config_sha256 == expected.config_sha256
            and existing.max_attempts == expected.max_attempts
        )

    @staticmethod
    def _dependency(connection: sqlite3.Connection, downstream: str, upstream: str) -> None:
        row = connection.execute(
            """SELECT 1 FROM perception_vnext_work_dependencies
               WHERE downstream_work_item_id=? AND upstream_work_item_id=?""",
            (downstream, upstream),
        ).fetchone()
        if row is None:
            connection.execute(
                """INSERT INTO perception_vnext_work_dependencies(downstream_work_item_id,upstream_work_item_id)
                   VALUES(?,?)""",
                (downstream, upstream),
            )

    @staticmethod
    def _run_is_sealed(connection: sqlite3.Connection, run_key: str) -> bool:
        row = connection.execute(
            "SELECT derived_work_sealed FROM perception_vnext_runs WHERE run_key=?",
            (run_key,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown durable perception run: {run_key}")
        return bool(int(row["derived_work_sealed"]))

    def _refresh_ready(
        self, connection: sqlite3.Connection, run_key: str | None, current: datetime
    ) -> int:
        clauses = ["work.state=?"]
        parameters: list[object] = [DurablePerceptionWorkState.PLANNED.value]
        if run_key is not None:
            clauses.append("work.run_key=?")
            parameters.append(run_key)
        rows = connection.execute(
            f"""SELECT work.work_item_id FROM perception_vnext_work_items AS work
               JOIN perception_vnext_runs AS run ON run.run_key=work.run_key
               WHERE {" AND ".join(clauses)}
                 AND (work.stage != ? OR run.derived_work_sealed=1)
                 AND NOT EXISTS (
                     SELECT 1 FROM perception_vnext_work_dependencies AS edge
                     JOIN perception_vnext_work_items AS upstream ON upstream.work_item_id=edge.upstream_work_item_id
                     WHERE edge.downstream_work_item_id=work.work_item_id AND upstream.state != ?
                 )""",
            (
                *parameters,
                PerceptionStage.FINALIZE.value,
                DurablePerceptionWorkState.SUCCEEDED.value,
            ),
        ).fetchall()
        if not rows:
            return 0
        ids = tuple(str(row["work_item_id"]) for row in rows)
        placeholders = ",".join("?" for _ in ids)
        cursor = connection.execute(
            f"""UPDATE perception_vnext_work_items SET state=?,updated_at=?
                 WHERE work_item_id IN ({placeholders}) AND state=?""",
            (
                DurablePerceptionWorkState.READY.value,
                _timestamp(current),
                *ids,
                DurablePerceptionWorkState.PLANNED.value,
            ),
        )
        return cursor.rowcount

    def _recover_expired(self, connection: sqlite3.Connection, current: datetime) -> int:
        rows = connection.execute(
            """SELECT * FROM perception_vnext_work_items
               WHERE state IN (?,?) AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?""",
            (
                DurablePerceptionWorkState.LEASED.value,
                DurablePerceptionWorkState.RUNNING.value,
                _timestamp(current),
            ),
        ).fetchall()
        changed = 0
        for row in rows:
            item = self._item_from_row(row)
            next_state = (
                DurablePerceptionWorkState.READY
                if item.attempt < item.max_attempts
                else DurablePerceptionWorkState.FAILED_PERMANENT
            )
            cursor = connection.execute(
                """UPDATE perception_vnext_work_items
                   SET state=?,fencing_token=NULL,leased_by=NULL,lease_expires_at=NULL,
                       retry_not_before_at=NULL,updated_at=?
                   WHERE work_item_id=? AND lease_epoch=? AND fencing_token=? AND state IN (?,?)""",
                (
                    next_state.value,
                    _timestamp(current),
                    item.work_item_id,
                    item.lease_epoch,
                    item.fencing_token,
                    DurablePerceptionWorkState.LEASED.value,
                    DurablePerceptionWorkState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                continue
            connection.execute(
                """UPDATE perception_vnext_work_attempts
                   SET completed_at=?,outcome='ABANDONED',error_code='LEASE_EXPIRED',
                       error_detail='worker lease expired before terminal acceptance'
                   WHERE work_item_id=? AND lease_epoch=? AND fencing_token=? AND outcome='ACTIVE'""",
                (_timestamp(current), item.work_item_id, item.lease_epoch, item.fencing_token),
            )
            changed += 1
        return changed

    @staticmethod
    def _promote_retry_wait(connection: sqlite3.Connection, current: datetime) -> int:
        cursor = connection.execute(
            """UPDATE perception_vnext_work_items SET state=?,retry_not_before_at=NULL,updated_at=?
               WHERE state=? AND retry_not_before_at IS NOT NULL AND retry_not_before_at <= ?""",
            (
                DurablePerceptionWorkState.READY.value,
                _timestamp(current),
                DurablePerceptionWorkState.RETRY_WAIT.value,
                _timestamp(current),
            ),
        )
        return cursor.rowcount

    def _live_lease(
        self, connection: sqlite3.Connection, lease: DurablePerceptionWorkLease, current: datetime
    ) -> DurablePerceptionWorkItem:
        if not isinstance(lease, DurablePerceptionWorkLease):
            raise TypeError("lease must be DurablePerceptionWorkLease")
        item = self._load_item(connection, lease.work_item_id)
        if (
            item.lease_epoch != lease.lease_epoch
            or item.fencing_token != lease.fencing_token
            or item.leased_by != lease.worker_id
        ):
            raise DurablePerceptionWorkFenceError("lease no longer owns the current fence")
        if item.lease_expires_at is None or _parse_timestamp(item.lease_expires_at) <= current:
            raise DurablePerceptionWorkFenceError("lease has expired")
        return item

    @staticmethod
    def _finish_attempt(
        connection: sqlite3.Connection,
        lease: DurablePerceptionWorkLease,
        current: datetime,
        outcome: str,
        code: str | None,
        detail: str | None,
    ) -> None:
        cursor = connection.execute(
            """UPDATE perception_vnext_work_attempts
               SET completed_at=?,outcome=?,error_code=?,error_detail=?
               WHERE work_item_id=? AND lease_epoch=? AND fencing_token=? AND outcome='ACTIVE'""",
            (
                _timestamp(current),
                outcome,
                code,
                detail,
                lease.work_item_id,
                lease.lease_epoch,
                lease.fencing_token,
            ),
        )
        if cursor.rowcount != 1:
            raise DurablePerceptionWorkFenceError("attempt changed before terminal acceptance")

    def _find_stage(
        self,
        connection: sqlite3.Connection,
        run_key: str,
        context_key: str | None,
        stage: PerceptionStage,
    ) -> DurablePerceptionWorkItem:
        if context_key is None:
            row = connection.execute(
                """SELECT * FROM perception_vnext_work_items
                   WHERE run_key=? AND context_key IS NULL AND stage=?""",
                (run_key, stage.value),
            ).fetchone()
        else:
            row = connection.execute(
                """SELECT * FROM perception_vnext_work_items
                   WHERE run_key=? AND context_key=? AND stage=?""",
                (run_key, context_key, stage.value),
            ).fetchone()
        if row is None:
            raise DurablePerceptionSchedulerError("missing required upstream vNext stage")
        return self._item_from_row(row)

    def _load_item(
        self, connection: sqlite3.Connection, work_item_id: str
    ) -> DurablePerceptionWorkItem:
        row = connection.execute(
            "SELECT * FROM perception_vnext_work_items WHERE work_item_id=?", (work_item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown durable perception work: {work_item_id}")
        return self._item_from_row(row)

    def _load_run(self, connection: sqlite3.Connection, run_key: str) -> DurablePerceptionRun:
        row = connection.execute(
            "SELECT * FROM perception_vnext_runs WHERE run_key=?", (run_key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown durable perception run: {run_key}")
        return self._run_from_row(row)

    @staticmethod
    def _same_run_configuration(
        existing: DurablePerceptionRun, expected: DurablePerceptionRun
    ) -> bool:
        return (
            existing.run_key == expected.run_key
            and existing.plan_key == expected.plan_key
            and existing.plan_semantic_sha256 == expected.plan_semantic_sha256
            and existing.codec_policy_version == expected.codec_policy_version
            and existing.scheduler_policy_version == expected.scheduler_policy_version
            and existing.config_sha256 == expected.config_sha256
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> DurablePerceptionRun:
        return DurablePerceptionRun(
            run_key=str(row["run_key"]),
            plan_key=str(row["plan_key"]),
            plan_semantic_sha256=str(row["plan_semantic_sha256"]),
            codec_policy_version=str(row["codec_policy_version"]),
            scheduler_policy_version=str(row["scheduler_policy_version"]),
            config_sha256=str(row["config_sha256"]),
            created_at=str(row["created_at"]),
            derived_work_sealed=bool(int(row["derived_work_sealed"])),
        )

    @staticmethod
    def _context_from_row(row: sqlite3.Row) -> DurablePerceptionContext:
        return DurablePerceptionContext(
            run_key=str(row["run_key"]),
            focus_segment_ordinal=int(row["focus_segment_ordinal"]),
            segment_key=str(row["segment_key"]),
            segment_semantic_sha256=str(row["segment_semantic_sha256"]),
            context_key=str(row["context_key"]),
            context_semantic_sha256=str(row["context_semantic_sha256"]),
            interval_start_ns=int(row["interval_start_ns"]),
            interval_end_ns=int(row["interval_end_ns"]),
        )

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> DurablePerceptionWorkItem:
        return DurablePerceptionWorkItem(
            work_item_id=str(row["work_item_id"]),
            work_logical_key=str(row["work_logical_key"]),
            run_key=str(row["run_key"]),
            context_key=None if row["context_key"] is None else str(row["context_key"]),
            focus_segment_ordinal=None
            if row["focus_segment_ordinal"] is None
            else int(row["focus_segment_ordinal"]),
            derived_from_work_item_id=None
            if row["derived_from_work_item_id"] is None
            else str(row["derived_from_work_item_id"]),
            stage=PerceptionStage(str(row["stage"])),
            input_sha256=str(row["input_sha256"]),
            config_sha256=str(row["config_sha256"]),
            max_attempts=int(row["max_attempts"]),
            state=DurablePerceptionWorkState(str(row["state"])),
            lease_epoch=int(row["lease_epoch"]),
            fencing_token=None if row["fencing_token"] is None else str(row["fencing_token"]),
            leased_by=None if row["leased_by"] is None else str(row["leased_by"]),
            lease_expires_at=None
            if row["lease_expires_at"] is None
            else str(row["lease_expires_at"]),
            attempt=int(row["attempt"]),
            retry_not_before_at=None
            if row["retry_not_before_at"] is None
            else str(row["retry_not_before_at"]),
            result_reference=None
            if row["result_reference"] is None
            else str(row["result_reference"]),
            result_sha256=None if row["result_sha256"] is None else str(row["result_sha256"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            for statement in _SCHEMA:
                connection.execute(statement)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(perception_vnext_runs)").fetchall()
            }
            if "derived_work_sealed" not in columns:
                connection.execute(
                    "ALTER TABLE perception_vnext_runs ADD COLUMN derived_work_sealed INTEGER NOT NULL DEFAULT 0"
                )
            work_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(perception_vnext_work_items)"
                ).fetchall()
            }
            if "derived_from_work_item_id" not in work_columns:
                connection.execute(
                    "ALTER TABLE perception_vnext_work_items ADD COLUMN derived_from_work_item_id TEXT"
                )
            connection.commit()

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
        except sqlite3.Error as error:
            raise DurablePerceptionSchedulerError(
                f"SQLite durable perception transaction failed: {error}"
            ) from error
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._path)
        except sqlite3.Error as error:
            raise DurablePerceptionSchedulerError(
                "could not open durable perception SQLite database"
            ) from error
        connection.row_factory = sqlite3.Row
        return connection


def _now(value: datetime | None) -> datetime:
    current = datetime.now(UTC) if value is None else value
    if not isinstance(current, datetime):
        raise TypeError("now must be datetime or None")
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DurablePerceptionSchedulerError("stored timestamp has no timezone")
    return parsed.astimezone(UTC)


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "DURABLE_PERCEPTION_RUN_KEY_NAMESPACE",
    "DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION",
    "DURABLE_PERCEPTION_WORK_KEY_NAMESPACE",
    "DurablePerceptionContext",
    "DurablePerceptionRun",
    "DurablePerceptionRunSnapshot",
    "DurablePerceptionSchedulerError",
    "DurablePerceptionStageCount",
    "DurablePerceptionWorkClaim",
    "DurablePerceptionWorkConflict",
    "DurablePerceptionWorkFenceError",
    "DurablePerceptionWorkItem",
    "DurablePerceptionWorkLease",
    "DurablePerceptionWorkState",
    "DurablePerceptionWorkStateError",
    "SQLitePerceptionWorkScheduler",
]
