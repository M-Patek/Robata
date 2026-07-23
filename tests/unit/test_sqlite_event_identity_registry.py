from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from uuid import NAMESPACE_URL, uuid5

import pytest

import robata.adapters.sqlite_event_identity_registry as sqlite_registry_module
from robata.adapters.sqlite_event_identity_registry import (
    SQLiteEventIdentityRegistryError,
    SQLiteEventIdentityRegistryRepository,
    SQLiteEventIdentityRegistryUncertainCommitError,
)
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRegistryError
from robata.event_pipeline.identity_registry import (
    CrossRecordingEventIdentityError,
    EventCurrentRevisionReference,
    EventIdentityAssignment,
    EventIdentityAssignmentDisposition,
    EventIdentityAssignmentRelation,
    EventIdentityCandidate,
    EventIdentityCandidateRelation,
    EventIdentityOutboxRecord,
    EventIdentityPolicyRef,
    EventIdentityRegistryMutation,
    EventIdentityRegistryService,
    EventIdentityRelation,
    EventIdentityResolution,
    EventRegistrySnapshot,
    StaleEventRegistryFenceError,
    event_identity_assignment_semantic_projection,
    event_identity_relation_semantic_projection,
)
from robata.runtime.observability import (
    RuntimeObserver,
    RuntimeProfileRecorder,
    RuntimeProfileSnapshot,
)
from tests.unit.test_event_identity_registry import (
    NOW,
    OUTPUT_ADMISSION_POLICY,
    _context,
    _digest,
    _enriched_output,
    _hypothesis,
    _SequenceAllocator,
    _service,
    _uuid,
)


def _database(tmp_path: Path) -> Path:
    return tmp_path / "event-identities.sqlite3"


def _operation_counter_value(
    snapshot: RuntimeProfileSnapshot,
    counter_name: str,
    operation: str,
) -> int:
    return sum(
        counter.value
        for counter in snapshot.counters
        if counter.name == counter_name
        and any(
            attribute.name == "operation" and attribute.value == operation
            for attribute in counter.attributes
        )
    )


def test_initialization_observes_exact_event_identity_transaction(
    tmp_path: Path,
) -> None:
    recorder = RuntimeProfileRecorder()

    SQLiteEventIdentityRegistryRepository(
        _database(tmp_path),
        runtime_observer=recorder,
    )

    snapshot = recorder.snapshot()
    domain = "sqlite.event_identity"
    assert (
        _operation_counter_value(
            snapshot,
            f"{domain}.transactions",
            "initialize_schema",
        )
        == 1
    )
    assert (
        _operation_counter_value(
            snapshot,
            f"{domain}.commits",
            "initialize_schema",
        )
        == 1
    )
    assert (
        _operation_counter_value(
            snapshot,
            f"{domain}.rollbacks",
            "initialize_schema",
        )
        == 0
    )
    assert sum(span.name == f"{domain}.transaction" for span in snapshot.spans) == 1


def _assign_one(
    repository: SQLiteEventIdentityRegistryRepository,
    *,
    fingerprint: str,
):
    context = _context()
    output = _enriched_output(context)
    hypothesis = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint=fingerprint,
        ordinal=0,
    )
    result = _service(repository, _SequenceAllocator()).assign_batch(
        admitted_context=context,
        hypotheses=(hypothesis,),
        enriched_outputs=(output,),
        decided_at=NOW,
    )
    return context, result


def _rebuild_assignment(
    assignment: EventIdentityAssignment,
    **updates: object,
) -> EventIdentityAssignment:
    values = assignment.model_dump(mode="python")
    values.update(updates)
    provisional = EventIdentityAssignment.model_construct(**values)
    digest = semantic_sha256(event_identity_assignment_semantic_projection(provisional))
    values.update(
        assignment_id=str(uuid5(NAMESPACE_URL, f"robata:event-identity-assignment:{digest}")),
        assignment_logical_key=f"event-identity-assignment:{digest}",
        assignment_semantic_sha256=digest,
    )
    return EventIdentityAssignment.model_validate(values, strict=True)


def _rebuild_relation(
    relation: EventIdentityRelation,
    **updates: object,
) -> EventIdentityRelation:
    values = relation.model_dump(mode="python")
    values.update(updates)
    provisional = EventIdentityRelation.model_construct(**values)
    digest = semantic_sha256(event_identity_relation_semantic_projection(provisional))
    values["relation_logical_key"] = f"event-identity-relation:{digest}"
    return EventIdentityRelation.model_validate(values, strict=True)


def _outbox_for_assignment(
    assignment: EventIdentityAssignment,
) -> EventIdentityOutboxRecord:
    return EventIdentityOutboxRecord(
        schema_version="1.0",
        outbox_id=str(
            uuid5(
                NAMESPACE_URL,
                f"robata:event-identity-outbox:{assignment.assignment_logical_key}",
            )
        ),
        topic="event.identity.assignment",
        recording_identity=assignment.recording_identity,
        key=assignment.recording_identity,
        assignment_logical_key=assignment.assignment_logical_key,
        payload_reference=assignment.assignment_logical_key,
        registry_generation=assignment.registry_generation,
    )


class _SplitResolver:
    def __init__(self) -> None:
        self._policy = EventIdentityPolicyRef(
            version="split-fixture-v1",
            semantic_sha256=_digest("split-fixture-v1"),
        )

    @property
    def policy(self) -> EventIdentityPolicyRef:
        return self._policy

    def resolve(
        self, *, snapshot: EventRegistrySnapshot, **_kwargs: object
    ) -> EventIdentityResolution:
        if not snapshot.identities:
            return EventIdentityResolution(
                disposition=EventIdentityAssignmentDisposition.CREATED,
                selected_event_id=None,
                candidates=(),
                reason="first identity in the fixture",
            )
        return EventIdentityResolution(
            disposition=EventIdentityAssignmentDisposition.CREATED,
            selected_event_id=None,
            candidates=(
                EventIdentityCandidate(
                    event_id=snapshot.identities[0].event_id,
                    score=0.9,
                    relation=EventIdentityCandidateRelation.SPLIT_FROM,
                    reason="fixture split lineage",
                ),
            ),
            reason="fixture created a split identity",
        )


class _MutationCaptured(RuntimeError):
    pass


class _CapturingRepository:
    def __init__(self, delegate: SQLiteEventIdentityRegistryRepository) -> None:
        self._delegate = delegate
        self.mutation: EventIdentityRegistryMutation | None = None

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        return self._delegate.snapshot(recording_identity)

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        self.mutation = mutation
        raise _MutationCaptured


class _SynchronizedSQLiteRepository(SQLiteEventIdentityRegistryRepository):
    def __init__(self, database_path: Path) -> None:
        super().__init__(database_path)
        self._initial_reads = Barrier(2)
        self._counter_lock = Lock()
        self.stale_commits = 0

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        snapshot = super().snapshot(recording_identity)
        if snapshot.generation == 0:
            self._initial_reads.wait(timeout=5)
        return snapshot

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        try:
            return super().commit(mutation)
        except StaleEventRegistryFenceError:
            with self._counter_lock:
                self.stale_commits += 1
            raise


class _FailingCommitSQLiteRepository(SQLiteEventIdentityRegistryRepository):
    def __init__(
        self,
        database_path: Path,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        super().__init__(database_path, runtime_observer=runtime_observer)
        self._inside_mutation = False
        self._fail_once = True

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        self._inside_mutation = True
        try:
            return super().commit(mutation)
        finally:
            self._inside_mutation = False

    def _commit(self, connection: sqlite3.Connection) -> None:
        if self._inside_mutation and self._fail_once:
            self._fail_once = False
            raise sqlite3.OperationalError("injected durable commit failure")
        super()._commit(connection)


class _CommitThenFailSQLiteRepository(SQLiteEventIdentityRegistryRepository):
    def __init__(
        self,
        database_path: Path,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        super().__init__(database_path, runtime_observer=runtime_observer)
        self._inside_mutation = False
        self._fail_once = True

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        self._inside_mutation = True
        try:
            return super().commit(mutation)
        finally:
            self._inside_mutation = False

    def _commit(self, connection: sqlite3.Connection) -> None:
        super()._commit(connection)
        if self._inside_mutation and self._fail_once:
            self._fail_once = False
            raise sqlite3.OperationalError("injected error after durable commit")


class _CommitThenChangeCurrentRevisionSQLiteRepository(SQLiteEventIdentityRegistryRepository):
    def __init__(self, database_path: Path) -> None:
        super().__init__(database_path)
        self._inside_mutation = False
        self._fail_once = True
        self.current_revision: EventCurrentRevisionReference | None = None

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        self._inside_mutation = True
        try:
            return super().commit(mutation)
        finally:
            self._inside_mutation = False

    def _commit(self, connection: sqlite3.Connection) -> None:
        super()._commit(connection)
        if not self._inside_mutation or not self._fail_once:
            return
        self._fail_once = False

        external = sqlite3.connect(self.database_path)
        try:
            external.execute("PRAGMA foreign_keys = ON")
            row = external.execute(
                """
                SELECT recording_identity, event_id
                FROM stable_event_identities
                ORDER BY recording_identity, event_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                raise AssertionError("durable identity fixture was not committed")
            revision = EventCurrentRevisionReference(
                recording_identity=row[0],
                event_id=row[1],
                revision_logical_key=(
                    f"action-event-revision:{_digest('concurrent-current-revision')}"
                ),
                revision_semantic_sha256=_digest("concurrent-current-revision-semantic"),
                effective_interval=NanosecondInterval(start_ns=10, end_ns=20),
            )
            external.execute(
                """
                INSERT INTO event_current_revisions (
                    recording_identity, event_id, payload_json
                ) VALUES (?, ?, ?)
                """,
                (
                    revision.recording_identity,
                    revision.event_id,
                    sqlite3.Binary(canonical_json_bytes(revision)),
                ),
            )
            external.commit()
            self.current_revision = revision
        finally:
            external.close()
        raise sqlite3.OperationalError("injected error after commit and current-revision advance")


def test_new_empty_database_sets_exact_sqlite_identity_and_version(tmp_path: Path) -> None:
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))
    connection = sqlite3.connect(repository.database_path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] != 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_v1_domain_outbox_payload_is_atomically_upcast_to_exact_wire(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    repository = SQLiteEventIdentityRegistryRepository(database_path)
    context, result = _assign_one(repository, fingerprint="v1-outbox-migration-fixture")
    assert len(result.outbox) == 1

    connection = sqlite3.connect(database_path)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'event_identity_outbox_no_update'"
        ).fetchone()[0]
        row = connection.execute(
            "SELECT outbox_id, payload_json FROM event_identity_outbox"
        ).fetchone()
        document = json.loads(bytes(row[1]))
        assert document.pop("schema_ref")
        legacy_payload = canonical_json_bytes(document)
        connection.execute("DROP TRIGGER event_identity_outbox_no_update")
        connection.execute(
            "UPDATE event_identity_outbox SET payload_json = ? WHERE outbox_id = ?",
            (sqlite3.Binary(legacy_payload), row[0]),
        )
        connection.execute(trigger_sql)
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    reopened = SQLiteEventIdentityRegistryRepository(database_path)
    assert reopened.list_outbox(context.recording_identity) == result.outbox
    connection = sqlite3.connect(database_path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        stored = json.loads(
            bytes(
                connection.execute("SELECT payload_json FROM event_identity_outbox").fetchone()[0]
            )
        )
    finally:
        connection.close()
    assert version == 2
    assert set(stored["schema_ref"]) == {"schema_id", "version", "artifact_id", "sha256"}


def test_write_side_schema_registry_failure_maps_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))

    def reject_exact_wire(*_args: object, **_kwargs: object) -> None:
        raise SchemaRegistryError("injected exact-schema failure")

    monkeypatch.setattr(
        sqlite_registry_module,
        "validate_registered_event_identity_outbox_wire_record",
        reject_exact_wire,
    )
    with pytest.raises(
        SQLiteEventIdentityRegistryError,
        match="before persistence",
    ):
        _assign_one(repository, fingerprint="write-schema-failure-fixture")

    snapshot = repository.snapshot(_context().recording_identity)
    assert snapshot.generation == 0
    assert snapshot.identities == ()
    assert snapshot.assignments == ()


def test_existing_repository_rejects_external_wal_mode_downgrade(tmp_path: Path) -> None:
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))
    connection = sqlite3.connect(repository.database_path)
    try:
        assert connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match="WAL mode"):
        repository.snapshot(_digest("wal-mode-fixture"))


def test_nonempty_unversioned_database_is_rejected_without_adoption(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE unrelated_records (value TEXT NOT NULL)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match="refusing to adopt"):
        SQLiteEventIdentityRegistryRepository(database_path)

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name"
        ).fetchall() == [("unrelated_records",)]
    finally:
        connection.close()


def test_existing_database_with_wrong_application_identity_is_rejected(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    SQLiteEventIdentityRegistryRepository(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA application_id = 0")
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match="application identity"):
        SQLiteEventIdentityRegistryRepository(database_path)


@pytest.mark.parametrize(
    "statement",
    (
        "DROP INDEX event_identity_outbox_recording_idx",
        "DROP TRIGGER event_identity_outbox_no_delete",
    ),
)
def test_existing_database_with_schema_drift_is_rejected(
    tmp_path: Path,
    statement: str,
) -> None:
    database_path = _database(tmp_path)
    SQLiteEventIdentityRegistryRepository(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match="canonical schema"):
        SQLiteEventIdentityRegistryRepository(database_path)


def test_restart_preserves_identities_assignments_relations_outbox_and_replay(
    tmp_path: Path,
) -> None:
    context = _context()
    output = _enriched_output(context)
    first_hypothesis = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="parent-event",
        ordinal=0,
    )
    second_hypothesis = _hypothesis(
        context,
        output,
        start_ns=30,
        end_ns=40,
        fingerprint="split-event",
        ordinal=1,
    )
    database_path = _database(tmp_path)
    repository = SQLiteEventIdentityRegistryRepository(database_path)
    service = EventIdentityRegistryService(
        repository=repository,
        resolver=_SplitResolver(),
        allocator=_SequenceAllocator(),
        output_admission_policy=OUTPUT_ADMISSION_POLICY,
    )

    service.assign_batch(
        admitted_context=context,
        hypotheses=(first_hypothesis,),
        enriched_outputs=(output,),
        decided_at=NOW,
    )
    second = service.assign_batch(
        admitted_context=context,
        hypotheses=(second_hypothesis,),
        enriched_outputs=(output,),
        decided_at=NOW,
    )
    assert len(second.relations) == 1

    reopened = SQLiteEventIdentityRegistryRepository(database_path)
    snapshot = reopened.snapshot(context.recording_identity)
    assert snapshot.generation == 2
    assert snapshot.fence == 3
    assert len(snapshot.identities) == 2
    assert len(snapshot.assignments) == 2
    assert reopened.list_relations(context.recording_identity) == second.relations
    assert len(reopened.list_outbox(context.recording_identity)) == 2

    replay_service = EventIdentityRegistryService(
        repository=reopened,
        resolver=_SplitResolver(),
        allocator=_SequenceAllocator(start=20_000),
        output_admission_policy=OUTPUT_ADMISSION_POLICY,
    )
    replay = replay_service.assign_batch(
        admitted_context=context,
        hypotheses=(second_hypothesis,),
        enriched_outputs=(output,),
        decided_at="2026-07-20T00:00:00Z",
    )
    assert replay.initial_generation == replay.final_generation == 2
    assert replay.assignments == second.assignments
    assert replay.new_identities == ()
    assert replay.outbox == ()


def test_concurrent_same_recording_commits_retry_exact_cas_and_converge(tmp_path: Path) -> None:
    context = _context()
    output = _enriched_output(context)
    hypotheses = (
        _hypothesis(
            context,
            output,
            start_ns=10,
            end_ns=20,
            fingerprint="concurrent-event",
            ordinal=0,
        ),
        _hypothesis(
            context,
            output,
            start_ns=30,
            end_ns=40,
            fingerprint="concurrent-event",
            ordinal=1,
        ),
    )
    repository = _SynchronizedSQLiteRepository(_database(tmp_path))
    service = _service(repository, _SequenceAllocator())

    def assign(index: int):
        return service.assign_batch(
            admitted_context=context,
            hypotheses=(hypotheses[index],),
            enriched_outputs=(output,),
            decided_at=NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(assign, (0, 1)))

    snapshot = repository.snapshot(context.recording_identity)
    assert repository.stale_commits == 1
    assert snapshot.generation == 2
    assert len(snapshot.identities) == 1
    assert len(snapshot.assignments) == 2
    assert {item.assignments[0].event_id for item in results} == {snapshot.identities[0].event_id}
    assert len(repository.list_outbox(context.recording_identity)) == 2


def test_event_id_ownership_is_global_and_survives_restart(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    fixed_event_id = _uuid(50_000)
    first_context = _context("sqlite-recording-a")
    first_output = _enriched_output(first_context)
    first_hypothesis = _hypothesis(
        first_context,
        first_output,
        start_ns=10,
        end_ns=20,
        fingerprint="event-a",
        ordinal=0,
    )
    first_repository = SQLiteEventIdentityRegistryRepository(database_path)
    _service(first_repository, _SequenceAllocator(fixed=fixed_event_id)).assign_batch(
        admitted_context=first_context,
        hypotheses=(first_hypothesis,),
        enriched_outputs=(first_output,),
        decided_at=NOW,
    )

    second_context = _context("sqlite-recording-b")
    second_output = _enriched_output(second_context)
    second_hypothesis = _hypothesis(
        second_context,
        second_output,
        start_ns=10,
        end_ns=20,
        fingerprint="event-b",
        ordinal=0,
    )
    reopened = SQLiteEventIdentityRegistryRepository(database_path)
    with pytest.raises(CrossRecordingEventIdentityError, match="another recording"):
        _service(reopened, _SequenceAllocator(fixed=fixed_event_id)).assign_batch(
            admitted_context=second_context,
            hypotheses=(second_hypothesis,),
            enriched_outputs=(second_output,),
            decided_at=NOW,
        )

    assert reopened.snapshot(first_context.recording_identity).generation == 1
    assert reopened.snapshot(second_context.recording_identity).generation == 0
    assert reopened.list_outbox(second_context.recording_identity) == ()


def test_failed_sqlite_commit_rolls_back_every_mutation_row(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    recorder = RuntimeProfileRecorder()
    context = _context()
    output = _enriched_output(context)
    hypothesis = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="rolled-back-event",
        ordinal=0,
    )
    repository = _FailingCommitSQLiteRepository(
        database_path,
        runtime_observer=recorder,
    )

    with pytest.raises(
        SQLiteEventIdentityRegistryUncertainCommitError,
        match="exact expected mutation",
    ):
        _service(repository, _SequenceAllocator()).assign_batch(
            admitted_context=context,
            hypotheses=(hypothesis,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )

    observation = recorder.snapshot()
    domain = "sqlite.event_identity"
    assert _operation_counter_value(observation, f"{domain}.transactions", "commit") == 1
    assert _operation_counter_value(observation, f"{domain}.commits", "commit") == 0
    assert _operation_counter_value(observation, f"{domain}.rollbacks", "commit") == 1
    assert (
        _operation_counter_value(
            observation,
            f"{domain}.transactions",
            "recover_uncertain_commit",
        )
        == 1
    )
    assert (
        _operation_counter_value(
            observation,
            f"{domain}.commits",
            "recover_uncertain_commit",
        )
        == 1
    )

    reopened = SQLiteEventIdentityRegistryRepository(database_path)
    snapshot = reopened.snapshot(context.recording_identity)
    assert snapshot.generation == 0
    assert snapshot.fence == 1
    assert snapshot.identities == ()
    assert snapshot.assignments == ()
    assert reopened.list_relations(context.recording_identity) == ()
    assert reopened.list_outbox(context.recording_identity) == ()


def test_error_after_durable_commit_reconciles_the_exact_mutation(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    recorder = RuntimeProfileRecorder()
    repository = _CommitThenFailSQLiteRepository(
        database_path,
        runtime_observer=recorder,
    )

    context, result = _assign_one(repository, fingerprint="durably-committed-event")

    assert result.initial_generation == 0
    assert result.final_generation == 1
    assert result.fence == 2
    assert len(result.assignments) == 1
    assert len(result.new_identities) == 1
    assert len(result.outbox) == 1
    observation = recorder.snapshot()
    domain = "sqlite.event_identity"
    assert _operation_counter_value(observation, f"{domain}.transactions", "commit") == 1
    assert _operation_counter_value(observation, f"{domain}.commits", "commit") == 0
    assert _operation_counter_value(observation, f"{domain}.rollbacks", "commit") == 0
    assert (
        _operation_counter_value(
            observation,
            f"{domain}.transactions",
            "recover_uncertain_commit",
        )
        == 1
    )
    assert (
        _operation_counter_value(
            observation,
            f"{domain}.commits",
            "recover_uncertain_commit",
        )
        == 1
    )

    reopened = SQLiteEventIdentityRegistryRepository(database_path)
    snapshot = reopened.snapshot(context.recording_identity)
    assert snapshot.generation == 1
    assert snapshot.fence == 2
    assert snapshot.assignments == result.assignments
    assert snapshot.identities == result.new_identities
    assert reopened.list_relations(context.recording_identity) == ()
    assert reopened.list_outbox(context.recording_identity) == result.outbox


def test_uncertain_reconciliation_ignores_legal_current_revision_advance(
    tmp_path: Path,
) -> None:
    database_path = _database(tmp_path)
    repository = _CommitThenChangeCurrentRevisionSQLiteRepository(database_path)

    context, result = _assign_one(repository, fingerprint="revision-race-event")

    assert result.final_generation == 1
    assert result.fence == 2
    assert repository.current_revision is not None
    snapshot = repository.snapshot(context.recording_identity)
    assert snapshot.current_revisions == (repository.current_revision,)
    assert snapshot.identities == result.new_identities
    assert snapshot.assignments == result.assignments
    assert repository.list_outbox(context.recording_identity) == result.outbox


def test_reopen_rejects_orphaned_current_revision_projection(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    repository = SQLiteEventIdentityRegistryRepository(database_path)
    context, _result = _assign_one(repository, fingerprint="foreign-key-fixture")
    orphan = EventCurrentRevisionReference(
        recording_identity=context.recording_identity,
        event_id=_uuid(999_999),
        revision_logical_key=f"action-event-revision:{_digest('orphan-revision')}",
        revision_semantic_sha256=_digest("orphan-revision-semantic"),
        effective_interval=NanosecondInterval(start_ns=1, end_ns=2),
    )

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        connection.execute(
            """
            INSERT INTO event_current_revisions (
                recording_identity, event_id, payload_json
            ) VALUES (?, ?, ?)
            """,
            (
                orphan.recording_identity,
                orphan.event_id,
                sqlite3.Binary(canonical_json_bytes(orphan)),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match="foreign-key"):
        SQLiteEventIdentityRegistryRepository(database_path)


def test_reopen_rejects_outbox_with_forged_exact_schema_pin(tmp_path: Path) -> None:
    database_path = _database(tmp_path)
    repository = SQLiteEventIdentityRegistryRepository(database_path)
    _assign_one(repository, fingerprint="outbox-schema-pin-fixture")

    connection = sqlite3.connect(database_path)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'event_identity_outbox_no_update'"
        ).fetchone()[0]
        payload = json.loads(
            bytes(
                connection.execute(
                    "SELECT payload_json FROM event_identity_outbox LIMIT 1"
                ).fetchone()[0]
            )
        )
        payload["schema_ref"]["sha256"] = "0" * 64
        connection.execute("DROP TRIGGER event_identity_outbox_no_update")
        connection.execute(
            "UPDATE event_identity_outbox SET payload_json = ?",
            (sqlite3.Binary(canonical_json_bytes(payload)),),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match="exact schema validation"):
        SQLiteEventIdentityRegistryRepository(database_path)


@pytest.mark.parametrize(
    ("generation", "fence", "message"),
    (
        (0, 1, "generation zero"),
        (2, 3, "complete partition history"),
    ),
)
def test_reopen_rejects_tampered_partition_generation_closure(
    tmp_path: Path,
    generation: int,
    fence: int,
    message: str,
) -> None:
    database_path = _database(tmp_path)
    repository = SQLiteEventIdentityRegistryRepository(database_path)
    context, _result = _assign_one(repository, fingerprint="generation-fixture")

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            UPDATE event_registry_partitions
            SET generation = ?, fence = ?
            WHERE recording_identity = ?
            """,
            (generation, fence, context.recording_identity),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SQLiteEventIdentityRegistryError, match=message):
        SQLiteEventIdentityRegistryRepository(database_path)


@pytest.mark.parametrize("alteration", ("reversed-endpoints", "mismatched-metadata"))
def test_commit_rejects_domain_valid_relation_outside_exact_candidate_closure(
    tmp_path: Path,
    alteration: str,
) -> None:
    context = _context()
    output = _enriched_output(context)
    hypotheses = (
        _hypothesis(
            context,
            output,
            start_ns=10,
            end_ns=20,
            fingerprint="closure-parent",
            ordinal=0,
        ),
        _hypothesis(
            context,
            output,
            start_ns=30,
            end_ns=40,
            fingerprint="closure-child",
            ordinal=1,
        ),
    )
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))
    first_service = EventIdentityRegistryService(
        repository=repository,
        resolver=_SplitResolver(),
        allocator=_SequenceAllocator(),
        output_admission_policy=OUTPUT_ADMISSION_POLICY,
    )
    first_service.assign_batch(
        admitted_context=context,
        hypotheses=(hypotheses[0],),
        enriched_outputs=(output,),
        decided_at=NOW,
    )

    capturing = _CapturingRepository(repository)
    second_service = EventIdentityRegistryService(
        repository=capturing,
        resolver=_SplitResolver(),
        allocator=_SequenceAllocator(start=20_000),
        output_admission_policy=OUTPUT_ADMISSION_POLICY,
    )
    with pytest.raises(_MutationCaptured):
        second_service.assign_batch(
            admitted_context=context,
            hypotheses=(hypotheses[1],),
            enriched_outputs=(output,),
            decided_at=NOW,
        )
    mutation = capturing.mutation
    assert mutation is not None
    assert len(mutation.relations) == 1
    relation = mutation.relations[0]
    if alteration == "reversed-endpoints":
        invalid_relation = _rebuild_relation(
            relation,
            from_event_id=relation.to_event_id,
            to_event_id=relation.from_event_id,
        )
    else:
        invalid_relation = _rebuild_relation(
            relation,
            score=0.8,
            reason="metadata no longer matches the assignment candidate",
            identity_policy_version="mismatched-relation-v1",
            identity_policy_sha256=_digest("mismatched-relation-v1"),
        )
    invalid_mutation = EventIdentityRegistryMutation.model_validate(
        mutation.model_copy(update={"relations": (invalid_relation,)}).model_dump(mode="python"),
        strict=True,
    )

    with pytest.raises(
        SQLiteEventIdentityRegistryError,
        match="exactly match assignment candidates",
    ):
        repository.commit(invalid_mutation)

    snapshot = repository.snapshot(context.recording_identity)
    assert snapshot.generation == 1
    assert len(snapshot.identities) == 1
    assert len(snapshot.assignments) == 1
    assert repository.list_relations(context.recording_identity) == ()
    assert len(repository.list_outbox(context.recording_identity)) == 1


def test_commit_rejects_created_assignment_that_points_to_an_older_identity(
    tmp_path: Path,
) -> None:
    context = _context()
    output = _enriched_output(context)
    hypotheses = (
        _hypothesis(
            context,
            output,
            start_ns=10,
            end_ns=20,
            fingerprint="reused-identity",
            ordinal=0,
        ),
        _hypothesis(
            context,
            output,
            start_ns=30,
            end_ns=40,
            fingerprint="reused-identity",
            ordinal=1,
        ),
    )
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))
    _service(repository, _SequenceAllocator()).assign_batch(
        admitted_context=context,
        hypotheses=(hypotheses[0],),
        enriched_outputs=(output,),
        decided_at=NOW,
    )

    capturing = _CapturingRepository(repository)
    with pytest.raises(_MutationCaptured):
        _service(capturing, _SequenceAllocator(start=20_000)).assign_batch(
            admitted_context=context,
            hypotheses=(hypotheses[1],),
            enriched_outputs=(output,),
            decided_at=NOW,
        )
    mutation = capturing.mutation
    assert mutation is not None
    assert mutation.identities == ()
    assert len(mutation.assignments) == 1
    invalid_assignment = _rebuild_assignment(
        mutation.assignments[0],
        disposition=EventIdentityAssignmentDisposition.CREATED,
        relation=(EventIdentityAssignmentRelation.NEW_IDENTITY,),
        candidates=(),
        reason="claims creation while retaining an older identity",
    )
    invalid_mutation = EventIdentityRegistryMutation(
        schema_version="1.0",
        recording_identity=mutation.recording_identity,
        expected_generation=mutation.expected_generation,
        fence=mutation.fence,
        next_generation=mutation.next_generation,
        identities=(),
        assignments=(invalid_assignment,),
        relations=(),
        outbox=(_outbox_for_assignment(invalid_assignment),),
    )

    with pytest.raises(
        SQLiteEventIdentityRegistryError,
        match="does not create its exact identity",
    ):
        repository.commit(invalid_mutation)

    snapshot = repository.snapshot(context.recording_identity)
    assert snapshot.generation == 1
    assert len(snapshot.identities) == 1
    assert len(snapshot.assignments) == 1
    assert repository.list_relations(context.recording_identity) == ()
    assert len(repository.list_outbox(context.recording_identity)) == 1


@pytest.mark.parametrize(
    "statement",
    (
        "UPDATE stable_event_identities SET created_generation = created_generation",
        "DELETE FROM event_identity_assignments",
        "UPDATE event_identity_outbox SET registry_generation = registry_generation",
        "INSERT OR REPLACE INTO stable_event_identities SELECT * FROM stable_event_identities",
        "INSERT OR REPLACE INTO event_identity_assignments "
        "SELECT * FROM event_identity_assignments",
        "INSERT OR REPLACE INTO event_identity_outbox SELECT * FROM event_identity_outbox",
    ),
)
def test_sqlite_schema_rejects_updates_and_deletes_of_append_only_rows(
    tmp_path: Path,
    statement: str,
) -> None:
    context = _context()
    output = _enriched_output(context)
    hypothesis = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="append-only-event",
        ordinal=0,
    )
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))
    _service(repository, _SequenceAllocator()).assign_batch(
        admitted_context=context,
        hypotheses=(hypothesis,),
        enriched_outputs=(output,),
        decided_at=NOW,
    )

    connection = sqlite3.connect(repository.database_path)
    try:
        assert connection.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(statement)
    finally:
        connection.rollback()
        connection.close()

    snapshot = repository.snapshot(context.recording_identity)
    assert snapshot.generation == 1
    assert len(snapshot.identities) == 1
    assert len(snapshot.assignments) == 1
    assert len(repository.list_outbox(context.recording_identity)) == 1


def test_sqlite_schema_rejects_relation_replace_without_recursive_triggers(
    tmp_path: Path,
) -> None:
    context = _context()
    output = _enriched_output(context)
    hypotheses = (
        _hypothesis(
            context,
            output,
            start_ns=10,
            end_ns=20,
            fingerprint="relation-parent",
            ordinal=0,
        ),
        _hypothesis(
            context,
            output,
            start_ns=30,
            end_ns=40,
            fingerprint="relation-child",
            ordinal=1,
        ),
    )
    repository = SQLiteEventIdentityRegistryRepository(_database(tmp_path))
    service = EventIdentityRegistryService(
        repository=repository,
        resolver=_SplitResolver(),
        allocator=_SequenceAllocator(),
        output_admission_policy=OUTPUT_ADMISSION_POLICY,
    )
    for hypothesis in hypotheses:
        service.assign_batch(
            admitted_context=context,
            hypotheses=(hypothesis,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )
    before = repository.list_relations(context.recording_identity)
    assert len(before) == 1

    connection = sqlite3.connect(repository.database_path)
    try:
        assert connection.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "INSERT OR REPLACE INTO event_identity_relations "
                "SELECT * FROM event_identity_relations"
            )
    finally:
        connection.rollback()
        connection.close()

    assert repository.list_relations(context.recording_identity) == before
