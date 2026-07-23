from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from robata.application.canonical.local_composition import CanonicalLocalRunReceipt
from robata.application.canonical.local_outbox_delivery import (
    LocalOutboxDeliveryOutcome,
    LocalOutboxDeliverySummary,
)
from robata.application.canonical.local_review_routing import LocalReviewRoutingSummary
from robata.contracts.hashing import canonical_json_bytes
from robata.review.routing import ReviewRoutingDisposition
from robata.runtime import canonical_profile as profile_module
from robata.runtime.canonical_profile import (
    CanonicalProfileManifest,
    CanonicalProfilePolicyFacts,
    CanonicalProfileReport,
    CanonicalProfileRunError,
    ProfileFileFact,
    ProfileGitFacts,
    ProfileRuntimeFacts,
    StateFileClass,
    build_canonical_profile_manifest,
    build_profile_reconciliation,
    discover_canonical_profile_durations,
    snapshot_state_tree,
    snapshot_work_queue,
    unique_runtime_counter_value,
)
from robata.runtime.observability import (
    RuntimeAttribute,
    RuntimeCounterSnapshot,
    RuntimeProfileSnapshot,
    RuntimeResourceMeasurement,
    RuntimeResourceSnapshot,
    RuntimeResourceStatus,
    RuntimeSpanSnapshot,
    RuntimeSpanStatus,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _policies() -> CanonicalProfilePolicyFacts:
    return CanonicalProfilePolicyFacts(
        composition_version="composition-v1",
        pipeline_version="pipeline-v1",
        execution_policy_semantic_sha256=_digest("execution"),
        runtime_policy_semantic_sha256=_digest("runtime"),
        input_planner_version="planner-v1",
        parser_version="parser-v1",
        inference_policy_versions=("coarse-v1", "dense-v1"),
    )


def _runtime() -> ProfileRuntimeFacts:
    return ProfileRuntimeFacts(
        python_version="3.13.5",
        python_implementation="CPython",
        platform="test-platform",
        machine="test-machine",
        logical_cpu_count=8,
    )


def _manifest(*, run_key: str = "profile-test") -> CanonicalProfileManifest:
    return CanonicalProfileManifest.create(
        source=ProfileFileFact(sha256=_digest("source"), byte_count=10),
        mapping_config=ProfileFileFact(sha256=_digest("mapping"), byte_count=11),
        uv_lock=ProfileFileFact(sha256=_digest("lock"), byte_count=12),
        schema_catalog=ProfileFileFact(sha256=_digest("catalog"), byte_count=13),
        git=ProfileGitFacts(head_commit="a" * 40, dirty=False),
        runtime=_runtime(),
        policies=_policies(),
        run_key=run_key,
        max_duration_ns=180_000_000_000,
        allow_unapproved_profile=True,
    )


def _unsupported_resource() -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(status=RuntimeResourceStatus.UNSUPPORTED)


def _observer() -> RuntimeProfileSnapshot:
    return RuntimeProfileSnapshot(
        elapsed_ns=100,
        process_cpu_ns=50,
        resources=RuntimeResourceSnapshot(
            rss_bytes=_unsupported_resource(),
            read_bytes_delta=_unsupported_resource(),
            write_bytes_delta=_unsupported_resource(),
        ),
    )


def _empty_state():
    return snapshot_state_tree(Path("this-profile-state-does-not-exist"))


def _receipt(*, replayed: bool = False) -> CanonicalLocalRunReceipt:
    return CanonicalLocalRunReceipt(
        schema_version="1.0",
        model_version="canonical-local-run-receipt-v4",
        ok=True,
        run_id="run-1",
        recording_identity="recording-1",
        status="NO_EVENTS",
        command_sha256=_digest("command"),
        completion_semantic_sha256=_digest("completion"),
        event_ids=(),
        revision_ids=(),
        outbox_ids=(),
        outbox_count=0,
        outbox_delivery=LocalOutboxDeliverySummary(
            model_version="canonical-local-outbox-delivery-v1",
            outcome=LocalOutboxDeliveryOutcome.NOT_APPLICABLE,
            outbox_ids=(),
            relay_attempt_count=0,
            pending_count=0,
            leased_count=0,
            retry_wait_count=0,
            delivered_count=0,
            dead_letter_count=0,
            unknown_count=0,
            budget_exhausted=False,
            last_error=None,
        ),
        media_quality_binding=None,
        supplemental_qa_evidence=None,
        review_routing=LocalReviewRoutingSummary(
            disposition=ReviewRoutingDisposition.NOT_ROUTED,
        ),
        replayed=replayed,
        fixture_inference_calls=0,
        network_call_count=0,
        evidence_class="LOCAL_CONFORMANCE",
        production_eligible=False,
    )


def test_manifest_binds_exact_inputs_without_local_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "schemas").mkdir(parents=True)
    (root / "uv.lock").write_bytes(b"locked\n")
    (root / "schemas" / "schema-catalog.json").write_bytes(b"{}\n")
    source = tmp_path / "first" / "source.mcap"
    source.parent.mkdir()
    source.write_bytes(b"source bytes")
    relocated = tmp_path / "second" / "renamed.mcap"
    relocated.parent.mkdir()
    relocated.write_bytes(source.read_bytes())
    mapping = tmp_path / "mapping.json"
    mapping.write_bytes(b'{"mapping":"v1"}\n')
    monkeypatch.setattr(
        profile_module,
        "_git_facts",
        lambda _root: ProfileGitFacts(head_commit="b" * 40, dirty=True),
    )
    monkeypatch.setattr(profile_module, "_runtime_facts", _runtime)
    monkeypatch.setattr(profile_module, "_policy_facts", _policies)

    first = build_canonical_profile_manifest(
        repository_root=root,
        source_path=source,
        mapping_config=mapping,
        run_key="same-run",
        max_duration_ns=9_000_000_000,
        allow_unapproved_profile=True,
    )
    second = build_canonical_profile_manifest(
        repository_root=root,
        source_path=relocated,
        mapping_config=mapping,
        run_key="same-run",
        max_duration_ns=9_000_000_000,
        allow_unapproved_profile=True,
    )

    assert first == second
    assert first.git.dirty is True
    assert first.source.sha256 == hashlib.sha256(b"source bytes").hexdigest()
    payload = canonical_json_bytes(first.model_dump(mode="json"))
    assert str(tmp_path).encode() not in payload
    assert first.evidence_class == "LOCAL_CONFORMANCE"
    assert first.production_eligible is False
    assert first.measurement_status == "NOT_MEASURED"
    assert first.qualification_status == "NOT_PRODUCTION_QUALIFIED"


def test_manifest_digest_rejects_semantic_tampering() -> None:
    manifest = _manifest()
    tampered = manifest.model_dump(mode="python")
    tampered["run_key"] = "different"

    with pytest.raises(ValidationError, match="manifest_sha256"):
        CanonicalProfileManifest.model_validate(tampered, strict=True)


def test_state_snapshot_classifies_bytes_and_reads_sqlite_without_mutation(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    (state / "frames").mkdir(parents=True)
    (state / "video-view").mkdir()
    (state / "frames" / "frame.png").write_bytes(b"png")
    (state / "video-view" / "cam_01.mp4").write_bytes(b"video")
    (state / "facts.jsonl").write_bytes(b"{}\n")
    cas_blob = state / "artifact-registry" / "blobs" / "sha256" / "aa" / ("a" * 64)
    cas_blob.parent.mkdir(parents=True)
    cas_blob.write_bytes(b"cas")
    (state / "other.bin").write_bytes(b"other")
    excluded = state / "profile.json"
    excluded.write_bytes(b"excluded")
    database = state / "ledger.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE zeta (value INTEGER NOT NULL)")
        connection.execute("CREATE TABLE alpha (value INTEGER PRIMARY KEY AUTOINCREMENT)")
        connection.executemany("INSERT INTO zeta (value) VALUES (?)", ((1,), (2,)))
        connection.execute("INSERT INTO alpha DEFAULT VALUES")
    database_bytes = database.read_bytes()
    files_before = {
        path.relative_to(state).as_posix(): path.stat().st_size
        for path in state.rglob("*")
        if path.is_file()
    }

    snapshot = snapshot_state_tree(state, excluded_paths=(excluded,))

    assert database.read_bytes() == database_bytes
    assert snapshot == snapshot_state_tree(state, excluded_paths=(excluded,))
    assert {
        path.relative_to(state).as_posix(): path.stat().st_size
        for path in state.rglob("*")
        if path.is_file()
    } == files_before
    by_class = {item.file_class: item for item in snapshot.classes}
    assert tuple(by_class) == tuple(StateFileClass)
    assert by_class[StateFileClass.FRAME_PNG].file_count == 1
    assert by_class[StateFileClass.EXPORTED_VIDEO].file_count == 1
    assert by_class[StateFileClass.SQLITE].file_count == 1
    assert by_class[StateFileClass.JSON].file_count == 1
    assert by_class[StateFileClass.CONTENT_ADDRESSED_BLOB].file_count == 1
    assert by_class[StateFileClass.OTHER].file_count == 1
    assert snapshot.file_count == 6
    assert len(snapshot.sqlite_databases) == 1
    sqlite_snapshot = snapshot.sqlite_databases[0]
    assert sqlite_snapshot.error is None
    assert tuple((item.table_name, item.row_count) for item in sqlite_snapshot.tables) == (
        ("alpha", 1),
        ("zeta", 2),
    )


def test_state_snapshot_rejects_nonempty_wal_without_mutating_it(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    database = state / "live.sqlite3"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE facts (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO facts (value) VALUES (1)")
        connection.commit()
        wal = database.with_name(f"{database.name}-wal")
        assert wal.stat().st_size > 0
        files_before = {
            path.relative_to(state).as_posix(): path.stat().st_size
            for path in state.rglob("*")
            if path.is_file()
        }

        snapshot = snapshot_state_tree(state)

        assert {
            path.relative_to(state).as_posix(): path.stat().st_size
            for path in state.rglob("*")
            if path.is_file()
        } == files_before
        assert len(snapshot.sqlite_databases) == 1
        error = snapshot.sqlite_databases[0].error
        assert error is not None
        assert error.error_type == "_SQLiteSnapshotUnstableError"
        assert "nonempty SQLite WAL" in error.detail
    finally:
        connection.close()


def test_state_snapshot_counts_hardlinked_view_bytes_once(tmp_path: Path) -> None:
    state = tmp_path / "state"
    blob = state / "artifact-registry" / "blobs" / "sha256" / "aa" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"one physical payload")
    view = state / "video-view" / "cam_01.mp4"
    view.parent.mkdir()
    os.link(blob, view)

    snapshot = snapshot_state_tree(state)
    reconciliation = build_profile_reconciliation(
        observer=_observer(),
        state_after=snapshot,
        manifest=_manifest(),
        execution_mode="UNKNOWN",
    )

    assert snapshot.file_identity_status == "AVAILABLE"
    assert snapshot.file_count == 2
    assert snapshot.unique_file_count == 1
    assert snapshot.byte_count == 2 * len(b"one physical payload")
    assert snapshot.unique_byte_count == len(b"one physical payload")
    assert snapshot.hardlink_duplicate_path_count == 1
    assert snapshot.hardlink_duplicate_path_bytes == len(b"one physical payload")
    assert reconciliation.artifact_bytes.physical_duplication_status == "AVAILABLE"
    assert reconciliation.artifact_bytes.unique_state_bytes == len(b"one physical payload")


def test_duration_discovery_selects_matching_source_and_half_open_interval(
    tmp_path: Path,
) -> None:
    source_sha256 = _digest("selected-source")
    state = tmp_path / "state"
    video_view = state / "mcap" / "source-id" / "video-view"
    video_view.mkdir(parents=True)
    (video_view / "camera-video-export-manifest.json").write_bytes(
        canonical_json_bytes({"source_content_sha256": source_sha256})
    )
    (video_view.parent / "media-quality-report.json").write_bytes(
        canonical_json_bytes(
            {
                "recording_duration_ns": "100",
                "requested_interval": {"start_ns": "10", "end_ns": "80"},
            }
        )
    )

    assert discover_canonical_profile_durations(
        state,
        source_sha256=source_sha256,
    ) == (100, 70)
    assert discover_canonical_profile_durations(
        state,
        source_sha256=_digest("other-source"),
    ) == (None, None)


def test_work_queue_snapshot_reports_all_states_and_oldest_backlog_age(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    database = state / "work-scheduler.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE work_items (state TEXT, created_at TEXT)")
        connection.executemany(
            "INSERT INTO work_items (state, created_at) VALUES (?, ?)",
            (
                ("READY", "2026-01-01T00:00:00Z"),
                ("SUCCEEDED", "2025-01-01T00:00:00Z"),
            ),
        )
    database_bytes = database.read_bytes()
    files_before = {
        path.relative_to(state).as_posix(): path.stat().st_size
        for path in state.rglob("*")
        if path.is_file()
    }

    snapshot = snapshot_work_queue(
        state,
        observed_at=datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC),
    )

    assert database.read_bytes() == database_bytes
    assert {
        path.relative_to(state).as_posix(): path.stat().st_size
        for path in state.rglob("*")
        if path.is_file()
    } == files_before
    assert snapshot.status == "AVAILABLE"
    assert snapshot.nonterminal_backlog_count == 1
    assert snapshot.oldest_nonterminal_age_ns == 10_000_000_000
    assert len(snapshot.state_counts) > 2
    by_state = {item.state.value: item.count for item in snapshot.state_counts}
    assert by_state["READY"] == 1
    assert by_state["SUCCEEDED"] == 1


def test_unique_runtime_counter_requires_one_value_by_name() -> None:
    base = _observer().model_dump(mode="python")
    unique = RuntimeProfileSnapshot(
        **{
            **base,
            "counters": (RuntimeCounterSnapshot(name="duration", value=12),),
        }
    )
    ambiguous = RuntimeProfileSnapshot(
        **{
            **base,
            "counters": (
                RuntimeCounterSnapshot(
                    name="duration",
                    attributes=(RuntimeAttribute(name="camera", value="1"),),
                    value=5,
                ),
                RuntimeCounterSnapshot(
                    name="duration",
                    attributes=(RuntimeAttribute(name="camera", value="2"),),
                    value=7,
                ),
            ),
        }
    )

    assert unique_runtime_counter_value(unique, "duration") == 12
    assert unique_runtime_counter_value(unique, "missing") is None
    assert unique_runtime_counter_value(ambiguous, "duration") is None


def test_runtime_span_reconciliation_accounts_for_nested_and_uncovered_time() -> None:
    snapshot = RuntimeProfileSnapshot(
        elapsed_ns=100,
        process_cpu_ns=50,
        resources=RuntimeResourceSnapshot(
            rss_bytes=_unsupported_resource(),
            read_bytes_delta=_unsupported_resource(),
            write_bytes_delta=_unsupported_resource(),
        ),
        spans=(
            RuntimeSpanSnapshot(
                sequence=1,
                name="root",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=10,
                ended_offset_ns=90,
                elapsed_ns=80,
            ),
            RuntimeSpanSnapshot(
                sequence=2,
                parent_sequence=1,
                name="child-a",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=20,
                ended_offset_ns=40,
                elapsed_ns=20,
            ),
            RuntimeSpanSnapshot(
                sequence=3,
                parent_sequence=1,
                name="child-b",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=50,
                ended_offset_ns=80,
                elapsed_ns=30,
            ),
        ),
    )

    reconciliation = profile_module.reconcile_runtime_spans(snapshot)

    assert reconciliation.status == "RECONCILED"
    assert reconciliation.root_span_count == 1
    assert reconciliation.top_level_sum_ns == 80
    assert reconciliation.top_level_union_ns == 80
    assert reconciliation.exclusive_sum_ns == 80
    assert reconciliation.uncovered_wall_ns == 20


def test_report_requires_exactly_one_result_and_matches_replay_mode() -> None:
    manifest = _manifest()
    state = _empty_state()
    receipt = _receipt(replayed=False)

    report = CanonicalProfileReport(
        schema_version="1.0",
        model_version="canonical-profile-report-v2",
        manifest=manifest,
        manifest_sha256=manifest.manifest_sha256,
        observer=_observer(),
        state_before=state,
        state_after=state,
        state_file_count_delta=0,
        state_byte_count_delta=0,
        work_queue_after=snapshot_work_queue(Path("this-profile-state-does-not-exist")),
        receipt=receipt,
        error=None,
        execution_mode="FRESH",
        source_span_duration_ns=90,
        recording_duration_ns=100,
        requested_duration_ns=80,
        reconciliation=build_profile_reconciliation(
            observer=_observer(),
            state_after=state,
            manifest=manifest,
            execution_mode="FRESH",
        ),
    )

    assert report.execution_mode == "FRESH"
    with pytest.raises(ValidationError, match="execution_mode"):
        report.model_copy(update={"execution_mode": "REPLAY"}).model_validate(
            {**report.model_dump(mode="python"), "execution_mode": "REPLAY"},
            strict=True,
        )
    with pytest.raises(ValidationError, match="exactly one"):
        CanonicalProfileReport(
            **{
                **report.model_dump(mode="python"),
                "receipt": None,
                "error": None,
                "execution_mode": "UNKNOWN",
            }
        )

    failed = CanonicalProfileReport(
        **{
            **report.model_dump(mode="python"),
            "receipt": None,
            "error": CanonicalProfileRunError(
                code="SOURCE_INVALID",
                error_type="CanonicalLocalCompositionError",
                detail="invalid source",
            ),
            "execution_mode": "UNKNOWN",
            "recording_duration_ns": None,
            "requested_duration_ns": None,
        }
    )
    assert failed.error is not None
