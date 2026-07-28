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
    CompletionProfileMeasurements,
    ProfileFileFact,
    ProfileGitFacts,
    ProfileMetricAvailability,
    ProfileRuntimeFacts,
    SQLiteOperationMeasurements,
    StateFileClass,
    build_canonical_profile_manifest,
    build_canonical_profile_measurements,
    build_profile_capacity,
    build_profile_reconciliation,
    canonical_profile_workload_fingerprint,
    compare_canonical_profile_reports,
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


def test_state_snapshot_excludes_hardlink_owned_by_source_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"source authority owns these bytes")
    state = tmp_path / "state"
    registry_blob = state / "artifact-registry" / "blobs" / "sha256" / "aa" / ("a" * 64)
    registry_blob.parent.mkdir(parents=True)
    os.link(source, registry_blob)
    derived = state / "derived.json"
    derived.write_bytes(b"derived")

    snapshot = snapshot_state_tree(
        state,
        externally_owned_paths=(source,),
    )

    assert snapshot.file_count == 2
    assert snapshot.unique_file_count == 1
    assert snapshot.unique_byte_count == len(b"derived")
    assert snapshot.hardlink_duplicate_path_count == 1
    assert snapshot.hardlink_duplicate_path_bytes == source.stat().st_size


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


def _available_resource(value: int) -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(status=RuntimeResourceStatus.AVAILABLE, value=value)


def _measured_observer() -> RuntimeProfileSnapshot:
    counters = (
        RuntimeCounterSnapshot(name="inference.call_parts", value=8),
        RuntimeCounterSnapshot(name="inference.call_splits", value=3),
        RuntimeCounterSnapshot(name="inference.coarse_unique_images", value=8),
        RuntimeCounterSnapshot(name="inference.dense_logical_calls", value=2),
        RuntimeCounterSnapshot(name="inference.dense_provider_images", value=4),
        RuntimeCounterSnapshot(name="inference.dense_unique_images", value=4),
        RuntimeCounterSnapshot(name="inference.input_tokens", value=1_200),
        RuntimeCounterSnapshot(name="inference.logical_calls", value=5),
        RuntimeCounterSnapshot(name="inference.output_token_responses", value=5),
        RuntimeCounterSnapshot(name="inference.output_tokens", value=150),
        RuntimeCounterSnapshot(name="inference.provider_batch_requests", value=5),
        RuntimeCounterSnapshot(name="inference.provider_batches", value=2),
        RuntimeCounterSnapshot(name="inference.provider_images", value=20),
        RuntimeCounterSnapshot(name="inference.provider_retries", value=1),
        RuntimeCounterSnapshot(name="inference.unique_images", value=12),
        RuntimeCounterSnapshot(name="sampling.windows", value=4),
        RuntimeCounterSnapshot(name="sqlite.inference_evidence.connections", value=3),
        RuntimeCounterSnapshot(
            name="sqlite.inference_evidence.transactions",
            attributes=(
                RuntimeAttribute(name="operation", value="read"),
                RuntimeAttribute(name="write", value=False),
            ),
            value=2,
        ),
        RuntimeCounterSnapshot(
            name="sqlite.inference_evidence.transactions",
            attributes=(
                RuntimeAttribute(name="operation", value="write"),
                RuntimeAttribute(name="write", value=True),
            ),
            value=4,
        ),
    )
    return RuntimeProfileSnapshot(
        elapsed_ns=1_000,
        process_cpu_ns=500,
        resources=RuntimeResourceSnapshot(
            rss_bytes=_available_resource(99),
            read_bytes_delta=_available_resource(123),
            write_bytes_delta=_available_resource(456),
        ),
        spans=(
            RuntimeSpanSnapshot(
                sequence=1,
                name="source.decode",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=0,
                ended_offset_ns=100,
                elapsed_ns=100,
                process_cpu_ns=40,
            ),
            RuntimeSpanSnapshot(
                sequence=2,
                name="source.decode",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=100,
                ended_offset_ns=200,
                elapsed_ns=100,
                process_cpu_ns=30,
            ),
            RuntimeSpanSnapshot(
                sequence=3,
                name="sqlite.inference_evidence.transaction",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=200,
                ended_offset_ns=400,
                elapsed_ns=200,
                process_cpu_ns=50,
            ),
        ),
        counters=counters,
    )


def _v3_report(
    *,
    replayed: bool,
    worker_count: int = 1,
    wall_time_ns: int = 1_000,
    run_key: str = "profile-run",
) -> CanonicalProfileReport:
    manifest = _manifest(run_key=run_key)
    state = _empty_state()
    receipt = _receipt(replayed=replayed).model_copy(update={"fixture_inference_calls": 5})
    observer = _measured_observer().model_copy(update={"elapsed_ns": wall_time_ns})
    measurements = build_canonical_profile_measurements(
        observer=observer,
        state_before=state,
        state_after=state,
        manifest=manifest,
        receipt=receipt,
        recording_worker_count=worker_count,
    )
    capacity = build_profile_capacity(
        observer=observer,
        manifest=manifest,
        receipt=receipt,
        execution_mode="REPLAY" if replayed else "FRESH",
        recording_duration_ns=500,
        requested_duration_ns=400,
        measurements=measurements,
    )
    return CanonicalProfileReport(
        schema_version="1.0",
        model_version="canonical-profile-report-v3",
        manifest=manifest,
        manifest_sha256=manifest.manifest_sha256,
        observer=observer,
        state_before=state,
        state_after=state,
        state_file_count_delta=0,
        state_byte_count_delta=0,
        work_queue_after=snapshot_work_queue(Path("this-profile-state-does-not-exist")),
        receipt=receipt,
        error=None,
        execution_mode="REPLAY" if replayed else "FRESH",
        source_span_duration_ns=None,
        recording_duration_ns=500,
        requested_duration_ns=400,
        reconciliation=build_profile_reconciliation(
            observer=observer,
            state_after=state,
            manifest=manifest,
            execution_mode="REPLAY" if replayed else "FRESH",
        ),
        measurements=measurements,
        capacity=capacity,
    )


def test_profile_measurements_expose_provider_multipliers_sqlite_scope_and_stage_cpu() -> None:
    report = _v3_report(replayed=False)

    assert report.measurements is not None
    assert report.capacity is not None
    assert report.measurements.workload_fingerprint == canonical_profile_workload_fingerprint(
        report.manifest
    )
    assert report.measurements.source_bytes == 10
    assert report.measurements.provider_mode.value == "LOCAL_OFFLINE_FIXTURE"
    assert report.measurements.sqlite.connection_count_status.value == "PARTIAL"
    assert report.measurements.sqlite.connection_count == 3
    assert report.measurements.sqlite.transaction_count_status.value == "PARTIAL"
    assert report.measurements.sqlite.transaction_count == 6
    assert report.measurements.sqlite.read_transaction_count == 2
    assert report.measurements.sqlite.write_transaction_count == 4
    assert report.measurements.sqlite.sqlite_read_bytes_status.value == "NOT_AVAILABLE"
    assert report.measurements.sqlite.sqlite_read_bytes is None
    assert report.measurements.sqlite.process_read_bytes.value == 123
    stages = {stage.stage: stage for stage in report.measurements.stages}
    assert stages["source.decode"].span_count == 2
    assert stages["source.decode"].inclusive_wall_time_ns == 200
    assert stages["source.decode"].inclusive_process_cpu_ns == 70
    assert report.capacity.provider_images == 20
    assert report.capacity.unique_images == 12
    assert report.capacity.windows == 4
    assert report.capacity.logical_calls == 5
    assert report.capacity.call_parts == 8
    assert report.capacity.call_splits == 3
    assert report.capacity.call_parts_per_logical_call == pytest.approx(1.6)
    assert report.capacity.logical_calls_per_window == pytest.approx(1.25)
    assert report.capacity.input_tokens == 1_200
    assert report.capacity.output_tokens == 150
    assert report.capacity.dense_logical_calls == 2
    assert report.capacity.dense_logical_call_fraction == pytest.approx(0.4)
    assert report.capacity.dense_unique_images == 4
    assert report.capacity.dense_upgrade_fraction == pytest.approx(4 / 12)
    assert report.capacity.dense_provider_image_fraction == pytest.approx(0.2)
    assert report.capacity.provider_images_per_unique_image == pytest.approx(20 / 12)
    assert report.capacity.production_eligible is False


def test_profile_comparison_keeps_scheduler_costs_by_stable_operation() -> None:
    report = _v3_report(replayed=False)
    assert report.measurements is not None

    baseline_operation = SQLiteOperationMeasurements(
        operation="plan_many",
        connection_count=1,
        connection_setup_duration_ns=25,
        transaction_count=1,
        begin_lock_wait_duration_ns=10,
        transaction_duration_ns=100,
        operation_duration_ns=60,
        commit_duration_ns=30,
        rollback_duration_ns=0,
        rows_committed=12,
        rows_rolled_back=0,
        retry_count=0,
        rollback_count=0,
        busy_or_locked_failure_count=0,
        fsync_count_status=ProfileMetricAvailability.NOT_AVAILABLE,
        fsync_count=None,
    )
    candidate_operation = baseline_operation.model_copy(
        update={
            "connection_setup_duration_ns": 15,
            "transaction_duration_ns": 50,
            "operation_duration_ns": 25,
            "commit_duration_ns": 10,
        }
    )
    baseline_measurements = report.measurements.model_copy(
        update={
            "sqlite": report.measurements.sqlite.model_copy(
                update={"operations": (baseline_operation,)}
            )
        }
    )
    candidate_measurements = report.measurements.model_copy(
        update={
            "sqlite": report.measurements.sqlite.model_copy(
                update={"operations": (candidate_operation,)}
            )
        }
    )
    baseline = report.model_copy(update={"measurements": baseline_measurements})
    candidate = report.model_copy(update={"measurements": candidate_measurements})

    comparison = compare_canonical_profile_reports(baseline, candidate)
    transaction = next(
        item
        for item in comparison.resources
        if item.metric == "sqlite.operation.plan_many.transaction_duration_ns"
    )
    fsync = next(
        item
        for item in comparison.resources
        if item.metric == "sqlite.operation.plan_many.fsync_count"
    )

    assert transaction.baseline_value == 100
    assert transaction.candidate_value == 50
    assert transaction.candidate_to_baseline_ratio == pytest.approx(0.5)
    assert fsync.baseline_availability is ProfileMetricAvailability.NOT_AVAILABLE
    assert fsync.candidate_availability is ProfileMetricAvailability.NOT_AVAILABLE
    assert fsync.candidate_to_baseline_ratio is None


def test_profile_measurements_aggregate_scheduler_operation_facts() -> None:
    attributes = (
        RuntimeAttribute(name="operation", value="plan_many"),
        RuntimeAttribute(name="synchronous", value="FULL"),
        RuntimeAttribute(name="write", value=True),
    )
    observer = RuntimeProfileSnapshot(
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
                name="sqlite.work_scheduler.connection_setup",
                attributes=attributes,
                status=RuntimeSpanStatus.OK,
                started_offset_ns=0,
                ended_offset_ns=10,
                elapsed_ns=10,
            ),
            RuntimeSpanSnapshot(
                sequence=2,
                name="sqlite.work_scheduler.transaction",
                attributes=attributes,
                status=RuntimeSpanStatus.OK,
                started_offset_ns=10,
                ended_offset_ns=100,
                elapsed_ns=90,
            ),
            RuntimeSpanSnapshot(
                sequence=3,
                name="sqlite.work_scheduler.begin",
                attributes=attributes,
                status=RuntimeSpanStatus.OK,
                started_offset_ns=10,
                ended_offset_ns=20,
                elapsed_ns=10,
            ),
            RuntimeSpanSnapshot(
                sequence=4,
                name="sqlite.work_scheduler.operation",
                attributes=attributes,
                status=RuntimeSpanStatus.OK,
                started_offset_ns=20,
                ended_offset_ns=80,
                elapsed_ns=60,
            ),
            RuntimeSpanSnapshot(
                sequence=5,
                name="sqlite.work_scheduler.commit",
                attributes=attributes,
                status=RuntimeSpanStatus.OK,
                started_offset_ns=80,
                ended_offset_ns=100,
                elapsed_ns=20,
            ),
        ),
        counters=(
            RuntimeCounterSnapshot(
                name="sqlite.work_scheduler.connections",
                attributes=attributes,
                value=1,
            ),
            RuntimeCounterSnapshot(
                name="sqlite.work_scheduler.rows_committed",
                attributes=attributes,
                value=12,
            ),
            RuntimeCounterSnapshot(
                name="sqlite.work_scheduler.transactions",
                attributes=attributes,
                value=1,
            ),
            RuntimeCounterSnapshot(
                name="sqlite.work_scheduler.work_retries",
                attributes=attributes,
                value=1,
            ),
        ),
    )
    state = _empty_state()
    measurements = build_canonical_profile_measurements(
        observer=observer,
        state_before=state,
        state_after=state,
        manifest=_manifest(),
        receipt=_receipt(),
    )

    operation = measurements.sqlite.operations
    assert len(operation) == 1
    assert operation[0].operation == "plan_many"
    assert operation[0].connection_count == 1
    assert operation[0].begin_lock_wait_duration_ns == 10
    assert operation[0].transaction_duration_ns == 90
    assert operation[0].operation_duration_ns == 60
    assert operation[0].commit_duration_ns == 20
    assert operation[0].rows_committed == 12
    assert operation[0].retry_count == 1
    assert operation[0].fsync_count_status is ProfileMetricAvailability.NOT_AVAILABLE
    assert operation[0].fsync_count is None


def test_profile_measurements_aggregate_completion_payload_and_root_facts() -> None:
    observer = RuntimeProfileSnapshot(
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
                name="completion.command.roots.leaf_prepare",
                attributes=(RuntimeAttribute(name="collection", value="run-memberships"),),
                status=RuntimeSpanStatus.OK,
                started_offset_ns=0,
                ended_offset_ns=10,
                elapsed_ns=10,
            ),
            RuntimeSpanSnapshot(
                sequence=2,
                name="completion.command.roots.compute",
                attributes=(RuntimeAttribute(name="collection", value="run-memberships"),),
                status=RuntimeSpanStatus.OK,
                started_offset_ns=10,
                ended_offset_ns=30,
                elapsed_ns=20,
            ),
            RuntimeSpanSnapshot(
                sequence=3,
                name="completion.command.detail.model_validate",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=30,
                ended_offset_ns=55,
                elapsed_ns=25,
            ),
            RuntimeSpanSnapshot(
                sequence=4,
                name="completion.commit.detail_artifact.insert",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=55,
                ended_offset_ns=75,
                elapsed_ns=20,
            ),
            RuntimeSpanSnapshot(
                sequence=5,
                name="completion.commit.database_commit",
                status=RuntimeSpanStatus.OK,
                started_offset_ns=75,
                ended_offset_ns=100,
                elapsed_ns=25,
            ),
        ),
        counters=(
            RuntimeCounterSnapshot(
                name="completion.command.command_bytes",
                value=300,
            ),
            RuntimeCounterSnapshot(
                name="completion.command.detail_bytes",
                value=200,
            ),
            RuntimeCounterSnapshot(
                name="completion.command.processing_run_bytes",
                value=100,
            ),
            RuntimeCounterSnapshot(
                name="completion.command.root_collections",
                value=11,
            ),
            RuntimeCounterSnapshot(
                name="completion.command.root_leaves",
                value=7,
            ),
        ),
    )
    state = _empty_state()
    measurements = build_canonical_profile_measurements(
        observer=observer,
        state_before=state,
        state_after=state,
        manifest=_manifest(),
        receipt=_receipt(),
    )

    completion = measurements.completion
    assert completion is not None
    assert completion.detail_bytes == 200
    assert completion.command_bytes == 300
    assert completion.processing_run_bytes == 100
    assert completion.ordered_root_collection_count == 11
    assert completion.ordered_root_leaf_count == 7
    assert completion.ordered_root_leaf_count_status is ProfileMetricAvailability.AVAILABLE
    stages = {stage.stage: stage for stage in measurements.stages}
    assert stages["completion.command.roots.leaf_prepare"].inclusive_wall_time_ns == 10
    assert stages["completion.command.roots.compute"].inclusive_wall_time_ns == 20
    assert stages["completion.command.detail.model_validate"].inclusive_wall_time_ns == 25
    assert stages["completion.commit.detail_artifact.insert"].inclusive_wall_time_ns == 20
    assert stages["completion.commit.database_commit"].inclusive_wall_time_ns == 25

    report = _v3_report(replayed=False)
    assert report.measurements is not None
    baseline_completion = CompletionProfileMeasurements(
        detail_bytes_status=ProfileMetricAvailability.AVAILABLE,
        detail_bytes=200,
        command_bytes_status=ProfileMetricAvailability.AVAILABLE,
        command_bytes=300,
        processing_run_bytes_status=ProfileMetricAvailability.AVAILABLE,
        processing_run_bytes=100,
        ordered_root_collection_count_status=ProfileMetricAvailability.AVAILABLE,
        ordered_root_collection_count=11,
        ordered_root_leaf_count_status=ProfileMetricAvailability.AVAILABLE,
        ordered_root_leaf_count=7,
    )
    candidate_completion = baseline_completion.model_copy(update={"command_bytes": 150})
    baseline = report.model_copy(
        update={
            "measurements": report.measurements.model_copy(
                update={"completion": baseline_completion}
            )
        }
    )
    candidate = report.model_copy(
        update={
            "measurements": report.measurements.model_copy(
                update={"completion": candidate_completion}
            )
        }
    )

    comparison = compare_canonical_profile_reports(baseline, candidate)
    command_bytes = next(
        item for item in comparison.resources if item.metric == "completion.command_bytes"
    )
    assert command_bytes.baseline_value == 300
    assert command_bytes.candidate_value == 150
    assert command_bytes.candidate_to_baseline_ratio == pytest.approx(0.5)


def test_profile_capacity_preserves_known_zero_output_tokens() -> None:
    manifest = _manifest()
    state = _empty_state()
    receipt = _receipt().model_copy(update={"fixture_inference_calls": 5})
    observer = _measured_observer().model_copy(
        update={
            "counters": tuple(
                counter
                for counter in _measured_observer().counters
                if counter.name != "inference.output_tokens"
            )
        }
    )
    measurements = build_canonical_profile_measurements(
        observer=observer,
        state_before=state,
        state_after=state,
        manifest=manifest,
        receipt=receipt,
    )

    capacity = build_profile_capacity(
        observer=observer,
        manifest=manifest,
        receipt=receipt,
        execution_mode="FRESH",
        recording_duration_ns=500,
        requested_duration_ns=400,
        measurements=measurements,
    )

    assert capacity.output_token_responses == 5
    assert capacity.output_tokens == 0


def test_profile_capacity_stays_unavailable_when_a_failed_run_has_no_provider_mode() -> None:
    manifest = _manifest()
    state = _empty_state()
    observer = _measured_observer()
    measurements = build_canonical_profile_measurements(
        observer=observer,
        state_before=state,
        state_after=state,
        manifest=manifest,
        receipt=None,
    )

    capacity = build_profile_capacity(
        observer=observer,
        manifest=manifest,
        receipt=None,
        execution_mode="UNKNOWN",
        recording_duration_ns=500,
        requested_duration_ns=400,
        measurements=measurements,
    )

    assert capacity.measurement_status.value == "NOT_AVAILABLE"
    assert capacity.unavailable_reasons == ("MISSING_PROVIDER_MODE",)
    assert capacity.recording_hours_per_wall_hour is None


def test_profile_comparison_exposes_fresh_replay_and_recording_worker_stage_ratios() -> None:
    fresh = _v3_report(replayed=False, worker_count=1, wall_time_ns=1_000)
    replay = _v3_report(replayed=True, worker_count=1, wall_time_ns=500)
    scaled = _v3_report(replayed=False, worker_count=2, wall_time_ns=500)

    fresh_replay = compare_canonical_profile_reports(fresh, replay)
    worker_scaling = compare_canonical_profile_reports(fresh, scaled)

    assert fresh_replay.capacity.comparable is True
    assert fresh_replay.capacity.comparison_kind.value == "FRESH_VS_REPLAY"
    assert fresh_replay.capacity.recording_hours_per_wall_hour_ratio == pytest.approx(2.0)
    assert fresh_replay.capacity.rate_ratios[-1].name == "windows_per_wall_hour"
    sqlite_transactions = next(
        item for item in fresh_replay.resources if item.metric == "sqlite.transaction_count"
    )
    assert sqlite_transactions.baseline_availability.value == "PARTIAL"
    assert sqlite_transactions.candidate_to_baseline_ratio == pytest.approx(1.0)
    source_stage = next(stage for stage in fresh_replay.stages if stage.stage == "source.decode")
    assert source_stage.inclusive_wall_time_ratio == pytest.approx(1.0)
    assert worker_scaling.capacity.comparison_kind.value == "RECORDING_WORKER_SCALING"
    assert worker_scaling.capacity.camera_hours_per_wall_hour_ratio == pytest.approx(2.0)


def test_fresh_replay_comparison_requires_a_shared_run_key() -> None:
    fresh = _v3_report(replayed=False, run_key="fresh-run")
    replay = _v3_report(replayed=True, run_key="unrelated-run")

    comparison = compare_canonical_profile_reports(fresh, replay)

    assert comparison.capacity.comparable is False
    assert comparison.capacity.comparison_kind.value == "FRESH_VS_REPLAY"
    assert comparison.capacity.non_comparable_reasons == ("RUN_KEY_CHANGED",)
    assert all(item.candidate_to_baseline_ratio is None for item in comparison.resources)
