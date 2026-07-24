from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.stream_scheduler import DurableStreamWindowScheduler
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.stream_finalization import WindowTerminalMember
from robata.runtime import local_streaming_smoke as streaming_smoke
from robata.runtime.local_streaming_benchmark import (
    BenchmarkCacheState,
    BenchmarkRunMode,
    HostRuntimePins,
    OfferedLoadUnit,
)
from robata.runtime.local_streaming_smoke import (
    LocalStreamingSmokeConfig,
    LocalStreamingSmokeManifest,
    LocalStreamingSmokeReport,
    run_local_streaming_smoke,
)


def _manifest(config: LocalStreamingSmokeConfig) -> LocalStreamingSmokeManifest:
    return LocalStreamingSmokeManifest.create(
        candidate_commit="1" * 40,
        candidate_worktree_state="DIRTY",
        candidate_worktree_status_sha256="2" * 64,
        lockfile_sha256="3" * 64,
        host=HostRuntimePins(
            cpu_model="test-cpu",
            logical_cpu_count=4,
            gpu_model="NONE",
            memory_bytes=8 * 1024**3,
            driver_version="NONE",
            operating_system="test-os",
            power_mode="test-power",
            runtime="CPython test",
        ),
        config=config,
    )


def _content_addressed_target(root: Path, digest: str) -> Path:
    return root / digest[:2] / f"{digest}.json"


def _content_addressed_temps(target: Path) -> tuple[Path, ...]:
    return tuple(target.parent.glob(f".{target.name}.*.tmp"))


def test_content_addressed_publish_is_atomic_and_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports"
    digest = "a" * 64
    payload = b'{"complete":true}'
    target = _content_addressed_target(root, digest)
    real_link = streaming_smoke.os.link
    real_fsync = streaming_smoke.os.fsync
    links: list[tuple[Path, Path]] = []
    fsync_count = 0

    def observed_link(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == target.parent
        assert source_path.read_bytes() == payload
        assert destination_path == target
        assert not target.exists()
        links.append((source_path, destination_path))
        real_link(source, destination)

    def observed_fsync(descriptor: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        real_fsync(descriptor)

    monkeypatch.setattr(streaming_smoke.os, "link", observed_link)
    monkeypatch.setattr(streaming_smoke.os, "fsync", observed_fsync)

    assert streaming_smoke._write_content_addressed(root, digest, payload) == target
    assert target.read_bytes() == payload
    assert links
    assert fsync_count >= 1
    assert not _content_addressed_temps(target)


def test_content_addressed_publish_reuses_preexisting_identical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports"
    digest = "b" * 64
    payload = b'{"complete":true}'
    target = _content_addressed_target(root, digest)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    def unexpected_link(_source: object, _destination: object) -> None:
        pytest.fail("identical content-addressed bytes must not be republished")

    monkeypatch.setattr(streaming_smoke.os, "link", unexpected_link)

    assert streaming_smoke._write_content_addressed(root, digest, payload) == target
    assert target.read_bytes() == payload
    assert not _content_addressed_temps(target)


@pytest.mark.parametrize(
    "existing",
    (
        b'{"different":true}',
        b'{"complete":true',
    ),
    ids=("conflicting", "truncated"),
)
def test_content_addressed_publish_rejects_preexisting_conflicting_result(
    tmp_path: Path,
    existing: bytes,
) -> None:
    root = tmp_path / "reports"
    digest = "c" * 64
    payload = b'{"complete":true}'
    target = _content_addressed_target(root, digest)
    target.parent.mkdir(parents=True)
    target.write_bytes(existing)

    with pytest.raises(RuntimeError, match="conflicts with existing bytes"):
        streaming_smoke._write_content_addressed(root, digest, payload)

    assert target.read_bytes() == existing
    assert not _content_addressed_temps(target)


def test_content_addressed_publish_link_failure_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports"
    digest = "d" * 64
    payload = b'{"complete":true}'
    target = _content_addressed_target(root, digest)

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(streaming_smoke.os, "link", fail_link)

    with pytest.raises(OSError, match="injected link failure"):
        streaming_smoke._write_content_addressed(root, digest, payload)

    assert not target.exists()
    assert not _content_addressed_temps(target)


def test_content_addressed_publish_never_overwrites_racing_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports"
    digest = "f" * 64
    payload = b'{"complete":true}'
    competing_payload = b'{"competing":true}'
    target = _content_addressed_target(root, digest)

    def competing_link(_source: object, destination: str | Path) -> None:
        destination_path = Path(destination)
        destination_path.write_bytes(competing_payload)
        raise FileExistsError("injected competing publisher")

    monkeypatch.setattr(streaming_smoke.os, "link", competing_link)

    with pytest.raises(RuntimeError, match="conflicts with existing bytes"):
        streaming_smoke._write_content_addressed(root, digest, payload)

    assert target.read_bytes() == competing_payload
    assert not _content_addressed_temps(target)


def test_content_addressed_publish_write_failure_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports"
    digest = "e" * 64
    payload = b'{"complete":true}'
    target = _content_addressed_target(root, digest)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected write durability failure")

    monkeypatch.setattr(streaming_smoke.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected write durability failure"):
        streaming_smoke._write_content_addressed(root, digest, payload)

    assert not target.exists()
    assert not _content_addressed_temps(target)


def test_actual_local_mock_runs_durable_scheduler_and_writes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_member_scans = 0
    original_terminal_members = DurableStreamWindowScheduler.terminal_members

    def counted_terminal_members(
        self: DurableStreamWindowScheduler,
    ) -> tuple[WindowTerminalMember, ...]:
        nonlocal terminal_member_scans
        terminal_member_scans += 1
        return original_terminal_members(self)

    monkeypatch.setattr(
        DurableStreamWindowScheduler,
        "terminal_members",
        counted_terminal_members,
    )
    manifest = _manifest(
        LocalStreamingSmokeConfig(
            source_duration_ms=3_000,
            window_duration_ms=1_000,
            window_hop_ms=1_000,
            window_batch_size=2,
            drain_batch_size=16,
            mock_fixed_latency_ms=1,
            mock_failure_probability_ppm=0,
            mock_failure_seed=7,
            mock_retry_limit=1,
        )
    )

    artifacts = run_local_streaming_smoke(
        manifest=manifest,
        output_root=tmp_path,
        execution_run_id="00000000-0000-0000-0000-000000000123",
        sleep=lambda _seconds: None,
    )

    report = artifacts.report
    assert report.execution_mode == "ACTUAL_LOCAL_MOCK"
    assert report.measurement_status == "NOT_MEASURED"
    assert report.production_eligible is False
    assert report.wp6_minimum_duration_met is False
    assert report.incremental_latency_p95_target_met
    assert report.incremental_latency_p99_target_met
    assert report.no_growing_backlog_met
    assert report.wp6_actual_smoke_gate_met is False
    assert report.metrics.declared_window_count == 3
    assert report.metrics.terminal_window_count == 3
    assert report.metrics.eligible_to_terminal_sample_count == 3
    assert report.metrics.active_backlog_high_water > 0
    assert report.metrics.active_backlog_end == 0
    assert report.metrics.active_backlog_after_drain_samples
    assert report.metrics.recording_seconds_per_wall_second > 0
    assert report.metrics.mock_provider_batch_request_count > 0
    assert (
        report.metrics.mock_provider_attempt_count
        >= report.metrics.mock_provider_batch_request_count
    )
    assert report.metrics.mock_provider_timeout_count == 0
    assert report.metrics.mock_provider_max_batch_size_observed <= 16
    assert report.recording_output_decision == "NO_EVENTS"
    assert len(report.recording_result_semantic_sha256) == 64
    assert report.benchmark_manifest_sha256 == manifest.benchmark_manifest.manifest_sha256
    assert report.authority_status == "AUTHORITATIVE_LOCAL_MOCK_SMOKE"
    assert tuple(item.unit for item in report.offered_load_projections) == tuple(OfferedLoadUnit)
    assert all(
        item.projection_authority == "LOCAL_MOCK_SCALE_OUT_ARITHMETIC_ONLY"
        and item.production_eligible is False
        for item in report.offered_load_projections
    )
    assert artifacts.database_path.is_file()
    assert artifacts.manifest_path.name == f"{manifest.manifest_sha256}.json"
    assert artifacts.report_path.name == f"{report.report_sha256}.json"
    assert artifacts.manifest_path.read_bytes() == canonical_json_bytes(manifest)
    stored_report = LocalStreamingSmokeReport.model_validate_json(
        artifacts.report_path.read_bytes(), strict=True
    )
    assert stored_report == report
    assert terminal_member_scans == 1

    benchmark = manifest.benchmark_manifest
    assert tuple(state.mode for state in benchmark.run_states) == tuple(BenchmarkRunMode)
    assert tuple(state.cache_state for state in benchmark.run_states) == (
        BenchmarkCacheState.EMPTY,
        BenchmarkCacheState.DISABLED,
        BenchmarkCacheState.RESTORED,
    )
    assert benchmark.artifact_retention_profile_sha256
    assert benchmark.policies.chunk_duration_ms == 1_000
    assert benchmark.policies.window_duration_ms == 1_000
    assert benchmark.policies.allowed_lateness_ms == 0
    assert benchmark.policies.ring_capacity_ms == 2_000
    assert all(
        len(digest) == 64 for field, digest in benchmark.policies if field.endswith("_sha256")
    )
    assert benchmark.mock_provider.latency.points[0].latency_ms == 1
    assert benchmark.mock_provider.failure.failure_probability_ppm == 0
    assert benchmark.mock_provider.retry.maximum_attempts == 2
    assert benchmark.mock_provider.request_timeout_ms == 30_000
    assert benchmark.mock_provider.request_limit_per_second == 1_000
    assert benchmark.mock_provider.max_batch_size == 16
    assert benchmark.mock_provider.max_concurrency_per_group == 1
    assert benchmark.protocol.warmup_count == 0
    assert benchmark.protocol.repetition_count == 1
    assert benchmark.protocol.smoke_duration_ms == 30 * 60 * 1_000
    assert tuple(item.unit for item in benchmark.offered_loads) == tuple(OfferedLoadUnit)
    assert all(item.offered_hours_per_day == 500 for item in benchmark.offered_loads)


def test_seeded_failures_leave_work_for_eos_recovery_without_losing_windows(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        LocalStreamingSmokeConfig(
            source_duration_ms=2_000,
            window_duration_ms=1_000,
            window_hop_ms=1_000,
            window_batch_size=1,
            drain_batch_size=2,
            mock_fixed_latency_ms=1,
            mock_failure_probability_ppm=1_000_000,
            mock_failure_seed=11,
            mock_retry_limit=1,
        )
    )

    artifacts = run_local_streaming_smoke(
        manifest=manifest,
        output_root=tmp_path,
        sleep=lambda _seconds: None,
    )

    metrics = artifacts.report.metrics
    assert metrics.injected_retryable_failure_count > 0
    assert metrics.mock_provider_attempt_count == metrics.drain_opportunity_count
    assert metrics.mock_provider_attempt_count == metrics.injected_retryable_failure_count
    assert metrics.bounded_drain_call_count == 0
    assert metrics.eos_recovery_used
    assert metrics.terminal_window_count == 2
    assert metrics.active_backlog_end == 0
    assert artifacts.report.no_growing_backlog_met is False


def test_provider_batches_enforce_manifest_size_and_request_timeout(
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    manifest = _manifest(
        LocalStreamingSmokeConfig(
            source_duration_ms=2_000,
            window_duration_ms=1_000,
            window_hop_ms=1_000,
            window_batch_size=2,
            drain_batch_size=16,
            mock_fixed_latency_ms=10,
            mock_failure_probability_ppm=0,
            mock_retry_limit=1,
            mock_request_timeout_ms=1,
            mock_request_limit_per_second=1,
            mock_max_batch_size=1,
        )
    )

    artifacts = run_local_streaming_smoke(
        manifest=manifest,
        output_root=tmp_path,
        sleep=sleeps.append,
    )

    metrics = artifacts.report.metrics
    assert manifest.benchmark_manifest.mock_provider.max_batch_size == 1
    assert metrics.mock_provider_max_batch_size_observed == 1
    assert metrics.mock_provider_batch_request_count > 0
    assert metrics.mock_provider_timeout_count == metrics.mock_provider_attempt_count
    assert metrics.injected_retryable_failure_count == 0
    assert metrics.mock_provider_rate_limit_wait_count > 0
    assert any(delay >= 0.999 for delay in sleeps)
    assert metrics.bounded_drain_call_count == 0
    assert metrics.eos_recovery_used
    assert metrics.active_backlog_end == 0
