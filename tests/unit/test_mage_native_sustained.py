from __future__ import annotations

import json

import pytest

from robata.benchmark.mage_native_sustained import (
    MageNativeGenerationTelemetrySample,
    MageNativeQualificationStatus,
    MageNativeRunIdentity,
    MageNativeRunMeasurement,
    MageNativeSustainedQualificationPolicy,
    MageNativeTelemetryDisposition,
    MageNativeTimeInterval,
    assess_cross_run_isolation,
    assess_run_compatibility,
    assess_run_freshness,
    build_mage_native_sustained_comparison_report,
    generation_gap_summary,
    interval_intersection_seconds,
    interval_union_seconds,
    merge_intervals,
    serial_vs_prefetch_speedup,
    summarize_run,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _identity(**overrides: str) -> MageNativeRunIdentity:
    values = {
        "model_identity_sha256": _digest(1),
        "checkpoint_sha256": _digest(2),
        "source_media_sha256": _digest(3),
        "segment_manifest_sha256": _digest(4),
        "prompt_sha256": _digest(5),
        "codec_policy_sha256": _digest(6),
        "camera_id": "camera-1",
    }
    values.update(overrides)
    return MageNativeRunIdentity.model_validate(values)


def _sample(
    ordinal: int,
    *,
    generation_start: float | None,
    generation_end: float | None,
    arm: str,
    request_id: str | None = None,
    inference_digest: str | None = None,
    artifact_digest: str | None = None,
    output_text_digest: str | None = None,
    processor: tuple[float, float] | None = None,
    disposition: MageNativeTelemetryDisposition = MageNativeTelemetryDisposition.FRESH_GENERATION,
    output_tokens: int = 40,
    max_new_tokens: int = 256,
    output_valid: bool = True,
    telemetry_event_version: str = "mage-endpoint-telemetry-v1",
) -> MageNativeGenerationTelemetrySample:
    generation_interval = (
        MageNativeTimeInterval(
            start_seconds=generation_start,
            end_seconds=generation_end,
        )
        if generation_start is not None and generation_end is not None
        else None
    )
    processor_interval = (
        MageNativeTimeInterval(start_seconds=processor[0], end_seconds=processor[1])
        if processor is not None
        else None
    )
    return MageNativeGenerationTelemetrySample(
        telemetry_event_version=telemetry_event_version,
        segment_ordinal=ordinal,
        request_id=request_id or f"{arm}-request-{ordinal}",
        inference_identity_sha256=inference_digest or _digest(100 + ordinal),
        result_artifact_identity_sha256=artifact_digest
        or _digest((1_000 if arm == "serial" else 2_000) + ordinal),
        output_text_sha256=output_text_digest or _digest(3_000 + ordinal),
        disposition=disposition,
        processor_interval=processor_interval,
        generation_interval=generation_interval,
        prompt_tokens=100,
        output_tokens=output_tokens,
        max_new_tokens=max_new_tokens,
        output_valid=output_valid,
        time_to_first_token_seconds=0.2 if generation_interval is not None else None,
    )


def _serial_run() -> MageNativeRunMeasurement:
    starts = (0.5, 1.6, 2.7, 3.8, 4.9)
    return MageNativeRunMeasurement(
        run_id="serial-run",
        execution_profile="SERIAL_NATIVE_V1",
        telemetry_event_version="mage-endpoint-telemetry-v1",
        identity=_identity(),
        expected_segment_count=5,
        media_duration_seconds=40.0,
        wall_seconds=6.0,
        model_load_seconds=17.11,
        model_load_included_in_wall=False,
        telemetry=tuple(
            _sample(
                ordinal,
                generation_start=start,
                generation_end=start + 1.0,
                arm="serial",
                processor=(max(0.0, start - 0.3), start - 0.1),
            )
            for ordinal, start in enumerate(starts)
        ),
    )


def _prefetch_run() -> MageNativeRunMeasurement:
    starts = (0.10, 1.12, 2.14, 3.16, 4.18)
    processors = ((0.01, 0.08), (0.70, 0.90), (1.75, 1.95), (2.80, 3.00), (3.85, 4.05))
    return MageNativeRunMeasurement(
        run_id="prefetch-run",
        execution_profile="BOUNDED_PREFETCH_NATIVE_V1",
        telemetry_event_version="mage-endpoint-telemetry-v1",
        identity=_identity(),
        expected_segment_count=5,
        media_duration_seconds=40.0,
        wall_seconds=5.2,
        model_load_seconds=17.11,
        model_load_included_in_wall=False,
        telemetry=tuple(
            _sample(
                ordinal,
                generation_start=start,
                generation_end=start + 1.0,
                arm="prefetch",
                processor=processors[ordinal],
            )
            for ordinal, start in enumerate(starts)
        ),
    )


def test_interval_union_and_intersection_do_not_double_count() -> None:
    left = (
        MageNativeTimeInterval(start_seconds=0.0, end_seconds=2.0),
        MageNativeTimeInterval(start_seconds=1.0, end_seconds=3.0),
        MageNativeTimeInterval(start_seconds=5.0, end_seconds=6.0),
    )
    right = (
        MageNativeTimeInterval(start_seconds=1.5, end_seconds=2.5),
        MageNativeTimeInterval(start_seconds=2.5, end_seconds=5.5),
    )

    merged = merge_intervals(left)

    assert tuple((item.start_seconds, item.end_seconds) for item in merged) == (
        (0.0, 3.0),
        (5.0, 6.0),
    )
    assert interval_union_seconds(left) == pytest.approx(4.0)
    assert interval_intersection_seconds(left, right) == pytest.approx(2.0)
    assert interval_union_seconds(()) == 0.0
    assert interval_intersection_seconds((), right) == 0.0


def test_generation_gap_summary_uses_nearest_rank_percentiles() -> None:
    intervals = tuple(
        MageNativeTimeInterval(start_seconds=start, end_seconds=end)
        for start, end in ((0.0, 1.0), (1.1, 2.1), (2.3, 3.3), (4.3, 5.3))
    )

    summary = generation_gap_summary(intervals)

    assert summary.gap_count == 3
    assert summary.total_seconds == pytest.approx(1.3)
    assert summary.p50_seconds == pytest.approx(0.2)
    assert summary.p95_seconds == pytest.approx(1.0)
    assert summary.max_seconds == pytest.approx(1.0)
    assert generation_gap_summary(intervals[:1]).model_dump() == {
        "gap_count": 0,
        "total_seconds": 0.0,
        "p50_seconds": 0.0,
        "p95_seconds": 0.0,
        "max_seconds": 0.0,
    }


def test_run_summary_reports_duty_rtf_tokens_and_processor_overlap() -> None:
    run = _prefetch_run()

    summary = summarize_run(run)

    assert summary.generation_sum_seconds == pytest.approx(5.0)
    assert summary.generation_union_seconds == pytest.approx(5.0)
    assert summary.generation_overlap_seconds == pytest.approx(0.0)
    assert summary.generation_gap.p95_seconds == pytest.approx(0.02)
    assert summary.time_to_first_token.sample_count == 5
    assert summary.time_to_first_token.p50_seconds == pytest.approx(0.2)
    assert summary.time_to_first_token.p95_seconds == pytest.approx(0.2)
    assert summary.time_to_first_token.max_seconds == pytest.approx(0.2)
    assert summary.generation_duty_cycle == pytest.approx(5.0 / 5.2)
    assert summary.wall_rtf == pytest.approx(40.0 / 5.2)
    assert summary.output_tokens == 200
    assert summary.output_tokens_per_generation_second == pytest.approx(40.0)
    assert summary.output_tokens_per_wall_second == pytest.approx(200.0 / 5.2)
    assert summary.processor_union_seconds == pytest.approx(0.87)
    assert summary.processor_generation_overlap_seconds == pytest.approx(0.8)
    assert summary.processor_overlap_fraction == pytest.approx(0.8 / 0.87)
    assert summary.model_load_seconds == pytest.approx(17.11)
    assert summary.model_load_included_in_wall is False


def test_compatibility_checks_every_fair_comparison_pin() -> None:
    serial = _serial_run()
    compatible = assess_run_compatibility(serial, _prefetch_run())
    assert compatible.compatible is True
    assert compatible.mismatch_codes == ()

    changed_first = _sample(
        0,
        generation_start=0.1,
        generation_end=1.1,
        arm="prefetch",
        inference_digest=_digest(999),
        max_new_tokens=128,
    )
    incompatible_prefetch = _prefetch_run().model_copy(
        update={
            "identity": _identity(
                checkpoint_sha256=_digest(22),
                source_media_sha256=_digest(33),
                segment_manifest_sha256=_digest(44),
                prompt_sha256=_digest(55),
            ),
            "telemetry": (changed_first, *_prefetch_run().telemetry[1:]),
        }
    )

    incompatible = assess_run_compatibility(serial, incompatible_prefetch)

    assert incompatible.compatible is False
    assert set(incompatible.mismatch_codes) == {
        "CHECKPOINT_MISMATCH",
        "SOURCE_MEDIA_MISMATCH",
        "SEGMENT_MANIFEST_MISMATCH",
        "PROMPT_MISMATCH",
        "INFERENCE_IDENTITY_SEQUENCE_MISMATCH",
        "DECODER_BUDGET_MISMATCH",
    }


def test_freshness_rejects_missing_duplicate_replay_and_wrong_version_rows() -> None:
    duplicate_request = "reused-request"
    duplicate_inference = _digest(900)
    run = MageNativeRunMeasurement(
        run_id="contaminated-run",
        execution_profile="SERIAL_NATIVE_V1",
        telemetry_event_version="mage-endpoint-telemetry-v1",
        identity=_identity(),
        expected_segment_count=3,
        media_duration_seconds=24.0,
        wall_seconds=4.0,
        telemetry=(
            _sample(
                0,
                generation_start=0.1,
                generation_end=1.1,
                arm="serial",
                request_id=duplicate_request,
                inference_digest=duplicate_inference,
            ),
            _sample(
                0,
                generation_start=None,
                generation_end=None,
                arm="serial",
                request_id=duplicate_request,
                inference_digest=duplicate_inference,
                disposition=MageNativeTelemetryDisposition.ARTIFACT_REPLAY,
                telemetry_event_version="stale-version",
            ),
        ),
    )

    freshness = assess_run_freshness(run)

    assert freshness.passed is False
    assert freshness.telemetry_count == 2
    assert freshness.fresh_generation_count == 1
    assert freshness.unique_request_id_count == 1
    assert freshness.unique_inference_identity_count == 1
    assert freshness.replay_count == 1
    assert freshness.missing_generation_interval_count == 1
    assert freshness.missing_segment_ordinals == (1, 2)
    assert freshness.duplicate_segment_ordinals == (0,)
    assert set(freshness.issue_codes) == {
        "TELEMETRY_CARDINALITY_MISMATCH",
        "SEGMENT_ORDINAL_SET_MISMATCH",
        "DUPLICATE_REQUEST_ID",
        "DUPLICATE_INFERENCE_IDENTITY",
        "REPLAY_CONTAMINATION",
        "MISSING_GENERATION_INTERVAL",
        "TELEMETRY_VERSION_MISMATCH",
    }


def test_cross_run_isolation_allows_deterministic_requests_but_rejects_artifact_copy() -> None:
    serial = _serial_run()
    first_serial = serial.telemetry[0]
    first_prefetch = (
        _prefetch_run()
        .telemetry[0]
        .model_copy(
            update={
                "request_id": first_serial.request_id,
                "result_artifact_identity_sha256": first_serial.result_artifact_identity_sha256,
            }
        )
    )
    copied = _prefetch_run().model_copy(
        update={
            "run_id": serial.run_id,
            "telemetry": (first_prefetch, *_prefetch_run().telemetry[1:]),
        }
    )

    isolation = assess_cross_run_isolation(serial, copied)

    assert isolation.passed is False
    assert isolation.overlapping_request_id_count == 1
    assert isolation.overlapping_result_artifact_identity_count == 1
    assert isolation.issue_codes == (
        "RUN_ID_REUSED_BETWEEN_ARMS",
        "RESULT_ARTIFACT_REUSED_BETWEEN_ARMS",
    )


def test_passing_report_is_machine_serializable_and_locally_scoped() -> None:
    serial = _serial_run()
    prefetch = _prefetch_run()

    report = build_mage_native_sustained_comparison_report(
        serial=serial,
        prefetch=prefetch,
    )

    assert report.qualification_status is MageNativeQualificationStatus.PASSED
    assert report.production_eligible is False
    assert report.evidence_class == "LOCAL_CONFORMANCE"
    assert report.prefetch_speedup == pytest.approx(6.0 / 5.2)
    assert serial_vs_prefetch_speedup(serial, prefetch) == pytest.approx(6.0 / 5.2)
    assert all(gate.passed for gate in report.gates)

    payload = report.as_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["report_version"] == "mage-native-sustained-comparison-v1"
    assert payload["qualification_status"] == "PASSED"
    assert "PREFETCH_GENERATION_DUTY_CYCLE" in encoded


def test_report_fails_replay_output_budget_and_performance_gates() -> None:
    serial = _serial_run()
    original_prefetch = _prefetch_run()
    contaminated_first = original_prefetch.telemetry[0].model_copy(
        update={
            "disposition": MageNativeTelemetryDisposition.ARTIFACT_REPLAY,
            "generation_interval": None,
            "output_tokens": 256,
            "output_valid": False,
            "request_id": serial.telemetry[0].request_id,
            "result_artifact_identity_sha256": serial.telemetry[0].result_artifact_identity_sha256,
            "output_text_sha256": _digest(9_999),
        }
    )
    slow_prefetch = original_prefetch.model_copy(
        update={
            "wall_seconds": 8.0,
            "telemetry": (contaminated_first, *original_prefetch.telemetry[1:]),
        }
    )

    report = build_mage_native_sustained_comparison_report(
        serial=serial,
        prefetch=slow_prefetch,
        policy=MageNativeSustainedQualificationPolicy(
            minimum_prefetch_speedup=1.0,
            minimum_prefetch_generation_duty_cycle=0.9,
            minimum_prefetch_wall_rtf=6.0,
            maximum_prefetch_generation_gap_p95_seconds=0.01,
        ),
    )

    assert report.qualification_status is MageNativeQualificationStatus.FAILED
    failed = {gate.gate_id for gate in report.gates if not gate.passed}
    assert {
        "PREFETCH_FRESH_TELEMETRY",
        "CROSS_RUN_ISOLATION",
        "VALID_OUTPUTS",
        "OUTPUT_BUDGET_NOT_EXHAUSTED",
        "OUTPUT_TEXT_HASH_PARITY",
        "PREFETCH_SPEEDUP",
        "PREFETCH_GENERATION_DUTY_CYCLE",
        "PREFETCH_WALL_RTF",
        "PREFETCH_GENERATION_GAP_P95",
    } <= failed
