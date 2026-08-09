from __future__ import annotations

import pytest

from robata.runtime.e2e_trace import (
    E2ETraceFragmentRole,
    E2ETraceMeasurementStatus,
    E2ETraceStage,
    build_e2e_trace_runtime_fragment,
    summarize_e2e_trace_stages,
)
from robata.runtime.observability import (
    ProcessResourceSample,
    RuntimeProfileRecorder,
    RuntimeResourceMeasurement,
    RuntimeResourceStatus,
    runtime_span,
)


def _available(value: int) -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(status=RuntimeResourceStatus.AVAILABLE, value=value)


def _resource_sample() -> ProcessResourceSample:
    return ProcessResourceSample(
        rss_bytes=_available(1024),
        read_bytes=_available(0),
        write_bytes=_available(0),
    )


def test_stage_summary_uses_interval_union_and_explicit_not_measured_states() -> None:
    clock_values = iter((0, 0, 10, 20, 30, 40, 50, 50))
    cpu_values = iter((0, 0, 1, 2, 3, 4, 5, 5))
    recorder = RuntimeProfileRecorder(
        clock_ns=lambda: next(clock_values),
        process_cpu_clock_ns=lambda: next(cpu_values),
        resource_sampler=_resource_sample,
    )

    with runtime_span(recorder, "inference.parent"), runtime_span(recorder, "inference.child"):
        pass
    with runtime_span(recorder, "source.decode"):
        pass

    summaries, unclassified = summarize_e2e_trace_stages(recorder.snapshot())
    by_stage = {item.stage: item for item in summaries}

    inference = by_stage[E2ETraceStage.INFERENCE]
    assert inference.measurement_status is E2ETraceMeasurementStatus.MEASURED
    assert inference.observed_span_count == 2
    assert inference.wall_time_union_ns == 30
    assert inference.inclusive_span_time_ns == 40
    assert (
        by_stage[E2ETraceStage.EVIDENCE].measurement_status
        is E2ETraceMeasurementStatus.NOT_MEASURED
    )
    assert by_stage[E2ETraceStage.EVIDENCE].wall_time_union_ns is None
    assert unclassified == 0


def test_runtime_fragment_rejects_uncovered_spans() -> None:
    recorder = RuntimeProfileRecorder(resource_sampler=_resource_sample)
    with runtime_span(recorder, "unknown.boundary"):
        pass
    fragment = build_e2e_trace_runtime_fragment(
        role=E2ETraceFragmentRole.LAUNCHER,
        runtime_profile=recorder.snapshot(),
    )
    assert fragment.unclassified_span_count == 1
    assert all(
        stage.measurement_status is E2ETraceMeasurementStatus.NOT_MEASURED
        for stage in fragment.stages
    )


def test_not_measured_stage_rejects_fake_zero_timing() -> None:
    summaries, _ = summarize_e2e_trace_stages(
        RuntimeProfileRecorder(resource_sampler=_resource_sample).snapshot()
    )
    assert all(item.wall_time_union_ns is None for item in summaries)
    with pytest.raises(ValueError, match="NOT_MEASURED"):
        type(summaries[0])(
            stage=E2ETraceStage.SOURCE,
            measurement_status=E2ETraceMeasurementStatus.NOT_MEASURED,
            observed_span_count=0,
            wall_time_union_ns=0,
        )


def test_postgres_operation_family_is_attributed_without_raw_operation_name() -> None:
    recorder = RuntimeProfileRecorder(resource_sampler=_resource_sample)

    with runtime_span(
        recorder,
        "postgres.authority.transaction",
        {"operation_family": "SCHEDULING", "write": True},
    ):
        pass

    summaries, unclassified = summarize_e2e_trace_stages(recorder.snapshot())
    scheduling = {item.stage: item for item in summaries}[E2ETraceStage.SCHEDULING]

    assert scheduling.measurement_status is E2ETraceMeasurementStatus.MEASURED
    assert scheduling.observed_span_count == 1
    assert unclassified == 0


def test_local_canonical_operation_names_are_classified_by_semantic_stage() -> None:
    recorder = RuntimeProfileRecorder(resource_sampler=_resource_sample)
    names = (
        "canonical.composition",
        "sqlite.work_scheduler.transaction",
        "sqlite.barrier.transaction",
        "sqlite.inference_evidence.transaction",
        "completion.storage.open",
        "completion.run.begin",
    )
    for name in names:
        with runtime_span(recorder, name):
            pass

    summaries, unclassified = summarize_e2e_trace_stages(recorder.snapshot())
    by_stage = {item.stage: item for item in summaries}

    assert by_stage[E2ETraceStage.ORCHESTRATION].observed_span_count == 2
    assert by_stage[E2ETraceStage.SCHEDULING].observed_span_count == 2
    assert by_stage[E2ETraceStage.EVIDENCE].observed_span_count == 1
    assert by_stage[E2ETraceStage.PUBLICATION].observed_span_count == 1
    assert unclassified == 0


def test_perception_vnext_spans_are_classified_by_physical_capability() -> None:
    recorder = RuntimeProfileRecorder(resource_sampler=_resource_sample)
    names = (
        "perception.media_scan",
        "perception.observe",
        "perception.project",
        "perception.temporal_reconcile",
        "perception.fusion",
        "perception.refine",
        "perception.finalize",
    )
    for name in names:
        with runtime_span(recorder, name):
            pass

    summaries, unclassified = summarize_e2e_trace_stages(recorder.snapshot())
    by_stage = {item.stage: item for item in summaries}

    assert by_stage[E2ETraceStage.SOURCE].observed_span_count == 1
    assert by_stage[E2ETraceStage.INFERENCE].observed_span_count == 2
    assert by_stage[E2ETraceStage.EVIDENCE].observed_span_count == 1
    assert by_stage[E2ETraceStage.REDUCTION].observed_span_count == 3
    assert unclassified == 0
