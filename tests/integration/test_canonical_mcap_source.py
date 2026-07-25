from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname
from uuid import NAMESPACE_URL, uuid5

import pytest

from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.adapters.mcap_single_pass import McapSinglePassH264Tee
from robata.adapters.pyav_decoder import PyAvH264DecoderProbe
from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter
from robata.adapters.sqlite_inference_evidence import MODEL_INFERENCE_SCHEMA_ID
from robata.adapters.sqlite_primary_completion import SQLitePrimaryCompletionRepository
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.application.canonical import local_composition as local_composition_module
from robata.application.canonical import mcap_source as mcap_source_module
from robata.application.canonical.bounded_media import PlannerEmission
from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
    run_local_canonical_mcap,
)
from robata.application.canonical.mcap_source import (
    CanonicalMcapSourceError,
    McapMediaProcessingPolicy,
    authorize_mcap_mapping,
    load_canonical_mcap_source,
)
from robata.application.canonical.media_quality import (
    registered_local_media_quality_report_document,
    validate_registered_local_media_quality_report_document,
)
from robata.application.canonical.pre_eos_execution import ProviderNeutralStreamStageExecutor
from robata.application.canonical.primary_completion import PrimaryCompletionEvidenceRole
from robata.application.canonical.runner import CanonicalPreEosInferenceInvocation
from robata.application.canonical.stream_scheduler import DurableStreamWindowScheduler
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRegistry
from robata.contracts.stream_common import StreamStage
from robata.inference.models import (
    InferenceStatus,
    ModelInference,
    ModelInferenceUsage,
    VisionTask,
)
from robata.inference.orchestrator import OrchestratedAttemptResult
from robata.review.routing import ReviewRoutingDisposition
from robata.runtime.observability import RuntimeProfileRecorder, RuntimeSpanStatus
from tests.support.six_camera_mcap import (
    SIX_CAMERA_MCAP_SHA256,
    SIX_CAMERA_TOPICS,
    write_six_camera_mcap,
)


def _write_mapping(path: Path) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {
                "profile_id": "canonical-mcap-integration-v1",
                "version": "canonical-mcap-integration-v1",
                "profile_kind": "TEST_FIXTURE",
                "approval_status": "UNAPPROVED",
                "approved": False,
                "mapping_policy": "EXACT_TOPIC",
                "required_schema": "foxglove.CompressedImage",
                "topics": {
                    camera_id.value: topic
                    for camera_id, topic in zip(
                        CAMERA_IDS,
                        SIX_CAMERA_TOPICS,
                        strict=True,
                    )
                },
            }
        )
    )
    return path


def test_incremental_sink_skips_scheduler_and_executor_for_empty_emissions() -> None:
    scheduler = MagicMock()
    executor = MagicMock()
    sink = mcap_source_module._IncrementalLocalStreamPlanningSink(
        scheduler=scheduler,
        executor=executor,
        runtime_observer=None,
    )
    empty_emission = MagicMock(spec=PlannerEmission)
    empty_emission.windows = ()

    sink.append_emission(empty_emission)

    scheduler.append_emission.assert_not_called()
    executor.drain_ready.assert_not_called()

    window_emission = MagicMock(spec=PlannerEmission)
    window_emission.windows = (object(),)
    executor.drain_ready.return_value = 5

    sink.append_emission(window_emission)

    scheduler.append_emission.assert_called_once_with(window_emission)
    executor.drain_ready.assert_called_once_with(max_items=5)


def test_real_mcap_builds_admitted_canonical_source_bundle(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )

    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert bundle.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert bundle.admitted_context.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert bundle.admitted_context.alignment_manifest.method.value == "mcap_log_time"
    assert tuple(bundle.frame_index.cameras.keys()) == CAMERA_IDS
    assert all(bundle.frame_index.cameras[camera_id].frames for camera_id in CAMERA_IDS)
    resolved = tuple(
        bundle.resolve_artifact(camera_id, bundle.frame_index.cameras[camera_id].frames[0])
        for camera_id in CAMERA_IDS
    )
    assert all(artifact is not None for artifact in resolved)
    assert all(artifact.artifact.media_type == "image/png" for artifact in resolved if artifact)
    report_path = tmp_path / "media-state" / "media-quality-report.json"
    registry = SchemaRegistry()
    report_document = registered_local_media_quality_report_document(
        bundle.media_quality_report,
        registry,
    )
    validate_registered_local_media_quality_report_document(report_document, registry)
    assert report_path.read_bytes() == canonical_json_bytes(report_document)
    supplemental = report_document["supplemental_targets"]
    assert isinstance(supplemental, dict)
    assert supplemental["candidate_count"] > 0
    assert supplemental["targets"]
    assert all(target["provenance"] for target in supplemental["targets"])
    assert "dropped_by_per_camera_budget" in supplemental
    assert "dropped_by_total_budget" in supplemental


def test_canonical_source_uses_one_spool_tee_and_no_legacy_media_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    state_dir = tmp_path / "media-state"
    tee_calls = 0
    original_traverse = McapSinglePassH264Tee.traverse

    def observed_traverse(
        tee: McapSinglePassH264Tee,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal tee_calls
        tee_calls += 1
        return original_traverse(tee, *args, **kwargs)  # type: ignore[arg-type]

    def reject_legacy_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("legacy per-camera source read must not run")

    monkeypatch.setattr(McapSinglePassH264Tee, "traverse", observed_traverse)
    monkeypatch.setattr(
        OfficialMcapInspector,
        "inspect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("canonical source must not run the legacy full inspector")
        ),
    )
    monkeypatch.setattr(PyAvH264DecoderProbe, "probe", reject_legacy_read)
    monkeypatch.setattr(PyAvH264Mp4Exporter, "export", reject_legacy_read)

    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert tee_calls == 1
    assert len(CAMERA_IDS) == mcap_source_module.MCAP_SPOOL_EXPORT_WORKERS
    assert not (state_dir / "h264-spools").exists()
    probe_facts = bundle.admitted_context.validation_report.probed_stream_facts
    assert len(probe_facts) == len(CAMERA_IDS)
    assert all(
        fact.decoder_probe.probe.name == "pyav-h264-remux-decode-validation"
        and fact.decoder_probe.decoded_frame_count == 2
        and fact.decoder_probe.decoded_width > 0
        and fact.decoder_probe.decoded_height > 0
        for fact in probe_facts
    )


def test_fresh_visual_fusion_avoids_mp4_redecode_and_replay_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    state_dir = tmp_path / "media-state"

    def reject_redecode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fresh visual fusion must not decode the exported MP4 again")

    original_materialize = mcap_source_module._decode_selected_camera_frames
    monkeypatch.setattr(mcap_source_module, "_decode_selected_camera_frames", reject_redecode)
    monkeypatch.setattr(PyAvH264Mp4Exporter, "_validate_exported_mp4", reject_redecode)

    fresh = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert fresh._artifacts
    assert all(
        len(ledger.decoded_observations) > 0 for ledger in fresh.media_quality_report.camera_ledgers
    )

    decode_calls = 0

    def observe_replay_decode(*args: object, **kwargs: object) -> object:
        nonlocal decode_calls
        decode_calls += 1
        return original_materialize(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        mcap_source_module,
        "_decode_selected_camera_frames",
        observe_replay_decode,
    )
    replay = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert decode_calls == len(CAMERA_IDS)
    assert replay.frame_index == fresh.frame_index
    assert replay.media_quality_report == fresh.media_quality_report


def test_canonical_mcap_composes_and_exactly_recovers_stream_scheduler(
    tmp_path: Path,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    state_dir = tmp_path / "media-state"
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    stream_run_id = "00000000-0000-0000-0000-000000000321"

    def clock() -> datetime:
        return datetime(2026, 7, 20, tzinfo=UTC)

    first = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=clock,
        execution_scheduler=scheduler,
        stream_run_id=stream_run_id,
    )
    recovered = DurableStreamWindowScheduler.recover_registered(
        execution_scheduler=scheduler,
        stream_run_id=stream_run_id,
        clock=clock,
    )
    assert len(recovered) == 1
    stream = recovered[0]
    assert stream.declarations()
    assert stream.expected_plan_seal() is not None
    assert stream.export_barrier().complete
    assert stream.backlog(now=clock()).finalization_published is False

    with sqlite3.connect(scheduler.database_path) as connection:
        before = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "capture_authority_receipts",
                "stream_plans",
                "expected_windows",
                "stream_work_plans",
                "work_items",
            )
        )

    second = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=clock,
        execution_scheduler=SQLiteWorkScheduler(scheduler.database_path),
        stream_run_id=stream_run_id,
    )
    with sqlite3.connect(scheduler.database_path) as connection:
        after = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "capture_authority_receipts",
                "stream_plans",
                "expected_windows",
                "stream_work_plans",
                "work_items",
            )
        )

    assert second.source_content_sha256 == first.source_content_sha256
    assert second.admitted_context == first.admitted_context
    assert after == before


def test_mcap_source_injects_pre_eos_stage_executor_into_incremental_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    eos_finalized = Event()
    observed_stages: list[StreamStage] = []
    original_finalize_eos = DurableStreamWindowScheduler.finalize_eos
    finalizer_factory = MagicMock(
        side_effect=mcap_source_module.LocalConformanceStreamFinalizer
    )

    def stage_terminal_executor(plan: object) -> None:
        assert not eos_finalized.is_set()
        assert isinstance(plan, mcap_source_module.StreamWorkItemPlan)
        observed_stages.append(plan.stage)
        # Returning None leaves the local conformance terminal selectable. The
        # P5 provider-neutral executor returns a typed terminal for QA/event work.
        return None

    def observed_finalize_eos(
        scheduler: DurableStreamWindowScheduler,
        inputs: object,
    ) -> object:
        result = original_finalize_eos(scheduler, inputs)  # type: ignore[arg-type]
        eos_finalized.set()
        return result

    monkeypatch.setattr(
        mcap_source_module,
        "LocalConformanceStreamFinalizer",
        finalizer_factory,
    )
    monkeypatch.setattr(
        DurableStreamWindowScheduler,
        "finalize_eos",
        observed_finalize_eos,
    )

    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        execution_scheduler=SQLiteWorkScheduler(tmp_path / "work.sqlite3"),
        stream_run_id="00000000-0000-0000-0000-000000000323",
        stage_terminal_executor=stage_terminal_executor,
    )

    assert bundle.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert eos_finalized.is_set()
    assert {
        StreamStage.QA_COARSE,
        StreamStage.QA_DENSE,
        StreamStage.EVENT_PROPOSAL,
    }.issubset(observed_stages)
    finalizer_factory.assert_called_once()
    assert (
        finalizer_factory.call_args.kwargs["stage_terminal_executor"]
        is stage_terminal_executor
    )
    schema_refs = finalizer_factory.call_args.kwargs["schema_refs"]
    assert schema_refs.model_inference is not None
    assert schema_refs.model_inference.schema_id == MODEL_INFERENCE_SCHEMA_ID
    assert schema_refs.model_inference.version == "1.0.0"



def _pre_eos_fixture_model(task: VisionTask) -> ModelInference:
    """Return a strict terminal shaped like the canonical evidence ledger output."""

    ordinal = {
        VisionTask.QA_COARSE: 1,
        VisionTask.QA_DENSE: 2,
        VisionTask.EVENT_PROPOSAL: 3,
    }[task]
    return ModelInference(
        schema_version="1.0",
        inference_id=str(uuid5(NAMESPACE_URL, f"p5-fixture-inference:{task.value}")),
        logical_invocation_id=str(
            uuid5(NAMESPACE_URL, f"p5-fixture-logical-invocation:{task.value}")
        ),
        request_id=str(uuid5(NAMESPACE_URL, f"p5-fixture-request:{task.value}")),
        idempotency_key=f"p5-fixture-idempotency-{task.value}",
        mcap_id="00000000-0000-0000-0000-000000000401",
        package_set_id="00000000-0000-0000-0000-000000000402",
        package_id=None,
        package_ids=(f"p5-fixture-package-{ordinal}",),
        camera_mapping_run_id="00000000-0000-0000-0000-000000000403",
        alignment_id="00000000-0000-0000-0000-000000000404",
        start_ns=0,
        end_ns=1_000_000_000,
        stage=task,
        provider="delayed-pre-eos-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="fixture-adapter-v1",
        prompt_version="fixture-prompt-v1",
        prompt_artifact_id="p5-fixture-prompt",
        prompt_sha256=f"{500 + ordinal:064x}",
        rendered_input_digest=f"{510 + ordinal:064x}",
        input_plan_id=None,
        input_plan_semantic_sha256=None,
        input_plan_part_ordinal=None,
        input_plan_part_count=None,
        input_plan_part_semantic_sha256=None,
        output_schema_id="p5-fixture-output",
        output_schema_version="1.0",
        output_schema_artifact_id="p5-fixture-output-schema",
        output_schema_sha256=f"{520 + ordinal:064x}",
        capability_snapshot_id="00000000-0000-0000-0000-000000000405",
        capability_snapshot_digest=f"{530 + ordinal:064x}",
        input_manifest_set_sha256=f"{540 + ordinal:064x}",
        input_config={"input_images": 1, "payload_bytes": 100},
        sampling_config={"policy": "p5-pre-eos-fixture-v1"},
        generation_config={"temperature": 0},
        provider_idempotency_key=f"p5-fixture-provider-idempotency-{task.value}",
        provider_request_id=f"p5-fixture-provider-request-{task.value}",
        experiment_id=None,
        shadow_route_id=None,
        primary_inference_id=None,
        shadow=False,
        attempt=1,
        retry_count=0,
        status=InferenceStatus.SUCCEEDED,
        queued_at="2026-07-20T00:00:00Z",
        started_at="2026-07-20T00:00:00Z",
        completed_at="2026-07-20T00:00:00Z",
        latency_ms=1,
        raw_output={"fixture": task.value},
        normalized_output={"label": task.value},
        output_valid=True,
        reported_confidence=None,
        calibrated_confidence=None,
        usage=ModelInferenceUsage(input_frames=1, input_images=1),
        failure=None,
        created_at="2026-07-20T00:00:00Z",
    )


def test_mcap_pre_eos_provider_neutral_executor_commits_typed_terminals_before_eos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    eos_finalized = Event()
    observed_stages: list[VisionTask] = []
    schema_ref = SchemaRegistry().resolve_version(MODEL_INFERENCE_SCHEMA_ID, "1.0.0").ref

    class DelayedFixturePipeline:
        async def execute_pre_eos_inference(
            self,
            invocation: CanonicalPreEosInferenceInvocation,
        ) -> OrchestratedAttemptResult:
            assert not eos_finalized.is_set()
            await asyncio.sleep(0)
            observed_stages.append(invocation.task)
            terminal = _pre_eos_fixture_model(invocation.task)
            return OrchestratedAttemptResult(
                terminal=terminal,
                selection=SimpleNamespace(
                    inference_id=terminal.inference_id,
                    logical_invocation_id=terminal.logical_invocation_id,
                ),
            )

    task_by_stage = {
        StreamStage.QA_COARSE: VisionTask.QA_COARSE,
        StreamStage.QA_DENSE: VisionTask.QA_DENSE,
        StreamStage.EVENT_PROPOSAL: VisionTask.EVENT_PROPOSAL,
    }

    def invocation_factory(
        plan: mcap_source_module.StreamWorkItemPlan,
    ) -> CanonicalPreEosInferenceInvocation | None:
        task = task_by_stage.get(plan.stage)
        if task is None:
            return None
        return CanonicalPreEosInferenceInvocation(
            task=task,
            mcap_id="00000000-0000-0000-0000-000000000401",
            camera_mapping_run_id="00000000-0000-0000-0000-000000000403",
            alignment_id="00000000-0000-0000-0000-000000000404",
            start_ns=0,
            end_ns=1_000_000_000,
            package_set_id="00000000-0000-0000-0000-000000000402",
            rendered_input_digest=f"{600 + len(observed_stages):064x}",
            input_config={"input_images": 1, "payload_bytes": 100},
            sampling_config={"policy": "p5-pre-eos-fixture-v1"},
            metadata={"stream_stage": task.value},
        )

    executor = ProviderNeutralStreamStageExecutor(
        pipeline=DelayedFixturePipeline(),
        invocation_factory=invocation_factory,
        artifact_root=tmp_path / "stream-artifacts",
        model_inference_schema_ref=schema_ref,
        terminal_policy_version="stream-terminal-policy-v1",
    )
    original_finalize_eos = DurableStreamWindowScheduler.finalize_eos

    def observed_finalize_eos(
        scheduler: DurableStreamWindowScheduler,
        inputs: object,
    ) -> object:
        result = original_finalize_eos(scheduler, inputs)  # type: ignore[arg-type]
        eos_finalized.set()
        return result

    monkeypatch.setattr(
        DurableStreamWindowScheduler,
        "finalize_eos",
        observed_finalize_eos,
    )
    scheduler_store = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        execution_scheduler=scheduler_store,
        stream_run_id="00000000-0000-0000-0000-000000000324",
        stream_artifact_root=tmp_path / "stream-artifacts",
        stage_terminal_executor=executor,
    )

    assert bundle.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert eos_finalized.is_set()
    assert observed_stages == [
        VisionTask.QA_COARSE,
        VisionTask.QA_DENSE,
        VisionTask.EVENT_PROPOSAL,
    ]
    (scheduler,) = DurableStreamWindowScheduler.recover_registered(
        execution_scheduler=SQLiteWorkScheduler(scheduler_store.database_path),
        stream_run_id="00000000-0000-0000-0000-000000000324",
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    typed_terminals = tuple(
        item.terminal_evidence
        for item in scheduler.work_items()
        if item.stage in task_by_stage
    )
    assert all(terminal is not None for terminal in typed_terminals)
    assert all(
        terminal.evidence_ref.schema_ref == schema_ref
        for terminal in typed_terminals
        if terminal is not None
    )
    assert all(
        ModelInference.model_validate_json(
            executor.artifact_path_for(terminal.evidence_ref).read_bytes(),
            strict=True,
        ).output_valid
        for terminal in typed_terminals
        if terminal is not None
    )


def test_sealed_spool_export_advances_while_eos_is_finalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    export_started = Event()
    eos_finalized = Event()
    export_service = mcap_source_module.RegisteredSixCameraVideoExportService
    original_export = export_service.export_staged_local
    original_finalize_eos = DurableStreamWindowScheduler.finalize_eos

    def observed_export(
        service: object,
        request: object,
        producer: object,
    ) -> object:
        export_started.set()
        assert eos_finalized.wait(timeout=5)
        return original_export(service, request, producer)  # type: ignore[arg-type]

    def observed_finalize_eos(
        scheduler: DurableStreamWindowScheduler,
        inputs: object,
    ) -> object:
        assert export_started.wait(timeout=5)
        result = original_finalize_eos(scheduler, inputs)  # type: ignore[arg-type]
        eos_finalized.set()
        return result

    monkeypatch.setattr(export_service, "export_staged_local", observed_export)
    monkeypatch.setattr(
        DurableStreamWindowScheduler,
        "finalize_eos",
        observed_finalize_eos,
    )

    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        execution_scheduler=SQLiteWorkScheduler(tmp_path / "work.sqlite3"),
        stream_run_id="00000000-0000-0000-0000-000000000322",
    )

    assert bundle.source_content_sha256 == SIX_CAMERA_MCAP_SHA256
    assert export_started.is_set()
    assert eos_finalized.is_set()


def test_runtime_observation_preserves_canonical_source_facts(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    state_dir = tmp_path / "media-state"
    baseline = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    recorder = RuntimeProfileRecorder()

    observed = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        runtime_observer=recorder,
    )
    snapshot = recorder.snapshot()

    assert observed.source_content_sha256 == baseline.source_content_sha256
    assert observed.admitted_context == baseline.admitted_context
    assert observed.requested_interval == baseline.requested_interval
    assert observed.sampling_plan == baseline.sampling_plan
    assert observed.frame_index == baseline.frame_index
    assert observed.media_quality_report == baseline.media_quality_report
    assert dict(observed._artifacts) == dict(baseline._artifacts)
    source_spans = tuple(span for span in snapshot.spans if span.parent_sequence is None)
    assert tuple(span.name for span in source_spans) == (
        "source.inspect",
        "source.mapping.resolve",
        "source.stream.capture_publish",
        "source.video.publication_validate",
        "source.video.ledger_load",
        "source.metadata.build",
        "source.frame_index",
        "source.quality.timing",
        "source.materialize",
        "source.quality.report",
        "source.quality.publish",
    )
    assert all(span.status is RuntimeSpanStatus.OK for span in snapshot.spans)
    assert any(
        span.name == "sqlite.artifact_registry.transaction" and span.parent_sequence is not None
        for span in snapshot.spans
    )
    assert all(
        {attribute.name: attribute.value for attribute in span.attributes}["camera_count"]
        == len(CAMERA_IDS)
        for span in source_spans
    )

    counters = {
        counter.name: counter.value
        for counter in snapshot.counters
        if counter.name.startswith("source.")
    }
    indexed_frame_count = sum(
        len(observed.frame_index.cameras[camera_id].frames) for camera_id in CAMERA_IDS
    )
    quality_observation_count = sum(
        len(ledger.decoded_observations) for ledger in observed.media_quality_report.camera_ledgers
    )
    source_timestamps = tuple(
        frame.source_timestamp_ns
        for camera_id in CAMERA_IDS
        for frame in observed.frame_index.cameras[camera_id].frames
    )
    assert counters == {
        "source.frame_observations": quality_observation_count,
        "source.frame_index.frames": indexed_frame_count,
        "source.materialized_artifacts": len(observed._artifacts),
        "source.message_count": indexed_frame_count,
        "source.recording_duration_ns": (
            observed.admitted_context.ready_manifest.recording.duration_ns
        ),
        "source.requested_duration_ns": observed.requested_interval.duration_ns,
        "source.span_duration_ns": max(source_timestamps) - min(source_timestamps),
    }
    assert (
        sum(
            counter.value
            for counter in snapshot.counters
            if counter.name == "sqlite.artifact_registry.transactions"
        )
        > 0
    )


def test_cache_miss_lazily_materializes_exact_verified_frame(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    recorder = RuntimeProfileRecorder()
    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        runtime_observer=recorder,
    )
    camera_id = CAMERA_IDS[1]
    frame = bundle.frame_index.cameras[camera_id].frames[-1]
    key = (camera_id, frame.source_frame_id)
    expected = bundle.resolve_artifact(camera_id, frame)
    assert expected is not None

    bundle._artifact_resolver.artifacts.pop(key)
    actual = bundle.resolve_artifact(camera_id, frame)

    assert actual == expected
    assert bundle._artifacts[key] == actual
    assert bundle.resolve_artifact(camera_id, frame) is actual
    tampered = frame.model_copy(update={"source_timestamp_ns": frame.source_timestamp_ns + 1})
    with pytest.raises(CanonicalMcapSourceError, match="canonical frame index"):
        bundle.resolve_artifact(camera_id, tampered)
    snapshot = recorder.snapshot()
    lazy_spans = tuple(span for span in snapshot.spans if span.name == "source.lazy_materialize")
    assert len(lazy_spans) == 1
    assert all(span.status is RuntimeSpanStatus.OK for span in lazy_spans)
    assert any(span.name == "sqlite.artifact_registry.transaction" for span in snapshot.spans)
    resolver_counts = {
        next(
            attribute.value for attribute in counter.attributes if attribute.name == "cache"
        ): counter.value
        for counter in snapshot.counters
        if counter.name == "source.artifact_resolver.requests"
    }
    assert resolver_counts == {"HIT": 2, "MISS": 1}
    assert (
        sum(
            counter.value
            for counter in snapshot.counters
            if counter.name == "source.lazy_materialized_artifacts"
        )
        == 1
    )


def test_max_duration_clamps_half_open_source_and_quality_window(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    recorder = RuntimeProfileRecorder()
    bundle = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        max_duration_ns=250_000_000,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        runtime_observer=recorder,
    )

    assert bundle.requested_interval.start_ns == 0
    assert bundle.requested_interval.end_ns == 250_000_000
    assert bundle.media_quality_report.window_limited is True
    assert bundle.media_quality_report.requested_max_duration_ns == 250_000_000
    assert all(
        len(ledger.decoded_observations) == 1
        for ledger in bundle.media_quality_report.camera_ledgers
    )
    for camera_id in CAMERA_IDS:
        outside = bundle.frame_index.cameras[camera_id].frames[-1]
        assert outside.alignment_projection.aligned_timestamp_ns >= bundle.requested_interval.end_ns
        assert bundle.resolve_artifact(camera_id, outside) is None
    materialize_span = next(
        span for span in recorder.snapshot().spans if span.name == "source.materialize"
    )
    assert {attribute.name: attribute.value for attribute in materialize_span.attributes} == {
        "camera_count": len(CAMERA_IDS),
        "max_duration_limited": True,
        "media_processing_policy_version": (
            mcap_source_module.DEFAULT_MCAP_MEDIA_PROCESSING_POLICY.version
        ),
        "semantic_target_rate": "2/1",
        "sentinel_rate": "2/1",
        "target_selection_tolerance_ns": 300_000_000,
        "sentinel_analysis_width": 64,
        "evidence_encoding": "png",
        "evidence_max_width": 320,
        "evidence_extractor_version": "canonical-mcap-png-320-v1",
    }


def test_media_processing_policy_bounds_sentinels_and_preserves_semantic_targets(
    tmp_path: Path,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    authorization = authorize_mcap_mapping(
        _write_mapping(tmp_path / "mapping.json"),
        allow_unapproved_profile=True,
    )
    baseline = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "baseline-media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    recorder = RuntimeProfileRecorder()
    policy = McapMediaProcessingPolicy(
        version="canonical-mcap-media-processing-test-v1",
        sentinel_rate_numerator=1,
        sentinel_analysis_width=8,
        evidence_max_width=8,
        evidence_extractor_version="canonical-mcap-png-8-test-v1",
    )
    bounded = load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=tmp_path / "bounded-media-state",
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        media_processing_policy=policy,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        runtime_observer=recorder,
    )

    assert bounded.frame_index == baseline.frame_index
    assert bounded.sampling_plan == baseline.sampling_plan
    assert {(camera_id, source_frame_id) for camera_id, source_frame_id in bounded._artifacts} == {
        (camera_id, source_frame_id) for camera_id, source_frame_id in baseline._artifacts
    }
    assert all(
        artifact.width <= policy.evidence_max_width for artifact in bounded._artifacts.values()
    )
    assert all(
        len(ledger.decoded_observations) == 1
        for ledger in bounded.media_quality_report.camera_ledgers
    )
    assert all(
        len(ledger.decoded_observations) == 2
        for ledger in baseline.media_quality_report.camera_ledgers
    )
    materialize_span = next(
        span for span in recorder.snapshot().spans if span.name == "source.materialize"
    )
    attributes = {attribute.name: attribute.value for attribute in materialize_span.attributes}
    assert attributes["media_processing_policy_version"] == policy.version
    assert attributes["sentinel_rate"] == "1/1"
    assert attributes["sentinel_analysis_width"] == 8
    assert attributes["evidence_encoding"] == "png"
    assert attributes["evidence_max_width"] == 8
    assert attributes["evidence_extractor_version"] == policy.evidence_extractor_version


def test_existing_quality_report_bytes_fail_closed(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "media-state"
    authorization = authorize_mcap_mapping(
        mapping,
        allow_unapproved_profile=True,
    )
    load_canonical_mcap_source(
        source,
        authorization=authorization,
        state_dir=state_dir,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    (state_dir / "media-quality-report.json").write_bytes(b"{}")

    with pytest.raises(
        CanonicalMcapSourceError,
        match="existing media quality report bytes are inconsistent",
    ):
        load_canonical_mcap_source(
            source,
            authorization=authorization,
            state_dir=state_dir,
            expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
            clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
        )


def test_real_mcap_command_commits_and_exactly_replays(tmp_path: Path) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "canonical-state"

    first = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key="real-mcap-replay",
        allow_unapproved_profile=True,
    )
    replay = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key="real-mcap-replay",
        allow_unapproved_profile=True,
    )

    assert first.replayed is False
    assert first.fixture_inference_calls > 0
    assert first.event_ids
    assert first.revision_ids
    assert first.outbox_ids
    assert first.production_eligible is False
    assert first.network_call_count == 0
    assert first.media_quality_binding is not None
    assert first.media_quality_binding.requires_review is True
    assert first.supplemental_qa_evidence is not None
    assert first.supplemental_qa_evidence.schema_version == "2.0"
    assert first.supplemental_qa_evidence.production_eligible is False
    assert first.review_routing.disposition is ReviewRoutingDisposition.ENQUEUED
    assert replay.replayed is True
    assert replay.fixture_inference_calls == 0
    assert replay.run_id == first.run_id
    assert replay.command_sha256 == first.command_sha256
    assert replay.completion_semantic_sha256 == first.completion_semantic_sha256
    assert replay.event_ids == first.event_ids
    assert replay.revision_ids == first.revision_ids
    assert replay.outbox_ids == first.outbox_ids
    assert replay.media_quality_binding == first.media_quality_binding
    assert replay.supplemental_qa_evidence == first.supplemental_qa_evidence
    assert replay.review_routing.disposition is ReviewRoutingDisposition.ALREADY_ENQUEUED
    committed = SQLitePrimaryCompletionRepository(
        state_dir / "primary-completion.sqlite3",
        registry=SchemaRegistry(),
    ).get(first.run_id)
    assert committed is not None
    assert tuple(reference.role for reference in committed.evidence_references) == (
        PrimaryCompletionEvidenceRole.MEDIA_QUALITY_REPORT,
        PrimaryCompletionEvidenceRole.STREAM_RECORDING_RESULT,
        PrimaryCompletionEvidenceRole.SUPPLEMENTAL_QA_EVIDENCE,
    )
    stream_recording_reference = next(
        reference
        for reference in committed.evidence_references
        if reference.role is PrimaryCompletionEvidenceRole.STREAM_RECORDING_RESULT
    )
    assert stream_recording_reference.schema_ref.schema_id == (
        "https://schemas.robata.dev/local-stream-recording-result"
    )
    assert stream_recording_reference.schema_ref.version == "4.0.0"
    assert all(reference.byte_count > 0 for reference in committed.evidence_references)
    with sqlite3.connect(state_dir / "work-scheduler.sqlite3") as connection:
        expected_count = connection.execute("SELECT COUNT(*) FROM expected_windows").fetchone()[0]
        result_count = connection.execute("SELECT COUNT(*) FROM stream_window_results").fetchone()[
            0
        ]
        finalization_count = connection.execute(
            "SELECT COUNT(*) FROM recording_finalizations"
        ).fetchone()[0]
        outbox_count, delivered_count = connection.execute(
            """
            SELECT COUNT(*), COUNT(delivered_at)
            FROM stream_delivery_outbox
            """
        ).fetchone()
    with sqlite3.connect(state_dir / "stream-outbox-sink.sqlite3") as connection:
        sink_count = connection.execute(
            "SELECT COUNT(*) FROM delivered_outbox_messages"
        ).fetchone()[0]
    assert expected_count > 0
    assert result_count == expected_count
    assert finalization_count == 1
    assert outbox_count == expected_count + 1
    assert delivered_count == outbox_count
    assert sink_count == outbox_count
    assert not tuple(state_dir.rglob("*.h264.spool"))


def test_recovered_mcap_completion_rejects_tampered_quality_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    mapping = _write_mapping(tmp_path / "mapping.json")
    state_dir = tmp_path / "canonical-state"
    run_key = "tampered-quality-recovery"

    first = run_local_canonical_mcap(
        source,
        mapping,
        state_dir,
        run_key=run_key,
        allow_unapproved_profile=True,
    )
    assert first.media_quality_binding is not None
    reports = tuple((state_dir / "mcap").glob("*/media-quality-report.json"))
    assert len(reports) == 1
    report_bytes = reports[0].read_bytes()
    reports[0].write_bytes(report_bytes + b"\n")
    reconciliation_calls: list[str] = []
    original_reconcile = local_composition_module._reconcile_local_outbox

    def observed_reconcile(*args: object, **kwargs: object):
        reconciliation_calls.append("outbox")
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        local_composition_module,
        "_reconcile_local_outbox",
        observed_reconcile,
    )

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_mcap(
            source,
            mapping,
            state_dir,
            run_key=run_key,
            allow_unapproved_profile=True,
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    assert "exact canonical JSON" in str(caught.value)
    assert reconciliation_calls == ["outbox"]

    reports[0].write_bytes(report_bytes)
    committed = SQLitePrimaryCompletionRepository(
        state_dir / "primary-completion.sqlite3",
        registry=SchemaRegistry(),
    ).get(first.run_id)
    assert committed is not None
    stream_reference = next(
        reference
        for reference in committed.evidence_references
        if reference.role is PrimaryCompletionEvidenceRole.STREAM_RECORDING_RESULT
    )
    stream_artifact_path = (
        state_dir
        / "stream-artifacts"
        / stream_reference.exact_bytes_sha256[:2]
        / f"{stream_reference.exact_bytes_sha256}.json"
    )
    stream_artifact_bytes = stream_artifact_path.read_bytes()
    stream_artifact_path.write_bytes(stream_artifact_bytes + b"\n")

    with pytest.raises(CanonicalLocalCompositionError) as caught_stream_artifact:
        run_local_canonical_mcap(
            source,
            mapping,
            state_dir,
            run_key=run_key,
            allow_unapproved_profile=True,
        )

    assert (
        caught_stream_artifact.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    )
    assert "stream artifact bytes do not match" in str(caught_stream_artifact.value)
    stream_artifact_path.write_bytes(stream_artifact_bytes)
    assert first.supplemental_qa_evidence is not None
    selected = next(
        outcome.selected_artifact
        for outcome in first.supplemental_qa_evidence.package.outcomes
        if outcome.selected_artifact is not None
    )
    artifact_path = Path(url2pathname(unquote(urlsplit(selected.artifact.uri).path)))
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(CanonicalLocalCompositionError) as caught_artifact:
        run_local_canonical_mcap(
            source,
            mapping,
            state_dir,
            run_key=run_key,
            allow_unapproved_profile=True,
        )

    assert caught_artifact.value.code is CanonicalLocalCompositionErrorCode.LOCAL_STATE_FAILED
    assert "artifact byte count is inconsistent" in str(caught_artifact.value)


def test_corrupt_mcap_fails_before_completion(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.mcap"
    source.write_bytes(write_six_camera_mcap(tmp_path / "valid.mcap").read_bytes()[:256])

    with pytest.raises(CanonicalLocalCompositionError) as caught:
        run_local_canonical_mcap(
            source,
            _write_mapping(tmp_path / "mapping.json"),
            tmp_path / "canonical-state",
            run_key="corrupt-input",
            allow_unapproved_profile=True,
        )

    assert caught.value.code is CanonicalLocalCompositionErrorCode.SOURCE_INVALID
