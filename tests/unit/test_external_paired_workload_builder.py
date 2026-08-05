from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from robata.benchmark.external_paired_qualification import ExternalPairedWorkloadManifest
from robata.benchmark.external_paired_workload_builder import (
    ExternalPairedTargetConfig,
    ExternalPairedWorkloadBuilderError,
    ExternalPairedWorkloadSourceConfig,
    build_external_paired_workload,
    write_external_paired_workload,
)
from robata.benchmark.local_real_model_e2e import (
    LocalFrameArtifact,
    LocalModelObservation,
    LocalRealModelE2EReport,
    LocalStorageObservation,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.adapter import JsonSchemaRef
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.models import InputMode, VisionTask
from robata.inference.orchestrator import InferencePolicy
from robata.inference.routing import ExperimentInputRepresentation, ExperimentIsolationProfile
from robata.runtime.e2e_participation import (
    E2EParticipationBoundary,
    E2EParticipationState,
    build_e2e_participation_manifest,
    write_e2e_participation_manifest,
)
from robata.runtime.e2e_trace import E2ETraceFragmentRole, build_e2e_trace_runtime_fragment
from robata.runtime.observability import RuntimeProfileRecorder, runtime_span

NOW = "2026-08-05T12:00:00Z"
TASK = VisionTask.ACTION_EVIDENCE
SCHEMA_DIGEST = "a" * 64


def _uuid(number: int) -> str:
    return str(UUID(int=number + 1))


def _digest(number: int) -> str:
    return f"{number:064x}"[-64:]


def _schema() -> JsonSchemaRef:
    return JsonSchemaRef(
        schema_id="test.provider.claim",
        version="1.0",
        artifact_id="test-provider-claim-v1",
        sha256=SCHEMA_DIGEST,
    )


def _source_files(tmp_path: Path) -> tuple[Path, tuple[LocalFrameArtifact, ...]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.mcap"
    source_bytes = b"frozen-test-mcap\n"
    source.write_bytes(source_bytes)
    object_root = tmp_path / "r2" / "sha256"
    artifacts: list[LocalFrameArtifact] = []
    for ordinal, camera_id in enumerate(CAMERA_IDS):
        data = f"camera-{ordinal}".encode("ascii")
        digest = exact_bytes_sha256(data)
        artifact_path = object_root / digest[:2] / f"{digest}.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(data)
        artifacts.append(
            LocalFrameArtifact(
                camera_id=camera_id.value,
                topic=f"/robot0/sensor/camera{ordinal}/compressed",
                source_timestamp_ns=1_781_051_907_271_600_000 + ordinal,
                messages_examined=1,
                decode_failures=(),
                width=4,
                height=3,
                uri=artifact_path.as_uri(),
                sha256=digest,
                byte_count=len(data),
            )
        )
    return source, tuple(artifacts)


def _report(tmp_path: Path) -> tuple[Path, tuple[LocalFrameArtifact, ...]]:
    source, artifacts = _source_files(tmp_path)
    report_path = tmp_path / "report.json"
    recorder = RuntimeProfileRecorder()
    for span_name in (
        "runtime.test",
        "source.test",
        "scheduler.test",
        "inference.test",
        "evidence.test",
        "reduction.test",
        "publication.test",
    ):
        with runtime_span(recorder, span_name):
            pass
    trace = build_e2e_trace_runtime_fragment(
        role=E2ETraceFragmentRole.LAUNCHER,
        runtime_profile=recorder.snapshot(),
    )
    declarations = {
        boundary: E2EParticipationState.PARTICIPATING for boundary in E2EParticipationBoundary
    }
    participation = build_e2e_participation_manifest(
        runtime_fragment=trace,
        declarations=declarations,
        trace_id=_uuid(999),
        observed_at=NOW,
    )
    participation_path = tmp_path / "participation.json"
    participation_sha256 = write_e2e_participation_manifest(participation, participation_path)
    source_bytes = source.read_bytes()
    report = LocalRealModelE2EReport(
        run_id=str(uuid4()),
        observed_at=NOW,
        source_path=str(source),
        source_sha256=exact_bytes_sha256(source_bytes),
        source_size_bytes=len(source_bytes),
        source_profile="test",
        source_message_count=1,
        mapping_profile_id="test-mapping",
        mapping_approval_status="APPROVED",
        camera_artifacts=artifacts,
        prompt="explicit test prompt",
        model=LocalModelObservation(
            model_transport="LOOPBACK_HTTP",
            model_identifier="local-test",
            model_version="test-v1",
            input_image_count=6,
            rendered_image_sizes=tuple((4, 3) for _ in CAMERA_IDS),
            prompt_tokens=1,
            output_tokens=1,
            load_seconds=0.0,
            generation_seconds=0.1,
            gpu_name="test-gpu",
            gpu_total_bytes=1,
            gpu_free_before_bytes=1,
            gpu_allocated_after_load_bytes=1,
            gpu_peak_allocated_bytes=1,
            output_text="{}",
            parsed_json={"ok": True},
        ),
        storage=LocalStorageObservation(
            object_store_root=str(tmp_path / "r2"),
            sqlite_path=str(tmp_path / "qualification.sqlite3"),
        ),
        trace=trace,
        stage_coverage={
            stage: "MEASURED"
            for stage in (
                "ORCHESTRATION",
                "SOURCE",
                "SCHEDULING",
                "INFERENCE",
                "EVIDENCE",
                "REDUCTION",
                "PUBLICATION",
            )
        },
        participation_coverage=participation.coverage,
        participation_manifest_sha256=participation_sha256,
        participation_manifest_path=str(participation_path),
        quality_observation={"structured_output_shape_valid": True},
        warnings=("test source is non-production",),
    )
    report_path.write_bytes(canonical_json_bytes(report.model_dump(mode="json")) + b"\n")
    return report_path, artifacts


def _plans(tmp_path: Path, artifacts: tuple[LocalFrameArtifact, ...]):
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    package_id = _uuid(10)
    frames = tuple(
        CatalogFrame(
            frame_id=_uuid(100 + ordinal),
            ordinal=0,
            aligned_timestamp_ns=artifact.source_timestamp_ns,
            source_timestamp_ns=artifact.source_timestamp_ns,
            source_artifact_uri=artifact.uri,
            source_artifact_sha256=artifact.sha256,
            source_artifact_bytes=artifact.byte_count,
            media_type=artifact.media_type,
            encoding="png",
            width=artifact.width,
            height=artifact.height,
        )
        for ordinal, artifact in enumerate(artifacts)
    )
    cameras = tuple(
        CatalogCamera(camera_id=camera_id, ordinal=ordinal, frames=(frames[ordinal],))
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )
    package = CatalogPackage(
        package_id=package_id,
        ordinal=0,
        semantic_content_sha256=_digest(20),
        manifest_bytes_sha256=_digest(21),
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(30),
        task=TASK,
        packages=(package,),
        created_at=NOW,
    )
    schema = _schema()

    def make_plan(model_name: str, index: int):
        target = InputPlanTarget(
            provider="runpod",
            model_name=model_name,
            model_version="1.0",
            adapter_version="runpod-adapter-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=_uuid(400 + index),
            capability_snapshot_sha256=_digest(401 + index),
        )
        items = tuple(
            RenderedProviderItem(
                provider_item_ordinal=ordinal,
                package_id=package.package_id,
                package_ordinal=0,
                camera_id=camera_id,
                camera_ordinal=ordinal,
                frame_id=frames[ordinal].frame_id,
                frame_ordinal=0,
                aligned_timestamp_ns=artifact.source_timestamp_ns,
                source_timestamp_ns=artifact.source_timestamp_ns,
                source_artifact_sha256=artifact.sha256,
                artifact=RenderedArtifact(
                    artifact_id=_uuid(500 + ordinal),
                    uri=artifact.uri,
                    sha256=artifact.sha256,
                    byte_count=artifact.byte_count,
                    media_type=artifact.media_type,
                    encoding="png",
                    width=artifact.width,
                    height=artifact.height,
                ),
                transform=FrameTransform.create(
                    operation=TransformOperation.NONE,
                    policy_version="test-identity-v1",
                ),
            )
            for ordinal, (camera_id, artifact) in enumerate(zip(CAMERA_IDS, artifacts, strict=True))
        )
        return planner.build(
            input_plan_id=_uuid(600 + index),
            created_at=NOW,
            request_catalog=catalog,
            target=target,
            rendered_items=items,
            prompt_output=PromptOutputContract(
                prompt_version="test-prompt-v1",
                prompt_sha256=_digest(70),
                rendered_message_sha256=_digest(71),
                provider_response_schema_sha256=schema.sha256,
                enriched_domain_schema_sha256=_digest(72),
                protocol_mode="json-schema",
                tool_mode="none",
            ),
            applicable_limits=ApplicableProviderLimits(
                max_images_per_request=6,
                max_pixels_per_image=12,
                max_payload_bytes_per_request=1000,
                max_input_tokens_per_request=100,
            ),
            call_parts=(
                CallPartSpec(
                    start_item_ordinal=0,
                    end_item_ordinal_exclusive=6,
                    measured_input_tokens=1,
                ),
            ),
            idempotency_policy_version="test-idempotency-v1",
            reduction_policy="ordered-concat",
            reduction_policy_version="test-reduction-v1",
        )

    def make_policy(model_name: str) -> InferencePolicy:
        return InferencePolicy(
            policy_version="test-policy-v1",
            task=TASK,
            provider="runpod",
            model_name=model_name,
            model_version="1.0",
            adapter_version="runpod-adapter-v1",
            prompt_version="test-prompt-v1",
            prompt_artifact_id="test-prompt",
            prompt_sha256=_digest(70),
            output_schema=schema,
            generation_config={"max_output_tokens": 16, "temperature": 0.0},
            timeout_ms=1000,
            selection_policy_version="test-selection-v1",
            required_input_mode=InputMode.MULTI_IMAGE,
            required_media_types=("image/png",),
        )

    return (
        make_plan("Qwen3-VL-4B", 0),
        make_policy("Qwen3-VL-4B"),
        make_plan("Mage-VL-4B", 1),
        make_policy("Mage-VL-4B"),
    )


def _target(
    *,
    tmp_path: Path,
    plan,
    policy,
    deployment_id: str,
    source_override: dict[str, object] | None = None,
) -> Path:
    source_values: dict[str, object] = {
        "experiment_id": "test-paired",
        "contract_version": "1.0",
        "route_id": "test-paired-route",
        "route_policy_version": "1.0",
        "arrival_schedule_sha256": _digest(80),
        "comparison_config": {"numeric_tolerance": 0.0},
        "input_representation": ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING,
        "isolation_profile": ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
        "mcap_id": str(_uuid(900)),
        "camera_mapping_run_id": str(_uuid(901)),
        "alignment_id": str(_uuid(902)),
        "start_ns": 1_781_051_907_271_600_000,
        "end_ns": 1_781_051_907_272_600_000,
        "input_config": {"source": "frozen-local-e2e"},
        "sampling_config": {"mode": "single"},
        "metadata": {"fixture": "builder"},
    }
    if source_override:
        source_values.update(source_override)
    config = ExternalPairedTargetConfig(
        deployment_id=deployment_id,
        policy=policy,
        input_plan=plan,
        input_plan_part_ordinal=0,
        source=ExternalPairedWorkloadSourceConfig.model_validate(source_values, strict=True),
    )
    path = tmp_path / f"{deployment_id}.json"
    path.write_bytes(canonical_json_bytes(config.model_dump(mode="json")) + b"\n")
    return path


def _files(tmp_path: Path):
    report_path, artifacts = _report(tmp_path)
    control_plan, control_policy, candidate_plan, candidate_policy = _plans(tmp_path, artifacts)
    control_path = _target(
        tmp_path=tmp_path,
        plan=control_plan,
        policy=control_policy,
        deployment_id="control-qwen",
    )
    candidate_path = _target(
        tmp_path=tmp_path,
        plan=candidate_plan,
        policy=candidate_policy,
        deployment_id="candidate-mage",
    )
    return report_path, control_path, candidate_path


def test_builder_creates_non_promotional_workload_from_frozen_report(tmp_path: Path) -> None:
    report_path, control_path, candidate_path = _files(tmp_path)
    result = build_external_paired_workload(
        report_path=report_path,
        control_target_path=control_path,
        candidate_target_path=candidate_path,
    )
    assert isinstance(result.workload, ExternalPairedWorkloadManifest)
    assert result.workload.format_version == "robata-external-paired-workload-v1"
    assert result.workload.source_workload_manifest_sha256 == result.source_report_sha256
    assert result.workload.input_identity_sha256 == result.input_identity_sha256
    assert result.workload.package_inputs[0].role == "primary"
    assert result.workload.control.deployment_id != result.workload.candidate.deployment_id
    output = tmp_path / "workload.json"
    assert write_external_paired_workload(result.workload, output) == result.workload_sha256
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    document = json.loads(payload)
    assert document["start_ns"] == "1781051907271600000"
    assert document["end_ns"] == "1781051907272600000"


def test_builder_rejects_candidate_source_binding_drift(tmp_path: Path) -> None:
    report_path, control_path, _candidate_path = _files(tmp_path)
    artifacts = _report(tmp_path / "other")[1]
    _control_plan, _control_policy, candidate_plan, candidate_policy = _plans(tmp_path, artifacts)
    candidate_path = _target(
        tmp_path=tmp_path,
        plan=candidate_plan,
        policy=candidate_policy,
        deployment_id="candidate-mage",
        source_override={"route_id": "different-route"},
    )
    with pytest.raises(ExternalPairedWorkloadBuilderError, match="source bindings"):
        build_external_paired_workload(
            report_path=report_path,
            control_target_path=control_path,
            candidate_target_path=candidate_path,
        )


def test_builder_rejects_tampered_camera_bytes(tmp_path: Path) -> None:
    report_path, control_path, candidate_path = _files(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact_path = Path(report["camera_artifacts"][0]["uri"][8:])
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(ExternalPairedWorkloadBuilderError, match="camera artifact digest"):
        build_external_paired_workload(
            report_path=report_path,
            control_target_path=control_path,
            candidate_target_path=candidate_path,
        )


def test_builder_rejects_different_rendered_input(tmp_path: Path) -> None:
    report_path, control_path, _candidate_path = _files(tmp_path)
    _source, artifacts = _source_files(tmp_path / "alt")
    _control_plan, _control_policy, candidate_plan, candidate_policy = _plans(tmp_path, artifacts)
    candidate_path = _target(
        tmp_path=tmp_path,
        plan=candidate_plan,
        policy=candidate_policy,
        deployment_id="candidate-mage",
    )
    with pytest.raises(ExternalPairedWorkloadBuilderError, match=r"source bindings|rendered input"):
        build_external_paired_workload(
            report_path=report_path,
            control_target_path=control_path,
            candidate_target_path=candidate_path,
        )


def test_builder_rejects_partial_participation_report(tmp_path: Path) -> None:
    report_path, control_path, candidate_path = _files(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["participation_coverage"] = "PARTIAL"
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    with pytest.raises(ExternalPairedWorkloadBuilderError, match="coverage must be COMPLETE"):
        build_external_paired_workload(
            report_path=report_path,
            control_target_path=control_path,
            candidate_target_path=candidate_path,
        )


def test_builder_rejects_tampered_participation_sidecar(tmp_path: Path) -> None:
    report_path, control_path, candidate_path = _files(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    participation_path = Path(report["participation_manifest_path"])
    participation_path.write_bytes(participation_path.read_bytes() + b"tampered")
    with pytest.raises(
        ExternalPairedWorkloadBuilderError,
        match="participation manifest digest does not match",
    ):
        build_external_paired_workload(
            report_path=report_path,
            control_target_path=control_path,
            candidate_target_path=candidate_path,
        )
