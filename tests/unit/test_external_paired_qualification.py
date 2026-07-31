from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest

from robata.benchmark.external_paired_qualification import (
    ExternalEvidenceMode,
    ExternalGateStatus,
    ExternalPairedQualificationError,
    ExternalPairedRole,
    ExternalPairedWorkloadManifest,
    ExternalPairedWorkloadTarget,
    run_external_paired_qualification,
    write_external_paired_qualification_report,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import JsonSchemaRef, PackageInput
from robata.inference.enrichment import PROVIDER_CLAIM_SCHEMA_ID
from robata.inference.experiment_execution import ExperimentComparisonStatus
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlan,
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.models import (
    ConcurrencyClass,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.orchestrator import InferencePolicy
from robata.inference.routing import (
    DispatchDisposition,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
)
from robata.inference.runpod import (
    RecordedRunPodExchange,
    RecordedRunPodTransport,
    RunPodApiKey,
    RunPodHttpRequest,
    RunPodHttpResponse,
)
from robata.runtime.e2e_trace import (
    E2ETraceFragmentRole,
    E2ETraceMeasurementStatus,
    E2ETraceStage,
    run_external_paired_qualification_with_trace,
    write_external_paired_e2e_trace,
)

NOW = "2026-07-30T12:00:00Z"
TASK = VisionTask.ACTION_EVIDENCE
RAW_CLAIM = '{"claims":[],"abstained":true}'


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


@dataclass
class _EchoRecordingTransport:
    """Offline endpoint double used only to create exact recorded exchanges."""

    role: ExternalPairedRole
    exchanges: list[RecordedRunPodExchange] = field(default_factory=list)
    requests: list[RunPodHttpRequest] = field(default_factory=list)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self.requests.append(request)
        binding = json.loads(request.body)["input"]["binding"]
        response = RunPodHttpResponse(
            status_code=200,
            body=canonical_json_bytes(
                {
                    "id": f"recorded-{self.role.value.lower()}-job",
                    "status": "COMPLETED",
                    "output": {
                        "contract_version": "robata-runpod-vision-response-v1",
                        "binding": binding,
                        "raw_output_json": RAW_CLAIM,
                        "usage": {
                            "input_tokens": 120,
                            "output_tokens": 7,
                            "cost_usd": 0.01,
                        },
                    },
                    "delayTime": 2,
                    "executionTime": 5,
                    "workerId": f"recorded-{self.role.value.lower()}-worker",
                }
            ),
        )
        self.exchanges.append(
            RecordedRunPodExchange(
                request_body_sha256=exact_bytes_sha256(request.body),
                response=response,
            )
        )
        return response


@dataclass(frozen=True)
class _FixtureFiles:
    environment: dict[str, str]
    control_capabilities: Path
    candidate_capabilities: Path
    workload: Path


def _capabilities(*, model_name: str, snapshot: int) -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(snapshot),
        snapshot_digest=_digest(snapshot),
        provider="runpod",
        model_name=model_name,
        model_version="1.0",
        supported_tasks=(TASK,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=6,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=10_000,
        max_input_tokens=1_000,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="p20-test-data-policy-v1",
        observed_at=NOW,
    )


def _schema_ref() -> JsonSchemaRef:
    ref = SchemaRegistry().resolve_alias(PROVIDER_CLAIM_SCHEMA_ID).ref
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _source() -> tuple[object, tuple[RenderedProviderItem, ...], tuple[PackageInput, ...]]:
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_uuid(100 + ordinal),
                    ordinal=0,
                    aligned_timestamp_ns=1_000_000 + ordinal,
                    source_timestamp_ns=2_000_000 + ordinal,
                    source_artifact_uri=f"object://p20-source/{ordinal}.png",
                    source_artifact_sha256=_digest(200 + ordinal),
                    source_artifact_bytes=128,
                    media_type="image/png",
                    encoding="png",
                    width=640,
                    height=480,
                ),
            ),
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )
    package = CatalogPackage(
        package_id=_uuid(50),
        ordinal=0,
        semantic_content_sha256=_digest(300),
        manifest_bytes_sha256=_digest(301),
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(60),
        task=TASK,
        packages=(package,),
        created_at=NOW,
    )
    items = tuple(
        RenderedProviderItem(
            provider_item_ordinal=ordinal,
            package_id=package.package_id,
            package_ordinal=package.ordinal,
            camera_id=camera.camera_id,
            camera_ordinal=camera.ordinal,
            frame_id=frame.frame_id,
            frame_ordinal=frame.ordinal,
            aligned_timestamp_ns=frame.aligned_timestamp_ns,
            source_timestamp_ns=frame.source_timestamp_ns,
            source_artifact_sha256=frame.source_artifact_sha256,
            artifact=RenderedArtifact(
                artifact_id=_uuid(400 + ordinal),
                uri=f"object://p20-rendered/{ordinal}.png",
                sha256=frame.source_artifact_sha256,
                byte_count=frame.source_artifact_bytes,
                media_type=frame.media_type,
                encoding=frame.encoding,
                width=frame.width,
                height=frame.height,
            ),
            transform=FrameTransform.create(
                operation=TransformOperation.NONE,
                policy_version="p20-identical-frame-v1",
            ),
        )
        for ordinal, (camera, frame) in enumerate(
            (camera, frame) for camera in package.cameras for frame in camera.frames
        )
    )
    package_inputs = (
        PackageInput(
            package_id=package.package_id,
            package_semantic_content_sha256=package.semantic_content_sha256,
            package_manifest_sha256=package.manifest_bytes_sha256,
            role="primary",
            ordinal=package.ordinal,
        ),
    )
    return catalog, items, package_inputs


def _plan(
    *,
    catalog: object,
    items: tuple[RenderedProviderItem, ...],
    capabilities: ModelCapabilities,
    plan_id: int,
    schema: JsonSchemaRef,
) -> InferenceInputPlan:
    return InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION).build(
        input_plan_id=_uuid(plan_id),
        created_at=NOW,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            model_version=capabilities.model_version,
            adapter_version="runpod-adapter-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=capabilities.snapshot_id,
            capability_snapshot_sha256=capabilities.snapshot_digest,
        ),
        rendered_items=items,
        prompt_output=PromptOutputContract(
            prompt_version="p20-prompt-v1",
            prompt_sha256=_digest(700),
            rendered_message_sha256=_digest(701),
            provider_response_schema_sha256=schema.sha256,
            enriched_domain_schema_sha256=_digest(702),
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=capabilities.max_images_per_request,
            max_pixels_per_image=capabilities.max_pixels_per_image,
            max_payload_bytes_per_request=capabilities.max_payload_bytes,
            max_input_tokens_per_request=capabilities.max_input_tokens,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=120,
            ),
        ),
        idempotency_policy_version="p20-idempotency-v1",
        reduction_policy="ordered-concat",
        reduction_policy_version="p20-reduction-v1",
    )


def _policy(capabilities: ModelCapabilities, schema: JsonSchemaRef) -> InferencePolicy:
    return InferencePolicy(
        policy_version="p20-external-paired-policy-v1",
        task=TASK,
        provider=capabilities.provider,
        model_name=capabilities.model_name,
        model_version=capabilities.model_version,
        adapter_version="runpod-adapter-v1",
        prompt_version="p20-prompt-v1",
        prompt_artifact_id="p20-action-evidence-prompt",
        prompt_sha256=_digest(700),
        output_schema=schema,
        generation_config={"max_output_tokens": 64, "temperature": 0.0},
        timeout_ms=1_000,
        selection_policy_version="p20-selection-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="p20-test-data-policy-v1",
    )


def _role_environment(
    *,
    role: ExternalPairedRole,
    capabilities: ModelCapabilities,
    offset: int,
) -> dict[str, str]:
    prefix = f"RUNPOD_{role.value}_"
    return {
        f"{prefix}ENDPOINT_URL": f"https://api.runpod.test/v2/{role.value.lower()}-{offset}/runsync",
        f"{prefix}MODEL_IDENTIFIER": capabilities.model_name,
        f"{prefix}MODEL_VERSION": capabilities.model_version,
        f"{prefix}HANDLER_IMAGE": f"registry.test/robata/{role.value.lower()}:{offset}",
        f"{prefix}HANDLER_IMAGE_SHA256": _digest(800 + offset),
        f"{prefix}CAPABILITY_SNAPSHOT_SHA256": capabilities.snapshot_digest,
        f"{prefix}INFERENCE_ENGINE": "vllm",
        f"{prefix}PRECISION_OR_QUANTIZATION": "bf16",
        f"{prefix}TOPOLOGY": "TWO_SINGLE_CARD_REPLICAS",
        f"{prefix}MAX_OUTPUT_TOKENS": "64",
        f"{prefix}ADAPTER_VERSION": "runpod-adapter-v1",
        f"{prefix}NATIVE_BATCH_ENABLED": "false",
        f"{prefix}NATIVE_BATCH_MAX_SIZE": "1",
        f"{prefix}MAX_CONCURRENT_REQUESTS": "1",
        f"{prefix}REQUEST_TIMEOUT_CAP_MS": "120000",
        f"{prefix}MAX_RESPONSE_BYTES": "4194304",
    }


def _fixture_files(tmp_path: Path) -> _FixtureFiles:
    control = _capabilities(model_name="Qwen3-VL-4B", snapshot=500)
    candidate = _capabilities(model_name="Mage-VL-4B", snapshot=501)
    schema = _schema_ref()
    catalog, items, package_inputs = _source()
    manifest = ExternalPairedWorkloadManifest(
        experiment_id="p20-mage-vs-qwen3-vl-4b",
        contract_version="1.0",
        route_id="p20-mage-vs-qwen3-vl-4b-paired",
        route_policy_version="1.0",
        source_workload_manifest_sha256=_digest(600),
        arrival_schedule_sha256=_digest(601),
        comparison_config={"default_severity": "MATERIAL", "numeric_tolerance": 0.0},
        input_representation=ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING,
        isolation_profile=ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
        input_identity_sha256=_digest(602),
        task=TASK,
        package_set_id=_uuid(70),
        mcap_id=_uuid(71),
        camera_mapping_run_id=_uuid(72),
        alignment_id=_uuid(73),
        start_ns=100,
        end_ns=200,
        package_inputs=package_inputs,
        input_config={"fixture": "p20-external-paired"},
        sampling_config={"policy": "p20-paired"},
        metadata={"fixture": "recorded-runpod"},
        control=ExternalPairedWorkloadTarget(
            deployment_id="p20-qwen3-vl-4b-control",
            policy=_policy(control, schema),
            input_plan=_plan(
                catalog=catalog,
                items=items,
                capabilities=control,
                plan_id=900,
                schema=schema,
            ),
            input_plan_part_ordinal=0,
        ),
        candidate=ExternalPairedWorkloadTarget(
            deployment_id="p20-mage-vl-4b-candidate",
            policy=_policy(candidate, schema),
            input_plan=_plan(
                catalog=catalog,
                items=items,
                capabilities=candidate,
                plan_id=901,
                schema=schema,
            ),
            input_plan_part_ordinal=0,
        ),
    )
    capabilities_directory = tmp_path / "capabilities"
    capabilities_directory.mkdir()
    control_path = capabilities_directory / "control.json"
    candidate_path = capabilities_directory / "candidate.json"
    workload_path = tmp_path / "workload.json"
    control_path.write_bytes(canonical_json_bytes(control.model_dump(mode="json")))
    candidate_path.write_bytes(canonical_json_bytes(candidate.model_dump(mode="json")))
    workload_path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    environment = {
        "RUNPOD_API_KEY": "runpod-test-secret-000000000000",
        **_role_environment(
            role=ExternalPairedRole.CONTROL,
            capabilities=control,
            offset=1,
        ),
        **_role_environment(
            role=ExternalPairedRole.CANDIDATE,
            capabilities=candidate,
            offset=2,
        ),
    }
    return _FixtureFiles(
        environment=environment,
        control_capabilities=control_path,
        candidate_capabilities=candidate_path,
        workload=workload_path,
    )


def _run(
    files: _FixtureFiles,
    *,
    transport_factory,
    evidence_directory: Path | None = None,
):
    return asyncio.run(
        run_external_paired_qualification(
            environment=files.environment,
            control_capabilities_path=files.control_capabilities,
            candidate_capabilities_path=files.candidate_capabilities,
            workload_path=files.workload,
            transport_factory=transport_factory,
            evidence_directory=evidence_directory,
        )
    )


def test_external_paired_launcher_replays_recorded_endpoints_without_promotion(
    tmp_path: Path,
) -> None:
    files = _fixture_files(tmp_path)
    live_recorders = {
        ExternalPairedRole.CONTROL: _EchoRecordingTransport(ExternalPairedRole.CONTROL),
        ExternalPairedRole.CANDIDATE: _EchoRecordingTransport(ExternalPairedRole.CANDIDATE),
    }
    _run(files, transport_factory=lambda role: live_recorders[role])
    recorded_exchanges = {
        role: tuple(transport.exchanges) for role, transport in live_recorders.items()
    }

    report = _run(
        files,
        transport_factory=lambda role: RecordedRunPodTransport(recorded_exchanges[role]),
    )

    assert report.evidence_mode is ExternalEvidenceMode.IN_MEMORY
    assert report.comparison.status is ExperimentComparisonStatus.AGREEMENT
    assert report.comparison.comparable is True
    assert all(
        dispatch.disposition is DispatchDisposition.OBSERVATION
        for dispatch in report.experiment_decision.dispatches
    )
    assert report.control_observation.terminal is not None
    assert report.candidate_observation.terminal is not None
    assert report.control_observation.terminal.shadow is True
    assert report.candidate_observation.terminal.shadow is True
    assert report.control_observation.transport_request_count == 1
    assert report.candidate_observation.transport_request_count == 1
    assert len(report.control_observation.raw_evidence) == 1
    assert len(report.candidate_observation.raw_evidence) == 1
    assert [item.status for item in report.external_gates] == [
        ExternalGateStatus.NOT_MEASURED,
        ExternalGateStatus.NOT_MEASURED,
        ExternalGateStatus.NOT_MEASURED,
        ExternalGateStatus.OBSERVED_PROVIDER_OUTCOME,
        ExternalGateStatus.NOT_MEASURED,
        ExternalGateStatus.NOT_MEASURED,
    ]
    assert report.production_eligible is False
    assert report.selection_eligible is False

    output = tmp_path / "external-paired-observation.json"
    write_external_paired_qualification_report(report, output)
    rendered = output.read_text(encoding="utf-8")
    assert "runpod-test-secret-000000000000" not in rendered
    assert '"production_eligible":false' in rendered

    durable_report = _run(
        files,
        transport_factory=lambda role: RecordedRunPodTransport(recorded_exchanges[role]),
        evidence_directory=tmp_path / "durable-evidence",
    )
    assert durable_report.evidence_mode is ExternalEvidenceMode.DURABLE_LOCAL_SQLITE
    assert (tmp_path / "durable-evidence" / "control-inference-evidence.sqlite").is_file()
    assert (tmp_path / "durable-evidence" / "candidate-inference-evidence.sqlite").is_file()

    from scripts.run_external_paired_model_qualification import main

    environment_path = tmp_path / "runpod.env"
    environment_path.write_text(
        "\n".join(f"{key}={value}" for key, value in files.environment.items()) + "\n",
        encoding="utf-8",
    )
    cli_output = tmp_path / "cli-external-paired-observation.json"
    exit_code = main(
        [
            "--env-file",
            str(environment_path),
            "--capabilities-dir",
            str(files.control_capabilities.parent),
            "--workload",
            str(files.workload),
            "--output",
            str(cli_output),
        ],
        environment={},
        transport_factory=lambda role: RecordedRunPodTransport(recorded_exchanges[role]),
    )
    assert exit_code == 0
    assert cli_output.is_file()


def test_external_paired_launcher_emits_noncanonical_e2e_trace_sidecar(
    tmp_path: Path,
) -> None:
    files = _fixture_files(tmp_path)
    live_recorders = {
        ExternalPairedRole.CONTROL: _EchoRecordingTransport(ExternalPairedRole.CONTROL),
        ExternalPairedRole.CANDIDATE: _EchoRecordingTransport(ExternalPairedRole.CANDIDATE),
    }
    _run(files, transport_factory=lambda role: live_recorders[role])
    recorded_exchanges = {
        role: tuple(transport.exchanges) for role, transport in live_recorders.items()
    }

    execution = asyncio.run(
        run_external_paired_qualification_with_trace(
            environment=files.environment,
            control_capabilities_path=files.control_capabilities,
            candidate_capabilities_path=files.candidate_capabilities,
            workload_path=files.workload,
            trace_id=_uuid(990),
            observed_at=NOW,
            transport_factory=lambda role: RecordedRunPodTransport(recorded_exchanges[role]),
        )
    )

    report = execution.report
    trace = execution.trace
    assert trace.trace_id == _uuid(990)
    assert trace.selection_eligible is False
    assert trace.production_eligible is False
    assert trace.launcher.role is E2ETraceFragmentRole.LAUNCHER
    assert trace.control.role is E2ETraceFragmentRole.CONTROL
    assert trace.candidate.role is E2ETraceFragmentRole.CANDIDATE
    assert any(
        span.name == "qualification.external_paired"
        for span in trace.launcher.runtime_profile.spans
    )
    control_inference = next(
        item for item in trace.control.stages if item.stage is E2ETraceStage.INFERENCE
    )
    assert control_inference.measurement_status is E2ETraceMeasurementStatus.MEASURED
    source_stage = next(item for item in trace.control.stages if item.stage is E2ETraceStage.SOURCE)
    assert source_stage.measurement_status is E2ETraceMeasurementStatus.NOT_MEASURED
    assert source_stage.wall_time_union_ns is None
    assert trace.correlation.control.request_id == report.control_observation.terminal.request_id
    assert (
        trace.correlation.candidate.request_id == report.candidate_observation.terminal.request_id
    )
    assert trace.quality_funnel[-1].measurement_status is E2ETraceMeasurementStatus.NOT_MEASURED
    assert trace.provider_cost_inputs[0].measurement_status is E2ETraceMeasurementStatus.MEASURED

    report_path = tmp_path / "paired-report.json"
    trace_path = tmp_path / "paired-trace.json"
    write_external_paired_qualification_report(report, report_path)
    write_external_paired_e2e_trace(trace, trace_path)
    assert trace.correlation.external_report_file_sha256 == exact_bytes_sha256(
        report_path.read_bytes()
    )
    rendered = trace_path.read_text(encoding="utf-8")
    assert "runpod-test-secret-000000000000" not in rendered
    assert '"coverage":"PARTIAL"' in rendered


def test_external_paired_launcher_retains_recorded_endpoint_failures_as_observations(
    tmp_path: Path,
) -> None:
    files = _fixture_files(tmp_path)
    live_recorders = {
        ExternalPairedRole.CONTROL: _EchoRecordingTransport(ExternalPairedRole.CONTROL),
        ExternalPairedRole.CANDIDATE: _EchoRecordingTransport(ExternalPairedRole.CANDIDATE),
    }
    _run(files, transport_factory=lambda role: live_recorders[role])
    failed_exchanges = {
        role: tuple(
            RecordedRunPodExchange(
                request_body_sha256=exchange.request_body_sha256,
                response=RunPodHttpResponse(status_code=503, body=b"recorded endpoint unavailable"),
            )
            for exchange in transport.exchanges
        )
        for role, transport in live_recorders.items()
    }

    report = _run(
        files,
        transport_factory=lambda role: RecordedRunPodTransport(failed_exchanges[role]),
    )

    assert report.comparison.status is ExperimentComparisonStatus.BOTH_FAILURE
    assert report.comparison.comparable is False
    assert report.control_observation.terminal is not None
    assert report.candidate_observation.terminal is not None
    assert report.control_observation.terminal.shadow is True
    assert report.candidate_observation.terminal.shadow is True
    assert report.control_observation.raw_evidence == ()
    assert report.candidate_observation.raw_evidence == ()
    assert report.external_gates[3].status is ExternalGateStatus.OBSERVED_PROVIDER_OUTCOME
    assert "BOTH_FAILURE" in report.external_gates[3].detail

    output = tmp_path / "failed-external-paired-observation.json"
    write_external_paired_qualification_report(report, output)
    assert '"status":"BOTH_FAILURE"' in output.read_text(encoding="utf-8")


def test_external_paired_launcher_rejects_capability_snapshot_drift_before_transport_creation(
    tmp_path: Path,
) -> None:
    files = _fixture_files(tmp_path)
    environment = {**files.environment, "RUNPOD_CONTROL_CAPABILITY_SNAPSHOT_SHA256": _digest(999)}
    factory_calls: list[ExternalPairedRole] = []

    def factory(role: ExternalPairedRole):
        factory_calls.append(role)
        raise AssertionError("transport must not be created for invalid capability bindings")

    with pytest.raises(
        ExternalPairedQualificationError,
        match="control capabilities snapshot digest does not match",
    ):
        asyncio.run(
            run_external_paired_qualification(
                environment=environment,
                control_capabilities_path=files.control_capabilities,
                candidate_capabilities_path=files.candidate_capabilities,
                workload_path=files.workload,
                transport_factory=factory,
            )
        )

    assert factory_calls == []
