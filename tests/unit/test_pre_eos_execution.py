from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

from robata.application.canonical.pre_eos_execution import (
    MODEL_INFERENCE_SCHEMA_ID,
    ProviderNeutralStreamStageExecutor,
)
from robata.application.canonical.runner import (
    CanonicalOfflinePipeline,
    CanonicalPreEosInferenceInvocation,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import StreamStage, TerminalOutcome
from robata.contracts.stream_planning import StreamWorkItemPlan
from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    PackageInput,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    ModelInference,
    VisionTask,
)
from robata.inference.orchestrator import (
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)

_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
_NOW_TEXT = "2026-07-25T12:00:00Z"
_OUTPUT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["label"],
    "properties": {"label": {"type": "string"}},
}


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _output_schema_ref() -> JsonSchemaRef:
    return JsonSchemaRef(
        schema_id="pre-eos-qa-output",
        version="1.0",
        artifact_id="pre-eos-qa-output-schema",
        sha256=semantic_sha256(_OUTPUT_SCHEMA),
    )


def _model_inference_schema_ref() -> SchemaRef:
    return SchemaRef(
        schema_id=MODEL_INFERENCE_SCHEMA_ID,
        version="1.0.0",
        artifact_id=_uuid(90),
        sha256=_digest(91),
    )


class _DelayedFixtureAdapter:
    provider = "delayed-fixture"

    def __init__(self) -> None:
        self.infer_calls = 0

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        assert (model_name, model_version) == ("fixture-vision", "1.0")
        return ModelCapabilities(
            schema_version="1.0",
            snapshot_id=_uuid(100),
            snapshot_digest=_digest(101),
            provider=self.provider,
            model_name="fixture-vision",
            model_version="1.0",
            supported_tasks=(VisionTask.QA_COARSE,),
            input_modes=(InputMode.MULTI_IMAGE,),
            accepted_media_types=("image/png",),
            max_images_per_request=16,
            max_pixels_per_image=2_000_000,
            max_payload_bytes=10_000_000,
            max_input_tokens=10_000,
            supports_json_schema=True,
            supports_provider_idempotency=True,
            concurrency_class=ConcurrencyClass.LIMITED,
            data_handling_policy_version="fixture-data-v1",
            observed_at=_NOW_TEXT,
        )

    async def infer(self, request: VisionInferenceRequest) -> VisionInferenceSuccess:
        self.infer_calls += 1
        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id="delayed-fixture-request",
            provider=request.provider,
            model_name=request.model_name,
            model_version=request.model_version,
            normalized_output=NormalizedOutputEnvelope(
                task=request.task,
                output_schema=request.output_schema,
                package_input_set_sha256=request.package_input_set_sha256,
                payload={"label": "clear"},
            ),
            raw_output_artifact_id="delayed-fixture-raw-output",
            schema_valid=True,
            reported_confidence=0.9,
            usage=VisionUsage(
                input_frames=1,
                input_images=1,
                input_tokens=12,
                output_tokens=4,
            ),
            latency_ms=1,
        )


def _pipeline() -> tuple[CanonicalOfflinePipeline, _DelayedFixtureAdapter, InMemoryInferenceLedger]:
    adapter = _DelayedFixtureAdapter()
    schema_ref = _output_schema_ref()
    policy = InferencePolicy(
        policy_version="pre-eos-qa-policy-v1",
        task=VisionTask.QA_COARSE,
        provider=adapter.provider,
        model_name="fixture-vision",
        model_version="1.0",
        adapter_version="fixture-adapter-v1",
        prompt_version="fixture-prompt-v1",
        prompt_artifact_id="pre-eos-qa-prompt",
        prompt_sha256=_digest(200),
        output_schema=schema_ref,
        generation_config={"temperature": 0.0},
        timeout_ms=1_000,
        selection_policy_version="pre-eos-select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="fixture-data-v1",
    )
    ledger = InMemoryInferenceLedger()
    orchestrator = InferenceOrchestrator(
        adapters={adapter.provider: adapter},
        task_policies={VisionTask.QA_COARSE: policy},
        schema_documents={schema_ref.artifact_id: _OUTPUT_SCHEMA},
        ledger=ledger,
        clock=lambda: _NOW,
    )
    # P5's narrow pipeline entry needs only the long-lived orchestrator; using a
    # partial instance makes the replay assertion independent of unrelated
    # package/reduction setup.
    pipeline = object.__new__(CanonicalOfflinePipeline)
    pipeline._orchestrator = orchestrator
    return pipeline, adapter, ledger


def _invocation() -> CanonicalPreEosInferenceInvocation:
    return CanonicalPreEosInferenceInvocation(
        task=VisionTask.QA_COARSE,
        package_set_id=_uuid(1),
        mcap_id=_uuid(2),
        camera_mapping_run_id=_uuid(3),
        alignment_id=_uuid(4),
        start_ns=0,
        end_ns=1_000_000_000,
        package_inputs=(
            PackageInput(
                package_id=_uuid(5),
                package_semantic_content_sha256=_digest(6),
                package_manifest_sha256=_digest(7),
                role="window",
                ordinal=0,
            ),
        ),
        rendered_input_digest=_digest(8),
        input_config={"input_images": 1, "payload_bytes": 100},
        sampling_config={"policy": "pre-eos-test-v1"},
        metadata={"stream_stage": "QA_COARSE"},
    )


def _qa_coarse_plan() -> StreamWorkItemPlan:
    plan = MagicMock(spec=StreamWorkItemPlan)
    plan.stage = StreamStage.QA_COARSE
    return plan


def test_pre_eos_executor_reuses_durable_selection_after_crash_before_stream_terminal(
    tmp_path: Path,
) -> None:
    pipeline, adapter, ledger = _pipeline()
    invocation = _invocation()
    schema_ref = _model_inference_schema_ref()
    executor = ProviderNeutralStreamStageExecutor(
        pipeline=pipeline,
        invocation_factory=lambda plan: invocation,
        artifact_root=tmp_path / "stream-artifacts",
        model_inference_schema_ref=schema_ref,
        terminal_policy_version="stream-terminal-policy-v1",
    )

    # The first call represents provider evidence committed just before the
    # scheduler's terminal acceptance crashes. Reconstructing the hook exercises
    # the same replay boundary a restarted source worker uses.
    first = executor.execute(_qa_coarse_plan())
    restarted = ProviderNeutralStreamStageExecutor(
        pipeline=pipeline,
        invocation_factory=lambda plan: invocation,
        artifact_root=tmp_path / "stream-artifacts",
        model_inference_schema_ref=schema_ref,
        terminal_policy_version="stream-terminal-policy-v1",
    )
    replayed = restarted.execute(_qa_coarse_plan())

    assert first is not None
    assert replayed == first
    assert first.outcome is TerminalOutcome.SUCCEEDED
    assert first.evidence_ref.schema_ref == schema_ref
    assert adapter.infer_calls == 1
    assert len(ledger.list_terminals()) == 1
    assert len(ledger.list_selections()) == 1
    persisted = ModelInference.model_validate_json(
        restarted.artifact_path_for(first.evidence_ref).read_bytes(),
        strict=True,
    )
    assert persisted.stage is VisionTask.QA_COARSE
    assert persisted.output_valid
