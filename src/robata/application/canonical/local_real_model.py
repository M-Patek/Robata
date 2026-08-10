"""Reusable local Qwen canonical-composition wiring.

This module binds the explicitly local, loopback-only Qwen adapter to the
restartable SQLite canonical composition. It is deliberately non-authoritative:
receipts remain LOCAL_CONFORMANCE and never imply production eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from uuid import NAMESPACE_URL, uuid5

from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger
from robata.application.canonical.local_composition import (
    CanonicalLocalRunReceipt,
    LocalCanonicalModelBinding,
    LocalCanonicalNativeBatchAdmission,
    run_local_canonical_mcap,
)
from robata.application.canonical.runner import NormalizedOutputLineagePolicy
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import JsonSchemaRef, VisionModelAdapter
from robata.inference.enrichment import (
    ENRICHED_OUTPUT_SCHEMA_ID,
    ENRICHED_OUTPUT_SCHEMA_VERSION,
    PROVIDER_CLAIM_SCHEMA_ID,
)
from robata.inference.local_hf_adapter import (
    LOCAL_HF_LOOPBACK_ADAPTER_VERSION,
    LOCAL_HF_LOOPBACK_BASE_URL,
    LocalHfLoopbackAdapterConfig,
    LocalHfLoopbackVisionAdapter,
    LocalHfTransport,
    local_hf_compact_prompt_normalization_contract,
    local_hf_compact_prompt_normalization_contract_sha256,
)
from robata.inference.models import (
    ConcurrencyClass,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.offline_fixture import StrictProviderClaimParser
from robata.inference.orchestrator import InferencePolicy
from robata.runtime.observability import RuntimeObserver, RuntimeProfileRecorder

LOCAL_QWEN_PROVIDER: Final[Literal["local-huggingface"]] = "local-huggingface"
LOCAL_QWEN_MODEL_NAME: Final = "Qwen3-VL-4B-Instruct"
LOCAL_QWEN_MODEL_VERSION: Final = "local-2026-08-06"
LOCAL_QWEN_OBSERVED_AT: Final = "2026-08-06T00:00:00Z"
LOCAL_QWEN_DATA_HANDLING_POLICY_VERSION: Final = "local-loopback-data-v1"
LOCAL_QWEN_CAPABILITY_PROJECTION_VERSION: Final = "local-qwen-capabilities-v2"
LOCAL_QWEN_POLICY_PROJECTION_VERSION: Final = "local-qwen-policy-v2"
LOCAL_QWEN_MAX_IMAGES: Final = 6
LOCAL_QWEN_MAX_NEW_TOKENS: Final = 128
LOCAL_QWEN_TIMEOUT_MS: Final = 300_000
LOCAL_QWEN_MAX_PIXELS_PER_IMAGE: Final = 33_177_600
LOCAL_QWEN_MAX_PAYLOAD_BYTES: Final = 24_000_000
LOCAL_QWEN_MAX_INPUT_TOKENS: Final = 100_000
LOCAL_QWEN_ADAPTER_VERSION: Final = LOCAL_HF_LOOPBACK_ADAPTER_VERSION
LOCAL_QWEN_NORMALIZED_LINEAGE_POLICY_VERSION: Final = "local-qwen-compact-normalized-lineage-v1"
LOCAL_QWEN_NORMALIZATION_CONTRACT_SHA256: Final = (
    local_hf_compact_prompt_normalization_contract_sha256()
)
LOCAL_QWEN_NORMALIZED_LINEAGE_PARSER_VERSION: Final = (
    f"local-hf-compact-provider-claim-v1-{LOCAL_QWEN_NORMALIZATION_CONTRACT_SHA256[:12]}"
)
# Native batching is an explicit local candidate, not a mutation of the serial
# control.  The policy contains the qualified multi-claim serial guard observed
# in the frozen r12 A/B evidence; endpoint failures are never downgraded to serial.
LOCAL_QWEN_NATIVE_BATCH_POLICY_VERSION: Final = "local-qwen-task-claim-group-hybrid-batch-v1"
LOCAL_QWEN_NATIVE_BATCH_CAPACITY_PROJECTION_VERSION: Final = "local-qwen-batch4-local-capacity-v1"
LOCAL_QWEN_MULTI_CLAIM_SERIAL_GUARD_POLICY_VERSION: Final = "local-qwen-multi-claim-serial-guard-v1"
LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE: Final = 4
LOCAL_QWEN_NATIVE_BATCH_MAX_CONCURRENT_CALL_PARTS: Final = 4


@dataclass(frozen=True, slots=True)
class LocalQwenCanonicalRunOptions:
    """Optional knobs for one local Qwen canonical invocation."""

    run_key: str = "qwen-local-2026-08-06"
    allow_unapproved_profile: bool = False
    runtime_observer: RuntimeObserver | None = None
    transport: LocalHfTransport | None = None


def _schema_ref(registry: SchemaRegistry, schema_id: str, version: str) -> JsonSchemaRef:
    resolved = registry.resolve_version(schema_id, version).ref
    return JsonSchemaRef(
        schema_id=resolved.schema_id,
        version=resolved.version,
        artifact_id=resolved.artifact_id,
        sha256=resolved.sha256,
    )


def _stable_uuid(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "robata:local-qwen:" + ":".join(parts)))


def _prompt_identity(task: VisionTask, slug: str) -> tuple[str, str, str]:
    prompt_version = f"local-qwen-{slug}-prompt-v2"
    prompt_artifact_id = _stable_uuid("prompt", prompt_version)
    prompt_sha256 = semantic_sha256(
        {
            "semantic_projection_version": LOCAL_QWEN_POLICY_PROJECTION_VERSION,
            "prompt_version": prompt_version,
            "task": task.value,
            "compact_prompt_contract": local_hf_compact_prompt_normalization_contract(),
        }
    )
    return prompt_version, prompt_artifact_id, prompt_sha256


def build_local_qwen_normalized_lineage_policy() -> NormalizedOutputLineagePolicy:
    """Bind compact-wire expansion to one replay-visible local parser identity."""

    return NormalizedOutputLineagePolicy(
        version=LOCAL_QWEN_NORMALIZED_LINEAGE_POLICY_VERSION,
        parser_version=LOCAL_QWEN_NORMALIZED_LINEAGE_PARSER_VERSION,
        provider=LOCAL_QWEN_PROVIDER,
        model_name=LOCAL_QWEN_MODEL_NAME,
        model_version=LOCAL_QWEN_MODEL_VERSION,
        adapter_version=LOCAL_QWEN_ADAPTER_VERSION,
        normalization_contract_sha256=LOCAL_QWEN_NORMALIZATION_CONTRACT_SHA256,
        allowed_tasks=tuple(VisionTask),
    )


def build_local_qwen_capabilities(
    *,
    observed_at: str = LOCAL_QWEN_OBSERVED_AT,
    checkpoint_manifest_sha256: str | None = None,
) -> ModelCapabilities:
    """Build the immutable capability snapshot used by the local Qwen binding."""

    if checkpoint_manifest_sha256 is not None:
        normalized_checkpoint = checkpoint_manifest_sha256.lower()
        if len(normalized_checkpoint) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_checkpoint
        ):
            raise ValueError("checkpoint_manifest_sha256 must be a lowercase SHA-256 digest")
        checkpoint_manifest_sha256 = normalized_checkpoint
    tasks = (
        VisionTask.QA_COARSE,
        VisionTask.QA_DENSE,
        VisionTask.EVENT_PROPOSAL,
        VisionTask.ACTION_EVIDENCE,
        VisionTask.BOUNDARY_REFINEMENT,
        VisionTask.FUSION_ADJUDICATION,
    )
    input_modes = (InputMode.MULTI_IMAGE,)
    media_types = ("image/png",)
    facts: dict[str, object] = {
        "provider": LOCAL_QWEN_PROVIDER,
        "model_name": LOCAL_QWEN_MODEL_NAME,
        "model_version": LOCAL_QWEN_MODEL_VERSION,
        "supported_tasks": [task.value for task in tasks],
        "input_modes": [mode.value for mode in input_modes],
        "accepted_media_types": list(media_types),
        "max_images_per_request": LOCAL_QWEN_MAX_IMAGES,
        "max_pixels_per_image": LOCAL_QWEN_MAX_PIXELS_PER_IMAGE,
        "max_payload_bytes": LOCAL_QWEN_MAX_PAYLOAD_BYTES,
        "max_input_tokens": LOCAL_QWEN_MAX_INPUT_TOKENS,
        "supports_json_schema": True,
        "supports_provider_idempotency": True,
        "concurrency_class": ConcurrencyClass.SERIAL.value,
        "data_handling_policy_version": LOCAL_QWEN_DATA_HANDLING_POLICY_VERSION,
        "adapter_version": LOCAL_QWEN_ADAPTER_VERSION,
        "checkpoint_manifest_sha256": (checkpoint_manifest_sha256 or "UNBOUND_LOCAL_CHECKPOINT"),
        "normalized_lineage_policy": {
            "version": LOCAL_QWEN_NORMALIZED_LINEAGE_POLICY_VERSION,
            "parser_version": LOCAL_QWEN_NORMALIZED_LINEAGE_PARSER_VERSION,
            "normalization_contract_sha256": LOCAL_QWEN_NORMALIZATION_CONTRACT_SHA256,
            "semantic_sha256": build_local_qwen_normalized_lineage_policy().semantic_sha256,
        },
        "prompt_template_sha256_by_task": {
            task.value: _prompt_identity(task, task.value.lower())[2] for task in tasks
        },
        "endpoint_url": LOCAL_HF_LOOPBACK_BASE_URL,
        "max_new_tokens": LOCAL_QWEN_MAX_NEW_TOKENS,
        "timeout_ms": LOCAL_QWEN_TIMEOUT_MS,
        "observed_at": observed_at,
    }
    snapshot_digest = semantic_sha256(
        {
            "semantic_projection_version": LOCAL_QWEN_CAPABILITY_PROJECTION_VERSION,
            **facts,
        }
    )
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_stable_uuid("capabilities", snapshot_digest),
        snapshot_digest=snapshot_digest,
        provider=LOCAL_QWEN_PROVIDER,
        model_name=LOCAL_QWEN_MODEL_NAME,
        model_version=LOCAL_QWEN_MODEL_VERSION,
        supported_tasks=tasks,
        input_modes=input_modes,
        accepted_media_types=media_types,
        max_images_per_request=LOCAL_QWEN_MAX_IMAGES,
        max_pixels_per_image=LOCAL_QWEN_MAX_PIXELS_PER_IMAGE,
        max_payload_bytes=LOCAL_QWEN_MAX_PAYLOAD_BYTES,
        max_input_tokens=LOCAL_QWEN_MAX_INPUT_TOKENS,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.SERIAL,
        data_handling_policy_version=LOCAL_QWEN_DATA_HANDLING_POLICY_VERSION,
        observed_at=observed_at,
    )


def _build_policy(
    *,
    task: VisionTask,
    policy_slug: str,
    prompt_slug: str,
    provider_schema: JsonSchemaRef,
    enriched_schema: JsonSchemaRef,
) -> InferencePolicy:
    prompt_version, prompt_artifact_id, prompt_sha256 = _prompt_identity(task, prompt_slug)
    return InferencePolicy(
        policy_version=f"local-qwen-{policy_slug}-policy-v2",
        task=task,
        provider=LOCAL_QWEN_PROVIDER,
        model_name=LOCAL_QWEN_MODEL_NAME,
        model_version=LOCAL_QWEN_MODEL_VERSION,
        adapter_version=LOCAL_QWEN_ADAPTER_VERSION,
        prompt_version=prompt_version,
        prompt_artifact_id=prompt_artifact_id,
        prompt_sha256=prompt_sha256,
        output_schema=provider_schema,
        enriched_output_schema=enriched_schema,
        generation_config={"temperature": 0, "max_new_tokens": LOCAL_QWEN_MAX_NEW_TOKENS},
        timeout_ms=LOCAL_QWEN_TIMEOUT_MS,
        selection_policy_version="select-first-valid-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version=LOCAL_QWEN_DATA_HANDLING_POLICY_VERSION,
    )


def build_local_qwen_policies(
    *,
    registry: SchemaRegistry | None = None,
) -> tuple[
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
    InferencePolicy,
]:
    """Build all task policies bound to the local Qwen capability snapshot."""

    resolved_registry = registry or SchemaRegistry()
    provider_schema = _schema_ref(resolved_registry, PROVIDER_CLAIM_SCHEMA_ID, "1.0.0")
    enriched_schema = _schema_ref(
        resolved_registry,
        ENRICHED_OUTPUT_SCHEMA_ID,
        ENRICHED_OUTPUT_SCHEMA_VERSION,
    )
    return (
        _build_policy(
            task=VisionTask.QA_COARSE,
            policy_slug="coarse-qa",
            prompt_slug="coarse-qa",
            provider_schema=provider_schema,
            enriched_schema=enriched_schema,
        ),
        _build_policy(
            task=VisionTask.QA_DENSE,
            policy_slug="dense-qa",
            prompt_slug="dense-qa",
            provider_schema=provider_schema,
            enriched_schema=enriched_schema,
        ),
        _build_policy(
            task=VisionTask.EVENT_PROPOSAL,
            policy_slug="event-proposal",
            prompt_slug="event-proposal",
            provider_schema=provider_schema,
            enriched_schema=enriched_schema,
        ),
        _build_policy(
            task=VisionTask.ACTION_EVIDENCE,
            policy_slug="action-evidence",
            prompt_slug="action-evidence",
            provider_schema=provider_schema,
            enriched_schema=enriched_schema,
        ),
        _build_policy(
            task=VisionTask.BOUNDARY_REFINEMENT,
            policy_slug="boundary-refinement",
            prompt_slug="boundary-refinement",
            provider_schema=provider_schema,
            enriched_schema=enriched_schema,
        ),
        _build_policy(
            task=VisionTask.FUSION_ADJUDICATION,
            policy_slug="fusion",
            prompt_slug="fusion",
            provider_schema=provider_schema,
            enriched_schema=enriched_schema,
        ),
    )


def build_local_qwen_model_binding(
    *,
    transport: LocalHfTransport | None = None,
    observed_at: str = LOCAL_QWEN_OBSERVED_AT,
    checkpoint_manifest_sha256: str | None = None,
) -> LocalCanonicalModelBinding:
    """Build a serial local model binding using the composition-owned ledger."""

    capabilities = build_local_qwen_capabilities(
        observed_at=observed_at,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
    )
    policies = build_local_qwen_policies()
    adapter_config = LocalHfLoopbackAdapterConfig(
        provider=LOCAL_QWEN_PROVIDER,
        default_max_new_tokens=LOCAL_QWEN_MAX_NEW_TOKENS,
        request_timeout_cap_ms=LOCAL_QWEN_TIMEOUT_MS,
    )

    def adapter_factory(
        evidence_ledger: SQLiteInferenceEvidenceLedger,
        parser: StrictProviderClaimParser,
    ) -> VisionModelAdapter:
        return LocalHfLoopbackVisionAdapter(
            capabilities=capabilities,
            parser=parser,
            evidence_ledger=evidence_ledger,
            config=adapter_config,
            transport=transport,
        )

    return LocalCanonicalModelBinding(
        capabilities=capabilities,
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
        adapter_factory=adapter_factory,
        normalized_output_lineage_policy=build_local_qwen_normalized_lineage_policy(),
        max_concurrent_call_parts=1,
        max_inference_batch_size=1,
    )


def build_local_qwen_batch_model_binding(
    *,
    transport: LocalHfTransport | None = None,
    observed_at: str = LOCAL_QWEN_OBSERVED_AT,
    checkpoint_manifest_sha256: str | None = None,
    max_inference_batch_size: int = LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE,
    max_concurrent_call_parts: int = LOCAL_QWEN_NATIVE_BATCH_MAX_CONCURRENT_CALL_PARTS,
) -> LocalCanonicalModelBinding:
    """Build the qualified Batch4 hybrid candidate as an explicit local route.

    Per-request capabilities and policies intentionally match the serial control.
    Only the internal runtime/capacity identity changes.  Selecting
    :func:`build_local_qwen_model_binding` is therefore the complete rollback.
    """

    if max_inference_batch_size != LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE:
        raise ValueError(
            "max_inference_batch_size must equal the qualified local Qwen native batch size "
            f"{LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE}"
        )
    capabilities = build_local_qwen_capabilities(
        observed_at=observed_at,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
    )
    policies = build_local_qwen_policies()
    adapter_config = LocalHfLoopbackAdapterConfig(
        provider=LOCAL_QWEN_PROVIDER,
        default_max_new_tokens=LOCAL_QWEN_MAX_NEW_TOKENS,
        request_timeout_cap_ms=LOCAL_QWEN_TIMEOUT_MS,
    )

    def adapter_factory(
        evidence_ledger: SQLiteInferenceEvidenceLedger,
        parser: StrictProviderClaimParser,
    ) -> VisionModelAdapter:
        return LocalHfLoopbackVisionAdapter(
            capabilities=capabilities,
            parser=parser,
            evidence_ledger=evidence_ledger,
            config=adapter_config,
            transport=transport,
        )

    return LocalCanonicalModelBinding(
        capabilities=capabilities,
        coarse_qa_policy=policies[0],
        dense_qa_policy=policies[1],
        event_proposal_policy=policies[2],
        action_evidence_policy=policies[3],
        boundary_refinement_policy=policies[4],
        inference_policy=policies[5],
        adapter_factory=adapter_factory,
        normalized_output_lineage_policy=build_local_qwen_normalized_lineage_policy(),
        native_batch_admission=LocalCanonicalNativeBatchAdmission(
            policy_version=LOCAL_QWEN_NATIVE_BATCH_POLICY_VERSION,
            max_batch_size=LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE,
            capacity_projection_version=(LOCAL_QWEN_NATIVE_BATCH_CAPACITY_PROJECTION_VERSION),
            serial_guard_policy_version=(LOCAL_QWEN_MULTI_CLAIM_SERIAL_GUARD_POLICY_VERSION),
        ),
        max_concurrent_call_parts=max_concurrent_call_parts,
        max_inference_batch_size=max_inference_batch_size,
    )


def run_local_qwen_canonical_mcap(
    *,
    source_path: Path,
    mapping_config: Path,
    state_dir: Path,
    run_key: str = "qwen-local-2026-08-06",
    allow_unapproved_profile: bool = False,
    runtime_observer: RuntimeObserver | None = None,
    transport: LocalHfTransport | None = None,
    checkpoint_manifest_sha256: str | None = None,
) -> CanonicalLocalRunReceipt:
    """Run the complete local canonical path without truncating recording duration."""

    observer = runtime_observer or RuntimeProfileRecorder()
    binding = build_local_qwen_model_binding(
        transport=transport,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
    )
    return run_local_canonical_mcap(
        source_path=source_path,
        mapping_config=mapping_config,
        state_dir=state_dir,
        run_key=run_key,
        allow_unapproved_profile=allow_unapproved_profile,
        max_duration_ns=None,
        runtime_observer=observer,
        model_binding=binding,
    )


# Short aliases make the reusable wiring convenient for callers that use the
# Qwen name rather than the longer local-hugging-face provider name.
build_qwen_capabilities = build_local_qwen_capabilities
build_qwen_batch_model_binding = build_local_qwen_batch_model_binding
build_qwen_model_binding = build_local_qwen_model_binding
build_qwen_policies = build_local_qwen_policies
run_local_qwen_canonical = run_local_qwen_canonical_mcap


__all__ = [
    "LOCAL_QWEN_ADAPTER_VERSION",
    "LOCAL_QWEN_CAPABILITY_PROJECTION_VERSION",
    "LOCAL_QWEN_DATA_HANDLING_POLICY_VERSION",
    "LOCAL_QWEN_MAX_IMAGES",
    "LOCAL_QWEN_MAX_INPUT_TOKENS",
    "LOCAL_QWEN_MAX_NEW_TOKENS",
    "LOCAL_QWEN_MAX_PAYLOAD_BYTES",
    "LOCAL_QWEN_MAX_PIXELS_PER_IMAGE",
    "LOCAL_QWEN_MODEL_NAME",
    "LOCAL_QWEN_MODEL_VERSION",
    "LOCAL_QWEN_MULTI_CLAIM_SERIAL_GUARD_POLICY_VERSION",
    "LOCAL_QWEN_NATIVE_BATCH_CAPACITY_PROJECTION_VERSION",
    "LOCAL_QWEN_NATIVE_BATCH_MAX_CONCURRENT_CALL_PARTS",
    "LOCAL_QWEN_NATIVE_BATCH_MAX_SIZE",
    "LOCAL_QWEN_NATIVE_BATCH_POLICY_VERSION",
    "LOCAL_QWEN_NORMALIZATION_CONTRACT_SHA256",
    "LOCAL_QWEN_NORMALIZED_LINEAGE_PARSER_VERSION",
    "LOCAL_QWEN_NORMALIZED_LINEAGE_POLICY_VERSION",
    "LOCAL_QWEN_OBSERVED_AT",
    "LOCAL_QWEN_POLICY_PROJECTION_VERSION",
    "LOCAL_QWEN_PROVIDER",
    "LOCAL_QWEN_TIMEOUT_MS",
    "LocalQwenCanonicalRunOptions",
    "build_local_qwen_batch_model_binding",
    "build_local_qwen_capabilities",
    "build_local_qwen_model_binding",
    "build_local_qwen_normalized_lineage_policy",
    "build_local_qwen_policies",
    "build_qwen_batch_model_binding",
    "build_qwen_capabilities",
    "build_qwen_model_binding",
    "build_qwen_policies",
    "run_local_qwen_canonical",
    "run_local_qwen_canonical_mcap",
]
