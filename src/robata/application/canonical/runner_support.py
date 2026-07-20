"""Validation and conversion helpers for the canonical offline runner."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical.models import (
    CANONICAL_OFFLINE_PIPELINE_VERSION,
    CanonicalOfflineConfigurationError,
    CanonicalOfflineExecutionPolicy,
    CanonicalRootWindow,
)
from robata.application.canonical_run_membership import CanonicalProcessingRunContext
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.temporal import PackageLineage, TemporalPackageSet
from robata.inference.adapter import JsonSchemaRef, PackageInput
from robata.inference.enrichment import (
    ProviderReferenceCatalog,
    ProviderReferenceCatalogEntry,
)
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import ModelCapabilities, ModelInference
from robata.inference.orchestrator import InferencePolicy
from robata.inference.preparation import InputPlanPreparer
from robata.sampling.dense import IntervalPart
from robata.sampling.materializer import MaterializedTemporalPackage


def _validate_materialized_chain(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalRootWindow,
    lineage: PackageLineage,
    planned_parts: tuple[IntervalPart, ...],
    materialized: tuple[MaterializedTemporalPackage, ...],
) -> None:
    if len(materialized) != len(planned_parts) or not materialized:
        raise CanonicalOfflineConfigurationError(
            "materialized package count does not match planned parts"
        )
    expected_lineage = PackageLineage(
        source_content_sha256=context.source_content_sha256,
        window_semantic_sha256=window.semantic_sha256,
        camera_mapping_semantic_sha256=context.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=context.alignment_semantic_sha256,
        sampling_plan_sha256=lineage.sampling_plan_sha256,
    )
    if lineage != expected_lineage:
        raise CanonicalOfflineConfigurationError(
            "materialization lineage does not match context and root window"
        )
    for ordinal, (planned, output) in enumerate(zip(planned_parts, materialized, strict=True)):
        package = output.package
        if (
            package.part.ordinal != ordinal
            or package.part.part_count != len(planned_parts)
            or package.part.requested_interval != planned.requested_interval
            or package.part.effective_interval != planned.effective_interval
            or package.part.overlap_before_ns != planned.overlap_before_ns
            or package.part.overlap_after_ns != planned.overlap_after_ns
        ):
            raise CanonicalOfflineConfigurationError(
                "materialized package coordinates differ from the planned part"
            )
        if (
            package.window_id != window.window_id
            or package.mcap_id != context.ready_manifest.mcap_id
            or package.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
            or package.alignment_id != context.alignment_manifest.alignment_id
            or package.lineage != lineage
            or package.sampling_plan_sha256 != lineage.sampling_plan_sha256
        ):
            raise CanonicalOfflineConfigurationError(
                "materialized package authority binding is inconsistent"
            )
        if (
            output.manifest_bytes != canonical_json_bytes(package)
            or output.package_manifest_sha256 != exact_bytes_sha256(output.manifest_bytes)
            or output.package_ref.package_id != package.package_id
            or output.package_ref.package_semantic_content_sha256 != package.semantic_content_sha256
            or output.package_ref.package_manifest_sha256 != output.package_manifest_sha256
        ):
            raise CanonicalOfflineConfigurationError(
                "materialized package exact-byte identity is inconsistent"
            )


def _validate_package_set_chain(
    *,
    context: AdmittedRecordingContextV2,
    window: CanonicalRootWindow,
    lineage: PackageLineage,
    package_set: TemporalPackageSet,
    materialized: tuple[MaterializedTemporalPackage, ...],
    reduction_policy_version: str,
) -> None:
    if (
        package_set.mcap_id != context.ready_manifest.mcap_id
        or package_set.window_id != window.window_id
        or package_set.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
        or package_set.alignment_id != context.alignment_manifest.alignment_id
        or package_set.lineage != lineage
        or package_set.requested_start_ns != window.requested_interval.start_ns
        or package_set.requested_end_ns != window.requested_interval.end_ns
        or package_set.start_ns != window.interval.start_ns
        or package_set.end_ns != window.interval.end_ns
        or package_set.reduction_policy_version != reduction_policy_version
    ):
        raise CanonicalOfflineConfigurationError(
            "package set does not match context, window, lineage, and reduction policy"
        )
    expected = tuple(
        (
            item.package.package_id,
            item.package.part.ordinal,
            item.package.semantic_content_sha256,
            item.package_manifest_sha256,
        )
        for item in materialized
    )
    actual = tuple(
        (
            member.package_id,
            member.ordinal,
            member.package_semantic_content_sha256,
            member.package_manifest_sha256,
        )
        for member in package_set.members
    )
    if actual != expected:
        raise CanonicalOfflineConfigurationError(
            "package set members differ from exact materialized packages"
        )


def _validated_capabilities(
    capabilities: ModelCapabilities,
    *,
    inference_policy: InferencePolicy,
    input_preparer: InputPlanPreparer,
) -> ModelCapabilities:
    try:
        result = ModelCapabilities.model_validate(
            capabilities.model_dump(mode="python"), strict=True
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CanonicalOfflineConfigurationError(
            "offline capability snapshot failed strict validation"
        ) from exc
    if (
        result.provider != inference_policy.provider
        or result.model_name != inference_policy.model_name
        or result.model_version != inference_policy.model_version
        or inference_policy.task not in result.supported_tasks
    ):
        raise CanonicalOfflineConfigurationError(
            "capability snapshot does not match the pinned inference policy"
        )
    if not result.supports_json_schema or not result.supports_provider_idempotency:
        raise CanonicalOfflineConfigurationError(
            "canonical retry path requires schema and provider idempotency support"
        )
    required_media = set(inference_policy.required_media_types)
    rendering_media = set(input_preparer.policy.accepted_media_types)
    accepted_media = set(result.accepted_media_types)
    if not required_media.issubset(accepted_media) or not rendering_media.issubset(accepted_media):
        raise CanonicalOfflineConfigurationError(
            "capability media types do not cover policy and rendering requirements"
        )
    return result


def _rendered_prompt_bytes(
    *,
    inference_policy: InferencePolicy,
    request_catalog_sha256: str,
    token_policy_version: str,
    entries: tuple[ProviderReferenceCatalogEntry, ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "protocol": "robata-provider-claim-v1",
            "task": inference_policy.task.value,
            "prompt_artifact": {
                "version": inference_policy.prompt_version,
                "artifact_id": inference_policy.prompt_artifact_id,
                "sha256": inference_policy.prompt_sha256,
            },
            "request_catalog_sha256": request_catalog_sha256,
            "token_policy_version": token_policy_version,
            "evidence_catalog": [entry.model_dump(mode="json") for entry in entries],
            "provider_response_schema": inference_policy.output_schema.model_dump(mode="json"),
        }
    )


def _validate_input_plan_chain(
    *,
    package_set: TemporalPackageSet,
    materialized: tuple[MaterializedTemporalPackage, ...],
    input_plan: InferenceInputPlan,
    reference_catalog: ProviderReferenceCatalog,
    inference_policy: InferencePolicy,
    execution_policy: CanonicalOfflineExecutionPolicy,
    capabilities: ModelCapabilities,
) -> None:
    expected_packages = tuple(
        (
            item.package.package_id,
            item.package.part.ordinal,
            item.package.semantic_content_sha256,
            item.package_manifest_sha256,
        )
        for item in materialized
    )
    subject_packages = tuple(
        (
            item.package_id,
            item.ordinal,
            item.semantic_content_sha256,
            item.manifest_bytes_sha256,
        )
        for item in input_plan.subject.packages
    )
    set_packages = tuple(
        (
            item.package_id,
            item.ordinal,
            item.package_semantic_content_sha256,
            item.package_manifest_sha256,
        )
        for item in package_set.members
    )
    if subject_packages != expected_packages or set_packages != expected_packages:
        raise CanonicalOfflineConfigurationError(
            "input plan subject does not exactly match package-set members"
        )
    target = input_plan.target
    if (
        target.provider != inference_policy.provider
        or target.model_name != inference_policy.model_name
        or target.model_version != inference_policy.model_version
        or target.adapter_version != inference_policy.adapter_version
        or target.capability_snapshot_id != capabilities.snapshot_id
        or target.capability_snapshot_sha256 != capabilities.snapshot_digest
    ):
        raise CanonicalOfflineConfigurationError(
            "input plan target differs from pinned policy or capability snapshot"
        )
    expected_limits = (
        capabilities.max_images_per_request,
        capabilities.max_pixels_per_image,
        capabilities.max_payload_bytes,
        capabilities.max_input_tokens,
    )
    actual_limits = (
        input_plan.applicable_limits.max_images_per_request,
        input_plan.applicable_limits.max_pixels_per_image,
        input_plan.applicable_limits.max_payload_bytes_per_request,
        input_plan.applicable_limits.max_input_tokens_per_request,
    )
    if actual_limits != expected_limits:
        raise CanonicalOfflineConfigurationError(
            "input plan limits differ from the capability snapshot"
        )
    if any(
        item.artifact.media_type not in capabilities.accepted_media_types
        for item in input_plan.rendered_items
    ):
        raise CanonicalOfflineConfigurationError(
            "input plan contains media outside the capability snapshot"
        )
    if (
        input_plan.call_plan.reduction_policy != execution_policy.reduction_policy
        or input_plan.call_plan.reduction_policy_version
        != execution_policy.reduction_policy_version
        or input_plan.prompt_output.prompt_version != inference_policy.prompt_version
        or input_plan.prompt_output.prompt_sha256 != inference_policy.prompt_sha256
        or input_plan.prompt_output.provider_response_schema_sha256
        != inference_policy.output_schema.sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "input plan prompt, schema, or reduction policy binding differs"
        )
    if (
        reference_catalog.input_plan_id != input_plan.input_plan_id
        or reference_catalog.input_plan_semantic_sha256 != input_plan.semantic_sha256
        or reference_catalog.request_catalog_id != input_plan.request_catalog.request_catalog_id
        or reference_catalog.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "reference catalog does not bind the finalized input plan"
        )


def _package_inputs(package_set: TemporalPackageSet) -> tuple[PackageInput, ...]:
    return tuple(
        PackageInput(
            package_id=member.package_id,
            package_semantic_content_sha256=member.package_semantic_content_sha256,
            package_manifest_sha256=member.package_manifest_sha256,
            role="TEMPORAL_EVIDENCE",
            ordinal=member.ordinal,
        )
        for member in package_set.members
    )


def _terminal_raw_artifact_id(terminal: ModelInference) -> str:
    raw = terminal.raw_output
    artifact_id = raw.get("artifact_id") if raw is not None else None
    if not isinstance(artifact_id, str) or not artifact_id:
        raise CanonicalOfflineConfigurationError(
            "selected terminal has no valid raw response artifact reference"
        )
    return artifact_id


def _require_canonical_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CanonicalOfflineConfigurationError(f"{label} must be a UUID") from exc
    if str(parsed) != value:
        raise CanonicalOfflineConfigurationError(f"{label} must use canonical lowercase UUID text")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _validate_processing_run_binding(
    *,
    processing_run: CanonicalProcessingRunContext,
    admitted_context: AdmittedRecordingContextV2,
    execution_policy: CanonicalOfflineExecutionPolicy,
) -> None:
    if (
        processing_run.recording_identity != admitted_context.recording_identity
        or processing_run.mcap_id != admitted_context.ready_manifest.mcap_id
        or processing_run.pipeline_version != CANONICAL_OFFLINE_PIPELINE_VERSION
        or processing_run.config_sha256 != execution_policy.semantic_sha256
    ):
        raise CanonicalOfflineConfigurationError(
            "processing run does not bind the admitted recording and execution policy"
        )


def _schema_ref(ref: JsonSchemaRef) -> SchemaRef:
    return SchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _json_schema_ref(ref: SchemaRef) -> JsonSchemaRef:
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _rfc3339_datetime(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an RFC3339 timezone")
    return parsed
