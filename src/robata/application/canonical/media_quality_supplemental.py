"""Bridge registered media-quality evidence to frozen supplemental QA targets."""

from __future__ import annotations

from robata.application.canonical.media_quality import (
    LocalMediaQualityReport,
    registered_local_media_quality_report_document,
)
from robata.application.canonical.media_quality_binding import (
    derive_local_media_quality_binding_document,
)
from robata.application.canonical.media_quality_source_binding import (
    RegisteredMediaQualitySourceBinding,
)
from robata.contracts.common import Nanoseconds
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.sampling.supplemental import (
    FrozenSupplementalTargetPlan,
    build_frozen_supplemental_target_plan,
)


def freeze_registered_media_quality_targets(
    report: LocalMediaQualityReport,
    *,
    registry: SchemaRegistry,
    source_binding: RegisteredMediaQualitySourceBinding,
    selection_tolerance_ns: Nanoseconds,
    tie_break_policy_version: str,
    dedupe_policy_version: str,
) -> FrozenSupplementalTargetPlan | None:
    """Validate one report/source binding and freeze its nonempty target plan."""

    if not isinstance(source_binding, RegisteredMediaQualitySourceBinding):
        raise TypeError("source_binding must be a RegisteredMediaQualitySourceBinding")
    binding = RegisteredMediaQualitySourceBinding.model_validate(
        source_binding.model_dump(mode="python"),
        strict=True,
    )
    registry.resolve_exact(binding.report_schema_ref)

    document = registered_local_media_quality_report_document(report, registry)
    schema_ref = SchemaRef.model_validate(document["schema_ref"], strict=True)
    quality_binding = derive_local_media_quality_binding_document(document, registry)
    if (
        binding.report_schema_ref != schema_ref
        or binding.report_semantic_sha256 != report.semantic_sha256
        or binding.supplemental_target_plan_semantic_sha256
        != report.supplemental_targets.semantic_sha256
        or binding.media_quality_binding_semantic_sha256 != quality_binding.semantic_sha256
    ):
        raise ValueError("media-quality source binding does not bind the report")

    targets = report.supplemental_targets.targets
    if not targets:
        return None
    return build_frozen_supplemental_target_plan(
        source_binding=binding,
        effective_interval=report.supplemental_targets.interval,
        targets=tuple((target.camera_id, target.target_ns) for target in targets),
        selection_tolerance_ns=selection_tolerance_ns,
        tie_break_policy_version=tie_break_policy_version,
        dedupe_policy_version=dedupe_policy_version,
        target_policy_version=report.supplemental_targets.policy_version,
    )


__all__ = ["freeze_registered_media_quality_targets"]
