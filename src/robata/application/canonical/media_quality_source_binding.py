"""Bind one exact media-quality report to admitted source/alignment lineage."""

from __future__ import annotations

from typing import Any

from robata.application.canonical.media_quality import (
    LocalMediaQualityReport,
    registered_local_media_quality_report_document,
)
from robata.application.canonical.media_quality_binding import (
    derive_local_media_quality_binding_document,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.sampling.supplemental import (
    RegisteredMediaQualitySourceBinding,
    media_quality_source_binding_projection,
)

MEDIA_QUALITY_SOURCE_BINDING_PROJECTION_VERSION = "media-quality-source-binding-semantic-v1"


def bind_registered_media_quality_source(
    report: LocalMediaQualityReport,
    *,
    registry: SchemaRegistry,
    source_content_sha256: str,
    camera_mapping_semantic_sha256: str,
    alignment_semantic_sha256: str,
) -> RegisteredMediaQualitySourceBinding:
    """Validate report-derived evidence before attaching admitted source lineage."""

    document = registered_local_media_quality_report_document(report, registry)
    report_ref = SchemaRef.model_validate(document["schema_ref"], strict=True)
    registry.resolve_exact(report_ref)
    quality_binding = derive_local_media_quality_binding_document(document, registry)
    values: dict[str, Any] = {
        "report_schema_ref": report_ref,
        "report_semantic_sha256": quality_binding.report_semantic_sha256,
        "supplemental_target_plan_semantic_sha256": (
            quality_binding.supplemental_target_plan_semantic_sha256
        ),
        "media_quality_binding_semantic_sha256": quality_binding.semantic_sha256,
        "source_content_sha256": source_content_sha256,
        "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": alignment_semantic_sha256,
        "projection_version": MEDIA_QUALITY_SOURCE_BINDING_PROJECTION_VERSION,
    }
    draft = RegisteredMediaQualitySourceBinding.model_construct(
        semantic_sha256="0" * 64,
        **values,
    )
    return RegisteredMediaQualitySourceBinding.model_validate(
        {
            **values,
            "semantic_sha256": semantic_sha256(media_quality_source_binding_projection(draft)),
        },
        strict=True,
    )


__all__ = [
    "MEDIA_QUALITY_SOURCE_BINDING_PROJECTION_VERSION",
    "RegisteredMediaQualitySourceBinding",
    "bind_registered_media_quality_source",
    "media_quality_source_binding_projection",
]
