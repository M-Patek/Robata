"""Local full-chain requirements preparation pipeline.

This orchestration connects QA -> feed-once frame cache -> annotation -> zero-GPU search without
introducing a real model/provider.  It is intentionally deterministic and suitable for acceptance
fixtures; model-backed adapters can be injected behind ``AnnotationPrincipal`` later.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robata.annotation import AnnotationBatchResult, AnnotationPipeline, AnnotationPrincipal
from robata.contracts.hashing import canonical_json_bytes
from robata.frame_cache import FrameFeedManifest, FramePayload, SharedFrameCache
from robata.qa import ClipMark, QAAssessment, QAClassifier
from robata.search import ClipSearchIndex, SearchHit


@dataclass(frozen=True, slots=True)
class RequirementsRunResult:
    assessment: QAAssessment
    frame_manifest: FrameFeedManifest | None
    annotations: AnnotationBatchResult
    search_index: ClipSearchIndex
    output_directory: Path | None
    provider_requests: int = 0
    execution_mode: str = "LOCAL_DEVELOPMENT_FAKE_MODEL"
    production_eligible: bool = False

    @property
    def search_entries(self) -> int:
        return len(self.search_index.entries())

    def search(self, query: str | Mapping[str, Any], *, limit: int = 50) -> tuple[SearchHit, ...]:
        return self.search_index.search(query, limit=limit)


class LocalRequirementsPipeline:
    """Run all non-provider requirements stages for one recording."""

    def __init__(
        self,
        *,
        cache: SharedFrameCache | None = None,
        qa_classifier: QAClassifier | None = None,
        annotation_principal: AnnotationPrincipal | None = None,
    ) -> None:
        self.cache = cache
        self.qa_classifier = qa_classifier or QAClassifier()
        self.annotation_pipeline = AnnotationPipeline(annotation_principal)

    def run(
        self,
        recording_id: str,
        duration_sec: float,
        clip_marks: Iterable[ClipMark | Mapping[str, object]] = (),
        *,
        source_uri: str | None = None,
        decoder: Callable[
            [], Iterable[FramePayload | bytes | Mapping[str, Any] | tuple[float, bytes]]
        ]
        | None = None,
        frame_rate: float = 2.0,
        output_directory: Path | None = None,
    ) -> RequirementsRunResult:
        assessment = self.qa_classifier.assess(recording_id, duration_sec, clip_marks)
        frame_manifest: FrameFeedManifest | None = None
        if decoder is not None:
            if source_uri is None:
                raise ValueError("source_uri is required when decoder is supplied")
            if self.cache is None:
                raise ValueError("cache is required when decoder is supplied")
            feed = self.cache.feed_once(recording_id, source_uri, decoder, frame_rate=frame_rate)
            frame_manifest = feed.manifest
        frame_manifests = {recording_id: frame_manifest} if frame_manifest else None
        annotations = self.annotation_pipeline.run((assessment,), frame_manifests=frame_manifests)
        search_index = ClipSearchIndex(annotations.drafts)
        result = RequirementsRunResult(
            assessment=assessment,
            frame_manifest=frame_manifest,
            annotations=annotations,
            search_index=search_index,
            output_directory=output_directory,
        )
        if output_directory is not None:
            self._publish(result, output_directory)
        return result

    @staticmethod
    def _publish(result: RequirementsRunResult, output_directory: Path) -> None:
        if not isinstance(output_directory, Path) or output_directory.name in {"", ".", ".."}:
            raise ValueError("output_directory must be a named Path")
        target = Path(os.path.abspath(output_directory))
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.mkdir(parents=True, exist_ok=False)
        try:
            records: dict[str, Any] = {
                "qa-assessment.json": result.assessment,
                "annotations.json": result.annotations,
                "search-index.json": result.search_index.entries(),
                "requirements-run.json": {
                    "provider_requests": result.provider_requests,
                    "execution_mode": result.execution_mode,
                    "production_eligible": result.production_eligible,
                    "recording_id": result.assessment.recording_id,
                    "qa_status": result.assessment.status.value,
                    "search_entries": result.search_entries,
                },
            }
            if result.frame_manifest is not None:
                records["frame-manifest.json"] = result.frame_manifest
            for filename, value in records.items():
                path = target / filename
                payload = canonical_json_bytes(value)
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception:
            for path in target.iterdir():
                path.unlink(missing_ok=True)
            target.rmdir()
            raise


RequirementsPipeline = LocalRequirementsPipeline
FullRequirementsPipeline = LocalRequirementsPipeline

__all__ = [
    "FullRequirementsPipeline",
    "LocalRequirementsPipeline",
    "RequirementsPipeline",
    "RequirementsRunResult",
]
