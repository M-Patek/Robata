"""Application boundary for structured retrieval and provenance resolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.retrieval.index import EventIndex, EventIndexError
from robata.retrieval.models import ClipManifest, RetrievalQuery, RetrievalResult


class RetrievalCapabilityError(RuntimeError):
    """Raised when a requested retrieval operation has no configured resolver."""


ClipResolver = Callable[[str, tuple[str, ...] | None], ClipManifest]
ProvenanceResolver = Callable[[str], dict[str, Any]]
SemanticReranker = Callable[[RetrievalResult, Sequence[float]], RetrievalResult]


class RetrievalService:
    """Structured-first retrieval with explicit optional extension points.

    The default implementation is completely offline. Semantic text queries
    use the deterministic lexical score in EventIndex; embedding reranking
    and clip extraction fail closed unless a resolver is supplied.
    """

    def __init__(
        self,
        *,
        index: EventIndex | None = None,
        clip_resolver: ClipResolver | None = None,
        provenance_resolver: ProvenanceResolver | None = None,
        semantic_reranker: SemanticReranker | None = None,
    ) -> None:
        self._index = index or EventIndex()
        self._clip_resolver = clip_resolver
        self._provenance_resolver = provenance_resolver
        self._semantic_reranker = semantic_reranker
        self._clips: dict[tuple[str, tuple[str, ...] | None], ClipManifest] = {}
        self._provenance: dict[str, dict[str, Any]] = {}

    @property
    def index(self) -> EventIndex:
        return self._index

    def build_index(self, source: dict[str, Any]) -> None:
        self._index.build_index(source)

    def register_event_revision(
        self,
        event_revision: dict[str, Any],
        *,
        select: bool | None = None,
    ) -> None:
        self._index.update_index(event_revision, select=select)

    def select_revision(
        self,
        *,
        event_id: str,
        revision_id: str,
        selection_decision_id: str,
        sequence: int | None = None,
    ) -> None:
        self._index.select_revision(
            event_id=event_id,
            revision_id=revision_id,
            selection_decision_id=selection_decision_id,
            sequence=sequence,
        )

    def query_events(self, query: RetrievalQuery) -> RetrievalResult:
        if not isinstance(query, RetrievalQuery):
            raise EventIndexError("query must be a RetrievalQuery")
        return self._index.query_index(query)

    def semantic_search(
        self,
        query: RetrievalQuery,
        embedding_vector: list[float] | None = None,
    ) -> RetrievalResult:
        result = self.query_events(query)
        if embedding_vector is None:
            return result
        if self._semantic_reranker is None:
            raise RetrievalCapabilityError(
                "embedding reranking is unavailable without a configured semantic_reranker"
            )
        reranked = self._semantic_reranker(result, tuple(embedding_vector))
        if not isinstance(reranked, RetrievalResult):
            raise RetrievalCapabilityError("semantic_reranker returned an invalid result")
        return reranked

    @staticmethod
    def _camera_mask(camera_mask: list[str] | None) -> tuple[str, ...] | None:
        if camera_mask is None:
            return None
        normalized = tuple(camera_mask)
        if len(set(normalized)) != len(normalized):
            raise EventIndexError("camera_mask cannot contain duplicates")
        unknown = set(normalized) - set(CAMERA_ID_VALUES)
        if unknown:
            raise EventIndexError(f"camera_mask contains unknown cameras: {sorted(unknown)}")
        return tuple(sorted(normalized))

    def register_clip_manifest(
        self,
        manifest: ClipManifest,
        *,
        camera_mask: list[str] | None = None,
    ) -> None:
        if not isinstance(manifest, ClipManifest):
            raise RetrievalCapabilityError("manifest must be a ClipManifest")
        mask = self._camera_mask(camera_mask)
        if manifest.start_ns >= manifest.end_ns:
            raise EventIndexError("clip interval must be non-empty")
        if any(
            artifact.camera_id is not None and artifact.camera_id not in CAMERA_ID_VALUES
            for artifact in manifest.clip_artifacts
        ):
            raise EventIndexError("clip manifest contains an unknown camera")
        key = (manifest.event_id, mask)
        existing = self._clips.get(key)
        if existing is not None and existing != manifest:
            raise EventIndexError("clip manifests are immutable")
        self._clips[key] = manifest

    def extract_clip(
        self,
        event_id: str,
        camera_mask: list[str] | None = None,
    ) -> ClipManifest:
        mask = self._camera_mask(camera_mask)
        if self._clip_resolver is not None:
            resolved_manifest = self._clip_resolver(event_id, mask)
            if not isinstance(resolved_manifest, ClipManifest):
                raise RetrievalCapabilityError("clip resolver returned an invalid manifest")
            return resolved_manifest
        registered_manifest = self._clips.get((event_id, mask))
        if registered_manifest is None:
            raise RetrievalCapabilityError(
                "clip extraction requires a registered manifest or configured resolver"
            )
        return registered_manifest

    def register_provenance(self, event_id: str, provenance: dict[str, Any]) -> None:
        if not event_id or not isinstance(provenance, dict):
            raise EventIndexError("event_id and provenance must be non-empty")
        existing = self._provenance.get(event_id)
        if existing is not None and existing != provenance:
            raise EventIndexError("provenance records are immutable")
        self._provenance[event_id] = dict(provenance)

    def resolve_provenance(self, event_id: str) -> dict[str, Any]:
        if self._provenance_resolver is not None:
            resolved_provenance = self._provenance_resolver(event_id)
            if not isinstance(resolved_provenance, dict):
                raise RetrievalCapabilityError("provenance resolver returned an invalid result")
            return dict(resolved_provenance)
        registered_provenance = self._provenance.get(event_id)
        if registered_provenance is None:
            raise RetrievalCapabilityError(
                "provenance resolution requires a registered record or configured resolver"
            )
        return dict(registered_provenance)


__all__ = ["RetrievalCapabilityError", "RetrievalService"]
