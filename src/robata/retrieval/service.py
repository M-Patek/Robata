"""Application boundary for structured retrieval and provenance resolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.retrieval import VectorSearchHit, VectorSearchQuery
from robata.ports.vector_projection import VectorProjectionError, VectorProjectionStore
from robata.retrieval.index import EventIndex, EventIndexError, EventIndexMembership
from robata.retrieval.models import (
    ClipManifest,
    RetrievalQuery,
    RetrievalResult,
    RetrievalResultItem,
)


class RetrievalCapabilityError(RuntimeError):
    """Raised when a requested retrieval operation has no configured resolver."""


ClipResolver = Callable[[str, tuple[str, ...] | None], ClipManifest]
ProvenanceResolver = Callable[[str], dict[str, Any]]
SemanticReranker = Callable[[RetrievalResult, Sequence[float]], RetrievalResult]
StructuredResultAuthorizer = Callable[[RetrievalResultItem, str], bool]


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
        vector_store: VectorProjectionStore | None = None,
        structured_result_authorizer: StructuredResultAuthorizer | None = None,
    ) -> None:
        self._index = index or EventIndex()
        self._clip_resolver = clip_resolver
        self._provenance_resolver = provenance_resolver
        self._semantic_reranker = semantic_reranker
        self._vector_store = vector_store
        self._structured_result_authorizer = structured_result_authorizer
        self._clips: dict[tuple[str, tuple[str, ...] | None], ClipManifest] = {}
        self._provenance: dict[str, dict[str, Any]] = {}

    @property
    def index(self) -> EventIndex:
        return self._index

    def build_index(self, source: dict[str, Any]) -> None:
        self._index.build_index(source)

    def apply_terminal_projection(
        self, projection: dict[str, Any]
    ) -> tuple[EventIndexMembership, ...]:
        """Apply an async terminal EventIndex projection idempotently."""

        return self._index.apply_projection(projection)

    # Compatibility aliases for adapters at the terminal-closure boundary.
    apply_event_revision_projection = apply_terminal_projection
    project_terminal = apply_terminal_projection
    register_terminal_projection = apply_terminal_projection

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

    @staticmethod
    def _page_result(result: RetrievalResult, query: RetrievalQuery) -> RetrievalResult:
        """Apply a public page after candidate authorization or reranking."""

        items = result.items[query.offset : query.offset + query.limit]
        return result.model_copy(
            update={
                "query": query,
                "items": items,
                "offset": query.offset,
                "limit": query.limit,
                "has_more": query.offset + len(items) < result.total,
            }
        )

    def _authorize_structured_result(
        self,
        result: RetrievalResult,
        *,
        tenant_id: str | None,
    ) -> RetrievalResult:
        """Filter an already-bounded candidate set before any vector lookup."""

        authorizer = self._structured_result_authorizer
        if authorizer is None:
            if tenant_id is not None:
                raise RetrievalCapabilityError(
                    "tenant-scoped retrieval requires a structured_result_authorizer"
                )
            return result
        if tenant_id is None:
            raise RetrievalCapabilityError(
                "tenant_id is required when structured tenant authorization is configured"
            )
        try:
            authorized = tuple(item for item in result.items if authorizer(item, tenant_id))
        except Exception as exc:
            raise RetrievalCapabilityError(
                f"structured tenant authorization failed: {exc}"
            ) from exc
        return result.model_copy(
            update={
                "items": authorized,
                "total": len(authorized),
                "offset": 0,
                "limit": 1000,
                "has_more": False,
            }
        )

    def query_events(
        self,
        query: RetrievalQuery,
        *,
        tenant_id: str | None = None,
    ) -> RetrievalResult:
        if not isinstance(query, RetrievalQuery):
            raise EventIndexError("query must be a RetrievalQuery")
        if self._structured_result_authorizer is None and tenant_id is None:
            return self._index.query_index(query)
        if query.offset + query.limit > 1000:
            raise RetrievalCapabilityError(
                "tenant-authorized structured queries are bounded to 1000 candidates"
            )
        candidate_query = query.model_copy(update={"offset": 0, "limit": 1000})
        result = self._index.query_index(candidate_query)
        return self._page_result(
            self._authorize_structured_result(result, tenant_id=tenant_id),
            query,
        )

    def semantic_search(
        self,
        query: RetrievalQuery,
        embedding_vector: list[float] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> RetrievalResult:
        if embedding_vector is None:
            return self.query_events(query, tenant_id=tenant_id)

        # Vector reranking must see the structured candidate set before
        # pagination; otherwise a high-scoring candidate on a later structured
        # page can never reach the requested result page.  The query contract
        # bounds this local candidate pool at 1000 rows.
        candidate_query = query.model_copy(update={"offset": 0, "limit": 1000})
        result = self.query_events(candidate_query, tenant_id=tenant_id)

        if self._vector_store is not None:
            # A structured query with no candidates is authoritative and must
            # not trigger an unbounded vector scan (or require an embedding
            # family that cannot affect an empty result).
            if not result.items:
                return self._page_result(result, query)
            if query.embedding_id is None:
                raise RetrievalCapabilityError("vector search requires RetrievalQuery.embedding_id")
            candidate_event_revision_ids = tuple(
                sorted({item.event_revision_id for item in result.items})
            )
            try:
                vector_result = self._vector_store.search(
                    VectorSearchQuery(
                        embedding_id=query.embedding_id,
                        query_vector=tuple(embedding_vector),
                        tenant_id=tenant_id,
                        # VectorSearchQuery requires canonical sorted/unique
                        # candidate IDs; structured result order remains the
                        # deterministic tie-breaker below.
                        candidate_event_revision_ids=candidate_event_revision_ids,
                        limit=max(len(result.items), 1),
                    )
                )
            except VectorProjectionError:
                # Projection infrastructure is optional. The structured set is
                # authoritative, so failure cannot turn valid retrieval into an
                # error or expand its tenant-bound candidate population.
                return self._page_result(result, query)
            except Exception:
                # A configured adapter is still optional infrastructure.  Do not
                # leak provider/database exceptions through the application API
                # or make optional projection availability an authority gate.
                return self._page_result(result, query)
            if not isinstance(vector_result, tuple) or any(
                not isinstance(hit, VectorSearchHit)
                or hit.embedding_id != query.embedding_id
                or hit.event_revision_id not in candidate_event_revision_ids
                for hit in vector_result
            ):
                return self._page_result(result, query)
            # A revision may have several derived artifact/package vectors.
            # Collapse them by the strongest matching artifact so a later,
            # lower-scoring row cannot overwrite the best score.
            score_by_revision: dict[str, float] = {}
            for hit in vector_result:
                prior = score_by_revision.get(hit.event_revision_id)
                if prior is None or hit.score > prior:
                    score_by_revision[hit.event_revision_id] = hit.score
            ranked: list[tuple[int, RetrievalResultItem]] = []
            for position, item in enumerate(result.items):
                score = score_by_revision.get(item.event_revision_id)
                ranked.append(
                    (
                        position,
                        item.model_copy(update={"semantic_score": score}),
                    )
                )
            ranked.sort(
                key=lambda pair: (
                    -(pair[1].semantic_score if pair[1].semantic_score is not None else -1.0),
                    pair[0],
                )
            )
            return self._page_result(
                result.model_copy(update={"items": tuple(item for _, item in ranked)}),
                query,
            )
        if self._semantic_reranker is None:
            raise RetrievalCapabilityError(
                "embedding reranking is unavailable without a configured semantic_reranker"
            )
        reranked = self._semantic_reranker(result, tuple(embedding_vector))
        if not isinstance(reranked, RetrievalResult):
            raise RetrievalCapabilityError("semantic_reranker returned an invalid result")
        return self._page_result(reranked, query)

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
