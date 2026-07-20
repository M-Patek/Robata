"""Retrieval models for structured event and clip queries.

Implements the retrieval contract from ARCHITECTURE_DESIGN_V1.md Section 16.5.
All models inherit StrictModel (frozen=True, extra="forbid", strict=True).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel

# ---------------------------------------------------------------------------
# Reusable field types
# ---------------------------------------------------------------------------
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


# ---------------------------------------------------------------------------
# Search filter
# ---------------------------------------------------------------------------
class SearchFilter(StrictModel):
    """Structured filter predicates for event retrieval.

    Filters are applied in order: action/hand/object first, then temporal,
    QA, confidence, and camera-coverage constraints.
    """

    # Action-level filters
    action_type: NonEmptyString | None = None
    active_hand: Literal["LEFT", "RIGHT", "BOTH", "UNKNOWN"] | None = None
    object_class_id: NonEmptyString | None = None
    object_label: NonEmptyString | None = None

    # Temporal filters
    start_ns_min: Nanoseconds | None = None
    start_ns_max: Nanoseconds | None = None
    end_ns_min: Nanoseconds | None = None
    end_ns_max: Nanoseconds | None = None

    # Recording filters
    mcap_id: NonEmptyString | None = None

    # QA / confidence filters
    min_confidence: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] | None = None
    require_current_revision: bool = True

    # Camera-coverage filters
    min_usable_camera_count: Annotated[int, Field(strict=True, ge=0, le=6)] | None = None
    required_camera_status: dict[str, NonEmptyString] | None = None


# ---------------------------------------------------------------------------
# Retrieval query
# ---------------------------------------------------------------------------
class RetrievalQuery(StrictModel):
    """A structured query for event retrieval.

    Structured filters are applied first; optional semantic reranking
    may follow.
    """

    schema_version: Literal["1.0"] = "1.0"
    filters: SearchFilter = Field(default_factory=SearchFilter)

    # Optional semantic / free-text expansion
    semantic_query: NonEmptyString | None = None
    embedding_id: NonEmptyString | None = None

    # Pagination
    limit: Annotated[int, Field(strict=True, ge=1, le=1000)] = 50
    offset: Annotated[int, Field(strict=True, ge=0)] = 0


# ---------------------------------------------------------------------------
# Clip artifact
# ---------------------------------------------------------------------------
class ClipArtifact(StrictModel):
    """One artifact within a generated clip manifest.

    Represents a single camera's clip or a synchronized multi-camera
    package artifact.
    """

    artifact_id: NonEmptyString
    camera_id: NonEmptyString | None = None
    uri: NonEmptyString
    sha256: NonEmptyString
    bytes: Annotated[int, Field(strict=True, ge=0)]
    media_type: NonEmptyString
    format: NonEmptyString
    trim_policy_version: SchemaVersion
    effective_start_ns: Nanoseconds
    effective_end_ns: Nanoseconds


# ---------------------------------------------------------------------------
# Clip manifest
# ---------------------------------------------------------------------------
class ClipManifest(StrictModel):
    """A fully traceable clip manifest for event playback.

    Resolves from event revision -> [start_ns, end_ns) -> alignment version
    -> six camera streams -> source MCAP byte/frame locations.
    """

    schema_version: Literal["1.0"] = "1.0"
    clip_manifest_id: NonEmptyString
    event_id: NonEmptyString
    event_revision_id: NonEmptyString
    mcap_id: NonEmptyString
    alignment_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString

    # Temporal bounds
    start_ns: Nanoseconds
    end_ns: Nanoseconds

    # Source provenance
    source_stream_artifacts: tuple[NonEmptyString, ...]
    source_manifest_digest: NonEmptyString

    # Generated clip artifacts (one per camera or one synchronized package)
    clip_artifacts: tuple[ClipArtifact, ...]

    # Metadata
    extractor_version: SchemaVersion
    created_at: NonEmptyString


# ---------------------------------------------------------------------------
# Retrieval result item
# ---------------------------------------------------------------------------
class RetrievalResultItem(StrictModel):
    """One event result within a retrieval response."""

    event_id: NonEmptyString
    event_revision_id: NonEmptyString
    mcap_id: NonEmptyString
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    action_type: NonEmptyString | None = None
    active_hand: NonEmptyString | None = None
    object_class_id: NonEmptyString | None = None
    object_label: NonEmptyString | None = None
    confidence_value: float | None = None
    semantic_score: float | None = None


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------
class RetrievalResult(StrictModel):
    """The result of a retrieval query."""

    schema_version: Literal["1.0"] = "1.0"
    query: RetrievalQuery
    items: tuple[RetrievalResultItem, ...]
    total: Annotated[int, Field(strict=True, ge=0)]
    offset: Annotated[int, Field(strict=True, ge=0)]
    limit: Annotated[int, Field(strict=True, ge=1)]
    has_more: bool


__all__ = [
    "ClipArtifact",
    "ClipManifest",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalResultItem",
    "SearchFilter",
]
