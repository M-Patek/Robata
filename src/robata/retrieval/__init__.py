"""Structured-first retrieval contracts and offline implementation."""

from robata.retrieval.index import EventIndex, EventIndexError
from robata.retrieval.models import (
    ClipArtifact,
    ClipManifest,
    RetrievalQuery,
    RetrievalResult,
    RetrievalResultItem,
    SearchFilter,
)
from robata.retrieval.service import RetrievalCapabilityError, RetrievalService

__all__ = [
    "ClipArtifact",
    "ClipManifest",
    "EventIndex",
    "EventIndexError",
    "RetrievalCapabilityError",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalResultItem",
    "RetrievalService",
    "SearchFilter",
]
