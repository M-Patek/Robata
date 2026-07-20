"""Provider-neutral index for immutable action-event revisions.

The implementation is intentionally in-memory so indexing semantics can be
verified without a database. Revisions are append-only; currentness is stored
as a separate selection projection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.hashing import semantic_sha256
from robata.retrieval.models import RetrievalQuery, RetrievalResult, RetrievalResultItem


class EventIndexError(ValueError):
    """An index operation violated an immutable-revision invariant."""


@dataclass(frozen=True, slots=True)
class _IndexedRevision:
    event_id: str
    revision_id: str
    record: Mapping[str, Any]
    search_text: str


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise EventIndexError(f"{field} must be a non-empty string")
    return value


def _nanoseconds(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise EventIndexError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
        return int(value)
    raise EventIndexError(f"{field} must be a canonical integer")


def _normalise(source: Mapping[str, Any]) -> dict[str, Any]:
    record = deepcopy(dict(source))
    for field in ("event_id", "event_revision_id", "mcap_id"):
        record[field] = _required_string(record, field)
    record["start_ns"] = _nanoseconds(record.get("start_ns"), "start_ns")
    record["end_ns"] = _nanoseconds(record.get("end_ns"), "end_ns")
    if record["start_ns"] >= record["end_ns"]:
        raise EventIndexError("event interval must be non-empty")

    for field in ("action_type", "active_hand", "object_class_id", "object_label"):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise EventIndexError(f"{field} must be non-empty when present")

    confidence = record.get("confidence_value")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise EventIndexError("confidence_value must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise EventIndexError("confidence_value must be within [0, 1]")
        record["confidence_value"] = confidence

    statuses = record.get("camera_statuses") or {}
    if not isinstance(statuses, Mapping):
        raise EventIndexError("camera_statuses must be a mapping")
    unknown = set(statuses) - set(CAMERA_ID_VALUES)
    if unknown:
        raise EventIndexError(f"unknown camera IDs: {sorted(unknown)}")
    record["camera_statuses"] = {str(key): str(value) for key, value in statuses.items()}
    usable = record.get("usable_camera_count")
    if usable is None:
        usable = sum(
            status.upper() in {"PASS", "SUPPORTING", "PARTIAL", "USABLE"}
            for status in record["camera_statuses"].values()
        )
    if isinstance(usable, bool) or not isinstance(usable, int) or not 0 <= usable <= 6:
        raise EventIndexError("usable_camera_count must be an integer in [0, 6]")
    record["usable_camera_count"] = usable

    record.pop("is_current", None)
    record.pop("current", None)
    text_fields = ("action_type", "active_hand", "object_class_id", "object_label", "text")
    record["_search_text"] = " ".join(
        str(record.get(field, "")) for field in text_fields
    ).casefold()
    record["_semantic_digest"] = semantic_sha256(
        {key: value for key, value in record.items() if not key.startswith("_")}
    )
    return record


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9_]+", query.casefold()))
    text_tokens = set(re.findall(r"[a-z0-9_]+", text.casefold()))
    return len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0.0


class EventIndex:
    """Append-only event revisions plus a deterministic current projection."""

    def __init__(self) -> None:
        self._revisions: dict[str, _IndexedRevision] = {}
        self._by_event: dict[str, list[str]] = {}
        self._current: dict[str, str] = {}
        self._selections: dict[str, list[tuple[str, int]]] = {}

    @property
    def revision_count(self) -> int:
        return len(self._revisions)

    def build_index(self, source: dict[str, Any]) -> None:
        if not isinstance(source, Mapping):
            raise EventIndexError("source must be a mapping")
        revisions = source.get("event_revisions", source.get("events", ()))
        selections = source.get("current_selections", source.get("selections", ()))
        if not isinstance(revisions, (list, tuple)) or not isinstance(selections, (list, tuple)):
            raise EventIndexError("revisions and selections must be sequences")
        self._revisions.clear()
        self._by_event.clear()
        self._current.clear()
        self._selections.clear()
        for revision in revisions:
            if not isinstance(revision, Mapping):
                raise EventIndexError("each revision must be a mapping")
            self.update_index(dict(revision), select=bool(revision.get("is_current", False)))
        for selection in selections:
            if not isinstance(selection, Mapping):
                raise EventIndexError("each selection must be a mapping")
            self.select_revision(
                event_id=_required_string(selection, "event_id"),
                revision_id=_required_string(selection, "selected_revision_id"),
                selection_decision_id=_required_string(selection, "selection_decision_id"),
                sequence=selection.get("selection_sequence"),
            )

    def update_index(self, event_revision: dict[str, Any], *, select: bool | None = None) -> None:
        if not isinstance(event_revision, Mapping):
            raise EventIndexError("event_revision must be a mapping")
        record = _normalise(event_revision)
        event_id = record["event_id"]
        revision_id = record["event_revision_id"]
        existing = self._revisions.get(revision_id)
        if existing is not None:
            if dict(existing.record) != record:
                raise EventIndexError("an existing revision cannot be mutated")
        else:
            self._revisions[revision_id] = _IndexedRevision(
                event_id=event_id,
                revision_id=revision_id,
                record=record,
                search_text=record["_search_text"],
            )
            self._by_event.setdefault(event_id, []).append(revision_id)
        should_select = bool(event_revision.get("is_current", False)) if select is None else select
        if should_select:
            sequence = len(self._selections.get(event_id, ())) + 1
            decision = semantic_sha256({"event_id": event_id, "revision_id": revision_id})[:32]
            self.select_revision(
                event_id=event_id,
                revision_id=revision_id,
                selection_decision_id=f"local-{decision}",
                sequence=sequence,
            )

    def select_revision(
        self,
        *,
        event_id: str,
        revision_id: str,
        selection_decision_id: str,
        sequence: object | None = None,
    ) -> None:
        revision = self._revisions.get(revision_id)
        if revision is None or revision.event_id != event_id:
            raise EventIndexError("selection must reference an owned revision")
        history = self._selections.setdefault(event_id, [])
        expected = len(history) + 1
        actual = expected if sequence is None else sequence
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise EventIndexError(f"selection_sequence must be {expected}")
        if not selection_decision_id:
            raise EventIndexError("selection_decision_id must be non-empty")
        history.append((selection_decision_id, actual))
        self._current[event_id] = revision_id

    def current_revision(self, event_id: str) -> Mapping[str, Any] | None:
        revision_id = self._current.get(event_id)
        revision = self._revisions.get(revision_id) if revision_id else None
        return deepcopy(dict(revision.record)) if revision else None

    def selection_history(self, event_id: str) -> tuple[tuple[str, int], ...]:
        return tuple(self._selections.get(event_id, ()))

    @staticmethod
    def _matches(record: Mapping[str, Any], query: RetrievalQuery) -> bool:
        filters = query.filters
        exact = {
            "action_type": filters.action_type,
            "active_hand": filters.active_hand,
            "object_class_id": filters.object_class_id,
            "object_label": filters.object_label,
            "mcap_id": filters.mcap_id,
        }
        if any(value is not None and record.get(field) != value for field, value in exact.items()):
            return False
        if filters.start_ns_min is not None and record["start_ns"] < filters.start_ns_min:
            return False
        if filters.start_ns_max is not None and record["start_ns"] > filters.start_ns_max:
            return False
        if filters.end_ns_min is not None and record["end_ns"] < filters.end_ns_min:
            return False
        if filters.end_ns_max is not None and record["end_ns"] > filters.end_ns_max:
            return False
        confidence = record.get("confidence_value")
        if filters.min_confidence is not None and (
            confidence is None or confidence < filters.min_confidence
        ):
            return False
        if (
            filters.min_usable_camera_count is not None
            and record["usable_camera_count"] < filters.min_usable_camera_count
        ):
            return False
        return all(
            record["camera_statuses"].get(camera_id) == expected
            for camera_id, expected in (filters.required_camera_status or {}).items()
        )

    def query_index(self, query: RetrievalQuery) -> RetrievalResult:
        candidates: list[tuple[_IndexedRevision, float | None]] = []
        for event_id, revisions in self._by_event.items():
            selected = self._current.get(event_id)
            for revision_id in revisions:
                if query.filters.require_current_revision and revision_id != selected:
                    continue
                revision = self._revisions[revision_id]
                if not self._matches(revision.record, query):
                    continue
                score = (
                    _lexical_score(query.semantic_query, revision.search_text)
                    if query.semantic_query
                    else None
                )
                if query.semantic_query and score == 0.0:
                    continue
                candidates.append((revision, score))
        candidates.sort(
            key=lambda item: (
                -(item[1] or 0.0) if query.semantic_query else 0.0,
                item[0].record["start_ns"],
                item[0].event_id,
                item[0].revision_id,
            )
        )
        total = len(candidates)
        page = candidates[query.offset : query.offset + query.limit]
        items = tuple(
            RetrievalResultItem(
                event_id=revision.event_id,
                event_revision_id=revision.revision_id,
                mcap_id=revision.record["mcap_id"],
                start_ns=revision.record["start_ns"],
                end_ns=revision.record["end_ns"],
                action_type=revision.record.get("action_type"),
                active_hand=revision.record.get("active_hand"),
                object_class_id=revision.record.get("object_class_id"),
                object_label=revision.record.get("object_label"),
                confidence_value=revision.record.get("confidence_value"),
                semantic_score=score,
            )
            for revision, score in page
        )
        return RetrievalResult(
            query=query,
            items=items,
            total=total,
            offset=query.offset,
            limit=query.limit,
            has_more=query.offset + len(items) < total,
        )


__all__ = ["EventIndex", "EventIndexError"]
