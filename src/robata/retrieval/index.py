"""Provider-neutral index for immutable action-event revisions.

The implementation is intentionally in-memory so indexing semantics can be
verified without a database. Revisions are append-only; currentness is stored
as a separate selection projection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.common import INT64_MAX, INT64_MIN
from robata.contracts.hashing import semantic_sha256
from robata.retrieval.models import RetrievalQuery, RetrievalResult, RetrievalResultItem


class EventIndexError(ValueError):
    """An index operation violated an immutable-revision invariant."""


# This is an internal projection version, not the ActionEvent/revision wire
# version.  It prevents an adapter from silently accepting a row shape it does
# not understand while keeping legacy hand-built rows (which omit the field)
# usable in the local index.
EVENT_INDEX_PROJECTION_VERSION = "canonical-event-index-projection-v1"


@dataclass(frozen=True, slots=True)
class _IndexedRevision:
    event_id: str
    revision_id: str
    record: Mapping[str, Any]
    search_text: str


@dataclass(frozen=True, slots=True)
class EventIndexMembership:
    """Replay receipt for one terminal revision projection.

    The key is the event/revision identity, not a transport locator or a
    processing-run UUID.  Replaying the same receipt is therefore a no-op;
    attempting to reuse the key with different semantic bytes is rejected.
    """

    event_id: str
    event_revision_id: str
    revision_semantic_sha256: str | None
    selection_decision_id: str | None = None
    selection_sequence: int | None = None

    @property
    def identity(self) -> str:
        """Stable, injective membership key independent of transport facts."""

        return (
            f"{len(self.event_id)}:{self.event_id}:"
            f"{len(self.event_revision_id)}:{self.event_revision_id}"
        )

    @property
    def revision_identity(self) -> tuple[str, str]:
        return self.event_id, self.event_revision_id


def _projection_version(record: Mapping[str, Any]) -> object:
    """Resolve the legacy and semantic spelling of the internal row version."""

    primary = record.get("projection_version")
    alias = record.get("semantic_projection_version")
    if primary is not None and alias is not None and primary != alias:
        raise EventIndexError("projection version spellings disagree")
    return primary if primary is not None else alias


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise EventIndexError(f"{field} must be a non-empty string")
    return value


def _nanoseconds(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise EventIndexError(f"{field} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
        parsed = int(value)
    else:
        raise EventIndexError(f"{field} must be a canonical integer")
    if parsed < INT64_MIN or parsed > INT64_MAX:
        raise EventIndexError(f"{field} must fit in signed int64")
    return parsed


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EventIndexError(f"{field} must be a lowercase SHA-256 digest when present")
    return value


def _normalise(source: Mapping[str, Any]) -> dict[str, Any]:
    record = deepcopy(dict(source))
    projection_version = _projection_version(record)
    if projection_version is not None and projection_version != EVENT_INDEX_PROJECTION_VERSION:
        raise EventIndexError(f"unsupported EventIndex projection version: {projection_version!r}")
    if projection_version is not None:
        record["projection_version"] = projection_version
        # Keep one canonical internal spelling so JSON/legacy alias replays
        # do not look like immutable row mutations.
        record.pop("semantic_projection_version", None)
    for field in ("event_id", "event_revision_id", "mcap_id"):
        record[field] = _required_string(record, field)
    for field in (
        "recording_identity",
        "revision_semantic_sha256",
        "payload_sha256",
        "lineage_sha256",
    ):
        value = record.get(field)
        if value is not None and (
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise EventIndexError(f"{field} must be a lowercase SHA-256 digest")
    record["start_ns"] = _nanoseconds(record.get("start_ns"), "start_ns")
    record["end_ns"] = _nanoseconds(record.get("end_ns"), "end_ns")
    if record["start_ns"] >= record["end_ns"]:
        raise EventIndexError("event interval must be non-empty")

    for field in ("action_type", "active_hand", "object_class_id", "object_label"):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise EventIndexError(f"{field} must be non-empty when present")
    text = record.get("text")
    if text is not None and (not isinstance(text, str) or not text):
        raise EventIndexError("text must be a non-empty string when present")

    confidence = record.get("confidence_value")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise EventIndexError("confidence_value must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise EventIndexError("confidence_value must be within [0, 1]")
        record["confidence_value"] = confidence

    statuses = record.get("camera_statuses")
    if statuses is None:
        statuses = {}
    if not isinstance(statuses, Mapping):
        raise EventIndexError("camera_statuses must be a mapping")
    if any(not isinstance(key, str) for key in statuses):
        raise EventIndexError("camera_statuses keys must be non-empty strings")
    unknown = set(statuses) - set(CAMERA_ID_VALUES)
    if unknown:
        raise EventIndexError(f"unknown camera IDs: {sorted(unknown)}")
    normalised_statuses: dict[str, str] = {}
    for key, value in statuses.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            raise EventIndexError("camera_statuses keys and values must be non-empty strings")
        normalised_statuses[key] = value
    record["camera_statuses"] = normalised_statuses
    usable = record.get("usable_camera_count")
    if usable is None:
        usable = sum(
            status.upper() in {"PASS", "SUPPORTING", "PARTIAL", "USABLE"}
            for status in record["camera_statuses"].values()
        )
    if isinstance(usable, bool) or not isinstance(usable, int) or not 0 <= usable <= 6:
        raise EventIndexError("usable_camera_count must be an integer in [0, 6]")
    record["usable_camera_count"] = usable
    for field in (
        "revision_semantic_sha256",
        "payload_sha256",
        "lineage_sha256",
        "recording_identity",
    ):
        record[field] = _optional_sha256(record.get(field), field)

    record.pop("is_current", None)
    record.pop("current", None)
    text_fields = ("action_type", "active_hand", "object_class_id", "object_label", "text")
    record["_search_text"] = " ".join(
        str(record.get(field, "")) for field in text_fields
    ).casefold()
    record["_semantic_digest"] = semantic_sha256(
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
            and key not in {"projection_version", "semantic_projection_version"}
        }
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
        self._selection_by_id: dict[str, tuple[str, str, int]] = {}
        self._memberships: dict[tuple[str, str], EventIndexMembership] = {}

    @property
    def revision_count(self) -> int:
        return len(self._revisions)

    def build_index(self, source: dict[str, Any]) -> None:
        if not isinstance(source, Mapping):
            raise EventIndexError("source must be a mapping")
        projection_version = _projection_version(source)
        if projection_version is not None and projection_version != EVENT_INDEX_PROJECTION_VERSION:
            raise EventIndexError(
                f"unsupported EventIndex projection version: {projection_version!r}"
            )
        revisions = source.get("event_revisions", source.get("events", ()))
        selections = source.get("current_selections", source.get("selections", ()))
        if not isinstance(revisions, (list, tuple)) or not isinstance(selections, (list, tuple)):
            raise EventIndexError("revisions and selections must be sequences")
        self._revisions.clear()
        self._by_event.clear()
        self._current.clear()
        self._selections.clear()
        self._selection_by_id.clear()
        self._memberships.clear()
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
            # Compare canonical semantic bytes rather than Python container
            # types: a JSON replay may decode a tuple as a list while retaining
            # exactly the same immutable row projection.
            if existing.record.get("_semantic_digest") != record.get("_semantic_digest"):
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
        for field, value in (
            ("event_id", event_id),
            ("revision_id", revision_id),
            ("selection_decision_id", selection_decision_id),
        ):
            if not isinstance(value, str) or not value:
                raise EventIndexError(f"{field} must be a non-empty string")
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
        prior = self._selection_by_id.get(selection_decision_id)
        if prior is not None and prior[:2] != (event_id, revision_id):
            # A decision ID is an identity-bearing fact. Keep the legacy
            # direct API's ability to append repeated observations for the
            # same event/revision, but never let the same decision be reused
            # for a different owner.
            raise EventIndexError("selection decision identity was reused for another revision")
        history.append((selection_decision_id, actual))
        self._selection_by_id.setdefault(selection_decision_id, (event_id, revision_id, actual))
        self._current[event_id] = revision_id

    def current_revision(self, event_id: str) -> Mapping[str, Any] | None:
        revision_id = self._current.get(event_id)
        revision = self._revisions.get(revision_id) if revision_id else None
        return deepcopy(dict(revision.record)) if revision else None

    def selection_history(self, event_id: str) -> tuple[tuple[str, int], ...]:
        return tuple(self._selections.get(event_id, ()))

    @property
    def membership_count(self) -> int:
        """Number of terminal event-revision memberships applied to this index."""

        return len(self._memberships)

    def membership(self, event_id: str, revision_id: str) -> EventIndexMembership | None:
        """Return the immutable replay receipt for one event revision."""

        return self._memberships.get((event_id, revision_id))

    def membership_for_revision(self, revision_id: str) -> EventIndexMembership | None:
        """Resolve a globally unique revision ID, if present."""

        matches = [
            item
            for (event_id, candidate), item in self._memberships.items()
            if candidate == revision_id
        ]
        if len(matches) > 1:
            raise EventIndexError("event_revision_id is not unique in the index")
        return matches[0] if matches else None

    def memberships(self) -> tuple[EventIndexMembership, ...]:
        """Return memberships in deterministic event/revision order."""

        return tuple(
            self._memberships[key]
            for key in sorted(self._memberships, key=lambda item: (item[0], item[1]))
        )

    def apply_projection(self, projection: Mapping[str, Any]) -> tuple[EventIndexMembership, ...]:
        """Apply a terminal EventIndex projection idempotently.

        ``projection`` follows the plain mapping emitted by
        ``canonical_event_index_batch_projection`` (``event_revisions`` plus
        ``current_selections``).  The operation is transactional: an invalid
        row or conflicting replay restores the prior index state.
        """

        if not isinstance(projection, Mapping):
            dump = getattr(projection, "model_dump", None)
            if not callable(dump):
                raise EventIndexError("projection must be a mapping")
            projection = dump(mode="json")
            if not isinstance(projection, Mapping):
                raise EventIndexError("projection model did not produce a mapping")
        projection_version = _projection_version(projection)
        if projection_version is not None and projection_version != EVENT_INDEX_PROJECTION_VERSION:
            raise EventIndexError(
                f"unsupported EventIndex projection version: {projection_version!r}"
            )
        # Accept canonical preparation models directly while keeping the
        # retrieval package free of a hard import at module load time.
        if (
            "publications" in projection
            or "action_event_publications" in projection
            or "detail" in projection
        ):
            from robata.application.canonical.projections import (
                canonical_event_index_batch_projection,
            )

            projection = canonical_event_index_batch_projection(projection)
        elif "payload" in projection and "revision" in projection:
            from robata.application.canonical.projections import (
                canonical_event_index_revision_projection,
            )

            projection = canonical_event_index_revision_projection(projection)
        revisions: Sequence[object]
        selections: Sequence[object]
        if "event_id" in projection and "event_revision_id" in projection:
            # Accept a single canonical row as a convenience; terminal
            # adapters normally send the batch shape below.
            revisions = (projection,)
            selections = ()
        else:
            revisions_value = projection.get(
                "event_revisions", projection.get("revisions", projection.get("events", ()))
            )
            selections_value = projection.get(
                "current_selections", projection.get("selections", projection.get("current", ()))
            )
            if not isinstance(revisions_value, (list, tuple)) or not isinstance(
                selections_value, (list, tuple)
            ):
                raise EventIndexError("projection revisions and selections must be sequences")
            revisions = revisions_value
            selections = selections_value
        snapshot = (
            dict(self._revisions),
            {event_id: list(items) for event_id, items in self._by_event.items()},
            dict(self._current),
            {event_id: list(items) for event_id, items in self._selections.items()},
            dict(self._selection_by_id),
            dict(self._memberships),
        )
        try:
            pending: list[EventIndexMembership] = []
            nested_selections: list[Mapping[str, Any]] = []
            for revision in revisions:
                if not isinstance(revision, Mapping):
                    raise EventIndexError("each projected revision must be a mapping")
                # A single-revision projection may carry its current-selection
                # command inline.  Keep it separate from immutable row bytes.
                row = dict(revision)
                nested = row.pop("selection", None)
                if nested is not None:
                    if not isinstance(nested, Mapping):
                        raise EventIndexError("projected selection must be a mapping")
                    nested_selections.append(nested)
                receipt = self._apply_projected_revision(row)
                pending.append(receipt)
            for selection in (*nested_selections, *selections):
                if not isinstance(selection, Mapping):
                    raise EventIndexError("each projected selection must be a mapping")
                event_id = _required_string(selection, "event_id")
                revision_id = _required_string(selection, "selected_revision_id")
                decision_id = _required_string(selection, "selection_decision_id")
                sequence = selection.get("selection_sequence")
                selection_receipt = self._apply_idempotent_selection(
                    event_id=event_id,
                    revision_id=revision_id,
                    selection_decision_id=decision_id,
                    sequence=sequence,
                )
                existing = self._memberships.get((event_id, revision_id))
                if existing is None:
                    # A selection may legally arrive in a separate replay batch
                    # after its immutable revision row.
                    existing = EventIndexMembership(
                        event_id=event_id,
                        event_revision_id=revision_id,
                        revision_semantic_sha256=(
                            self._revisions[revision_id].record.get("revision_semantic_sha256")
                            if revision_id in self._revisions
                            else None
                        ),
                    )
                updated = EventIndexMembership(
                    event_id=existing.event_id,
                    event_revision_id=existing.event_revision_id,
                    revision_semantic_sha256=existing.revision_semantic_sha256,
                    selection_decision_id=selection_receipt[0],
                    selection_sequence=selection_receipt[1],
                )
                self._memberships[(event_id, revision_id)] = updated
                pending.append(updated)
            return tuple(
                self._memberships[key]
                for key in sorted(
                    {(item.event_id, item.event_revision_id) for item in pending},
                    key=lambda item: (item[0], item[1]),
                )
            )
        except Exception:
            (
                revisions_snapshot,
                by_event_snapshot,
                current_snapshot,
                selections_snapshot,
                selection_by_id_snapshot,
                memberships_snapshot,
            ) = snapshot
            self._revisions = revisions_snapshot
            self._by_event = by_event_snapshot
            self._current = current_snapshot
            self._selections = selections_snapshot
            self._selection_by_id = selection_by_id_snapshot
            self._memberships = memberships_snapshot
            raise

    # Explicit aliases used by adapters at the terminal-closure boundary.
    apply_terminal_projection = apply_projection
    apply_event_revision_projection = apply_projection
    register_terminal_projection = apply_projection
    project_terminal = apply_projection

    def _apply_projected_revision(self, revision: Mapping[str, Any]) -> EventIndexMembership:
        projection_version = _projection_version(revision)
        if projection_version is not None and projection_version != EVENT_INDEX_PROJECTION_VERSION:
            raise EventIndexError(
                f"unsupported EventIndex projection version: {projection_version!r}"
            )
        record = _normalise(revision)
        event_id = record["event_id"]
        revision_id = record["event_revision_id"]
        existing = self._revisions.get(revision_id)
        if existing is not None:
            if existing.event_id != event_id:
                raise EventIndexError("an event revision identity cannot change event ownership")
            # Compare canonical semantic bytes rather than Python container
            # types: a JSON replay may decode a tuple as a list while retaining
            # exactly the same immutable row projection.
            if existing.record.get("_semantic_digest") != record.get("_semantic_digest"):
                raise EventIndexError("an existing revision cannot be mutated")
        else:
            self._revisions[revision_id] = _IndexedRevision(
                event_id=event_id,
                revision_id=revision_id,
                record=record,
                search_text=record["_search_text"],
            )
            self._by_event.setdefault(event_id, []).append(revision_id)
        semantic = record.get("revision_semantic_sha256")
        if semantic is not None and not isinstance(semantic, str):
            raise EventIndexError("revision_semantic_sha256 must be a string when present")
        key = (event_id, revision_id)
        prior = self._memberships.get(key)
        if prior is not None and prior.revision_semantic_sha256 != semantic:
            raise EventIndexError("event revision membership semantic identity changed")
        receipt = prior or EventIndexMembership(event_id, revision_id, semantic)
        self._memberships[key] = receipt
        return receipt

    def _apply_idempotent_selection(
        self,
        *,
        event_id: str,
        revision_id: str,
        selection_decision_id: str,
        sequence: object,
    ) -> tuple[str, int]:
        revision = self._revisions.get(revision_id)
        if revision is None or revision.event_id != event_id:
            raise EventIndexError("selection must reference an owned revision")
        prior = self._selection_by_id.get(selection_decision_id)
        if prior is not None:
            if prior[:2] != (event_id, revision_id) or (
                sequence is not None and sequence != prior[2]
            ):
                raise EventIndexError(
                    "selection decision identity was replayed with different facts"
                )
            return selection_decision_id, prior[2]
        if sequence is None:
            sequence = len(self._selections.get(event_id, ())) + 1
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise EventIndexError("selection_sequence must be an integer")
        history = self._selections.setdefault(event_id, [])
        expected = len(history) + 1
        if sequence != expected:
            raise EventIndexError(f"selection_sequence must be {expected}")
        history.append((selection_decision_id, sequence))
        self._selection_by_id[selection_decision_id] = (event_id, revision_id, sequence)
        self._current[event_id] = revision_id
        return selection_decision_id, sequence

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


__all__ = [
    "EVENT_INDEX_PROJECTION_VERSION",
    "EventIndex",
    "EventIndexError",
    "EventIndexMembership",
]
