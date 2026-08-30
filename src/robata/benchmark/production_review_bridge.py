"""Bridge an agent review queue into an explicit review decision pack.

The production cohort intentionally starts without labels.  The visual-review
agent can therefore prepare *suggestions*, but it must never silently turn those
suggestions into gold.  This module provides the small, audit-light handoff:
an explicit reviewer decision file may accept, edit, split, reject, or abstain
each suggestion.  Only an explicit ``accept``/``edit``/``split`` decision with
reviewer provenance produces an ``ACCEPTED`` item in the ordinary production
review-pack shape.

The bridge performs no model inference, media decoding, ontology mapping, or
identity calculation.  It is a local benchmark helper; callers still decide
whether an accepted pack is suitable for a production claim.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


class ProductionReviewBridgeError(ValueError):
    """Raised when an agent review queue or decision file is malformed."""


_AGENT_FORMAT = "robata-production-agent-reviewed-segment-pack-v1"
_REVIEW_PACK_FORMAT = "robata-production-human-review-pack-v1"
_DECISION_FORMAT = "robata-production-review-decisions-v1"
_AUTHORITY = "LOCAL_NONPRODUCTION_ONLY"
_DECISIONS = frozenset({"accept", "edit", "split", "reject", "abstain", "pending"})
_LABEL_FIELDS = ("verb", "noun", "attributes", "location", "hand")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionReviewBridgeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionReviewBridgeError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionReviewBridgeError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionReviewBridgeError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ProductionReviewBridgeError(f"{field} must be a finite number")
    return number


def _normalise_decision(value: object, *, field: str) -> str:
    decision = _text(value, field=field).casefold()
    if decision not in _DECISIONS:
        raise ProductionReviewBridgeError(f"{field} must be one of {', '.join(sorted(_DECISIONS))}")
    return decision


def _assert_no_model_payload(value: object, *, field: str) -> None:
    """Reject model/gold payloads in the explicit decision envelope.

    Reviewer notes may mention models in prose, but a decision entry must not
    carry a nested prediction or gold object that could be mistaken for an
    independent reference.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProductionReviewBridgeError(f"{field} keys must be strings")
            key = "".join(character for character in raw_key.casefold() if character.isalnum())
            if key in {
                "modeloutputs",
                "predictions",
                "prediction",
                "gold",
                "groundtruth",
                "officialreference",
            }:
                raise ProductionReviewBridgeError(
                    f"{field}.{raw_key} cannot contain model or gold payload"
                )
            _assert_no_model_payload(child, field=f"{field}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_model_payload(child, field=f"{field}[{index}]")
        return
    raise ProductionReviewBridgeError(f"{field} must be JSON-compatible")


def _segment_from_agent(raw: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Project an agent suggestion to the production gold segment fields."""

    verb = _text(raw.get("verb"), field=f"{field}.verb")
    noun = _text(raw.get("noun"), field=f"{field}.noun")
    start = _number(raw.get("start_seconds"), field=f"{field}.start_seconds", minimum=0.0)
    end = _number(raw.get("end_seconds"), field=f"{field}.end_seconds", minimum=0.0)
    if end <= start:
        raise ProductionReviewBridgeError(f"{field}.end_seconds must exceed start_seconds")
    result: dict[str, Any] = {
        "verb": verb,
        "noun": noun,
        "attributes": raw.get("attributes"),
        "location": raw.get("location"),
        "hand": raw.get("hand"),
        "start_seconds": start,
        "end_seconds": end,
    }
    for key in _LABEL_FIELDS:
        value = result[key]
        if value is not None and not isinstance(value, str):
            raise ProductionReviewBridgeError(f"{field}.{key} must be a string or null")
    return result


def _decision_map(
    decisions: Mapping[str, Any],
) -> tuple[str, str, str, dict[str, Mapping[str, Any]]]:
    if decisions.get("format") != _DECISION_FORMAT:
        raise ProductionReviewBridgeError("unsupported decision format")
    if decisions.get("authority") != _AUTHORITY:
        raise ProductionReviewBridgeError("decision authority must be LOCAL_NONPRODUCTION_ONLY")
    reviewer_id = _text(decisions.get("reviewer_id"), field="decisions.reviewer_id")
    reviewed_at = _text(decisions.get("reviewed_at"), field="decisions.reviewed_at")
    source_pack = _text(decisions.get("source_agent_pack"), field="decisions.source_agent_pack")
    raw_rows = _sequence(decisions.get("decisions"), field="decisions.decisions")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, field=f"decisions.decisions[{index}]")
        window_id = _text(row.get("window_id"), field=f"decisions.decisions[{index}].window_id")
        if window_id in result:
            raise ProductionReviewBridgeError(f"duplicate decision for {window_id}")
        _assert_no_model_payload(row, field=f"decisions.decisions[{index}]")
        # Validate now so a malformed decision cannot be silently treated as pending.
        _normalise_decision(row.get("decision"), field=f"decisions.decisions[{index}].decision")
        result[window_id] = row
    return reviewer_id, reviewed_at, source_pack, result


def build_decision_template(agent_pack: Mapping[str, Any]) -> dict[str, Any]:
    """Create an explicit, all-pending decision envelope for a review queue."""

    if agent_pack.get("format") != _AGENT_FORMAT:
        raise ProductionReviewBridgeError("unsupported agent review pack format")
    if agent_pack.get("authority") != _AUTHORITY:
        raise ProductionReviewBridgeError("agent review pack authority is unsupported")
    items = _sequence(agent_pack.get("items"), field="agent_pack.items")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        item = _mapping(raw, field=f"agent_pack.items[{index}]")
        window_id = _text(item.get("window_id"), field=f"agent_pack.items[{index}].window_id")
        rows.append(
            {
                "window_id": window_id,
                "decision": "pending",
                "segments": [],
                "notes": "Review the agent suggestion; accept, edit, split, reject, or abstain.",
            }
        )
    return {
        "format": _DECISION_FORMAT,
        "authority": _AUTHORITY,
        "source_agent_pack": "",
        "reviewer_id": "",
        "reviewed_at": "",
        "decisions": rows,
        "contract": {
            "explicit_reviewer_decision_required": True,
            "agent_suggestions_are_not_gold": True,
            "model_predictions_are_not_accepted_as_decisions": True,
            "empty_decision_means_pending": True,
        },
    }


def apply_review_decisions(
    blank_review_pack: Mapping[str, Any],
    agent_pack: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply explicit review decisions to a blank production review pack.

    ``accept`` copies the agent segment suggestion after reviewer approval;
    ``edit`` and ``split`` require explicit replacement segments; ``reject`` and
    ``abstain`` remain outside the evaluator denominator.  Unmentioned windows
    remain pending.  No model output is copied into the resulting pack.
    """

    if blank_review_pack.get("format") != _REVIEW_PACK_FORMAT:
        raise ProductionReviewBridgeError("unsupported blank review-pack format")
    if blank_review_pack.get("authority") != _AUTHORITY:
        raise ProductionReviewBridgeError("blank review-pack authority is unsupported")
    if agent_pack.get("format") != _AGENT_FORMAT:
        raise ProductionReviewBridgeError("unsupported agent review pack format")
    if agent_pack.get("authority") != _AUTHORITY:
        raise ProductionReviewBridgeError("agent review pack authority is unsupported")
    reviewer_id, reviewed_at, source_pack, decision_rows = _decision_map(decisions)

    agent_by_window: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_sequence(agent_pack.get("items"), field="agent_pack.items")):
        item = _mapping(raw, field=f"agent_pack.items[{index}]")
        window_id = _text(item.get("window_id"), field=f"agent_pack.items[{index}].window_id")
        if window_id in agent_by_window:
            raise ProductionReviewBridgeError(f"duplicate agent window {window_id}")
        if item.get("accepted_as_gold") is True:
            raise ProductionReviewBridgeError(
                f"agent suggestion {window_id} is already marked gold"
            )
        agent_by_window[window_id] = item

    output = copy.deepcopy(dict(blank_review_pack))
    output["authority"] = _AUTHORITY
    output["production_eligible"] = False
    output.setdefault("bridge", {})
    output["bridge"] = {
        "format": "robata-production-review-bridge-v1",
        "source_agent_pack": source_pack,
        "explicit_decisions_applied": True,
        "agent_suggestions_are_not_gold_without_decision": True,
        "model_predictions_copied": False,
    }
    controls = dict(output.get("controls", {}))
    controls.update(
        {
            "labels_inferred": False,
            "model_predictions_copied": False,
            "agent_suggestions_reviewed": True,
            "ontology_modified": False,
            "mapper_modified": False,
            "sha_or_digest_computed": False,
        }
    )
    output["controls"] = controls

    seen_decisions: set[str] = set()
    for index, raw in enumerate(_sequence(output.get("items"), field="review_pack.items")):
        item = _mapping(raw, field=f"review_pack.items[{index}]")
        window_id = _text(item.get("window_id"), field=f"review_pack.items[{index}].window_id")
        decision_row = decision_rows.get(window_id)
        if decision_row is None:
            continue
        seen_decisions.add(window_id)
        decision = _normalise_decision(
            decision_row.get("decision"), field=f"decision.{window_id}.decision"
        )
        item_out = dict(item)
        gold = dict(_mapping(item_out.get("gold"), field=f"{window_id}.gold"))
        agent_item = agent_by_window.get(window_id)
        if agent_item is None:
            raise ProductionReviewBridgeError(
                f"decision references unknown agent window {window_id}"
            )
        if decision == "accept":
            raw_segments = _sequence(
                agent_item.get("segments"), field=f"agent.{window_id}.segments"
            )
            segments = [
                _segment_from_agent(
                    _mapping(value, field=f"agent.{window_id}.segments[{seg_index}]"),
                    field=f"agent.{window_id}.segments[{seg_index}]",
                )
                for seg_index, value in enumerate(raw_segments)
            ]
            if not segments:
                raise ProductionReviewBridgeError(
                    f"accept decision for {window_id} has no agent segments"
                )
            status = "ACCEPTED"
            adjudication_status = "ACCEPTED"
        elif decision in {"edit", "split"}:
            if decision_row.get("segments") is None:
                raise ProductionReviewBridgeError(
                    f"{decision} decision for {window_id} requires segments"
                )
            raw_segments = _sequence(
                decision_row.get("segments"), field=f"decision.{window_id}.segments"
            )
            if not raw_segments:
                raise ProductionReviewBridgeError(
                    f"{decision} decision for {window_id} requires segments"
                )
            segments = [
                _segment_from_agent(
                    _mapping(value, field=f"decision.{window_id}.segments[{seg_index}]"),
                    field=f"decision.{window_id}.segments[{seg_index}]",
                )
                for seg_index, value in enumerate(raw_segments)
            ]
            status = "ACCEPTED"
            adjudication_status = "ACCEPTED"
        elif decision == "reject":
            segments = []
            status = "REJECTED"
            adjudication_status = "COMPLETED"
        elif decision == "abstain":
            segments = []
            status = "ABSTAIN"
            adjudication_status = "COMPLETED"
        else:
            segments = []
            status = "PENDING_HUMAN_REVIEW"
            adjudication_status = "PENDING"

        gold.update(
            {
                "status": status,
                "segments": segments,
                "label_fields": list(_LABEL_FIELDS),
                "provenance": {
                    "source": source_pack or "agent_review_queue",
                    "reviewer_id": reviewer_id,
                    "reviewed_at": reviewed_at,
                    "adjudication_status": adjudication_status,
                    "decision": decision,
                    "agent_suggestion_used": decision == "accept",
                },
            }
        )
        item_out["gold"] = gold
        adjudication = dict(
            _mapping(item_out.get("adjudication", {}), field=f"{window_id}.adjudication")
        )
        adjudication.update(
            {
                "status": (
                    "COMPLETED"
                    if decision in {"accept", "edit", "split", "reject", "abstain"}
                    else "PENDING"
                ),
                "reviewer_a": reviewer_id if decision != "pending" else None,
                "reviewer_b": None,
                "decision": decision,
                "disagreement_notes": decision_row.get("notes"),
            }
        )
        item_out["adjudication"] = adjudication
        output["items"][index] = item_out

    unknown = sorted(set(decision_rows) - seen_decisions)
    if unknown:
        raise ProductionReviewBridgeError(
            "decision file references windows absent from blank review pack: " + ", ".join(unknown)
        )
    return output


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON object for the CLI without invoking a model or decoder."""

    from pathlib import Path

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionReviewBridgeError(f"could not read JSON {path}: {exc}") from exc
    return dict(_mapping(value, field=path))


__all__ = [
    "ProductionReviewBridgeError",
    "apply_review_decisions",
    "build_decision_template",
    "load_json",
]
