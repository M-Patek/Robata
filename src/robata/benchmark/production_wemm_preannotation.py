"""Build a label-neutral WeMM production pre-annotation envelope.

WeMM is an embedding/retrieval model, so it can only rank text phrases that a
caller supplies; it cannot manufacture a production ontology by itself.  This
module provides the small, benchmark-local envelope around that fact.  Phrase
rows are explicitly provisional and may be arbitrary production-facing text,
``unknown`` and ``abstain`` are first-class outcomes, and human review remains
required for every proposal.

The module is intentionally inference-free.  A future WeMM runner can feed its
raw camera rankings to :func:`build_preannotation_envelope`, while tests and
review tooling can use the same contract without importing torch or decoding
media.  EPIC action IDs are rejected at this boundary so an EPIC benchmark
sidecar cannot silently become a production pre-annotation.
"""

from __future__ import annotations

import copy
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from .production_wemm_interval_proposal import (
    BOUNDARY_SOURCE as TEMPORAL_BOUNDARY_SOURCE,
)
from .production_wemm_interval_proposal import (
    BOUNDARY_STATUS as TEMPORAL_BOUNDARY_STATUS,
)
from .production_wemm_temporal import FORMAT as TEMPORAL_RESOLUTION_FORMAT
from .production_wemm_temporal_refinement import (
    APPLIED_FORMAT as TEMPORAL_REFINEMENT_APPLIED_FORMAT,
)
from .production_wemm_temporal_refinement import (
    FORMAT as TEMPORAL_REFINEMENT_PLAN_FORMAT,
)
from .production_wemm_temporal_score_refinement import (
    FORMAT as TEMPORAL_SCORE_REFINEMENT_FORMAT,
)
from .production_wemm_temporal_score_refinement import (
    RESULT_FORMAT as TEMPORAL_SCORE_RESULT_FORMAT,
)

FORMAT: Final = "robata-production-wemm-preannotation-v1"
REVIEW_FORMAT: Final = "robata-production-wemm-preannotation-review-pack-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "REVIEW_REQUIRED"
LABEL_SPACE: Final = "OPEN_PROVISIONAL_PHRASES"
DECISIONS: Final = ("pending", "accept", "edit", "split", "reject", "abstain")
PROPOSAL_STATUSES: Final = ("PROPOSED", "UNKNOWN", "ABSTAIN", "SPLIT")
BOUNDARY_STATUSES: Final = (
    "MEASURED",
    "WINDOW_BOUND_ONLY",
    "NOT_MEASURED",
    "NOT_OBSERVABLE",
)
REFINED_BOUNDARY_STATUSES: Final = ("MODEL_REFINED", "MODEL_REFINEMENT_PENDING")
FIELD_STATUSES: Final = ("MEASURED", "NOT_MEASURED", "NOT_OBSERVABLE")
TEMPORAL_BOUNDARY_METHODS: Final = frozenset(
    {"observed_probe_span", "probe_center_midpoint", "mixed_probe_boundary"}
)
TEMPORAL_REFINEMENT_BOUNDARY_SOURCE: Final = "wemm_short_refinement"
TEMPORAL_REFINEMENT_BOUNDARY_METHOD: Final = "short_probe_model"

# Adaptive temporal sidecars are deliberately kept separate from the dense
# ``temporal_resolution``/``temporal_segments`` contract.  The runner emits
# these fields only when the opt-in adaptive route is selected; older readers
# can therefore ignore them without changing the historical window shape.
TEMPORAL_REFINEMENT_FIELDS: Final = (
    "temporal_refinement_plan",
    "temporal_refinement_fine_plan",
    "temporal_refinement_score_resolution",
    "temporal_refinement",
    "refined_segments",
    "refined_temporal_segments",
    "temporal_refinement_segments",
)
TEMPORAL_SIDECAR_FIELDS: Final = (
    "temporal_resolution",
    "temporal_segments",
    *TEMPORAL_REFINEMENT_FIELDS,
)

# These are identity-bearing fields from the EPIC action catalog.  Production
# proposals use a free ``provisional_id`` (or no ID); accepting an EPIC pair
# here would make provenance impossible to audit later.
_EPIC_KEYS: Final = frozenset(
    {
        "actionid",
        "verbid",
        "nounid",
        "epicactionkey",
        "epicontology",
        "ontologyid",
    }
)
_GOLD_KEYS: Final = frozenset(
    {
        "gold",
        "groundtruth",
        "officiallabel",
        "humanannotation",
        "adjudicatedlabel",
    }
)


class ProductionWemmPreannotationError(ValueError):
    """Raised when a pre-annotation input violates the review-only contract."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmPreannotationError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmPreannotationError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmPreannotationError(f"{field} must be a string")
    result = unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise ProductionWemmPreannotationError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionWemmPreannotationError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionWemmPreannotationError(f"{field} must be finite")
    return result


def _optional_finite(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field=field)


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Copy JSON-compatible model output without interpreting its semantics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmPreannotationError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmPreannotationError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmPreannotationError(f"{field} must be JSON-compatible")


def _key_token(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _assert_no_epic_or_gold(value: object, *, field: str = "input") -> None:
    """Reject catalog identity/gold containers while allowing arbitrary text."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProductionWemmPreannotationError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProductionWemmPreannotationError(f"{field} keys must be strings")
            token = _key_token(raw_key)
            if token in {"epicontology", "epicactionkey"}:
                raise ProductionWemmPreannotationError(
                    f"{field}.{raw_key} declares EPIC ontology provenance"
                )
            if token == "epicontologyused" and child is True:
                raise ProductionWemmPreannotationError(
                    f"{field}.{raw_key} declares EPIC ontology use"
                )
            # ``action_key``/``verb_key``/``noun_key`` are also used by local
            # production sidecars as opaque string identifiers.  Reject only
            # the numeric pair shape characteristic of EPIC class IDs.
            if token in {"actionkey", "verbkey", "nounkey"}:
                if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                    values = list(child)
                    numeric = all(
                        isinstance(item, int) and not isinstance(item, bool) for item in values
                    )
                    pair_shape = token == "actionkey" and len(values) == 2
                    scalar_shape = token in {"verbkey", "nounkey"} and len(values) == 1
                    if numeric and (pair_shape or scalar_shape):
                        raise ProductionWemmPreannotationError(
                            f"{field}.{raw_key} contains an EPIC ontology identity; "
                            "use provisional_id"
                        )
                elif isinstance(child, int) and not isinstance(child, bool):
                    raise ProductionWemmPreannotationError(
                        f"{field}.{raw_key} contains an EPIC ontology identity; use provisional_id"
                    )
            if token in _EPIC_KEYS:
                raise ProductionWemmPreannotationError(
                    f"{field}.{raw_key} is an EPIC ontology identity; use provisional_id"
                )
            if token in _GOLD_KEYS or any(
                fragment in token
                for fragment in ("groundtruth", "officiallabel", "humanannotation")
            ):
                raise ProductionWemmPreannotationError(
                    f"{field}.{raw_key} contains gold/review data"
                )
            _assert_no_epic_or_gold(child, field=f"{field}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_epic_or_gold(child, field=f"{field}[{index}]")
        return
    raise ProductionWemmPreannotationError(f"{field} must be JSON-compatible")


def _field(value: object, *, field: str, status: object = None) -> dict[str, Any]:
    """Normalize one structured field without deriving it from label text."""

    explicit_status: str | None = None
    explicit_value = value
    if isinstance(value, Mapping) and ("value" in value or "status" in value):
        explicit_value = value.get("value")
        raw_status = value.get("status")
        if raw_status is not None:
            explicit_status = _text(raw_status, field=f"{field}.status").upper()
    if status is not None:
        explicit_status = _text(status, field=f"{field}.status").upper()
    if explicit_status is None:
        explicit_status = "NOT_MEASURED" if explicit_value is None else "MEASURED"
    if explicit_status not in FIELD_STATUSES:
        raise ProductionWemmPreannotationError(
            f"{field}.status must be one of {', '.join(FIELD_STATUSES)}"
        )
    if explicit_value is not None and not isinstance(
        explicit_value, (str, int, float, bool, list, dict)
    ):
        raise ProductionWemmPreannotationError(f"{field}.value must be JSON-compatible")
    if isinstance(explicit_value, str):
        explicit_value = unicodedata.normalize("NFKC", explicit_value).strip() or None
    if explicit_status != "MEASURED":
        # A non-measured field must not carry a value that looks authoritative.
        explicit_value = None
    return {"value": _copy_json(explicit_value, field=f"{field}.value"), "status": explicit_status}


def _label_text(row: Mapping[str, Any]) -> str | None:
    for key in ("label_text", "provisional_label", "text", "label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return unicodedata.normalize("NFKC", value).strip()
    return None


def _structured_labels(row: Mapping[str, Any], *, field: str) -> dict[str, dict[str, Any]]:
    labels = row.get("structured_labels")
    source: Mapping[str, Any] = labels if isinstance(labels, Mapping) else row
    return {
        key: _field(source.get(key), field=f"{field}.{key}")
        for key in ("verb", "noun", "attributes", "location", "hand")
    }


def _interval(row: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    start_value = row.get("start_seconds", row.get("start_time_sec"))
    end_value = row.get("end_seconds", row.get("end_time_sec"))
    if start_value is None and end_value is None:
        raw_status = row.get("boundary_status", "NOT_MEASURED")
        status = _text(raw_status, field=f"{field}.boundary_status").upper()
        if status not in BOUNDARY_STATUSES:
            raise ProductionWemmPreannotationError(
                f"{field}.boundary_status must be one of {', '.join(BOUNDARY_STATUSES)}"
            )
        return {"start_seconds": None, "end_seconds": None, "status": status}
    if start_value is None or end_value is None:
        raise ProductionWemmPreannotationError(
            f"{field} must provide both start_seconds and end_seconds or neither"
        )
    start = _finite(start_value, field=f"{field}.start_seconds")
    end = _finite(end_value, field=f"{field}.end_seconds")
    if start < 0 or end <= start:
        raise ProductionWemmPreannotationError(f"{field} interval must satisfy 0 <= start < end")
    raw_status = row.get("boundary_status", "MEASURED")
    status = _text(raw_status, field=f"{field}.boundary_status").upper()
    if status not in BOUNDARY_STATUSES:
        raise ProductionWemmPreannotationError(
            f"{field}.boundary_status must be one of {', '.join(BOUNDARY_STATUSES)}"
        )
    return {"start_seconds": start, "end_seconds": end, "status": status}


def _evidence(row: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    raw = row.get("evidence", [])
    if isinstance(raw, str):
        raw = [raw]
    values = _sequence(raw, field=field)
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            text = unicodedata.normalize("NFKC", value).strip()
            if text:
                result.append({"text": text, "camera_id": row.get("camera_id")})
            continue
        item = _mapping(value, field=f"{field}[{index}]")
        camera = item.get("camera_id")
        if camera is not None:
            camera = _text(camera, field=f"{field}[{index}].camera_id")
        text = item.get("text", item.get("description"))
        if text is not None:
            text = _text(text, field=f"{field}[{index}].text")
        copied = _copy_json(item, field=f"{field}[{index}]")
        if not isinstance(copied, dict):  # pragma: no cover - _copy_json invariant
            raise ProductionWemmPreannotationError(f"{field}[{index}] must be an object")
        if camera is not None:
            copied["camera_id"] = camera
        if text is not None:
            copied["text"] = text
        result.append(copied)
    return result


def _score(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field=field)


def _candidate(row: Mapping[str, Any], *, field: str, position: int) -> dict[str, Any]:
    label = _label_text(row)
    labels = _structured_labels(row, field=f"{field}.structured_labels")
    rank_raw = row.get("rank", position)
    if isinstance(rank_raw, bool) or not isinstance(rank_raw, int) or rank_raw <= 0:
        raise ProductionWemmPreannotationError(f"{field}.rank must be a positive integer")
    score = _score(row.get("score", row.get("visual_score")), field=f"{field}.score")
    copied = _copy_json(row, field=field)
    if not isinstance(copied, dict):  # pragma: no cover - _copy_json invariant
        raise ProductionWemmPreannotationError(f"{field} must be an object")
    return {
        "rank": rank_raw,
        "rank_inferred": "rank" not in row,
        "label_text": label,
        "structured_labels": labels,
        "score": score,
        "camera_id": row.get("camera_id"),
        "evidence": _evidence(row, field=f"{field}.evidence"),
        "raw": copied,
    }


def _margin(candidates: Sequence[Mapping[str, Any]]) -> float | None:
    scores = [item.get("score") for item in candidates[:2]]
    if len(scores) < 2 or not all(isinstance(score, (int, float)) for score in scores):
        return None
    first, second = cast(float, scores[0]), cast(float, scores[1])
    return first - second


def _proposal(row: Mapping[str, Any], *, field: str, index: int, window_id: str) -> dict[str, Any]:
    proposal_id = row.get("proposal_id", row.get("id", f"{window_id}-proposal-{index + 1:02d}"))
    proposal_id = _text(proposal_id, field=f"{field}.proposal_id")
    candidates_raw = row.get("top_k", row.get("candidates", []))
    candidates = _sequence(candidates_raw, field=f"{field}.top_k")
    top_k = [
        _candidate(
            _mapping(value, field=f"{field}.top_k[{position}]"),
            field=f"{field}.top_k[{position}]",
            position=position + 1,
        )
        for position, value in enumerate(candidates)
    ]
    interval = _interval(row, field=f"{field}.proposal_interval")
    labels = _structured_labels(row, field=f"{field}.structured_labels")
    label = _label_text(row)
    if label is None:
        verb = labels["verb"]["value"]
        noun = labels["noun"]["value"]
        if isinstance(verb, str) and isinstance(noun, str):
            label = f"{verb} {noun}"
    confidence = _score(row.get("confidence"), field=f"{field}.confidence")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ProductionWemmPreannotationError(f"{field}.confidence must be between 0 and 1")
    decision = _text(row.get("decision", "pending"), field=f"{field}.decision").casefold()
    if decision not in DECISIONS:
        raise ProductionWemmPreannotationError(
            f"{field}.decision must be one of {', '.join(DECISIONS)}"
        )
    proposal_status = _text(
        row.get("proposal_status", "UNKNOWN" if row.get("unknown") is True else "PROPOSED"),
        field=f"{field}.proposal_status",
    ).upper()
    if proposal_status not in PROPOSAL_STATUSES:
        raise ProductionWemmPreannotationError(
            f"{field}.proposal_status must be one of {', '.join(PROPOSAL_STATUSES)}"
        )
    camera_support = row.get("camera_support", row.get("camera_ids", []))
    if isinstance(camera_support, Mapping):
        camera_support_value: Any = _copy_json(camera_support, field=f"{field}.camera_support")
    else:
        camera_support_value = [
            _text(value, field=f"{field}.camera_support[{position}]")
            for position, value in enumerate(
                _sequence(camera_support, field=f"{field}.camera_support")
            )
        ]
    return {
        "proposal_id": proposal_id,
        "window_id": window_id,
        "proposal_status": proposal_status,
        "label_text": label,
        "structured_labels": labels,
        "proposal_interval": interval,
        "confidence": confidence,
        "evidence": _evidence(row, field=f"{field}.evidence"),
        "camera_support": camera_support_value,
        "top_k": top_k,
        "margin": _margin(top_k),
        "decision": decision,
        "split_hint": bool(row.get("split_hint", False)),
        "review_required": True,
        "automatic_eligible": False,
        "raw": _copy_json(row, field=field),
    }


def build_preannotation_envelope(
    source: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    *,
    raw_model_output: object | None = None,
    model: Mapping[str, Any] | None = None,
    candidate_profile: str = "temporary_phrase_candidates",
    model_invoked: bool = False,
) -> dict[str, Any]:
    """Normalize raw WeMM observations into a review-only envelope.

    ``windows`` are processing/review units.  Their boundaries are copied as
    source context only; a proposal receives an interval only when the model
    explicitly supplied one.  Callers may pass an empty candidate list to
    represent an ``unknown``/``abstain`` window.
    """

    _assert_no_epic_or_gold(source, field="source")
    _assert_no_epic_or_gold(windows, field="windows")
    if raw_model_output is not None:
        _assert_no_epic_or_gold(raw_model_output, field="raw_model_output")
    if model is not None:
        _assert_no_epic_or_gold(model, field="model")
    if not isinstance(model_invoked, bool):
        raise ProductionWemmPreannotationError("model_invoked must be boolean")
    source_copy = _copy_json(source, field="source")
    if not isinstance(source_copy, dict):  # pragma: no cover
        raise ProductionWemmPreannotationError("source must be an object")
    result_windows: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    for window_index, raw_window in enumerate(windows):
        window = _mapping(raw_window, field=f"windows[{window_index}]")
        window_id = _text(window.get("window_id"), field=f"windows[{window_index}].window_id")
        if window_id in seen_windows:
            raise ProductionWemmPreannotationError(f"duplicate window_id: {window_id}")
        seen_windows.add(window_id)
        # Window times are source context; retaining them does not turn them
        # into action boundaries.
        window_start = _optional_finite(
            window.get("start_seconds"), field=f"{window_id}.start_seconds"
        )
        window_end = _optional_finite(window.get("end_seconds"), field=f"{window_id}.end_seconds")
        if (window_start is None) != (window_end is None) or (
            window_start is not None
            and window_end is not None
            and (window_start < 0 or window_end <= window_start)
        ):
            raise ProductionWemmPreannotationError(
                f"invalid source window interval for {window_id}"
            )
        raw_rows = window.get("proposals", window.get("predictions", window.get("candidates", [])))
        rows = _sequence(raw_rows, field=f"{window_id}.proposals")
        proposals = [
            _proposal(
                _mapping(value, field=f"{window_id}.proposals[{index}]"),
                field=f"{window_id}.proposals[{index}]",
                index=index,
                window_id=window_id,
            )
            for index, value in enumerate(rows)
        ]
        window_decision = _text(
            window.get("decision", "pending"), field=f"{window_id}.decision"
        ).casefold()
        if window_decision not in DECISIONS:
            raise ProductionWemmPreannotationError(
                f"{window_id}.decision must be one of {', '.join(DECISIONS)}"
            )
        explicit_window_status = window.get("window_status")
        if explicit_window_status is None:
            if window_decision == "abstain":
                window_status = "ABSTAIN"
            elif not proposals:
                window_status = "UNKNOWN"
            else:
                window_status = "PROPOSALS_AVAILABLE"
        else:
            window_status = _text(
                explicit_window_status, field=f"{window_id}.window_status"
            ).upper()
            if window_status not in {"PROPOSALS_AVAILABLE", "UNKNOWN", "ABSTAIN", "SPLIT"}:
                raise ProductionWemmPreannotationError(f"invalid window_status for {window_id}")
        result_windows.append(
            {
                "window_id": window_id,
                "ordinal": window.get("ordinal", window_index),
                "source_interval": {
                    "start_seconds": window_start,
                    "end_seconds": window_end,
                    "status": "WINDOW_CONTEXT_ONLY" if window_start is not None else "NOT_MEASURED",
                },
                "camera_ids": _copy_json(
                    window.get("camera_ids", []), field=f"{window_id}.camera_ids"
                ),
                "proposals": proposals,
                "window_decision": window_decision,
                "window_status": window_status,
                "raw_candidates": _copy_json(raw_rows, field=f"{window_id}.raw_candidates"),
                "review_required": True,
            }
        )
    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": STATUS,
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "label_space": {
            "kind": LABEL_SPACE,
            "candidate_profile": _text(candidate_profile, field="candidate_profile"),
            "epic_ontology_used": False,
            "mapper_used": False,
            "unknown_allowed": True,
            "abstain_allowed": True,
            "split_allowed": True,
            "reviewed_vocabulary_required": True,
        },
        "source": source_copy,
        "model": _copy_json(
            model or {"name": "WeMM-Embedding", "route": "video_embedding"}, field="model"
        ),
        "windows": result_windows,
        "raw_model_output": _copy_json(raw_model_output, field="raw_model_output"),
        "controls": {
            "model_invoked": model_invoked,
            "gold_read": False,
            "gold_written": False,
            "raw_candidates_overwritten": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_sha_used": False,
        },
        "review_contract": {
            "decision_options": list(DECISIONS[1:]),
            "required_fields": [
                "start_seconds",
                "end_seconds",
                "verb",
                "noun",
                "attributes",
                "location",
                "hand",
                "confidence",
                "evidence",
                "camera_support",
                "top_k",
                "margin",
            ],
            "window_context_fields": ["start_seconds", "end_seconds"],
            "window_context_only": True,
            "status_fields": [
                "window_status",
                "window_decision",
                "proposal_status",
                "decision",
                "split_hint",
            ],
            "provenance_fields": [
                "camera_ids",
                "camera_support",
                "top_k",
                "margin",
            ],
            "accepted_as_gold": False,
            "human_review_required": True,
            "temporal_sidecar_fields": list(TEMPORAL_SIDECAR_FIELDS),
            "temporal_segments_review_only": True,
            "temporal_segments_are_action_boundary_proposals": True,
            "temporal_refinement_review_only": True,
            "refined_segments_review_only": True,
            "refined_segments_are_action_boundary_proposals": True,
        },
    }


def build_review_pack(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Expose proposals as a human review queue without copying a gold label."""

    _validate_envelope_shape(envelope)
    # Keep a compact model/catalog provenance snapshot in the review queue.
    # The raw camera observations remain in the pre-annotation sidecar; this
    # queue must not duplicate those potentially large payloads, but reviewers
    # and downstream aggregators still need to know which model and provisional
    # phrase catalog produced the proposals.
    model = envelope.get("model")
    model_snapshot = _copy_json(model, field="model") if model is not None else {}
    raw_model_output = envelope.get("raw_model_output")
    raw_mapping = raw_model_output if isinstance(raw_model_output, Mapping) else {}
    raw_catalog = raw_mapping.get("catalog")
    catalog_snapshot: dict[str, Any] = {}
    if isinstance(raw_catalog, Mapping):
        for key in (
            "format",
            "phrase_count",
            "epic_ontology_used",
            "mapper_used",
            "provisional",
        ):
            if key in raw_catalog:
                catalog_snapshot[key] = _copy_json(
                    raw_catalog[key], field=f"raw_model_output.catalog.{key}"
                )
    # Dense temporal resolution is an additive, review-only sidecar attached
    # by the native WeMM runner after the historical envelope is built.  Keep
    # it out of ``items`` so existing window-oriented consumers remain
    # compatible, while preserving both the canonical object and its compact
    # alias for reviewers and later batch aggregation.
    temporal_resolution_snapshot, temporal_segments_snapshot = _temporal_resolution_snapshots(
        envelope
    )

    # Adaptive temporal resolution is intentionally additive.  Keep the
    # complete request/score/applied sidecars in the review pack so a reviewer
    # can inspect how a refined interval was obtained, while retaining the
    # historical coarse ``temporal_resolution`` fields unchanged.  Validation
    # below is inference-free and only enforces JSON/provenance/review-only
    # invariants.
    temporal_refinement_snapshots = _temporal_refinement_snapshots(envelope)

    items: list[dict[str, Any]] = []
    for window in envelope["windows"]:
        proposals = window.get("proposals", [])
        window_decision = window.get("window_decision", "pending")
        window_status = window.get("window_status")
        if window_status is None:
            if window_decision == "abstain":
                window_status = "ABSTAIN"
            elif not proposals:
                window_status = "UNKNOWN"
            else:
                window_status = "PROPOSALS_AVAILABLE"
        item = {
            "window_id": window["window_id"],
            "ordinal": window.get("ordinal"),
            "source_interval": copy.deepcopy(window.get("source_interval")),
            "camera_ids": copy.deepcopy(window.get("camera_ids", [])),
            "proposals": copy.deepcopy(proposals),
            # Preserve window-level routing outcomes.  In particular, an empty
            # proposal list is not enough to distinguish an explicit
            # ``ABSTAIN`` from an ``UNKNOWN`` window once the queue is built.
            "window_decision": window_decision,
            "window_status": window_status,
            # This is the pre-annotation input row, not a second inference
            # path.  Retaining it keeps unnormalised candidate provenance
            # available to reviewers while preserving the normalized proposal
            # fields above.
            "raw_candidates": copy.deepcopy(window.get("raw_candidates", [])),
            "review_required": window.get("review_required", True),
            "decision_options": list(DECISIONS[1:]),
            "review_status": "PENDING",
        }
        items.append(item)
    review_pack: dict[str, Any] = {
        "format": REVIEW_FORMAT,
        "authority": AUTHORITY,
        "status": "PENDING_REVIEW",
        "production_eligible": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "source": copy.deepcopy(envelope["source"]),
        "label_space": copy.deepcopy(envelope["label_space"]),
        "model": model_snapshot,
        "items": items,
        # Keep the review queue's contract explicit instead of requiring
        # consumers to infer it from one proposal.  These are the same fields
        # carried by each pre-annotation proposal; the window interval remains
        # source context only (it is never an action boundary).
        "review_contract": {
            "decision_options": list(DECISIONS[1:]),
            "required_fields": [
                "start_seconds",
                "end_seconds",
                "verb",
                "noun",
                "attributes",
                "location",
                "hand",
                "confidence",
                "evidence",
                "camera_support",
                "top_k",
                "margin",
            ],
            "window_context_fields": ["start_seconds", "end_seconds"],
            "window_context_only": True,
            "status_fields": [
                "window_status",
                "window_decision",
                "proposal_status",
                "decision",
                "split_hint",
            ],
            "provenance_fields": [
                "camera_ids",
                "camera_support",
                "top_k",
                "margin",
            ],
            "accepted_as_gold": False,
            "human_review_required": True,
            "temporal_sidecar_fields": list(TEMPORAL_SIDECAR_FIELDS),
            "temporal_segments_review_only": True,
            "temporal_segments_are_action_boundary_proposals": True,
            "temporal_refinement_review_only": True,
            "refined_segments_review_only": True,
            "refined_segments_are_action_boundary_proposals": True,
        },
        "model_artifact": {
            "format": envelope["format"],
            "raw_model_output_retained": True,
            "top_k_retained": True,
            "catalog": catalog_snapshot,
        },
        "controls": {
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "human_adjudication": "NOT_PERFORMED",
        },
    }
    if temporal_resolution_snapshot is not None:
        review_pack["temporal_resolution"] = temporal_resolution_snapshot
        # Keep a compact alias for review clients that only need proposed
        # segments.  It is copied from the detached sidecar, never from the
        # caller's mutable envelope.
        review_pack["temporal_segments"] = copy.deepcopy(
            temporal_resolution_snapshot.get("segments", [])
        )
    elif temporal_segments_snapshot is not None:
        # Alias-only envelopes are valid inputs from lightweight review tools;
        # never silently drop their model-boundary evidence.
        review_pack["temporal_segments"] = copy.deepcopy(temporal_segments_snapshot)
    for key, value in temporal_refinement_snapshots.items():
        review_pack[key] = value
    # ``refined_temporal_segments`` is a descriptive alias used by review
    # clients that need to distinguish the adaptive rows from coarse
    # ``temporal_segments``.  It is always detached from the canonical list.
    if "refined_segments" in temporal_refinement_snapshots:
        review_pack["refined_temporal_segments"] = copy.deepcopy(
            temporal_refinement_snapshots["refined_segments"]
        )
    return review_pack


def _validate_envelope_shape(envelope: Mapping[str, Any]) -> None:
    if envelope.get("format") != FORMAT:
        raise ProductionWemmPreannotationError(f"format must be {FORMAT!r}")
    if envelope.get("authority") != AUTHORITY:
        raise ProductionWemmPreannotationError("authority must remain local-only")
    if envelope.get("production_eligible") is not False:
        raise ProductionWemmPreannotationError("production_eligible must be false")
    if envelope.get("official_gold_status") not in {"NOT_ESTABLISHED", "PENDING_HUMAN_REVIEW"}:
        raise ProductionWemmPreannotationError("official_gold_status must remain unestablished")
    label_space = _mapping(envelope.get("label_space"), field="label_space")
    if label_space.get("kind") != LABEL_SPACE or label_space.get("epic_ontology_used") is not False:
        raise ProductionWemmPreannotationError("label_space must be open and non-EPIC")
    _temporal_resolution_snapshots(envelope)
    _validate_temporal_refinement_fields(envelope)
    windows = _sequence(envelope.get("windows"), field="windows")
    for index, raw_window in enumerate(windows):
        window = _mapping(raw_window, field=f"windows[{index}]")
        _text(window.get("window_id"), field=f"windows[{index}].window_id")
        for proposal_index, raw_proposal in enumerate(
            _sequence(window.get("proposals", []), field=f"windows[{index}].proposals")
        ):
            proposal = _mapping(raw_proposal, field=f"windows[{index}].proposals[{proposal_index}]")
            if (
                proposal.get("review_required") is not True
                or proposal.get("automatic_eligible") is not False
            ):
                raise ProductionWemmPreannotationError("every proposal must require human review")
            decision = proposal.get("decision")
            if decision not in DECISIONS:
                raise ProductionWemmPreannotationError("proposal decision is invalid")
            if proposal.get("proposal_status") not in PROPOSAL_STATUSES:
                raise ProductionWemmPreannotationError("proposal status is invalid")
            _assert_no_epic_or_gold(proposal, field=f"windows[{index}].proposals[{proposal_index}]")


def _validate_temporal_resolution(value: object) -> None:
    """Validate the optional model-driven interval sidecar invariants."""

    if value is None:
        return
    temporal = _mapping(value, field="temporal_resolution")
    if temporal.get("format") != TEMPORAL_RESOLUTION_FORMAT:
        raise ProductionWemmPreannotationError(
            f"temporal_resolution.format must be {TEMPORAL_RESOLUTION_FORMAT!r}"
        )
    if temporal.get("authority") != AUTHORITY:
        raise ProductionWemmPreannotationError(
            "temporal_resolution.authority must remain local-only"
        )
    if temporal.get("status") != "PROPOSALS_ONLY":
        raise ProductionWemmPreannotationError(
            "temporal_resolution.status must be 'PROPOSALS_ONLY'"
        )
    if temporal.get("production_eligible") is not False:
        raise ProductionWemmPreannotationError(
            "temporal_resolution.production_eligible must be false"
        )
    context = _mapping(
        temporal.get("context_interval"), field="temporal_resolution.context_interval"
    )
    if "context_only" in context and context.get("context_only") is not True:
        raise ProductionWemmPreannotationError(
            "temporal_resolution.context_interval.context_only must be true"
        )
    if "is_action_boundary" in context and context.get("is_action_boundary") is not False:
        raise ProductionWemmPreannotationError(
            "temporal_resolution.context_interval.is_action_boundary must be false"
        )
    if "action_boundary" in context and context.get("action_boundary") is not False:
        raise ProductionWemmPreannotationError(
            "temporal_resolution.context_interval.action_boundary must be false"
        )
    _validate_temporal_segments(
        temporal.get("segments", []),
        field="temporal_resolution.segments",
    )


def _validate_context_markers(segment: Mapping[str, Any], *, field: str) -> None:
    """Validate temporal markers, accepting their known legacy omission."""

    for key in ("context_only", "window_context_only"):
        if key in segment and segment.get(key) is not True:
            raise ProductionWemmPreannotationError(f"{field}.{key} must be true")
    for key in ("is_action_boundary", "action_boundary"):
        if key in segment and segment.get(key) is not False:
            raise ProductionWemmPreannotationError(f"{field}.{key} must be false")


def _validate_boundary_provenance(
    segment: Mapping[str, Any],
    *,
    field: str,
    source: str,
    methods: frozenset[str] | None = None,
) -> None:
    boundary_source = segment.get("boundary_source")
    if boundary_source != source:
        raise ProductionWemmPreannotationError(f"{field}.boundary_source must be {source!r}")
    boundary_method_value = segment.get("boundary_method")
    boundary_method = _text(boundary_method_value, field=f"{field}.boundary_method")
    if methods is not None and boundary_method not in methods:
        raise ProductionWemmPreannotationError(
            f"{field}.boundary_method must be one of {', '.join(sorted(methods))}"
        )


def _validate_temporal_segments(value: object, *, field: str) -> list[dict[str, Any]]:
    """Validate and detach coarse model-bound rows for either alias."""

    segments = _sequence(value, field=field)
    copied_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_segment in enumerate(segments):
        segment_field = f"{field}[{index}]"
        segment = _mapping(raw_segment, field=segment_field)
        segment_id = _text(segment.get("segment_id"), field=f"{segment_field}.segment_id")
        if segment_id in seen_ids:
            raise ProductionWemmPreannotationError(f"{segment_field}.segment_id must be unique")
        seen_ids.add(segment_id)
        if segment.get("review_required") is not True:
            raise ProductionWemmPreannotationError(f"{segment_field} must require human review")
        if segment.get("automatic_eligible") is not False:
            raise ProductionWemmPreannotationError(f"{segment_field} cannot be automatic eligible")
        if segment.get("boundary_status") != TEMPORAL_BOUNDARY_STATUS:
            raise ProductionWemmPreannotationError(
                f"{segment_field} must use {TEMPORAL_BOUNDARY_STATUS}"
            )
        _validate_context_markers(segment, field=segment_field)
        _validate_boundary_provenance(
            segment,
            field=segment_field,
            source=TEMPORAL_BOUNDARY_SOURCE,
            methods=TEMPORAL_BOUNDARY_METHODS,
        )
        start = _finite(
            segment.get("start_seconds"),
            field=f"{segment_field}.start_seconds",
        )
        end = _finite(
            segment.get("end_seconds"),
            field=f"{segment_field}.end_seconds",
        )
        if start < 0 or end <= start:
            raise ProductionWemmPreannotationError(
                f"{segment_field} interval must satisfy 0 <= start < end"
            )
        _assert_no_epic_or_gold(segment, field=segment_field)
        copied = _copy_json(segment, field=segment_field)
        if not isinstance(copied, dict):  # pragma: no cover - helper invariant
            raise ProductionWemmPreannotationError(f"{segment_field} must be an object")
        copied.setdefault("context_only", True)
        copied.setdefault("window_context_only", True)
        copied.setdefault("is_action_boundary", False)
        copied.setdefault("action_boundary", False)
        copied_rows.append(copied)
    return copied_rows


def _temporal_resolution_snapshots(
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Validate canonical/alias coarse temporal sidecars and require agreement."""

    resolution_snapshot: dict[str, Any] | None = None
    resolution_segments: list[dict[str, Any]] | None = None
    raw_resolution = envelope.get("temporal_resolution")
    if raw_resolution is not None:
        temporal = _mapping(raw_resolution, field="temporal_resolution")
        _validate_temporal_resolution(temporal)
        copied = _copy_json(temporal, field="temporal_resolution")
        if not isinstance(copied, dict):  # pragma: no cover - helper invariant
            raise ProductionWemmPreannotationError("temporal_resolution must be an object")
        resolution_segments = _validate_temporal_segments(
            copied.get("segments", []),
            field="temporal_resolution.segments",
        )
        copied["segments"] = resolution_segments
        context_copy = copied.get("context_interval")
        if isinstance(context_copy, dict):
            context_copy.setdefault("context_only", True)
            context_copy.setdefault("is_action_boundary", False)
            context_copy.setdefault("action_boundary", False)
        resolution_snapshot = copied

    alias_segments: list[dict[str, Any]] | None = None
    raw_alias = envelope.get("temporal_segments")
    if raw_alias is not None:
        alias_segments = _validate_temporal_segments(
            raw_alias,
            field="temporal_segments",
        )
    if (
        resolution_segments is not None
        and alias_segments is not None
        and resolution_segments != alias_segments
    ):
        raise ProductionWemmPreannotationError(
            "temporal_segments does not match temporal_resolution.segments"
        )
    return resolution_snapshot, alias_segments


def _validate_refined_segment(value: object, *, field: str) -> dict[str, Any]:
    """Validate one adaptive refined row without making it production data."""

    segment = _mapping(value, field=field)
    _assert_no_epic_or_gold(segment, field=field)
    _text(segment.get("segment_id"), field=f"{field}.segment_id")
    _validate_context_markers(segment, field=field)
    _validate_boundary_provenance(
        segment,
        field=field,
        source=TEMPORAL_REFINEMENT_BOUNDARY_SOURCE,
        methods=frozenset({TEMPORAL_REFINEMENT_BOUNDARY_METHOD}),
    )
    if segment.get("review_required") is not True:
        raise ProductionWemmPreannotationError(f"{field}.review_required must be true")
    if segment.get("automatic_eligible") is not False:
        raise ProductionWemmPreannotationError(f"{field}.automatic_eligible must be false")
    boundary_status = _text(
        segment.get("boundary_status"), field=f"{field}.boundary_status"
    ).upper()
    if boundary_status not in REFINED_BOUNDARY_STATUSES:
        raise ProductionWemmPreannotationError(
            f"{field}.boundary_status must be one of {', '.join(REFINED_BOUNDARY_STATUSES)}"
        )

    # Every refined row must retain the coarse model interval as a separate
    # coordinate frame.  This prevents an adaptive row from becoming an
    # unexplained replacement for the historical dense proposal.
    coarse = _mapping(segment.get("coarse_interval"), field=f"{field}.coarse_interval")
    coarse_start = _finite(
        coarse.get("start_seconds"), field=f"{field}.coarse_interval.start_seconds"
    )
    coarse_end = _finite(coarse.get("end_seconds"), field=f"{field}.coarse_interval.end_seconds")
    if coarse_start < 0.0 or coarse_end <= coarse_start:
        raise ProductionWemmPreannotationError(
            f"{field}.coarse_interval must satisfy 0 <= start < end"
        )

    start_value = segment.get("start_seconds")
    end_value = segment.get("end_seconds")
    if boundary_status == "MODEL_REFINED":
        if start_value is None or end_value is None:
            raise ProductionWemmPreannotationError(
                f"{field} MODEL_REFINED rows require start_seconds and end_seconds"
            )
        start = _finite(start_value, field=f"{field}.start_seconds")
        end = _finite(end_value, field=f"{field}.end_seconds")
        if start < 0.0 or end <= start:
            raise ProductionWemmPreannotationError(
                f"{field} interval must satisfy 0 <= start < end"
            )
    elif start_value is not None or end_value is not None:
        # Pending rows are deliberately unresolved.  A half-populated row is
        # ambiguous and must not be interpreted as a measured interval.
        raise ProductionWemmPreannotationError(
            f"{field} MODEL_REFINEMENT_PENDING rows must omit start/end"
        )

    copied = _copy_json(segment, field=field)
    if not isinstance(copied, dict):  # pragma: no cover - helper invariant
        raise ProductionWemmPreannotationError(f"{field} must be an object")
    copied.setdefault("context_only", True)
    copied.setdefault("window_context_only", True)
    copied.setdefault("is_action_boundary", False)
    copied.setdefault("action_boundary", False)
    # Canonicalize the status in the detached snapshot so preannotation and
    # aggregate readers cannot disagree merely because an upstream review tool
    # used lowercase spelling.
    copied["boundary_status"] = boundary_status
    return copied


def _validate_temporal_refinement_sidecar(value: object, *, field: str) -> dict[str, Any]:
    """Validate a plan/result/applied adaptive sidecar generically.

    The planner and score resolver own their detailed schemas.  At the
    pre-annotation boundary we intentionally validate only the fields needed
    to keep the sidecar review-only and JSON-safe; this lets future additive
    diagnostics pass through without a second schema fork.
    """

    sidecar = _mapping(value, field=field)
    expected_formats = {
        "temporal_refinement_plan": TEMPORAL_REFINEMENT_PLAN_FORMAT,
        "temporal_refinement_fine_plan": TEMPORAL_SCORE_REFINEMENT_FORMAT,
        "temporal_refinement_score_resolution": TEMPORAL_SCORE_RESULT_FORMAT,
        "temporal_refinement": TEMPORAL_REFINEMENT_APPLIED_FORMAT,
    }
    expected_format = expected_formats.get(field)
    if expected_format is None:
        raise ProductionWemmPreannotationError(f"{field} is not a supported temporal sidecar")
    if sidecar.get("format") != expected_format:
        raise ProductionWemmPreannotationError(f"{field}.format must be {expected_format!r}")
    if sidecar.get("authority") != AUTHORITY:
        raise ProductionWemmPreannotationError(f"{field}.authority must remain local-only")
    _assert_no_epic_or_gold(sidecar, field=field)
    copied = _copy_json(sidecar, field=field)
    if not isinstance(copied, dict):  # pragma: no cover - helper invariant
        raise ProductionWemmPreannotationError(f"{field} must be an object")
    if copied.get("production_eligible") is not False:
        raise ProductionWemmPreannotationError(f"{field}.production_eligible must be false")
    return copied


def _temporal_refinement_snapshots(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached adaptive fields present on an envelope."""

    snapshots: dict[str, Any] = {}

    def _validated_rows(value: object, *, field: str) -> list[dict[str, Any]]:
        rows = _sequence(value, field=field)
        seen_ids: set[str] = set()
        result: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            row_field = f"{field}[{index}]"
            checked = _validate_refined_segment(row, field=row_field)
            row_id = _text(checked.get("segment_id"), field=f"{row_field}.segment_id")
            if row_id in seen_ids:
                raise ProductionWemmPreannotationError(f"{row_field}.segment_id must be unique")
            seen_ids.add(row_id)
            result.append(checked)
        return result

    for key in TEMPORAL_REFINEMENT_FIELDS:
        if key not in envelope:
            continue
        value = envelope[key]
        if key in {
            "refined_segments",
            "refined_temporal_segments",
            "temporal_refinement_segments",
        }:
            snapshots[key] = _validated_rows(value, field=key)
        else:
            snapshots[key] = _validate_temporal_refinement_sidecar(value, field=key)

    # The applied sidecar commonly embeds the same refined rows as the
    # top-level field.  Require equality when both are supplied so stale
    # copies cannot silently diverge in a review pack.
    applied = snapshots.get("temporal_refinement")
    nested_rows_by_key: dict[str, list[dict[str, Any]]] = {}
    if isinstance(applied, Mapping):
        for nested_key in (
            "refined_segments",
            "refined_temporal_segments",
            "temporal_refinement_segments",
            "segments",
        ):
            nested_value = applied.get(nested_key)
            if nested_value is None:
                continue
            nested_rows_by_key[nested_key] = _validated_rows(
                nested_value, field=f"temporal_refinement.{nested_key}"
            )
            if isinstance(applied, dict):
                # Keep every nested alias detached; do not collapse one alias
                # into another before comparing their source snapshots.
                applied[nested_key] = copy.deepcopy(nested_rows_by_key[nested_key])
    nested_rows = nested_rows_by_key.get("refined_segments")
    top_rows = snapshots.get("refined_segments")
    alias_rows = snapshots.get("refined_temporal_segments")
    named_rows = snapshots.get("temporal_refinement_segments")
    if top_rows is not None and alias_rows is not None and top_rows != alias_rows:
        raise ProductionWemmPreannotationError(
            "refined_temporal_segments does not match refined_segments"
        )
    if top_rows is not None and named_rows is not None and top_rows != named_rows:
        raise ProductionWemmPreannotationError(
            "temporal_refinement_segments does not match refined_segments"
        )
    if alias_rows is not None and named_rows is not None and alias_rows != named_rows:
        raise ProductionWemmPreannotationError(
            "temporal_refinement_segments does not match refined_temporal_segments"
        )
    canonical_rows = (
        top_rows if top_rows is not None else alias_rows if alias_rows is not None else named_rows
    )
    if canonical_rows is not None:
        for nested_key, nested_rows_value in nested_rows_by_key.items():
            if nested_rows_value != canonical_rows:
                raise ProductionWemmPreannotationError(
                    f"temporal_refinement.{nested_key} does not match refined_segments"
                )
    # Accept a sidecar-only envelope and expose the canonical top-level row
    # list for review consumers.  The runner normally supplies this field
    # explicitly, but promotion here keeps rebuild/interop paths lossless.
    if top_rows is None:
        if nested_rows is not None:
            snapshots["refined_segments"] = nested_rows
        elif alias_rows is not None:
            snapshots["refined_segments"] = alias_rows
        elif named_rows is not None:
            snapshots["refined_segments"] = named_rows
        elif nested_rows_by_key:
            snapshots["refined_segments"] = next(iter(nested_rows_by_key.values()))
    return snapshots


def _validate_temporal_refinement_fields(envelope: Mapping[str, Any]) -> None:
    """Validate adaptive sidecars attached to a pre-annotation envelope."""

    # Calling the same extractor used by ``build_review_pack`` keeps the two
    # public entry points in lockstep and performs all detached JSON checks.
    _temporal_refinement_snapshots(envelope)


def validate_preannotation_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached copy suitable for a sidecar writer."""

    _validate_envelope_shape(envelope)
    _assert_no_epic_or_gold(envelope, field="envelope")
    copied = _copy_json(envelope, field="envelope")
    if not isinstance(copied, dict):  # pragma: no cover - mapping invariant
        raise ProductionWemmPreannotationError("envelope must be an object")
    return copied


def load_json(value: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmPreannotationError(f"could not read JSON {value}: {exc}") from exc
    return dict(_mapping(payload, field="input"))


__all__ = [
    "AUTHORITY",
    "BOUNDARY_STATUSES",
    "DECISIONS",
    "FIELD_STATUSES",
    "FORMAT",
    "LABEL_SPACE",
    "PROPOSAL_STATUSES",
    "REFINED_BOUNDARY_STATUSES",
    "REVIEW_FORMAT",
    "STATUS",
    "TEMPORAL_REFINEMENT_FIELDS",
    "TEMPORAL_SIDECAR_FIELDS",
    "ProductionWemmPreannotationError",
    "build_preannotation_envelope",
    "build_review_pack",
    "load_json",
    "validate_preannotation_envelope",
]
