"""Build a read-only Qwen-review queue from production WeMM artifacts.

The production WeMM runner emits review-only, open-phrase observations.  This
module turns those *already recorded* observations into a smaller queue for a
later Qwen ambiguity pass.  It deliberately does not decide an action, infer an
action span, or estimate model quality.  In particular, an eight-second
processing window remains source context, not a proposed action segment.

The selector accepts one of the following WeMM-only shapes:

* a ``robata-production-wemm-preannotation-v1`` sidecar;
* a per-recording ``robata-production-wemm-preannotation-review-pack-v1``;
* a flattened ``robata-production-wemm-review-pack-aggregate-v1`` queue.

It copies compact Top-K and provenance references into the output, leaving the
original sidecars untouched.  No media is decoded; no model is loaded; and
gold, Qwen, Mage, EPIC, Mapper, or evaluator artifacts are neither joined nor
consulted.  Thresholds are routing knobs only, never calibrated probabilities
or quality gates.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PREANNOTATION_FORMAT: Final = "robata-production-wemm-preannotation-v1"
REVIEW_PACK_FORMAT: Final = "robata-production-wemm-preannotation-review-pack-v1"
REVIEW_AGGREGATE_FORMAT: Final = "robata-production-wemm-review-pack-aggregate-v1"
AMBIGUITY_SELECTION_FORMAT: Final = "robata-production-wemm-ambiguity-selection-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_QUALITY_STATUS: Final = "NOT_MEASURED"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
FIELD_NAMES: Final = ("verb", "noun", "attributes", "location", "hand")
CRITICAL_FIELD_NAMES: Final = ("verb", "noun")
OPTIONAL_FIELD_NAMES: Final = ("attributes", "location", "hand")
BOUNDARY_STATUSES: Final = frozenset(
    {"MEASURED", "WINDOW_BOUND_ONLY", "NOT_MEASURED", "NOT_OBSERVABLE"}
)

# Stable order makes reports and test fixtures reproducible.  Unrecognised
# future reason codes are still retained, after this known order.
_REASON_ORDER: Final = (
    "NO_PROPOSALS_AVAILABLE",
    "MULTIPLE_PROPOSALS_AVAILABLE",
    "TOP_K_EMPTY",
    "TOP_K_INSUFFICIENT",
    "MARGIN_UNAVAILABLE",
    "LOW_TOP1_TOP2_MARGIN",
    "TOP_K_NEAR_TIE_VERB_CONFLICT",
    "TOP_K_NEAR_TIE_NOUN_CONFLICT",
    "TOP_K_NEAR_TIE_LABEL_CONFLICT",
    "CAMERA_EVIDENCE_UNAVAILABLE",
    "CROSS_CAMERA_TOP1_DISAGREEMENT",
    "LOW_CAMERA_TOP1_CONSENSUS",
    "MISSING_VERB_FIELD",
    "MISSING_NOUN_FIELD",
    "MISSING_CONFIDENCE",
    "MISSING_EVIDENCE",
    "MISSING_CAMERA_SUPPORT",
    "OPTIONAL_FIELDS_UNMEASURED",
    "PROPOSAL_BOUNDARY_UNMEASURED",
    "SOURCE_INTERVAL_UNAVAILABLE",
    "SOURCE_INTERVAL_CONTEXT_ONLY",
    "RECORDING_LEADING_CONTEXT_EDGE",
    "RECORDING_TRAILING_CONTEXT_EDGE",
    "ADJACENT_TOP1_TRANSITION",
    "NONCONTIGUOUS_WINDOW_CONTEXT",
)
_REASON_INDEX: Final = {reason: index for index, reason in enumerate(_REASON_ORDER)}


class ProductionWemmAmbiguitySelectorError(ValueError):
    """Raised when selector input or routing policy is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmAmbiguitySelectorError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmAmbiguitySelectorError(f"{field} must be an array")
    return value


def _text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _copy_json(value: object, *, field: str = "value") -> Any:
    """Detach JSON-compatible output without adding an identity/digest."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmAmbiguitySelectorError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmAmbiguitySelectorError(f"{field} keys must be strings")
            result[key] = _copy_json(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_json(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmAmbiguitySelectorError(f"{field} must be JSON-compatible")


def _load_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(payload, Mapping):
        return None, "JSON root must be an object"
    return dict(payload), None


def load_json(value: str | Path) -> dict[str, Any]:
    """Load one JSON object for callers that want explicit preflight control."""

    path = Path(value).expanduser().resolve()
    payload, error = _load_file(path)
    if payload is None or error is not None:
        raise ProductionWemmAmbiguitySelectorError(
            f"could not read JSON {path}: {error or 'invalid object'}"
        )
    return payload


def _known_format(payload: Mapping[str, Any]) -> bool:
    return payload.get("format") in {
        PREANNOTATION_FORMAT,
        REVIEW_PACK_FORMAT,
        REVIEW_AGGREGATE_FORMAT,
    }


def _discover_input_files(path: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Discover only WeMM preannotation/review locations.

    A recursive scan of arbitrary ``.json`` files would unnecessarily open
    Qwen/Mage diagnostics sitting beside a run.  The restricted discovery
    surface deliberately limits directories to explicit ``review`` and
    ``preannotations`` trees plus root-level aggregate reports.
    """

    rejected: list[dict[str, str]] = []
    if path.is_file():
        return [path.resolve()], rejected
    if not path.exists():
        return [], [{"path": str(path), "reason": "PATH_NOT_FOUND"}]
    if not path.is_dir():
        return [], [{"path": str(path), "reason": "PATH_NOT_FILE_OR_DIRECTORY"}]

    aggregate_candidates: set[Path] = set()
    for child in path.glob("aggregate*.json"):
        if child.is_file():
            aggregate_candidates.add(child.resolve())
    for child in path.glob("*review*aggregate*.json"):
        if child.is_file():
            aggregate_candidates.add(child.resolve())
    # Prefer an explicitly materialized flattened queue when one is present.
    # Otherwise a directory containing both an aggregate and its source packs
    # would silently duplicate every window.
    if aggregate_candidates:
        return sorted(aggregate_candidates), rejected

    candidates: set[Path] = set()
    for candidate in path.rglob("*.json"):
        if not candidate.is_file():
            continue
        if any(
            parent.name.casefold() in {"review", "preannotations"} for parent in candidate.parents
        ):
            candidates.add(candidate.resolve())
    if not candidates:
        rejected.append({"path": str(path), "reason": "NO_WEMM_REVIEW_OR_PREANNOTATION_FOUND"})
    return sorted(candidates), rejected


def _collect_documents(
    inputs: Mapping[str, Any] | Sequence[Any] | str | Path,
) -> tuple[list[tuple[str, Mapping[str, Any]]], list[dict[str, str]], list[dict[str, str]]]:
    """Load explicit WeMM-only documents, retaining malformed input visibility."""

    documents: list[tuple[str, Mapping[str, Any]]] = []
    invalid: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    def add_mapping(path: str, payload: Mapping[str, Any]) -> None:
        if not _known_format(payload):
            rejected.append({"path": path, "reason": "UNSUPPORTED_OR_NON_WEMM_FORMAT"})
            return
        documents.append((path, payload))

    def add_path(raw_path: str | Path) -> None:
        path = Path(raw_path).expanduser().resolve()
        paths, discovery_rejected = _discover_input_files(path)
        rejected.extend(discovery_rejected)
        for candidate in paths:
            payload, error = _load_file(candidate)
            if payload is None or error is not None:
                invalid.append({"path": str(candidate), "reason": error or "INVALID_JSON"})
                continue
            add_mapping(str(candidate), payload)

    if isinstance(inputs, Mapping):
        add_mapping("<mapping>", inputs)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray, Path)):
        for index, raw in enumerate(inputs):
            if isinstance(raw, Mapping):
                add_mapping(f"<sequence>[{index}]", raw)
            elif isinstance(raw, (str, Path)):
                add_path(raw)
            else:
                rejected.append(
                    {"path": f"<sequence>[{index}]", "reason": "INPUT_NOT_OBJECT_OR_PATH"}
                )
    else:
        add_path(inputs)  # type: ignore[arg-type]
    return documents, invalid, rejected


def _recording_id(source: Mapping[str, Any], *, fallback: str) -> str:
    return (
        _text(source.get("recording_id") or source.get("source_id") or source.get("path"))
        or fallback
    )


def _source_ref(
    source: Mapping[str, Any],
    *,
    recording_id: str,
    input_path: str,
    source_kind: str,
) -> dict[str, Any]:
    """Preserve a compact stable reference plus the source snapshot."""

    path_key = "preannotation_path" if source_kind == "preannotation" else "review_pack_path"
    result: dict[str, Any] = {
        "recording_id": recording_id,
        path_key: input_path if not input_path.startswith("<") else None,
        "source_path": source.get("path"),
        "archive_member": source.get("archive_member"),
        "archive_path": source.get("archive_path"),
        "source_preflight_status": source.get("source_preflight_status"),
        "qa_status": source.get("qa_status"),
        "source": _copy_json(source, field="source_ref.source"),
    }
    return result


def _source_ref_from_item(
    item: Mapping[str, Any],
    *,
    recording_id: str,
    input_path: str,
    source_snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw = item.get("source_ref")
    result = _copy_json(raw, field="item.source_ref") if isinstance(raw, Mapping) else {}
    if not isinstance(result, dict):  # pragma: no cover - _copy_json invariant
        raise ProductionWemmAmbiguitySelectorError("item.source_ref must be an object")
    result.setdefault("recording_id", recording_id)
    result.setdefault("review_pack_path", input_path if not input_path.startswith("<") else None)
    source = source_snapshots.get(recording_id)
    if source is not None:
        result.setdefault("source", _copy_json(source, field="item.recording_source"))
        result.setdefault("source_path", source.get("path"))
        result.setdefault("archive_member", source.get("archive_member"))
        result.setdefault("archive_path", source.get("archive_path"))
    return result


def _normalise_document(
    path: str,
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Project one known WeMM artifact into window-context rows only."""

    fmt = payload.get("format")
    invalid: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    if fmt == PREANNOTATION_FORMAT:
        source = payload.get("source")
        if not isinstance(source, Mapping):
            return rows, [{"path": path, "reason": "PREANNOTATION_SOURCE_NOT_OBJECT"}]
        recording_id = _recording_id(source, fallback=path)
        windows = payload.get("windows")
        if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
            return rows, [{"path": path, "reason": "PREANNOTATION_WINDOWS_NOT_ARRAY"}]
        reference = _source_ref(
            source,
            recording_id=recording_id,
            input_path=path,
            source_kind="preannotation",
        )
        # The review contract belongs to the source document, rather than to
        # the selector.  Carry a detached snapshot on every normalized row so
        # a later reviewer can interpret status/decision fields without having
        # to reopen the source sidecar.
        document_review_contract = payload.get("review_contract", {})
        if not isinstance(document_review_contract, Mapping):
            document_review_contract = {}
        for index, raw in enumerate(windows):
            if not isinstance(raw, Mapping):
                invalid.append({"path": f"{path}.windows[{index}]", "reason": "WINDOW_NOT_OBJECT"})
                continue
            row_review_contract = raw.get("review_contract", document_review_contract)
            if not isinstance(row_review_contract, Mapping):
                row_review_contract = document_review_contract
            rows.append(
                {
                    "recording_id": recording_id,
                    "window_id": _text(raw.get("window_id")),
                    "ordinal": raw.get("ordinal", index),
                    "source_interval": raw.get("source_interval"),
                    "camera_ids": raw.get("camera_ids", []),
                    "proposals": raw.get("proposals", []),
                    "window_status": raw.get("window_status"),
                    "window_decision": raw.get("window_decision"),
                    "raw_candidates": raw.get("raw_candidates", []),
                    "review_contract": _copy_json(
                        row_review_contract, field="window.review_contract"
                    ),
                    "source_ref": reference,
                    "provenance": {
                        "input_path": path,
                        "input_format": PREANNOTATION_FORMAT,
                        "source_preflight_status": source.get("source_preflight_status"),
                        "qa_status": source.get("qa_status"),
                        "window_context_only": True,
                    },
                    "source_window_count": source.get("window_count"),
                    "model": payload.get("model"),
                }
            )
        return rows, invalid

    if fmt == REVIEW_PACK_FORMAT:
        source = payload.get("source")
        if not isinstance(source, Mapping):
            return rows, [{"path": path, "reason": "REVIEW_PACK_SOURCE_NOT_OBJECT"}]
        recording_id = _recording_id(source, fallback=path)
        items = payload.get("items")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            return rows, [{"path": path, "reason": "REVIEW_PACK_ITEMS_NOT_ARRAY"}]
        reference = _source_ref(
            source,
            recording_id=recording_id,
            input_path=path,
            source_kind="review_pack",
        )
        document_review_contract = payload.get("review_contract", {})
        if not isinstance(document_review_contract, Mapping):
            document_review_contract = {}
        for index, raw in enumerate(items):
            if not isinstance(raw, Mapping):
                invalid.append({"path": f"{path}.items[{index}]", "reason": "ITEM_NOT_OBJECT"})
                continue
            row_review_contract = raw.get("review_contract", document_review_contract)
            if not isinstance(row_review_contract, Mapping):
                row_review_contract = document_review_contract
            rows.append(
                {
                    "recording_id": recording_id,
                    "window_id": _text(raw.get("window_id")),
                    "ordinal": raw.get("ordinal", index),
                    "source_interval": raw.get("source_interval"),
                    "camera_ids": raw.get("camera_ids", []),
                    "proposals": raw.get("proposals", []),
                    "window_status": raw.get("window_status"),
                    "window_decision": raw.get("window_decision"),
                    "raw_candidates": raw.get("raw_candidates", []),
                    "review_contract": _copy_json(
                        row_review_contract, field="window.review_contract"
                    ),
                    "source_ref": reference,
                    "provenance": {
                        "input_path": path,
                        "input_format": REVIEW_PACK_FORMAT,
                        "source_preflight_status": source.get("source_preflight_status"),
                        "qa_status": source.get("qa_status"),
                        "window_context_only": True,
                    },
                    "source_window_count": source.get("window_count"),
                    "model": payload.get("model"),
                }
            )
        return rows, invalid

    if fmt == REVIEW_AGGREGATE_FORMAT:
        raw_recordings = payload.get("recordings", [])
        source_snapshots: dict[str, Mapping[str, Any]] = {}
        if isinstance(raw_recordings, Sequence) and not isinstance(
            raw_recordings, (str, bytes, bytearray)
        ):
            for raw_recording in raw_recordings:
                if not isinstance(raw_recording, Mapping):
                    continue
                recording_id_value = _text(raw_recording.get("recording_id"))
                source_ref = raw_recording.get("source")
                if recording_id_value is not None and isinstance(source_ref, Mapping):
                    recording_id = recording_id_value
                    source = (
                        source_ref.get("source")
                        if isinstance(source_ref.get("source"), Mapping)
                        else source_ref
                    )
                    if isinstance(source, Mapping):
                        source_snapshots[recording_id] = source
        items = payload.get("items")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            return rows, [{"path": path, "reason": "REVIEW_AGGREGATE_ITEMS_NOT_ARRAY"}]
        document_review_contract = payload.get("review_contract", {})
        if not isinstance(document_review_contract, Mapping):
            document_review_contract = {}
        for index, raw in enumerate(items):
            if not isinstance(raw, Mapping):
                invalid.append({"path": f"{path}.items[{index}]", "reason": "ITEM_NOT_OBJECT"})
                continue
            source_ref_raw = raw.get("source_ref")
            source_ref_map = source_ref_raw if isinstance(source_ref_raw, Mapping) else {}
            recording_id_value = _text(
                raw.get("recording_id") or source_ref_map.get("recording_id")
            )
            if recording_id_value is None:
                invalid.append({"path": f"{path}.items[{index}]", "reason": "RECORDING_ID_MISSING"})
                continue
            recording_id = recording_id_value
            source = source_snapshots.get(recording_id)
            source_count = source.get("window_count") if source is not None else None
            provenance_raw = raw.get("provenance")
            provenance = (
                _copy_json(provenance_raw, field="item.provenance")
                if isinstance(provenance_raw, Mapping)
                else {}
            )
            if not isinstance(provenance, dict):  # pragma: no cover - _copy_json invariant
                raise ProductionWemmAmbiguitySelectorError("item.provenance must be an object")
            provenance.update(
                {
                    "input_path": path,
                    "input_format": REVIEW_AGGREGATE_FORMAT,
                    "window_context_only": True,
                }
            )
            row_review_contract = raw.get("review_contract", document_review_contract)
            if not isinstance(row_review_contract, Mapping):
                row_review_contract = document_review_contract
            rows.append(
                {
                    "recording_id": recording_id,
                    "window_id": _text(raw.get("window_id")),
                    "ordinal": raw.get("ordinal", index),
                    "source_interval": raw.get("source_interval"),
                    "camera_ids": raw.get("camera_ids", []),
                    "proposals": raw.get("proposals", []),
                    "window_status": raw.get("window_status"),
                    "window_decision": raw.get("window_decision"),
                    "raw_candidates": raw.get("raw_candidates", []),
                    "review_contract": _copy_json(
                        row_review_contract, field="window.review_contract"
                    ),
                    "source_ref": _source_ref_from_item(
                        raw,
                        recording_id=recording_id,
                        input_path=path,
                        source_snapshots=source_snapshots,
                    ),
                    "provenance": provenance,
                    "source_window_count": source_count,
                    "model": None,
                }
            )
        return rows, invalid

    return rows, [{"path": path, "reason": "UNSUPPORTED_OR_NON_WEMM_FORMAT"}]


def _normalised_structured_field(proposal: Mapping[str, Any], field: str) -> tuple[str, Any]:
    labels = proposal.get("structured_labels")
    if not isinstance(labels, Mapping):
        return "MISSING", None
    raw = labels.get(field)
    if isinstance(raw, Mapping):
        status = _text(raw.get("status"))
        value = raw.get("value")
        if status:
            return status.upper(), value
        return ("MEASURED", value) if value is not None else ("NOT_MEASURED", None)
    return ("MEASURED", raw) if raw is not None else ("NOT_MEASURED", None)


def _candidate_label(candidate: Mapping[str, Any]) -> str | None:
    label = _text(candidate.get("label_text"))
    if label:
        return label
    labels = candidate.get("structured_labels")
    if isinstance(labels, Mapping):
        verb_raw = labels.get("verb")
        noun_raw = labels.get("noun")
        verb = _text(verb_raw.get("value") if isinstance(verb_raw, Mapping) else verb_raw)
        noun = _text(noun_raw.get("value") if isinstance(noun_raw, Mapping) else noun_raw)
        if verb and noun:
            return f"{verb} {noun}"
    return None


def _candidate_field(candidate: Mapping[str, Any], field: str) -> str | None:
    labels = candidate.get("structured_labels")
    if not isinstance(labels, Mapping):
        return None
    raw = labels.get(field)
    return _text(raw.get("value") if isinstance(raw, Mapping) else raw)


def _rank(candidate: Mapping[str, Any], fallback: int) -> int:
    value = candidate.get("rank")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return fallback


def _ordered_candidates(proposal: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = proposal.get("top_k", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    candidates = [row for row in raw if isinstance(row, Mapping)]
    indexed = list(enumerate(candidates, start=1))

    def sort_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, float, str]:
        fallback, candidate = item
        score = _finite(candidate.get("score"))
        score_key = -score if score is not None else math.inf
        return (_rank(candidate, fallback), score_key, _candidate_label(candidate) or "")

    return [candidate for _, candidate in sorted(indexed, key=sort_key)]


def _camera_vote_rows(
    proposal: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, int], dict[str, float | None]]:
    """Recover per-camera rank-1 labels from retained candidate evidence.

    The fused proposal is not assumed to be a camera consensus.  A camera vote
    is only derived where the recorded Top-K candidate evidence exposes a
    camera id and rank; absent evidence stays absent rather than being guessed
    from the fused label.
    """

    best: dict[str, tuple[int, float | None, str]] = {}

    def consider(
        camera: object, label: str | None, rank_value: object, score_value: object
    ) -> None:
        camera_id = _text(camera)
        if camera_id is None or label is None:
            return
        rank = (
            rank_value
            if isinstance(rank_value, int) and not isinstance(rank_value, bool) and rank_value > 0
            else 10**6
        )
        score = _finite(score_value)
        candidate = (rank, score, label)
        previous = best.get(camera_id)
        if previous is None:
            best[camera_id] = candidate
            return
        previous_rank, previous_score, previous_label = previous
        # Lowest recorded per-camera rank wins.  Score only resolves an
        # impossible duplicate-rank tie; label is a deterministic final tie.
        if (rank, -(score if score is not None else -math.inf), label.casefold()) < (
            previous_rank,
            -(previous_score if previous_score is not None else -math.inf),
            previous_label.casefold(),
        ):
            best[camera_id] = candidate

    for index, candidate in enumerate(candidates, start=1):
        fallback_rank = _rank(candidate, index)
        label = _candidate_label(candidate)
        evidence = candidate.get("evidence", [])
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
            for raw_evidence in evidence:
                if not isinstance(raw_evidence, Mapping):
                    continue
                consider(
                    raw_evidence.get("camera_id"),
                    _text(raw_evidence.get("label_text")) or label,
                    raw_evidence.get("rank", fallback_rank),
                    raw_evidence.get("score", candidate.get("score")),
                )
        raw_candidate = candidate.get("raw")
        if isinstance(raw_candidate, Mapping):
            for evidence_key in ("per_camera", "camera_evidence"):
                raw_evidence_rows = raw_candidate.get(evidence_key, [])
                if not isinstance(raw_evidence_rows, Sequence) or isinstance(
                    raw_evidence_rows, (str, bytes, bytearray)
                ):
                    continue
                for raw_evidence in raw_evidence_rows:
                    if not isinstance(raw_evidence, Mapping):
                        continue
                    consider(
                        raw_evidence.get("camera_id"),
                        _text(raw_evidence.get("label_text")) or label,
                        raw_evidence.get("rank", fallback_rank),
                        raw_evidence.get("score", candidate.get("score")),
                    )
        evidence_rows = (
            [row for row in evidence if isinstance(row, Mapping)]
            if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray))
            else []
        )
        if not evidence_rows:
            consider(candidate.get("camera_id"), label, fallback_rank, candidate.get("score"))

    # Older sidecars may carry only proposal-level top-1 evidence.  It is still
    # useful if present, but never manufactures missing non-top1 candidates.
    proposal_label = _text(proposal.get("label_text"))
    proposal_evidence = proposal.get("evidence", [])
    if isinstance(proposal_evidence, Sequence) and not isinstance(
        proposal_evidence, (str, bytes, bytearray)
    ):
        for raw_evidence in proposal_evidence:
            if not isinstance(raw_evidence, Mapping):
                continue
            camera_id = _text(raw_evidence.get("camera_id"))
            # Candidate-level evidence is more specific.  Proposal evidence is
            # only a fallback for cameras absent from the retained Top-K rows;
            # it must not overwrite a conflicting camera rank-1 observation.
            if camera_id is not None and camera_id not in best:
                consider(
                    camera_id,
                    _text(raw_evidence.get("label_text")) or proposal_label,
                    raw_evidence.get("rank", 1),
                    raw_evidence.get("score"),
                )

    labels = {camera_id: value[2] for camera_id, value in sorted(best.items())}
    ranks = {camera_id: value[0] for camera_id, value in sorted(best.items())}
    scores = {camera_id: value[1] for camera_id, value in sorted(best.items())}
    return labels, ranks, scores


def _compact_candidate(candidate: Mapping[str, Any], *, fallback_rank: int) -> dict[str, Any]:
    evidence = candidate.get("evidence", [])
    evidence_rows = (
        [row for row in evidence if isinstance(row, Mapping)]
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray))
        else []
    )
    return {
        "rank": _rank(candidate, fallback_rank),
        "label_text": _candidate_label(candidate),
        "structured_labels": _copy_json(
            candidate.get("structured_labels", {}), field="candidate.structured_labels"
        ),
        "score": _finite(candidate.get("score")),
        "camera_id": _text(candidate.get("camera_id")),
        "evidence_camera_ids": sorted(
            {camera for row in evidence_rows if (camera := _text(row.get("camera_id"))) is not None}
        ),
    }


def _sort_reasons(reasons: Sequence[str]) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda reason: (_REASON_INDEX.get(reason, len(_REASON_ORDER)), reason),
    )


def _proposal_diagnostic(
    proposal: Mapping[str, Any],
    *,
    proposal_index: int,
    expected_camera_count: int,
    margin_threshold: float,
    camera_consensus_threshold: float,
    top_k_conflict_threshold: float,
) -> dict[str, Any]:
    """Return one non-semantic ambiguity diagnostic for a recorded proposal."""

    candidates = _ordered_candidates(proposal)
    top1 = candidates[0] if candidates else None
    top2 = candidates[1] if len(candidates) > 1 else None
    top1_score = _finite(top1.get("score")) if top1 is not None else None
    top2_score = _finite(top2.get("score")) if top2 is not None else None
    recorded_margin = _finite(proposal.get("margin"))
    inferred_margin = (
        top1_score - top2_score if top1_score is not None and top2_score is not None else None
    )
    margin = recorded_margin if recorded_margin is not None else inferred_margin
    reasons: list[str] = []
    selection_reasons: list[str] = []
    if not candidates:
        reasons.append("TOP_K_EMPTY")
        selection_reasons.append("TOP_K_EMPTY")
    elif len(candidates) == 1:
        reasons.append("TOP_K_INSUFFICIENT")
        selection_reasons.append("TOP_K_INSUFFICIENT")
    elif margin is None:
        reasons.append("MARGIN_UNAVAILABLE")
        selection_reasons.append("MARGIN_UNAVAILABLE")
    elif margin <= margin_threshold:
        reasons.append("LOW_TOP1_TOP2_MARGIN")
        selection_reasons.append("LOW_TOP1_TOP2_MARGIN")

    if (
        top1 is not None
        and top2 is not None
        and margin is not None
        and margin <= top_k_conflict_threshold
    ):
        top1_label = _candidate_label(top1)
        top2_label = _candidate_label(top2)
        top1_verb, top2_verb = _candidate_field(top1, "verb"), _candidate_field(top2, "verb")
        top1_noun, top2_noun = _candidate_field(top1, "noun"), _candidate_field(top2, "noun")
        if top1_verb and top2_verb and top1_verb.casefold() != top2_verb.casefold():
            reasons.append("TOP_K_NEAR_TIE_VERB_CONFLICT")
            selection_reasons.append("TOP_K_NEAR_TIE_VERB_CONFLICT")
        if top1_noun and top2_noun and top1_noun.casefold() != top2_noun.casefold():
            reasons.append("TOP_K_NEAR_TIE_NOUN_CONFLICT")
            selection_reasons.append("TOP_K_NEAR_TIE_NOUN_CONFLICT")
        if (
            top1_label
            and top2_label
            and top1_label.casefold() != top2_label.casefold()
            and not {"TOP_K_NEAR_TIE_VERB_CONFLICT", "TOP_K_NEAR_TIE_NOUN_CONFLICT"}.intersection(
                reasons
            )
        ):
            reasons.append("TOP_K_NEAR_TIE_LABEL_CONFLICT")
            selection_reasons.append("TOP_K_NEAR_TIE_LABEL_CONFLICT")

    camera_labels, camera_ranks, camera_scores = _camera_vote_rows(proposal, candidates)
    vote_counter = Counter(label.casefold() for label in camera_labels.values())
    canonical_label_by_key = {label.casefold(): label for label in camera_labels.values()}
    winning_key = (
        min(vote_counter, key=lambda key: (-vote_counter[key], key)) if vote_counter else None
    )
    winning_votes = vote_counter[winning_key] if winning_key is not None else 0
    observed_count = len(camera_labels)
    consensus_fraction = winning_votes / observed_count if observed_count else None
    if not camera_labels:
        reasons.append("CAMERA_EVIDENCE_UNAVAILABLE")
        selection_reasons.append("CAMERA_EVIDENCE_UNAVAILABLE")
    elif len(vote_counter) > 1:
        reasons.append("CROSS_CAMERA_TOP1_DISAGREEMENT")
        if consensus_fraction is not None and consensus_fraction < camera_consensus_threshold:
            reasons.append("LOW_CAMERA_TOP1_CONSENSUS")
            selection_reasons.append("LOW_CAMERA_TOP1_CONSENSUS")

    field_statuses: dict[str, str] = {}
    missing_critical: list[str] = []
    missing_optional: list[str] = []
    for field in FIELD_NAMES:
        status, _ = _normalised_structured_field(proposal, field)
        field_statuses[field] = status
        if status != "MEASURED":
            if field in CRITICAL_FIELD_NAMES:
                missing_critical.append(field)
                reasons.append(f"MISSING_{field.upper()}_FIELD")
                selection_reasons.append(f"MISSING_{field.upper()}_FIELD")
            else:
                missing_optional.append(field)
    if missing_optional:
        reasons.append("OPTIONAL_FIELDS_UNMEASURED")
    if _finite(proposal.get("confidence")) is None:
        reasons.append("MISSING_CONFIDENCE")
        selection_reasons.append("MISSING_CONFIDENCE")
    evidence = proposal.get("evidence", [])
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes, bytearray))
        or not evidence
    ):
        reasons.append("MISSING_EVIDENCE")
        selection_reasons.append("MISSING_EVIDENCE")
    camera_support = proposal.get("camera_support", [])
    if not isinstance(camera_support, Mapping) and (
        not isinstance(camera_support, Sequence)
        or isinstance(camera_support, (str, bytes, bytearray))
        or not camera_support
    ):
        reasons.append("MISSING_CAMERA_SUPPORT")
        selection_reasons.append("MISSING_CAMERA_SUPPORT")

    interval = proposal.get("proposal_interval")
    interval_map = interval if isinstance(interval, Mapping) else {}
    boundary_status = _text(interval_map.get("status"))
    normalized_boundary_status = boundary_status.upper() if boundary_status else "MISSING"
    if normalized_boundary_status != "MEASURED":
        reasons.append("PROPOSAL_BOUNDARY_UNMEASURED")

    compact_candidates = [
        _compact_candidate(candidate, fallback_rank=index)
        for index, candidate in enumerate(candidates, start=1)
    ]
    return {
        "proposal_id": _text(proposal.get("proposal_id")) or f"proposal-{proposal_index + 1:02d}",
        "proposal_index": proposal_index,
        "proposal_status": _text(proposal.get("proposal_status")) or "UNKNOWN",
        "label_text": (
            _text(proposal.get("label_text"))
            or (_candidate_label(top1) if top1 is not None else None)
        ),
        "confidence": _finite(proposal.get("confidence")),
        "proposal_interval": _copy_json(interval_map, field="proposal.proposal_interval"),
        "boundary_status": normalized_boundary_status,
        "field_statuses": field_statuses,
        "missing_critical_fields": missing_critical,
        "missing_optional_fields": missing_optional,
        "top_k": compact_candidates,
        "top_k_count": len(compact_candidates),
        "top1_label": _candidate_label(top1) if top1 is not None else None,
        "top2_label": _candidate_label(top2) if top2 is not None else None,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "margin_source": "recorded"
        if recorded_margin is not None
        else "reconstructed"
        if inferred_margin is not None
        else "unavailable",
        "camera_consensus": {
            "expected_camera_count": expected_camera_count,
            "observed_camera_count": observed_count,
            "per_camera_top1": camera_labels,
            "per_camera_rank": camera_ranks,
            "per_camera_score": camera_scores,
            "top1_label": (
                canonical_label_by_key.get(winning_key) if winning_key is not None else None
            ),
            "top1_votes": winning_votes,
            "top1_consensus_fraction": consensus_fraction,
            "distinct_top1_label_count": len(vote_counter),
        },
        "observed_reason_codes": _sort_reasons(reasons),
        "selection_reason_codes": _sort_reasons(selection_reasons),
    }


def _valid_policy_number(
    value: object, *, field: str, minimum: float, maximum: float | None = None
) -> float:
    result = _finite(value)
    if result is None or result < minimum or (maximum is not None and result > maximum):
        range_text = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ProductionWemmAmbiguitySelectorError(f"{field} must be finite and {range_text}")
    return result


def _normalise_policy(
    *,
    margin_threshold: float,
    camera_consensus_threshold: float,
    top_k_conflict_threshold: float | None,
    include_optional_field_gaps: bool,
    include_unmeasured_boundaries: bool,
    include_recording_edges: bool,
    include_adjacent_transitions: bool,
    max_selected: int | None,
    expected_camera_count: int,
) -> dict[str, Any]:
    margin = _valid_policy_number(margin_threshold, field="margin_threshold", minimum=0.0)
    consensus = _valid_policy_number(
        camera_consensus_threshold,
        field="camera_consensus_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    conflict = _valid_policy_number(
        margin if top_k_conflict_threshold is None else top_k_conflict_threshold,
        field="top_k_conflict_threshold",
        minimum=0.0,
    )
    if not all(
        isinstance(value, bool)
        for value in (
            include_optional_field_gaps,
            include_unmeasured_boundaries,
            include_recording_edges,
            include_adjacent_transitions,
        )
    ):
        raise ProductionWemmAmbiguitySelectorError("selection switches must be boolean")
    if (
        isinstance(expected_camera_count, bool)
        or not isinstance(expected_camera_count, int)
        or expected_camera_count <= 0
    ):
        raise ProductionWemmAmbiguitySelectorError(
            "expected_camera_count must be a positive integer"
        )
    if max_selected is not None and (
        isinstance(max_selected, bool) or not isinstance(max_selected, int) or max_selected <= 0
    ):
        raise ProductionWemmAmbiguitySelectorError(
            "max_selected must be a positive integer or null"
        )
    return {
        "margin_threshold": margin,
        "camera_consensus_threshold": consensus,
        "top_k_conflict_threshold": conflict,
        "include_optional_field_gaps": include_optional_field_gaps,
        "include_unmeasured_boundaries": include_unmeasured_boundaries,
        "include_recording_edges": include_recording_edges,
        "include_adjacent_transitions": include_adjacent_transitions,
        "max_selected": max_selected,
        "expected_camera_count": expected_camera_count,
        "threshold_interpretation": "ROUTING_PRIORITY_ONLY_NOT_A_QUALITY_THRESHOLD",
    }


def _ordinal(row: Mapping[str, Any], fallback: int) -> int:
    value = row.get("ordinal")
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _source_interval_diagnostic(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    raw = row.get("source_interval")
    interval = raw if isinstance(raw, Mapping) else {}
    start = _finite(interval.get("start_seconds"))
    end = _finite(interval.get("end_seconds"))
    status_raw = _text(interval.get("status"))
    status = status_raw.upper() if status_raw else "MISSING"
    reasons: list[str] = []
    if start is None or end is None or end <= start:
        reasons.append("SOURCE_INTERVAL_UNAVAILABLE")
    elif status == "WINDOW_CONTEXT_ONLY":
        reasons.append("SOURCE_INTERVAL_CONTEXT_ONLY")
    elif status != "WINDOW_CONTEXT_ONLY":
        # It is still source context, but an unexpected upstream status must
        # remain visible rather than being reinterpreted as an action span.
        reasons.append("SOURCE_INTERVAL_UNAVAILABLE")
    return {
        "start_seconds": start,
        "end_seconds": end,
        "status": status,
        "is_action_boundary": False,
    }, reasons


def _window_diagnostic(
    row: Mapping[str, Any],
    *,
    row_index: int,
    policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    window_id = _text(row.get("window_id"))
    recording_id = _text(row.get("recording_id"))
    if window_id is None or recording_id is None:
        return None
    camera_ids = row.get("camera_ids", [])
    declared_cameras: list[str] = []
    if isinstance(camera_ids, Sequence) and not isinstance(camera_ids, (str, bytes, bytearray)):
        declared_cameras = sorted(
            {camera_id for camera in camera_ids if (camera_id := _text(camera)) is not None}
        )
    expected_camera_count = int(policy["expected_camera_count"])
    proposals_raw = row.get("proposals", [])
    proposals = (
        [proposal for proposal in proposals_raw if isinstance(proposal, Mapping)]
        if isinstance(proposals_raw, Sequence)
        and not isinstance(proposals_raw, (str, bytes, bytearray))
        else []
    )
    proposal_diagnostics = [
        _proposal_diagnostic(
            proposal,
            proposal_index=index,
            expected_camera_count=expected_camera_count,
            margin_threshold=float(policy["margin_threshold"]),
            camera_consensus_threshold=float(policy["camera_consensus_threshold"]),
            top_k_conflict_threshold=float(policy["top_k_conflict_threshold"]),
        )
        for index, proposal in enumerate(proposals)
    ]
    source_interval, source_reasons = _source_interval_diagnostic(row)
    observed_reasons = list(source_reasons)
    selection_reasons: list[str] = []
    if not proposals:
        observed_reasons.append("NO_PROPOSALS_AVAILABLE")
        selection_reasons.append("NO_PROPOSALS_AVAILABLE")
    elif len(proposals) > 1:
        observed_reasons.append("MULTIPLE_PROPOSALS_AVAILABLE")
        # Multiple draft proposals are not a semantic failure, but they are a
        # useful Qwen review signal because the selector cannot adjudicate them.
        selection_reasons.append("MULTIPLE_PROPOSALS_AVAILABLE")
    for diagnostic in proposal_diagnostics:
        observed_reasons.extend(diagnostic["observed_reason_codes"])
        selection_reasons.extend(diagnostic["selection_reason_codes"])
        if policy["include_optional_field_gaps"] and diagnostic["missing_optional_fields"]:
            selection_reasons.append("OPTIONAL_FIELDS_UNMEASURED")
        if policy["include_unmeasured_boundaries"] and diagnostic["boundary_status"] != "MEASURED":
            selection_reasons.append("PROPOSAL_BOUNDARY_UNMEASURED")
    if "SOURCE_INTERVAL_UNAVAILABLE" in source_reasons:
        selection_reasons.append("SOURCE_INTERVAL_UNAVAILABLE")

    source_ref = row.get("source_ref")
    provenance = row.get("provenance")
    return {
        "recording_id": recording_id,
        "window_id": window_id,
        "selection_key": f"{recording_id}::{window_id}",
        "ordinal": _ordinal(row, row_index),
        "source_interval": source_interval,
        "declared_camera_ids": declared_cameras,
        "source_ref": _copy_json(source_ref, field="window.source_ref")
        if isinstance(source_ref, Mapping)
        else {"recording_id": recording_id},
        "provenance": _copy_json(provenance, field="window.provenance")
        if isinstance(provenance, Mapping)
        else {},
        "model": _copy_json(row.get("model"), field="window.model")
        if isinstance(row.get("model"), Mapping)
        else None,
        # Keep source-level review state attached to the diagnostic row.  The
        # selector only computes routing reasons; it must not erase fields a
        # reviewer needs to recover the original pre-annotation decision.
        "window_status": row.get("window_status"),
        "window_decision": row.get("window_decision"),
        "raw_candidates": _copy_json(row.get("raw_candidates", []), field="window.raw_candidates"),
        "review_contract": _copy_json(
            row.get("review_contract", {}), field="window.review_contract"
        ),
        "source_window_count": row.get("source_window_count"),
        "proposal_diagnostics": proposal_diagnostics,
        "observed_reason_codes": _sort_reasons(observed_reasons),
        "selection_reason_codes": _sort_reasons(selection_reasons),
        "review_required": True,
        "automatic_eligible": False,
        "source_context_is_action_boundary": False,
    }


def _add_contextual_reasons(rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> None:
    """Add recording-edge and neighbor-transition diagnostics without segmentation."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)
    for grouped_rows in grouped.values():
        ordered = sorted(grouped_rows, key=lambda row: (int(row["ordinal"]), str(row["window_id"])))
        if not ordered:
            continue
        for index, row in enumerate(ordered):
            observed = list(row["observed_reason_codes"])
            selected = list(row["selection_reason_codes"])
            source_window_count = row.get("source_window_count")
            known_count = (
                source_window_count
                if isinstance(source_window_count, int)
                and not isinstance(source_window_count, bool)
                and source_window_count > 0
                else None
            )
            ordinal = int(row["ordinal"])
            is_leading = ordinal == 0
            is_trailing = known_count is not None and ordinal == known_count - 1
            if is_leading:
                observed.append("RECORDING_LEADING_CONTEXT_EDGE")
                if policy["include_recording_edges"]:
                    selected.append("RECORDING_LEADING_CONTEXT_EDGE")
            if is_trailing:
                observed.append("RECORDING_TRAILING_CONTEXT_EDGE")
                if policy["include_recording_edges"]:
                    selected.append("RECORDING_TRAILING_CONTEXT_EDGE")
            if index > 0:
                previous = ordered[index - 1]
                if ordinal != int(previous["ordinal"]) + 1:
                    observed.append("NONCONTIGUOUS_WINDOW_CONTEXT")
                    if policy["include_adjacent_transitions"]:
                        selected.append("NONCONTIGUOUS_WINDOW_CONTEXT")
                current_top = _window_top1_label(row)
                previous_top = _window_top1_label(previous)
                if (
                    ordinal == int(previous["ordinal"]) + 1
                    and current_top
                    and previous_top
                    and current_top.casefold() != previous_top.casefold()
                ):
                    observed.append("ADJACENT_TOP1_TRANSITION")
                    previous_observed = list(previous["observed_reason_codes"])
                    previous_observed.append("ADJACENT_TOP1_TRANSITION")
                    previous["observed_reason_codes"] = _sort_reasons(previous_observed)
                    if policy["include_adjacent_transitions"]:
                        selected.append("ADJACENT_TOP1_TRANSITION")
                        previous_selected = list(previous["selection_reason_codes"])
                        previous_selected.append("ADJACENT_TOP1_TRANSITION")
                        previous["selection_reason_codes"] = _sort_reasons(previous_selected)
            row["observed_reason_codes"] = _sort_reasons(observed)
            row["selection_reason_codes"] = _sort_reasons(selected)


def _window_top1_label(row: Mapping[str, Any]) -> str | None:
    proposals = row.get("proposal_diagnostics", [])
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes, bytearray)):
        return None
    for proposal in proposals:
        if isinstance(proposal, Mapping):
            label = _text(proposal.get("top1_label") or proposal.get("label_text"))
            if label:
                return label
    return None


def _priority_score(reason_codes: Sequence[str]) -> int:
    """A transparent queue-order score, not a confidence or quality metric."""

    weights = {
        "NO_PROPOSALS_AVAILABLE": 7,
        "TOP_K_EMPTY": 7,
        "TOP_K_INSUFFICIENT": 6,
        "MARGIN_UNAVAILABLE": 5,
        "LOW_TOP1_TOP2_MARGIN": 5,
        "TOP_K_NEAR_TIE_VERB_CONFLICT": 4,
        "TOP_K_NEAR_TIE_NOUN_CONFLICT": 4,
        "TOP_K_NEAR_TIE_LABEL_CONFLICT": 3,
        "CAMERA_EVIDENCE_UNAVAILABLE": 5,
        "LOW_CAMERA_TOP1_CONSENSUS": 5,
        "MISSING_VERB_FIELD": 5,
        "MISSING_NOUN_FIELD": 5,
        "MISSING_CONFIDENCE": 3,
        "MISSING_EVIDENCE": 3,
        "MISSING_CAMERA_SUPPORT": 3,
        "MULTIPLE_PROPOSALS_AVAILABLE": 3,
        "SOURCE_INTERVAL_UNAVAILABLE": 4,
        "PROPOSAL_BOUNDARY_UNMEASURED": 2,
        "OPTIONAL_FIELDS_UNMEASURED": 1,
        "RECORDING_LEADING_CONTEXT_EDGE": 1,
        "RECORDING_TRAILING_CONTEXT_EDGE": 1,
        "ADJACENT_TOP1_TRANSITION": 2,
        "NONCONTIGUOUS_WINDOW_CONTEXT": 3,
    }
    return sum(weights.get(reason, 1) for reason in set(reason_codes))


def _selected_row(row: Mapping[str, Any]) -> dict[str, Any]:
    proposal_rows = row.get("proposal_diagnostics", [])
    compact_proposals: list[dict[str, Any]] = []
    if isinstance(proposal_rows, Sequence) and not isinstance(
        proposal_rows, (str, bytes, bytearray)
    ):
        for raw in proposal_rows:
            if not isinstance(raw, Mapping):
                continue
            compact_proposals.append(_copy_json(raw, field="selected.proposal"))
    reasons = _sort_reasons(
        [str(reason) for reason in row.get("selection_reason_codes", []) if _text(reason)]
    )
    return {
        "selection_key": row["selection_key"],
        "selection_status": "QWEN_REVIEW_CANDIDATE",
        "review_required": True,
        "automatic_eligible": False,
        "recording_id": row["recording_id"],
        "window_id": row["window_id"],
        "ordinal": row["ordinal"],
        "source_interval": _copy_json(row["source_interval"], field="selected.source_interval"),
        "source_context_is_action_boundary": False,
        "declared_camera_ids": _copy_json(row["declared_camera_ids"], field="selected.cameras"),
        "source_ref": _copy_json(row["source_ref"], field="selected.source_ref"),
        "provenance": _copy_json(row["provenance"], field="selected.provenance"),
        "model": _copy_json(row.get("model"), field="selected.model"),
        # Preserve source routing state and the unnormalised candidate card.
        # These are evidence for the later review step, not a new decision.
        "window_status": row.get("window_status"),
        "window_decision": row.get("window_decision"),
        "raw_candidates": _copy_json(
            row.get("raw_candidates", []), field="selected.raw_candidates"
        ),
        "review_contract": _copy_json(
            row.get("review_contract", {}), field="selected.review_contract"
        ),
        "reason_codes": reasons,
        "observed_reason_codes": _copy_json(
            row["observed_reason_codes"], field="selected.observed_reasons"
        ),
        "priority_score": _priority_score(reasons),
        "priority_interpretation": "QUEUE_ORDER_ONLY_NOT_MODEL_QUALITY",
        "proposal_diagnostics": compact_proposals,
        "raw_top_k_retained_in_source_sidecar": True,
    }


def select_production_wemm_ambiguities(
    inputs: Mapping[str, Any] | Sequence[Any] | str | Path,
    *,
    margin_threshold: float = 0.01,
    camera_consensus_threshold: float = 2.0 / 3.0,
    top_k_conflict_threshold: float | None = None,
    include_optional_field_gaps: bool = False,
    include_unmeasured_boundaries: bool = False,
    include_recording_edges: bool = False,
    include_adjacent_transitions: bool = False,
    max_selected: int | None = None,
    expected_camera_count: int = 6,
) -> dict[str, Any]:
    """Select ambiguous WeMM processing-window rows for later Qwen review.

    By default, systematic optional-field, unmeasured-boundary, edge, and
    adjacent-transition gaps are reported but do not route every window to
    Qwen.  Turn on their explicit switches when a later stage intends to use
    Qwen for those completeness/context tasks.  This keeps a missing WeMM
    action boundary distinct from an inferred action boundary, keeps the Qwen
    path selective, and keeps thresholds from becoming quality claims.
    """

    policy = _normalise_policy(
        margin_threshold=margin_threshold,
        camera_consensus_threshold=camera_consensus_threshold,
        top_k_conflict_threshold=top_k_conflict_threshold,
        include_optional_field_gaps=include_optional_field_gaps,
        include_unmeasured_boundaries=include_unmeasured_boundaries,
        include_recording_edges=include_recording_edges,
        include_adjacent_transitions=include_adjacent_transitions,
        max_selected=max_selected,
        expected_camera_count=expected_camera_count,
    )
    documents, invalid_inputs, rejected_inputs = _collect_documents(inputs)
    raw_rows: list[dict[str, Any]] = []
    for path, payload in documents:
        normalised, invalid = _normalise_document(path, payload)
        raw_rows.extend(normalised)
        invalid_inputs.extend(invalid)

    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        diagnostic = _window_diagnostic(row, row_index=index, policy=policy)
        if diagnostic is None:
            invalid_inputs.append(
                {
                    "path": str(row.get("provenance", {}).get("input_path", "<input>")),
                    "reason": "WINDOW_OR_RECORDING_ID_MISSING",
                }
            )
            continue
        diagnostics.append(diagnostic)
    _add_contextual_reasons(diagnostics, policy)

    duplicate_keys = sorted(
        key
        for key, count in Counter(str(row["selection_key"]) for row in diagnostics).items()
        if count > 1
    )
    for row in diagnostics:
        if str(row["selection_key"]) in duplicate_keys:
            observed = list(row["observed_reason_codes"])
            observed.append("NONCONTIGUOUS_WINDOW_CONTEXT")
            row["observed_reason_codes"] = _sort_reasons(observed)

    selected = [row for row in diagnostics if row["selection_reason_codes"]]
    selected.sort(
        key=lambda row: (
            -_priority_score(row["selection_reason_codes"]),
            str(row["recording_id"]),
            int(row["ordinal"]),
            str(row["window_id"]),
        )
    )
    truncated_count = 0
    if max_selected is not None and len(selected) > max_selected:
        truncated_count = len(selected) - max_selected
        selected = selected[:max_selected]

    observed_reason_counts = Counter(
        reason for row in diagnostics for reason in row["observed_reason_codes"]
    )
    selected_reason_counts = Counter(
        reason for row in selected for reason in row["selection_reason_codes"]
    )
    recording_count = len({str(row["recording_id"]) for row in diagnostics})
    # Keep source review contracts visible at the queue level as well as on
    # each selected row.  A mixed input may legitimately contain contracts
    # from different artifact versions; do not silently merge those objects.
    source_contracts: list[dict[str, Any]] = []

    def remember_contract(value: object, *, field: str) -> None:
        if not isinstance(value, Mapping):
            return
        detached = _copy_json(value, field=field)
        if isinstance(detached, dict) and detached not in source_contracts:
            source_contracts.append(detached)

    # Prefer document-level contracts so an empty-but-valid source document
    # still carries its review semantics into the queue report.  Row-level
    # overrides are also recorded for forward-compatible artifacts.
    for document_index, (_, payload) in enumerate(documents):
        remember_contract(
            payload.get("review_contract"),
            field=f"report.review_contracts[{document_index}]",
        )
    for row_index, row in enumerate(raw_rows):
        remember_contract(
            row.get("review_contract"), field=f"report.row_review_contracts[{row_index}]"
        )
    report_review_contract: dict[str, Any] | None = (
        source_contracts[0] if len(source_contracts) == 1 else None
    )
    return {
        "format": AMBIGUITY_SELECTION_FORMAT,
        "authority": AUTHORITY,
        "status": "ROUTING_QUEUE_READY" if selected else "NO_QWEN_REVIEW_ROWS_SELECTED",
        "production_eligible": False,
        "official_quality_status": OFFICIAL_QUALITY_STATUS,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "quality_claim": False,
        "routing_scope": "WEMM_ONLY_OFFLINE_AMBIGUITY_SELECTION",
        "review_contract": report_review_contract,
        "source_contracts": source_contracts,
        "policy": policy,
        "summary": {
            "recording_count": recording_count,
            "input_window_count": len(diagnostics),
            "selected_window_count": len(selected),
            "unselected_window_count": len(diagnostics)
            - len([row for row in diagnostics if row["selection_reason_codes"]]),
            "selection_fraction_before_cap": (
                len([row for row in diagnostics if row["selection_reason_codes"]])
                / len(diagnostics)
                if diagnostics
                else 0.0
            ),
            "selection_truncated_count": truncated_count,
            "selected_reason_counts": dict(
                sorted(
                    selected_reason_counts.items(),
                    key=lambda pair: (
                        _REASON_INDEX.get(str(pair[0]), 999),
                        str(pair[0]),
                    ),
                )
            ),
            "observed_reason_counts": dict(
                sorted(
                    observed_reason_counts.items(),
                    key=lambda pair: (
                        _REASON_INDEX.get(str(pair[0]), 999),
                        str(pair[0]),
                    ),
                )
            ),
            "duplicate_selection_keys": duplicate_keys,
            "windows_are_action_segments": False,
        },
        # Only selected rows appear here, so a future Qwen runner cannot
        # accidentally treat every diagnostic window as an invocation request.
        "windows": [_selected_row(row) for row in selected],
        "input_artifacts": {
            "document_count": len(documents),
            "formats": dict(
                sorted(Counter(str(payload.get("format")) for _, payload in documents).items())
            ),
            "invalid_inputs": invalid_inputs,
            "rejected_inputs": rejected_inputs,
        },
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "qwen_read": False,
            "qwen_invoked": False,
            "mage_read": False,
            "mage_invoked": False,
            "epic_ontology_read": False,
            "mapper_read": False,
            "raw_wemm_output_modified": False,
            "hash_or_digest_computed": False,
            "heldout_100_opened": False,
        },
        "limitations": [
            (
                "This is a routing/review queue, not an annotation, action decision, "
                "or quality report."
            ),
            (
                "Margins and consensus fractions are recorded retrieval diagnostics, "
                "not calibrated confidence."
            ),
            (
                "Processing-window source intervals remain context only and are never "
                "action boundaries."
            ),
            (
                "Unmeasured optional fields and action boundaries are reported separately; "
                "they route only when explicitly enabled."
            ),
            (
                "Selected rows preserve source window status, decision, raw candidate "
                "cards, and review-contract snapshots; full raw model output remains "
                "in the source sidecar."
            ),
            (
                "When input artifacts carry different review contracts, the report-level "
                "review_contract is null and source_contracts lists each distinct source "
                "contract instead of silently merging them."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operational summary without claiming model quality."""

    summary = _mapping(report.get("summary", {}), field="report.summary")
    policy = _mapping(report.get("policy", {}), field="report.policy")
    lines = [
        "# Production WeMM ambiguity selection",
        "",
        (
            "> **ROUTING-ONLY / NON-GOLD.** This queue does not establish annotation "
            "quality or production eligibility."
        ),
        "",
        (
            "- Input processing windows (not action segments): "
            f"`{summary.get('input_window_count', 0)}`"
        ),
        f"- Selected for later Qwen review: `{summary.get('selected_window_count', 0)}`",
        (
            "- Selection fraction before cap: "
            f"`{float(summary.get('selection_fraction_before_cap', 0.0)):.1%}`"
        ),
        f"- Margin routing threshold: `{policy.get('margin_threshold')}`",
        f"- Camera-consensus routing threshold: `{policy.get('camera_consensus_threshold')}`",
        f"- Optional gaps route enabled: `{policy.get('include_optional_field_gaps')}`",
        f"- Unmeasured boundaries route enabled: `{policy.get('include_unmeasured_boundaries')}`",
        "",
        "## Selected reason counts",
        "",
        "| Reason | Windows |",
        "|---|---:|",
    ]
    selected_counts = summary.get("selected_reason_counts", {})
    if isinstance(selected_counts, Mapping) and selected_counts:
        for reason, count in selected_counts.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| _none_ | 0 |")
    lines.extend(
        [
            "",
            (
                "Thresholds only prioritize review workload. A selected row still "
                "requires a complete native-video Qwen pass and explicit human review; "
                "an unselected row is not accepted automatically."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AMBIGUITY_SELECTION_FORMAT",
    "AUTHORITY",
    "OFFICIAL_GOLD_STATUS",
    "OFFICIAL_QUALITY_STATUS",
    "ProductionWemmAmbiguitySelectorError",
    "load_json",
    "render_markdown",
    "select_production_wemm_ambiguities",
]
