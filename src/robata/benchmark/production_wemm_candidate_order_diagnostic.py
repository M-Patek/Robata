"""Small candidate-order intervention for the production WeMM/Qwen shadow path.

The production WeMM ambiguity selector stores a ranked ``top_k`` list.  The
native Qwen verifier renders that list in the order it receives while also
showing the original numeric ranks.  This module creates a *derived* one-window
selection in which only the presentation order of that list changes.  Rank,
label, score, source interval, and all source provenance remain untouched.

The artifact is diagnostic-only.  It does not invoke a model, decode media,
read gold/EPIC/Mapper/Mage artifacts, infer action boundaries, or compute an
identity digest.  It is intended for a paired ``as_is``/``reverse`` (and
optionally deterministic ``shuffle``) compact Qwen run on one camera.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

AMBIGUITY_SELECTION_FORMAT: Final = "robata-production-wemm-ambiguity-selection-v1"
ORDER_DIAGNOSTIC_FORMAT: Final = "robata-production-wemm-candidate-order-diagnostic-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
ORDER_MODES: Final = ("as_is", "reverse", "shuffle")


class ProductionWemmCandidateOrderDiagnosticError(ValueError):
    """Raised when a candidate-order diagnostic cannot be built."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmCandidateOrderDiagnosticError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmCandidateOrderDiagnosticError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise ProductionWemmCandidateOrderDiagnosticError(f"{field} must be non-empty text")
    result = str(value).strip()
    if not result and not allow_empty:
        raise ProductionWemmCandidateOrderDiagnosticError(f"{field} must be non-empty text")
    return result


def _load_json(value: Mapping[str, Any] | str | Path, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = Path(value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmCandidateOrderDiagnosticError(
            f"could not read {field} {path}: {exc}"
        ) from exc
    return dict(_mapping(payload, field=field))


def _candidate_label(candidate: Mapping[str, Any]) -> str:
    """Use the same compatibility spellings as the native verifier."""

    for key in ("label_text", "canonical_label", "mapped_action", "raw_label"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    structured = candidate.get("structured_labels")
    if isinstance(structured, Mapping):
        values: list[str] = []
        for key in ("verb", "noun"):
            value = structured.get(key)
            if isinstance(value, Mapping):
                value = value.get("value")
            if value is not None and str(value).strip():
                values.append(str(value).strip())
        if values:
            return " ".join(values)
    return " ".join(
        value.strip()
        for value in (candidate.get("verb"), candidate.get("noun"))
        if isinstance(value, str) and value.strip()
    )


def _rank(candidate: Mapping[str, Any], *, field: str) -> int:
    value = candidate.get("rank")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProductionWemmCandidateOrderDiagnosticError(
            f"{field}.rank must be a positive integer"
        )
    return value


def _order_signature(candidate: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Return audit metadata without normalising the candidate itself."""

    return {
        "rank": _rank(candidate, field=field),
        "label": _candidate_label(candidate),
        # Keep the score exactly as supplied.  It is copied for comparison and
        # reporting only; no rounding or recalibration occurs.
        "score": copy.deepcopy(candidate.get("score")),
    }


def _validate_top_k(raw: object, *, field: str) -> list[dict[str, Any]]:
    values = _sequence(raw, field=field)
    result: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for index, value in enumerate(values):
        candidate = _mapping(value, field=f"{field}[{index}]")
        rank = _rank(candidate, field=f"{field}[{index}]")
        if rank in seen_ranks:
            raise ProductionWemmCandidateOrderDiagnosticError(
                f"{field} contains duplicate rank {rank}"
            )
        seen_ranks.add(rank)
        label = _candidate_label(candidate)
        if not label:
            raise ProductionWemmCandidateOrderDiagnosticError(f"{field}[{index}] lacks a label")
        result.append(dict(copy.deepcopy(candidate)))
    if not result:
        raise ProductionWemmCandidateOrderDiagnosticError(f"{field} must not be empty")
    return result


def order_top_k(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    seed: str = "",
    window_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a presentation permutation while preserving source ranks.

    ``mode=reverse`` is an exact reverse of the supplied array.  ``shuffle``
    uses a local deterministic PRNG seeded by ``seed|window_id``; source rank
    fields are deliberately *not* renumbered.  The returned metadata is safe
    to persist in a diagnostic sidecar and contains only rank/label/score
    snapshots, not media bytes or derived identities.
    """

    normalised_mode = str(mode).strip().lower().replace("-", "_")
    if normalised_mode not in ORDER_MODES:
        raise ProductionWemmCandidateOrderDiagnosticError("mode must be as_is, reverse, or shuffle")
    window_text = str(window_id).strip()
    rows = _validate_top_k(candidates, field="top_k")
    before = [_order_signature(row, field=f"top_k[{index}]") for index, row in enumerate(rows)]
    ordered = list(rows)
    if normalised_mode == "reverse":
        ordered.reverse()
    elif normalised_mode == "shuffle":
        random.Random(f"{seed}|{window_text}").shuffle(ordered)
        # A one-in-N random identity permutation is valid, but unhelpful for a
        # paired intervention.  Make the smallest deterministic swap so a
        # requested shuffle always changes presentation when possible.
        if len(ordered) > 1 and all(
            left is right for left, right in zip(ordered, rows, strict=True)
        ):
            ordered[0], ordered[1] = ordered[1], ordered[0]
    after = [_order_signature(row, field=f"top_k[{index}]") for index, row in enumerate(ordered)]

    # Compare by source rank rather than list position.  This proves that the
    # intervention changed presentation only; it intentionally does not use a
    # digest or hash-based identity.
    before_by_rank = sorted(before, key=lambda item: int(item["rank"]))
    after_by_rank = sorted(after, key=lambda item: int(item["rank"]))
    if before_by_rank != after_by_rank:
        raise ProductionWemmCandidateOrderDiagnosticError(
            "candidate rank/label/score changed during order intervention"
        )
    return ordered, {
        "mode": normalised_mode,
        "seed": str(seed) if normalised_mode == "shuffle" else None,
        "window_id": window_text or None,
        "algorithm": (
            "identity"
            if normalised_mode == "as_is"
            else "exact_array_reverse"
            if normalised_mode == "reverse"
            else "random.Random(seed|window_id).shuffle_with_nonidentity_fallback"
        ),
        "order_changed": before != after,
        "rank_label_score_preserved": True,
        "before": before,
        "after": after,
    }


def _proposal_diagnostics(row: Mapping[str, Any], *, window_id: str) -> list[dict[str, Any]]:
    raw = _sequence(row.get("proposal_diagnostics"), field=f"{window_id}.proposal_diagnostics")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        result.append(dict(_mapping(value, field=f"{window_id}.proposal_diagnostics[{index}]")))
    if not result:
        raise ProductionWemmCandidateOrderDiagnosticError(
            f"{window_id}.proposal_diagnostics must not be empty"
        )
    return result


def _find_window(document: Mapping[str, Any], window_id: str | None) -> dict[str, Any]:
    raw = _sequence(document.get("windows"), field="ambiguity_selection.windows")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        row = dict(_mapping(value, field=f"ambiguity_selection.windows[{index}]"))
        current_id = _text(
            row.get("window_id"), field=f"ambiguity_selection.windows[{index}].window_id"
        )
        if current_id in seen:
            raise ProductionWemmCandidateOrderDiagnosticError(
                f"duplicate window_id in ambiguity selection: {current_id}"
            )
        seen.add(current_id)
        rows.append(row)
    if not rows:
        raise ProductionWemmCandidateOrderDiagnosticError(
            "ambiguity_selection.windows must not be empty"
        )
    if window_id is None:
        return rows[0]
    wanted = _text(window_id, field="window_id")
    for row in rows:
        if str(row.get("window_id")) == wanted:
            return row
    raise ProductionWemmCandidateOrderDiagnosticError(f"window_id not found: {wanted}")


def _restrict_camera(
    row: dict[str, Any], *, camera_id: str | None, window_id: str
) -> dict[str, Any] | None:
    if camera_id is None:
        return None
    camera = _text(camera_id, field="camera_id")
    raw_ids = row.get("declared_camera_ids", row.get("camera_ids"))
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes, bytearray)):
        source_ref = row.get("source_ref")
        source = source_ref.get("source") if isinstance(source_ref, Mapping) else None
        raw_ids = source.get("camera_ids") if isinstance(source, Mapping) else None
    declared = [str(value).strip() for value in raw_ids or [] if str(value).strip()]
    if declared and camera not in declared:
        raise ProductionWemmCandidateOrderDiagnosticError(
            f"camera_id {camera!r} is not declared for {window_id}"
        )
    original = list(declared)
    row["declared_camera_ids"] = [camera]
    return {
        "requested_camera_id": camera,
        "original_declared_camera_ids": original,
        "applied_to": "window.declared_camera_ids",
    }


def _summary_for_one_window(document: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    summary = (
        dict(document.get("summary", {})) if isinstance(document.get("summary"), Mapping) else {}
    )
    summary.update(
        {
            "selected_recording_count": 1,
            "selected_window_count": 1,
            "input_window_count": 1,
            "diagnostic_source_window_count": 1,
            "diagnostic_camera_count": len(row.get("declared_camera_ids", []))
            if isinstance(row.get("declared_camera_ids"), Sequence)
            and not isinstance(row.get("declared_camera_ids"), (str, bytes, bytearray))
            else None,
        }
    )
    return summary


def build_candidate_order_diagnostic(
    selection: Mapping[str, Any] | str | Path,
    *,
    mode: str,
    window_id: str | None = None,
    proposal_index: int | None = None,
    seed: str = "pilot-order-v1",
    camera_id: str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one derived selection row for an order intervention."""

    document = _load_json(selection, field="ambiguity_selection")
    if document.get("format") != AMBIGUITY_SELECTION_FORMAT:
        raise ProductionWemmCandidateOrderDiagnosticError(
            f"ambiguity_selection has unsupported format; expected {AMBIGUITY_SELECTION_FORMAT}"
        )
    source_row = _find_window(document, window_id)
    row = copy.deepcopy(source_row)
    selected_window_id = _text(row.get("window_id"), field="window_id")
    diagnostics = _proposal_diagnostics(row, window_id=selected_window_id)
    if proposal_index is None:
        if len(diagnostics) != 1:
            raise ProductionWemmCandidateOrderDiagnosticError(
                "window has multiple proposal_diagnostics; pass proposal_index explicitly"
            )
        selected_index = 0
    else:
        if (
            isinstance(proposal_index, bool)
            or not isinstance(proposal_index, int)
            or proposal_index < 0
        ):
            raise ProductionWemmCandidateOrderDiagnosticError(
                "proposal_index must be a non-negative integer"
            )
        if proposal_index >= len(diagnostics):
            raise ProductionWemmCandidateOrderDiagnosticError(
                "proposal_index is outside proposal_diagnostics"
            )
        selected_index = proposal_index
    diagnostic = diagnostics[selected_index]
    ordered, order_meta = order_top_k(
        _sequence(
            diagnostic.get("top_k"),
            field=f"{selected_window_id}.proposal_diagnostics[{selected_index}].top_k",
        ),
        mode=mode,
        seed=seed,
        window_id=selected_window_id,
    )
    diagnostic["top_k"] = ordered
    row["proposal_diagnostics"] = diagnostics
    camera_meta = _restrict_camera(row, camera_id=camera_id, window_id=selected_window_id)
    row["source_context_is_action_boundary"] = False

    result = copy.deepcopy(document)
    result["format"] = AMBIGUITY_SELECTION_FORMAT
    result["status"] = "CANDIDATE_ORDER_DIAGNOSTIC"
    result["production_eligible"] = False
    result["quality_claim"] = False
    result["official_gold_status"] = "NOT_ESTABLISHED"
    result["official_quality_status"] = "NOT_MEASURED"
    result["windows"] = [row]
    result["summary"] = _summary_for_one_window(result, row)
    controls = (
        dict(result.get("controls", {})) if isinstance(result.get("controls"), Mapping) else {}
    )
    controls.update(
        {
            "model_invoked": False,
            "qwen_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "epic_ontology_read": False,
            "mapper_read": False,
            "mage_read": False,
            "hash_or_digest_computed": False,
            "heldout_100_opened": False,
            "candidate_order_intervention": True,
        }
    )
    result["controls"] = controls
    result["diagnostic"] = {
        "format": ORDER_DIAGNOSTIC_FORMAT,
        "purpose": "measure Qwen sensitivity to WeMM candidate presentation order",
        "source_selection": str(source_path) if source_path is not None else None,
        "window_id": selected_window_id,
        "recording_id": row.get("recording_id"),
        "proposal_index": selected_index,
        "top_k_path": f"windows[0].proposal_diagnostics[{selected_index}].top_k",
        "camera_scope": camera_meta or {"requested_camera_id": None},
        "order": order_meta,
        "raw_wemm_output_modified": False,
        "action_boundaries_inferred": False,
        "quality_claim": False,
        "official_quality_status": "NOT_MEASURED",
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "epic_ontology_read": False,
            "mapper_read": False,
            "mage_read": False,
            "hash_or_digest_computed": False,
        },
    }
    return result


def build_candidate_order_variants(
    selection: Mapping[str, Any] | str | Path,
    *,
    modes: Sequence[str] = ORDER_MODES,
    window_id: str | None = None,
    proposal_index: int | None = None,
    seed: str = "pilot-order-v1",
    camera_id: str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build multiple independent derived selections from the same source row."""

    result: dict[str, dict[str, Any]] = {}
    for mode in modes:
        normalised = str(mode).strip().lower().replace("-", "_")
        if normalised in result:
            raise ProductionWemmCandidateOrderDiagnosticError(
                f"duplicate diagnostic mode: {normalised}"
            )
        result[normalised] = build_candidate_order_diagnostic(
            selection,
            mode=normalised,
            window_id=window_id,
            proposal_index=proposal_index,
            seed=seed,
            camera_id=camera_id,
            source_path=source_path,
        )
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable artifact summary."""

    diagnostic = report.get("diagnostic")
    diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
    order = diagnostic.get("order")
    order = order if isinstance(order, Mapping) else {}
    before = order.get("before", [])
    after = order.get("after", [])

    def _cells(values: object) -> str:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            return ""
        return ", ".join(
            f"r{item.get('rank')} {item.get('label')} ({item.get('score')})"
            for item in values
            if isinstance(item, Mapping)
        )

    camera_scope = diagnostic.get("camera_scope")
    camera_value = (
        camera_scope.get("requested_camera_id") if isinstance(camera_scope, Mapping) else None
    )
    lines = [
        "# WeMM candidate-order diagnostic",
        "",
        f"- **Status:** `{report.get('status')}`",
        f"- **Window:** `{diagnostic.get('window_id')}`",
        f"- **Recording:** `{diagnostic.get('recording_id')}`",
        f"- **Mode:** `{order.get('mode')}`",
        f"- **Camera:** `{camera_value}`",
        f"- **Rank/label/score preserved:** `{order.get('rank_label_score_preserved')}`",
        "",
        (
            "The source selection and raw WeMM output are unchanged; only the "
            "array presentation order under the selected proposal was changed. "
            "This is a non-production diagnostic and does not establish "
            "annotation quality."
        ),
        "",
        f"- **Before:** {_cells(before)}",
        f"- **After:** {_cells(after)}",
        "",
        (
            "No model was invoked, media was decoded, gold/EPIC/Mapper/Mage "
            "artifacts were read, or hash/digest was computed while creating "
            "this artifact."
        ),
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "AMBIGUITY_SELECTION_FORMAT",
    "AUTHORITY",
    "ORDER_DIAGNOSTIC_FORMAT",
    "ORDER_MODES",
    "ProductionWemmCandidateOrderDiagnosticError",
    "build_candidate_order_diagnostic",
    "build_candidate_order_variants",
    "order_top_k",
    "render_markdown",
]
