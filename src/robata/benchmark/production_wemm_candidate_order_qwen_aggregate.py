"""Post-hoc aggregation for the WeMM candidate-order Qwen diagnostic.

The diagnostic runner executes the same native-video window with ``as_is``,
``reverse``, and (optionally) deterministic ``shuffle`` candidate presentation
orders.  This module compares only recorded verifier outputs.  It never loads
a model, decodes media, reads gold/EPIC/Mapper/Mage artifacts, or computes an
identity hash/digest.  The report is an order-sensitivity diagnostic, not a
quality or production-accuracy claim.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

AGGREGATE_FORMAT: Final = "robata-production-wemm-candidate-order-qwen-aggregate-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
DEFAULT_MODES: Final = ("as_is", "reverse", "shuffle")


class ProductionWemmCandidateOrderQwenAggregateError(ValueError):
    """Raised when a candidate-order Qwen sidecar is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmCandidateOrderQwenAggregateError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmCandidateOrderQwenAggregateError(f"{field} must be an array")
    return value


def load_json(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmCandidateOrderQwenAggregateError(
            f"could not read diagnostic sidecar {path}: {exc}"
        ) from exc
    return dict(_mapping(payload, field="diagnostic sidecar"))


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parsed(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("parsed_verification")
    return value if isinstance(value, Mapping) else {}


def _selected_verdict(parsed: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = parsed.get("candidate_verdicts")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    selected_rank = parsed.get("selected_rank")
    verdicts = [item for item in raw if isinstance(item, Mapping)]
    if isinstance(selected_rank, int) and not isinstance(selected_rank, bool):
        for verdict in verdicts:
            if verdict.get("rank") == selected_rank:
                return verdict
    return verdicts[0] if len(verdicts) == 1 else None


def _evidence(parsed: Mapping[str, Any]) -> tuple[str, ...]:
    verdict = _selected_verdict(parsed)
    if verdict is None:
        return ()
    raw = verdict.get("evidence", [])
    if isinstance(raw, str):
        return (raw.strip(),) if raw.strip() else ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _evidence_normalized(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(" ".join(str(value).casefold().split()) for value in values if str(value).strip())


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _row_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    parsed = _parsed(row)
    # The resident pilot checkpoint intentionally stores a flattened row
    # (decision/rank/parse_status/evidence beside the raw verifier report),
    # whereas per-call native sidecars nest these values under
    # ``parsed_verification``.  Accept both shapes without changing either
    # artifact.
    selected_rank = parsed.get("selected_rank", row.get("selected_rank"))
    if not isinstance(selected_rank, int) or isinstance(selected_rank, bool) or selected_rank < 1:
        selected_rank = None
    parse_status = _text(parsed.get("parse_status", row.get("parse_status"))) or "MISSING"
    evidence = _evidence(parsed)
    if not evidence:
        raw_evidence = row.get("evidence", [])
        if isinstance(raw_evidence, str):
            evidence = (raw_evidence.strip(),) if raw_evidence.strip() else ()
        elif isinstance(raw_evidence, Sequence) and not isinstance(
            raw_evidence, (str, bytes, bytearray)
        ):
            evidence = tuple(str(item).strip() for item in raw_evidence if str(item).strip())
    decision = _text(parsed.get("decision")) or _text(row.get("decision")) or "abstain"
    generation_seconds = _finite(row.get("generation_seconds"))
    return {
        "status": _text(row.get("status")) or "MISSING",
        "decision": decision,
        "selected_rank": selected_rank,
        "parse_status": parse_status,
        "accept_contract_ok": parsed.get("accept_contract_ok", row.get("accept_contract_ok")),
        "evidence": list(evidence),
        "evidence_normalized": list(_evidence_normalized(evidence)),
        "generation_seconds": generation_seconds,
        "output_tokens": row.get("output_tokens"),
    }


def _distinct(values: Sequence[object]) -> list[object]:
    result: list[object] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _pairwise_flip(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    compared = 0
    flips = 0
    rows: list[str] = []
    for window_id, modes in groups.items():
        if left not in modes or right not in modes:
            continue
        compared += 1
        left_decision = modes[left]["decision"]
        right_decision = modes[right]["decision"]
        left_rank = modes[left]["selected_rank"]
        right_rank = modes[right]["selected_rank"]
        if left_decision != right_decision or left_rank != right_rank:
            flips += 1
            rows.append(window_id)
    return {
        "left": left,
        "right": right,
        "compared_windows": compared,
        "difference_windows": flips,
        "difference_rate": flips / compared if compared else None,
        "window_ids": rows,
        "definition": "decision or selected_rank differs between the two presentation orders",
    }


def aggregate_candidate_order_qwen_diagnostic(
    sidecar: Mapping[str, Any] | str | Path,
    *,
    expected_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-window/per-order Qwen rows without invoking anything."""

    payload = load_json(sidecar)
    raw_rows = _sequence(payload.get("rows", []), field="sidecar.rows")
    modes = tuple(
        str(mode).strip().lower().replace("-", "_")
        for mode in (expected_modes or payload.get("modes") or DEFAULT_MODES)
        if str(mode).strip()
    )
    if not modes:
        raise ProductionWemmCandidateOrderQwenAggregateError("expected_modes must not be empty")
    if len(set(modes)) != len(modes):
        raise ProductionWemmCandidateOrderQwenAggregateError("expected_modes contains duplicates")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    row_errors: list[str] = []
    raw_row_count = 0
    for index, value in enumerate(raw_rows):
        raw_row_count += 1
        row = _mapping(value, field=f"sidecar.rows[{index}]")
        window_id = _text(row.get("window_id"))
        mode = _text(row.get("mode"))
        if window_id is None or mode is None:
            row_errors.append(f"row {index}: window_id and mode are required")
            continue
        mode = mode.casefold().replace("-", "_")
        if mode not in modes:
            row_errors.append(f"row {index}: unexpected mode {mode!r}")
            continue
        if mode in grouped[window_id]:
            row_errors.append(f"row {index}: duplicate window/mode {window_id}/{mode}")
            continue
        projection = _row_projection(row)
        projection["recording_id"] = _text(row.get("recording_id"))
        grouped[window_id][mode] = projection

    per_window: list[dict[str, Any]] = []
    for window_id in sorted(grouped):
        mode_rows = grouped[window_id]
        complete = all(mode in mode_rows for mode in modes)
        present = {mode: mode_rows[mode] for mode in modes if mode in mode_rows}
        valid = {
            mode: row
            for mode, row in present.items()
            if row.get("status") == "SUCCEEDED" and row.get("parse_status") == "PARSED"
        }
        ranks = [
            row.get("selected_rank")
            for row in valid.values()
            if row.get("selected_rank") is not None
        ]
        decisions = [row.get("decision") for row in valid.values()]
        evidence = [tuple(row.get("evidence_normalized", ())) for row in valid.values()]
        rank_invariant = bool(valid) and len(ranks) == len(valid) and len(set(ranks)) == 1
        decision_invariant = (
            bool(valid) and len(decisions) == len(valid) and len(set(decisions)) == 1
        )
        evidence_invariant = bool(valid) and len(evidence) == len(valid) and len(set(evidence)) == 1
        per_window.append(
            {
                "window_id": window_id,
                "recording_id": next(
                    (
                        row.get("recording_id")
                        for row in present.values()
                        if row.get("recording_id")
                    ),
                    None,
                ),
                "complete": complete,
                "valid_mode_count": len(valid),
                "modes": present,
                "rank_invariant": rank_invariant,
                "decision_invariant": decision_invariant,
                "evidence_invariant": evidence_invariant,
                "rank_changed": bool(valid) and len(set(ranks)) > 1,
                "decision_changed": bool(valid) and len(set(decisions)) > 1,
                "evidence_changed": bool(valid) and len(set(evidence)) > 1,
            }
        )

    complete_windows = [row for row in per_window if row["complete"]]
    valid_complete = [row for row in complete_windows if row["valid_mode_count"] == len(modes)]
    all_projections = [mode_row for window in per_window for mode_row in window["modes"].values()]
    parse_failures = [
        row
        for row in all_projections
        if row.get("status") != "SUCCEEDED" or row.get("parse_status") != "PARSED"
    ]
    rank_counts = Counter(
        str(row["selected_rank"]) for row in all_projections if row.get("selected_rank") is not None
    )
    decision_counts = Counter(str(row.get("decision")) for row in all_projections)
    generation_values = [
        float(row["generation_seconds"])
        for row in all_projections
        if row.get("generation_seconds") is not None
    ]
    rank_changed_windows = [row["window_id"] for row in valid_complete if row["rank_changed"]]
    decision_changed_windows = [
        row["window_id"] for row in valid_complete if row["decision_changed"]
    ]
    evidence_changed_windows = [
        row["window_id"] for row in valid_complete if row["evidence_changed"]
    ]
    pairwise = {
        f"{left}_vs_{right}": _pairwise_flip(grouped, left, right)
        for left, right in (("as_is", "reverse"), ("as_is", "shuffle"), ("reverse", "shuffle"))
        if left in modes and right in modes
    }
    if not valid_complete:
        conclusion_result = "INSUFFICIENT_COMPLETE_CALLS"
        conclusion_text = "No window has valid outputs for every requested presentation mode."
    elif decision_changed_windows or rank_changed_windows:
        conclusion_result = "ORDER_SENSITIVE"
        conclusion_text = (
            f"{len(decision_changed_windows)} complete windows changed decision and "
            f"{len(rank_changed_windows)} changed selected rank across presentation orders. "
            "Treat the Qwen output as order-sensitive until a stronger paired diagnostic is run."
        )
    elif evidence_changed_windows:
        conclusion_result = "EVIDENCE_TEXT_SENSITIVE_ONLY"
        conclusion_text = (
            f"No complete window changed decision or selected rank, but "
            f"{len(evidence_changed_windows)} changed evidence wording across orders."
        )
    else:
        conclusion_result = "NO_ORDER_SENSITIVITY_OBSERVED"
        conclusion_text = (
            "Complete windows kept the same selected rank, decision, and normalized evidence "
            "across all requested presentation orders."
        )
    return {
        "format": AGGREGATE_FORMAT,
        "authority": AUTHORITY,
        "status": "COMPLETE" if not row_errors else "PARTIAL",
        "conclusion": {"result": conclusion_result, "text": conclusion_text},
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "source": {
            "sidecar": str(sidecar) if isinstance(sidecar, (str, Path)) else None,
            "format": payload.get("format"),
            "window_count": len(per_window),
            "expected_modes": list(modes),
        },
        "metrics": {
            "raw_row_count": raw_row_count,
            "window_count": len(per_window),
            "complete_window_count": len(complete_windows),
            "valid_complete_window_count": len(valid_complete),
            "expected_calls": len(per_window) * len(modes),
            "valid_calls": len(all_projections) - len(parse_failures),
            "parse_failure_count": len(parse_failures),
            "parse_failure_rate": len(parse_failures) / len(all_projections)
            if all_projections
            else None,
            "rank_distribution": dict(sorted(rank_counts.items(), key=lambda item: item[0])),
            "decision_distribution": dict(
                sorted(decision_counts.items(), key=lambda item: item[0])
            ),
            "rank_invariant_windows": sum(bool(row["rank_invariant"]) for row in valid_complete),
            "rank_changed_windows": len(rank_changed_windows),
            "rank_changed_window_ids": rank_changed_windows,
            "decision_invariant_windows": sum(
                bool(row["decision_invariant"]) for row in valid_complete
            ),
            "decision_changed_windows": len(decision_changed_windows),
            "decision_changed_window_ids": decision_changed_windows,
            "evidence_invariant_windows": sum(
                bool(row["evidence_invariant"]) for row in valid_complete
            ),
            "evidence_changed_windows": len(evidence_changed_windows),
            "evidence_changed_window_ids": evidence_changed_windows,
            "pairwise": pairwise,
            "generation_seconds": {
                "count": len(generation_values),
                "total": round(sum(generation_values), 6) if generation_values else None,
                "mean": round(statistics.mean(generation_values), 6) if generation_values else None,
                "median": round(statistics.median(generation_values), 6)
                if generation_values
                else None,
            },
        },
        "row_errors": row_errors,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "epic_ontology_read": False,
            "mapper_read": False,
            "mage_read": False,
            "hash_or_digest_computed": False,
            "heldout_100_opened": False,
            "posthoc_only": True,
        },
        "per_window": per_window,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render compact metrics and per-window order comparisons."""

    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    conclusion = report.get("conclusion")
    conclusion = conclusion if isinstance(conclusion, Mapping) else {}
    pairwise = metrics.get("pairwise")
    pairwise = pairwise if isinstance(pairwise, Mapping) else {}
    lines = [
        "# WeMM candidate-order Qwen diagnostic aggregate",
        "",
        f"- **Status:** `{report.get('status')}`",
        f"- **Conclusion:** `{conclusion.get('result')}`",
        f"- **Official quality:** `{report.get('official_quality_status')}`",
        "",
        str(conclusion.get("text") or ""),
        "",
        "## Metrics",
        "",
        (
            f"- Windows: `{metrics.get('window_count', 0)}` "
            f"(complete `{metrics.get('complete_window_count', 0)}`, "
            f"valid complete `{metrics.get('valid_complete_window_count', 0)}`)."
        ),
        (
            f"- Calls: `{metrics.get('valid_calls', 0)}/"
            f"{metrics.get('raw_row_count', 0)}` valid; "
            f"parse failures `{metrics.get('parse_failure_count', 0)}`."
        ),
        f"- Rank distribution: `{metrics.get('rank_distribution', {})}`.",
        f"- Decision distribution: `{metrics.get('decision_distribution', {})}`.",
        f"- Complete-window rank changes: `{metrics.get('rank_changed_windows', 0)}`.",
        f"- Complete-window decision changes: `{metrics.get('decision_changed_windows', 0)}`.",
        (
            "- Complete-window evidence wording changes: "
            f"`{metrics.get('evidence_changed_windows', 0)}`."
        ),
        "",
        "## Pairwise differences",
        "",
        "| Pair | Complete windows | Different rank/decision | Rate |",
        "|---|---:|---:|---:|",
    ]
    for key, value in pairwise.items():
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "| "
            f"{key} | {value.get('compared_windows', 0)} | "
            f"{value.get('difference_windows', 0)} | "
            f"{value.get('difference_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Per-window mode consistency",
            "",
            (
                "| Window | Complete | Ranks | Decisions | Rank invariant | "
                "Decision invariant | Evidence changed |"
            ),
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for window in report.get("per_window", []):
        if not isinstance(window, Mapping):
            continue
        modes = window.get("modes")
        modes = modes if isinstance(modes, Mapping) else {}
        ranks = ", ".join(
            f"{mode}:r{value.get('selected_rank')}"
            for mode, value in modes.items()
            if isinstance(value, Mapping)
        )
        decisions = ", ".join(
            f"{mode}:{value.get('decision')}"
            for mode, value in modes.items()
            if isinstance(value, Mapping)
        )
        lines.append(
            "| "
            f"{window.get('window_id')} | {window.get('complete')} | "
            f"{ranks} | {decisions} | {window.get('rank_invariant')} | "
            f"{window.get('decision_invariant')} | {window.get('evidence_changed')} |"
        )
    lines.extend(
        [
            "",
            (
                "This is a post-hoc order-sensitivity diagnostic only. It does not "
                "establish annotation quality, gold agreement, or production "
                "eligibility; no model/media/gold/ontology/hash operation is "
                "performed during aggregation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AGGREGATE_FORMAT",
    "AUTHORITY",
    "DEFAULT_MODES",
    "ProductionWemmCandidateOrderQwenAggregateError",
    "aggregate_candidate_order_qwen_diagnostic",
    "load_json",
    "render_markdown",
]
