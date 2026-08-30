"""Lightweight WeMM Top-K -> Qwen structured candidate verifier.

This module is deliberately production-shadow only.  WeMM supplies the only
allowed action vocabulary; Qwen receives a complete bounded native video and
may verify, reject, split, or abstain, but may not invent labels.  The parser
is strict about provenance and boundaries while preserving raw model output
for review.  No gold data, ontology mutation, media decoding, or identity
digest is performed here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

VERSION: Final = "robata-production-wemm-qwen-candidate-verifier-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
OFFICIAL_GOLD_STATUS: Final = "NOT_ESTABLISHED"
PRODUCTION_ELIGIBLE: Final = False
_SUPPORT: Final = {"supported", "unsupported", "unclear"}
_DECISIONS: Final = {"accept", "edit", "split", "reject", "abstain"}
_BOUNDARY: Final = {"measured", "unclear", "not_measured"}
_FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")
_FIELD_STATUS: Final = {"measured", "unclear", "not_observable", "not_measured"}


class ProductionWemmQwenCandidateVerifierError(ValueError):
    """Invalid candidate/verifier payload."""


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmQwenCandidateVerifierError(f"{name} must be an object")
    return value


def _as_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmQwenCandidateVerifierError(f"{name} must be an array")
    return list(value)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _candidate_rank(value: Mapping[str, Any]) -> int | None:
    raw = value.get("rank")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return None
    return raw


def _candidate_key(value: Mapping[str, Any]) -> tuple[str, str]:
    return (_text(value.get("verb")).casefold(), _text(value.get("noun")).casefold())


def _candidate_label(value: Mapping[str, Any]) -> str:
    """Return the production label using compatibility spellings only."""

    for key in ("canonical_label", "label_text", "mapped_action", "raw_label"):
        label = _text(value.get(key))
        if label:
            return label
    verb, noun = _text(value.get("verb")), _text(value.get("noun"))
    return " ".join(x for x in (verb, noun) if x)


def _candidate_rows(candidates: Sequence[Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, item in enumerate(candidates):
        row = _as_mapping(item, f"candidates[{index}]")
        rank = _candidate_rank(row)
        if rank is None:
            raise ProductionWemmQwenCandidateVerifierError(
                f"candidate rank must be a positive integer at index {index}"
            )
        if rank in result:
            raise ProductionWemmQwenCandidateVerifierError(
                f"candidate rank is duplicated at index {index}"
            )
        result[rank] = row
    return result


def build_candidate_verifier_prompt(
    candidates: Sequence[Any],
    *,
    window_duration_seconds: float,
    window_id: str,
    verdict_scope: str = "selected_only",
    include_optional_fields: bool = False,
) -> str:
    """Render a deterministic prompt binding Qwen to WeMM candidates.

    ``include_optional_fields`` is an opt-in field-complete shadow profile.  It
    intentionally remains incompatible with ``all_candidates``: the latter is
    a rank-comparison diagnostic, while the former asks for one complete
    candidate record.  The default compact profile is unchanged.
    """

    duration = _number(window_duration_seconds)
    if duration is None or duration <= 0:
        raise ProductionWemmQwenCandidateVerifierError("window duration must be positive")
    if verdict_scope not in {"selected_only", "all_candidates", "pairwise"}:
        raise ProductionWemmQwenCandidateVerifierError(
            "verdict_scope must be selected_only, all_candidates, or pairwise"
        )
    if not isinstance(include_optional_fields, bool):
        raise ProductionWemmQwenCandidateVerifierError("include_optional_fields must be a boolean")
    if include_optional_fields and verdict_scope != "selected_only":
        raise ProductionWemmQwenCandidateVerifierError(
            "include_optional_fields requires selected_only verdict scope"
        )
    rows = []
    seen_ranks: set[int] = set()
    for item in candidates:
        row = _as_mapping(item, "candidate")
        rank = _candidate_rank(row)
        if rank is None:
            raise ProductionWemmQwenCandidateVerifierError("candidate rank must be positive")
        if rank in seen_ranks:
            raise ProductionWemmQwenCandidateVerifierError("candidate rank is duplicated")
        seen_ranks.add(rank)
        rows.append(
            {
                "rank": rank,
                "label_id": _text(row.get("label_id")),
                "verb": _text(row.get("verb")),
                "noun": _text(row.get("noun")),
                "label": _candidate_label(row),
            }
        )
    compact = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    if verdict_scope == "all_candidates":
        response_instructions = (
            "Set verdict_scope=all_candidates. Inspect every supplied candidate and emit "
            "exactly one candidate_verdicts item per supplied rank, with only rank and "
            "support=supported|unsupported|unclear. Then set selected_rank to the "
            "best-supported rank; do not default to rank 1. Use decision=accept only when "
            "the selected candidate is supported; otherwise use abstain. Do not emit "
            "unsupported alternatives outside the supplied Top-K."
        )
        schema = (
            '{"verdict_scope":"all_candidates","candidate_verdicts":'
            '[{"rank":1,"support":"unsupported"},{"rank":2,"support":"supported"}],'
            '"decision":"accept","selected_rank":2,"segments":[]}'
        )
        evidence_instruction = (
            "For this diagnostic all-candidates mode, omit evidence, fields, and "
            "boundaries to keep every rank. Return the compact object under 180 tokens."
        )
    elif verdict_scope == "pairwise":
        if len(rows) != 2:
            raise ProductionWemmQwenCandidateVerifierError(
                "pairwise verdict scope requires exactly two candidates"
            )
        pair_ranks = [rows[0]["rank"], rows[1]["rank"]]
        response_instructions = (
            f"Set verdict_scope=pairwise. Compare exactly the two supplied candidates "
            f"(ranks {pair_ranks[0]} and {pair_ranks[1]}) against the complete video. "
            "Do not use rank order as evidence and do not default to the first candidate. "
            "Emit exactly one candidate_verdicts item for the better-supported rank, "
            "with support=supported or unclear, selected_rank equal to that rank, and "
            "decision=accept only when supported."
        )
        schema = (
            f'{{"verdict_scope":"pairwise","candidate_verdicts":[{{"rank":{pair_ranks[1]},'
            '"support":"supported","evidence":["distinct hand motion"]}],'
            f'"decision":"accept","selected_rank":{pair_ranks[1]},"segments":[]}}'
        )
        evidence_instruction = (
            "Give one concise evidence string (at most 10 words); omit fields and "
            "boundaries in this diagnostic. Keep the response under 140 tokens."
        )
    else:
        if include_optional_fields:
            response_instructions = (
                "Set verdict_scope=selected_only. Inspect every supplied candidate and "
                "choose exactly one best-supported rank (it may be any supplied rank; do "
                "not default to rank 1). Emit exactly ONE candidate_verdicts item for that "
                "rank, even when support is unclear. The item MUST include fields for verb, "
                "noun, attributes, location, and hand; each field MUST be an object with an "
                "explicit status of measured, unclear, or not_observable (use value=null "
                "when not observable). Include a non-empty evidence array, a boundary object "
                "with status measured or unclear, and selected_rank equal to the emitted rank. "
                "Include a required numeric confidence from 0 to 1 (do not omit it, even when "
                "support is unclear). Never invent a label or rank outside WeMM Top-K."
            )
            # Use a value-neutral skeleton rather than a concrete action
            # example.  Qwen otherwise copies the example's verb/noun into the
            # annotation.  Nulls/unclear tokens describe shape only and must
            # be replaced with values grounded in the supplied video.
            schema = (
                '{"verdict_scope":"selected_only","candidate_verdicts":[{"rank":1,'
                '"support":"unclear","fields":{"verb":{"value":null,"status":"unclear"},'
                '"noun":{"value":null,"status":"unclear"},"attributes":{"value":null,'
                '"status":"not_observable"},"location":{"value":null,"status":"not_observable"},'
                '"hand":{"value":null,"status":"not_observable"}},'
                '"evidence":["<observable evidence>"],'
                '"confidence":0.0,"boundary":{"status":"unclear"}}],"decision":"abstain",'
                '"selected_rank":1,"segments":[{"candidate_rank":1,"boundary":'
                '{"status":"unclear"}}]}'
                "  (all values are placeholders; replace them, and use another supplied "
                "rank when appropriate)"
            )
            evidence_instruction = (
                "Keep one concise evidence string (at most 12 words) and include the "
                "required numeric confidence between 0 and 1, and keep the response under "
                "240 tokens."
            )
        else:
            response_instructions = (
                "Set verdict_scope=selected_only. Inspect every supplied candidate and do "
                "NOT default to rank 1; selected_rank may be any supplied rank. The Top-K "
                "remains available to you, but to avoid output truncation emit at most ONE "
                "candidate_verdicts item: the best-supported candidate, or the single most "
                "plausible candidate with support=unclear when none is supported. Do not emit "
                "unsupported alternatives."
            )
            schema = (
                '{"verdict_scope":"selected_only","candidate_verdicts":[{"rank":1,"support":"supported",'
                '"evidence":["hand lifts garment"],"boundary":{"status":"measured",'
                '"start_time_sec":0.2,"end_time_sec":1.1}}],"decision":"accept","selected_rank":1,'
                '"segments":[{"candidate_rank":1,"boundary":{"status":"measured",'
                '"start_time_sec":0.2,"end_time_sec":1.1}}]}'
            )
            evidence_instruction = (
                "Keep one evidence string to at most 8 words; omit optional fields and "
                "confidence unless strictly necessary. Include at most one segment and keep "
                "the response under 120 tokens."
            )
    candidate_contract = (
        "For every supplied candidate emit only its rank and support token; do not add "
        "evidence or fields."
        if verdict_scope == "all_candidates"
        else (
            "For the selected candidate emit support, explicit evidence, and a "
            "measured/unclear boundary. "
            + (
                "In the field-complete profile, also emit all five structured fields "
                "with explicit statuses."
                if include_optional_fields
                else "Structured fields are optional in the compact profile."
            )
        )
    )
    return (
        "You are a visual evidence verifier for production shadow annotation.\n"
        f"Window {window_id!r} is a complete bounded native video of {duration:g} seconds.\n"
        "Use the complete bounded native video, not a chunk or prose summary.\n"
        "Verify only the supplied WeMM Top-K candidates. You must never invent, translate, "
        "or replace a candidate label; never use EPIC labels or an outside ontology.\n"
        f"{candidate_contract} Support must be supported|unsupported|unclear; boundaries "
        "stay within the window.\n"
        "Return exactly one compact JSON object and nothing else. Use the key `support` "
        "(never `label`) for the verdict token; use `candidate_rank` in segments. "
        f"{response_instructions} "
        f"{evidence_instruction} "
        "The enum decision must be one of accept, edit, split, reject, abstain. "
        "If a boundary is unclear, emit status=unclear without times.\n"
        f"Schema example: {schema}\n"
        f"WeMM Top-K (JSON): {compact}"
    )


def _normalise_field(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        status = _text(value.get("status")).lower() or (
            "not_observable" if value.get("value") is None else "supported"
        )
        return {"value": value.get("value"), "status": status}
    if value is None:
        return {"value": None, "status": "not_observable"}
    return {"value": value, "status": "supported"}


def _normalise_boundary(value: object, duration: float, reasons: list[str]) -> dict[str, Any]:
    boundary = _as_mapping(value, "boundary") if isinstance(value, Mapping) else {}
    status = _text(boundary.get("status")).lower() or "not_measured"
    if status not in _BOUNDARY:
        status = "unclear"
        reasons.append("BOUNDARY_STATUS_INVALID")
    start = _number(boundary.get("start_time_sec", boundary.get("start_seconds")))
    end = _number(boundary.get("end_time_sec", boundary.get("end_seconds")))
    if status == "measured":
        if start is None or end is None or end <= start:
            reasons.append("BOUNDARY_INVALID")
            status = "not_measured"
            start = end = None
        elif start < 0 or end > duration:
            reasons.append("BOUNDARY_OUT_OF_WINDOW")
            # Preserve the fact that the model attempted a boundary, but do
            # not clamp it into validity.
            status = "not_measured"
            start = end = None
    else:
        start = end = None
    result: dict[str, Any] = {"status": status}
    if start is not None:
        result["start_time_sec"] = start
    if end is not None:
        result["end_time_sec"] = end
    return result


def _match_rank_by_label(value: object, allowed: Mapping[int, Mapping[str, Any]]) -> int | None:
    """Resolve a segment label only against the supplied Top-K rows."""

    text = _text(value).casefold()
    if not text:
        return None
    for rank, row in allowed.items():
        choices = {
            _candidate_label(row).casefold(),
            _text(row.get("verb")).casefold(),
            _text(row.get("noun")).casefold(),
            _text(row.get("label_id")).casefold(),
        }
        if text in choices:
            return rank
    return None


def parse_qwen_candidate_verification_output(
    raw_text: str,
    candidates: Sequence[Any],
    *,
    window_duration_seconds: float,
    require_optional_fields: bool = False,
) -> dict[str, Any]:
    """Parse and constrain one Qwen verification response.

    A candidate rank outside WeMM Top-K makes the whole response invalid.  A
    mismatched optional label only downgrades that verdict to ``unclear`` so
    the raw response remains useful for review.  When
    ``require_optional_fields`` is enabled, the selected-only response must
    contain exactly one verdict with explicit statuses for all production
    fields, non-empty evidence, and a boundary object.
    """

    duration = _number(window_duration_seconds)
    if duration is None or duration <= 0:
        raise ProductionWemmQwenCandidateVerifierError("window duration must be positive")
    if not isinstance(require_optional_fields, bool):
        raise ProductionWemmQwenCandidateVerifierError("require_optional_fields must be a boolean")
    allowed = _candidate_rows(candidates)
    try:
        decoded = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        return {
            "parse_status": "INVALID",
            "decision": "abstain",
            "candidate_verdicts": [],
            "segments": [],
            "errors": [f"invalid JSON: {exc}"],
            "warnings": [],
            "acceptance_reason_codes": ["PARSE_INVALID"],
            "accept_contract_ok": False,
        }
    if not isinstance(decoded, Mapping):
        return {
            "parse_status": "INVALID",
            "decision": "abstain",
            "candidate_verdicts": [],
            "segments": [],
            "errors": ["root must be an object"],
            "warnings": [],
            "acceptance_reason_codes": ["ROOT_NOT_OBJECT"],
            "accept_contract_ok": False,
        }
    errors: list[str] = []
    warnings: list[str] = []
    verdicts: list[dict[str, Any]] = []
    raw_verdicts = decoded.get("candidate_verdicts", [])
    if not isinstance(raw_verdicts, Sequence) or isinstance(raw_verdicts, (str, bytes, bytearray)):
        errors.append("candidate_verdicts must be an array")
        raw_verdicts = []
    for index, item in enumerate(raw_verdicts):
        if not isinstance(item, Mapping):
            errors.append(f"candidate_verdicts[{index}] must be an object")
            continue
        rank = _candidate_rank(item)
        if rank is None or rank not in allowed:
            errors.append("rank is not in WeMM Top-K")
            continue
        source = allowed[rank]
        reasons: list[str] = []
        support = _text(item.get("support")).lower()
        # A frequent near-schema response uses ``label`` for the verdict
        # token and omits ``support``.  Accept only the three contract tokens;
        # arbitrary prose is never promoted to a verdict.
        label_alias = _text(item.get("label")).lower()
        if not support and label_alias in _SUPPORT:
            support = label_alias
            warnings.append("MODEL_SCHEMA_ALIAS_LABEL_AS_SUPPORT")
        support = support or "unclear"
        if support not in _SUPPORT:
            reasons.append("SUPPORT_INVALID")
            support = "unclear"
        supplied_label = _text(item.get("label")).casefold()
        source_label = _candidate_label(source).casefold()
        # A support-token alias is not an action label and must not trigger a
        # false candidate mismatch.
        if supplied_label in _SUPPORT:
            supplied_label = ""
        if supplied_label and source_label and supplied_label != source_label:
            reasons.append("CANDIDATE_LABEL_MISMATCH")
            support = "unclear"
        fields: dict[str, dict[str, Any]] = {}
        raw_fields = item.get("fields", item.get("structured_fields"))
        if isinstance(raw_fields, Mapping):
            for field in _FIELDS:
                if field in raw_fields:
                    fields[field] = _normalise_field(raw_fields[field])
        boundary_payload = item.get("boundary")
        if boundary_payload is None and ("start" in item or "end" in item):
            boundary_payload = {
                "status": "measured",
                "start_time_sec": item.get("start"),
                "end_time_sec": item.get("end"),
            }
        boundary = _normalise_boundary(boundary_payload, duration, reasons)
        evidence = item.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence] if evidence.strip() else []
        elif not isinstance(evidence, Sequence):
            evidence = []
        confidence = _number(item.get("confidence"))
        verdict: dict[str, Any] = {
            "rank": rank,
            "label": _candidate_label(source),
            "support": support,
            "conflict": bool(item.get("conflict", False)),
            "reason_codes": reasons,
            "fields": fields,
            "boundary": boundary,
            "evidence": [str(x) for x in evidence if str(x).strip()],
            "evidence_status": (
                "measured" if any(str(x).strip() for x in evidence) else "not_measured"
            ),
        }
        if "conflict_reasons" in item:
            verdict["conflict_reasons"] = [
                str(x) for x in _as_list(item["conflict_reasons"], "conflict_reasons")
            ]
        if confidence is not None:
            verdict["confidence"] = confidence
        verdicts.append(verdict)
    if errors:
        return {
            "parse_status": "INVALID",
            "decision": "abstain",
            "candidate_verdicts": verdicts,
            "segments": [],
            "errors": errors,
            "warnings": warnings,
            "acceptance_reason_codes": ["PARSE_INVALID"],
            "accept_contract_ok": False,
        }
    emitted_ranks = {v.get("rank") for v in verdicts}
    verdict_scope = _text(decoded.get("verdict_scope")).lower()
    if verdict_scope not in {"", "all_candidates", "selected_only", "pairwise"}:
        warnings.append("VERDICT_SCOPE_INVALID")
        verdict_scope = ""
    missing_ranks = sorted(set(allowed) - emitted_ranks)
    # The compact production prompt intentionally permits a selected-only
    # response.  Keep the warning for legacy/all-candidate responses so a
    # caller can still distinguish a truncated answer from an explicit
    # one-candidate verification.
    if missing_ranks and verdict_scope not in {"selected_only", "pairwise"}:
        warnings.append("CANDIDATE_VERDICTS_INCOMPLETE")
    decision = _text(decoded.get("decision")).lower() or "abstain"
    if decision not in _DECISIONS:
        # Keep a prose decision visible but normalize the narrow, safe case:
        # a valid selected rank with an explicitly supported verdict.  This is
        # schema compatibility, not semantic label generation.
        selected_hint = decoded.get("selected_rank")
        selected_verdict = next(
            (v for v in verdicts if isinstance(v, Mapping) and v.get("rank") == selected_hint),
            None,
        )
        if (
            isinstance(selected_hint, int)
            and isinstance(selected_verdict, Mapping)
            and selected_verdict.get("support") == "supported"
        ):
            decision = "accept"
            warnings.append("MODEL_SCHEMA_ALIAS_PROSE_DECISION")
        else:
            warnings.append("DECISION_INVALID")
            decision = "abstain"
    selected = decoded.get("selected_rank")
    selected_rank = (
        selected if isinstance(selected, int) and not isinstance(selected, bool) else None
    )
    if selected_rank is not None and selected_rank not in allowed:
        # Qwen occasionally emits ``selected_rank: 0`` in all-candidates
        # diagnostics when it finds no supported candidate.  Zero is not a
        # candidate rank, but for an explicit abstention with every supplied
        # candidate marked unclear/unsupported it is an unambiguous
        # no-selection sentinel.  Normalize that narrow schema alias instead
        # of discarding an otherwise useful complete diagnostic.  Keep the
        # strict error for accept/other decisions and for selected-only mode.
        no_supported_candidate = all(verdict.get("support") != "supported" for verdict in verdicts)
        # A pairwise prompt has exactly two supplied candidates.  Unlike the
        # all-candidates route, do not normalize an incomplete pairwise
        # response: both ranks must have an explicit non-supported verdict
        # before the model's ``0`` sentinel can mean "no selection".
        complete_pairwise_no_selection = (
            verdict_scope == "pairwise"
            and len(allowed) == 2
            and len(verdicts) == len(allowed)
            and {verdict.get("rank") for verdict in verdicts} == set(allowed)
            and no_supported_candidate
        )
        allow_zero_no_selection = (
            verdict_scope == "all_candidates" or complete_pairwise_no_selection
        )
        if (
            allow_zero_no_selection
            and decision == "abstain"
            and selected_rank == 0
            and no_supported_candidate
        ):
            warnings.append("MODEL_SCHEMA_ALIAS_NO_SELECTION")
            selected_rank = None
        else:
            errors.append("selected_rank is not in WeMM Top-K")
            selected_rank = None
    segments: list[dict[str, Any]] = []
    raw_segments = decoded.get("segments", [])
    if isinstance(raw_segments, Sequence) and not isinstance(raw_segments, (str, bytes, bytearray)):
        for item in raw_segments:
            if not isinstance(item, Mapping):
                warnings.append("SEGMENT_INVALID")
                continue
            rank = item.get("candidate_rank", item.get("rank"))
            if not isinstance(rank, int) or isinstance(rank, bool) or rank not in allowed:
                rank = _match_rank_by_label(item.get("label"), allowed)
            if rank is None or rank not in allowed:
                warnings.append("SEGMENT_CANDIDATE_NOT_IN_TOP_K")
                continue
            segment_reasons: list[str] = []
            boundary_value = item.get("boundary")
            if boundary_value is None and ("start" in item or "end" in item):
                boundary_value = {
                    "status": "measured",
                    "start_time_sec": item.get("start"),
                    "end_time_sec": item.get("end"),
                }
                warnings.append("MODEL_SCHEMA_ALIAS_SEGMENT_BOUNDARY")
            boundary = _normalise_boundary(boundary_value, duration, segment_reasons)
            segments.append(
                {"candidate_rank": rank, "boundary": boundary, "reason_codes": segment_reasons}
            )
            # If the model put the boundary on a compact segment object, carry
            # it to the matching verdict for downstream acceptance checks.
            for verdict in verdicts:
                if (
                    verdict.get("rank") == rank
                    and verdict.get("boundary", {}).get("status") != "measured"
                ):
                    verdict["boundary"] = boundary
    if require_optional_fields:
        # This profile is intentionally strict and selected-only.  Keep all
        # model material in ``candidate_verdicts`` while making a malformed
        # field-complete response ineligible for downstream acceptance.
        if verdict_scope != "selected_only":
            errors.append("OPTIONAL_PROFILE_REQUIRES_SELECTED_ONLY")
        if len(verdicts) != 1:
            errors.append("OPTIONAL_PROFILE_REQUIRES_ONE_VERDICT")
        if selected_rank is None:
            errors.append("OPTIONAL_PROFILE_SELECTED_RANK_MISSING")
        elif len(verdicts) == 1 and verdicts[0].get("rank") != selected_rank:
            errors.append("OPTIONAL_PROFILE_SELECTED_RANK_MISMATCH")
        raw_selected = None
        if isinstance(raw_verdicts, Sequence) and not isinstance(
            raw_verdicts, (str, bytes, bytearray)
        ):
            for raw_item in raw_verdicts:
                if isinstance(raw_item, Mapping) and raw_item.get("rank") == selected_rank:
                    raw_selected = raw_item
                    break
        if raw_selected is None and len(raw_verdicts) == 1:
            candidate = raw_verdicts[0]
            raw_selected = candidate if isinstance(candidate, Mapping) else None
        raw_fields = (
            raw_selected.get("fields", raw_selected.get("structured_fields"))
            if isinstance(raw_selected, Mapping)
            else None
        )
        if not isinstance(raw_fields, Mapping):
            errors.append("OPTIONAL_PROFILE_FIELDS_MISSING")
        else:
            for field in _FIELDS:
                payload = raw_fields.get(field)
                if not isinstance(payload, Mapping):
                    errors.append(f"OPTIONAL_PROFILE_FIELD_INVALID:{field}")
                    continue
                if "value" not in payload:
                    errors.append(f"OPTIONAL_PROFILE_FIELD_VALUE_MISSING:{field}")
                status = _text(payload.get("status")).lower()
                if status not in _FIELD_STATUS:
                    errors.append(f"OPTIONAL_PROFILE_FIELD_STATUS_INVALID:{field}")
            # The selected candidate is the only permitted ontology value.
            # Reject a field-complete response that copies a different action
            # from a schema example (a common failure mode for compact JSON
            # prompts) instead of silently accepting the wrong verb/noun.
            if selected_rank is not None and selected_rank in allowed:
                source = allowed[selected_rank]
                for field in ("verb", "noun"):
                    payload = raw_fields.get(field)
                    if not isinstance(payload, Mapping):
                        continue
                    status = _text(payload.get("status")).lower()
                    value = _text(payload.get("value")).casefold()
                    expected = _text(source.get(field)).casefold()
                    if status == "measured" and value and expected and value != expected:
                        errors.append(f"OPTIONAL_PROFILE_{field.upper()}_MISMATCH")
        evidence_values = verdicts[0].get("evidence", []) if len(verdicts) == 1 else []
        if (
            not isinstance(evidence_values, Sequence)
            or isinstance(evidence_values, (str, bytes, bytearray))
            or not any(str(value).strip() for value in evidence_values)
        ):
            errors.append("OPTIONAL_PROFILE_EVIDENCE_MISSING")
        elif any(
            "<" in str(value)
            or ">" in str(value)
            or str(value).strip().casefold() in {"observable evidence", "evidence"}
            for value in evidence_values
        ):
            errors.append("OPTIONAL_PROFILE_EVIDENCE_PLACEHOLDER")
        # Confidence is part of the field-complete production contract.  Keep
        # the compact profile permissive, but reject a field-complete record
        # that omits it or emits a non-finite/out-of-range value.
        raw_confidence = (
            raw_selected.get("confidence") if isinstance(raw_selected, Mapping) else None
        )
        confidence = _number(raw_confidence)
        if confidence is None or confidence < 0.0 or confidence > 1.0:
            errors.append("OPTIONAL_PROFILE_CONFIDENCE_INVALID")
        if not isinstance(raw_selected, Mapping) or not isinstance(
            raw_selected.get("boundary"), Mapping
        ):
            errors.append("OPTIONAL_PROFILE_BOUNDARY_MISSING")
        elif verdicts and verdicts[0].get("boundary", {}).get("status") not in {
            "measured",
            "unclear",
        }:
            errors.append("OPTIONAL_PROFILE_BOUNDARY_INVALID")
        if len(segments) != 1:
            errors.append("OPTIONAL_PROFILE_REQUIRES_ONE_SEGMENT")
        elif selected_rank is not None and segments[0].get("candidate_rank") != selected_rank:
            errors.append("OPTIONAL_PROFILE_SEGMENT_RANK_MISMATCH")
    if errors:
        return {
            "parse_status": "INVALID",
            "decision": "abstain",
            "candidate_verdicts": verdicts,
            "segments": segments,
            "errors": errors,
            "warnings": warnings,
            "acceptance_reason_codes": ["PARSE_INVALID"],
            "accept_contract_ok": False,
        }
    selected_verdict = next(
        (
            verdict
            for verdict in verdicts
            if isinstance(verdict, Mapping) and verdict.get("rank") == selected_rank
        ),
        None,
    )
    acceptance_reason_codes: list[str] = []
    if decision == "accept":
        if selected_verdict is None or selected_verdict.get("support") != "supported":
            acceptance_reason_codes.append("ACCEPT_REQUIRES_SUPPORTED_SELECTED_CANDIDATE")
        if selected_verdict is not None and not selected_verdict.get("evidence"):
            acceptance_reason_codes.append("ACCEPT_REQUIRES_EVIDENCE")
    # ``accept_contract_ok`` is intentionally observational in the parser:
    # legacy compact sidecars retain their original decision, while strict
    # downstream gates can fail closed using these explicit diagnostics.
    accept_contract_ok = not errors and decision == "accept" and not acceptance_reason_codes
    result: dict[str, Any] = {
        "parse_status": "INVALID" if errors else "PARSED",
        "decision": "abstain" if errors else decision,
        "candidate_verdicts": verdicts,
        "segments": segments,
        "errors": errors,
        "warnings": warnings,
        "acceptance_reason_codes": acceptance_reason_codes,
        "accept_contract_ok": accept_contract_ok,
    }
    if verdict_scope:
        result["verdict_scope"] = verdict_scope
    if selected_rank is not None:
        result["selected_rank"] = selected_rank
    return result


def _window_duration(window: Mapping[str, Any]) -> float:
    interval = window.get("source_interval", window.get("interval", [0.0, 0.0]))
    if (
        isinstance(interval, Sequence)
        and not isinstance(interval, (str, bytes, bytearray))
        and len(interval) >= 2
    ):
        start, end = _number(interval[0]), _number(interval[1])
        if start is not None and end is not None and end > start:
            return end - start
    start, end = _number(window.get("start_seconds")), _number(window.get("end_seconds"))
    if start is not None and end is not None and end > start:
        return end - start
    return 0.0


def _constraint_diagnostics(
    parsed: object,
    candidates: Sequence[Any],
    *,
    duration: float,
) -> list[str]:
    """Audit a recorded parsed response without trusting its provenance.

    Native runner rows normally carry a parser result, but a join can also be
    replayed from an externally supplied ``parsed_verification`` object.  The
    audit is intentionally observational: it reports rank/boundary violations
    in the diagnostics block while leaving the historical decision semantics
    unchanged.  This makes stale or hand-edited sidecars visible without
    silently rewriting old results.
    """

    if not isinstance(parsed, Mapping):
        return ["PARSED_VERIFICATION_NOT_OBJECT"]
    try:
        allowed = set(_candidate_rows(candidates))
    except ProductionWemmQwenCandidateVerifierError:
        return ["CANDIDATE_RANK_SET_INVALID"]
    violations: list[str] = []
    selected = parsed.get("selected_rank")
    if selected is not None and (
        isinstance(selected, bool) or not isinstance(selected, int) or selected not in allowed
    ):
        violations.append("SELECTED_RANK_OUT_OF_TOP_K")
    raw_verdicts = parsed.get("candidate_verdicts", [])
    if isinstance(raw_verdicts, Sequence) and not isinstance(raw_verdicts, (str, bytes, bytearray)):
        for item in raw_verdicts:
            if not isinstance(item, Mapping):
                violations.append("VERDICT_ROW_NOT_OBJECT")
                continue
            rank = item.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank not in allowed:
                violations.append("VERDICT_RANK_OUT_OF_TOP_K")
            boundary = item.get("boundary")
            if (
                isinstance(boundary, Mapping)
                and _text(boundary.get("status")).lower() == "measured"
            ):
                start = _number(boundary.get("start_time_sec", boundary.get("start_seconds")))
                end = _number(boundary.get("end_time_sec", boundary.get("end_seconds")))
                if start is None or end is None or start < 0 or end > duration or end <= start:
                    violations.append("VERDICT_BOUNDARY_OUT_OF_WINDOW")
    else:
        violations.append("VERDICT_ROWS_NOT_ARRAY")
    raw_segments = parsed.get("segments", [])
    if isinstance(raw_segments, Sequence) and not isinstance(raw_segments, (str, bytes, bytearray)):
        for item in raw_segments:
            if not isinstance(item, Mapping):
                violations.append("SEGMENT_ROW_NOT_OBJECT")
                continue
            rank = item.get("candidate_rank", item.get("rank"))
            if isinstance(rank, bool) or not isinstance(rank, int) or rank not in allowed:
                violations.append("SEGMENT_RANK_OUT_OF_TOP_K")
            boundary = item.get("boundary")
            if (
                isinstance(boundary, Mapping)
                and _text(boundary.get("status")).lower() == "measured"
            ):
                start = _number(boundary.get("start_time_sec", boundary.get("start_seconds")))
                end = _number(boundary.get("end_time_sec", boundary.get("end_seconds")))
                if start is None or end is None or start < 0 or end > duration or end <= start:
                    violations.append("SEGMENT_BOUNDARY_OUT_OF_WINDOW")
    return sorted(set(violations))


def diagnose_candidate_bound_consensus_gate(
    joined_report: Mapping[str, Any],
    *,
    expected_camera_count: int = 6,
    min_camera_coverage: float = 0.5,
    min_consensus_fraction: float = 0.5,
    min_retrieval_margin: float = 0.01,
    require_evidence_for_accept: bool = False,
) -> dict[str, Any]:
    """Compute a post-hoc candidate-bound camera consensus gate.

    This is deliberately a *diagnostic* projection over an existing join
    report.  It does not alter the verifier's recorded ``decision`` and does
    not invoke a model or decode media.  A camera vote is eligible only when
    the recorded row is native/complete, parsed, accepted, and its selected
    rank is both present in the WeMM Top-K and explicitly supported.  The gate
    then requires enough eligible cameras, a strict winning-rank majority, and
    a minimum WeMM rank-1/rank-2 score margin.

    The thresholds are intentionally exposed in the output and are not a
    production policy.  They provide a small, reproducible way to diagnose
    whether over-acceptance is caused by sparse camera coverage, disagreement,
    or an intrinsically low retrieval margin.
    """

    if (
        not isinstance(expected_camera_count, int)
        or isinstance(expected_camera_count, bool)
        or expected_camera_count <= 0
    ):
        raise ProductionWemmQwenCandidateVerifierError(
            "expected_camera_count must be a positive integer"
        )
    for name, value in (
        ("min_camera_coverage", min_camera_coverage),
        ("min_consensus_fraction", min_consensus_fraction),
    ):
        numeric = _number(value)
        if numeric is None or not 0.0 <= numeric <= 1.0:
            raise ProductionWemmQwenCandidateVerifierError(f"{name} must be between 0 and 1")
    margin_threshold = _number(min_retrieval_margin)
    if margin_threshold is None or margin_threshold < 0.0:
        raise ProductionWemmQwenCandidateVerifierError(
            "min_retrieval_margin must be finite and non-negative"
        )
    if not isinstance(require_evidence_for_accept, bool):
        raise ProductionWemmQwenCandidateVerifierError(
            "require_evidence_for_accept must be boolean"
        )
    windows = joined_report.get("windows", [])
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
        raise ProductionWemmQwenCandidateVerifierError("joined_report.windows must be an array")

    gate_rows: list[dict[str, Any]] = []
    for index, raw_window in enumerate(windows):
        window = _as_mapping(raw_window, f"joined_report.windows[{index}]")
        window_id = _text(window.get("window_id")) or f"window-{index}"
        raw_top_k = window.get("top_k", [])
        top_k = (
            list(raw_top_k)
            if isinstance(raw_top_k, Sequence)
            and not isinstance(raw_top_k, (str, bytes, bytearray))
            else []
        )
        allowed_ranks: set[int] = set()
        score_by_rank: dict[int, float] = {}
        for candidate in top_k:
            if not isinstance(candidate, Mapping):
                continue
            rank = _candidate_rank(candidate)
            if rank is None:
                continue
            allowed_ranks.add(rank)
            score = _number(candidate.get("score"))
            if score is not None:
                score_by_rank[rank] = score
        margin = (
            score_by_rank[1] - score_by_rank[2]
            if 1 in score_by_rank and 2 in score_by_rank
            else None
        )
        raw_cameras = window.get("camera_reports", [])
        cameras = (
            list(raw_cameras)
            if isinstance(raw_cameras, Sequence)
            and not isinstance(raw_cameras, (str, bytes, bytearray))
            else []
        )
        votes: dict[int, int] = {}
        eligible_camera_ids: list[str] = []
        reasons: list[str] = []
        candidate_bound_failures = 0
        evidence_failures = 0
        for camera_index, raw_camera in enumerate(cameras):
            camera = _as_mapping(raw_camera, f"{window_id}.camera_reports[{camera_index}]")
            camera_id = _text(camera.get("camera_id")) or f"camera-{camera_index}"
            parsed = camera.get("parsed_verification", {})
            parsed_map = parsed if isinstance(parsed, Mapping) else {}
            selected = parsed_map.get("selected_rank")
            selected_verdict = None
            verdicts = parsed_map.get("candidate_verdicts", [])
            if isinstance(verdicts, Sequence) and not isinstance(verdicts, (str, bytes, bytearray)):
                selected_verdict = next(
                    (
                        verdict
                        for verdict in verdicts
                        if isinstance(verdict, Mapping) and verdict.get("rank") == selected
                    ),
                    None,
                )
            rank_bound = (
                isinstance(selected, int)
                and not isinstance(selected, bool)
                and selected in allowed_ranks
            )
            supported = (
                isinstance(selected_verdict, Mapping)
                and selected_verdict.get("support") == "supported"
            )
            native_complete = camera.get("native_video_complete") is True
            parsed_ok = parsed_map.get("parse_status") == "PARSED"
            accepted = _text(camera.get("decision")).lower() == "accept"
            if not rank_bound:
                candidate_bound_failures += 1
            evidence_ok = True
            if require_evidence_for_accept:
                selected_verdict = next(
                    (
                        value
                        for value in parsed_map.get("candidate_verdicts", [])
                        if isinstance(value, Mapping) and value.get("rank") == selected
                    ),
                    None,
                )
                evidence_values = (
                    selected_verdict.get("evidence", [])
                    if isinstance(selected_verdict, Mapping)
                    else []
                )
                evidence_ok = (
                    isinstance(evidence_values, Sequence)
                    and not isinstance(evidence_values, (str, bytes, bytearray))
                    and any(str(value).strip() for value in evidence_values)
                )
                if not evidence_ok:
                    evidence_failures += 1
            if (
                native_complete
                and parsed_ok
                and accepted
                and rank_bound
                and supported
                and evidence_ok
                and isinstance(selected, int)
                and not isinstance(selected, bool)
            ):
                selected_rank = selected
                votes[selected_rank] = votes.get(selected_rank, 0) + 1
                eligible_camera_ids.append(camera_id)
        observed_camera_count = len(cameras)
        eligible_vote_count = sum(votes.values())
        winning_rank: int | None = None
        winning_votes = 0
        if votes:
            winning_rank, winning_votes = max(votes.items(), key=lambda item: (item[1], -item[0]))
        camera_coverage = eligible_vote_count / float(expected_camera_count)
        consensus_fraction = (
            winning_votes / float(eligible_vote_count) if eligible_vote_count else 0.0
        )
        if not cameras:
            reasons.append("NO_CAMERA_ROWS")
        if eligible_vote_count == 0:
            reasons.append("NO_ELIGIBLE_CAMERA_VOTES")
        if camera_coverage < float(min_camera_coverage):
            reasons.append("INSUFFICIENT_CAMERA_COVERAGE")
        if eligible_vote_count and not (winning_votes * 2 > eligible_vote_count):
            reasons.append("NO_STRICT_CAMERA_CONSENSUS")
        if eligible_vote_count and consensus_fraction < float(min_consensus_fraction):
            reasons.append("LOW_CAMERA_CONSENSUS")
        if margin is None:
            reasons.append("RETRIEVAL_MARGIN_UNAVAILABLE")
        elif margin < margin_threshold:
            reasons.append("LOW_RETRIEVAL_MARGIN")
        if candidate_bound_failures:
            reasons.append("CANDIDATE_BOUND_VIOLATION")
        if evidence_failures:
            reasons.append("CANDIDATE_EVIDENCE_MISSING")
        gate_decision = "accept" if not reasons else "abstain"
        winning_label = None
        if winning_rank is not None:
            for candidate in top_k:
                if isinstance(candidate, Mapping) and _candidate_rank(candidate) == winning_rank:
                    winning_label = _candidate_label(candidate)
                    break
        gate_rows.append(
            {
                "window_id": window_id,
                "recorded_decision": _text(window.get("decision")).lower() or "abstain",
                "gate_decision": gate_decision,
                "gate_reason_codes": sorted(set(reasons)),
                "observed_camera_count": observed_camera_count,
                "eligible_camera_count": eligible_vote_count,
                "eligible_camera_ids": eligible_camera_ids,
                "expected_camera_count": expected_camera_count,
                "camera_coverage": round(camera_coverage, 6),
                "winning_rank": winning_rank,
                "winning_label": winning_label,
                "winning_votes": winning_votes,
                "consensus_fraction": round(consensus_fraction, 6),
                "retrieval_margin_top1_top2": None if margin is None else round(margin, 6),
                "candidate_bound_failure_count": candidate_bound_failures,
                "candidate_evidence_failure_count": evidence_failures,
                "require_evidence_for_accept": require_evidence_for_accept,
            }
        )
    accepted_count = sum(row["gate_decision"] == "accept" for row in gate_rows)
    return {
        "format": "robata-production-wemm-qwen-candidate-consensus-gate-v1",
        "authority": AUTHORITY,
        "official_quality_status": "NOT_MEASURED",
        "accuracy_status": "NOT_MEASURED",
        "production_eligible": False,
        "quality_claim": False,
        "diagnostic_only": True,
        "policy": {
            "expected_camera_count": expected_camera_count,
            "min_camera_coverage": float(min_camera_coverage),
            "min_consensus_fraction": float(min_consensus_fraction),
            "min_retrieval_margin": float(margin_threshold),
            "strict_majority": True,
            "decision_mutated": False,
            "require_evidence_for_accept": require_evidence_for_accept,
        },
        "summary": {
            "window_count": len(gate_rows),
            "gate_accept_count": accepted_count,
            "gate_abstain_count": len(gate_rows) - accepted_count,
            "recorded_accept_count": sum(row["recorded_decision"] == "accept" for row in gate_rows),
        },
        "windows": gate_rows,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_included": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "hash_or_digest_computed": False,
        },
    }


def render_candidate_bound_consensus_gate_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact, non-gold consensus-gate diagnostic report."""

    policy = report.get("policy", {})
    summary = report.get("summary", {})
    lines = [
        "# Candidate-bound six-camera consensus gate",
        "",
        f"Status: `diagnostic_only`; accuracy: "
        f"`{_text(report.get('accuracy_status')) or 'NOT_MEASURED'}`; "
        "production eligible: `false`.",
        f"Policy: expected cameras `{policy.get('expected_camera_count', 6)}`, coverage >= "
        f"`{policy.get('min_camera_coverage', 0.5)}`, strict majority, margin >= "
        f"`{policy.get('min_retrieval_margin', 0.01)}`, evidence required for accept="
        f"`{policy.get('require_evidence_for_accept', False)}`.",
        f"Summary: `{summary.get('gate_accept_count', 0)}` gate accepts / "
        f"`{summary.get('gate_abstain_count', 0)}` gate abstentions; recorded decisions "
        "are unchanged.",
        "",
        "| Window | Recorded | Gate | Cameras eligible/observed | Agreement | Margin | Reasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    windows = report.get("windows", [])
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes, bytearray)):
        for row in windows:
            if not isinstance(row, Mapping):
                continue
            reasons = ", ".join(str(x) for x in row.get("gate_reason_codes", [])) or "-"
            margin = (
                row.get("retrieval_margin_top1_top2")
                if row.get("retrieval_margin_top1_top2") is not None
                else "-"
            )
            lines.append(
                f"| {_text(row.get('window_id'))} | {_text(row.get('recorded_decision'))} | "
                f"{_text(row.get('gate_decision'))} | {row.get('eligible_camera_count', 0)}/"
                f"{row.get('observed_camera_count', 0)} | "
                f"{row.get('consensus_fraction', 0.0):.3f} | "
                f"{margin} | "
                f"{reasons} |"
            )
    return "\n".join(lines) + "\n"


def verify_wemm_qwen_candidate_sidecars(
    candidate_pack: Mapping[str, Any],
    qwen_sidecar: Mapping[str, Any],
    *,
    require_evidence_for_accept: bool = False,
) -> dict[str, Any]:
    """Join recorded WeMM candidates and Qwen verifier responses.

    ``require_evidence_for_accept`` is an opt-in strict shadow gate.  The
    default preserves historical compact-sidecar semantics, while strict mode
    downgrades an otherwise supported Qwen accept that lacks a non-empty
    candidate-bound evidence field to ``abstain``.
    """

    if not isinstance(require_evidence_for_accept, bool):
        raise ProductionWemmQwenCandidateVerifierError(
            "require_evidence_for_accept must be boolean"
        )

    pack_windows = _as_list(candidate_pack.get("windows", []), "candidate_pack.windows")
    qwen_windows = _as_list(qwen_sidecar.get("windows", []), "qwen_sidecar.windows")
    # Keep join/provenance diagnostics explicit.  A window may legitimately
    # have multiple Qwen rows (one per camera), so duplicate ``window_id``
    # values are informational; repeated ``(window_id, camera_id)`` pairs are
    # the actionable duplicate condition.
    candidate_ids: list[str] = []
    for index, raw_window in enumerate(pack_windows):
        row = _as_mapping(raw_window, f"candidate_pack.windows[{index}]")
        candidate_ids.append(_text(row.get("window_id")) or f"window-{index}")
    qwen_ids: list[str] = []
    qwen_camera_keys: list[tuple[str, str]] = []
    for index, raw_row in enumerate(qwen_windows):
        row = _as_mapping(raw_row, f"qwen window[{index}]")
        qwen_id = _text(row.get("window_id"))
        qwen_ids.append(qwen_id)
        qwen_camera_keys.append((qwen_id, _text(row.get("camera_id"))))
    candidate_id_set = set(candidate_ids)
    qwen_id_set = {value for value in qwen_ids if value}
    qwen_id_counts: dict[str, int] = {}
    for value in qwen_ids:
        if value:
            qwen_id_counts[value] = qwen_id_counts.get(value, 0) + 1
    camera_key_counts: dict[tuple[str, str], int] = {}
    for key in qwen_camera_keys:
        camera_key_counts[key] = camera_key_counts.get(key, 0) + 1
    duplicate_camera_rows = [
        {"window_id": window_id, "camera_id": camera_id, "count": count}
        for (window_id, camera_id), count in sorted(camera_key_counts.items())
        if window_id and camera_id and count > 1
    ]
    qwen_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for raw_row in qwen_windows:
        row = _as_mapping(raw_row, "qwen window")
        qwen_by_id.setdefault(_text(row.get("window_id")), []).append(row)
    reports: list[dict[str, Any]] = []
    validation_rows: list[tuple[str, bool, Mapping[str, Any], list[str]]] = []
    for index, raw_window in enumerate(pack_windows):
        window = _as_mapping(raw_window, f"candidate_pack.windows[{index}]")
        window_id = _text(window.get("window_id")) or f"window-{index}"
        context = window.get("model_context", {})
        context_map = (
            _as_mapping(context, f"{window_id}.model_context")
            if isinstance(context, Mapping)
            else {}
        )
        wemm = context_map.get("wemm", {})
        wemm_map = _as_mapping(wemm, f"{window_id}.wemm") if isinstance(wemm, Mapping) else {}
        candidates = wemm_map.get("top_k", wemm_map.get("predictions", []))
        candidates_list = _as_list(candidates, f"{window_id}.wemm.top_k")
        qwen_rows = qwen_by_id.get(window_id, [])
        duration = _window_duration(window)
        if duration <= 0:
            duration = 1.0
        camera_reports: list[dict[str, Any]] = []
        for qwen in qwen_rows:
            native_ok = bool(
                qwen.get("input_mode") == "native_video"
                and qwen.get("native_video_complete") is True
            )
            parsed = qwen.get("parsed_verification")
            if not isinstance(parsed, Mapping):
                raw = _text(qwen.get("raw_text"))
                parsed = parse_qwen_candidate_verification_output(
                    raw, candidates_list, window_duration_seconds=duration
                )
            else:
                parsed = dict(parsed)
            constraint_codes = _constraint_diagnostics(parsed, candidates_list, duration=duration)
            validation_rows.append(
                (
                    f"{window_id}:{_text(qwen.get('camera_id')) or 'unknown'}",
                    bool(
                        qwen.get("input_mode") == "native_video"
                        and qwen.get("native_video_complete") is True
                    ),
                    parsed,
                    constraint_codes,
                )
            )
            camera_decision = _text(parsed.get("decision")).lower() or "abstain"
            camera_reasons = (
                [str(x) for x in parsed.get("errors", [])]
                if isinstance(parsed.get("errors"), Sequence)
                else []
            )
            if not native_ok:
                camera_reasons.append("NATIVE_VIDEO_NOT_COMPLETE")
                camera_decision = "abstain"
            if parsed.get("parse_status") != "PARSED":
                camera_reasons.append("QWEN_VERIFICATION_PARSE_INVALID")
                camera_decision = "abstain"
            if camera_decision == "accept":
                selected = parsed.get("selected_rank")
                raw_verdicts = parsed.get("candidate_verdicts", [])
                supported = (
                    [
                        value
                        for value in raw_verdicts
                        if isinstance(value, Mapping) and value.get("support") == "supported"
                    ]
                    if isinstance(raw_verdicts, Sequence)
                    and not isinstance(raw_verdicts, (str, bytes, bytearray))
                    else []
                )
                if not isinstance(selected, int) or not any(
                    v.get("rank") == selected for v in supported
                ):
                    camera_reasons.append("ACCEPT_REQUIRES_SUPPORTED_SELECTED_CANDIDATE")
                    camera_decision = "abstain"
                if "CANDIDATE_VERDICTS_INCOMPLETE" in parsed.get("warnings", []):
                    camera_reasons.append("CANDIDATE_VERDICTS_INCOMPLETE")
                    camera_decision = "abstain"
                if require_evidence_for_accept:
                    selected_verdict = next(
                        (
                            value
                            for value in parsed.get("candidate_verdicts", [])
                            if isinstance(value, Mapping) and value.get("rank") == selected
                        ),
                        None,
                    )
                    evidence_values = (
                        selected_verdict.get("evidence", [])
                        if isinstance(selected_verdict, Mapping)
                        else []
                    )
                    if (
                        not isinstance(evidence_values, Sequence)
                        or isinstance(evidence_values, (str, bytes, bytearray))
                        or not any(str(value).strip() for value in evidence_values)
                    ):
                        camera_reasons.append("ACCEPT_REQUIRES_EVIDENCE")
                        camera_decision = "abstain"
            camera_reports.append(
                {
                    "camera_id": qwen.get("camera_id"),
                    "decision": camera_decision,
                    "native_video_complete": native_ok,
                    "reason_codes": camera_reasons,
                    "constraint_diagnostics": constraint_codes,
                    "parsed_verification": parsed,
                }
            )
        # Aggregate camera-level verifier decisions without turning a
        # disagreement into a fabricated label.  A single recorded camera is
        # sufficient for the focused pilot; when several cameras are present,
        # require the most-supported rank to win a strict majority.
        reasons: list[str] = []
        accepted = [r for r in camera_reports if r.get("decision") == "accept"]
        rank_counts: dict[int, int] = {}
        for item in accepted:
            rank = item.get("parsed_verification", {}).get("selected_rank")
            if isinstance(rank, int):
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
        decision = "abstain"
        aggregate_parsed: Mapping[str, Any] = (
            camera_reports[0].get("parsed_verification", {})
            if camera_reports
            else {
                "parse_status": "MISSING",
                "decision": "abstain",
                "candidate_verdicts": [],
                "segments": [],
                "errors": ["QWEN_ROW_MISSING"],
                "warnings": [],
            }
        )
        if rank_counts:
            winning_rank, winning_count = max(
                rank_counts.items(), key=lambda item: (item[1], -item[0])
            )
            if len(camera_reports) <= 1 or winning_count * 2 > len(camera_reports):
                decision = "accept"
                aggregate_parsed = next(
                    (
                        r["parsed_verification"]
                        for r in accepted
                        if r["parsed_verification"].get("selected_rank") == winning_rank
                    ),
                    aggregate_parsed,
                )
            else:
                reasons.append("CAMERA_DECISION_DISAGREEMENT")
        elif len(camera_reports) == 1:
            # Split/edit/reject/abstain are already structured verifier
            # decisions; do not erase a valid split merely because it has no
            # single selected rank.
            single_decision = _text(camera_reports[0].get("decision")).lower()
            if single_decision in {"split", "edit", "reject", "abstain"}:
                decision = single_decision
                aggregate_parsed = camera_reports[0].get("parsed_verification", aggregate_parsed)
        if not qwen_rows:
            reasons.append("QWEN_ROW_MISSING")
        for camera in camera_reports:
            reasons.extend(str(x) for x in camera.get("reason_codes", []))
        reports.append(
            {
                "window_id": window_id,
                "candidate_source": "wemm_top_k_only",
                "candidate_count": len(candidates_list),
                "qwen_native_video_complete": bool(camera_reports)
                and all(bool(r.get("native_video_complete")) for r in camera_reports),
                "decision": decision,
                "reason_codes": reasons,
                "parsed_verification": aggregate_parsed,
                "camera_reports": camera_reports,
                "top_k": candidates_list,
            }
        )
    parse_invalid_rows = 0
    native_incomplete_rows = 0
    rank_constraint_error_rows = 0
    boundary_error_rows = 0
    for _row_key, native_ok, parsed_row, constraint_codes in validation_rows:
        if not native_ok:
            native_incomplete_rows += 1
        parse_status = _text(parsed_row.get("parse_status"))
        errors = (
            [str(value) for value in parsed_row.get("errors", [])]
            if isinstance(parsed_row.get("errors"), Sequence)
            else []
        )
        nested_reasons: list[str] = []
        for parsed_key in ("candidate_verdicts", "segments"):
            values = parsed_row.get(parsed_key, [])
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                for value in values:
                    if isinstance(value, Mapping) and isinstance(
                        value.get("reason_codes"), Sequence
                    ):
                        nested_reasons.extend(str(code) for code in value.get("reason_codes", []))
        errors.extend(nested_reasons)
        errors.extend(constraint_codes)
        if parse_status != "PARSED":
            parse_invalid_rows += 1
        if any("rank" in error.casefold() or "top-k" in error.casefold() for error in errors):
            rank_constraint_error_rows += 1
        if any(error.startswith("BOUNDARY_") or "BOUNDARY" in error for error in errors):
            boundary_error_rows += 1
    diagnostics = {
        "candidate_window_count": len(candidate_ids),
        "qwen_row_count": len(qwen_windows),
        "qwen_window_count": len(qwen_id_set),
        # A report row is emitted for every candidate window, including a
        # missing-Qwen row.  Count only IDs that actually had at least one
        # sidecar row so the diagnostic distinguishes coverage from output
        # row count.
        "joined_window_count": sum(1 for window_id in candidate_ids if qwen_by_id.get(window_id)),
        "missing_window_ids": sorted(candidate_id_set - qwen_id_set),
        "extra_window_ids": sorted(qwen_id_set - candidate_id_set),
        "qwen_rows_per_window": dict(sorted(qwen_id_counts.items())),
        "duplicate_camera_rows": duplicate_camera_rows,
        "native_incomplete_row_count": native_incomplete_rows,
        "parse_invalid_row_count": parse_invalid_rows,
        "rank_constraint_error_row_count": rank_constraint_error_rows,
        "boundary_error_row_count": boundary_error_rows,
    }
    return {
        "format": VERSION,
        "authority": AUTHORITY,
        "official_gold_status": OFFICIAL_GOLD_STATUS,
        "official_quality_status": "NOT_MEASURED",
        # This shadow report deliberately has no gold-backed accuracy.  Keep
        # the field explicit so downstream consumers cannot mistake the
        # verifier's accept rate for annotation accuracy.
        "accuracy_status": "NOT_MEASURED",
        "production_eligible": PRODUCTION_ELIGIBLE,
        "quality_claim": False,
        "diagnostics": diagnostics,
        "windows": reports,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_included": False,
            "epic_ontology_used": False,
            "mapper_used": False,
            "hash_or_digest_computed": False,
            "require_evidence_for_accept": require_evidence_for_accept,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    windows = report.get("windows", [])
    diagnostics = report.get("diagnostics", {})
    lines = [
        "# WeMM → Qwen candidate verification",
        "",
        f"> Official quality: `{_text(report.get('official_quality_status')) or 'NOT_MEASURED'}`; "
        "this is a non-gold production shadow.",
        f"> Accuracy status: `{_text(report.get('accuracy_status')) or 'NOT_MEASURED'}`.",
        "",
        "## Join diagnostics",
        "",
        f"- Candidate windows: `{diagnostics.get('candidate_window_count', '—')}`; "
        f"Qwen rows: `{diagnostics.get('qwen_row_count', '—')}`; joined windows: "
        f"`{diagnostics.get('joined_window_count', '—')}`.",
        f"- Missing IDs: `{len(diagnostics.get('missing_window_ids', []))}`; extra IDs: "
        f"`{len(diagnostics.get('extra_window_ids', []))}`; duplicate camera rows: "
        f"`{len(diagnostics.get('duplicate_camera_rows', []))}`.",
        f"- Native-incomplete rows: `{diagnostics.get('native_incomplete_row_count', '—')}`; "
        f"parse-invalid rows: `{diagnostics.get('parse_invalid_row_count', '—')}`; "
        f"rank errors: `{diagnostics.get('rank_constraint_error_row_count', '—')}`; "
        f"boundary errors: `{diagnostics.get('boundary_error_row_count', '—')}`.",
        "",
        "| Window | Decision | Candidate source | Reasons |",
        "|---|---|---|---|",
    ]
    if isinstance(windows, Sequence):
        for row in windows:
            if not isinstance(row, Mapping):
                continue
            reasons = ", ".join(str(x) for x in row.get("reason_codes", [])) or "—"
            lines.append(
                f"| {_text(row.get('window_id'))} | {_text(row.get('decision'))} | "
                f"{_text(row.get('candidate_source'))} | {reasons} |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "VERSION",
    "ProductionWemmQwenCandidateVerifierError",
    "build_candidate_verifier_prompt",
    "diagnose_candidate_bound_consensus_gate",
    "parse_qwen_candidate_verification_output",
    "render_candidate_bound_consensus_gate_markdown",
    "render_markdown",
    "verify_wemm_qwen_candidate_sidecars",
]
