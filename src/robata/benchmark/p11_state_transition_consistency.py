"""Deterministic state-pair consistency projection for Qwen diagnostics.

Qwen's native-video answer is retained as an unrestricted *candidate*.  This
benchmark-local projection derives a transition only when the candidate itself
contains two compatible endpoint states (for example ``off`` -> ``on`` or
``closed`` -> ``open``).  It does not infer polarity from a feature norm, use a
mapper/adapter, invoke another model, or read a private oracle.

The purpose is to separate two common failures that were previously reported as
one: a model can describe the two endpoint states correctly while emitting an
inconsistent ``direction`` field, and a frozen endpoint can still produce a
stale action story.  The projection repairs neither visual evidence nor
semantic identity; it only records the conservative state-pair consequence and
abstains when the evidence is insufficient.  Private semantic scoring is kept
in :func:`evaluate_posthoc` and never mutates the public result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VERSION = "p11-state-transition-consistency-v1"
PUBLIC_VERSION = "p11-state-transition-consistency-public-v1"
POSTHOC_VERSION = "p11-state-transition-consistency-posthoc-v1"
AUTHORITY = "LOCAL_NONPRODUCTION_ONLY"
VARIANTS: tuple[str, ...] = ("normal", "reverse", "pre_pre", "post_post")
EPS = 1e-6

INVERSE_DIRECTION = {
    "off_to_on": "on_to_off",
    "on_to_off": "off_to_on",
    "closed_to_open": "open_to_closed",
    "open_to_closed": "closed_to_open",
}

# Public candidate fields are deliberately allow-listed.  In particular, an
# unrestricted raw response is parsed, but arbitrary nested fields are not
# copied into the public diagnostic (this prevents an accidental oracle leak).
_CANDIDATE_FIELDS = frozenset(
    {
        "object",
        "active_part",
        "pre_state",
        "post_state",
        "direction",
        "state_relation",
        "evidence",
        "confidence",
        "abstention",
    }
)
_PRIVATE_TOKENS = frozenset(
    {
        "oracle",
        "semanticoracle",
        "oraclecomparison",
        "expectedobject",
        "expectedactivepart",
        "expectedprestate",
        "expectedpoststate",
        "expectedchangedirection",
        "groundtruth",
        "officialreference",
        "mapperoutput",
        "adapteroutput",
        "rawsemanticoutput",
    }
)
# Structural cardinalities emitted by the P9/P10 runners are metadata, not
# semantic labels.  Keep this allow-list explicit and aligned with both
# artifacts so a P10 result can be consumed without weakening the boundary.
_STRUCTURAL_EXPECTED_TOKENS = frozenset({"expectedgenerations", "expectedarms"})


class StateTransitionError(ValueError):
    """Raised when a public candidate or feature artifact is malformed."""


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _walk_keys(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.append(str(key))
            result.extend(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            result.extend(_walk_keys(child))
    return result


def _public_boundary(document: Mapping[str, Any], *, field: str) -> None:
    hits = sorted(
        {
            key
            for key in _walk_keys(document)
            if _token(key) in _PRIVATE_TOKENS
            or (
                _token(key).startswith("expected")
                and _token(key) not in _STRUCTURAL_EXPECTED_TOKENS
            )
        }
    )
    if hits:
        raise StateTransitionError(f"{field} contains private semantic fields: {hits}")
    label_blind = document.get(
        "label_blind_inference",
        document.get("label_blind_input", document.get("label_blind")),
    )
    # P10 keeps authority metadata in a dedicated ``authority_flags`` block
    # while P9 exposes the attestation at the document root.  Accept the
    # explicit P10 flag as a compatibility spelling; do not infer label-blind
    # status from a generic ``public_only`` bit.
    if label_blind is None:
        flags = document.get("authority_flags")
        if isinstance(flags, Mapping):
            label_blind = flags.get("label_blind")
    if label_blind is not True:
        raise StateTransitionError(f"{field} must attest label-blind input")
    if document.get("production_eligible") not in (None, False):
        raise StateTransitionError(f"{field}.production_eligible must be false")
    for key in (
        "training_invoked",
        "training_authorized",
        "mapper_or_adapter_changed",
        "hash_or_sha_used",
        "heldout_opened",
        "heldout_100_opened",
        "production_path_changed",
    ):
        if document.get(key) not in (None, False, 0, ""):
            raise StateTransitionError(f"{field}.{key} must be false")


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _normal_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"\s+", " ", value.strip().casefold())


def normalize_state(value: object) -> tuple[str, str] | None:
    """Map only unambiguous endpoint phrases to an axis/state pair.

    This is intentionally conservative.  A phrase containing an action (for
    example ``opening`` or ``turning off``) is not treated as a state; callers
    should abstain instead of converting a scene prior into polarity.
    """

    text = _normal_text(value)
    if text is None:
        return None
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if compact in {"off", "stopped", "inactive", "notflowing", "noflow", "notrunning"}:
        return ("on_off", "off")
    if compact in {"on", "active", "flowing", "running", "waterflowing"}:
        return ("on_off", "on")
    if compact in {"closed", "shut", "latched"}:
        return ("open_closed", "closed")
    if compact in {"open", "opened", "ajar", "unlatched"}:
        return ("open_closed", "open")
    # Permit a few direct functional-effect phrases, but reject verb phrases.
    if "not flowing" in text or "no water" in text or "water stopped" in text:
        return ("on_off", "off")
    if "water flowing" in text or "water is running" in text:
        return ("on_off", "on")
    return None


def derive_transition(pre_state: object, post_state: object) -> dict[str, Any]:
    """Derive a conservative relation/direction from two endpoint states."""

    pre = normalize_state(pre_state)
    post = normalize_state(post_state)
    result: dict[str, Any] = {
        "pre_normalized": pre[1] if pre else None,
        "post_normalized": post[1] if post else None,
        "state_axis": pre[0] if pre and post and pre[0] == post[0] else None,
        "relation": "unclear",
        "direction": None,
        "status": "unclear",
    }
    if pre is None or post is None:
        result["reason"] = "endpoint_state_not_unambiguous"
        return result
    if pre[0] != post[0]:
        result["reason"] = "endpoint_state_axes_conflict"
        return result
    if pre[1] == post[1]:
        result.update({"relation": "no_change", "direction": "no_direction", "status": "valid"})
        return result
    direction = f"{pre[1]}_to_{post[1]}"
    if direction not in INVERSE_DIRECTION:
        result["reason"] = "endpoint_state_pair_not_supported"
        return result
    result.update({"relation": "change", "direction": direction, "status": "valid"})
    return result


def _parse_raw_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    value: object = row.get("prediction")
    if value is None:
        value = row.get("raw_model_output")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = {}
    if not isinstance(value, Mapping):
        value = row
    result: dict[str, Any] = {}
    for field in _CANDIDATE_FIELDS:
        raw = value.get(field)
        if field == "confidence":
            parsed = _finite(raw)
            if parsed is not None:
                result[field] = max(0.0, min(1.0, parsed))
        elif field == "evidence":
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                result[field] = [str(item) for item in raw[:4] if isinstance(item, str)]
        elif raw is not None:
            result[field] = str(raw).strip() if isinstance(raw, str) else raw
    return result


def _variant_rows(document: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cases = document.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)):
        raise StateTransitionError("candidate document.cases must be an array")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            raise StateTransitionError("candidate cases must be objects")
        case_id = str(raw_case.get("case_id") or "")
        if not case_id:
            raise StateTransitionError("candidate case_id is required")
        nested = raw_case.get("variants")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            members = nested
        else:
            members = (raw_case,)
        for member in members:
            if not isinstance(member, Mapping):
                raise StateTransitionError(f"{case_id} variant must be an object")
            variant = str(member.get("variant") or member.get("variant_kind") or "normal")
            if variant in {"pair2_model336", "pair4_model336", "best_pair_model336"}:
                variant = "normal"
            if variant.endswith("_model336"):
                variant = variant.removesuffix("_model336")
            if variant not in VARIANTS:
                continue
            key = (case_id, variant)
            if key in rows:
                raise StateTransitionError(f"duplicate candidate row {case_id}/{variant}")
            rows[key] = {
                "case_id": case_id,
                "video_group": raw_case.get("video_group", member.get("video_group")),
                "variant": variant,
                "source": member,
                "prediction": _parse_raw_prediction(member),
            }
    return rows


def _number_from(value: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        candidate = value.get(name)
        parsed = _finite(candidate)
        if parsed is not None:
            return parsed
    return None


def _surface_feature_document(document: Mapping[str, Any], surface: str) -> Mapping[str, Any]:
    """Project one P10 feature surface onto its arm rows for diagnostics.

    P10 stores per-arm rows under ``cases`` and compact per-case metrics under
    ``case_metrics_by_surface``.  The latter is deliberately selected only
    when the caller asks for a named surface; the default P11 behavior remains
    byte-for-byte compatible with the original feature artifact.  Arm rows are
    copied in memory and receive only scalar metrics from the selected surface.
    """

    by_surface = document.get("case_metrics_by_surface")
    if not isinstance(by_surface, Mapping) or surface not in by_surface:
        available = (
            sorted(str(key) for key in by_surface) if isinstance(by_surface, Mapping) else []
        )
        raise StateTransitionError(
            f"feature surface {surface!r} is unavailable; choose one of {available}"
        )
    compact = by_surface[surface]
    if not isinstance(compact, Sequence) or isinstance(compact, (str, bytes, bytearray)):
        raise StateTransitionError(f"feature surface {surface!r} metrics must be an array")
    compact_by_case: dict[str, Mapping[str, Any]] = {
        str(row.get("case_id")): row
        for row in compact
        if isinstance(row, Mapping) and row.get("case_id") is not None
    }
    raw_rows = document.get("cases", document.get("rows", []))
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        raise StateTransitionError("feature document cases must be an array")
    projected_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        case_id = str(row.get("case_id") or "")
        compact_metrics = compact_by_case.get(case_id)
        if compact_metrics is not None:
            # Preserve the arm identity while replacing only the scalar metric
            # block.  The compact P10 row has no semantic labels.
            row["temporal_metrics"] = dict(compact_metrics)
        projected_rows.append(row)
    return {
        "artifact_version": document.get("artifact_version"),
        "authority_flags": document.get("authority_flags", {}),
        "label_blind": True,
        "production_eligible": False,
        "rows": projected_rows,
    }


def _feature_rows(
    document: Mapping[str, Any] | None,
    *,
    surface: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if document is None:
        return {}
    _public_boundary(document, field="feature_document")
    if surface is not None:
        document = _surface_feature_document(document, surface)
    # P10 currently retains one row per arm under ``cases`` and a compact
    # one-row-per-case projection under ``case_metrics``.  Accept either
    # surface (with the explicit ``rows`` spelling used by older probes) so a
    # public feature artifact cannot lose its scalar evidence merely because
    # its storage view changed.
    raw_rows = document.get(
        "rows",
        document.get("cases", document.get("case_metrics", [])),
    )
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        raise StateTransitionError("feature document rows must be an array")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        case_id = str(raw.get("case_id") or "")
        variant = str(raw.get("variant") or raw.get("variant_kind") or "")
        # P10 ``case_metrics`` is intentionally one row per case and its
        # ``normal_endpoint_delta`` describes the normal arm.  Treat that
        # compact surface as a normal-row observation; do not fan it out to
        # reverse/freeze variants whose metrics are distinct diagnostics.
        if not variant and "normal_endpoint_delta" in raw:
            variant = "normal"
        if variant.endswith("_model336"):
            variant = variant.removesuffix("_model336")
        if variant in {"pair2", "pair4", "best_pair"}:
            variant = "normal"
        if not case_id or variant not in VARIANTS:
            continue
        metrics: Mapping[str, Any] = raw
        for key in ("temporal_metrics", "feature_metrics", "metrics", "temporal_summary"):
            nested = raw.get(key)
            if isinstance(nested, Mapping):
                metrics = nested
                break
        # P10 repeats a case-level metric block on each arm row.  Select the
        # metric that belongs to the arm before looking at generic aliases;
        # otherwise a normal endpoint delta would incorrectly mark PRE/PRE or
        # POST/POST controls as a visual change.
        if variant in {"pre_pre", "post_post"}:
            duplicate_name = f"{variant}_duplicate_delta"
            delta = _number_from(
                metrics,
                (
                    duplicate_name,
                    f"{variant}_duplicate_delta_norm",
                    "freeze_delta_norm",
                    "duplicate_delta_norm",
                ),
            )
        else:
            delta = _number_from(
                metrics,
                (
                    "endpoint_delta_norm",
                    "delta_norm",
                    "normal_endpoint_delta_norm",
                    # P10's public aggregate uses the shorter, explicitly
                    # representation-scoped name.  Keep this alias here so
                    # the projection can consume the frozen feature audit
                    # directly instead of silently treating an observed
                    # metric as absent.
                    "normal_endpoint_delta",
                    "change_score",
                    "active_delta_norm",
                ),
            )
        result[(case_id, variant)] = {
            "available": delta is not None,
            "change_present": None if delta is None else bool(delta > EPS),
            "delta_norm": delta,
            "source": str(document.get("artifact_version") or "feature_artifact"),
        }
    return result


def build_public_result(
    candidate_document: object,
    feature_document: Mapping[str, Any] | None = None,
    *,
    feature_surface: str | None = None,
) -> dict[str, Any]:
    """Build a label-blind state-pair projection without mutating inputs.

    ``feature_surface`` is an optional benchmark diagnostic selector (for
    example ``pooler_output`` in a P10 artifact).  It never changes the
    candidate parser or production mapper path and defaults to the historical
    feature rows.
    """

    if not isinstance(candidate_document, Mapping):
        raise StateTransitionError("candidate document must be an object")
    _public_boundary(candidate_document, field="candidate_document")
    candidates = _variant_rows(candidate_document)
    features = _feature_rows(feature_document, surface=feature_surface)
    case_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (case_id, variant), row in sorted(candidates.items()):
        prediction = row["prediction"]
        transition = derive_transition(prediction.get("pre_state"), prediction.get("post_state"))
        feature = features.get((case_id, variant))
        # Structural freeze arms are only a fallback classification; a supplied
        # feature artifact can still veto it if it reports a non-null delta.
        structural_freeze = variant in {"pre_pre", "post_post"}
        feature_change = feature is not None and feature.get("change_present") is True
        freeze_supported = structural_freeze and not feature_change
        reasons: list[str] = []
        if freeze_supported:
            projected_relation = "no_change"
            projected_direction = "no_direction"
            disposition = "abstain_no_change"
            reasons.append("duplicate_endpoint_control_overrides_candidate_story")
        elif feature is not None and not feature.get("available"):
            projected_relation = transition["relation"]
            projected_direction = transition["direction"]
            disposition = "review"
            reasons.append("feature_change_evidence_unavailable")
        elif feature is not None and not feature_change:
            projected_relation = "no_change"
            projected_direction = "no_direction"
            disposition = "abstain_no_change"
            reasons.append("feature_endpoint_delta_is_null")
        elif transition["status"] != "valid":
            projected_relation = "unclear"
            projected_direction = None
            disposition = "review"
            reasons.append(str(transition.get("reason") or "state_pair_unclear"))
        else:
            projected_relation = transition["relation"]
            projected_direction = transition["direction"]
            disposition = "review"
            reasons.append("semantic_identity_and_polarity_not independently verified")
            raw_direction = prediction.get("direction")
            if raw_direction and raw_direction != projected_direction:
                reasons.append("candidate_direction_conflicts_with_state_pair")
        raw_direction = prediction.get("direction")
        direction_consistent = (
            isinstance(raw_direction, str)
            and projected_direction is not None
            and raw_direction == projected_direction
        )
        row_out = {
            "case_id": case_id,
            "video_group": row.get("video_group"),
            "variant": variant,
            "candidate": prediction,
            "derived_transition": transition,
            "feature_evidence": feature
            or {
                "available": False,
                "change_present": None,
                "delta_norm": None,
                "source": None,
            },
            "projection": {
                "relation": projected_relation,
                "direction": projected_direction,
                "disposition": disposition,
                "direction_consistent_with_raw": direction_consistent,
                "direction_source": "candidate_endpoint_state_pair"
                if projected_direction is not None
                else None,
                "semantic_identity_verified": False,
                "reasons": list(dict.fromkeys(reasons)),
            },
        }
        case_variants[case_id].append(row_out)
    rows = [row for values in case_variants.values() for row in values]
    normal = [row for row in rows if row["variant"] == "normal"]
    freeze = [row for row in rows if row["variant"] in {"pre_pre", "post_post"}]
    valid_state = [row for row in normal if row["derived_transition"]["status"] == "valid"]
    conflicts = [
        row
        for row in normal
        if "candidate_direction_conflicts_with_state_pair" in row["projection"]["reasons"]
    ]
    return {
        "artifact_version": PUBLIC_VERSION,
        "authority": AUTHORITY,
        "label_blind_inference": True,
        "public_only": True,
        "model_invoked": False,
        "generation_invoked": False,
        "feature_model_invoked": False,
        "training_invoked": False,
        "mapper_or_adapter_changed": False,
        "hash_or_sha_used": False,
        "heldout_100_opened": False,
        "production_eligible": False,
        "candidate_artifact_version": candidate_document.get("artifact_version"),
        "feature_surface": feature_surface,
        "rows": rows,
        "summary": {
            "case_count": len(case_variants),
            "row_count": len(rows),
            "normal_rows": len(normal),
            "normal_valid_state_pair_rows": len(valid_state),
            "normal_valid_state_pair_rate": len(valid_state) / len(normal) if normal else None,
            "raw_direction_consistency_rows": sum(
                row["projection"]["direction_consistent_with_raw"] for row in normal
            ),
            "raw_direction_consistency_rate": sum(
                row["projection"]["direction_consistent_with_raw"] for row in normal
            )
            / len(normal)
            if normal
            else None,
            "state_pair_direction_conflict_rows": len(conflicts),
            "freeze_rows": len(freeze),
            "freeze_no_change_rows": sum(
                row["projection"]["relation"] == "no_change" for row in freeze
            ),
            "freeze_no_change_rate": sum(
                row["projection"]["relation"] == "no_change" for row in freeze
            )
            / len(freeze)
            if freeze
            else None,
            "strict_visual_joint_rows": 0,
            "strict_visual_joint_note": "semantic object/part identity remains unverified",
        },
        "interpretation": (
            "This projection derives a transition only from the candidate's two endpoint "
            "states and uses duplicate controls as no-change guards. It is not a visual "
            "polarity classifier and does not certify semantic object/part identity."
        ),
    }


def evaluate_posthoc(
    public_document: Mapping[str, Any], oracle_document: Mapping[str, Any]
) -> dict[str, Any]:
    """Score projected transitions against a separate private oracle."""

    _public_boundary(public_document, field="public_document")
    if not isinstance(oracle_document, Mapping):
        raise StateTransitionError("oracle document must be an object")
    oracle_rows = oracle_document.get("cases", oracle_document)
    oracle_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    iterable: Sequence[Any]
    if isinstance(oracle_rows, Mapping):
        iterable = [
            dict(value, case_id=key) if isinstance(value, Mapping) else {}
            for key, value in oracle_rows.items()
        ]
    elif isinstance(oracle_rows, Sequence) and not isinstance(oracle_rows, (str, bytes, bytearray)):
        iterable = oracle_rows
    else:
        raise StateTransitionError("oracle cases must be an array or mapping")
    for raw in iterable:
        if not isinstance(raw, Mapping):
            continue
        case_id = str(raw.get("case_id") or "")
        variants = raw.get("variants")
        if isinstance(variants, Mapping):
            for variant, value in variants.items():
                if isinstance(value, Mapping):
                    oracle_index[(case_id, str(variant))] = value
        else:
            oracle_index[(case_id, str(raw.get("variant") or "normal"))] = raw
    rows: list[dict[str, Any]] = []
    for row in public_document.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("case_id")), str(row.get("variant")))
        oracle = oracle_index.get(key)
        if oracle is None:
            continue
        projected = row.get("projection", {})
        candidate = row.get("candidate", {})
        expected_direction = oracle.get("expected_change_direction", oracle.get("direction"))
        expected_relation = oracle.get("expected_state_relation", oracle.get("relation", "change"))
        rows.append(
            {
                "case_id": key[0],
                "variant": key[1],
                "raw_direction_match": candidate.get("direction") == expected_direction,
                "derived_direction_match": projected.get("direction") == expected_direction,
                "derived_relation_match": projected.get("relation") == expected_relation,
                "object_match": candidate.get("object")
                == oracle.get("expected_object", oracle.get("object")),
                "active_part_match": candidate.get("active_part")
                == oracle.get("expected_active_part", oracle.get("active_part")),
            }
        )

    def rate(field: str) -> float | None:
        return sum(bool(row[field]) for row in rows) / len(rows) if rows else None

    return {
        "artifact_version": POSTHOC_VERSION,
        "authority": "LOCAL_NONPRODUCTION_POSTHOC",
        "posthoc_only": True,
        "private_fields_copied_to_public": False,
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "raw_direction_accuracy": rate("raw_direction_match"),
            "derived_direction_accuracy": rate("derived_direction_match"),
            "derived_relation_accuracy": rate("derived_relation_match"),
            "object_accuracy": rate("object_match"),
            "active_part_accuracy": rate("active_part_match"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--feature-surface",
        help="optional P10 surface (for example pooler_output) for diagnostic comparison",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    features = (
        json.loads(args.features.read_text(encoding="utf-8")) if args.features is not None else None
    )
    result = build_public_result(candidate, features, feature_surface=args.feature_surface)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


__all__ = [
    "AUTHORITY",
    "INVERSE_DIRECTION",
    "POSTHOC_VERSION",
    "PUBLIC_VERSION",
    "StateTransitionError",
    "build_public_result",
    "derive_transition",
    "evaluate_posthoc",
    "normalize_state",
]


if __name__ == "__main__":
    raise SystemExit(main())
