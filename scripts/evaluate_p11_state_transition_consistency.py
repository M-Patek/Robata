#!/usr/bin/env python3
# ruff: noqa: E501
"""Evaluate a P11 public projection against a private endpoint sidecar.

This evaluator is deliberately separate from the P11 public projection.  It
first validates the completed public artifact (including its label-blind
attestation and absence of private semantic keys), and only then opens the
private endpoint review file.  The private file is used as an evaluator-only
join through each case's nested ``semantic_oracle`` block.  The emitted JSON
and Markdown contain scalar rates/counts only; candidate rows and private
semantic values are never copied into the result.

No model, media, mapper/adapter, training, hash/digest, or production path is
invoked by this command.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".agent_tmp"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.p11_state_transition_consistency import (  # noqa: E402
    PUBLIC_VERSION,
    StateTransitionError,
)

P11_POSTHOC_ARTIFACT_VERSION = "p11-state-transition-consistency-posthoc-v1"
# Short aliases keep the script convenient to import alongside the benchmark
# module, which exposes ``POSTHOC_VERSION`` under the same artifact identity.
POSTHOC_VERSION = P11_POSTHOC_ARTIFACT_VERSION
AUTHORITY = "LOCAL_NONPRODUCTION_POSTHOC"
DEFAULT_PROJECTION = TMP / "p11_p9_p10_projection_20260826.json"
DEFAULT_PRIVATE = TMP / "mechanism_endpoint_review_private_20260826.json"
DEFAULT_OUTPUT = TMP / "p11_p9_p10_projection_posthoc_20260826.json"
DEFAULT_REPORT = TMP / "p11_p9_p10_projection_posthoc_20260826.md"

VARIANTS: tuple[str, ...] = ("normal", "reverse", "pre_pre", "post_post")
_DIRECTIONS = frozenset({"off_to_on", "on_to_off", "closed_to_open", "open_to_closed"})
_INVERSE_DIRECTION = {
    "off_to_on": "on_to_off",
    "on_to_off": "off_to_on",
    "closed_to_open": "open_to_closed",
    "open_to_closed": "closed_to_open",
}
_PRIVATE_KEY_TOKENS = frozenset(
    {
        "oracle",
        "semanticoracle",
        "oraclecomparison",
        "expectedobject",
        "expectedactivepart",
        "expectedprestate",
        "expectedpoststate",
        "expectedchangedirection",
        "expectedstaterelation",
        "groundtruth",
        "officialreference",
        "rawsemanticoutput",
        "semantic",
        "mapperoutput",
        "adapteroutput",
    }
)


class P11PosthocError(ValueError):
    """Raised when a P11 post-hoc input violates its evaluator boundary."""


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise P11PosthocError(f"{description} is not a file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P11PosthocError(f"unable to read {description}: {resolved}") from error
    if not isinstance(value, dict):
        raise P11PosthocError(f"{description} must be a JSON object: {resolved}")
    return value


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 4)


def _public_validation(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the P11 public boundary and return scalar structural metrics."""

    if document.get("artifact_version") != PUBLIC_VERSION:
        raise P11PosthocError(
            f"unexpected P11 projection artifact version: {document.get('artifact_version')!r}"
        )
    if document.get("label_blind_inference") is not True:
        raise P11PosthocError("P11 projection must attest label_blind_inference=true")
    if document.get("public_only") is not True:
        raise P11PosthocError("P11 projection must attest public_only=true")
    for key in (
        "model_invoked",
        "generation_invoked",
        "feature_model_invoked",
        "training_invoked",
        "mapper_or_adapter_changed",
        "hash_or_sha_used",
        "heldout_100_opened",
        "production_eligible",
    ):
        value = document.get(key)
        if value not in (False, 0, None, ""):
            raise P11PosthocError(f"P11 projection authorization flag {key!r} must be false")

    leaked = sorted({key for key in _walk_keys(document) if _token(key) in _PRIVATE_KEY_TOKENS})
    if leaked:
        raise P11PosthocError(f"P11 projection contains private semantic keys: {leaked}")

    rows = document.get("rows")
    if not isinstance(rows, list):
        raise P11PosthocError("P11 projection rows must be an array")
    seen: set[tuple[str, str]] = set()
    variant_counts: defaultdict[str, int] = defaultdict(int)
    normal_count = 0
    freeze_count = 0
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise P11PosthocError("P11 projection row must be an object")
        case_id = str(raw.get("case_id") or "")
        variant = str(raw.get("variant") or "")
        if not case_id or variant not in VARIANTS:
            raise P11PosthocError("P11 projection row has invalid case_id/variant")
        pair = (case_id, variant)
        if pair in seen:
            raise P11PosthocError(f"duplicate P11 projection row: {case_id}/{variant}")
        seen.add(pair)
        variant_counts[variant] += 1
        if variant == "normal":
            normal_count += 1
        else:
            freeze_count += variant in {"pre_pre", "post_post"}
        candidate = raw.get("candidate")
        projection = raw.get("projection")
        if not isinstance(candidate, Mapping) or not isinstance(projection, Mapping):
            raise P11PosthocError(f"{case_id}/{variant} is missing candidate/projection objects")
        feature = raw.get("feature_evidence")
        if feature is not None and not isinstance(feature, Mapping):
            raise P11PosthocError(f"{case_id}/{variant} has invalid feature_evidence")
        if isinstance(feature, Mapping) and feature.get("feature_values_serialized") not in (
            None,
            False,
        ):
            raise P11PosthocError("P11 projection cannot carry serialized feature values")

    summary = document.get("summary")
    if summary is not None and not isinstance(summary, Mapping):
        raise P11PosthocError("P11 projection summary must be an object")
    return {
        "valid": True,
        "artifact_version": str(document.get("artifact_version")),
        "row_count": len(rows),
        "case_count": len({case_id for case_id, _ in seen}),
        "normal_rows": normal_count,
        "freeze_rows": freeze_count,
        "variant_counts": dict(sorted(variant_counts.items())),
    }


def validate_public_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    """Public name for the fail-closed projection validator."""

    try:
        return _public_validation(document)
    except (P11PosthocError, StateTransitionError):
        raise


def _normal_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _surface(value: object) -> str | None:
    """Map endpoint prose to a coarse state surface for evaluator matching."""

    text = _normal_text(value)
    if not text:
        return None
    # Negated forms must be considered before their positive token.
    if any(
        token in text for token in ("not flowing", "no flow", "not running", "stopped", "inactive")
    ):
        return "off"
    if any(token in text for token in ("flowing", "water is running", "running", "active")):
        return "on"
    if any(token in text for token in ("closed", "occluded", "shut", "latched")):
        return "closed"
    if any(token in text for token in ("interior exposed", "open", "opened", "ajar", "unlatched")):
        return "open"
    return None


def _phrase_match(candidate: object, expected: object) -> bool:
    """Match short object/part labels without requiring exact private prose."""

    candidate_text = _normal_text(candidate)
    expected_text = _normal_text(expected)
    if not candidate_text or not expected_text:
        return False
    candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate_text))
    alternatives = [part.strip() for part in re.split(r"/|,|\bor\b", expected_text) if part.strip()]
    if not alternatives:
        alternatives = [expected_text]
    for alternative in alternatives:
        expected_tokens = set(re.findall(r"[a-z0-9]+", alternative))
        if not expected_tokens:
            continue
        # Either side may be an informative extension (e.g. ``refrigerator``
        # vs ``refrigerator door``), but unrelated nouns do not match.
        if candidate_tokens <= expected_tokens or expected_tokens <= candidate_tokens:
            return True
    return False


def _direction(value: object) -> str | None:
    text = _normal_text(value).replace("-", "_").replace(" ", "_")
    if text in _DIRECTIONS:
        return text
    if text in {"no_direction", "none", "nochange", "no_change"}:
        return "no_direction"
    return None


def _private_cases(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index nested semantic-oracle records without exposing their values."""

    if document.get("label_blind_model_input") is not True:
        raise P11PosthocError("private endpoint review must attest label_blind_model_input=true")
    for key in ("model_invoked", "gpu_invoked", "hash_or_sha_used"):
        if document.get(key) not in (False, 0, None, ""):
            raise P11PosthocError(
                f"private endpoint review authorization flag {key!r} must be false"
            )
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise P11PosthocError("private endpoint review cases must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in cases:
        if not isinstance(raw, Mapping):
            continue
        case_id = str(raw.get("case_id") or "")
        oracle = raw.get("semantic_oracle")
        if case_id and isinstance(oracle, Mapping):
            if case_id in result:
                raise P11PosthocError(f"duplicate private endpoint case: {case_id}")
            result[case_id] = raw
    if not result:
        raise P11PosthocError("private endpoint review has no nested semantic_oracle cases")
    return result


def _expected_for_variant(oracle: Mapping[str, Any], variant: str) -> dict[str, Any]:
    nested = oracle.get("semantic_oracle")
    if not isinstance(nested, Mapping):
        raise P11PosthocError("private case is missing semantic_oracle")
    normal_direction = _direction(nested.get("expected_change_direction"))
    normal_pre = _surface(nested.get("expected_pre_state"))
    normal_post = _surface(nested.get("expected_post_state"))
    if variant == "normal":
        return {
            "relation": "change" if normal_direction in _DIRECTIONS else "unclear",
            "direction": normal_direction,
            "pre": normal_pre,
            "post": normal_post,
        }
    if variant == "reverse":
        return {
            "relation": "change" if normal_direction in _DIRECTIONS else "unclear",
            "direction": (
                _INVERSE_DIRECTION.get(normal_direction) if normal_direction is not None else None
            ),
            "pre": normal_post,
            "post": normal_pre,
        }
    if variant == "pre_pre":
        return {
            "relation": "no_change",
            "direction": "no_direction",
            "pre": normal_pre,
            "post": normal_pre,
        }
    if variant == "post_post":
        return {
            "relation": "no_change",
            "direction": "no_direction",
            "pre": normal_post,
            "post": normal_post,
        }
    raise P11PosthocError(f"unsupported P11 variant: {variant}")


def _score_rows(
    public_document: Mapping[str, Any],
    private_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create evaluator-local booleans; this list is not emitted."""

    scored: list[dict[str, Any]] = []
    rows = public_document.get("rows")
    if not isinstance(rows, list):
        raise P11PosthocError("P11 projection rows must be an array")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = str(row.get("case_id") or "")
        variant = str(row.get("variant") or "")
        oracle = private_index.get(case_id)
        if oracle is None:
            continue
        expected = _expected_for_variant(oracle, variant)
        nested = oracle.get("semantic_oracle")
        assert isinstance(nested, Mapping)
        candidate = row.get("candidate")
        projection = row.get("projection")
        transition = row.get("derived_transition")
        feature = row.get("feature_evidence")
        if not isinstance(candidate, Mapping):
            candidate = {}
        if not isinstance(projection, Mapping):
            projection = {}
        if not isinstance(transition, Mapping):
            transition = {}
        if not isinstance(feature, Mapping):
            feature = {}
        expected_direction = expected["direction"]
        expected_relation = expected["relation"]
        direction_applicable = (
            expected_direction in _DIRECTIONS or expected_direction == "no_direction"
        )
        relation_applicable = expected_relation in {"change", "no_change"}
        candidate_relation = _normal_text(candidate.get("state_relation")).replace(" ", "_")
        projected_relation = _normal_text(projection.get("relation")).replace(" ", "_")
        scored.append(
            {
                "case_id": case_id,
                "variant": variant,
                "object_match": _phrase_match(
                    candidate.get("object"), nested.get("expected_object")
                ),
                "active_part_match": _phrase_match(
                    candidate.get("active_part"), nested.get("expected_active_part")
                ),
                "pre_state_match": expected["pre"] is not None
                and _surface(candidate.get("pre_state")) == expected["pre"],
                "post_state_match": expected["post"] is not None
                and _surface(candidate.get("post_state")) == expected["post"],
                "raw_direction_match": _direction(candidate.get("direction")) == expected_direction,
                "derived_direction_match": _direction(transition.get("direction"))
                == expected_direction,
                "projected_direction_match": _direction(projection.get("direction"))
                == expected_direction,
                "raw_relation_match": candidate_relation == expected_relation,
                "derived_relation_match": _normal_text(transition.get("relation")).replace(" ", "_")
                == expected_relation,
                "projected_relation_match": projected_relation == expected_relation,
                "direction_applicable": direction_applicable,
                "relation_applicable": relation_applicable,
                "feature_available": feature.get("available") is True,
                "feature_change_present": feature.get("change_present") is True,
                "feature_no_change": (
                    feature.get("available") is True and feature.get("change_present") is not True
                ),
                "projection_direction_consistent": projection.get("direction_consistent_with_raw")
                is True,
            }
        )
    return scored


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def applicable_rate(key: str, gate: str) -> float | None:
        applicable = [row for row in rows if row.get(gate)]
        return _rate(applicable, key)

    return {
        "count": len(rows),
        "object_match_rate": _rate(rows, "object_match"),
        "active_part_match_rate": _rate(rows, "active_part_match"),
        "pre_state_match_rate": _rate(rows, "pre_state_match"),
        "post_state_match_rate": _rate(rows, "post_state_match"),
        "raw_relation_match_rate": applicable_rate("raw_relation_match", "relation_applicable"),
        "projected_relation_match_rate": applicable_rate(
            "projected_relation_match", "relation_applicable"
        ),
        "raw_direction_match_rate": applicable_rate("raw_direction_match", "direction_applicable"),
        "derived_direction_match_rate": applicable_rate(
            "derived_direction_match", "direction_applicable"
        ),
        "projected_direction_match_rate": applicable_rate(
            "projected_direction_match", "direction_applicable"
        ),
        "strict_raw_joint_rate": _rate(
            [
                row
                for row in rows
                if row.get("relation_applicable") and row.get("direction_applicable")
            ],
            "strict_raw_joint",
        ),
        "strict_projected_joint_rate": _rate(
            [
                row
                for row in rows
                if row.get("relation_applicable") and row.get("direction_applicable")
            ],
            "strict_projected_joint",
        ),
        "feature_available_rate": _rate(rows, "feature_available"),
        "feature_change_present_rate": _rate(rows, "feature_change_present"),
        "feature_no_change_rate": _rate(rows, "feature_no_change"),
    }


def _decorate_joint_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["strict_raw_joint"] = bool(
            row.get("object_match")
            and row.get("active_part_match")
            and row.get("raw_relation_match")
            and row.get("raw_direction_match")
        )
        row["strict_projected_joint"] = bool(
            row.get("object_match")
            and row.get("active_part_match")
            and row.get("projected_relation_match")
            and row.get("projected_direction_match")
        )
        row["freeze_gate_pass"] = bool(
            row.get("variant") in {"pre_pre", "post_post"}
            and row.get("projected_relation_match")
            and row.get("projected_direction_match")
        )


def evaluate_documents(
    public_document: Mapping[str, Any],
    private_document: Mapping[str, Any],
    *,
    public_source: str = "",
    private_source: str = "",
) -> tuple[dict[str, Any], str]:
    """Validate and post-hoc score already-loaded P11 documents.

    The returned document intentionally contains no row-level candidate or
    oracle values.  ``private_document`` is only retained in local variables.
    """

    validation = _public_validation(public_document)
    private_index = _private_cases(private_document)
    public_rows = public_document.get("rows")
    assert isinstance(public_rows, list)
    public_case_ids = {
        str(row.get("case_id") or "")
        for row in public_rows
        if isinstance(row, Mapping) and row.get("case_id")
    }
    missing_cases = sorted(public_case_ids - set(private_index))
    if missing_cases:
        raise P11PosthocError(
            f"private endpoint review is missing nested semantic cases: {missing_cases}"
        )
    rows = _score_rows(public_document, private_index)
    _decorate_joint_flags(rows)
    if not rows:
        raise P11PosthocError("P11 projection has no rows matching private endpoint cases")

    by_variant = {
        variant: _metric_summary([row for row in rows if row["variant"] == variant])
        for variant in VARIANTS
    }
    normal = [row for row in rows if row["variant"] == "normal"]
    reverse = [row for row in rows if row["variant"] == "reverse"]
    freeze = [row for row in rows if row["variant"] in {"pre_pre", "post_post"}]
    reverse_applicable = [row for row in reverse if row.get("direction_applicable")]
    freeze_applicable = [row for row in freeze if row.get("relation_applicable")]
    projection_summary = public_document.get("summary")
    if not isinstance(projection_summary, Mapping):
        projection_summary = {}
    summary = _metric_summary(rows)
    summary.update(
        {
            "normal_count": len(normal),
            "reverse_count": len(reverse),
            "freeze_count": len(freeze),
            "reverse_direction_sensitivity_rate": _rate(
                reverse_applicable, "projected_direction_match"
            ),
            "freeze_no_change_gate_rate": _rate(freeze_applicable, "freeze_gate_pass"),
            "public_raw_direction_consistency_rate": _finite(
                projection_summary.get("raw_direction_consistency_rate")
            ),
            "public_state_pair_conflict_rows": int(
                projection_summary.get("state_pair_direction_conflict_rows") or 0
            ),
            "normal_feature_change_rate": _rate(normal, "feature_change_present"),
            "freeze_feature_no_change_rate": _rate(
                [row for row in freeze if row.get("feature_available")],
                "feature_no_change",
            ),
            "normal_raw_direction_match_rate": _rate(
                [row for row in normal if row.get("direction_applicable")],
                "raw_direction_match",
            ),
            "normal_derived_direction_match_rate": _rate(
                [row for row in normal if row.get("direction_applicable")],
                "derived_direction_match",
            ),
            "normal_projected_direction_match_rate": _rate(
                [row for row in normal if row.get("direction_applicable")],
                "projected_direction_match",
            ),
            "normal_strict_projected_joint_rate": _rate(
                [
                    row
                    for row in normal
                    if row.get("relation_applicable") and row.get("direction_applicable")
                ],
                "strict_projected_joint",
            ),
            # Compatibility aliases mirror the compact summary names used by
            # the P11 module's evaluator while retaining arm-specific rates
            # above.  These remain scalar values only.
            "raw_direction_accuracy": _rate(
                [row for row in normal if row.get("direction_applicable")],
                "raw_direction_match",
            ),
            "derived_direction_accuracy": _rate(
                [row for row in normal if row.get("direction_applicable")],
                "projected_direction_match",
            ),
            "derived_relation_accuracy": _rate(
                [row for row in normal if row.get("relation_applicable")],
                "projected_relation_match",
            ),
            "object_accuracy": _rate(normal, "object_match"),
            "active_part_accuracy": _rate(normal, "active_part_match"),
        }
    )

    # The output surface is intentionally scalar-only.  Case IDs and private
    # expected values are not copied; per-arm aggregates are scalar mappings.
    output: dict[str, Any] = {
        "artifact_version": P11_POSTHOC_ARTIFACT_VERSION,
        "authority": AUTHORITY,
        "public_projection_source": public_source,
        "private_source": private_source,
        "public_projection_validated_first": True,
        "public_only": False,
        "private_oracle_read": True,
        "semantic_scoring": True,
        "private_fields_copied_to_public": False,
        "model_invoked_by_evaluator": False,
        "media_opened_by_evaluator": False,
        "mapper_or_adapter_changed": False,
        "training_invoked": False,
        "hash_or_sha_used": False,
        "heldout_opened": False,
        "production_eligible": False,
        "public_validation": validation,
        "summary": summary,
        "by_variant": by_variant,
        "private_case_count": len(private_index),
        "joined_row_count": len(rows),
    }
    report = _render_report(output)
    return output, report


def _fmt_rate(value: object) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.1%}"


def _render_report(output: Mapping[str, Any]) -> str:
    summary = output.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    by_variant = output.get("by_variant")
    if not isinstance(by_variant, Mapping):
        by_variant = {}
    lines = [
        "# P11 state-transition consistency post-hoc evaluation",
        "",
        "The public P11 projection was validated before the private endpoint review was read.",
        "Only scalar metrics are emitted; private semantic values and candidate rows are not copied.",
        "",
        f"- Joined rows: **{int(output.get('joined_row_count') or 0)}**; private cases indexed: **{int(output.get('private_case_count') or 0)}**.",
        f"- Raw direction accuracy: **{_fmt_rate(summary.get('raw_direction_match_rate'))}**; projected direction accuracy: **{_fmt_rate(summary.get('projected_direction_match_rate'))}**.",
        f"- Raw relation accuracy: **{_fmt_rate(summary.get('raw_relation_match_rate'))}**; projected relation accuracy: **{_fmt_rate(summary.get('projected_relation_match_rate'))}**.",
        f"- Normal-arm raw/projected direction: **{_fmt_rate(summary.get('normal_raw_direction_match_rate'))}** / **{_fmt_rate(summary.get('normal_projected_direction_match_rate'))}**.",
        f"- Reverse projected-direction sensitivity: **{_fmt_rate(summary.get('reverse_direction_sensitivity_rate'))}**.",
        f"- PRE/PRE + POST/POST projected no-change gate: **{_fmt_rate(summary.get('freeze_no_change_gate_rate'))}**.",
        f"- Public endpoint-pair raw-direction consistency: **{_fmt_rate(summary.get('public_raw_direction_consistency_rate'))}**.",
        "",
        "## Scalar metrics by arm",
        "",
        "| arm | n | object | part | relation (projected) | direction (projected) | strict projected | feature change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        metrics = by_variant.get(variant)
        if not isinstance(metrics, Mapping):
            metrics = {}
        lines.append(
            f"| {variant} | {int(metrics.get('count') or 0)} | "
            f"{_fmt_rate(metrics.get('object_match_rate'))} | "
            f"{_fmt_rate(metrics.get('active_part_match_rate'))} | "
            f"{_fmt_rate(metrics.get('projected_relation_match_rate'))} | "
            f"{_fmt_rate(metrics.get('projected_direction_match_rate'))} | "
            f"{_fmt_rate(metrics.get('strict_projected_joint_rate'))} | "
            f"{_fmt_rate(metrics.get('feature_change_present_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Endpoint-state projection repairs an internally inconsistent direction field only as a review diagnostic.",
            "- Feature deltas and duplicate controls provide change/no-change evidence, not semantic polarity or object identity certification.",
            "- This artifact is non-production post-hoc analysis and must not be used as a promotion or training input.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(
    *,
    projection_path: Path = DEFAULT_PROJECTION,
    private_path: Path = DEFAULT_PRIVATE,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Read/validate public projection, then read private nested semantics."""

    # Ordering is intentional: private_path is not touched until this call
    # returns successfully.
    public = _load_json(projection_path, description="P11 public projection")
    _public_validation(public)
    private = _load_json(private_path, description="private endpoint review")
    output, report = evaluate_documents(
        public,
        private,
        public_source=str(Path(projection_path).expanduser()),
        private_source=str(Path(private_path).expanduser()),
    )
    output_path = Path(output_path).expanduser()
    report_path = Path(report_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(report, encoding="utf-8")
    summary = output.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "artifact_version": P11_POSTHOC_ARTIFACT_VERSION,
        "output": str(output_path),
        "report": str(report_path),
        "joined_row_count": output.get("joined_row_count", 0),
        "raw_direction_match_rate": summary.get("raw_direction_match_rate"),
        "projected_direction_match_rate": summary.get("projected_direction_match_rate"),
        "freeze_no_change_gate_rate": summary.get("freeze_no_change_gate_rate"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "projection_positional",
        nargs="?",
        type=Path,
        help="P11 public projection JSON (default: frozen .agent_tmp artifact)",
    )
    parser.add_argument(
        "--projection",
        "--public-projection",
        "--public",
        dest="projection_option",
        type=Path,
    )
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    projection = args.projection_option or args.projection_positional or DEFAULT_PROJECTION
    print(
        json.dumps(
            evaluate(
                projection_path=projection,
                private_path=args.private,
                output_path=args.output,
                report_path=args.report,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


__all__ = [
    "AUTHORITY",
    "DEFAULT_OUTPUT",
    "DEFAULT_PRIVATE",
    "DEFAULT_PROJECTION",
    "DEFAULT_REPORT",
    "P11_POSTHOC_ARTIFACT_VERSION",
    "POSTHOC_VERSION",
    "P11PosthocError",
    "evaluate",
    "evaluate_documents",
    "main",
    "validate_public_projection",
]


if __name__ == "__main__":
    raise SystemExit(main())
