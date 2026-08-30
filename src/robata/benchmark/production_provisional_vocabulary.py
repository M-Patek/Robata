"""Build and compare a non-gold production action vocabulary.

The selected production recording has no independently adjudicated action
segments.  This module therefore treats the existing agent visual review as a
*proposal* only.  It materialises a small, source-facing vocabulary and an
explicitly unapproved projection to the existing EPIC action catalog.  The
projection is useful for finding coverage gaps, but it must never be used as
gold or silently modify the EPIC ontology/Mapper.

All functions are sidecar-only: they read JSON objects and perform no model
inference, media decoding, training, ontology mutation, or digest/hash work.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any, Final

PROVISIONAL_VOCABULARY_VERSION: Final = "robata-production-provisional-action-vocabulary-v1"
PROVISIONAL_COMPARISON_VERSION: Final = "robata-production-provisional-vocabulary-comparison-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
PROVISIONAL_STATUS: Final = "PROVISIONAL_UNAPPROVED"
MEASUREMENT_STATUS: Final = "NOT_MEASURED"

# This family is intentionally small.  It is a routing aid for the observed
# garment noun, not an assertion that the words are interchangeable in the
# production annotation ontology.
TEXTILE_FAMILY: Final = (
    "cloth",
    "clothes",
    "clothing",
    "fabric",
    "garment",
    "pants",
    "sheets",
    "shirt",
    "shorts",
)

_INFLECTIONS: Final[dict[str, str]] = {
    "adjusts": "adjust",
    "adjusting": "adjust",
    "arranges": "arrange",
    "arranging": "arrange",
    "flattens": "flatten",
    "flattening": "flatten",
    "folds": "fold",
    "folding": "fold",
    "picks": "pick",
    "picking": "pick",
    "places": "place",
    "placing": "place",
    "presses": "press",
    "pressing": "press",
    "smooths": "smooth",
    "smoothing": "smooth",
    "spreads": "spread",
    "spreading": "spread",
}


class ProvisionalVocabularyError(ValueError):
    """Raised when a non-gold vocabulary input violates its contract."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvisionalVocabularyError(f"{field} must be an object")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProvisionalVocabularyError(f"{field} must be a string")
    result = unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise ProvisionalVocabularyError(f"{field} must be non-empty")
    return result


def _normalise(value: object, *, field: str) -> str:
    text = _text(value, field=field).casefold()
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [_INFLECTIONS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def _pair(value: Mapping[str, Any], *, field: str) -> tuple[str, str]:
    verb = _normalise(value.get("verb"), field=f"{field}.verb")
    noun = _normalise(value.get("noun"), field=f"{field}.noun")
    if not verb or not noun:
        raise ProvisionalVocabularyError(f"{field} must contain verb and noun")
    return verb, noun


def _label(pair: tuple[str, str]) -> str:
    return f"{pair[0]} {pair[1]}"


def _noun_family(noun: str) -> str:
    return "textile_generic" if noun in TEXTILE_FAMILY else noun


def _assert_non_gold_review_pack(pack: Mapping[str, Any]) -> None:
    """Refuse packs that could accidentally be treated as official labels."""

    expected = "robata-production-agent-reviewed-segment-pack-v1"
    if pack.get("format") != expected:
        raise ProvisionalVocabularyError(f"review pack format must be {expected!r}")
    if pack.get("authority") != AUTHORITY:
        raise ProvisionalVocabularyError("review pack authority is not local-only")
    if pack.get("production_eligible") is True:
        raise ProvisionalVocabularyError("agent review pack cannot be production eligible")
    contract = _mapping(pack.get("review_contract"), field="review_contract")
    if contract.get("accepted_as_gold") is not False:
        raise ProvisionalVocabularyError("review contract must keep accepted_as_gold=false")
    if contract.get("official_gold_status") not in {None, "PENDING_HUMAN_REVIEW"}:
        raise ProvisionalVocabularyError("review pack unexpectedly claims official gold")
    if contract.get("not_an_evaluator_input") is not True:
        raise ProvisionalVocabularyError("review pack must be excluded from evaluator")
    controls = pack.get("controls")
    if isinstance(controls, Mapping):
        for key in (
            "gold_read",
            "gold_written",
            "predictions_copied_to_gold",
            "model_predictions_copied",
        ):
            if controls.get(key) is True:
                raise ProvisionalVocabularyError(f"review pack control {key} must be false")


def _review_pair(segment: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    pair = _pair(segment, field=field)
    label_text = segment.get("label_text")
    if isinstance(label_text, str) and label_text.strip():
        surface_label = unicodedata.normalize("NFKC", label_text).strip()
    else:
        surface_label = _label(pair)
    result: dict[str, Any] = {
        "verb": pair[0],
        "noun": pair[1],
        "canonical_label": _label(pair),
        "surface_label": surface_label,
        "confidence": segment.get("confidence"),
        "boundary_status": segment.get("boundary_status"),
        "source_segment_id": segment.get("segment_id"),
    }
    for key in ("attributes", "location", "hand", "start_seconds", "end_seconds"):
        if key in segment:
            result[key] = segment.get(key)
    return result


def _epic_labels(ontology: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if ontology is None:
        return []
    rows = ontology.get("labels", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ProvisionalVocabularyError("ontology.labels must be an array")
    labels: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"ontology.labels[{index}]")
        try:
            # Frozen WeMM catalogs use ``verb_key``/``noun_key`` while small
            # fixtures often use the shorter ``verb``/``noun`` spelling.
            pair = (
                _normalise(
                    row.get("verb", row.get("verb_key")),
                    field=f"ontology.labels[{index}].verb",
                ),
                _normalise(
                    row.get("noun", row.get("noun_key")),
                    field=f"ontology.labels[{index}].noun",
                ),
            )
        except ProvisionalVocabularyError:
            continue
        action_key = row.get("action_key")
        if not isinstance(action_key, Sequence) or isinstance(action_key, (str, bytes)):
            action_key = None
        text_map = row.get("texts")
        canonical = None
        if isinstance(text_map, Mapping) and isinstance(text_map.get("canonical"), str):
            canonical = text_map.get("canonical")
        labels.append(
            {
                "action_key": list(action_key) if action_key is not None else None,
                "verb": pair[0],
                "noun": pair[1],
                "label": canonical or _label(pair),
            }
        )
    return labels


def _mapping_candidates(
    pair: tuple[str, str],
    ontology_labels: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Return conservative exact/family candidates; never invent synonyms."""

    exact: list[dict[str, Any]] = []
    family: list[dict[str, Any]] = []
    for row in ontology_labels:
        row_pair = (str(row.get("verb", "")), str(row.get("noun", "")))
        if row_pair == pair:
            exact.append(dict(row))
        elif row_pair[0] == pair[0] and _noun_family(row_pair[1]) == _noun_family(pair[1]):
            family.append(dict(row))
    if exact:
        return exact, "EXACT_PAIR"
    if family:
        return family, "VERB_EXACT_NOUN_FAMILY_ONLY"
    return [], "NO_EXACT_OR_CONSERVATIVE_FAMILY_MATCH"


def _noun_alias_candidates(
    noun: str,
    ontology_labels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """List noun-family EPIC labels without implying an action mapping.

    This separate list is intentionally broader than ``_mapping_candidates``.
    It lets a reviewer see which EPIC labels share the coarse textile noun,
    while the verb/action decision remains unresolved.
    """

    family = _noun_family(noun)
    return [
        dict(row) for row in ontology_labels if _noun_family(str(row.get("noun", ""))) == family
    ]


def _surface_aliases(pair: tuple[str, str], source_label: str) -> list[str]:
    aliases = {source_label, _label(pair)}
    verb = pair[0]
    noun = pair[1]
    # Morphology only.  Deliberately do not map ``pick up`` to ``take`` or
    # ``smooth`` to ``rub``; such decisions require a production owner.
    if verb == "pick up":
        aliases.update({f"picking up {noun}", f"pickup {noun}"})
    elif verb.endswith("e"):
        aliases.add(f"{verb[:-1]}ing {noun}")
    else:
        aliases.add(f"{verb}ing {noun}")
    if noun == "garment":
        aliases.update({"garment", "clothing", "cloth", "clothes", "fabric"})
    return sorted(alias for alias in aliases if alias)


def build_provisional_vocabulary(
    review_pack: Mapping[str, Any],
    *,
    ontology: Mapping[str, Any] | None = None,
    review_artifact: str = ".agent_tmp/production_agent_reviewed_segments_4s_16f_20260827.json",
    ontology_artifact: str | None = None,
) -> dict[str, Any]:
    """Build a stable, explicitly unapproved production vocabulary artifact."""

    _assert_non_gold_review_pack(review_pack)
    items = review_pack.get("items", [])
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ProvisionalVocabularyError("review pack items must be an array")
    ontology_rows = _epic_labels(ontology)
    records: list[dict[str, Any]] = []
    record_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []

    for item_index, raw_item in enumerate(items):
        item = _mapping(raw_item, field=f"items[{item_index}]")
        window_id = _text(item.get("window_id"), field=f"items[{item_index}].window_id")
        segments = item.get("segments", [])
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes, bytearray)):
            raise ProvisionalVocabularyError(f"items[{item_index}].segments must be an array")
        window_record_ids: list[str] = []
        for segment_index, raw_segment in enumerate(segments):
            segment = _mapping(raw_segment, field=f"items[{item_index}].segments[{segment_index}]")
            parsed = _review_pair(segment, field=f"items[{item_index}].segments[{segment_index}]")
            pair = (parsed["verb"], parsed["noun"])
            record = record_by_pair.get(pair)
            if record is None:
                record_id = f"prod-provisional-{len(records) + 1:03d}"
                candidates, basis = _mapping_candidates(pair, ontology_rows)
                noun_aliases = _noun_alias_candidates(pair[1], ontology_rows)
                record = {
                    "provisional_id": record_id,
                    "status": PROVISIONAL_STATUS,
                    "source_pair": {
                        "verb": pair[0],
                        "noun": pair[1],
                        "canonical_label": _label(pair),
                    },
                    "surface_aliases": _surface_aliases(pair, parsed["surface_label"]),
                    "noun_family": {
                        "key": _noun_family(pair[1]),
                        "status": "PROVISIONAL_ALIAS_FAMILY"
                        if _noun_family(pair[1]) == "textile_generic"
                        else "EXACT_SURFACE_NOUN",
                        "approval_required": _noun_family(pair[1]) == "textile_generic",
                    },
                    "epic_projection": {
                        "status": "UNAPPROVED_CANDIDATE_MAPPING",
                        "mapping_basis": basis,
                        "candidate_action_keys": [row["action_key"] for row in candidates],
                        "candidate_labels": [row["label"] for row in candidates],
                        "noun_alias_candidates": [row["label"] for row in noun_aliases],
                        "noun_alias_action_keys": [row["action_key"] for row in noun_aliases],
                        "noun_alias_status": "UNAPPROVED_NOUN_ONLY_ALIAS",
                        "requires_human_mapping": True,
                    },
                    "observations": [],
                }
                records.append(record)
                record_by_pair[pair] = record
            observation = {
                "window_id": window_id,
                "segment": parsed,
                "reference_status": "AGENT_SURROGATE_NON_GOLD",
                "official_gold_status": "NOT_ESTABLISHED",
            }
            record["observations"].append(observation)
            window_record_ids.append(record["provisional_id"])
        windows.append(
            {
                "window_id": window_id,
                "record_ids": list(dict.fromkeys(window_record_ids)),
                "reference_status": "AGENT_SURROGATE_NON_GOLD",
                "official_gold_status": "NOT_ESTABLISHED",
                "human_adjudication": "NOT_PERFORMED",
            }
        )

    unresolved = sum(
        not bool(record["epic_projection"]["candidate_action_keys"]) for record in records
    )
    return {
        "format": PROVISIONAL_VOCABULARY_VERSION,
        "authority": AUTHORITY,
        "status": PROVISIONAL_STATUS,
        "production_eligible": False,
        "quality": {
            "measurement_status": MEASUREMENT_STATUS,
            "quality_claim": False,
            "official_gold_status": "NOT_ESTABLISHED",
            "reason": "source cohort has no independently adjudicated production action labels",
        },
        "purpose": "routing_diagnostic_and_mapping_gap_inventory",
        "source": {
            "review_artifact": review_artifact,
            "window_count": len(windows),
            "record_count": len(records),
            "reviewer_type": "AGENT_SURROGATE_VISUAL_REVIEW",
            "accepted_as_gold": False,
            "human_adjudication": "NOT_PERFORMED",
        },
        "ontology_projection": {
            "artifact": ontology_artifact,
            "catalog_label_count": len(ontology_rows),
            "mapping_status": "UNAPPROVED",
            "exact_or_family_mapped_record_count": len(records) - unresolved,
            "unresolved_record_count": unresolved,
            "note": "Projection is read-only and does not alter EPIC ontology or Mapper.",
        },
        "normalization": {
            "inflection_policy": "explicit morphology only",
            "verb_synonym_aliases": {},
            "noun_family": {
                "key": "textile_generic",
                "members": list(TEXTILE_FAMILY),
                "semantic_equivalence": "UNVERIFIED",
            },
            "no_automatic_semantic_promotion": True,
        },
        "records": records,
        "windows": windows,
        "review_requirements": [
            "approve or edit each production verb/noun pair",
            "confirm action boundaries and split compound windows where needed",
            "approve any noun-family or verb synonym mapping independently",
            "only then create a production gold/evaluator input",
        ],
        "controls": {
            "model_invoked": False,
            "source_media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "official_evaluator_invoked": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
    }


def _candidate_pair(raw: object) -> tuple[str, str] | None:
    if not isinstance(raw, Mapping):
        return None
    pair_value = raw.get("pair")
    if (
        isinstance(pair_value, Sequence)
        and not isinstance(pair_value, (str, bytes))
        and len(pair_value) == 2
        and all(isinstance(value, str) for value in pair_value)
    ):
        try:
            return (
                _normalise(pair_value[0], field="candidate.pair[0]"),
                _normalise(pair_value[1], field="candidate.pair[1]"),
            )
        except ProvisionalVocabularyError:
            return None
    try:
        return _normalise(raw.get("verb"), field="candidate.verb"), _normalise(
            raw.get("noun"), field="candidate.noun"
        )
    except ProvisionalVocabularyError:
        return None


def compare_provisional_vocabulary(
    vocabulary: Mapping[str, Any], comparator: Mapping[str, Any], *, top_k: int = 5
) -> dict[str, Any]:
    """Compute routing-only overlap against the provisional vocabulary.

    This deliberately reports ``SURROGATE_ONLY`` rather than precision/recall.
    A candidate can overlap the provisional surface pair or noun family without
    proving that the model understood the action.
    """

    if vocabulary.get("format") != PROVISIONAL_VOCABULARY_VERSION:
        raise ProvisionalVocabularyError("unexpected provisional vocabulary format")
    if vocabulary.get("status") != PROVISIONAL_STATUS:
        raise ProvisionalVocabularyError("vocabulary must remain PROVISIONAL_UNAPPROVED")
    if comparator.get("status") != "NON_GOLD_EXPLORATORY":
        raise ProvisionalVocabularyError("comparator must be NON_GOLD_EXPLORATORY")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ProvisionalVocabularyError("top_k must be a positive integer")

    records = vocabulary.get("records", [])
    windows = vocabulary.get("windows", [])
    if not isinstance(records, Sequence) or not isinstance(windows, Sequence):
        raise ProvisionalVocabularyError("vocabulary records/windows must be arrays")
    record_by_id = {
        str(row.get("provisional_id")): row
        for row in records
        if isinstance(row, Mapping) and row.get("provisional_id")
    }
    refs: dict[str, list[tuple[str, str]]] = {}
    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            continue
        wid = str(raw_window.get("window_id", ""))
        pairs: list[tuple[str, str]] = []
        for record_id in raw_window.get("record_ids", []):
            record = record_by_id.get(str(record_id))
            if not isinstance(record, Mapping):
                continue
            source = record.get("source_pair")
            if isinstance(source, Mapping):
                with suppress(ProvisionalVocabularyError):
                    pairs.append(_pair(source, field=f"record[{record_id}].source_pair"))
        if wid and pairs:
            refs[wid] = pairs

    route_payloads = comparator.get("routes", {})
    if not isinstance(route_payloads, Mapping):
        raise ProvisionalVocabularyError("comparator.routes must be an object")
    routes: dict[str, Any] = {}
    for route, payload in route_payloads.items():
        if not isinstance(payload, Mapping):
            continue
        candidate_windows = payload.get("candidate_windows", {})
        if not isinstance(candidate_windows, Mapping):
            continue
        per_window: dict[str, Any] = {}
        counts: Counter[str] = Counter()
        denomin = len(refs)
        for window_id, reference_pairs in refs.items():
            raw_candidates = candidate_windows.get(window_id, [])
            if not isinstance(raw_candidates, Sequence):
                raw_candidates = []
            candidates = [pair for pair in (_candidate_pair(x) for x in raw_candidates) if pair]
            candidates = list(dict.fromkeys(candidates))
            top = candidates[:top_k]
            strict_rank = next(
                (index for index, candidate in enumerate(top, 1) if candidate in reference_pairs),
                None,
            )
            family_rank = next(
                (
                    index
                    for index, candidate in enumerate(top, 1)
                    if any(
                        candidate[0] == reference[0]
                        and _noun_family(candidate[1]) == _noun_family(reference[1])
                        for reference in reference_pairs
                    )
                ),
                None,
            )
            verb_rank = next(
                (
                    index
                    for index, candidate in enumerate(top, 1)
                    if any(candidate[0] == reference[0] for reference in reference_pairs)
                ),
                None,
            )
            noun_rank = next(
                (
                    index
                    for index, candidate in enumerate(top, 1)
                    if any(
                        _noun_family(candidate[1]) == _noun_family(reference[1])
                        for reference in reference_pairs
                    )
                ),
                None,
            )
            for name, rank in (
                ("strict_pair", strict_rank),
                ("family_pair", family_rank),
                ("verb_only", verb_rank),
                ("noun_family_only", noun_rank),
            ):
                counts[f"{name}_top1"] += rank == 1
                counts[f"{name}_at_k"] += rank is not None
            per_window[window_id] = {
                "reference_pairs": [list(pair) for pair in reference_pairs],
                "candidate_pairs": [list(pair) for pair in top],
                "strict_pair_rank": strict_rank,
                "family_pair_rank": family_rank,
                "verb_only_rank": verb_rank,
                "noun_family_only_rank": noun_rank,
            }

        fractions = {
            name: {
                "top1": round(counts[f"{name}_top1"] / denomin, 4) if denomin else 0.0,
                "at_k": round(counts[f"{name}_at_k"] / denomin, 4) if denomin else 0.0,
            }
            for name in ("strict_pair", "family_pair", "verb_only", "noun_family_only")
        }

        routes[str(route)] = {
            "measurement_status": "SURROGATE_ONLY",
            "windows": denomin,
            "top_k": top_k,
            "metrics": fractions,
            "per_window": per_window,
        }

    return {
        "format": PROVISIONAL_COMPARISON_VERSION,
        "authority": AUTHORITY,
        "status": "AGENT_SURROGATE_MEASURED_NON_GOLD",
        "quality_claim": False,
        "official_quality_status": MEASUREMENT_STATUS,
        "reference_status": "AGENT_SURROGATE_NON_GOLD",
        "vocabulary_artifact": vocabulary.get("format"),
        "route_metrics": routes,
        "interpretation": [
            "Overlap is a routing diagnostic against an agent surrogate, not production accuracy.",
            "Family overlap keeps the verb strict and groups only the documented "
            "textile noun family.",
            "No candidate mapping is promoted to an official ontology or gold label.",
        ],
        "controls": {
            "model_invoked": False,
            "source_media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "official_evaluator_invoked": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
    }


def vocabulary_markdown(vocabulary: Mapping[str, Any]) -> str:
    """Render a concise human-readable artifact report."""

    lines = [
        "# Production provisional action vocabulary",
        "",
        "> **PROVISIONAL_UNAPPROVED.** Agent-surrogate review only; not gold and not",
        "> an evaluator input. No model inference, media decode, ontology mutation, or hash work.",
        "",
        f"- Windows: `{vocabulary.get('source', {}).get('window_count', 0)}`",
        f"- Distinct provisional records: `{vocabulary.get('source', {}).get('record_count', 0)}`",
        "- EPIC catalog candidates: "
        f"`{vocabulary.get('ontology_projection', {}).get('catalog_label_count', 0)}`",
        "- Unresolved records: "
        f"`{vocabulary.get('ontology_projection', {}).get('unresolved_record_count', 0)}`",
        "",
        "## Records",
        "",
        "| ID | Provisional pair | Surface aliases | EPIC projection |",
        "|---|---|---|---|",
    ]
    for record in vocabulary.get("records", []):
        if not isinstance(record, Mapping):
            continue
        projection = record.get("epic_projection", {})
        labels = projection.get("candidate_labels", []) if isinstance(projection, Mapping) else []
        lines.append(
            "| "
            f"{record.get('provisional_id')} | "
            f"{record.get('source_pair', {}).get('canonical_label', '—')} | "
            f"{', '.join(str(x) for x in record.get('surface_aliases', []))} | "
            f"{', '.join(str(x) for x in labels) or 'UNRESOLVED — human mapping required'} |"
        )
    lines.extend(
        [
            "",
            "## Approval boundary",
            "",
            "- `official_gold_status=NOT_ESTABLISHED`; `production_eligible=false`.",
            "- Morphology-only aliases are emitted; semantic verb synonyms are "
            "deliberately absent.",
            "- Textile noun-family matches are explicitly provisional and require owner approval.",
            "- Create production gold only after independent review confirms labels "
            "and boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def comparison_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Production provisional vocabulary routing diagnostic",
        "",
        "> **AGENT_SURROGATE_MEASURED_NON_GOLD.** This is not production precision/recall.",
        "",
        "| Route | Strict @1 | Strict @K | Family @1 | Family @K | Verb @K | Noun-family @K |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route, payload in comparison.get("route_metrics", {}).items():
        metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
        strict = metrics.get("strict_pair", {})
        family = metrics.get("family_pair", {})
        verb = metrics.get("verb_only", {})
        noun = metrics.get("noun_family_only", {})
        lines.append(
            f"| {route} | {strict.get('top1', 0.0):.1%} | "
            f"{strict.get('at_k', 0.0):.1%} | "
            f"{family.get('top1', 0.0):.1%} | {family.get('at_k', 0.0):.1%} | "
            f"{verb.get('at_k', 0.0):.1%} | {noun.get('at_k', 0.0):.1%} |"
        )
    lines.extend(
        [
            "",
            "The denominator is the ten-window agent-surrogate review queue. Keep this",
            "separate from official production quality until labels and boundaries are approved.",
            "",
        ]
    )
    return "\n".join(lines)
