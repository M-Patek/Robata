#!/usr/bin/env python3
"""Compare frozen production shadows in a provisional action space.

This is a routing diagnostic for the unlabelled ``sample-medium`` cohort.  It
projects WeMM's direct visual prototypes and Qwen's recorded prose into the
small, agent-reviewed textile vocabulary, but never treats that review as
official gold and never invokes a model.  The output is therefore explicitly
non-gold and must not be used as a production quality claim.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


_VERB_MAP = {
    "adjusts": "adjust",
    "adjusting": "adjust",
    "folds": "fold",
    "folding": "fold",
    "flattens": "flatten",
    "flattening": "flatten",
    "moves": "move",
    "moving": "move",
    "picks": "pick",
    "picking": "pick",
    "places": "place",
    "placing": "place",
    "smooths": "smooth",
    "smoothing": "smooth",
    "spreads": "spread",
    "spreading": "spread",
}
_TEXTILE_NOUNS = {
    "cloth",
    "clothes",
    "clothing",
    "fabric",
    "garment",
    "shirt",
    "sheets",
}
_ALLOWED_VERBS = {"pick up", "spread", "adjust", "flatten", "smooth", "fold"}


def _normalise(value: object) -> str:
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _verb(value: object) -> str:
    text = _normalise(value)
    if text in {"pick up", "picks up", "picking up", "pickup"}:
        return "pick up"
    tokens = text.split()
    if tokens:
        tokens[-1] = _VERB_MAP.get(tokens[-1], tokens[-1])
    return " ".join(tokens)


def _noun(value: object) -> str:
    text = _normalise(value)
    return "garment" if text in _TEXTILE_NOUNS else text


def _pair(verb: object, noun: object) -> tuple[str, str] | None:
    result = (_verb(verb), _noun(noun))
    if result[0] not in _ALLOWED_VERBS or not result[1]:
        return None
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _references(vocabulary: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    records = {
        str(row.get("provisional_id")): row
        for row in vocabulary.get("records", [])
        if isinstance(row, dict) and row.get("provisional_id")
    }
    result: dict[str, list[tuple[str, str]]] = {}
    for window in vocabulary.get("windows", []):
        if not isinstance(window, dict):
            continue
        pairs: list[tuple[str, str]] = []
        for record_id in window.get("record_ids", []):
            record = records.get(str(record_id))
            source = record.get("source_pair") if isinstance(record, dict) else None
            if isinstance(source, dict):
                pair = _pair(source.get("verb"), source.get("noun"))
                if pair is not None:
                    pairs.append(pair)
        if pairs:
            result[str(window.get("window_id"))] = list(dict.fromkeys(pairs))
    return result


def _wemm_candidates(sidecar: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for window in sidecar.get("windows", []):
        if not isinstance(window, dict):
            continue
        model = window.get("model", {})
        predictions = model.get("predictions", []) if isinstance(model, dict) else []
        candidates: list[dict[str, Any]] = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            pair = _pair(prediction.get("verb"), prediction.get("noun"))
            if pair is None:
                continue
            candidates.append(
                {
                    "verb": pair[0],
                    "noun": pair[1],
                    "score": float(prediction.get("score", 0.0) or 0.0),
                    "source": "wemm_direct_provisional_prototype",
                }
            )
        result[str(window.get("window_id"))] = candidates
    return result


def _decode_raw(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return decoded if isinstance(decoded, dict) else None


def _qwen_candidates(sidecar: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in sidecar.get("windows", []):
        if not isinstance(row, dict):
            continue
        decoded = _decode_raw(row.get("raw_text"))
        if decoded is None:
            continue
        pair = _pair(decoded.get("verb"), decoded.get("noun"))
        if pair is None:
            continue
        try:
            confidence = float(decoded.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        grouped[str(row.get("window_id"))][pair].append(confidence)

    result: dict[str, list[dict[str, Any]]] = {}
    for window_id, pairs in grouped.items():
        candidates = []
        for (verb, noun), confidences in pairs.items():
            mean_confidence = sum(confidences) / len(confidences)
            support = len(confidences) / 6.0
            candidates.append(
                {
                    "verb": verb,
                    "noun": noun,
                    "score": round(0.5 * support + 0.5 * mean_confidence, 6),
                    "support_count": len(confidences),
                    "source": "qwen_recorded_prose_projection",
                }
            )
        result[window_id] = sorted(
            candidates,
            key=lambda row: (-float(row["score"]), str(row["verb"]), str(row["noun"])),
        )
    return result


def _hybrid_candidates(
    wemm: dict[str, list[dict[str, Any]]],
    qwen: dict[str, list[dict[str, Any]]],
    *,
    wemm_weight: float = 0.65,
    qwen_weight: float = 0.35,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for window_id in set(wemm) | set(qwen):
        by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for row in wemm.get(window_id, []):
            by_pair[(row["verb"], row["noun"])] = {
                "verb": row["verb"],
                "noun": row["noun"],
                "wemm_score": float(row["score"]),
                "qwen_score": 0.0,
            }
        for row in qwen.get(window_id, []):
            item = by_pair.setdefault(
                (row["verb"], row["noun"]),
                {"verb": row["verb"], "noun": row["noun"], "wemm_score": 0.0},
            )
            item["qwen_score"] = float(row["score"])
        rows = []
        for item in by_pair.values():
            item["score"] = round(
                wemm_weight * item.get("wemm_score", 0.0)
                + qwen_weight * item.get("qwen_score", 0.0),
                6,
            )
            item["source"] = "wemm_qwen_provisional_score_fusion"
            rows.append(item)
        result[window_id] = sorted(
            rows,
            key=lambda row: (-float(row["score"]), str(row["verb"]), str(row["noun"])),
        )
    return result


def _metrics(
    references: dict[str, list[tuple[str, str]]],
    candidates: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    for window_id, reference_pairs in references.items():
        rows = candidates.get(window_id, [])[:top_k]
        pairs = [(str(row["verb"]), str(row["noun"])) for row in rows]
        rank = next(
            (index for index, pair in enumerate(pairs, 1) if pair in reference_pairs),
            None,
        )
        ranks.append(rank)
    denominator = len(ranks)
    covered = [rank for rank in ranks if rank is not None]
    return {
        "windows": denominator,
        "top_k": top_k,
        "strict_pair_at_1": sum(rank == 1 for rank in ranks) / denominator if denominator else 0.0,
        "strict_pair_at_k": len(covered) / denominator if denominator else 0.0,
        "mrr": sum(1.0 / rank for rank in covered) / denominator if denominator else 0.0,
        "abstention": sum(not candidates.get(window_id) for window_id in references) / denominator
        if denominator
        else 0.0,
        "ranks": ranks,
    }


def compare(
    vocabulary: dict[str, Any],
    wemm_sidecar: dict[str, Any],
    qwen_sidecar: dict[str, Any],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    references = _references(vocabulary)
    wemm = _wemm_candidates(wemm_sidecar)
    qwen = _qwen_candidates(qwen_sidecar)
    hybrid = _hybrid_candidates(wemm, qwen)
    routes = {"wemm": wemm, "qwen": qwen, "hybrid": hybrid}
    return {
        "format": "robata-production-provisional-route-comparison-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "AGENT_SURROGATE_MEASURED_NON_GOLD",
        "quality_claim": False,
        "official_quality_status": "NOT_MEASURED",
        "reference_status": "AGENT_SURROGATE_NON_GOLD",
        "prototype_space": "six coarse textile action labels",
        "prototype_source": "agent-surrogate review; unapproved",
        "routes": {
            name: {
                "metrics": _metrics(references, payload, top_k=top_k),
                "candidate_windows": payload,
            }
            for name, payload in routes.items()
        },
        "controls": {
            "model_invoked": False,
            "source_media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
        "interpretation": [
            "Direct WeMM prototypes test whether the Qwen prose bottleneck can be bypassed.",
            "Qwen candidates are conservative lexical projections; unknown verbs abstain.",
            "All scores are routing overlap against an agent surrogate, not production accuracy.",
        ],
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Production provisional route comparison",
        "",
        "> **AGENT_SURROGATE_MEASURED_NON_GOLD.** Official production quality remains "
        "`NOT_MEASURED`.",
        "",
        "| Route | strict @1 | strict @K | MRR | abstention |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in result.get("routes", {}).items():
        metrics = payload.get("metrics", {})
        lines.append(
            f"| {name} | {metrics.get('strict_pair_at_1', 0.0):.1%} | "
            f"{metrics.get('strict_pair_at_k', 0.0):.1%} | "
            f"{metrics.get('mrr', 0.0):.3f} | "
            f"{metrics.get('abstention', 0.0):.1%} |"
        )
    lines.extend(
        [
            "",
            "The prototype labels are provisional visual-review suggestions. They do not",
            "establish official production labels, boundaries, or model precision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--wemm", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = compare(
        _load(args.vocabulary),
        _load(args.wemm),
        _load(args.qwen),
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
