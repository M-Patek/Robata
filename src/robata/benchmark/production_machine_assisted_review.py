"""Machine-assisted production review drafts, kept separate from human gold.

The production-shaped sample currently has no supplied action annotations.  This
module provides a deliberately small bridge for the next step: combine model
claims (for example WeMM candidates and structured Qwen/Mage observations) into
an *editable draft* that a reviewer can inspect.  It never writes to the
``gold`` section and never calls a model.  A draft is therefore useful for
reducing review effort, but it is not an accuracy denominator or independent
ground truth.

The contract is benchmark-local and intentionally contains no content hash or
digest.  Evidence references are descriptive (model, camera, rank, and source
field) so the visual review surface can be regenerated without changing the
gold boundary.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final

MACHINE_ASSISTED_REVIEW_VERSION: Final = "robata-production-machine-assisted-review-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
MODEL_NAMES: Final = ("wemm", "qwen", "mage")
_GOLD_FRAGMENTS: Final = (
    "gold",
    "groundtruth",
    "officiallabel",
    "officialreference",
    "humanlabel",
    "annotation",
    "adjudication",
)
_KNOWN_PREDICTION_KEYS: Final = frozenset(
    {"verb", "noun", "action_key", "label", "candidate", "score", "rank", "confidence"}
)


class MachineAssistedReviewError(ValueError):
    """Raised when a draft input violates the label-separation contract."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MachineAssistedReviewError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MachineAssistedReviewError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MachineAssistedReviewError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MachineAssistedReviewError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise MachineAssistedReviewError(f"{field} must be a finite non-negative number")
    return number


def _normalise_key(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _assert_no_gold(value: object, *, field: str) -> None:
    """Reject official/human label fields in model-input sidecars."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise MachineAssistedReviewError(f"{field} mapping keys must be strings")
            normalised = _normalise_key(raw_key)
            if any(fragment in normalised for fragment in _GOLD_FRAGMENTS):
                raise MachineAssistedReviewError(f"{field}.{raw_key} contains gold/annotation data")
            _assert_no_gold(child, field=f"{field}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_gold(child, field=f"{field}[{index}]")
        return
    raise MachineAssistedReviewError(f"{field} must be JSON-compatible")


def _normalise_label(value: object) -> str | None:
    if isinstance(value, Mapping) and "value" in value:
        # Canonical structured envelopes represent every field as a
        # value/status object.  Read only the explicit value; status and
        # evidence are not interpreted here.
        value = value.get("value")
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text or None


def _candidate_pair(value: Mapping[str, Any]) -> tuple[str, str] | None:
    """Extract an explicit verb/noun pair without parsing free prose."""

    verb = _normalise_label(value.get("verb"))
    noun = _normalise_label(value.get("noun"))
    if verb and noun:
        return verb, noun
    # Canonical structured claims keep labels under ``structured_labels``;
    # accepting that explicit shape lets this review-only draft consume a
    # direct Qwen/Mage sidecar without parsing its evidence prose.
    for labels_key in ("structured_labels", "labels"):
        labels = value.get(labels_key)
        if isinstance(labels, Mapping):
            pair = _candidate_pair(labels)
            if pair is not None:
                return pair
    nested = value.get("candidate")
    if isinstance(nested, Mapping):
        return _candidate_pair(nested)
    nested = value.get("action")
    if isinstance(nested, Mapping):
        return _candidate_pair(nested)
    # ``action_key``/``label`` are accepted only when they have an explicit
    # two-token form.  We do not turn arbitrary prose into a label.
    for key in ("action_key", "label"):
        raw = value.get(key)
        if not isinstance(raw, str):
            continue
        pieces = re.split(r"\s*[/|:,]\s*|\s+", raw.strip(), maxsplit=1)
        if len(pieces) == 2:
            left = _normalise_label(pieces[0])
            right = _normalise_label(pieces[1])
            if left and right:
                return left, right
    return None


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    number = float(value)
    if not math.isfinite(number):
        return 1.0
    return max(0.0, min(1.0, number))


def _rank(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return rank if rank > 0 else None


def _iter_predictions(slot: Mapping[str, Any], *, model: str) -> list[dict[str, Any]]:
    raw_predictions = slot.get(
        "predictions",
        slot.get("results", slot.get("candidates", slot.get("segments", []))),
    )
    # Native structured rows commonly carry an empty ``predictions`` array
    # alongside populated segments.  Prefer an explicit non-empty segment or
    # candidate list before consulting parsed metadata.
    if (
        isinstance(raw_predictions, Sequence)
        and not isinstance(raw_predictions, (str, bytes, bytearray))
        and not raw_predictions
    ):
        for key in ("segments", "candidates", "results"):
            alternative = slot.get(key)
            if (
                isinstance(alternative, Sequence)
                and not isinstance(alternative, (str, bytes, bytearray))
                and alternative
            ):
                raw_predictions = alternative
                break
    parsed = slot.get("parsed_structured")
    if (
        (raw_predictions is None or raw_predictions == [])
        and isinstance(parsed, Mapping)
        and isinstance(parsed.get("segments"), Sequence)
        and not isinstance(parsed.get("segments"), (str, bytes, bytearray))
    ):
        raw_predictions = parsed.get("segments")
    if raw_predictions is None:
        return []
    predictions = _sequence(raw_predictions, field=f"{model}.predictions")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(predictions):
        row = _mapping(raw, field=f"{model}.predictions[{index}]")
        pair = _candidate_pair(row)
        if pair is None:
            continue
        rank = _rank(row.get("rank"))
        confidence = _confidence(row.get("confidence", row.get("score")))
        weight = confidence * (1.0 / rank if rank is not None else 1.0)
        if isinstance(row.get("verb"), str) and isinstance(row.get("noun"), str):
            source_field = "verb+noun"
        elif isinstance(row.get("structured_labels"), Mapping) or isinstance(
            row.get("labels"), Mapping
        ):
            source_field = "structured_labels"
        else:
            source_field = "candidate/action_key/label"
        rows.append(
            {
                "verb": pair[0],
                "noun": pair[1],
                "rank": rank,
                "confidence": confidence,
                "weight": weight,
                "model": model,
                "prediction_index": index,
                # Native production rows keep camera_id on the row envelope,
                # not inside each segment/candidate.  Preserve that locator so
                # reviewer evidence can identify which camera contributed it.
                "camera_id": row.get("camera_id", slot.get("camera_id")),
                "source_field": source_field,
            }
        )
    return rows


def _model_slots(sidecar: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    outputs = sidecar.get("model_outputs", sidecar.get("models"))
    if outputs is None:
        outputs = sidecar
    if not isinstance(outputs, Mapping):
        raise MachineAssistedReviewError("model_outputs must be an object")
    result: dict[str, Mapping[str, Any]] = {}
    for model in MODEL_NAMES:
        raw = outputs.get(model)
        if raw is None:
            continue
        result[model] = _mapping(raw, field=f"model_outputs.{model}")
    return result


def _direct_model_from_format(sidecar: Mapping[str, Any]) -> str | None:
    """Infer only the known model route from a direct sidecar format."""

    format_text = str(sidecar.get("format", "")).casefold()
    if "qwen" in format_text:
        return "qwen"
    if "mage" in format_text:
        return "mage"
    if "wemm" in format_text:
        return "wemm"
    return None


def _normalise_window_sidecar(
    window: Mapping[str, Any], *, direct_model: str | None
) -> Mapping[str, Any]:
    """Expose common model-slot shapes to the draft collector.

    This is a structural adapter only.  It preserves the source model's
    explicit predictions/segments and never repairs timestamps or parses
    free-form prose.
    """

    def attach_camera_id(
        slots: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]]:
        camera_id = window.get("camera_id")
        result: dict[str, Mapping[str, Any]] = {}
        for model, raw_slot in slots.items():
            slot = dict(_mapping(raw_slot, field=f"model_outputs.{model}"))
            if isinstance(camera_id, str) and camera_id.strip():
                for key in ("predictions", "results", "candidates", "segments"):
                    values = slot.get(key)
                    if not isinstance(values, Sequence) or isinstance(
                        values, (str, bytes, bytearray)
                    ):
                        continue
                    annotated: list[Any] = []
                    for value in values:
                        if isinstance(value, Mapping) and "camera_id" not in value:
                            item = dict(value)
                            item["camera_id"] = camera_id
                            annotated.append(item)
                        else:
                            annotated.append(value)
                    slot[key] = annotated
                slot.setdefault("camera_id", camera_id)
            result[str(model)] = slot
        return result

    if "model_outputs" in window:
        return attach_camera_id(_mapping(window.get("model_outputs"), field="model_outputs"))
    if "models" in window:
        return attach_camera_id(_mapping(window.get("models"), field="models"))
    if "model" in window:
        model = direct_model or "wemm"
        return attach_camera_id({model: _mapping(window.get("model"), field="model")})
    if direct_model is not None:
        slot = dict(window)
        parsed = slot.get("parsed_structured")
        if (
            "segments" not in slot
            and isinstance(parsed, Mapping)
            and isinstance(parsed.get("segments"), Sequence)
        ):
            slot["segments"] = parsed.get("segments")
        return attach_camera_id({direct_model: slot})
    # Keep the historical behavior for an explicit model-key mapping.
    return window


def _merge_model_window_maps(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Merge per-camera model slots for one source window.

    Production native sidecars record one row per camera.  Keeping only the
    last row silently discards evidence (and can turn a populated window into
    an abstention), so concatenate explicit candidate/segment arrays while
    retaining each model's runtime status.  This is a review-draft operation,
    not semantic de-duplication or timestamp repair.
    """

    result: dict[str, Mapping[str, Any]] = {
        str(model): _mapping(slot, field=f"model_outputs.{model}")
        for model, slot in left.items()
        if isinstance(slot, Mapping)
    }
    for model, raw_slot in right.items():
        slot = _mapping(raw_slot, field=f"model_outputs.{model}")
        existing = result.get(str(model))
        if existing is None:
            result[str(model)] = slot
            continue
        merged = dict(existing)
        for key in ("predictions", "results", "candidates", "segments"):
            old_values = existing.get(key)
            new_values = slot.get(key)
            old_array = (
                list(old_values)
                if isinstance(old_values, Sequence)
                and not isinstance(old_values, (str, bytes, bytearray))
                else []
            )
            new_array = (
                list(new_values)
                if isinstance(new_values, Sequence)
                and not isinstance(new_values, (str, bytes, bytearray))
                else []
            )
            if old_array or new_array or key in existing or key in slot:
                merged[key] = [*old_array, *new_array]
        old_status = str(existing.get("status", "")).upper()
        new_status = str(slot.get("status", "")).upper()
        if new_status == "SUCCEEDED" or not old_status:
            merged["status"] = slot.get("status", existing.get("status"))
        result[str(model)] = merged
    return result


def _draft_for_item(
    item: Mapping[str, Any], model_sidecar: Mapping[str, Any] | None
) -> dict[str, Any]:
    window_id = _text(item.get("window_id"), field="review item.window_id")
    start = _finite_nonnegative(item.get("start_seconds"), field=f"{window_id}.start_seconds")
    end = _finite_nonnegative(item.get("end_seconds"), field=f"{window_id}.end_seconds")
    if end <= start:
        raise MachineAssistedReviewError(
            f"{window_id}.end_seconds must be greater than start_seconds"
        )

    observations: list[dict[str, Any]] = []
    if model_sidecar is not None:
        _assert_no_gold(model_sidecar, field=f"model_sidecar[{window_id}]")
        for model, slot in _model_slots(model_sidecar).items():
            observations.extend(_iter_predictions(slot, model=model))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[(observation["verb"], observation["noun"])].append(observation)

    ranked_groups = sorted(
        grouped.items(),
        key=lambda pair: (
            -sum(float(row["weight"]) for row in pair[1]),
            -len({row["model"] for row in pair[1]}),
            pair[0],
        ),
    )
    if not ranked_groups:
        return {
            "window_id": window_id,
            "start_seconds": start,
            "end_seconds": end,
            "status": "ABSTAIN",
            "draft_type": "PROVISIONAL",
            "review_priority": "HIGH",
            "segments": [],
            "candidate_votes": [],
            "agreement": {"distinct_models": 0, "top_support_models": [], "top_weight": 0.0},
        }

    (verb, noun), support = ranked_groups[0]
    support_models = sorted({str(row["model"]) for row in support})
    priority = (
        "LOW" if len(support_models) >= 3 else "MEDIUM" if len(support_models) == 2 else "HIGH"
    )
    # This is a review-assistance score, not a calibrated probability.
    draft_confidence = min(0.95, 0.35 + 0.2 * len(support_models) + 0.1 * min(3, len(support)))
    evidence = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "model",
                "rank",
                "confidence",
                "camera_id",
                "source_field",
                "prediction_index",
            }
            and value is not None
        }
        for row in support
    ]
    segment = {
        "start_seconds": start,
        "end_seconds": end,
        "verb": verb,
        "noun": noun,
        "attributes": None,
        "location": None,
        "hand": None,
        "boundary_status": "WINDOW_BOUND_ONLY",
        "confidence": round(draft_confidence, 4),
        "evidence": evidence,
    }
    candidate_votes = [
        {
            "verb": pair[0],
            "noun": pair[1],
            "support_models": sorted({str(row["model"]) for row in rows}),
            "support_count": len(rows),
            "weight": round(sum(float(row["weight"]) for row in rows), 6),
        }
        for pair, rows in ranked_groups
    ]
    return {
        "window_id": window_id,
        "start_seconds": start,
        "end_seconds": end,
        "status": "MACHINE_ASSISTED_DRAFT",
        "draft_type": "PROVISIONAL",
        "review_priority": priority,
        "segments": [segment],
        "candidate_votes": candidate_votes,
        "agreement": {
            "distinct_models": len(support_models),
            "top_support_models": support_models,
            "top_weight": round(sum(float(row["weight"]) for row in support), 6),
        },
    }


def build_machine_assisted_review(
    review_pack: Mapping[str, Any],
    model_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a separate machine-assisted draft sidecar.

    ``review_pack`` is only used for source/window geometry.  Its gold sections
    are not copied, read, or modified.  ``model_sidecar`` may contain claims,
    but any field that looks like official/human annotation is rejected.
    """

    pack = _mapping(review_pack, field="review_pack")
    if pack.get("format") != "robata-production-human-review-pack-v1":
        raise MachineAssistedReviewError("unsupported review pack format")
    items = _sequence(pack.get("items"), field="review_pack.items")
    sidecar_windows: dict[str, Mapping[str, Any]] = {}
    direct_model: str | None = None
    if model_sidecar is not None:
        sidecar = _mapping(model_sidecar, field="model_sidecar")
        direct_model = _direct_model_from_format(sidecar)
        raw_windows = sidecar.get("windows")
        if raw_windows is not None:
            for index, raw in enumerate(_sequence(raw_windows, field="model_sidecar.windows")):
                window = _mapping(raw, field=f"model_sidecar.windows[{index}]")
                window_id = _text(
                    window.get("window_id"),
                    field=f"model_sidecar.windows[{index}].window_id",
                )
                normalized = _normalise_window_sidecar(window, direct_model=direct_model)
                previous = sidecar_windows.get(window_id)
                sidecar_windows[window_id] = (
                    normalized
                    if previous is None
                    else _merge_model_window_maps(previous, normalized)
                )
        else:
            sidecar_windows = {
                str(key): _mapping(value, field=f"model_sidecar.{key}")
                for key, value in sidecar.items()
                if key in MODEL_NAMES
            }

    drafts: list[dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        item = _mapping(raw_item, field=f"review_pack.items[{index}]")
        window_id = _text(item.get("window_id"), field=f"review_pack.items[{index}].window_id")
        window_sidecar = sidecar_windows.get(window_id)
        drafts.append(_draft_for_item(item, window_sidecar))

    return {
        "format": MACHINE_ASSISTED_REVIEW_VERSION,
        "authority": AUTHORITY,
        "source_review_pack_format": pack.get("format"),
        "items": drafts,
        "review_contract": {
            "draft_status": "MACHINE_ASSISTED_DRAFT",
            "provisional_status": "PROVISIONAL",
            "status_is_provisional": True,
            "gold_written": False,
            "independent_review_required": True,
            "window_boundaries_are_not_action_boundaries": True,
            "unobserved_fields": ["attributes", "location", "hand"],
        },
        "controls": {
            "machine_assistance_used": model_sidecar is not None,
            "gold_read": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "sha_or_digest_computed": False,
        },
    }


__all__ = [
    "AUTHORITY",
    "MACHINE_ASSISTED_REVIEW_VERSION",
    "MachineAssistedReviewError",
    "build_machine_assisted_review",
]
