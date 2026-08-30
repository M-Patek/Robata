"""Production-native WeMM vocabulary retrieval (benchmark-local).

The original production WeMM shadow accepts an EPIC ``(verb_id, noun_id)``
catalog.  That is useful for the EPIC experiment, but it is the wrong label
space for a production recording whose labels are defined by the owner/Terra
review process.  This module is a deliberately separate adapter which runs
the same native WeMM video encoder against an explicitly supplied *production
vocabulary*.

The vocabulary is only a routing aid.  In particular, an owner-approved
coarse vocabulary is not official gold and cannot make a quality claim.  The
adapter does not mutate the EPIC ontology or Mapper and does not copy model
predictions into a gold artifact.  It preserves per-camera Top-K evidence so a
later Qwen/Mage structured annotation or human review can consume the
candidates.

No identity/hash/digest work is performed here.  Media decoding and model
inference are opt-in through :func:`run_production_wemm_vocabulary_shadow`;
the pure validation/ranking helpers remain runnable without optional media
dependencies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from .production_wemm_shadow import decode_production_windows
from .wemm_action_retrieval import cosine_similarity, render_action_label_texts
from .wemm_embedding_backend import WemmEmbeddingBackend
from .wemm_multiview_retrieval import fuse_camera_rankings

PRODUCTION_WEMM_VOCABULARY_VERSION: Final = "robata-production-wemm-vocabulary-shadow-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
VOCABULARY_FORMAT: Final = "robata-production-coarse-vocabulary-owner-approval-v1"
VOCABULARY_PROFILE: Final = "PRODUCTION_OWNER_APPROVED_COARSE_VOCABULARY"
QUALITY_STATUS: Final = "NOT_MEASURED"
LABEL_VARIANTS: Final = ("canonical", "verb_noun", "natural")
LabelVariant = Literal["canonical", "verb_noun", "natural"]


class ProductionWemmVocabularyError(ValueError):
    """Raised when a production vocabulary or retrieval input is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmVocabularyError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmVocabularyError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmVocabularyError(f"{field} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ProductionWemmVocabularyError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionWemmVocabularyError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ProductionWemmVocabularyError(f"{field} must be finite")
    return number


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmVocabularyError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ProductionVocabularyLabel:
    """One owner-scoped production label and its deterministic text surfaces."""

    label_id: str
    verb_code: str
    verb: str
    noun: str
    canonical_label: str
    texts: tuple[tuple[str, str], ...]

    def text_for(self, variant: str) -> str:
        if variant not in LABEL_VARIANTS:
            raise ProductionWemmVocabularyError(f"unsupported label variant: {variant!r}")
        for key, value in self.texts:
            if key == variant:
                return value
        raise ProductionWemmVocabularyError(f"label has no text variant: {variant!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "verb_code": self.verb_code,
            "verb": self.verb,
            "noun": self.noun,
            "canonical_label": self.canonical_label,
            "texts": {key: value for key, value in self.texts},
        }


@dataclass(frozen=True, slots=True)
class RetrievedProductionLabel:
    """A ranked production-vocabulary candidate."""

    rank: int
    label: ProductionVocabularyLabel
    label_variant: str
    visual_cosine: float
    visual_score: float

    @property
    def label_id(self) -> str:
        return self.label.label_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "label_id": self.label.label_id,
            "verb_code": self.label.verb_code,
            "verb": self.label.verb,
            "noun": self.label.noun,
            "label_text": self.label.text_for(self.label_variant),
            "canonical_label": self.label.canonical_label,
            "label_variant": self.label_variant,
            "visual_cosine": self.visual_cosine,
            "visual_score": self.visual_score,
            "fused_score": self.visual_score,
        }


def _load_json(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionWemmVocabularyError(
                f"could not read production vocabulary {path}: {exc}"
            ) from exc
        return _mapping(payload, field="production_vocabulary")
    return value


def _validate_vocabulary_envelope(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the owner boundary without treating it as gold."""

    if document.get("format") != VOCABULARY_FORMAT:
        raise ProductionWemmVocabularyError(
            f"production vocabulary format must be {VOCABULARY_FORMAT!r}"
        )
    if document.get("authority") != AUTHORITY:
        raise ProductionWemmVocabularyError("production vocabulary authority is not local-only")
    if document.get("owner_approved") is not True:
        raise ProductionWemmVocabularyError("production vocabulary is not owner-approved")
    # Owner approval allows a local routing experiment only; these guards keep
    # a later artifact from silently becoming an evaluator/gold input.
    for key in ("production_eligible", "official_gold", "accepted_as_gold"):
        if document.get(key) is not False:
            raise ProductionWemmVocabularyError(f"production vocabulary {key} must remain false")
    if document.get("official_gold_status") not in {None, "NOT_ESTABLISHED"}:
        raise ProductionWemmVocabularyError(
            "production vocabulary unexpectedly claims official gold"
        )
    body = _mapping(document.get("vocabulary"), field="production_vocabulary.vocabulary")
    return body


def load_production_vocabulary(
    value: Mapping[str, Any] | str | Path,
) -> tuple[tuple[ProductionVocabularyLabel, ...], dict[str, Any]]:
    """Load and validate an owner-scoped production vocabulary.

    The returned labels use ``verb_code`` as their opaque local identifier;
    no EPIC class IDs are introduced.  The second return value is provenance
    metadata suitable for a sidecar.
    """

    document = _load_json(value)
    body = _validate_vocabulary_envelope(document)
    raw_pairs = _sequence(
        body.get("verb_noun_pairs"), field="production_vocabulary.verb_noun_pairs"
    )
    labels: list[ProductionVocabularyLabel] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_pairs):
        row = _mapping(raw, field=f"verb_noun_pairs[{index}]")
        verb = _text(row.get("verb"), field=f"verb_noun_pairs[{index}].verb")
        noun = _text(row.get("noun"), field=f"verb_noun_pairs[{index}].noun")
        verb_code = _text(row.get("verb_code"), field=f"verb_noun_pairs[{index}].verb_code")
        canonical = row.get("canonical_label")
        canonical_label = (
            _text(canonical, field=f"verb_noun_pairs[{index}].canonical_label")
            if canonical is not None
            else f"{verb} {noun}"
        )
        pair = (verb.casefold(), noun.casefold())
        if verb_code in seen_ids or pair in seen_pairs:
            raise ProductionWemmVocabularyError(
                "production vocabulary contains duplicate verb_code or verb/noun pair"
            )
        seen_ids.add(verb_code)
        seen_pairs.add(pair)
        try:
            rendered = render_action_label_texts(verb, noun)
        except Exception as exc:
            raise ProductionWemmVocabularyError(f"could not render label {verb_code!r}") from exc
        # The owner may choose a production-facing canonical surface that is
        # more specific than the normalized ``verb noun`` pair.  Keep that
        # surface authoritative for the canonical embedding while retaining
        # the deterministic generic variants for the representation sweep.
        rendered["canonical"] = canonical_label
        labels.append(
            ProductionVocabularyLabel(
                label_id=verb_code,
                verb_code=verb_code,
                verb=verb,
                noun=noun,
                canonical_label=canonical_label,
                texts=tuple((name, rendered[name]) for name in LABEL_VARIANTS),
            )
        )
    if not labels:
        raise ProductionWemmVocabularyError("production vocabulary contains no labels")
    source = document.get("source")
    source_meta = dict(source) if isinstance(source, Mapping) else {}
    return tuple(labels), {
        "format": document.get("format"),
        "status": document.get("status"),
        "owner_approved": True,
        "production_eligible": False,
        "official_gold_status": document.get("official_gold_status", "NOT_ESTABLISHED"),
        "approval_scope": document.get("approval_scope"),
        "source": source_meta,
        "pair_count": len(labels),
        "labels": [label.to_dict() for label in labels],
    }


def rank_production_vocabulary(
    labels: Sequence[ProductionVocabularyLabel],
    *,
    query_embedding: Sequence[float],
    label_embeddings: Mapping[str, Sequence[float]],
    label_variant: LabelVariant = "canonical",
    top_k: int | None = None,
) -> tuple[RetrievedProductionLabel, ...]:
    """Rank production labels by visual cosine similarity."""

    if label_variant not in LABEL_VARIANTS:
        raise ProductionWemmVocabularyError(f"unsupported label variant: {label_variant!r}")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)):
        raise ProductionWemmVocabularyError("labels must be a sequence")
    if not labels:
        raise ProductionWemmVocabularyError("labels must not be empty")
    if top_k is not None:
        top_k = _positive_int(top_k, field="top_k")
    ranked: list[RetrievedProductionLabel] = []
    for label in labels:
        if label.label_id not in label_embeddings:
            raise ProductionWemmVocabularyError(f"missing label embedding for {label.label_id!r}")
        try:
            cosine = cosine_similarity(query_embedding, label_embeddings[label.label_id])
        except Exception as exc:
            raise ProductionWemmVocabularyError(
                f"invalid embedding for {label.label_id!r}: {exc}"
            ) from exc
        visual_score = (cosine + 1.0) / 2.0
        ranked.append(
            RetrievedProductionLabel(
                rank=0,
                label=label,
                label_variant=label_variant,
                visual_cosine=cosine,
                visual_score=visual_score,
            )
        )
    ranked.sort(key=lambda row: (-row.visual_score, row.label.label_id))
    limit = len(ranked) if top_k is None else min(top_k, len(ranked))
    return tuple(
        RetrievedProductionLabel(
            rank=index,
            label=row.label,
            label_variant=row.label_variant,
            visual_cosine=row.visual_cosine,
            visual_score=row.visual_score,
        )
        for index, row in enumerate(ranked[:limit], 1)
    )


def _prediction_row(item: RetrievedProductionLabel, *, camera_id: str) -> dict[str, Any]:
    row = item.to_dict()
    row.update(
        {
            "camera_id": camera_id,
            "source": "wemm_visual_embedding_production_vocabulary",
        }
    )
    return row


def _fuse_predictions(
    per_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    labels_by_id: Mapping[str, ProductionVocabularyLabel],
    camera_order: Sequence[str],
    top_k: int,
    fusion: str,
    score_normalization: str,
    label_variant: LabelVariant,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse rankings while keeping opaque production label IDs.

    ``fuse_camera_rankings`` accepts string ``candidate_id`` values, so no
    synthetic EPIC verb/noun IDs are needed for this route.
    """

    payload = {
        camera_id: {
            "candidates": [
                {
                    "candidate_id": str(row["label_id"]),
                    "rank": row.get("rank"),
                    "score": row.get("visual_score", row.get("fused_score")),
                    "verb": row.get("verb"),
                    "noun": row.get("noun"),
                    "label_text": row.get("label_text"),
                }
                for row in rows
            ]
        }
        for camera_id, rows in per_camera.items()
    }
    try:
        fused = fuse_camera_rankings(
            payload,
            camera_order=camera_order,
            expected_cameras=camera_order,
            top_k=top_k,
            fusion=fusion,
            score_normalization=score_normalization,
            missing_score="omit",
            include_embeddings=False,
        )
    except Exception as exc:
        raise ProductionWemmVocabularyError(f"production camera fusion failed: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for raw in fused.get("candidates", ()):
        candidate = _mapping(raw, field="fused.candidate")
        label_id = _text(candidate.get("action_key"), field="fused.candidate.action_key")
        label = labels_by_id.get(label_id)
        if label is None:
            raise ProductionWemmVocabularyError(
                f"fused ranking returned unknown production label {label_id!r}"
            )
        rows.append(
            {
                "rank": candidate.get("rank"),
                "label_id": label.label_id,
                "verb_code": label.verb_code,
                "verb": label.verb,
                "noun": label.noun,
                "label_text": label.text_for(label_variant),
                "canonical_label": label.canonical_label,
                "label_variant": label_variant,
                "score": candidate.get("fused_score", candidate.get("score")),
                "fused_score": candidate.get("fused_score", candidate.get("score")),
                "camera_coverage": candidate.get("camera_coverage"),
                "camera_coverage_fraction": candidate.get("camera_coverage_fraction"),
                "source": "wemm_multiview_production_vocabulary_fusion",
            }
        )
    return rows, fused


def run_production_wemm_vocabulary_shadow(
    manifest: Mapping[str, Any] | str | Path,
    *,
    vocabulary: Mapping[str, Any] | str | Path,
    model_directory: str | Path,
    frame_count: int = 4,
    top_k: int = 10,
    dimension: int = 2048,
    device: str = "cuda",
    label_variant: LabelVariant = "canonical",
    max_windows: int | None = None,
    fusion: str = "mean",
    score_normalization: str = "unit",
    validate_crcs: bool = False,
) -> dict[str, Any]:
    """Run native WeMM retrieval against a production vocabulary.

    The result is an exploratory sidecar.  It deliberately contains no EPIC
    class IDs and leaves ``production_eligible`` false.
    """

    manifest_doc = _load_json(manifest)
    labels, vocabulary_meta = load_production_vocabulary(vocabulary)
    frame_count = _positive_int(frame_count, field="frame_count")
    if frame_count < 2 or frame_count > 64:
        raise ProductionWemmVocabularyError("frame_count must be between 2 and 64")
    top_k = _positive_int(top_k, field="top_k")
    dimension = _positive_int(dimension, field="dimension")
    if not isinstance(device, str) or not device.strip():
        raise ProductionWemmVocabularyError("device must be non-empty")
    if label_variant not in LABEL_VARIANTS:
        raise ProductionWemmVocabularyError(f"unsupported label variant: {label_variant!r}")
    if not isinstance(validate_crcs, bool):
        raise ProductionWemmVocabularyError("validate_crcs must be boolean")
    if max_windows is not None:
        max_windows = _positive_int(max_windows, field="max_windows")

    # ``decode_production_windows`` performs the source-bound six-camera
    # validation used by the existing shadow route.  The model path remains
    # native and bounded; no re-encoded temporary movie is introduced.
    raw_windows = manifest_doc.get("windows")
    windows = _sequence(raw_windows, field="manifest.windows")
    if max_windows is not None:
        windows = windows[:max_windows]
    if not windows:
        raise ProductionWemmVocabularyError("manifest has no selected windows")
    bounded_manifest = {**dict(manifest_doc), "windows": list(windows)}
    try:
        groups = decode_production_windows(
            bounded_manifest, frame_count=frame_count, validate_crcs=validate_crcs
        )
    except Exception as exc:
        if isinstance(exc, ProductionWemmVocabularyError):
            raise
        raise ProductionWemmVocabularyError(f"production media decode failed: {exc}") from exc

    backend = WemmEmbeddingBackend(
        model_directory=model_directory,
        device=device,
        dimension=dimension,
    )
    try:
        label_texts = [label.text_for(label_variant) for label in labels]
        label_vectors = backend.encode_texts(label_texts, batch_size=32)
        if len(label_vectors) != len(labels):
            raise ProductionWemmVocabularyError(
                f"WeMM returned {len(label_vectors)} label vectors; expected {len(labels)}"
            )
        label_embeddings = {
            label.label_id: vector for label, vector in zip(labels, label_vectors, strict=True)
        }
        labels_by_id = {label.label_id: label for label in labels}
        camera_order = tuple(sorted(groups))
        output_windows: list[dict[str, Any]] = []
        for raw_window in windows:
            window = _mapping(raw_window, field="manifest.windows[]")
            window_id = _text(window.get("window_id"), field="window_id")
            ordinal = window.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                ordinal = len(output_windows)
            start = _finite(window.get("start_seconds"), field=f"{window_id}.start_seconds")
            end = _finite(window.get("end_seconds"), field=f"{window_id}.end_seconds")
            if end <= start:
                raise ProductionWemmVocabularyError(f"{window_id} has an invalid interval")
            per_camera: dict[str, list[dict[str, Any]]] = {}
            input_observations: list[dict[str, Any]] = []
            for camera_id in camera_order:
                try:
                    group = groups[camera_id][window_id]
                except KeyError as exc:
                    raise ProductionWemmVocabularyError(
                        f"missing decoded group for {camera_id}/{window_id}"
                    ) from exc
                query_vector = backend.encode_video_frames(
                    [group.frames], metadata_groups=[group.metadata()]
                )[0]
                ranked = rank_production_vocabulary(
                    labels,
                    query_embedding=query_vector,
                    label_embeddings=label_embeddings,
                    label_variant=label_variant,
                    top_k=top_k,
                )
                per_camera[camera_id] = [
                    _prediction_row(item, camera_id=camera_id) for item in ranked
                ]
                observation = group.to_dict()
                if backend.observations:
                    observation["model_observation"] = backend.observations[-1].to_dict()
                input_observations.append(observation)
            fused_predictions, fusion_report = _fuse_predictions(
                per_camera,
                labels_by_id=labels_by_id,
                camera_order=camera_order,
                top_k=top_k,
                fusion=fusion,
                score_normalization=score_normalization,
                label_variant=label_variant,
            )
            output_windows.append(
                {
                    "ordinal": ordinal,
                    "window_id": window_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "model": {
                        "model": "wemm",
                        "native_route": "complete_bounded_video_embedding_production_vocabulary",
                        "status": "SUCCEEDED",
                        "predictions": fused_predictions,
                        "per_camera_predictions": per_camera,
                        "input_observations": input_observations,
                        "fusion": fusion_report,
                    },
                }
            )
        source = manifest_doc.get("source")
        source_map = dict(source) if isinstance(source, Mapping) else {}
        source_path = source_map.get("path")
        return {
            "format": PRODUCTION_WEMM_VOCABULARY_VERSION,
            "authority": AUTHORITY,
            "status": "SUCCEEDED",
            "official_quality_status": QUALITY_STATUS,
            "official_gold_status": "NOT_ESTABLISHED",
            "quality_claim": False,
            "production_eligible": False,
            "source": {
                "path": source_path,
                "manifest_format": manifest_doc.get("format"),
                "window_count": len(output_windows),
                "camera_count": len(camera_order),
            },
            "model": {
                "identifier": "WeMM-Embedding-2B",
                "model_directory": str(Path(model_directory).expanduser().resolve()),
                "dimension": dimension,
                "label_variant": label_variant,
                "frame_count": frame_count,
            },
            "vocabulary": {
                **vocabulary_meta,
                "profile": VOCABULARY_PROFILE,
                "epic_ontology_used": False,
                "mapper_used": False,
            },
            "windows": output_windows,
            "quality": {
                "measurement_status": QUALITY_STATUS,
                "reason": (
                    "production vocabulary is owner-scoped non-gold; independent "
                    "human gold is not established"
                ),
                "values": {},
            },
            "controls": {
                "model_invoked": True,
                "gold_included": False,
                "predictions_are_gold": False,
                "existing_mapper_invoked": False,
                "ontology_modified": False,
                "mapper_modified": False,
                "training_invoked": False,
                "heldout_100_opened": False,
                "hash_or_sha_used": False,
                "ground_truth_used_in_encoder_input": False,
            },
            "limitations": [
                "The owner-approved vocabulary is a routing aid, not official gold.",
                (
                    "Top-K retrieval does not produce attributes, location, hand, "
                    "or action boundaries."
                ),
                "No EPIC ontology or Mapper was used or modified.",
                "Scores are embedding similarities, not calibrated probabilities.",
            ],
            "backend_observations": backend.observation_payload(),
        }
    finally:
        backend.close()


__all__ = [
    "AUTHORITY",
    "LABEL_VARIANTS",
    "PRODUCTION_WEMM_VOCABULARY_VERSION",
    "ProductionVocabularyLabel",
    "ProductionWemmVocabularyError",
    "RetrievedProductionLabel",
    "load_production_vocabulary",
    "rank_production_vocabulary",
    "run_production_wemm_vocabulary_shadow",
]
