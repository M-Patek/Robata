"""Run a review-only WeMM retrieval pass over an open phrase catalog.

The production corpus does not have an EPIC action ontology.  This benchmark
seam therefore accepts a small, owner/Terra-facing phrase catalog and keeps the
catalog identifiers opaque (``provisional_id``).  A bounded six-camera video is
encoded natively by :class:`~robata.benchmark.wemm_embedding_backend.WemmEmbeddingBackend`,
text prototypes are encoded once, and camera rankings are fused before being
wrapped in the review-only pre-annotation envelope.

This module deliberately does *not* invoke Qwen/Mage, infer action boundaries,
read or write gold, call the Mapper, or alter a production ontology.  The
serial schedule is the default; an explicit ``include_pipeline`` option only
overlaps bounded decode and inference and leaves the envelope unchanged.  The
``dry_run_open_phrase_plan`` helper performs only manifest/catalog validation so
that a full 37-recording plan can be inspected without loading a model.
"""

from __future__ import annotations

import gc
import json
import math
import re
import threading
import unicodedata
from collections.abc import Hashable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .production_wemm_decode_cache import ProductionWemmDecodeCache
from .production_wemm_preannotation import (
    ProductionWemmPreannotationError,
    build_preannotation_envelope,
    validate_preannotation_envelope,
)
from .production_wemm_shadow import (
    decode_production_windows,
    iter_decode_production_window_chunks,
)
from .wemm_action_retrieval import cosine_similarity
from .wemm_embedding_backend import WemmEmbeddingBackend
from .wemm_multiview_retrieval import fuse_camera_rankings
from .wemm_pipeline_benchmark import PipelinePhase, run_bounded_pipeline

OPEN_RUN_FORMAT: Final = "robata-production-wemm-open-run-v1"
PHRASE_CATALOG_FORMAT: Final = "robata-production-open-phrase-catalog-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
LABEL_VARIANTS: Final = ("canonical", "verb_noun", "natural")
CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
DEFAULT_QUEUE_CAPACITY: Final = 1

# Keep a reference to the production implementations so unit tests and
# downstream callers that historically monkeypatch ``decode_production_windows``
# continue to work.  The normal path uses the source-bound iterator below;
# a replaced legacy symbol is treated as an explicit compatibility seam.
_DEFAULT_DECODE_PRODUCTION_WINDOWS = decode_production_windows
_DEFAULT_ITER_DECODE_PRODUCTION_WINDOW_CHUNKS = iter_decode_production_window_chunks


class ProductionWemmOpenRunnerError(RuntimeError):
    """Raised when an open-vocabulary WeMM run cannot be prepared or completed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionWemmOpenRunnerError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProductionWemmOpenRunnerError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProductionWemmOpenRunnerError(f"{field} must be a string")
    result = unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise ProductionWemmOpenRunnerError(f"{field} must be non-empty")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ProductionWemmOpenRunnerError(f"{field} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmOpenRunnerError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ProductionWemmOpenRunnerError(f"{field} must be finite")
    return result


def _optional_positive_int(value: object, *, field: str) -> int | None:
    """Validate an optional integer resize override before media work."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWemmOpenRunnerError(f"{field} must be a positive integer or None")
    return value


def _json_copy(value: object, *, field: str = "value") -> Any:
    """Copy JSON-shaped metadata and reject unserialisable objects early."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionWemmOpenRunnerError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProductionWemmOpenRunnerError(f"{field} keys must be strings")
            result[key] = _json_copy(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_copy(child, field=f"{field}[{index}]") for index, child in enumerate(value)]
    raise ProductionWemmOpenRunnerError(f"{field} must be JSON-compatible")


def _close_decoded_group_images(groups: Mapping[str, Mapping[str, Any]] | None) -> None:
    """Release bounded decoded PIL images after their embeddings are emitted.

    The review envelope stores observations and rankings, never pixels.  PIL
    image cores can otherwise remain resident through a loop-local frame-group
    reference until the next collection cycle on long recordings.
    """

    if groups is None:
        return
    for camera_groups in groups.values():
        if not isinstance(camera_groups, Mapping):
            continue
        for frame_group in camera_groups.values():
            frames = getattr(frame_group, "frames", ())
            if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes, bytearray)):
                continue
            for image in frames:
                close = getattr(image, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()


def _expand_model_observations(
    observations: Sequence[Any], *, expected_count: int
) -> tuple[dict[str, Any], ...]:
    """Pair batch-level backend telemetry with flattened input rows.

    ``WemmEmbeddingBackend`` emits one observation per processor/model call,
    while the review envelope stores one input observation per camera/window.
    Repeating a batch observation for its members is intentional; the added
    ``batch_member_index`` makes the association explicit and avoids the old
    bug where every row received the *last* batch's telemetry.

    A custom lightweight backend may expose no observations at all.  In that
    case return an empty tuple and let the caller omit the optional field.
    If a backend reports an inconsistent item count, also omit the pairing
    rather than writing misleading metadata into the envelope.
    """

    if not observations:
        return ()
    expanded: list[dict[str, Any]] = []
    for observation_index, observation in enumerate(observations):
        to_dict = getattr(observation, "to_dict", None)
        raw = to_dict() if callable(to_dict) else observation
        if not isinstance(raw, Mapping):
            return ()
        payload = dict(_json_copy(raw, field="backend_observation"))
        raw_count = payload.get("item_count", 1)
        if isinstance(raw_count, bool):
            return ()
        try:
            item_count = int(raw_count)
        except (TypeError, ValueError, OverflowError):
            return ()
        if item_count <= 0:
            return ()
        for member_index in range(item_count):
            member = dict(payload)
            if item_count > 1:
                member["batch_member_index"] = member_index
                member["batch_observation_index"] = observation_index
            expanded.append(member)
    if len(expanded) != expected_count:
        return ()
    return tuple(expanded)


def _load_json(value: Mapping[str, Any] | Sequence[Any] | str | Path) -> Any:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionWemmOpenRunnerError(f"could not read JSON {path}: {exc}") from exc
    return value


def _key_token(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _assert_non_epic_catalog(value: object, *, field: str = "catalog") -> None:
    """Reject obvious EPIC/gold identity fields without hashing or deep policy."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProductionWemmOpenRunnerError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ProductionWemmOpenRunnerError(f"{field} keys must be strings")
            token = _key_token(raw_key)
            if token in {
                "epicontology",
                "epicactionkey",
                "actionid",
                "verbid",
                "nounid",
                "ontologyid",
                "groundtruth",
                "officiallabel",
                "humanannotation",
                "adjudicatedlabel",
            }:
                raise ProductionWemmOpenRunnerError(
                    f"{field}.{raw_key} is an EPIC/gold identity; use provisional_id"
                )
            if token == "epicontologyused" and child is True:
                raise ProductionWemmOpenRunnerError(f"{field}.{raw_key} declares EPIC ontology use")
            if token in {"actionkey", "verbkey", "nounkey"}:
                if isinstance(child, int) and not isinstance(child, bool):
                    raise ProductionWemmOpenRunnerError(
                        f"{field}.{raw_key} contains a numeric EPIC identity"
                    )
                if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                    values = list(child)
                    if values and all(
                        isinstance(item, int) and not isinstance(item, bool) for item in values
                    ):
                        raise ProductionWemmOpenRunnerError(
                            f"{field}.{raw_key} contains a numeric EPIC identity"
                        )
            _assert_non_epic_catalog(child, field=f"{field}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_non_epic_catalog(child, field=f"{field}[{index}]")
        return
    raise ProductionWemmOpenRunnerError(f"{field} must be JSON-compatible")


def _slug(value: str, index: int) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return f"phrase_{index + 1:03d}_{token or 'unknown'}"


@dataclass(frozen=True, slots=True)
class OpenPhrase:
    """One provisional production phrase and its optional structured hints."""

    provisional_id: str
    label_text: str
    texts: tuple[tuple[str, str], ...]
    structured_labels: dict[str, Any]

    def text_for(self, variant: str) -> str:
        if variant not in LABEL_VARIANTS:
            raise ProductionWemmOpenRunnerError(f"unsupported label variant: {variant!r}")
        for key, value in self.texts:
            if key == variant:
                return value
        return self.label_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "provisional_id": self.provisional_id,
            "label_text": self.label_text,
            "texts": {key: value for key, value in self.texts},
            "structured_labels": _json_copy(self.structured_labels),
        }


def _phrase_rows(document: object) -> tuple[Sequence[Any], Mapping[str, Any]]:
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes, bytearray)):
        return document, {"format": PHRASE_CATALOG_FORMAT, "source": {}}
    body = _mapping(document, field="phrase_catalog")
    # ``phrases`` is the canonical shape; a compact ``labels``/``candidates``
    # alias makes it easy to bootstrap from a Terra review export.
    rows: object = body.get("phrases")
    if rows is None:
        rows = body.get("labels", body.get("candidates"))
    if rows is None:
        nested = body.get("vocabulary")
        if isinstance(nested, Mapping):
            rows = nested.get("phrases", nested.get("labels"))
    if rows is None:
        raise ProductionWemmOpenRunnerError(
            "phrase_catalog must contain phrases (or labels/candidates)"
        )
    return _sequence(rows, field="phrase_catalog.phrases"), body


def load_open_phrase_catalog(
    value: Mapping[str, Any] | Sequence[Any] | str | Path,
) -> tuple[tuple[OpenPhrase, ...], dict[str, Any]]:
    """Load an open phrase catalog and retain only non-identity metadata."""

    document = _load_json(value)
    _assert_non_epic_catalog(document)
    rows, envelope = _phrase_rows(document)
    labels: list[OpenPhrase] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for index, raw in enumerate(rows):
        if isinstance(raw, str):
            row: Mapping[str, Any] = {"label_text": raw}
        else:
            row = _mapping(raw, field=f"phrase_catalog.phrases[{index}]")
        text_value: object | None = None
        for key in ("label_text", "phrase", "text", "canonical_label", "label", "name"):
            if row.get(key) is not None:
                text_value = row.get(key)
                break
        label_text = _text(text_value, field=f"phrases[{index}].label_text")
        normalized_text = label_text.casefold()
        if normalized_text in seen_texts:
            raise ProductionWemmOpenRunnerError(
                f"phrase catalog contains duplicate label text: {label_text!r}"
            )
        seen_texts.add(normalized_text)
        raw_id = row.get("provisional_id", row.get("id", row.get("label_id")))
        provisional_id = (
            _text(raw_id, field=f"phrases[{index}].provisional_id")
            if raw_id is not None
            else _slug(label_text, index)
        )
        if provisional_id in seen_ids:
            raise ProductionWemmOpenRunnerError(
                f"phrase catalog contains duplicate provisional_id: {provisional_id!r}"
            )
        seen_ids.add(provisional_id)
        raw_texts = row.get("texts", {})
        texts: dict[str, str] = {"canonical": label_text}
        if isinstance(raw_texts, Mapping):
            for variant in LABEL_VARIANTS:
                candidate = raw_texts.get(variant)
                if candidate is not None:
                    texts[variant] = _text(candidate, field=f"phrases[{index}].texts.{variant}")
        for variant in LABEL_VARIANTS:
            candidate = row.get(variant)
            if candidate is not None:
                texts[variant] = _text(candidate, field=f"phrases[{index}].{variant}")
        structured = row.get("structured_labels", {})
        if structured is None:
            structured = {}
        structured_mapping = dict(_mapping(structured, field=f"phrases[{index}].structured_labels"))
        # Keep only the contract fields; values are hints and remain reviewable.
        structured_mapping = {
            key: _json_copy(
                structured_mapping[key],
                field=f"phrases[{index}].structured_labels.{key}",
            )
            for key in ("verb", "noun", "attributes", "location", "hand")
            if key in structured_mapping
        }
        labels.append(
            OpenPhrase(
                provisional_id=provisional_id,
                label_text=label_text,
                texts=tuple(
                    (variant, texts.get(variant, label_text)) for variant in LABEL_VARIANTS
                ),
                structured_labels=structured_mapping,
            )
        )
    if not labels:
        raise ProductionWemmOpenRunnerError("phrase catalog contains no phrases")
    metadata: dict[str, Any] = {
        "format": envelope.get("format", PHRASE_CATALOG_FORMAT),
        "authority": envelope.get("authority", AUTHORITY),
        "status": envelope.get("status", "PROVISIONAL_NON_GOLD"),
        "production_eligible": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "epic_ontology_used": False,
        "phrase_count": len(labels),
        "phrases": [label.to_dict() for label in labels],
    }
    source = envelope.get("source")
    if source is not None:
        metadata["source"] = _json_copy(source, field="phrase_catalog.source")
    return tuple(labels), metadata


def _load_manifest(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    document = _load_json(value)
    manifest = dict(_mapping(document, field="manifest"))
    source = _mapping(manifest.get("source"), field="manifest.source")
    source_path = _text(source.get("path"), field="manifest.source.path")
    manifest["source"] = dict(source)
    manifest["source"]["path"] = source_path
    windows = _sequence(manifest.get("windows"), field="manifest.windows")
    if not windows:
        raise ProductionWemmOpenRunnerError("manifest.windows must not be empty")
    for index, raw in enumerate(windows):
        window = _mapping(raw, field=f"manifest.windows[{index}]")
        _text(window.get("window_id"), field=f"manifest.windows[{index}].window_id")
        start = _finite(window.get("start_seconds"), field=f"windows[{index}].start_seconds")
        end = _finite(window.get("end_seconds"), field=f"windows[{index}].end_seconds")
        if start < 0 or end <= start:
            raise ProductionWemmOpenRunnerError(f"invalid interval for windows[{index}]")
    return manifest


def _selected_windows(manifest: Mapping[str, Any], max_windows: int | None) -> list[dict[str, Any]]:
    windows = [
        dict(_mapping(raw, field="manifest.windows[]"))
        for raw in _sequence(manifest["windows"], field="manifest.windows")
    ]
    if max_windows is not None:
        if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows <= 0:
            raise ProductionWemmOpenRunnerError("max_windows must be a positive integer")
        windows = windows[:max_windows]
    if not windows:
        raise ProductionWemmOpenRunnerError("no windows selected")
    return windows


def _catalog_source(labels: Sequence[OpenPhrase], metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": metadata.get("format", PHRASE_CATALOG_FORMAT),
        "phrase_count": len(labels),
        "epic_ontology_used": False,
        "mapper_used": False,
        "provisional": True,
    }


def _rank_camera(
    labels: Sequence[OpenPhrase],
    label_vectors: Mapping[str, Sequence[float]],
    query_vector: Sequence[float],
    *,
    camera_id: str,
    label_variant: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        cosine = cosine_similarity(query_vector, label_vectors[label.provisional_id])
        rows.append(
            {
                "action_key": label.provisional_id,
                "provisional_id": label.provisional_id,
                "label_text": label.text_for(label_variant),
                "label_variant": label_variant,
                "score": (cosine + 1.0) / 2.0,
                "visual_cosine": cosine,
                "camera_id": camera_id,
                "structured_labels": label.structured_labels,
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), str(row["provisional_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:top_k]


def _fused_rows(
    fused: Mapping[str, Any],
    labels_by_id: Mapping[str, OpenPhrase],
    *,
    label_variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _sequence(fused.get("candidates"), field="fused.candidates"):
        candidate = _mapping(raw, field="fused.candidates[]")
        raw_key = candidate.get("action_key")
        if isinstance(raw_key, list):
            raise ProductionWemmOpenRunnerError(
                "open candidate unexpectedly contains pair identity"
            )
        provisional_id = _text(raw_key, field="fused.candidates[].action_key")
        label = labels_by_id.get(provisional_id)
        if label is None:
            raise ProductionWemmOpenRunnerError(
                f"fused candidate references unknown provisional_id {provisional_id!r}"
            )
        camera_evidence = candidate.get("camera_evidence", candidate.get("per_camera", []))
        rows.append(
            {
                "rank": int(candidate.get("rank", len(rows) + 1)),
                "provisional_id": provisional_id,
                "label_text": label.text_for(label_variant),
                "label_variant": label_variant,
                "structured_labels": label.structured_labels,
                "score": _finite(
                    candidate.get("fused_score", candidate.get("score")),
                    field="fused_score",
                ),
                "camera_id": None,
                "camera_support": candidate.get("camera_coverage"),
                "evidence": _json_copy(camera_evidence, field="fused.camera_evidence"),
                "raw": _json_copy(candidate, field="fused.candidate"),
            }
        )
    return rows


def dry_run_open_phrase_plan(
    manifest: Mapping[str, Any] | str | Path,
    *,
    phrase_catalog: Mapping[str, Any] | Sequence[Any] | str | Path,
    max_windows: int | None = None,
) -> dict[str, Any]:
    """Validate an open run without decoding media or loading WeMM."""

    manifest_doc = _load_manifest(manifest)
    labels, catalog_meta = load_open_phrase_catalog(phrase_catalog)
    windows = _selected_windows(manifest_doc, max_windows)
    source = _mapping(manifest_doc.get("source"), field="manifest.source")
    cameras = _sequence(source.get("cameras"), field="manifest.source.cameras")
    camera_ids = [
        _text(_mapping(camera, field="camera").get("camera_id"), field="camera_id")
        for camera in cameras
    ]
    return {
        "format": OPEN_RUN_FORMAT,
        "authority": AUTHORITY,
        "status": "DRY_RUN",
        "production_eligible": False,
        "model_invoked": False,
        "source": {
            "path": source.get("path"),
            "manifest_format": manifest_doc.get("format"),
            "window_count": len(windows),
            "camera_count": len(camera_ids),
            "camera_ids": camera_ids,
        },
        "catalog": _catalog_source(labels, catalog_meta),
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "epic_ontology_used": False,
            "mapper_used": False,
            "hash_or_sha_used": False,
        },
        "limitations": [
            "Dry-run validates only manifest/catalog shape; no media or model was used.",
            "All selected windows remain review-only and have no measured quality.",
        ],
    }


def run_production_wemm_open(
    manifest: Mapping[str, Any] | str | Path,
    *,
    phrase_catalog: Mapping[str, Any] | Sequence[Any] | str | Path,
    model_directory: str | Path,
    frame_count: int = 4,
    top_k: int = 10,
    dimension: int = 256,
    device: str = "cuda",
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    label_variant: str = "canonical",
    max_windows: int | None = None,
    fusion: str = "mean",
    score_normalization: str = "none",
    validate_crcs: bool = False,
    backend: WemmEmbeddingBackend | None = None,
    window_chunk_size: int = 1,
    inference_batch_size: int = 1,
    include_pipeline: bool = False,
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    decode_cache: ProductionWemmDecodeCache | None = None,
    decode_scope_key: Hashable | None = None,
) -> dict[str, Any]:
    """Run native WeMM retrieval and return a validated review envelope.

    ``backend`` is an optional resident encoder supplied by a serial batch
    runner.  When omitted the function preserves its historical one-shot
    lifecycle and owns/ closes a freshly-created backend.  Supplying a backend
    avoids reloading the multi-gigabyte checkpoint for every recording while
    keeping the output contract unchanged.

    ``window_chunk_size`` bounds how many processing windows are decoded and
    held as PIL images at once.  A value of one is intentionally the safe
    default for long multi-camera recordings; larger values trade memory for
    fewer source scans.

    ``inference_batch_size`` is an opt-in model microbatch width.  The default
    of one preserves the historical singleton path.  Values greater than one
    flatten the bounded chunk in window-major/camera-minor order and call the
    backend's native video microbatch seam; the resulting rows are projected
    back to the same per-camera ranking/envelope shape.  This option changes
    only compute scheduling, not the annotation contract.

    ``video_min_pixels`` and ``video_max_pixels`` optionally override the
    shortest/longest edge bounds used by the direct-video processor.  They are
    useful for controlled frame/grid experiments and are recorded in model
    metadata when supplied; omitted values retain the backend defaults.

    ``include_pipeline`` is a separate opt-in producer/consumer schedule.  It
    decodes the bounded chunks on one worker while the resident backend consumes
    the previous chunk on another worker.  The serial schedule remains the
    default, and the queue is bounded by ``queue_capacity``.  Pipeline mode is
    deliberately a scheduling choice only: chunks are consumed FIFO and the
    resulting review envelope has the same window/camera order and fields as
    the serial route.

    ``decode_cache`` is an optional process-local benchmark seam.  Callers must
    provide an explicit ``decode_scope_key``; this runner does not derive an
    identity, persist decoded media, or compute a digest.  Cache replays return
    close-safe frame copies for matrix arms.  Pipeline mode keeps a separate
    source decode so its producer timing remains meaningful.
    """

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or not 2 <= frame_count <= 64
    ):
        raise ProductionWemmOpenRunnerError("frame_count must be between 2 and 64")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ProductionWemmOpenRunnerError("top_k must be a positive integer")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ProductionWemmOpenRunnerError("dimension must be a positive integer")
    if not isinstance(device, str) or not device.strip():
        raise ProductionWemmOpenRunnerError("device must be non-empty")
    video_min_pixels = _optional_positive_int(video_min_pixels, field="video_min_pixels")
    video_max_pixels = _optional_positive_int(video_max_pixels, field="video_max_pixels")
    if (
        video_min_pixels is not None
        and video_max_pixels is not None
        and video_min_pixels > video_max_pixels
    ):
        raise ProductionWemmOpenRunnerError("video_min_pixels must be <= video_max_pixels")
    if label_variant not in LABEL_VARIANTS:
        raise ProductionWemmOpenRunnerError(f"unsupported label variant: {label_variant!r}")
    if not isinstance(validate_crcs, bool):
        raise ProductionWemmOpenRunnerError("validate_crcs must be boolean")
    if (
        isinstance(window_chunk_size, bool)
        or not isinstance(window_chunk_size, int)
        or window_chunk_size <= 0
    ):
        raise ProductionWemmOpenRunnerError("window_chunk_size must be a positive integer")
    if (
        isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
        or inference_batch_size <= 0
        or inference_batch_size > 64
    ):
        raise ProductionWemmOpenRunnerError(
            "inference_batch_size must be an integer between 1 and 64"
        )
    if not isinstance(include_pipeline, bool):
        raise ProductionWemmOpenRunnerError("include_pipeline must be boolean")
    if (
        isinstance(queue_capacity, bool)
        or not isinstance(queue_capacity, int)
        or queue_capacity <= 0
    ):
        raise ProductionWemmOpenRunnerError("queue_capacity must be a positive integer")
    if (decode_cache is None) != (decode_scope_key is None):
        raise ProductionWemmOpenRunnerError(
            "decode_cache and decode_scope_key must be supplied together"
        )

    manifest_doc = _load_manifest(manifest)
    labels, catalog_meta = load_open_phrase_catalog(phrase_catalog)
    windows = _selected_windows(manifest_doc, max_windows)

    owns_backend = backend is None
    if backend is None:
        backend = WemmEmbeddingBackend(
            model_directory=model_directory,
            device=device,
            dimension=dimension,
            video_min_pixels=video_min_pixels,
            video_max_pixels=video_max_pixels,
        )
    elif video_min_pixels is not None or video_max_pixels is not None:
        raise ProductionWemmOpenRunnerError(
            "video pixel overrides require a backend created with those bounds"
        )
    observation_start = len(backend.observations)
    labels_by_id = {label.provisional_id: label for label in labels}
    try:
        label_texts = [label.text_for(label_variant) for label in labels]
        # A resident production batch reuses the same phrase prototypes across
        # recordings.  Newer WeMM backends expose an explicit in-process cache;
        # retain the fallback for lightweight test doubles and older adapters.
        encode_texts_cached = getattr(backend, "encode_texts_cached", None)
        if callable(encode_texts_cached):
            label_vectors_raw = encode_texts_cached(label_texts, batch_size=32)
        else:
            label_vectors_raw = backend.encode_texts(label_texts, batch_size=32)
        if len(label_vectors_raw) != len(labels):
            raise ProductionWemmOpenRunnerError(
                f"WeMM returned {len(label_vectors_raw)} label vectors; expected {len(labels)}"
            )
        label_vectors = {
            label.provisional_id: vector
            for label, vector in zip(labels, label_vectors_raw, strict=True)
        }
        output_windows: list[dict[str, Any]] = []
        raw_camera_runs: list[dict[str, Any]] = []
        camera_order: tuple[str, ...] = ()
        # Pipeline timing is retained only for the opt-in producer/consumer
        # diagnostic path.  Keeping it separate from the historical serial
        # envelope avoids changing the default wire shape while making queue,
        # decode, and model contention observable in real runs.
        pipeline_timing: dict[str, Any] | None = None

        def _iter_decoded_chunks() -> Any:
            """Yield bounded decoded groups, with a legacy monkeypatch seam.

            The source-bound iterator is the normal path and scans each MCAP
            exactly once while retaining only the active window chunk.  A few
            downstream tests historically replaced ``decode_production_windows``
            with a fixture; preserve that explicit seam rather than silently
            bypassing their fake media source.
            """

            legacy_decode = (
                decode_production_windows is not _DEFAULT_DECODE_PRODUCTION_WINDOWS
                and iter_decode_production_window_chunks
                is _DEFAULT_ITER_DECODE_PRODUCTION_WINDOW_CHUNKS
            )
            if legacy_decode:
                for chunk_start in range(0, len(windows), window_chunk_size):
                    window_chunk = windows[chunk_start : chunk_start + window_chunk_size]
                    bounded_manifest = {**manifest_doc, "windows": window_chunk}
                    try:
                        yield decode_production_windows(
                            bounded_manifest,
                            frame_count=frame_count,
                            validate_crcs=validate_crcs,
                        )
                    except Exception as exc:
                        raise ProductionWemmOpenRunnerError(
                            f"production media decode failed: {exc}"
                        ) from exc
                return

            def _decode_factory() -> Any:
                return iter_decode_production_window_chunks(
                    {**manifest_doc, "windows": list(windows)},
                    frame_count=frame_count,
                    validate_crcs=validate_crcs,
                    window_chunk_size=window_chunk_size,
                )

            try:
                if decode_cache is not None:
                    yield from decode_cache.iter_chunks(decode_scope_key, _decode_factory)
                else:
                    yield from _decode_factory()
            except ProductionWemmOpenRunnerError:
                raise
            except Exception as exc:
                raise ProductionWemmOpenRunnerError(
                    f"production media decode failed: {exc}"
                ) from exc

        def _process_decoded_chunk(
            chunk_index: int,
            groups: Mapping[str, Mapping[str, Any]],
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            """Encode and project one decoded chunk without retaining pixels.

            The helper is shared by the historical serial loop and the opt-in
            producer/consumer loop.  Keeping all ranking/envelope construction
            here is what makes the latter a scheduling change rather than a
            second annotation implementation.
            """

            nonlocal camera_order
            chunk_start = chunk_index * window_chunk_size
            window_chunk = windows[chunk_start : chunk_start + window_chunk_size]
            if not window_chunk:
                raise ProductionWemmOpenRunnerError(
                    f"decoded chunk {chunk_index} has no requested windows"
                )
            try:
                camera_order = tuple(sorted(groups))
                # Flatten one bounded decode chunk in the same deterministic
                # order used by the serial implementation.  A single model
                # call can then cover several camera/window groups while the
                # envelope is rebuilt per window below.
                flat_refs: list[tuple[str, Mapping[str, Any], Any]] = []
                for raw_window in window_chunk:
                    window = _mapping(raw_window, field="manifest.windows[]")
                    window_id = _text(window.get("window_id"), field="window_id")
                    for camera_id in camera_order:
                        group = groups[camera_id][window_id]
                        flat_refs.append((camera_id, window, group))

                observation_cursor = len(backend.observations)
                if inference_batch_size == 1:
                    query_vectors = tuple(
                        backend.encode_video_frames(
                            [group.frames], metadata_groups=[group.metadata()]
                        )[0]
                        for _camera_id, _window, group in flat_refs
                    )
                else:
                    if not hasattr(backend, "encode_video_frames_batch"):
                        raise ProductionWemmOpenRunnerError(
                            "backend lacks encode_video_frames_batch for inference_batch_size > 1"
                        )
                    query_vectors = backend.encode_video_frames_batch(
                        [group.frames for _camera_id, _window, group in flat_refs],
                        metadata_groups=[
                            group.metadata() for _camera_id, _window, group in flat_refs
                        ],
                        batch_size=inference_batch_size,
                    )
                if len(query_vectors) != len(flat_refs):
                    raise ProductionWemmOpenRunnerError(
                        "WeMM returned an unexpected number of video embeddings: "
                        f"{len(query_vectors)} != {len(flat_refs)}"
                    )
                row_observations = _expand_model_observations(
                    backend.observations[observation_cursor:],
                    expected_count=len(flat_refs),
                )

                # Rebuild the historical per-window envelope from the ordered
                # flattened rows.  This keeps fusion and review consumers
                # completely unaware of the batching choice.
                chunk_output_windows: list[dict[str, Any]] = []
                chunk_raw_camera_runs: list[dict[str, Any]] = []
                vector_index = 0
                for local_index, raw_window in enumerate(window_chunk):
                    window = _mapping(raw_window, field="manifest.windows[]")
                    window_id = _text(window.get("window_id"), field="window_id")
                    ordinal = window.get("ordinal", chunk_start + local_index)
                    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                        ordinal = chunk_start + local_index
                    start = _finite(window.get("start_seconds"), field=f"{window_id}.start_seconds")
                    end = _finite(window.get("end_seconds"), field=f"{window_id}.end_seconds")
                    per_camera: dict[str, list[dict[str, Any]]] = {}
                    observations: list[dict[str, Any]] = []
                    for camera_id in camera_order:
                        group = groups[camera_id][window_id]
                        query_vector = query_vectors[vector_index]
                        row_observation = (
                            row_observations[vector_index] if row_observations else None
                        )
                        vector_index += 1
                        ranked = _rank_camera(
                            labels,
                            label_vectors,
                            query_vector,
                            camera_id=camera_id,
                            label_variant=label_variant,
                            top_k=min(top_k, len(labels)),
                        )
                        per_camera[camera_id] = ranked
                        observation = group.to_dict()
                        if row_observation is not None:
                            observation["model_observation"] = row_observation
                        observations.append(observation)
                    fused = fuse_camera_rankings(
                        per_camera,
                        camera_order=camera_order,
                        expected_cameras=camera_order,
                        top_k=min(top_k, len(labels)),
                        fusion=fusion,
                        score_normalization=score_normalization,
                        include_embeddings=False,
                    )
                    fused_rows = _fused_rows(fused, labels_by_id, label_variant=label_variant)
                    top_candidates = [
                        {
                            "rank": row["rank"],
                            "provisional_id": row["provisional_id"],
                            "label_text": row["label_text"],
                            "label_variant": row["label_variant"],
                            "structured_labels": row["structured_labels"],
                            "score": row["score"],
                            "camera_id": None,
                            "camera_support": row["camera_support"],
                            "evidence": row["evidence"],
                            "raw": row["raw"],
                        }
                        for row in fused_rows
                    ]
                    best = top_candidates[0] if top_candidates else None
                    proposal: dict[str, Any]
                    if best is None:
                        proposal = {
                            "proposal_id": f"{window_id}-p01",
                            "proposal_status": "UNKNOWN",
                            "unknown": True,
                            "top_k": [],
                            "camera_support": [],
                            "decision": "abstain",
                        }
                    else:
                        label = labels_by_id[str(best["provisional_id"])]
                        evidence = []
                        for item in _sequence(best.get("evidence", []), field="proposal.evidence"):
                            evidence.append(item)
                        proposal = {
                            "proposal_id": f"{window_id}-p01",
                            "proposal_status": "PROPOSED",
                            "provisional_id": label.provisional_id,
                            "label_text": label.text_for(label_variant),
                            "structured_labels": label.structured_labels,
                            "confidence": best["score"],
                            "camera_support": [
                                str(camera_id)
                                for camera_id in camera_order
                                if any(
                                    isinstance(item, Mapping) and item.get("camera_id") == camera_id
                                    for item in evidence
                                )
                            ],
                            "evidence": evidence,
                            "top_k": top_candidates,
                            "decision": "pending",
                            "boundary_status": "NOT_MEASURED",
                        }
                    chunk_output_windows.append(
                        {
                            "window_id": window_id,
                            "ordinal": ordinal,
                            "start_seconds": start,
                            "end_seconds": end,
                            "camera_ids": list(camera_order),
                            "proposals": [proposal],
                        }
                    )
                    chunk_raw_camera_runs.append(
                        {
                            "window_id": window_id,
                            "ordinal": ordinal,
                            "camera_rankings": per_camera,
                            "fused": fused,
                            "input_observations": observations,
                        }
                    )
                return chunk_output_windows, chunk_raw_camera_runs
            finally:
                # The chunk's PIL images are intentionally not included in any
                # output sidecar.  Release them before decoding the next chunk
                # so CPython/PyAV cannot accumulate a full recording's frames.
                _close_decoded_group_images(groups)
                gc.collect()

        # The iterator owns one decoder per camera for the whole source scan;
        # each yielded mapping is processed and released before the next chunk.
        # This avoids the historical per-window source rescans and decoder
        # resets while keeping the peak PIL footprint bounded.  Pipeline mode
        # changes only when the next chunk is requested; it does not alter the
        # flattening, ranking, or envelope projection above.
        if not include_pipeline:
            for chunk_index, groups in enumerate(_iter_decoded_chunks()):
                chunk_windows, chunk_raw = _process_decoded_chunk(chunk_index, groups)
                output_windows.extend(chunk_windows)
                raw_camera_runs.extend(chunk_raw)
        else:
            pipeline_iter = iter(_iter_decoded_chunks())
            chunk_count = (len(windows) + window_chunk_size - 1) // window_chunk_size
            # ``run_bounded_pipeline`` intentionally does not know how to
            # dispose of arbitrary payloads when a sibling worker fails.  Keep
            # only the bounded set of currently queued/in-flight decoded
            # chunks here so a cancelled producer item cannot retain PIL
            # frames until process teardown.  Successful items are removed by
            # the consumer after ``_process_decoded_chunk`` closes their
            # images; at most ``queue_capacity + 1`` chunks are referenced.
            pending_pipeline_groups: dict[int, Mapping[str, Mapping[str, Any]]] = {}
            pending_pipeline_lock = threading.Lock()

            def prepare_chunk(
                ordinal: int, recorder: Any
            ) -> tuple[int, Mapping[str, Mapping[str, Any]]]:
                with recorder.phase(PipelinePhase.MEDIA_DECODE):
                    try:
                        groups = next(pipeline_iter)
                    except StopIteration as exc:
                        raise ProductionWemmOpenRunnerError(
                            "decoder ended before the planned pipeline chunks"
                        ) from exc
                with pending_pipeline_lock:
                    pending_pipeline_groups[ordinal] = groups
                return ordinal, groups

            def consume_chunk(
                payload: tuple[int, Mapping[str, Mapping[str, Any]]], recorder: Any
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                chunk_index, groups = payload
                # The phase surrounds exactly the same processing helper as the
                # serial path; no duplicate model/ranking implementation exists.
                try:
                    with recorder.phase(PipelinePhase.MODEL):
                        return _process_decoded_chunk(chunk_index, groups)
                finally:
                    with pending_pipeline_lock:
                        pending_pipeline_groups.pop(chunk_index, None)

            try:
                pipeline_run = run_bounded_pipeline(
                    range(chunk_count),
                    prepare=prepare_chunk,
                    consume=consume_chunk,
                    key=lambda _item, ordinal: f"open-chunk-{ordinal:03d}",
                    queue_capacity=queue_capacity,
                )
                if not pipeline_run.succeeded:
                    report = pipeline_run.report
                    raise ProductionWemmOpenRunnerError(
                        "producer-consumer pipeline failed: "
                        f"{report.error_type or 'unknown'}: {report.error_detail or ''}".strip()
                    )
                if len(pipeline_run.outputs) != chunk_count:
                    raise ProductionWemmOpenRunnerError(
                        "producer-consumer pipeline returned "
                        f"{len(pipeline_run.outputs)}/{chunk_count} chunks"
                    )
                for chunk_result in pipeline_run.outputs:
                    chunk_windows, chunk_raw = chunk_result
                    output_windows.extend(chunk_windows)
                    raw_camera_runs.extend(chunk_raw)
                pipeline_timing = pipeline_run.report.to_dict()
            finally:
                # The bounded harness drains cancelled payloads without
                # invoking the consumer callback.  Close those remaining
                # groups explicitly before releasing the iterator.
                with pending_pipeline_lock:
                    pending_groups = tuple(pending_pipeline_groups.values())
                    pending_pipeline_groups.clear()
                for groups in pending_groups:
                    _close_decoded_group_images(groups)
                close_pipeline = getattr(pipeline_iter, "close", None)
                if callable(close_pipeline):
                    close_pipeline()
        if len(output_windows) != len(windows):
            raise ProductionWemmOpenRunnerError(
                "production media decode yielded "
                f"{len(output_windows)}/{len(windows)} requested windows"
            )
        source = dict(_mapping(manifest_doc.get("source"), field="manifest.source"))
        source["manifest_format"] = manifest_doc.get("format")
        source.pop("cameras", None)
        source["window_count"] = len(output_windows)
        source["camera_count"] = len(camera_order)
        model_metadata: dict[str, Any] = {
            "name": "WeMM-Embedding-2B",
            "route": "complete_bounded_video_embedding_open_phrases",
            "dimension": dimension,
            "frame_count": frame_count,
            "window_chunk_size": window_chunk_size,
            "inference_batch_size": inference_batch_size,
            "label_variant": label_variant,
            "fusion": fusion,
            "score_normalization": score_normalization,
        }
        prototype_stats = getattr(backend, "text_prototype_cache_stats", None)
        if callable(prototype_stats):
            model_metadata["text_prototype_cache"] = dict(prototype_stats())
        if video_min_pixels is not None:
            model_metadata["video_min_pixels"] = video_min_pixels
        if video_max_pixels is not None:
            model_metadata["video_max_pixels"] = video_max_pixels
        if include_pipeline:
            # Keep scheduling metadata opt-in so the historical serial output
            # shape remains byte-for-byte compatible for existing consumers.
            model_metadata["producer_consumer"] = True
            model_metadata["queue_capacity"] = queue_capacity
        if decode_cache is not None:
            model_metadata["decode_cache"] = {
                "scope": "process_local",
                "scope_key_supplied": True,
                **decode_cache.stats().to_dict(),
                "pipeline_uses_separate_decode": bool(include_pipeline),
            }
        raw_model_output: dict[str, Any] = {
            "format": OPEN_RUN_FORMAT,
            "catalog": _catalog_source(labels, catalog_meta),
            "windows": raw_camera_runs,
            "backend_observations": backend.observation_payload()[observation_start:],
        }
        if pipeline_timing is not None:
            raw_model_output["pipeline_timing"] = pipeline_timing
        envelope = build_preannotation_envelope(
            source,
            output_windows,
            raw_model_output=raw_model_output,
            model=model_metadata,
            candidate_profile="open_phrase_catalog",
            model_invoked=True,
        )
        return validate_preannotation_envelope(envelope)
    except ProductionWemmPreannotationError as exc:
        raise ProductionWemmOpenRunnerError(f"pre-annotation envelope failed: {exc}") from exc
    finally:
        if owns_backend:
            backend.close()


__all__ = [
    "AUTHORITY",
    "DEFAULT_QUEUE_CAPACITY",
    "LABEL_VARIANTS",
    "OPEN_RUN_FORMAT",
    "PHRASE_CATALOG_FORMAT",
    "OpenPhrase",
    "ProductionWemmOpenRunnerError",
    "dry_run_open_phrase_plan",
    "load_open_phrase_catalog",
    "run_production_wemm_open",
]
