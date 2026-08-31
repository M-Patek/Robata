#!/usr/bin/env python3
"""Run the benchmark-local WeMM→EPIC joint-action retrieval experiment.

This command is intentionally separate from Robata's production retrieval and
mapper paths.  It reads a bounded development manifest, decodes each annotated
interval into an ordered frame group, obtains a WeMM shared-space embedding,
and compares visual-only, text-only, and hybrid candidate rankings.  The
default ``manifest`` catalog is a small development diagnostic catalog; a
training-derived pair file should be supplied for a leakage-free catalog when
one is available.

No model larger than WeMM-Embedding-2B is loaded, no ontology or mapper is
modified, and no content hashes are produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.wemm_action_retrieval import (  # noqa: E402
    LabelVariant,
    WemmRetrievalError,
    build_joint_action_catalog,
    compare_rankings,
    full_cartesian_action_pairs,
    project_retrieval_to_mapper,
    rank_joint_actions,
    text_scores_for_prediction,
)
from robata.benchmark.wemm_embedding_backend import (  # noqa: E402
    WemmBackendUnavailable,
    WemmEmbeddingBackend,
)

# The metric report below always includes Recall@10.  Do not let a smaller
# ``top_k`` silently truncate that metric.  The native WeMM processor also has
# a hard 64-frame limit for one direct video request; callers must obey it
# before decoding any pixels.
_REPORT_KS = (1, 3, 5, 10)
_MAX_VIDEO_FRAMES = 64

# Pair catalogs are accepted in a small number of explicitly named split
# forms.  The split allow-list is deliberately conservative: a source and
# split are useful provenance labels, but they do not establish label
# blindness on their own.  ``label_blind=true`` is required separately below.
_VERIFIED_CATALOG_SPLITS = frozenset(
    {
        "train",
        "training",
        "train_disjoint_from_dev27",
        "ontology",
        "catalog",
        "full",
        "all",
        "cross_split",
        "cross-split",
    }
)

# These names are annotation/target fields, not model predictions.  They are
# tolerated at the *row* level for backwards-compatible smoke reports (and are
# surfaced as label-bearing provenance), but are never read as text features.
# A prediction object containing one of them is rejected because that would
# make target leakage indistinguishable from a model output.
_TEXT_TARGET_FIELDS = frozenset(
    {
        "ground_truth",
        "verb_class",
        "noun_class",
        "verb_id",
        "noun_id",
        "action_key",
        "target",
        "target_action",
        "label",
        "official_reference",
        "annotation",
        "answer",
    }
)
_TEXT_KEY_FIELDS = ("uid", "case_id", "annotation_id", "id", "video_id")


def _read_class_table(path: Path) -> dict[int, str]:
    """Read only official ``id,key`` columns without computing an identity hash."""

    entries: dict[int, str] = {}
    with path.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "key"}.issubset(reader.fieldnames):
            raise ValueError(f"class table must contain id,key columns: {path}")
        for row_number, row in enumerate(reader, start=2):
            raw_id = str(row.get("id") or "").strip()
            key = str(row.get("key") or "").strip()
            if not raw_id.isdigit() or not key:
                raise ValueError(f"invalid class table row {path}:{row_number}")
            class_id = int(raw_id)
            if class_id in entries:
                raise ValueError(f"duplicate class id {class_id} in {path}")
            entries[class_id] = key
    if not entries:
        raise ValueError(f"class table is empty: {path}")
    return dict(sorted(entries.items()))


def _finite_seconds(value: Any, *, field: str, row_index: int) -> float:
    """Coerce one manifest time value to a finite, non-negative float.

    JSON manifests normally contain numbers.  Numeric strings are accepted for
    compatibility with CSV-to-JSON exports, while booleans, nulls, NaN and
    infinities are rejected explicitly rather than being silently converted by
    ``float``.
    """

    if isinstance(value, bool) or value is None:
        raise ValueError(f"manifest row {row_index} {field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"manifest row {row_index} {field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"manifest row {row_index} {field} must be finite")
    if parsed < 0.0:
        raise ValueError(f"manifest row {row_index} {field} must be non-negative")
    return parsed


def _path_within_root(candidate: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved directory."""

    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _load_manifest(path: Path, dataset_root: Path, max_cases: int | None) -> list[dict[str, Any]]:
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be a positive integer")
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {resolved_root}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.expanduser().resolve().read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {line_number} must be a JSON object")
        relpath = row.get("video_relpath")
        if not isinstance(relpath, str) or not relpath.strip():
            raise ValueError(f"manifest row {line_number} has no video_relpath")
        # ``video_relpath`` is intentionally a relative field.  Resolve after
        # normalising separators so Windows-style traversal is checked on all
        # platforms, then reject both absolute paths and symlink/traversal
        # escapes from the declared dataset root.
        normalized_relpath = relpath.strip().replace("\\", "/")
        relative_path = Path(normalized_relpath)
        # Check both path flavours so a manifest authored on another OS cannot
        # smuggle a drive/UNC absolute path through a host-native ``Path``.
        foreign_absolute = (
            PurePosixPath(normalized_relpath).is_absolute()
            or PureWindowsPath(normalized_relpath).is_absolute()
            or bool(PureWindowsPath(normalized_relpath).drive)
        )
        if relative_path.is_absolute() or foreign_absolute:
            raise ValueError(f"manifest row {line_number} video_relpath must be relative")
        video_path = (resolved_root / relative_path).resolve()
        if not _path_within_root(video_path, resolved_root):
            raise ValueError(
                f"manifest row {line_number} video path escapes dataset root: {relpath!r}"
            )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        if "start_seconds" not in row or "end_seconds" not in row:
            raise ValueError(
                f"manifest row {line_number} must contain start_seconds and end_seconds"
            )
        start_seconds = _finite_seconds(
            row.get("start_seconds"), field="start_seconds", row_index=line_number
        )
        end_seconds = _finite_seconds(
            row.get("end_seconds"), field="end_seconds", row_index=line_number
        )
        if end_seconds <= start_seconds:
            raise ValueError(
                f"manifest row {line_number} end_seconds must be greater than start_seconds"
            )

        # Store canonical values so downstream decode and the audit report use
        # exactly the same finite interval representation.
        row["start_seconds"] = start_seconds
        row["end_seconds"] = end_seconds
        row["video_path"] = str(video_path)
        rows.append(row)
        if max_cases is not None and len(rows) >= max_cases:
            break
    if not rows:
        raise ValueError("manifest contains no usable rows")
    _validate_unique_row_keys(rows)
    return rows


def _pair_from_row(row: Mapping[str, Any]) -> tuple[int, int] | None:
    truth = row.get("ground_truth")
    if not isinstance(truth, Mapping):
        truth = row
    try:
        verb = truth.get("verb_class", truth.get("verb_id"))
        noun = truth.get("noun_class", truth.get("noun_id"))
        if verb is None or noun is None:
            return None
        return int(verb), int(noun)
    except (TypeError, ValueError):
        return None


def _pair_file_payload_and_provenance(
    path: Path,
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]]:
    """Read action pairs and retain the declared, non-cryptographic provenance.

    A bare list remains accepted for a vertical smoke, but it is *never*
    provenance-verified.  Mapping payloads must declare all three independent
    provenance facts: a non-empty source, an allow-listed non-evaluation split,
    and ``provenance.label_blind`` explicitly equal to ``true``.  A source and
    split without the label-blind declaration are intentionally insufficient.
    """

    resolved_path = path.expanduser().resolve()
    raw_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    declared_format: str | None = None
    source: Any = None
    split: Any = None
    label_blind_value: Any = None
    if isinstance(raw_payload, Mapping):
        pairs_payload = raw_payload.get("action_pairs", raw_payload.get("pairs"))
        declared = raw_payload.get("format")
        if isinstance(declared, str) and declared.strip():
            declared_format = declared.strip()
        provenance: Any = raw_payload.get("provenance")
        source = raw_payload.get("source") or raw_payload.get("source_dataset")
        split = raw_payload.get("split") or raw_payload.get("source_split")
        label_blind_value = raw_payload.get("label_blind")
        if isinstance(provenance, Mapping):
            source = provenance.get("source") or provenance.get("dataset") or source
            split = provenance.get("split") or provenance.get("source_split") or split
            label_blind_value = provenance.get("label_blind", label_blind_value)
        container_format = "mapping"
        payload = pairs_payload
    else:
        # Legacy bare-list files remain usable for a vertical smoke only.
        container_format = "list"
        payload = raw_payload

    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("pair file must contain a JSON list or {action_pairs: [...]}")
    pairs: list[tuple[int, int]] = []
    for item in payload:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"invalid action pair: {item!r}")
        converted: list[int] = []
        for value, field in zip(item, ("verb", "noun"), strict=True):
            if isinstance(value, bool):
                raise ValueError(f"invalid {field} ID in action pair: {item!r}")
            if isinstance(value, Integral):
                converted.append(int(value))
                continue
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                converted.append(int(value.strip()))
                continue
            raise ValueError(f"invalid {field} ID in action pair: {item!r}")
        pairs.append((converted[0], converted[1]))
    if not pairs:
        raise ValueError("pair file is empty")

    source_text = str(source).strip() if source is not None else None
    split_text = str(split).strip() if split is not None else None
    split_key = split_text.casefold() if split_text else ""
    label_blind = label_blind_value if isinstance(label_blind_value, bool) else None
    verified = bool(source_text) and split_key in _VERIFIED_CATALOG_SPLITS and label_blind is True
    metadata = {
        "path": str(resolved_path),
        "format": declared_format or f"json_{container_format}",
        "container_format": container_format,
        "source": source_text,
        "split": split_text,
        "label_blind": label_blind,
        "verified": verified,
        # The runner intentionally uses target frequencies for an unverified
        # catalog so that a smoke result is visibly target-informed rather than
        # silently presenting it as a quality measurement.
        "evaluation_labels_used": not verified,
    }
    return tuple(pairs), metadata


def _pair_file_payload(path: Path) -> tuple[tuple[tuple[int, int], ...], bool]:
    """Backwards-compatible pair reader returning the verification bit."""

    pairs, metadata = _pair_file_payload_and_provenance(path)
    return pairs, bool(metadata["verified"])


def _pair_file_provenance(path: Path) -> dict[str, Any]:
    """Return sanitized pair-file provenance without hashes or digests."""

    _pairs, metadata = _pair_file_payload_and_provenance(path)
    return metadata


def _load_pair_file(path: Path) -> tuple[tuple[int, int], ...]:
    """Backwards-compatible pair-file reader (provenance is checked by caller)."""

    pairs, _verified = _pair_file_payload(path)
    return pairs


def _catalog_pairs(
    rows: Sequence[Mapping[str, Any]],
    verbs: Mapping[int, str],
    nouns: Mapping[int, str],
    *,
    pair_file: Path | None,
    full_cartesian: bool,
) -> tuple[tuple[tuple[int, int], ...], str, bool]:
    if pair_file is not None and full_cartesian:
        raise ValueError("--catalog-pairs and --full-cartesian are mutually exclusive")
    if pair_file is not None:
        pairs, verified = _pair_file_payload(pair_file)
        return (
            pairs,
            "explicit_pair_file" if verified else "explicit_pair_file_unverified",
            not verified,
        )
    if full_cartesian:
        return full_cartesian_action_pairs(verbs, nouns), "full_cartesian", False
    pairs = tuple(sorted({pair for row in rows if (pair := _pair_from_row(row)) is not None}))
    if not pairs:
        raise ValueError("manifest has no ground-truth pairs; supply --catalog-pairs")
    # This fallback is useful for a fast vertical smoke only.  Make the
    # development-label dependence explicit in the report and never call it a
    # production or held-out measurement.
    return pairs, "development_manifest_pairs", True


def _catalog_provenance_details(
    *,
    pair_file: Path | None,
    full_cartesian: bool,
    catalog_source: str,
    catalog_uses_dev_labels: bool,
) -> dict[str, Any]:
    """Describe the catalog source without inventing an identity digest."""

    if pair_file is not None:
        # Read the same declared fields that determine the verification bit.
        # Keeping these values in the report makes a sidecar self-describing;
        # no hash is computed because provenance here is intentionally factual,
        # not a cryptographic identity assertion.
        return _pair_file_provenance(pair_file)
    if full_cartesian:
        return {
            "path": None,
            "format": "generated_full_cartesian",
            "container_format": "generated",
            "source": "verb and noun class tables",
            "split": "full_cartesian",
            "label_blind": True,
            "verified": True,
            "evaluation_labels_used": False,
        }
    return {
        "path": None,
        "format": "generated_manifest_pairs",
        "container_format": "generated",
        "source": "development manifest ground-truth pairs",
        "split": "development_manifest",
        "label_blind": False,
        "verified": False,
        "evaluation_labels_used": True,
    }


def _catalog_ground_truth_coverage(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[Any]
) -> dict[str, Any]:
    """Expose whether the fixed candidate catalog can score every labelled row.

    A participant-disjoint training catalog may legitimately omit action pairs
    that occur only in the evaluation participants.  Ranking can still be
    useful as a retrieval diagnostic in that situation, but Recall@K/MRR would
    count those unavailable targets as misses.  Keep the run observable while
    marking its aggregate quality metrics ineligible instead of silently
    treating catalog coverage as model error.
    """

    targets = {pair for row in rows if (pair := _pair_from_row(row)) is not None}
    catalog = {tuple(label.action_key) for label in labels if hasattr(label, "action_key")}
    missing = tuple(sorted(targets - catalog))
    return {
        "unique_manifest_target_pairs": len(targets),
        "catalog_pairs": len(catalog),
        "covered_target_pairs": len(targets & catalog),
        "missing_target_pairs": [list(pair) for pair in missing],
        "complete": not missing,
    }


def _decode_interval(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    *,
    intervention: str,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Decode a bounded interval and return RGB PIL frames plus source metadata."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("frame_count must be an integer")
    if frame_count < 2 or frame_count > _MAX_VIDEO_FRAMES:
        raise ValueError(f"frame_count must be between 2 and {_MAX_VIDEO_FRAMES}")
    start_seconds = _finite_seconds(start_seconds, field="start_seconds", row_index=0)
    end_seconds = _finite_seconds(end_seconds, field="end_seconds", row_index=0)
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0 or total_frames <= 0:
            raise RuntimeError(f"video has no usable fps/frame count: {path}")
        duration = total_frames / fps
        left = max(0.0, min(float(start_seconds), max(0.0, duration - 1.0 / fps)))
        right = max(left + 1.0 / fps, min(float(end_seconds), duration))
        times = np.linspace(left, max(left, right - 1.0 / fps), num=frame_count)
        selected_indices = [
            max(0, min(total_frames - 1, round(float(timestamp) * fps))) for timestamp in times
        ]
        if intervention == "reverse":
            selected_indices = list(reversed(selected_indices))
        elif intervention not in {"normal", "freeze_pre", "freeze_post"}:
            raise ValueError(f"unknown intervention: {intervention}")
        if intervention == "freeze_pre":
            selected_indices = [selected_indices[0]] * frame_count
        elif intervention == "freeze_post":
            selected_indices = [selected_indices[-1]] * frame_count
        frames: list[Image.Image] = []
        for frame_index in selected_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {frame_index} from {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).convert("RGB"))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or frames[0].width)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or frames[0].height)
        metadata: dict[str, Any] = {
            "total_num_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "frames_indices": selected_indices,
            "source_window_start_seconds": left,
            "source_window_end_seconds": right,
            "intervention": intervention,
        }
        return frames, metadata
    finally:
        capture.release()


def _row_key(row: Mapping[str, Any], index: int) -> str:
    for field in ("uid", "case_id", "annotation_id", "id"):
        raw_value = row.get(field)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            return value
    return f"row-{index}"


def _validate_unique_row_keys(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Reject duplicate stable IDs before ranking dictionaries can overwrite rows."""

    seen: dict[str, int] = {}
    keys: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest row {index} must be an object")
        key = _row_key(row, index)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(f"duplicate manifest row key {key!r} at rows {previous} and {index}")
        seen[key] = index
        keys.append(key)
    return tuple(keys)


def _nonempty_field(row: Mapping[str, Any], field: str) -> str | None:
    raw_value = row.get(field)
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _target_fields(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(str(key) for key in row if str(key).strip().casefold() in _TEXT_TARGET_FIELDS)
    )


def _prediction_row(row: Mapping[str, Any], index: int) -> tuple[Mapping[str, Any] | None, bool]:
    """Extract only model-authored text fields and flag target-bearing rows."""

    prediction = row.get("prediction")
    if prediction is not None:
        if not isinstance(prediction, Mapping):
            raise ValueError(f"text result {index} prediction must be an object")
        forbidden = _target_fields(prediction)
        if forbidden:
            raise ValueError(f"text result {index} prediction contains target fields: {forbidden}")
        return prediction, bool(_target_fields(row))
    if "verb" in row or "noun" in row or "raw_text" in row:
        # Legacy reports put prediction fields at the row level.  Copy only
        # the three fields consumed by the scorer; never pass annotation or
        # diagnostic fields through as model text.
        allowed = {
            key: row[key] for key in ("verb", "noun", "raw_text", "confidence") if key in row
        }
        return allowed, bool(_target_fields(row))
    return None, bool(_target_fields(row))


def _text_report_payload(path: Path | None) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {
            "provided": False,
            "alignment": "none",
            "alignment_explicit": False,
            "ordinal_consistent": False,
            "label_blind_declared": None,
            "label_bearing_fields_present": False,
            "source": None,
            "split": None,
            "matched_count": 0,
            "missing_count": 0,
            "quality_valid": False,
        }
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("text report must be an object containing a results list")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("text report must contain a results list")
    results: list[Mapping[str, Any]] = []
    label_bearing = bool(_target_fields(payload))
    for index, raw_row in enumerate(raw_results):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"text result {index} must be an object")
        # Validate/sanitize prediction fields now, before any alignment logic.
        _prediction, row_label_bearing = _prediction_row(raw_row, index)
        label_bearing = label_bearing or row_label_bearing
        results.append(raw_row)
    provenance: Any = payload.get("provenance")
    metadata: Any = payload.get("metadata")
    source = payload.get("source") or payload.get("source_dataset")
    split = payload.get("split") or payload.get("source_split")
    alignment_value: Any = payload.get("alignment")
    label_blind_value: Any = payload.get("label_blind")
    if isinstance(provenance, Mapping):
        label_bearing = label_bearing or bool(_target_fields(provenance))
        source = provenance.get("source") or provenance.get("dataset") or source
        split = provenance.get("split") or provenance.get("source_split") or split
        alignment_value = provenance.get("alignment") or alignment_value
        label_blind_value = provenance.get("label_blind", label_blind_value)
    if isinstance(metadata, Mapping):
        label_bearing = label_bearing or bool(_target_fields(metadata))
        source = metadata.get("source") or metadata.get("dataset") or source
        split = metadata.get("split") or metadata.get("source_split") or split
        alignment_value = metadata.get("alignment") or alignment_value
        label_blind_value = metadata.get("label_blind", label_blind_value)
    payload_label_blind = label_blind_value
    alignment = alignment_value
    alignment_text = str(alignment).strip().casefold() if alignment is not None else ""
    return results, {
        "provided": True,
        "alignment": alignment_text or None,
        "alignment_explicit": bool(alignment_text),
        "label_blind_declared": payload_label_blind
        if isinstance(payload_label_blind, bool)
        else None,
        "label_bearing_fields_present": label_bearing,
        "source": str(source).strip() if source is not None else None,
        "split": str(split).strip() if split is not None else None,
    }


def _manifest_aliases(row: Mapping[str, Any], index: int) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = [("row_key", _row_key(row, index))]
    for field in _TEXT_KEY_FIELDS:
        value = _nonempty_field(row, field)
        if value is not None:
            aliases.append((field, value))
    # De-duplicate aliases while preserving deterministic order.
    return tuple(dict.fromkeys(aliases))


def _prediction_aliases(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    for field in _TEXT_KEY_FIELDS:
        value = _nonempty_field(row, field)
        if value is not None:
            aliases.append((field, value))
    return tuple(dict.fromkeys(aliases))


def _align_text_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Align a text report by stable IDs, with a tightly-scoped ordinal fallback."""

    if not predictions:
        result = dict(provenance)
        result.update(
            {
                "matched_count": 0,
                "missing_count": len(rows),
                "ordinal_consistent": False,
                "quality_valid": False,
            }
        )
        return [], result
    for index, row in enumerate(predictions):
        if not isinstance(row, Mapping):
            raise ValueError(f"text result {index} must be an object")
        _prediction_row(row, index)
    manifest_alias_map: dict[tuple[str, str], set[int]] = {}
    for index, row in enumerate(rows):
        for alias in _manifest_aliases(row, index):
            manifest_alias_map.setdefault(alias, set()).add(index)
    # Repeated aliases such as a video-level ID are not safe join keys.  Ignore
    # them so a legacy report can still use its verified ordinal column rather
    # than being rejected as "multiple manifest rows".
    unique_manifest_aliases = {
        alias for alias, matches in manifest_alias_map.items() if len(matches) == 1
    }
    aligned: list[Mapping[str, Any] | None] = [None] * len(rows)
    used_indices: set[int] = set()
    alignment = str(provenance.get("alignment") or "").casefold()
    # Parse ordinal columns up front.  An explicit ordinal contract (and a
    # legacy report with a consistent 0..N-1 ordinal) takes precedence over
    # ambiguous video-level aliases, which may repeat across many intervals.
    ordinals: list[int] = []
    have_ordinals = True
    for index, row in enumerate(predictions):
        raw_ordinal = row.get("ordinal", row.get("index"))
        if raw_ordinal is None:
            have_ordinals = False
            break
        if isinstance(raw_ordinal, bool):
            raise ValueError(f"text result {index} ordinal must be an integer")
        try:
            ordinal = int(raw_ordinal)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"text result {index} ordinal must be an integer") from exc
        if ordinal != raw_ordinal and not (
            isinstance(raw_ordinal, str) and raw_ordinal.strip() == str(ordinal)
        ):
            raise ValueError(f"text result {index} ordinal must be integral")
        ordinals.append(ordinal)
    ordinal_consistent = (
        len(predictions) == len(rows) and have_ordinals and ordinals == list(range(len(rows)))
    )
    if alignment not in {"", "ordinal"}:
        raise ValueError(f"unsupported text alignment: {alignment!r}")
    use_ordinal = alignment == "ordinal" or ordinal_consistent
    if use_ordinal:
        if len(predictions) != len(rows):
            raise ValueError("ordinal text report requires exactly one result per manifest row")
        if alignment != "ordinal" and not ordinal_consistent:
            raise ValueError("ordinal text report must contain consistent 0..N-1 ordinals")
        if alignment == "ordinal" and have_ordinals and not ordinal_consistent:
            raise ValueError("ordinal text report must contain consistent 0..N-1 ordinals")
        aligned = list(predictions)
        alignment = "ordinal"
    else:
        prediction_aliases = [
            tuple(alias for alias in _prediction_aliases(row) if alias in unique_manifest_aliases)
            for row in predictions
        ]
        keyed = [bool(aliases) for aliases in prediction_aliases]
        if any(keyed):
            if not all(keyed):
                raise ValueError("text report mixes keyed and unkeyed rows")
            for prediction_index, (raw_row, aliases) in enumerate(
                zip(predictions, prediction_aliases, strict=True)
            ):
                matches: set[int] = set()
                for alias in aliases:
                    matches.update(manifest_alias_map.get(alias, set()))
                if not matches:
                    raise ValueError(f"text result {prediction_index} has no matching manifest row")
                if len(matches) != 1:
                    raise ValueError(
                        f"text result {prediction_index} matches multiple manifest rows"
                    )
                target_index = next(iter(matches))
                if target_index in used_indices:
                    raise ValueError(f"duplicate text prediction for manifest row {target_index}")
                aligned[target_index] = raw_row
                used_indices.add(target_index)
            alignment = "stable_key"
        else:
            # Positional alignment is accepted only for an exact row count and
            # a declared ordinal contract or a verified 0..N-1 ordinal column.
            if len(predictions) != len(rows):
                raise ValueError("unkeyed text report requires exactly one result per manifest row")
            raise ValueError(
                "unkeyed text report requires alignment='ordinal' and consistent ordinals"
            )
    aligned_rows = [row for row in aligned if row is not None]
    label_blind_declared = provenance.get("label_blind_declared") is True
    label_bearing = bool(provenance.get("label_bearing_fields_present"))
    source = str(provenance.get("source") or "").strip()
    split = str(provenance.get("split") or "").strip()
    quality_valid = (
        label_blind_declared
        and not label_bearing
        and bool(source)
        and bool(split)
        and alignment in {"stable_key", "ordinal"}
        and len(aligned_rows) == len(rows)
    )
    result = dict(provenance)
    result.update(
        {
            "alignment": alignment,
            "alignment_explicit": bool(provenance.get("alignment_explicit")),
            "ordinal_consistent": ordinal_consistent,
            "matched_count": len(aligned_rows),
            "missing_count": len(rows) - len(aligned_rows),
            "quality_valid": quality_valid,
        }
    )
    # Keep one placeholder row per manifest row for the scorer's positional
    # contract.  Missing keyed predictions intentionally score as empty text.
    return [row if row is not None else {} for row in aligned], result


def _load_text_predictions(path: Path | None) -> list[Mapping[str, Any]]:
    """Load legacy-compatible results without aligning them to a manifest."""

    if path is None:
        return []
    predictions, _provenance = _text_report_payload(path)
    return predictions


def _load_and_align_text_predictions(
    path: Path | None, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if path is None:
        _predictions, provenance = _text_report_payload(path)
        return [], provenance
    predictions, provenance = _text_report_payload(path)
    return _align_text_predictions(rows, predictions, provenance)


def _text_prediction_for(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], index: int
) -> Mapping[str, Any] | None:
    if 0 <= index < len(predictions):
        row = predictions[index]
        prediction = row.get("prediction")
        if isinstance(prediction, Mapping):
            return prediction
        if "verb" in row or "noun" in row or "raw_text" in row:
            return {
                key: row[key] for key in ("verb", "noun", "raw_text", "confidence") if key in row
            }
    return None


def _plain_audit_value(value: Any) -> Any:
    """Convert decoder metadata to JSON-native scalars without hashing data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _plain_audit_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_audit_value(item) for item in value]
    # numpy scalar values and similar decoder metadata expose an ``item``
    # method.  Convert those before falling back to a descriptive string.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain_audit_value(item())
        except Exception:  # pragma: no cover - defensive for exotic metadata
            pass
    return str(value)


def _audit_number(value: Any) -> float | int | None:
    """Return a finite JSON number for optional audit metadata."""

    plain = _plain_audit_value(value)
    if isinstance(plain, bool) or plain is None:
        return None
    if isinstance(plain, int):
        return plain
    if isinstance(plain, float):
        return plain if math.isfinite(plain) else None
    try:
        parsed = float(plain)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_input_audit_entry(
    row: Mapping[str, Any],
    index: int,
    metadata: Mapping[str, Any] | None,
    *,
    requested_frame_count: int,
    intervention: str,
    decoded_frame_count: int | None = None,
) -> dict[str, Any]:
    """Build one reproducibility record for a decoded manifest interval.

    This records paths, interval geometry and frame-selection metadata only.
    It intentionally excludes pixels, embeddings and hashes.
    """

    decoder = metadata if isinstance(metadata, Mapping) else {}
    frame_indices_raw = decoder.get("frames_indices", decoder.get("frame_indices"))
    frame_indices = _plain_audit_value(frame_indices_raw)
    if not isinstance(frame_indices, list):
        frame_indices = None
    actual_frame_count = len(frame_indices) if frame_indices is not None else decoded_frame_count
    fps = _audit_number(decoder.get("fps", decoder.get("source_fps")))
    total_num_frames = _audit_number(decoder.get("total_num_frames"))
    duration = _audit_number(decoder.get("duration", decoder.get("duration_seconds")))
    source_start = _audit_number(
        decoder.get("source_window_start_seconds", decoder.get("source_start_seconds"))
    )
    source_end = _audit_number(
        decoder.get("source_window_end_seconds", decoder.get("source_end_seconds"))
    )
    start_seconds = _audit_number(row.get("start_seconds"))
    end_seconds = _audit_number(row.get("end_seconds"))
    video_path = row.get("video_path")
    if video_path is not None:
        video_path = str(video_path)
    video_relpath = row.get("video_relpath")
    if video_relpath is not None:
        video_relpath = str(video_relpath)

    entry: dict[str, Any] = {
        "ordinal": index,
        "index": index,
        "row_key": _row_key(row, index),
        "uid": _plain_audit_value(row.get("uid")),
        "case_id": _plain_audit_value(row.get("case_id")),
        "annotation_id": _plain_audit_value(row.get("annotation_id")),
        "participant_id": _plain_audit_value(row.get("participant_id", row.get("participant"))),
        "video_id": _plain_audit_value(row.get("video_id")),
        "camera_id": _plain_audit_value(row.get("camera_id")),
        "video_relpath": video_relpath,
        "video_path": video_path,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "interval_start_seconds": start_seconds,
        "interval_end_seconds": end_seconds,
        "interval_seconds": [start_seconds, end_seconds],
        "requested_frame_count": requested_frame_count,
        "frame_count": actual_frame_count
        if actual_frame_count is not None
        else requested_frame_count,
        "decoded_frame_count": actual_frame_count
        if actual_frame_count is not None
        else requested_frame_count,
        "fps": fps,
        "source_fps": fps,
        "total_num_frames": total_num_frames,
        "duration": duration,
        "duration_seconds": duration,
        "frames_indices": frame_indices,
        "frame_indices": frame_indices,
        "source_window_start_seconds": source_start,
        "source_window_end_seconds": source_end,
        "source_window_seconds": [source_start, source_end],
        "intervention": _plain_audit_value(decoder.get("intervention", intervention)),
        "width": _audit_number(decoder.get("width")),
        "height": _audit_number(decoder.get("height")),
    }
    return cast(dict[str, Any], _plain_audit_value(entry))


def _quality_validity_reason(
    *,
    catalog_quality_valid: bool,
    catalog_provenance_valid: bool,
    catalog_target_coverage_complete: bool,
    text_report_provided: bool,
    text_report_quality_valid: bool,
) -> str:
    """Explain aggregate/mode validity in a stable, machine-readable sentence."""

    reasons: list[str] = []
    if not catalog_provenance_valid:
        reasons.append("catalog provenance is not verified label_blind=true")
    if not catalog_target_coverage_complete:
        reasons.append("catalog omits manifest ground-truth pairs")
    if not text_report_provided:
        reasons.append("text report not provided; text and hybrid are placeholders")
    elif not text_report_quality_valid:
        reasons.append("text report alignment/provenance is not quality-valid")
    if reasons:
        return "; ".join(reasons)
    if not catalog_quality_valid:  # defensive; all known reasons are above
        return "catalog metrics are not quality-valid"
    return "OK"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--verb-classes", type=Path, required=True)
    parser.add_argument("--noun-classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-report", type=Path)
    parser.add_argument("--catalog-pairs", type=Path)
    parser.add_argument("--full-cartesian", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument(
        "--dimension",
        type=int,
        default=2048,
        help="full WeMM-Embedding-2B dimension (MRL values such as 256 are allowed)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=16,
        help="bounded batch size for ontology text embeddings",
    )
    parser.add_argument(
        "--intervention",
        choices=("normal", "reverse", "freeze_pre", "freeze_post"),
        default="normal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--visual-weight", type=float, default=0.7)
    parser.add_argument("--text-weight", type=float, default=0.3)
    return parser


def _validate_run_args(args: argparse.Namespace) -> None:
    """Reject resource/metric settings that would make the report misleading."""

    frame_count = getattr(args, "frame_count", None)
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("frame_count must be an integer")
    if frame_count < 2 or frame_count > _MAX_VIDEO_FRAMES:
        raise ValueError(f"frame_count must be between 2 and {_MAX_VIDEO_FRAMES}")
    top_k = getattr(args, "top_k", None)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < max(_REPORT_KS):
        raise ValueError(
            f"top_k must be at least {max(_REPORT_KS)} to report Recall@{max(_REPORT_KS)}"
        )
    text_batch_size = getattr(args, "text_batch_size", None)
    if (
        isinstance(text_batch_size, bool)
        or not isinstance(text_batch_size, int)
        or text_batch_size <= 0
    ):
        raise ValueError("text_batch_size must be a positive integer")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_args(args)
    started = time.perf_counter()
    rows = _load_manifest(args.manifest, args.dataset_root, args.max_cases)
    verbs = _read_class_table(args.verb_classes)
    nouns = _read_class_table(args.noun_classes)
    pairs, catalog_source, catalog_uses_dev_labels = _catalog_pairs(
        rows,
        verbs,
        nouns,
        pair_file=args.catalog_pairs,
        full_cartesian=args.full_cartesian,
    )
    catalog_provenance = _catalog_provenance_details(
        pair_file=args.catalog_pairs,
        full_cartesian=args.full_cartesian,
        catalog_source=catalog_source,
        catalog_uses_dev_labels=catalog_uses_dev_labels,
    )
    # Evaluation-label frequencies are useful only for the explicitly
    # target-informed development smoke.  Never attach them to a full or
    # independently sourced catalog, where they would leak target priors into
    # an otherwise label-blind report.
    counts = (
        Counter(pair for row in rows if (pair := _pair_from_row(row)) is not None)
        if catalog_uses_dev_labels
        else None
    )
    labels = build_joint_action_catalog(
        verb_table_or_entries=verbs,
        noun_table_or_entries=nouns,
        action_pairs=pairs,
        observed_counts=counts,
    )
    catalog_target_coverage = _catalog_ground_truth_coverage(rows, labels)
    text_predictions, text_alignment_provenance = _load_and_align_text_predictions(
        args.text_report, rows
    )
    # Validity is mode-specific.  A visual-only invocation still emits the
    # legacy text/hybrid ranking keys for schema compatibility, but those rows
    # are placeholders until a quality-valid, label-blind text sidecar is
    # supplied.  Keep the booleans explicit so consumers cannot infer validity
    # from the mere presence of a metrics block.
    catalog_provenance_valid = bool(catalog_provenance.get("verified")) and not (
        catalog_uses_dev_labels
    )
    catalog_quality_valid = catalog_provenance_valid and bool(catalog_target_coverage["complete"])
    text_report_quality_valid = (
        bool(text_alignment_provenance.get("quality_valid"))
        if args.text_report is not None
        else False
    )
    mode_metrics_valid = {
        "visual": catalog_quality_valid,
        "text": catalog_quality_valid and text_report_quality_valid,
        "hybrid": catalog_quality_valid and text_report_quality_valid,
    }
    overall_metrics_valid = all(mode_metrics_valid.values())
    backend = WemmEmbeddingBackend(
        args.model_dir,
        device=args.device,
        dimension=args.dimension,
    )
    try:
        if backend.variant != "2B":
            raise WemmBackendUnavailable(
                "this benchmark runner is restricted to WeMM-Embedding-2B; "
                f"inferred variant={backend.variant}"
            )

        # Decode and encode one bounded interval at a time.  The previous
        # batch-at-front implementation retained every PIL frame until the
        # model call completed, allowing a large manifest to consume all host
        # memory.  A single group is still limited to 64 frames by the backend.
        query_embeddings: list[tuple[float, ...]] = []
        row_input_audit: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            frame_group, metadata = _decode_interval(
                Path(str(row["video_path"])),
                # ``_load_manifest`` enforces both fields for real runs.  The
                # defaults retain compatibility with injected decoder smoke
                # tests that provide only a synthetic ``video_path``.
                float(row.get("start_seconds", 0.0)),
                float(row.get("end_seconds", 0.0)),
                args.frame_count,
                intervention=args.intervention,
            )
            row_input_audit.append(
                _row_input_audit_entry(
                    row,
                    index,
                    metadata,
                    requested_frame_count=args.frame_count,
                    intervention=args.intervention,
                    decoded_frame_count=len(frame_group),
                )
            )
            try:
                encoded = backend.encode_video_frames((frame_group,), metadata_groups=(metadata,))
            finally:
                # Drop references even when processor/model encoding fails.
                frame_group.clear()
            if len(encoded) != 1:
                raise RuntimeError("WeMM query count does not match one manifest row")
            query_embeddings.extend(encoded)
        if len(query_embeddings) != len(rows):
            raise RuntimeError("WeMM query count does not match manifest count")

        variants: tuple[LabelVariant, ...] = ("canonical", "verb_noun", "natural")
        all_label_texts = [label.text_for(variant) for variant in variants for label in labels]
        all_label_vectors = backend.encode_texts(
            all_label_texts,
            batch_size=args.text_batch_size,
        )
        expected_label_vectors = len(variants) * len(labels)
        if len(all_label_vectors) != expected_label_vectors:
            raise RuntimeError(
                "WeMM label count does not match catalog: "
                f"{len(all_label_vectors)} != {expected_label_vectors}"
            )
        variant_results: dict[str, Any] = {}
        for variant_index, variant in enumerate(variants):
            start = variant_index * len(labels)
            label_vectors = all_label_vectors[start : start + len(labels)]
            if len(label_vectors) != len(labels):
                raise RuntimeError("WeMM label count does not match catalog count")
            label_embedding_map = {
                label.action_key: vector
                for label, vector in zip(labels, label_vectors, strict=True)
            }
            visual_rankings: dict[str, tuple[Any, ...]] = {}
            text_rankings: dict[str, tuple[Any, ...]] = {}
            hybrid_rankings: dict[str, tuple[Any, ...]] = {}
            projections: dict[str, dict[str, Any]] = {}
            for index, (row, query_embedding) in enumerate(
                zip(rows, query_embeddings, strict=True)
            ):
                key = _row_key(row, index)
                prediction = _text_prediction_for(rows, text_predictions, index)
                text_scores = text_scores_for_prediction(
                    prediction,
                    labels,
                    event_text=(
                        prediction.get("raw_text") if isinstance(prediction, Mapping) else None
                    ),
                )
                visual = rank_joint_actions(
                    labels=labels,
                    query_embedding=query_embedding,
                    label_embeddings=label_embedding_map,
                    label_variant=variant,
                    mode="visual",
                    top_k=args.top_k,
                )
                text = rank_joint_actions(
                    labels=labels,
                    label_variant=variant,
                    text_scores=text_scores,
                    mode="text",
                    top_k=args.top_k,
                )
                hybrid = rank_joint_actions(
                    labels=labels,
                    query_embedding=query_embedding,
                    label_embeddings=label_embedding_map,
                    label_variant=variant,
                    mode="hybrid",
                    visual_weight=args.visual_weight,
                    text_weight=args.text_weight,
                    top_k=args.top_k,
                    text_scores=text_scores,
                )
                visual_rankings[key] = visual
                text_rankings[key] = text
                hybrid_rankings[key] = hybrid
                projections[key] = {
                    "visual": project_retrieval_to_mapper(
                        visual, min_score=args.min_score, min_margin=args.min_margin
                    ),
                    "text": project_retrieval_to_mapper(
                        text, min_score=args.min_score, min_margin=args.min_margin
                    ),
                    "hybrid": project_retrieval_to_mapper(
                        hybrid, min_score=args.min_score, min_margin=args.min_margin
                    ),
                }
            rankings = {
                "visual": visual_rankings,
                "text": text_rankings,
                "hybrid": hybrid_rankings,
            }
            variant_results[variant] = {
                "catalog_size": len(labels),
                "metric_validity": dict(mode_metrics_valid),
                "rankings": {
                    mode: {
                        key: [item.to_dict() for item in ranking] for key, ranking in values.items()
                    }
                    for mode, values in rankings.items()
                },
                "metrics": compare_rankings(rows, rankings),
                "projections": projections,
            }
        report = {
            "report_version": "wemm-epic-retrieval-runner-v1",
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "production_eligible": False,
            "model": {
                "identity": backend.identity,
                "requested_variant": backend.variant,
                "supported_dimensions": list(backend.supported_dimensions),
                "larger_model_invoked": False,
                "dimension": args.dimension,
            },
            "input": {
                "manifest": str(args.manifest.expanduser().resolve()),
                "dataset_root": str(args.dataset_root.expanduser().resolve()),
                "case_count": len(rows),
                "frame_count": args.frame_count,
                "intervention": args.intervention,
                "catalog_source": catalog_source,
                "catalog_uses_development_labels": catalog_uses_dev_labels,
                "catalog_provenance_verified": catalog_provenance_valid,
                "catalog_provenance": catalog_provenance,
                "catalog_ground_truth_coverage": catalog_target_coverage,
                "catalog_size": len(labels),
                "label_variants": ["canonical", "verb_noun", "natural"],
                "text_batch_size": args.text_batch_size,
                "text_alignment_provenance": text_alignment_provenance,
                "metric_k_values": list(_REPORT_KS),
                "top_k_covers_reported_metrics": args.top_k >= max(_REPORT_KS),
                # One compact, JSON-native audit row per decoded interval.
                # ``video_interval_audit`` is the descriptive name; the
                # ``row_input_audit`` alias preserves compatibility with early
                # benchmark sidecars that used that spelling.
                "video_interval_audit": row_input_audit,
                "row_input_audit": row_input_audit,
            },
            "controls": {
                "ontology_modified": False,
                "mapper_modified": False,
                "mapper_training_invoked": False,
                "production_path_changed": False,
                "heldout_100_opened": False,
                "hash_or_sha_used": False,
                "ground_truth_used_in_encoder_input": False,
                "ground_truth_used_in_catalog": catalog_uses_dev_labels,
                "qwen_loaded_concurrently": False,
                "catalog_provenance_verified": catalog_provenance_valid,
                "catalog_target_coverage_complete": catalog_target_coverage["complete"],
                "text_report_label_blind": text_report_quality_valid,
                "visual_metrics_valid": mode_metrics_valid["visual"],
                "text_metrics_valid": mode_metrics_valid["text"],
                "hybrid_metrics_valid": mode_metrics_valid["hybrid"],
                "quality_metrics_valid": overall_metrics_valid,
            },
            "fusion": {
                "visual_weight": args.visual_weight,
                "text_weight": args.text_weight,
                "score_normalization": "visual cosine [-1,1] -> [0,1]; text score clipped [0,1]",
            },
            "quality_validity": {
                "catalog_metrics_valid": catalog_quality_valid,
                "visual_metrics_valid": mode_metrics_valid["visual"],
                "text_report_metrics_valid": text_report_quality_valid,
                "text_metrics_valid": mode_metrics_valid["text"],
                "hybrid_metrics_valid": mode_metrics_valid["hybrid"],
                "mode_metrics_valid": dict(mode_metrics_valid),
                "overall_metrics_valid": overall_metrics_valid,
                "reason": _quality_validity_reason(
                    catalog_quality_valid=catalog_quality_valid,
                    catalog_provenance_valid=catalog_provenance_valid,
                    catalog_target_coverage_complete=bool(catalog_target_coverage["complete"]),
                    text_report_provided=args.text_report is not None,
                    text_report_quality_valid=text_report_quality_valid,
                ),
            },
            "processor_observations": backend.observation_payload(),
            "labels": [label.to_dict() for label in labels],
            "results": variant_results,
            "elapsed_seconds": time.perf_counter() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return report
    finally:
        # Always release a resident model, including decode/processor/model
        # failures.  This runner is often invoked repeatedly for interventions.
        backend.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, ValueError, WemmRetrievalError, WemmBackendUnavailable) as exc:
        raise SystemExit(f"WeMM retrieval failed: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": report["input"]["case_count"],
                "catalog_size": report["input"]["catalog_size"],
                "intervention": report["input"]["intervention"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
