"""Read-only aggregation for production WeMM pre-annotation sidecars.

The production WeMM runner emits one review-only JSON envelope per recording.
This module deliberately sits *after* that runner: it discovers and validates
completed envelopes, then computes operational coverage and distribution
summaries.  It never opens media, invokes a model, reads a review bridge or
evaluator denominator, and never treats fixed processing windows as action
boundaries.

The input may be a sidecar, a batch-run checkpoint, or a directory containing
``preannotations/*.json``.  Review packs and arbitrary JSON files are ignored
when discovering a directory; when explicitly supplied they are reported as
rejected rather than being silently interpreted as pre-annotations.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .production_wemm_batch_runner import BATCH_RUN_FORMAT
from .production_wemm_preannotation import (
    FORMAT,
    ProductionWemmPreannotationError,
    validate_preannotation_envelope,
)

AGGREGATE_FORMAT: Final = "robata-production-wemm-preannotation-aggregate-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
STATUS: Final = "READ_ONLY_SUMMARY"
FIELDS: Final = ("verb", "noun", "attributes", "location", "hand")


class ProductionWemmPreannotationAggregateError(ValueError):
    """Raised for an invalid aggregate request (not for one bad sidecar)."""


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _load_file(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"could not read JSON: {exc}"
    if not isinstance(payload, Mapping):
        return None, "JSON root must be an object"
    return payload, None


def load_json(value: str | Path) -> dict[str, Any]:
    """Load one JSON object for callers that want explicit preflight control."""

    path = _resolve(value)
    payload, error = _load_file(path)
    if error is not None or payload is None:
        raise ProductionWemmPreannotationAggregateError(
            f"could not read JSON {path}: {error or 'invalid object'}"
        )
    return dict(payload)


def _resolve(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def _is_batch(value: Mapping[str, Any]) -> bool:
    fmt = value.get("format")
    return isinstance(fmt, str) and fmt == BATCH_RUN_FORMAT


def _is_preannotation(value: Mapping[str, Any]) -> bool:
    fmt = value.get("format")
    return isinstance(fmt, str) and fmt == FORMAT


def _discover_paths(path: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Discover only sidecars from a directory, preserving provenance."""

    rejected: list[dict[str, str]] = []
    if path.is_file():
        return [path], rejected
    if not path.exists():
        return [], [{"path": str(path), "reason": "PATH_NOT_FOUND"}]
    if not path.is_dir():
        return [], [{"path": str(path), "reason": "PATH_NOT_FILE_OR_DIRECTORY"}]

    found: set[Path] = set()
    # A checkpoint is useful even when sidecars are not directly below this
    # directory.  Only the conventional batch-run file is considered.
    for checkpoint in sorted(path.glob("batch-run*.json")):
        if checkpoint.is_file():
            found.add(checkpoint.resolve())
    # Nested preannotations are the only directory JSON files considered.
    for candidate in sorted(path.rglob("*.json")):
        if candidate.is_file() and any(
            parent.name.casefold() == "preannotations" for parent in candidate.parents
        ):
            found.add(candidate.resolve())
    if not found:
        rejected.append({"path": str(path), "reason": "NO_PREANNOTATION_OR_BATCH_CHECKPOINT_FOUND"})
    return sorted(found), rejected


def _iter_checkpoint_sidecars(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    rejected: list[dict[str, str]],
) -> Iterator[tuple[Path, Mapping[str, Any] | None, str | None]]:
    """Yield checkpoint sidecars one at a time to keep peak memory bounded."""

    items = _sequence(checkpoint.get("items"))
    for index, raw in enumerate(items):
        item = _mapping(raw)
        if item is None:
            rejected.append(
                {"path": f"{checkpoint_path}#items[{index}]", "reason": "ITEM_NOT_OBJECT"}
            )
            continue
        raw_path = item.get("preannotation_path")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            rejected.append(
                {
                    "path": f"{checkpoint_path}#items[{index}]",
                    "reason": "MISSING_PREANNOTATION_PATH",
                }
            )
            continue
        sidecar_path = _resolve(str(raw_path), base=checkpoint_path.parent)
        payload, error = _load_file(sidecar_path)
        yield sidecar_path, payload, error


def _iter_input_documents(
    inputs: Mapping[str, Any] | Sequence[Any] | str | Path,
    rejected: list[dict[str, str]],
) -> Iterator[tuple[str, Mapping[str, Any] | None, str | None]]:
    """Streaming counterpart to :func:`_input_documents` used by aggregation."""

    if isinstance(inputs, Mapping):
        if _is_batch(inputs):
            for path, payload, error in _iter_checkpoint_sidecars(
                inputs, Path("<mapping-checkpoint>"), rejected
            ):
                yield str(path), payload, error
        elif _is_preannotation(inputs):
            yield "<mapping-sidecar>", inputs, None
        else:
            rejected.append({"path": "<mapping>", "reason": "ROOT_NOT_PREANNOTATION_OR_CHECKPOINT"})
        return
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray, Path)):
        for index, raw in enumerate(inputs):
            if isinstance(raw, (str, Path)):
                candidate = _resolve(raw)
                payload, error = _load_file(candidate)
                if error is not None:
                    yield str(candidate), None, error
                elif _is_preannotation(payload or {}):
                    yield str(candidate), payload, None
                else:
                    rejected.append({"path": str(candidate), "reason": "ROOT_NOT_PREANNOTATION"})
                continue
            item = _mapping(raw)
            if item is None or not _is_preannotation(item):
                rejected.append(
                    {"path": f"<sequence>[{index}]", "reason": "ROOT_NOT_PREANNOTATION"}
                )
                continue
            yield f"<sequence>[{index}]", item, None
        return

    path = _resolve(inputs)  # type: ignore[arg-type]
    discovered, discovery_rejected = _discover_paths(path)
    rejected.extend(discovery_rejected)
    seen_sidecar_paths: set[Path] = set()
    for candidate in discovered:
        payload, error = _load_file(candidate)
        if error is not None:
            yield str(candidate), None, error
            continue
        if _is_batch(payload or {}):
            for sidecar, sidecar_payload, sidecar_error in _iter_checkpoint_sidecars(
                payload or {}, candidate, rejected
            ):
                resolved = sidecar.resolve()
                if resolved in seen_sidecar_paths:
                    continue
                seen_sidecar_paths.add(resolved)
                yield str(sidecar), sidecar_payload, sidecar_error
        elif _is_preannotation(payload or {}):
            resolved = candidate.resolve()
            if resolved not in seen_sidecar_paths:
                seen_sidecar_paths.add(resolved)
                yield str(candidate), payload, None
        else:
            rejected.append({"path": str(candidate), "reason": "ROOT_NOT_PREANNOTATION"})


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
    }


def _distribution(counter: Counter[str], *, limit: int = 50) -> dict[str, Any]:
    total = sum(counter.values())
    rows = [
        {"value": value, "count": count, "share": count / total if total else 0.0}
        for value, count in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]
    entropy = None
    if total:
        entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())
    return {"total": total, "unique": len(counter), "rows": rows, "entropy_bits": entropy}


def _field_bucket(status: object, value: object) -> str:
    token = str(status or "").upper()
    if token not in {"MEASURED", "NOT_MEASURED", "NOT_OBSERVABLE"}:
        return "MISSING"
    return token


def aggregate_production_wemm_preannotations(
    inputs: Mapping[str, Any] | Sequence[Any] | str | Path,
    *,
    expected_camera_count: int = 6,
) -> dict[str, Any]:
    """Aggregate completed sidecars without inference, media or gold access."""

    if (
        isinstance(expected_camera_count, bool)
        or not isinstance(expected_camera_count, int)
        or expected_camera_count <= 0
    ):
        raise ProductionWemmPreannotationAggregateError(
            "expected_camera_count must be a positive integer"
        )
    rejected_inputs: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    artifact_count = 0
    valid_artifact_count = 0

    recording_ids: Counter[str] = Counter()
    window_ids: Counter[str] = Counter()
    artifact_recordings: set[str] = set()
    declared_camera_ids: set[str] = set()
    observed_camera_ids: set[str] = set()
    # ``declared_camera_windows`` comes from the envelope's camera_ids and is
    # useful for provenance.  ``camera_windows`` is stricter: it counts only
    # raw input_observations, i.e. camera-window inputs actually observed by
    # the runner.
    declared_camera_windows: set[tuple[str, str]] = set()
    camera_windows: set[tuple[str, str]] = set()
    source_window_count = 0
    proposal_count = 0
    topk_cardinality: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    proposal_labels: Counter[str] = Counter()
    top1_labels: Counter[str] = Counter()
    proposal_verbs: Counter[str] = Counter()
    proposal_nouns: Counter[str] = Counter()
    top1_verbs: Counter[str] = Counter()
    top1_nouns: Counter[str] = Counter()
    scores: list[float] = []
    margins: list[float] = []
    missing_scores = 0
    missing_margins = 0
    interval_statuses: Counter[str] = Counter()
    source_interval_statuses: Counter[str] = Counter()
    field_stats: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    field_values: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    confidence_values: list[float] = []
    confidence_missing = 0
    evidence_present = 0
    evidence_missing = 0
    duplicate_recordings: list[str] = []
    duplicate_windows: list[str] = []
    observation_count = decoded_frame_count = examined_message_count = 0
    observation_with_failures = 0
    warning_count = 0
    warning_strings: Counter[str] = Counter()
    warning_types: Counter[str] = Counter()
    artifact_rows: list[dict[str, Any]] = []
    model_names: Counter[str] = Counter()
    model_routes: Counter[str] = Counter()
    catalog_formats: Counter[str] = Counter()
    catalog_phrase_counts: Counter[str] = Counter()
    catalog_epic_flags: Counter[str] = Counter()
    catalog_mapper_flags: Counter[str] = Counter()
    source_camera_counts: Counter[str] = Counter()

    for path, payload, error in _iter_input_documents(inputs, rejected_inputs):
        artifact_count += 1
        if error is not None or payload is None:
            invalid.append({"path": path, "reason": error or "INVALID_JSON"})
            continue
        if not _is_preannotation(payload):
            rejected_inputs.append({"path": path, "reason": "ROOT_NOT_PREANNOTATION"})
            continue
        try:
            # Validation is intentionally performed, but the detached copy
            # is not retained: large raw_model_output blocks may be tens of
            # megabytes per recording, so aggregation should remain bounded.
            validate_preannotation_envelope(payload)
            envelope = payload
        except (ProductionWemmPreannotationError, TypeError, ValueError) as exc:
            invalid.append({"path": path, "reason": str(exc)})
            continue
        valid_artifact_count += 1
        source = _mapping(envelope.get("source")) or {}
        recording_id = _text(source.get("recording_id") or source.get("source_id") or path)
        if recording_id is None:
            recording_id = path
        recording_ids[recording_id] += 1
        artifact_recordings.add(recording_id)
        source_camera_count = source.get("camera_count")
        if isinstance(source_camera_count, int) and not isinstance(source_camera_count, bool):
            source_camera_counts[str(source_camera_count)] += 1
        raw_windows = _sequence(envelope.get("windows"))
        source_window_count += len(raw_windows)
        model = _mapping(envelope.get("model")) or {}
        model_name = _text(model.get("name"))
        model_route = _text(model.get("route"))
        if model_name:
            model_names[model_name] += 1
        if model_route:
            model_routes[model_route] += 1
        catalog = _mapping((_mapping(envelope.get("raw_model_output")) or {}).get("catalog")) or {}
        catalog_format = _text(catalog.get("format"))
        if catalog_format:
            catalog_formats[catalog_format] += 1
        phrase_count = catalog.get("phrase_count")
        if isinstance(phrase_count, int) and not isinstance(phrase_count, bool):
            catalog_phrase_counts[str(phrase_count)] += 1
        for key, counter in (
            ("epic_ontology_used", catalog_epic_flags),
            ("mapper_used", catalog_mapper_flags),
        ):
            value = catalog.get(key)
            if isinstance(value, bool):
                counter[str(value).lower()] += 1
        artifact_rows.append(
            {
                "path": path,
                "recording_id": recording_id,
                "format": envelope.get("format"),
                "window_count": len(raw_windows),
                "proposal_count": sum(
                    len(_sequence((_mapping(window) or {}).get("proposals")))
                    for window in raw_windows
                ),
            }
        )
        raw_model = _mapping(envelope.get("raw_model_output")) or {}
        raw_model_windows = {
            str(item.get("window_id")): item
            for item in _sequence(raw_model.get("windows"))
            if isinstance(item, Mapping) and item.get("window_id") is not None
        }
        for raw_window in raw_windows:
            window = _mapping(raw_window)
            if window is None:
                continue  # validator normally makes this unreachable
            window_id = _text(window.get("window_id")) or "<missing-window>"
            window_ids[window_id] += 1
            cameras = [str(item) for item in _sequence(window.get("camera_ids")) if _text(item)]
            declared_camera_ids.update(cameras)
            for camera_id in cameras:
                declared_camera_windows.add((window_id, camera_id))
            source_interval = _mapping(window.get("source_interval")) or {}
            source_interval_statuses[str(source_interval.get("status") or "MISSING").upper()] += 1
            proposals = _sequence(window.get("proposals"))
            proposal_count += len(proposals)
            for raw_proposal in proposals:
                proposal = _mapping(raw_proposal)
                if proposal is None:
                    continue
                label = _text(proposal.get("label_text"))
                if label:
                    proposal_labels[label.casefold()] += 1
                labels = _mapping(proposal.get("structured_labels")) or {}
                for field in FIELDS:
                    cell = _mapping(labels.get(field)) or {}
                    status = _field_bucket(cell.get("status"), cell.get("value"))
                    field_stats[field][status] += 1
                    value = _text(cell.get("value"))
                    if status == "MEASURED" and value:
                        field_values[field][value.casefold()] += 1
                    if field == "verb" and value and status == "MEASURED":
                        proposal_verbs[value.casefold()] += 1
                    if field == "noun" and value and status == "MEASURED":
                        proposal_nouns[value.casefold()] += 1
                interval = _mapping(proposal.get("proposal_interval")) or {}
                interval_statuses[str(interval.get("status") or "MISSING").upper()] += 1
                confidence = _finite(proposal.get("confidence"))
                if confidence is None:
                    confidence_missing += 1
                else:
                    confidence_values.append(confidence)
                if _sequence(proposal.get("evidence")):
                    evidence_present += 1
                else:
                    evidence_missing += 1
                candidates = _sequence(proposal.get("top_k"))
                topk_cardinality[str(len(candidates))] += 1
                top1 = _mapping(candidates[0]) if candidates else None
                if top1 is not None:
                    top_label = _text(top1.get("label_text"))
                    if top_label:
                        top1_labels[top_label.casefold()] += 1
                    top_labels = _mapping(top1.get("structured_labels")) or {}
                    for field, counter in (("verb", top1_verbs), ("noun", top1_nouns)):
                        cell = _mapping(top_labels.get(field)) or {}
                        value = _text(cell.get("value"))
                        if value:
                            counter[value.casefold()] += 1
                for position, raw_candidate in enumerate(candidates, start=1):
                    candidate = _mapping(raw_candidate)
                    if candidate is None:
                        continue
                    rank = candidate.get("rank", position)
                    if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
                        rank_counts[str(rank)] += 1
                    score = _finite(candidate.get("score"))
                    if score is None:
                        missing_scores += 1
                    else:
                        scores.append(score)
                margin = _finite(proposal.get("margin"))
                if margin is None:
                    missing_margins += 1
                else:
                    margins.append(margin)

            raw_window_model = raw_model_windows.get(window_id) or {}
            observations = _sequence(raw_window_model.get("input_observations"))
            for raw_observation in observations:
                observation = _mapping(raw_observation)
                if observation is None:
                    continue
                observation_camera_id = _text(observation.get("camera_id"))
                observation_count += 1
                if observation_camera_id:
                    observed_camera_ids.add(observation_camera_id)
                    camera_windows.add((window_id, observation_camera_id))
                decoded = observation.get("decoded_frames")
                if isinstance(decoded, int) and not isinstance(decoded, bool):
                    decoded_frame_count += decoded
                examined = observation.get("messages_examined")
                if isinstance(examined, int) and not isinstance(examined, bool):
                    examined_message_count += examined
                failures = _sequence(observation.get("decode_failures"))
                if failures:
                    observation_with_failures += 1
                for failure in failures:
                    warning = _text(failure) or "UNKNOWN_DECODE_WARNING"
                    warning_count += 1
                    warning_strings[warning] += 1
                    warning_types[warning.split(":", 1)[0].strip()] += 1

    duplicate_recordings = sorted(key for key, count in recording_ids.items() if count > 1)
    duplicate_windows = sorted(key for key, count in window_ids.items() if count > 1)
    total_proposals = proposal_count

    def completion(field: str) -> dict[str, Any]:
        stats = field_stats[field]
        measured = stats.get("MEASURED", 0)
        return {
            "total": total_proposals,
            "status_counts": dict(sorted(stats.items())),
            "value_counts": _distribution(field_values[field]),
            "measured_count": measured,
            "measured_rate": measured / total_proposals if total_proposals else None,
        }

    expected_inputs = source_window_count * expected_camera_count
    coverage = {
        "artifact_count": artifact_count,
        "valid_artifact_count": valid_artifact_count,
        "invalid_artifact_count": len(invalid),
        "rejected_input_count": len(rejected_inputs),
        "recording_count": len(artifact_recordings),
        "window_count": source_window_count,
        "processing_window_count": source_window_count,
        "proposal_count": proposal_count,
        "observed_camera_count": len(observed_camera_ids),
        "declared_camera_count": len(declared_camera_ids),
        "expected_camera_count": expected_camera_count,
        "camera_window_input_count": len(camera_windows),
        "declared_camera_window_count": len(declared_camera_windows),
        "expected_camera_window_input_count": expected_inputs,
        "camera_window_coverage": (
            len(camera_windows) / expected_inputs if expected_inputs else None
        ),
        "observation_count": observation_count,
    }
    return {
        "format": AGGREGATE_FORMAT,
        "authority": AUTHORITY,
        "status": STATUS if valid_artifact_count else "NO_VALID_ARTIFACTS",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "inputs": {
            "kind": type(inputs).__name__,
            "source": str(inputs) if isinstance(inputs, (str, Path)) else "in_memory",
            "discovery": "preannotations_only_or_batch_checkpoint",
        },
        "coverage": coverage,
        "artifacts": artifact_rows,
        "provenance": {
            "model_names": dict(sorted(model_names.items())),
            "model_routes": dict(sorted(model_routes.items())),
            "catalog_formats": dict(sorted(catalog_formats.items())),
            "catalog_phrase_counts": dict(
                sorted(catalog_phrase_counts.items(), key=lambda pair: int(pair[0]))
            ),
            "catalog_epic_ontology_used": dict(sorted(catalog_epic_flags.items())),
            "catalog_mapper_used": dict(sorted(catalog_mapper_flags.items())),
            "source_camera_counts": dict(
                sorted(source_camera_counts.items(), key=lambda pair: int(pair[0]))
            ),
        },
        "top_k": {
            "proposal_count": proposal_count,
            "cardinality_histogram": dict(
                sorted(topk_cardinality.items(), key=lambda pair: int(pair[0]))
            ),
            "rank_counts": dict(sorted(rank_counts.items(), key=lambda pair: int(pair[0]))),
            "score_kind": "retrieval_score_not_probability",
            "score_summary": _summary(scores),
            "margin_summary": _summary(margins),
            "missing_score_count": missing_scores,
            "missing_margin_count": missing_margins,
            "confidence_summary": _summary(confidence_values),
            "confidence_missing_count": confidence_missing,
        },
        "vocabulary_bias": {
            "proposal_labels": _distribution(proposal_labels),
            "top1_labels": _distribution(top1_labels),
            "proposal_verbs": _distribution(proposal_verbs),
            "proposal_nouns": _distribution(proposal_nouns),
            "top1_verbs": _distribution(top1_verbs),
            "top1_nouns": _distribution(top1_nouns),
        },
        "field_completeness": {
            "fields": {field: completion(field) for field in FIELDS},
            "proposal_interval": {
                "total": proposal_count,
                "status_counts": dict(sorted(interval_statuses.items())),
                "action_boundaries_inferred": False,
            },
            "evidence": {
                "present_count": evidence_present,
                "missing_count": evidence_missing,
                "present_rate": evidence_present / proposal_count if proposal_count else None,
            },
        },
        "decode_warnings": {
            "observation_count": observation_count,
            "warning_bearing_camera_window_count": observation_with_failures,
            "warning_entry_count": warning_count,
            "warning_strings": dict(sorted(warning_strings.items())),
            "warning_types": dict(sorted(warning_types.items())),
            "decoded_frame_count": decoded_frame_count,
            "examined_message_count": examined_message_count,
        },
        "source_window_status": dict(sorted(source_interval_statuses.items())),
        "duplicates": {
            "recording_ids": duplicate_recordings,
            "window_ids": duplicate_windows,
            "duplicate_recording_count": len(duplicate_recordings),
            "duplicate_window_count": len(duplicate_windows),
        },
        "invalid_artifacts": invalid,
        "rejected_inputs": rejected_inputs,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "review_bridge_read": False,
            "evaluator_invoked": False,
            "window_boundaries_as_actions": False,
            "hash_or_sha_used": False,
            "inference_performed_for_summary": False,
        },
        "limitations": [
            "No official/source-bound gold was read; quality is NOT_MEASURED.",
            "Processing windows are compute/review units, not inferred action boundaries.",
            "Scores and margins are retrieval scores, not calibrated probabilities.",
            "Review packs, bridge artifacts and evaluator denominators are not read.",
            "Invalid or duplicate artifacts are reported and never silently merged.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, auditable report without embedding raw model output."""

    coverage = _mapping(report.get("coverage")) or {}
    topk = _mapping(report.get("top_k")) or {}
    decode = _mapping(report.get("decode_warnings")) or {}
    vocab = _mapping(report.get("vocabulary_bias")) or {}
    fields = _mapping(report.get("field_completeness")) or {}
    provenance = _mapping(report.get("provenance")) or {}
    lines = [
        "# Production WeMM pre-annotation aggregate",
        "",
        f"- Status: **{report.get('status', 'UNKNOWN')}**",
        f"- Official quality: **{report.get('official_quality_status', 'NOT_MEASURED')}**",
        f"- Official gold: **{report.get('official_gold_status', 'NOT_ESTABLISHED')}**",
        f"- Model(s): `{json.dumps(provenance.get('model_names', {}), sort_keys=True)}`",
        f"- Catalog(s): `{json.dumps(provenance.get('catalog_formats', {}), sort_keys=True)}`",
        "",
        "## Coverage",
        "",
        f"- Artifacts: {coverage.get('valid_artifact_count', 0)} valid / "
        f"{coverage.get('invalid_artifact_count', 0)} invalid",
        f"- Recordings: {coverage.get('recording_count', 0)}",
        f"- Processing windows (not action spans): {coverage.get('processing_window_count', 0)}",
        f"- Proposals: {coverage.get('proposal_count', 0)}",
        f"- Camera-window inputs: {coverage.get('camera_window_input_count', 0)} / "
        f"{coverage.get('expected_camera_window_input_count', 0)} expected",
        "",
        "## Top-K and margins",
        "",
        f"- Score summary: `{json.dumps(topk.get('score_summary', {}), sort_keys=True)}`",
        f"- Margin summary: `{json.dumps(topk.get('margin_summary', {}), sort_keys=True)}`",
        "- Top-K cardinality: "
        f"`{json.dumps(topk.get('cardinality_histogram', {}), sort_keys=True)}`",
        "",
        "## Vocabulary bias",
        "",
        f"- Proposal labels: `{json.dumps(vocab.get('proposal_labels', {}), sort_keys=True)}`",
        f"- Top-1 labels: `{json.dumps(vocab.get('top1_labels', {}), sort_keys=True)}`",
        "",
        "## Structured-field completeness",
        "",
    ]
    field_rows = _mapping(fields.get("fields")) or {}
    for field in FIELDS:
        row = _mapping(field_rows.get(field)) or {}
        lines.append(
            f"- `{field}` measured: {row.get('measured_count', 0)} / {row.get('total', 0)}"
        )
    lines.extend(
        [
            "",
            "## Decode warnings",
            "",
            f"- Observations: {decode.get('observation_count', 0)}; warning-bearing: "
            f"{decode.get('warning_bearing_camera_window_count', 0)}",
            f"- Warning entries: {decode.get('warning_entry_count', 0)}",
            f"- Warning types: `{json.dumps(decode.get('warning_types', {}), sort_keys=True)}`",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in _sequence(report.get("limitations")):
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


__all__ = [
    "AGGREGATE_FORMAT",
    "AUTHORITY",
    "ProductionWemmPreannotationAggregateError",
    "aggregate_production_wemm_preannotations",
    "load_json",
    "render_markdown",
]
