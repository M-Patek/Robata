"""Build an offline A/B report for Mage DCVC Provider V2.

The tool never starts a GPU process. Inputs are retained measurements for the same
exact 40-second/five-segment sample. One is observed-v1; Provider V2 inputs contain
one max_side=0 control and one or more bounded candidates. seq_len_frames is never
treated as a limit on recurrent DCVC work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256  # noqa: E402

EVIDENCE_VERSION = "mage-dcvc-provider-qualification-evidence-v1"
REPORT_VERSION = "mage-dcvc-provider-v2-qualification-report-v1"
BASELINE = "OBSERVED_V1"
PROVIDER_V2 = "PROVIDER_V2"
RECURRENT_WORK = "FULL_RECURRENT_CHAIN_THROUGH_LAST_SAMPLED_FRAME"
DURATION_NS = 40_000_000_000
SEGMENT_COUNT = 5
MIN_FULL_SPEEDUP = 1.05
MIN_BOUNDED_SPEEDUP = 1.25
MAX_VRAM = 0.85
MAX_TEMP_C = 85.0
MAX_BOUNDARY_DRIFT_S = 0.25
MAX_CONFIDENCE_DELTA = 0.05
KINDS = ("qa", "event", "evidence", "track", "fusion")
HEX = frozenset("0123456789abcdef")


class MageDcvcQualificationInputError(ValueError):
    """Raised when retained evidence is unsafe or incomparable."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evidence", type=Path)
    parser.add_argument("--provider-v2-evidence", action="append", type=Path)
    parser.add_argument("--observed-preparation-dir", type=Path)
    parser.add_argument("--observed-generation-dir", type=Path)
    parser.add_argument("--provider-v2-full-preparation-dir", type=Path)
    parser.add_argument("--provider-v2-full-generation-dir", type=Path)
    parser.add_argument("--provider-v2-bounded-preparation-dir", action="append", type=Path)
    parser.add_argument("--provider-v2-bounded-generation-dir", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MageDcvcQualificationInputError(f"could not read evidence: {path}") from error
    if not isinstance(value, dict):
        raise MageDcvcQualificationInputError(f"evidence must be an object: {path}")
    return value


def _obj(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MageDcvcQualificationInputError(f"{label} must be an object")
    return value


def _arr(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise MageDcvcQualificationInputError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MageDcvcQualificationInputError(f"{label} must be a nonempty string")
    return value


def _sha(value: object, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(character not in HEX for character in digest):
        raise MageDcvcQualificationInputError(f"{label} must be a lowercase SHA-256")
    return digest


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MageDcvcQualificationInputError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MageDcvcQualificationInputError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise MageDcvcQualificationInputError(f"{label} must be finite and >= {minimum}")
    return result


def _confidence(value: object, label: str) -> float:
    result = _number(value, label)
    if result > 1.0:
        raise MageDcvcQualificationInputError(f"{label} must be <= 1")
    return result


def _required(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    if missing:
        raise MageDcvcQualificationInputError(f"{label} missing required keys: {sorted(missing)}")


def _validate_telemetry(value: object, label: str) -> None:
    telemetry = _obj(value, label)
    _required(telemetry, {"measurement_status", "device_summaries", "errors"}, label)
    status = _text(telemetry["measurement_status"], f"{label}.measurement_status")
    if status not in {"MEASURED", "PARTIAL", "UNAVAILABLE"}:
        raise MageDcvcQualificationInputError(f"{label} status is unsupported")
    devices = _arr(telemetry["device_summaries"], f"{label}.device_summaries")
    if status == "MEASURED" and not devices:
        raise MageDcvcQualificationInputError(f"{label} measured telemetry needs a device")
    if status == "UNAVAILABLE" and devices:
        raise MageDcvcQualificationInputError(f"{label} unavailable telemetry has devices")
    for index, raw in enumerate(devices):
        device = _obj(raw, f"{label}.device_summaries[{index}]")
        fields = {
            "gpu_index",
            "gpu_name",
            "utilization_gpu_percent_mean",
            "utilization_gpu_percent_max",
            "memory_used_fraction_max",
            "memory_used_mib_max",
            "memory_total_mib",
            "temperature_celsius_max",
            "power_draw_watts_max",
            "sample_count",
        }
        _required(device, fields, f"{label}.device_summaries[{index}]")
        _integer(device["gpu_index"], f"{label}.gpu_index")
        _text(device["gpu_name"], f"{label}.gpu_name")
        for field in ("utilization_gpu_percent_mean", "utilization_gpu_percent_max"):
            if _number(device[field], f"{label}.{field}") > 100:
                raise MageDcvcQualificationInputError(f"{label}.{field} exceeds 100")
        if _number(device["memory_used_fraction_max"], f"{label}.memory") > 1:
            raise MageDcvcQualificationInputError(f"{label} memory fraction exceeds one")
        used = _integer(device["memory_used_mib_max"], f"{label}.used_mib")
        total = _integer(device["memory_total_mib"], f"{label}.total_mib", 1)
        if used > total:
            raise MageDcvcQualificationInputError(f"{label} used memory exceeds total")
        _number(device["temperature_celsius_max"], f"{label}.temperature")
        _number(device["power_draw_watts_max"], f"{label}.power")
        _integer(device["sample_count"], f"{label}.sample_count", 1)
    for index, error in enumerate(_arr(telemetry["errors"], f"{label}.errors")):
        _text(error, f"{label}.errors[{index}]")


def _validate_projection_atom(value: object, label: str, kind: str) -> None:
    atom = _obj(value, label)
    _required(atom, {"match_key", "semantic_sha256"}, label)
    _text(atom["match_key"], f"{label}.match_key")
    _sha(atom["semantic_sha256"], f"{label}.semantic_sha256")
    if kind in {"event", "track"}:
        _required(atom, {"label", "start_ns", "end_ns", "confidence"}, label)
        _text(atom["label"], f"{label}.label")
    elif kind == "evidence":
        _required(atom, {"supports", "confidence"}, label)
        if not isinstance(atom["supports"], bool):
            raise MageDcvcQualificationInputError(f"{label}.supports must be boolean")
    elif kind == "fusion":
        _required(atom, {"disposition", "start_ns", "end_ns", "confidence"}, label)
        _text(atom["disposition"], f"{label}.disposition")
    if "start_ns" in atom:
        start = _integer(atom["start_ns"], f"{label}.start_ns")
        end = _integer(atom["end_ns"], f"{label}.end_ns", 1)
        if end <= start or end > DURATION_NS:
            raise MageDcvcQualificationInputError(f"{label} has an invalid interval")
    if "confidence" in atom:
        _confidence(atom["confidence"], f"{label}.confidence")


def _validate_evidence(evidence: Mapping[str, Any], label: str) -> None:
    top = {
        "format_version",
        "evidence_class",
        "production_eligible",
        "variant_id",
        "provider",
        "sample",
        "control_identity",
        "cold_preparation",
        "endpoint",
        "projected_semantics",
    }
    _required(evidence, top, label)
    if evidence["format_version"] != EVIDENCE_VERSION:
        raise MageDcvcQualificationInputError(f"{label}.format_version is unsupported")
    if (
        evidence["evidence_class"] != "LOCAL_MEASUREMENT"
        or evidence["production_eligible"] is not False
    ):
        raise MageDcvcQualificationInputError(f"{label} must be local and production-ineligible")
    _text(evidence["variant_id"], f"{label}.variant_id")

    provider = _obj(evidence["provider"], f"{label}.provider")
    provider_fields = {
        "generation",
        "recipe_version",
        "implementation_sha256",
        "effective_config_sha256",
        "cache_namespace_identity",
        "max_side",
        "recurrent_work_semantics",
        "sequence_length_is_compute_cap",
        "inference_identity_binds_provider_recipe",
    }
    _required(provider, provider_fields, f"{label}.provider")
    if provider["generation"] not in {BASELINE, PROVIDER_V2}:
        raise MageDcvcQualificationInputError(f"{label} provider generation is unsupported")
    _text(provider["recipe_version"], f"{label}.provider.recipe_version")
    for field in ("implementation_sha256", "effective_config_sha256", "cache_namespace_identity"):
        _sha(provider[field], f"{label}.provider.{field}")
    _integer(provider["max_side"], f"{label}.provider.max_side")
    if provider["recurrent_work_semantics"] != RECURRENT_WORK:
        raise MageDcvcQualificationInputError(
            f"{label} must state {RECURRENT_WORK}; sampled frames do not cap recurrent work"
        )
    if provider["sequence_length_is_compute_cap"] is not False:
        raise MageDcvcQualificationInputError(
            f"{label} may not claim seq_len_frames limits DCVC recurrent work"
        )
    if not isinstance(provider["inference_identity_binds_provider_recipe"], bool):
        raise MageDcvcQualificationInputError(f"{label} identity binding flag must be boolean")

    sample = _obj(evidence["sample"], f"{label}.sample")
    _required(
        sample,
        {"source_media_sha256", "segment_manifest_sha256", "media_duration_ns", "segments"},
        f"{label}.sample",
    )
    _sha(sample["source_media_sha256"], f"{label}.sample.source_media_sha256")
    _sha(sample["segment_manifest_sha256"], f"{label}.sample.segment_manifest_sha256")
    if sample["media_duration_ns"] != DURATION_NS:
        raise MageDcvcQualificationInputError(f"{label} must use the exact 40-second sample")
    segments = _arr(sample["segments"], f"{label}.sample.segments")
    if len(segments) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError(f"{label} must have exactly five segments")
    expected_start = 0
    for ordinal, raw in enumerate(segments):
        segment = _obj(raw, f"{label}.sample.segments[{ordinal}]")
        _required(
            segment,
            {"ordinal", "start_ns", "end_ns", "source_content_sha256", "source_byte_count"},
            f"{label}.segment",
        )
        if segment["ordinal"] != ordinal or segment["start_ns"] != expected_start:
            raise MageDcvcQualificationInputError(
                f"{label} segments must be ordered and contiguous"
            )
        end = _integer(segment["end_ns"], f"{label}.segment.end_ns", 1)
        if end <= expected_start:
            raise MageDcvcQualificationInputError(f"{label} segment duration must be positive")
        expected_start = end
        _sha(segment["source_content_sha256"], f"{label}.segment.content_sha256")
        _integer(segment["source_byte_count"], f"{label}.segment.byte_count", 1)
    if expected_start != DURATION_NS:
        raise MageDcvcQualificationInputError(f"{label} segments must cover 40 seconds")

    control = _obj(evidence["control_identity"], f"{label}.control_identity")
    control_fields = {
        "checkpoint_manifest_sha256",
        "model_identity_sha256",
        "prompt_sha256",
        "decoder_identity_sha256",
        "max_new_tokens",
    }
    _required(control, control_fields, f"{label}.control_identity")
    for field in control_fields - {"max_new_tokens"}:
        _sha(control[field], f"{label}.control_identity.{field}")
    _integer(control["max_new_tokens"], f"{label}.control_identity.max_new_tokens", 1)

    cold = _obj(evidence["cold_preparation"], f"{label}.cold_preparation")
    cold_fields = {
        "wall_seconds",
        "provider_process_start_count",
        "provider_model_load_count",
        "provider_startup_seconds",
        "provider_model_load_seconds",
        "startup_load_included_in_wall",
        "per_segment",
        "gpu_telemetry",
    }
    _required(cold, cold_fields, f"{label}.cold_preparation")
    _number(cold["wall_seconds"], f"{label}.cold.wall_seconds", 1e-12)
    _integer(cold["provider_process_start_count"], f"{label}.cold.start_count", 1)
    _integer(cold["provider_model_load_count"], f"{label}.cold.load_count", 1)
    _number(cold["provider_startup_seconds"], f"{label}.cold.startup_seconds")
    _number(cold["provider_model_load_seconds"], f"{label}.cold.load_seconds")
    if not isinstance(cold["startup_load_included_in_wall"], bool):
        raise MageDcvcQualificationInputError(f"{label} startup inclusion flag must be boolean")
    prepared = _arr(cold["per_segment"], f"{label}.cold.per_segment")
    if len(prepared) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError(f"{label} needs five preparation rows")
    for ordinal, raw in enumerate(prepared):
        item = _obj(raw, f"{label}.cold.per_segment[{ordinal}]")
        fields = {
            "ordinal",
            "preparation_seconds",
            "asset_identity_sha256",
            "asset_exact_sha256",
            "meta_identity_sha256",
            "meta_exact_sha256",
            "meta_semantic_sha256",
            "verified",
        }
        _required(item, fields, f"{label}.cold.per_segment[{ordinal}]")
        if item["ordinal"] != ordinal:
            raise MageDcvcQualificationInputError(f"{label} preparation rows must be ordered")
        _number(item["preparation_seconds"], f"{label}.prep.seconds", 1e-12)
        for field in fields - {"ordinal", "preparation_seconds", "verified"}:
            _sha(item[field], f"{label}.prep.{field}")
        if not isinstance(item["verified"], bool):
            raise MageDcvcQualificationInputError(f"{label}.prep.verified must be boolean")
    _validate_telemetry(cold["gpu_telemetry"], f"{label}.cold.gpu_telemetry")

    endpoint = _obj(evidence["endpoint"], f"{label}.endpoint")
    endpoint_fields = {
        "timed_wall_seconds",
        "model_load_seconds",
        "model_load_included_in_wall",
        "per_segment",
        "gpu_telemetry",
    }
    _required(endpoint, endpoint_fields, f"{label}.endpoint")
    _number(endpoint["timed_wall_seconds"], f"{label}.endpoint.wall_seconds", 1e-12)
    _number(endpoint["model_load_seconds"], f"{label}.endpoint.model_load_seconds")
    if not isinstance(endpoint["model_load_included_in_wall"], bool):
        raise MageDcvcQualificationInputError(f"{label} endpoint load inclusion must be boolean")
    results = _arr(endpoint["per_segment"], f"{label}.endpoint.per_segment")
    if len(results) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError(f"{label} needs five endpoint rows")
    result_fields = {
        "ordinal",
        "inference_identity_sha256",
        "result_artifact_identity_sha256",
        "result_artifact_exact_sha256",
        "output_text_sha256",
        "observation_semantic_sha256",
    }
    for ordinal, raw in enumerate(results):
        result = _obj(raw, f"{label}.endpoint.per_segment[{ordinal}]")
        _required(result, result_fields, f"{label}.endpoint.per_segment[{ordinal}]")
        if result["ordinal"] != ordinal:
            raise MageDcvcQualificationInputError(f"{label} endpoint rows must be ordered")
        for field in result_fields - {"ordinal"}:
            _sha(result[field], f"{label}.endpoint.{field}")
    _validate_telemetry(endpoint["gpu_telemetry"], f"{label}.endpoint.gpu_telemetry")

    projections = _obj(evidence["projected_semantics"], f"{label}.projected_semantics")
    _required(projections, set(KINDS), f"{label}.projected_semantics")
    for kind in KINDS:
        seen: set[str] = set()
        for ordinal, raw in enumerate(_arr(projections[kind], f"{label}.{kind}")):
            atom_label = f"{label}.projected_semantics.{kind}[{ordinal}]"
            _validate_projection_atom(raw, atom_label, kind)
            key = str(_obj(raw, atom_label)["match_key"])
            if key in seen:
                raise MageDcvcQualificationInputError(f"{label}.{kind} duplicate match_key {key}")
            seen.add(key)


def _sample_identity(evidence: Mapping[str, Any]) -> tuple[object, ...]:
    sample = _obj(evidence["sample"], "sample")
    segments = tuple(
        tuple(
            _obj(raw, "segment")[field]
            for field in (
                "ordinal",
                "start_ns",
                "end_ns",
                "source_content_sha256",
                "source_byte_count",
            )
        )
        for raw in _arr(sample["segments"], "segments")
    )
    return (
        sample["source_media_sha256"],
        sample["segment_manifest_sha256"],
        sample["media_duration_ns"],
        segments,
    )


def _control_identity(evidence: Mapping[str, Any]) -> tuple[object, ...]:
    control = _obj(evidence["control_identity"], "control_identity")
    fields = (
        "checkpoint_manifest_sha256",
        "model_identity_sha256",
        "prompt_sha256",
        "decoder_identity_sha256",
        "max_new_tokens",
    )
    return tuple(control[field] for field in fields)


def _telemetry_rollup(*documents: Mapping[str, Any]) -> dict[str, object]:
    statuses = [str(document["measurement_status"]) for document in documents]
    devices = [
        _obj(raw, "device")
        for document in documents
        for raw in _arr(document["device_summaries"], "device_summaries")
    ]
    total_samples = sum(int(device["sample_count"]) for device in devices)
    weighted_utilization = None
    if total_samples:
        weighted_utilization = (
            sum(
                float(device["utilization_gpu_percent_mean"]) * int(device["sample_count"])
                for device in devices
            )
            / total_samples
        )
    return {
        "measurement_statuses": statuses,
        "complete": all(status == "MEASURED" for status in statuses),
        "device_summaries": [dict(device) for device in devices],
        "sample_count": total_samples,
        "utilization_gpu_percent_weighted_mean": weighted_utilization,
        "utilization_gpu_percent_max": max(
            (float(device["utilization_gpu_percent_max"]) for device in devices), default=None
        ),
        "memory_used_fraction_max": max(
            (float(device["memory_used_fraction_max"]) for device in devices), default=None
        ),
        "memory_used_mib_max": max(
            (int(device["memory_used_mib_max"]) for device in devices), default=None
        ),
        "temperature_celsius_max": max(
            (float(device["temperature_celsius_max"]) for device in devices), default=None
        ),
        "power_draw_watts_max": max(
            (float(device["power_draw_watts_max"]) for device in devices), default=None
        ),
        "errors": [
            str(error) for document in documents for error in _arr(document["errors"], "errors")
        ],
    }


def _variant_summary(evidence: Mapping[str, Any]) -> dict[str, object]:
    provider = _obj(evidence["provider"], "provider")
    sample = _obj(evidence["sample"], "sample")
    cold = _obj(evidence["cold_preparation"], "cold_preparation")
    endpoint = _obj(evidence["endpoint"], "endpoint")
    media_seconds = int(sample["media_duration_ns"]) / 1_000_000_000
    cold_wall = float(cold["wall_seconds"])
    endpoint_hot_wall = float(endpoint["timed_wall_seconds"])
    endpoint_cold_wall = endpoint_hot_wall + (
        0.0 if endpoint["model_load_included_in_wall"] else float(endpoint["model_load_seconds"])
    )
    total_wall = cold_wall + endpoint_cold_wall
    telemetry = _telemetry_rollup(
        _obj(cold["gpu_telemetry"], "cold.gpu_telemetry"),
        _obj(endpoint["gpu_telemetry"], "endpoint.gpu_telemetry"),
    )
    projections = _obj(evidence["projected_semantics"], "projected_semantics")
    return {
        "variant_id": evidence["variant_id"],
        "provider": dict(provider),
        "sample": {
            "source_media_sha256": sample["source_media_sha256"],
            "segment_manifest_sha256": sample["segment_manifest_sha256"],
            "media_duration_seconds": media_seconds,
            "segment_count": SEGMENT_COUNT,
        },
        "cold_preparation": {
            "wall_seconds": cold_wall,
            "realtime_factor": media_seconds / cold_wall,
            "slower_than_realtime_ratio": cold_wall / media_seconds,
            "provider_process_start_count": cold["provider_process_start_count"],
            "provider_model_load_count": cold["provider_model_load_count"],
            "provider_startup_seconds": cold["provider_startup_seconds"],
            "provider_model_load_seconds": cold["provider_model_load_seconds"],
            "startup_load_included_in_wall": cold["startup_load_included_in_wall"],
            "per_segment": [
                dict(_obj(raw, "preparation")) for raw in _arr(cold["per_segment"], "per_segment")
            ],
        },
        "endpoint": {
            "timed_wall_seconds": endpoint_hot_wall,
            "hot_realtime_factor": media_seconds / endpoint_hot_wall,
            "model_load_seconds": endpoint["model_load_seconds"],
            "model_load_included_in_wall": endpoint["model_load_included_in_wall"],
            "cold_wall_including_model_load_seconds": endpoint_cold_wall,
            "per_segment": [
                dict(_obj(raw, "endpoint result"))
                for raw in _arr(endpoint["per_segment"], "per_segment")
            ],
        },
        "cold_to_result": {
            "wall_seconds": total_wall,
            "realtime_factor": media_seconds / total_wall,
        },
        "gpu_telemetry": telemetry,
        "projected_semantics": {
            kind: [dict(_obj(raw, kind)) for raw in _arr(projections[kind], kind)] for kind in KINDS
        },
    }


def _projection_map(evidence: Mapping[str, Any], kind: str) -> dict[str, Mapping[str, Any]]:
    projections = _obj(evidence["projected_semantics"], "projected_semantics")
    return {
        str(atom["match_key"]): atom
        for atom in (_obj(raw, kind) for raw in _arr(projections[kind], kind))
    }


def _ratio(numerator: int, denominator: int, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def _projection_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], kind: str
) -> dict[str, object]:
    left = _projection_map(baseline, kind)
    right = _projection_map(candidate, kind)
    left_keys, right_keys = set(left), set(right)
    common = sorted(left_keys & right_keys)
    precision = _ratio(len(common), len(right_keys), float(not left_keys))
    recall = _ratio(len(common), len(left_keys), float(not right_keys))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, object] = {
        "baseline_count": len(left_keys),
        "candidate_count": len(right_keys),
        "matched_count": len(common),
        "missing_match_keys": sorted(left_keys - right_keys),
        "extra_match_keys": sorted(right_keys - left_keys),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_semantic_agreement": _ratio(
            sum(left[key]["semantic_sha256"] == right[key]["semantic_sha256"] for key in common),
            len(common),
        ),
    }
    if kind in {"event", "track"}:
        result["label_agreement"] = _ratio(
            sum(left[key]["label"] == right[key]["label"] for key in common), len(common)
        )
    if kind == "evidence":
        result["support_agreement"] = _ratio(
            sum(left[key]["supports"] == right[key]["supports"] for key in common), len(common)
        )
    if kind == "fusion":
        result["disposition_agreement"] = _ratio(
            sum(left[key]["disposition"] == right[key]["disposition"] for key in common),
            len(common),
        )
    if kind in {"event", "track", "fusion"}:
        result["maximum_boundary_drift_seconds"] = max(
            (
                max(
                    abs(int(left[key]["start_ns"]) - int(right[key]["start_ns"])),
                    abs(int(left[key]["end_ns"]) - int(right[key]["end_ns"])),
                )
                / 1_000_000_000
                for key in common
            ),
            default=0.0,
        )
    if kind in {"event", "evidence", "track", "fusion"}:
        result["maximum_confidence_delta"] = max(
            (
                abs(float(left[key]["confidence"]) - float(right[key]["confidence"]))
                for key in common
            ),
            default=0.0,
        )
    return result


def _gate(
    gate_id: str, passed: bool, observed: object, threshold: object, detail: str
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def _prep_rows(evidence: Mapping[str, Any]) -> Sequence[object]:
    return _arr(_obj(evidence["cold_preparation"], "cold")["per_segment"], "per_segment")


def _endpoint_rows(evidence: Mapping[str, Any]) -> Sequence[object]:
    return _arr(_obj(evidence["endpoint"], "endpoint")["per_segment"], "per_segment")


def _prep_hashes(evidence: Mapping[str, Any], field: str) -> tuple[str, ...]:
    return tuple(str(_obj(raw, "preparation")[field]) for raw in _prep_rows(evidence))


def _endpoint_hashes(evidence: Mapping[str, Any], field: str) -> tuple[str, ...]:
    return tuple(str(_obj(raw, "endpoint")[field]) for raw in _endpoint_rows(evidence))


def _asset_hashes(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    return _prep_hashes(evidence, "asset_exact_sha256")


def _output_hashes(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    return _endpoint_hashes(evidence, "output_text_sha256")


def _candidate_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    baseline_provider = _obj(baseline["provider"], "baseline.provider")
    provider = _obj(candidate["provider"], "candidate.provider")
    baseline_cold = _obj(baseline["cold_preparation"], "baseline.cold")
    cold = _obj(candidate["cold_preparation"], "candidate.cold")
    summary = _variant_summary(candidate)
    telemetry = _obj(summary["gpu_telemetry"], "gpu_telemetry")
    full = int(provider["max_side"]) == 0
    speedup = float(baseline_cold["wall_seconds"]) / float(cold["wall_seconds"])
    deltas = {kind: _projection_delta(baseline, candidate, kind) for kind in KINDS}
    verified = all(bool(_obj(raw, "preparation")["verified"]) for raw in _prep_rows(candidate))
    gates = [
        _gate(
            "SAME_SAMPLE",
            _sample_identity(baseline) == _sample_identity(candidate),
            _sample_identity(candidate)[:3],
            "exact baseline identity and segment sequence",
            "source, manifest, bounds, content hashes, and byte counts must match",
        ),
        _gate(
            "CONTROL_IDENTITY_PARITY",
            _control_identity(baseline) == _control_identity(candidate),
            dict(_obj(candidate["control_identity"], "control_identity")),
            "exact checkpoint/model/prompt/decoder/token parity",
            "the provider is the only intended experimental variable",
        ),
        _gate(
            "PROVIDER_RECIPE_IDENTITY_BOUND",
            bool(provider["inference_identity_binds_provider_recipe"]),
            provider["inference_identity_binds_provider_recipe"],
            True,
            "recipe, effective config, and implementation must enter inference identity",
        ),
        _gate(
            "CACHE_NAMESPACE_ISOLATION",
            provider["cache_namespace_identity"] != baseline_provider["cache_namespace_identity"],
            provider["cache_namespace_identity"],
            "different from observed-v1",
            "Provider V2 may not overwrite observed-v1 evidence",
        ),
        _gate(
            "PROVIDER_V2_IMPLEMENTATION_AND_CONFIG_ISOLATION",
            provider["recipe_version"] != baseline_provider["recipe_version"]
            and provider["implementation_sha256"] != baseline_provider["implementation_sha256"]
            and provider["effective_config_sha256"] != baseline_provider["effective_config_sha256"],
            {
                "recipe_version": provider["recipe_version"],
                "implementation_sha256": provider["implementation_sha256"],
                "effective_config_sha256": provider["effective_config_sha256"],
            },
            "all different from observed-v1",
            "Provider V2 implementation, recipe, and effective config are versioned inputs",
        ),
        _gate(
            "PROVIDER_ASSET_META_AND_INFERENCE_IDENTITY_ISOLATION",
            all(
                _prep_hashes(candidate, field) != _prep_hashes(baseline, field)
                for field in (
                    "asset_identity_sha256",
                    "meta_identity_sha256",
                    "meta_semantic_sha256",
                )
            )
            and _endpoint_hashes(candidate, "inference_identity_sha256")
            != _endpoint_hashes(baseline, "inference_identity_sha256"),
            {
                "asset_identities": list(_prep_hashes(candidate, "asset_identity_sha256")),
                "meta_identities": list(_prep_hashes(candidate, "meta_identity_sha256")),
                "meta_semantics": list(_prep_hashes(candidate, "meta_semantic_sha256")),
                "inference_identities": list(
                    _endpoint_hashes(candidate, "inference_identity_sha256")
                ),
            },
            "all identity projections isolated from observed-v1",
            "new recipe/config metadata must not reuse old durable or inference identities",
        ),
        _gate(
            "SINGLE_PROVIDER_START_AND_LOAD",
            cold["provider_process_start_count"] == 1 and cold["provider_model_load_count"] == 1,
            {
                "process_starts": cold["provider_process_start_count"],
                "model_loads": cold["provider_model_load_count"],
            },
            {"process_starts": 1, "model_loads": 1},
            "the persistent preparation provider must load once for all five segments",
        ),
        _gate(
            "COLD_WALL_INCLUDES_PROVIDER_STARTUP_AND_LOAD",
            bool(cold["startup_load_included_in_wall"]),
            cold["startup_load_included_in_wall"],
            True,
            "cold-wall speedup is valid only when provider startup and load are included",
        ),
        _gate(
            "ASSET_AND_META_VERIFICATION",
            verified,
            [bool(_obj(raw, "preparation")["verified"]) for raw in _prep_rows(candidate)],
            "all five verified",
            "asset and metadata identities are admission evidence",
        ),
        _gate(
            "GPU_TELEMETRY_COMPLETE",
            bool(telemetry["complete"]),
            telemetry["measurement_statuses"],
            ["MEASURED", "MEASURED"],
            "cold preparation and endpoint telemetry are both required for adoption",
        ),
        _gate(
            "PEAK_VRAM_SAFETY",
            telemetry["memory_used_fraction_max"] is not None
            and float(telemetry["memory_used_fraction_max"]) <= MAX_VRAM,
            telemetry["memory_used_fraction_max"],
            {"maximum": MAX_VRAM},
            "peak includes codec preparation and endpoint generation",
        ),
        _gate(
            "TEMPERATURE_SAFETY",
            telemetry["temperature_celsius_max"] is not None
            and float(telemetry["temperature_celsius_max"]) <= MAX_TEMP_C,
            telemetry["temperature_celsius_max"],
            {"maximum_celsius": MAX_TEMP_C},
            "peak includes codec preparation and endpoint generation",
        ),
        _gate(
            "COLD_PREPARATION_SPEEDUP",
            speedup >= (MIN_FULL_SPEEDUP if full else MIN_BOUNDED_SPEEDUP),
            speedup,
            {"minimum": MIN_FULL_SPEEDUP if full else MIN_BOUNDED_SPEEDUP},
            "observed-v1 cold wall divided by Provider V2 cold wall",
        ),
    ]
    if full:
        projection_parity = all(
            float(deltas[kind]["f1"]) == 1.0
            and float(deltas[kind]["exact_semantic_agreement"]) == 1.0
            for kind in KINDS
        )
        gates.extend(
            [
                _gate(
                    "FULL_RESOLUTION_CODEC_ASSET_PARITY",
                    _asset_hashes(baseline) == _asset_hashes(candidate),
                    list(_asset_hashes(candidate)),
                    list(_asset_hashes(baseline)),
                    "max_side=0 must preserve exact codec assets in a new namespace",
                ),
                _gate(
                    "FULL_RESOLUTION_MAGE_OUTPUT_PARITY",
                    _output_hashes(baseline) == _output_hashes(candidate),
                    list(_output_hashes(candidate)),
                    list(_output_hashes(baseline)),
                    "all five Mage output text hashes must match observed-v1",
                ),
                _gate(
                    "FULL_RESOLUTION_PROJECTED_SEMANTIC_PARITY",
                    projection_parity,
                    {
                        kind: {
                            "f1": deltas[kind]["f1"],
                            "exact": deltas[kind]["exact_semantic_agreement"],
                        }
                        for kind in KINDS
                    },
                    "exact QA/event/evidence/track/fusion parity",
                    "full resolution is the Provider V2 behavioral migration control",
                ),
            ]
        )
    else:
        qa, event = deltas["qa"], deltas["event"]
        evidence_delta, track, fusion = deltas["evidence"], deltas["track"], deltas["fusion"]
        gates.extend(
            [
                _gate(
                    "QA_QUALITY",
                    float(qa["f1"]) >= 1.0 and float(qa["exact_semantic_agreement"]) >= 1.0,
                    qa,
                    {"minimum_f1": 1.0, "minimum_exact_agreement": 1.0},
                    "bounded codec must not change projected QA facts",
                ),
                _gate(
                    "EVENT_QUALITY",
                    float(event["f1"]) >= 0.95
                    and float(event["label_agreement"]) >= 0.98
                    and float(event["maximum_boundary_drift_seconds"]) <= MAX_BOUNDARY_DRIFT_S
                    and float(event["maximum_confidence_delta"]) <= MAX_CONFIDENCE_DELTA,
                    event,
                    {
                        "minimum_f1": 0.95,
                        "minimum_label_agreement": 0.98,
                        "maximum_boundary_drift_seconds": MAX_BOUNDARY_DRIFT_S,
                        "maximum_confidence_delta": MAX_CONFIDENCE_DELTA,
                    },
                    "event labels, boundaries, and confidence are compared by match_key",
                ),
                _gate(
                    "EVIDENCE_QUALITY",
                    float(evidence_delta["f1"]) >= 0.95
                    and float(evidence_delta["support_agreement"]) >= 0.98
                    and float(evidence_delta["maximum_confidence_delta"]) <= MAX_CONFIDENCE_DELTA,
                    evidence_delta,
                    {
                        "minimum_f1": 0.95,
                        "minimum_support_agreement": 0.98,
                        "maximum_confidence_delta": MAX_CONFIDENCE_DELTA,
                    },
                    "evidence support direction and confidence must remain stable",
                ),
                _gate(
                    "TRACK_QUALITY",
                    float(track["f1"]) >= 0.95
                    and float(track["label_agreement"]) >= 0.98
                    and float(track["maximum_boundary_drift_seconds"]) <= MAX_BOUNDARY_DRIFT_S
                    and float(track["maximum_confidence_delta"]) <= MAX_CONFIDENCE_DELTA,
                    track,
                    {
                        "minimum_f1": 0.95,
                        "minimum_label_agreement": 0.98,
                        "maximum_boundary_drift_seconds": MAX_BOUNDARY_DRIFT_S,
                        "maximum_confidence_delta": MAX_CONFIDENCE_DELTA,
                    },
                    "temporal reconciliation remains independently qualified",
                ),
                _gate(
                    "FUSION_QUALITY",
                    float(fusion["f1"]) >= 0.95
                    and float(fusion["disposition_agreement"]) >= 1.0
                    and float(fusion["maximum_boundary_drift_seconds"]) <= MAX_BOUNDARY_DRIFT_S
                    and float(fusion["maximum_confidence_delta"]) <= MAX_CONFIDENCE_DELTA,
                    fusion,
                    {
                        "minimum_f1": 0.95,
                        "minimum_disposition_agreement": 1.0,
                        "maximum_boundary_drift_seconds": MAX_BOUNDARY_DRIFT_S,
                        "maximum_confidence_delta": MAX_CONFIDENCE_DELTA,
                    },
                    "bounded compression may not change the fused disposition",
                ),
            ]
        )
    passed = all(bool(item["passed"]) for item in gates)
    return {
        "variant": summary,
        "comparison_to_observed_v1": {
            "cold_preparation_speedup_ratio": speedup,
            "asset_exact_hash_parity": _asset_hashes(baseline) == _asset_hashes(candidate),
            "mage_output_text_hash_parity": _output_hashes(baseline) == _output_hashes(candidate),
            "projected_semantic_deltas": deltas,
        },
        "gates": gates,
        "qualification_status": "PASSED" if passed else "FAILED",
        "locally_adoptable": passed,
    }


def build_qualification_report(
    *, baseline_evidence: Path, provider_v2_evidence: Sequence[Path]
) -> dict[str, object]:
    baseline = _read(baseline_evidence)
    _validate_evidence(baseline, "baseline")
    baseline_provider = _obj(baseline["provider"], "baseline.provider")
    if baseline_provider["generation"] != BASELINE or baseline_provider["max_side"] != 0:
        raise MageDcvcQualificationInputError("baseline must be observed-v1 with max_side=0")

    candidates = [_read(path) for path in provider_v2_evidence]
    if not candidates:
        raise MageDcvcQualificationInputError("Provider V2 evidence is required")
    for index, candidate in enumerate(candidates):
        _validate_evidence(candidate, f"provider_v2[{index}]")
        if _obj(candidate["provider"], "provider")["generation"] != PROVIDER_V2:
            raise MageDcvcQualificationInputError("all candidates must be Provider V2")
    candidates.sort(
        key=lambda item: (
            int(_obj(item["provider"], "provider")["max_side"]),
            str(item["variant_id"]),
        )
    )
    max_sides = [int(_obj(item["provider"], "provider")["max_side"]) for item in candidates]
    if max_sides.count(0) != 1:
        raise MageDcvcQualificationInputError("exactly one Provider V2 max_side=0 run is required")
    if not any(value > 0 for value in max_sides):
        raise MageDcvcQualificationInputError("at least one bounded max_side run is required")
    if len(set(max_sides)) != len(max_sides):
        raise MageDcvcQualificationInputError("Provider V2 max_side values must be unique")

    variant_ids = [str(baseline["variant_id"]), *(str(item["variant_id"]) for item in candidates)]
    if len(set(variant_ids)) != len(variant_ids):
        raise MageDcvcQualificationInputError("variant_id values must be unique")
    namespaces = [
        str(_obj(item["provider"], "provider")["cache_namespace_identity"]) for item in candidates
    ]
    configs = [
        str(_obj(item["provider"], "provider")["effective_config_sha256"]) for item in candidates
    ]
    if len(set(namespaces)) != len(namespaces):
        raise MageDcvcQualificationInputError(
            "each Provider V2 config needs a distinct cache namespace"
        )
    if len(set(configs)) != len(configs):
        raise MageDcvcQualificationInputError(
            "each max_side needs a distinct effective config identity"
        )

    comparisons = [_candidate_comparison(baseline, candidate) for candidate in candidates]
    full = next(
        item
        for item in comparisons
        if int(_obj(_obj(item["variant"], "variant")["provider"], "provider")["max_side"]) == 0
    )
    bounded = [
        item
        for item in comparisons
        if int(_obj(_obj(item["variant"], "variant")["provider"], "provider")["max_side"]) > 0
    ]
    passing = (
        [item for item in bounded if item["locally_adoptable"]] if full["locally_adoptable"] else []
    )
    if passing:
        recommended = min(
            passing,
            key=lambda item: float(
                _obj(_obj(item["variant"], "variant")["cold_preparation"], "cold")["wall_seconds"]
            ),
        )
        recommendation = "ADOPT_FASTEST_QUALITY_QUALIFIED_BOUNDED_PROVIDER_V2_LOCALLY"
    elif full["locally_adoptable"]:
        recommended = full
        recommendation = "ADOPT_PROVIDER_V2_FULL_RESOLUTION_LOCALLY"
    else:
        recommended = None
        recommendation = "KEEP_OBSERVED_V1_BASELINE"

    payload: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "authority": "LOCAL_QUALIFICATION_NON_PRODUCTION",
        "production_eligible": False,
        "production_prohibition": (
            "LOCAL_SINGLE_DEVICE_EVIDENCE_DOES_NOT_QUALIFY_DUAL_H100_OR_PRODUCTION_ADOPTION"
        ),
        "qualification_scope": {
            "media_duration_seconds": 40.0,
            "segment_count": SEGMENT_COUNT,
            "camera_count": 1,
            "worker_count": 1,
            "generation_concurrency": 1,
            "real_gpu_benchmark_executed_by_this_tool": False,
            "recurrent_work_semantics": RECURRENT_WORK,
            "sequence_length_is_compute_cap": False,
            "note": (
                "num_sampled_frames and seq_len_frames do not limit the recurrent readiness "
                "chain; work continues through the last sampled frame"
            ),
        },
        "thresholds": {
            "minimum_full_resolution_cold_speedup_ratio": MIN_FULL_SPEEDUP,
            "minimum_bounded_cold_speedup_ratio": MIN_BOUNDED_SPEEDUP,
            "maximum_peak_vram_fraction": MAX_VRAM,
            "maximum_temperature_celsius": MAX_TEMP_C,
            "minimum_qa_f1": 1.0,
            "minimum_qa_exact_agreement": 1.0,
            "minimum_event_f1": 0.95,
            "minimum_event_label_agreement": 0.98,
            "minimum_evidence_f1": 0.95,
            "minimum_evidence_support_agreement": 0.98,
            "minimum_track_f1": 0.95,
            "minimum_track_label_agreement": 0.98,
            "minimum_fusion_f1": 0.95,
            "minimum_fusion_disposition_agreement": 1.0,
            "maximum_boundary_drift_seconds": MAX_BOUNDARY_DRIFT_S,
            "maximum_confidence_delta": MAX_CONFIDENCE_DELTA,
        },
        "observed_v1_baseline": _variant_summary(baseline),
        "provider_v2_full_resolution": full,
        "provider_v2_bounded_candidates": bounded,
        "qualification_status": "PASSED" if full["locally_adoptable"] else "FAILED",
        "recommendation": recommendation,
        "recommended_variant_id": (
            None if recommended is None else _obj(recommended["variant"], "variant")["variant_id"]
        ),
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


RETAINED_REPORT_VERSION = "mage-dcvc-provider-v2-local-artifact-ab-report-v1"
RETAINED_EVIDENCE_VERSION = "mage-dcvc-provider-v2-local-artifact-evidence-v1"
VISUAL_ASSET_EXCLUSIONS = frozenset({"meta.json"})
DOWNSTREAM_KINDS = {
    "qa": ("qa", "qa_projection_semantic_sha256"),
    "event": ("event", "event_projection_semantic_sha256"),
    "evidence": ("evidence", "evidence_projection_semantic_sha256"),
    "track": ("event-track", "revision_semantic_sha256"),
    "fusion": ("fusion-decision", "fusion_semantic_sha256"),
}


def _retained_file_reference(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise MageDcvcQualificationInputError(
            f"retained artifact is missing or unreadable: {resolved}"
        ) from error
    return {
        "path": str(resolved),
        "byte_count": len(payload),
        "exact_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _retained_jsonl(path: Path) -> list[dict[str, Any]]:
    reference = path.expanduser().resolve()
    try:
        lines = reference.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MageDcvcQualificationInputError(
            f"retained JSONL is missing or unreadable: {reference}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise MageDcvcQualificationInputError(
                f"invalid retained JSONL at {reference}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise MageDcvcQualificationInputError(
                f"retained JSONL row must be an object: {reference}:{line_number}"
            )
        rows.append(value)
    return rows


def _retained_normalized_path(value: object, label: str) -> str:
    return str(Path(_text(value, label)).expanduser()).replace("/", "\\").casefold()


def _retained_segment_from_source(value: object, label: str) -> tuple[int, str]:
    name = Path(_text(value, label)).stem
    parts = name.split("-", maxsplit=2)
    if len(parts) < 2 or not parts[0].isdigit():
        raise MageDcvcQualificationInputError(
            f"{label} must begin with a zero-padded ordinal and segment hash"
        )
    ordinal = int(parts[0])
    segment_sha256 = _sha(parts[1], f"{label}.segment_sha256")
    return ordinal, segment_sha256


def _retained_gpu_telemetry(path: Path) -> dict[str, object]:
    document = _read(path)
    status = _text(document.get("measurement_status"), f"{path}.measurement_status")
    raw_summary = document.get("summary", [])
    if not isinstance(raw_summary, list):
        raise MageDcvcQualificationInputError(f"{path}.summary must be an array")
    devices: list[dict[str, object]] = []
    for index, raw in enumerate(raw_summary):
        device = _obj(raw, f"{path}.summary[{index}]")
        devices.append(
            {
                "gpu_index": _integer(device.get("gpu_index"), "gpu_index"),
                "gpu_name": _text(device.get("gpu_name"), "gpu_name"),
                "sample_count": _integer(device.get("sample_count"), "sample_count"),
                "utilization_gpu_percent_mean": _number(
                    device.get("utilization_gpu_percent_mean"), "utilization mean"
                ),
                "utilization_gpu_percent_max": _number(
                    device.get("utilization_gpu_percent_max"), "utilization max"
                ),
                "memory_used_mib_max": _number(
                    device.get("memory_used_mib_max"), "memory used max"
                ),
                "memory_total_mib": _number(device.get("memory_total_mib"), "memory total", 1e-12),
                "memory_used_fraction_max": _number(
                    device.get("memory_used_fraction_max"), "memory fraction max"
                ),
                "temperature_celsius_max": _number(
                    device.get("temperature_celsius_max"), "temperature max"
                ),
                "power_draw_watts_max": _number(device.get("power_draw_watts_max"), "power max"),
            }
        )
    errors = document.get("errors", [])
    if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
        raise MageDcvcQualificationInputError(f"{path}.errors must be an array of strings")
    return {
        "measurement_status": status,
        "devices": devices,
        "errors": list(errors),
        "source_artifact": _retained_file_reference(path),
    }


def _retained_assets(raw: object, label: str) -> list[dict[str, object]]:
    assets: list[dict[str, object]] = []
    for index, value in enumerate(_arr(raw, label)):
        asset = _obj(value, f"{label}[{index}]")
        assets.append(
            {
                "relative_path": _text(asset.get("relative_path"), "asset.relative_path"),
                "byte_count": _integer(asset.get("byte_count"), "asset.byte_count", 1),
                "sha256": _sha(asset.get("sha256"), "asset.sha256"),
            }
        )
    assets.sort(key=lambda item: str(item["relative_path"]))
    if len({item["relative_path"] for item in assets}) != len(assets):
        raise MageDcvcQualificationInputError(f"{label} has duplicate asset paths")
    return assets


def _retained_v1_sidecars(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cache_root = Path(_text(manifest.get("qualified_cache_root"), "qualified_cache_root"))
    sidecar_root = cache_root / ".robata-entries"
    sidecars: dict[str, dict[str, Any]] = {}
    for path in sorted(sidecar_root.glob("*.json")):
        document = _read(path)
        key = _retained_normalized_path(document.get("source_path"), "sidecar.source_path")
        if key in sidecars:
            raise MageDcvcQualificationInputError("duplicate observed-v1 source sidecar")
        document["_source_artifact"] = _retained_file_reference(path)
        sidecars[key] = document
    return sidecars


def _retained_preparation_evidence(root: Path, generation: str) -> dict[str, object]:
    base = root.expanduser().resolve()
    observed = generation == BASELINE
    manifest_path = base / ("cache-manifest-v1.json" if observed else "cache-manifest-v2.json")
    manifest = _read(manifest_path)
    expected_version = (
        "mage-codec-cache-manifest-v1" if observed else "mage-codec-cache-manifest-v2"
    )
    if manifest.get("manifest_version") != expected_version:
        raise MageDcvcQualificationInputError(f"{manifest_path} must use {expected_version}")
    if manifest.get("entry_count") != SEGMENT_COUNT or manifest.get("built_count") != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError(
            f"{manifest_path} must retain five successfully built entries"
        )
    raw_entries = list(_arr(manifest.get("entries"), f"{manifest_path}.entries"))
    if len(raw_entries) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError(f"{manifest_path} must retain five entries")

    v1_sidecars = _retained_v1_sidecars(manifest) if observed else {}
    prewarm_report: dict[str, Any] | None = None
    jobs_by_source: dict[str, Mapping[str, Any]] = {}
    source_references = [_retained_file_reference(manifest_path)]
    if not observed:
        report_path = base / "prewarm-report-v2.json"
        prewarm_report = _read(report_path)
        if prewarm_report.get("report_version") != "mage-dcvc-prewarm-report-v2":
            raise MageDcvcQualificationInputError(
                f"{report_path} must use mage-dcvc-prewarm-report-v2"
            )
        source_references.append(_retained_file_reference(report_path))
        for index, raw in enumerate(_arr(prewarm_report.get("jobs"), "prewarm jobs")):
            job = _obj(raw, f"prewarm jobs[{index}]")
            key = _retained_normalized_path(job.get("source_path"), "job.source_path")
            if key in jobs_by_source:
                raise MageDcvcQualificationInputError("duplicate Provider V2 prewarm job")
            jobs_by_source[key] = job
        if len(jobs_by_source) != SEGMENT_COUNT:
            raise MageDcvcQualificationInputError("Provider V2 needs five retained prewarm jobs")

    prepared: list[dict[str, object]] = []
    for raw in raw_entries:
        entry = _obj(raw, "cache manifest entry")
        source_path = _text(entry.get("source_path"), "entry.source_path")
        source_key = _retained_normalized_path(source_path, "entry.source_path")
        ordinal, segment_sha256 = _retained_segment_from_source(source_path, "entry.source_path")
        if entry.get("admission") != "BUILT":
            raise MageDcvcQualificationInputError("all retained preparation entries must be BUILT")

        if observed:
            sidecar = v1_sidecars.get(source_key)
            if sidecar is None:
                raise MageDcvcQualificationInputError(
                    f"observed-v1 sidecar not found for {source_path}"
                )
            preparation_seconds: float | None = None
            timing_status = "NOT_RECORDED"
            timing_note = (
                "Observed-v1 retained only the aggregate prewarm wall. Per-segment time is "
                "not derived by dividing the aggregate."
            )
            source_artifact = _obj(sidecar.pop("_source_artifact"), "source artifact")
        else:
            directory = Path(
                _text(entry.get("provider_cache_directory"), "provider_cache_directory")
            )
            sidecar_path = directory / ".robata-dcvc-preparation-v2.json"
            sidecar = _read(sidecar_path)
            source_references.append(_retained_file_reference(sidecar_path))
            source_artifact = source_references[-1]
            job = jobs_by_source.get(source_key)
            if job is None:
                raise MageDcvcQualificationInputError(
                    f"Provider V2 prewarm job not found for {source_path}"
                )
            preparation_seconds = _number(
                entry.get("preparation_wall_seconds"), "entry.preparation_wall_seconds", 1e-12
            )
            job_seconds = _number(job.get("response_wall_seconds"), "job.response_wall_seconds")
            if not math.isclose(preparation_seconds, job_seconds, rel_tol=0.0, abs_tol=1e-9):
                raise MageDcvcQualificationInputError(
                    f"Provider V2 timing mismatch for segment {ordinal}"
                )
            timing_status = "MEASURED"
            timing_note = "Retained Provider V2 worker response wall for this segment."

        assets = _retained_assets(sidecar.get("assets"), "preparation assets")
        meta = [item for item in assets if item["relative_path"] == "meta.json"]
        if len(meta) != 1:
            raise MageDcvcQualificationInputError("each preparation entry must retain meta.json")
        visual_assets = [
            item for item in assets if item["relative_path"] not in VISUAL_ASSET_EXCLUSIONS
        ]
        if not visual_assets:
            raise MageDcvcQualificationInputError("preparation entry has no visual payload assets")
        source_content_sha256 = _sha(sidecar.get("source_content_sha256"), "source_content_sha256")
        source_byte_count = _integer(sidecar.get("source_byte_count"), "source_byte_count", 1)
        prepared.append(
            {
                "ordinal": ordinal,
                "segment_semantic_sha256": segment_sha256,
                "source_path": source_path,
                "source_content_sha256": source_content_sha256,
                "source_byte_count": source_byte_count,
                "preparation_seconds": preparation_seconds,
                "timing_status": timing_status,
                "timing_note": timing_note,
                "visual_payload_assets": visual_assets,
                "provider_metadata_asset": meta[0],
                "all_assets": assets,
                "sidecar_source_artifact": dict(source_artifact),
            }
        )
    prepared.sort(key=lambda item: int(item["ordinal"]))
    if [item["ordinal"] for item in prepared] != list(range(SEGMENT_COUNT)):
        raise MageDcvcQualificationInputError("preparation entries must cover ordinals 0..4")

    recipe = _obj(manifest.get("recipe", {}), "observed recipe") if observed else {}
    effective_projection = (
        _obj(recipe.get("effective_projection", {}), "observed effective projection")
        if observed
        else {}
    )
    provider_process_start_count: int | None
    provider_model_load_count: int | None
    provider_model_load_seconds: float | None
    startup_load_included_in_wall: bool | None
    if observed:
        provider_process_start_count = None
        provider_model_load_count = None
        provider_model_load_seconds = None
        startup_load_included_in_wall = None
        measurement_note = (
            "Observed-v1 does not authoritatively retain provider process starts, model load "
            "count, model load seconds, or per-segment preparation timings. Those fields remain "
            "null; only the aggregate wall is measured."
        )
        recipe_version = _text(recipe.get("recipe_version"), "recipe.recipe_version")
        max_side = _integer(effective_projection.get("max_side"), "recipe.max_side")
        provider_version: str | None = None
        effective_config_sha256: str | None = None
        implementation_sha256: str | None = None
        recipe_semantic_sha256: str | None = _sha(
            recipe.get("semantic_sha256"), "recipe.semantic_sha256"
        )
    else:
        assert prewarm_report is not None
        worker = _obj(prewarm_report.get("worker_process"), "worker_process")
        provider_process_start_count = _integer(
            worker.get("process_start_count"), "worker_process.process_start_count", 1
        )
        provider_model_load_count = _integer(
            prewarm_report.get("inferred_process_model_load_count"),
            "inferred_process_model_load_count",
            1,
        )
        provider_model_load_seconds = _number(
            prewarm_report.get("inferred_process_model_load_seconds"),
            "inferred_process_model_load_seconds",
        )
        startup_load_included_in_wall = True
        measurement_note = (
            "Provider V2 aggregate wall is from the cache manifest and includes the persistent "
            "worker lifecycle measured by the companion prewarm report."
        )
        recipe_version = _text(manifest.get("recipe_version"), "recipe_version")
        effective = _obj(manifest.get("effective_config"), "effective_config")
        max_side = _integer(effective.get("max_side"), "effective_config.max_side")
        provider_version = _text(manifest.get("provider_version"), "provider_version")
        effective_config_sha256 = _sha(
            prewarm_report.get("effective_config_sha256"), "effective_config_sha256"
        )
        implementation_sha256 = _sha(
            manifest.get("provider_implementation_sha256"),
            "provider_implementation_sha256",
        )
        recipe_semantic_sha256 = None

    telemetry_path = base / "gpu-telemetry.json"
    source_references.append(_retained_file_reference(telemetry_path))
    payload: dict[str, object] = {
        "provider_generation": generation,
        "manifest_version": expected_version,
        "recipe_version": recipe_version,
        "provider_version": provider_version,
        "namespace_identity": _sha(manifest.get("namespace_identity"), "namespace_identity"),
        "checkpoint_manifest_sha256": _sha(
            manifest.get("checkpoint_manifest_sha256"), "checkpoint_manifest_sha256"
        ),
        "codec_policy_sha256": _sha(manifest.get("codec_policy_sha256"), "codec_policy_sha256"),
        "max_side": max_side,
        "effective_config_sha256": effective_config_sha256,
        "provider_implementation_sha256": implementation_sha256,
        "recipe_semantic_sha256": recipe_semantic_sha256,
        "wall_seconds": _number(
            manifest.get("prewarm_wall_seconds"), "prewarm_wall_seconds", 1e-12
        ),
        "wall_measurement_source": "cache_manifest.prewarm_wall_seconds",
        "provider_process_start_count": provider_process_start_count,
        "provider_model_load_count": provider_model_load_count,
        "provider_model_load_seconds": provider_model_load_seconds,
        "startup_load_included_in_wall": startup_load_included_in_wall,
        "per_segment_timing_status": "NOT_RECORDED" if observed else "MEASURED",
        "measurement_note": measurement_note,
        "per_segment": prepared,
        "gpu_telemetry": _retained_gpu_telemetry(telemetry_path),
        "source_artifacts": source_references,
    }
    payload["evidence_semantic_sha256"] = semantic_sha256(payload)
    return payload


def _retained_projection(kind: str, document: Mapping[str, Any]) -> object:
    if kind == "qa":
        return {"camera_facts": document.get("camera_facts")}
    if kind == "event":
        return {
            "hypotheses": [
                {
                    "action": item.get("action"),
                    "actor": item.get("actor"),
                    "object": item.get("object"),
                    "interval": item.get("interval"),
                    "model_reported_confidence": item.get("model_reported_confidence"),
                    "start_confidence": item.get("start_confidence"),
                    "end_confidence": item.get("end_confidence"),
                    "started_before_context": item.get("started_before_context"),
                    "continues_after_context": item.get("continues_after_context"),
                }
                for item in (
                    _obj(raw, "event hypothesis")
                    for raw in _arr(document.get("hypotheses"), "event hypotheses")
                )
            ]
        }
    if kind == "evidence":
        projected_events: list[list[dict[str, object]]] = []
        for raw_event in _arr(document.get("event_evidence"), "event evidence"):
            event = _obj(raw_event, "event evidence item")
            cameras = _obj(event.get("cameras"), "event evidence cameras")
            projected_cameras: list[dict[str, object]] = []
            for camera_id in sorted(cameras):
                camera = _obj(cameras[camera_id], "camera evidence")
                projected_cameras.append(
                    {
                        "camera_id": camera_id,
                        "observed_interval": camera.get("observed_interval"),
                        "relation": camera.get("relation"),
                        "selected_for_inference": camera.get("selected_for_inference"),
                        "visibility": camera.get("visibility"),
                    }
                )
            projected_events.append(projected_cameras)
        projected_events.sort(key=canonical_json_bytes)
        return {"event_evidence": projected_events}
    if kind == "track":
        return {
            "action": document.get("action"),
            "actor": document.get("actor"),
            "object": document.get("object"),
            "interval": document.get("interval"),
            "state": document.get("state"),
            "model_reported_confidence_values": document.get("model_reported_confidence_values"),
            "start_confidence": document.get("start_confidence"),
            "end_confidence": document.get("end_confidence"),
            "continues_after_context": document.get("continues_after_context"),
        }
    if kind == "fusion":
        assessments: list[dict[str, object]] = []
        for raw in _arr(document.get("camera_assessments"), "camera assessments"):
            camera = _obj(raw, "camera assessment")
            assessments.append(
                {
                    "camera_id": camera.get("camera_id"),
                    "observable": camera.get("observable"),
                    "selected": camera.get("selected"),
                    "supporting": camera.get("supporting"),
                    "contradicting": camera.get("contradicting"),
                    "mean_reliability": camera.get("mean_reliability"),
                    "mean_support_value": camera.get("mean_support_value"),
                }
            )
        assessments.sort(key=lambda item: str(item["camera_id"]))
        return {
            "action": document.get("action"),
            "interval": document.get("interval"),
            "confidence": document.get("confidence"),
            "ambiguity_reasons": document.get("ambiguity_reasons"),
            "refine_reasons": document.get("refine_reasons"),
            "observable_camera_count": document.get("observable_camera_count"),
            "selected_camera_count": document.get("selected_camera_count"),
            "supporting_camera_count": document.get("supporting_camera_count"),
            "contradicting_camera_count": document.get("contradicting_camera_count"),
            "camera_assessments": assessments,
        }
    raise MageDcvcQualificationInputError(f"unsupported downstream kind: {kind}")


def _retained_downstream(root: Path) -> dict[str, object]:
    artifact_root = root / "stream-artifacts"
    result: dict[str, object] = {}
    for kind, (directory, durable_field) in DOWNSTREAM_KINDS.items():
        paths = sorted((artifact_root / directory).rglob("*.json"))
        if not paths:
            raise MageDcvcQualificationInputError(
                f"retained generation has no {directory} artifacts: {artifact_root}"
            )
        content_hashes: list[str] = []
        durable_hashes: list[str] = []
        for path in paths:
            document = _read(path)
            content_hashes.append(semantic_sha256(_retained_projection(kind, document)))
            durable_hashes.append(_sha(document.get(durable_field), f"{kind}.{durable_field}"))
        result[kind] = {
            "artifact_count": len(paths),
            "content_projection_semantic_sha256_values": sorted(content_hashes),
            "durable_semantic_sha256_values": sorted(durable_hashes),
        }
    return result


def _retained_generation_evidence(root: Path) -> dict[str, object]:
    base = root.expanduser().resolve()
    harness_path = base / "harness-result.json"
    stream_path = base / "stream-report.json"
    telemetry_path = base / "endpoint-generation-telemetry.jsonl"
    full_gpu_path = base / "full-wall-gpu-telemetry.json"
    stream_gpu_path = base / "stream-gpu-telemetry.json"
    harness = _read(harness_path)
    stream = _read(stream_path)
    telemetry_rows = _retained_jsonl(telemetry_path)
    if harness.get("status") != "SUCCEEDED" or stream.get("ok") is not True:
        raise MageDcvcQualificationInputError(f"retained generation did not succeed: {base}")
    if len(telemetry_rows) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError(
            f"retained generation needs five telemetry rows: {base}"
        )

    plan = _obj(stream.get("plan"), "stream.plan")
    recording = _obj(plan.get("recording"), "stream.plan.recording")
    recording_interval = _obj(recording.get("interval"), "recording.interval")
    if recording_interval.get("start_ns") != 0 or recording_interval.get("end_ns") != DURATION_NS:
        raise MageDcvcQualificationInputError("retained generation must cover exactly 40 seconds")
    raw_segments = _arr(plan.get("storage_segments"), "storage_segments")
    if len(raw_segments) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError("retained generation needs five storage segments")
    segments: list[dict[str, object]] = []
    expected_start = 0
    for ordinal, raw in enumerate(raw_segments):
        segment = _obj(raw, "storage segment")
        interval = _obj(segment.get("interval"), "storage segment interval")
        start_ns = _integer(interval.get("start_ns"), "storage segment start")
        end_ns = _integer(interval.get("end_ns"), "storage segment end", 1)
        if segment.get("ordinal") != ordinal or start_ns != expected_start or end_ns <= start_ns:
            raise MageDcvcQualificationInputError("storage segments must be ordered and contiguous")
        expected_start = end_ns
        segments.append(
            {
                "ordinal": ordinal,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "segment_semantic_sha256": _sha(
                    segment.get("segment_semantic_sha256"), "segment_semantic_sha256"
                ),
            }
        )
    if expected_start != DURATION_NS:
        raise MageDcvcQualificationInputError("storage segments must cover exactly 40 seconds")

    execution = _obj(stream.get("execution"), "stream.execution")
    contexts = _arr(execution.get("contexts"), "execution.contexts")
    if len(contexts) != SEGMENT_COUNT:
        raise MageDcvcQualificationInputError("retained generation needs five contexts")
    rows_by_context: dict[str, Mapping[str, Any]] = {}
    for row in telemetry_rows:
        context_id = _text(row.get("context_id"), "telemetry.context_id")
        if context_id in rows_by_context:
            raise MageDcvcQualificationInputError("duplicate generation telemetry context")
        rows_by_context[context_id] = row

    generated: list[dict[str, object]] = []
    result_references: list[dict[str, object]] = []
    for ordinal, raw_context in enumerate(contexts):
        context = _obj(raw_context, "execution context")
        context_id = _text(context.get("context_manifest_key"), "context_manifest_key")
        row = rows_by_context.get(context_id)
        if row is None:
            raise MageDcvcQualificationInputError(
                f"no endpoint telemetry for context ordinal {ordinal}"
            )
        artifact_identity = _sha(row.get("result_artifact_identity"), "result_artifact_identity")
        result_path = base / "endpoint-results" / f"{artifact_identity}.json"
        result_document = _read(result_path)
        reference = _retained_file_reference(result_path)
        if reference["exact_sha256"] != _sha(
            row.get("result_artifact_exact_sha256"), "result_artifact_exact_sha256"
        ):
            raise MageDcvcQualificationInputError(
                f"endpoint result exact hash mismatch for segment {ordinal}"
            )
        result_references.append(reference)
        output_text = _text(result_document.get("output_text"), "endpoint output_text")
        try:
            normalized_output = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise MageDcvcQualificationInputError(
                f"endpoint output is not JSON for segment {ordinal}"
            ) from error
        if not isinstance(normalized_output, dict):
            raise MageDcvcQualificationInputError(
                f"endpoint output must be a JSON object for segment {ordinal}"
            )
        runtime = _obj(row.get("runtime_telemetry"), "runtime_telemetry")
        if context.get("focus_segment_ordinal") != ordinal:
            raise MageDcvcQualificationInputError("context ordinals are not ordered")
        generated.append(
            {
                "ordinal": ordinal,
                "segment_semantic_sha256": segments[ordinal]["segment_semantic_sha256"],
                "context_manifest_key": context_id,
                "inference_identity_sha256": _sha(
                    row.get("inference_identity_sha256"), "inference_identity_sha256"
                ),
                "input_manifest_sha256": _sha(
                    row.get("input_manifest_sha256"), "input_manifest_sha256"
                ),
                "result_artifact_identity_sha256": artifact_identity,
                "result_artifact_exact_sha256": reference["exact_sha256"],
                "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
                "normalized_output_semantic_sha256": semantic_sha256(normalized_output),
                "prompt_tokens": _integer(row.get("prompt_tokens"), "prompt_tokens", 1),
                "output_tokens": _integer(row.get("output_tokens"), "output_tokens"),
                "output_budget_exhausted": bool(row.get("output_budget_exhausted")),
                "processor_seconds": _number(runtime.get("processor_seconds"), "processor_seconds"),
                "input_materialization_seconds": _number(
                    runtime.get("input_materialization_seconds"),
                    "input_materialization_seconds",
                ),
                "generate_seconds": _number(runtime.get("generate_seconds"), "generate_seconds"),
                "decode_seconds": _number(runtime.get("decode_seconds"), "decode_seconds"),
                "total_request_seconds": _number(
                    runtime.get("total_request_seconds"), "total_request_seconds"
                ),
                "time_to_first_token_seconds": _number(
                    runtime.get("time_to_first_token_seconds"), "time_to_first_token_seconds"
                ),
                "output_tokens_per_second": _number(
                    runtime.get("output_tokens_per_second"), "output_tokens_per_second"
                ),
                "observation_seconds": _number(
                    context.get("observation_seconds"), "observation_seconds"
                ),
                "endpoint_result_source_artifact": reference,
            }
        )

    first = telemetry_rows[0]
    for row in telemetry_rows[1:]:
        for field in (
            "codec_policy_sha256",
            "model_identity_sha256",
            "decoder_id",
            "max_new_tokens",
        ):
            if row.get(field) != first.get(field):
                raise MageDcvcQualificationInputError(
                    f"generation control {field} changed within one run"
                )
    endpoint = _obj(execution.get("endpoint"), "execution.endpoint")
    model_identity = _obj(endpoint.get("model_identity"), "endpoint.model_identity")
    measurement = _obj(harness.get("measurement"), "harness.measurement")
    execution_timing = _obj(execution.get("execution_timing"), "execution_timing")
    single_route = _obj(stream.get("single_route"), "single_route")
    source_references = [
        _retained_file_reference(harness_path),
        _retained_file_reference(stream_path),
        _retained_file_reference(telemetry_path),
        _retained_file_reference(full_gpu_path),
        _retained_file_reference(stream_gpu_path),
        *result_references,
    ]
    payload: dict[str, object] = {
        "status": "SUCCEEDED",
        "sample": {
            "recording_key": _text(recording.get("recording_key"), "recording_key"),
            "recording_exact_sha256": _sha(
                recording.get("recording_exact_sha256"), "recording_exact_sha256"
            ),
            "source_path": _text(stream.get("source_path"), "source_path"),
            "source_byte_count": _integer(stream.get("source_byte_count"), "source_byte_count", 1),
            "media_duration_ns": DURATION_NS,
            "plan_semantic_sha256": _sha(plan.get("plan_semantic_sha256"), "plan_semantic_sha256"),
            "segments": segments,
        },
        "controls": {
            "checkpoint_manifest_sha256": _sha(
                model_identity.get("checkpoint_manifest_sha256"),
                "checkpoint_manifest_sha256",
            ),
            "model_identity_sha256": _sha(
                first.get("model_identity_sha256"), "model_identity_sha256"
            ),
            "decoder_id": _text(first.get("decoder_id"), "decoder_id"),
            "decoder_identity_sha256_values": [
                _sha(row.get("decoder_identity_sha256"), "decoder_identity_sha256")
                for row in telemetry_rows
            ],
            "max_new_tokens": _integer(first.get("max_new_tokens"), "max_new_tokens", 1),
            "prompt_token_counts": [item["prompt_tokens"] for item in generated],
            "selected_camera": _text(stream.get("selected_camera"), "selected_camera"),
            "worker_count": _integer(single_route.get("worker_count"), "worker_count", 1),
            "generation_concurrency": _integer(
                single_route.get("generation_concurrency"), "generation_concurrency", 1
            ),
            "max_inflight_observations": _integer(
                single_route.get("max_inflight_observations"),
                "max_inflight_observations",
                1,
            ),
            "execution_profile": _text(execution.get("execution_profile"), "execution_profile"),
        },
        "codec_policy_sha256": _sha(first.get("codec_policy_sha256"), "codec_policy_sha256"),
        "measurement": {
            "overall_wall_seconds": _number(
                measurement.get("overall_wall_seconds"), "overall_wall_seconds", 1e-12
            ),
            "stream_command_wall_seconds": _number(
                measurement.get("stream_wall_seconds"), "stream_wall_seconds", 1e-12
            ),
            "stream_run_wall_seconds": _number(
                execution_timing.get("run_wall_seconds"), "run_wall_seconds", 1e-12
            ),
            "endpoint_startup_to_health_seconds": _number(
                measurement.get("endpoint_startup_to_health_seconds"),
                "endpoint_startup_to_health_seconds",
            ),
            "model_load_seconds": _number(first.get("model_load_seconds"), "model_load_seconds"),
            "model_load_included_in_overall_wall": measurement.get(
                "model_load_included_in_overall_wall"
            ),
            "codec_preparation_included_in_overall_wall": measurement.get(
                "codec_preparation_included_in_overall_wall"
            ),
            "per_segment": generated,
        },
        "downstream": _retained_downstream(base),
        "full_wall_gpu_telemetry": _retained_gpu_telemetry(full_gpu_path),
        "stream_gpu_telemetry": _retained_gpu_telemetry(stream_gpu_path),
        "source_artifacts": source_references,
    }
    if payload["measurement"]["codec_preparation_included_in_overall_wall"] is not False:
        raise MageDcvcQualificationInputError(
            "generation overall wall must explicitly exclude codec preparation"
        )
    payload["evidence_semantic_sha256"] = semantic_sha256(payload)
    return payload


def _retained_build_variant(
    *, preparation_root: Path, generation_root: Path, generation: str
) -> dict[str, object]:
    preparation = _retained_preparation_evidence(preparation_root, generation)
    generation_evidence = _retained_generation_evidence(generation_root)
    sample = _obj(generation_evidence["sample"], "generation.sample")
    plan_segments = _arr(sample.get("segments"), "generation.sample.segments")
    prep_segments = _arr(preparation.get("per_segment"), "preparation.per_segment")
    plan_identities = [
        _obj(item, "plan segment").get("segment_semantic_sha256") for item in plan_segments
    ]
    prep_identities = [
        _obj(item, "preparation segment").get("segment_semantic_sha256") for item in prep_segments
    ]
    if prep_identities != plan_identities:
        raise MageDcvcQualificationInputError(
            "preparation segment plan does not match the generation request"
        )
    controls = _obj(generation_evidence["controls"], "generation.controls")
    if preparation["checkpoint_manifest_sha256"] != controls["checkpoint_manifest_sha256"]:
        raise MageDcvcQualificationInputError(
            "preparation checkpoint does not match generation model checkpoint"
        )
    if preparation["codec_policy_sha256"] != generation_evidence["codec_policy_sha256"]:
        raise MageDcvcQualificationInputError(
            "preparation codec policy does not match generation requests"
        )
    max_side = _integer(preparation["max_side"], "preparation.max_side")
    variant_id = (
        "observed-v1-max-side-0" if generation == BASELINE else f"provider-v2-max-side-{max_side}"
    )
    payload: dict[str, object] = {
        "evidence_version": RETAINED_EVIDENCE_VERSION,
        "evidence_class": "RETAINED_LOCAL_MEASUREMENT",
        "production_eligible": False,
        "variant_id": variant_id,
        "provider_generation": generation,
        "max_side": max_side,
        "preparation": preparation,
        "generation": generation_evidence,
    }
    payload["evidence_semantic_sha256"] = semantic_sha256(payload)
    return payload


def _retained_variant_projection(variant: Mapping[str, Any], field: str) -> tuple[object, ...]:
    preparation = _obj(variant.get("preparation"), "variant.preparation")
    segments = _arr(preparation.get("per_segment"), "preparation.per_segment")
    if field == "visual":
        return tuple(
            _obj(item, "preparation segment").get("visual_payload_assets") for item in segments
        )
    if field == "metadata":
        return tuple(
            _obj(item, "preparation segment").get("provider_metadata_asset") for item in segments
        )
    generation = _obj(variant.get("generation"), "variant.generation")
    measurement = _obj(generation.get("measurement"), "generation.measurement")
    generated = _arr(measurement.get("per_segment"), "generation.per_segment")
    if field == "output_text":
        return tuple(
            _obj(item, "generated segment").get("output_text_sha256") for item in generated
        )
    if field == "normalized_output":
        return tuple(
            _obj(item, "generated segment").get("normalized_output_semantic_sha256")
            for item in generated
        )
    if field == "inference_identity":
        return tuple(
            _obj(item, "generated segment").get("inference_identity_sha256") for item in generated
        )
    raise MageDcvcQualificationInputError(f"unsupported retained projection: {field}")


def _retained_sample_projection(variant: Mapping[str, Any]) -> object:
    generation = _obj(variant.get("generation"), "variant.generation")
    return generation.get("sample")


def _retained_control_projection(variant: Mapping[str, Any]) -> object:
    generation = _obj(variant.get("generation"), "variant.generation")
    return generation.get("controls")


def _retained_downstream_projection(
    variant: Mapping[str, Any], field: str
) -> dict[str, tuple[object, ...]]:
    generation = _obj(variant.get("generation"), "variant.generation")
    downstream = _obj(generation.get("downstream"), "variant.downstream")
    return {
        kind: tuple(_arr(_obj(downstream.get(kind), kind).get(field), f"{kind}.{field}"))
        for kind in KINDS
    }


def _retained_gpu_rollup(variant: Mapping[str, Any]) -> dict[str, object]:
    preparation = _obj(variant.get("preparation"), "variant.preparation")
    generation = _obj(variant.get("generation"), "variant.generation")
    telemetry_documents = [
        _obj(preparation.get("gpu_telemetry"), "preparation gpu telemetry"),
        _obj(generation.get("full_wall_gpu_telemetry"), "generation gpu telemetry"),
    ]
    statuses = [document.get("measurement_status") for document in telemetry_documents]
    devices = [
        _obj(raw, "gpu device")
        for document in telemetry_documents
        for raw in _arr(document.get("devices"), "gpu devices")
    ]
    memory_values = [float(item["memory_used_fraction_max"]) for item in devices]
    temperature_values = [float(item["temperature_celsius_max"]) for item in devices]
    return {
        "complete": statuses == ["MEASURED", "MEASURED"]
        and all(not document.get("errors") for document in telemetry_documents)
        and bool(devices),
        "measurement_statuses": statuses,
        "memory_used_fraction_max": max(memory_values) if memory_values else None,
        "temperature_celsius_max": max(temperature_values) if temperature_values else None,
    }


def _retained_gate(
    gate_id: str, passed: bool, observed: object, required: object, note: str
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "passed": passed,
        "observed": observed,
        "required": required,
        "note": note,
    }


def _retained_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    candidate_preparation = _obj(candidate.get("preparation"), "candidate.preparation")
    candidate_generation = _obj(candidate.get("generation"), "candidate.generation")
    baseline_preparation = _obj(baseline.get("preparation"), "baseline.preparation")
    baseline_generation = _obj(baseline.get("generation"), "baseline.generation")
    candidate_measurement = _obj(candidate_generation.get("measurement"), "candidate.measurement")
    baseline_measurement = _obj(baseline_generation.get("measurement"), "baseline.measurement")
    same_sample = _retained_sample_projection(candidate) == _retained_sample_projection(baseline)
    same_controls = _retained_control_projection(candidate) == _retained_control_projection(
        baseline
    )
    visual_parity = _retained_variant_projection(
        candidate, "visual"
    ) == _retained_variant_projection(baseline, "visual")
    metadata_parity = _retained_variant_projection(
        candidate, "metadata"
    ) == _retained_variant_projection(baseline, "metadata")
    output_text_parity = _retained_variant_projection(
        candidate, "output_text"
    ) == _retained_variant_projection(baseline, "output_text")
    normalized_output_parity = _retained_variant_projection(
        candidate, "normalized_output"
    ) == _retained_variant_projection(baseline, "normalized_output")
    content_field = "content_projection_semantic_sha256_values"
    durable_field = "durable_semantic_sha256_values"
    downstream_content_baseline = _retained_downstream_projection(baseline, content_field)
    downstream_content_candidate = _retained_downstream_projection(candidate, content_field)
    downstream_durable_baseline = _retained_downstream_projection(baseline, durable_field)
    downstream_durable_candidate = _retained_downstream_projection(candidate, durable_field)
    downstream_content_parity = {
        kind: downstream_content_candidate[kind] == downstream_content_baseline[kind]
        for kind in KINDS
    }
    downstream_durable_parity = {
        kind: downstream_durable_candidate[kind] == downstream_durable_baseline[kind]
        for kind in KINDS
    }
    preparation_speedup = float(baseline_preparation["wall_seconds"]) / float(
        candidate_preparation["wall_seconds"]
    )
    generation_stream_speedup = float(baseline_measurement["stream_run_wall_seconds"]) / float(
        candidate_measurement["stream_run_wall_seconds"]
    )
    baseline_cold_total = float(baseline_preparation["wall_seconds"]) + float(
        baseline_measurement["overall_wall_seconds"]
    )
    candidate_cold_total = float(candidate_preparation["wall_seconds"]) + float(
        candidate_measurement["overall_wall_seconds"]
    )
    cold_total_speedup = baseline_cold_total / candidate_cold_total
    gpu = _retained_gpu_rollup(candidate)
    full_resolution = candidate.get("max_side") == 0
    provider_lifecycle = (
        candidate_preparation.get("provider_process_start_count") == 1
        and candidate_preparation.get("provider_model_load_count") == 1
        and candidate_preparation.get("startup_load_included_in_wall") is True
    )
    gates = [
        _retained_gate(
            "SAME_SAMPLE_AND_SEGMENT_PLAN",
            same_sample,
            same_sample,
            True,
            "Recording identity, source size, duration, plan, and five segment intervals match.",
        ),
        _retained_gate(
            "GENERATION_CONTROL_PARITY",
            same_controls,
            same_controls,
            True,
            "Model, decoder, prompt token counts, camera, and single-worker controls match.",
        ),
        _retained_gate(
            "PERSISTENT_PROVIDER_SINGLE_START_AND_LOAD",
            provider_lifecycle,
            {
                "process_start_count": candidate_preparation.get("provider_process_start_count"),
                "model_load_count": candidate_preparation.get("provider_model_load_count"),
                "startup_load_included_in_wall": candidate_preparation.get(
                    "startup_load_included_in_wall"
                ),
            },
            {
                "process_start_count": 1,
                "model_load_count": 1,
                "startup_load_included_in_wall": True,
            },
            "Provider V2 must measure one persistent process and one model load.",
        ),
        _retained_gate(
            "PER_SEGMENT_PREPARATION_TIMING_RETAINED",
            candidate_preparation.get("per_segment_timing_status") == "MEASURED",
            candidate_preparation.get("per_segment_timing_status"),
            "MEASURED",
            "Candidate per-segment timings must be real worker measurements.",
        ),
        _retained_gate(
            "OUTPUT_TEXT_AND_NORMALIZED_JSON_PARITY",
            output_text_parity and normalized_output_parity,
            {
                "output_text_exact": output_text_parity,
                "normalized_json_semantic": normalized_output_parity,
            },
            {"output_text_exact": True, "normalized_json_semantic": True},
            "All five model outputs must remain exactly and semantically equal.",
        ),
        _retained_gate(
            "DOWNSTREAM_CONTENT_PARITY",
            all(downstream_content_parity.values()),
            downstream_content_parity,
            {kind: True for kind in KINDS},
            "QA, event, evidence, track, and fusion content projections must match.",
        ),
        _retained_gate(
            "GPU_TELEMETRY_COMPLETE",
            bool(gpu["complete"]),
            gpu["measurement_statuses"],
            ["MEASURED", "MEASURED"],
            "Preparation and full generation wall both require measured GPU telemetry.",
        ),
        _retained_gate(
            "PEAK_VRAM_SAFETY",
            gpu["memory_used_fraction_max"] is not None
            and float(gpu["memory_used_fraction_max"]) <= MAX_VRAM,
            gpu["memory_used_fraction_max"],
            {"maximum": MAX_VRAM},
            "Peak includes preparation and the endpoint generation harness.",
        ),
        _retained_gate(
            "TEMPERATURE_SAFETY",
            gpu["temperature_celsius_max"] is not None
            and float(gpu["temperature_celsius_max"]) <= MAX_TEMP_C,
            gpu["temperature_celsius_max"],
            {"maximum_celsius": MAX_TEMP_C},
            "Peak includes preparation and the endpoint generation harness.",
        ),
        _retained_gate(
            "PREPARATION_SPEEDUP",
            preparation_speedup >= (MIN_FULL_SPEEDUP if full_resolution else MIN_BOUNDED_SPEEDUP),
            preparation_speedup,
            {"minimum": MIN_FULL_SPEEDUP if full_resolution else MIN_BOUNDED_SPEEDUP},
            "Observed-v1 aggregate preparation wall divided by candidate aggregate wall.",
        ),
    ]
    if full_resolution:
        gates.append(
            _retained_gate(
                "FULL_RESOLUTION_VISUAL_PAYLOAD_EXACT_PARITY",
                visual_parity,
                visual_parity,
                True,
                "Provider metadata is excluded; every visual payload asset must hash-match.",
            )
        )
    passed = all(bool(item["passed"]) for item in gates)
    return {
        "candidate_variant_id": candidate.get("variant_id"),
        "production_eligible": False,
        "comparability": {
            "same_sample_and_segment_plan": same_sample,
            "generation_control_parity": same_controls,
        },
        "preparation": {
            "observed_wall_seconds": baseline_preparation["wall_seconds"],
            "candidate_wall_seconds": candidate_preparation["wall_seconds"],
            "speedup_ratio": preparation_speedup,
            "visual_payload_exact_parity_excluding_meta_json": visual_parity,
            "provider_metadata_exact_parity": metadata_parity,
            "provider_metadata_parity_note": (
                "meta.json binds provider/config telemetry and is expected to differ across "
                "provider generations; it is reported separately from visual payload quality."
            ),
        },
        "generation": {
            "observed_stream_run_wall_seconds": baseline_measurement["stream_run_wall_seconds"],
            "candidate_stream_run_wall_seconds": candidate_measurement["stream_run_wall_seconds"],
            "stream_speedup_ratio": generation_stream_speedup,
            "output_text_exact_parity": output_text_parity,
            "normalized_output_semantic_parity": normalized_output_parity,
            "inference_identity_parity": _retained_variant_projection(
                candidate, "inference_identity"
            )
            == _retained_variant_projection(baseline, "inference_identity"),
        },
        "downstream": {
            "content_projection_parity": downstream_content_parity,
            "durable_identity_parity": downstream_durable_parity,
            "durable_identity_note": (
                "Durable hashes bind provenance and may differ even when content projections "
                "are equal; content parity is the behavioral gate."
            ),
        },
        "cold_total": {
            "definition": "preparation.aggregate_wall + generation.harness_overall_wall",
            "observed_seconds": baseline_cold_total,
            "candidate_seconds": candidate_cold_total,
            "speedup_ratio": cold_total_speedup,
        },
        "gpu_safety": gpu,
        "gates": gates,
        "qualification_status": "LOCAL_QUALIFIED" if passed else "FAILED",
        "locally_adoptable": passed,
    }


def build_retained_artifact_report(
    *,
    observed_preparation_dir: Path,
    observed_generation_dir: Path,
    provider_v2_full_preparation_dir: Path,
    provider_v2_full_generation_dir: Path,
    provider_v2_bounded_preparation_dirs: Sequence[Path],
    provider_v2_bounded_generation_dirs: Sequence[Path],
) -> dict[str, object]:
    if len(provider_v2_bounded_preparation_dirs) != len(provider_v2_bounded_generation_dirs):
        raise MageDcvcQualificationInputError(
            "bounded preparation and generation directory counts must match"
        )
    if not provider_v2_bounded_preparation_dirs:
        raise MageDcvcQualificationInputError(
            "at least one bounded Provider V2 artifact pair is required"
        )
    observed = _retained_build_variant(
        preparation_root=observed_preparation_dir,
        generation_root=observed_generation_dir,
        generation=BASELINE,
    )
    full = _retained_build_variant(
        preparation_root=provider_v2_full_preparation_dir,
        generation_root=provider_v2_full_generation_dir,
        generation=PROVIDER_V2,
    )
    if full["max_side"] != 0:
        raise MageDcvcQualificationInputError(
            "Provider V2 full-resolution control must retain max_side=0"
        )
    bounded = [
        _retained_build_variant(
            preparation_root=preparation,
            generation_root=generation,
            generation=PROVIDER_V2,
        )
        for preparation, generation in zip(
            provider_v2_bounded_preparation_dirs,
            provider_v2_bounded_generation_dirs,
            strict=True,
        )
    ]
    if any(item["max_side"] == 0 for item in bounded):
        raise MageDcvcQualificationInputError("bounded candidates must use max_side > 0")
    bounded.sort(key=lambda item: int(item["max_side"]))
    comparisons = [_retained_comparison(observed, full)] + [
        _retained_comparison(observed, item) for item in bounded
    ]
    qualified = [item for item in comparisons if item["locally_adoptable"]]
    recommended = min(
        qualified,
        key=lambda item: float(_obj(item["cold_total"], "cold_total")["candidate_seconds"]),
        default=None,
    )
    payload: dict[str, object] = {
        "format_version": RETAINED_REPORT_VERSION,
        "evidence_class": "RETAINED_LOCAL_MEASUREMENT",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "scope": {
            "builder_starts_gpu_process": False,
            "retained_real_gpu_measurements_consumed": True,
            "sample_duration_ns": DURATION_NS,
            "segment_count": SEGMENT_COUNT,
            "camera_count": 1,
            "worker_count": 1,
            "note": (
                "This report qualifies only the retained local single-camera/single-worker A/B. "
                "It is not production or multi-camera evidence."
            ),
        },
        "observed_v1_measurement_limitation": {
            "per_segment_preparation_seconds": None,
            "provider_process_start_count": None,
            "provider_model_load_count": None,
            "provider_model_load_seconds": None,
            "status": "NOT_RECORDED",
            "note": (
                "The aggregate observed-v1 preparation wall is measured. Missing per-segment "
                "and lifecycle values are not inferred or allocated from that aggregate."
            ),
        },
        "thresholds": {
            "minimum_full_resolution_preparation_speedup_ratio": MIN_FULL_SPEEDUP,
            "minimum_bounded_preparation_speedup_ratio": MIN_BOUNDED_SPEEDUP,
            "maximum_peak_vram_fraction": MAX_VRAM,
            "maximum_temperature_celsius": MAX_TEMP_C,
        },
        "variants": {
            "observed_v1": observed,
            "provider_v2_full_resolution": full,
            "provider_v2_bounded": bounded,
        },
        "comparisons": comparisons,
        "qualification_status": "LOCAL_QUALIFIED" if recommended is not None else "FAILED",
        "recommended_variant_id": (
            None if recommended is None else recommended["candidate_variant_id"]
        ),
        "recommendation_note": (
            "A local recommendation does not change production_eligible=false and must not be "
            "presented as H100, multi-camera, or production qualification."
        ),
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def _build_from_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    legacy_requested = arguments.baseline_evidence is not None or bool(
        arguments.provider_v2_evidence
    )
    retained_values = (
        arguments.observed_preparation_dir,
        arguments.observed_generation_dir,
        arguments.provider_v2_full_preparation_dir,
        arguments.provider_v2_full_generation_dir,
    )
    retained_requested = any(item is not None for item in retained_values) or bool(
        arguments.provider_v2_bounded_preparation_dir
        or arguments.provider_v2_bounded_generation_dir
    )
    if legacy_requested and retained_requested:
        raise MageDcvcQualificationInputError(
            "legacy evidence mode and retained-artifact mode are mutually exclusive"
        )
    if retained_requested:
        if any(item is None for item in retained_values):
            raise MageDcvcQualificationInputError(
                "retained-artifact mode requires observed and full preparation/generation dirs"
            )
        return build_retained_artifact_report(
            observed_preparation_dir=arguments.observed_preparation_dir,
            observed_generation_dir=arguments.observed_generation_dir,
            provider_v2_full_preparation_dir=arguments.provider_v2_full_preparation_dir,
            provider_v2_full_generation_dir=arguments.provider_v2_full_generation_dir,
            provider_v2_bounded_preparation_dirs=(
                arguments.provider_v2_bounded_preparation_dir or []
            ),
            provider_v2_bounded_generation_dirs=(
                arguments.provider_v2_bounded_generation_dir or []
            ),
        )
    if arguments.baseline_evidence is None or not arguments.provider_v2_evidence:
        raise MageDcvcQualificationInputError(
            "provide either legacy evidence files or a complete retained-artifact set"
        )
    return build_qualification_report(
        baseline_evidence=arguments.baseline_evidence,
        provider_v2_evidence=arguments.provider_v2_evidence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = _build_from_arguments(arguments)
        destination = arguments.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(payload))
    except (MageDcvcQualificationInputError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "detail": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
