"""Compare two non-authoritative local Mage NF4 attention experiments.

The comparison is intentionally fail-closed.  Even a passing result is only an
``ADOPTABLE_EXPERIMENT``: the current Mage runtime identity does not bind attention
backend selection, so this tool can never authorize a production change.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.hashing import (  # noqa: E402
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)

REPORT_VERSION = "mage-nf4-attention-experiment-v1"
COMPARISON_VERSION = "mage-nf4-attention-comparison-v1"
REPORT_AUTHORITY = "NON_AUTHORITATIVE_EXPERIMENT"
NF4_LOAD_PROFILE = "bitsandbytes_4bit_nf4_v1"
MINIMUM_SPEEDUP_RATIO = 1.05
MAXIMUM_PEAK_VRAM_FRACTION = 0.85
ADOPTABLE = "ADOPTABLE_EXPERIMENT"
NOT_ADOPTABLE = "NOT_ADOPTABLE_EXPERIMENT"
PRODUCTION_PROHIBITION = "PROHIBITED_UNTIL_VERSIONED_RUNTIME_IDENTITY_BINDS_ATTENTION"


class MageAttentionComparisonInputError(ValueError):
    """A benchmark report is malformed or outside the non-authoritative scope."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MageAttentionComparisonInputError(
            f"could not read benchmark report: {resolved}"
        ) from error
    if not isinstance(value, dict):
        raise MageAttentionComparisonInputError(f"benchmark report must be an object: {resolved}")
    return value


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MageAttentionComparisonInputError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise MageAttentionComparisonInputError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MageAttentionComparisonInputError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MageAttentionComparisonInputError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MageAttentionComparisonInputError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MageAttentionComparisonInputError(f"{name} must be finite and positive")
    return number


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MageAttentionComparisonInputError(f"{name} must be a nonnegative integer")
    return value


def _validate_report(report: Mapping[str, Any], *, label: str) -> None:
    if report.get("report_version") != REPORT_VERSION:
        raise MageAttentionComparisonInputError(f"{label} report_version is unsupported")
    if (
        report.get("authority") != REPORT_AUTHORITY
        or report.get("production_eligible") is not False
    ):
        raise MageAttentionComparisonInputError(
            f"{label} must be explicitly non-authoritative and production-ineligible"
        )

    scope = _object(report.get("experimental_scope"), f"{label}.experimental_scope")
    if scope.get("runtime_identity_binds_attention_backend") is not False:
        raise MageAttentionComparisonInputError(
            f"{label} must state that runtime identity does not bind attention"
        )
    if scope.get("production_launcher_modified") is not False:
        raise MageAttentionComparisonInputError(
            f"{label} must not claim that the production launcher was modified"
        )
    if scope.get("production_adoption_requires_versioned_identity") is not True:
        raise MageAttentionComparisonInputError(
            f"{label} must require a future versioned identity before production adoption"
        )

    attention = _string(report.get("attention_requested"), f"{label}.attention_requested")
    if attention not in {"eager", "sdpa"}:
        raise MageAttentionComparisonInputError(f"{label} attention backend is unsupported")
    resolved = _object(report.get("attention_resolved"), f"{label}.attention_resolved")
    for field in ("top_level", "text", "vision"):
        value = resolved.get(field)
        if value is not None and not isinstance(value, str):
            raise MageAttentionComparisonInputError(
                f"{label}.attention_resolved.{field} must be a string or null"
            )

    runtime = _object(report.get("runtime_identity"), f"{label}.runtime_identity")
    _string(runtime.get("identity_version"), f"{label}.runtime_identity.identity_version")
    if runtime.get("load_profile") != NF4_LOAD_PROFILE:
        raise MageAttentionComparisonInputError(f"{label} must use the pinned NF4 load profile")
    if runtime.get("attention_backend_bound") is not False:
        raise MageAttentionComparisonInputError(
            f"{label} runtime identity must explicitly leave attention unbound"
        )

    _sha256(report.get("checkpoint_manifest_sha256"), f"{label}.checkpoint_manifest_sha256")
    _sha256(
        report.get("codec_cache_manifest_semantic_sha256"),
        f"{label}.codec_cache_manifest_semantic_sha256",
    )
    _sha256(
        report.get("codec_cache_namespace_identity"),
        f"{label}.codec_cache_namespace_identity",
    )
    _sha256(report.get("codec_policy_sha256"), f"{label}.codec_policy_sha256")
    _sha256(report.get("prompt_sha256"), f"{label}.prompt_sha256")
    _nonnegative_int(report.get("max_new_tokens"), f"{label}.max_new_tokens")
    if int(report["max_new_tokens"]) <= 0:
        raise MageAttentionComparisonInputError(f"{label}.max_new_tokens must be positive")

    videos = _array(report.get("input_videos"), f"{label}.input_videos")
    if not videos:
        raise MageAttentionComparisonInputError(f"{label}.input_videos must not be empty")
    for ordinal, raw_video in enumerate(videos):
        video = _object(raw_video, f"{label}.input_videos[{ordinal}]")
        if video.get("ordinal") != ordinal:
            raise MageAttentionComparisonInputError(
                f"{label}.input_videos ordinals must be contiguous and ordered"
            )
        _string(video.get("source_path"), f"{label}.input_videos[{ordinal}].source_path")
        _sha256(
            video.get("source_content_sha256"),
            f"{label}.input_videos[{ordinal}].source_content_sha256",
        )
        byte_count = _nonnegative_int(
            video.get("source_byte_count"),
            f"{label}.input_videos[{ordinal}].source_byte_count",
        )
        if byte_count <= 0:
            raise MageAttentionComparisonInputError(
                f"{label}.input_videos[{ordinal}].source_byte_count must be positive"
            )
        _sha256(
            video.get("logical_cache_identity"),
            f"{label}.input_videos[{ordinal}].logical_cache_identity",
        )

    warmup = _object(report.get("warmup"), f"{label}.warmup")
    warmup_budget = _nonnegative_int(warmup.get("max_new_tokens"), f"{label}.warmup.max_new_tokens")
    if warmup_budget <= 0:
        raise MageAttentionComparisonInputError(f"{label}.warmup.max_new_tokens must be positive")
    _nonnegative_int(warmup.get("actual_output_tokens"), f"{label}.warmup.actual_output_tokens")
    _positive_number(warmup.get("generation_seconds"), f"{label}.warmup.generation_seconds")
    _sha256(warmup.get("output_text_sha256"), f"{label}.warmup.output_text_sha256")

    _positive_number(report.get("model_load_seconds"), f"{label}.model_load_seconds")
    _positive_number(report.get("timed_wall_seconds"), f"{label}.timed_wall_seconds")
    _positive_number(report.get("generation_sum_seconds"), f"{label}.generation_sum_seconds")

    results = _array(report.get("results"), f"{label}.results")
    if len(results) != len(videos):
        raise MageAttentionComparisonInputError(
            f"{label}.results must contain exactly one result per input video"
        )
    for ordinal, raw_result in enumerate(results):
        result = _object(raw_result, f"{label}.results[{ordinal}]")
        video = _object(videos[ordinal], f"{label}.input_videos[{ordinal}]")
        if result.get("ordinal") != ordinal or result.get("video_path") != video.get("source_path"):
            raise MageAttentionComparisonInputError(
                f"{label}.results must preserve input-video order and paths"
            )
        _sha256(
            result.get("output_text_sha256"),
            f"{label}.results[{ordinal}].output_text_sha256",
        )
        _nonnegative_int(result.get("prompt_tokens"), f"{label}.results[{ordinal}].prompt_tokens")
        _nonnegative_int(result.get("output_tokens"), f"{label}.results[{ordinal}].output_tokens")
        _positive_number(
            result.get("generation_seconds"), f"{label}.results[{ordinal}].generation_seconds"
        )

    telemetry = _object(report.get("gpu_telemetry"), f"{label}.gpu_telemetry")
    _string(telemetry.get("measurement_status"), f"{label}.gpu_telemetry.measurement_status")
    _array(telemetry.get("summary"), f"{label}.gpu_telemetry.summary")


def _attention_is_resolved(report: Mapping[str, Any]) -> bool:
    requested = str(report["attention_requested"])
    resolved = _object(report["attention_resolved"], "attention_resolved")
    observed = tuple(
        value
        for value in (resolved.get("top_level"), resolved.get("text"), resolved.get("vision"))
        if value is not None
    )
    return (
        report.get("attention_resolution_verified") is True
        and bool(observed)
        and all(value == requested for value in observed)
    )


def _video_identity(report: Mapping[str, Any]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            video["ordinal"],
            video["source_path"],
            video["source_content_sha256"],
            video["source_byte_count"],
            video["logical_cache_identity"],
        )
        for video in (
            _object(item, "input_video") for item in _array(report["input_videos"], "input_videos")
        )
    )


def _output_identity(report: Mapping[str, Any]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            result["ordinal"],
            result["video_path"],
            result["output_text_sha256"],
            result["prompt_tokens"],
            result["output_tokens"],
        )
        for result in (_object(item, "result") for item in _array(report["results"], "results"))
    )


def _peak_vram_fraction(report: Mapping[str, Any]) -> float | None:
    telemetry = _object(report["gpu_telemetry"], "gpu_telemetry")
    if telemetry.get("measurement_status") != "MEASURED":
        return None
    summaries = _array(telemetry.get("summary"), "gpu_telemetry.summary")
    fractions: list[float] = []
    for ordinal, raw_summary in enumerate(summaries):
        summary = _object(raw_summary, f"gpu_telemetry.summary[{ordinal}]")
        value = summary.get("memory_used_fraction_max")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        fraction = float(value)
        if not math.isfinite(fraction) or fraction < 0.0:
            return None
        fractions.append(fraction)
    return max(fractions) if fractions else None


def _gate(gate_id: str, passed: bool, detail: str) -> dict[str, object]:
    return {"gate_id": gate_id, "passed": passed, "detail": detail}


def build_comparison_payload(*, baseline_report: Path, candidate_report: Path) -> dict[str, object]:
    baseline = _read_report(baseline_report)
    candidate = _read_report(candidate_report)
    _validate_report(baseline, label="baseline")
    _validate_report(candidate, label="candidate")

    baseline_wall = float(baseline["timed_wall_seconds"])
    candidate_wall = float(candidate["timed_wall_seconds"])
    baseline_generation = float(baseline["generation_sum_seconds"])
    candidate_generation = float(candidate["generation_sum_seconds"])
    wall_speedup = baseline_wall / candidate_wall
    generation_speedup = baseline_generation / candidate_generation
    baseline_peak_vram = _peak_vram_fraction(baseline)
    candidate_peak_vram = _peak_vram_fraction(candidate)

    checkpoint_parity = (
        baseline["checkpoint_manifest_sha256"] == candidate["checkpoint_manifest_sha256"]
    )
    cache_parity = (
        baseline["codec_cache_manifest_semantic_sha256"]
        == candidate["codec_cache_manifest_semantic_sha256"]
        and baseline["codec_cache_namespace_identity"]
        == candidate["codec_cache_namespace_identity"]
        and baseline["codec_policy_sha256"] == candidate["codec_policy_sha256"]
    )
    prompt_parity = baseline["prompt_sha256"] == candidate["prompt_sha256"]
    video_parity = _video_identity(baseline) == _video_identity(candidate)
    token_budget_parity = (
        baseline["max_new_tokens"] == candidate["max_new_tokens"]
        and _object(baseline["warmup"], "baseline.warmup")["max_new_tokens"]
        == _object(candidate["warmup"], "candidate.warmup")["max_new_tokens"]
    )
    output_parity = (
        _output_identity(baseline) == _output_identity(candidate)
        and _object(baseline["warmup"], "baseline.warmup")["output_text_sha256"]
        == _object(candidate["warmup"], "candidate.warmup")["output_text_sha256"]
    )
    attention_pair = {
        baseline["attention_requested"],
        candidate["attention_requested"],
    } == {"eager", "sdpa"}
    attention_resolved = _attention_is_resolved(baseline) and _attention_is_resolved(candidate)
    runtime_identity_parity = baseline["runtime_identity"] == candidate["runtime_identity"]
    telemetry_complete = baseline_peak_vram is not None and candidate_peak_vram is not None
    vram_safe = (
        telemetry_complete
        and baseline_peak_vram <= MAXIMUM_PEAK_VRAM_FRACTION
        and candidate_peak_vram <= MAXIMUM_PEAK_VRAM_FRACTION
    )
    wall_fast_enough = wall_speedup >= MINIMUM_SPEEDUP_RATIO
    generation_fast_enough = generation_speedup >= MINIMUM_SPEEDUP_RATIO

    gates = [
        _gate(
            "ATTENTION_PAIR",
            attention_pair,
            "reports must compare exactly eager and sdpa",
        ),
        _gate(
            "ATTENTION_RESOLUTION",
            attention_resolved,
            "each loaded model must resolve every observed attention field to its request",
        ),
        _gate(
            "RUNTIME_IDENTITY_PARITY",
            runtime_identity_parity,
            "runtime identities must match and continue to leave attention unbound",
        ),
        _gate("CHECKPOINT_PARITY", checkpoint_parity, "checkpoint manifest SHA must match"),
        _gate(
            "CODEC_CACHE_PARITY",
            cache_parity,
            "cache manifest, namespace, and codec policy identities must match",
        ),
        _gate("PROMPT_PARITY", prompt_parity, "prompt SHA must match"),
        _gate(
            "VIDEO_PARITY",
            video_parity,
            "ordered video paths, content SHAs, byte counts, and cache identities must match",
        ),
        _gate(
            "TOKEN_BUDGET_PARITY",
            token_budget_parity,
            "timed and warm-up token budgets must match",
        ),
        _gate(
            "OUTPUT_TEXT_SHA_PARITY",
            output_parity,
            "every timed and warm-up output text SHA and token count must match exactly",
        ),
        _gate(
            "GPU_TELEMETRY_COMPLETE",
            telemetry_complete,
            "both reports must contain MEASURED peak VRAM fractions",
        ),
        _gate(
            "PEAK_VRAM_SAFETY",
            bool(vram_safe),
            f"both peak VRAM fractions must be <= {MAXIMUM_PEAK_VRAM_FRACTION:.2f}",
        ),
        _gate(
            "TIMED_WALL_SPEEDUP",
            wall_fast_enough,
            f"candidate timed-wall speedup must be >= {MINIMUM_SPEEDUP_RATIO:.2f}x",
        ),
        _gate(
            "GENERATION_SPEEDUP",
            generation_fast_enough,
            f"candidate generation-sum speedup must be >= {MINIMUM_SPEEDUP_RATIO:.2f}x",
        ),
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    payload: dict[str, object] = {
        "comparison_version": COMPARISON_VERSION,
        "authority": REPORT_AUTHORITY,
        "production_eligible": False,
        "recommendation": ADOPTABLE if passed else NOT_ADOPTABLE,
        "production_adoption": PRODUCTION_PROHIBITION,
        "qualification_status": "PASSED" if passed else "FAILED",
        "baseline": {
            "report_path": str(baseline_report.expanduser().resolve()),
            "report_exact_sha256": exact_bytes_sha256(
                baseline_report.expanduser().resolve().read_bytes()
            ),
            "attention": baseline["attention_requested"],
            "timed_wall_seconds": baseline_wall,
            "generation_sum_seconds": baseline_generation,
            "peak_vram_fraction": baseline_peak_vram,
        },
        "candidate": {
            "report_path": str(candidate_report.expanduser().resolve()),
            "report_exact_sha256": exact_bytes_sha256(
                candidate_report.expanduser().resolve().read_bytes()
            ),
            "attention": candidate["attention_requested"],
            "timed_wall_seconds": candidate_wall,
            "generation_sum_seconds": candidate_generation,
            "peak_vram_fraction": candidate_peak_vram,
        },
        "thresholds": {
            "minimum_speedup_ratio": MINIMUM_SPEEDUP_RATIO,
            "maximum_peak_vram_fraction": MAXIMUM_PEAK_VRAM_FRACTION,
        },
        "metrics": {
            "timed_wall_speedup_ratio": wall_speedup,
            "timed_wall_reduction_percent": (1.0 - candidate_wall / baseline_wall) * 100.0,
            "generation_speedup_ratio": generation_speedup,
            "generation_reduction_percent": (1.0 - candidate_generation / baseline_generation)
            * 100.0,
        },
        "gates": gates,
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def _write_report(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(resolved)
    return exact_bytes_sha256(resolved.read_bytes())


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = build_comparison_payload(
            baseline_report=arguments.baseline_report,
            candidate_report=arguments.candidate_report,
        )
        output_sha256 = _write_report(arguments.output, payload)
    except (MageAttentionComparisonInputError, OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"ok": False, "code": "MAGE_NF4_ATTENTION_COMPARISON_FAILED", "detail": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "authority": REPORT_AUTHORITY,
                "production_eligible": False,
                "qualification_status": payload["qualification_status"],
                "recommendation": payload["recommendation"],
                "production_adoption": PRODUCTION_PROHIBITION,
                "output": str(arguments.output.expanduser().resolve()),
                "output_exact_sha256": output_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
