#!/usr/bin/env python3
"""Run Qwen's candidate verifier on complete native video windows.

This is a bounded production-shadow runner.  WeMM Top-K labels are supplied by
the candidate pack; Qwen is never allowed to create a new label.  Each request
contains one complete bounded native video (never a stream/chunk).  The output
is a non-gold verifier sidecar suitable for the lightweight joiner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_qwen_candidate_verifier import (  # noqa: E402
    ProductionWemmQwenCandidateVerifierError,
    build_candidate_verifier_prompt,
    parse_qwen_candidate_verification_output,
)
from robata.benchmark.qwen_native_video import sample_qwen_native_video  # noqa: E402
from robata.inference.local_hf_runtime import (  # noqa: E402
    LocalHfVideoGenerationRequest,
    LocalHuggingFaceVisionRuntime,
)

REPORT_FORMAT = "robata-production-wemm-qwen-candidate-verification-native-v1"
AMBIGUITY_SELECTION_FORMAT = "robata-production-wemm-ambiguity-selection-v1"
# Production candidate verification deliberately does not derive content
# identifiers for decoded frames.  The compatibility switch is retained for
# older callers, but this native route binds it off and records that fact in
# every report.
FRAME_DIGEST_POLICY = "DISABLED_FOR_PRODUCTION_CANDIDATE_VERIFIER"
DEFAULT_CANDIDATES = ROOT / ".agent_tmp" / "production_review_candidate_pack_4s_20260828_r2.json"
DEFAULT_MANIFEST = ROOT / ".agent_tmp" / "production_sample_cohort_manifest_4s_20260827.json"
DEFAULT_VIDEO_ROOT = ROOT / ".local" / "production-worker-it" / "state2" / "video-view"
DEFAULT_MODEL_DIR = Path(r"D:\HuggingFace\Qwen3-VL-4B-Instruct")
DEFAULT_OFFLOAD = ROOT / ".local" / "qwen-candidate-verifier-offload"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProductionWemmQwenCandidateVerifierError(f"{path} must contain an object")
    return value


def _interval(window: Mapping[str, Any]) -> tuple[float, float]:
    """Read either the legacy array or selector mapping interval shape.

    The ambiguity selector deliberately stores a mapping with an explicit
    ``WINDOW_CONTEXT_ONLY`` status, while the historical candidate pack used a
    two-element array.  Both are source-context intervals; neither is an
    inferred action boundary.
    """

    value = window.get("source_interval", window.get("interval"))
    if isinstance(value, Mapping):
        start_value = value.get("start_seconds", value.get("start_time_sec"))
        end_value = value.get("end_seconds", value.get("end_time_sec"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            raise ProductionWemmQwenCandidateVerifierError("candidate window lacks source_interval")
        start_value, end_value = value[0], value[1]
    else:
        raise ProductionWemmQwenCandidateVerifierError("candidate window lacks source_interval")
    try:
        start, end = float(start_value), float(end_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmQwenCandidateVerifierError(
            "candidate window interval is not numeric"
        ) from exc
    if end <= start:
        raise ProductionWemmQwenCandidateVerifierError("candidate window interval is invalid")
    return start, end


def _structured_value(candidate: Mapping[str, Any], field: str) -> Any:
    labels = candidate.get("structured_labels")
    if not isinstance(labels, Mapping):
        return None
    value = labels.get(field)
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def _normalise_selector_candidate(
    candidate: Mapping[str, Any], *, fallback_rank: int
) -> dict[str, Any]:
    """Adapt one compact selector Top-K row to the verifier candidate shape.

    Selector rows intentionally retain only compact structured labels.  The
    verifier prompt/parser accepts the legacy ``verb``/``noun`` fields, so we
    materialize those aliases without changing the original label or score.
    """

    result = dict(candidate)
    rank = candidate.get("rank", fallback_rank)
    try:
        rank_value = int(rank)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionWemmQwenCandidateVerifierError(
            "selector Top-K rank must be a positive integer"
        ) from exc
    if rank_value < 1:
        raise ProductionWemmQwenCandidateVerifierError(
            "selector Top-K rank must be a positive integer"
        )
    result["rank"] = rank_value
    label = result.get("label_text") or result.get("canonical_label")
    if not isinstance(label, str) or not label.strip():
        label = " ".join(
            str(part).strip()
            for part in (
                _structured_value(candidate, "verb"),
                _structured_value(candidate, "noun"),
            )
            if part is not None and str(part).strip()
        )
    if label:
        result.setdefault("label_text", label)
        result.setdefault("canonical_label", label)
    result.setdefault("verb", _structured_value(candidate, "verb"))
    result.setdefault("noun", _structured_value(candidate, "noun"))
    # Selector uses ``provisional_id``; ``label_id`` is an optional verifier
    # field and is only copied when available.
    if "label_id" not in result and "provisional_id" in candidate:
        result["label_id"] = candidate.get("provisional_id")
    return result


def _selector_top_k(
    window: Mapping[str, Any], *, proposal_index: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics = window.get("proposal_diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes, bytearray)):
        raise ProductionWemmQwenCandidateVerifierError(
            "ambiguity-selection window lacks proposal_diagnostics"
        )
    diagnostics_list = [item for item in diagnostics if isinstance(item, Mapping)]
    if len(diagnostics_list) != len(diagnostics):
        raise ProductionWemmQwenCandidateVerifierError(
            "ambiguity-selection proposal_diagnostics must contain objects"
        )
    if proposal_index is None:
        if len(diagnostics_list) != 1:
            raise ProductionWemmQwenCandidateVerifierError(
                "ambiguity-selection window has multiple proposals; "
                "pass --proposal-index explicitly"
            )
        selected_index = 0
    else:
        if isinstance(proposal_index, bool) or proposal_index < 0:
            raise ProductionWemmQwenCandidateVerifierError(
                "--proposal-index must be a non-negative integer"
            )
        if proposal_index >= len(diagnostics_list):
            raise ProductionWemmQwenCandidateVerifierError(
                "--proposal-index is outside proposal_diagnostics"
            )
        selected_index = proposal_index
    diagnostic = diagnostics_list[selected_index]
    raw = diagnostic.get("top_k", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ProductionWemmQwenCandidateVerifierError(
            "ambiguity-selection proposal top_k must be an array"
        )
    top_k = [
        _normalise_selector_candidate(item, fallback_rank=index + 1)
        for index, item in enumerate(raw)
        if isinstance(item, Mapping)
    ]
    if len(top_k) != len(raw):
        raise ProductionWemmQwenCandidateVerifierError(
            "ambiguity-selection proposal top_k must contain objects"
        )
    ranks = [int(item["rank"]) for item in top_k]
    if len(set(ranks)) != len(ranks):
        raise ProductionWemmQwenCandidateVerifierError(
            "ambiguity-selection proposal top_k contains duplicate ranks"
        )
    return top_k, {
        "candidate_format": AMBIGUITY_SELECTION_FORMAT,
        "adapter": "selector_proposal_diagnostics_v1",
        "proposal_index": selected_index,
        "proposal_id": diagnostic.get("proposal_id"),
        "proposal_count": len(diagnostics_list),
    }


def _top_k_with_adapter(
    window: Mapping[str, Any], *, proposal_index: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if "proposal_diagnostics" in window:
        return _selector_top_k(window, proposal_index=proposal_index)
    context = window.get("model_context")
    context = context if isinstance(context, Mapping) else {}
    wemm = context.get("wemm")
    wemm = wemm if isinstance(wemm, Mapping) else {}
    raw = wemm.get("top_k", wemm.get("predictions", []))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [dict(item) for item in raw if isinstance(item, Mapping)], {
            "candidate_format": "legacy_candidate_pack_v1",
            "adapter": "legacy_model_context_v1",
            "proposal_index": None,
            "proposal_id": None,
            "proposal_count": None,
        }
    return [], {
        "candidate_format": "unknown",
        "adapter": "none",
        "proposal_index": None,
        "proposal_id": None,
        "proposal_count": None,
    }


def _top_k(window: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Backward-compatible candidate accessor for legacy callers/tests."""

    return _top_k_with_adapter(window)[0]


def _source_window(
    window: Mapping[str, Any], manifest_by_id: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Resolve camera metadata, falling back to selector declarations.

    The historical runner requires a separate manifest.  Selector rows carry
    their own declared camera list, so allowing that row as a source fallback
    makes the adapter usable with a final full-corpus selection queue while
    preserving the explicit manifest when present.
    """

    window_id = str(window.get("window_id"))
    source = manifest_by_id.get(window_id)
    if source is not None:
        return source
    if window.get("proposal_diagnostics") is not None:
        return window
    return None


def _camera_ids(source: Mapping[str, Any]) -> list[str]:
    for key in ("camera_ids", "declared_camera_ids"):
        value = source.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            camera_ids = [str(item) for item in value if str(item).strip()]
            if camera_ids:
                return camera_ids
    cameras = source.get("cameras")
    if isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes, bytearray)):
        camera_ids = [
            str(item.get("camera_id"))
            for item in cameras
            if isinstance(item, Mapping) and str(item.get("camera_id", "")).strip()
        ]
        if camera_ids:
            return camera_ids
    source_ref = source.get("source_ref")
    if isinstance(source_ref, Mapping):
        nested = source_ref.get("source")
        if isinstance(nested, Mapping):
            value = nested.get("camera_ids")
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [str(item) for item in value]
    return []


def _request(
    video: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    compute_frame_sha256: bool = False,
) -> LocalHfVideoGenerationRequest:
    return LocalHfVideoGenerationRequest(
        video_payloads=video.frame_payloads,
        frame_indices=video.frame_indices,
        frame_timestamps_seconds=video.frame_timestamps_seconds,
        source_fps=video.source_fps,
        total_num_frames=video.total_num_frames,
        width=video.width,
        height=video.height,
        duration_seconds=video.duration_seconds,
        source_window_start_seconds=getattr(
            video, "source_window_start_seconds", video.interval_start_seconds
        ),
        source_window_end_seconds=getattr(
            video, "source_window_end_seconds", video.interval_end_seconds
        ),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stop_after_first_complete_json_object=True,
        compute_frame_sha256=compute_frame_sha256,
    )


def _run_with_runtime(
    args: argparse.Namespace,
    runtime: LocalHuggingFaceVisionRuntime,
    *,
    load_observation: Any | None = None,
    close_runtime: bool = False,
) -> dict[str, Any]:
    """Execute one verifier shard with a caller-owned resident runtime.

    The one-shot CLI wrapper below still owns its runtime.  Batch callers can
    instead pass one resident ``LocalHuggingFaceVisionRuntime`` and invoke the
    public :func:`run_with_runtime` once per staged recording, avoiding a 4B
    model reload at every recording boundary.
    """

    include_optional_fields = bool(getattr(args, "include_optional_fields", False))
    verdict_scope = getattr(args, "verdict_scope", "selected_only")
    if include_optional_fields and verdict_scope != "selected_only":
        raise ProductionWemmQwenCandidateVerifierError(
            "--include-optional-fields requires --verdict-scope selected_only"
        )
    candidates = _load(args.candidates)
    manifest = _load(args.manifest)
    candidate_windows = candidates.get("windows", [])
    manifest_windows = manifest.get("windows", [])
    if not isinstance(candidate_windows, Sequence) or not isinstance(manifest_windows, Sequence):
        raise ProductionWemmQwenCandidateVerifierError("candidate/manifest windows must be arrays")
    manifest_by_id = {
        str(w.get("window_id")): w for w in manifest_windows if isinstance(w, Mapping)
    }
    selected = [w for w in candidate_windows if isinstance(w, Mapping)]
    if args.window_id:
        selected = [w for w in selected if str(w.get("window_id")) in set(args.window_id)]
    if args.limit is not None:
        if args.limit <= 0:
            raise ProductionWemmQwenCandidateVerifierError("--limit must be positive")
        selected = selected[: args.limit]
    requested = set(args.camera_id or [])
    rows: list[dict[str, Any]] = []
    model_invoked = False
    source_media_decoded = False
    started = time.perf_counter()
    if load_observation is None:
        # The concrete runtime exposes ``loaded``/``load_observation`` so a
        # resident caller does not even re-enter ``load`` at each recording.
        # Duck-typed fakes used by existing tests may expose only ``load``;
        # retain that compatibility fallback.
        if bool(getattr(runtime, "loaded", False)):
            load_observation = runtime.load_observation
        else:
            load_observation = runtime.load()
    try:
        for raw_window in selected:
            window = dict(raw_window)
            window_id = str(window.get("window_id"))
            source = _source_window(window, manifest_by_id)
            if source is None:
                continue
            start, end = _interval(window)
            duration = end - start
            top_k, candidate_adapter = _top_k_with_adapter(
                window, proposal_index=getattr(args, "proposal_index", None)
            )
            camera_ids = _camera_ids(source)
            if requested:
                camera_ids = [c for c in camera_ids if str(c) in requested]
            for camera_id in camera_ids:
                camera_id = str(camera_id)
                row: dict[str, Any] = {
                    "window_id": window_id,
                    "ordinal": window.get("ordinal"),
                    "interval": [start, end],
                    "camera_id": camera_id,
                    "input_mode": "native_video",
                    "native_video_complete": True,
                    "status": "FAILED",
                    "candidate_adapter": candidate_adapter,
                    "provenance": {
                        "frame_digest_policy": FRAME_DIGEST_POLICY,
                        "frame_sha256_computed": False,
                    },
                }
                try:
                    prompt = build_candidate_verifier_prompt(
                        top_k,
                        window_duration_seconds=duration,
                        window_id=window_id,
                        verdict_scope=verdict_scope,
                        include_optional_fields=include_optional_fields,
                    )
                    video = sample_qwen_native_video(
                        args.video_root / f"{camera_id}.mp4",
                        start_seconds=start,
                        end_seconds=end,
                        frame_count=args.frame_count,
                        context_before_seconds=0.0,
                        context_after_seconds=0.0,
                        jpeg_quality=args.jpeg_quality,
                    )
                    source_media_decoded = True
                    model_invoked = True
                    observation = runtime.generate_video(
                        request=_request(
                            video,
                            prompt,
                            args.max_new_tokens,
                            compute_frame_sha256=False,
                        )
                    )
                    # A runtime that returns a non-empty digest despite the
                    # production opt-out violates this route's provenance
                    # contract; do not silently publish a contradictory row.
                    if getattr(observation, "frame_sha256", ()):
                        raise ProductionWemmQwenCandidateVerifierError(
                            "production verifier runtime returned frame_sha256 despite "
                            "the disabled digest policy"
                        )
                    parsed = parse_qwen_candidate_verification_output(
                        observation.output_text,
                        top_k,
                        window_duration_seconds=duration,
                        require_optional_fields=include_optional_fields,
                    )
                    row.update(
                        {
                            "status": "SUCCEEDED",
                            "raw_text": observation.output_text,
                            "parsed_verification": parsed,
                            "frame_indices": list(observation.frame_indices),
                            "frame_timestamps_seconds": list(observation.frame_timestamps_seconds),
                            "rendered_frame_sizes": [
                                list(x) for x in observation.rendered_frame_sizes
                            ],
                            "prompt_tokens": observation.prompt_tokens,
                            "output_tokens": observation.output_tokens,
                            "generation_seconds": observation.generation_seconds,
                            "gpu_peak_allocated_bytes": observation.gpu_peak_allocated_bytes,
                            # Preserve the scalar native-processor telemetry so a
                            # successful verifier run is auditable without retaining
                            # image payloads or recomputing the media path.
                            "visual_input": (
                                observation.visual_input.as_dict()
                                if observation.visual_input is not None
                                else None
                            ),
                            "prompt_version": (
                                f"wemm-top-k-bound-native-video-{verdict_scope}-"
                                f"{'fields' if include_optional_fields else 'compact'}-v1"
                            ),
                            "verifier_profile": (
                                "selected_only_fields"
                                if include_optional_fields
                                else f"{verdict_scope}_compact"
                            ),
                        }
                    )
                except Exception as exc:  # preserve per-camera failure and continue
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    finally:
        if close_runtime:
            runtime.close()
    return {
        "format": REPORT_FORMAT,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "official_gold_status": "NOT_ESTABLISHED",
        "official_quality_status": "NOT_MEASURED",
        "production_eligible": False,
        "quality_claim": False,
        "source": {
            "candidates": str(args.candidates),
            "manifest": str(args.manifest),
            "video_root": str(args.video_root),
        },
        "model": {
            "identifier": "Qwen3-VL-4B-Instruct",
            "route": "complete_native_video_candidate_verifier",
            "frame_count": args.frame_count,
            "max_image_side": args.max_image_side,
            "max_new_tokens": args.max_new_tokens,
            "verifier_profile": (
                "selected_only_fields" if include_optional_fields else f"{verdict_scope}_compact"
            ),
            "load_seconds": getattr(load_observation, "load_seconds", None),
            "gpu_name": getattr(load_observation, "gpu_name", None),
            "candidate_input_format": candidates.get("format", "unknown"),
            "candidate_adapter": (
                "ambiguity_selection_v1"
                if candidates.get("format") == AMBIGUITY_SELECTION_FORMAT
                else "legacy_candidate_pack_v1"
            ),
            "frame_digest_policy": FRAME_DIGEST_POLICY,
        },
        "provenance": {
            "frame_digest_policy": FRAME_DIGEST_POLICY,
            "frame_sha256_computed": False,
        },
        "windows": rows,
        "controls": {
            "model_invoked": model_invoked,
            "source_media_decoded": source_media_decoded,
            "gold_included": False,
            "epic_ontology_used": False,
            "mapper_used": False,
            "hash_or_digest_computed": False,
            "frame_sha256_computed": False,
            "complete_native_video_only": True,
        },
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_with_runtime(
    args: argparse.Namespace,
    runtime: LocalHuggingFaceVisionRuntime,
    *,
    load_observation: Any | None = None,
) -> dict[str, Any]:
    """Run one shard without taking ownership of the Qwen runtime.

    The caller is responsible for constructing, loading, and closing
    ``runtime``.  This preserves the exact native-video request, parser, raw
    per-camera rows, and no-digest provenance contract of :func:`run` while
    allowing a recording-level batch loop to keep one model resident.
    """

    return _run_with_runtime(
        args,
        runtime,
        load_observation=load_observation,
        close_runtime=False,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """One-shot compatibility wrapper that owns the runtime lifecycle."""

    runtime = LocalHuggingFaceVisionRuntime(
        model_directory=args.model_dir,
        offload_directory=args.offload_dir,
        max_image_side=args.max_image_side,
        gpu_weight_memory_gib=args.gpu_weight_memory_gib,
        cpu_weight_memory_gib=args.cpu_weight_memory_gib,
    )
    try:
        return _run_with_runtime(args, runtime, close_runtime=False)
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--offload-dir", type=Path, default=DEFAULT_OFFLOAD)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--proposal-index",
        type=int,
        help=(
            "select one proposal_diagnostics index from an ambiguity-selection row; "
            "required when a selected window contains multiple proposals"
        ),
    )
    parser.add_argument("--window-id", action="append", help="restrict to one or more window IDs")
    parser.add_argument("--camera-id", action="append")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-weight-memory-gib", type=int, default=5)
    parser.add_argument("--cpu-weight-memory-gib", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--verdict-scope",
        choices=("selected_only", "all_candidates", "pairwise"),
        default="selected_only",
        help="candidate verdict emission mode; all_candidates/pairwise are compact diagnostics",
    )
    parser.add_argument(
        "--include-optional-fields",
        action="store_true",
        help=(
            "opt into one selected-only field-complete verdict (verb/noun/attributes/"
            "location/hand statuses, confidence, evidence, boundary)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, ProductionWemmQwenCandidateVerifierError) as exc:
        print(f"candidate verifier native run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(report["windows"]),
                "model_invoked": True,
                "official_quality_status": "NOT_MEASURED",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
