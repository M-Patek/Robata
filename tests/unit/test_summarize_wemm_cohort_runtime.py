from __future__ import annotations

from typing import Any

import pytest

from scripts.summarize_wemm_cohort_runtime import _markdown, build_compact_report


def _runtime() -> dict[str, Any]:
    return {
        "status": "MEASURED_NONPRODUCTION",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "official_gold_status": "NOT_ESTABLISHED",
        "official_quality_status": "NOT_MEASURED",
        "production_eligible": False,
        "source": {
            "window_count": 2,
            "camera_count": 6,
            "camera_window_input_count": 12,
            "represented_window_seconds": 8.0,
        },
        "arms": [
            {
                "arm_id": "serial",
                "batch_size": 1,
                "frame_count": 4,
                "input_count": 12,
                "decode_seconds_shared": 10.0,
                "inference_seconds": 2.0,
                "estimated_e2e_seconds": 3.0,
                "source_camera_normalized_realtime": 2.0,
                "rank_diagnostic": {
                    "top_label_counts_not_gold": {"a": 2},
                    "camera_consistency_not_gold": {
                        "window_count": 2,
                        "mean_modal_top1_fraction": 0.75,
                        "all_camera_same_fraction": 0.5,
                    },
                },
            },
            {
                "arm_id": "batch2",
                "batch_size": 2,
                "frame_count": 4,
                "input_count": 12,
                "decode_seconds_shared": 10.0,
                "inference_seconds": 2.0,
                "estimated_e2e_seconds": 2.0,
                "source_camera_normalized_realtime": 3.0,
                "observations": [
                    {
                        "phase_timings": {
                            "processor": 0.2,
                            "model": 0.3,
                            "postprocess": 0.05,
                            "total": 0.6,
                        }
                    },
                    {
                        "phase_timings": {
                            "processor": 0.4,
                            "model": 0.5,
                            "postprocess": 0.15,
                            "total": 1.1,
                        }
                    },
                ],
                "rank_diagnostic": {"top_label_counts_not_gold": {"a": 2}},
                "parity_vs_serial": {
                    "mean_cosine": 0.99,
                    "top1_equal_fraction": 1.0,
                    "full_order_equal_fraction": 0.5,
                    "row_order_preserved": True,
                    "within_tolerance": False,
                },
            },
        ],
    }


def _pipeline() -> dict[str, Any]:
    return {
        "pipeline": {
            "batch_size": 2,
            "window_chunk_size": 1,
            "queue_capacity": 1,
            "timing": {
                "queue_capacity": 1,
                "wall_seconds": 4.0,
                "overlap_seconds": 1.0,
                "estimated_speedup": 1.2,
                "producer_utilization": 0.9,
                "consumer_utilization": 0.3,
                "producer_backpressure_ns": 0,
                "phase_totals": [
                    {
                        "name": "media_decode",
                        "count": 2,
                        "total_ns": 10_000_000_000,
                        "mean_ns": 5_000_000_000,
                        "max_ns": 6_000_000_000,
                    },
                    {
                        "name": "model",
                        "count": 2,
                        "total_ns": 2_000_000_000,
                        "mean_ns": 1_000_000_000,
                        "max_ns": 1_500_000_000,
                    },
                ],
            },
        }
    }


def _matrix() -> dict[str, Any]:
    return {
        "arms": [
            {
                "arm_id": "f4_current_budget",
                "frame_count": 4,
                "total_pixel_budget": 262144,
                "observed_grid_thw": [[2, 14, 16]],
                "estimated_e2e_seconds": 3.0,
                "inference_speedup_vs_serial": 2.0,
                "batch_top1_parity": 1.0,
                "batch_full_order_parity": 0.8,
                "batch_mean_cosine_vs_serial": 0.99,
                "provisional_top1_distribution": {"a": 2},
            }
        ]
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_all_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_all_keys(child))
        return result
    return set()


def test_compact_report_keeps_runtime_diagnostics_without_heavy_fields() -> None:
    report = build_compact_report(_runtime(), pipeline=_pipeline(), matrix=_matrix())

    assert report["quality_claim"] is False
    assert report["throughput"][1]["arm_id"] == "batch2"
    assert report["throughput"][1]["parity_vs_serial"]["row_order_preserved"] is True
    assert report["throughput"][0]["camera_consistency"]["all_camera_same_fraction"] == 0.5
    assert report["frame_grid_matrix"][0]["observed_grid_thw"] == [[2, 14, 16]]
    # ``model`` is now an intentional phase name; no model metadata or
    # heavyweight identity material should be introduced elsewhere.
    keys = _all_keys(report)
    keys.discard("model")
    assert not ({"decode", "hash", "sha", "digest"} & keys)


def test_phase_breakdown_and_ingest_cached_summary_are_explicit() -> None:
    report = build_compact_report(_runtime(), pipeline=_pipeline(), matrix=_matrix())

    phase = report["phase_breakdown"]
    assert phase[0]["status"] == "NOT_RECORDED"
    assert phase[0]["total_source"] == "arm.inference_seconds"
    assert phase[0]["mean_seconds"]["total"] == 2.0
    assert phase[1]["status"] == "MEASURED"
    assert phase[1]["phase_observation_count"] == 2
    assert phase[1]["sum_seconds"]["processor"] == pytest.approx(0.6)
    assert phase[1]["mean_seconds"]["model"] == pytest.approx(0.4)
    assert phase[1]["sum_seconds"]["postprocess"] == pytest.approx(0.2)
    assert phase[1]["sum_seconds"]["total"] == pytest.approx(1.7)
    assert phase[1]["unattributed_seconds"] == pytest.approx(0.3)

    split = report["ingest_vs_cached_inference"]
    assert split["status"] == "DIAGNOSTIC_ESTIMATE"
    assert split["runtime_arms"][1]["ingest_seconds"] == 10.0
    assert split["runtime_arms"][1]["cached_inference_seconds"] == 2.0
    assert split["pipeline"]["ingest_seconds"] == 10.0
    assert split["pipeline"]["cached_inference_seconds"] == 2.0
    assert "persistent cache-hit" in split["cache_semantics"]


def test_embedded_pipeline_is_used_when_sidecar_is_omitted() -> None:
    runtime = _runtime()
    runtime["pipeline"] = _pipeline()
    report = build_compact_report(runtime, matrix=_matrix())

    assert report["pipeline"]["wall_seconds"] == 4.0
    assert report["ingest_vs_cached_inference"]["pipeline"]["ingest_seconds"] == 10.0


def test_markdown_contains_non_gold_boundary() -> None:
    report = build_compact_report(_runtime(), pipeline=_pipeline(), matrix=_matrix())
    rendered = _markdown(report)

    assert "No independent source-bound gold" in rendered
    assert "not accuracy" in rendered
    assert "batch2" in rendered
