from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, ClassVar

import pytest

import robata.benchmark.wemm_frame_grid_quality_matrix as matrix_module


class _Stats:
    def __init__(self, cache: _Cache) -> None:
        self.hits = cache.hits
        self.misses = cache.misses

    def to_dict(self) -> dict[str, int]:
        return {
            "scope_count": 2 if self.misses else 0,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "eviction_count": 0,
            "cached_chunk_count": 2 if self.misses else 0,
            "request_count": self.hits + self.misses,
        }


class _Cache:
    instances: ClassVar[list[_Cache]] = []

    def __init__(self, *, max_scopes: int) -> None:
        assert max_scopes == 2
        self.keys: list[Any] = []
        self.hits = 0
        self.misses = 0
        self.cleared = False
        self._seen: set[Any] = set()
        self.__class__.instances.append(self)

    def stats(self) -> _Stats:
        return _Stats(self)

    def clear(self) -> None:
        self.cleared = True


def _runtime_report(*, frame_count: int, pixel_budget: int) -> dict[str, Any]:
    grid = [2, 14, 16] if frame_count == 4 else [4, 10, 12]
    return {
        "status": "MEASURED_NONPRODUCTION",
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "arms": [
            {
                "control": True,
                "batch_size": 1,
                "inference_seconds": 1.0,
                "estimated_e2e_seconds": 2.0,
            },
            {
                "batch_size": 4,
                "input_count": 60,
                "model_call_count": 15,
                "video_item_count": 60,
                "decode_seconds_shared": 3.0,
                "inference_seconds": 1.5,
                "estimated_e2e_seconds": 4.5,
                "source_camera_normalized_realtime": 10.0,
                "observations": [
                    {"modality": "video", "video_grid_thw": [grid]},
                ],
                "rank_diagnostic": {
                    "top1_top2_margin_not_calibrated": {"mean": 0.1},
                    "camera_consistency_not_gold": {"mean_modal_top1_fraction": 1.0},
                },
                "parity_vs_serial": {
                    "row_count_equal": True,
                    "dimension_equal": True,
                    "mean_cosine": 1.0,
                    "min_cosine": 1.0,
                    "max_abs_delta": 0.0,
                    "mean_abs_delta": 0.0,
                    "top1_equal_fraction": 1.0,
                    "full_order_equal_fraction": 1.0,
                    "within_tolerance": True,
                    "row_order_preserved": True,
                    "order_context_count": 60,
                    "mismatch_count": 0,
                    "mismatches_truncated": False,
                    "mismatches": [],
                },
            },
        ],
    }


@pytest.fixture
def patched_matrix(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _Cache.instances.clear()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(matrix_module, "ProductionWemmDecodeCache", _Cache)

    def fake_runtime(manifest: Any, **kwargs: Any) -> dict[str, Any]:
        cache = kwargs["decode_cache"]
        key = kwargs["decode_scope_key"]
        calls.append({"kwargs": kwargs, "key": key, "manifest": manifest})
        if key in cache._seen:
            cache.hits += 1
        else:
            cache._seen.add(key)
            cache.misses += 1
        return _runtime_report(
            frame_count=int(kwargs["frame_count"]),
            pixel_budget=int(kwargs["pixel_budget"]),
        )

    monkeypatch.setattr(matrix_module, "run_wemm_cohort_runtime_benchmark", fake_runtime)
    return {"calls": calls}


def test_quality_matrix_reuses_decode_by_frame_count_and_keeps_batch4(
    patched_matrix: dict[str, Any],
) -> None:
    report = matrix_module.run_wemm_frame_grid_quality_matrix(
        "fixture-cohort.json",
        phrase_catalog="terra.json",
        model_directory="WeMM-Embedding-2B",
        device="cpu",
        pipeline_arm="f8_higher_budget",
    )

    calls = patched_matrix["calls"]
    assert len(calls) == 4
    assert [call["kwargs"]["frame_count"] for call in calls] == [4, 8, 4, 8]
    assert [call["kwargs"]["pixel_budget"] for call in calls] == [262144, 262144, 524288, 524288]
    assert all(call["kwargs"]["batch_sizes"] == (4,) for call in calls)
    assert calls[0]["key"] == calls[2]["key"]
    assert calls[1]["key"] == calls[3]["key"]
    assert calls[0]["key"] != calls[1]["key"]
    assert [call["kwargs"]["include_pipeline"] for call in calls] == [False, False, False, True]
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["official_gold_status"] == "NOT_ESTABLISHED"
    assert report["production_eligible"] is False
    assert report["matrix"]["batch_size"] == 4
    assert [arm["arm_id"] for arm in report["matrix"]["arms"]] == [
        "f4_current_budget",
        "f8_current_budget",
        "f4_higher_budget",
        "f8_higher_budget",
    ]
    assert report["matrix"]["arms"][0]["observed_video_grid_thw"] == [[2, 14, 16]]
    assert report["matrix"]["arms"][3]["expected_video_grid_thw"] == [4, 14, 16]
    assert [arm["decode_cache"]["cache_hit"] for arm in report["matrix"]["arms"]] == [
        False,
        False,
        True,
        True,
    ]
    assert report["decode_cache"]["hit_count"] == 2
    assert report["decode_cache"]["miss_count"] == 2
    assert _Cache.instances and _Cache.instances[0].cleared
    json.dumps(report)


def test_quality_matrix_rejects_unknown_pipeline_arm() -> None:
    with pytest.raises(matrix_module.WemmFrameGridQualityMatrixError, match="pipeline_arm"):
        matrix_module.run_wemm_frame_grid_quality_matrix(
            {},
            phrase_catalog={},
            model_directory="fixture",
            pipeline_arm="unknown",
        )


def test_quality_matrix_cli_forwards_options_and_writes_report() -> None:
    script = Path(__file__).parents[2] / "scripts" / "run_wemm_frame_grid_quality_matrix.py"
    namespace = runpy.run_path(str(script), run_name="robata_wemm_frame_grid_quality_matrix_cli")
    calls: list[dict[str, Any]] = []
    fake_report = {
        "status": "SUCCEEDED",
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "decode_cache": {"hit_count": 2, "miss_count": 2},
        "matrix": {"arms": [{"arm_id": "f4_current_budget"}]},
    }

    def fake_matrix(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, "kwargs": kwargs})
        return fake_report

    # ``runpy.run_path`` returns a snapshot; replace the imported symbol in
    # ``main``'s actual globals so the forwarding seam is used.
    namespace["main"].__globals__["run_wemm_frame_grid_quality_matrix"] = fake_matrix
    output = Path(".agent_tmp") / "wemm_frame_grid_quality_matrix_cli_test.json"
    argv = [
        "--manifest",
        "cohort.json",
        "--phrase-catalog",
        "terra.json",
        "--model-dir",
        "model",
        "--output",
        str(output),
        "--device",
        "cpu",
        "--pipeline-arm",
        "f4_current_budget",
        "--queue-capacity",
        "3",
    ]
    try:
        assert namespace["main"](argv) == 0
        assert calls[0]["kwargs"]["device"] == "cpu"
        assert calls[0]["kwargs"]["pipeline_arm"] == "f4_current_budget"
        assert calls[0]["kwargs"]["queue_capacity"] == 3
        assert output.is_file()
        assert json.loads(output.read_text(encoding="utf-8")) == fake_report
    finally:
        output.unlink(missing_ok=True)
