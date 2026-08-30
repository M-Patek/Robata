from __future__ import annotations

import json
from pathlib import Path

from robata.benchmark.production_wemm_preannotation import build_preannotation_envelope
from robata.benchmark.production_wemm_preannotation_aggregate import (
    AGGREGATE_FORMAT,
    aggregate_production_wemm_preannotations,
    render_markdown,
)


def _envelope(recording_id: str = "rec-01", *, windows: int = 2) -> dict[str, object]:
    source = {
        "recording_id": recording_id,
        "path": f"{recording_id}.mcap",
        "camera_count": 6,
    }
    rows: list[dict[str, object]] = []
    for ordinal in range(windows):
        rows.append(
            {
                "window_id": f"{recording_id}-w{ordinal:04d}",
                "ordinal": ordinal,
                "start_seconds": ordinal * 8.0,
                "end_seconds": (ordinal + 1) * 8.0,
                "camera_ids": [f"cam_{index:02d}" for index in range(1, 7)],
                "proposals": [
                    {
                        "label_text": "open cupboard" if ordinal == 0 else "close cupboard",
                        "structured_labels": {
                            "verb": "open" if ordinal == 0 else "close",
                            "noun": "cupboard",
                            "attributes": None,
                            "location": {"value": "kitchen", "status": "MEASURED"},
                            "hand": {"status": "NOT_OBSERVABLE"},
                        },
                        "confidence": 0.8 - ordinal * 0.1,
                        "evidence": [{"camera_id": "cam_01", "text": "door moves"}],
                        "top_k": [
                            {
                                "label_text": "open cupboard" if ordinal == 0 else "close cupboard",
                                "verb": "open" if ordinal == 0 else "close",
                                "noun": "cupboard",
                                "score": 0.8 - ordinal * 0.1,
                            },
                            {
                                "label_text": "open drawer",
                                "verb": "open",
                                "noun": "drawer",
                                "score": 0.7,
                            },
                        ],
                    }
                ],
            }
        )
    raw_windows = []
    for row in rows:
        observations = [
            {
                "camera_id": camera,
                "decoded_frames": 4,
                "messages_examined": 10,
                "decode_failures": (["H264_DECODE_ERROR:bad frame"] if camera == "cam_02" else []),
            }
            for camera in [f"cam_{index:02d}" for index in range(1, 7)]
        ]
        raw_windows.append({"window_id": row["window_id"], "input_observations": observations})
    return build_preannotation_envelope(
        source,
        rows,
        raw_model_output={"windows": raw_windows},
        model={"name": "WeMM-Embedding-2B"},
    )


def test_aggregate_direct_sidecar_reports_structural_metrics() -> None:
    report = aggregate_production_wemm_preannotations(_envelope())
    assert report["format"] == AGGREGATE_FORMAT
    assert report["status"] == "READ_ONLY_SUMMARY"
    coverage = report["coverage"]
    assert coverage["valid_artifact_count"] == 1
    assert coverage["recording_count"] == 1
    assert coverage["processing_window_count"] == 2
    assert coverage["proposal_count"] == 2
    assert coverage["camera_window_input_count"] == 12
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["controls"]["gold_read"] is False
    assert report["controls"]["media_decoded"] is False
    assert report["top_k"]["cardinality_histogram"] == {"2": 2}
    assert report["decode_warnings"]["warning_entry_count"] == 2
    assert report["decode_warnings"]["warning_types"] == {"H264_DECODE_ERROR": 2}
    fields = report["field_completeness"]["fields"]
    assert fields["verb"]["measured_count"] == 2
    assert fields["attributes"]["measured_count"] == 0
    assert fields["hand"]["status_counts"]["NOT_OBSERVABLE"] == 2
    assert "action boundaries" in " ".join(report["limitations"])
    assert "Top-K" in render_markdown(report)


def test_directory_discovers_only_preannotations_and_checkpoint_paths(tmp_path: Path) -> None:
    root = tmp_path / "run"
    pre = root / "preannotations"
    review = root / "review"
    pre.mkdir(parents=True)
    review.mkdir()
    sidecar = pre / "rec.json"
    sidecar.write_text(json.dumps(_envelope("rec-dir", windows=1)), encoding="utf-8")
    (review / "review-pack.json").write_text(json.dumps(_envelope("wrong")), encoding="utf-8")
    report = aggregate_production_wemm_preannotations(root)
    assert report["coverage"]["valid_artifact_count"] == 1
    assert report["coverage"]["recording_count"] == 1


def test_batch_checkpoint_and_bad_sidecar_are_reported_without_silent_merge(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    pre = root / "preannotations"
    pre.mkdir(parents=True)
    good = pre / "good.json"
    good.write_text(json.dumps(_envelope("rec-same", windows=1)), encoding="utf-8")
    bad = pre / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    checkpoint = root / "batch-run.json"
    checkpoint.write_text(
        json.dumps(
            {
                "format": "robata-production-wemm-batch-run-v1",
                "items": [
                    {"preannotation_path": "preannotations/good.json"},
                    {"preannotation_path": "preannotations/bad.json"},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = aggregate_production_wemm_preannotations(checkpoint)
    assert report["coverage"]["valid_artifact_count"] == 1
    assert report["coverage"]["invalid_artifact_count"] == 1


def test_duplicate_recording_and_window_ids_are_visible() -> None:
    first = _envelope("same", windows=1)
    second = _envelope("same", windows=1)
    # Sequence input is intentionally supported for in-memory audit tests.
    report = aggregate_production_wemm_preannotations([first, second])
    assert report["duplicates"]["recording_ids"] == ["same"]
    assert report["duplicates"]["window_ids"] == ["same-w0000"]
