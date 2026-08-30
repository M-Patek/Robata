from __future__ import annotations

from robata.benchmark.production_review_queue_batches import (
    ProductionReviewQueueBatchError,
    build_review_queue_batches,
    validate_review_queue_batches,
)


def _draft(window_count: int = 3) -> dict[str, object]:
    windows: list[dict[str, object]] = []
    for index in range(window_count):
        window_id = f"recording-01-w{index:04d}"
        top_k = [
            {"rank": rank, "label_text": f"action {rank}", "score": 0.9 - rank / 100}
            for rank in range(1, 7)
        ]
        segment = {
            "segment_id": f"{window_id}-p01",
            "label_text": "pick up garment",
            "structured_labels": {"verb": "pick up", "noun": "garment"},
            "confidence": 0.8,
            "evidence": [{"camera_id": "cam_01", "text": "hand moves"}],
            "camera_support": [f"cam_{camera:02d}" for camera in range(1, 7)],
            "top_k": top_k,
            "margin": 0.1,
            "proposal_status": "PROPOSED",
            "split_hint": False,
            "start_seconds": None,
            "end_seconds": None,
            "boundary_status": "NOT_MEASURED",
            "timestamp_basis": None,
            "window_context": {"is_action_boundary": False},
        }
        windows.append(
            {
                "window_id": window_id,
                "recording_id": "recording-01",
                "ordinal": index,
                "source_interval": {
                    "start_seconds": float(index * 8),
                    "end_seconds": float(index * 8 + 8),
                    "status": "WINDOW_CONTEXT_ONLY",
                },
                "source_provenance": {
                    "qa_status": "PENDING",
                    "source_preflight_status": "PASS",
                    "review_pack_path": "review.json",
                    "archive_member": "file/recording-01.mcap",
                    "source_path": "recording-01.mcap",
                },
                "source_ref": {"recording_id": "recording-01"},
                "annotation_draft": {"segments": [segment]},
            }
        )
    return {
        "format": "robata-production-wemm-annotation-draft-v1",
        "production_eligible": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "official_quality_status": "NOT_MEASURED",
        "label_space": {"kind": "OPEN_PROVISIONAL_PHRASES"},
        "metrics": {"camera_window_input_count": window_count * 6},
        "windows": windows,
    }


def test_builds_ten_window_style_queue_and_keeps_review_blank() -> None:
    plan = build_review_queue_batches(_draft(), batch_size=2, draft_path="draft.json")

    assert plan["summary"] == {
        "recording_count": 1,
        "window_count": 3,
        "camera_window_input_count": 18,
        "batch_count": 2,
        "batch_size_windows": 2,
        "last_batch_size": 1,
        "top_k6_count": 3,
        "review_boundary_status": {"NOT_MEASURED": 3},
        "qwen_placeholder_count": 3,
        "official_gold_status": "NOT_ESTABLISHED",
        "official_quality_status": "NOT_MEASURED",
    }
    batch = plan["batches"][0]
    item = batch["items"][0]
    assert len(item["wemm"]["top_k"]) == 6
    assert item["wemm"]["margin"] == 0.1
    assert item["qwen"]["status"] == "NOT_RUN"
    assert item["review"]["decision_options"] == [
        "accept",
        "edit",
        "split",
        "reject",
        "abstain",
    ]
    assert item["review"]["annotation"]["start_seconds"] is None
    assert item["review"]["annotation"]["boundary_status"] == "NOT_MEASURED"
    assert item["review_boundary"] == {
        "start_seconds": None,
        "end_seconds": None,
        "boundary_status": "NOT_MEASURED",
        "timestamp_basis": None,
    }
    assert validate_review_queue_batches(plan)["status"] == "VALID"


def test_rejects_non_six_top_k() -> None:
    draft = _draft(1)
    draft["windows"][0]["annotation_draft"]["segments"][0]["top_k"] = []  # type: ignore[index]
    try:
        build_review_queue_batches(draft)
    except ProductionReviewQueueBatchError as exc:
        assert "exactly 6" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("expected Top-K validation failure")
