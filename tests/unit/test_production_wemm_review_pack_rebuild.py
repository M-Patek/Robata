from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_preannotation import build_preannotation_envelope
from robata.benchmark.production_wemm_review_pack_rebuild import (
    REBUILD_FORMAT,
    ProductionWemmReviewPackRebuildError,
    rebuild_production_wemm_review_packs,
    render_markdown,
)


def _preannotation(recording_id: str) -> dict[str, object]:
    return build_preannotation_envelope(
        {
            "recording_id": recording_id,
            "path": f"{recording_id}.mcap",
            "archive_member": f"file/{recording_id}.mcap",
            "archive_path": "source.zip",
            "camera_count": 6,
            "source_preflight_status": "PASS",
            "qa_status": "PENDING",
        },
        [
            {
                "window_id": f"{recording_id}-w0000",
                "ordinal": 0,
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "camera_ids": [f"cam_{index:02d}" for index in range(1, 7)],
                "proposals": [
                    {
                        "label_text": "pick up garment",
                        "structured_labels": {"verb": "pick up", "noun": "garment"},
                        "top_k": [
                            {"label_text": "pick up garment", "score": 0.8},
                            {"label_text": "fold garment", "score": 0.7},
                        ],
                    }
                ],
            }
        ],
        raw_model_output={
            "catalog": {
                "format": "robata-production-open-phrase-catalog-v1",
                "phrase_count": 6,
                "epic_ontology_used": False,
                "mapper_used": False,
                "provisional": True,
            }
        },
        model={"name": "WeMM-Embedding-2B", "route": "video_embedding"},
        model_invoked=True,
    )


def _write_checkpoint(root: Path) -> Path:
    pre = root / "preannotations"
    pre.mkdir(parents=True)
    for recording_id in ("rec-complete", "rec-running", "rec-planned", "rec-failed"):
        (pre / f"{recording_id}.json").write_text(
            json.dumps(_preannotation(recording_id)), encoding="utf-8"
        )
    checkpoint = {
        "format": "robata-production-wemm-batch-run-v1",
        "status": "RUNNING",
        "items": [
            {
                "ordinal": 1,
                "status": "COMPLETE",
                "preannotation_path": "preannotations/rec-complete.json",
            },
            {
                "ordinal": 2,
                "status": "RUNNING",
                "preannotation_path": "preannotations/rec-running.json",
            },
            {
                "ordinal": 3,
                "status": "PLANNED",
                "preannotation_path": "preannotations/rec-planned.json",
            },
            {
                "ordinal": 4,
                "status": "FAILED",
                "preannotation_path": "preannotations/rec-failed.json",
            },
        ],
    }
    path = root / "batch-run.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    return path


def test_rebuilds_only_complete_preannotations_to_independent_directory() -> None:
    with tempfile.TemporaryDirectory(prefix="robata-wemm-rebuild-") as temp_dir:
        root = Path(temp_dir) / "batch"
        checkpoint = _write_checkpoint(root)
        output = Path(temp_dir) / "rebuilt-review"

        report = rebuild_production_wemm_review_packs(checkpoint, output)

        assert report["format"] == REBUILD_FORMAT
        assert report["status"] == "PARTIAL"
        assert report["summary"]["written_count"] == 1  # type: ignore[index]
        assert report["summary"]["unresolved_item_count"] == 3  # type: ignore[index]
        assert report["controls"]["model_invoked"] is False  # type: ignore[index]
        assert report["controls"]["gold_written"] is False  # type: ignore[index]
        reasons = " ".join(row["reason"] for row in report["rejected_inputs"])  # type: ignore[index]
        assert "STATUS_NOT_COMPLETE:RUNNING" in reasons
        assert "STATUS_NOT_COMPLETE:PLANNED" in reasons
        assert "STATUS_NOT_COMPLETE:FAILED" in reasons

        rebuilt = json.loads(Path(report["written"][0]["path"]).read_text(encoding="utf-8"))  # type: ignore[index]
        assert rebuilt["source"]["recording_id"] == "rec-complete"
        assert rebuilt["model"]["name"] == "WeMM-Embedding-2B"
        assert rebuilt["model_artifact"]["catalog"]["phrase_count"] == 6
        assert rebuilt["items"][0]["proposals"][0]["margin"] == pytest.approx(0.1)
        assert rebuilt["rebuild_provenance"]["source_status"] == "COMPLETE"
        assert rebuilt["rebuild_provenance"]["recording_filename"] == "rec-complete.mcap"
        assert "gold" not in rebuilt
        assert "COMPLETE items" in render_markdown(report)


def test_existing_rebuilt_file_is_not_overwritten_by_default() -> None:
    with tempfile.TemporaryDirectory(prefix="robata-wemm-rebuild-") as temp_dir:
        root = Path(temp_dir) / "batch"
        checkpoint = _write_checkpoint(root)
        output = Path(temp_dir) / "rebuilt-review"
        first = rebuild_production_wemm_review_packs(checkpoint, output)
        output_path = Path(first["written"][0]["path"])  # type: ignore[index]
        original = output_path.read_text(encoding="utf-8")

        second = rebuild_production_wemm_review_packs(checkpoint, output)

        assert second["summary"]["written_count"] == 0  # type: ignore[index]
        assert second["summary"]["skipped_existing_count"] == 1  # type: ignore[index]
        assert output_path.read_text(encoding="utf-8") == original
        assert second["controls"]["runner_outputs_overwritten"] is False  # type: ignore[index]


def test_rejects_output_in_runner_review_directory() -> None:
    with tempfile.TemporaryDirectory(prefix="robata-wemm-rebuild-") as temp_dir:
        root = Path(temp_dir) / "batch"
        checkpoint = _write_checkpoint(root)
        with pytest.raises(ProductionWemmReviewPackRebuildError, match="must not be the runner"):
            rebuild_production_wemm_review_packs(checkpoint, root / "review")


def test_non_checkpoint_input_is_reported_without_opening_sidecar_as_runner_item() -> None:
    with tempfile.TemporaryDirectory(prefix="robata-wemm-rebuild-") as temp_dir:
        sidecar = Path(temp_dir) / "preannotation.json"
        sidecar.write_text(json.dumps(_preannotation("rec")), encoding="utf-8")
        report = rebuild_production_wemm_review_packs(sidecar, Path(temp_dir) / "output")
        assert report["status"] == "NO_COMPLETE_ITEMS"
        assert report["summary"]["written_count"] == 0  # type: ignore[index]
        assert report["rejected_inputs"][0]["reason"] == "ROOT_NOT_BATCH_CHECKPOINT"  # type: ignore[index]
