from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

from robata.benchmark.production_annotation_handoff import (
    PRODUCTION_ANNOTATION_HANDOFF_VERSION,
    build_production_annotation_handoff,
)
from robata.benchmark.production_structured_annotation import (
    build_structured_annotation_envelope,
)


def _segment(
    *,
    start: float = 1.0,
    end: float = 2.0,
    verb: str = "fold",
    noun: str = "garment",
    evidence: object = ("visible fold",),
) -> dict[str, object]:
    return {
        "start_time_sec": start,
        "end_time_sec": end,
        "structured_labels": {
            "verb": verb,
            "noun": noun,
            "attributes": "blue",
            "location": "on table",
            "hand": "right hand",
        },
        "confidence": 0.8,
        "evidence": list(evidence) if isinstance(evidence, tuple) else evidence,
    }


def _envelope() -> dict[str, object]:
    return build_structured_annotation_envelope(
        {
            "wemm": {
                "format": "robata-production-wemm-shadow-v1",
                "source": {"path": "source.mcap", "camera_count": 1},
                "windows": [
                    {
                        "window_id": "w00",
                        "ordinal": 0,
                        "start_seconds": 0.0,
                        "end_seconds": 4.0,
                        "model": {
                            "status": "SUCCEEDED",
                            "predictions": [
                                {"rank": 1, "verb": "fold", "noun": "garment", "score": 0.91},
                                {"rank": 2, "verb": "smooth", "noun": "garment", "score": 0.8},
                            ],
                        },
                    }
                ],
            },
            "qwen": {
                "format": "robata-production-qwen-structured-native-shadow-v1",
                "source": {"manifest": "cohort.json", "camera_count": 1},
                "windows": [
                    {
                        "window_id": "w00",
                        "ordinal": 0,
                        "interval": [0.0, 4.0],
                        "camera_id": "cam_01",
                        "status": "SUCCEEDED",
                        "segments": [
                            _segment(),
                            _segment(start=3.0, end=5.0, verb="move", evidence=[]),
                        ],
                    }
                ],
            },
        },
        source_path="source.mcap",
        window_specs=[{"window_id": "w00", "ordinal": 0, "start_seconds": 0.0, "end_seconds": 4.0}],
        camera_count=1,
    )


def test_handoff_separates_reviewable_candidates_and_rejected_claims() -> None:
    report = build_production_annotation_handoff(_envelope())
    assert report["format"] == PRODUCTION_ANNOTATION_HANDOFF_VERSION
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["official_gold_status"] == "NOT_ESTABLISHED"
    assert report["quality_claim"] is False
    assert report["human_adjudication"] == "NOT_PERFORMED"
    assert report["production_eligible"] is False
    assert report["automatic_eligible"] is False
    assert report["automatic_qualification"] is False
    assert report["contract"]["automatic_eligible_always_false"] is True
    assert report["quality"]["measurement_status"] == "NOT_MEASURED"
    window = report["windows"][0]
    assert window["status"] == "REVIEW_REQUIRED"
    assert len(window["annotation_candidates"]) == 1
    candidate = window["annotation_candidates"][0]
    assert candidate["status"] == "PENDING_HUMAN_REVIEW"
    assert candidate["automatic_eligible"] is False
    assert candidate["label_text"] == "fold blue garment on table with right hand"
    assert candidate["accepted"] is False
    assert len(window["rejected_claims"]) == 1
    assert "BOUNDARY_OUT_OF_SOURCE" in window["rejected_claims"][0]["reason_codes"]
    assert window["model_context"]["wemm"]["top_k"][0]["verb"] == "fold"
    assert window["decision"] == "pending"
    assert report["controls"]["gold_written"] is False


def test_handoff_abstains_when_no_structurally_reviewable_claim_exists() -> None:
    envelope = _envelope()
    envelope["windows"][0]["models"]["qwen"]["segments"] = []  # type: ignore[index]
    envelope["windows"][0]["models"]["qwen"]["measurement_status"] = "NOT_MEASURED"  # type: ignore[index]
    report = build_production_annotation_handoff(envelope)
    window = report["windows"][0]
    assert window["status"] == "ABSTAIN"
    assert window["annotation_candidates"] == []
    assert window["model_context"]["wemm"]["top_k"]
    assert report["metrics"]["abstained_window_count"] == 1


def test_handoff_cli_smoke() -> None:
    root = Path(".agent_tmp") / f"annotation-handoff-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        source = root / "envelope.json"
        output = root / "handoff.json"
        output_md = root / "handoff.md"
        source.write_text(json.dumps(_envelope()), encoding="utf-8")
        script = Path(__file__).parents[2] / "scripts" / "build_production_annotation_handoff.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                str(source),
                "--output",
                str(output),
                "--output-md",
                str(output_md),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["format"] == PRODUCTION_ANNOTATION_HANDOFF_VERSION
        assert "review-only" in output_md.read_text(encoding="utf-8")
    finally:
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
