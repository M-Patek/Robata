from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from robata.benchmark.production_structured_annotation import (
    build_structured_annotation_envelope,
)
from robata.benchmark.production_structured_evidence_verifier import (
    PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION,
    render_markdown,
    verify_production_structured_evidence,
)


def _segment(
    *,
    verb: str = "fold",
    noun: str = "garment",
    start: float | None = 1.0,
    end: float | None = 2.0,
    evidence: object = ("visible fold",),
    boundary_status: str = "MEASURED",
) -> dict[str, object]:
    return {
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": boundary_status,
        "structured_labels": {
            "verb": {"value": verb, "status": "MEASURED"},
            "noun": {"value": noun, "status": "MEASURED"},
            "attributes": {"value": None, "status": "NOT_OBSERVABLE"},
            "location": {"value": None, "status": "NOT_OBSERVABLE"},
            "hand": {"value": None, "status": "NOT_OBSERVABLE"},
        },
        "confidence": 0.8,
        "evidence": list(evidence) if isinstance(evidence, tuple) else evidence,
        "evidence_status": "MEASURED",
        "status": "MEASURED",
    }


def _envelope(
    segments: list[dict[str, object]],
    *,
    candidates: list[dict[str, object]] | None = None,
    candidate_marker: str | None = None,
) -> dict[str, object]:
    qwen_row: dict[str, object] = {
        "window_id": "w00",
        "ordinal": 0,
        "interval": [0.0, 4.0],
        "camera_id": "cam_01",
        "status": "SUCCEEDED",
        "segments": segments,
    }
    if candidates is not None:
        qwen_row["candidates"] = candidates
    if candidate_marker is not None:
        qwen_row["candidate_state"] = candidate_marker
    return build_structured_annotation_envelope(
        {
            "qwen": {
                "format": "robata-production-qwen-structured-native-shadow-v1",
                "source": {"manifest": "cohort.json", "camera_count": 1},
                "windows": [qwen_row],
            }
        },
        source_path="source.mcap",
        camera_count=1,
    )


def test_valid_structured_claim_is_reviewable_but_never_accepted() -> None:
    report = verify_production_structured_evidence(_envelope([_segment()]))
    assert report["format"] == PRODUCTION_STRUCTURED_EVIDENCE_VERIFIER_VERSION
    assert report["status"] == "REVIEW_REQUIRED"
    window = report["windows"][0]
    claim = window["structured_claims"][0]
    assert claim["source_bound_positive_interval"] is True
    assert claim["verb_measured"] is True
    assert claim["noun_measured"] is True
    assert claim["evidence_presence"] is True
    assert claim["eligible_for_review"] is True
    assert claim["accepted"] is False
    assert window["decision"] == "review"
    assert window["abstained"] is False
    assert report["quality"]["measurement_status"] == "NOT_MEASURED"
    assert report["controls"]["model_invoked"] is False


def test_invalid_claim_is_retained_with_explicit_reasons() -> None:
    invalid = _segment(start=3.0, end=5.0, evidence=[])
    invalid["structured_labels"]["verb"] = {"value": None, "status": "NOT_OBSERVABLE"}  # type: ignore[index]
    report = verify_production_structured_evidence(_envelope([invalid]))
    window = report["windows"][0]
    claim = window["structured_claims"][0]
    assert window["status"] == "ABSTAIN"
    assert claim["eligible_for_review"] is False
    assert claim["source_bound_positive_interval"] is False
    assert "BOUNDARY_OUT_OF_SOURCE" in claim["reason_codes"]
    assert "VERB_NOT_MEASURED" in claim["reason_codes"]
    assert "EVIDENCE_MISSING" in claim["reason_codes"]
    # The canonical envelope deliberately nulls an out-of-window boundary but
    # retains an explicit diagnostic marker; the verifier keeps that row out
    # of reviewable claims.
    assert claim["raw_claim"]["boundary_error"] == "SEGMENT_BOUNDARY_OUTSIDE_WINDOW"
    assert "NO_SOURCE_BOUND_STRUCTURED_CLAIMS" in window["abstention"]["reason_codes"]


def test_filler_verbs_warn_without_semantic_rewrite() -> None:
    report = verify_production_structured_evidence(_envelope([_segment(verb="reaches")]))
    claim = report["windows"][0]["structured_claims"][0]
    assert claim["eligible_for_review"] is True
    assert claim["filler_verb_warning"] == "FILLER_VERB_PRESENT:reaches"
    assert "FILLER_VERB_PRESENT:reaches" in claim["review_reason_codes"]
    assert claim["verb"] == "reaches"


def test_candidate_only_rows_are_not_structured_claims_and_empty_qwen_top_k_is_unmeasured() -> None:
    report = verify_production_structured_evidence(
        _envelope([], candidates=[{"verb": "fold", "noun": "garment"}])
    )
    window = report["windows"][0]
    assert window["candidate_state"] == "OBSERVED"
    assert window["candidate_only_claim_count"] == 1
    assert window["candidate_only_claims"][0]["claim_kind"] == "CANDIDATE_ONLY"
    assert window["candidate_only_claims"][0]["accepted"] is False
    assert window["status"] == "ABSTAIN"

    empty = verify_production_structured_evidence(_envelope([], candidates=[]))
    empty_window = empty["windows"][0]
    assert empty_window["candidate_state"] == "NOT_MEASURED"
    assert empty_window["candidate_top_k_observed"] is False

    observed_input = _envelope([], candidates=[])
    observed_input["windows"][0]["models"]["qwen"]["candidate_state"] = "OBSERVED_EMPTY"  # type: ignore[index]
    observed_empty = verify_production_structured_evidence(observed_input)
    assert observed_empty["windows"][0]["candidate_state"] == "OBSERVED_EMPTY"


def test_raw_qwen_prose_or_structured_response_is_not_invented_as_top_k() -> None:
    sidecar = {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "source": {"manifest": "cohort.json", "camera_count": 1},
        "windows": [
            {
                "window_id": "w00",
                "interval": [0.0, 4.0],
                "camera_id": "cam_01",
                "status": "SUCCEEDED",
                "raw_text": '{"segments": []}',
            }
        ],
    }
    report = verify_production_structured_evidence(sidecar)
    window = report["windows"][0]
    assert window["candidate_state"] == "NOT_MEASURED"
    assert window["candidate_only_claim_count"] == 0


def test_parse_failure_is_visible_even_when_model_runtime_succeeded() -> None:
    envelope = _envelope([])
    qwen = envelope["windows"][0]["models"]["qwen"]  # type: ignore[index]
    qwen["parse_observations"] = [  # type: ignore[index]
        {
            "camera_id": "cam_01",
            "parse_status": "INVALID",
            "errors": ["SEGMENT_BOUNDARY_OUTSIDE_WINDOW"],
            "warnings": [],
        }
    ]
    report = verify_production_structured_evidence(envelope)
    window = report["windows"][0]
    assert window["runtime_status"] == "SUCCEEDED"
    assert window["structured_status"] == "STRUCTURED_INVALID"
    assert "PARSE_INVALID" in window["reason_codes"]
    assert "PARSE_ERROR:SEGMENT_BOUNDARY_OUTSIDE_WINDOW" in window["reason_codes"]
    assert window["parse_diagnostics"][0]["parse_status"] == "INVALID"


def test_unsupported_timestamp_basis_cannot_be_source_bound() -> None:
    segment = _segment()
    segment["timestamp_basis"] = "camera_local_ticks"
    report = verify_production_structured_evidence(_envelope([segment]))
    claim = report["windows"][0]["structured_claims"][0]
    assert claim["eligible_for_review"] is False
    assert claim["source_bound_positive_interval"] is False
    assert "TIMESTAMP_BASIS_UNSUPPORTED" in claim["reason_codes"]


def test_direct_sidecar_accepts_media_locator_compatibility_keys() -> None:
    sidecar = {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {"media_path": "sample-medium.mcap", "camera_count": 1},
        "windows": [
            {
                "window_id": "w00",
                "ordinal": 0,
                "interval": [0.0, 4.0],
                "camera_id": "cam_01",
                "status": "SUCCEEDED",
                "segments": [_segment()],
            }
        ],
    }
    report = verify_production_structured_evidence(sidecar)
    assert report["source"]["path"] == "sample-medium.mcap"
    assert report["windows"][0]["reviewable_claim_count"] == 1


def test_cli_smoke() -> None:
    root = Path(".agent_tmp") / f"structured-evidence-verifier-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        source = root / "input.json"
        output = root / "report.json"
        output_md = root / "report.md"
        source.write_text(json.dumps(_envelope([_segment()])), encoding="utf-8")
        script = Path(__file__).parents[2] / "scripts" / "verify_production_structured_evidence.py"
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
        assert payload["status"] == "REVIEW_REQUIRED"
        assert "structured evidence" in output_md.read_text(encoding="utf-8")
        assert "NOT_MEASURED" in render_markdown(payload)
    finally:
        shutil.rmtree(root, ignore_errors=True)
