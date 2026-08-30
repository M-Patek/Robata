from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from robata.benchmark.production_annotation_projection import (
    ProductionAnnotationProjectionError,
    project_production_annotations,
    render_markdown,
)


def _segment(
    verb: str = "fold",
    noun: str = "garment",
    *,
    start: float = 1.0,
    end: float = 2.0,
    boundary_status: str = "MEASURED",
    evidence: object = ("visible fold",),
    confidence: float = 0.8,
    attributes: object = None,
    location: object = None,
    hand: object = None,
) -> dict[str, object]:
    return {
        "start_time_sec": start,
        "end_time_sec": end,
        "boundary_status": boundary_status,
        "structured_labels": {
            "verb": {"value": verb, "status": "MEASURED"},
            "noun": {"value": noun, "status": "MEASURED"},
            "attributes": {
                "value": attributes,
                "status": "MEASURED" if attributes is not None else "NOT_OBSERVABLE",
            },
            "location": {
                "value": location,
                "status": "MEASURED" if location is not None else "NOT_OBSERVABLE",
            },
            "hand": {
                "value": hand,
                "status": "MEASURED" if hand is not None else "NOT_OBSERVABLE",
            },
        },
        "confidence": confidence,
        "evidence": list(evidence) if isinstance(evidence, tuple) else evidence,
        "evidence_status": "MEASURED",
    }


def _qwen(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "windows": rows,
    }


def _row(
    camera: str,
    segments: list[dict[str, object]],
    *,
    interval: tuple[float, float] = (0.0, 4.0),
    parse_status: str = "PARSED",
) -> dict[str, object]:
    return {
        "window_id": "w00",
        "camera_id": camera,
        "interval": list(interval),
        "parsed_structured": {"parse_status": parse_status},
        "segments": segments,
    }


def _wemm() -> dict[str, object]:
    return {
        "format": "robata-production-wemm-shadow-v1",
        "windows": [
            {
                "window_id": "w00",
                "model": {
                    "status": "SUCCEEDED",
                    "predictions": [
                        {"rank": 1, "verb": "fold", "noun": "garment", "score": 0.9},
                        {"rank": 2, "verb": "smooth", "noun": "garment", "score": 0.8},
                    ],
                },
            }
        ],
    }


def test_source_bound_claims_are_deduplicated_and_chronologically_projected() -> None:
    report = project_production_annotations(
        _qwen(
            [
                _row("cam_02", [_segment(start=1.0, end=2.0)]),
                _row("cam_01", [_segment(start=1.2, end=2.2, confidence=0.9)]),
                _row("cam_03", [_segment("smooth", "garment", start=1.5, end=2.5)]),
            ]
        ),
        _wemm(),
    )
    window = report["windows"][0]
    candidates = window["annotation"]["segments"]
    assert window["annotation"]["status"] == "REVIEW_REQUIRED"
    assert candidates[0]["label_text"] == "fold garment"
    assert candidates[0]["camera_ids"] == ["cam_01", "cam_02"]
    assert candidates[0]["support_count"] == 2
    assert candidates[0]["status"] == "CONSENSUS"
    assert candidates[0]["review_required"] is True
    assert candidates[1]["label_text"] == "smooth garment"
    assert candidates[0]["annotation_order"] == 1
    assert report["metrics"]["conflict_window_count"] == 1
    assert report["parameters"]["label_order"] == "verb_then_noun"
    assert report["parameters"]["structured_pair_order"] == "verb_then_noun"
    assert report["parameters"]["label_text_order"] == ("verb attributes noun location with hand")


def test_duplicate_same_camera_claims_are_retained_but_one_candidate() -> None:
    report = project_production_annotations(
        _qwen(
            [
                _row(
                    "cam_01",
                    [_segment(start=1.0, end=2.0), _segment(start=1.1, end=2.1)],
                )
            ]
        )
    )
    candidate = report["windows"][0]["annotation"]["segments"][0]
    assert candidate["label_text"] == "fold garment"
    assert candidate["claim_count"] == 2
    assert candidate["support_count"] == 1
    assert report["metrics"]["raw_claim_count"] == 2


def test_optional_fields_are_projected_fieldwise_with_explicit_alternatives() -> None:
    report = project_production_annotations(
        _qwen(
            [
                _row(
                    "cam_01",
                    [_segment(attributes="blue", location="table", hand="left hand")],
                ),
                _row(
                    "cam_02",
                    [_segment(attributes="blue", location="counter", hand="left hand")],
                ),
            ]
        )
    )
    candidate = report["windows"][0]["annotation"]["segments"][0]
    labels = candidate["structured_labels"]
    assert labels["attributes"]["value"] == "blue"
    assert labels["attributes"]["status"] == "MEASURED"
    assert labels["location"]["value"] == "table"
    assert labels["location"]["status"] == "MEASURED"
    assert candidate["location"] == "table"
    # A bare location is preserved verbatim; adding ``at`` would infer a
    # spatial relation that was not part of the observation.
    assert candidate["label_text"] == "fold blue garment table with left hand"
    assert candidate["label_text_order"] == "verb attributes noun location with hand"
    assert candidate["field_conflicts"] == ["location"]
    assert len(candidate["field_alternatives"]["location"]) == 2
    assert candidate["review_required"] is True


def test_source_and_field_statuses_are_retained_separately_from_projection_status() -> None:
    segment = _segment(attributes=None, location=None, hand=None)
    segment["status"] = "MEASURED"
    row = _row("cam_01", [segment])
    row["timestamp_basis"] = "source_absolute_seconds"
    report = project_production_annotations(_qwen([row]))
    candidate = report["windows"][0]["annotation"]["segments"][0]
    assert candidate["status"] == "SINGLE_SOURCE"
    assert candidate["source_status"] == "MEASURED"
    assert candidate["principal_status"] == "MEASURED"
    assert candidate["source_statuses"] == ["MEASURED"]
    assert candidate["timestamp_basis"] == "source_absolute_seconds"
    assert candidate["timestamp_basis_status"] == "MEASURED"
    assert candidate["evidence_status"] == "MEASURED"
    assert candidate["field_statuses"] == {
        "attributes": "NOT_OBSERVABLE",
        "location": "NOT_OBSERVABLE",
        "hand": "NOT_OBSERVABLE",
    }
    raw_claim = report["windows"][0]["raw_claims"][0]
    assert raw_claim["source_status"] == "MEASURED"
    assert raw_claim["timestamp_basis"] == "source_absolute_seconds"


def test_unsupported_timestamp_basis_is_retained_but_not_projected() -> None:
    row = _row("cam_01", [_segment()])
    row["timestamp_basis"] = "window_relative_seconds"
    report = project_production_annotations(_qwen([row]))
    window = report["windows"][0]
    assert window["annotation"]["segments"] == []
    assert window["abstention"]["abstained"] is True
    assert "TIMESTAMP_BASIS_UNSUPPORTED" in window["abstention"]["reason_codes"]
    claim = window["raw_claims"][0]
    assert claim["timestamp_basis"] == "window_relative_seconds"
    assert claim["timestamp_basis_status"] == "UNSUPPORTED"
    assert claim["valid_source_bound"] is False


def test_invalid_or_out_of_window_claims_are_not_shifted_and_cause_abstention() -> None:
    report = project_production_annotations(
        _qwen(
            [
                _row("cam_01", [_segment(start=-1.0, end=1.0)]),
                _row("cam_02", [_segment(start=4.0, end=5.0)]),
                _row("cam_03", [_segment(start=1.0, end=2.0, boundary_status="NOT_MEASURED")]),
                _row("cam_04", [_segment(start=1.0, end=2.0, evidence=[])]),
            ]
        )
    )
    window = report["windows"][0]
    assert window["annotation"]["segments"] == []
    assert window["abstention"]["abstained"] is True
    assert "BOUNDARY_OUT_OF_SOURCE" in window["abstention"]["reason_codes"]
    assert report["metrics"]["invalid_claim_count"] == 4
    intervals = [claim["interval"] for claim in window["raw_claims"]]
    assert intervals[1] == [4.0, 5.0]


def test_missing_evidence_and_parse_invalid_claims_are_retained_not_projected() -> None:
    report = project_production_annotations(
        _qwen(
            [
                _row("cam_01", [_segment(evidence=[])]),
                _row("cam_02", [_segment()], parse_status="INVALID"),
            ]
        )
    )
    claims = report["windows"][0]["raw_claims"]
    assert len(claims) == 2
    assert all(not claim["valid_source_bound"] for claim in claims)
    assert "EVIDENCE_MISSING" in claims[0]["reasons"]
    assert "PARSE_INVALID" in claims[1]["reasons"]
    assert report["windows"][0]["annotation"]["status"] == "ABSTAIN"


def test_wemm_top_k_is_context_only_and_raw_provenance_is_copied() -> None:
    wemm = _wemm()
    report = project_production_annotations(_qwen([_row("cam_01", [_segment()])]), wemm, top_k=1)
    projection = report["windows"][0]["wemm"]
    assert len(projection["top_k"]) == 1
    assert len(projection["raw_provenance"]["model"]["predictions"]) == 2
    assert report["controls"]["wemm_used_as_annotation_evidence"] is False
    wemm["windows"][0]["model"]["predictions"][0]["score"] = 0.1  # type: ignore[index]
    assert projection["top_k"][0]["score"] == 0.9


def test_label_text_uses_morphology_only_and_report_is_surrogate() -> None:
    report = project_production_annotations(
        _qwen([_row("cam_01", [_segment("flatten", "clothes")])])
    )
    segment = report["windows"][0]["annotation"]["segments"][0]
    assert segment["label_text"] == "flatten clothes"
    assert report["status"] == "SURROGATE_ONLY"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["quality_claim"] is False
    assert report["production_eligible"] is False
    assert all(
        value is False
        for key, value in report["controls"].items()
        if key not in {"raw_claims_preserved"}
    )


def test_envelope_falls_back_to_canonical_source_bound_segments() -> None:
    envelope = {
        "format": "robata-production-structured-annotation-envelope-v1",
        "windows": [
            {
                "window_id": "w00",
                "start_time_sec": 0.0,
                "end_time_sec": 4.0,
                "models": {
                    "qwen": {
                        "segments": [_segment()],
                        "candidate_sources": [
                            {
                                "camera_id": "cam_01",
                                "raw_text": (
                                    '{"segments": [{"start_time_sec": 1, "end_time_sec": 2}]}'
                                ),
                            },
                        ],
                    }
                },
            }
        ],
    }
    report = project_production_annotations(envelope)
    assert report["windows"][0]["annotation"]["segments"][0]["label_text"] == "fold garment"
    assert report["windows"][0]["annotation"]["segments"][0]["camera_ids"] == []


def test_invalid_configuration_is_rejected() -> None:
    try:
        project_production_annotations(_qwen([]), top_k=0)
    except ProductionAnnotationProjectionError as exc:
        assert "top_k" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid top_k")


def test_cli_smoke(tmp_path: Path) -> None:
    def reserve(name: str) -> Path:
        fd, raw = tempfile.mkstemp(prefix=f"annotation-projection-{name}-", dir=tmp_path)
        os.close(fd)
        Path(raw).unlink()
        return Path(raw)

    qwen_path, wemm_path, output = (reserve(name) for name in ("qwen", "wemm", "out"))
    output_md = output.with_suffix(".md")
    try:
        qwen_path.write_text(json.dumps(_qwen([_row("cam_01", [_segment()])])), encoding="utf-8")
        wemm_path.write_text(json.dumps(_wemm()), encoding="utf-8")
        script = (
            Path(__file__).parents[2] / "scripts" / "project_production_annotation_projection.py"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--qwen",
                str(qwen_path),
                "--wemm",
                str(wemm_path),
                "--output-json",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["status"] == "SURROGATE_ONLY"
        assert output_md.exists()
        assert "SURROGATE_ONLY" in render_markdown(payload)
    finally:
        for path in (qwen_path, wemm_path, output, output_md):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
