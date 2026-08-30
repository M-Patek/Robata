from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.production_structured_annotation import (
    QWEN_STRUCTURED_NATIVE_PROMPT,
    STRUCTURED_ANNOTATION_ENVELOPE_VERSION,
    TIMESTAMP_BASIS,
    TIMESTAMP_MAPPING_VERSION,
    WINDOW_RELATIVE_TIMESTAMP_BASIS,
    ProductionStructuredAnnotationError,
    build_structured_annotation_envelope,
    map_qwen_relative_timestamps,
    normalize_structured_annotation_envelope,
    parse_qwen_structured_output,
)


def _wemm(*, segments: list[dict[str, object]] | None = None) -> dict[str, object]:
    prediction = {
        "rank": 1,
        "action_key": [1, 2],
        "verb": "open",
        "noun": "drawer",
        "score": 0.91,
    }
    row: dict[str, object] = {
        "ordinal": 0,
        "window_id": "w00",
        "start_seconds": 0.0,
        "end_seconds": 4.0,
        "model": {
            "status": "SUCCEEDED",
            "predictions": [prediction],
        },
    }
    if segments is not None:
        row["model"]["segments"] = segments  # type: ignore[index]
    return {
        "format": "robata-production-wemm-shadow-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {"path": "sample.mcap", "camera_count": 6},
        "windows": [row],
    }


def _qwen() -> dict[str, object]:
    return {
        "format": "robata-production-qwen-shadow-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {"manifest": "cohort.json", "video_root": "video"},
        "camera_ids": ["cam_01"],
        "windows": [
            {
                "window_id": "w00",
                "interval": [0.0, 4.2],
                "camera_id": "cam_01",
                "status": "SUCCEEDED",
                "raw_text": '{"verb":"opens","noun":"drawer","confidence":0.7}',
            }
        ],
        "controls": {
            "model_invoked": True,
            "gold_included": False,
            "predictions_are_gold": False,
        },
    }


def test_build_preserves_top_k_and_explicitly_blocks_missing_mage() -> None:
    envelope = build_structured_annotation_envelope({"wemm": _wemm(), "qwen": _qwen()})

    assert envelope["format"] == STRUCTURED_ANNOTATION_ENVELOPE_VERSION
    window = envelope["windows"][0]
    assert window["start_time_sec"] == 0.0
    assert window["end_time_sec"] == 4.0
    assert window["models"]["wemm"]["candidates"] == [
        {"rank": 1, "action_key": [1, 2], "verb": "open", "noun": "drawer", "score": 0.91}
    ]
    assert window["models"]["qwen"]["candidates"][0]["verb"] == "opens"
    assert window["models"]["mage"]["status"] == "BLOCKED"
    assert window["models"]["mage"]["candidates"] == []


def test_missing_fields_have_status_and_window_is_not_segment_boundary() -> None:
    envelope = build_structured_annotation_envelope({"wemm": _wemm()})
    wemm = envelope["windows"][0]["models"]["wemm"]
    assert wemm["segments"] == []
    assert wemm["measurement_status"] == "NOT_MEASURED"

    segment_source = {
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "structured_labels": {
            "verb": "open",
            "noun": "drawer",
            "attributes": None,
        },
    }
    envelope = build_structured_annotation_envelope({"wemm": _wemm(segments=[segment_source])})
    segment = envelope["windows"][0]["models"]["wemm"]["segments"][0]
    assert segment["start_time_sec"] == 1.0
    assert segment["end_time_sec"] == 2.0
    assert segment["structured_labels"]["verb"] == {"value": "open", "status": "MEASURED"}
    assert segment["structured_labels"]["attributes"] == {
        "value": None,
        "status": "NOT_OBSERVABLE",
    }
    assert segment["structured_labels"]["location"]["status"] == "NOT_MEASURED"


def test_fixed_window_cannot_be_used_as_segment_boundary() -> None:
    envelope = build_structured_annotation_envelope({"wemm": _wemm()})
    assert envelope["windows"][0]["models"]["wemm"]["segments"] == []


def test_interval_mismatch_from_canonical_sources_fails() -> None:
    qwen = _qwen()
    qwen["windows"][0]["start_seconds"] = 8.0  # type: ignore[index]
    qwen["windows"][0]["end_seconds"] = 12.0  # type: ignore[index]
    with pytest.raises(ProductionStructuredAnnotationError, match="interval"):
        build_structured_annotation_envelope({"wemm": _wemm(), "qwen": qwen})


def test_gold_and_review_data_are_not_accepted() -> None:
    sidecar = _wemm()
    broken = deepcopy(sidecar)
    broken["review"] = {"segments": []}
    with pytest.raises(ProductionStructuredAnnotationError, match=r"gold|review|annotation"):
        build_structured_annotation_envelope({"wemm": broken})


@pytest.mark.parametrize(
    ("container", "flag"),
    [
        ("label_space", "epic_ontology_used"),
        ("controls", "mapper_used"),
    ],
)
def test_explicit_epic_or_mapper_use_is_benchmark_only(container: str, flag: str) -> None:
    sidecar = _wemm()
    sidecar[container] = {flag: True}

    with pytest.raises(
        ProductionStructuredAnnotationError,
        match=r"benchmark-only|EPIC|Mapper",
    ):
        build_structured_annotation_envelope({"wemm": sidecar})


def test_explicit_epic_format_is_benchmark_only() -> None:
    sidecar = _wemm()
    sidecar["format"] = "epic-kitchens-100-sidecar-v1"

    with pytest.raises(
        ProductionStructuredAnnotationError,
        match=r"benchmark-only|EPIC",
    ):
        build_structured_annotation_envelope({"wemm": sidecar})


def test_explicit_false_epic_mapper_controls_preserve_production_fixture() -> None:
    sidecar = _wemm()
    sidecar["label_space"] = {
        "kind": "OPEN_PROVISIONAL_PHRASES",
        "epic_ontology_used": False,
        "mapper_used": False,
    }
    sidecar["controls"] = {
        "epic_ontology_used": False,
        "mapper_used": False,
    }

    envelope = build_structured_annotation_envelope({"wemm": sidecar})
    assert envelope["windows"][0]["models"]["wemm"]["status"] == "SUCCEEDED"


def test_round_trip_normalizer_is_independent() -> None:
    envelope = build_structured_annotation_envelope({"wemm": _wemm()})
    copy_value = normalize_structured_annotation_envelope(envelope)
    assert copy_value == envelope
    copy_value["windows"][0]["models"]["wemm"]["candidates"][0]["score"] = 0.1
    assert envelope["windows"][0]["models"]["wemm"]["candidates"][0]["score"] == 0.91


def test_qwen_structured_parser_keeps_missing_statuses_and_rejects_prose() -> None:
    parsed = parse_qwen_structured_output(
        '{"segments":[{"start_time_sec":1.0,"end_time_sec":2.0,'
        '"structured_labels":{"verb":"open","noun":"drawer",'
        '"attributes":null,"location":null,"hand":null},'
        '"confidence":0.8,"evidence":["door moves"]}]}'
    )
    assert parsed["parse_status"] == "PARSED"
    segment = parsed["segments"][0]
    assert segment["structured_labels"]["verb"]["status"] == "MEASURED"
    assert segment["structured_labels"]["attributes"]["status"] == "NOT_OBSERVABLE"
    assert QWEN_STRUCTURED_NATIVE_PROMPT.startswith("Review the complete bounded native video")

    invalid = parse_qwen_structured_output("The person opens the drawer.")
    assert invalid["parse_status"] == "INVALID"
    assert invalid["segments"] == []


def test_qwen_structured_parser_does_not_accept_duplicate_keys_or_extra_root_fields() -> None:
    duplicate = parse_qwen_structured_output('{"segments":[],"segments":[]}')
    assert duplicate["parse_status"] == "INVALID"
    extra = parse_qwen_structured_output('{"segments":[],"verb":"open"}')
    assert extra["parse_status"] == "INVALID"


def test_candidate_and_prediction_sources_are_both_retained() -> None:
    sidecar = _wemm()
    model = sidecar["windows"][0]["model"]  # type: ignore[index]
    model["candidates"] = [{"label": "open drawer", "score": 0.9}]  # type: ignore[index]
    envelope = build_structured_annotation_envelope({"wemm": sidecar})
    section = envelope["windows"][0]["models"]["wemm"]
    assert section["candidates"] == [{"label": "open drawer", "score": 0.9}]
    assert section["candidate_groups"] == [
        {
            "source_field": "candidates",
            "candidates": [{"label": "open drawer", "score": 0.9}],
        },
        {
            "source_field": "predictions",
            "candidates": [
                {
                    "rank": 1,
                    "action_key": [1, 2],
                    "verb": "open",
                    "noun": "drawer",
                    "score": 0.91,
                }
            ],
        },
    ]


def test_envelope_records_timestamp_basis_and_marks_out_of_window_boundary() -> None:
    sidecar = _wemm(
        segments=[
            {
                "start_time_sec": 1.0,
                "end_time_sec": 5.0,
                "structured_labels": {"verb": "open", "noun": "drawer"},
            }
        ]
    )
    envelope = build_structured_annotation_envelope({"wemm": sidecar})
    assert envelope["contract"]["timestamp_basis"] == "source_absolute_seconds"
    assert envelope["windows"][0]["timestamp_basis"] == "source_absolute_seconds"
    segment = envelope["windows"][0]["models"]["wemm"]["segments"][0]
    assert segment["boundary_status"] == "NOT_MEASURED"
    assert segment["start_time_sec"] is None
    assert segment["status"] == "FAILED"
    assert segment["boundary_error"] == "SEGMENT_BOUNDARY_OUTSIDE_WINDOW"


def test_normalizer_rejects_unknown_timestamp_basis() -> None:
    envelope = build_structured_annotation_envelope({"wemm": _wemm()})
    envelope["contract"]["timestamp_basis"] = "window_relative_seconds"  # type: ignore[index]
    with pytest.raises(ProductionStructuredAnnotationError, match="timestamp_basis"):
        normalize_structured_annotation_envelope(envelope)


def test_parser_rejects_partial_boundaries_and_conflicting_aliases() -> None:
    partial = parse_qwen_structured_output(
        '{"segments":[{"start_time_sec":1.0,"structured_labels":{"verb":"open","noun":"drawer"}}]}'
    )
    assert partial["parse_status"] == "INVALID"
    assert any("PARTIAL_BOUNDARY" in error for error in partial["errors"])

    timestamp_conflict = parse_qwen_structured_output(
        '{"segments":[{"start_time_sec":1.0,"start_seconds":2.0,'
        '"end_time_sec":3.0,"structured_labels":{"verb":"open",'
        '"noun":"drawer"}}]}'
    )
    assert timestamp_conflict["parse_status"] == "INVALID"
    assert any("ALIAS_CONFLICT" in error for error in timestamp_conflict["errors"])

    labels_conflict = parse_qwen_structured_output(
        '{"segments":[{"structured_labels":{"verb":"open",'
        '"noun":"drawer"},"labels":{"verb":"close",'
        '"noun":"drawer"}}]}'
    )
    assert labels_conflict["parse_status"] == "INVALID"
    assert any("ALIAS_CONFLICT" in error for error in labels_conflict["errors"])

    labels_missing_key = parse_qwen_structured_output(
        '{"segments":[{"structured_labels":{"verb":"open",'
        '"noun":"drawer"},"labels":{"verb":"open"}}]}'
    )
    assert labels_missing_key["parse_status"] == "INVALID"
    assert any("ALIAS_CONFLICT" in error for error in labels_missing_key["errors"])

    equivalent_aliases = parse_qwen_structured_output(
        '{"segments":[{"structured_labels":{"verb":"open",'
        '"noun":"drawer"},"labels":{"verb":{"value":"open",'
        '"status":"MEASURED"},"noun":{"value":"drawer",'
        '"status":"MEASURED"}}}]}'
    )
    assert equivalent_aliases["parse_status"] == "PARSED"


def test_parser_accepts_explicit_relative_timestamp_basis() -> None:
    parsed = parse_qwen_structured_output(
        '{"timestamp_basis":"window_relative_seconds","segments":['
        '{"start_time_sec":0.5,"end_time_sec":1.5,'
        '"structured_labels":{"verb":"open","noun":"drawer"}}]}'
    )
    assert parsed["parse_status"] == "PARSED"
    assert parsed["timestamp_basis"] == WINDOW_RELATIVE_TIMESTAMP_BASIS
    assert parsed["timestamp_basis_status"] == "MEASURED"
    assert parsed["timestamp_basis_explicit"] is True
    assert parsed["segments"][0]["start_time_sec"] == 0.5

    legacy = parse_qwen_structured_output('{"segments":[]}')
    assert legacy["timestamp_basis"] == TIMESTAMP_BASIS
    assert legacy["timestamp_basis_explicit"] is False


def test_relative_timestamp_mapping_preserves_raw_and_mapped_values() -> None:
    parsed = parse_qwen_structured_output(
        '{"timestamp_basis":"window_relative_seconds","segments":['
        '{"start_time_sec":0.5,"end_time_sec":1.5,'
        '"structured_labels":{"verb":"open","noun":"drawer"},'
        '"evidence":["drawer moves"]}]}'
    )
    mapped = map_qwen_relative_timestamps(
        parsed,
        window_start_seconds=4.0,
        window_end_seconds=8.0,
    )
    assert mapped["timestamp_basis"] == TIMESTAMP_BASIS
    assert mapped["timestamp_basis_status"] == "MEASURED"
    assert mapped["timestamp_mapping_status"] == "MAPPED"
    assert mapped["timestamp_mapping"]["version"] == TIMESTAMP_MAPPING_VERSION
    segment = mapped["segments"][0]
    assert segment["start_time_sec"] == 4.5
    assert segment["end_time_sec"] == 5.5
    assert segment["mapped_start_time_sec"] == 4.5
    assert segment["mapped_end_time_sec"] == 5.5
    assert segment["raw_start_time_sec"] == 0.5
    assert segment["raw_end_time_sec"] == 1.5
    assert segment["raw_timestamp_basis"] == WINDOW_RELATIVE_TIMESTAMP_BASIS
    # The parser result is not mutated, so callers can retain the raw parsed
    # representation alongside the mapped sidecar.
    assert parsed["segments"][0]["start_time_sec"] == 0.5


def test_source_absolute_mapping_is_an_unchanged_deep_copy() -> None:
    parsed = parse_qwen_structured_output(
        '{"segments":[{"start_time_sec":4.5,"end_time_sec":5.5,'
        '"structured_labels":{"verb":"open","noun":"drawer"}}]}'
    )
    mapped = map_qwen_relative_timestamps(
        parsed,
        window_start_seconds=4.0,
        window_end_seconds=8.0,
    )
    assert mapped == parsed
    assert mapped is not parsed
    mapped["segments"][0]["start_time_sec"] = 99.0
    assert parsed["segments"][0]["start_time_sec"] == 4.5


def test_relative_timestamp_mapping_rejects_out_of_window_offsets() -> None:
    parsed = parse_qwen_structured_output(
        '{"timestamp_basis":"window_relative_seconds","segments":['
        '{"start_time_sec":3.5,"end_time_sec":4.5,'
        '"structured_labels":{"verb":"open","noun":"drawer"}}]}'
    )
    with pytest.raises(
        ProductionStructuredAnnotationError,
        match="RELATIVE_BOUNDARY_OUT_OF_WINDOW",
    ):
        map_qwen_relative_timestamps(
            parsed,
            window_start_seconds=4.0,
            window_end_seconds=8.0,
        )


def test_relative_mapping_metadata_survives_envelope_build() -> None:
    parsed = parse_qwen_structured_output(
        '{"timestamp_basis":"window_relative_seconds","segments":['
        '{"start_time_sec":0.5,"end_time_sec":1.5,'
        '"structured_labels":{"verb":"open","noun":"drawer"}}]}'
    )
    mapped = map_qwen_relative_timestamps(
        parsed,
        window_start_seconds=4.0,
        window_end_seconds=8.0,
    )
    qwen = {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "source": {"manifest": "cohort.json", "video_root": "video"},
        "windows": [
            {
                "window_id": "w00",
                "interval": [4.0, 8.0],
                "camera_id": "cam_01",
                "status": "SUCCEEDED",
                "timestamp_basis": TIMESTAMP_BASIS,
                "parsed_structured": mapped,
                "segments": mapped["segments"],
            }
        ],
    }
    envelope = build_structured_annotation_envelope(
        {"qwen": qwen},
        source_path="source.mcap",
        window_specs=[
            {
                "ordinal": 0,
                "window_id": "w00",
                "start_seconds": 4.0,
                "end_seconds": 8.0,
            }
        ],
        camera_count=1,
    )
    segment = envelope["windows"][0]["models"]["qwen"]["segments"][0]
    assert segment["start_time_sec"] == 4.5
    assert segment["raw_start_time_sec"] == 0.5
    assert segment["mapped_end_time_sec"] == 5.5
    assert segment["timestamp_mapping_status"] == "MEASURED"


def test_parser_warnings_and_observations_survive_envelope_normalization() -> None:
    sidecar = _qwen()
    row = sidecar["windows"][0]  # type: ignore[index]
    row["segments"] = [  # type: ignore[index]
        {
            "structured_labels": {"verb": "open", "noun": "drawer"},
            "confidence": 0.8,
        }
    ]
    row["parsed_structured"] = {  # type: ignore[index]
        "parse_status": "PARSED",
        "segments": row["segments"],  # type: ignore[index]
        "errors": [],
        "warnings": ["ROOT_ARRAY_COMPAT", "FILLER_VERB_PRESENT:reaches"],
    }
    row["generation_warnings"] = ["MAX_NEW_TOKENS_REACHED"]  # type: ignore[index]
    envelope = build_structured_annotation_envelope(
        {"qwen": sidecar},
        source_path="sample.mcap",
        window_specs=[
            {
                "ordinal": 0,
                "window_id": "w00",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
            }
        ],
        camera_count=1,
    )
    section = envelope["windows"][0]["models"]["qwen"]
    assert section["candidate_groups"]
    assert section["parse_observations"][0]["warnings"] == [
        "ROOT_ARRAY_COMPAT",
        "FILLER_VERB_PRESENT:reaches",
    ]
    assert section["parse_observations"][0]["generation_warnings"] == ["MAX_NEW_TOKENS_REACHED"]
    assert section["warnings"] == [
        "ROOT_ARRAY_COMPAT",
        "FILLER_VERB_PRESENT:reaches",
        "MAX_NEW_TOKENS_REACHED",
    ]
