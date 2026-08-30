from __future__ import annotations

import json

import pytest

from robata.benchmark.production_wemm_qwen_candidate_verifier import (
    ProductionWemmQwenCandidateVerifierError,
    build_candidate_verifier_prompt,
    diagnose_candidate_bound_consensus_gate,
    parse_qwen_candidate_verification_output,
    render_candidate_bound_consensus_gate_markdown,
    render_markdown,
    verify_wemm_qwen_candidate_sidecars,
)


def _candidate_pack() -> dict[str, object]:
    return {
        "windows": [
            {
                "window_id": "w00",
                "ordinal": 0,
                "source_interval": [0.0, 4.0],
                "model_context": {
                    "wemm": {
                        "status": "SUCCEEDED",
                        "top_k": [
                            {
                                "rank": 1,
                                "label_id": "pick_up",
                                "verb": "pick up",
                                "noun": "garment",
                                "canonical_label": "pick up garment",
                                "score": 0.8,
                            },
                            {
                                "rank": 2,
                                "label_id": "fold",
                                "verb": "fold",
                                "noun": "garment",
                                "canonical_label": "fold garment",
                                "score": 0.7,
                            },
                        ],
                    }
                },
            }
        ]
    }


def _qwen(raw_text: str, *, native: bool = True) -> dict[str, object]:
    row: dict[str, object] = {
        "window_id": "w00",
        "status": "SUCCEEDED",
        "raw_text": raw_text,
        "input_mode": "native_video" if native else "frames",
        "native_video_complete": native,
    }
    return {"windows": [row]}


def test_prompt_binds_qwen_to_wemm_top_k_and_complete_video() -> None:
    prompt = build_candidate_verifier_prompt(
        _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"],  # type: ignore[index]
        window_duration_seconds=4.0,
        window_id="w00",
    )
    assert "complete bounded native video" in prompt
    assert "never invent, translate, or replace" in prompt
    assert '"rank":1' in prompt
    assert "EPIC labels" in prompt


def test_prompt_rejects_duplicate_candidate_ranks() -> None:
    candidates = [
        {"rank": 1, "canonical_label": "pick up garment"},
        {"rank": 1, "canonical_label": "fold garment"},
    ]
    with pytest.raises(ProductionWemmQwenCandidateVerifierError, match="duplicated"):
        build_candidate_verifier_prompt(
            candidates,
            window_duration_seconds=4.0,
            window_id="w00",
        )


def test_prompt_all_candidates_compact_profile_requires_every_rank() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    prompt = build_candidate_verifier_prompt(
        candidates,
        window_duration_seconds=4.0,
        window_id="w00",
        verdict_scope="all_candidates",
    )
    assert "verdict_scope=all_candidates" in prompt
    assert "every supplied candidate" in prompt
    assert "omit evidence" in prompt
    assert '"verdict_scope":"all_candidates"' in prompt


def test_prompt_pairwise_profile_requires_two_candidates_and_forced_choice() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    prompt = build_candidate_verifier_prompt(
        candidates,
        window_duration_seconds=4.0,
        window_id="w00",
        verdict_scope="pairwise",
    )
    assert "verdict_scope=pairwise" in prompt
    assert "exactly the two supplied candidates" in prompt
    with pytest.raises(ProductionWemmQwenCandidateVerifierError, match="exactly two"):
        build_candidate_verifier_prompt(
            candidates[:1],
            window_duration_seconds=4.0,
            window_id="w00",
            verdict_scope="pairwise",
        )


def test_prompt_field_complete_profile_requires_explicit_fields() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    prompt = build_candidate_verifier_prompt(
        candidates,
        window_duration_seconds=4.0,
        window_id="w00",
        include_optional_fields=True,
    )
    assert "exactly ONE candidate_verdicts item" in prompt
    assert "attributes" in prompt
    assert "location" in prompt
    assert "hand" in prompt
    assert "explicit status" in prompt
    assert "required numeric confidence" in prompt
    assert "Never invent a label" in prompt


def test_prompt_field_complete_profile_rejects_all_candidates_scope() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    with pytest.raises(ProductionWemmQwenCandidateVerifierError, match="requires selected_only"):
        build_candidate_verifier_prompt(
            candidates,
            window_duration_seconds=4.0,
            window_id="w00",
            verdict_scope="all_candidates",
            include_optional_fields=True,
        )


def test_parse_all_candidates_compact_profile_retains_rank_comparison() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "all_candidates",
                "candidate_verdicts": [
                    {"rank": 1, "support": "unsupported"},
                    {"rank": 2, "support": "supported"},
                ],
                "decision": "accept",
                "selected_rank": 2,
                "segments": [],
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    assert output["verdict_scope"] == "all_candidates"
    assert output["selected_rank"] == 2
    assert [v["support"] for v in output["candidate_verdicts"]] == ["unsupported", "supported"]
    assert "CANDIDATE_VERDICTS_INCOMPLETE" not in output["warnings"]


def test_parse_pairwise_profile_allows_single_winner_verdict() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    pair = [candidates[0], candidates[1]]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "pairwise",
                "candidate_verdicts": [
                    {
                        "rank": 2,
                        "support": "supported",
                        "evidence": ["hands fold garment"],
                        "confidence": 0.7,
                    }
                ],
                "decision": "accept",
                "selected_rank": 2,
                "segments": [],
            }
        ),
        pair,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    assert output["verdict_scope"] == "pairwise"
    assert output["selected_rank"] == 2
    assert output["candidate_verdicts"][0]["confidence"] == 0.7


def test_parse_rejects_selected_rank_outside_top_k() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "all_candidates",
                "candidate_verdicts": [
                    {"rank": 1, "support": "unclear"},
                    {"rank": 2, "support": "unclear"},
                ],
                "decision": "accept",
                "selected_rank": 0,
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "INVALID"
    assert output["decision"] == "abstain"
    assert output.get("selected_rank") is None
    assert "selected_rank is not in WeMM Top-K" in output["errors"]


def test_parse_all_candidates_normalizes_zero_no_selection_on_abstain() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "all_candidates",
                "candidate_verdicts": [
                    {"rank": 1, "support": "unclear"},
                    {"rank": 2, "support": "unsupported"},
                ],
                "decision": "abstain",
                "selected_rank": 0,
                "segments": [],
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    assert output["decision"] == "abstain"
    assert output.get("selected_rank") is None
    assert "MODEL_SCHEMA_ALIAS_NO_SELECTION" in output["warnings"]


def test_parse_pairwise_normalizes_zero_no_selection_on_abstain() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    pair = [candidates[0], candidates[1]]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "pairwise",
                "candidate_verdicts": [
                    {"rank": 1, "support": "unclear"},
                    {"rank": 2, "support": "unsupported"},
                ],
                "decision": "abstain",
                "selected_rank": 0,
                "segments": [],
            }
        ),
        pair,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    assert output["decision"] == "abstain"
    assert output.get("selected_rank") is None
    assert "MODEL_SCHEMA_ALIAS_NO_SELECTION" in output["warnings"]


def test_parse_pairwise_zero_no_selection_does_not_expand_accept() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    pair = [candidates[0], candidates[1]]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "pairwise",
                "candidate_verdicts": [
                    {"rank": 1, "support": "supported"},
                    {"rank": 2, "support": "unclear"},
                ],
                "decision": "accept",
                "selected_rank": 0,
                "segments": [],
            }
        ),
        pair,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "INVALID"
    assert output["decision"] == "abstain"
    assert output.get("selected_rank") is None
    assert "selected_rank is not in WeMM Top-K" in output["errors"]


def test_parse_accept_requires_supported_candidate_and_preserves_fields() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "candidate_verdicts": [
                    {
                        "rank": 1,
                        "support": "supported",
                        "conflict": False,
                        "fields": {
                            "verb": {"value": "pick up", "status": "supported"},
                            "noun": {"value": "garment", "status": "supported"},
                            "attributes": {"value": None, "status": "not_observable"},
                            "location": {"value": "table", "status": "supported"},
                            "hand": {"value": "both hands", "status": "supported"},
                        },
                        "boundary": {
                            "status": "measured",
                            "start_time_sec": 1.0,
                            "end_time_sec": 2.5,
                        },
                        "evidence": ["hands lift garment"],
                        "confidence": 0.9,
                    },
                    {
                        "rank": 2,
                        "support": "unsupported",
                        "conflict": True,
                        "conflict_reasons": ["fold not visible"],
                        "evidence": ["no fold"],
                    },
                ],
                "decision": "accept",
                "selected_rank": 1,
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    assert output["decision"] == "accept"
    assert output["selected_rank"] == 1
    assert output["candidate_verdicts"][0]["fields"]["location"]["value"] == "table"
    assert output["candidate_verdicts"][0]["boundary"]["status"] == "measured"


def test_parse_field_complete_profile_accepts_explicit_optional_fields() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {
                        "rank": 2,
                        "support": "supported",
                        "fields": {
                            "verb": {"value": "fold", "status": "measured"},
                            "noun": {"value": "garment", "status": "measured"},
                            "attributes": {"value": None, "status": "not_observable"},
                            "location": {"value": "table", "status": "measured"},
                            "hand": {"value": "both hands", "status": "measured"},
                        },
                        "evidence": ["hands fold garment"],
                        "confidence": 0.8,
                        "boundary": {
                            "status": "measured",
                            "start_time_sec": 0.5,
                            "end_time_sec": 2.0,
                        },
                    }
                ],
                "decision": "accept",
                "selected_rank": 2,
                "segments": [
                    {
                        "candidate_rank": 2,
                        "boundary": {
                            "status": "measured",
                            "start_time_sec": 0.5,
                            "end_time_sec": 2.0,
                        },
                    }
                ],
            }
        ),
        candidates,
        window_duration_seconds=4.0,
        require_optional_fields=True,
    )
    assert output["parse_status"] == "PARSED"
    assert output["selected_rank"] == 2
    assert output["candidate_verdicts"][0]["fields"]["hand"]["status"] == "measured"


def test_parse_field_complete_profile_rejects_missing_field_status_and_evidence() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {
                        "rank": 1,
                        "support": "supported",
                        "fields": {
                            "verb": {"value": "pick up"},
                            "noun": {"value": "garment", "status": "measured"},
                            "attributes": {"value": None, "status": "not_observable"},
                            "location": {"value": None, "status": "not_observable"},
                            "hand": {"value": None, "status": "not_observable"},
                        },
                        "boundary": {"status": "unclear"},
                    }
                ],
                "decision": "abstain",
                "selected_rank": 1,
                "segments": [{"candidate_rank": 1, "boundary": {"status": "unclear"}}],
            }
        ),
        candidates,
        window_duration_seconds=4.0,
        require_optional_fields=True,
    )
    assert output["parse_status"] == "INVALID"
    assert "OPTIONAL_PROFILE_FIELD_STATUS_INVALID:verb" in output["errors"]
    assert "OPTIONAL_PROFILE_EVIDENCE_MISSING" in output["errors"]


def test_parse_field_complete_profile_rejects_schema_copy_and_placeholder_evidence() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {
                        "rank": 1,
                        "support": "supported",
                        "fields": {
                            "verb": {"value": "fold", "status": "measured"},
                            "noun": {"value": "shirt", "status": "measured"},
                            "attributes": {"value": None, "status": "not_observable"},
                            "location": {"value": None, "status": "not_observable"},
                            "hand": {"value": None, "status": "not_observable"},
                        },
                        "evidence": ["<observable evidence>"],
                        "confidence": 0.5,
                        "boundary": {"status": "unclear"},
                    }
                ],
                "decision": "accept",
                "selected_rank": 1,
                "segments": [{"candidate_rank": 1, "boundary": {"status": "unclear"}}],
            }
        ),
        candidates,
        window_duration_seconds=4.0,
        require_optional_fields=True,
    )
    assert output["parse_status"] == "INVALID"
    assert "OPTIONAL_PROFILE_VERB_MISMATCH" in output["errors"]
    assert "OPTIONAL_PROFILE_NOUN_MISMATCH" in output["errors"]
    assert "OPTIONAL_PROFILE_EVIDENCE_PLACEHOLDER" in output["errors"]


def test_parse_selected_only_scope_allows_compact_single_verdict() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "verdict_scope": "selected_only",
                "candidate_verdicts": [
                    {"rank": 2, "support": "supported", "evidence": ["hands fold garment"]}
                ],
                "decision": "accept",
                "selected_rank": 2,
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    assert output["verdict_scope"] == "selected_only"
    assert output["selected_rank"] == 2
    assert "CANDIDATE_VERDICTS_INCOMPLETE" not in output["warnings"]


def test_parse_rejects_label_not_in_top_k_and_out_of_window_boundary() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    output = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "candidate_verdicts": [
                    {
                        "rank": 1,
                        "label": "fold garment",
                        "support": "supported",
                        "evidence": ["visible"],
                        "boundary": {
                            "status": "measured",
                            "start_time_sec": 3.0,
                            "end_time_sec": 5.0,
                        },
                    }
                ],
                "decision": "accept",
                "selected_rank": 1,
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert output["parse_status"] == "PARSED"
    verdict = output["candidate_verdicts"][0]
    assert verdict["support"] == "unclear"
    assert "CANDIDATE_LABEL_MISMATCH" in verdict["reason_codes"]
    assert verdict["boundary"]["status"] == "not_measured"
    assert "BOUNDARY_OUT_OF_WINDOW" in verdict["reason_codes"]
    assert output["decision"] == "accept"


def test_sidecar_verifier_downgrades_accept_when_native_provenance_missing() -> None:
    raw = json.dumps(
        {
            "candidate_verdicts": [
                {
                    "rank": 1,
                    "support": "supported",
                    "fields": {
                        "verb": {"value": "pick up", "status": "supported"},
                        "noun": {"value": "garment", "status": "supported"},
                    },
                    "evidence": ["hands lift garment"],
                }
            ],
            "decision": "accept",
            "selected_rank": 1,
        }
    )
    report = verify_wemm_qwen_candidate_sidecars(_candidate_pack(), _qwen(raw, native=False))
    window = report["windows"][0]
    assert window["candidate_source"] == "wemm_top_k_only"
    assert window["decision"] == "abstain"
    assert "NATIVE_VIDEO_NOT_COMPLETE" in window["reason_codes"]
    assert report["production_eligible"] is False
    assert report["official_gold_status"] == "NOT_ESTABLISHED"
    assert report["accuracy_status"] == "NOT_MEASURED"


def test_sidecar_strict_evidence_gate_abstains_on_bare_supported_accept() -> None:
    raw = json.dumps(
        {
            "candidate_verdicts": [{"rank": 1, "support": "supported"}],
            "decision": "accept",
            "selected_rank": 1,
        }
    )
    report = verify_wemm_qwen_candidate_sidecars(
        _candidate_pack(),
        _qwen(raw),
        require_evidence_for_accept=True,
    )
    row = report["windows"][0]
    assert row["decision"] == "abstain"
    assert "ACCEPT_REQUIRES_EVIDENCE" in row["reason_codes"]
    assert report["controls"]["require_evidence_for_accept"] is True


def test_consensus_gate_can_require_candidate_evidence() -> None:
    joined = _joined_gate_report(1, 1, 1, 1, 1, 1)
    # The fixture intentionally has no evidence field on its supported
    # candidate verdicts, so strict mode must not turn camera agreement into
    # an accepted vote.
    report = diagnose_candidate_bound_consensus_gate(
        joined,
        require_evidence_for_accept=True,
    )
    row = report["windows"][0]
    assert row["gate_decision"] == "abstain"
    assert "NO_ELIGIBLE_CAMERA_VOTES" in row["gate_reason_codes"]
    assert "CANDIDATE_EVIDENCE_MISSING" in row["gate_reason_codes"]
    assert "CANDIDATE_BOUND_VIOLATION" not in row["gate_reason_codes"]
    assert row["candidate_evidence_failure_count"] == 6
    assert row["require_evidence_for_accept"] is True


def test_sidecar_diagnostics_report_join_and_constraint_failures() -> None:
    malformed = {
        "candidate_verdicts": [
            {
                "rank": 9,
                "support": "supported",
                "boundary": {"status": "measured", "start_time_sec": 0.0, "end_time_sec": 9.0},
            },
            {
                "rank": 1,
                "support": "unclear",
                "boundary": {"status": "measured", "start_time_sec": 0.0, "end_time_sec": 9.0},
            },
        ],
        "decision": "accept",
        "selected_rank": 9,
    }
    qwen = {
        "windows": [
            {
                "window_id": "w00",
                "camera_id": "cam_01",
                "input_mode": "frames",
                "native_video_complete": False,
                "raw_text": json.dumps(malformed),
            },
            {
                "window_id": "w00",
                "camera_id": "cam_02",
                "input_mode": "native_video",
                "native_video_complete": True,
                "parsed_verification": {
                    "parse_status": "PARSED",
                    "decision": "abstain",
                    "selected_rank": 1,
                    "candidate_verdicts": [
                        {
                            "rank": 1,
                            "support": "unclear",
                            "boundary": {
                                "status": "measured",
                                "start_time_sec": 0.0,
                                "end_time_sec": 9.0,
                            },
                        }
                    ],
                    "segments": [],
                    "errors": [],
                    "warnings": [],
                },
            },
            {
                "window_id": "w99",
                "camera_id": "cam_01",
                "input_mode": "native_video",
                "native_video_complete": True,
                "raw_text": json.dumps({"candidate_verdicts": [], "decision": "abstain"}),
            },
        ]
    }
    report = verify_wemm_qwen_candidate_sidecars(_candidate_pack(), qwen)
    diagnostics = report["diagnostics"]
    assert diagnostics["candidate_window_count"] == 1
    assert diagnostics["joined_window_count"] == 1
    assert diagnostics["extra_window_ids"] == ["w99"]
    assert diagnostics["native_incomplete_row_count"] == 1
    assert diagnostics["parse_invalid_row_count"] == 1
    assert diagnostics["rank_constraint_error_row_count"] == 1
    assert diagnostics["boundary_error_row_count"] == 2


def test_sidecar_verifier_does_not_accept_qwen_only_rank() -> None:
    raw = json.dumps(
        {
            "candidate_verdicts": [{"rank": 9, "support": "supported", "evidence": ["visible"]}],
            "decision": "accept",
            "selected_rank": 9,
        }
    )
    parsed = parse_qwen_candidate_verification_output(
        raw,
        _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"],  # type: ignore[index]
        window_duration_seconds=4.0,
    )
    assert parsed["parse_status"] == "INVALID"
    assert any("rank is not in WeMM Top-K" in error for error in parsed["errors"])


def test_split_requires_supported_candidate_segments() -> None:
    candidates = _candidate_pack()["windows"][0]["model_context"]["wemm"]["top_k"]  # type: ignore[index]
    parsed = parse_qwen_candidate_verification_output(
        json.dumps(
            {
                "candidate_verdicts": [
                    {"rank": 1, "support": "supported", "evidence": ["lift"]},
                    {"rank": 2, "support": "supported", "evidence": ["fold"]},
                ],
                "decision": "split",
                "segments": [
                    {"candidate_rank": 1, "boundary": {"status": "unclear"}},
                    {"candidate_rank": 2, "boundary": {"status": "unclear"}},
                ],
            }
        ),
        candidates,
        window_duration_seconds=4.0,
    )
    assert parsed["parse_status"] == "PARSED"
    report = verify_wemm_qwen_candidate_sidecars(
        _candidate_pack(),
        {"windows": [{**_qwen("{}", native=True)["windows"][0], "parsed_verification": parsed}]},
    )
    assert report["windows"][0]["decision"] == "split"


def test_markdown_is_concise() -> None:
    raw = json.dumps({"candidate_verdicts": [], "decision": "abstain"})
    report = verify_wemm_qwen_candidate_sidecars(_candidate_pack(), _qwen(raw))
    markdown = render_markdown(report)
    assert "WeMM → Qwen" in markdown
    assert "NOT_MEASURED" in markdown
    assert "Accuracy status" in markdown


def _joined_gate_report(*ranks: int, margin: float = 0.1) -> dict[str, object]:
    candidates = [
        {"rank": 1, "raw_label": "pick up garment", "score": 0.8},
        {"rank": 2, "raw_label": "fold garment", "score": 0.8 - margin},
    ]
    cameras = []
    for index, rank in enumerate(ranks):
        cameras.append(
            {
                "camera_id": f"cam_{index + 1:02d}",
                "decision": "accept",
                "native_video_complete": True,
                "parsed_verification": {
                    "parse_status": "PARSED",
                    "decision": "accept",
                    "selected_rank": rank,
                    "candidate_verdicts": [{"rank": rank, "support": "supported"}],
                    "segments": [],
                    "errors": [],
                    "warnings": [],
                },
            }
        )
    return {
        "windows": [
            {
                "window_id": "w00",
                "decision": "accept",
                "top_k": candidates,
                "camera_reports": cameras,
            }
        ]
    }


def test_consensus_gate_accepts_unanimous_six_camera_candidate_bound_vote() -> None:
    report = diagnose_candidate_bound_consensus_gate(_joined_gate_report(1, 1, 1, 1, 1, 1))
    row = report["windows"][0]
    assert report["diagnostic_only"] is True
    assert report["policy"]["decision_mutated"] is False
    assert row["gate_decision"] == "accept"
    assert row["camera_coverage"] == 1.0
    assert row["consensus_fraction"] == 1.0
    assert row["retrieval_margin_top1_top2"] == 0.1


def test_consensus_gate_abstains_on_sparse_camera_coverage_without_mutating_decision() -> None:
    report = diagnose_candidate_bound_consensus_gate(_joined_gate_report(1))
    row = report["windows"][0]
    assert row["recorded_decision"] == "accept"
    assert row["gate_decision"] == "abstain"
    assert "INSUFFICIENT_CAMERA_COVERAGE" in row["gate_reason_codes"]
    assert report["summary"]["recorded_accept_count"] == 1


def test_consensus_gate_abstains_on_tie_and_low_retrieval_margin() -> None:
    report = diagnose_candidate_bound_consensus_gate(
        _joined_gate_report(1, 1, 1, 2, 2, 2, margin=0.001),
        min_retrieval_margin=0.01,
    )
    row = report["windows"][0]
    assert row["gate_decision"] == "abstain"
    assert "NO_STRICT_CAMERA_CONSENSUS" in row["gate_reason_codes"]
    assert "LOW_RETRIEVAL_MARGIN" in row["gate_reason_codes"]
    markdown = render_candidate_bound_consensus_gate_markdown(report)
    assert "diagnostic_only" in markdown
    assert "LOW_RETRIEVAL_MARGIN" in markdown
