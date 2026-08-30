from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.production_wemm_rank_error_analysis import (
    ProductionWemmRankErrorAnalysisError,
    analyze_production_wemm_rank_errors,
    extend_production_wemm_rank_error_analysis,
    render_markdown,
)


def _route(variant: str, windows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "label_variant": variant,
        "provenance": {"epic_ontology_used": False, "mapper_used": False},
        "metrics": {},
        "per_window": {str(row["window_id"]): row for row in windows},
    }


def _comparison() -> dict[str, object]:
    windows = [
        {
            "window_id": "w0",
            "reference_pairs": [["pick up", "garment"]],
            "candidates": [
                {"rank": 1, "pair": ["pick up", "garment"], "score": 0.80},
                {"rank": 2, "pair": ["fold", "garment"], "score": 0.79},
            ],
        },
        {
            "window_id": "w1",
            "reference_pairs": [["smooth", "garment"]],
            "candidates": [
                {"rank": 1, "pair": ["fold", "garment"], "score": 0.701},
                {"rank": 2, "pair": ["flatten", "garment"], "score": 0.700},
                {"rank": 3, "pair": ["smooth", "garment"], "score": 0.699},
            ],
        },
        {
            "window_id": "w2",
            "reference_pairs": [["smooth", "garment"], ["adjust", "garment"]],
            "candidates": [
                {"rank": 1, "pair": ["fold", "garment"], "score": 0.65},
                {"rank": 2, "pair": ["adjust", "garment"], "score": 0.60},
            ],
        },
    ]
    return {
        "format": "robata-production-wemm-vocabulary-variant-comparison-v1",
        "status": "SURROGATE_ONLY",
        "quality_claim": False,
        "production_eligible": False,
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "reference": {
            "status": "INDEPENDENT_SURROGATE_REFERENCE",
            "eligible_window_count": 3,
            "excluded_window_count": 2,
        },
        "source_binding": {"status": "MATCHED"},
        "routes": {
            "canonical": _route("canonical", windows),
            "natural": _route("natural", windows),
        },
    }


def _comparison_with_camera_diagnostics() -> dict[str, object]:
    report = deepcopy(_comparison())
    route = report["routes"]["canonical"]  # type: ignore[index]
    route["per_window"] = deepcopy(route["per_window"])  # type: ignore[index]
    route["per_window"]["w0"]["camera_diagnostics"] = {  # type: ignore[index]
        "status": "AVAILABLE",
        "observed_camera_count": 4,
        "expected_camera_count": 4,
        "coverage_fraction": 1.0,
        "per_camera": [
            {
                "camera_id": "cam_01",
                "top1_action": "pick up garment",
                "top1_score": 0.80,
                "top2_action": "fold garment",
                "top2_score": 0.70,
                "top1_top2_margin": 0.10,
                "ranked_actions": ["pick up garment", "fold garment"],
            },
            {
                "camera_id": "cam_02",
                "top1_action": "pick up garment",
                "top1_score": 0.79,
                "top2_action": "fold garment",
                "top2_score": 0.71,
                "top1_top2_margin": 0.08,
                "ranked_actions": ["pick up garment", "fold garment"],
            },
            {
                "camera_id": "cam_03",
                "top1_action": "fold garment",
                "top1_score": 0.81,
                "top2_action": "pick up garment",
                "top2_score": 0.70,
                "top1_top2_margin": 0.11,
                "ranked_actions": ["fold garment", "pick up garment"],
            },
            {
                "camera_id": "cam_04",
                "top1_action": "pick up garment",
                "top1_score": 0.78,
                "top2_action": "fold garment",
                "top2_score": 0.75,
                "top1_top2_margin": 0.03,
                "ranked_actions": ["pick up garment", "fold garment"],
            },
        ],
    }
    # Keep another route aligned; it has no camera diagnostics and should be
    # reported as unavailable rather than treated as zero consensus.
    return report


def test_rank_distance_histogram_confusions_and_margin_bins() -> None:
    report = analyze_production_wemm_rank_errors(_comparison())
    canonical = report["routes"]["canonical"]  # type: ignore[index]
    action = canonical["action_level"]  # type: ignore[index]
    assert canonical["action_instance_count"] == 4  # type: ignore[index]
    assert action["top1_hits"] == 1
    assert action["rank_histogram"] == {
        "rank_1": 1,
        "rank_2": 1,
        "rank_3": 1,
        "not_in_top_k": 1,
    }
    # The split window contributes one row for each reference action; the
    # action-level top-1 rate therefore differs from window-level any-reference.
    assert action["top1_rate"] == pytest.approx(0.25)
    assert canonical["window_level"]["top1_rate_any_reference"] == pytest.approx(1 / 3)  # type: ignore[index]
    assert action["near_miss_rank_2_3_count"] == 2
    assert action["rank_4_plus_error_count"] == 0
    assert action["not_in_top_k_count"] == 1

    pair = canonical["confusion_pairs"]["single_reference_windows_only"][0]  # type: ignore[index]
    assert pair["predicted_top1"] == "fold garment"
    assert pair["ground_truth"] == "smooth garment"
    assert pair["count"] == 1

    margin_bins = canonical["margin_bins"]["bins"]  # type: ignore[index]
    assert sum(int(bucket["count"]) for bucket in margin_bins) == 3
    assert margin_bins[0]["label"] == "[0,0.001)"
    assert margin_bins[0]["top1_hits"] == 0
    assert margin_bins[-1]["label"] == "[0.02,+inf)"

    detail = canonical["per_action_instance"]  # type: ignore[index]
    assert detail[0]["gt_rank"] == 1
    assert detail[1]["rank_distance_from_top1"] == 2
    assert detail[2]["split_reference"] is True


def test_report_is_surrogate_only_and_markdown_is_readable() -> None:
    report = analyze_production_wemm_rank_errors(_comparison())
    assert report["status"] == "SURROGATE_ONLY"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["controls"]["model_invoked"] is False  # type: ignore[index]
    markdown = render_markdown(report)
    assert "Rank-1 error distance" in markdown
    assert "confusion pairs" in markdown
    assert "NOT_MEASURED" in markdown


def test_additive_exact_buckets_hard_negatives_clusters_and_variant_pairs() -> None:
    report = analyze_production_wemm_rank_errors(_comparison())
    canonical = report["routes"]["canonical"]  # type: ignore[index]
    action = canonical["action_level"]  # type: ignore[index]
    assert report["analysis"]["rank_buckets"] == [  # type: ignore[index]
        "rank_1",
        "rank_2",
        "rank_3",
        "rank_4_plus",
        "not_in_top_k",
    ]
    assert action["exact_rank_histogram"] == {  # type: ignore[index]
        "rank_1": 1,
        "rank_2": 1,
        "rank_3": 1,
        "not_in_top_k": 1,
    }
    assert action["rank_bucket_histogram"] == {  # type: ignore[index]
        "rank_1": 1,
        "rank_2": 1,
        "rank_3": 1,
        "rank_4_plus": 0,
        "not_in_top_k": 1,
    }
    assert canonical["hard_negative_analysis"]["hard_negative_count"] > 0  # type: ignore[index]
    assert canonical["per_action_instance"][1]["top1_hard_negative_type"] == "same_noun"  # type: ignore[index]
    clusters = canonical["confusion_clusters"]["single_reference_windows_only"]  # type: ignore[index]
    assert clusters[0]["nodes"] == ["fold garment", "smooth garment"]
    comparison = report["prototype_variant_comparison"]  # type: ignore[index]
    assert comparison["baseline_variant"] == "canonical"
    assert "canonical_vs_natural" in comparison["pairwise"]
    assert len(comparison["per_window"]) == 3


def test_camera_consensus_is_separate_from_fused_rank_and_has_rank_summary() -> None:
    report = analyze_production_wemm_rank_errors(_comparison_with_camera_diagnostics())
    canonical = report["routes"]["canonical"]  # type: ignore[index]
    camera = canonical["camera_consensus"]  # type: ignore[index]
    assert camera["status"] == "AVAILABLE"
    assert camera["windows_total"] == 3
    assert camera["windows_measured"] == 1
    assert camera["strict_majority_count"] == 1
    assert camera["strict_majority_rate"] == 1.0
    assert camera["winner_matches_reference_count"] == 1
    assert camera["fused_top1_agrees_with_consensus_count"] == 1
    assert camera["bins"][1]["windows"] == 1  # [0.5,1) consensus bucket
    row = canonical["per_window"][0]  # type: ignore[index]
    assert row["camera_consensus"]["consensus_winner"] == "pick up garment"  # type: ignore[index]
    action_row = canonical["per_action_instance"][0]  # type: ignore[index]
    assert action_row["camera_consensus_matches_gt"] is True
    assert action_row["camera_gt_rank_min"] == 1
    assert action_row["camera_gt_rank_mean"] == 1.25
    assert action_row["camera_gt_rank_median"] == 1.0
    # A route without camera rows is explicitly unavailable, not a zero score.
    assert report["routes"]["natural"]["camera_consensus"]["status"] == "NOT_AVAILABLE"  # type: ignore[index]
    assert "Camera-consensus diagnostics" in render_markdown(report)


def test_existing_v1_rank_report_can_be_extended_without_model_or_hash_work() -> None:
    original = analyze_production_wemm_rank_errors(_comparison())
    extended = extend_production_wemm_rank_error_analysis(original)
    assert extended["format"] == original["format"]
    assert extended["analysis"]["extension_format"].endswith("-v1")  # type: ignore[index]
    assert extended["controls"]["model_invoked"] is False
    assert extended["controls"]["hash_or_digest_computed"] is False
    assert extended["prototype_variant_comparison"]["variants"] == [  # type: ignore[index]
        "canonical",
        "natural",
    ]


def test_rejects_production_or_epic_claims() -> None:
    bad = _comparison()
    bad["production_eligible"] = True
    with pytest.raises(ProductionWemmRankErrorAnalysisError, match="production_eligible"):
        analyze_production_wemm_rank_errors(bad)

    bad = _comparison()
    bad["routes"]["canonical"]["provenance"]["epic_ontology_used"] = True  # type: ignore[index]
    with pytest.raises(ProductionWemmRankErrorAnalysisError, match="epic_ontology_used"):
        analyze_production_wemm_rank_errors(bad)


def test_rejects_misaligned_variants_and_duplicate_candidates() -> None:
    bad = _comparison()
    bad["routes"]["natural"]["per_window"] = {}  # type: ignore[index]
    with pytest.raises(ProductionWemmRankErrorAnalysisError, match="not aligned"):
        analyze_production_wemm_rank_errors(bad)

    bad = _comparison()
    candidate = bad["routes"]["canonical"]["per_window"]["w0"]["candidates"][1]  # type: ignore[index]
    candidate["rank"] = 1
    with pytest.raises(ProductionWemmRankErrorAnalysisError, match="duplicate candidate rank"):
        analyze_production_wemm_rank_errors(bad)
