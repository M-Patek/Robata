from __future__ import annotations

import pytest

from robata.benchmark.production_wemm_vocabulary_comparison import (
    ProductionWemmVariantComparisonError,
    compare_production_wemm_vocabulary_variants,
    render_markdown,
)


def _review() -> dict[str, object]:
    return {
        "format": "robata-production-owner-confirmation-v1",
        "source": {"media_path": "data/source/sample-medium.mcap"},
        "windows": [
            {
                "window_id": "w00",
                "decision": "accept",
                "segments": [{"verb": "pick up", "noun": "garment"}],
            },
            {
                "window_id": "w01",
                "decision": "accept",
                "segments": [{"verb": "spread", "noun": "garment"}],
            },
            {
                "window_id": "w02",
                "decision": "abstain",
                "segments": [],
            },
        ],
    }


def _sidecar(variant: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format": "robata-production-wemm-vocabulary-shadow-v1",
        "status": "SUCCEEDED",
        "production_eligible": False,
        "source": {"path": "D:\\Github\\Robata\\data\\source\\sample-medium.mcap"},
        "model": {"label_variant": variant},
        "vocabulary": {
            "format": "robata-production-coarse-vocabulary-owner-approval-v1",
            "profile": "PRODUCTION_OWNER_APPROVED_COARSE_VOCABULARY",
            "epic_ontology_used": False,
            "mapper_used": False,
        },
        "windows": rows,
    }


def test_compare_reports_variant_topk_mrr_and_provenance() -> None:
    canonical = _sidecar(
        "canonical",
        [
            {
                "window_id": "w00",
                "model": {
                    "predictions": [
                        {"rank": 1, "label_id": "pick_up", "verb": "pick up", "noun": "garment"}
                    ]
                },
            },
            {
                "window_id": "w01",
                "model": {
                    "predictions": [
                        {"rank": 1, "label_id": "fold", "verb": "fold", "noun": "garment"},
                        {"rank": 2, "label_id": "spread", "verb": "spread", "noun": "garment"},
                    ]
                },
            },
            {"window_id": "w02", "model": {"predictions": []}},
        ],
    )
    verb_noun = _sidecar(
        "verb_noun",
        [
            {
                "window_id": "w00",
                "model": {
                    "predictions": [
                        {"rank": 1, "label_id": "fold", "verb": "fold", "noun": "garment"},
                        {"rank": 2, "label_id": "pick_up", "verb": "pick up", "noun": "garment"},
                    ]
                },
            },
            {
                "window_id": "w01",
                "model": {
                    "predictions": [
                        {"rank": 1, "label_id": "spread", "verb": "spread", "noun": "garment"}
                    ]
                },
            },
            {"window_id": "w02", "model": {"predictions": []}},
        ],
    )
    natural = _sidecar(
        "natural",
        canonical["windows"],  # type: ignore[arg-type]
    )

    report = compare_production_wemm_vocabulary_variants(
        _review(),
        {"canonical": canonical, "verb_noun": verb_noun, "natural": natural},
    )
    assert report["status"] == "SURROGATE_ONLY"
    assert report["reference"]["eligible_window_count"] == 2  # type: ignore[index]
    assert report["routes"]["canonical"]["metrics"]["top1"]["rate"] == 0.5  # type: ignore[index]
    assert report["routes"]["canonical"]["metrics"]["top5"]["rate"] == 1.0  # type: ignore[index]
    assert report["routes"]["canonical"]["metrics"]["mrr"] == pytest.approx(0.75)  # type: ignore[index]
    assert report["routes"]["verb_noun"]["metrics"]["top1"]["rate"] == 0.5  # type: ignore[index]
    assert report["source_binding"]["status"] == "MATCHED"  # type: ignore[index]
    assert report["controls"]["model_invoked"] is False  # type: ignore[index]
    assert report["metric_units"]["top_k"] == "window_level"  # type: ignore[index]
    cardinality = report["routes"]["canonical"]["metrics"]["candidate_list_cardinality"]  # type: ignore[index]
    assert cardinality["min"] == 1  # type: ignore[index]
    assert cardinality["max"] == 2  # type: ignore[index]
    assert cardinality["full_list_at_k"]["10"] is True  # type: ignore[index]
    rendered = render_markdown(report)
    assert "SURROGATE_ONLY" in rendered
    assert "Candidate-list cardinality" in rendered
    assert "Window R@1" in rendered


def test_compare_rejects_epic_sidecar_or_source_conflict() -> None:
    sidecar = _sidecar(
        "canonical",
        [{"window_id": "w00", "model": {"predictions": []}}],
    )
    bad_epic = dict(sidecar)
    bad_epic["format"] = "robata-production-wemm-shadow-v1"
    with pytest.raises(ProductionWemmVariantComparisonError, match="format"):
        compare_production_wemm_vocabulary_variants(_review(), {"canonical": bad_epic})

    conflict = dict(sidecar)
    conflict["source"] = {"path": "other-video.mcap"}
    with pytest.raises(ProductionWemmVariantComparisonError, match="conflicting"):
        compare_production_wemm_vocabulary_variants(_review(), {"canonical": conflict})


def test_independent_review_recommendation_excludes_abstain_rows() -> None:
    """Independent Terra's recommendation field is authoritative for eligibility."""

    review = {
        "format": "robata-production-independent-review-v1",
        "source": {"media_path": "data/source/sample-medium.mcap"},
        "items": [
            {
                "window_id": "w00",
                "recommendation": "EDIT",
                "segments": [{"verb": "pick up", "noun": "garment"}],
            },
            {
                "window_id": "w01",
                "recommendation": "ABSTAIN",
                # The reviewer retains a diagnostic candidate, but it must not
                # become a positive retrieval target.
                "segments": [{"verb": "adjust", "noun": "garment"}],
            },
            {
                "window_id": "w02",
                "recommendation": "SPLIT",
                "segments": [
                    {"verb": "smooth", "noun": "garment"},
                    {"verb": "adjust", "noun": "garment"},
                ],
            },
        ],
    }
    canonical = _sidecar(
        "canonical",
        [
            {
                "window_id": "w00",
                "model": {"predictions": [{"rank": 1, "verb": "pick up", "noun": "garment"}]},
            },
            {
                "window_id": "w01",
                "model": {"predictions": [{"rank": 1, "verb": "adjust", "noun": "garment"}]},
            },
            {
                "window_id": "w02",
                "model": {"predictions": [{"rank": 1, "verb": "smooth", "noun": "garment"}]},
            },
        ],
    )
    report = compare_production_wemm_vocabulary_variants(
        review,
        {"canonical": canonical},
    )
    assert report["reference"]["eligible_window_count"] == 2  # type: ignore[index]
    assert report["reference"]["excluded_window_count"] == 1  # type: ignore[index]
    assert report["routes"]["canonical"]["metrics"]["top1"]["rate"] == 1.0  # type: ignore[index]


def test_compare_rejects_mapper_enabled_sidecar() -> None:
    sidecar = _sidecar(
        "canonical",
        [{"window_id": "w00", "model": {"predictions": []}}],
    )
    sidecar["vocabulary"]["mapper_used"] = True  # type: ignore[index]
    with pytest.raises(ProductionWemmVariantComparisonError, match="mapper_used=false"):
        compare_production_wemm_vocabulary_variants(_review(), {"canonical": sidecar})


def test_compare_rejects_invalid_cutoffs() -> None:
    with pytest.raises(ProductionWemmVariantComparisonError, match="positive"):
        compare_production_wemm_vocabulary_variants(
            _review(), {"canonical": _sidecar("canonical", [])}, ks=(0,)
        )


def test_compare_projects_optional_camera_consensus_without_changing_rank_metrics() -> None:
    sidecar = _sidecar(
        "canonical",
        [
            {
                "window_id": "w00",
                "model": {
                    "predictions": [
                        {
                            "rank": 1,
                            "label_id": "pick_up",
                            "verb": "pick up",
                            "noun": "garment",
                            "score": 0.8,
                        },
                        {
                            "rank": 2,
                            "label_id": "fold",
                            "verb": "fold",
                            "noun": "garment",
                            "score": 0.7,
                        },
                    ],
                    "fusion": {
                        "camera_order": ["cam_01", "cam_02"],
                        "camera_coverage": {"expected_count": 2},
                    },
                    "per_camera_predictions": {
                        "cam_01": [
                            {
                                "rank": 1,
                                "verb": "pick up",
                                "noun": "garment",
                                "score": 0.8,
                            },
                            {
                                "rank": 2,
                                "verb": "fold",
                                "noun": "garment",
                                "score": 0.7,
                            },
                        ],
                        "cam_02": [
                            {
                                "rank": 1,
                                "verb": "fold",
                                "noun": "garment",
                                "score": 0.81,
                            },
                            {
                                "rank": 2,
                                "verb": "pick up",
                                "noun": "garment",
                                "score": 0.79,
                            },
                        ],
                    },
                },
            },
        ],
    )
    report = compare_production_wemm_vocabulary_variants(_review(), {"canonical": sidecar})
    diagnostics = report["routes"]["canonical"]["per_window"]["w00"]["camera_diagnostics"]  # type: ignore[index]
    assert diagnostics["status"] == "AVAILABLE"
    assert diagnostics["observed_camera_count"] == 2
    assert diagnostics["consensus_fraction"] == 0.5
    assert diagnostics["strict_majority"] is False
    assert diagnostics["top1_votes"][0]["votes"] == 1
    assert diagnostics["per_camera"][0]["ranked_actions"] == [
        "pick up garment",
        "fold garment",
    ]
    # Existing retrieval metrics remain window-level and unchanged.
    assert report["routes"]["canonical"]["metrics"]["top1"]["rate"] == 0.5  # type: ignore[index]
