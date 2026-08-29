from __future__ import annotations

import pytest

from robata.benchmark.wemm_action_retrieval import (
    WemmRetrievalError,
    build_joint_action_catalog,
    build_retrieval_report,
    compare_rankings,
    cosine_similarity,
    evaluate_rankings,
    full_cartesian_action_pairs,
    project_retrieval_to_mapper,
    rank_joint_actions,
    render_action_label_texts,
    text_scores_for_prediction,
    validate_embedding_matrix,
)


def _labels():
    return build_joint_action_catalog(
        verb_table_or_entries={0: "open", 1: "close", 2: "wash"},
        noun_table_or_entries={0: "door", 1: "plate"},
        action_pairs=((0, 0), (1, 0), (2, 1)),
    )


def test_label_text_variants_are_deterministic():
    rendered = render_action_label_texts("open", "door")
    assert rendered == {
        "canonical": "open door",
        "verb_noun": "verb: open; noun: door",
        "natural": "a person is opening a door",
    }
    assert render_action_label_texts("put-on", "apple")["natural"] == (
        "a person is putting on an apple"
    )


def test_catalog_is_sorted_and_full_cartesian_is_explicit():
    labels = _labels()
    assert [label.action_key for label in labels] == [(0, 0), (1, 0), (2, 1)]
    assert full_cartesian_action_pairs({1: "close", 0: "open"}, {2: "cup", 1: "plate"}) == (
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    )
    assert labels[0].text_for("natural") == "a person is opening a door"


def test_validate_embeddings_normalizes_and_rejects_bad_shapes():
    assert validate_embedding_matrix([[3.0, 4.0]], expected_rows=1) == ((0.6, 0.8),)
    with pytest.raises(WemmRetrievalError, match="zero-norm"):
        validate_embedding_matrix([[0.0, 0.0]], expected_rows=1)
    with pytest.raises(WemmRetrievalError, match="expected"):
        validate_embedding_matrix([[1.0, 0.0]], expected_rows=2)
    with pytest.raises(WemmRetrievalError, match="inconsistent"):
        validate_embedding_matrix([[1.0, 0.0], [1.0]], expected_rows=2)


def test_visual_text_and_hybrid_rankings_keep_scores_and_tie_break_deterministic():
    labels = _labels()
    vectors = {
        (0, 0): (1.0, 0.0),
        (1, 0): (0.8, 0.6),
        (2, 1): (0.0, 1.0),
    }
    visual = rank_joint_actions(
        labels=labels,
        query_embedding=(1.0, 0.0),
        label_embeddings=vectors,
        mode="visual",
        top_k=3,
    )
    assert [item.action_key for item in visual] == [(0, 0), (1, 0), (2, 1)]
    assert visual[0].visual_cosine == pytest.approx(1.0)
    text = rank_joint_actions(
        labels=labels,
        text_scores={(0, 0): 0.2, (1, 0): 0.9, (2, 1): 0.1},
        mode="text",
        top_k=2,
    )
    assert [item.action_key for item in text] == [(1, 0), (0, 0)]
    hybrid = rank_joint_actions(
        labels=labels,
        query_embedding=(1.0, 0.0),
        label_embeddings=vectors,
        text_scores={(0, 0): 0.2, (1, 0): 0.9, (2, 1): 0.1},
        mode="hybrid",
        visual_weight=0.7,
        text_weight=0.3,
    )
    assert hybrid[0].action_key == (1, 0)
    assert tuple(item.rank for item in hybrid) == (1, 2, 3)


def test_rank_rejects_missing_inputs_and_invalid_weights():
    labels = _labels()
    with pytest.raises(WemmRetrievalError, match="require embeddings"):
        rank_joint_actions(labels=labels, mode="visual")
    with pytest.raises(WemmRetrievalError, match=r"both.*zero"):
        rank_joint_actions(
            labels=labels,
            query_embedding=(1.0, 0.0),
            label_embeddings={label.action_key: (1.0, 0.0) for label in labels},
            text_scores={},
            mode="hybrid",
            visual_weight=0.0,
            text_weight=0.0,
        )
    with pytest.raises(WemmRetrievalError, match="missing label"):
        rank_joint_actions(
            labels=labels,
            query_embedding=(1.0, 0.0),
            label_embeddings={(0, 0): (1.0, 0.0)},
            mode="visual",
        )


def test_text_scores_use_fields_and_retained_raw_phrase():
    labels = _labels()
    scores = text_scores_for_prediction(
        {"verb": "opening", "noun": "door", "raw_text": "open door"},
        labels,
    )
    assert scores[(0, 0)] > scores[(2, 1)]
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_evaluate_and_compare_handle_duplicate_or_fallback_row_ids():
    labels = _labels()
    ranking = rank_joint_actions(
        labels=labels,
        text_scores={(0, 0): 1.0, (1, 0): 0.0, (2, 1): 0.0},
        mode="text",
        top_k=3,
    )
    rows = [
        {"uid": None, "ground_truth": {"verb_class": 0, "noun_class": 0}, "video_group": "a"},
        {"uid": None, "ground_truth": {"verb_class": 2, "noun_class": 1}, "video_group": "b"},
    ]
    rankings = {"row-0": ranking, "row-1": tuple(reversed(ranking))}
    metrics = evaluate_rankings(rows, rankings, ks=(1, 3))
    assert metrics["scored_query_count"] == 2
    assert metrics["recall_at_k"]["3"] == 1.0
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["mrr"] == 1.0
    comparison = compare_rankings(rows, {"text": rankings})
    assert len(comparison["case_deltas"]) == 2
    assert comparison["case_deltas"][1]["text"]["top1"] == [2, 1]


def test_projection_and_report_are_nonproduction_and_explicit():
    labels = _labels()
    ranking = rank_joint_actions(
        labels=labels,
        text_scores={(0, 0): 0.8, (1, 0): 0.1, (2, 1): 0.0},
        mode="text",
        top_k=3,
    )
    mapped = project_retrieval_to_mapper(ranking, min_score=0.5, min_margin=0.1)
    assert mapped["status"] == "MAPPED"
    assert mapped["joint_selected"] == {
        "verb_id": 0,
        "noun_id": 0,
        "verb_key": "open",
        "noun_key": "door",
    }
    abstained = project_retrieval_to_mapper(ranking, min_score=0.99)
    assert abstained["status"] == "ABSTAIN"
    report = build_retrieval_report(
        rows=[{"uid": "x", "ground_truth": {"verb_class": 0, "noun_class": 0}}],
        rankings_by_mode={"text": {"x": ranking}},
        model_identity="fake-wemm",
        label_variant="canonical",
        catalog_size=len(labels),
        media_mode="fake",
        dimension=4,
        visual_weight=0.7,
        text_weight=0.3,
    )
    assert report["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert report["production_eligible"] is False
    assert report["controls"]["heldout_100_opened"] is False
    assert report["quality"]["metrics"]["text"]["top1_accuracy"] == 1.0


def test_cosine_rejects_dimension_and_zero_norm():
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    with pytest.raises(WemmRetrievalError, match="dimensions"):
        cosine_similarity((1.0,), (1.0, 0.0))
    with pytest.raises(WemmRetrievalError, match="non-zero"):
        cosine_similarity((0.0,), (1.0,))


def test_catalog_rejects_boolean_fractional_and_malformed_inputs():
    with pytest.raises(WemmRetrievalError, match="invalid action pair"):
        build_joint_action_catalog(
            verb_table_or_entries={0: "open"},
            noun_table_or_entries={0: "door"},
            action_pairs=((True, 0),),
        )
    with pytest.raises(WemmRetrievalError, match="invalid action pair"):
        build_joint_action_catalog(
            verb_table_or_entries={0: "open"},
            noun_table_or_entries={0: "door"},
            action_pairs=((1.5, 0),),
        )
    with pytest.raises(WemmRetrievalError, match="observed_counts"):
        build_joint_action_catalog(
            verb_table_or_entries={0: "open"},
            noun_table_or_entries={0: "door"},
            action_pairs=((0, 0),),
            observed_counts=[],
        )


def test_embedding_and_numeric_parameters_report_public_errors():
    with pytest.raises(WemmRetrievalError, match="non-numeric"):
        validate_embedding_matrix([["not-a-number"]], expected_rows=1)
    with pytest.raises(WemmRetrievalError, match="non-numeric"):
        validate_embedding_matrix([[True]], expected_rows=1)
    with pytest.raises(WemmRetrievalError, match="expected_rows"):
        validate_embedding_matrix([], expected_rows=True)
    labels = _labels()
    with pytest.raises(WemmRetrievalError, match="positive integer"):
        rank_joint_actions(labels=labels, text_scores={}, mode="text", top_k=1.5)
    with pytest.raises(WemmRetrievalError, match="positive integers"):
        evaluate_rankings([], {}, ks=(1.5,))


def test_compare_keeps_truth_bound_to_each_duplicate_explicit_id():
    labels = _labels()
    ranking_b = rank_joint_actions(
        labels=labels,
        text_scores={(0, 0): 0.0, (1, 0): 0.0, (2, 1): 1.0},
        mode="text",
    )
    rows = [
        {"uid": "duplicate", "ground_truth": {"verb_class": 0, "noun_class": 0}},
        {"uid": "duplicate", "ground_truth": {"verb_class": 2, "noun_class": 1}},
    ]
    comparison = compare_rankings(rows, {"text": {"duplicate": ranking_b}})
    assert [case["ground_truth"] for case in comparison["case_deltas"]] == [
        [0, 0],
        [2, 1],
    ]
    assert comparison["case_deltas"][0]["text"]["top1_correct"] is False
    assert comparison["case_deltas"][1]["text"]["top1_correct"] is True


def test_evaluation_reports_missing_and_empty_rankings_without_crashing():
    rows = [
        {"uid": "missing", "ground_truth": {"verb_class": 0, "noun_class": 0}},
        {"uid": "empty", "ground_truth": {"verb_class": 1, "noun_class": 0}},
        {"uid": "unscored"},
    ]
    metrics = evaluate_rankings(rows, {"empty": ()}, ks=(1,))
    assert metrics["query_count"] == 3
    assert metrics["scored_query_count"] == 1
    assert metrics["missing_ranking_ids"] == ["missing"]
    assert metrics["unscored_query_ids"] == ["unscored"]
    assert metrics["top1_accuracy"] == 0.0
    comparison = compare_rankings(rows, {"text": {"empty": ()}})
    assert comparison["case_deltas"][1]["text"]["top1"] is None


def test_zero_valued_explicit_row_id_is_not_treated_as_missing():
    labels = _labels()
    ranking = rank_joint_actions(
        labels=labels,
        text_scores={(0, 0): 1.0, (1, 0): 0.0, (2, 1): 0.0},
        mode="text",
    )
    metrics = evaluate_rankings(
        [{"uid": 0, "ground_truth": {"verb_class": 0, "noun_class": 0}}],
        {"0": ranking},
        ks=(1,),
    )
    assert metrics["scored_query_count"] == 1
    assert metrics["missing_ranking_ids"] == []


def test_projection_and_prediction_reject_malformed_inputs():
    with pytest.raises(WemmRetrievalError, match="ranking must be a sequence"):
        project_retrieval_to_mapper(None)  # type: ignore[arg-type]
    with pytest.raises(WemmRetrievalError, match="prediction must be a mapping"):
        text_scores_for_prediction([], _labels())  # type: ignore[arg-type]
    with pytest.raises(WemmRetrievalError, match="unsupported retrieval mode"):
        rank_joint_actions(labels=_labels(), mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(WemmRetrievalError, match="unsupported label variant"):
        rank_joint_actions(labels=_labels(), mode="text", text_scores={}, label_variant="x")  # type: ignore[arg-type]
