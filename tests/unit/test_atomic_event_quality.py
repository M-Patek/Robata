from __future__ import annotations

from robata.benchmark.atomic_event_quality import (
    AtomicEventReference,
    evaluate_raw_output_records,
    score_atomic_event_text,
)


def _score(verb: str, noun: str, text: str, *, all_nouns: tuple[str, ...] = ()):
    return score_atomic_event_text(
        text,
        reference=AtomicEventReference(verb=verb, noun=noun, all_nouns=all_nouns),
    )


def test_nine_case_atomic_sentinel_covers_transitions_and_continuous_control() -> None:
    cases = (
        ("turn-on", "tap", "The hand turns the faucet on and water starts flowing."),
        ("wash", "pot", "The hand scrubs the inside of the pot with a sponge."),
        ("turn-off", "tap", "The hand turns the tap off and the water stops flowing."),
        ("open", "cupboard", "The hand pulls the cabinet door open."),
        ("take", "plate", "The hand picks up a plate from the cupboard."),
        ("close", "cupboard", "The hand closes the cupboard door."),
        ("take", "knife", "The right hand lifts the knife from the counter."),
        ("cut", "carrot", "The knife continuously cuts the carrot into pieces."),
        ("put-down", "knife", "The hand places the knife down on the counter."),
    )

    scores = [_score(verb, noun, text) for verb, noun, text in cases]

    assert all(score.atomic_action_match for score in scores)
    assert all(score.object_match for score in scores)
    assert all(score.direction_match is not False for score in scores)
    assert all(score.mapper_input_sufficient for score in scores)


def test_turn_off_does_not_credit_adjacent_washing_under_running_water() -> None:
    score = _score(
        "turn-off",
        "tap",
        "The person is washing a pot in the sink while water runs from the faucet.",
    )

    assert score.atomic_action_match is False
    assert score.object_match is True
    assert score.direction_match is False
    assert score.adjacent_action_only is True
    assert score.generic_or_state_only is False
    assert score.mapper_input_sufficient is False
    assert "ADJACENT_ACTION_INSTEAD_OF_TARGET" in score.reasons


def test_vague_motion_and_holding_state_are_not_take_or_turn_off_events() -> None:
    vague = _score("turn-off", "extractor fan", "The hand moves toward a pot, then moves away.")
    holding = _score("take", "bag:cereal", "The person is holding a clear plastic bag.")

    assert vague.generic_or_state_only is True
    assert vague.atomic_action_match is False
    assert vague.object_match is False
    assert holding.generic_or_state_only is True
    assert holding.atomic_action_match is False
    assert holding.object_match is True
    assert holding.object_specificity_match is False
    assert holding.mapper_input_sufficient is False


def test_opposite_direction_and_multi_action_story_fail_mapper_sufficiency() -> None:
    opposite = _score(
        "close",
        "refrigerator",
        "The hand opens the fridge door and removes a container from inside.",
    )
    story = _score(
        "close",
        "cupboard",
        "The hand opens the cabinet, places a pot on the counter, then closes the cabinet.",
    )

    assert opposite.opposite_direction_asserted is True
    assert opposite.direction_match is False
    assert opposite.mapper_input_sufficient is False
    assert story.atomic_action_match is True
    assert story.object_match is True
    assert story.multiple_action_story is True
    assert story.mapper_input_sufficient is False


def test_hedged_alternative_action_is_reported_even_when_target_words_appear() -> None:
    score = _score(
        "turn-off",
        "hob",
        "The hand presses a cooktop button, likely to adjust the heat or turn the hob off.",
    )

    assert score.atomic_action_match is True
    assert score.object_match is True
    assert score.hedged is True
    assert score.mapper_input_sufficient is False


def test_record_evaluation_uses_official_reference_and_aggregates_families() -> None:
    report = evaluate_raw_output_records(
        [
            {
                "uid": "good",
                "raw_output_text": "The hand opens the refrigerator door.",
                "official_reference": {
                    "verb": "open",
                    "noun": "fridge",
                    "narration": "open fridge",
                    "all_nouns": ["fridge"],
                },
            },
            {
                "uid": "bad",
                "raw_output_text": "The person is holding a pot under a running faucet.",
                "official_reference": {
                    "verb": "turn-off",
                    "noun": "tap",
                    "narration": "turn off tap",
                    "all_nouns": "['tap']",
                },
            },
        ]
    )

    assert report["summary"]["case_count"] == 2
    assert report["summary"]["atomic_action_match_count"] == 1
    assert report["summary"]["object_match_count"] == 2
    assert report["summary"]["mapper_input_sufficient_count"] == 1
    assert report["summary"]["by_action_family"]["open"]["case_count"] == 1
    assert report["cases"][1]["quality"]["direction_match"] is False


def test_cutting_board_noun_does_not_count_as_cutting_action() -> None:
    score = _score(
        "cut",
        "carrot",
        "The person picks up a carrot from the cutting board and places it into a bowl.",
    )

    assert score.atomic_action_match is False
    assert score.adjacent_action_only is True
    assert score.multiple_action_story is True


def test_plain_put_uses_narration_preposition_to_distinguish_put_in() -> None:
    score = score_atomic_event_text(
        "The hand places the carrots into the bin.",
        reference=AtomicEventReference(
            verb="put",
            noun="carrots",
            narration="put carrots in bin",
        ),
    )

    assert score.action_family.value == "put-in"
    assert score.atomic_action_match is True
    assert score.multiple_action_story is False
