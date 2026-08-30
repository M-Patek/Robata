"""Lightweight, Mapper-before quality scoring for free-form video observations.

The scorer is intentionally benchmark-only.  It compares a model's retained free
text with an EPIC human annotation after generation and answers the questions that
matter before ontology mapping:

* did the text assert the expected atomic action rather than only a scene state;
* did it name the directly manipulated object or control;
* for directional actions, did it preserve the correct transition direction; and
* did it stay focused on one event instead of narrating adjacent actions.

This is a small, inspectable lexical sentinel, not a replacement for human review
or a learned semantic judge.  It has no model/runtime dependency and deliberately
does not create hashes, identities, gates, or production authority.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

RAW_ATOMIC_EVENT_QUALITY_VERSION = "epic-raw-atomic-event-quality-v1"


class AtomicActionFamily(StrEnum):
    """Action families whose direction and granularity can be inspected in prose."""

    TAKE = "take"
    PUT_DOWN = "put-down"
    PUT_IN = "put-in"
    OPEN = "open"
    CLOSE = "close"
    TURN_ON = "turn-on"
    TURN_OFF = "turn-off"
    WASH = "wash"
    CUT = "cut"
    POUR = "pour"
    STIR = "stir"
    FOLD = "fold"
    SHAKE = "shake"
    THROW = "throw"
    SCOOP = "scoop"
    MOVE = "move"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class AtomicEventReference:
    """The small official-reference projection needed by the raw-text scorer."""

    verb: str
    noun: str
    narration: str = ""
    all_nouns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.verb.strip() or not self.noun.strip():
            raise ValueError("atomic event reference requires verb and noun")


@dataclass(frozen=True, slots=True)
class AtomicEventQualityScore:
    """One post-generation, pre-Mapper quality judgement."""

    action_family: AtomicActionFamily
    atomic_action_match: bool
    object_match: bool
    object_specificity_match: bool | None
    direction_required: bool
    direction_match: bool | None
    opposite_direction_asserted: bool
    generic_or_state_only: bool
    adjacent_action_only: bool
    hedged: bool
    multiple_action_story: bool
    mapper_input_sufficient: bool
    matched_action_families: tuple[str, ...]
    expected_object_surfaces: tuple[str, ...]
    expected_specificity_surfaces: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_observation_text(value: Any) -> str:
    """Normalize punctuation without inventing semantic aliases."""

    return _SPACE.sub(" ", " ".join(_TOKEN.findall(str(value or "").casefold()))).strip()


def _patterns(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value) for value in values)


_ACTION_PATTERNS: dict[AtomicActionFamily, tuple[re.Pattern[str], ...]] = {
    AtomicActionFamily.TAKE: _patterns(
        r"\bpick(?:s|ed|ing)?\s+up\b",
        r"\btak(?:e|es|en|ing)\b",
        r"\btook\b",
        r"\blift(?:s|ed|ing)?\b",
        r"\bremov(?:e|es|ed|ing)\b",
        r"\bpull(?:s|ed|ing)?\b.{0,70}\bout\b",
    ),
    AtomicActionFamily.PUT_DOWN: _patterns(
        r"\bput(?:s|ting)?\b.{0,60}\bdown\b",
        r"\bplac(?:e|es|ed|ing)\b.{0,70}\b(?:down|on|onto|atop)\b",
        r"\bset(?:s|ting)?\b.{0,70}\b(?:down|on|onto|atop)\b",
        r"\bl(?:ay|ays|aid|aying)\b",
    ),
    AtomicActionFamily.PUT_IN: _patterns(
        r"\bput(?:s|ting)?\b.{0,70}\b(?:in|into|inside)\b",
        r"\bplac(?:e|es|ed|ing)\b.{0,70}\b(?:in|into|inside)\b",
        r"\binsert(?:s|ed|ing)?\b",
        r"\bpour(?:s|ed|ing)?\b.{0,70}\binto\b",
    ),
    AtomicActionFamily.OPEN: _patterns(r"\bopen(?:s|ed|ing)?\b"),
    AtomicActionFamily.CLOSE: _patterns(r"\bclos(?:e|es|ed|ing)\b", r"\bshut(?:s|ting)?\b"),
    AtomicActionFamily.TURN_ON: _patterns(
        r"\bturn(?:s|ed|ing)?\b.{0,35}\bon\b",
        r"\bswitch(?:es|ed|ing)?\b.{0,35}\bon\b",
        r"\bactivat(?:e|es|ed|ing)\b",
        r"\bwater\b.{0,30}\b(?:begins|starts)\b.{0,25}\b(?:flow|flowing|run|running)\b",
    ),
    AtomicActionFamily.TURN_OFF: _patterns(
        r"\bturn(?:s|ed|ing)?\b.{0,35}\boff\b",
        r"\bswitch(?:es|ed|ing)?\b.{0,35}\boff\b",
        r"\bdeactivat(?:e|es|ed|ing)\b",
        r"\b(?:water|flow|faucet|tap|fan|burner|hob|stove)\b.{0,35}"
        r"\b(?:stops|stopped|ceases|ceased)\b",
        r"\b(?:stops|stopped)\b.{0,35}\b(?:water|flow|faucet|tap|fan|burner|hob|stove)\b",
    ),
    AtomicActionFamily.WASH: _patterns(
        r"\bwash(?:es|ed|ing)?\b",
        r"\brins(?:e|es|ed|ing)\b",
        r"\bscrub(?:s|bed|bing)?\b",
        r"\bclean(?:s|ed|ing)?\b",
    ),
    AtomicActionFamily.CUT: _patterns(
        r"\bcut(?:s|ting)?\b(?!\s+board\b)",
        r"\bchop(?:s|ped|ping)?\b",
        r"\bslic(?:e|es|ed|ing)\b",
        r"\btrim(?:s|med|ming)?\b",
    ),
    AtomicActionFamily.POUR: _patterns(
        r"\bpour(?:s|ed|ing)?\b",
        r"\bdrain(?:s|ed|ing)?\b",
        r"\bempt(?:y|ies|ied|ying)\b",
    ),
    AtomicActionFamily.STIR: _patterns(r"\bstir(?:s|red|ring)?\b"),
    AtomicActionFamily.FOLD: _patterns(r"\bfold(?:s|ed|ing)?\b"),
    AtomicActionFamily.SHAKE: _patterns(r"\bshak(?:e|es|en|ing)\b", r"\bshook\b"),
    AtomicActionFamily.THROW: _patterns(
        r"\bthrow(?:s|ing)?\b", r"\bthrew\b", r"\bdiscard(?:s|ed|ing)?\b"
    ),
    AtomicActionFamily.SCOOP: _patterns(r"\bscoop(?:s|ed|ing)?\b", r"\bshovel(?:s|ed|ing)?\b"),
    AtomicActionFamily.MOVE: _patterns(
        r"\bmov(?:e|es|ed|ing)\b",
        r"\breposition(?:s|ed|ing)?\b",
        r"\badjust(?:s|ed|ing)?\b",
    ),
}

_VERB_FAMILY: dict[str, AtomicActionFamily] = {
    "take": AtomicActionFamily.TAKE,
    "pick up": AtomicActionFamily.TAKE,
    "put": AtomicActionFamily.PUT_DOWN,
    "put down": AtomicActionFamily.PUT_DOWN,
    "put in": AtomicActionFamily.PUT_IN,
    "open": AtomicActionFamily.OPEN,
    "close": AtomicActionFamily.CLOSE,
    "turn on": AtomicActionFamily.TURN_ON,
    "turn off": AtomicActionFamily.TURN_OFF,
    "wash": AtomicActionFamily.WASH,
    "rinse": AtomicActionFamily.WASH,
    "cut": AtomicActionFamily.CUT,
    "cut into": AtomicActionFamily.CUT,
    "cut off": AtomicActionFamily.CUT,
    "chop": AtomicActionFamily.CUT,
    "slice": AtomicActionFamily.CUT,
    "pour out": AtomicActionFamily.POUR,
    "stir": AtomicActionFamily.STIR,
    "fold": AtomicActionFamily.FOLD,
    "shake": AtomicActionFamily.SHAKE,
    "throw": AtomicActionFamily.THROW,
    "shovel up": AtomicActionFamily.SCOOP,
    "move": AtomicActionFamily.MOVE,
    "adjust": AtomicActionFamily.MOVE,
}

_DIRECTIONAL_FAMILIES = frozenset(
    {
        AtomicActionFamily.TAKE,
        AtomicActionFamily.PUT_DOWN,
        AtomicActionFamily.PUT_IN,
        AtomicActionFamily.OPEN,
        AtomicActionFamily.CLOSE,
        AtomicActionFamily.TURN_ON,
        AtomicActionFamily.TURN_OFF,
        AtomicActionFamily.POUR,
    }
)

_OPPOSITES: dict[AtomicActionFamily, frozenset[AtomicActionFamily]] = {
    AtomicActionFamily.TAKE: frozenset({AtomicActionFamily.PUT_DOWN, AtomicActionFamily.PUT_IN}),
    AtomicActionFamily.PUT_DOWN: frozenset({AtomicActionFamily.TAKE}),
    AtomicActionFamily.PUT_IN: frozenset({AtomicActionFamily.TAKE}),
    AtomicActionFamily.OPEN: frozenset({AtomicActionFamily.CLOSE}),
    AtomicActionFamily.CLOSE: frozenset({AtomicActionFamily.OPEN}),
    AtomicActionFamily.TURN_ON: frozenset({AtomicActionFamily.TURN_OFF}),
    AtomicActionFamily.TURN_OFF: frozenset({AtomicActionFamily.TURN_ON}),
}

_STATE_PATTERNS = _patterns(
    r"\b(?:is|are|was|were)\s+holding\b",
    r"\bremains?\b",
    r"\b(?:is|are)\s+positioned\b",
    r"\b(?:is|are)\s+sitting\b",
    r"\bscene\s+is\s+static\b",
    r"\bno\s+(?:visible\s+)?(?:movement|action|interaction)\b",
    r"\bwater\s+(?:is\s+)?(?:flowing|running)\b",
)
_VAGUE_MOTION_PATTERNS = _patterns(
    r"\b(?:moves?|reaches?)\s+(?:toward|towards|away|near)\b",
    r"\binteract(?:s|ed|ing)?\s+with\b",
    r"\bmanipulat(?:e|es|ed|ing)\b",
    r"\bwithout\s+touching\b",
)
_HEDGE_PATTERNS = _patterns(
    r"\bappears?\s+to\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bperhaps\b",
    r"\bas\s+if\b",
    r"\b(?:may|might|could)\s+be\b",
    r"\bor\s+(?:about\s+to\s+)?(?:open|close|turn|put|take|pick|move|adjust)\b",
)


def _matches(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _matched_families(text: str) -> tuple[AtomicActionFamily, ...]:
    return tuple(
        family for family, patterns in _ACTION_PATTERNS.items() if _matches(patterns, text)
    )


def _fallback_action_match(expected_verb: str, text: str) -> bool:
    expected_tokens = normalize_observation_text(expected_verb).split()
    if not expected_tokens:
        return False
    text_tokens = set(text.split())
    informative = [token for token in expected_tokens if token not in {"in", "into", "off", "on"}]
    return bool(informative) and all(token in text_tokens for token in informative)


_NOUN_EQUIVALENTS: tuple[frozenset[str], ...] = (
    frozenset({"tap", "faucet", "spigot"}),
    frozenset({"hob", "induction hob", "cooktop", "stove", "burner"}),
    frozenset({"extractor fan", "extractor", "range hood", "vent hood", "hood fan"}),
    frozenset({"cupboard", "cabinet"}),
    frozenset({"refrigerator", "fridge"}),
    frozenset({"bin", "trash bin", "garbage bin", "waste bin"}),
    frozenset({"scissors", "shears"}),
    frozenset({"tea towel", "dish towel", "towel", "cloth"}),
    frozenset({"carrot sticks", "carrots", "carrot"}),
    frozenset({"packing bag", "packaging bag", "packaging", "bag"}),
)


def _singular_token(token: str) -> str:
    irregular = {"knives": "knife", "leaves": "leaf", "loaves": "loaf"}
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ves"):
        return token[:-3] + "f"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _surface_tokens(value: str) -> tuple[str, ...]:
    return tuple(_singular_token(token) for token in normalize_observation_text(value).split())


def _contains_surface(text: str, surface: str) -> bool:
    wanted = _surface_tokens(surface)
    if not wanted:
        return False
    available = tuple(_singular_token(token) for token in text.split())
    width = len(wanted)
    return any(
        available[index : index + width] == wanted for index in range(len(available) - width + 1)
    )


def _equivalent_surfaces(value: str) -> set[str]:
    normalized = normalize_observation_text(value)
    result = {normalized} if normalized else set()
    for group in _NOUN_EQUIVALENTS:
        if normalized in group:
            result.update(group)
    return result


def _object_surfaces(reference: AtomicEventReference) -> tuple[tuple[str, ...], tuple[str, ...]]:
    heads: set[str] = set()
    specificity: set[str] = set()
    values = (reference.noun, *reference.all_nouns)
    for value in values:
        raw_parts = [part for part in re.split(r"[:;]", str(value)) if part.strip()]
        if not raw_parts:
            continue
        head = normalize_observation_text(raw_parts[0])
        heads.update(_equivalent_surfaces(head))
        if len(raw_parts) > 1:
            for modifier in raw_parts[1:]:
                normalized_modifier = normalize_observation_text(modifier)
                if normalized_modifier:
                    specificity.update(_equivalent_surfaces(normalized_modifier))
    return tuple(sorted(heads)), tuple(sorted(specificity))


def _coerce_text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        # Some EPIC projections retain the original Python-list-looking CSV cell.
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            items = re.findall(r"['\"]([^'\"]+)['\"]", stripped)
            return tuple(item.strip() for item in items if item.strip())
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def reference_from_record(record: Mapping[str, Any]) -> AtomicEventReference:
    """Read the official-reference shape used by the Qwen and Mage runners."""

    raw_reference = record.get("official_reference")
    if not isinstance(raw_reference, Mapping):
        raw_reference = record.get("ground_truth")
    if not isinstance(raw_reference, Mapping):
        raise ValueError("raw output record has no official_reference or ground_truth")
    return AtomicEventReference(
        verb=str(raw_reference.get("verb") or ""),
        noun=str(raw_reference.get("noun") or ""),
        narration=str(raw_reference.get("narration") or ""),
        all_nouns=_coerce_text_tuple(raw_reference.get("all_nouns")),
    )


def _expected_action_family(reference: AtomicEventReference) -> AtomicActionFamily:
    verb = normalize_observation_text(reference.verb)
    if verb == "put":
        narration = f" {normalize_observation_text(reference.narration)} "
        if " into " in narration or " in " in narration or " inside " in narration:
            return AtomicActionFamily.PUT_IN
    return _VERB_FAMILY.get(verb, AtomicActionFamily.OTHER)


def score_atomic_event_text(
    raw_text: str,
    *,
    reference: AtomicEventReference,
) -> AtomicEventQualityScore:
    """Score one free-form observation against a post-generation reference."""

    text = normalize_observation_text(raw_text)
    expected_verb = normalize_observation_text(reference.verb)
    action_family = _expected_action_family(reference)
    matched = _matched_families(text)
    if action_family is AtomicActionFamily.OTHER:
        action_match = _fallback_action_match(expected_verb, text)
    else:
        action_match = action_family in matched

    expected_surfaces, specificity_surfaces = _object_surfaces(reference)
    object_match = any(_contains_surface(text, surface) for surface in expected_surfaces)
    specificity_match = (
        any(_contains_surface(text, surface) for surface in specificity_surfaces)
        if specificity_surfaces
        else None
    )

    opposite_families = _OPPOSITES.get(action_family, frozenset())
    opposite_asserted = any(family in matched for family in opposite_families)
    direction_required = action_family in _DIRECTIONAL_FAMILIES
    direction_match = action_match and not opposite_asserted if direction_required else None

    any_atomic_action = bool(matched) or (
        action_family is AtomicActionFamily.OTHER and action_match
    )
    vague_motion = _matches(_VAGUE_MOTION_PATTERNS, text)
    state_cue = _matches(_STATE_PATTERNS, text)
    only_vague_family = bool(matched) and set(matched) <= {AtomicActionFamily.MOVE}
    generic_or_state_only = (not any_atomic_action and (state_cue or vague_motion or not text)) or (
        only_vague_family and action_family is not AtomicActionFamily.MOVE
    )
    adjacent_action_only = any_atomic_action and not action_match
    hedged = _matches(_HEDGE_PATTERNS, text)
    substantive_families = tuple(
        family for family in matched if family is not AtomicActionFamily.MOVE
    )
    multiple_action_story = len(set(substantive_families)) > 1

    specificity_sufficient = specificity_match is not False
    mapper_input_sufficient = bool(
        action_match
        and object_match
        and specificity_sufficient
        and direction_match is not False
        and not opposite_asserted
        and not generic_or_state_only
        and not hedged
        and not multiple_action_story
    )

    reasons: list[str] = []
    if not action_match:
        reasons.append("EXPECTED_ATOMIC_ACTION_MISSING")
    if not object_match:
        reasons.append("DIRECT_OBJECT_OR_CONTROL_MISSING")
    if specificity_match is False:
        reasons.append("OBJECT_SPECIFICITY_MISSING")
    if direction_match is False:
        reasons.append("DIRECTION_OR_STATE_CHANGE_MISSING")
    if opposite_asserted:
        reasons.append("OPPOSITE_DIRECTION_ASSERTED")
    if generic_or_state_only:
        reasons.append("GENERIC_MOTION_OR_STATE_ONLY")
    if adjacent_action_only:
        reasons.append("ADJACENT_ACTION_INSTEAD_OF_TARGET")
    if hedged:
        reasons.append("HEDGED_OR_ALTERNATIVE_ACTION")
    if multiple_action_story:
        reasons.append("MULTIPLE_ACTION_STORY")

    return AtomicEventQualityScore(
        action_family=action_family,
        atomic_action_match=action_match,
        object_match=object_match,
        object_specificity_match=specificity_match,
        direction_required=direction_required,
        direction_match=direction_match,
        opposite_direction_asserted=opposite_asserted,
        generic_or_state_only=generic_or_state_only,
        adjacent_action_only=adjacent_action_only,
        hedged=hedged,
        multiple_action_story=multiple_action_story,
        mapper_input_sufficient=mapper_input_sufficient,
        matched_action_families=tuple(family.value for family in matched),
        expected_object_surfaces=expected_surfaces,
        expected_specificity_surfaces=specificity_surfaces,
        reasons=tuple(reasons),
    )


def score_raw_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Score one retained runner row while preserving its human-readable fields."""

    reference = reference_from_record(record)
    raw_text = record.get("raw_output_text")
    if raw_text is None:
        raw_text = record.get("output_text")
    if not isinstance(raw_text, str):
        raw_text = ""
    score = score_atomic_event_text(raw_text, reference=reference)
    return {
        "uid": record.get("uid") or record.get("case_id"),
        "video_id": record.get("video_id"),
        "expected": {
            "verb": reference.verb,
            "noun": reference.noun,
            "narration": reference.narration,
        },
        "raw_output_text": raw_text,
        "quality": score.to_dict(),
    }


def summarize_atomic_event_scores(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate case projections without hiding per-family failure modes."""

    total = len(cases)
    metric_names = (
        "atomic_action_match",
        "object_match",
        "generic_or_state_only",
        "adjacent_action_only",
        "hedged",
        "multiple_action_story",
        "mapper_input_sufficient",
    )
    counts = Counter[str]()
    reason_counts = Counter[str]()
    direction_total = 0
    direction_match_count = 0
    family_rows: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        quality = case.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError("every case must contain a quality mapping")
        for name in metric_names:
            counts[name] += bool(quality.get(name))
        if quality.get("direction_required") is True:
            direction_total += 1
            direction_match_count += quality.get("direction_match") is True
        for reason in quality.get("reasons", ()):
            reason_counts[str(reason)] += 1
        family = str(quality.get("action_family") or AtomicActionFamily.OTHER.value)
        family_rows.setdefault(family, []).append(quality)

    def ratio(count: int, denominator: int = total) -> float | None:
        return count / denominator if denominator else None

    by_family: dict[str, Any] = {}
    for family, rows in sorted(family_rows.items()):
        family_count = len(rows)
        by_family[family] = {
            "case_count": family_count,
            "atomic_action_match_count": sum(bool(row.get("atomic_action_match")) for row in rows),
            "object_match_count": sum(bool(row.get("object_match")) for row in rows),
            "mapper_input_sufficient_count": sum(
                bool(row.get("mapper_input_sufficient")) for row in rows
            ),
            "mapper_input_sufficient_rate": ratio(
                sum(bool(row.get("mapper_input_sufficient")) for row in rows), family_count
            ),
        }

    return {
        "evaluation_version": RAW_ATOMIC_EVENT_QUALITY_VERSION,
        "case_count": total,
        "atomic_action_match_count": counts["atomic_action_match"],
        "atomic_action_match_rate": ratio(counts["atomic_action_match"]),
        "object_match_count": counts["object_match"],
        "object_match_rate": ratio(counts["object_match"]),
        "directional_case_count": direction_total,
        "direction_match_count": direction_match_count,
        "direction_match_rate": ratio(direction_match_count, direction_total),
        "generic_or_state_only_count": counts["generic_or_state_only"],
        "adjacent_action_only_count": counts["adjacent_action_only"],
        "hedged_count": counts["hedged"],
        "multiple_action_story_count": counts["multiple_action_story"],
        "mapper_input_sufficient_count": counts["mapper_input_sufficient"],
        "mapper_input_sufficient_rate": ratio(counts["mapper_input_sufficient"]),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "by_action_family": by_family,
    }


def evaluate_raw_output_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score and summarize a runner output sequence."""

    cases = [score_raw_output_record(record) for record in records]
    return {"summary": summarize_atomic_event_scores(cases), "cases": cases}
