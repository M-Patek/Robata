"""Zero-GPU structured-label clip search MVP.



The MVP indexes annotation segments only.  It performs deterministic verb-family normalization,

faceted filtering, and direct clip playback targets; embedding/vector providers can be added later

behind the same ``ClipSearchIndex`` port without changing callers.

"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import AliasChoices, Field, StringConstraints, model_validator

from robata.annotation import AnnotationSegmentDraft, StructuredLabels
from robata.contracts.common import StrictModel
from robata.qa import ClipMark

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]

NonNegative = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


# Manual mapping is intentionally explicit and reviewable.  It covers common annotation verbs and

# keeps the provider-independent MVP deterministic (synonyms resolve to one canonical family).

VERB_FAMILY_MAP: dict[str, str] = {
    "wipe": "clean",
    "wiping": "clean",
    "scrub": "clean",
    "scrubbing": "clean",
    "wash": "clean",
    "washing": "clean",
    "clean": "clean",
    "cleaning": "clean",
    "brush": "clean",
    "brushing": "clean",
    "polish": "clean",
    "polishing": "clean",
    "cut": "cut",
    "cutting": "cut",
    "chop": "cut",
    "chopping": "cut",
    "slice": "cut",
    "slicing": "cut",
    "dice": "cut",
    "dicing": "cut",
    "peel": "cut",
    "peeling": "cut",
    "open": "open",
    "opening": "open",
    "close": "close",
    "closing": "close",
    "pick": "pick",
    "picking": "pick",
    "grab": "pick",
    "grabbing": "pick",
    "take": "pick",
    "put": "place",
    "placing": "place",
    "place": "place",
    "set": "place",
    "setting": "place",
    "move": "move",
    "moving": "move",
    "carry": "move",
    "carrying": "move",
    "bring": "move",
    "hold": "hold",
    "holding": "hold",
    "grip": "hold",
    "gripping": "hold",
    "press": "press",
    "pressing": "press",
    "push": "press",
    "pushing": "press",
    "pull": "pull",
    "pulling": "pull",
    "drag": "pull",
    "dragging": "pull",
    "turn": "rotate",
    "turning": "rotate",
    "twist": "rotate",
    "twisting": "rotate",
    "rotate": "rotate",
    "rotating": "rotate",
    "spin": "rotate",
    "spinning": "rotate",
    "insert": "insert",
    "inserting": "insert",
    "remove": "remove",
    "removing": "remove",
    "attach": "attach",
    "attaching": "attach",
    "detach": "detach",
    "detaching": "detach",
    "pour": "pour",
    "pouring": "pour",
    "fill": "fill",
    "filling": "fill",
    "empty": "empty",
    "emptying": "empty",
    "stir": "mix",
    "stirring": "mix",
    "mix": "mix",
    "mixing": "mix",
    "shake": "shake",
    "shaking": "shake",
    "fold": "fold",
    "folding": "fold",
    "unfold": "fold",
    "unfolding": "fold",
    "write": "write",
    "writing": "write",
    "draw": "draw",
    "drawing": "draw",
    "point": "point",
    "pointing": "point",
    "look": "look",
    "looking": "look",
    "touch": "touch",
    "touching": "touch",
    "scan": "scan",
    "scanning": "scan",
    "measure": "measure",
    "measuring": "measure",
    "interact": "interaction",
    "interacting": "interaction",
    "interaction": "interaction",
    # Common Chinese labels used by the annotation UI.
    "擦": "clean",
    "擦拭": "clean",
    "清洁": "clean",
    "清洗": "clean",
    "洗": "clean",
    "切": "cut",
    "切割": "cut",
    "拿": "pick",
    "抓": "pick",
    "放": "place",
    "打开": "open",
    "关闭": "close",
    "倒": "pour",
    "搅拌": "mix",
    "按": "press",
    "推": "press",
    "拉": "pull",
}


# Aliases useful for callers that use the existing lemmatizer vocabulary.

VERB_LEMMAS = VERB_FAMILY_MAP

_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "with",
    "and",
    "or",
    "clip",
    "clips",
    "video",
    "videos",
    "show",
    "me",
    "find",
    "where",
    "that",
    "which",
    "something",
    "??",
    "?",
}

_ATTRIBUTE_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "small",
    "large",
    "big",
    "metal",
    "plastic",
    "wooden",
    "dirty",
    "clean",
    "fast",
    "slow",
    "inside",
    "outside",
    "front",
    "back",
    "top",
    "bottom",
}

_LOCATION_WORDS = {
    "left",
    "right",
    "center",
    "middle",
    "inside",
    "outside",
    "table",
    "floor",
    "room",
}

_HAND_WORDS = {"left", "right", "both", "either", "??"}


class ClipIndexEntry(StrictModel):
    """One searchable annotation segment."""

    clip_id: NonEmptyString

    video_id: NonEmptyString

    start_sec: NonNegative = Field(
        validation_alias=AliasChoices("start_sec", "start_time_sec"),
        serialization_alias="start_sec",
    )

    end_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] = Field(
        validation_alias=AliasChoices("end_sec", "end_time_sec"),
        serialization_alias="end_sec",
    )

    structured_labels: StructuredLabels

    source_uri: NonEmptyString | None = None

    qa_clip_marks: tuple[ClipMark, ...] = ()

    @model_validator(mode="after")
    def _validate_interval(self) -> ClipIndexEntry:

        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")

        return self

    @property
    def verb_family(self) -> str:

        return VerbNormalizer.normalize(self.structured_labels.verb)

    @property
    def playback_target(self) -> str:

        base = self.source_uri or self.video_id

        separator = "&" if "?" in base else "?"

        return f"{base}{separator}start={self.start_sec:g}&end={self.end_sec:g}"


class SearchQuery(StrictModel):
    """Parsed natural-language or faceted query."""

    text: str = ""

    verb_family: str | None = None

    noun: str | None = None

    attributes: tuple[str, ...] = ()

    location: str | None = None

    hand: str | None = None


class SearchHit(StrictModel):
    """Directly playable clip result."""

    clip_id: NonEmptyString

    video_id: NonEmptyString

    start_sec: NonNegative = Field(
        validation_alias=AliasChoices("start_sec", "start_time_sec"),
        serialization_alias="start_sec",
    )

    end_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] = Field(
        validation_alias=AliasChoices("end_sec", "end_time_sec"),
        serialization_alias="end_sec",
    )

    verb: NonEmptyString

    verb_family: NonEmptyString

    noun: NonEmptyString

    attributes: tuple[NonEmptyString, ...] = ()

    location: str | None = None

    hand: str | None = None

    score: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]

    playback_target: NonEmptyString

    qa_clip_marks: tuple[ClipMark, ...] = ()

    @property
    def start_time_sec(self) -> float:

        return self.start_sec

    @property
    def end_time_sec(self) -> float:

        return self.end_sec

    @property
    def clip_start(self) -> float:

        return self.start_sec

    @property
    def clip_end(self) -> float:

        return self.end_sec

    @property
    def playback_url(self) -> str:

        return self.playback_target


class VerbNormalizer:
    """Normalize free-text verbs into stable action families."""

    @staticmethod
    def normalize(verb: str) -> str:

        if not isinstance(verb, str) or not verb.strip():
            raise ValueError("verb must be a non-empty string")

        raw = verb.strip().casefold()

        if raw in VERB_FAMILY_MAP:
            return VERB_FAMILY_MAP[raw]

        token = re.sub(r"[^a-z0-9_-]+", " ", raw).strip()

        if token in VERB_FAMILY_MAP:
            return VERB_FAMILY_MAP[token]

        # Basic morphology for unseen words; preserve unknown values as their own family so the

        # index remains explainable instead of dropping a segment.

        if token.endswith("ing") and token[:-3] in VERB_FAMILY_MAP:
            return VERB_FAMILY_MAP[token[:-3]]

        if token.endswith("ed") and token[:-2] in VERB_FAMILY_MAP:
            return VERB_FAMILY_MAP[token[:-2]]

        return token

    @staticmethod
    def normalize_many(values: Iterable[str]) -> tuple[str, ...]:

        return tuple(dict.fromkeys(VerbNormalizer.normalize(value) for value in values))


class NaturalLanguageQueryParser:
    """Small deterministic parser for the zero-GPU MVP."""

    @staticmethod
    def parse(query: str | Mapping[str, Any] | SearchQuery) -> SearchQuery:

        if isinstance(query, SearchQuery):
            return query

        if isinstance(query, Mapping):
            payload = dict(query)

            if payload.get("verb") and not payload.get("verb_family"):
                payload["verb_family"] = VerbNormalizer.normalize(str(payload["verb"]))

            if payload.get("attributes") is not None and isinstance(payload["attributes"], str):
                payload["attributes"] = (payload["attributes"],)

            return SearchQuery.model_validate(payload)

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty text, mapping, or SearchQuery")

        text = query.strip()

        folded_text = text.casefold()

        tokens = [token for token in re.findall(r"[\w-]+", folded_text, flags=re.UNICODE) if token]

        verb_family: str | None = None

        consumed: set[int] = set()

        for alias in sorted(
            (key for key in VERB_FAMILY_MAP if not key.isascii()),
            key=len,
            reverse=True,
        ):
            if alias in folded_text:
                verb_family = VERB_FAMILY_MAP[alias]

                folded_text = folded_text.replace(alias, " ", 1)

                tokens = [
                    token for token in re.findall(r"[\w-]+", folded_text, flags=re.UNICODE) if token
                ]

                break

        for index, token in enumerate(tokens):
            if token in VERB_FAMILY_MAP:
                verb_family = VERB_FAMILY_MAP[token]

                consumed.add(index)

                break

        attributes = tuple(dict.fromkeys(token for token in tokens if token in _ATTRIBUTE_WORDS))

        # A bare "table" is usually the noun ("wipe table"); treat it as a location only

        # when introduced by a preposition ("on the table").

        location = None

        location_indices: set[int] = set()

        for index, token in enumerate(tokens):
            if token not in _LOCATION_WORDS:
                continue

            previous = tokens[index - 1] if index else ""

            if token in {"table", "floor", "room"} and previous not in {
                "at",
                "on",
                "in",
                "inside",
                "outside",
            }:
                continue

            if token in {"left", "right"} and index == 0:
                continue

            location = token

            location_indices.add(index)

            break

        hand = next((token for token in tokens if token in _HAND_WORDS), None)

        noun_tokens = [
            token
            for index, token in enumerate(tokens)
            if index not in consumed
            and index not in location_indices
            and token not in _STOPWORDS
            and token not in _ATTRIBUTE_WORDS
            and token not in _HAND_WORDS
        ]

        noun = noun_tokens[0] if noun_tokens else None

        return SearchQuery(
            text=text,
            verb_family=verb_family,
            noun=noun,
            attributes=attributes,
            location=location,
            hand=hand,
        )


@dataclass(frozen=True, slots=True)
class _ScoredEntry:
    entry: ClipIndexEntry

    score: float


class ClipSearchIndex:
    """In-memory deterministic clip index suitable for an MVP and unit tests."""

    def __init__(
        self, entries: Iterable[ClipIndexEntry | AnnotationSegmentDraft | Mapping[str, Any]] = ()
    ) -> None:

        self._entries: dict[str, ClipIndexEntry] = {}

        self._parser = NaturalLanguageQueryParser()

        self.add_many(entries)

    def add(
        self, entry: ClipIndexEntry | AnnotationSegmentDraft | Mapping[str, Any]
    ) -> ClipIndexEntry:

        normalized = self._coerce_entry(entry)

        self._entries[normalized.clip_id] = normalized

        return normalized

    upsert = add

    def add_many(
        self, entries: Iterable[ClipIndexEntry | AnnotationSegmentDraft | Mapping[str, Any]]
    ) -> int:

        count = 0

        for entry in entries:
            self.add(entry)

            count += 1

        return count

    def remove(self, clip_id: str) -> bool:

        return self._entries.pop(clip_id, None) is not None

    def get(self, clip_id: str) -> ClipIndexEntry:

        try:
            return self._entries[clip_id]

        except KeyError as exc:
            raise KeyError(f"unknown clip_id: {clip_id}") from exc

    def entries(self) -> tuple[ClipIndexEntry, ...]:

        return tuple(self._entries[key] for key in sorted(self._entries))

    def search(
        self,
        query: str | Mapping[str, Any] | SearchQuery,
        *,
        limit: int = 50,
        video_id: str | None = None,
        include_warning: bool = True,
    ) -> tuple[SearchHit, ...]:

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        parsed = self._parser.parse(query)

        scored: list[_ScoredEntry] = []

        for entry in self._entries.values():
            if video_id is not None and entry.video_id != video_id:
                continue

            if not include_warning and entry.qa_clip_marks:
                continue

            score = self._score(entry, parsed)

            if score <= 0:
                continue

            scored.append(_ScoredEntry(entry, score))

        scored.sort(
            key=lambda item: (
                -item.score,
                item.entry.video_id,
                item.entry.start_sec,
                item.entry.clip_id,
            )
        )

        return tuple(self._to_hit(item) for item in scored[:limit])

    query = search

    def filter(
        self,
        *,
        verb_family: str | None = None,
        noun: str | None = None,
        attributes: Sequence[str] = (),
        location: str | None = None,
        hand: str | None = None,
        limit: int = 50,
    ) -> tuple[SearchHit, ...]:
        return self.search(
            SearchQuery(
                verb_family=VerbNormalizer.normalize(verb_family) if verb_family else None,
                noun=noun,
                attributes=tuple(attributes),
                location=location,
                hand=hand,
            ),
            limit=limit,
        )

    search_text = search
    search_facets = filter

    def facet_values(self) -> dict[str, tuple[str, ...]]:

        entries = self.entries()

        return {
            "verb_family": tuple(sorted({entry.verb_family for entry in entries})),
            "noun": tuple(sorted({entry.structured_labels.noun.casefold() for entry in entries})),
            "attribute": tuple(
                sorted(
                    {
                        attr.casefold()
                        for entry in entries
                        for attr in entry.structured_labels.attributes
                    }
                )
            ),
            "location": tuple(
                sorted(
                    {
                        entry.structured_labels.location.casefold()
                        for entry in entries
                        if entry.structured_labels.location
                    }
                )
            ),
            "hand": tuple(
                sorted(
                    {
                        entry.structured_labels.hand.casefold()
                        for entry in entries
                        if entry.structured_labels.hand
                    }
                )
            ),
        }

    def _coerce_entry(
        self, value: ClipIndexEntry | AnnotationSegmentDraft | Mapping[str, Any]
    ) -> ClipIndexEntry:

        if isinstance(value, ClipIndexEntry):
            return value

        if isinstance(value, AnnotationSegmentDraft):
            return ClipIndexEntry(
                clip_id=value.segment_id,
                video_id=value.video_id,
                start_sec=value.start_sec,
                end_sec=value.end_sec,
                structured_labels=value.structured_labels,
                qa_clip_marks=value.qa_clip_marks,
                source_uri=None,
            )

        if isinstance(value, Mapping):
            payload = dict(value)

            if "clip_id" not in payload and "segment_id" in payload:
                payload["clip_id"] = payload.pop("segment_id")

            if "structured_labels" not in payload:
                payload["structured_labels"] = {
                    key: payload.pop(key)
                    for key in ("verb", "noun", "attributes", "location", "hand")
                    if key in payload
                }

            return ClipIndexEntry.model_validate(payload)

        raise TypeError("entry must be ClipIndexEntry, AnnotationSegmentDraft, or mapping")

    @staticmethod
    def _score(entry: ClipIndexEntry, query: SearchQuery) -> float:

        labels = entry.structured_labels

        score = 1.0  # unconstrained text returns all clips deterministically

        if query.verb_family:
            if entry.verb_family != query.verb_family:
                return 0.0

            score += 5.0

        if query.noun:
            if labels.noun.casefold() != query.noun.casefold():
                return 0.0

            score += 4.0

        if query.attributes:
            available = {value.casefold() for value in labels.attributes}

            matched = sum(attribute.casefold() in available for attribute in query.attributes)

            if matched == 0:
                return 0.0

            score += 2.0 * matched / len(query.attributes)

        if query.location:
            if not labels.location or labels.location.casefold() != query.location.casefold():
                return 0.0

            score += 1.5

        if query.hand:
            if not labels.hand or labels.hand.casefold() != query.hand.casefold():
                return 0.0

            score += 1.5

        return score

    @staticmethod
    def _to_hit(scored: _ScoredEntry) -> SearchHit:

        labels = scored.entry.structured_labels

        return SearchHit(
            clip_id=scored.entry.clip_id,
            video_id=scored.entry.video_id,
            start_sec=scored.entry.start_sec,
            end_sec=scored.entry.end_sec,
            verb=labels.verb,
            verb_family=scored.entry.verb_family,
            noun=labels.noun,
            attributes=labels.attributes,
            location=labels.location,
            hand=labels.hand,
            score=scored.score,
            playback_target=scored.entry.playback_target,
            qa_clip_marks=scored.entry.qa_clip_marks,
        )


StructuredClipIndex = ClipSearchIndex
ClipIndex = ClipSearchIndex
SearchService = ClipSearchIndex
VideoClipSearch = ClipSearchIndex
VideoSearchMVP = ClipSearchIndex


__all__ = [
    "VERB_FAMILY_MAP",
    "VERB_LEMMAS",
    "ClipIndex",
    "ClipIndexEntry",
    "ClipSearchIndex",
    "NaturalLanguageQueryParser",
    "SearchHit",
    "SearchQuery",
    "SearchService",
    "StructuredClipIndex",
    "VerbNormalizer",
    "VideoClipSearch",
    "VideoSearchMVP",
]
