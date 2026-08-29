"""Benchmark-local WeMM visual-to-EPIC joint-action retrieval.

This module is deliberately separate from the production retrieval service and
the existing semantic mapper.  It answers one narrow research question:
whether a shared WeMM multimodal representation can recover the correct EPIC
``(verb, noun)`` action candidate from a visual clip, and whether that signal
adds anything to the current text/lexical retrieval score.

The implementation has no model dependency.  A small encoder protocol makes
the ranking and metric code runnable with deterministic test doubles; the
optional Transformers backend lives in :mod:`wemm_embedding_backend`.
Ground-truth fields are accepted only by the post-hoc metric functions.  They
are never part of the encoder input.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal


def _shortlist_score(hint: str, candidate: str) -> float:
    """Score a free-text hint against a candidate without external helpers.

    WeMM's text/hybrid diagnostic path only needs a small lexical baseline.
    Keeping it local avoids coupling the embedding experiment to the older
    EPIC ontology identity module (and keeps the benchmark free of digest
    generation).
    """

    left = re.sub(r"[^a-z0-9 ]+", "", _text(hint).casefold()).strip()
    right = re.sub(r"[^a-z0-9 ]+", "", _text(candidate).casefold()).strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    sequence = difflib.SequenceMatcher(None, left, right).ratio()
    return max(overlap, sequence * 0.9)


WEMM_RETRIEVAL_VERSION = "wemm-epic-joint-retrieval-v1"
WEMM_LABEL_TEXT_VERSION = "wemm-epic-label-text-v1"
LABEL_VARIANTS = ("canonical", "verb_noun", "natural")
LabelVariant = Literal["canonical", "verb_noun", "natural"]
RetrievalMode = Literal["visual", "text", "hybrid"]


class WemmRetrievalError(ValueError):
    """Raised when a benchmark retrieval input is malformed."""


def _coerce_integer(value: object, *, field: str) -> int:
    """Coerce an integer-shaped value without silently truncating floats/bools.

    The benchmark accepts integer strings because class tables and manifests are
    commonly read from CSV/JSON, but values such as ``1.5`` and ``True`` are
    malformed identifiers rather than valid class IDs.  Keeping this conversion
    in one place also ensures malformed inputs consistently raise the public
    benchmark error instead of leaking ``TypeError``/``ValueError``.
    """

    if isinstance(value, bool):
        raise WemmRetrievalError(f"{field} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    raise WemmRetrievalError(f"{field} must be an integer")


def _coerce_float(value: object, *, field: str) -> float:
    """Coerce a finite numeric value while normalising error reporting."""

    if isinstance(value, bool):
        raise WemmRetrievalError(f"{field} must be a finite number")
    try:
        numeric_value: Any = value
        converted = float(numeric_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmRetrievalError(f"{field} must be a finite number") from exc
    if not math.isfinite(converted):
        raise WemmRetrievalError(f"{field} must be a finite number")
    return converted


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().replace("_", " ").replace("-", " ").split())


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", _text(value).casefold()).strip()


def _article(noun: str) -> str:
    return "an" if noun[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"


_IRREGULAR_GERUNDS = {
    "put": "putting",
    "take": "taking",
    "make": "making",
    "use": "using",
    "close": "closing",
    "open": "opening",
    "place": "placing",
    "pick": "picking",
    "turn": "turning",
    "pour": "pouring",
    "wash": "washing",
}


def _gerund(verb: str) -> str:
    """Render a readable, deterministic natural-language verb surface."""

    words = _text(verb).split()
    if not words:
        return "performing"
    if len(words) > 1:
        # EPIC contains phrasal verbs such as ``put-on`` and ``turn-off``.
        # Inflect the lexical head while retaining the particle.
        return f"{_gerund(' '.join(words[:-1]))} {words[-1]}"
    last = words[-1]
    if last in _IRREGULAR_GERUNDS:
        words[-1] = _IRREGULAR_GERUNDS[last]
    elif last.endswith("ie"):
        words[-1] = last[:-2] + "ying"
    elif last.endswith("e") and not last.endswith("ee"):
        words[-1] = last[:-1] + "ing"
    else:
        words[-1] = last + "ing"
    return " ".join(words)


def render_action_label_texts(verb_key: str, noun_key: str) -> dict[str, str]:
    """Return the three pre-registered text surfaces used by the experiment."""

    verb = _text(verb_key)
    noun = _text(noun_key)
    if not verb or not noun:
        raise WemmRetrievalError("verb and noun keys must be non-empty")
    return {
        "canonical": f"{verb} {noun}",
        "verb_noun": f"verb: {verb}; noun: {noun}",
        "natural": f"a person is {_gerund(verb)} {_article(noun)} {noun}",
    }


def _entries(table_or_entries: object, *, kind: str) -> dict[int, str]:
    raw = getattr(table_or_entries, "entries", table_or_entries)
    if not isinstance(raw, Mapping):
        raise WemmRetrievalError(f"{kind} entries must be a mapping")
    result: dict[int, str] = {}
    for key, value in raw.items():
        try:
            class_id = _coerce_integer(key, field=f"{kind} IDs")
        except WemmRetrievalError as exc:
            raise WemmRetrievalError(f"invalid {kind} class ID: {key!r}") from exc
        label = _text(value)
        if class_id < 0 or not label:
            raise WemmRetrievalError(f"invalid {kind} class entry: {key!r} -> {value!r}")
        if class_id in result:
            raise WemmRetrievalError(f"duplicate {kind} class ID: {class_id}")
        result[class_id] = label
    if not result:
        raise WemmRetrievalError(f"{kind} entries are empty")
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class JointActionLabel:
    """One immutable, benchmark-local joint action candidate."""

    verb_id: int
    noun_id: int
    verb_key: str
    noun_key: str
    texts: tuple[tuple[str, str], ...]
    observed_count: int = 0

    @property
    def action_key(self) -> tuple[int, int]:
        return (self.verb_id, self.noun_id)

    def text_for(self, variant: str) -> str:
        variant = str(variant)
        for name, value in self.texts:
            if name == variant:
                return value
        raise WemmRetrievalError(f"unsupported label variant: {variant!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb_id": self.verb_id,
            "noun_id": self.noun_id,
            "verb_key": self.verb_key,
            "noun_key": self.noun_key,
            "action_key": [self.verb_id, self.noun_id],
            "texts": {name: value for name, value in self.texts},
            "observed_count": self.observed_count,
        }


def build_joint_action_catalog(
    *,
    verb_table_or_entries: object,
    noun_table_or_entries: object,
    action_pairs: Iterable[tuple[int, int]],
    observed_counts: Mapping[tuple[int, int], int] | None = None,
) -> tuple[JointActionLabel, ...]:
    """Build a catalog without changing either official class table.

    ``action_pairs`` is intentionally explicit.  Callers can provide pairs
    observed in a training/ontology artifact, or use a full Cartesian product
    when that is the declared experiment.  The evaluator never derives this
    catalog from the current row's ground truth.
    """

    if isinstance(action_pairs, (str, bytes, bytearray)):
        raise WemmRetrievalError("action_pairs must be an iterable of ID pairs")
    try:
        pair_iter = iter(action_pairs)
    except TypeError as exc:
        raise WemmRetrievalError("action_pairs must be an iterable of ID pairs") from exc
    verbs = _entries(verb_table_or_entries, kind="verb")
    nouns = _entries(noun_table_or_entries, kind="noun")
    if observed_counts is not None and not isinstance(observed_counts, Mapping):
        raise WemmRetrievalError("observed_counts must be a mapping")
    counts = observed_counts or {}
    selected: set[tuple[int, int]] = set()
    labels: list[JointActionLabel] = []
    for raw_pair in pair_iter:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            raise WemmRetrievalError(f"action pair must contain two IDs: {raw_pair!r}")
        try:
            pair = (
                _coerce_integer(raw_pair[0], field="verb ID"),
                _coerce_integer(raw_pair[1], field="noun ID"),
            )
        except WemmRetrievalError as exc:
            raise WemmRetrievalError(f"invalid action pair: {raw_pair!r}") from exc
        if pair in selected:
            continue
        if pair[0] not in verbs or pair[1] not in nouns:
            raise WemmRetrievalError(f"action pair references an unknown class: {pair}")
        count = counts.get(pair, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise WemmRetrievalError(f"observed count must be a non-negative integer: {pair}")
        surfaces = render_action_label_texts(verbs[pair[0]], nouns[pair[1]])
        labels.append(
            JointActionLabel(
                verb_id=pair[0],
                noun_id=pair[1],
                verb_key=verbs[pair[0]],
                noun_key=nouns[pair[1]],
                texts=tuple((name, surfaces[name]) for name in LABEL_VARIANTS),
                observed_count=count,
            )
        )
        selected.add(pair)
    if not labels:
        raise WemmRetrievalError("action catalog is empty")
    return tuple(sorted(labels, key=lambda item: item.action_key))


def full_cartesian_action_pairs(
    verb_table_or_entries: object, noun_table_or_entries: object
) -> tuple[tuple[int, int], ...]:
    """Return all verb x noun pairs as an explicit, non-mutating projection."""

    verbs = _entries(verb_table_or_entries, kind="verb")
    nouns = _entries(noun_table_or_entries, kind="noun")
    return tuple((verb_id, noun_id) for verb_id in verbs for noun_id in nouns)


def _as_rows(value: object) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WemmRetrievalError("encoder output must be a two-dimensional sequence")
    rows: list[list[float]] = []
    for raw_row in value:
        if hasattr(raw_row, "tolist"):
            raw_row = raw_row.tolist()
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes, bytearray)):
            raise WemmRetrievalError("encoder returned a malformed embedding row")
        try:
            if any(isinstance(item, bool) for item in raw_row):
                raise WemmRetrievalError("encoder returned a non-numeric embedding")
            row = [float(item) for item in raw_row]
        except WemmRetrievalError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise WemmRetrievalError("encoder returned a non-numeric embedding") from exc
        if not row or any(not math.isfinite(item) for item in row):
            raise WemmRetrievalError("encoder returned a non-finite or empty embedding")
        norm = math.sqrt(sum(item * item for item in row))
        if not math.isfinite(norm) or norm <= 0.0:
            raise WemmRetrievalError("encoder returned a zero-norm embedding")
        rows.append([item / norm for item in row])
    return rows


def validate_embedding_matrix(
    value: object, *, expected_rows: int
) -> tuple[tuple[float, ...], ...]:
    """Coerce and L2-normalize encoder output for deterministic cosine scoring."""

    try:
        expected = _coerce_integer(expected_rows, field="expected_rows")
    except WemmRetrievalError as exc:
        raise WemmRetrievalError("expected_rows must be a non-negative integer") from exc
    if expected < 0:
        raise WemmRetrievalError("expected_rows must be a non-negative integer")
    rows = _as_rows(value)
    if len(rows) != expected:
        raise WemmRetrievalError(f"encoder returned {len(rows)} rows; expected {expected}")
    dimensions = len(rows[0]) if rows else None
    if dimensions is not None and any(len(row) != dimensions for row in rows):
        raise WemmRetrievalError("encoder returned inconsistent embedding dimensions")
    return tuple(tuple(row) for row in rows)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if (
        not isinstance(left, Sequence)
        or isinstance(left, (str, bytes, bytearray))
        or not isinstance(right, Sequence)
        or isinstance(right, (str, bytes, bytearray))
    ):
        raise WemmRetrievalError("cosine inputs must be numeric sequences")
    if len(left) != len(right) or not left:
        raise WemmRetrievalError("embedding dimensions do not match")
    try:
        if any(isinstance(item, bool) for item in (*left, *right)):
            raise WemmRetrievalError("cosine inputs must be numeric")
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except WemmRetrievalError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmRetrievalError("cosine inputs must be numeric") from exc
    if any(not math.isfinite(item) for item in (*left_values, *right_values)):
        raise WemmRetrievalError("cosine inputs must be finite")
    left_norm = math.sqrt(sum(item * item for item in left_values))
    right_norm = math.sqrt(sum(item * item for item in right_values))
    if (
        not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
        or left_norm <= 0.0
        or right_norm <= 0.0
    ):
        raise WemmRetrievalError("cosine inputs must have non-zero norm")
    score = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    if not math.isfinite(score):
        raise WemmRetrievalError("cosine inputs produced a non-finite score")
    return max(-1.0, min(1.0, score / (left_norm * right_norm)))


def _unit_cosine(score: float) -> float:
    return (max(-1.0, min(1.0, float(score))) + 1.0) / 2.0


@dataclass(frozen=True, slots=True)
class RetrievedAction:
    """One ranked candidate with raw and fused scores retained."""

    rank: int
    label: JointActionLabel
    label_variant: str
    visual_cosine: float | None
    visual_score: float | None
    text_score: float | None
    fused_score: float

    @property
    def action_key(self) -> tuple[int, int]:
        return self.label.action_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "action_key": list(self.action_key),
            "verb_id": self.label.verb_id,
            "noun_id": self.label.noun_id,
            "verb_key": self.label.verb_key,
            "noun_key": self.label.noun_key,
            "label_text": self.label.text_for(self.label_variant),
            "label_variant": self.label_variant,
            "visual_cosine": self.visual_cosine,
            "visual_score": self.visual_score,
            "text_score": self.text_score,
            "fused_score": self.fused_score,
        }


def rank_joint_actions(
    *,
    labels: Sequence[JointActionLabel],
    query_embedding: Sequence[float] | None = None,
    label_embeddings: Mapping[tuple[int, int], Sequence[float]] | None = None,
    label_variant: LabelVariant = "canonical",
    text_scores: Mapping[tuple[int, int], float] | None = None,
    mode: RetrievalMode = "visual",
    visual_weight: float = 1.0,
    text_weight: float = 0.0,
    top_k: int | None = None,
) -> tuple[RetrievedAction, ...]:
    """Rank joint candidates for visual-only, text-only, or weighted hybrid use."""

    if not isinstance(label_variant, str) or label_variant not in LABEL_VARIANTS:
        raise WemmRetrievalError(f"unsupported label variant: {label_variant!r}")
    if not isinstance(mode, str) or mode not in {"visual", "text", "hybrid"}:
        raise WemmRetrievalError(f"unsupported retrieval mode: {mode!r}")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)):
        raise WemmRetrievalError("labels must be a sequence")
    if not labels:
        raise WemmRetrievalError("labels must be non-empty")
    if any(not isinstance(label, JointActionLabel) for label in labels):
        raise WemmRetrievalError("labels contain a malformed joint action")
    if len({label.action_key for label in labels}) != len(labels):
        raise WemmRetrievalError("labels contain duplicate action keys")
    if mode in {"visual", "hybrid"} and (query_embedding is None or label_embeddings is None):
        raise WemmRetrievalError("visual and hybrid modes require embeddings")
    if mode in {"text", "hybrid"} and text_scores is None:
        raise WemmRetrievalError("text and hybrid modes require text scores")
    if mode in {"visual", "hybrid"} and not isinstance(label_embeddings, Mapping):
        raise WemmRetrievalError("label_embeddings must be a mapping")
    if mode in {"text", "hybrid"} and not isinstance(text_scores, Mapping):
        raise WemmRetrievalError("text_scores must be a mapping")
    try:
        visual_weight = _coerce_float(visual_weight, field="visual_weight")
        text_weight = _coerce_float(text_weight, field="text_weight")
    except WemmRetrievalError as exc:
        raise WemmRetrievalError("fusion weights must be finite and non-negative") from exc
    if visual_weight < 0.0 or text_weight < 0.0:
        raise WemmRetrievalError("fusion weights must be finite and non-negative")
    if mode == "visual":
        visual_weight, text_weight = 1.0, 0.0
    elif mode == "text":
        visual_weight, text_weight = 0.0, 1.0
    elif visual_weight + text_weight <= 0.0:
        raise WemmRetrievalError("hybrid weights cannot both be zero")
    total_weight = visual_weight + text_weight
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise WemmRetrievalError("fusion weights must have a finite positive sum")
    text_score_table = text_scores

    scored: list[RetrievedAction] = []
    for label in labels:
        key = label.action_key
        visual_cosine: float | None = None
        visual_score: float | None = None
        if mode in {"visual", "hybrid"}:
            if key not in label_embeddings:  # type: ignore[operator]
                raise WemmRetrievalError(f"missing label embedding for action {key}")
            visual_cosine = cosine_similarity(query_embedding, label_embeddings[key])  # type: ignore[arg-type,index]
            visual_score = _unit_cosine(visual_cosine)
        text_score: float | None = None
        if mode in {"text", "hybrid"}:
            assert text_score_table is not None
            try:
                raw_text_score = _coerce_float(
                    text_score_table.get(key, 0.0), field=f"text score for {key}"
                )
            except WemmRetrievalError as exc:
                raise WemmRetrievalError(f"text score for {key} is non-finite") from exc
            text_score = max(0.0, min(1.0, raw_text_score))
        fused = (
            (visual_weight * (visual_score if visual_score is not None else 0.0))
            + (text_weight * (text_score if text_score is not None else 0.0))
        ) / total_weight
        if not math.isfinite(fused):
            raise WemmRetrievalError(f"fused score for {key} is non-finite")
        scored.append(
            RetrievedAction(
                rank=0,
                label=label,
                label_variant=label_variant,
                visual_cosine=visual_cosine,
                visual_score=visual_score,
                text_score=text_score,
                fused_score=fused,
            )
        )
    scored.sort(
        key=lambda item: (
            -item.fused_score,
            -(item.visual_score if item.visual_score is not None else -1.0),
            -(item.text_score if item.text_score is not None else -1.0),
            item.label.verb_id,
            item.label.noun_id,
        )
    )
    if top_k is None:
        limit = len(scored)
    else:
        try:
            limit = _coerce_integer(top_k, field="top_k")
        except WemmRetrievalError as exc:
            raise WemmRetrievalError("top_k must be a positive integer") from exc
    if limit <= 0:
        raise WemmRetrievalError("top_k must be a positive integer")
    return tuple(
        RetrievedAction(
            rank=index,
            label=item.label,
            label_variant=item.label_variant,
            visual_cosine=item.visual_cosine,
            visual_score=item.visual_score,
            text_score=item.text_score,
            fused_score=item.fused_score,
        )
        for index, item in enumerate(scored[:limit], start=1)
    )


def text_scores_for_prediction(
    prediction: Mapping[str, Any] | None,
    labels: Sequence[JointActionLabel],
    *,
    event_text: str | None = None,
) -> dict[tuple[int, int], float]:
    """Compute the current lightweight text retrieval score per joint label.

    This intentionally mirrors the existing mapper's independent verb/noun
    lexical signal, but keeps the result at the joint-pair level so a visual
    score can be fused without changing the Mapper implementation.
    """

    if prediction is not None and not isinstance(prediction, Mapping):
        raise WemmRetrievalError("prediction must be a mapping")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)):
        raise WemmRetrievalError("labels must be a sequence")
    if any(not isinstance(label, JointActionLabel) for label in labels):
        raise WemmRetrievalError("labels contain a malformed joint action")
    if len({label.action_key for label in labels}) != len(labels):
        raise WemmRetrievalError("labels contain duplicate action keys")
    prediction = prediction or {}
    raw_verb = str(prediction.get("verb") or "")
    raw_noun = str(prediction.get("noun") or "")
    raw_event = str(event_text or prediction.get("raw_text") or "")
    scores: dict[tuple[int, int], float] = {}
    for label in labels:
        try:
            verb_score = (
                _coerce_float(
                    _shortlist_score(raw_verb, label.verb_key),
                    field=f"verb score for {label.action_key}",
                )
                if raw_verb
                else 0.0
            )
            noun_score = (
                _coerce_float(
                    _shortlist_score(raw_noun, label.noun_key),
                    field=f"noun score for {label.action_key}",
                )
                if raw_noun
                else 0.0
            )
            phrase_score = (
                _coerce_float(
                    _shortlist_score(raw_event, label.text_for("canonical")),
                    field=f"phrase score for {label.action_key}",
                )
                if raw_event
                else 0.0
            )
        except WemmRetrievalError as exc:
            raise WemmRetrievalError(
                f"text scorer returned an invalid score for {label.action_key}"
            ) from exc
        # The field scores remain dominant, while a retained raw sentence can
        # supply a small complementary signal when one field is missing.
        field_score = (verb_score + noun_score) / 2.0
        scores[label.action_key] = max(0.0, min(1.0, 0.45 * field_score + 0.10 * phrase_score))
    return scores


def project_retrieval_to_mapper(
    ranking: Sequence[RetrievedAction], *, min_score: float = 0.0, min_margin: float = 0.0
) -> dict[str, Any]:
    """Project a joint ranking into the existing mapper-shaped diagnostic view.

    This is deterministic projection only; it does not invoke or retrain the
    existing Mapper.  Keeping the shape compatible lets the established
    post-hoc metric code compare candidate-only and projected results.
    """

    try:
        min_score_value = _coerce_float(min_score, field="min_score")
        min_margin_value = _coerce_float(min_margin, field="min_margin")
    except WemmRetrievalError as exc:
        raise WemmRetrievalError("min_score and min_margin must be finite numbers") from exc
    if not 0.0 <= min_score_value <= 1.0:
        raise WemmRetrievalError("min_score must be in [0,1]")
    if not 0.0 <= min_margin_value <= 1.0:
        raise WemmRetrievalError("min_margin must be in [0,1]")
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes, bytearray)):
        raise WemmRetrievalError("ranking must be a sequence")
    if not ranking:
        return {
            "status": "ABSTAIN",
            "selected_id": None,
            "selected_key": None,
            "selected_score": 0.0,
            "margin": 0.0,
            "reason": "EMPTY_RETRIEVAL",
            "candidates": [],
            "source": "wemm_joint_retrieval_projection",
        }
    for item in ranking:
        if not isinstance(item, RetrievedAction):
            raise WemmRetrievalError("ranking contains a malformed retrieved action")
        if not math.isfinite(item.fused_score):
            raise WemmRetrievalError("ranking contains a non-finite fused score")
    best = ranking[0]
    second_score = ranking[1].fused_score if len(ranking) > 1 else 0.0
    margin = best.fused_score - second_score
    status = (
        "MAPPED"
        if best.fused_score >= min_score_value and margin >= min_margin_value
        else "ABSTAIN"
    )
    candidates = [
        {
            "class_id": item.label.verb_id,
            "key": item.label.verb_key,
            "score": item.fused_score,
            "joint_action": [item.label.verb_id, item.label.noun_id],
        }
        for item in ranking
    ]
    return {
        "status": status,
        "selected_id": best.label.verb_id if status == "MAPPED" else None,
        "selected_key": best.label.verb_key if status == "MAPPED" else None,
        "selected_score": best.fused_score,
        "margin": margin,
        "reason": "OK" if status == "MAPPED" else "LOW_SCORE_OR_MARGIN",
        "candidates": candidates,
        "source": "wemm_joint_retrieval_projection",
        "joint_selected": {
            "verb_id": best.label.verb_id if status == "MAPPED" else None,
            "noun_id": best.label.noun_id if status == "MAPPED" else None,
            "verb_key": best.label.verb_key if status == "MAPPED" else None,
            "noun_key": best.label.noun_key if status == "MAPPED" else None,
        },
    }


def _row_key(row: Mapping[str, Any], index: int) -> str:
    if not isinstance(row, Mapping):
        raise WemmRetrievalError("row must be a mapping")
    for field in ("uid", "case_id", "annotation_id", "id"):
        raw_value = row.get(field)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            return value
    return f"row-{index}"


def _ground_truth_pair(row: Mapping[str, Any]) -> tuple[int, int] | None:
    if not isinstance(row, Mapping):
        return None
    raw = row.get("ground_truth")
    if not isinstance(raw, Mapping):
        raw = row
    try:
        verb = raw.get("verb_class", raw.get("verb_id"))
        noun = raw.get("noun_class", raw.get("noun_id"))
        if verb is None or noun is None:
            return None
        return (
            _coerce_integer(verb, field="ground-truth verb ID"),
            _coerce_integer(noun, field="ground-truth noun ID"),
        )
    except WemmRetrievalError:
        return None


def evaluate_rankings(
    rows: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[RetrievedAction]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Calculate Recall@K/MRR/top-1 and group coverage after retrieval."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise WemmRetrievalError("rows must be a sequence of mappings")
    if not isinstance(rankings, Mapping):
        raise WemmRetrievalError("rankings must be a mapping")
    try:
        raw_ks = tuple(ks)
    except TypeError as exc:
        raise WemmRetrievalError("ks must contain positive integers") from exc
    try:
        canonical_ks = tuple(sorted({_coerce_integer(k, field="k") for k in raw_ks}))
    except WemmRetrievalError as exc:
        raise WemmRetrievalError("ks must contain positive integers") from exc
    if not canonical_ks or any(k <= 0 for k in canonical_ks):
        raise WemmRetrievalError("ks must contain positive integers")
    scored_rows: list[
        tuple[str, tuple[int, int], Sequence[RetrievedAction], Mapping[str, Any]]
    ] = []
    missing_rankings: list[str] = []
    unscored: list[str] = []
    for index, row in enumerate(rows):
        key = _row_key(row, index)
        truth = _ground_truth_pair(row)
        # A row without a usable target cannot be scored even if its ranking is
        # absent.  Classify it as unscored first so the report does not conflate
        # data quality with retrieval coverage.
        if truth is None:
            unscored.append(key)
            continue
        ranking = rankings.get(key)
        if ranking is None:
            missing_rankings.append(key)
            continue
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes, bytearray)):
            raise WemmRetrievalError(f"ranking for {key!r} must be a sequence")
        scored_rows.append((key, truth, ranking, row))
    recall_counts = {str(k): 0 for k in canonical_ks}
    reciprocal_ranks: list[float] = []
    top1 = 0
    margins: list[float] = []
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"scored": 0, "top1": 0})
    for _key, truth, ranking, row in scored_rows:
        for item in ranking:
            if not isinstance(item, RetrievedAction):
                raise WemmRetrievalError("ranking contains a malformed retrieved action")
            if not math.isfinite(item.fused_score):
                raise WemmRetrievalError("ranking contains a non-finite fused score")
        # The sequence order is authoritative.  A caller may reorder a ranking
        # (for example, to compare an intervention) without rewriting the
        # diagnostic ``rank`` fields retained on each item.
        positions = [
            position for position, item in enumerate(ranking, start=1) if item.action_key == truth
        ]
        position = positions[0] if positions else None
        reciprocal_ranks.append(1.0 / position if position else 0.0)
        top1 += int(position == 1)
        for k in canonical_ks:
            recall_counts[str(k)] += int(position is not None and position <= k)
        if ranking:
            margins.append(
                ranking[0].fused_score - (ranking[1].fused_score if len(ranking) > 1 else 0.0)
            )
        group = str(
            row.get("video_group") or row.get("video_id") or row.get("source_group") or "unknown"
        )
        groups[group]["scored"] += 1
        groups[group]["top1"] += int(position == 1)
    denominator = len(scored_rows)
    return {
        "query_count": len(rows),
        "scored_query_count": denominator,
        "unscored_query_ids": unscored,
        "missing_ranking_ids": missing_rankings,
        "recall_at_k": {
            key: (value / denominator if denominator else None)
            for key, value in recall_counts.items()
        },
        "mrr": sum(reciprocal_ranks) / denominator if denominator else None,
        "top1_accuracy": top1 / denominator if denominator else None,
        "mean_top1_margin": sum(margins) / len(margins) if margins else None,
        "video_groups": {
            group: {
                "scored": values["scored"],
                "top1": values["top1"],
                "top1_accuracy": values["top1"] / values["scored"] if values["scored"] else None,
            }
            for group, values in sorted(groups.items())
        },
        "group_count": len(groups),
    }


def compare_rankings(
    rows: Sequence[Mapping[str, Any]],
    rankings_by_mode: Mapping[str, Mapping[str, Sequence[RetrievedAction]]],
) -> dict[str, Any]:
    """Summarize visual/text/hybrid metrics and per-case winner changes."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise WemmRetrievalError("rows must be a sequence of mappings")
    if not isinstance(rankings_by_mode, Mapping):
        raise WemmRetrievalError("rankings_by_mode must be a mapping")
    metrics = {mode: evaluate_rankings(rows, ranking) for mode, ranking in rankings_by_mode.items()}
    case_deltas: list[dict[str, Any]] = []
    keyed_rows = [(_row_key(row, index), row) for index, row in enumerate(rows)]
    for key, row in keyed_rows:
        truth = _ground_truth_pair(row)
        if truth is None:
            continue
        row_result: dict[str, Any] = {"id": key, "ground_truth": list(truth)}
        for mode, ranking in rankings_by_mode.items():
            items = ranking.get(key, ())
            row_result[mode] = {
                "top1": list(items[0].action_key) if items else None,
                "top1_correct": bool(items and items[0].action_key == truth),
                "top5_contains": any(item.action_key == truth for item in items[:5]),
            }
        case_deltas.append(row_result)
    return {"metrics": metrics, "case_deltas": case_deltas}


def build_retrieval_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    rankings_by_mode: Mapping[str, Mapping[str, Sequence[RetrievedAction]]],
    model_identity: str,
    label_variant: str,
    catalog_size: int,
    media_mode: str,
    dimension: int,
    visual_weight: float,
    text_weight: float,
) -> dict[str, Any]:
    """Build a self-describing, non-production experiment report."""

    comparison = compare_rankings(rows, rankings_by_mode)
    return {
        "report_version": WEMM_RETRIEVAL_VERSION,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "model": {
            "identity": model_identity,
            "family": "WeMM-Embedding",
            "requested_variant": "2B",
            "larger_model_invoked": False,
            "dimension": dimension,
        },
        "input": {
            "media_mode": media_mode,
            "label_variant": label_variant,
            "label_text_version": WEMM_LABEL_TEXT_VERSION,
            "catalog_size": catalog_size,
            "query_count": len(rows),
        },
        "fusion": {
            "visual_weight": visual_weight,
            "text_weight": text_weight,
            "score_normalization": "visual cosine [-1,1] -> [0,1]; text score clipped [0,1]",
        },
        "controls": {
            "ontology_modified": False,
            "mapper_training_invoked": False,
            "production_path_changed": False,
            "heldout_100_opened": False,
            "hash_or_sha_used": False,
            "ground_truth_used_in_encoder_input": False,
        },
        "architecture_comparison": {
            "current_pipeline": "Qwen free-text/structured observation -> text retrieval/mapper",
            "wemm_visual": (
                "complete clip or image sequence -> WeMM embedding -> joint action "
                "labels -> cosine top-k"
            ),
            "hybrid": (
                "WeMM visual score + current text retrieval score -> deterministic joint projection"
            ),
            "mapper_note": (
                "Existing mapper code is not modified; projection is mapper-shaped "
                "diagnostic output."
            ),
        },
        "quality": comparison,
    }


__all__ = [
    "LABEL_VARIANTS",
    "WEMM_LABEL_TEXT_VERSION",
    "WEMM_RETRIEVAL_VERSION",
    "JointActionLabel",
    "RetrievedAction",
    "WemmRetrievalError",
    "build_joint_action_catalog",
    "build_retrieval_report",
    "compare_rankings",
    "cosine_similarity",
    "evaluate_rankings",
    "full_cartesian_action_pairs",
    "project_retrieval_to_mapper",
    "rank_joint_actions",
    "render_action_label_texts",
    "text_scores_for_prediction",
    "validate_embedding_matrix",
]
