"""Deterministic, benchmark-only WeMM multi-camera candidate fusion.

This module is intentionally a small pure-function seam between a future
multi-camera WeMM runner and the existing retrieval/Mapper code.  It accepts
already-computed per-camera rankings (or embeddings plus a per-camera query
embedding), performs deterministic rank/score fusion, and returns a JSON-shaped
candidate document suitable for placing in a benchmark model-output sidecar.

No model, media decoder, ontology, Mapper, production API, network client, or
digest/hash implementation is imported here.  The output is diagnostic only:
``production_eligible`` is always false and all camera evidence remains attached
to each fused candidate.

The default policy is score ``mean`` with ``unit`` (clip-to-``[0, 1]``)
normalisation.  A candidate's score is averaged over cameras where it is
present (``missing_score='omit'``); ``missing_score='zero'`` is available when
the experiment wants an absent top-k item to count as explicit zero evidence.
Missing *camera views* are never silently treated as model evidence.  They are
reported in the run-level ``camera_coverage`` object and excluded from the
score denominator.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Final, Literal, TypeGuard, cast

WEMM_MULTIVIEW_RETRIEVAL_VERSION: Final = "wemm-multiview-retrieval-v1"
"""Version of this benchmark-local fusion projection."""

VERSION: Final = WEMM_MULTIVIEW_RETRIEVAL_VERSION
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"

FusionMethod = Literal["mean", "score_mean", "rank_mean", "rank", "rrf", "max", "sum"]
ScoreNormalization = Literal["unit", "clip", "none", "minmax", "cosine", "rank"]
MissingScorePolicy = Literal["omit", "ignore", "zero", "fill_zero"]

FUSION_METHODS: Final = ("mean", "score_mean", "rank_mean", "rank", "rrf", "max", "sum")
SCORE_NORMALIZATIONS: Final = ("unit", "clip", "none", "minmax", "cosine", "rank")
MISSING_SCORE_POLICIES: Final = ("omit", "ignore", "zero", "fill_zero")


class WemmMultiviewRetrievalError(ValueError):
    """Raised when a multi-camera ranking violates the local contract."""


# A shorter compatibility spelling is useful to callers that do not otherwise
# use the WeMM-specific class name.
MultiviewFusionError = WemmMultiviewRetrievalError


def _coerce_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise WemmMultiviewRetrievalError(f"{field_name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise WemmMultiviewRetrievalError(f"{field_name} must be an integer")


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise WemmMultiviewRetrievalError(f"{field_name} must be a finite number")
    # ``float`` also accepts Decimal and numpy scalar values without importing
    # either optional dependency.  Reject strings explicitly: a score copied
    # from a malformed JSON producer should not be silently accepted.
    if isinstance(value, str):
        raise WemmMultiviewRetrievalError(f"{field_name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmMultiviewRetrievalError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise WemmMultiviewRetrievalError(f"{field_name} must be a finite number")
    return number


def _is_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_list(value: object, *, field_name: str) -> list[Any]:
    # Torch/numpy-like values are accepted only through their non-invasive
    # ``tolist`` conversion.  No runtime is imported or invoked.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not _is_sequence(value):
        raise WemmMultiviewRetrievalError(f"{field_name} must be an array")
    return list(value)


def _vector(value: object, *, field_name: str) -> tuple[float, ...]:
    values = _as_list(value, field_name=field_name)
    if not values:
        raise WemmMultiviewRetrievalError(f"{field_name} must not be empty")
    result: list[float] = []
    for index, item in enumerate(values):
        result.append(_coerce_float(item, field_name=f"{field_name}[{index}]"))
    norm = math.sqrt(sum(item * item for item in result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise WemmMultiviewRetrievalError(f"{field_name} must have non-zero norm")
    return tuple(result)


def _unit_vector(value: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(item * item for item in value))
    return tuple(item / norm for item in value)


def _cosine(left: Sequence[float], right: Sequence[float], *, field_name: str) -> float:
    if len(left) != len(right):
        raise WemmMultiviewRetrievalError(f"{field_name} embedding dimensions do not match")
    left_unit = _unit_vector(left)
    right_unit = _unit_vector(right)
    value = sum(a * b for a, b in zip(left_unit, right_unit, strict=True))
    if not math.isfinite(value):
        raise WemmMultiviewRetrievalError(f"{field_name} produced a non-finite cosine")
    return max(-1.0, min(1.0, value))


def _json_copy(value: object, *, field_name: str) -> Any:
    """Validate/copy JSON-shaped metadata without importing a JSON package."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WemmMultiviewRetrievalError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise WemmMultiviewRetrievalError(f"{field_name} mapping keys must be strings")
            copied[key] = _json_copy(child, field_name=f"{field_name}.{key}")
        return copied
    if _is_sequence(value):
        return [
            _json_copy(child, field_name=f"{field_name}[{index}]")
            for index, child in enumerate(value)
        ]
    raise WemmMultiviewRetrievalError(f"{field_name} must be JSON-compatible")


def _action_key(value: object, *, field_name: str = "action_key") -> tuple[int, int] | int | str:
    """Canonicalise a pair or scalar action identifier for stable ordering."""

    if isinstance(value, Mapping):
        if "action_key" in value:
            return _action_key(value["action_key"], field_name=field_name)
        if "joint_action" in value:
            return _action_key(value["joint_action"], field_name=field_name)
        if "verb_id" in value and "noun_id" in value:
            verb = _coerce_integer(value["verb_id"], field_name=f"{field_name}.verb_id")
            noun = _coerce_integer(value["noun_id"], field_name=f"{field_name}.noun_id")
            if verb < 0 or noun < 0:
                raise WemmMultiviewRetrievalError(f"{field_name} IDs must be non-negative")
            return (verb, noun)
        for key in ("candidate_id", "id", "action"):
            if key in value:
                return _action_key(value[key], field_name=field_name)
        raise WemmMultiviewRetrievalError(f"{field_name} is missing an action identifier")
    if _is_sequence(value):
        values = list(value)
        if len(values) != 2:
            raise WemmMultiviewRetrievalError(
                f"{field_name} pair must contain exactly verb and noun IDs"
            )
        verb = _coerce_integer(values[0], field_name=f"{field_name}[0]")
        noun = _coerce_integer(values[1], field_name=f"{field_name}[1]")
        if verb < 0 or noun < 0:
            raise WemmMultiviewRetrievalError(f"{field_name} IDs must be non-negative")
        return (verb, noun)
    if isinstance(value, bool):
        raise WemmMultiviewRetrievalError(f"{field_name} must not be boolean")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise WemmMultiviewRetrievalError(f"{field_name} must be a pair or non-empty identifier")


def _action_sort_key(value: tuple[int, int] | int | str) -> tuple[int, str]:
    if isinstance(value, tuple):
        return (0, f"{value[0]:+020d}:{value[1]:+020d}")
    if isinstance(value, int):
        return (1, f"{value:+020d}")
    return (2, value)


def _action_json(value: tuple[int, int] | int | str) -> list[int] | int | str:
    return list(value) if isinstance(value, tuple) else value


def _normalise_method(value: object, *, field_name: str = "fusion") -> str:
    if not isinstance(value, str):
        raise WemmMultiviewRetrievalError(f"{field_name} must be one of {FUSION_METHODS}")
    method = value.strip().casefold().replace("-", "_")
    aliases = {
        "score": "mean",
        "score_average": "mean",
        "average": "mean",
        "rank_average": "rank_mean",
        "borda": "rank_mean",
        "reciprocal_rank": "rrf",
    }
    method = aliases.get(method, method)
    if method not in {"mean", "score_mean", "rank_mean", "rank", "rrf", "max", "sum"}:
        raise WemmMultiviewRetrievalError(f"unsupported fusion method: {value!r}")
    return "mean" if method == "score_mean" else ("rank_mean" if method == "rank" else method)


def _normalise_score_method(value: object) -> str:
    if not isinstance(value, str):
        raise WemmMultiviewRetrievalError(
            f"score_normalization must be one of {SCORE_NORMALIZATIONS}"
        )
    method = value.strip().casefold().replace("-", "_")
    aliases = {"clip01": "unit", "unit_interval": "unit", "identity": "none"}
    method = aliases.get(method, method)
    if method not in set(SCORE_NORMALIZATIONS):
        raise WemmMultiviewRetrievalError(f"unsupported score normalization: {value!r}")
    return method


def _normalise_missing_policy(value: object) -> str:
    if not isinstance(value, str):
        raise WemmMultiviewRetrievalError(f"missing_score must be one of {MISSING_SCORE_POLICIES}")
    method = value.strip().casefold().replace("-", "_")
    if method not in set(MISSING_SCORE_POLICIES):
        raise WemmMultiviewRetrievalError(f"unsupported missing score policy: {value!r}")
    return "omit" if method == "ignore" else ("zero" if method == "fill_zero" else method)


def normalize_scores(
    scores: Sequence[object] | Mapping[object, object],
    *,
    method: ScoreNormalization | str = "unit",
) -> tuple[float, ...] | dict[object, float]:
    """Return finite scores in a deterministic unit interval.

    ``unit``/``clip`` preserve already-normalised scores and clamp outliers;
    ``none`` is strict and rejects values outside ``[0, 1]``; ``minmax`` uses
    the values in this one camera; and ``cosine`` maps ``[-1, 1]`` to
    ``[0, 1]``.  ``rank`` is accepted for convenience and interprets the
    sequence as descending rank order (the first item receives ``1``).
    """

    normalisation = _normalise_score_method(method)
    mapping_keys: list[object] | None
    if isinstance(scores, Mapping):
        mapping_keys = list(scores.keys())
        values = list(scores.values())
    elif _is_sequence(scores):
        mapping_keys = None
        values = list(scores)
    else:
        raise WemmMultiviewRetrievalError("scores must be an array or object")
    if normalisation == "rank":
        # Rank normalisation is positional and deliberately accepts opaque
        # candidate labels; no numeric coercion is needed.
        count = len(values)
        result_values = [1.0 if count <= 1 else 1.0 - index / (count - 1) for index in range(count)]
    else:
        converted = [
            _coerce_float(value, field_name=f"scores[{index}]")
            for index, value in enumerate(values)
        ]
        if normalisation == "none":
            if any(value < 0.0 or value > 1.0 for value in converted):
                raise WemmMultiviewRetrievalError(
                    "scores must be within [0, 1] for normalization='none'"
                )
            result_values = converted
        elif normalisation in {"unit", "clip"}:
            result_values = [max(0.0, min(1.0, value)) for value in converted]
        elif normalisation == "cosine":
            if any(value < -1.0 or value > 1.0 for value in converted):
                raise WemmMultiviewRetrievalError("cosine scores must be within [-1, 1]")
            result_values = [(value + 1.0) / 2.0 for value in converted]
        else:  # minmax
            if not converted:
                result_values = []
            else:
                low = min(converted)
                high = max(converted)
                if high == low:
                    # A constant list carries no relative evidence.  Assigning
                    # a deterministic full unit value keeps a one-item ranking
                    # useful while retaining the all-equal tie.
                    result_values = [1.0] * len(converted)
                else:
                    result_values = [(value - low) / (high - low) for value in converted]
    if mapping_keys is not None:
        return {key: result_values[index] for index, key in enumerate(mapping_keys)}
    return tuple(result_values)


normalize_camera_scores = normalize_scores


def normalize_score(value: object, *, method: ScoreNormalization | str = "unit") -> float:
    """Normalize one score using the scalar unit-interval policies.

    Batch-dependent policies (``minmax`` and ``rank``) are intentionally
    rejected here; use :func:`normalize_scores` for those policies.
    """

    normalisation = _normalise_score_method(method)
    number = _coerce_float(value, field_name="score")
    if normalisation in {"minmax", "rank"}:
        raise WemmMultiviewRetrievalError(
            f"{normalisation} normalization requires a score sequence"
        )
    if normalisation == "none":
        if not 0.0 <= number <= 1.0:
            raise WemmMultiviewRetrievalError(
                "score must be within [0, 1] for normalization='none'"
            )
        return number
    if normalisation == "cosine":
        if not -1.0 <= number <= 1.0:
            raise WemmMultiviewRetrievalError("cosine score must be within [-1, 1]")
        return (number + 1.0) / 2.0
    return max(0.0, min(1.0, number))


@dataclass(slots=True)
class _ParsedCandidate:
    action_key: tuple[int, int] | int | str
    rank: int
    raw_score: float | None
    embedding: tuple[float, ...] | None
    score_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_score: float = 0.0


def _candidate_action_from_mapping(candidate: Mapping[str, Any]) -> object:
    for key in ("action_key", "joint_action", "candidate_id", "id", "action"):
        if key in candidate:
            return candidate[key]
    if "verb_id" in candidate and "noun_id" in candidate:
        return {"verb_id": candidate["verb_id"], "noun_id": candidate["noun_id"]}
    raise WemmMultiviewRetrievalError("candidate is missing action_key")


_KNOWN_CANDIDATE_FIELDS = frozenset(
    {
        "action_key",
        "joint_action",
        "candidate_id",
        "id",
        "action",
        "verb_id",
        "noun_id",
        "rank",
        "score",
        "fused_score",
        "embedding",
        "vector",
        "action_embedding",
    }
)


def _extract_score(candidate: Mapping[str, Any]) -> float | None:
    for key in ("score", "fused_score", "visual_score", "similarity", "cosine", "confidence"):
        if key in candidate and candidate[key] is not None:
            return _coerce_float(candidate[key], field_name=f"candidate.{key}")
    return None


def _extract_embedding(
    candidate: Mapping[str, Any], *, field_name: str
) -> tuple[float, ...] | None:
    for key in ("embedding", "vector", "action_embedding"):
        if key in candidate and candidate[key] is not None:
            return _vector(candidate[key], field_name=f"{field_name}.{key}")
    return None


def _parse_candidate(
    raw: object,
    *,
    index: int,
    action_hint: object | None = None,
) -> _ParsedCandidate:
    # ``rank_joint_actions`` (the existing single-camera WeMM benchmark seam)
    # returns immutable ``RetrievedAction`` rows rather than dictionaries.
    # Accept its JSON projection through a narrow duck-typed ``to_dict`` hook
    # so the multi-view adapter can consume already-computed rankings directly
    # without importing the model/ontology path.  Any non-mapping projection is
    # rejected as malformed below instead of being treated as an opaque action.
    if not isinstance(raw, Mapping):
        to_dict = getattr(raw, "to_dict", None)
        if callable(to_dict):
            try:
                projected = to_dict()
            except Exception as exc:
                raise WemmMultiviewRetrievalError(
                    "candidate.to_dict() failed while projecting a retrieval row"
                ) from exc
            if isinstance(projected, Mapping):
                raw = projected
    if isinstance(raw, Mapping):
        action = _candidate_action_from_mapping(raw)
        rank_raw = raw.get("rank", index + 1)
        rank = _coerce_integer(rank_raw, field_name="candidate.rank")
        if rank <= 0:
            raise WemmMultiviewRetrievalError("candidate.rank must be a positive integer")
        score = _extract_score(raw)
        embedding = _extract_embedding(raw, field_name="candidate")
        metadata = {
            str(key): _json_copy(value, field_name=f"candidate.{key}")
            for key, value in raw.items()
            if key not in _KNOWN_CANDIDATE_FIELDS
        }
    elif action_hint is not None:
        action = action_hint
        rank = index + 1
        if isinstance(raw, Mapping):  # pragma: no cover - handled above
            score = _extract_score(raw)
            embedding = _extract_embedding(raw, field_name="candidate")
            metadata = {}
        elif isinstance(raw, Real) and not isinstance(raw, bool):
            score = _coerce_float(raw, field_name="candidate.score")
            embedding = None
            metadata = {}
        elif _is_sequence(raw):
            embedding = _vector(raw, field_name="candidate.embedding")
            score = None
            metadata = {}
        elif raw is None:
            score = None
            embedding = None
            metadata = {}
        else:
            raise WemmMultiviewRetrievalError(
                "candidate mapping value must be a score or embedding"
            )
    else:
        # A scalar/identifier in a ranking is a rank-only candidate.  A pair
        # list is interpreted as ``[verb_id, noun_id]`` by ``_action_key``.
        action = raw
        rank = index + 1
        score = None
        embedding = None
        metadata = {}
    return _ParsedCandidate(
        action_key=_action_key(action),
        rank=rank,
        raw_score=score,
        embedding=embedding,
        score_source="provided" if score is not None else "rank",
        metadata=metadata,
    )


def _payload_parts(payload: object) -> tuple[object, object | None]:
    """Return ``(ranked_candidates, query_embedding)`` from a camera payload."""

    if payload is None:
        return (), None
    if isinstance(payload, Mapping):
        query = payload.get("query_embedding", payload.get("query_vector"))
        for key in ("candidates", "ranking", "ranked_actions", "actions"):
            if key in payload:
                return payload[key], query
        # A compact score/embedding object is accepted as well:
        # ``{"scores": {action: 0.7}, "embeddings": {action: [...]}}``.
        if "scores" in payload or "embeddings" in payload:
            scores = payload.get("scores", {})
            embeddings = payload.get("embeddings", {})
            if not isinstance(scores, Mapping) or not isinstance(embeddings, Mapping):
                raise WemmMultiviewRetrievalError("camera scores and embeddings must be objects")
            merged: dict[object, dict[str, Any]] = {}
            for key, value in scores.items():
                merged[key] = {"score": value}
            for key, value in embeddings.items():
                merged.setdefault(key, {})["embedding"] = value
            return merged, query
        # Finally treat unknown keys as an action -> score map.  Reserved
        # camera metadata is ignored only when it is explicitly known.
        reserved = {"query_embedding", "query_vector", "camera_id", "camera", "metadata"}
        compact = {key: value for key, value in payload.items() if key not in reserved}
        return compact, query
    return payload, None


def _parse_camera_candidates(
    payload: object,
) -> tuple[list[_ParsedCandidate], tuple[float, ...] | None]:
    raw_candidates, raw_query = _payload_parts(payload)
    query = _vector(raw_query, field_name="query_embedding") if raw_query is not None else None
    parsed: list[_ParsedCandidate] = []
    if isinstance(raw_candidates, Mapping):
        mapping_items = list(raw_candidates.items())

        # A compact score map is a ranking by definition, not an arbitrary
        # insertion-ordered object.  Sort score-bearing entries by descending
        # score (then action key) before assigning implicit ranks.  Explicit
        # rank fields remain authoritative and are sorted below.
        def mapping_sort_key(item: tuple[object, object]) -> tuple[int, float, tuple[int, str]]:
            raw_action, raw_value = item
            rank_value: object | None = None
            score_value: float | None = None
            if isinstance(raw_value, Mapping):
                rank_value = raw_value.get("rank")
                score_value = _extract_score(raw_value)
            elif isinstance(raw_value, Real) and not isinstance(raw_value, bool):
                score_value = _coerce_float(raw_value, field_name="candidate.score")
            if rank_value is not None:
                rank = _coerce_integer(rank_value, field_name="candidate.rank")
                return (0, float(rank), _action_sort_key(_action_key(raw_action)))
            if score_value is not None:
                return (1, -score_value, _action_sort_key(_action_key(raw_action)))
            return (2, 0.0, _action_sort_key(_action_key(raw_action)))

        mapping_items.sort(key=mapping_sort_key)
        for index, (raw_action, raw_value) in enumerate(mapping_items):
            # A value mapping may carry its own action key/metadata; otherwise
            # the mapping key is the action identifier.
            if isinstance(raw_value, Mapping):
                candidate_mapping = dict(raw_value)
                candidate_mapping.setdefault("action_key", raw_action)
                parsed.append(_parse_candidate(candidate_mapping, index=index))
            else:
                parsed.append(_parse_candidate(raw_value, index=index, action_hint=raw_action))
    elif _is_sequence(raw_candidates):
        for index, raw_candidate in enumerate(raw_candidates):
            # Convenience form: ``[[verb, noun], score]``.  A plain pair is
            # kept as a rank-only action key.
            candidate_values = list(raw_candidate) if _is_sequence(raw_candidate) else []
            if (
                len(candidate_values) == 2
                and _is_sequence(candidate_values[0])
                and isinstance(candidate_values[1], Real)
                and not isinstance(candidate_values[1], bool)
            ):
                pair, score = candidate_values
                parsed.append(
                    _parse_candidate(
                        {"action_key": pair, "score": score, "rank": index + 1}, index=index
                    )
                )
            else:
                parsed.append(_parse_candidate(raw_candidate, index=index))
    else:
        raise WemmMultiviewRetrievalError("camera candidates must be an array or object")

    seen: set[tuple[int, int] | int | str] = set()
    seen_ranks: set[int] = set()
    for item in parsed:
        if item.action_key in seen:
            raise WemmMultiviewRetrievalError(
                f"camera ranking contains duplicate action {item.action_key!r}"
            )
        if item.rank in seen_ranks:
            raise WemmMultiviewRetrievalError("camera ranking contains duplicate ranks")
        seen.add(item.action_key)
        seen_ranks.add(item.rank)
    parsed.sort(key=lambda item: (item.rank, _action_sort_key(item.action_key)))
    # Explicit ranks may be sparse, but sequence-only rankings are contiguous;
    # retaining the supplied rank is important evidence and avoids rewriting a
    # producer's rank semantics.
    return parsed, query


def _camera_entries(
    camera_rankings: object,
) -> dict[str, object]:
    if camera_rankings is None:
        return {}
    if isinstance(camera_rankings, Mapping):
        if "cameras" in camera_rankings and isinstance(camera_rankings["cameras"], Sequence):
            camera_rankings = camera_rankings["cameras"]
        else:
            result: dict[str, object] = {}
            for raw_camera, payload in camera_rankings.items():
                if not isinstance(raw_camera, str) or not raw_camera.strip():
                    raise WemmMultiviewRetrievalError("camera IDs must be non-empty strings")
                camera_id = raw_camera.strip()
                if camera_id in result:
                    raise WemmMultiviewRetrievalError(f"duplicate camera ID: {camera_id!r}")
                result[camera_id] = payload
            return result
    if not _is_sequence(camera_rankings):
        raise WemmMultiviewRetrievalError("camera_rankings must be an object or camera array")
    result = {}
    for index, raw_camera in enumerate(camera_rankings):
        if not isinstance(raw_camera, Mapping):
            raise WemmMultiviewRetrievalError(f"camera[{index}] must be an object")
        raw_id = raw_camera.get("camera_id", raw_camera.get("camera"))
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise WemmMultiviewRetrievalError(
                f"camera[{index}].camera_id must be a non-empty string"
            )
        camera_id = raw_id.strip()
        if camera_id in result:
            raise WemmMultiviewRetrievalError(f"duplicate camera ID: {camera_id!r}")
        if any(
            key in raw_camera
            for key in ("candidates", "ranking", "ranked_actions", "scores", "embeddings")
        ):
            result[camera_id] = {
                key: value
                for key, value in raw_camera.items()
                if key not in {"camera_id", "camera"}
            }
        else:
            result[camera_id] = raw_camera.get("payload")
    return result


def _camera_order(
    observed: Sequence[str],
    *,
    camera_order: Sequence[str] | None,
    expected_cameras: Sequence[str] | None,
) -> tuple[str, ...]:
    def clean(values: Sequence[str], field_name: str) -> tuple[str, ...]:
        if not _is_sequence(values):
            raise WemmMultiviewRetrievalError(f"{field_name} must be an array")
        cleaned: list[str] = []
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise WemmMultiviewRetrievalError(
                    f"{field_name}[{index}] must be a non-empty string"
                )
            item = value.strip()
            if item in cleaned:
                raise WemmMultiviewRetrievalError(
                    f"{field_name} contains duplicate camera {item!r}"
                )
            cleaned.append(item)
        if not cleaned:
            raise WemmMultiviewRetrievalError(f"{field_name} must not be empty")
        return tuple(cleaned)

    explicit = clean(camera_order, "camera_order") if camera_order is not None else None
    expected = clean(expected_cameras, "expected_cameras") if expected_cameras is not None else None
    if explicit is not None and expected is not None and explicit != expected:
        raise WemmMultiviewRetrievalError(
            "camera_order and expected_cameras must match when both supplied"
        )
    result = explicit or expected or tuple(sorted(observed))
    unknown = sorted(set(observed) - set(result))
    if unknown:
        raise WemmMultiviewRetrievalError(
            f"camera_order does not contain observed cameras: {', '.join(unknown)}"
        )
    return result


def _camera_weights(
    weights: Mapping[str, object] | None,
    *,
    order: Sequence[str],
) -> dict[str, float]:
    if weights is None:
        return {camera: 1.0 for camera in order}
    if not isinstance(weights, Mapping):
        raise WemmMultiviewRetrievalError("camera_weights must be an object")
    unknown = sorted(set(weights) - set(order))
    if unknown:
        raise WemmMultiviewRetrievalError(
            f"camera_weights contains unknown cameras: {', '.join(str(item) for item in unknown)}"
        )
    result: dict[str, float] = {camera: 1.0 for camera in order}
    for camera, value in weights.items():
        number = _coerce_float(value, field_name=f"camera_weights.{camera}")
        if number < 0.0:
            raise WemmMultiviewRetrievalError("camera weights must be non-negative")
        result[str(camera)] = number
    if not any(value > 0.0 for value in result.values()):
        raise WemmMultiviewRetrievalError("camera weights must contain a positive value")
    return result


def _rank_score(rank: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (rank - 1) / (count - 1)))


def _candidate_evidence(
    item: _ParsedCandidate,
    *,
    camera_id: str,
    include_embeddings: bool,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "camera_id": camera_id,
        "rank": item.rank,
        "raw_score": item.raw_score,
        "score": item.normalized_score,
        "normalized_score": item.normalized_score,
        "score_source": item.score_source,
    }
    if include_embeddings and item.embedding is not None:
        evidence["embedding"] = list(item.embedding)
    evidence.update(item.metadata)
    return evidence


def fuse_camera_rankings(
    camera_rankings: Mapping[str, object] | Sequence[Mapping[str, object]],
    *,
    camera_order: Sequence[str] | None = None,
    expected_cameras: Sequence[str] | None = None,
    top_k: int | None = 10,
    fusion: FusionMethod | str = "mean",
    fusion_method: FusionMethod | str | None = None,
    method: FusionMethod | str | None = None,
    score_normalization: ScoreNormalization | str = "unit",
    missing_score: MissingScorePolicy | str = "omit",
    missing_view_policy: MissingScorePolicy | str | None = None,
    camera_weights: Mapping[str, object] | None = None,
    include_embeddings: bool = True,
) -> dict[str, Any]:
    """Fuse ranked action candidates from multiple cameras.

    ``camera_rankings`` may be ``{camera_id: candidates}`` or an array of
    ``{"camera_id": ..., "candidates": ...}`` objects.  A candidate can use
    ``action_key`` (a ``[verb_id, noun_id]`` pair), ``verb_id``/``noun_id``, or
    a scalar identifier.  Scores are read from ``score`` (with a few retained
    retrieval aliases) and embeddings from ``embedding``.  A camera payload may
    supply ``query_embedding``; when a candidate has an embedding but no score,
    its cosine similarity to that query is used.

    The returned mapping contains only JSON-compatible values and is safe to
    assign to a model-output sidecar slot's ``predictions`` field.  It is a
    diagnostic projection, not an ontology or Mapper result.
    """

    entries = _camera_entries(camera_rankings)
    if not entries and camera_order is None and expected_cameras is None:
        raise WemmMultiviewRetrievalError("camera_rankings must contain at least one camera")
    order = _camera_order(
        tuple(entries), camera_order=camera_order, expected_cameras=expected_cameras
    )
    fusion_value = fusion_method if fusion_method is not None else fusion
    if method is not None:
        if fusion_method is not None and _normalise_method(
            method, field_name="method"
        ) != _normalise_method(fusion_method, field_name="fusion_method"):
            raise WemmMultiviewRetrievalError("method and fusion_method disagree")
        fusion_value = method
    fusion_name = _normalise_method(fusion_value)
    normalisation = _normalise_score_method(score_normalization)
    if missing_view_policy is not None:
        if missing_score != "omit" and _normalise_missing_policy(
            missing_score
        ) != _normalise_missing_policy(missing_view_policy):
            raise WemmMultiviewRetrievalError("missing_score and missing_view_policy disagree")
        missing_score = missing_view_policy
    missing_name = _normalise_missing_policy(missing_score)
    if not isinstance(include_embeddings, bool):
        raise WemmMultiviewRetrievalError("include_embeddings must be boolean")
    if top_k is None:
        limit: int | None = None
    else:
        limit = _coerce_integer(top_k, field_name="top_k")
        if limit <= 0:
            raise WemmMultiviewRetrievalError("top_k must be a positive integer or None")
    weights = _camera_weights(camera_weights, order=order)

    parsed_by_camera: dict[str, list[_ParsedCandidate]] = {}
    query_by_camera: dict[str, tuple[float, ...] | None] = {}
    embedding_dimensions: int | None = None
    for camera_id in order:
        payload = entries.get(camera_id)
        parsed, query = _parse_camera_candidates(payload) if camera_id in entries else ([], None)
        if query is not None:
            if embedding_dimensions is None:
                embedding_dimensions = len(query)
            elif len(query) != embedding_dimensions:
                raise WemmMultiviewRetrievalError(
                    "camera query embeddings have inconsistent dimensions"
                )
        for item in parsed:
            if item.embedding is not None:
                if embedding_dimensions is None:
                    embedding_dimensions = len(item.embedding)
                elif len(item.embedding) != embedding_dimensions:
                    raise WemmMultiviewRetrievalError(
                        "candidate embeddings have inconsistent dimensions"
                    )
            if item.raw_score is None and item.embedding is not None and query is not None:
                item.raw_score = (
                    _cosine(item.embedding, query, field_name=f"{camera_id}.candidate") + 1.0
                ) / 2.0
                item.score_source = "embedding_cosine"
            parsed_by_camera[camera_id] = parsed
            query_by_camera[camera_id] = query

        if camera_id not in parsed_by_camera:
            parsed_by_camera[camera_id] = parsed
            query_by_camera[camera_id] = query

        # Normalise explicit values camera-by-camera.  Rank-derived values are
        # filled below, which permits rank-only candidate lists for rank fusion
        # and gives score fusion an honest, deterministic fallback.
        explicit = [item for item in parsed if item.raw_score is not None]
        if normalisation == "rank":
            # Rank normalisation is relative to this camera's complete ranked
            # list, not to each one-item scalar passed to ``normalize_scores``.
            for item in parsed:
                item.normalized_score = _rank_score(item.rank, len(parsed))
                item.score_source = "rank"
            continue
        explicit_values: list[float]
        if normalisation == "minmax":
            normalised_values = normalize_scores(
                [item.raw_score for item in explicit], method=normalisation
            )
            explicit_values = list(cast(tuple[float, ...], normalised_values))
        else:
            explicit_values = []
            for item in explicit:
                assert item.raw_score is not None
                # An embedding-derived value is already cosine-mapped to the
                # unit interval.  Do not map it a second time when the caller
                # selects ``score_normalization='cosine'``.
                if item.score_source == "embedding_cosine":
                    if normalisation == "none" and not 0.0 <= item.raw_score <= 1.0:
                        raise WemmMultiviewRetrievalError(
                            "embedding-derived scores must be within [0, 1]"
                        )
                    explicit_values.append(max(0.0, min(1.0, item.raw_score)))
                else:
                    normalised_value = normalize_scores([item.raw_score], method=normalisation)
                    explicit_values.extend(cast(tuple[float, ...], normalised_value))
        explicit_index = 0
        for item in parsed:
            if item.raw_score is not None:
                item.normalized_score = explicit_values[explicit_index]
                explicit_index += 1
            else:
                item.normalized_score = _rank_score(item.rank, len(parsed))
                item.score_source = "rank"

    observed_cameras = [camera for camera in order if parsed_by_camera.get(camera)]
    missing_cameras = [camera for camera in order if camera not in observed_cameras]
    expected_count = len(order)
    observed_count = len(observed_cameras)
    coverage_fraction = observed_count / expected_count if expected_count else 0.0

    by_action: dict[tuple[int, int] | int | str, dict[str, _ParsedCandidate]] = {}
    for camera_id in order:
        for item in parsed_by_camera[camera_id]:
            by_action.setdefault(item.action_key, {})[camera_id] = item

    def aggregate(action_evidence: Mapping[str, _ParsedCandidate]) -> float:
        values: list[tuple[float, float]] = []
        for camera_id in order:
            item = action_evidence.get(camera_id)
            if item is None:
                if missing_name == "zero" and camera_id in observed_cameras:
                    values.append((0.0, weights[camera_id]))
                continue
            value = item.normalized_score
            if fusion_name == "rank_mean":
                value = _rank_score(item.rank, len(parsed_by_camera[camera_id]))
            elif fusion_name == "rrf":
                value = 1.0 / item.rank
            values.append((value, weights[camera_id]))
        if not values:
            return 0.0
        denominator = sum(weight for _, weight in values)
        if denominator <= 0.0:
            return 0.0
        if fusion_name == "max":
            return max(value for value, _ in values)
        if fusion_name == "sum":
            # Keep all fused scores in the same unit interval as mean: divide
            # by the total available camera weight rather than returning a
            # camera-count-dependent scale.
            return min(1.0, sum(value * weight for value, weight in values) / denominator)
        return sum(value * weight for value, weight in values) / denominator

    fused_rows: list[dict[str, Any]] = []
    for action, action_evidence in by_action.items():
        fused_score = aggregate(action_evidence)
        if not math.isfinite(fused_score):  # defensive boundary for future policies
            raise WemmMultiviewRetrievalError(f"fused score for {action!r} is non-finite")
        support = len(action_evidence)
        support_fraction = support / observed_count if observed_count else 0.0
        expected_support_fraction = support / expected_count if expected_count else 0.0
        evidence = [
            _candidate_evidence(
                action_evidence[camera_id],
                camera_id=camera_id,
                include_embeddings=include_embeddings,
            )
            for camera_id in order
            if camera_id in action_evidence
        ]
        row: dict[str, Any] = {
            "rank": 0,
            "action_key": _action_json(action),
            "score": fused_score,
            "fused_score": fused_score,
            "camera_coverage": support,
            "camera_coverage_fraction": support_fraction,
            "expected_camera_coverage_fraction": expected_support_fraction,
            "per_camera": evidence,
            "camera_evidence": evidence,
        }
        # Preserve useful action labels from the first deterministic camera
        # evidence without inventing ontology fields.
        first_metadata = action_evidence[order[0]].metadata if order[0] in action_evidence else {}
        for key in ("verb_id", "noun_id", "verb_key", "noun_key", "label_text", "label_variant"):
            if key in first_metadata:
                row[key] = first_metadata[key]
        fused_rows.append(row)

    fused_rows.sort(
        key=lambda row: (
            -float(row["fused_score"]),
            -int(row["camera_coverage"]),
            -max((float(item["normalized_score"]) for item in row["per_camera"]), default=0.0),
            _action_sort_key(_action_key(row["action_key"])),
        )
    )
    if limit is not None:
        fused_rows = fused_rows[:limit]
    for index, row in enumerate(fused_rows, start=1):
        row["rank"] = index

    return {
        "version": WEMM_MULTIVIEW_RETRIEVAL_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "retrieval_only": True,
        "model_invoked": False,
        "gpu_invoked": False,
        "ontology_modified": False,
        "mapper_modified": False,
        "camera_order": list(order),
        "camera_coverage": {
            "expected_count": expected_count,
            "observed_count": observed_count,
            "fraction": coverage_fraction,
            "observed_cameras": observed_cameras,
            "missing_cameras": missing_cameras,
        },
        "camera_coverage_fraction": coverage_fraction,
        "top_k": limit,
        "fusion": {
            "method": fusion_name,
            "score_normalization": normalisation,
            "missing_score": missing_name,
            "camera_weights": {camera: weights[camera] for camera in order},
        },
        "candidates": fused_rows,
    }


# Descriptive aliases keep the seam easy to discover without requiring callers
# to know which spelling was chosen for the first benchmark experiment.
fuse_multiview_candidates = fuse_camera_rankings
fuse_multiview_action_candidates = fuse_camera_rankings
fuse_ranked_camera_actions = fuse_camera_rankings
build_multiview_candidates = fuse_camera_rankings


__all__ = [
    "AUTHORITY",
    "FUSION_METHODS",
    "MISSING_SCORE_POLICIES",
    "SCORE_NORMALIZATIONS",
    "VERSION",
    "WEMM_MULTIVIEW_RETRIEVAL_VERSION",
    "MultiviewFusionError",
    "WemmMultiviewRetrievalError",
    "build_multiview_candidates",
    "fuse_camera_rankings",
    "fuse_multiview_action_candidates",
    "fuse_multiview_candidates",
    "fuse_ranked_camera_actions",
    "normalize_camera_scores",
    "normalize_score",
    "normalize_scores",
]
