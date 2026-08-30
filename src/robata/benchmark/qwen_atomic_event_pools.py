"""Frozen, benchmark-only case pools for the raw-first Qwen experiments.

The pools are deliberately small, ordinary Python data.  They do not read a
candidate manifest, model output, media, or annotations from disk.  In
particular, ``posthoc_official_action`` is retained solely for scoring and
reporting *after* a generation; neither selection nor record resolution uses
it.

``H8`` is a historical selector diagnostic, ``D12`` is the development screen,
``C24`` is the isolated confirmation screen, and ``M9`` is the matched Mage
sentinel drawn from ``C24``.  H8 may overlap C24's explicitly historical
regression anchors.  D12 and C24 may not overlap by UID, video group, or
participant.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, cast


class QwenAtomicEventPoolName(StrEnum):
    """Names of the fixed local experiment pools."""

    H8 = "H8"
    D12 = "D12"
    C24 = "C24"
    M9 = "M9"


class AtomicEventStratum(StrEnum):
    """Broad action-family strata used to keep the small pools balanced."""

    SWITCH_DIRECTION = "switch/direction"
    OPEN_CLOSE = "open/close"
    TAKE_PUT = "take/put"
    CONTINUOUS_ADJACENT = "continuous/adjacent"


@dataclass(frozen=True, slots=True)
class FrozenAtomicEventPoolCase:
    """One UID selected before any future experiment output is observed.

    ``posthoc_official_action`` is an official-label display value for later
    scoring only.  It is not prompt material and must not influence record
    lookup, pool membership, or partition validation.
    """

    uid: str
    participant_id: str
    video_id: str
    stratum: AtomicEventStratum
    posthoc_official_action: str
    historical: bool = False


class FrozenPoolError(ValueError):
    """Base error for malformed frozen-pool data or record resolution."""


class FrozenPoolPartitionError(FrozenPoolError):
    """Raised when a pool mutation breaks a fixed partition invariant."""


class FrozenPoolRecordResolutionError(FrozenPoolError):
    """Raised when selected UIDs cannot be resolved uniquely from supplied records."""


def _case(
    uid: str,
    stratum: AtomicEventStratum,
    posthoc_official_action: str,
    *,
    historical: bool = False,
) -> FrozenAtomicEventPoolCase:
    participant_id, video_number, _ = uid.split("_", maxsplit=2)
    return FrozenAtomicEventPoolCase(
        uid=uid,
        participant_id=participant_id,
        video_id=f"{participant_id}_{video_number}",
        stratum=stratum,
        posthoc_official_action=posthoc_official_action,
        historical=historical,
    )


# Historical selector diagnostic: these known moments are not new held-out
# evidence.  The action text remains post-generation metadata only.
H8: Final[tuple[FrozenAtomicEventPoolCase, ...]] = (
    _case(
        "P37_102_15",
        AtomicEventStratum.SWITCH_DIRECTION,
        "turn off tap",
        historical=True,
    ),
    _case(
        "P35_101_13",
        AtomicEventStratum.OPEN_CLOSE,
        "open cupboard",
        historical=True,
    ),
    _case(
        "P37_102_33",
        AtomicEventStratum.SWITCH_DIRECTION,
        "turn on tap",
        historical=True,
    ),
    _case(
        "P37_102_35",
        AtomicEventStratum.SWITCH_DIRECTION,
        "turn off tap",
        historical=True,
    ),
    _case(
        "P37_102_46",
        AtomicEventStratum.CONTINUOUS_ADJACENT,
        "cut carrot",
        historical=True,
    ),
    _case(
        "P35_101_78",
        AtomicEventStratum.SWITCH_DIRECTION,
        "turn on tap",
        historical=True,
    ),
    _case(
        "P35_101_81",
        AtomicEventStratum.CONTINUOUS_ADJACENT,
        "wash pot",
        historical=True,
    ),
    _case(
        "P35_101_90",
        AtomicEventStratum.SWITCH_DIRECTION,
        "turn off tap",
        historical=True,
    ),
)


# Fresh development screen.  Its P01/P06/P09/P20 participant and video groups
# are deliberately disjoint from C24.
D12: Final[tuple[FrozenAtomicEventPoolCase, ...]] = (
    _case("P01_13_14", AtomicEventStratum.SWITCH_DIRECTION, "turn on tap"),
    _case("P01_13_19", AtomicEventStratum.SWITCH_DIRECTION, "turn off tap"),
    _case("P06_14_0", AtomicEventStratum.SWITCH_DIRECTION, "turn off extractor fan"),
    _case("P09_08_1", AtomicEventStratum.OPEN_CLOSE, "open refrigerator"),
    _case("P09_08_3", AtomicEventStratum.OPEN_CLOSE, "close refrigerator"),
    _case("P20_07_23", AtomicEventStratum.OPEN_CLOSE, "close dishwasher"),
    _case("P01_13_23", AtomicEventStratum.TAKE_PUT, "take glass"),
    _case("P06_14_6", AtomicEventStratum.TAKE_PUT, "take bowl"),
    _case("P20_07_7", AtomicEventStratum.TAKE_PUT, "put down bowl"),
    _case("P09_08_12", AtomicEventStratum.CONTINUOUS_ADJACENT, "wash blueberry"),
    _case("P09_08_13", AtomicEventStratum.CONTINUOUS_ADJACENT, "rinse blueberry"),
    _case("P20_07_19", AtomicEventStratum.CONTINUOUS_ADJACENT, "rinse rag"),
)


# Grouped confirmation.  Its three historical regression anchors are reported
# separately from the 21 fresh confirmation cases.
C24: Final[tuple[FrozenAtomicEventPoolCase, ...]] = (
    _case("P03_25_12", AtomicEventStratum.SWITCH_DIRECTION, "turn on tap"),
    _case("P03_25_14", AtomicEventStratum.SWITCH_DIRECTION, "turn off tap"),
    _case("P11_21_6", AtomicEventStratum.SWITCH_DIRECTION, "turn off timer"),
    _case("P26_30_2", AtomicEventStratum.SWITCH_DIRECTION, "turn on tap"),
    _case("P28_23_5", AtomicEventStratum.SWITCH_DIRECTION, "turn on microwave"),
    _case(
        "P35_101_90",
        AtomicEventStratum.SWITCH_DIRECTION,
        "turn off tap",
        historical=True,
    ),
    _case("P02_15_41", AtomicEventStratum.OPEN_CLOSE, "open fridge"),
    _case("P02_15_49", AtomicEventStratum.OPEN_CLOSE, "close fridge"),
    _case("P03_26_1", AtomicEventStratum.OPEN_CLOSE, "open fridge"),
    _case("P03_26_3", AtomicEventStratum.OPEN_CLOSE, "close fridge"),
    _case("P28_23_1", AtomicEventStratum.OPEN_CLOSE, "open microwave"),
    _case("P28_23_3", AtomicEventStratum.OPEN_CLOSE, "close microwave"),
    _case("P02_15_10", AtomicEventStratum.TAKE_PUT, "put pie"),
    _case("P03_25_9", AtomicEventStratum.TAKE_PUT, "take grape"),
    _case("P11_18_1", AtomicEventStratum.TAKE_PUT, "pick up bowl"),
    _case("P11_18_16", AtomicEventStratum.TAKE_PUT, "put down bowl"),
    _case("P26_30_1", AtomicEventStratum.TAKE_PUT, "pick up small pot"),
    _case("P28_23_2", AtomicEventStratum.TAKE_PUT, "put dish into microwave"),
    _case("P02_15_21", AtomicEventStratum.CONTINUOUS_ADJACENT, "dry tupperware"),
    _case("P03_25_1", AtomicEventStratum.CONTINUOUS_ADJACENT, "fill glass"),
    _case("P04_26_1", AtomicEventStratum.CONTINUOUS_ADJACENT, "slice chilli"),
    _case("P11_18_12", AtomicEventStratum.CONTINUOUS_ADJACENT, "pour cereal"),
    _case(
        "P35_101_81",
        AtomicEventStratum.CONTINUOUS_ADJACENT,
        "wash pot",
        historical=True,
    ),
    _case(
        "P37_102_46",
        AtomicEventStratum.CONTINUOUS_ADJACENT,
        "cut carrot",
        historical=True,
    ),
)


_C24_BY_UID = {case.uid: case for case in C24}
M9: Final[tuple[FrozenAtomicEventPoolCase, ...]] = tuple(
    _C24_BY_UID[uid]
    for uid in (
        "P03_25_12",
        "P03_25_14",
        "P28_23_3",
        "P02_15_10",
        "P11_18_1",
        "P26_30_1",
        "P04_26_1",
        "P35_101_81",
        "P37_102_46",
    )
)
del _C24_BY_UID


FROZEN_QWEN_ATOMIC_EVENT_POOLS: Final[
    Mapping[QwenAtomicEventPoolName, tuple[FrozenAtomicEventPoolCase, ...]]
] = MappingProxyType(
    {
        QwenAtomicEventPoolName.H8: H8,
        QwenAtomicEventPoolName.D12: D12,
        QwenAtomicEventPoolName.C24: C24,
        QwenAtomicEventPoolName.M9: M9,
    }
)
_POOL_NAMES: Final[tuple[QwenAtomicEventPoolName, ...]] = (
    QwenAtomicEventPoolName.H8,
    QwenAtomicEventPoolName.D12,
    QwenAtomicEventPoolName.C24,
    QwenAtomicEventPoolName.M9,
)

_EXPECTED_POOL_SIZES: Final[Mapping[QwenAtomicEventPoolName, int]] = MappingProxyType(
    {
        QwenAtomicEventPoolName.H8: 8,
        QwenAtomicEventPoolName.D12: 12,
        QwenAtomicEventPoolName.C24: 24,
        QwenAtomicEventPoolName.M9: 9,
    }
)
_EXPECTED_STRATA: Final[Mapping[QwenAtomicEventPoolName, Mapping[AtomicEventStratum, int]]] = (
    MappingProxyType(
        {
            QwenAtomicEventPoolName.H8: MappingProxyType(
                {
                    AtomicEventStratum.SWITCH_DIRECTION: 5,
                    AtomicEventStratum.OPEN_CLOSE: 1,
                    AtomicEventStratum.TAKE_PUT: 0,
                    AtomicEventStratum.CONTINUOUS_ADJACENT: 2,
                }
            ),
            QwenAtomicEventPoolName.D12: MappingProxyType(
                {
                    AtomicEventStratum.SWITCH_DIRECTION: 3,
                    AtomicEventStratum.OPEN_CLOSE: 3,
                    AtomicEventStratum.TAKE_PUT: 3,
                    AtomicEventStratum.CONTINUOUS_ADJACENT: 3,
                }
            ),
            QwenAtomicEventPoolName.C24: MappingProxyType(
                {
                    AtomicEventStratum.SWITCH_DIRECTION: 6,
                    AtomicEventStratum.OPEN_CLOSE: 6,
                    AtomicEventStratum.TAKE_PUT: 6,
                    AtomicEventStratum.CONTINUOUS_ADJACENT: 6,
                }
            ),
            QwenAtomicEventPoolName.M9: MappingProxyType(
                {
                    AtomicEventStratum.SWITCH_DIRECTION: 2,
                    AtomicEventStratum.OPEN_CLOSE: 1,
                    AtomicEventStratum.TAKE_PUT: 3,
                    AtomicEventStratum.CONTINUOUS_ADJACENT: 3,
                }
            ),
        }
    )
)
_UID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<participant>P\d{2})_(?P<video>\d+)_(?P<annotation>\d+)$"
)


def get_frozen_pool(
    pool: QwenAtomicEventPoolName | str,
) -> tuple[FrozenAtomicEventPoolCase, ...]:
    """Return a fixed pool in its declared run order without performing I/O."""

    return FROZEN_QWEN_ATOMIC_EVENT_POOLS[_coerce_pool_name(pool)]


def validate_pool_partition(
    pools: Mapping[QwenAtomicEventPoolName | str, Sequence[FrozenAtomicEventPoolCase]]
    | None = None,
) -> None:
    """Fail closed when a mutated frozen-pool partition loses its safeguards.

    Validation intentionally uses only UID/group/stratum metadata.  It does not
    consume a model response and does not use an official action to select,
    filter, or partition a case.
    """

    source = FROZEN_QWEN_ATOMIC_EVENT_POOLS if pools is None else pools
    normalized = _normalize_pool_mapping(
        cast(Mapping[object, Sequence[FrozenAtomicEventPoolCase]], source)
    )
    for name in _POOL_NAMES:
        cases = normalized[name]
        _validate_pool_cases(name, cases)
        _validate_pool_shape(name, cases)

    d12 = normalized[QwenAtomicEventPoolName.D12]
    c24 = normalized[QwenAtomicEventPoolName.C24]
    m9 = normalized[QwenAtomicEventPoolName.M9]

    shared_uids = _shared_values(d12, c24, lambda case: case.uid)
    if shared_uids:
        raise FrozenPoolPartitionError(
            "D12 and C24 must not share UIDs: " + ", ".join(sorted(shared_uids))
        )
    shared_videos = _shared_values(d12, c24, lambda case: case.video_id)
    if shared_videos:
        raise FrozenPoolPartitionError(
            "D12 and C24 must not share video groups: " + ", ".join(sorted(shared_videos))
        )
    shared_participants = _shared_values(d12, c24, lambda case: case.participant_id)
    if shared_participants:
        raise FrozenPoolPartitionError(
            "D12 and C24 must not share participants: " + ", ".join(sorted(shared_participants))
        )

    c24_by_uid = {case.uid: case for case in c24}
    missing_from_c24 = [case.uid for case in m9 if case.uid not in c24_by_uid]
    if missing_from_c24:
        raise FrozenPoolPartitionError(
            "M9 must be a C24 UID subset; missing from C24: " + ", ".join(missing_from_c24)
        )
    mismatched_m9_metadata = [
        case.uid
        for case in m9
        if (
            case.participant_id != c24_by_uid[case.uid].participant_id
            or case.video_id != c24_by_uid[case.uid].video_id
            or case.stratum != c24_by_uid[case.uid].stratum
            or case.historical != c24_by_uid[case.uid].historical
        )
    ]
    if mismatched_m9_metadata:
        raise FrozenPoolPartitionError(
            "M9 metadata must match its C24 cases: " + ", ".join(mismatched_m9_metadata)
        )


def resolve_pool_records[RecordT](
    pool: QwenAtomicEventPoolName | str | Sequence[FrozenAtomicEventPoolCase],
    records: Mapping[str, RecordT] | Iterable[RecordT],
    *,
    uid_getter: Callable[[RecordT], str] | None = None,
) -> tuple[RecordT, ...]:
    """Resolve supplied records in frozen-pool order using UID only.

    A mapping is indexed by its keys.  For an iterable, the default lookup uses
    a mapping item's ``uid`` field or an object's ``uid`` attribute; callers can
    supply ``uid_getter`` for another record shape.  The returned records are
    the original supplied values, not copies enriched with official labels.
    """

    cases, label = _resolve_pool_cases(pool)
    _validate_pool_cases(label, cases)
    record_index = _record_index(records, uid_getter=uid_getter)
    missing = [case.uid for case in cases if case.uid not in record_index]
    if missing:
        raise FrozenPoolRecordResolutionError(
            f"{label}: records missing for UIDs: " + ", ".join(missing)
        )
    return tuple(record_index[case.uid] for case in cases)


def _coerce_pool_name(value: object) -> QwenAtomicEventPoolName:
    if not isinstance(value, str):
        known = ", ".join(name.value for name in _POOL_NAMES)
        raise FrozenPoolPartitionError(f"unknown frozen pool {value!r}; expected one of {known}")
    try:
        return QwenAtomicEventPoolName(value)
    except ValueError as error:
        known = ", ".join(name.value for name in _POOL_NAMES)
        raise FrozenPoolPartitionError(
            f"unknown frozen pool {value!r}; expected one of {known}"
        ) from error


def _normalize_pool_mapping(
    pools: Mapping[object, Sequence[FrozenAtomicEventPoolCase]],
) -> dict[QwenAtomicEventPoolName, tuple[FrozenAtomicEventPoolCase, ...]]:
    normalized: dict[QwenAtomicEventPoolName, tuple[FrozenAtomicEventPoolCase, ...]] = {}
    for raw_name, cases in pools.items():
        name = _coerce_pool_name(raw_name)
        if name in normalized:
            raise FrozenPoolPartitionError(f"pool mapping repeats {name.value}")
        normalized[name] = tuple(cases)
    missing = tuple(name for name in _POOL_NAMES if name not in normalized)
    if missing:
        raise FrozenPoolPartitionError(
            "pool mapping must contain H8, D12, C24, and M9 (missing "
            + ", ".join(name.value for name in missing)
            + ")"
        )
    return normalized


def _validate_pool_cases(
    name: QwenAtomicEventPoolName | str,
    cases: Sequence[FrozenAtomicEventPoolCase],
) -> None:
    label = name.value if isinstance(name, QwenAtomicEventPoolName) else str(name)
    for case in cases:
        if not isinstance(case, FrozenAtomicEventPoolCase):
            raise FrozenPoolPartitionError(
                f"{label}: each entry must be a FrozenAtomicEventPoolCase"
            )
    duplicates = _duplicate_values(case.uid for case in cases)
    if duplicates:
        raise FrozenPoolPartitionError(f"{label}: duplicate UIDs: " + ", ".join(sorted(duplicates)))
    for case in cases:
        _validate_case_identity(label, case)


def _validate_case_identity(label: str, case: FrozenAtomicEventPoolCase) -> None:
    match = _UID_PATTERN.fullmatch(case.uid)
    if match is None:
        raise FrozenPoolPartitionError(f"{label}: invalid UID {case.uid!r}")
    expected_participant = match.group("participant")
    expected_video = f"{expected_participant}_{match.group('video')}"
    if case.participant_id != expected_participant:
        raise FrozenPoolPartitionError(
            f"{label}: {case.uid} participant_id must be {expected_participant}"
        )
    if case.video_id != expected_video:
        raise FrozenPoolPartitionError(f"{label}: {case.uid} video_id must be {expected_video}")
    if not isinstance(case.stratum, AtomicEventStratum):
        raise FrozenPoolPartitionError(f"{label}: {case.uid} has an unknown stratum")
    if not isinstance(case.historical, bool):
        raise FrozenPoolPartitionError(f"{label}: {case.uid} historical must be a bool")
    if (
        not isinstance(case.posthoc_official_action, str)
        or not case.posthoc_official_action.strip()
    ):
        raise FrozenPoolPartitionError(
            f"{label}: {case.uid} must retain a non-empty posthoc official action"
        )


def _validate_pool_shape(
    name: QwenAtomicEventPoolName,
    cases: Sequence[FrozenAtomicEventPoolCase],
) -> None:
    if len(cases) != _EXPECTED_POOL_SIZES[name]:
        raise FrozenPoolPartitionError(
            f"{name.value}: expected {_EXPECTED_POOL_SIZES[name]} cases, got {len(cases)}"
        )
    observed_strata = Counter(case.stratum for case in cases)
    expected_strata = _EXPECTED_STRATA[name]
    observed_counts = {stratum: observed_strata[stratum] for stratum in AtomicEventStratum}
    if observed_counts != dict(expected_strata):
        expected = ", ".join(
            f"{stratum.value}={count}" for stratum, count in expected_strata.items()
        )
        observed = ", ".join(
            f"{stratum.value}={observed_counts[stratum]}" for stratum in AtomicEventStratum
        )
        raise FrozenPoolPartitionError(
            f"{name.value}: unexpected stratum balance; expected {expected}; got {observed}"
        )
    historical_count = sum(case.historical for case in cases)
    expected_historical = {
        QwenAtomicEventPoolName.H8: 8,
        QwenAtomicEventPoolName.D12: 0,
        QwenAtomicEventPoolName.C24: 3,
        QwenAtomicEventPoolName.M9: 2,
    }[name]
    if historical_count != expected_historical:
        raise FrozenPoolPartitionError(
            f"{name.value}: expected {expected_historical} historical cases, got {historical_count}"
        )


def _shared_values(
    first: Sequence[FrozenAtomicEventPoolCase],
    second: Sequence[FrozenAtomicEventPoolCase],
    key: Callable[[FrozenAtomicEventPoolCase], str],
) -> set[str]:
    return {key(case) for case in first} & {key(case) for case in second}


def _duplicate_values(values: Iterable[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def _resolve_pool_cases(
    pool: QwenAtomicEventPoolName | str | Sequence[FrozenAtomicEventPoolCase],
) -> tuple[tuple[FrozenAtomicEventPoolCase, ...], str]:
    if isinstance(pool, (QwenAtomicEventPoolName, str)):
        name = _coerce_pool_name(pool)
        return FROZEN_QWEN_ATOMIC_EVENT_POOLS[name], name.value
    return tuple(pool), "provided pool"


def _record_index[RecordT](
    records: Mapping[str, RecordT] | Iterable[RecordT],
    *,
    uid_getter: Callable[[RecordT], str] | None,
) -> dict[str, RecordT]:
    indexed: dict[str, RecordT] = {}
    if isinstance(records, Mapping):
        items = records.items()
        for raw_uid, record in items:
            uid = _required_uid(raw_uid, context="record mapping key")
            _add_record(indexed, uid, record)
        return indexed

    getter = uid_getter or _default_record_uid
    for record in records:
        uid = _required_uid(getter(record), context="record UID")
        _add_record(indexed, uid, record)
    return indexed


def _add_record[RecordT](indexed: dict[str, RecordT], uid: str, record: RecordT) -> None:
    if uid in indexed:
        raise FrozenPoolRecordResolutionError(f"records contain duplicate UID: {uid}")
    indexed[uid] = record


def _required_uid(value: object, *, context: str) -> str:
    uid = str(value).strip()
    if not uid:
        raise FrozenPoolRecordResolutionError(f"{context} must be a non-empty UID")
    return uid


def _default_record_uid[RecordT](record: RecordT) -> str:
    value = record.get("uid") if isinstance(record, Mapping) else getattr(record, "uid", None)
    if value is None:
        raise FrozenPoolRecordResolutionError(
            "records supplied as an iterable require a uid field, uid attribute, or uid_getter"
        )
    return str(value)


__all__ = [
    "C24",
    "D12",
    "FROZEN_QWEN_ATOMIC_EVENT_POOLS",
    "H8",
    "M9",
    "AtomicEventStratum",
    "FrozenAtomicEventPoolCase",
    "FrozenPoolError",
    "FrozenPoolPartitionError",
    "FrozenPoolRecordResolutionError",
    "QwenAtomicEventPoolName",
    "get_frozen_pool",
    "resolve_pool_records",
    "validate_pool_partition",
]
