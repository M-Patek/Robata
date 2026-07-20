"""Deterministic grouped data splits for benchmark evaluation.

MCAPs are the statistical unit. When metadata is supplied, records sharing a
session, actor, scene, collection day, or rig are joined transitively and the
whole connected component is assigned to one split. Missing stratification
metadata is never replaced with synthetic membership counts.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

from pydantic import Field, StringConstraints, ValidationError

from robata.benchmark.models import StratificationDimension
from robata.contracts.common import StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
SplitName = Literal["development", "validation", "frozen_test"]
_SPLIT_NAMES: tuple[SplitName, ...] = ("development", "validation", "frozen_test")
_ROUNDING_METHOD: Final[Literal["LARGEST_REMAINDER_MCAP_TARGET_THEN_GROUP_GREEDY"]] = (
    "LARGEST_REMAINDER_MCAP_TARGET_THEN_GROUP_GREEDY"
)


class SplitMetadataError(ValueError):
    """Required grouping or stratification metadata is absent or contradictory."""


class SplitRecord(StrictModel):
    """One MCAP and the metadata used to prevent benchmark leakage.

    The four nullable grouping fields are required in mapping input: ``None``
    means the dimension is explicitly not applicable. Omitting the fields is
    treated as incomplete metadata and fails closed.
    """

    mcap_id: NonEmptyString
    session_id: NonEmptyString
    actor: NonEmptyString | None
    scene: NonEmptyString | None
    collection_day: NonEmptyString | None
    rig: NonEmptyString | None
    strata: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)


class SplitConfig(StrictModel):
    """Registered split ratios and deterministic seed."""

    version: NonEmptyString
    development_ratio: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    validation_ratio: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    frozen_test_ratio: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    random_seed: int


class DataSplitResult(StrictModel):
    """One immutable split result plus auditable allocation evidence."""

    development: tuple[NonEmptyString, ...]
    validation: tuple[NonEmptyString, ...]
    frozen_test: tuple[NonEmptyString, ...]
    stratification_report: dict[str, dict[str, int]]
    grouping_metadata_complete: bool = False
    leakage_group_ids: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)
    rounding_method: Literal["LARGEST_REMAINDER_MCAP_TARGET_THEN_GROUP_GREEDY"] = _ROUNDING_METHOD
    notes: tuple[NonEmptyString, ...] = ()


@dataclass(frozen=True, slots=True)
class _Group:
    group_id: str
    records: tuple[SplitRecord, ...]
    strata_counts: Counter[tuple[str, str]]

    @property
    def size(self) -> int:
        return len(self.records)


class DataSplitter:
    """Assign complete leakage groups to registered benchmark splits."""

    def __init__(self, config: SplitConfig) -> None:
        self._config = config
        total = config.development_ratio + config.validation_ratio + config.frozen_test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")

    def split(
        self,
        mcap_ids: Sequence[str],
        stratify_by: Sequence[StratificationDimension] = (),
        *,
        records: Sequence[SplitRecord | Mapping[str, Any]]
        | Mapping[str, Mapping[str, Any]]
        | None = None,
        metadata: Sequence[SplitRecord | Mapping[str, Any]]
        | Mapping[str, Mapping[str, Any]]
        | None = None,
    ) -> DataSplitResult:
        """Split MCAPs without separating a known leakage group.

        The legacy call without ``records`` remains available only for an
        unstratified, non-certifying local split. Supplying a stratification
        request without record metadata raises ``SplitMetadataError``.
        """

        if records is not None and metadata is not None:
            raise SplitMetadataError("provide records or metadata, not both")
        if metadata is not None:
            records = metadata
        ids = _validate_mcap_ids(mcap_ids)
        dimensions = _validate_dimensions(stratify_by)
        if records is None:
            if dimensions:
                raise SplitMetadataError(
                    "records are required to report real stratification membership"
                )
            normalized = tuple(
                SplitRecord(
                    mcap_id=mcap_id,
                    session_id=mcap_id,
                    actor=None,
                    scene=None,
                    collection_day=None,
                    rig=None,
                )
                for mcap_id in ids
            )
            metadata_complete = False
        else:
            normalized = _normalize_records(records, ids)
            _validate_record_strata(normalized, dimensions)
            metadata_complete = True

        groups = _build_groups(normalized, dimensions)
        ratios = self._ratios()
        targets = _largest_remainder(len(ids), ratios)
        stratum_totals: Counter[tuple[str, str]] = Counter()
        for group in groups:
            stratum_totals.update(group.strata_counts)
        stratum_targets = {
            key: _largest_remainder(total, ratios) for key, total in sorted(stratum_totals.items())
        }
        assignments = _allocate_groups(
            groups,
            targets=targets,
            stratum_targets=stratum_targets,
            seed=self._config.random_seed,
        )
        split_ids: dict[SplitName, list[str]] = {name: [] for name in _SPLIT_NAMES}
        group_ids: dict[str, str] = {}
        for group in groups:
            split_name = assignments[group.group_id]
            split_ids[split_name].extend(record.mcap_id for record in group.records)
            for record in group.records:
                group_ids[record.mcap_id] = group.group_id
        for values in split_ids.values():
            values.sort()

        report = _build_report(
            split_ids,
            groups=groups,
            assignments=assignments,
            dimensions=dimensions,
            targets=targets,
            stratum_targets=stratum_targets,
        )
        notes = (
            (
                "Grouping metadata absent; each MCAP was treated as its own session "
                "and the split is non-certifying.",
            )
            if not metadata_complete
            else (
                "Target counts use largest-remainder rounding; indivisible leakage "
                "groups may cause actual count deltas.",
            )
        )
        return DataSplitResult(
            development=tuple(split_ids["development"]),
            validation=tuple(split_ids["validation"]),
            frozen_test=tuple(split_ids["frozen_test"]),
            stratification_report=report,
            grouping_metadata_complete=metadata_complete,
            leakage_group_ids=group_ids if metadata_complete else {},
            notes=notes,
        )

    def validate_no_leakage(
        self,
        splits: DataSplitResult,
        *,
        records: Sequence[SplitRecord | Mapping[str, Any]]
        | Mapping[str, Mapping[str, Any]]
        | None = None,
        metadata: Sequence[SplitRecord | Mapping[str, Any]]
        | Mapping[str, Mapping[str, Any]]
        | None = None,
    ) -> bool:
        """Check duplicate MCAPs and, when available, leakage-group separation."""

        if records is not None and metadata is not None:
            return False
        if metadata is not None:
            records = metadata
        memberships: dict[str, SplitName] = {}
        for split_name, values in zip(
            _SPLIT_NAMES,
            (splits.development, splits.validation, splits.frozen_test),
            strict=True,
        ):
            if len(values) != len(set(values)):
                return False
            for mcap_id in values:
                if mcap_id in memberships:
                    return False
                memberships[mcap_id] = split_name

        if records is not None:
            try:
                normalized = _normalize_records(records, tuple(sorted(memberships)))
                groups = _build_groups(normalized, ())
            except (SplitMetadataError, ValidationError, TypeError, ValueError):
                return False
            return all(
                len({memberships[record.mcap_id] for record in group.records}) == 1
                for group in groups
            )

        if splits.grouping_metadata_complete:
            if set(splits.leakage_group_ids) != set(memberships):
                return False
            group_splits: dict[str, set[SplitName]] = {}
            for mcap_id, group_id in splits.leakage_group_ids.items():
                group_splits.setdefault(group_id, set()).add(memberships[mcap_id])
            return all(len(split_names) == 1 for split_names in group_splits.values())
        return True

    def _ratios(self) -> dict[SplitName, float]:
        return {
            "development": self._config.development_ratio,
            "validation": self._config.validation_ratio,
            "frozen_test": self._config.frozen_test_ratio,
        }


def _validate_mcap_ids(mcap_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(mcap_ids, (str, bytes)):
        raise TypeError("mcap_ids must be a sequence of identifiers")
    values = tuple(mcap_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("every mcap_id must be a nonempty string")
    if len(values) != len(set(values)):
        raise ValueError("mcap_ids must be unique")
    return tuple(sorted(values))


def _validate_dimensions(
    dimensions: Sequence[StratificationDimension],
) -> tuple[StratificationDimension, ...]:
    result = tuple(dimensions)
    names = [dimension.dimension for dimension in result]
    if len(names) != len(set(names)):
        raise ValueError("stratification dimension names must be unique")
    for dimension in result:
        if not dimension.values:
            raise ValueError(f"stratification dimension {dimension.dimension!r} has no values")
        if len(dimension.values) != len(dimension.proportions):
            raise ValueError(
                f"stratification dimension {dimension.dimension!r} values/proportions differ"
            )
        if len(dimension.values) != len(set(dimension.values)):
            raise ValueError(
                f"stratification dimension {dimension.dimension!r} has duplicate values"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in dimension.proportions):
            raise ValueError("stratification proportions must be finite and nonnegative")
        if abs(sum(dimension.proportions) - 1.0) > 1e-9:
            raise ValueError("stratification proportions must sum to 1.0")
    return result


def _normalize_records(
    records: Sequence[SplitRecord | Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    mcap_ids: tuple[str, ...],
) -> tuple[SplitRecord, ...]:
    normalized: list[SplitRecord] = []
    if isinstance(records, Mapping):
        sequence: Sequence[SplitRecord | Mapping[str, Any]] = tuple(
            _record_mapping_for_id(mcap_id, payload) for mcap_id, payload in records.items()
        )
    else:
        sequence = records
    for record in sequence:
        if isinstance(record, SplitRecord):
            normalized.append(record)
        elif isinstance(record, Mapping):
            try:
                normalized.append(SplitRecord.model_validate(dict(record), strict=True))
            except ValidationError as exc:
                raise SplitMetadataError(f"invalid split record: {exc}") from exc
        else:
            raise TypeError("records must contain SplitRecord or mapping values")
    ids = [record.mcap_id for record in normalized]
    if len(ids) != len(set(ids)):
        raise SplitMetadataError("records contain duplicate mcap_id values")
    expected = set(mcap_ids)
    actual = set(ids)
    if actual != expected:
        raise SplitMetadataError(
            f"records must cover exactly mcap_ids: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return tuple(sorted(normalized, key=lambda record: record.mcap_id))


def _record_mapping_for_id(
    mcap_id: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(mcap_id, str) or not mcap_id:
        raise SplitMetadataError("metadata keys must be nonempty mcap IDs")
    normalized = dict(payload)
    embedded_id = normalized.get("mcap_id")
    if embedded_id is not None and embedded_id != mcap_id:
        raise SplitMetadataError(
            f"metadata key {mcap_id!r} conflicts with embedded mcap_id {embedded_id!r}"
        )
    normalized["mcap_id"] = mcap_id
    return normalized


def _validate_record_strata(
    records: tuple[SplitRecord, ...],
    dimensions: tuple[StratificationDimension, ...],
) -> None:
    for record in records:
        for dimension in dimensions:
            value = record.strata.get(dimension.dimension)
            if value is None:
                raise SplitMetadataError(
                    f"record {record.mcap_id!r} lacks stratum {dimension.dimension!r}"
                )
            if value not in dimension.values:
                raise SplitMetadataError(
                    f"record {record.mcap_id!r} has unknown {dimension.dimension!r} value {value!r}"
                )


def _build_groups(
    records: tuple[SplitRecord, ...],
    dimensions: tuple[StratificationDimension, ...],
) -> tuple[_Group, ...]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = (
            ("session_id", record.session_id),
            ("actor", record.actor),
            ("scene", record.scene),
            ("collection_day", record.collection_day),
            ("rig", record.rig),
        )
        for name, value in keys:
            if value is None:
                continue
            key = (name, value)
            previous = seen.setdefault(key, index)
            union(index, previous)

    components: dict[int, list[SplitRecord]] = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record)
    groups: list[_Group] = []
    for members in components.values():
        ordered = tuple(sorted(members, key=lambda record: record.mcap_id))
        group_id = hashlib.sha256(
            ("robata-benchmark-leakage-group-v1\0" + "\0".join(r.mcap_id for r in ordered)).encode()
        ).hexdigest()
        strata = Counter(
            (dimension.dimension, record.strata[dimension.dimension])
            for record in ordered
            for dimension in dimensions
        )
        groups.append(_Group(group_id=group_id, records=ordered, strata_counts=strata))
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _largest_remainder(
    total: int,
    ratios: Mapping[SplitName, float],
) -> dict[SplitName, int]:
    raw = {name: total * ratios[name] for name in _SPLIT_NAMES}
    result = {name: math.floor(raw[name]) for name in _SPLIT_NAMES}
    remaining = total - sum(result.values())
    priority = {name: index for index, name in enumerate(_SPLIT_NAMES)}
    order = sorted(
        _SPLIT_NAMES,
        key=lambda name: (-(raw[name] - result[name]), priority[name]),
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def _allocate_groups(
    groups: tuple[_Group, ...],
    *,
    targets: Mapping[SplitName, int],
    stratum_targets: Mapping[tuple[str, str], Mapping[SplitName, int]],
    seed: int,
) -> dict[str, SplitName]:
    actual = {name: 0 for name in _SPLIT_NAMES}
    actual_strata = {name: Counter[tuple[str, str]]() for name in _SPLIT_NAMES}
    global_strata = {
        key: sum(group.strata_counts[key] for group in groups) for key in stratum_targets
    }
    ordered = sorted(
        groups,
        key=lambda group: (
            -group.size,
            sum(count * global_strata[key] for key, count in group.strata_counts.items()),
            _seeded_rank(seed, group.group_id),
        ),
    )
    assignments: dict[str, SplitName] = {}
    total_weight = len(stratum_targets) + 1
    for group in ordered:
        candidates: list[tuple[int, int, SplitName]] = []
        for candidate in _SPLIT_NAMES:
            total_penalty = 0
            for split_name in _SPLIT_NAMES:
                count = actual[split_name] + (group.size if split_name == candidate else 0)
                total_penalty += (count - targets[split_name]) ** 2
            stratum_penalty = 0
            for key, target_by_split in stratum_targets.items():
                for split_name in _SPLIT_NAMES:
                    count = actual_strata[split_name][key]
                    if split_name == candidate:
                        count += group.strata_counts[key]
                    stratum_penalty += (count - target_by_split[split_name]) ** 2
            score = total_penalty * total_weight + stratum_penalty
            candidates.append(
                (score, _seeded_rank(seed, f"{group.group_id}:{candidate}"), candidate)
            )
        _, _, selected = min(candidates)
        assignments[group.group_id] = selected
        actual[selected] += group.size
        actual_strata[selected].update(group.strata_counts)
    return assignments


def _build_report(
    split_ids: Mapping[SplitName, list[str]],
    *,
    groups: tuple[_Group, ...],
    assignments: Mapping[str, SplitName],
    dimensions: tuple[StratificationDimension, ...],
    targets: Mapping[SplitName, int],
    stratum_targets: Mapping[tuple[str, str], Mapping[SplitName, int]],
) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for split_name in _SPLIT_NAMES:
        actual_count = len(split_ids[split_name])
        row = {
            "count": actual_count,
            "group_count": sum(assignments[group.group_id] == split_name for group in groups),
            "target_count": targets[split_name],
            "allocation_delta": actual_count - targets[split_name],
        }
        assigned_groups = tuple(
            group for group in groups if assignments[group.group_id] == split_name
        )
        for dimension in dimensions:
            for value in dimension.values:
                key = (dimension.dimension, value)
                actual = sum(
                    group.strata_counts[(dimension.dimension, value)] for group in assigned_groups
                )
                target = stratum_targets.get(key, {}).get(split_name, 0)
                row[f"stratum:{dimension.dimension}:{value}"] = actual
                row[f"stratum_target:{dimension.dimension}:{value}"] = target
                row[f"stratum_delta:{dimension.dimension}:{value}"] = actual - target
        report[split_name] = row
    return report


def _seeded_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest, "big")


__all__ = [
    "DataSplitResult",
    "DataSplitter",
    "SplitConfig",
    "SplitMetadataError",
    "SplitRecord",
    "StratificationDimension",
]
