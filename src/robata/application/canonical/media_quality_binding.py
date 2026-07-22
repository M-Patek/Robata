"""Internal canonical binding for local media-quality observations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from robata.application.canonical.media_quality import (
    LocalMediaQualityReport,
    LocalQualityFlag,
    registered_local_media_quality_report_document,
    validate_registered_local_media_quality_report_document,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRegistry

LOCAL_MEDIA_QUALITY_BINDING_PROJECTION_VERSION: Literal[
    "canonical-local-media-quality-binding-semantic-v1"
] = "canonical-local-media-quality-binding-semantic-v1"


class LocalMediaQualityFlagCount(StrictModel):
    """Number of concrete local observations carrying one exact flag."""

    flag: LocalQualityFlag
    occurrence_count: Annotated[int, Field(strict=True, ge=1)]


class LocalMediaQualityBinding(StrictModel):
    """Non-promotional quality evidence bound to one deterministic source report."""

    projection_version: Literal["canonical-local-media-quality-binding-semantic-v1"] = (
        LOCAL_MEDIA_QUALITY_BINDING_PROJECTION_VERSION
    )
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    production_eligible: Literal[False] = False
    quality_policy_version: SchemaVersion
    neighbor_target_policy_version: SchemaVersion
    report_semantic_sha256: Sha256Digest
    supplemental_target_plan_semantic_sha256: Sha256Digest
    flag_counts: tuple[LocalMediaQualityFlagCount, ...]
    requires_review: bool
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        flags = tuple(item.flag for item in self.flag_counts)
        if flags != tuple(sorted(set(flags), key=lambda value: value.value)):
            raise ValueError("media-quality flag counts must be unique and canonically ordered")
        if self.requires_review is not bool(self.flag_counts):
            raise ValueError("requires_review must reflect the presence of quality observations")
        expected = semantic_sha256(local_media_quality_binding_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("media-quality binding semantic_sha256 is inconsistent")
        return self


def derive_local_media_quality_binding(
    report: LocalMediaQualityReport,
) -> LocalMediaQualityBinding:
    """Validate and reduce a local report without inferring semantic defects."""

    if not isinstance(report, LocalMediaQualityReport):
        raise TypeError("report must be a LocalMediaQualityReport")

    # This validates the registered wire document and both semantic digests.
    registered_local_media_quality_report_document(report, SchemaRegistry())
    return _create_binding(
        quality_policy_version=report.policy_version,
        neighbor_target_policy_version=report.supplemental_targets.policy_version,
        report_semantic_sha256=report.semantic_sha256,
        supplemental_target_plan_semantic_sha256=(report.supplemental_targets.semantic_sha256),
        flag_counts=_quality_flag_counts(report),
    )


def derive_local_media_quality_binding_document(
    document: Mapping[str, object],
    registry: SchemaRegistry,
) -> LocalMediaQualityBinding:
    """Derive the same binding from exact persisted report bytes during recovery."""

    validated = validate_registered_local_media_quality_report_document(document, registry)
    supplemental = _mapping(validated["supplemental_targets"], "supplemental_targets")
    return _create_binding(
        quality_policy_version=_string(validated["policy_version"], "policy_version"),
        neighbor_target_policy_version=_string(
            supplemental["policy_version"],
            "supplemental_targets.policy_version",
        ),
        report_semantic_sha256=_string(
            validated["semantic_sha256"],
            "semantic_sha256",
        ),
        supplemental_target_plan_semantic_sha256=_string(
            supplemental["semantic_sha256"],
            "supplemental_targets.semantic_sha256",
        ),
        flag_counts=_document_quality_flag_counts(validated),
    )


def _create_binding(
    *,
    quality_policy_version: str,
    neighbor_target_policy_version: str,
    report_semantic_sha256: str,
    supplemental_target_plan_semantic_sha256: str,
    flag_counts: tuple[LocalMediaQualityFlagCount, ...],
) -> LocalMediaQualityBinding:
    projection = _local_media_quality_binding_projection_values(
        quality_policy_version=quality_policy_version,
        neighbor_target_policy_version=neighbor_target_policy_version,
        report_semantic_sha256=report_semantic_sha256,
        supplemental_target_plan_semantic_sha256=(supplemental_target_plan_semantic_sha256),
        flag_counts=flag_counts,
        requires_review=bool(flag_counts),
    )
    return LocalMediaQualityBinding(
        quality_policy_version=quality_policy_version,
        neighbor_target_policy_version=neighbor_target_policy_version,
        report_semantic_sha256=report_semantic_sha256,
        supplemental_target_plan_semantic_sha256=(supplemental_target_plan_semantic_sha256),
        flag_counts=flag_counts,
        requires_review=bool(flag_counts),
        semantic_sha256=semantic_sha256(projection),
    )


def local_media_quality_binding_projection(
    binding: LocalMediaQualityBinding,
) -> dict[str, object]:
    """Return the complete semantic projection, excluding its own digest."""

    if not isinstance(binding, LocalMediaQualityBinding):
        raise TypeError("binding must be a LocalMediaQualityBinding")
    return _local_media_quality_binding_projection_values(
        quality_policy_version=binding.quality_policy_version,
        neighbor_target_policy_version=binding.neighbor_target_policy_version,
        report_semantic_sha256=binding.report_semantic_sha256,
        supplemental_target_plan_semantic_sha256=(binding.supplemental_target_plan_semantic_sha256),
        flag_counts=binding.flag_counts,
        requires_review=binding.requires_review,
    )


def _local_media_quality_binding_projection_values(
    *,
    quality_policy_version: str,
    neighbor_target_policy_version: str,
    report_semantic_sha256: str,
    supplemental_target_plan_semantic_sha256: str,
    flag_counts: tuple[LocalMediaQualityFlagCount, ...],
    requires_review: bool,
) -> dict[str, object]:
    return {
        "projection_version": LOCAL_MEDIA_QUALITY_BINDING_PROJECTION_VERSION,
        "evidence_class": "LOCAL_CONFORMANCE",
        "production_eligible": False,
        "quality_policy_version": quality_policy_version,
        "neighbor_target_policy_version": neighbor_target_policy_version,
        "report_semantic_sha256": report_semantic_sha256,
        "supplemental_target_plan_semantic_sha256": (supplemental_target_plan_semantic_sha256),
        "flag_counts": [
            {
                "flag": item.flag.value,
                "occurrence_count": item.occurrence_count,
            }
            for item in flag_counts
        ],
        "requires_review": requires_review,
    }


def _document_quality_flag_counts(
    document: Mapping[str, object],
) -> tuple[LocalMediaQualityFlagCount, ...]:
    counts: Counter[LocalQualityFlag] = Counter()
    ledgers = _sequence(document["camera_ledgers"], "camera_ledgers")
    camera_ids: list[CameraId] = []
    for ordinal, raw_ledger in enumerate(ledgers):
        ledger = _mapping(raw_ledger, f"camera_ledgers[{ordinal}]")
        camera_id = CameraId(_string(ledger["camera_id"], f"camera_ledgers[{ordinal}].camera_id"))
        camera_ids.append(camera_id)
        aggregate_flags: set[LocalQualityFlag] = set()
        observations = _sequence(
            ledger["decoded_observations"],
            f"camera_ledgers[{ordinal}].decoded_observations",
        )
        for observation_ordinal, raw_observation in enumerate(observations):
            observation = _mapping(
                raw_observation,
                (f"camera_ledgers[{ordinal}].decoded_observations[{observation_ordinal}]"),
            )
            flags = _quality_flags(
                observation["flags"],
                (f"camera_ledgers[{ordinal}].decoded_observations[{observation_ordinal}].flags"),
            )
            counts.update(flags)
            aggregate_flags.update(flags)
        cadence_gaps = _sequence(
            ledger["cadence_gaps"],
            f"camera_ledgers[{ordinal}].cadence_gaps",
        )
        sequence_gaps = _sequence(
            ledger["sequence_gaps"],
            f"camera_ledgers[{ordinal}].sequence_gaps",
        )
        if cadence_gaps:
            aggregate_flags.add(LocalQualityFlag.OBSERVED_CADENCE_GAP)
            counts[LocalQualityFlag.OBSERVED_CADENCE_GAP] += len(cadence_gaps)
        if sequence_gaps:
            aggregate_flags.add(LocalQualityFlag.OBSERVED_SEQUENCE_GAP)
            counts[LocalQualityFlag.OBSERVED_SEQUENCE_GAP] += len(sequence_gaps)
        ledger_flags = _quality_flags(
            ledger["flags"],
            f"camera_ledgers[{ordinal}].flags",
        )
        expected_flags = tuple(sorted(aggregate_flags, key=lambda value: value.value))
        if ledger_flags != expected_flags:
            raise ValueError("camera media-quality flags disagree with detailed observations")
    if tuple(camera_ids) != CAMERA_IDS:
        raise ValueError("media-quality camera ledgers must use canonical camera order")

    skew = _mapping(document["cross_camera_skew"], "cross_camera_skew")
    threshold_ns = int(_string(skew["threshold_ns"], "cross_camera_skew.threshold_ns"))
    samples = _sequence(skew["samples"], "cross_camera_skew.samples")
    flagged_skew_count = 0
    for ordinal, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"cross_camera_skew.samples[{ordinal}]")
        skew_ns = int(
            _string(
                sample["skew_ns"],
                f"cross_camera_skew.samples[{ordinal}].skew_ns",
            )
        )
        flagged_skew_count += skew_ns > threshold_ns
    skew_flags = _quality_flags(skew["flags"], "cross_camera_skew.flags")
    expected_skew_flags = (
        (LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW,) if flagged_skew_count else ()
    )
    if skew_flags != expected_skew_flags:
        raise ValueError("cross-camera skew flags disagree with detailed samples")
    if flagged_skew_count:
        counts[LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW] = flagged_skew_count

    return tuple(
        LocalMediaQualityFlagCount(flag=flag, occurrence_count=counts[flag])
        for flag in sorted(counts, key=lambda value: value.value)
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _quality_flags(value: object, label: str) -> tuple[LocalQualityFlag, ...]:
    return tuple(
        LocalQualityFlag(_string(item, f"{label}[{ordinal}]"))
        for ordinal, item in enumerate(_sequence(value, label))
    )


def _quality_flag_counts(
    report: LocalMediaQualityReport,
) -> tuple[LocalMediaQualityFlagCount, ...]:
    counts: Counter[LocalQualityFlag] = Counter()
    if tuple(ledger.camera_id for ledger in report.camera_ledgers) != CAMERA_IDS:
        raise ValueError("media-quality camera ledgers must use canonical camera order")

    for ledger in report.camera_ledgers:
        aggregate_flags: set[LocalQualityFlag] = set()
        for observation in ledger.decoded_observations:
            if observation.camera_id is not ledger.camera_id:
                raise ValueError("frame quality observation is bound to the wrong camera")
            counts.update(observation.flags)
            aggregate_flags.update(observation.flags)
        if any(gap.camera_id is not ledger.camera_id for gap in ledger.cadence_gaps):
            raise ValueError("cadence-gap observation is bound to the wrong camera")
        if any(gap.camera_id is not ledger.camera_id for gap in ledger.sequence_gaps):
            raise ValueError("sequence-gap observation is bound to the wrong camera")
        if ledger.cadence_gaps:
            aggregate_flags.add(LocalQualityFlag.OBSERVED_CADENCE_GAP)
            counts[LocalQualityFlag.OBSERVED_CADENCE_GAP] += len(ledger.cadence_gaps)
        if ledger.sequence_gaps:
            aggregate_flags.add(LocalQualityFlag.OBSERVED_SEQUENCE_GAP)
            counts[LocalQualityFlag.OBSERVED_SEQUENCE_GAP] += len(ledger.sequence_gaps)
        if ledger.flags != tuple(sorted(aggregate_flags, key=lambda value: value.value)):
            raise ValueError("camera media-quality flags disagree with detailed observations")

    skew = report.cross_camera_skew
    flagged_skew_count = sum(sample.skew_ns > skew.threshold_ns for sample in skew.samples)
    expected_skew_flags = (
        (LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW,) if flagged_skew_count else ()
    )
    if skew.flags != expected_skew_flags:
        raise ValueError("cross-camera skew flags disagree with detailed samples")
    if flagged_skew_count:
        counts[LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW] = flagged_skew_count

    return tuple(
        LocalMediaQualityFlagCount(flag=flag, occurrence_count=counts[flag])
        for flag in sorted(counts, key=lambda value: value.value)
    )


__all__ = [
    "LOCAL_MEDIA_QUALITY_BINDING_PROJECTION_VERSION",
    "LocalMediaQualityBinding",
    "LocalMediaQualityFlagCount",
    "derive_local_media_quality_binding",
    "derive_local_media_quality_binding_document",
    "local_media_quality_binding_projection",
]
