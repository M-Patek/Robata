"""Read-only evidence ledgers for historical atomic-event experiment reports.

The A0--A3 artifacts predate the current input-observation contract.  This
module projects what those artifacts actually retain into a small local table;
it marks absent observability facts as unknown instead of reconstructing or
inferring them.  In particular, a legacy ``input_size`` is retained as a
pre-runtime transform value and is never presented as final thumbnail geometry.

No media is decoded, no model is invoked, and no legacy lexical result is
recomputed.  The input mappings are read only and are not mutated.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class LegacyExperimentEvidenceError(ValueError):
    """Raised when a purported historical report cannot be read as records."""


class LegacyEvidenceAvailability(StrEnum):
    """Whether an artifact explicitly retained a requested observation."""

    KNOWN = "known"
    UNKNOWN = "unknown"


class HistoricalAtomicExperimentArm(StrEnum):
    """The four historical H8 Qwen arm labels used by the baseline ledger."""

    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"


_PROFILE_TO_ARM: Mapping[str, HistoricalAtomicExperimentArm] = {
    "context-focus": HistoricalAtomicExperimentArm.A0,
    "context-focus-microburst": HistoricalAtomicExperimentArm.A1,
    "context-focus-microburst-stn-roi": HistoricalAtomicExperimentArm.A2,
    "context-focus-microburst-hybrid-roi": HistoricalAtomicExperimentArm.A3,
}


@dataclass(frozen=True, slots=True)
class LegacyEvidenceField:
    """One retained fact or an explicit statement that the legacy row lacks it."""

    availability: LegacyEvidenceAvailability
    value: Any = None
    note: str = ""

    @classmethod
    def known(cls, value: Any, *, note: str = "") -> LegacyEvidenceField:
        return cls(LegacyEvidenceAvailability.KNOWN, value=value, note=note)

    @classmethod
    def unknown(cls, *, note: str) -> LegacyEvidenceField:
        return cls(LegacyEvidenceAvailability.UNKNOWN, note=note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability.value,
            "value": self.value,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LegacyAtomicExperimentCaseLedger:
    """The retained/missing observability matrix for one historical case row."""

    ordinal: int
    uid: str | None
    source_frames: LegacyEvidenceField
    transform_trace: LegacyEvidenceField
    roi: LegacyEvidenceField
    legacy_pre_runtime_input_size: LegacyEvidenceField
    final_thumbnail_geometry: LegacyEvidenceField
    processor_grid: LegacyEvidenceField
    processor_tensor_shape: LegacyEvidenceField
    raw_output: LegacyEvidenceField
    lexical_outcome: LegacyEvidenceField
    latency: LegacyEvidenceField

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "uid": self.uid,
            "source_frames": self.source_frames.to_dict(),
            "transform_trace": self.transform_trace.to_dict(),
            "roi": self.roi.to_dict(),
            "legacy_pre_runtime_input_size": self.legacy_pre_runtime_input_size.to_dict(),
            "final_thumbnail_geometry": self.final_thumbnail_geometry.to_dict(),
            "processor_grid": self.processor_grid.to_dict(),
            "processor_tensor_shape": self.processor_tensor_shape.to_dict(),
            "raw_output": self.raw_output.to_dict(),
            "lexical_outcome": self.lexical_outcome.to_dict(),
            "latency": self.latency.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LegacyAtomicExperimentArmLedger:
    """Read-only projection for one A0--A3 report and its recorded stop outcome."""

    arm: HistoricalAtomicExperimentArm | str | None
    input_profile: str | None
    cases: tuple[LegacyAtomicExperimentCaseLedger, ...]
    stop_outcome: LegacyEvidenceField

    def to_dict(self) -> dict[str, Any]:
        arm = self.arm.value if isinstance(self.arm, HistoricalAtomicExperimentArm) else self.arm
        return {
            "arm": arm,
            "input_profile": self.input_profile,
            "case_count": len(self.cases),
            "stop_outcome": self.stop_outcome.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


_CASE_FIELDS = (
    "source_frames",
    "transform_trace",
    "roi",
    "legacy_pre_runtime_input_size",
    "final_thumbnail_geometry",
    "processor_grid",
    "processor_tensor_shape",
    "raw_output",
    "lexical_outcome",
    "latency",
)


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _nested_mapping(mapping: Mapping[str, Any], *names: str) -> Mapping[str, Any] | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, Mapping):
            return value
    return None


def _first_nested(mapping: Mapping[str, Any], *names: str) -> Any:
    value = _first(mapping, *names)
    if value is not None:
        return value
    for container in (
        _nested_mapping(mapping, "spatial_focus"),
        _nested_mapping(mapping, "hybrid_focus"),
        _nested_mapping(mapping, "focus_microburst"),
        _nested_mapping(mapping, "processor_observation"),
        _nested_mapping(mapping, "processor_geometry"),
    ):
        if container is not None:
            value = _first(container, *names)
            if value is not None:
                return value
    return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _as_case_records(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    if isinstance(report, Mapping):
        for field in ("cases", "records"):
            if field not in report:
                continue
            value = report[field]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                raise LegacyExperimentEvidenceError(f"legacy report {field} must be a sequence")
            if not all(isinstance(item, Mapping) for item in value):
                raise LegacyExperimentEvidenceError(f"legacy report {field} must contain objects")
            return report, tuple(value)
        # A single JSON-like result row is also a useful diagnostic input.
        return report, (report,)
    if isinstance(report, Sequence) and not isinstance(report, (str, bytes, bytearray)):
        if not all(isinstance(item, Mapping) for item in report):
            raise LegacyExperimentEvidenceError("legacy result records must be objects")
        return {}, tuple(report)
    raise LegacyExperimentEvidenceError("legacy report must be an object or a sequence of objects")


def infer_historical_arm(
    report: Mapping[str, Any],
    *,
    arm: HistoricalAtomicExperimentArm | str | None = None,
) -> HistoricalAtomicExperimentArm | str | None:
    """Name an A0--A3 arm only from an explicit label or retained profile."""

    if arm is not None:
        if isinstance(arm, HistoricalAtomicExperimentArm):
            return arm
        normalized = str(arm).strip().upper()
        try:
            return HistoricalAtomicExperimentArm(normalized)
        except ValueError:
            return str(arm).strip() or None

    explicit = _text(_first(report, "arm", "arm_id", "experiment_arm"))
    if explicit is not None:
        normalized = explicit.upper()
        try:
            return HistoricalAtomicExperimentArm(normalized)
        except ValueError:
            return explicit
    profile = _text(_first(report, "input_profile", "profile"))
    return _PROFILE_TO_ARM.get(profile) if profile is not None else None


def _source_frames(case: Mapping[str, Any]) -> LegacyEvidenceField:
    source_frames = _first(case, "source_frames", "source_frame_indices")
    context = _first(case, "context_frame_indices", "context_source_frame_indices")
    focus = _first(case, "focus_frame_indices", "focus_source_frame_indices")
    timestamps = _first(case, "source_frame_timestamps", "frame_timestamps")
    if source_frames is not None:
        return LegacyEvidenceField.known(source_frames, note="legacy source-frame field retained")
    if context is not None or focus is not None or timestamps is not None:
        value = {
            "context_frame_indices": context,
            "focus_frame_indices": focus,
            "source_frame_timestamps": timestamps,
        }
        return LegacyEvidenceField.known(value, note="legacy selected source-frame fields retained")
    return LegacyEvidenceField.unknown(note="legacy row has no selected source-frame field")


def _transform_trace(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first_nested(
        case,
        "per_frame_transform_trace",
        "transform_trace",
        "frame_transforms",
        "per_frame_transforms",
    )
    if value is not None:
        return LegacyEvidenceField.known(value, note="per-frame transform trace retained")
    return LegacyEvidenceField.unknown(
        note="legacy ROI diagnostics do not prove a per-frame transform trace"
    )


def _roi(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first_nested(case, "roi_xyxy", "roi", "suggested_fixed_roi_xyxy")
    if value is not None:
        return LegacyEvidenceField.known(value, note="legacy ROI field retained")
    return LegacyEvidenceField.unknown(note="legacy row has no ROI field")


def _pre_runtime_input_size(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first_nested(case, "input_size", "transform_size", "render_input_size")
    if value is not None:
        return LegacyEvidenceField.known(
            value,
            note="legacy pre-runtime transform size; not final thumbnail geometry",
        )
    return LegacyEvidenceField.unknown(note="legacy row has no pre-runtime transform-size field")


def _final_thumbnail_geometry(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first_nested(
        case,
        "final_thumbnail_geometry",
        "post_thumbnail_geometry",
        "post_thumbnail_sizes",
        "thumbnail_sizes",
        "rendered_thumbnail_sizes",
    )
    if value is not None:
        return LegacyEvidenceField.known(value, note="post-thumbnail geometry retained")
    return LegacyEvidenceField.unknown(
        note="legacy input_size is pre-runtime and cannot establish final thumbnail geometry"
    )


def _processor_grid(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first_nested(case, "video_grid_thw", "processor_grid", "video_grid")
    if value is not None:
        return LegacyEvidenceField.known(value, note="processor grid retained")
    return LegacyEvidenceField.unknown(note="legacy row has no processor grid")


def _processor_tensor_shape(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first_nested(
        case,
        "processor_tensor_shape",
        "processor_tensor_shapes",
        "tensor_shape",
        "tensor_shapes",
    )
    if value is not None:
        return LegacyEvidenceField.known(value, note="processor tensor shape retained")
    return LegacyEvidenceField.unknown(note="legacy row has no processor tensor shape")


def _raw_output(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first(case, "raw_output_text", "output_text")
    if isinstance(value, str):
        return LegacyEvidenceField.known(value, note="retained model text")
    return LegacyEvidenceField.unknown(note="legacy row has no retained raw output text")


def _lexical_outcome(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first(case, "quality", "lexical_outcome", "raw_quality")
    if isinstance(value, Mapping):
        return LegacyEvidenceField.known(value, note="retained lexical result; not recomputed")
    return LegacyEvidenceField.unknown(note="legacy row has no retained lexical result")


def _latency(case: Mapping[str, Any]) -> LegacyEvidenceField:
    value = _first(case, "generation_seconds", "latency_seconds", "elapsed_seconds")
    if value is not None:
        return LegacyEvidenceField.known(value, note="retained timing value")
    return LegacyEvidenceField.unknown(note="legacy row has no retained timing value")


def project_legacy_case_record(
    record: Mapping[str, Any],
    *,
    ordinal: int = 0,
) -> LegacyAtomicExperimentCaseLedger:
    """Project a legacy result row without modifying it or scoring its text."""

    if not isinstance(record, Mapping):
        raise LegacyExperimentEvidenceError("legacy case record must be an object")
    raw_uid = record.get("uid", record.get("case_id"))
    uid = str(raw_uid) if raw_uid is not None else None
    return LegacyAtomicExperimentCaseLedger(
        ordinal=ordinal,
        uid=uid,
        source_frames=_source_frames(record),
        transform_trace=_transform_trace(record),
        roi=_roi(record),
        legacy_pre_runtime_input_size=_pre_runtime_input_size(record),
        final_thumbnail_geometry=_final_thumbnail_geometry(record),
        processor_grid=_processor_grid(record),
        processor_tensor_shape=_processor_tensor_shape(record),
        raw_output=_raw_output(record),
        lexical_outcome=_lexical_outcome(record),
        latency=_latency(record),
    )


def _stop_outcome(
    document: Mapping[str, Any],
    supplied: Mapping[str, Any] | str | None,
) -> LegacyEvidenceField:
    source: Mapping[str, Any] | None
    if isinstance(supplied, Mapping):
        source = supplied
    elif isinstance(supplied, str):
        return LegacyEvidenceField.known(supplied, note="recorded supplied branch outcome")
    elif supplied is None:
        source = document
    else:
        raise LegacyExperimentEvidenceError("stop outcome must be an object, string, or null")

    value = _first(source, "stop_outcome", "decision", "branch_outcome", "outcome")
    if value is not None:
        return LegacyEvidenceField.known(value, note="recorded historical branch outcome")
    return LegacyEvidenceField.unknown(note="no recorded stop or branch outcome in legacy artifact")


def project_legacy_experiment_report(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    arm: HistoricalAtomicExperimentArm | str | None = None,
    stop_outcome: Mapping[str, Any] | str | None = None,
) -> LegacyAtomicExperimentArmLedger:
    """Project one historical A0--A3 report into an explicit evidence ledger."""

    document, records = _as_case_records(report)
    input_profile = _text(_first(document, "input_profile", "profile"))
    return LegacyAtomicExperimentArmLedger(
        arm=infer_historical_arm(document, arm=arm),
        input_profile=input_profile,
        cases=tuple(
            project_legacy_case_record(record, ordinal=ordinal)
            for ordinal, record in enumerate(records)
        ),
        stop_outcome=_stop_outcome(document, stop_outcome),
    )


def _arm_key(arm: HistoricalAtomicExperimentArm | str | None, ordinal: int) -> str:
    if isinstance(arm, HistoricalAtomicExperimentArm):
        return arm.value
    if isinstance(arm, str) and arm:
        return arm
    return f"unknown-{ordinal}"


def build_historical_a0_a3_ledger(
    reports: Mapping[
        HistoricalAtomicExperimentArm | str,
        Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ],
    *,
    stop_outcomes: Mapping[str, Mapping[str, Any] | str] | None = None,
) -> dict[str, Any]:
    """Build one comparison-ready, read-only ledger from A0--A3 legacy reports.

    ``stop_outcomes`` lets a comparison artifact (for example P29.3's recorded
    stop decision) be attached to its arm without changing either historical
    source mapping.
    """

    projected: list[LegacyAtomicExperimentArmLedger] = []
    for supplied_arm, report in reports.items():
        supplied_stop = None
        if stop_outcomes is not None:
            supplied_stop = stop_outcomes.get(str(supplied_arm))
        projected.append(
            project_legacy_experiment_report(
                report,
                arm=supplied_arm,
                stop_outcome=supplied_stop,
            )
        )

    unknown_counts: Counter[str] = Counter()
    for arm_ledger in projected:
        for case in arm_ledger.cases:
            for field_name in _CASE_FIELDS:
                field = getattr(case, field_name)
                if field.availability is LegacyEvidenceAvailability.UNKNOWN:
                    unknown_counts[field_name] += 1

    arm_rows = [ledger.to_dict() for ledger in projected]
    stop_rows = {
        _arm_key(ledger.arm, ordinal): ledger.stop_outcome.to_dict()
        for ordinal, ledger in enumerate(projected)
    }
    return {
        "arms": arm_rows,
        "case_count": sum(len(ledger.cases) for ledger in projected),
        "unknown_case_field_counts": dict(sorted(unknown_counts.items())),
        "stop_outcomes": stop_rows,
        "interpretation_boundary": (
            "Known legacy scalar outcomes do not establish source-to-processor visual parity; "
            "unknown fields remain unknown."
        ),
    }


# The generic name is useful for callers that are not concerned with the arm
# labels, while the A0--A3 name keeps the intended historical scope obvious.
build_legacy_experiment_evidence_ledger = build_historical_a0_a3_ledger
