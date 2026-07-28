"""Internal, non-promotional P5 PNG/JPEG comparison evidence.

The report deliberately separates encoding speed, materialized size, quality, and
end-to-end results. It binds an opt-in alternate policy to exact outcome evidence,
but never turns local or representative evidence into a production authorization.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.measurement_truth import EvidenceClass, MeasurementStatus
from robata.contracts.phase_contract_decisions import (
    OptimizationPhase,
    PhaseContractDecisionKind,
    PhaseContractDecisionRegister,
    default_phase_contract_decision_register,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]

ALTERNATE_ENCODING_QUALIFICATION_VERSION: Final[Literal["alternate-encoding-qualification-v1"]] = (
    "alternate-encoding-qualification-v1"
)


class AlternateMediaEncoding(StrEnum):
    """The baseline and opt-in encodings covered by the P5 comparison."""

    PNG = "png"
    JPEG = "jpeg"


class EncodingParityStatus(StrEnum):
    """Whether two frozen outcome surfaces have been compared."""

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_MEASURED = "NOT_MEASURED"


def _projection(model: StrictModel, *, exclude: set[str]) -> dict[str, object]:
    return model.model_dump(mode="json", exclude=exclude)


class EncodingPolicyProvenance(StrictModel):
    """Exact encoder config, rather than a MIME type alone."""

    provenance_version: Literal["encoding-policy-provenance-v1"] = "encoding-policy-provenance-v1"
    policy_digest: Sha256Digest
    policy_version: SchemaVersion
    encoding: AlternateMediaEncoding
    media_type: Literal["image/png", "image/jpeg"]
    encoder_implementation: NonEmptyString
    encoder_version: SchemaVersion
    quality_setting: PositiveInt | None = None
    chroma_subsampling: NonEmptyString
    resize_policy_version: SchemaVersion
    color_conversion_version: SchemaVersion
    metadata_policy_version: SchemaVersion

    @classmethod
    def create(cls, **values: object) -> Self:
        if "policy_digest" in values:
            raise ValueError("policy_digest is derived")
        draft = cls.model_construct(
            provenance_version="encoding-policy-provenance-v1",
            policy_digest="0" * 64,
            **values,
        )
        return cls.model_validate(
            {
                **draft.model_dump(mode="python"),
                "policy_digest": semantic_sha256(_projection(draft, exclude={"policy_digest"})),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        expected_media_type = (
            "image/png" if self.encoding == AlternateMediaEncoding.PNG else "image/jpeg"
        )
        if self.media_type != expected_media_type:
            raise ValueError("media_type does not match encoding")
        if self.encoding == AlternateMediaEncoding.PNG and self.quality_setting is not None:
            raise ValueError("PNG provenance cannot declare a lossy quality setting")
        if self.encoding == AlternateMediaEncoding.JPEG and self.quality_setting is None:
            raise ValueError("JPEG provenance requires a quality setting")
        if self.policy_digest != semantic_sha256(_projection(self, exclude={"policy_digest"})):
            raise ValueError("policy_digest does not match encoder provenance")
        return self


class SelectedFrameDimension(StrictModel):
    """One geometry/count row in a selected-frame inventory."""

    width: PositiveInt
    height: PositiveInt
    frame_count: PositiveInt


class SelectedFrameInventory(StrictModel):
    """Exact selected-frame set and its materialized geometry distribution."""

    selection_semantic_sha256: Sha256Digest
    selected_frame_count: NonNegativeInt
    dimension_counts: tuple[SelectedFrameDimension, ...] = ()

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        geometries = tuple((item.width, item.height) for item in self.dimension_counts)
        if geometries != tuple(sorted(geometries)) or len(set(geometries)) != len(geometries):
            raise ValueError("dimension_counts must be sorted and unique")
        if sum(item.frame_count for item in self.dimension_counts) != self.selected_frame_count:
            raise ValueError("dimension_counts do not total selected_frame_count")
        if bool(self.dimension_counts) != bool(self.selected_frame_count):
            raise ValueError("selected frame count and geometry inventory disagree")
        return self


class SelectedFrameComparison(StrictModel):
    """Parity of the source selections before image encoding changes bytes."""

    parity_status: EncodingParityStatus
    baseline: SelectedFrameInventory
    alternate: SelectedFrameInventory

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        same = self.baseline == self.alternate
        if self.parity_status == EncodingParityStatus.MATCH and not same:
            raise ValueError("selected-frame MATCH requires identical inventories")
        if self.parity_status == EncodingParityStatus.MISMATCH and same:
            raise ValueError("selected-frame MISMATCH requires different inventories")
        return self


class OutcomeParity(StrictModel):
    """Exact evidence for one QA, event, or boundary outcome surface."""

    parity_status: EncodingParityStatus
    baseline_count: NonNegativeInt | None = None
    alternate_count: NonNegativeInt | None = None
    baseline_semantic_sha256: Sha256Digest | None = None
    alternate_semantic_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        values = (
            self.baseline_count,
            self.alternate_count,
            self.baseline_semantic_sha256,
            self.alternate_semantic_sha256,
        )
        if self.parity_status == EncodingParityStatus.NOT_MEASURED:
            if any(value is not None for value in values):
                raise ValueError("NOT_MEASURED outcome cannot retain values")
            return self
        if any(value is None for value in values):
            raise ValueError("measured outcome requires counts and semantic digests")
        same = (
            self.baseline_count == self.alternate_count
            and self.baseline_semantic_sha256 == self.alternate_semantic_sha256
        )
        if self.parity_status == EncodingParityStatus.MATCH and not same:
            raise ValueError("outcome MATCH requires equal counts and semantic digests")
        if self.parity_status == EncodingParityStatus.MISMATCH and same:
            raise ValueError("outcome MISMATCH requires a changed count or semantic digest")
        return self


class PerClassOutcomeParity(StrictModel):
    """Separated QA, event, and boundary comparison for one label class."""

    class_id: NonEmptyString
    qa: OutcomeParity
    event: OutcomeParity
    boundary: OutcomeParity

    @property
    def fully_matching(self) -> bool:
        return all(
            item.parity_status == EncodingParityStatus.MATCH
            for item in (self.qa, self.event, self.boundary)
        )


class EncodingSpeedDelta(StrictModel):
    """Encode-only timing; it is deliberately not an end-to-end surrogate."""

    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    baseline_encode_duration_ns: Nanoseconds | None = None
    alternate_encode_duration_ns: Nanoseconds | None = None
    delta_ns: int | None = None

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        values = (
            self.baseline_encode_duration_ns,
            self.alternate_encode_duration_ns,
            self.delta_ns,
        )
        if self.measurement_status == MeasurementStatus.NOT_MEASURED:
            if any(value is not None for value in values):
                raise ValueError("NOT_MEASURED speed cannot retain values")
            return self
        if any(value is None for value in values):
            raise ValueError("MEASURED speed requires baseline, alternate, and delta")
        if self.baseline_encode_duration_ns < 0 or self.alternate_encode_duration_ns < 0:
            raise ValueError("speed durations must be nonnegative")
        if self.delta_ns != self.alternate_encode_duration_ns - self.baseline_encode_duration_ns:
            raise ValueError("speed delta does not match durations")
        return self


class EncodingSizeDelta(StrictModel):
    """Total selected-artifact bytes, independent of encode timing."""

    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    baseline_total_bytes: NonNegativeInt | None = None
    alternate_total_bytes: NonNegativeInt | None = None
    delta_bytes: int | None = None

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        values = (self.baseline_total_bytes, self.alternate_total_bytes, self.delta_bytes)
        if self.measurement_status == MeasurementStatus.NOT_MEASURED:
            if any(value is not None for value in values):
                raise ValueError("NOT_MEASURED size cannot retain values")
            return self
        if any(value is None for value in values):
            raise ValueError("MEASURED size requires baseline, alternate, and delta")
        if self.delta_bytes != self.alternate_total_bytes - self.baseline_total_bytes:
            raise ValueError("size delta does not match bytes")
        return self


class EncodingQualityDelta(StrictModel):
    """Label-grounded quality metric, independent of byte and latency changes."""

    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    metric_name: NonEmptyString | None = None
    metric_policy_digest: Sha256Digest | None = None
    baseline_score: UnitInterval | None = None
    alternate_score: UnitInterval | None = None
    delta_score: FiniteFloat | None = None

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        values = (
            self.metric_name,
            self.metric_policy_digest,
            self.baseline_score,
            self.alternate_score,
            self.delta_score,
        )
        if self.measurement_status == MeasurementStatus.NOT_MEASURED:
            if any(value is not None for value in values):
                raise ValueError("NOT_MEASURED quality cannot retain values")
            return self
        if any(value is None for value in values):
            raise ValueError("MEASURED quality requires metric identity and scores")
        if not math.isclose(
            self.delta_score,
            self.alternate_score - self.baseline_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("quality delta does not match scores")
        return self


class EncodingEndToEndDelta(StrictModel):
    """Complete provider-path duration, never substituted with encode-only timing."""

    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    baseline_duration_ns: Nanoseconds | None = None
    alternate_duration_ns: Nanoseconds | None = None
    delta_ns: int | None = None

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        values = (self.baseline_duration_ns, self.alternate_duration_ns, self.delta_ns)
        if self.measurement_status == MeasurementStatus.NOT_MEASURED:
            if any(value is not None for value in values):
                raise ValueError("NOT_MEASURED end-to-end cannot retain values")
            return self
        if any(value is None for value in values):
            raise ValueError("MEASURED end-to-end requires baseline, alternate, and delta")
        if self.baseline_duration_ns < 0 or self.alternate_duration_ns < 0:
            raise ValueError("end-to-end durations must be nonnegative")
        if self.delta_ns != self.alternate_duration_ns - self.baseline_duration_ns:
            raise ValueError("end-to-end delta does not match durations")
        return self


class RepresentativeParityEvidence(StrictModel):
    """External labels and provider facts required before default-policy consideration."""

    evidence_version: Literal["alternate-encoding-representative-parity-v1"] = (
        "alternate-encoding-representative-parity-v1"
    )
    evidence_digest: Sha256Digest
    parity_status: EncodingParityStatus = EncodingParityStatus.NOT_MEASURED
    representative_workload_digest: Sha256Digest | None = None
    external_labels_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    provider_acceptance_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    provider_replay_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    external_labels_evidence_digest: Sha256Digest | None = None
    provider_acceptance_evidence_digest: Sha256Digest | None = None
    provider_replay_evidence_digest: Sha256Digest | None = None

    @classmethod
    def create(cls, **values: object) -> Self:
        if "evidence_digest" in values:
            raise ValueError("evidence_digest is derived")
        draft = cls.model_construct(
            evidence_version="alternate-encoding-representative-parity-v1",
            evidence_digest="0" * 64,
            **values,
        )
        return cls.model_validate(
            {
                **draft.model_dump(mode="python"),
                "evidence_digest": semantic_sha256(_projection(draft, exclude={"evidence_digest"})),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        statuses = (
            self.external_labels_status,
            self.provider_acceptance_status,
            self.provider_replay_status,
        )
        if (
            self.parity_status != EncodingParityStatus.NOT_MEASURED
            and self.representative_workload_digest is None
        ):
            raise ValueError("measured representative parity requires workload digest")
        if any(status == MeasurementStatus.MEASURED for status in statuses) and (
            self.representative_workload_digest is None
        ):
            raise ValueError("representative measurements require workload digest")
        for label, status, digest in (
            ("external labels", self.external_labels_status, self.external_labels_evidence_digest),
            (
                "provider acceptance",
                self.provider_acceptance_status,
                self.provider_acceptance_evidence_digest,
            ),
            ("provider replay", self.provider_replay_status, self.provider_replay_evidence_digest),
        ):
            if (status == MeasurementStatus.MEASURED) != (digest is not None):
                raise ValueError(f"{label} status and evidence digest must agree")
        if (
            self.provider_replay_status == MeasurementStatus.MEASURED
            and self.provider_acceptance_status != MeasurementStatus.MEASURED
        ):
            raise ValueError("provider replay requires provider acceptance")
        if self.parity_status == EncodingParityStatus.MATCH and any(
            status != MeasurementStatus.MEASURED for status in statuses
        ):
            raise ValueError("representative MATCH requires labels and provider evidence")
        if self.parity_status == EncodingParityStatus.MATCH and any(
            digest is None
            for digest in (
                self.external_labels_evidence_digest,
                self.provider_acceptance_evidence_digest,
                self.provider_replay_evidence_digest,
            )
        ):
            raise ValueError("representative MATCH requires label and provider evidence digests")
        if self.evidence_digest != semantic_sha256(_projection(self, exclude={"evidence_digest"})):
            raise ValueError("evidence_digest does not match representative parity")
        return self


class RepresentativeParitySignoff(StrictModel):
    """A non-cyclic reviewer signoff over the unsigned comparison digest."""

    signoff_version: Literal["alternate-encoding-parity-signoff-v1"] = (
        "alternate-encoding-parity-signoff-v1"
    )
    signoff_digest: Sha256Digest
    comparison_digest: Sha256Digest
    signoff_id: NonEmptyString
    signer: NonEmptyString

    @classmethod
    def create(cls, **values: object) -> Self:
        if "signoff_digest" in values:
            raise ValueError("signoff_digest is derived")
        draft = cls.model_construct(
            signoff_version="alternate-encoding-parity-signoff-v1",
            signoff_digest="0" * 64,
            **values,
        )
        return cls.model_validate(
            {
                **draft.model_dump(mode="python"),
                "signoff_digest": semantic_sha256(_projection(draft, exclude={"signoff_digest"})),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_signoff(self) -> Self:
        if self.signoff_digest != semantic_sha256(_projection(self, exclude={"signoff_digest"})):
            raise ValueError("signoff_digest does not match signoff")
        return self


class AlternateEncodingQualificationReport(StrictModel):
    """Content-addressed P5 comparison that cannot authorize production by itself."""

    report_version: Literal["alternate-encoding-qualification-v1"] = (
        ALTERNATE_ENCODING_QUALIFICATION_VERSION
    )
    comparison_digest: Sha256Digest
    report_sha256: Sha256Digest
    phase_contract_decisions: PhaseContractDecisionRegister = Field(
        default_factory=default_phase_contract_decision_register
    )
    baseline_policy: EncodingPolicyProvenance
    alternate_policy: EncodingPolicyProvenance
    selected_frames: SelectedFrameComparison
    speed: EncodingSpeedDelta
    size: EncodingSizeDelta
    quality: EncodingQualityDelta
    end_to_end: EncodingEndToEndDelta
    class_outcomes: tuple[PerClassOutcomeParity, ...] = Field(min_length=1)
    representative_parity: RepresentativeParityEvidence
    parity_signoff: RepresentativeParitySignoff | None = None
    evidence_class: EvidenceClass = EvidenceClass.LOCAL_CONFORMANCE
    default_promotion_eligible: bool
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        baseline_policy: EncodingPolicyProvenance,
        alternate_policy: EncodingPolicyProvenance,
        selected_frames: SelectedFrameComparison,
        speed: EncodingSpeedDelta,
        size: EncodingSizeDelta,
        quality: EncodingQualityDelta,
        end_to_end: EncodingEndToEndDelta,
        class_outcomes: tuple[PerClassOutcomeParity, ...],
        representative_parity: RepresentativeParityEvidence,
        parity_signoff: RepresentativeParitySignoff | None = None,
        evidence_class: EvidenceClass = EvidenceClass.LOCAL_CONFORMANCE,
        phase_contract_decisions: PhaseContractDecisionRegister | None = None,
    ) -> Self:
        draft = cls.model_construct(
            report_version=ALTERNATE_ENCODING_QUALIFICATION_VERSION,
            comparison_digest="0" * 64,
            report_sha256="0" * 64,
            phase_contract_decisions=(
                default_phase_contract_decision_register()
                if phase_contract_decisions is None
                else phase_contract_decisions
            ),
            baseline_policy=baseline_policy,
            alternate_policy=alternate_policy,
            selected_frames=selected_frames,
            speed=speed,
            size=size,
            quality=quality,
            end_to_end=end_to_end,
            class_outcomes=class_outcomes,
            representative_parity=representative_parity,
            parity_signoff=parity_signoff,
            evidence_class=EvidenceClass(evidence_class),
            default_promotion_eligible=False,
            production_eligible=False,
        )
        comparison_digest = semantic_sha256(_comparison_projection(draft))
        with_comparison = draft.model_copy(update={"comparison_digest": comparison_digest})
        final = with_comparison.model_copy(
            update={"default_promotion_eligible": _default_promotion_eligible(with_comparison)}
        )
        return cls.model_validate(
            {
                **final.model_dump(mode="python"),
                "report_sha256": semantic_sha256(_report_projection(final)),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        decision = self.phase_contract_decisions.require_dispatchable(OptimizationPhase.P5)
        if decision.decision != PhaseContractDecisionKind.INTERNAL_VERSIONED_CHANGE:
            raise ValueError("P5 requires internal versioned decision")
        if (
            self.baseline_policy.encoding != AlternateMediaEncoding.PNG
            or self.alternate_policy.encoding != AlternateMediaEncoding.JPEG
        ):
            raise ValueError("P5 requires PNG baseline and opt-in JPEG alternate")
        class_ids = tuple(item.class_id for item in self.class_outcomes)
        if class_ids != tuple(sorted(class_ids)) or len(set(class_ids)) != len(class_ids):
            raise ValueError("class outcomes must be sorted and unique")
        if (
            self.quality.measurement_status == MeasurementStatus.MEASURED
            and self.representative_parity.external_labels_status != MeasurementStatus.MEASURED
        ):
            raise ValueError("quality cannot be MEASURED without external labels")
        if self.end_to_end.measurement_status == MeasurementStatus.MEASURED and (
            self.representative_parity.provider_acceptance_status != MeasurementStatus.MEASURED
            or self.representative_parity.provider_replay_status != MeasurementStatus.MEASURED
        ):
            raise ValueError("end-to-end evidence requires provider acceptance and replay")
        if self.representative_parity.parity_status == EncodingParityStatus.MATCH:
            if self.selected_frames.parity_status != EncodingParityStatus.MATCH or not all(
                item.fully_matching for item in self.class_outcomes
            ):
                raise ValueError(
                    "representative MATCH requires selected-frame and per-class parity"
                )
            if any(
                item.measurement_status != MeasurementStatus.MEASURED
                for item in (self.speed, self.size, self.quality, self.end_to_end)
            ):
                raise ValueError("representative MATCH requires every delta surface")
        if self.parity_signoff is not None and (
            self.representative_parity.parity_status != EncodingParityStatus.MATCH
            or self.parity_signoff.comparison_digest != self.comparison_digest
        ):
            raise ValueError("signoff must bind representative MATCH comparison")
        if self.comparison_digest != semantic_sha256(_comparison_projection(self)):
            raise ValueError("comparison_digest does not match unsigned comparison")
        if self.default_promotion_eligible != _default_promotion_eligible(self):
            raise ValueError("default_promotion_eligible does not match evidence")
        if self.report_sha256 != semantic_sha256(_report_projection(self)):
            raise ValueError("report_sha256 does not match report")
        return self


def _comparison_projection(report: AlternateEncodingQualificationReport) -> dict[str, object]:
    return _projection(
        report,
        exclude={
            "comparison_digest",
            "report_sha256",
            "parity_signoff",
            "default_promotion_eligible",
        },
    )


def _report_projection(report: AlternateEncodingQualificationReport) -> dict[str, object]:
    return _projection(report, exclude={"report_sha256"})


def _default_promotion_eligible(report: AlternateEncodingQualificationReport) -> bool:
    return (
        report.evidence_class == EvidenceClass.REPRESENTATIVE_BENCHMARK
        and report.representative_parity.parity_status == EncodingParityStatus.MATCH
        and report.parity_signoff is not None
        and report.selected_frames.parity_status == EncodingParityStatus.MATCH
        and all(item.fully_matching for item in report.class_outcomes)
        and all(
            item.measurement_status == MeasurementStatus.MEASURED
            for item in (report.speed, report.size, report.quality, report.end_to_end)
        )
    )


__all__ = [
    "ALTERNATE_ENCODING_QUALIFICATION_VERSION",
    "AlternateEncodingQualificationReport",
    "AlternateMediaEncoding",
    "EncodingEndToEndDelta",
    "EncodingParityStatus",
    "EncodingPolicyProvenance",
    "EncodingQualityDelta",
    "EncodingSizeDelta",
    "EncodingSpeedDelta",
    "OutcomeParity",
    "PerClassOutcomeParity",
    "RepresentativeParityEvidence",
    "RepresentativeParitySignoff",
    "SelectedFrameComparison",
    "SelectedFrameDimension",
    "SelectedFrameInventory",
]
