from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.benchmark.alternate_encoding_qualification import (
    AlternateEncodingQualificationReport,
    AlternateMediaEncoding,
    EncodingEndToEndDelta,
    EncodingParityStatus,
    EncodingPolicyProvenance,
    EncodingQualityDelta,
    EncodingSizeDelta,
    EncodingSpeedDelta,
    OutcomeParity,
    PerClassOutcomeParity,
    RepresentativeParityEvidence,
    RepresentativeParitySignoff,
    SelectedFrameComparison,
    SelectedFrameDimension,
    SelectedFrameInventory,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.measurement_truth import EvidenceClass, MeasurementStatus


def _digest(value: str) -> str:
    return semantic_sha256({"alternate-encoding-test": value})


def _policy(encoding: AlternateMediaEncoding) -> EncodingPolicyProvenance:
    return EncodingPolicyProvenance.create(
        policy_version=f"mcap-{encoding.value}-policy-v1",
        encoding=encoding,
        media_type="image/png" if encoding == AlternateMediaEncoding.PNG else "image/jpeg",
        encoder_implementation="pyav",
        encoder_version="pyav-18.0.0",
        quality_setting=None if encoding == AlternateMediaEncoding.PNG else 2,
        chroma_subsampling="rgb24" if encoding == AlternateMediaEncoding.PNG else "yuvj420p",
        resize_policy_version="evidence-max-width-320-v1",
        color_conversion_version="rgb24-conversion-v1",
        metadata_policy_version="bitexact-metadata-v1",
    )


def _inventory(seed: str = "selection") -> SelectedFrameInventory:
    return SelectedFrameInventory(
        selection_semantic_sha256=_digest(seed),
        selected_frame_count=3,
        dimension_counts=(
            SelectedFrameDimension(width=320, height=180, frame_count=2),
            SelectedFrameDimension(width=320, height=240, frame_count=1),
        ),
    )


def _outcome(
    status: EncodingParityStatus = EncodingParityStatus.MATCH,
) -> PerClassOutcomeParity:
    def surface(name: str) -> OutcomeParity:
        if status == EncodingParityStatus.NOT_MEASURED:
            return OutcomeParity(parity_status=status)
        baseline = _digest(f"{name}-baseline")
        alternate = baseline if status == EncodingParityStatus.MATCH else _digest(f"{name}-jpeg")
        return OutcomeParity(
            parity_status=status,
            baseline_count=4,
            alternate_count=4,
            baseline_semantic_sha256=baseline,
            alternate_semantic_sha256=alternate,
        )

    return PerClassOutcomeParity(
        class_id="ACTION",
        qa=surface("qa"),
        event=surface("event"),
        boundary=surface("boundary"),
    )


def _measured_deltas() -> tuple[
    EncodingSpeedDelta,
    EncodingSizeDelta,
    EncodingQualityDelta,
    EncodingEndToEndDelta,
]:
    return (
        EncodingSpeedDelta(
            measurement_status=MeasurementStatus.MEASURED,
            baseline_encode_duration_ns=100,
            alternate_encode_duration_ns=80,
            delta_ns=-20,
        ),
        EncodingSizeDelta(
            measurement_status=MeasurementStatus.MEASURED,
            baseline_total_bytes=1000,
            alternate_total_bytes=700,
            delta_bytes=-300,
        ),
        EncodingQualityDelta(
            measurement_status=MeasurementStatus.MEASURED,
            metric_name="qa_macro_f1",
            metric_policy_digest=_digest("metric-policy"),
            baseline_score=0.8,
            alternate_score=0.8,
            delta_score=0.0,
        ),
        EncodingEndToEndDelta(
            measurement_status=MeasurementStatus.MEASURED,
            baseline_duration_ns=1000,
            alternate_duration_ns=900,
            delta_ns=-100,
        ),
    )


def _representative(
    status: EncodingParityStatus = EncodingParityStatus.MATCH,
) -> RepresentativeParityEvidence:
    if status == EncodingParityStatus.NOT_MEASURED:
        return RepresentativeParityEvidence.create()
    return RepresentativeParityEvidence.create(
        parity_status=status,
        representative_workload_digest=_digest("representative-workload"),
        external_labels_status=MeasurementStatus.MEASURED,
        provider_acceptance_status=MeasurementStatus.MEASURED,
        provider_replay_status=MeasurementStatus.MEASURED,
        external_labels_evidence_digest=_digest("external-labels"),
        provider_acceptance_evidence_digest=_digest("provider-acceptance"),
        provider_replay_evidence_digest=_digest("provider-replay"),
    )


def _report(
    *,
    parity: EncodingParityStatus = EncodingParityStatus.MATCH,
    outcome: EncodingParityStatus = EncodingParityStatus.MATCH,
    signoff: RepresentativeParitySignoff | None = None,
    evidence_class: EvidenceClass = EvidenceClass.REPRESENTATIVE_BENCHMARK,
) -> AlternateEncodingQualificationReport:
    speed, size, quality, end_to_end = _measured_deltas()
    return AlternateEncodingQualificationReport.create(
        baseline_policy=_policy(AlternateMediaEncoding.PNG),
        alternate_policy=_policy(AlternateMediaEncoding.JPEG),
        selected_frames=SelectedFrameComparison(
            parity_status=parity,
            baseline=_inventory(),
            alternate=(
                _inventory()
                if parity != EncodingParityStatus.MISMATCH
                else _inventory("jpeg-selection")
            ),
        ),
        speed=speed,
        size=size,
        quality=quality,
        end_to_end=end_to_end,
        class_outcomes=(_outcome(outcome),),
        representative_parity=_representative(parity),
        parity_signoff=signoff,
        evidence_class=evidence_class,
    )


def test_unknown_external_labels_and_provider_remain_not_measured() -> None:
    evidence = RepresentativeParityEvidence.create()

    assert evidence.parity_status == EncodingParityStatus.NOT_MEASURED
    assert evidence.external_labels_status == MeasurementStatus.NOT_MEASURED
    assert evidence.provider_acceptance_status == MeasurementStatus.NOT_MEASURED
    assert evidence.provider_replay_status == MeasurementStatus.NOT_MEASURED


def test_representative_default_change_requires_a_bound_signoff() -> None:
    unsigned = _report()

    assert unsigned.default_promotion_eligible is False
    signoff = RepresentativeParitySignoff.create(
        comparison_digest=unsigned.comparison_digest,
        signoff_id="review-p5-001",
        signer="reviewer",
    )
    signed = _report(signoff=signoff)

    assert signed.default_promotion_eligible is True
    assert signed.production_eligible is False


def test_representative_match_rejects_any_per_class_outcome_mismatch() -> None:
    with pytest.raises(ValueError, match="selected-frame and per-class parity"):
        _report(outcome=EncodingParityStatus.MISMATCH)


def test_quality_cannot_be_measured_without_external_labels() -> None:
    speed, size, quality, end_to_end = _measured_deltas()

    with pytest.raises(ValueError, match="external labels"):
        AlternateEncodingQualificationReport.create(
            baseline_policy=_policy(AlternateMediaEncoding.PNG),
            alternate_policy=_policy(AlternateMediaEncoding.JPEG),
            selected_frames=SelectedFrameComparison(
                parity_status=EncodingParityStatus.NOT_MEASURED,
                baseline=_inventory(),
                alternate=_inventory(),
            ),
            speed=speed,
            size=size,
            quality=quality,
            end_to_end=end_to_end,
            class_outcomes=(_outcome(EncodingParityStatus.NOT_MEASURED),),
            representative_parity=RepresentativeParityEvidence.create(),
        )


def test_report_digest_and_derived_eligibility_reject_tampering() -> None:
    unsigned = _report()
    signoff = RepresentativeParitySignoff.create(
        comparison_digest=unsigned.comparison_digest,
        signoff_id="review-p5-002",
        signer="reviewer",
    )
    signed = _report(signoff=signoff)
    values = signed.model_dump(mode="python")
    values["default_promotion_eligible"] = False

    with pytest.raises(ValidationError, match="default_promotion_eligible"):
        AlternateEncodingQualificationReport.model_validate(values, strict=True)


def test_outcome_not_measured_cannot_smuggle_values() -> None:
    with pytest.raises(ValueError, match="NOT_MEASURED"):
        OutcomeParity(parity_status=EncodingParityStatus.NOT_MEASURED, baseline_count=1)


def test_selected_frame_inventory_rejects_unordered_or_incomplete_dimensions() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        SelectedFrameInventory(
            selection_semantic_sha256=_digest("unordered"),
            selected_frame_count=2,
            dimension_counts=(
                SelectedFrameDimension(width=320, height=240, frame_count=1),
                SelectedFrameDimension(width=320, height=180, frame_count=1),
            ),
        )

    with pytest.raises(ValueError, match="total selected_frame_count"):
        SelectedFrameInventory(
            selection_semantic_sha256=_digest("incomplete"),
            selected_frame_count=2,
            dimension_counts=(SelectedFrameDimension(width=320, height=180, frame_count=1),),
        )
