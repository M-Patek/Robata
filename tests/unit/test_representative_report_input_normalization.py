from __future__ import annotations

import pytest

from robata.benchmark.qualification import RepresentativeProductionQualificationReport
from tests.unit.test_representative_production_qualification import _report_inputs


def test_production_report_create_normalizes_nested_mapping_inputs() -> None:
    typed = _report_inputs()
    raw = {
        **typed,
        "scope": typed["scope"].model_dump(mode="python"),
        "service_capacity": typed["service_capacity"].model_dump(mode="python"),
        "provider_saturation": typed["provider_saturation"],
        "quality": typed["quality"].model_dump(mode="python"),
        "recovery_evidence": tuple(
            item.model_dump(mode="python") for item in typed["recovery_evidence"]
        ),
    }

    typed_report = RepresentativeProductionQualificationReport.create(**typed)
    raw_report = RepresentativeProductionQualificationReport.create(**raw)

    assert raw_report == typed_report


@pytest.mark.parametrize('mode', ('python', 'json'))
def test_production_report_create_normalizes_provider_report_mapping(mode: str) -> None:
    typed = _report_inputs()
    raw = {
        **typed,
        'provider_saturation': typed['provider_saturation'].model_dump(mode=mode),
    }

    assert RepresentativeProductionQualificationReport.create(**raw) == (
        RepresentativeProductionQualificationReport.create(**typed)
    )


def test_production_report_create_rejects_boolean_numeric_mapping_values() -> None:
    typed = _report_inputs()
    raw_service = typed["service_capacity"].model_dump(mode="python")
    raw_service["backlog_start_count"] = True

    with pytest.raises(ValueError):
        RepresentativeProductionQualificationReport.create(
            **{**typed, "service_capacity": raw_service}
        )

    raw_quality = typed["quality"].model_dump(mode="python")
    raw_quality["decision"]["approved_gates"][0]["actual_value"] = True

    with pytest.raises(ValueError):
        RepresentativeProductionQualificationReport.create(
            **{**typed, "quality": raw_quality}
        )


def test_production_report_create_rejects_provider_boolean_numeric_mapping_value() -> None:
    typed = _report_inputs()
    raw_provider = typed["provider_saturation"].model_dump(mode="json")
    raw_provider["points"][0]["telemetry"]["adapter_transport_retry_count"] = True

    with pytest.raises(ValueError, match="cannot coerce a boolean"):
        RepresentativeProductionQualificationReport.create(
            **{**typed, "provider_saturation": raw_provider}
        )
