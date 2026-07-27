from __future__ import annotations

from dataclasses import replace

import pytest

from robata.benchmark.provider_qualification import ProviderSaturationPoint
from tests.unit.test_provider_qualification import (
    _capacity,
    _configuration,
    _session,
    _telemetry,
)


def test_provider_saturation_rejects_capacity_count_that_disagrees_with_telemetry() -> None:
    configuration = _configuration()
    session = _session(configuration, 901)
    mismatched_capacity = replace(_capacity(), provider_images=1)

    with pytest.raises(ValueError, match="capacity provider_images total"):
        ProviderSaturationPoint(
            configuration_digest=configuration.configuration_digest,
            qualification_session=session,
            run_namespace=session.run_namespace,
            offered_concurrency=1,
            capacity=mismatched_capacity,
            telemetry=_telemetry(session),
        )
