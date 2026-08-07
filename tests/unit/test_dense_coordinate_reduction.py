from __future__ import annotations

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

import pytest

from robata.application.canonical.runner import _reduce_dense_coordinate_result
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.pipeline import CameraQAStatus
from robata.inference.enrichment import (
    EnrichedProviderClaim,
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.qa_pipeline.dense import CameraDenseResult, DenseQAOutputRef


def _id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:test:dense-coordinate-reduction:{value}"))


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _result(
    *,
    marker: str,
    observation: ProviderObservation,
    start_ns: int,
) -> CameraDenseResult:
    package_id = _id("package:0")
    output = DenseQAOutputRef(
        artifact_id=_id(f"output:{marker}"),
        semantic_sha256=_digest(f"output:{marker}"),
        enrichment_logical_key=f"dense-output:{_digest(marker)}",
        inference_id=_id(f"inference:{marker}"),
        input_plan_id=_id("input-plan"),
        input_plan_semantic_sha256=_digest("input-plan"),
    )
    claim = EnrichedProviderClaim(
        claim_id=_id(f"claim:{marker}"),
        claim_ordinal=0,
        kind=ProviderClaimKind.QA_OBSERVATION,
        package_id=package_id,
        package_ordinal=0,
        camera_id=CAMERA_IDS[0],
        interval=ProviderClaimInterval(start_ns=start_ns, end_ns=start_ns + 10),
        label="dense-test",
        observation=observation,
        evidence=(),
        model_reported_confidence=None,
        conflict_codes=(),
    )
    return CameraDenseResult(
        package_id=package_id,
        package_ordinal=0,
        camera_id=CAMERA_IDS[0],
        local_status=CameraQAStatus(observation.value),
        source_output=output,
        claim=claim,
    )


@pytest.mark.parametrize(
    ("current_observation", "candidate_observation"),
    (
        (ProviderObservation.GOOD, ProviderObservation.DEGRADED),
        (ProviderObservation.DEGRADED, ProviderObservation.UNKNOWN),
        (ProviderObservation.UNKNOWN, ProviderObservation.UNUSABLE),
    ),
)
def test_duplicate_dense_coordinate_prefers_more_conservative_status(
    current_observation: ProviderObservation,
    candidate_observation: ProviderObservation,
) -> None:
    current = _result(marker="current", observation=current_observation, start_ns=200)
    candidate = _result(marker="candidate", observation=candidate_observation, start_ns=100)

    assert _reduce_dense_coordinate_result(current, candidate) is candidate
    assert _reduce_dense_coordinate_result(candidate, current) is candidate


def test_duplicate_dense_coordinate_prefers_earliest_interval_on_status_tie() -> None:
    late = _result(marker="late", observation=ProviderObservation.DEGRADED, start_ns=200)
    early = _result(marker="early", observation=ProviderObservation.DEGRADED, start_ns=100)

    assert _reduce_dense_coordinate_result(late, early) is early
    assert _reduce_dense_coordinate_result(early, late) is early
