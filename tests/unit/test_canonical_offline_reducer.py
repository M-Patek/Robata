from __future__ import annotations

from collections.abc import Callable

import pytest

from robata.application.canonical_offline import (
    CanonicalOfflineConfigurationError,
    _fusion_claim_reduction_digest,
    _reduce_provider_claim_payloads,
)
from robata.inference.enrichment import (
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderObservation,
    ProviderTaskClaim,
)
from tests.unit.test_inference_enrichment import _enrich, _fixture


def _token(value: int) -> str:
    return f"ref:{value:064x}"


def _claim(
    *,
    label: str = "pick",
    evidence_tokens: tuple[str, ...] = (_token(1),),
    model_reported_score: float | None = 0.5,
    conflict_codes: tuple[str, ...] = ("camera-disagreement",),
) -> ProviderTaskClaim:
    return ProviderTaskClaim(
        claim_ordinal=0,
        kind=ProviderClaimKind.FUSION_HYPOTHESIS,
        package_ordinal=None,
        camera_ordinal=None,
        interval=ProviderClaimInterval(start_ns=100, end_ns=200),
        label=label,
        observation=ProviderObservation.PROPOSED,
        evidence_tokens=evidence_tokens,
        model_reported_score=model_reported_score,
        conflict_codes=conflict_codes,
    )


def _payload(*claims: ProviderTaskClaim) -> ProviderClaimPayload:
    return ProviderClaimPayload(
        claims=tuple(
            claim.model_copy(update={"claim_ordinal": ordinal})
            for ordinal, claim in enumerate(claims)
        ),
        abstained=False,
    )


def test_exact_cross_part_duplicate_ignores_only_local_ordinal() -> None:
    lead = _claim(label="lead")
    duplicate = _claim()

    reduced = _reduce_provider_claim_payloads(
        (
            _payload(lead, duplicate),
            _payload(duplicate),
        )
    )

    assert reduced.abstained is False
    assert tuple(claim.claim_ordinal for claim in reduced.claims) == (0, 1)
    assert tuple(claim.label for claim in reduced.claims) == ("lead", "pick")
    assert reduced.claims[1].model_dump(
        mode="json", exclude={"claim_ordinal"}
    ) == duplicate.model_dump(mode="json", exclude={"claim_ordinal"})


@pytest.mark.parametrize(
    "vary_claim",
    [
        pytest.param(
            lambda claim: claim.model_copy(update={"model_reported_score": 0.75}),
            id="score",
        ),
        pytest.param(
            lambda claim: claim.model_copy(update={"conflict_codes": ("timing-disagreement",)}),
            id="conflict",
        ),
        pytest.param(
            lambda claim: claim.model_copy(update={"evidence_tokens": (_token(2),)}),
            id="evidence",
        ),
    ],
)
def test_non_ordinal_claim_differences_survive_exact_reduction(
    vary_claim: Callable[[ProviderTaskClaim], ProviderTaskClaim],
) -> None:
    original = _claim()
    variation = vary_claim(original)

    reduced = _reduce_provider_claim_payloads(
        (
            _payload(original),
            _payload(variation),
        )
    )

    assert reduced.abstained is False
    assert tuple(claim.claim_ordinal for claim in reduced.claims) == (0, 1)
    assert reduced.claims[0].model_dump(mode="json", exclude={"claim_ordinal"}) != reduced.claims[
        1
    ].model_dump(mode="json", exclude={"claim_ordinal"})


def test_duplicate_claims_within_one_part_fail_closed() -> None:
    duplicate = _claim()

    with pytest.raises(
        CanonicalOfflineConfigurationError,
        match="one call part contains duplicate provider claims",
    ):
        _reduce_provider_claim_payloads((_payload(duplicate, duplicate),))


def test_enriched_claim_reduction_identity_excludes_row_ids_and_storage_locator() -> None:
    claim = _enrich(_fixture()).claims[0]
    evidence = claim.evidence[0]
    relocated_evidence = evidence.model_copy(
        update={
            "package_id": "00000000-0000-0000-0000-000000000101",
            "frame_id": "00000000-0000-0000-0000-000000000102",
            "source_artifact_uri": "object://relocated/same-content",
        }
    )
    relocated = claim.model_copy(
        update={
            "package_id": "00000000-0000-0000-0000-000000000103",
            "evidence": (relocated_evidence,),
        }
    )

    assert _fusion_claim_reduction_digest(relocated) == _fusion_claim_reduction_digest(claim)


def test_enriched_claim_reduction_identity_includes_evidence_content_digest() -> None:
    claim = _enrich(_fixture()).claims[0]
    evidence = claim.evidence[0]
    changed = claim.model_copy(
        update={"evidence": (evidence.model_copy(update={"source_artifact_sha256": "f" * 64}),)}
    )

    assert _fusion_claim_reduction_digest(changed) != _fusion_claim_reduction_digest(claim)


def test_all_required_parts_may_abstain() -> None:
    abstained = ProviderClaimPayload(claims=(), abstained=True)

    reduced = _reduce_provider_claim_payloads((abstained, abstained))

    assert reduced == ProviderClaimPayload(claims=(), abstained=True)


@pytest.mark.parametrize("abstained_first", [True, False])
def test_claims_and_abstention_cannot_be_mixed(abstained_first: bool) -> None:
    abstained = ProviderClaimPayload(claims=(), abstained=True)
    claimed = _payload(_claim())
    payloads = (abstained, claimed) if abstained_first else (claimed, abstained)

    with pytest.raises(
        CanonicalOfflineConfigurationError,
        match="required part payloads cannot mix claims and abstentions",
    ):
        _reduce_provider_claim_payloads(payloads)
