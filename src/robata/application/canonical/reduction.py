"""Deterministic provider-claim and all-part fusion reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import model_validator

from robata.application.canonical.models import (
    CanonicalOfflineConfigurationError,
    CanonicalOfflinePartResult,
    CanonicalOfflinePartStatus,
    NonEmptyString,
    NonNegativeInt,
)
from robata.application.canonical.projections import (
    _canonical_fusion_reduction_projection_values,
    _fusion_claim_reduction_digest,
    _stable_uuid,
    canonical_fusion_reduction_projection,
)
from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp
from robata.event_pipeline.identity_registry import PlatformEnrichedOutputReference
from robata.inference.call_barrier import InferenceCallPartCompletion, InferenceCallReduction
from robata.inference.enrichment import (
    EnrichedProviderClaim,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderTaskClaim,
)
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import InferenceStatus


class CanonicalFusionPartSource(StrictModel):
    part_ordinal: NonNegativeInt
    part_semantic_sha256: Sha256Digest
    completion_id: OpaqueUuid
    inference_id: OpaqueUuid
    selected_attempt_output_sha256: Sha256Digest
    enrichment: PlatformEnrichedOutputReference
    abstained: bool


class CanonicalFusionClaimSource(StrictModel):
    part_ordinal: NonNegativeInt
    source_claim_ordinal: NonNegativeInt
    source_claim_id: OpaqueUuid
    enrichment_logical_key: NodeLogicalKey


class CanonicalReducedFusionClaim(StrictModel):
    fusion_output_ordinal: NonNegativeInt
    claim_semantic_sha256: Sha256Digest
    representative: EnrichedProviderClaim
    sources: tuple[CanonicalFusionClaimSource, ...]

    @model_validator(mode="after")
    def validate_reduced_claim(self) -> Self:
        if self.representative.kind is not ProviderClaimKind.FUSION_HYPOTHESIS:
            raise ValueError("fusion reduction accepts only fusion hypotheses")
        expected_sources = tuple(
            sorted(
                self.sources,
                key=lambda item: (
                    item.part_ordinal,
                    item.source_claim_ordinal,
                    item.source_claim_id,
                ),
            )
        )
        if not self.sources or self.sources != expected_sources:
            raise ValueError("fusion claim sources must be nonempty and canonical")
        source_keys = tuple(
            (item.part_ordinal, item.source_claim_ordinal, item.source_claim_id)
            for item in self.sources
        )
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("fusion claim sources must be unique")
        if self.claim_semantic_sha256 != _fusion_claim_reduction_digest(self.representative):
            raise ValueError("fusion claim semantic digest is inconsistent")
        return self


class CanonicalFusionReduction(StrictModel):
    """Ephemeral reduced fusion view; it never impersonates a selected attempt."""

    schema_version: Literal["1.0"]
    reduction_id: OpaqueUuid
    reduction_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    input_plan_semantic_sha256: Sha256Digest
    barrier_reduction_id: OpaqueUuid
    barrier_reduction_semantic_sha256: Sha256Digest
    reduction_policy: NonEmptyString
    reduction_policy_version: SchemaVersion
    outcome: Literal["CLAIMS", "NO_SURVIVING_EVENTS", "ALL_PARTS_ABSTAINED"]
    parts: tuple[CanonicalFusionPartSource, ...]
    claims: tuple[CanonicalReducedFusionClaim, ...]
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_reduction(self) -> Self:
        if tuple(item.part_ordinal for item in self.parts) != tuple(range(len(self.parts))):
            raise ValueError("fusion reduction parts must be complete and ordered")
        enrichment_keys = tuple(item.enrichment.enrichment_logical_key for item in self.parts)
        if len(set(enrichment_keys)) != len(enrichment_keys):
            raise ValueError("fusion reduction enrichments must be unique")
        if tuple(item.fusion_output_ordinal for item in self.claims) != tuple(
            range(len(self.claims))
        ):
            raise ValueError("reduced fusion ordinals must be contiguous from zero")
        part_by_ordinal = {item.part_ordinal: item for item in self.parts}
        for claim in self.claims:
            for source in claim.sources:
                part = part_by_ordinal.get(source.part_ordinal)
                if (
                    part is None
                    or source.enrichment_logical_key != part.enrichment.enrichment_logical_key
                ):
                    raise ValueError("reduced claim source is outside the part manifest")
        if self.outcome == "ALL_PARTS_ABSTAINED":
            if self.claims or not self.parts or not all(item.abstained for item in self.parts):
                raise ValueError("all-parts abstention has an inconsistent reduction shape")
        elif self.outcome == "NO_SURVIVING_EVENTS":
            if self.claims or not self.parts or any(item.abstained for item in self.parts):
                raise ValueError("empty fusion reduction has an inconsistent shape")
        elif not self.claims or any(item.abstained for item in self.parts):
            raise ValueError("claim reduction requires every required part to complete")
        expected = semantic_sha256(canonical_fusion_reduction_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("fusion reduction semantic digest is inconsistent")
        if self.reduction_logical_key != f"fusion-reduction:{expected}":
            raise ValueError("fusion reduction logical key is inconsistent")
        if self.reduction_id != _stable_uuid("canonical-fusion-reduction", expected):
            raise ValueError("fusion reduction ID is inconsistent")
        return self


class _OrderedProviderClaimReducer:
    """Exact-duplicate reducer for ordered required call-part payloads."""

    def reduce(
        self,
        *,
        input_plan: InferenceInputPlan,
        ordered_completions: tuple[InferenceCallPartCompletion, ...],
    ) -> Mapping[str, object]:
        if len(ordered_completions) != len(input_plan.call_plan.parts):
            raise CanonicalOfflineConfigurationError(
                "provider reducer requires the complete call-part set"
            )
        payloads: list[ProviderClaimPayload] = []
        for part, completion in zip(input_plan.call_plan.parts, ordered_completions, strict=True):
            if (
                completion.part_ordinal != part.ordinal
                or completion.part_semantic_sha256 != part.part_semantic_sha256
                or completion.status is not InferenceStatus.SUCCEEDED
                or completion.normalized_output is None
            ):
                raise CanonicalOfflineConfigurationError(
                    "provider reducer received an invalid ordered completion"
                )
            payloads.append(
                ProviderClaimPayload.model_validate(
                    completion.normalized_output,
                    strict=False,
                )
            )
        return _reduce_provider_claim_payloads(tuple(payloads)).model_dump(mode="json")


def _reduce_provider_claim_payloads(
    payloads: Sequence[ProviderClaimPayload],
) -> ProviderClaimPayload:
    """Reduce ordered part payloads using exact equality except local ordinals."""

    ordered = tuple(payloads)
    if not ordered:
        raise CanonicalOfflineConfigurationError(
            "provider claim reduction requires at least one part payload"
        )
    abstention_flags = tuple(item.abstained for item in ordered)
    if all(abstention_flags):
        return ProviderClaimPayload(claims=(), abstained=True)
    if any(abstention_flags):
        raise CanonicalOfflineConfigurationError(
            "required part payloads cannot mix claims and abstentions"
        )

    reduced: list[ProviderTaskClaim] = []
    seen_across_parts: set[str] = set()
    for payload in ordered:
        seen_in_part: set[str] = set()
        for claim in payload.claims:
            digest = semantic_sha256(claim.model_dump(mode="json", exclude={"claim_ordinal"}))
            if digest in seen_in_part:
                raise CanonicalOfflineConfigurationError(
                    "one call part contains duplicate provider claims"
                )
            seen_in_part.add(digest)
            if digest in seen_across_parts:
                continue
            seen_across_parts.add(digest)
            reduced.append(claim.model_copy(update={"claim_ordinal": len(reduced)}))
    return ProviderClaimPayload(claims=tuple(reduced), abstained=False)


def _build_canonical_fusion_reduction(
    *,
    input_plan: InferenceInputPlan,
    barrier_reduction: InferenceCallReduction,
    part_results: Sequence[CanonicalOfflinePartResult],
    created_at: str,
) -> CanonicalFusionReduction:
    ordered_results = tuple(part_results)
    planned_parts = input_plan.call_plan.parts
    if (
        len(ordered_results) != len(planned_parts)
        or tuple(item.part_ordinal for item in ordered_results) != tuple(range(len(planned_parts)))
        or barrier_reduction.input_plan_semantic_sha256 != input_plan.semantic_sha256
        or barrier_reduction.ordered_completion_ids
        != tuple(item.completion.completion_id for item in ordered_results)
        or barrier_reduction.reduction_policy != input_plan.call_plan.reduction_policy
        or barrier_reduction.reduction_policy_version
        != input_plan.call_plan.reduction_policy_version
    ):
        raise CanonicalOfflineConfigurationError(
            "fusion reduction inputs do not match the complete ordered call plan"
        )

    part_sources: list[CanonicalFusionPartSource] = []
    grouped_claims: dict[
        str,
        list[tuple[CanonicalOfflinePartResult, EnrichedProviderClaim]],
    ] = {}
    for planned, result in zip(planned_parts, ordered_results, strict=True):
        enriched = result.enriched_output
        selected = result.selected_output
        if (
            result.status is not CanonicalOfflinePartStatus.ENRICHED
            or result.part_semantic_sha256 != planned.part_semantic_sha256
            or enriched is None
            or selected is None
        ):
            raise CanonicalOfflineConfigurationError(
                "fusion reduction requires exact enriched lineage for every part"
            )
        enrichment_ref = PlatformEnrichedOutputReference.from_output(enriched)
        part_sources.append(
            CanonicalFusionPartSource(
                part_ordinal=result.part_ordinal,
                part_semantic_sha256=result.part_semantic_sha256,
                completion_id=result.completion.completion_id,
                inference_id=result.terminal.inference_id,
                selected_attempt_output_sha256=selected.output_sha256,
                enrichment=enrichment_ref,
                abstained=enriched.abstained,
            )
        )
        seen_in_part: set[str] = set()
        for claim in enriched.claims:
            digest = _fusion_claim_reduction_digest(claim)
            if digest in seen_in_part:
                raise CanonicalOfflineConfigurationError(
                    "one enriched part contains duplicate fusion claim semantics"
                )
            seen_in_part.add(digest)
            grouped_claims.setdefault(digest, []).append((result, claim))

    abstention_flags = tuple(item.abstained for item in part_sources)
    if any(abstention_flags) and not all(abstention_flags):
        raise CanonicalOfflineConfigurationError(
            "fusion reduction cannot mix required abstentions and claims"
        )

    reduced_claims: list[CanonicalReducedFusionClaim] = []
    ordered_groups = sorted(
        grouped_claims.items(),
        key=lambda item: (
            -1 if item[1][0][1].interval is None else item[1][0][1].interval.start_ns,
            -1 if item[1][0][1].interval is None else item[1][0][1].interval.end_ns,
            item[0],
        ),
    )
    for digest, group in ordered_groups:
        ordered_group = tuple(
            sorted(
                group,
                key=lambda item: (
                    item[0].part_ordinal,
                    item[1].claim_ordinal,
                    item[1].claim_id,
                ),
            )
        )
        sources = tuple(
            CanonicalFusionClaimSource(
                part_ordinal=result.part_ordinal,
                source_claim_ordinal=claim.claim_ordinal,
                source_claim_id=claim.claim_id,
                enrichment_logical_key=result.enriched_output.enrichment_logical_key,
            )
            for result, claim in ordered_group
            if result.enriched_output is not None
        )
        reduced_claims.append(
            CanonicalReducedFusionClaim(
                fusion_output_ordinal=len(reduced_claims),
                claim_semantic_sha256=digest,
                representative=ordered_group[0][1],
                sources=sources,
            )
        )

    if all(abstention_flags):
        outcome: Literal["CLAIMS", "NO_SURVIVING_EVENTS", "ALL_PARTS_ABSTAINED"] = (
            "ALL_PARTS_ABSTAINED"
        )
    elif reduced_claims:
        outcome = "CLAIMS"
    else:
        outcome = "NO_SURVIVING_EVENTS"
    projection = _canonical_fusion_reduction_projection_values(
        schema_version="1.0",
        input_plan_semantic_sha256=input_plan.semantic_sha256,
        barrier_reduction_semantic_sha256=barrier_reduction.reduction_semantic_sha256,
        reduction_policy=input_plan.call_plan.reduction_policy,
        reduction_policy_version=input_plan.call_plan.reduction_policy_version,
        outcome=outcome,
        parts=part_sources,
        claims=reduced_claims,
    )
    digest = semantic_sha256(projection)
    return CanonicalFusionReduction(
        schema_version="1.0",
        reduction_id=_stable_uuid("canonical-fusion-reduction", digest),
        reduction_logical_key=f"fusion-reduction:{digest}",
        semantic_sha256=digest,
        input_plan_semantic_sha256=input_plan.semantic_sha256,
        barrier_reduction_id=barrier_reduction.reduction_id,
        barrier_reduction_semantic_sha256=(barrier_reduction.reduction_semantic_sha256),
        reduction_policy=input_plan.call_plan.reduction_policy,
        reduction_policy_version=input_plan.call_plan.reduction_policy_version,
        outcome=outcome,
        parts=tuple(part_sources),
        claims=tuple(reduced_claims),
        created_at=created_at,
    )


__all__ = [
    "CanonicalFusionClaimSource",
    "CanonicalFusionPartSource",
    "CanonicalFusionReduction",
    "CanonicalReducedFusionClaim",
]
