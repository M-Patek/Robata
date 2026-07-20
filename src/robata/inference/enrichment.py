"""Fail-closed provider-claim and orchestrator-enrichment boundary.

Provider payloads contain only local claims and opaque correlation tokens.  The
enricher resolves those tokens against an immutable input-plan catalog and is
the only component in this slice allowed to add persisted identity or lineage.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import ArtifactUri
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import (
    NodeLogicalKey,
    NodeType,
    OpaqueUuid,
    Rfc3339Timestamp,
)
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.inference.adapter import JsonSchemaRef
from robata.inference.input_plan import (
    InferenceCallPart,
    InferenceInputPlan,
    RenderedProviderItem,
)
from robata.inference.models import (
    InferenceAttemptSelection,
    VisionTask,
    inference_attempt_selection_logical_key,
)

PROVIDER_CLAIM_SCHEMA_ID = "https://schemas.robata.dev/provider-claim-payload"
ENRICHED_OUTPUT_SCHEMA_ID = "https://schemas.robata.dev/orchestrator-enriched-output"
ENRICHED_OUTPUT_SCHEMA_VERSION = "2.0.0"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
ClaimLabel = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
CameraOrdinal = Annotated[int, Field(strict=True, ge=0, le=5)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
ProviderCorrelationToken = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^ref:[0-9a-f]{64}$"),
]
type FrozenTuple[T] = Annotated[tuple[T, ...], Field(strict=False)]


class ProviderClaimKind(StrEnum):
    """Closed set of provider-authored claim shapes."""

    ACTION_OBSERVATION = "ACTION_OBSERVATION"
    BOUNDARY_OBSERVATION = "BOUNDARY_OBSERVATION"
    CROSS_VIEW_HYPOTHESIS = "CROSS_VIEW_HYPOTHESIS"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    FUSION_HYPOTHESIS = "FUSION_HYPOTHESIS"
    QA_OBSERVATION = "QA_OBSERVATION"


class ProviderObservation(StrEnum):
    """Task observations that are labels, never system decisions."""

    CONFLICT = "CONFLICT"
    DEGRADED = "DEGRADED"
    GOOD = "GOOD"
    MISSING = "MISSING"
    NO_BOUNDARY = "NO_BOUNDARY"
    NO_EVENT = "NO_EVENT"
    OBSERVED = "OBSERVED"
    OCCLUDED = "OCCLUDED"
    PARTIAL = "PARTIAL"
    PROPOSED = "PROPOSED"
    SUPPORTING = "SUPPORTING"
    UNKNOWN = "UNKNOWN"
    UNUSABLE = "UNUSABLE"


_CAMERA_CLAIM_KINDS = frozenset(
    {
        ProviderClaimKind.ACTION_OBSERVATION,
        ProviderClaimKind.BOUNDARY_OBSERVATION,
        ProviderClaimKind.QA_OBSERVATION,
    }
)
_ALLOWED_OBSERVATIONS = {
    ProviderClaimKind.QA_OBSERVATION: frozenset(
        {
            ProviderObservation.GOOD,
            ProviderObservation.DEGRADED,
            ProviderObservation.UNUSABLE,
            ProviderObservation.UNKNOWN,
        }
    ),
    ProviderClaimKind.EVENT_PROPOSAL: frozenset({ProviderObservation.PROPOSED}),
    ProviderClaimKind.ACTION_OBSERVATION: frozenset(
        {
            ProviderObservation.SUPPORTING,
            ProviderObservation.PARTIAL,
            ProviderObservation.NO_EVENT,
            ProviderObservation.OCCLUDED,
            ProviderObservation.UNUSABLE,
            ProviderObservation.MISSING,
        }
    ),
    ProviderClaimKind.CROSS_VIEW_HYPOTHESIS: frozenset(
        {ProviderObservation.SUPPORTING, ProviderObservation.PARTIAL}
    ),
    ProviderClaimKind.BOUNDARY_OBSERVATION: frozenset(
        {
            ProviderObservation.OBSERVED,
            ProviderObservation.NO_BOUNDARY,
            ProviderObservation.OCCLUDED,
            ProviderObservation.UNUSABLE,
            ProviderObservation.MISSING,
        }
    ),
    ProviderClaimKind.FUSION_HYPOTHESIS: frozenset(
        {ProviderObservation.PROPOSED, ProviderObservation.CONFLICT}
    ),
}
_TASK_KINDS = {
    VisionTask.QA_COARSE: frozenset({ProviderClaimKind.QA_OBSERVATION}),
    VisionTask.QA_DENSE: frozenset({ProviderClaimKind.QA_OBSERVATION}),
    VisionTask.EVENT_PROPOSAL: frozenset({ProviderClaimKind.EVENT_PROPOSAL}),
    VisionTask.ACTION_EVIDENCE: frozenset(
        {
            ProviderClaimKind.ACTION_OBSERVATION,
            ProviderClaimKind.CROSS_VIEW_HYPOTHESIS,
        }
    ),
    VisionTask.BOUNDARY_REFINEMENT: frozenset({ProviderClaimKind.BOUNDARY_OBSERVATION}),
    VisionTask.FUSION_ADJUDICATION: frozenset({ProviderClaimKind.FUSION_HYPOTHESIS}),
}
_EVIDENCE_REQUIRED = frozenset(
    {
        ProviderObservation.CONFLICT,
        ProviderObservation.DEGRADED,
        ProviderObservation.GOOD,
        ProviderObservation.NO_EVENT,
        ProviderObservation.OBSERVED,
        ProviderObservation.PARTIAL,
        ProviderObservation.PROPOSED,
        ProviderObservation.SUPPORTING,
    }
)


class ProviderClaimEnrichmentError(ValueError):
    """Provider output cannot be enriched without inventing authority."""


class ProviderClaimInterval(StrictModel):
    """One provider-reported half-open interval."""

    start_ns: Nanoseconds
    end_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if self.start_ns >= self.end_ns:
            raise ValueError("provider claim interval must be nonempty")
        return self


class ProviderTaskClaim(StrictModel):
    """A provider-facing claim with no persisted IDs or trusted confidence."""

    claim_ordinal: NonNegativeInt
    kind: ProviderClaimKind
    package_ordinal: NonNegativeInt | None
    camera_ordinal: CameraOrdinal | None
    interval: ProviderClaimInterval | None
    label: ClaimLabel | None
    observation: ProviderObservation
    evidence_tokens: FrozenTuple[ProviderCorrelationToken]
    model_reported_score: UnitInterval | None
    conflict_codes: FrozenTuple[ClaimLabel]

    @model_validator(mode="after")
    def validate_claim_shape(self) -> Self:
        if self.observation not in _ALLOWED_OBSERVATIONS[self.kind]:
            raise ValueError("observation is not valid for the provider claim kind")
        if len(set(self.evidence_tokens)) != len(self.evidence_tokens):
            raise ValueError("provider evidence tokens must be unique within a claim")
        if len(set(self.conflict_codes)) != len(self.conflict_codes):
            raise ValueError("provider conflict codes must be unique within a claim")

        if self.kind in _CAMERA_CLAIM_KINDS:
            if self.package_ordinal is None or self.camera_ordinal is None:
                raise ValueError("camera claims require local package and camera ordinals")
        elif self.package_ordinal is not None or self.camera_ordinal is not None:
            raise ValueError("cross-view claims cannot author package or camera identity")

        if (
            self.kind
            in {
                ProviderClaimKind.EVENT_PROPOSAL,
                ProviderClaimKind.CROSS_VIEW_HYPOTHESIS,
                ProviderClaimKind.FUSION_HYPOTHESIS,
            }
            and self.interval is None
        ):
            raise ValueError("temporal hypotheses require a provider-reported interval")
        if self.kind is ProviderClaimKind.QA_OBSERVATION and self.interval is None:
            raise ValueError("QA observations require an observed interval")
        if self.observation is ProviderObservation.MISSING and (
            self.interval is not None or self.evidence_tokens
        ):
            raise ValueError("MISSING claims cannot assert intervals or evidence")
        if self.observation in _EVIDENCE_REQUIRED and not self.evidence_tokens:
            raise ValueError("observing claims require at least one evidence token")
        if self.conflict_codes and self.kind is not ProviderClaimKind.FUSION_HYPOTHESIS:
            raise ValueError("conflict codes are valid only for fusion hypotheses")
        return self


class ProviderClaimPayload(StrictModel):
    """Exact provider-facing document validated by its registered schema."""

    claims: FrozenTuple[ProviderTaskClaim]
    abstained: bool

    @model_validator(mode="after")
    def validate_ordinals(self) -> Self:
        ordinals = tuple(claim.claim_ordinal for claim in self.claims)
        if ordinals != tuple(range(len(self.claims))):
            raise ValueError("provider claim ordinals must be contiguous from zero")
        if self.abstained and self.claims:
            raise ValueError("an abstained provider response cannot contain claims")
        return self


class ProviderReferenceCatalogEntry(StrictModel):
    """One opaque token and its model-local catalog coordinates."""

    correlation_token: ProviderCorrelationToken
    provider_item_ordinal: NonNegativeInt
    package_ordinal: NonNegativeInt
    camera_ordinal: CameraOrdinal
    frame_ordinal: NonNegativeInt


class ProviderReferenceCatalog(StrictModel):
    """Immutable join catalog derived from one exact inference input plan."""

    schema_version: Literal["1.0"]
    reference_catalog_id: OpaqueUuid
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    request_catalog_id: OpaqueUuid
    request_catalog_sha256: Sha256Digest
    task: VisionTask
    token_policy_version: SchemaVersion
    entries: FrozenTuple[ProviderReferenceCatalogEntry]
    semantic_sha256: Sha256Digest
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        ordinals = tuple(entry.provider_item_ordinal for entry in self.entries)
        if ordinals != tuple(range(len(self.entries))):
            raise ValueError("provider catalog item ordinals must be contiguous from zero")
        tokens = tuple(entry.correlation_token for entry in self.entries)
        if len(set(tokens)) != len(tokens):
            raise ValueError("provider reference catalog tokens must be unique")
        for entry in self.entries:
            expected = _correlation_token(
                request_catalog_sha256=self.request_catalog_sha256,
                token_policy_version=self.token_policy_version,
                provider_item_ordinal=entry.provider_item_ordinal,
                package_ordinal=entry.package_ordinal,
                camera_ordinal=entry.camera_ordinal,
                frame_ordinal=entry.frame_ordinal,
            )
            if entry.correlation_token != expected:
                raise ValueError("provider correlation token is inconsistent")
        expected_digest = semantic_sha256(provider_reference_catalog_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("provider reference catalog semantic_sha256 is inconsistent")
        return self

    @classmethod
    def build(
        cls,
        *,
        input_plan: InferenceInputPlan,
        reference_catalog_id: str,
        token_policy_version: str,
        created_at: str,
    ) -> ProviderReferenceCatalog:
        """Build tokens from the exact provider rendering and request catalog."""

        entries = _expected_reference_entries(input_plan, token_policy_version)
        projection = {
            "input_plan_semantic_sha256": input_plan.semantic_sha256,
            "request_catalog_sha256": input_plan.request_catalog.semantic_sha256,
            "task": input_plan.subject.task.value,
            "token_policy_version": token_policy_version,
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }
        return cls(
            schema_version="1.0",
            reference_catalog_id=reference_catalog_id,
            input_plan_id=input_plan.input_plan_id,
            input_plan_semantic_sha256=input_plan.semantic_sha256,
            request_catalog_id=input_plan.request_catalog.request_catalog_id,
            request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
            task=input_plan.subject.task,
            token_policy_version=token_policy_version,
            entries=entries,
            semantic_sha256=semantic_sha256(projection),
            created_at=created_at,
        )

    @staticmethod
    def derive_entries(
        *,
        request_catalog_sha256: str,
        rendered_items: Sequence[RenderedProviderItem],
        token_policy_version: str,
    ) -> tuple[ProviderReferenceCatalogEntry, ...]:
        """Derive prompt-safe tokens before the final input plan exists."""

        return _reference_entries_for_rendering(
            request_catalog_sha256=request_catalog_sha256,
            rendered_items=rendered_items,
            token_policy_version=token_policy_version,
        )

    def entries_for_part(
        self,
        *,
        input_plan: InferenceInputPlan,
        part_ordinal: int,
    ) -> tuple[ProviderReferenceCatalogEntry, ...]:
        """Return the exact token allowlist visible to one declared call part."""

        part = _enrichment_part(input_plan, part_ordinal)
        if (
            self.input_plan_id != input_plan.input_plan_id
            or self.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or self.entries != _expected_reference_entries(input_plan, self.token_policy_version)
        ):
            raise ProviderClaimEnrichmentError(
                "provider reference catalog does not match input plan"
            )
        return tuple(
            entry
            for entry in self.entries
            if part.start_item_ordinal
            <= entry.provider_item_ordinal
            < part.end_item_ordinal_exclusive
        )


class RawProviderResponseArtifact(StrictModel):
    """Immutable reference to exact untrusted provider response bytes."""

    schema_version: Literal["1.0"]
    artifact_id: OpaqueUuid
    exact_bytes_sha256: Sha256Digest
    byte_count: PositiveInt
    media_type: NonEmptyString
    provider_request_id: NonEmptyString
    inference_id: OpaqueUuid
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @classmethod
    def from_bytes(
        cls,
        *,
        data: bytes,
        artifact_id: str,
        media_type: str,
        provider_request_id: str,
        inference_id: str,
        provider: str,
        model_name: str,
        model_version: str,
        created_at: str,
    ) -> RawProviderResponseArtifact:
        if not data:
            raise ValueError("raw provider response bytes must be nonempty")
        return cls(
            schema_version="1.0",
            artifact_id=artifact_id,
            exact_bytes_sha256=exact_bytes_sha256(data),
            byte_count=len(data),
            media_type=media_type,
            provider_request_id=provider_request_id,
            inference_id=inference_id,
            provider=provider,
            model_name=model_name,
            model_version=model_version,
            created_at=created_at,
        )


class ParsedProviderClaimArtifact(StrictModel):
    """Parsed claims remain a separate immutable untrusted artifact."""

    schema_version: Literal["1.0"]
    artifact_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    raw_response: RawProviderResponseArtifact
    provider_claim_schema: JsonSchemaRef
    task: VisionTask
    payload: ProviderClaimPayload
    parser_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if self.provider_claim_schema.schema_id != PROVIDER_CLAIM_SCHEMA_ID:
            raise ValueError("parsed provider claims require the provider-claim schema")
        allowed = _TASK_KINDS[self.task]
        if any(claim.kind not in allowed for claim in self.payload.claims):
            raise ValueError("provider claim kind does not match the selected task")
        if self.task is VisionTask.FUSION_ADJUDICATION:
            if self.payload.abstained == bool(self.payload.claims):
                raise ValueError("fusion output must either abstain or contain hypotheses")
        elif self.payload.abstained:
            raise ValueError("only fusion adjudication may emit abstained=true")
        expected = semantic_sha256(parsed_provider_claim_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("parsed provider claim semantic_sha256 is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        raw_response: RawProviderResponseArtifact,
        provider_claim_schema: JsonSchemaRef,
        task: VisionTask,
        payload: ProviderClaimPayload,
        parser_version: str,
        created_at: str,
    ) -> ParsedProviderClaimArtifact:
        projection = {
            "raw_provider_bytes_sha256": raw_response.exact_bytes_sha256,
            "provider_claim_schema": _json_schema_logical_projection(provider_claim_schema),
            "task": task.value,
            "payload": payload.model_dump(mode="json"),
            "parser_version": parser_version,
        }
        return cls(
            schema_version="1.0",
            artifact_id=artifact_id,
            semantic_sha256=semantic_sha256(projection),
            raw_response=raw_response,
            provider_claim_schema=provider_claim_schema,
            task=task,
            payload=payload,
            parser_version=parser_version,
            created_at=created_at,
        )


class SelectedAttemptOutput(StrictModel):
    """Selected attempt evidence joining raw bytes and parsed claims."""

    inference_id: OpaqueUuid
    selection_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    selection_decision_logical_key: NodeLogicalKey
    selection_policy_version: SchemaVersion
    raw_response_artifact_id: OpaqueUuid
    raw_response_sha256: Sha256Digest
    parsed_claim_artifact_id: OpaqueUuid
    parsed_claim_sha256: Sha256Digest
    output_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        expected_selection_key = inference_attempt_selection_logical_key(
            logical_invocation_id=self.logical_invocation_id,
            policy_version=self.selection_policy_version,
        )
        if self.selection_decision_logical_key != expected_selection_key:
            raise ValueError("selected attempt selection logical key is inconsistent")
        expected = _selected_attempt_output_digest(
            raw_response_sha256=self.raw_response_sha256,
            parsed_claim_sha256=self.parsed_claim_sha256,
            selection_decision_logical_key=self.selection_decision_logical_key,
        )
        if self.output_sha256 != expected:
            raise ValueError("selected attempt output_sha256 is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        parsed_claims: ParsedProviderClaimArtifact,
        selection: InferenceAttemptSelection,
    ) -> SelectedAttemptOutput:
        raw = parsed_claims.raw_response
        if selection.inference_id != raw.inference_id:
            raise ValueError("selection does not reference the parsed inference attempt")
        output_digest = _selected_attempt_output_digest(
            raw_response_sha256=raw.exact_bytes_sha256,
            parsed_claim_sha256=parsed_claims.semantic_sha256,
            selection_decision_logical_key=selection.selection_decision_logical_key,
        )
        return cls(
            inference_id=raw.inference_id,
            selection_id=selection.selection_id,
            logical_invocation_id=selection.logical_invocation_id,
            selection_decision_logical_key=selection.selection_decision_logical_key,
            selection_policy_version=selection.policy_version,
            raw_response_artifact_id=raw.artifact_id,
            raw_response_sha256=raw.exact_bytes_sha256,
            parsed_claim_artifact_id=parsed_claims.artifact_id,
            parsed_claim_sha256=parsed_claims.semantic_sha256,
            output_sha256=output_digest,
        )


class EnrichmentAuthorityContext(StrictModel):
    """Authoritative lineage injected only after provider parsing succeeds."""

    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    inference_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    prompt_version: SchemaVersion
    prompt_artifact_id: OpaqueUuid
    prompt_sha256: Sha256Digest
    work_node_type: NodeType
    work_node_logical_key: NodeLogicalKey


class EnrichedEvidenceReference(StrictModel):
    """One provider token resolved to authoritative package/frame lineage."""

    correlation_token: ProviderCorrelationToken
    provider_item_ordinal: NonNegativeInt
    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest
    camera_id: CameraId
    camera_ordinal: CameraOrdinal
    frame_id: OpaqueUuid
    frame_ordinal: NonNegativeInt
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    source_artifact_uri: ArtifactUri
    source_artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_camera_ordinal(self) -> Self:
        if CAMERA_IDS[self.camera_ordinal] is not self.camera_id:
            raise ValueError("enriched camera ordinal must match canonical camera identity")
        return self


class ModelReportedUncalibrated(StrictModel):
    """A model score retained without assigning probability semantics."""

    confidence_id: OpaqueUuid
    kind: Literal["MODEL_REPORTED_UNCALIBRATED"]
    semantics: Literal["provider_self_report"]
    producer_type: Literal["MODEL_ATTEMPT"]
    producer_id: OpaqueUuid
    producer_version: SchemaVersion
    source_claim_ordinal: NonNegativeInt
    value: UnitInterval


class EnrichedProviderClaim(StrictModel):
    """Provider claim plus only orchestrator-owned identity and evidence refs."""

    claim_id: OpaqueUuid
    claim_ordinal: NonNegativeInt
    kind: ProviderClaimKind
    package_id: OpaqueUuid | None
    package_ordinal: NonNegativeInt | None
    camera_id: CameraId | None
    interval: ProviderClaimInterval | None
    label: ClaimLabel | None
    observation: ProviderObservation
    evidence: FrozenTuple[EnrichedEvidenceReference]
    model_reported_confidence: ModelReportedUncalibrated | None
    conflict_codes: FrozenTuple[ClaimLabel]

    @model_validator(mode="after")
    def validate_binding_shape(self) -> Self:
        camera_claim = self.kind in _CAMERA_CLAIM_KINDS
        if camera_claim != (
            self.package_id is not None
            and self.package_ordinal is not None
            and self.camera_id is not None
        ):
            raise ValueError("enriched camera identity shape does not match claim kind")
        tokens = tuple(item.correlation_token for item in self.evidence)
        if len(set(tokens)) != len(tokens):
            raise ValueError("enriched evidence references must be unique within a claim")
        if self.camera_id is not None and any(
            item.camera_id is not self.camera_id or item.package_id != self.package_id
            for item in self.evidence
        ):
            raise ValueError("camera claim evidence must match its authoritative camera")
        return self


class OrchestratorEnrichedOutput(StrictModel):
    """Authoritative output validated under a schema distinct from provider claims."""

    schema_version: Literal["2.0"]
    artifact_id: OpaqueUuid
    enrichment_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    task: VisionTask
    selected_attempt: SelectedAttemptOutput
    request_catalog_id: OpaqueUuid
    request_catalog_sha256: Sha256Digest
    reference_catalog_id: OpaqueUuid
    reference_catalog_sha256: Sha256Digest
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    provider_claim_schema: JsonSchemaRef
    enriched_output_schema: JsonSchemaRef
    enrichment_policy_version: SchemaVersion
    authority: EnrichmentAuthorityContext
    claims: FrozenTuple[EnrichedProviderClaim]
    abstained: bool
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        if self.provider_claim_schema.schema_id != PROVIDER_CLAIM_SCHEMA_ID:
            raise ValueError("provider claim schema reference is not authoritative")
        if self.enriched_output_schema.schema_id != ENRICHED_OUTPUT_SCHEMA_ID:
            raise ValueError("enriched output requires its dedicated registered schema")
        if self.enriched_output_schema.version != ENRICHED_OUTPUT_SCHEMA_VERSION:
            raise ValueError("enriched output requires the exact v2 registered schema")
        if self.provider_claim_schema.sha256 == self.enriched_output_schema.sha256:
            raise ValueError("provider and enriched schemas must be distinct artifacts")
        if self.selected_attempt.inference_id != self.authority.inference_id:
            raise ValueError("selected attempt does not match authoritative inference")
        if self.selected_attempt.logical_invocation_id != self.authority.logical_invocation_id:
            raise ValueError("selected attempt does not match authoritative logical invocation")
        ordinals = tuple(claim.claim_ordinal for claim in self.claims)
        if ordinals != tuple(range(len(self.claims))):
            raise ValueError("enriched claim ordinals must be contiguous from zero")
        digest = enrichment_logical_digest(
            selected_attempt_output_sha256=self.selected_attempt.output_sha256,
            request_catalog_sha256=self.request_catalog_sha256,
            target_schema_sha256=self.enriched_output_schema.sha256,
            enrichment_policy_version=self.enrichment_policy_version,
        )
        if self.enrichment_logical_key != f"orchestrator-enrichment:{digest}":
            raise ValueError("enrichment logical key is inconsistent")
        for claim in self.claims:
            expected_claim_id = _stable_uuid("enriched-provider-claim", digest, claim.claim_ordinal)
            if claim.claim_id != expected_claim_id:
                raise ValueError("enriched provider claim ID is inconsistent")
            confidence = claim.model_reported_confidence
            if confidence is not None:
                expected_id = _stable_uuid(
                    "model-reported-uncalibrated", digest, claim.claim_ordinal
                )
                if (
                    confidence.confidence_id != expected_id
                    or confidence.producer_id != self.selected_attempt.inference_id
                    or confidence.source_claim_ordinal != claim.claim_ordinal
                ):
                    raise ValueError("model-reported confidence lineage is inconsistent")
        expected_semantic = semantic_sha256(orchestrator_enriched_output_projection(self))
        if self.semantic_sha256 != expected_semantic:
            raise ValueError("enriched output semantic_sha256 is inconsistent")
        return self


class ProviderClaimEnricher:
    """Validate provider claims and inject authority from one exact input plan."""

    def __init__(self, schema_registry: SchemaRegistry) -> None:
        self._schema_registry = schema_registry

    def enrich(
        self,
        *,
        input_plan: InferenceInputPlan,
        reference_catalog: ProviderReferenceCatalog,
        parsed_claims: ParsedProviderClaimArtifact,
        selected_attempt: SelectedAttemptOutput,
        authority: EnrichmentAuthorityContext,
        enriched_output_schema: JsonSchemaRef,
        enrichment_policy_version: str,
        artifact_id: str,
        created_at: str,
        input_plan_part_ordinal: int | None = None,
    ) -> OrchestratorEnrichedOutput:
        part = _enrichment_part(input_plan, input_plan_part_ordinal)
        self._validate_boundary(
            input_plan=input_plan,
            part=part,
            reference_catalog=reference_catalog,
            parsed_claims=parsed_claims,
            selected_attempt=selected_attempt,
            authority=authority,
            enriched_output_schema=enriched_output_schema,
        )
        logical_digest = enrichment_logical_digest(
            selected_attempt_output_sha256=selected_attempt.output_sha256,
            request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
            target_schema_sha256=enriched_output_schema.sha256,
            enrichment_policy_version=enrichment_policy_version,
        )
        claims = self._enrich_claims(
            input_plan=input_plan,
            part=part,
            reference_catalog=reference_catalog,
            parsed_claims=parsed_claims,
            selected_attempt=selected_attempt,
            logical_digest=logical_digest,
        )
        values = {
            "schema_version": "2.0",
            "artifact_id": artifact_id,
            "enrichment_logical_key": f"orchestrator-enrichment:{logical_digest}",
            "task": parsed_claims.task,
            "selected_attempt": selected_attempt,
            "request_catalog_id": input_plan.request_catalog.request_catalog_id,
            "request_catalog_sha256": input_plan.request_catalog.semantic_sha256,
            "reference_catalog_id": reference_catalog.reference_catalog_id,
            "reference_catalog_sha256": reference_catalog.semantic_sha256,
            "input_plan_id": input_plan.input_plan_id,
            "input_plan_semantic_sha256": input_plan.semantic_sha256,
            "provider_claim_schema": parsed_claims.provider_claim_schema,
            "enriched_output_schema": enriched_output_schema,
            "enrichment_policy_version": enrichment_policy_version,
            "authority": authority,
            "claims": claims,
            "abstained": parsed_claims.payload.abstained,
            "created_at": created_at,
        }
        projection = _enriched_projection_from_values(values)
        output = OrchestratorEnrichedOutput.model_validate(
            {**values, "semantic_sha256": semantic_sha256(projection)},
            strict=True,
        )
        self._schema_registry.validate_pinned(
            _registry_ref(enriched_output_schema), output.model_dump(mode="json")
        )
        return output

    def _validate_boundary(
        self,
        *,
        input_plan: InferenceInputPlan,
        part: InferenceCallPart,
        reference_catalog: ProviderReferenceCatalog,
        parsed_claims: ParsedProviderClaimArtifact,
        selected_attempt: SelectedAttemptOutput,
        authority: EnrichmentAuthorityContext,
        enriched_output_schema: JsonSchemaRef,
    ) -> None:
        catalog = input_plan.request_catalog
        if (
            parsed_claims.task is not input_plan.subject.task
            or catalog.task is not parsed_claims.task
        ):
            raise ProviderClaimEnrichmentError("provider claims do not match the input-plan task")
        expected_reference_entries = _expected_reference_entries(
            input_plan, reference_catalog.token_policy_version
        )
        if (
            reference_catalog.input_plan_id != input_plan.input_plan_id
            or reference_catalog.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or reference_catalog.request_catalog_id != catalog.request_catalog_id
            or reference_catalog.request_catalog_sha256 != catalog.semantic_sha256
            or reference_catalog.task is not catalog.task
            or reference_catalog.entries != expected_reference_entries
        ):
            raise ProviderClaimEnrichmentError(
                "provider reference catalog does not match input plan"
            )
        if (
            parsed_claims.provider_claim_schema.sha256
            != input_plan.prompt_output.provider_response_schema_sha256
        ):
            raise ProviderClaimEnrichmentError("provider claim schema is not bound by input plan")
        if enriched_output_schema.sha256 != input_plan.prompt_output.enriched_domain_schema_sha256:
            raise ProviderClaimEnrichmentError("enriched output schema is not bound by input plan")
        if enriched_output_schema.schema_id != ENRICHED_OUTPUT_SCHEMA_ID:
            raise ProviderClaimEnrichmentError("unexpected enriched output schema identity")
        if enriched_output_schema.version != ENRICHED_OUTPUT_SCHEMA_VERSION:
            raise ProviderClaimEnrichmentError("enriched output schema must be pinned to v2")
        if parsed_claims.provider_claim_schema.sha256 == enriched_output_schema.sha256:
            raise ProviderClaimEnrichmentError("provider and enriched schemas must be distinct")
        raw = parsed_claims.raw_response
        if (
            selected_attempt.inference_id != raw.inference_id
            or selected_attempt.raw_response_artifact_id != raw.artifact_id
            or selected_attempt.raw_response_sha256 != raw.exact_bytes_sha256
            or selected_attempt.parsed_claim_artifact_id != parsed_claims.artifact_id
            or selected_attempt.parsed_claim_sha256 != parsed_claims.semantic_sha256
            or authority.inference_id != selected_attempt.inference_id
            or authority.logical_invocation_id != selected_attempt.logical_invocation_id
        ):
            raise ProviderClaimEnrichmentError("selected attempt artifact lineage is inconsistent")
        if (
            raw.provider != input_plan.target.provider
            or raw.model_name != input_plan.target.model_name
            or raw.model_version != input_plan.target.model_version
        ):
            raise ProviderClaimEnrichmentError("raw provider target does not match input plan")
        if (
            authority.prompt_version != input_plan.prompt_output.prompt_version
            or authority.prompt_sha256 != input_plan.prompt_output.prompt_sha256
        ):
            raise ProviderClaimEnrichmentError("prompt authority does not match input plan")

        provider_ref = _registry_ref(parsed_claims.provider_claim_schema)
        enriched_ref = _registry_ref(enriched_output_schema)
        self._schema_registry.resolve_exact(provider_ref)
        self._schema_registry.resolve_exact(enriched_ref)
        self._schema_registry.validate_pinned(
            provider_ref, parsed_claims.payload.model_dump(mode="json")
        )
        self._validate_camera_coverage(input_plan, part, parsed_claims)

    @staticmethod
    def _validate_camera_coverage(
        input_plan: InferenceInputPlan,
        part: InferenceCallPart,
        parsed_claims: ParsedProviderClaimArtifact,
    ) -> None:
        camera_kind = {
            VisionTask.QA_COARSE: ProviderClaimKind.QA_OBSERVATION,
            VisionTask.QA_DENSE: ProviderClaimKind.QA_OBSERVATION,
            VisionTask.ACTION_EVIDENCE: ProviderClaimKind.ACTION_OBSERVATION,
            VisionTask.BOUNDARY_REFINEMENT: ProviderClaimKind.BOUNDARY_OBSERVATION,
        }.get(parsed_claims.task)
        if camera_kind is None:
            return
        expected = {
            (item.package_ordinal, item.camera_ordinal)
            for item in input_plan.rendered_items[
                part.start_item_ordinal : part.end_item_ordinal_exclusive
            ]
        }
        actual_list = [
            (claim.package_ordinal, claim.camera_ordinal)
            for claim in parsed_claims.payload.claims
            if claim.kind is camera_kind
        ]
        if len(actual_list) != len(set(actual_list)) or set(actual_list) != expected:
            raise ProviderClaimEnrichmentError(
                "camera claims must cover each request-catalog package/camera exactly once"
            )

    @staticmethod
    def _enrich_claims(
        *,
        input_plan: InferenceInputPlan,
        part: InferenceCallPart,
        reference_catalog: ProviderReferenceCatalog,
        parsed_claims: ParsedProviderClaimArtifact,
        selected_attempt: SelectedAttemptOutput,
        logical_digest: str,
    ) -> tuple[EnrichedProviderClaim, ...]:
        entries = {entry.correlation_token: entry for entry in reference_catalog.entries}
        allowed_coordinates = {
            (item.package_ordinal, item.camera_ordinal)
            for item in input_plan.rendered_items[
                part.start_item_ordinal : part.end_item_ordinal_exclusive
            ]
        }
        result: list[EnrichedProviderClaim] = []
        for claim in parsed_claims.payload.claims:
            package = None
            camera = None
            if claim.kind in _CAMERA_CLAIM_KINDS:
                assert claim.package_ordinal is not None
                assert claim.camera_ordinal is not None
                try:
                    package = input_plan.request_catalog.packages[claim.package_ordinal]
                    camera = package.cameras[claim.camera_ordinal]
                except IndexError as exc:
                    raise ProviderClaimEnrichmentError(
                        "provider local package or camera ordinal is out of catalog"
                    ) from exc
                if (package.ordinal, camera.ordinal) not in allowed_coordinates:
                    raise ProviderClaimEnrichmentError(
                        "provider local package or camera ordinal is outside the call part"
                    )

            evidence: list[EnrichedEvidenceReference] = []
            for token in claim.evidence_tokens:
                entry = entries.get(token)
                if entry is None:
                    raise ProviderClaimEnrichmentError(
                        "provider evidence token is outside the request catalog"
                    )
                if not (
                    part.start_item_ordinal
                    <= entry.provider_item_ordinal
                    < part.end_item_ordinal_exclusive
                ):
                    raise ProviderClaimEnrichmentError(
                        "provider evidence token is outside the selected call part"
                    )
                if (
                    package is not None
                    and camera is not None
                    and (
                        entry.package_ordinal != package.ordinal
                        or entry.camera_ordinal != camera.ordinal
                    )
                ):
                    raise ProviderClaimEnrichmentError(
                        "provider evidence token does not match claim-local ordinals"
                    )
                bound_package = input_plan.request_catalog.packages[entry.package_ordinal]
                bound_camera = bound_package.cameras[entry.camera_ordinal]
                try:
                    frame = bound_camera.frames[entry.frame_ordinal]
                except IndexError as exc:
                    raise ProviderClaimEnrichmentError(
                        "provider frame ordinal is outside the request catalog"
                    ) from exc
                evidence.append(
                    EnrichedEvidenceReference(
                        correlation_token=token,
                        provider_item_ordinal=entry.provider_item_ordinal,
                        package_id=bound_package.package_id,
                        package_ordinal=bound_package.ordinal,
                        package_semantic_content_sha256=bound_package.semantic_content_sha256,
                        package_manifest_sha256=bound_package.manifest_bytes_sha256,
                        camera_id=bound_camera.camera_id,
                        camera_ordinal=bound_camera.ordinal,
                        frame_id=frame.frame_id,
                        frame_ordinal=frame.ordinal,
                        aligned_timestamp_ns=frame.aligned_timestamp_ns,
                        source_timestamp_ns=frame.source_timestamp_ns,
                        source_artifact_uri=frame.source_artifact_uri,
                        source_artifact_sha256=frame.source_artifact_sha256,
                    )
                )
            evidence.sort(key=lambda item: item.provider_item_ordinal)

            confidence = None
            if claim.model_reported_score is not None:
                confidence = ModelReportedUncalibrated(
                    confidence_id=_stable_uuid(
                        "model-reported-uncalibrated", logical_digest, claim.claim_ordinal
                    ),
                    kind="MODEL_REPORTED_UNCALIBRATED",
                    semantics="provider_self_report",
                    producer_type="MODEL_ATTEMPT",
                    producer_id=selected_attempt.inference_id,
                    producer_version=parsed_claims.raw_response.model_version,
                    source_claim_ordinal=claim.claim_ordinal,
                    value=claim.model_reported_score,
                )
            result.append(
                EnrichedProviderClaim(
                    claim_id=_stable_uuid(
                        "enriched-provider-claim", logical_digest, claim.claim_ordinal
                    ),
                    claim_ordinal=claim.claim_ordinal,
                    kind=claim.kind,
                    package_id=package.package_id if package is not None else None,
                    package_ordinal=package.ordinal if package is not None else None,
                    camera_id=camera.camera_id if camera is not None else None,
                    interval=claim.interval,
                    label=claim.label,
                    observation=claim.observation,
                    evidence=tuple(evidence),
                    model_reported_confidence=confidence,
                    conflict_codes=claim.conflict_codes,
                )
            )
        return tuple(result)


def provider_reference_catalog_projection(
    catalog: ProviderReferenceCatalog,
) -> dict[str, object]:
    """Return the row-ID and wall-clock independent reference-catalog projection."""

    return {
        "input_plan_semantic_sha256": catalog.input_plan_semantic_sha256,
        "request_catalog_sha256": catalog.request_catalog_sha256,
        "task": catalog.task.value,
        "token_policy_version": catalog.token_policy_version,
        "entries": [entry.model_dump(mode="json") for entry in catalog.entries],
    }


def parsed_provider_claim_projection(
    artifact: ParsedProviderClaimArtifact,
) -> dict[str, object]:
    """Return parsed-claim semantics without locator identities or clock fields."""

    return {
        "raw_provider_bytes_sha256": artifact.raw_response.exact_bytes_sha256,
        "provider_claim_schema": _json_schema_logical_projection(artifact.provider_claim_schema),
        "task": artifact.task.value,
        "payload": artifact.payload.model_dump(mode="json"),
        "parser_version": artifact.parser_version,
    }


def _json_schema_logical_projection(reference: JsonSchemaRef) -> dict[str, object]:
    return {
        "schema_id": reference.schema_id,
        "version": reference.version,
        "sha256": reference.sha256,
    }


def enrichment_logical_digest(
    *,
    selected_attempt_output_sha256: str,
    request_catalog_sha256: str,
    target_schema_sha256: str,
    enrichment_policy_version: str,
) -> Sha256Digest:
    """Derive the exact Architecture V1.1 Section 25.1 enrichment key digest."""

    return semantic_sha256(
        {
            "selected_attempt_output_sha256": selected_attempt_output_sha256,
            "request_catalog_sha256": request_catalog_sha256,
            "target_schema_sha256": target_schema_sha256,
            "enrichment_policy_version": enrichment_policy_version,
        }
    )


def _selected_attempt_output_digest(
    *,
    raw_response_sha256: str,
    parsed_claim_sha256: str,
    selection_decision_logical_key: str,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "raw_provider_bytes_sha256": raw_response_sha256,
            "parsed_provider_claim_sha256": parsed_claim_sha256,
            "selection_decision_logical_key": selection_decision_logical_key,
        }
    )


def orchestrator_enriched_output_projection(
    output: OrchestratorEnrichedOutput,
) -> dict[str, object]:
    """Return immutable enriched content excluding only artifact locator and clock."""

    values = output.model_dump(mode="json", exclude={"semantic_sha256"})
    values.pop("artifact_id")
    values.pop("created_at")
    return values


def _enriched_projection_from_values(values: dict[str, object]) -> dict[str, object]:
    projection: dict[str, object] = {}
    for key, value in values.items():
        if key not in {"artifact_id", "created_at"}:
            if hasattr(value, "model_dump"):
                projection[key] = value.model_dump(mode="json")
            elif isinstance(value, tuple):
                projection[key] = [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in value
                ]
            elif isinstance(value, StrEnum):
                projection[key] = value.value
            else:
                projection[key] = value
    return projection


def _expected_reference_entries(
    input_plan: InferenceInputPlan,
    token_policy_version: str,
) -> tuple[ProviderReferenceCatalogEntry, ...]:
    return _reference_entries_for_rendering(
        request_catalog_sha256=input_plan.request_catalog.semantic_sha256,
        rendered_items=input_plan.rendered_items,
        token_policy_version=token_policy_version,
    )


def _enrichment_part(
    input_plan: InferenceInputPlan,
    part_ordinal: int | None,
) -> InferenceCallPart:
    if not isinstance(input_plan, InferenceInputPlan):
        raise ProviderClaimEnrichmentError("input plan failed immutable contract validation")
    parts = input_plan.call_plan.parts
    if part_ordinal is None:
        if len(parts) != 1:
            raise ProviderClaimEnrichmentError(
                "multi-part enrichment requires an explicit input-plan part ordinal"
            )
        return parts[0]
    if isinstance(part_ordinal, bool) or not isinstance(part_ordinal, int):
        raise ProviderClaimEnrichmentError("input-plan part ordinal must be an integer")
    if part_ordinal < 0 or part_ordinal >= len(parts):
        raise ProviderClaimEnrichmentError("input-plan part ordinal is out of range")
    return parts[part_ordinal]


def _reference_entries_for_rendering(
    *,
    request_catalog_sha256: str,
    rendered_items: Sequence[RenderedProviderItem],
    token_policy_version: str,
) -> tuple[ProviderReferenceCatalogEntry, ...]:
    entries: list[ProviderReferenceCatalogEntry] = []
    for ordinal, item in enumerate(rendered_items):
        if item.provider_item_ordinal != ordinal:
            raise ValueError("rendered provider item ordinals must be contiguous")
        entries.append(
            ProviderReferenceCatalogEntry(
                correlation_token=_correlation_token(
                    request_catalog_sha256=request_catalog_sha256,
                    token_policy_version=token_policy_version,
                    provider_item_ordinal=ordinal,
                    package_ordinal=item.package_ordinal,
                    camera_ordinal=item.camera_ordinal,
                    frame_ordinal=item.frame_ordinal,
                ),
                provider_item_ordinal=ordinal,
                package_ordinal=item.package_ordinal,
                camera_ordinal=item.camera_ordinal,
                frame_ordinal=item.frame_ordinal,
            )
        )
    return tuple(entries)


def _correlation_token(
    *,
    request_catalog_sha256: str,
    token_policy_version: str,
    provider_item_ordinal: int,
    package_ordinal: int,
    camera_ordinal: int,
    frame_ordinal: int,
) -> str:
    digest = semantic_sha256(
        {
            "request_catalog_sha256": request_catalog_sha256,
            "token_policy_version": token_policy_version,
            "provider_item_ordinal": provider_item_ordinal,
            "package_ordinal": package_ordinal,
            "camera_ordinal": camera_ordinal,
            "frame_ordinal": frame_ordinal,
        }
    )
    return f"ref:{digest}"


def _registry_ref(ref: JsonSchemaRef) -> SchemaRef:
    return SchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _stable_uuid(namespace: str, digest: str, ordinal: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{digest}:{ordinal}"))


__all__ = [
    "ENRICHED_OUTPUT_SCHEMA_ID",
    "ENRICHED_OUTPUT_SCHEMA_VERSION",
    "PROVIDER_CLAIM_SCHEMA_ID",
    "EnrichedEvidenceReference",
    "EnrichedProviderClaim",
    "EnrichmentAuthorityContext",
    "ModelReportedUncalibrated",
    "OrchestratorEnrichedOutput",
    "ParsedProviderClaimArtifact",
    "ProviderClaimEnricher",
    "ProviderClaimEnrichmentError",
    "ProviderClaimInterval",
    "ProviderClaimKind",
    "ProviderClaimPayload",
    "ProviderCorrelationToken",
    "ProviderObservation",
    "ProviderReferenceCatalog",
    "ProviderReferenceCatalogEntry",
    "ProviderTaskClaim",
    "RawProviderResponseArtifact",
    "SelectedAttemptOutput",
    "enrichment_logical_digest",
    "orchestrator_enriched_output_projection",
    "parsed_provider_claim_projection",
    "provider_reference_catalog_projection",
]
