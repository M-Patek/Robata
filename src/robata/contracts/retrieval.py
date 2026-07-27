"""Versioned contracts for the optional vector-retrieval projection.

The canonical event index remains the authoritative retrieval path.  This module
describes the *derived* embedding projection only: model metadata, subject
lineage, vector rows, backend locators, access/retention policy, and bounded
backfill requests.  None of the transport metadata below participates in event,
revision, artifact, or recording identity.

The contracts are intentionally provider-neutral.  A Postgres/Supabase adapter
may use ``VectorLocator`` to describe where a row was stored and pgvector to
search it, while an explicit encoder (CPU, API, or RunPod) is responsible for
creating the vector.  A failed or unavailable projection must therefore never
re-open a completed QA result.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AliasChoices, Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
# Pydantic accepts NaN/Infinity for a plain strict ``float`` by default. Those
# values are not meaningful in a cosine index (and are not stable JSON), so the
# reusable type is finite at every contract boundary, including encoder output.
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


RETRIEVAL_PROJECTION_CONTRACT_VERSION: Literal["1.0"] = "1.0"
VECTOR_PROJECTION_SEMANTIC_PROJECTION_VERSION: Literal["vector-projection-v1"] = (
    "vector-projection-v1"
)
# These versions make the structured subject/idempotency key migration
# explicit.  Delimiter-concatenated opaque IDs are not injective.
VECTOR_PROJECTION_SUBJECT_KEY_VERSION: Literal["vector-subject-key-v2"] = (
    "vector-subject-key-v2"
)
VECTOR_PROJECTION_IDEMPOTENCY_KEY_VERSION: Literal["vector-idempotency-key-v2"] = (
    "vector-idempotency-key-v2"
)
VectorProjectionSemanticProjectionVersion = VECTOR_PROJECTION_SEMANTIC_PROJECTION_VERSION


class EmbeddingModality(StrEnum):
    """Input modality used to produce an embedding."""

    TEXT = "TEXT"
    VISION = "VISION"


class EmbeddingNormalization(StrEnum):
    """Normalization promised by an encoder and consumed by a vector index."""

    NONE = "NONE"
    L2 = "L2"


class EmbeddingProvider(StrEnum):
    """Where an encoder runs; this is not part of a canonical event identity."""

    CPU = "CPU"
    API = "API"
    RUNPOD = "RUNPOD"


class VectorIndexKind(StrEnum):
    """Optional pgvector index strategy."""

    NONE = "NONE"
    HNSW = "HNSW"
    IVFFLAT = "IVFFLAT"


class VectorProjectionStatus(StrEnum):
    """Lifecycle of one asynchronous derived vector row."""

    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"
    RETIRED = "RETIRED"


class VectorSubjectType(StrEnum):
    """Canonical subject represented by a derived vector row."""

    EVENT_REVISION = "EVENT_REVISION"
    CLIP_ARTIFACT = "CLIP_ARTIFACT"
    TEMPORAL_PACKAGE = "TEMPORAL_PACKAGE"


class VectorBackend(StrEnum):
    """Storage backend named by a locator; pgvector is a storage/search layer."""

    POSTGRES = "POSTGRES"
    SUPABASE = "SUPABASE"
    MEMORY = "MEMORY"


class EmbeddingSpec(StrictModel):
    """Versioned model and index metadata for a vector family.

    ``embedding_id`` is the stable logical family identifier.  Changing model,
    model version, modality, dimension, normalization, or index policy requires
    a new spec (and consequently a new embedding ID), rather than mutating an
    existing vector row in place.
    """

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    embedding_id: NonEmptyString
    model: NonEmptyString
    model_version: SchemaVersion
    modality: EmbeddingModality
    dimension: Annotated[int, Field(strict=True, gt=0, le=65535)]
    normalization: EmbeddingNormalization = EmbeddingNormalization.NONE
    encoder_provider: EmbeddingProvider
    index_policy_version: SchemaVersion
    index_kind: VectorIndexKind = VectorIndexKind.NONE

    @property
    def semantic_projection(self) -> dict[str, object]:
        """Identity-bearing model metadata, excluding transport locators."""

        return {
            "semantic_projection_version": VECTOR_PROJECTION_SEMANTIC_PROJECTION_VERSION,
            "contract_version": self.contract_version,
            "embedding_id": self.embedding_id,
            "model": self.model,
            "model_version": self.model_version,
            "modality": self.modality.value,
            "dimension": self.dimension,
            "normalization": self.normalization.value,
            "encoder_provider": self.encoder_provider.value,
            "index_policy_version": self.index_policy_version,
            "index_kind": self.index_kind.value,
        }

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return semantic_sha256(self.semantic_projection)


class VectorProjectionSubject(StrictModel):
    """Lineage key for one vector row.

    A row is anchored to an immutable event revision and may additionally bind a
    clip/package artifact.  Artifact IDs are optional for text/event embeddings,
    but when an artifact ID is supplied its exact digest is required as well.
    """

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    subject_type: VectorSubjectType = VectorSubjectType.EVENT_REVISION
    event_revision_id: NonEmptyString
    artifact_id: NonEmptyString | None = None
    artifact_sha256: Sha256Digest | None = None
    source_semantic_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> Self:
        if (self.artifact_id is None) != (self.artifact_sha256 is None):
            raise ValueError("artifact_id and artifact_sha256 must be supplied together")
        if (
            self.subject_type
            in {VectorSubjectType.CLIP_ARTIFACT, VectorSubjectType.TEMPORAL_PACKAGE}
            and self.artifact_id is None
        ):
            raise ValueError(
                f"{self.subject_type.value} subjects require artifact_id and artifact_sha256"
            )
        return self

    @property
    def projection_key(self) -> str:
        """Transport-independent, injective key for the subject lineage.

        The key is versioned and digest-backed instead of joining opaque IDs
        with delimiters.  This prevents an event ID containing ``:artifact:``
        from colliding with a separate artifact-bound subject.
        """

        return f"{VECTOR_PROJECTION_SUBJECT_KEY_VERSION}:{self.semantic_sha256}"

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return semantic_sha256(
            {
                "contract_version": self.contract_version,
                "subject_type": self.subject_type.value,
                "event_revision_id": self.event_revision_id,
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "source_semantic_sha256": self.source_semantic_sha256,
            }
        )


class VectorLocator(StrictModel):
    """Backend locator metadata for a vector row.

    Table names, row IDs, index names, and provider versions are useful for
    reconciliation, but deliberately have no content identity semantics.
    """

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    backend: VectorBackend
    relation: NonEmptyString
    vector_column: NonEmptyString = "embedding"
    row_key: NonEmptyString
    index_name: NonEmptyString | None = None
    provider_version: SchemaVersion | None = None

    @property
    def metadata_projection(self) -> dict[str, str]:
        projection = {
            "contract_version": self.contract_version,
            "backend": self.backend.value,
            "relation": self.relation,
            "vector_column": self.vector_column,
            "row_key": self.row_key,
        }
        if self.index_name is not None:
            projection["index_name"] = self.index_name
        if self.provider_version is not None:
            projection["provider_version"] = self.provider_version
        return projection

    @property
    def content_identity(self) -> None:
        """Locators cannot silently become vector or event identity."""

        return None


class VectorRetentionPolicy(StrictModel):
    """Versioned retention decision for optional vector rows."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    retention_policy_version: SchemaVersion
    ttl_days: NonNegativeInt | None = None
    retain_until: Rfc3339Timestamp | None = None
    delete_with_source: bool = True
    legal_hold: bool = False

    @model_validator(mode="after")
    def validate_retention(self) -> Self:
        if self.ttl_days is None and self.retain_until is None and not self.legal_hold:
            raise ValueError("retention policy must define ttl_days, retain_until, or legal_hold")
        if self.legal_hold and self.ttl_days is not None:
            raise ValueError("legal_hold cannot be combined with ttl_days")
        return self


class VectorAccessPolicy(StrictModel):
    """Application-level RLS metadata carried alongside a projection request."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    policy_version: SchemaVersion
    tenant_id: NonEmptyString
    allowed_roles: tuple[NonEmptyString, ...] = ()
    row_level_security_required: bool = True

    @model_validator(mode="after")
    def validate_roles(self) -> Self:
        if len(set(self.allowed_roles)) != len(self.allowed_roles):
            raise ValueError("allowed_roles must be unique")
        return self


class VectorProjectionRequest(StrictModel):
    """Idempotent asynchronous write request for a derived vector."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    subject: VectorProjectionSubject
    embedding: EmbeddingSpec
    vector: tuple[FiniteFloat, ...] = Field(min_length=1)
    retention: VectorRetentionPolicy
    access_policy: VectorAccessPolicy | None = None
    requested_at: Rfc3339Timestamp | None = None

    @field_validator("vector")
    @classmethod
    def validate_finite_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not math.isfinite(item) for item in value):
            raise ValueError("vector must contain finite values")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if len(self.vector) != self.embedding.dimension:
            raise ValueError(
                f"vector dimension {len(self.vector)} does not match embedding dimension "
                f"{self.embedding.dimension}"
            )
        if self.embedding.normalization is EmbeddingNormalization.L2:
            norm = math.sqrt(sum(item * item for item in self.vector))
            if norm == 0.0:
                raise ValueError("L2-normalized vectors cannot be all zero")
        return self

    @property
    def vector_sha256(self) -> Sha256Digest:
        return semantic_sha256({"vector": list(self.vector)})

    @property
    def idempotency_key(self) -> str:
        """Identity used for retry/duplicate writes; locators are absent."""

        return semantic_sha256(
            {
                "idempotency_key_version": VECTOR_PROJECTION_IDEMPOTENCY_KEY_VERSION,
                "contract_version": self.contract_version,
                "subject": self.subject.projection_key,
                "embedding_id": self.embedding.embedding_id,
                "embedding_semantic_sha256": self.embedding.semantic_sha256,
            }
        )


class VectorProjection(StrictModel):
    """Persisted/observed state of one asynchronous vector projection."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    projection_id: NonEmptyString
    request: VectorProjectionRequest
    status: VectorProjectionStatus = VectorProjectionStatus.PENDING
    locator: VectorLocator | None = None
    error: NonEmptyString | None = None
    attempts: PositiveInt = 1
    created_at: Rfc3339Timestamp | None = None
    updated_at: Rfc3339Timestamp | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.status is VectorProjectionStatus.READY and self.locator is None:
            raise ValueError("READY vector projection requires a locator")
        if self.status is VectorProjectionStatus.FAILED and self.error is None:
            raise ValueError("FAILED vector projection requires an error")
        if self.status is not VectorProjectionStatus.FAILED and self.error is not None:
            raise ValueError("only FAILED vector projections may carry an error")
        return self

    @property
    def idempotency_key(self) -> str:
        return self.request.idempotency_key

    @property
    def vector_sha256(self) -> Sha256Digest:
        return self.request.vector_sha256


class VectorProjectionReceipt(StrictModel):
    """Idempotent acknowledgement for enqueue/upsert."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    idempotency_key: NonEmptyString
    projection_id: NonEmptyString
    status: VectorProjectionStatus
    duplicate: bool = False
    queued: bool = True


class VectorSearchQuery(StrictModel):
    """Vector query constrained to a versioned embedding family.

    ``vector`` and ``embedding_vector`` are accepted aliases for callers that
    already use the retrieval service terminology; serialization remains
    canonical as ``query_vector``.
    """

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    embedding_id: NonEmptyString
    tenant_id: NonEmptyString | None = None
    query_vector: tuple[FiniteFloat, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("query_vector", "vector", "embedding_vector")
    )
    candidate_event_revision_ids: tuple[NonEmptyString, ...] = ()
    candidate_subject_ids: tuple[NonEmptyString, ...] = ()
    limit: Annotated[int, Field(strict=True, ge=1, le=1000)] = 50
    min_score: Annotated[
        float,
        Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False),
    ] | None = None

    @field_validator("query_vector")
    @classmethod
    def validate_query_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("query_vector must contain finite values")
        return value

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> Self:
        for name, values in (
            ("candidate_event_revision_ids", self.candidate_event_revision_ids),
            ("candidate_subject_ids", self.candidate_subject_ids),
        ):
            if tuple(values) != tuple(sorted(values)) or len(set(values)) != len(values):
                raise ValueError(f"{name} must be sorted and unique")
        return self


class VectorSearchHit(StrictModel):
    """A vector candidate that can be joined back to structured EventIndex."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    projection_id: NonEmptyString
    event_revision_id: NonEmptyString
    artifact_id: NonEmptyString | None = None
    embedding_id: NonEmptyString
    score: Annotated[float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)]
    rank: Annotated[int, Field(strict=True, ge=0)]
    locator: VectorLocator | None = None


class EmbeddingBackfillRequest(StrictModel):
    """Bounded, replay-safe request for asynchronous vector backfill."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    embedding: EmbeddingSpec
    event_revision_ids: tuple[NonEmptyString, ...] = ()
    artifact_ids: tuple[NonEmptyString, ...] = ()
    batch_size: Annotated[int, Field(strict=True, ge=1, le=10000)] = 100
    cursor: NonEmptyString | None = None
    requested_at: Rfc3339Timestamp | None = None

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        if not self.event_revision_ids and not self.artifact_ids and self.cursor is None:
            raise ValueError("backfill must specify targets or a cursor")
        if self.cursor is not None and (self.event_revision_ids or self.artifact_ids):
            raise ValueError("backfill cursor cannot be combined with explicit targets")
        for name, values in (
            ("event_revision_ids", self.event_revision_ids),
            ("artifact_ids", self.artifact_ids),
        ):
            if tuple(values) != tuple(sorted(values)) or len(set(values)) != len(values):
                raise ValueError(f"{name} must be sorted and unique")
        return self

    @property
    def idempotency_key(self) -> str:
        return semantic_sha256(
            {
                "embedding_semantic_sha256": self.embedding.semantic_sha256,
                "event_revision_ids": list(self.event_revision_ids),
                "artifact_ids": list(self.artifact_ids),
                "batch_size": self.batch_size,
                "cursor": self.cursor,
            }
        )


class EmbeddingEncoderInput(StrictModel):
    """Input lineage passed to an explicit text/vision encoder."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    subject: VectorProjectionSubject
    embedding: EmbeddingSpec
    text: NonEmptyString | None = None
    artifact_locator: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_input_kind(self) -> Self:
        if self.embedding.modality is EmbeddingModality.TEXT:
            if self.text is None:
                raise ValueError("TEXT encoder input requires text")
            if self.artifact_locator is not None:
                raise ValueError("TEXT encoder input cannot carry artifact_locator")
        elif self.embedding.modality is EmbeddingModality.VISION:
            if self.artifact_locator is None:
                raise ValueError("VISION encoder input requires artifact_locator")
            if self.subject.subject_type not in {
                VectorSubjectType.CLIP_ARTIFACT,
                VectorSubjectType.TEMPORAL_PACKAGE,
            }:
                raise ValueError("VISION encoder input requires an artifact subject")
            if self.text is not None:
                raise ValueError("VISION encoder input cannot carry text")
        return self


class EncodedEmbedding(StrictModel):
    """Encoder output before it enters the asynchronous vector projection."""

    contract_version: Literal["1.0"] = RETRIEVAL_PROJECTION_CONTRACT_VERSION
    subject: VectorProjectionSubject
    embedding: EmbeddingSpec
    vector: tuple[FiniteFloat, ...] = Field(min_length=1)
    encoder_run_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if len(self.vector) != self.embedding.dimension:
            raise ValueError("encoded vector dimension does not match embedding spec")
        if self.embedding.normalization is EmbeddingNormalization.L2:
            norm = math.sqrt(sum(item * item for item in self.vector))
            if norm == 0.0:
                raise ValueError("L2-normalized vectors cannot be all zero")
        return self

    def to_projection_request(
        self,
        *,
        retention: VectorRetentionPolicy,
        access_policy: VectorAccessPolicy | None = None,
        requested_at: Rfc3339Timestamp | None = None,
    ) -> VectorProjectionRequest:
        return VectorProjectionRequest(
            subject=self.subject,
            embedding=self.embedding,
            vector=self.vector,
            retention=retention,
            access_policy=access_policy,
            requested_at=requested_at,
        )


# Compatibility aliases make the boundary discoverable without introducing a
# second schema or a second identity formula.
EmbeddingMetadata = EmbeddingSpec
EmbeddingModelSpec = EmbeddingSpec
VectorProjectionKey = VectorProjectionSubject
VectorEmbedding = VectorProjection
EmbeddingProjection = VectorProjection
VectorStoreRequest = VectorProjectionRequest
VectorSearchRequest = VectorSearchQuery
VectorHit = VectorSearchHit
VectorRetention = VectorRetentionPolicy
VectorProjectionLocator = VectorLocator
VectorRetentionContract = VectorRetentionPolicy
VectorRlsPolicy = VectorAccessPolicy
EmbeddingModelVersion = EmbeddingSpec
VectorQuery = VectorSearchQuery


__all__ = [
    "RETRIEVAL_PROJECTION_CONTRACT_VERSION",
    "VECTOR_PROJECTION_IDEMPOTENCY_KEY_VERSION",
    "VECTOR_PROJECTION_SEMANTIC_PROJECTION_VERSION",
    "VECTOR_PROJECTION_SUBJECT_KEY_VERSION",
    "EmbeddingBackfillRequest",
    "EmbeddingEncoderInput",
    "EmbeddingMetadata",
    "EmbeddingModality",
    "EmbeddingModelSpec",
    "EmbeddingModelVersion",
    "EmbeddingNormalization",
    "EmbeddingProjection",
    "EmbeddingProvider",
    "EmbeddingSpec",
    "EncodedEmbedding",
    "FiniteFloat",
    "VectorAccessPolicy",
    "VectorBackend",
    "VectorEmbedding",
    "VectorHit",
    "VectorIndexKind",
    "VectorLocator",
    "VectorProjection",
    "VectorProjectionKey",
    "VectorProjectionLocator",
    "VectorProjectionReceipt",
    "VectorProjectionRequest",
    "VectorProjectionSemanticProjectionVersion",
    "VectorProjectionStatus",
    "VectorProjectionSubject",
    "VectorQuery",
    "VectorRetention",
    "VectorRetentionContract",
    "VectorRetentionPolicy",
    "VectorRlsPolicy",
    "VectorSearchHit",
    "VectorSearchQuery",
    "VectorSearchRequest",
    "VectorStoreRequest",
    "VectorSubjectType",
]

