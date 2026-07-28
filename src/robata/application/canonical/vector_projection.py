"""Detached, revision-bound P14 vector projection intent and dispatch.

This module intentionally sits outside terminal completion and EventIndex
projection.  It turns an already encoded optional vector into an asynchronous
request only after binding it to the immutable event-revision semantic digest.
Adapter errors become observable optional-projection outcomes; they never
change completion, event identity, current revision selection, or retrieval
authority.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, Rfc3339Timestamp
from robata.contracts.retrieval import (
    EncodedEmbedding,
    VectorAccessPolicy,
    VectorIndexKind,
    VectorProjectionReceipt,
    VectorProjectionRequest,
    VectorRetentionPolicy,
)
from robata.ports.vector_projection import (
    VectorProjectionError,
    VectorProjectionErrorCode,
    VectorProjectionStore,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]

CANONICAL_VECTOR_POLICY_PROJECTION_VERSION: Final = "canonical-vector-policy-semantic-v1"
CANONICAL_VECTOR_INTENT_PROJECTION_VERSION: Final = "canonical-vector-intent-semantic-v1"
CANONICAL_VECTOR_POLICY_KEY_NAMESPACE: Final = "canonical-vector-policy-v1"
CANONICAL_VECTOR_INTENT_KEY_NAMESPACE: Final = "canonical-vector-intent-v1"


class VectorDistanceMetric(StrEnum):
    """Distance interpretation pinned beside an immutable embedding family."""

    COSINE = "COSINE"


class CanonicalVectorIndexParameter(StrictModel):
    """One exact, canonical physical-index parameter citation."""

    name: NonEmptyString
    value: NonEmptyString


class CanonicalVectorProjectionPolicy(StrictModel):
    """Internal versioned encoder/preprocessing/distance/index policy.

    ``EmbeddingSpec`` remains the released v1 contract.  This policy binds
    implementation facts that cannot be added to that existing semantic
    identity without a contract migration.
    """

    policy_version: SchemaVersion
    embedding_id: NonEmptyString
    embedding_semantic_sha256: Sha256Digest
    preprocessing_policy_version: SchemaVersion
    preprocessing_semantic_sha256: Sha256Digest
    distance_metric: VectorDistanceMetric = VectorDistanceMetric.COSINE
    index_kind: VectorIndexKind
    index_parameters: tuple[CanonicalVectorIndexParameter, ...] = ()
    projection_version: Literal["canonical-vector-policy-semantic-v1"] = (
        CANONICAL_VECTOR_POLICY_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        policy_version: str,
        embedding_id: str,
        embedding_semantic_sha256: str,
        preprocessing_policy_version: str,
        preprocessing_semantic_sha256: str,
        index_kind: VectorIndexKind,
        index_parameters: tuple[CanonicalVectorIndexParameter, ...] = (),
    ) -> Self:
        """Create a content-addressed internal policy without mutating v1."""

        values: dict[str, object] = {
            "policy_version": policy_version,
            "embedding_id": embedding_id,
            "embedding_semantic_sha256": embedding_semantic_sha256,
            "preprocessing_policy_version": preprocessing_policy_version,
            "preprocessing_semantic_sha256": preprocessing_semantic_sha256,
            "distance_metric": VectorDistanceMetric.COSINE,
            "index_kind": index_kind,
            "index_parameters": index_parameters,
            "projection_version": CANONICAL_VECTOR_POLICY_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{CANONICAL_VECTOR_POLICY_KEY_NAMESPACE}:{'0' * 64}",
            policy_version=policy_version,
            embedding_id=embedding_id,
            embedding_semantic_sha256=embedding_semantic_sha256,
            preprocessing_policy_version=preprocessing_policy_version,
            preprocessing_semantic_sha256=preprocessing_semantic_sha256,
            distance_metric=VectorDistanceMetric.COSINE,
            index_kind=index_kind,
            index_parameters=index_parameters,
            projection_version=CANONICAL_VECTOR_POLICY_PROJECTION_VERSION,
            production_eligible=False,
        )
        digest = semantic_sha256(canonical_vector_projection_policy_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{CANONICAL_VECTOR_POLICY_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @field_validator("index_parameters")
    @classmethod
    def validate_index_parameters(
        cls,
        value: tuple[CanonicalVectorIndexParameter, ...],
    ) -> tuple[CanonicalVectorIndexParameter, ...]:
        names = tuple(parameter.name for parameter in value)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("index parameters must have unique canonical names")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.index_kind is VectorIndexKind.NONE and self.index_parameters:
            raise ValueError("NONE index kind cannot carry physical index parameters")
        digest = semantic_sha256(canonical_vector_projection_policy_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("vector projection policy semantic identity is inconsistent")
        if self.logical_key != f"{CANONICAL_VECTOR_POLICY_KEY_NAMESPACE}:{digest}":
            raise ValueError("vector projection policy logical key is inconsistent")
        return self


def canonical_vector_projection_policy_projection(
    policy: CanonicalVectorProjectionPolicy,
) -> dict[str, object]:
    """Return the identity-bearing representation of an internal P14 policy."""

    return {
        "semantic_projection_version": policy.projection_version,
        "policy_version": policy.policy_version,
        "embedding_id": policy.embedding_id,
        "embedding_semantic_sha256": policy.embedding_semantic_sha256,
        "preprocessing_policy_version": policy.preprocessing_policy_version,
        "preprocessing_semantic_sha256": policy.preprocessing_semantic_sha256,
        "distance_metric": policy.distance_metric.value,
        "index_kind": policy.index_kind.value,
        "index_parameters": [item.model_dump(mode="json") for item in policy.index_parameters],
        "production_eligible": policy.production_eligible,
    }


class CanonicalVectorProjectionIntent(StrictModel):
    """A frozen optional projection request bound to an event revision digest."""

    event_revision_id: NonEmptyString
    event_revision_semantic_sha256: Sha256Digest
    policy: CanonicalVectorProjectionPolicy
    encoded_embedding: EncodedEmbedding
    retention: VectorRetentionPolicy
    access_policy: VectorAccessPolicy | None = None
    requested_at: Rfc3339Timestamp | None = None
    projection_version: Literal["canonical-vector-intent-semantic-v1"] = (
        CANONICAL_VECTOR_INTENT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        event_revision_id: str,
        event_revision_semantic_sha256: str,
        policy: CanonicalVectorProjectionPolicy,
        encoded_embedding: EncodedEmbedding,
        retention: VectorRetentionPolicy,
        access_policy: VectorAccessPolicy | None = None,
        requested_at: str | None = None,
    ) -> Self:
        """Freeze one dispatch intent, excluding wall-clock enqueue observation."""

        values: dict[str, object] = {
            "event_revision_id": event_revision_id,
            "event_revision_semantic_sha256": event_revision_semantic_sha256,
            "policy": policy,
            "encoded_embedding": encoded_embedding,
            "retention": retention,
            "access_policy": access_policy,
            "requested_at": requested_at,
            "projection_version": CANONICAL_VECTOR_INTENT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{CANONICAL_VECTOR_INTENT_KEY_NAMESPACE}:{'0' * 64}",
            event_revision_id=event_revision_id,
            event_revision_semantic_sha256=event_revision_semantic_sha256,
            policy=policy,
            encoded_embedding=encoded_embedding,
            retention=retention,
            access_policy=access_policy,
            requested_at=requested_at,
            projection_version=CANONICAL_VECTOR_INTENT_PROJECTION_VERSION,
            production_eligible=False,
        )
        digest = semantic_sha256(canonical_vector_projection_intent_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{CANONICAL_VECTOR_INTENT_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_revision_binding(self) -> Self:
        subject = self.encoded_embedding.subject
        if subject.event_revision_id != self.event_revision_id:
            raise ValueError("encoded vector subject must bind the intent event revision")
        if subject.source_semantic_sha256 != self.event_revision_semantic_sha256:
            raise ValueError(
                "encoded vector source digest must bind the event revision semantic digest"
            )
        if self.policy.embedding_id != self.encoded_embedding.embedding.embedding_id:
            raise ValueError("vector projection policy embedding ID differs from encoded embedding")
        if (
            self.policy.embedding_semantic_sha256
            != self.encoded_embedding.embedding.semantic_sha256
        ):
            raise ValueError(
                "vector projection policy embedding digest differs from encoded embedding"
            )
        digest = semantic_sha256(canonical_vector_projection_intent_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("vector projection intent semantic identity is inconsistent")
        if self.logical_key != f"{CANONICAL_VECTOR_INTENT_KEY_NAMESPACE}:{digest}":
            raise ValueError("vector projection intent logical key is inconsistent")
        return self

    def to_request(self) -> VectorProjectionRequest:
        """Build the existing derived-store request without a completion dependency."""

        return self.encoded_embedding.to_projection_request(
            retention=self.retention,
            access_policy=self.access_policy,
            requested_at=self.requested_at,
        )


def canonical_vector_projection_intent_projection(
    intent: CanonicalVectorProjectionIntent,
) -> dict[str, object]:
    """Return the frozen semantic input for an optional async dispatch."""

    return {
        "semantic_projection_version": intent.projection_version,
        "event_revision_id": intent.event_revision_id,
        "event_revision_semantic_sha256": intent.event_revision_semantic_sha256,
        "policy": canonical_vector_projection_policy_projection(intent.policy),
        "policy_semantic_sha256": intent.policy.semantic_sha256,
        "encoded_embedding": intent.encoded_embedding.model_dump(mode="json"),
        "retention": intent.retention.model_dump(mode="json"),
        "access_policy": (
            None if intent.access_policy is None else intent.access_policy.model_dump(mode="json")
        ),
        "production_eligible": intent.production_eligible,
    }


class CanonicalVectorProjectionDispatchStatus(StrEnum):
    """Non-authoritative outcome of attempting optional vector dispatch."""

    QUEUED = "QUEUED"
    FAILED = "FAILED"


class CanonicalVectorProjectionDispatch(StrictModel):
    """Observable dispatch outcome that cannot represent completion authority."""

    intent_logical_key: NodeLogicalKey
    intent_semantic_sha256: Sha256Digest
    idempotency_key: NonEmptyString
    status: CanonicalVectorProjectionDispatchStatus
    receipt: VectorProjectionReceipt | None = None
    error_code: VectorProjectionErrorCode | None = None
    error_message: NonEmptyString | None = None
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is CanonicalVectorProjectionDispatchStatus.QUEUED:
            if (
                self.receipt is None
                or self.error_code is not None
                or self.error_message is not None
            ):
                raise ValueError("queued vector dispatch requires only a projection receipt")
        elif self.receipt is not None or self.error_code is None or self.error_message is None:
            raise ValueError("failed vector dispatch requires an explicit adapter error")
        if self.intent_logical_key.rsplit(":", 1)[-1] != self.intent_semantic_sha256:
            raise ValueError("dispatch intent logical key and digest are inconsistent")
        return self


class CanonicalVectorProjectionBridge:
    """Dispatch optional revision-bound vectors without entering the runner path."""

    def __init__(self, *, store: VectorProjectionStore) -> None:
        if not callable(getattr(store, "enqueue", None)):
            raise TypeError("store must provide an enqueue() vector projection operation")
        self._store = store

    def enqueue(
        self,
        intent: CanonicalVectorProjectionIntent,
    ) -> CanonicalVectorProjectionDispatch:
        """Try asynchronous enqueue and retain adapter failure as optional state."""

        if not isinstance(intent, CanonicalVectorProjectionIntent):
            raise TypeError("intent must be a CanonicalVectorProjectionIntent")
        request = intent.to_request()
        try:
            receipt = self._store.enqueue(request)
        except VectorProjectionError as error:
            return CanonicalVectorProjectionDispatch(
                intent_logical_key=intent.logical_key,
                intent_semantic_sha256=intent.semantic_sha256,
                idempotency_key=request.idempotency_key,
                status=CanonicalVectorProjectionDispatchStatus.FAILED,
                error_code=error.code,
                error_message=str(error),
            )
        except Exception as error:
            return CanonicalVectorProjectionDispatch(
                intent_logical_key=intent.logical_key,
                intent_semantic_sha256=intent.semantic_sha256,
                idempotency_key=request.idempotency_key,
                status=CanonicalVectorProjectionDispatchStatus.FAILED,
                error_code=VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
                error_message=str(error) or type(error).__name__,
            )
        return CanonicalVectorProjectionDispatch(
            intent_logical_key=intent.logical_key,
            intent_semantic_sha256=intent.semantic_sha256,
            idempotency_key=request.idempotency_key,
            status=CanonicalVectorProjectionDispatchStatus.QUEUED,
            receipt=receipt,
        )


__all__ = [
    "CANONICAL_VECTOR_INTENT_KEY_NAMESPACE",
    "CANONICAL_VECTOR_INTENT_PROJECTION_VERSION",
    "CANONICAL_VECTOR_POLICY_KEY_NAMESPACE",
    "CANONICAL_VECTOR_POLICY_PROJECTION_VERSION",
    "CanonicalVectorIndexParameter",
    "CanonicalVectorProjectionBridge",
    "CanonicalVectorProjectionDispatch",
    "CanonicalVectorProjectionDispatchStatus",
    "CanonicalVectorProjectionIntent",
    "CanonicalVectorProjectionPolicy",
    "VectorDistanceMetric",
    "canonical_vector_projection_intent_projection",
    "canonical_vector_projection_policy_projection",
]
