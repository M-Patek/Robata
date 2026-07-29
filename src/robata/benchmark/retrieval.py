"""Local qualification helpers for the optional retrieval projection.

The structured :class:`robata.retrieval.EventIndex` is authoritative.  These
helpers only measure the asynchronous vector projection and deliberately mark
database/RLS/ANN/cost claims as local evidence until a real Supabase/Postgres
run is supplied.  They are dependency-free so replay and failure tests can run
without a cloud SDK or an encoder.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class RetrievalEvidenceClass(StrEnum):
    """Evidence labels used by local retrieval reports.

    A string subclass keeps this module independent of the larger qualification
    model while accepting the same values as the governance evidence boundary.
    """

    LOCAL_CONFORMANCE = "LOCAL_CONFORMANCE"
    LOCAL_BENCHMARK = "LOCAL_BENCHMARK"
    NOT_MEASURED = "NOT_MEASURED"


class RetrievalBackfillCounters(StrictModel):
    """Idempotent backfill outcomes, split from authoritative QA completion."""

    requested: NonNegativeInt = 0
    enqueued: NonNegativeInt = 0
    completed: NonNegativeInt = 0
    duplicate: NonNegativeInt = 0
    failed: NonNegativeInt = 0
    deferred: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        if self.enqueued > self.requested:
            raise ValueError("enqueued work cannot exceed requested work")
        if self.duplicate > self.requested:
            raise ValueError("duplicate work cannot exceed requested work")
        if self.completed + self.failed + self.deferred > self.enqueued:
            raise ValueError("backfill outcomes cannot exceed enqueued work")
        return self


class RetrievalFilterProfile(StrictModel):
    """Selectivity counters proving structured filtering remains first."""

    query_count: NonNegativeInt = 0
    structured_candidate_count: NonNegativeInt = 0
    vector_candidate_count: NonNegativeInt = 0
    result_count: NonNegativeInt = 0
    vector_selectivity: UnitInterval = 0.0
    result_selectivity: UnitInterval = 0.0
    structured_first: Literal[True] = True

    @model_validator(mode="after")
    def validate_selectivity(self) -> Self:
        if self.vector_candidate_count > self.structured_candidate_count:
            raise ValueError("vector candidates cannot exceed structured candidates")
        if self.result_count > self.vector_candidate_count:
            raise ValueError("results cannot exceed vector candidates")
        if self.structured_candidate_count == 0:
            if self.vector_selectivity != 0.0 or self.result_selectivity != 0.0:
                raise ValueError("zero structured candidates require zero selectivity")
        else:
            vector = self.vector_candidate_count / self.structured_candidate_count
            result = self.result_count / self.structured_candidate_count
            if not math.isclose(self.vector_selectivity, vector, abs_tol=1e-12):
                raise ValueError("vector_selectivity does not match counters")
            if not math.isclose(self.result_selectivity, result, abs_tol=1e-12):
                raise ValueError("result_selectivity does not match counters")
        return self


def build_filter_profile(
    observations: Sequence[RetrievalFilterObservation],
) -> RetrievalFilterProfile:
    """Aggregate structured-first filter observations deterministically."""

    if len({item.query_id for item in observations}) != len(observations):
        raise ValueError("filter observation query IDs must be unique")
    structured = sum(item.structured_count for item in observations)
    vector = sum(item.vector_count for item in observations)
    results = sum(item.final_count for item in observations)
    denominator = structured or 1
    return RetrievalFilterProfile(
        query_count=len(observations),
        structured_candidate_count=structured,
        vector_candidate_count=vector,
        result_count=results,
        vector_selectivity=vector / denominator if structured else 0.0,
        result_selectivity=results / denominator if structured else 0.0,
    )


class AsyncRetrievalBackfillTracker:
    """Local idempotent tracker for optional embedding backfill work."""

    def __init__(self) -> None:
        self._completed: set[str] = set()

    @property
    def completed_identities(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed))

    async def run(
        self,
        identities: Iterable[str],
        *,
        worker: Callable[[str], bool | Awaitable[bool]],
        concurrency: int = 4,
        fail_fast: bool = False,
    ) -> RetrievalBackfillCounters:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
            raise ValueError("concurrency must be a positive integer")
        requested = 0
        unique: list[str] = []
        seen: set[str] = set()
        duplicate = 0
        for identity in identities:
            requested += 1
            if not isinstance(identity, str) or not identity:
                raise ValueError("backfill identities must be non-empty strings")
            if identity in seen or identity in self._completed:
                duplicate += 1
                continue
            seen.add(identity)
            unique.append(identity)
        semaphore = asyncio.Semaphore(concurrency)

        async def one(identity: str) -> tuple[str, bool | None]:
            try:
                async with semaphore:
                    result = worker(identity)
                    result = await result if inspect.isawaitable(result) else result
                    if not isinstance(result, bool):
                        raise ValueError("backfill worker must return bool")
                    return identity, result
            except Exception:
                if fail_fast:
                    raise
                return identity, None

        outcomes = tuple(await asyncio.gather(*(one(item) for item in unique)))
        completed = sum(result is True for _, result in outcomes)
        deferred = sum(result is False for _, result in outcomes)
        failed = sum(result is None for _, result in outcomes)
        for identity, result in outcomes:
            if result is True:
                self._completed.add(identity)
        return RetrievalBackfillCounters(
            requested=requested,
            enqueued=len(unique),
            completed=completed,
            duplicate=duplicate,
            failed=failed,
            deferred=deferred,
        )


async def run_async_backfill(
    identities: Iterable[str],
    *,
    worker: Callable[[str], bool | Awaitable[bool]],
    tracker: AsyncRetrievalBackfillTracker | None = None,
    concurrency: int = 4,
    fail_fast: bool = False,
) -> RetrievalBackfillCounters:
    """Convenience wrapper around :class:`AsyncRetrievalBackfillTracker`."""

    active = tracker or AsyncRetrievalBackfillTracker()
    return await active.run(identities, worker=worker, concurrency=concurrency, fail_fast=fail_fast)


class RetrievalLatencyProfile(StrictModel):
    """Deterministic nearest-rank latency summary in milliseconds."""

    sample_count: NonNegativeInt = 0
    p50_ms: NonNegativeFloat = 0.0
    p95_ms: NonNegativeFloat = 0.0
    p99_ms: NonNegativeFloat = 0.0

    @classmethod
    def from_samples(cls, samples_ms: Iterable[float]) -> Self:
        raw_values = tuple(samples_ms)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values
        ):
            raise ValueError("latency samples must be numeric")
        values = sorted(float(value) for value in raw_values)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("latency samples must be finite and non-negative")
        if not values:
            return cls()

        def nearest(percentile: float) -> float:
            # Nearest-rank: ceil(p*n)-1, with a one-based rank.
            index = max(0, min(len(values) - 1, math.ceil(percentile * len(values)) - 1))
            return values[index]

        return cls(
            sample_count=len(values),
            p50_ms=nearest(0.50),
            p95_ms=nearest(0.95),
            p99_ms=nearest(0.99),
        )


class RetrievalRecallProfile(StrictModel):
    """Recall-at-k for vector candidates after structured filtering."""

    query_count: NonNegativeInt = 0
    relevant_count: NonNegativeInt = 0
    recall_at_k: dict[str, UnitInterval] = Field(default_factory=dict)
    filtered_candidate_count: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_recall(self) -> Self:
        if any(
            not key.isdigit() or int(key) < 1 or str(int(key)) != key for key in self.recall_at_k
        ):
            raise ValueError("recall_at_k keys must be canonical positive decimal k values")
        return self


class RetrievalCostProfile(StrictModel):
    """Counters needed to report retrieval cost without claiming cloud pricing."""

    structured_queries: NonNegativeInt = 0
    vector_queries: NonNegativeInt = 0
    vector_rows_scanned: NonNegativeInt = 0
    encoder_calls: NonNegativeInt = 0
    provider_tokens: NonNegativeInt = 0
    cpu_ms: NonNegativeFloat = 0.0
    estimated_cost: NonNegativeFloat = 0.0


class RetrievalCostMetrics(RetrievalCostProfile):
    """Cost counters with transparent formula fields for local reports."""

    embedding_items: NonNegativeInt = 0
    embedding_calls: NonNegativeInt = 0
    embedding_input_tokens: NonNegativeInt = 0
    vector_writes: NonNegativeInt = 0
    vector_searches: NonNegativeInt = 0
    vector_storage_bytes: NonNegativeInt = 0
    encoder_usd_per_1k_tokens: NonNegativeFloat = 0.0
    vector_usd_per_1k_operations: NonNegativeFloat = 0.0
    storage_usd_per_gib_month: NonNegativeFloat = 0.0
    estimated_encoder_cost_usd: NonNegativeFloat = 0.0
    estimated_vector_cost_usd: NonNegativeFloat = 0.0
    estimated_storage_cost_usd: NonNegativeFloat = 0.0

    @classmethod
    def create(
        cls,
        *,
        embedding_items: int = 0,
        embedding_calls: int = 0,
        embedding_input_tokens: int = 0,
        vector_writes: int = 0,
        vector_searches: int = 0,
        vector_storage_bytes: int = 0,
        encoder_usd_per_1k_tokens: float = 0.0,
        vector_usd_per_1k_operations: float = 0.0,
        storage_usd_per_gib_month: float = 0.0,
    ) -> Self:
        for value in (
            embedding_items,
            embedding_calls,
            embedding_input_tokens,
            vector_writes,
            vector_searches,
            vector_storage_bytes,
        ):
            if value < 0:
                raise ValueError("cost counters must be non-negative")
        encoder_cost = embedding_input_tokens / 1000.0 * encoder_usd_per_1k_tokens
        vector_cost = (vector_writes + vector_searches) / 1000.0 * vector_usd_per_1k_operations
        storage_cost = vector_storage_bytes / (1024**3) * storage_usd_per_gib_month
        return cls(
            embedding_items=embedding_items,
            embedding_calls=embedding_calls,
            embedding_input_tokens=embedding_input_tokens,
            vector_writes=vector_writes,
            vector_searches=vector_searches,
            vector_storage_bytes=vector_storage_bytes,
            encoder_usd_per_1k_tokens=encoder_usd_per_1k_tokens,
            vector_usd_per_1k_operations=vector_usd_per_1k_operations,
            storage_usd_per_gib_month=storage_usd_per_gib_month,
            estimated_encoder_cost_usd=encoder_cost,
            estimated_vector_cost_usd=vector_cost,
            estimated_storage_cost_usd=storage_cost,
            estimated_cost=encoder_cost + vector_cost + storage_cost,
            encoder_calls=embedding_calls,
        )


class RetrievalLatencySummary(RetrievalLatencyProfile):
    """Compatibility spelling with nearest-rank percentile semantics."""

    @property
    def count(self) -> int:
        return self.sample_count

    @classmethod
    def from_samples(cls, samples_ms: Iterable[float]) -> Self:
        raw_values = tuple(samples_ms)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values
        ):
            raise ValueError("latency samples must be numeric")
        values = sorted(float(value) for value in raw_values)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("latency samples must be finite and non-negative")
        if not values:
            return cls()

        def nearest(percentile: float) -> float:
            import math as _math

            return values[max(0, min(len(values) - 1, _math.ceil(percentile * len(values)) - 1))]

        return cls(
            sample_count=len(values),
            p50_ms=nearest(0.50),
            p95_ms=nearest(0.95),
            p99_ms=nearest(0.99),
        )


class BackfillWriteDisposition(StrEnum):
    """Outcome returned by an idempotent vector sink."""

    WRITTEN = "WRITTEN"
    REUSED = "REUSED"
    FAILED = "FAILED"


RETRIEVAL_BACKFILL_TARGET_KEY_VERSION = "retrieval-backfill-target-v2"


class RetrievalBackfillTarget(StrictModel):
    """Stable event-revision/artifact identity used by a backfill job."""

    event_revision_id: str = Field(min_length=1, strict=True)
    artifact_identity: str | None = Field(default=None, min_length=1, strict=True)

    @property
    def identity_key(self) -> str:
        """Return an injective, versioned identity for backfill dedupe."""

        return semantic_sha256(
            {
                "key_version": RETRIEVAL_BACKFILL_TARGET_KEY_VERSION,
                "event_revision_id": self.event_revision_id,
                "artifact_identity": self.artifact_identity,
            }
        )


class RetrievalEmbeddingWrite(StrictModel):
    """Encoder output handed to a vector sink."""

    target: RetrievalBackfillTarget
    vector: tuple[float, ...] = Field(min_length=1)
    encoder_name: str = Field(min_length=1, strict=True)
    model_version: str = Field(min_length=1, strict=True)
    dimension: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def validate_vector(self) -> Self:
        if len(self.vector) != self.dimension:
            raise ValueError("encoded vector dimension does not match dimension")
        if any(not math.isfinite(value) for value in self.vector):
            raise ValueError("encoded vector values must be finite")
        return self


class RetrievalBackfillObservation(StrictModel):
    target: RetrievalBackfillTarget
    status: Literal["WRITTEN", "REUSED", "FAILED"]
    error: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if self.status == "FAILED" and not self.error:
            raise ValueError("FAILED backfill observations require an error")
        if self.status != "FAILED" and self.error is not None:
            raise ValueError("only FAILED backfill observations may carry an error")
        return self


class RetrievalBackfillReport(StrictModel):
    target_count: NonNegativeInt
    unique_target_count: NonNegativeInt
    duplicate_target_count: NonNegativeInt
    written_count: NonNegativeInt = 0
    reused_count: NonNegativeInt = 0
    failed_count: NonNegativeInt = 0
    observations: tuple[RetrievalBackfillObservation, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.unique_target_count + self.duplicate_target_count != self.target_count:
            raise ValueError("backfill target counts are inconsistent")
        if self.written_count + self.reused_count + self.failed_count != self.unique_target_count:
            raise ValueError("backfill outcome counts are inconsistent")
        if self.observations:
            if len(self.observations) != self.unique_target_count:
                raise ValueError("backfill observations must cover each unique target")
            identities = tuple(item.target.identity_key for item in self.observations)
            if len(set(identities)) != len(identities):
                raise ValueError("backfill observations must contain unique targets")
            observed = {
                "WRITTEN": sum(item.status == "WRITTEN" for item in self.observations),
                "REUSED": sum(item.status == "REUSED" for item in self.observations),
                "FAILED": sum(item.status == "FAILED" for item in self.observations),
            }
            if (
                observed["WRITTEN"] != self.written_count
                or observed["REUSED"] != self.reused_count
                or observed["FAILED"] != self.failed_count
            ):
                raise ValueError("backfill outcome counts do not match observations")
        return self


async def run_embedding_backfill(
    targets: Sequence[RetrievalBackfillTarget],
    *,
    encoder: Callable[[RetrievalBackfillTarget], Awaitable[Sequence[float]] | Sequence[float]],
    writer: Callable[
        [RetrievalEmbeddingWrite],
        Awaitable[BackfillWriteDisposition | bool | None] | BackfillWriteDisposition | bool | None,
    ],
    encoder_name: str,
    model_version: str,
    dimension: int,
    concurrency: int = 1,
) -> RetrievalBackfillReport:
    """Encode and enqueue targets with bounded concurrency and replay-safe dedupe."""

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or dimension < 1
        or concurrency < 1
    ):
        raise ValueError("dimension and concurrency must be positive integers")
    ordered: list[RetrievalBackfillTarget] = []
    seen: set[str] = set()
    duplicate_count = 0
    for target in targets:
        if not isinstance(target, RetrievalBackfillTarget):
            target = RetrievalBackfillTarget.model_validate(target, strict=True)
        if target.identity_key in seen:
            duplicate_count += 1
            continue
        seen.add(target.identity_key)
        ordered.append(target)

    semaphore = asyncio.Semaphore(concurrency)

    async def invoke(value: Any, *args: Any) -> Any:
        result = value(*args)
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            return await result
        return result

    async def one(target: RetrievalBackfillTarget) -> RetrievalBackfillObservation:
        async with semaphore:
            try:
                encoded = await invoke(encoder, target)
                vector = tuple(float(item) for item in encoded)
                write = RetrievalEmbeddingWrite(
                    target=target,
                    vector=vector,
                    encoder_name=encoder_name,
                    model_version=model_version,
                    dimension=dimension,
                )
                disposition = await invoke(writer, write)
                status: Literal["WRITTEN", "REUSED", "FAILED"]
                if disposition is BackfillWriteDisposition.REUSED or disposition == "REUSED":
                    status = "REUSED"
                elif (
                    disposition is BackfillWriteDisposition.WRITTEN
                    or disposition == "WRITTEN"
                    or disposition is True
                ):
                    status = "WRITTEN"
                elif (
                    disposition is BackfillWriteDisposition.FAILED
                    or disposition == "FAILED"
                    or disposition is False
                    or disposition is None
                ):
                    status = "FAILED"
                else:
                    raise ValueError("writer returned an unknown disposition")
                return RetrievalBackfillObservation(
                    target=target,
                    status=status,
                    error=(
                        "writer returned no successful disposition" if status == "FAILED" else None
                    ),
                )
            except Exception as exc:  # failure is recorded; other targets continue
                return RetrievalBackfillObservation(
                    target=target,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )

    observations = tuple(await asyncio.gather(*(one(target) for target in ordered)))
    return RetrievalBackfillReport(
        target_count=len(targets),
        unique_target_count=len(ordered),
        duplicate_target_count=duplicate_count,
        written_count=sum(item.status == "WRITTEN" for item in observations),
        reused_count=sum(item.status == "REUSED" for item in observations),
        failed_count=sum(item.status == "FAILED" for item in observations),
        observations=observations,
    )


class RetrievalFilterObservation(StrictModel):
    """One query's structured-first candidate and final result sets.

    The ID form is preferred; count fields are accepted for compact benchmark
    callers and are converted into equivalent counters without inventing IDs.
    """

    query_id: str = Field(min_length=1, strict=True)
    structured_ids: tuple[str, ...] = ()
    vector_ids: tuple[str, ...] = ()
    result_ids: tuple[str, ...] = ()
    structured_candidate_count: NonNegativeInt | None = None
    vector_candidate_count: NonNegativeInt | None = None
    result_count: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        structured = set(self.structured_ids)
        vector = set(self.vector_ids)
        results = set(self.result_ids)
        structured_count = (
            len(self.structured_ids)
            if self.structured_candidate_count is None
            else self.structured_candidate_count
        )
        vector_count = (
            len(self.vector_ids)
            if self.vector_candidate_count is None
            else self.vector_candidate_count
        )
        result_count = len(self.result_ids) if self.result_count is None else self.result_count
        for name, values in (
            ("structured", self.structured_ids),
            ("vector", self.vector_ids),
            ("result", self.result_ids),
        ):
            if values and (
                tuple(values) != tuple(sorted(values)) or len(set(values)) != len(values)
            ):
                raise ValueError(f"{name} IDs must be sorted and unique")
        if (
            self.structured_ids
            and self.structured_candidate_count is not None
            and structured_count != len(self.structured_ids)
        ):
            raise ValueError("structured_candidate_count does not match structured IDs")
        if (
            self.vector_ids
            and self.vector_candidate_count is not None
            and vector_count != len(self.vector_ids)
        ):
            raise ValueError("vector_candidate_count does not match vector IDs")
        if (
            self.result_ids
            and self.result_count is not None
            and result_count != len(self.result_ids)
        ):
            raise ValueError("result_count does not match result IDs")
        if self.vector_ids and not vector.issubset(structured):
            raise ValueError("vector candidates must be drawn from structured candidates")
        if self.result_ids and not results.issubset(vector | structured):
            raise ValueError("results must be drawn from structured candidates")
        if vector_count > structured_count:
            raise ValueError("vector candidates cannot exceed structured candidates")
        if result_count > vector_count:
            raise ValueError("results cannot exceed vector candidates")
        return self

    @property
    def structured_first(self) -> bool:
        return True

    @property
    def vector_selectivity(self) -> float:
        structured_count = (
            len(self.structured_ids)
            if self.structured_candidate_count is None
            else self.structured_candidate_count
        )
        vector_count = (
            len(self.vector_ids)
            if self.vector_candidate_count is None
            else self.vector_candidate_count
        )
        return vector_count / structured_count if structured_count else 0.0

    @property
    def structured_count(self) -> int:
        return (
            len(self.structured_ids)
            if self.structured_candidate_count is None
            else self.structured_candidate_count
        )

    @property
    def vector_count(self) -> int:
        return (
            len(self.vector_ids)
            if self.vector_candidate_count is None
            else self.vector_candidate_count
        )

    @property
    def final_count(self) -> int:
        return len(self.result_ids) if self.result_count is None else self.result_count


class RetrievalFilterMetrics(StrictModel):
    query_count: NonNegativeInt = 0
    structured_first: Literal[True] = True
    vector_selectivity: UnitInterval = 0.0
    structured_candidate_count: NonNegativeInt = 0
    vector_candidate_count: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_selectivity(self) -> Self:
        if self.vector_candidate_count > self.structured_candidate_count:
            raise ValueError("vector candidates cannot exceed structured candidates")
        expected = (
            self.vector_candidate_count / self.structured_candidate_count
            if self.structured_candidate_count
            else 0.0
        )
        if not math.isclose(self.vector_selectivity, expected, abs_tol=1e-12):
            raise ValueError("vector_selectivity does not match counters")
        return self


def build_filter_metrics(
    observations: Sequence[RetrievalFilterObservation],
) -> RetrievalFilterMetrics:
    if len({item.query_id for item in observations}) != len(observations):
        raise ValueError("filter observation query IDs must be unique")
    if not observations:
        return RetrievalFilterMetrics()
    return RetrievalFilterMetrics(
        query_count=len(observations),
        vector_selectivity=(
            sum(item.vector_count for item in observations)
            / sum(item.structured_count for item in observations)
            if sum(item.structured_count for item in observations)
            else 0.0
        ),
        structured_candidate_count=sum(item.structured_count for item in observations),
        vector_candidate_count=sum(item.vector_count for item in observations),
    )


class VectorRecallPoint(StrictModel):
    k: int = Field(strict=True, ge=1)
    recall: UnitInterval
    query_count: NonNegativeInt


def calculate_vector_recall(
    relevant_by_query: Mapping[str, Sequence[str]],
    retrieved_by_query: Mapping[str, Sequence[str]],
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> tuple[VectorRecallPoint, ...]:
    """Calculate global recall@k over relevant identities.

    The first mapping contains the structured/relevant identities; the second
    contains vector-ranked candidates.  Global denominators keep sparse queries
    from silently disappearing from the report.
    """

    raw_cutoffs = tuple(cutoffs)
    if any(isinstance(k, bool) or not isinstance(k, int) for k in raw_cutoffs):
        raise ValueError("cutoffs must be sorted and unique positive integers")
    ks = tuple(sorted(set(raw_cutoffs)))
    if any(k < 1 for k in ks):
        raise ValueError("cutoffs must be sorted and unique positive integers")
    if ks != raw_cutoffs:
        raise ValueError("cutoffs must be sorted and unique")
    query_ids = tuple(sorted(set(relevant_by_query) | set(retrieved_by_query)))
    total_relevant = sum(len(set(relevant_by_query.get(query_id, ()))) for query_id in query_ids)
    points: list[VectorRecallPoint] = []
    for k in ks:
        hits = sum(
            len(
                set(relevant_by_query.get(query_id, ()))
                & set(retrieved_by_query.get(query_id, ())[:k])
            )
            for query_id in query_ids
        )
        points.append(
            VectorRecallPoint(
                k=k,
                recall=hits / total_relevant if total_relevant else 0.0,
                query_count=len(query_ids),
            )
        )
    return tuple(points)


class RetrievalQualificationProfile(StrictModel):
    """Frozen local retrieval profile with an explicit external boundary."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_class: Literal["LOCAL_CONFORMANCE", "LOCAL_BENCHMARK", "NOT_MEASURED"] = (
        "LOCAL_CONFORMANCE"
    )
    structured_authoritative: bool = True
    backfill: RetrievalBackfillCounters = Field(default_factory=RetrievalBackfillCounters)
    recall: RetrievalRecallProfile = Field(default_factory=RetrievalRecallProfile)
    latency: RetrievalLatencyProfile = Field(default_factory=RetrievalLatencyProfile)
    cost: RetrievalCostProfile = Field(default_factory=RetrievalCostProfile)
    filters: RetrievalFilterMetrics = Field(default_factory=RetrievalFilterMetrics)
    workload_fingerprint: Sha256Digest | None = None
    encoder: str | None = None
    model_version: str | None = None
    dimension: NonNegativeInt = 0
    embeddings_non_blocking: bool = True
    production_eligible: bool = False
    external_database_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED"
    profile_digest: Sha256Digest

    @classmethod
    def create(
        cls,
        *,
        evidence_class: Literal["LOCAL_CONFORMANCE", "LOCAL_BENCHMARK", "NOT_MEASURED"] = (
            "LOCAL_CONFORMANCE"
        ),
        backfill: RetrievalBackfillCounters | None = None,
        recall: RetrievalRecallProfile | None = None,
        latency: RetrievalLatencyProfile | None = None,
        cost: RetrievalCostProfile | None = None,
        filters: RetrievalFilterMetrics | None = None,
        workload_fingerprint: Sha256Digest | None = None,
        encoder: str | None = None,
        model_version: str | None = None,
        dimension: int = 0,
        external_database_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED",
    ) -> Self:
        resolved_backfill = backfill or RetrievalBackfillCounters()
        resolved_recall = recall or RetrievalRecallProfile()
        resolved_latency = latency or RetrievalLatencyProfile()
        resolved_cost = cost or RetrievalCostProfile()
        resolved_filters = filters or RetrievalFilterMetrics()
        projection = retrieval_profile_projection(
            evidence_class=evidence_class,
            structured_authoritative=True,
            backfill=resolved_backfill,
            recall=resolved_recall,
            latency=resolved_latency,
            cost=resolved_cost,
            filters=resolved_filters,
            workload_fingerprint=workload_fingerprint,
            encoder=encoder,
            model_version=model_version,
            dimension=dimension,
            embeddings_non_blocking=True,
            production_eligible=False,
            external_database_status=external_database_status,
        )
        return cls(
            schema_version="1.0",
            evidence_class=evidence_class,
            structured_authoritative=True,
            backfill=resolved_backfill,
            recall=resolved_recall,
            latency=resolved_latency,
            cost=resolved_cost,
            filters=resolved_filters,
            workload_fingerprint=workload_fingerprint,
            encoder=encoder,
            model_version=model_version,
            dimension=dimension,
            embeddings_non_blocking=True,
            production_eligible=False,
            external_database_status=external_database_status,
            profile_digest=semantic_sha256(projection),
        )

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if not self.structured_authoritative:
            raise ValueError("structured retrieval must remain authoritative")
        if not self.embeddings_non_blocking:
            raise ValueError("embedding projection must remain non-blocking")
        if self.production_eligible:
            raise ValueError("retrieval projection cannot self-promote production eligibility")
        expected = semantic_sha256(
            retrieval_profile_projection(
                evidence_class=self.evidence_class,
                structured_authoritative=self.structured_authoritative,
                backfill=self.backfill,
                recall=self.recall,
                latency=self.latency,
                cost=self.cost,
                filters=self.filters,
                workload_fingerprint=self.workload_fingerprint,
                encoder=self.encoder,
                model_version=self.model_version,
                dimension=self.dimension,
                embeddings_non_blocking=self.embeddings_non_blocking,
                production_eligible=self.production_eligible,
                external_database_status=self.external_database_status,
            )
        )
        if self.profile_digest != expected:
            raise ValueError("profile_digest does not match retrieval profile")
        return self


def retrieval_profile_projection(
    *,
    evidence_class: str,
    structured_authoritative: bool,
    backfill: RetrievalBackfillCounters,
    recall: RetrievalRecallProfile,
    latency: RetrievalLatencyProfile,
    cost: RetrievalCostProfile,
    filters: RetrievalFilterMetrics | None = None,
    workload_fingerprint: str | None = None,
    encoder: str | None = None,
    model_version: str | None = None,
    dimension: int = 0,
    embeddings_non_blocking: bool = True,
    production_eligible: bool = False,
    external_database_status: str = "NOT_MEASURED",
) -> dict[str, object]:
    """Return the digest preimage for a retrieval profile."""

    return {
        "schema_version": "1.0",
        "evidence_class": evidence_class,
        "structured_authoritative": structured_authoritative,
        "backfill": backfill.model_dump(mode="json"),
        "recall": recall.model_dump(mode="json"),
        "latency": latency.model_dump(mode="json"),
        "cost": cost.model_dump(mode="json"),
        "filters": (filters or RetrievalFilterMetrics()).model_dump(mode="json"),
        "workload_fingerprint": workload_fingerprint,
        "encoder": encoder,
        "model_version": model_version,
        "dimension": dimension,
        "embeddings_non_blocking": embeddings_non_blocking,
        "production_eligible": production_eligible,
        "external_database_status": external_database_status,
    }


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    *,
    k: int,
) -> float:
    """Compute deterministic recall@k, treating identities as sets."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    expected = set(relevant)
    if not expected:
        return 0.0
    return len(set(retrieved[:k]) & expected) / len(expected)


def build_recall_profile(
    observations: Sequence[tuple[Sequence[str], Sequence[str]]],
    *,
    ks: Sequence[int] = (1, 5, 10),
    filtered_candidate_count: int = 0,
) -> RetrievalRecallProfile:
    """Build recall metrics from ``(retrieved, relevant)`` query pairs."""

    raw_ks = tuple(ks)
    if any(isinstance(k, bool) or not isinstance(k, int) for k in raw_ks):
        raise ValueError("ks must be positive integers")
    normalized_ks = tuple(sorted(set(raw_ks)))
    if any(k < 1 for k in normalized_ks):
        raise ValueError("ks must be positive")
    if normalized_ks != raw_ks:
        raise ValueError("ks must be sorted and unique")
    values = {
        str(k): (
            sum(recall_at_k(retrieved, relevant, k=k) for retrieved, relevant in observations)
            / len(observations)
            if observations
            else 0.0
        )
        for k in normalized_ks
    }
    return RetrievalRecallProfile(
        query_count=len(observations),
        relevant_count=sum(len(set(relevant)) for _, relevant in observations),
        recall_at_k=values,
        filtered_candidate_count=filtered_candidate_count,
    )


def build_retrieval_profile(
    *,
    backfill: RetrievalBackfillCounters | RetrievalBackfillReport | None = None,
    observations: Sequence[tuple[Sequence[str], Sequence[str]]] = (),
    latency_samples_ms: Iterable[float] = (),
    latency: RetrievalLatencyProfile | None = None,
    cost: RetrievalCostProfile | None = None,
    filters: RetrievalFilterMetrics | None = None,
    recall: Sequence[VectorRecallPoint] | RetrievalRecallProfile | None = None,
    workload_fingerprint: Sha256Digest | None = None,
    encoder: str | None = None,
    model_version: str | None = None,
    dimension: int = 0,
    evidence_class: Literal["LOCAL_CONFORMANCE", "LOCAL_BENCHMARK", "NOT_MEASURED"] = (
        "LOCAL_CONFORMANCE"
    ),
    external_database_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED",
    ks: Sequence[int] = (1, 5, 10),
    filtered_candidate_count: int = 0,
) -> RetrievalQualificationProfile:
    """Build a digest-bound local profile from replayable observations."""

    if isinstance(backfill, RetrievalBackfillReport):
        backfill_counts = RetrievalBackfillCounters(
            requested=backfill.target_count,
            enqueued=backfill.unique_target_count,
            # A reused row is a successful completion of this backfill request
            # even though it did not issue a new vector write. Keep ``completed``
            # aligned with the unique target outcomes; ``duplicate`` retains replay signal.
            completed=backfill.written_count + backfill.reused_count,
            duplicate=backfill.reused_count + backfill.duplicate_target_count,
            failed=backfill.failed_count,
        )
    else:
        backfill_counts = backfill or RetrievalBackfillCounters()
    if isinstance(recall, RetrievalRecallProfile):
        recall_profile = recall
    elif recall is not None:
        recall_profile = RetrievalRecallProfile(
            query_count=recall[0].query_count if recall else 0,
            relevant_count=0,
            recall_at_k={str(item.k): item.recall for item in recall},
            filtered_candidate_count=filtered_candidate_count,
        )
    else:
        recall_profile = build_recall_profile(
            observations,
            ks=ks,
            filtered_candidate_count=filtered_candidate_count,
        )
    return RetrievalQualificationProfile.create(
        evidence_class=evidence_class,
        backfill=backfill_counts,
        recall=recall_profile,
        latency=latency or RetrievalLatencyProfile.from_samples(latency_samples_ms),
        cost=cost,
        filters=filters,
        workload_fingerprint=workload_fingerprint,
        encoder=encoder,
        model_version=model_version,
        dimension=dimension,
        external_database_status=external_database_status,
    )


# Short aliases make the profile discoverable from benchmark and retrieval callers.
RetrievalProfile = RetrievalQualificationProfile
RetrievalBackfill = RetrievalBackfillCounters


__all__ = [
    "RETRIEVAL_BACKFILL_TARGET_KEY_VERSION",
    "AsyncRetrievalBackfillTracker",
    "BackfillWriteDisposition",
    "RetrievalBackfill",
    "RetrievalBackfillCounters",
    "RetrievalBackfillReport",
    "RetrievalBackfillTarget",
    "RetrievalCostMetrics",
    "RetrievalCostProfile",
    "RetrievalEmbeddingWrite",
    "RetrievalEvidenceClass",
    "RetrievalFilterMetrics",
    "RetrievalFilterObservation",
    "RetrievalFilterProfile",
    "RetrievalLatencyProfile",
    "RetrievalLatencySummary",
    "RetrievalProfile",
    "RetrievalQualificationProfile",
    "RetrievalRecallProfile",
    "VectorRecallPoint",
    "build_filter_metrics",
    "build_filter_profile",
    "build_recall_profile",
    "build_retrieval_profile",
    "calculate_vector_recall",
    "recall_at_k",
    "retrieval_profile_projection",
    "run_async_backfill",
    "run_embedding_backfill",
]
