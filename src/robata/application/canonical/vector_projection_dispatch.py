"""Durable, detached dispatch for optional P14 vector projections.

The structured EventIndex remains the retrieval authority.  This sidecar seals
an already encoded, revision-bound vector intent before attempting the optional
adapter handoff.  A worker persists one immutable handoff record per job, so a
restart retries only jobs whose adapter call was not recorded.  The underlying
adapter still owns physical queue durability, FAILED lifecycle state, and any
real pgvector/RLS implementation; this module does not represent an encoder or
database capability.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, model_validator

from robata.application.canonical.vector_projection import (
    CanonicalVectorProjectionBridge,
    CanonicalVectorProjectionDispatch,
    CanonicalVectorProjectionDispatchStatus,
    CanonicalVectorProjectionIntent,
    canonical_vector_projection_intent_projection,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey
from robata.contracts.retrieval import VectorProjectionReceipt, VectorProjectionStatus
from robata.ports.vector_projection import (
    VectorProjectionError,
    VectorProjectionErrorCode,
    VectorProjectionStore,
)
from robata.tempfiles import make_temp_file

CANONICAL_VECTOR_DISPATCH_JOB_PROJECTION_VERSION: Final = (
    "canonical-vector-dispatch-job-semantic-v1"
)
CANONICAL_VECTOR_DISPATCH_JOB_KEY_NAMESPACE: Final = "canonical-vector-dispatch-job-v1"
CANONICAL_VECTOR_DISPATCH_RECORD_PROJECTION_VERSION: Final = (
    "canonical-vector-dispatch-record-semantic-v1"
)
CANONICAL_VECTOR_DISPATCH_RECORD_KEY_NAMESPACE: Final = "canonical-vector-dispatch-record-v1"

RetryOrdinal = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]

# These failures describe an unavailable optional handoff rather than an
# immutable input/lineage conflict.  A retry is always a new job and never
# mutates the original handoff evidence.
_RETRYABLE_DISPATCH_ERRORS: Final = frozenset(
    {
        VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
        VectorProjectionErrorCode.INDEX_UNAVAILABLE,
        VectorProjectionErrorCode.RETRYABLE,
    }
)


class VectorProjectionDispatchError(RuntimeError):
    """Base error for the detached vector dispatch sidecar."""


class VectorProjectionDispatchConflict(VectorProjectionDispatchError):
    """An immutable job path contains unrelated canonical bytes."""


class VectorProjectionDispatchStorageError(VectorProjectionDispatchError):
    """A sidecar path is malformed, noncanonical, or cannot be persisted."""


class VectorProjectionDispatchRetryNotAllowed(VectorProjectionDispatchError):
    """A recorded failure is not a safe retry candidate."""


class CanonicalVectorProjectionJobStatus(StrEnum):
    """Outcome of sealing an immutable optional-vector handoff job."""

    ENQUEUED = "ENQUEUED"
    REPLAYED = "REPLAYED"


class CanonicalVectorProjectionDispatchExecutionStatus(StrEnum):
    """Outcome of recording one worker handoff observation."""

    RECORDED = "RECORDED"
    REPLAYED = "REPLAYED"


class CanonicalVectorProjectionDispatchJob(StrictModel):
    """One immutable, non-authoritative vector adapter handoff request.

    ``intent.requested_at`` is deliberately normalized to ``None`` before the
    job is sealed.  It is a wall-clock enqueue observation, not part of the
    revision-bound intent identity, and keeping it would make two safe
    recoveries write divergent bytes under one semantic job path.
    """

    schema_version: Literal["1.0"] = "1.0"
    intent: CanonicalVectorProjectionIntent
    retry_ordinal: RetryOrdinal = 0
    retry_of_job_logical_key: NodeLogicalKey | None = None
    retry_of_job_semantic_sha256: Sha256Digest | None = None
    retry_of_dispatch_semantic_sha256: Sha256Digest | None = None
    projection_version: Literal["canonical-vector-dispatch-job-semantic-v1"] = (
        CANONICAL_VECTOR_DISPATCH_JOB_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(cls, *, intent: CanonicalVectorProjectionIntent) -> Self:
        """Seal a first adapter handoff without invoking an adapter."""

        return cls._create(
            intent=_frozen_intent(intent),
            retry_ordinal=0,
            retry_of_job_logical_key=None,
            retry_of_job_semantic_sha256=None,
            retry_of_dispatch_semantic_sha256=None,
        )

    @classmethod
    def create_retry(
        cls,
        *,
        prior_job: CanonicalVectorProjectionDispatchJob,
        prior_record: CanonicalVectorProjectionDispatchRecord,
    ) -> Self:
        """Create a new immutable retry job from one recorded retryable failure."""

        checked_job = _require_job(prior_job)
        checked_record = _require_record(prior_record)
        _validate_record_for_job(checked_record, checked_job)
        dispatch = checked_record.dispatch
        if dispatch.status is not CanonicalVectorProjectionDispatchStatus.FAILED:
            raise VectorProjectionDispatchRetryNotAllowed(
                "only a FAILED vector dispatch record may be retried"
            )
        if dispatch.error_code not in _RETRYABLE_DISPATCH_ERRORS:
            error_code = "UNKNOWN" if dispatch.error_code is None else dispatch.error_code.value
            raise VectorProjectionDispatchRetryNotAllowed(
                f"vector dispatch failure {error_code} is not retryable"
            )
        return cls._create(
            intent=checked_job.intent,
            retry_ordinal=checked_job.retry_ordinal + 1,
            retry_of_job_logical_key=checked_job.logical_key,
            retry_of_job_semantic_sha256=checked_job.semantic_sha256,
            retry_of_dispatch_semantic_sha256=checked_record.semantic_sha256,
        )

    @classmethod
    def _create(
        cls,
        *,
        intent: CanonicalVectorProjectionIntent,
        retry_ordinal: int,
        retry_of_job_logical_key: NodeLogicalKey | None,
        retry_of_job_semantic_sha256: Sha256Digest | None,
        retry_of_dispatch_semantic_sha256: Sha256Digest | None,
    ) -> Self:
        values: dict[str, object] = {
            "schema_version": "1.0",
            "intent": intent,
            "retry_ordinal": retry_ordinal,
            "retry_of_job_logical_key": retry_of_job_logical_key,
            "retry_of_job_semantic_sha256": retry_of_job_semantic_sha256,
            "retry_of_dispatch_semantic_sha256": retry_of_dispatch_semantic_sha256,
            "projection_version": CANONICAL_VECTOR_DISPATCH_JOB_PROJECTION_VERSION,
            "production_eligible": False,
        }
        digest = semantic_sha256(_job_projection_values(values))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{CANONICAL_VECTOR_DISPATCH_JOB_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @property
    def is_retry(self) -> bool:
        """Whether this job explicitly follows a recorded failed handoff."""

        return self.retry_ordinal > 0

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        retry_references = (
            self.retry_of_job_logical_key,
            self.retry_of_job_semantic_sha256,
            self.retry_of_dispatch_semantic_sha256,
        )
        if self.intent.requested_at is not None:
            raise ValueError("durable vector dispatch intent cannot retain requested_at")
        if self.retry_ordinal == 0:
            if any(reference is not None for reference in retry_references):
                raise ValueError("initial vector dispatch job cannot reference a prior handoff")
        elif any(reference is None for reference in retry_references):
            raise ValueError("retry vector dispatch job requires complete prior handoff linkage")
        if (
            self.retry_of_job_logical_key is not None
            and self.retry_of_job_semantic_sha256 is not None
            and self.retry_of_job_logical_key.rsplit(":", 1)[-1]
            != self.retry_of_job_semantic_sha256
        ):
            raise ValueError("retry vector dispatch job logical key and digest are inconsistent")
        digest = semantic_sha256(canonical_vector_projection_dispatch_job_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("vector dispatch job semantic identity is inconsistent")
        if self.logical_key != f"{CANONICAL_VECTOR_DISPATCH_JOB_KEY_NAMESPACE}:{digest}":
            raise ValueError("vector dispatch job logical key is inconsistent")
        return self


class CanonicalVectorProjectionDispatchRecord(StrictModel):
    """Immutable adapter-handoff outcome for one durable vector dispatch job."""

    schema_version: Literal["1.0"] = "1.0"
    job_logical_key: NodeLogicalKey
    job_semantic_sha256: Sha256Digest
    dispatch: CanonicalVectorProjectionDispatch
    projection_version: Literal["canonical-vector-dispatch-record-semantic-v1"] = (
        CANONICAL_VECTOR_DISPATCH_RECORD_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        job: CanonicalVectorProjectionDispatchJob,
        dispatch: CanonicalVectorProjectionDispatch,
    ) -> Self:
        """Bind an observed handoff outcome to precisely one sealed job."""

        checked_job = _require_job(job)
        checked_dispatch = _require_dispatch(dispatch)
        values: dict[str, object] = {
            "schema_version": "1.0",
            "job_logical_key": checked_job.logical_key,
            "job_semantic_sha256": checked_job.semantic_sha256,
            "dispatch": checked_dispatch,
            "projection_version": CANONICAL_VECTOR_DISPATCH_RECORD_PROJECTION_VERSION,
            "production_eligible": False,
        }
        digest = semantic_sha256(_record_projection_values(values))
        record = cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{CANONICAL_VECTOR_DISPATCH_RECORD_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )
        _validate_record_for_job(record, checked_job)
        return record

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.job_logical_key.rsplit(":", 1)[-1] != self.job_semantic_sha256:
            raise ValueError("vector dispatch record job logical key is inconsistent")
        digest = semantic_sha256(canonical_vector_projection_dispatch_record_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("vector dispatch record semantic identity is inconsistent")
        if self.logical_key != f"{CANONICAL_VECTOR_DISPATCH_RECORD_KEY_NAMESPACE}:{digest}":
            raise ValueError("vector dispatch record logical key is inconsistent")
        return self


class CanonicalVectorProjectionJobDispatch(StrictModel):
    """Result of durably creating or replaying one dispatch job."""

    status: CanonicalVectorProjectionJobStatus
    job: CanonicalVectorProjectionDispatchJob
    replayed: bool

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = (
            CanonicalVectorProjectionJobStatus.REPLAYED
            if self.replayed
            else CanonicalVectorProjectionJobStatus.ENQUEUED
        )
        if self.status is not expected:
            raise ValueError("vector dispatch job status must match replayed")
        return self


class CanonicalVectorProjectionDispatchExecution(StrictModel):
    """One recorded optional adapter handoff, never a completion outcome."""

    status: CanonicalVectorProjectionDispatchExecutionStatus
    job: CanonicalVectorProjectionDispatchJob
    record: CanonicalVectorProjectionDispatchRecord
    replayed: bool

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = (
            CanonicalVectorProjectionDispatchExecutionStatus.REPLAYED
            if self.replayed
            else CanonicalVectorProjectionDispatchExecutionStatus.RECORDED
        )
        if self.status is not expected:
            raise ValueError("vector dispatch execution status must match replayed")
        _validate_record_for_job(self.record, self.job)
        return self


class CanonicalVectorProjectionDispatchStore:
    """Exact-byte append-only store for jobs and one handoff record per job.

    A record path is keyed by the job digest rather than the record digest.  A
    worker race can observe duplicate acknowledgements that differ only in
    adapter observation details; the first valid immutable record for the job
    remains the durable handoff evidence, while every later worker replays it.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).absolute()
        _ensure_regular_directory(self._root, "vector dispatch sidecar root")
        _ensure_regular_directory(self._jobs_directory, "vector dispatch job directory")
        _ensure_regular_directory(self._records_directory, "vector dispatch record directory")

    @property
    def root(self) -> Path:
        """Return the non-resolved root that owns the sidecar files."""

        return self._root

    @property
    def _jobs_directory(self) -> Path:
        return self._root / "jobs"

    @property
    def _records_directory(self) -> Path:
        return self._root / "records"

    def job_path(self, semantic_sha256: Sha256Digest | str) -> Path:
        """Return the immutable path for one dispatch job identity."""

        return self._jobs_directory / f"{_validate_digest(semantic_sha256)}.json"

    def record_path(self, job_semantic_sha256: Sha256Digest | str) -> Path:
        """Return the immutable one-record path for one dispatch job."""

        return self._records_directory / f"{_validate_digest(job_semantic_sha256)}.json"

    def put_or_get_job(
        self,
        job: CanonicalVectorProjectionDispatchJob,
    ) -> tuple[CanonicalVectorProjectionDispatchJob, bool]:
        """Persist one job once, or replay only byte-identical sealed input."""

        checked = _require_job(job)
        stored, replayed = self._publish_or_read(
            path=self.job_path(checked.semantic_sha256),
            expected=canonical_json_bytes(checked),
            parser=_parse_exact_job,
            label="vector dispatch job",
            require_identical_existing=True,
        )
        if stored != checked:
            raise VectorProjectionDispatchConflict(
                "vector dispatch job path contains different immutable canonical bytes"
            )
        return stored, replayed

    def get_job(
        self,
        semantic_sha256: Sha256Digest | str,
    ) -> CanonicalVectorProjectionDispatchJob | None:
        """Load one job only if canonical bytes and filename both verify."""

        digest = _validate_digest(semantic_sha256)
        path = self.job_path(digest)
        if not path.exists() and not path.is_symlink():
            return None
        job = _parse_exact_job(self._read_exact(path, "vector dispatch job"))
        if job.semantic_sha256 != digest:
            raise VectorProjectionDispatchStorageError(
                "vector dispatch job path does not match its semantic digest"
            )
        return job

    def list_jobs(self) -> tuple[CanonicalVectorProjectionDispatchJob, ...]:
        """Load every durable job in stable semantic-digest order."""

        _ensure_regular_directory(self._jobs_directory, "vector dispatch job directory")
        jobs: list[CanonicalVectorProjectionDispatchJob] = []
        for path in sorted(self._jobs_directory.glob("*.json"), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file():
                raise VectorProjectionDispatchStorageError(
                    "vector dispatch job is not a regular file"
                )
            job = _parse_exact_job(self._read_exact(path, "vector dispatch job"))
            if path != self.job_path(job.semantic_sha256):
                raise VectorProjectionDispatchStorageError(
                    "vector dispatch job path does not match its semantic digest"
                )
            jobs.append(job)
        return tuple(jobs)

    def get_record(
        self,
        job_semantic_sha256: Sha256Digest | str,
    ) -> CanonicalVectorProjectionDispatchRecord | None:
        """Load the single immutable handoff record for a durable job."""

        digest = _validate_digest(job_semantic_sha256)
        path = self.record_path(digest)
        if not path.exists() and not path.is_symlink():
            return None
        record = _parse_exact_record(self._read_exact(path, "vector dispatch record"))
        if record.job_semantic_sha256 != digest:
            raise VectorProjectionDispatchStorageError(
                "vector dispatch record path does not match its job digest"
            )
        job = self.get_job(digest)
        if job is None:
            raise VectorProjectionDispatchStorageError(
                "vector dispatch record does not have a durable job"
            )
        _validate_record_for_job(record, job)
        return record

    def put_or_get_record(
        self,
        *,
        job: CanonicalVectorProjectionDispatchJob,
        record: CanonicalVectorProjectionDispatchRecord,
    ) -> tuple[CanonicalVectorProjectionDispatchRecord, bool]:
        """Persist first valid handoff evidence or replay an earlier observation."""

        checked_job = _require_job(job)
        checked_record = _require_record(record)
        _validate_record_for_job(checked_record, checked_job)
        stored, replayed = self._publish_or_read(
            path=self.record_path(checked_job.semantic_sha256),
            expected=canonical_json_bytes(checked_record),
            parser=_parse_exact_record,
            label="vector dispatch record",
            require_identical_existing=False,
        )
        _validate_record_for_job(stored, checked_job)
        return stored, replayed

    def _publish_or_read[TArtifact](
        self,
        *,
        path: Path,
        expected: bytes,
        parser: Callable[[bytes], TArtifact],
        label: str,
        require_identical_existing: bool,
    ) -> tuple[TArtifact, bool]:
        _ensure_regular_directory(path.parent, f"{label} directory")
        if path.exists() or path.is_symlink():
            actual = self._read_exact(path, label)
            if require_identical_existing and actual != expected:
                raise VectorProjectionDispatchConflict(
                    f"existing {label} contains different immutable canonical bytes"
                )
            return parser(actual), True

        descriptor, temporary = make_temp_file(
            path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                actual = self._read_exact(path, label)
                if require_identical_existing and actual != expected:
                    raise VectorProjectionDispatchConflict(
                        f"concurrent {label} contains different immutable canonical bytes"
                    ) from None
                return parser(actual), True
            return parser(expected), False
        except OSError as error:
            raise VectorProjectionDispatchStorageError(
                f"cannot publish {label}: {error}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_exact(path: Path, label: str) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise VectorProjectionDispatchStorageError(f"{label} is not a regular file")
        try:
            return path.read_bytes()
        except OSError as error:
            raise VectorProjectionDispatchStorageError(f"cannot read {label}: {error}") from error


class CanonicalVectorProjectionDispatchBridge:
    """Seal optional projection work without placing it on the primary path."""

    def __init__(self, store: CanonicalVectorProjectionDispatchStore) -> None:
        if not isinstance(store, CanonicalVectorProjectionDispatchStore):
            raise TypeError("store must be a CanonicalVectorProjectionDispatchStore")
        self._store = store

    def enqueue(
        self,
        intent: CanonicalVectorProjectionIntent,
    ) -> CanonicalVectorProjectionJobDispatch:
        """Durably seal a first handoff request before any adapter call."""

        job = CanonicalVectorProjectionDispatchJob.create(intent=intent)
        stored, replayed = self._store.put_or_get_job(job)
        return CanonicalVectorProjectionJobDispatch(
            status=(
                CanonicalVectorProjectionJobStatus.REPLAYED
                if replayed
                else CanonicalVectorProjectionJobStatus.ENQUEUED
            ),
            job=stored,
            replayed=replayed,
        )

    def retry_failed(
        self,
        job: CanonicalVectorProjectionDispatchJob,
    ) -> CanonicalVectorProjectionJobDispatch:
        """Seal a distinct retry job from a stored retryable failure record."""

        prior_job = _require_job(job)
        stored_prior = self._store.get_job(prior_job.semantic_sha256)
        if stored_prior is None:
            raise VectorProjectionDispatchRetryNotAllowed(
                "cannot retry a vector dispatch job that was never durably stored"
            )
        if stored_prior != prior_job:
            raise VectorProjectionDispatchConflict(
                "stored vector dispatch job differs from requested retry source"
            )
        prior_record = self._store.get_record(prior_job.semantic_sha256)
        if prior_record is None:
            raise VectorProjectionDispatchRetryNotAllowed(
                "cannot retry a vector dispatch job without a stored failure record"
            )
        retry = CanonicalVectorProjectionDispatchJob.create_retry(
            prior_job=prior_job,
            prior_record=prior_record,
        )
        stored, replayed = self._store.put_or_get_job(retry)
        return CanonicalVectorProjectionJobDispatch(
            status=(
                CanonicalVectorProjectionJobStatus.REPLAYED
                if replayed
                else CanonicalVectorProjectionJobStatus.ENQUEUED
            ),
            job=stored,
            replayed=replayed,
        )


class CanonicalVectorProjectionDispatchWorker:
    """Recover sealed vector handoffs without becoming a retrieval authority."""

    def __init__(
        self,
        *,
        store: CanonicalVectorProjectionDispatchStore,
        vector_store: VectorProjectionStore,
    ) -> None:
        if not isinstance(store, CanonicalVectorProjectionDispatchStore):
            raise TypeError("store must be a CanonicalVectorProjectionDispatchStore")
        if not callable(getattr(vector_store, "enqueue", None)) or not callable(
            getattr(vector_store, "retry_failed", None)
        ):
            raise TypeError("vector_store must provide enqueue() and retry_failed()")
        self._store = store
        self._vector_store = vector_store

    def drain(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[CanonicalVectorProjectionDispatchExecution, ...]:
        """Record or replay handoffs for sealed jobs in stable bounded order."""

        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("drain limit must be a non-negative integer")
        jobs = self._store.list_jobs()
        if limit is not None:
            jobs = jobs[:limit]
        return tuple(self.dispatch_job(job) for job in jobs)

    def dispatch_job(
        self,
        job: CanonicalVectorProjectionDispatchJob,
    ) -> CanonicalVectorProjectionDispatchExecution:
        """Perform at-least-once adapter handoff for one sealed job."""

        checked_job = _require_job(job)
        existing = self._store.get_record(checked_job.semantic_sha256)
        if existing is not None:
            return _execution(checked_job, existing, replayed=True)

        dispatch = self._dispatch_to_adapter(checked_job)
        candidate = CanonicalVectorProjectionDispatchRecord.create(
            job=checked_job,
            dispatch=dispatch,
        )
        stored, replayed = self._store.put_or_get_record(job=checked_job, record=candidate)
        return _execution(checked_job, stored, replayed=replayed)

    def _dispatch_to_adapter(
        self,
        job: CanonicalVectorProjectionDispatchJob,
    ) -> CanonicalVectorProjectionDispatch:
        if job.is_retry:
            try:
                receipt = self._vector_store.retry_failed(
                    job.intent.encoded_embedding.subject,
                    job.intent.encoded_embedding.embedding.embedding_id,
                )
            except Exception:
                # A crash can happen after a previous retry already moved a
                # physical row back to PENDING.  enqueue is idempotent for the
                # frozen request, so it is the safe recovery path for every
                # retry_failed() error, including NOT_FOUND and stale state.
                return self._enqueue(job.intent)
            return _accepted_dispatch(job.intent, receipt)
        return self._enqueue(job.intent)

    def _enqueue(
        self,
        intent: CanonicalVectorProjectionIntent,
    ) -> CanonicalVectorProjectionDispatch:
        try:
            dispatch = CanonicalVectorProjectionBridge(store=self._vector_store).enqueue(intent)
            if dispatch.status is CanonicalVectorProjectionDispatchStatus.FAILED:
                return dispatch
            if dispatch.receipt is None:
                return _failed_dispatch(
                    intent,
                    VectorProjectionErrorCode.CONFLICT,
                    "vector adapter accepted a dispatch without a receipt",
                )
            return _accepted_dispatch(intent, dispatch.receipt)
        except VectorProjectionError as error:
            return _failed_dispatch(intent, error.code, str(error))
        except Exception as error:
            return _failed_dispatch(
                intent,
                VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
                str(error) or type(error).__name__,
            )


def canonical_vector_projection_dispatch_job_projection(
    job: CanonicalVectorProjectionDispatchJob,
) -> dict[str, object]:
    """Return the internal identity projection for a durable handoff job."""

    return _job_projection_values(
        {
            "intent": job.intent,
            "retry_ordinal": job.retry_ordinal,
            "retry_of_job_logical_key": job.retry_of_job_logical_key,
            "retry_of_job_semantic_sha256": job.retry_of_job_semantic_sha256,
            "retry_of_dispatch_semantic_sha256": job.retry_of_dispatch_semantic_sha256,
            "projection_version": job.projection_version,
            "production_eligible": job.production_eligible,
        }
    )


def canonical_vector_projection_dispatch_record_projection(
    record: CanonicalVectorProjectionDispatchRecord,
) -> dict[str, object]:
    """Return the identity projection for one immutable adapter observation."""

    return _record_projection_values(
        {
            "job_logical_key": record.job_logical_key,
            "job_semantic_sha256": record.job_semantic_sha256,
            "dispatch": record.dispatch,
            "projection_version": record.projection_version,
            "production_eligible": record.production_eligible,
        }
    )


def _job_projection_values(values: dict[str, object]) -> dict[str, object]:
    intent = values["intent"]
    if not isinstance(intent, CanonicalVectorProjectionIntent):
        raise TypeError("vector dispatch job intent is invalid")
    return {
        "semantic_projection_version": values["projection_version"],
        "intent": canonical_vector_projection_intent_projection(intent),
        "intent_semantic_sha256": intent.semantic_sha256,
        "retry_ordinal": values["retry_ordinal"],
        "retry_of_job_logical_key": values["retry_of_job_logical_key"],
        "retry_of_job_semantic_sha256": values["retry_of_job_semantic_sha256"],
        "retry_of_dispatch_semantic_sha256": values["retry_of_dispatch_semantic_sha256"],
        "production_eligible": values["production_eligible"],
        "identity_scope": "detached-optional-vector-handoff-not-retrieval-or-completion-authority",
    }


def _record_projection_values(values: dict[str, object]) -> dict[str, object]:
    dispatch = values["dispatch"]
    if not isinstance(dispatch, CanonicalVectorProjectionDispatch):
        raise TypeError("vector dispatch record dispatch is invalid")
    return {
        "semantic_projection_version": values["projection_version"],
        "job_logical_key": values["job_logical_key"],
        "job_semantic_sha256": values["job_semantic_sha256"],
        "dispatch": dispatch.model_dump(mode="json"),
        "production_eligible": values["production_eligible"],
        "identity_scope": "detached-optional-vector-handoff-observation",
    }


def _execution(
    job: CanonicalVectorProjectionDispatchJob,
    record: CanonicalVectorProjectionDispatchRecord,
    *,
    replayed: bool,
) -> CanonicalVectorProjectionDispatchExecution:
    return CanonicalVectorProjectionDispatchExecution(
        status=(
            CanonicalVectorProjectionDispatchExecutionStatus.REPLAYED
            if replayed
            else CanonicalVectorProjectionDispatchExecutionStatus.RECORDED
        ),
        job=job,
        record=record,
        replayed=replayed,
    )


def _accepted_dispatch(
    intent: CanonicalVectorProjectionIntent,
    receipt: VectorProjectionReceipt,
) -> CanonicalVectorProjectionDispatch:
    if receipt.status is VectorProjectionStatus.FAILED:
        return _failed_dispatch(
            intent,
            VectorProjectionErrorCode.RETRYABLE,
            "vector adapter reports an existing projection in FAILED state",
        )
    if receipt.status is VectorProjectionStatus.RETIRED:
        return _failed_dispatch(
            intent,
            VectorProjectionErrorCode.CONFLICT,
            "vector adapter reports an existing projection in RETIRED state",
        )
    return CanonicalVectorProjectionDispatch(
        intent_logical_key=intent.logical_key,
        intent_semantic_sha256=intent.semantic_sha256,
        idempotency_key=intent.to_request().idempotency_key,
        status=CanonicalVectorProjectionDispatchStatus.QUEUED,
        receipt=receipt,
    )


def _failed_dispatch(
    intent: CanonicalVectorProjectionIntent,
    error_code: VectorProjectionErrorCode,
    error_message: str,
) -> CanonicalVectorProjectionDispatch:
    message = error_message[:256] or error_code.value
    return CanonicalVectorProjectionDispatch(
        intent_logical_key=intent.logical_key,
        intent_semantic_sha256=intent.semantic_sha256,
        idempotency_key=intent.to_request().idempotency_key,
        status=CanonicalVectorProjectionDispatchStatus.FAILED,
        error_code=error_code,
        error_message=message,
    )


def _frozen_intent(value: object) -> CanonicalVectorProjectionIntent:
    intent = _require_intent(value)
    # The source instance has already passed strict validation; only discard
    # the nonsemantic enqueue observation without coercing nested enums/tuples.
    return intent.model_copy(update={"requested_at": None})


def _validate_record_for_job(
    record: CanonicalVectorProjectionDispatchRecord,
    job: CanonicalVectorProjectionDispatchJob,
) -> None:
    if (
        record.job_logical_key != job.logical_key
        or record.job_semantic_sha256 != job.semantic_sha256
    ):
        raise VectorProjectionDispatchConflict(
            "vector dispatch record does not belong to the sealed job"
        )
    if (
        record.dispatch.intent_logical_key != job.intent.logical_key
        or record.dispatch.intent_semantic_sha256 != job.intent.semantic_sha256
        or record.dispatch.idempotency_key != job.intent.to_request().idempotency_key
    ):
        raise VectorProjectionDispatchConflict(
            "vector dispatch record does not bind the sealed revision intent"
        )


def _ensure_regular_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VectorProjectionDispatchStorageError(f"cannot create {label}: {error}") from error
    if path.is_symlink() or not path.is_dir():
        raise VectorProjectionDispatchStorageError(f"{label} must be a regular directory")


def _validate_digest(value: Sha256Digest | str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("semantic_sha256 must be a lowercase SHA-256 digest")
    return value


def _parse_exact_job(raw: bytes) -> CanonicalVectorProjectionDispatchJob:
    return _parse_exact_json_model(raw, CanonicalVectorProjectionDispatchJob, "vector dispatch job")


def _parse_exact_record(raw: bytes) -> CanonicalVectorProjectionDispatchRecord:
    return _parse_exact_json_model(
        raw, CanonicalVectorProjectionDispatchRecord, "vector dispatch record"
    )


def _parse_exact_json_model[TModel: StrictModel](
    raw: bytes,
    model_type: type[TModel],
    label: str,
) -> TModel:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise VectorProjectionDispatchStorageError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise VectorProjectionDispatchStorageError(f"{label} root must be an object")
    if canonical_json_bytes(document) != raw:
        raise VectorProjectionDispatchStorageError(f"{label} bytes are not exact canonical JSON")
    try:
        model = model_type.model_validate_json(raw, strict=True)
    except ValueError as error:
        raise VectorProjectionDispatchStorageError(f"invalid {label}: {error}") from error
    if canonical_json_bytes(model) != raw:
        raise VectorProjectionDispatchStorageError(f"{label} bytes are inconsistent with its model")
    return model


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _require_intent(value: object) -> CanonicalVectorProjectionIntent:
    if not isinstance(value, CanonicalVectorProjectionIntent):
        raise TypeError("intent must be a CanonicalVectorProjectionIntent")
    try:
        return CanonicalVectorProjectionIntent.model_validate_json(
            canonical_json_bytes(value), strict=True
        )
    except ValueError as error:
        raise ValueError(f"invalid vector projection intent: {error}") from error


def _require_job(value: object) -> CanonicalVectorProjectionDispatchJob:
    if not isinstance(value, CanonicalVectorProjectionDispatchJob):
        raise TypeError("job must be a CanonicalVectorProjectionDispatchJob")
    try:
        return CanonicalVectorProjectionDispatchJob.model_validate_json(
            canonical_json_bytes(value), strict=True
        )
    except ValueError as error:
        raise ValueError(f"invalid vector dispatch job: {error}") from error


def _require_dispatch(value: object) -> CanonicalVectorProjectionDispatch:
    if not isinstance(value, CanonicalVectorProjectionDispatch):
        raise TypeError("dispatch must be a CanonicalVectorProjectionDispatch")
    try:
        return CanonicalVectorProjectionDispatch.model_validate_json(
            canonical_json_bytes(value), strict=True
        )
    except ValueError as error:
        raise ValueError(f"invalid vector projection dispatch: {error}") from error


def _require_record(value: object) -> CanonicalVectorProjectionDispatchRecord:
    if not isinstance(value, CanonicalVectorProjectionDispatchRecord):
        raise TypeError("record must be a CanonicalVectorProjectionDispatchRecord")
    try:
        return CanonicalVectorProjectionDispatchRecord.model_validate_json(
            canonical_json_bytes(value), strict=True
        )
    except ValueError as error:
        raise ValueError(f"invalid vector dispatch record: {error}") from error


__all__ = [
    "CANONICAL_VECTOR_DISPATCH_JOB_KEY_NAMESPACE",
    "CANONICAL_VECTOR_DISPATCH_JOB_PROJECTION_VERSION",
    "CANONICAL_VECTOR_DISPATCH_RECORD_KEY_NAMESPACE",
    "CANONICAL_VECTOR_DISPATCH_RECORD_PROJECTION_VERSION",
    "CanonicalVectorProjectionDispatchBridge",
    "CanonicalVectorProjectionDispatchExecution",
    "CanonicalVectorProjectionDispatchExecutionStatus",
    "CanonicalVectorProjectionDispatchJob",
    "CanonicalVectorProjectionDispatchRecord",
    "CanonicalVectorProjectionDispatchStore",
    "CanonicalVectorProjectionDispatchWorker",
    "CanonicalVectorProjectionJobDispatch",
    "CanonicalVectorProjectionJobStatus",
    "VectorProjectionDispatchConflict",
    "VectorProjectionDispatchError",
    "VectorProjectionDispatchRetryNotAllowed",
    "VectorProjectionDispatchStorageError",
    "canonical_vector_projection_dispatch_job_projection",
    "canonical_vector_projection_dispatch_record_projection",
]
