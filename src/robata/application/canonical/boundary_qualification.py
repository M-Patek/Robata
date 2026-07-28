"""Detached durable P12 qualification over exact boundary-refinement evidence.

The canonical boundary result remains authoritative.  This module only seals an
already-validated dual-role refinement execution into an immutable sidecar job,
then lets an independent worker produce a candidate-only qualification report.
Neither enqueueing nor draining a job reads or changes primary completion,
event revisions, or released completion payloads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import model_validator

from robata.application.canonical.result_validation import CanonicalBoundaryRefinementExecution
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey
from robata.event_pipeline.boundary_qualification import (
    BoundaryQualificationCameraObservation,
    BoundaryQualificationCase,
    BoundaryQualificationEngine,
    BoundaryQualificationObservationOutcome,
    BoundaryQualificationPolicy,
    BoundaryQualificationReport,
    BoundaryQualificationRoleInput,
    boundary_qualification_case_projection,
    boundary_qualification_policy_projection,
    verify_boundary_qualification_report,
)
from robata.event_pipeline.boundary_refinement import (
    BoundaryCameraOutcome,
    BoundaryRefinementRoleResult,
    CameraBoundaryEvidence,
)
from robata.qa_pipeline.boundary_quality import BoundaryCameraQualityEvidence
from robata.tempfiles import make_temp_file

CANONICAL_BOUNDARY_QUALIFICATION_JOB_PROJECTION_VERSION: Final = (
    "canonical-boundary-qualification-job-semantic-v1"
)
CANONICAL_BOUNDARY_QUALIFICATION_JOB_KEY_NAMESPACE: Final = (
    "canonical-boundary-qualification-job-v1"
)
CANONICAL_BOUNDARY_QUALIFICATION_CAMERA_SLOT_KEY_NAMESPACE: Final = (
    "canonical-boundary-qualification-camera-slot-v1"
)
CANONICAL_BOUNDARY_QUALIFICATION_CAMERA_SLOT_PROJECTION_VERSION: Final = (
    "canonical-boundary-qualification-camera-slot-semantic-v1"
)


class BoundaryQualificationSidecarError(RuntimeError):
    """Base error for the detached P12 job and report boundary."""


class BoundaryQualificationSidecarConflict(BoundaryQualificationSidecarError):
    """A content-addressed job or report path contains different bytes."""


class BoundaryQualificationSidecarStorageError(BoundaryQualificationSidecarError):
    """A sidecar file is malformed, noncanonical, or cannot be persisted."""


class BoundaryQualificationDispatchStatus(StrEnum):
    """Outcome of storing an immutable P12 sidecar job."""

    ENQUEUED = "ENQUEUED"
    REPLAYED = "REPLAYED"


class BoundaryQualificationPublicationStatus(StrEnum):
    """Outcome of deriving one candidate-only report from a sealed job."""

    PUBLISHED = "PUBLISHED"
    REPLAYED = "REPLAYED"


class CanonicalBoundaryQualificationJob(StrictModel):
    """A durable, non-authoritative comparison request for one action boundary."""

    schema_version: Literal["1.0"] = "1.0"
    case: BoundaryQualificationCase
    policy: BoundaryQualificationPolicy
    projection_version: Literal["canonical-boundary-qualification-job-semantic-v1"] = (
        CANONICAL_BOUNDARY_QUALIFICATION_JOB_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        case: BoundaryQualificationCase,
        policy: BoundaryQualificationPolicy,
    ) -> Self:
        """Seal exact candidate inputs under a content-addressed job identity."""

        checked_case = _require_model(case, BoundaryQualificationCase, "case")
        checked_policy = _require_model(policy, BoundaryQualificationPolicy, "policy")
        values: dict[str, object] = {
            "schema_version": "1.0",
            "case": checked_case,
            "policy": checked_policy,
            "projection_version": CANONICAL_BOUNDARY_QUALIFICATION_JOB_PROJECTION_VERSION,
            "production_eligible": False,
        }
        digest = semantic_sha256(_job_projection_values(values))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{CANONICAL_BOUNDARY_QUALIFICATION_JOB_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        digest = semantic_sha256(canonical_boundary_qualification_job_projection(self))
        if (
            self.semantic_sha256 != digest
            or self.logical_key != f"{CANONICAL_BOUNDARY_QUALIFICATION_JOB_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("boundary qualification job semantic identity is inconsistent")
        return self


class BoundaryQualificationDispatchResult(StrictModel):
    """The sealed job and whether recovery found its exact bytes already present."""

    status: BoundaryQualificationDispatchStatus
    job: CanonicalBoundaryQualificationJob
    replayed: bool

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = (
            BoundaryQualificationDispatchStatus.REPLAYED
            if self.replayed
            else BoundaryQualificationDispatchStatus.ENQUEUED
        )
        if self.status is not expected:
            raise ValueError("dispatch status must match replayed")
        return self


class BoundaryQualificationPublicationResult(StrictModel):
    """The immutable candidate report published for one durable job."""

    status: BoundaryQualificationPublicationStatus
    job_logical_key: NodeLogicalKey
    job_semantic_sha256: Sha256Digest
    report: BoundaryQualificationReport
    replayed: bool

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = (
            BoundaryQualificationPublicationStatus.REPLAYED
            if self.replayed
            else BoundaryQualificationPublicationStatus.PUBLISHED
        )
        if self.status is not expected:
            raise ValueError("publication status must match replayed")
        if self.job_logical_key.rsplit(":", 1)[-1] != self.job_semantic_sha256:
            raise ValueError("publication job logical key is inconsistent")
        return self


class BoundaryQualificationSidecarStore:
    """Exact-canonical storage for independent P12 jobs and reports.

    The job is written before the worker derives a report.  Jobs are never
    deleted or marked complete, so a restart can safely recompute and replay
    reports without creating another candidate identity.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).absolute()
        _ensure_regular_directory(self._root, "boundary qualification sidecar root")
        _ensure_regular_directory(self._jobs_directory, "boundary qualification job directory")
        _ensure_regular_directory(
            self._reports_directory, "boundary qualification report directory"
        )

    @property
    def root(self) -> Path:
        """Return the non-resolved root that owns the sidecar files."""

        return self._root

    @property
    def _jobs_directory(self) -> Path:
        return self._root / "jobs"

    @property
    def _reports_directory(self) -> Path:
        return self._root / "reports"

    def job_path(self, semantic_sha256: Sha256Digest | str) -> Path:
        """Return the immutable path for one job semantic identity."""

        return self._jobs_directory / f"{_validate_digest(semantic_sha256)}.json"

    def report_path(self, semantic_sha256: Sha256Digest | str) -> Path:
        """Return the immutable path for one report semantic identity."""

        return self._reports_directory / f"{_validate_digest(semantic_sha256)}.json"

    def put_or_get_job(
        self,
        job: CanonicalBoundaryQualificationJob,
    ) -> tuple[CanonicalBoundaryQualificationJob, bool]:
        """Persist a sealed P12 input once, or replay only byte-identical input."""

        checked = _require_model(job, CanonicalBoundaryQualificationJob, "job")
        stored, replayed = self._publish_or_read(
            path=self.job_path(checked.semantic_sha256),
            expected=canonical_json_bytes(checked),
            parser=_parse_exact_job,
            label="boundary qualification job",
        )
        if stored != checked:
            raise BoundaryQualificationSidecarConflict(
                "boundary qualification job path contains different immutable canonical bytes"
            )
        return stored, replayed

    def get_job(
        self,
        semantic_sha256: Sha256Digest | str,
    ) -> CanonicalBoundaryQualificationJob | None:
        """Load a job only if its canonical bytes and filename both verify."""

        digest = _validate_digest(semantic_sha256)
        path = self.job_path(digest)
        if not path.exists() and not path.is_symlink():
            return None
        job = _parse_exact_job(self._read_exact(path, "boundary qualification job"))
        if job.semantic_sha256 != digest:
            raise BoundaryQualificationSidecarStorageError(
                "boundary qualification job path does not match its semantic digest"
            )
        return job

    def list_jobs(self) -> tuple[CanonicalBoundaryQualificationJob, ...]:
        """Load every immutable job in deterministic semantic-digest order."""

        _ensure_regular_directory(self._jobs_directory, "boundary qualification job directory")
        jobs: list[CanonicalBoundaryQualificationJob] = []
        for path in sorted(self._jobs_directory.glob("*.json"), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file():
                raise BoundaryQualificationSidecarStorageError(
                    "boundary qualification job is not a regular file"
                )
            job = _parse_exact_job(self._read_exact(path, "boundary qualification job"))
            if path != self.job_path(job.semantic_sha256):
                raise BoundaryQualificationSidecarStorageError(
                    "boundary qualification job path does not match its semantic digest"
                )
            jobs.append(job)
        return tuple(jobs)

    def put_or_get_report(
        self,
        report: BoundaryQualificationReport,
    ) -> tuple[BoundaryQualificationReport, bool]:
        """Persist a deterministic candidate report once, or replay it exactly."""

        checked = _require_report(report)
        stored, replayed = self._publish_or_read(
            path=self.report_path(checked.semantic_sha256),
            expected=canonical_json_bytes(checked),
            parser=_parse_exact_report,
            label="boundary qualification report",
        )
        if stored != checked:
            raise BoundaryQualificationSidecarConflict(
                "boundary qualification report path contains different immutable canonical bytes"
            )
        return stored, replayed

    def get_report(
        self,
        semantic_sha256: Sha256Digest | str,
    ) -> BoundaryQualificationReport | None:
        """Load a report only if it deterministically reproduces from its job inputs."""

        digest = _validate_digest(semantic_sha256)
        path = self.report_path(digest)
        if not path.exists() and not path.is_symlink():
            return None
        report = _parse_exact_report(self._read_exact(path, "boundary qualification report"))
        if report.semantic_sha256 != digest:
            raise BoundaryQualificationSidecarStorageError(
                "boundary qualification report path does not match its semantic digest"
            )
        return report

    def _publish_or_read[TArtifact](
        self,
        *,
        path: Path,
        expected: bytes,
        parser: Callable[[bytes], TArtifact],
        label: str,
    ) -> tuple[TArtifact, bool]:
        _ensure_regular_directory(path.parent, f"{label} directory")
        if path.exists() or path.is_symlink():
            actual = self._read_exact(path, label)
            if actual != expected:
                raise BoundaryQualificationSidecarConflict(
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
                if actual != expected:
                    raise BoundaryQualificationSidecarConflict(
                        f"concurrent {label} contains different immutable canonical bytes"
                    ) from None
                return parser(actual), True
            return parser(expected), False
        except OSError as error:
            raise BoundaryQualificationSidecarStorageError(
                f"cannot publish {label}: {error}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_exact(path: Path, label: str) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise BoundaryQualificationSidecarStorageError(f"{label} is not a regular file")
        try:
            return path.read_bytes()
        except OSError as error:
            raise BoundaryQualificationSidecarStorageError(
                f"cannot read {label}: {error}"
            ) from error


class CanonicalBoundaryQualificationBridge:
    """Seal accepted boundary evidence without changing primary canonical semantics."""

    def __init__(self, store: BoundaryQualificationSidecarStore) -> None:
        if not isinstance(store, BoundaryQualificationSidecarStore):
            raise TypeError("store must be a BoundaryQualificationSidecarStore")
        self._store = store

    def enqueue(
        self,
        *,
        execution: CanonicalBoundaryRefinementExecution,
        policy: BoundaryQualificationPolicy,
        quality_evidence: Sequence[BoundaryCameraQualityEvidence] = (),
    ) -> BoundaryQualificationDispatchResult:
        """Persist exact accepted boundary inputs for a later independent worker.

        Applications call this only after their primary completion becomes
        durable.  This class deliberately has no dependency on that completion
        surface, which prevents P12 from becoming a hidden completion stage.
        """

        job = boundary_qualification_job_from_execution(
            execution=execution,
            policy=policy,
            quality_evidence=quality_evidence,
        )
        stored, replayed = self._store.put_or_get_job(job)
        return BoundaryQualificationDispatchResult(
            status=(
                BoundaryQualificationDispatchStatus.REPLAYED
                if replayed
                else BoundaryQualificationDispatchStatus.ENQUEUED
            ),
            job=stored,
            replayed=replayed,
        )


class CanonicalBoundaryQualificationWorker:
    """Drain P12 jobs independently of completion, events, and stream stages."""

    def __init__(self, store: BoundaryQualificationSidecarStore) -> None:
        if not isinstance(store, BoundaryQualificationSidecarStore):
            raise TypeError("store must be a BoundaryQualificationSidecarStore")
        self._store = store

    def drain(self) -> tuple[BoundaryQualificationPublicationResult, ...]:
        """Derive or replay every candidate report in stable job order."""

        publications: list[BoundaryQualificationPublicationResult] = []
        for job in self._store.list_jobs():
            report = BoundaryQualificationEngine(job.policy).compare(job.case)
            stored, replayed = self._store.put_or_get_report(report)
            publications.append(
                BoundaryQualificationPublicationResult(
                    status=(
                        BoundaryQualificationPublicationStatus.REPLAYED
                        if replayed
                        else BoundaryQualificationPublicationStatus.PUBLISHED
                    ),
                    job_logical_key=job.logical_key,
                    job_semantic_sha256=job.semantic_sha256,
                    report=stored,
                    replayed=replayed,
                )
            )
        return tuple(publications)


def boundary_qualification_job_from_execution(
    *,
    execution: CanonicalBoundaryRefinementExecution,
    policy: BoundaryQualificationPolicy,
    quality_evidence: Sequence[BoundaryCameraQualityEvidence] = (),
) -> CanonicalBoundaryQualificationJob:
    """Freeze a fully validated dual-role execution into a candidate-only job."""

    checked_execution = _require_model(
        execution,
        CanonicalBoundaryRefinementExecution,
        "boundary refinement execution",
    )
    case = boundary_qualification_case_from_execution(
        execution=checked_execution,
        quality_evidence=quality_evidence,
    )
    return CanonicalBoundaryQualificationJob.create(case=case, policy=policy)


def boundary_qualification_case_from_execution(
    *,
    execution: CanonicalBoundaryRefinementExecution,
    quality_evidence: Sequence[BoundaryCameraQualityEvidence] = (),
) -> BoundaryQualificationCase:
    """Copy authoritative dual-role inputs without changing their reducer output."""

    checked_execution = _require_model(
        execution,
        CanonicalBoundaryRefinementExecution,
        "boundary refinement execution",
    )
    result = checked_execution.result
    onset = _role_input_from_result(checked_execution.onset.role_result)
    offset = _role_input_from_result(checked_execution.offset.role_result)
    quality = tuple(
        sorted(
            (
                _require_model(item, BoundaryCameraQualityEvidence, "quality evidence")
                for item in quality_evidence
            ),
            key=lambda item: (item.camera_id.value, item.logical_key),
        )
    )
    return BoundaryQualificationCase(
        source_boundary_result_logical_key=result.logical_key,
        source_boundary_result_semantic_sha256=result.semantic_sha256,
        source_boundary_result_exact_sha256=exact_bytes_sha256(canonical_json_bytes(result)),
        source_action_logical_key=checked_execution.action.logical_key,
        source_action_semantic_sha256=checked_execution.action.semantic_sha256,
        mcap_id=result.mcap_id,
        recording_identity=checked_execution.onset.role_result.recording_identity,
        source_content_sha256=result.source_content_sha256,
        camera_mapping_semantic_sha256=result.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=result.alignment_semantic_sha256,
        roles=(onset, offset),
        quality_evidence=quality,
        production_eligible=False,
    )


def canonical_boundary_qualification_job_projection(
    job: CanonicalBoundaryQualificationJob,
) -> dict[str, object]:
    """Return the internal-only job identity projection."""

    return _job_projection_values(
        {
            "schema_version": job.schema_version,
            "case": job.case,
            "policy": job.policy,
            "projection_version": job.projection_version,
            "production_eligible": job.production_eligible,
        }
    )


def _job_projection_values(values: dict[str, object]) -> dict[str, object]:
    case = values["case"]
    policy = values["policy"]
    if not isinstance(case, BoundaryQualificationCase):
        raise TypeError("boundary qualification job case is invalid")
    if not isinstance(policy, BoundaryQualificationPolicy):
        raise TypeError("boundary qualification job policy is invalid")
    return {
        "semantic_projection_version": values["projection_version"],
        "case": boundary_qualification_case_projection(case),
        "policy": boundary_qualification_policy_projection(policy),
        "production_eligible": values["production_eligible"],
        "identity_scope": "detached-candidate-qualification-not-event-or-completion-authority",
    }


def _role_input_from_result(
    result: BoundaryRefinementRoleResult,
) -> BoundaryQualificationRoleInput:
    observations = tuple(
        _camera_observation_from_slot(
            role_result=result,
            slot=result.camera_evidence[camera_id],
        )
        for camera_id in CAMERA_IDS
    )
    return BoundaryQualificationRoleInput(
        role=result.role,
        source_role_result_logical_key=result.logical_key,
        source_role_result_semantic_sha256=result.semantic_sha256,
        source_role_result_exact_sha256=exact_bytes_sha256(canonical_json_bytes(result)),
        window_interval=result.window_interval,
        minimum_observed_cameras=result.policy.minimum_observed_cameras,
        camera_observations=observations,
        baseline_boundary_estimate_ns=result.boundary_estimate_ns,
        baseline_uncertainty_ns=result.uncertainty_ns,
        baseline_boundary_interval=result.boundary_interval,
        baseline_outcome=result.outcome,
        production_eligible=False,
    )


def _camera_observation_from_slot(
    *,
    role_result: BoundaryRefinementRoleResult,
    slot: CameraBoundaryEvidence,
) -> BoundaryQualificationCameraObservation:
    if slot.camera_id not in CAMERA_IDS or slot.role is not role_result.role:
        raise BoundaryQualificationSidecarError("boundary camera slot differs from its role result")
    semantic_digest = semantic_sha256(
        {
            "semantic_projection_version": (
                CANONICAL_BOUNDARY_QUALIFICATION_CAMERA_SLOT_PROJECTION_VERSION
            ),
            "source_role_result_semantic_sha256": role_result.semantic_sha256,
            "camera_id": slot.camera_id.value,
            "slot": slot.model_dump(mode="json"),
        }
    )
    return BoundaryQualificationCameraObservation(
        camera_id=slot.camera_id,
        source_camera_evidence_logical_key=(
            f"{CANONICAL_BOUNDARY_QUALIFICATION_CAMERA_SLOT_KEY_NAMESPACE}:{semantic_digest}"
        ),
        source_camera_evidence_semantic_sha256=semantic_digest,
        source_camera_evidence_exact_sha256=exact_bytes_sha256(canonical_json_bytes(slot)),
        outcome=_qualification_observation_outcome(slot.outcome),
        observed_interval=slot.observed_interval,
        boundary_estimate_ns=slot.boundary_estimate_ns,
        uncertainty_ns=slot.uncertainty_ns,
        production_eligible=False,
    )


def _qualification_observation_outcome(
    outcome: BoundaryCameraOutcome,
) -> BoundaryQualificationObservationOutcome:
    return BoundaryQualificationObservationOutcome(outcome.value)


def _ensure_regular_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise BoundaryQualificationSidecarStorageError(f"cannot create {label}: {error}") from error
    if path.is_symlink() or not path.is_dir():
        raise BoundaryQualificationSidecarStorageError(f"{label} must be a regular directory")


def _validate_digest(value: Sha256Digest | str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("semantic_sha256 must be a lowercase SHA-256 digest")
    return value


def _parse_exact_job(raw: bytes) -> CanonicalBoundaryQualificationJob:
    job = _parse_exact_json_model(
        raw, CanonicalBoundaryQualificationJob, "boundary qualification job"
    )
    return _require_model(job, CanonicalBoundaryQualificationJob, "job")


def _parse_exact_report(raw: bytes) -> BoundaryQualificationReport:
    report = _parse_exact_json_model(
        raw, BoundaryQualificationReport, "boundary qualification report"
    )
    try:
        verified = verify_boundary_qualification_report(report)
    except ValueError as error:
        raise BoundaryQualificationSidecarStorageError(
            f"boundary qualification report does not reproduce: {error}"
        ) from error
    if canonical_json_bytes(verified) != raw:
        raise BoundaryQualificationSidecarStorageError(
            "boundary qualification report bytes are inconsistent with its model"
        )
    return verified


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
        raise BoundaryQualificationSidecarStorageError(f"invalid {label} JSON: {error}") from error
    if not isinstance(document, dict):
        raise BoundaryQualificationSidecarStorageError(f"{label} root must be an object")
    if canonical_json_bytes(document) != raw:
        raise BoundaryQualificationSidecarStorageError(
            f"{label} bytes are not exact canonical JSON"
        )
    try:
        return model_type.model_validate_json(raw, strict=True)
    except ValueError as error:
        raise BoundaryQualificationSidecarStorageError(f"invalid {label}: {error}") from error


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _require_model[TModel: StrictModel](
    value: object,
    model_type: type[TModel],
    label: str,
) -> TModel:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be a {model_type.__name__}")
    try:
        return model_type.model_validate_json(canonical_json_bytes(value), strict=True)
    except ValueError as error:
        raise ValueError(f"invalid {label}: {error}") from error


def _require_report(report: object) -> BoundaryQualificationReport:
    if not isinstance(report, BoundaryQualificationReport):
        raise TypeError("report must be a BoundaryQualificationReport")
    try:
        return verify_boundary_qualification_report(
            BoundaryQualificationReport.model_validate_json(
                canonical_json_bytes(report), strict=True
            )
        )
    except ValueError as error:
        raise ValueError(f"invalid report: {error}") from error


__all__ = [
    "CANONICAL_BOUNDARY_QUALIFICATION_CAMERA_SLOT_KEY_NAMESPACE",
    "CANONICAL_BOUNDARY_QUALIFICATION_CAMERA_SLOT_PROJECTION_VERSION",
    "CANONICAL_BOUNDARY_QUALIFICATION_JOB_KEY_NAMESPACE",
    "CANONICAL_BOUNDARY_QUALIFICATION_JOB_PROJECTION_VERSION",
    "BoundaryQualificationDispatchResult",
    "BoundaryQualificationDispatchStatus",
    "BoundaryQualificationPublicationResult",
    "BoundaryQualificationPublicationStatus",
    "BoundaryQualificationSidecarConflict",
    "BoundaryQualificationSidecarError",
    "BoundaryQualificationSidecarStorageError",
    "BoundaryQualificationSidecarStore",
    "CanonicalBoundaryQualificationBridge",
    "CanonicalBoundaryQualificationJob",
    "CanonicalBoundaryQualificationWorker",
    "boundary_qualification_case_from_execution",
    "boundary_qualification_job_from_execution",
    "canonical_boundary_qualification_job_projection",
]
