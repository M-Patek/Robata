"""Detached P11 recording-association dispatch after primary completion.

This module intentionally has no dependency on primary-completion persistence.
It freezes association inputs only after an immutable completion is available,
persists an independent immutable job, and lets a separate worker publish the
derived report.  The worker can be delayed, retried, or unavailable without
changing completion, released stream result versions, or event identities.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import model_validator

from robata.application.canonical.primary_completion import CommittedPrimaryCompletion
from robata.application.canonical.recording_association import (
    CanonicalRecordingAssociationBridge,
    RecordingAssociationPublicationResult,
    RecordingAssociationPublicationStatus,
    RecordingAssociationReportStore,
)
from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey
from robata.event_pipeline.evidence import (
    ActionEvidenceResult,
    action_observation_semantic_projection,
)
from robata.event_pipeline.recording_association import (
    AssociationAcceptedEvidenceRef,
    AssociationSourceActionRef,
    CompletedRecordingAssociationBinding,
    RecordingAssociationBridgeEvidence,
    RecordingAssociationEngine,
    RecordingAssociationInput,
    RecordingAssociationPolicy,
)
from robata.inference.enrichment import ProviderObservation
from robata.tempfiles import make_temp_file

CANONICAL_RECORDING_ASSOCIATION_JOB_PROJECTION_VERSION: Final = (
    "canonical-recording-association-job-semantic-v1"
)
CANONICAL_RECORDING_ASSOCIATION_JOB_KEY_NAMESPACE: Final = "canonical-recording-association-job-v1"
CANONICAL_COMPLETED_RECORDING_KEY_NAMESPACE: Final = "canonical-completed-recording-v1"
LOCAL_RECORDING_ASSOCIATION_POLICY_VERSION: Final = "canonical-local-recording-association-v1"


class RecordingAssociationDispatchError(RuntimeError):
    """Base error for detached association dispatch."""


class RecordingAssociationDispatchConflict(RecordingAssociationDispatchError):
    """An immutable job identity has different exact canonical bytes."""


class RecordingAssociationDispatchStorageError(RecordingAssociationDispatchError):
    """Stored detached association work is malformed or unsafe to use."""


class RecordingAssociationDispatchStatus(StrEnum):
    """Outcome of one attempt to persist an independent derivation job."""

    ENQUEUED = "ENQUEUED"
    REPLAYED = "REPLAYED"
    NO_ACCEPTED_EVIDENCE = "NO_ACCEPTED_EVIDENCE"


class RecordingAssociationDispatchResult(StrictModel):
    """Observable non-authoritative enqueue outcome."""

    status: RecordingAssociationDispatchStatus
    job: CanonicalRecordingAssociationJob | None
    replayed: bool

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is RecordingAssociationDispatchStatus.NO_ACCEPTED_EVIDENCE:
            if self.job is not None or self.replayed:
                raise ValueError("no-evidence dispatch cannot carry a job or replay")
            return self
        if self.job is None:
            raise ValueError("enqueued or replayed dispatch requires a job")
        expected = (
            RecordingAssociationDispatchStatus.REPLAYED
            if self.replayed
            else RecordingAssociationDispatchStatus.ENQUEUED
        )
        if self.status is not expected:
            raise ValueError("dispatch status must match replayed")
        return self


class CanonicalRecordingAssociationJob(StrictModel):
    """One immutable, detached association derivation request.

    The job includes only already accepted evidence.  Its source-action and
    accepted-observation identities remain independent of event identities and
    of the released V3/V4 recording reduction shapes.
    """

    schema_version: Literal["1.0"] = "1.0"
    projection_version: Literal["canonical-recording-association-job-semantic-v1"] = (
        CANONICAL_RECORDING_ASSOCIATION_JOB_PROJECTION_VERSION
    )
    recording: CompletedRecordingAssociationBinding
    policy: RecordingAssociationPolicy
    inputs: tuple[RecordingAssociationInput, ...]
    bridge_evidence: tuple[RecordingAssociationBridgeEvidence, ...] = ()
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        recording: CompletedRecordingAssociationBinding,
        policy: RecordingAssociationPolicy,
        inputs: Sequence[RecordingAssociationInput],
        bridge_evidence: Sequence[RecordingAssociationBridgeEvidence] = (),
    ) -> Self:
        """Freeze a canonical job without entering any primary authority."""

        ordered_inputs = tuple(sorted(inputs, key=_input_sort_key))
        ordered_bridges = tuple(sorted(bridge_evidence, key=_bridge_sort_key))
        values: dict[str, object] = {
            "schema_version": "1.0",
            "projection_version": CANONICAL_RECORDING_ASSOCIATION_JOB_PROJECTION_VERSION,
            "recording": recording,
            "policy": policy,
            "inputs": ordered_inputs,
            "bridge_evidence": ordered_bridges,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            **values,
            semantic_sha256="0" * 64,
            logical_key=f"{CANONICAL_RECORDING_ASSOCIATION_JOB_KEY_NAMESPACE}:{'0' * 64}",
        )
        digest = semantic_sha256(canonical_recording_association_job_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{CANONICAL_RECORDING_ASSOCIATION_JOB_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_job(self) -> Self:
        if self.schema_version != "1.0":
            raise ValueError("recording association job uses an unsupported schema version")
        if self.projection_version != CANONICAL_RECORDING_ASSOCIATION_JOB_PROJECTION_VERSION:
            raise ValueError("recording association job uses an unsupported projection version")
        if self.production_eligible:
            raise ValueError("recording association jobs are not production eligible")
        if not self.inputs:
            raise ValueError("recording association job requires accepted inputs")
        expected_inputs = tuple(sorted(self.inputs, key=_input_sort_key))
        if self.inputs != expected_inputs:
            raise ValueError("recording association inputs must be canonically ordered")
        source_keys = tuple(item.source_action.source_action_logical_key for item in self.inputs)
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("recording association inputs must have unique source actions")
        if len(self.inputs) > self.policy.max_inputs:
            raise ValueError("recording association inputs exceed the policy limit")
        if any(
            item.mcap_id != self.recording.mcap_id
            or item.source_content_sha256 != self.recording.source_content_sha256
            or item.camera_mapping_semantic_sha256 != self.recording.camera_mapping_semantic_sha256
            or item.alignment_semantic_sha256 != self.recording.alignment_semantic_sha256
            for item in self.inputs
        ):
            raise ValueError("association input lineage differs from completed recording")
        expected_bridges = tuple(sorted(self.bridge_evidence, key=_bridge_sort_key))
        if self.bridge_evidence != expected_bridges:
            raise ValueError("recording association bridge evidence must be canonically ordered")
        available_sources = {item.source_action for item in self.inputs}
        if any(
            any(source not in available_sources for source in bridge.source_actions)
            for bridge in self.bridge_evidence
        ):
            raise ValueError("association bridge evidence references an unknown source action")
        expected = semantic_sha256(canonical_recording_association_job_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("recording association job semantic identity is inconsistent")
        if self.logical_key != f"{CANONICAL_RECORDING_ASSOCIATION_JOB_KEY_NAMESPACE}:{expected}":
            raise ValueError("recording association job logical identity is inconsistent")
        return self


class RecordingAssociationJobExecution(StrictModel):
    """One independent worker result for an immutable association job."""

    job_logical_key: NodeLogicalKey
    job_semantic_sha256: Sha256Digest
    publication: RecordingAssociationPublicationResult

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.job_logical_key.rsplit(":", 1)[-1] != self.job_semantic_sha256:
            raise ValueError("association job execution has inconsistent job identity")
        if self.publication.status is RecordingAssociationPublicationStatus.NO_ACCEPTED_EVIDENCE:
            raise ValueError("persisted association jobs must contain accepted inputs")
        return self


def canonical_recording_association_job_projection(
    job: CanonicalRecordingAssociationJob,
) -> dict[str, object]:
    """Return the identity-bearing representation of one detached job."""

    return {
        "semantic_projection_version": job.projection_version,
        "recording": job.recording.model_dump(mode="json"),
        "policy": job.policy.model_dump(mode="json"),
        "inputs": [item.model_dump(mode="json") for item in job.inputs],
        "bridge_evidence": [item.model_dump(mode="json") for item in job.bridge_evidence],
        "production_eligible": job.production_eligible,
    }


class RecordingAssociationJobStore:
    """Exact-byte append-only store for independent association jobs."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).absolute()
        _ensure_regular_directory(self._root, "recording association dispatch root")
        _ensure_regular_directory(self._jobs_directory, "recording association job directory")

    @property
    def root(self) -> Path:
        """Return the dispatch root without resolving potentially hostile links."""

        return self._root

    @property
    def _jobs_directory(self) -> Path:
        return self._root / "jobs"

    def job_path(self, semantic_sha256: Sha256Digest | str) -> Path:
        """Return the content-addressed path for one immutable job."""

        digest = _validate_digest(semantic_sha256)
        return self._jobs_directory / f"{digest}.json"

    def put_or_get(
        self,
        job: CanonicalRecordingAssociationJob,
    ) -> tuple[CanonicalRecordingAssociationJob, bool]:
        """Persist one job once, or replay only byte-identical canonical JSON."""

        checked = _require_job(job)
        path = self.job_path(checked.semantic_sha256)
        expected = canonical_json_bytes(checked)
        stored, replayed = self._publish_or_read(path, expected)
        if (
            stored != checked
            or stored.semantic_sha256 != checked.semantic_sha256
            or stored.logical_key != checked.logical_key
        ):
            raise RecordingAssociationDispatchConflict(
                "association job path contains different immutable canonical bytes"
            )
        return stored, replayed

    def list_jobs(self) -> tuple[CanonicalRecordingAssociationJob, ...]:
        """Load all pending immutable jobs in deterministic digest order.

        Jobs remain immutable after a report is published.  Reprocessing a job
        is safe because report publication is itself content-addressed.
        """

        _ensure_regular_directory(self._jobs_directory, "recording association job directory")
        paths = tuple(sorted(self._jobs_directory.glob("*.json"), key=lambda item: item.name))
        jobs: list[CanonicalRecordingAssociationJob] = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RecordingAssociationDispatchStorageError(
                    "recording association job is not a regular file"
                )
            job = _parse_exact_job(self._read_exact(path))
            if path != self.job_path(job.semantic_sha256):
                raise RecordingAssociationDispatchStorageError(
                    "recording association job path does not match its semantic digest"
                )
            jobs.append(job)
        return tuple(jobs)

    def _publish_or_read(
        self,
        path: Path,
        expected: bytes,
    ) -> tuple[CanonicalRecordingAssociationJob, bool]:
        _ensure_regular_directory(path.parent, "recording association job directory")
        if path.exists() or path.is_symlink():
            actual = self._read_exact(path)
            if actual != expected:
                raise RecordingAssociationDispatchConflict(
                    "existing association job contains different immutable canonical bytes"
                )
            return _parse_exact_job(actual), True

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
                actual = self._read_exact(path)
                if actual != expected:
                    raise RecordingAssociationDispatchConflict(
                        "existing association job contains different immutable canonical bytes"
                    ) from None
                return _parse_exact_job(actual), True
            return _parse_exact_job(expected), False
        except OSError as error:
            raise RecordingAssociationDispatchStorageError(
                f"cannot publish recording association job: {error}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_exact(path: Path) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise RecordingAssociationDispatchStorageError(
                "recording association job is not a regular file"
            )
        try:
            return path.read_bytes()
        except OSError as error:
            raise RecordingAssociationDispatchStorageError(
                f"cannot read recording association job: {error}"
            ) from error


class CanonicalRecordingAssociationWorker:
    """Drain immutable jobs without entering primary completion authority."""

    def __init__(
        self,
        *,
        jobs: RecordingAssociationJobStore,
        reports: RecordingAssociationReportStore,
    ) -> None:
        if not isinstance(jobs, RecordingAssociationJobStore):
            raise TypeError("jobs must be a RecordingAssociationJobStore")
        if not isinstance(reports, RecordingAssociationReportStore):
            raise TypeError("reports must be a RecordingAssociationReportStore")
        self._jobs = jobs
        self._reports = reports

    def drain(self) -> tuple[RecordingAssociationJobExecution, ...]:
        """Derive every queued report, relying on immutable report replay."""

        executions: list[RecordingAssociationJobExecution] = []
        for job in self._jobs.list_jobs():
            bridge = CanonicalRecordingAssociationBridge(
                RecordingAssociationEngine(job.policy),
                self._reports,
            )
            publication = bridge.derive_and_publish(
                recording=job.recording,
                inputs=job.inputs,
                bridge_evidence=job.bridge_evidence,
            )
            executions.append(
                RecordingAssociationJobExecution(
                    job_logical_key=job.logical_key,
                    job_semantic_sha256=job.semantic_sha256,
                    publication=publication,
                )
            )
        return tuple(executions)


def local_recording_association_policy() -> RecordingAssociationPolicy:
    """Return the explicit local, non-production association policy."""

    return RecordingAssociationPolicy.create(
        version=LOCAL_RECORDING_ASSOCIATION_POLICY_VERSION,
        max_gap_ns=0,
        min_input_confidence_millionths=500_000,
        min_bridge_confidence_millionths=500_000,
        allow_label_transitions=False,
    )


def job_from_committed_primary_completion(
    committed: CommittedPrimaryCompletion,
    *,
    policy: RecordingAssociationPolicy | None = None,
) -> CanonicalRecordingAssociationJob | None:
    """Project accepted ACTION_EVIDENCE leaves after immutable completion.

    V3/V4 recording-result summaries intentionally lack camera-overlap and
    confidence fields.  This builder therefore consumes only the exact
    completion-detail ACTION_EVIDENCE executions and declines to create a job
    when no eligible positive, scored camera evidence is available.
    """

    if not isinstance(committed, CommittedPrimaryCompletion):
        raise TypeError("committed must be a CommittedPrimaryCompletion")
    resolved_policy = local_recording_association_policy() if policy is None else policy
    if not isinstance(resolved_policy, RecordingAssociationPolicy):
        raise TypeError("policy must be a RecordingAssociationPolicy or None")

    inputs = tuple(
        input_value
        for execution in committed.detail.action_evidence_executions
        if (
            input_value := _input_from_action_evidence(
                execution.evidence_result,
                completion_exact_sha256=committed.completion.detailed_result.exact_bytes_sha256,
            )
        )
        is not None
    )
    if not inputs:
        return None
    if len(inputs) > resolved_policy.max_inputs:
        raise RecordingAssociationDispatchError(
            "accepted association inputs exceed the configured association policy limit"
        )

    first = inputs[0]
    if first.mcap_id != committed.detail.mcap_id:
        raise RecordingAssociationDispatchError(
            "accepted association evidence mcap differs from completed primary detail"
        )
    recording = CompletedRecordingAssociationBinding(
        completed_run_id=committed.detail.run_id,
        completed_recording_logical_key=(
            f"{CANONICAL_COMPLETED_RECORDING_KEY_NAMESPACE}:{committed.detail.semantic_sha256}"
        ),
        completed_recording_semantic_sha256=committed.detail.semantic_sha256,
        completed_recording_exact_sha256=committed.completion.detailed_result.exact_bytes_sha256,
        mcap_id=first.mcap_id,
        source_content_sha256=first.source_content_sha256,
        camera_mapping_semantic_sha256=first.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=first.alignment_semantic_sha256,
    )
    return CanonicalRecordingAssociationJob.create(
        recording=recording,
        policy=resolved_policy,
        inputs=inputs,
    )


def enqueue_recording_association_after_completion(
    *,
    committed: CommittedPrimaryCompletion,
    jobs: RecordingAssociationJobStore,
    policy: RecordingAssociationPolicy | None = None,
) -> RecordingAssociationDispatchResult:
    """Persist detached work only after primary completion has succeeded."""

    if not isinstance(jobs, RecordingAssociationJobStore):
        raise TypeError("jobs must be a RecordingAssociationJobStore")
    job = job_from_committed_primary_completion(committed, policy=policy)
    if job is None:
        return RecordingAssociationDispatchResult(
            status=RecordingAssociationDispatchStatus.NO_ACCEPTED_EVIDENCE,
            job=None,
            replayed=False,
        )
    stored, replayed = jobs.put_or_get(job)
    return RecordingAssociationDispatchResult(
        status=(
            RecordingAssociationDispatchStatus.REPLAYED
            if replayed
            else RecordingAssociationDispatchStatus.ENQUEUED
        ),
        job=stored,
        replayed=replayed,
    )


def _input_from_action_evidence(
    result: ActionEvidenceResult,
    *,
    completion_exact_sha256: Sha256Digest,
) -> RecordingAssociationInput | None:
    """Build one source-action input from positive scored camera observations."""

    candidate_label = result.candidate_label
    source_logical_key = result.logical_key
    source_semantic_sha256 = result.semantic_sha256
    source_action = AssociationSourceActionRef(
        source_action_logical_key=source_logical_key,
        source_action_semantic_sha256=source_semantic_sha256,
    )
    accepted: list[AssociationAcceptedEvidenceRef] = []
    for camera in result.camera_evidence.values():
        for observation in camera.observations:
            if (
                observation.observation
                not in {ProviderObservation.SUPPORTING, ProviderObservation.PARTIAL}
                or observation.interval is None
                or observation.label != candidate_label
                or observation.model_reported_score is None
            ):
                continue
            observation_semantic_sha256 = semantic_sha256(
                action_observation_semantic_projection(observation)
            )
            expected_logical_key = f"action-observation:{observation_semantic_sha256}"
            if observation.source_action_observation_logical_key != expected_logical_key:
                raise RecordingAssociationDispatchError(
                    "accepted camera observation has inconsistent semantic identity"
                )
            accepted.append(
                AssociationAcceptedEvidenceRef(
                    source_action=source_action,
                    accepted_evidence_logical_key=observation.source_action_observation_logical_key,
                    accepted_evidence_semantic_sha256=observation_semantic_sha256,
                    # Observation leaves are embedded in the exact immutable
                    # completion detail; use that persisted root as their exact
                    # byte anchor instead of claiming a separate source artifact.
                    accepted_evidence_exact_sha256=completion_exact_sha256,
                    camera_id=observation.camera_id,
                    interval=observation.interval,
                    label=candidate_label,
                    confidence_millionths=round(observation.model_reported_score * 1_000_000),
                )
            )
    if not accepted:
        return None
    return RecordingAssociationInput.create(
        source_action=source_action,
        mcap_id=result.mcap_id,
        source_content_sha256=result.source_content_sha256,
        camera_mapping_semantic_sha256=result.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=result.alignment_semantic_sha256,
        interval=_accepted_interval(accepted),
        label=candidate_label,
        accepted_evidence=accepted,
    )


def _accepted_interval(
    accepted: Sequence[AssociationAcceptedEvidenceRef],
) -> NanosecondInterval:
    """Return the positive accepted-evidence envelope for one source action."""

    if not accepted:
        raise ValueError("accepted evidence must be nonempty")
    return NanosecondInterval(
        start_ns=min(item.interval.start_ns for item in accepted),
        end_ns=max(item.interval.end_ns for item in accepted),
    )


def _input_sort_key(
    input_value: RecordingAssociationInput,
) -> tuple[int, int, str]:
    return (
        input_value.interval.start_ns,
        input_value.interval.end_ns,
        input_value.source_action.source_action_logical_key,
    )


def _bridge_sort_key(
    evidence: RecordingAssociationBridgeEvidence,
) -> tuple[str, str, str]:
    return (
        evidence.source_actions[0].source_action_logical_key,
        evidence.source_actions[1].source_action_logical_key,
        evidence.accepted_bridge_evidence_logical_key,
    )


def _ensure_regular_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RecordingAssociationDispatchStorageError(f"cannot create {label}: {error}") from error
    if path.is_symlink() or not path.is_dir():
        raise RecordingAssociationDispatchStorageError(f"{label} must be a regular directory")


def _validate_digest(value: Sha256Digest | str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("semantic_sha256 must be a lowercase SHA-256 digest")
    return value


def _parse_exact_job(raw: bytes) -> CanonicalRecordingAssociationJob:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RecordingAssociationDispatchStorageError(
            f"invalid recording association job JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise RecordingAssociationDispatchStorageError(
            "recording association job root must be an object"
        )
    if canonical_json_bytes(document) != raw:
        raise RecordingAssociationDispatchStorageError(
            "recording association job bytes are not exact canonical JSON"
        )
    try:
        job = CanonicalRecordingAssociationJob.model_validate_json(raw, strict=True)
    except ValueError as error:
        raise RecordingAssociationDispatchStorageError(
            f"invalid recording association job: {error}"
        ) from error
    if canonical_json_bytes(job) != raw:
        raise RecordingAssociationDispatchStorageError(
            "recording association job bytes are inconsistent with its model"
        )
    return job


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _require_job(value: object) -> CanonicalRecordingAssociationJob:
    if not isinstance(value, CanonicalRecordingAssociationJob):
        raise TypeError("job must be a CanonicalRecordingAssociationJob")
    try:
        return CanonicalRecordingAssociationJob.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise ValueError(f"invalid recording association job: {error}") from error


__all__ = [
    "CANONICAL_COMPLETED_RECORDING_KEY_NAMESPACE",
    "CANONICAL_RECORDING_ASSOCIATION_JOB_KEY_NAMESPACE",
    "CANONICAL_RECORDING_ASSOCIATION_JOB_PROJECTION_VERSION",
    "LOCAL_RECORDING_ASSOCIATION_POLICY_VERSION",
    "CanonicalRecordingAssociationJob",
    "CanonicalRecordingAssociationWorker",
    "RecordingAssociationDispatchConflict",
    "RecordingAssociationDispatchError",
    "RecordingAssociationDispatchResult",
    "RecordingAssociationDispatchStatus",
    "RecordingAssociationDispatchStorageError",
    "RecordingAssociationJobExecution",
    "RecordingAssociationJobStore",
    "canonical_recording_association_job_projection",
    "enqueue_recording_association_after_completion",
    "job_from_committed_primary_completion",
    "local_recording_association_policy",
]
