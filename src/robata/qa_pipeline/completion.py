"""Deterministic completion gate for coarse and dense six-camera QA."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.pipeline import CameraQAStatus
from robata.qa_pipeline.coarse import (
    CameraCoarseResult,
    CoarseQAOutputRef,
    CoarseQAResult,
    coarse_qa_semantic_projection,
)
from robata.qa_pipeline.dense import (
    CameraDenseResult,
    CoarseQAObservationRef,
    DenseQAOutcome,
    DenseQAPlanner,
    DenseQAPlanningPolicy,
    DenseQAResult,
    DenseQAStatus,
    DenseQAWorkItem,
    DenseQAWorkManifest,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

LOCAL_QA_COMPLETION_POLICY_VERSION = "local-qa-completion-v2"
QA_COMPLETION_PROJECTION_VERSION = "qa-completion-semantic-v2"


class QACompletionStatus(StrEnum):
    QA_COMPLETE = "QA_COMPLETE"
    DENSE_REQUIRED = "DENSE_REQUIRED"
    QA_INCOMPLETE = "QA_INCOMPLETE"


class QACompletionProjectionError(ValueError):
    """Coarse QA cannot be completed without losing or inventing evidence."""


class CameraFinalQAResult(StrictModel):
    """One camera finalized with complete coarse and optional dense lineage."""

    camera_id: CameraId
    status: Literal[CameraQAStatus.GOOD]
    observations: tuple[CoarseQAObservationRef, ...]
    dense_observations: tuple[CameraDenseResult, ...] = ()
    policy_version: SchemaVersion
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_camera(self) -> Self:
        if not self.observations:
            raise ValueError("a final camera result requires observations")
        if any(item.camera_id is not self.camera_id for item in self.observations):
            raise ValueError("final coarse observations must match their camera")
        if tuple(item.package_ordinal for item in self.observations) != tuple(
            range(len(self.observations))
        ):
            raise ValueError("final camera observations must cover every package in order")
        if any(
            item.camera_id is not self.camera_id or item.local_status is not CameraQAStatus.GOOD
            for item in self.dense_observations
        ):
            raise ValueError("final dense observations must be matching GOOD evidence")
        return self


class QACompletionAggregate(StrictModel):
    """Local six-camera final aggregate without invented confidence or quality."""

    status: Literal[QACompletionStatus.QA_COMPLETE]
    camera_results: tuple[CameraFinalQAResult, ...]
    policy_version: SchemaVersion
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if tuple(item.camera_id for item in self.camera_results) != CAMERA_IDS:
            raise ValueError("final aggregate requires six cameras in canonical order")
        if any(item.policy_version != self.policy_version for item in self.camera_results):
            raise ValueError("final camera policy must match the aggregate")
        return self


class QACompletionResult(StrictModel):
    """Stable result of the local coarse/dense QA completion gate."""

    completion_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    package_set_id: NonEmptyString
    mcap_id: NonEmptyString
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    package_ids: tuple[OpaqueUuid, ...]
    coarse_result_semantic_sha256: Sha256Digest
    coarse_policy_version: SchemaVersion
    source_outputs: tuple[CoarseQAOutputRef, ...]
    coarse_coverage: tuple[CoarseQAObservationRef, ...]
    dense_work_manifest: DenseQAWorkManifest
    dense_result: DenseQAResult | None
    dense_failure_code: NonEmptyString | None
    final_aggregate: QACompletionAggregate | None
    status: QACompletionStatus
    policy_version: SchemaVersion
    semantic_projection_version: Literal["qa-completion-semantic-v2"] = "qa-completion-semantic-v2"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not self.package_ids or len(set(self.package_ids)) != len(self.package_ids):
            raise ValueError("QA completion package IDs must be nonempty and unique")
        output_ids = tuple(item.artifact_id for item in self.source_outputs)
        if output_ids != tuple(sorted(set(output_ids))):
            raise ValueError("source outputs must be unique and canonically ordered")

        expected_coordinates = tuple(
            (ordinal, camera_id)
            for ordinal in range(len(self.package_ids))
            for camera_id in CAMERA_IDS
        )
        actual_coordinates = tuple(
            (item.package_ordinal, item.camera_id) for item in self.coarse_coverage
        )
        if actual_coordinates != expected_coordinates:
            raise ValueError("QA completion must preserve package-by-six coverage")
        declared_outputs = set(self.source_outputs)
        for item in self.coarse_coverage:
            if item.package_id != self.package_ids[item.package_ordinal]:
                raise ValueError("coverage does not match package IDs")
            if item.source_output not in declared_outputs:
                raise ValueError("coverage references an undeclared enriched output")

        expected_status = _completion_status(
            self.coarse_coverage,
            self.dense_result,
            self.dense_failure_code,
        )
        if self.status is not expected_status:
            raise ValueError("completion status does not match coarse/dense evidence")
        _validate_manifest_binding(self)
        _validate_aggregate_binding(self)

        expected_digest = semantic_sha256(qa_completion_semantic_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("QA completion semantic_sha256 is inconsistent")
        if self.completion_id != _stable_id(expected_digest):
            raise ValueError("QA completion ID is inconsistent")
        return self


class QACompletionProjector:
    """Plan dense work and finalize it only from a matching dense result."""

    def __init__(
        self,
        policy_version: str = LOCAL_QA_COMPLETION_POLICY_VERSION,
        *,
        dense_planning_policy: DenseQAPlanningPolicy | None = None,
    ) -> None:
        if not isinstance(policy_version, str) or not policy_version or len(policy_version) > 128:
            raise ValueError("policy_version must be a non-empty schema-version string")
        self._policy_version = policy_version
        planning_policy = dense_planning_policy or DenseQAPlanningPolicy(version=policy_version)
        if planning_policy.version != policy_version:
            raise ValueError("dense planning policy version must match completion policy")
        self._dense_planner = DenseQAPlanner(planning_policy)

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def project(
        self,
        coarse_result: CoarseQAResult,
        dense_result: DenseQAResult | None = None,
        *,
        recording_interval: NanosecondInterval | None = None,
    ) -> QACompletionResult:
        return self._project(
            coarse_result,
            dense_result,
            dense_failure_code=None,
            recording_interval=recording_interval,
        )

    def block(
        self,
        coarse_result: CoarseQAResult,
        failure_code: str,
        *,
        recording_interval: NanosecondInterval | None = None,
    ) -> QACompletionResult:
        """Finalize required dense work as incomplete after a terminal local failure."""

        if not isinstance(failure_code, str) or not failure_code:
            raise ValueError("failure_code must be a non-empty string")
        return self._project(
            coarse_result,
            None,
            dense_failure_code=failure_code,
            recording_interval=recording_interval,
        )

    def _project(
        self,
        coarse_result: CoarseQAResult,
        dense_result: DenseQAResult | None,
        *,
        dense_failure_code: str | None,
        recording_interval: NanosecondInterval | None,
    ) -> QACompletionResult:
        coarse = _validated_coarse_result(coarse_result)
        dense = _validated_dense_result(dense_result)
        coverage = tuple(_observation_ref(item) for item in coarse.package_camera_results)
        manifest = self._dense_planner.plan(
            coarse,
            recording_interval=recording_interval,
        )
        _validate_dense_result_binding(manifest, dense)
        status = _completion_status(coverage, dense, dense_failure_code)
        aggregate = (
            _final_aggregate(coverage, dense, self._policy_version)
            if status is QACompletionStatus.QA_COMPLETE
            else None
        )
        values: dict[str, Any] = {
            "package_set_id": coarse.package_set_id,
            "mcap_id": coarse.mcap_id,
            "input_plan_id": coarse.input_plan_id,
            "input_plan_semantic_sha256": coarse.input_plan_semantic_sha256,
            "package_ids": coarse.package_ids,
            "coarse_result_semantic_sha256": semantic_sha256(coarse_qa_semantic_projection(coarse)),
            "coarse_policy_version": coarse.policy_version,
            "source_outputs": coarse.source_outputs,
            "coarse_coverage": coverage,
            "dense_work_manifest": manifest,
            "dense_result": dense,
            "dense_failure_code": dense_failure_code,
            "final_aggregate": aggregate,
            "status": status,
            "policy_version": self._policy_version,
            "semantic_projection_version": QA_COMPLETION_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = QACompletionResult.model_construct(
            completion_id=_stable_id("0" * 64),
            semantic_sha256="0" * 64,
            **values,
        )
        digest = semantic_sha256(qa_completion_semantic_projection(draft))
        return QACompletionResult.model_validate(
            {
                **values,
                "completion_id": _stable_id(digest),
                "semantic_sha256": digest,
            },
            strict=True,
        )


def qa_completion_semantic_projection(result: QACompletionResult) -> dict[str, Any]:
    """Return the sole versioned identity projection for QA completion."""

    return {
        "semantic_projection_version": result.semantic_projection_version,
        "package_set_id": result.package_set_id,
        "mcap_id": result.mcap_id,
        "input_plan_id": result.input_plan_id,
        "input_plan_semantic_sha256": result.input_plan_semantic_sha256,
        "package_ids": result.package_ids,
        "coarse_result_semantic_sha256": result.coarse_result_semantic_sha256,
        "coarse_policy_version": result.coarse_policy_version,
        "source_outputs": result.source_outputs,
        "coarse_coverage": result.coarse_coverage,
        "dense_work_manifest": result.dense_work_manifest,
        "dense_result": result.dense_result,
        "dense_failure_code": result.dense_failure_code,
        "final_aggregate": result.final_aggregate,
        "status": result.status,
        "policy_version": result.policy_version,
        "production_eligible": result.production_eligible,
    }


def _validated_coarse_result(value: CoarseQAResult) -> CoarseQAResult:
    if not isinstance(value, CoarseQAResult):
        raise QACompletionProjectionError("coarse_result must be a CoarseQAResult")
    try:
        return CoarseQAResult.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QACompletionProjectionError(
            "coarse_result failed immutable contract validation"
        ) from exc


def _validated_dense_result(value: DenseQAResult | None) -> DenseQAResult | None:
    if value is None:
        return None
    if not isinstance(value, DenseQAResult):
        raise QACompletionProjectionError("dense_result must be a DenseQAResult")
    try:
        return DenseQAResult.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QACompletionProjectionError(
            "dense_result failed immutable contract validation"
        ) from exc


def _observation_ref(item: CameraCoarseResult) -> CoarseQAObservationRef:
    interval = item.claim.interval
    if interval is None:
        raise QACompletionProjectionError("coarse observation is missing its exact interval")
    return CoarseQAObservationRef(
        package_id=item.package_id,
        package_ordinal=item.package_ordinal,
        camera_id=item.camera_id,
        interval=NanosecondInterval(start_ns=interval.start_ns, end_ns=interval.end_ns),
        provider_observation=item.claim.observation,
        local_status=item.local_status,
        source_claim_id=item.claim.claim_id,
        source_output=item.source_output,
    )


def _completion_status(
    coverage: tuple[CoarseQAObservationRef, ...],
    dense_result: DenseQAResult | None,
    dense_failure_code: str | None,
) -> QACompletionStatus:
    statuses = {item.local_status for item in coverage}
    if CameraQAStatus.UNKNOWN in statuses:
        if dense_result is not None or dense_failure_code is not None:
            raise ValueError("unknown coarse evidence cannot enter dense execution")
        return QACompletionStatus.QA_INCOMPLETE
    if statuses & {CameraQAStatus.DEGRADED, CameraQAStatus.UNUSABLE}:
        if dense_result is not None and dense_failure_code is not None:
            raise ValueError("dense result and dense failure are mutually exclusive")
        if dense_failure_code is not None:
            return QACompletionStatus.QA_INCOMPLETE
        if dense_result is None:
            return QACompletionStatus.DENSE_REQUIRED
        return (
            QACompletionStatus.QA_COMPLETE
            if dense_result.status is DenseQAStatus.COMPLETE
            else QACompletionStatus.QA_INCOMPLETE
        )
    if statuses == {CameraQAStatus.GOOD}:
        if dense_result is not None or dense_failure_code is not None:
            raise ValueError("all-GOOD coarse evidence requires zero dense children")
        return QACompletionStatus.QA_COMPLETE
    raise ValueError("coarse coverage contains unsupported statuses")


def _final_aggregate(
    coverage: tuple[CoarseQAObservationRef, ...],
    dense_result: DenseQAResult | None,
    policy_version: str,
) -> QACompletionAggregate:
    dense_observations = (
        tuple(
            observation
            for unit in dense_result.units
            for observation in unit.evidence.package_camera_results
        )
        if dense_result is not None
        else ()
    )
    return QACompletionAggregate(
        status=QACompletionStatus.QA_COMPLETE,
        camera_results=tuple(
            CameraFinalQAResult(
                camera_id=camera_id,
                status=CameraQAStatus.GOOD,
                observations=tuple(item for item in coverage if item.camera_id is camera_id),
                dense_observations=tuple(
                    item for item in dense_observations if item.camera_id is camera_id
                ),
                policy_version=policy_version,
            )
            for camera_id in CAMERA_IDS
        ),
        policy_version=policy_version,
    )


def _validate_manifest_binding(result: QACompletionResult) -> None:
    manifest = result.dense_work_manifest
    if manifest.policy_version != result.policy_version:
        raise ValueError("dense manifest policy does not match completion")
    coarse_status = _completion_status(result.coarse_coverage, None, None)
    expected_outcome = {
        QACompletionStatus.QA_COMPLETE: DenseQAOutcome.SKIPPED_NOT_NEEDED,
        QACompletionStatus.DENSE_REQUIRED: DenseQAOutcome.DENSE_REQUIRED,
        QACompletionStatus.QA_INCOMPLETE: DenseQAOutcome.BLOCKED_INCOMPLETE,
    }[coarse_status]
    if manifest.outcome is not expected_outcome:
        raise ValueError("dense outcome does not match coarse control status")
    expected_sources = tuple(
        item
        for item in result.coarse_coverage
        if item.local_status in {CameraQAStatus.DEGRADED, CameraQAStatus.UNUSABLE}
    )
    if coarse_status is not QACompletionStatus.DENSE_REQUIRED:
        expected_sources = ()
    if tuple(item.source for item in manifest.items) != expected_sources:
        raise ValueError("dense manifest does not preserve exact suspicious lineage")
    _validate_dense_result_binding(manifest, result.dense_result)


def _validate_dense_result_binding(
    manifest: DenseQAWorkManifest,
    dense_result: DenseQAResult | None,
) -> None:
    if manifest.outcome is not DenseQAOutcome.DENSE_REQUIRED:
        if dense_result is not None:
            raise ValueError("zero-child dense outcome cannot retain a dense result")
        return
    if dense_result is None:
        return
    if (
        dense_result.mcap_id != manifest.mcap_id
        or dense_result.coarse_result_digest != manifest.coarse_result_digest
        or dense_result.work_manifest_id != manifest.manifest_id
        or dense_result.work_manifest_digest != manifest.semantic_digest
        or dense_result.policy_version != manifest.policy_version
    ):
        raise ValueError("dense result does not bind the exact work manifest")
    expected_units = tuple((item.unit_id, item.semantic_digest) for item in manifest.units)
    actual_units = tuple(
        (item.work_unit_id, item.work_unit_semantic_digest) for item in dense_result.units
    )
    if actual_units != expected_units:
        raise ValueError("dense result does not preserve planned unit order")


def _validate_aggregate_binding(result: QACompletionResult) -> None:
    aggregate = result.final_aggregate
    if result.status is not QACompletionStatus.QA_COMPLETE:
        if aggregate is not None:
            raise ValueError("non-complete QA cannot publish a final aggregate")
        return
    if aggregate is None or aggregate.policy_version != result.policy_version:
        raise ValueError("QA_COMPLETE requires a matching final aggregate")
    dense_observations = (
        tuple(
            observation
            for unit in result.dense_result.units
            for observation in unit.evidence.package_camera_results
        )
        if result.dense_result is not None
        else ()
    )
    for camera in aggregate.camera_results:
        expected = tuple(
            item for item in result.coarse_coverage if item.camera_id is camera.camera_id
        )
        if camera.observations != expected:
            raise ValueError("final aggregate does not preserve complete coarse coverage")
        expected_dense = tuple(
            item for item in dense_observations if item.camera_id is camera.camera_id
        )
        if camera.dense_observations != expected_dense:
            raise ValueError("final aggregate does not preserve complete dense evidence")
    if result.dense_result is None and any(
        item.local_status is not CameraQAStatus.GOOD for item in result.coarse_coverage
    ):
        raise ValueError("QA_COMPLETE requires GOOD coarse or complete dense evidence")


def _stable_id(digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:qa-completion:{digest}"))


__all__ = [
    "LOCAL_QA_COMPLETION_POLICY_VERSION",
    "QA_COMPLETION_PROJECTION_VERSION",
    "CameraFinalQAResult",
    "CoarseQAObservationRef",
    "DenseQAOutcome",
    "DenseQAWorkItem",
    "DenseQAWorkManifest",
    "QACompletionAggregate",
    "QACompletionProjectionError",
    "QACompletionProjector",
    "QACompletionResult",
    "QACompletionStatus",
    "qa_completion_semantic_projection",
]
