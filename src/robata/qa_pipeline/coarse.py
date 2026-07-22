"""Concrete local projection of enriched coarse-QA observations.

This module deliberately does not sample frames or invoke a model. Those
responsibilities already belong to the canonical sampling and inference
boundaries. The projector below is the first concrete coarse-QA domain step:
it accepts only authoritative QA_COARSE enriched outputs, proves that each
call-part covers its package/camera coordinates, conservatively reduces
coordinates split across call-parts, and emits a non-promotable local decision
without inventing calibrated confidence or formal QA records.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.pipeline import CameraQAStatus
from robata.contracts.temporal import TemporalPackageSet, TemporalPackageSetMember
from robata.inference.enrichment import (
    EnrichedProviderClaim,
    OrchestratorEnrichedOutput,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.inference.input_plan import InferenceInputPlan, RenderedProviderItem
from robata.inference.models import VisionTask
from robata.qa_pipeline.suspicion_reducer import SuspiciousInterval

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

LOCAL_COARSE_QA_POLICY_VERSION = "local-coarse-qa-projector-v2"
COARSE_QA_SEMANTIC_PROJECTION_VERSION = "canonical-coarse-qa-result-v1"

__all__ = [
    "COARSE_QA_SEMANTIC_PROJECTION_VERSION",
    "LOCAL_COARSE_QA_POLICY_VERSION",
    "CameraCoarseResult",
    "CoarseQAOutputRef",
    "CoarseQAPipeline",
    "CoarseQAProjectionError",
    "CoarseQAProjector",
    "CoarseQAResult",
    "CoarseQAStatus",
    "SamplingPlan",
    "SuspiciousInterval",
    "coarse_qa_semantic_projection",
]


class SamplingPlan(StrictModel):
    """Compatibility-only coarse sampling configuration.

    Sampling is not performed by this module. The type remains exported for
    callers that still describe a coarse rate, while the canonical materializer
    owns frame selection and package construction.
    """

    target_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    frames_per_window: Annotated[int, Field(strict=True, ge=1)] = 64
    window_overlap_sec: Annotated[float, Field(strict=True, ge=0.0)] = 0.0
    policy_version: SchemaVersion


class CoarseQAStatus(StrEnum):
    """Local coarse-stage control result, never a production qualification."""

    COMPLETE = "COMPLETE"
    REQUIRES_DENSE = "REQUIRES_DENSE"
    INCOMPLETE = "INCOMPLETE"


class CoarseQAProjectionError(ValueError):
    """Raised when enriched coarse evidence cannot be projected safely."""


class CoarseQAOutputRef(StrictModel):
    """Stable reference to one authoritative enriched inference artifact."""

    artifact_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    enrichment_logical_key: NodeLogicalKey
    inference_id: OpaqueUuid


class CameraCoarseResult(StrictModel):
    """One package/camera observation with its complete enriched claim retained."""

    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    camera_id: CameraId
    local_status: CameraQAStatus
    source_output: CoarseQAOutputRef
    claim: EnrichedProviderClaim

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if (
            self.claim.kind is not ProviderClaimKind.QA_OBSERVATION
            or self.claim.package_id != self.package_id
            or self.claim.package_ordinal != self.package_ordinal
            or self.claim.camera_id is not self.camera_id
        ):
            raise ValueError("coarse result must retain its exact enriched QA claim binding")
        expected = _camera_status(self.claim.observation)
        if self.local_status is not expected:
            raise ValueError("coarse local status must be derived from the provider observation")
        return self


class CoarseQAResult(StrictModel):
    """Deterministic, internal-only result of complete coarse-QA projection."""

    package_set_id: NonEmptyString
    mcap_id: NonEmptyString
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    package_ids: tuple[OpaqueUuid, ...]
    source_outputs: tuple[CoarseQAOutputRef, ...]
    package_camera_results: tuple[CameraCoarseResult, ...]
    local_status: CoarseQAStatus
    complete: bool
    requires_dense: bool
    policy_version: SchemaVersion
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not self.package_ids:
            raise ValueError("coarse QA requires at least one package")
        if len(set(self.package_ids)) != len(self.package_ids):
            raise ValueError("coarse QA package IDs must be unique")
        if len({item.artifact_id for item in self.source_outputs}) != len(self.source_outputs):
            raise ValueError("coarse QA source output references must be unique")

        expected_coordinates = tuple(
            (package_ordinal, camera_id)
            for package_ordinal in range(len(self.package_ids))
            for camera_id in CAMERA_IDS
        )
        actual_coordinates = tuple(
            (item.package_ordinal, item.camera_id) for item in self.package_camera_results
        )
        if actual_coordinates != expected_coordinates:
            raise ValueError(
                "coarse QA package/camera results must be complete and canonically ordered"
            )
        for item in self.package_camera_results:
            if item.package_id != self.package_ids[item.package_ordinal]:
                raise ValueError("coarse QA package result does not match package_ids")

        source_refs = set(self.source_outputs)
        if any(item.source_output not in source_refs for item in self.package_camera_results):
            raise ValueError("coarse QA result references an undeclared enriched output")

        expected_status = _result_status(self.package_camera_results)
        if self.local_status is not expected_status:
            raise ValueError("coarse QA local status is inconsistent with camera observations")
        expected_complete = expected_status is CoarseQAStatus.COMPLETE
        expected_dense = expected_status is CoarseQAStatus.REQUIRES_DENSE
        if self.complete is not expected_complete or self.requires_dense is not expected_dense:
            raise ValueError("coarse QA control flags are inconsistent with local status")
        return self


def coarse_qa_semantic_projection(result: CoarseQAResult) -> dict[str, Any]:
    """Return the semantic projection shared with the canonical logical node."""

    return {
        "semantic_projection_version": COARSE_QA_SEMANTIC_PROJECTION_VERSION,
        "result": result.model_dump(mode="json"),
    }


class CoarseQAProjector:
    """Project authoritative enriched QA claims into a local control result."""

    def __init__(self, policy_version: str = LOCAL_COARSE_QA_POLICY_VERSION) -> None:
        if not isinstance(policy_version, str) or not policy_version or len(policy_version) > 128:
            raise ValueError("policy_version must be a non-empty schema-version string")
        self._policy_version = policy_version

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def project(
        self,
        *,
        package_set: TemporalPackageSet,
        input_plan: InferenceInputPlan,
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
    ) -> CoarseQAResult:
        """Validate complete coarse coverage and derive a fail-closed local result."""

        checked_package_set = _validated_package_set(package_set)
        checked_plan = _validated_input_plan(input_plan)
        _validate_plan_binding(checked_package_set, checked_plan)

        members = checked_package_set.members
        expected_coordinates = {
            (member.ordinal, camera_id) for member in members for camera_id in CAMERA_IDS
        }
        rendered_coordinates = {
            (item.package_ordinal, item.camera_id) for item in checked_plan.rendered_items
        }
        if rendered_coordinates != expected_coordinates:
            missing = _coordinate_labels(expected_coordinates - rendered_coordinates)
            extra = _coordinate_labels(rendered_coordinates - expected_coordinates)
            raise CoarseQAProjectionError(
                "QA_COARSE input plan must render every package/camera coordinate exactly; "
                f"missing={missing}, extra={extra}"
            )

        rendered_by_ordinal = {
            item.provider_item_ordinal: item for item in checked_plan.rendered_items
        }
        source_outputs: dict[str, CoarseQAOutputRef] = {}
        projected: dict[tuple[int, CameraId], CameraCoarseResult] = {}
        for output_value in enriched_outputs:
            output = _validated_enriched_output(output_value)
            _validate_output_binding(checked_package_set, checked_plan, output)
            if output.artifact_id in source_outputs:
                raise CoarseQAProjectionError(
                    f"duplicate enriched output artifact: {output.artifact_id}"
                )
            source_ref = CoarseQAOutputRef(
                artifact_id=output.artifact_id,
                semantic_sha256=output.semantic_sha256,
                enrichment_logical_key=output.enrichment_logical_key,
                inference_id=output.selected_attempt.inference_id,
            )
            source_outputs[output.artifact_id] = source_ref

            for claim in output.claims:
                coordinate = _validate_claim(
                    claim=claim,
                    members=members,
                    rendered_by_ordinal=rendered_by_ordinal,
                )
                assert claim.package_id is not None
                assert claim.package_ordinal is not None
                assert claim.camera_id is not None
                candidate = CameraCoarseResult(
                    package_id=claim.package_id,
                    package_ordinal=claim.package_ordinal,
                    camera_id=claim.camera_id,
                    local_status=_camera_status(claim.observation),
                    source_output=source_ref,
                    claim=claim,
                )
                current = projected.get(coordinate)
                if current is None or _camera_status_rank(candidate.local_status) > (
                    _camera_status_rank(current.local_status)
                ):
                    projected[coordinate] = candidate

        actual_coordinates = set(projected)
        if actual_coordinates != expected_coordinates:
            missing = _coordinate_labels(expected_coordinates - actual_coordinates)
            extra = _coordinate_labels(actual_coordinates - expected_coordinates)
            raise CoarseQAProjectionError(
                "QA_COARSE enriched claims must cover every package/camera exactly once; "
                f"missing={missing}, extra={extra}"
            )

        ordered_results = tuple(
            projected[(member.ordinal, camera_id)] for member in members for camera_id in CAMERA_IDS
        )
        local_status = _result_status(ordered_results)
        return CoarseQAResult(
            package_set_id=checked_package_set.package_set_id,
            mcap_id=checked_package_set.mcap_id,
            input_plan_id=checked_plan.input_plan_id,
            input_plan_semantic_sha256=checked_plan.semantic_sha256,
            package_ids=tuple(member.package_id for member in members),
            source_outputs=tuple(
                source_outputs[artifact_id] for artifact_id in sorted(source_outputs)
            ),
            package_camera_results=ordered_results,
            local_status=local_status,
            complete=local_status is CoarseQAStatus.COMPLETE,
            requires_dense=local_status is CoarseQAStatus.REQUIRES_DENSE,
            policy_version=self._policy_version,
            production_eligible=False,
        )


# Keep the established public name while replacing the old classifier/sampler
# skeleton with the concrete projector API above.
CoarseQAPipeline = CoarseQAProjector


def _validated_package_set(value: TemporalPackageSet) -> TemporalPackageSet:
    if not isinstance(value, TemporalPackageSet):
        raise CoarseQAProjectionError("package_set must be a TemporalPackageSet")
    try:
        return TemporalPackageSet.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoarseQAProjectionError("package_set failed immutable contract validation") from exc


def _validated_input_plan(value: InferenceInputPlan) -> InferenceInputPlan:
    if not isinstance(value, InferenceInputPlan):
        raise CoarseQAProjectionError("input_plan must be an InferenceInputPlan")
    try:
        return InferenceInputPlan.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoarseQAProjectionError("input_plan failed immutable contract validation") from exc


def _validated_enriched_output(
    value: OrchestratorEnrichedOutput,
) -> OrchestratorEnrichedOutput:
    if not isinstance(value, OrchestratorEnrichedOutput):
        raise CoarseQAProjectionError(
            "enriched_outputs must contain OrchestratorEnrichedOutput values"
        )
    try:
        return OrchestratorEnrichedOutput.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoarseQAProjectionError(
            "enriched output failed immutable contract validation"
        ) from exc


def _validate_plan_binding(
    package_set: TemporalPackageSet,
    input_plan: InferenceInputPlan,
) -> None:
    if (
        input_plan.subject.task is not VisionTask.QA_COARSE
        or input_plan.request_catalog.task is not VisionTask.QA_COARSE
    ):
        raise CoarseQAProjectionError("coarse QA requires an exact QA_COARSE input plan")

    members = package_set.members
    subject_packages = input_plan.subject.packages
    catalog_packages = input_plan.request_catalog.packages
    if len(subject_packages) != len(members) or len(catalog_packages) != len(members):
        raise CoarseQAProjectionError("input plan package count does not match package set")

    for member, subject, catalog in zip(members, subject_packages, catalog_packages, strict=True):
        expected = (
            member.package_id,
            member.ordinal,
            member.package_semantic_content_sha256,
            member.package_manifest_sha256,
        )
        if (
            subject.package_id,
            subject.ordinal,
            subject.semantic_content_sha256,
            subject.manifest_bytes_sha256,
        ) != expected or (
            catalog.package_id,
            catalog.ordinal,
            catalog.semantic_content_sha256,
            catalog.manifest_bytes_sha256,
        ) != expected:
            raise CoarseQAProjectionError(
                "input plan packages do not match the exact temporal package set"
            )

    for item in input_plan.rendered_items:
        if item.package_ordinal >= len(members):
            raise CoarseQAProjectionError("rendered item package ordinal is outside package set")
        member = members[item.package_ordinal]
        if item.package_id != member.package_id:
            raise CoarseQAProjectionError("rendered item package ID does not match package set")
        if not member.start_ns <= item.aligned_timestamp_ns < member.end_ns:
            raise CoarseQAProjectionError(
                "rendered QA frame timestamp must lie inside its temporal package"
            )


def _validate_output_binding(
    package_set: TemporalPackageSet,
    input_plan: InferenceInputPlan,
    output: OrchestratorEnrichedOutput,
) -> None:
    if output.task is not VisionTask.QA_COARSE:
        raise CoarseQAProjectionError("coarse QA projector rejects non-QA_COARSE output")
    if output.abstained:
        raise CoarseQAProjectionError("QA_COARSE output cannot abstain")
    if (
        output.input_plan_id != input_plan.input_plan_id
        or output.input_plan_semantic_sha256 != input_plan.semantic_sha256
        or output.request_catalog_id != input_plan.request_catalog.request_catalog_id
        or output.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
    ):
        raise CoarseQAProjectionError("enriched output is not bound to the exact input plan")
    if (
        output.provider_claim_schema.sha256
        != input_plan.prompt_output.provider_response_schema_sha256
        or output.enriched_output_schema.sha256
        != input_plan.prompt_output.enriched_domain_schema_sha256
    ):
        raise CoarseQAProjectionError("enriched output schemas are not bound by input plan")
    if (
        output.authority.mcap_id != package_set.mcap_id
        or output.authority.camera_mapping_run_id != package_set.camera_mapping_run_id
        or output.authority.alignment_id != package_set.alignment_id
    ):
        raise CoarseQAProjectionError("enriched output authority does not match package lineage")
    if (
        output.authority.prompt_version != input_plan.prompt_output.prompt_version
        or output.authority.prompt_sha256 != input_plan.prompt_output.prompt_sha256
    ):
        raise CoarseQAProjectionError("enriched output prompt authority does not match input plan")


def _validate_claim(
    *,
    claim: EnrichedProviderClaim,
    members: tuple[TemporalPackageSetMember, ...],
    rendered_by_ordinal: dict[int, RenderedProviderItem],
) -> tuple[int, CameraId]:
    if (
        claim.kind is not ProviderClaimKind.QA_OBSERVATION
        or claim.package_id is None
        or claim.package_ordinal is None
        or claim.camera_id is None
        or claim.interval is None
    ):
        raise CoarseQAProjectionError(
            "coarse QA accepts only package/camera-bound QA_OBSERVATION claims"
        )
    if claim.package_ordinal >= len(members):
        raise CoarseQAProjectionError("QA claim package ordinal is outside package set")
    member = members[claim.package_ordinal]
    if claim.package_id != member.package_id:
        raise CoarseQAProjectionError("QA claim package ID does not match package set")
    if claim.interval.start_ns < member.start_ns or claim.interval.end_ns > member.end_ns:
        raise CoarseQAProjectionError("QA claim interval must lie inside its temporal package")

    if claim.observation in {ProviderObservation.GOOD, ProviderObservation.DEGRADED} and not (
        claim.evidence
    ):
        raise CoarseQAProjectionError("observed GOOD/DEGRADED QA claims require evidence")
    for evidence in claim.evidence:
        item = rendered_by_ordinal.get(evidence.provider_item_ordinal)
        if item is None:
            raise CoarseQAProjectionError("QA evidence is outside the rendered input plan")
        if (
            evidence.package_id != item.package_id
            or evidence.package_ordinal != item.package_ordinal
            or evidence.camera_id is not item.camera_id
            or evidence.camera_ordinal != item.camera_ordinal
            or evidence.frame_id != item.frame_id
            or evidence.frame_ordinal != item.frame_ordinal
            or evidence.source_artifact_sha256 != item.source_artifact_sha256
        ):
            raise CoarseQAProjectionError("QA evidence does not resolve to the input plan")
    return claim.package_ordinal, claim.camera_id


def _camera_status(observation: ProviderObservation) -> CameraQAStatus:
    try:
        return {
            ProviderObservation.GOOD: CameraQAStatus.GOOD,
            ProviderObservation.DEGRADED: CameraQAStatus.DEGRADED,
            ProviderObservation.UNUSABLE: CameraQAStatus.UNUSABLE,
            ProviderObservation.UNKNOWN: CameraQAStatus.UNKNOWN,
        }[observation]
    except KeyError as exc:
        raise ValueError("unsupported coarse QA observation") from exc


def _camera_status_rank(status: CameraQAStatus) -> int:
    """Rank partial call-part observations by conservative local severity."""

    return {
        CameraQAStatus.GOOD: 0,
        CameraQAStatus.DEGRADED: 1,
        CameraQAStatus.UNUSABLE: 2,
        CameraQAStatus.UNKNOWN: 3,
    }[status]


def _result_status(results: Sequence[CameraCoarseResult]) -> CoarseQAStatus:
    statuses = {item.local_status for item in results}
    if CameraQAStatus.UNKNOWN in statuses:
        return CoarseQAStatus.INCOMPLETE
    if statuses & {CameraQAStatus.DEGRADED, CameraQAStatus.UNUSABLE}:
        return CoarseQAStatus.REQUIRES_DENSE
    if statuses == {CameraQAStatus.GOOD}:
        return CoarseQAStatus.COMPLETE
    raise ValueError("coarse QA observations contain an unsupported local status")


def _coordinate_labels(coordinates: set[tuple[int, CameraId]]) -> tuple[str, ...]:
    return tuple(
        f"package[{package_ordinal}]/{camera_id.value}"
        for package_ordinal, camera_id in sorted(
            coordinates, key=lambda item: (item[0], CAMERA_IDS.index(item[1]))
        )
    )
