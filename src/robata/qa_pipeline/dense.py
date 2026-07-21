"""Deterministic QA_DENSE planning and evidence projection.

This module owns no sampling, provider, or storage side effects.  It turns a
complete coarse-QA result into stable dense work units, then validates
normalized terminal evidence for those units.  Every value remains local
conformance evidence and is therefore never production eligible.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.pipeline import CameraQAStatus
from robata.inference.enrichment import (
    EnrichedProviderClaim,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.qa_pipeline.coarse import (
    CameraCoarseResult,
    CoarseQAOutputRef,
    CoarseQAResult,
    CoarseQAStatus,
    coarse_qa_semantic_projection,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

LOCAL_DENSE_QA_POLICY_VERSION = "local-dense-qa-v1"
DENSE_QA_WORK_PROJECTION_VERSION = "dense-qa-work-semantic-v1"
DENSE_QA_RESULT_PROJECTION_VERSION = "dense-qa-result-semantic-v1"


class DenseQAProjectionError(ValueError):
    """Dense work or evidence cannot be projected without guessing."""


class DenseQAOutcome(StrEnum):
    SKIPPED_NOT_NEEDED = "SKIPPED_NOT_NEEDED"
    DENSE_REQUIRED = "DENSE_REQUIRED"
    BLOCKED_INCOMPLETE = "BLOCKED_INCOMPLETE"


class DenseQAStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class CoarseQAObservationRef(StrictModel):
    """Exact package-camera observation and enriched-output lineage."""

    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    camera_id: CameraId
    interval: NanosecondInterval
    provider_observation: ProviderObservation
    local_status: CameraQAStatus
    source_claim_id: OpaqueUuid
    source_output: CoarseQAOutputRef

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.local_status is not _camera_status(self.provider_observation):
            raise ValueError("local status must match the provider observation")
        return self


class DenseQAWorkItem(StrictModel):
    """One suspicious coarse coordinate with synchronized six-view context."""

    source: CoarseQAObservationRef
    context_camera_ids: tuple[CameraId, ...]
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_item(self) -> Self:
        if self.context_camera_ids != CAMERA_IDS:
            raise ValueError("dense work must retain all six context cameras")
        if self.source.local_status not in {
            CameraQAStatus.DEGRADED,
            CameraQAStatus.UNUSABLE,
        }:
            raise ValueError("dense work requires DEGRADED or UNUSABLE coarse evidence")
        return self


class DenseQAWorkUnit(StrictModel):
    """Merged, padded, and clipped interval to materialize exactly once."""

    unit_id: OpaqueUuid
    semantic_digest: Sha256Digest
    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    context_truncated: bool
    source_items: tuple[DenseQAWorkItem, ...]
    target_interval_ids: tuple[OpaqueUuid, ...]
    target_camera_ids: tuple[CameraId, ...]
    context_camera_ids: tuple[CameraId, ...]
    purpose: Literal["QA_DENSE"] = "QA_DENSE"
    policy_version: SchemaVersion
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_unit(self) -> Self:
        if not self.source_items:
            raise ValueError("a dense work unit requires source items")
        if self.context_camera_ids != CAMERA_IDS:
            raise ValueError("dense work units require canonical six-camera context")
        expected_sources = tuple(
            sorted(
                set(self.source_items),
                key=lambda item: (
                    item.source.package_ordinal,
                    CAMERA_IDS.index(item.source.camera_id),
                ),
            )
        )
        if self.source_items != expected_sources:
            raise ValueError("dense unit sources must be unique and canonically ordered")
        if any(
            item.source.interval.start_ns < self.requested_interval.start_ns
            or item.source.interval.end_ns > self.requested_interval.end_ns
            for item in self.source_items
        ):
            raise ValueError("dense unit requested interval must contain every source")
        if (
            self.effective_interval.start_ns > self.requested_interval.start_ns
            or self.effective_interval.end_ns < self.requested_interval.end_ns
        ):
            raise ValueError("dense unit effective interval cannot discard source evidence")
        expected_cameras = tuple(
            camera_id
            for camera_id in CAMERA_IDS
            if any(item.source.camera_id is camera_id for item in self.source_items)
        )
        if self.target_camera_ids != expected_cameras:
            raise ValueError("dense target cameras must be derived from source evidence")
        expected_intervals = tuple(
            _target_interval_id(item.source) for item in _unique_source_intervals(self.source_items)
        )
        if self.target_interval_ids != expected_intervals:
            raise ValueError("dense target interval IDs are inconsistent")
        digest = semantic_sha256(dense_qa_work_unit_projection(self))
        if self.semantic_digest != digest or self.unit_id != _stable_id("dense-unit", digest):
            raise ValueError("dense work unit identity is inconsistent")
        return self


class DenseQAWorkManifest(StrictModel):
    """Stable dense fan-out, including an explicit zero-child outcome."""

    manifest_id: OpaqueUuid
    semantic_digest: Sha256Digest
    mcap_id: NonEmptyString
    coarse_result_digest: Sha256Digest
    outcome: DenseQAOutcome
    items: tuple[DenseQAWorkItem, ...]
    units: tuple[DenseQAWorkUnit, ...]
    policy_version: SchemaVersion
    projection_version: Literal["dense-qa-work-semantic-v1"] = "dense-qa-work-semantic-v1"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected_items = tuple(
            sorted(
                set(self.items),
                key=lambda item: (
                    item.source.package_ordinal,
                    CAMERA_IDS.index(item.source.camera_id),
                ),
            )
        )
        if self.items != expected_items:
            raise ValueError("dense work must be unique and canonically ordered")
        if self.outcome is DenseQAOutcome.DENSE_REQUIRED:
            if not self.items or not self.units:
                raise ValueError("DENSE_REQUIRED requires items and work units")
            flattened = tuple(item for unit in self.units for item in unit.source_items)
            if tuple(sorted(flattened, key=_work_item_sort_key)) != self.items:
                raise ValueError("dense units must partition the exact manifest items")
            if tuple(unit.effective_interval.start_ns for unit in self.units) != tuple(
                sorted(unit.effective_interval.start_ns for unit in self.units)
            ):
                raise ValueError("dense units must be stored in temporal order")
            if any(unit.policy_version != self.policy_version for unit in self.units):
                raise ValueError("dense unit policy must match its manifest")
        elif self.items or self.units:
            raise ValueError("a non-required dense outcome must have zero children")
        digest = semantic_sha256(dense_qa_work_manifest_projection(self))
        if self.semantic_digest != digest or self.manifest_id != _stable_id(
            "dense-manifest", digest
        ):
            raise ValueError("dense work manifest identity is inconsistent")
        return self


class DenseQAPlanningPolicy(StrictModel):
    """Only the interval operations that affect dense work identity."""

    version: SchemaVersion = LOCAL_DENSE_QA_POLICY_VERSION
    padding_ns: Annotated[int, Field(strict=True, ge=0)] = 500_000_000
    merge_gap_ns: Annotated[int, Field(strict=True, ge=0)] = 0
    production_eligible: Literal[False] = False


class DenseQAPlanner:
    """Derive deterministic dense units from authoritative coarse evidence."""

    def __init__(self, policy: DenseQAPlanningPolicy | None = None) -> None:
        self._policy = policy or DenseQAPlanningPolicy()

    @property
    def policy(self) -> DenseQAPlanningPolicy:
        return self._policy

    def plan(
        self,
        coarse_result: CoarseQAResult,
        *,
        recording_interval: NanosecondInterval | None = None,
    ) -> DenseQAWorkManifest:
        coarse = _validated_coarse_result(coarse_result)
        bounds = _validated_interval(recording_interval)
        coverage = tuple(_observation_ref(item) for item in coarse.package_camera_results)
        suspicious = tuple(
            DenseQAWorkItem(source=item, context_camera_ids=CAMERA_IDS)
            for item in coverage
            if item.local_status in {CameraQAStatus.DEGRADED, CameraQAStatus.UNUSABLE}
        )
        if coarse.local_status is CoarseQAStatus.COMPLETE:
            outcome = DenseQAOutcome.SKIPPED_NOT_NEEDED
        elif coarse.local_status is CoarseQAStatus.INCOMPLETE:
            outcome = DenseQAOutcome.BLOCKED_INCOMPLETE
            suspicious = ()
        else:
            outcome = DenseQAOutcome.DENSE_REQUIRED

        units = self._build_units(coarse.mcap_id, suspicious, bounds)
        values: dict[str, Any] = {
            "mcap_id": coarse.mcap_id,
            "coarse_result_digest": semantic_sha256(coarse_qa_semantic_projection(coarse)),
            "outcome": outcome,
            "items": suspicious,
            "units": units,
            "policy_version": self._policy.version,
            "projection_version": DENSE_QA_WORK_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = DenseQAWorkManifest.model_construct(
            manifest_id=_stable_id("dense-manifest", "0" * 64),
            semantic_digest="0" * 64,
            **values,
        )
        digest = semantic_sha256(dense_qa_work_manifest_projection(draft))
        return DenseQAWorkManifest.model_validate(
            {
                **values,
                "manifest_id": _stable_id("dense-manifest", digest),
                "semantic_digest": digest,
            },
            strict=True,
        )

    def _build_units(
        self,
        mcap_id: str,
        items: tuple[DenseQAWorkItem, ...],
        bounds: NanosecondInterval | None,
    ) -> tuple[DenseQAWorkUnit, ...]:
        if not items:
            return ()
        ordered = sorted(
            items,
            key=lambda item: (
                item.source.interval.start_ns,
                item.source.interval.end_ns,
                item.source.package_ordinal,
                CAMERA_IDS.index(item.source.camera_id),
            ),
        )
        groups: list[list[DenseQAWorkItem]] = []
        group_end = 0
        for item in ordered:
            interval = item.source.interval
            if bounds is not None and (
                interval.start_ns < bounds.start_ns or interval.end_ns > bounds.end_ns
            ):
                raise DenseQAProjectionError("coarse interval lies outside recording bounds")
            if not groups or interval.start_ns > group_end + self._policy.merge_gap_ns:
                groups.append([item])
                group_end = interval.end_ns
            else:
                groups[-1].append(item)
                group_end = max(group_end, interval.end_ns)
        return tuple(self._make_unit(mcap_id, group, bounds) for group in groups)

    def _make_unit(
        self,
        mcap_id: str,
        group: list[DenseQAWorkItem],
        bounds: NanosecondInterval | None,
    ) -> DenseQAWorkUnit:
        sources = tuple(sorted(group, key=_work_item_sort_key))
        requested = NanosecondInterval(
            start_ns=min(item.source.interval.start_ns for item in sources),
            end_ns=max(item.source.interval.end_ns for item in sources),
        )
        padded_start = requested.start_ns - self._policy.padding_ns
        padded_end = requested.end_ns + self._policy.padding_ns
        effective_start = max(padded_start, bounds.start_ns) if bounds else padded_start
        effective_end = min(padded_end, bounds.end_ns) if bounds else padded_end
        try:
            effective = NanosecondInterval(start_ns=effective_start, end_ns=effective_end)
        except ValueError as exc:
            raise DenseQAProjectionError(
                "dense padding and clipping produced an empty interval"
            ) from exc
        values: dict[str, Any] = {
            "requested_interval": requested,
            "effective_interval": effective,
            "context_truncated": (effective_start != padded_start or effective_end != padded_end),
            "source_items": sources,
            "target_interval_ids": tuple(
                _target_interval_id(item.source) for item in _unique_source_intervals(sources)
            ),
            "target_camera_ids": tuple(
                camera_id
                for camera_id in CAMERA_IDS
                if any(item.source.camera_id is camera_id for item in sources)
            ),
            "context_camera_ids": CAMERA_IDS,
            "purpose": "QA_DENSE",
            "policy_version": self._policy.version,
            "production_eligible": False,
        }
        draft = DenseQAWorkUnit.model_construct(
            unit_id=_stable_id("dense-unit", "0" * 64),
            semantic_digest="0" * 64,
            **values,
        )
        digest = semantic_sha256(dense_qa_work_unit_projection(draft))
        return DenseQAWorkUnit.model_validate(
            {
                **values,
                "unit_id": _stable_id("dense-unit", digest),
                "semantic_digest": digest,
            },
            strict=True,
        )


def dense_qa_work_unit_projection(unit: DenseQAWorkUnit) -> dict[str, Any]:
    return {
        "projection_version": DENSE_QA_WORK_PROJECTION_VERSION,
        "requested_interval": unit.requested_interval,
        "effective_interval": unit.effective_interval,
        "context_truncated": unit.context_truncated,
        "source_items": unit.source_items,
        "target_interval_ids": unit.target_interval_ids,
        "target_camera_ids": unit.target_camera_ids,
        "context_camera_ids": unit.context_camera_ids,
        "purpose": unit.purpose,
        "policy_version": unit.policy_version,
        "production_eligible": unit.production_eligible,
    }


def dense_qa_work_manifest_projection(manifest: DenseQAWorkManifest) -> dict[str, Any]:
    return {
        "projection_version": manifest.projection_version,
        "mcap_id": manifest.mcap_id,
        "coarse_result_digest": manifest.coarse_result_digest,
        "outcome": manifest.outcome,
        "items": manifest.items,
        "units": manifest.units,
        "policy_version": manifest.policy_version,
        "production_eligible": manifest.production_eligible,
    }


class DenseQAPackageRef(StrictModel):
    """Exact materialized package member consumed by dense inference."""

    package_id: OpaqueUuid
    ordinal: NonNegativeInt
    interval: NanosecondInterval
    semantic_content_sha256: Sha256Digest
    manifest_sha256: Sha256Digest


class DenseQAInputPlanRef(StrictModel):
    """Run-independent QA_DENSE input-plan identity and ordered packages."""

    input_plan_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    task: Literal["QA_DENSE"] = "QA_DENSE"
    package_ids: tuple[OpaqueUuid, ...]

    @model_validator(mode="after")
    def validate_packages(self) -> Self:
        if not self.package_ids or len(set(self.package_ids)) != len(self.package_ids):
            raise ValueError("dense input plan requires unique ordered packages")
        return self


class DenseQAOutputRef(StrictModel):
    """One authoritative enriched QA_DENSE output."""

    artifact_id: OpaqueUuid
    semantic_sha256: Sha256Digest
    enrichment_logical_key: NodeLogicalKey
    inference_id: OpaqueUuid
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    task: Literal["QA_DENSE"] = "QA_DENSE"


class CameraDenseResult(StrictModel):
    """One package-camera dense observation retaining the exact enriched claim."""

    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    camera_id: CameraId
    local_status: CameraQAStatus
    source_output: DenseQAOutputRef
    claim: EnrichedProviderClaim
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if (
            self.claim.kind is not ProviderClaimKind.QA_OBSERVATION
            or self.claim.package_id != self.package_id
            or self.claim.package_ordinal != self.package_ordinal
            or self.claim.camera_id is not self.camera_id
        ):
            raise ValueError("dense result must retain its exact enriched QA claim binding")
        if self.local_status is not _camera_status(self.claim.observation):
            raise ValueError("dense local status must derive from the provider observation")
        return self


class DenseQAUnitEvidence(StrictModel):
    """Normalized terminal evidence for one planned dense work unit."""

    unit_id: OpaqueUuid
    unit_semantic_digest: Sha256Digest
    mcap_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_id: NonEmptyString
    alignment_semantic_sha256: Sha256Digest
    package_set_id: NonEmptyString
    split_plan_digest: Sha256Digest
    member_manifest_sha256: Sha256Digest
    packages: tuple[DenseQAPackageRef, ...]
    input_plan: DenseQAInputPlanRef
    source_outputs: tuple[DenseQAOutputRef, ...]
    package_camera_results: tuple[CameraDenseResult, ...]
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if not self.packages:
            raise ValueError("dense evidence requires materialized packages")
        if tuple(package.ordinal for package in self.packages) != tuple(range(len(self.packages))):
            raise ValueError("dense packages must be stored in ordinal order")
        package_ids = tuple(package.package_id for package in self.packages)
        if len(set(package_ids)) != len(package_ids):
            raise ValueError("dense package IDs must be unique")
        if self.input_plan.package_ids != package_ids:
            raise ValueError("dense input plan does not bind the exact package sequence")
        for previous, current in zip(self.packages, self.packages[1:], strict=False):
            if (
                current.interval.start_ns <= previous.interval.start_ns
                or current.interval.end_ns <= previous.interval.end_ns
                or current.interval.start_ns > previous.interval.end_ns
            ):
                raise ValueError("dense packages must make progress without temporal gaps")

        output_ids = tuple(item.artifact_id for item in self.source_outputs)
        if output_ids != tuple(sorted(set(output_ids))):
            raise ValueError("dense output references must be unique and canonically ordered")
        if any(
            item.input_plan_id != self.input_plan.input_plan_id
            or item.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
            for item in self.source_outputs
        ):
            raise ValueError("dense outputs do not bind the exact input plan")

        expected_coordinates = tuple(
            (package.ordinal, camera_id) for package in self.packages for camera_id in CAMERA_IDS
        )
        actual_coordinates = tuple(
            (item.package_ordinal, item.camera_id) for item in self.package_camera_results
        )
        if actual_coordinates != expected_coordinates:
            raise ValueError("dense evidence requires exact package-by-six terminal coverage")
        outputs = set(self.source_outputs)
        for item in self.package_camera_results:
            package = self.packages[item.package_ordinal]
            if item.package_id != package.package_id or item.source_output not in outputs:
                raise ValueError("dense observation references undeclared lineage")
            _validate_dense_claim(item.claim, package)
        return self


class DenseQAUnitResult(StrictModel):
    """Deterministic decision over one normalized dense evidence bundle."""

    result_id: OpaqueUuid
    semantic_digest: Sha256Digest
    work_unit_id: OpaqueUuid
    work_unit_semantic_digest: Sha256Digest
    evidence: DenseQAUnitEvidence
    status: DenseQAStatus
    policy_version: SchemaVersion
    projection_version: Literal["dense-qa-result-semantic-v1"] = "dense-qa-result-semantic-v1"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (
            self.evidence.unit_id != self.work_unit_id
            or self.evidence.unit_semantic_digest != self.work_unit_semantic_digest
        ):
            raise ValueError("dense unit result does not bind its planned work unit")
        expected = _dense_status(self.evidence.package_camera_results)
        if self.status is not expected:
            raise ValueError("dense unit status does not match terminal observations")
        digest = semantic_sha256(dense_qa_unit_result_projection(self))
        if self.semantic_digest != digest or self.result_id != _stable_id(
            "dense-unit-result", digest
        ):
            raise ValueError("dense unit result identity is inconsistent")
        return self


class DenseQAResult(StrictModel):
    """Complete projection of every unit in one dense work manifest."""

    result_id: OpaqueUuid
    semantic_digest: Sha256Digest
    mcap_id: NonEmptyString
    coarse_result_digest: Sha256Digest
    work_manifest_id: OpaqueUuid
    work_manifest_digest: Sha256Digest
    units: tuple[DenseQAUnitResult, ...]
    status: DenseQAStatus
    policy_version: SchemaVersion
    projection_version: Literal["dense-qa-result-semantic-v1"] = "dense-qa-result-semantic-v1"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not self.units:
            raise ValueError("a dense QA result requires unit results")
        if tuple(item.work_unit_id for item in self.units) != tuple(
            dict.fromkeys(item.work_unit_id for item in self.units)
        ):
            raise ValueError("dense unit results must be unique")
        if any(
            item.evidence.mcap_id != self.mcap_id or item.policy_version != self.policy_version
            for item in self.units
        ):
            raise ValueError("dense unit results do not share aggregate lineage")
        expected = (
            DenseQAStatus.COMPLETE
            if all(item.status is DenseQAStatus.COMPLETE for item in self.units)
            else DenseQAStatus.INCOMPLETE
        )
        if self.status is not expected:
            raise ValueError("dense aggregate status does not match unit results")
        digest = semantic_sha256(dense_qa_result_projection(self))
        if self.semantic_digest != digest or self.result_id != _stable_id("dense-result", digest):
            raise ValueError("dense QA result identity is inconsistent")
        return self


class DenseQAProjector:
    """Validate exact terminal evidence and finalize every planned dense unit."""

    def __init__(self, policy_version: str = LOCAL_DENSE_QA_POLICY_VERSION) -> None:
        if not isinstance(policy_version, str) or not policy_version or len(policy_version) > 128:
            raise ValueError("policy_version must be a non-empty schema-version string")
        self._policy_version = policy_version

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def project(
        self,
        manifest: DenseQAWorkManifest,
        unit_evidence: Sequence[DenseQAUnitEvidence],
    ) -> DenseQAResult:
        work = _validated_manifest(manifest)
        if work.outcome is not DenseQAOutcome.DENSE_REQUIRED:
            raise DenseQAProjectionError("only DENSE_REQUIRED manifests accept dense evidence")
        if work.policy_version != self._policy_version:
            raise DenseQAProjectionError("dense projector policy does not match the manifest")
        evidence_by_unit: dict[str, DenseQAUnitEvidence] = {}
        for value in unit_evidence:
            evidence = _validated_unit_evidence(value)
            if evidence.unit_id in evidence_by_unit:
                raise DenseQAProjectionError("dense evidence contains a duplicate work unit")
            evidence_by_unit[evidence.unit_id] = evidence
        expected_ids = {unit.unit_id for unit in work.units}
        if set(evidence_by_unit) != expected_ids:
            raise DenseQAProjectionError("dense evidence must cover every planned unit exactly")

        unit_results = tuple(
            self._project_unit(unit, evidence_by_unit[unit.unit_id], work.mcap_id)
            for unit in work.units
        )
        status = (
            DenseQAStatus.COMPLETE
            if all(item.status is DenseQAStatus.COMPLETE for item in unit_results)
            else DenseQAStatus.INCOMPLETE
        )
        values: dict[str, Any] = {
            "mcap_id": work.mcap_id,
            "coarse_result_digest": work.coarse_result_digest,
            "work_manifest_id": work.manifest_id,
            "work_manifest_digest": work.semantic_digest,
            "units": unit_results,
            "status": status,
            "policy_version": self._policy_version,
            "projection_version": DENSE_QA_RESULT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = DenseQAResult.model_construct(
            result_id=_stable_id("dense-result", "0" * 64),
            semantic_digest="0" * 64,
            **values,
        )
        digest = semantic_sha256(dense_qa_result_projection(draft))
        return DenseQAResult.model_validate(
            {
                **values,
                "result_id": _stable_id("dense-result", digest),
                "semantic_digest": digest,
            },
            strict=True,
        )

    def _project_unit(
        self,
        unit: DenseQAWorkUnit,
        evidence: DenseQAUnitEvidence,
        mcap_id: str,
    ) -> DenseQAUnitResult:
        if evidence.unit_semantic_digest != unit.semantic_digest or evidence.mcap_id != mcap_id:
            raise DenseQAProjectionError("dense evidence does not bind its planned unit")
        if (
            evidence.packages[0].interval.start_ns != unit.effective_interval.start_ns
            or evidence.packages[-1].interval.end_ns != unit.effective_interval.end_ns
        ):
            raise DenseQAProjectionError("dense packages do not cover the exact work interval")
        status = _dense_status(evidence.package_camera_results)
        values: dict[str, Any] = {
            "work_unit_id": unit.unit_id,
            "work_unit_semantic_digest": unit.semantic_digest,
            "evidence": evidence,
            "status": status,
            "policy_version": self._policy_version,
            "projection_version": DENSE_QA_RESULT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = DenseQAUnitResult.model_construct(
            result_id=_stable_id("dense-unit-result", "0" * 64),
            semantic_digest="0" * 64,
            **values,
        )
        digest = semantic_sha256(dense_qa_unit_result_projection(draft))
        return DenseQAUnitResult.model_validate(
            {
                **values,
                "result_id": _stable_id("dense-unit-result", digest),
                "semantic_digest": digest,
            },
            strict=True,
        )


def dense_qa_unit_result_projection(result: DenseQAUnitResult) -> dict[str, Any]:
    return {
        "projection_version": result.projection_version,
        "work_unit_id": result.work_unit_id,
        "work_unit_semantic_digest": result.work_unit_semantic_digest,
        "evidence": result.evidence,
        "status": result.status,
        "policy_version": result.policy_version,
        "production_eligible": result.production_eligible,
    }


def dense_qa_result_projection(result: DenseQAResult) -> dict[str, Any]:
    return {
        "projection_version": result.projection_version,
        "mcap_id": result.mcap_id,
        "coarse_result_digest": result.coarse_result_digest,
        "work_manifest_id": result.work_manifest_id,
        "work_manifest_digest": result.work_manifest_digest,
        "units": result.units,
        "status": result.status,
        "policy_version": result.policy_version,
        "production_eligible": result.production_eligible,
    }


def _validated_coarse_result(value: CoarseQAResult) -> CoarseQAResult:
    if not isinstance(value, CoarseQAResult):
        raise DenseQAProjectionError("coarse_result must be a CoarseQAResult")
    try:
        return CoarseQAResult.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DenseQAProjectionError("coarse_result failed immutable contract validation") from exc


def _validated_interval(
    value: NanosecondInterval | None,
) -> NanosecondInterval | None:
    if value is None:
        return None
    if not isinstance(value, NanosecondInterval):
        raise DenseQAProjectionError("recording_interval must be a NanosecondInterval")
    try:
        return NanosecondInterval.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DenseQAProjectionError("recording_interval failed validation") from exc


def _validated_manifest(value: DenseQAWorkManifest) -> DenseQAWorkManifest:
    if not isinstance(value, DenseQAWorkManifest):
        raise DenseQAProjectionError("manifest must be a DenseQAWorkManifest")
    try:
        return DenseQAWorkManifest.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DenseQAProjectionError("manifest failed immutable contract validation") from exc


def _validated_unit_evidence(value: DenseQAUnitEvidence) -> DenseQAUnitEvidence:
    if not isinstance(value, DenseQAUnitEvidence):
        raise DenseQAProjectionError("unit_evidence must contain DenseQAUnitEvidence values")
    try:
        return DenseQAUnitEvidence.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DenseQAProjectionError("dense evidence failed immutable contract validation") from exc


def _observation_ref(item: CameraCoarseResult) -> CoarseQAObservationRef:
    interval = item.claim.interval
    if interval is None:
        raise DenseQAProjectionError("coarse observation is missing its exact interval")
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


def _validate_dense_claim(
    claim: EnrichedProviderClaim,
    package: DenseQAPackageRef,
) -> None:
    interval = claim.interval
    if interval is None:
        raise ValueError("dense QA claims require an exact package interval")
    if interval.start_ns < package.interval.start_ns or interval.end_ns > package.interval.end_ns:
        raise ValueError("dense QA claim interval lies outside its package")
    if claim.observation in {ProviderObservation.GOOD, ProviderObservation.DEGRADED} and not (
        claim.evidence
    ):
        raise ValueError("observed GOOD/DEGRADED dense claims require evidence")
    for evidence in claim.evidence:
        if (
            evidence.package_id != package.package_id
            or evidence.package_ordinal != package.ordinal
            or evidence.camera_id is not claim.camera_id
            or evidence.package_semantic_content_sha256 != package.semantic_content_sha256
            or evidence.package_manifest_sha256 != package.manifest_sha256
        ):
            raise ValueError("dense claim evidence does not resolve to its package/camera")


def _camera_status(observation: ProviderObservation) -> CameraQAStatus:
    try:
        return {
            ProviderObservation.GOOD: CameraQAStatus.GOOD,
            ProviderObservation.DEGRADED: CameraQAStatus.DEGRADED,
            ProviderObservation.UNUSABLE: CameraQAStatus.UNUSABLE,
            ProviderObservation.UNKNOWN: CameraQAStatus.UNKNOWN,
        }[observation]
    except KeyError as exc:
        raise ValueError("unsupported QA observation") from exc


def _dense_status(results: Sequence[CameraDenseResult]) -> DenseQAStatus:
    return (
        DenseQAStatus.COMPLETE
        if results and all(item.local_status is CameraQAStatus.GOOD for item in results)
        else DenseQAStatus.INCOMPLETE
    )


def _work_item_sort_key(item: DenseQAWorkItem) -> tuple[int, int]:
    return item.source.package_ordinal, CAMERA_IDS.index(item.source.camera_id)


def _unique_source_intervals(
    items: Sequence[DenseQAWorkItem],
) -> tuple[DenseQAWorkItem, ...]:
    selected: dict[tuple[str, int, int], DenseQAWorkItem] = {}
    for item in sorted(
        items,
        key=lambda value: (
            value.source.interval.start_ns,
            value.source.interval.end_ns,
            value.source.package_id,
            CAMERA_IDS.index(value.source.camera_id),
        ),
    ):
        key = (
            item.source.package_id,
            item.source.interval.start_ns,
            item.source.interval.end_ns,
        )
        selected.setdefault(key, item)
    return tuple(selected.values())


def _target_interval_id(source: CoarseQAObservationRef) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            "robata:dense-target-interval:"
            f"{source.package_id}:{source.interval.start_ns}:{source.interval.end_ns}",
        )
    )


def _stable_id(kind: str, digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{kind}:{digest}"))


__all__ = [
    "DENSE_QA_RESULT_PROJECTION_VERSION",
    "DENSE_QA_WORK_PROJECTION_VERSION",
    "LOCAL_DENSE_QA_POLICY_VERSION",
    "CameraDenseResult",
    "CoarseQAObservationRef",
    "DenseQAInputPlanRef",
    "DenseQAOutcome",
    "DenseQAOutputRef",
    "DenseQAPackageRef",
    "DenseQAPlanner",
    "DenseQAPlanningPolicy",
    "DenseQAProjectionError",
    "DenseQAProjector",
    "DenseQAResult",
    "DenseQAStatus",
    "DenseQAUnitEvidence",
    "DenseQAUnitResult",
    "DenseQAWorkItem",
    "DenseQAWorkManifest",
    "DenseQAWorkUnit",
    "dense_qa_result_projection",
    "dense_qa_unit_result_projection",
    "dense_qa_work_manifest_projection",
    "dense_qa_work_unit_projection",
]
