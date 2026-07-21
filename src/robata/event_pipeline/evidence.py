"""Canonical ACTION_EVIDENCE projection plus the fail-closed legacy surface.

The projector in this module does not sample frames or invoke a provider. It
accepts only the already-enriched outputs for one immutable ACTION_EVIDENCE
input plan, proves their ordered call-part and package bindings, and emits a
deterministic, non-promotable six-camera result.

Provider/run locators are retained for audit where useful, but never enter the
logical identity projections below. Stable identity is derived from the
candidate, window/package/input-plan semantics, selected output content,
selection policy identity, and normalized claim/evidence coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.pipeline import (
    ActionEvidence,
    CandidateEvent,
    TemporalVisualPackage,
)
from robata.contracts.temporal import TemporalPackageSet, TemporalPackageSetMember
from robata.event_pipeline.candidate import CanonicalCandidateEvent
from robata.inference.enrichment import (
    EnrichedEvidenceReference,
    EnrichedProviderClaim,
    OrchestratorEnrichedOutput,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.inference.input_plan import InferenceCallPart, InferenceInputPlan
from robata.inference.models import VisionTask

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

ACTION_EVIDENCE_RESULT_PROJECTION_VERSION = "action-evidence-result-semantic-v1"
ACTION_OBSERVATION_PROJECTION_VERSION = "action-observation-semantic-v1"
CROSS_VIEW_HYPOTHESIS_PROJECTION_VERSION = "cross-view-hypothesis-semantic-v1"


class ActionEvidenceProjectionError(ValueError):
    """ACTION_EVIDENCE output cannot be admitted into canonical evidence."""


class ActionEvidenceOutcome(StrEnum):
    """Local, non-production outcome for one candidate evidence projection."""

    SUPPORTED = "SUPPORTED"
    NO_ACTION = "NO_ACTION"
    INDETERMINATE = "INDETERMINATE"


class ActionEvidencePackageRef(StrictModel):
    """One materialized ACTION_DENSE package retained by the result."""

    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    interval: NanosecondInterval
    semantic_content_sha256: Sha256Digest
    manifest_sha256: Sha256Digest


class ActionEvidenceOutputRef(StrictModel):
    """Audit lineage for one selected, enriched call-part output.

    The UUID and enrichment key fields are provenance locators only. See
    :func:`action_evidence_output_semantic_projection` for the intentionally
    smaller stable identity projection.
    """

    part_ordinal: NonNegativeInt
    part_semantic_sha256: Sha256Digest
    source_inference_id: OpaqueUuid
    source_artifact_id: OpaqueUuid
    selected_output_sha256: Sha256Digest
    selection_decision_logical_key: NodeLogicalKey
    enrichment_logical_key: NodeLogicalKey


class ActionEvidenceClaimRef(StrictModel):
    """One exact enriched claim locator within a selected call-part output."""

    output: ActionEvidenceOutputRef
    claim_ordinal: NonNegativeInt
    claim_id: OpaqueUuid


class NormalizedActionObservation(StrictModel):
    """One package-camera observation normalized from authoritative enrichment."""

    source_action_observation_logical_key: NodeLogicalKey
    source: ActionEvidenceClaimRef
    package_id: OpaqueUuid
    package_ordinal: NonNegativeInt
    camera_id: CameraId
    interval: NanosecondInterval | None
    label: str | None
    observation: ProviderObservation
    evidence: tuple[EnrichedEvidenceReference, ...]
    model_reported_score: UnitInterval | None
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.observation not in _ACTION_OBSERVATIONS:
            raise ValueError("invalid ACTION_OBSERVATION status")
        if self.observation in _POSITIVE_OR_NEGATIVE_OBSERVATIONS and (
            self.interval is None or not self.evidence
        ):
            raise ValueError("SUPPORTING, PARTIAL, and NO_EVENT require interval and evidence")
        if self.observation is ProviderObservation.MISSING and (
            self.interval is not None or self.evidence
        ):
            raise ValueError("MISSING cannot assert interval or evidence")
        expected_evidence = tuple(sorted(self.evidence, key=_evidence_sort_key))
        if self.evidence != expected_evidence:
            raise ValueError("action observation evidence must be canonically ordered")
        digest = semantic_sha256(action_observation_semantic_projection(self))
        if self.source_action_observation_logical_key != f"action-observation:{digest}":
            raise ValueError("action observation logical identity is inconsistent")
        return self


class CameraActionEvidence(StrictModel):
    """One canonical camera slot retaining observations from every call part."""

    camera_id: CameraId
    observations: tuple[NormalizedActionObservation, ...]
    outcome: ActionEvidenceOutcome
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_slot(self) -> Self:
        if any(item.camera_id is not self.camera_id for item in self.observations):
            raise ValueError("camera evidence slot contains a foreign camera observation")
        expected = tuple(sorted(self.observations, key=_action_observation_sort_key))
        if self.observations != expected or len(
            {item.source_action_observation_logical_key for item in self.observations}
        ) != len(self.observations):
            raise ValueError("camera observations must be unique and canonically ordered")
        if self.outcome is not _camera_outcome(self.observations):
            raise ValueError("camera evidence outcome does not match its observations")
        return self


class NormalizedCrossViewHypothesis(StrictModel):
    """One cross-view action hypothesis with authoritative evidence references."""

    source_cross_view_logical_key: NodeLogicalKey
    source: ActionEvidenceClaimRef
    interval: NanosecondInterval
    label: str | None
    observation: Literal[ProviderObservation.SUPPORTING, ProviderObservation.PARTIAL]
    evidence: tuple[EnrichedEvidenceReference, ...]
    model_reported_score: UnitInterval | None
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_hypothesis(self) -> Self:
        if not self.evidence:
            raise ValueError("cross-view hypotheses require evidence")
        expected_evidence = tuple(sorted(self.evidence, key=_evidence_sort_key))
        if self.evidence != expected_evidence:
            raise ValueError("cross-view evidence must be canonically ordered")
        digest = semantic_sha256(cross_view_hypothesis_semantic_projection(self))
        if self.source_cross_view_logical_key != f"cross-view-hypothesis:{digest}":
            raise ValueError("cross-view hypothesis logical identity is inconsistent")
        return self


class ActionEvidenceResult(StrictModel):
    """Canonical local ACTION_EVIDENCE projection for one candidate."""

    task: Literal[VisionTask.ACTION_EVIDENCE] = VisionTask.ACTION_EVIDENCE
    candidate_event_id: OpaqueUuid
    candidate_logical_key: NodeLogicalKey
    candidate_label: str
    candidate_effective_interval: NanosecondInterval
    requested_dense_interval: NanosecondInterval
    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    window_semantic_sha256: Sha256Digest
    window_interval: NanosecondInterval
    package_set_id: OpaqueUuid
    split_plan_digest: Sha256Digest
    member_manifest_sha256: Sha256Digest
    packages: tuple[ActionEvidencePackageRef, ...]
    input_plan_id: OpaqueUuid
    input_plan_semantic_sha256: Sha256Digest
    source_outputs: tuple[ActionEvidenceOutputRef, ...]
    camera_evidence: SixCameraMap[CameraActionEvidence]
    cross_view_hypotheses: tuple[NormalizedCrossViewHypothesis, ...]
    outcome: ActionEvidenceOutcome
    projection_version: Literal["action-evidence-result-semantic-v1"] = (
        "action-evidence-result-semantic-v1"
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if tuple(item.package_ordinal for item in self.packages) != tuple(
            range(len(self.packages))
        ):
            raise ValueError("action evidence packages must be stored in ordinal order")
        if len({item.package_id for item in self.packages}) != len(self.packages):
            raise ValueError("action evidence package IDs must be unique")
        if tuple(item.part_ordinal for item in self.source_outputs) != tuple(
            range(len(self.source_outputs))
        ):
            raise ValueError("action evidence outputs must be stored in call-part order")
        if len({item.part_semantic_sha256 for item in self.source_outputs}) != len(
            self.source_outputs
        ):
            raise ValueError("action evidence output part identities must be unique")
        if (
            self.candidate_effective_interval.start_ns < self.window_interval.start_ns
            or self.candidate_effective_interval.end_ns > self.window_interval.end_ns
            or self.window_interval.start_ns < self.requested_dense_interval.start_ns
            or self.window_interval.end_ns > self.requested_dense_interval.end_ns
        ):
            raise ValueError("action evidence candidate/window bounds are inconsistent")
        for camera_id in CAMERA_IDS:
            if self.camera_evidence[camera_id].camera_id is not camera_id:
                raise ValueError("action evidence camera slot identity is inconsistent")
        _validate_result_claim_lineage(self)
        expected_hypotheses = tuple(sorted(self.cross_view_hypotheses, key=_cross_view_sort_key))
        if self.cross_view_hypotheses != expected_hypotheses or len(
            {item.source_cross_view_logical_key for item in self.cross_view_hypotheses}
        ) != len(self.cross_view_hypotheses):
            raise ValueError("cross-view hypotheses must be unique and canonically ordered")
        if self.outcome is not _result_outcome(self.camera_evidence, self.cross_view_hypotheses):
            raise ValueError("action evidence outcome does not match normalized evidence")
        digest = semantic_sha256(action_evidence_result_semantic_projection(self))
        if self.semantic_sha256 != digest or self.logical_key != f"action-evidence:{digest}":
            raise ValueError("action evidence result logical identity is inconsistent")
        return self


_ACTION_OBSERVATIONS = frozenset(
    {
        ProviderObservation.SUPPORTING,
        ProviderObservation.PARTIAL,
        ProviderObservation.NO_EVENT,
        ProviderObservation.OCCLUDED,
        ProviderObservation.UNUSABLE,
        ProviderObservation.MISSING,
    }
)
_POSITIVE_OBSERVATIONS = frozenset({ProviderObservation.SUPPORTING, ProviderObservation.PARTIAL})
_POSITIVE_OR_NEGATIVE_OBSERVATIONS = _POSITIVE_OBSERVATIONS | frozenset(
    {ProviderObservation.NO_EVENT}
)


def action_evidence_output_semantic_projection(
    output: ActionEvidenceOutputRef,
) -> dict[str, object]:
    """Return stable selected-output lineage, excluding locator UUIDs and keys."""

    return {
        "part_ordinal": output.part_ordinal,
        "part_semantic_sha256": output.part_semantic_sha256,
        "selected_output_sha256": output.selected_output_sha256,
        "selection_decision_logical_key": output.selection_decision_logical_key,
    }


def action_observation_semantic_projection(
    observation: NormalizedActionObservation,
) -> dict[str, object]:
    """Return the run-independent identity projection of one camera claim."""

    return {
        "semantic_projection_version": ACTION_OBSERVATION_PROJECTION_VERSION,
        "source_output": action_evidence_output_semantic_projection(observation.source.output),
        "claim_ordinal": observation.source.claim_ordinal,
        "package_ordinal": observation.package_ordinal,
        "camera_id": observation.camera_id,
        "interval": _interval_projection(observation.interval),
        "label": observation.label,
        "observation": observation.observation,
        "evidence_coordinates": [
            _evidence_semantic_projection(item) for item in observation.evidence
        ],
        "model_reported_score": observation.model_reported_score,
    }


def cross_view_hypothesis_semantic_projection(
    hypothesis: NormalizedCrossViewHypothesis,
) -> dict[str, object]:
    """Return the run-independent identity projection of one cross-view claim."""

    return {
        "semantic_projection_version": CROSS_VIEW_HYPOTHESIS_PROJECTION_VERSION,
        "source_output": action_evidence_output_semantic_projection(hypothesis.source.output),
        "claim_ordinal": hypothesis.source.claim_ordinal,
        "interval": hypothesis.interval.model_dump(mode="json"),
        "label": hypothesis.label,
        "observation": hypothesis.observation,
        "evidence_coordinates": [
            _evidence_semantic_projection(item) for item in hypothesis.evidence
        ],
        "model_reported_score": hypothesis.model_reported_score,
    }


def action_evidence_result_semantic_projection(
    result: ActionEvidenceResult,
) -> dict[str, object]:
    """Return the stable candidate/window/package/action result projection."""

    return {
        "semantic_projection_version": result.projection_version,
        "candidate_logical_key": result.candidate_logical_key,
        "candidate_label": result.candidate_label,
        "candidate_effective_interval": result.candidate_effective_interval.model_dump(mode="json"),
        "requested_dense_interval": result.requested_dense_interval.model_dump(mode="json"),
        "source_content_sha256": result.source_content_sha256,
        "camera_mapping_semantic_sha256": result.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": result.alignment_semantic_sha256,
        "window_semantic_sha256": result.window_semantic_sha256,
        "window_interval": result.window_interval.model_dump(mode="json"),
        "split_plan_digest": result.split_plan_digest,
        "member_manifest_sha256": result.member_manifest_sha256,
        "package_members": [
            {
                "package_ordinal": item.package_ordinal,
                "interval": item.interval.model_dump(mode="json"),
                "semantic_content_sha256": item.semantic_content_sha256,
            }
            for item in result.packages
        ],
        "input_plan_semantic_sha256": result.input_plan_semantic_sha256,
        "source_outputs": [
            action_evidence_output_semantic_projection(item) for item in result.source_outputs
        ],
        "camera_observation_logical_keys": {
            camera_id.value: [
                item.source_action_observation_logical_key
                for item in result.camera_evidence[camera_id].observations
            ]
            for camera_id in CAMERA_IDS
        },
        "camera_outcomes": {
            camera_id.value: result.camera_evidence[camera_id].outcome for camera_id in CAMERA_IDS
        },
        "cross_view_hypothesis_logical_keys": [
            item.source_cross_view_logical_key for item in result.cross_view_hypotheses
        ],
        "outcome": result.outcome,
        "production_eligible": result.production_eligible,
    }


class ActionEvidenceProjector:
    """Normalize one candidate's ordered ACTION_EVIDENCE call-part outputs."""

    policy_version: SchemaVersion = ACTION_EVIDENCE_RESULT_PROJECTION_VERSION

    def project(
        self,
        *,
        input_plan: InferenceInputPlan,
        package_set: TemporalPackageSet,
        candidate: CanonicalCandidateEvent,
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
    ) -> ActionEvidenceResult:
        _validate_projection_lineage(
            input_plan=input_plan,
            package_set=package_set,
            candidate=candidate,
            enriched_outputs=enriched_outputs,
        )
        packages = tuple(_package_ref(member) for member in package_set.members)
        output_refs: list[ActionEvidenceOutputRef] = []
        by_camera: dict[CameraId, list[NormalizedActionObservation]] = {
            camera_id: [] for camera_id in CAMERA_IDS
        }
        hypotheses: list[NormalizedCrossViewHypothesis] = []

        for part, output in zip(input_plan.call_plan.parts, enriched_outputs, strict=True):
            output_ref = _output_ref(part, output)
            output_refs.append(output_ref)
            visible_items = {
                item.provider_item_ordinal: item
                for item in input_plan.rendered_items[
                    part.start_item_ordinal : part.end_item_ordinal_exclusive
                ]
            }
            for claim in output.claims:
                _validate_claim_evidence_visible(claim, visible_items)
                if claim.kind is ProviderClaimKind.ACTION_OBSERVATION:
                    observation = _normalize_action_observation(
                        claim=claim,
                        output_ref=output_ref,
                        package_set=package_set,
                    )
                    by_camera[observation.camera_id].append(observation)
                elif claim.kind is ProviderClaimKind.CROSS_VIEW_HYPOTHESIS:
                    hypotheses.append(
                        _normalize_cross_view_hypothesis(
                            claim=claim,
                            output_ref=output_ref,
                            package_set=package_set,
                        )
                    )
                else:
                    raise ActionEvidenceProjectionError(
                        "non-action claim reached ACTION_EVIDENCE projector"
                    )

        camera_evidence = SixCameraMap[CameraActionEvidence](
            {
                camera_id: CameraActionEvidence(
                    camera_id=camera_id,
                    observations=tuple(
                        sorted(by_camera[camera_id], key=_action_observation_sort_key)
                    ),
                    outcome=_camera_outcome(by_camera[camera_id]),
                )
                for camera_id in CAMERA_IDS
            }
        )
        normalized_hypotheses = tuple(sorted(hypotheses, key=_cross_view_sort_key))
        outcome = _result_outcome(camera_evidence, normalized_hypotheses)
        values: dict[str, Any] = {
            "task": VisionTask.ACTION_EVIDENCE,
            "candidate_event_id": candidate.candidate_event_id,
            "candidate_logical_key": candidate.candidate_logical_key,
            "candidate_label": candidate.label,
            "candidate_effective_interval": candidate.effective_interval,
            "requested_dense_interval": candidate.requested_dense_interval,
            "mcap_id": candidate.mcap_id,
            "source_content_sha256": candidate.source_content_sha256,
            "camera_mapping_semantic_sha256": candidate.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": candidate.alignment_semantic_sha256,
            "window_semantic_sha256": package_set.lineage.window_semantic_sha256,
            "window_interval": NanosecondInterval(
                start_ns=package_set.start_ns, end_ns=package_set.end_ns
            ),
            "package_set_id": package_set.package_set_id,
            "split_plan_digest": package_set.split_plan_digest,
            "member_manifest_sha256": package_set.member_manifest_sha256,
            "packages": packages,
            "input_plan_id": input_plan.input_plan_id,
            "input_plan_semantic_sha256": input_plan.semantic_sha256,
            "source_outputs": tuple(output_refs),
            "camera_evidence": camera_evidence,
            "cross_view_hypotheses": normalized_hypotheses,
            "outcome": outcome,
            "projection_version": self.policy_version,
            "production_eligible": False,
        }
        draft = ActionEvidenceResult.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"action-evidence:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(action_evidence_result_semantic_projection(draft))
        return ActionEvidenceResult.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"action-evidence:{digest}",
            },
            strict=True,
        )


def _validate_projection_lineage(
    *,
    input_plan: InferenceInputPlan,
    package_set: TemporalPackageSet,
    candidate: CanonicalCandidateEvent,
    enriched_outputs: Sequence[OrchestratorEnrichedOutput],
) -> None:
    if input_plan.subject.task is not VisionTask.ACTION_EVIDENCE:
        raise ActionEvidenceProjectionError("projector requires ACTION_EVIDENCE input plan")
    if input_plan.request_catalog.task is not VisionTask.ACTION_EVIDENCE:
        raise ActionEvidenceProjectionError("request catalog task is not ACTION_EVIDENCE")
    if (
        package_set.mcap_id != candidate.mcap_id
        or package_set.lineage.source_content_sha256 != candidate.source_content_sha256
        or package_set.lineage.camera_mapping_semantic_sha256
        != candidate.camera_mapping_semantic_sha256
        or package_set.lineage.alignment_semantic_sha256 != candidate.alignment_semantic_sha256
        or package_set.requested_start_ns != candidate.requested_dense_interval.start_ns
        or package_set.requested_end_ns != candidate.requested_dense_interval.end_ns
        or candidate.effective_interval.start_ns < package_set.start_ns
        or candidate.effective_interval.end_ns > package_set.end_ns
    ):
        raise ActionEvidenceProjectionError(
            "candidate does not match the ACTION_DENSE package window"
        )
    subject = input_plan.subject.packages
    if len(subject) != len(package_set.members) or any(
        planned.package_id != member.package_id
        or planned.ordinal != member.ordinal
        or planned.semantic_content_sha256 != member.package_semantic_content_sha256
        or planned.manifest_bytes_sha256 != member.package_manifest_sha256
        for planned, member in zip(subject, package_set.members, strict=True)
    ):
        raise ActionEvidenceProjectionError(
            "ACTION_EVIDENCE input plan does not bind exact package members"
        )
    parts = input_plan.call_plan.parts
    if not enriched_outputs or len(enriched_outputs) != len(parts):
        raise ActionEvidenceProjectionError(
            "ACTION_EVIDENCE projection requires one enriched output per call part"
        )
    for part, output in zip(parts, enriched_outputs, strict=True):
        if (
            output.task is not VisionTask.ACTION_EVIDENCE
            or output.abstained
            or output.input_plan_id != input_plan.input_plan_id
            or output.input_plan_semantic_sha256 != input_plan.semantic_sha256
            or output.request_catalog_id != input_plan.request_catalog.request_catalog_id
            or output.request_catalog_sha256 != input_plan.request_catalog.semantic_sha256
            or output.authority.mcap_id != package_set.mcap_id
            or output.authority.camera_mapping_run_id != package_set.camera_mapping_run_id
            or output.authority.alignment_id != package_set.alignment_id
        ):
            raise ActionEvidenceProjectionError(
                "ACTION_EVIDENCE enrichment is foreign to the candidate input plan"
            )
        expected_coordinates = {
            (item.package_ordinal, item.camera_ordinal)
            for item in input_plan.rendered_items[
                part.start_item_ordinal : part.end_item_ordinal_exclusive
            ]
        }
        actual_coordinates = [
            (claim.package_ordinal, CAMERA_IDS.index(claim.camera_id))
            for claim in output.claims
            if claim.kind is ProviderClaimKind.ACTION_OBSERVATION and claim.camera_id is not None
        ]
        if (
            len(actual_coordinates) != len(set(actual_coordinates))
            or set(actual_coordinates) != expected_coordinates
        ):
            raise ActionEvidenceProjectionError(
                "each call part must cover its exact package-camera coordinates"
            )


def _output_ref(
    part: InferenceCallPart,
    output: OrchestratorEnrichedOutput,
) -> ActionEvidenceOutputRef:
    selected = output.selected_attempt
    return ActionEvidenceOutputRef(
        part_ordinal=part.ordinal,
        part_semantic_sha256=part.part_semantic_sha256,
        source_inference_id=output.authority.inference_id,
        source_artifact_id=output.artifact_id,
        selected_output_sha256=selected.output_sha256,
        selection_decision_logical_key=selected.selection_decision_logical_key,
        enrichment_logical_key=output.enrichment_logical_key,
    )


def _claim_ref(
    output_ref: ActionEvidenceOutputRef,
    claim: EnrichedProviderClaim,
) -> ActionEvidenceClaimRef:
    return ActionEvidenceClaimRef(
        output=output_ref,
        claim_ordinal=claim.claim_ordinal,
        claim_id=claim.claim_id,
    )


def _normalize_action_observation(
    *,
    claim: EnrichedProviderClaim,
    output_ref: ActionEvidenceOutputRef,
    package_set: TemporalPackageSet,
) -> NormalizedActionObservation:
    if (
        claim.kind is not ProviderClaimKind.ACTION_OBSERVATION
        or claim.package_id is None
        or claim.package_ordinal is None
        or claim.camera_id is None
        or claim.package_ordinal >= len(package_set.members)
    ):
        raise ActionEvidenceProjectionError("action observation lacks package-camera binding")
    member = package_set.members[claim.package_ordinal]
    if claim.package_id != member.package_id:
        raise ActionEvidenceProjectionError("action observation cites a foreign package")
    interval = _claim_interval(claim)
    _validate_action_shape(claim, interval)
    if interval is not None:
        _require_interval_inside_member(interval, member)
    evidence = tuple(sorted(claim.evidence, key=_evidence_sort_key))
    source = _claim_ref(output_ref, claim)
    label = _normalized_label(claim.label)
    values: dict[str, Any] = {
        "source": source,
        "package_id": claim.package_id,
        "package_ordinal": claim.package_ordinal,
        "camera_id": claim.camera_id,
        "interval": interval,
        "label": label,
        "observation": claim.observation,
        "evidence": evidence,
        "model_reported_score": (
            claim.model_reported_confidence.value
            if claim.model_reported_confidence is not None
            else None
        ),
        "production_eligible": False,
    }
    draft = NormalizedActionObservation.model_construct(
        source_action_observation_logical_key=f"action-observation:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(action_observation_semantic_projection(draft))
    return NormalizedActionObservation.model_validate(
        {
            **values,
            "source_action_observation_logical_key": f"action-observation:{digest}",
        },
        strict=True,
    )


def _normalize_cross_view_hypothesis(
    *,
    claim: EnrichedProviderClaim,
    output_ref: ActionEvidenceOutputRef,
    package_set: TemporalPackageSet,
) -> NormalizedCrossViewHypothesis:
    if (
        claim.kind is not ProviderClaimKind.CROSS_VIEW_HYPOTHESIS
        or claim.package_id is not None
        or claim.package_ordinal is not None
        or claim.camera_id is not None
        or claim.interval is None
        or claim.observation not in _POSITIVE_OBSERVATIONS
        or not claim.evidence
    ):
        raise ActionEvidenceProjectionError("cross-view hypothesis has invalid shape")
    interval = NanosecondInterval(start_ns=claim.interval.start_ns, end_ns=claim.interval.end_ns)
    _require_interval_inside_window(interval, package_set)
    evidence = tuple(sorted(claim.evidence, key=_evidence_sort_key))
    source = _claim_ref(output_ref, claim)
    label = _normalized_label(claim.label)
    values: dict[str, Any] = {
        "source": source,
        "interval": interval,
        "label": label,
        "observation": claim.observation,
        "evidence": evidence,
        "model_reported_score": (
            claim.model_reported_confidence.value
            if claim.model_reported_confidence is not None
            else None
        ),
        "production_eligible": False,
    }
    draft = NormalizedCrossViewHypothesis.model_construct(
        source_cross_view_logical_key=f"cross-view-hypothesis:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(cross_view_hypothesis_semantic_projection(draft))
    return NormalizedCrossViewHypothesis.model_validate(
        {
            **values,
            "source_cross_view_logical_key": f"cross-view-hypothesis:{digest}",
        },
        strict=True,
    )


def _validate_action_shape(
    claim: EnrichedProviderClaim,
    interval: NanosecondInterval | None,
) -> None:
    if claim.observation not in _ACTION_OBSERVATIONS:
        raise ActionEvidenceProjectionError("invalid ACTION_OBSERVATION status")
    if claim.observation in _POSITIVE_OR_NEGATIVE_OBSERVATIONS and (
        interval is None or not claim.evidence
    ):
        raise ActionEvidenceProjectionError(
            "SUPPORTING, PARTIAL, and NO_EVENT require interval and evidence"
        )
    if claim.observation is ProviderObservation.MISSING and (
        interval is not None or claim.evidence
    ):
        raise ActionEvidenceProjectionError("MISSING cannot assert interval or evidence")


def _validate_claim_evidence_visible(
    claim: EnrichedProviderClaim,
    visible_items: Mapping[int, object],
) -> None:
    for evidence in claim.evidence:
        rendered = visible_items.get(evidence.provider_item_ordinal)
        if rendered is None:
            raise ActionEvidenceProjectionError("claim evidence is outside its selected call part")
        if (
            getattr(rendered, "package_id", None) != evidence.package_id
            or getattr(rendered, "package_ordinal", None) != evidence.package_ordinal
            or getattr(rendered, "camera_id", None) is not evidence.camera_id
            or getattr(rendered, "camera_ordinal", None) != evidence.camera_ordinal
            or getattr(rendered, "frame_id", None) != evidence.frame_id
            or getattr(rendered, "frame_ordinal", None) != evidence.frame_ordinal
            or getattr(rendered, "aligned_timestamp_ns", None) != evidence.aligned_timestamp_ns
            or getattr(rendered, "source_timestamp_ns", None) != evidence.source_timestamp_ns
            or getattr(rendered, "source_artifact_sha256", None) != evidence.source_artifact_sha256
        ):
            raise ActionEvidenceProjectionError(
                "claim evidence differs from authoritative input-plan coordinates"
            )


def _validate_result_claim_lineage(result: ActionEvidenceResult) -> None:
    packages = {item.package_ordinal: item for item in result.packages}
    outputs = set(result.source_outputs)
    for camera_id in CAMERA_IDS:
        for observation in result.camera_evidence[camera_id].observations:
            package = packages.get(observation.package_ordinal)
            if (
                package is None
                or observation.package_id != package.package_id
                or observation.source.output not in outputs
                or any(
                    item.package_id != package.package_id
                    or item.package_ordinal != package.package_ordinal
                    or item.camera_id is not camera_id
                    or item.package_semantic_content_sha256 != package.semantic_content_sha256
                    for item in observation.evidence
                )
            ):
                raise ValueError("action observation references undeclared lineage")
            if observation.interval is not None and not _interval_inside(
                observation.interval, package.interval
            ):
                raise ValueError("action observation interval lies outside its package")
    for hypothesis in result.cross_view_hypotheses:
        if hypothesis.source.output not in outputs or not _interval_inside(
            hypothesis.interval, result.window_interval
        ):
            raise ValueError("cross-view hypothesis references undeclared lineage")
        for evidence in hypothesis.evidence:
            package = packages.get(evidence.package_ordinal)
            if (
                package is None
                or evidence.package_id != package.package_id
                or evidence.package_semantic_content_sha256 != package.semantic_content_sha256
            ):
                raise ValueError("cross-view evidence references an undeclared package")


def _package_ref(member: TemporalPackageSetMember) -> ActionEvidencePackageRef:
    return ActionEvidencePackageRef(
        package_id=member.package_id,
        package_ordinal=member.ordinal,
        interval=NanosecondInterval(start_ns=member.start_ns, end_ns=member.end_ns),
        semantic_content_sha256=member.package_semantic_content_sha256,
        manifest_sha256=member.package_manifest_sha256,
    )


def _claim_interval(claim: EnrichedProviderClaim) -> NanosecondInterval | None:
    if claim.interval is None:
        return None
    return NanosecondInterval(
        start_ns=claim.interval.start_ns,
        end_ns=claim.interval.end_ns,
    )


def _require_interval_inside_member(
    interval: NanosecondInterval,
    member: TemporalPackageSetMember,
) -> None:
    if interval.start_ns < member.start_ns or interval.end_ns > member.end_ns:
        raise ActionEvidenceProjectionError(
            "action observation interval is outside its package/window bounds"
        )


def _require_interval_inside_window(
    interval: NanosecondInterval,
    package_set: TemporalPackageSet,
) -> None:
    if interval.start_ns < package_set.start_ns or interval.end_ns > package_set.end_ns:
        raise ActionEvidenceProjectionError(
            "cross-view hypothesis interval is outside the candidate window"
        )


def _interval_inside(
    inner: NanosecondInterval,
    outer: NanosecondInterval,
) -> bool:
    return inner.start_ns >= outer.start_ns and inner.end_ns <= outer.end_ns


def _interval_projection(interval: NanosecondInterval | None) -> object:
    return interval.model_dump(mode="json") if interval is not None else None


def _normalized_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = label.strip().lower()
    if not normalized:
        raise ActionEvidenceProjectionError("action evidence label cannot be blank")
    return normalized


def _evidence_semantic_projection(
    evidence: EnrichedEvidenceReference,
) -> dict[str, object]:
    return {
        "provider_item_ordinal": evidence.provider_item_ordinal,
        "package_ordinal": evidence.package_ordinal,
        "package_semantic_content_sha256": evidence.package_semantic_content_sha256,
        "camera_ordinal": evidence.camera_ordinal,
        "frame_ordinal": evidence.frame_ordinal,
        "aligned_timestamp_ns": str(evidence.aligned_timestamp_ns),
        "source_timestamp_ns": str(evidence.source_timestamp_ns),
        "source_artifact_sha256": evidence.source_artifact_sha256,
    }


def _evidence_sort_key(
    evidence: EnrichedEvidenceReference,
) -> tuple[int, int, int, int]:
    return (
        evidence.provider_item_ordinal,
        evidence.package_ordinal,
        evidence.camera_ordinal,
        evidence.frame_ordinal,
    )


def _action_observation_sort_key(
    observation: NormalizedActionObservation,
) -> tuple[int, int, int, str]:
    return (
        observation.source.output.part_ordinal,
        observation.package_ordinal,
        observation.source.claim_ordinal,
        observation.source_action_observation_logical_key,
    )


def _cross_view_sort_key(
    hypothesis: NormalizedCrossViewHypothesis,
) -> tuple[int, int, int, str]:
    return (
        hypothesis.interval.start_ns,
        hypothesis.interval.end_ns,
        hypothesis.source.output.part_ordinal,
        hypothesis.source_cross_view_logical_key,
    )


def _camera_outcome(
    observations: Sequence[NormalizedActionObservation],
) -> ActionEvidenceOutcome:
    if any(item.observation in _POSITIVE_OBSERVATIONS for item in observations):
        return ActionEvidenceOutcome.SUPPORTED
    if observations and all(
        item.observation is ProviderObservation.NO_EVENT for item in observations
    ):
        return ActionEvidenceOutcome.NO_ACTION
    return ActionEvidenceOutcome.INDETERMINATE


def _result_outcome(
    camera_evidence: SixCameraMap[CameraActionEvidence],
    hypotheses: Sequence[NormalizedCrossViewHypothesis],
) -> ActionEvidenceOutcome:
    if hypotheses or any(
        camera_evidence[camera_id].outcome is ActionEvidenceOutcome.SUPPORTED
        for camera_id in CAMERA_IDS
    ):
        return ActionEvidenceOutcome.SUPPORTED
    if all(
        camera_evidence[camera_id].outcome is ActionEvidenceOutcome.NO_ACTION
        for camera_id in CAMERA_IDS
    ):
        return ActionEvidenceOutcome.NO_ACTION
    return ActionEvidenceOutcome.INDETERMINATE


class ActionEvidenceExtractor:
    """Legacy extractor retained only as an explicit fail-closed surface."""

    def extract_evidence(
        self,
        candidate: CandidateEvent,
        dense_package: TemporalVisualPackage,
    ) -> ActionEvidence:
        _ = candidate
        _ = dense_package
        raise NotImplementedError(
            "ActionEvidenceExtractor.extract_evidence is non-runnable; use ActionEvidenceProjector"
        )


__all__ = [
    "ACTION_EVIDENCE_RESULT_PROJECTION_VERSION",
    "ActionEvidenceClaimRef",
    "ActionEvidenceExtractor",
    "ActionEvidenceOutcome",
    "ActionEvidenceOutputRef",
    "ActionEvidencePackageRef",
    "ActionEvidenceProjectionError",
    "ActionEvidenceProjector",
    "ActionEvidenceResult",
    "CameraActionEvidence",
    "NormalizedActionObservation",
    "NormalizedCrossViewHypothesis",
    "action_evidence_output_semantic_projection",
    "action_evidence_result_semantic_projection",
    "action_observation_semantic_projection",
    "cross_view_hypothesis_semantic_projection",
]
