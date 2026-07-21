"""Deterministic local fusion from candidate evidence to coarse physical actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.event_pipeline.candidate import CandidateReductionResult, CanonicalCandidateEvent
from robata.event_pipeline.evidence import (
    ActionEvidenceOutcome,
    ActionEvidenceResult,
    NormalizedActionObservation,
    NormalizedCrossViewHypothesis,
)
from robata.inference.enrichment import ProviderObservation

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
ActionLabel = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
]
AmbiguityCode = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$"),
]

PROVISIONAL_FUSION_POLICY_PROJECTION_VERSION: Final = "provisional-action-fusion-policy-semantic-v1"
PROVISIONAL_PHYSICAL_ACTION_PROJECTION_VERSION: Final = "provisional-physical-action-semantic-v1"
PROVISIONAL_PHYSICAL_ACTION_LOGICAL_KEY_NAMESPACE: Final = "provisional-physical-action-v1"
PROVISIONAL_FUSION_RESULT_PROJECTION_VERSION: Final = "provisional-action-fusion-result-semantic-v1"
PROVISIONAL_FUSION_RESULT_LOGICAL_KEY_NAMESPACE: Final = "provisional-action-fusion-v1"
LOCAL_PROVISIONAL_FUSION_POLICY_VERSION: Final = "local-provisional-action-fusion-v1"

_POSITIVE_OBSERVATIONS = frozenset({ProviderObservation.SUPPORTING, ProviderObservation.PARTIAL})


class ProvisionalFusionError(ValueError):
    """The exact candidate/evidence closure cannot be fused deterministically."""


class ProvisionalFusionOutcome(StrEnum):
    NO_ACTIONS = "NO_ACTIONS"
    ACTIONS = "ACTIONS"


class ProvisionalFusionPolicy(StrictModel):
    """Frozen development policy; O-11/O-12 still own production association."""

    version: SchemaVersion
    association_gap_ns: NonNegativeInt = 0
    max_physical_actions: PositiveInt = 128
    projection_version: Literal["provisional-action-fusion-policy-semantic-v1"] = (
        PROVISIONAL_FUSION_POLICY_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        version: str,
        association_gap_ns: int = 0,
        max_physical_actions: int = 128,
    ) -> Self:
        values: dict[str, Any] = {
            "version": version,
            "association_gap_ns": association_gap_ns,
            "max_physical_actions": max_physical_actions,
            "projection_version": PROVISIONAL_FUSION_POLICY_PROJECTION_VERSION,
            "production_eligible": False,
        }
        digest = semantic_sha256(_provisional_fusion_policy_projection_values(values))
        return cls.model_validate({**values, "semantic_sha256": digest}, strict=True)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.semantic_sha256 != semantic_sha256(provisional_fusion_policy_projection(self)):
            raise ValueError("provisional fusion policy semantic identity is inconsistent")
        return self


class ProvisionalCandidateEvidenceRef(StrictModel):
    """One candidate and its exact normalized ACTION_EVIDENCE result."""

    candidate_event_id: OpaqueUuid
    candidate_logical_key: NodeLogicalKey
    candidate_label: ActionLabel
    action_evidence_logical_key: NodeLogicalKey
    action_evidence_semantic_sha256: Sha256Digest
    outcome: ActionEvidenceOutcome


class ProvisionalActionObservationRef(StrictModel):
    """Run-independent reference to one normalized camera observation."""

    source_action_evidence_logical_key: NodeLogicalKey
    source_action_observation_logical_key: NodeLogicalKey
    camera_id: CameraId
    interval: NanosecondInterval | None
    source_label: str | None
    resolved_label: ActionLabel
    observation: ProviderObservation


class ProvisionalCrossViewHypothesisRef(StrictModel):
    """Run-independent reference to one normalized cross-view hypothesis."""

    source_action_evidence_logical_key: NodeLogicalKey
    source_cross_view_logical_key: NodeLogicalKey
    interval: NanosecondInterval
    source_label: str | None
    resolved_label: ActionLabel
    observation: Literal[ProviderObservation.SUPPORTING, ProviderObservation.PARTIAL]


class ProvisionalCameraEvidence(StrictModel):
    """One explicit camera slot for a provisional physical action."""

    camera_id: CameraId
    observations: tuple[ProvisionalActionObservationRef, ...]
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        expected = tuple(sorted(self.observations, key=_observation_ref_sort_key))
        if self.observations != expected:
            raise ValueError("provisional camera observations must be canonically ordered")
        keys = tuple(item.source_action_observation_logical_key for item in self.observations)
        if len(keys) != len(set(keys)):
            raise ValueError("provisional camera observations must be unique")
        if any(item.camera_id is not self.camera_id for item in self.observations):
            raise ValueError("provisional camera slot contains a foreign observation")
        return self


class ProvisionalPhysicalAction(StrictModel):
    """One run-independent coarse physical-action derivation, never a stable event ID."""

    ordinal: NonNegativeInt
    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    coarse_interval: NanosecondInterval
    label: ActionLabel
    source_candidates: tuple[ProvisionalCandidateEvidenceRef, ...]
    camera_evidence: SixCameraMap[ProvisionalCameraEvidence]
    cross_view_hypotheses: tuple[ProvisionalCrossViewHypothesisRef, ...]
    ambiguity_codes: tuple[AmbiguityCode, ...]
    policy_semantic_sha256: Sha256Digest
    projection_version: Literal["provisional-physical-action-semantic-v1"] = (
        PROVISIONAL_PHYSICAL_ACTION_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        if self.source_candidates != tuple(
            sorted(self.source_candidates, key=_candidate_evidence_ref_sort_key)
        ):
            raise ValueError("provisional action sources must be canonically ordered")
        source_keys = tuple(item.action_evidence_logical_key for item in self.source_candidates)
        if not source_keys or len(source_keys) != len(set(source_keys)):
            raise ValueError("provisional action requires unique source evidence results")
        if self.cross_view_hypotheses != tuple(
            sorted(self.cross_view_hypotheses, key=_cross_view_ref_sort_key)
        ):
            raise ValueError("provisional cross-view hypotheses must be canonically ordered")
        cross_keys = tuple(
            item.source_cross_view_logical_key for item in self.cross_view_hypotheses
        )
        if len(cross_keys) != len(set(cross_keys)):
            raise ValueError("provisional cross-view hypotheses must be unique")
        if self.ambiguity_codes != tuple(sorted(set(self.ambiguity_codes))):
            raise ValueError("provisional ambiguity codes must be unique and ordered")
        for camera_id in CAMERA_IDS:
            if self.camera_evidence[camera_id].camera_id is not camera_id:
                raise ValueError("provisional action camera slot identity is inconsistent")
        positive_intervals = _positive_action_intervals(self)
        if not positive_intervals:
            raise ValueError("provisional action requires positive normalized evidence")
        expected_interval = NanosecondInterval(
            start_ns=min(item.start_ns for item in positive_intervals),
            end_ns=max(item.end_ns for item in positive_intervals),
        )
        if self.coarse_interval != expected_interval:
            raise ValueError("provisional action interval is not the positive-evidence envelope")
        if any(label != self.label for label in _positive_action_labels(self)):
            raise ValueError("provisional action contains incompatible positive labels")
        allowed_sources = set(source_keys)
        if any(
            item.source_action_evidence_logical_key not in allowed_sources
            for camera_id in CAMERA_IDS
            for item in self.camera_evidence[camera_id].observations
        ) or any(
            item.source_action_evidence_logical_key not in allowed_sources
            for item in self.cross_view_hypotheses
        ):
            raise ValueError("provisional action evidence references an undeclared source")
        digest = semantic_sha256(provisional_physical_action_semantic_projection(self))
        if (
            self.semantic_sha256 != digest
            or self.logical_key != f"{PROVISIONAL_PHYSICAL_ACTION_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("provisional physical-action identity is inconsistent")
        return self


class ProvisionalFusionResult(StrictModel):
    """Deterministic 0/1/N provisional fusion result for one candidate closure."""

    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    source_candidate_reduction_logical_key: NodeLogicalKey
    source_candidate_reduction_semantic_sha256: Sha256Digest
    source_candidates: tuple[ProvisionalCandidateEvidenceRef, ...]
    policy: ProvisionalFusionPolicy
    actions: tuple[ProvisionalPhysicalAction, ...]
    outcome: ProvisionalFusionOutcome
    projection_version: Literal["provisional-action-fusion-result-semantic-v1"] = (
        PROVISIONAL_FUSION_RESULT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @property
    def no_actions(self) -> bool:
        return not self.actions

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.source_candidates != tuple(
            sorted(self.source_candidates, key=_candidate_evidence_ref_sort_key)
        ):
            raise ValueError("provisional fusion sources must be canonically ordered")
        candidate_keys = tuple(item.candidate_logical_key for item in self.source_candidates)
        evidence_keys = tuple(item.action_evidence_logical_key for item in self.source_candidates)
        if (
            not self.source_candidates
            or len(candidate_keys) != len(set(candidate_keys))
            or len(evidence_keys) != len(set(evidence_keys))
        ):
            raise ValueError("provisional fusion requires unique candidate/evidence closure")
        expected_actions = tuple(sorted(self.actions, key=_physical_action_sort_key))
        actual_ordinals = tuple(item.ordinal for item in self.actions)
        if self.actions != expected_actions or actual_ordinals != tuple(range(len(self.actions))):
            raise ValueError("provisional actions must be canonically ordered and ordinaled")
        action_keys = tuple(item.logical_key for item in self.actions)
        if len(action_keys) != len(set(action_keys)):
            raise ValueError("provisional actions must have unique logical identities")
        expected_outcome = (
            ProvisionalFusionOutcome.ACTIONS
            if self.actions
            else ProvisionalFusionOutcome.NO_ACTIONS
        )
        if self.outcome is not expected_outcome:
            raise ValueError("provisional fusion outcome does not match its actions")
        supported = {
            item.action_evidence_logical_key
            for item in self.source_candidates
            if item.outcome is ActionEvidenceOutcome.SUPPORTED
        }
        consumed = {
            source.action_evidence_logical_key
            for action in self.actions
            for source in action.source_candidates
        }
        if consumed != supported:
            raise ValueError("provisional actions must consume every supported evidence result")
        if any(
            item.outcome is ActionEvidenceOutcome.INDETERMINATE for item in self.source_candidates
        ):
            raise ValueError("indeterminate action evidence cannot enter provisional fusion")
        if any(
            action.mcap_id != self.mcap_id
            or action.source_content_sha256 != self.source_content_sha256
            or action.camera_mapping_semantic_sha256 != self.camera_mapping_semantic_sha256
            or action.alignment_semantic_sha256 != self.alignment_semantic_sha256
            or action.policy_semantic_sha256 != self.policy.semantic_sha256
            or action.production_eligible
            for action in self.actions
        ):
            raise ValueError("provisional action lineage differs from its fusion result")
        digest = semantic_sha256(provisional_fusion_result_semantic_projection(self))
        if (
            self.semantic_sha256 != digest
            or self.logical_key != f"{PROVISIONAL_FUSION_RESULT_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("provisional fusion result identity is inconsistent")
        return self


@dataclass(frozen=True)
class _PositiveSeed:
    kind: Literal["CAMERA", "CROSS_VIEW"]
    source_action_evidence_logical_key: NodeLogicalKey
    logical_key: NodeLogicalKey
    interval: NanosecondInterval
    label: str
    source_label: str | None
    camera_id: CameraId | None


class ProvisionalPhysicalActionFuser:
    """Associate normalized positive evidence into deterministic coarse actions."""

    def __init__(self, policy: ProvisionalFusionPolicy) -> None:
        self._policy = ProvisionalFusionPolicy.model_validate(
            policy.model_dump(mode="python"), strict=True
        )

    @property
    def policy(self) -> ProvisionalFusionPolicy:
        return self._policy

    def fuse(
        self,
        candidate_reduction: CandidateReductionResult,
        evidence_results: Sequence[ActionEvidenceResult],
    ) -> ProvisionalFusionResult:
        reduction = CandidateReductionResult.model_validate(
            candidate_reduction.model_dump(mode="python"), strict=True
        )
        if not reduction.candidates:
            raise ProvisionalFusionError(
                "provisional fusion is scheduled only for a nonempty candidate closure"
            )
        ordered_results = _bind_candidate_evidence(reduction, evidence_results)
        source_refs = tuple(
            sorted(
                (
                    _candidate_evidence_ref(candidate, result)
                    for candidate, result in ordered_results
                ),
                key=_candidate_evidence_ref_sort_key,
            )
        )
        first = ordered_results[0][1]
        seeds = _positive_seeds(ordered_results)
        components = _connected_components(seeds, self._policy.association_gap_ns)
        if len(components) > self._policy.max_physical_actions:
            raise ProvisionalFusionError("provisional fusion exceeded max_physical_actions")
        results_by_key = {result.logical_key: result for _, result in ordered_results}
        source_ref_by_key = {item.action_evidence_logical_key: item for item in source_refs}
        actions = tuple(
            _build_action(
                ordinal=ordinal,
                component=component,
                results_by_key=results_by_key,
                source_ref_by_key=source_ref_by_key,
                policy=self._policy,
                mcap_id=first.mcap_id,
                source_content_sha256=first.source_content_sha256,
                camera_mapping_semantic_sha256=first.camera_mapping_semantic_sha256,
                alignment_semantic_sha256=first.alignment_semantic_sha256,
            )
            for ordinal, component in enumerate(components)
        )
        values: dict[str, Any] = {
            "mcap_id": first.mcap_id,
            "source_content_sha256": first.source_content_sha256,
            "camera_mapping_semantic_sha256": first.camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": first.alignment_semantic_sha256,
            "source_candidate_reduction_logical_key": reduction.logical_key,
            "source_candidate_reduction_semantic_sha256": reduction.semantic_sha256,
            "source_candidates": source_refs,
            "policy": self._policy,
            "actions": actions,
            "outcome": (
                ProvisionalFusionOutcome.ACTIONS if actions else ProvisionalFusionOutcome.NO_ACTIONS
            ),
            "projection_version": PROVISIONAL_FUSION_RESULT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = ProvisionalFusionResult.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{PROVISIONAL_FUSION_RESULT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(provisional_fusion_result_semantic_projection(draft))
        return ProvisionalFusionResult.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{PROVISIONAL_FUSION_RESULT_LOGICAL_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )


def _bind_candidate_evidence(
    reduction: CandidateReductionResult,
    evidence_results: Sequence[ActionEvidenceResult],
) -> tuple[tuple[CanonicalCandidateEvent, ActionEvidenceResult], ...]:
    checked = tuple(
        ActionEvidenceResult.model_validate(item.model_dump(mode="python"), strict=True)
        for item in evidence_results
    )
    by_candidate: dict[str, ActionEvidenceResult] = {}
    for result in checked:
        if result.candidate_logical_key in by_candidate:
            raise ProvisionalFusionError("duplicate ACTION_EVIDENCE result for one candidate")
        by_candidate[result.candidate_logical_key] = result
    expected_keys = {item.candidate_logical_key for item in reduction.candidates}
    if set(by_candidate) != expected_keys:
        raise ProvisionalFusionError(
            "ACTION_EVIDENCE results do not exactly cover the candidate reduction"
        )
    ordered: list[tuple[CanonicalCandidateEvent, ActionEvidenceResult]] = []
    for candidate in reduction.candidates:
        result = by_candidate[candidate.candidate_logical_key]
        if (
            result.candidate_event_id != candidate.candidate_event_id
            or result.candidate_label != candidate.label
            or result.candidate_effective_interval != candidate.effective_interval
            or result.requested_dense_interval != candidate.requested_dense_interval
            or result.mcap_id != candidate.mcap_id
            or result.source_content_sha256 != candidate.source_content_sha256
            or result.camera_mapping_semantic_sha256 != candidate.camera_mapping_semantic_sha256
            or result.alignment_semantic_sha256 != candidate.alignment_semantic_sha256
        ):
            raise ProvisionalFusionError(
                "ACTION_EVIDENCE result is foreign to its candidate lineage"
            )
        if result.outcome is ActionEvidenceOutcome.INDETERMINATE:
            raise ProvisionalFusionError(
                "indeterminate ACTION_EVIDENCE cannot enter provisional fusion"
            )
        ordered.append((candidate, result))
    return tuple(ordered)


def _candidate_evidence_ref(
    candidate: CanonicalCandidateEvent,
    result: ActionEvidenceResult,
) -> ProvisionalCandidateEvidenceRef:
    return ProvisionalCandidateEvidenceRef(
        candidate_event_id=candidate.candidate_event_id,
        candidate_logical_key=candidate.candidate_logical_key,
        candidate_label=candidate.label,
        action_evidence_logical_key=result.logical_key,
        action_evidence_semantic_sha256=result.semantic_sha256,
        outcome=result.outcome,
    )


def _positive_seeds(
    ordered_results: Sequence[tuple[CanonicalCandidateEvent, ActionEvidenceResult]],
) -> tuple[_PositiveSeed, ...]:
    seeds: list[_PositiveSeed] = []
    seen: set[str] = set()
    for _, result in ordered_results:
        for camera_id in CAMERA_IDS:
            for observation in result.camera_evidence[camera_id].observations:
                if observation.observation not in _POSITIVE_OBSERVATIONS:
                    continue
                if observation.interval is None:
                    raise ProvisionalFusionError("positive camera evidence lacks an interval")
                label = observation.label or result.candidate_label
                seed = _PositiveSeed(
                    kind="CAMERA",
                    source_action_evidence_logical_key=result.logical_key,
                    logical_key=observation.source_action_observation_logical_key,
                    interval=observation.interval,
                    label=label,
                    source_label=observation.label,
                    camera_id=camera_id,
                )
                if seed.logical_key in seen:
                    raise ProvisionalFusionError("positive evidence seed is duplicated")
                seen.add(seed.logical_key)
                seeds.append(seed)
        for hypothesis in result.cross_view_hypotheses:
            label = hypothesis.label or result.candidate_label
            seed = _PositiveSeed(
                kind="CROSS_VIEW",
                source_action_evidence_logical_key=result.logical_key,
                logical_key=hypothesis.source_cross_view_logical_key,
                interval=hypothesis.interval,
                label=label,
                source_label=hypothesis.label,
                camera_id=None,
            )
            if seed.logical_key in seen:
                raise ProvisionalFusionError("positive evidence seed is duplicated")
            seen.add(seed.logical_key)
            seeds.append(seed)
    supported = {
        result.logical_key
        for _, result in ordered_results
        if result.outcome is ActionEvidenceOutcome.SUPPORTED
    }
    seeded = {item.source_action_evidence_logical_key for item in seeds}
    if supported != seeded:
        raise ProvisionalFusionError("supported ACTION_EVIDENCE lacks a positive fusion seed")
    return tuple(sorted(seeds, key=_positive_seed_sort_key))


def _connected_components(
    seeds: Sequence[_PositiveSeed], association_gap_ns: int
) -> tuple[tuple[_PositiveSeed, ...], ...]:
    if not seeds:
        return ()
    parents = list(range(len(seeds)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            if seeds[left].label == seeds[right].label and _within_gap(
                seeds[left].interval,
                seeds[right].interval,
                association_gap_ns,
            ):
                union(left, right)
    grouped: dict[int, list[_PositiveSeed]] = {}
    for index, seed in enumerate(seeds):
        grouped.setdefault(find(index), []).append(seed)
    components = tuple(
        tuple(sorted(items, key=_positive_seed_sort_key)) for items in grouped.values()
    )
    return tuple(sorted(components, key=_component_sort_key))


def _build_action(
    *,
    ordinal: int,
    component: Sequence[_PositiveSeed],
    results_by_key: dict[str, ActionEvidenceResult],
    source_ref_by_key: dict[str, ProvisionalCandidateEvidenceRef],
    policy: ProvisionalFusionPolicy,
    mcap_id: str,
    source_content_sha256: str,
    camera_mapping_semantic_sha256: str,
    alignment_semantic_sha256: str,
) -> ProvisionalPhysicalAction:
    if not component:
        raise ProvisionalFusionError("provisional action component cannot be empty")
    label = component[0].label
    if any(item.label != label for item in component):
        raise ProvisionalFusionError("provisional action component contains incompatible labels")
    interval = NanosecondInterval(
        start_ns=min(item.interval.start_ns for item in component),
        end_ns=max(item.interval.end_ns for item in component),
    )
    source_keys = tuple(sorted({item.source_action_evidence_logical_key for item in component}))
    sources = tuple(source_ref_by_key[item] for item in source_keys)
    positive_camera_keys = {item.logical_key for item in component if item.kind == "CAMERA"}
    camera_slots: dict[CameraId, ProvisionalCameraEvidence] = {}
    contradictory = False
    for camera_id in CAMERA_IDS:
        observations: list[ProvisionalActionObservationRef] = []
        for source_key in source_keys:
            result = results_by_key[source_key]
            for observation in result.camera_evidence[camera_id].observations:
                is_positive_member = (
                    observation.observation in _POSITIVE_OBSERVATIONS
                    and observation.source_action_observation_logical_key in positive_camera_keys
                )
                is_context = observation.observation not in _POSITIVE_OBSERVATIONS and (
                    observation.interval is None
                    or _intervals_intersect(observation.interval, interval)
                )
                if not is_positive_member and not is_context:
                    continue
                ref = _observation_ref(result, observation)
                observations.append(ref)
                contradictory = contradictory or (ref.observation is ProviderObservation.NO_EVENT)
        camera_slots[camera_id] = ProvisionalCameraEvidence(
            camera_id=camera_id,
            observations=tuple(sorted(observations, key=_observation_ref_sort_key)),
            production_eligible=False,
        )
    cross_keys = {item.logical_key for item in component if item.kind == "CROSS_VIEW"}
    cross_refs = tuple(
        sorted(
            (
                _cross_view_ref(result, hypothesis)
                for source_key in source_keys
                for result in (results_by_key[source_key],)
                for hypothesis in result.cross_view_hypotheses
                if hypothesis.source_cross_view_logical_key in cross_keys
            ),
            key=_cross_view_ref_sort_key,
        )
    )
    ambiguity_codes: set[str] = set()
    if any(item.source_label is None for item in component):
        ambiguity_codes.add("LABEL_FALLBACK_TO_CANDIDATE")
    if any(item.candidate_label != label for item in sources):
        ambiguity_codes.add("CANDIDATE_LABEL_DISAGREEMENT")
    if contradictory:
        ambiguity_codes.add("CONTRADICTORY_NO_EVENT_EVIDENCE")
    values: dict[str, Any] = {
        "ordinal": ordinal,
        "mcap_id": mcap_id,
        "source_content_sha256": source_content_sha256,
        "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": alignment_semantic_sha256,
        "coarse_interval": interval,
        "label": label,
        "source_candidates": tuple(sorted(sources, key=_candidate_evidence_ref_sort_key)),
        "camera_evidence": SixCameraMap[ProvisionalCameraEvidence](camera_slots),
        "cross_view_hypotheses": cross_refs,
        "ambiguity_codes": tuple(sorted(ambiguity_codes)),
        "policy_semantic_sha256": policy.semantic_sha256,
        "projection_version": PROVISIONAL_PHYSICAL_ACTION_PROJECTION_VERSION,
        "production_eligible": False,
    }
    draft = ProvisionalPhysicalAction.model_construct(
        semantic_sha256="0" * 64,
        logical_key=f"{PROVISIONAL_PHYSICAL_ACTION_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
        **values,
    )
    digest = semantic_sha256(provisional_physical_action_semantic_projection(draft))
    return ProvisionalPhysicalAction.model_validate(
        {
            **values,
            "semantic_sha256": digest,
            "logical_key": f"{PROVISIONAL_PHYSICAL_ACTION_LOGICAL_KEY_NAMESPACE}:{digest}",
        },
        strict=True,
    )


def _observation_ref(
    result: ActionEvidenceResult,
    observation: NormalizedActionObservation,
) -> ProvisionalActionObservationRef:
    return ProvisionalActionObservationRef(
        source_action_evidence_logical_key=result.logical_key,
        source_action_observation_logical_key=observation.source_action_observation_logical_key,
        camera_id=observation.camera_id,
        interval=observation.interval,
        source_label=observation.label,
        resolved_label=observation.label or result.candidate_label,
        observation=observation.observation,
    )


def _cross_view_ref(
    result: ActionEvidenceResult,
    hypothesis: NormalizedCrossViewHypothesis,
) -> ProvisionalCrossViewHypothesisRef:
    return ProvisionalCrossViewHypothesisRef(
        source_action_evidence_logical_key=result.logical_key,
        source_cross_view_logical_key=hypothesis.source_cross_view_logical_key,
        interval=hypothesis.interval,
        source_label=hypothesis.label,
        resolved_label=hypothesis.label or result.candidate_label,
        observation=hypothesis.observation,
    )


def _provisional_fusion_policy_projection_values(
    values: dict[str, object],
) -> dict[str, object]:
    return {
        "semantic_projection_version": values["projection_version"],
        "version": values["version"],
        "association_gap_ns": values["association_gap_ns"],
        "max_physical_actions": values["max_physical_actions"],
        "association_basis": "normalized-label-and-half-open-interval-connected-components",
        "production_eligible": values["production_eligible"],
    }


def provisional_fusion_policy_projection(policy: ProvisionalFusionPolicy) -> dict[str, object]:
    return _provisional_fusion_policy_projection_values(policy.model_dump(mode="python"))


def _candidate_ref_projection(item: ProvisionalCandidateEvidenceRef) -> dict[str, object]:
    return {
        "candidate_logical_key": item.candidate_logical_key,
        "candidate_label": item.candidate_label,
        "action_evidence_logical_key": item.action_evidence_logical_key,
        "action_evidence_semantic_sha256": item.action_evidence_semantic_sha256,
        "outcome": item.outcome,
    }


def provisional_physical_action_semantic_projection(
    action: ProvisionalPhysicalAction,
) -> dict[str, object]:
    return {
        "semantic_projection_version": action.projection_version,
        "source_content_sha256": action.source_content_sha256,
        "camera_mapping_semantic_sha256": action.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": action.alignment_semantic_sha256,
        "coarse_interval": action.coarse_interval.model_dump(mode="json"),
        "label": action.label,
        "source_candidates": [_candidate_ref_projection(item) for item in action.source_candidates],
        "camera_evidence": {
            camera_id.value: [
                item.model_dump(mode="json")
                for item in action.camera_evidence[camera_id].observations
            ]
            for camera_id in CAMERA_IDS
        },
        "cross_view_hypotheses": [
            item.model_dump(mode="json") for item in action.cross_view_hypotheses
        ],
        "ambiguity_codes": list(action.ambiguity_codes),
        "policy_semantic_sha256": action.policy_semantic_sha256,
        "production_eligible": action.production_eligible,
    }


def provisional_fusion_result_semantic_projection(
    result: ProvisionalFusionResult,
) -> dict[str, object]:
    return {
        "semantic_projection_version": result.projection_version,
        "source_content_sha256": result.source_content_sha256,
        "camera_mapping_semantic_sha256": result.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": result.alignment_semantic_sha256,
        "source_candidate_reduction_logical_key": (result.source_candidate_reduction_logical_key),
        "source_candidate_reduction_semantic_sha256": (
            result.source_candidate_reduction_semantic_sha256
        ),
        "source_candidates": [_candidate_ref_projection(item) for item in result.source_candidates],
        "policy_semantic_sha256": result.policy.semantic_sha256,
        "action_logical_keys": [item.logical_key for item in result.actions],
        "outcome": result.outcome,
        "production_eligible": result.production_eligible,
    }


def _positive_action_intervals(
    action: ProvisionalPhysicalAction,
) -> tuple[NanosecondInterval, ...]:
    camera_intervals = tuple(
        item.interval
        for camera_id in CAMERA_IDS
        for item in action.camera_evidence[camera_id].observations
        if item.observation in _POSITIVE_OBSERVATIONS and item.interval is not None
    )
    return camera_intervals + tuple(item.interval for item in action.cross_view_hypotheses)


def _positive_action_labels(action: ProvisionalPhysicalAction) -> tuple[str, ...]:
    camera_labels = tuple(
        item.resolved_label
        for camera_id in CAMERA_IDS
        for item in action.camera_evidence[camera_id].observations
        if item.observation in _POSITIVE_OBSERVATIONS
    )
    return camera_labels + tuple(item.resolved_label for item in action.cross_view_hypotheses)


def _within_gap(
    left: NanosecondInterval,
    right: NanosecondInterval,
    gap_ns: int,
) -> bool:
    return left.start_ns <= right.end_ns + gap_ns and right.start_ns <= left.end_ns + gap_ns


def _intervals_intersect(left: NanosecondInterval, right: NanosecondInterval) -> bool:
    return left.start_ns < right.end_ns and right.start_ns < left.end_ns


def _candidate_evidence_ref_sort_key(
    item: ProvisionalCandidateEvidenceRef,
) -> tuple[str, str]:
    return item.candidate_logical_key, item.action_evidence_logical_key


def _observation_ref_sort_key(
    item: ProvisionalActionObservationRef,
) -> tuple[int, int, str, str]:
    return (
        item.interval.start_ns if item.interval is not None else -(2**63),
        item.interval.end_ns if item.interval is not None else -(2**63),
        item.observation.value,
        item.source_action_observation_logical_key,
    )


def _cross_view_ref_sort_key(
    item: ProvisionalCrossViewHypothesisRef,
) -> tuple[int, int, str, str]:
    return (
        item.interval.start_ns,
        item.interval.end_ns,
        item.resolved_label,
        item.source_cross_view_logical_key,
    )


def _positive_seed_sort_key(item: _PositiveSeed) -> tuple[str, int, int, str, str]:
    return item.label, item.interval.start_ns, item.interval.end_ns, item.kind, item.logical_key


def _component_sort_key(
    component: Sequence[_PositiveSeed],
) -> tuple[int, int, str, tuple[str, ...]]:
    return (
        min(item.interval.start_ns for item in component),
        max(item.interval.end_ns for item in component),
        component[0].label,
        tuple(item.logical_key for item in component),
    )


def _physical_action_sort_key(
    item: ProvisionalPhysicalAction,
) -> tuple[int, int, str, str]:
    return (
        item.coarse_interval.start_ns,
        item.coarse_interval.end_ns,
        item.label,
        item.logical_key,
    )


__all__ = [
    "LOCAL_PROVISIONAL_FUSION_POLICY_VERSION",
    "PROVISIONAL_FUSION_POLICY_PROJECTION_VERSION",
    "PROVISIONAL_FUSION_RESULT_LOGICAL_KEY_NAMESPACE",
    "PROVISIONAL_FUSION_RESULT_PROJECTION_VERSION",
    "PROVISIONAL_PHYSICAL_ACTION_LOGICAL_KEY_NAMESPACE",
    "PROVISIONAL_PHYSICAL_ACTION_PROJECTION_VERSION",
    "ProvisionalActionObservationRef",
    "ProvisionalCameraEvidence",
    "ProvisionalCandidateEvidenceRef",
    "ProvisionalCrossViewHypothesisRef",
    "ProvisionalFusionError",
    "ProvisionalFusionOutcome",
    "ProvisionalFusionPolicy",
    "ProvisionalFusionResult",
    "ProvisionalPhysicalAction",
    "ProvisionalPhysicalActionFuser",
    "provisional_fusion_policy_projection",
    "provisional_fusion_result_semantic_projection",
    "provisional_physical_action_semantic_projection",
]
