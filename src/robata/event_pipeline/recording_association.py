"""Deterministic, non-production recording-level association evidence.

This module deliberately derives an asynchronous report from accepted evidence.  It
does not allocate an event or track identity, and callers must keep it outside the
primary recording-completion path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
ConfidenceMillionths = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
ActionLabel = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
]

RECORDING_ASSOCIATION_POLICY_PROJECTION_VERSION: Final = "recording-association-policy-semantic-v1"
RECORDING_ASSOCIATION_REPORT_PROJECTION_VERSION: Final = "recording-association-report-semantic-v1"
RECORDING_ASSOCIATION_REPORT_LOGICAL_KEY_NAMESPACE: Final = "recording-association-report-v1"


class RecordingAssociationError(ValueError):
    """The requested association closure cannot be derived deterministically."""


class RecordingAssociationOutcome(StrEnum):
    """A report outcome, intentionally distinct from an event claim."""

    ASSOCIATIONS_DERIVED = "ASSOCIATIONS_DERIVED"
    NO_ASSOCIATIONS = "NO_ASSOCIATIONS"


class AssociationBridgeKind(StrEnum):
    """The explicit evidence claim that permits a non-baseline association."""

    GAP_CONTINUITY = "GAP_CONTINUITY"
    LABEL_TRANSITION = "LABEL_TRANSITION"


class AssociationPairDisposition(StrEnum):
    """One deterministic decision about an unordered pair of inputs."""

    MERGED = "MERGED"
    SPLIT = "SPLIT"
    AMBIGUOUS = "AMBIGUOUS"


class AssociationInputDisposition(StrEnum):
    """The final association status of one source action in this report."""

    ASSOCIATED = "ASSOCIATED"
    UNASSOCIATED = "UNASSOCIATED"
    AMBIGUOUS = "AMBIGUOUS"


class AssociationReasonCode(StrEnum):
    """Stable explanations, not event semantics or review conclusions."""

    TOUCHING_OR_OVERLAPPING_SHARED_CAMERA = "TOUCHING_OR_OVERLAPPING_SHARED_CAMERA"
    GAP_CONTINUITY_EVIDENCE = "GAP_CONTINUITY_EVIDENCE"
    LABEL_TRANSITION_EVIDENCE = "LABEL_TRANSITION_EVIDENCE"
    GAP_EXCEEDS_POLICY = "GAP_EXCEEDS_POLICY"
    INSUFFICIENT_SHARED_CAMERA_SUPPORT = "INSUFFICIENT_SHARED_CAMERA_SUPPORT"
    GAP_REQUIRES_BRIDGE_EVIDENCE = "GAP_REQUIRES_BRIDGE_EVIDENCE"
    LABEL_TRANSITIONS_DISABLED = "LABEL_TRANSITIONS_DISABLED"
    LABEL_TRANSITION_REQUIRES_BRIDGE_EVIDENCE = "LABEL_TRANSITION_REQUIRES_BRIDGE_EVIDENCE"
    BRIDGE_DOES_NOT_SPAN_INPUTS = "BRIDGE_DOES_NOT_SPAN_INPUTS"
    BRIDGE_CONFIDENCE_BELOW_POLICY = "BRIDGE_CONFIDENCE_BELOW_POLICY"
    BRIDGE_CAMERA_NOT_SUPPORTED_BY_INPUTS = "BRIDGE_CAMERA_NOT_SUPPORTED_BY_INPUTS"
    AMBIGUOUS_TIED_BRIDGE_SUPPORT = "AMBIGUOUS_TIED_BRIDGE_SUPPORT"
    MAX_CLUSTER_MEMBERS_EXCEEDED = "MAX_CLUSTER_MEMBERS_EXCEEDED"
    ASSOCIATED_BY_EXPLICIT_EVIDENCE = "ASSOCIATED_BY_EXPLICIT_EVIDENCE"
    NO_SUPPORTED_ASSOCIATION = "NO_SUPPORTED_ASSOCIATION"
    AMBIGUOUS_ALTERNATIVE_RETAINED = "AMBIGUOUS_ALTERNATIVE_RETAINED"


class CompletedRecordingAssociationBinding(StrictModel):
    """Exact completed-recording binding required before a report can be derived."""

    completed_run_id: OpaqueUuid
    completed_recording_logical_key: NodeLogicalKey
    completed_recording_semantic_sha256: Sha256Digest
    completed_recording_exact_sha256: Sha256Digest
    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        _require_logical_key_digest(
            self.completed_recording_logical_key,
            self.completed_recording_semantic_sha256,
            "completed recording",
        )
        return self


class AssociationSourceActionRef(StrictModel):
    """Exact logical identity of one accepted source action."""

    source_action_logical_key: NodeLogicalKey
    source_action_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_logical_key_digest(
            self.source_action_logical_key,
            self.source_action_semantic_sha256,
            "source action",
        )
        return self


class AssociationAcceptedEvidenceRef(StrictModel):
    """One positive, accepted camera observation used by association."""

    source_action: AssociationSourceActionRef
    accepted_evidence_logical_key: NodeLogicalKey
    accepted_evidence_semantic_sha256: Sha256Digest
    accepted_evidence_exact_sha256: Sha256Digest
    camera_id: CameraId
    interval: NanosecondInterval
    label: ActionLabel
    confidence_millionths: ConfidenceMillionths

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_logical_key_digest(
            self.accepted_evidence_logical_key,
            self.accepted_evidence_semantic_sha256,
            "accepted evidence",
        )
        return self


class RecordingAssociationInput(StrictModel):
    """A complete association-relevant projection of one accepted action."""

    source_action: AssociationSourceActionRef
    mcap_id: OpaqueUuid
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    interval: NanosecondInterval
    label: ActionLabel
    accepted_evidence: tuple[AssociationAcceptedEvidenceRef, ...]
    semantic_sha256: Sha256Digest
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        source_action: AssociationSourceActionRef,
        mcap_id: str,
        source_content_sha256: str,
        camera_mapping_semantic_sha256: str,
        alignment_semantic_sha256: str,
        interval: NanosecondInterval,
        label: str,
        accepted_evidence: Sequence[AssociationAcceptedEvidenceRef],
    ) -> Self:
        values: dict[str, Any] = {
            "source_action": source_action,
            "mcap_id": mcap_id,
            "source_content_sha256": source_content_sha256,
            "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": alignment_semantic_sha256,
            "interval": interval,
            "label": label,
            "accepted_evidence": tuple(sorted(accepted_evidence, key=_accepted_evidence_sort_key)),
            "production_eligible": False,
        }
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": semantic_sha256(
                    _recording_association_input_projection_values(values)
                ),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        if not self.accepted_evidence:
            raise ValueError("recording association input requires accepted evidence")
        expected = tuple(sorted(self.accepted_evidence, key=_accepted_evidence_sort_key))
        if self.accepted_evidence != expected:
            raise ValueError("recording association accepted evidence must be canonically ordered")
        evidence_keys = tuple(item.accepted_evidence_logical_key for item in self.accepted_evidence)
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("recording association accepted evidence must be unique")
        if any(
            item.source_action != self.source_action or item.label != self.label
            for item in self.accepted_evidence
        ):
            raise ValueError("accepted evidence must bind the input source action and label")
        expected_interval = NanosecondInterval(
            start_ns=min(item.interval.start_ns for item in self.accepted_evidence),
            end_ns=max(item.interval.end_ns for item in self.accepted_evidence),
        )
        if self.interval != expected_interval:
            raise ValueError("input interval must be the accepted-evidence envelope")
        if self.semantic_sha256 != semantic_sha256(recording_association_input_projection(self)):
            raise ValueError("recording association input semantic identity is inconsistent")
        return self


class RecordingAssociationBridgeEvidence(StrictModel):
    """Exact evidence explicitly asserting continuity between two source actions."""

    source_actions: tuple[AssociationSourceActionRef, AssociationSourceActionRef]
    accepted_bridge_evidence_logical_key: NodeLogicalKey
    accepted_bridge_evidence_semantic_sha256: Sha256Digest
    accepted_bridge_evidence_exact_sha256: Sha256Digest
    camera_id: CameraId
    interval: NanosecondInterval
    confidence_millionths: ConfidenceMillionths
    kind: AssociationBridgeKind

    @model_validator(mode="after")
    def validate_bridge(self) -> Self:
        if self.source_actions[0] == self.source_actions[1]:
            raise ValueError("bridge evidence must bind two distinct source actions")
        if self.source_actions != tuple(sorted(self.source_actions, key=_source_action_sort_key)):
            raise ValueError("bridge source actions must be canonically ordered")
        _require_logical_key_digest(
            self.accepted_bridge_evidence_logical_key,
            self.accepted_bridge_evidence_semantic_sha256,
            "accepted bridge evidence",
        )
        return self


class RecordingAssociationPolicy(StrictModel):
    """Frozen non-production rule set for the derived association report."""

    version: SchemaVersion
    max_gap_ns: NonNegativeInt = 0
    min_input_confidence_millionths: ConfidenceMillionths = 500_000
    min_bridge_confidence_millionths: ConfidenceMillionths = 500_000
    allow_label_transitions: bool = False
    max_inputs: PositiveInt = 128
    projection_version: Literal["recording-association-policy-semantic-v1"] = (
        RECORDING_ASSOCIATION_POLICY_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        version: str,
        max_gap_ns: int = 0,
        min_input_confidence_millionths: int = 500_000,
        min_bridge_confidence_millionths: int = 500_000,
        allow_label_transitions: bool = False,
        max_inputs: int = 128,
    ) -> Self:
        values: dict[str, Any] = {
            "version": version,
            "max_gap_ns": max_gap_ns,
            "min_input_confidence_millionths": min_input_confidence_millionths,
            "min_bridge_confidence_millionths": min_bridge_confidence_millionths,
            "allow_label_transitions": allow_label_transitions,
            "max_inputs": max_inputs,
            "projection_version": RECORDING_ASSOCIATION_POLICY_PROJECTION_VERSION,
            "production_eligible": False,
        }
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": semantic_sha256(
                    _recording_association_policy_projection_values(values)
                ),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.semantic_sha256 != semantic_sha256(recording_association_policy_projection(self)):
            raise ValueError("recording association policy semantic identity is inconsistent")
        return self


class RecordingAssociationPairDecision(StrictModel):
    """Explainable decision for one unordered pair of accepted actions."""

    source_actions: tuple[AssociationSourceActionRef, AssociationSourceActionRef]
    disposition: AssociationPairDisposition
    reason_codes: tuple[AssociationReasonCode, ...]
    temporal_gap_ns: NonNegativeInt
    shared_camera_ids: tuple[CameraId, ...]
    bridge_evidence_logical_keys: tuple[NodeLogicalKey, ...] = ()
    selected_bridge_evidence_logical_key: NodeLogicalKey | None = None
    confidence_millionths: ConfidenceMillionths | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.source_actions[0] == self.source_actions[1]:
            raise ValueError("pair decision must bind two distinct source actions")
        if self.source_actions != tuple(sorted(self.source_actions, key=_source_action_sort_key)):
            raise ValueError("pair decision source actions must be canonically ordered")
        if not self.reason_codes or self.reason_codes != tuple(
            sorted(set(self.reason_codes), key=lambda item: item.value)
        ):
            raise ValueError("pair decision reasons must be unique and canonically ordered")
        if self.shared_camera_ids != _canonical_camera_ids(self.shared_camera_ids):
            raise ValueError("pair decision cameras must be unique and canonically ordered")
        if self.bridge_evidence_logical_keys != tuple(
            sorted(set(self.bridge_evidence_logical_keys))
        ):
            raise ValueError("pair decision bridge references must be unique and ordered")
        if (
            self.selected_bridge_evidence_logical_key is not None
            and self.selected_bridge_evidence_logical_key not in self.bridge_evidence_logical_keys
        ):
            raise ValueError("selected bridge evidence must be present in the decision evidence")
        if self.disposition is AssociationPairDisposition.MERGED:
            if self.confidence_millionths is None:
                raise ValueError("merged pair decision requires explicit confidence")
            if not self.shared_camera_ids:
                raise ValueError("merged pair decision requires shared camera evidence")
        return self


class RecordingAssociationInputDecision(StrictModel):
    """Final report-level status for a source action, including no-link evidence."""

    source_action: AssociationSourceActionRef
    disposition: AssociationInputDisposition
    reason_codes: tuple[AssociationReasonCode, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if not self.reason_codes or self.reason_codes != tuple(
            sorted(set(self.reason_codes), key=lambda item: item.value)
        ):
            raise ValueError("input decision reasons must be unique and canonically ordered")
        return self


class RecordingAssociationCluster(StrictModel):
    """An ordinal report grouping, deliberately not an event or track identity."""

    ordinal: NonNegativeInt
    source_actions: tuple[AssociationSourceActionRef, ...]
    interval: NanosecondInterval
    labels: tuple[ActionLabel, ...]
    continuity_confidence_millionths: ConfidenceMillionths

    @model_validator(mode="after")
    def validate_cluster(self) -> Self:
        if len(self.source_actions) < 2:
            raise ValueError("association cluster requires at least two source actions")
        if self.source_actions != tuple(sorted(self.source_actions, key=_source_action_sort_key)):
            raise ValueError("cluster source actions must be canonically ordered")
        keys = tuple(item.source_action_logical_key for item in self.source_actions)
        if len(keys) != len(set(keys)):
            raise ValueError("cluster source actions must be unique")
        if self.labels != tuple(sorted(set(self.labels))):
            raise ValueError("cluster labels must be unique and canonically ordered")
        return self


class RecordingAssociationReport(StrictModel):
    """Content-addressed asynchronous association report, never a stable event record."""

    recording: CompletedRecordingAssociationBinding
    policy: RecordingAssociationPolicy
    inputs: tuple[RecordingAssociationInput, ...]
    bridge_evidence: tuple[RecordingAssociationBridgeEvidence, ...]
    pair_decisions: tuple[RecordingAssociationPairDecision, ...]
    input_decisions: tuple[RecordingAssociationInputDecision, ...]
    clusters: tuple[RecordingAssociationCluster, ...]
    outcome: RecordingAssociationOutcome
    projection_version: Literal["recording-association-report-semantic-v1"] = (
        RECORDING_ASSOCIATION_REPORT_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if not self.inputs:
            raise ValueError("recording association report requires at least one input")
        if self.inputs != tuple(sorted(self.inputs, key=_recording_input_sort_key)):
            raise ValueError("report inputs must be canonically ordered")
        input_keys = tuple(item.source_action.source_action_logical_key for item in self.inputs)
        if len(input_keys) != len(set(input_keys)):
            raise ValueError("report inputs must have unique source actions")
        _validate_input_lineage(self.recording, self.inputs)
        if self.bridge_evidence != tuple(sorted(self.bridge_evidence, key=_bridge_sort_key)):
            raise ValueError("report bridge evidence must be canonically ordered")
        bridge_keys = tuple(
            item.accepted_bridge_evidence_logical_key for item in self.bridge_evidence
        )
        if len(bridge_keys) != len(set(bridge_keys)):
            raise ValueError("report bridge evidence must be unique")
        _validate_bridges(self.inputs, self.bridge_evidence)
        expected_pair_keys = {
            _pair_key(left.source_action, right.source_action)
            for index, left in enumerate(self.inputs)
            for right in self.inputs[index + 1 :]
        }
        actual_pair_keys = tuple(_decision_pair_key(item) for item in self.pair_decisions)
        if self.pair_decisions != tuple(sorted(self.pair_decisions, key=_pair_decision_sort_key)):
            raise ValueError("report pair decisions must be canonically ordered")
        if set(actual_pair_keys) != expected_pair_keys or len(actual_pair_keys) != len(
            set(actual_pair_keys)
        ):
            raise ValueError("report must retain exactly one decision for every input pair")
        expected_input_refs = tuple(item.source_action for item in self.inputs)
        actual_input_refs = tuple(item.source_action for item in self.input_decisions)
        if actual_input_refs != expected_input_refs:
            raise ValueError("report must retain exactly one ordered input decision per input")
        if self.clusters != tuple(sorted(self.clusters, key=_cluster_sort_key)) or tuple(
            item.ordinal for item in self.clusters
        ) != tuple(range(len(self.clusters))):
            raise ValueError("report clusters must be canonically ordered and ordinaled")
        cluster_members = [
            action.source_action_logical_key
            for cluster in self.clusters
            for action in cluster.source_actions
        ]
        if len(cluster_members) != len(set(cluster_members)):
            raise ValueError("a source action cannot appear in multiple association clusters")
        if any(key not in set(input_keys) for key in cluster_members):
            raise ValueError("association cluster references a foreign input")
        expected_outcome = (
            RecordingAssociationOutcome.ASSOCIATIONS_DERIVED
            if self.clusters
            else RecordingAssociationOutcome.NO_ASSOCIATIONS
        )
        if self.outcome is not expected_outcome:
            raise ValueError("association report outcome does not match its clusters")
        digest = semantic_sha256(recording_association_report_projection(self))
        if (
            self.semantic_sha256 != digest
            or self.logical_key != f"{RECORDING_ASSOCIATION_REPORT_LOGICAL_KEY_NAMESPACE}:{digest}"
        ):
            raise ValueError("recording association report semantic identity is inconsistent")
        return self


@dataclass(frozen=True)
class _CandidateEdge:
    pair_key: tuple[str, str]
    left: AssociationSourceActionRef
    right: AssociationSourceActionRef
    reason: AssociationReasonCode
    temporal_gap_ns: int
    confidence_millionths: int
    shared_camera_ids: tuple[CameraId, ...]
    bridge_evidence: tuple[RecordingAssociationBridgeEvidence, ...] = ()


class RecordingAssociationEngine:
    """Derive a deterministic report from explicit accepted evidence only."""

    def __init__(self, policy: RecordingAssociationPolicy) -> None:
        self._policy = RecordingAssociationPolicy.model_validate(
            policy.model_dump(mode="python"), strict=True
        )

    @property
    def policy(self) -> RecordingAssociationPolicy:
        return self._policy

    def derive(
        self,
        *,
        recording: CompletedRecordingAssociationBinding,
        inputs: Sequence[RecordingAssociationInput],
        bridge_evidence: Sequence[RecordingAssociationBridgeEvidence] = (),
    ) -> RecordingAssociationReport:
        checked_recording = CompletedRecordingAssociationBinding.model_validate(
            recording.model_dump(mode="python"), strict=True
        )
        checked_inputs = tuple(
            sorted(
                (
                    RecordingAssociationInput.model_validate(
                        item.model_dump(mode="python"), strict=True
                    )
                    for item in inputs
                ),
                key=_recording_input_sort_key,
            )
        )
        if not checked_inputs:
            raise RecordingAssociationError("recording association requires at least one input")
        if len(checked_inputs) > self._policy.max_inputs:
            raise RecordingAssociationError("recording association input count exceeds policy")
        input_keys = tuple(item.source_action.source_action_logical_key for item in checked_inputs)
        if len(input_keys) != len(set(input_keys)):
            raise RecordingAssociationError(
                "recording association inputs duplicate a source action"
            )
        _validate_input_lineage(checked_recording, checked_inputs)
        checked_bridges = tuple(
            sorted(
                (
                    RecordingAssociationBridgeEvidence.model_validate(
                        item.model_dump(mode="python"), strict=True
                    )
                    for item in bridge_evidence
                ),
                key=_bridge_sort_key,
            )
        )
        bridge_keys = tuple(item.accepted_bridge_evidence_logical_key for item in checked_bridges)
        if len(bridge_keys) != len(set(bridge_keys)):
            raise RecordingAssociationError("recording association bridge evidence is duplicated")
        _validate_bridges(checked_inputs, checked_bridges)

        input_by_key = {
            item.source_action.source_action_logical_key: item for item in checked_inputs
        }
        bridges_by_pair: dict[tuple[str, str], tuple[RecordingAssociationBridgeEvidence, ...]] = {
            key: tuple(sorted(value, key=_bridge_sort_key))
            for key, value in _bridges_by_pair(checked_bridges).items()
        }
        initial_decisions: dict[tuple[str, str], RecordingAssociationPairDecision] = {}
        candidate_edges: dict[tuple[str, str], _CandidateEdge] = {}

        for index, left in enumerate(checked_inputs):
            for right in checked_inputs[index + 1 :]:
                pair_key = _pair_key(left.source_action, right.source_action)
                decision, edge = self._classify_pair(
                    left,
                    right,
                    bridges_by_pair.get(pair_key, ()),
                )
                initial_decisions[pair_key] = decision
                if edge is not None:
                    candidate_edges[pair_key] = edge

        ambiguous_pairs = _ambiguous_bridge_pairs(candidate_edges, input_by_key)
        final_decisions = dict(initial_decisions)
        accepted_edges: list[_CandidateEdge] = []
        for pair_key, edge in candidate_edges.items():
            if pair_key in ambiguous_pairs:
                final_decisions[pair_key] = _ambiguous_pair_decision(edge)
            else:
                accepted_edges.append(edge)

        disjoint = _DisjointSet(input_keys)
        for edge in sorted(accepted_edges, key=_candidate_edge_sort_key):
            if (
                disjoint.component_size(edge.left.source_action_logical_key)
                + (
                    disjoint.component_size(edge.right.source_action_logical_key)
                    if disjoint.find(edge.left.source_action_logical_key)
                    != disjoint.find(edge.right.source_action_logical_key)
                    else 0
                )
                > self._policy.max_inputs
            ):
                final_decisions[edge.pair_key] = _split_for_cluster_limit(edge)
                continue
            disjoint.union(
                edge.left.source_action_logical_key, edge.right.source_action_logical_key
            )

        clusters = _build_clusters(checked_inputs, final_decisions, disjoint)
        input_decisions = _build_input_decisions(
            checked_inputs,
            tuple(final_decisions.values()),
            clusters,
        )
        ordered_decisions = tuple(sorted(final_decisions.values(), key=_pair_decision_sort_key))
        values: dict[str, Any] = {
            "recording": checked_recording,
            "policy": self._policy,
            "inputs": checked_inputs,
            "bridge_evidence": checked_bridges,
            "pair_decisions": ordered_decisions,
            "input_decisions": input_decisions,
            "clusters": clusters,
            "outcome": (
                RecordingAssociationOutcome.ASSOCIATIONS_DERIVED
                if clusters
                else RecordingAssociationOutcome.NO_ASSOCIATIONS
            ),
            "projection_version": RECORDING_ASSOCIATION_REPORT_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = RecordingAssociationReport.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{RECORDING_ASSOCIATION_REPORT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(recording_association_report_projection(draft))
        return RecordingAssociationReport.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{RECORDING_ASSOCIATION_REPORT_LOGICAL_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    def _classify_pair(
        self,
        left: RecordingAssociationInput,
        right: RecordingAssociationInput,
        bridges: Sequence[RecordingAssociationBridgeEvidence],
    ) -> tuple[RecordingAssociationPairDecision, _CandidateEdge | None]:
        pair_key = _pair_key(left.source_action, right.source_action)
        gap_ns = _temporal_gap(left.interval, right.interval)
        shared_camera_ids = _shared_supported_cameras(
            left,
            right,
            self._policy.min_input_confidence_millionths,
        )
        bridge_keys = tuple(item.accepted_bridge_evidence_logical_key for item in bridges)
        if left.label == right.label and gap_ns == 0:
            shared_camera_ids = _shared_touching_or_overlapping_cameras(
                left,
                right,
                self._policy.min_input_confidence_millionths,
            )
            if not shared_camera_ids:
                return (
                    _pair_decision(
                        left,
                        right,
                        AssociationPairDisposition.SPLIT,
                        (AssociationReasonCode.INSUFFICIENT_SHARED_CAMERA_SUPPORT,),
                        gap_ns,
                        shared_camera_ids,
                        bridge_keys,
                    ),
                    None,
                )
            confidence = _shared_input_confidence(left, right, shared_camera_ids)
            edge = _CandidateEdge(
                pair_key=pair_key,
                left=left.source_action,
                right=right.source_action,
                reason=AssociationReasonCode.TOUCHING_OR_OVERLAPPING_SHARED_CAMERA,
                temporal_gap_ns=gap_ns,
                confidence_millionths=confidence,
                shared_camera_ids=shared_camera_ids,
            )
            return _merged_pair_decision(edge), edge

        if gap_ns > self._policy.max_gap_ns:
            return (
                _pair_decision(
                    left,
                    right,
                    AssociationPairDisposition.SPLIT,
                    (AssociationReasonCode.GAP_EXCEEDS_POLICY,),
                    gap_ns,
                    shared_camera_ids,
                    bridge_keys,
                ),
                None,
            )
        if not shared_camera_ids:
            return (
                _pair_decision(
                    left,
                    right,
                    AssociationPairDisposition.SPLIT,
                    (AssociationReasonCode.INSUFFICIENT_SHARED_CAMERA_SUPPORT,),
                    gap_ns,
                    shared_camera_ids,
                    bridge_keys,
                ),
                None,
            )

        requires_transition = left.label != right.label
        if requires_transition and not self._policy.allow_label_transitions:
            return (
                _pair_decision(
                    left,
                    right,
                    AssociationPairDisposition.SPLIT,
                    (AssociationReasonCode.LABEL_TRANSITIONS_DISABLED,),
                    gap_ns,
                    shared_camera_ids,
                    bridge_keys,
                ),
                None,
            )
        expected_kind = (
            AssociationBridgeKind.LABEL_TRANSITION
            if requires_transition
            else AssociationBridgeKind.GAP_CONTINUITY
        )
        candidates, failure_reason = _eligible_bridges(
            left,
            right,
            bridges,
            expected_kind,
            self._policy.min_bridge_confidence_millionths,
        )
        if not candidates:
            reason = failure_reason or (
                AssociationReasonCode.LABEL_TRANSITION_REQUIRES_BRIDGE_EVIDENCE
                if requires_transition
                else AssociationReasonCode.GAP_REQUIRES_BRIDGE_EVIDENCE
            )
            return (
                _pair_decision(
                    left,
                    right,
                    AssociationPairDisposition.SPLIT,
                    (reason,),
                    gap_ns,
                    shared_camera_ids,
                    bridge_keys,
                ),
                None,
            )
        top_confidence = max(item.confidence_millionths for item in candidates)
        top = tuple(item for item in candidates if item.confidence_millionths == top_confidence)
        reason = (
            AssociationReasonCode.LABEL_TRANSITION_EVIDENCE
            if requires_transition
            else AssociationReasonCode.GAP_CONTINUITY_EVIDENCE
        )
        edge = _CandidateEdge(
            pair_key=pair_key,
            left=left.source_action,
            right=right.source_action,
            reason=reason,
            temporal_gap_ns=gap_ns,
            confidence_millionths=top_confidence,
            shared_camera_ids=shared_camera_ids,
            bridge_evidence=top,
        )
        if len(top) > 1:
            return _ambiguous_pair_decision(edge), edge
        return _merged_pair_decision(edge), edge


class _DisjointSet:
    def __init__(self, keys: Sequence[str]) -> None:
        self._parents = {key: key for key in keys}
        self._sizes = {key: 1 for key in keys}

    def find(self, key: str) -> str:
        parent = self._parents[key]
        while parent != self._parents[parent]:
            parent = self._parents[parent]
        self._parents[key] = parent
        return parent

    def component_size(self, key: str) -> int:
        return self._sizes[self.find(key)]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if (self._sizes[left_root], left_root) < (self._sizes[right_root], right_root):
            left_root, right_root = right_root, left_root
        self._parents[right_root] = left_root
        self._sizes[left_root] += self._sizes[right_root]


def recording_association_policy_projection(
    policy: RecordingAssociationPolicy,
) -> dict[str, object]:
    """Return the explicit semantic projection for a P11 association policy."""

    return _recording_association_policy_projection_values(policy.model_dump(mode="python"))


def recording_association_input_projection(
    input_value: RecordingAssociationInput,
) -> dict[str, object]:
    """Return the complete accepted-evidence projection for one association input."""

    return _recording_association_input_projection_values(input_value.model_dump(mode="python"))


def recording_association_report_projection(
    report: RecordingAssociationReport,
) -> dict[str, object]:
    """Return report semantics while excluding its derived content-addressed identity."""

    return {
        "semantic_projection_version": report.projection_version,
        "recording": report.recording.model_dump(mode="json"),
        "policy": recording_association_policy_projection(report.policy),
        "inputs": [recording_association_input_projection(item) for item in report.inputs],
        "bridge_evidence": [item.model_dump(mode="json") for item in report.bridge_evidence],
        "pair_decisions": [item.model_dump(mode="json") for item in report.pair_decisions],
        "input_decisions": [item.model_dump(mode="json") for item in report.input_decisions],
        "clusters": [item.model_dump(mode="json") for item in report.clusters],
        "outcome": report.outcome,
        "production_eligible": report.production_eligible,
        "identity_scope": "derived-recording-association-report-not-event-or-track-identity",
    }


def verify_recording_association_report(
    report: RecordingAssociationReport,
) -> RecordingAssociationReport:
    """Reject a syntactically valid report if it is not reproducible from its evidence."""

    checked = RecordingAssociationReport.model_validate(
        report.model_dump(mode="python"), strict=True
    )
    expected = RecordingAssociationEngine(checked.policy).derive(
        recording=checked.recording,
        inputs=checked.inputs,
        bridge_evidence=checked.bridge_evidence,
    )
    if expected.model_dump(mode="json") != checked.model_dump(mode="json"):
        raise ValueError("recording association report does not match deterministic policy replay")
    return checked


def _recording_association_policy_projection_values(values: dict[str, Any]) -> dict[str, object]:
    return {
        "semantic_projection_version": values["projection_version"],
        "version": values["version"],
        "max_gap_ns": values["max_gap_ns"],
        "min_input_confidence_millionths": values["min_input_confidence_millionths"],
        "min_bridge_confidence_millionths": values["min_bridge_confidence_millionths"],
        "allow_label_transitions": values["allow_label_transitions"],
        "max_inputs": values["max_inputs"],
        "association_basis": "accepted-camera-evidence-plus-explicit-bridge-evidence",
        "production_eligible": values["production_eligible"],
    }


def _recording_association_input_projection_values(values: dict[str, Any]) -> dict[str, object]:
    source_action = AssociationSourceActionRef.model_validate(values["source_action"], strict=True)
    accepted_evidence = tuple(
        AssociationAcceptedEvidenceRef.model_validate(item, strict=True)
        for item in values["accepted_evidence"]
    )
    interval = NanosecondInterval.model_validate(values["interval"], strict=True)
    return {
        "source_action": source_action.model_dump(mode="json"),
        "mcap_id": values["mcap_id"],
        "source_content_sha256": values["source_content_sha256"],
        "camera_mapping_semantic_sha256": values["camera_mapping_semantic_sha256"],
        "alignment_semantic_sha256": values["alignment_semantic_sha256"],
        "interval": interval.model_dump(mode="json"),
        "label": values["label"],
        "accepted_evidence": [item.model_dump(mode="json") for item in accepted_evidence],
        "production_eligible": values["production_eligible"],
    }


def _require_logical_key_digest(logical_key: str, digest: str, subject: str) -> None:
    if logical_key.rsplit(":", 1)[-1] != digest:
        raise ValueError(f"{subject} logical key must end with its semantic SHA-256")


def _source_action_sort_key(item: AssociationSourceActionRef) -> tuple[str, str]:
    return item.source_action_logical_key, item.source_action_semantic_sha256


def _accepted_evidence_sort_key(
    item: AssociationAcceptedEvidenceRef,
) -> tuple[int, int, str, str]:
    return (
        item.interval.start_ns,
        item.interval.end_ns,
        item.camera_id.value,
        item.accepted_evidence_logical_key,
    )


def _recording_input_sort_key(item: RecordingAssociationInput) -> tuple[int, int, str, str]:
    return (
        item.interval.start_ns,
        item.interval.end_ns,
        item.label,
        item.source_action.source_action_logical_key,
    )


def _bridge_sort_key(
    item: RecordingAssociationBridgeEvidence,
) -> tuple[str, str, str]:
    return (
        item.source_actions[0].source_action_logical_key,
        item.source_actions[1].source_action_logical_key,
        item.accepted_bridge_evidence_logical_key,
    )


def _ordered_source_action_pair(
    left: AssociationSourceActionRef,
    right: AssociationSourceActionRef,
) -> tuple[AssociationSourceActionRef, AssociationSourceActionRef]:
    if _source_action_sort_key(left) <= _source_action_sort_key(right):
        return left, right
    return right, left


def _pair_key(
    left: AssociationSourceActionRef,
    right: AssociationSourceActionRef,
) -> tuple[str, str]:
    if left.source_action_logical_key <= right.source_action_logical_key:
        return left.source_action_logical_key, right.source_action_logical_key
    return right.source_action_logical_key, left.source_action_logical_key


def _decision_pair_key(item: RecordingAssociationPairDecision) -> tuple[str, str]:
    return _pair_key(*item.source_actions)


def _pair_decision_sort_key(item: RecordingAssociationPairDecision) -> tuple[str, str]:
    return _decision_pair_key(item)


def _cluster_sort_key(item: RecordingAssociationCluster) -> tuple[int, int, tuple[str, ...]]:
    return (
        item.interval.start_ns,
        item.interval.end_ns,
        tuple(action.source_action_logical_key for action in item.source_actions),
    )


def _canonical_camera_ids(camera_ids: Sequence[CameraId]) -> tuple[CameraId, ...]:
    actual = set(camera_ids)
    return tuple(camera_id for camera_id in CAMERA_IDS if camera_id in actual)


def _validate_input_lineage(
    recording: CompletedRecordingAssociationBinding,
    inputs: Sequence[RecordingAssociationInput],
) -> None:
    for item in inputs:
        if (
            item.mcap_id != recording.mcap_id
            or item.source_content_sha256 != recording.source_content_sha256
            or item.camera_mapping_semantic_sha256 != recording.camera_mapping_semantic_sha256
            or item.alignment_semantic_sha256 != recording.alignment_semantic_sha256
        ):
            raise RecordingAssociationError(
                "association input does not bind the completed recording"
            )


def _validate_bridges(
    inputs: Sequence[RecordingAssociationInput],
    bridges: Sequence[RecordingAssociationBridgeEvidence],
) -> None:
    input_by_key = {item.source_action.source_action_logical_key: item for item in inputs}
    for bridge in bridges:
        source_keys = tuple(item.source_action_logical_key for item in bridge.source_actions)
        if any(key not in input_by_key for key in source_keys):
            raise RecordingAssociationError(
                "bridge evidence references a foreign association input"
            )
        for ref in bridge.source_actions:
            if input_by_key[ref.source_action_logical_key].source_action != ref:
                raise RecordingAssociationError(
                    "bridge evidence source action identity is inconsistent"
                )


def _bridges_by_pair(
    bridges: Sequence[RecordingAssociationBridgeEvidence],
) -> dict[tuple[str, str], list[RecordingAssociationBridgeEvidence]]:
    result: dict[tuple[str, str], list[RecordingAssociationBridgeEvidence]] = defaultdict(list)
    for bridge in bridges:
        result[_pair_key(*bridge.source_actions)].append(bridge)
    return result


def _temporal_gap(left: NanosecondInterval, right: NanosecondInterval) -> int:
    if left.end_ns < right.start_ns:
        return right.start_ns - left.end_ns
    if right.end_ns < left.start_ns:
        return left.start_ns - right.end_ns
    return 0


def _camera_confidences(input_value: RecordingAssociationInput) -> dict[CameraId, int]:
    values: dict[CameraId, int] = {}
    for evidence in input_value.accepted_evidence:
        values[evidence.camera_id] = max(
            values.get(evidence.camera_id, 0), evidence.confidence_millionths
        )
    return values


def _shared_supported_cameras(
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
    minimum_confidence: int,
) -> tuple[CameraId, ...]:
    left_confidences = _camera_confidences(left)
    right_confidences = _camera_confidences(right)
    return tuple(
        camera_id
        for camera_id in CAMERA_IDS
        if (
            left_confidences.get(camera_id, 0) >= minimum_confidence
            and right_confidences.get(camera_id, 0) >= minimum_confidence
        )
    )


def _shared_touching_or_overlapping_cameras(
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
    minimum_confidence: int,
) -> tuple[CameraId, ...]:
    """Require contact between actual camera evidence, not just action envelopes."""

    shared: list[CameraId] = []
    for camera_id in CAMERA_IDS:
        left_evidence = tuple(
            item
            for item in left.accepted_evidence
            if (item.camera_id is camera_id and item.confidence_millionths >= minimum_confidence)
        )
        right_evidence = tuple(
            item
            for item in right.accepted_evidence
            if (item.camera_id is camera_id and item.confidence_millionths >= minimum_confidence)
        )
        if any(
            _temporal_gap(left_item.interval, right_item.interval) == 0
            for left_item in left_evidence
            for right_item in right_evidence
        ):
            shared.append(camera_id)
    return tuple(shared)


def _shared_input_confidence(
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
    shared_camera_ids: Sequence[CameraId],
) -> int:
    left_confidences = _camera_confidences(left)
    right_confidences = _camera_confidences(right)
    return min(
        min(left_confidences[camera_id], right_confidences[camera_id])
        for camera_id in shared_camera_ids
    )


def _bridge_spans_inputs(
    bridge: RecordingAssociationBridgeEvidence,
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
) -> bool:
    earlier, later = sorted(
        (left, right),
        key=lambda item: (item.interval.start_ns, item.interval.end_ns),
    )
    return (
        bridge.interval.start_ns <= earlier.interval.end_ns
        and bridge.interval.end_ns >= later.interval.start_ns
    )


def _eligible_bridges(
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
    bridges: Sequence[RecordingAssociationBridgeEvidence],
    expected_kind: AssociationBridgeKind,
    minimum_confidence: int,
) -> tuple[tuple[RecordingAssociationBridgeEvidence, ...], AssociationReasonCode | None]:
    candidates: list[RecordingAssociationBridgeEvidence] = []
    failure_reasons: set[AssociationReasonCode] = set()
    left_confidences = _camera_confidences(left)
    right_confidences = _camera_confidences(right)
    for bridge in bridges:
        if bridge.kind is not expected_kind:
            continue
        if bridge.confidence_millionths < minimum_confidence:
            failure_reasons.add(AssociationReasonCode.BRIDGE_CONFIDENCE_BELOW_POLICY)
            continue
        if (
            left_confidences.get(bridge.camera_id, 0) < minimum_confidence
            or right_confidences.get(bridge.camera_id, 0) < minimum_confidence
        ):
            failure_reasons.add(AssociationReasonCode.BRIDGE_CAMERA_NOT_SUPPORTED_BY_INPUTS)
            continue
        if not _bridge_spans_inputs(bridge, left, right):
            failure_reasons.add(AssociationReasonCode.BRIDGE_DOES_NOT_SPAN_INPUTS)
            continue
        candidates.append(bridge)
    if candidates:
        return tuple(sorted(candidates, key=_bridge_sort_key)), None
    if not failure_reasons:
        return (), None
    return (), sorted(failure_reasons, key=lambda item: item.value)[0]


def _pair_decision(
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
    disposition: AssociationPairDisposition,
    reasons: Sequence[AssociationReasonCode],
    gap_ns: int,
    shared_camera_ids: Sequence[CameraId],
    bridge_keys: Sequence[NodeLogicalKey],
    *,
    selected_bridge: NodeLogicalKey | None = None,
    confidence_millionths: int | None = None,
) -> RecordingAssociationPairDecision:
    return RecordingAssociationPairDecision(
        source_actions=_ordered_source_action_pair(left.source_action, right.source_action),
        disposition=disposition,
        reason_codes=tuple(sorted(set(reasons), key=lambda item: item.value)),
        temporal_gap_ns=gap_ns,
        shared_camera_ids=_canonical_camera_ids(shared_camera_ids),
        bridge_evidence_logical_keys=tuple(sorted(set(bridge_keys))),
        selected_bridge_evidence_logical_key=selected_bridge,
        confidence_millionths=confidence_millionths,
    )


def _merged_pair_decision(edge: _CandidateEdge) -> RecordingAssociationPairDecision:
    bridge_keys = tuple(item.accepted_bridge_evidence_logical_key for item in edge.bridge_evidence)
    selected = bridge_keys[0] if len(bridge_keys) == 1 else None
    return RecordingAssociationPairDecision(
        source_actions=_ordered_source_action_pair(edge.left, edge.right),
        disposition=AssociationPairDisposition.MERGED,
        reason_codes=(edge.reason,),
        temporal_gap_ns=edge.temporal_gap_ns,
        shared_camera_ids=edge.shared_camera_ids,
        bridge_evidence_logical_keys=bridge_keys,
        selected_bridge_evidence_logical_key=selected,
        confidence_millionths=edge.confidence_millionths,
    )


def _ambiguous_pair_decision(edge: _CandidateEdge) -> RecordingAssociationPairDecision:
    bridge_keys = tuple(item.accepted_bridge_evidence_logical_key for item in edge.bridge_evidence)
    return RecordingAssociationPairDecision(
        source_actions=_ordered_source_action_pair(edge.left, edge.right),
        disposition=AssociationPairDisposition.AMBIGUOUS,
        reason_codes=(AssociationReasonCode.AMBIGUOUS_TIED_BRIDGE_SUPPORT,),
        temporal_gap_ns=edge.temporal_gap_ns,
        shared_camera_ids=edge.shared_camera_ids,
        bridge_evidence_logical_keys=bridge_keys,
        confidence_millionths=edge.confidence_millionths,
    )


def _split_for_cluster_limit(edge: _CandidateEdge) -> RecordingAssociationPairDecision:
    bridge_keys = tuple(item.accepted_bridge_evidence_logical_key for item in edge.bridge_evidence)
    return RecordingAssociationPairDecision(
        source_actions=_ordered_source_action_pair(edge.left, edge.right),
        disposition=AssociationPairDisposition.SPLIT,
        reason_codes=(AssociationReasonCode.MAX_CLUSTER_MEMBERS_EXCEEDED,),
        temporal_gap_ns=edge.temporal_gap_ns,
        shared_camera_ids=edge.shared_camera_ids,
        bridge_evidence_logical_keys=bridge_keys,
        confidence_millionths=edge.confidence_millionths,
    )


def _candidate_edge_sort_key(edge: _CandidateEdge) -> tuple[int, str, str, str]:
    return (
        0 if edge.reason is AssociationReasonCode.TOUCHING_OR_OVERLAPPING_SHARED_CAMERA else 1,
        edge.left.source_action_logical_key,
        edge.right.source_action_logical_key,
        edge.reason.value,
    )


def _ambiguous_bridge_pairs(
    edges: dict[tuple[str, str], _CandidateEdge],
    input_by_key: dict[str, RecordingAssociationInput],
) -> set[tuple[str, str]]:
    ambiguous: set[tuple[str, str]] = {
        pair_key for pair_key, edge in edges.items() if len(edge.bridge_evidence) > 1
    }
    per_side: dict[tuple[str, str], list[_CandidateEdge]] = defaultdict(list)
    for edge in edges.values():
        if not edge.bridge_evidence:
            continue
        for source, other in ((edge.left, edge.right), (edge.right, edge.left)):
            source_input = input_by_key[source.source_action_logical_key]
            other_input = input_by_key[other.source_action_logical_key]
            side = (
                "EARLIER"
                if (
                    other_input.interval.start_ns,
                    other_input.interval.end_ns,
                    other.source_action_logical_key,
                )
                < (
                    source_input.interval.start_ns,
                    source_input.interval.end_ns,
                    source.source_action_logical_key,
                )
                else "LATER"
            )
            per_side[(source.source_action_logical_key, side)].append(edge)
    for choices in per_side.values():
        top = max(item.confidence_millionths for item in choices)
        tied = [item for item in choices if item.confidence_millionths == top]
        if len(tied) > 1:
            ambiguous.update(item.pair_key for item in tied)
    return ambiguous


def _build_clusters(
    inputs: Sequence[RecordingAssociationInput],
    decisions: dict[tuple[str, str], RecordingAssociationPairDecision],
    disjoint: _DisjointSet,
) -> tuple[RecordingAssociationCluster, ...]:
    input_by_key = {item.source_action.source_action_logical_key: item for item in inputs}
    groups: dict[str, list[str]] = defaultdict(list)
    for key in input_by_key:
        groups[disjoint.find(key)].append(key)
    unsorted_clusters: list[RecordingAssociationCluster] = []
    for member_keys in groups.values():
        if len(member_keys) < 2:
            continue
        members = tuple(sorted(member_keys))
        member_inputs = tuple(input_by_key[key] for key in members)
        internal_decisions = tuple(
            decision
            for decision in decisions.values()
            if decision.disposition is AssociationPairDisposition.MERGED
            and all(
                action.source_action_logical_key in members for action in decision.source_actions
            )
        )
        confidence = min(
            item.confidence_millionths
            for item in internal_decisions
            if item.confidence_millionths is not None
        )
        unsorted_clusters.append(
            RecordingAssociationCluster(
                ordinal=0,
                source_actions=tuple(item.source_action for item in member_inputs),
                interval=NanosecondInterval(
                    start_ns=min(item.interval.start_ns for item in member_inputs),
                    end_ns=max(item.interval.end_ns for item in member_inputs),
                ),
                labels=tuple(sorted({item.label for item in member_inputs})),
                continuity_confidence_millionths=confidence,
            )
        )
    ordered = sorted(unsorted_clusters, key=_cluster_sort_key)
    return tuple(
        item.model_copy(update={"ordinal": ordinal}) for ordinal, item in enumerate(ordered)
    )


def _build_input_decisions(
    inputs: Sequence[RecordingAssociationInput],
    pair_decisions: Sequence[RecordingAssociationPairDecision],
    clusters: Sequence[RecordingAssociationCluster],
) -> tuple[RecordingAssociationInputDecision, ...]:
    clustered = {
        action.source_action_logical_key
        for cluster in clusters
        for action in cluster.source_actions
    }
    ambiguous = {
        action.source_action_logical_key
        for decision in pair_decisions
        if decision.disposition is AssociationPairDisposition.AMBIGUOUS
        for action in decision.source_actions
    }
    result: list[RecordingAssociationInputDecision] = []
    for input_value in inputs:
        key = input_value.source_action.source_action_logical_key
        if key in clustered:
            reasons: set[AssociationReasonCode] = {
                AssociationReasonCode.ASSOCIATED_BY_EXPLICIT_EVIDENCE
            }
            if key in ambiguous:
                reasons.add(AssociationReasonCode.AMBIGUOUS_ALTERNATIVE_RETAINED)
            disposition = AssociationInputDisposition.ASSOCIATED
        elif key in ambiguous:
            reasons = {AssociationReasonCode.AMBIGUOUS_TIED_BRIDGE_SUPPORT}
            disposition = AssociationInputDisposition.AMBIGUOUS
        else:
            reasons = {AssociationReasonCode.NO_SUPPORTED_ASSOCIATION}
            disposition = AssociationInputDisposition.UNASSOCIATED
        result.append(
            RecordingAssociationInputDecision(
                source_action=input_value.source_action,
                disposition=disposition,
                reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
            )
        )
    return tuple(result)


__all__ = [
    "RECORDING_ASSOCIATION_POLICY_PROJECTION_VERSION",
    "RECORDING_ASSOCIATION_REPORT_LOGICAL_KEY_NAMESPACE",
    "RECORDING_ASSOCIATION_REPORT_PROJECTION_VERSION",
    "AssociationAcceptedEvidenceRef",
    "AssociationBridgeKind",
    "AssociationInputDisposition",
    "AssociationPairDisposition",
    "AssociationReasonCode",
    "AssociationSourceActionRef",
    "CompletedRecordingAssociationBinding",
    "RecordingAssociationBridgeEvidence",
    "RecordingAssociationCluster",
    "RecordingAssociationEngine",
    "RecordingAssociationError",
    "RecordingAssociationInput",
    "RecordingAssociationInputDecision",
    "RecordingAssociationOutcome",
    "RecordingAssociationPairDecision",
    "RecordingAssociationPolicy",
    "RecordingAssociationReport",
    "recording_association_input_projection",
    "recording_association_policy_projection",
    "recording_association_report_projection",
    "verify_recording_association_report",
]
