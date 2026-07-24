"""Deterministic recording-level reduction over finalized stream windows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal, Self, cast

from pydantic import model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.local_stream_causal import (
    LocalStreamWindowSemanticEvidenceV2,
    LocalStreamWindowSemanticStatus,
)
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    NonEmptyString,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_finalization import (
    RecordingFinalizationMap,
    WindowTerminalClosure,
)
from robata.contracts.stream_inference import (
    StreamInferenceTerminalReference,
    StreamWindowResult,
)

LOCAL_STREAM_RECORDING_RESULT_PROJECTION_VERSION: Final = (
    "local-stream-recording-result-semantic-v1"
)
LOCAL_STREAM_RECORDING_REDUCTION_POLICY_VERSION: Final = (
    "local-conformance-stream-recording-reduction-v1"
)
LOCAL_STREAM_RECORDING_RESULT_KEY_NAMESPACE: Final = "local-stream-recording-result-v1"
LOCAL_CONFORMANCE_EVIDENCE_CLASS: Final = "LOCAL_CONFORMANCE"
LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID: Final = (
    "https://schemas.robata.dev/local-stream-recording-result"
)
LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION: Final = "1.0.0"
LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION: Final = "2.0.0"
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID: Final = (
    "https://schemas.robata.dev/local-stream-window-semantic-evidence"
)
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION: Final = "1.0.0"
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_PROJECTION_VERSION: Final = (
    "local-stream-window-semantic-evidence-v1"
)
LOCAL_STREAM_RECORDING_RESULT_V2_PROJECTION_VERSION: Final = (
    "local-stream-recording-result-semantic-v2"
)
LOCAL_STREAM_RECORDING_REDUCTION_V2_POLICY_VERSION: Final = (
    "local-conformance-stream-recording-reduction-v2"
)
LOCAL_STREAM_RECORDING_RESULT_V2_KEY_NAMESPACE: Final = "local-stream-recording-result-v2"

type LocalStreamOutputDecision = Literal["ADMITTED", "NO_EVENTS", "ABSTAINED"]
type LocalStreamSemanticKind = Literal[
    "EVENT_PROPOSAL",
    "CANDIDATE",
    "ACTION",
    "BOUNDARY",
    "HYPOTHESIS",
]


LOCAL_STREAM_RECORDING_RESULT_V3_SCHEMA_VERSION: Final = "3.0.0"
LOCAL_STREAM_RECORDING_RESULT_V3_PROJECTION_VERSION: Final = (
    "local-stream-recording-result-semantic-v3"
)
LOCAL_STREAM_RECORDING_REDUCTION_V3_POLICY_VERSION: Final = (
    "local-conformance-stream-recording-reduction-v3"
)
LOCAL_STREAM_RECORDING_RESULT_V3_KEY_NAMESPACE: Final = "local-stream-recording-result-v3"
LOCAL_STREAM_CAUSAL_BUNDLE_ROOT_VERSION: Final = "local-stream-causal-bundle-root-v1"
LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION: Final = "4.0.0"
LOCAL_STREAM_RECORDING_RESULT_V4_PROJECTION_VERSION: Final = (
    "local-stream-recording-result-semantic-v4"
)
LOCAL_STREAM_RECORDING_REDUCTION_V4_POLICY_VERSION: Final = (
    "local-conformance-stream-recording-reduction-v4"
)
LOCAL_STREAM_RECORDING_RESULT_V4_KEY_NAMESPACE: Final = "local-stream-recording-result-v4"


class LocalStreamRecordingReductionError(ValueError):
    """Final stream evidence cannot form one complete recording reduction."""


class LocalStreamRecordingResult(StrictModel):
    """Non-production recording result derived from one sealed stream history."""

    schema_version: Literal["1.0"] = "1.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = LOCAL_CONFORMANCE_EVIDENCE_CLASS
    production_eligible: Literal[False] = False
    recording_result_projection_version: SchemaVersion = (
        LOCAL_STREAM_RECORDING_RESULT_PROJECTION_VERSION
    )
    reduction_policy_version: SchemaVersion = LOCAL_STREAM_RECORDING_REDUCTION_POLICY_VERSION
    capture_scope_digest: Sha256Digest
    plan_key: NonEmptyString
    expected_plan_seal_semantic_sha256: Sha256Digest
    window_terminal_closure_semantic_sha256: Sha256Digest
    recording_finalization_semantic_sha256: Sha256Digest
    final_recording_identity: Sha256Digest
    ordered_window_result_semantic_sha256_values: tuple[Sha256Digest, ...]
    ordered_window_terminal_outcomes: tuple[TerminalOutcome, ...]
    accepted_terminals: tuple[StreamInferenceTerminalReference, ...]
    output_decision: LocalStreamOutputDecision
    recording_result_key: NonEmptyString
    recording_result_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (
            self.recording_result_projection_version
            != LOCAL_STREAM_RECORDING_RESULT_PROJECTION_VERSION
        ):
            raise ValueError("recording result uses the registered projection version")
        if self.reduction_policy_version != LOCAL_STREAM_RECORDING_REDUCTION_POLICY_VERSION:
            raise ValueError("recording result uses the local reduction policy version")
        if len(self.ordered_window_result_semantic_sha256_values) != len(
            self.ordered_window_terminal_outcomes
        ):
            raise ValueError("ordered window result digests and outcomes must align")
        if len(set(self.ordered_window_result_semantic_sha256_values)) != len(
            self.ordered_window_result_semantic_sha256_values
        ):
            raise ValueError("ordered window result digests must be unique")

        terminal_keys = tuple(_terminal_reference_key(item) for item in self.accepted_terminals)
        terminal_digests = tuple(item.terminal_semantic_sha256 for item in self.accepted_terminals)
        if (
            terminal_keys != tuple(sorted(terminal_keys))
            or len(set(terminal_keys)) != len(terminal_keys)
            or len(set(terminal_digests)) != len(terminal_digests)
        ):
            raise ValueError("accepted terminal references must be unique and canonical")
        if self.output_decision != _reduce_output_decision(self.ordered_window_terminal_outcomes):
            raise ValueError("output_decision does not match ordered window outcomes")

        expected = local_stream_recording_result_semantic_sha256(self)
        if self.recording_result_semantic_sha256 != expected:
            raise ValueError("recording_result_semantic_sha256 does not match result projection")
        if self.recording_result_key != derive_local_stream_recording_result_key(expected):
            raise ValueError("recording_result_key does not match result digest")
        return self


class LocalStreamQaCameraReference(StrictModel):
    """One exact camera member of the canonical six-camera QA closure."""

    camera_id: CameraId
    semantic_sha256: Sha256Digest


class LocalStreamSemanticIntervalReference(StrictModel):
    """Run-independent semantic object assigned to every overlapping stream window."""

    kind: LocalStreamSemanticKind
    logical_key: NonEmptyString
    semantic_sha256: Sha256Digest
    interval: NanosecondInterval


def _semantic_reference_key(
    reference: LocalStreamSemanticIntervalReference,
) -> tuple[int, int, str, str, str]:
    return (
        reference.interval.start_ns,
        reference.interval.end_ns,
        reference.kind,
        reference.logical_key,
        reference.semantic_sha256,
    )


def _validate_semantic_closure_parts(
    *,
    qa_camera_references: tuple[LocalStreamQaCameraReference, ...],
    proposal_references: tuple[LocalStreamSemanticIntervalReference, ...],
    candidate_references: tuple[LocalStreamSemanticIntervalReference, ...],
    action_references: tuple[LocalStreamSemanticIntervalReference, ...],
    boundary_references: tuple[LocalStreamSemanticIntervalReference, ...],
    hypothesis_references: tuple[LocalStreamSemanticIntervalReference, ...],
) -> None:
    if tuple(item.camera_id for item in qa_camera_references) != CAMERA_IDS:
        raise ValueError("stream semantic truth requires six QA cameras in canonical order")
    for expected_kind, references in (
        ("EVENT_PROPOSAL", proposal_references),
        ("CANDIDATE", candidate_references),
        ("ACTION", action_references),
        ("BOUNDARY", boundary_references),
        ("HYPOTHESIS", hypothesis_references),
    ):
        if any(item.kind != expected_kind for item in references):
            raise ValueError(f"{expected_kind} closure contains a foreign semantic kind")
        keys = tuple(_semantic_reference_key(item) for item in references)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError(f"{expected_kind} closure must be unique and canonical")


class LocalStreamCanonicalTruth(StrictModel):
    """Exact recording semantics projected from the existing canonical mock result."""

    six_camera_qa_semantic_sha256: Sha256Digest
    qa_camera_references: tuple[LocalStreamQaCameraReference, ...]
    event_proposal_result_semantic_sha256: Sha256Digest
    proposal_references: tuple[LocalStreamSemanticIntervalReference, ...]
    candidate_reduction_semantic_sha256: Sha256Digest
    candidate_references: tuple[LocalStreamSemanticIntervalReference, ...]
    provisional_fusion_semantic_sha256: Sha256Digest | None
    action_references: tuple[LocalStreamSemanticIntervalReference, ...]
    boundary_closure_semantic_sha256: Sha256Digest
    boundary_references: tuple[LocalStreamSemanticIntervalReference, ...]
    output_decision: LocalStreamOutputDecision
    output_decision_semantic_sha256: Sha256Digest | None
    hypothesis_references: tuple[LocalStreamSemanticIntervalReference, ...]

    @model_validator(mode="after")
    def validate_truth(self) -> Self:
        _validate_semantic_closure_parts(
            qa_camera_references=self.qa_camera_references,
            proposal_references=self.proposal_references,
            candidate_references=self.candidate_references,
            action_references=self.action_references,
            boundary_references=self.boundary_references,
            hypothesis_references=self.hypothesis_references,
        )
        if (self.output_decision == "ADMITTED") != bool(self.hypothesis_references):
            raise ValueError("ADMITTED must exactly match a nonempty hypothesis closure")
        if self.output_decision == "NO_EVENTS" and (
            self.proposal_references
            or self.candidate_references
            or self.action_references
            or self.boundary_references
        ):
            raise ValueError("NO_EVENTS cannot retain downstream event semantics")
        return self


class LocalStreamWindowSemanticEvidence(StrictModel):
    """Content-addressed canonical-mock semantics assigned to one sealed window."""

    schema_version: Literal["1.0"] = "1.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = LOCAL_CONFORMANCE_EVIDENCE_CLASS
    production_eligible: Literal[False] = False
    projection_version: Literal["local-stream-window-semantic-evidence-v1"] = (
        LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_PROJECTION_VERSION
    )
    plan_key: NonEmptyString
    expected_ordinal: int
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    window_result_semantic_sha256: Sha256Digest
    effective_interval: NanosecondInterval
    six_camera_qa_semantic_sha256: Sha256Digest
    qa_camera_references: tuple[LocalStreamQaCameraReference, ...]
    event_proposal_result_semantic_sha256: Sha256Digest
    proposal_references: tuple[LocalStreamSemanticIntervalReference, ...]
    candidate_reduction_semantic_sha256: Sha256Digest
    candidate_references: tuple[LocalStreamSemanticIntervalReference, ...]
    provisional_fusion_semantic_sha256: Sha256Digest | None
    action_references: tuple[LocalStreamSemanticIntervalReference, ...]
    boundary_closure_semantic_sha256: Sha256Digest
    boundary_references: tuple[LocalStreamSemanticIntervalReference, ...]
    output_decision: LocalStreamOutputDecision
    output_decision_semantic_sha256: Sha256Digest | None
    hypothesis_references: tuple[LocalStreamSemanticIntervalReference, ...]
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.expected_ordinal < 0:
            raise ValueError("expected_ordinal must be nonnegative")
        _validate_semantic_closure_parts(
            qa_camera_references=self.qa_camera_references,
            proposal_references=self.proposal_references,
            candidate_references=self.candidate_references,
            action_references=self.action_references,
            boundary_references=self.boundary_references,
            hypothesis_references=self.hypothesis_references,
        )
        if any(
            not _intervals_overlap(item.interval, self.effective_interval)
            for item in (
                *self.proposal_references,
                *self.candidate_references,
                *self.action_references,
                *self.boundary_references,
                *self.hypothesis_references,
            )
        ):
            raise ValueError("window semantic reference does not overlap its effective interval")
        if self.semantic_sha256 != local_stream_window_semantic_evidence_sha256(self):
            raise ValueError("window semantic evidence digest is inconsistent")
        return self


class LocalStreamRecordingResultV2(StrictModel):
    """Recording truth rebuilt from content-addressed window semantic evidence."""

    schema_version: Literal["2.0"] = "2.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = LOCAL_CONFORMANCE_EVIDENCE_CLASS
    production_eligible: Literal[False] = False
    recording_result_projection_version: Literal["local-stream-recording-result-semantic-v2"] = (
        LOCAL_STREAM_RECORDING_RESULT_V2_PROJECTION_VERSION
    )
    reduction_policy_version: Literal["local-conformance-stream-recording-reduction-v2"] = (
        LOCAL_STREAM_RECORDING_REDUCTION_V2_POLICY_VERSION
    )
    capture_scope_digest: Sha256Digest
    plan_key: NonEmptyString
    expected_plan_seal_semantic_sha256: Sha256Digest
    window_terminal_closure_semantic_sha256: Sha256Digest
    recording_finalization_semantic_sha256: Sha256Digest
    final_recording_identity: Sha256Digest
    ordered_window_result_semantic_sha256_values: tuple[Sha256Digest, ...]
    ordered_window_terminal_outcomes: tuple[TerminalOutcome, ...]
    ordered_window_semantic_evidence_refs: tuple[ArtifactEvidenceRef, ...]
    ordered_window_semantic_evidence_sha256_values: tuple[Sha256Digest, ...]
    six_camera_qa_semantic_sha256: Sha256Digest
    qa_camera_references: tuple[LocalStreamQaCameraReference, ...]
    event_proposal_result_semantic_sha256: Sha256Digest
    proposal_references: tuple[LocalStreamSemanticIntervalReference, ...]
    candidate_reduction_semantic_sha256: Sha256Digest
    candidate_references: tuple[LocalStreamSemanticIntervalReference, ...]
    provisional_fusion_semantic_sha256: Sha256Digest | None
    action_references: tuple[LocalStreamSemanticIntervalReference, ...]
    boundary_closure_semantic_sha256: Sha256Digest
    boundary_references: tuple[LocalStreamSemanticIntervalReference, ...]
    output_decision: LocalStreamOutputDecision
    output_decision_semantic_sha256: Sha256Digest | None
    hypothesis_references: tuple[LocalStreamSemanticIntervalReference, ...]
    cross_window_duplicate_reference_count: int
    recording_result_key: NonEmptyString
    recording_result_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        count = len(self.ordered_window_result_semantic_sha256_values)
        if not count or any(
            len(items) != count
            for items in (
                self.ordered_window_terminal_outcomes,
                self.ordered_window_semantic_evidence_refs,
                self.ordered_window_semantic_evidence_sha256_values,
            )
        ):
            raise ValueError("recording result window closures must be nonempty and aligned")
        if self.cross_window_duplicate_reference_count < 0:
            raise ValueError("cross-window duplicate count must be nonnegative")
        LocalStreamCanonicalTruth(
            six_camera_qa_semantic_sha256=self.six_camera_qa_semantic_sha256,
            qa_camera_references=self.qa_camera_references,
            event_proposal_result_semantic_sha256=(self.event_proposal_result_semantic_sha256),
            proposal_references=self.proposal_references,
            candidate_reduction_semantic_sha256=(self.candidate_reduction_semantic_sha256),
            candidate_references=self.candidate_references,
            provisional_fusion_semantic_sha256=self.provisional_fusion_semantic_sha256,
            action_references=self.action_references,
            boundary_closure_semantic_sha256=self.boundary_closure_semantic_sha256,
            boundary_references=self.boundary_references,
            output_decision=self.output_decision,
            output_decision_semantic_sha256=self.output_decision_semantic_sha256,
            hypothesis_references=self.hypothesis_references,
        )
        expected = local_stream_recording_result_v2_semantic_sha256(self)
        if self.recording_result_semantic_sha256 != expected:
            raise ValueError("recording_result_semantic_sha256 does not match v2 projection")
        if self.recording_result_key != derive_local_stream_recording_result_v2_key(expected):
            raise ValueError("recording_result_key does not match v2 result digest")
        return self


class LocalStreamMergedHypothesis(StrictModel):
    """One label-preserving union of causal proposal fragments.

    The source ordinals remain part of the local result: the union is a
    recording-level convenience projection, not a replacement for the
    content-addressed window evidence that established it.
    """

    label: NonEmptyString
    interval: NanosecondInterval
    source_ordinals: tuple[int, ...]
    source_proposal_semantic_sha256_values: tuple[Sha256Digest, ...]

    @model_validator(mode="after")
    def validate_merged_proposal(self) -> Self:
        if not self.source_ordinals or len(self.source_ordinals) != len(
            self.source_proposal_semantic_sha256_values
        ):
            raise ValueError("merged hypothesis sources must be nonempty and aligned")
        if any(ordinal < 0 for ordinal in self.source_ordinals):
            raise ValueError("merged hypothesis source ordinals must be nonnegative")
        if self.source_ordinals != tuple(sorted(self.source_ordinals)):
            raise ValueError("merged hypothesis source ordinals must be canonical")
        return self


LocalStreamMergedProposal = LocalStreamMergedHypothesis


class LocalStreamRecordingResultV3(StrictModel):
    """Ordinal-bound causal reduction over W1 results and S2 window evidence."""

    schema_version: Literal["3.0"] = "3.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = LOCAL_CONFORMANCE_EVIDENCE_CLASS
    production_eligible: Literal[False] = False
    recording_result_projection_version: Literal["local-stream-recording-result-semantic-v3"] = (
        LOCAL_STREAM_RECORDING_RESULT_V3_PROJECTION_VERSION
    )
    reduction_policy_version: Literal["local-conformance-stream-recording-reduction-v3"] = (
        LOCAL_STREAM_RECORDING_REDUCTION_V3_POLICY_VERSION
    )
    capture_scope_digest: Sha256Digest
    plan_key: NonEmptyString
    expected_plan_seal_semantic_sha256: Sha256Digest
    window_terminal_closure_semantic_sha256: Sha256Digest
    recording_finalization_semantic_sha256: Sha256Digest
    final_recording_identity: Sha256Digest
    source_timeline_origin_ns: int | None = None
    canonical_requested_interval: NanosecondInterval | None = None
    ordered_window_result_semantic_sha256_values: tuple[Sha256Digest, ...]
    ordered_window_terminal_outcomes: tuple[TerminalOutcome, ...]
    ordered_window_semantic_evidence_refs: tuple[ArtifactEvidenceRef, ...]
    ordered_window_semantic_evidence_sha256_values: tuple[Sha256Digest, ...]
    ordered_window_semantic_statuses: tuple[LocalStreamWindowSemanticStatus, ...]
    merged_hypotheses: tuple[LocalStreamMergedHypothesis, ...]
    output_decision: LocalStreamOutputDecision
    recording_result_key: NonEmptyString
    recording_result_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.source_timeline_origin_ns is None) != (self.canonical_requested_interval is None):
            raise ValueError(
                "timeline origin and canonical requested interval must be supplied together"
            )
        if isinstance(self.source_timeline_origin_ns, bool):
            raise ValueError("source_timeline_origin_ns must be an integer")
        count = len(self.ordered_window_result_semantic_sha256_values)
        if not count or any(
            len(items) != count
            for items in (
                self.ordered_window_terminal_outcomes,
                self.ordered_window_semantic_evidence_refs,
                self.ordered_window_semantic_evidence_sha256_values,
                self.ordered_window_semantic_statuses,
            )
        ):
            raise ValueError("v3 recording result window closures must be nonempty and aligned")
        merged_key = tuple(
            (item.label, item.interval.start_ns, item.interval.end_ns)
            for item in self.merged_hypotheses
        )
        if merged_key != tuple(sorted(merged_key)) or len(merged_key) != len(set(merged_key)):
            raise ValueError("merged hypotheses must be unique and canonical")
        if self.output_decision != _causal_output_decision(
            self.ordered_window_semantic_statuses,
            self.merged_hypotheses,
        ):
            raise ValueError("output_decision does not match causal window semantics")
        expected = local_stream_recording_result_v3_semantic_sha256(self)
        if self.recording_result_semantic_sha256 != expected:
            raise ValueError("recording_result_semantic_sha256 does not match v3 projection")
        if self.recording_result_key != derive_local_stream_recording_result_v3_key(expected):
            raise ValueError("recording_result_key does not match v3 result digest")
        return self


class LocalStreamRecordingResultV4(StrictModel):
    """RR4 causal result with RFC 8785-safe timestamp wire representation."""

    schema_version: Literal["4.0"] = "4.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = LOCAL_CONFORMANCE_EVIDENCE_CLASS
    production_eligible: Literal[False] = False
    recording_result_projection_version: Literal["local-stream-recording-result-semantic-v4"] = (
        LOCAL_STREAM_RECORDING_RESULT_V4_PROJECTION_VERSION
    )
    reduction_policy_version: Literal["local-conformance-stream-recording-reduction-v4"] = (
        LOCAL_STREAM_RECORDING_REDUCTION_V4_POLICY_VERSION
    )
    capture_scope_digest: Sha256Digest
    plan_key: NonEmptyString
    expected_plan_seal_semantic_sha256: Sha256Digest
    window_terminal_closure_semantic_sha256: Sha256Digest
    recording_finalization_semantic_sha256: Sha256Digest
    final_recording_identity: Sha256Digest
    source_timeline_origin_ns: Nanoseconds | None = None
    canonical_requested_interval: NanosecondInterval | None = None
    ordered_window_result_semantic_sha256_values: tuple[Sha256Digest, ...]
    ordered_window_terminal_outcomes: tuple[TerminalOutcome, ...]
    ordered_window_semantic_evidence_refs: tuple[ArtifactEvidenceRef, ...]
    ordered_window_semantic_evidence_sha256_values: tuple[Sha256Digest, ...]
    ordered_window_semantic_statuses: tuple[LocalStreamWindowSemanticStatus, ...]
    merged_hypotheses: tuple[LocalStreamMergedHypothesis, ...]
    output_decision: LocalStreamOutputDecision
    recording_result_key: NonEmptyString
    recording_result_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.source_timeline_origin_ns is None) != (self.canonical_requested_interval is None):
            raise ValueError(
                "timeline origin and canonical requested interval must be supplied together"
            )
        count = len(self.ordered_window_result_semantic_sha256_values)
        if not count or any(
            len(items) != count
            for items in (
                self.ordered_window_terminal_outcomes,
                self.ordered_window_semantic_evidence_refs,
                self.ordered_window_semantic_evidence_sha256_values,
                self.ordered_window_semantic_statuses,
            )
        ):
            raise ValueError("v4 recording result window closures must be nonempty and aligned")
        merged_key = tuple(
            (item.label, item.interval.start_ns, item.interval.end_ns)
            for item in self.merged_hypotheses
        )
        if merged_key != tuple(sorted(merged_key)) or len(merged_key) != len(set(merged_key)):
            raise ValueError("merged hypotheses must be unique and canonical")
        if self.output_decision != _causal_output_decision(
            self.ordered_window_semantic_statuses,
            self.merged_hypotheses,
        ):
            raise ValueError("output_decision does not match causal window semantics")
        expected = local_stream_recording_result_v4_semantic_sha256(self)
        if self.recording_result_semantic_sha256 != expected:
            raise ValueError("recording_result_semantic_sha256 does not match v4 projection")
        if self.recording_result_key != derive_local_stream_recording_result_v4_key(expected):
            raise ValueError("recording_result_key does not match v4 result digest")
        return self


def local_stream_recording_result_semantic_projection(
    result: LocalStreamRecordingResult,
) -> dict[str, object]:
    return {
        "recording_result_projection_version": result.recording_result_projection_version,
        "reduction_policy_version": result.reduction_policy_version,
        "evidence_class": result.evidence_class,
        "production_eligible": result.production_eligible,
        "capture_scope_digest": result.capture_scope_digest,
        "plan_key": result.plan_key,
        "expected_plan_seal_semantic_sha256": (result.expected_plan_seal_semantic_sha256),
        "window_terminal_closure_semantic_sha256": (result.window_terminal_closure_semantic_sha256),
        "recording_finalization_semantic_sha256": (result.recording_finalization_semantic_sha256),
        "final_recording_identity": result.final_recording_identity,
        "ordered_window_result_semantic_sha256_values": list(
            result.ordered_window_result_semantic_sha256_values
        ),
        "ordered_window_terminal_outcomes": [
            outcome.value for outcome in result.ordered_window_terminal_outcomes
        ],
        "ordered_accepted_terminal_semantic_sha256_values": [
            terminal.terminal_semantic_sha256 for terminal in result.accepted_terminals
        ],
        "output_decision": result.output_decision,
    }


def local_stream_recording_result_semantic_sha256(
    result: LocalStreamRecordingResult,
) -> Sha256Digest:
    return semantic_sha256(local_stream_recording_result_semantic_projection(result))


def derive_local_stream_recording_result_key(digest: Sha256Digest) -> str:
    return f"{LOCAL_STREAM_RECORDING_RESULT_KEY_NAMESPACE}:{digest}"


def create_local_stream_recording_result(
    *,
    schema_ref: SchemaRef,
    window_results: Sequence[StreamWindowResult],
    terminal_closure: WindowTerminalClosure,
    recording_finalization: RecordingFinalizationMap,
) -> LocalStreamRecordingResult:
    """Reduce exact finalized windows in their sealed expected ordinal order."""

    supplied = tuple(window_results)
    if any(not isinstance(result, StreamWindowResult) for result in supplied):
        raise TypeError("window_results must contain StreamWindowResult values")
    if not isinstance(terminal_closure, WindowTerminalClosure):
        raise TypeError("terminal_closure must be WindowTerminalClosure")
    if not isinstance(recording_finalization, RecordingFinalizationMap):
        raise TypeError("recording_finalization must be RecordingFinalizationMap")
    if (
        recording_finalization.expected_plan_seal_semantic_sha256
        != terminal_closure.plan_seal_semantic_sha256
        or recording_finalization.window_terminal_closure_semantic_sha256
        != terminal_closure.terminal_closure_digest
    ):
        raise LocalStreamRecordingReductionError(
            "recording finalization does not bind the supplied plan seal and terminal closure"
        )

    by_window: dict[tuple[str, str], StreamWindowResult] = {}
    for result in supplied:
        key = (
            result.window_subject.subject_key,
            result.window_subject.subject_semantic_sha256,
        )
        if key in by_window:
            raise LocalStreamRecordingReductionError("window results must be unique")
        by_window[key] = result

    ordered: list[StreamWindowResult] = []
    if len(by_window) != terminal_closure.expected_member_count:
        raise LocalStreamRecordingReductionError(
            "window results must cover every sealed expected ordinal"
        )
    for member in terminal_closure.members:
        key = (member.window_key, member.window_semantic_sha256)
        matched_result = by_window.get(key)
        if matched_result is None:
            raise LocalStreamRecordingReductionError(
                f"sealed expected ordinal {member.expected_ordinal} lacks a window result"
            )
        result_payload = canonical_json_bytes(matched_result)
        member_reference = member.terminal_evidence_ref
        if (
            matched_result.terminal_outcome is not member.terminal_outcome
            or member_reference.exact_sha256 != exact_bytes_sha256(result_payload)
            or member_reference.byte_count != len(result_payload)
            or member_reference.schema_ref != matched_result.schema_ref
        ):
            raise LocalStreamRecordingReductionError(
                f"window result at ordinal {member.expected_ordinal} conflicts with "
                "terminal closure"
            )
        ordered.append(matched_result)

    ordered_results = tuple(ordered)
    if any(
        result.window_subject.capture_scope_digest != recording_finalization.capture_scope_digest
        for result in ordered_results
    ):
        raise LocalStreamRecordingReductionError(
            "window results and recording finalization must share one capture scope"
        )

    mappings = recording_finalization.ordered_subject_mappings
    if len(mappings) != len(ordered_results):
        raise LocalStreamRecordingReductionError(
            "recording finalization must map every sealed window exactly once"
        )
    for ordinal, (result, mapping) in enumerate(zip(ordered_results, mappings, strict=True)):
        if (
            mapping.incremental_subject_type is not StreamSubjectType.INCREMENTAL_WINDOW
            or mapping.incremental_subject_key != result.window_subject.subject_key
            or mapping.incremental_subject_semantic_sha256
            != result.window_subject.subject_semantic_sha256
        ):
            raise LocalStreamRecordingReductionError(
                f"recording finalization mapping at ordinal {ordinal} conflicts with its window"
            )

    accepted_terminals = _flatten_accepted_terminals(ordered_results)
    outcomes = tuple(result.terminal_outcome for result in ordered_results)
    values = {
        "schema_ref": schema_ref,
        "capture_scope_digest": recording_finalization.capture_scope_digest,
        "plan_key": terminal_closure.plan_key,
        "expected_plan_seal_semantic_sha256": (terminal_closure.plan_seal_semantic_sha256),
        "window_terminal_closure_semantic_sha256": (terminal_closure.terminal_closure_digest),
        "recording_finalization_semantic_sha256": (
            recording_finalization.finalization_semantic_sha256
        ),
        "final_recording_identity": recording_finalization.final_recording_identity,
        "ordered_window_result_semantic_sha256_values": tuple(
            result.window_result_semantic_sha256 for result in ordered_results
        ),
        "ordered_window_terminal_outcomes": outcomes,
        "accepted_terminals": accepted_terminals,
        "output_decision": _reduce_output_decision(outcomes),
    }
    draft = LocalStreamRecordingResult.model_construct(
        recording_result_key="x",
        recording_result_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = local_stream_recording_result_semantic_sha256(draft)
    return LocalStreamRecordingResult(
        recording_result_key=derive_local_stream_recording_result_key(digest),
        recording_result_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def local_stream_window_semantic_evidence_projection(
    evidence: LocalStreamWindowSemanticEvidence,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        evidence.model_dump(mode="json", exclude={"schema_ref", "semantic_sha256"}),
    )


def local_stream_window_semantic_evidence_sha256(
    evidence: LocalStreamWindowSemanticEvidence,
) -> Sha256Digest:
    return semantic_sha256(local_stream_window_semantic_evidence_projection(evidence))


def create_local_stream_window_semantic_evidence(
    *,
    schema_ref: SchemaRef,
    plan_key: str,
    expected_ordinal: int,
    window_result: StreamWindowResult,
    effective_interval: NanosecondInterval,
    canonical_truth: LocalStreamCanonicalTruth,
) -> LocalStreamWindowSemanticEvidence:
    """Assign exact canonical mock semantics to one overlapping stream window."""

    def overlapping(
        references: tuple[LocalStreamSemanticIntervalReference, ...],
    ) -> tuple[LocalStreamSemanticIntervalReference, ...]:
        return tuple(
            reference
            for reference in references
            if _intervals_overlap(reference.interval, effective_interval)
        )

    values: dict[str, object] = {
        "schema_ref": schema_ref,
        "plan_key": plan_key,
        "expected_ordinal": expected_ordinal,
        "window_key": window_result.window_subject.subject_key,
        "window_semantic_sha256": window_result.window_subject.subject_semantic_sha256,
        "window_result_semantic_sha256": window_result.window_result_semantic_sha256,
        "effective_interval": effective_interval,
        "six_camera_qa_semantic_sha256": canonical_truth.six_camera_qa_semantic_sha256,
        "qa_camera_references": canonical_truth.qa_camera_references,
        "event_proposal_result_semantic_sha256": (
            canonical_truth.event_proposal_result_semantic_sha256
        ),
        "proposal_references": overlapping(canonical_truth.proposal_references),
        "candidate_reduction_semantic_sha256": (
            canonical_truth.candidate_reduction_semantic_sha256
        ),
        "candidate_references": overlapping(canonical_truth.candidate_references),
        "provisional_fusion_semantic_sha256": (canonical_truth.provisional_fusion_semantic_sha256),
        "action_references": overlapping(canonical_truth.action_references),
        "boundary_closure_semantic_sha256": (canonical_truth.boundary_closure_semantic_sha256),
        "boundary_references": overlapping(canonical_truth.boundary_references),
        "output_decision": canonical_truth.output_decision,
        "output_decision_semantic_sha256": (canonical_truth.output_decision_semantic_sha256),
        "hypothesis_references": overlapping(canonical_truth.hypothesis_references),
    }
    draft = LocalStreamWindowSemanticEvidence.model_construct(
        semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    return LocalStreamWindowSemanticEvidence(
        semantic_sha256=local_stream_window_semantic_evidence_sha256(draft),
        **cast(dict[str, Any], values),
    )


def local_stream_recording_result_v2_semantic_projection(
    result: LocalStreamRecordingResultV2,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        result.model_dump(
            mode="json",
            exclude={
                "schema_ref",
                "recording_result_key",
                "recording_result_semantic_sha256",
            },
        ),
    )


def local_stream_recording_result_v2_semantic_sha256(
    result: LocalStreamRecordingResultV2,
) -> Sha256Digest:
    return semantic_sha256(local_stream_recording_result_v2_semantic_projection(result))


def derive_local_stream_recording_result_v2_key(digest: Sha256Digest) -> str:
    return f"{LOCAL_STREAM_RECORDING_RESULT_V2_KEY_NAMESPACE}:{digest}"


def create_local_stream_recording_result_v2(
    *,
    schema_ref: SchemaRef,
    window_results: Sequence[StreamWindowResult],
    window_semantic_evidence: Sequence[
        tuple[LocalStreamWindowSemanticEvidence, ArtifactEvidenceRef]
    ],
    terminal_closure: WindowTerminalClosure,
    recording_finalization: RecordingFinalizationMap,
) -> LocalStreamRecordingResultV2:
    """Rebuild recording semantics solely from every sealed window artifact."""

    base = create_local_stream_recording_result(
        schema_ref=schema_ref,
        window_results=window_results,
        terminal_closure=terminal_closure,
        recording_finalization=recording_finalization,
    )
    disallowed = {
        TerminalOutcome.FAILED,
        TerminalOutcome.CANCELLED,
        TerminalOutcome.EXPIRED,
        TerminalOutcome.QUARANTINED,
        TerminalOutcome.LATE_INPUT,
        TerminalOutcome.INCOMPLETE,
        TerminalOutcome.INVALIDATED,
    }
    if any(outcome in disallowed for outcome in base.ordered_window_terminal_outcomes):
        raise LocalStreamRecordingReductionError(
            "failed or incomplete required window cannot produce recording truth"
        )

    results_by_window = {
        (
            result.window_subject.subject_key,
            result.window_subject.subject_semantic_sha256,
        ): result
        for result in window_results
    }
    supplied_evidence = tuple(window_semantic_evidence)
    if any(
        not isinstance(evidence, LocalStreamWindowSemanticEvidence)
        or not isinstance(reference, ArtifactEvidenceRef)
        for evidence, reference in supplied_evidence
    ):
        raise TypeError("window_semantic_evidence must contain typed evidence/reference pairs")
    evidence_by_window: dict[
        tuple[str, str],
        tuple[LocalStreamWindowSemanticEvidence, ArtifactEvidenceRef],
    ] = {}
    for evidence, reference in supplied_evidence:
        key = (evidence.window_key, evidence.window_semantic_sha256)
        if key in evidence_by_window:
            raise LocalStreamRecordingReductionError(
                "window semantic evidence must be unique per sealed window"
            )
        payload = canonical_json_bytes(evidence)
        if (
            reference.schema_ref != evidence.schema_ref
            or reference.media_type != "application/json"
            or reference.exact_sha256 != exact_bytes_sha256(payload)
            or reference.byte_count != len(payload)
        ):
            raise LocalStreamRecordingReductionError(
                "window semantic evidence differs from its content-addressed reference"
            )
        evidence_by_window[key] = (evidence, reference)

    ordered_pairs: list[tuple[LocalStreamWindowSemanticEvidence, ArtifactEvidenceRef]] = []
    for member in terminal_closure.members:
        key = (member.window_key, member.window_semantic_sha256)
        result = results_by_window.get(key)
        pair = evidence_by_window.get(key)
        if result is None or pair is None:
            raise LocalStreamRecordingReductionError(
                f"sealed expected ordinal {member.expected_ordinal} lacks semantic evidence"
            )
        evidence, _ = pair
        if (
            evidence.plan_key != terminal_closure.plan_key
            or evidence.expected_ordinal != member.expected_ordinal
            or evidence.window_result_semantic_sha256 != result.window_result_semantic_sha256
        ):
            raise LocalStreamRecordingReductionError(
                f"semantic evidence at ordinal {member.expected_ordinal} conflicts with its window"
            )
        ordered_pairs.append(pair)
    if len(ordered_pairs) != len(supplied_evidence):
        raise LocalStreamRecordingReductionError("semantic evidence includes an undeclared window")

    evidences = tuple(item[0] for item in ordered_pairs)
    first = evidences[0]
    if any(
        (
            evidence.six_camera_qa_semantic_sha256,
            evidence.qa_camera_references,
            evidence.event_proposal_result_semantic_sha256,
            evidence.candidate_reduction_semantic_sha256,
            evidence.provisional_fusion_semantic_sha256,
            evidence.boundary_closure_semantic_sha256,
            evidence.output_decision,
            evidence.output_decision_semantic_sha256,
        )
        != (
            first.six_camera_qa_semantic_sha256,
            first.qa_camera_references,
            first.event_proposal_result_semantic_sha256,
            first.candidate_reduction_semantic_sha256,
            first.provisional_fusion_semantic_sha256,
            first.boundary_closure_semantic_sha256,
            first.output_decision,
            first.output_decision_semantic_sha256,
        )
        for evidence in evidences[1:]
    ):
        raise LocalStreamRecordingReductionError(
            "window semantic evidence disagrees on recording-global roots"
        )

    proposal_refs, proposal_duplicates = _deduplicate_semantic_references(
        tuple(evidence.proposal_references for evidence in evidences)
    )
    candidate_refs, candidate_duplicates = _deduplicate_semantic_references(
        tuple(evidence.candidate_references for evidence in evidences)
    )
    action_refs, action_duplicates = _deduplicate_semantic_references(
        tuple(evidence.action_references for evidence in evidences)
    )
    boundary_refs, boundary_duplicates = _deduplicate_semantic_references(
        tuple(evidence.boundary_references for evidence in evidences)
    )
    hypothesis_refs, hypothesis_duplicates = _deduplicate_semantic_references(
        tuple(evidence.hypothesis_references for evidence in evidences)
    )
    truth = LocalStreamCanonicalTruth(
        six_camera_qa_semantic_sha256=first.six_camera_qa_semantic_sha256,
        qa_camera_references=first.qa_camera_references,
        event_proposal_result_semantic_sha256=(first.event_proposal_result_semantic_sha256),
        proposal_references=proposal_refs,
        candidate_reduction_semantic_sha256=(first.candidate_reduction_semantic_sha256),
        candidate_references=candidate_refs,
        provisional_fusion_semantic_sha256=first.provisional_fusion_semantic_sha256,
        action_references=action_refs,
        boundary_closure_semantic_sha256=first.boundary_closure_semantic_sha256,
        boundary_references=boundary_refs,
        output_decision=first.output_decision,
        output_decision_semantic_sha256=first.output_decision_semantic_sha256,
        hypothesis_references=hypothesis_refs,
    )
    values: dict[str, object] = {
        "schema_ref": schema_ref,
        "capture_scope_digest": base.capture_scope_digest,
        "plan_key": base.plan_key,
        "expected_plan_seal_semantic_sha256": (base.expected_plan_seal_semantic_sha256),
        "window_terminal_closure_semantic_sha256": (base.window_terminal_closure_semantic_sha256),
        "recording_finalization_semantic_sha256": (base.recording_finalization_semantic_sha256),
        "final_recording_identity": base.final_recording_identity,
        "ordered_window_result_semantic_sha256_values": (
            base.ordered_window_result_semantic_sha256_values
        ),
        "ordered_window_terminal_outcomes": base.ordered_window_terminal_outcomes,
        "ordered_window_semantic_evidence_refs": tuple(item[1] for item in ordered_pairs),
        "ordered_window_semantic_evidence_sha256_values": tuple(
            evidence.semantic_sha256 for evidence in evidences
        ),
        "six_camera_qa_semantic_sha256": truth.six_camera_qa_semantic_sha256,
        "qa_camera_references": truth.qa_camera_references,
        "event_proposal_result_semantic_sha256": (truth.event_proposal_result_semantic_sha256),
        "proposal_references": truth.proposal_references,
        "candidate_reduction_semantic_sha256": (truth.candidate_reduction_semantic_sha256),
        "candidate_references": truth.candidate_references,
        "provisional_fusion_semantic_sha256": (truth.provisional_fusion_semantic_sha256),
        "action_references": truth.action_references,
        "boundary_closure_semantic_sha256": (truth.boundary_closure_semantic_sha256),
        "boundary_references": truth.boundary_references,
        "output_decision": truth.output_decision,
        "output_decision_semantic_sha256": (truth.output_decision_semantic_sha256),
        "hypothesis_references": truth.hypothesis_references,
        "cross_window_duplicate_reference_count": (
            proposal_duplicates
            + candidate_duplicates
            + action_duplicates
            + boundary_duplicates
            + hypothesis_duplicates
        ),
    }
    draft = LocalStreamRecordingResultV2.model_construct(
        recording_result_key="x",
        recording_result_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = local_stream_recording_result_v2_semantic_sha256(draft)
    return LocalStreamRecordingResultV2(
        recording_result_key=derive_local_stream_recording_result_v2_key(digest),
        recording_result_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def validate_local_stream_recording_result_v2_truth(
    result: LocalStreamRecordingResultV2,
    expected: LocalStreamCanonicalTruth,
) -> None:
    actual = LocalStreamCanonicalTruth(
        six_camera_qa_semantic_sha256=result.six_camera_qa_semantic_sha256,
        qa_camera_references=result.qa_camera_references,
        event_proposal_result_semantic_sha256=(result.event_proposal_result_semantic_sha256),
        proposal_references=result.proposal_references,
        candidate_reduction_semantic_sha256=(result.candidate_reduction_semantic_sha256),
        candidate_references=result.candidate_references,
        provisional_fusion_semantic_sha256=result.provisional_fusion_semantic_sha256,
        action_references=result.action_references,
        boundary_closure_semantic_sha256=result.boundary_closure_semantic_sha256,
        boundary_references=result.boundary_references,
        output_decision=result.output_decision,
        output_decision_semantic_sha256=result.output_decision_semantic_sha256,
        hypothesis_references=result.hypothesis_references,
    )
    if actual != expected:
        raise LocalStreamRecordingReductionError(
            "stream recording truth disagrees with canonical recording semantics"
        )


def local_stream_recording_result_v3_semantic_projection(
    result: LocalStreamRecordingResultV3,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        result.model_dump(
            mode="json",
            exclude={
                "schema_ref",
                "recording_result_key",
                "recording_result_semantic_sha256",
            },
        ),
    )


def local_stream_recording_result_v3_semantic_sha256(
    result: LocalStreamRecordingResultV3,
) -> Sha256Digest:
    return semantic_sha256(local_stream_recording_result_v3_semantic_projection(result))


def derive_local_stream_recording_result_v3_key(digest: Sha256Digest) -> str:
    return f"{LOCAL_STREAM_RECORDING_RESULT_V3_KEY_NAMESPACE}:{digest}"


def _validate_v3_ordered_window_pairs(
    *,
    window_results: tuple[StreamWindowResult, ...],
    window_semantic_evidence: tuple[LocalStreamWindowSemanticEvidenceV2, ...],
    terminal_closure: WindowTerminalClosure,
) -> None:
    """Bind one W1 and one S2 at each sealed ordinal, without reordering input."""

    if (
        len(window_results) != terminal_closure.expected_member_count
        or len(window_semantic_evidence) != terminal_closure.expected_member_count
    ):
        raise LocalStreamRecordingReductionError(
            "every sealed expected ordinal requires exactly one W1 and one S2"
        )
    for ordinal, (member, result, evidence) in enumerate(
        zip(
            terminal_closure.members,
            window_results,
            window_semantic_evidence,
            strict=True,
        )
    ):
        if member.expected_ordinal != ordinal:
            raise LocalStreamRecordingReductionError("terminal closure members are out of order")
        if (
            result.window_subject.subject_key != member.window_key
            or result.window_subject.subject_semantic_sha256 != member.window_semantic_sha256
            or evidence.expected_ordinal != ordinal
            or evidence.expected_ordinal != member.expected_ordinal
            or evidence.window_key != member.window_key
            or evidence.window_semantic_sha256 != member.window_semantic_sha256
        ):
            raise LocalStreamRecordingReductionError(
                f"W1/S2 pair at ordinal {ordinal} is missing or out of order"
            )
        payload = canonical_json_bytes(evidence)
        evidence_ref = result.result_evidence_ref
        if (
            result.result_semantic_evidence_sha256 != evidence.semantic_sha256
            or evidence_ref.schema_ref != evidence.schema_ref
            or evidence_ref.media_type != "application/json"
            or evidence_ref.exact_sha256 != exact_bytes_sha256(payload)
            or evidence_ref.byte_count != len(payload)
        ):
            raise LocalStreamRecordingReductionError(
                f"W1 result evidence does not exactly bind S2 at ordinal {ordinal}"
            )


def _normalize_v3_causal_interval(
    interval: NanosecondInterval,
    *,
    source_timeline_origin_ns: int | None,
    canonical_requested_interval: NanosecondInterval | None,
) -> NanosecondInterval:
    """Apply the finalizer's absolute-source to recording-local interval mapping."""

    if source_timeline_origin_ns is None and canonical_requested_interval is None:
        return interval
    if source_timeline_origin_ns is None or canonical_requested_interval is None:
        raise LocalStreamRecordingReductionError(
            "timeline origin and canonical requested interval must be supplied together"
        )
    start_ns = max(
        canonical_requested_interval.start_ns + interval.start_ns - source_timeline_origin_ns,
        canonical_requested_interval.start_ns,
    )
    end_ns = min(
        canonical_requested_interval.start_ns + interval.end_ns - source_timeline_origin_ns,
        canonical_requested_interval.end_ns,
    )
    if start_ns >= end_ns:
        raise LocalStreamRecordingReductionError(
            "causal proposal has no recording-relative overlap with requested interval"
        )
    return NanosecondInterval(start_ns=start_ns, end_ns=end_ns)


def _merge_causal_hypotheses(
    evidences: tuple[LocalStreamWindowSemanticEvidenceV2, ...],
    *,
    source_timeline_origin_ns: int | None,
    canonical_requested_interval: NanosecondInterval | None,
) -> tuple[LocalStreamMergedHypothesis, ...]:
    fragments = sorted(
        (
            (
                evidence.proposal_label,
                _normalize_v3_causal_interval(
                    cast(NanosecondInterval, evidence.proposal_interval),
                    source_timeline_origin_ns=source_timeline_origin_ns,
                    canonical_requested_interval=canonical_requested_interval,
                ),
                ordinal,
                evidence.proposal_semantic_sha256,
            )
            for ordinal, evidence in enumerate(evidences)
            if evidence.semantic_status == "PROPOSED"
        ),
        key=lambda item: (
            cast(str, item[0]),
            item[1].start_ns,
            item[1].end_ns,
            cast(str, item[3]),
        ),
    )
    merged: list[LocalStreamMergedHypothesis] = []
    for label, interval, ordinal, digest in fragments:
        # The S2 validator makes these fields non-null for PROPOSED evidence.
        assert label is not None and interval is not None and digest is not None
        if merged and merged[-1].label == label and interval.start_ns <= merged[-1].interval.end_ns:
            previous = merged.pop()
            merged.append(
                LocalStreamMergedHypothesis(
                    label=label,
                    interval=NanosecondInterval(
                        start_ns=previous.interval.start_ns,
                        end_ns=max(previous.interval.end_ns, interval.end_ns),
                    ),
                    source_ordinals=(*previous.source_ordinals, ordinal),
                    source_proposal_semantic_sha256_values=(
                        *previous.source_proposal_semantic_sha256_values,
                        digest,
                    ),
                )
            )
        else:
            merged.append(
                LocalStreamMergedHypothesis(
                    label=label,
                    interval=interval,
                    source_ordinals=(ordinal,),
                    source_proposal_semantic_sha256_values=(digest,),
                )
            )
    return tuple(merged)


def _causal_output_decision(
    statuses: tuple[LocalStreamWindowSemanticStatus, ...],
    merged_hypotheses: tuple[LocalStreamMergedHypothesis, ...],
) -> LocalStreamOutputDecision:
    if any(status == "ABSTAINED" for status in statuses):
        return "ABSTAINED"
    if not merged_hypotheses:
        return "NO_EVENTS"
    return "ADMITTED"


def create_local_stream_recording_result_v3(
    *,
    schema_ref: SchemaRef,
    window_results: Sequence[StreamWindowResult],
    window_semantic_evidence: Sequence[LocalStreamWindowSemanticEvidenceV2],
    terminal_closure: WindowTerminalClosure,
    recording_finalization: RecordingFinalizationMap,
    source_timeline_origin_ns: int | None = None,
    canonical_requested_interval: NanosecondInterval | None = None,
) -> LocalStreamRecordingResultV3:
    """Reduce ordinal-aligned W1/S2 evidence into a causal local recording result.

    Unlike V1/V2, V3 intentionally treats order as evidence.  A caller must supply
    both W1 results and their S2 payloads in the sealed expected-ordinal order;
    accepting and silently sorting either sequence would weaken the causal binding.
    """

    if (source_timeline_origin_ns is None) != (canonical_requested_interval is None):
        raise ValueError(
            "timeline origin and canonical requested interval must be supplied together"
        )
    if isinstance(source_timeline_origin_ns, bool):
        raise TypeError("source_timeline_origin_ns must be an integer")
    results = tuple(window_results)
    evidences = tuple(window_semantic_evidence)
    if any(not isinstance(item, StreamWindowResult) for item in results):
        raise TypeError("window_results must contain StreamWindowResult values")
    if any(not isinstance(item, LocalStreamWindowSemanticEvidenceV2) for item in evidences):
        raise TypeError(
            "window_semantic_evidence must contain LocalStreamWindowSemanticEvidenceV2 values"
        )
    _validate_v3_ordered_window_pairs(
        window_results=results,
        window_semantic_evidence=evidences,
        terminal_closure=terminal_closure,
    )
    base = create_local_stream_recording_result(
        schema_ref=schema_ref,
        window_results=results,
        terminal_closure=terminal_closure,
        recording_finalization=recording_finalization,
    )
    disallowed = {
        TerminalOutcome.FAILED,
        TerminalOutcome.CANCELLED,
        TerminalOutcome.EXPIRED,
        TerminalOutcome.QUARANTINED,
        TerminalOutcome.LATE_INPUT,
        TerminalOutcome.INCOMPLETE,
        TerminalOutcome.INVALIDATED,
    }
    if any(outcome in disallowed for outcome in base.ordered_window_terminal_outcomes):
        raise LocalStreamRecordingReductionError(
            "failed or incomplete required window cannot produce causal recording truth"
        )
    for ordinal, evidence in enumerate(evidences):
        if evidence.plan_key != base.plan_key:
            raise LocalStreamRecordingReductionError(
                f"S2 evidence at ordinal {ordinal} conflicts with the sealed plan"
            )

    merged_hypotheses = _merge_causal_hypotheses(
        evidences,
        source_timeline_origin_ns=source_timeline_origin_ns,
        canonical_requested_interval=canonical_requested_interval,
    )
    values: dict[str, object] = {
        "schema_ref": schema_ref,
        "capture_scope_digest": base.capture_scope_digest,
        "plan_key": base.plan_key,
        "expected_plan_seal_semantic_sha256": base.expected_plan_seal_semantic_sha256,
        "window_terminal_closure_semantic_sha256": base.window_terminal_closure_semantic_sha256,
        "recording_finalization_semantic_sha256": base.recording_finalization_semantic_sha256,
        "final_recording_identity": base.final_recording_identity,
        "source_timeline_origin_ns": source_timeline_origin_ns,
        "canonical_requested_interval": canonical_requested_interval,
        "ordered_window_result_semantic_sha256_values": (
            base.ordered_window_result_semantic_sha256_values
        ),
        "ordered_window_terminal_outcomes": base.ordered_window_terminal_outcomes,
        "ordered_window_semantic_evidence_refs": tuple(
            result.result_evidence_ref for result in results
        ),
        "ordered_window_semantic_evidence_sha256_values": tuple(
            evidence.semantic_sha256 for evidence in evidences
        ),
        "ordered_window_semantic_statuses": tuple(
            evidence.semantic_status for evidence in evidences
        ),
        "merged_hypotheses": merged_hypotheses,
        "output_decision": _causal_output_decision(
            tuple(evidence.semantic_status for evidence in evidences),
            merged_hypotheses,
        ),
    }
    draft = LocalStreamRecordingResultV3.model_construct(
        recording_result_key="x",
        recording_result_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = local_stream_recording_result_v3_semantic_sha256(draft)
    return LocalStreamRecordingResultV3(
        recording_result_key=derive_local_stream_recording_result_v3_key(digest),
        recording_result_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


# Builder spelling is retained for integrations that describe RR3 as a builder.
build_local_stream_recording_result_v3 = create_local_stream_recording_result_v3


def validate_local_stream_recording_result_v3_truth(
    result: LocalStreamRecordingResultV3 | LocalStreamRecordingResultV4,
    expected: LocalStreamCanonicalTruth,
    *,
    source_timeline_origin_ns: int | None = None,
    canonical_requested_interval: NanosecondInterval | None = None,
) -> None:
    """Check V3/V4 business compatibility with canonical truth, never exact digests.

    V3 uses independently produced local causal fragments, so comparing their
    content hashes to canonical-mock references would be invalid.  Compatibility is
    instead decision equality plus bidirectional interval overlap for admitted
    proposals.  Labels intentionally remain local-causal labels.
    """

    if result.output_decision != expected.output_decision:
        raise LocalStreamRecordingReductionError(
            "causal recording output decision is incompatible with canonical truth"
        )
    if result.output_decision != "ADMITTED":
        return
    # Builder-produced V3 hypotheses are already recording-local when their
    # normalization context is persisted.  For an older/external local result,
    # callers may supply the same explicit context to normalize before compare.
    if result.source_timeline_origin_ns is not None:
        actual_intervals = tuple(item.interval for item in result.merged_hypotheses)
    else:
        actual_intervals = tuple(
            _normalize_v3_causal_interval(
                item.interval,
                source_timeline_origin_ns=source_timeline_origin_ns,
                canonical_requested_interval=canonical_requested_interval,
            )
            for item in result.merged_hypotheses
        )
    expected_intervals = tuple(item.interval for item in expected.proposal_references)
    if not actual_intervals:
        raise LocalStreamRecordingReductionError(
            "admitted causal recording result requires a merged hypothesis"
        )
    if expected_intervals and (
        any(
            not any(_intervals_overlap(actual, canonical) for canonical in expected_intervals)
            for actual in actual_intervals
        )
        or any(
            not any(_intervals_overlap(canonical, actual) for actual in actual_intervals)
            for canonical in expected_intervals
        )
    ):
        raise LocalStreamRecordingReductionError(
            "causal proposal intervals are incompatible with canonical truth"
        )


def local_stream_recording_result_v4_semantic_projection(
    result: LocalStreamRecordingResultV4,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        result.model_dump(
            mode="json",
            exclude={
                "schema_ref",
                "recording_result_key",
                "recording_result_semantic_sha256",
            },
        ),
    )


def local_stream_recording_result_v4_semantic_sha256(
    result: LocalStreamRecordingResultV4,
) -> Sha256Digest:
    return semantic_sha256(local_stream_recording_result_v4_semantic_projection(result))


def derive_local_stream_recording_result_v4_key(digest: Sha256Digest) -> str:
    return f"{LOCAL_STREAM_RECORDING_RESULT_V4_KEY_NAMESPACE}:{digest}"


def create_local_stream_recording_result_v4(
    *,
    schema_ref: SchemaRef,
    window_results: Sequence[StreamWindowResult],
    window_semantic_evidence: Sequence[LocalStreamWindowSemanticEvidenceV2],
    terminal_closure: WindowTerminalClosure,
    recording_finalization: RecordingFinalizationMap,
    source_timeline_origin_ns: Nanoseconds | None = None,
    canonical_requested_interval: NanosecondInterval | None = None,
) -> LocalStreamRecordingResultV4:
    """Build RR4 causal truth, encoding epoch nanoseconds as decimal JSON strings."""

    if (source_timeline_origin_ns is None) != (canonical_requested_interval is None):
        raise ValueError(
            "timeline origin and canonical requested interval must be supplied together"
        )
    results = tuple(window_results)
    evidences = tuple(window_semantic_evidence)
    if any(not isinstance(item, StreamWindowResult) for item in results):
        raise TypeError("window_results must contain StreamWindowResult values")
    if any(not isinstance(item, LocalStreamWindowSemanticEvidenceV2) for item in evidences):
        raise TypeError(
            "window_semantic_evidence must contain LocalStreamWindowSemanticEvidenceV2 values"
        )
    _validate_v3_ordered_window_pairs(
        window_results=results,
        window_semantic_evidence=evidences,
        terminal_closure=terminal_closure,
    )
    base = create_local_stream_recording_result(
        schema_ref=schema_ref,
        window_results=results,
        terminal_closure=terminal_closure,
        recording_finalization=recording_finalization,
    )
    disallowed = {
        TerminalOutcome.FAILED,
        TerminalOutcome.CANCELLED,
        TerminalOutcome.EXPIRED,
        TerminalOutcome.QUARANTINED,
        TerminalOutcome.LATE_INPUT,
        TerminalOutcome.INCOMPLETE,
        TerminalOutcome.INVALIDATED,
    }
    if any(outcome in disallowed for outcome in base.ordered_window_terminal_outcomes):
        raise LocalStreamRecordingReductionError(
            "failed or incomplete required window cannot produce causal recording truth"
        )
    for ordinal, evidence in enumerate(evidences):
        if evidence.plan_key != base.plan_key:
            raise LocalStreamRecordingReductionError(
                f"S2 evidence at ordinal {ordinal} conflicts with the sealed plan"
            )

    merged_hypotheses = _merge_causal_hypotheses(
        evidences,
        source_timeline_origin_ns=source_timeline_origin_ns,
        canonical_requested_interval=canonical_requested_interval,
    )
    values: dict[str, object] = {
        "schema_ref": schema_ref,
        "capture_scope_digest": base.capture_scope_digest,
        "plan_key": base.plan_key,
        "expected_plan_seal_semantic_sha256": base.expected_plan_seal_semantic_sha256,
        "window_terminal_closure_semantic_sha256": base.window_terminal_closure_semantic_sha256,
        "recording_finalization_semantic_sha256": base.recording_finalization_semantic_sha256,
        "final_recording_identity": base.final_recording_identity,
        "source_timeline_origin_ns": source_timeline_origin_ns,
        "canonical_requested_interval": canonical_requested_interval,
        "ordered_window_result_semantic_sha256_values": (
            base.ordered_window_result_semantic_sha256_values
        ),
        "ordered_window_terminal_outcomes": base.ordered_window_terminal_outcomes,
        "ordered_window_semantic_evidence_refs": tuple(
            result.result_evidence_ref for result in results
        ),
        "ordered_window_semantic_evidence_sha256_values": tuple(
            evidence.semantic_sha256 for evidence in evidences
        ),
        "ordered_window_semantic_statuses": tuple(
            evidence.semantic_status for evidence in evidences
        ),
        "merged_hypotheses": merged_hypotheses,
        "output_decision": _causal_output_decision(
            tuple(evidence.semantic_status for evidence in evidences),
            merged_hypotheses,
        ),
    }
    draft = LocalStreamRecordingResultV4.model_construct(
        recording_result_key="x",
        recording_result_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = local_stream_recording_result_v4_semantic_sha256(draft)
    return LocalStreamRecordingResultV4(
        recording_result_key=derive_local_stream_recording_result_v4_key(digest),
        recording_result_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


build_local_stream_recording_result_v4 = create_local_stream_recording_result_v4


def _deduplicate_semantic_references(
    groups: tuple[tuple[LocalStreamSemanticIntervalReference, ...], ...],
) -> tuple[tuple[LocalStreamSemanticIntervalReference, ...], int]:
    by_key: dict[tuple[str, str], LocalStreamSemanticIntervalReference] = {}
    supplied_count = 0
    for group in groups:
        for reference in group:
            supplied_count += 1
            key = (reference.kind, reference.logical_key)
            existing = by_key.get(key)
            if existing is not None and existing != reference:
                raise LocalStreamRecordingReductionError(
                    "one semantic logical key resolves to conflicting window references"
                )
            by_key.setdefault(key, reference)
    ordered = tuple(sorted(by_key.values(), key=_semantic_reference_key))
    return ordered, supplied_count - len(ordered)


def _intervals_overlap(left: NanosecondInterval, right: NanosecondInterval) -> bool:
    return left.start_ns < right.end_ns and right.start_ns < left.end_ns


def _flatten_accepted_terminals(
    ordered_results: tuple[StreamWindowResult, ...],
) -> tuple[StreamInferenceTerminalReference, ...]:
    by_digest: dict[str, StreamInferenceTerminalReference] = {}
    for result in ordered_results:
        for terminal in result.accepted_terminals:
            existing = by_digest.get(terminal.terminal_semantic_sha256)
            if existing is not None and existing != terminal:
                raise LocalStreamRecordingReductionError(
                    "one accepted terminal digest resolves to conflicting references"
                )
            by_digest.setdefault(terminal.terminal_semantic_sha256, terminal)
    return tuple(sorted(by_digest.values(), key=_terminal_reference_key))


def _terminal_reference_key(
    terminal: StreamInferenceTerminalReference,
) -> tuple[str, str, str, str, str]:
    return (
        terminal.terminal_semantic_sha256,
        terminal.window_semantic_sha256,
        terminal.stream_inference_logical_id,
        terminal.inference_attempt_id,
        terminal.artifact_ref.exact_sha256,
    )


def _reduce_output_decision(
    outcomes: tuple[TerminalOutcome, ...],
) -> LocalStreamOutputDecision:
    if outcomes and all(outcome is TerminalOutcome.NO_EVENTS for outcome in outcomes):
        return "NO_EVENTS"
    if outcomes and all(
        outcome in {TerminalOutcome.SUCCEEDED, TerminalOutcome.NO_EVENTS} for outcome in outcomes
    ):
        return "ADMITTED"
    return "ABSTAINED"


__all__ = [
    "LOCAL_CONFORMANCE_EVIDENCE_CLASS",
    "LOCAL_STREAM_RECORDING_REDUCTION_POLICY_VERSION",
    "LOCAL_STREAM_RECORDING_REDUCTION_V2_POLICY_VERSION",
    "LOCAL_STREAM_RECORDING_REDUCTION_V3_POLICY_VERSION",
    "LOCAL_STREAM_RECORDING_REDUCTION_V4_POLICY_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_KEY_NAMESPACE",
    "LOCAL_STREAM_RECORDING_RESULT_PROJECTION_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID",
    "LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_V2_KEY_NAMESPACE",
    "LOCAL_STREAM_RECORDING_RESULT_V2_PROJECTION_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_V3_KEY_NAMESPACE",
    "LOCAL_STREAM_RECORDING_RESULT_V3_PROJECTION_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_V3_SCHEMA_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_V4_KEY_NAMESPACE",
    "LOCAL_STREAM_RECORDING_RESULT_V4_PROJECTION_VERSION",
    "LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION",
    "LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_PROJECTION_VERSION",
    "LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID",
    "LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION",
    "LocalStreamCanonicalTruth",
    "LocalStreamMergedHypothesis",
    "LocalStreamMergedProposal",
    "LocalStreamOutputDecision",
    "LocalStreamQaCameraReference",
    "LocalStreamRecordingReductionError",
    "LocalStreamRecordingResult",
    "LocalStreamRecordingResultV2",
    "LocalStreamRecordingResultV3",
    "LocalStreamRecordingResultV4",
    "LocalStreamSemanticIntervalReference",
    "LocalStreamWindowSemanticEvidence",
    "build_local_stream_recording_result_v3",
    "build_local_stream_recording_result_v4",
    "create_local_stream_recording_result",
    "create_local_stream_recording_result_v2",
    "create_local_stream_recording_result_v3",
    "create_local_stream_recording_result_v4",
    "create_local_stream_window_semantic_evidence",
    "derive_local_stream_recording_result_key",
    "derive_local_stream_recording_result_v2_key",
    "derive_local_stream_recording_result_v3_key",
    "derive_local_stream_recording_result_v4_key",
    "local_stream_recording_result_semantic_projection",
    "local_stream_recording_result_semantic_sha256",
    "local_stream_recording_result_v2_semantic_projection",
    "local_stream_recording_result_v2_semantic_sha256",
    "local_stream_recording_result_v3_semantic_projection",
    "local_stream_recording_result_v3_semantic_sha256",
    "local_stream_recording_result_v4_semantic_projection",
    "local_stream_recording_result_v4_semantic_sha256",
    "local_stream_window_semantic_evidence_projection",
    "local_stream_window_semantic_evidence_sha256",
    "validate_local_stream_recording_result_v2_truth",
    "validate_local_stream_recording_result_v3_truth",
]
