"""Terminal closure and immutable EOS finalization mapping contracts."""

from __future__ import annotations

from typing import Any, Literal, Self, cast

from pydantic import model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    NonEmptyString,
    NonNegativeInt,
    StreamSubjectType,
    TerminalOutcome,
)
from robata.contracts.stream_planning import (
    ExpectedWindowDeclaration,
    ExpectedWindowPlanSeal,
    compute_ordered_expected_member_root,
    derive_work_item_id,
)

TERMINAL_MEMBER_PROJECTION_VERSION = "window-terminal-member-semantic-v1"
TERMINAL_MEMBER_ROOT_VERSION = "window-terminal-member-root-v1"
TERMINAL_CLOSURE_PROJECTION_VERSION = "window-terminal-closure-semantic-v1"
FINALIZATION_PROJECTION_VERSION = "recording-finalization-map-semantic-v1"
FINALIZATION_POLICY_VERSION = "recording-finalization-policy-v1"
FINALIZATION_KEY_NAMESPACE = "recording-finalization-map-v1"
STREAM_FINALIZATION_WIRE_VERSION: Literal["1.0"] = "1.0"
WINDOW_TERMINAL_MEMBER_SCHEMA_ID = "https://schemas.robata.dev/window-terminal-member"
WINDOW_TERMINAL_MEMBER_SCHEMA_VERSION = "1.0.0"
WINDOW_TERMINAL_CLOSURE_SCHEMA_ID = "https://schemas.robata.dev/window-terminal-closure"
WINDOW_TERMINAL_CLOSURE_SCHEMA_VERSION = "1.0.0"
RECORDING_FINALIZATION_SCHEMA_ID = "https://schemas.robata.dev/recording-finalization-map"
RECORDING_FINALIZATION_SCHEMA_VERSION = "1.0.0"


class WindowTerminalMember(StrictModel):
    """One exact terminal outcome for one sealed expected ordinal."""

    schema_version: Literal["1.0"] = STREAM_FINALIZATION_WIRE_VERSION
    schema_ref: SchemaRef
    plan_key: NonEmptyString
    expected_ordinal: NonNegativeInt
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    terminal_outcome: TerminalOutcome
    terminal_work_item_id: OpaqueUuid
    terminal_work_logical_key: NonEmptyString
    terminal_evidence_ref: ArtifactEvidenceRef
    terminal_policy_version: NonEmptyString
    member_projection_version: NonEmptyString = TERMINAL_MEMBER_PROJECTION_VERSION
    member_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if self.member_projection_version != TERMINAL_MEMBER_PROJECTION_VERSION:
            raise ValueError("terminal member uses the registered projection version")
        if self.window_key != f"incremental-window-v1:{self.window_semantic_sha256}":
            raise ValueError("terminal window key must bind to window_semantic_sha256")
        work_prefix = "stream-work-v1:"
        if not self.terminal_work_logical_key.startswith(work_prefix):
            raise ValueError("terminal work logical key must use the stream-work-v1 namespace")
        work_digest = self.terminal_work_logical_key.removeprefix(work_prefix)
        if (
            len(work_digest) != 64
            or any(character not in "0123456789abcdef" for character in work_digest)
            or self.terminal_work_item_id != derive_work_item_id(work_digest)
        ):
            raise ValueError("terminal work item ID must bind to terminal work logical key")
        expected = window_terminal_member_semantic_sha256(self)
        if self.member_semantic_sha256 != expected:
            raise ValueError("member_semantic_sha256 does not match terminal member fields")
        return self


def window_terminal_member_semantic_projection(
    member: WindowTerminalMember,
) -> dict[str, object]:
    return {
        "member_projection_version": member.member_projection_version,
        "plan_key": member.plan_key,
        "expected_ordinal": member.expected_ordinal,
        "window_key": member.window_key,
        "window_semantic_sha256": member.window_semantic_sha256,
        "terminal_outcome": member.terminal_outcome.value,
        "terminal_work_item_id": member.terminal_work_item_id,
        "terminal_work_logical_key": member.terminal_work_logical_key,
        "terminal_evidence_ref": member.terminal_evidence_ref.model_dump(mode="json"),
        "terminal_policy_version": member.terminal_policy_version,
    }


def window_terminal_member_semantic_sha256(member: WindowTerminalMember) -> Sha256Digest:
    return semantic_sha256(window_terminal_member_semantic_projection(member))


def create_window_terminal_member(
    *,
    schema_ref: SchemaRef,
    plan_key: str,
    expected_ordinal: int,
    window_key: str,
    window_semantic_sha256: Sha256Digest,
    terminal_outcome: TerminalOutcome,
    terminal_work_item_id: OpaqueUuid,
    terminal_work_logical_key: str,
    terminal_evidence_ref: ArtifactEvidenceRef,
    terminal_policy_version: str,
) -> WindowTerminalMember:
    values = {
        "schema_ref": schema_ref,
        "plan_key": plan_key,
        "expected_ordinal": expected_ordinal,
        "window_key": window_key,
        "window_semantic_sha256": window_semantic_sha256,
        "terminal_outcome": terminal_outcome,
        "terminal_work_item_id": terminal_work_item_id,
        "terminal_work_logical_key": terminal_work_logical_key,
        "terminal_evidence_ref": terminal_evidence_ref,
        "terminal_policy_version": terminal_policy_version,
    }
    draft = WindowTerminalMember.model_construct(
        member_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = window_terminal_member_semantic_sha256(draft)
    return WindowTerminalMember(
        member_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


class WindowTerminalClosure(StrictModel):
    """Execution reconciliation over the already sealed expected set."""

    schema_version: Literal["1.0"] = STREAM_FINALIZATION_WIRE_VERSION
    schema_ref: SchemaRef
    plan_key: NonEmptyString
    plan_seal_semantic_sha256: Sha256Digest
    expected_member_count: NonNegativeInt
    members: tuple[WindowTerminalMember, ...]
    terminal_member_root: Sha256Digest
    terminal_closure_digest: Sha256Digest
    projection_version: NonEmptyString = TERMINAL_CLOSURE_PROJECTION_VERSION

    @model_validator(mode="after")
    def validate_closure(self) -> Self:
        if self.projection_version != TERMINAL_CLOSURE_PROJECTION_VERSION:
            raise ValueError("terminal closure uses the registered projection version")
        if len(self.members) != self.expected_member_count:
            raise ValueError("terminal closure must contain every expected member")
        ordinals = tuple(member.expected_ordinal for member in self.members)
        if ordinals != tuple(range(self.expected_member_count)):
            raise ValueError("terminal closure ordinals must be contiguous and ordered")
        if any(member.plan_key != self.plan_key for member in self.members):
            raise ValueError("terminal member plan_key does not match closure")
        expected_root = compute_terminal_member_root(
            plan_seal_semantic_sha256=self.plan_seal_semantic_sha256,
            members=self.members,
        )
        if self.terminal_member_root != expected_root:
            raise ValueError("terminal_member_root does not match ordered members")
        expected_digest = terminal_closure_semantic_sha256(self)
        if self.terminal_closure_digest != expected_digest:
            raise ValueError("terminal_closure_digest does not match closure projection")
        return self

    @property
    def complete(self) -> bool:
        return len(self.members) == self.expected_member_count


def compute_terminal_member_root(
    *,
    plan_seal_semantic_sha256: Sha256Digest,
    members: tuple[WindowTerminalMember, ...],
) -> Sha256Digest:
    return semantic_sha256(
        {
            "version": TERMINAL_MEMBER_ROOT_VERSION,
            "plan_seal_semantic_sha256": plan_seal_semantic_sha256,
            "ordered_member_semantic_sha256_values": [
                member.member_semantic_sha256 for member in members
            ],
        }
    )


def terminal_closure_semantic_projection(
    closure: WindowTerminalClosure,
) -> dict[str, object]:
    return {
        "projection_version": closure.projection_version,
        "plan_key": closure.plan_key,
        "plan_seal_semantic_sha256": closure.plan_seal_semantic_sha256,
        "expected_member_count": closure.expected_member_count,
        "terminal_member_root": closure.terminal_member_root,
    }


def terminal_closure_semantic_sha256(closure: WindowTerminalClosure) -> Sha256Digest:
    return semantic_sha256(terminal_closure_semantic_projection(closure))


def create_window_terminal_closure(
    *,
    schema_ref: SchemaRef,
    plan_seal: ExpectedWindowPlanSeal,
    expected_declarations: tuple[ExpectedWindowDeclaration, ...],
    members: tuple[WindowTerminalMember, ...],
) -> WindowTerminalClosure:
    expected = tuple(sorted(expected_declarations, key=lambda declaration: declaration.ordinal))
    if len(expected) != plan_seal.expected_member_count:
        raise ValueError("expected declarations must match the sealed member count")
    if tuple(declaration.ordinal for declaration in expected) != tuple(range(len(expected))):
        raise ValueError("expected declarations must be contiguous and ordered")
    if any(declaration.plan_key != plan_seal.plan_key for declaration in expected):
        raise ValueError("expected declaration plan_key does not match the seal")
    if (
        compute_ordered_expected_member_root(expected)
        != plan_seal.ordered_expected_member_root_sha256
    ):
        raise ValueError("expected declarations do not match the sealed member root")

    ordered = tuple(sorted(members, key=lambda member: member.expected_ordinal))
    if len(ordered) != len(expected):
        raise ValueError("terminal closure must contain every sealed expected member")
    for declaration, member in zip(expected, ordered, strict=True):
        if (
            member.plan_key != plan_seal.plan_key
            or member.expected_ordinal != declaration.ordinal
            or member.window_key != declaration.window_key
            or member.window_semantic_sha256 != declaration.window_semantic_sha256
        ):
            raise ValueError("terminal member does not match its sealed expected declaration")

    root = compute_terminal_member_root(
        plan_seal_semantic_sha256=plan_seal.seal_semantic_sha256,
        members=ordered,
    )
    projection = {
        "projection_version": TERMINAL_CLOSURE_PROJECTION_VERSION,
        "plan_key": plan_seal.plan_key,
        "plan_seal_semantic_sha256": plan_seal.seal_semantic_sha256,
        "expected_member_count": len(ordered),
        "terminal_member_root": root,
    }
    digest = semantic_sha256(projection)
    return WindowTerminalClosure(
        schema_ref=schema_ref,
        plan_key=plan_seal.plan_key,
        plan_seal_semantic_sha256=plan_seal.seal_semantic_sha256,
        expected_member_count=len(ordered),
        members=ordered,
        terminal_member_root=root,
        terminal_closure_digest=digest,
    )


class FinalizationSubjectMapping(StrictModel):
    """One immutable incremental-to-final subject link."""

    incremental_subject_type: StreamSubjectType
    incremental_subject_key: NonEmptyString
    incremental_subject_semantic_sha256: Sha256Digest
    final_subject_type: NonEmptyString
    final_subject_key: NonEmptyString
    final_subject_semantic_sha256: Sha256Digest


class RecordingFinalizationMap(StrictModel):
    """EOS link from incremental history to final recording identities."""

    schema_version: Literal["1.0"] = STREAM_FINALIZATION_WIRE_VERSION
    schema_ref: SchemaRef
    capture_scope_key: NonEmptyString
    capture_scope_digest: Sha256Digest
    final_source_subject_type: NonEmptyString
    final_source_subject_id: OpaqueUuid
    final_source_exact_sha256: Sha256Digest
    final_recording_identity: Sha256Digest
    final_duration_ns: int
    final_mapping_semantic_sha256: Sha256Digest
    final_alignment_semantic_sha256: Sha256Digest
    expected_plan_seal_semantic_sha256: Sha256Digest
    window_terminal_closure_semantic_sha256: Sha256Digest
    export_manifest_semantic_sha256: Sha256Digest
    ordered_subject_mappings: tuple[FinalizationSubjectMapping, ...]
    finalization_projection_version: NonEmptyString = FINALIZATION_PROJECTION_VERSION
    finalization_policy_version: NonEmptyString = FINALIZATION_POLICY_VERSION
    finalization_key: NonEmptyString
    finalization_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_map(self) -> Self:
        if self.finalization_projection_version != FINALIZATION_PROJECTION_VERSION:
            raise ValueError("finalization map uses the registered projection version")
        if self.finalization_policy_version != FINALIZATION_POLICY_VERSION:
            raise ValueError("finalization map uses the registered policy version")
        if self.final_duration_ns < 0:
            raise ValueError("final_duration_ns must be nonnegative")
        if self.capture_scope_key != f"pre-eos-capture-v1:{self.capture_scope_digest}":
            raise ValueError("capture_scope_key must bind to capture_scope_digest")
        keys = tuple(
            (mapping.incremental_subject_type, mapping.incremental_subject_key)
            for mapping in self.ordered_subject_mappings
        )
        if len(set(keys)) != len(keys):
            raise ValueError("incremental subjects may map at most once")
        if any(
            not mapping.incremental_subject_key.endswith(
                f":{mapping.incremental_subject_semantic_sha256}"
            )
            for mapping in self.ordered_subject_mappings
        ):
            raise ValueError("incremental subject keys must bind to their semantic digests")
        expected = recording_finalization_semantic_sha256(self)
        if self.finalization_semantic_sha256 != expected:
            raise ValueError("finalization_semantic_sha256 does not match map projection")
        if self.finalization_key != derive_finalization_key(expected):
            raise ValueError("finalization_key does not match map digest")
        return self


def recording_finalization_semantic_projection(
    finalization: RecordingFinalizationMap,
) -> dict[str, object]:
    return {
        "finalization_projection_version": finalization.finalization_projection_version,
        "finalization_policy_version": finalization.finalization_policy_version,
        "capture_scope_key": finalization.capture_scope_key,
        "capture_scope_digest": finalization.capture_scope_digest,
        "final_source_subject_type": finalization.final_source_subject_type,
        "final_source_subject_id": finalization.final_source_subject_id,
        "final_source_exact_sha256": finalization.final_source_exact_sha256,
        "final_recording_identity": finalization.final_recording_identity,
        "final_duration_ns": str(finalization.final_duration_ns),
        "final_mapping_semantic_sha256": finalization.final_mapping_semantic_sha256,
        "final_alignment_semantic_sha256": finalization.final_alignment_semantic_sha256,
        "expected_plan_seal_semantic_sha256": finalization.expected_plan_seal_semantic_sha256,
        "window_terminal_closure_semantic_sha256": (
            finalization.window_terminal_closure_semantic_sha256
        ),
        "export_manifest_semantic_sha256": finalization.export_manifest_semantic_sha256,
        "ordered_subject_mappings": [
            mapping.model_dump(mode="json") for mapping in finalization.ordered_subject_mappings
        ],
    }


def recording_finalization_semantic_sha256(
    finalization: RecordingFinalizationMap,
) -> Sha256Digest:
    return semantic_sha256(recording_finalization_semantic_projection(finalization))


def derive_finalization_key(finalization_semantic_sha256: Sha256Digest) -> str:
    return f"{FINALIZATION_KEY_NAMESPACE}:{finalization_semantic_sha256}"


def create_recording_finalization_map(
    *,
    schema_ref: SchemaRef,
    capture_scope_key: str,
    capture_scope_digest: Sha256Digest,
    final_source_subject_type: str,
    final_source_subject_id: OpaqueUuid,
    final_source_exact_sha256: Sha256Digest,
    final_recording_identity: Sha256Digest,
    final_duration_ns: int,
    final_mapping_semantic_sha256: Sha256Digest,
    final_alignment_semantic_sha256: Sha256Digest,
    expected_plan_seal_semantic_sha256: Sha256Digest,
    window_terminal_closure_semantic_sha256: Sha256Digest,
    export_manifest_semantic_sha256: Sha256Digest,
    ordered_subject_mappings: tuple[FinalizationSubjectMapping, ...],
) -> RecordingFinalizationMap:
    values = {
        "schema_ref": schema_ref,
        "capture_scope_key": capture_scope_key,
        "capture_scope_digest": capture_scope_digest,
        "final_source_subject_type": final_source_subject_type,
        "final_source_subject_id": final_source_subject_id,
        "final_source_exact_sha256": final_source_exact_sha256,
        "final_recording_identity": final_recording_identity,
        "final_duration_ns": final_duration_ns,
        "final_mapping_semantic_sha256": final_mapping_semantic_sha256,
        "final_alignment_semantic_sha256": final_alignment_semantic_sha256,
        "expected_plan_seal_semantic_sha256": expected_plan_seal_semantic_sha256,
        "window_terminal_closure_semantic_sha256": window_terminal_closure_semantic_sha256,
        "export_manifest_semantic_sha256": export_manifest_semantic_sha256,
        "ordered_subject_mappings": ordered_subject_mappings,
    }
    draft = RecordingFinalizationMap.model_construct(
        finalization_key="x",
        finalization_semantic_sha256="0" * 64,
        **cast(dict[str, Any], values),
    )
    digest = recording_finalization_semantic_sha256(draft)
    return RecordingFinalizationMap(
        finalization_key=derive_finalization_key(digest),
        finalization_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


__all__ = [
    "FINALIZATION_KEY_NAMESPACE",
    "FINALIZATION_POLICY_VERSION",
    "FINALIZATION_PROJECTION_VERSION",
    "RECORDING_FINALIZATION_SCHEMA_ID",
    "RECORDING_FINALIZATION_SCHEMA_VERSION",
    "STREAM_FINALIZATION_WIRE_VERSION",
    "TERMINAL_CLOSURE_PROJECTION_VERSION",
    "TERMINAL_MEMBER_PROJECTION_VERSION",
    "TERMINAL_MEMBER_ROOT_VERSION",
    "WINDOW_TERMINAL_CLOSURE_SCHEMA_ID",
    "WINDOW_TERMINAL_CLOSURE_SCHEMA_VERSION",
    "WINDOW_TERMINAL_MEMBER_SCHEMA_ID",
    "WINDOW_TERMINAL_MEMBER_SCHEMA_VERSION",
    "FinalizationSubjectMapping",
    "RecordingFinalization",
    "RecordingFinalizationMap",
    "TerminalClosureMember",
    "WindowTerminalClosure",
    "WindowTerminalMember",
    "compute_terminal_member_root",
    "create_recording_finalization_map",
    "create_window_terminal_closure",
    "create_window_terminal_member",
    "derive_finalization_key",
    "recording_finalization_semantic_projection",
    "recording_finalization_semantic_sha256",
    "terminal_closure_semantic_projection",
    "terminal_closure_semantic_sha256",
    "window_terminal_member_semantic_projection",
    "window_terminal_member_semantic_sha256",
]

RecordingFinalization = RecordingFinalizationMap
TerminalClosureMember = WindowTerminalMember
