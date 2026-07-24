"""Window-local causal evidence for the provider-neutral stream executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal, Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import model_validator

from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    NonEmptyString,
    NonNegativeInt,
    StreamStage,
)

LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID: Final = (
    "https://schemas.robata.dev/local-stream-window-inference-plan"
)
LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION: Final = "1.0.0"
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID: Final = (
    "https://schemas.robata.dev/local-stream-window-semantic-evidence"
)
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION: Final = "2.0.0"
LOCAL_STREAM_WINDOW_INFERENCE_PLAN_PROJECTION_VERSION: Final = (
    "local-stream-window-inference-plan-semantic-v1"
)
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_PROJECTION_VERSION: Final = (
    "local-stream-window-semantic-evidence-v2"
)
LOCAL_STREAM_WINDOW_CAUSAL_POLICY_VERSION: Final = "local-conformance-window-causal-executor-v1"
LOCAL_STREAM_WINDOW_INFERENCE_PLAN_KEY_NAMESPACE: Final = "local-stream-window-inference-plan-v1"
LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_KEY_NAMESPACE: Final = (
    "local-stream-window-semantic-evidence-v2"
)
_PLAN_NAMESPACE = uuid5(NAMESPACE_URL, "robata:local-stream-window-inference-plan-v1")
_CAUSAL_UPSTREAM_STAGES: Final = (
    StreamStage.WINDOW,
    StreamStage.QA_COARSE,
    StreamStage.QA_DENSE,
    StreamStage.EVENT_PROPOSAL,
)


class LocalStreamStageEvidenceReference(StrictModel):
    """One exact upstream terminal receipt selected by a causal plan."""

    stage: StreamStage
    work_logical_key: NonEmptyString
    terminal_evidence_ref: ArtifactEvidenceRef
    evidence_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_stage(self) -> Self:
        if self.stage not in _CAUSAL_UPSTREAM_STAGES:
            raise ValueError("causal stage evidence must be one of the four upstream stages")
        return self


def _plan_projection(plan: LocalStreamWindowInferencePlan) -> dict[str, object]:
    return {
        "projection_version": plan.projection_version,
        "causal_policy_version": plan.causal_policy_version,
        "plan_key": plan.plan_key,
        "expected_ordinal": plan.expected_ordinal,
        "window_key": plan.window_key,
        "window_semantic_sha256": plan.window_semantic_sha256,
        "effective_interval": plan.effective_interval.model_dump(mode="json"),
        "input_plan_semantic_sha256": plan.input_plan_semantic_sha256,
        "six_camera_slot_closure_semantic_sha256": plan.six_camera_slot_closure_semantic_sha256,
        "ordered_upstream_stage_evidence": [
            item.model_dump(mode="json") for item in plan.ordered_upstream_stage_evidence
        ],
    }


class LocalStreamWindowInferencePlan(StrictModel):
    """A deterministic local input plan assembled before a window result."""

    schema_version: Literal["1.0"] = "1.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    production_eligible: Literal[False] = False
    projection_version: Literal["local-stream-window-inference-plan-semantic-v1"] = (
        LOCAL_STREAM_WINDOW_INFERENCE_PLAN_PROJECTION_VERSION
    )
    causal_policy_version: Literal["local-conformance-window-causal-executor-v1"] = (
        LOCAL_STREAM_WINDOW_CAUSAL_POLICY_VERSION
    )
    plan_key: NonEmptyString
    expected_ordinal: NonNegativeInt
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    effective_interval: NanosecondInterval
    input_plan_semantic_sha256: Sha256Digest
    six_camera_slot_closure_semantic_sha256: Sha256Digest
    ordered_upstream_stage_evidence: tuple[LocalStreamStageEvidenceReference, ...]
    inference_plan_key: NonEmptyString
    input_plan_id: str
    plan_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.projection_version != LOCAL_STREAM_WINDOW_INFERENCE_PLAN_PROJECTION_VERSION:
            raise ValueError("window inference plan uses the registered projection version")
        if self.causal_policy_version != LOCAL_STREAM_WINDOW_CAUSAL_POLICY_VERSION:
            raise ValueError("window inference plan uses the registered causal policy")
        if (
            tuple(item.stage for item in self.ordered_upstream_stage_evidence)
            != _CAUSAL_UPSTREAM_STAGES
        ):
            raise ValueError("causal upstream stage evidence must be ordered and complete")
        if self.window_key != f"incremental-window-v1:{self.window_semantic_sha256}":
            raise ValueError("causal plan window key must bind to its semantic digest")
        expected = local_stream_window_inference_plan_semantic_sha256(self)
        if self.plan_semantic_sha256 != expected:
            raise ValueError("plan_semantic_sha256 does not match the plan projection")
        if self.inference_plan_key != derive_local_stream_window_inference_plan_key(expected):
            raise ValueError("inference_plan_key does not match plan_semantic_sha256")
        if self.input_plan_id != derive_local_stream_window_inference_plan_id(expected):
            raise ValueError("input_plan_id does not match plan_semantic_sha256")
        return self


def local_stream_window_inference_plan_semantic_projection(
    plan: LocalStreamWindowInferencePlan,
) -> dict[str, object]:
    return _plan_projection(plan)


def local_stream_window_inference_plan_semantic_sha256(
    plan: LocalStreamWindowInferencePlan,
) -> Sha256Digest:
    return semantic_sha256(_plan_projection(plan))


def derive_local_stream_window_inference_plan_key(digest: Sha256Digest) -> str:
    return f"{LOCAL_STREAM_WINDOW_INFERENCE_PLAN_KEY_NAMESPACE}:{digest}"


def derive_local_stream_window_inference_plan_id(digest: Sha256Digest) -> str:
    return str(uuid5(_PLAN_NAMESPACE, derive_local_stream_window_inference_plan_key(digest)))


def create_local_stream_window_inference_plan(
    *,
    schema_ref: SchemaRef,
    plan_key: str,
    expected_ordinal: int,
    window_key: str,
    window_semantic_sha256: Sha256Digest,
    effective_interval: NanosecondInterval,
    input_plan_semantic_sha256: Sha256Digest,
    six_camera_slot_closure_semantic_sha256: Sha256Digest,
    ordered_upstream_stage_evidence: Sequence[LocalStreamStageEvidenceReference],
) -> LocalStreamWindowInferencePlan:
    values: dict[str, Any] = {
        "schema_ref": schema_ref,
        "plan_key": plan_key,
        "expected_ordinal": expected_ordinal,
        "window_key": window_key,
        "window_semantic_sha256": window_semantic_sha256,
        "effective_interval": effective_interval,
        "input_plan_semantic_sha256": input_plan_semantic_sha256,
        "six_camera_slot_closure_semantic_sha256": six_camera_slot_closure_semantic_sha256,
        "ordered_upstream_stage_evidence": tuple(ordered_upstream_stage_evidence),
        "inference_plan_key": "x",
        "input_plan_id": "00000000-0000-0000-0000-000000000000",
        "plan_semantic_sha256": "0" * 64,
    }
    draft = LocalStreamWindowInferencePlan.model_construct(**values)
    digest = local_stream_window_inference_plan_semantic_sha256(draft)
    values["inference_plan_key"] = derive_local_stream_window_inference_plan_key(digest)
    values["input_plan_id"] = derive_local_stream_window_inference_plan_id(digest)
    values["plan_semantic_sha256"] = digest
    return LocalStreamWindowInferencePlan(**values)


LocalStreamWindowSemanticStatus = Literal["PROPOSED", "NO_EVENTS", "ABSTAINED"]


class LocalStreamWindowSemanticEvidenceV2(StrictModel):
    """Window-local semantic availability derived before EOS."""

    schema_version: Literal["2.0"] = "2.0"
    schema_ref: SchemaRef
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    production_eligible: Literal[False] = False
    projection_version: Literal["local-stream-window-semantic-evidence-v2"] = (
        LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_PROJECTION_VERSION
    )
    causal_policy_version: Literal["local-conformance-window-causal-executor-v1"] = (
        LOCAL_STREAM_WINDOW_CAUSAL_POLICY_VERSION
    )
    plan_key: NonEmptyString
    plan_semantic_sha256: Sha256Digest
    window_inference_plan_ref: ArtifactEvidenceRef
    expected_ordinal: NonNegativeInt
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    effective_interval: NanosecondInterval
    input_plan_semantic_sha256: Sha256Digest
    six_camera_slot_closure_semantic_sha256: Sha256Digest
    semantic_status: LocalStreamWindowSemanticStatus
    proposal_label: NonEmptyString | None = None
    proposal_interval: NanosecondInterval | None = None
    proposal_semantic_sha256: Sha256Digest | None = None
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.projection_version != LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_PROJECTION_VERSION:
            raise ValueError("window semantic evidence uses the registered projection version")
        if self.causal_policy_version != LOCAL_STREAM_WINDOW_CAUSAL_POLICY_VERSION:
            raise ValueError("window semantic evidence uses the registered causal policy")
        if self.window_key != f"incremental-window-v1:{self.window_semantic_sha256}":
            raise ValueError("window semantic evidence window key is inconsistent")
        if self.semantic_status == "PROPOSED":
            if (
                self.proposal_label is None
                or self.proposal_interval is None
                or self.proposal_semantic_sha256 is None
            ):
                raise ValueError("PROPOSED evidence requires a proposal fragment")
            if (
                self.proposal_interval.start_ns < self.effective_interval.start_ns
                or self.proposal_interval.end_ns > self.effective_interval.end_ns
            ):
                raise ValueError("window proposal interval must be contained by effective interval")
        elif any(
            value is not None
            for value in (
                self.proposal_label,
                self.proposal_interval,
                self.proposal_semantic_sha256,
            )
        ):
            raise ValueError("non-PROPOSED evidence cannot contain a proposal fragment")
        if self.semantic_sha256 != local_stream_window_semantic_evidence_v2_semantic_sha256(self):
            raise ValueError("semantic_sha256 does not match window evidence projection")
        return self


def local_stream_window_semantic_evidence_v2_semantic_projection(
    evidence: LocalStreamWindowSemanticEvidenceV2,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        evidence.model_dump(mode="json", exclude={"schema_ref", "semantic_sha256"}),
    )


def local_stream_window_semantic_evidence_v2_semantic_sha256(
    evidence: LocalStreamWindowSemanticEvidenceV2,
) -> Sha256Digest:
    return semantic_sha256(local_stream_window_semantic_evidence_v2_semantic_projection(evidence))


def derive_local_stream_window_semantic_evidence_v2_key(digest: Sha256Digest) -> str:
    return f"{LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_KEY_NAMESPACE}:{digest}"


def create_local_stream_window_semantic_evidence_v2(
    *,
    schema_ref: SchemaRef,
    plan: LocalStreamWindowInferencePlan,
    plan_ref: ArtifactEvidenceRef,
    semantic_status: LocalStreamWindowSemanticStatus = "PROPOSED",
    proposal_label: str | None = "fixture-action",
    proposal_interval: NanosecondInterval | None = None,
    proposal_semantic_sha256: Sha256Digest | None = None,
) -> LocalStreamWindowSemanticEvidenceV2:
    plan_payload = canonical_json_bytes(plan)
    if (
        plan_ref.schema_ref != plan.schema_ref
        or plan_ref.exact_sha256 != exact_bytes_sha256(plan_payload)
        or plan_ref.byte_count != len(plan_payload)
        or plan_ref.media_type != "application/json"
    ):
        raise ValueError("window inference plan reference does not bind its exact artifact")
    interval = plan.effective_interval if proposal_interval is None else proposal_interval
    proposal_digest = (
        None
        if semantic_status != "PROPOSED"
        else proposal_semantic_sha256
        or semantic_sha256(
            {
                "projection_version": "local-stream-window-proposal-v1",
                "label": proposal_label,
                "interval": interval.model_dump(mode="json"),
                "window_semantic_sha256": plan.window_semantic_sha256,
            }
        )
    )
    values: dict[str, Any] = {
        "schema_ref": schema_ref,
        "plan_key": plan.plan_key,
        "plan_semantic_sha256": plan.plan_semantic_sha256,
        "window_inference_plan_ref": plan_ref,
        "expected_ordinal": plan.expected_ordinal,
        "window_key": plan.window_key,
        "window_semantic_sha256": plan.window_semantic_sha256,
        "effective_interval": plan.effective_interval,
        "input_plan_semantic_sha256": plan.input_plan_semantic_sha256,
        "six_camera_slot_closure_semantic_sha256": plan.six_camera_slot_closure_semantic_sha256,
        "semantic_status": semantic_status,
        "proposal_label": proposal_label if semantic_status == "PROPOSED" else None,
        "proposal_interval": interval if semantic_status == "PROPOSED" else None,
        "proposal_semantic_sha256": proposal_digest,
    }
    draft = LocalStreamWindowSemanticEvidenceV2.model_construct(semantic_sha256="0" * 64, **values)
    digest = local_stream_window_semantic_evidence_v2_semantic_sha256(draft)
    return LocalStreamWindowSemanticEvidenceV2(semantic_sha256=digest, **values)


__all__ = [
    "LOCAL_STREAM_WINDOW_CAUSAL_POLICY_VERSION",
    "LOCAL_STREAM_WINDOW_INFERENCE_PLAN_PROJECTION_VERSION",
    "LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID",
    "LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION",
    "LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_PROJECTION_VERSION",
    "LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID",
    "LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION",
    "LocalStreamStageEvidenceReference",
    "LocalStreamWindowInferencePlan",
    "LocalStreamWindowSemanticEvidenceV2",
    "create_local_stream_window_inference_plan",
    "create_local_stream_window_semantic_evidence_v2",
    "derive_local_stream_window_inference_plan_id",
    "derive_local_stream_window_inference_plan_key",
    "derive_local_stream_window_semantic_evidence_v2_key",
    "local_stream_window_inference_plan_semantic_projection",
    "local_stream_window_inference_plan_semantic_sha256",
    "local_stream_window_semantic_evidence_v2_semantic_projection",
    "local_stream_window_semantic_evidence_v2_semantic_sha256",
]
