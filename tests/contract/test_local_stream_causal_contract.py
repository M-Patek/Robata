from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.local_stream_causal import (
    LocalStreamStageEvidenceReference,
    LocalStreamWindowInferencePlan,
    LocalStreamWindowSemanticEvidenceV2,
    create_local_stream_window_inference_plan,
    create_local_stream_window_semantic_evidence_v2,
)
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import StreamArtifactRef, StreamStage
from scripts.export_persisted_wire_schema import export_schema

_DIGEST = "a" * 64
_SCHEMA_REF = SchemaRef(
    schema_id="https://schemas.robata.dev/local-test",
    version="1.0.0",
    artifact_id="00000000-0000-0000-0000-000000000001",
    sha256=_DIGEST,
)


def _artifact(seed: str | bytes) -> StreamArtifactRef:
    raw = seed.encode("utf-8") if isinstance(seed, str) else seed
    return StreamArtifactRef(
        artifact_id="00000000-0000-0000-0000-000000000002",
        exact_sha256=exact_bytes_sha256(raw),
        byte_count=len(raw),
        media_type="application/json",
        schema_ref=_SCHEMA_REF,
    )


def _plan() -> LocalStreamWindowInferencePlan:
    evidence = tuple(
        LocalStreamStageEvidenceReference(
            stage=stage,
            work_logical_key=f"work:{stage.value.lower()}",
            terminal_evidence_ref=_artifact(stage.value),
            evidence_semantic_sha256=_DIGEST,
        )
        for stage in (
            StreamStage.WINDOW,
            StreamStage.QA_COARSE,
            StreamStage.QA_DENSE,
            StreamStage.EVENT_PROPOSAL,
        )
    )
    window_digest = _DIGEST
    return create_local_stream_window_inference_plan(
        schema_ref=_SCHEMA_REF,
        plan_key="expected-plan:v1",
        expected_ordinal=0,
        window_key=f"incremental-window-v1:{window_digest}",
        window_semantic_sha256=window_digest,
        effective_interval=NanosecondInterval(start_ns=0, end_ns=1_000_000_000),
        input_plan_semantic_sha256=_DIGEST,
        six_camera_slot_closure_semantic_sha256=_DIGEST,
        ordered_upstream_stage_evidence=evidence,
    )


def test_causal_plan_is_deterministic_and_stage_ordered() -> None:
    first = _plan()
    second = _plan()

    assert first == second
    assert first.inference_plan_key.endswith(first.plan_semantic_sha256)
    assert first.input_plan_id


def test_causal_plan_rejects_stage_reordering() -> None:
    plan = _plan().model_dump(mode="python")
    plan["ordered_upstream_stage_evidence"] = tuple(
        reversed(plan["ordered_upstream_stage_evidence"])
    )

    with pytest.raises(ValidationError, match="ordered and complete"):
        LocalStreamWindowInferencePlan.model_validate(plan, strict=True)


def test_semantic_evidence_binds_exact_plan_artifact_and_rejects_tamper() -> None:
    plan = _plan()
    plan_payload = canonical_json_bytes(plan)
    evidence = create_local_stream_window_semantic_evidence_v2(
        schema_ref=_SCHEMA_REF,
        plan=plan,
        plan_ref=_artifact(plan_payload),
    )

    assert evidence.plan_semantic_sha256 == plan.plan_semantic_sha256
    assert evidence.window_inference_plan_ref.exact_sha256 == exact_bytes_sha256(plan_payload)

    with pytest.raises(ValueError, match="exact artifact"):
        create_local_stream_window_semantic_evidence_v2(
            schema_ref=_SCHEMA_REF,
            plan=plan,
            plan_ref=_artifact("wrong-plan"),
        )

    tampered = deepcopy(evidence.model_dump(mode="python"))
    tampered["proposal_label"] = "tampered"
    with pytest.raises(ValidationError, match="semantic_sha256"):
        LocalStreamWindowSemanticEvidenceV2.model_validate(tampered, strict=True)


def test_causal_candidate_schema_exports_are_deterministic(tmp_path: Path) -> None:
    cases = (
        (
            "local-stream-window-inference-plan",
            "https://schemas.robata.dev/v1/local-stream-window-inference-plan.schema.json",
            tmp_path / "plan.schema.json",
        ),
        (
            "local-stream-window-semantic-evidence-v2",
            "https://schemas.robata.dev/v2/local-stream-window-semantic-evidence.schema.json",
            tmp_path / "evidence.schema.json",
        ),
    )
    for name, document_id, path in cases:
        export_schema(name, path)
        first = path.read_bytes()
        export_schema(name, path)
        assert path.read_bytes() == first
        document = __import__("json").loads(first)
        assert document["$id"] == document_id
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert semantic_sha256(document["properties"]) != _DIGEST
