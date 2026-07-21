from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.admission.context import AdmittedRecordingContextV2
from robata.application.canonical_offline import (
    CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION,
    CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION,
    CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION,
    CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION,
    CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION,
    CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE,
    CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION,
    CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
    CanonicalOfflineConfigurationError,
    CanonicalOutputAdmissionDecision,
    CanonicalRootWindow,
    _canonical_output_decision_projection_values,
    _stable_uuid,
    canonical_call_barrier_logical_node,
    canonical_call_part_logical_node,
    canonical_event_hypothesis_logical_node,
    canonical_input_plan_logical_node,
    canonical_lineage,
    canonical_output_decision_logical_node,
    canonical_output_decision_projection,
    canonical_package_set_logical_node,
    canonical_parsed_claim_logical_node,
    canonical_root_window_logical_node,
)
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.pipeline import SamplingPurpose
from robata.event_pipeline.identity_registry import (
    EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE,
    AdmissionEvidenceClass,
    OutputAdmissionProof,
    PlatformEnrichedOutputReference,
    ProductionOutputAdmissionPolicyRef,
)
from robata.inference.enrichment import OrchestratorEnrichedOutput
from robata.inference.input_plan import (
    CALL_BARRIER_LOGICAL_KEY_NAMESPACE,
    CALL_PART_LOGICAL_KEY_NAMESPACE,
    INPUT_PLAN_LOGICAL_KEY_NAMESPACE,
)
from robata.sampling.package_set import PackageSetBuilder, sampling_plan_digest
from tests.unit.test_event_identity_registry import (
    OUTPUT_ADMISSION_POLICY,
    _context,
    _enriched_output,
    _hypothesis,
    _hypothesis_fact,
    _output_proof,
)
from tests.unit.test_inference_enrichment import _fixture as _enrichment_fixture
from tests.unit.test_inference_enrichment import _parsed
from tests.unit.test_inference_input_plan import _fixture as _input_plan_fixture
from tests.unit.test_sampling_dense_package_set import (
    SECOND,
    _build_package_set,
    _plan,
    _window,
)
from tests.unit.test_sampling_materializer import _v2_context

NOW = "2026-07-20T12:00:00Z"


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _root_window(
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
    created_at: str = NOW,
) -> CanonicalRootWindow:
    context = _v2_context()
    return CanonicalRootWindow.from_context(
        context=context,
        requested_interval=NanosecondInterval(start_ns=0, end_ns=SECOND),
        purpose=purpose,
        window_policy_version="root-window-v1",
        created_at=created_at,
    )


def _decision_values(
    decision: CanonicalOutputAdmissionDecision,
    *,
    fusion_reduction_logical_key: str | None = None,
    production_output_admission: OutputAdmissionProof | None = None,
) -> dict[str, object]:
    proof = production_output_admission or decision.production_output_admission
    key = fusion_reduction_logical_key or decision.fusion_reduction_logical_key
    projection = _canonical_output_decision_projection_values(
        decision=decision.decision,
        evidence_class=decision.evidence_class,
        production_eligible=decision.production_eligible,
        recording_identity=decision.recording_identity,
        source_enrichments=decision.source_enrichments,
        fusion_reduction_logical_key=key,
        fusion_reduction_semantic_sha256=decision.fusion_reduction_semantic_sha256,
        policy_version=decision.policy_version,
        policy_sha256=decision.policy_sha256,
        admitted_claim_ordinals=decision.admitted_claim_ordinals,
        reason_code=decision.reason_code,
        production_output_admission=proof,
    )
    digest = semantic_sha256(projection)
    return {
        "schema_version": "2.0",
        "decision_id": _stable_uuid(CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE, digest),
        "decision": decision.decision,
        "evidence_class": decision.evidence_class,
        "production_eligible": decision.production_eligible,
        "semantic_sha256": digest,
        "recording_identity": decision.recording_identity,
        "source_enrichments": decision.source_enrichments,
        "fusion_reduction_logical_key": key,
        "fusion_reduction_semantic_sha256": decision.fusion_reduction_semantic_sha256,
        "policy_version": decision.policy_version,
        "policy_sha256": decision.policy_sha256,
        "admitted_claim_ordinals": decision.admitted_claim_ordinals,
        "reason_code": decision.reason_code,
        "production_output_admission": proof,
    }


def _local_admitted_decision() -> tuple[
    CanonicalOutputAdmissionDecision,
    AdmittedRecordingContextV2,
    OrchestratorEnrichedOutput,
]:
    context = _context()
    output = _enriched_output(context)
    source_refs = (PlatformEnrichedOutputReference.from_output(output),)
    fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="canonical-node-producer",
        ordinal=0,
    )
    proof = _output_proof(context, output, fact)
    fusion_digest = _digest("canonical-fusion-reduction")
    provisional = CanonicalOutputAdmissionDecision.model_construct(
        schema_version="2.0",
        decision_id=_uuid(1),
        decision="ADMITTED",
        evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        production_eligible=False,
        semantic_sha256="0" * 64,
        recording_identity=context.recording_identity,
        source_enrichments=source_refs,
        fusion_reduction_logical_key=f"fusion-reduction:{fusion_digest}",
        fusion_reduction_semantic_sha256=fusion_digest,
        policy_version=OUTPUT_ADMISSION_POLICY.version,
        policy_sha256=OUTPUT_ADMISSION_POLICY.semantic_sha256,
        admitted_claim_ordinals=(0,),
        reason_code="CLAIMS_ADMITTED",
        production_output_admission=proof,
    )
    decision = CanonicalOutputAdmissionDecision.model_validate(
        _decision_values(provisional), strict=True
    )
    return decision, context, output


def test_root_window_node_excludes_association_ids_and_clock() -> None:
    first = _root_window()
    relocated = first.model_copy(
        update={
            "mcap_id": _uuid(91_001),
            "camera_mapping_run_id": _uuid(91_002),
            "alignment_id": _uuid(91_003),
            "created_at": "2026-07-21T12:00:00Z",
        }
    )

    assert canonical_root_window_logical_node(relocated) == (
        canonical_root_window_logical_node(first)
    )


def test_root_window_semantic_change_changes_node_identity() -> None:
    first = _root_window()
    context = _v2_context()
    changed = CanonicalRootWindow.from_context(
        context=context,
        requested_interval=NanosecondInterval(start_ns=1, end_ns=SECOND),
        purpose=SamplingPurpose.ACTION_DENSE,
        window_policy_version="root-window-v1",
        created_at=NOW,
    )

    assert canonical_root_window_logical_node(changed) != canonical_root_window_logical_node(first)


def test_root_window_supports_qa_coarse_with_distinct_semantic_identity() -> None:
    context = _v2_context()
    plan = _plan(max_per_camera=20, max_total=120)
    action_dense = _root_window()
    qa_coarse = _root_window(purpose=SamplingPurpose.QA_COARSE)
    lineage = canonical_lineage(
        context=context,
        window=qa_coarse,
        sampling_plan=plan,
    )

    assert qa_coarse.purpose is SamplingPurpose.QA_COARSE
    assert qa_coarse.semantic_sha256 != action_dense.semantic_sha256
    assert canonical_root_window_logical_node(qa_coarse) != (
        canonical_root_window_logical_node(action_dense)
    )
    assert lineage.sampling_plan_sha256 == sampling_plan_digest(
        plan,
        purpose=SamplingPurpose.QA_COARSE,
    )
    assert lineage.sampling_plan_sha256 != sampling_plan_digest(plan)


def test_package_set_node_excludes_all_locators_row_ids_and_clock() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    builder = PackageSetBuilder("reduce-v1")
    first = _build_package_set(builder, _window(0, SECOND), plan)
    moved_window = SimpleNamespace(
        window_id="window-relocated",
        mcap_id="mcap-relocated",
        camera_mapping_run_id="mapping-relocated",
        requested_interval=NanosecondInterval(start_ns=0, end_ns=SECOND),
        interval=NanosecondInterval(start_ns=0, end_ns=SECOND),
    )
    relocated = _build_package_set(
        builder,
        moved_window,
        plan,
        alignment_id="alignment-relocated",
        package_id_prefix="relocated-package",
    ).model_copy(update={"created_at": "2026-07-21T12:00:00Z"})

    assert first.package_set_id == relocated.package_set_id
    assert canonical_package_set_logical_node(relocated) == (
        canonical_package_set_logical_node(first)
    )


def test_package_set_semantic_content_change_changes_node_identity() -> None:
    plan = _plan(max_per_camera=20, max_total=120)
    builder = PackageSetBuilder("reduce-v1")
    first = _build_package_set(builder, _window(0, SECOND), plan, content_seed="first")
    changed = _build_package_set(builder, _window(0, SECOND), plan, content_seed="second")

    assert canonical_package_set_logical_node(changed) != canonical_package_set_logical_node(first)


def test_parsed_claim_node_excludes_raw_and_parsed_row_facts() -> None:
    fixture = _enrichment_fixture()
    first = fixture.parsed
    relocated = _parsed(
        payload=fixture.payload,
        provider_schema=fixture.provider_schema,
        row_offset=50,
        inference_id=_uuid(92_001),
    )
    relocated = relocated.model_copy(
        update={
            "raw_response": relocated.raw_response.model_copy(
                update={"created_at": "2026-07-21T12:00:00Z"}
            ),
            "created_at": "2026-07-21T12:00:01Z",
        }
    )

    assert first.artifact_id != relocated.artifact_id
    assert first.raw_response.artifact_id != relocated.raw_response.artifact_id
    assert first.raw_response.provider_request_id != relocated.raw_response.provider_request_id
    assert canonical_parsed_claim_logical_node(relocated) == (
        canonical_parsed_claim_logical_node(first)
    )


def test_parsed_claim_semantic_change_changes_node_identity() -> None:
    fixture = _enrichment_fixture()
    changed_claim = fixture.payload.claims[0].model_copy(update={"label": "release"})
    changed_payload = fixture.payload.model_copy(
        update={"claims": (changed_claim, *fixture.payload.claims[1:])}
    )
    changed = _parsed(
        payload=changed_payload,
        provider_schema=fixture.provider_schema,
        row_offset=60,
        inference_id=_uuid(92_002),
    )

    assert canonical_parsed_claim_logical_node(changed) != (
        canonical_parsed_claim_logical_node(fixture.parsed)
    )


def test_input_plan_node_converges_across_row_locator_and_clock_changes() -> None:
    first = _input_plan_fixture(row_offset=0)[3]
    relocated = _input_plan_fixture(row_offset=50)[3]

    assert first.semantic_sha256 == relocated.semantic_sha256
    assert canonical_input_plan_logical_node(relocated) == canonical_input_plan_logical_node(first)


def test_input_plan_chain_nodes_publish_only_v2_identity_namespaces() -> None:
    plan = _input_plan_fixture()[3]
    input_node = canonical_input_plan_logical_node(plan)
    part_node = canonical_call_part_logical_node(plan, plan.call_plan.parts[0])
    barrier_node = canonical_call_barrier_logical_node(plan)

    assert (
        input_node.key_namespace,
        input_node.identity_policy_version,
    ) == (
        INPUT_PLAN_LOGICAL_KEY_NAMESPACE,
        CANONICAL_INPUT_PLAN_IDENTITY_POLICY_VERSION,
    )
    assert (
        part_node.key_namespace,
        part_node.identity_policy_version,
    ) == (
        CALL_PART_LOGICAL_KEY_NAMESPACE,
        CANONICAL_CALL_PART_IDENTITY_POLICY_VERSION,
    )
    assert (
        barrier_node.key_namespace,
        barrier_node.identity_policy_version,
    ) == (
        CALL_BARRIER_LOGICAL_KEY_NAMESPACE,
        CANONICAL_CALL_BARRIER_IDENTITY_POLICY_VERSION,
    )
    assert all(
        node.node_logical_key.startswith(f"{node.key_namespace}:")
        for node in (input_node, part_node, barrier_node)
    )
    assert {
        input_node.key_namespace,
        part_node.key_namespace,
        barrier_node.key_namespace,
    }.isdisjoint(
        {
            "inference-input-plan",
            "inference-input-call-part",
            "inference-input-barrier",
        }
    )


def test_call_part_node_requires_complete_plan_admission() -> None:
    plan = _input_plan_fixture()[3]
    part = plan.call_plan.parts[0]

    node = canonical_call_part_logical_node(plan, part)

    assert node.semantic_sha256 == part.part_semantic_sha256


def test_call_part_node_rejects_part_detached_from_complete_plan() -> None:
    plan = _input_plan_fixture()[3]
    part = plan.call_plan.parts[0]
    detached = part.model_copy(update={"end_item_ordinal_exclusive": 3})

    with pytest.raises(
        (CanonicalOfflineConfigurationError, ValidationError),
        match=r"call part|input plan",
    ):
        canonical_call_part_logical_node(plan, detached)


def test_output_and_event_nodes_publish_only_v2_identity_domains() -> None:
    decision, context, output = _local_admitted_decision()
    hypothesis = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="canonical-node-producer",
        ordinal=0,
        output_proof=decision.production_output_admission,
    )

    output_node = canonical_output_decision_logical_node(decision)
    event_node = canonical_event_hypothesis_logical_node(hypothesis)

    assert (
        output_node.key_namespace,
        output_node.identity_policy_version,
    ) == (
        CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE,
        CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION,
    )
    assert (
        event_node.key_namespace,
        event_node.identity_policy_version,
    ) == (
        EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE,
        CANONICAL_EVENT_HYPOTHESIS_IDENTITY_POLICY_VERSION,
    )
    assert (
        canonical_output_decision_projection(decision)["semantic_projection_version"]
        == CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION
    )
    assert output_node.node_logical_key.startswith(
        f"{CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE}:"
    )
    assert event_node.node_logical_key.startswith(f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:")


def test_output_decision_rejects_v1_status_and_uuid_domain() -> None:
    decision, _, _ = _local_admitted_decision()

    old_status = _decision_values(decision)
    old_status["decision"] = "PRODUCTION_ADMITTED"
    with pytest.raises(ValidationError, match="ADMITTED"):
        CanonicalOutputAdmissionDecision.model_validate(old_status, strict=True)

    old_identity = _decision_values(decision)
    old_identity["decision_id"] = _stable_uuid(
        "canonical-output-admission",
        decision.semantic_sha256,
    )
    with pytest.raises(ValidationError, match="decision ID"):
        CanonicalOutputAdmissionDecision.model_validate(old_identity, strict=True)


def test_output_decision_rejects_fusion_key_digest_mismatch() -> None:
    decision, _, _ = _local_admitted_decision()
    wrong_key = f"fusion-reduction:{_digest('different-fusion-reduction')}"

    with pytest.raises(
        (CanonicalOfflineConfigurationError, ValidationError),
        match=r"fusion reduction|fusion_reduction",
    ):
        forged = CanonicalOutputAdmissionDecision.model_validate(
            _decision_values(decision, fusion_reduction_logical_key=wrong_key),
            strict=True,
        )
        canonical_output_decision_logical_node(forged)


def test_output_decision_rejects_proof_policy_mismatch() -> None:
    decision, context, output = _local_admitted_decision()
    other_policy = ProductionOutputAdmissionPolicyRef(
        version="other-output-admission-v1",
        semantic_sha256=_digest("other-output-admission-v1"),
    )
    fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="canonical-node-producer",
        ordinal=0,
    )
    mismatched_proof = _output_proof(context, output, fact, policy=other_policy)

    with pytest.raises(
        (CanonicalOfflineConfigurationError, ValidationError),
        match="policy",
    ):
        forged = CanonicalOutputAdmissionDecision.model_validate(
            _decision_values(
                decision,
                production_output_admission=mismatched_proof,
            ),
            strict=True,
        )
        canonical_output_decision_logical_node(forged)


def test_parsed_claim_node_rejects_stale_semantic_digest() -> None:
    parsed = _enrichment_fixture().parsed
    forged = parsed.model_copy(update={"semantic_sha256": _digest("stale")})

    with pytest.raises(ValidationError, match="semantic_sha256"):
        canonical_parsed_claim_logical_node(forged)
