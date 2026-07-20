from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from threading import Barrier, Lock
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.admission import (
    AdmissionContextResolver,
    AlignmentAdmissionOutcome,
    PrimaryAdmissionEvaluation,
    PrimaryAdmissionPolicy,
    SourceAdmissionOutcome,
)
from robata.contracts.common import NanosecondInterval
from robata.contracts.schema_registry import SchemaRegistry
from robata.event_pipeline.identity_registry import (
    ADMISSION_PROOF_SEMANTIC_PROJECTION_VERSION,
    EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE,
    EVENT_HYPOTHESIS_SEMANTIC_PROJECTION_VERSION,
    OUTPUT_ADMISSION_SEMANTIC_PROJECTION_VERSION,
    AdmissionEvidenceClass,
    AdmissionProof,
    CrossRecordingEventIdentityError,
    EventIdentityAssignmentDisposition,
    EventIdentityInputError,
    EventIdentityPolicyRef,
    EventIdentityRegistryService,
    ExactFingerprintEventIdentityResolver,
    InMemoryEventIdentityRegistryRepository,
    OutputAdmissionProof,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    ProductionAdmittedHypothesisFact,
    ProductionOutputAdmissionPolicyRef,
    ProductionQualificationUnavailableError,
    admission_proof_projection,
    event_hypothesis_semantic_projection,
    output_admission_projection,
    platform_enriched_output_logical_projection,
)
from robata.inference.enrichment import SelectedAttemptOutput
from tests.contract.test_admission_evidence_v2_contract import (
    _alignment_manifest,
    _ready_manifest,
    _validation_report,
)
from tests.unit.test_inference_enrichment import _enrich, _fixture, _parsed, _selection

NOW = "2026-07-19T16:00:00Z"
OUTPUT_ADMISSION_POLICY = ProductionOutputAdmissionPolicyRef(
    version="fusion-output-admission-v1",
    semantic_sha256=sha256(b"fusion-output-admission-v1").hexdigest(),
)


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _context(label: str = "recording"):
    registry = SchemaRegistry()
    report = _validation_report(
        registry,
        recording_identity=_digest(label),
    )
    ready = _ready_manifest(registry, report)
    alignment = _alignment_manifest(registry, ready)
    policy = PrimaryAdmissionPolicy.create(
        version="primary-v2",
        admissible_alignment_outcomes=(AlignmentAdmissionOutcome.VALID,),
    )
    evaluation = PrimaryAdmissionEvaluation(
        recording_identity=report.recording_identity,
        ready_manifest_id=ready.ready_manifest_id,
        ready_manifest_semantic_sha256=ready.ready_manifest_semantic_sha256,
        source_outcome=SourceAdmissionOutcome.READY,
        alignment_outcome=AlignmentAdmissionOutcome.VALID,
        alignment_id=alignment.alignment_id,
        alignment_semantic_sha256=alignment.alignment_semantic_sha256,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admissible=True,
        reason_code="ADMISSIBLE",
    )
    return AdmissionContextResolver().resolve_v2(
        evaluation=evaluation,
        policy=policy,
        validation_report=report,
        ready_manifest=ready,
        alignment_manifest=alignment,
        registry=registry,
    )


def _enriched_output(context):
    fixture = _fixture()
    authority = fixture.authority.model_copy(
        update={
            "recording_identity": context.recording_identity,
            "mcap_id": context.ready_manifest.mcap_id,
            "camera_mapping_run_id": context.ready_manifest.camera_mapping_run_id,
            "alignment_id": context.alignment_manifest.alignment_id,
        }
    )
    return _enrich(replace(fixture, authority=authority))


def _equivalent_enriched_outputs(context):
    fixture = _fixture()
    authority = fixture.authority.model_copy(
        update={
            "recording_identity": context.recording_identity,
            "mcap_id": context.ready_manifest.mcap_id,
            "camera_mapping_run_id": context.ready_manifest.camera_mapping_run_id,
            "alignment_id": context.alignment_manifest.alignment_id,
        }
    )
    first = _enrich(fixture, authority=authority)
    second_parsed = _parsed(
        payload=fixture.payload,
        provider_schema=fixture.provider_schema,
        row_offset=50,
        inference_id=_uuid(602),
    )
    second_selection = _selection(
        inference_id=second_parsed.raw_response.inference_id,
        logical_invocation_id=fixture.selection.logical_invocation_id,
        policy_version=fixture.selection.policy_version,
        row_offset=50,
    )
    second_selected = SelectedAttemptOutput.create(second_parsed, second_selection)
    second_authority = authority.model_copy(update={"inference_id": second_selected.inference_id})
    second = _enrich(
        fixture,
        parsed=second_parsed,
        selected=second_selected,
        authority=second_authority,
    )
    return first, second


def _hypothesis_fact(
    *,
    start_ns: int,
    end_ns: int,
    fingerprint: str,
    ordinal: int,
) -> ProductionAdmittedHypothesisFact:
    return ProductionAdmittedHypothesisFact(
        fusion_output_ordinal=ordinal,
        effective_interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        semantic_fingerprint_sha256=_digest(fingerprint),
        fusion_logical_key=f"fusion:{_digest(f'fusion-{ordinal}')}",
    )


def _output_proof(
    context,
    output,
    *facts: ProductionAdmittedHypothesisFact,
    policy: ProductionOutputAdmissionPolicyRef = OUTPUT_ADMISSION_POLICY,
) -> OutputAdmissionProof:
    return OutputAdmissionProof.create(
        recording_identity=context.recording_identity,
        source_enrichments=(PlatformEnrichedOutputReference.from_output(output),),
        admitted_hypothesis_facts=facts,
        policy=policy,
    )


def _hypothesis(
    context,
    output,
    *,
    start_ns: int,
    end_ns: int,
    fingerprint: str,
    ordinal: int,
    output_proof: OutputAdmissionProof | None = None,
) -> PlatformEnrichedEventHypothesis:
    references = (PlatformEnrichedOutputReference.from_output(output),)
    fact = _hypothesis_fact(
        start_ns=start_ns,
        end_ns=end_ns,
        fingerprint=fingerprint,
        ordinal=ordinal,
    )
    return PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=ordinal,
        source_enrichments=references,
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=output_proof or _output_proof(context, output, fact),
    )


def _evidence_lineage(
    context,
    output,
    evidence_class: AdmissionEvidenceClass,
) -> tuple[AdmissionProof, OutputAdmissionProof, PlatformEnrichedEventHypothesis]:
    fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="evidence-tier-event",
        ordinal=0,
    )
    references = (PlatformEnrichedOutputReference.from_output(output),)
    admission = AdmissionProof.from_context(
        context,
        evidence_class=evidence_class,
    )
    output_proof = OutputAdmissionProof.create(
        recording_identity=context.recording_identity,
        source_enrichments=references,
        admitted_hypothesis_facts=(fact,),
        policy=OUTPUT_ADMISSION_POLICY,
        evidence_class=evidence_class,
    )
    hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=fact.fusion_output_ordinal,
        source_enrichments=references,
        production_admission=admission,
        production_output_admission=output_proof,
    )
    return admission, output_proof, hypothesis


class _SequenceAllocator:
    version = "test-sequence-v1"

    def __init__(self, *, start: int = 10_000, fixed: str | None = None) -> None:
        self._next = start
        self._fixed = fixed
        self._lock = Lock()

    def allocate(self, **_kwargs: object) -> str:
        with self._lock:
            if self._fixed is not None:
                return self._fixed
            value = self._next
            self._next += 1
            return _uuid(value)


class _SynchronizedInitialSnapshotRepository(InMemoryEventIdentityRegistryRepository):
    def __init__(self) -> None:
        super().__init__()
        self._initial_reads = Barrier(2)

    def snapshot(self, recording_identity: str):
        snapshot = super().snapshot(recording_identity)
        if snapshot.generation == 0:
            self._initial_reads.wait(timeout=5)
        return snapshot


def _service(repository, allocator) -> EventIdentityRegistryService:
    policy = EventIdentityPolicyRef(
        version="exact-fingerprint-v1",
        semantic_sha256=_digest("exact-fingerprint-v1"),
    )
    return EventIdentityRegistryService(
        repository=repository,
        resolver=ExactFingerprintEventIdentityResolver(policy),
        allocator=allocator,
        output_admission_policy=OUTPUT_ADMISSION_POLICY,
        max_cas_retries=4,
    )


def test_nonproduction_evidence_class_is_hash_bearing() -> None:
    context = _context()
    output = _enriched_output(context)
    lineages = {
        evidence_class: _evidence_lineage(context, output, evidence_class)
        for evidence_class in (
            AdmissionEvidenceClass.LOCAL_CONFORMANCE,
            AdmissionEvidenceClass.GOVERNED_BENCHMARK,
        )
    }

    for evidence_class, (admission, output_proof, hypothesis) in lineages.items():
        assert admission.decision == output_proof.decision == "ADMITTED"
        assert admission.evidence_class is output_proof.evidence_class is evidence_class
        assert admission.production_eligible is False
        assert output_proof.production_eligible is False
        assert (
            admission_proof_projection(admission)["semantic_projection_version"]
            == ADMISSION_PROOF_SEMANTIC_PROJECTION_VERSION
        )
        assert (
            output_admission_projection(output_proof)["semantic_projection_version"]
            == OUTPUT_ADMISSION_SEMANTIC_PROJECTION_VERSION
        )
        assert (
            event_hypothesis_semantic_projection(hypothesis)["semantic_projection_version"]
            == EVENT_HYPOTHESIS_SEMANTIC_PROJECTION_VERSION
        )
        assert hypothesis.event_hypothesis_logical_key.startswith(
            f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:"
        )

    assert len({item[1].semantic_sha256 for item in lineages.values()}) == 2
    assert len({item[2].semantic_sha256 for item in lineages.values()}) == 2
    assert len({item[2].event_hypothesis_logical_key for item in lineages.values()}) == 2


def test_evidence_metadata_rejects_inconsistent_production_eligibility() -> None:
    context = _context()
    output = _enriched_output(context)
    admission, output_proof, _ = _evidence_lineage(
        context,
        output,
        AdmissionEvidenceClass.LOCAL_CONFORMANCE,
    )

    admission_values = admission.model_dump(mode="python")
    admission_values["production_eligible"] = True
    with pytest.raises(ValidationError, match="production_eligible"):
        AdmissionProof.model_validate(admission_values, strict=True)

    output_values = output_proof.model_dump(mode="python")
    output_values["production_eligible"] = True
    with pytest.raises(ValidationError, match="production_eligible"):
        OutputAdmissionProof.model_validate(output_values, strict=True)


def test_v1_admission_status_and_event_namespace_fail_closed() -> None:
    context = _context()
    output = _enriched_output(context)
    admission, output_proof, hypothesis = _evidence_lineage(
        context,
        output,
        AdmissionEvidenceClass.LOCAL_CONFORMANCE,
    )

    for model, proof in (
        (AdmissionProof, admission),
        (OutputAdmissionProof, output_proof),
    ):
        values = proof.model_dump(mode="python")
        values["decision"] = "PRODUCTION_ADMITTED"
        with pytest.raises(ValidationError, match="ADMITTED"):
            model.model_validate(values, strict=True)

    hypothesis_values = hypothesis.model_dump(mode="python")
    hypothesis_values["event_hypothesis_logical_key"] = (
        f"event-hypothesis:{hypothesis.semantic_sha256}"
    )
    with pytest.raises(ValidationError, match="unexpected namespace"):
        PlatformEnrichedEventHypothesis.model_validate(hypothesis_values, strict=True)

    output_values = output_proof.model_dump(mode="python")
    output_values["schema_version"] = "1.0"
    with pytest.raises(ValidationError, match=r"2\.0"):
        OutputAdmissionProof.model_validate(output_values, strict=True)

    hypothesis_values = hypothesis.model_dump(mode="python")
    hypothesis_values["schema_version"] = "1.0"
    with pytest.raises(ValidationError, match=r"2\.0"):
        PlatformEnrichedEventHypothesis.model_validate(hypothesis_values, strict=True)


def test_production_qualification_cannot_be_self_minted() -> None:
    context = _context()
    output = _enriched_output(context)
    fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="production-qualified",
        ordinal=0,
    )

    with pytest.raises(
        ProductionQualificationUnavailableError,
        match="governed qualification gateway",
    ):
        AdmissionProof.from_context(
            context,
            evidence_class=AdmissionEvidenceClass.PRODUCTION_QUALIFIED,
        )
    with pytest.raises(
        ProductionQualificationUnavailableError,
        match="governed qualification gateway",
    ):
        OutputAdmissionProof.create(
            recording_identity=context.recording_identity,
            source_enrichments=(PlatformEnrichedOutputReference.from_output(output),),
            admitted_hypothesis_facts=(fact,),
            policy=OUTPUT_ADMISSION_POLICY,
            evidence_class=AdmissionEvidenceClass.PRODUCTION_QUALIFIED,
        )

    admission = AdmissionProof.from_context(context)
    admission_values = admission.model_dump(mode="python")
    admission_values.update(
        evidence_class=AdmissionEvidenceClass.PRODUCTION_QUALIFIED,
        production_eligible=True,
    )
    with pytest.raises(ValidationError, match="governed qualification gateway"):
        AdmissionProof.model_validate(admission_values, strict=True)

    output_proof = _output_proof(context, output, fact)
    output_values = output_proof.model_dump(mode="python")
    output_values.update(
        evidence_class=AdmissionEvidenceClass.PRODUCTION_QUALIFIED,
        production_eligible=True,
    )
    with pytest.raises(ValidationError, match="governed qualification gateway"):
        OutputAdmissionProof.model_validate(output_values, strict=True)

    policy = EventIdentityPolicyRef(
        version="exact-fingerprint-v1",
        semantic_sha256=_digest("exact-fingerprint-v1"),
    )
    with pytest.raises(
        ProductionQualificationUnavailableError,
        match="governed qualification gateway",
    ):
        EventIdentityRegistryService(
            repository=InMemoryEventIdentityRegistryRepository(),
            resolver=ExactFingerprintEventIdentityResolver(policy),
            allocator=_SequenceAllocator(),
            output_admission_policy=OUTPUT_ADMISSION_POLICY,
            admission_evidence_class=AdmissionEvidenceClass.PRODUCTION_QUALIFIED,
        )


def test_canonical_batch_reuses_exact_fingerprint_and_replay_is_idempotent() -> None:
    context = _context()
    output = _enriched_output(context)
    early_fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="same-event",
        ordinal=0,
    )
    late_fact = _hypothesis_fact(
        start_ns=30,
        end_ns=40,
        fingerprint="same-event",
        ordinal=1,
    )
    output_proof = _output_proof(context, output, late_fact, early_fact)
    early = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="same-event",
        ordinal=0,
        output_proof=output_proof,
    )
    late = _hypothesis(
        context,
        output,
        start_ns=30,
        end_ns=40,
        fingerprint="same-event",
        ordinal=1,
        output_proof=output_proof,
    )
    repository = InMemoryEventIdentityRegistryRepository()
    service = _service(repository, _SequenceAllocator())

    first = service.assign_batch(
        admitted_context=context,
        hypotheses=(late, early),
        enriched_outputs=(output,),
        decided_at=NOW,
    )

    assert tuple(item.event_hypothesis_logical_key for item in first.assignments) == (
        early.event_hypothesis_logical_key,
        late.event_hypothesis_logical_key,
    )
    assert tuple(item.disposition for item in first.assignments) == (
        EventIdentityAssignmentDisposition.CREATED,
        EventIdentityAssignmentDisposition.REUSED,
    )
    assert len({item.event_id for item in first.assignments}) == 1
    assert first.initial_generation == 0
    assert first.final_generation == 1
    assert len(first.new_identities) == 1
    assert len(first.outbox) == 2

    replay = service.assign_batch(
        admitted_context=context,
        hypotheses=(early, late),
        enriched_outputs=(output,),
        decided_at="2026-07-20T00:00:00Z",
    )
    assert replay.initial_generation == replay.final_generation == 1
    assert replay.assignments == first.assignments
    assert replay.new_identities == ()
    assert replay.outbox == ()
    assert len(replay.replayed_assignment_logical_keys) == 2
    assert len(repository.list_outbox(context.recording_identity)) == 2


def test_logical_identity_converges_across_exact_attempt_and_artifact_locators() -> None:
    context = _context()
    first_output, second_output = _equivalent_enriched_outputs(context)
    first_reference = PlatformEnrichedOutputReference.from_output(first_output)
    second_reference = PlatformEnrichedOutputReference.from_output(second_output)
    fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="same-logical-event",
        ordinal=0,
    )

    assert first_output.enrichment_logical_key == second_output.enrichment_logical_key
    assert first_output.semantic_sha256 != second_output.semantic_sha256
    assert first_output.selected_attempt.inference_id != second_output.selected_attempt.inference_id
    assert first_output.selected_attempt.selection_id != second_output.selected_attempt.selection_id
    assert first_reference != second_reference
    assert platform_enriched_output_logical_projection(first_reference) == (
        platform_enriched_output_logical_projection(second_reference)
    )

    first_proof = OutputAdmissionProof.create(
        recording_identity=context.recording_identity,
        source_enrichments=(first_reference,),
        admitted_hypothesis_facts=(fact,),
        policy=OUTPUT_ADMISSION_POLICY,
    )
    second_proof = OutputAdmissionProof.create(
        recording_identity=context.recording_identity,
        source_enrichments=(second_reference,),
        admitted_hypothesis_facts=(fact,),
        policy=OUTPUT_ADMISSION_POLICY,
    )
    first_hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=fact.fusion_output_ordinal,
        source_enrichments=(first_reference,),
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=first_proof,
    )
    second_hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=fact.fusion_output_ordinal,
        source_enrichments=(second_reference,),
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=second_proof,
    )

    assert first_proof.semantic_sha256 == second_proof.semantic_sha256
    assert first_hypothesis.semantic_sha256 == second_hypothesis.semantic_sha256
    assert first_hypothesis.event_hypothesis_logical_key == (
        second_hypothesis.event_hypothesis_logical_key
    )

    service = _service(InMemoryEventIdentityRegistryRepository(), _SequenceAllocator())
    first_result = service.assign_batch(
        admitted_context=context,
        hypotheses=(first_hypothesis,),
        enriched_outputs=(first_output,),
        decided_at=NOW,
    )
    replay = service.assign_batch(
        admitted_context=context,
        hypotheses=(second_hypothesis,),
        enriched_outputs=(second_output,),
        decided_at=NOW,
    )
    assert replay.assignments == first_result.assignments
    assert replay.replayed_assignment_logical_keys == (
        first_result.assignments[0].assignment_logical_key,
    )

    with pytest.raises(EventIdentityInputError, match="forged enrichment"):
        service.assign_batch(
            admitted_context=context,
            hypotheses=(second_hypothesis,),
            enriched_outputs=(first_output,),
            decided_at=NOW,
        )


def test_logical_identity_separates_distinct_enrichment_logical_keys() -> None:
    context = _context()
    output = _enriched_output(context)
    first_reference = PlatformEnrichedOutputReference.from_output(output)
    second_reference = first_reference.model_copy(
        update={"enrichment_logical_key": f"orchestrator-enrichment:{_digest('different')}"}
    )
    fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="same-event-facts",
        ordinal=0,
    )
    first_proof = OutputAdmissionProof.create(
        recording_identity=context.recording_identity,
        source_enrichments=(first_reference,),
        admitted_hypothesis_facts=(fact,),
        policy=OUTPUT_ADMISSION_POLICY,
    )
    second_proof = OutputAdmissionProof.create(
        recording_identity=context.recording_identity,
        source_enrichments=(second_reference,),
        admitted_hypothesis_facts=(fact,),
        policy=OUTPUT_ADMISSION_POLICY,
    )
    first_hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=fact.fusion_output_ordinal,
        source_enrichments=(first_reference,),
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=first_proof,
    )
    second_hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=fact.fusion_output_ordinal,
        source_enrichments=(second_reference,),
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=second_proof,
    )

    assert first_proof.semantic_sha256 != second_proof.semantic_sha256
    assert first_hypothesis.semantic_sha256 != second_hypothesis.semantic_sha256
    assert first_hypothesis.event_hypothesis_logical_key != (
        second_hypothesis.event_hypothesis_logical_key
    )


def test_registry_rejects_forged_enrichment_lineage() -> None:
    context = _context()
    output = _enriched_output(context)
    reference = PlatformEnrichedOutputReference.from_output(output).model_copy(
        update={"enriched_output_semantic_sha256": _digest("forged-output")}
    )
    hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=NanosecondInterval(start_ns=10, end_ns=20),
        semantic_fingerprint_sha256=_digest("event"),
        fusion_logical_key=f"fusion:{_digest('fusion')}",
        fusion_output_ordinal=0,
        source_enrichments=(reference,),
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=OutputAdmissionProof.create(
            recording_identity=context.recording_identity,
            source_enrichments=(reference,),
            admitted_hypothesis_facts=(
                ProductionAdmittedHypothesisFact(
                    fusion_output_ordinal=0,
                    effective_interval=NanosecondInterval(start_ns=10, end_ns=20),
                    semantic_fingerprint_sha256=_digest("event"),
                    fusion_logical_key=f"fusion:{_digest('fusion')}",
                ),
            ),
            policy=OUTPUT_ADMISSION_POLICY,
        ),
    )

    with pytest.raises(EventIdentityInputError, match="forged enrichment"):
        _service(
            InMemoryEventIdentityRegistryRepository(),
            _SequenceAllocator(),
        ).assign_batch(
            admitted_context=context,
            hypotheses=(hypothesis,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )


def test_registry_rejects_an_unconfigured_output_admission_policy() -> None:
    context = _context()
    output = _enriched_output(context)
    references = (PlatformEnrichedOutputReference.from_output(output),)
    other_policy = ProductionOutputAdmissionPolicyRef(
        version="other-output-admission-v1",
        semantic_sha256=_digest("other-output-admission-v1"),
    )
    hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=context.recording_identity,
        effective_interval=NanosecondInterval(start_ns=10, end_ns=20),
        semantic_fingerprint_sha256=_digest("event"),
        fusion_logical_key=f"fusion:{_digest('fusion-other-policy')}",
        fusion_output_ordinal=0,
        source_enrichments=references,
        production_admission=AdmissionProof.from_context(context),
        production_output_admission=OutputAdmissionProof.create(
            recording_identity=context.recording_identity,
            source_enrichments=references,
            admitted_hypothesis_facts=(
                ProductionAdmittedHypothesisFact(
                    fusion_output_ordinal=0,
                    effective_interval=NanosecondInterval(start_ns=10, end_ns=20),
                    semantic_fingerprint_sha256=_digest("event"),
                    fusion_logical_key=f"fusion:{_digest('fusion-other-policy')}",
                ),
            ),
            policy=other_policy,
        ),
    )

    with pytest.raises(EventIdentityInputError, match="configured output admission policy"):
        _service(
            InMemoryEventIdentityRegistryRepository(),
            _SequenceAllocator(),
        ).assign_batch(
            admitted_context=context,
            hypotheses=(hypothesis,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("effective_interval", NanosecondInterval(start_ns=11, end_ns=21)),
        ("semantic_fingerprint_sha256", _digest("forged-fingerprint")),
    ),
)
def test_registry_rejects_output_proof_reused_for_changed_hypothesis_facts(
    field: str,
    replacement: object,
) -> None:
    context = _context()
    output = _enriched_output(context)
    admitted = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="admitted-event",
        ordinal=0,
    )
    forged = admitted.model_copy(update={field: replacement})

    with pytest.raises(EventIdentityInputError, match="event hypothesis failed validation"):
        _service(
            InMemoryEventIdentityRegistryRepository(),
            _SequenceAllocator(),
        ).assign_batch(
            admitted_context=context,
            hypotheses=(forged,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )


def test_registry_rejects_an_extra_hypothesis_not_covered_by_the_proof() -> None:
    context = _context()
    output = _enriched_output(context)
    admitted = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="admitted-event",
        ordinal=0,
    )
    extra = admitted.model_copy(
        update={
            "effective_interval": NanosecondInterval(start_ns=30, end_ns=40),
            "semantic_fingerprint_sha256": _digest("unadmitted-event"),
            "fusion_logical_key": f"fusion:{_digest('fusion-1')}",
            "fusion_output_ordinal": 1,
        }
    )

    with pytest.raises(EventIdentityInputError, match="event hypothesis failed validation"):
        _service(
            InMemoryEventIdentityRegistryRepository(),
            _SequenceAllocator(),
        ).assign_batch(
            admitted_context=context,
            hypotheses=(admitted, extra),
            enriched_outputs=(output,),
            decided_at=NOW,
        )


def test_registry_rejects_proof_facts_not_present_in_the_hypothesis_batch() -> None:
    context = _context()
    output = _enriched_output(context)
    admitted_fact = _hypothesis_fact(
        start_ns=10,
        end_ns=20,
        fingerprint="admitted-event",
        ordinal=0,
    )
    absent_fact = _hypothesis_fact(
        start_ns=30,
        end_ns=40,
        fingerprint="absent-event",
        ordinal=1,
    )
    proof = _output_proof(context, output, admitted_fact, absent_fact)
    admitted = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="admitted-event",
        ordinal=0,
        output_proof=proof,
    )

    with pytest.raises(EventIdentityInputError, match="exactly cover the hypothesis batch"):
        _service(
            InMemoryEventIdentityRegistryRepository(),
            _SequenceAllocator(),
        ).assign_batch(
            admitted_context=context,
            hypotheses=(admitted,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )


def test_concurrent_same_recording_batches_retry_and_converge() -> None:
    context = _context()
    output = _enriched_output(context)
    first_hypothesis = _hypothesis(
        context,
        output,
        start_ns=10,
        end_ns=20,
        fingerprint="concurrent-event",
        ordinal=0,
    )
    second_hypothesis = _hypothesis(
        context,
        output,
        start_ns=30,
        end_ns=40,
        fingerprint="concurrent-event",
        ordinal=1,
    )
    repository = _SynchronizedInitialSnapshotRepository()
    service = _service(repository, _SequenceAllocator())

    def assign(hypothesis: PlatformEnrichedEventHypothesis):
        return service.assign_batch(
            admitted_context=context,
            hypotheses=(hypothesis,),
            enriched_outputs=(output,),
            decided_at=NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(assign, (first_hypothesis, second_hypothesis)))

    event_ids = {result.assignments[0].event_id for result in results}
    snapshot = repository.snapshot(context.recording_identity)
    assert len(event_ids) == 1
    assert len(snapshot.identities) == 1
    assert len(snapshot.assignments) == 2
    assert snapshot.generation == 2


def test_allocator_cannot_reuse_an_event_id_across_recordings() -> None:
    fixed_event_id = _uuid(50_000)
    repository = InMemoryEventIdentityRegistryRepository()
    allocator = _SequenceAllocator(fixed=fixed_event_id)
    service = _service(repository, allocator)
    first_context = _context("recording-a")
    first_output = _enriched_output(first_context)
    first = _hypothesis(
        first_context,
        first_output,
        start_ns=10,
        end_ns=20,
        fingerprint="event-a",
        ordinal=0,
    )
    service.assign_batch(
        admitted_context=first_context,
        hypotheses=(first,),
        enriched_outputs=(first_output,),
        decided_at=NOW,
    )

    second_context = _context("recording-b")
    second_output = _enriched_output(second_context)
    second = _hypothesis(
        second_context,
        second_output,
        start_ns=10,
        end_ns=20,
        fingerprint="event-b",
        ordinal=0,
    )
    with pytest.raises(
        CrossRecordingEventIdentityError,
        match="another recording",
    ):
        service.assign_batch(
            admitted_context=second_context,
            hypotheses=(second,),
            enriched_outputs=(second_output,),
            decided_at=NOW,
        )
