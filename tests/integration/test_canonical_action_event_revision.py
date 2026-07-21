from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.models import CanonicalOfflineRunStatus
from robata.application.canonical.result_validation import CanonicalOfflineRunResult
from robata.application.canonical_offline import (
    CanonicalActionEventRevisionError,
    prepare_initial_action_event_publications,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.revisions import RevisionEligibility
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    EventIdentityPolicyRef,
    EventIdentityRegistryService,
    ExactFingerprintEventIdentityResolver,
    InMemoryEventIdentityRegistryRepository,
    PreparedEventIdentityBatch,
)
from robata.inference.enrichment import OrchestratorEnrichedOutput
from tests.integration.test_canonical_offline import (
    _claim_bytes,
    _digest,
    _Harness,
    _harness,
    _run,
    _SequenceEventIdAllocator,
)


@pytest.fixture(scope="module")
def canonical_action_event_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[_Harness, CanonicalOfflineRunResult, tuple[OrchestratorEnrichedOutput, ...]]:
    root = Path(tmp_path_factory.mktemp("canonical-action-event-revision"))
    harness = _harness(_claim_bytes, logical_registry_root=root)
    result = _run(harness)
    assert result.status is CanonicalOfflineRunStatus.SUCCEEDED
    outputs = tuple(
        item.enriched_output for item in result.part_results if item.enriched_output is not None
    )
    assert outputs
    return harness, result, outputs


def _prepare_identities(
    *,
    repository: InMemoryEventIdentityRegistryRepository,
    harness: _Harness,
    result: CanonicalOfflineRunResult,
    outputs: tuple[OrchestratorEnrichedOutput, ...],
    decided_at: str,
) -> PreparedEventIdentityBatch:
    service = EventIdentityRegistryService(
        repository=None,
        resolver=ExactFingerprintEventIdentityResolver(
            EventIdentityPolicyRef(
                version="exact-fingerprint-v1",
                semantic_sha256=_digest("exact-fingerprint-v1"),
            )
        ),
        allocator=_SequenceEventIdAllocator(),
        output_admission_policy=harness.execution_policy.output_admission_policy,
    )
    return service.prepare_batch(
        snapshot=repository.snapshot(harness.context.recording_identity),
        admitted_context=harness.context,
        hypotheses=result.hypotheses,
        enriched_outputs=outputs,
        decided_at=decided_at,
    )


def test_prepares_local_genesis_revision_without_writes(
    canonical_action_event_case: tuple[
        _Harness,
        CanonicalOfflineRunResult,
        tuple[OrchestratorEnrichedOutput, ...],
    ],
) -> None:
    harness, result, outputs = canonical_action_event_case
    repository = InMemoryEventIdentityRegistryRepository()
    initial = repository.snapshot(harness.context.recording_identity)
    completed_at = result.processing_run.completed_at
    assert completed_at is not None
    prepared = _prepare_identities(
        repository=repository,
        harness=harness,
        result=result,
        outputs=outputs,
        decided_at=completed_at,
    )

    batch = prepare_initial_action_event_publications(
        context=harness.context,
        result=result,
        prepared_identities=prepared,
        execution_policy=harness.execution_policy,
    )

    assert batch.outcome == "PREPARED"
    assert (batch.expected_generation, batch.expected_fence) == (0, 1)
    assert len(batch.publications) == 1
    publication = batch.publications[0]
    assert publication.subject.node_type == "ACTION_EVENT"
    assert publication.assignment.event_id == publication.payload.event_id
    assert publication.payload.recording_identity == harness.context.recording_identity
    assert publication.payload.evidence_class is AdmissionEvidenceClass.LOCAL_CONFORMANCE
    assert publication.payload.production_eligible is False
    assert publication.payload.event_status == "NEEDS_REVIEW"
    assert tuple(item.camera_id for item in publication.payload.camera_sources) == CAMERA_IDS
    assert sum(item.citation_status == "CITED" for item in publication.payload.camera_sources) == 1
    assert sum(len(item.cited_frames) for item in publication.payload.camera_sources) == 1
    assert publication.revision.supersedes_revision_id is None
    assert publication.revision.eligibility_at_publication is RevisionEligibility.ELIGIBLE
    assert publication.selection.selection_sequence == 1
    assert publication.selection.previous_selection_decision_id is None
    assert publication.current.selected_revision_id == publication.revision.revision_id
    assert (
        publication.current_revision.revision_logical_key
        == publication.revision.revision_logical_key
    )
    assert repository.snapshot(harness.context.recording_identity) == initial
    assert repository.list_outbox(harness.context.recording_identity) == ()


def test_exact_identity_replay_rebuilds_identical_publications(
    canonical_action_event_case: tuple[
        _Harness,
        CanonicalOfflineRunResult,
        tuple[OrchestratorEnrichedOutput, ...],
    ],
) -> None:
    harness, result, outputs = canonical_action_event_case
    repository = InMemoryEventIdentityRegistryRepository()
    completed_at = result.processing_run.completed_at
    assert completed_at is not None
    first_prepared = _prepare_identities(
        repository=repository,
        harness=harness,
        result=result,
        outputs=outputs,
        decided_at=completed_at,
    )
    first_batch = prepare_initial_action_event_publications(
        context=harness.context,
        result=result,
        prepared_identities=first_prepared,
        execution_policy=harness.execution_policy,
    )
    mutation = first_prepared.mutation
    assert mutation is not None
    committed = repository.commit(mutation)

    replay_prepared = _prepare_identities(
        repository=repository,
        harness=harness,
        result=result,
        outputs=outputs,
        decided_at="2026-07-20T12:00:00Z",
    )
    replay_batch = prepare_initial_action_event_publications(
        context=harness.context,
        result=result,
        prepared_identities=replay_prepared,
        execution_policy=harness.execution_policy,
    )

    assert replay_prepared.mutation is None
    assert replay_prepared.assignments == first_prepared.assignments
    assert (replay_batch.expected_generation, replay_batch.expected_fence) == (1, 2)
    assert replay_batch.publications == first_batch.publications
    assert repository.snapshot(harness.context.recording_identity) == committed
    assert len(repository.list_outbox(harness.context.recording_identity)) == 1


def test_assignment_hypothesis_mismatch_fails_without_writes(
    canonical_action_event_case: tuple[
        _Harness,
        CanonicalOfflineRunResult,
        tuple[OrchestratorEnrichedOutput, ...],
    ],
) -> None:
    harness, result, outputs = canonical_action_event_case
    repository = InMemoryEventIdentityRegistryRepository()
    initial = repository.snapshot(harness.context.recording_identity)
    completed_at = result.processing_run.completed_at
    assert completed_at is not None
    prepared = _prepare_identities(
        repository=repository,
        harness=harness,
        result=result,
        outputs=outputs,
        decided_at=completed_at,
    )
    forged_assignment = prepared.assignments[0].model_copy(
        update={"event_hypothesis_semantic_sha256": _digest("forged-hypothesis")}
    )
    forged_prepared = prepared.model_copy(update={"assignments": (forged_assignment,)})

    with pytest.raises(
        CanonicalActionEventRevisionError,
        match="does not bind the exact event hypothesis",
    ):
        prepare_initial_action_event_publications(
            context=harness.context,
            result=result,
            prepared_identities=forged_prepared,
            execution_policy=harness.execution_policy,
        )

    assert repository.snapshot(harness.context.recording_identity) == initial
    assert repository.list_outbox(harness.context.recording_identity) == ()
