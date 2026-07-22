from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from robata.contracts.pipeline import CandidateEvent, TemporalVisualPackage
from robata.contracts.qa import QAClassifier
from robata.event_pipeline.candidate import CandidateEventManager
from robata.event_pipeline.evidence import ActionEvidenceExtractor
from robata.event_pipeline.proposer import (
    EventProposer,
    EventProposerConfig,
)
from robata.event_pipeline.proposer import (
    MCAPRecording as ProposalMCAPRecording,
)


def _unimplemented_calls() -> tuple[tuple[str, Callable[[], object]], ...]:
    candidate = cast(CandidateEvent, object())
    proposer = EventProposer(
        EventProposerConfig(
            version="event-proposer-v1",
            min_proposal_duration_ns=1,
            max_proposals_per_recording=1,
            overlap_threshold=0.5,
        )
    )
    candidate_manager = CandidateEventManager()
    classifier = QAClassifier()
    evidence = ActionEvidenceExtractor()

    return (
        (
            "EventProposer.propose",
            lambda: proposer.propose(
                cast(ProposalMCAPRecording, object()),
                cast(tuple[TemporalVisualPackage, ...], ()),
                (),
            ),
        ),
        (
            "CandidateEventManager.merge_candidates",
            lambda: candidate_manager.merge_candidates((candidate,)),
        ),
        (
            "CandidateEventManager.split_candidate",
            lambda: candidate_manager.split_candidate(candidate, (1,)),
        ),
        (
            "CandidateEventManager.validate_candidate",
            lambda: candidate_manager.validate_candidate(candidate),
        ),
        (
            "extract_evidence",
            lambda: evidence.extract_evidence(
                candidate,
                cast(TemporalVisualPackage, object()),
            ),
        ),
        (
            "QAClassifier.assess",
            lambda: classifier.assess("recording", 1.0),
        ),
    )


@pytest.mark.parametrize(("entry_point", "invoke"), _unimplemented_calls())
def test_unimplemented_pipeline_entry_points_fail_closed(
    entry_point: str,
    invoke: Callable[[], object],
) -> None:
    with pytest.raises(NotImplementedError, match=entry_point):
        invoke()
