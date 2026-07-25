# Event Semantics

## Scope and path anchors
- Event implementation: `src/robata/event_pipeline/**`
- Main entry files: `candidate.py`, `evidence.py`, `proposer.py`
- Unit coverage: `test_event_pipeline_core.py`, `test_event_projection_guards.py`, `test_supplemental_temporal_package.py`

## How to dispatch
`event-semantics / P<n> - <candidate, evidence, hypothesis, proposal, or boundary task>`

## Construction phases
1. **Candidates and evidence** - turn QA/inference facts into deterministic event inputs.
2. **Hypotheses and proposals** - compose typed event and action semantics.
3. **Temporal refinement** - improve onset/offset windows while retaining upstream evidence.
4. **Semantic stability** - exercise ordering, replay, and projection guards.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py`
- Broader: `python -m pytest tests/integration/test_canonical_offline.py tests/integration/test_canonical_action_event_revision.py`

## Read alongside
Read `sampling-qa` and `inference-evidence` for input evidence. Read `identity-delivery` if a change alters fields used to identify or revise an event, and `canonical-integration` for stage ordering.
