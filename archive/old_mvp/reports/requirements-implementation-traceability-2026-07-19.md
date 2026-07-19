# REQUIREMENTS.md implementation traceability (July 19, 2026)

| Requirement | Local implementation | Acceptance evidence | Production gate |
|---|---|---|---|
| Clip-level QA `{start_sec,end_sec,issue,confidence}` | `robata.qa.ClipMark` | `tests/unit/test_requirements_components.py` | Real detector/model quality not claimed |
| 21 issue taxonomy and local/whole mapping | `QAIssue`, `ISSUE_DISPOSITION`, `QA_ISSUE_GROUPS` | taxonomy count/policy tests | Cross-video review remains human/process gate |
| pass/warning/fail; only fail deletes | `QAClassifier`, `QAAssessment` | QA policy tests and `verify_requirements.py` | Source deletion adapter intentionally absent |
| Pass + warning annotation; fail exclusion | `AnnotationPipeline`, `AnnotationPrincipal` | annotation pipeline tests | Real annotation principal/model approval required |
| Warning marks propagated downstream | `AnnotationSegmentDraft.qa_clip_marks` | warning overlap test | SLAM/skeleton consumers not in this scope |
| Feed once shared frames | `SharedFrameCache.feed_once` | decode-at-most-once test | R2/object-store adapter pending |
| Zero-GPU structured-label search | `ClipSearchIndex`, `VerbNormalizer`, query parser | search playback test | Embedding/vector stage pending |
| T+1/T+3 SLA accounting | `SLAPlanner` | deadline test | Operational scheduler/alerts pending |
| 500 h/day throughput target | `CapacityPlanner`, `ThroughputLedger` | assumptions/report test | Governed corpus and measured provider throughput required |
| No real provider traffic | fixed run metadata | acceptance JSON (`provider_requests=0`) | Real provider integration is a separate approved change |

## Current status

The local preparation chain is complete and executable without model SDKs, network access, Redis,
PostgreSQL, Supabase, R2, or RunPod dependencies. It is **not production eligible**: local fake
model and synthetic throughput evidence remain `NOT_MEASURED`, and `production_eligible=false`.
