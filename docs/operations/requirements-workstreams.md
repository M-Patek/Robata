# Requirements workstreams runbook

This runbook documents the completed provider-neutral path for `REQUIREMENTS.md`. It is an
implementation boundary, not a production model/provider approval.

## QA

`robata.qa.QAClassifier.assess(recording_id, duration_sec, clip_marks)` accepts one or more
`ClipMark` values:

```python
ClipMark(start_sec=12.5, end_sec=14.0, issue=QAIssue.BLACK_SCREEN, confidence=0.93)
```

The classifier retains every mark (including multiple overlapping marks), computes a union-based
`effective_duration_sec`, and derives:

- `pass`: no marks; annotation eligible;
- `warning`: local marks; recording retained and marks are passed downstream;
- `fail`: whole-recording device/compliance/completeness defect; only this state requests source
delete.

Black Screen and Too Dark / Overexposed escalate to `fail` only when marks cover the complete
recording. The source recording is never modified.

`QAIssue` and `ISSUE_DISPOSITION` expose the complete 21-item vocabulary and local-vs-whole-recording
policy. `Lack of diversity` and `Performed other existing Tasks` are represented for cross-video
review and do not force per-video deletion.

## Feed once frame cache

`SharedFrameCache.feed_once(video_id, source_uri, decoder, frame_rate=2.0)` writes content-addressed
frame blobs and an immutable `FrameFeedManifest`. Concurrent/repeated calls for the same video
return the existing manifest without invoking `decoder` again. Annotation receives the manifest's
`FrameRef` values; it does not pull or decode the source a second time.

The local implementation is filesystem-only. A future object-store adapter may persist the same
manifest and URI contract without changing callers.

## Annotation principal

`AnnotationPipeline.run(assessments, frame_manifests=...)` invokes an injected
`AnnotationPrincipal` only for `pass` and `warning` assessments. A `fail` video is explicitly listed
in `skipped_fail_video_ids`. Every draft uses structured labels:

```text
verb / noun / attributes / location / hand
```

Warning `ClipMark` values intersecting a draft are attached even if an upstream principal omitted
them. `DeterministicAnnotationPrincipal` is the local fake implementation used for tests and
acceptance evidence; replacing it with a model adapter does not change the pipeline contract.

## Zero-GPU search MVP

`ClipSearchIndex` indexes `AnnotationSegmentDraft` or `ClipIndexEntry` records. It normalizes verb
synonyms into deterministic action families (for example `wipe`, `scrub`, and `wash` ? `clean`),
parses natural-language/faceted queries, filters by noun/attributes/location/hand, and returns
`SearchHit` records with exact `start_sec`, `end_sec`, and a direct playback target.

No GPU, network, Supabase, pgvector, or embedding dependency is used by this MVP. Text/visual
embeddings can be added later as a ranking layer while retaining facet filters.

## SLA and capacity accounting

`SLAPlanner` computes upload-time deadlines (QA T+1 and annotation T+3). `ThroughputLedger` and
`CapacityPlanner` report observations in recording-hours/day and GPU-hours/day. The documented
assumption is 2?H100 = 48 GPU-hours/day, with approximately 2h QA + 30h 7B pre-annotation = 32h/day.
A report is not certifying unless a governed corpus is supplied and `production_eligible` is
explicitly enabled; local fake-model smoke runs remain `NOT_MEASURED`.

## Acceptance command

```powershell
.\.venv\Scripts\python.exe scripts\verify_requirements.py
```

The command is offline and deterministic. It verifies pass/warning/fail behavior, fail exclusion,
feed-once idempotency, clip search playback targets, SLA/capacity accounting, and emits the fixed
flags `provider_requests=0`, `execution_mode=LOCAL_DEVELOPMENT_FAKE_MODEL`, and
`production_eligible=false`.
