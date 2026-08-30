# Production WeMM temporal resolution (local review path)

**Date:** 2026-08-30  
**Status:** local, non-production diagnostic  
**Scope:** WeMM production-shaped pre-annotation only

## Why this exists

The production WeMM runner needs a finite visual context, but a context window is
not an action annotation.  Copying a fixed four-second window into
`start_seconds`/`end_seconds` creates systematic boundary errors: the action may
start before the window, finish after it, or occupy only a short portion of the
window.  Temporal resolution therefore consumes a sequence of overlapping
context observations and proposes an interval from the model score trajectory.

This path does **not** make WeMM a frame-accurate segmenter.  It makes the
distinction explicit and gives a reviewer a measurable, source-relative
proposal with the evidence needed to accept, edit, split, reject, or abstain.

## Current route

```text
six-camera source
  -> bounded overlapping context windows (visual input only)
  -> WeMM video embedding + production phrase Top-K
  -> per-action score trajectory on the context grid
  -> camera-support filtering + hysteresis
  -> model-estimated interval sidecar
  -> human review
```

The historical per-window proposal remains unchanged.  Dense temporal output
is an additive `temporal_resolution` sidecar and does not relabel a window or
change the open production vocabulary.

### Context policy

`--temporal-mode dense_score` enables dense contexts.  If no stride is supplied,
the batch runner derives `window_seconds / 4` (four probes per context).  For a
four-second context this is a one-second stride.  The manifest records both the
unique source coverage and the extra overlap workload.  Dense windows are
therefore useful for temporal evidence but cost more decode/inference work.

The normal compatibility route remains `--temporal-mode none`, with the
historical non-overlapping window policy.

## Resolver semantics

For every action appearing in any context Top-K list, the resolver creates a
trajectory over all selected contexts.  A missing Top-K row is recorded as zero
**ranking support** for that probe; it is not interpreted as visual evidence
that the action is impossible.

Camera support is explicit.  A probe can be retained in the trajectory while
being excluded from boundary decisions when it does not meet
`--temporal-min-camera-support`.

Hysteresis prevents a single score fluctuation from opening and closing an
interval:

* `--temporal-start-threshold` opens a track;
* `--temporal-stop-threshold` releases a track and must not exceed the start
  threshold;
* `--temporal-merge-gap-seconds` joins nearby eligible probes; and
* `--temporal-min-duration-seconds` removes unusably short proposals.

Temporal score policy is explicit:

* `top1` (the dense-mode default) lets only the deterministic fused Top-K
  winner in each context contribute temporal support.  Lower-ranked candidates
  remain in the retained Top-K evidence for review, but their high raw
  similarity cannot create parallel full-recording tracks.
* `absolute` uses the raw fused similarity for controlled comparisons.  It is
  intentionally opt-in because current production artifacts cluster around
  0.70--0.74 and therefore do not provide a useful absolute boundary signal.

Boundary modes are deliberately named:

* `observed_probe` uses the active probe span; and
* `midpoint` places a transition between adjacent probe centres when a
  neighbouring probe exists.

Both modes are estimates.  The output records transition diagnostics, whether
the threshold was crossed, score delta, neighbouring probe, camera support,
and contributing window IDs.

## Review-only contract

Every temporal segment has:

* `boundary_status: MODEL_PROBE_BOUND`;
* `boundary_source: wemm_temporal_score`;
* `review_required: true`; and
* `automatic_eligible: false`.

The top-level temporal result also remains `production_eligible: false`, with
quality and official-gold status unmeasured/unestablished.  A reviewer may
project a confirmed interval into the normal annotation workflow, but the
resolver itself never writes measured or gold boundaries.

The review pack exposes the sidecar additively as:

```json
{
  "temporal_resolution": {"...": "resolver report", "segments": []},
  "temporal_segments": []
}
```

Existing `items[]` continue to represent windows and retain their original
`source_interval.status: WINDOW_CONTEXT_ONLY`.  Batch aggregation preserves
the sidecar under a separate temporal section and adds recording/source
lineage to each flattened segment; it does not mix temporal segments into the
window count or proposal count.

## Important current limitation

Historical WeMM artifacts show raw similarities clustered in a narrow range
(roughly 0.70--0.74 for several garment phrases).  Consequently, an absolute
threshold can keep a candidate active across most of a recording and produce a
wide, weakly localized interval.  Dense score trajectories remove the
architectural error of treating a context edge as an action edge, but they do
not by themselves prove accurate onset/offset localization.

The next controlled experiment should compare explicit score policies rather
than silently changing thresholds:

1. the implemented `top1` winner-gated policy;
2. `absolute` score with the current hysteresis;
3. a future relative prominence or rank-margin policy; and
4. camera-consensus variants on the same context grid.

Use the small reviewed diagnostic cohort first.  Keep the held-out-100 set
frozen until a policy is selected and reviewed.  Report interval quality,
candidate recall, accepted precision/coverage, and regressions separately.

## Commands

Open-runner diagnostic:

```powershell
python scripts/run_production_wemm_open.py `
  --manifest <manifest.json> `
  --phrase-catalog <catalog.json> `
  --model-dir <local-WeMM-snapshot> `
  --temporal-mode dense_score `
  --temporal-score-policy top1 `
  --temporal-boundary-mode midpoint
```

Resumable batch diagnostic:

```powershell
python scripts/run_production_wemm_batches.py `
  --source-preflight <preflight.json> `
  --phrase-catalog <catalog.json> `
  --model-dir <local-WeMM-snapshot> `
  --temporal-mode dense_score `
  --window-seconds 4 `
  --window-stride-seconds 1 `
  --temporal-score-policy top1
```

Changing temporal mode, stride, thresholds, camera support, or boundary mode
on a resumed batch is rejected.  Start a fresh diagnostic output directory for
a different arm so score trajectories and review decisions cannot be mixed.

## Evidence and quality status

This route is an experiment and review aid, not a production qualification:

* no official labels are read or written;
* no ontology or Mapper decision is made;
* Qwen and Mage are not invoked by the resolver; and
* a temporal proposal is not evidence that the model has correctly understood
  the action.

Quality must be measured against independently reviewed action intervals.  At
minimum, compare normal, reversed, and frozen score trajectories, and retain
the full Top-K/camera evidence for every accepted or rejected proposal.
