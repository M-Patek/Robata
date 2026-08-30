# WeMM rank-distance and camera-consensus diagnostics — 2026-08-28

> **Diagnostic only.** The comparison uses the independent Terra review as a
> surrogate reference. Official production gold remains `NOT_ESTABLISHED`, and
> quality remains `NOT_MEASURED`.

## Scope

The existing rank report already measured exact rank distance, rank buckets,
lexical hard negatives, directed confusion pairs/clusters, and fused Top-1 vs
Top-2 score margins. It did not retain the per-camera rankings from the WeMM
sidecar, so rank errors could not be stratified by camera agreement. The
additive projection now retains compact per-camera top-1/top-2 summaries and
ranked action keys; legacy sidecars report `NOT_AVAILABLE`.

Artifacts:

- `.agent_tmp/p24_wemm_terra_camera_comparison_20260828.json`
- `.agent_tmp/p24_wemm_rank_camera_analysis_20260828.json`
- `.agent_tmp/p24_wemm_rank_camera_analysis_20260828.md`

## Surrogate cohort result

Eight eligible windows from `sample-medium.mcap` were scored. Each sidecar had
six observed cameras and six recorded candidates per window. The camera rows are
post-hoc identity projections; agreement is not independent semantic evidence.

| Variant | Window R@1 | Window R@5 | MRR | Mean camera consensus | Strict majority | Consensus winner = reference | Fused Top-1 = consensus |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 25.0% | 87.5% | 0.425 | 72.9% | 5/8 | 2/8 | 6/8 |
| verb_noun | 25.0% | 100.0% | 0.473 | 79.2% | 8/8 | 2/8 | 7/8 |
| natural | 37.5% | 87.5% | 0.540 | 75.0% | 5/8 | 2/8 | 7/8 |

Canonical rank-distance remains predominantly far-error: 1/7 Top-1 errors
were rank 2–3, while 6/7 were rank 4 or lower. The camera-majority winner
matching the surrogate reference only 2/8 windows shows that high camera
agreement alone cannot justify automatic publication.

## Interpretation and next gate

The implementation is sufficient for the intended diagnostic split:

1. **Retrieval/ranking:** exact rank distribution, MRR, margins and hard
   negatives identify whether a wrong Top-1 is near or far from the reference.
2. **View agreement:** coverage, consensus fraction, strict-majority status and
   fused-vs-consensus agreement expose camera disagreement as a routing feature.
3. **Evidence boundary:** all metrics remain surrogate-only until independent
   source-bound gold exists; no mapper, ontology, model or production contract
   was changed.

Camera consensus should therefore gate review/routing, not replace semantic
verification. Re-run the same report against adjudicated source-bound gold when
available, then evaluate whether consensus correlates with true rank accuracy.
