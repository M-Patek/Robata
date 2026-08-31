# Recovery candidate matrix — 2026-08-31

This matrix is a file-level triage record.  A row marked **hold** is not a
request to delete or merge its source worktree.

| Candidate | Base/shape | Focused evidence | Decision |
|---|---|---:|---|
| Mage source interval projection | current-main isolated branch; 3 files | 4 tests + ruff/compile | **included** |
| Atomic-event L0–L6 taxonomy | current-main isolated branch; 2 files | 11 tests + ruff/compile | **included** |
| Qwen hard-negative recurrence gate | current-main isolated branch; 2 files | 3 tests + ruff/compile | **included** |
| P11 state-transition consistency | `aebb4bf`, separate current-main worktree; 3 files | 9 tests + strict source mypy | **hold for next batch** |
| Production gold collection | old snapshot; sidecar/fixture assumptions | 13 behavioral passes; CLI fixture absent | **hold; fixture review** |
| Qwen anonymous pairwise core | old snapshot; core is stdlib-only | 9 tests | **hold; runner adaptation** |
| Qwen spatial visibility core | old snapshot; review-only | 6 tests | **hold; contract review** |
| Production selective routing | old snapshot; overlaps current diagnostics | 12 tests | **hold; semantic comparison** |
| Production model candidate projection | old snapshot; overlaps current output contracts | 7 tests | **do not merge wholesale** |
| Production quality evaluator | old snapshot; overlaps current evaluator | 10 tests | **do not merge wholesale** |
| Mage native codec consolidated PR | stale/runtime/container commit | no new tests; strict identity gates | **reject; rewrite separately** |
| Qwen/Mage observability mega PR | stale base; broad deletions/schema rewrites | collection errors and lint failures | **reject; do not cherry-pick** |

## Rules used

1. A candidate must be based on the current main line or be copied file-by-file
   into a fresh branch.
2. A candidate must not change published schemas, production routing, or model
   selection as part of a recovery batch.
3. Focused tests and formatting/compile checks must pass before inclusion.
4. Old branch commits are evidence sources, not merge units.  Generated media,
   temporary files, and unresolved fixtures stay quarantined.

