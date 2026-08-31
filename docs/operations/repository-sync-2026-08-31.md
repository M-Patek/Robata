# Repository synchronization snapshot — 2026-08-31

This is a current-state handoff for the local repository cleanup. It supersedes
the older recovery notes for synchronization decisions; historical audit files
remain unchanged.

## Canonical source

- Root checkout: `D:\\Github\\Robata`, branch `main`.
- Local `main` and the last known `origin/main` tracking ref both point to
  `e8056d1` (the native-video-minframes review merge, PR #41).
- A live `git fetch origin main --prune` was attempted during this cleanup but
  GitHub reset the HTTPS connection. Therefore the equality above is the last
  known tracking state, not a new live-network assertion.
- The root worktree has no tracked changes. The only ordinary untracked items
  are the quarantined `audit-qwen-branch/` snapshot and
  `audit_interval_grep.txt`; neither is a merge source.

## What was (and was not) missing from main

The current main line already contains the reviewed WeMM retrieval/fusion and
temporal interval work, the Qwen native-video route/minimum-frame fix, Mage
source-interval projection, the L0–L6 taxonomy/hard-negative gate, P11 state
transition consistency, and production-gold collection.

The WeMM sidecars were compared file-by-file rather than merged wholesale:

- `local-core-rebase-20260830` and `wemm-temporal-integration-20260830` are
  older temporal snapshots and would remove later main-line behavior.
- `local-wemm-followup-20260830` and
  `wemm-qwen-verifier-publish-20260830` are duplicate/old-format copies with
  small regressions, not new source.
- `production-model-driven-interval-20260830` is an ancestor/old interval
  copy. `wemm-fusion-retrieval-publish-20260830` is also an older published
  snapshot.

No WeMM worktree contained a tracked path that was both absent from main and a
safe, tested improvement. **No WeMM sidecar was cherry-picked or pushed.**

The Mage recovery variants were likewise held: they are stale rebase stacks,
lack the required current-line test coverage, and some carry receipt/image
identity machinery outside this cleanup scope. They remain available as
historical evidence, not as merge candidates.

## Reversible local cleanup completed

The following already-merged, clean checkouts were removed one at a time with
their branch refs retained:

- `canonical-main-20260830`
- `native-route-integration-20260831`
- `native-video-minframes-review-20260831`
- `p11-recovery-pr-20260831`
- `production-gold-collection-20260831`
- `recovery-integration-20260831`
- `mage-native-coldpath-20260809`
- `mage-native-sustained-20260808`
- `single-route-20260808`

Before removal, the native bridge media/temporal manifest from
`native-route-integration-20260831` and the canonical temporary output were
moved to the ignored local archive at
`archive/worktree-cleanup-20260831/`. Pure root caches were moved to
`archive/local-cache-20260831/` with manifests. No source, schema, model, or
production input directory was deleted.

## Deliberately retained

Dirty worktrees, old branches with unresolved evidence, `root-local-snapshot`,
`root-reconciliation`, the Qwen/Mage observability snapshot, the temporary
Qwen end-to-end checkout, `data/source/`, `.agent_tmp/`, and the inaccessible
`audit-tmp-mage-recovery/` directory were not touched. They are not evidence
that main is missing code; they are retained review material or local data.

## Publication state

This record is intentionally a small, documentation-only synchronization
change. It does not modify schemas, Web/API/UI, model selection, or runtime
behavior. After GitHub connectivity/authentication is restored, publish this
branch as a small PR and merge only this record; do not publish the rejected
WeMM/Mage snapshots wholesale.
