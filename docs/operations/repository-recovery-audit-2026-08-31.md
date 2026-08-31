# Repository recovery and consolidation audit — 2026-08-31

## Scope

This audit records the safe recovery from the stale local Qwen/Mage worktrees.
The repository contract is anchored to `origin/main`; no root-worktree reset,
clean, stash application, force deletion, schema rewrite, or production runtime
change was performed.

## Baseline and recovery

- Root worktree: `D:/Github/Robata`
- Root branch: `main`
- Baseline: `origin/main` at `1f76fc4`
- Local `main` vs `origin/main`: `0 ahead / 0 behind`
- Existing local backup tag: `backup/pre-consolidation-20260831`
- Existing stash was preserved; it is an older 2026-07-24 WIP and was not applied.
- All existing non-root branches and worktrees remain available for review.

An untracked directory named `audit-qwen-branch/` is present in the root.  It is
treated as a quarantine snapshot produced by the earlier rollback attempt.  It
has not been staged, deleted, or used as a merge source.  A second inaccessible
`audit-tmp-mage-recovery/` directory was also left untouched.

## Rejected historical PR candidates

### Mage recovery (`codex/mage-recovery-consolidated-20260831`, `355f59a`)

Not safe to merge as a whole.  It is based on an older tree, changes runtime and
container paths, has no corresponding new test suite in the commit, and carries
strict image/package/receipt identity gates that are outside the current
benchmark-only recovery scope.  It therefore remains an archival review target,
not a production merge candidate.

### Qwen/Mage observability (`codex/qwen-mage-quality-observability-20260823`,
`d550546`)

Not safe to merge.  Its merge base predates the current main line and its tree
diff is a large mixture of additions, deletions, and rewrites.  It removes
registered v1 schema files, changes the schema catalog and dependency metadata,
and cannot be collected cleanly because retained code references missing modules
and optional dependencies.  It is not a clean source for cherry-picking an
entire commit.

## Safe, isolated recovery batch

Branch: `codex/recovery-integration-20260831`, based directly on `origin/main`.

1. `e9c51a5` — benchmark-only Mage source-interval projection.
2. `ce08130` — benchmark-only L0–L6 atomic-event error taxonomy.

The two commits add no model/runtime/codec path and do not alter published
schemas.  Added source and tests are self-contained and avoid content identity
or receipt enforcement.

Validation from the isolated worktree:

```text
15 targeted tests passed
compileall passed
ruff check passed
ruff format --check passed
git diff --check origin/main..HEAD passed
```

The isolated branch is clean and is exactly two commits ahead of `origin/main`.

## Decision and next action

Keep `main` as the only canonical baseline.  Publish the isolated recovery batch
for review, and only merge it after the remote checks pass.  Continue extracting
individual benchmark modules from old snapshots only when each module is based on
the current main line, has a narrow dependency surface, and passes focused tests.
Do not merge either historical PR wholesale and do not perform destructive
worktree cleanup while any snapshot ownership is unresolved.

## Current worktree inventory (read-only audit)

The repository currently has 34 registered worktrees (including the root).  The
root is on the canonical baseline but has an untracked quarantine snapshot, so a
blanket `git add .`, `reset`, or `clean` would be unsafe.  The remaining worktrees
fall into these groups:

- **Current-line candidates:** `recovery-integration` (three commits including
  this report), `reconciliation-candidates`, and `reconciliation-hard-negative`.
- **Older but clean research branches:** Mage native/cold-path, WeMM, and
  production publish worktrees; they are behind the current main line and need
  file-level extraction rather than branch merges.
- **Dirty snapshots:** `root-reconciliation` (hundreds of untracked files),
  `local-core-rebase`, `local-wemm-followup`, and several production worktrees.
  These remain untouched until ownership and file-level provenance are known.
- **Obsolete experiments:** old Mage stream/single-route and dated production
  readiness worktrees.  They are candidates for later archival only; no force
  removal was performed in this phase.

This inventory is descriptive, not an instruction to delete anything.  The
canonical source of truth remains the clean `main` commit above.

## Additional isolated candidates

- `c758c18` (`qwen_hard_negative_recurrence_gate.py` and its focused test) was
  copied from `reconciliation-hard-negative` as a separate benchmark commit.
  Its three focused tests, formatting, compilation, and diff checks pass.  It
  remains isolated on this review branch and does not alter inference or
  production routing.
- `42a55db` (`p11_state_transition_consistency.py` plus evaluator and test) was
  extracted into a separate clean worktree and is **not** included here yet.
  It passes nine focused tests and static checks, but is substantially larger
  and contains a broader review-boundary surface.  It will be considered only
  after the smaller recovery batch is accepted.

The quarantined `audit-qwen-branch/` snapshot currently contains approximately
2,096 files (including generated bytecode and placeholder media, about 52 MB).
That count is evidence for quarantine only; none of those files has been staged
or treated as source.

## Publication status

The remote fetch completed and confirmed `main` is current.  Publishing the
review branch was attempted but the environment rejected the outbound write
because explicit destination authorization/credentials were unavailable.  No
remote branch or PR was created by this phase.  The local branch and commit list
are complete and can be pushed later after explicit authorization.
