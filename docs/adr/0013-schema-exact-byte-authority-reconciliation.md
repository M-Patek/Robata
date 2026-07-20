# ADR 0013: Schema Exact-Byte Authority Reconciliation

- Status: Accepted one-time repository repair
- Date: 2026-07-20
- Governing authority: Architecture V1.1 Section 25.7 and ADR 0003

## Decision

`schemas/schema-catalog.json` is the authority for each published schema's exact
`SchemaRef`. Six early V1 artifacts were cataloged from CRLF bytes while Git's
historical blobs were normalized to LF. Windows checkouts could therefore pass
worktree validation even though an archive made from the Git tree failed the
catalog pins.

The six Git blobs listed explicitly in `.gitattributes` are reconciled once to
their already-published catalog digests. Their schema IDs, versions, catalog
entries, and semantic content do not change. The exceptions disable Git text
normalization only for those exact paths. All other JSON, Python, and Markdown
files use LF.

This repair does not permit an in-place V1 edit. After reconciliation, the
release contract compares every catalog digest with the corresponding Git index
blob and evaluates attributes from the index. Any later byte change, including
formatting or line endings, requires a new schema version.

The historical immutability checker recognizes an authority reconciliation only
when all three facts are machine-proven: the complete catalog entry is unchanged,
the baseline Git blob does not match that entry's existing SHA-256 pin, and the
candidate Git blob exactly matches the same pin. The result reports every
reconciled `(schema_id, version)` and its count. There is no bypass flag or path
allowlist. A baseline blob that already matches its pin receives no exception, and
a candidate that changes the pin, any other catalog field, or bytes away from the
pin is rejected.

## Consequences

- The repair commit must include `.gitattributes` and all six reconciled schema
  blobs together.
- Worktree-only schema checks are insufficient release evidence because checkout
  filters can hide a Git-object mismatch.
- Historical release immutability still requires comparison with a protected
  prior release/tag; the current catalog proves internal exact-byte consistency,
  not append-only history by itself.
- Once the repair commit is the protected baseline, these six blobs match their
  pins. The reconciliation condition can no longer authorize any later mutation.
