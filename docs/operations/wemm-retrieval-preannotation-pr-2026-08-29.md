# WeMM retrieval and production pre-annotation

**Status:** `LOCAL_NONPRODUCTION_ONLY`

This change adds a benchmark and shadow path for WeMM-Embedding-2B.  It does
not change the production Web/API/UI route, the Mapper, or any published
schema.  Production outputs remain editable pre-annotations and are never
gold.

## Architecture

```text
bounded native video
        |
        v
 WeMM video embedding
        |
 cosine retrieval against caller-supplied text prototypes
        |
 per-camera Top-K -> deterministic six-camera fusion
        |
 review-only pre-annotation (proposal + Top-K + evidence + provenance)
```

The EPIC runner keeps its action-pair catalog for a label-blind retrieval
baseline.  The production adapter uses an explicit owner/Terra-facing phrase
catalog with opaque provisional IDs; it never imports EPIC IDs into the
production pre-annotation envelope.  Context windows are not action
boundaries: boundary fields remain null unless a source-bound interval is
explicitly measured.

## Local measurements

On the fixed `sample-medium.mcap` cohort (40.8335 seconds, 10 windows, six
cameras, 60 camera-window inputs), the resident native-video path measured:

| path | wall time | model time |
|---|---:|---:|
| serial | 86.128 s | 28.124 s |
| batch 2 | 74.691 s | 16.687 s |
| batch 4 | 70.986 s | 12.982 s |

Cold MCAP/H.264/PIL decode is approximately 56-63 seconds and dominates this
cohort.  The quality screen over eight non-abstain Terra-surrogate windows
reached R@1 `50.0%`, R@5 `100.0%`, and MRR `0.650` at eight frames and the high
pixel budget.  These are development diagnostics, not production accuracy.

## Production shadow coverage

The source corpus currently contains 37 recordings (789 planned context
windows and 4,734 camera-window inputs).  One recording is malformed at source
preflight; 788 windows have review-only WeMM proposals.  All proposals remain
`PENDING_REVIEW`, and no accepted source-bound gold exists, so production
precision/recall and eligibility are `NOT_MEASURED`/`false`.

Each proposal retains the raw Top-K list, per-camera support, score/margin,
context interval, and provenance controls.  The editable draft supports
`accept`, `edit`, `split`, `reject`, and `abstain` without silently turning a
processing window into an action segment.

## Checks

The focused retrieval, decoding, batching, production-shadow, and post-hoc
tests pass on a clean checkout (`223 passed` with the optional model runtime
available; the three model-runtime tests skip when PyTorch is absent).  Optional model/media
dependencies (`av`, `mcap`, `mcap-protobuf-support`, Pillow, PyTorch,
Transformers, and `qwen-vl-utils`) are imported only at execution seams; pure
ranking and envelope validation remain runnable with test doubles.

No model weights, MCAP payloads, pixels, hashes, or digests are committed.
