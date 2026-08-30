# Production WeMM full-corpus progress (2026-08-29)

**Status:** `PRODUCTION_SHADOW_PREANNOTATION_COMPLETE`

This is an operational progress record. It is not a semantic-quality result,
human gold set, or production qualification decision.

## Corpus and coverage

| Layer | Count | Interpretation |
|---|---:|---|
| MCAP recordings in the source archive | 37 | 36 readable, 1 malformed/truncated |
| Source-preflight pass | 36 | Structural media readability only; not clip-level QA |
| Context windows | 788 | Fixed processing units, not action boundaries |
| Camera-window inputs | 4,728 | Six cameras per window |
| WeMM proposals | 788 | One fused production-vocabulary proposal per window |
| Ambiguity routing queue | 565 windows / 3,390 camera rows | Low margin, low consensus, or verb conflict |

Authoritative source-preflight artifact:
`.agent_tmp/production_corpus_source_preflight_20260828.json`.

## WeMM result currently available

The full production-only WeMM review aggregate is
`.agent_tmp/production_wemm_full_postprocess_20260828/review_aggregate.json`.
It records `epic_ontology_used=false`, `mapper_used=false`, and retains each
proposal's camera evidence, Top-K list, score, and margin. The observed labels
are the current six-item open production phrase catalog:

```text
pick up garment | flatten garment | fold garment |
smooth garment | spread garment | adjust garment
```

These frequencies describe model proposals, not correctness or the final
production vocabulary. Every proposal remains `PENDING` and every source
interval is `WINDOW_CONTEXT_ONLY`.

## Editable draft generated

The inference-free bridge generated:

`.agent_tmp/production_wemm_annotation_draft_full_20260829/annotation-draft.json`

and its Markdown companion. The draft contains 788 editable provisional
segments, one per context window, with:

- required verb/noun/optional-field slots;
- null `start_seconds`/`end_seconds` when no source-bound action interval was
  measured;
- explicit context interval and `is_action_boundary=false`;
- confidence, camera support, evidence, Top-K, margin, and raw proposal;
- `accept | edit | split | reject | abstain` pending decisions;
- production-only provenance and `gold_written=false`.

No model was invoked, no media was decoded, and no gold or published schema was
modified while creating this draft.

## Qwen ambiguity diagnostic

The completed pairwise pilot is
`.agent_tmp/qwen_pairwise_top2_pilot24_cam01_20260829/aggregate-reparsed.json`.
After the narrow no-selection parser normalization: 24/24 rows are parseable,
23 selected a candidate (all WeMM rank 1), 9 abstained, and rank-2 rescue was
0. This tests verifier behavior only; it does not establish production
accuracy. The earlier full ambiguity artifact is a dry-run/profile-failure
record and must not be counted as a completed Qwen run.

## Admission gaps and next work

1. All 37 recordings still have `qa_status=PENDING`; source preflight pass is
   not visual QA. The draft is therefore shadow-only.
2. The malformed recording must be repaired and independently re-preflighted;
   an experimental repair with one decoded frame per camera is not sufficient.
3. A source-bound human review must fill boundaries and decisions before formal
   precision/recall or production eligibility can be reported.
4. Only after the review contract is stable should the 565-window Qwen queue be
   run with the approved local profile and `--resume`; Mage remains a separate
   native-codec comparison route.

Official quality, gold, and production eligibility remain `NOT_MEASURED`,
`NOT_ESTABLISHED`, and `false` respectively.
