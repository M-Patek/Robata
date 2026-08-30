# Production surrogate quality comparison — 2026-08-27

> **AGENT_SURROGATE_MEASURED_NON_GOLD.** This is an accelerated visual-review
> diagnostic.  Official production quality remains `NOT_MEASURED`.

## Reference and controls

Codex inspected the six-camera, 16-frame surfaces for the ten contiguous
four-second windows and explicitly accepted the refined coarse observations for
exploratory comparison.  The resulting pack is
`.agent_tmp/production_review_pack_agent_surrogate_4s_16f_20260827.json`.
It is marked `AGENT_SURROGATE_NON_GOLD`,
`official_gold_status=NOT_ESTABLISHED`,
`human_adjudication=NOT_PERFORMED`, and `production_eligible=false`.
It is not training data and does not establish official production gold.

Frozen WeMM/Qwen sidecars were reused; no model or media rerun was performed.
Mage remains excluded because the source-bound native codec/cache toolchain is
unavailable.

## Exploratory result

The joined report is `.agent_tmp/production_surrogate_quality_20260827.json`
(Markdown: `.agent_tmp/production_surrogate_quality_20260827.md`).  It keeps
`official_quality_status=NOT_MEASURED` even though the underlying generic
evaluator reports a local calculation over the surrogate pack.

Strict exact matching is zero for all routes because the frozen resolver uses
the EPIC vocabulary (`cloth`, etc.), while the production review contract uses
the free noun `garment`.  With the explicit exploratory textile-family alias,
Qwen reaches family @5 **40%** on primary labels and **80%** when reviewer
alternatives are allowed; WeMM and the current WeMM-heavy hybrid remain **0%**.

This supports a Qwen-first candidate-routing hypothesis and an explicit need
for an approved production action vocabulary/mapping.  It is not a precision,
recall, boundary, structured-field, training, or production-capacity claim, and
the hybrid weights remain frozen.

## Next gate

Replace the surrogate with independently reviewed source-bound segments (or an
approved production ontology/mapping), then run the frozen evaluator once.  Do
not open held-out-100 or expand the model matrix before that gate.
