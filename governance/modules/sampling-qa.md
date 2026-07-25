# Sampling and QA

## Scope and path anchors
- Sampling: `src/robata/sampling/**` - start with `sampling/adaptive.py`
- QA pipeline: `src/robata/qa_pipeline/**` - start with `coarse.py` and `dense.py`
- QA bridge helpers: `application/canonical/{media_quality_source_binding,media_quality_supplemental,supplemental_qa_evidence}.py`

## How to dispatch
`sampling-qa / P<n> - <sampling, coarse QA, dense QA, context, or class-projection task>`

## Construction phases
1. **Adaptive selection** - choose frames/windows while retaining explainable coverage.
2. **Coarse and dense QA** - produce structured local QA evidence from selected media.
3. **Required-class projection** - map evidence to the product's 21 issue classes and clip marks.
4. **Quality tuning** - measure recall/cost tradeoffs with governed data when it becomes available.

## Relevant tests
- Fast: `python -m pytest tests/unit/test_sampling_adaptive.py tests/unit/test_qa_pipeline_core.py tests/unit/test_local_qa_product.py tests/unit/test_media_quality_supplemental.py`
- Broader: `python -m pytest tests/integration/test_canonical_offline.py`

## Read alongside
Read `source-media` before changing source windows, ROI, or retained-frame behavior. Read `inference-evidence` for model-call inputs and `canonical-integration` for the QA-to-run bridge.
