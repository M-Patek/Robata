# Local validation workstreams (2026-07-19)

This document records the local-only completion of the engineering items listed in the project
execution table.  It does **not** promote a fake model to production and it does not install or
contact a cloud/provider service.

## Completed checks

| Workstream | Local implementation/evidence | Guardrail |
|---|---|---|
| Windows ProcessPool / PNG reuse PoC | `robata.runtime.process_pool_poc`, `scripts/probe_process_pool.py`; `ReusablePngEncoder` is byte-identical to isolated PyAV encoding for same-size RGB frames. | Spawn support is reported, not assumed; codec reuse remains opt-in and non-certifying. |
| Local model adapter boundary | `robata.adapters.local_vision_model`; lazy optional `TransformersVisionModelAdapter.load_local(..., local_files_only=True)`. | No automatic download; missing dependency raises `OptionalDependencyUnavailable`; runner must return validated `VisionInferenceOutcome`. |
| QA sample + 21 issue matrix | `scripts/validate_qa_sample.py`; `sample-medium.mcap` inspection and six-camera pass fixture. | Matrix validates warning/fail policy; no detector quality claim is made. |
| Search MVP | `scripts/validate_search_mvp.py`; verb-family normalization, noun/location/hand facets and direct playback targets. | Structured-label, zero-GPU index only; no embeddings or network. |
| SharedFrameCache stress | `scripts/stress_frame_cache.py`; concurrent same-video calls produce one decode attempt per video. | Filesystem cache only; output is engineering evidence. |
| Worker integration | `scripts/validate_worker_integration.py`; `InMemoryTaskQueue + PipelineWorker` executes QA -> annotation -> search. | Queue is local/in-memory; durable Redis/PostgreSQL adapters remain future work. |
| Synthetic benchmark | `scripts/benchmark_synthetic.py`; serial/parallel output hashes and throughput accounting. | `measurement_status=NOT_MEASURED`; fake smoke cannot be certifying. |
| Capacity calibration | `scripts/calibrate_capacity.py`; 1/2/4 H100 × 7B/32B assumption matrix. | `measurement_status=ASSUMPTION`; `production_eligible=false`. |

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\validate_qa_sample.py
.\.venv\Scripts\python.exe scripts\validate_search_mvp.py
.\.venv\Scripts\python.exe scripts\stress_frame_cache.py
.\.venv\Scripts\python.exe scripts\validate_worker_integration.py
.\.venv\Scripts\python.exe scripts\benchmark_synthetic.py
.\.venv\Scripts\python.exe scripts\calibrate_capacity.py
.\.venv\Scripts\python.exe scripts\verify_requirements.py --output reports\requirements-acceptance-2026-07-19.json
```

For one aggregate run (including sample MCAP inspection), use `scripts/run_local_workstreams.py`.

The aggregate acceptance output is stored in
`reports/requirements-acceptance-2026-07-19.json`.  The expected invariant flags are:

```text
provider_requests=0
execution_mode=LOCAL_DEVELOPMENT_FAKE_MODEL
production_eligible=false
```

## Promotion gate

A production throughput/quality claim still requires a governed corpus, measured model runs,
calibration, and an explicit approval token.  `certify_summary` rejects fake/local execution
modes even when a caller supplies a corpus identifier.

