from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qualify_local_mage_dcvc_provider_v2.py"


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "dcvc_provider_qualification_test", SCRIPT
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _telemetry(status: str = "MEASURED") -> dict[str, object]:
    devices: list[dict[str, object]] = []
    if status != "UNAVAILABLE":
        devices.append(
            {
                "gpu_index": 0,
                "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "utilization_gpu_percent_mean": 62.0,
                "utilization_gpu_percent_max": 100.0,
                "memory_used_fraction_max": 0.64,
                "memory_used_mib_max": 5240,
                "memory_total_mib": 8188,
                "temperature_celsius_max": 72.0,
                "power_draw_watts_max": 96.0,
                "sample_count": 20,
            }
        )
    return {
        "measurement_status": status,
        "device_summaries": devices,
        "errors": [] if status == "MEASURED" else ["telemetry unavailable"],
    }


def _projections() -> dict[str, list[dict[str, object]]]:
    return {
        "qa": [
            {"match_key": f"qa-{ordinal}", "semantic_sha256": _digest(400 + ordinal)}
            for ordinal in range(5)
        ],
        "event": [
            {
                "match_key": "event-a",
                "semantic_sha256": _digest(420),
                "label": "pick_up_cup",
                "start_ns": 9_000_000_000,
                "end_ns": 12_000_000_000,
                "confidence": 0.91,
            },
            {
                "match_key": "event-b",
                "semantic_sha256": _digest(421),
                "label": "place_cup",
                "start_ns": 22_000_000_000,
                "end_ns": 25_000_000_000,
                "confidence": 0.89,
            },
        ],
        "evidence": [
            {
                "match_key": "evidence-a",
                "semantic_sha256": _digest(430),
                "supports": True,
                "confidence": 0.92,
            },
            {
                "match_key": "evidence-b",
                "semantic_sha256": _digest(431),
                "supports": True,
                "confidence": 0.88,
            },
        ],
        "track": [
            {
                "match_key": "track-a",
                "semantic_sha256": _digest(440),
                "label": "pick_up_cup",
                "start_ns": 9_000_000_000,
                "end_ns": 12_000_000_000,
                "confidence": 0.90,
            }
        ],
        "fusion": [
            {
                "match_key": "fusion-a",
                "semantic_sha256": _digest(450),
                "disposition": "RESOLVED",
                "start_ns": 9_000_000_000,
                "end_ns": 12_000_000_000,
                "confidence": 0.90,
            }
        ],
    }


def _evidence(
    *,
    generation: str,
    variant_id: str,
    max_side: int,
    seed: int,
    cold_wall: float,
    full_asset_parity: bool = False,
    output_parity: bool = True,
    telemetry_status: str = "MEASURED",
) -> dict[str, object]:
    baseline = generation == "OBSERVED_V1"
    assets = [
        100 + ordinal if baseline or full_asset_parity else seed + 100 + ordinal
        for ordinal in range(5)
    ]
    outputs = [
        300 + ordinal if baseline or output_parity else seed + 300 + ordinal for ordinal in range(5)
    ]
    return {
        "format_version": "mage-dcvc-provider-qualification-evidence-v1",
        "evidence_class": "LOCAL_MEASUREMENT",
        "production_eligible": False,
        "variant_id": variant_id,
        "provider": {
            "generation": generation,
            "recipe_version": "observed-v1" if baseline else "explicit-provider-v2",
            "implementation_sha256": _digest(10 if baseline else seed + 10),
            "effective_config_sha256": _digest(20 if baseline else seed + 20),
            "cache_namespace_identity": _digest(30 if baseline else seed + 30),
            "max_side": max_side,
            "recurrent_work_semantics": "FULL_RECURRENT_CHAIN_THROUGH_LAST_SAMPLED_FRAME",
            "sequence_length_is_compute_cap": False,
            "inference_identity_binds_provider_recipe": not baseline,
        },
        "sample": {
            "source_media_sha256": _digest(40),
            "segment_manifest_sha256": _digest(41),
            "media_duration_ns": 40_000_000_000,
            "segments": [
                {
                    "ordinal": ordinal,
                    "start_ns": ordinal * 8_000_000_000,
                    "end_ns": (ordinal + 1) * 8_000_000_000,
                    "source_content_sha256": _digest(50 + ordinal),
                    "source_byte_count": 10_000 + ordinal,
                }
                for ordinal in range(5)
            ],
        },
        "control_identity": {
            "checkpoint_manifest_sha256": _digest(60),
            "model_identity_sha256": _digest(61),
            "prompt_sha256": _digest(62),
            "decoder_identity_sha256": _digest(63),
            "max_new_tokens": 256,
        },
        "cold_preparation": {
            "wall_seconds": cold_wall,
            "provider_process_start_count": 5 if baseline else 1,
            "provider_model_load_count": 5 if baseline else 1,
            "provider_startup_seconds": 2.0 if baseline else 1.0,
            "provider_model_load_seconds": 20.0 if baseline else 4.0,
            "startup_load_included_in_wall": True,
            "per_segment": [
                {
                    "ordinal": ordinal,
                    "preparation_seconds": cold_wall / 5,
                    "asset_identity_sha256": _digest(seed + 200 + ordinal),
                    "asset_exact_sha256": _digest(assets[ordinal]),
                    "meta_identity_sha256": _digest(seed + 220 + ordinal),
                    "meta_exact_sha256": _digest(seed + 240 + ordinal),
                    "meta_semantic_sha256": _digest(seed + 260 + ordinal),
                    "verified": True,
                }
                for ordinal in range(5)
            ],
            "gpu_telemetry": _telemetry(telemetry_status),
        },
        "endpoint": {
            "timed_wall_seconds": 20.0,
            "model_load_seconds": 17.0,
            "model_load_included_in_wall": False,
            "per_segment": [
                {
                    "ordinal": ordinal,
                    "inference_identity_sha256": _digest(seed + 500 + ordinal),
                    "result_artifact_identity_sha256": _digest(seed + 520 + ordinal),
                    "result_artifact_exact_sha256": _digest(seed + 540 + ordinal),
                    "output_text_sha256": _digest(outputs[ordinal]),
                    "observation_semantic_sha256": _digest(560 + ordinal),
                }
                for ordinal in range(5)
            ],
            "gpu_telemetry": _telemetry(telemetry_status),
        },
        "projected_semantics": _projections(),
    }


def _write(tmp_path: Path, name: str, evidence: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    baseline = _write(
        tmp_path,
        "baseline.json",
        _evidence(
            generation="OBSERVED_V1",
            variant_id="observed-v1",
            max_side=0,
            seed=1_000,
            cold_wall=180.0,
        ),
    )
    full = _write(
        tmp_path,
        "full.json",
        _evidence(
            generation="PROVIDER_V2",
            variant_id="provider-v2-full",
            max_side=0,
            seed=2_000,
            cold_wall=150.0,
            full_asset_parity=True,
        ),
    )
    bounded_672 = _write(
        tmp_path,
        "bounded-672.json",
        _evidence(
            generation="PROVIDER_V2",
            variant_id="provider-v2-672",
            max_side=672,
            seed=3_000,
            cold_wall=60.0,
        ),
    )
    bounded_448 = _write(
        tmp_path,
        "bounded-448.json",
        _evidence(
            generation="PROVIDER_V2",
            variant_id="provider-v2-448",
            max_side=448,
            seed=4_000,
            cold_wall=35.0,
        ),
    )
    return baseline, full, bounded_672, bounded_448


def _gate(comparison: dict[str, object], gate_id: str) -> dict[str, object]:
    gates = comparison["gates"]
    assert isinstance(gates, list)
    return next(gate for gate in gates if gate["gate_id"] == gate_id)


def test_report_selects_fastest_quality_qualified_bounded_candidate(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, bounded_448 = _inputs(tmp_path)

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[bounded_672, full, bounded_448],
    )

    assert payload["qualification_status"] == "PASSED"
    assert payload["recommended_variant_id"] == "provider-v2-448"
    assert payload["recommendation"].startswith("ADOPT_FASTEST")
    assert payload["production_eligible"] is False
    assert payload["qualification_scope"]["real_gpu_benchmark_executed_by_this_tool"] is False
    assert payload["qualification_scope"]["sequence_length_is_compute_cap"] is False
    baseline_summary = payload["observed_v1_baseline"]
    assert baseline_summary["cold_preparation"]["realtime_factor"] == pytest.approx(40 / 180)
    full_summary = payload["provider_v2_full_resolution"]
    assert full_summary["variant"]["cold_preparation"]["provider_model_load_count"] == 1
    assert full_summary["variant"]["cold_to_result"]["wall_seconds"] == 187.0
    bounded = payload["provider_v2_bounded_candidates"]
    assert [item["variant"]["provider"]["max_side"] for item in bounded] == [448, 672]
    assert all(item["locally_adoptable"] for item in bounded)
    assert len(payload["semantic_sha256"]) == 64


def test_full_resolution_requires_exact_assets_outputs_and_projection_semantics(
    tmp_path: Path,
) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    full_document = json.loads(full.read_text(encoding="utf-8"))
    full_document["endpoint"]["per_segment"][2]["output_text_sha256"] = _digest(999_999)
    full.write_text(json.dumps(full_document), encoding="utf-8")

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_672],
    )

    full_result = payload["provider_v2_full_resolution"]
    assert full_result["locally_adoptable"] is False
    assert _gate(full_result, "FULL_RESOLUTION_MAGE_OUTPUT_PARITY")["passed"] is False
    assert payload["qualification_status"] == "FAILED"
    assert payload["recommendation"] == "KEEP_OBSERVED_V1_BASELINE"


def test_bounded_quality_delta_blocks_candidate(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    bounded_document = json.loads(bounded_672.read_text(encoding="utf-8"))
    fusion = bounded_document["projected_semantics"]["fusion"][0]
    fusion["disposition"] = "AMBIGUOUS"
    fusion["confidence"] = 0.70
    fusion["end_ns"] = 13_000_000_000
    bounded_672.write_text(json.dumps(bounded_document), encoding="utf-8")

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_672],
    )

    candidate = payload["provider_v2_bounded_candidates"][0]
    assert candidate["locally_adoptable"] is False
    gate = _gate(candidate, "FUSION_QUALITY")
    assert gate["passed"] is False
    assert gate["observed"]["disposition_agreement"] == 0.0
    assert gate["observed"]["maximum_boundary_drift_seconds"] == 1.0
    assert payload["recommended_variant_id"] == "provider-v2-full"


def test_missing_gpu_telemetry_is_reported_and_blocks_adoption(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    bounded_document = _evidence(
        generation="PROVIDER_V2",
        variant_id="provider-v2-672",
        max_side=672,
        seed=3_000,
        cold_wall=60.0,
        telemetry_status="UNAVAILABLE",
    )
    bounded_672.write_text(json.dumps(bounded_document), encoding="utf-8")

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_672],
    )

    candidate = payload["provider_v2_bounded_candidates"][0]
    assert _gate(candidate, "GPU_TELEMETRY_COMPLETE")["passed"] is False
    assert _gate(candidate, "PEAK_VRAM_SAFETY")["observed"] is None
    assert candidate["variant"]["gpu_telemetry"]["sample_count"] == 0
    assert payload["recommended_variant_id"] == "provider-v2-full"


def test_rejects_seq_len_compute_cap_claim(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    candidate = json.loads(bounded_672.read_text(encoding="utf-8"))
    candidate["provider"]["sequence_length_is_compute_cap"] = True
    bounded_672.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(module.MageDcvcQualificationInputError, match="seq_len_frames"):
        module.build_qualification_report(
            baseline_evidence=baseline,
            provider_v2_evidence=[full, bounded_672],
        )


def test_rejects_any_sample_or_control_identity_drift(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    candidate = json.loads(bounded_672.read_text(encoding="utf-8"))
    candidate["sample"]["segments"][0]["source_content_sha256"] = _digest(777_777)
    candidate["control_identity"]["prompt_sha256"] = _digest(888_888)
    bounded_672.write_text(json.dumps(candidate), encoding="utf-8")

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_672],
    )

    comparison = payload["provider_v2_bounded_candidates"][0]
    assert _gate(comparison, "SAME_SAMPLE")["passed"] is False
    assert _gate(comparison, "CONTROL_IDENTITY_PARITY")["passed"] is False
    assert comparison["locally_adoptable"] is False


def test_rejects_duplicate_effective_config_or_namespace(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, bounded_448 = _inputs(tmp_path)
    document_448 = json.loads(bounded_448.read_text(encoding="utf-8"))
    document_672 = json.loads(bounded_672.read_text(encoding="utf-8"))
    document_448["provider"]["effective_config_sha256"] = document_672["provider"][
        "effective_config_sha256"
    ]
    bounded_448.write_text(json.dumps(document_448), encoding="utf-8")

    with pytest.raises(module.MageDcvcQualificationInputError, match="effective config"):
        module.build_qualification_report(
            baseline_evidence=baseline,
            provider_v2_evidence=[full, bounded_672, bounded_448],
        )


def test_persistent_provider_must_start_and_load_once(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    candidate = json.loads(bounded_672.read_text(encoding="utf-8"))
    candidate["cold_preparation"]["provider_process_start_count"] = 5
    candidate["cold_preparation"]["provider_model_load_count"] = 5
    bounded_672.write_text(json.dumps(candidate), encoding="utf-8")

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_672],
    )
    comparison = payload["provider_v2_bounded_candidates"][0]

    assert _gate(comparison, "SINGLE_PROVIDER_START_AND_LOAD")["passed"] is False
    assert comparison["locally_adoptable"] is False


def test_argument_order_does_not_change_report_identity(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, bounded_448 = _inputs(tmp_path)

    left = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_448, bounded_672],
    )
    right = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[bounded_672, full, bounded_448],
    )

    assert left == right
    assert left["semantic_sha256"] == right["semantic_sha256"]


def test_main_writes_canonical_nonproduction_report(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    output = tmp_path / "qualification.json"

    exit_code = module.main(
        [
            "--baseline-evidence",
            str(baseline),
            "--provider-v2-evidence",
            str(bounded_672),
            "--provider-v2-evidence",
            str(full),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["authority"] == "LOCAL_QUALIFICATION_NON_PRODUCTION"
    assert payload["production_eligible"] is False
    assert output.read_bytes() == module.canonical_json_bytes(payload)


def test_versioned_identity_and_cold_wall_inclusion_are_fail_closed(tmp_path: Path) -> None:
    module = _module()
    baseline, full, bounded_672, _ = _inputs(tmp_path)
    baseline_document = json.loads(baseline.read_text(encoding="utf-8"))
    candidate = json.loads(bounded_672.read_text(encoding="utf-8"))
    for field in ("recipe_version", "implementation_sha256", "effective_config_sha256"):
        candidate["provider"][field] = baseline_document["provider"][field]
    for ordinal in range(5):
        for field in ("asset_identity_sha256", "meta_identity_sha256", "meta_semantic_sha256"):
            candidate["cold_preparation"]["per_segment"][ordinal][field] = baseline_document[
                "cold_preparation"
            ]["per_segment"][ordinal][field]
        candidate["endpoint"]["per_segment"][ordinal]["inference_identity_sha256"] = (
            baseline_document["endpoint"]["per_segment"][ordinal]["inference_identity_sha256"]
        )
    candidate["cold_preparation"]["startup_load_included_in_wall"] = False
    bounded_672.write_text(json.dumps(candidate), encoding="utf-8")

    payload = module.build_qualification_report(
        baseline_evidence=baseline,
        provider_v2_evidence=[full, bounded_672],
    )
    comparison = payload["provider_v2_bounded_candidates"][0]

    for gate_id in (
        "PROVIDER_V2_IMPLEMENTATION_AND_CONFIG_ISOLATION",
        "PROVIDER_ASSET_META_AND_INFERENCE_IDENTITY_ISOLATION",
        "COLD_WALL_INCLUDES_PROVIDER_STARTUP_AND_LOAD",
    ):
        assert _gate(comparison, gate_id)["passed"] is False
    assert comparison["locally_adoptable"] is False


def _retained_gpu_document() -> dict[str, object]:
    return {
        "measurement_status": "MEASURED",
        "summary": [
            {
                "gpu_index": 0,
                "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "sample_count": 10,
                "utilization_gpu_percent_mean": 50.0,
                "utilization_gpu_percent_max": 100.0,
                "memory_used_mib_max": 5000.0,
                "memory_total_mib": 8188.0,
                "memory_used_fraction_max": 0.62,
                "temperature_celsius_max": 72.0,
                "power_draw_watts_max": 95.0,
            }
        ],
        "errors": [],
    }


def test_retained_observed_prep_does_not_infer_missing_segment_timings(
    tmp_path: Path,
) -> None:
    module = _module()
    preparation_root = tmp_path / "observed"
    cache_root = tmp_path / "cache" / _digest(801)
    sidecar_root = cache_root / ".robata-entries"
    preparation_root.mkdir()
    sidecar_root.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    for ordinal in range(5):
        segment_sha256 = _digest(810 + ordinal)
        source = tmp_path / "segments" / f"{ordinal:06d}-{segment_sha256}.mp4"
        logical_identity = _digest(820 + ordinal)
        entries.append(
            {
                "admission": "BUILT",
                "entry_semantic_sha256": _digest(830 + ordinal),
                "logical_cache_identity": logical_identity,
                "provider_cache_directory": str(cache_root / f"entry-{ordinal}"),
                "source_path": str(source),
            }
        )
        sidecar = {
            "assets": [
                {
                    "relative_path": "canvas_000.jpg",
                    "byte_count": 10 + ordinal,
                    "sha256": _digest(840 + ordinal),
                },
                {
                    "relative_path": "meta.json",
                    "byte_count": 20 + ordinal,
                    "sha256": _digest(850 + ordinal),
                },
            ],
            "source_path": str(source),
            "source_content_sha256": _digest(860 + ordinal),
            "source_byte_count": 100 + ordinal,
        }
        (sidecar_root / f"{logical_identity}.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    manifest = {
        "manifest_version": "mage-codec-cache-manifest-v1",
        "entry_count": 5,
        "built_count": 5,
        "entries": entries,
        "qualified_cache_root": str(cache_root),
        "namespace_identity": _digest(870),
        "checkpoint_manifest_sha256": _digest(871),
        "codec_policy_sha256": _digest(872),
        "prewarm_wall_seconds": 100.0,
        "recipe": {
            "recipe_version": "mage-dcvc-readiness-observed-v1",
            "semantic_sha256": _digest(873),
            "effective_projection": {"max_side": 0},
        },
    }
    (preparation_root / "cache-manifest-v1.json").write_text(json.dumps(manifest), encoding="utf-8")
    (preparation_root / "gpu-telemetry.json").write_text(
        json.dumps(_retained_gpu_document()), encoding="utf-8"
    )

    evidence = module._retained_preparation_evidence(preparation_root, module.BASELINE)

    assert evidence["wall_seconds"] == 100.0
    assert evidence["per_segment_timing_status"] == "NOT_RECORDED"
    assert evidence["provider_process_start_count"] is None
    assert evidence["provider_model_load_count"] is None
    assert evidence["provider_model_load_seconds"] is None
    assert all(item["preparation_seconds"] is None for item in evidence["per_segment"])
    assert all(item["timing_status"] == "NOT_RECORDED" for item in evidence["per_segment"])


def _retained_comparison_variant(
    *, variant_id: str, max_side: int, preparation_wall: float, metadata_seed: int
) -> dict[str, object]:
    visual_assets = [
        {
            "relative_path": "canvas_000.jpg",
            "byte_count": 10,
            "sha256": _digest(900),
        }
    ]
    preparation_segments = [
        {
            "visual_payload_assets": visual_assets,
            "provider_metadata_asset": {
                "relative_path": "meta.json",
                "byte_count": 20,
                "sha256": _digest(metadata_seed + ordinal),
            },
        }
        for ordinal in range(5)
    ]
    generated = [
        {
            "output_text_sha256": _digest(910 + ordinal),
            "normalized_output_semantic_sha256": _digest(920 + ordinal),
            "inference_identity_sha256": _digest(930 + ordinal),
        }
        for ordinal in range(5)
    ]
    downstream = {
        kind: {
            "content_projection_semantic_sha256_values": [_digest(940 + index)],
            "durable_semantic_sha256_values": [_digest(metadata_seed + index)],
        }
        for index, kind in enumerate(("qa", "event", "evidence", "track", "fusion"))
    }
    gpu = {
        "measurement_status": "MEASURED",
        "devices": [
            {
                "memory_used_fraction_max": 0.64,
                "temperature_celsius_max": 72.0,
            }
        ],
        "errors": [],
    }
    sample = {
        "recording_key": "sample",
        "recording_exact_sha256": _digest(950),
        "source_byte_count": 100,
        "media_duration_ns": 40_000_000_000,
        "segments": [_digest(960 + ordinal) for ordinal in range(5)],
    }
    controls = {
        "model_identity_sha256": _digest(970),
        "decoder_identity_sha256_values": [_digest(980 + ordinal) for ordinal in range(5)],
        "max_new_tokens": 256,
        "worker_count": 1,
    }
    return {
        "variant_id": variant_id,
        "max_side": max_side,
        "preparation": {
            "wall_seconds": preparation_wall,
            "provider_process_start_count": 1,
            "provider_model_load_count": 1,
            "startup_load_included_in_wall": True,
            "per_segment_timing_status": "MEASURED",
            "per_segment": preparation_segments,
            "gpu_telemetry": gpu,
        },
        "generation": {
            "sample": sample,
            "controls": controls,
            "measurement": {
                "overall_wall_seconds": 50.0,
                "stream_run_wall_seconds": 20.0,
                "per_segment": generated,
            },
            "downstream": downstream,
            "full_wall_gpu_telemetry": gpu,
        },
    }


def test_retained_comparison_separates_visual_payload_from_provider_metadata() -> None:
    module = _module()
    baseline = _retained_comparison_variant(
        variant_id="observed", max_side=0, preparation_wall=100.0, metadata_seed=1000
    )
    candidate = _retained_comparison_variant(
        variant_id="bounded", max_side=448, preparation_wall=25.0, metadata_seed=1100
    )

    comparison = module._retained_comparison(baseline, candidate)

    assert comparison["locally_adoptable"] is True
    assert comparison["preparation"]["visual_payload_exact_parity_excluding_meta_json"] is True
    assert comparison["preparation"]["provider_metadata_exact_parity"] is False
    assert all(comparison["downstream"]["content_projection_parity"].values())
    assert not any(comparison["downstream"]["durable_identity_parity"].values())
    assert comparison["production_eligible"] is False


def test_retained_cli_rejects_unpaired_bounded_artifact_directories(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "qualification.json"

    exit_code = module.main(
        [
            "--observed-preparation-dir",
            str(tmp_path / "observed-prep"),
            "--observed-generation-dir",
            str(tmp_path / "observed-generation"),
            "--provider-v2-full-preparation-dir",
            str(tmp_path / "full-prep"),
            "--provider-v2-full-generation-dir",
            str(tmp_path / "full-generation"),
            "--provider-v2-bounded-preparation-dir",
            str(tmp_path / "bounded-prep"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()
