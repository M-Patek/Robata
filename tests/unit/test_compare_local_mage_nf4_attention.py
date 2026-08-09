from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "compare_local_mage_nf4_attention.py"
)


def _module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "compare_local_mage_nf4_attention_test", SCRIPT_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _report(
    *,
    attention: str,
    wall_seconds: float,
    generation_seconds: float,
    peak_vram_fraction: float = 0.64,
    output_digest: str | None = None,
) -> dict[str, object]:
    resolved = {
        "top_level": attention,
        "text": attention,
        "vision": attention,
    }
    return {
        "report_version": "mage-nf4-attention-experiment-v1",
        "authority": "NON_AUTHORITATIVE_EXPERIMENT",
        "production_eligible": False,
        "experimental_scope": {
            "attention_override": "PROCESS_LOCAL_PRIVATE_MONKEYPATCH",
            "runtime_identity_binds_attention_backend": False,
            "production_launcher_modified": False,
            "production_adoption_requires_versioned_identity": True,
        },
        "attention_requested": attention,
        "attention_resolved": resolved,
        "attention_resolution_verified": True,
        "runtime_identity": {
            "identity_version": "mage-video-runtime-identity-v1",
            "load_profile": "bitsandbytes_4bit_nf4_v1",
            "attention_backend_bound": False,
        },
        "checkpoint_manifest_sha256": _digest(1),
        "codec_cache_manifest_semantic_sha256": _digest(2),
        "codec_cache_namespace_identity": _digest(3),
        "codec_policy_sha256": _digest(4),
        "prompt_sha256": _digest(5),
        "input_videos": [
            {
                "ordinal": 0,
                "source_path": "D:/samples/segment-0.mp4",
                "source_content_sha256": _digest(6),
                "source_byte_count": 100,
                "logical_cache_identity": _digest(7),
            },
            {
                "ordinal": 1,
                "source_path": "D:/samples/segment-1.mp4",
                "source_content_sha256": _digest(8),
                "source_byte_count": 200,
                "logical_cache_identity": _digest(9),
            },
        ],
        "max_new_tokens": 256,
        "warmup": {
            "max_new_tokens": 32,
            "actual_output_tokens": 12,
            "generation_seconds": 1.0,
            "output_text_sha256": _digest(10),
        },
        "model_load_seconds": 17.0,
        "timed_wall_seconds": wall_seconds,
        "generation_sum_seconds": generation_seconds,
        "results": [
            {
                "ordinal": 0,
                "video_path": "D:/samples/segment-0.mp4",
                "output_text_sha256": output_digest or _digest(11),
                "prompt_tokens": 20,
                "output_tokens": 30,
                "generation_seconds": generation_seconds / 2,
                "total_request_seconds": generation_seconds / 2,
                "time_to_first_token_seconds": 0.2,
                "output_tokens_per_second": 10.0,
            },
            {
                "ordinal": 1,
                "video_path": "D:/samples/segment-1.mp4",
                "output_text_sha256": _digest(12),
                "prompt_tokens": 20,
                "output_tokens": 30,
                "generation_seconds": generation_seconds / 2,
                "total_request_seconds": generation_seconds / 2,
                "time_to_first_token_seconds": 0.2,
                "output_tokens_per_second": 10.0,
            },
        ],
        "gpu_telemetry": {
            "format_version": "robata-nvidia-smi-gpu-telemetry-v1",
            "measurement_status": "MEASURED",
            "summary": [
                {
                    "gpu_index": 0,
                    "gpu_name": "RTX 4060 Laptop GPU",
                    "memory_used_fraction_max": peak_vram_fraction,
                }
            ],
            "samples": [],
            "errors": [],
        },
    }


def _write_pair(
    tmp_path: Path,
    *,
    baseline: dict[str, object] | None = None,
    candidate: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    baseline_path = tmp_path / "eager.json"
    candidate_path = tmp_path / "sdpa.json"
    baseline_path.write_text(
        json.dumps(
            baseline or _report(attention="eager", wall_seconds=10.0, generation_seconds=9.0)
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            candidate or _report(attention="sdpa", wall_seconds=9.0, generation_seconds=8.0)
        ),
        encoding="utf-8",
    )
    return baseline_path, candidate_path


def _gate(payload: dict[str, object], gate_id: str) -> dict[str, object]:
    return next(gate for gate in payload["gates"] if gate["gate_id"] == gate_id)


def test_exact_parity_safe_vram_and_five_percent_speedup_is_only_experiment_adoptable(
    tmp_path: Path,
) -> None:
    module = _module()
    baseline, candidate = _write_pair(tmp_path)

    payload = module.build_comparison_payload(
        baseline_report=baseline,
        candidate_report=candidate,
    )

    assert payload["qualification_status"] == "PASSED"
    assert payload["recommendation"] == "ADOPTABLE_EXPERIMENT"
    assert payload["authority"] == "NON_AUTHORITATIVE_EXPERIMENT"
    assert payload["production_eligible"] is False
    assert payload["production_adoption"].startswith("PROHIBITED_")
    assert payload["thresholds"] == {
        "minimum_speedup_ratio": 1.05,
        "maximum_peak_vram_fraction": 0.85,
    }
    assert all(gate["passed"] for gate in payload["gates"])
    assert len(payload["semantic_sha256"]) == 64


def test_any_output_text_sha_mismatch_blocks_recommendation(tmp_path: Path) -> None:
    module = _module()
    candidate_report = _report(
        attention="sdpa",
        wall_seconds=9.0,
        generation_seconds=8.0,
        output_digest=_digest(99),
    )
    baseline, candidate = _write_pair(tmp_path, candidate=candidate_report)

    payload = module.build_comparison_payload(
        baseline_report=baseline,
        candidate_report=candidate,
    )

    assert payload["recommendation"] == "NOT_ADOPTABLE_EXPERIMENT"
    assert _gate(payload, "OUTPUT_TEXT_SHA_PARITY")["passed"] is False


def test_peak_vram_above_eighty_five_percent_blocks_recommendation(tmp_path: Path) -> None:
    module = _module()
    candidate_report = _report(
        attention="sdpa",
        wall_seconds=9.0,
        generation_seconds=8.0,
        peak_vram_fraction=0.851,
    )
    baseline, candidate = _write_pair(tmp_path, candidate=candidate_report)

    payload = module.build_comparison_payload(
        baseline_report=baseline,
        candidate_report=candidate,
    )

    assert payload["recommendation"] == "NOT_ADOPTABLE_EXPERIMENT"
    assert _gate(payload, "PEAK_VRAM_SAFETY")["passed"] is False


def test_less_than_five_percent_speedup_blocks_recommendation(tmp_path: Path) -> None:
    module = _module()
    candidate_report = _report(
        attention="sdpa",
        wall_seconds=9.6,
        generation_seconds=8.7,
    )
    baseline, candidate = _write_pair(tmp_path, candidate=candidate_report)

    payload = module.build_comparison_payload(
        baseline_report=baseline,
        candidate_report=candidate,
    )

    assert payload["recommendation"] == "NOT_ADOPTABLE_EXPERIMENT"
    assert _gate(payload, "TIMED_WALL_SPEEDUP")["passed"] is False
    assert _gate(payload, "GENERATION_SPEEDUP")["passed"] is False


def test_checkpoint_cache_prompt_video_and_budget_are_independent_fail_closed_gates(
    tmp_path: Path,
) -> None:
    module = _module()
    candidate_report = _report(
        attention="sdpa",
        wall_seconds=9.0,
        generation_seconds=8.0,
    )
    candidate_report["checkpoint_manifest_sha256"] = _digest(101)
    candidate_report["codec_cache_namespace_identity"] = _digest(102)
    candidate_report["prompt_sha256"] = _digest(103)
    candidate_report["input_videos"][0]["source_content_sha256"] = _digest(104)
    candidate_report["max_new_tokens"] = 128
    baseline, candidate = _write_pair(tmp_path, candidate=candidate_report)

    payload = module.build_comparison_payload(
        baseline_report=baseline,
        candidate_report=candidate,
    )

    assert payload["recommendation"] == "NOT_ADOPTABLE_EXPERIMENT"
    for gate_id in (
        "CHECKPOINT_PARITY",
        "CODEC_CACHE_PARITY",
        "PROMPT_PARITY",
        "VIDEO_PARITY",
        "TOKEN_BUDGET_PARITY",
    ):
        assert _gate(payload, gate_id)["passed"] is False


def test_rejects_report_that_claims_production_eligibility(tmp_path: Path) -> None:
    module = _module()
    candidate_report = _report(
        attention="sdpa",
        wall_seconds=9.0,
        generation_seconds=8.0,
    )
    candidate_report["production_eligible"] = True
    baseline, candidate = _write_pair(tmp_path, candidate=candidate_report)

    with pytest.raises(module.MageAttentionComparisonInputError, match="production-ineligible"):
        module.build_comparison_payload(
            baseline_report=baseline,
            candidate_report=candidate,
        )


def test_main_writes_canonical_nonproduction_comparison(tmp_path: Path) -> None:
    module = _module()
    baseline, candidate = _write_pair(tmp_path)
    output = tmp_path / "comparison.json"

    exit_code = module.main(
        [
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recommendation"] == "ADOPTABLE_EXPERIMENT"
    assert payload["production_eligible"] is False
    assert output.read_bytes().endswith(b"}")
