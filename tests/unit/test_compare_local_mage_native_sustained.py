from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "compare_local_mage_native_sustained.py"
    )
    spec = importlib.util.spec_from_file_location("compare_local_mage_native_sustained", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _report(*, profile: str, wall: float, run_key: str) -> dict[str, object]:
    contexts = [
        {
            "context_manifest_key": f"context-{ordinal}",
            "focus_segment_ordinal": ordinal,
        }
        for ordinal in range(2)
    ]
    return {
        "selected_camera": "cam_01",
        "plan": {
            "plan_semantic_sha256": _digest(4),
            "recording": {
                "recording_exact_sha256": _digest(3),
                "interval": {"start_ns": 0, "end_ns": 16_000_000_000},
            },
        },
        "execution": {
            "execution_profile": profile,
            "execution_timing": {"run_wall_seconds": wall},
            "endpoint": {
                "model_identity": {"checkpoint_manifest_sha256": _digest(2)},
            },
            "contexts": contexts,
            "durable_execution": {"run": {"run_key": run_key}},
            "run_manifest": {"logical_key": f"manifest:{run_key}"},
            "gpu_telemetry": {"measurement_status": "MEASURED"},
        },
    }


def _events(*, arm: str, generation: tuple[tuple[float, float], ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal, (start, end) in enumerate(generation):
        request_start = start - 0.05
        rows.append(
            {
                "event_version": "mage-video-generation-telemetry-event-v3",
                "request_id": f"request-{ordinal}",
                "context_id": f"context-{ordinal}",
                "inference_identity_sha256": _digest(100 + ordinal),
                "result_artifact_identity": _digest((200 if arm == "serial" else 300) + ordinal),
                "model_identity_sha256": _digest(1),
                "codec_policy_sha256": _digest(5),
                "decoder_identity_sha256": _digest(6),
                "prompt_tokens": 20,
                "output_tokens": 20,
                "max_new_tokens": 256,
                "model_load_seconds": 17.11,
                "model_load_included_in_run_wall": False,
                "runtime_telemetry": {
                    "request_started_monotonic_seconds": 100.0 + request_start,
                    "request_completed_monotonic_seconds": 100.0 + end + 0.05,
                    "processor_started_monotonic_seconds": 100.0 + request_start,
                    "processor_completed_monotonic_seconds": 100.0 + start,
                    "generation_started_monotonic_seconds": 100.0 + start,
                    "generation_completed_monotonic_seconds": 100.0 + end,
                    "decode_started_monotonic_seconds": 100.0 + end,
                    "decode_completed_monotonic_seconds": 100.0 + end + 0.05,
                    "time_to_first_token_seconds": 0.1,
                },
            }
        )
    return rows


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    serial_report = tmp_path / "serial-report.json"
    prefetch_report = tmp_path / "prefetch-report.json"
    serial_telemetry = tmp_path / "serial.jsonl"
    prefetch_telemetry = tmp_path / "prefetch.jsonl"
    serial_report.write_text(
        json.dumps(_report(profile="SERIAL_NATIVE_V1", wall=3.0, run_key="serial")),
        encoding="utf-8",
    )
    prefetch_report.write_text(
        json.dumps(
            _report(
                profile="BOUNDED_PREFETCH_NATIVE_V1",
                wall=2.0,
                run_key="prefetch",
            )
        ),
        encoding="utf-8",
    )
    serial_events = _events(arm="serial", generation=((0.1, 1.1), (1.5, 2.5)))
    prefetch_events = _events(arm="prefetch", generation=((0.0, 0.9), (1.0, 1.9)))
    result_dir = tmp_path / "endpoint-results"
    result_dir.mkdir()
    for rows in (serial_events, prefetch_events):
        for ordinal, row in enumerate(rows):
            artifact = {
                "artifact_identity": row["result_artifact_identity"],
                "request_id": row["request_id"],
                "output_tokens": row["output_tokens"],
                "output_text": f"deterministic-output-{ordinal}",
            }
            artifact_bytes = (
                json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            row["result_artifact_exact_sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
            (result_dir / f"{row['result_artifact_identity']}.json").write_bytes(artifact_bytes)
    serial_telemetry.write_text(
        "\n".join(json.dumps(row) for row in serial_events) + "\n",
        encoding="utf-8",
    )
    prefetch_telemetry.write_text(
        "\n".join(json.dumps(row) for row in prefetch_events) + "\n",
        encoding="utf-8",
    )
    return serial_report, serial_telemetry, prefetch_report, prefetch_telemetry


def test_builds_passing_serial_vs_prefetch_payload(tmp_path: Path) -> None:
    module = _module()
    serial_report, serial_telemetry, prefetch_report, prefetch_telemetry = _write_inputs(tmp_path)

    payload = module.build_comparison_payload(
        serial_report=serial_report,
        serial_telemetry=serial_telemetry,
        prefetch_report=prefetch_report,
        prefetch_telemetry=prefetch_telemetry,
    )

    comparison = payload["comparison"]
    assert comparison["qualification_status"] == "PASSED"
    assert comparison["prefetch_speedup"] == 1.5
    assert comparison["cross_run_isolation"]["overlapping_request_id_count"] == 2
    assert comparison["cross_run_isolation"]["passed"] is True
    assert (
        next(gate for gate in comparison["gates"] if gate["gate_id"] == "OUTPUT_TEXT_HASH_PARITY")[
            "passed"
        ]
        is True
    )
    assert comparison["serial_summary"]["time_to_first_token"]["p95_seconds"] == 0.1
    assert payload["gpu_telemetry"]["serial"]["measurement_status"] == "MEASURED"
    assert len(payload["semantic_sha256"]) == 64


def test_main_writes_canonical_output(tmp_path: Path) -> None:
    module = _module()
    serial_report, serial_telemetry, prefetch_report, prefetch_telemetry = _write_inputs(tmp_path)
    output = tmp_path / "comparison.json"

    exit_code = module.main(
        [
            "--serial-report",
            str(serial_report),
            "--serial-telemetry",
            str(serial_telemetry),
            "--prefetch-report",
            str(prefetch_report),
            "--prefetch-telemetry",
            str(prefetch_telemetry),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["comparison"]["qualification_status"]
        == "PASSED"
    )
