from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# The runner is intentionally reusable from a clean checkout. During this local
# qualification it can point at an isolated source tree through ROBATA_CODE_ROOT.
REPO_ROOT = Path(os.environ.get("ROBATA_CODE_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from robata.adapters.mcap_inspector import OfficialMcapInspector  # noqa: E402
from robata.application.canonical.local_composition import (  # noqa: E402
    LocalCanonicalModelBinding,
    run_local_canonical_mcap,
)
from robata.application.canonical.local_real_model import (  # noqa: E402
    LOCAL_QWEN_ADAPTER_VERSION,
    LOCAL_QWEN_MODEL_NAME,
    LOCAL_QWEN_MODEL_VERSION,
    LOCAL_QWEN_PROVIDER,
    build_local_qwen_model_binding,
)
from robata.application.canonical.local_stream_finalization import (  # noqa: E402
    LocalStreamExecutorConfig,
)
from robata.contracts.hashing import (  # noqa: E402
    canonical_json_bytes,
    exact_bytes_sha256,
)
from robata.inference.local_hf_endpoint import (  # noqa: E402
    LOCAL_HF_CHECKPOINT_MANIFEST_VERSION,
)
from robata.runtime.e2e_participation import (  # noqa: E402
    E2EParticipationBoundary,
    E2EParticipationDeclaration,
    E2EParticipationState,
    build_e2e_participation_manifest,
    write_e2e_participation_manifest,
)
from robata.runtime.e2e_trace import (  # noqa: E402
    E2ETraceFragmentRole,
    build_e2e_trace_runtime_fragment,
)
from robata.runtime.observability import RuntimeProfileRecorder  # noqa: E402

MODEL_PROVIDER = LOCAL_QWEN_PROVIDER
MODEL_NAME = LOCAL_QWEN_MODEL_NAME
MODEL_VERSION = LOCAL_QWEN_MODEL_VERSION


def build_qwen_binding(
    *,
    checkpoint_manifest_sha256: str,
) -> LocalCanonicalModelBinding:
    """Return the checkpoint-bound local Qwen binding used by the CLI entrypoint."""

    return build_local_qwen_model_binding(checkpoint_manifest_sha256=checkpoint_manifest_sha256)


def _write_json(path: Path, payload: object) -> str:
    data = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return exact_bytes_sha256(data)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_endpoint_health(
    payload: object,
    *,
    expected_checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("local Qwen endpoint health must be a JSON object")
    expected = {
        "status": "READY",
        "model_identifier": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "loaded": True,
        "concurrency": 1,
    }
    mismatches: dict[str, dict[str, object]] = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    checkpoint_identity = payload.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, dict):
        mismatches["checkpoint_identity"] = {
            "expected": "checkpoint identity object",
            "actual": checkpoint_identity,
        }
    else:
        for key, value in {
            "manifest_version": LOCAL_HF_CHECKPOINT_MANIFEST_VERSION,
            "manifest_sha256": expected_checkpoint_manifest_sha256,
        }.items():
            if checkpoint_identity.get(key) != value:
                mismatches[f"checkpoint_identity.{key}"] = {
                    "expected": value,
                    "actual": checkpoint_identity.get(key),
                }
        file_count = checkpoint_identity.get("included_file_count")
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count <= 0:
            mismatches["checkpoint_identity.included_file_count"] = {
                "expected": "positive integer",
                "actual": file_count,
            }
    if mismatches:
        raise RuntimeError(f"local Qwen endpoint health mismatch: {mismatches}")
    return payload


def _endpoint_health(*, expected_checkpoint_manifest_sha256: str) -> dict[str, Any]:
    request = Request("http://127.0.0.1:8101/healthz", method="GET")
    with urlopen(request, timeout=5.0) as response:
        payload = json.loads(response.read(64 * 1024))
    return _validate_endpoint_health(
        payload,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
    )


class _GpuTelemetrySampler:
    """Best-effort local NVIDIA telemetry; never canonical evidence."""

    def __init__(self, *, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns = 0
        self._binary = shutil.which("nvidia-smi")

    def start(self) -> None:
        self._started_ns = time.perf_counter_ns()
        if self._binary is None:
            self._errors.append("nvidia-smi not found")
            return
        self._thread = threading.Thread(
            target=self._run,
            name="robata-qwen-gpu-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self._interval_seconds * 4))
            if self._thread.is_alive():
                self._errors.append("GPU telemetry thread did not stop before timeout")
        return self._payload()

    def _run(self) -> None:
        assert self._binary is not None
        command = [
            self._binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            offset_ns = time.perf_counter_ns() - self._started_ns
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or f"exit {completed.returncode}"
                    if len(self._errors) < 20:
                        self._errors.append(f"nvidia-smi: {detail}")
                else:
                    for line in completed.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) != 7:
                            if len(self._errors) < 20:
                                self._errors.append(f"unexpected nvidia-smi row: {line}")
                            continue
                        try:
                            power_milliwatts = round(float(fields[5]) * 1000)
                            self._samples.append(
                                {
                                    "offset_ns": offset_ns,
                                    "gpu_index": int(fields[0]),
                                    "gpu_name": fields[1],
                                    "utilization_gpu_percent": int(fields[2]),
                                    "memory_used_mib": int(fields[3]),
                                    "memory_total_mib": int(fields[4]),
                                    "power_milliwatts": power_milliwatts,
                                    "temperature_celsius": int(fields[6]),
                                }
                            )
                        except ValueError as error:
                            if len(self._errors) < 20:
                                self._errors.append(f"could not parse nvidia-smi row: {error}")
            except (OSError, subprocess.SubprocessError) as error:
                if len(self._errors) < 20:
                    self._errors.append(f"{type(error).__name__}: {error}")
            self._stop.wait(self._interval_seconds)

    def _payload(self) -> dict[str, Any]:
        by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in self._samples:
            by_gpu[int(sample["gpu_index"])].append(sample)
        summary: list[dict[str, Any]] = []
        for index, samples in sorted(by_gpu.items()):
            utilization = [int(item["utilization_gpu_percent"]) for item in samples]
            memory_used = [int(item["memory_used_mib"]) for item in samples]
            memory_total = max(int(item["memory_total_mib"]) for item in samples)
            power = [int(item["power_milliwatts"]) for item in samples]
            temperature = [int(item["temperature_celsius"]) for item in samples]
            summary.append(
                {
                    "gpu_index": index,
                    "gpu_name": str(samples[0]["gpu_name"]),
                    "sample_count": len(samples),
                    "utilization_gpu_percent_max": max(utilization),
                    "utilization_gpu_milli_percent_mean": round(
                        sum(utilization) * 1000 / len(utilization)
                    ),
                    "memory_used_mib_max": max(memory_used),
                    "memory_used_fraction_ppm_max": (
                        0 if memory_total == 0 else max(memory_used) * 1_000_000 // memory_total
                    ),
                    "memory_total_mib": memory_total,
                    "power_milliwatts_max": max(power),
                    "temperature_celsius_max": max(temperature),
                }
            )
        return {
            "format_version": "robata-local-gpu-telemetry-v1",
            "measurement_status": "MEASURED" if self._samples else "NOT_MEASURED",
            "sample_interval_milliseconds": round(self._interval_seconds * 1000),
            "sample_count": len(self._samples),
            "summary": summary,
            "samples": self._samples,
            "errors": self._errors,
        }


def _source_facts(source: Path, state_dir: Path) -> dict[str, Any]:
    """Freeze source facts without treating them as canonical admission."""

    resolved = source.resolve()
    inspection = OfficialMcapInspector().inspect(resolved)
    channels = [asdict(channel) for channel in inspection.channels]
    # RFC 8785 canonical JSON intentionally rejects integers outside the
    # IEEE-754 safe domain. MCAP nanosecond timestamps are much larger, so
    # keep them as decimal strings in the qualification sidecar.
    for channel in channels:
        for key in ("first_message_time_ns", "last_message_time_ns"):
            if channel[key] is not None:
                channel[key] = str(channel[key])
    camera_channels = [
        channel
        for channel in channels
        if channel["schema_name"] == "foxglove.CompressedImage" and "/camera" in channel["topic"]
    ]
    camera_channels.sort(key=lambda channel: channel["topic"])
    media_reports = sorted(
        state_dir.glob("mcap/*/media-quality-report.json"),
        key=lambda path: str(path),
    )
    media: dict[str, Any] | None = None
    if media_reports:
        try:
            payload = json.loads(media_reports[-1].read_text(encoding="utf-8"))
            ledgers = payload.get("camera_ledgers", [])
            frame_counts = {
                str(ledger.get("camera_id")): ledger.get(
                    "timing_count", len(ledger.get("decoded_observations", []))
                )
                for ledger in ledgers
                if isinstance(ledger, dict) and ledger.get("camera_id") is not None
            }
            quality_observation_counts = {
                str(ledger.get("camera_id")): len(ledger.get("decoded_observations", []))
                for ledger in ledgers
                if isinstance(ledger, dict) and ledger.get("camera_id") is not None
            }
            media = {
                "path": str(media_reports[-1]),
                "format_version": payload.get("format_version"),
                "policy_version": payload.get("policy_version"),
                "recording_duration_ns": payload.get("recording_duration_ns"),
                "requested_interval": payload.get("requested_interval"),
                "requested_max_duration_ns": payload.get("requested_max_duration_ns"),
                "window_limited": payload.get("window_limited"),
                "camera_frame_counts": frame_counts,
                "camera_quality_observation_counts": quality_observation_counts,
                "camera_ledger_count": len(ledgers),
                "semantic_sha256": payload.get("semantic_sha256"),
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            media = {"path": str(media_reports[-1]), "read_error": str(error)}

    first = inspection.first_message_time_ns
    last = inspection.last_message_time_ns
    return {
        "path": str(resolved),
        "size_bytes": inspection.source_size_bytes,
        "sha256": inspection.source_sha256,
        "header_profile": inspection.header_profile,
        "header_library": inspection.header_library,
        "summary_available": inspection.summary_available,
        "message_count": inspection.message_count,
        "channel_count": inspection.channel_count,
        "first_message_time_ns": None if first is None else str(first),
        "last_message_time_ns": None if last is None else str(last),
        "native_span_ns": None if first is None or last is None else str(last - first),
        "camera_channel_count": len(camera_channels),
        "camera_channels": camera_channels,
        "media_admission": media,
    }


def _sqlite_audit(state_dir: Path) -> dict[str, Any]:
    """Audit local SQLite stores without changing them."""

    databases: list[dict[str, Any]] = []
    for path in sorted(state_dir.rglob("*.sqlite*"), key=lambda value: str(value)):
        if not path.is_file() or path.name.endswith(("-wal", "-shm")):
            continue
        entry: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "integrity_check": None,
            "foreign_key_violations": None,
            "tables": {},
            "error": None,
        }
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                entry["integrity_check"] = connection.execute("PRAGMA integrity_check").fetchone()[
                    0
                ]
                entry["foreign_key_violations"] = len(
                    connection.execute("PRAGMA foreign_key_check").fetchall()
                )
                table_rows = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
                for (table_name,) in table_rows:
                    quoted = '"' + str(table_name).replace('"', '""') + '"'
                    count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
                    table_info = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
                    table_entry: dict[str, Any] = {"row_count": count}
                    column_names = {str(row[1]) for row in table_info}
                    if "status" in column_names:
                        status_rows = connection.execute(
                            f"SELECT status, COUNT(*) FROM {quoted} GROUP BY status ORDER BY status"
                        ).fetchall()
                        table_entry["status_counts"] = {
                            str(status): status_count for status, status_count in status_rows
                        }
                    entry["tables"][str(table_name)] = table_entry
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        databases.append(entry)

    cas_root = state_dir / "raw-provider-cas"
    cas_files = (
        [path for path in cas_root.rglob("*") if path.is_file()] if cas_root.exists() else []
    )
    cas_mismatches: list[dict[str, Any]] = []
    cas_bytes = 0
    for path in cas_files:
        cas_bytes += path.stat().st_size
        expected = path.name.lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            continue
        actual = _file_sha256(path)
        if actual != expected:
            cas_mismatches.append({"path": str(path), "expected": expected, "actual": actual})
    return {
        "sqlite": databases,
        "sqlite_database_count": len(databases),
        "sqlite_integrity_failures": sum(
            1
            for item in databases
            if item["error"] is not None
            or item["integrity_check"] != "ok"
            or item["foreign_key_violations"]
        ),
        "raw_provider_cas": {
            "root": str(cas_root),
            "file_count": len(cas_files),
            "bytes": cas_bytes,
            "sha256_mismatch_count": len(cas_mismatches),
            "sha256_mismatches": cas_mismatches[:20],
        },
    }


def _nearest_rank(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[min(rank - 1, len(ordered) - 1)]


def _inference_audit(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "inference-evidence.sqlite3"
    if not path.is_file():
        return {"measurement_status": "NOT_MEASURED", "path": str(path)}
    result: dict[str, Any] = {
        "measurement_status": "MEASURED",
        "path": str(path),
    }
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            lineage_tables = (
                "inference_intents",
                "model_inference_terminals",
                "raw_provider_artifacts",
                "raw_provider_responses",
                "parsed_provider_claims",
                "inference_attempt_selections",
                "selected_attempt_outputs",
                "enriched_provider_outputs",
            )
            lineage_counts = {
                table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                for table in lineage_tables
            }
            terminal_groups: Counter[tuple[str, str, str, str, str, bool]] = Counter()
            latencies: dict[str, list[int]] = defaultdict(list)
            for (payload_json,) in connection.execute(
                "SELECT payload_json FROM model_inference_terminals"
            ):
                payload = json.loads(payload_json)
                stage = str(payload.get("stage"))
                terminal_groups[
                    (
                        stage,
                        str(payload.get("provider")),
                        str(payload.get("model_name")),
                        str(payload.get("model_version")),
                        str(payload.get("status")),
                        bool(payload.get("output_valid")),
                    )
                ] += 1
                latency_ms = payload.get("latency_ms")
                if isinstance(latency_ms, int) and not isinstance(latency_ms, bool):
                    latencies[stage].append(latency_ms)

            observation_counts: dict[str, Counter[str]] = defaultdict(Counter)
            dense_inputs: dict[str, list[str]] = defaultdict(list)
            for (payload_json,) in connection.execute(
                "SELECT payload_json FROM enriched_provider_outputs"
            ):
                payload = json.loads(payload_json)
                task = str(payload.get("task"))
                for claim in payload.get("claims", []):
                    observation = claim.get("observation")
                    if observation is None:
                        continue
                    observation_text = str(observation)
                    observation_counts[task][observation_text] += 1
                    if task == "QA_DENSE":
                        coordinate = f"{claim.get('package_ordinal')}:{claim.get('camera_id')}"
                        dense_inputs[coordinate].append(observation_text)

            severity = {"GOOD": 0, "DEGRADED": 1, "UNKNOWN": 2, "UNUSABLE": 3}
            dense_reduced = {
                coordinate: max(values, key=lambda value: severity.get(value, 4))
                for coordinate, values in sorted(dense_inputs.items())
            }
            terminal_count = lineage_counts["model_inference_terminals"]
            parser_versions = {
                str(version): count
                for version, count in connection.execute(
                    "SELECT parser_version, COUNT(*) FROM parsed_provider_claims "
                    "GROUP BY parser_version ORDER BY parser_version"
                )
            }
            raw_media_types = {
                str(media_type): count
                for media_type, count in connection.execute(
                    "SELECT media_type, COUNT(*) FROM raw_provider_artifacts "
                    "GROUP BY media_type ORDER BY media_type"
                )
            }
            result.update(
                {
                    "lineage_counts": lineage_counts,
                    "parsed_claim_parser_versions": parser_versions,
                    "raw_provider_media_types": raw_media_types,
                    "lineage_complete": terminal_count > 0
                    and all(lineage_counts[table] == terminal_count for table in lineage_tables),
                    "terminal_groups": [
                        {
                            "stage": key[0],
                            "provider": key[1],
                            "model_name": key[2],
                            "model_version": key[3],
                            "status": key[4],
                            "output_valid": key[5],
                            "count": count,
                        }
                        for key, count in sorted(terminal_groups.items())
                    ],
                    "provider_latency_ms": {
                        stage: {
                            "count": len(values),
                            "min": min(values),
                            "p50": _nearest_rank(values, 50),
                            "p95": _nearest_rank(values, 95),
                            "max": max(values),
                            "sum": sum(values),
                            "mean_milli_ms": round(sum(values) * 1000 / len(values)),
                        }
                        for stage, values in sorted(latencies.items())
                        if values
                    },
                    "observation_counts": {
                        task: dict(sorted(counts.items()))
                        for task, counts in sorted(observation_counts.items())
                    },
                    "dense_coordinate_inputs": {
                        coordinate: values for coordinate, values in sorted(dense_inputs.items())
                    },
                    "dense_coordinate_reduced": dense_reduced,
                    "dense_unresolved_coordinates": [
                        coordinate
                        for coordinate, observation in dense_reduced.items()
                        if observation != "GOOD"
                    ],
                }
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as error:
        result["measurement_status"] = "ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _pipeline_audit(state_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scheduler_path = state_dir / "work-scheduler.sqlite3"
    if scheduler_path.is_file():
        try:
            connection = sqlite3.connect(f"file:{scheduler_path.as_posix()}?mode=ro", uri=True)
            try:
                result["scheduler"] = {
                    "work_item_states": {
                        str(state): count
                        for state, count in connection.execute(
                            "SELECT state, COUNT(*) FROM work_items GROUP BY state ORDER BY state"
                        )
                    },
                    "work_item_stage_states": [
                        {"stage": stage, "state": state, "count": count}
                        for stage, state, count in connection.execute(
                            "SELECT stage, state, COUNT(*) FROM work_items "
                            "GROUP BY stage, state ORDER BY stage, state"
                        )
                    ],
                    "attempt_outcomes": {
                        str(outcome): count
                        for outcome, count in connection.execute(
                            "SELECT outcome, COUNT(*) FROM work_attempts "
                            "GROUP BY outcome ORDER BY outcome"
                        )
                    },
                    "expected_windows": connection.execute(
                        "SELECT COUNT(*) FROM expected_windows"
                    ).fetchone()[0],
                    "stream_window_results": connection.execute(
                        "SELECT COUNT(*) FROM stream_window_results"
                    ).fetchone()[0],
                    "stream_window_evidence_commits": connection.execute(
                        "SELECT COUNT(*) FROM stream_window_evidence_commits"
                    ).fetchone()[0],
                    "stream_delivery_outbox": connection.execute(
                        "SELECT COUNT(*) FROM stream_delivery_outbox"
                    ).fetchone()[0],
                    "stream_outbox_delivery_statuses": {
                        str(status): count
                        for status, count in connection.execute(
                            "SELECT status, COUNT(*) FROM stream_outbox_deliveries "
                            "GROUP BY status ORDER BY status"
                        )
                    },
                    "recording_finalizations": connection.execute(
                        "SELECT COUNT(*) FROM recording_finalizations"
                    ).fetchone()[0],
                }
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as error:
            result["scheduler"] = {"error": f"{type(error).__name__}: {error}"}

    stream_root = state_dir / "stream-artifacts"
    if stream_root.is_dir():
        artifact_files = [path for path in stream_root.rglob("*.json") if path.is_file()]
        fixture_intents = 0
        fixture_accepted = 0
        real_qwen_artifacts = 0
        for path in artifact_files:
            try:
                payload = path.read_bytes()
                document = json.loads(payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if b"Qwen3-VL-4B-Instruct" in payload or b"local-huggingface" in payload:
                real_qwen_artifacts += 1
            if b"local-conformance-window-mock-v1" not in payload:
                continue
            schema_id = (
                document.get("schema_ref", {}).get("schema_id")
                if isinstance(document, dict)
                else None
            )
            if schema_id == "https://schemas.robata.dev/stream-inference-intent":
                fixture_intents += 1
            if schema_id == "https://schemas.robata.dev/stream-accepted-call-evidence":
                fixture_accepted += 1
        result.setdefault("scheduler", {}).update(
            {
                "stream_artifact_count": len(artifact_files),
                "fixture_intent_artifacts": fixture_intents,
                "fixture_accepted_call_artifacts": fixture_accepted,
                "real_qwen_stream_artifacts": real_qwen_artifacts,
            }
        )

    completion_path = state_dir / "primary-completion.sqlite3"
    if completion_path.is_file():
        try:
            connection = sqlite3.connect(f"file:{completion_path.as_posix()}?mode=ro", uri=True)
            try:
                primary_runs = connection.execute(
                    "SELECT primary_status, COUNT(*) FROM primary_runs "
                    "GROUP BY primary_status ORDER BY primary_status"
                ).fetchall()
                result["completion"] = {
                    "primary_run_statuses": {str(status): count for status, count in primary_runs},
                    "primary_completions": connection.execute(
                        "SELECT COUNT(*) FROM primary_completions"
                    ).fetchone()[0],
                    "detailed_results": connection.execute(
                        "SELECT COUNT(*) FROM detailed_results"
                    ).fetchone()[0],
                    "primary_outbox": connection.execute(
                        "SELECT COUNT(*) FROM primary_outbox"
                    ).fetchone()[0],
                    "primary_outbox_deliveries": connection.execute(
                        "SELECT COUNT(*) FROM primary_outbox_deliveries"
                    ).fetchone()[0],
                }
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as error:
            result["completion"] = {"error": f"{type(error).__name__}: {error}"}
    return result


def _scheduler_succeeded(scheduler: dict[str, Any]) -> bool:
    states = scheduler.get("work_item_states")
    attempts = scheduler.get("attempt_outcomes")
    expected_windows = scheduler.get("expected_windows")
    if not isinstance(states, dict) or not states or set(states) != {"SUCCEEDED"}:
        return False
    if not isinstance(attempts, dict) or not attempts or set(attempts) != {"SUCCEEDED"}:
        return False
    if not isinstance(expected_windows, int) or expected_windows < 1:
        return False
    return (
        scheduler.get("stream_window_results") == expected_windows
        and scheduler.get("stream_window_evidence_commits") == expected_windows
    )


def _source_admitted(source: dict[str, Any]) -> bool:
    admission = source.get("media_admission")
    return (
        isinstance(admission, dict)
        and admission.get("camera_ledger_count") == 6
        and isinstance(admission.get("semantic_sha256"), str)
        and bool(admission.get("path"))
    )


def _semantic_stage_outcomes(
    *,
    error: BaseException | None,
    failure_boundary: E2EParticipationBoundary | None,
    source: dict[str, Any],
    inference: dict[str, Any],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    scheduler = pipeline.get("scheduler", {})
    completion = pipeline.get("completion", {})
    unresolved = inference.get("dense_unresolved_coordinates", [])
    source_admitted = _source_admitted(source)
    lineage_complete = inference.get("lineage_complete") is True
    if unresolved:
        reduction_status = "FAILED_QUALITY_GATE"
    elif error is None and lineage_complete:
        reduction_status = "SUCCEEDED"
    elif failure_boundary is E2EParticipationBoundary.REDUCTION:
        reduction_status = "FAILED"
    else:
        reduction_status = "NOT_REACHED_OR_UNPROVEN"
    return {
        "source": {
            "status": (
                "ADMITTED"
                if source_admitted
                else "FAILED"
                if failure_boundary is E2EParticipationBoundary.SOURCE
                else "NOT_ADMITTED_OR_UNPROVEN"
            ),
            "basis": (
                "persisted media-quality-report with six camera ledgers"
                if source_admitted
                else "no persisted six-camera media admission was observed"
            ),
        },
        "scheduling": {
            "status": "SUCCEEDED" if _scheduler_succeeded(scheduler) else "INCOMPLETE",
            "work_item_states": scheduler.get("work_item_states"),
            "attempt_outcomes": scheduler.get("attempt_outcomes"),
        },
        "inference": {
            "status": "SUCCEEDED" if lineage_complete else "INCOMPLETE",
            "terminal_groups": inference.get("terminal_groups"),
        },
        "evidence": {
            "status": "SUCCEEDED" if lineage_complete else "INCOMPLETE",
            "lineage_counts": inference.get("lineage_counts"),
        },
        "reduction": {
            "status": reduction_status,
            "failure_boundary": None if failure_boundary is None else failure_boundary.value,
            "unresolved_dense_coordinates": unresolved,
        },
        "publication": {
            "status": (
                "COMMITTED" if completion.get("primary_completions", 0) > 0 else "NOT_COMMITTED"
            ),
            "primary_run_statuses": completion.get("primary_run_statuses"),
            "primary_completions": completion.get("primary_completions"),
            "primary_outbox": completion.get("primary_outbox"),
        },
        "stream_experiment_route": {
            "status": (
                "FIXTURE_ONLY"
                if scheduler.get("fixture_accepted_call_artifacts", 0)
                == scheduler.get("stream_window_results", -1)
                and scheduler.get("real_qwen_stream_artifacts", -1) == 0
                else "MIXED_OR_REAL"
            ),
            "fixture_intent_artifacts": scheduler.get("fixture_intent_artifacts"),
            "fixture_accepted_call_artifacts": scheduler.get("fixture_accepted_call_artifacts"),
            "real_qwen_stream_artifacts": scheduler.get("real_qwen_stream_artifacts"),
            "stream_window_results": scheduler.get("stream_window_results"),
        },
    }


def _canonical_status(error: BaseException | None) -> str | None:
    if error is None:
        return None
    match = re.search(r"canonical run ended as ([A-Z_]+)", str(error))
    return match.group(1) if match else None


def _error_code(error: BaseException | None) -> str | None:
    if error is None:
        return None
    value = getattr(error, "code", None)
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _failure_boundary(
    *,
    error: BaseException | None,
    error_phase: str | None,
    fragment: Any,
) -> E2EParticipationBoundary | None:
    status = _canonical_status(error)
    by_status = {
        "INFERENCE_FAILED": E2EParticipationBoundary.INFERENCE,
        "INCOMPLETE": E2EParticipationBoundary.REDUCTION,
        "QA_INCOMPLETE": E2EParticipationBoundary.REDUCTION,
        "SOURCE_FAILED": E2EParticipationBoundary.SOURCE,
        "SCHEDULING_FAILED": E2EParticipationBoundary.SCHEDULING,
        "EVIDENCE_FAILED": E2EParticipationBoundary.EVIDENCE,
        "PUBLICATION_FAILED": E2EParticipationBoundary.PUBLICATION,
    }
    if status in by_status:
        return by_status[status]
    by_code = {
        "INVALID_REQUEST": E2EParticipationBoundary.ORCHESTRATION,
        "SOURCE_INVALID": E2EParticipationBoundary.SOURCE,
        "BACKPRESSURE": E2EParticipationBoundary.SCHEDULING,
        "RUN_NOT_COMPLETABLE": E2EParticipationBoundary.REDUCTION,
        "COMPLETION_FAILED": E2EParticipationBoundary.PUBLICATION,
    }
    code = _error_code(error)
    if code in by_code:
        return by_code[code]
    if error is None:
        return None
    if error_phase in {"BINDING", "ENDPOINT_PREFLIGHT"}:
        return E2EParticipationBoundary.ORCHESTRATION
    observed = [
        boundary
        for boundary, stage in zip(E2EParticipationBoundary, fragment.stages, strict=True)
        if stage.observed_span_count > 0
    ]
    return observed[-1] if observed else E2EParticipationBoundary.ORCHESTRATION


def _declarations(
    *,
    error: BaseException | None,
    error_phase: str | None,
    fragment: Any,
) -> tuple[E2EParticipationDeclaration, ...]:
    """Declare the intended path while marking an observed stop honestly."""

    status = _canonical_status(error) or _error_code(error) or type(error).__name__
    failed_boundary = _failure_boundary(
        error=error,
        error_phase=error_phase,
        fragment=fragment,
    )
    failed_index = (
        tuple(E2EParticipationBoundary).index(failed_boundary)
        if failed_boundary is not None
        else None
    )
    declarations: list[E2EParticipationDeclaration] = []
    for index, boundary in enumerate(E2EParticipationBoundary):
        if error is None:
            declarations.append(
                E2EParticipationDeclaration(
                    boundary=boundary, state=E2EParticipationState.PARTICIPATING
                )
            )
            continue
        if failed_boundary is not None and boundary is failed_boundary:
            declarations.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=E2EParticipationState.FAILED,
                    required=True,
                    reason=f"{status}: {error}",
                )
            )
            continue
        stage = fragment.stages[index]
        if failed_index is not None and index > failed_index and stage.observed_span_count == 0:
            declarations.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=E2EParticipationState.BYPASSED,
                    required=False,
                    reason=f"not reached after {status} at {failed_boundary.value}",
                )
            )
            continue
        declarations.append(
            E2EParticipationDeclaration(
                boundary=boundary, state=E2EParticipationState.PARTICIPATING
            )
        )
    return tuple(declarations)


def _checkpoint_manifest_sha256(value: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError("must be a lowercase or uppercase SHA-256 hex digest")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--mapping-config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-key", default="qwen-control-20260806")
    parser.add_argument("--max-duration-seconds", type=int, default=41)
    parser.add_argument("--allow-unapproved-profile", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256",
        type=_checkpoint_manifest_sha256,
        default=os.environ.get("ROBATA_QWEN_CHECKPOINT_MANIFEST_SHA256"),
        help=(
            "pinned SHA-256 from the independently generated local checkpoint manifest; "
            "or set ROBATA_QWEN_CHECKPOINT_MANIFEST_SHA256"
        ),
    )
    parser.add_argument("--gpu-sample-interval-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_checkpoint_manifest_sha256 is None:
        raise ValueError(
            "--expected-checkpoint-manifest-sha256 or "
            "ROBATA_QWEN_CHECKPOINT_MANIFEST_SHA256 is required"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = RuntimeProfileRecorder()
    binding: LocalCanonicalModelBinding | None = None
    receipt = None
    error: BaseException | None = None
    error_phase: str | None = None
    source_error: BaseException | None = None
    endpoint_health: dict[str, Any] | None = None
    source: dict[str, Any] = {
        "path": str(args.source.resolve()),
        "measurement_status": "NOT_MEASURED",
    }
    if args.gpu_sample_interval_seconds <= 0:
        raise ValueError("--gpu-sample-interval-seconds must be positive")
    gpu_sampler = _GpuTelemetrySampler(interval_seconds=args.gpu_sample_interval_seconds)
    gpu_sampler.start()
    try:
        error_phase = "BINDING"
        binding = build_qwen_binding(
            checkpoint_manifest_sha256=args.expected_checkpoint_manifest_sha256
        )
        error_phase = "ENDPOINT_PREFLIGHT"
        endpoint_health = _endpoint_health(
            expected_checkpoint_manifest_sha256=args.expected_checkpoint_manifest_sha256
        )
        error_phase = "SOURCE_INSPECTION"
        try:
            source = _source_facts(args.source, args.state_dir)
        except Exception as source_exception:
            source_error = source_exception
            source = {
                "path": str(args.source.resolve()),
                "error": f"{type(source_exception).__name__}: {source_exception}",
            }
        error_phase = "CANONICAL_RUN"
        receipt = run_local_canonical_mcap(
            source_path=args.source,
            mapping_config=args.mapping_config,
            state_dir=args.state_dir,
            run_key=args.run_key,
            allow_unapproved_profile=args.allow_unapproved_profile,
            max_duration_ns=args.max_duration_seconds * 1_000_000_000,
            runtime_observer=recorder,
            model_binding=binding,
            executor_config=LocalStreamExecutorConfig(max_concurrency=1),
        )
        error_phase = None
    except Exception as run_exception:
        error = run_exception
        payload = {
            "ok": False,
            "error_type": type(run_exception).__name__,
            "detail": str(run_exception),
            "canonical_status": _canonical_status(run_exception),
            "error_code": _error_code(run_exception),
            "error_phase": error_phase,
        }
        _write_json(args.output_dir / "error.json", payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)

    gpu_telemetry = gpu_sampler.stop()
    try:
        source = _source_facts(args.source, args.state_dir)
        source_error = None
    except Exception as source_exception:
        source_error = source_exception
        if source.get("measurement_status") == "NOT_MEASURED":
            source = {
                "path": str(args.source.resolve()),
                "error": f"{type(source_exception).__name__}: {source_exception}",
            }
    gpu_telemetry_path = args.output_dir / "gpu-telemetry.json"
    gpu_telemetry_digest = _write_json(gpu_telemetry_path, gpu_telemetry)

    fragment = build_e2e_trace_runtime_fragment(
        role=E2ETraceFragmentRole.CONTROL,
        runtime_profile=recorder.snapshot(),
    )
    trace_path = args.output_dir / "trace.json"
    trace_digest = _write_json(trace_path, fragment.model_dump(mode="json"))
    participation = build_e2e_participation_manifest(
        runtime_fragment=fragment,
        declarations=_declarations(
            error=error,
            error_phase=error_phase,
            fragment=fragment,
        ),
        trace_digest=trace_digest,
    )
    participation_path = args.output_dir / "participation.json"
    participation_digest = write_e2e_participation_manifest(participation, participation_path)

    if source_error is not None:
        source["collection_error"] = f"{type(source_error).__name__}: {source_error}"
    storage = _sqlite_audit(args.state_dir)
    inference_audit = _inference_audit(args.state_dir)
    pipeline_audit = _pipeline_audit(args.state_dir)
    failure_boundary = _failure_boundary(
        error=error,
        error_phase=error_phase,
        fragment=fragment,
    )
    stage_outcomes = _semantic_stage_outcomes(
        error=error,
        failure_boundary=failure_boundary,
        source=source,
        inference=inference_audit,
        pipeline=pipeline_audit,
    )
    canonical_status = _canonical_status(error)
    report = {
        "format_version": "robata-local-qwen-canonical-e2e-v3",
        "audit_revision": "persisted-stage-evidence-and-stream-schema-scan-v3",
        "ok": error is None,
        "status": "SUCCEEDED" if error is None else (canonical_status or "FAILED"),
        "error": (
            None
            if error is None
            else {
                "error_type": type(error).__name__,
                "detail": str(error),
                "canonical_status": canonical_status,
                "error_code": _error_code(error),
                "error_phase": error_phase,
                "failure_boundary": (None if failure_boundary is None else failure_boundary.value),
            }
        ),
        "model": {
            "provider": MODEL_PROVIDER,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "adapter_version": LOCAL_QWEN_ADAPTER_VERSION,
            "endpoint": "http://127.0.0.1:8101",
            "worker_concurrency": 1,
            "execution_mode": "single-worker-production-policy-local-conformance",
            "health_preflight": endpoint_health,
            "expected_checkpoint_manifest_sha256": args.expected_checkpoint_manifest_sha256,
        },
        "source": {
            **source,
            "requested_max_duration_seconds": args.max_duration_seconds,
            "allow_unapproved_profile": args.allow_unapproved_profile,
        },
        "receipt": None if receipt is None else receipt.model_dump(mode="json"),
        "storage_audit": storage,
        "inference_audit": inference_audit,
        "pipeline_audit": pipeline_audit,
        "semantic_stage_outcomes": stage_outcomes,
        "runtime": {
            "trace_path": str(trace_path),
            "trace_sha256": trace_digest,
            "participation_path": str(participation_path),
            "participation_sha256": participation_digest,
            "participation_coverage": participation.coverage.value,
            "stage_measurements": [stage.model_dump(mode="json") for stage in fragment.stages],
            "unclassified_span_count": fragment.unclassified_span_count,
            "runtime_elapsed_ns": fragment.runtime_profile.elapsed_ns,
            "process_cpu_ns": fragment.runtime_profile.process_cpu_ns,
            "resource_snapshot": fragment.runtime_profile.resources.model_dump(mode="json"),
            "gpu_telemetry_path": str(gpu_telemetry_path),
            "gpu_telemetry_sha256": gpu_telemetry_digest,
            "gpu_telemetry_summary": gpu_telemetry["summary"],
            "gpu_telemetry_sample_count": gpu_telemetry["sample_count"],
            "gpu_telemetry_errors": gpu_telemetry["errors"],
        },
        "evidence_class": "LOCAL_CONFORMANCE",
        "production_eligible": False,
        "canonical_authority": False,
    }
    report_path = args.output_dir / "report.json"
    report_digest = _write_json(report_path, report)
    summary = {
        "ok": error is None,
        "status": report["status"],
        "report": str(report_path),
        "report_sha256": report_digest,
        "trace": str(trace_path),
        "participation": str(participation_path),
        "gpu_telemetry": str(gpu_telemetry_path),
        "coverage": participation.coverage.value,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
