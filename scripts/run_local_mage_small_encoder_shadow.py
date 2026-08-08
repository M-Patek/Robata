"""Run a sequential Mage-native versus Mage-compatible encoder-lite shadow A/B.

The command intentionally keeps one resident model, one camera, one generation in
flight, and five non-overlapping segments from the existing qualification sample.
The candidate only receives Mage merger-space tensors and is never publication
authority. It writes a plain experimental report (not a published wire schema).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAGE_SMALL_ENCODER_REPORT_VERSION = "mage-small-encoder-shadow-report-v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _seal_report(report: dict[str, object]) -> dict[str, object]:
    if "report_sha256" in report:
        raise ValueError("report is already sealed")
    sealed = dict(report)
    sealed["report_sha256"] = _canonical_sha256(sealed)
    return sealed


def _reset_cuda_peak_memory(torch_module: Any, device: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_mib(torch_module: Any, device: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    torch_module.cuda.synchronize()
    return float(torch_module.cuda.max_memory_allocated(device) / (1024 * 1024))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_identity() -> dict[str, object]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "accelerate": _package_version("accelerate"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _git_and_source_identity(repo_root: Path) -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    source_files = {
        "runner": Path(__file__).resolve(),
        "encoder": repo_root / "src" / "robata" / "inference" / "mage_small_encoder.py",
        "evaluator": repo_root / "src" / "robata" / "benchmark" / "mage_small_encoder.py",
        "adapter": repo_root / "src" / "robata" / "inference" / "mage_video_adapter.py",
    }
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain", "--untracked-files=all"),
        "source_sha256": {
            name: _sha256_file(path) for name, path in source_files.items() if path.is_file()
        },
    }


def _verified_checkpoint_identity(*, model_dir: Path, manifest_path: Path) -> tuple[Any, str]:
    from robata.inference.mage_checkpoint_identity import (
        load_mage_checkpoint_manifest,
        verify_mage_checkpoint_manifest,
    )

    manifest = load_mage_checkpoint_manifest(manifest_path=manifest_path)
    verify_mage_checkpoint_manifest(manifest=manifest, model_directory=model_dir)
    if manifest.model_identifier != "Mage-VL":
        raise ValueError(f"unexpected checkpoint model identifier: {manifest.model_identifier}")
    return manifest, _sha256_file(manifest_path)


def _gpu_sample() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        row = result.stdout.strip().splitlines()[0].split(",")
        if len(row) < 5:
            return None
        return {
            "utilization_percent": float(row[0].strip()),
            "memory_used_mib": float(row[1].strip()),
            "memory_total_mib": float(row[2].strip()),
            "power_watts": float(row[3].strip()),
            "temperature_celsius": float(row[4].strip()),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class _GpuSampler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._samples: list[dict[str, float | str]] = []
        self._thread: threading.Thread | None = None
        self._phase = "IDLE"
        self._phase_lock = threading.Lock()

    def set_phase(self, value: str) -> None:
        with self._phase_lock:
            self._phase = value

    def start(self) -> None:
        def collect() -> None:
            while not self._stop.is_set():
                sample = _gpu_sample()
                if sample is not None:
                    with self._phase_lock:
                        phase = self._phase
                    self._samples.append({**sample, "phase": phase})
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=collect, name="mage-shadow-gpu", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float | str]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return list(self._samples)


def _aggregate_gpu(samples: list[dict[str, float | str]]) -> dict[str, object]:
    if not samples:
        return {"status": "NOT_MEASURED", "sample_count": 0}

    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    util = values("utilization_percent")
    memory = values("memory_used_mib")
    power = values("power_watts")
    temperature = values("temperature_celsius")
    return {
        "status": "MEASURED",
        "sample_count": len(samples),
        "utilization_percent_mean": sum(util) / len(util),
        "utilization_percent_max": max(util),
        "memory_used_mib_mean": sum(memory) / len(memory),
        "memory_used_mib_max": max(memory),
        "memory_total_mib": max(values("memory_total_mib")),
        "power_watts_mean": sum(power) / len(power),
        "power_watts_max": max(power),
        "temperature_celsius_max": max(temperature),
    }


def _aggregate_gpu_by_phase(
    samples: list[dict[str, float | str]],
) -> dict[str, dict[str, object]]:
    phases = sorted({str(sample["phase"]) for sample in samples if sample.get("phase") != "IDLE"})
    return {
        phase: _aggregate_gpu([sample for sample in samples if sample.get("phase") == phase])
        for phase in phases
    }


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
            "population_stdev": None,
        }
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "sum": sum(values),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "population_stdev": statistics.pstdev(ordered),
    }


def _load_contexts(qualification_root: Path) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    report = json.loads((qualification_root / "report.json").read_text(encoding="utf-8"))
    contexts: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for execution_context in sorted(
        report["execution"]["contexts"], key=lambda item: int(item["focus_segment_ordinal"])
    ):
        digest = execution_context["context_manifest_semantic_sha256"]
        context_payload: dict[str, Any] | None = None
        context_artifact_path: Path | None = None
        for candidate in (qualification_root / "artifacts" / "context").glob("*/*.json"):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("context_manifest_semantic_sha256") == digest:
                context_payload = payload
                context_artifact_path = candidate
                break
        if context_payload is None or context_artifact_path is None:
            raise FileNotFoundError(f"context artifact not found for {digest}")
        contexts.append((execution_context, context_payload, context_artifact_path))
    if len(contexts) != 5:
        raise ValueError(f"expected five qualification contexts, got {len(contexts)}")
    return contexts


def _typed_context(raw_context: dict[str, Any]) -> Any:
    from robata.contracts.perception_stream import PerceptionContextManifest

    try:
        context = PerceptionContextManifest.model_validate(raw_context, strict=False)
    except Exception as error:
        raise ValueError("qualification context artifact failed contract validation") from error
    if context.context_manifest_semantic_sha256 != raw_context["context_manifest_semantic_sha256"]:
        raise ValueError("qualification context semantic digest changed during validation")
    return context


def _segment_model(context: dict[str, Any], execution_context: dict[str, Any]) -> tuple[Any, Any]:
    from robata.contracts.cameras import CameraId
    from robata.inference.mage_video_adapter import MageVideoDurableCameraSegment

    path = Path(execution_context["durable_path"]).expanduser().resolve(strict=True)
    camera = context["cameras"]["cam_01"]
    content_sha = _sha256_file(path)
    segment = MageVideoDurableCameraSegment(
        camera_id=CameraId.CAM_01,
        segment_semantic_sha256_values=tuple(
            item["segment_semantic_sha256"] for item in context["ordered_segments"]
        ),
        codec_stream_exact_sha256=camera["codec_stream_exact_sha256"],
        durable_path=str(path),
        content_sha256=content_sha,
        byte_count=path.stat().st_size,
    )
    return segment, _typed_context(context)


def _load_model(model_dir: Path) -> tuple[Any, Any, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    ).eval()
    return processor, model, float(time.perf_counter() - started)


def _run_generation(
    *, model: Any, processor: Any, inputs: dict[str, Any], max_new_tokens: int
) -> tuple[str, float, int, int]:
    import torch

    device = next(model.parameters()).device
    synchronize = torch.cuda.is_available()
    materialized = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in inputs.items()
    }
    if materialized.get("pixel_values") is not None:
        materialized["pixel_values"] = materialized["pixel_values"].to(model.dtype)
    prompt_tokens = int(materialized["input_ids"].shape[1])
    with torch.inference_mode():
        if synchronize:
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model.generate(
            **materialized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if synchronize:
            torch.cuda.synchronize()
    seconds = float(time.perf_counter() - started)
    generated_only = generated[:, prompt_tokens:]
    text = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
    return text, seconds, prompt_tokens, int(generated_only.shape[1])


def _run_shadow_generation(
    *,
    model: Any,
    processor: Any,
    shadow_inputs: Any,
    max_new_tokens: int,
) -> tuple[str, float, int]:
    import torch

    synchronize = torch.cuda.is_available()
    with torch.inference_mode():
        if synchronize:
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model.generate(
            input_ids=shadow_inputs.input_ids,
            inputs_embeds=shadow_inputs.inputs_embeds,
            attention_mask=shadow_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if synchronize:
            torch.cuda.synchronize()
    seconds = float(time.perf_counter() - started)
    prompt_tokens = int(shadow_inputs.input_ids.shape[1])
    generated_only = generated[:, prompt_tokens:]
    text = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
    return text, seconds, int(generated_only.shape[1])


def _prepare_processor_inputs(
    *,
    processor: Any,
    segment: Any,
    prompt: str,
    codec_config: dict[str, object],
    max_pixels: int,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prepared = processor(
        text=[text],
        videos=[segment.durable_path],
        video_backend="codec",
        codec_config=codec_config,
        max_pixels=max_pixels,
        return_tensors="pt",
        padding=True,
    )
    if not isinstance(prepared, Mapping):
        raise TypeError("Mage processor must return a mapping of decoder inputs")
    return dict(prepared)


def _warmup_pair(
    *,
    model: Any,
    processor: Any,
    shadow_encoder: Any,
    native_inputs: dict[str, Any],
    max_new_tokens: int,
) -> None:
    """Warm both decoder paths once; warmup output is never scored."""

    _run_generation(
        model=model,
        processor=processor,
        inputs=native_inputs,
        max_new_tokens=max_new_tokens,
    )
    shadow_inputs = shadow_encoder.prepare(native_inputs)
    _run_shadow_generation(
        model=model,
        processor=processor,
        shadow_inputs=shadow_inputs,
        max_new_tokens=max_new_tokens,
    )
    del shadow_inputs
    import torch

    torch.cuda.empty_cache()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    import torch

    from robata.benchmark.mage_small_encoder import (
        MAX_MATCHED_BOUNDARY_MAE_SECONDS,
        SMALL_ENCODER_EVALUATOR_VERSION,
        aggregate_small_encoder_shadow_run,
        evaluate_small_encoder_pair,
        parse_compact_output,
    )
    from robata.inference.mage_small_encoder import (
        MageCompatibleSmallEncoder,
        MageSmallEncoderPolicy,
    )
    from robata.inference.mage_video_adapter import (
        MageVideoObservationAdapterConfig,
        build_mage_video_unified_observation_prompt,
    )

    qualification_root = Path(arguments.qualification_root).expanduser().resolve()
    model_dir = Path(arguments.model_dir).expanduser().resolve()
    manifest_path = Path(arguments.checkpoint_manifest_path).expanduser().resolve()
    codec_cache_dir = Path(arguments.codec_cache_dir).expanduser().resolve()
    if not codec_cache_dir.is_dir():
        raise FileNotFoundError(f"codec cache directory not found: {codec_cache_dir}")
    os.environ["ONLINE_CODEC_CACHE_DIR"] = str(codec_cache_dir)
    codec_config = json.loads(arguments.codec_config_json)
    if not isinstance(codec_config, dict):
        raise ValueError("codec config JSON must decode to an object")
    processor_max_pixels = int(codec_config.get("max_pixels", 0))
    if processor_max_pixels <= 0:
        raise ValueError("codec config max_pixels must be a positive integer")
    manifest, manifest_file_sha256 = _verified_checkpoint_identity(
        model_dir=model_dir,
        manifest_path=manifest_path,
    )
    contexts = _load_contexts(qualification_root)
    processor, model, load_seconds = _load_model(model_dir)
    device = next(model.parameters()).device
    policy = MageSmallEncoderPolicy(
        visual_layer_count=arguments.visual_layer_count,
        max_temporal_runs=arguments.max_temporal_runs,
    )
    shadow_encoder = MageCompatibleSmallEncoder(model=model, policy=policy)
    adapter_config = MageVideoObservationAdapterConfig(max_new_tokens=arguments.max_new_tokens)
    sampler = _GpuSampler()
    segment_results: list[dict[str, object]] = []
    segment_evaluations = []
    native_times: list[float] = []
    candidate_times: list[float] = []
    candidate_preparation_times: list[float] = []
    candidate_total_times: list[float] = []
    execution_orders: list[dict[str, object]] = []
    context_artifact_hashes: dict[str, str] = {}
    recording_duration_seconds = 0.0
    warmup_done = False
    sampler_started = False

    for execution_context, raw_context, context_artifact_path in contexts:
        segment, context = _segment_model(raw_context, execution_context)
        context_ordinal = int(execution_context["focus_segment_ordinal"])
        context_artifact_hash = _sha256_file(context_artifact_path)
        context_artifact_hashes[str(context_ordinal)] = context_artifact_hash
        prompt = build_mage_video_unified_observation_prompt(
            context=context, segment=segment, config=adapter_config
        )
        prompt_exact_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        processor_inputs = _prepare_processor_inputs(
            processor=processor,
            segment=segment,
            prompt=prompt,
            codec_config=codec_config,
            max_pixels=processor_max_pixels,
        )
        native_inputs = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in processor_inputs.items()
        }
        if native_inputs.get("pixel_values") is not None:
            native_inputs["pixel_values"] = native_inputs["pixel_values"].to(model.dtype)
        if not warmup_done:
            _warmup_pair(
                model=model,
                processor=processor,
                shadow_encoder=shadow_encoder,
                native_inputs=native_inputs,
                max_new_tokens=arguments.warmup_max_new_tokens,
            )
            sampler.start()
            sampler_started = True
            warmup_done = True
        segment_duration = float(context.context_interval.duration_ns) / 1_000_000_000
        recording_duration_seconds += segment_duration
        for repetition_ordinal in range(arguments.repetitions):
            native_first = (context_ordinal + repetition_ordinal) % 2 == 0
            order = "NATIVE_THEN_CANDIDATE" if native_first else "CANDIDATE_THEN_NATIVE"
            sampler.set_phase("IDLE")
            native_text = ""
            native_seconds = 0.0
            native_prompt_tokens = 0
            native_output_tokens = 0
            native_peak_memory_mib: float | None = None
            shadow_text = ""
            shadow_seconds = 0.0
            shadow_output_tokens = 0
            candidate_peak_memory_mib: float | None = None
            shadow_inputs: Any | None = None
            if native_first:
                _reset_cuda_peak_memory(torch, device)
                sampler.set_phase("NATIVE_GENERATION")
                native_text, native_seconds, native_prompt_tokens, native_output_tokens = (
                    _run_generation(
                        model=model,
                        processor=processor,
                        inputs=native_inputs,
                        max_new_tokens=arguments.max_new_tokens,
                    )
                )
                native_peak_memory_mib = _cuda_peak_memory_mib(torch, device)
                _reset_cuda_peak_memory(torch, device)
                sampler.set_phase("CANDIDATE_PREPARATION")
                shadow_inputs = shadow_encoder.prepare(native_inputs)
                prep_seconds = float(shadow_inputs.telemetry.total_seconds)
                sampler.set_phase("CANDIDATE_GENERATION")
                shadow_text, shadow_seconds, shadow_output_tokens = _run_shadow_generation(
                    model=model,
                    processor=processor,
                    shadow_inputs=shadow_inputs,
                    max_new_tokens=arguments.max_new_tokens,
                )
                candidate_peak_memory_mib = _cuda_peak_memory_mib(torch, device)
            else:
                _reset_cuda_peak_memory(torch, device)
                sampler.set_phase("CANDIDATE_PREPARATION")
                shadow_inputs = shadow_encoder.prepare(native_inputs)
                prep_seconds = float(shadow_inputs.telemetry.total_seconds)
                sampler.set_phase("CANDIDATE_GENERATION")
                shadow_text, shadow_seconds, shadow_output_tokens = _run_shadow_generation(
                    model=model,
                    processor=processor,
                    shadow_inputs=shadow_inputs,
                    max_new_tokens=arguments.max_new_tokens,
                )
                candidate_peak_memory_mib = _cuda_peak_memory_mib(torch, device)
                _reset_cuda_peak_memory(torch, device)
                sampler.set_phase("NATIVE_GENERATION")
                native_text, native_seconds, native_prompt_tokens, native_output_tokens = (
                    _run_generation(
                        model=model,
                        processor=processor,
                        inputs=native_inputs,
                        max_new_tokens=arguments.max_new_tokens,
                    )
                )
                native_peak_memory_mib = _cuda_peak_memory_mib(torch, device)
            sampler.set_phase("IDLE")
            native_times.append(native_seconds)
            candidate_times.append(shadow_seconds)
            candidate_preparation_times.append(prep_seconds)
            candidate_total_times.append(shadow_seconds + prep_seconds)
            execution_orders.append(
                {
                    "segment_ordinal": context_ordinal,
                    "repetition_ordinal": repetition_ordinal,
                    "order": order,
                }
            )
            native_parsed = parse_compact_output(native_text)
            shadow_parsed = parse_compact_output(shadow_text)
            evaluation = evaluate_small_encoder_pair(
                native_output_text=native_text,
                candidate_output_text=shadow_text,
            )
            segment_evaluations.append(evaluation)
            segment_results.append(
                {
                    "ordinal": context_ordinal,
                    "repetition_ordinal": repetition_ordinal,
                    "execution_order": order,
                    "context_artifact_exact_sha256": context_artifact_hash,
                    "context_manifest_key": context.context_manifest_key,
                    "context_manifest_semantic_sha256": context.context_manifest_semantic_sha256,
                    "prompt_exact_sha256": prompt_exact_sha256,
                    "segment_path": segment.durable_path,
                    "segment_content_sha256": segment.content_sha256,
                    "segment_duration_seconds": segment_duration,
                    "native": {
                        "generation_seconds": native_seconds,
                        "prompt_tokens": native_prompt_tokens,
                        "output_tokens": native_output_tokens,
                        "rtf_generation": segment_duration / native_seconds
                        if native_seconds
                        else None,
                        "output_text": native_text,
                        "json_syntax_valid": native_parsed.json_syntax_valid,
                        "compact_contract_valid": native_parsed.compact_contract_valid,
                        "action_labels": list(native_parsed.normalized_labels),
                        "observation_count": len(native_parsed.actions),
                        "torch_peak_memory_allocated_mib": native_peak_memory_mib,
                    },
                    "small_encoder_shadow": {
                        "generation_seconds": shadow_seconds,
                        "prompt_tokens": int(shadow_inputs.input_ids.shape[1]),
                        "output_tokens": shadow_output_tokens,
                        "rtf_generation": segment_duration / shadow_seconds
                        if shadow_seconds
                        else None,
                        "output_text": shadow_text,
                        "json_syntax_valid": shadow_parsed.json_syntax_valid,
                        "compact_contract_valid": shadow_parsed.compact_contract_valid,
                        "action_labels": list(shadow_parsed.normalized_labels),
                        "observation_count": len(shadow_parsed.actions),
                        "telemetry": shadow_inputs.telemetry.as_projection(),
                        "torch_peak_memory_allocated_mib": candidate_peak_memory_mib,
                        "shadow_only": True,
                    },
                    "comparison": evaluation.as_projection(),
                }
            )
            del shadow_inputs
            torch.cuda.empty_cache()
        del processor_inputs, native_inputs
        torch.cuda.empty_cache()

    gpu_samples = sampler.stop() if sampler_started else []
    qualification_result = aggregate_small_encoder_shadow_run(
        evaluations=tuple(segment_evaluations),
        native_generation_seconds=sum(native_times),
        candidate_generation_seconds=sum(candidate_times),
        candidate_preparation_seconds=sum(candidate_preparation_times),
    )
    qualification = qualification_result.as_projection()
    report: dict[str, object] = {
        "report_version": MAGE_SMALL_ENCODER_REPORT_VERSION,
        "authority": "MAGE_NATIVE",
        "candidate": {
            "policy": policy.model_dump(mode="json"),
            "policy_semantic_sha256": policy.semantic_sha256,
            "shadow_only": True,
        },
        "model": {
            "identifier": manifest.model_identifier,
            "revision": manifest.model_revision,
            "model_directory": str(model_dir),
            "load_seconds": load_seconds,
            "execution_device": str(device),
            "runtime_profile": "bitsandbytes_4bit_nf4_v1",
            "checkpoint_manifest_sha256": manifest.manifest_sha256,
            "checkpoint_manifest_file_sha256": manifest_file_sha256,
            "runtime_identity": _runtime_identity(),
        },
        "implementation": _git_and_source_identity(Path(__file__).resolve().parents[1]),
        "evaluator": {
            "version": SMALL_ENCODER_EVALUATOR_VERSION,
            "max_matched_boundary_mae_seconds": MAX_MATCHED_BOUNDARY_MAE_SECONDS,
        },
        "input": {
            "qualification_root": str(qualification_root),
            "camera": "cam_01",
            "segment_count": len(contexts),
            "repetitions": arguments.repetitions,
            "paired_call_count": len(segment_results),
            "recording_duration_seconds": recording_duration_seconds,
            "effective_media_seconds": recording_duration_seconds * arguments.repetitions,
            "source_report_sha256": _sha256_file(qualification_root / "report.json"),
            "codec_cache_dir": str(codec_cache_dir),
            "codec_config": codec_config,
            "codec_config_semantic_sha256": _canonical_sha256(codec_config),
            "processor_max_pixels": processor_max_pixels,
            "context_artifact_exact_sha256": context_artifact_hashes,
        },
        "prompt": {
            "config": adapter_config.model_dump(mode="json"),
            "config_semantic_sha256": _canonical_sha256(adapter_config.model_dump(mode="json")),
            "prompt_outside_processor_timing": True,
        },
        "execution": {
            "worker_count": 1,
            "generation_concurrency": 1,
            "max_inflight": 1,
            "warmup_max_new_tokens": arguments.warmup_max_new_tokens,
            "warmup_scored": False,
            "orders": execution_orders,
            "native_authority": True,
        },
        "latency": {
            "native_generation": _latency_summary(native_times),
            "candidate_generation": _latency_summary(candidate_times),
            "candidate_preparation": _latency_summary(candidate_preparation_times),
            "candidate_total": _latency_summary(candidate_total_times),
        },
        "segments": segment_results,
        "gpu": _aggregate_gpu(gpu_samples),
        "gpu_by_phase": _aggregate_gpu_by_phase(gpu_samples),
        "qualification": qualification,
        "verdict": (
            "SHADOW_QUALIFIED_FOR_NEXT_CANARY_ONLY"
            if qualification_result.qualified
            else "REJECTED_SHADOW_KEEP_MAGE_NATIVE_AUTHORITY"
        ),
    }
    return _seal_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-root",
        default=r"D:\Github\Robata\.local\mage-vnext-qualification-20260808-r9",
    )
    parser.add_argument("--model-dir", default=r"D:\HuggingFace\Mage-VL")
    parser.add_argument(
        "--codec-cache-dir",
        default=r"D:\Github\Robata\.tmp_mage_cache_final",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--visual-layer-count", type=int, default=24)
    parser.add_argument("--max-temporal-runs", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--checkpoint-manifest-path",
        default=r"D:\Github\Robata\.local\mage-qual-v7-endpoint\checkpoint-manifest-v2.json",
    )
    parser.add_argument(
        "--codec-config-json",
        default=json.dumps(
            {
                "engine": "dcvc-rt",
                "target_canvas": 8,
                "group_size": 8,
                "images_per_group": 1,
                "patch": 16,
                "max_pixels": 65536,
                "min_group_frames": 8,
                "max_group_frames": 8,
                "dcvc": {
                    "qp": 42,
                    "reset_interval": 64,
                    "intra_period": -1,
                    "max_side": 448,
                    "seq_len_frames": 8,
                    "device": "cuda",
                    "max_group_frames": 8,
                    "readiness_coverage_bins": 3,
                    "readiness_delta_ratio": 0.05,
                    "bitcost_pct": 99,
                    "decode_backsearch_max": 16,
                    "pkg_dir": r"D:\HuggingFace\Mage-VL\neural_codec",
                },
            }
        ),
    )
    arguments = parser.parse_args()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run(arguments)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
