"""Real local MCAP-to-Hugging-Face qualification path.

This module is intentionally non-canonical.  It exercises a real six-camera
MCAP, content-addressed local filesystem storage, one real local vision model,
and durable SQLite observations without claiming PostgreSQL/R2 production
qualification or canonical publication authority.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import Field, StringConstraints, model_validator

from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.local_hf_endpoint import (
    LocalHfEncodedImage,
    LocalHfEndpointRequest,
    LocalHfEndpointResponse,
)
from robata.runtime.e2e_participation import (
    E2EParticipationBoundary,
    E2EParticipationCoverage,
    E2EParticipationDeclaration,
    E2EParticipationState,
    build_e2e_participation_manifest,
    write_e2e_participation_manifest,
)
from robata.runtime.e2e_trace import (
    E2ETraceFragmentRole,
    E2ETraceRuntimeFragment,
    build_e2e_trace_runtime_fragment,
)
from robata.runtime.observability import RuntimeProfileRecorder, runtime_span

LOCAL_REAL_MODEL_E2E_VERSION: Literal["local-real-model-e2e-v1"] = "local-real-model-e2e-v1"
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class LocalRealModelE2EError(RuntimeError):
    """The bounded local qualification could not complete."""


class LocalFrameArtifact(StrictModel):
    """One real decoded camera frame stored in the local R2 stand-in."""

    camera_id: NonEmptyString
    topic: NonEmptyString
    source_timestamp_ns: Nanoseconds
    messages_examined: PositiveInt
    decode_failures: tuple[NonEmptyString, ...]
    width: PositiveInt
    height: PositiveInt
    media_type: Literal["image/png"] = "image/png"
    uri: NonEmptyString
    sha256: Sha256Digest
    byte_count: PositiveInt


class LocalModelObservation(StrictModel):
    """Measured output and resource facts for one real local model call."""

    provider: Literal["local-huggingface"] = "local-huggingface"
    model_transport: Literal["IN_PROCESS", "LOOPBACK_HTTP"]
    model_identifier: NonEmptyString
    model_version: NonEmptyString
    quantization: Literal["bnb-nf4-double-quant"] = "bnb-nf4-double-quant"
    precision: Literal["bfloat16-compute"] = "bfloat16-compute"
    input_image_count: PositiveInt
    rendered_image_sizes: tuple[tuple[PositiveInt, PositiveInt], ...]
    prompt_tokens: PositiveInt
    output_tokens: NonNegativeInt
    load_seconds: NonNegativeFloat
    generation_seconds: NonNegativeFloat
    gpu_name: NonEmptyString
    gpu_total_bytes: PositiveInt
    gpu_free_before_bytes: NonNegativeInt
    gpu_allocated_after_load_bytes: NonNegativeInt
    gpu_peak_allocated_bytes: NonNegativeInt
    output_text: NonEmptyString
    parsed_json: dict[str, object] | None


class LocalStorageObservation(StrictModel):
    """Local development substitutes; these are never production authorities."""

    object_store_kind: Literal["LOCAL_FILESYSTEM_R2_STAND_IN"] = "LOCAL_FILESYSTEM_R2_STAND_IN"
    object_store_root: NonEmptyString
    sqlite_kind: Literal["LOCAL_QUALIFICATION_SIDECAR"] = "LOCAL_QUALIFICATION_SIDECAR"
    sqlite_path: NonEmptyString


class LocalRealModelE2EReport(StrictModel):
    """Operator-readable result of one bounded real-model local execution."""

    report_version: Literal["local-real-model-e2e-v1"] = LOCAL_REAL_MODEL_E2E_VERSION
    run_id: NonEmptyString
    observed_at: NonEmptyString
    execution_class: Literal["LOCAL_QUALIFICATION"] = "LOCAL_QUALIFICATION"
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    source_path: NonEmptyString
    source_sha256: Sha256Digest
    source_size_bytes: PositiveInt
    source_profile: NonEmptyString
    source_message_count: PositiveInt
    mapping_profile_id: NonEmptyString
    mapping_approval_status: NonEmptyString
    camera_artifacts: tuple[LocalFrameArtifact, ...]
    prompt: NonEmptyString
    model: LocalModelObservation
    storage: LocalStorageObservation
    trace: E2ETraceRuntimeFragment
    stage_coverage: dict[NonEmptyString, NonEmptyString]
    participation_coverage: E2EParticipationCoverage
    participation_manifest_sha256: Sha256Digest
    participation_manifest_path: NonEmptyString
    quality_observation: dict[NonEmptyString, object]
    canary_status: Literal["NOT_EXECUTED"] = "NOT_EXECUTED"
    shadow_status: Literal["NOT_EXECUTED"] = "NOT_EXECUTED"
    canonical_pipeline_status: Literal["NOT_COMPOSED"] = "NOT_COMPOSED"
    production_eligible: Literal[False] = False
    canonical_authority: Literal[False] = False
    warnings: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_six_cameras(self) -> LocalRealModelE2EReport:
        if len(self.camera_artifacts) != 6:
            raise ValueError("local real-model qualification requires exactly six cameras")
        if self.model.input_image_count != len(self.camera_artifacts):
            raise ValueError("model input count must match decoded camera artifacts")
        return self


@dataclass(frozen=True, slots=True)
class _DecodedFrame:
    camera_id: str
    topic: str
    source_timestamp_ns: int
    messages_examined: int
    decode_failures: tuple[str, ...]
    width: int
    height: int
    png_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ModelExecution:
    observation: LocalModelObservation


def run_local_real_model_e2e(
    *,
    source_path: Path,
    mapping_config: Path,
    model_directory: Path,
    state_directory: Path,
    allow_unapproved_profile: bool,
    model_identifier: str = "Qwen3-VL-4B-Instruct",
    model_version: str = "local",
    max_image_side: int = 448,
    max_new_tokens: int = 64,
    gpu_weight_memory_gib: int = 4,
    cpu_weight_memory_gib: int = 1,
    endpoint_url: str | None = None,
    endpoint_timeout_seconds: float = 300.0,
    prompt: str | None = None,
) -> tuple[LocalRealModelE2EReport, Path, Path, Path]:
    """Run one real six-camera sample through one local quantized VLM."""

    source = _require_file(source_path, "source_path")
    mapping = _require_file(mapping_config, "mapping_config")
    model_dir = _require_directory(model_directory, "model_directory")
    state_root = Path(state_directory).resolve()
    if (
        isinstance(max_image_side, bool)
        or not isinstance(max_image_side, int)
        or max_image_side < 224
    ):
        raise LocalRealModelE2EError("max_image_side must be an integer of at least 224")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise LocalRealModelE2EError("max_new_tokens must be positive")
    if not isinstance(allow_unapproved_profile, bool):
        raise LocalRealModelE2EError("allow_unapproved_profile must be a boolean")
    if (
        isinstance(endpoint_timeout_seconds, bool)
        or not isinstance(endpoint_timeout_seconds, (int, float))
        or endpoint_timeout_seconds <= 0
    ):
        raise LocalRealModelE2EError("endpoint_timeout_seconds must be positive")
    resolved_endpoint_url = (
        _normalize_loopback_endpoint_url(endpoint_url) if endpoint_url is not None else None
    )
    state_root.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    run_root = state_root / "runs" / run_id
    object_root = state_root / "r2"
    run_root.mkdir(parents=True, exist_ok=False)
    object_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = state_root / "local-qualification.sqlite3"
    resolved_prompt = prompt or (
        "You are observing six synchronized robot camera channels. Return one compact strict "
        "JSON object only, with no markdown and no more than 100 words. Required keys: "
        '"scene_summary" (one sentence), "observed_objects" (at most five short strings), '
        '"observable_actions" (at most three short strings), "cross_camera_consistency" '
        '(one short sentence), and "uncertainties" (at most three short strings). Arrays '
        "must be JSON arrays, never comma-separated strings. Use only visible evidence and "
        "do not speculate."
    )

    recorder = RuntimeProfileRecorder()
    mapping_document: dict[str, object]
    with runtime_span(recorder, "source.mapping.read_validate"):
        mapping_document = _load_mapping(mapping, allow_unapproved_profile)
    topics = _mapping_topics(mapping_document)
    with runtime_span(recorder, "source.mcap.inspect"):
        inspection = OfficialMcapInspector().inspect(source)
        _validate_inspection(inspection, topics)
    with runtime_span(recorder, "source.frame.decode", {"camera_count": 6}):
        decoded = _decode_first_synchronized_frames(source, topics)
    with runtime_span(recorder, "r2.local.put", {"artifact_count": len(decoded)}):
        artifacts = tuple(_publish_frame(object_root, frame) for frame in decoded)

    if resolved_endpoint_url is None:
        cache_root = state_root / "model-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_MODULES_CACHE", str(cache_root / "modules"))
        os.environ.setdefault("HF_HOME", str(cache_root / "hf-home"))
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        model_execution = _run_huggingface_model(
            recorder=recorder,
            model_directory=model_dir,
            model_identifier=model_identifier,
            model_version=model_version,
            artifacts=artifacts,
            prompt=resolved_prompt,
            max_image_side=max_image_side,
            max_new_tokens=max_new_tokens,
            gpu_weight_memory_gib=gpu_weight_memory_gib,
            cpu_weight_memory_gib=cpu_weight_memory_gib,
            offload_directory=state_root / "model-offload",
        )
    else:
        model_execution = _run_huggingface_endpoint(
            recorder=recorder,
            endpoint_url=resolved_endpoint_url,
            model_identifier=model_identifier,
            model_version=model_version,
            artifacts=artifacts,
            prompt=resolved_prompt,
            max_new_tokens=max_new_tokens,
            timeout_seconds=float(endpoint_timeout_seconds),
        )

    with runtime_span(recorder, "sqlite.local_qualification.persist"):
        _persist_preliminary_observation(
            sqlite_path=sqlite_path,
            run_id=run_id,
            observed_at=_utc_now_text(),
            source_sha256=inspection.source_sha256,
            model=model_execution.observation,
            artifacts=artifacts,
        )
    with runtime_span(recorder, "publication.local_qualification.compose"):
        quality = _quality_observation(model_execution.observation, artifacts)

    runtime_profile = recorder.snapshot()
    trace = build_e2e_trace_runtime_fragment(
        role=E2ETraceFragmentRole.LAUNCHER,
        runtime_profile=runtime_profile,
    )
    stage_coverage = {stage.stage.value: stage.measurement_status.value for stage in trace.stages}
    warnings = _warnings(mapping_document, artifacts)
    observed_at = _utc_now_text()
    participation = build_e2e_participation_manifest(
        runtime_fragment=trace,
        declarations=(
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.ORCHESTRATION,
                state=E2EParticipationState.NOT_CONFIGURED,
                required=False,
                reason="bounded local qualification has no separate orchestration service",
            ),
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.SOURCE,
                state=E2EParticipationState.PARTICIPATING,
                required=True,
            ),
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.SCHEDULING,
                state=E2EParticipationState.NOT_CONFIGURED,
                required=False,
                reason="local qualification invokes one bounded model call without a scheduler",
            ),
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.INFERENCE,
                state=E2EParticipationState.PARTICIPATING,
                required=True,
            ),
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.EVIDENCE,
                state=E2EParticipationState.PARTICIPATING,
                required=True,
                reason="local filesystem and SQLite stand-ins participate in this route",
            ),
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.REDUCTION,
                state=E2EParticipationState.NOT_CONFIGURED,
                required=False,
                reason="single-call local qualification does not execute canonical reduction",
            ),
            E2EParticipationDeclaration(
                boundary=E2EParticipationBoundary.PUBLICATION,
                state=E2EParticipationState.PARTICIPATING,
                required=True,
                reason="only the local qualification report publication boundary participates",
            ),
        ),
        trace_id=run_id,
        observed_at=observed_at,
    )
    participation_path = run_root / "participation.json"
    participation_digest = write_e2e_participation_manifest(
        participation,
        participation_path,
    )
    report = LocalRealModelE2EReport(
        run_id=run_id,
        observed_at=observed_at,
        source_path=str(source),
        source_sha256=inspection.source_sha256,
        source_size_bytes=inspection.source_size_bytes,
        source_profile=inspection.header_profile,
        source_message_count=inspection.message_count,
        mapping_profile_id=str(mapping_document["profile_id"]),
        mapping_approval_status=str(mapping_document["approval_status"]),
        camera_artifacts=artifacts,
        prompt=resolved_prompt,
        model=model_execution.observation,
        storage=LocalStorageObservation(
            object_store_root=str(object_root),
            sqlite_path=str(sqlite_path),
        ),
        trace=trace,
        stage_coverage=stage_coverage,
        participation_coverage=participation.coverage,
        participation_manifest_sha256=participation_digest,
        participation_manifest_path=str(participation_path),
        quality_observation=quality,
        warnings=warnings,
    )
    report_path = run_root / "report.json"
    trace_path = run_root / "trace.json"
    # Operational trace/report fields include MCAP nanosecond timestamps.  Those
    # observations are intentionally not canonical wire projections; RFC 8785
    # rejects integers above JavaScript's safe-number domain.  Preserve the exact
    # integer facts in ordinary deterministic JSON and reserve canonical JSON for
    # identity-bearing contracts.
    report_bytes = _observational_json_bytes(report.model_dump(mode="json"))
    trace_bytes = _observational_json_bytes(trace.model_dump(mode="json"))
    _atomic_write(report_path, report_bytes)
    _atomic_write(trace_path, trace_bytes)
    _persist_final_report(sqlite_path, run_id, report_bytes, report_path, trace_path)
    return report, report_path, trace_path, participation_path


def _load_mapping(path: Path, allow_unapproved: bool) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LocalRealModelE2EError(f"invalid mapping config: {error}") from error
    if not isinstance(document, dict):
        raise LocalRealModelE2EError("mapping config must be a JSON object")
    required = ("profile_id", "approval_status", "approved", "topics")
    if any(key not in document for key in required):
        raise LocalRealModelE2EError("mapping config is missing required fields")
    approved = document["approved"]
    if approved is not True and not allow_unapproved:
        raise LocalRealModelE2EError(
            "mapping profile is not approved; pass --allow-unapproved-profile explicitly"
        )
    return document


def _mapping_topics(document: dict[str, object]) -> tuple[tuple[str, str], ...]:
    topics = document.get("topics")
    if not isinstance(topics, dict):
        raise LocalRealModelE2EError("mapping topics must be an object")
    expected = tuple(f"cam_{index:02d}" for index in range(1, 7))
    if tuple(sorted(topics)) != expected:
        raise LocalRealModelE2EError("mapping must contain exactly cam_01 through cam_06")
    resolved: list[tuple[str, str]] = []
    for camera_id in expected:
        topic = topics[camera_id]
        if not isinstance(topic, str) or not topic:
            raise LocalRealModelE2EError(f"mapping topic for {camera_id} must be nonempty")
        resolved.append((camera_id, topic))
    if len({topic for _, topic in resolved}) != 6:
        raise LocalRealModelE2EError("mapping camera topics must be unique")
    return tuple(resolved)


def _validate_inspection(inspection: Any, topics: tuple[tuple[str, str], ...]) -> None:
    channels = {channel.topic: channel for channel in inspection.channels}
    for camera_id, topic in topics:
        channel = channels.get(topic)
        if channel is None:
            raise LocalRealModelE2EError(f"{camera_id} topic is absent from MCAP: {topic}")
        if channel.schema_name != "foxglove.CompressedImage":
            raise LocalRealModelE2EError(f"{camera_id} does not use foxglove.CompressedImage")
        if (channel.codec or "").lower() != "h264":
            raise LocalRealModelE2EError(f"{camera_id} does not declare H.264")


def _decode_first_synchronized_frames(
    source: Path,
    topics: tuple[tuple[str, str], ...],
    *,
    max_messages_per_camera: int = 120,
) -> tuple[_DecodedFrame, ...]:
    try:
        av_module = import_module("av")
        reader_module = import_module("mcap.reader")
        decoder_module = import_module("mcap_protobuf.decoder")
    except ImportError as error:
        raise LocalRealModelE2EError(
            "real MCAP decoding requires the robata[mcap] optional dependencies"
        ) from error
    make_reader = reader_module.make_reader
    decoder_factory = decoder_module.DecoderFactory
    topic_to_camera = {topic: camera_id for camera_id, topic in topics}
    decoders = {topic: av_module.CodecContext.create("h264", "r") for topic in topic_to_camera}
    examined = {topic: 0 for topic in topic_to_camera}
    failures: dict[str, list[str]] = {topic: [] for topic in topic_to_camera}
    found: dict[str, _DecodedFrame] = {}
    try:
        with source.open("rb") as stream:
            reader = make_reader(
                stream,
                validate_crcs=True,
                decoder_factories=[decoder_factory()],
            )
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=tuple(topic_to_camera)
            ):
                topic = channel.topic
                if topic in found or topic not in topic_to_camera:
                    continue
                examined[topic] += 1
                payload = getattr(decoded, "data", None)
                if schema is None or schema.name != "foxglove.CompressedImage":
                    failures[topic].append("INVALID_COMPRESSED_IMAGE_SCHEMA")
                    continue
                if not isinstance(payload, bytes):
                    failures[topic].append("INVALID_COMPRESSED_IMAGE_PAYLOAD")
                    continue
                packet = av_module.Packet(payload)
                packet.pts = message.log_time
                packet.dts = message.log_time
                packet.time_base = Fraction(1, 1_000_000_000)
                try:
                    frames = decoders[topic].decode(packet)
                except Exception as error:
                    failures[topic].append(f"H264_DECODE_ERROR:{type(error).__name__}")
                    frames = []
                if frames:
                    image = frames[0].to_image().convert("RGB")
                    buffer = BytesIO()
                    image.save(buffer, format="PNG", optimize=False)
                    png_bytes = buffer.getvalue()
                    found[topic] = _DecodedFrame(
                        camera_id=topic_to_camera[topic],
                        topic=topic,
                        source_timestamp_ns=message.log_time,
                        messages_examined=examined[topic],
                        decode_failures=tuple(failures[topic]),
                        width=image.width,
                        height=image.height,
                        png_bytes=png_bytes,
                    )
                if examined[topic] >= max_messages_per_camera and topic not in found:
                    raise LocalRealModelE2EError(
                        f"no decodable frame for {topic} within {max_messages_per_camera} messages"
                    )
                if len(found) == len(topics):
                    break
    except LocalRealModelE2EError:
        raise
    except OSError as error:
        raise LocalRealModelE2EError(f"could not read MCAP source: {error}") from error
    except Exception as error:
        raise LocalRealModelE2EError(
            f"MCAP frame extraction failed: {type(error).__name__}: {error}"
        ) from error
    missing = [topic for _, topic in topics if topic not in found]
    if missing:
        raise LocalRealModelE2EError(f"no decodable frame for camera topics: {missing}")
    return tuple(found[topic] for _, topic in topics)


def _publish_frame(root: Path, frame: _DecodedFrame) -> LocalFrameArtifact:
    digest = exact_bytes_sha256(frame.png_bytes)
    target = root / "sha256" / digest[:2] / f"{digest}.png"
    if target.exists():
        if target.read_bytes() != frame.png_bytes:
            raise LocalRealModelE2EError("local object-store digest collision")
    else:
        _atomic_write(target, frame.png_bytes)
    return LocalFrameArtifact(
        camera_id=frame.camera_id,
        topic=frame.topic,
        source_timestamp_ns=frame.source_timestamp_ns,
        messages_examined=frame.messages_examined,
        decode_failures=frame.decode_failures,
        width=frame.width,
        height=frame.height,
        uri=target.resolve().as_uri(),
        sha256=digest,
        byte_count=len(frame.png_bytes),
    )


def _run_huggingface_model(
    *,
    recorder: RuntimeProfileRecorder,
    model_directory: Path,
    model_identifier: str,
    model_version: str,
    artifacts: tuple[LocalFrameArtifact, ...],
    prompt: str,
    max_image_side: int,
    max_new_tokens: int,
    gpu_weight_memory_gib: int,
    cpu_weight_memory_gib: int,
    offload_directory: Path,
) -> _ModelExecution:
    try:
        with runtime_span(recorder, "inference.runtime.imports"):
            torch = import_module("torch")
            transformers = import_module("transformers")
            image_module = import_module("PIL.Image")
    except ImportError as error:
        raise LocalRealModelE2EError(
            "local real-model execution requires torch, transformers, accelerate, "
            "bitsandbytes, safetensors, and Pillow"
        ) from error
    with runtime_span(recorder, "inference.runtime.initialize"):
        if not bool(torch.cuda.is_available()):
            raise LocalRealModelE2EError("CUDA is required for the bounded 4-bit local model run")
        auto_processor = transformers.AutoProcessor
        auto_model = transformers.AutoModelForImageTextToText
        quantization_class = transformers.BitsAndBytesConfig
        offload_directory.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_free_before, gpu_total = torch.cuda.mem_get_info()
        gpu_name = str(torch.cuda.get_device_name(0))
        quantization = quantization_class(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    load_started = time.perf_counter()
    with runtime_span(recorder, "inference.model.load", {"quantization": "nf4-4bit"}):
        processor = auto_processor.from_pretrained(
            model_directory,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = auto_model.from_pretrained(
            model_directory,
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={
                0: f"{gpu_weight_memory_gib}GiB",
                "cpu": f"{cpu_weight_memory_gib}GiB",
            },
            offload_folder=str(offload_directory),
            low_cpu_mem_usage=True,
        )
    load_seconds = time.perf_counter() - load_started
    allocated_after_load = int(torch.cuda.memory_allocated())
    pil_images: list[Any] = []
    rendered_sizes: list[tuple[int, int]] = []
    with runtime_span(
        recorder,
        "inference.input.prepare",
        {"image_count": len(artifacts), "max_image_side": max_image_side},
    ):
        for artifact in artifacts:
            path = _file_uri_path(artifact.uri)
            image = image_module.open(path).convert("RGB")
            image.thumbnail((max_image_side, max_image_side))
            pil_images.append(image)
            rendered_sizes.append((int(image.width), int(image.height)))
        content: list[dict[str, object]] = [
            {"type": "image", "image": image} for image in pil_images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=pil_images, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if callable(getattr(value, "to", None)) else value
            for key, value in inputs.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[1])
    generation_started = time.perf_counter()
    with (
        runtime_span(
            recorder,
            "inference.model.generate",
            {"image_count": len(pil_images), "max_new_tokens": max_new_tokens},
        ),
        torch.inference_mode(),
    ):
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generation_seconds = time.perf_counter() - generation_started
    with runtime_span(recorder, "inference.output.decode"):
        generated_only = generated[:, prompt_tokens:]
        output_tokens = int(generated_only.shape[1])
        output_text = str(
            processor.batch_decode(generated_only, skip_special_tokens=True)[0]
        ).strip()
        if not output_text:
            raise LocalRealModelE2EError("local model returned an empty output")
        peak = int(torch.cuda.max_memory_allocated())
        observation = LocalModelObservation(
            model_transport="IN_PROCESS",
            model_identifier=model_identifier,
            model_version=model_version,
            input_image_count=len(pil_images),
            rendered_image_sizes=tuple(rendered_sizes),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            load_seconds=float(load_seconds),
            generation_seconds=float(generation_seconds),
            gpu_name=gpu_name,
            gpu_total_bytes=int(gpu_total),
            gpu_free_before_bytes=int(gpu_free_before),
            gpu_allocated_after_load_bytes=allocated_after_load,
            gpu_peak_allocated_bytes=peak,
            output_text=output_text,
            parsed_json=_parse_json_output(output_text),
        )
    with runtime_span(recorder, "inference.runtime.release"):
        del generated, generated_only, inputs, model, processor
        for image in pil_images:
            close = getattr(image, "close", None)
            if callable(close):
                close()
        torch.cuda.empty_cache()
    return _ModelExecution(observation=observation)


def _run_huggingface_endpoint(
    *,
    recorder: RuntimeProfileRecorder,
    endpoint_url: str,
    model_identifier: str,
    model_version: str,
    artifacts: tuple[LocalFrameArtifact, ...],
    prompt: str,
    max_new_tokens: int,
    timeout_seconds: float,
) -> _ModelExecution:
    with runtime_span(
        recorder,
        "inference.endpoint.encode_request",
        {"image_count": len(artifacts), "transport": "loopback-http"},
    ):
        encoded_images: list[LocalHfEncodedImage] = []
        for artifact in artifacts:
            payload = _file_uri_path(artifact.uri).read_bytes()
            if exact_bytes_sha256(payload) != artifact.sha256:
                raise LocalRealModelE2EError(
                    f"local frame artifact digest changed before inference: {artifact.camera_id}"
                )
            encoded_images.append(
                LocalHfEncodedImage(
                    camera_id=artifact.camera_id,
                    sha256=artifact.sha256,
                    base64_data=base64.b64encode(payload).decode("ascii"),
                )
            )
        endpoint_request = LocalHfEndpointRequest(
            request_id=str(uuid4()),
            images=encoded_images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        payload = _observational_json_bytes(endpoint_request.model_dump(mode="json"))
        request = Request(
            f"{endpoint_url}/v1/local-vision/infer",
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "robata-local-real-model-e2e/1",
            },
            method="POST",
        )
    try:
        with (
            runtime_span(
                recorder,
                "inference.endpoint.request",
                {
                    "endpoint": endpoint_url,
                    "image_count": len(encoded_images),
                    "transport": "loopback-http",
                },
            ),
            urlopen(request, timeout=timeout_seconds) as response,
        ):
            response_payload = response.read(1_000_001)
    except HTTPError as error:
        detail = error.read(16_384).decode("utf-8", errors="replace")
        raise LocalRealModelE2EError(
            f"local model endpoint returned HTTP {error.code}: {detail}"
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        raise LocalRealModelE2EError(f"local model endpoint request failed: {error}") from error
    with runtime_span(recorder, "inference.endpoint.validate_response"):
        if len(response_payload) > 1_000_000:
            raise LocalRealModelE2EError("local model endpoint response exceeded 1000000 bytes")
        try:
            endpoint_response = LocalHfEndpointResponse.model_validate_json(response_payload)
        except ValueError as error:
            raise LocalRealModelE2EError(
                f"local model endpoint returned an invalid response contract: {error}"
            ) from error
        if endpoint_response.request_id != endpoint_request.request_id:
            raise LocalRealModelE2EError("local model endpoint response request_id mismatch")
        if endpoint_response.model_identifier != model_identifier:
            raise LocalRealModelE2EError("local model endpoint response model_identifier mismatch")
        if endpoint_response.model_version != model_version:
            raise LocalRealModelE2EError("local model endpoint response model_version mismatch")
        if endpoint_response.input_image_count != len(artifacts):
            raise LocalRealModelE2EError("local model endpoint response image count mismatch")
        observation = LocalModelObservation(
            model_transport="LOOPBACK_HTTP",
            model_identifier=endpoint_response.model_identifier,
            model_version=endpoint_response.model_version,
            quantization=endpoint_response.quantization,
            precision=endpoint_response.precision,
            input_image_count=endpoint_response.input_image_count,
            rendered_image_sizes=endpoint_response.rendered_image_sizes,
            prompt_tokens=endpoint_response.prompt_tokens,
            output_tokens=endpoint_response.output_tokens,
            load_seconds=endpoint_response.load_seconds,
            generation_seconds=endpoint_response.generation_seconds,
            gpu_name=endpoint_response.gpu_name,
            gpu_total_bytes=endpoint_response.gpu_total_bytes,
            gpu_free_before_bytes=endpoint_response.gpu_free_before_bytes,
            gpu_allocated_after_load_bytes=endpoint_response.gpu_allocated_after_load_bytes,
            gpu_peak_allocated_bytes=endpoint_response.gpu_peak_allocated_bytes,
            output_text=endpoint_response.output_text,
            parsed_json=_parse_json_output(endpoint_response.output_text),
        )
    return _ModelExecution(observation=observation)


def _normalize_loopback_endpoint_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalRealModelE2EError("endpoint_url must be a nonempty string")
    parsed = urlparse(value.strip())
    if parsed.scheme != "http":
        raise LocalRealModelE2EError("endpoint_url must use plain HTTP on loopback")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise LocalRealModelE2EError("endpoint_url must target a loopback host")
    if parsed.username is not None or parsed.password is not None:
        raise LocalRealModelE2EError("endpoint_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise LocalRealModelE2EError("endpoint_url must not contain query or fragment components")
    if parsed.path not in {"", "/"}:
        raise LocalRealModelE2EError("endpoint_url must not contain an API path")
    try:
        port = parsed.port
    except ValueError as error:
        raise LocalRealModelE2EError("endpoint_url contains an invalid port") from error
    if port is None:
        raise LocalRealModelE2EError("endpoint_url must include an explicit port")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"http://{host}:{port}"


def _parse_json_output(text: str) -> dict[str, object] | None:
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]))
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _quality_observation(
    model: LocalModelObservation,
    artifacts: tuple[LocalFrameArtifact, ...],
) -> dict[str, object]:
    return {
        "decoded_camera_count": len(artifacts),
        "all_six_cameras_present": len(artifacts) == 6,
        "model_output_nonempty": bool(model.output_text.strip()),
        "strict_json_parsed": model.parsed_json is not None,
        "structured_output_shape_valid": _structured_output_shape_valid(model.parsed_json),
        "ground_truth_quality": "NOT_MEASURED",
        "paired_model_comparison": "NOT_MEASURED",
        "production_load_saturation": "NOT_MEASURED",
    }


def _structured_output_shape_valid(value: dict[str, object] | None) -> bool:
    if value is None:
        return False
    text_fields = ("scene_summary", "cross_camera_consistency")
    array_fields = ("observed_objects", "observable_actions", "uncertainties")
    if set(value) != {*text_fields, *array_fields}:
        return False
    if any(not isinstance(value[field], str) or not value[field] for field in text_fields):
        return False
    for field in array_fields:
        items = value[field]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item for item in items
        ):
            return False
    return True


def _warnings(
    mapping: dict[str, object], artifacts: tuple[LocalFrameArtifact, ...]
) -> tuple[str, ...]:
    warnings = [
        "This LOCAL_QUALIFICATION run is non-canonical and production_eligible=false.",
        "The local filesystem object root is an R2 stand-in, not Cloudflare R2.",
        "The SQLite database is an observational sidecar, not PostgreSQL/Supabase authority.",
        "Canary, shadow, reduction, canonical completion, outbox, and RLS were not executed.",
    ]
    if mapping.get("approved") is not True:
        warnings.append(
            "The selected MCAP mapping profile is UNAPPROVED and was explicitly allowed."
        )
    if any(artifact.decode_failures for artifact in artifacts):
        warnings.append(
            "At least one camera required recovery after an initial H.264 decode error."
        )
    return tuple(warnings)


def _persist_preliminary_observation(
    *,
    sqlite_path: Path,
    run_id: str,
    observed_at: str,
    source_sha256: str,
    model: LocalModelObservation,
    artifacts: tuple[LocalFrameArtifact, ...],
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS local_qualification_runs (
                run_id TEXT PRIMARY KEY,
                observed_at TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                production_eligible INTEGER NOT NULL CHECK (production_eligible = 0),
                report_sha256 TEXT,
                report_path TEXT,
                trace_path TEXT
            );
            CREATE TABLE IF NOT EXISTS local_qualification_artifacts (
                run_id TEXT NOT NULL REFERENCES local_qualification_runs(run_id),
                camera_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                source_timestamp_ns INTEGER NOT NULL,
                artifact_sha256 TEXT NOT NULL,
                artifact_uri TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                PRIMARY KEY (run_id, camera_id)
            );
            CREATE TABLE IF NOT EXISTS local_qualification_model_outputs (
                run_id TEXT PRIMARY KEY REFERENCES local_qualification_runs(run_id),
                model_identifier TEXT NOT NULL,
                quantization TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                load_seconds REAL NOT NULL,
                generation_seconds REAL NOT NULL,
                gpu_peak_allocated_bytes INTEGER NOT NULL,
                output_text TEXT NOT NULL,
                parsed_json TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO local_qualification_runs "
            "(run_id, observed_at, source_sha256, status, production_eligible) "
            "VALUES (?, ?, ?, 'RUNNING', 0)",
            (run_id, observed_at, source_sha256),
        )
        connection.executemany(
            "INSERT INTO local_qualification_artifacts "
            "(run_id, camera_id, topic, source_timestamp_ns, artifact_sha256, "
            "artifact_uri, byte_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    artifact.camera_id,
                    artifact.topic,
                    artifact.source_timestamp_ns,
                    artifact.sha256,
                    artifact.uri,
                    artifact.byte_count,
                )
                for artifact in artifacts
            ],
        )
        connection.execute(
            "INSERT INTO local_qualification_model_outputs "
            "(run_id, model_identifier, quantization, prompt_tokens, output_tokens, "
            "load_seconds, generation_seconds, gpu_peak_allocated_bytes, output_text, "
            "parsed_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                model.model_identifier,
                model.quantization,
                model.prompt_tokens,
                model.output_tokens,
                model.load_seconds,
                model.generation_seconds,
                model.gpu_peak_allocated_bytes,
                model.output_text,
                json.dumps(model.parsed_json, ensure_ascii=False, sort_keys=True)
                if model.parsed_json is not None
                else None,
            ),
        )


def _persist_final_report(
    sqlite_path: Path,
    run_id: str,
    report_bytes: bytes,
    report_path: Path,
    trace_path: Path,
) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        cursor = connection.execute(
            "UPDATE local_qualification_runs SET status = 'SUCCEEDED', report_sha256 = ?, "
            "report_path = ?, trace_path = ? WHERE run_id = ?",
            (
                exact_bytes_sha256(report_bytes),
                str(report_path),
                str(trace_path),
                run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LocalRealModelE2EError("could not finalize local qualification SQLite row")


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "file":
        raise LocalRealModelE2EError(f"local model input is not a file URI: {uri}")
    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    if os.name == "nt" and path.startswith("/") and len(path) >= 3 and path[2] == ":":
        path = path[1:]
    return Path(path)


def _observational_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(contents)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_file(path: Path, field: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise LocalRealModelE2EError(f"{field} is not a file: {resolved}")
    return resolved


def _require_directory(path: Path, field: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise LocalRealModelE2EError(f"{field} is not a directory: {resolved}")
    return resolved


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "LOCAL_REAL_MODEL_E2E_VERSION",
    "LocalFrameArtifact",
    "LocalModelObservation",
    "LocalRealModelE2EError",
    "LocalRealModelE2EReport",
    "LocalStorageObservation",
    "run_local_real_model_e2e",
]
