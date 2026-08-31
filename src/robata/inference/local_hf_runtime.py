"""Single-resident local Hugging Face vision runtime for bounded development use."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any, Final


class LocalHuggingFaceRuntimeError(RuntimeError):
    """The local optional model runtime could not load or generate."""


LOCAL_HF_MAX_BATCH_REQUESTS: Final = 8
LOCAL_HF_MAX_IMAGES_PER_REQUEST: Final = 6
LOCAL_HF_MAX_VIDEO_FRAMES: Final = 32


def _model_execution_device(model: Any) -> Any:
    """Select a non-offloaded execution device for processor tensors."""

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for mapped_device in device_map.values():
            if mapped_device not in {"cpu", "disk"}:
                return f"cuda:{mapped_device}" if isinstance(mapped_device, int) else mapped_device
    return next(model.parameters()).device


@dataclass(frozen=True, slots=True)
class LocalHfLoadObservation:
    load_seconds: float
    gpu_name: str
    gpu_total_bytes: int
    gpu_free_before_bytes: int
    gpu_allocated_after_load_bytes: int


@dataclass(frozen=True, slots=True)
class LocalHfGenerationObservation:
    rendered_image_sizes: tuple[tuple[int, int], ...]
    prompt_tokens: int
    output_tokens: int
    generation_seconds: float
    gpu_peak_allocated_bytes: int
    output_text: str


@dataclass(frozen=True, slots=True)
class LocalHfVideoGenerationRequest:
    """Bounded complete-event input for Qwen3-VL's native ``video`` path.

    This is an internal local/benchmark contract. Frame bytes remain ordered and
    the source timeline is explicit so the processor can construct
    ``pixel_values_videos`` and ``video_grid_thw`` instead of treating frames as
    unrelated image messages.
    """

    video_payloads: tuple[bytes, ...]
    frame_indices: tuple[int, ...]
    frame_timestamps_seconds: tuple[float, ...]
    source_fps: float
    total_num_frames: int
    width: int
    height: int
    duration_seconds: float | None
    prompt: str
    max_new_tokens: int
    # The source interval is kept separate from the sampled frame timestamps.
    # Older local callers did not provide it, so ``None`` means that the
    # interval is derived from the first/last sampled frame.
    source_window_start_seconds: float | None = None
    source_window_end_seconds: float | None = None
    # Keep complete bounded-video generation the only normal route.  This flag
    # is an explicit opt-in for sparse diagnostic pairs, never for chunked
    # streaming input.
    allow_sparse_temporal_coverage: bool = False
    # Native Qwen output is usually a JSON object.  When enabled, the runtime
    # asks Transformers to stop after one strict complete object.  The raw
    # decoded suffix remains authoritative for the caller's parser.
    stop_after_first_complete_json_object: bool = False
    # Compatibility-only switch retained for old benchmark callers.  Content
    # identifiers are intentionally disabled in this runtime; setting it to
    # true fails before model loading rather than computing one.
    compute_frame_sha256: bool = False


@dataclass(frozen=True, slots=True)
class LocalHfVisualInputObservation:
    """Scalar observability for the exact native-video processor input."""

    rendered_frame_sizes: tuple[tuple[int, int], ...]
    processor_video_metadata: dict[str, Any]
    video_grid_thw: tuple[tuple[int, ...], ...]
    processor_tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    input_mode: str = "native_video"

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "rendered_frame_sizes": [list(size) for size in self.rendered_frame_sizes],
            "processor_video_metadata": dict(self.processor_video_metadata),
            "video_grid_thw": [list(row) for row in self.video_grid_thw],
            "processor_tensor_shapes": {
                name: list(shape) for name, shape in self.processor_tensor_shapes
            },
        }


@dataclass(frozen=True, slots=True)
class LocalHfVideoGenerationObservation:
    """Generation output plus native-video evidence needed for replay/audit."""

    rendered_frame_sizes: tuple[tuple[int, int], ...]
    frame_indices: tuple[int, ...]
    frame_timestamps_seconds: tuple[float, ...]
    source_fps: float
    total_num_frames: int
    width: int
    height: int
    duration_seconds: float | None
    prompt_tokens: int
    output_tokens: int
    generation_seconds: float
    gpu_peak_allocated_bytes: int
    output_text: str
    source_window_start_seconds: float | None = None
    source_window_end_seconds: float | None = None
    frame_sha256: tuple[str, ...] = ()
    input_mode: str = "native_video"
    # Optional scalar processor evidence.  It is deliberately after all
    # legacy fields so old positional construction remains valid.
    visual_input: LocalHfVisualInputObservation | None = None


@dataclass(frozen=True, slots=True)
class LocalHfBatchGenerationRequest:
    """One logical request participating in a single physical generation batch."""

    image_payloads: tuple[bytes, ...]
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class LocalHfBatchMemberObservation:
    """Per-member results without falsely attributed physical timing."""

    rendered_image_sizes: tuple[tuple[int, int], ...]
    prompt_tokens: int
    output_tokens: int
    output_text: str


@dataclass(frozen=True, slots=True)
class LocalHfBatchGenerationObservation:
    """Logical results plus telemetry shared by the one physical generate call."""

    members: tuple[LocalHfBatchMemberObservation, ...]
    physical_generation_seconds: float
    physical_gpu_peak_allocated_bytes: int


class LocalHuggingFaceVisionRuntime:
    """Keep exactly one quantized image-text model resident on one local GPU."""

    def __init__(
        self,
        *,
        model_directory: Path,
        offload_directory: Path,
        max_image_side: int = 448,
        gpu_weight_memory_gib: int = 4,
        cpu_weight_memory_gib: int = 1,
    ) -> None:
        self._model_directory = Path(model_directory).resolve()
        self._offload_directory = Path(offload_directory).resolve()
        self._max_image_side = _positive_int(max_image_side, "max_image_side")
        self._gpu_weight_memory_gib = _positive_int(gpu_weight_memory_gib, "gpu_weight_memory_gib")
        self._cpu_weight_memory_gib = _positive_int(cpu_weight_memory_gib, "cpu_weight_memory_gib")
        if not self._model_directory.is_dir():
            raise LocalHuggingFaceRuntimeError(
                f"model_directory is not a directory: {self._model_directory}"
            )
        self._lock = RLock()
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._image_module: Any | None = None
        self._load_observation: LocalHfLoadObservation | None = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def load_observation(self) -> LocalHfLoadObservation:
        with self._lock:
            if self._load_observation is None:
                raise LocalHuggingFaceRuntimeError("model is not loaded")
            return self._load_observation

    def load(self) -> LocalHfLoadObservation:
        """Load once with NF4 double quantization and BF16 compute."""

        with self._lock:
            if self._load_observation is not None:
                return self._load_observation
            try:
                torch = import_module("torch")
                transformers = import_module("transformers")
                image_module = import_module("PIL.Image")
            except ImportError as error:
                raise LocalHuggingFaceRuntimeError(
                    "local model runtime requires torch, transformers, accelerate, "
                    "bitsandbytes, safetensors, and Pillow"
                ) from error
            if not bool(torch.cuda.is_available()):
                raise LocalHuggingFaceRuntimeError("CUDA is required for the 4-bit local runtime")
            self._offload_directory.mkdir(parents=True, exist_ok=True)
            torch.cuda.empty_cache()
            gpu_free_before, gpu_total = torch.cuda.mem_get_info()
            gpu_name = str(torch.cuda.get_device_name(0))
            quantization = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            started = time.perf_counter()
            processor = transformers.AutoProcessor.from_pretrained(
                self._model_directory,
                local_files_only=True,
                trust_remote_code=True,
            )
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                self._model_directory,
                local_files_only=True,
                trust_remote_code=True,
                quantization_config=quantization,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory={
                    0: f"{self._gpu_weight_memory_gib}GiB",
                    "cpu": f"{self._cpu_weight_memory_gib}GiB",
                },
                offload_folder=str(self._offload_directory),
                low_cpu_mem_usage=True,
            )
            observation = LocalHfLoadObservation(
                load_seconds=float(time.perf_counter() - started),
                gpu_name=gpu_name,
                gpu_total_bytes=int(gpu_total),
                gpu_free_before_bytes=int(gpu_free_before),
                gpu_allocated_after_load_bytes=int(torch.cuda.memory_allocated()),
            )
            self._torch = torch
            self._processor = processor
            self._model = model
            self._image_module = image_module
            self._load_observation = observation
            return observation

    def generate(
        self,
        *,
        image_payloads: Sequence[bytes],
        prompt: str,
        max_new_tokens: int,
    ) -> LocalHfGenerationObservation:
        """Generate once; calls are serialized because one GPU owns one resident model."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise LocalHuggingFaceRuntimeError("prompt must be nonempty")
        _positive_int(max_new_tokens, "max_new_tokens")
        if not image_payloads:
            raise LocalHuggingFaceRuntimeError("at least one image payload is required")
        with self._lock:
            load = self.load()
            del load
            assert self._torch is not None
            assert self._processor is not None
            assert self._model is not None
            assert self._image_module is not None
            torch = self._torch
            processor = self._processor
            model = self._model
            images: list[Any] = []
            rendered_sizes: list[tuple[int, int]] = []
            try:
                for index, payload in enumerate(image_payloads):
                    if not isinstance(payload, bytes) or not payload:
                        raise LocalHuggingFaceRuntimeError(
                            "every image payload must be nonempty bytes"
                        )
                    try:
                        with self._image_module.open(BytesIO(payload)) as source_image:
                            image = source_image.convert("RGB")
                    except (OSError, ValueError) as error:
                        raise LocalHuggingFaceRuntimeError(
                            f"image payload {index} is not a decodable image"
                        ) from error
                    image.thumbnail((self._max_image_side, self._max_image_side))
                    images.append(image)
                    rendered_sizes.append((int(image.width), int(image.height)))
                content: list[dict[str, object]] = [
                    {"type": "image", "image": image} for image in images
                ]
                content.append({"type": "text", "text": prompt})
                text = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(text=[text], images=images, return_tensors="pt")
                device = _model_execution_device(model)
                inputs = {
                    key: value.to(device) if callable(getattr(value, "to", None)) else value
                    for key, value in inputs.items()
                }
                prompt_tokens = int(inputs["input_ids"].shape[1])
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                generation_seconds = float(time.perf_counter() - started)
                generated_only = generated[:, prompt_tokens:]
                output_tokens = int(generated_only.shape[1])
                output_text = str(
                    processor.batch_decode(generated_only, skip_special_tokens=True)[0]
                ).strip()
                if not output_text:
                    raise LocalHuggingFaceRuntimeError("local model returned an empty output")
                return LocalHfGenerationObservation(
                    rendered_image_sizes=tuple(rendered_sizes),
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    generation_seconds=generation_seconds,
                    gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    output_text=output_text,
                )
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if callable(close):
                        close()

    def generate_video(
        self,
        *,
        request: LocalHfVideoGenerationRequest,
    ) -> LocalHfVideoGenerationObservation:
        """Generate one bounded complete event through Qwen's native video path.

        Unlike :meth:`generate`, this method emits one ``video`` multimodal token
        and passes source ``fps``/``frames_indices`` metadata to the Qwen3-VL
        processor. It intentionally does not expose a streaming/chunked interface:
        callers must materialize the complete bounded action window first.
        """

        _validate_video_request(request)
        with self._lock:
            load = self.load()
            del load
            assert self._torch is not None
            assert self._processor is not None
            assert self._model is not None
            assert self._image_module is not None
            torch = self._torch
            processor = self._processor
            model = self._model
            images: list[Any] = []
            rendered_sizes: list[tuple[int, int]] = []
            try:
                for index, payload in enumerate(request.video_payloads):
                    try:
                        with self._image_module.open(BytesIO(payload)) as source_image:
                            image = source_image.convert("RGB")
                    except (OSError, ValueError) as error:
                        raise LocalHuggingFaceRuntimeError(
                            f"video frame payload {index} is not decodable"
                        ) from error
                    image.thumbnail((self._max_image_side, self._max_image_side))
                    images.append(image)
                    rendered_sizes.append((int(image.width), int(image.height)))

                content: list[dict[str, object]] = [
                    {"type": "video", "video": images},
                    {"type": "text", "text": request.prompt},
                ]
                text = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                metadata: dict[str, Any] = {
                    "total_num_frames": request.total_num_frames,
                    "fps": float(request.source_fps),
                    "width": request.width,
                    "height": request.height,
                    "frames_indices": list(request.frame_indices),
                }
                if request.duration_seconds is not None:
                    metadata["duration"] = float(request.duration_seconds)
                inputs = processor(
                    text=[text],
                    videos=[images],
                    video_metadata=[metadata],
                    do_sample_frames=False,
                    return_metadata=True,
                    return_tensors="pt",
                )
                # ``video_metadata`` is processor evidence, not a model forward
                # argument. Keep it out of generate while retaining it in the
                # request/observation lineage above.
                model_inputs = {
                    key: value for key, value in inputs.items() if key != "video_metadata"
                }
                if "pixel_values_videos" not in model_inputs:
                    raise LocalHuggingFaceRuntimeError(
                        "native video processor did not return pixel_values_videos"
                    )
                if "video_grid_thw" not in model_inputs:
                    raise LocalHuggingFaceRuntimeError(
                        "native video processor did not return video_grid_thw"
                    )
                visual_input = LocalHfVisualInputObservation(
                    rendered_frame_sizes=tuple(rendered_sizes),
                    processor_video_metadata=dict(metadata),
                    video_grid_thw=_video_grid_rows(model_inputs["video_grid_thw"]),
                    processor_tensor_shapes=_processor_tensor_shapes(model_inputs),
                )
                device = _model_execution_device(model)
                model_inputs = {
                    key: value.to(device) if callable(getattr(value, "to", None)) else value
                    for key, value in model_inputs.items()
                }
                prompt_tokens = int(model_inputs["input_ids"].shape[1])
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": request.max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                }
                if request.stop_after_first_complete_json_object:
                    generation_kwargs["stopping_criteria"] = [
                        _FirstCompleteJsonObjectStoppingCriterion(
                            processor=processor,
                            prompt_tokens=prompt_tokens,
                            torch_module=torch,
                        )
                    ]
                with torch.inference_mode():
                    generated = model.generate(
                        **model_inputs,
                        **generation_kwargs,
                    )
                generation_seconds = float(time.perf_counter() - started)
                generated_only = generated[:, prompt_tokens:]
                output_tokens = int(generated_only.shape[1])
                output_text = str(
                    processor.batch_decode(generated_only, skip_special_tokens=True)[0]
                ).strip()
                if not output_text:
                    raise LocalHuggingFaceRuntimeError("local model returned an empty output")
                source_start, source_end = _resolved_source_window(request)
                return LocalHfVideoGenerationObservation(
                    rendered_frame_sizes=tuple(rendered_sizes),
                    frame_indices=request.frame_indices,
                    frame_timestamps_seconds=request.frame_timestamps_seconds,
                    source_fps=request.source_fps,
                    total_num_frames=request.total_num_frames,
                    width=request.width,
                    height=request.height,
                    duration_seconds=request.duration_seconds,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    generation_seconds=generation_seconds,
                    gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    output_text=output_text,
                    source_window_start_seconds=source_start,
                    source_window_end_seconds=source_end,
                    frame_sha256=(),
                    visual_input=visual_input,
                )
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if callable(close):
                        close()

    def generate_batch(
        self,
        *,
        requests: Sequence[LocalHfBatchGenerationRequest],
    ) -> LocalHfBatchGenerationObservation:
        """Generate a small logical batch with one physical model call."""

        normalized_requests = _validated_batch_requests(requests)
        max_new_tokens = normalized_requests[0].max_new_tokens
        with self._lock:
            load = self.load()
            del load
            assert self._torch is not None
            assert self._processor is not None
            assert self._model is not None
            assert self._image_module is not None
            torch = self._torch
            processor = self._processor
            model = self._model
            images: list[Any] = []
            rendered_sizes_by_member: list[tuple[tuple[int, int], ...]] = []
            texts: list[str] = []
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise LocalHuggingFaceRuntimeError(
                    "processor must expose a tokenizer for batched generation"
                )
            original_padding_side = getattr(tokenizer, "padding_side", None)
            if not isinstance(original_padding_side, str) or not original_padding_side:
                raise LocalHuggingFaceRuntimeError(
                    "processor tokenizer must expose a nonempty padding_side for batched generation"
                )
            try:
                tokenizer.padding_side = "left"
                for member_index, request in enumerate(normalized_requests):
                    member_images: list[Any] = []
                    member_rendered_sizes: list[tuple[int, int]] = []
                    for image_index, payload in enumerate(request.image_payloads):
                        try:
                            with self._image_module.open(BytesIO(payload)) as source_image:
                                image = source_image.convert("RGB")
                        except (OSError, ValueError) as error:
                            raise LocalHuggingFaceRuntimeError(
                                "batch member "
                                f"{member_index} image payload {image_index} is not decodable"
                            ) from error
                        images.append(image)
                        member_images.append(image)
                        image.thumbnail((self._max_image_side, self._max_image_side))
                        member_rendered_sizes.append((int(image.width), int(image.height)))
                    content: list[dict[str, object]] = [
                        {"type": "image", "image": image} for image in member_images
                    ]
                    content.append({"type": "text", "text": request.prompt})
                    texts.append(
                        str(
                            processor.apply_chat_template(
                                [{"role": "user", "content": content}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                    )
                    rendered_sizes_by_member.append(tuple(member_rendered_sizes))

                inputs = processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                device = _model_execution_device(model)
                inputs = {
                    key: value.to(device) if callable(getattr(value, "to", None)) else value
                    for key, value in inputs.items()
                }
                prompt_token_counts, physical_prompt_width = _batch_prompt_token_counts(
                    inputs,
                    expected_members=len(normalized_requests),
                )
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                physical_generation_seconds = float(time.perf_counter() - started)
                generated_only = generated[:, physical_prompt_width:]
                _validate_generated_batch_size(
                    generated_only,
                    expected_members=len(normalized_requests),
                )
                decoded = tuple(
                    str(value).strip()
                    for value in processor.batch_decode(
                        generated_only,
                        skip_special_tokens=True,
                    )
                )
                if len(decoded) != len(normalized_requests):
                    raise LocalHuggingFaceRuntimeError(
                        "processor returned a different number of decoded batch members"
                    )
                for member_index, output_text in enumerate(decoded):
                    if not output_text:
                        raise LocalHuggingFaceRuntimeError(
                            f"local model returned an empty output for batch member {member_index}"
                        )
                output_token_counts = _batch_output_token_counts(
                    generated_only,
                    expected_members=len(normalized_requests),
                    pad_token_id=_pad_token_id(processor=processor, model=model),
                )
                members = tuple(
                    LocalHfBatchMemberObservation(
                        rendered_image_sizes=rendered_sizes_by_member[index],
                        prompt_tokens=prompt_token_counts[index],
                        output_tokens=output_token_counts[index],
                        output_text=decoded[index],
                    )
                    for index in range(len(normalized_requests))
                )
                return LocalHfBatchGenerationObservation(
                    members=members,
                    physical_generation_seconds=physical_generation_seconds,
                    physical_gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                )
            finally:
                tokenizer.padding_side = original_padding_side
                for image in images:
                    close = getattr(image, "close", None)
                    if callable(close):
                        close()

    def close(self) -> None:
        """Release the resident model when a one-shot caller is done."""

        with self._lock:
            model = self._model
            processor = self._processor
            self._model = None
            self._processor = None
            self._image_module = None
            self._load_observation = None
            torch = self._torch
            self._torch = None
            del model, processor
            if torch is not None:
                torch.cuda.empty_cache()


def _validated_batch_requests(
    requests: Sequence[LocalHfBatchGenerationRequest],
) -> tuple[LocalHfBatchGenerationRequest, ...]:
    try:
        normalized_candidates = tuple(requests)
    except TypeError as error:
        raise LocalHuggingFaceRuntimeError("batch requests must be a finite sequence") from error
    if not normalized_candidates:
        raise LocalHuggingFaceRuntimeError("at least one batch request is required")
    if len(normalized_candidates) > LOCAL_HF_MAX_BATCH_REQUESTS:
        raise LocalHuggingFaceRuntimeError(
            f"batch request count must not exceed {LOCAL_HF_MAX_BATCH_REQUESTS}"
        )

    normalized: list[LocalHfBatchGenerationRequest] = []
    expected_max_new_tokens: int | None = None
    for member_index, request in enumerate(normalized_candidates):
        if not isinstance(request, LocalHfBatchGenerationRequest):
            raise LocalHuggingFaceRuntimeError(
                f"batch member {member_index} must be a LocalHfBatchGenerationRequest"
            )
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise LocalHuggingFaceRuntimeError(
                f"batch member {member_index} prompt must be nonempty"
            )
        max_new_tokens = _positive_int(
            request.max_new_tokens,
            f"batch member {member_index} max_new_tokens",
        )
        image_payloads = tuple(request.image_payloads)
        if not image_payloads:
            raise LocalHuggingFaceRuntimeError(
                f"batch member {member_index} requires at least one image payload"
            )
        if len(image_payloads) > LOCAL_HF_MAX_IMAGES_PER_REQUEST:
            raise LocalHuggingFaceRuntimeError(
                "batch member "
                f"{member_index} image count must not exceed {LOCAL_HF_MAX_IMAGES_PER_REQUEST}"
            )
        for image_index, payload in enumerate(image_payloads):
            if not isinstance(payload, bytes) or not payload:
                raise LocalHuggingFaceRuntimeError(
                    "batch member "
                    f"{member_index} image payload {image_index} must be nonempty bytes"
                )
        if expected_max_new_tokens is None:
            expected_max_new_tokens = max_new_tokens
        elif max_new_tokens != expected_max_new_tokens:
            raise LocalHuggingFaceRuntimeError("all batch members must use the same max_new_tokens")
        normalized.append(
            LocalHfBatchGenerationRequest(
                image_payloads=image_payloads,
                prompt=request.prompt,
                max_new_tokens=max_new_tokens,
            )
        )
    return tuple(normalized)


def _batch_prompt_token_counts(
    inputs: dict[str, Any],
    *,
    expected_members: int,
) -> tuple[tuple[int, ...], int]:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise LocalHuggingFaceRuntimeError("processor batch is missing input_ids")
    input_shape: Any = getattr(input_ids, "shape", ())
    if len(input_shape) != 2 or int(input_shape[0]) != expected_members:
        raise LocalHuggingFaceRuntimeError("processor returned an invalid input_ids batch shape")
    physical_prompt_width = int(input_shape[1])
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        raise LocalHuggingFaceRuntimeError(
            "processor batch is missing attention_mask for non-padding prompt accounting"
        )
    attention_shape: Any = getattr(attention_mask, "shape", ())
    if tuple(int(value) for value in attention_shape) != (
        expected_members,
        physical_prompt_width,
    ):
        raise LocalHuggingFaceRuntimeError("processor returned an invalid attention_mask shape")
    counts = tuple(_tensor_row_sum(attention_mask[index]) for index in range(expected_members))
    if any(count <= 0 for count in counts):
        raise LocalHuggingFaceRuntimeError("processor returned an empty prompt member")
    return counts, physical_prompt_width


def _validate_generated_batch_size(generated: Any, *, expected_members: int) -> None:
    shape: Any = getattr(generated, "shape", ())
    if len(shape) != 2 or int(shape[0]) != expected_members:
        raise LocalHuggingFaceRuntimeError("model returned an invalid generated batch shape")


def _batch_output_token_counts(
    generated_only: Any,
    *,
    expected_members: int,
    pad_token_id: int | None,
) -> tuple[int, ...]:
    if pad_token_id is None:
        return tuple(int(generated_only[index].shape[0]) for index in range(expected_members))
    return tuple(
        _tensor_row_sum(generated_only[index] != pad_token_id) for index in range(expected_members)
    )


def _tensor_row_sum(row: Any) -> int:
    value = row.sum()
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def _pad_token_id(*, processor: Any, model: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    candidates = (
        getattr(tokenizer, "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _strict_complete_json_object(value: str) -> bool:
    """Return whether *value* is exactly one strict JSON object.

    The native stopping hook must not stop on prose containing an embedded
    object.  Duplicate keys and non-standard constants are rejected as well;
    the regular post-hoc parser remains authoritative for malformed output.
    """

    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate.startswith("{"):
        return False

    def reject_constant(token: str) -> Any:
        raise ValueError(token)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = item
        return result

    try:
        decoded = json.loads(
            candidate,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(decoded, dict)


class _FirstCompleteJsonObjectStoppingCriterion:
    """Transformers stopping criterion for one complete native JSON object."""

    def __init__(self, *, processor: Any, prompt_tokens: int, torch_module: Any) -> None:
        self._processor = processor
        self._prompt_tokens = _positive_int(prompt_tokens, "prompt_tokens")
        self._torch = torch_module

    def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> Any:
        batch_size = self._batch_size(input_ids)
        if batch_size is None:
            return False
        try:
            generated_only = input_ids[:, self._prompt_tokens :]
            decoded = self._processor.batch_decode(
                generated_only,
                skip_special_tokens=True,
            )
            if (
                not isinstance(decoded, Sequence)
                or isinstance(decoded, (str, bytes, bytearray))
                or len(decoded) != batch_size
            ):
                complete = [False] * batch_size
            else:
                complete = [_strict_complete_json_object(str(item)) for item in decoded]
        except Exception:
            # A stopping hook is diagnostic only.  If a tokenizer/fake cannot
            # decode at an intermediate step, leave generation running.
            complete = [False] * batch_size
        try:
            kwargs: dict[str, Any] = {"dtype": self._torch.bool}
            device = getattr(input_ids, "device", None)
            if device is not None:
                kwargs["device"] = device
            return self._torch.tensor(complete, **kwargs)
        except Exception:
            return complete[0] if len(complete) == 1 else all(complete)

    @staticmethod
    def _batch_size(input_ids: Any) -> int | None:
        try:
            shape = input_ids.shape
            value = int(shape[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return value if value > 0 else None


def _processor_tensor_shapes(inputs: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Extract scalar tensor shapes without requiring torch at import time."""

    rows: list[tuple[str, tuple[int, ...]]] = []
    for key, value in inputs.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        try:
            normalized = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            continue
        if any(dimension < 0 for dimension in normalized):
            continue
        rows.append((str(key), normalized))
    return tuple(rows)


def _video_grid_rows(value: Any) -> tuple[tuple[int, ...], ...]:
    """Normalize a processor ``video_grid_thw`` tensor/list to JSON-safe rows."""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
        raise LocalHuggingFaceRuntimeError("video_grid_thw must be a two-dimensional sequence")
    result: list[tuple[int, ...]] = []
    for row in candidate:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise LocalHuggingFaceRuntimeError("video_grid_thw must contain rows")
        try:
            normalized = tuple(int(item) for item in row)
        except (TypeError, ValueError) as error:
            raise LocalHuggingFaceRuntimeError("video_grid_thw values must be integers") from error
        if not normalized or any(item <= 0 for item in normalized):
            raise LocalHuggingFaceRuntimeError("video_grid_thw values must be positive")
        result.append(normalized)
    if not result:
        raise LocalHuggingFaceRuntimeError("video_grid_thw must contain at least one row")
    return tuple(result)


def _resolved_source_window(
    request: LocalHfVideoGenerationRequest,
) -> tuple[float, float]:
    start = (
        float(request.source_window_start_seconds)
        if request.source_window_start_seconds is not None
        else float(request.frame_timestamps_seconds[0])
    )
    end = (
        float(request.source_window_end_seconds)
        if request.source_window_end_seconds is not None
        else float(
            request.duration_seconds
            if request.duration_seconds is not None
            and request.source_window_start_seconds is not None
            else request.frame_timestamps_seconds[-1]
        )
    )
    return start, end


def _validate_video_request(request: LocalHfVideoGenerationRequest) -> None:
    if not isinstance(request, LocalHfVideoGenerationRequest):
        raise LocalHuggingFaceRuntimeError("request must be LocalHfVideoGenerationRequest")
    try:
        payloads = tuple(request.video_payloads)
        frame_indices = tuple(request.frame_indices)
        frame_timestamps = tuple(request.frame_timestamps_seconds)
    except TypeError as error:
        raise LocalHuggingFaceRuntimeError(
            "native video payloads, indices, and timestamps must be finite sequences"
        ) from error
    count = len(payloads)
    if count <= 0:
        raise LocalHuggingFaceRuntimeError("native video requires at least one frame")
    if count > LOCAL_HF_MAX_VIDEO_FRAMES:
        raise LocalHuggingFaceRuntimeError(
            f"native video frame count must not exceed {LOCAL_HF_MAX_VIDEO_FRAMES}"
        )
    if len(frame_indices) != count:
        raise LocalHuggingFaceRuntimeError("frame_indices count must match video_payloads")
    if len(frame_timestamps) != count:
        raise LocalHuggingFaceRuntimeError(
            "frame_timestamps_seconds count must match video_payloads"
        )
    for index, payload in enumerate(payloads):
        if not isinstance(payload, bytes) or not payload:
            raise LocalHuggingFaceRuntimeError(
                f"native video frame payload {index} must be nonempty bytes"
            )
    if isinstance(request.total_num_frames, bool) or not isinstance(request.total_num_frames, int):
        raise LocalHuggingFaceRuntimeError("total_num_frames must be an integer")
    if request.total_num_frames <= 0:
        raise LocalHuggingFaceRuntimeError("total_num_frames must be positive")
    if isinstance(request.width, bool) or not isinstance(request.width, int) or request.width <= 0:
        raise LocalHuggingFaceRuntimeError("width must be a positive integer")
    if (
        isinstance(request.height, bool)
        or not isinstance(request.height, int)
        or request.height <= 0
    ):
        raise LocalHuggingFaceRuntimeError("height must be a positive integer")
    if isinstance(request.source_fps, bool) or not isinstance(request.source_fps, (int, float)):
        raise LocalHuggingFaceRuntimeError("source_fps must be a finite positive number")
    if not math.isfinite(float(request.source_fps)) or float(request.source_fps) <= 0:
        raise LocalHuggingFaceRuntimeError("source_fps must be a finite positive number")
    previous_index = -1
    previous_timestamp = -1.0
    tolerance = max(1e-3, 0.5 / float(request.source_fps))
    for index, (frame_index, timestamp) in enumerate(
        zip(frame_indices, frame_timestamps, strict=True)
    ):
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise LocalHuggingFaceRuntimeError(f"frame index {index} must be an integer")
        if frame_index < 0 or frame_index >= request.total_num_frames:
            raise LocalHuggingFaceRuntimeError(f"frame index {index} is outside source bounds")
        if frame_index <= previous_index:
            raise LocalHuggingFaceRuntimeError("frame_indices must be strictly increasing")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise LocalHuggingFaceRuntimeError(
                f"frame timestamp {index} must be a finite non-negative number"
            )
        if not math.isfinite(float(timestamp)) or float(timestamp) < 0:
            raise LocalHuggingFaceRuntimeError(
                f"frame timestamp {index} must be a finite non-negative number"
            )
        if float(timestamp) <= previous_timestamp:
            raise LocalHuggingFaceRuntimeError(
                "frame_timestamps_seconds must be strictly increasing"
            )
        expected = frame_index / float(request.source_fps)
        if abs(float(timestamp) - expected) > tolerance:
            raise LocalHuggingFaceRuntimeError(
                f"frame timestamp {index} does not match source fps/frame index"
            )
        previous_index = frame_index
        previous_timestamp = float(timestamp)
    if request.duration_seconds is not None:
        if isinstance(request.duration_seconds, bool) or not isinstance(
            request.duration_seconds, (int, float)
        ):
            raise LocalHuggingFaceRuntimeError("duration_seconds must be positive when provided")
        if (
            not math.isfinite(float(request.duration_seconds))
            or float(request.duration_seconds) <= 0
        ):
            raise LocalHuggingFaceRuntimeError("duration_seconds must be positive when provided")
    if not isinstance(request.allow_sparse_temporal_coverage, bool):
        raise LocalHuggingFaceRuntimeError("allow_sparse_temporal_coverage must be a boolean")
    if not isinstance(request.stop_after_first_complete_json_object, bool):
        raise LocalHuggingFaceRuntimeError(
            "stop_after_first_complete_json_object must be a boolean"
        )
    if not isinstance(request.compute_frame_sha256, bool):
        raise LocalHuggingFaceRuntimeError("compute_frame_sha256 must be a boolean")
    if request.compute_frame_sha256:
        raise LocalHuggingFaceRuntimeError(
            "content identifiers are disabled for native-video generation"
        )
    raw_start = request.source_window_start_seconds
    raw_end = request.source_window_end_seconds
    if raw_start is None:
        source_start = float(frame_timestamps[0])
    elif isinstance(raw_start, bool) or not isinstance(raw_start, (int, float)):
        raise LocalHuggingFaceRuntimeError(
            "source_window_start_seconds must be a finite non-negative number"
        )
    else:
        source_start = float(raw_start)
    if raw_end is None:
        source_end = float(
            request.duration_seconds
            if request.duration_seconds is not None
            and request.source_window_start_seconds is not None
            else frame_timestamps[-1]
        )
    elif isinstance(raw_end, bool) or not isinstance(raw_end, (int, float)):
        raise LocalHuggingFaceRuntimeError(
            "source_window_end_seconds must be a finite non-negative number"
        )
    else:
        source_end = float(raw_end)
    for value, field in (
        (source_start, "source_window_start_seconds"),
        (source_end, "source_window_end_seconds"),
    ):
        if not math.isfinite(value) or value < 0:
            raise LocalHuggingFaceRuntimeError(f"{field} must be finite and non-negative")
    if source_end < source_start:
        raise LocalHuggingFaceRuntimeError(
            "source_window_end_seconds must not precede source_window_start_seconds"
        )
    if (
        request.duration_seconds is not None
        and source_end > float(request.duration_seconds) + tolerance
    ):
        raise LocalHuggingFaceRuntimeError(
            "source_window_end_seconds must not exceed duration_seconds"
        )
    if not request.allow_sparse_temporal_coverage:
        if float(frame_timestamps[0]) > source_start + tolerance:
            raise LocalHuggingFaceRuntimeError(
                "sampled frames do not cover the start of the source window"
            )
        if float(frame_timestamps[-1]) < source_end - tolerance and not _covers_physical_video_tail(
            request,
            source_end=source_end,
            tolerance=tolerance,
        ):
            raise LocalHuggingFaceRuntimeError(
                "sampled frames do not cover the end of the source window"
            )
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise LocalHuggingFaceRuntimeError("prompt must be nonempty")
    _positive_int(request.max_new_tokens, "max_new_tokens")


def _covers_physical_video_tail(
    request: LocalHfVideoGenerationRequest,
    *,
    source_end: float,
    tolerance: float,
) -> bool:
    """Recognize the final decodable frame as coverage of its media tail.

    Frame timestamps name the instant at which a frame begins, while a video
    with ``N`` frames at ``fps`` conventionally has duration ``N / fps``.  A
    complete tail window therefore ends one frame interval after the last
    frame's timestamp.  Permit that exact physical endpoint only when the
    request contains the final source frame and its declared duration agrees
    with the supplied frame timeline.  Ordinary incomplete windows retain the
    strict sampled-end coverage check above.
    """

    if request.duration_seconds is None:
        return False
    duration = float(request.duration_seconds)
    if abs(source_end - duration) > tolerance:
        return False
    if request.frame_indices[-1] != request.total_num_frames - 1:
        return False
    nominal_duration = request.total_num_frames / float(request.source_fps)
    return abs(duration - nominal_duration) <= tolerance


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalHuggingFaceRuntimeError(f"{field} must be a positive integer")
    return value


__all__ = [
    "LOCAL_HF_MAX_BATCH_REQUESTS",
    "LOCAL_HF_MAX_IMAGES_PER_REQUEST",
    "LOCAL_HF_MAX_VIDEO_FRAMES",
    "LocalHfBatchGenerationObservation",
    "LocalHfBatchGenerationRequest",
    "LocalHfBatchMemberObservation",
    "LocalHfGenerationObservation",
    "LocalHfLoadObservation",
    "LocalHfVideoGenerationObservation",
    "LocalHfVideoGenerationRequest",
    "LocalHfVisualInputObservation",
    "LocalHuggingFaceRuntimeError",
    "LocalHuggingFaceVisionRuntime",
]
