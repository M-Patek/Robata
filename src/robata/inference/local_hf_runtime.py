"""Single-resident local Hugging Face vision runtime for bounded development use."""

from __future__ import annotations

import time
from collections.abc import Sequence
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
                device = next(model.parameters()).device
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
                device = next(model.parameters()).device
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


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalHuggingFaceRuntimeError(f"{field} must be a positive integer")
    return value


__all__ = [
    "LOCAL_HF_MAX_BATCH_REQUESTS",
    "LOCAL_HF_MAX_IMAGES_PER_REQUEST",
    "LocalHfBatchGenerationObservation",
    "LocalHfBatchGenerationRequest",
    "LocalHfBatchMemberObservation",
    "LocalHfGenerationObservation",
    "LocalHfLoadObservation",
    "LocalHuggingFaceRuntimeError",
    "LocalHuggingFaceVisionRuntime",
]
