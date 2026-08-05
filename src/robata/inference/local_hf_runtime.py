"""Single-resident local Hugging Face vision runtime for bounded development use."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any


class LocalHuggingFaceRuntimeError(RuntimeError):
    """The local optional model runtime could not load or generate."""


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


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalHuggingFaceRuntimeError(f"{field} must be a positive integer")
    return value


__all__ = [
    "LocalHfGenerationObservation",
    "LocalHfLoadObservation",
    "LocalHuggingFaceRuntimeError",
    "LocalHuggingFaceVisionRuntime",
]
