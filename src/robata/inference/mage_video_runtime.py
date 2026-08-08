"""Native Mage video/codec inference runtime.

The runtime keeps model weights resident, but deliberately never retains or
persists hidden, KV, or recurrent decoder state.  Those values may exist only
inside one ``model.generate`` call.  Durable input and result handling lives in
:mod:`robata.inference.mage_video_endpoint`.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from threading import Condition, Lock, RLock
from typing import Any, Final

MAGE_VIDEO_RUNTIME_IDENTITY_VERSION: Final = "mage-video-runtime-identity-v1"
_LOGGER = logging.getLogger(__name__)


class MageVideoLoadProfile(StrEnum):
    """Versioned model-execution configurations supported by the Mage runtime."""

    NATIVE_BF16 = "native_bf16_v1"
    BITSANDBYTES_4BIT_NF4 = "bitsandbytes_4bit_nf4_v1"
    # Short alias for configuration surfaces that use the common BnB acronym.
    BNB_4BIT_NF4 = "bitsandbytes_4bit_nf4_v1"


MAGE_VIDEO_LOAD_PROFILE_NATIVE_BF16: Final = MageVideoLoadProfile.NATIVE_BF16
MAGE_VIDEO_LOAD_PROFILE_BITSANDBYTES_4BIT_NF4: Final = MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4


@dataclass(frozen=True, slots=True)
class MageVideoRuntimeIdentity:
    """Versioned runtime identity for one declared model-execution profile."""

    identity_version: str = MAGE_VIDEO_RUNTIME_IDENTITY_VERSION
    load_profile: MageVideoLoadProfile = MageVideoLoadProfile.NATIVE_BF16


class MageVideoRuntimeError(RuntimeError):
    """Mage's optional native video runtime could not load or generate."""


class MageVideoCodecDependencyError(MageVideoRuntimeError):
    """A selected codec preprocessing engine is unavailable on this host."""


@dataclass(frozen=True, slots=True)
class MageVideoLoadObservation:
    """One immutable observation from loading the resident model."""

    load_seconds: float
    execution_device: str
    runtime_identity: MageVideoRuntimeIdentity = field(default_factory=MageVideoRuntimeIdentity)


@dataclass(frozen=True, slots=True)
class MageVideoGenerationTelemetry:
    """Non-wire timing breakdown for one native generation request."""

    processor_lock_wait_seconds: float
    processor_seconds: float
    generation_lock_wait_seconds: float
    input_materialization_seconds: float
    generate_seconds: float
    decode_seconds: float
    total_request_seconds: float


@dataclass(frozen=True, slots=True)
class MageVideoGenerationObservation:
    """Durable-result-safe output from a native video generation call."""

    input_video_count: int
    prompt_tokens: int
    output_tokens: int
    generation_seconds: float
    output_text: str
    telemetry: MageVideoGenerationTelemetry | None = None


CodecDependencyChecker = Callable[[Mapping[str, Any], Path], None]


@dataclass(frozen=True, slots=True)
class _MageVideoResidentResources:
    """A short-lived lease over resident objects protected from ``close``."""

    torch: Any
    processor: Any
    model: Any


class MageVideoRuntime:
    """Run one Mage video decoder through its native codec processor path.

    ``video_paths`` is intentionally a sequence even though the current v2
    endpoint permits exactly one item. The boundary is therefore ready for
    independent camera encoders to feed one decoder in a later version without
    changing the runtime method shape.

    Runtime concurrency is deliberately split into three bounded regions:

    * one lifecycle condition protects loading, closing, and resident-object
      leases;
    * one processor lock serializes the mutable native processor and codec
      preparation; and
    * one generation lock serializes ``model.generate``.

    This permits a CPU codec preparation for request N+1 to overlap the GPU
    generation for request N, without ever allowing two generations or two
    processor calls to overlap. A lease remains held from preparation through
    decoding so ``close`` cannot free the model or processor mid-request.
    """

    def __init__(
        self,
        *,
        model_directory: Path,
        offload_directory: Path | None = None,
        load_profile: MageVideoLoadProfile | str | None = None,
        runtime_identity: MageVideoRuntimeIdentity | None = None,
        codec_dependency_checker: CodecDependencyChecker | None = None,
    ) -> None:
        self._model_directory = Path(model_directory).expanduser().resolve()
        if not self._model_directory.is_dir():
            raise MageVideoRuntimeError(
                f"model_directory is not a directory: {self._model_directory}"
            )
        self._offload_directory = (
            Path(offload_directory).expanduser().resolve()
            if offload_directory is not None
            else None
        )
        self._codec_dependency_checker = (
            codec_dependency_checker or require_mage_video_codec_dependencies
        )
        self._runtime_identity = _resolve_runtime_identity(
            load_profile=load_profile,
            declared_identity=runtime_identity,
        )

        # Do not use a single lock across codec preparation and GPU generation.
        # The condition protects resident lifetime only; the other two locks
        # independently bound the unsafe mutable provider surfaces.
        self._state_condition = Condition(RLock())
        self._processor_lock = Lock()
        self._generation_lock = Lock()
        self._loading = False
        self._closing = False
        self._active_operations = 0
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_observation: MageVideoLoadObservation | None = None

    @property
    def loaded(self) -> bool:
        """Whether model and processor weights are currently resident and usable."""

        with self._state_condition:
            return (
                not self._closing
                and self._torch is not None
                and self._processor is not None
                and self._model is not None
            )

    @property
    def runtime_identity(self) -> MageVideoRuntimeIdentity:
        """Return the configured, versioned model-execution identity."""

        return self._runtime_identity

    @property
    def load_profile(self) -> MageVideoLoadProfile:
        """Return the explicit profile that controls model-loading semantics."""

        return self._runtime_identity.load_profile

    @property
    def load_observation(self) -> MageVideoLoadObservation:
        """Return the immutable load observation after successful startup."""

        with self._state_condition:
            if self._closing or self._load_observation is None:
                raise MageVideoRuntimeError("model is not loaded")
            return self._load_observation

    def load(self) -> MageVideoLoadObservation:
        """Load the local checkpoint with the declared, versioned profile.

        The expensive provider imports/checkpoint load happen outside the
        lifecycle condition. Concurrent callers wait for the same load rather
        than creating another resident model, and ``close`` waits for an
        in-flight load before releasing the finished resident objects.
        """

        with self._state_condition:
            while self._closing:
                self._state_condition.wait()
            if self._load_observation is not None:
                return self._load_observation
            while self._loading:
                self._state_condition.wait()
                while self._closing:
                    self._state_condition.wait()
                if self._load_observation is not None:
                    return self._load_observation
            self._loading = True

        try:
            try:
                torch = import_module("torch")
                transformers = import_module("transformers")
            except ImportError as error:
                raise MageVideoRuntimeError(
                    "Mage video runtime requires torch, transformers, accelerate, and the local "
                    "Mage checkpoint remote-code files"
                ) from error

            if self._offload_directory is not None:
                try:
                    self._offload_directory.mkdir(parents=True, exist_ok=True)
                except OSError as error:
                    raise MageVideoRuntimeError(
                        f"could not create model offload directory: {self._offload_directory}"
                    ) from error

            model_kwargs = _build_model_load_kwargs(
                profile=self.load_profile,
                torch=torch,
                transformers=transformers,
                offload_directory=self._offload_directory,
            )
            started = time.perf_counter()
            try:
                processor = transformers.AutoProcessor.from_pretrained(
                    self._model_directory,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self._model_directory,
                    **model_kwargs,
                )
                evaluated_model = model.eval()
                if evaluated_model is not None:
                    model = evaluated_model
                _validate_loaded_model_execution(
                    model=model,
                    profile=self.load_profile,
                    torch=torch,
                )
                execution_device = str(_model_device(model))
            except MageVideoRuntimeError:
                raise
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise MageVideoRuntimeError("could not load the local Mage video model") from error

            observation = MageVideoLoadObservation(
                load_seconds=float(time.perf_counter() - started),
                execution_device=execution_device,
                runtime_identity=self._runtime_identity,
            )
        except BaseException:
            with self._state_condition:
                self._loading = False
                self._state_condition.notify_all()
            raise

        with self._state_condition:
            self._torch = torch
            self._processor = processor
            self._model = model
            self._load_observation = observation
            self._loading = False
            self._state_condition.notify_all()
            return observation

    def generate(
        self,
        *,
        video_paths: Sequence[Path | str],
        prompt: str,
        max_new_tokens: int,
        codec_config: Mapping[str, Any],
    ) -> MageVideoGenerationObservation:
        """Generate from durable video paths through ``video_backend='codec'``.

        Native codec preparation remains a processor call with
        ``videos=[path]`` and ``video_backend='codec'``. There is deliberately
        no image fallback. The declared preprocessing device is normalized into
        the native configuration before the processor sees it; it is never
        inferred from the model execution device or environment.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise MageVideoRuntimeError("prompt must be nonempty")
        _positive_int(max_new_tokens, "max_new_tokens")
        paths = _normalise_video_paths(video_paths)
        if len(paths) != 1:
            raise MageVideoRuntimeError(
                "Mage video runtime for the v2 endpoint accepts exactly one video path; "
                "the sequence interface is reserved for future multi-camera encoding"
            )
        for path in paths:
            if not path.is_file():
                raise MageVideoRuntimeError(f"video path is not a file: {path}")

        native_codec_config = _normalise_native_codec_config(codec_config, self._model_directory)
        # Validate host requirements before a costly model load. The actual
        # codec implementation still owns per-video cache and decoding errors.
        self._codec_dependency_checker(native_codec_config, self._model_directory)
        resident = self._acquire_resident_lease()
        request_started = time.perf_counter()
        processor_lock_wait_seconds = 0.0
        processor_seconds = 0.0
        generation_lock_wait_seconds = 0.0
        input_materialization_seconds = 0.0
        generation_seconds = 0.0
        decode_seconds = 0.0

        try:
            # The processor owns native codec invocation and mutable prompt
            # templating state. Keep this region narrow so the next CPU
            # preparation can overlap an earlier GPU generation, but not another
            # processor call.
            lock_wait_started = time.perf_counter()
            with self._processor_lock:
                processor_lock_wait_seconds += time.perf_counter() - lock_wait_started
                processor_started = time.perf_counter()
                content: list[dict[str, str]] = [{"type": "video"} for _ in paths]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                text = resident.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = resident.processor(
                    text=[text],
                    videos=[str(path) for path in paths],
                    video_backend="codec",
                    codec_config=native_codec_config,
                    max_pixels=int(native_codec_config["max_pixels"]),
                    return_tensors="pt",
                    padding=True,
                )
                processor_seconds += time.perf_counter() - processor_started

            # The model remains single-flight. Input device materialization is
            # intentionally inside this region: it may allocate GPU tensors and
            # must not race model execution on a memory-constrained worker.
            lock_wait_started = time.perf_counter()
            with self._generation_lock:
                generation_lock_wait_seconds += time.perf_counter() - lock_wait_started
                materialization_started = time.perf_counter()
                device = _model_device(resident.model)
                materialized_inputs = _move_inputs_to_device(inputs, device)
                pixel_values = materialized_inputs.get("pixel_values")
                model_dtype = getattr(resident.model, "dtype", None)
                if pixel_values is not None and model_dtype is not None:
                    move = getattr(pixel_values, "to", None)
                    if callable(move):
                        materialized_inputs["pixel_values"] = move(model_dtype)
                input_materialization_seconds = time.perf_counter() - materialization_started
                prompt_tokens = _token_count(materialized_inputs.get("input_ids"), "input_ids")
                started = time.perf_counter()
                # ``use_cache`` may create attention KV only for this call. It
                # is not stored on ``self`` or returned to the endpoint.
                with resident.torch.inference_mode():
                    generated = resident.model.generate(
                        **materialized_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                generation_seconds = float(time.perf_counter() - started)
                generated_only = generated[:, prompt_tokens:]
                output_tokens = _token_count(generated_only, "generated output")

            # ``batch_decode`` is also processor state. It must not race a
            # subsequent codec preparation, but it intentionally happens after
            # the GPU generation lock has been released.
            lock_wait_started = time.perf_counter()
            with self._processor_lock:
                processor_lock_wait_seconds += time.perf_counter() - lock_wait_started
                decode_started = time.perf_counter()
                output_text = _decode_generated_text(resident.processor, generated_only)
                decode_seconds = time.perf_counter() - decode_started
        except MageVideoRuntimeError:
            raise
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            _LOGGER.exception("native Mage video generation failed")
            raise MageVideoRuntimeError("native Mage video generation failed") from error
        finally:
            self._release_resident_lease()

        if not output_text:
            raise MageVideoRuntimeError("Mage video model returned an empty output")
        total_request_seconds = time.perf_counter() - request_started
        return MageVideoGenerationObservation(
            input_video_count=len(paths),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_seconds=generation_seconds,
            output_text=output_text,
            telemetry=MageVideoGenerationTelemetry(
                processor_lock_wait_seconds=processor_lock_wait_seconds,
                processor_seconds=processor_seconds,
                generation_lock_wait_seconds=generation_lock_wait_seconds,
                input_materialization_seconds=input_materialization_seconds,
                generate_seconds=generation_seconds,
                decode_seconds=decode_seconds,
                total_request_seconds=total_request_seconds,
            ),
        )

    def close(self) -> None:
        """Release resident weights without racing an active preparation/generation.

        New work is barred before waiting for active leases and an in-flight
        checkpoint load. Cleanup remains inside that lifecycle barrier, so a
        subsequent ``load`` cannot start while CUDA cache disposal is running.
        """

        with self._state_condition:
            while self._closing:
                self._state_condition.wait()
            self._closing = True
            while self._loading or self._active_operations:
                self._state_condition.wait()
            model = self._model
            processor = self._processor
            torch = self._torch
            self._model = None
            self._processor = None
            self._torch = None
            self._load_observation = None

        try:
            del model, processor
            cuda = getattr(torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        finally:
            with self._state_condition:
                self._closing = False
                self._state_condition.notify_all()

    def _acquire_resident_lease(self) -> _MageVideoResidentResources:
        """Load if needed, then protect resident objects until request completion."""

        while True:
            self.load()
            with self._state_condition:
                while self._closing:
                    self._state_condition.wait()
                if (
                    self._torch is not None
                    and self._processor is not None
                    and self._model is not None
                ):
                    self._active_operations += 1
                    return _MageVideoResidentResources(
                        torch=self._torch,
                        processor=self._processor,
                        model=self._model,
                    )
                # A close could have completed between ``load`` returning and
                # this lease acquisition. Reload against the newly empty state.

    def _release_resident_lease(self) -> None:
        with self._state_condition:
            if self._active_operations <= 0:
                raise MageVideoRuntimeError("Mage runtime resident lease underflow")
            self._active_operations -= 1
            self._state_condition.notify_all()


def _resolve_runtime_identity(
    *,
    load_profile: MageVideoLoadProfile | str | None,
    declared_identity: MageVideoRuntimeIdentity | None,
) -> MageVideoRuntimeIdentity:
    if declared_identity is not None and not isinstance(
        declared_identity, MageVideoRuntimeIdentity
    ):
        raise MageVideoRuntimeError(
            "runtime_identity must be a MageVideoRuntimeIdentity when it is declared"
        )
    if declared_identity is not None:
        _validate_runtime_identity(declared_identity)

    if load_profile is None:
        profile = (
            declared_identity.load_profile
            if declared_identity is not None
            else MageVideoLoadProfile.NATIVE_BF16
        )
    else:
        profile = _normalise_load_profile(load_profile)
    configured_identity = MageVideoRuntimeIdentity(load_profile=profile)

    if declared_identity is not None and declared_identity != configured_identity:
        raise MageVideoRuntimeError(
            "configured Mage load profile does not match the declared runtime identity"
        )
    return configured_identity


def _validate_runtime_identity(identity: MageVideoRuntimeIdentity) -> None:
    if identity.identity_version != MAGE_VIDEO_RUNTIME_IDENTITY_VERSION:
        raise MageVideoRuntimeError(
            f"declared Mage runtime identity version is unsupported: {identity.identity_version!r}"
        )
    if not isinstance(identity.load_profile, MageVideoLoadProfile):
        raise MageVideoRuntimeError("declared Mage runtime identity has an invalid load profile")


def _normalise_load_profile(load_profile: MageVideoLoadProfile | str) -> MageVideoLoadProfile:
    if isinstance(load_profile, MageVideoLoadProfile):
        return load_profile
    if isinstance(load_profile, str):
        aliases = {
            "native-bf16-v1": MageVideoLoadProfile.NATIVE_BF16,
            "bitsandbytes-4bit-nf4-v1": MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4,
            "bnb_4bit_nf4_v1": MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4,
        }
        canonical = aliases.get(load_profile, load_profile)
        try:
            return MageVideoLoadProfile(canonical)
        except ValueError as error:
            raise MageVideoRuntimeError(
                "load_profile must be 'native_bf16_v1' or 'bitsandbytes_4bit_nf4_v1'"
            ) from error
    raise MageVideoRuntimeError(
        "load_profile must be a MageVideoLoadProfile or one of its versioned string values"
    )


def _build_model_load_kwargs(
    *,
    profile: MageVideoLoadProfile,
    torch: Any,
    transformers: Any,
    offload_directory: Path | None,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": _bfloat16_dtype(torch),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if offload_directory is not None:
        model_kwargs["offload_folder"] = str(offload_directory)

    if profile is MageVideoLoadProfile.NATIVE_BF16:
        return model_kwargs
    if profile is MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4:
        model_kwargs["quantization_config"] = _build_nf4_quantization_config(
            torch=torch,
            transformers=transformers,
        )
        return model_kwargs
    raise MageVideoRuntimeError(f"unsupported Mage load profile: {profile!r}")


def _bfloat16_dtype(torch: Any) -> Any:
    dtype = getattr(torch, "bfloat16", None)
    if dtype is None:
        raise MageVideoRuntimeError(
            "the selected Mage load profile requires a PyTorch build with bfloat16 support"
        )
    return dtype


def _build_nf4_quantization_config(*, torch: Any, transformers: Any) -> Any:
    _require_cuda_for_nf4(torch)
    try:
        import_module("bitsandbytes")
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        raise MageVideoRuntimeError(
            "Mage load profile 'bitsandbytes_4bit_nf4_v1' requires a working bitsandbytes "
            "installation compatible with the active CUDA PyTorch build; install bitsandbytes "
            "and a CUDA-enabled PyTorch build."
        ) from error

    configuration_class = getattr(transformers, "BitsAndBytesConfig", None)
    if not callable(configuration_class):
        raise MageVideoRuntimeError(
            "Mage load profile 'bitsandbytes_4bit_nf4_v1' requires a Transformers release "
            "that provides BitsAndBytesConfig; upgrade transformers and bitsandbytes together."
        )
    try:
        return configuration_class(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_bfloat16_dtype(torch),
            bnb_4bit_use_double_quant=True,
        )
    except (
        ImportError,
        OSError,
        PackageNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise MageVideoRuntimeError(
            "could not configure Transformers BitsAndBytesConfig for "
            "'bitsandbytes_4bit_nf4_v1'; install compatible transformers and bitsandbytes "
            "packages, then verify the CUDA runtime."
        ) from error


def _require_cuda_for_nf4(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        raise MageVideoRuntimeError(
            "Mage load profile 'bitsandbytes_4bit_nf4_v1' requires CUDA; use "
            "'native_bf16_v1' only on a suitable BF16-capable production accelerator."
        )
    try:
        cuda_available = bool(is_available())
    except (AttributeError, RuntimeError) as error:
        raise MageVideoRuntimeError(
            "could not determine CUDA availability for Mage load profile "
            "'bitsandbytes_4bit_nf4_v1'; verify the CUDA PyTorch installation."
        ) from error
    if not cuda_available:
        raise MageVideoRuntimeError(
            "Mage load profile 'bitsandbytes_4bit_nf4_v1' requires CUDA; use "
            "'native_bf16_v1' only on a suitable BF16-capable production accelerator."
        )


def _validate_loaded_model_execution(
    *,
    model: Any,
    profile: MageVideoLoadProfile,
    torch: Any,
) -> None:
    expected_dtype = _bfloat16_dtype(torch)
    actual_dtype = getattr(model, "dtype", None)
    if not _same_dtype(actual_dtype, expected_dtype):
        raise _profile_mismatch(
            profile,
            "model dtype is not bfloat16",
        )

    quantization_config = _model_quantization_config(model)
    if profile is MageVideoLoadProfile.NATIVE_BF16:
        if quantization_config is not None:
            raise _profile_mismatch(
                profile,
                "native BF16 execution must not expose a quantization configuration",
            )
        if bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
            getattr(model, "is_loaded_in_8bit", False)
        ):
            raise _profile_mismatch(profile, "native BF16 execution reported quantized weights")
        return

    if profile is not MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4:
        raise MageVideoRuntimeError(f"unsupported Mage load profile: {profile!r}")
    if quantization_config is None:
        raise _profile_mismatch(
            profile,
            "quantized execution did not expose its BitsAndBytesConfig",
        )
    if not bool(_configuration_value(quantization_config, "load_in_4bit")):
        raise _profile_mismatch(profile, "BitsAndBytesConfig.load_in_4bit is not true")
    quantization_type = _configuration_value(quantization_config, "bnb_4bit_quant_type")
    if str(quantization_type).lower() != "nf4":
        raise _profile_mismatch(profile, "BitsAndBytesConfig quantization type is not NF4")
    compute_dtype = _configuration_value(quantization_config, "bnb_4bit_compute_dtype")
    if not _same_dtype(compute_dtype, expected_dtype):
        raise _profile_mismatch(profile, "BitsAndBytesConfig compute dtype is not bfloat16")
    if not bool(_configuration_value(quantization_config, "bnb_4bit_use_double_quant")):
        raise _profile_mismatch(profile, "BitsAndBytesConfig double quantization is not enabled")
    loaded_in_4bit = getattr(model, "is_loaded_in_4bit", None)
    if loaded_in_4bit is not None and not bool(loaded_in_4bit):
        raise _profile_mismatch(profile, "model did not report 4-bit loaded weights")


def _model_quantization_config(model: Any) -> Any | None:
    configuration = getattr(model, "config", None)
    quantization_config = _configuration_value(configuration, "quantization_config")
    if quantization_config is not None:
        return quantization_config
    return getattr(model, "quantization_config", None)


def _configuration_value(configuration: Any, field: str) -> Any | None:
    if isinstance(configuration, Mapping):
        return configuration.get(field)
    return getattr(configuration, field, None)


def _same_dtype(actual: Any, expected: Any) -> bool:
    return _dtype_name(actual) == _dtype_name(expected)


def _dtype_name(dtype: Any) -> str:
    return str(dtype).strip().lower().removeprefix("torch.")


def _profile_mismatch(profile: MageVideoLoadProfile, detail: str) -> MageVideoRuntimeError:
    return MageVideoRuntimeError(
        "loaded Mage model execution configuration does not match declared load profile "
        f"{profile.value!r}: {detail}"
    )


def require_mage_video_codec_dependencies(
    codec_config: Mapping[str, Any],
    model_directory: Path,
) -> None:
    """Fail fast with an engine-specific installation error.

    The native processor will execute the selected codec preprocessing engine
    later.  Checking its executable/package entrypoints here makes missing
    optional codec dependencies clear before model loading or generation.
    """

    engine = codec_config.get("engine")
    if engine in {"hevc", "cv-preinfer"}:
        executable = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
        executable_path = Path(executable).expanduser()
        if shutil.which(executable) is None and not executable_path.is_file():
            raise MageVideoCodecDependencyError(
                "Mage traditional codec preprocessing requires the 'cv-preinfer' executable "
                "provided by codec-video-prep. Install that dependency and put it on PATH, "
                "or set CV_PREINFER_BIN to the executable path."
            )
        return
    if engine == "dcvc-rt":
        dcvc = codec_config.get("dcvc")
        if not isinstance(dcvc, Mapping):
            raise MageVideoCodecDependencyError(
                "Mage neural codec preprocessing requires a mapping at codec_config['dcvc']"
            )
        package_directory = Path(
            str(dcvc.get("pkg_dir") or (Path(model_directory) / "neural_codec"))
        ).expanduser()
        generator = package_directory / "dcvc_readiness_gen.py"
        source_directory = package_directory / "DCVC" / "src"
        intra_checkpoint = Path(
            str(dcvc.get("intra_ckpt") or (package_directory / "dcvc_rt_intra.tar"))
        ).expanduser()
        inter_checkpoint = Path(
            str(dcvc.get("inter_ckpt") or (package_directory / "dcvc_rt_inter.tar"))
        ).expanduser()
        missing: list[str] = []
        if not generator.is_file():
            missing.append(str(generator))
        if not source_directory.is_dir():
            missing.append(str(source_directory))
        if not intra_checkpoint.is_file():
            missing.append(str(intra_checkpoint))
        if not inter_checkpoint.is_file():
            missing.append(str(inter_checkpoint))
        if missing:
            raise MageVideoCodecDependencyError(
                "Mage neural codec preprocessing is unavailable; missing " + ", ".join(missing)
            )
        return
    raise MageVideoCodecDependencyError(
        "codec_config.engine must be 'hevc', 'cv-preinfer', or 'dcvc-rt'"
    )


def _normalise_video_paths(video_paths: Sequence[Path | str]) -> tuple[Path, ...]:
    if isinstance(video_paths, (str, bytes)) or not isinstance(video_paths, Sequence):
        raise MageVideoRuntimeError("video_paths must be a nonempty sequence of paths")
    paths = tuple(Path(path).expanduser().resolve() for path in video_paths)
    if not paths:
        raise MageVideoRuntimeError("at least one video path is required")
    return paths


def _normalise_native_codec_config(
    codec_config: Mapping[str, Any],
    model_directory: Path,
) -> dict[str, Any]:
    """Validate an identity-bound native codec configuration.

    ``preprocess_device`` is a required v2 policy field. It is deliberately
    independent of the decoder execution device: local 4-bit workers can run
    DCVC preparation on CPU while the decoder remains on CUDA, whereas a
    production BF16 worker may explicitly choose CUDA preprocessing.
    """

    if not isinstance(codec_config, Mapping):
        raise MageVideoRuntimeError("codec_config must be a mapping")
    native = dict(codec_config)
    engine = native.get("engine")
    if not isinstance(engine, str):
        raise MageVideoRuntimeError("codec_config.engine must be a string")
    if engine not in {"hevc", "cv-preinfer", "dcvc-rt"}:
        raise MageVideoRuntimeError(
            "codec_config.engine must be 'hevc', 'cv-preinfer', or 'dcvc-rt'"
        )
    preprocess_device = native.get("preprocess_device")
    if not isinstance(preprocess_device, str) or preprocess_device not in {"cpu", "cuda"}:
        raise MageVideoRuntimeError(
            "codec_config.preprocess_device must explicitly be 'cpu' or 'cuda'"
        )
    # ``preprocess_device`` is an endpoint/runtime policy input, not a field
    # accepted by Mage's public ``CodecConfig`` constructor. Consume it here
    # and bind it to the engine-specific configuration below.
    native.pop("preprocess_device", None)
    max_pixels = native.get("max_pixels")
    native["max_pixels"] = _positive_int(max_pixels, "codec_config.max_pixels")
    if engine == "dcvc-rt":
        raw_dcvc = native.get("dcvc", {})
        if not isinstance(raw_dcvc, Mapping):
            raise MageVideoRuntimeError("codec_config.dcvc must be a mapping for the neural codec")
        dcvc = dict(raw_dcvc)
        existing_device = dcvc.get("device")
        if existing_device is not None and existing_device != preprocess_device:
            raise MageVideoRuntimeError(
                "codec_config.dcvc.device must match codec_config.preprocess_device"
            )
        # This is intentionally not a model-device default. The endpoint's
        # explicitly identity-bound policy owns the DCVC device selection.
        dcvc["device"] = preprocess_device
        dcvc.setdefault("pkg_dir", str(model_directory / "neural_codec"))
        native["dcvc"] = dcvc
    return native


def _move_inputs_to_device(inputs: Any, device: Any) -> dict[str, Any]:
    items = getattr(inputs, "items", None)
    if not callable(items):
        raise MageVideoRuntimeError("native processor did not return a mapping of model inputs")
    moved: dict[str, Any] = {}
    for key, value in items():
        move = getattr(value, "to", None)
        moved[str(key)] = move(device) if callable(move) else value
    return moved


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise MageVideoRuntimeError("loaded Mage model does not expose a device")
    try:
        return next(parameters()).device
    except (AttributeError, StopIteration, TypeError) as error:
        raise MageVideoRuntimeError("loaded Mage model does not expose a device") from error


def _token_count(value: Any, field: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise MageVideoRuntimeError(f"native processor returned invalid {field} token dimensions")
    try:
        count = int(shape[1])
    except (IndexError, TypeError, ValueError) as error:
        raise MageVideoRuntimeError(
            f"native processor returned invalid {field} token dimensions"
        ) from error
    if count < 0:
        raise MageVideoRuntimeError(f"native processor returned negative {field} token count")
    return count


def _decode_generated_text(processor: Any, generated_only: Any) -> str:
    batch_decode = getattr(processor, "batch_decode", None)
    if callable(batch_decode):
        decoded = batch_decode(generated_only, skip_special_tokens=True)
    else:
        tokenizer = getattr(processor, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise MageVideoRuntimeError("native processor cannot decode generated tokens")
        decoded = [decode(generated_only[0], skip_special_tokens=True)]
    if isinstance(decoded, (str, bytes)) or not isinstance(decoded, Sequence) or not decoded:
        raise MageVideoRuntimeError("native processor returned an invalid decoded response")
    return str(decoded[0]).strip()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MageVideoRuntimeError(f"{field} must be a positive integer")
    return value


__all__ = [
    "MAGE_VIDEO_LOAD_PROFILE_BITSANDBYTES_4BIT_NF4",
    "MAGE_VIDEO_LOAD_PROFILE_NATIVE_BF16",
    "MAGE_VIDEO_RUNTIME_IDENTITY_VERSION",
    "MageVideoCodecDependencyError",
    "MageVideoGenerationObservation",
    "MageVideoGenerationTelemetry",
    "MageVideoLoadObservation",
    "MageVideoLoadProfile",
    "MageVideoRuntime",
    "MageVideoRuntimeError",
    "MageVideoRuntimeIdentity",
    "require_mage_video_codec_dependencies",
]
