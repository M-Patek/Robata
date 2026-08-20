"""Native Mage video/codec inference runtime.

The runtime keeps model weights resident, but deliberately never retains or
persists hidden, KV, or recurrent decoder state.  Those values may exist only
inside one ``model.generate`` call.  Durable input and result handling lives in
:mod:`robata.inference.mage_video_endpoint`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from pathlib import Path, PurePosixPath
from threading import Condition, Lock, RLock
from typing import Any, Final, cast

from robata.contracts.hashing import semantic_sha256
from robata.inference.device_execution_guard import (
    DeviceExecutionGuard,
    DeviceExecutionGuardBusy,
    DeviceExecutionGuardError,
    ExclusiveFileDeviceGuard,
)
from robata.inference.mage_native_codec import (
    MageNativeCodecError,
    require_mage_codec_dependencies,
)

MAGE_VIDEO_RUNTIME_IDENTITY_VERSION: Final = "mage-video-runtime-identity-v1"
MAGE_VIDEO_GENERATION_TELEMETRY_VERSION: Final = "mage-video-generation-telemetry-v3"
MAGE_VIDEO_CODEC_CACHE_BINDING_VERSION: Final = "mage-video-codec-cache-binding-v1"
MAGE_VIDEO_TRADITIONAL_CODEC_CACHE_BINDING_VERSION: Final = (
    "mage-video-traditional-codec-cache-binding-v1"
)
_LOGGER = logging.getLogger(__name__)
_PROVIDER_CACHE_BINDING_LOCK = RLock()
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
class MageVideoCodecCacheBinding:
    """Internal binding to one already verified native-codec cache directory.

    This object is an endpoint/runtime hand-off, not a published wire contract and
    not an authoritative replay artifact. Its only purpose is to make the exact
    cache directory selected by strict admission explicit to the resident runtime.
    """

    source_path: Path
    provider_cache_directory: Path
    binding_version: str = field(default=MAGE_VIDEO_CODEC_CACHE_BINDING_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be pathlib.Path")
        if not isinstance(self.provider_cache_directory, Path):
            raise TypeError("provider_cache_directory must be pathlib.Path")
        source = self.source_path.expanduser().resolve()
        directory = self.provider_cache_directory.expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"codec cache binding source is not a file: {source}")
        if not directory.is_dir():
            raise ValueError(f"codec cache binding directory is not a directory: {directory}")
        for relative_name in ("meta.json", "src_patch_position.npy"):
            if not (directory / relative_name).is_file():
                raise ValueError(
                    "codec cache binding directory lacks a required provider asset: "
                    f"{relative_name}"
                )
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "provider_cache_directory", directory)


@dataclass(frozen=True, slots=True)
class MageVideoExactCodecCacheAsset:
    """One exact provider output file carried into runtime admission."""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str):
            raise TypeError("relative_path must be a string")
        relative = PurePosixPath(self.relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("relative_path must be a safe relative POSIX path")
        if not isinstance(self.byte_count, int) or isinstance(self.byte_count, bool):
            raise TypeError("byte_count must be an integer")
        if self.byte_count <= 0:
            raise ValueError("byte_count must be positive")
        _require_sha256_digest(self.sha256, "asset sha256")


@dataclass(frozen=True, slots=True)
class MageVideoTraditionalCodecCacheBinding:
    """Exact traditional-codec replay binding admitted by a verified manifest.

    Unlike :class:`MageVideoCodecCacheBinding`, which remains the DCVC Provider V2
    hand-off, this additive binding carries the exact traditional provider,
    toolchain, configuration, source, and output-asset identities. The resident
    runtime re-verifies every asset and calls only Mage's result loader; it never
    invokes ``cv-preinfer`` for a bound replay.
    """

    source_path: Path
    provider_cache_directory: Path
    codec_engine: str
    codec_config_sha256: str
    checkpoint_manifest_sha256: str
    codec_policy_sha256: str
    provider_identity_sha256: str
    toolchain_identity_sha256: str
    effective_config_sha256: str
    entry_semantic_sha256: str
    asset_set_sha256: str
    assets: tuple[MageVideoExactCodecCacheAsset, ...]
    binding_version: str = field(
        default=MAGE_VIDEO_TRADITIONAL_CODEC_CACHE_BINDING_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be pathlib.Path")
        if not isinstance(self.provider_cache_directory, Path):
            raise TypeError("provider_cache_directory must be pathlib.Path")
        if self.codec_engine not in {"hevc", "cv-preinfer"}:
            raise ValueError("traditional codec binding engine must be 'hevc' or 'cv-preinfer'")
        for field_name in (
            "codec_config_sha256",
            "checkpoint_manifest_sha256",
            "codec_policy_sha256",
            "provider_identity_sha256",
            "toolchain_identity_sha256",
            "effective_config_sha256",
            "entry_semantic_sha256",
            "asset_set_sha256",
        ):
            _require_sha256_digest(getattr(self, field_name), field_name)
        if not isinstance(self.assets, tuple) or not self.assets:
            raise ValueError("traditional codec binding assets must be a nonempty tuple")
        if not all(isinstance(asset, MageVideoExactCodecCacheAsset) for asset in self.assets):
            raise TypeError("traditional codec binding assets have an invalid item")
        paths = tuple(asset.relative_path for asset in self.assets)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("traditional codec binding assets must be unique and sorted")
        if not {"meta.json", "src_patch_position.npy"}.issubset(paths):
            raise ValueError("traditional codec binding lacks required Mage provider assets")

        unresolved_source = self.source_path.expanduser()
        unresolved_directory = self.provider_cache_directory.expanduser()
        if unresolved_source.is_symlink():
            raise ValueError("traditional codec binding source must not be a symlink")
        if unresolved_directory.is_symlink():
            raise ValueError("traditional codec binding directory must not be a symlink")
        source = unresolved_source.resolve()
        directory = unresolved_directory.resolve()
        if not source.is_file():
            raise ValueError(f"traditional codec binding source is not a file: {source}")
        if not directory.is_dir():
            raise ValueError(f"traditional codec binding directory is not a directory: {directory}")
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "provider_cache_directory", directory)

    def verify_exact_assets(self) -> None:
        """Fail closed if the provider directory no longer matches its asset manifest."""

        _verify_traditional_codec_cache_assets(
            directory=self.provider_cache_directory,
            expected_assets=self.assets,
            expected_asset_set_sha256=self.asset_set_sha256,
        )


type MageVideoExactCodecCacheBinding = (
    MageVideoCodecCacheBinding | MageVideoTraditionalCodecCacheBinding
)


@dataclass(frozen=True, slots=True)
class MageVideoLoadObservation:
    """One immutable observation from loading the resident model."""

    load_seconds: float
    execution_device: str
    runtime_identity: MageVideoRuntimeIdentity = field(default_factory=MageVideoRuntimeIdentity)


@dataclass(frozen=True, slots=True)
class MageVideoGenerationTelemetry:
    """Versioned, non-wire timing breakdown for one native generation request.

    Monotonic timestamps are process-local observations. They are deliberately
    excluded from endpoint v2 responses, result artifacts, and inference
    identity; only the optional operational telemetry sink may retain them.
    """

    request_started_monotonic_seconds: float
    request_completed_monotonic_seconds: float
    processor_started_monotonic_seconds: float
    processor_completed_monotonic_seconds: float
    input_materialization_started_monotonic_seconds: float
    input_materialization_completed_monotonic_seconds: float
    generation_started_monotonic_seconds: float
    first_output_token_monotonic_seconds: float | None
    generation_completed_monotonic_seconds: float
    decode_started_monotonic_seconds: float
    decode_completed_monotonic_seconds: float
    processor_lock_wait_seconds: float
    processor_seconds: float
    generation_lock_wait_seconds: float
    input_materialization_seconds: float
    generate_seconds: float
    decode_seconds: float
    total_request_seconds: float
    time_to_first_token_seconds: float | None
    output_tokens_per_second: float | None
    telemetry_version: str = field(
        default=MAGE_VIDEO_GENERATION_TELEMETRY_VERSION,
        init=False,
    )


class _MageVideoFirstTokenStoppingCriterion:
    """Observe the first decoded token without copying token tensors to CPU.

    Transformers invokes stopping criteria after appending each generated
    token. Returning ``False`` preserves generation behavior while the
    monotonic callback provides low-overhead TTFT instrumentation.
    """

    def __init__(self, *, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.first_output_token_monotonic_seconds: float | None = None

    def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
        if self.first_output_token_monotonic_seconds is None:
            self.first_output_token_monotonic_seconds = self._clock()
        return False


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
        codec_cache_root: Path | None = None,
        load_profile: MageVideoLoadProfile | str | None = None,
        runtime_identity: MageVideoRuntimeIdentity | None = None,
        codec_dependency_checker: CodecDependencyChecker | None = None,
        shared_device_guard_file: Path | None = None,
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
        self._codec_cache_root = (
            Path(codec_cache_root).expanduser().resolve() if codec_cache_root is not None else None
        )
        if self._codec_cache_root is not None and not self._codec_cache_root.is_dir():
            raise MageVideoRuntimeError(
                f"codec_cache_root is not an existing directory: {self._codec_cache_root}"
            )
        self._codec_dependency_checker = (
            codec_dependency_checker or require_mage_video_codec_dependencies
        )
        self._runtime_identity = _resolve_runtime_identity(
            load_profile=load_profile,
            declared_identity=runtime_identity,
        )
        try:
            self._shared_device_guard: DeviceExecutionGuard | None = (
                ExclusiveFileDeviceGuard(shared_device_guard_file)
                if shared_device_guard_file is not None
                else None
            )
        except DeviceExecutionGuardError as error:
            raise MageVideoRuntimeError("shared device guard configuration is invalid") from error
        self._shared_device_guard_file = (
            self._shared_device_guard.path
            if isinstance(self._shared_device_guard, ExclusiveFileDeviceGuard)
            else None
        )

        # Do not use a single in-process lock across codec preparation and GPU generation.
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
    def shared_device_guard_file(self) -> Path | None:
        """Return the operational cross-process guard path, if configured.

        The path is scheduling state only and deliberately does not alter
        :attr:`runtime_identity` or any durable inference identity.
        """

        return self._shared_device_guard_file

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
        codec_cache_binding: MageVideoExactCodecCacheBinding | None = None,
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

        native_codec_config = _normalise_native_codec_config(
            codec_config,
            self._model_directory,
            codec_cache_root=self._codec_cache_root,
        )
        exact_cache_binding = _validate_codec_cache_binding(
            binding=codec_cache_binding,
            paths=paths,
            requested_codec_config=codec_config,
            native_codec_config=native_codec_config,
            configured_cache_root=self._codec_cache_root,
        )
        # A verified traditional binding replays provider output through Mage's
        # own loader and therefore must not require or execute the external
        # ``cv-preinfer`` tool. Unbound requests and the existing DCVC Provider
        # V2 binding retain their original dependency diagnostics unchanged.
        if not isinstance(exact_cache_binding, MageVideoTraditionalCodecCacheBinding):
            self._codec_dependency_checker(native_codec_config, self._model_directory)

        def prepare_inputs(processor: Any, text: str) -> Mapping[str, Any]:
            with _bind_exact_provider_cache_entry(
                processor=processor,
                binding=exact_cache_binding,
            ):
                return cast(
                    Mapping[str, Any],
                    processor(
                        text=[text],
                        videos=[str(path) for path in paths],
                        video_backend="codec",
                        codec_config=native_codec_config,
                        max_pixels=int(native_codec_config["max_pixels"]),
                        return_tensors="pt",
                        padding=True,
                    ),
                )

        return self._generate_with_processor(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_video_count=len(paths),
            prepare_inputs=prepare_inputs,
        )

    def generate_fixed_frames(
        self,
        *,
        frames: Sequence[Any],
        prompt: str,
        max_new_tokens: int,
    ) -> MageVideoGenerationObservation:
        """Generate from one already decoded, fixed ordered frame sequence.

        This is an explicit diagnostic/control path. It invokes Mage's native
        ``video_backend='frames'`` processor without codec preparation, hidden
        resampling, recurrent state, or fallback to the codec path. Durable frame
        identity and timestamps remain the caller's responsibility; the runtime
        only consumes the verified in-memory images for one generation call.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise MageVideoRuntimeError("prompt must be nonempty")
        _positive_int(max_new_tokens, "max_new_tokens")
        ordered_frames = tuple(frames)
        if not ordered_frames:
            raise MageVideoRuntimeError("fixed-frame generation requires at least one frame")
        if any(frame is None for frame in ordered_frames):
            raise MageVideoRuntimeError("fixed-frame generation received a null frame")

        def prepare_inputs(processor: Any, text: str) -> Mapping[str, Any]:
            return cast(
                Mapping[str, Any],
                processor(
                    text=[text],
                    videos=[list(ordered_frames)],
                    video_backend="frames",
                    num_frames=len(ordered_frames),
                    max_frames=len(ordered_frames),
                    return_tensors="pt",
                    padding=True,
                ),
            )

        return self._generate_with_processor(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_video_count=1,
            prepare_inputs=prepare_inputs,
        )

    def _generate_with_processor(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        input_video_count: int,
        prepare_inputs: Callable[[Any, str], Mapping[str, Any]],
    ) -> MageVideoGenerationObservation:
        resident = self._acquire_resident_lease()
        request_started = time.perf_counter()
        processor_lock_wait_seconds = 0.0
        processor_seconds = 0.0
        generation_lock_wait_seconds = 0.0
        input_materialization_seconds = 0.0
        generation_seconds = 0.0
        decode_seconds = 0.0

        try:
            # Prompt templating and visual preprocessing share processor state.
            # Keep this region narrow so the next preparation can overlap an
            # earlier GPU generation, but never another processor call.
            lock_wait_started = time.perf_counter()
            with self._processor_lock:
                processor_lock_wait_seconds += time.perf_counter() - lock_wait_started
                processor_started = time.perf_counter()
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = resident.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = prepare_inputs(resident.processor, text)
                processor_completed = time.perf_counter()
                processor_seconds += processor_completed - processor_started

            # The model remains single-flight. Input device materialization is
            # intentionally inside this region: it may allocate GPU tensors and
            # must not race model execution on a memory-constrained worker.
            lock_wait_started = time.perf_counter()
            with self._generation_lock:
                generation_lock_wait_seconds += time.perf_counter() - lock_wait_started
                device_guard = self._shared_device_guard
                guard = device_guard.hold() if device_guard is not None else _null_device_guard()
                try:
                    with guard:
                        materialization_started = time.perf_counter()
                        device = _model_device(resident.model)
                        materialized_inputs = _move_inputs_to_device(inputs, device)
                        pixel_values = materialized_inputs.get("pixel_values")
                        model_dtype = getattr(resident.model, "dtype", None)
                        if pixel_values is not None and model_dtype is not None:
                            move = getattr(pixel_values, "to", None)
                            if callable(move):
                                materialized_inputs["pixel_values"] = move(model_dtype)
                        materialization_completed = time.perf_counter()
                        input_materialization_seconds = (
                            materialization_completed - materialization_started
                        )
                        prompt_tokens = _token_count(
                            materialized_inputs.get("input_ids"), "input_ids"
                        )
                        first_token_criterion = _MageVideoFirstTokenStoppingCriterion(
                            clock=time.perf_counter
                        )
                        generation_started_monotonic_seconds = time.perf_counter()
                        with resident.torch.inference_mode():
                            generated = resident.model.generate(
                                **materialized_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                use_cache=True,
                                stopping_criteria=[first_token_criterion],
                            )
                        generation_completed_monotonic_seconds = time.perf_counter()
                        generation_seconds = float(
                            generation_completed_monotonic_seconds
                            - generation_started_monotonic_seconds
                        )
                        generated_only = generated[:, prompt_tokens:]
                        output_tokens = _token_count(generated_only, "generated output")
                        move_generated = getattr(generated_only, "to", None)
                        if callable(move_generated):
                            generated_only = move_generated("cpu")
                        del generated
                        del materialized_inputs
                        del inputs
                        first_output_token_monotonic_seconds = (
                            first_token_criterion.first_output_token_monotonic_seconds
                        )
                        time_to_first_token_seconds = (
                            None
                            if first_output_token_monotonic_seconds is None
                            else max(
                                0.0,
                                first_output_token_monotonic_seconds
                                - generation_started_monotonic_seconds,
                            )
                        )
                        output_tokens_per_second = (
                            None
                            if generation_seconds <= 0.0
                            else float(output_tokens) / generation_seconds
                        )
                except DeviceExecutionGuardBusy as error:
                    raise MageVideoRuntimeError(
                        "shared accelerator is busy with DCVC preparation"
                    ) from error
                except DeviceExecutionGuardError as error:
                    raise MageVideoRuntimeError("shared device guard failed") from error

            # ``batch_decode`` is processor state. It must not race a subsequent
            # visual preparation, but intentionally happens after the GPU lane
            # has been released.
            lock_wait_started = time.perf_counter()
            with self._processor_lock:
                processor_lock_wait_seconds += time.perf_counter() - lock_wait_started
                decode_started = time.perf_counter()
                output_text = _decode_generated_text(resident.processor, generated_only)
                decode_completed = time.perf_counter()
                decode_seconds = decode_completed - decode_started
        except MageVideoRuntimeError:
            raise
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            _LOGGER.exception("native Mage video generation failed")
            raise MageVideoRuntimeError("native Mage video generation failed") from error
        finally:
            self._release_resident_lease()

        if not output_text:
            raise MageVideoRuntimeError("Mage video model returned an empty output")
        request_completed = time.perf_counter()
        total_request_seconds = request_completed - request_started
        return MageVideoGenerationObservation(
            input_video_count=input_video_count,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_seconds=generation_seconds,
            output_text=output_text,
            telemetry=MageVideoGenerationTelemetry(
                request_started_monotonic_seconds=request_started,
                request_completed_monotonic_seconds=request_completed,
                processor_started_monotonic_seconds=processor_started,
                processor_completed_monotonic_seconds=processor_completed,
                input_materialization_started_monotonic_seconds=materialization_started,
                input_materialization_completed_monotonic_seconds=materialization_completed,
                generation_started_monotonic_seconds=generation_started_monotonic_seconds,
                first_output_token_monotonic_seconds=first_output_token_monotonic_seconds,
                generation_completed_monotonic_seconds=generation_completed_monotonic_seconds,
                decode_started_monotonic_seconds=decode_started,
                decode_completed_monotonic_seconds=decode_completed,
                processor_lock_wait_seconds=processor_lock_wait_seconds,
                processor_seconds=processor_seconds,
                generation_lock_wait_seconds=generation_lock_wait_seconds,
                input_materialization_seconds=input_materialization_seconds,
                generate_seconds=generation_seconds,
                decode_seconds=decode_seconds,
                total_request_seconds=total_request_seconds,
                time_to_first_token_seconds=time_to_first_token_seconds,
                output_tokens_per_second=output_tokens_per_second,
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


def _validate_codec_cache_binding(
    *,
    binding: MageVideoExactCodecCacheBinding | None,
    paths: Sequence[Path],
    requested_codec_config: Mapping[str, Any],
    native_codec_config: Mapping[str, Any],
    configured_cache_root: Path | None,
) -> MageVideoExactCodecCacheBinding | None:
    if binding is None:
        return None
    if len(paths) != 1 or binding.source_path != paths[0]:
        raise MageVideoRuntimeError(
            "codec cache binding source does not match the generation input"
        )
    if configured_cache_root is None:
        raise MageVideoRuntimeError(
            "an exact provider cache binding requires a configured codec cache root"
        )
    if binding.provider_cache_directory.parent != configured_cache_root:
        raise MageVideoRuntimeError(
            "provider cache binding is outside the configured qualified cache root"
        )

    if isinstance(binding, MageVideoCodecCacheBinding):
        if native_codec_config.get("engine") != "dcvc-rt":
            raise MageVideoRuntimeError(
                "a DCVC Provider V2 cache binding is valid only for engine='dcvc-rt'"
            )
        return binding

    if not isinstance(binding, MageVideoTraditionalCodecCacheBinding):
        raise MageVideoRuntimeError("codec_cache_binding has an unsupported binding family")
    if native_codec_config.get("engine") != binding.codec_engine:
        raise MageVideoRuntimeError(
            "traditional codec cache binding engine does not match the generation policy"
        )
    if mage_video_codec_config_sha256(requested_codec_config) != binding.codec_config_sha256:
        raise MageVideoRuntimeError(
            "traditional codec cache binding configuration does not match the generation policy"
        )
    try:
        binding.verify_exact_assets()
    except (OSError, TypeError, ValueError) as error:
        raise MageVideoRuntimeError(
            "traditional codec cache assets changed before replay"
        ) from error
    return binding


@contextmanager
def _bind_exact_provider_cache_entry(
    *,
    processor: Any,
    binding: MageVideoExactCodecCacheBinding | None,
) -> Iterator[None]:
    """Route one processor call directly to a strictly admitted cache entry.

    Mage's remote-code processor computes its cache key inside a dynamically copied
    module. That copy does not include `preprocessor_config.json`, so recomputing
    the upstream locator can disagree with the externally qualified Provider V2
    locator. The checkpoint does not expose a cache-entry argument, therefore this
    adapter replaces only `process_codec_video` for the bounded processor call and
    invokes the checkpoint's own result loader on the exact admitted directory.
    There is no fallback to codec preparation on a bound request.
    """

    if binding is None:
        with _PROVIDER_CACHE_BINDING_LOCK:
            yield
        return

    codec_module = _mage_codec_processing_module(processor)
    original_process = getattr(codec_module, "process_codec_video", None)
    load_codec_result = getattr(codec_module, "_load_codec_result", None)
    if not callable(original_process) or not callable(load_codec_result):
        raise MageVideoRuntimeError(
            "qualified Mage codec module lacks the exact-cache consumption surface"
        )

    def consume_bound_entry(video_url: str, config: Any) -> Any:
        try:
            source = Path(video_url).expanduser().resolve()
        except (OSError, TypeError, ValueError) as error:
            raise MageVideoRuntimeError(
                "native processor supplied an invalid bound-cache source"
            ) from error
        if source != binding.source_path:
            raise MageVideoRuntimeError(
                "native processor requested a source other than the admitted cache binding"
            )
        expected_engine = (
            "dcvc-rt" if isinstance(binding, MageVideoCodecCacheBinding) else binding.codec_engine
        )
        if getattr(config, "engine", None) != expected_engine:
            raise MageVideoRuntimeError(
                "native processor changed engine for the admitted exact codec cache"
            )
        return load_codec_result(binding.provider_cache_directory)

    # `process_codec_video` is a module-level function imported inside the
    # checkpoint processor's call. Keep the replacement process-global but narrow,
    # and serialize it across any accidental second runtime in this interpreter.
    with _PROVIDER_CACHE_BINDING_LOCK:
        codec_module.process_codec_video = consume_bound_entry
        try:
            yield
        finally:
            codec_module.process_codec_video = original_process


def _mage_codec_processing_module(processor: Any) -> Any:
    module_name = type(processor).__module__
    package_name, separator, _leaf = module_name.rpartition(".")
    candidates = (
        (f"{package_name}.codec_video_processing_mage_vl", "codec_video_processing_mage_vl")
        if separator
        else ("codec_video_processing_mage_vl",)
    )
    last_error: ImportError | None = None
    for candidate in candidates:
        try:
            return import_module(candidate)
        except ImportError as error:
            last_error = error
    raise MageVideoRuntimeError(
        "could not import the qualified Mage codec processing module"
    ) from last_error


class _NullDeviceGuard:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None


def _null_device_guard() -> _NullDeviceGuard:
    return _NullDeviceGuard()


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


def mage_video_codec_config_sha256(codec_config: Mapping[str, Any]) -> str:
    """Hash the exact request-side codec configuration without operational cache paths."""

    if not isinstance(codec_config, Mapping):
        raise MageVideoRuntimeError("codec_config must be a mapping")
    if "cache_root" in codec_config:
        raise MageVideoRuntimeError(
            "codec_config identity must not include an operational cache_root"
        )
    try:
        return semantic_sha256(dict(codec_config))
    except (TypeError, ValueError) as error:
        raise MageVideoRuntimeError("codec_config is not canonical JSON compatible") from error


def _require_sha256_digest(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _verify_traditional_codec_cache_assets(
    *,
    directory: Path,
    expected_assets: tuple[MageVideoExactCodecCacheAsset, ...],
    expected_asset_set_sha256: str,
) -> None:
    observed_paths: list[str] = []
    observed_directories: list[str] = []
    for candidate in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError("traditional codec cache assets must not contain symlinks")
        relative = candidate.relative_to(directory).as_posix()
        if candidate.is_dir():
            observed_directories.append(relative)
        elif candidate.is_file():
            observed_paths.append(relative)
    expected_paths = [asset.relative_path for asset in expected_assets]
    expected_directories = sorted(
        {
            parent.as_posix()
            for asset in expected_assets
            for parent in PurePosixPath(asset.relative_path).parents
            if parent.as_posix() != "."
        }
    )
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise ValueError("traditional codec cache asset set changed")

    for asset in expected_assets:
        path = directory / Path(PurePosixPath(asset.relative_path))
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
        if byte_count != asset.byte_count or digest.hexdigest() != asset.sha256:
            raise ValueError(f"traditional codec cache asset bytes changed: {asset.relative_path}")

    observed_asset_set_sha256 = semantic_sha256(
        [
            {
                "relative_path": asset.relative_path,
                "byte_count": asset.byte_count,
                "sha256": asset.sha256,
            }
            for asset in expected_assets
        ]
    )
    if observed_asset_set_sha256 != expected_asset_set_sha256:
        raise ValueError("traditional codec cache asset-set identity changed")


def require_mage_video_codec_dependencies(
    codec_config: Mapping[str, Any],
    model_directory: Path,
) -> None:
    """Fail closed before model load using the shared native codec adapter."""

    try:
        require_mage_codec_dependencies(codec_config, model_directory=model_directory)
    except MageNativeCodecError as error:
        raise MageVideoCodecDependencyError(str(error)) from error


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
    *,
    codec_cache_root: Path | None = None,
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
    if codec_cache_root is not None:
        native["cache_root"] = codec_cache_root
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
    "MAGE_VIDEO_CODEC_CACHE_BINDING_VERSION",
    "MAGE_VIDEO_GENERATION_TELEMETRY_VERSION",
    "MAGE_VIDEO_LOAD_PROFILE_BITSANDBYTES_4BIT_NF4",
    "MAGE_VIDEO_LOAD_PROFILE_NATIVE_BF16",
    "MAGE_VIDEO_RUNTIME_IDENTITY_VERSION",
    "MAGE_VIDEO_TRADITIONAL_CODEC_CACHE_BINDING_VERSION",
    "MageVideoCodecCacheBinding",
    "MageVideoCodecDependencyError",
    "MageVideoExactCodecCacheAsset",
    "MageVideoExactCodecCacheBinding",
    "MageVideoGenerationObservation",
    "MageVideoGenerationTelemetry",
    "MageVideoLoadObservation",
    "MageVideoLoadProfile",
    "MageVideoRuntime",
    "MageVideoRuntimeError",
    "MageVideoRuntimeIdentity",
    "MageVideoTraditionalCodecCacheBinding",
    "mage_video_codec_config_sha256",
    "require_mage_video_codec_dependencies",
]
