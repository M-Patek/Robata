"""Optional local Transformers backend for the benchmark-only WeMM experiment.

The rest of the WeMM experiment is intentionally runnable with fake vectors, so
the core benchmark does not acquire a hard dependency on ``torch`` or
``qwen-vl-utils``.  This module is the only place that imports those optional
packages.  It is used from a separate process and never from the production
retrieval service.

The backend follows the public WeMM model-card contract: render a multimodal
chat message, call ``model.embedding(**inputs)``, and retain the final
embedding (optionally Matryoshka-truncated and L2-normalised).  It records
processor observations such as frame count and ``video_grid_thw`` without
persisting pixels or content digests.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any


class WemmBackendUnavailable(RuntimeError):
    """Optional WeMM runtime dependencies or local model files are unavailable."""


# These are the public model-card bounds.  Keeping them in one place prevents
# the processor's unusually large Qwen3.5 defaults from silently expanding a
# benchmark input and makes the actual geometry auditable in the report.
WEMM_IMAGE_KWARGS: dict[str, int] = {
    "min_pixels": 64 * 32 * 32,
    "max_pixels": 8192 * 32 * 32,
}
WEMM_VIDEO_KWARGS: dict[str, int] = {
    "min_pixels": 4 * 32 * 32,
    "max_pixels": 256 * 32 * 32,
    "total_pixels": 8192 * 32 * 32,
    "sample_fps": 1,
    "fps": 1,
    "max_frames": 64,
}
# Explicit arms used by the bounded local experiment.  The backend accepts a
# generic positive batch size for fixture tests; this tuple documents the two
# supported trial settings without changing any production/default route.
WEMM_VIDEO_MICROBATCH_SIZES: tuple[int, int] = (2, 4)

# Transformers 5.x routes visual resize controls through modality-scoped
# ``images_kwargs``/``videos_kwargs``.  The per-item ``min_pixels`` fields above
# are retained for qwen-vl-utils/chat-template compatibility, but are ignored by
# a direct Qwen3VLProcessor call.  Keep the translated SizeDicts explicit so a
# direct frame-list path cannot silently fall back to the model card's enormous
# 4K+ defaults.
WEMM_IMAGE_SIZE: dict[str, int] = {
    "shortest_edge": WEMM_IMAGE_KWARGS["min_pixels"],
    "longest_edge": WEMM_IMAGE_KWARGS["max_pixels"],
}
WEMM_VIDEO_SIZE: dict[str, int] = {
    "shortest_edge": WEMM_VIDEO_KWARGS["min_pixels"],
    "longest_edge": WEMM_VIDEO_KWARGS["max_pixels"],
}


def _positive_pixel_override(value: object, *, field: str) -> int | None:
    """Validate an optional per-run pixel bound without touching the model."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer or None")
    return value


# ``transformers>=5`` validates VideoMetadata strictly.  Runner-only fields
# (for example source window/intervention annotations) belong in observations,
# not in the processor side channel.
_PROCESSOR_VIDEO_METADATA_KEYS = frozenset(
    {
        "total_num_frames",
        "fps",
        "width",
        "height",
        "duration",
        "video_backend",
        "frames_indices",
    }
)


@dataclass(frozen=True, slots=True)
class WemmInputObservation:
    """Auditable shape facts for one encoder call (no hashes or raw pixels)."""

    modality: str
    item_count: int
    frame_count: int | None
    embedding_dimension: int
    requested_dimension: int | None
    input_keys: tuple[str, ...]
    video_grid_thw: tuple[tuple[int, ...], ...] = ()
    elapsed_seconds: float = 0.0
    # The fields below are intentionally optional.  Existing serial/image/text
    # observations therefore retain their historical wire shape, while the
    # explicit video-microbatch probe can expose bounded, non-content telemetry.
    batch_size: int | None = None
    batch_index: int | None = None
    frame_counts: tuple[int, ...] = ()
    tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    phase_timings: tuple[tuple[str, float], ...] = ()
    # Non-default per-run video resize controls are retained as optional
    # telemetry.  Leaving these fields empty preserves the historical
    # singleton wire shape for callers using the default budget.
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_size: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modality": self.modality,
            "item_count": self.item_count,
            "frame_count": self.frame_count,
            "embedding_dimension": self.embedding_dimension,
            "requested_dimension": self.requested_dimension,
            "input_keys": list(self.input_keys),
            "video_grid_thw": [list(row) for row in self.video_grid_thw],
            "elapsed_seconds": self.elapsed_seconds,
        }
        # Do not add experimental keys to the long-standing singleton path.
        # ``encode_video_frames_batch`` sets all of these fields explicitly.
        if self.batch_size is not None:
            payload["batch_size"] = self.batch_size
            payload["batch"] = self.batch_size
        if self.batch_index is not None:
            payload["batch_index"] = self.batch_index
        if self.frame_counts:
            payload["frame_counts"] = list(self.frame_counts)
        if self.tensor_shapes:
            shapes = {name: list(shape) for name, shape in self.tensor_shapes}
            # ``processor_tensor_shapes`` is the project-wide name used by
            # existing visual traces; ``tensor_shapes`` is a compact alias for
            # callers that only need the microbatch probe telemetry.
            payload["processor_tensor_shapes"] = shapes
            payload["tensor_shapes"] = shapes
        if self.phase_timings:
            payload["phase_timings"] = {
                name: float(seconds) for name, seconds in self.phase_timings
            }
        if self.video_min_pixels is not None:
            payload["video_min_pixels"] = self.video_min_pixels
        if self.video_max_pixels is not None:
            payload["video_max_pixels"] = self.video_max_pixels
        if self.video_size:
            payload["video_size"] = {name: int(value) for name, value in self.video_size}
        return payload


def _optional_imports() -> tuple[Any, Any, Any, Any]:
    """Load optional runtime dependencies lazily with an actionable error."""

    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        transformers = importlib.import_module("transformers")
        # Transformers exposes many classes through a module-level lazy
        # ``__getattr__``.  Looking in ``vars(transformers)`` bypasses that
        # resolver and raises ``KeyError`` on otherwise valid installations.
        # Attribute access works for both lazy module exports and ordinary
        # test doubles while preserving the single optional-import boundary.
        auto_model = transformers.AutoModel
        auto_processor = transformers.AutoProcessor
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise WemmBackendUnavailable(
            "WeMM runtime requires torch and transformers; install them in the "
            "isolated benchmark environment"
        ) from exc
    return torch, functional, auto_model, auto_processor


def _mapping_from_config(config: object) -> Mapping[str, Any]:
    """Return a shallow mapping for a Transformers config when available."""

    if isinstance(config, Mapping):
        return config
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
        except Exception:
            value = None
        if isinstance(value, Mapping):
            return value
    try:
        value = vars(config)
    except TypeError:
        value = None
    return value if isinstance(value, Mapping) else {}


def _read_local_config(model_directory: Path) -> Mapping[str, Any]:
    """Read non-sensitive model metadata without loading weights."""

    path = model_directory / "config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _validate_local_safetensors(model_directory: Path) -> None:
    """Fail early on an empty/corrupt local shard without hashing its contents."""

    shards = sorted(model_directory.glob("*.safetensors"))
    if not shards:
        # Transformers will provide the usual missing-weight error for a
        # directory that genuinely uses another supported format.
        return
    for shard in shards:
        try:
            with shard.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated header length")
                header_size = int.from_bytes(prefix, "little")
                if header_size <= 0 or header_size > shard.stat().st_size - 8:
                    raise ValueError(f"invalid header length {header_size}")
                header = json.loads(handle.read(header_size).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise WemmBackendUnavailable(f"invalid local safetensors shard {shard}: {exc}") from exc
        if not isinstance(header, Mapping):
            raise WemmBackendUnavailable(
                f"invalid local safetensors shard {shard}: header is not an object"
            )


def _supported_dimensions(config: object) -> tuple[int, ...]:
    """Read the model-card Matryoshka dimensions, ignoring malformed entries."""

    values = _mapping_from_config(config).get("matryoshka_dimensions")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer > 0 and integer not in result:
            result.append(integer)
    return tuple(result)


def _configured_full_dimension(config: object) -> int | None:
    """Infer the untruncated embedding width from a model config."""

    mapping = _mapping_from_config(config)
    supported = _supported_dimensions(mapping)
    if supported:
        return max(supported)
    text_config = mapping.get("text_config")
    text_mapping = _mapping_from_config(text_config)
    for key in ("hidden_size", "out_hidden_size"):
        value = text_mapping.get(key, mapping.get(key))
        if isinstance(value, bool):
            continue
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer > 0:
            return integer
    return None


def _variant_from_config(config: object, model_directory: Path) -> str:
    """Infer the WeMM size for report identity without assuming a checkpoint size."""

    mapping = _mapping_from_config(config)
    full_dimension = _configured_full_dimension(mapping)
    by_dimension = {2048: "2B", 2560: "4B", 4096: "9B"}
    if full_dimension in by_dimension:
        return by_dimension[full_dimension]
    name = model_directory.name.casefold().replace("_", "-")
    for variant in ("2b", "4b", "9b"):
        if variant in name:
            return variant.upper()
    return "unknown"


def _without_side_channels(inputs: Any) -> Any:
    """Drop processor metadata that WeMM's ``embedding`` does not accept."""

    if not isinstance(inputs, Mapping) and not hasattr(inputs, "items"):
        return inputs
    return {key: value for key, value in inputs.items() if str(key) != "video_metadata"}


def _as_int(value: object) -> int | None:
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _shape_rows(value: object) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    if value and not isinstance(value[0], Sequence):
        value = [value]
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            continue
        converted = tuple(item for item in (_as_int(item) for item in row) if item is not None)
        if converted:
            rows.append(converted)
    return tuple(rows)


def _tensor_shape(value: object) -> tuple[int, ...] | None:
    """Return a bounded tensor/array shape without materialising its values."""

    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        values = tuple(int(component) for component in shape)
    except (TypeError, ValueError, OverflowError):
        return None
    if not values or any(component < 0 for component in values):
        return None
    return values


def _tensor_shapes(value: object) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Capture input tensor shapes as small, deterministic metadata rows."""

    if not isinstance(value, Mapping) and not hasattr(value, "items"):
        return ()
    rows: list[tuple[str, tuple[int, ...]]] = []
    try:
        entries = value.items()
    except Exception:
        return ()
    for key, child in entries:
        shape = _tensor_shape(child)
        if shape is not None:
            rows.append((str(key), shape))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


def _processor_video_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only keys accepted by Transformers' strict VideoMetadata type."""

    return {
        str(key): item for key, item in value.items() if str(key) in _PROCESSOR_VIDEO_METADATA_KEYS
    }


def _synthetic_frame_metadata(frames: Sequence[Any]) -> dict[str, Any]:
    """Build valid processor metadata when a direct frame caller has none.

    Transformers 5.13 requires a non-empty ``video_metadata`` side channel even
    for ``return_metadata=False`` direct-frame calls.  A one-Hz synthetic clock
    keeps such calls executable while preserving the real source metadata path
    used by the runner whenever it is available.
    """

    width = height = 1
    if frames:
        first = frames[0]
        size = getattr(first, "size", None)
        if (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes, bytearray))
            and len(size) >= 2
        ):
            try:
                width = max(1, int(size[0]))
                height = max(1, int(size[1]))
            except (TypeError, ValueError, OverflowError):
                pass
        shape = getattr(first, "shape", None)
        if (
            isinstance(shape, Sequence)
            and not isinstance(shape, (str, bytes, bytearray))
            and len(shape) >= 2
        ):
            try:
                height = max(1, int(shape[-2]))
                width = max(1, int(shape[-1]))
            except (TypeError, ValueError, OverflowError):
                pass
    count = len(frames)
    return {
        "total_num_frames": count,
        "fps": 1.0,
        "width": width,
        "height": height,
        "duration": float(count),
        "video_backend": "direct",
        "frames_indices": list(range(count)),
    }


def _model_device(model: Any) -> Any:
    """Choose a usable execution device for a regular or accelerate model."""

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for mapped in device_map.values():
            if mapped not in {"cpu", "disk", "meta"}:
                return f"cuda:{mapped}" if isinstance(mapped, int) else mapped
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cpu"


def _move_inputs(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _embedding_anchor_id(tokenizer: Any) -> int | None:
    """Return the dedicated embedding-token ID when the tokenizer exposes it."""

    if tokenizer is None:
        return None
    try:
        vocabulary = tokenizer.get_vocab()
    except Exception:
        vocabulary = None
    if isinstance(vocabulary, Mapping) and "<embedding>" not in vocabulary:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        value = convert("<embedding>")
    except Exception:
        return None
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return integer if integer >= 0 else None


def _assert_embedding_anchor(processor: Any, inputs: Any) -> None:
    """Fail closed if tokenization truncates the final ``<embedding>`` anchor.

    Some lightweight test doubles do not expose a tokenizer; in that case the
    check is intentionally skipped.  Real WeMM processors append the anchor
    through their tokenizer post-processor, so checking the final attended
    token catches accidental truncation/template regressions before inference.
    """

    tokenizer = getattr(processor, "tokenizer", None)
    anchor_id = _embedding_anchor_id(tokenizer)
    if anchor_id is None or (not isinstance(inputs, Mapping) and not hasattr(inputs, "get")):
        return
    ids = _shape_rows(inputs.get("input_ids"))
    if not ids:
        return
    masks = _shape_rows(inputs.get("attention_mask"))
    for row_index, row in enumerate(ids):
        if masks and row_index < len(masks):
            active = [value for value, mask in zip(row, masks[row_index], strict=False) if mask]
            final = active[-1] if active else None
        else:
            final = row[-1] if row else None
        if final != anchor_id:
            raise WemmBackendUnavailable(
                "WeMM tokenizer output is missing the final <embedding> anchor"
            )


def _tensor_to_rows(
    value: Any,
    *,
    torch: Any,
    functional: Any,
    dimension: int | None,
    supported_dimensions: Sequence[int] = (),
) -> tuple[tuple[float, ...], ...]:
    if not hasattr(value, "shape"):
        raise WemmBackendUnavailable("WeMM model.embedding returned no tensor")
    tensor = value.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] <= 0:
        raise WemmBackendUnavailable(
            f"WeMM embedding must be [batch,dim], got {tuple(tensor.shape)}"
        )
    if dimension is not None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise WemmBackendUnavailable(
                f"requested dimension {dimension!r} must be a positive integer"
            )
        if supported_dimensions and dimension not in supported_dimensions:
            supported = ", ".join(str(item) for item in supported_dimensions)
            raise WemmBackendUnavailable(
                f"requested dimension {dimension} is not supported; choose one of {supported}"
            )
        if dimension > tensor.shape[-1]:
            raise WemmBackendUnavailable(
                f"requested dimension {dimension!r} is invalid for {tensor.shape[-1]}"
            )
        tensor = tensor[..., :dimension]
    tensor = functional.normalize(tensor, dim=-1)
    rows = tensor.detach().cpu().tolist()
    result = tuple(tuple(float(item) for item in row) for row in rows)
    if any(not math.isfinite(item) for row in result for item in row):
        raise WemmBackendUnavailable("WeMM returned a non-finite embedding")
    return result


class WemmEmbeddingBackend:
    """Lazy local WeMM encoder used only by benchmark scripts.

    ``model_directory`` must be an already downloaded local snapshot.  The
    backend uses ``local_files_only=True`` by default so running an experiment
    cannot silently fetch a different revision.  ``video_min_pixels`` and
    ``video_max_pixels`` are optional per-run direct-video resize bounds; when
    omitted, the existing model-card bounds are used.
    """

    def __init__(
        self,
        model_directory: str | Path,
        *,
        device: str = "cuda",
        dimension: int | None = None,
        dtype: str = "bfloat16",
        local_files_only: bool = True,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
    ) -> None:
        self.model_directory = Path(model_directory).expanduser().resolve()
        self.device_name = device
        if dimension is not None and (
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer or None")
        self.dimension = dimension
        self.dtype_name = dtype
        self.local_files_only = local_files_only
        self._video_min_pixels_override = _positive_pixel_override(
            video_min_pixels, field="video_min_pixels"
        )
        self._video_max_pixels_override = _positive_pixel_override(
            video_max_pixels, field="video_max_pixels"
        )
        if self.video_min_pixels > self.video_max_pixels:
            raise ValueError("video_min_pixels must be <= video_max_pixels")
        self._config: Mapping[str, Any] = _read_local_config(self.model_directory)
        self._variant = _variant_from_config(self._config, self.model_directory)
        self._supported_dimensions = _supported_dimensions(self._config)
        self._torch: Any | None = None
        self._functional: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self.observations: list[WemmInputObservation] = []
        # Text prototypes are invariant for a resident backend and are often
        # requested once per recording by the production pre-annotation loop.
        # Keep a small in-process cache so reusing the model also reuses the
        # prototype vectors.  The key is the explicit ordered text tuple; no
        # identity digest or persisted cache is involved.
        self._text_prototype_cache: dict[tuple[str, ...], tuple[tuple[float, ...], ...]] = {}
        self._text_prototype_cache_hits = 0
        self._text_prototype_cache_misses = 0
        self._embedding_chat_template = (
            "sentence_transformers"
            if (
                self.model_directory / "additional_chat_templates" / "sentence_transformers.jinja"
            ).is_file()
            else None
        )

    @property
    def identity(self) -> str:
        return f"WeMM-Embedding-{self._variant}@{self.model_directory}"

    @property
    def variant(self) -> str:
        """Return the inferred size label (for example ``2B`` or ``4B``)."""

        return self._variant

    @property
    def supported_dimensions(self) -> tuple[int, ...]:
        """Return the model-card Matryoshka dimensions, if declared."""

        return self._supported_dimensions

    @property
    def video_min_pixels(self) -> int:
        """Effective shortest-edge pixel bound for direct video inputs."""

        return self._video_min_pixels_override or WEMM_VIDEO_KWARGS["min_pixels"]

    @property
    def video_max_pixels(self) -> int:
        """Effective longest-edge pixel bound for direct video inputs."""

        return self._video_max_pixels_override or WEMM_VIDEO_KWARGS["max_pixels"]

    @property
    def video_size(self) -> dict[str, int]:
        """Return a fresh Transformers ``SizeDict`` for this backend run."""

        return {
            "shortest_edge": self.video_min_pixels,
            "longest_edge": self.video_max_pixels,
        }

    @property
    def _video_budget_is_custom(self) -> bool:
        return (
            self.video_min_pixels != WEMM_VIDEO_KWARGS["min_pixels"]
            or self.video_max_pixels != WEMM_VIDEO_KWARGS["max_pixels"]
        )

    def _video_content_kwargs(self) -> dict[str, Any]:
        """Copy the native content hints with this run's pixel bounds."""

        values = dict(WEMM_VIDEO_KWARGS)
        values["min_pixels"] = self.video_min_pixels
        values["max_pixels"] = self.video_max_pixels
        return values

    def _video_observation_fields(
        self,
    ) -> tuple[int | None, int | None, tuple[tuple[str, int], ...]]:
        """Return additive resize telemetry only for non-default overrides."""

        if not self._video_budget_is_custom:
            return None, None, ()
        size = self.video_size
        return (
            self.video_min_pixels,
            self.video_max_pixels,
            tuple(sorted((str(key), int(value)) for key, value in size.items())),
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _apply_chat_template(self, messages: Any) -> Any:
        """Render the embedding template when the snapshot supplies one.

        Transformers 5.13 expands the ordinary Qwen video template with nested
        vision boundaries.  The released WeMM snapshot includes a dedicated
        Sentence-Transformers template with a bare video placeholder, matching
        the model-card embedding path.  Older processors without that template
        use their normal fallback.
        """

        assert self._processor is not None
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        if self._embedding_chat_template is not None:
            kwargs["chat_template"] = self._embedding_chat_template
        return self._processor.apply_chat_template(messages, **kwargs)

    def _load(self) -> None:
        if self.loaded:
            return
        if not self.model_directory.is_dir():
            raise WemmBackendUnavailable(
                f"WeMM model directory does not exist: {self.model_directory}"
            )
        _validate_local_safetensors(self.model_directory)
        if (
            self.dimension is not None
            and self._supported_dimensions
            and self.dimension not in self._supported_dimensions
        ):
            supported = ", ".join(str(item) for item in self._supported_dimensions)
            raise WemmBackendUnavailable(
                f"requested dimension {self.dimension} is not supported; choose one of {supported}"
            )
        self._torch, self._functional, auto_model, auto_processor = _optional_imports()
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        try:
            self._processor = auto_processor.from_pretrained(self.model_directory, **kwargs)
        except Exception as exc:
            raise WemmBackendUnavailable(f"failed to load WeMM processor: {exc}") from exc
        model_kwargs = dict(kwargs)
        dtype = getattr(self._torch, self.dtype_name, None)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        # device_map=auto avoids hard-coding a GPU ordinal and remains compatible
        # with a small laptop GPU plus CPU offload.  A caller can request CPU.
        if self.device_name != "cpu":
            model_kwargs["device_map"] = "auto"
        try:
            self._model = auto_model.from_pretrained(self.model_directory, **model_kwargs)
        except TypeError:
            # Transformers 5.x renamed torch_dtype to dtype in a few loaders.
            model_kwargs.pop("torch_dtype", None)
            if dtype is not None:
                model_kwargs["dtype"] = dtype
            try:
                self._model = auto_model.from_pretrained(self.model_directory, **model_kwargs)
            except Exception as exc:
                raise WemmBackendUnavailable(f"failed to load WeMM model: {exc}") from exc
        except Exception as exc:
            raise WemmBackendUnavailable(f"failed to load WeMM model: {exc}") from exc
        model_config = getattr(self._model, "config", None)
        if model_config is not None:
            self._config = _mapping_from_config(model_config)
            self._variant = _variant_from_config(self._config, self.model_directory)
            self._supported_dimensions = _supported_dimensions(self._config)
        if (
            self.dimension is not None
            and self._supported_dimensions
            and self.dimension not in self._supported_dimensions
        ):
            supported = ", ".join(str(item) for item in self._supported_dimensions)
            raise WemmBackendUnavailable(
                f"requested dimension {self.dimension} is not supported; choose one of {supported}"
            )
        if hasattr(self._model, "eval"):
            self._model.eval()
        self._device = _model_device(self._model)
        if self.device_name == "cpu" and hasattr(self._model, "to"):
            self._model.to("cpu")
            self._device = "cpu"

    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        modality: str,
        images: Any = None,
        videos: Any = None,
        frame_count: int | None = None,
        video_metadata: Sequence[Any] | None = None,
        video_kwargs: Mapping[str, Any] | None = None,
        image_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._functional is not None
        prompt = self._apply_chat_template(messages)
        kwargs: dict[str, Any] = {}
        if videos is not None:
            scoped_video_kwargs = dict(video_kwargs or {})
            if self._video_budget_is_custom or "size" not in scoped_video_kwargs:
                scoped_video_kwargs["size"] = self.video_size
            if video_metadata is not None:
                scoped_video_kwargs["video_metadata"] = [
                    _processor_video_metadata(item) if isinstance(item, Mapping) else item
                    for item in video_metadata
                ]
            kwargs["videos_kwargs"] = scoped_video_kwargs
        if image_kwargs is not None:
            kwargs["images_kwargs"] = dict(image_kwargs)
        kwargs.update(
            {
                "text": prompt,
                "images": images,
                "videos": videos,
                "return_tensors": "pt",
            }
        )
        inputs = self._processor(**kwargs)
        _assert_embedding_anchor(self._processor, inputs)
        # Transformers 5.x may retain ``video_metadata`` in BatchFeature even
        # though it is only a processor-side side channel.  The custom WeMM
        # ``embedding`` method forwards unknown kwargs to Qwen's inner model,
        # so passing that key would fail at runtime.  Keep it out of the model
        # call while retaining grid/shape facts for the observation below.
        processor_inputs = inputs
        model_inputs = _without_side_channels(inputs)
        model_inputs = _move_inputs(model_inputs, self._device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.embedding(**model_inputs)
        rows = _tensor_to_rows(
            output,
            torch=self._torch,
            functional=self._functional,
            dimension=self.dimension,
            supported_dimensions=self._supported_dimensions,
        )
        grid = _shape_rows(
            processor_inputs.get("video_grid_thw") if hasattr(processor_inputs, "get") else None
        )
        video_min_pixels, video_max_pixels, video_size = (
            self._video_observation_fields() if modality == "video" else (None, None, ())
        )
        self.observations.append(
            WemmInputObservation(
                modality=modality,
                item_count=len(rows),
                frame_count=frame_count,
                embedding_dimension=len(rows[0]),
                requested_dimension=self.dimension,
                input_keys=tuple(sorted(str(key) for key in model_inputs))
                if hasattr(model_inputs, "keys")
                else (),
                video_grid_thw=grid,
                elapsed_seconds=time.perf_counter() - started,
                video_min_pixels=video_min_pixels,
                video_max_pixels=video_max_pixels,
                video_size=video_size,
            )
        )
        return rows

    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 16,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode text labels in bounded batches while preserving input order."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        values = [str(text).strip() for text in texts]
        if any(not value for value in values):
            raise ValueError("WeMM text input cannot be empty")
        if not values:
            return ()
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None and self._functional is not None
        rows: list[tuple[float, ...]] = []
        for offset in range(0, len(values), batch_size):
            chunk = values[offset : offset + batch_size]
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": value}]}] for value in chunk
            ]
            prompts = self._apply_chat_template(messages)
            inputs = self._processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            _assert_embedding_anchor(self._processor, inputs)
            model_inputs = _move_inputs(_without_side_channels(inputs), self._device)
            started = time.perf_counter()
            with self._torch.inference_mode():
                output = self._model.embedding(**model_inputs)
            encoded = _tensor_to_rows(
                output,
                torch=self._torch,
                functional=self._functional,
                dimension=self.dimension,
                supported_dimensions=self._supported_dimensions,
            )
            if len(encoded) != len(chunk):
                raise WemmBackendUnavailable(
                    f"WeMM text embedding returned {len(encoded)} rows; expected {len(chunk)}"
                )
            rows.extend(encoded)
            self.observations.append(
                WemmInputObservation(
                    modality="text",
                    item_count=len(encoded),
                    frame_count=None,
                    embedding_dimension=len(encoded[0]),
                    requested_dimension=self.dimension,
                    input_keys=tuple(sorted(str(key) for key in model_inputs))
                    if hasattr(model_inputs, "keys")
                    else (),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        return tuple(rows)

    def encode_texts_cached(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 16,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode and cache an ordered text-prototype set for this backend.

        This is an explicit opt-in wrapper around :meth:`encode_texts` so
        existing callers retain their historical observation/timing semantics.
        A cache entry is scoped to this resident backend and the exact ordered
        text values, which is sufficient for the immutable model/prototype
        contract used by the WeMM batch runner.
        """

        values = tuple(str(text).strip() for text in texts)
        if any(not value for value in values):
            raise ValueError("WeMM text input cannot be empty")
        cached = self._text_prototype_cache.get(values)
        if cached is not None:
            self._text_prototype_cache_hits += 1
            return cached
        self._text_prototype_cache_misses += 1
        encoded = self.encode_texts(values, batch_size=batch_size)
        self._text_prototype_cache[values] = encoded
        return encoded

    def text_prototype_cache_stats(self) -> dict[str, int]:
        """Return lightweight cache telemetry for resident-run reports."""

        return {
            "entries": len(self._text_prototype_cache),
            "hits": self._text_prototype_cache_hits,
            "misses": self._text_prototype_cache_misses,
        }

    def encode_images(self, images: Iterable[str | Path | Any]) -> tuple[tuple[float, ...], ...]:
        """Encode local image paths/PIL images through the shared image space."""

        rows: list[tuple[float, ...]] = []
        for image in images:
            value: Any = image
            if isinstance(image, (str, Path)):
                value = str(Path(image).expanduser().resolve())
            rows.extend(
                self._encode_messages(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": value, **WEMM_IMAGE_KWARGS}],
                        }
                    ],
                    modality="image",
                    images=[value],
                    image_kwargs={"size": dict(WEMM_IMAGE_SIZE)},
                )
            )
        return tuple(rows)

    def encode_video_frames(
        self,
        frame_groups: Iterable[Sequence[Any]],
        *,
        metadata_groups: Iterable[Mapping[str, Any]] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode already-decoded ordered frame groups through native video input.

        Supplying frames and explicit metadata lets the runner pass a bounded
        interval without writing a re-encoded temporary movie.  It also avoids
        silently falling back to independent image messages, which would break
        temporal semantics.  ``metadata_groups`` follows the local Qwen native
        runtime shape: ``total_num_frames``, ``fps``, ``width``, ``height`` and
        ``frames_indices`` (with optional ``duration``).
        """

        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None and self._functional is not None
        sentinel = object()
        paired_groups: Iterable[tuple[Any, Any]]
        if metadata_groups is None:
            paired_groups = ((frames, None) for frames in frame_groups)
        else:
            from itertools import zip_longest

            paired_groups = zip_longest(frame_groups, metadata_groups, fillvalue=sentinel)
        rows: list[tuple[float, ...]] = []
        for _index, pair in enumerate(paired_groups):
            raw_frames, raw_info = pair
            if raw_frames is sentinel or raw_info is sentinel:
                raise ValueError("metadata_groups must match frame_groups")
            frames = list(raw_frames)
            if len(frames) < 2 or len(frames) > WEMM_VIDEO_KWARGS["max_frames"]:
                raise ValueError("video frame group must contain between 2 and 64 frames")
            if raw_info is not None and not isinstance(raw_info, Mapping):
                raise ValueError("video metadata entries must be mappings")
            info: dict[str, Any] = (
                dict(raw_info) if raw_info is not None else _synthetic_frame_metadata(frames)
            )
            processor_info = _processor_video_metadata(info)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                            **self._video_content_kwargs(),
                        }
                    ],
                }
            ]
            prompt = self._apply_chat_template(messages)
            scoped_video_kwargs: dict[str, Any] = {
                "size": self.video_size,
                "do_sample_frames": False,
            }
            # Transformers 5.13 requires a non-empty metadata side channel for
            # direct frame lists, even when return_metadata=False is requested.
            # Real callers provide source metadata; otherwise use the bounded
            # synthetic clock generated above.
            scoped_video_kwargs["return_metadata"] = True
            scoped_video_kwargs["video_metadata"] = [processor_info]
            processor_kwargs: dict[str, Any] = {
                "text": prompt,
                "videos": [frames],
                "truncation": False,
                "return_tensors": "pt",
                "videos_kwargs": scoped_video_kwargs,
            }
            # Temporal sampling remains disabled because metadata already
            # describes the selected source ordinals.  Resize controls are
            # modality-scoped; flat min/max/total_pixels kwargs are not valid
            # for Qwen3VLProcessor 5.x and would silently use huge defaults.
            inputs = self._processor(**processor_kwargs)
            _assert_embedding_anchor(self._processor, inputs)
            # ``video_metadata`` is a descriptive side channel, not a model
            # tensor.  Keeping it out of ``embedding`` mirrors the native Qwen
            # runtime and prevents accidental kwargs errors.
            model_inputs = {key: value for key, value in inputs.items() if key != "video_metadata"}
            model_inputs = _move_inputs(model_inputs, self._device)
            started = time.perf_counter()
            with self._torch.inference_mode():
                output = self._model.embedding(**model_inputs)
            encoded = _tensor_to_rows(
                output,
                torch=self._torch,
                functional=self._functional,
                dimension=self.dimension,
                supported_dimensions=self._supported_dimensions,
            )
            rows.extend(encoded)
            grid = _shape_rows(
                model_inputs.get("video_grid_thw") if hasattr(model_inputs, "get") else None
            )
            video_min_pixels, video_max_pixels, video_size = self._video_observation_fields()
            self.observations.append(
                WemmInputObservation(
                    modality="video",
                    item_count=len(encoded),
                    frame_count=len(frames),
                    embedding_dimension=len(encoded[0]),
                    requested_dimension=self.dimension,
                    input_keys=tuple(sorted(str(key) for key in model_inputs))
                    if hasattr(model_inputs, "keys")
                    else (),
                    video_grid_thw=grid,
                    elapsed_seconds=time.perf_counter() - started,
                    video_min_pixels=video_min_pixels,
                    video_max_pixels=video_max_pixels,
                    video_size=video_size,
                )
            )
        return tuple(rows)

    def encode_video_frames_batch(
        self,
        frame_groups: Iterable[Sequence[Any]],
        *,
        metadata_groups: Iterable[Mapping[str, Any]] | None = None,
        batch_size: int = 2,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode bounded frame groups in explicit video microbatches.

        This is an opt-in benchmark seam.  The historical
        :meth:`encode_video_frames` method remains singleton/serial and is not
        routed through this method.  ``batch_size`` bounds both the number of
        videos handed to one processor/model call and the number of embeddings
        retained before they are appended to the ordered result.  ``2`` and
        ``4`` are the intended Batch2/Batch4 experiment arms, although any
        positive value is accepted for small fixture probes.

        Frame groups and metadata are consumed incrementally and paired by
        position.  The returned rows always follow input order; no camera,
        window, timestamp, or content identity is inferred or persisted here.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None and self._functional is not None

        frame_iter = iter(frame_groups)
        metadata_iter = iter(metadata_groups) if metadata_groups is not None else None
        sentinel = object()
        rows: list[tuple[float, ...]] = []
        batch_index = 0

        while True:
            raw_frames_batch = list(islice(frame_iter, batch_size))
            if not raw_frames_batch:
                if metadata_iter is not None and next(metadata_iter, sentinel) is not sentinel:
                    raise ValueError("metadata_groups must match frame_groups")
                break

            raw_metadata_batch: list[Any]
            if metadata_iter is None:
                raw_metadata_batch = [None] * len(raw_frames_batch)
            else:
                raw_metadata_batch = list(islice(metadata_iter, len(raw_frames_batch)))
                if len(raw_metadata_batch) != len(raw_frames_batch):
                    raise ValueError("metadata_groups must match frame_groups")

            prepared_frames: list[list[Any]] = []
            processor_metadata: list[dict[str, Any]] = []
            frame_counts: list[int] = []
            prepare_started = time.perf_counter()
            for index, (raw_frames, raw_info) in enumerate(
                zip(raw_frames_batch, raw_metadata_batch, strict=True)
            ):
                try:
                    frames = list(raw_frames)
                except TypeError as exc:
                    raise ValueError(f"video frame group {index} must be iterable") from exc
                if len(frames) < 2 or len(frames) > WEMM_VIDEO_KWARGS["max_frames"]:
                    raise ValueError("video frame group must contain between 2 and 64 frames")
                if raw_info is not None and not isinstance(raw_info, Mapping):
                    raise ValueError("video metadata entries must be mappings")
                info: dict[str, Any] = (
                    dict(raw_info) if raw_info is not None else _synthetic_frame_metadata(frames)
                )
                prepared_frames.append(frames)
                processor_metadata.append(_processor_video_metadata(info))
                frame_counts.append(len(frames))

            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": frames,
                                **self._video_content_kwargs(),
                            }
                        ],
                    }
                ]
                for frames in prepared_frames
            ]
            prompt = self._apply_chat_template(messages)
            processor_started = time.perf_counter()
            scoped_video_kwargs: dict[str, Any] = {
                "size": self.video_size,
                "do_sample_frames": False,
                # Transformers 5.x requires this side channel for direct frame
                # lists, even when the model only consumes tensor inputs.
                "return_metadata": True,
                "video_metadata": processor_metadata,
            }
            inputs = self._processor(
                text=prompt,
                videos=prepared_frames,
                truncation=False,
                return_tensors="pt",
                videos_kwargs=scoped_video_kwargs,
            )
            processor_seconds = time.perf_counter() - processor_started
            _assert_embedding_anchor(self._processor, inputs)
            model_inputs = _without_side_channels(inputs)
            processor_shapes = _tensor_shapes(inputs)
            model_inputs = _move_inputs(model_inputs, self._device)

            model_started = time.perf_counter()
            with self._torch.inference_mode():
                output = self._model.embedding(**model_inputs)
            model_seconds = time.perf_counter() - model_started

            postprocess_started = time.perf_counter()
            encoded = _tensor_to_rows(
                output,
                torch=self._torch,
                functional=self._functional,
                dimension=self.dimension,
                supported_dimensions=self._supported_dimensions,
            )
            postprocess_seconds = time.perf_counter() - postprocess_started
            if len(encoded) != len(prepared_frames):
                raise WemmBackendUnavailable(
                    "WeMM video microbatch returned "
                    f"{len(encoded)} rows; expected {len(prepared_frames)}"
                )
            rows.extend(encoded)

            grid = _shape_rows(inputs.get("video_grid_thw") if hasattr(inputs, "get") else None)
            video_min_pixels, video_max_pixels, video_size = self._video_observation_fields()
            total_seconds = max(0.0, time.perf_counter() - prepare_started)
            self.observations.append(
                WemmInputObservation(
                    modality="video",
                    item_count=len(encoded),
                    frame_count=(
                        frame_counts[0]
                        if frame_counts and all(item == frame_counts[0] for item in frame_counts)
                        else None
                    ),
                    embedding_dimension=len(encoded[0]),
                    requested_dimension=self.dimension,
                    input_keys=tuple(sorted(str(key) for key in model_inputs))
                    if hasattr(model_inputs, "keys")
                    else (),
                    video_grid_thw=grid,
                    elapsed_seconds=total_seconds,
                    batch_size=len(prepared_frames),
                    batch_index=batch_index,
                    frame_counts=tuple(frame_counts),
                    tensor_shapes=processor_shapes,
                    phase_timings=(
                        ("prepare", max(0.0, processor_started - prepare_started)),
                        ("processor", max(0.0, processor_seconds)),
                        ("model", max(0.0, model_seconds)),
                        ("postprocess", max(0.0, postprocess_seconds)),
                        ("total", total_seconds),
                    ),
                    video_min_pixels=video_min_pixels,
                    video_max_pixels=video_max_pixels,
                    video_size=video_size,
                )
            )
            batch_index += 1
        return tuple(rows)

    def encode_videos(
        self,
        videos: Iterable[str | Path],
        *,
        frame_counts: Sequence[int | None] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode complete bounded local videos using WeMM's native video path."""

        paths = [str(Path(video).expanduser().resolve()) for video in videos]
        counts = list(frame_counts or ())
        if frame_counts is not None and len(counts) != len(paths):
            raise ValueError("frame_counts must match the number of videos")
        try:
            process_vision_info = importlib.import_module("qwen_vl_utils").__dict__[
                "process_vision_info"
            ]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise WemmBackendUnavailable(
                "native WeMM video input requires qwen-vl-utils[decord]==0.0.14"
            ) from exc
        rows: list[tuple[float, ...]] = []
        for index, path in enumerate(paths):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": path,
                            **self._video_content_kwargs(),
                        }
                    ],
                }
            ]
            images, videos_payload, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            metadata: Sequence[Any] | None = None
            if videos_payload:
                try:
                    videos_payload, metadata = zip(*videos_payload, strict=True)
                except (TypeError, ValueError) as exc:
                    raise WemmBackendUnavailable(
                        "qwen-vl-utils returned malformed video metadata"
                    ) from exc
                videos_payload = list(videos_payload)
                metadata = list(metadata)
            elif videos_payload is not None:
                videos_payload = []
            # Reuse the same processor call helper, but it needs the prepared
            # visual payloads rather than the source message.  Keep this branch
            # explicit so the exact native path remains observable.
            self._load()
            assert self._processor is not None
            prompt = self._apply_chat_template(messages)
            raw_video_kwargs = dict(video_kwargs or {})
            # Keep explicit values authoritative if qwen-vl-utils happens to
            # return one of these keys in its kwargs mapping.  In particular,
            # ``video_metadata`` must not be passed twice on Transformers 5.x.
            processor_kwargs: dict[str, Any] = {
                "text": prompt,
                "images": images,
                "videos": videos_payload,
                "return_tensors": "pt",
            }
            # qwen-vl-utils may return flat video kwargs (for example fps or
            # do_sample_frames).  Put the resize SizeDict in the supported
            # modality-scoped namespace and remove stale flat aliases.
            flat_video_min = raw_video_kwargs.pop("min_pixels", None)
            flat_video_max = raw_video_kwargs.pop("max_pixels", None)
            raw_video_kwargs.pop("total_pixels", None)
            video_modality_kwargs = dict(raw_video_kwargs.pop("videos_kwargs", {}) or {})
            video_modality_kwargs.update(raw_video_kwargs)
            if self._video_budget_is_custom:
                video_modality_kwargs["size"] = self.video_size
            elif flat_video_min is not None and flat_video_max is not None:
                video_modality_kwargs["size"] = {
                    "shortest_edge": int(flat_video_min),
                    "longest_edge": int(flat_video_max),
                }
            else:
                video_modality_kwargs.setdefault("size", self.video_size)
            processor_kwargs["videos_kwargs"] = video_modality_kwargs
            if metadata is not None:
                video_modality_kwargs["video_metadata"] = [
                    _processor_video_metadata(item) if isinstance(item, Mapping) else item
                    for item in metadata
                ]
            inputs = self._processor(**processor_kwargs)
            _assert_embedding_anchor(self._processor, inputs)
            model_inputs = _without_side_channels(inputs)
            model_inputs = _move_inputs(model_inputs, self._device)
            assert (
                self._torch is not None and self._functional is not None and self._model is not None
            )
            started = time.perf_counter()
            with self._torch.inference_mode():
                output = self._model.embedding(**model_inputs)
            encoded = _tensor_to_rows(
                output,
                torch=self._torch,
                functional=self._functional,
                dimension=self.dimension,
                supported_dimensions=self._supported_dimensions,
            )
            rows.extend(encoded)
            grid = _shape_rows(
                model_inputs.get("video_grid_thw") if hasattr(model_inputs, "get") else None
            )
            video_min_pixels, video_max_pixels, video_size = self._video_observation_fields()
            self.observations.append(
                WemmInputObservation(
                    modality="video",
                    item_count=len(encoded),
                    frame_count=counts[index] if index < len(counts) else None,
                    embedding_dimension=len(encoded[0]),
                    requested_dimension=self.dimension,
                    input_keys=tuple(sorted(str(key) for key in model_inputs))
                    if hasattr(model_inputs, "keys")
                    else (),
                    video_grid_thw=grid,
                    elapsed_seconds=time.perf_counter() - started,
                    video_min_pixels=video_min_pixels,
                    video_max_pixels=video_max_pixels,
                    video_size=video_size,
                )
            )
        return tuple(rows)

    def close(self) -> None:
        """Release the resident model between independent benchmark arms."""

        self._text_prototype_cache.clear()
        self._text_prototype_cache_hits = 0
        self._text_prototype_cache_misses = 0
        self._model = None
        self._processor = None
        self._device = None
        with suppress(Exception):
            import gc

            gc.collect()
        if self._torch is not None and getattr(self._torch, "cuda", None) is not None:
            with suppress(Exception):
                self._torch.cuda.empty_cache()

    def observation_payload(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.observations]


__all__ = [
    "WEMM_VIDEO_MICROBATCH_SIZES",
    "WemmBackendUnavailable",
    "WemmEmbeddingBackend",
    "WemmInputObservation",
]
