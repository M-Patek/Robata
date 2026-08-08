"""Mage-compatible visual encoder-lite shadow path.

The candidate in this module never replaces the native Mage authority. It reuses
Mage's own visual tower and merger, optionally applies an early layer budget, and
keeps a bounded set of complete temporal placeholder runs before invoking the same
Qwen decoder. Arbitrary external embeddings are rejected: only 2560-dimensional
features emitted by the resident Mage merger are accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from threading import RLock
from time import perf_counter
from typing import Any, Final, Literal

from pydantic import Field

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256

MAGE_SMALL_ENCODER_POLICY_VERSION: Final = "mage-small-encoder-shadow-v2"
MAGE_SMALL_ENCODER_ID: Final = "mage-visual-encoder-lite"
MAGE_SMALL_ENCODER_REVISION: Final = "mage-vl-local-2026-08-07"
MAGE_SMALL_ENCODER_SELECTION_MODE: Final = "UNIFORM_TEMPORAL_RUN_KEEP_NO_EMPTY_SPANS_V2"


class MageSmallEncoderError(ValueError):
    """The shadow encoder could not preserve Mage token/placeholder invariants."""


def _load_torch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as error:
        raise MageSmallEncoderError(
            "Mage small-encoder execution requires the optional torch runtime"
        ) from error


class MageSmallEncoderPolicy(StrictModel):
    """Identity-bound, non-authoritative encoder budget.

    ``visual_layer_count`` is an early-exit budget over Mage's own 24-layer visual
    tower. ``max_temporal_runs`` selects evenly distributed complete placeholder
    runs, always including the first and last run. No cross-time mean pooling is
    performed because that would erase boundary evidence without a trained adapter.
    """

    policy_version: Literal["mage-small-encoder-shadow-v2"] = MAGE_SMALL_ENCODER_POLICY_VERSION
    encoder_id: Literal["mage-visual-encoder-lite"] = MAGE_SMALL_ENCODER_ID
    encoder_revision: str = MAGE_SMALL_ENCODER_REVISION
    selection_mode: Literal["UNIFORM_TEMPORAL_RUN_KEEP_NO_EMPTY_SPANS_V2"] = (
        MAGE_SMALL_ENCODER_SELECTION_MODE
    )
    visual_layer_count: int = Field(default=24, ge=1, le=24)
    max_temporal_runs: int = Field(default=4, ge=2, le=64)
    shadow_only: Literal[True] = True

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return semantic_sha256(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class MageSmallEncoderTelemetry:
    """Non-wire timing and shape telemetry for one shadow preparation."""

    policy_semantic_sha256: Sha256Digest
    input_token_count: int
    compressed_token_count: int
    visual_token_count: int
    compressed_visual_token_count: int
    temporal_run_count: int
    kept_temporal_run_indices: tuple[int, ...]
    visual_layer_count: int
    max_temporal_runs: int
    visual_seconds: float
    selection_seconds: float
    embedding_seconds: float
    digest_seconds: float
    total_seconds: float
    feature_content_sha256: Sha256Digest

    @property
    def token_reduction_ratio(self) -> float:
        if self.visual_token_count <= 0:
            return 0.0
        return 1.0 - (self.compressed_visual_token_count / self.visual_token_count)

    def as_projection(self) -> dict[str, object]:
        return {
            "policy_semantic_sha256": self.policy_semantic_sha256,
            "input_token_count": self.input_token_count,
            "compressed_token_count": self.compressed_token_count,
            "visual_token_count": self.visual_token_count,
            "compressed_visual_token_count": self.compressed_visual_token_count,
            "temporal_run_count": self.temporal_run_count,
            "kept_temporal_run_indices": list(self.kept_temporal_run_indices),
            "visual_layer_count": self.visual_layer_count,
            "max_temporal_runs": self.max_temporal_runs,
            "visual_seconds": self.visual_seconds,
            "selection_seconds": self.selection_seconds,
            "embedding_seconds": self.embedding_seconds,
            "digest_seconds": self.digest_seconds,
            "total_seconds": self.total_seconds,
            "token_reduction_ratio": self.token_reduction_ratio,
            "feature_content_sha256": self.feature_content_sha256,
            "shadow_only": True,
        }


@dataclass(frozen=True, slots=True)
class MageSmallEncoderInputs:
    """Decoder-ready inputs plus shadow-only lineage/telemetry."""

    input_ids: Any
    attention_mask: Any
    inputs_embeds: Any
    telemetry: MageSmallEncoderTelemetry

    @property
    def shadow_only(self) -> bool:
        return True


def _feature_content_digest(value: Any) -> Sha256Digest:
    """Digest a stable float32 projection, not an unpublished tensor wire format."""

    torch = _load_torch()

    detached = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    projection = (
        str(value.dtype).encode("utf-8")
        + b"|"
        + repr(tuple(int(dimension) for dimension in value.shape)).encode("ascii")
        + b"|"
        + detached.numpy().tobytes()
    )
    return sha256(projection).hexdigest()


def _placeholder_runs(input_ids: Any, image_token_id: int) -> tuple[tuple[int, int], ...]:
    if getattr(input_ids, "ndim", None) != 1:
        raise MageSmallEncoderError("small encoder accepts exactly one flattened input-id row")
    values = [int(item) for item in input_ids.detach().to("cpu").tolist()]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*values, None]):
        if value == image_token_id and start is None:
            start = index
        elif value != image_token_id and start is not None:
            runs.append((start, index))
            start = None
    return tuple(runs)


def uniform_temporal_run_indices(*, run_count: int, max_temporal_runs: int) -> tuple[int, ...]:
    """Select deterministic, evenly spaced runs while preserving both boundaries."""

    if run_count <= 0:
        raise MageSmallEncoderError("at least one Mage visual placeholder run is required")
    if max_temporal_runs < 2:
        raise MageSmallEncoderError("max_temporal_runs must be >= 2")
    if run_count <= max_temporal_runs:
        return tuple(range(run_count))
    denominator = max_temporal_runs - 1
    return tuple(
        (position * (run_count - 1)) // denominator for position in range(max_temporal_runs)
    )


def select_mage_visual_token_runs(
    *,
    input_ids: Any,
    attention_mask: Any,
    visual_features: Any,
    image_token_id: int,
    max_temporal_runs: int,
    vision_start_token_id: int | None = None,
    vision_end_token_id: int | None = None,
) -> tuple[Any, Any, Any, tuple[int, ...], int]:
    """Keep complete Mage-space temporal runs and rewrite matching placeholders.

    All non-visual text and timestamp tokens remain in order. For dropped runs, the
    matching vision-start/end wrapper is removed as well; timestamp marker tokens are
    retained, so the decoder never receives an empty visual span. Features from
    distinct temporal runs are never mixed.
    """

    if getattr(input_ids, "ndim", None) != 2 or int(input_ids.shape[0]) != 1:
        raise MageSmallEncoderError("small encoder requires batch size 1")
    if getattr(attention_mask, "shape", None) != getattr(input_ids, "shape", None):
        raise MageSmallEncoderError("attention_mask shape must match input_ids")
    if getattr(visual_features, "ndim", None) != 2:
        raise MageSmallEncoderError("visual features must have shape [tokens, hidden]")

    ids = input_ids[0]
    mask = attention_mask[0]
    runs = _placeholder_runs(ids, image_token_id)
    visual_count = int(sum(end - start for start, end in runs))
    if visual_count != int(visual_features.shape[0]):
        raise MageSmallEncoderError(
            "Mage visual token count does not match placeholder count: "
            f"features={int(visual_features.shape[0])}, placeholders={visual_count}"
        )
    kept_indices = uniform_temporal_run_indices(
        run_count=len(runs),
        max_temporal_runs=max_temporal_runs,
    )
    kept_set = frozenset(kept_indices)
    removed_positions: set[int] = set()
    for run_index, (start, end) in enumerate(runs):
        if run_index in kept_set:
            continue
        if vision_start_token_id is None or vision_end_token_id is None:
            raise MageSmallEncoderError(
                "vision_start_token_id and vision_end_token_id are required when dropping runs"
            )
        wrapper_start = start - 1
        wrapper_end = end
        if wrapper_start < 0 or int(ids[wrapper_start]) != vision_start_token_id:
            raise MageSmallEncoderError(
                "dropped Mage visual run has no matching vision-start token"
            )
        if wrapper_end >= int(ids.shape[0]) or int(ids[wrapper_end]) != vision_end_token_id:
            raise MageSmallEncoderError("dropped Mage visual run has no matching vision-end token")
        removed_positions.add(wrapper_start)
        removed_positions.add(wrapper_end)
        removed_positions.update(range(start, end))

    selected_features: list[Any] = []
    feature_offset = 0
    for run_index, (start, end) in enumerate(runs):
        run_features = visual_features[feature_offset : feature_offset + (end - start)]
        feature_offset += end - start
        if run_index in kept_set:
            selected_features.extend(run_features.unbind(dim=0))

    output_ids = [
        int(ids[index]) for index in range(int(ids.shape[0])) if index not in removed_positions
    ]
    output_mask = [
        int(mask[index]) for index in range(int(mask.shape[0])) if index not in removed_positions
    ]

    torch = _load_torch()

    device = input_ids.device
    output_input_ids = torch.tensor([output_ids], dtype=input_ids.dtype, device=device)
    output_attention_mask = torch.tensor(
        [output_mask],
        dtype=attention_mask.dtype,
        device=device,
    )
    output_features = torch.stack(selected_features).to(
        device=device,
        dtype=visual_features.dtype,
    )
    return (
        output_input_ids,
        output_attention_mask,
        output_features,
        kept_indices,
        len(runs),
    )


class MageCompatibleSmallEncoder:
    """Run a bounded Mage visual tower and produce shadow-only decoder inputs.

    Early exit temporarily swaps the resident visual layer list. Calls through this
    instance are locked, but callers must also serialize native generation against
    ``prepare``. The qualification runner enforces one worker and one model call in
    flight; this class is intentionally not a high-concurrency serving primitive.
    """

    def __init__(self, *, model: Any, policy: MageSmallEncoderPolicy) -> None:
        self.model = model
        self.policy = policy
        self._visual_lock = RLock()
        self._validate_model()

    def _validate_model(self) -> None:
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is None or not hasattr(visual, "encoder") or not hasattr(visual, "merger"):
            raise MageSmallEncoderError("model does not expose the Mage visual tower")
        layer_count = len(getattr(visual.encoder, "layers", ()))
        if layer_count <= 0:
            raise MageSmallEncoderError("Mage visual tower has no encoder layers")
        if self.policy.visual_layer_count > layer_count:
            raise MageSmallEncoderError(
                f"visual_layer_count must be within 1..{layer_count}, "
                f"got {self.policy.visual_layer_count}"
            )

    def _visual_features(self, inputs: Mapping[str, Any]) -> Any:
        visual = self.model.model.visual
        pixel_values = inputs.get("pixel_values")
        grid_thw = inputs.get("image_grid_thw")
        patch_positions = inputs.get("patch_positions")
        if pixel_values is None or grid_thw is None or patch_positions is None:
            raise MageSmallEncoderError(
                "native Mage processor inputs must include pixel_values, image_grid_thw, "
                "and patch_positions"
            )
        with self._visual_lock:
            original_layers = visual.encoder.layers
            try:
                torch = _load_torch()

                visual.encoder.layers = torch.nn.ModuleList(
                    list(original_layers)[: self.policy.visual_layer_count]
                )
                model_dtype = getattr(visual.embeddings.patch_embedding.weight, "dtype", None)
                prepared = (
                    pixel_values.to(dtype=model_dtype) if model_dtype is not None else pixel_values
                )
                with torch.inference_mode():
                    output = visual(
                        prepared,
                        grid_thw=grid_thw,
                        patch_positions=patch_positions,
                    )
                return output.last_hidden_state.reshape(-1, output.last_hidden_state.shape[-1])
            finally:
                visual.encoder.layers = original_layers

    def prepare(self, inputs: Mapping[str, Any]) -> MageSmallEncoderInputs:
        torch = _load_torch()

        started = perf_counter()
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise MageSmallEncoderError(
                "native Mage processor inputs must include input_ids and attention_mask"
            )
        synchronize = bool(torch.cuda.is_available() and getattr(input_ids, "is_cuda", False))
        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        with torch.inference_mode():
            if synchronize:
                torch.cuda.synchronize(input_ids.device)
            visual_started = perf_counter()
            visual_features = self._visual_features(inputs)
            if synchronize:
                torch.cuda.synchronize(visual_features.device)
            visual_seconds = perf_counter() - visual_started

            selection_started = perf_counter()
            (
                compressed_ids,
                compressed_mask,
                compressed_features,
                kept_indices,
                temporal_run_count,
            ) = select_mage_visual_token_runs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                image_token_id=image_token_id,
                max_temporal_runs=self.policy.max_temporal_runs,
                vision_start_token_id=int(
                    getattr(self.model.config, "vision_start_token_id", 151652)
                ),
                vision_end_token_id=int(getattr(self.model.config, "vision_end_token_id", 151653)),
            )
            if synchronize:
                torch.cuda.synchronize(compressed_features.device)
            selection_seconds = perf_counter() - selection_started

            embedding_started = perf_counter()
            embedding_layer = self.model.get_input_embeddings()
            expected_hidden_size = int(embedding_layer.weight.shape[1])
            if int(compressed_features.shape[1]) != expected_hidden_size:
                raise MageSmallEncoderError(
                    "Mage merger features do not match the decoder hidden size: "
                    f"features={int(compressed_features.shape[1])}, "
                    f"decoder={expected_hidden_size}"
                )
            input_embeddings = embedding_layer(compressed_ids)
            placeholder_mask = compressed_ids[0] == image_token_id
            if int(placeholder_mask.sum()) != int(compressed_features.shape[0]):
                raise MageSmallEncoderError("compressed placeholder and feature counts diverged")
            input_embeddings[0, placeholder_mask, :] = compressed_features.to(
                input_embeddings.dtype
            )
            if synchronize:
                torch.cuda.synchronize(input_embeddings.device)
            embedding_seconds = perf_counter() - embedding_started

            digest_started = perf_counter()
            feature_content_sha256 = _feature_content_digest(compressed_features)
            digest_seconds = perf_counter() - digest_started
        telemetry = MageSmallEncoderTelemetry(
            policy_semantic_sha256=self.policy.semantic_sha256,
            input_token_count=int(input_ids.shape[1]),
            compressed_token_count=int(compressed_ids.shape[1]),
            visual_token_count=int(visual_features.shape[0]),
            compressed_visual_token_count=int(compressed_features.shape[0]),
            temporal_run_count=temporal_run_count,
            kept_temporal_run_indices=kept_indices,
            visual_layer_count=self.policy.visual_layer_count,
            max_temporal_runs=self.policy.max_temporal_runs,
            visual_seconds=float(visual_seconds),
            selection_seconds=float(selection_seconds),
            embedding_seconds=float(embedding_seconds),
            digest_seconds=float(digest_seconds),
            total_seconds=float(perf_counter() - started),
            feature_content_sha256=feature_content_sha256,
        )
        return MageSmallEncoderInputs(
            input_ids=compressed_ids,
            attention_mask=compressed_mask,
            inputs_embeds=input_embeddings,
            telemetry=telemetry,
        )


__all__ = [
    "MAGE_SMALL_ENCODER_ID",
    "MAGE_SMALL_ENCODER_POLICY_VERSION",
    "MAGE_SMALL_ENCODER_REVISION",
    "MAGE_SMALL_ENCODER_SELECTION_MODE",
    "MageCompatibleSmallEncoder",
    "MageSmallEncoderError",
    "MageSmallEncoderInputs",
    "MageSmallEncoderPolicy",
    "MageSmallEncoderTelemetry",
    "select_mage_visual_token_runs",
    "uniform_temporal_run_indices",
]
