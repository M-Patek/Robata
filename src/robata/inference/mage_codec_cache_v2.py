"""Strict cache admission records for the explicit Mage DCVC Provider V2.

The v2 cache family is separate from the observed-v1 cache.  It binds the qualified
checkpoint, declared codec policy, exact provider implementation, explicit effective
configuration, worker-authored preparation artifact, and every output byte.  Paths are
operational locators only and never substitute for identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, JsonValue, StringConstraints, ValidationError, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.inference.mage_checkpoint_identity import MageCheckpointManifest
from robata.inference.mage_dcvc_preparation_protocol import (
    MAGE_DCVC_PREPARATION_SIDECAR_NAME,
    MAGE_DCVC_PROVIDER_VERSION,
    MAGE_DCVC_RECIPE_VERSION,
    MageDcvcEffectiveConfig,
    MageDcvcPreparationArtifact,
    MageDcvcPreparedAsset,
)
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    build_mage_video_codec_policy_identity,
)

MAGE_CODEC_CACHE_ENTRY_V2_VERSION: Final = "mage-codec-cache-entry-v2"
MAGE_CODEC_CACHE_MANIFEST_V2_VERSION: Final = "mage-codec-cache-manifest-v2"
MAGE_CODEC_CACHE_NAMESPACE_V2_VERSION: Final = "mage-codec-cache-namespace-v2"
MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME: Final = ".robata-cache-entry-v2.json"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


class MageCodecCacheV2Error(RuntimeError):
    """A Provider V2 cache record or asset failed strict verification."""


class MageCodecCacheEntryV2(StrictModel):
    """One source-bound Provider V2 output admitted to a qualified namespace."""

    entry_version: Literal["mage-codec-cache-entry-v2"] = MAGE_CODEC_CACHE_ENTRY_V2_VERSION
    namespace_version: Literal["mage-codec-cache-namespace-v2"] = (
        MAGE_CODEC_CACHE_NAMESPACE_V2_VERSION
    )
    provider_version: Literal["robata-mage-dcvc-provider-v2"] = MAGE_DCVC_PROVIDER_VERSION
    recipe_version: Literal["mage-dcvc-readiness-explicit-v2"] = MAGE_DCVC_RECIPE_VERSION
    source_path: NonEmptyString
    source_content_sha256: Sha256Digest
    source_byte_count: PositiveInt
    checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    provider_implementation_sha256: Sha256Digest
    effective_config_sha256: Sha256Digest
    preparation_identity: Sha256Digest
    preparation_artifact_semantic_sha256: Sha256Digest
    preparation_artifact_exact_sha256: Sha256Digest
    logical_cache_identity: Sha256Digest
    namespace_identity: Sha256Digest
    provider_cache_directory_name: NonEmptyString
    assets: tuple[MageDcvcPreparedAsset, ...]
    asset_set_sha256: Sha256Digest
    provider_metadata: dict[str, JsonValue]
    entry_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        if Path(self.provider_cache_directory_name).name != self.provider_cache_directory_name:
            raise ValueError("provider_cache_directory_name must contain one path component")
        paths = tuple(asset.relative_path for asset in self.assets)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("assets must be nonempty, unique, and sorted")
        expected_namespace = mage_codec_v2_namespace_identity(
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            provider_implementation_sha256=self.provider_implementation_sha256,
            effective_config_sha256=self.effective_config_sha256,
        )
        if self.namespace_identity != expected_namespace:
            raise ValueError("namespace_identity does not match Provider V2 inputs")
        expected_logical = mage_codec_v2_logical_cache_identity(
            source_content_sha256=self.source_content_sha256,
            source_byte_count=self.source_byte_count,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            provider_implementation_sha256=self.provider_implementation_sha256,
            effective_config_sha256=self.effective_config_sha256,
            preparation_identity=self.preparation_identity,
        )
        if self.logical_cache_identity != expected_logical:
            raise ValueError("logical_cache_identity does not match Provider V2 inputs")
        if self.asset_set_sha256 != _asset_set_sha256(self.assets):
            raise ValueError("asset_set_sha256 does not match assets")
        if self.entry_semantic_sha256 != _entry_semantic_sha256(self):
            raise ValueError("entry_semantic_sha256 does not match entry")
        return self


class MageCodecCacheManifestEntryV2(StrictModel):
    """Operational locator for one exact v2 entry."""

    source_path: NonEmptyString
    logical_cache_identity: Sha256Digest
    entry_semantic_sha256: Sha256Digest
    provider_cache_directory: NonEmptyString
    admission: Literal["BUILT", "VERIFIED_HIT"]
    preparation_wall_seconds: NonNegativeFloat


class MageCodecCacheManifestV2(StrictModel):
    """One exact Provider V2 cache namespace and its qualification observations."""

    manifest_version: Literal["mage-codec-cache-manifest-v2"] = MAGE_CODEC_CACHE_MANIFEST_V2_VERSION
    namespace_version: Literal["mage-codec-cache-namespace-v2"] = (
        MAGE_CODEC_CACHE_NAMESPACE_V2_VERSION
    )
    provider_version: Literal["robata-mage-dcvc-provider-v2"] = MAGE_DCVC_PROVIDER_VERSION
    recipe_version: Literal["mage-dcvc-readiness-explicit-v2"] = MAGE_DCVC_RECIPE_VERSION
    checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    provider_implementation_sha256: Sha256Digest
    effective_config: MageDcvcEffectiveConfig
    namespace_identity: Sha256Digest
    cache_base_root: NonEmptyString
    qualified_cache_root: NonEmptyString
    entry_count: PositiveInt
    built_count: NonNegativeInt
    verified_hit_count: NonNegativeInt
    prewarm_wall_seconds: NonNegativeFloat
    entries: tuple[MageCodecCacheManifestEntryV2, ...]
    manifest_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if (
            self.provider_implementation_sha256
            != self.effective_config.provider_implementation_sha256
        ):
            raise ValueError("manifest implementation does not match effective config")
        if self.namespace_identity != mage_codec_v2_namespace_identity(
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            provider_implementation_sha256=self.provider_implementation_sha256,
            effective_config_sha256=self.effective_config.effective_config_sha256,
        ):
            raise ValueError("manifest namespace identity does not match inputs")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count does not match entries")
        if self.built_count + self.verified_hit_count != self.entry_count:
            raise ValueError("manifest admission counts do not sum to entry_count")
        sources = tuple(item.source_path for item in self.entries)
        if sources != tuple(sorted(sources)) or len(sources) != len(set(sources)):
            raise ValueError("manifest entries must be unique and sorted by source_path")
        if self.manifest_semantic_sha256 != _manifest_semantic_sha256(self):
            raise ValueError("manifest_semantic_sha256 does not match manifest")
        return self


def validate_mage_dcvc_effective_config_for_policy(
    *, effective_config: MageDcvcEffectiveConfig, codec_policy: MageVideoCodecPolicy
) -> None:
    """Reject declared/effective divergence before V2 preparation or endpoint launch."""

    if codec_policy.codec_mode != "neural":
        raise MageCodecCacheV2Error("Provider V2 requires neural codec mode")
    parameters = codec_policy.neural_parameters
    if parameters is None:
        raise MageCodecCacheV2Error("Provider V2 requires explicit neural parameters")
    if parameters.sequence_length_frames != 0:
        raise MageCodecCacheV2Error(
            "Provider V2 qualification requires sequence_length_frames=0; it is not a compute cap"
        )
    if parameters.canvas_token_side is not None:
        raise MageCodecCacheV2Error("Provider V2 qualification requires canvas_token_side=null")
    comparisons = {
        "preparation_device": effective_config.preparation_device == codec_policy.preprocess_device,
        "qp": effective_config.qp == parameters.quantization_parameter,
        "reset_interval": effective_config.reset_interval == parameters.reset_interval,
        "intra_period": effective_config.intra_period == parameters.intra_period,
        "max_side": effective_config.max_side == parameters.max_side,
        "target_canvas": effective_config.target_canvas == codec_policy.target_canvas,
        "sampled_frame_count": effective_config.sampled_frame_count
        == (codec_policy.target_canvas // codec_policy.images_per_group) * codec_policy.group_size,
        "sequence_length_frames": effective_config.sequence_length_frames == 0,
        "canvas_token_side": effective_config.canvas_token_side is None,
        "group_size": effective_config.group_size == codec_policy.group_size,
        "images_per_group": effective_config.images_per_group == codec_policy.images_per_group,
        "patch": effective_config.patch == codec_policy.patch_size == 16,
        "max_pixels": effective_config.max_pixels == codec_policy.max_pixels,
        "min_group_frames": effective_config.min_group_frames == codec_policy.min_group_frames,
        "max_group_frames": effective_config.max_group_frames == 128,
        "grouping_mode": effective_config.grouping_mode == "readiness",
        "readiness_sum_threshold_mode": effective_config.readiness_sum_threshold_mode == "auto",
        "readiness_coverage_bins": effective_config.readiness_coverage_bins
        == parameters.readiness_coverage_bins,
        "readiness_delta_ratio": effective_config.readiness_delta_ratio
        == parameters.readiness_delta_ratio,
        "bitcost_grid": effective_config.bitcost_grid == "sub",
        "bitcost_percentile": effective_config.bitcost_percentile == parameters.bitcost_percentile,
        "bitcost_log_scale": effective_config.bitcost_log_scale is True,
        "decode_backsearch_max": effective_config.decode_backsearch_max
        == parameters.decode_backsearch_max,
        "canvas_format": effective_config.canvas_format == "jpg",
        "per_frame_cap_ratio": effective_config.per_frame_cap_ratio == 1.2,
        "bottom_attenuation": effective_config.bottom_attenuation == 0.5,
        "bottom_band_ratio": effective_config.bottom_band_ratio == 0.1,
        "threshold_scale": effective_config.threshold_scale == 1.0,
        "random_select": effective_config.random_select is False,
        "random_seed": effective_config.random_seed == 0,
        "encoded_frame_extent": effective_config.encoded_frame_extent
        == "through-last-sampled-frame",
    }
    mismatches = sorted(name for name, matches in comparisons.items() if not matches)
    if mismatches:
        raise MageCodecCacheV2Error(
            "declared codec policy differs from effective Provider V2 config: "
            + ", ".join(mismatches)
        )


def mage_codec_v2_namespace_identity(
    *,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    provider_implementation_sha256: Sha256Digest,
    effective_config_sha256: Sha256Digest,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "namespace_version": MAGE_CODEC_CACHE_NAMESPACE_V2_VERSION,
            "provider_version": MAGE_DCVC_PROVIDER_VERSION,
            "recipe_version": MAGE_DCVC_RECIPE_VERSION,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": codec_policy_sha256,
            "provider_implementation_sha256": provider_implementation_sha256,
            "effective_config_sha256": effective_config_sha256,
        }
    )


def mage_codec_v2_logical_cache_identity(
    *,
    source_content_sha256: Sha256Digest,
    source_byte_count: int,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    provider_implementation_sha256: Sha256Digest,
    effective_config_sha256: Sha256Digest,
    preparation_identity: Sha256Digest,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "entry_version": MAGE_CODEC_CACHE_ENTRY_V2_VERSION,
            "source_content_sha256": source_content_sha256,
            "source_byte_count": source_byte_count,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": codec_policy_sha256,
            "provider_implementation_sha256": provider_implementation_sha256,
            "effective_config_sha256": effective_config_sha256,
            "preparation_identity": preparation_identity,
        }
    )


def upstream_mage_codec_cache_directory_name(
    *,
    video_path: Path,
    codec_policy: MageVideoCodecPolicy,
    model_directory: Path,
) -> str:
    """Reproduce the checkpoint's current provider cache key for strict pre-admission.

    This path-dependent upstream locator is not a Robata identity. The v2 logical identity
    remains content-addressed. Its implementation is recipe-bound and parity-tested against
    the qualified checkpoint before real use.
    """

    if codec_policy.codec_mode != "neural" or codec_policy.neural_parameters is None:
        raise MageCodecCacheV2Error("upstream DCVC cache naming requires neural parameters")
    parameters = codec_policy.neural_parameters
    model_root = Path(model_directory).expanduser().resolve()
    preprocessor_path = model_root / "preprocessor_config.json"
    try:
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MageCodecCacheV2Error("could not read Mage preprocessor_config.json") from error
    dcvc_block = (preprocessor.get("codec") or {}).get("dcvc") or {}
    if not isinstance(dcvc_block, dict):
        raise MageCodecCacheV2Error("preprocessor codec.dcvc must be an object")
    video = Path(video_path).expanduser().resolve()
    raw = (
        f"{video}|eng=dcvc-rt|tc={codec_policy.target_canvas}|gs={codec_policy.group_size}"
        f"|ipg={codec_policy.images_per_group}|patch={codec_policy.patch_size}"
        f"|mp={codec_policy.max_pixels}|mask=off"
        f"|dqp={parameters.quantization_parameter}|drst={parameters.reset_interval}"
        f"|dip={parameters.intra_period}|dms={parameters.max_side}"
        f"|dpatch=16|dseq={parameters.sequence_length_frames}"
        f"|dcts={parameters.canvas_token_side}"
        "|dcvcblk=" + json.dumps(dcvc_block, sort_keys=True, separators=(",", ":"))
    )
    key = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()
    return f"{video.stem}_{key}"


def build_mage_codec_cache_manifest_v2(
    *,
    checkpoint_manifest: MageCheckpointManifest,
    codec_policy: MageVideoCodecPolicy,
    effective_config: MageDcvcEffectiveConfig,
    cache_base_root: Path,
    model_directory: Path,
    observations: Sequence[tuple[Path, Path, Literal["BUILT", "VERIFIED_HIT"], float]],
    prewarm_wall_seconds: float,
) -> MageCodecCacheManifestV2:
    """Build a v2 cache manifest from independently verified worker output directories."""

    validate_mage_dcvc_effective_config_for_policy(
        effective_config=effective_config,
        codec_policy=codec_policy,
    )
    base_root = Path(cache_base_root).expanduser().resolve()
    policy_sha = build_mage_video_codec_policy_identity(codec_policy).policy_sha256
    namespace = mage_codec_v2_namespace_identity(
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_sha,
        provider_implementation_sha256=effective_config.provider_implementation_sha256,
        effective_config_sha256=effective_config.effective_config_sha256,
    )
    qualified_root = base_root / namespace
    entries: list[MageCodecCacheManifestEntryV2] = []
    for source_path, output_directory, admission, wall_seconds in observations:
        source = Path(source_path).expanduser().resolve()
        output = Path(output_directory).expanduser().resolve()
        if output.parent != qualified_root:
            raise MageCodecCacheV2Error("worker output is outside the qualified v2 cache root")
        expected_name = upstream_mage_codec_cache_directory_name(
            video_path=source,
            codec_policy=codec_policy,
            model_directory=model_directory,
        )
        if output.name != expected_name:
            raise MageCodecCacheV2Error("worker output directory does not match Mage cache locator")
        entry = build_mage_codec_cache_entry_v2(
            checkpoint_manifest=checkpoint_manifest,
            codec_policy=codec_policy,
            effective_config=effective_config,
            source_path=source,
            provider_cache_directory=output,
            namespace_identity=namespace,
        )
        if admission == "BUILT":
            _write_entry_sidecar(directory=output, entry=entry)
        verified = verify_mage_codec_cache_entry_v2(
            cache_directory=output,
            expected_entry=entry,
            effective_config=effective_config,
        )
        entries.append(
            MageCodecCacheManifestEntryV2(
                source_path=str(source),
                logical_cache_identity=verified.logical_cache_identity,
                entry_semantic_sha256=verified.entry_semantic_sha256,
                provider_cache_directory=str(output),
                admission=admission,
                preparation_wall_seconds=max(0.0, float(wall_seconds)),
            )
        )
    ordered = tuple(sorted(entries, key=lambda item: item.source_path))
    built = sum(item.admission == "BUILT" for item in ordered)
    values: dict[str, Any] = {
        "checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
        "codec_policy_sha256": policy_sha,
        "provider_implementation_sha256": effective_config.provider_implementation_sha256,
        "effective_config": effective_config,
        "namespace_identity": namespace,
        "cache_base_root": str(base_root),
        "qualified_cache_root": str(qualified_root),
        "entry_count": len(ordered),
        "built_count": built,
        "verified_hit_count": len(ordered) - built,
        "prewarm_wall_seconds": max(0.0, float(prewarm_wall_seconds)),
        "entries": ordered,
    }
    if not ordered:
        raise MageCodecCacheV2Error("at least one Provider V2 cache entry is required")
    provisional = MageCodecCacheManifestV2.model_construct(
        **values,
        manifest_semantic_sha256="0" * 64,
    )
    return MageCodecCacheManifestV2(
        **values,
        manifest_semantic_sha256=_manifest_semantic_sha256(provisional),
    )


def build_mage_codec_cache_entry_v2(
    *,
    checkpoint_manifest: MageCheckpointManifest,
    codec_policy: MageVideoCodecPolicy,
    effective_config: MageDcvcEffectiveConfig,
    source_path: Path,
    provider_cache_directory: Path,
    namespace_identity: Sha256Digest,
) -> MageCodecCacheEntryV2:
    source = Path(source_path).expanduser().resolve()
    directory = Path(provider_cache_directory).expanduser().resolve()
    source_sha, source_bytes = _file_sha256(source)
    artifact, artifact_exact_sha = _load_preparation_artifact(directory)
    _validate_provider_metadata(artifact=artifact, effective_config=effective_config)
    if (
        artifact.source_content_sha256 != source_sha
        or artifact.source_byte_count != source_bytes
        or artifact.effective_config_sha256 != effective_config.effective_config_sha256
        or artifact.provider_implementation_sha256
        != effective_config.provider_implementation_sha256
    ):
        raise MageCodecCacheV2Error("worker preparation artifact does not match source/config")
    policy_sha = build_mage_video_codec_policy_identity(codec_policy).policy_sha256
    logical = mage_codec_v2_logical_cache_identity(
        source_content_sha256=source_sha,
        source_byte_count=source_bytes,
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_sha,
        provider_implementation_sha256=effective_config.provider_implementation_sha256,
        effective_config_sha256=effective_config.effective_config_sha256,
        preparation_identity=artifact.preparation_identity,
    )
    values: dict[str, Any] = {
        "source_path": str(source),
        "source_content_sha256": source_sha,
        "source_byte_count": source_bytes,
        "checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
        "codec_policy_sha256": policy_sha,
        "provider_implementation_sha256": effective_config.provider_implementation_sha256,
        "effective_config_sha256": effective_config.effective_config_sha256,
        "preparation_identity": artifact.preparation_identity,
        "preparation_artifact_semantic_sha256": artifact.artifact_semantic_sha256,
        "preparation_artifact_exact_sha256": artifact_exact_sha,
        "logical_cache_identity": logical,
        "namespace_identity": namespace_identity,
        "provider_cache_directory_name": directory.name,
        "assets": artifact.assets,
        "asset_set_sha256": _asset_set_sha256(artifact.assets),
        "provider_metadata": artifact.provider_metadata,
    }
    provisional = MageCodecCacheEntryV2.model_construct(
        **values,
        entry_semantic_sha256="0" * 64,
    )
    return MageCodecCacheEntryV2(
        **values,
        entry_semantic_sha256=_entry_semantic_sha256(provisional),
    )


def verify_mage_codec_cache_entry_v2(
    *,
    cache_directory: Path,
    expected_entry: MageCodecCacheEntryV2 | None = None,
    effective_config: MageDcvcEffectiveConfig | None = None,
) -> MageCodecCacheEntryV2:
    directory = Path(cache_directory).expanduser().resolve()
    sidecar = directory / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME
    try:
        raw = sidecar.read_bytes()
        entry = MageCodecCacheEntryV2.model_validate_json(raw, strict=True)
    except (OSError, ValidationError) as error:
        raise MageCodecCacheV2Error("Provider V2 cache entry sidecar is invalid") from error
    if canonical_json_bytes(entry.model_dump(mode="json")) != raw:
        raise MageCodecCacheV2Error("Provider V2 cache entry sidecar must be canonical JSON")
    if expected_entry is not None and entry != expected_entry:
        raise MageCodecCacheV2Error("Provider V2 cache entry differs from expected entry")
    source_sha, source_bytes = _file_sha256(Path(entry.source_path))
    if source_sha != entry.source_content_sha256 or source_bytes != entry.source_byte_count:
        raise MageCodecCacheV2Error("Provider V2 source content changed")
    artifact, artifact_exact_sha = _load_preparation_artifact(directory)
    if effective_config is not None:
        _validate_provider_metadata(artifact=artifact, effective_config=effective_config)
    if (
        artifact.artifact_semantic_sha256 != entry.preparation_artifact_semantic_sha256
        or artifact_exact_sha != entry.preparation_artifact_exact_sha256
        or artifact.preparation_identity != entry.preparation_identity
        or artifact.assets != entry.assets
        or artifact.provider_metadata != entry.provider_metadata
    ):
        raise MageCodecCacheV2Error("Provider V2 preparation artifact differs from cache entry")
    _verify_assets(directory=directory, assets=entry.assets)
    return entry


def verify_mage_codec_cache_manifest_v2(
    *, manifest: MageCodecCacheManifestV2
) -> tuple[MageCodecCacheEntryV2, ...]:
    base = Path(manifest.cache_base_root).expanduser().resolve()
    qualified = Path(manifest.qualified_cache_root).expanduser().resolve()
    if qualified.parent != base or qualified.name != manifest.namespace_identity:
        raise MageCodecCacheV2Error("Provider V2 qualified cache root binding mismatch")
    verified: list[MageCodecCacheEntryV2] = []
    for observation in manifest.entries:
        directory = Path(observation.provider_cache_directory).expanduser().resolve()
        if directory.parent != qualified:
            raise MageCodecCacheV2Error("Provider V2 cache directory escaped qualified root")
        entry = verify_mage_codec_cache_entry_v2(
            cache_directory=directory, effective_config=manifest.effective_config
        )
        if (
            entry.source_path != observation.source_path
            or entry.logical_cache_identity != observation.logical_cache_identity
            or entry.entry_semantic_sha256 != observation.entry_semantic_sha256
            or entry.checkpoint_manifest_sha256 != manifest.checkpoint_manifest_sha256
            or entry.codec_policy_sha256 != manifest.codec_policy_sha256
            or entry.provider_implementation_sha256 != manifest.provider_implementation_sha256
            or entry.effective_config_sha256 != manifest.effective_config.effective_config_sha256
            or entry.namespace_identity != manifest.namespace_identity
        ):
            raise MageCodecCacheV2Error("Provider V2 manifest entry binding mismatch")
        verified.append(entry)
    return tuple(verified)


def load_mage_codec_cache_manifest_v2(*, path: Path) -> MageCodecCacheManifestV2:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        manifest = MageCodecCacheManifestV2.model_validate_json(raw, strict=True)
    except (OSError, ValidationError) as error:
        raise MageCodecCacheV2Error("Provider V2 cache manifest is invalid") from error
    if canonical_json_bytes(manifest.model_dump(mode="json")) != raw:
        raise MageCodecCacheV2Error("Provider V2 cache manifest must use canonical JSON")
    return manifest


def write_mage_codec_cache_manifest_v2(*, manifest: MageCodecCacheManifestV2, path: Path) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
        temporary.replace(resolved)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise MageCodecCacheV2Error("could not publish Provider V2 cache manifest") from error


def _checkpoint_model_root(
    checkpoint_manifest: MageCheckpointManifest,
    output_directory: Path,
) -> Path:
    # The checkpoint model path is intentionally not stored in MageCheckpointManifest.
    # Callers building a manifest must therefore use the explicit helper below instead
    # of this private placeholder. Keeping this guard prevents path inference from cache.
    raise MageCodecCacheV2Error(
        "model_directory is required to validate the upstream Mage cache locator"
    )


def _validate_provider_metadata(
    *, artifact: MageDcvcPreparationArtifact, effective_config: MageDcvcEffectiveConfig
) -> None:
    metadata = artifact.provider_metadata
    expected: dict[str, JsonValue] = {
        "provider_version": effective_config.provider_version,
        "recipe_version": effective_config.recipe_version,
        "effective_config_sha256": effective_config.effective_config_sha256,
        "provider_implementation_sha256": effective_config.provider_implementation_sha256,
        "engine": effective_config.engine,
        "preparation_device": effective_config.preparation_device,
        "device_concurrency_policy": effective_config.device_concurrency_policy,
        "max_side": effective_config.max_side,
        "configured_sampled_frame_count": effective_config.sampled_frame_count,
        "sequence_length_frames": effective_config.sequence_length_frames,
        "canvas_token_side": effective_config.canvas_token_side,
        "encoded_frame_extent": effective_config.encoded_frame_extent,
        "segment_state_policy": "reset-per-job",
    }
    mismatches = sorted(key for key, value in expected.items() if metadata.get(key) != value)
    sampled = metadata.get("effective_sampled_frame_count")
    if (
        isinstance(sampled, bool)
        or not isinstance(sampled, int)
        or sampled <= 0
        or sampled > effective_config.sampled_frame_count
    ):
        mismatches.append("effective_sampled_frame_count")

    max_encoded_frame_id = metadata.get("max_encoded_frame_id")
    if (
        isinstance(max_encoded_frame_id, bool)
        or not isinstance(max_encoded_frame_id, int)
        or max_encoded_frame_id < 0
        or (
            isinstance(sampled, int)
            and not isinstance(sampled, bool)
            and max_encoded_frame_id < sampled - 1
        )
    ):
        mismatches.append("max_encoded_frame_id")
    if metadata.get("engine_load_count") != 1:
        mismatches.append("engine_load_count")
    engine_load_seconds = metadata.get("engine_load_seconds")
    if (
        isinstance(engine_load_seconds, bool)
        or not isinstance(engine_load_seconds, (int, float))
        or not math.isfinite(float(engine_load_seconds))
        or float(engine_load_seconds) < 0.0
    ):
        mismatches.append("engine_load_seconds")
    worker_completed_job_count = metadata.get("worker_completed_job_count")
    if (
        isinstance(worker_completed_job_count, bool)
        or not isinstance(worker_completed_job_count, int)
        or worker_completed_job_count <= 0
    ):
        mismatches.append("worker_completed_job_count")
    if metadata.get("sequence_reset_count_for_job") != 1:
        mismatches.append("sequence_reset_count_for_job")
    if mismatches:
        raise MageCodecCacheV2Error(
            "worker-authored provider metadata differs from effective config: "
            + ", ".join(sorted(set(mismatches)))
        )


def _load_preparation_artifact(
    directory: Path,
) -> tuple[MageDcvcPreparationArtifact, Sha256Digest]:
    sidecar = directory / MAGE_DCVC_PREPARATION_SIDECAR_NAME
    try:
        raw = sidecar.read_bytes()
        artifact = MageDcvcPreparationArtifact.model_validate_json(raw, strict=True)
    except (OSError, ValidationError) as error:
        raise MageCodecCacheV2Error("Provider V2 preparation sidecar is invalid") from error
    if canonical_json_bytes(artifact.model_dump(mode="json")) != raw:
        raise MageCodecCacheV2Error("Provider V2 preparation sidecar must be canonical JSON")
    return artifact, hashlib.sha256(raw).hexdigest()


def _verify_assets(*, directory: Path, assets: Sequence[MageDcvcPreparedAsset]) -> None:
    for asset in assets:
        path = (directory / Path(PurePosixPath(asset.relative_path))).resolve()
        if directory not in path.parents or not path.is_file() or path.is_symlink():
            raise MageCodecCacheV2Error("Provider V2 asset path is unsafe or missing")
        digest, byte_count = _file_sha256(path)
        if digest != asset.sha256 or byte_count != asset.byte_count:
            raise MageCodecCacheV2Error("Provider V2 asset bytes changed")


def _write_entry_sidecar(*, directory: Path, entry: MageCodecCacheEntryV2) -> None:
    path = directory / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME
    if path.exists():
        raise MageCodecCacheV2Error("Provider V2 cache entry sidecar already exists")
    path.write_bytes(canonical_json_bytes(entry.model_dump(mode="json")))


def _file_sha256(path: Path) -> tuple[Sha256Digest, int]:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageCodecCacheV2Error(f"could not read exact file: {resolved}") from error
    if byte_count <= 0:
        raise MageCodecCacheV2Error(f"required exact file is empty: {resolved}")
    return digest.hexdigest(), byte_count


def _asset_set_sha256(assets: Sequence[MageDcvcPreparedAsset]) -> Sha256Digest:
    return semantic_sha256([asset.model_dump(mode="json") for asset in assets])


def _entry_semantic_sha256(entry: MageCodecCacheEntryV2) -> Sha256Digest:
    return semantic_sha256(entry.model_dump(mode="json", exclude={"entry_semantic_sha256"}))


def _manifest_semantic_sha256(manifest: MageCodecCacheManifestV2) -> Sha256Digest:
    return semantic_sha256(
        manifest.model_dump(
            mode="json",
            exclude={"manifest_semantic_sha256", "cache_base_root", "qualified_cache_root"},
        )
    )


__all__ = [
    "MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME",
    "MAGE_CODEC_CACHE_ENTRY_V2_VERSION",
    "MAGE_CODEC_CACHE_MANIFEST_V2_VERSION",
    "MAGE_CODEC_CACHE_NAMESPACE_V2_VERSION",
    "MageCodecCacheEntryV2",
    "MageCodecCacheManifestEntryV2",
    "MageCodecCacheManifestV2",
    "MageCodecCacheV2Error",
    "build_mage_codec_cache_entry_v2",
    "build_mage_codec_cache_manifest_v2",
    "load_mage_codec_cache_manifest_v2",
    "mage_codec_v2_logical_cache_identity",
    "mage_codec_v2_namespace_identity",
    "upstream_mage_codec_cache_directory_name",
    "validate_mage_dcvc_effective_config_for_policy",
    "verify_mage_codec_cache_entry_v2",
    "verify_mage_codec_cache_manifest_v2",
    "write_mage_codec_cache_manifest_v2",
]
