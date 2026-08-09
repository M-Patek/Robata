"""Identity-namespaced, exact prewarm cache for Mage's native codec path.

Cache bytes are operational acceleration state, never authoritative inference
state.  A cache hit is admitted only when immutable media bytes, the verified
checkpoint, the declared codec policy, the observed provider recipe, the build
environment, and every generated asset still match a canonical sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal

from pydantic import Field, JsonValue, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.inference.mage_checkpoint_identity import MageCheckpointManifest
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    build_mage_video_codec_policy_identity,
)

MAGE_CODEC_CACHE_ENTRY_VERSION: Final = "mage-codec-cache-entry-v1"
MAGE_CODEC_CACHE_MANIFEST_VERSION: Final = "mage-codec-cache-manifest-v1"
MAGE_CODEC_EFFECTIVE_RECIPE_VERSION: Final = "mage-dcvc-readiness-observed-v1"
MAGE_CODEC_CACHE_NAMESPACE_VERSION: Final = "mage-codec-cache-namespace-v1"
MAGE_CODEC_CACHE_SIDECAR_NAME: Final = ".robata-entry-v1.json"
MAGE_CODEC_CACHE_SIDECAR_DIRECTORY: Final = ".robata-entries"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

_IMPLEMENTATION_FILES: Final = (
    "codec_video_processing_mage_vl.py",
    "preprocessor_config.json",
    "processing_mage_vl.py",
    "neural_codec/codec_dcvc_config.py",
    "neural_codec/dcvc_readiness_gen.py",
    "neural_codec/dcvc_rt_engine.py",
    "neural_codec/codec_tools/pipeline/process_video_bitcost_readiness.py",
)


class MageCodecCacheError(RuntimeError):
    """A Mage native-codec cache entry could not be built or verified safely."""


class MageCodecCacheAsset(StrictModel):
    """One exact file in a provider-generated cache directory."""

    relative_path: NonEmptyString
    byte_count: PositiveInt
    sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_relative_path(self) -> MageCodecCacheAsset:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("relative_path must be a safe relative POSIX path")
        return self


class MageCodecEffectiveRecipe(StrictModel):
    """Pinned source, config, and host inputs for the currently observed provider path."""

    recipe_version: Literal["mage-dcvc-readiness-observed-v1"] = MAGE_CODEC_EFFECTIVE_RECIPE_VERSION
    implementation_files: tuple[MageCodecCacheAsset, ...]
    effective_projection: dict[str, JsonValue]
    environment_projection: dict[str, JsonValue]
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_recipe(self) -> MageCodecEffectiveRecipe:
        if tuple(item.relative_path for item in self.implementation_files) != tuple(
            sorted(item.relative_path for item in self.implementation_files)
        ):
            raise ValueError("implementation_files must be sorted")
        if self.semantic_sha256 != _recipe_semantic_sha256(self):
            raise ValueError("recipe semantic_sha256 does not match its projection")
        return self


class MageCodecCacheEntry(StrictModel):
    """Canonical exact-byte sidecar for one prewarmed source path."""

    entry_version: Literal["mage-codec-cache-entry-v1"] = MAGE_CODEC_CACHE_ENTRY_VERSION
    namespace_version: Literal["mage-codec-cache-namespace-v1"] = MAGE_CODEC_CACHE_NAMESPACE_VERSION
    source_path: NonEmptyString
    source_content_sha256: Sha256Digest
    source_byte_count: PositiveInt
    checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    recipe_sha256: Sha256Digest
    logical_cache_identity: Sha256Digest
    namespace_identity: Sha256Digest
    provider_cache_directory_name: NonEmptyString
    assets: tuple[MageCodecCacheAsset, ...]
    asset_set_sha256: Sha256Digest
    entry_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_entry(self) -> MageCodecCacheEntry:
        if Path(self.provider_cache_directory_name).name != self.provider_cache_directory_name:
            raise ValueError("provider_cache_directory_name must contain one path component")
        if tuple(item.relative_path for item in self.assets) != tuple(
            sorted(item.relative_path for item in self.assets)
        ):
            raise ValueError("assets must be sorted")
        if self.logical_cache_identity != mage_codec_logical_cache_identity(
            source_content_sha256=self.source_content_sha256,
            source_byte_count=self.source_byte_count,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            recipe_sha256=self.recipe_sha256,
        ):
            raise ValueError("logical_cache_identity does not match its bound inputs")
        if self.namespace_identity != mage_codec_namespace_identity(
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            recipe_sha256=self.recipe_sha256,
        ):
            raise ValueError("namespace_identity does not match its bound inputs")
        if self.asset_set_sha256 != _asset_set_sha256(self.assets):
            raise ValueError("asset_set_sha256 does not match assets")
        if self.entry_semantic_sha256 != _entry_semantic_sha256(self):
            raise ValueError("entry_semantic_sha256 does not match entry")
        return self


class MageCodecCacheManifestEntry(StrictModel):
    """One hit/build observation in a cache qualification run."""

    source_path: NonEmptyString
    logical_cache_identity: Sha256Digest
    entry_semantic_sha256: Sha256Digest
    provider_cache_directory: NonEmptyString
    admission: Literal["BUILT", "VERIFIED_HIT"]


class MageCodecCacheManifest(StrictModel):
    """Operational manifest for one identity-qualified cache namespace."""

    manifest_version: Literal["mage-codec-cache-manifest-v1"] = MAGE_CODEC_CACHE_MANIFEST_VERSION
    namespace_version: Literal["mage-codec-cache-namespace-v1"] = MAGE_CODEC_CACHE_NAMESPACE_VERSION
    checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    recipe: MageCodecEffectiveRecipe
    namespace_identity: Sha256Digest
    cache_base_root: NonEmptyString
    qualified_cache_root: NonEmptyString
    entry_count: PositiveInt
    built_count: NonNegativeInt
    verified_hit_count: NonNegativeInt
    prewarm_wall_seconds: NonNegativeFloat
    entries: tuple[MageCodecCacheManifestEntry, ...]
    manifest_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> MageCodecCacheManifest:
        if self.namespace_identity != mage_codec_namespace_identity(
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            recipe_sha256=self.recipe.semantic_sha256,
        ):
            raise ValueError("manifest namespace_identity does not match its inputs")
        if Path(self.qualified_cache_root).name != self.namespace_identity:
            raise ValueError("qualified_cache_root must end with namespace_identity")
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count does not match entries")
        if self.built_count + self.verified_hit_count != self.entry_count:
            raise ValueError("built_count and verified_hit_count do not cover entries")
        if tuple(item.source_path for item in self.entries) != tuple(
            sorted(item.source_path for item in self.entries)
        ):
            raise ValueError("manifest entries must be sorted by source_path")
        if self.manifest_semantic_sha256 != _manifest_semantic_sha256(self):
            raise ValueError("manifest_semantic_sha256 does not match manifest")
        return self


ProcessorFactory = Callable[[Path], Any]


def mage_codec_namespace_identity(
    *,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    recipe_sha256: Sha256Digest,
) -> Sha256Digest:
    """Name one cache root by model, declared policy, and effective execution recipe."""

    return semantic_sha256(
        {
            "namespace_version": MAGE_CODEC_CACHE_NAMESPACE_VERSION,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": codec_policy_sha256,
            "recipe_sha256": recipe_sha256,
        }
    )


def mage_codec_logical_cache_identity(
    *,
    source_content_sha256: Sha256Digest,
    source_byte_count: int,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    recipe_sha256: Sha256Digest,
) -> Sha256Digest:
    """Address one cache asset independently of its durable mount path."""

    return semantic_sha256(
        {
            "entry_version": MAGE_CODEC_CACHE_ENTRY_VERSION,
            "source_content_sha256": source_content_sha256,
            "source_byte_count": source_byte_count,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": codec_policy_sha256,
            "recipe_sha256": recipe_sha256,
        }
    )


def build_mage_codec_effective_recipe(
    *,
    model_directory: Path,
    codec_policy: MageVideoCodecPolicy,
    environment_projection: Mapping[str, JsonValue] | None = None,
) -> MageCodecEffectiveRecipe:
    """Capture the real v2 readiness behavior instead of the misleading declared subset."""

    if codec_policy.codec_mode != "neural":
        raise MageCodecCacheError("native DCVC prewarm requires codec_mode='neural'")
    model_root = Path(model_directory).expanduser().resolve()
    implementation: list[MageCodecCacheAsset] = []
    for relative in _IMPLEMENTATION_FILES:
        digest, byte_count = _exact_file_sha256(model_root / Path(PurePosixPath(relative)))
        implementation.append(
            MageCodecCacheAsset(relative_path=relative, byte_count=byte_count, sha256=digest)
        )
    ordered = tuple(sorted(implementation, key=lambda item: item.relative_path))
    preprocessor = _read_json_object(model_root / "preprocessor_config.json")
    dcvc_block = (preprocessor.get("codec") or {}).get("dcvc") or {}
    if not isinstance(dcvc_block, dict):
        raise MageCodecCacheError("preprocessor_config codec.dcvc must be an object")
    native = codec_policy.native_codec_config()
    declared_dcvc = native.get("dcvc")
    if not isinstance(declared_dcvc, Mapping):
        raise MageCodecCacheError("neural policy produced no dcvc mapping")
    projection: dict[str, JsonValue] = {
        "engine": "dcvc-rt",
        "num_sampled_frames_formula": (
            "min((target_canvas/images_per_group)*group_size,total_frames)"
        ),
        "group_size": codec_policy.group_size,
        "images_per_group": codec_policy.images_per_group,
        "patch": codec_policy.patch_size,
        "max_pixels": codec_policy.max_pixels,
        "min_group_frames": codec_policy.min_group_frames,
        "max_group_frames_source": "preprocessor_config.codec.dcvc.max_group_frames",
        "max_group_frames": int(dcvc_block.get("max_group_frames", 128)),
        "qp_source": "preprocessor_config.codec.dcvc",
        "qp": int(dcvc_block.get("qp", 42)),
        "reset_interval": int(dcvc_block.get("reset_interval", 64)),
        "intra_period": int(dcvc_block.get("intra_period", -1)),
        "max_side": int(dcvc_block.get("max_side", 0)),
        "sequence_length_behavior": (
            "declared seq_len_frames does not limit current online readiness"
        ),
        "readiness_coverage_bins": int(declared_dcvc["readiness_coverage_bins"]),
        "readiness_delta_ratio": float(declared_dcvc["readiness_delta_ratio"]),
        "bitcost_percentile": int(declared_dcvc["bitcost_pct"]),
        "decode_backsearch_max": int(declared_dcvc["decode_backsearch_max"]),
        "preprocess_device": codec_policy.preprocess_device,
    }
    environment = dict(environment_projection or _runtime_environment_projection())
    provisional = MageCodecEffectiveRecipe.model_construct(
        implementation_files=ordered,
        effective_projection=projection,
        environment_projection=environment,
        semantic_sha256="0" * 64,
    )
    return MageCodecEffectiveRecipe(
        implementation_files=ordered,
        effective_projection=projection,
        environment_projection=environment,
        semantic_sha256=_recipe_semantic_sha256(provisional),
    )


def prewarm_mage_codec_cache(
    *,
    model_directory: Path,
    checkpoint_manifest: MageCheckpointManifest,
    codec_policy: MageVideoCodecPolicy,
    cache_base_root: Path,
    video_paths: Sequence[Path],
    prompt: str = "Observe this video segment.",
    processor_factory: ProcessorFactory | None = None,
    environment_projection: Mapping[str, JsonValue] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MageCodecCacheManifest:
    """Run the exact resident AutoProcessor codec path once for every cache miss."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise MageCodecCacheError("prewarm prompt must be nonempty")
    if codec_policy.codec_mode != "neural":
        raise MageCodecCacheError("prewarm supports only codec_mode='neural'")
    if not isinstance(checkpoint_manifest, MageCheckpointManifest):
        raise TypeError("checkpoint_manifest must be MageCheckpointManifest")
    model_root = Path(model_directory).expanduser().resolve()
    base_root = Path(cache_base_root).expanduser().resolve()
    videos = tuple(sorted({Path(item).expanduser().resolve() for item in video_paths}, key=str))
    if not videos:
        raise MageCodecCacheError("at least one video path is required")
    recipe = build_mage_codec_effective_recipe(
        model_directory=model_root,
        codec_policy=codec_policy,
        environment_projection=environment_projection,
    )
    policy_identity = build_mage_video_codec_policy_identity(codec_policy)
    namespace_identity = mage_codec_namespace_identity(
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_identity.policy_sha256,
        recipe_sha256=recipe.semantic_sha256,
    )
    qualified_root = base_root / namespace_identity
    qualified_root.mkdir(parents=True, exist_ok=True)
    _reject_unbound_provider_directories(qualified_root)

    existing = _verified_entries_by_source(
        cache_root=qualified_root,
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_identity.policy_sha256,
        recipe_sha256=recipe.semantic_sha256,
        namespace_identity=namespace_identity,
    )
    pending: list[tuple[Path, str, int, str]] = []
    observations: list[MageCodecCacheManifestEntry] = []
    for video in videos:
        digest, byte_count = _exact_file_sha256(video)
        logical = mage_codec_logical_cache_identity(
            source_content_sha256=digest,
            source_byte_count=byte_count,
            checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
            codec_policy_sha256=policy_identity.policy_sha256,
            recipe_sha256=recipe.semantic_sha256,
        )
        cached = existing.get(str(video))
        if cached is not None:
            if cached.logical_cache_identity != logical:
                raise MageCodecCacheError("cache sidecar source path now has different bytes")
            provider_dir = qualified_root / cached.provider_cache_directory_name
            observations.append(
                MageCodecCacheManifestEntry(
                    source_path=str(video),
                    logical_cache_identity=logical,
                    entry_semantic_sha256=cached.entry_semantic_sha256,
                    provider_cache_directory=str(provider_dir),
                    admission="VERIFIED_HIT",
                )
            )
        else:
            pending.append((video, digest, byte_count, logical))

    started = clock()
    if pending:
        processor = (processor_factory or _default_processor_factory)(model_root)
        native_config = _native_processor_codec_config(
            codec_policy=codec_policy,
            model_directory=model_root,
            cache_root=qualified_root,
        )
        overlay_root = qualified_root / ".robata-python-overlay"
        _write_dcvc_namespace_overlay(overlay_root=overlay_root, model_root=model_root)
        environment = {
            "PYTHONPATH": _prepend_pythonpath(overlay_root, os.environ.get("PYTHONPATH")),
            "DCVC_INTRA_TAR": str(model_root / "neural_codec" / "dcvc_rt_intra.tar"),
            "DCVC_INTER_TAR": str(model_root / "neural_codec" / "dcvc_rt_inter.tar"),
            "DCVC_DEVICE": codec_policy.preprocess_device,
            "DCVC_ENGINE_DIR": str(model_root / "neural_codec"),
            "DCVC_REPO_DIR": str(model_root / "neural_codec"),
        }
        with _temporary_environment(environment):
            for video, digest, byte_count, logical in pending:
                before = _provider_cache_directories(qualified_root)
                _invoke_processor_prewarm(
                    processor=processor,
                    video_path=video,
                    prompt=prompt,
                    codec_config=native_config,
                    max_pixels=codec_policy.max_pixels,
                )
                created = _provider_cache_directories(qualified_root) - before
                if len(created) != 1:
                    raise MageCodecCacheError(
                        "exact provider prewarm created "
                        f"{len(created)} cache directories for {video}"
                    )
                provider_dir = qualified_root / next(iter(created))
                assets = _collect_asset_files(provider_dir)
                entry = _build_entry(
                    source_path=video,
                    source_content_sha256=digest,
                    source_byte_count=byte_count,
                    checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
                    codec_policy_sha256=policy_identity.policy_sha256,
                    recipe_sha256=recipe.semantic_sha256,
                    logical_cache_identity=logical,
                    namespace_identity=namespace_identity,
                    provider_cache_directory_name=provider_dir.name,
                    assets=assets,
                )
                _write_canonical_json(
                    mage_codec_cache_sidecar_path(provider_dir),
                    entry.model_dump(mode="json"),
                )
                verified = verify_mage_codec_cache_entry(
                    cache_directory=provider_dir,
                    expected_source_path=video,
                    expected_checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
                    expected_codec_policy_sha256=policy_identity.policy_sha256,
                    expected_recipe_sha256=recipe.semantic_sha256,
                    expected_namespace_identity=namespace_identity,
                )
                observations.append(
                    MageCodecCacheManifestEntry(
                        source_path=str(video),
                        logical_cache_identity=logical,
                        entry_semantic_sha256=verified.entry_semantic_sha256,
                        provider_cache_directory=str(provider_dir),
                        admission="BUILT",
                    )
                )
    elapsed = max(0.0, float(clock() - started))
    ordered = tuple(sorted(observations, key=lambda item: item.source_path))
    built_count = sum(item.admission == "BUILT" for item in ordered)
    provisional = MageCodecCacheManifest.model_construct(
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_identity.policy_sha256,
        recipe=recipe,
        namespace_identity=namespace_identity,
        cache_base_root=str(base_root),
        qualified_cache_root=str(qualified_root),
        entry_count=len(ordered),
        built_count=built_count,
        verified_hit_count=len(ordered) - built_count,
        prewarm_wall_seconds=elapsed,
        entries=ordered,
        manifest_semantic_sha256="0" * 64,
    )
    return MageCodecCacheManifest(
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        codec_policy_sha256=policy_identity.policy_sha256,
        recipe=recipe,
        namespace_identity=namespace_identity,
        cache_base_root=str(base_root),
        qualified_cache_root=str(qualified_root),
        entry_count=len(ordered),
        built_count=built_count,
        verified_hit_count=len(ordered) - built_count,
        prewarm_wall_seconds=elapsed,
        entries=ordered,
        manifest_semantic_sha256=_manifest_semantic_sha256(provisional),
    )


def mage_codec_cache_sidecar_path(cache_directory: Path) -> Path:
    """Return the short, deterministic sidecar path outside the provider directory."""

    directory = Path(cache_directory).expanduser().resolve()
    sidecar_key = semantic_sha256({"provider_cache_directory_name": directory.name})
    return directory.parent / MAGE_CODEC_CACHE_SIDECAR_DIRECTORY / f"{sidecar_key}.json"


def verify_mage_codec_cache_manifest(
    *, manifest: MageCodecCacheManifest
) -> tuple[MageCodecCacheEntry, ...]:
    """Rehash every source, sidecar, and provider asset in a prewarm manifest."""

    cache_base_root = Path(manifest.cache_base_root).expanduser().resolve()
    qualified_cache_root = Path(manifest.qualified_cache_root).expanduser().resolve()
    if qualified_cache_root.parent != cache_base_root:
        raise MageCodecCacheError("qualified cache root is outside cache_base_root")
    if qualified_cache_root.name != manifest.namespace_identity:
        raise MageCodecCacheError("qualified cache root namespace binding mismatch")

    verified: list[MageCodecCacheEntry] = []
    for item in manifest.entries:
        provider_cache_directory = Path(item.provider_cache_directory).expanduser().resolve()
        if provider_cache_directory.parent != qualified_cache_root:
            raise MageCodecCacheError(
                "manifest provider cache directory is outside qualified cache root"
            )
        entry = verify_mage_codec_cache_entry(
            cache_directory=provider_cache_directory,
            expected_source_path=Path(item.source_path),
            expected_checkpoint_manifest_sha256=manifest.checkpoint_manifest_sha256,
            expected_codec_policy_sha256=manifest.codec_policy_sha256,
            expected_recipe_sha256=manifest.recipe.semantic_sha256,
            expected_namespace_identity=manifest.namespace_identity,
        )
        if entry.logical_cache_identity != item.logical_cache_identity:
            raise MageCodecCacheError("manifest logical cache identity mismatch")
        if entry.entry_semantic_sha256 != item.entry_semantic_sha256:
            raise MageCodecCacheError("manifest entry semantic identity mismatch")
        verified.append(entry)
    return tuple(verified)


def verify_mage_codec_cache_entry(
    *,
    cache_directory: Path,
    expected_source_path: Path | None = None,
    expected_checkpoint_manifest_sha256: Sha256Digest | None = None,
    expected_codec_policy_sha256: Sha256Digest | None = None,
    expected_recipe_sha256: Sha256Digest | None = None,
    expected_namespace_identity: Sha256Digest | None = None,
) -> MageCodecCacheEntry:
    """Fail closed unless a cache directory and every declared asset match exactly."""

    directory = Path(cache_directory).expanduser().resolve()
    sidecar_path = mage_codec_cache_sidecar_path(directory)
    try:
        raw = sidecar_path.read_bytes()
        entry = MageCodecCacheEntry.model_validate_json(raw, strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise MageCodecCacheError(f"cache sidecar is absent or invalid: {directory}") from error
    if canonical_json_bytes(entry.model_dump(mode="json")) != raw:
        raise MageCodecCacheError("cache sidecar is not canonical JSON")
    if entry.provider_cache_directory_name != directory.name:
        raise MageCodecCacheError("provider cache directory binding mismatch")
    if expected_source_path is not None:
        source = Path(expected_source_path).expanduser().resolve()
        digest, byte_count = _exact_file_sha256(source)
        if entry.source_path != str(source):
            raise MageCodecCacheError("cache source path mismatch")
        if entry.source_content_sha256 != digest or entry.source_byte_count != byte_count:
            raise MageCodecCacheError("cache source content mismatch")
    expected_pairs = (
        ("checkpoint", entry.checkpoint_manifest_sha256, expected_checkpoint_manifest_sha256),
        ("codec policy", entry.codec_policy_sha256, expected_codec_policy_sha256),
        ("recipe", entry.recipe_sha256, expected_recipe_sha256),
        ("namespace", entry.namespace_identity, expected_namespace_identity),
    )
    for label, actual, expected in expected_pairs:
        if expected is not None and actual != expected:
            raise MageCodecCacheError(f"cache {label} identity mismatch")
    if _collect_asset_files(directory) != entry.assets:
        raise MageCodecCacheError("cache asset bytes do not match sidecar")
    return entry


def write_mage_codec_cache_manifest(*, manifest: MageCodecCacheManifest, path: Path) -> None:
    """Persist canonical operational evidence without publishing a wire schema."""

    if not isinstance(manifest, MageCodecCacheManifest):
        raise TypeError("manifest must be MageCodecCacheManifest")
    _write_canonical_json(Path(path).expanduser().resolve(), manifest.model_dump(mode="json"))


def load_mage_codec_cache_manifest(*, path: Path) -> MageCodecCacheManifest:
    """Load a canonical self-validating cache manifest."""

    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        manifest = MageCodecCacheManifest.model_validate_json(raw, strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise MageCodecCacheError(f"could not load cache manifest: {resolved}") from error
    if canonical_json_bytes(manifest.model_dump(mode="json")) != raw:
        raise MageCodecCacheError("cache manifest is not canonical JSON")
    return manifest


def _verified_entries_by_source(
    *,
    cache_root: Path,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    recipe_sha256: Sha256Digest,
    namespace_identity: Sha256Digest,
) -> dict[str, MageCodecCacheEntry]:
    entries: dict[str, MageCodecCacheEntry] = {}
    for provider_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        if provider_dir.name.startswith(".robata-"):
            continue
        entry = verify_mage_codec_cache_entry(
            cache_directory=provider_dir,
            expected_checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            expected_codec_policy_sha256=codec_policy_sha256,
            expected_recipe_sha256=recipe_sha256,
            expected_namespace_identity=namespace_identity,
        )
        if entry.source_path in entries:
            raise MageCodecCacheError("cache namespace contains duplicate source-path sidecars")
        entries[entry.source_path] = entry
    return entries


def _reject_unbound_provider_directories(cache_root: Path) -> None:
    for child in cache_root.iterdir():
        if (
            child.is_dir()
            and not child.name.startswith(".robata-")
            and not mage_codec_cache_sidecar_path(child).is_file()
        ):
            raise MageCodecCacheError(
                f"qualified cache root contains an unbound provider directory: {child}"
            )


def _provider_cache_directories(cache_root: Path) -> set[str]:
    return {
        child.name
        for child in cache_root.iterdir()
        if child.is_dir()
        and not child.name.startswith(".robata-")
        and (child / "meta.json").is_file()
        and (child / "src_patch_position.npy").is_file()
    }


def _native_processor_codec_config(
    *, codec_policy: MageVideoCodecPolicy, model_directory: Path, cache_root: Path
) -> dict[str, Any]:
    native = codec_policy.native_codec_config()
    preprocess_device = native.pop("preprocess_device")
    dcvc = dict(native.get("dcvc") or {})
    dcvc["device"] = preprocess_device
    dcvc.setdefault("pkg_dir", str(model_directory / "neural_codec"))
    native["dcvc"] = dcvc
    native["cache_root"] = cache_root
    return native


def _invoke_processor_prewarm(
    *,
    processor: Any,
    video_path: Path,
    prompt: str,
    codec_config: Mapping[str, Any],
    max_pixels: int,
) -> None:
    content: list[dict[str, str]] = [{"type": "video"}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        raise MageCodecCacheError("Mage processor exposes no apply_chat_template")
    text = apply_template(messages, tokenize=False, add_generation_prompt=True)
    if not callable(processor):
        raise MageCodecCacheError("Mage processor is not callable")
    try:
        result = processor(
            text=[text],
            videos=[str(video_path)],
            video_backend="codec",
            codec_config=dict(codec_config),
            max_pixels=max_pixels,
            return_tensors="pt",
            padding=True,
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise MageCodecCacheError(
            f"exact Mage processor prewarm failed for {video_path}"
        ) from error
    if result is None:
        raise MageCodecCacheError("Mage processor prewarm returned no inputs")
    del result


def _default_processor_factory(model_directory: Path) -> Any:
    try:
        transformers = import_module("transformers")
        return transformers.AutoProcessor.from_pretrained(
            model_directory,
            local_files_only=True,
            trust_remote_code=True,
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise MageCodecCacheError("could not load the local Mage AutoProcessor") from error


def _build_entry(
    *,
    source_path: Path,
    source_content_sha256: Sha256Digest,
    source_byte_count: int,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    recipe_sha256: Sha256Digest,
    logical_cache_identity: Sha256Digest,
    namespace_identity: Sha256Digest,
    provider_cache_directory_name: str,
    assets: tuple[MageCodecCacheAsset, ...],
) -> MageCodecCacheEntry:
    asset_set_sha256 = _asset_set_sha256(assets)
    provisional = MageCodecCacheEntry.model_construct(
        source_path=str(source_path),
        source_content_sha256=source_content_sha256,
        source_byte_count=source_byte_count,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        codec_policy_sha256=codec_policy_sha256,
        recipe_sha256=recipe_sha256,
        logical_cache_identity=logical_cache_identity,
        namespace_identity=namespace_identity,
        provider_cache_directory_name=provider_cache_directory_name,
        assets=assets,
        asset_set_sha256=asset_set_sha256,
        entry_semantic_sha256="0" * 64,
    )
    return MageCodecCacheEntry(
        source_path=str(source_path),
        source_content_sha256=source_content_sha256,
        source_byte_count=source_byte_count,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        codec_policy_sha256=codec_policy_sha256,
        recipe_sha256=recipe_sha256,
        logical_cache_identity=logical_cache_identity,
        namespace_identity=namespace_identity,
        provider_cache_directory_name=provider_cache_directory_name,
        assets=assets,
        asset_set_sha256=asset_set_sha256,
        entry_semantic_sha256=_entry_semantic_sha256(provisional),
    )


def _collect_asset_files(directory: Path) -> tuple[MageCodecCacheAsset, ...]:
    if (
        not (directory / "meta.json").is_file()
        or not (directory / "src_patch_position.npy").is_file()
    ):
        raise MageCodecCacheError("provider cache lacks meta.json or src_patch_position.npy")
    meta = _read_json_object(directory / "meta.json")
    canvas_files = meta.get("canvas_files")
    if not isinstance(canvas_files, list) or not canvas_files:
        raise MageCodecCacheError("provider cache declares no canvas_files")
    for name in canvas_files:
        if not isinstance(name, str) or Path(name).name != name:
            raise MageCodecCacheError("provider cache has an unsafe canvas filename")
        if not (directory / name).is_file():
            raise MageCodecCacheError(f"provider cache canvas is missing: {name}")
    assets: list[MageCodecCacheAsset] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == MAGE_CODEC_CACHE_SIDECAR_NAME:
            continue
        digest, byte_count = _exact_file_sha256(path)
        assets.append(
            MageCodecCacheAsset(
                relative_path=path.relative_to(directory).as_posix(),
                byte_count=byte_count,
                sha256=digest,
            )
        )
    return tuple(assets)


def _recipe_semantic_sha256(recipe: MageCodecEffectiveRecipe) -> Sha256Digest:
    return semantic_sha256(
        {
            "recipe_version": recipe.recipe_version,
            "implementation_files": [
                item.model_dump(mode="json") for item in recipe.implementation_files
            ],
            "effective_projection": recipe.effective_projection,
            "environment_projection": recipe.environment_projection,
        }
    )


def _entry_semantic_sha256(entry: MageCodecCacheEntry) -> Sha256Digest:
    return semantic_sha256(
        {
            "entry_version": entry.entry_version,
            "namespace_version": entry.namespace_version,
            "source_content_sha256": entry.source_content_sha256,
            "source_byte_count": entry.source_byte_count,
            "checkpoint_manifest_sha256": entry.checkpoint_manifest_sha256,
            "codec_policy_sha256": entry.codec_policy_sha256,
            "recipe_sha256": entry.recipe_sha256,
            "logical_cache_identity": entry.logical_cache_identity,
            "namespace_identity": entry.namespace_identity,
            "provider_cache_directory_name": entry.provider_cache_directory_name,
            "asset_set_sha256": entry.asset_set_sha256,
        }
    )


def _manifest_semantic_sha256(manifest: MageCodecCacheManifest) -> Sha256Digest:
    return semantic_sha256(
        {
            "manifest_version": manifest.manifest_version,
            "namespace_version": manifest.namespace_version,
            "checkpoint_manifest_sha256": manifest.checkpoint_manifest_sha256,
            "codec_policy_sha256": manifest.codec_policy_sha256,
            "recipe_sha256": manifest.recipe.semantic_sha256,
            "namespace_identity": manifest.namespace_identity,
            "entry_semantic_sha256": [item.entry_semantic_sha256 for item in manifest.entries],
        }
    )


def _asset_set_sha256(assets: Sequence[MageCodecCacheAsset]) -> Sha256Digest:
    return semantic_sha256([item.model_dump(mode="json") for item in assets])


def _runtime_environment_projection() -> dict[str, JsonValue]:
    projection: dict[str, JsonValue] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "opencv": _package_version("opencv-python"),
        "numpy": _package_version("numpy"),
        "pillow": _package_version("Pillow"),
    }
    try:
        torch = import_module("torch")
        cuda = getattr(torch, "cuda", None)
        projection["torch_cuda"] = str(getattr(getattr(torch, "version", None), "cuda", None))
        if cuda is not None and bool(cuda.is_available()):
            projection["cuda_device_name"] = str(cuda.get_device_name(0))
            capability = cuda.get_device_capability(0)
            projection["cuda_capability"] = [int(capability[0]), int(capability[1])]
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        projection["torch_cuda"] = None
    return projection


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "ABSENT"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise MageCodecCacheError(f"could not read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise MageCodecCacheError(f"JSON document must be an object: {path}")
    return value


def _exact_file_sha256(path: Path) -> tuple[Sha256Digest, int]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise MageCodecCacheError(f"required file does not exist: {resolved}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageCodecCacheError(f"could not read file: {resolved}") from error
    if byte_count <= 0:
        raise MageCodecCacheError(f"required file is empty: {resolved}")
    return digest.hexdigest(), byte_count


def _write_dcvc_namespace_overlay(*, overlay_root: Path, model_root: Path) -> None:
    package = overlay_root / "src"
    target = (model_root / "neural_codec" / "DCVC" / "src").resolve()
    if not target.is_dir():
        raise MageCodecCacheError(f"Mage DCVC source package is missing: {target}")
    package.mkdir(parents=True, exist_ok=True)
    init_path = package / "__init__.py"
    content = (
        f"# Generated by Robata mage-codec-cache-namespace-v1.\n__path__ = [{str(target)!r}]\n"
    )
    if init_path.exists() and init_path.read_text(encoding="utf-8") != content:
        raise MageCodecCacheError("DCVC namespace overlay exists with different content")
    init_path.write_text(content, encoding="utf-8", newline="\n")


def _prepend_pythonpath(overlay_root: Path, current: str | None) -> str:
    return str(overlay_root) if not current else os.pathsep.join((str(overlay_root), current))


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_canonical_json(path: Path, payload: object) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        temporary.replace(resolved)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise MageCodecCacheError(f"could not write canonical JSON: {resolved}") from error


__all__ = [
    "MAGE_CODEC_CACHE_ENTRY_VERSION",
    "MAGE_CODEC_CACHE_MANIFEST_VERSION",
    "MAGE_CODEC_CACHE_NAMESPACE_VERSION",
    "MAGE_CODEC_CACHE_SIDECAR_DIRECTORY",
    "MAGE_CODEC_CACHE_SIDECAR_NAME",
    "MAGE_CODEC_EFFECTIVE_RECIPE_VERSION",
    "MageCodecCacheAsset",
    "MageCodecCacheEntry",
    "MageCodecCacheError",
    "MageCodecCacheManifest",
    "MageCodecCacheManifestEntry",
    "MageCodecEffectiveRecipe",
    "build_mage_codec_effective_recipe",
    "load_mage_codec_cache_manifest",
    "mage_codec_cache_sidecar_path",
    "mage_codec_logical_cache_identity",
    "mage_codec_namespace_identity",
    "prewarm_mage_codec_cache",
    "verify_mage_codec_cache_entry",
    "verify_mage_codec_cache_manifest",
    "write_mage_codec_cache_manifest",
]
