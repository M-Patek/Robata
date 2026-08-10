"""Exact replay cache for Mage's traditional H.264/HEVC codec backend.

This internal family is additive to DCVC Provider V2. It binds source bytes,
codec policy/configuration, provider implementation, codec-video-prep package
and executable, pinned container image, provider options, and every output
asset. Replay uses Mage's own result loader and never executes cv-preinfer.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    MageVideoEndpointRequest,
    build_mage_video_codec_policy_identity,
)
from robata.inference.mage_video_runtime import (
    MageVideoExactCodecCacheAsset,
    MageVideoTraditionalCodecCacheBinding,
    mage_video_codec_config_sha256,
)

MAGE_TRADITIONAL_CODEC_PROVIDER_VERSION: Final = "robata-mage-traditional-codec-provider-v1"
MAGE_TRADITIONAL_CODEC_TOOLCHAIN_VERSION: Final = "mage-traditional-codec-toolchain-v1"
MAGE_TRADITIONAL_CODEC_EFFECTIVE_CONFIG_VERSION: Final = (
    "mage-traditional-codec-effective-config-v1"
)
MAGE_TRADITIONAL_CODEC_CACHE_ENTRY_VERSION: Final = "mage-traditional-codec-cache-entry-v1"
MAGE_TRADITIONAL_CODEC_CACHE_MANIFEST_VERSION: Final = "mage-traditional-codec-cache-manifest-v1"
MAGE_TRADITIONAL_CODEC_CACHE_NAMESPACE_VERSION: Final = "mage-traditional-codec-cache-namespace-v1"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
TraditionalEngine = Literal["hevc", "cv-preinfer"]


class MageTraditionalCodecCacheError(RuntimeError):
    """A traditional codec identity or provider asset failed verification."""


class MageTraditionalCodecPreparedAsset(StrictModel):
    """One exact file emitted by codec-video-prep for Mage's result loader."""

    relative_path: NonEmptyString
    byte_count: PositiveInt
    sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        relative = PurePosixPath(self.relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("relative_path must be a safe relative POSIX path")
        return self


class MageTraditionalCodecToolchainIdentity(StrictModel):
    """Exact package, executable, command contract, image, and platform identity."""

    toolchain_version: Literal["mage-traditional-codec-toolchain-v1"] = (
        MAGE_TRADITIONAL_CODEC_TOOLCHAIN_VERSION
    )
    package_name: Literal["codec-video-prep"] = "codec-video-prep"
    package_version: NonEmptyString
    package_artifact_sha256: Sha256Digest
    executable_sha256: Sha256Digest
    provider_command_contract_sha256: Sha256Digest
    container_image_reference: NonEmptyString
    container_image_digest: Sha256Digest
    container_platform: NonEmptyString
    toolchain_identity_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if not self.container_image_reference.endswith(f"@sha256:{self.container_image_digest}"):
            raise ValueError("container image reference must be digest pinned")
        if self.toolchain_identity_sha256 != _toolchain_identity_sha256(self):
            raise ValueError("toolchain identity does not match its projection")
        return self


class MageTraditionalCodecEffectiveConfig(StrictModel):
    """Exact request config plus path-independent provider invocation options."""

    effective_config_version: Literal["mage-traditional-codec-effective-config-v1"] = (
        MAGE_TRADITIONAL_CODEC_EFFECTIVE_CONFIG_VERSION
    )
    engine: TraditionalEngine
    native_codec_config: dict[str, JsonValue]
    codec_config_sha256: Sha256Digest
    provider_options: dict[str, JsonValue]
    effective_config_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.native_codec_config.get("engine") != self.engine:
            raise ValueError("native config engine does not match effective engine")
        if "cache_root" in self.native_codec_config:
            raise ValueError("native config must not include operational cache_root")
        if self.codec_config_sha256 != mage_video_codec_config_sha256(self.native_codec_config):
            raise ValueError("codec config identity does not match native config")
        if self.effective_config_sha256 != _effective_config_sha256(self):
            raise ValueError("effective config identity does not match its projection")
        return self


class MageTraditionalCodecCacheEntry(StrictModel):
    """One source-bound set of exact traditional provider outputs."""

    entry_version: Literal["mage-traditional-codec-cache-entry-v1"] = (
        MAGE_TRADITIONAL_CODEC_CACHE_ENTRY_VERSION
    )
    namespace_version: Literal["mage-traditional-codec-cache-namespace-v1"] = (
        MAGE_TRADITIONAL_CODEC_CACHE_NAMESPACE_VERSION
    )
    provider_version: Literal["robata-mage-traditional-codec-provider-v1"] = (
        MAGE_TRADITIONAL_CODEC_PROVIDER_VERSION
    )
    source_path: NonEmptyString
    source_content_sha256: Sha256Digest
    source_byte_count: PositiveInt
    checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    codec_config_sha256: Sha256Digest
    provider_implementation_sha256: Sha256Digest
    toolchain_identity_sha256: Sha256Digest
    effective_config_sha256: Sha256Digest
    provider_identity_sha256: Sha256Digest
    logical_cache_identity: Sha256Digest
    namespace_identity: Sha256Digest
    provider_cache_directory_name: NonEmptyString
    assets: tuple[MageTraditionalCodecPreparedAsset, ...]
    asset_set_sha256: Sha256Digest
    entry_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        if Path(self.provider_cache_directory_name).name != self.provider_cache_directory_name:
            raise ValueError("provider cache directory name must have one component")
        paths = tuple(asset.relative_path for asset in self.assets)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("assets must be nonempty, unique, and sorted")
        if not {"meta.json", "src_patch_position.npy"}.issubset(paths):
            raise ValueError("traditional output lacks required Mage assets")
        if self.asset_set_sha256 != _asset_set_sha256(self.assets):
            raise ValueError("asset set identity does not match assets")
        provider = mage_traditional_codec_provider_identity(
            provider_implementation_sha256=self.provider_implementation_sha256,
            toolchain_identity_sha256=self.toolchain_identity_sha256,
            effective_config_sha256=self.effective_config_sha256,
        )
        if self.provider_identity_sha256 != provider:
            raise ValueError("provider identity does not match provider inputs")
        namespace = mage_traditional_codec_namespace_identity(
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            codec_config_sha256=self.codec_config_sha256,
            provider_identity_sha256=provider,
        )
        if self.namespace_identity != namespace:
            raise ValueError("namespace identity does not match cache inputs")
        logical = mage_traditional_codec_logical_cache_identity(
            source_content_sha256=self.source_content_sha256,
            source_byte_count=self.source_byte_count,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            codec_config_sha256=self.codec_config_sha256,
            provider_identity_sha256=provider,
        )
        if self.logical_cache_identity != logical:
            raise ValueError("logical cache identity does not match source/provider inputs")
        if self.entry_semantic_sha256 != _entry_semantic_sha256(self):
            raise ValueError("entry identity does not match its projection")
        return self


class MageTraditionalCodecCacheManifestEntry(StrictModel):
    """One exact cache directory and its complete replay identity."""

    provider_cache_directory: NonEmptyString
    entry: MageTraditionalCodecCacheEntry


class MageTraditionalCodecCacheManifest(StrictModel):
    """Operational manifest for one exact traditional cache namespace."""

    manifest_version: Literal["mage-traditional-codec-cache-manifest-v1"] = (
        MAGE_TRADITIONAL_CODEC_CACHE_MANIFEST_VERSION
    )
    namespace_version: Literal["mage-traditional-codec-cache-namespace-v1"] = (
        MAGE_TRADITIONAL_CODEC_CACHE_NAMESPACE_VERSION
    )
    provider_version: Literal["robata-mage-traditional-codec-provider-v1"] = (
        MAGE_TRADITIONAL_CODEC_PROVIDER_VERSION
    )
    checkpoint_manifest_sha256: Sha256Digest
    codec_policy_sha256: Sha256Digest
    codec_config_sha256: Sha256Digest
    provider_implementation_sha256: Sha256Digest
    toolchain: MageTraditionalCodecToolchainIdentity
    effective_config: MageTraditionalCodecEffectiveConfig
    provider_identity_sha256: Sha256Digest
    namespace_identity: Sha256Digest
    cache_base_root: NonEmptyString
    qualified_cache_root: NonEmptyString
    entries: tuple[MageTraditionalCodecCacheManifestEntry, ...]
    manifest_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        sources = tuple(item.entry.source_path for item in self.entries)
        if not sources or sources != tuple(sorted(sources)) or len(sources) != len(set(sources)):
            raise ValueError("manifest source paths must be nonempty, unique, and sorted")
        if self.codec_config_sha256 != self.effective_config.codec_config_sha256:
            raise ValueError("manifest codec config differs from effective config")
        provider = mage_traditional_codec_provider_identity(
            provider_implementation_sha256=self.provider_implementation_sha256,
            toolchain_identity_sha256=self.toolchain.toolchain_identity_sha256,
            effective_config_sha256=self.effective_config.effective_config_sha256,
        )
        if self.provider_identity_sha256 != provider:
            raise ValueError("manifest provider identity does not match its inputs")
        namespace = mage_traditional_codec_namespace_identity(
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            codec_policy_sha256=self.codec_policy_sha256,
            codec_config_sha256=self.codec_config_sha256,
            provider_identity_sha256=provider,
        )
        if self.namespace_identity != namespace:
            raise ValueError("manifest namespace identity does not match its inputs")
        if Path(self.qualified_cache_root).name != namespace:
            raise ValueError("qualified cache root must end with namespace identity")
        constants = (
            self.checkpoint_manifest_sha256,
            self.codec_policy_sha256,
            self.codec_config_sha256,
            self.provider_implementation_sha256,
            self.toolchain.toolchain_identity_sha256,
            self.effective_config.effective_config_sha256,
            provider,
            namespace,
        )
        for item in self.entries:
            entry = item.entry
            if (
                entry.checkpoint_manifest_sha256,
                entry.codec_policy_sha256,
                entry.codec_config_sha256,
                entry.provider_implementation_sha256,
                entry.toolchain_identity_sha256,
                entry.effective_config_sha256,
                entry.provider_identity_sha256,
                entry.namespace_identity,
            ) != constants:
                raise ValueError("cache entry identity diverges from manifest")
            if Path(item.provider_cache_directory).name != entry.provider_cache_directory_name:
                raise ValueError("entry provider directory locator changed")
        if self.manifest_semantic_sha256 != _manifest_semantic_sha256(self):
            raise ValueError("manifest identity does not match its projection")
        return self


def build_mage_traditional_codec_toolchain_identity(
    *,
    package_version: str,
    package_artifact_sha256: Sha256Digest,
    executable_sha256: Sha256Digest,
    provider_command_contract_sha256: Sha256Digest,
    container_image_reference: str,
    container_image_digest: Sha256Digest,
    container_platform: str,
) -> MageTraditionalCodecToolchainIdentity:
    """Build a self-validating codec-video-prep/container identity."""

    values: dict[str, Any] = {
        "package_version": package_version,
        "package_artifact_sha256": package_artifact_sha256,
        "executable_sha256": executable_sha256,
        "provider_command_contract_sha256": provider_command_contract_sha256,
        "container_image_reference": container_image_reference,
        "container_image_digest": container_image_digest,
        "container_platform": container_platform,
    }
    provisional = MageTraditionalCodecToolchainIdentity.model_construct(
        **values, toolchain_identity_sha256="0" * 64
    )
    return MageTraditionalCodecToolchainIdentity(
        **values,
        toolchain_identity_sha256=_toolchain_identity_sha256(provisional),
    )


def build_mage_traditional_codec_effective_config(
    *,
    codec_policy: MageVideoCodecPolicy,
    engine: TraditionalEngine = "hevc",
    provider_options: Mapping[str, JsonValue] | None = None,
) -> MageTraditionalCodecEffectiveConfig:
    """Bind endpoint policy and all path-independent cv-preinfer options."""

    if codec_policy.codec_mode != "traditional":
        raise MageTraditionalCodecCacheError(
            "traditional effective config requires codec_mode='traditional'"
        )
    native_config = dict(codec_policy.native_codec_config())
    if native_config.get("engine") != engine:
        raise MageTraditionalCodecCacheError(
            "effective engine must exactly match endpoint native config"
        )
    values: dict[str, Any] = {
        "engine": engine,
        "native_codec_config": native_config,
        "codec_config_sha256": mage_video_codec_config_sha256(native_config),
        "provider_options": dict(provider_options or {}),
    }
    provisional = MageTraditionalCodecEffectiveConfig.model_construct(
        **values, effective_config_sha256="0" * 64
    )
    return MageTraditionalCodecEffectiveConfig(
        **values,
        effective_config_sha256=_effective_config_sha256(provisional),
    )


def build_mage_traditional_codec_cache_manifest(
    *,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy: MageVideoCodecPolicy,
    provider_implementation_sha256: Sha256Digest,
    toolchain: MageTraditionalCodecToolchainIdentity,
    effective_config: MageTraditionalCodecEffectiveConfig,
    cache_base_root: Path,
    observations: Sequence[tuple[Path, Path]],
) -> MageTraditionalCodecCacheManifest:
    """Build a manifest from already materialized traditional outputs."""

    if codec_policy.codec_mode != "traditional":
        raise MageTraditionalCodecCacheError("traditional cache requires codec_mode='traditional'")
    policy_sha256 = build_mage_video_codec_policy_identity(codec_policy).policy_sha256
    codec_config_sha256 = mage_video_codec_config_sha256(codec_policy.native_codec_config())
    if effective_config.codec_config_sha256 != codec_config_sha256:
        raise MageTraditionalCodecCacheError(
            "effective config does not match endpoint codec policy"
        )
    provider_identity_sha256 = mage_traditional_codec_provider_identity(
        provider_implementation_sha256=provider_implementation_sha256,
        toolchain_identity_sha256=toolchain.toolchain_identity_sha256,
        effective_config_sha256=effective_config.effective_config_sha256,
    )
    namespace_identity = mage_traditional_codec_namespace_identity(
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        codec_policy_sha256=policy_sha256,
        codec_config_sha256=codec_config_sha256,
        provider_identity_sha256=provider_identity_sha256,
    )
    base_root = Path(cache_base_root).expanduser().resolve()
    qualified_root = base_root / namespace_identity
    if qualified_root.is_symlink() or not qualified_root.is_dir():
        raise MageTraditionalCodecCacheError(
            "qualified traditional cache root is missing or unsafe"
        )
    normalized = tuple(
        sorted(
            (
                (Path(source).expanduser().resolve(), Path(directory).expanduser().resolve())
                for source, directory in observations
            ),
            key=lambda pair: str(pair[0]),
        )
    )
    if not normalized or len({source for source, _ in normalized}) != len(normalized):
        raise MageTraditionalCodecCacheError("observations must have unique source paths")

    entries: list[MageTraditionalCodecCacheManifestEntry] = []
    for source, directory in normalized:
        if directory.is_symlink() or directory.parent != qualified_root:
            raise MageTraditionalCodecCacheError("provider output is outside its exact namespace")
        source_sha256, source_byte_count = _exact_file_sha256(source)
        assets = _collect_assets(directory)
        asset_set_sha256 = _asset_set_sha256(assets)
        logical = mage_traditional_codec_logical_cache_identity(
            source_content_sha256=source_sha256,
            source_byte_count=source_byte_count,
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            codec_policy_sha256=policy_sha256,
            codec_config_sha256=codec_config_sha256,
            provider_identity_sha256=provider_identity_sha256,
        )
        values: dict[str, Any] = {
            "source_path": str(source),
            "source_content_sha256": source_sha256,
            "source_byte_count": source_byte_count,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": policy_sha256,
            "codec_config_sha256": codec_config_sha256,
            "provider_implementation_sha256": provider_implementation_sha256,
            "toolchain_identity_sha256": toolchain.toolchain_identity_sha256,
            "effective_config_sha256": effective_config.effective_config_sha256,
            "provider_identity_sha256": provider_identity_sha256,
            "logical_cache_identity": logical,
            "namespace_identity": namespace_identity,
            "provider_cache_directory_name": directory.name,
            "assets": assets,
            "asset_set_sha256": asset_set_sha256,
        }
        provisional = MageTraditionalCodecCacheEntry.model_construct(
            **values, entry_semantic_sha256="0" * 64
        )
        entry = MageTraditionalCodecCacheEntry(
            **values, entry_semantic_sha256=_entry_semantic_sha256(provisional)
        )
        entries.append(
            MageTraditionalCodecCacheManifestEntry(
                provider_cache_directory=str(directory), entry=entry
            )
        )

    manifest_values: dict[str, Any] = {
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "codec_policy_sha256": policy_sha256,
        "codec_config_sha256": codec_config_sha256,
        "provider_implementation_sha256": provider_implementation_sha256,
        "toolchain": toolchain,
        "effective_config": effective_config,
        "provider_identity_sha256": provider_identity_sha256,
        "namespace_identity": namespace_identity,
        "cache_base_root": str(base_root),
        "qualified_cache_root": str(qualified_root),
        "entries": tuple(entries),
    }
    provisional_manifest = MageTraditionalCodecCacheManifest.model_construct(
        **manifest_values, manifest_semantic_sha256="0" * 64
    )
    return MageTraditionalCodecCacheManifest(
        **manifest_values,
        manifest_semantic_sha256=_manifest_semantic_sha256(provisional_manifest),
    )


def mage_traditional_codec_provider_identity(
    *,
    provider_implementation_sha256: Sha256Digest,
    toolchain_identity_sha256: Sha256Digest,
    effective_config_sha256: Sha256Digest,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "provider_version": MAGE_TRADITIONAL_CODEC_PROVIDER_VERSION,
            "provider_implementation_sha256": provider_implementation_sha256,
            "toolchain_identity_sha256": toolchain_identity_sha256,
            "effective_config_sha256": effective_config_sha256,
        }
    )


def mage_traditional_codec_namespace_identity(
    *,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    codec_config_sha256: Sha256Digest,
    provider_identity_sha256: Sha256Digest,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "namespace_version": MAGE_TRADITIONAL_CODEC_CACHE_NAMESPACE_VERSION,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": codec_policy_sha256,
            "codec_config_sha256": codec_config_sha256,
            "provider_identity_sha256": provider_identity_sha256,
        }
    )


def mage_traditional_codec_logical_cache_identity(
    *,
    source_content_sha256: Sha256Digest,
    source_byte_count: int,
    checkpoint_manifest_sha256: Sha256Digest,
    codec_policy_sha256: Sha256Digest,
    codec_config_sha256: Sha256Digest,
    provider_identity_sha256: Sha256Digest,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "entry_version": MAGE_TRADITIONAL_CODEC_CACHE_ENTRY_VERSION,
            "source_content_sha256": source_content_sha256,
            "source_byte_count": source_byte_count,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "codec_policy_sha256": codec_policy_sha256,
            "codec_config_sha256": codec_config_sha256,
            "provider_identity_sha256": provider_identity_sha256,
        }
    )


def verify_mage_traditional_codec_cache_manifest(
    *,
    manifest: MageTraditionalCodecCacheManifest,
    expected_checkpoint_manifest_sha256: Sha256Digest | None = None,
    expected_codec_policy_sha256: Sha256Digest | None = None,
    expected_provider_identity_sha256: Sha256Digest | None = None,
    expected_toolchain_identity_sha256: Sha256Digest | None = None,
    expected_container_image_digest: Sha256Digest | None = None,
) -> tuple[MageTraditionalCodecCacheEntry, ...]:
    """Re-hash every source and provider output under an exact namespace."""

    if not isinstance(manifest, MageTraditionalCodecCacheManifest):
        raise TypeError("manifest must be MageTraditionalCodecCacheManifest")
    expected_pairs = (
        (
            "checkpoint manifest",
            manifest.checkpoint_manifest_sha256,
            expected_checkpoint_manifest_sha256,
        ),
        ("codec policy", manifest.codec_policy_sha256, expected_codec_policy_sha256),
        ("provider", manifest.provider_identity_sha256, expected_provider_identity_sha256),
        (
            "toolchain",
            manifest.toolchain.toolchain_identity_sha256,
            expected_toolchain_identity_sha256,
        ),
        (
            "container image",
            manifest.toolchain.container_image_digest,
            expected_container_image_digest,
        ),
    )
    for label, actual, expected in expected_pairs:
        if expected is not None and actual != expected:
            raise MageTraditionalCodecCacheError(f"traditional codec {label} identity mismatch")

    qualified_root = Path(manifest.qualified_cache_root).expanduser()
    if qualified_root.is_symlink():
        raise MageTraditionalCodecCacheError("qualified cache root must not be a symlink")
    qualified_root = qualified_root.resolve()
    if not qualified_root.is_dir():
        raise MageTraditionalCodecCacheError("qualified cache root is missing")
    if qualified_root.parent != Path(manifest.cache_base_root).expanduser().resolve():
        raise MageTraditionalCodecCacheError("cache root binding changed")
    if qualified_root.name != manifest.namespace_identity:
        raise MageTraditionalCodecCacheError("namespace locator changed")

    expected_directories = {
        Path(item.provider_cache_directory).expanduser().resolve().name for item in manifest.entries
    }
    observed_directories = {
        child.name
        for child in qualified_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    }
    if observed_directories != expected_directories:
        raise MageTraditionalCodecCacheError("namespace contains unbound provider directories")

    verified: list[MageTraditionalCodecCacheEntry] = []
    for item in manifest.entries:
        verified.append(
            verify_mage_traditional_codec_cache_entry(
                item=item, qualified_cache_root=qualified_root
            )
        )
    return tuple(verified)


def verify_mage_traditional_codec_cache_entry(
    *,
    item: MageTraditionalCodecCacheManifestEntry,
    qualified_cache_root: Path,
) -> MageTraditionalCodecCacheEntry:
    """Verify one source and exact asset set without invoking external tools."""

    if not isinstance(item, MageTraditionalCodecCacheManifestEntry):
        raise TypeError("item must be MageTraditionalCodecCacheManifestEntry")
    root = Path(qualified_cache_root).expanduser().resolve()
    directory = Path(item.provider_cache_directory).expanduser()
    if directory.is_symlink():
        raise MageTraditionalCodecCacheError("provider directory must not be a symlink")
    directory = directory.resolve()
    if directory.parent != root or directory.name != item.entry.provider_cache_directory_name:
        raise MageTraditionalCodecCacheError("provider directory escaped namespace")
    if not directory.is_dir():
        raise MageTraditionalCodecCacheError("provider directory is missing")

    source = Path(item.entry.source_path).expanduser()
    if source.is_symlink():
        raise MageTraditionalCodecCacheError("source must not be a symlink")
    source = source.resolve()
    source_sha256, source_byte_count = _exact_file_sha256(source)
    if (
        source_sha256 != item.entry.source_content_sha256
        or source_byte_count != item.entry.source_byte_count
    ):
        raise MageTraditionalCodecCacheError("source bytes changed")
    assets = _collect_assets(directory)
    if assets != item.entry.assets or _asset_set_sha256(assets) != item.entry.asset_set_sha256:
        raise MageTraditionalCodecCacheError("provider assets changed")
    return item.entry


class MageTraditionalCodecCacheAdmission:
    """Endpoint guard converting a deployment-pinned manifest to runtime bindings."""

    def __init__(
        self,
        *,
        manifest: MageTraditionalCodecCacheManifest,
        expected_checkpoint_manifest_sha256: Sha256Digest,
        expected_codec_policy: MageVideoCodecPolicy,
        expected_provider_identity_sha256: Sha256Digest,
        expected_toolchain_identity_sha256: Sha256Digest,
        expected_container_image_digest: Sha256Digest,
    ) -> None:
        if expected_codec_policy.codec_mode != "traditional":
            raise MageTraditionalCodecCacheError(
                "traditional admission requires codec_mode='traditional'"
            )
        policy_sha256 = build_mage_video_codec_policy_identity(expected_codec_policy).policy_sha256
        codec_config_sha256 = mage_video_codec_config_sha256(
            expected_codec_policy.native_codec_config()
        )
        if manifest.codec_config_sha256 != codec_config_sha256:
            raise MageTraditionalCodecCacheError("cache config does not match endpoint policy")
        verified = verify_mage_traditional_codec_cache_manifest(
            manifest=manifest,
            expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
            expected_codec_policy_sha256=policy_sha256,
            expected_provider_identity_sha256=expected_provider_identity_sha256,
            expected_toolchain_identity_sha256=expected_toolchain_identity_sha256,
            expected_container_image_digest=expected_container_image_digest,
        )
        self._manifest = manifest
        self._expected_checkpoint_manifest_sha256 = expected_checkpoint_manifest_sha256
        self._expected_codec_policy_sha256 = policy_sha256
        self._entries = {
            entry.source_path: item for entry, item in zip(verified, manifest.entries, strict=True)
        }

    @classmethod
    def from_manifest_path(
        cls,
        *,
        manifest_path: Path,
        expected_checkpoint_manifest_sha256: Sha256Digest,
        expected_codec_policy: MageVideoCodecPolicy,
        expected_provider_identity_sha256: Sha256Digest,
        expected_toolchain_identity_sha256: Sha256Digest,
        expected_container_image_digest: Sha256Digest,
    ) -> MageTraditionalCodecCacheAdmission:
        return cls(
            manifest=load_mage_traditional_codec_cache_manifest(path=manifest_path),
            expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
            expected_codec_policy=expected_codec_policy,
            expected_provider_identity_sha256=expected_provider_identity_sha256,
            expected_toolchain_identity_sha256=expected_toolchain_identity_sha256,
            expected_container_image_digest=expected_container_image_digest,
        )

    @property
    def cache_root(self) -> Path:
        """Exact namespace root passed to MageVideoRuntime."""

        return Path(self._manifest.qualified_cache_root).expanduser().resolve()

    @property
    def manifest(self) -> MageTraditionalCodecCacheManifest:
        return self._manifest

    def __call__(
        self,
        request: MageVideoEndpointRequest,
        paths: Sequence[Path],
        /,
    ) -> MageVideoTraditionalCodecCacheBinding:
        if request.codec_policy.codec_mode != "traditional":
            raise MageTraditionalCodecCacheError("traditional cache cannot admit a neural request")
        if len(paths) != 1:
            raise MageTraditionalCodecCacheError("traditional cache admits exactly one source path")
        source = Path(paths[0]).expanduser().resolve()
        item = self._entries.get(str(source))
        if item is None:
            raise MageTraditionalCodecCacheError("request source is absent from traditional cache")
        entry = item.entry
        segment = request.camera_encodings[0].segment_manifest
        request_policy_sha256 = build_mage_video_codec_policy_identity(
            request.codec_policy
        ).policy_sha256
        request_config_sha256 = mage_video_codec_config_sha256(
            request.codec_policy.native_codec_config()
        )
        if request.model_identity.checkpoint_manifest_sha256 != (
            self._expected_checkpoint_manifest_sha256
        ):
            raise MageTraditionalCodecCacheError(
                "request checkpoint does not match traditional cache"
            )
        if request_policy_sha256 != self._expected_codec_policy_sha256:
            raise MageTraditionalCodecCacheError("request policy does not match traditional cache")
        if request_config_sha256 != entry.codec_config_sha256:
            raise MageTraditionalCodecCacheError(
                "request codec config does not match traditional cache"
            )
        if (
            segment.content_sha256 != entry.source_content_sha256
            or segment.byte_count != entry.source_byte_count
            or Path(segment.durable_path).expanduser().resolve() != source
        ):
            raise MageTraditionalCodecCacheError(
                "request segment does not match traditional cache source"
            )
        assets = tuple(
            MageVideoExactCodecCacheAsset(
                relative_path=asset.relative_path,
                byte_count=asset.byte_count,
                sha256=asset.sha256,
            )
            for asset in entry.assets
        )
        return MageVideoTraditionalCodecCacheBinding(
            source_path=source,
            provider_cache_directory=Path(item.provider_cache_directory),
            codec_engine=self._manifest.effective_config.engine,
            codec_config_sha256=entry.codec_config_sha256,
            checkpoint_manifest_sha256=entry.checkpoint_manifest_sha256,
            codec_policy_sha256=entry.codec_policy_sha256,
            provider_identity_sha256=entry.provider_identity_sha256,
            toolchain_identity_sha256=entry.toolchain_identity_sha256,
            effective_config_sha256=entry.effective_config_sha256,
            entry_semantic_sha256=entry.entry_semantic_sha256,
            asset_set_sha256=entry.asset_set_sha256,
            assets=assets,
        )


def write_mage_traditional_codec_cache_manifest(
    *, manifest: MageTraditionalCodecCacheManifest, path: Path
) -> None:
    """Create one canonical manifest without ever replacing an existing identity."""

    if not isinstance(manifest, MageTraditionalCodecCacheManifest):
        raise TypeError("manifest must be MageTraditionalCodecCacheManifest")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))

    if os.path.lexists(destination):
        _verify_exact_existing_manifest(
            destination=destination,
            manifest=manifest,
            expected_bytes=expected_bytes,
        )
        return

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(expected_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # The same-directory hard link publishes a fully written inode atomically.
            # Unlike replace(), it cannot overwrite an identity won by another writer.
            os.link(temporary, destination)
        except FileExistsError:
            pass
        except OSError as error:
            raise MageTraditionalCodecCacheError(
                "could not atomically publish traditional codec cache manifest"
            ) from error
    except OSError as error:
        raise MageTraditionalCodecCacheError(
            "could not stage traditional codec cache manifest"
        ) from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            raise MageTraditionalCodecCacheError(
                "could not remove staged traditional codec cache manifest"
            ) from error

    _verify_exact_existing_manifest(
        destination=destination,
        manifest=manifest,
        expected_bytes=expected_bytes,
    )


def _verify_exact_existing_manifest(
    *,
    destination: Path,
    manifest: MageTraditionalCodecCacheManifest,
    expected_bytes: bytes,
) -> None:
    if destination.is_symlink() or not destination.is_file():
        raise MageTraditionalCodecCacheError("existing traditional codec cache manifest is unsafe")
    try:
        actual_bytes = destination.read_bytes()
        actual_manifest = MageTraditionalCodecCacheManifest.model_validate_json(
            actual_bytes, strict=True
        )
    except (OSError, TypeError, ValueError) as error:
        raise MageTraditionalCodecCacheError(
            "existing traditional codec cache manifest is invalid"
        ) from error
    if actual_bytes != expected_bytes or actual_manifest != manifest:
        raise MageTraditionalCodecCacheError(
            "existing traditional codec cache manifest differs from requested identity"
        )


def load_mage_traditional_codec_cache_manifest(*, path: Path) -> MageTraditionalCodecCacheManifest:
    """Load canonical self-validating internal manifest bytes."""

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        manifest = MageTraditionalCodecCacheManifest.model_validate_json(raw, strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise MageTraditionalCodecCacheError(
            f"could not load traditional codec cache manifest: {source}"
        ) from error
    if canonical_json_bytes(manifest.model_dump(mode="json")) != raw:
        raise MageTraditionalCodecCacheError(
            "traditional codec cache manifest is not canonical JSON"
        )
    return manifest


def _toolchain_identity_sha256(
    toolchain: MageTraditionalCodecToolchainIdentity,
) -> Sha256Digest:
    return semantic_sha256(toolchain.model_dump(mode="json", exclude={"toolchain_identity_sha256"}))


def _effective_config_sha256(
    config: MageTraditionalCodecEffectiveConfig,
) -> Sha256Digest:
    return semantic_sha256(config.model_dump(mode="json", exclude={"effective_config_sha256"}))


def _asset_set_sha256(
    assets: tuple[MageTraditionalCodecPreparedAsset, ...],
) -> Sha256Digest:
    return semantic_sha256([asset.model_dump(mode="json") for asset in assets])


def _entry_semantic_sha256(entry: MageTraditionalCodecCacheEntry) -> Sha256Digest:
    return semantic_sha256(entry.model_dump(mode="json", exclude={"entry_semantic_sha256"}))


def _manifest_semantic_sha256(
    manifest: MageTraditionalCodecCacheManifest,
) -> Sha256Digest:
    return semantic_sha256(manifest.model_dump(mode="json", exclude={"manifest_semantic_sha256"}))


def _collect_assets(directory: Path) -> tuple[MageTraditionalCodecPreparedAsset, ...]:
    root = Path(directory).expanduser()
    if root.is_symlink():
        raise MageTraditionalCodecCacheError("provider cache directory must not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise MageTraditionalCodecCacheError("provider cache directory is missing")
    assets: list[MageTraditionalCodecPreparedAsset] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise MageTraditionalCodecCacheError("provider cache assets must not be symlinks")
        if path.is_file():
            sha256, byte_count = _exact_file_sha256(path)
            assets.append(
                MageTraditionalCodecPreparedAsset(
                    relative_path=path.relative_to(root).as_posix(),
                    byte_count=byte_count,
                    sha256=sha256,
                )
            )
    ordered = tuple(assets)
    paths = {asset.relative_path for asset in ordered}
    if not {"meta.json", "src_patch_position.npy"}.issubset(paths):
        raise MageTraditionalCodecCacheError("traditional output lacks required Mage assets")
    return ordered


def _exact_file_sha256(path: Path) -> tuple[Sha256Digest, int]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise MageTraditionalCodecCacheError(f"exact file must not be a symlink: {source}")
    source = source.resolve()
    if not source.is_file():
        raise MageTraditionalCodecCacheError(f"exact file is missing: {source}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageTraditionalCodecCacheError(f"could not read exact file: {source}") from error
    if byte_count <= 0:
        raise MageTraditionalCodecCacheError(f"exact file is empty: {source}")
    return digest.hexdigest(), byte_count


__all__ = [
    "MAGE_TRADITIONAL_CODEC_CACHE_ENTRY_VERSION",
    "MAGE_TRADITIONAL_CODEC_CACHE_MANIFEST_VERSION",
    "MAGE_TRADITIONAL_CODEC_CACHE_NAMESPACE_VERSION",
    "MAGE_TRADITIONAL_CODEC_EFFECTIVE_CONFIG_VERSION",
    "MAGE_TRADITIONAL_CODEC_PROVIDER_VERSION",
    "MAGE_TRADITIONAL_CODEC_TOOLCHAIN_VERSION",
    "MageTraditionalCodecCacheAdmission",
    "MageTraditionalCodecCacheEntry",
    "MageTraditionalCodecCacheError",
    "MageTraditionalCodecCacheManifest",
    "MageTraditionalCodecCacheManifestEntry",
    "MageTraditionalCodecEffectiveConfig",
    "MageTraditionalCodecPreparedAsset",
    "MageTraditionalCodecToolchainIdentity",
    "build_mage_traditional_codec_cache_manifest",
    "build_mage_traditional_codec_effective_config",
    "build_mage_traditional_codec_toolchain_identity",
    "load_mage_traditional_codec_cache_manifest",
    "mage_traditional_codec_logical_cache_identity",
    "mage_traditional_codec_namespace_identity",
    "mage_traditional_codec_provider_identity",
    "verify_mage_traditional_codec_cache_entry",
    "verify_mage_traditional_codec_cache_manifest",
    "write_mage_traditional_codec_cache_manifest",
]
