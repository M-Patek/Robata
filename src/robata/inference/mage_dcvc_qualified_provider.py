"""Reproducible qualified Mage model trees for the explicit DCVC Provider V2.

The original checkpoint is never edited.  A qualified tree contains exact copies (or
operator-selected hard links) of the upstream files plus the exact Provider V2 Python
implementation and a canonical bundle manifest.  The ordinary Mage checkpoint manifest
therefore binds provider code into the existing model/inference identity.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.inference.mage_checkpoint_identity import (
    MageCheckpointManifest,
    build_mage_checkpoint_manifest,
    verify_mage_checkpoint_manifest,
    write_mage_checkpoint_manifest,
)
from robata.inference.mage_dcvc_preparation_protocol import MAGE_DCVC_PROVIDER_VERSION

MAGE_DCVC_QUALIFIED_BUNDLE_VERSION: Final = "mage-dcvc-qualified-provider-bundle-v2"
MAGE_DCVC_QUALIFIED_MANIFEST_VERSION: Final = "mage-dcvc-qualified-provider-manifest-v2"
MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY: Final = "robata_provider_v2"
MAGE_DCVC_QUALIFIED_BUNDLE_MANIFEST_NAME: Final = "provider_bundle_manifest.json"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class MageDcvcQualifiedProviderError(RuntimeError):
    """A qualified provider tree could not be created or verified safely."""


class MageDcvcQualifiedProviderFile(StrictModel):
    """One exact Provider V2 implementation file copied into the qualified tree."""

    relative_path: NonEmptyString
    byte_count: PositiveInt
    sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("relative_path must be a safe relative POSIX path")
        return self


class MageDcvcQualifiedProviderBundle(StrictModel):
    """Canonical provider payload embedded in the qualified Mage checkpoint tree."""

    bundle_version: Literal["mage-dcvc-qualified-provider-bundle-v2"] = (
        MAGE_DCVC_QUALIFIED_BUNDLE_VERSION
    )
    provider_version: Literal["robata-mage-dcvc-provider-v2"] = MAGE_DCVC_PROVIDER_VERSION
    source_checkpoint_manifest_sha256: Sha256Digest
    source_model_identifier: NonEmptyString
    source_model_revision: NonEmptyString
    qualified_model_identifier: NonEmptyString
    qualified_model_revision: NonEmptyString
    provider_files: tuple[MageDcvcQualifiedProviderFile, ...]
    bundle_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        paths = tuple(item.relative_path for item in self.provider_files)
        if not paths:
            raise ValueError("provider_files must not be empty")
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("provider_files must be unique and sorted")
        if self.qualified_model_revision == self.source_model_revision:
            raise ValueError("qualified_model_revision must differ from source_model_revision")
        if self.bundle_semantic_sha256 != _qualified_bundle_sha256(self):
            raise ValueError("bundle_semantic_sha256 does not match bundle projection")
        return self


class MageDcvcQualifiedProviderManifest(StrictModel):
    """Operational record tying a qualified tree to its new checkpoint identity."""

    manifest_version: Literal["mage-dcvc-qualified-provider-manifest-v2"] = (
        MAGE_DCVC_QUALIFIED_MANIFEST_VERSION
    )
    provider_version: Literal["robata-mage-dcvc-provider-v2"] = MAGE_DCVC_PROVIDER_VERSION
    copy_mode: Literal["copy", "hardlink"]
    qualified_model_directory: NonEmptyString
    bundle: MageDcvcQualifiedProviderBundle
    qualified_checkpoint_manifest: MageCheckpointManifest
    manifest_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        checkpoint = self.qualified_checkpoint_manifest
        if checkpoint.model_identifier != self.bundle.qualified_model_identifier:
            raise ValueError("qualified checkpoint model identifier does not match bundle")
        if checkpoint.model_revision != self.bundle.qualified_model_revision:
            raise ValueError("qualified checkpoint model revision does not match bundle")
        if self.manifest_semantic_sha256 != _qualified_manifest_sha256(self):
            raise ValueError("manifest_semantic_sha256 does not match manifest projection")
        return self


def qualify_mage_dcvc_provider_v2(
    *,
    source_model_directory: Path,
    source_checkpoint_manifest: MageCheckpointManifest,
    target_model_directory: Path,
    qualified_model_identifier: str,
    qualified_model_revision: str,
    provider_source_files: Sequence[Path],
    manifest_path: Path,
    copy_mode: Literal["copy", "hardlink"] = "copy",
) -> MageDcvcQualifiedProviderManifest:
    """Create and atomically publish a Provider V2-qualified Mage model tree.

    ``provider_source_files`` are copied under
    ``neural_codec/robata_provider_v2`` and then included by the existing Mage
    checkpoint inclusion policy.  An already-published target is never overwritten.
    """

    if not isinstance(source_checkpoint_manifest, MageCheckpointManifest):
        raise TypeError("source_checkpoint_manifest must be MageCheckpointManifest")
    source = Path(source_model_directory).expanduser().resolve()
    target = Path(target_model_directory).expanduser().resolve()
    manifest_target = Path(manifest_path).expanduser().resolve()
    _validate_distinct_trees(source=source, target=target)
    verify_mage_checkpoint_manifest(
        manifest=source_checkpoint_manifest,
        model_directory=source,
    )
    sources = _normalise_provider_sources(provider_source_files)
    if target.exists():
        raise MageDcvcQualifiedProviderError(
            f"qualified target already exists; verify/reuse or choose a new target: {target}"
        )
    if manifest_target.exists():
        raise MageDcvcQualifiedProviderError(
            "qualified manifest already exists; verify/reuse or choose a new path: "
            f"{manifest_target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.robata-dcvc-v2-{uuid.uuid4().hex}.tmp"
    _require_safe_staging(staging=staging, target=target)
    try:
        _copy_model_tree(source=source, staging=staging, copy_mode=copy_mode)
        provider_directory = staging / "neural_codec" / MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY
        provider_directory.mkdir(parents=True, exist_ok=False)
        provider_files: list[MageDcvcQualifiedProviderFile] = []
        for provider_source in sources:
            destination = provider_directory / provider_source.name
            shutil.copy2(provider_source, destination)
            digest, byte_count = _file_sha256(destination)
            provider_files.append(
                MageDcvcQualifiedProviderFile(
                    relative_path=(
                        Path("neural_codec")
                        / MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY
                        / destination.name
                    ).as_posix(),
                    byte_count=byte_count,
                    sha256=digest,
                )
            )
        ordered_files = tuple(sorted(provider_files, key=lambda item: item.relative_path))
        provisional_bundle = MageDcvcQualifiedProviderBundle.model_construct(
            source_checkpoint_manifest_sha256=source_checkpoint_manifest.manifest_sha256,
            source_model_identifier=source_checkpoint_manifest.model_identifier,
            source_model_revision=source_checkpoint_manifest.model_revision,
            qualified_model_identifier=qualified_model_identifier,
            qualified_model_revision=qualified_model_revision,
            provider_files=ordered_files,
            bundle_semantic_sha256="0" * 64,
        )
        bundle = MageDcvcQualifiedProviderBundle(
            source_checkpoint_manifest_sha256=source_checkpoint_manifest.manifest_sha256,
            source_model_identifier=source_checkpoint_manifest.model_identifier,
            source_model_revision=source_checkpoint_manifest.model_revision,
            qualified_model_identifier=qualified_model_identifier,
            qualified_model_revision=qualified_model_revision,
            provider_files=ordered_files,
            bundle_semantic_sha256=_qualified_bundle_sha256(provisional_bundle),
        )
        (provider_directory / MAGE_DCVC_QUALIFIED_BUNDLE_MANIFEST_NAME).write_bytes(
            canonical_json_bytes(bundle.model_dump(mode="json"))
        )
        qualified_checkpoint = build_mage_checkpoint_manifest(
            model_directory=staging,
            model_identifier=qualified_model_identifier,
            model_revision=qualified_model_revision,
        )
        verify_mage_checkpoint_manifest(
            manifest=qualified_checkpoint,
            model_directory=staging,
        )
        staging.replace(target)
        provisional = MageDcvcQualifiedProviderManifest.model_construct(
            copy_mode=copy_mode,
            qualified_model_directory=str(target),
            bundle=bundle,
            qualified_checkpoint_manifest=qualified_checkpoint,
            manifest_semantic_sha256="0" * 64,
        )
        manifest = MageDcvcQualifiedProviderManifest(
            copy_mode=copy_mode,
            qualified_model_directory=str(target),
            bundle=bundle,
            qualified_checkpoint_manifest=qualified_checkpoint,
            manifest_semantic_sha256=_qualified_manifest_sha256(provisional),
        )
        _write_canonical_manifest(manifest=manifest, path=manifest_target)
        return manifest
    except Exception:
        if staging.exists():
            _require_safe_staging(staging=staging, target=target)
            shutil.rmtree(staging)
        raise


def load_mage_dcvc_qualified_provider_manifest(
    *, manifest_path: Path
) -> MageDcvcQualifiedProviderManifest:
    """Load canonical qualification evidence without trusting embedded paths."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        manifest = MageDcvcQualifiedProviderManifest.model_validate_json(raw, strict=True)
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise MageDcvcQualifiedProviderError("qualified provider manifest is invalid") from error
    if canonical_json_bytes(manifest.model_dump(mode="json")) != raw:
        raise MageDcvcQualifiedProviderError("qualified provider manifest must use canonical JSON")
    return manifest


def verify_mage_dcvc_qualified_provider(*, manifest: MageDcvcQualifiedProviderManifest) -> None:
    """Rehash checkpoint and embedded provider files before the tree is used."""

    if not isinstance(manifest, MageDcvcQualifiedProviderManifest):
        raise TypeError("manifest must be MageDcvcQualifiedProviderManifest")
    root = Path(manifest.qualified_model_directory).expanduser().resolve()
    verify_mage_checkpoint_manifest(
        manifest=manifest.qualified_checkpoint_manifest,
        model_directory=root,
    )
    provider_root = (root / "neural_codec" / MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY).resolve()
    if provider_root.parent != (root / "neural_codec").resolve():
        raise MageDcvcQualifiedProviderError("qualified provider directory escaped model root")
    bundle_path = provider_root / MAGE_DCVC_QUALIFIED_BUNDLE_MANIFEST_NAME
    try:
        bundle_raw = bundle_path.read_bytes()
        embedded = MageDcvcQualifiedProviderBundle.model_validate_json(bundle_raw, strict=True)
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise MageDcvcQualifiedProviderError("embedded provider bundle is invalid") from error
    if canonical_json_bytes(embedded.model_dump(mode="json")) != bundle_raw:
        raise MageDcvcQualifiedProviderError("embedded provider bundle must use canonical JSON")
    if embedded != manifest.bundle:
        raise MageDcvcQualifiedProviderError("embedded provider bundle does not match manifest")
    for provider_file in embedded.provider_files:
        path = (root / Path(PurePosixPath(provider_file.relative_path))).resolve()
        if provider_root not in path.parents:
            raise MageDcvcQualifiedProviderError("provider implementation path escaped bundle root")
        digest, byte_count = _file_sha256(path)
        if digest != provider_file.sha256 or byte_count != provider_file.byte_count:
            raise MageDcvcQualifiedProviderError(
                f"provider implementation bytes changed: {provider_file.relative_path}"
            )


def verify_mage_dcvc_qualified_provider_sources(
    *,
    manifest: MageDcvcQualifiedProviderManifest,
    provider_source_files: Sequence[Path],
) -> None:
    """Prove the executing provider sources are the bytes embedded in the qualified tree.

    The qualified checkpoint identity protects the copied provider bundle.  Runtime
    processes still import Robata's installed modules, so admission must additionally
    bind those executing source files to the exact embedded bytes rather than trusting
    matching filenames or a previously recorded digest.
    """

    verify_mage_dcvc_qualified_provider(manifest=manifest)
    sources = _normalise_provider_sources(provider_source_files)
    expected = {
        PurePosixPath(item.relative_path).name: item for item in manifest.bundle.provider_files
    }
    actual_names = {path.name for path in sources}
    if actual_names != set(expected):
        raise MageDcvcQualifiedProviderError(
            "executing provider source set does not match qualified provider bundle"
        )
    for source in sources:
        digest, byte_count = _file_sha256(source)
        provider_file = expected[source.name]
        if digest != provider_file.sha256 or byte_count != provider_file.byte_count:
            raise MageDcvcQualifiedProviderError(
                f"executing provider source bytes differ from qualified bundle: {source.name}"
            )


def write_qualified_checkpoint_manifest(
    *, manifest: MageDcvcQualifiedProviderManifest, path: Path
) -> None:
    """Write the ordinary checkpoint manifest for endpoint launch tooling."""

    write_mage_checkpoint_manifest(
        manifest=manifest.qualified_checkpoint_manifest,
        manifest_path=path,
    )


def _normalise_provider_sources(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved = tuple(sorted({Path(path).expanduser().resolve() for path in paths}, key=str))
    if not resolved:
        raise MageDcvcQualifiedProviderError("at least one provider source file is required")
    if len({path.name.casefold() for path in resolved}) != len(resolved):
        raise MageDcvcQualifiedProviderError("provider source basenames must be unique")
    for path in resolved:
        if not path.is_file():
            raise MageDcvcQualifiedProviderError(f"provider source file is missing: {path}")
        if path.is_symlink():
            raise MageDcvcQualifiedProviderError("provider source files cannot be symlinks")
    return resolved


def _copy_model_tree(
    *, source: Path, staging: Path, copy_mode: Literal["copy", "hardlink"]
) -> None:
    if staging.exists():
        raise MageDcvcQualifiedProviderError("qualified staging directory already exists")

    def copy_file(src: str, dst: str) -> str:
        source_path = Path(src)
        destination_path = Path(dst)
        if source_path.is_symlink():
            raise MageDcvcQualifiedProviderError("source Mage tree contains a symlink")
        if copy_mode == "hardlink":
            os.link(source_path, destination_path)
            return str(destination_path)
        return shutil.copy2(str(source_path), str(destination_path))

    try:
        shutil.copytree(source, staging, copy_function=copy_file, symlinks=False)
    except OSError as error:
        raise MageDcvcQualifiedProviderError("could not copy qualified Mage tree") from error


def _validate_distinct_trees(*, source: Path, target: Path) -> None:
    if source == target or source in target.parents or target in source.parents:
        raise MageDcvcQualifiedProviderError(
            "source and qualified target must be separate non-nested directories"
        )
    if not source.is_dir():
        raise MageDcvcQualifiedProviderError(f"source Mage directory is missing: {source}")


def _require_safe_staging(*, staging: Path, target: Path) -> None:
    resolved = staging.resolve()
    if resolved.parent != target.parent.resolve():
        raise MageDcvcQualifiedProviderError("staging directory escaped target parent")
    if not resolved.name.startswith(
        f".{target.name}.robata-dcvc-v2-"
    ) or not resolved.name.endswith(".tmp"):
        raise MageDcvcQualifiedProviderError("staging directory name is not owned by qualifier")


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
        raise MageDcvcQualifiedProviderError(f"could not read file: {resolved}") from error
    if byte_count <= 0:
        raise MageDcvcQualifiedProviderError(f"required file is empty: {resolved}")
    return digest.hexdigest(), byte_count


def _qualified_bundle_sha256(bundle: MageDcvcQualifiedProviderBundle) -> Sha256Digest:
    return semantic_sha256(bundle.model_dump(mode="json", exclude={"bundle_semantic_sha256"}))


def _qualified_manifest_sha256(manifest: MageDcvcQualifiedProviderManifest) -> Sha256Digest:
    return semantic_sha256(
        {
            "manifest_version": manifest.manifest_version,
            "provider_version": manifest.provider_version,
            "copy_mode": manifest.copy_mode,
            "bundle": manifest.bundle.model_dump(mode="json"),
            "qualified_checkpoint_manifest_sha256": (
                manifest.qualified_checkpoint_manifest.manifest_sha256
            ),
        }
    )


def _write_canonical_manifest(*, manifest: MageDcvcQualifiedProviderManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise MageDcvcQualifiedProviderError(
            "could not publish qualified provider manifest"
        ) from error


__all__ = [
    "MAGE_DCVC_QUALIFIED_BUNDLE_MANIFEST_NAME",
    "MAGE_DCVC_QUALIFIED_BUNDLE_VERSION",
    "MAGE_DCVC_QUALIFIED_MANIFEST_VERSION",
    "MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY",
    "MageDcvcQualifiedProviderBundle",
    "MageDcvcQualifiedProviderError",
    "MageDcvcQualifiedProviderFile",
    "MageDcvcQualifiedProviderManifest",
    "load_mage_dcvc_qualified_provider_manifest",
    "qualify_mage_dcvc_provider_v2",
    "verify_mage_dcvc_qualified_provider",
    "write_qualified_checkpoint_manifest",
]
