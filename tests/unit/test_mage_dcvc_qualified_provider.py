from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from robata.inference.mage_checkpoint_identity import build_mage_checkpoint_manifest
from robata.inference.mage_dcvc_qualified_provider import (
    MAGE_DCVC_QUALIFIED_BUNDLE_MANIFEST_NAME,
    MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY,
    MageDcvcQualifiedProviderError,
    load_mage_dcvc_qualified_provider_manifest,
    qualify_mage_dcvc_provider_v2,
    verify_mage_dcvc_qualified_provider,
    verify_mage_dcvc_qualified_provider_sources,
    write_qualified_checkpoint_manifest,
)


def _model_tree(tmp_path: Path) -> tuple[Path, object]:
    root = tmp_path / "Mage-VL"
    files = {
        "config.json": b'{"model_type":"mage_vl"}',
        "preprocessor_config.json": b'{"codec":{"dcvc":{"max_side":0}}}',
        "modeling_mage_vl.py": b"class Mage: pass\n",
        "model.safetensors": b"small-test-weight",
        "neural_codec/dcvc_readiness_gen.py": b"def main(): pass\n",
        "neural_codec/DCVC/src/__init__.py": b"# package\n",
        "README.md": b"incidental but copied\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    checkpoint = build_mage_checkpoint_manifest(
        model_directory=root,
        model_identifier="Mage-VL",
        model_revision="upstream-test",
    )
    return root, checkpoint


def _provider_sources(tmp_path: Path) -> tuple[Path, ...]:
    root = tmp_path / "provider-source"
    root.mkdir()
    protocol = root / "protocol.py"
    worker = root / "worker.py"
    protocol.write_bytes(b"PROVIDER_VERSION = 'v2'\n")
    worker.write_bytes(b"def serve(): return 'v2'\n")
    return protocol, worker


def test_qualifier_preserves_source_and_binds_provider_into_new_checkpoint(tmp_path: Path) -> None:
    source, checkpoint = _model_tree(tmp_path)
    source_snapshot = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    target = tmp_path / "Mage-VL-Robata-DCVC-V2"
    manifest_path = tmp_path / "evidence" / "qualified-provider.json"
    provider_sources = _provider_sources(tmp_path)

    manifest = qualify_mage_dcvc_provider_v2(
        source_model_directory=source,
        source_checkpoint_manifest=checkpoint,
        target_model_directory=target,
        qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
        qualified_model_revision="upstream-test+robata-dcvc-provider-v2-20260809",
        provider_source_files=provider_sources,
        manifest_path=manifest_path,
    )

    assert source_snapshot == {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert manifest.bundle.source_checkpoint_manifest_sha256 == checkpoint.manifest_sha256
    assert manifest.qualified_checkpoint_manifest.manifest_sha256 != checkpoint.manifest_sha256
    assert manifest.qualified_checkpoint_manifest.model_revision.endswith("20260809")
    provider_root = target / "neural_codec" / MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY
    assert (provider_root / "protocol.py").is_file()
    assert (provider_root / "worker.py").is_file()
    assert (provider_root / MAGE_DCVC_QUALIFIED_BUNDLE_MANIFEST_NAME).is_file()
    assert load_mage_dcvc_qualified_provider_manifest(manifest_path=manifest_path) == manifest
    verify_mage_dcvc_qualified_provider(manifest=manifest)
    verify_mage_dcvc_qualified_provider_sources(
        manifest=manifest, provider_source_files=provider_sources
    )

    endpoint_checkpoint_path = tmp_path / "evidence" / "checkpoint-manifest-v2.json"
    write_qualified_checkpoint_manifest(manifest=manifest, path=endpoint_checkpoint_path)
    assert endpoint_checkpoint_path.is_file()


def test_runtime_source_admission_rejects_different_or_incomplete_bytes(tmp_path: Path) -> None:
    source, checkpoint = _model_tree(tmp_path)
    provider_sources = _provider_sources(tmp_path)
    manifest = qualify_mage_dcvc_provider_v2(
        source_model_directory=source,
        source_checkpoint_manifest=checkpoint,
        target_model_directory=tmp_path / "qualified-source-admission",
        qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
        qualified_model_revision="upstream-test+source-admission",
        provider_source_files=provider_sources,
        manifest_path=tmp_path / "source-admission.json",
    )

    protocol, worker = provider_sources
    verify_mage_dcvc_qualified_provider_sources(
        manifest=manifest, provider_source_files=(protocol, worker)
    )
    with pytest.raises(MageDcvcQualifiedProviderError, match="source set"):
        verify_mage_dcvc_qualified_provider_sources(
            manifest=manifest, provider_source_files=(protocol,)
        )
    worker.write_bytes(worker.read_bytes() + b"# local drift\n")
    with pytest.raises(MageDcvcQualifiedProviderError, match="source bytes differ"):
        verify_mage_dcvc_qualified_provider_sources(
            manifest=manifest, provider_source_files=(protocol, worker)
        )


def test_qualifier_rejects_nested_or_existing_target(tmp_path: Path) -> None:
    source, checkpoint = _model_tree(tmp_path)
    provider_sources = _provider_sources(tmp_path)

    with pytest.raises(MageDcvcQualifiedProviderError, match="non-nested"):
        qualify_mage_dcvc_provider_v2(
            source_model_directory=source,
            source_checkpoint_manifest=checkpoint,
            target_model_directory=source / "qualified",
            qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
            qualified_model_revision="upstream-test+v2",
            provider_source_files=provider_sources,
            manifest_path=tmp_path / "nested.json",
        )

    target = tmp_path / "already-there"
    target.mkdir()
    with pytest.raises(MageDcvcQualifiedProviderError, match="already exists"):
        qualify_mage_dcvc_provider_v2(
            source_model_directory=source,
            source_checkpoint_manifest=checkpoint,
            target_model_directory=target,
            qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
            qualified_model_revision="upstream-test+v2",
            provider_source_files=provider_sources,
            manifest_path=tmp_path / "existing.json",
        )


def test_verifier_detects_provider_or_checkpoint_tampering(tmp_path: Path) -> None:
    source, checkpoint = _model_tree(tmp_path)
    target = tmp_path / "qualified"
    manifest = qualify_mage_dcvc_provider_v2(
        source_model_directory=source,
        source_checkpoint_manifest=checkpoint,
        target_model_directory=target,
        qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
        qualified_model_revision="upstream-test+v2",
        provider_source_files=_provider_sources(tmp_path),
        manifest_path=tmp_path / "qualified.json",
    )
    worker = target / "neural_codec" / MAGE_DCVC_QUALIFIED_PROVIDER_DIRECTORY / "worker.py"
    worker.write_bytes(worker.read_bytes() + b"# tampered\n")

    with pytest.raises(Exception, match="changed included files"):
        verify_mage_dcvc_qualified_provider(manifest=manifest)


def test_hardlink_mode_is_explicit_and_still_verifiable(tmp_path: Path) -> None:
    source, checkpoint = _model_tree(tmp_path)
    target = tmp_path / "qualified-hardlink"
    manifest = qualify_mage_dcvc_provider_v2(
        source_model_directory=source,
        source_checkpoint_manifest=checkpoint,
        target_model_directory=target,
        qualified_model_identifier="Mage-VL-Robata-DCVC-V2",
        qualified_model_revision="upstream-test+v2-hardlink",
        provider_source_files=_provider_sources(tmp_path),
        manifest_path=tmp_path / "qualified-hardlink.json",
        copy_mode="hardlink",
    )

    assert manifest.copy_mode == "hardlink"
    verify_mage_dcvc_qualified_provider(manifest=manifest)
