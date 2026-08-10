from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.mage_traditional_codec_cache import (
    MageTraditionalCodecCacheAdmission,
    MageTraditionalCodecCacheError,
    MageTraditionalCodecCacheManifest,
    build_mage_traditional_codec_cache_manifest,
    build_mage_traditional_codec_effective_config,
    build_mage_traditional_codec_toolchain_identity,
    load_mage_traditional_codec_cache_manifest,
    mage_traditional_codec_namespace_identity,
    mage_traditional_codec_provider_identity,
    verify_mage_traditional_codec_cache_manifest,
    write_mage_traditional_codec_cache_manifest,
)
from robata.inference.mage_video_endpoint import (
    MageVideoCameraEncoding,
    MageVideoCodecPolicy,
    MageVideoDecoderRequest,
    MageVideoEndpointRequest,
    MageVideoModelIdentity,
    MageVideoRuntimeIdentityBinding,
    build_mage_video_codec_policy_identity,
    build_mage_video_context_manifest,
    build_mage_video_segment_manifest,
)
from robata.inference.mage_video_runtime import (
    MageVideoRuntimeIdentity,
    MageVideoTraditionalCodecCacheBinding,
    mage_video_codec_config_sha256,
)

CHECKPOINT_SHA256 = "a" * 64
PROVIDER_IMPLEMENTATION_SHA256 = "b" * 64
PACKAGE_ARTIFACT_SHA256 = "c" * 64
EXECUTABLE_SHA256 = "d" * 64
COMMAND_CONTRACT_SHA256 = "e" * 64
IMAGE_DIGEST = "f" * 64


def _fixture(tmp_path: Path) -> dict[str, Any]:
    policy = MageVideoCodecPolicy(
        preprocess_device="cpu",
        target_canvas=8,
        group_size=8,
        images_per_group=4,
        patch_size=16,
        max_pixels=150_000,
        min_group_frames=8,
        max_group_frames=64,
    )
    toolchain = build_mage_traditional_codec_toolchain_identity(
        package_version="0.2.5",
        package_artifact_sha256=PACKAGE_ARTIFACT_SHA256,
        executable_sha256=EXECUTABLE_SHA256,
        provider_command_contract_sha256=COMMAND_CONTRACT_SHA256,
        container_image_reference=f"python:3.12-slim@sha256:{IMAGE_DIGEST}",
        container_image_digest=IMAGE_DIGEST,
        container_platform="linux/amd64",
    )
    effective_config = build_mage_traditional_codec_effective_config(
        codec_policy=policy,
        provider_options={
            "decode_backend": "ffmpeg_native",
            "grouping_mode": "readiness",
            "parallel_decode_cv_reader": False,
        },
    )
    provider_identity = mage_traditional_codec_provider_identity(
        provider_implementation_sha256=PROVIDER_IMPLEMENTATION_SHA256,
        toolchain_identity_sha256=toolchain.toolchain_identity_sha256,
        effective_config_sha256=effective_config.effective_config_sha256,
    )
    policy_sha256 = build_mage_video_codec_policy_identity(policy).policy_sha256
    codec_config_sha256 = mage_video_codec_config_sha256(policy.native_codec_config())
    namespace_identity = mage_traditional_codec_namespace_identity(
        checkpoint_manifest_sha256=CHECKPOINT_SHA256,
        codec_policy_sha256=policy_sha256,
        codec_config_sha256=codec_config_sha256,
        provider_identity_sha256=provider_identity,
    )
    cache_base_root = tmp_path / "cache"
    provider_directory = cache_base_root / namespace_identity / "segment-00"
    provider_directory.mkdir(parents=True)
    source_path = tmp_path / "segment-00.mp4"
    source_payload = b"h264-segment-bytes"
    source_path.write_bytes(source_payload)
    (provider_directory / "canvas_000.jpg").write_bytes(b"jpeg-canvas")
    (provider_directory / "meta.json").write_bytes(b'{"canvas_files":["canvas_000.jpg"],"fps":30}')
    (provider_directory / "src_patch_position.npy").write_bytes(b"numpy-position-bytes")
    manifest = build_mage_traditional_codec_cache_manifest(
        checkpoint_manifest_sha256=CHECKPOINT_SHA256,
        codec_policy=policy,
        provider_implementation_sha256=PROVIDER_IMPLEMENTATION_SHA256,
        toolchain=toolchain,
        effective_config=effective_config,
        cache_base_root=cache_base_root,
        observations=[(source_path, provider_directory)],
    )
    return {
        "policy": policy,
        "toolchain": toolchain,
        "effective_config": effective_config,
        "provider_identity": provider_identity,
        "cache_base_root": cache_base_root,
        "provider_directory": provider_directory,
        "source_path": source_path,
        "source_payload": source_payload,
        "manifest": manifest,
    }


def _request(
    fixture: dict[str, Any], *, policy: MageVideoCodecPolicy | None = None
) -> MageVideoEndpointRequest:
    source_path = fixture["source_path"]
    source_payload = fixture["source_payload"]
    assert isinstance(source_path, Path)
    assert isinstance(source_payload, bytes)
    segment = build_mage_video_segment_manifest(
        segment_id="segment-00",
        camera_id="cam-01",
        durable_path=str(source_path),
        media_type="video/mp4",
        content_sha256=exact_bytes_sha256(source_payload),
        byte_count=len(source_payload),
    )
    context = build_mage_video_context_manifest(
        context_id="context-00",
        context_payload_sha256=exact_bytes_sha256(b"context"),
        segment_manifest_identities=[segment.manifest_identity],
    )
    return MageVideoEndpointRequest(
        request_id="traditional-request-00",
        model_identity=MageVideoModelIdentity(
            model_identifier="microsoft/Mage-VL",
            model_revision="traditional-cache-test",
            checkpoint_manifest_sha256=CHECKPOINT_SHA256,
            runtime_identity=MageVideoRuntimeIdentityBinding.from_runtime_identity(
                MageVideoRuntimeIdentity()
            ),
        ),
        codec_policy=policy or fixture["policy"],
        context_manifest=context,
        camera_encodings=[
            MageVideoCameraEncoding(
                encoder_id="camera-encoder-cam-01",
                segment_manifest=segment,
            )
        ],
        decoder=MageVideoDecoderRequest(
            decoder_id="mage-decoder",
            prompt="Observe the segment.",
            max_new_tokens=32,
        ),
    )


def _admission(fixture: dict[str, Any]) -> MageTraditionalCodecCacheAdmission:
    return MageTraditionalCodecCacheAdmission(
        manifest=fixture["manifest"],
        expected_checkpoint_manifest_sha256=CHECKPOINT_SHA256,
        expected_codec_policy=fixture["policy"],
        expected_provider_identity_sha256=fixture["provider_identity"],
        expected_toolchain_identity_sha256=fixture["toolchain"].toolchain_identity_sha256,
        expected_container_image_digest=IMAGE_DIGEST,
    )


def test_manifest_binds_package_image_config_source_and_exact_assets(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    manifest_path = tmp_path / "traditional-cache-manifest.json"

    write_mage_traditional_codec_cache_manifest(manifest=manifest, path=manifest_path)
    loaded = load_mage_traditional_codec_cache_manifest(path=manifest_path)
    verified = verify_mage_traditional_codec_cache_manifest(
        manifest=loaded,
        expected_checkpoint_manifest_sha256=CHECKPOINT_SHA256,
        expected_codec_policy_sha256=build_mage_video_codec_policy_identity(
            fixture["policy"]
        ).policy_sha256,
        expected_provider_identity_sha256=fixture["provider_identity"],
        expected_toolchain_identity_sha256=fixture["toolchain"].toolchain_identity_sha256,
        expected_container_image_digest=IMAGE_DIGEST,
    )

    assert loaded == manifest
    assert loaded.toolchain.package_version == "0.2.5"
    assert loaded.toolchain.package_artifact_sha256 == PACKAGE_ARTIFACT_SHA256
    assert loaded.toolchain.executable_sha256 == EXECUTABLE_SHA256
    assert loaded.toolchain.container_image_reference.endswith(IMAGE_DIGEST)
    assert loaded.effective_config.provider_options["decode_backend"] == "ffmpeg_native"
    assert [asset.relative_path for asset in verified[0].assets] == [
        "canvas_000.jpg",
        "meta.json",
        "src_patch_position.npy",
    ]


def test_manifest_publish_reuses_only_exact_existing_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = tmp_path / "traditional-cache-manifest.json"

    write_mage_traditional_codec_cache_manifest(manifest=fixture["manifest"], path=manifest_path)
    first_bytes = manifest_path.read_bytes()
    write_mage_traditional_codec_cache_manifest(manifest=fixture["manifest"], path=manifest_path)

    assert manifest_path.read_bytes() == first_bytes
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


def test_manifest_publish_never_overwrites_existing_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = tmp_path / "traditional-cache-manifest.json"
    foreign_bytes = b"foreign-manifest-identity"
    manifest_path.write_bytes(foreign_bytes)

    with pytest.raises(
        MageTraditionalCodecCacheError,
        match="existing traditional codec cache manifest is invalid",
    ):
        write_mage_traditional_codec_cache_manifest(
            manifest=fixture["manifest"], path=manifest_path
        )

    assert manifest_path.read_bytes() == foreign_bytes
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


def test_admission_returns_additive_traditional_runtime_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = _request(fixture)
    binding = _admission(fixture)(request, [fixture["source_path"]])

    assert isinstance(binding, MageVideoTraditionalCodecCacheBinding)
    assert binding.codec_engine == "hevc"
    assert binding.codec_config_sha256 == fixture["manifest"].codec_config_sha256
    assert binding.provider_identity_sha256 == fixture["provider_identity"]
    assert binding.toolchain_identity_sha256 == fixture["toolchain"].toolchain_identity_sha256
    assert binding.provider_cache_directory == fixture["provider_directory"].resolve()


@pytest.mark.parametrize(
    "relative_path",
    ["canvas_000.jpg", "meta.json", "src_patch_position.npy"],
)
def test_manifest_verification_fails_closed_for_asset_tampering(
    tmp_path: Path, relative_path: str
) -> None:
    fixture = _fixture(tmp_path)
    (fixture["provider_directory"] / relative_path).write_bytes(b"tampered")

    with pytest.raises(MageTraditionalCodecCacheError, match="provider assets changed"):
        verify_mage_traditional_codec_cache_manifest(manifest=fixture["manifest"])


def test_admission_requires_deployment_pinned_toolchain_and_image(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(MageTraditionalCodecCacheError, match="toolchain identity mismatch"):
        MageTraditionalCodecCacheAdmission(
            manifest=fixture["manifest"],
            expected_checkpoint_manifest_sha256=CHECKPOINT_SHA256,
            expected_codec_policy=fixture["policy"],
            expected_provider_identity_sha256=fixture["provider_identity"],
            expected_toolchain_identity_sha256="0" * 64,
            expected_container_image_digest=IMAGE_DIGEST,
        )
    with pytest.raises(MageTraditionalCodecCacheError, match="container image identity mismatch"):
        MageTraditionalCodecCacheAdmission(
            manifest=fixture["manifest"],
            expected_checkpoint_manifest_sha256=CHECKPOINT_SHA256,
            expected_codec_policy=fixture["policy"],
            expected_provider_identity_sha256=fixture["provider_identity"],
            expected_toolchain_identity_sha256=fixture["toolchain"].toolchain_identity_sha256,
            expected_container_image_digest="0" * 64,
        )


def test_toolchain_package_or_image_mutation_invalidates_manifest_model(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture["manifest"].model_dump(mode="json")
    payload["toolchain"]["package_version"] = "0.2.6"
    with pytest.raises(ValidationError, match="toolchain identity"):
        MageTraditionalCodecCacheManifest.model_validate(payload, strict=True)

    payload = fixture["manifest"].model_dump(mode="json")
    payload["toolchain"]["container_image_reference"] = "python:3.12-slim:latest"
    with pytest.raises(ValidationError, match="digest pinned"):
        MageTraditionalCodecCacheManifest.model_validate(payload, strict=True)


def test_request_policy_mismatch_is_rejected_before_runtime(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    admission = _admission(fixture)
    different_policy = fixture["policy"].model_copy(update={"max_pixels": 200_000})
    request = _request(fixture, policy=different_policy)

    with pytest.raises(MageTraditionalCodecCacheError, match="request policy"):
        admission(request, [fixture["source_path"]])


def test_unbound_provider_directory_invalidates_namespace(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unexpected = fixture["provider_directory"].parent / "unbound-entry"
    unexpected.mkdir()

    with pytest.raises(MageTraditionalCodecCacheError, match="unbound provider directories"):
        verify_mage_traditional_codec_cache_manifest(manifest=fixture["manifest"])
