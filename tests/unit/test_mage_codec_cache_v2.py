from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from robata.contracts.hashing import canonical_json_bytes
from robata.inference.mage_checkpoint_identity import build_mage_checkpoint_manifest
from robata.inference.mage_codec_cache_v2 import (
    MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME,
    MageCodecCacheV2Error,
    build_mage_codec_cache_manifest_v2,
    load_mage_codec_cache_manifest_v2,
    mage_codec_v2_namespace_identity,
    upstream_mage_codec_cache_directory_name,
    validate_mage_dcvc_effective_config_for_policy,
    verify_mage_codec_cache_manifest_v2,
    write_mage_codec_cache_manifest_v2,
)
from robata.inference.mage_dcvc_preparation_protocol import (
    MAGE_DCVC_PREPARATION_SIDECAR_NAME,
    MageDcvcEffectiveConfig,
    MageDcvcPreparationArtifact,
    MageDcvcPreparedAsset,
    mage_dcvc_artifact_semantic_sha256,
    mage_dcvc_effective_config_sha256,
    mage_dcvc_preparation_identity,
)
from robata.inference.mage_video_endpoint import (
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
    build_mage_video_codec_policy_identity,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model(tmp_path: Path) -> tuple[Path, Any]:
    root = tmp_path / "Mage-VL-Robata-DCVC-V2"
    for relative, payload in {
        "config.json": b'{"model_type":"mage_vl"}',
        "model.safetensors": b"weights",
        "modeling_mage_vl.py": b"class Model: pass\n",
        "preprocessor_config.json": canonical_json_bytes(
            {
                "codec": {
                    "dcvc": {
                        "qp": 42,
                        "reset_interval": 64,
                        "intra_period": -1,
                        "max_side": 0,
                        "max_group_frames": 128,
                        "threshold_scale": 1.0,
                    }
                }
            }
        ),
        "neural_codec/dcvc_rt_intra.tar": b"intra",
        "neural_codec/dcvc_rt_inter.tar": b"inter",
        "neural_codec/robata_provider_v2/worker.py": b"def serve(): pass\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = build_mage_checkpoint_manifest(
        model_directory=root,
        model_identifier="Mage-VL-Robata-DCVC-V2",
        model_revision="test+robata-dcvc-provider-v2",
    )
    return root, manifest


def _policy(max_side: int = 0, *, sequence_length_frames: int = 0) -> MageVideoCodecPolicy:
    return MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device="cuda",
        target_canvas=8,
        group_size=8,
        images_per_group=1,
        patch_size=16,
        max_pixels=65_536,
        min_group_frames=8,
        max_group_frames=64,
        neural_parameters=MageVideoNeuralCodecParameters(
            quantization_parameter=42,
            reset_interval=64,
            intra_period=-1,
            max_side=max_side,
            sequence_length_frames=sequence_length_frames,
            canvas_token_side=None,
            readiness_coverage_bins=3,
            readiness_delta_ratio=0.05,
            bitcost_percentile=99,
            decode_backsearch_max=16,
        ),
    )


def _config(max_side: int = 0, *, implementation: str = "1" * 64) -> MageDcvcEffectiveConfig:
    values: dict[str, Any] = {
        "provider_implementation_sha256": implementation,
        "intra_checkpoint_sha256": "2" * 64,
        "inter_checkpoint_sha256": "3" * 64,
        "preparation_device": "cuda",
        "device_concurrency_policy": "exclusive-shared-device-v1",
        "max_side": max_side,
        "target_canvas": 8,
        "sampled_frame_count": 64,
        "group_size": 8,
        "images_per_group": 1,
        "max_pixels": 65_536,
        "min_group_frames": 8,
    }
    provisional = MageDcvcEffectiveConfig.model_construct(
        **values,
        effective_config_sha256="0" * 64,
    )
    return MageDcvcEffectiveConfig(
        **values,
        effective_config_sha256=mage_dcvc_effective_config_sha256(provisional),
    )


def _write_worker_output(
    *, directory: Path, source: Path, config: MageDcvcEffectiveConfig
) -> MageDcvcPreparationArtifact:
    directory.mkdir(parents=True)
    files = {
        "canvas_000.jpg": b"canvas",
        "frame_ids.npy": b"frames",
        "src_patch_position.npy": b"positions",
        "meta.json": canonical_json_bytes(
            {
                "canvas_files": ["canvas_000.jpg"],
                "robata_dcvc_provider": {
                    "provider_version": config.provider_version,
                    "recipe_version": config.recipe_version,
                    "effective_config_sha256": config.effective_config_sha256,
                    "provider_implementation_sha256": config.provider_implementation_sha256,
                    "effective_max_side": config.max_side,
                    "encoded_frame_extent": config.encoded_frame_extent,
                },
            }
        ),
    }
    assets: list[MageDcvcPreparedAsset] = []
    for relative, payload in sorted(files.items()):
        path = directory / relative
        path.write_bytes(payload)
        assets.append(
            MageDcvcPreparedAsset(
                relative_path=relative,
                byte_count=len(payload),
                sha256=_sha(payload),
            )
        )
    source_payload = source.read_bytes()
    preparation_identity = mage_dcvc_preparation_identity(
        source_content_sha256=_sha(source_payload),
        source_byte_count=len(source_payload),
        effective_config_sha256=config.effective_config_sha256,
    )
    metadata = {
        "provider_version": config.provider_version,
        "recipe_version": config.recipe_version,
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "engine": config.engine,
        "preparation_device": config.preparation_device,
        "device_concurrency_policy": config.device_concurrency_policy,
        "max_side": config.max_side,
        "configured_sampled_frame_count": config.sampled_frame_count,
        "effective_sampled_frame_count": 60,
        "max_encoded_frame_id": 119,
        "engine_load_count": 1,
        "engine_load_seconds": 0.75,
        "worker_completed_job_count": 1,
        "sequence_reset_count_for_job": 1,
        "sequence_length_frames": config.sequence_length_frames,
        "canvas_token_side": config.canvas_token_side,
        "encoded_frame_extent": config.encoded_frame_extent,
        "segment_state_policy": "reset-per-job",
    }
    values: dict[str, Any] = {
        "preparation_identity": preparation_identity,
        "effective_config_sha256": config.effective_config_sha256,
        "provider_implementation_sha256": config.provider_implementation_sha256,
        "source_content_sha256": _sha(source_payload),
        "source_byte_count": len(source_payload),
        "assets": tuple(assets),
        "provider_metadata": metadata,
    }
    provisional = MageDcvcPreparationArtifact.model_construct(
        **values,
        artifact_semantic_sha256="0" * 64,
    )
    artifact = MageDcvcPreparationArtifact(
        **values,
        artifact_semantic_sha256=mage_dcvc_artifact_semantic_sha256(provisional),
    )
    (directory / MAGE_DCVC_PREPARATION_SIDECAR_NAME).write_bytes(
        canonical_json_bytes(artifact.model_dump(mode="json"))
    )
    return artifact


def _build(tmp_path: Path) -> tuple[Any, Path, Path]:
    model_root, checkpoint = _model(tmp_path)
    policy = _policy()
    config = _config()
    policy_sha = build_mage_video_codec_policy_identity(policy).policy_sha256
    namespace = mage_codec_v2_namespace_identity(
        checkpoint_manifest_sha256=checkpoint.manifest_sha256,
        codec_policy_sha256=policy_sha,
        provider_implementation_sha256=config.provider_implementation_sha256,
        effective_config_sha256=config.effective_config_sha256,
    )
    source = tmp_path / "segments" / "segment-000.mp4"
    source.parent.mkdir()
    source.write_bytes(b"immutable-video")
    name = upstream_mage_codec_cache_directory_name(
        video_path=source,
        codec_policy=policy,
        model_directory=model_root,
    )
    cache_base = tmp_path / "cache"
    output = cache_base / namespace / name
    _write_worker_output(directory=output, source=source, config=config)
    manifest = build_mage_codec_cache_manifest_v2(
        checkpoint_manifest=checkpoint,
        codec_policy=policy,
        effective_config=config,
        cache_base_root=cache_base,
        model_directory=model_root,
        observations=[(source, output, "BUILT", 1.25)],
        prewarm_wall_seconds=1.25,
    )
    return manifest, source, output


def test_v2_manifest_binds_config_implementation_receipt_and_assets(tmp_path: Path) -> None:
    manifest, _source, output = _build(tmp_path)

    assert manifest.manifest_version == "mage-codec-cache-manifest-v2"
    assert manifest.recipe_version == "mage-dcvc-readiness-explicit-v2"
    assert manifest.entry_count == manifest.built_count == 1
    verified = verify_mage_codec_cache_manifest_v2(manifest=manifest)
    assert len(verified) == 1
    assert verified[0].provider_implementation_sha256 == "1" * 64
    assert verified[0].provider_metadata["max_side"] == 0
    assert (output / ".robata-cache-entry-v2.json").is_file()

    path = tmp_path / "evidence" / "cache-v2.json"
    write_mage_codec_cache_manifest_v2(manifest=manifest, path=path)
    assert load_mage_codec_cache_manifest_v2(path=path) == manifest


def test_v2_namespace_changes_with_implementation_or_effective_config() -> None:
    policy_sha = build_mage_video_codec_policy_identity(_policy()).policy_sha256
    baseline = mage_codec_v2_namespace_identity(
        checkpoint_manifest_sha256="4" * 64,
        codec_policy_sha256=policy_sha,
        provider_implementation_sha256="1" * 64,
        effective_config_sha256=_config().effective_config_sha256,
    )
    changed_implementation = mage_codec_v2_namespace_identity(
        checkpoint_manifest_sha256="4" * 64,
        codec_policy_sha256=policy_sha,
        provider_implementation_sha256="5" * 64,
        effective_config_sha256=_config(implementation="5" * 64).effective_config_sha256,
    )
    changed_max_side = mage_codec_v2_namespace_identity(
        checkpoint_manifest_sha256="4" * 64,
        codec_policy_sha256=build_mage_video_codec_policy_identity(_policy(448)).policy_sha256,
        provider_implementation_sha256="1" * 64,
        effective_config_sha256=_config(448).effective_config_sha256,
    )

    assert len({baseline, changed_implementation, changed_max_side}) == 3


def test_v2_rejects_seq_len_compute_shortcut_and_policy_divergence() -> None:
    with pytest.raises(MageCodecCacheV2Error, match="sequence_length_frames=0"):
        validate_mage_dcvc_effective_config_for_policy(
            effective_config=_config(),
            codec_policy=_policy(sequence_length_frames=8),
        )
    with pytest.raises(MageCodecCacheV2Error, match="max_side"):
        validate_mage_dcvc_effective_config_for_policy(
            effective_config=_config(448),
            codec_policy=_policy(0),
        )


@pytest.mark.parametrize("corruption", ["asset", "receipt", "source", "path"])
def test_v2_verification_fails_closed_for_tampering(tmp_path: Path, corruption: str) -> None:
    manifest, source, output = _build(tmp_path)
    if corruption == "asset":
        (output / "canvas_000.jpg").write_bytes(b"changed")
        match = "asset bytes"
    elif corruption == "receipt":
        receipt = output / MAGE_DCVC_PREPARATION_SIDECAR_NAME
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        match = "canonical JSON"
    elif corruption == "source":
        source.write_bytes(b"changed")
        match = "source content"
    else:
        escaped = tmp_path / "outside" / output.name
        changed = manifest.entries[0].model_copy(
            update={"provider_cache_directory": str(escaped.resolve())}
        )
        manifest = manifest.model_copy(update={"entries": (changed,)})
        match = "escaped qualified root"

    with pytest.raises(MageCodecCacheV2Error, match=match):
        verify_mage_codec_cache_manifest_v2(manifest=manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_encoded_frame_id", 1),
        ("engine_load_count", 2),
        ("engine_load_seconds", -1.0),
        ("worker_completed_job_count", 0),
        ("sequence_reset_count_for_job", 0),
    ],
)
def test_v2_rejects_unproven_persistent_worker_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, _source, output = _build(tmp_path)
    receipt = output / MAGE_DCVC_PREPARATION_SIDECAR_NAME
    original = MageDcvcPreparationArtifact.model_validate_json(receipt.read_bytes(), strict=True)
    metadata = dict(original.provider_metadata)
    metadata[field] = value  # type: ignore[assignment]
    values = original.model_dump(mode="python", exclude={"artifact_semantic_sha256"})
    values["assets"] = original.assets
    values["provider_metadata"] = metadata
    provisional = MageDcvcPreparationArtifact.model_construct(
        **values, artifact_semantic_sha256="0" * 64
    )
    changed = MageDcvcPreparationArtifact(
        **values, artifact_semantic_sha256=mage_dcvc_artifact_semantic_sha256(provisional)
    )
    receipt.write_bytes(canonical_json_bytes(changed.model_dump(mode="json")))
    (output / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME).unlink()

    model_root = tmp_path / "Mage-VL-Robata-DCVC-V2"
    checkpoint = build_mage_checkpoint_manifest(
        model_directory=model_root,
        model_identifier="Mage-VL-Robata-DCVC-V2",
        model_revision="test+robata-dcvc-provider-v2",
    )
    with pytest.raises(MageCodecCacheV2Error, match=field):
        build_mage_codec_cache_manifest_v2(
            checkpoint_manifest=checkpoint,
            codec_policy=_policy(),
            effective_config=manifest.effective_config,
            cache_base_root=Path(manifest.cache_base_root),
            model_directory=model_root,
            observations=[(Path(manifest.entries[0].source_path), output, "BUILT", 1.0)],
            prewarm_wall_seconds=1.0,
        )


def test_v2_manifest_rebuild_accepts_only_verified_existing_entry(tmp_path: Path) -> None:
    manifest, source, output = _build(tmp_path)
    model_root = tmp_path / "Mage-VL-Robata-DCVC-V2"
    policy = _policy()
    config = manifest.effective_config
    checkpoint = build_mage_checkpoint_manifest(
        model_directory=model_root,
        model_identifier="Mage-VL-Robata-DCVC-V2",
        model_revision="test+robata-dcvc-provider-v2",
    )

    replay = build_mage_codec_cache_manifest_v2(
        checkpoint_manifest=checkpoint,
        codec_policy=policy,
        effective_config=config,
        cache_base_root=tmp_path / "cache",
        model_directory=model_root,
        observations=[(source, output, "VERIFIED_HIT", 0.001)],
        prewarm_wall_seconds=0.001,
    )

    assert replay.built_count == 0
    assert replay.verified_hit_count == 1
    assert replay.entries[0].entry_semantic_sha256 == manifest.entries[0].entry_semantic_sha256


def _rebuild_verified_hit(
    *,
    tmp_path: Path,
    manifest: Any,
    source: Path,
    output: Path,
) -> Any:
    model_root = tmp_path / "Mage-VL-Robata-DCVC-V2"
    checkpoint = build_mage_checkpoint_manifest(
        model_directory=model_root,
        model_identifier="Mage-VL-Robata-DCVC-V2",
        model_revision="test+robata-dcvc-provider-v2",
    )
    return build_mage_codec_cache_manifest_v2(
        checkpoint_manifest=checkpoint,
        codec_policy=_policy(),
        effective_config=manifest.effective_config,
        cache_base_root=Path(manifest.cache_base_root),
        model_directory=model_root,
        observations=[(source, output, "VERIFIED_HIT", 0.001)],
        prewarm_wall_seconds=0.001,
    )


def test_v2_verified_hit_republishes_missing_entry_sidecar_exactly(tmp_path: Path) -> None:
    manifest, source, output = _build(tmp_path)
    sidecar = output / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME
    expected_bytes = sidecar.read_bytes()
    sidecar.unlink()

    replay = _rebuild_verified_hit(
        tmp_path=tmp_path,
        manifest=manifest,
        source=source,
        output=output,
    )

    assert sidecar.read_bytes() == expected_bytes
    assert replay.built_count == 0
    assert replay.verified_hit_count == 1
    assert replay.entries[0].entry_semantic_sha256 == manifest.entries[0].entry_semantic_sha256
    assert verify_mage_codec_cache_manifest_v2(manifest=replay)
    assert not tuple(output.glob(f".{sidecar.name}.*.tmp"))


@pytest.mark.parametrize("corruption", ["source", "artifact", "asset"])
def test_v2_missing_sidecar_is_not_published_for_tampered_backing(
    tmp_path: Path, corruption: str
) -> None:
    manifest, source, output = _build(tmp_path)
    sidecar = output / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME
    sidecar.unlink()
    if corruption == "source":
        source.write_bytes(b"tampered-source")
        match = "does not match source/config"
    elif corruption == "artifact":
        preparation = output / MAGE_DCVC_PREPARATION_SIDECAR_NAME
        preparation.write_bytes(preparation.read_bytes() + b"\n")
        match = "canonical JSON"
    else:
        (output / "canvas_000.jpg").write_bytes(b"tampered-asset")
        match = "asset bytes"

    with pytest.raises(MageCodecCacheV2Error, match=match):
        _rebuild_verified_hit(
            tmp_path=tmp_path,
            manifest=manifest,
            source=source,
            output=output,
        )

    assert not sidecar.exists()
    assert not tuple(output.glob(f".{sidecar.name}.*.tmp"))


def test_v2_truncated_existing_sidecar_fails_closed_without_overwrite(tmp_path: Path) -> None:
    manifest, source, output = _build(tmp_path)
    sidecar = output / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME
    truncated = b'{"entry_version":"mage-codec-cache-entry-v2"'
    sidecar.write_bytes(truncated)

    with pytest.raises(MageCodecCacheV2Error, match="differs from expected entry"):
        _rebuild_verified_hit(
            tmp_path=tmp_path,
            manifest=manifest,
            source=source,
            output=output,
        )

    assert sidecar.read_bytes() == truncated
    assert not tuple(output.glob(f".{sidecar.name}.*.tmp"))


def test_v2_mismatched_valid_sidecar_fails_closed_without_overwrite(tmp_path: Path) -> None:
    manifest, source, output = _build(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    _other_manifest, _other_source, other_output = _build(other_root)
    mismatched = (other_output / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME).read_bytes()
    sidecar = output / MAGE_CODEC_CACHE_ENTRY_V2_SIDECAR_NAME
    sidecar.write_bytes(mismatched)

    with pytest.raises(MageCodecCacheV2Error, match="differs from expected entry"):
        _rebuild_verified_hit(
            tmp_path=tmp_path,
            manifest=manifest,
            source=source,
            output=output,
        )

    assert sidecar.read_bytes() == mismatched
    assert not tuple(output.glob(f".{sidecar.name}.*.tmp"))
