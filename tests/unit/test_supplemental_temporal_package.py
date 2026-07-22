from __future__ import annotations

import struct
import zlib
from dataclasses import replace
from hashlib import sha256

import pytest

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.qa_pipeline.supplemental import (
    DeterministicSupplementalQaDenseConsumer,
    SupplementalQaDenseStatus,
    _validate_png,
)
from robata.sampling.grid import SelectionStatus
from robata.sampling.supplemental import (
    SUPPLEMENTAL_DEDUPE_POLICY_VERSION,
    SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION,
    ExplicitTargetPackageMaterializer,
    FrozenSupplementalTargetPlan,
    MaterializedArtifactManifest,
    MaterializedFrameArtifactFact,
    MaterializedSupplementalPackage,
    ProviderNeutralSupplementalPackage,
    RegisteredMediaQualitySourceBinding,
    SupplementalPackageMaterializationError,
    build_frozen_supplemental_target_plan,
    media_quality_source_binding_projection,
)
from tests.unit.test_sampling_materializer import (
    _alignment,
    _digest,
    _frame_index,
    _lineage,
    _sampling_plan,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _artifact_contents(camera_id: CameraId, source_order: int) -> bytes:
    pixel = bytes(((CAMERA_IDS.index(camera_id) * 32 + source_order) % 256,))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00" + pixel))
        + _png_chunk(b"IEND", b"")
    )


def _artifact_resolver(
    camera_id: CameraId,
    frame,
) -> MaterializedFrameArtifactFact:
    contents = _artifact_contents(camera_id, frame.source_order)
    return MaterializedFrameArtifactFact(
        artifact=MaterializedArtifactManifest(
            artifact_id=f"supplemental-artifact-{camera_id.value}-{frame.source_order}",
            uri=f"memory://supplemental/{camera_id.value}/{frame.source_order}.png",
            sha256=sha256(contents).hexdigest(),
            bytes=len(contents),
            media_type="image/png",
        ),
        width=1,
        height=1,
        quality_flags=("FIXTURE_BYTES_VERIFIED",),
    )


def _artifact_bytes_resolver(artifact: MaterializedArtifactManifest) -> bytes:
    for camera_id in CAMERA_IDS:
        for source_order in range(3):
            contents = _artifact_contents(camera_id, source_order)
            if sha256(contents).hexdigest() == artifact.sha256:
                return contents
    raise ValueError("unknown supplemental artifact")


def _schema_ref() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/media-quality-report",
        version="1.0.0",
        artifact_id="0056daa0-49fd-038e-f075-293244a6677a",
        sha256="3e275f18bb36986b94ac96d977b0614aa15352565efb365a623d893b0123b798",
    )


def _source_binding(
    source_target_plan_digest: str | None = None,
    *,
    report_schema_ref: SchemaRef | None = None,
) -> RegisteredMediaQualitySourceBinding:
    values = {
        "report_schema_ref": report_schema_ref or _schema_ref(),
        "report_semantic_sha256": _digest("media-quality-report"),
        "supplemental_target_plan_semantic_sha256": (
            source_target_plan_digest or _digest("neighbor-target-plan")
        ),
        "media_quality_binding_semantic_sha256": _digest("media-quality-binding"),
        "source_content_sha256": _digest("source-content"),
        "camera_mapping_semantic_sha256": _digest("mapping"),
        "alignment_semantic_sha256": _digest("alignment"),
        "projection_version": "media-quality-source-binding-semantic-v1",
    }
    draft = RegisteredMediaQualitySourceBinding.model_construct(
        semantic_sha256="0" * 64,
        **values,
    )
    return RegisteredMediaQualitySourceBinding(
        **values,
        semantic_sha256=semantic_sha256(media_quality_source_binding_projection(draft)),
    )


def _plan(
    *,
    source_target_plan_digest: str | None = None,
    extra_missing_target: bool = False,
) -> FrozenSupplementalTargetPlan:

    targets = [(camera_id, 250_000_000) for camera_id in reversed(CAMERA_IDS)]
    targets.append((CameraId.CAM_01, 260_000_000))
    if extra_missing_target:
        targets.append((CameraId.CAM_06, 1_490_000_000))
    return build_frozen_supplemental_target_plan(
        source_binding=_source_binding(source_target_plan_digest),
        effective_interval=NanosecondInterval(start_ns=0, end_ns=1_500_000_000),
        targets=targets,
        selection_tolerance_ns=300_000_000,
        tie_break_policy_version=SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION,
        dedupe_policy_version=SUPPLEMENTAL_DEDUPE_POLICY_VERSION,
        target_policy_version="local-neighbor-targets-v1",
    )


def _materialized(
    *,
    alias: int = 0,
    plan: FrozenSupplementalTargetPlan | None = None,
    extra_missing_target: bool = False,
) -> MaterializedSupplementalPackage:
    resolved_plan = plan or _plan(extra_missing_target=extra_missing_target)
    lineage = _lineage(_sampling_plan())
    alignment = _alignment(alias)
    frame_index = _frame_index(alignment, lineage, alias=alias, empty_camera=None)
    return ExplicitTargetPackageMaterializer().materialize(
        plan=resolved_plan,
        frame_index=frame_index,
        artifact_resolver=_artifact_resolver,
        created_at=f"2026-07-{19 + alias:02d}T01:00:00Z",
    )


def test_explicit_targets_are_frozen_before_package_identity() -> None:
    plan = _plan()

    assert tuple(target.ordinal for target in plan.targets) == tuple(range(7))
    assert tuple((target.camera_id, target.target_ns) for target in plan.targets[:2]) == (
        (CameraId.CAM_01, 250_000_000),
        (CameraId.CAM_02, 250_000_000),
    )
    changed = _plan(source_target_plan_digest=_digest("different-neighbor-target-plan"))
    assert changed.semantic_sha256 != plan.semantic_sha256
    assert changed.plan_id != plan.plan_id


def test_frozen_plan_rejects_empty_targets_and_foreign_report_schema() -> None:
    kwargs = {
        "source_binding": _source_binding(),
        "effective_interval": NanosecondInterval(start_ns=0, end_ns=1_500_000_000),
        "targets": (),
        "selection_tolerance_ns": 300_000_000,
        "tie_break_policy_version": SUPPLEMENTAL_TIE_BREAK_POLICY_VERSION,
        "dedupe_policy_version": SUPPLEMENTAL_DEDUPE_POLICY_VERSION,
        "target_policy_version": "local-neighbor-targets-v1",
    }
    with pytest.raises(ValueError, match="at least one target"):
        build_frozen_supplemental_target_plan(**kwargs)

    kwargs["targets"] = ((CameraId.CAM_01, 250_000_000),)
    kwargs["source_binding"] = _source_binding(
        report_schema_ref=_schema_ref().model_copy(
            update={"schema_id": "https://schemas.robata.dev/event-hypothesis"}
        )
    )
    with pytest.raises(ValueError, match=r"media-quality-report@1\.0\.0"):
        build_frozen_supplemental_target_plan(**kwargs)


def test_materializer_records_selected_and_deduplicated_targets() -> None:
    materialized = _materialized()
    outcomes = materialized.package.outcomes

    cam_01 = tuple(item for item in outcomes if item.target.camera_id is CameraId.CAM_01)
    assert tuple(item.status for item in cam_01) == (
        SelectionStatus.SELECTED,
        SelectionStatus.DEDUPLICATED_FRAME,
    )
    assert cam_01[1].reused_selected_target_ordinal == cam_01[0].target.ordinal
    assert cam_01[0].selected_artifact is not None
    assert cam_01[1].selected_artifact is None
    assert materialized.package.selected_artifact_count == 6
    assert materialized.package.production_eligible is False


def test_source_locator_dedupe_is_scoped_to_its_camera() -> None:
    plan = _plan()
    lineage = _lineage(_sampling_plan())
    alignment = _alignment()
    frame_index = _frame_index(alignment, lineage, empty_camera=None)
    cameras = {}
    for camera_id, camera_index in frame_index.cameras.items():
        cameras[camera_id] = camera_index.model_copy(
            update={
                "frames": tuple(
                    frame.model_copy(
                        update={"source_locator": {"message_offset": frame.source_order}}
                    )
                    for frame in camera_index.frames
                )
            }
        )
    shared_locator_index = frame_index.model_copy(update={"cameras": SixCameraMap(cameras)})

    materialized = ExplicitTargetPackageMaterializer().materialize(
        plan=plan,
        frame_index=shared_locator_index,
        artifact_resolver=_artifact_resolver,
        created_at="2026-07-22T01:00:00Z",
    )

    assert materialized.package.selected_artifact_count == 6
    assert (
        tuple(
            item.target.camera_id
            for item in materialized.package.outcomes
            if item.status is SelectionStatus.SELECTED
        )
        == CAMERA_IDS
    )


def test_package_rejects_duplicate_selected_source_and_wrong_dedupe_winner() -> None:
    package = _materialized().package
    outcomes = list(package.outcomes)
    cam_01_indexes = tuple(
        index for index, item in enumerate(outcomes) if item.target.camera_id is CameraId.CAM_01
    )
    first_index, second_index = cam_01_indexes
    first, second = outcomes[first_index], outcomes[second_index]
    assert first.selected_artifact is not None

    outcomes[second_index] = second.model_copy(
        update={
            "status": SelectionStatus.SELECTED,
            "selected_artifact": first.selected_artifact,
            "reused_selected_target_ordinal": None,
        }
    )
    duplicate = package.model_copy(
        update={
            "outcomes": tuple(outcomes),
            "selected_artifact_count": package.selected_artifact_count + 1,
        }
    )
    with pytest.raises(ValueError, match="exactly one selected target"):
        ProviderNeutralSupplementalPackage.model_validate(
            duplicate.model_dump(mode="python"), strict=True
        )

    outcomes = list(package.outcomes)
    outcomes[first_index] = first.model_copy(
        update={
            "status": SelectionStatus.DEDUPLICATED_FRAME,
            "selected_artifact": None,
            "reused_selected_target_ordinal": second.target.ordinal,
        }
    )
    outcomes[second_index] = second.model_copy(
        update={
            "status": SelectionStatus.SELECTED,
            "selected_artifact": first.selected_artifact,
            "reused_selected_target_ordinal": None,
        }
    )
    wrong_winner = package.model_copy(update={"outcomes": tuple(outcomes)})
    with pytest.raises(ValueError, match="dedupe winner policy"):
        ProviderNeutralSupplementalPackage.model_validate(
            wrong_winner.model_dump(mode="python"), strict=True
        )


def test_materializer_rejects_unsupported_selection_policy_names() -> None:
    with pytest.raises(SupplementalPackageMaterializationError, match="tie-break policy"):
        _materialized(plan=_plan().model_copy(update={"tie_break_policy_version": "unknown-v1"}))
    with pytest.raises(SupplementalPackageMaterializationError, match="dedupe policy"):
        _materialized(plan=_plan().model_copy(update={"dedupe_policy_version": "unknown-v1"}))


def test_semantic_identity_excludes_row_ids_uris_and_wall_clock() -> None:
    first = _materialized(alias=0)
    second = _materialized(alias=1)

    assert first.package.package_id == second.package.package_id
    assert first.package.semantic_content_sha256 == second.package.semantic_content_sha256
    assert first.manifest_sha256 != second.manifest_sha256


def test_materializer_rejects_foreign_source_lineage() -> None:
    plan = _plan().model_copy(update={"source_content_sha256": _digest("foreign-source")})
    alignment = _alignment()
    lineage = _lineage(_sampling_plan())
    frame_index = _frame_index(alignment, lineage, empty_camera=None)

    with pytest.raises(
        SupplementalPackageMaterializationError,
        match="does not bind the frame index lineage",
    ):
        ExplicitTargetPackageMaterializer().materialize(
            plan=plan,
            frame_index=frame_index,
            artifact_resolver=_artifact_resolver,
            created_at="2026-07-19T01:00:00Z",
        )


def test_qa_dense_mock_consumes_exact_package_artifacts_without_semantic_claims() -> None:
    materialized = _materialized()
    consumer = DeterministicSupplementalQaDenseConsumer()
    input_plan = consumer.prepare(materialized)
    result = consumer.consume(
        materialized, input_plan, artifact_bytes_resolver=_artifact_bytes_resolver
    )

    assert input_plan.package_id == materialized.package.package_id
    assert input_plan.package_manifest_sha256 == materialized.manifest_sha256
    assert result.input_plan_semantic_sha256 == input_plan.semantic_sha256
    assert result.package_semantic_content_sha256 == materialized.package.semantic_content_sha256
    assert result.status is SupplementalQaDenseStatus.COMPLETE
    assert all(item.effective_artifact_sha256 is not None for item in result.consumptions)
    assert "occlusion" not in result.model_dump_json().lower()
    assert "blur" not in result.model_dump_json().lower()


def test_qa_dense_mock_preserves_missing_target_as_incomplete() -> None:
    materialized = _materialized(extra_missing_target=True)
    consumer = DeterministicSupplementalQaDenseConsumer()
    result = consumer.consume(
        materialized,
        consumer.prepare(materialized),
        artifact_bytes_resolver=_artifact_bytes_resolver,
    )

    assert result.status is SupplementalQaDenseStatus.INCOMPLETE
    assert result.consumptions[-1].package_status is SelectionStatus.NO_FRAME_WITHIN_TOLERANCE
    assert result.consumptions[-1].effective_artifact_sha256 is None


def test_exact_manifest_and_semantic_tampering_fail_closed() -> None:
    materialized = _materialized()
    with pytest.raises(ValueError, match="manifest_sha256"):
        replace(materialized, manifest_sha256=_digest("tampered-manifest"))

    payload = materialized.package.model_dump(mode="python")
    payload["selected_artifact_count"] = 1
    with pytest.raises(ValueError, match="selected_artifact_count"):
        ProviderNeutralSupplementalPackage.model_validate(payload, strict=True)


def test_png_decode_policy_rejects_missing_pixels_crc_and_dimensions() -> None:
    valid = _artifact_contents(CameraId.CAM_01, 0)
    signature = bytes.fromhex("89504e470d0a1a0a")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    no_pixels = signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")

    with pytest.raises(ValueError, match="incomplete"):
        _validate_png(no_pixels, expected_width=1, expected_height=1)
    with pytest.raises(ValueError, match="dimensions"):
        _validate_png(valid, expected_width=2, expected_height=1)

    bad_crc = bytearray(valid)
    bad_crc[-1] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        _validate_png(bytes(bad_crc), expected_width=1, expected_height=1)


def test_png_decode_policy_rejects_corrupt_compressed_pixels() -> None:
    signature = bytes.fromhex("89504e470d0a1a0a")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    corrupt = (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"not-zlib")
        + _png_chunk(b"IEND", b"")
    )

    with pytest.raises(ValueError, match="pixel stream"):
        _validate_png(corrupt, expected_width=1, expected_height=1)
