from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID

import pytest

from robata.benchmark.qwen_r12_request_corpus import (
    QWEN_R12_20260806_EXPECTED,
    QWEN_R12_20260806_MANIFEST_SEMANTIC_SHA256,
    BatchCompatibilityProjection,
    QwenRequestCorpusError,
    QwenRequestCorpusExpected,
    batch_compatibility_projection,
    load_qwen_request_corpus,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.inference.adapter import JsonSchemaRef, PackageInput, VisionInferenceRequest
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.models import VisionTask
from robata.inference.orchestrator import InferenceIntent

_NOW = "2026-08-09T12:00:00Z"
_PROVIDER = "local-huggingface"
_MODEL = "Qwen3-VL-4B-Instruct"
_MODEL_VERSION = "local"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REAL_R12 = Path(
    r"D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r12-20260806"
    r"\inference-evidence.sqlite3"
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _build_intent(
    root: Path,
    *,
    seed: int,
    task: VisionTask,
    timeout_ms: int = 10_000,
    image_count: int = 6,
) -> tuple[InferenceIntent, tuple[Path, ...]]:
    root.mkdir(parents=True, exist_ok=True)
    base = seed * 1_000
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    schema = JsonSchemaRef(
        schema_id="https://schemas.example.test/provider-claim",
        version="1.0.0",
        artifact_id=f"provider-claim-{seed}",
        sha256=_digest(base + 1),
    )
    cameras: list[CatalogCamera] = []
    rendered_items: list[RenderedProviderItem] = []
    paths: list[Path] = []
    for ordinal, camera_id in enumerate(CAMERA_IDS):
        payload = _PNG_SIGNATURE + f"fixture-{seed}-{ordinal}".encode("ascii")
        path = root / f"fixture-{seed}-{ordinal}.png"
        path.write_bytes(payload)
        paths.append(path)
        frame = CatalogFrame(
            frame_id=_uuid(base + 100 + ordinal),
            ordinal=0,
            aligned_timestamp_ns=1_000_000_000 + ordinal,
            source_timestamp_ns=1_700_000_000_000_000_000 + ordinal,
            source_artifact_uri=path.resolve().as_uri(),
            source_artifact_sha256=exact_bytes_sha256(payload),
            source_artifact_bytes=len(payload),
            media_type="image/png",
            encoding="png",
            width=320,
            height=260,
        )
        cameras.append(CatalogCamera(camera_id=camera_id, ordinal=ordinal, frames=(frame,)))
        rendered_items.append(
            RenderedProviderItem(
                provider_item_ordinal=ordinal,
                package_id=_uuid(base + 300),
                package_ordinal=0,
                camera_id=camera_id,
                camera_ordinal=ordinal,
                frame_id=frame.frame_id,
                frame_ordinal=0,
                aligned_timestamp_ns=frame.aligned_timestamp_ns,
                source_timestamp_ns=frame.source_timestamp_ns,
                source_artifact_sha256=frame.source_artifact_sha256,
                artifact=RenderedArtifact(
                    artifact_id=_uuid(base + 400 + ordinal),
                    uri=path.resolve().as_uri(),
                    sha256=exact_bytes_sha256(payload),
                    byte_count=len(payload),
                    media_type="image/png",
                    encoding="png",
                    width=320,
                    height=260,
                ),
                transform=FrameTransform.create(
                    operation=TransformOperation.NONE,
                    policy_version="render-v1",
                ),
            )
        )
    package = CatalogPackage(
        package_id=_uuid(base + 300),
        ordinal=0,
        semantic_content_sha256=_digest(base + 301),
        manifest_bytes_sha256=_digest(base + 302),
        cameras=tuple(cameras),
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(base + 303),
        task=task,
        packages=(package,),
        created_at=_NOW,
    )
    plan = planner.build(
        input_plan_id=_uuid(base + 304),
        created_at=_NOW,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider=_PROVIDER,
            model_name=_MODEL,
            model_version=_MODEL_VERSION,
            adapter_version="local-hf-loopback-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=_uuid(base + 500),
            capability_snapshot_sha256=_digest(base + 501),
        ),
        rendered_items=tuple(rendered_items),
        prompt_output=PromptOutputContract(
            prompt_version=f"local-{task.value.lower()}-prompt-v1",
            prompt_sha256=_digest(base + 503),
            rendered_message_sha256=_digest(base + 504),
            provider_response_schema_sha256=schema.sha256,
            enriched_domain_schema_sha256=_digest(base + 505),
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=image_count,
            max_pixels_per_image=320 * 260,
            max_payload_bytes_per_request=1_000_000,
            max_input_tokens_per_request=1_000,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=image_count,
                measured_input_tokens=13,
            ),
            *(
                (
                    CallPartSpec(
                        start_item_ordinal=image_count,
                        end_item_ordinal_exclusive=len(CAMERA_IDS),
                        measured_input_tokens=3,
                    ),
                )
                if image_count < len(CAMERA_IDS)
                else ()
            ),
        ),
        idempotency_policy_version="local-idempotency-v1",
        reduction_policy="single-part",
        reduction_policy_version="local-reduction-v1",
    )
    part = plan.call_plan.parts[0]
    request = VisionInferenceRequest(
        schema_version="1.0",
        logical_invocation_id=_uuid(base + 600),
        request_id=_uuid(base + 601),
        idempotency_key=f"fixture-request-{seed}",
        provider=_PROVIDER,
        model_name=_MODEL,
        model_version=_MODEL_VERSION,
        package_set_id=_uuid(base + 602),
        package_inputs=(
            PackageInput(
                package_id=package.package_id,
                package_semantic_content_sha256=package.semantic_content_sha256,
                package_manifest_sha256=package.manifest_bytes_sha256,
                role="primary",
                ordinal=0,
            ),
        ),
        package_input_set_sha256=_digest(base + 603),
        task=task,
        prompt_version=plan.prompt_output.prompt_version,
        prompt_artifact_id=f"prompt-{seed}",
        prompt_sha256=plan.prompt_output.prompt_sha256,
        rendered_input_digest=part.item_manifest_sha256,
        input_plan_id=plan.input_plan_id,
        input_plan_semantic_sha256=plan.semantic_sha256,
        input_plan_part_ordinal=part.ordinal,
        input_plan_part_count=part.part_count,
        input_plan_part_semantic_sha256=part.part_semantic_sha256,
        input_plan=plan,
        output_schema=schema,
        capability_snapshot_id=plan.target.capability_snapshot_id,
        capability_snapshot_digest=plan.target.capability_snapshot_sha256,
        model_policy_version="local-model-policy-v1",
        generation_config={"max_new_tokens": 12, "temperature": 0},
        provider_idempotency_key=part.idempotency_key,
        timeout_ms=timeout_ms,
        metadata={},
    )
    intent = InferenceIntent(
        schema_version="1.0",
        inference_id=_uuid(base + 700),
        logical_invocation_id=request.logical_invocation_id,
        request_id=request.request_id,
        idempotency_key=request.idempotency_key,
        task=request.task,
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        adapter_version=plan.target.adapter_version,
        mcap_id=_uuid(base + 701),
        camera_mapping_run_id=_uuid(base + 702),
        alignment_id=_uuid(base + 703),
        start_ns=seed * 1_000_000_000,
        end_ns=(seed + 1) * 1_000_000_000,
        input_config={"input_images": image_count},
        sampling_config={"policy": "fixture-v1"},
        input_plan_id=request.input_plan_id,
        input_plan_semantic_sha256=request.input_plan_semantic_sha256,
        input_plan_part_ordinal=request.input_plan_part_ordinal,
        input_plan_part_count=request.input_plan_part_count,
        input_plan_part_semantic_sha256=request.input_plan_part_semantic_sha256,
        attempt=1,
        retry_count=0,
        shadow=False,
        request=request,
        queued_at=_NOW,
        created_at=_NOW,
    )
    return intent, tuple(paths)


def _write_database(path: Path, intents: tuple[InferenceIntent, ...]) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE inference_intents (
                inference_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                payload_json BLOB NOT NULL,
                payload_sha256 TEXT NOT NULL
            )
            """
        )
        for intent in intents:
            payload = canonical_json_bytes(intent)
            connection.execute(
                """
                INSERT INTO inference_intents (
                    inference_id, request_id, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    intent.inference_id,
                    intent.request_id,
                    sqlite3.Binary(payload),
                    exact_bytes_sha256(payload),
                ),
            )
    return exact_bytes_sha256(path.read_bytes())


def _expected(
    database_sha256: str,
    *,
    task_counts: tuple[tuple[VisionTask, int], ...],
    intent_count: int,
    reference_count: int,
    unique_image_count: int,
    selected_images_per_intent: int = 6,
) -> QwenRequestCorpusExpected:
    return QwenRequestCorpusExpected(
        database_sha256=database_sha256,
        intent_count=intent_count,
        task_counts=task_counts,
        reference_count=reference_count,
        unique_image_count=unique_image_count,
        selected_images_per_intent=selected_images_per_intent,
    )


def test_loader_preserves_order_verifies_files_and_freezes_manifest(tmp_path: Path) -> None:
    coarse, coarse_paths = _build_intent(tmp_path / "images", seed=1, task=VisionTask.QA_COARSE)
    dense, dense_paths = _build_intent(tmp_path / "images", seed=2, task=VisionTask.QA_DENSE)
    database = tmp_path / "evidence.sqlite3"
    database_sha256 = _write_database(database, (coarse, dense))
    expected = _expected(
        database_sha256,
        task_counts=((VisionTask.QA_COARSE, 1), (VisionTask.QA_DENSE, 1)),
        intent_count=2,
        reference_count=12,
        unique_image_count=12,
    )

    corpus = load_qwen_request_corpus(database, expected=expected)
    replay = load_qwen_request_corpus(database, expected=expected)

    assert tuple(case.request.task for case in corpus.cases) == (
        VisionTask.QA_COARSE,
        VisionTask.QA_DENSE,
    )
    assert tuple(bucket.task for bucket in corpus.task_buckets) == (
        VisionTask.QA_COARSE,
        VisionTask.QA_DENSE,
    )
    assert corpus.reference_count == 12
    assert corpus.unique_image_count == 12
    assert corpus.database_sha256 == database_sha256
    assert corpus.semantic_sha256 == semantic_sha256(corpus.manifest_projection())
    assert corpus.canonical_manifest_bytes() == canonical_json_bytes(corpus.manifest_projection())
    assert replay.semantic_sha256 == corpus.semantic_sha256
    assert tuple(image.path for image in corpus.cases[0].selected_images) == coarse_paths
    assert tuple(image.path for image in corpus.cases[1].selected_images) == dense_paths
    assert all(image.path.is_absolute() for case in corpus.cases for image in case.selected_images)
    with pytest.raises(FrozenInstanceError):
        corpus.reference_count = 0  # type: ignore[misc]


def test_batch_projection_matches_orchestrator_dimensions_for_representative_request(
    tmp_path: Path,
) -> None:
    intent, _ = _build_intent(tmp_path / "images", seed=3, task=VisionTask.QA_COARSE)

    observed = batch_compatibility_projection(intent.request)

    assert observed == BatchCompatibilityProjection(
        provider=_PROVIDER,
        model_name=_MODEL,
        model_version=_MODEL_VERSION,
        task=VisionTask.QA_COARSE,
        model_policy_version="local-model-policy-v1",
        output_schema_sha256=intent.request.output_schema.sha256,
        timeout_ms=10_000,
        input_shape=(("image/png", "png", 320, 260),) * 6,
    )


def test_loader_rejects_payload_digest_drift(tmp_path: Path) -> None:
    intent, _ = _build_intent(tmp_path / "images", seed=4, task=VisionTask.QA_COARSE)
    database = tmp_path / "evidence.sqlite3"
    _write_database(database, (intent,))
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE inference_intents SET payload_sha256 = ?", ("0" * 64,))
    database_sha256 = exact_bytes_sha256(database.read_bytes())

    with pytest.raises(QwenRequestCorpusError, match="payload_sha256"):
        load_qwen_request_corpus(
            database,
            expected=_expected(
                database_sha256,
                task_counts=((VisionTask.QA_COARSE, 1),),
                intent_count=1,
                reference_count=6,
                unique_image_count=6,
            ),
        )


@pytest.mark.parametrize("tamper", ["missing", "not-png", "digest"])
def test_loader_rejects_missing_or_tampered_images(tmp_path: Path, tamper: str) -> None:
    intent, paths = _build_intent(tmp_path / "images", seed=5, task=VisionTask.QA_COARSE)
    database = tmp_path / "evidence.sqlite3"
    database_sha256 = _write_database(database, (intent,))
    if tamper == "missing":
        paths[0].unlink()
    elif tamper == "not-png":
        paths[0].write_bytes(b"not a png")
    else:
        paths[0].write_bytes(_PNG_SIGNATURE + b"different exact bytes")

    with pytest.raises(QwenRequestCorpusError):
        load_qwen_request_corpus(
            database,
            expected=_expected(
                database_sha256,
                task_counts=((VisionTask.QA_COARSE, 1),),
                intent_count=1,
                reference_count=6,
                unique_image_count=6,
            ),
        )


def test_loader_rejects_duplicate_request_identity(tmp_path: Path) -> None:
    intent, _ = _build_intent(tmp_path / "images", seed=6, task=VisionTask.QA_COARSE)
    database = tmp_path / "evidence.sqlite3"
    database_sha256 = _write_database(database, (intent, intent))

    with pytest.raises(QwenRequestCorpusError, match="duplicate inference identity"):
        load_qwen_request_corpus(
            database,
            expected=_expected(
                database_sha256,
                task_counts=((VisionTask.QA_COARSE, 2),),
                intent_count=2,
                reference_count=12,
                unique_image_count=6,
            ),
        )


def test_loader_rejects_six_image_profile_mismatch(tmp_path: Path) -> None:
    intent, _ = _build_intent(tmp_path / "images", seed=7, task=VisionTask.QA_COARSE, image_count=5)
    database = tmp_path / "evidence.sqlite3"
    database_sha256 = _write_database(database, (intent,))

    with pytest.raises(QwenRequestCorpusError, match="selected 5 images; expected 6"):
        load_qwen_request_corpus(
            database,
            expected=_expected(
                database_sha256,
                task_counts=((VisionTask.QA_COARSE, 1),),
                intent_count=1,
                reference_count=5,
                unique_image_count=5,
            ),
        )


def test_loader_rejects_compatibility_drift_within_task(tmp_path: Path) -> None:
    first, _ = _build_intent(tmp_path / "images", seed=8, task=VisionTask.QA_COARSE)
    second, _ = _build_intent(
        tmp_path / "images", seed=9, task=VisionTask.QA_COARSE, timeout_ms=20_000
    )
    database = tmp_path / "evidence.sqlite3"
    database_sha256 = _write_database(database, (first, second))

    with pytest.raises(QwenRequestCorpusError, match="batch compatibility drift"):
        load_qwen_request_corpus(
            database,
            expected=_expected(
                database_sha256,
                task_counts=((VisionTask.QA_COARSE, 2),),
                intent_count=2,
                reference_count=12,
                unique_image_count=12,
            ),
        )


def test_loader_rejects_database_exact_sha_drift(tmp_path: Path) -> None:
    intent, _ = _build_intent(tmp_path / "images", seed=10, task=VisionTask.QA_COARSE)
    database = tmp_path / "evidence.sqlite3"
    _write_database(database, (intent,))

    with pytest.raises(QwenRequestCorpusError, match="database exact SHA-256"):
        load_qwen_request_corpus(
            database,
            expected=QwenRequestCorpusExpected(database_sha256="0" * 64),
        )


@pytest.mark.skipif(not _REAL_R12.is_file(), reason="real local r12 corpus is not available")
def test_real_r12_corpus_is_exact_and_deterministic() -> None:
    corpus = load_qwen_request_corpus(_REAL_R12, expected=QWEN_R12_20260806_EXPECTED)

    assert len(corpus.cases) == 51
    assert corpus.reference_count == 306
    assert corpus.unique_image_count == 276
    assert tuple((bucket.task, bucket.case_count) for bucket in corpus.task_buckets) == (
        (VisionTask.QA_COARSE, 41),
        (VisionTask.QA_DENSE, 10),
    )
    assert len({case.intent.inference_id for case in corpus.cases}) == 51
    assert len({case.request.request_id for case in corpus.cases}) == 51
    assert corpus.semantic_sha256 == QWEN_R12_20260806_MANIFEST_SEMANTIC_SHA256
    assert corpus.semantic_sha256 == semantic_sha256(corpus.manifest_projection())
