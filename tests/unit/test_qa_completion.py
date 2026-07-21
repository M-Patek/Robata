from __future__ import annotations

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

import pytest

from robata.application.canonical.logical_nodes import (
    canonical_coarse_qa_logical_node,
    canonical_qa_completion_logical_node,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.pipeline import CameraQAStatus
from robata.inference.enrichment import (
    EnrichedEvidenceReference,
    EnrichedProviderClaim,
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderObservation,
)
from robata.qa_pipeline.coarse import (
    CameraCoarseResult,
    CoarseQAOutputRef,
    CoarseQAResult,
    CoarseQAStatus,
    coarse_qa_semantic_projection,
)
from robata.qa_pipeline.completion import (
    DenseQAOutcome,
    QACompletionProjectionError,
    QACompletionProjector,
    QACompletionResult,
    QACompletionStatus,
)
from robata.qa_pipeline.dense import (
    CameraDenseResult,
    DenseQAInputPlanRef,
    DenseQAOutputRef,
    DenseQAPackageRef,
    DenseQAPlanningPolicy,
    DenseQAProjectionError,
    DenseQAProjector,
    DenseQAStatus,
    DenseQAUnitEvidence,
    DenseQAWorkManifest,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:test:qa-completion:{value}"))


def _source_output(package_ordinal: int) -> CoarseQAOutputRef:
    return CoarseQAOutputRef(
        artifact_id=_id(f"output:{package_ordinal}"),
        semantic_sha256=_digest(f"output:{package_ordinal}"),
        enrichment_logical_key=(
            f"orchestrator-enrichment:{_digest(f'enrichment:{package_ordinal}')}"
        ),
        inference_id=_id(f"inference:{package_ordinal}"),
    )


def _coarse_result(
    statuses: dict[tuple[int, CameraId], ProviderObservation] | None = None,
) -> CoarseQAResult:
    selected = statuses or {}
    package_ids = tuple(_id(f"package:{ordinal}") for ordinal in range(2))
    outputs_by_package = {ordinal: _source_output(ordinal) for ordinal in range(2)}
    source_outputs = tuple(sorted(outputs_by_package.values(), key=lambda item: item.artifact_id))
    results: list[CameraCoarseResult] = []
    for package_ordinal, package_id in enumerate(package_ids):
        start_ns = package_ordinal * 100
        end_ns = start_ns + 100
        for camera_ordinal, camera_id in enumerate(CAMERA_IDS):
            observation = selected.get((package_ordinal, camera_id), ProviderObservation.GOOD)
            local_status = {
                ProviderObservation.GOOD: CameraQAStatus.GOOD,
                ProviderObservation.DEGRADED: CameraQAStatus.DEGRADED,
                ProviderObservation.UNUSABLE: CameraQAStatus.UNUSABLE,
                ProviderObservation.UNKNOWN: CameraQAStatus.UNKNOWN,
            }[observation]
            coordinate = f"{package_ordinal}:{camera_id.value}"
            evidence = EnrichedEvidenceReference(
                correlation_token=f"ref:{_digest(f'token:{coordinate}')}",
                provider_item_ordinal=package_ordinal * 6 + camera_ordinal,
                package_id=package_id,
                package_ordinal=package_ordinal,
                package_semantic_content_sha256=_digest(f"package-semantic:{package_ordinal}"),
                package_manifest_sha256=_digest(f"package-manifest:{package_ordinal}"),
                camera_id=camera_id,
                camera_ordinal=camera_ordinal,
                frame_id=_id(f"frame:{coordinate}"),
                frame_ordinal=0,
                aligned_timestamp_ns=start_ns + 10 + camera_ordinal,
                source_timestamp_ns=1_000 + start_ns + camera_ordinal,
                source_artifact_uri=f"object://qa-completion/{coordinate}",
                source_artifact_sha256=_digest(f"frame:{coordinate}"),
            )
            claim = EnrichedProviderClaim(
                claim_id=_id(f"claim:{coordinate}"),
                claim_ordinal=package_ordinal * 6 + camera_ordinal,
                kind=ProviderClaimKind.QA_OBSERVATION,
                package_id=package_id,
                package_ordinal=package_ordinal,
                camera_id=camera_id,
                interval=ProviderClaimInterval(start_ns=start_ns, end_ns=end_ns),
                label="local-screen",
                observation=observation,
                evidence=(evidence,),
                model_reported_confidence=None,
                conflict_codes=(),
            )
            results.append(
                CameraCoarseResult(
                    package_id=package_id,
                    package_ordinal=package_ordinal,
                    camera_id=camera_id,
                    local_status=local_status,
                    source_output=outputs_by_package[package_ordinal],
                    claim=claim,
                )
            )

    status_set = {item.local_status for item in results}
    if CameraQAStatus.UNKNOWN in status_set:
        local_status = CoarseQAStatus.INCOMPLETE
    elif status_set & {CameraQAStatus.DEGRADED, CameraQAStatus.UNUSABLE}:
        local_status = CoarseQAStatus.REQUIRES_DENSE
    else:
        local_status = CoarseQAStatus.COMPLETE
    return CoarseQAResult(
        package_set_id="qa-completion-package-set-v1",
        mcap_id=_id("mcap"),
        input_plan_id=_id("input-plan"),
        input_plan_semantic_sha256=_digest("input-plan"),
        package_ids=package_ids,
        source_outputs=source_outputs,
        package_camera_results=tuple(results),
        local_status=local_status,
        complete=local_status is CoarseQAStatus.COMPLETE,
        requires_dense=local_status is CoarseQAStatus.REQUIRES_DENSE,
        policy_version="local-coarse-qa-projector-v1",
        production_eligible=False,
    )


def _dense_evidence(
    manifest: DenseQAWorkManifest,
    statuses: dict[tuple[int, CameraId], ProviderObservation] | None = None,
) -> tuple[DenseQAUnitEvidence, ...]:
    selected = statuses or {}
    bundles: list[DenseQAUnitEvidence] = []
    for unit_ordinal, unit in enumerate(manifest.units):
        package_id = _id(f"dense-package:{unit.unit_id}")
        package_semantic = _digest(f"dense-package-semantic:{unit.unit_id}")
        package_manifest = _digest(f"dense-package-manifest:{unit.unit_id}")
        package = DenseQAPackageRef(
            package_id=package_id,
            ordinal=0,
            interval=unit.effective_interval,
            semantic_content_sha256=package_semantic,
            manifest_sha256=package_manifest,
        )
        input_plan = DenseQAInputPlanRef(
            input_plan_id=_id(f"dense-input-plan:{unit.unit_id}"),
            semantic_sha256=_digest(f"dense-input-plan:{unit.unit_id}"),
            package_ids=(package_id,),
        )
        output = DenseQAOutputRef(
            artifact_id=_id(f"dense-output:{unit.unit_id}"),
            semantic_sha256=_digest(f"dense-output:{unit.unit_id}"),
            enrichment_logical_key=(
                f"orchestrator-enrichment:{_digest(f'dense-enrichment:{unit.unit_id}')}"
            ),
            inference_id=_id(f"dense-inference:{unit.unit_id}"),
            input_plan_id=input_plan.input_plan_id,
            input_plan_semantic_sha256=input_plan.semantic_sha256,
        )
        results: list[CameraDenseResult] = []
        for camera_ordinal, camera_id in enumerate(CAMERA_IDS):
            observation = selected.get((unit_ordinal, camera_id), ProviderObservation.GOOD)
            local_status = {
                ProviderObservation.GOOD: CameraQAStatus.GOOD,
                ProviderObservation.DEGRADED: CameraQAStatus.DEGRADED,
                ProviderObservation.UNUSABLE: CameraQAStatus.UNUSABLE,
                ProviderObservation.UNKNOWN: CameraQAStatus.UNKNOWN,
            }[observation]
            coordinate = f"{unit.unit_id}:{camera_id.value}"
            evidence = EnrichedEvidenceReference(
                correlation_token=f"ref:{_digest(f'dense-token:{coordinate}')}",
                provider_item_ordinal=camera_ordinal,
                package_id=package_id,
                package_ordinal=0,
                package_semantic_content_sha256=package_semantic,
                package_manifest_sha256=package_manifest,
                camera_id=camera_id,
                camera_ordinal=camera_ordinal,
                frame_id=_id(f"dense-frame:{coordinate}"),
                frame_ordinal=0,
                aligned_timestamp_ns=unit.effective_interval.start_ns + camera_ordinal,
                source_timestamp_ns=10_000 + camera_ordinal,
                source_artifact_uri=f"object://dense/{coordinate}",
                source_artifact_sha256=_digest(f"dense-frame:{coordinate}"),
            )
            claim = EnrichedProviderClaim(
                claim_id=_id(f"dense-claim:{coordinate}"),
                claim_ordinal=camera_ordinal,
                kind=ProviderClaimKind.QA_OBSERVATION,
                package_id=package_id,
                package_ordinal=0,
                camera_id=camera_id,
                interval=ProviderClaimInterval(
                    start_ns=unit.effective_interval.start_ns,
                    end_ns=unit.effective_interval.end_ns,
                ),
                label="local-dense-screen",
                observation=observation,
                evidence=(evidence,),
                model_reported_confidence=None,
                conflict_codes=(),
            )
            results.append(
                CameraDenseResult(
                    package_id=package_id,
                    package_ordinal=0,
                    camera_id=camera_id,
                    local_status=local_status,
                    source_output=output,
                    claim=claim,
                )
            )
        bundles.append(
            DenseQAUnitEvidence(
                unit_id=unit.unit_id,
                unit_semantic_digest=unit.semantic_digest,
                mcap_id=manifest.mcap_id,
                camera_mapping_run_id=_id("dense-camera-mapping"),
                camera_mapping_semantic_sha256=_digest("dense-camera-mapping"),
                alignment_id=_id("dense-alignment"),
                alignment_semantic_sha256=_digest("dense-alignment"),
                package_set_id=_id(f"dense-package-set:{unit.unit_id}"),
                split_plan_digest=_digest(f"dense-split:{unit.unit_id}"),
                member_manifest_sha256=_digest(f"dense-members:{unit.unit_id}"),
                packages=(package,),
                input_plan=input_plan,
                source_outputs=(output,),
                package_camera_results=tuple(results),
            )
        )
    return tuple(bundles)


def test_all_good_completes_with_explicit_empty_dense_work_and_six_cameras() -> None:
    coarse = _coarse_result()
    projector = QACompletionProjector()

    result = projector.project(coarse)
    replay = projector.project(coarse)

    assert result == replay
    assert result.status is QACompletionStatus.QA_COMPLETE
    assert result.dense_work_manifest.outcome is DenseQAOutcome.SKIPPED_NOT_NEEDED
    assert result.dense_work_manifest.items == ()
    assert result.final_aggregate is not None
    assert tuple(item.camera_id for item in result.final_aggregate.camera_results) == CAMERA_IDS
    assert all(len(item.observations) == 2 for item in result.final_aggregate.camera_results)
    assert result.coarse_coverage[0].interval.start_ns == 0
    assert result.coarse_coverage[0].interval.end_ns == 100
    assert result.coarse_coverage[-1].interval.start_ns == 100
    assert result.coarse_coverage[-1].interval.end_ns == 200
    assert result.source_outputs == coarse.source_outputs
    assert result.production_eligible is False
    assert result.final_aggregate.production_eligible is False
    assert "confidence" not in result.final_aggregate.model_dump(mode="json")


def test_completion_reuses_the_attached_coarse_logical_node_digest() -> None:
    coarse = _coarse_result()
    coarse_node = canonical_coarse_qa_logical_node(coarse)

    result = QACompletionProjector().project(coarse)

    assert semantic_sha256(coarse_qa_semantic_projection(coarse)) == coarse_node.semantic_sha256
    assert result.coarse_result_semantic_sha256 == coarse_node.semantic_sha256


def test_completion_logical_node_binds_the_exact_result_digest() -> None:
    coarse = _coarse_result()
    first = QACompletionProjector().project(coarse)
    changed = QACompletionProjector(policy_version="local-qa-completion-v3").project(coarse)

    first_node = canonical_qa_completion_logical_node(first)
    changed_node = canonical_qa_completion_logical_node(changed)

    assert first_node.node_type == "QA_COMPLETION_RESULT"
    assert first_node.node_logical_key == f"qa-completion-result:{first.semantic_sha256}"
    assert first_node.semantic_sha256 == first.semantic_sha256
    assert changed_node != first_node


def test_degraded_and_unusable_create_deterministic_exact_dense_work() -> None:
    coarse = _coarse_result(
        {
            (0, CameraId.CAM_05): ProviderObservation.UNUSABLE,
            (1, CameraId.CAM_02): ProviderObservation.DEGRADED,
        }
    )
    projector = QACompletionProjector(policy_version="local-qa-completion-v2")

    result = projector.project(coarse)
    replay = projector.project(coarse)

    assert result == replay
    assert result.status is QACompletionStatus.DENSE_REQUIRED
    assert result.final_aggregate is None
    manifest = result.dense_work_manifest
    assert manifest.outcome is DenseQAOutcome.DENSE_REQUIRED
    assert tuple(
        (item.source.package_ordinal, item.source.camera_id) for item in manifest.items
    ) == ((0, CameraId.CAM_05), (1, CameraId.CAM_02))
    assert tuple(
        (item.source.interval.start_ns, item.source.interval.end_ns) for item in manifest.items
    ) == ((0, 100), (100, 200))
    assert all(item.context_camera_ids == CAMERA_IDS for item in manifest.items)
    assert all(item.source.source_output in coarse.source_outputs for item in manifest.items)
    assert manifest.policy_version == "local-qa-completion-v2"
    assert all(item.production_eligible is False for item in manifest.items)


def test_dense_planning_merges_adjacent_work_then_pads_and_clips() -> None:
    coarse = _coarse_result(
        {
            (0, CameraId.CAM_05): ProviderObservation.UNUSABLE,
            (1, CameraId.CAM_02): ProviderObservation.DEGRADED,
        }
    )
    policy = DenseQAPlanningPolicy(
        version="local-qa-completion-v2",
        padding_ns=25,
    )

    result = QACompletionProjector(
        policy_version=policy.version,
        dense_planning_policy=policy,
    ).project(
        coarse,
        recording_interval=NanosecondInterval(start_ns=0, end_ns=200),
    )

    assert len(result.dense_work_manifest.units) == 1
    unit = result.dense_work_manifest.units[0]
    assert unit.requested_interval == NanosecondInterval(start_ns=0, end_ns=200)
    assert unit.effective_interval == NanosecondInterval(start_ns=0, end_ns=200)
    assert unit.context_truncated is True
    assert unit.target_camera_ids == (CameraId.CAM_02, CameraId.CAM_05)
    assert unit.context_camera_ids == CAMERA_IDS


def test_complete_dense_result_finalizes_qa_with_exact_six_camera_lineage() -> None:
    coarse = _coarse_result({(0, CameraId.CAM_05): ProviderObservation.UNUSABLE})
    completion = QACompletionProjector(policy_version="local-qa-completion-v2")
    pending = completion.project(coarse)
    dense = DenseQAProjector(policy_version=completion.policy_version).project(
        pending.dense_work_manifest,
        _dense_evidence(pending.dense_work_manifest),
    )

    result = completion.project(coarse, dense)

    assert dense.status is DenseQAStatus.COMPLETE
    assert result.status is QACompletionStatus.QA_COMPLETE
    assert result.dense_result == dense
    assert result.dense_failure_code is None
    assert result.final_aggregate is not None
    assert all(
        len(camera.dense_observations) == len(pending.dense_work_manifest.units)
        for camera in result.final_aggregate.camera_results
    )
    assert all(
        observation.local_status is CameraQAStatus.GOOD
        for camera in result.final_aggregate.camera_results
        for observation in camera.dense_observations
    )
    assert result.production_eligible is False


def test_unknown_dense_result_and_terminal_failure_both_fail_closed() -> None:
    coarse = _coarse_result({(1, CameraId.CAM_04): ProviderObservation.DEGRADED})
    completion = QACompletionProjector()
    pending = completion.project(coarse)
    dense = DenseQAProjector(policy_version=completion.policy_version).project(
        pending.dense_work_manifest,
        _dense_evidence(
            pending.dense_work_manifest,
            {(0, CameraId.CAM_04): ProviderObservation.UNKNOWN},
        ),
    )

    incomplete = completion.project(coarse, dense)
    blocked = completion.block(coarse, "DENSE_INFERENCE_FAILED")

    assert dense.status is DenseQAStatus.INCOMPLETE
    assert incomplete.status is QACompletionStatus.QA_INCOMPLETE
    assert incomplete.final_aggregate is None
    assert blocked.status is QACompletionStatus.QA_INCOMPLETE
    assert blocked.dense_work_manifest.outcome is DenseQAOutcome.DENSE_REQUIRED
    assert blocked.dense_work_manifest.units
    assert blocked.dense_result is None
    assert blocked.dense_failure_code == "DENSE_INFERENCE_FAILED"
    assert blocked.final_aggregate is None


def test_dense_projection_rejects_missing_six_camera_terminal_evidence() -> None:
    coarse = _coarse_result({(0, CameraId.CAM_01): ProviderObservation.DEGRADED})
    completion = QACompletionProjector()
    manifest = completion.project(coarse).dense_work_manifest
    evidence = _dense_evidence(manifest)[0]
    broken = evidence.model_copy(
        update={"package_camera_results": evidence.package_camera_results[:-1]}
    )

    with pytest.raises(DenseQAProjectionError, match="immutable contract validation"):
        DenseQAProjector(policy_version=completion.policy_version).project(
            manifest,
            (broken,),
        )


def test_unknown_is_incomplete_and_does_not_guess_dense_work() -> None:
    coarse = _coarse_result({(1, CameraId.CAM_04): ProviderObservation.UNKNOWN})

    result = QACompletionProjector().project(coarse)

    assert result.status is QACompletionStatus.QA_INCOMPLETE
    assert result.dense_work_manifest.outcome is DenseQAOutcome.BLOCKED_INCOMPLETE
    assert result.dense_work_manifest.items == ()
    assert result.final_aggregate is None
    unknown = result.coarse_coverage[9]
    assert unknown.camera_id is CameraId.CAM_04
    assert unknown.provider_observation is ProviderObservation.UNKNOWN
    assert unknown.local_status is CameraQAStatus.UNKNOWN


@pytest.mark.parametrize("tamper", ["coverage", "source-digest"])
def test_completion_rejects_broken_package_coverage_or_source_lineage(tamper: str) -> None:
    coarse = _coarse_result()
    if tamper == "coverage":
        broken = coarse.model_copy(
            update={"package_camera_results": coarse.package_camera_results[:-1]}
        )
    else:
        first = coarse.source_outputs[0].model_copy(
            update={"semantic_sha256": _digest("tampered-output")}
        )
        broken = coarse.model_copy(update={"source_outputs": (first, *coarse.source_outputs[1:])})

    with pytest.raises(QACompletionProjectionError, match="immutable contract validation"):
        QACompletionProjector().project(broken)


def test_completion_digest_fails_closed_after_tampering() -> None:
    result = QACompletionProjector().project(_coarse_result())
    tampered = result.model_copy(update={"semantic_sha256": "0" * 64})

    with pytest.raises(ValueError, match="semantic_sha256 is inconsistent"):
        QACompletionResult.model_validate(tampered.model_dump(mode="python"), strict=True)
