from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from robata.inference import (
    EvaluationConflictError,
    EvaluationError,
    EvaluationService,
    InferenceFailure,
    InferenceStatus,
    ModelInference,
    ModelInferenceUsage,
    Retryability,
    ShadowRouter,
    ShadowRouteStatus,
    ShadowRoutingError,
    ShadowSelectionReason,
    VisionTask,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-19T12:00:00Z"


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _inference(
    *,
    inference_id: int,
    shadow: bool,
    normalized_output: dict[str, object] | None,
    status: InferenceStatus = InferenceStatus.SUCCEEDED,
    output_valid: bool = True,
    prompt_sha256: str | None = None,
) -> ModelInference:
    failure = None
    if status is not InferenceStatus.SUCCEEDED:
        failure = InferenceFailure(
            code="PROVIDER_UNAVAILABLE",
            detail="offline fixture failure",
            retryability=Retryability.RETRYABLE,
        )
    return ModelInference(
        schema_version="1.0",
        inference_id=_uuid(inference_id),
        logical_invocation_id=_uuid(inference_id + 100),
        request_id=_uuid(inference_id + 200),
        idempotency_key=f"request-{inference_id}",
        mcap_id=_uuid(3),
        package_set_id=_uuid(4),
        package_id=None,
        package_ids=(_uuid(5),),
        camera_mapping_run_id=_uuid(6),
        alignment_id=_uuid(7),
        start_ns=100,
        end_ns=200,
        stage=VisionTask.ACTION_EVIDENCE,
        provider="gpt" if shadow else "qwen",
        model_name="shadow-model" if shadow else "primary-model",
        model_version="1.0",
        adapter_version="1.0",
        prompt_version="1.0",
        prompt_artifact_id="prompt-action-evidence",
        prompt_sha256=prompt_sha256 or _digest(11),
        rendered_input_digest=_digest(12),
        output_schema_id="action-evidence",
        output_schema_version="1.0",
        output_schema_artifact_id="schema-action-evidence",
        output_schema_sha256=_digest(13),
        capability_snapshot_id=_uuid(inference_id + 300),
        capability_snapshot_digest=_digest(inference_id + 20),
        input_manifest_set_sha256=_digest(14),
        input_config={"camera_count": 6},
        sampling_config={"policy": "dense-v1"},
        generation_config={"temperature": 0.0},
        provider_request_id=f"provider-request-{inference_id}",
        experiment_id="experiment-1" if shadow else None,
        shadow_route_id=_uuid(10) if shadow else None,
        primary_inference_id=_uuid(1) if shadow else None,
        shadow=shadow,
        attempt=1,
        retry_count=0,
        status=status,
        queued_at=NOW_TEXT,
        started_at=NOW_TEXT,
        completed_at=NOW_TEXT,
        latency_ms=20 if shadow else 10,
        raw_output=normalized_output,
        normalized_output=normalized_output,
        output_valid=output_valid,
        reported_confidence={"value": 0.8} if output_valid else None,
        calibrated_confidence={"value": 0.8} if output_valid else None,
        usage=ModelInferenceUsage(
            input_frames=6,
            input_images=6,
            input_tokens=100,
            output_tokens=20,
            cost=0.25,
            currency="USD",
        ),
        failure=failure,
        created_at=NOW_TEXT,
    )


def test_shadow_sampling_is_stable_and_zero_ratio_needs_no_package_id() -> None:
    arguments = {
        "package_set_member_manifest_digest": _digest(20),
        "task": VisionTask.ACTION_EVIDENCE,
        "experiment_contract_digest": _digest(21),
        "shadow_policy_version": "1.0",
        "shadow_sample_ratio": 0.37,
    }

    assert ShadowRouter().select_random(**arguments) == ShadowRouter().select_random(**arguments)
    assert ShadowRouter().select_random(**{**arguments, "shadow_sample_ratio": 0.0}) is False
    assert ShadowRouter().select_random(**{**arguments, "shadow_sample_ratio": 1.0}) is True
    assert ShadowRouter().route(**{**arguments, "shadow_sample_ratio": 0.0}) is None
    with pytest.raises(ShadowRoutingError, match="SHA-256"):
        ShadowRouter().select_random(
            **{**arguments, "package_set_member_manifest_digest": "not-a-digest"}
        )


def test_shadow_route_unions_reasons_without_mutating_primary() -> None:
    router = ShadowRouter(clock=lambda: NOW)
    common = {
        "package_set_id": _uuid(4),
        "package_set_member_manifest_digest": _digest(20),
        "task": VisionTask.ACTION_EVIDENCE,
        "experiment_contract_digest": _digest(21),
        "shadow_policy_version": "1.0",
        "shadow_sample_ratio": 1.0,
    }
    random_route = router.route(**common)
    assert random_route is not None
    assert random_route.reasons == (ShadowSelectionReason.RANDOM,)

    primary = _inference(
        inference_id=1,
        shadow=False,
        normalized_output={"action": {"type": "grasp"}},
    )
    primary_snapshot = primary.model_dump(mode="json")
    combined_route = router.route(
        **common,
        primary_inference=primary,
        calibrated_confidence={"value": 0.2},
    )

    assert combined_route is not None
    assert combined_route.shadow_route_id == random_route.shadow_route_id
    assert combined_route.created_at == random_route.created_at
    assert combined_route.primary_inference_id == primary.inference_id
    assert combined_route.reasons == (
        ShadowSelectionReason.RANDOM,
        ShadowSelectionReason.HARD_CASE,
    )
    assert len(router.routes) == 1
    assert primary.model_dump(mode="json") == primary_snapshot

    reimported_route = ShadowRouter(clock=lambda: NOW).route(
        **{**common, "package_set_id": _uuid(40)}
    )
    assert reimported_route is not None
    assert reimported_route.shadow_route_id == random_route.shadow_route_id
    with pytest.raises(ShadowRoutingError, match="package_set_id changed"):
        router.route(**{**common, "package_set_id": _uuid(40)})


def test_shadow_budget_gate_retains_explicit_outcome() -> None:
    router = ShadowRouter(clock=lambda: NOW)
    route = router.route(
        package_set_id=_uuid(4),
        package_set_member_manifest_digest=_digest(20),
        task=VisionTask.ACTION_EVIDENCE,
        experiment_contract_digest=_digest(21),
        shadow_policy_version="1.0",
        shadow_sample_ratio=1.0,
    )
    assert route is not None

    assert router.budget_gate(
        route=route,
        daily_spend_limit=100,
        current_daily_spend=100,
    ) == (False, ShadowRouteStatus.SKIPPED_BUDGET)
    assert router.budget_gate(
        route=route,
        queue_depth_limit=5,
        current_queue_depth=5,
    ) == (False, ShadowRouteStatus.DEFERRED)
    assert router.budget_gate(route=route) == (True, ShadowRouteStatus.QUEUED)
    assert route.status is ShadowRouteStatus.SELECTED

    with pytest.raises(ShadowRoutingError, match="supplied together"):
        router.budget_gate(route=route, daily_spend_limit=10)


def test_evaluation_is_recursive_idempotent_and_append_only() -> None:
    primary = _inference(
        inference_id=1,
        shadow=False,
        normalized_output={
            "action": {"objects": ["cup", "lid"], "score": 0.5, "type": "grasp"},
            "metadata": {"trace": "primary"},
        },
    )
    shadow = _inference(
        inference_id=2,
        shadow=True,
        normalized_output={
            "action": {"objects": ["cup"], "score": 0.51, "type": "reach"},
            "metadata": {"trace": "shadow"},
        },
    )
    primary_snapshot = primary.model_dump(mode="json")
    service = EvaluationService(clock=lambda: NOW)
    result = service.evaluate_pair(
        qwen_inference=primary,
        gpt_inference=shadow,
        comparison_contract_version="1.0",
        comparison_config={
            "ignored_paths": ["metadata.trace"],
            "numeric_tolerance": 0.02,
        },
    )

    assert result.status == "OPEN"
    assert [delta.path for delta in result.field_deltas] == [
        "action.objects[1]",
        "action.type",
    ]
    assert result.metrics is not None
    assert result.metrics.primary.latency_ms == 10
    assert result.metrics.shadow.latency_ms == 20
    assert (
        service.evaluate_pair(
            qwen_inference=primary,
            gpt_inference=shadow,
            comparison_contract_version="1.0",
            comparison_config={
                "ignored_paths": ["metadata.trace"],
                "numeric_tolerance": 0.02,
            },
        )
        == result
    )
    assert len(service.evaluations) == 1

    sample = service.persist_disagreement(
        evaluation_result=result,
        shadow_reason=ShadowSelectionReason.RANDOM,
        mcap_id=primary.mcap_id,
        start_ns=primary.start_ns,
        end_ns=primary.end_ns,
        package_set_id=primary.package_set_id,
        camera_mapping_run_id=primary.camera_mapping_run_id,
        alignment_id=primary.alignment_id,
    )
    replay = service.persist_disagreement(
        evaluation_result=result,
        shadow_reason=ShadowSelectionReason.RANDOM,
        mcap_id=primary.mcap_id,
        start_ns=primary.start_ns,
        end_ns=primary.end_ns,
        package_set_id=primary.package_set_id,
        camera_mapping_run_id=primary.camera_mapping_run_id,
        alignment_id=primary.alignment_id,
    )

    assert replay == sample
    assert sample.status == "OPEN"
    assert sample.qwen_inference_id == primary.inference_id
    assert sample.gpt_inference_id == shadow.inference_id
    assert len(service.disagreements) == 1
    assert primary.model_dump(mode="json") == primary_snapshot


def test_evaluation_fails_closed_for_incomparable_or_mutated_pair() -> None:
    primary = _inference(
        inference_id=1,
        shadow=False,
        normalized_output={"action": {"type": "grasp"}},
    )
    shadow = _inference(
        inference_id=2,
        shadow=True,
        normalized_output={"action": {"type": "reach"}},
    )
    service = EvaluationService(clock=lambda: NOW)

    with pytest.raises(EvaluationError, match="prompt_sha256"):
        service.evaluate_pair(
            qwen_inference=primary,
            gpt_inference=shadow.model_copy(update={"prompt_sha256": _digest(99)}),
            comparison_contract_version="1.0",
        )

    service.evaluate_pair(
        qwen_inference=primary,
        gpt_inference=shadow,
        comparison_contract_version="1.0",
    )
    changed_primary = primary.model_copy(
        update={"normalized_output": {"action": {"type": "release"}}}
    )
    with pytest.raises(EvaluationConflictError, match="different evidence"):
        service.evaluate_pair(
            qwen_inference=changed_primary,
            gpt_inference=shadow,
            comparison_contract_version="1.0",
        )


def test_provider_failure_is_an_evaluation_outcome() -> None:
    primary = _inference(
        inference_id=1,
        shadow=False,
        normalized_output={"action": {"type": "grasp"}},
    )
    failed_shadow = _inference(
        inference_id=2,
        shadow=True,
        normalized_output=None,
        status=InferenceStatus.FAILED,
        output_valid=False,
    )
    service = EvaluationService(clock=lambda: NOW)

    result = service.evaluate_pair(
        qwen_inference=primary,
        gpt_inference=failed_shadow,
        comparison_contract_version="1.0",
    )

    assert result.status == "PROVIDER_FAILURE"
    assert result.field_deltas == ()
    assert result.metrics is not None
    assert result.metrics.shadow.failure_code == "PROVIDER_UNAVAILABLE"
    sample = service.persist_disagreement(
        evaluation_result=result,
        shadow_reason=ShadowSelectionReason.RANDOM,
        mcap_id=primary.mcap_id,
        start_ns=primary.start_ns,
        end_ns=primary.end_ns,
        package_set_id=primary.package_set_id,
        camera_mapping_run_id=primary.camera_mapping_run_id,
        alignment_id=primary.alignment_id,
    )
    assert sample.status == "PROVIDER_FAILURE"


def test_agreement_is_not_persisted_as_disagreement() -> None:
    output = {"action": {"type": "grasp"}}
    primary = _inference(inference_id=1, shadow=False, normalized_output=output)
    shadow = _inference(inference_id=2, shadow=True, normalized_output=output)
    service = EvaluationService(clock=lambda: NOW)
    result = service.evaluate_pair(
        qwen_inference=primary,
        gpt_inference=shadow,
        comparison_contract_version="1.0",
    )
    assert result.status == "AGREEMENT"

    with pytest.raises(EvaluationError, match="only OPEN"):
        service.persist_disagreement(
            evaluation_result=result,
            shadow_reason=ShadowSelectionReason.RANDOM,
            mcap_id=primary.mcap_id,
            start_ns=primary.start_ns,
            end_ns=primary.end_ns,
            package_set_id=primary.package_set_id,
            camera_mapping_run_id=primary.camera_mapping_run_id,
            alignment_id=primary.alignment_id,
        )


def test_disagreement_interval_is_fail_closed() -> None:
    primary = _inference(
        inference_id=1,
        shadow=False,
        normalized_output={"action": {"type": "grasp"}},
    )
    shadow = _inference(
        inference_id=2,
        shadow=True,
        normalized_output={"action": {"type": "reach"}},
    )
    service = EvaluationService(clock=lambda: NOW)
    result = service.evaluate_pair(
        qwen_inference=primary,
        gpt_inference=shadow,
        comparison_contract_version="1.0",
    )

    with pytest.raises(EvaluationError, match="nonempty"):
        service.persist_disagreement(
            evaluation_result=result,
            shadow_reason=ShadowSelectionReason.HARD_CASE,
            mcap_id=primary.mcap_id,
            start_ns=200,
            end_ns=200,
            package_set_id=primary.package_set_id,
            camera_mapping_run_id=primary.camera_mapping_run_id,
            alignment_id=primary.alignment_id,
        )
