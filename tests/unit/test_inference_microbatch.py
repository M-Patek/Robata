from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter

import pytest

from robata.inference.adapter import (
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
)
from robata.inference.models import InferenceStatus, ModelCapabilities, VisionTask
from robata.inference.orchestrator import (
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from tests.unit.test_inference_orchestrator import (
    NOW,
    SCHEMA,
    _capabilities,
    _digest,
    _package_input,
    _policy,
    _request_kwargs,
    _success,
    _uuid,
)

Outcome = VisionInferenceSuccess | VisionInferenceFailure


def _request(
    ordinal: int,
    *,
    task: VisionTask = VisionTask.ACTION_EVIDENCE,
    role: str = "primary",
) -> dict[str, object]:
    package = _package_input().model_copy(
        update={
            "package_id": _uuid(1_000 + ordinal),
            "package_semantic_content_sha256": _digest(2_000 + ordinal),
            "package_manifest_sha256": _digest(3_000 + ordinal),
            "role": role,
        }
    )
    return {
        **_request_kwargs(),
        "task": task,
        "package_set_id": _uuid(4_000 + ordinal),
        "package_inputs": (package,),
        "rendered_input_digest": _digest(5_000 + ordinal),
    }


def _outcome(request: VisionInferenceRequest) -> VisionInferenceSuccess:
    return _success(request).model_copy(
        update={
            "provider_request_id": f"batch:{request.request_id}",
            "raw_output_artifact_id": f"raw:{request.request_id}",
        }
    )


class BatchFakeAdapter:
    provider = "fake"

    def __init__(
        self,
        *,
        capabilities: ModelCapabilities,
        outcome_factory: Callable[[VisionInferenceRequest], Outcome] = _outcome,
        batch_error: BaseException | None = None,
        batch_delay_seconds: float = 0,
    ) -> None:
        self._capabilities = capabilities
        self._outcome_factory = outcome_factory
        self._batch_error = batch_error
        self._batch_delay_seconds = batch_delay_seconds
        self.infer_calls: list[VisionInferenceRequest] = []
        self.batch_calls: list[tuple[VisionInferenceRequest, ...]] = []

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        assert (model_name, model_version) == ("local-fake", "1.0")
        return self._capabilities

    async def infer(self, request: VisionInferenceRequest) -> Outcome:
        self.infer_calls.append(request)
        return self._outcome_factory(request)

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[Outcome, ...]:
        self.batch_calls.append(requests)
        if self._batch_delay_seconds:
            await asyncio.sleep(self._batch_delay_seconds)
        if self._batch_error is not None:
            raise self._batch_error
        return tuple(self._outcome_factory(request) for request in requests)


def _policies(
    *tasks: VisionTask,
    timeout_ms: int = 500,
) -> dict[VisionTask, InferencePolicy]:
    policies: dict[VisionTask, InferencePolicy] = {}
    for index, task in enumerate(tasks):
        policies[task] = _policy().model_copy(
            update={
                "task": task,
                "policy_version": f"policy-{index + 1}",
                "prompt_version": f"prompt-{index + 1}",
                "timeout_ms": timeout_ms,
            }
        )
    return policies


def _orchestrator(
    adapter: BatchFakeAdapter,
    policies: dict[VisionTask, InferencePolicy],
    *,
    max_batch_size: int,
    max_queue_delay_ms: int = 5,
) -> tuple[InferenceOrchestrator, InMemoryInferenceLedger]:
    ledger = InMemoryInferenceLedger()
    return (
        InferenceOrchestrator(
            adapters={"fake": adapter},
            task_policies=policies,
            schema_documents={
                policy.output_schema.artifact_id: SCHEMA for policy in policies.values()
            },
            ledger=ledger,
            max_batch_size=max_batch_size,
            max_batch_queue_delay_ms=max_queue_delay_ms,
            clock=lambda: NOW,
        ),
        ledger,
    )


def test_compatible_requests_use_one_real_batch_dispatch_in_input_order() -> None:
    async def scenario() -> None:
        policies = _policies(VisionTask.ACTION_EVIDENCE)
        adapter = BatchFakeAdapter(capabilities=_capabilities())
        orchestrator, _ = _orchestrator(adapter, policies, max_batch_size=3)

        results = await asyncio.gather(
            *(orchestrator.orchestrate(**_request(index)) for index in range(3))
        )

        assert adapter.infer_calls == []
        assert len(adapter.batch_calls) == 1
        dispatched = adapter.batch_calls[0]
        assert tuple(item.request_id for item in dispatched) == tuple(
            result.request_id for result in results
        )
        assert all(result.status is InferenceStatus.SUCCEEDED for result in results)

    asyncio.run(scenario())


def test_purpose_and_input_shape_are_never_mixed_in_one_batch() -> None:
    async def scenario() -> None:
        tasks = (VisionTask.ACTION_EVIDENCE, VisionTask.QA_COARSE)
        capabilities = _capabilities().model_copy(update={"supported_tasks": tasks})
        adapter = BatchFakeAdapter(capabilities=capabilities)
        orchestrator, _ = _orchestrator(
            adapter,
            _policies(*tasks),
            max_batch_size=8,
            max_queue_delay_ms=2,
        )

        await asyncio.gather(
            orchestrator.orchestrate(**_request(1)),
            orchestrator.orchestrate(**_request(2, task=VisionTask.QA_COARSE)),
            orchestrator.orchestrate(**_request(3, role="secondary")),
        )

        assert len(adapter.batch_calls) == 3
        assert all(len(batch) == 1 for batch in adapter.batch_calls)
        assert all(len({request.task for request in batch}) == 1 for batch in adapter.batch_calls)
        assert all(
            len(
                {
                    tuple((item.role, item.ordinal) for item in request.package_inputs)
                    for request in batch
                }
            )
            == 1
            for batch in adapter.batch_calls
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("batch_error", "delay_seconds", "timeout_ms", "expected_code"),
    (
        (RuntimeError("batch transport failed"), 0, 500, "ADAPTER_EXCEPTION"),
        (None, 0.05, 10, "ADAPTER_TIMEOUT"),
    ),
)
def test_batch_dispatch_error_or_timeout_cannot_publish_a_successful_prefix(
    batch_error: BaseException | None,
    delay_seconds: float,
    timeout_ms: int,
    expected_code: str,
) -> None:
    async def scenario() -> None:
        policies = _policies(VisionTask.ACTION_EVIDENCE, timeout_ms=timeout_ms)
        adapter = BatchFakeAdapter(
            capabilities=_capabilities(),
            batch_error=batch_error,
            batch_delay_seconds=delay_seconds,
        )
        orchestrator, ledger = _orchestrator(adapter, policies, max_batch_size=2)

        results = await asyncio.gather(
            orchestrator.orchestrate(**_request(1)),
            orchestrator.orchestrate(**_request(2)),
        )

        assert len(adapter.batch_calls) == 1
        assert all(result.status is not InferenceStatus.SUCCEEDED for result in results)
        assert {result.failure.code for result in results if result.failure} == {expected_code}
        assert ledger.list_selections() == ()

    asyncio.run(scenario())


def test_serial_and_batch_dispatch_produce_identical_semantic_results() -> None:
    async def execute(max_batch_size: int) -> tuple[object, ...]:
        policies = _policies(VisionTask.ACTION_EVIDENCE)
        adapter = BatchFakeAdapter(capabilities=_capabilities())
        orchestrator, _ = _orchestrator(
            adapter,
            policies,
            max_batch_size=max_batch_size,
        )
        return tuple(
            await asyncio.gather(
                *(orchestrator.orchestrate(**_request(index)) for index in range(3))
            )
        )

    serial = asyncio.run(execute(1))
    batched = asyncio.run(execute(3))

    assert batched == serial


def test_queue_delay_flushes_a_partial_batch_within_the_bound() -> None:
    async def scenario() -> None:
        policies = _policies(VisionTask.ACTION_EVIDENCE)
        adapter = BatchFakeAdapter(capabilities=_capabilities())
        orchestrator, _ = _orchestrator(
            adapter,
            policies,
            max_batch_size=8,
            max_queue_delay_ms=5,
        )

        started = perf_counter()
        result = await orchestrator.orchestrate(**_request(1))
        elapsed = perf_counter() - started

        assert result.status is InferenceStatus.SUCCEEDED
        assert len(adapter.batch_calls) == 1
        assert elapsed < 0.2

    asyncio.run(scenario())
