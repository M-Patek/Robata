from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from robata.application.canonical.runner import (
    CanonicalOfflinePipeline,
    _CountingBatchVisionModelAdapter,
    _CountingVisionModelAdapter,
)
from robata.contracts.pipeline import SamplingPurpose
from robata.inference.models import InferenceStatus, VisionTask
from robata.runtime.observability import RuntimeProfileRecorder


def _pipeline(*, max_concurrent_call_parts: int) -> CanonicalOfflinePipeline:
    pipeline = object.__new__(CanonicalOfflinePipeline)
    pipeline._max_concurrent_call_parts = max_concurrent_call_parts
    return pipeline


def _input_plan(*, part_count: int) -> SimpleNamespace:
    parts = tuple(SimpleNamespace(ordinal=ordinal) for ordinal in range(part_count))
    return SimpleNamespace(call_plan=SimpleNamespace(parts=parts))


async def _execute_parts(
    pipeline: CanonicalOfflinePipeline,
    input_plan: SimpleNamespace,
) -> tuple[object, ...]:
    return await pipeline._execute_call_plan_parts(
        task=VisionTask.FUSION_ADJUDICATION,
        inference_policy=object(),
        context=object(),
        window=object(),
        sampling_plan=object(),
        package_set=object(),
        package_inputs=(),
        input_plan=input_plan,
        reference_catalog=object(),
        dependency_config=None,
    )


def test_execute_call_plan_parts_bounds_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pipeline = _pipeline(max_concurrent_call_parts=2)
        input_plan = _input_plan(part_count=5)
        release = asyncio.Event()
        saturated = asyncio.Event()
        active = 0
        peak_active = 0
        started: list[int] = []

        async def execute_one(**kwargs: object) -> int:
            nonlocal active, peak_active
            part = kwargs["part"]
            assert isinstance(part, SimpleNamespace)
            active += 1
            peak_active = max(peak_active, active)
            started.append(part.ordinal)
            if active == 2:
                saturated.set()
            try:
                await release.wait()
                return part.ordinal
            finally:
                active -= 1

        monkeypatch.setattr(pipeline, "_execute_one_call_part", execute_one)
        execution = asyncio.create_task(_execute_parts(pipeline, input_plan))

        await asyncio.wait_for(saturated.wait(), timeout=1)
        await asyncio.sleep(0)
        assert active == 2
        assert started == [0, 1]

        release.set()
        assert await asyncio.wait_for(execution, timeout=1) == (0, 1, 2, 3, 4)
        assert peak_active == 2

    asyncio.run(scenario())


def test_execute_call_plan_parts_preserves_input_order_after_reverse_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pipeline = _pipeline(max_concurrent_call_parts=3)
        input_plan = _input_plan(part_count=3)
        releases = tuple(asyncio.Event() for _ in range(3))
        all_started = asyncio.Event()
        started = 0
        completed: list[int] = []

        async def execute_one(**kwargs: object) -> int:
            nonlocal started
            part = kwargs["part"]
            assert isinstance(part, SimpleNamespace)
            started += 1
            if started == 3:
                all_started.set()
            await releases[part.ordinal].wait()
            completed.append(part.ordinal)
            return part.ordinal

        monkeypatch.setattr(pipeline, "_execute_one_call_part", execute_one)
        execution = asyncio.create_task(_execute_parts(pipeline, input_plan))
        await asyncio.wait_for(all_started.wait(), timeout=1)

        for ordinal in reversed(range(3)):
            releases[ordinal].set()
            await asyncio.sleep(0)

        assert await asyncio.wait_for(execution, timeout=1) == (0, 1, 2)
        assert completed == [2, 1, 0]

    asyncio.run(scenario())


def test_execute_call_plan_parts_cancels_and_awaits_siblings_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pipeline = _pipeline(max_concurrent_call_parts=3)
        input_plan = _input_plan(part_count=3)
        all_started = asyncio.Event()
        blocker = asyncio.Event()
        started: set[int] = set()
        cancelled: set[int] = set()
        cleaned_up: set[int] = set()

        async def execute_one(**kwargs: object) -> int:
            part = kwargs["part"]
            assert isinstance(part, SimpleNamespace)
            started.add(part.ordinal)
            if len(started) == 3:
                all_started.set()
            try:
                await all_started.wait()
                if part.ordinal == 1:
                    raise RuntimeError("injected part failure")
                await blocker.wait()
                return part.ordinal
            except asyncio.CancelledError:
                cancelled.add(part.ordinal)
                await asyncio.sleep(0)
                cleaned_up.add(part.ordinal)
                raise

        monkeypatch.setattr(pipeline, "_execute_one_call_part", execute_one)

        with pytest.raises(RuntimeError, match="injected part failure"):
            await asyncio.wait_for(_execute_parts(pipeline, input_plan), timeout=1)

        assert cancelled == {0, 2}
        assert cleaned_up == {0, 2}

    asyncio.run(scenario())


def _counter_total(recorder: RuntimeProfileRecorder, name: str) -> int:
    return sum(counter.value for counter in recorder.snapshot().counters if counter.name == name)


def test_planned_call_part_counters_track_amplification_and_global_unique_images() -> None:
    recorder = RuntimeProfileRecorder()
    pipeline = object.__new__(CanonicalOfflinePipeline)
    pipeline._runtime_observer = recorder
    pipeline._observed_unique_image_frame_ids = set()
    pipeline._observed_coarse_image_frame_ids = set()
    pipeline._observed_dense_image_frame_ids = set()
    pipeline._observed_window_keys = set()
    input_plan = SimpleNamespace(
        target=SimpleNamespace(provider="fixture-provider"),
        rendered_items=(
            SimpleNamespace(
                frame_id="frame-a",
                artifact=SimpleNamespace(media_type="image/jpeg", sha256="sha-a"),
            ),
            SimpleNamespace(
                frame_id="frame-b-first-selection",
                artifact=SimpleNamespace(media_type="image/jpeg", sha256="sha-b"),
            ),
            SimpleNamespace(
                frame_id="frame-b-reselected",
                artifact=SimpleNamespace(media_type="image/jpeg", sha256="sha-b"),
            ),
            SimpleNamespace(
                frame_id="frame-c",
                artifact=SimpleNamespace(media_type="image/jpeg", sha256="sha-c"),
            ),
            SimpleNamespace(
                frame_id="video-d",
                artifact=SimpleNamespace(media_type="video/mp4", sha256="sha-video"),
            ),
        ),
    )
    first_part = SimpleNamespace(
        ordinal=0,
        start_item_ordinal=0,
        end_item_ordinal_exclusive=2,
        measured_input_tokens=5,
    )
    dense_part = SimpleNamespace(
        ordinal=1,
        start_item_ordinal=2,
        end_item_ordinal_exclusive=5,
        measured_input_tokens=7,
    )

    pipeline._observe_planned_call_part(
        task=VisionTask.ACTION_EVIDENCE,
        window=SimpleNamespace(purpose=SamplingPurpose.ACTION_DENSE, window_id="window-a"),
        input_plan=input_plan,
        part=first_part,
    )
    pipeline._observe_planned_call_part(
        task=VisionTask.ACTION_EVIDENCE,
        window=SimpleNamespace(purpose=SamplingPurpose.ACTION_DENSE, window_id="window-a"),
        input_plan=input_plan,
        part=dense_part,
    )
    pipeline._observe_terminal_retry_count(
        task=VisionTask.ACTION_EVIDENCE,
        terminal=SimpleNamespace(provider="fixture-provider", retry_count=2),
    )

    # A logical call is the unsplit input plan.  The two dispatch parts are
    # exposed separately through call_splits and provider-image amplification.
    assert _counter_total(recorder, "inference.logical_calls") == 1
    assert _counter_total(recorder, "inference.call_splits") == 1
    assert _counter_total(recorder, "inference.provider_images") == 4
    # frame-b entries have identical bytes but distinct selected-frame identities.
    assert _counter_total(recorder, "inference.unique_images") == 4
    assert _counter_total(recorder, "inference.input_tokens") == 12
    assert _counter_total(recorder, "inference.dense_logical_calls") == 1
    assert _counter_total(recorder, "inference.dense_provider_images") == 4
    assert _counter_total(recorder, "inference.dense_unique_images") == 4
    assert _counter_total(recorder, "sampling.windows") == 1
    assert _counter_total(recorder, "inference.provider_retries") == 2
    unique_counter = next(
        counter
        for counter in recorder.snapshot().counters
        if counter.name == "inference.unique_images"
    )
    assert unique_counter.attributes == ()


class _SingleOutcomeAdapter:
    provider = "fixture-provider"

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def infer(self, request: object) -> object:
        del request
        return self._outcome


class _BatchOutcomeAdapter:
    provider = "fixture-provider"

    def __init__(self, outcomes: list[tuple[object, ...]]) -> None:
        self._outcomes = outcomes

    async def infer_batch(self, requests: tuple[object, ...]) -> tuple[object, ...]:
        del requests
        return self._outcomes.pop(0)


def _outcome(*, output_tokens: int | None) -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(output_tokens=output_tokens))


def test_provider_batch_and_known_output_token_counters_include_size_one_batches() -> None:
    async def scenario() -> RuntimeProfileRecorder:
        recorder = RuntimeProfileRecorder()
        request = SimpleNamespace(task=VisionTask.QA_COARSE)
        single = _CountingVisionModelAdapter(
            _SingleOutcomeAdapter(_outcome(output_tokens=0)),
            runtime_observer=recorder,
        )
        batch = _CountingBatchVisionModelAdapter(
            _BatchOutcomeAdapter(
                [
                    (_outcome(output_tokens=1),),
                    (_outcome(output_tokens=None), _outcome(output_tokens=3)),
                ]
            ),
            runtime_observer=recorder,
        )

        await single.infer(request)
        await batch.infer_batch((request,))
        await batch.infer_batch((request, request))
        return recorder

    recorder = asyncio.run(scenario())

    assert _counter_total(recorder, "inference.provider_batches") == 3
    assert _counter_total(recorder, "inference.provider_batch_requests") == 4
    assert _counter_total(recorder, "inference.output_token_responses") == 3
    assert _counter_total(recorder, "inference.output_tokens") == 4


class _NoReadLedger:
    def get_selection(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("call-part execution must reuse the returned selection")

    def get_terminal(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("call-part execution must reuse the returned terminal")


class _ReturnedSelectionOrchestrator:
    def __init__(self, terminal: object, selection: object) -> None:
        self._terminal = terminal
        self._selection = selection
        self.calls: list[dict[str, object]] = []

    async def orchestrate_with_selection(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(terminal=self._terminal, selection=self._selection)


def test_orchestrate_call_part_reuses_returned_selection_without_ledger_readbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh success must not read the just-persisted selection/terminal a second time."""

    monkeypatch.setattr(
        "robata.application.canonical.runner.sampling_plan_digest",
        lambda _sampling_plan, *, purpose: f"sampling:{purpose.value}",
    )
    terminal = SimpleNamespace(
        status=InferenceStatus.SUCCEEDED,
        inference_id="terminal-1",
        logical_invocation_id="invocation-1",
        input_plan_part_ordinal=0,
        output_valid=True,
    )
    selection = SimpleNamespace(
        inference_id=terminal.inference_id,
        logical_invocation_id=terminal.logical_invocation_id,
    )
    orchestrator = _ReturnedSelectionOrchestrator(terminal, selection)
    pipeline = object.__new__(CanonicalOfflinePipeline)
    pipeline._orchestrator = orchestrator
    pipeline._ledger = _NoReadLedger()
    pipeline._execution_policy = SimpleNamespace(
        max_attempts=1,
        semantic_sha256="execution-policy-sha",
    )
    context = SimpleNamespace(
        ready_manifest=SimpleNamespace(
            mcap_id="mcap-1",
            camera_mapping_run_id="mapping-1",
        ),
        alignment_manifest=SimpleNamespace(alignment_id="alignment-1"),
    )
    window = SimpleNamespace(
        interval=SimpleNamespace(start_ns=0, end_ns=1_000),
        purpose=SamplingPurpose.ACTION_DENSE,
    )
    part = SimpleNamespace(ordinal=0, item_manifest_sha256="part-manifest-sha")

    returned_terminal, returned_selection, attempts = asyncio.run(
        pipeline._orchestrate_call_part(
            task=VisionTask.ACTION_EVIDENCE,
            inference_policy=object(),
            context=context,
            window=window,
            sampling_plan=SimpleNamespace(version="sampling-v1"),
            package_set=SimpleNamespace(package_set_id="package-set-1"),
            package_inputs=(),
            input_plan=object(),
            part=part,
        )
    )

    assert returned_terminal is terminal
    assert returned_selection is selection
    assert attempts == 1
    assert len(orchestrator.calls) == 1
