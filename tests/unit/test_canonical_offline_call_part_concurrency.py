from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from robata.application.canonical.runner import CanonicalOfflinePipeline
from robata.inference.models import VisionTask


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
