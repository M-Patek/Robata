from __future__ import annotations

import pytest

from robata.application.canonical.perception_routing import (
    DEFAULT_PERCEPTION_ROUTE_PROFILE,
    LEGACY_QWEN_WINDOW_PROFILE,
    MAGE_STREAM_COMPOSITION_VERSION,
    MAGE_STREAM_VNEXT_PROFILE,
    PerceptionExecutionMode,
    PerceptionRouteKind,
    PerceptionRouteSelectionError,
    require_explicit_legacy_window_route,
    resolve_perception_route,
)
from robata.perception.durable_scheduler import (
    DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
)
from robata.perception.pipeline import PerceptionStage


def test_default_route_is_provider_neutral_mage_native_video() -> None:
    decision = resolve_perception_route()

    assert DEFAULT_PERCEPTION_ROUTE_PROFILE == MAGE_STREAM_VNEXT_PROFILE
    assert decision.profile is PerceptionRouteKind.MAGE_STREAM_VNEXT
    assert decision.execution_mode is PerceptionExecutionMode.MAGE_STREAM
    assert decision.composition_version == MAGE_STREAM_COMPOSITION_VERSION
    assert decision.scheduler_policy_version == DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION
    assert decision.native_media_type == "video/mp4"
    assert decision.normal_model_stage is PerceptionStage.PERCEPTION_OBSERVE
    assert decision.refinement_model_stage is PerceptionStage.PERCEPTION_REFINE
    assert decision.explicit_legacy_only is False
    assert all(
        "QWEN" not in stage.value and "MAGE" not in stage.value
        for stage in decision.provider_neutral_stages
    )


def test_qwen_window_route_requires_explicit_legacy_admission() -> None:
    with pytest.raises(PerceptionRouteSelectionError, match="explicit"):
        resolve_perception_route(LEGACY_QWEN_WINDOW_PROFILE)

    decision = resolve_perception_route(
        LEGACY_QWEN_WINDOW_PROFILE,
        allow_explicit_legacy_qwen=True,
    )
    assert decision.profile is PerceptionRouteKind.LEGACY_QWEN_WINDOW
    assert decision.explicit_legacy_only is True


def test_legacy_window_helper_rejects_implicit_or_mage_selection() -> None:
    with pytest.raises(PerceptionRouteSelectionError, match="explicit"):
        require_explicit_legacy_window_route(None)
    with pytest.raises(PerceptionRouteSelectionError, match="explicit"):
        require_explicit_legacy_window_route(MAGE_STREAM_VNEXT_PROFILE)

    decision = require_explicit_legacy_window_route(LEGACY_QWEN_WINDOW_PROFILE)
    assert decision.execution_mode is PerceptionExecutionMode.LEGACY_WINDOW
    assert decision.scheduler_policy_version == "stream-window-dag-v4"
