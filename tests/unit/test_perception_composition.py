from __future__ import annotations

import pytest

from robata.application.canonical.perception_composition import (
    LEGACY_QWEN_WINDOW_PROFILE,
    MAGE_STREAM_COMPOSITION_VERSION,
    MAGE_STREAM_VNEXT_PROFILE,
    PerceptionCompositionMode,
    PerceptionCompositionSelectionError,
    require_legacy_window_composition,
    resolve_perception_composition,
)
from robata.application.canonical.stream_scheduler import (
    StreamSchedulerCompositionError,
    require_legacy_window_scheduler_profile,
)


def test_default_composition_is_mage_native_stream_without_qwen_autoload() -> None:
    decision = resolve_perception_composition()

    assert decision.profile.value == MAGE_STREAM_VNEXT_PROFILE
    assert decision.mode is PerceptionCompositionMode.MAGE_STREAM
    assert decision.composition_version == MAGE_STREAM_COMPOSITION_VERSION
    assert decision.native_media_type == "video/mp4"
    assert decision.qwen_autoload is False
    assert decision.explicit_legacy_only is False


def test_legacy_composition_requires_an_explicit_legacy_profile() -> None:
    with pytest.raises(PerceptionCompositionSelectionError, match="explicit"):
        require_legacy_window_composition(None)

    with pytest.raises(PerceptionCompositionSelectionError, match="explicit"):
        require_legacy_window_composition(MAGE_STREAM_VNEXT_PROFILE)

    decision = require_legacy_window_composition(LEGACY_QWEN_WINDOW_PROFILE)
    assert decision.mode is PerceptionCompositionMode.LEGACY_WINDOW
    assert decision.explicit_legacy_only is True
    assert decision.qwen_autoload is False


def test_window_scheduler_admission_cannot_be_selected_by_default() -> None:
    with pytest.raises(StreamSchedulerCompositionError, match="explicit"):
        require_legacy_window_scheduler_profile(None)

    decision = require_legacy_window_scheduler_profile(LEGACY_QWEN_WINDOW_PROFILE)
    assert decision.mode is PerceptionCompositionMode.LEGACY_WINDOW


def test_production_runtime_defaults_to_mage_and_requires_explicit_legacy() -> None:
    from robata.application.canonical.production_runtime import (
        _resolve_production_perception_composition,
    )

    assert (
        _resolve_production_perception_composition(None).mode
        is PerceptionCompositionMode.MAGE_STREAM
    )
    assert (
        _resolve_production_perception_composition(LEGACY_QWEN_WINDOW_PROFILE).mode
        is PerceptionCompositionMode.LEGACY_WINDOW
    )
