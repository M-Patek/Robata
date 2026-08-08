"""Provider-neutral route selection for stream-oriented perception.

Mage stream vNext is the repository default. The Qwen/window path remains readable
and runnable only through an explicit legacy selection; this module never imports
or loads either model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from robata.perception.durable_scheduler import (
    DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
)
from robata.perception.pipeline import PerceptionStage

MAGE_STREAM_VNEXT_PROFILE: Final = "mage_stream_vnext_v1"
LEGACY_QWEN_WINDOW_PROFILE: Final = "legacy_window_v1"
DEFAULT_PERCEPTION_ROUTE_PROFILE: Final = MAGE_STREAM_VNEXT_PROFILE
MAGE_STREAM_COMPOSITION_VERSION: Final = "mage-stream-composition-v1"
LEGACY_WINDOW_COMPOSITION_VERSION: Final = "legacy-window-composition-v1"


class PerceptionRouteSelectionError(ValueError):
    """A caller attempted to use a non-default route without explicit admission."""


class PerceptionRouteKind(StrEnum):
    MAGE_STREAM_VNEXT = MAGE_STREAM_VNEXT_PROFILE
    LEGACY_QWEN_WINDOW = LEGACY_QWEN_WINDOW_PROFILE


class PerceptionExecutionMode(StrEnum):
    """Physical graph selected for one new perception run."""

    MAGE_STREAM = "MAGE_STREAM"
    LEGACY_WINDOW = "LEGACY_WINDOW"


@dataclass(frozen=True, slots=True)
class PerceptionRouteDecision:
    profile: PerceptionRouteKind
    execution_mode: PerceptionExecutionMode
    composition_version: str
    scheduler_policy_version: str
    provider_neutral_stages: tuple[PerceptionStage, ...]
    native_media_type: str
    normal_model_stage: PerceptionStage
    refinement_model_stage: PerceptionStage
    explicit_legacy_only: bool


def resolve_perception_route(
    profile: str | PerceptionRouteKind | None = None,
    *,
    allow_explicit_legacy_qwen: bool = False,
) -> PerceptionRouteDecision:
    """Resolve the default Mage route without importing provider runtimes."""

    selected = PerceptionRouteKind(profile or DEFAULT_PERCEPTION_ROUTE_PROFILE)
    stages = tuple(PerceptionStage)  # type: tuple[PerceptionStage, ...]
    if selected is PerceptionRouteKind.LEGACY_QWEN_WINDOW:
        if not allow_explicit_legacy_qwen:
            raise PerceptionRouteSelectionError(
                "legacy Qwen/window routing requires explicit caller admission"
            )
        return PerceptionRouteDecision(
            profile=selected,
            execution_mode=PerceptionExecutionMode.LEGACY_WINDOW,
            composition_version=LEGACY_WINDOW_COMPOSITION_VERSION,
            scheduler_policy_version="stream-window-dag-v4",
            provider_neutral_stages=stages,
            native_media_type="image/png",
            normal_model_stage=PerceptionStage.PERCEPTION_OBSERVE,
            refinement_model_stage=PerceptionStage.PERCEPTION_REFINE,
            explicit_legacy_only=True,
        )
    return PerceptionRouteDecision(
        profile=selected,
        execution_mode=PerceptionExecutionMode.MAGE_STREAM,
        composition_version=MAGE_STREAM_COMPOSITION_VERSION,
        scheduler_policy_version=DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
        provider_neutral_stages=stages,
        native_media_type="video/mp4",
        normal_model_stage=PerceptionStage.PERCEPTION_OBSERVE,
        refinement_model_stage=PerceptionStage.PERCEPTION_REFINE,
        explicit_legacy_only=False,
    )


def require_explicit_legacy_window_route(
    profile: str | PerceptionRouteKind | None,
) -> PerceptionRouteDecision:
    """Admit the historical window graph only through an explicit legacy profile."""

    if profile is None:
        raise PerceptionRouteSelectionError(
            "legacy window routing requires explicit profile='legacy_window_v1'"
        )
    try:
        selected = PerceptionRouteKind(profile)
    except ValueError as error:
        raise PerceptionRouteSelectionError(f"unknown perception profile: {profile!r}") from error
    if selected is not PerceptionRouteKind.LEGACY_QWEN_WINDOW:
        raise PerceptionRouteSelectionError(
            "legacy window routing requires explicit profile='legacy_window_v1'"
        )
    return resolve_perception_route(selected, allow_explicit_legacy_qwen=True)


__all__ = [
    "DEFAULT_PERCEPTION_ROUTE_PROFILE",
    "LEGACY_QWEN_WINDOW_PROFILE",
    "LEGACY_WINDOW_COMPOSITION_VERSION",
    "MAGE_STREAM_COMPOSITION_VERSION",
    "MAGE_STREAM_VNEXT_PROFILE",
    "PerceptionExecutionMode",
    "PerceptionRouteDecision",
    "PerceptionRouteKind",
    "PerceptionRouteSelectionError",
    "require_explicit_legacy_window_route",
    "resolve_perception_route",
]
