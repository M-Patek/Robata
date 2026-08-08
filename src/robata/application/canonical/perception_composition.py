"""Composition-root selection for Mage stream vNext and legacy window runs.

This module is the provider-neutral boundary used by local and production
composition roots.  The default is the Mage native codec/video stream route.
The historical window/Qwen route remains readable for replay and compatibility,
but callers must explicitly select ``legacy_window_v1`` before constructing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from robata.application.canonical.perception_routing import (
    DEFAULT_PERCEPTION_ROUTE_PROFILE,
    LEGACY_QWEN_WINDOW_PROFILE,
    MAGE_STREAM_VNEXT_PROFILE,
    PerceptionRouteDecision,
    PerceptionRouteKind,
    resolve_perception_route,
)
from robata.perception.durable_scheduler import (
    DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
    SQLitePerceptionWorkScheduler,
)
from robata.perception.pipeline import PerceptionStage

MAGE_STREAM_COMPOSITION_VERSION: Final = "mage-stream-composition-v1"
LEGACY_WINDOW_COMPOSITION_VERSION: Final = "legacy-window-composition-v1"


class PerceptionCompositionMode(StrEnum):
    """Physical composition selected for one new run."""

    MAGE_STREAM = "MAGE_STREAM"
    LEGACY_WINDOW = "LEGACY_WINDOW"


class PerceptionCompositionSelectionError(ValueError):
    """A composition root attempted an implicit legacy selection."""


@dataclass(frozen=True, slots=True)
class PerceptionCompositionDecision:
    """Immutable route facts shared by local and production composition roots."""

    route: PerceptionRouteDecision
    mode: PerceptionCompositionMode
    composition_version: str
    scheduler_policy_version: str
    qwen_autoload: bool

    @property
    def profile(self) -> PerceptionRouteKind:
        return self.route.profile

    @property
    def provider_neutral_stages(self) -> tuple[PerceptionStage, ...]:
        return self.route.provider_neutral_stages

    @property
    def native_media_type(self) -> str:
        return self.route.native_media_type

    @property
    def explicit_legacy_only(self) -> bool:
        return self.route.explicit_legacy_only


def resolve_perception_composition(
    profile: str | PerceptionRouteKind | None = None,
    *,
    allow_explicit_legacy_qwen: bool = False,
) -> PerceptionCompositionDecision:
    """Resolve one composition, defaulting to the Mage stream route.

    ``None`` is intentionally equivalent to ``mage_stream_vnext_v1``.  A
    legacy selection is only accepted when the caller supplies both the legacy
    profile and explicit admission; this prevents a production/local bootstrap
    default from silently constructing the old window DAG or loading Qwen.
    """

    route = resolve_perception_route(
        profile,
        allow_explicit_legacy_qwen=allow_explicit_legacy_qwen,
    )
    if route.profile is PerceptionRouteKind.LEGACY_QWEN_WINDOW:
        return PerceptionCompositionDecision(
            route=route,
            mode=PerceptionCompositionMode.LEGACY_WINDOW,
            composition_version=LEGACY_WINDOW_COMPOSITION_VERSION,
            scheduler_policy_version="stream-window-dag-v4",
            qwen_autoload=False,
        )
    return PerceptionCompositionDecision(
        route=route,
        mode=PerceptionCompositionMode.MAGE_STREAM,
        composition_version=MAGE_STREAM_COMPOSITION_VERSION,
        scheduler_policy_version=DURABLE_PERCEPTION_SCHEDULER_POLICY_VERSION,
        qwen_autoload=False,
    )


def create_default_vnext_perception_scheduler(
    database_path: str | Path,
    *,
    profile: str | PerceptionRouteKind | None = None,
) -> SQLitePerceptionWorkScheduler:
    """Construct the vNext scheduler only for the Mage default composition.

    Historical window/replay callers retain their explicit legacy selector. A
    new/default perception run receives the provider-neutral SQLite context-work
    authority instead of constructing ``stream-window-dag-v4``.
    """

    try:
        decision = resolve_perception_composition(profile)
    except ValueError as error:
        raise PerceptionCompositionSelectionError(str(error)) from error
    if decision.mode is not PerceptionCompositionMode.MAGE_STREAM:
        raise PerceptionCompositionSelectionError(
            "durable vNext scheduler is only available for mage_stream_vnext_v1; "
            "legacy replay must use its explicit legacy scheduler"
        )
    return SQLitePerceptionWorkScheduler(database_path)


def require_legacy_window_composition(
    profile: str | PerceptionRouteKind | None,
) -> PerceptionCompositionDecision:
    """Resolve the compatibility window composition from an explicit profile.

    This helper is intended for old scheduler/replay entry points.  It refuses
    a missing or Mage profile rather than allowing a legacy scheduler to become
    the default by accident.
    """

    if profile is None:
        raise PerceptionCompositionSelectionError(
            "legacy window composition requires explicit profile='legacy_window_v1'"
        )
    try:
        selected = PerceptionRouteKind(profile)
    except ValueError as error:
        raise PerceptionCompositionSelectionError(
            f"unknown perception profile: {profile!r}"
        ) from error
    if selected is not PerceptionRouteKind.LEGACY_QWEN_WINDOW:
        raise PerceptionCompositionSelectionError(
            "legacy window composition requires explicit profile='legacy_window_v1'"
        )
    return resolve_perception_composition(
        selected,
        allow_explicit_legacy_qwen=True,
    )


__all__ = [
    "DEFAULT_PERCEPTION_ROUTE_PROFILE",
    "LEGACY_QWEN_WINDOW_PROFILE",
    "LEGACY_WINDOW_COMPOSITION_VERSION",
    "MAGE_STREAM_COMPOSITION_VERSION",
    "MAGE_STREAM_VNEXT_PROFILE",
    "PerceptionCompositionDecision",
    "PerceptionCompositionMode",
    "PerceptionCompositionSelectionError",
    "create_default_vnext_perception_scheduler",
    "require_legacy_window_composition",
    "resolve_perception_composition",
]
