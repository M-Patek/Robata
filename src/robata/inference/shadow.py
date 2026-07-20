"""Deterministic, provider-neutral shadow inference routing.

Architecture V1.1 - Section 11 (shadow path).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Final
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.common import SchemaVersion
from robata.inference.models import (
    InferenceStatus,
    ModelInference,
    NonNegativeInt,
    ShadowRoute,
    ShadowRouteStatus,
    ShadowSelectionReason,
    UnitInterval,
    VisionTask,
)

Clock = Callable[[], datetime]

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_LOW_CONFIDENCE_KEYS: Final = frozenset(
    {"confidence", "calibrated_confidence", "overall", "score", "value"}
)
_DISAGREEMENT_KEYS: Final = frozenset(
    {
        "disagreement",
        "disagreement_score",
        "high_disagreement",
        "high_view_disagreement",
        "view_disagreement",
        "view_disagreement_score",
    }
)
_AMBIGUOUS_QA_KEYS: Final = frozenset({"ambiguous", "ambiguous_qa", "qa_ambiguous", "uncertain_qa"})
_BOUNDARY_KEYS: Final = frozenset(
    {
        "boundary_uncertainty",
        "boundary_uncertainty_score",
        "uncertain",
        "uncertain_boundary",
    }
)
_AMBIGUOUS_STATES: Final = frozenset({"AMBIGUOUS", "INCOMPLETE", "UNKNOWN", "UNCERTAIN"})


class ShadowRoutingError(ValueError):
    """Raised when a shadow route cannot be evaluated safely."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _rfc3339(clock: Clock) -> str:
    value = clock()
    if not isinstance(value, datetime):
        raise TypeError("shadow router clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ShadowRoutingError("shadow router clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_digest(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ShadowRoutingError(f"{field} must be a nonempty string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ShadowRoutingError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _ratio(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowRoutingError("shadow_sample_ratio must be a finite number in [0, 1]")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ShadowRoutingError("shadow_sample_ratio must be a finite number in [0, 1]")
    return ratio


def _iter_named_values(value: object) -> tuple[tuple[str, object], ...]:
    """Flatten JSON-like signal mappings while retaining normalized leaf names."""

    leaves: list[tuple[str, object]] = []

    def visit(item: object, name: str = "") -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(item[key], str(key).strip().lower())
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, name)
        else:
            leaves.append((name, item))

    visit(value)
    return tuple(leaves)


def _is_true(value: object) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().upper() in {
        "1",
        "HIGH",
        "TRUE",
        "YES",
        *_AMBIGUOUS_STATES,
    }


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class ShadowRouter:
    """Reproducible shadow selection with an isolated, explicit budget gate."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        low_confidence_threshold: float = 0.5,
        disagreement_threshold: float = 0.5,
        boundary_uncertainty_threshold: float = 0.5,
    ) -> None:
        self._clock = clock or _utc_now
        self._low_confidence_threshold = _ratio(low_confidence_threshold)
        self._disagreement_threshold = _ratio(disagreement_threshold)
        self._boundary_uncertainty_threshold = _ratio(boundary_uncertainty_threshold)
        self._routes: dict[str, ShadowRoute] = {}

    @property
    def routes(self) -> tuple[ShadowRoute, ...]:
        """Return an insertion-ordered snapshot of selected routes."""

        return tuple(self._routes.values())

    def select_random(
        self,
        *,
        package_set_member_manifest_digest: str,
        task: VisionTask,
        experiment_contract_digest: str,
        shadow_policy_version: SchemaVersion,
        shadow_sample_ratio: UnitInterval,
    ) -> bool:
        """Map the architecture-defined immutable preimage to ``[0, 1)``."""

        manifest_digest = _canonical_digest(
            package_set_member_manifest_digest,
            field="package_set_member_manifest_digest",
        )
        experiment_digest = _canonical_digest(
            experiment_contract_digest,
            field="experiment_contract_digest",
        )
        policy_version = self._policy_version(shadow_policy_version)
        task_value = self._task_value(task)
        ratio = _ratio(shadow_sample_ratio)
        if ratio == 0.0:
            return False
        if ratio == 1.0:
            return True

        preimage = "\x1f".join(
            (manifest_digest, task_value, experiment_digest, policy_version)
        ).encode("utf-8")
        sample = int.from_bytes(hashlib.sha256(preimage).digest(), "big") / (1 << 256)
        return sample < ratio

    def select_hard_case(
        self,
        *,
        primary_inference: ModelInference,
        calibrated_confidence: dict[str, object] | None = None,
        qa_signals: dict[str, object] | None = None,
        boundary_signals: dict[str, object] | None = None,
        policy_version: SchemaVersion,
    ) -> bool:
        """Apply deterministic hard-case rules to stored primary evidence."""

        self._policy_version(policy_version)
        if primary_inference.shadow:
            raise ShadowRoutingError("hard-case selection requires a primary inference")
        if primary_inference.status is InferenceStatus.INVALID_OUTPUT:
            return True
        if (
            primary_inference.status is InferenceStatus.SUCCEEDED
            and not primary_inference.output_valid
        ):
            return True

        confidence = calibrated_confidence or primary_inference.calibrated_confidence
        for key, value in _iter_named_values(confidence):
            number = _numeric(value)
            if (
                key in _LOW_CONFIDENCE_KEYS
                and number is not None
                and 0.0 <= number < self._low_confidence_threshold
            ):
                return True

        for key, value in _iter_named_values(qa_signals):
            if key in _AMBIGUOUS_QA_KEYS and _is_true(value):
                return True
            if (
                key in {"status", "state", "result"}
                and isinstance(value, str)
                and value.strip().upper() in _AMBIGUOUS_STATES
            ):
                return True
            if key in _DISAGREEMENT_KEYS:
                number = _numeric(value)
                if _is_true(value) or (
                    number is not None and number >= self._disagreement_threshold
                ):
                    return True

        for key, value in _iter_named_values(boundary_signals):
            if (
                key in {"status", "state"}
                and isinstance(value, str)
                and value.strip().upper() in _AMBIGUOUS_STATES
            ):
                return True
            if key in _BOUNDARY_KEYS:
                number = _numeric(value)
                if _is_true(value) or (
                    number is not None and number >= self._boundary_uncertainty_threshold
                ):
                    return True
        return False

    def route(
        self,
        *,
        package_set_id: str | None = None,
        package_set_member_manifest_digest: str,
        task: VisionTask,
        experiment_contract_digest: str,
        shadow_policy_version: SchemaVersion,
        shadow_sample_ratio: UnitInterval,
        primary_inference: ModelInference | None = None,
        calibrated_confidence: dict[str, object] | None = None,
        qa_signals: dict[str, object] | None = None,
        boundary_signals: dict[str, object] | None = None,
    ) -> ShadowRoute | None:
        """Return one deduplicated route containing the union of selection reasons."""

        manifest_digest = _canonical_digest(
            package_set_member_manifest_digest,
            field="package_set_member_manifest_digest",
        )
        experiment_digest = _canonical_digest(
            experiment_contract_digest,
            field="experiment_contract_digest",
        )
        policy_version = self._policy_version(shadow_policy_version)
        task_value = self._task_value(task)
        ratio = _ratio(shadow_sample_ratio)
        reasons: list[ShadowSelectionReason] = []
        if self.select_random(
            package_set_member_manifest_digest=manifest_digest,
            task=task,
            experiment_contract_digest=experiment_digest,
            shadow_policy_version=policy_version,
            shadow_sample_ratio=ratio,
        ):
            reasons.append(ShadowSelectionReason.RANDOM)
        if primary_inference is not None and self.select_hard_case(
            primary_inference=primary_inference,
            calibrated_confidence=calibrated_confidence,
            qa_signals=qa_signals,
            boundary_signals=boundary_signals,
            policy_version=policy_version,
        ):
            reasons.append(ShadowSelectionReason.HARD_CASE)

        resolved_package_set_id = package_set_id
        if resolved_package_set_id is None and primary_inference is not None:
            resolved_package_set_id = primary_inference.package_set_id
        if resolved_package_set_id is None:
            if not reasons:
                return None
            raise ShadowRoutingError("package_set_id is required for a selected shadow route")

        route_key = "\x1f".join(
            (
                manifest_digest,
                task_value,
                experiment_digest,
                policy_version,
            )
        )

        previous = self._routes.get(route_key)
        if not reasons and previous is None:
            return None
        if previous is not None:
            if previous.package_set_id != resolved_package_set_id:
                raise ShadowRoutingError(
                    "package_set_id changed for an immutable shadow route identity"
                )
            merged = tuple(dict.fromkeys((*previous.reasons, *reasons)))
            primary_inference_id = previous.primary_inference_id
            if primary_inference is not None:
                primary_inference_id = primary_inference.inference_id
            updated = previous.model_copy(
                update={
                    "primary_inference_id": primary_inference_id,
                    "reasons": merged,
                }
            )
            self._routes[route_key] = ShadowRoute.model_validate(updated.model_dump())
            return self._routes[route_key]

        route = ShadowRoute(
            schema_version="1.0",
            shadow_route_id=str(uuid5(NAMESPACE_URL, f"robata:shadow-route:{route_key}")),
            primary_inference_id=(
                primary_inference.inference_id if primary_inference is not None else None
            ),
            package_set_id=resolved_package_set_id,
            package_set_member_manifest_digest=manifest_digest,
            task=task,
            reasons=tuple(reasons),
            sample_ratio=ratio,
            policy_version=policy_version,
            status=ShadowRouteStatus.SELECTED,
            created_at=_rfc3339(self._clock),
        )
        self._routes[route_key] = route
        return route

    def budget_gate(
        self,
        *,
        route: ShadowRoute,
        daily_spend_limit: NonNegativeInt | None = None,
        current_daily_spend: NonNegativeInt | None = None,
        queue_depth_limit: NonNegativeInt | None = None,
        current_queue_depth: NonNegativeInt | None = None,
    ) -> tuple[bool, ShadowRouteStatus]:
        """Classify a selected route without mutating it or the primary path."""

        self._validate_limit_pair(
            limit=daily_spend_limit,
            current=current_daily_spend,
            name="daily_spend",
        )
        self._validate_limit_pair(
            limit=queue_depth_limit,
            current=current_queue_depth,
            name="queue_depth",
        )
        if (
            daily_spend_limit is not None
            and current_daily_spend is not None
            and current_daily_spend >= daily_spend_limit
        ):
            return False, ShadowRouteStatus.SKIPPED_BUDGET
        if (
            queue_depth_limit is not None
            and current_queue_depth is not None
            and current_queue_depth >= queue_depth_limit
        ):
            return False, ShadowRouteStatus.DEFERRED
        if route.status not in {ShadowRouteStatus.SELECTED, ShadowRouteStatus.DEFERRED}:
            raise ShadowRoutingError("budget gate accepts only SELECTED or DEFERRED routes")
        return True, ShadowRouteStatus.QUEUED

    @staticmethod
    def _validate_limit_pair(*, limit: int | None, current: int | None, name: str) -> None:
        for suffix, value in (("limit", limit), ("current", current)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ShadowRoutingError(f"{name}_{suffix} must be a nonnegative integer")
        if (limit is None) is not (current is None):
            raise ShadowRoutingError(f"{name}_limit and {name}_current must be supplied together")

    @staticmethod
    def _policy_version(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ShadowRoutingError("shadow policy version must be a nonempty string")
        return value

    @staticmethod
    def _task_value(task: object) -> str:
        if isinstance(task, VisionTask):
            return task.value
        if isinstance(task, str):
            try:
                return VisionTask(task).value
            except ValueError as error:
                raise ShadowRoutingError(f"unsupported vision task: {task!r}") from error
        raise ShadowRoutingError("task must be a VisionTask")


__all__ = [
    "ShadowRouter",
    "ShadowRoutingError",
]
