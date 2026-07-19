"""Shadow routing for GPT shadow inference.

Architecture V1.1 — Section 11 (GPT shadow path).
"""

from __future__ import annotations

from robata.contracts.common import SchemaVersion
from robata.inference.models import (
    ModelInference,
    NonNegativeInt,
    ShadowRoute,
    ShadowRouteStatus,
    UnitInterval,
    VisionTask,
)


# ---------------------------------------------------------------------------
# ShadowRouter
# ---------------------------------------------------------------------------


class ShadowRouter:
    """Reproducible shadow route selection and budget gating.

    Implements the union of stable random sampling and hard-case sampling
    as described in Architecture V1.1 Section 11.2.
    """

    def select_random(
        self,
        *,
        package_set_member_manifest_digest: str,
        task: VisionTask,
        experiment_contract_digest: str,
        shadow_policy_version: SchemaVersion,
        shadow_sample_ratio: UnitInterval,
    ) -> bool:
        """Stable hash-based random sampling.

        Maps ``hash(package_set_member_manifest_digest, task,
        experiment_contract_digest, shadow_policy_version)`` to ``[0, 1)``
        and compares with ``shadow_sample_ratio``.

        Args:
            package_set_member_manifest_digest: Immutable ordered package content digest.
            task: The vision task being evaluated.
            experiment_contract_digest: Digest of the experiment contract.
            shadow_policy_version: Version of the shadow routing policy.
            shadow_sample_ratio: Fraction of packages to route (0.0 to 1.0).

        Returns:
            True if the package should be routed to GPT shadow.
        """
        raise NotImplementedError

    def select_hard_case(
        self,
        *,
        primary_inference: ModelInference,
        calibrated_confidence: dict[str, object] | None = None,
        qa_signals: dict[str, object] | None = None,
        boundary_signals: dict[str, object] | None = None,
        policy_version: SchemaVersion,
    ) -> bool:
        """Hard-case sampling based on low confidence and anomaly signals.

        Routes after the relevant Qwen/fusion result when configured rules
        detect low calibrated confidence, high view disagreement, ambiguous QA,
        uncertain boundaries, invalid-output repair, or another versioned signal.

        Args:
            primary_inference: The completed primary (Qwen) inference result.
            calibrated_confidence: Optional calibrated confidence values.
            qa_signals: Optional QA anomaly signals.
            boundary_signals: Optional boundary uncertainty signals.
            policy_version: Version of the hard-case selection policy.

        Returns:
            True if the result qualifies as a hard-case shadow route.
        """
        raise NotImplementedError

    def route(
        self,
        *,
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
        """Union of random and hard-case selection with deduplication.

        Evaluates both selection mechanisms and returns a single ``ShadowRoute``
        if either (or both) selects the package. Reasons are recorded as an
        append-only set. Returns ``None`` when neither mechanism selects.

        Args:
            package_set_member_manifest_digest: Immutable ordered package content digest.
            task: The vision task being evaluated.
            experiment_contract_digest: Digest of the experiment contract.
            shadow_policy_version: Version of the shadow routing policy.
            shadow_sample_ratio: Fraction of packages for random sampling.
            primary_inference: Optional primary inference for hard-case evaluation.
            calibrated_confidence: Optional calibrated confidence values.
            qa_signals: Optional QA anomaly signals.
            boundary_signals: Optional boundary uncertainty signals.

        Returns:
            A ``ShadowRoute`` if selected, otherwise ``None``.
        """
        raise NotImplementedError

    def budget_gate(
        self,
        *,
        route: ShadowRoute,
        daily_spend_limit: NonNegativeInt | None = None,
        current_daily_spend: NonNegativeInt | None = None,
        queue_depth_limit: NonNegativeInt | None = None,
        current_queue_depth: NonNegativeInt | None = None,
    ) -> tuple[bool, ShadowRouteStatus]:
        """Check spend and capacity limits before enqueuing a shadow route.

        A budget gate may mark selected work ``SKIPPED_BUDGET`` or defer it;
        selection must never disappear silently.

        Args:
            route: The shadow route to evaluate.
            daily_spend_limit: Optional daily spend limit in currency units.
            current_daily_spend: Optional current daily spend in currency units.
            queue_depth_limit: Optional maximum allowed shadow queue depth.
            current_queue_depth: Optional current shadow queue depth.

        Returns:
            A tuple of ``(allowed, status)`` where ``allowed`` is True if the
            route passes the gate, and ``status`` is the resulting route status.
        """
        raise NotImplementedError


__all__ = [
    "ShadowRouter",
]
