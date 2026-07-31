"""Typed inputs for the executable Robata production-composition gate.

The configuration is deliberately a separately mounted immutable document. It
contains pinned deployment facts, never database or provider credentials. The
two referenced release artifacts are checked as exact bytes before they can
authorize the primary route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import StringConstraints, ValidationError, model_validator

from robata.application.canonical.production_composition import ProductionPrimaryRunPodBinding
from robata.application.canonical.production_runtime import ProductionCaptureAuthorityBinding
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.models import ModelCapabilities
from robata.inference.routing import ModelDeployment, ProductionRoute
from robata.inference.runpod import RunPodRetryPolicy
from robata.queue.outbox import OutboxRetryPolicy

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]


class ProductionBootstrapConfigurationError(RuntimeError):
    """The mounted production bootstrap input is absent, malformed, or unsafe."""


class PrimaryRouteReleaseDecision(StrictModel):
    """Immutable decision facts authorizing one pinned primary deployment."""

    schema_version: Literal["robata-primary-route-release-decision-v1"] = (
        "robata-primary-route-release-decision-v1"
    )
    decision: Literal["APPROVED"]
    route_id: NonEmptyString
    policy_version: NonEmptyString
    deployment: ModelDeployment
    qualification_report_ref: NonEmptyString
    qualification_report_sha256: Sha256Digest
    primary_binding_sha256: Sha256Digest
    handler_image_sha256: Sha256Digest


class ProductionRuntimeBootstrapConfiguration(StrictModel):
    """Non-secret facts required to activate the production composition root."""

    schema_version: Literal["robata-production-runtime-bootstrap-v1"] = (
        "robata-production-runtime-bootstrap-v1"
    )
    primary_binding: ProductionPrimaryRunPodBinding
    primary_capabilities: ModelCapabilities
    primary_retry_policy: RunPodRetryPolicy
    primary_route: ProductionRoute
    capture_authority: ProductionCaptureAuthorityBinding
    outbox_retry_policy_version: NonEmptyString
    outbox_max_attempts: int
    outbox_base_delay_seconds: float
    outbox_max_delay_seconds: float
    primary_parser_version: NonEmptyString = "robata-production-runpod-parser-v1"
    qualification_report_file: NonEmptyString
    release_decision_file: NonEmptyString

    @model_validator(mode="after")
    def validate_primary_binding(self) -> Self:
        endpoint = self.primary_binding.endpoint
        deployment = endpoint.deployment_configuration
        if deployment is None:
            raise ValueError("primary binding endpoint must include deployment_configuration")
        if (
            self.primary_capabilities.snapshot_digest
            != self.primary_binding.capability_snapshot_sha256
        ):
            raise ValueError("primary capability snapshot does not match primary binding")
        if (
            self.primary_capabilities.provider != endpoint.provider
            or self.primary_capabilities.model_name != deployment.model_identifier
            or self.primary_capabilities.model_version != deployment.model_version
        ):
            raise ValueError("primary capability facts do not match the pinned RunPod endpoint")
        if self.primary_route.deployment.model_name != deployment.model_identifier:
            raise ValueError("primary route model does not match the pinned RunPod endpoint")
        _checked_artifact_path(self.qualification_report_file, "qualification_report_file")
        _checked_artifact_path(self.release_decision_file, "release_decision_file")
        try:
            self.outbox_retry_policy()
        except (TypeError, ValueError) as error:
            raise ValueError(f"outbox retry policy is invalid: {error}") from error
        return self

    def outbox_retry_policy(self) -> OutboxRetryPolicy:
        """Build the immutable retry policy supplied to the canonical outbox."""

        return OutboxRetryPolicy(
            version=self.outbox_retry_policy_version,
            max_attempts=self.outbox_max_attempts,
            base_delay_seconds=self.outbox_base_delay_seconds,
            max_delay_seconds=self.outbox_max_delay_seconds,
        )

    def release_verifier(self) -> ExactArtifactPrimaryReleaseVerifier:
        """Return a verifier bound to this route and its exact frozen evidence."""

        return ExactArtifactPrimaryReleaseVerifier(
            primary_binding=self.primary_binding,
            primary_route=self.primary_route,
            qualification_report_file=Path(self.qualification_report_file),
            release_decision_file=Path(self.release_decision_file),
        )


class ExactArtifactPrimaryReleaseVerifier:
    """Verify a primary route against byte-pinned qualification and release files."""

    def __init__(
        self,
        *,
        primary_binding: ProductionPrimaryRunPodBinding,
        primary_route: ProductionRoute,
        qualification_report_file: Path,
        release_decision_file: Path,
    ) -> None:
        if not isinstance(primary_binding, ProductionPrimaryRunPodBinding):
            raise TypeError("primary_binding must be ProductionPrimaryRunPodBinding")
        if not isinstance(primary_route, ProductionRoute):
            raise TypeError("primary_route must be ProductionRoute")
        self._primary_binding = primary_binding
        self._primary_route = primary_route
        self._qualification_report_file = _checked_artifact_path(
            qualification_report_file,
            "qualification_report_file",
        )
        self._release_decision_file = _checked_artifact_path(
            release_decision_file,
            "release_decision_file",
        )

    def verify_primary(self, *, authorization: object, deployment: object) -> bool:
        """Return true only for the exact pinned route and exact evidence bytes."""

        if (
            authorization != self._primary_route.authorization
            or deployment != self._primary_route.deployment
        ):
            return False
        try:
            qualification_report = self._qualification_report_file.read_bytes()
            release_decision_bytes = self._release_decision_file.read_bytes()
            release_decision = PrimaryRouteReleaseDecision.model_validate_json(
                release_decision_bytes,
                strict=True,
            )
        except (OSError, ValueError):
            return False
        return (
            exact_bytes_sha256(qualification_report)
            == self._primary_route.authorization.qualification_report_sha256
            and exact_bytes_sha256(release_decision_bytes)
            == self._primary_route.authorization.release_decision_sha256
            and release_decision.route_id == self._primary_route.route_id
            and release_decision.policy_version == self._primary_route.policy_version
            and release_decision.deployment == self._primary_route.deployment
            and release_decision.qualification_report_ref
            == self._primary_route.authorization.qualification_report_ref
            and release_decision.qualification_report_sha256
            == self._primary_route.authorization.qualification_report_sha256
            and release_decision.primary_binding_sha256
            == self._primary_binding.configuration_sha256
            and release_decision.handler_image_sha256 == self._primary_binding.handler_image_sha256
        )


def load_production_runtime_bootstrap_configuration(
    path: Path,
) -> ProductionRuntimeBootstrapConfiguration:
    """Read one exact mounted configuration document without consulting process globals."""

    checked_path = _checked_artifact_path(path, "production runtime configuration")
    try:
        raw = checked_path.read_bytes()
    except OSError as error:
        raise ProductionBootstrapConfigurationError(
            "cannot read production runtime configuration"
        ) from error
    if not raw:
        raise ProductionBootstrapConfigurationError("production runtime configuration is empty")
    try:
        return ProductionRuntimeBootstrapConfiguration.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise ProductionBootstrapConfigurationError(
            "production runtime configuration does not match the reviewed bootstrap contract"
        ) from error


def _checked_artifact_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{label} must name an existing regular file")
    return path


__all__ = [
    "ExactArtifactPrimaryReleaseVerifier",
    "PrimaryRouteReleaseDecision",
    "ProductionBootstrapConfigurationError",
    "ProductionRuntimeBootstrapConfiguration",
    "load_production_runtime_bootstrap_configuration",
]
