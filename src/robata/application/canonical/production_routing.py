"""Closed production-plane route contracts for canonical composition.

This module intentionally owns only the authoritative primary-route identity used
by the PostgreSQL production boundary. Experiment and canary orchestration may
live elsewhere, but cannot become an implicit runtime dependency of the
production composition.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=256)]


class ModelDeployment(StrictModel):
    """Pinned non-secret facts for the single authoritative model deployment."""

    schema_version: Literal["1.0"] = "1.0"
    deployment_id: NonEmptyString
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    capability_snapshot_id: OpaqueUuid
    capability_snapshot_digest: Sha256Digest
    endpoint_config_digest: Sha256Digest
    max_concurrent_requests: PositiveInt


class ProductionRouteAuthorization(StrictModel):
    """Exact evidence references required before a primary route is admitted."""

    qualification_report_ref: NonEmptyString
    qualification_report_sha256: Sha256Digest
    release_decision_ref: NonEmptyString
    release_decision_sha256: Sha256Digest


class ProductionRoute(StrictModel):
    """One authoritative route and its independently reviewed evidence binding."""

    schema_version: Literal["1.0"] = "1.0"
    route_id: NonEmptyString
    policy_version: SchemaVersion
    deployment: ModelDeployment
    authorization: ProductionRouteAuthorization

    @property
    def configuration_digest(self) -> Sha256Digest:
        return semantic_sha256(self.model_dump(mode="json"))


def endpoint_config_digest(endpoint_config: StrictModel) -> Sha256Digest:
    """Return the stable digest that binds a route deployment to its endpoint."""

    if not isinstance(endpoint_config, StrictModel):
        raise TypeError("endpoint_config must be a StrictModel")
    return semantic_sha256(endpoint_config.model_dump(mode="json"))


__all__ = [
    "ModelDeployment",
    "ProductionRoute",
    "ProductionRouteAuthorization",
    "endpoint_config_digest",
]
