"""Tests for the fail-closed PostgreSQL canonical production boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from robata.adapters.pgvector_runtime import PgVectorConnectionConfig, PgVectorRuntimeConfig
from robata.adapters.r2_object_store import R2ObjectStoreConfig
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.application.canonical.production_composition import (
    CanonicalPostgresConnectionConfig,
    CanonicalPostgresRuntimeConfig,
    ProductionCompositionContract,
    ProductionCompositionError,
    ProductionCompositionErrorCode,
    ProductionPrimaryRunPodBinding,
)
from robata.application.canonical.production_routing import (
    ModelDeployment,
    ProductionRoute,
    ProductionRouteAuthorization,
    endpoint_config_digest,
)
from robata.contracts.retrieval import VectorBackend
from robata.inference.runpod import RunPodDeploymentConfiguration, RunPodEndpointConfig


def _postgres_connection(*, user: str, application_name: str) -> CanonicalPostgresConnectionConfig:
    return CanonicalPostgresConnectionConfig(
        host="db.example.test",
        database="robata",
        user=user,
        port=5432,
        sslmode="verify-full",
        sslrootcert="/etc/ssl/certs/robata-ca.pem",
        connect_timeout_seconds=10,
        application_name=application_name,
    )


def _postgres_config() -> CanonicalPostgresRuntimeConfig:
    return CanonicalPostgresRuntimeConfig(
        application=_postgres_connection(user="robata_api", application_name="robata-api"),
        worker=_postgres_connection(user="robata_worker", application_name="robata-worker"),
    )


def _pgvector_config() -> PgVectorRuntimeConfig:
    connection = PgVectorConnectionConfig(
        host="db.example.test",
        database="robata",
        user="robata_vector_api",
        sslmode="verify-full",
        sslrootcert="/etc/ssl/certs/robata-ca.pem",
        application_name="robata-vector-api",
    )
    worker = connection.model_copy(update={"user": "robata_vector_worker"})
    return PgVectorRuntimeConfig(
        primary=connection,
        worker=worker,
        worker_role="robata_vector_worker",
        dimension=3,
        backend=VectorBackend.POSTGRES,
    )


def _endpoint() -> RunPodEndpointConfig:
    return RunPodEndpointConfig(
        provider="runpod",
        endpoint_url="https://api.runpod.test/v2/mage-4b/runsync",
        adapter_version="runpod-adapter-v1",
        deployment_configuration=RunPodDeploymentConfiguration(
            model_identifier="mage-vl-4b",
            model_version="1.0",
            inference_engine="vllm",
            precision_or_quantization="bf16",
            topology="TWO_SINGLE_CARD_REPLICAS",
            max_output_tokens=1024,
            supported_topologies=("TWO_SINGLE_CARD_REPLICAS",),
        ),
        max_concurrent_requests=2,
    )


def _contract() -> ProductionCompositionContract:
    endpoint = _endpoint()
    return ProductionCompositionContract(
        canonical_postgres=_postgres_config(),
        r2=R2ObjectStoreConfig(
            endpoint_url="https://account-id.r2.cloudflarestorage.com",
            bucket="robata-production",
            prefix="artifacts",
        ),
        pgvector=_pgvector_config(),
        primary_runpod=ProductionPrimaryRunPodBinding(
            endpoint=endpoint,
            handler_image_sha256="a" * 64,
            capability_snapshot_sha256="b" * 64,
        ),
    )


def _route(contract: ProductionCompositionContract) -> ProductionRoute:
    binding = contract.primary_runpod
    deployment_configuration = binding.endpoint.deployment_configuration
    assert deployment_configuration is not None
    return ProductionRoute(
        route_id="production-mage-primary",
        policy_version="1.0",
        deployment=ModelDeployment(
            deployment_id="mage-primary",
            provider="runpod",
            model_name=deployment_configuration.model_identifier,
            model_version=deployment_configuration.model_version,
            adapter_version=binding.endpoint.adapter_version,
            capability_snapshot_id="11111111-1111-4111-8111-111111111111",
            capability_snapshot_digest=binding.capability_snapshot_sha256,
            endpoint_config_digest=endpoint_config_digest(binding.endpoint),
            max_concurrent_requests=binding.endpoint.max_concurrent_requests,
        ),
        authorization=ProductionRouteAuthorization(
            qualification_report_ref="reports/mage.json",
            qualification_report_sha256="c" * 64,
            release_decision_ref="releases/mage.json",
            release_decision_sha256="d" * 64,
        ),
    )


class _PostgresAuthority:
    backend_kind = "POSTGRESQL"

    def __init__(self) -> None:
        self.verified = False

    def verify_startup(self) -> None:
        self.verified = True


class _ReleaseVerifier:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls = 0

    def verify_primary(self, *, authorization: object, deployment: object) -> bool:
        del authorization, deployment
        self.calls += 1
        return self.result


def test_declared_composition_is_explicitly_not_production_eligible() -> None:
    contract = _contract()

    readiness = contract.readiness

    assert readiness.state == "DECLARED_NOT_RUNNABLE"
    assert readiness.canonical_authority_backend == "POSTGRESQL"
    assert readiness.production_eligible is False
    assert readiness.primary_route_release_verifier == "REQUIRED"
    assert "password" not in str(readiness.model_dump(mode="json")).lower()


def test_environment_requires_tls_rls_and_distinct_canonical_roles() -> None:
    environment = {
        "CANONICAL_POSTGRES_HOST": "db.example.test",
        "CANONICAL_POSTGRES_DATABASE": "robata",
        "CANONICAL_POSTGRES_APP_USER": "robata_api",
        "CANONICAL_POSTGRES_APP_PASSWORD": "app-secret",
        "CANONICAL_POSTGRES_WORKER_USER": "robata_worker",
        "CANONICAL_POSTGRES_WORKER_PASSWORD": "worker-secret",
        "CANONICAL_POSTGRES_SSLROOTCERT": "/etc/ssl/certs/robata-ca.pem",
    }

    config = CanonicalPostgresRuntimeConfig.from_environment(environment)

    assert config.application.user != config.worker.user
    assert config.require_rls is True
    assert "app-secret" not in str(config.model_dump(mode="json"))
    assert "worker-secret" not in str(config.model_dump(mode="json"))
    with pytest.raises(ValueError, match="cannot disable RLS"):
        CanonicalPostgresRuntimeConfig.from_environment(
            {**environment, "CANONICAL_POSTGRES_REQUIRE_RLS": "false"}
        )


def test_startup_rejects_local_authority_before_route_checks(tmp_path: Path) -> None:
    contract = _contract()
    local_authority = SQLiteWorkScheduler(tmp_path / "local-work.sqlite3")

    with pytest.raises(ProductionCompositionError) as error:
        contract.require_authoritative_startup(canonical_authority=local_authority)

    assert error.value.code is ProductionCompositionErrorCode.LOCAL_CANONICAL_AUTHORITY_FORBIDDEN


def test_startup_requires_postgres_authority_and_matching_route() -> None:
    contract = _contract()
    authority = _PostgresAuthority()

    with pytest.raises(ProductionCompositionError) as missing_route:
        contract.require_authoritative_startup(canonical_authority=authority)

    assert authority.verified is True
    assert missing_route.value.code is ProductionCompositionErrorCode.MISSING_PRIMARY_ROUTE

    with pytest.raises(ProductionCompositionError) as mismatch:
        contract.require_authoritative_startup(
            canonical_authority=authority,
            primary_route=_route(contract).model_copy(
                update={
                    "deployment": _route(contract).deployment.model_copy(
                        update={"model_name": "wrong"}
                    )
                }
            ),
        )

    assert mismatch.value.code is ProductionCompositionErrorCode.PRIMARY_ROUTE_MISMATCH


def test_startup_requires_independent_release_verification_then_stays_fail_closed() -> None:
    contract = _contract()
    authority = _PostgresAuthority()
    route = _route(contract)

    with pytest.raises(ProductionCompositionError) as missing_verifier:
        contract.require_authoritative_startup(
            canonical_authority=authority,
            primary_route=route,
        )
    assert (
        missing_verifier.value.code
        is ProductionCompositionErrorCode.MISSING_PRIMARY_ROUTE_RELEASE_VERIFIER
    )

    verifier = _ReleaseVerifier(False)
    with pytest.raises(ProductionCompositionError) as unverified:
        contract.require_authoritative_startup(
            canonical_authority=authority,
            primary_route=route,
            release_verifier=verifier,
        )
    assert verifier.calls == 1
    assert (
        unverified.value.code is ProductionCompositionErrorCode.PRIMARY_ROUTE_RELEASE_NOT_VERIFIED
    )

    verifier = _ReleaseVerifier(True)
    with pytest.raises(ProductionCompositionError) as unimplemented:
        contract.require_authoritative_startup(
            canonical_authority=authority,
            primary_route=route,
            release_verifier=verifier,
        )
    assert (
        unimplemented.value.code is ProductionCompositionErrorCode.CANONICAL_ADAPTERS_UNIMPLEMENTED
    )
