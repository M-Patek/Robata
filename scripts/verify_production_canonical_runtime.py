"""Construct and verify Robata's complete production adapter graph without work dispatch.

This is a deployment gate, not a worker loop. It verifies the exact mounted
route/release inputs and constructs the same PostgreSQL/R2/pgvector/RunPod
composition a worker or API process must receive before serving traffic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.adapters.pgvector_runtime import (  # noqa: E402
    PgVectorRuntimeConfig,
    create_verified_pgvector_projection_store_from_environment,
)
from robata.adapters.postgres_completion_evidence import (  # noqa: E402
    PostgresInferenceEvidenceLedger,
)
from robata.adapters.r2_object_store import create_r2_object_store_from_environment  # noqa: E402
from robata.application.canonical.production_bootstrap import (  # noqa: E402
    load_production_runtime_bootstrap_configuration,
)
from robata.application.canonical.production_composition import (  # noqa: E402
    CanonicalPostgresRuntimeConfig,
    ProductionCompositionContract,
)
from robata.application.canonical.production_runtime import (  # noqa: E402
    CanonicalPostgresRuntimeCredentials,
    ProductionTenantContext,
    build_production_canonical_runtime,
)
from robata.contracts.schema_registry import default_schema_registry  # noqa: E402
from robata.inference.offline_fixture import StrictProviderClaimParser  # noqa: E402
from robata.inference.runpod import (  # noqa: E402
    RunPodApiKey,
    RunPodVisionAdapter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "mounted production-runtime bootstrap document; defaults to "
            "ROBATA_PRODUCTION_RUNTIME_CONFIG"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None, *, environment: Mapping[str, str] | None = None) -> int:
    """Construct every production adapter once and report no secret material."""

    arguments = _parser().parse_args(argv)
    values = os.environ if environment is None else environment
    try:
        configuration_path = arguments.config or Path(
            _required(values, "ROBATA_PRODUCTION_RUNTIME_CONFIG")
        )
        bootstrap = load_production_runtime_bootstrap_configuration(configuration_path)
        registry = default_schema_registry()
        r2_object_store = create_r2_object_store_from_environment(values)
        contract = ProductionCompositionContract(
            canonical_postgres=CanonicalPostgresRuntimeConfig.from_environment(values),
            r2=r2_object_store.config,
            pgvector=_pgvector_config(values),
            primary_runpod=bootstrap.primary_binding,
        )
        pgvector_projection = create_verified_pgvector_projection_store_from_environment(values)

        def primary_adapter_factory(
            raw_store: PostgresInferenceEvidenceLedger,
        ) -> RunPodVisionAdapter:
            return RunPodVisionAdapter(
                config=bootstrap.primary_binding.endpoint,
                credential=RunPodApiKey(_required(values, "RUNPOD_API_KEY")),
                capabilities=bootstrap.primary_capabilities,
                retry_policy=bootstrap.primary_retry_policy,
                raw_store=raw_store,
                parser=StrictProviderClaimParser(
                    registry,
                    parser_version=bootstrap.primary_parser_version,
                ),
            )

        runtime = build_production_canonical_runtime(
            contract=contract,
            credentials=CanonicalPostgresRuntimeCredentials.from_environment(values),
            tenant=ProductionTenantContext(tenant_id=_required(values, "ROBATA_TENANT_ID")),
            capture_authority=bootstrap.capture_authority,
            r2_object_store=r2_object_store,
            pgvector_projection=pgvector_projection,
            primary_adapter_factory=primary_adapter_factory,
            primary_route=bootstrap.primary_route,
            release_verifier=bootstrap.release_verifier(),
            outbox_retry_policy=bootstrap.outbox_retry_policy(),
            schema_registry=registry,
        )
        _write_json(
            {
                "ok": True,
                "canonical_backend": runtime.worker_authority.backend_kind,
                "canonical_schema": runtime.worker_authority.schema,
                "primary_route_id": runtime.primary_route.route_id,
                "read_model_backend": runtime.read_model.backend_kind,
            }
        )
    except Exception as error:
        _write_json(
            {
                "ok": False,
                "error": type(error).__name__,
                "detail": "production canonical runtime gate did not verify",
            }
        )
        return 1
    return 0


def _pgvector_config(environment: Mapping[str, str]) -> PgVectorRuntimeConfig:
    """Parse the reviewed pgvector target configuration without opening a connection."""

    return PgVectorRuntimeConfig.from_environment(environment)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _write_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
