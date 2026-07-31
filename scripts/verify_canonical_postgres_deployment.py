"""Verify the deployed Robata canonical PostgreSQL roles, migrations, and RLS.

The command makes database reads only. It does not invoke RunPod, R2, or a
worker loop; use it after migrations and role grants, before admitting work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.adapters.postgres_authority import PostgresCanonicalAuthority  # noqa: E402
from robata.adapters.postgres_migrations import PostgresMigrationRunner  # noqa: E402
from robata.application.canonical.production_composition import (  # noqa: E402
    CanonicalPostgresRuntimeConfig,
)
from robata.application.canonical.production_runtime import (  # noqa: E402
    CanonicalPostgresRuntimeCredentials,
    ProductionTenantContext,
    _verify_canonical_authority,
    create_canonical_postgres_connection_factory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tenant-id",
        help="tenant used to prove transaction-local RLS enforcement; defaults to ROBATA_TENANT_ID",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, environment: Mapping[str, str] | None = None) -> int:
    """Check both runtime roles without printing credentials."""

    args = _parser().parse_args(argv)
    values = os.environ if environment is None else environment
    try:
        config = CanonicalPostgresRuntimeConfig.from_environment(values)
        credentials = CanonicalPostgresRuntimeCredentials.from_environment(values)
        tenant = ProductionTenantContext(
            tenant_id=args.tenant_id or _required(values, "ROBATA_TENANT_ID")
        )
        app_factory = create_canonical_postgres_connection_factory(
            config.application,
            credentials.application,
        )
        worker_factory = create_canonical_postgres_connection_factory(
            config.worker,
            credentials.worker,
        )
        verified = PostgresMigrationRunner(
            worker_factory,
            REPOSITORY_ROOT / "db" / "migrations",
        ).verify()
        for role, factory in (("application", app_factory), ("worker", worker_factory)):
            authority = PostgresCanonicalAuthority(
                factory,
                schema=config.schema_name,
                tenant_setting=config.tenant_context_setting,
                tenant_id=tenant.tenant_id,
            )
            runtime_role: Literal["application", "worker"] = (
                "application" if role == "application" else "worker"
            )
            _verify_canonical_authority(authority, runtime_role=runtime_role)
            _write_json({"ok": True, "role": role, "migration_ids": verified})
    except Exception as error:
        _write_json({"ok": False, "error": type(error).__name__, "detail": str(error)})
        return 1
    return 0


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _write_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
