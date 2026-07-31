"""Recover staged or pre-0005 PostgreSQL raw-evidence R2 mirrors.

This is an operator command, not a serving worker. It uses the canonical worker
role and an explicit tenant context to resume only deterministic immutable R2
writes. It never rewrites existing PostgreSQL raw evidence or published wire
contracts.
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

from robata.adapters.postgres_authority import PostgresCanonicalAuthority  # noqa: E402
from robata.adapters.postgres_r2_artifacts import PostgresR2ArtifactAuthority  # noqa: E402
from robata.adapters.r2_object_store import create_r2_object_store_from_environment  # noqa: E402
from robata.application.canonical.production_composition import (  # noqa: E402
    CanonicalPostgresRuntimeConfig,
)
from robata.application.canonical.production_runtime import (  # noqa: E402
    CanonicalPostgresCredentials,
    create_canonical_postgres_connection_factory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "resume-staged", "backfill-unmirrored"),
        default="all",
        help="which durable R2 recovery operation to perform",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="maximum receipt or raw-evidence rows per operation",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, environment: Mapping[str, str] | None = None) -> int:
    """Run a bounded tenant-scoped R2 recovery pass without dispatching inference."""

    arguments = _parser().parse_args(argv)
    values = os.environ if environment is None else environment
    if isinstance(arguments.limit, bool) or arguments.limit < 1:
        _write_json({"ok": False, "error": "ValueError", "detail": "limit must be positive"})
        return 1
    try:
        postgres = CanonicalPostgresRuntimeConfig.from_environment(values)
        credentials = CanonicalPostgresCredentials.from_environment(
            values,
            variable="CANONICAL_POSTGRES_WORKER_PASSWORD",
        )
        tenant_id = _required(values, "ROBATA_TENANT_ID")
        authority = PostgresCanonicalAuthority(
            create_canonical_postgres_connection_factory(postgres.worker, credentials),
            schema=postgres.schema_name,
            tenant_setting=postgres.tenant_context_setting,
            tenant_id=tenant_id,
        )
        artifacts = PostgresR2ArtifactAuthority(
            authority,
            create_r2_object_store_from_environment(values),
            tenant_id=tenant_id,
        )
        artifacts.verify_startup()
        resumed = ()
        backfilled = ()
        if arguments.mode in {"all", "resume-staged"}:
            resumed = artifacts.reconcile_staged(limit=arguments.limit)
        if arguments.mode in {"all", "backfill-unmirrored"}:
            backfilled = artifacts.backfill_unmirrored_raw_provider_responses(limit=arguments.limit)
        _write_json(
            {
                "ok": True,
                "mode": arguments.mode,
                "tenant_id": tenant_id,
                "resumed_count": len(resumed),
                "backfilled_count": len(backfilled),
                "artifact_ids": sorted(
                    {receipt.artifact_id for receipt in (*resumed, *backfilled)}
                ),
            }
        )
    except Exception as error:
        _write_json(
            {
                "ok": False,
                "error": type(error).__name__,
                "detail": "R2 raw-evidence recovery did not verify",
            }
        )
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
