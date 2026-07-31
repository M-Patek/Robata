"""Apply or verify the immutable Robata canonical PostgreSQL migration set.

This command intentionally uses a dedicated migrator login.  Runtime app and
worker credentials are not accepted as a substitute for a deployment role.
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

from robata.adapters.postgres_migrations import PostgresMigrationRunner  # noqa: E402
from robata.application.canonical.production_composition import (  # noqa: E402
    CanonicalPostgresConnectionConfig,
)
from robata.application.canonical.production_runtime import (  # noqa: E402
    CanonicalPostgresCredentials,
    create_canonical_postgres_connection_factory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify exact applied migration bytes instead of applying missing migrations",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, environment: Mapping[str, str] | None = None) -> int:
    """Run the dedicated migration role against an explicit environment mapping."""

    args = _parser().parse_args(argv)
    values = os.environ if environment is None else environment
    try:
        connection = _migrator_connection(values)
        credentials = CanonicalPostgresCredentials.from_environment(
            values,
            variable="CANONICAL_POSTGRES_MIGRATOR_PASSWORD",
        )
        runner = PostgresMigrationRunner(
            create_canonical_postgres_connection_factory(connection, credentials),
            REPOSITORY_ROOT / "db" / "migrations",
        )
        if args.verify:
            verified = runner.verify()
            _write_json({"ok": True, "mode": "VERIFY", "migration_ids": verified})
        else:
            result = runner.apply()
            _write_json(
                {
                    "ok": True,
                    "mode": "APPLY",
                    "applied_ids": result.applied_ids,
                    "already_applied_ids": result.already_applied_ids,
                }
            )
    except Exception as error:
        _write_json({"ok": False, "error": type(error).__name__, "detail": str(error)})
        return 1
    return 0


def _migrator_connection(environment: Mapping[str, str]) -> CanonicalPostgresConnectionConfig:
    return CanonicalPostgresConnectionConfig(
        host=_required(environment, "CANONICAL_POSTGRES_HOST"),
        database=_required(environment, "CANONICAL_POSTGRES_DATABASE"),
        user=_required(environment, "CANONICAL_POSTGRES_MIGRATOR_USER"),
        port=_positive_int(environment, "CANONICAL_POSTGRES_PORT", 5432),
        sslmode=_verify_full(environment, "CANONICAL_POSTGRES_SSLMODE"),
        sslrootcert=_required(environment, "CANONICAL_POSTGRES_SSLROOTCERT"),
        connect_timeout_seconds=_positive_int(
            environment,
            "CANONICAL_POSTGRES_CONNECT_TIMEOUT_SECONDS",
            10,
        ),
        application_name=environment.get(
            "CANONICAL_POSTGRES_MIGRATOR_APPLICATION_NAME",
            "robata-canonical-migrator",
        ),
    )


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _verify_full(environment: Mapping[str, str], name: str) -> Literal["verify-full"]:
    value = environment.get(name, "verify-full")
    if value != "verify-full":
        raise ValueError(f"{name} must be verify-full")
    return "verify-full"


def _write_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
