"""Validate explicitly configured R2 and pgvector adapters before composition."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.adapters.pgvector_runtime import (  # noqa: E402
    PgVectorCredentials,
    PgVectorRuntimeConfig,
    create_pgvector_projection_store,
    create_verified_pgvector_projection_store,
)
from robata.adapters.r2_object_store import (  # noqa: E402
    R2Credentials,
    R2ObjectStoreConfig,
    create_boto3_r2_client,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402
from robata.ports.object_storage import ObjectStoreError  # noqa: E402
from robata.ports.vector_projection import VectorProjectionError  # noqa: E402

_SECRET_ENVIRONMENT_KEYS = (
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "PGVECTOR_PASSWORD",
    "PGVECTOR_WORKER_PASSWORD",
)
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2", action="store_true", help="validate and construct the R2 S3 client")
    parser.add_argument(
        "--pgvector", action="store_true", help="validate and construct the pgvector store"
    )
    parser.add_argument(
        "--verify-pgvector",
        action="store_true",
        help="connect to pgvector and verify the reviewed backend/RLS deployment",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        metavar="PATH",
        help="read explicit local KEY=VALUE configuration without loading it into the process",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run selected checks against one explicit environment mapping.

    R2 construction checks configuration and the optional SDK only; it does not
    write, read, or reconcile an object. pgvector remains lazy unless the caller
    opts into ``--verify-pgvector``, which performs target database I/O.
    """

    args = _parser().parse_args(argv)
    base_values = os.environ if environment is None else environment
    values: Mapping[str, str] = base_values
    try:
        values = _load_environment_file(base_values, args.env_file)
        if args.verify_pgvector and not args.pgvector:
            return _write_failure(
                "INVALID_REQUEST", "--verify-pgvector requires --pgvector", values
            )
        if not args.r2 and not args.pgvector:
            return _write_failure(
                "INVALID_REQUEST", "select at least one of --r2 or --pgvector", values
            )
        checks = _run_checks(
            values,
            include_r2=args.r2,
            include_pgvector=args.pgvector,
            verify_pgvector=args.verify_pgvector,
        )
    except (ObjectStoreError, VectorProjectionError) as error:
        return _write_failure(error.code.value, str(error), values)
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        return _write_failure("INVALID_CONFIGURATION", str(error), values)
    except Exception:
        return _write_failure(
            "PREFLIGHT_FAILED", "optional adapter preflight raised an unexpected error", values
        )
    _write_json(
        {
            "ok": True,
            "checks": checks,
            "qualification_status": "NOT_MEASURED",
            "production_eligible": False,
        }
    )
    return 0


def _load_environment_file(
    base_values: Mapping[str, str], environment_file: Path | None
) -> dict[str, str]:
    """Merge a deliberately selected local env file without mutating process state."""

    values = dict(base_values)
    if environment_file is None:
        return values
    for line_number, raw_line in enumerate(
        environment_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw_value = raw_line.partition("=")
        key = key.strip()
        if not separator or _ENVIRONMENT_KEY.fullmatch(key) is None:
            raise ValueError(
                f"environment file has an invalid KEY=VALUE assignment at line {line_number}"
            )
        values[key] = _parse_environment_value(raw_value.strip(), line_number)
    return values


def _parse_environment_value(value: str, line_number: int) -> str:
    if not value:
        return value
    if value[0] not in {"'", '"'}:
        return value
    if len(value) < 2 or value[-1] != value[0]:
        raise ValueError(f"environment file has an unclosed quoted value at line {line_number}")
    return value[1:-1]


def _run_checks(
    environment: Mapping[str, str],
    *,
    include_r2: bool,
    include_pgvector: bool,
    verify_pgvector: bool,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if include_r2:
        r2_config = R2ObjectStoreConfig.from_environment(environment)
        create_boto3_r2_client(r2_config, R2Credentials.from_environment(environment))
        checks.append(
            {
                "adapter": "r2",
                "state": "CONFIGURED",
                "endpoint_url": r2_config.endpoint_url,
                "bucket": r2_config.bucket,
                "prefix": r2_config.normalized_prefix,
            }
        )
    if include_pgvector:
        pgvector_config = PgVectorRuntimeConfig.from_environment(environment)
        primary_credentials = PgVectorCredentials.from_environment(
            environment, variable="PGVECTOR_PASSWORD"
        )
        worker_credentials = PgVectorCredentials.from_environment(
            environment, variable="PGVECTOR_WORKER_PASSWORD"
        )
        if verify_pgvector:
            create_verified_pgvector_projection_store(
                pgvector_config,
                primary_credentials,
                worker_credentials=worker_credentials,
            )
            state = "VERIFIED"
        else:
            create_pgvector_projection_store(
                pgvector_config,
                primary_credentials,
                worker_credentials=worker_credentials,
            )
            state = "CONFIGURED"
        checks.append(
            {
                "adapter": "pgvector",
                "state": state,
                "backend": pgvector_config.backend.value,
                "relation": pgvector_config.relation,
                "dimension": pgvector_config.dimension,
                "require_rls": True,
            }
        )
    return checks


def _write_failure(code: str, detail: str, environment: Mapping[str, str]) -> int:
    _write_json(
        {"ok": False, "code": code, "detail": _redact(detail, environment)},
        stream=sys.stderr,
    )
    return 2


def _redact(detail: str, environment: Mapping[str, str]) -> str:
    redacted = detail
    for key in _SECRET_ENVIRONMENT_KEYS:
        value = environment.get(key)
        if value:
            redacted = redacted.replace(value, "REDACTED")
    return redacted


def _write_json(payload: object, *, stream: TextIO | None = None) -> None:
    print(
        canonical_json_bytes(payload).decode("utf-8"), file=sys.stdout if stream is None else stream
    )


if __name__ == "__main__":
    raise SystemExit(main())
