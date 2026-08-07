"""Validate or explicitly stage one pinned R2 MCAP for the canonical CLI."""

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

from robata.adapters.r2_object_store import (  # noqa: E402
    R2Credentials,
    R2ObjectStoreConfig,
    create_r2_object_store_from_environment,
)
from robata.application.canonical.r2_mcap_staging import (  # noqa: E402
    R2McapSourceStagingError,
    load_r2_mcap_source_manifest,
    r2_mcap_source_manifest_projection,
    stage_r2_mcap_source,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402
from robata.ports.object_storage import ObjectStoreError  # noqa: E402

_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEYS = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        metavar="PATH",
        help="exact-canonical robata-r2-mcap-source-v1 JSON manifest",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        metavar="PATH",
        help="local MCAP path for the existing run_canonical_mcap.py command",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="explicitly read and integrity-verify the R2 object before local publication",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        metavar="PATH",
        help="read explicit KEY=VALUE configuration without modifying process environment",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run a no-network launch-input check, or explicitly stage one R2 source.

    Without ``--stage`` this command validates the source manifest and R2
    configuration only.  It does not import boto3, open a network connection,
    create the destination directory, or modify local state.  ``--stage`` is
    the sole external-I/O path and performs an R2 HEAD/GET followed by an
    immutable local file publication.
    """

    args = _parser().parse_args(argv)
    values: Mapping[str, str] = os.environ if environment is None else environment
    try:
        resolved_environment = _load_environment_file(values, args.env_file)
        r2_config = R2ObjectStoreConfig.from_environment(resolved_environment)
        # Validate credentials without instantiating boto3 in CONFIG_ONLY mode.
        R2Credentials.from_environment(resolved_environment)
        manifest = load_r2_mcap_source_manifest(args.source_manifest)
        payload: dict[str, object] = {
            "ok": True,
            "mode": "STAGED" if args.stage else "CONFIG_ONLY",
            "external_calls": args.stage,
            "production_eligible": False,
            "r2": {
                "endpoint_url": r2_config.endpoint_url,
                "bucket": r2_config.bucket,
                "prefix": r2_config.normalized_prefix,
            },
            "source": r2_mcap_source_manifest_projection(manifest),
            "destination": str(args.destination),
        }
        if args.stage:
            receipt = stage_r2_mcap_source(
                manifest=manifest,
                object_store=create_r2_object_store_from_environment(resolved_environment),
                destination=args.destination,
            )
            payload["receipt"] = {
                "byte_count": receipt.byte_count,
                "content_sha256": receipt.content_sha256,
                "reused_existing_file": receipt.reused_existing_file,
            }
    except ObjectStoreError as error:
        return _write_failure(
            error.code.value,
            str(error),
            resolved_environment if "resolved_environment" in locals() else values,
        )
    except R2McapSourceStagingError as error:
        return _write_failure(
            "INVALID_SOURCE",
            str(error),
            resolved_environment if "resolved_environment" in locals() else values,
        )
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        return _write_failure(
            "INVALID_CONFIGURATION",
            str(error),
            resolved_environment if "resolved_environment" in locals() else values,
        )
    except Exception:
        return _write_failure(
            "STAGING_FAILED",
            "R2 MCAP staging raised an unexpected error",
            resolved_environment if "resolved_environment" in locals() else values,
        )
    _write_json(payload)
    return 0


def _load_environment_file(
    base_values: Mapping[str, str], environment_file: Path | None
) -> dict[str, str]:
    """Merge one deliberate local env file without writing process globals."""

    values = dict(base_values)
    if environment_file is None:
        return values
    for line_number, raw_line in enumerate(
        environment_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or _ENVIRONMENT_KEY.fullmatch(key) is None:
            raise ValueError(
                f"environment file has an invalid KEY=VALUE assignment at line {line_number}"
            )
        values[key] = _parse_environment_value(raw_value.strip(), line_number)
    return values


def _parse_environment_value(value: str, line_number: int) -> str:
    if not value or value[0] not in {"'", '"'}:
        return value
    if len(value) < 2 or value[-1] != value[0]:
        raise ValueError(f"environment file has an unclosed quoted value at line {line_number}")
    return value[1:-1]


def _write_failure(code: str, detail: str, environment: Mapping[str, str]) -> int:
    _write_json(
        {"ok": False, "code": code, "detail": _redact(detail, environment)}, stream=sys.stderr
    )
    return 2


def _redact(detail: str, environment: Mapping[str, str]) -> str:
    redacted = detail
    for key in _SECRET_KEYS:
        value = environment.get(key)
        if isinstance(value, str) and value:
            redacted = redacted.replace(value, "REDACTED")
    return redacted


def _write_json(payload: object, *, stream: TextIO | None = None) -> None:
    print(
        canonical_json_bytes(payload).decode("utf-8"),
        file=sys.stdout if stream is None else stream,
    )


if __name__ == "__main__":
    raise SystemExit(main())
