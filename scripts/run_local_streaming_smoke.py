"""Run the executable WP6 local mock streaming smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from robata.contracts.hashing import canonical_json_bytes
from robata.runtime.local_streaming_smoke import (
    DEFAULT_SOURCE_DURATION_MS,
    LocalStreamingSmokeConfig,
    create_manifest_from_repository,
    run_local_streaming_smoke,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("tmp/local-streaming-smoke"))
    parser.add_argument("--source-duration-ms", type=int, default=DEFAULT_SOURCE_DURATION_MS)
    parser.add_argument("--chunk-duration-ms", type=int, default=1_000)
    parser.add_argument("--window-duration-ms", type=int, default=2_000)
    parser.add_argument("--window-hop-ms", type=int, default=1_000)
    parser.add_argument("--allowed-lateness-ms", type=int, default=0)
    parser.add_argument("--ring-capacity-ms", type=int, default=2_000)
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--drain-batch-size", type=int, default=64)
    parser.add_argument("--mock-fixed-latency-ms", type=int, default=5)
    parser.add_argument("--mock-failure-probability-ppm", type=int, default=10_000)
    parser.add_argument("--mock-failure-seed", type=int, default=29)
    parser.add_argument("--mock-retry-limit", type=int, default=2)
    parser.add_argument("--mock-request-timeout-ms", type=int, default=30_000)
    parser.add_argument("--mock-request-limit-per-second", type=int, default=1_000)
    parser.add_argument("--mock-max-batch-size", type=int, default=16)
    parser.add_argument("--incremental-latency-p95-target-ms", type=int, default=5_000)
    parser.add_argument("--incremental-latency-p99-target-ms", type=int, default=15_000)
    parser.add_argument("--sqlite-synchronous", choices=("FULL", "NORMAL"), default="NORMAL")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    repository_root = Path(__file__).resolve().parents[1]
    config = LocalStreamingSmokeConfig(
        source_duration_ms=arguments.source_duration_ms,
        chunk_duration_ms=arguments.chunk_duration_ms,
        window_duration_ms=arguments.window_duration_ms,
        window_hop_ms=arguments.window_hop_ms,
        allowed_lateness_ms=arguments.allowed_lateness_ms,
        ring_capacity_ms=arguments.ring_capacity_ms,
        window_batch_size=arguments.window_batch_size,
        drain_batch_size=arguments.drain_batch_size,
        mock_fixed_latency_ms=arguments.mock_fixed_latency_ms,
        mock_failure_probability_ppm=arguments.mock_failure_probability_ppm,
        mock_failure_seed=arguments.mock_failure_seed,
        mock_retry_limit=arguments.mock_retry_limit,
        mock_request_timeout_ms=arguments.mock_request_timeout_ms,
        mock_request_limit_per_second=arguments.mock_request_limit_per_second,
        mock_max_batch_size=arguments.mock_max_batch_size,
        incremental_latency_p95_target_ms=(arguments.incremental_latency_p95_target_ms),
        incremental_latency_p99_target_ms=(arguments.incremental_latency_p99_target_ms),
        sqlite_synchronous=arguments.sqlite_synchronous,
    )
    manifest = create_manifest_from_repository(
        repository_root=repository_root,
        config=config,
    )
    artifacts = run_local_streaming_smoke(
        manifest=manifest,
        output_root=arguments.output_root,
    )
    print(
        canonical_json_bytes(
            {
                "database_path": str(artifacts.database_path),
                "manifest_path": str(artifacts.manifest_path),
                "report_path": str(artifacts.report_path),
                "report": artifacts.report,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
