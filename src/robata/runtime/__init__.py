"""Runtime evidence and benchmark helpers for local execution."""

from robata.runtime.benchmark import (
    BenchmarkSummary,
    ThroughputSample,
    measure_callable,
    run_repeated,
    summarize_samples,
)
from robata.runtime.execution import (
    EXECUTION_AUDIT_FILENAME,
    EXECUTION_MANIFEST_FILENAME,
    EXECUTION_MODE,
    EXECUTION_SCHEMA_VERSION,
    ExecutionEvidenceError,
    PublishedExecutionEvidence,
    build_execution_manifest,
    execution_manifest_semantic_sha256,
    verify_execution_evidence,
    write_execution_evidence,
)
from robata.runtime.preflight import EXPECTED_EXECUTION_SPEC_SHA256, REQUIRED_IMPORTS, run_preflight
from robata.runtime.verification import LocalMainlineVerificationError, verify_local_mainline_output

__all__ = [
    "EXECUTION_AUDIT_FILENAME",
    "EXECUTION_MANIFEST_FILENAME",
    "EXECUTION_MODE",
    "EXECUTION_SCHEMA_VERSION",
    "EXPECTED_EXECUTION_SPEC_SHA256",
    "REQUIRED_IMPORTS",
    "BenchmarkSummary",
    "ExecutionEvidenceError",
    "LocalMainlineVerificationError",
    "PublishedExecutionEvidence",
    "ThroughputSample",
    "build_execution_manifest",
    "execution_manifest_semantic_sha256",
    "measure_callable",
    "run_preflight",
    "run_repeated",
    "summarize_samples",
    "verify_execution_evidence",
    "verify_local_mainline_output",
    "write_execution_evidence",
]
