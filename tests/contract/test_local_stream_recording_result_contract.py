from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from robata.application.canonical.stream_recording_reduction import (
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION,
    LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION,
    LocalStreamRecordingResult,
    LocalStreamRecordingResultV2,
    create_local_stream_recording_result,
    create_local_stream_recording_result_v2,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from scripts.export_persisted_wire_schema import export_schema
from tests.unit.test_stream_recording_reduction import (
    _inputs,
    _window_semantic_evidence,
)


def _registered_result(
    registry: SchemaRegistry,
) -> LocalStreamRecordingResult:
    registered = registry.resolve_version(
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION,
    )
    window_results, terminal_closure, recording_finalization = _inputs()
    return create_local_stream_recording_result(
        schema_ref=registered.ref,
        window_results=window_results,
        terminal_closure=terminal_closure,
        recording_finalization=recording_finalization,
    )


def _registered_v2_result(
    registry: SchemaRegistry,
) -> LocalStreamRecordingResultV2:
    registered = registry.resolve_version(
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
        LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION,
    )
    window_results, terminal_closure, recording_finalization = _inputs()
    semantic_evidence = _window_semantic_evidence(
        window_results,
        terminal_closure,
    )
    semantic_schema = registry.resolve_version(
        LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID,
        LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION,
    ).ref
    rebound_evidence = tuple(
        (
            evidence.model_copy(update={"schema_ref": semantic_schema}),
            reference.model_copy(
                update={
                    "schema_ref": semantic_schema,
                    "exact_sha256": exact_bytes_sha256(
                        canonical_json_bytes(
                            evidence.model_copy(update={"schema_ref": semantic_schema})
                        )
                    ),
                    "byte_count": len(
                        canonical_json_bytes(
                            evidence.model_copy(update={"schema_ref": semantic_schema})
                        )
                    ),
                }
            ),
        )
        for evidence, reference in semantic_evidence
    )
    return create_local_stream_recording_result_v2(
        schema_ref=registered.ref,
        window_results=window_results,
        window_semantic_evidence=rebound_evidence,
        terminal_closure=terminal_closure,
        recording_finalization=recording_finalization,
    )


def _validate_registered_result(
    registry: SchemaRegistry,
    payload: dict[str, object],
) -> LocalStreamRecordingResult:
    registered = registry.resolve_version(
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION,
    )
    registry.validate_pinned(registered.ref, payload)
    return LocalStreamRecordingResult.model_validate_json(
        canonical_json_bytes(payload),
        strict=True,
    )


def test_registered_schema_exactly_matches_pydantic_export(tmp_path: Path) -> None:
    registry = SchemaRegistry()
    registered = registry.resolve_version(
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_VERSION,
    )
    exported = tmp_path / "local-stream-recording-result.schema.json"

    export_schema("local-stream-recording-result", exported)

    assert registered.document_bytes == registered.path.read_bytes()
    assert registered.document_bytes == exported.read_bytes()
    assert registered.ref.sha256 == exact_bytes_sha256(registered.document_bytes)


def test_registered_v2_schemas_exactly_match_pydantic_exports(tmp_path: Path) -> None:
    registry = SchemaRegistry()
    cases = (
        (
            LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_ID,
            LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_SCHEMA_VERSION,
            "local-stream-window-semantic-evidence",
            tmp_path / "local-stream-window-semantic-evidence.schema.json",
        ),
        (
            LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
            LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION,
            "local-stream-recording-result-v2",
            tmp_path / "local-stream-recording-result-v2.schema.json",
        ),
    )

    for schema_id, version, export_name, export_path in cases:
        registered = registry.resolve_version(schema_id, version)
        export_schema(export_name, export_path)

        assert registered.document_bytes == registered.path.read_bytes()
        assert registered.document_bytes == export_path.read_bytes()
        assert registered.ref.sha256 == exact_bytes_sha256(registered.document_bytes)


def test_registered_validation_accepts_reducer_result_and_rejects_tampered_digest() -> None:
    registry = SchemaRegistry()
    result = _registered_result(registry)
    payload = result.model_dump(mode="json")

    assert _validate_registered_result(registry, payload) == result

    tampered = deepcopy(payload)
    tampered["recording_result_semantic_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="recording_result_semantic_sha256"):
        _validate_registered_result(registry, tampered)


def test_registered_v2_validation_rejects_tampered_recording_digest() -> None:
    registry = SchemaRegistry()
    result = _registered_v2_result(registry)
    payload = result.model_dump(mode="json")
    registered = registry.resolve_version(
        LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
        LOCAL_STREAM_RECORDING_RESULT_V2_SCHEMA_VERSION,
    )

    registry.validate_pinned(registered.ref, payload)
    assert (
        LocalStreamRecordingResultV2.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
        == result
    )

    tampered = deepcopy(payload)
    tampered["recording_result_semantic_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="recording_result_semantic_sha256"):
        LocalStreamRecordingResultV2.model_validate_json(
            canonical_json_bytes(tampered),
            strict=True,
        )
