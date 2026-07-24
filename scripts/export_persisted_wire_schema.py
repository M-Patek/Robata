"""Export deterministic JSON Schema candidates for registered persisted wires."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True, slots=True)
class WireSchemaSpec:
    module: str
    model: str
    document_id: str
    title: str


WIRE_SCHEMAS = {
    "local-stream-window-inference-plan": WireSchemaSpec(
        module="robata.contracts.local_stream_causal",
        model="LocalStreamWindowInferencePlan",
        document_id=(
            "https://schemas.robata.dev/v1/local-stream-window-inference-plan.schema.json"
        ),
        title="LocalStreamWindowInferencePlan",
    ),
    "local-stream-window-semantic-evidence-v2": WireSchemaSpec(
        module="robata.contracts.local_stream_causal",
        model="LocalStreamWindowSemanticEvidenceV2",
        document_id=(
            "https://schemas.robata.dev/v2/local-stream-window-semantic-evidence.schema.json"
        ),
        title="LocalStreamWindowSemanticEvidenceV2",
    ),
    "local-stream-window-semantic-evidence": WireSchemaSpec(
        module="robata.application.canonical.stream_recording_reduction",
        model="LocalStreamWindowSemanticEvidence",
        document_id=(
            "https://schemas.robata.dev/v1/local-stream-window-semantic-evidence.schema.json"
        ),
        title="LocalStreamWindowSemanticEvidence",
    ),
    "local-stream-recording-result-v2": WireSchemaSpec(
        module="robata.application.canonical.stream_recording_reduction",
        model="LocalStreamRecordingResultV2",
        document_id=("https://schemas.robata.dev/v2/local-stream-recording-result.schema.json"),
        title="LocalStreamRecordingResultV2",
    ),
    "local-stream-recording-result-v3": WireSchemaSpec(
        module="robata.application.canonical.stream_recording_reduction",
        model="LocalStreamRecordingResultV3",
        document_id=("https://schemas.robata.dev/v3/local-stream-recording-result.schema.json"),
        title="LocalStreamRecordingResultV3",
    ),
    "local-stream-recording-result-v4": WireSchemaSpec(
        module="robata.application.canonical.stream_recording_reduction",
        model="LocalStreamRecordingResultV4",
        document_id=("https://schemas.robata.dev/v4/local-stream-recording-result.schema.json"),
        title="LocalStreamRecordingResultV4",
    ),
    "pre-eos-capture-subject": WireSchemaSpec(
        module="robata.contracts.stream_source",
        model="PreEosCaptureSubject",
        document_id="https://schemas.robata.dev/v1/pre-eos-capture-subject.schema.json",
        title="PreEosCaptureSubject",
    ),
    "stream-segment": WireSchemaSpec(
        module="robata.contracts.stream_source",
        model="StreamSegmentManifest",
        document_id="https://schemas.robata.dev/v1/stream-segment.schema.json",
        title="StreamSegmentManifest",
    ),
    "incremental-window": WireSchemaSpec(
        module="robata.contracts.stream_window",
        model="IncrementalWindow",
        document_id="https://schemas.robata.dev/v1/incremental-window.schema.json",
        title="IncrementalWindow",
    ),
    "stream-inference": WireSchemaSpec(
        module="robata.contracts.stream_window",
        model="StreamInferenceLogicalIdentity",
        document_id="https://schemas.robata.dev/v1/stream-inference.schema.json",
        title="StreamInferenceLogicalIdentity",
    ),
    "stream-inference-attempt": WireSchemaSpec(
        module="robata.contracts.stream_window",
        model="StreamInferenceAttemptIdentity",
        document_id="https://schemas.robata.dev/v1/stream-inference-attempt.schema.json",
        title="StreamInferenceAttemptIdentity",
    ),
    "stream-inference-intent": WireSchemaSpec(
        module="robata.contracts.stream_inference",
        model="StreamInferenceIntent",
        document_id="https://schemas.robata.dev/v1/stream-inference-intent.schema.json",
        title="StreamInferenceIntent",
    ),
    "stream-accepted-call-evidence": WireSchemaSpec(
        module="robata.contracts.stream_inference",
        model="StreamAcceptedCallEvidence",
        document_id=("https://schemas.robata.dev/v1/stream-accepted-call-evidence.schema.json"),
        title="StreamAcceptedCallEvidence",
    ),
    "stream-inference-terminal": WireSchemaSpec(
        module="robata.contracts.stream_inference",
        model="StreamInferenceTerminal",
        document_id="https://schemas.robata.dev/v1/stream-inference-terminal.schema.json",
        title="StreamInferenceTerminal",
    ),
    "stream-window-result": WireSchemaSpec(
        module="robata.contracts.stream_inference",
        model="StreamWindowResult",
        document_id="https://schemas.robata.dev/v1/stream-window-result.schema.json",
        title="StreamWindowResult",
    ),
    "stream-work-plan": WireSchemaSpec(
        module="robata.contracts.stream_planning",
        model="StreamWorkItemPlan",
        document_id="https://schemas.robata.dev/v1/stream-work-plan.schema.json",
        title="StreamWorkItemPlan",
    ),
    "stream-work-message": WireSchemaSpec(
        module="robata.queue.stream_wire",
        model="StreamWorkMessage",
        document_id="https://schemas.robata.dev/v1/stream-work-message.schema.json",
        title="StreamWorkMessage",
    ),
    "expected-window-plan": WireSchemaSpec(
        module="robata.contracts.stream_planning",
        model="ExpectedWindowPlan",
        document_id="https://schemas.robata.dev/v1/expected-window-plan.schema.json",
        title="ExpectedWindowPlan",
    ),
    "expected-window-declaration": WireSchemaSpec(
        module="robata.contracts.stream_planning",
        model="ExpectedWindowDeclaration",
        document_id=("https://schemas.robata.dev/v1/expected-window-declaration.schema.json"),
        title="ExpectedWindowDeclaration",
    ),
    "expected-window-plan-seal": WireSchemaSpec(
        module="robata.contracts.stream_planning",
        model="ExpectedWindowPlanSeal",
        document_id="https://schemas.robata.dev/v1/expected-window-plan-seal.schema.json",
        title="ExpectedWindowPlanSeal",
    ),
    "window-terminal-member": WireSchemaSpec(
        module="robata.contracts.stream_finalization",
        model="WindowTerminalMember",
        document_id="https://schemas.robata.dev/v1/window-terminal-member.schema.json",
        title="WindowTerminalMember",
    ),
    "window-terminal-closure": WireSchemaSpec(
        module="robata.contracts.stream_finalization",
        model="WindowTerminalClosure",
        document_id="https://schemas.robata.dev/v1/window-terminal-closure.schema.json",
        title="WindowTerminalClosure",
    ),
    "recording-finalization-map": WireSchemaSpec(
        module="robata.contracts.stream_finalization",
        model="RecordingFinalizationMap",
        document_id="https://schemas.robata.dev/v1/recording-finalization-map.schema.json",
        title="RecordingFinalizationMap",
    ),
    "local-stream-recording-result": WireSchemaSpec(
        module="robata.application.canonical.stream_recording_reduction",
        model="LocalStreamRecordingResult",
        document_id=("https://schemas.robata.dev/v1/local-stream-recording-result.schema.json"),
        title="LocalStreamRecordingResult",
    ),
    "artifact-registry-entry-v3": WireSchemaSpec(
        module="robata.contracts.artifacts_v3",
        model="ArtifactRegistryEntryV3",
        document_id="https://schemas.robata.dev/v3/artifact-registry-entry.schema.json",
        title="ArtifactRegistryEntryV3",
    ),
    "artifact-registry-snapshot-v3": WireSchemaSpec(
        module="robata.contracts.artifacts_v3",
        model="ArtifactRegistrySnapshotV3",
        document_id="https://schemas.robata.dev/v3/artifact-registry-snapshot.schema.json",
        title="ArtifactRegistrySnapshotV3",
    ),
    "local-supplemental-qa-evidence": WireSchemaSpec(
        module="robata.qa_pipeline.supplemental_wire",
        model="LocalSupplementalQaEvidence",
        document_id=("https://schemas.robata.dev/v2/local-supplemental-qa-evidence.schema.json"),
        title="LocalSupplementalQaEvidence",
    ),
    "event-identity-outbox-record": WireSchemaSpec(
        module="robata.event_pipeline.identity_registry",
        model="EventIdentityOutboxWireRecord",
        document_id=("https://schemas.robata.dev/v1/event-identity-outbox-record.schema.json"),
        title="EventIdentityOutboxRecord",
    ),
    "review-task": WireSchemaSpec(
        module="robata.review.models",
        model="ReviewTask",
        document_id="https://schemas.robata.dev/v1/review-task.schema.json",
        title="ReviewTask",
    ),
    "review-annotation": WireSchemaSpec(
        module="robata.review.models",
        model="ReviewAnnotation",
        document_id="https://schemas.robata.dev/v1/review-annotation.schema.json",
        title="ReviewAnnotation",
    ),
    "review-reopen-command": WireSchemaSpec(
        module="robata.review.models",
        model="ReviewReopenCommand",
        document_id="https://schemas.robata.dev/v1/review-reopen-command.schema.json",
        title="ReviewReopenCommand",
    ),
}


def _close_model_objects(value: Any) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["additionalProperties"] = False
            value["required"] = list(properties)
        for nested in value.values():
            _close_model_objects(nested)
    elif isinstance(value, list):
        for nested in value:
            _close_model_objects(nested)


def export_schema(name: str, output: Path) -> None:
    spec = WIRE_SCHEMAS[name]
    module = importlib.import_module(spec.module)
    model = getattr(module, spec.model)
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError(f"{spec.module}.{spec.model} is not a Pydantic model")

    document = model.model_json_schema(mode="serialization")
    _close_model_objects(document)
    document["$schema"] = JSON_SCHEMA_DIALECT
    document["$id"] = spec.document_id
    document["title"] = spec.title
    encoded = (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=tuple(WIRE_SCHEMAS))
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    export_schema(arguments.name, arguments.output)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
