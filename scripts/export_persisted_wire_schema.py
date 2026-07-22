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
