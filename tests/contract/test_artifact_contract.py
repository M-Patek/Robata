from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError
from referencing import Registry, Resource

from robata.contracts.artifacts import ArtifactRegistryEntry, ArtifactRegistrySnapshot
from robata.contracts.schema_registry import SchemaValidationError

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "schemas" / "v2"
ENTRY_SCHEMA = "artifact-registry-entry"
SNAPSHOT_SCHEMA = "artifact-registry-snapshot"

MAPPING_SCHEMA_ID = "https://schemas.robata.dev/v2/mapping-profile.schema.json"
CONFIG_SCHEMA_ID = "https://schemas.robata.dev/v2/export-config.schema.json"
TIMESTAMP_SCHEMA_ID = "https://schemas.robata.dev/v1/camera-video-timestamp-row.schema.json"
MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/v1/camera-video-export-manifest.schema.json"

MEDIA_TYPES = {
    "CAMERA_VIDEO_EXPORT_MANIFEST": "application/json",
    "CAMERA_VIDEO_MP4": "video/mp4",
    "CAMERA_VIDEO_TIMESTAMP_MAP": "application/x-ndjson",
    "EXPORT_CONFIG": "application/json",
    "JSON_SCHEMA": "application/schema+json",
    "MAPPING_PROFILE": "application/json",
    "RAW_MCAP": "application/x-mcap",
}


class _ArtifactSchemaRegistry:
    """Load only this independent schema pair while the global V2 set is assembled."""

    def __init__(self) -> None:
        documents: dict[str, dict[str, Any]] = {}
        resources: list[tuple[str, Resource[Any]]] = []
        for schema_name in (ENTRY_SCHEMA, SNAPSHOT_SCHEMA):
            filename = f"{schema_name}.schema.json"
            document = json.loads((SCHEMA_DIRECTORY / filename).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(document)
            documents[schema_name] = document
            resources.append((document["$id"], Resource.from_contents(document)))
        self._documents = documents
        self._registry: Registry[Any] = Registry().with_resources(resources)

    def validate[Payload](self, schema_name: str, payload: Payload) -> Payload:
        validator = Draft202012Validator(
            self._documents[schema_name],
            registry=self._registry,
            format_checker=FormatChecker(),
        )
        try:
            validator.validate(payload)
        except JsonSchemaValidationError as error:
            raise SchemaValidationError(
                schema_name,
                "$",
                error.message,
                error.validator if isinstance(error.validator, str) else None,
            ) from error
        return payload


@pytest.fixture(scope="module")
def registry() -> _ArtifactSchemaRegistry:
    return _ArtifactSchemaRegistry()


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _digest(number: int) -> str:
    return f"{number:064x}"


def _schema_reference(number: int, schema_id: str, version: str) -> dict[str, Any]:
    return {
        "schema_id": schema_id,
        "version": version,
        "artifact_id": _uuid(number),
        "sha256": _digest(2000 + number),
    }


def _entry(
    number: int,
    artifact_type: str,
    *,
    uri: str,
    object_version: str = "1.0",
    parents: list[dict[str, Any]] | None = None,
    payload_schema_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "artifact_id": _uuid(number),
        "artifact_type": artifact_type,
        "semantic_sha256": _digest(1000 + number),
        "locator": {
            "uri": uri,
            "object_version": object_version,
        },
        "sha256": _digest(2000 + number),
        "bytes": 1000 + number,
        "media_type": MEDIA_TYPES[artifact_type],
        "producer": {
            "name": "robata-contract-fixture",
            "version": "1.0.0",
            "canonical_config_sha256": _digest(3000),
        },
        "lifecycle": {
            "state": "ACTIVE",
            "policy_version": "retention-v1",
        },
        "parents": parents or [],
        "payload_schema_ref": payload_schema_ref,
        "created_at": "2026-07-18T12:00:00Z",
    }


def _parent(number: int, relation: str) -> dict[str, Any]:
    return {
        "artifact_id": _uuid(number),
        "relation": relation,
    }


def _input_parents() -> list[dict[str, Any]]:
    return [
        _parent(12, "EXPORT_CONFIG"),
        _parent(11, "MAPPING_PROFILE"),
        _parent(10, "SOURCE_CONTENT"),
    ]


def _manifest_parents() -> list[dict[str, Any]]:
    return [
        *_input_parents(),
        *(_parent(number, "TIMESTAMP_OUTPUT") for number in range(30, 36)),
        *(_parent(number, "VIDEO_OUTPUT") for number in range(20, 26)),
    ]


def _snapshot() -> dict[str, Any]:
    schema_entries = [
        _entry(
            1,
            "JSON_SCHEMA",
            uri=MAPPING_SCHEMA_ID,
            object_version="2.0",
        ),
        _entry(
            2,
            "JSON_SCHEMA",
            uri=CONFIG_SCHEMA_ID,
            object_version="2.0",
        ),
        _entry(
            3,
            "JSON_SCHEMA",
            uri=TIMESTAMP_SCHEMA_ID,
            object_version="1.0",
        ),
        _entry(
            4,
            "JSON_SCHEMA",
            uri=MANIFEST_SCHEMA_ID,
            object_version="1.0",
        ),
    ]
    source = _entry(
        10,
        "RAW_MCAP",
        uri="object://recordings/session.mcap",
        object_version="source-v1",
    )
    mapping = _entry(
        11,
        "MAPPING_PROFILE",
        uri="object://profiles/six-camera-mapping.json",
        object_version="mapping-v1",
        payload_schema_ref=_schema_reference(1, MAPPING_SCHEMA_ID, "2.0"),
    )
    config = _entry(
        12,
        "EXPORT_CONFIG",
        uri="object://configs/video-export.json",
        object_version="config-v1",
        payload_schema_ref=_schema_reference(2, CONFIG_SCHEMA_ID, "2.0"),
    )
    videos = [
        _entry(
            number,
            "CAMERA_VIDEO_MP4",
            uri=f"object://exports/cam_{number - 19:02d}.mp4",
            object_version="export-v1",
            parents=_input_parents(),
        )
        for number in range(20, 26)
    ]
    timestamp_maps = [
        _entry(
            number,
            "CAMERA_VIDEO_TIMESTAMP_MAP",
            uri=f"object://exports/cam_{number - 29:02d}.timestamps.ndjson",
            object_version="export-v1",
            parents=_input_parents(),
            payload_schema_ref=_schema_reference(3, TIMESTAMP_SCHEMA_ID, "1.0"),
        )
        for number in range(30, 36)
    ]
    manifest = _entry(
        40,
        "CAMERA_VIDEO_EXPORT_MANIFEST",
        uri="object://exports/camera-video-export-manifest.json",
        object_version="export-v1",
        parents=_manifest_parents(),
        payload_schema_ref=_schema_reference(4, MANIFEST_SCHEMA_ID, "1.0"),
    )
    return {
        "schema_version": "2.0",
        "entries": [
            *schema_entries,
            source,
            mapping,
            config,
            *videos,
            *timestamp_maps,
            manifest,
        ],
    }


def _entry_model(payload: dict[str, Any]) -> ArtifactRegistryEntry:
    return ArtifactRegistryEntry.model_validate_json(json.dumps(payload))


def _snapshot_model(payload: dict[str, Any]) -> ArtifactRegistrySnapshot:
    return ArtifactRegistrySnapshot.model_validate_json(json.dumps(payload))


def _find_entry(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    return next(entry for entry in payload["entries"] if entry["artifact_type"] == artifact_type)


def _assert_entry_rejected_by_both(
    registry: _ArtifactSchemaRegistry,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(SchemaValidationError):
        registry.validate(ENTRY_SCHEMA, payload)
    with pytest.raises(ValidationError):
        _entry_model(payload)


def test_raw_entry_round_trips_through_schema_and_model(
    registry: _ArtifactSchemaRegistry,
) -> None:
    payload = _entry(
        10,
        "RAW_MCAP",
        uri="object://recordings/session.mcap",
        object_version="source-v1",
    )

    assert registry.validate(ENTRY_SCHEMA, payload) is payload
    entry = _entry_model(payload)

    assert entry.model_dump(mode="json") == payload
    assert isinstance(entry.parents, tuple)


def test_complete_registry_snapshot_round_trips(
    registry: _ArtifactSchemaRegistry,
) -> None:
    payload = _snapshot()

    assert registry.validate(SNAPSHOT_SCHEMA, payload) is payload
    snapshot = _snapshot_model(payload)
    serialized = snapshot.model_dump(mode="json")

    assert serialized == payload
    assert registry.validate(SNAPSHOT_SCHEMA, serialized) is serialized
    assert len(snapshot.entries) == 20


def test_entry_and_snapshot_are_deeply_immutable() -> None:
    snapshot = _snapshot_model(_snapshot())

    with pytest.raises(ValidationError):
        snapshot.schema_version = "2.0"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        snapshot.entries[0].locator.uri = "object://other/schema.json"  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.entries[0] = snapshot.entries[1]  # type: ignore[index]


def test_duplicate_parent_id_is_rejected_even_with_distinct_relations() -> None:
    payload = _find_entry(_snapshot(), "CAMERA_VIDEO_MP4")
    payload["parents"][1]["artifact_id"] = payload["parents"][0]["artifact_id"]

    with pytest.raises(ValidationError, match="repeat an artifact_id"):
        _entry_model(payload)


def test_parent_order_is_canonical() -> None:
    payload = _find_entry(_snapshot(), "CAMERA_VIDEO_MP4")
    payload["parents"][0], payload["parents"][1] = (
        payload["parents"][1],
        payload["parents"][0],
    )

    with pytest.raises(ValidationError, match="canonical"):
        _entry_model(payload)


def test_self_parent_is_rejected() -> None:
    payload = _find_entry(_snapshot(), "CAMERA_VIDEO_MP4")
    payload["parents"][2]["artifact_id"] = payload["artifact_id"]

    with pytest.raises(ValidationError, match="own parent"):
        _entry_model(payload)


@pytest.mark.parametrize(
    "artifact_type",
    [
        "JSON_SCHEMA",
        "RAW_MCAP",
    ],
)
def test_root_artifacts_reject_parents(
    registry: _ArtifactSchemaRegistry,
    artifact_type: str,
) -> None:
    payload = _entry(50, artifact_type, uri=f"object://root/{artifact_type.lower()}")
    payload["parents"] = [_parent(10, "SOURCE_CONTENT")]

    _assert_entry_rejected_by_both(registry, payload)


@pytest.mark.parametrize(
    "artifact_type",
    [
        "CAMERA_VIDEO_MP4",
        "CAMERA_VIDEO_TIMESTAMP_MAP",
    ],
)
def test_camera_outputs_require_all_three_inputs(
    registry: _ArtifactSchemaRegistry,
    artifact_type: str,
) -> None:
    payload = deepcopy(_find_entry(_snapshot(), artifact_type))
    payload["parents"].pop()

    _assert_entry_rejected_by_both(registry, payload)


def test_manifest_requires_six_video_and_six_timestamp_outputs(
    registry: _ArtifactSchemaRegistry,
) -> None:
    payload = _find_entry(_snapshot(), "CAMERA_VIDEO_EXPORT_MANIFEST")
    payload["parents"].pop()

    _assert_entry_rejected_by_both(registry, payload)


@pytest.mark.parametrize(
    "artifact_type",
    [
        "MAPPING_PROFILE",
        "EXPORT_CONFIG",
        "CAMERA_VIDEO_TIMESTAMP_MAP",
        "CAMERA_VIDEO_EXPORT_MANIFEST",
    ],
)
def test_structured_artifacts_require_payload_schema_reference(
    registry: _ArtifactSchemaRegistry,
    artifact_type: str,
) -> None:
    payload = deepcopy(_find_entry(_snapshot(), artifact_type))
    payload["payload_schema_ref"] = None

    _assert_entry_rejected_by_both(registry, payload)


@pytest.mark.parametrize(
    "artifact_type",
    [
        "JSON_SCHEMA",
        "RAW_MCAP",
        "CAMERA_VIDEO_MP4",
    ],
)
def test_binary_and_schema_artifacts_forbid_payload_schema_reference(
    registry: _ArtifactSchemaRegistry,
    artifact_type: str,
) -> None:
    payload = deepcopy(_find_entry(_snapshot(), artifact_type))
    payload["payload_schema_ref"] = _schema_reference(1, MAPPING_SCHEMA_ID, "2.0")

    _assert_entry_rejected_by_both(registry, payload)


def test_artifact_type_has_an_exact_media_type(
    registry: _ArtifactSchemaRegistry,
) -> None:
    payload = _find_entry(_snapshot(), "CAMERA_VIDEO_MP4")
    payload["media_type"] = "application/octet-stream"

    _assert_entry_rejected_by_both(registry, payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_id",
        "locator_version",
        "semantic_identity",
        "entry_order",
    ],
)
def test_snapshot_rejects_noncanonical_or_duplicate_identity(mutation: str) -> None:
    payload = _snapshot()
    if mutation == "artifact_id":
        payload["entries"][-1]["artifact_id"] = payload["entries"][-2]["artifact_id"]
    elif mutation == "locator_version":
        payload["entries"][-1]["locator"] = deepcopy(payload["entries"][0]["locator"])
    elif mutation == "semantic_identity":
        videos = [
            entry for entry in payload["entries"] if entry["artifact_type"] == "CAMERA_VIDEO_MP4"
        ]
        videos[1]["semantic_sha256"] = videos[0]["semantic_sha256"]
    else:
        payload["entries"][0], payload["entries"][1] = (
            payload["entries"][1],
            payload["entries"][0],
        )

    with pytest.raises(ValidationError):
        _snapshot_model(payload)


def test_snapshot_rejects_parent_absent_from_snapshot() -> None:
    payload = _snapshot()
    payload["entries"] = [
        entry for entry in payload["entries"] if entry["artifact_id"] != _uuid(10)
    ]

    with pytest.raises(ValidationError, match="absent from the snapshot"):
        _snapshot_model(payload)


def test_snapshot_rejects_parent_relation_to_wrong_artifact_type() -> None:
    payload = _snapshot()
    video = _find_entry(payload, "CAMERA_VIDEO_MP4")
    video["parents"][2]["artifact_id"] = _uuid(1)

    with pytest.raises(ValidationError, match="SOURCE_CONTENT must reference RAW_MCAP"):
        _snapshot_model(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_type",
        "schema_id",
        "version",
        "sha256",
    ],
)
def test_snapshot_resolves_exact_payload_schema_artifact(mutation: str) -> None:
    payload = _snapshot()
    mapping = _find_entry(payload, "MAPPING_PROFILE")
    reference = mapping["payload_schema_ref"]
    if mutation == "missing":
        reference["artifact_id"] = _uuid(99)
    elif mutation == "wrong_type":
        source = _find_entry(payload, "RAW_MCAP")
        reference.update(
            {
                "artifact_id": source["artifact_id"],
                "schema_id": source["locator"]["uri"],
                "version": source["locator"]["object_version"],
                "sha256": source["sha256"],
            }
        )
    elif mutation == "schema_id":
        reference["schema_id"] = "https://schemas.robata.dev/v2/other.schema.json"
    elif mutation == "version":
        reference["version"] = "2.1"
    else:
        reference["sha256"] = _digest(9999)

    with pytest.raises(ValidationError):
        _snapshot_model(payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("artifact_id", "00000000-0000-4000-8000-00000000000A"),
        ("semantic_sha256", "A" * 64),
        ("bytes", 0),
        ("created_at", "2026-07-18T12:00:00"),
    ],
)
def test_entry_rejects_invalid_wire_scalars(
    registry: _ArtifactSchemaRegistry,
    field: str,
    invalid_value: object,
) -> None:
    payload = _entry(
        10,
        "RAW_MCAP",
        uri="object://recordings/session.mcap",
        object_version="source-v1",
    )
    payload[field] = invalid_value

    _assert_entry_rejected_by_both(registry, payload)


def test_model_rejects_calendar_invalid_rfc3339_timestamp() -> None:
    payload = _entry(
        10,
        "RAW_MCAP",
        uri="object://recordings/session.mcap",
        object_version="source-v1",
    )
    payload["created_at"] = "2026-02-30T12:00:00Z"

    with pytest.raises(ValidationError, match="valid RFC3339"):
        _entry_model(payload)


def test_entry_rejects_bare_locator_and_unknown_field(
    registry: _ArtifactSchemaRegistry,
) -> None:
    payload = _entry(
        10,
        "RAW_MCAP",
        uri="recordings/session.mcap",
        object_version="source-v1",
    )
    _assert_entry_rejected_by_both(registry, payload)

    payload = _entry(
        10,
        "RAW_MCAP",
        uri="object://recordings/session.mcap",
        object_version="source-v1",
    )
    payload["mutable"] = True
    _assert_entry_rejected_by_both(registry, payload)


def test_snapshot_schema_is_closed_and_nonempty(
    registry: _ArtifactSchemaRegistry,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "entries": [],
    }
    with pytest.raises(SchemaValidationError):
        registry.validate(SNAPSHOT_SCHEMA, payload)
    with pytest.raises(ValidationError):
        _snapshot_model(payload)

    payload = _snapshot()
    payload["next_page"] = None
    with pytest.raises(SchemaValidationError):
        registry.validate(SNAPSHOT_SCHEMA, payload)
    with pytest.raises(ValidationError):
        _snapshot_model(payload)
