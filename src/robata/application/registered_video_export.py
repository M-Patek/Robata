"""Registry-authoritative V2 publication for the local six-camera video export."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from robata.alignment.rational_time import round_half_even
from robata.application.artifact_view import (
    ArtifactViewError,
    materialize_camera_video_view,
)
from robata.application.video_export import (
    MANIFEST_FILENAME,
    LocalVideoExportRequest,
    SixCameraVideoExportService,
    VideoExportRunError,
    VideoExportRunErrorCode,
    _drop_provenance,
    _sync_directory,
    _verify_sidecar_rows,
    _write_new_file,
)
from robata.contracts.artifacts import (
    ArtifactLifecycle,
    ArtifactLocator,
    ArtifactParent,
    ArtifactParentRelation,
    ArtifactProducer,
    ArtifactRegistryEntry,
    ArtifactRegistrySnapshot,
    ArtifactType,
    SchemaArtifactReference,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import Sha256Digest
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    recording_identity,
    semantic_sha256,
)
from robata.contracts.schema_registry import SchemaRegistry, SchemaRegistryError
from robata.contracts.video_export import (
    CameraVideoTimestampRow,
    DroppedMessageReasonCode,
    MappingProfileReference,
    MediaTimeMapping,
    SourceVideoStream,
    TailDurationPolicy,
    VideoExportAlignmentStatus,
    VideoExportExecutionMode,
)
from robata.contracts.video_export_v2 import (
    CameraVideoExportManifestV2,
    CameraVideoExportRecordV2,
    ExportConfigArtifactPayloadV2,
    MappingProfileArtifactPayloadV2,
    TimestampSidecarArtifactV2,
    camera_video_manifest_v2_semantic_projection,
)
from robata.ports.artifact_registry import (
    ArtifactBlobSource,
    ArtifactRegistry,
    ArtifactRegistryError,
)
from robata.ports.video_export import CameraVideoExporter, ExportedCameraVideoFacts
from robata.tempfiles import make_staging_directory

MAPPING_PROFILE_SCHEMA_ID = "https://schemas.robata.dev/mapping-profile"
EXPORT_CONFIG_SCHEMA_ID = "https://schemas.robata.dev/export-config"
TIMESTAMP_ROW_SCHEMA_ID = "https://schemas.robata.dev/camera-video-timestamp-row"
VIDEO_MANIFEST_SCHEMA_ID = "https://schemas.robata.dev/camera-video-export-manifest"
V1_SCHEMA_VERSION = "1.0.0"
V2_SCHEMA_VERSION = "2.0.0"
ARTIFACT_LIFECYCLE_POLICY_VERSION = "local-evidence-v1"


@dataclass(frozen=True, slots=True)
class PublishedRegisteredVideoExport:
    """One committed derivation and its verified local materialized view."""

    output_directory: Path
    manifest: CameraVideoExportManifestV2
    manifest_sha256: Sha256Digest
    manifest_artifact_id: str
    logical_key: str
    derivation_reused: bool
    materialized_view_reused: bool

    @property
    def reused(self) -> bool:
        """Compatibility summary; authoritative reuse dimensions stay separate."""

        return self.derivation_reused


@dataclass(frozen=True, slots=True)
class _ResolvedPayloadSchema:
    registered: Any
    reference: SchemaArtifactReference
    artifact: ArtifactRegistryEntry


@dataclass(frozen=True, slots=True)
class _InputArtifacts:
    schemas: tuple[_ResolvedPayloadSchema, ...]
    mapping_schema: _ResolvedPayloadSchema
    export_config_schema: _ResolvedPayloadSchema
    timestamp_schema: _ResolvedPayloadSchema
    manifest_schema: _ResolvedPayloadSchema
    source: ArtifactRegistryEntry
    mapping: ArtifactRegistryEntry
    export_config: ArtifactRegistryEntry
    blob_sources: Mapping[str, ArtifactBlobSource]


@dataclass(frozen=True, slots=True)
class _BuiltDerivation:
    manifest: CameraVideoExportManifestV2
    manifest_bytes: bytes
    manifest_entry: ArtifactRegistryEntry
    snapshot: ArtifactRegistrySnapshot
    blob_sources: Mapping[str, ArtifactBlobSource]


class RegisteredSixCameraVideoExportService:
    """Publish V2 artifacts to the registry before exposing an output view."""

    def __init__(
        self,
        exporter: CameraVideoExporter,
        artifact_registry: ArtifactRegistry,
        schema_registry: SchemaRegistry | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        max_parallel_exports: int = 1,
    ) -> None:
        self._verified_export = SixCameraVideoExportService(
            exporter,
            max_parallel_exports=max_parallel_exports,
        )
        self._artifact_registry = artifact_registry
        self._schema_registry = schema_registry or SchemaRegistry()
        self._clock = clock or (lambda: datetime.now(UTC))

    def export_local(
        self,
        request: LocalVideoExportRequest,
    ) -> PublishedRegisteredVideoExport:
        """Resolve or atomically register one local six-camera V2 derivation."""

        try:
            return self._export_local(request)
        except VideoExportRunError:
            raise
        except ArtifactViewError as error:
            raise VideoExportRunError(
                VideoExportRunErrorCode.MATERIALIZED_VIEW_FAILED,
                f"could not publish the registered export view: {error}",
            ) from error
        except ArtifactRegistryError as error:
            raise VideoExportRunError(
                VideoExportRunErrorCode.ARTIFACT_REGISTRY_FAILED,
                f"artifact registry rejected the export: {error}",
            ) from error
        except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
            raise VideoExportRunError(
                VideoExportRunErrorCode.MANIFEST_INVALID,
                f"registered camera-video export is invalid: {error}",
            ) from error
        except OSError as error:
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_IO_ERROR,
                f"registered camera-video export I/O failed: {error}",
            ) from error

    def _export_local(
        self,
        request: LocalVideoExportRequest,
    ) -> PublishedRegisteredVideoExport:
        SixCameraVideoExportService._validate_request(request)
        output_directory = self._resolve_output_directory(request.output_directory)
        created_at = self._created_at()
        inputs = self._build_inputs(request, created_at)
        logical_key = _logical_derivation_key(request, inputs)

        committed = self._artifact_registry.lookup_derivation(logical_key)
        if committed is not None:
            SixCameraVideoExportService._verify_source_unchanged(request)
            manifest, manifest_bytes = self._load_committed_manifest(
                committed.snapshot,
                committed.manifest_artifact_id,
                inputs,
                request,
            )
            view = materialize_camera_video_view(
                registry=self._artifact_registry,
                snapshot=committed.snapshot,
                manifest_artifact_id=committed.manifest_artifact_id,
                manifest=manifest,
                output_directory=output_directory,
            )
            return PublishedRegisteredVideoExport(
                output_directory=view.output_directory,
                manifest=manifest,
                manifest_sha256=exact_bytes_sha256(manifest_bytes),
                manifest_artifact_id=committed.manifest_artifact_id,
                logical_key=logical_key,
                derivation_reused=True,
                materialized_view_reused=view.reused,
            )

        if output_directory.exists():
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_EXISTS,
                "an unregistered output directory cannot be reused or promoted",
            )

        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging_directory = make_staging_directory(
            output_directory.parent,
            prefix=f".{output_directory.name}.artifacts-",
        )
        try:
            facts = self._verified_export._export_all_cameras(
                request,
                staging_directory,
            )
            SixCameraVideoExportService._verify_source_unchanged(request)
            built = self._build_derivation(
                request,
                inputs,
                facts,
                staging_directory,
                created_at,
            )
            _write_new_file(
                staging_directory / MANIFEST_FILENAME,
                built.manifest_bytes,
            )
            _sync_directory(staging_directory)
            publication_manifest = built.manifest
            publication_manifest_bytes = built.manifest_bytes
            try:
                published = self._artifact_registry.publish_derivation(
                    snapshot=built.snapshot,
                    logical_key=logical_key,
                    manifest_artifact_id=built.manifest_entry.artifact_id,
                    blob_sources=built.blob_sources,
                )
            except ArtifactRegistryError:
                recovered = self._artifact_registry.lookup_derivation(logical_key)
                if recovered is None:
                    raise
                publication_manifest, publication_manifest_bytes = self._load_committed_manifest(
                    recovered.snapshot,
                    recovered.manifest_artifact_id,
                    inputs,
                    request,
                )
                published = recovered
        finally:
            if staging_directory.exists():
                with suppress(OSError):
                    shutil.rmtree(staging_directory)

        view = materialize_camera_video_view(
            registry=self._artifact_registry,
            snapshot=published.snapshot,
            manifest_artifact_id=published.manifest_artifact_id,
            manifest=publication_manifest,
            output_directory=output_directory,
        )
        return PublishedRegisteredVideoExport(
            output_directory=view.output_directory,
            manifest=publication_manifest,
            manifest_sha256=exact_bytes_sha256(publication_manifest_bytes),
            manifest_artifact_id=published.manifest_artifact_id,
            logical_key=logical_key,
            derivation_reused=published.reused,
            materialized_view_reused=view.reused,
        )

    @staticmethod
    def _resolve_output_directory(output_directory: Path) -> Path:
        try:
            resolved = output_directory.resolve()
        except OSError as error:
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_IO_ERROR,
                f"could not resolve export destination: {error}",
            ) from error
        if not resolved.name:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "export destination must name a child directory",
            )
        return resolved

    def _created_at(self) -> str:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise VideoExportRunError(
                VideoExportRunErrorCode.INVALID_REQUEST,
                "artifact publication clock must return a timezone-aware datetime",
            )
        return (
            value.astimezone(UTC)
            .isoformat(timespec="microseconds")
            .replace(
                "+00:00",
                "Z",
            )
        )

    def _build_inputs(
        self,
        request: LocalVideoExportRequest,
        created_at: str,
    ) -> _InputArtifacts:
        mapping_schema = self._resolve_payload_schema(
            MAPPING_PROFILE_SCHEMA_ID,
            V2_SCHEMA_VERSION,
            created_at,
        )
        export_config_schema = self._resolve_payload_schema(
            EXPORT_CONFIG_SCHEMA_ID,
            V2_SCHEMA_VERSION,
            created_at,
        )
        timestamp_schema = self._resolve_payload_schema(
            TIMESTAMP_ROW_SCHEMA_ID,
            V1_SCHEMA_VERSION,
            created_at,
        )
        manifest_schema = self._resolve_payload_schema(
            VIDEO_MANIFEST_SCHEMA_ID,
            V2_SCHEMA_VERSION,
            created_at,
        )
        schemas = (
            mapping_schema,
            export_config_schema,
            timestamp_schema,
            manifest_schema,
        )

        mapping_payload = MappingProfileArtifactPayloadV2(
            schema_version="2.0",
            profile_id=request.mapping_profile.profile_id,
            version=request.mapping_profile.version,
            profile_kind=request.mapping_profile.profile_kind,
            approval_status=cast(
                Literal["APPROVED", "UNAPPROVED"],
                request.mapping_profile.approval_status,
            ),
            approved=request.mapping_profile.approved,
            mapping_policy="EXACT_TOPIC",
            required_schema="foxglove.CompressedImage",
            topics=request.mapping_profile.topics,
        )
        mapping_document = mapping_payload.model_dump(mode="json")
        self._schema_registry.validate_pinned(
            mapping_schema.registered.ref,
            mapping_document,
        )
        mapping_bytes = canonical_json_bytes(mapping_payload)

        export_config_payload = ExportConfigArtifactPayloadV2(
            schema_version="2.0",
            exporter=request.exporter.as_contract(),
        )
        export_config_document = export_config_payload.model_dump(mode="json")
        self._schema_registry.validate_pinned(
            export_config_schema.registered.ref,
            export_config_document,
        )
        export_config_bytes = canonical_json_bytes(export_config_payload)
        export_config_semantic_sha256 = semantic_sha256(export_config_document)

        source = self._entry_or_existing(
            artifact_type=ArtifactType.RAW_MCAP,
            semantic_sha256=request.inspection.source_sha256,
            exact_sha256=request.inspection.source_sha256,
            bytes_count=request.inspection.source_size_bytes,
            media_type="application/x-mcap",
            producer=ArtifactProducer(
                name="robata.raw-source-import",
                version="1.0.0",
                canonical_config_sha256=semantic_sha256({"policy": "exact-source-content-v1"}),
            ),
            parents=(),
            payload_schema_ref=None,
            created_at=created_at,
        )
        mapping = self._entry_or_existing(
            artifact_type=ArtifactType.MAPPING_PROFILE,
            semantic_sha256=request.mapping_profile_digest,
            exact_sha256=exact_bytes_sha256(mapping_bytes),
            bytes_count=len(mapping_bytes),
            media_type="application/json",
            producer=ArtifactProducer(
                name="robata.mapping-profile",
                version=request.mapping_profile.version,
                canonical_config_sha256=request.mapping_profile_digest,
            ),
            parents=(),
            payload_schema_ref=mapping_schema.reference,
            created_at=created_at,
        )
        export_config = self._entry_or_existing(
            artifact_type=ArtifactType.EXPORT_CONFIG,
            semantic_sha256=export_config_semantic_sha256,
            exact_sha256=exact_bytes_sha256(export_config_bytes),
            bytes_count=len(export_config_bytes),
            media_type="application/json",
            producer=_export_producer(request),
            parents=(),
            payload_schema_ref=export_config_schema.reference,
            created_at=created_at,
        )

        blob_sources: dict[str, ArtifactBlobSource] = {
            schema.artifact.artifact_id: schema.registered.path for schema in schemas
        }
        blob_sources.update(
            {
                source.artifact_id: request.source,
                mapping.artifact_id: mapping_bytes,
                export_config.artifact_id: export_config_bytes,
            }
        )
        return _InputArtifacts(
            schemas=schemas,
            mapping_schema=mapping_schema,
            export_config_schema=export_config_schema,
            timestamp_schema=timestamp_schema,
            manifest_schema=manifest_schema,
            source=source,
            mapping=mapping,
            export_config=export_config,
            blob_sources=blob_sources,
        )

    def _resolve_payload_schema(
        self,
        schema_id: str,
        version: str,
        created_at: str,
    ) -> _ResolvedPayloadSchema:
        registered = self._schema_registry.resolve_version(schema_id, version)
        catalog_ref = registered.ref
        reference = SchemaArtifactReference(
            schema_id=catalog_ref.schema_id,
            version=catalog_ref.version,
            artifact_id=catalog_ref.artifact_id,
            sha256=catalog_ref.sha256,
        )
        schema_bytes = registered.document_bytes
        if exact_bytes_sha256(schema_bytes) != reference.sha256:
            raise ValueError(f"catalog returned changed schema bytes for {schema_id} {version}")
        artifact = self._entry_or_existing(
            artifact_type=ArtifactType.JSON_SCHEMA,
            semantic_sha256=reference.sha256,
            exact_sha256=reference.sha256,
            bytes_count=len(schema_bytes),
            media_type="application/schema+json",
            producer=ArtifactProducer(
                name="robata.schema-registry",
                version="1.0.0",
                canonical_config_sha256=reference.sha256,
            ),
            parents=(),
            payload_schema_ref=None,
            created_at=created_at,
            expected_artifact_id=reference.artifact_id,
            locator=ArtifactLocator(
                uri=reference.schema_id,
                object_version=reference.version,
            ),
        )
        return _ResolvedPayloadSchema(
            registered=registered,
            reference=reference,
            artifact=artifact,
        )

    def _entry_or_existing(
        self,
        *,
        artifact_type: ArtifactType,
        semantic_sha256: Sha256Digest,
        exact_sha256: Sha256Digest,
        bytes_count: int,
        media_type: str,
        producer: ArtifactProducer,
        parents: tuple[ArtifactParent, ...],
        payload_schema_ref: SchemaArtifactReference | None,
        created_at: str,
        expected_artifact_id: str | None = None,
        locator: ArtifactLocator | None = None,
    ) -> ArtifactRegistryEntry:
        artifact_id = self._artifact_registry.allocate_artifact_id(
            artifact_type,
            semantic_sha256,
        )
        if expected_artifact_id is not None and expected_artifact_id != artifact_id:
            raise ValueError(
                "catalog artifact ID is not allocated from the schema digest: "
                f"{expected_artifact_id}"
            )
        candidate = ArtifactRegistryEntry(
            schema_version="2.0",
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            semantic_sha256=semantic_sha256,
            locator=locator
            or ArtifactLocator(
                uri=f"robata-artifact://local/{artifact_id}",
                object_version=exact_sha256,
            ),
            sha256=exact_sha256,
            bytes=bytes_count,
            media_type=media_type,
            producer=producer,
            lifecycle=ArtifactLifecycle(
                state="ACTIVE",
                policy_version=ARTIFACT_LIFECYCLE_POLICY_VERSION,
            ),
            parents=_canonical_parents(parents),
            payload_schema_ref=payload_schema_ref,
            created_at=created_at,
        )
        existing = self._artifact_registry.lookup_artifact(
            artifact_type,
            semantic_sha256,
        )
        if existing is None:
            return candidate
        if existing.model_dump(mode="json", exclude={"created_at"}) != candidate.model_dump(
            mode="json",
            exclude={"created_at"},
        ):
            raise ValueError(
                f"existing artifact metadata conflicts with semantic identity {artifact_id}"
            )
        return existing

    def _build_derivation(
        self,
        request: LocalVideoExportRequest,
        inputs: _InputArtifacts,
        facts: tuple[ExportedCameraVideoFacts, ...],
        staging_directory: Path,
        created_at: str,
    ) -> _BuiltDerivation:
        parent_entries = (inputs.source, inputs.mapping, inputs.export_config)
        records: list[CameraVideoExportRecordV2] = []
        output_entries: list[ArtifactRegistryEntry] = []
        blob_sources = dict(inputs.blob_sources)
        for camera_id, camera_facts in zip(CAMERA_IDS, facts, strict=True):
            if camera_facts.camera_id is not camera_id:
                raise ValueError("camera exporter facts are not in canonical six-camera order")
            video_path = staging_directory / f"{camera_id.value}.mp4"
            sidecar_path = staging_directory / f"{camera_id.value}.timestamps.jsonl"
            self._validate_pinned_sidecar(sidecar_path, inputs.timestamp_schema)

            parents = (
                ArtifactParent(
                    artifact_id=inputs.export_config.artifact_id,
                    relation=ArtifactParentRelation.EXPORT_CONFIG,
                ),
                ArtifactParent(
                    artifact_id=inputs.mapping.artifact_id,
                    relation=ArtifactParentRelation.MAPPING_PROFILE,
                ),
                ArtifactParent(
                    artifact_id=inputs.source.artifact_id,
                    relation=ArtifactParentRelation.SOURCE_CONTENT,
                ),
            )
            video_entry = self._entry_or_existing(
                artifact_type=ArtifactType.CAMERA_VIDEO_MP4,
                semantic_sha256=_camera_output_semantic_sha256(
                    camera_id=camera_id,
                    artifact_type=ArtifactType.CAMERA_VIDEO_MP4,
                    exact_sha256=camera_facts.video_sha256,
                    bytes_count=camera_facts.video_size_bytes,
                    parent_entries=parent_entries,
                    payload_schema_sha256=None,
                ),
                exact_sha256=camera_facts.video_sha256,
                bytes_count=camera_facts.video_size_bytes,
                media_type="video/mp4",
                producer=_export_producer(request),
                parents=parents,
                payload_schema_ref=None,
                created_at=created_at,
            )
            sidecar_entry = self._entry_or_existing(
                artifact_type=ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
                semantic_sha256=_camera_output_semantic_sha256(
                    camera_id=camera_id,
                    artifact_type=ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
                    exact_sha256=camera_facts.sidecar_sha256,
                    bytes_count=camera_facts.sidecar_size_bytes,
                    parent_entries=parent_entries,
                    payload_schema_sha256=inputs.timestamp_schema.reference.sha256,
                ),
                exact_sha256=camera_facts.sidecar_sha256,
                bytes_count=camera_facts.sidecar_size_bytes,
                media_type="application/x-ndjson",
                producer=_export_producer(request),
                parents=parents,
                payload_schema_ref=inputs.timestamp_schema.reference,
                created_at=created_at,
            )
            records.append(_camera_record_v2(camera_facts, video_entry, sidecar_entry))
            output_entries.extend((video_entry, sidecar_entry))
            blob_sources[video_entry.artifact_id] = video_path
            blob_sources[sidecar_entry.artifact_id] = sidecar_path

        manifest = _build_manifest_v2(request, inputs, tuple(records))
        self._schema_registry.validate_pinned(
            inputs.manifest_schema.registered.ref,
            manifest.model_dump(mode="json"),
        )
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_entry = self._entry_or_existing(
            artifact_type=ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST,
            semantic_sha256=manifest.semantic_content_sha256,
            exact_sha256=exact_bytes_sha256(manifest_bytes),
            bytes_count=len(manifest_bytes),
            media_type="application/json",
            producer=_export_producer(request),
            parents=_manifest_parents(manifest),
            payload_schema_ref=inputs.manifest_schema.reference,
            created_at=created_at,
        )
        entries = (
            *(schema.artifact for schema in inputs.schemas),
            inputs.source,
            inputs.mapping,
            inputs.export_config,
            *output_entries,
            manifest_entry,
        )
        snapshot = ArtifactRegistrySnapshot(
            schema_version="2.0",
            entries=tuple(sorted(entries, key=lambda entry: entry.artifact_id)),
        )
        blob_sources[manifest_entry.artifact_id] = manifest_bytes
        return _BuiltDerivation(
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            manifest_entry=manifest_entry,
            snapshot=snapshot,
            blob_sources=blob_sources,
        )

    def _load_committed_manifest(
        self,
        snapshot: ArtifactRegistrySnapshot,
        manifest_artifact_id: str,
        inputs: _InputArtifacts,
        request: LocalVideoExportRequest,
    ) -> tuple[CameraVideoExportManifestV2, bytes]:
        entries_by_id = {entry.artifact_id: entry for entry in snapshot.entries}
        manifest_entry = entries_by_id.get(manifest_artifact_id)
        if (
            manifest_entry is None
            or manifest_entry.artifact_type is not ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST
        ):
            raise ValueError("registered derivation has no typed camera-video manifest")
        manifest_bytes = self._artifact_registry.resolve_blob(manifest_artifact_id).read_bytes()
        if (
            exact_bytes_sha256(manifest_bytes) != manifest_entry.sha256
            or len(manifest_bytes) != manifest_entry.bytes
        ):
            raise ValueError("registered manifest blob disagrees with its artifact entry")
        manifest = CameraVideoExportManifestV2.model_validate_json(
            manifest_bytes,
            strict=True,
        )
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise ValueError("registered V2 manifest is not canonical JSON")
        self._schema_registry.validate_pinned(
            inputs.manifest_schema.registered.ref,
            manifest.model_dump(mode="json"),
        )
        self._verify_committed_derivation(
            snapshot,
            manifest_entry,
            manifest,
            manifest_bytes,
            inputs,
            request,
        )
        for record in manifest.cameras:
            sidecar_path = self._artifact_registry.resolve_blob(
                record.timestamp_sidecar_artifact.artifact.artifact_id
            )
            self._validate_pinned_sidecar(sidecar_path, inputs.timestamp_schema)
            _verify_sidecar_rows(
                sidecar_path,
                camera_id=record.camera_id,
                export_profile_id=manifest.exporter.export_profile_id,
                export_profile_version=manifest.exporter.profile_version,
                expected_count=record.timestamp_sidecar_artifact.row_count,
                expected_first_source_ns=record.export_first_observed_source_message_ns,
                expected_last_source_ns=record.export_last_observed_source_message_ns,
                expected_first_pts=record.media_time_mapping.first_pts,
                expected_last_pts=record.media_time_mapping.last_pts,
                expected_last_duration_ns=record.media_time_mapping.last_duration,
                expected_keyframe_count=record.keyframe_count,
                schema_ref=inputs.timestamp_schema.registered.ref,
            )
        return manifest, manifest_bytes

    def _verify_committed_derivation(
        self,
        snapshot: ArtifactRegistrySnapshot,
        manifest_entry: ArtifactRegistryEntry,
        manifest: CameraVideoExportManifestV2,
        manifest_bytes: bytes,
        inputs: _InputArtifacts,
        request: LocalVideoExportRequest,
    ) -> None:
        expected_recording_identity = recording_identity(
            request.namespace,
            request.inspection.source_sha256,
        )
        if (
            manifest.schema_ref != inputs.manifest_schema.reference
            or manifest.recording_identity != expected_recording_identity
            or manifest.source_content_sha256 != request.inspection.source_sha256
            or manifest.source_size_bytes != request.inspection.source_size_bytes
            or manifest.source_artifact_id != inputs.source.artifact_id
            or manifest.mapping_profile_artifact_id != inputs.mapping.artifact_id
            or manifest.export_config_artifact_id != inputs.export_config.artifact_id
            or manifest.mapping_profile.version != request.mapping_profile.version
            or manifest.mapping_profile.digest != request.mapping_profile_digest
            or manifest.mapping_profile.approved
            or manifest.exporter != request.exporter.as_contract()
        ):
            raise VideoExportRunError(
                VideoExportRunErrorCode.OUTPUT_EXISTS,
                "registered derivation belongs to different input semantics",
            )

        entries_by_id = {entry.artifact_id: entry for entry in snapshot.entries}
        expected_output_producer = _export_producer(request)
        expected_ids = {
            *(schema.artifact.artifact_id for schema in inputs.schemas),
            inputs.source.artifact_id,
            inputs.mapping.artifact_id,
            inputs.export_config.artifact_id,
            manifest_entry.artifact_id,
        }
        expected_ids.update(
            artifact_id
            for record in manifest.cameras
            for artifact_id in (
                record.video_artifact.artifact_id,
                record.timestamp_sidecar_artifact.artifact.artifact_id,
            )
        )
        if set(entries_by_id) != expected_ids or len(expected_ids) != 20:
            raise ValueError("registered V2 derivation must contain exactly 20 artifacts")

        for schema in inputs.schemas:
            if not _entries_match_ignoring_created_at(
                entries_by_id[schema.artifact.artifact_id],
                schema.artifact,
            ):
                raise ValueError("registered derivation substituted a payload schema artifact")
        for expected in (inputs.source, inputs.mapping, inputs.export_config):
            if not _entries_match_ignoring_created_at(
                entries_by_id[expected.artifact_id],
                expected,
            ):
                raise ValueError("registered derivation substituted an input artifact")
        for record in manifest.cameras:
            video = record.video_artifact
            sidecar = record.timestamp_sidecar_artifact.artifact
            if entries_by_id[video.artifact_id] != video:
                raise ValueError(f"{record.camera_id.value} video entry differs from snapshot")
            if entries_by_id[sidecar.artifact_id] != sidecar:
                raise ValueError(f"{record.camera_id.value} timestamp entry differs from snapshot")
            if (
                video.producer != expected_output_producer
                or sidecar.producer != expected_output_producer
                or video.payload_schema_ref is not None
                or sidecar.payload_schema_ref != inputs.timestamp_schema.reference
            ):
                raise ValueError(
                    f"{record.camera_id.value} output producer or payload schema is invalid"
                )
            channel = request.channels[record.camera_id]
            if (
                record.source.topic != channel.topic
                or record.source.channel_id != channel.channel_id
                or record.source.schema_name != channel.schema_name
                or record.source.codec != (channel.codec or "").strip().lower()
                or record.input_message_count != channel.message_count
                or record.source_first_observed_message_ns != channel.first_message_time_ns
                or record.source_last_observed_message_ns != channel.last_message_time_ns
            ):
                raise VideoExportRunError(
                    VideoExportRunErrorCode.OUTPUT_EXISTS,
                    f"registered source facts differ for {record.camera_id.value}",
                )
            for artifact in (video, sidecar):
                expected_semantic = _camera_output_semantic_sha256(
                    camera_id=record.camera_id,
                    artifact_type=artifact.artifact_type,
                    exact_sha256=artifact.sha256,
                    bytes_count=artifact.bytes,
                    parent_entries=(inputs.source, inputs.mapping, inputs.export_config),
                    payload_schema_sha256=(
                        None
                        if artifact.payload_schema_ref is None
                        else artifact.payload_schema_ref.sha256
                    ),
                )
                if artifact.semantic_sha256 != expected_semantic:
                    raise ValueError(f"{record.camera_id.value} output semantic digest is invalid")

        if (
            manifest_entry.semantic_sha256 != manifest.semantic_content_sha256
            or manifest_entry.sha256 != exact_bytes_sha256(manifest_bytes)
            or manifest_entry.bytes != len(manifest_bytes)
            or manifest_entry.payload_schema_ref != inputs.manifest_schema.reference
            or manifest_entry.parents != _manifest_parents(manifest)
        ):
            raise ValueError("external manifest entry disagrees with canonical manifest bytes")

    def _validate_pinned_sidecar(
        self,
        path: Path,
        timestamp_schema: _ResolvedPayloadSchema,
    ) -> None:
        contents = path.read_bytes()
        if not contents.endswith(b"\n"):
            raise ValueError("timestamp sidecar must end with a newline")
        for index, raw_line in enumerate(contents.splitlines()):
            try:
                payload = json.loads(raw_line)
                row = CameraVideoTimestampRow.model_validate_json(raw_line, strict=True)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(f"timestamp sidecar row {index} is invalid") from error
            if canonical_json_bytes(row) != raw_line:
                raise ValueError(f"timestamp sidecar row {index} is not canonical JSON")
            self._schema_registry.validate_pinned(
                timestamp_schema.registered.ref,
                payload,
            )


def _build_manifest_v2(
    request: LocalVideoExportRequest,
    inputs: _InputArtifacts,
    records: tuple[CameraVideoExportRecordV2, ...],
) -> CameraVideoExportManifestV2:
    fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": inputs.manifest_schema.reference,
        "semantic_content_sha256": "0" * 64,
        "execution_mode": VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE,
        "recording_identity": recording_identity(
            request.namespace,
            request.inspection.source_sha256,
        ),
        "source_content_sha256": request.inspection.source_sha256,
        "source_size_bytes": request.inspection.source_size_bytes,
        "source_artifact_id": inputs.source.artifact_id,
        "mapping_profile_artifact_id": inputs.mapping.artifact_id,
        "export_config_artifact_id": inputs.export_config.artifact_id,
        "mapping_profile": MappingProfileReference(
            version=request.mapping_profile.version,
            digest=request.mapping_profile_digest,
            approved=False,
        ),
        "ready_manifest_id": None,
        "ready_manifest_semantic_sha256": None,
        "alignment_id": None,
        "alignment_semantic_sha256": None,
        "alignment_status": VideoExportAlignmentStatus.UNVERIFIED,
        "exporter": request.exporter.as_contract(),
        "cameras": records,
    }
    draft = CameraVideoExportManifestV2.model_construct(**fields)
    fields["semantic_content_sha256"] = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft)
    )
    return CameraVideoExportManifestV2.model_validate(fields, strict=True)


def _camera_record_v2(
    facts: ExportedCameraVideoFacts,
    video_entry: ArtifactRegistryEntry,
    sidecar_entry: ArtifactRegistryEntry,
) -> CameraVideoExportRecordV2:
    last_duration = round_half_even(
        facts.tail_duration_ns * facts.time_base_denominator,
        1_000_000_000 * facts.time_base_numerator,
    )
    return CameraVideoExportRecordV2(
        camera_id=facts.camera_id,
        source=SourceVideoStream(
            topic=facts.topic,
            channel_id=facts.channel_id,
            schema_name="foxglove.CompressedImage",
            codec="h264",
        ),
        input_message_count=facts.source_message_count,
        source_first_observed_message_ns=facts.source_first_log_time_ns,
        source_last_observed_message_ns=facts.source_last_log_time_ns,
        export_first_observed_source_message_ns=facts.export_first_source_log_time_ns,
        export_last_observed_source_message_ns=facts.export_last_source_log_time_ns,
        leading_drops=_drop_provenance(
            facts.leading_access_unit_count,
            DroppedMessageReasonCode.BEFORE_FIRST_DECODABLE_KEYFRAME,
            facts.leading_first_log_time_ns,
            facts.leading_last_log_time_ns,
        ),
        trailing_drops=_drop_provenance(
            facts.trailing_access_unit_count,
            DroppedMessageReasonCode.AFTER_LAST_COMPLETE_SAMPLE,
            facts.trailing_first_log_time_ns,
            facts.trailing_last_log_time_ns,
        ),
        exported_packet_count=facts.exported_packet_count,
        exported_frame_count=facts.decoded_frame_count,
        keyframe_count=facts.keyframe_count,
        width=facts.width,
        height=facts.height,
        video_artifact=video_entry,
        timestamp_sidecar_artifact=TimestampSidecarArtifactV2(
            artifact=sidecar_entry,
            row_count=facts.sidecar_row_count,
        ),
        media_time_mapping=MediaTimeMapping(
            zero_source_ns=facts.export_first_source_log_time_ns,
            time_base_numerator=facts.time_base_numerator,
            time_base_denominator=facts.time_base_denominator,
            first_pts=facts.first_pts_ns,
            last_pts=facts.last_pts_ns,
            last_duration=last_duration,
            tail_duration_policy=TailDurationPolicy(facts.tail_duration_policy),
            rounding="HALF_EVEN",
            max_rounding_error_ns=facts.max_timestamp_mapping_error_ns,
        ),
    )


def _logical_derivation_key(
    request: LocalVideoExportRequest,
    inputs: _InputArtifacts,
) -> str:
    digest = semantic_sha256(
        {
            "schema_version": "2.0",
            "manifest_schema_sha256": inputs.manifest_schema.reference.sha256,
            "timestamp_schema_sha256": inputs.timestamp_schema.reference.sha256,
            "recording_identity": recording_identity(
                request.namespace,
                request.inspection.source_sha256,
            ),
            "source_semantic_sha256": inputs.source.semantic_sha256,
            "mapping_semantic_sha256": inputs.mapping.semantic_sha256,
            "export_config_semantic_sha256": inputs.export_config.semantic_sha256,
            "execution_mode": VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE.value,
        }
    )
    return f"camera-video-export:v2:{digest}"


def _camera_output_semantic_sha256(
    *,
    camera_id: CameraId,
    artifact_type: ArtifactType,
    exact_sha256: Sha256Digest,
    bytes_count: int,
    parent_entries: tuple[ArtifactRegistryEntry, ...],
    payload_schema_sha256: Sha256Digest | None,
) -> Sha256Digest:
    parent_semantics = {
        entry.artifact_type.value: entry.semantic_sha256 for entry in parent_entries
    }
    return semantic_sha256(
        {
            "schema_version": "2.0",
            "camera_id": camera_id.value,
            "artifact_type": artifact_type.value,
            "sha256": exact_sha256,
            "bytes": bytes_count,
            "parent_semantic_sha256": parent_semantics,
            "payload_schema_sha256": payload_schema_sha256,
        }
    )


def _entries_match_ignoring_created_at(
    registered: ArtifactRegistryEntry,
    expected: ArtifactRegistryEntry,
) -> bool:
    return registered.model_dump(mode="json", exclude={"created_at"}) == expected.model_dump(
        mode="json",
        exclude={"created_at"},
    )


def _manifest_parents(
    manifest: CameraVideoExportManifestV2,
) -> tuple[ArtifactParent, ...]:
    return _canonical_parents(
        (
            ArtifactParent(
                artifact_id=manifest.export_config_artifact_id,
                relation=ArtifactParentRelation.EXPORT_CONFIG,
            ),
            ArtifactParent(
                artifact_id=manifest.mapping_profile_artifact_id,
                relation=ArtifactParentRelation.MAPPING_PROFILE,
            ),
            ArtifactParent(
                artifact_id=manifest.source_artifact_id,
                relation=ArtifactParentRelation.SOURCE_CONTENT,
            ),
            *(
                ArtifactParent(
                    artifact_id=record.timestamp_sidecar_artifact.artifact.artifact_id,
                    relation=ArtifactParentRelation.TIMESTAMP_OUTPUT,
                )
                for record in manifest.cameras
            ),
            *(
                ArtifactParent(
                    artifact_id=record.video_artifact.artifact_id,
                    relation=ArtifactParentRelation.VIDEO_OUTPUT,
                )
                for record in manifest.cameras
            ),
        )
    )


def _canonical_parents(
    parents: tuple[ArtifactParent, ...],
) -> tuple[ArtifactParent, ...]:
    return tuple(
        sorted(
            parents,
            key=lambda parent: (parent.relation.value, parent.artifact_id),
        )
    )


def _export_producer(request: LocalVideoExportRequest) -> ArtifactProducer:
    descriptor = request.exporter
    return ArtifactProducer(
        name=descriptor.name,
        version=descriptor.version,
        canonical_config_sha256=descriptor.canonical_config_sha256,
    )


__all__ = [
    "PublishedRegisteredVideoExport",
    "RegisteredSixCameraVideoExportService",
]
