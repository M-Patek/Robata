"""Database storage models using SQLAlchemy 2.0 declarative syntax.

Implements the core tables from ARCHITECTURE_DESIGN_V1.md Section 16.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

try:
    from sqlalchemy import (
        BigInteger,
        Boolean,
        Column,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        String,
        Text,
        UniqueConstraint,
        create_engine,
    )
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
except ImportError:
    # SQLAlchemy is optional; provide stubs for type checking
    DeclarativeBase = object  # type: ignore[assignment, misc]
    Mapped = Any  # type: ignore[assignment, misc]
    mapped_column = lambda *a, **k: None  # type: ignore[assignment]
    UUID = lambda native=True: String  # type: ignore[assignment]
    BigInteger = Integer  # type: ignore[assignment]


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Artifact(Base):
    """Immutable artifact registry entry."""

    __tablename__ = "artifact"

    artifact_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    object_version: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(Text, nullable=False)
    producer: Mapped[str] = mapped_column(Text, nullable=False)
    producer_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class MCAPRecording(Base):
    """One MCAP recording after ingestion."""

    __tablename__ = "mcap_recording"

    mcap_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    recording_identity: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    source_artifact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("artifact.artifact_id", ondelete="RESTRICT"), nullable=False
    )
    start_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timebase: Mapped[str] = mapped_column(Text, nullable=False)
    camera_count: Mapped[int] = mapped_column(Integer, nullable=False, default=6)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class VideoStream(Base):
    """One raw video stream within an MCAP."""

    __tablename__ = "video_stream"

    stream_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    channel_id: Mapped[int] = mapped_column(Integer, nullable=False)
    codec: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    nominal_fps: Mapped[float] = mapped_column(Float, nullable=False)
    source_start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)


class CameraMappingRun(Base):
    """A versioned camera mapping run."""

    __tablename__ = "camera_mapping_run"

    mapping_run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    mapping_policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class CameraMapping(Base):
    """One camera-to-stream mapping."""

    __tablename__ = "camera_mapping"
    __table_args__ = (
        UniqueConstraint("mapping_run_id", "camera_id", name="uq_mapping_run_camera"),
    )

    mapping_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mapping_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("camera_mapping_run.mapping_run_id", ondelete="RESTRICT"),
        nullable=False,
    )
    camera_id: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    stream_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_stream.stream_id", ondelete="RESTRICT"), nullable=False
    )


class AlignmentRun(Base):
    """One alignment run for an MCAP."""

    __tablename__ = "alignment_run"

    alignment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    mapping_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("camera_mapping_run.mapping_run_id", ondelete="RESTRICT"),
        nullable=False,
    )
    method: Mapped[str] = mapped_column(Text, nullable=False)
    algorithm_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TemporalWindow(Base):
    """One temporal window for processing."""

    __tablename__ = "temporal_window"

    window_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    alignment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("alignment_run.alignment_id", ondelete="RESTRICT"), nullable=False
    )
    requested_start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    requested_end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    parent_window_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("temporal_window.window_id", ondelete="RESTRICT"), nullable=True
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TemporalPackage(Base):
    """One temporal visual package."""

    __tablename__ = "temporal_package"

    package_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    window_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("temporal_window.window_id", ondelete="RESTRICT"), nullable=False
    )
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    alignment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("alignment_run.alignment_id", ondelete="RESTRICT"), nullable=False
    )
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    semantic_content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_bytes_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TemporalPackageCamera(Base):
    """Per-camera entry within a temporal package."""

    __tablename__ = "temporal_package_camera"
    __table_args__ = (
        UniqueConstraint("package_id", "camera_id", name="uq_package_camera"),
    )

    package_camera_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    package_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("temporal_package.package_id", ondelete="RESTRICT"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    target_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    actual_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missed_targets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ModelInference(Base):
    """One model inference attempt."""

    __tablename__ = "model_inference"

    inference_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    logical_invocation_id: Mapped[str] = mapped_column(Text, nullable=False)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    package_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("temporal_package.package_id", ondelete="RESTRICT"), nullable=True
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class WorkItem(Base):
    """One work item in the processing pipeline."""

    __tablename__ = "work_item"

    work_item_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    work_logical_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    mcap_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mcap_recording.mcap_id", ondelete="RESTRICT"), nullable=False
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    subject_type: Mapped[str] = mapped_column(Text, nullable=False)
    subject_id: Mapped[str] = mapped_column(Text, nullable=False)
    input_digest: Mapped[str] = mapped_column(Text, nullable=False)
    config_digest: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False, default="PENDING")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sla_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_expiry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    lease_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fencing_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class WorkBarrier(Base):
    """A barrier for fan-out/reduction operations."""

    __tablename__ = "work_barrier"

    barrier_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    logical_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    subject_id: Mapped[str] = mapped_column(Text, nullable=False)
    expected_member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class WorkBarrierMember(Base):
    """One member of a work barrier."""

    __tablename__ = "work_barrier_member"
    __table_args__ = (
        UniqueConstraint("barrier_id", "work_item_id", name="uq_barrier_work_item"),
    )

    member_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    barrier_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("work_barrier.barrier_id", ondelete="RESTRICT"), nullable=False
    )
    work_item_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("work_item.work_item_id", ondelete="RESTRICT"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    criticality: Mapped[str] = mapped_column(Text, nullable=False)
    terminal_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)


class OutboxEvent(Base):
    """Transactional outbox event for successor publication."""

    __tablename__ = "outbox"

    outbox_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    payload_reference: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def get_engine(database_url: str | None = None) -> Any:
    """Create a SQLAlchemy engine.

    Args:
        database_url: Database connection URL. Defaults to SQLite in-memory.

    Returns:
        SQLAlchemy engine instance.
    """
    url = database_url or "sqlite:///:memory:"
    return create_engine(url, echo=False)


def create_tables(engine: Any) -> None:
    """Create all tables.

    Args:
        engine: SQLAlchemy engine.
    """
    Base.metadata.create_all(engine)


def drop_tables(engine: Any) -> None:
    """Drop all tables.

    Args:
        engine: SQLAlchemy engine.
    """
    Base.metadata.drop_all(engine)


__all__ = [
    "AlignmentRun",
    "Artifact",
    "Base",
    "CameraMapping",
    "CameraMappingRun",
    "MCAPRecording",
    "ModelInference",
    "OutboxEvent",
    "TemporalPackage",
    "TemporalPackageCamera",
    "TemporalWindow",
    "VideoStream",
    "WorkBarrier",
    "WorkBarrierMember",
    "WorkItem",
    "create_tables",
    "drop_tables",
    "get_engine",
]
