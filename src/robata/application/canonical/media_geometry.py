"""Bounded, deterministic geometry views for selected media evidence.

Geometry correction is deliberately a derived-view concern.  The source packet,
timestamp and frame identity remain authoritative; this module only produces a
small in-memory payload for a selected window/model input and a fact that binds
that payload to its exact source lineage.  The default implementation is a
deterministic nearest-neighbour pass-through/resample so the local CPU path does
not require OpenCV or a camera-calibration SDK.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Annotated, Final, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256, semantic_sha256
from robata.ports.decoded_frame import DecodedFrameView

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

GEOMETRY_PROJECTION_VERSION: Final = "canonical-media-geometry-v1"
GEOMETRY_LINEAGE_PROJECTION_VERSION: Final = "canonical-derived-geometry-lineage-v1"
GEOMETRY_MAP_POLICY_VERSION: Final = "canonical-geometry-map-nearest-v1"
DEFAULT_PREPROCESS_POLICY_VERSION: Final = "canonical-frame-preprocess-v1"


class GeometryView(StrEnum):
    """Supported derived camera views.

    ``RAW`` is the source-coordinate view.  The other values are supplemental
    views and must carry a camera calibration profile.
    """

    RAW = "RAW"
    FISHEYE_UNDISTORTED = "FISHEYE_UNDISTORTED"
    PERSPECTIVE = "PERSPECTIVE"
    EGOCENTRIC = "EGOCENTRIC"


class GeometryInterpolation(StrEnum):
    """Interpolation modes understood by the local bounded implementation."""

    NEAREST = "NEAREST"


class GeometryProcessingError(ValueError):
    """Raised when a supplemental geometry request cannot be proven safely."""


class CameraCalibrationProfile(StrictModel):
    """Versioned camera calibration metadata used by a derived view.

    Matrix values are metadata only in the local fallback.  A hardware/OpenCV
    adapter may use the same profile to construct a map, but the profile digest
    is part of every derived frame identity regardless of the adapter.
    """

    profile_id: NonEmptyString
    version: SchemaVersion
    camera_id: CameraId
    source_width: PositiveInt
    source_height: PositiveInt
    intrinsic_matrix: tuple[float | int, ...] = ()
    distortion_coefficients: tuple[float | int, ...] = ()
    projection_matrix: tuple[float | int, ...] = ()
    map_policy_version: SchemaVersion = GEOMETRY_MAP_POLICY_VERSION

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        values = (
            *self.intrinsic_matrix,
            *self.distortion_coefficients,
            *self.projection_matrix,
        )
        if any(type(value) not in (int, float) or not isfinite(float(value)) for value in values):
            raise ValueError("calibration matrix values must be finite floats")
        if self.intrinsic_matrix and len(self.intrinsic_matrix) not in (9,):
            raise ValueError("intrinsic_matrix must contain nine values when supplied")
        if self.projection_matrix and len(self.projection_matrix) not in (9,):
            raise ValueError("projection_matrix must contain nine values when supplied")
        return self

    @property
    def semantic_sha256(self) -> Sha256Digest:
        """Digest of the complete calibration profile, excluding no fields."""

        return camera_calibration_profile_sha256(self)

    @property
    def calibration_sha256(self) -> Sha256Digest:
        """Descriptive alias used by lineage consumers."""

        return self.semantic_sha256


class FramePreprocessPolicy(StrictModel):
    """Versioned policy for one selected derived frame view.

    Geometry is supplemental by construction.  It can be requested for selected
    windows/model inputs, but it never replaces the raw source or blocks a source
    quality result.  ``pass_through`` is explicit so a local CPU fallback remains
    deterministic and auditable.
    """

    version: SchemaVersion = DEFAULT_PREPROCESS_POLICY_VERSION
    view: GeometryView = GeometryView.RAW
    output_width: PositiveInt | None = None
    output_height: PositiveInt | None = None
    interpolation: GeometryInterpolation = GeometryInterpolation.NEAREST
    pass_through: bool = True
    selected_only: bool = True
    supplemental_only: bool = True

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("output_width and output_height must be supplied together")
        if self.view is GeometryView.RAW and not self.pass_through:
            raise ValueError("RAW preprocessing cannot disable pass-through")
        if not self.selected_only:
            raise ValueError("geometry preprocessing must remain selected-window bounded")
        if not self.supplemental_only:
            raise ValueError("geometry preprocessing must remain supplemental")
        return self

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return frame_preprocess_policy_sha256(self)


DEFAULT_FRAME_PREPROCESS_POLICY: Final = FramePreprocessPolicy()


class DerivedGeometryFrame(StrictModel):
    """Exact facts for one ephemeral or persisted derived geometry payload."""

    frame_id: NonEmptyString
    source_frame_id: NonEmptyString
    source_content_sha256: Sha256Digest
    camera_id: CameraId
    source_timestamp_ns: Nanoseconds
    aligned_timestamp_ns: Nanoseconds
    view: GeometryView
    calibration_sha256: Sha256Digest | None
    preprocess_policy_version: SchemaVersion
    preprocess_policy_sha256: Sha256Digest
    source_width: PositiveInt
    source_height: PositiveInt
    output_width: PositiveInt
    output_height: PositiveInt
    exact_bytes_sha256: Sha256Digest
    bytes: PositiveInt
    media_type: NonEmptyString = "application/octet-stream"
    uri: str | None = None
    lineage_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_geometry_fact(self) -> Self:
        if self.view is not GeometryView.RAW and self.calibration_sha256 is None:
            raise ValueError("supplemental geometry facts require a calibration digest")
        if self.uri is not None and (not isinstance(self.uri, str) or not self.uri):
            raise ValueError("geometry artifact uri must be a non-empty string when supplied")
        expected_lineage = semantic_sha256(
            geometry_lineage_projection(
                source_frame_id=self.source_frame_id,
                source_content_sha256=self.source_content_sha256,
                camera_id=self.camera_id,
                source_timestamp_ns=self.source_timestamp_ns,
                aligned_timestamp_ns=self.aligned_timestamp_ns,
                view=self.view,
                calibration_sha256=self.calibration_sha256,
                preprocess_policy_version=self.preprocess_policy_version,
                preprocess_policy_sha256=self.preprocess_policy_sha256,
                source_width=self.source_width,
                source_height=self.source_height,
                output_width=self.output_width,
                output_height=self.output_height,
                exact_bytes_sha256=self.exact_bytes_sha256,
                bytes=self.bytes,
            )
        )
        if self.lineage_sha256 != expected_lineage:
            raise ValueError("lineage_sha256 does not match the complete geometry lineage")
        expected_identity = semantic_sha256(
            {
                "projection_version": GEOMETRY_PROJECTION_VERSION,
                "lineage_sha256": expected_lineage,
            }
        )
        expected_frame_id = str(
            uuid5(NAMESPACE_URL, f"robata:derived-geometry:{expected_identity}")
        )
        if self.frame_id != expected_frame_id:
            raise ValueError("frame_id does not match the geometry lineage")
        return self

    @property
    def derived_frame_id(self) -> str:
        """Stable alias for callers that distinguish source and derived IDs."""

        return self.frame_id


@dataclass(frozen=True, slots=True)
class DerivedGeometryArtifact:
    """Payload plus exact facts; callers may persist the bytes through an artifact port."""

    fact: DerivedGeometryFrame
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.fact, DerivedGeometryFrame):
            raise TypeError("fact must be a DerivedGeometryFrame")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise TypeError("derived geometry payload must be non-empty immutable bytes")
        if len(self.payload) != self.fact.bytes:
            raise ValueError("derived geometry payload length differs from its fact")
        if exact_bytes_sha256(self.payload) != self.fact.exact_bytes_sha256:
            raise ValueError("derived geometry payload digest differs from its fact")


class GeometryMapCache:
    """Small bounded cache of deterministic source-pixel maps."""

    def __init__(self, *, max_entries: int = 64) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._maps: OrderedDict[tuple[str, str, int, int, int, int], tuple[int, ...]] = (
            OrderedDict()
        )

    @property
    def size(self) -> int:
        return len(self._maps)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def get_or_create(
        self,
        *,
        calibration: CameraCalibrationProfile | None,
        policy: FramePreprocessPolicy,
        source_width: int,
        source_height: int,
        output_width: int,
        output_height: int,
    ) -> tuple[int, ...]:
        if not isinstance(policy, FramePreprocessPolicy):
            raise TypeError("policy must be a FramePreprocessPolicy")
        if calibration is not None and not isinstance(calibration, CameraCalibrationProfile):
            raise TypeError("calibration must be a CameraCalibrationProfile or None")
        key = (
            "none" if calibration is None else calibration.semantic_sha256,
            policy.semantic_sha256,
            source_width,
            source_height,
            output_width,
            output_height,
        )
        cached = self._maps.get(key)
        if cached is not None:
            self._maps.move_to_end(key)
            return cached
        mapping = _build_nearest_map(
            source_width,
            source_height,
            output_width,
            output_height,
        )
        self._maps[key] = mapping
        self._maps.move_to_end(key)
        while len(self._maps) > self._max_entries:
            self._maps.popitem(last=False)
        return mapping


DEFAULT_GEOMETRY_MAP_CACHE: Final = GeometryMapCache()


def camera_calibration_profile_projection(
    profile: CameraCalibrationProfile,
) -> dict[str, object]:
    """Return the explicit semantic projection used for profile identity."""

    if not isinstance(profile, CameraCalibrationProfile):
        raise TypeError("profile must be a CameraCalibrationProfile")
    return {
        "projection_version": GEOMETRY_PROJECTION_VERSION,
        "profile_id": profile.profile_id,
        "version": profile.version,
        "camera_id": profile.camera_id.value,
        "source_width": profile.source_width,
        "source_height": profile.source_height,
        "intrinsic_matrix": profile.intrinsic_matrix,
        "distortion_coefficients": profile.distortion_coefficients,
        "projection_matrix": profile.projection_matrix,
        "map_policy_version": profile.map_policy_version,
    }


def camera_calibration_profile_sha256(profile: CameraCalibrationProfile) -> Sha256Digest:
    return semantic_sha256(camera_calibration_profile_projection(profile))


def frame_preprocess_policy_projection(policy: FramePreprocessPolicy) -> dict[str, object]:
    """Return the complete semantic projection of a preprocess policy."""

    if not isinstance(policy, FramePreprocessPolicy):
        raise TypeError("policy must be a FramePreprocessPolicy")
    return {
        "projection_version": GEOMETRY_PROJECTION_VERSION,
        "version": policy.version,
        "view": policy.view.value,
        "output_width": policy.output_width,
        "output_height": policy.output_height,
        "interpolation": policy.interpolation.value,
        "pass_through": policy.pass_through,
        "selected_only": policy.selected_only,
        "supplemental_only": policy.supplemental_only,
    }


def frame_preprocess_policy_sha256(policy: FramePreprocessPolicy) -> Sha256Digest:
    return semantic_sha256(frame_preprocess_policy_projection(policy))


def geometry_lineage_projection(
    *,
    source_frame_id: str,
    source_content_sha256: str,
    camera_id: CameraId,
    source_timestamp_ns: int,
    aligned_timestamp_ns: int,
    view: GeometryView,
    calibration_sha256: str | None,
    preprocess_policy_version: str,
    preprocess_policy_sha256: str,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    exact_bytes_sha256: str,
    bytes: int,
) -> dict[str, object]:
    """Build the exact lineage projection shared by facts and frame IDs."""

    return {
        "projection_version": GEOMETRY_LINEAGE_PROJECTION_VERSION,
        "source_frame_id": source_frame_id,
        "source_content_sha256": source_content_sha256,
        "camera_id": camera_id.value,
        "source_timestamp_ns": str(source_timestamp_ns),
        "aligned_timestamp_ns": str(aligned_timestamp_ns),
        "view": view.value,
        "calibration_sha256": calibration_sha256,
        "preprocess_policy_version": preprocess_policy_version,
        "preprocess_policy_sha256": preprocess_policy_sha256,
        "source_width": source_width,
        "source_height": source_height,
        "output_width": output_width,
        "output_height": output_height,
        "exact_bytes_sha256": exact_bytes_sha256,
        "bytes": bytes,
    }


def materialize_geometry_view(
    frame: DecodedFrameView,
    *,
    source_frame_id: str,
    source_content_sha256: Sha256Digest,
    camera_id: CameraId,
    source_timestamp_ns: int,
    aligned_timestamp_ns: int,
    policy: FramePreprocessPolicy | None = None,
    calibration: CameraCalibrationProfile | None = None,
    map_cache: GeometryMapCache = DEFAULT_GEOMETRY_MAP_CACHE,
    media_type: str = "application/octet-stream",
    uri: str | None = None,
) -> DerivedGeometryArtifact:
    """Produce one bounded derived view and exact source/time lineage.

    This function never retains a full-resolution RGB frame.  The caller supplies
    the compact :class:`DecodedFrameView`; the returned payload is either its
    grayscale bytes or a bounded nearest-neighbour projection of those bytes.
    """

    if not isinstance(frame, DecodedFrameView):
        raise TypeError("frame must be a DecodedFrameView")
    if policy is None:
        policy = DEFAULT_FRAME_PREPROCESS_POLICY
    if not isinstance(policy, FramePreprocessPolicy):
        raise TypeError("policy must be a FramePreprocessPolicy")
    resolved_policy = policy
    if not isinstance(camera_id, CameraId):
        raise TypeError("camera_id must be a CameraId")
    if not isinstance(source_frame_id, str) or not source_frame_id:
        raise ValueError("source_frame_id must be a non-empty string")
    if not isinstance(source_content_sha256, str) or len(source_content_sha256) != 64:
        raise ValueError("source_content_sha256 must be a SHA-256 digest")
    if not isinstance(source_timestamp_ns, int) or isinstance(source_timestamp_ns, bool):
        raise TypeError("source_timestamp_ns must be an integer")
    if not isinstance(aligned_timestamp_ns, int) or isinstance(aligned_timestamp_ns, bool):
        raise TypeError("aligned_timestamp_ns must be an integer")
    if frame.timestamp_ns != aligned_timestamp_ns:
        raise GeometryProcessingError(
            "decoded frame timestamp differs from aligned geometry timestamp"
        )
    if calibration is not None:
        if not isinstance(calibration, CameraCalibrationProfile):
            raise TypeError("calibration must be a CameraCalibrationProfile or None")
        if calibration.camera_id is not camera_id:
            raise GeometryProcessingError("calibration camera differs from the source camera")
        if (calibration.source_width, calibration.source_height) != (frame.width, frame.height):
            raise GeometryProcessingError("calibration dimensions differ from the decoded frame")
    if resolved_policy.view is not GeometryView.RAW and calibration is None:
        raise GeometryProcessingError("supplemental geometry views require calibration")
    if resolved_policy.view is not GeometryView.RAW and not resolved_policy.pass_through:
        raise GeometryProcessingError(
            "the local geometry adapter only supports deterministic pass-through maps"
        )
    if not isinstance(map_cache, GeometryMapCache):
        raise TypeError("map_cache must be a GeometryMapCache")
    output_width = (
        frame.width if resolved_policy.output_width is None else resolved_policy.output_width
    )
    output_height = (
        frame.height if resolved_policy.output_height is None else resolved_policy.output_height
    )
    calibration_sha256 = None if calibration is None else calibration.semantic_sha256
    mapping = map_cache.get_or_create(
        calibration=calibration,
        policy=resolved_policy,
        source_width=frame.width,
        source_height=frame.height,
        output_width=output_width,
        output_height=output_height,
    )
    payload = bytes(frame.gray_pixels[index] for index in mapping)
    exact_digest = exact_bytes_sha256(payload)
    lineage = geometry_lineage_projection(
        source_frame_id=source_frame_id,
        source_content_sha256=source_content_sha256,
        camera_id=camera_id,
        source_timestamp_ns=source_timestamp_ns,
        aligned_timestamp_ns=aligned_timestamp_ns,
        view=resolved_policy.view,
        calibration_sha256=calibration_sha256,
        preprocess_policy_version=resolved_policy.version,
        preprocess_policy_sha256=resolved_policy.semantic_sha256,
        source_width=frame.width,
        source_height=frame.height,
        output_width=output_width,
        output_height=output_height,
        exact_bytes_sha256=exact_digest,
        bytes=len(payload),
    )
    lineage_digest = semantic_sha256(lineage)
    identity = semantic_sha256(
        {
            "projection_version": GEOMETRY_PROJECTION_VERSION,
            "lineage_sha256": lineage_digest,
        }
    )
    fact = DerivedGeometryFrame(
        frame_id=str(uuid5(NAMESPACE_URL, f"robata:derived-geometry:{identity}")),
        source_frame_id=source_frame_id,
        source_content_sha256=source_content_sha256,
        camera_id=camera_id,
        source_timestamp_ns=source_timestamp_ns,
        aligned_timestamp_ns=aligned_timestamp_ns,
        view=resolved_policy.view,
        calibration_sha256=calibration_sha256,
        preprocess_policy_version=resolved_policy.version,
        preprocess_policy_sha256=resolved_policy.semantic_sha256,
        source_width=frame.width,
        source_height=frame.height,
        output_width=output_width,
        output_height=output_height,
        exact_bytes_sha256=exact_digest,
        bytes=len(payload),
        media_type=media_type,
        uri=uri,
        lineage_sha256=lineage_digest,
    )
    return DerivedGeometryArtifact(fact=fact, payload=payload)


def _build_nearest_map(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[int, ...]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (source_width, source_height, output_width, output_height)
    ):
        raise ValueError("geometry map dimensions must be positive integers")
    indexes: list[int] = []
    for y in range(output_height):
        source_y = min(source_height - 1, (y * source_height) // output_height)
        row = source_y * source_width
        for x in range(output_width):
            source_x = min(source_width - 1, (x * source_width) // output_width)
            indexes.append(row + source_x)
    return tuple(indexes)


__all__ = [
    "DEFAULT_FRAME_PREPROCESS_POLICY",
    "DEFAULT_GEOMETRY_MAP_CACHE",
    "GEOMETRY_LINEAGE_PROJECTION_VERSION",
    "GEOMETRY_MAP_POLICY_VERSION",
    "GEOMETRY_PROJECTION_VERSION",
    "CameraCalibrationProfile",
    "DerivedGeometryArtifact",
    "DerivedGeometryFrame",
    "FramePreprocessPolicy",
    "GeometryInterpolation",
    "GeometryMapCache",
    "GeometryProcessingError",
    "GeometryView",
    "camera_calibration_profile_projection",
    "camera_calibration_profile_sha256",
    "frame_preprocess_policy_projection",
    "frame_preprocess_policy_sha256",
    "geometry_lineage_projection",
    "materialize_geometry_view",
]
