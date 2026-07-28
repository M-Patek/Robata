"""Target-media support and provenance boundary for optional NVDEC adapters.

The CUDA/DeepStream runtime remains optional. A target deployment supplies a backend
through the existing media ports, while this module records the concrete runtime and
enforces that CPU fallback happens only before target output becomes visible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.runtime.observability import RuntimeObserver, runtime_increment

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NVDEC_MEDIA_RUNTIME_PROVENANCE_VERSION: Final[Literal["media-runtime-provenance-v2"]] = (
    "media-runtime-provenance-v2"
)


class NvdecFallbackReason(StrEnum):
    """GPU conditions for which retrying with the CPU reference is safe."""

    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    DEVICE_FAILED = "DEVICE_FAILED"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"


class MediaRuntimeBackend(StrEnum):
    """The media runtime actually used for one bounded operation."""

    CPU_REFERENCE = "CPU_REFERENCE"
    NVDEC_TARGET = "NVDEC_TARGET"
    CPU_FALLBACK = "CPU_FALLBACK"


class NvdecInputProfile(StrictModel):
    """Codec/profile/dimension facts used for target support selection."""

    codec: NonEmptyString
    profile: NonEmptyString
    width: PositiveInt
    height: PositiveInt

    @model_validator(mode="after")
    def validate_normalized(self) -> Self:
        if self.codec != self.codec.strip().lower():
            raise ValueError("codec must be lowercase and trimmed")
        if self.profile != self.profile.strip().lower():
            raise ValueError("profile must be lowercase and trimmed")
        return self


class NvdecSupportedInput(StrictModel):
    """One declared target decoder support envelope."""

    codec: NonEmptyString
    profiles: tuple[NonEmptyString, ...]
    max_width: PositiveInt
    max_height: PositiveInt

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.codec != self.codec.strip().lower():
            raise ValueError("codec must be lowercase and trimmed")
        if not self.profiles:
            raise ValueError("profiles must not be empty")
        if any(profile != profile.strip().lower() for profile in self.profiles):
            raise ValueError("profiles must be lowercase and trimmed")
        if tuple(sorted(set(self.profiles))) != self.profiles:
            raise ValueError("profiles must be unique and ordered")
        return self

    def supports(self, profile: NvdecInputProfile) -> bool:
        if not isinstance(profile, NvdecInputProfile):
            raise TypeError("profile must be an NvdecInputProfile")
        return (
            profile.codec == self.codec
            and profile.profile in self.profiles
            and profile.width <= self.max_width
            and profile.height <= self.max_height
        )


class MediaRuntimeProvenance(StrictModel):
    """Internal media-runtime fact, separate from inference CapabilitySnapshot."""

    schema_version: Literal["1.0"]
    provenance_version: Literal["media-runtime-provenance-v2"]
    provenance_sha256: Sha256Digest
    backend: MediaRuntimeBackend
    implementation: NonEmptyString
    implementation_version: NonEmptyString
    selected_input: NvdecInputProfile | None = None
    supported_inputs: tuple[NvdecSupportedInput, ...] = ()
    fallback_reasons: tuple[NvdecFallbackReason, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        backend: MediaRuntimeBackend,
        implementation: str,
        implementation_version: str,
        selected_input: NvdecInputProfile | None = None,
        supported_inputs: tuple[NvdecSupportedInput, ...] = (),
        fallback_reasons: tuple[NvdecFallbackReason, ...] = (),
    ) -> Self:
        draft = cls.model_construct(
            schema_version="1.0",
            provenance_version=NVDEC_MEDIA_RUNTIME_PROVENANCE_VERSION,
            provenance_sha256="0" * 64,
            backend=backend,
            implementation=implementation,
            implementation_version=implementation_version,
            selected_input=selected_input,
            supported_inputs=supported_inputs,
            fallback_reasons=fallback_reasons,
        )
        return cls.model_validate(
            {
                **draft.model_dump(mode="python"),
                "provenance_sha256": semantic_sha256(
                    media_runtime_provenance_projection(draft, include_digest=False)
                ),
            },
            strict=True,
        )

    @classmethod
    def cpu_reference(cls) -> Self:
        """Return reference-path provenance without claiming target hardware."""

        return cls.create(
            backend=MediaRuntimeBackend.CPU_REFERENCE,
            implementation="robata.pyav_h264_mp4_exporter",
            implementation_version="0.1.0",
        )

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.backend is MediaRuntimeBackend.NVDEC_TARGET:
            if self.selected_input is None or not self.supported_inputs:
                raise ValueError("NVDEC provenance requires selected input and support matrix")
            if not any(item.supports(self.selected_input) for item in self.supported_inputs):
                raise ValueError("selected NVDEC input is outside the declared support matrix")
            if self.fallback_reasons:
                raise ValueError("selected NVDEC provenance cannot carry fallback reasons")
        elif self.backend is MediaRuntimeBackend.CPU_FALLBACK:
            if not self.fallback_reasons:
                raise ValueError("CPU fallback provenance requires fallback reasons")
        elif self.fallback_reasons:
            raise ValueError("CPU reference provenance cannot carry fallback reasons")
        if self.fallback_reasons != tuple(
            sorted(set(self.fallback_reasons), key=lambda reason: reason.value)
        ):
            raise ValueError("fallback reasons must be unique and ordered")
        if self.provenance_sha256 != semantic_sha256(
            media_runtime_provenance_projection(self, include_digest=False)
        ):
            raise ValueError("provenance_sha256 does not match media runtime facts")
        return self


def media_runtime_provenance_projection(
    provenance: MediaRuntimeProvenance,
    *,
    include_digest: bool = True,
) -> dict[str, Any]:
    """Return the content-addressed internal media-runtime projection."""

    if not isinstance(provenance, MediaRuntimeProvenance):
        raise TypeError("provenance must be a MediaRuntimeProvenance")
    projection = provenance.model_dump(mode="json")
    if not include_digest:
        projection.pop("provenance_sha256", None)
    return projection


def media_output_snapshot(paths: Iterable[Path]) -> tuple[tuple[str, str], ...]:
    """Return an exact recursive snapshot used to reject post-output fallback.

    The snapshot includes every path under a caller-owned output root. It is a
    safety guard only, never an artifact identity or cache key.
    """

    checked = tuple(Path(path) for path in paths)
    entries: list[tuple[str, str]] = []
    for index, root in enumerate(checked):
        prefix = str(index)
        if root.is_symlink():
            entries.append((prefix, "SYMLINK"))
            continue
        if not root.exists():
            entries.append((prefix, "MISSING"))
            continue
        if root.is_file():
            entries.append((prefix, _media_output_file_state(root)))
            continue
        if not root.is_dir():
            entries.append((prefix, "OTHER"))
            continue
        entries.append((prefix, "DIRECTORY"))
        try:
            children = tuple(sorted(root.rglob("*"), key=lambda path: path.as_posix()))
        except OSError as error:
            entries.append((prefix, f"ERROR:{type(error).__name__}"))
            continue
        for child in children:
            relative = child.relative_to(root).as_posix()
            key = f"{prefix}/{relative}"
            if child.is_symlink():
                entries.append((key, "SYMLINK"))
            elif child.is_file():
                entries.append((key, _media_output_file_state(child)))
            elif child.is_dir():
                entries.append((key, "DIRECTORY"))
            else:
                entries.append((key, "OTHER"))
    return tuple(entries)


def _media_output_file_state(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"FILE:{path.stat().st_size}:{digest}"
    except OSError as error:
        return f"ERROR:{type(error).__name__}"


class NvdecBackendUnavailableError(RuntimeError):
    """Signal a target failure that may permit a CPU retry before publication.

    Set ``publication_started`` once target work has made any dependent output or
    callback-visible evidence externally observable, not only after final files exist.
    """

    def __init__(
        self,
        reason: NvdecFallbackReason,
        message: str,
        *,
        publication_started: bool = False,
    ) -> None:
        if not isinstance(reason, NvdecFallbackReason):
            raise TypeError("reason must be an NvdecFallbackReason")
        if not isinstance(publication_started, bool):
            raise TypeError("publication_started must be a bool")
        super().__init__(message)
        self.reason = reason
        self.publication_started = publication_started


def nvdec_fallback_allowed(
    error: NvdecBackendUnavailableError,
    *,
    output_changed: bool = False,
) -> bool:
    """Return whether a CPU retry remains safe before dependent publication."""

    if not isinstance(error, NvdecBackendUnavailableError):
        raise TypeError("error must be an NvdecBackendUnavailableError")
    if not isinstance(output_changed, bool):
        raise TypeError("output_changed must be a bool")
    return not error.publication_started and not output_changed


def record_nvdec_fallback(
    runtime_observer: RuntimeObserver | None,
    *,
    operation: str,
    reason: NvdecFallbackReason,
) -> None:
    """Record a CPU retry without allowing telemetry to affect media correctness."""

    runtime_increment(
        runtime_observer,
        "media.nvdec.fallbacks",
        attributes={"operation": operation, "reason": reason.value},
    )


__all__ = [
    "NVDEC_MEDIA_RUNTIME_PROVENANCE_VERSION",
    "MediaRuntimeBackend",
    "MediaRuntimeProvenance",
    "NvdecBackendUnavailableError",
    "NvdecFallbackReason",
    "NvdecInputProfile",
    "NvdecSupportedInput",
    "media_output_snapshot",
    "media_runtime_provenance_projection",
    "nvdec_fallback_allowed",
    "record_nvdec_fallback",
]
