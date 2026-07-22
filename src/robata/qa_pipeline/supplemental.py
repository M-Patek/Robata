"""Deterministic QA_DENSE consumption of local supplemental visual packages."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.sampling.grid import SelectionStatus
from robata.sampling.materializer import MaterializedArtifactManifest
from robata.sampling.supplemental import (
    MaterializedSupplementalPackage,
    ProviderNeutralSupplementalPackage,
    SupplementalEvidenceClass,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

SUPPLEMENTAL_QA_DENSE_INPUT_PROJECTION_VERSION = "supplemental-qa-dense-input-semantic-v2"
SUPPLEMENTAL_QA_DENSE_RESULT_PROJECTION_VERSION = "supplemental-qa-dense-result-semantic-v2"
LOCAL_SUPPLEMENTAL_QA_DENSE_POLICY_VERSION = "local-supplemental-qa-dense-consumer-v2-png-decode-v2"


class SupplementalQaDenseStatus(StrEnum):
    """Whether every requested coordinate resolved to consumable frame evidence."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class SupplementalQaDenseInputPlan(StrictModel):
    """Run-independent input identity for one exact supplemental package."""

    input_plan_id: NonEmptyString
    semantic_sha256: Sha256Digest
    task: Literal["QA_DENSE"] = "QA_DENSE"
    package_id: NonEmptyString
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest
    target_plan_id: NonEmptyString
    target_plan_semantic_sha256: Sha256Digest
    consumer_policy_version: SchemaVersion
    projection_version: Literal["supplemental-qa-dense-input-semantic-v2"] = (
        "supplemental-qa-dense-input-semantic-v2"
    )
    evidence_class: Literal[SupplementalEvidenceClass.LOCAL_CONFORMANCE] = (
        SupplementalEvidenceClass.LOCAL_CONFORMANCE
    )
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected_digest = semantic_sha256(supplemental_qa_dense_input_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match supplemental QA_DENSE input")
        if self.input_plan_id != _stable_id("supplemental-qa-dense-input-v2", expected_digest):
            raise ValueError("input_plan_id does not match supplemental QA_DENSE input")
        return self


def supplemental_qa_dense_input_projection(
    plan: SupplementalQaDenseInputPlan,
) -> dict[str, object]:
    """Exclude exact manifest bytes from logical identity while retaining audit pin."""

    return {
        "projection_version": plan.projection_version,
        "task": plan.task,
        "package_id": plan.package_id,
        "package_semantic_content_sha256": plan.package_semantic_content_sha256,
        "target_plan_id": plan.target_plan_id,
        "target_plan_semantic_sha256": plan.target_plan_semantic_sha256,
        "consumer_policy_version": plan.consumer_policy_version,
        "evidence_class": plan.evidence_class.value,
        "production_eligible": plan.production_eligible,
    }


class SupplementalQaDenseConsumption(StrictModel):
    """One target result the deterministic mock inspected from the package."""

    target_ordinal: NonNegativeInt
    camera_id: CameraId
    target_ns: Nanoseconds
    package_status: SelectionStatus
    effective_artifact_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_consumption(self) -> Self:
        has_evidence = self.package_status in {
            SelectionStatus.SELECTED,
            SelectionStatus.DEDUPLICATED_FRAME,
        }
        if has_evidence != (self.effective_artifact_sha256 is not None):
            raise ValueError("consumption artifact availability disagrees with package status")
        return self


class SupplementalQaDenseResult(StrictModel):
    """Local mock result bound to exact input, package, and consumed artifacts."""

    result_id: NonEmptyString
    semantic_sha256: Sha256Digest
    task: Literal["QA_DENSE"] = "QA_DENSE"
    input_plan_id: NonEmptyString
    input_plan_semantic_sha256: Sha256Digest
    package_id: NonEmptyString
    package_semantic_content_sha256: Sha256Digest
    package_manifest_sha256: Sha256Digest
    consumptions: tuple[SupplementalQaDenseConsumption, ...]
    status: SupplementalQaDenseStatus
    consumer_policy_version: SchemaVersion
    projection_version: Literal["supplemental-qa-dense-result-semantic-v2"] = (
        "supplemental-qa-dense-result-semantic-v2"
    )
    evidence_class: Literal[SupplementalEvidenceClass.LOCAL_CONFORMANCE] = (
        SupplementalEvidenceClass.LOCAL_CONFORMANCE
    )
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if tuple(item.target_ordinal for item in self.consumptions) != tuple(
            range(len(self.consumptions))
        ):
            raise ValueError("supplemental QA_DENSE must consume every target in order")
        expected_status = (
            SupplementalQaDenseStatus.COMPLETE
            if self.consumptions
            and all(item.effective_artifact_sha256 is not None for item in self.consumptions)
            else SupplementalQaDenseStatus.INCOMPLETE
        )
        if self.status is not expected_status:
            raise ValueError("supplemental QA_DENSE status does not match consumed evidence")
        expected_digest = semantic_sha256(supplemental_qa_dense_result_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match supplemental QA_DENSE result")
        if self.result_id != _stable_id("supplemental-qa-dense-result-v2", expected_digest):
            raise ValueError("result_id does not match supplemental QA_DENSE result")
        return self


def supplemental_qa_dense_result_projection(
    result: SupplementalQaDenseResult,
) -> dict[str, object]:
    """Return the complete versioned local-consumption projection."""

    return {
        "projection_version": result.projection_version,
        "task": result.task,
        "input_plan_id": result.input_plan_id,
        "input_plan_semantic_sha256": result.input_plan_semantic_sha256,
        "package_id": result.package_id,
        "package_semantic_content_sha256": result.package_semantic_content_sha256,
        "consumptions": result.consumptions,
        "status": result.status.value,
        "consumer_policy_version": result.consumer_policy_version,
        "evidence_class": result.evidence_class.value,
        "production_eligible": result.production_eligible,
    }


class DeterministicSupplementalQaDenseConsumer:
    """Consume package evidence without pretending to classify visual semantics."""

    def __init__(
        self,
        policy_version: str = LOCAL_SUPPLEMENTAL_QA_DENSE_POLICY_VERSION,
    ) -> None:
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("consumer policy version must be non-empty")
        self._policy_version = policy_version

    def prepare(
        self, materialized: MaterializedSupplementalPackage
    ) -> SupplementalQaDenseInputPlan:
        """Create the provider-neutral input identity before local mock consumption."""

        if not isinstance(materialized, MaterializedSupplementalPackage):
            raise TypeError("materialized must be a MaterializedSupplementalPackage")
        package = _validated_package(materialized.package)
        values: dict[str, Any] = {
            "task": "QA_DENSE",
            "package_id": package.package_id,
            "package_semantic_content_sha256": package.semantic_content_sha256,
            "package_manifest_sha256": materialized.manifest_sha256,
            "target_plan_id": package.target_plan_id,
            "target_plan_semantic_sha256": package.target_plan_semantic_sha256,
            "consumer_policy_version": self._policy_version,
            "projection_version": SUPPLEMENTAL_QA_DENSE_INPUT_PROJECTION_VERSION,
            "evidence_class": SupplementalEvidenceClass.LOCAL_CONFORMANCE,
            "production_eligible": False,
        }
        draft = SupplementalQaDenseInputPlan.model_construct(
            input_plan_id="pending",
            semantic_sha256="0" * 64,
            **values,
        )
        digest = semantic_sha256(supplemental_qa_dense_input_projection(draft))
        return SupplementalQaDenseInputPlan.model_validate(
            {
                **values,
                "input_plan_id": _stable_id("supplemental-qa-dense-input-v2", digest),
                "semantic_sha256": digest,
            },
            strict=True,
        )

    def consume(
        self,
        materialized: MaterializedSupplementalPackage,
        input_plan: SupplementalQaDenseInputPlan,
        *,
        artifact_bytes_resolver: Callable[[MaterializedArtifactManifest], bytes],
    ) -> SupplementalQaDenseResult:
        """Read and verify every selected artifact before retaining its digest."""

        if not isinstance(materialized, MaterializedSupplementalPackage):
            raise TypeError("materialized must be a MaterializedSupplementalPackage")
        package = _validated_package(materialized.package)
        plan = SupplementalQaDenseInputPlan.model_validate(
            input_plan.model_dump(mode="python"), strict=True
        )
        if (
            plan.package_id != package.package_id
            or plan.package_semantic_content_sha256 != package.semantic_content_sha256
            or plan.package_manifest_sha256 != materialized.manifest_sha256
            or plan.target_plan_id != package.target_plan_id
            or plan.target_plan_semantic_sha256 != package.target_plan_semantic_sha256
            or plan.consumer_policy_version != self._policy_version
        ):
            raise ValueError("supplemental QA_DENSE input does not bind the package")

        if not callable(artifact_bytes_resolver):
            raise TypeError("artifact_bytes_resolver must be callable")
        selected_artifacts: dict[int, str] = {}
        for item in package.outcomes:
            if item.selected_artifact is None:
                continue
            artifact = item.selected_artifact.artifact
            if artifact.media_type != "image/png":
                raise ValueError("supplemental QA_DENSE artifact media type is unsupported")
            contents = artifact_bytes_resolver(artifact)
            if not isinstance(contents, bytes):
                raise TypeError("artifact_bytes_resolver must return exact bytes")
            if len(contents) != artifact.bytes:
                raise ValueError("supplemental QA_DENSE artifact byte count is inconsistent")
            if sha256(contents).hexdigest() != artifact.sha256:
                raise ValueError("supplemental QA_DENSE artifact SHA-256 is inconsistent")
            _validate_png(
                contents,
                expected_width=item.selected_artifact.width,
                expected_height=item.selected_artifact.height,
            )
            selected_artifacts[item.target.ordinal] = artifact.sha256
        consumptions = tuple(
            SupplementalQaDenseConsumption(
                target_ordinal=item.target.ordinal,
                camera_id=item.target.camera_id,
                target_ns=item.target.target_ns,
                package_status=item.status,
                effective_artifact_sha256=(
                    item.selected_artifact.artifact.sha256
                    if item.selected_artifact is not None
                    else (
                        selected_artifacts[item.reused_selected_target_ordinal]
                        if item.reused_selected_target_ordinal is not None
                        else None
                    )
                ),
            )
            for item in package.outcomes
        )
        status = (
            SupplementalQaDenseStatus.COMPLETE
            if consumptions
            and all(item.effective_artifact_sha256 is not None for item in consumptions)
            else SupplementalQaDenseStatus.INCOMPLETE
        )
        values: dict[str, Any] = {
            "task": "QA_DENSE",
            "input_plan_id": plan.input_plan_id,
            "input_plan_semantic_sha256": plan.semantic_sha256,
            "package_id": package.package_id,
            "package_semantic_content_sha256": package.semantic_content_sha256,
            "package_manifest_sha256": materialized.manifest_sha256,
            "consumptions": consumptions,
            "status": status,
            "consumer_policy_version": self._policy_version,
            "projection_version": SUPPLEMENTAL_QA_DENSE_RESULT_PROJECTION_VERSION,
            "evidence_class": SupplementalEvidenceClass.LOCAL_CONFORMANCE,
            "production_eligible": False,
        }
        draft = SupplementalQaDenseResult.model_construct(
            result_id="pending",
            semantic_sha256="0" * 64,
            **values,
        )
        digest = semantic_sha256(supplemental_qa_dense_result_projection(draft))
        return SupplementalQaDenseResult.model_validate(
            {
                **values,
                "result_id": _stable_id("supplemental-qa-dense-result-v2", digest),
                "semantic_sha256": digest,
            },
            strict=True,
        )


def _validate_png(contents: bytes, *, expected_width: int, expected_height: int) -> None:
    """Decode the governed 8-bit non-interlaced PNG subset without retaining pixels."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not contents.startswith(signature):
        raise ValueError("supplemental QA_DENSE artifact lacks a PNG signature")

    offset = len(signature)
    width = height = channels = 0
    saw_ihdr = False
    saw_idat = False
    idat_closed = False
    saw_iend = False
    compressed = bytearray()
    while offset < len(contents):
        if len(contents) - offset < 12:
            raise ValueError("supplemental QA_DENSE PNG chunk is truncated")
        chunk_length = int.from_bytes(contents[offset : offset + 4], "big")
        chunk_type = contents[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(contents):
            raise ValueError("supplemental QA_DENSE PNG chunk length is corrupt")
        chunk_data = contents[offset + 8 : offset + 8 + chunk_length]
        expected_crc = int.from_bytes(contents[chunk_end - 4 : chunk_end], "big")
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("supplemental QA_DENSE PNG chunk CRC is corrupt")

        if not saw_ihdr:
            if chunk_type != b"IHDR" or chunk_length != 13:
                raise ValueError("supplemental QA_DENSE PNG requires a leading IHDR")
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth, color_type, compression, filtering, interlace = chunk_data[8:13]
            channels_by_color_type = {0: 1, 2: 3, 4: 2, 6: 4}
            channels = channels_by_color_type.get(color_type, 0)
            if width != expected_width or height != expected_height:
                raise ValueError("supplemental QA_DENSE PNG dimensions are inconsistent")
            if (
                width < 1
                or height < 1
                or bit_depth != 8
                or channels == 0
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ValueError("supplemental QA_DENSE PNG encoding is unsupported")
            saw_ihdr = True
        elif chunk_type == b"IHDR":
            raise ValueError("supplemental QA_DENSE PNG contains duplicate IHDR")
        elif chunk_type == b"IDAT":
            if idat_closed:
                raise ValueError("supplemental QA_DENSE PNG IDAT chunks must be consecutive")
            saw_idat = True
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            if chunk_length != 0 or chunk_end != len(contents):
                raise ValueError("supplemental QA_DENSE PNG IEND is inconsistent")
            saw_iend = True
        else:
            if saw_idat:
                idat_closed = True
            if chunk_type[0] & 0x20 == 0:
                raise ValueError("supplemental QA_DENSE PNG has an unsupported critical chunk")
        offset = chunk_end

    if not saw_ihdr or not saw_idat or not saw_iend:
        raise ValueError("supplemental QA_DENSE PNG is incomplete")

    row_bytes = width * channels
    decoded_size = height * (row_bytes + 1)
    if decoded_size > 256 * 1024 * 1024:
        raise ValueError("supplemental QA_DENSE PNG decoded size exceeds policy")
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(bytes(compressed), decoded_size + 1)
    except zlib.error as error:
        raise ValueError("supplemental QA_DENSE PNG pixel stream is corrupt") from error
    if (
        len(decoded) != decoded_size
        or not decoder.eof
        or decoder.unconsumed_tail
        or decoder.unused_data
    ):
        raise ValueError("supplemental QA_DENSE PNG pixel stream is inconsistent")

    previous = bytes(row_bytes)
    for row_index in range(height):
        start = row_index * (row_bytes + 1)
        filter_type = decoded[start]
        if filter_type > 4:
            raise ValueError("supplemental QA_DENSE PNG row filter is invalid")
        encoded_row = decoded[start + 1 : start + 1 + row_bytes]
        reconstructed = bytearray(row_bytes)
        for index, encoded in enumerate(encoded_row):
            left = reconstructed[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            else:
                predictor = _paeth_predictor(left, above, upper_left)
            reconstructed[index] = (encoded + predictor) & 0xFF
        previous = bytes(reconstructed)


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _validated_package(
    package: ProviderNeutralSupplementalPackage,
) -> ProviderNeutralSupplementalPackage:
    if not isinstance(package, ProviderNeutralSupplementalPackage):
        raise TypeError("package must be a ProviderNeutralSupplementalPackage")
    return ProviderNeutralSupplementalPackage.model_validate(
        package.model_dump(mode="python"), strict=True
    )


def _stable_id(kind: str, digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{kind}:{digest}"))


__all__ = [
    "LOCAL_SUPPLEMENTAL_QA_DENSE_POLICY_VERSION",
    "SUPPLEMENTAL_QA_DENSE_INPUT_PROJECTION_VERSION",
    "SUPPLEMENTAL_QA_DENSE_RESULT_PROJECTION_VERSION",
    "DeterministicSupplementalQaDenseConsumer",
    "SupplementalQaDenseConsumption",
    "SupplementalQaDenseInputPlan",
    "SupplementalQaDenseResult",
    "SupplementalQaDenseStatus",
    "supplemental_qa_dense_input_projection",
    "supplemental_qa_dense_result_projection",
]
