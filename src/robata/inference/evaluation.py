"""Provider-neutral paired evaluation for isolated shadow inference.

Architecture V1.1 - Section 11.3 (paired evaluation and disagreements).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import CanonicalizationError, semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.inference.models import (
    InferenceStatus,
    ModelDisagreementSample,
    ModelInference,
    ModelInferenceUsage,
    NonEmptyString,
    NonNegativeInt,
    ShadowSelectionReason,
)

Clock = Callable[[], datetime]


class EvaluationError(ValueError):
    """Raised when a pair is not comparable under the evaluation contract."""


class EvaluationConflictError(EvaluationError):
    """Raised when an immutable evaluation identity is replayed with new content."""


class FieldDelta(StrictModel):
    """A single deterministic field-level disagreement."""

    path: NonEmptyString
    qwen: object | None = None
    gpt: object | None = None
    severity: NonEmptyString


class InferenceEvaluationMetrics(StrictModel):
    """Operational evidence for one side of a paired evaluation."""

    status: InferenceStatus
    output_valid: bool
    retry_count: NonNegativeInt
    latency_ms: NonNegativeInt
    usage: ModelInferenceUsage
    failure_code: NonEmptyString | None = None


class PairedEvaluationMetrics(StrictModel):
    """Primary and shadow operational evidence retained with the comparison."""

    primary: InferenceEvaluationMetrics
    shadow: InferenceEvaluationMetrics


class EvaluationResult(StrictModel):
    """Structured result of one comparable primary/shadow pair."""

    schema_version: Literal["1.0"]
    evaluation_id: OpaqueUuid
    qwen_inference_id: OpaqueUuid
    gpt_inference_id: OpaqueUuid
    shadow_route_id: OpaqueUuid
    field_deltas: tuple[FieldDelta, ...]
    status: NonEmptyString
    comparison_contract_version: SchemaVersion
    comparison_config_digest: Sha256Digest | None = None
    metrics: PairedEvaluationMetrics | None = None
    created_at: Rfc3339Timestamp


@dataclass(frozen=True, slots=True)
class _EvaluationRecord:
    source_digest: str
    result: EvaluationResult
    primary: ModelInference
    shadow: ModelInference


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _rfc3339(clock: Clock) -> str:
    value = clock()
    if not isinstance(value, datetime):
        raise TypeError("evaluation clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvaluationError("evaluation clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _uuid(prefix: str, *parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "\x1f".join((f"robata:{prefix}", *parts))))


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise EvaluationError(f"{field} must be a finite nonnegative number")
    return number


class EvaluationService:
    """Compare stored primary/shadow attempts and retain append-only evidence."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or _utc_now
        self._evaluations: dict[tuple[str, str, str], _EvaluationRecord] = {}
        self._disagreements: dict[tuple[str, str, str], ModelDisagreementSample] = {}

    @property
    def evaluations(self) -> tuple[EvaluationResult, ...]:
        """Return an insertion-ordered snapshot of paired evaluations."""

        return tuple(record.result for record in self._evaluations.values())

    @property
    def disagreements(self) -> tuple[ModelDisagreementSample, ...]:
        """Return an insertion-ordered snapshot of append-only evidence."""

        return tuple(self._disagreements.values())

    def evaluate_pair(
        self,
        *,
        qwen_inference: ModelInference,
        gpt_inference: ModelInference,
        comparison_contract_version: SchemaVersion,
        comparison_config: dict[str, object] | None = None,
    ) -> EvaluationResult:
        """Validate pair comparability, then compute a deterministic evaluation."""

        contract_version = self._contract_version(comparison_contract_version)
        config = deepcopy(comparison_config or {})
        config_digest = self._semantic_digest(config, field="comparison_config")
        self._validate_pair(qwen_inference, gpt_inference)

        key = (qwen_inference.inference_id, gpt_inference.inference_id, contract_version)
        source_digest = self._semantic_digest(
            {
                "comparison_config": config,
                "primary": qwen_inference,
                "shadow": gpt_inference,
            },
            field="evaluation source",
        )
        existing = self._evaluations.get(key)
        if existing is not None:
            if existing.source_digest != source_digest:
                raise EvaluationConflictError(
                    "inference pair and comparison contract already contain different evidence"
                )
            return existing.result

        provider_failure = self._is_provider_failure(qwen_inference) or self._is_provider_failure(
            gpt_inference
        )
        if provider_failure:
            field_deltas: tuple[FieldDelta, ...] = ()
            status = "PROVIDER_FAILURE"
        else:
            primary_output = qwen_inference.normalized_output
            shadow_output = gpt_inference.normalized_output
            if primary_output is None or shadow_output is None:
                raise EvaluationError(
                    "successful comparable attempts must contain normalized output"
                )
            field_deltas = self.compute_disagreement(
                qwen_normalized_output=primary_output,
                gpt_normalized_output=shadow_output,
                comparison_config=config,
            )
            status = "OPEN" if field_deltas else "AGREEMENT"

        result = EvaluationResult(
            schema_version="1.0",
            evaluation_id=_uuid("evaluation", *key),
            qwen_inference_id=qwen_inference.inference_id,
            gpt_inference_id=gpt_inference.inference_id,
            shadow_route_id=self._shadow_route_id(gpt_inference),
            field_deltas=field_deltas,
            status=status,
            comparison_contract_version=contract_version,
            comparison_config_digest=config_digest,
            metrics=PairedEvaluationMetrics(
                primary=self._metrics(qwen_inference),
                shadow=self._metrics(gpt_inference),
            ),
            created_at=_rfc3339(self._clock),
        )
        self._evaluations[key] = _EvaluationRecord(
            source_digest=source_digest,
            result=result,
            primary=qwen_inference,
            shadow=gpt_inference,
        )
        return result

    def compute_disagreement(
        self,
        *,
        qwen_normalized_output: dict[str, object],
        gpt_normalized_output: dict[str, object],
        comparison_config: dict[str, object],
    ) -> tuple[FieldDelta, ...]:
        """Recursively compare normalized JSON-like outputs in stable path order."""

        if not isinstance(qwen_normalized_output, dict) or not isinstance(
            gpt_normalized_output, dict
        ):
            raise EvaluationError("normalized outputs must be dictionaries")
        if not isinstance(comparison_config, dict):
            raise EvaluationError("comparison_config must be a dictionary")

        ignored_paths = self._string_sequence(
            comparison_config.get(
                "ignored_paths",
                comparison_config.get("ignore_paths", ()),
            ),
            field="ignored_paths",
        )
        global_tolerance = _nonnegative_number(
            comparison_config.get("numeric_tolerance", 0.0),
            field="numeric_tolerance",
        )
        path_tolerances = self._path_numbers(
            comparison_config.get("numeric_tolerances", {}),
            field="numeric_tolerances",
        )
        severity_by_path = self._path_strings(
            comparison_config.get("severity_by_path", {}),
            field="severity_by_path",
        )
        default_severity = comparison_config.get("default_severity", "MATERIAL")
        if not isinstance(default_severity, str) or not default_severity:
            raise EvaluationError("default_severity must be a nonempty string")

        deltas: list[FieldDelta] = []

        def is_ignored(path: str) -> bool:
            return any(
                path == ignored or path.startswith(f"{ignored}.") or path.startswith(f"{ignored}[")
                for ignored in ignored_paths
            )

        def severity(path: str) -> str:
            if path in severity_by_path:
                return severity_by_path[path]
            matches = [candidate for candidate in severity_by_path if path.startswith(candidate)]
            if matches:
                return severity_by_path[max(matches, key=len)]
            return default_severity

        def tolerance(path: str) -> float:
            if path in path_tolerances:
                return path_tolerances[path]
            matches = [candidate for candidate in path_tolerances if path.startswith(candidate)]
            return path_tolerances[max(matches, key=len)] if matches else global_tolerance

        def add(path: str, primary: object | None, shadow: object | None) -> None:
            if not is_ignored(path):
                deltas.append(
                    FieldDelta(
                        path=path,
                        qwen=deepcopy(primary),
                        gpt=deepcopy(shadow),
                        severity=severity(path),
                    )
                )

        def compare(primary: object, shadow: object, path: str) -> None:
            if is_ignored(path):
                return
            if isinstance(primary, Mapping) and isinstance(shadow, Mapping):
                if any(not isinstance(key, str) for key in (*primary.keys(), *shadow.keys())):
                    raise EvaluationError("normalized output dictionary keys must be strings")
                for key in sorted(set(primary) | set(shadow)):
                    child_path = f"{path}.{key}" if path else key
                    if key not in primary:
                        add(child_path, None, shadow[key])
                    elif key not in shadow:
                        add(child_path, primary[key], None)
                    else:
                        compare(primary[key], shadow[key], child_path)
                return
            if (
                isinstance(primary, Sequence)
                and not isinstance(primary, (str, bytes))
                and isinstance(shadow, Sequence)
                and not isinstance(shadow, (str, bytes))
            ):
                for index in range(max(len(primary), len(shadow))):
                    child_path = f"{path or '$'}[{index}]"
                    if index >= len(primary):
                        add(child_path, None, shadow[index])
                    elif index >= len(shadow):
                        add(child_path, primary[index], None)
                    else:
                        compare(primary[index], shadow[index], child_path)
                return
            if self._numbers_equal(primary, shadow, tolerance(path or "$")):
                return
            if primary != shadow:
                add(path or "$", primary, shadow)

        compare(qwen_normalized_output, gpt_normalized_output, "")
        return tuple(deltas)

    def persist_disagreement(
        self,
        *,
        evaluation_result: EvaluationResult,
        shadow_reason: ShadowSelectionReason,
        mcap_id: OpaqueUuid,
        start_ns: Nanoseconds,
        end_ns: Nanoseconds,
        package_set_id: OpaqueUuid,
        camera_mapping_run_id: OpaqueUuid,
        alignment_id: OpaqueUuid,
    ) -> ModelDisagreementSample:
        """Idempotently append one immutable disagreement/provider-failure sample."""

        if evaluation_result.status not in {"OPEN", "PROVIDER_FAILURE"}:
            raise EvaluationError(
                "only OPEN disagreements and PROVIDER_FAILURE outcomes can be persisted"
            )
        if evaluation_result.status == "OPEN" and not evaluation_result.field_deltas:
            raise EvaluationError("OPEN disagreement must contain at least one field delta")
        if evaluation_result.status == "PROVIDER_FAILURE" and evaluation_result.field_deltas:
            raise EvaluationError("PROVIDER_FAILURE outcome cannot contain field deltas")
        if isinstance(start_ns, bool) or isinstance(end_ns, bool) or start_ns >= end_ns:
            raise EvaluationError("disagreement interval must be nonempty")
        key = (
            evaluation_result.qwen_inference_id,
            evaluation_result.gpt_inference_id,
            evaluation_result.comparison_contract_version,
        )
        self._validate_persistence_context(
            key=key,
            evaluation_result=evaluation_result,
            mcap_id=mcap_id,
            start_ns=start_ns,
            end_ns=end_ns,
            package_set_id=package_set_id,
            camera_mapping_run_id=camera_mapping_run_id,
            alignment_id=alignment_id,
        )
        config_digest = evaluation_result.comparison_config_digest or self._semantic_digest(
            {
                "comparison_contract_version": evaluation_result.comparison_contract_version,
                "field_deltas": evaluation_result.field_deltas,
            },
            field="comparison fallback configuration",
        )
        candidate = ModelDisagreementSample(
            schema_version="1.0",
            disagreement_id=_uuid("disagreement", *key),
            mcap_id=mcap_id,
            start_ns=start_ns,
            end_ns=end_ns,
            package_set_id=package_set_id,
            camera_mapping_run_id=camera_mapping_run_id,
            alignment_id=alignment_id,
            qwen_inference_id=evaluation_result.qwen_inference_id,
            gpt_inference_id=evaluation_result.gpt_inference_id,
            comparison_contract_version=evaluation_result.comparison_contract_version,
            comparison_config_digest=config_digest,
            shadow_route_id=evaluation_result.shadow_route_id,
            shadow_reason=shadow_reason,
            field_deltas=tuple(
                delta.model_dump(mode="json") for delta in evaluation_result.field_deltas
            ),
            status=evaluation_result.status,
            adjudication=None,
            created_at=evaluation_result.created_at,
        )
        existing = self._disagreements.get(key)
        if existing is not None:
            if existing != candidate:
                raise EvaluationConflictError(
                    "inference pair and comparison contract already have different evidence"
                )
            return existing
        self._disagreements[key] = candidate
        return candidate

    def _validate_pair(self, primary: ModelInference, shadow: ModelInference) -> None:
        if primary.inference_id == shadow.inference_id:
            raise EvaluationError("primary and shadow inference IDs must differ")
        if primary.shadow:
            raise EvaluationError("qwen_inference must be a primary inference")
        if not shadow.shadow:
            raise EvaluationError("gpt_inference must be marked as shadow")
        if primary.provider == shadow.provider:
            raise EvaluationError("primary and shadow providers must differ")
        if shadow.primary_inference_id not in {None, primary.inference_id}:
            raise EvaluationError(
                "shadow primary_inference_id does not reference the paired primary"
            )
        if not shadow.shadow_route_id:
            raise EvaluationError("shadow inference must reference a shadow route")

        comparable_fields = (
            "mcap_id",
            "package_set_id",
            "package_id",
            "package_ids",
            "camera_mapping_run_id",
            "alignment_id",
            "start_ns",
            "end_ns",
            "stage",
            "prompt_version",
            "prompt_artifact_id",
            "prompt_sha256",
            "rendered_input_digest",
            "output_schema_id",
            "output_schema_version",
            "output_schema_artifact_id",
            "output_schema_sha256",
            "input_manifest_set_sha256",
            "input_config",
            "sampling_config",
        )
        mismatches = [
            field
            for field in comparable_fields
            if getattr(primary, field) != getattr(shadow, field)
        ]
        if mismatches:
            raise EvaluationError(
                "inference pair is not comparable; mismatched fields: " + ", ".join(mismatches)
            )
        if primary.package_set_id is None and primary.package_id is None:
            raise EvaluationError(
                "comparable inference pair must reference immutable package content"
            )
        if primary.start_ns >= primary.end_ns:
            raise EvaluationError("inference interval must be nonempty")

    def _validate_persistence_context(
        self,
        *,
        key: tuple[str, str, str],
        evaluation_result: EvaluationResult,
        mcap_id: str,
        start_ns: int,
        end_ns: int,
        package_set_id: str,
        camera_mapping_run_id: str,
        alignment_id: str,
    ) -> None:
        record = self._evaluations.get(key)
        if record is None:
            return
        if record.result != evaluation_result:
            raise EvaluationConflictError(
                "evaluation result differs from stored append-only evidence"
            )
        expected = (
            record.primary.mcap_id,
            record.primary.start_ns,
            record.primary.end_ns,
            record.primary.package_set_id,
            record.primary.camera_mapping_run_id,
            record.primary.alignment_id,
        )
        supplied = (
            mcap_id,
            start_ns,
            end_ns,
            package_set_id,
            camera_mapping_run_id,
            alignment_id,
        )
        if expected != supplied:
            raise EvaluationError("disagreement lineage differs from the evaluated inference pair")

    @staticmethod
    def _metrics(inference: ModelInference) -> InferenceEvaluationMetrics:
        return InferenceEvaluationMetrics(
            status=inference.status,
            output_valid=inference.output_valid,
            retry_count=inference.retry_count,
            latency_ms=inference.latency_ms,
            usage=inference.usage,
            failure_code=inference.failure.code if inference.failure is not None else None,
        )

    @staticmethod
    def _is_provider_failure(inference: ModelInference) -> bool:
        return inference.status is not InferenceStatus.SUCCEEDED or not inference.output_valid

    @staticmethod
    def _numbers_equal(primary: object, shadow: object, tolerance: float) -> bool:
        if isinstance(primary, bool) or isinstance(shadow, bool):
            return False
        if isinstance(primary, (int, float)) and isinstance(shadow, (int, float)):
            primary_number = float(primary)
            shadow_number = float(shadow)
            return (
                math.isfinite(primary_number)
                and math.isfinite(shadow_number)
                and abs(primary_number - shadow_number) <= tolerance
            )
        return False

    @staticmethod
    def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise EvaluationError(f"{field} must be a list or tuple of strings")
        if any(not isinstance(item, str) or not item for item in value):
            raise EvaluationError(f"{field} must contain only nonempty strings")
        return tuple(value)

    @staticmethod
    def _path_numbers(value: object, *, field: str) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise EvaluationError(f"{field} must be a mapping")
        result: dict[str, float] = {}
        for path, tolerance in value.items():
            if not isinstance(path, str) or not path:
                raise EvaluationError(f"{field} keys must be nonempty strings")
            result[path] = _nonnegative_number(tolerance, field=f"{field}.{path}")
        return result

    @staticmethod
    def _path_strings(value: object, *, field: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise EvaluationError(f"{field} must be a mapping")
        result: dict[str, str] = {}
        for path, severity in value.items():
            if not isinstance(path, str) or not path:
                raise EvaluationError(f"{field} keys must be nonempty strings")
            if not isinstance(severity, str) or not severity:
                raise EvaluationError(f"{field} values must be nonempty strings")
            result[path] = severity
        return result

    @staticmethod
    def _semantic_digest(value: object, *, field: str) -> str:
        try:
            return semantic_sha256(value)
        except (CanonicalizationError, TypeError, ValueError) as error:
            raise EvaluationError(f"{field} is not canonical JSON data") from error

    @staticmethod
    def _shadow_route_id(inference: ModelInference) -> str:
        if not inference.shadow_route_id:
            raise EvaluationError("shadow inference must reference a shadow route")
        return inference.shadow_route_id

    @staticmethod
    def _contract_version(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise EvaluationError("comparison_contract_version must be a nonempty string")
        return value


__all__ = [
    "EvaluationConflictError",
    "EvaluationError",
    "EvaluationResult",
    "EvaluationService",
    "FieldDelta",
    "InferenceEvaluationMetrics",
    "PairedEvaluationMetrics",
]
