"""Internal, replayable calibration lineage for accepted inference evidence.

The registered ``ModelInference`` payload is terminal and append-only.  This
module deliberately keeps calibration facts in a separate internal branch so a
later fit cannot rewrite a provider score or silently alter a published Product
QA projection.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from math import exp, isfinite
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, Self

from pydantic import Field, StringConstraints, field_serializer, field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.inference.enrichment import OrchestratorEnrichedOutput, SelectedAttemptOutput
from robata.inference.models import (
    InferenceAttemptSelection,
    InferenceStatus,
    ModelInference,
    VisionTask,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
ACCEPTED_INFERENCE_CALIBRATION_EXTRACTOR_VERSION: Final = (
    "accepted-inference-calibration-extractor-v1"
)
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class _FrozenJsonMapping(Mapping[str, object]):
    """A recursively frozen JSON object that remains safe to place in a ledger cache."""

    __values: Mapping[str, object]
    __slots__ = ("__values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        object.__setattr__(self, "_FrozenJsonMapping__values", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("frozen JSON mappings cannot be mutated")

    def __getitem__(self, key: str) -> object:
        return self.__values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self


def _freeze_json_object(value: Mapping[str, object], *, field_name: str) -> _FrozenJsonMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _freeze_json_value(value, path=field_name)
    if not isinstance(frozen, _FrozenJsonMapping):
        raise AssertionError("JSON object freezer returned a non-mapping value")
    return frozen


def _freeze_json_value(value: object, *, path: str) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} must not contain a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen_values: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string object key")
            frozen_values[key] = _freeze_json_value(nested, path=f"{path}.{key}")
        return _FrozenJsonMapping(frozen_values)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    raise ValueError(f"{path} must contain only JSON-compatible values")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(nested) for nested in value]
    return value


class CalibrationFittingMethod(StrEnum):
    """Deterministic fit families supported by the local calibration bridge."""

    IDENTITY = "IDENTITY"
    PLATT_LOGISTIC = "PLATT_LOGISTIC"
    ISOTONIC_LINEAR = "ISOTONIC_LINEAR"


class CalibrationAssociationOutcome(StrEnum):
    """Whether a selected inference could use its frozen calibration artifact."""

    APPLIED = "APPLIED"
    RAW_FALLBACK_MISSING_ARTIFACT = "RAW_FALLBACK_MISSING_ARTIFACT"
    RAW_FALLBACK_INAPPLICABLE = "RAW_FALLBACK_INAPPLICABLE"
    RAW_FALLBACK_UNAVAILABLE_SCORE = "RAW_FALLBACK_UNAVAILABLE_SCORE"


class CalibrationScoreSource(StrEnum):
    """The one scalar source a P9 binding may calibrate per accepted selection."""

    TERMINAL_REPORTED_CONFIDENCE = "TERMINAL_REPORTED_CONFIDENCE"
    ENRICHED_CLAIM_REPORTED_CONFIDENCE = "ENRICHED_CLAIM_REPORTED_CONFIDENCE"


def accepted_calibration_score_input(
    *,
    score_source: CalibrationScoreSource,
    source_claim_ordinal: int | None,
    inference: ModelInference,
    selected_output: SelectedAttemptOutput,
    enriched_output: OrchestratorEnrichedOutput,
) -> tuple[float | None, dict[str, object]]:
    """Extract one score and its complete immutable accepted-evidence projection."""

    if not isinstance(score_source, CalibrationScoreSource):
        raise TypeError("score_source must be a CalibrationScoreSource")
    if not isinstance(inference, ModelInference):
        raise TypeError("inference must be a ModelInference")
    if not isinstance(selected_output, SelectedAttemptOutput):
        raise TypeError("selected_output must be a SelectedAttemptOutput")
    if not isinstance(enriched_output, OrchestratorEnrichedOutput):
        raise TypeError("enriched_output must be an OrchestratorEnrichedOutput")
    if (
        selected_output.inference_id != inference.inference_id
        or enriched_output.selected_attempt != selected_output
    ):
        raise ValueError("calibration score source does not bind the accepted output")

    inputs: dict[str, object] = {
        "extractor_version": ACCEPTED_INFERENCE_CALIBRATION_EXTRACTOR_VERSION,
        "score_source": score_source.value,
        "selected_attempt_output_sha256": selected_output.output_sha256,
        "enriched_output_artifact_id": enriched_output.artifact_id,
        "enriched_output_semantic_sha256": enriched_output.semantic_sha256,
        "enrichment_logical_key": enriched_output.enrichment_logical_key,
    }
    if score_source is CalibrationScoreSource.TERMINAL_REPORTED_CONFIDENCE:
        if source_claim_ordinal is not None:
            raise ValueError("terminal calibration score source cannot name a claim ordinal")
        inputs["score_path"] = "model_inference.reported_confidence.value"
        value = _reported_terminal_score(inference.reported_confidence)
        if value is None:
            inputs["score_available"] = False
            return None, inputs
        inputs["score_available"] = True
        return value, inputs

    if isinstance(source_claim_ordinal, bool) or not isinstance(source_claim_ordinal, int):
        raise ValueError("enriched claim calibration score source requires a nonnegative ordinal")
    if source_claim_ordinal < 0:
        raise ValueError("enriched claim calibration score ordinal must be nonnegative")
    inputs["source_claim_ordinal"] = source_claim_ordinal
    inputs["score_path"] = (
        "orchestrator_enriched_output.claims["
        f"{source_claim_ordinal}].model_reported_confidence.value"
    )
    claim = next(
        (item for item in enriched_output.claims if item.claim_ordinal == source_claim_ordinal),
        None,
    )
    if claim is None or claim.model_reported_confidence is None:
        inputs["score_available"] = False
        return None, inputs
    confidence = claim.model_reported_confidence
    inputs.update(
        {
            "source_claim_id": claim.claim_id,
            "source_confidence_id": confidence.confidence_id,
            "source_confidence_kind": confidence.kind,
            "source_confidence_semantics": confidence.semantics,
            "score_available": True,
        }
    )
    return confidence.value, inputs


class CalibrationApplicability(StrictModel):
    """Every immutable term which must match before applying a fitted mapping."""

    score_family: NonEmptyString
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    runtime_revision: NonEmptyString
    preprocess_revision: NonEmptyString
    prompt_version: SchemaVersion
    prompt_sha256: Sha256Digest
    stage: VisionTask

    @classmethod
    def from_inference(
        cls,
        inference: ModelInference,
        *,
        score_family: str,
        runtime_revision: str,
        preprocess_revision: str,
    ) -> Self:
        """Bind a fit to the exact model and input-preparation tuple."""

        if not isinstance(inference, ModelInference):
            raise TypeError("inference must be a ModelInference")
        return cls(
            score_family=score_family,
            provider=inference.provider,
            model_name=inference.model_name,
            model_version=inference.model_version,
            adapter_version=inference.adapter_version,
            runtime_revision=runtime_revision,
            preprocess_revision=preprocess_revision,
            prompt_version=inference.prompt_version,
            prompt_sha256=inference.prompt_sha256,
            stage=inference.stage,
        )

    def mismatch_reasons(
        self,
        inference: ModelInference,
        *,
        score_family: str,
        runtime_revision: str,
        preprocess_revision: str,
    ) -> tuple[str, ...]:
        """Return stable reasons rather than treating partial matches as valid."""

        if not isinstance(inference, ModelInference):
            raise TypeError("inference must be a ModelInference")
        values = (
            ("score_family", self.score_family, score_family),
            ("provider", self.provider, inference.provider),
            ("model_name", self.model_name, inference.model_name),
            ("model_version", self.model_version, inference.model_version),
            ("adapter_version", self.adapter_version, inference.adapter_version),
            ("runtime_revision", self.runtime_revision, runtime_revision),
            ("preprocess_revision", self.preprocess_revision, preprocess_revision),
            ("prompt_version", self.prompt_version, inference.prompt_version),
            ("prompt_sha256", self.prompt_sha256, inference.prompt_sha256),
            ("stage", self.stage, inference.stage),
        )
        return tuple(name for name, expected, actual in values if expected != actual)


class CalibrationTrainingPopulation(StrictModel):
    """Content-addressed labelled population used to fit one calibration artifact."""

    population_artifact_id: NonEmptyString
    population_sha256: Sha256Digest
    label_set_sha256: Sha256Digest
    member_count: PositiveInt
    labelled_member_count: PositiveInt

    @model_validator(mode="after")
    def validate_labelled_count(self) -> Self:
        if self.labelled_member_count > self.member_count:
            raise ValueError("labelled member count cannot exceed population member count")
        return self


class CalibrationGroupedSplitLineage(StrictModel):
    """Leakage-safe development/calibration/frozen-evaluation split lineage."""

    split_artifact_id: NonEmptyString
    split_sha256: Sha256Digest
    grouping_key: NonEmptyString
    leakage_policy_version: SchemaVersion
    development_group_sha256: Sha256Digest
    calibration_group_sha256: Sha256Digest
    frozen_evaluation_group_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_disjoint_group_manifests(self) -> Self:
        group_digests = (
            self.development_group_sha256,
            self.calibration_group_sha256,
            self.frozen_evaluation_group_sha256,
        )
        if len(set(group_digests)) != len(group_digests):
            raise ValueError("development, calibration, and evaluation groups must be distinct")
        return self


class CalibrationPolicyDecision(StrictModel):
    """A separately-versioned policy result; it is not a calibrated probability."""

    policy_version: SchemaVersion
    decision: NonEmptyString
    input_kind: Literal["RAW_SCORE", "CALIBRATED_PROBABILITY"]
    threshold: UnitInterval | None = None


class CalibrationArtifact(StrictModel):
    """Frozen content-addressed fitting artifact with complete applicability lineage."""

    schema_version: Literal["1.0"]
    artifact_id: NonEmptyString
    artifact_sha256: Sha256Digest
    applicability: CalibrationApplicability
    fitting_method: CalibrationFittingMethod
    fitting_parameters: Mapping[str, object]
    training_population: CalibrationTrainingPopulation
    grouped_split_lineage: CalibrationGroupedSplitLineage
    fitted_at: Rfc3339Timestamp
    valid_from: Rfc3339Timestamp
    valid_until: Rfc3339Timestamp | None = None
    created_at: Rfc3339Timestamp

    @classmethod
    def create(
        cls,
        *,
        applicability: CalibrationApplicability,
        fitting_method: CalibrationFittingMethod,
        fitting_parameters: Mapping[str, object],
        training_population: CalibrationTrainingPopulation,
        grouped_split_lineage: CalibrationGroupedSplitLineage,
        fitted_at: str,
        valid_from: str,
        valid_until: str | None,
        created_at: str,
    ) -> Self:
        """Build an artifact whose ID is exactly its semantic content digest."""

        frozen_parameters = _freeze_json_object(fitting_parameters, field_name="fitting_parameters")
        projection = _artifact_projection(
            schema_version="1.0",
            applicability=applicability,
            fitting_method=fitting_method,
            fitting_parameters=frozen_parameters,
            training_population=training_population,
            grouped_split_lineage=grouped_split_lineage,
            fitted_at=fitted_at,
            valid_from=valid_from,
            valid_until=valid_until,
            created_at=created_at,
        )
        digest = semantic_sha256(projection)
        return cls(
            schema_version="1.0",
            artifact_id=f"calibration-artifact:{digest}",
            artifact_sha256=digest,
            applicability=applicability,
            fitting_method=fitting_method,
            fitting_parameters=frozen_parameters,
            training_population=training_population,
            grouped_split_lineage=grouped_split_lineage,
            fitted_at=fitted_at,
            valid_from=valid_from,
            valid_until=valid_until,
            created_at=created_at,
        )

    @model_validator(mode="after")
    def validate_content_address_and_fit(self) -> Self:
        projection = _artifact_projection(
            schema_version=self.schema_version,
            applicability=self.applicability,
            fitting_method=self.fitting_method,
            fitting_parameters=self.fitting_parameters,
            training_population=self.training_population,
            grouped_split_lineage=self.grouped_split_lineage,
            fitted_at=self.fitted_at,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            created_at=self.created_at,
        )
        digest = semantic_sha256(projection)
        if self.artifact_sha256 != digest or self.artifact_id != f"calibration-artifact:{digest}":
            raise ValueError("calibration artifact identity does not match its content digest")
        if _parse_timestamp(self.valid_from) < _parse_timestamp(self.fitted_at):
            raise ValueError("calibration validity cannot begin before the fit timestamp")
        if self.valid_until is not None and _parse_timestamp(self.valid_until) < _parse_timestamp(
            self.valid_from
        ):
            raise ValueError("calibration validity cannot end before it begins")
        if _parse_timestamp(self.created_at) < _parse_timestamp(self.fitted_at):
            raise ValueError("calibration artifact creation cannot precede fitting")
        _validate_fitting_parameters(self.fitting_method, self.fitting_parameters)
        return self

    @field_validator("fitting_parameters")
    @classmethod
    def freeze_fitting_parameters(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return _freeze_json_object(value, field_name="fitting_parameters")

    @field_serializer("fitting_parameters")
    def serialize_fitting_parameters(self, value: Mapping[str, object]) -> object:
        return _thaw_json_value(value)

    def mismatch_reasons(
        self,
        inference: ModelInference,
        *,
        score_family: str,
        runtime_revision: str,
        preprocess_revision: str,
        evaluated_at: str,
    ) -> tuple[str, ...]:
        """Fail closed unless every applicability and validity term matches."""

        reasons = list(
            self.applicability.mismatch_reasons(
                inference,
                score_family=score_family,
                runtime_revision=runtime_revision,
                preprocess_revision=preprocess_revision,
            )
        )
        evaluated = _parse_timestamp(evaluated_at)
        if evaluated < _parse_timestamp(self.valid_from) or (
            self.valid_until is not None and evaluated > _parse_timestamp(self.valid_until)
        ):
            reasons.append("validity_window")
        return tuple(reasons)

    def calibrate(self, raw_score: float) -> float:
        """Apply the frozen deterministic fitting mapping to one raw score."""

        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError("raw_score must be a finite number")
        score = float(raw_score)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("raw_score must be within [0, 1]")
        parameters = self.fitting_parameters
        if self.fitting_method is CalibrationFittingMethod.IDENTITY:
            return score
        if self.fitting_method is CalibrationFittingMethod.PLATT_LOGISTIC:
            slope = _finite_parameter(parameters["slope"], "slope")
            intercept = _finite_parameter(parameters["intercept"], "intercept")
            logit = slope * score + intercept
            if logit >= 0.0:
                return 1.0 / (1.0 + exp(-logit))
            exponent = exp(logit)
            return exponent / (1.0 + exponent)
        if self.fitting_method is CalibrationFittingMethod.ISOTONIC_LINEAR:
            knots = _isotonic_knots(parameters)
            if score <= knots[0][0]:
                return knots[0][1]
            if score >= knots[-1][0]:
                return knots[-1][1]
            for left, right in pairwise(knots):
                if left[0] <= score <= right[0]:
                    span = right[0] - left[0]
                    return left[1] + ((score - left[0]) / span) * (right[1] - left[1])
        raise AssertionError("validated calibration fitting method did not return")


class CalibrationAssociation(StrictModel):
    """Append-only calibration result bound to one accepted inference selection."""

    schema_version: Literal["1.0"]
    association_id: NonEmptyString
    selection_id: OpaqueUuid
    inference_id: OpaqueUuid
    score_family: NonEmptyString
    runtime_revision: NonEmptyString
    preprocess_revision: NonEmptyString
    evaluated_at: Rfc3339Timestamp
    raw_score: UnitInterval | None
    calibrated_probability: UnitInterval | None
    deterministic_inputs: Mapping[str, object]
    deterministic_inputs_sha256: Sha256Digest
    policy_decision: CalibrationPolicyDecision | None = None
    outcome: CalibrationAssociationOutcome
    mismatch_reasons: tuple[NonEmptyString, ...]
    calibration_artifact_id: NonEmptyString | None = None
    calibration_artifact_sha256: Sha256Digest | None = None
    created_at: Rfc3339Timestamp

    @classmethod
    def create(
        cls,
        *,
        selection_id: str,
        inference: ModelInference,
        score_family: str,
        runtime_revision: str,
        preprocess_revision: str,
        evaluated_at: str,
        raw_score: float | None,
        deterministic_inputs: Mapping[str, object],
        policy_decision: CalibrationPolicyDecision | None,
        calibration_artifact: CalibrationArtifact | None,
        created_at: str,
    ) -> Self:
        """Create either a calibrated result or an explicit raw-score fallback."""

        if not isinstance(inference, ModelInference):
            raise TypeError("inference must be a ModelInference")
        raw = None if raw_score is None else _unit_interval(raw_score, "raw_score")
        frozen_inputs = _freeze_json_object(deterministic_inputs, field_name="deterministic_inputs")
        artifact_id = None if calibration_artifact is None else calibration_artifact.artifact_id
        artifact_sha256 = (
            None if calibration_artifact is None else calibration_artifact.artifact_sha256
        )
        reasons: tuple[str, ...]
        if raw is None:
            outcome = CalibrationAssociationOutcome.RAW_FALLBACK_UNAVAILABLE_SCORE
            calibrated = None
            reasons = ("raw_score_unavailable",)
            artifact_id = None
            artifact_sha256 = None
        elif calibration_artifact is None:
            outcome = CalibrationAssociationOutcome.RAW_FALLBACK_MISSING_ARTIFACT
            calibrated = None
            reasons = ("artifact_missing",)
        else:
            reasons = calibration_artifact.mismatch_reasons(
                inference,
                score_family=score_family,
                runtime_revision=runtime_revision,
                preprocess_revision=preprocess_revision,
                evaluated_at=evaluated_at,
            )
            if reasons:
                outcome = CalibrationAssociationOutcome.RAW_FALLBACK_INAPPLICABLE
                calibrated = None
            else:
                outcome = CalibrationAssociationOutcome.APPLIED
                calibrated = calibration_artifact.calibrate(raw)
        return cls(
            schema_version="1.0",
            association_id=calibration_association_id(
                selection_id=selection_id,
                score_family=score_family,
            ),
            selection_id=selection_id,
            inference_id=inference.inference_id,
            score_family=score_family,
            runtime_revision=runtime_revision,
            preprocess_revision=preprocess_revision,
            evaluated_at=evaluated_at,
            raw_score=raw,
            calibrated_probability=calibrated,
            deterministic_inputs=frozen_inputs,
            deterministic_inputs_sha256=calibration_inputs_sha256(frozen_inputs),
            policy_decision=policy_decision,
            outcome=outcome,
            mismatch_reasons=reasons,
            calibration_artifact_id=artifact_id,
            calibration_artifact_sha256=artifact_sha256,
            created_at=created_at,
        )

    @model_validator(mode="after")
    def validate_result_shape(self) -> Self:
        expected_id = calibration_association_id(
            selection_id=self.selection_id,
            score_family=self.score_family,
        )
        if self.association_id != expected_id:
            raise ValueError("calibration association identity is inconsistent")
        if self.deterministic_inputs_sha256 != calibration_inputs_sha256(self.deterministic_inputs):
            raise ValueError("calibration association deterministic inputs digest is inconsistent")
        artifact_fields_match = (self.calibration_artifact_id is None) == (
            self.calibration_artifact_sha256 is None
        )
        if not artifact_fields_match:
            raise ValueError("calibration artifact identity and digest must be present together")
        if self.outcome is CalibrationAssociationOutcome.APPLIED:
            if (
                self.raw_score is None
                or self.calibrated_probability is None
                or self.calibration_artifact_id is None
                or self.mismatch_reasons
            ):
                raise ValueError("applied calibration requires an artifact, raw score, and result")
            if self.policy_decision is not None and (
                self.policy_decision.input_kind != "CALIBRATED_PROBABILITY"
            ):
                raise ValueError("applied calibration policy must use the calibrated probability")
        elif self.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_MISSING_ARTIFACT:
            if (
                self.raw_score is None
                or self.calibrated_probability is not None
                or self.calibration_artifact_id is not None
                or self.mismatch_reasons != ("artifact_missing",)
            ):
                raise ValueError("missing calibration artifact must retain the raw-score fallback")
            if self.policy_decision is not None and self.policy_decision.input_kind != "RAW_SCORE":
                raise ValueError("raw-score fallback policy must use the raw score")
        elif self.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_INAPPLICABLE:
            if (
                self.raw_score is None
                or self.calibrated_probability is not None
                or self.calibration_artifact_id is None
                or not self.mismatch_reasons
            ):
                raise ValueError(
                    "inapplicable calibration must retain an explicit raw-score fallback"
                )
            if self.policy_decision is not None and self.policy_decision.input_kind != "RAW_SCORE":
                raise ValueError("raw-score fallback policy must use the raw score")
        elif self.outcome is CalibrationAssociationOutcome.RAW_FALLBACK_UNAVAILABLE_SCORE:
            if (
                self.raw_score is not None
                or self.calibrated_probability is not None
                or self.calibration_artifact_id is not None
                or self.mismatch_reasons != ("raw_score_unavailable",)
                or self.policy_decision is not None
            ):
                raise ValueError("unavailable raw score cannot claim calibration lineage")
        else:
            raise AssertionError("unhandled calibration association outcome")
        return self

    @field_validator("deterministic_inputs")
    @classmethod
    def freeze_deterministic_inputs(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return _freeze_json_object(value, field_name="deterministic_inputs")

    @field_serializer("deterministic_inputs")
    def serialize_deterministic_inputs(self, value: Mapping[str, object]) -> object:
        return _thaw_json_value(value)


def calibration_artifact_digest(artifact: CalibrationArtifact) -> Sha256Digest:
    """Return the content digest used by metric/report consumers as a generic ID."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be a CalibrationArtifact")
    return artifact.artifact_sha256


class CalibrationBinding(StrictModel):
    """One explicitly scalar, inference-side calibration configuration."""

    schema_version: Literal["1.0"] = "1.0"
    task: VisionTask
    score_family: NonEmptyString
    runtime_revision: NonEmptyString
    preprocess_revision: NonEmptyString
    score_source: CalibrationScoreSource
    source_claim_ordinal: NonNegativeInt | None = None
    calibration_artifact: CalibrationArtifact | None = None

    @model_validator(mode="after")
    def validate_score_source(self) -> Self:
        requires_claim_ordinal = (
            self.score_source is CalibrationScoreSource.ENRICHED_CLAIM_REPORTED_CONFIDENCE
        )
        if requires_claim_ordinal != (self.source_claim_ordinal is not None):
            raise ValueError(
                "source_claim_ordinal must match the configured calibration score source"
            )
        return self


class CalibrationAssociationStore(Protocol):
    """Append-only internal storage for calibration facts detached from wire contracts."""

    def append_calibration_artifact(self, artifact: CalibrationArtifact) -> CalibrationArtifact:
        """Persist or replay one content-addressed calibration artifact."""
        ...

    def append_calibration_association(
        self,
        association: CalibrationAssociation,
    ) -> CalibrationAssociation:
        """Persist or replay one selected-inference calibration association."""
        ...


class CalibrationBridgeError(RuntimeError):
    """The configured detached calibration lineage could not be recorded safely."""


class AcceptedInferenceCalibrationBridge:
    """Append P9 calibration only after canonical evidence accepts a selected output.

    A binding targets one scalar source.  It deliberately does not update the
    terminal inference or any Product QA object, so a later calibration fit cannot
    change existing wire values, cascade states, or semantic projections.
    """

    def __init__(
        self,
        *,
        store: CalibrationAssociationStore,
        bindings: Sequence[CalibrationBinding],
    ) -> None:
        if not callable(getattr(store, "append_calibration_artifact", None)) or not callable(
            getattr(store, "append_calibration_association", None)
        ):
            raise TypeError("store must implement the calibration association append port")
        configured: dict[VisionTask, list[CalibrationBinding]] = {}
        seen: set[tuple[VisionTask, str]] = set()
        for binding in bindings:
            if not isinstance(binding, CalibrationBinding):
                raise TypeError("bindings must contain CalibrationBinding values")
            try:
                checked = CalibrationBinding.model_validate(
                    binding.model_dump(mode="python"),
                    strict=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("calibration binding failed strict validation") from exc
            key = (checked.task, checked.score_family)
            if key in seen:
                raise ValueError(
                    "one task cannot configure the same calibration score family twice"
                )
            seen.add(key)
            configured.setdefault(checked.task, []).append(checked)
        self._store = store
        self._bindings_by_task = {
            task: tuple(sorted(task_bindings, key=lambda binding: binding.score_family))
            for task, task_bindings in configured.items()
        }

    @property
    def store(self) -> CalibrationAssociationStore:
        """Return the append-only store that must share the accepted evidence ledger."""

        return self._store

    def record_accepted(
        self,
        *,
        task: VisionTask,
        inference: ModelInference,
        selection: InferenceAttemptSelection,
        selected_output: SelectedAttemptOutput,
        enriched_output: OrchestratorEnrichedOutput,
    ) -> tuple[CalibrationAssociation, ...]:
        """Append replay-stable associations for one already accepted canonical output."""

        if not isinstance(task, VisionTask):
            raise TypeError("task must be a VisionTask")
        checked_inference = _strict_calibration_instance(
            inference,
            ModelInference,
            "inference",
        )
        checked_selection = _strict_calibration_instance(
            selection,
            InferenceAttemptSelection,
            "selection",
        )
        checked_selected_output = _strict_calibration_instance(
            selected_output,
            SelectedAttemptOutput,
            "selected_output",
        )
        checked_enriched_output = _strict_calibration_instance(
            enriched_output,
            OrchestratorEnrichedOutput,
            "enriched_output",
        )
        if (
            checked_inference.stage is not task
            or checked_inference.shadow
            or checked_inference.status is not InferenceStatus.SUCCEEDED
            or not checked_inference.output_valid
            or checked_inference.failure is not None
            or checked_selection.inference_id != checked_inference.inference_id
            or checked_selection.logical_invocation_id != checked_inference.logical_invocation_id
            or checked_selected_output.inference_id != checked_inference.inference_id
            or checked_selected_output.selection_id != checked_selection.selection_id
            or checked_selected_output.logical_invocation_id
            != checked_inference.logical_invocation_id
            or checked_enriched_output.task is not task
            or checked_enriched_output.selected_attempt != checked_selected_output
        ):
            raise CalibrationBridgeError(
                "calibration bridge requires one accepted selected inference"
            )

        associations: list[CalibrationAssociation] = []
        for binding in self._bindings_by_task.get(task, ()):
            raw_score, deterministic_inputs = self._score_input(
                binding=binding,
                inference=checked_inference,
                selected_output=checked_selected_output,
                enriched_output=checked_enriched_output,
            )
            artifact = binding.calibration_artifact if raw_score is not None else None
            try:
                if artifact is not None:
                    stored_artifact = self._store.append_calibration_artifact(artifact)
                    if stored_artifact != artifact:
                        raise CalibrationBridgeError(
                            "calibration store returned a different frozen artifact"
                        )
                association = CalibrationAssociation.create(
                    selection_id=checked_selection.selection_id,
                    inference=checked_inference,
                    score_family=binding.score_family,
                    runtime_revision=binding.runtime_revision,
                    preprocess_revision=binding.preprocess_revision,
                    evaluated_at=checked_selection.selected_at,
                    raw_score=raw_score,
                    deterministic_inputs=deterministic_inputs,
                    policy_decision=None,
                    calibration_artifact=artifact,
                    created_at=checked_selection.selected_at,
                )
                stored_association = self._store.append_calibration_association(association)
            except CalibrationBridgeError:
                raise
            except Exception as exc:
                raise CalibrationBridgeError(
                    "calibration association could not be appended to the accepted evidence ledger"
                ) from exc
            if stored_association != association:
                raise CalibrationBridgeError(
                    "calibration store returned a conflicting selected-inference association"
                )
            associations.append(association)
        return tuple(associations)

    def _score_input(
        self,
        *,
        binding: CalibrationBinding,
        inference: ModelInference,
        selected_output: SelectedAttemptOutput,
        enriched_output: OrchestratorEnrichedOutput,
    ) -> tuple[float | None, dict[str, object]]:
        return accepted_calibration_score_input(
            score_source=binding.score_source,
            source_claim_ordinal=binding.source_claim_ordinal,
            inference=inference,
            selected_output=selected_output,
            enriched_output=enriched_output,
        )


def _strict_calibration_instance[CalibrationModelT: StrictModel](
    value: CalibrationModelT,
    expected_type: type[CalibrationModelT],
    label: str,
) -> CalibrationModelT:
    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    try:
        return expected_type.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CalibrationBridgeError(f"{label} failed strict calibration validation") from exc


def _reported_terminal_score(value: object) -> float | None:
    if not isinstance(value, Mapping) or set(value) != {"value"}:
        return None
    raw = value.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    score = float(raw)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def calibration_association_id(*, selection_id: str, score_family: str) -> str:
    """Return the one immutable association key for a selected score family."""

    return "calibration-association:" + semantic_sha256(
        {"selection_id": selection_id, "score_family": score_family}
    )


def calibration_inputs_sha256(inputs: Mapping[str, object]) -> Sha256Digest:
    """Hash deterministic score inputs independently of model and policy values."""

    frozen_inputs = _freeze_json_object(inputs, field_name="deterministic_inputs")
    return semantic_sha256(_thaw_json_value(frozen_inputs))


def _artifact_projection(
    *,
    schema_version: str,
    applicability: CalibrationApplicability,
    fitting_method: CalibrationFittingMethod,
    fitting_parameters: Mapping[str, object],
    training_population: CalibrationTrainingPopulation,
    grouped_split_lineage: CalibrationGroupedSplitLineage,
    fitted_at: str,
    valid_from: str,
    valid_until: str | None,
    created_at: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "applicability": applicability,
        "fitting_method": fitting_method,
        "fitting_parameters": _thaw_json_value(fitting_parameters),
        "training_population": training_population,
        "grouped_split_lineage": grouped_split_lineage,
        "fitted_at": fitted_at,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "created_at": created_at,
    }


def _parse_timestamp(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("calibration timestamp must be timezone-aware")
    return parsed


def _finite_parameter(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"calibration parameter {name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"calibration parameter {name} must be finite")
    return number


def _unit_interval(value: object, name: str) -> float:
    number = _finite_parameter(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _isotonic_knots(parameters: Mapping[str, object]) -> tuple[tuple[float, float], ...]:
    raw_knots = parameters.get("knots")
    if not isinstance(raw_knots, (list, tuple)) or len(raw_knots) < 2:
        raise ValueError("isotonic calibration requires at least two knots")
    knots: list[tuple[float, float]] = []
    for index, item in enumerate(raw_knots):
        if not isinstance(item, Mapping) or set(item) != {"raw_score", "probability"}:
            raise ValueError("isotonic calibration knots require raw_score and probability")
        score = _unit_interval(item["raw_score"], f"knots[{index}].raw_score")
        probability = _unit_interval(item["probability"], f"knots[{index}].probability")
        knots.append((score, probability))
    for left, right in pairwise(knots):
        if left[0] >= right[0] or left[1] > right[1]:
            raise ValueError("isotonic calibration knots must be strictly ordered and monotonic")
    return tuple(knots)


def _validate_fitting_parameters(
    method: CalibrationFittingMethod,
    parameters: Mapping[str, object],
) -> None:
    if method is CalibrationFittingMethod.IDENTITY:
        if parameters:
            raise ValueError("identity calibration does not accept fitting parameters")
        return
    if method is CalibrationFittingMethod.PLATT_LOGISTIC:
        if set(parameters) != {"slope", "intercept"}:
            raise ValueError("Platt calibration requires slope and intercept parameters")
        _finite_parameter(parameters["slope"], "slope")
        _finite_parameter(parameters["intercept"], "intercept")
        return
    if method is CalibrationFittingMethod.ISOTONIC_LINEAR:
        if set(parameters) != {"knots"}:
            raise ValueError("isotonic calibration requires only knot parameters")
        _isotonic_knots(parameters)
        return
    raise AssertionError("unhandled calibration fitting method")


__all__ = [
    "ACCEPTED_INFERENCE_CALIBRATION_EXTRACTOR_VERSION",
    "AcceptedInferenceCalibrationBridge",
    "CalibrationApplicability",
    "CalibrationArtifact",
    "CalibrationAssociation",
    "CalibrationAssociationOutcome",
    "CalibrationAssociationStore",
    "CalibrationBinding",
    "CalibrationBridgeError",
    "CalibrationFittingMethod",
    "CalibrationGroupedSplitLineage",
    "CalibrationPolicyDecision",
    "CalibrationScoreSource",
    "CalibrationTrainingPopulation",
    "accepted_calibration_score_input",
    "calibration_artifact_digest",
    "calibration_association_id",
    "calibration_inputs_sha256",
]
