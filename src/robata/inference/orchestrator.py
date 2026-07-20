"""Provider-neutral, fail-closed vision inference orchestration.

The module implements Architecture V1.1 sections 9.3 and 10 without binding
the application layer to a concrete model provider or persistence engine.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol, cast
from urllib.parse import urljoin
from uuid import NAMESPACE_URL, uuid5

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import Field, StringConstraints
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable

from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import (
    CanonicalizationError,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.inference.adapter import (
    JsonSchemaRef,
    PackageInput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionModelAdapter,
)
from robata.inference.input_plan import InferenceCallPart, InferenceInputPlan
from robata.inference.models import (
    CapabilitySnapshot,
    InferenceAttemptSelection,
    InferenceFailure,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    ModelInference,
    ModelInferenceUsage,
    Retryability,
    VisionTask,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Clock = Callable[[], datetime]


class InferenceOrchestrationError(RuntimeError):
    """Base class for local orchestration failures."""


class OrchestrationConfigurationError(InferenceOrchestrationError):
    """Raised when required pinned policy, schema, or provider data is absent."""


class CapabilityValidationError(InferenceOrchestrationError):
    """Raised before dispatch when provider capabilities do not satisfy policy."""


class InferenceLedgerError(InferenceOrchestrationError):
    """Raised when append-only inference records conflict."""


class InferencePolicy(StrictModel):
    """Immutable, task-specific provider and artifact selection policy."""

    schema_version: Literal["1.0"] = "1.0"
    policy_version: SchemaVersion
    task: VisionTask
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    prompt_version: SchemaVersion
    prompt_artifact_id: NonEmptyString
    prompt_sha256: Sha256Digest
    output_schema: JsonSchemaRef
    enriched_output_schema: JsonSchemaRef | None = None
    generation_config: dict[str, object]
    timeout_ms: PositiveInt
    selection_policy_version: SchemaVersion
    required_input_mode: InputMode
    required_media_types: tuple[NonEmptyString, ...] = ()
    required_data_handling_policy_version: SchemaVersion | None = None


class InferenceIntent(StrictModel):
    """Durable dispatch intent written before an adapter can run."""

    schema_version: Literal["1.0"] = "1.0"
    inference_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    request_id: OpaqueUuid
    idempotency_key: NonEmptyString
    task: VisionTask
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    adapter_version: SchemaVersion
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    input_config: dict[str, object]
    sampling_config: dict[str, object]
    input_plan_id: OpaqueUuid | None = None
    input_plan_semantic_sha256: Sha256Digest | None = None
    input_plan_part_ordinal: Annotated[int, Field(strict=True, ge=0)] | None = None
    input_plan_part_count: PositiveInt | None = None
    input_plan_part_semantic_sha256: Sha256Digest | None = None
    experiment_id: NonEmptyString | None = None
    shadow_route_id: NonEmptyString | None = None
    primary_inference_id: NonEmptyString | None = None
    attempt: PositiveInt
    retry_count: Annotated[int, Field(strict=True, ge=0)]
    shadow: bool
    request: VisionInferenceRequest
    queued_at: Rfc3339Timestamp
    created_at: Rfc3339Timestamp


class InferenceLedger(Protocol):
    """Minimal persistence port required by the orchestrator."""

    def append_intent(self, intent: InferenceIntent) -> InferenceIntent:
        """Append an intent, or return an identical existing intent."""
        ...

    def get_intent(self, inference_id: str) -> InferenceIntent | None:
        """Look up an intent by attempt identity."""
        ...

    def append_terminal(self, inference: ModelInference) -> ModelInference:
        """Append a terminal attempt, or return an identical existing record."""
        ...

    def get_terminal(self, inference_id: str) -> ModelInference | None:
        """Look up a terminal attempt by identity."""
        ...

    def append_selection(self, selection: InferenceAttemptSelection) -> InferenceAttemptSelection:
        """Select at most one attempt for a logical invocation and policy."""
        ...

    def get_selection(
        self, logical_invocation_id: str, policy_version: str
    ) -> InferenceAttemptSelection | None:
        """Return a previously selected attempt."""
        ...


class InferenceExecutionGate(Protocol):
    """Dependency-injected quota/concurrency/deadline gate."""

    async def acquire(self, *, capabilities: ModelCapabilities, timeout_ms: int) -> None:
        """Allow dispatch or raise before the adapter is invoked."""
        ...


class InMemoryInferenceLedger:
    """Append-only reference ledger suitable for local execution and tests."""

    def __init__(self) -> None:
        self._intents: dict[str, InferenceIntent] = {}
        self._terminals: dict[str, ModelInference] = {}
        self._selections: dict[tuple[str, str], InferenceAttemptSelection] = {}

    def append_intent(self, intent: InferenceIntent) -> InferenceIntent:
        existing = self._intents.get(intent.inference_id)
        if existing is not None and existing != intent:
            raise InferenceLedgerError(f"conflicting intent: {intent.inference_id}")
        self._intents[intent.inference_id] = intent
        return intent

    def get_intent(self, inference_id: str) -> InferenceIntent | None:
        return self._intents.get(inference_id)

    def append_terminal(self, inference: ModelInference) -> ModelInference:
        existing = self._terminals.get(inference.inference_id)
        if existing is not None and existing != inference:
            raise InferenceLedgerError(f"conflicting terminal attempt: {inference.inference_id}")
        if self.get_intent(inference.inference_id) is None:
            raise InferenceLedgerError("terminal attempt requires a persisted intent")
        self._terminals[inference.inference_id] = inference
        return inference

    def get_terminal(self, inference_id: str) -> ModelInference | None:
        return self._terminals.get(inference_id)

    def append_selection(self, selection: InferenceAttemptSelection) -> InferenceAttemptSelection:
        key = (selection.logical_invocation_id, selection.policy_version)
        existing = self._selections.get(key)
        if existing is not None and existing != selection:
            raise InferenceLedgerError(
                "logical invocation already has a different selected attempt"
            )
        terminal = self.get_terminal(selection.inference_id)
        if terminal is None or terminal.status is not InferenceStatus.SUCCEEDED:
            raise InferenceLedgerError("selection requires a successful terminal attempt")
        if terminal.shadow or not terminal.output_valid:
            raise InferenceLedgerError("shadow or invalid output cannot be selected")
        self._selections[key] = selection
        return selection

    def get_selection(
        self, logical_invocation_id: str, policy_version: str
    ) -> InferenceAttemptSelection | None:
        return self._selections.get((logical_invocation_id, policy_version))

    def list_intents(self) -> tuple[InferenceIntent, ...]:
        return tuple(self._intents.values())

    def list_terminals(self) -> tuple[ModelInference, ...]:
        return tuple(self._terminals.values())

    def list_selections(self) -> tuple[InferenceAttemptSelection, ...]:
        return tuple(self._selections.values())


class _NoopExecutionGate:
    async def acquire(self, *, capabilities: ModelCapabilities, timeout_ms: int) -> None:
        del capabilities, timeout_ms


class _CompiledSchema:
    def __init__(
        self,
        reference: JsonSchemaRef,
        document: Mapping[str, object],
        *,
        registry: Registry[Any],
    ) -> None:
        self.reference = reference
        self.document = dict(document)
        try:
            Draft202012Validator.check_schema(self.document)
        except SchemaError as exc:
            raise OrchestrationConfigurationError(
                f"invalid JSON Schema {reference.schema_id}@{reference.version}: {exc.message}"
            ) from exc
        self.validator = Draft202012Validator(self.document, registry=registry)

    def validate(self, instance: object) -> None:
        try:
            self.validator.validate(instance)
        except Unresolvable as exc:
            raise OrchestrationConfigurationError(
                f"schema dependency is not locally resolvable for "
                f"{self.reference.schema_id}@{self.reference.version}: {exc.ref}"
            ) from exc


class _DuplicateJsonKeyError(ValueError):
    """Raised when an exact schema artifact contains an ambiguous JSON object."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKeyError(f"duplicate JSON object key: {key!r}")
        document[key] = value
    return document


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_exact_schema_artifact(
    *, artifact_id: str, raw: bytes, expected_sha256: str | None = None
) -> Mapping[str, object]:
    if expected_sha256 is not None and exact_bytes_sha256(raw) != expected_sha256:
        raise OrchestrationConfigurationError(f"schema artifact digest mismatch: {artifact_id!r}")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise OrchestrationConfigurationError(
            f"schema artifact must not contain a UTF-8 BOM: {artifact_id!r}"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OrchestrationConfigurationError(
            f"schema artifact is not strict UTF-8: {artifact_id!r}"
        ) from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJsonKeyError, ValueError) as exc:
        raise OrchestrationConfigurationError(
            f"schema artifact is not strict JSON: {artifact_id!r}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise OrchestrationConfigurationError(
            f"schema artifact root must be a JSON object: {artifact_id!r}"
        )
    return cast(Mapping[str, object], document)


def _schema_references(
    value: object,
    *,
    base_uri: str,
) -> tuple[tuple[str, str], ...]:
    references: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        nested_base = base_uri
        schema_id = value.get("$id")
        if schema_id is not None:
            if not isinstance(schema_id, str) or not schema_id:
                raise OrchestrationConfigurationError(
                    "exact schema artifact $id must be a nonempty string"
                )
            nested_base = urljoin(base_uri, schema_id)
        for keyword in ("$ref", "$dynamicRef"):
            reference = value.get(keyword)
            if reference is not None:
                if not isinstance(reference, str) or not reference:
                    raise OrchestrationConfigurationError(
                        f"exact schema artifact {keyword} must be a nonempty string"
                    )
                references.append((nested_base, reference))
        for nested in value.values():
            references.extend(_schema_references(nested, base_uri=nested_base))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            references.extend(_schema_references(nested, base_uri=base_uri))
    return tuple(references)


def _local_exact_schema_registry(
    documents: Mapping[str, Mapping[str, object]],
) -> Registry[Any]:
    resources: list[tuple[str, Resource[Any]]] = []
    resource_by_artifact: dict[str, Resource[Any]] = {}
    artifact_by_document_id: dict[str, str] = {}
    for artifact_id, document in documents.items():
        try:
            Draft202012Validator.check_schema(dict(document))
        except SchemaError as exc:
            raise OrchestrationConfigurationError(
                f"invalid exact JSON Schema artifact {artifact_id!r}: {exc.message}"
            ) from exc
        try:
            resource = Resource.from_contents(document)
        except CannotDetermineSpecification as exc:
            raise OrchestrationConfigurationError(
                f"exact schema artifact cannot determine its JSON Schema dialect: {artifact_id!r}"
            ) from exc
        resource_by_artifact[artifact_id] = resource
        document_id = document.get("$id")
        if document_id is None:
            continue
        if not isinstance(document_id, str) or not document_id:
            raise OrchestrationConfigurationError(
                f"exact schema artifact has an invalid $id: {artifact_id!r}"
            )
        previous = artifact_by_document_id.get(document_id)
        if previous is not None:
            raise OrchestrationConfigurationError(
                f"exact schema artifacts {previous!r} and {artifact_id!r} share $id {document_id!r}"
            )
        artifact_by_document_id[document_id] = artifact_id
        resources.append((document_id, resource))

    registry = Registry[Any]().with_resources(resources).crawl()
    for artifact_id, document in documents.items():
        document_id = document.get("$id")
        base_uri = document_id if isinstance(document_id, str) else ""
        for reference_base, reference in _schema_references(document, base_uri=base_uri):
            try:
                if reference_base:
                    registry.resolver(reference_base).lookup(reference)
                else:
                    registry.resolver_with_root(resource_by_artifact[artifact_id]).lookup(reference)
            except Unresolvable as exc:
                raise OrchestrationConfigurationError(
                    f"exact schema dependency is not locally resolvable in "
                    f"{artifact_id!r}: {reference}"
                ) from exc
    return registry


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(clock: Clock) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OrchestrationConfigurationError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _stable_uuid(namespace: str, digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{digest}"))


def _exception_detail(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _bind_input_plan_measurements(
    input_config: dict[str, object],
    input_plan: InferenceInputPlan,
    part: InferenceCallPart | None,
) -> None:
    """Expose the selected call-part usage to the capability gate."""

    if part is None:
        measurements = {
            "input_images": input_plan.measured_limits.max_images_per_request,
            "max_pixels_per_image": input_plan.measured_limits.max_pixels_per_image,
            "payload_bytes": input_plan.measured_limits.max_payload_bytes_per_request,
            "input_tokens": input_plan.measured_limits.max_input_tokens_per_request,
        }
        _merge_input_measurements(input_config, measurements)
        return
    items = input_plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive]
    image_items = tuple(item for item in items if item.artifact.media_type.startswith("image/"))
    measurements = {
        "input_images": len(image_items),
        "max_pixels_per_image": max(
            (item.artifact.width * item.artifact.height for item in image_items),
            default=0,
        ),
        "payload_bytes": sum(item.artifact.byte_count for item in items),
        "input_tokens": part.measured_input_tokens,
    }
    _merge_input_measurements(input_config, measurements)


def _merge_input_measurements(
    input_config: dict[str, object],
    measurements: Mapping[str, int],
) -> None:
    for field, value in measurements.items():
        existing = input_config.get(field)
        if existing is not None and existing != value:
            raise OrchestrationConfigurationError(
                f"input_config.{field} conflicts with the input plan measurement"
            )
        input_config[field] = value


class InferenceOrchestrator:
    """Coordinate one provider-neutral inference attempt end to end."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, VisionModelAdapter] | None = None,
        task_policies: Mapping[VisionTask, InferencePolicy] | None = None,
        schema_documents: Mapping[str, Mapping[str, object]] | None = None,
        schema_artifacts: Mapping[str, bytes | bytearray | memoryview] | None = None,
        ledger: InferenceLedger | None = None,
        execution_gate: InferenceExecutionGate | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._adapters = dict(adapters or {})
        self._policies = dict(task_policies or {})
        self._schema_documents = {
            artifact_id: dict(document)
            for artifact_id, document in (schema_documents or {}).items()
        }
        self._schema_artifacts = {
            artifact_id: bytes(raw) for artifact_id, raw in (schema_artifacts or {}).items()
        }
        self._exact_schema_documents: dict[str, Mapping[str, object]] | None = None
        self._exact_schema_registry: Registry[Any] | None = None
        self._ledger = ledger or InMemoryInferenceLedger()
        self._execution_gate = execution_gate or _NoopExecutionGate()
        self._clock = clock or _utc_now
        self._compiled_schemas: dict[str, _CompiledSchema] = {}

    @property
    def ledger(self) -> InferenceLedger:
        return self._ledger

    @property
    def intents(self) -> tuple[InferenceIntent, ...]:
        if not isinstance(self._ledger, InMemoryInferenceLedger):
            raise InferenceLedgerError("configured ledger does not expose in-memory snapshots")
        return self._ledger.list_intents()

    @property
    def attempts(self) -> tuple[ModelInference, ...]:
        if not isinstance(self._ledger, InMemoryInferenceLedger):
            raise InferenceLedgerError("configured ledger does not expose in-memory snapshots")
        return self._ledger.list_terminals()

    @property
    def selections(self) -> tuple[InferenceAttemptSelection, ...]:
        if not isinstance(self._ledger, InMemoryInferenceLedger):
            raise InferenceLedgerError("configured ledger does not expose in-memory snapshots")
        return self._ledger.list_selections()

    def selected_attempt(
        self, *, logical_invocation_id: str, policy_version: str
    ) -> InferenceAttemptSelection | None:
        return self._ledger.get_selection(logical_invocation_id, policy_version)

    def _policy(self, task: VisionTask) -> InferencePolicy:
        policy = self._policies.get(task)
        if policy is None:
            raise OrchestrationConfigurationError(f"no inference policy configured for {task}")
        if policy.task is not task:
            raise OrchestrationConfigurationError(
                f"policy key {task} does not match policy task {policy.task}"
            )
        return policy

    def _adapter(self, policy: InferencePolicy) -> VisionModelAdapter:
        adapter = self._adapters.get(policy.provider)
        if adapter is None:
            raise OrchestrationConfigurationError(
                f"no inference adapter configured for provider {policy.provider!r}"
            )
        if adapter.provider != policy.provider:
            raise OrchestrationConfigurationError(
                "adapter registry key does not match adapter provider identity"
            )
        return adapter

    def _schema(self, reference: JsonSchemaRef) -> _CompiledSchema:
        existing = self._compiled_schemas.get(reference.artifact_id)
        if existing is not None:
            if existing.reference != reference:
                raise OrchestrationConfigurationError(
                    f"schema artifact {reference.artifact_id!r} has conflicting references"
                )
            return existing

        artifact_id = reference.artifact_id
        has_document = artifact_id in self._schema_documents
        has_artifact = artifact_id in self._schema_artifacts
        if has_document and has_artifact:
            raise OrchestrationConfigurationError(
                f"schema artifact has both exact and synthetic sources: {artifact_id!r}"
            )
        if has_artifact:
            raw = self._schema_artifacts[artifact_id]
            if exact_bytes_sha256(raw) != reference.sha256:
                raise OrchestrationConfigurationError(
                    f"schema artifact digest mismatch: {artifact_id!r}"
                )
            registry = self._require_exact_schema_registry()
            assert self._exact_schema_documents is not None
            document = self._exact_schema_documents[artifact_id]
        elif has_document:
            document = self._schema_documents[artifact_id]
            registry = self._require_exact_schema_registry()
        else:
            raise OrchestrationConfigurationError(
                f"schema artifact is not configured: {artifact_id!r}"
            )
        if not has_artifact:
            try:
                digest = semantic_sha256(document)
            except (CanonicalizationError, TypeError, ValueError) as exc:
                raise OrchestrationConfigurationError(
                    f"schema artifact cannot be canonicalized: {artifact_id!r}"
                ) from exc
            if digest != reference.sha256:
                raise OrchestrationConfigurationError(
                    f"schema artifact digest mismatch: {artifact_id!r}"
                )
        compiled = _CompiledSchema(reference, document, registry=registry)
        self._compiled_schemas[artifact_id] = compiled
        return compiled

    def _require_exact_schema_registry(self) -> Registry[Any]:
        if self._exact_schema_registry is not None:
            return self._exact_schema_registry
        pinned_references: dict[str, JsonSchemaRef] = {}
        for policy in self._policies.values():
            reference = policy.output_schema
            existing = pinned_references.get(reference.artifact_id)
            if existing is not None and existing != reference:
                raise OrchestrationConfigurationError(
                    f"schema artifact {reference.artifact_id!r} has conflicting policy pins"
                )
            pinned_references[reference.artifact_id] = reference
        for artifact_id, reference in pinned_references.items():
            raw = self._schema_artifacts.get(artifact_id)
            if raw is not None and exact_bytes_sha256(raw) != reference.sha256:
                raise OrchestrationConfigurationError(
                    f"schema artifact digest mismatch: {artifact_id!r}"
                )
        documents = {
            artifact_id: _parse_exact_schema_artifact(artifact_id=artifact_id, raw=raw)
            for artifact_id, raw in self._schema_artifacts.items()
        }
        registry = _local_exact_schema_registry(documents)
        self._exact_schema_documents = documents
        self._exact_schema_registry = registry
        return registry

    @staticmethod
    def _validate_package_inputs(
        package_inputs: Sequence[PackageInput],
    ) -> tuple[PackageInput, ...]:
        inputs = tuple(package_inputs)
        if not inputs:
            raise OrchestrationConfigurationError("at least one package input is required")
        expected_ordinals = tuple(range(len(inputs)))
        ordinals = tuple(item.ordinal for item in inputs)
        if ordinals != expected_ordinals:
            raise OrchestrationConfigurationError(
                "package input ordinals must be contiguous and ordered from zero"
            )
        package_ids = tuple(item.package_id for item in inputs)
        if len(set(package_ids)) != len(package_ids):
            raise OrchestrationConfigurationError("package inputs must have unique package IDs")
        return inputs

    @staticmethod
    def _validate_capabilities(
        *,
        policy: InferencePolicy,
        capabilities: ModelCapabilities,
        package_inputs: Sequence[PackageInput],
        input_config: Mapping[str, object],
    ) -> None:
        expected_identity = (policy.provider, policy.model_name, policy.model_version)
        actual_identity = (
            capabilities.provider,
            capabilities.model_name,
            capabilities.model_version,
        )
        if actual_identity != expected_identity:
            raise CapabilityValidationError("capability snapshot model identity mismatch")
        if policy.task not in capabilities.supported_tasks:
            raise CapabilityValidationError(f"model does not support task {policy.task}")
        if policy.required_input_mode not in capabilities.input_modes:
            raise CapabilityValidationError(
                f"model does not support input mode {policy.required_input_mode}"
            )
        unsupported_media = set(policy.required_media_types) - set(
            capabilities.accepted_media_types
        )
        if unsupported_media:
            raise CapabilityValidationError(
                f"model does not accept required media types: {sorted(unsupported_media)}"
            )
        if not capabilities.supports_json_schema:
            raise CapabilityValidationError("model does not support JSON Schema output")
        required_policy = policy.required_data_handling_policy_version
        if (
            required_policy is not None
            and capabilities.data_handling_policy_version != required_policy
        ):
            raise CapabilityValidationError("data handling policy version mismatch")
        InferenceOrchestrator._validate_optional_limit(
            input_config,
            field="input_images",
            limit=capabilities.max_images_per_request,
        )
        InferenceOrchestrator._validate_optional_limit(
            input_config,
            field="max_pixels_per_image",
            limit=capabilities.max_pixels_per_image,
        )
        InferenceOrchestrator._validate_optional_limit(
            input_config,
            field="payload_bytes",
            limit=capabilities.max_payload_bytes,
        )
        InferenceOrchestrator._validate_optional_limit(
            input_config,
            field="input_tokens",
            limit=capabilities.max_input_tokens,
        )

    @staticmethod
    def _validate_optional_limit(
        config: Mapping[str, object], *, field: str, limit: int | None
    ) -> None:
        value = config.get(field)
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OrchestrationConfigurationError(f"input_config.{field} must be a nonnegative int")
        if limit is not None and value > limit:
            raise CapabilityValidationError(f"input_config.{field} exceeds model limit")

    async def _select_adapter(self, task: VisionTask) -> str:
        """Select the pinned provider identifier for a task."""
        return self._policy(task).provider

    async def _select_model(self, task: VisionTask) -> tuple[str, str]:
        """Select the pinned model name and version for a task."""
        policy = self._policy(task)
        return policy.model_name, policy.model_version

    async def _select_prompt(self, task: VisionTask) -> tuple[str, str, str]:
        """Select the pinned prompt artifacts for a task."""
        policy = self._policy(task)
        return policy.prompt_version, policy.prompt_artifact_id, policy.prompt_sha256

    async def _select_capability_snapshot(
        self,
        provider: str,
        model_name: str,
        model_version: str,
    ) -> CapabilitySnapshot:
        """Discover and return the immutable capability reference."""
        adapter = self._adapters.get(provider)
        if adapter is None or adapter.provider != provider:
            raise OrchestrationConfigurationError(
                f"no matching adapter configured for provider {provider!r}"
            )
        capabilities = await adapter.capabilities(model_name, model_version)
        if (
            capabilities.provider != provider
            or capabilities.model_name != model_name
            or capabilities.model_version != model_version
        ):
            raise CapabilityValidationError("capability snapshot identity mismatch")
        return CapabilitySnapshot(
            schema_version="1.0",
            snapshot_id=capabilities.snapshot_id,
            snapshot_digest=capabilities.snapshot_digest,
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            model_version=capabilities.model_version,
            observed_at=capabilities.observed_at,
        )

    @staticmethod
    def _package_digest(package_inputs: Sequence[PackageInput]) -> Sha256Digest:
        return semantic_sha256([item.model_dump(mode="json") for item in package_inputs])

    @staticmethod
    def _validate_input_plan(
        *,
        input_plan: InferenceInputPlan,
        task: VisionTask,
        package_inputs: Sequence[PackageInput],
        rendered_input_digest: str | None,
        part_ordinal: int | None,
    ) -> InferenceCallPart | None:
        if not isinstance(input_plan, InferenceInputPlan):
            raise OrchestrationConfigurationError("input_plan must be an InferenceInputPlan")
        try:
            InferenceInputPlan.model_validate(input_plan.model_dump(mode="python"))
        except ValueError as exc:
            raise OrchestrationConfigurationError(
                "input_plan failed immutable contract validation"
            ) from exc
        if input_plan.request_catalog.task is not task or input_plan.subject.task is not task:
            raise OrchestrationConfigurationError("input_plan task does not match the request")
        expected_packages = tuple(
            (
                item.package_id,
                item.ordinal,
                item.package_semantic_content_sha256,
                item.package_manifest_sha256,
            )
            for item in package_inputs
        )
        actual_packages = tuple(
            (
                item.package_id,
                item.ordinal,
                item.semantic_content_sha256,
                item.manifest_bytes_sha256,
            )
            for item in input_plan.subject.packages
        )
        if actual_packages != expected_packages:
            raise OrchestrationConfigurationError(
                "input_plan subject packages do not match the request package inputs"
            )
        if part_ordinal is None:
            if (
                rendered_input_digest is not None
                and rendered_input_digest != input_plan.rendering_sha256
            ):
                raise OrchestrationConfigurationError(
                    "rendered_input_digest does not match input_plan.rendering_sha256"
                )
            return None
        if isinstance(part_ordinal, bool) or not isinstance(part_ordinal, int):
            raise OrchestrationConfigurationError("input_plan_part_ordinal must be an integer")
        if part_ordinal < 0 or part_ordinal >= len(input_plan.call_plan.parts):
            raise OrchestrationConfigurationError(
                "input_plan_part_ordinal is outside the call plan"
            )
        part = input_plan.call_plan.parts[part_ordinal]
        if rendered_input_digest is not None and rendered_input_digest != part.item_manifest_sha256:
            raise OrchestrationConfigurationError(
                "rendered_input_digest does not match the selected call part"
            )
        return part

    @staticmethod
    def _validate_input_plan_target_and_limits(
        *,
        input_plan: InferenceInputPlan,
        policy: InferencePolicy,
        capabilities: ModelCapabilities,
    ) -> None:
        target = input_plan.target
        if (
            target.provider != policy.provider
            or target.model_name != policy.model_name
            or target.model_version != policy.model_version
            or target.adapter_version != policy.adapter_version
            or target.capability_snapshot_id != capabilities.snapshot_id
            or target.capability_snapshot_sha256 != capabilities.snapshot_digest
        ):
            raise OrchestrationConfigurationError(
                "input_plan target does not match the selected provider capability snapshot"
            )
        prompt = input_plan.prompt_output
        if (
            prompt.prompt_version != policy.prompt_version
            or prompt.prompt_sha256 != policy.prompt_sha256
        ):
            raise OrchestrationConfigurationError(
                "input_plan prompt contract does not match the selected policy"
            )
        if prompt.provider_response_schema_sha256 != policy.output_schema.sha256:
            raise OrchestrationConfigurationError(
                "input_plan provider-response schema digest does not match the selected policy"
            )
        if (
            policy.enriched_output_schema is not None
            and prompt.enriched_domain_schema_sha256 != policy.enriched_output_schema.sha256
        ):
            raise OrchestrationConfigurationError(
                "input_plan enriched schema digest does not match the selected policy"
            )
        expected_limits = (
            capabilities.max_images_per_request,
            capabilities.max_pixels_per_image,
            capabilities.max_payload_bytes,
            capabilities.max_input_tokens,
        )
        actual_limits = (
            input_plan.applicable_limits.max_images_per_request,
            input_plan.applicable_limits.max_pixels_per_image,
            input_plan.applicable_limits.max_payload_bytes_per_request,
            input_plan.applicable_limits.max_input_tokens_per_request,
        )
        if actual_limits != expected_limits:
            raise OrchestrationConfigurationError(
                "input_plan limits are not the pinned capability limits"
            )

    @staticmethod
    def _logical_digest(
        *,
        package_input_set_sha256: str,
        task: VisionTask,
        provider: str,
        model_name: str,
        model_version: str,
        adapter_version: str,
        prompt_version: str,
        prompt_artifact_id: str,
        prompt_sha256: str,
        rendered_input_digest: str,
        input_plan_semantic_sha256: str | None,
        input_plan_part_semantic_sha256: str | None,
        output_schema: JsonSchemaRef,
        capability_snapshot: CapabilitySnapshot,
        generation_config: Mapping[str, object],
    ) -> Sha256Digest:
        projection: dict[str, object] = {
            "package_input_set_sha256": package_input_set_sha256,
            "task": task,
            "provider": provider,
            "model_name": model_name,
            "model_version": model_version,
            "adapter_version": adapter_version,
            "prompt_version": prompt_version,
            "prompt_artifact_id": prompt_artifact_id,
            "prompt_sha256": prompt_sha256,
            "rendered_input_digest": rendered_input_digest,
            "input_plan_semantic_sha256": input_plan_semantic_sha256,
            "output_schema": output_schema,
            "capability_snapshot_digest": capability_snapshot.snapshot_digest,
            "generation_config": dict(generation_config),
        }
        if input_plan_part_semantic_sha256 is not None:
            projection["input_plan_part_semantic_sha256"] = input_plan_part_semantic_sha256
        return semantic_sha256(projection)

    @staticmethod
    def _attempt_digest(
        logical_digest: str, *, attempt: int, retry_count: int, shadow: bool
    ) -> Sha256Digest:
        return semantic_sha256(
            {
                "logical_invocation_digest": logical_digest,
                "attempt": attempt,
                "retry_count": retry_count,
                "shadow": shadow,
            }
        )

    @staticmethod
    def _usage(value: object) -> ModelInferenceUsage:
        if hasattr(value, "model_dump"):
            return ModelInferenceUsage.model_validate(value.model_dump(mode="python"))
        raise InferenceOrchestrationError("adapter returned an invalid usage object")

    @staticmethod
    def _failure(*, code: str, detail: str, retryability: Retryability) -> InferenceFailure:
        return InferenceFailure(code=code, detail=detail, retryability=retryability)

    def _terminal(
        self,
        *,
        intent: InferenceIntent,
        started_at: str,
        status: InferenceStatus,
        provider_request_id: str | None,
        latency_ms: int,
        usage: ModelInferenceUsage,
        raw_output: dict[str, object] | None,
        normalized_output: dict[str, object] | None,
        output_valid: bool,
        reported_confidence: float | None,
        failure: InferenceFailure | None,
    ) -> ModelInference:
        request = intent.request
        package_id = (
            request.package_inputs[0].package_id if len(request.package_inputs) == 1 else None
        )
        confidence: dict[str, object] | None = (
            {"value": reported_confidence} if reported_confidence is not None else None
        )
        return ModelInference(
            schema_version="1.0",
            inference_id=intent.inference_id,
            logical_invocation_id=intent.logical_invocation_id,
            request_id=intent.request_id,
            idempotency_key=intent.idempotency_key,
            mcap_id=intent.mcap_id,
            package_set_id=request.package_set_id,
            package_id=package_id,
            package_ids=tuple(item.package_id for item in request.package_inputs),
            camera_mapping_run_id=intent.camera_mapping_run_id,
            alignment_id=intent.alignment_id,
            start_ns=intent.start_ns,
            end_ns=intent.end_ns,
            stage=request.task,
            provider=request.provider,
            model_name=request.model_name,
            model_version=request.model_version,
            adapter_version=intent.adapter_version,
            prompt_version=request.prompt_version,
            prompt_artifact_id=request.prompt_artifact_id,
            prompt_sha256=request.prompt_sha256,
            rendered_input_digest=request.rendered_input_digest,
            input_plan_id=request.input_plan_id,
            input_plan_semantic_sha256=request.input_plan_semantic_sha256,
            input_plan_part_ordinal=request.input_plan_part_ordinal,
            input_plan_part_count=request.input_plan_part_count,
            input_plan_part_semantic_sha256=request.input_plan_part_semantic_sha256,
            output_schema_id=request.output_schema.schema_id,
            output_schema_version=request.output_schema.version,
            output_schema_artifact_id=request.output_schema.artifact_id,
            output_schema_sha256=request.output_schema.sha256,
            capability_snapshot_id=request.capability_snapshot_id,
            capability_snapshot_digest=request.capability_snapshot_digest,
            input_manifest_set_sha256=request.package_input_set_sha256,
            input_config=dict(intent.input_config),
            sampling_config=dict(intent.sampling_config),
            generation_config=dict(request.generation_config),
            provider_idempotency_key=request.provider_idempotency_key,
            provider_request_id=provider_request_id,
            experiment_id=intent.experiment_id,
            shadow_route_id=intent.shadow_route_id,
            primary_inference_id=intent.primary_inference_id,
            shadow=intent.shadow,
            attempt=intent.attempt,
            retry_count=intent.retry_count,
            status=status,
            queued_at=intent.queued_at,
            started_at=started_at,
            completed_at=_timestamp(self._clock),
            latency_ms=latency_ms,
            raw_output=raw_output,
            normalized_output=normalized_output,
            output_valid=output_valid,
            reported_confidence=confidence,
            calibrated_confidence=None,
            usage=usage,
            failure=failure,
            created_at=intent.created_at,
        )

    @staticmethod
    def _outcome_identity_error(
        outcome: VisionInferenceSuccess | VisionInferenceFailure,
        request: VisionInferenceRequest,
    ) -> str | None:
        if outcome.provider != request.provider:
            return "adapter outcome provider does not match request"
        if outcome.model_name != request.model_name:
            return "adapter outcome model name does not match request"
        if outcome.model_version != request.model_version:
            return "adapter outcome model version does not match request"
        return None

    @staticmethod
    def _envelope_error(
        outcome: VisionInferenceSuccess, request: VisionInferenceRequest
    ) -> str | None:
        envelope = outcome.normalized_output
        if envelope.task is not request.task:
            return "normalized output task does not match request"
        if envelope.output_schema != request.output_schema:
            return "normalized output schema reference does not match request"
        if envelope.package_input_set_sha256 != request.package_input_set_sha256:
            return "normalized output package digest does not match request"
        if envelope.input_plan_semantic_sha256 != request.input_plan_semantic_sha256:
            return "normalized output input-plan digest does not match request"
        if envelope.input_plan_part_ordinal != request.input_plan_part_ordinal:
            return "normalized output input-plan part ordinal does not match request"
        if envelope.input_plan_part_semantic_sha256 != request.input_plan_part_semantic_sha256:
            return "normalized output input-plan part digest does not match request"
        return None

    def _select_success(
        self, inference: ModelInference, *, policy_version: str
    ) -> InferenceAttemptSelection | None:
        if (
            inference.shadow
            or inference.status is not InferenceStatus.SUCCEEDED
            or not inference.output_valid
        ):
            return None
        existing = self._ledger.get_selection(inference.logical_invocation_id, policy_version)
        if existing is not None:
            return existing
        selected_at = _timestamp(self._clock)
        selection_digest = semantic_sha256(
            {
                "logical_invocation_id": inference.logical_invocation_id,
                "policy_version": policy_version,
            }
        )
        selection = InferenceAttemptSelection(
            schema_version="1.0",
            selection_id=_stable_uuid("inference-selection", selection_digest),
            inference_id=inference.inference_id,
            logical_invocation_id=inference.logical_invocation_id,
            policy_version=policy_version,
            selected_at=selected_at,
        )
        return self._ledger.append_selection(selection)

    def _persist_terminal(
        self, inference: ModelInference, *, selection_policy_version: str
    ) -> ModelInference:
        stored = self._ledger.append_terminal(inference)
        self._select_success(stored, policy_version=selection_policy_version)
        return stored

    def _local_failure_terminal(
        self,
        *,
        intent: InferenceIntent,
        started_at: str,
        status: InferenceStatus,
        code: str,
        detail: str,
        retryability: Retryability,
    ) -> ModelInference:
        return self._terminal(
            intent=intent,
            started_at=started_at,
            status=status,
            provider_request_id=None,
            latency_ms=0,
            usage=ModelInferenceUsage(
                input_frames=0,
                input_images=0,
                input_tokens=None,
                output_tokens=None,
                cost=None,
                currency=None,
            ),
            raw_output=None,
            normalized_output=None,
            output_valid=False,
            reported_confidence=None,
            failure=self._failure(
                code=code,
                detail=detail,
                retryability=retryability,
            ),
        )

    async def orchestrate(
        self,
        *,
        task: VisionTask,
        package_set_id: str | None,
        mcap_id: str,
        camera_mapping_run_id: str,
        alignment_id: str,
        start_ns: int,
        end_ns: int,
        package_inputs: Sequence[PackageInput] = (),
        rendered_input_digest: str | None = None,
        input_plan: InferenceInputPlan | None = None,
        input_plan_part_ordinal: int | None = None,
        input_config: Mapping[str, object] | None = None,
        sampling_config: Mapping[str, object] | None = None,
        metadata: Mapping[str, str] | None = None,
        attempt: int = 1,
        retry_count: int = 0,
        shadow: bool = False,
        experiment_id: str | None = None,
        shadow_route_id: str | None = None,
        primary_inference_id: str | None = None,
    ) -> ModelInference:
        """Execute and durably record a provider-neutral inference attempt."""
        if isinstance(start_ns, bool) or isinstance(end_ns, bool) or start_ns >= end_ns:
            raise OrchestrationConfigurationError("start_ns must be less than end_ns")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise OrchestrationConfigurationError("attempt must be a positive integer")
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
            or retry_count >= attempt
        ):
            raise OrchestrationConfigurationError(
                "retry_count must be nonnegative and less than attempt"
            )
        if not shadow and (
            experiment_id is not None
            or shadow_route_id is not None
            or primary_inference_id is not None
        ):
            raise OrchestrationConfigurationError("shadow lineage fields require shadow=True")

        inputs = self._validate_package_inputs(package_inputs)
        input_plan_part: InferenceCallPart | None = None
        if input_plan is not None:
            input_plan_part = self._validate_input_plan(
                input_plan=input_plan,
                task=task,
                package_inputs=inputs,
                rendered_input_digest=rendered_input_digest,
                part_ordinal=input_plan_part_ordinal,
            )
            rendered_input_digest = (
                input_plan_part.item_manifest_sha256
                if input_plan_part is not None
                else input_plan.rendering_sha256
            )
        elif input_plan_part_ordinal is not None:
            raise OrchestrationConfigurationError("input_plan_part_ordinal requires an input plan")
        if rendered_input_digest is None:
            raise OrchestrationConfigurationError("rendered_input_digest is required")
        inputs_config = dict(input_config or {})
        sampling = dict(sampling_config or {})
        request_metadata = dict(metadata or {})
        if input_plan is not None:
            plan_metadata = {
                "input_plan_id": input_plan.input_plan_id,
                "input_plan_semantic_sha256": input_plan.semantic_sha256,
                "input_plan_call_plan_sha256": input_plan.call_plan.call_plan_sha256,
            }
            if input_plan_part is not None:
                plan_metadata.update(
                    {
                        "input_plan_part_ordinal": str(input_plan_part.ordinal),
                        "input_plan_part_semantic_sha256": (input_plan_part.part_semantic_sha256),
                    }
                )
            for key, value in plan_metadata.items():
                existing = request_metadata.get(key)
                if existing is not None and existing != value:
                    raise OrchestrationConfigurationError(
                        f"metadata.{key} conflicts with the input plan"
                    )
                request_metadata[key] = value

        policy = self._policy(task)
        adapter = self._adapter(policy)
        self._schema(policy.output_schema)
        capabilities = await adapter.capabilities(policy.model_name, policy.model_version)
        if input_plan is not None:
            self._validate_input_plan_target_and_limits(
                input_plan=input_plan,
                policy=policy,
                capabilities=capabilities,
            )
            _bind_input_plan_measurements(inputs_config, input_plan, input_plan_part)
        self._validate_capabilities(
            policy=policy,
            capabilities=capabilities,
            package_inputs=inputs,
            input_config=inputs_config,
        )
        capability_snapshot = CapabilitySnapshot(
            schema_version="1.0",
            snapshot_id=capabilities.snapshot_id,
            snapshot_digest=capabilities.snapshot_digest,
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            model_version=capabilities.model_version,
            observed_at=capabilities.observed_at,
        )

        package_digest = self._package_digest(inputs)
        try:
            logical_digest = self._logical_digest(
                package_input_set_sha256=package_digest,
                task=task,
                provider=policy.provider,
                model_name=policy.model_name,
                model_version=policy.model_version,
                adapter_version=policy.adapter_version,
                prompt_version=policy.prompt_version,
                prompt_artifact_id=policy.prompt_artifact_id,
                prompt_sha256=policy.prompt_sha256,
                rendered_input_digest=rendered_input_digest,
                input_plan_semantic_sha256=(
                    input_plan.semantic_sha256 if input_plan is not None else None
                ),
                input_plan_part_semantic_sha256=(
                    input_plan_part.part_semantic_sha256 if input_plan_part is not None else None
                ),
                output_schema=policy.output_schema,
                capability_snapshot=capability_snapshot,
                generation_config=policy.generation_config,
            )
        except (CanonicalizationError, TypeError, ValueError) as exc:
            raise OrchestrationConfigurationError(
                "inference identity inputs are not canonical JSON"
            ) from exc
        logical_invocation_id = _stable_uuid("logical-invocation", logical_digest)
        attempt_digest = self._attempt_digest(
            logical_digest,
            attempt=attempt,
            retry_count=retry_count,
            shadow=shadow,
        )
        inference_id = _stable_uuid("inference-attempt", attempt_digest)
        request_id = _stable_uuid("inference-request", attempt_digest)
        idempotency_key = f"robata-inference:{attempt_digest}"

        request = VisionInferenceRequest(
            schema_version="1.0",
            logical_invocation_id=logical_invocation_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            provider=policy.provider,
            model_name=policy.model_name,
            model_version=policy.model_version,
            package_set_id=package_set_id,
            package_inputs=inputs,
            package_input_set_sha256=package_digest,
            task=task,
            prompt_version=policy.prompt_version,
            prompt_artifact_id=policy.prompt_artifact_id,
            prompt_sha256=policy.prompt_sha256,
            rendered_input_digest=rendered_input_digest,
            input_plan_id=input_plan.input_plan_id if input_plan is not None else None,
            input_plan_semantic_sha256=(
                input_plan.semantic_sha256 if input_plan is not None else None
            ),
            input_plan_part_ordinal=(
                input_plan_part.ordinal if input_plan_part is not None else None
            ),
            input_plan_part_count=(
                input_plan_part.part_count if input_plan_part is not None else None
            ),
            input_plan_part_semantic_sha256=(
                input_plan_part.part_semantic_sha256 if input_plan_part is not None else None
            ),
            input_plan=input_plan,
            output_schema=policy.output_schema,
            capability_snapshot_id=capability_snapshot.snapshot_id,
            capability_snapshot_digest=capability_snapshot.snapshot_digest,
            model_policy_version=policy.policy_version,
            generation_config=dict(policy.generation_config),
            provider_idempotency_key=(
                input_plan_part.idempotency_key
                if input_plan_part is not None
                else f"robata-provider:{attempt_digest}"
            ),
            timeout_ms=policy.timeout_ms,
            metadata=request_metadata,
        )

        existing_intent = self._ledger.get_intent(inference_id)
        now = _timestamp(self._clock)
        queued_at = existing_intent.queued_at if existing_intent is not None else now
        created_at = existing_intent.created_at if existing_intent is not None else now
        intent = InferenceIntent(
            schema_version="1.0",
            inference_id=inference_id,
            logical_invocation_id=logical_invocation_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            task=task,
            provider=policy.provider,
            model_name=policy.model_name,
            model_version=policy.model_version,
            adapter_version=policy.adapter_version,
            mcap_id=mcap_id,
            camera_mapping_run_id=camera_mapping_run_id,
            alignment_id=alignment_id,
            start_ns=start_ns,
            end_ns=end_ns,
            input_config=inputs_config,
            sampling_config=sampling,
            input_plan_id=input_plan.input_plan_id if input_plan is not None else None,
            input_plan_semantic_sha256=(
                input_plan.semantic_sha256 if input_plan is not None else None
            ),
            input_plan_part_ordinal=(
                input_plan_part.ordinal if input_plan_part is not None else None
            ),
            input_plan_part_count=(
                input_plan_part.part_count if input_plan_part is not None else None
            ),
            input_plan_part_semantic_sha256=(
                input_plan_part.part_semantic_sha256 if input_plan_part is not None else None
            ),
            experiment_id=experiment_id,
            shadow_route_id=shadow_route_id,
            primary_inference_id=primary_inference_id,
            attempt=attempt,
            retry_count=retry_count,
            shadow=shadow,
            request=request,
            queued_at=queued_at,
            created_at=created_at,
        )
        persisted_intent = self._ledger.append_intent(intent)
        existing_terminal = self._ledger.get_terminal(inference_id)
        if existing_terminal is not None:
            self._select_success(
                existing_terminal,
                policy_version=policy.selection_policy_version,
            )
            return existing_terminal
        started_at = _timestamp(self._clock)

        try:
            await self._execution_gate.acquire(
                capabilities=capabilities,
                timeout_ms=policy.timeout_ms,
            )
        except asyncio.CancelledError:
            cancelled = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.CANCELLED,
                code="ORCHESTRATION_CANCELLED",
                detail="inference cancelled while awaiting execution gate",
                retryability=Retryability.RETRYABLE,
            )
            self._persist_terminal(
                cancelled,
                selection_policy_version=policy.selection_policy_version,
            )
            raise
        except TimeoutError as exc:
            timeout = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.TIMEOUT,
                code="EXECUTION_GATE_TIMEOUT",
                detail=_exception_detail(exc),
                retryability=Retryability.RETRYABLE,
            )
            return self._persist_terminal(
                timeout,
                selection_policy_version=policy.selection_policy_version,
            )
        except Exception as exc:
            rejected = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.FAILED,
                code="EXECUTION_GATE_REJECTED",
                detail=_exception_detail(exc),
                retryability=Retryability.RETRYABLE,
            )
            return self._persist_terminal(
                rejected,
                selection_policy_version=policy.selection_policy_version,
            )

        try:
            async with asyncio.timeout(policy.timeout_ms / 1000):
                outcome = await adapter.infer(request)
        except asyncio.CancelledError:
            cancelled = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.CANCELLED,
                code="ADAPTER_CANCELLED",
                detail="inference cancelled during adapter dispatch",
                retryability=Retryability.RETRYABLE,
            )
            self._persist_terminal(
                cancelled,
                selection_policy_version=policy.selection_policy_version,
            )
            raise
        except TimeoutError as exc:
            timeout = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.TIMEOUT,
                code="ADAPTER_TIMEOUT",
                detail=_exception_detail(exc),
                retryability=Retryability.RETRYABLE,
            )
            return self._persist_terminal(
                timeout,
                selection_policy_version=policy.selection_policy_version,
            )
        except Exception as exc:
            failed = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.FAILED,
                code="ADAPTER_EXCEPTION",
                detail=_exception_detail(exc),
                retryability=Retryability.RETRYABLE,
            )
            return self._persist_terminal(
                failed,
                selection_policy_version=policy.selection_policy_version,
            )

        if not isinstance(outcome, (VisionInferenceSuccess, VisionInferenceFailure)):
            invalid_contract = self._local_failure_terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.FAILED,
                code="ADAPTER_CONTRACT_VIOLATION",
                detail="adapter returned an unsupported outcome type",
                retryability=Retryability.PERMANENT,
            )
            return self._persist_terminal(
                invalid_contract,
                selection_policy_version=policy.selection_policy_version,
            )

        identity_error = self._outcome_identity_error(outcome, request)
        if identity_error is not None:
            identity_failure = self._terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.FAILED,
                provider_request_id=outcome.provider_request_id,
                latency_ms=outcome.latency_ms,
                usage=self._usage(outcome.usage),
                raw_output=(
                    {"artifact_id": outcome.raw_output_artifact_id}
                    if outcome.raw_output_artifact_id is not None
                    else None
                ),
                normalized_output=None,
                output_valid=False,
                reported_confidence=None,
                failure=self._failure(
                    code="ADAPTER_OUTCOME_IDENTITY_MISMATCH",
                    detail=identity_error,
                    retryability=Retryability.PERMANENT,
                ),
            )
            return self._persist_terminal(
                identity_failure,
                selection_policy_version=policy.selection_policy_version,
            )

        if isinstance(outcome, VisionInferenceFailure):
            provider_failure = self._terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=outcome.status,
                provider_request_id=outcome.provider_request_id,
                latency_ms=outcome.latency_ms,
                usage=self._usage(outcome.usage),
                raw_output=(
                    {"artifact_id": outcome.raw_output_artifact_id}
                    if outcome.raw_output_artifact_id is not None
                    else None
                ),
                normalized_output=None,
                output_valid=False,
                reported_confidence=None,
                failure=outcome.failure,
            )
            return self._persist_terminal(
                provider_failure,
                selection_policy_version=policy.selection_policy_version,
            )

        payload = dict(outcome.normalized_output.payload)
        raw_output: dict[str, object] = {"artifact_id": outcome.raw_output_artifact_id}
        envelope_error = self._envelope_error(outcome, request)
        if envelope_error is not None:
            invalid_envelope = self._terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.INVALID_OUTPUT,
                provider_request_id=outcome.provider_request_id,
                latency_ms=outcome.latency_ms,
                usage=self._usage(outcome.usage),
                raw_output=raw_output,
                normalized_output=payload,
                output_valid=False,
                reported_confidence=outcome.reported_confidence,
                failure=self._failure(
                    code="OUTPUT_ENVELOPE_MISMATCH",
                    detail=envelope_error,
                    retryability=Retryability.PERMANENT,
                ),
            )
            return self._persist_terminal(
                invalid_envelope,
                selection_policy_version=policy.selection_policy_version,
            )

        compiled_schema = self._schema(request.output_schema)
        try:
            compiled_schema.validate(payload)
        except JsonSchemaValidationError as exc:
            invalid_output = self._terminal(
                intent=persisted_intent,
                started_at=started_at,
                status=InferenceStatus.INVALID_OUTPUT,
                provider_request_id=outcome.provider_request_id,
                latency_ms=outcome.latency_ms,
                usage=self._usage(outcome.usage),
                raw_output=raw_output,
                normalized_output=payload,
                output_valid=False,
                reported_confidence=outcome.reported_confidence,
                failure=self._failure(
                    code="OUTPUT_SCHEMA_INVALID",
                    detail=exc.message,
                    retryability=Retryability.PERMANENT,
                ),
            )
            return self._persist_terminal(
                invalid_output,
                selection_policy_version=policy.selection_policy_version,
            )

        success = self._terminal(
            intent=persisted_intent,
            started_at=started_at,
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=outcome.provider_request_id,
            latency_ms=outcome.latency_ms,
            usage=self._usage(outcome.usage),
            raw_output=raw_output,
            normalized_output=payload,
            output_valid=True,
            reported_confidence=outcome.reported_confidence,
            failure=None,
        )
        return self._persist_terminal(
            success,
            selection_policy_version=policy.selection_policy_version,
        )

    async def apply_rate_limits(
        self,
        *,
        provider: str,
        model_name: str,
        concurrency_limit: int | None = None,
        quota_limit: int | None = None,
    ) -> None:
        """Run the injected execution gate for a selected provider/model."""
        for name, value in (
            ("concurrency_limit", concurrency_limit),
            ("quota_limit", quota_limit),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise OrchestrationConfigurationError(f"{name} must be a positive integer")
        candidates = [
            policy
            for policy in self._policies.values()
            if policy.provider == provider and policy.model_name == model_name
        ]
        if not candidates:
            raise OrchestrationConfigurationError(
                f"rate-limit policy is not configured for {provider}/{model_name}"
            )
        model_versions = {policy.model_version for policy in candidates}
        if len(model_versions) != 1:
            raise OrchestrationConfigurationError(
                f"rate-limit model version is ambiguous for {provider}/{model_name}"
            )
        policy = min(candidates, key=lambda candidate: candidate.timeout_ms)
        adapter = self._adapter(policy)
        capabilities = await adapter.capabilities(policy.model_name, policy.model_version)
        await self._execution_gate.acquire(
            capabilities=capabilities,
            timeout_ms=policy.timeout_ms,
        )

    async def validate_output(
        self,
        *,
        task: VisionTask,
        output_schema_id: str,
        normalized_output: dict[str, object],
    ) -> bool:
        """Validate a normalized payload against the pinned task schema."""
        policy = self._policy(task)
        if policy.output_schema.schema_id != output_schema_id:
            raise OrchestrationConfigurationError("output schema ID does not match the task policy")
        try:
            self._schema(policy.output_schema).validate(normalized_output)
        except JsonSchemaValidationError:
            return False
        return True

    async def persist_attempt(
        self,
        *,
        inference: ModelInference,
        raw_output: dict[str, object] | None = None,
        normalized_output: dict[str, object] | None = None,
        output_valid: bool | None = None,
        usage: ModelInferenceUsage | None = None,
        failure: dict[str, object] | None = None,
    ) -> ModelInference:
        """Append a terminal attempt, retaining any explicitly supplied fields."""
        updates: dict[str, object] = {}
        if raw_output is not None:
            updates["raw_output"] = raw_output
        if normalized_output is not None:
            updates["normalized_output"] = normalized_output
        if output_valid is not None:
            updates["output_valid"] = output_valid
        if usage is not None:
            updates["usage"] = usage
        if failure is not None:
            updates["failure"] = InferenceFailure.model_validate(failure)
        if updates:
            candidate = ModelInference.model_validate(
                inference.model_copy(update=updates).model_dump(mode="python")
            )
        else:
            candidate = inference
        policy = self._policies.get(candidate.stage)
        if policy is None:
            raise OrchestrationConfigurationError(
                f"no selection policy configured for task {candidate.stage}"
            )
        return self._persist_terminal(
            candidate,
            selection_policy_version=policy.selection_policy_version,
        )


__all__ = [
    "CapabilityValidationError",
    "InMemoryInferenceLedger",
    "InferenceExecutionGate",
    "InferenceIntent",
    "InferenceLedger",
    "InferenceLedgerError",
    "InferenceOrchestrationError",
    "InferenceOrchestrator",
    "InferencePolicy",
    "OrchestrationConfigurationError",
]
