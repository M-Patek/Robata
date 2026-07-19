"""Optional local vision-model adapter boundaries.

The production model is intentionally not bundled with Robata.  This module provides a
provider-neutral adapter that can wrap a locally loaded runner (including a small
``transformers`` model) without importing model libraries at module import time or making
network requests.  The runner is responsible for translating model output into the validated
``VisionInferenceOutcome`` contract.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

from robata.contracts.mainline import (
    TemporalVisualPackage,
    VisionInferenceFailure,
    VisionInferenceOutcome,
    VisionInferenceRequest,
    VisionInferenceSuccess,
)


class LocalVisionModelAdapterError(RuntimeError):
    """Raised when a local adapter cannot satisfy the inference boundary."""


class OptionalDependencyUnavailable(LocalVisionModelAdapterError):
    """Raised when an optional local model dependency is not installed."""


class VisionRunner(Protocol):
    """Callable runner expected by :class:`LocalVisionModelAdapter`."""

    def __call__(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage,
        artifact_root: Path,
    ) -> VisionInferenceOutcome:
        ...


class LocalVisionModelAdapter:
    """Validate and expose a locally loaded vision runner as ``VisionModelAdapter``.

    The adapter deliberately accepts an already-loaded runner rather than downloading a model.
    ``external_provider_requests`` is always zero and is useful in acceptance reports.
    """

    external_provider_requests = 0

    def __init__(
        self,
        runner: VisionRunner,
        *,
        provider: str = "local",
        model_name: str = "local-vision-model",
        model_version: str = "unversioned",
        supports_parallel_inference: bool = False,
    ) -> None:
        if not callable(runner):
            raise TypeError("runner must be callable")
        for field, value in (
            ("provider", provider),
            ("model_name", model_name),
            ("model_version", model_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if not isinstance(supports_parallel_inference, bool):
            raise TypeError("supports_parallel_inference must be a bool")
        self._runner = runner
        self._provider = provider.strip()
        self._model_name = model_name.strip()
        self._model_version = model_version.strip()
        self._supports_parallel_inference = supports_parallel_inference

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def supports_parallel_inference(self) -> bool:
        return self._supports_parallel_inference

    def infer(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage,
        artifact_root: Path,
    ) -> VisionInferenceOutcome:
        if not isinstance(request, VisionInferenceRequest):
            raise TypeError("request must be VisionInferenceRequest")
        if not isinstance(package, TemporalVisualPackage):
            raise TypeError("package must be TemporalVisualPackage")
        if not isinstance(artifact_root, Path) or not artifact_root.is_dir():
            raise LocalVisionModelAdapterError("artifact_root must be an existing directory")
        if request.provider != self.provider:
            raise LocalVisionModelAdapterError(
                f"request provider {request.provider!r} does not match adapter {self.provider!r}"
            )
        if request.model_name != self.model_name:
            raise LocalVisionModelAdapterError(
                f"request model_name {request.model_name!r} does not match "
                f"adapter {self.model_name!r}"
            )
        if request.model_version != self.model_version:
            raise LocalVisionModelAdapterError(
                f"request model_version {request.model_version!r} does not match "
                f"adapter {self.model_version!r}"
            )
        if (
            request.package_id != package.package_id
            or request.package_content_sha256 != package.content_sha256
            or request.mcap_id != package.mcap_id
            or request.interval != package.interval
        ):
            raise LocalVisionModelAdapterError("request package identity does not match package")
        outcome = self._runner(request, package, artifact_root)
        if not isinstance(outcome, (VisionInferenceSuccess, VisionInferenceFailure)):
            raise LocalVisionModelAdapterError(
                "runner must return a validated VisionInferenceSuccess or VisionInferenceFailure"
            )
        self._validate_outcome_identity(request, outcome)
        return outcome

    @staticmethod
    def _validate_outcome_identity(
        request: VisionInferenceRequest,
        outcome: VisionInferenceSuccess | VisionInferenceFailure,
    ) -> None:
        if outcome.inference_id != request.inference_id:
            raise LocalVisionModelAdapterError("outcome inference_id does not match request")
        if outcome.request_id != request.request_id:
            raise LocalVisionModelAdapterError("outcome request_id does not match request")
        if outcome.task is not request.task:
            raise LocalVisionModelAdapterError("outcome task does not match request")
        if outcome.provider != request.provider:
            raise LocalVisionModelAdapterError("outcome provider does not match request")
        if outcome.model_name != request.model_name:
            raise LocalVisionModelAdapterError("outcome model_name does not match request")
        if outcome.model_version != request.model_version:
            raise LocalVisionModelAdapterError("outcome model_version does not match request")


class TransformersVisionModelAdapter(LocalVisionModelAdapter):
    """Lazy optional adapter for an already-local ``transformers`` checkpoint.

    A custom ``runner`` is required for inference because Robata's output contract is richer than
    a generic text-generation result.  ``load_local`` loads processor/model only from a local
    path with ``local_files_only=True`` and never contacts a model hub.
    """

    def __init__(
        self,
        runner: VisionRunner,
        *,
        processor: Any = None,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(runner, **kwargs)
        self.processor = processor
        self.model = model

    @classmethod
    def load_local(
        cls,
        model_path: str | Path,
        *,
        runner: VisionRunner,
        provider: str = "transformers-local",
        model_name: str | None = None,
        model_version: str = "local",
        supports_parallel_inference: bool = False,
    ) -> TransformersVisionModelAdapter:
        path = Path(model_path)
        if not path.exists() or not path.is_dir():
            raise LocalVisionModelAdapterError(f"local model path does not exist: {path}")
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise OptionalDependencyUnavailable(
                "transformers is not installed; install the optional local-model extra"
            ) from error
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                str(path), local_files_only=True
            )
            model = transformers.AutoModel.from_pretrained(str(path), local_files_only=True)
        except Exception as error:
            raise LocalVisionModelAdapterError(
                f"could not load local transformers checkpoint without network access: {error}"
            ) from error
        resolved_name = model_name or path.name
        return cls(
            runner,
            processor=processor,
            model=model,
            provider=provider,
            model_name=resolved_name,
            model_version=model_version,
            supports_parallel_inference=supports_parallel_inference,
        )

    @classmethod
    def require_transformers(cls) -> Any:
        """Import the optional dependency on demand for diagnostics/tests."""

        try:
            return importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise OptionalDependencyUnavailable(
                "transformers is not installed; no model download was attempted"
            ) from error


__all__ = [
    "LocalVisionModelAdapter",
    "LocalVisionModelAdapterError",
    "OptionalDependencyUnavailable",
    "TransformersVisionModelAdapter",
    "VisionRunner",
]
