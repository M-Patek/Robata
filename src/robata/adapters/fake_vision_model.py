"""Deterministic, network-free vision-model adapter for the local mainline."""

from __future__ import annotations

from pathlib import Path

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.mainline import (
    ActionEvidence,
    BoundaryCameraClaim,
    BoundaryRefinement,
    BoundaryStatus,
    CameraActionClaim,
    CameraEvidenceStatus,
    CameraQAClaim,
    CameraQAStatus,
    CrossViewHypothesis,
    EventProposal,
    EventProposalOutput,
    FusionAdjudicationOutput,
    FusionHypothesisClaim,
    InferenceFailureDetail,
    InferenceStatus,
    ProposalCameraClaim,
    QAOutput,
    Retryability,
    TemporalVisualPackage,
    VisionInferenceFailure,
    VisionInferenceOutcome,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionProviderOutput,
    VisionTask,
    VisionUsage,
)

FAKE_PROVIDER = "fake"
FAKE_MODEL_NAME = "deterministic-mainline"
FAKE_MODEL_VERSION = "1.0"
_ACTION_TYPE = "object_interaction"


class DeterministicFakeVisionModelAdapter:
    """Return strict provider claims without making an external request.

    The adapter deliberately emits only provider-local ordinals. Authoritative package,
    candidate, frame, inference, and event identifiers remain owned by the orchestrator.
    """

    provider = FAKE_PROVIDER
    model_name = FAKE_MODEL_NAME
    model_version = FAKE_MODEL_VERSION

    def __init__(self, *, no_event: bool = False) -> None:
        if type(no_event) is not bool:
            raise TypeError("no_event must be a bool")
        self._no_event = no_event

    @property
    def external_provider_requests(self) -> int:
        """External provider traffic is impossible for this in-process adapter."""

        return 0

    def infer(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage | None = None,
        artifact_root: Path | None = None,
    ) -> VisionInferenceOutcome:
        usage = _usage(request)
        identity_error = self._identity_error(request)
        if identity_error is not None:
            return self._failure(request, usage, "FAKE_MODEL_IDENTITY_MISMATCH", identity_error)
        media_error = _media_error(request, package, artifact_root)
        if media_error is not None:
            return self._failure(request, usage, "FAKE_MEDIA_CONTEXT_MISMATCH", media_error)

        empty_cameras = tuple(
            camera_id.value
            for camera_id in CAMERA_IDS
            if not request.camera_inputs[camera_id].frame_ids
        )
        if empty_cameras:
            detail = "camera inputs contain no frames: " + ", ".join(empty_cameras)
            return self._failure(request, usage, "FAKE_INPUT_MISSING_FRAMES", detail)

        output = self._output(request)
        return VisionInferenceSuccess(
            schema_version="1.0",
            inference_id=request.inference_id,
            request_id=request.request_id,
            task=request.task,
            status=InferenceStatus.SUCCEEDED,
            provider=self.provider,
            model_name=self.model_name,
            model_version=self.model_version,
            output=output,
            raw_output_sha256=exact_bytes_sha256(canonical_json_bytes(output)),
            schema_valid=True,
            usage=usage,
            latency_ms=0,
        )

    def _identity_error(self, request: VisionInferenceRequest) -> str | None:
        expected = (self.provider, self.model_name, self.model_version)
        actual = (request.provider, request.model_name, request.model_version)
        if actual == expected:
            return None
        return (
            "request selected a different provider/model identity; "
            f"expected {expected!r}, received {actual!r}"
        )

    def _failure(
        self,
        request: VisionInferenceRequest,
        usage: VisionUsage,
        code: str,
        detail: str,
    ) -> VisionInferenceFailure:
        return VisionInferenceFailure(
            schema_version="1.0",
            inference_id=request.inference_id,
            request_id=request.request_id,
            task=request.task,
            status=InferenceStatus.FAILED,
            provider=self.provider,
            model_name=self.model_name,
            model_version=self.model_version,
            output=None,
            raw_output_sha256=None,
            schema_valid=False,
            usage=usage,
            latency_ms=0,
            failure=InferenceFailureDetail(
                code=code,
                detail=detail,
                retryability=Retryability.PERMANENT,
            ),
        )

    def _output(self, request: VisionInferenceRequest) -> VisionProviderOutput:
        if request.task in {VisionTask.QA_COARSE, VisionTask.QA_DENSE}:
            return _qa_output(request)
        if request.task is VisionTask.EVENT_PROPOSAL:
            return _proposal_output(request, no_event=self._no_event)
        if request.task is VisionTask.ACTION_EVIDENCE:
            return _action_output(request, no_event=self._no_event)
        if request.task is VisionTask.BOUNDARY_REFINEMENT:
            return _boundary_output(request, no_event=self._no_event)
        if request.task is VisionTask.FUSION_ADJUDICATION:
            return _fusion_output(request, no_event=self._no_event)
        raise AssertionError(f"unsupported vision task: {request.task}")


def _usage(request: VisionInferenceRequest) -> VisionUsage:
    input_frames = sum(len(item.frame_ids) for item in request.camera_inputs.values())
    return VisionUsage(
        input_frames=input_frames,
        input_images=input_frames,
        input_tokens=None,
        output_tokens=None,
        cost=None,
        currency=None,
    )


def _ordinals(request: VisionInferenceRequest, camera_id: CameraId) -> tuple[int, ...]:
    return tuple(range(len(request.camera_inputs[camera_id].frame_ids)))


def _centered_event(scope: NanosecondInterval) -> NanosecondInterval:
    inset = scope.duration_ns // 4
    start_ns = scope.start_ns + inset
    end_ns = scope.end_ns - inset
    if start_ns >= end_ns:
        return scope
    return NanosecondInterval(start_ns=start_ns, end_ns=end_ns)


def _qa_output(request: VisionInferenceRequest) -> QAOutput:
    return QAOutput(
        cameras=SixCameraMap[CameraQAClaim](
            {
                camera_id: CameraQAClaim(
                    camera_id=camera_id,
                    observed_interval=request.interval,
                    status=CameraQAStatus.GOOD,
                    issues=(),
                    reported_score=0.99,
                    frame_ordinals=_ordinals(request, camera_id),
                )
                for camera_id in CAMERA_IDS
            }
        )
    )


def _proposal_output(
    request: VisionInferenceRequest,
    *,
    no_event: bool,
) -> EventProposalOutput:
    if no_event:
        return EventProposalOutput(proposals=())
    return EventProposalOutput(
        proposals=(
            EventProposal(
                ordinal=0,
                interval=_centered_event(request.interval),
                label_hint=_ACTION_TYPE,
                reported_score=0.95,
                cameras=SixCameraMap[ProposalCameraClaim](
                    {
                        camera_id: ProposalCameraClaim(
                            camera_id=camera_id,
                            status=CameraEvidenceStatus.SUPPORTING,
                            frame_ordinals=_ordinals(request, camera_id),
                        )
                        for camera_id in CAMERA_IDS
                    }
                ),
            ),
        )
    )


def _action_output(
    request: VisionInferenceRequest,
    *,
    no_event: bool,
) -> ActionEvidence:
    event_interval = None if no_event else _centered_event(request.interval)
    status = CameraEvidenceStatus.NO_EVENT if no_event else CameraEvidenceStatus.SUPPORTING
    cameras = SixCameraMap[CameraActionClaim](
        {
            camera_id: CameraActionClaim(
                camera_id=camera_id,
                status=status,
                event_interval=event_interval,
                observed_interval=request.interval,
                visibility=1.0,
                observed_frame_count=len(request.camera_inputs[camera_id].frame_ids),
                coverage_fraction=1.0,
                reported_score=0.97,
                frame_ordinals=_ordinals(request, camera_id),
                reason=None,
            )
            for camera_id in CAMERA_IDS
        }
    )
    hypotheses = (
        ()
        if no_event
        else (
            CrossViewHypothesis(
                ordinal=0,
                interval=_centered_event(request.interval),
                action_type=_ACTION_TYPE,
                reported_score=0.97,
            ),
        )
    )
    return ActionEvidence(cameras=cameras, cross_view_hypotheses=hypotheses)


def _boundary_output(
    request: VisionInferenceRequest,
    *,
    no_event: bool,
) -> BoundaryRefinement:
    event = _centered_event(request.interval)
    width_ns = max(1, event.duration_ns // 4)
    onset = NanosecondInterval(
        start_ns=event.start_ns,
        end_ns=min(event.end_ns, event.start_ns + width_ns),
    )
    offset = NanosecondInterval(
        start_ns=max(event.start_ns, event.end_ns - width_ns),
        end_ns=event.end_ns,
    )
    return BoundaryRefinement(
        cameras=SixCameraMap[BoundaryCameraClaim](
            {
                camera_id: BoundaryCameraClaim(
                    camera_id=camera_id,
                    status=(BoundaryStatus.NO_BOUNDARY if no_event else BoundaryStatus.OBSERVED),
                    observed_interval=request.interval,
                    onset_interval=None if no_event else onset,
                    offset_interval=None if no_event else offset,
                    reported_score=0.96,
                    frame_ordinals=_ordinals(request, camera_id),
                    reason="deterministic no-event mode" if no_event else None,
                )
                for camera_id in CAMERA_IDS
            }
        )
    )


def _fusion_output(
    request: VisionInferenceRequest,
    *,
    no_event: bool,
) -> FusionAdjudicationOutput:
    if no_event:
        return FusionAdjudicationOutput(hypotheses=(), abstained=True)
    return FusionAdjudicationOutput(
        hypotheses=(
            FusionHypothesisClaim(
                ordinal=0,
                interval=_centered_event(request.interval),
                action_type=_ACTION_TYPE,
                conflict_codes=(),
                reported_score=0.97,
            ),
        ),
        abstained=False,
    )


FakeVisionModelAdapter = DeterministicFakeVisionModelAdapter


def _media_error(
    request: VisionInferenceRequest,
    package: TemporalVisualPackage | None,
    artifact_root: Path | None,
) -> str | None:
    if package is None and artifact_root is None:
        return None
    if package is None or artifact_root is None:
        return "package and artifact_root must be supplied together"
    if not isinstance(package, TemporalVisualPackage) or not isinstance(artifact_root, Path):
        return "media context has the wrong type"
    if (
        request.package_id != package.package_id
        or request.package_content_sha256 != package.content_sha256
        or request.mcap_id != package.mcap_id
        or request.interval != package.interval
    ):
        return "request package identity does not match the supplied package"
    if not artifact_root.is_dir():
        return "artifact_root is not an existing directory"
    return None


__all__ = [
    "FAKE_MODEL_NAME",
    "FAKE_MODEL_VERSION",
    "FAKE_PROVIDER",
    "DeterministicFakeVisionModelAdapter",
    "FakeVisionModelAdapter",
]
