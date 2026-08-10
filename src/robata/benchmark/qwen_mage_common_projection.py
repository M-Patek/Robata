"""Common cam_01 five-segment projection helpers for Mage/Qwen qualification.

This module intentionally compares a derived-frame Qwen input with the already accepted
native-video Mage artifacts on the same camera and aligned five non-overlapping intervals.
It does not claim the inputs are byte-identical or that model-to-model agreement is ground
truth.  Both outputs are projected through the same MageObservation and deterministic
QA/event/evidence/track/fusion contracts so throughput and semantic disagreement are
visible without changing a published schema.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from robata.application.canonical.local_real_model import LOCAL_QWEN_MODEL_VERSION
from robata.benchmark.qwen_r12_request_corpus import QwenRequestCorpus
from robata.contracts.cameras import CameraId
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.perception_stream import (
    CognitionGateSignal,
    MageObservation,
    PerceptionContextManifest,
    create_mage_observation,
)
from robata.inference.mage_video_adapter import (
    MageVideoAcceptedObservationBinding,
    MageVideoObservationAdapterError,
    MageVideoObservationPayload,
    _decode_compact_json_object,
    _expand_action_observations,
    _expand_semantic_qa,
    _normalise_compact_numeric_leaves,
    _prepare_compact_payload,
)
from robata.perception.fusion import PerceptionFusionEngine, PerceptionFusionPolicy
from robata.perception.pipeline import StreamPerceptionPipeline, StreamPerceptionRunResult
from robata.perception.projectors import (
    EventProjector,
    EvidenceProjector,
    MediaHealthReport,
    QaProjector,
)
from robata.perception.tracking import EventTrackPolicy, EventTrackReconciler

COMMON_PROJECTION_FIXTURE_VERSION: Final = "qwen-mage-common-cam01-five-segment-v1"
COMMON_FRAME_SELECTION_VERSION: Final = "cam01-six-evenly-spaced-r12-frames-v1"
COMMON_QWEN_ARTIFACT_VERSION: Final = "qwen-common-projection-artifact-v1"
COMMON_QWEN_PROMPT_VERSION: Final = "qwen-mage-common-projection-prompt-v2"
COMMON_QWEN_MODEL_FAMILY: Final = "qwen_video_frames"
COMMON_TRACK_POLICY_VERSION: Final = "mage-stream-track-v1"
COMMON_FUSION_POLICY_VERSION: Final = "mage-stream-fusion-v1"
COMMON_REFINE_POLICY_VERSION: Final = "mage-stream-refine-v1"
COMMON_REFINE_PROMPT_VERSION: Final = "mage-stream-refine-prompt-v1"
COMMON_MEDIA_CAMERA: Final = CameraId.CAM_01
COMMON_SEGMENT_COUNT: Final = 5
COMMON_FRAMES_PER_SEGMENT: Final = 6


class CommonProjectionError(RuntimeError):
    """The frozen common-projection fixture or candidate output is invalid."""


@dataclass(frozen=True, slots=True)
class CommonFrameReference:
    ordinal: int
    aligned_timestamp_ns: int
    path: Path
    sha256: str
    byte_count: int
    width: int
    height: int

    def projection(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "aligned_timestamp_ns": str(self.aligned_timestamp_ns),
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class CommonProjectionCase:
    ordinal: int
    binding: MageVideoAcceptedObservationBinding
    mage_observation: MageObservation
    media_health: MediaHealthReport
    selected_frames: tuple[CommonFrameReference, ...]

    @property
    def context(self) -> PerceptionContextManifest:
        return self.binding.context

    @property
    def duration_seconds(self) -> float:
        interval = self.context.context_interval
        return float((int(interval.end_ns) - int(interval.start_ns)) / 1_000_000_000.0)

    def projection(self) -> dict[str, object]:
        interval = self.context.context_interval
        return {
            "ordinal": self.ordinal,
            "context_manifest_semantic_sha256": self.context.context_manifest_semantic_sha256,
            "start_ns": str(interval.start_ns),
            "end_ns": str(interval.end_ns),
            "duration_seconds": self.duration_seconds,
            "prompt_exact_sha256": exact_bytes_sha256(self.binding.prompt.encode("utf-8")),
            "mage_request_identity_sha256": self.binding.request_identity_sha256,
            "mage_result_artifact_exact_sha256": self.binding.result_artifact_exact_sha256,
            "mage_generation_seconds": self.binding.endpoint_response.generation_seconds,
            "mage_observation_semantic_sha256": self.mage_observation.observation_semantic_sha256,
            "selected_frames": [frame.projection() for frame in self.selected_frames],
        }


@dataclass(frozen=True, slots=True)
class CommonProjectionFixture:
    version: str
    corpus_semantic_sha256: str
    cases: tuple[CommonProjectionCase, ...]
    semantic_sha256: str

    @property
    def duration_seconds(self) -> float:
        return sum(case.duration_seconds for case in self.cases)

    def projection(self) -> dict[str, object]:
        return {
            "version": self.version,
            "frame_selection_version": COMMON_FRAME_SELECTION_VERSION,
            "corpus_semantic_sha256": self.corpus_semantic_sha256,
            "camera_id": COMMON_MEDIA_CAMERA.value,
            "input_equivalence": (
                "SAME_CAMERA_AND_ALIGNED_INTERVALS_DERIVED_MODALITIES_NOT_BYTE_IDENTICAL"
            ),
            "cases": [case.projection() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class CommonQwenProjection:
    observation: MageObservation
    raw_output_text: str
    raw_output_exact_sha256: str
    inference_artifact_exact_sha256: str
    diagnostics: tuple[dict[str, object], ...]


class _UnusedObservationProvider:
    def observe(self, context: PerceptionContextManifest) -> MageObservation:
        del context
        raise RuntimeError("precomputed common projection must not call a provider")


def _file_uri_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme.casefold() != "file" or parsed.netloc not in {"", "localhost"}:
        raise CommonProjectionError(f"unsupported frame URI: {value}")
    decoded = unquote(parsed.path)
    if re.fullmatch(r"/[A-Za-z]:/.*", decoded):
        decoded = decoded[1:]
    path = Path(decoded).resolve()
    if not path.is_file():
        raise CommonProjectionError(f"frame path does not exist: {path}")
    return path


def _all_cam01_frames(corpus: QwenRequestCorpus) -> tuple[CommonFrameReference, ...]:
    if not corpus.cases:
        raise CommonProjectionError("Qwen corpus is empty")
    first = corpus.cases[0].request
    if first.input_plan is None:
        raise CommonProjectionError("frozen r12 first request has no input plan")
    packages = first.input_plan.request_catalog.packages
    if len(packages) != 1:
        raise CommonProjectionError("frozen r12 first request must contain one package")
    cameras = [camera for camera in packages[0].cameras if camera.camera_id is COMMON_MEDIA_CAMERA]
    if len(cameras) != 1:
        raise CommonProjectionError("frozen r12 first request must contain one cam_01 catalog")
    frames: list[CommonFrameReference] = []
    for frame in cameras[0].frames:
        path = _file_uri_path(frame.source_artifact_uri)
        payload = path.read_bytes()
        if len(payload) != frame.source_artifact_bytes:
            raise CommonProjectionError(f"frame byte count changed: {path}")
        if exact_bytes_sha256(payload) != frame.source_artifact_sha256:
            raise CommonProjectionError(f"frame digest changed: {path}")
        frames.append(
            CommonFrameReference(
                ordinal=frame.ordinal,
                aligned_timestamp_ns=frame.aligned_timestamp_ns,
                path=path,
                sha256=frame.source_artifact_sha256,
                byte_count=frame.source_artifact_bytes,
                width=frame.width,
                height=frame.height,
            )
        )
    ordered = tuple(sorted(frames, key=lambda item: (item.aligned_timestamp_ns, item.ordinal)))
    if len(ordered) != 41:
        raise CommonProjectionError("frozen r12 cam_01 catalog must contain exactly 41 frames")
    if len({item.aligned_timestamp_ns for item in ordered}) != len(ordered):
        raise CommonProjectionError("cam_01 aligned frame timestamps must be unique")
    return ordered


def select_evenly_spaced_frames(
    frames: Sequence[CommonFrameReference],
    *,
    start_ns: int,
    end_ns: int,
    count: int = COMMON_FRAMES_PER_SEGMENT,
    include_end: bool = False,
) -> tuple[CommonFrameReference, ...]:
    if start_ns < 0 or end_ns <= start_ns:
        raise CommonProjectionError("segment interval must be positive and ordered")
    if count <= 0:
        raise CommonProjectionError("selected frame count must be positive")
    candidates = tuple(
        frame
        for frame in frames
        if start_ns <= frame.aligned_timestamp_ns
        and (
            frame.aligned_timestamp_ns <= end_ns
            if include_end
            else frame.aligned_timestamp_ns < end_ns
        )
    )
    if len(candidates) < count:
        raise CommonProjectionError(
            f"segment contains {len(candidates)} frames but {count} are required"
        )
    if count == 1:
        return (candidates[len(candidates) // 2],)
    denominator = count - 1
    indices = tuple(
        (index * (len(candidates) - 1) + denominator // 2) // denominator for index in range(count)
    )
    selected = tuple(candidates[index] for index in indices)
    if len({item.sha256 for item in selected}) != count:
        raise CommonProjectionError("even frame selection produced duplicate content")
    return selected


def _load_strict_documents(root: Path, kind: str, model: type[Any]) -> tuple[Any, ...]:
    directory = root / kind
    paths = tuple(sorted(directory.rglob("*.json"))) if directory.is_dir() else ()
    if not paths:
        raise CommonProjectionError(f"missing frozen Mage {kind} artifacts: {directory}")
    documents: list[Any] = []
    for path in paths:
        try:
            documents.append(model.model_validate_json(path.read_bytes(), strict=True))
        except (OSError, TypeError, ValueError, ValidationError) as error:
            raise CommonProjectionError(f"invalid frozen Mage {kind} artifact: {path}") from error
    return tuple(documents)


def load_common_projection_fixture(
    *,
    corpus: QwenRequestCorpus,
    mage_stream_artifact_root: Path,
) -> CommonProjectionFixture:
    root = mage_stream_artifact_root.expanduser().resolve()
    bindings = _load_strict_documents(
        root, "accepted-inference-binding", MageVideoAcceptedObservationBinding
    )
    observations = _load_strict_documents(root, "observation", MageObservation)
    health_reports = _load_strict_documents(root, "media-health", MediaHealthReport)
    binding_by_context = {item.context.context_manifest_semantic_sha256: item for item in bindings}
    observation_by_context = {
        item.context.context_manifest_semantic_sha256: item for item in observations
    }
    health_by_context = {item.context_manifest_semantic_sha256: item for item in health_reports}
    if not (set(binding_by_context) == set(observation_by_context) == set(health_by_context)):
        raise CommonProjectionError("Mage binding/observation/media-health context sets differ")
    if len(binding_by_context) != COMMON_SEGMENT_COUNT:
        raise CommonProjectionError("common fixture requires exactly five Mage contexts")

    frames = _all_cam01_frames(corpus)
    cases: list[CommonProjectionCase] = []
    ordered_bindings = sorted(bindings, key=lambda item: item.context.focus_segment_ordinal)
    if [item.context.focus_segment_ordinal for item in ordered_bindings] != list(
        range(COMMON_SEGMENT_COUNT)
    ):
        raise CommonProjectionError("Mage focus segment ordinals must be exactly 0..4")
    for binding in ordered_bindings:
        context = binding.context
        context_digest = context.context_manifest_semantic_sha256
        selected_cameras = tuple(
            camera_id
            for camera_id, camera in context.cameras.items()
            if camera.selected_for_inference
        )
        if selected_cameras != (COMMON_MEDIA_CAMERA,):
            raise CommonProjectionError("common fixture must select only cam_01")
        interval = context.context_interval
        selected = select_evenly_spaced_frames(
            frames,
            start_ns=interval.start_ns,
            end_ns=interval.end_ns,
            include_end=binding.context.focus_segment_ordinal == COMMON_SEGMENT_COUNT - 1,
        )
        observation = observation_by_context[context_digest]
        if observation.context != context:
            raise CommonProjectionError("Mage observation context differs from accepted binding")
        cases.append(
            CommonProjectionCase(
                ordinal=context.focus_segment_ordinal,
                binding=binding,
                mage_observation=observation,
                media_health=health_by_context[context_digest],
                selected_frames=selected,
            )
        )

    draft = CommonProjectionFixture(
        version=COMMON_PROJECTION_FIXTURE_VERSION,
        corpus_semantic_sha256=corpus.semantic_sha256,
        cases=tuple(cases),
        semantic_sha256="0" * 64,
    )
    digest = exact_bytes_sha256(canonical_json_bytes(draft.projection()))
    return CommonProjectionFixture(
        version=draft.version,
        corpus_semantic_sha256=draft.corpus_semantic_sha256,
        cases=draft.cases,
        semantic_sha256=digest,
    )


def load_selected_frame_payloads(case: CommonProjectionCase) -> tuple[bytes, ...]:
    payloads: list[bytes] = []
    for frame in case.selected_frames:
        payload = frame.path.read_bytes()
        if len(payload) != frame.byte_count or exact_bytes_sha256(payload) != frame.sha256:
            raise CommonProjectionError(f"selected frame changed after fixture load: {frame.path}")
        payloads.append(payload)
    return tuple(payloads)


def build_qwen_common_prompt(case: CommonProjectionCase) -> str:
    """Render a Qwen-specific grammar with the same compact observation semantics.

    The frozen Mage prompt remains referenced by exact digest, but is not reused as
    executable Qwen syntax: the first qualification showed that Qwen interpreted the
    nested schema description as an output array.  This v2 prompt makes the JSON root
    and forbidden schema-metadata keys explicit without weakening strict parsing.
    """

    interval = case.context.context_interval
    duration_seconds = case.duration_seconds
    frame_offsets = [
        (frame.aligned_timestamp_ns - int(interval.start_ns)) / 1_000_000_000.0
        for frame in case.selected_frames
    ]
    document = {
        "protocol": COMMON_QWEN_PROMPT_VERSION,
        "semantic_contract_source": {
            "mage_prompt_exact_sha256": exact_bytes_sha256(case.binding.prompt.encode("utf-8")),
            "observation_schema_version": case.mage_observation.observation_schema_version,
        },
        "input": {
            "camera_id": COMMON_MEDIA_CAMERA.value,
            "modality": "six ordered still frames sampled from one continuous segment",
            "segment_duration_seconds": duration_seconds,
            "frame_offset_seconds": frame_offsets,
        },
        "task": (
            "Inspect the ordered frames as one temporal segment. Report visible physical "
            "actions and the selected camera quality."
        ),
        "response_contract": {
            "json_root": "object",
            "exact_root_keys": ["selected_camera_qa", "observations"],
            "selected_camera_qa": {
                "disposition": "USABLE|DEGRADED|UNUSABLE|UNKNOWN",
                "issues": "array of {code, detail}; use [] when none",
                "confidence": "optional number from 0 to 1",
            },
            "observations": {
                "type": "array",
                "item_exact_keys": ["action", "interval", "confidence", "visibility"],
                "action": "short observed physical-action phrase",
                "interval": {
                    "start_offset_seconds": (
                        f"number from 0 inclusive to {duration_seconds} exclusive"
                    ),
                    "end_offset_seconds": (
                        f"number after start and no greater than {duration_seconds}"
                    ),
                },
                "confidence": "optional number from 0 to 1",
                "visibility": "optional number from 0 to 1",
            },
        },
        "rules": [
            "Return exactly one JSON object and no markdown or prose.",
            "The root must contain selected_camera_qa and observations.",
            "Do not output schema metadata keys such as item_keys, type, or root_keys.",
            "Use seconds relative to the segment start, not frame ordinals.",
            "Do not invent an action that is not visible in the supplied frames.",
            "Return an empty observations array when no physical action is visible.",
        ],
        "example_shape": {
            "selected_camera_qa": {"disposition": "USABLE", "issues": []},
            "observations": [
                {
                    "action": "fold green shirt",
                    "interval": {
                        "start_offset_seconds": 1.0,
                        "end_offset_seconds": 3.0,
                    },
                    "confidence": 0.9,
                    "visibility": 0.9,
                }
            ],
        },
    }
    return str(canonical_json_bytes(document).decode("utf-8"))


def qwen_projection_artifact_sha256(
    *,
    case: CommonProjectionCase,
    checkpoint_manifest_sha256: str,
    output_text: str,
) -> str:
    return str(
        exact_bytes_sha256(
            canonical_json_bytes(
                {
                    "artifact_version": COMMON_QWEN_ARTIFACT_VERSION,
                    "model_family": COMMON_QWEN_MODEL_FAMILY,
                    "model_revision": LOCAL_QWEN_MODEL_VERSION,
                    "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                    "context_manifest_semantic_sha256": (
                        case.context.context_manifest_semantic_sha256
                    ),
                    "prompt_version": COMMON_QWEN_PROMPT_VERSION,
                    "prompt_exact_sha256": exact_bytes_sha256(
                        build_qwen_common_prompt(case).encode("utf-8")
                    ),
                    "source_mage_prompt_exact_sha256": exact_bytes_sha256(
                        case.binding.prompt.encode("utf-8")
                    ),
                    "frame_selection_version": COMMON_FRAME_SELECTION_VERSION,
                    "selected_frame_sha256_values": [
                        frame.sha256 for frame in case.selected_frames
                    ],
                    "output_text": output_text,
                }
            )
        )
    )


def project_qwen_compact_output(
    *,
    case: CommonProjectionCase,
    checkpoint_manifest_sha256: str,
    output_text: str,
    created_at: str | None = None,
) -> CommonQwenProjection:
    if not isinstance(output_text, str) or not output_text.strip():
        raise CommonProjectionError("Qwen common-projection output must be nonempty")
    artifact_sha256 = qwen_projection_artifact_sha256(
        case=case,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        output_text=output_text,
    )
    try:
        raw_payload = _decode_compact_json_object(output_text)
        compact_payload = _prepare_compact_payload(
            raw_payload,
            observation_schema_version=case.mage_observation.observation_schema_version,
        )
        normalized_payload = _normalise_compact_numeric_leaves(compact_payload)
        payload = MageVideoObservationPayload.model_validate_json(
            canonical_json_bytes(normalized_payload), strict=True
        )
        semantic_qa = _expand_semantic_qa(
            selected_camera=COMMON_MEDIA_CAMERA,
            selected_qa=payload.selected_camera_qa,
        )
        expanded = _expand_action_observations(
            context=case.context,
            selected_camera=COMMON_MEDIA_CAMERA,
            payloads=payload.observations,
            out_of_context_action_policy="REJECT_ACTION_V1",
            inference_artifact_exact_sha256=artifact_sha256,
        )
    except (MageVideoObservationAdapterError, TypeError, ValueError, ValidationError) as error:
        raise CommonProjectionError(
            f"Qwen output for common case {case.ordinal} is not a strict compact observation"
        ) from error
    observation = create_mage_observation(
        observation_schema_version=payload.observation_schema_version,
        context=case.context,
        model_family=COMMON_QWEN_MODEL_FAMILY,
        model_revision=LOCAL_QWEN_MODEL_VERSION,
        model_artifact_manifest_sha256=checkpoint_manifest_sha256,
        prompt_version=COMMON_QWEN_PROMPT_VERSION,
        inference_artifact_exact_sha256=artifact_sha256,
        cognition_gate=CognitionGateSignal(
            score=None,
            threshold=0.5,
            would_admit=None,
            gate_policy_version="qwen-common-gate-shadow-v1",
        ),
        semantic_qa=semantic_qa,
        observations=expanded.actions,
        created_at=created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return CommonQwenProjection(
        observation=observation,
        raw_output_text=output_text,
        raw_output_exact_sha256=exact_bytes_sha256(output_text.encode("utf-8")),
        inference_artifact_exact_sha256=artifact_sha256,
        diagnostics=tuple(item.model_dump(mode="json") for item in expanded.diagnostics),
    )


def run_common_downstream(
    *,
    cases: Sequence[CommonProjectionCase],
    observations: Sequence[MageObservation],
    observation_elapsed_seconds: Sequence[float],
) -> StreamPerceptionRunResult:
    ordered_cases = tuple(cases)
    ordered_observations = tuple(observations)
    ordered_elapsed = tuple(float(item) for item in observation_elapsed_seconds)
    if not ordered_cases or len(ordered_cases) != len(ordered_observations):
        raise CommonProjectionError("common downstream case/observation counts differ")
    if len(ordered_elapsed) != len(ordered_cases) or any(item < 0 for item in ordered_elapsed):
        raise CommonProjectionError("common downstream elapsed values are invalid")
    pipeline = StreamPerceptionPipeline(
        provider=_UnusedObservationProvider(),
        qa_projector=QaProjector(),
        event_projector=EventProjector(),
        evidence_projector=EvidenceProjector(),
        reconciler=EventTrackReconciler(EventTrackPolicy(version=COMMON_TRACK_POLICY_VERSION)),
        fusion_engine=PerceptionFusionEngine(
            PerceptionFusionPolicy(version=COMMON_FUSION_POLICY_VERSION)
        ),
        refine_policy_version=COMMON_REFINE_POLICY_VERSION,
        refine_prompt_version=COMMON_REFINE_PROMPT_VERSION,
    )
    session = pipeline.open_session()
    for case, observation, elapsed in zip(
        ordered_cases, ordered_observations, ordered_elapsed, strict=True
    ):
        session.consume_precomputed(
            context=case.context,
            media_health=case.media_health,
            observation=observation,
            observation_elapsed_seconds=elapsed,
        )
    return session.finalize()


def downstream_projection(result: StreamPerceptionRunResult) -> dict[str, object]:
    return {
        "context_count": len(result.contexts),
        "normal_model_call_count": result.normal_model_call_count,
        "refinement_model_call_count": result.refinement_model_call_count,
        "event_tracks": [
            {
                "event_track_key": track.event_track_key,
                "state": track.state.value,
                "action": track.action,
                "start_ns": str(track.interval.start_ns),
                "end_ns": str(track.interval.end_ns),
                "source_hypothesis_count": len(track.source_hypotheses),
                "semantic_sha256": track.revision_semantic_sha256,
            }
            for track in result.event_tracks
        ],
        "fusion_decisions": [
            {
                "fusion_key": decision.fusion_key,
                "action": decision.action,
                "start_ns": str(decision.interval.start_ns),
                "end_ns": str(decision.interval.end_ns),
                "confidence": decision.confidence,
                "ambiguity_reasons": [item.value for item in decision.ambiguity_reasons],
                "requires_refinement": decision.requires_refinement,
                "semantic_sha256": decision.fusion_semantic_sha256,
            }
            for decision in result.fusion_decisions
        ],
        "refine_request_count": len(result.refine_requests),
        "stage_measurements": [
            {
                "stage": measurement.stage.value,
                "invocation_count": measurement.invocation_count,
                "elapsed_seconds": measurement.elapsed_seconds,
            }
            for measurement in result.stage_measurements
        ],
    }


_STOPWORDS: Final = frozenset({"a", "an", "the", "person", "is", "are", "was", "were"})


def action_tokens(action: str) -> frozenset[str]:
    values = {
        token
        for token in re.findall(r"[a-z0-9]+", action.casefold().replace("_", " "))
        if token not in _STOPWORDS
    }
    return frozenset(values)


def action_token_f1(left: str, right: str) -> float:
    left_tokens = action_tokens(left)
    right_tokens = action_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def interval_iou(left_start: int, left_end: int, right_start: int, right_end: int) -> float:
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return 0.0 if union <= 0 else intersection / union


def compare_observations(
    *,
    mage: Sequence[MageObservation],
    qwen: Sequence[MageObservation],
) -> dict[str, object]:
    mage_actions = [item for observation in mage for item in observation.observations]
    qwen_actions = [item for observation in qwen for item in observation.observations]
    remaining = set(range(len(qwen_actions)))
    matches: list[dict[str, object]] = []
    for mage_index, expected in enumerate(mage_actions):
        candidates: list[tuple[float, float, float, int]] = []
        for qwen_index in sorted(remaining):
            actual = qwen_actions[qwen_index]
            label_f1 = action_token_f1(expected.action, actual.action)
            temporal_iou = interval_iou(
                expected.interval.start_ns,
                expected.interval.end_ns,
                actual.interval.start_ns,
                actual.interval.end_ns,
            )
            candidates.append(((label_f1 + temporal_iou) / 2.0, label_f1, temporal_iou, qwen_index))
        if not candidates:
            matches.append(
                {
                    "mage_index": mage_index,
                    "qwen_index": None,
                    "mage_action": expected.action,
                    "qwen_action": None,
                    "label_token_f1": 0.0,
                    "temporal_iou": 0.0,
                }
            )
            continue
        _, label_f1, temporal_iou, qwen_index = max(
            candidates, key=lambda item: (item[0], item[1], item[2], -item[3])
        )
        remaining.remove(qwen_index)
        actual = qwen_actions[qwen_index]
        matches.append(
            {
                "mage_index": mage_index,
                "qwen_index": qwen_index,
                "mage_action": expected.action,
                "qwen_action": actual.action,
                "label_token_f1": label_f1,
                "temporal_iou": temporal_iou,
            }
        )
    label_values = [cast(float, item["label_token_f1"]) for item in matches]
    temporal_values = [cast(float, item["temporal_iou"]) for item in matches]
    return {
        "authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
        "is_ground_truth_accuracy": False,
        "mage_action_count": len(mage_actions),
        "qwen_action_count": len(qwen_actions),
        "matched_mage_action_count": sum(item["qwen_index"] is not None for item in matches),
        "unmatched_qwen_action_count": len(remaining),
        "mean_label_token_f1": sum(label_values) / len(label_values) if label_values else None,
        "mean_temporal_iou": sum(temporal_values) / len(temporal_values)
        if temporal_values
        else None,
        "matches": matches,
        "unmatched_qwen_actions": [qwen_actions[index].action for index in sorted(remaining)],
    }


__all__ = [
    "COMMON_FRAMES_PER_SEGMENT",
    "COMMON_FRAME_SELECTION_VERSION",
    "COMMON_PROJECTION_FIXTURE_VERSION",
    "COMMON_QWEN_ARTIFACT_VERSION",
    "COMMON_QWEN_MODEL_FAMILY",
    "COMMON_QWEN_PROMPT_VERSION",
    "CommonFrameReference",
    "CommonProjectionCase",
    "CommonProjectionError",
    "CommonProjectionFixture",
    "CommonQwenProjection",
    "action_token_f1",
    "action_tokens",
    "build_qwen_common_prompt",
    "compare_observations",
    "downstream_projection",
    "interval_iou",
    "load_common_projection_fixture",
    "load_selected_frame_payloads",
    "project_qwen_compact_output",
    "qwen_projection_artifact_sha256",
    "run_common_downstream",
    "select_evenly_spaced_frames",
]
