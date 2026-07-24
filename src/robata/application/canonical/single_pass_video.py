"""Durable single-pass H.264 spooling and staged six-camera export.

The producer owns one narrow recovery boundary.  A fresh run traverses MCAP once,
publishes an immutable six-spool set at EOS, and then remuxes those spools in a
bounded thread pool.  A resumed run validates the seal, rebuilds the bounded
planner by merging the six spools on their original traversal index, and never
opens the MCAP source.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from robata.adapters.mcap_inspector import McapPreflight
from robata.adapters.mcap_single_pass import (
    PLANNING_MODE_EOS_SPOOL_REPLAY,
    PLANNING_MODE_LIVE_BOOTSTRAP,
    PLANNING_MODE_LIVE_INDEXED,
    AppendOnlyH264SpoolBranch,
    H264PacketEnvelope,
    H264SpoolFacts,
    McapSinglePassH264Tee,
    SinglePassTraversalResult,
    iter_h264_spool,
)
from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter
from robata.application.canonical.bounded_media import (
    ACCESS_UNIT_FRAMING_VERSION,
    BoundedMediaPolicy,
    BoundedSinglePassMediaPlanner,
    PlannerFinish,
    RingSnapshot,
    SinglePassPlanningSink,
)
from robata.application.video_export import StagedSixCameraVideoExport
from robata.contracts import (
    CAMERA_IDS,
    CameraId,
    Sha256Digest,
    SixCameraMap,
    canonical_json_bytes,
    exact_bytes_sha256,
)
from robata.ports import ChannelInspection, ExportedCameraVideoFacts, McapInspection
from robata.tempfiles import make_staging_directory

SPOOL_SET_VERSION: Final = "robata-h264-spool-set-v2"
SPOOL_SEAL_FILENAME: Final = "h264-spool-set.seal.json"
_READ_CHUNK_BYTES: Final = 1024 * 1024


class SinglePassVideoProductionError(RuntimeError):
    """A sealed spool set or its staged export cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class H264SpoolSetFacts:
    """Exact immutable facts accepted from the spool-set seal."""

    directory: Path
    seal_path: Path
    seal_sha256: Sha256Digest
    source_size_bytes: int
    source_sha256: Sha256Digest
    source_message_count: int
    selected_packet_count: int
    camera_packet_counts: SixCameraMap[int]
    final_end_ns: int
    planner_policy_sha256: Sha256Digest
    spools: SixCameraMap[H264SpoolFacts]
    inspection: McapInspection
    planning_mode: str
    preflight_message_indexes_complete: bool


@dataclass(frozen=True, slots=True)
class SinglePassVideoProductionFacts:
    """Composition-facing facts for one fresh or recovered staged production."""

    staged_export: StagedSixCameraVideoExport
    traversal: SinglePassTraversalResult
    spool_set: H264SpoolSetFacts
    planner_finish: PlannerFinish
    planner_ring_snapshots: tuple[RingSnapshot, ...]
    reused_spool_set: bool
    max_parallel_exports: int


class DurableSinglePassVideoProducer:
    """Callable staged producer backed by one sealed durable spool set."""

    def __init__(
        self,
        *,
        inspection: McapInspection,
        channels: SixCameraMap[ChannelInspection],
        planner_policy: BoundedMediaPolicy | None,
        spool_directory: Path,
        final_end_ns: int | None = None,
        max_parallel_exports: int = 2,
        align_timestamp: Callable[[CameraId, int], int] | None = None,
        planning_sink: SinglePassPlanningSink | None = None,
        preflight: McapPreflight | None = None,
        planner_policy_factory: Callable[[int], BoundedMediaPolicy] | None = None,
        planner_source_scope_digest: str | None = None,
        tee: McapSinglePassH264Tee | None = None,
        exporter: PyAvH264Mp4Exporter | None = None,
    ) -> None:
        if isinstance(max_parallel_exports, bool) or not isinstance(max_parallel_exports, int):
            raise TypeError("max_parallel_exports must be an integer")
        if not 1 <= max_parallel_exports <= len(CAMERA_IDS):
            raise ValueError("max_parallel_exports must be between one and six")
        if planner_policy is None:
            if (
                preflight is None
                or preflight.message_indexes_complete
                or planner_policy_factory is None
                or final_end_ns is not None
                or planner_source_scope_digest is None
            ):
                raise ValueError(
                    "deferred planner policy requires an incomplete-index preflight, "
                    "a policy factory, a planner source scope, and no provisional final_end_ns"
                )
            resolved_final_end: int | None = None
            resolved_planner_scope = planner_source_scope_digest
        else:
            if preflight is not None and not preflight.message_indexes_complete:
                raise ValueError("incomplete-index preflight requires deferred planner policy")
            resolved_final_end = (
                inspection.last_message_time_ns + 1
                if final_end_ns is None and inspection.last_message_time_ns is not None
                else final_end_ns
            )
            if isinstance(resolved_final_end, bool) or not isinstance(resolved_final_end, int):
                raise ValueError(
                    "final_end_ns requires an inspected end time or an explicit integer"
                )
            if resolved_final_end <= planner_policy.source_origin_ns:
                raise ValueError("final_end_ns must be after the planner source origin")
            resolved_planner_scope = (
                planner_policy.source_scope_digest
                if planner_source_scope_digest is None
                else planner_source_scope_digest
            )
            if planner_policy.source_scope_digest != resolved_planner_scope:
                raise ValueError("planner policy differs from the declared planner source scope")
        self._inspection = inspection
        self._channels = channels
        self._planner_policy = planner_policy
        self._spool_directory = Path(spool_directory)
        self._final_end_ns = resolved_final_end
        self._max_parallel_exports = max_parallel_exports
        self._align_timestamp = align_timestamp
        self._planning_sink = planning_sink
        self._preflight = preflight
        self._planner_policy_factory = planner_policy_factory
        self._planner_source_scope_digest = resolved_planner_scope
        self._tee = tee or McapSinglePassH264Tee()
        self._exporter = exporter or PyAvH264Mp4Exporter()
        self._last_facts: SinglePassVideoProductionFacts | None = None
        self._prepared: (
            tuple[
                H264SpoolSetFacts,
                BoundedSinglePassMediaPlanner,
                SinglePassTraversalResult,
                bool,
            ]
            | None
        ) = None

    def __call__(self, staging_directory: Path, /) -> StagedSixCameraVideoExport:
        facts = self.produce(staging_directory)
        self._last_facts = facts
        return facts.staged_export

    def produce(self, staging_directory: Path) -> SinglePassVideoProductionFacts:
        """Produce staged MP4 artifacts from a fresh or recovered sealed spool set."""

        staging_directory = Path(staging_directory)
        if not staging_directory.is_dir() or staging_directory.is_symlink():
            raise SinglePassVideoProductionError(
                "staging_directory must be an existing private directory"
            )

        spool_set, planner, traversal, reused = self._prepare()

        camera_facts = self._export_spools(spool_set, staging_directory)
        staged = StagedSixCameraVideoExport(
            source_size_bytes=spool_set.source_size_bytes,
            source_sha256=spool_set.source_sha256,
            camera_facts=camera_facts,
        )
        facts = SinglePassVideoProductionFacts(
            staged_export=staged,
            traversal=traversal,
            spool_set=spool_set,
            planner_finish=_required_planner_finish(traversal),
            planner_ring_snapshots=planner.ring_snapshots(),
            reused_spool_set=reused,
            max_parallel_exports=self._max_parallel_exports,
        )
        self._last_facts = facts
        return facts

    def prepare(self) -> H264SpoolSetFacts:
        """Capture or recover the accepted spool set without exporting it twice."""

        return self._prepare()[0]

    def release_exported_spool_set(self) -> None:
        """Release spools after the caller has verified the registered publication."""

        if self._last_facts is None and self._prepared is None:
            raise SinglePassVideoProductionError(
                "cannot release spool set before source preparation"
            )
        directory = self._spool_directory.resolve()
        parent = self._spool_directory.parent.resolve()
        if directory.parent != parent or directory == parent:
            raise SinglePassVideoProductionError("spool directory is outside its owner")
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            return
        except OSError as error:
            raise SinglePassVideoProductionError(
                f"cannot release exported spool set: {error}"
            ) from error

    def _prepare(
        self,
    ) -> tuple[
        H264SpoolSetFacts,
        BoundedSinglePassMediaPlanner,
        SinglePassTraversalResult,
        bool,
    ]:
        if self._prepared is not None:
            return self._prepared
        if self._spool_directory.exists():
            spool_set = self._load_spool_set()
            planner, traversal = self._replay_planner(spool_set)
            reused = True
        else:
            spool_set, planner, traversal = self._capture_fresh_spool_set()
            reused = False
        self._accept_inspection(spool_set.inspection)
        self._prepared = (spool_set, planner, traversal, reused)
        return self._prepared

    @property
    def inspection(self) -> McapInspection:
        if self._prepared is None:
            raise RuntimeError("single-pass inspection is available only after prepare")
        return self._inspection

    @property
    def channels(self) -> SixCameraMap[ChannelInspection]:
        if self._prepared is None:
            raise RuntimeError("single-pass channels are available only after prepare")
        return self._channels

    @property
    def last_facts(self) -> SinglePassVideoProductionFacts:
        if self._last_facts is None:
            raise RuntimeError("single-pass video facts are available only after production")
        return self._last_facts

    @property
    def max_parallel_exports(self) -> int:
        return self._max_parallel_exports

    @property
    def planner_finish(self) -> PlannerFinish:
        if self._prepared is None:
            raise RuntimeError("planner finish is available only after prepare")
        return _required_planner_finish(self._prepared[2])

    @property
    def planner_policy(self) -> BoundedMediaPolicy:
        if self._prepared is None:
            raise RuntimeError("planner policy is available only after prepare")
        return _required_policy(self._planner_policy)

    def _capture_fresh_spool_set(
        self,
    ) -> tuple[
        H264SpoolSetFacts,
        BoundedSinglePassMediaPlanner,
        SinglePassTraversalResult,
    ]:
        parent = self._spool_directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = make_staging_directory(
            parent,
            prefix=f".{self._spool_directory.name}.capture-",
        )
        published = False
        planner = (
            BoundedSinglePassMediaPlanner(self._planner_policy)
            if self._planner_policy is not None
            else None
        )
        bootstrap_planners: list[BoundedSinglePassMediaPlanner] = []

        def create_bootstrap_planner(source_origin_ns: int) -> BoundedSinglePassMediaPlanner:
            if self._planner_policy_factory is None or bootstrap_planners:
                raise SinglePassVideoProductionError(
                    "bootstrap planner factory is unavailable or was already used"
                )
            policy = self._planner_policy_factory(source_origin_ns)
            if policy.source_scope_digest != self._planner_source_scope_digest:
                raise SinglePassVideoProductionError(
                    "bootstrap planner policy differs from the declared planner source scope"
                )
            created = BoundedSinglePassMediaPlanner(policy)
            bootstrap_planners.append(created)
            return created

        branches = {
            camera_id: AppendOnlyH264SpoolBranch(
                camera_id,
                temporary / _spool_filename(camera_id),
            )
            for camera_id in CAMERA_IDS
        }
        try:
            traversal = self._tee.traverse(
                self._inspection.source,
                self._channels,
                planner,
                branches,
                align_timestamp=self._align_timestamp,
                planning_sink=self._planning_sink,
                final_end_ns=self._final_end_ns,
                preflight=self._preflight,
                expected_source_sha256=self._inspection.source_sha256,
                planner_source_scope_digest=self._planner_source_scope_digest,
                bootstrap_planner_factory=(create_bootstrap_planner if planner is None else None),
            )
            spool_facts = SixCameraMap[H264SpoolFacts].model_validate(
                {camera_id: branches[camera_id].facts for camera_id in CAMERA_IDS},
                strict=True,
            )
            if planner is None:
                final_channels = _mapped_channels_from_inspection(
                    self._channels,
                    traversal.inspection,
                )
                if self._planner_policy_factory is None:
                    raise SinglePassVideoProductionError(
                        "deferred planner policy factory is absent"
                    )
                source_origin_ns, self._final_end_ns = _mapped_channel_bounds(final_channels)
                if bootstrap_planners:
                    planner = bootstrap_planners[0]
                    policy = planner.policy
                    self._planner_policy = policy
                    if (
                        traversal.planning_mode != PLANNING_MODE_LIVE_BOOTSTRAP
                        or policy.source_origin_ns != source_origin_ns
                        or traversal.final_end_ns != self._final_end_ns
                    ):
                        raise SinglePassVideoProductionError(
                            "bootstrap planner differs from final mapped source bounds"
                        )
                else:
                    policy = self._planner_policy_factory(source_origin_ns)
                    self._planner_policy = policy
                    deferred_spool_set = H264SpoolSetFacts(
                        directory=temporary,
                        seal_path=temporary / SPOOL_SEAL_FILENAME,
                        seal_sha256="0" * 64,
                        source_size_bytes=traversal.source_size_bytes,
                        source_sha256=traversal.source_sha256,
                        source_message_count=traversal.source_message_count,
                        selected_packet_count=traversal.selected_packet_count,
                        camera_packet_counts=traversal.camera_packet_counts,
                        final_end_ns=self._final_end_ns,
                        planner_policy_sha256=_policy_sha256(policy),
                        spools=spool_facts,
                        inspection=traversal.inspection,
                        planning_mode=PLANNING_MODE_EOS_SPOOL_REPLAY,
                        preflight_message_indexes_complete=False,
                    )
                    planner, traversal = self._replay_planner(deferred_spool_set)
            self._verify_fresh_traversal(traversal)
            for camera_id in CAMERA_IDS:
                _sync_file(spool_facts[camera_id].path)
            seal_document = self._seal_document(traversal, spool_facts)
            seal_bytes = canonical_json_bytes(seal_document)
            _write_new_file(temporary / SPOOL_SEAL_FILENAME, seal_bytes)
            _sync_directory(temporary)
            try:
                temporary.rename(self._spool_directory)
            except FileExistsError as exc:
                raise SinglePassVideoProductionError(
                    "spool set appeared during atomic publication"
                ) from exc
            published = True
            _sync_directory(parent)
        except SinglePassVideoProductionError:
            raise
        except Exception as exc:
            raise SinglePassVideoProductionError(
                f"fresh single-pass capture failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if not published and temporary.exists():
                shutil.rmtree(temporary)
        final_spools = SixCameraMap[H264SpoolFacts].model_validate(
            {
                camera_id: H264SpoolFacts(
                    camera_id=camera_id,
                    path=self._spool_directory / _spool_filename(camera_id),
                    packet_count=spool_facts[camera_id].packet_count,
                    size_bytes=spool_facts[camera_id].size_bytes,
                    sha256=spool_facts[camera_id].sha256,
                )
                for camera_id in CAMERA_IDS
            },
            strict=True,
        )
        spool_set = H264SpoolSetFacts(
            directory=self._spool_directory,
            seal_path=self._spool_directory / SPOOL_SEAL_FILENAME,
            seal_sha256=exact_bytes_sha256(seal_bytes),
            source_size_bytes=traversal.source_size_bytes,
            source_sha256=traversal.source_sha256,
            source_message_count=traversal.source_message_count,
            selected_packet_count=traversal.selected_packet_count,
            camera_packet_counts=traversal.camera_packet_counts,
            final_end_ns=traversal.final_end_ns,
            planner_policy_sha256=_policy_sha256(_required_policy(self._planner_policy)),
            spools=final_spools,
            inspection=traversal.inspection,
            planning_mode=traversal.planning_mode,
            preflight_message_indexes_complete=(
                self._preflight.message_indexes_complete
                if self._preflight is not None
                else traversal.planning_mode == PLANNING_MODE_LIVE_INDEXED
            ),
        )
        return spool_set, planner, traversal

    def _verify_fresh_traversal(self, traversal: SinglePassTraversalResult) -> None:
        expected_selected = sum(channel.message_count for channel in self._channels.values())
        if (
            traversal.source_size_bytes != self._inspection.source_size_bytes
            or traversal.source_sha256 != self._inspection.source_sha256
            or traversal.source_message_count != self._inspection.message_count
            or traversal.selected_packet_count != expected_selected
            or self._final_end_ns is None
            or traversal.final_end_ns != self._final_end_ns
            or traversal.planner_finish is None
        ):
            raise SinglePassVideoProductionError(
                "single-pass traversal differs from inspected source identity or counts"
            )
        for camera_id in CAMERA_IDS:
            if traversal.camera_packet_counts[camera_id] != self._channels[camera_id].message_count:
                raise SinglePassVideoProductionError(
                    f"single-pass count differs for {camera_id.value}"
                )

    def _accept_inspection(self, inspection: McapInspection) -> None:
        if (
            inspection.source != self._inspection.source
            or inspection.source_size_bytes != self._inspection.source_size_bytes
            or inspection.source_sha256 != self._inspection.source_sha256
            or inspection.message_count != self._inspection.message_count
        ):
            raise SinglePassVideoProductionError(
                "accepted inspection differs from the expected source identity"
            )
        self._channels = _mapped_channels_from_inspection(self._channels, inspection)
        self._inspection = inspection

    def _seal_document(
        self,
        traversal: SinglePassTraversalResult,
        spool_facts: SixCameraMap[H264SpoolFacts],
    ) -> dict[str, object]:
        return {
            "camera_packet_counts": {
                camera_id.value: traversal.camera_packet_counts[camera_id]
                for camera_id in CAMERA_IDS
            },
            "channels": _channel_document(
                _mapped_channels_from_inspection(self._channels, traversal.inspection)
            ),
            "final_end_ns": str(traversal.final_end_ns),
            "planner_policy": _policy_document(_required_policy(self._planner_policy)),
            "planner_policy_sha256": _policy_sha256(_required_policy(self._planner_policy)),
            "planning_mode": traversal.planning_mode,
            "preflight_message_indexes_complete": (
                self._preflight.message_indexes_complete
                if self._preflight is not None
                else traversal.planning_mode == PLANNING_MODE_LIVE_INDEXED
            ),
            "selected_packet_count": traversal.selected_packet_count,
            "inspection": _inspection_document(traversal.inspection),
            "source": {
                "message_count": traversal.source_message_count,
                "sha256": traversal.source_sha256,
                "size_bytes": traversal.source_size_bytes,
            },
            "spool_set_version": SPOOL_SET_VERSION,
            "spools": [
                {
                    "camera_id": camera_id.value,
                    "packet_count": spool_facts[camera_id].packet_count,
                    "path": _spool_filename(camera_id),
                    "sha256": spool_facts[camera_id].sha256,
                    "size_bytes": spool_facts[camera_id].size_bytes,
                }
                for camera_id in CAMERA_IDS
            ],
        }

    def _load_spool_set(self) -> H264SpoolSetFacts:
        directory = self._spool_directory
        if not directory.is_dir() or directory.is_symlink():
            raise SinglePassVideoProductionError(
                "existing spool path is not a reusable sealed directory"
            )
        seal_path = directory / SPOOL_SEAL_FILENAME
        if not seal_path.is_file() or seal_path.is_symlink():
            raise SinglePassVideoProductionError(
                "existing spool directory has no complete seal and cannot be reused"
            )
        raw = seal_path.read_bytes()
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SinglePassVideoProductionError("spool-set seal is not valid JSON") from exc
        if not isinstance(document, dict) or canonical_json_bytes(document) != raw:
            raise SinglePassVideoProductionError("spool-set seal is not canonical JSON")
        if document.get("spool_set_version") != SPOOL_SET_VERSION:
            raise SinglePassVideoProductionError("spool-set seal version is unsupported")
        planning_mode = _planning_mode(document.get("planning_mode"))
        preflight_message_indexes_complete = _boolean(
            document.get("preflight_message_indexes_complete"),
            "preflight_message_indexes_complete",
        )
        if (preflight_message_indexes_complete and planning_mode != PLANNING_MODE_LIVE_INDEXED) or (
            not preflight_message_indexes_complete and planning_mode == PLANNING_MODE_LIVE_INDEXED
        ):
            raise SinglePassVideoProductionError(
                "spool-set planning mode differs from its preflight capability"
            )
        policy = _required_policy(self._planner_policy)
        policy_document = _policy_document(policy)
        policy_sha256 = _policy_sha256(policy)
        if (
            document.get("planner_policy") != policy_document
            or document.get("planner_policy_sha256") != policy_sha256
        ):
            raise SinglePassVideoProductionError("spool-set planner policy differs")

        source = _object(document.get("source"), "source")
        source_size = _integer(source.get("size_bytes"), "source.size_bytes", minimum=1)
        source_digest = _digest(source.get("sha256"), "source.sha256")
        source_messages = _integer(
            source.get("message_count"),
            "source.message_count",
            minimum=1,
        )
        if (
            source_size != self._inspection.source_size_bytes
            or source_digest != self._inspection.source_sha256
            or source_messages != self._inspection.message_count
        ):
            raise SinglePassVideoProductionError("spool-set source identity differs")
        sealed_inspection = _inspection_from_document(
            document.get("inspection"),
            source=self._inspection.source,
            source_size_bytes=source_size,
            source_sha256=source_digest,
            source_message_count=source_messages,
        )
        sealed_channels = _mapped_channels_from_inspection(self._channels, sealed_inspection)
        if document.get("channels") != _channel_document(sealed_channels):
            raise SinglePassVideoProductionError("spool-set channel mapping differs")

        final_end_ns = _integer_string(document.get("final_end_ns"), "final_end_ns")
        if final_end_ns != self._final_end_ns:
            raise SinglePassVideoProductionError("spool-set final_end_ns differs")
        selected_packet_count = _integer(
            document.get("selected_packet_count"),
            "selected_packet_count",
            minimum=1,
        )
        raw_counts = _object(document.get("camera_packet_counts"), "camera_packet_counts")
        counts = SixCameraMap[int].model_validate(
            {
                camera_id: _integer(
                    raw_counts.get(camera_id.value),
                    f"camera_packet_counts.{camera_id.value}",
                    minimum=1,
                )
                for camera_id in CAMERA_IDS
            },
            strict=True,
        )
        if selected_packet_count != sum(counts.values()):
            raise SinglePassVideoProductionError("spool-set selected count does not reconcile")
        if any(
            counts[camera_id] != self._channels[camera_id].message_count for camera_id in CAMERA_IDS
        ):
            raise SinglePassVideoProductionError("spool-set camera counts differ")

        raw_spools = document.get("spools")
        if not isinstance(raw_spools, list) or len(raw_spools) != len(CAMERA_IDS):
            raise SinglePassVideoProductionError("spool-set must contain six spool facts")
        spool_map: dict[CameraId, H264SpoolFacts] = {}
        for camera_id, raw_spool in zip(CAMERA_IDS, raw_spools, strict=True):
            spool = _object(raw_spool, f"spools.{camera_id.value}")
            filename = _spool_filename(camera_id)
            if spool.get("camera_id") != camera_id.value or spool.get("path") != filename:
                raise SinglePassVideoProductionError(
                    f"spool-set order or path differs for {camera_id.value}"
                )
            path = directory / filename
            if not path.is_file() or path.is_symlink():
                raise SinglePassVideoProductionError(
                    f"sealed spool is missing for {camera_id.value}"
                )
            expected_size = _integer(
                spool.get("size_bytes"),
                f"spools.{camera_id.value}.size_bytes",
                minimum=1,
            )
            expected_digest = _digest(
                spool.get("sha256"),
                f"spools.{camera_id.value}.sha256",
            )
            packet_count = _integer(
                spool.get("packet_count"),
                f"spools.{camera_id.value}.packet_count",
                minimum=1,
            )
            actual_size, actual_digest = _hash_file(path)
            if actual_size != expected_size or actual_digest != expected_digest:
                raise SinglePassVideoProductionError(
                    f"sealed spool bytes differ for {camera_id.value}"
                )
            if packet_count != counts[camera_id]:
                raise SinglePassVideoProductionError(
                    f"sealed spool count differs for {camera_id.value}"
                )
            spool_map[camera_id] = H264SpoolFacts(
                camera_id=camera_id,
                path=path,
                packet_count=packet_count,
                size_bytes=expected_size,
                sha256=expected_digest,
            )

        return H264SpoolSetFacts(
            directory=directory,
            seal_path=seal_path,
            seal_sha256=exact_bytes_sha256(raw),
            source_size_bytes=source_size,
            source_sha256=source_digest,
            source_message_count=source_messages,
            selected_packet_count=selected_packet_count,
            camera_packet_counts=counts,
            final_end_ns=final_end_ns,
            planner_policy_sha256=policy_sha256,
            spools=SixCameraMap[H264SpoolFacts].model_validate(spool_map, strict=True),
            inspection=sealed_inspection,
            planning_mode=planning_mode,
            preflight_message_indexes_complete=preflight_message_indexes_complete,
        )

    def _replay_planner(
        self,
        spool_set: H264SpoolSetFacts,
    ) -> tuple[BoundedSinglePassMediaPlanner, SinglePassTraversalResult]:
        planner = BoundedSinglePassMediaPlanner(_required_policy(self._planner_policy))
        iterators: dict[CameraId, Iterator[H264PacketEnvelope]] = {
            camera_id: iter_h264_spool(spool_set.spools[camera_id].path) for camera_id in CAMERA_IDS
        }
        heap: list[tuple[int, str, CameraId, H264PacketEnvelope]] = []
        observed = {camera_id: 0 for camera_id in CAMERA_IDS}
        for camera_id in CAMERA_IDS:
            first = next(iterators[camera_id], None)
            if first is None:
                raise SinglePassVideoProductionError(f"sealed spool is empty for {camera_id.value}")
            heapq.heappush(
                heap,
                (first.packet.traversal_index, camera_id.value, camera_id, first),
            )

        selected_count = 0
        while heap:
            _index, _camera_value, camera_id, envelope = heapq.heappop(heap)
            packet = envelope.packet
            if packet.camera_id is not camera_id or packet.source_order != observed[camera_id]:
                raise SinglePassVideoProductionError(
                    f"sealed spool ordering differs for {camera_id.value}"
                )
            emission = planner.push(packet)
            if self._planning_sink is not None:
                self._planning_sink.append_emission(emission)
            observed[camera_id] += 1
            selected_count += 1
            following = next(iterators[camera_id], None)
            if following is not None:
                heapq.heappush(
                    heap,
                    (
                        following.packet.traversal_index,
                        camera_id.value,
                        camera_id,
                        following,
                    ),
                )

        if selected_count != spool_set.selected_packet_count or any(
            observed[camera_id] != spool_set.camera_packet_counts[camera_id]
            for camera_id in CAMERA_IDS
        ):
            raise SinglePassVideoProductionError("sealed spool replay count does not reconcile")
        finish = planner.finish(spool_set.final_end_ns)
        if self._planning_sink is not None:
            self._planning_sink.seal(finish)
        traversal = SinglePassTraversalResult(
            source_size_bytes=spool_set.source_size_bytes,
            source_sha256=spool_set.source_sha256,
            source_message_count=spool_set.source_message_count,
            selected_packet_count=selected_count,
            camera_packet_counts=spool_set.camera_packet_counts,
            final_end_ns=spool_set.final_end_ns,
            planner_finish=finish,
            inspection=spool_set.inspection,
            planning_mode=spool_set.planning_mode,
        )
        return planner, traversal

    def _export_spools(
        self,
        spool_set: H264SpoolSetFacts,
        staging_directory: Path,
    ) -> tuple[ExportedCameraVideoFacts, ...]:
        with ThreadPoolExecutor(
            max_workers=self._max_parallel_exports,
            thread_name_prefix="robata-spool-export",
        ) as executor:
            futures = {
                camera_id: executor.submit(
                    self._export_camera_spool,
                    camera_id,
                    spool_set.spools[camera_id],
                    staging_directory,
                )
                for camera_id in CAMERA_IDS
            }
            return tuple(futures[camera_id].result() for camera_id in CAMERA_IDS)

    def _export_camera_spool(
        self,
        camera_id: CameraId,
        spool: H264SpoolFacts,
        staging_directory: Path,
    ) -> ExportedCameraVideoFacts:
        session = self._exporter.begin_incremental(
            camera_id,
            self._channels[camera_id],
            staging_directory / f"{camera_id.value}.mp4",
            staging_directory / f"{camera_id.value}.timestamps.jsonl",
        )
        observed = 0
        try:
            for envelope in iter_h264_spool(spool.path):
                session.append_access_unit(
                    envelope,
                    envelope.packet.reference(),
                    framing_version=ACCESS_UNIT_FRAMING_VERSION,
                )
                observed += 1
            if observed != spool.packet_count:
                raise SinglePassVideoProductionError(
                    f"export replay count differs for {camera_id.value}"
                )
            session.seal()
            return session.facts
        finally:
            session.abort()


def read_sealed_mcap_inspection(
    spool_directory: Path,
    *,
    source: Path,
    expected_source_sha256: Sha256Digest,
) -> McapInspection:
    """Restore complete inspection facts from a canonical V2 seal without opening source."""

    directory = Path(spool_directory)
    seal_path = directory / SPOOL_SEAL_FILENAME
    if not directory.is_dir() or directory.is_symlink():
        raise SinglePassVideoProductionError("sealed spool directory is not reusable")
    if not seal_path.is_file() or seal_path.is_symlink():
        raise SinglePassVideoProductionError("sealed spool directory has no complete seal")
    raw = seal_path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SinglePassVideoProductionError("spool-set seal is not valid JSON") from exc
    if not isinstance(document, dict) or canonical_json_bytes(document) != raw:
        raise SinglePassVideoProductionError("spool-set seal is not canonical JSON")
    if document.get("spool_set_version") != SPOOL_SET_VERSION:
        raise SinglePassVideoProductionError("spool-set seal version is unsupported")
    source_document = _object(document.get("source"), "source")
    source_size = _integer(source_document.get("size_bytes"), "source.size_bytes", minimum=1)
    source_digest = _digest(source_document.get("sha256"), "source.sha256")
    source_messages = _integer(
        source_document.get("message_count"),
        "source.message_count",
        minimum=1,
    )
    if source_digest != expected_source_sha256:
        raise SinglePassVideoProductionError("spool-set source digest differs")
    return _inspection_from_document(
        document.get("inspection"),
        source=Path(source),
        source_size_bytes=source_size,
        source_sha256=source_digest,
        source_message_count=source_messages,
    )


def _mapped_channels_from_inspection(
    expected: SixCameraMap[ChannelInspection],
    inspection: McapInspection,
) -> SixCameraMap[ChannelInspection]:
    by_id = {channel.channel_id: channel for channel in inspection.channels}
    resolved: dict[CameraId, ChannelInspection] = {}
    for camera_id in CAMERA_IDS:
        prior = expected[camera_id]
        channel = by_id.get(prior.channel_id)
        if (
            channel is None
            or channel.topic != prior.topic
            or channel.schema_name != prior.schema_name
            or channel.message_encoding != prior.message_encoding
            or channel.schema_encoding != prior.schema_encoding
            or channel.schema_content_sha256 != prior.schema_content_sha256
        ):
            raise SinglePassVideoProductionError(
                f"accepted inspection changes mapped identity for {camera_id.value}"
            )
        resolved[camera_id] = channel
    return SixCameraMap[ChannelInspection].model_validate(resolved, strict=True)


def _mapped_channel_bounds(
    channels: SixCameraMap[ChannelInspection],
) -> tuple[int, int]:
    first: list[int] = []
    last: list[int] = []
    for camera_id in CAMERA_IDS:
        channel = channels[camera_id]
        if (
            channel.message_count < 1
            or channel.first_message_time_ns is None
            or channel.last_message_time_ns is None
            or not channel.monotonic
        ):
            raise SinglePassVideoProductionError(
                f"accepted {camera_id.value} channel lacks monotonic timestamp bounds"
            )
        first.append(channel.first_message_time_ns)
        last.append(channel.last_message_time_ns)
    return min(first), max(last) + 1


def _required_policy(policy: BoundedMediaPolicy | None) -> BoundedMediaPolicy:
    if policy is None:
        raise SinglePassVideoProductionError("planner policy has not been resolved")
    return policy


def _required_planner_finish(traversal: SinglePassTraversalResult) -> PlannerFinish:
    if traversal.planner_finish is None:
        raise SinglePassVideoProductionError("planner was not finalized before production")
    return traversal.planner_finish


def _planning_mode(value: object) -> str:
    if value not in {
        PLANNING_MODE_LIVE_INDEXED,
        PLANNING_MODE_LIVE_BOOTSTRAP,
        PLANNING_MODE_EOS_SPOOL_REPLAY,
    }:
        raise SinglePassVideoProductionError("spool-set planning_mode is unsupported")
    assert isinstance(value, str)
    return value


def _inspection_document(inspection: McapInspection) -> dict[str, object]:
    return {
        "channel_count": inspection.channel_count,
        "channels": [
            {
                "channel_id": channel.channel_id,
                "codec": channel.codec,
                "first_message_time_ns": _optional_integer_document(channel.first_message_time_ns),
                "frame_id": channel.frame_id,
                "last_message_time_ns": _optional_integer_document(channel.last_message_time_ns),
                "message_count": channel.message_count,
                "message_encoding": channel.message_encoding,
                "monotonic": channel.monotonic,
                "schema_content_sha256": channel.schema_content_sha256,
                "schema_encoding": channel.schema_encoding,
                "schema_name": channel.schema_name,
                "topic": channel.topic,
            }
            for channel in inspection.channels
        ],
        "first_message_time_ns": _optional_integer_document(inspection.first_message_time_ns),
        "header_library": inspection.header_library,
        "header_profile": inspection.header_profile,
        "last_message_time_ns": _optional_integer_document(inspection.last_message_time_ns),
        "message_count": inspection.message_count,
        "summary_available": inspection.summary_available,
    }


def _inspection_from_document(
    value: object,
    *,
    source: Path,
    source_size_bytes: int,
    source_sha256: Sha256Digest,
    source_message_count: int,
) -> McapInspection:
    document = _exact_object(
        value,
        "inspection",
        {
            "channel_count",
            "channels",
            "first_message_time_ns",
            "header_library",
            "header_profile",
            "last_message_time_ns",
            "message_count",
            "summary_available",
        },
    )
    raw_channels = document.get("channels")
    if not isinstance(raw_channels, list):
        raise SinglePassVideoProductionError("spool-set inspection.channels is not an array")
    channels: list[ChannelInspection] = []
    channel_fields = {
        "channel_id",
        "codec",
        "first_message_time_ns",
        "frame_id",
        "last_message_time_ns",
        "message_count",
        "message_encoding",
        "monotonic",
        "schema_content_sha256",
        "schema_encoding",
        "schema_name",
        "topic",
    }
    for index, raw_channel in enumerate(raw_channels):
        field = f"inspection.channels[{index}]"
        channel = _exact_object(raw_channel, field, channel_fields)
        schema_digest_value = channel.get("schema_content_sha256")
        schema_digest = (
            None
            if schema_digest_value is None
            else _digest(schema_digest_value, f"{field}.schema_content_sha256")
        )
        channels.append(
            ChannelInspection(
                channel_id=_integer(channel.get("channel_id"), f"{field}.channel_id", minimum=0),
                topic=_string(channel.get("topic"), f"{field}.topic"),
                schema_name=_optional_string(channel.get("schema_name"), f"{field}.schema_name"),
                message_encoding=_string(
                    channel.get("message_encoding"),
                    f"{field}.message_encoding",
                ),
                message_count=_integer(
                    channel.get("message_count"),
                    f"{field}.message_count",
                    minimum=0,
                ),
                first_message_time_ns=_optional_integer_string(
                    channel.get("first_message_time_ns"),
                    f"{field}.first_message_time_ns",
                ),
                last_message_time_ns=_optional_integer_string(
                    channel.get("last_message_time_ns"),
                    f"{field}.last_message_time_ns",
                ),
                monotonic=_boolean(channel.get("monotonic"), f"{field}.monotonic"),
                codec=_optional_string(channel.get("codec"), f"{field}.codec"),
                frame_id=_optional_string(channel.get("frame_id"), f"{field}.frame_id"),
                schema_encoding=_optional_string(
                    channel.get("schema_encoding"),
                    f"{field}.schema_encoding",
                ),
                schema_content_sha256=schema_digest,
            )
        )
    channel_count = _integer(
        document.get("channel_count"),
        "inspection.channel_count",
        minimum=0,
    )
    message_count = _integer(
        document.get("message_count"),
        "inspection.message_count",
        minimum=0,
    )
    if channel_count != len(channels) or message_count != source_message_count:
        raise SinglePassVideoProductionError(
            "spool-set inspection counts differ from its source facts"
        )
    if tuple(channel.channel_id for channel in channels) != tuple(
        sorted(channel.channel_id for channel in channels)
    ) or len({channel.channel_id for channel in channels}) != len(channels):
        raise SinglePassVideoProductionError(
            "spool-set inspection channels are not unique canonical channel order"
        )
    return McapInspection(
        source=Path(source),
        source_size_bytes=source_size_bytes,
        source_sha256=source_sha256,
        header_profile=_string(document.get("header_profile"), "inspection.header_profile"),
        header_library=_string(document.get("header_library"), "inspection.header_library"),
        summary_available=_boolean(
            document.get("summary_available"),
            "inspection.summary_available",
        ),
        channel_count=channel_count,
        message_count=message_count,
        first_message_time_ns=_optional_integer_string(
            document.get("first_message_time_ns"),
            "inspection.first_message_time_ns",
        ),
        last_message_time_ns=_optional_integer_string(
            document.get("last_message_time_ns"),
            "inspection.last_message_time_ns",
        ),
        channels=tuple(channels),
    )


def _optional_integer_document(value: int | None) -> str | None:
    return None if value is None else str(value)


def _spool_filename(camera_id: CameraId) -> str:
    return f"{camera_id.value}.h264.spool"


def _policy_document(policy: BoundedMediaPolicy) -> dict[str, object]:
    return {
        "alignment_semantic_sha256": policy.alignment_semantic_sha256,
        "allowed_lateness_ns": str(policy.allowed_lateness_ns),
        "mapping_semantic_sha256": policy.mapping_semantic_sha256,
        "quality_period_ns": str(policy.quality_period_ns),
        "quality_policy_version": policy.quality_policy_version,
        "quality_selection_tolerance_ns": str(policy.quality_selection_tolerance_ns),
        "quality_target_phase_ns": str(policy.quality_target_phase_ns),
        "ring_duration_ns": str(policy.ring_duration_ns),
        "ring_max_bytes_per_camera": policy.ring_max_bytes_per_camera,
        "segment_duration_ns": str(policy.segment_duration_ns),
        "segmentation_policy_version": policy.segmentation_policy_version,
        "source_origin_ns": str(policy.source_origin_ns),
        "source_scope_digest": policy.source_scope_digest,
        "window_hop_ns": str(policy.window_hop_ns),
        "window_policy_version": policy.window_policy_version,
        "window_purpose": policy.window_purpose.value,
        "window_width_ns": str(policy.window_width_ns),
    }


def _policy_sha256(policy: BoundedMediaPolicy) -> Sha256Digest:
    return exact_bytes_sha256(canonical_json_bytes(_policy_document(policy)))


def _channel_document(
    channels: SixCameraMap[ChannelInspection],
) -> dict[str, object]:
    return {
        camera_id.value: {
            "channel_id": channel.channel_id,
            "first_message_time_ns": (
                str(channel.first_message_time_ns)
                if channel.first_message_time_ns is not None
                else None
            ),
            "last_message_time_ns": (
                str(channel.last_message_time_ns)
                if channel.last_message_time_ns is not None
                else None
            ),
            "message_count": channel.message_count,
            "topic": channel.topic,
        }
        for camera_id in CAMERA_IDS
        for channel in (channels[camera_id],)
    }


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SinglePassVideoProductionError(f"spool-set {field} is not an object")
    return value


def _exact_object(value: object, field: str, expected_fields: set[str]) -> dict[str, Any]:
    document = _object(value, field)
    if set(document) != expected_fields:
        raise SinglePassVideoProductionError(f"spool-set {field} fields differ")
    return document


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SinglePassVideoProductionError(f"spool-set {field} is not a nonempty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise SinglePassVideoProductionError(f"spool-set {field} is not a boolean")
    return value


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SinglePassVideoProductionError(f"spool-set {field} is not an integer")
    if minimum is not None and value < minimum:
        raise SinglePassVideoProductionError(f"spool-set {field} is below its minimum")
    return value


def _integer_string(value: object, field: str) -> int:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SinglePassVideoProductionError(f"spool-set {field} is not an integer string")
    try:
        result = int(value)
    except ValueError as exc:
        raise SinglePassVideoProductionError(f"spool-set {field} is not an integer string") from exc
    if str(result) != value:
        raise SinglePassVideoProductionError(f"spool-set {field} is not canonical")
    return result


def _optional_integer_string(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _integer_string(value, field)


def _digest(value: object, field: str) -> Sha256Digest:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SinglePassVideoProductionError(f"spool-set {field} is not SHA-256")
    return value


def _hash_file(path: Path) -> tuple[int, Sha256Digest]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def _write_new_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "SPOOL_SEAL_FILENAME",
    "SPOOL_SET_VERSION",
    "DurableSinglePassVideoProducer",
    "H264SpoolSetFacts",
    "SinglePassVideoProductionError",
    "SinglePassVideoProductionFacts",
    "read_sealed_mcap_inspection",
]
