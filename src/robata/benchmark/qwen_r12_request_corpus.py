"""Read-only, exact-byte loader for persisted Qwen r12 inference requests.

The loader turns an immutable inference-evidence SQLite snapshot into a small
benchmark corpus descriptor.  It deliberately keeps image bytes out of the
returned objects: files are read once for exact-byte verification, then only
paths and pinned digests remain in memory.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.inference.adapter import VisionInferenceRequest
from robata.inference.models import VisionTask
from robata.inference.orchestrator import InferenceIntent

QWEN_REQUEST_CORPUS_MANIFEST_VERSION = "qwen-request-corpus-manifest-v1"
QWEN_R12_20260806_MANIFEST_SEMANTIC_SHA256 = (
    "d4bd44f5e573b2abc13000cf9421134ac0e8d00fe92890fc6a7fa265c84425ed"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QwenRequestCorpusError(ValueError):
    """The persisted request corpus is missing, mutable, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class QwenRequestCorpusExpected:
    """Optional fixture constraints supplied by the caller, never inferred as truth."""

    database_sha256: str
    intent_count: int | None = None
    task_counts: tuple[tuple[VisionTask, int], ...] = ()
    reference_count: int | None = None
    unique_image_count: int | None = None
    selected_images_per_intent: int | None = None
    require_single_compatibility_per_task: bool = True
    require_contiguous_task_buckets: bool = True

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.database_sha256) is None:
            raise ValueError("database_sha256 must be a lowercase SHA-256 digest")
        for name in (
            "intent_count",
            "reference_count",
            "unique_image_count",
            "selected_images_per_intent",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when supplied")
        if any(count < 1 for _, count in self.task_counts):
            raise ValueError("expected task counts must be positive")
        tasks = tuple(task for task, _ in self.task_counts)
        if len(set(tasks)) != len(tasks):
            raise ValueError("expected task buckets must not repeat a task")


# Exact constraints for the preserved six-camera r12 corpus.  The loader itself
# remains generic; callers opt into this profile explicitly.
QWEN_R12_20260806_EXPECTED = QwenRequestCorpusExpected(
    database_sha256="005e146fcad99ed5d24ef20a71a017e1b9147e8508913811bf63b69cae9a37b4",
    intent_count=51,
    task_counts=((VisionTask.QA_COARSE, 41), (VisionTask.QA_DENSE, 10)),
    reference_count=306,
    unique_image_count=276,
    selected_images_per_intent=6,
)


@dataclass(frozen=True, slots=True)
class BatchCompatibilityProjection:
    """Public stable projection matching orchestrator batch compatibility dimensions."""

    provider: str
    model_name: str
    model_version: str
    task: VisionTask
    model_policy_version: str
    output_schema_sha256: str
    timeout_ms: int
    input_shape: tuple[tuple[str, str, int, int], ...]

    def manifest_projection(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task.value,
            "model_policy_version": self.model_policy_version,
            "output_schema_sha256": self.output_schema_sha256,
            "timeout_ms": self.timeout_ms,
            "input_shape": [list(shape) for shape in self.input_shape],
        }


@dataclass(frozen=True, slots=True)
class QwenSelectedImage:
    """One verified selected call-part item without retained image bytes."""

    selected_ordinal: int
    provider_item_ordinal: int
    camera_id: str
    uri: str
    path: Path
    sha256: str
    byte_count: int
    media_type: str
    encoding: str
    width: int
    height: int

    def manifest_projection(self) -> dict[str, object]:
        # A filesystem locator is deliberately not semantic.  The digest and
        # provider-facing shape are sufficient to reproduce corpus identity.
        return {
            "selected_ordinal": self.selected_ordinal,
            "provider_item_ordinal": self.provider_item_ordinal,
            "camera_id": self.camera_id,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
            "encoding": self.encoding,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class QwenRequestCase:
    """One ordered, strictly validated persisted inference request."""

    ordinal: int
    sqlite_rowid: int
    payload_sha256: str
    intent: InferenceIntent
    request: VisionInferenceRequest
    input_plan_part_ordinal: int
    input_plan_part_count: int
    input_plan_part_semantic_sha256: str
    selected_images: tuple[QwenSelectedImage, ...]
    compatibility: BatchCompatibilityProjection

    def manifest_projection(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "inference_id": self.intent.inference_id,
            "logical_invocation_id": self.intent.logical_invocation_id,
            "request_id": self.request.request_id,
            "payload_sha256": self.payload_sha256,
            "task": self.request.task.value,
            "input_plan_id": self.request.input_plan_id,
            "input_plan_semantic_sha256": self.request.input_plan_semantic_sha256,
            "input_plan_part_ordinal": self.input_plan_part_ordinal,
            "input_plan_part_count": self.input_plan_part_count,
            "input_plan_part_semantic_sha256": self.input_plan_part_semantic_sha256,
            "compatibility": self.compatibility.manifest_projection(),
            "selected_images": [image.manifest_projection() for image in self.selected_images],
        }


@dataclass(frozen=True, slots=True)
class QwenTaskBucket:
    """One contiguous task run in original SQLite insertion order."""

    task: VisionTask
    start_case_ordinal: int
    end_case_ordinal_exclusive: int
    case_count: int
    compatibility: BatchCompatibilityProjection

    def manifest_projection(self) -> dict[str, object]:
        return {
            "task": self.task.value,
            "start_case_ordinal": self.start_case_ordinal,
            "end_case_ordinal_exclusive": self.end_case_ordinal_exclusive,
            "case_count": self.case_count,
            "compatibility": self.compatibility.manifest_projection(),
        }


@dataclass(frozen=True, slots=True)
class QwenRequestCorpus:
    """Immutable verified request corpus and its canonical semantic identity."""

    database_path: Path
    database_sha256: str
    database_byte_count: int
    cases: tuple[QwenRequestCase, ...]
    task_buckets: tuple[QwenTaskBucket, ...]
    reference_count: int
    unique_image_count: int
    semantic_sha256: str

    def manifest_projection(self) -> dict[str, object]:
        return {
            "manifest_version": QWEN_REQUEST_CORPUS_MANIFEST_VERSION,
            "database": {
                "exact_bytes_sha256": self.database_sha256,
                "byte_count": self.database_byte_count,
            },
            "counts": {
                "intents": len(self.cases),
                "references": self.reference_count,
                "unique_images": self.unique_image_count,
            },
            "task_buckets": [bucket.manifest_projection() for bucket in self.task_buckets],
            "cases": [case.manifest_projection() for case in self.cases],
        }

    def canonical_manifest_bytes(self) -> bytes:
        return canonical_json_bytes(self.manifest_projection())


@dataclass(frozen=True, slots=True)
class _VerifiedFile:
    path: Path
    sha256: str
    byte_count: int


def batch_compatibility_projection(
    request: VisionInferenceRequest,
) -> BatchCompatibilityProjection:
    """Derive exactly the public dimensions used by bounded microbatch dispatch."""

    plan = request.input_plan
    if plan is None:
        input_shape = tuple(("package", item.role, 0, 0) for item in request.package_inputs)
    else:
        items = plan.rendered_items
        if request.input_plan_part_ordinal is not None:
            part = plan.call_plan.parts[request.input_plan_part_ordinal]
            items = items[part.start_item_ordinal : part.end_item_ordinal_exclusive]
        input_shape = tuple(
            (
                item.artifact.media_type,
                item.artifact.encoding,
                item.artifact.width,
                item.artifact.height,
            )
            for item in items
        )
    return BatchCompatibilityProjection(
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        task=request.task,
        model_policy_version=request.model_policy_version,
        output_schema_sha256=request.output_schema.sha256,
        timeout_ms=request.timeout_ms,
        input_shape=input_shape,
    )


def load_qwen_request_corpus(
    database_path: Path | str,
    *,
    expected: QwenRequestCorpusExpected,
) -> QwenRequestCorpus:
    """Load and fully verify a persisted inference-intent corpus read-only."""

    path = Path(database_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise QwenRequestCorpusError(f"corpus database is not a file: {path}")
    database_bytes = path.read_bytes()
    database_sha256 = exact_bytes_sha256(database_bytes)
    database_byte_count = len(database_bytes)
    del database_bytes
    if database_sha256 != expected.database_sha256:
        raise QwenRequestCorpusError("corpus database exact SHA-256 does not match expected")

    rows = _read_intent_rows(path)
    cases: list[QwenRequestCase] = []
    seen_inference_ids: set[str] = set()
    seen_request_ids: set[str] = set()
    verified_files: dict[Path, _VerifiedFile] = {}
    reference_count = 0

    for ordinal, row in enumerate(rows):
        sqlite_rowid, stored_inference_id, stored_request_id, raw, stored_payload_sha256 = row
        payload_sha256 = exact_bytes_sha256(raw)
        if stored_payload_sha256 != payload_sha256:
            raise QwenRequestCorpusError(
                f"intent row {sqlite_rowid} payload_sha256 does not match exact payload bytes"
            )
        try:
            intent = InferenceIntent.model_validate_json(raw, strict=True)
        except (TypeError, ValueError) as exc:
            raise QwenRequestCorpusError(
                f"intent row {sqlite_rowid} failed strict InferenceIntent validation"
            ) from exc
        if canonical_json_bytes(intent) != raw:
            raise QwenRequestCorpusError(f"intent row {sqlite_rowid} payload is not canonical JSON")
        request = intent.request
        _validate_row_and_request_binding(
            intent=intent,
            stored_inference_id=stored_inference_id,
            stored_request_id=stored_request_id,
            sqlite_rowid=sqlite_rowid,
        )
        if intent.inference_id in seen_inference_ids:
            raise QwenRequestCorpusError(f"duplicate inference identity: {intent.inference_id}")
        if request.request_id in seen_request_ids:
            raise QwenRequestCorpusError(f"duplicate request identity: {request.request_id}")
        seen_inference_ids.add(intent.inference_id)
        seen_request_ids.add(request.request_id)

        plan = request.input_plan
        part_ordinal = request.input_plan_part_ordinal
        part_count = request.input_plan_part_count
        part_sha256 = request.input_plan_part_semantic_sha256
        if plan is None or part_ordinal is None or part_count is None or part_sha256 is None:
            raise QwenRequestCorpusError(
                f"intent row {sqlite_rowid} must bind one immutable input-plan call part"
            )
        part = plan.call_plan.parts[part_ordinal]
        selected_items = plan.rendered_items[
            part.start_item_ordinal : part.end_item_ordinal_exclusive
        ]
        if not selected_items:
            raise QwenRequestCorpusError(f"intent row {sqlite_rowid} selects no rendered items")
        if (
            expected.selected_images_per_intent is not None
            and len(selected_items) != expected.selected_images_per_intent
        ):
            raise QwenRequestCorpusError(
                f"intent row {sqlite_rowid} selected {len(selected_items)} images; "
                f"expected {expected.selected_images_per_intent}"
            )

        selected_images: list[QwenSelectedImage] = []
        for selected_ordinal, item in enumerate(selected_items):
            artifact = item.artifact
            if artifact.media_type != "image/png" or artifact.encoding.lower() != "png":
                raise QwenRequestCorpusError(
                    f"intent row {sqlite_rowid} selected a non-PNG rendered artifact"
                )
            image_path = _path_from_file_uri(artifact.uri)
            verified = _verify_png_file(
                image_path,
                expected_sha256=artifact.sha256,
                expected_byte_count=artifact.byte_count,
                cache=verified_files,
            )
            selected_images.append(
                QwenSelectedImage(
                    selected_ordinal=selected_ordinal,
                    provider_item_ordinal=item.provider_item_ordinal,
                    camera_id=item.camera_id.value,
                    uri=artifact.uri,
                    path=verified.path,
                    sha256=verified.sha256,
                    byte_count=verified.byte_count,
                    media_type=artifact.media_type,
                    encoding=artifact.encoding,
                    width=artifact.width,
                    height=artifact.height,
                )
            )
        reference_count += len(selected_images)
        cases.append(
            QwenRequestCase(
                ordinal=ordinal,
                sqlite_rowid=sqlite_rowid,
                payload_sha256=payload_sha256,
                intent=intent,
                request=request,
                input_plan_part_ordinal=part_ordinal,
                input_plan_part_count=part_count,
                input_plan_part_semantic_sha256=part_sha256,
                selected_images=tuple(selected_images),
                compatibility=batch_compatibility_projection(request),
            )
        )

    # A second exact hash closes the race where a source database changes while
    # its read-only snapshot or external files are being inspected.
    ending_sha256 = exact_bytes_sha256(path.read_bytes())
    if ending_sha256 != database_sha256:
        raise QwenRequestCorpusError("corpus database changed during read-only verification")

    task_buckets = _build_task_buckets(
        tuple(cases),
        require_single_compatibility=expected.require_single_compatibility_per_task,
        require_contiguous=expected.require_contiguous_task_buckets,
    )
    _validate_expected_counts(
        expected,
        cases=tuple(cases),
        task_buckets=task_buckets,
        reference_count=reference_count,
        unique_image_count=len(verified_files),
    )
    draft = QwenRequestCorpus(
        database_path=path,
        database_sha256=database_sha256,
        database_byte_count=database_byte_count,
        cases=tuple(cases),
        task_buckets=task_buckets,
        reference_count=reference_count,
        unique_image_count=len(verified_files),
        semantic_sha256="0" * 64,
    )
    return QwenRequestCorpus(
        database_path=draft.database_path,
        database_sha256=draft.database_sha256,
        database_byte_count=draft.database_byte_count,
        cases=draft.cases,
        task_buckets=draft.task_buckets,
        reference_count=draft.reference_count,
        unique_image_count=draft.unique_image_count,
        semantic_sha256=semantic_sha256(draft.manifest_projection()),
    )


def _read_intent_rows(path: Path) -> tuple[tuple[int, str, str, bytes, str], ...]:
    uri = f"{path.as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
                raise QwenRequestCorpusError("SQLite connection is not query-only")
            raw_rows = connection.execute(
                """
                SELECT rowid, inference_id, request_id, payload_json, payload_sha256
                FROM inference_intents
                ORDER BY rowid ASC
                """
            ).fetchall()
    except QwenRequestCorpusError:
        raise
    except sqlite3.Error as exc:
        raise QwenRequestCorpusError("could not read inference_intents in read-only mode") from exc
    rows: list[tuple[int, str, str, bytes, str]] = []
    for row in raw_rows:
        raw = row["payload_json"]
        if not isinstance(raw, bytes):
            raise QwenRequestCorpusError(
                "inference intent payload_json must be stored as BLOB bytes"
            )
        rows.append(
            (
                int(row["rowid"]),
                _required_text(row, "inference_id"),
                _required_text(row, "request_id"),
                raw,
                _required_text(row, "payload_sha256"),
            )
        )
    if not rows:
        raise QwenRequestCorpusError("inference_intents corpus is empty")
    return tuple(rows)


def _required_text(row: sqlite3.Row, column: str) -> str:
    value = row[column]
    if not isinstance(value, str) or not value:
        raise QwenRequestCorpusError(f"inference_intents.{column} must be nonempty text")
    return value


def _validate_row_and_request_binding(
    *,
    intent: InferenceIntent,
    stored_inference_id: str,
    stored_request_id: str,
    sqlite_rowid: int,
) -> None:
    request = intent.request
    if stored_inference_id != intent.inference_id:
        raise QwenRequestCorpusError(f"intent row {sqlite_rowid} inference_id binding drift")
    if stored_request_id != intent.request_id or intent.request_id != request.request_id:
        raise QwenRequestCorpusError(f"intent row {sqlite_rowid} request_id binding drift")
    bindings = (
        ("logical_invocation_id", intent.logical_invocation_id, request.logical_invocation_id),
        ("idempotency_key", intent.idempotency_key, request.idempotency_key),
        ("task", intent.task, request.task),
        ("provider", intent.provider, request.provider),
        ("model_name", intent.model_name, request.model_name),
        ("model_version", intent.model_version, request.model_version),
        ("input_plan_id", intent.input_plan_id, request.input_plan_id),
        (
            "input_plan_semantic_sha256",
            intent.input_plan_semantic_sha256,
            request.input_plan_semantic_sha256,
        ),
        (
            "input_plan_part_ordinal",
            intent.input_plan_part_ordinal,
            request.input_plan_part_ordinal,
        ),
        ("input_plan_part_count", intent.input_plan_part_count, request.input_plan_part_count),
        (
            "input_plan_part_semantic_sha256",
            intent.input_plan_part_semantic_sha256,
            request.input_plan_part_semantic_sha256,
        ),
    )
    for name, outer, inner in bindings:
        if outer != inner:
            raise QwenRequestCorpusError(f"intent row {sqlite_rowid} {name} binding drift")


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "file" or parsed.query or parsed.fragment:
        raise QwenRequestCorpusError(f"rendered artifact URI must be a plain file URI: {uri}")
    if parsed.netloc not in ("", "localhost"):
        raise QwenRequestCorpusError(f"rendered artifact file URI authority is unsupported: {uri}")
    local_text = url2pathname(parsed.path)
    if not local_text:
        raise QwenRequestCorpusError(f"rendered artifact file URI has no path: {uri}")
    try:
        return Path(local_text).resolve(strict=True)
    except OSError as exc:
        raise QwenRequestCorpusError(f"rendered artifact file is missing: {uri}") from exc


def _verify_png_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    cache: dict[Path, _VerifiedFile],
) -> _VerifiedFile:
    if path.suffix.lower() != ".png":
        raise QwenRequestCorpusError(f"rendered artifact is not a .png file: {path}")
    cached = cache.get(path)
    if cached is None:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise QwenRequestCorpusError(f"could not read rendered artifact: {path}") from exc
        if not payload.startswith(_PNG_SIGNATURE):
            raise QwenRequestCorpusError(f"rendered artifact does not have a PNG signature: {path}")
        cached = _VerifiedFile(
            path=path,
            sha256=exact_bytes_sha256(payload),
            byte_count=len(payload),
        )
        cache[path] = cached
    if cached.sha256 != expected_sha256:
        raise QwenRequestCorpusError(f"rendered artifact SHA-256 drift: {path}")
    if cached.byte_count != expected_byte_count:
        raise QwenRequestCorpusError(f"rendered artifact byte_count drift: {path}")
    return cached


def _build_task_buckets(
    cases: tuple[QwenRequestCase, ...],
    *,
    require_single_compatibility: bool,
    require_contiguous: bool,
) -> tuple[QwenTaskBucket, ...]:
    buckets: list[QwenTaskBucket] = []
    start = 0
    while start < len(cases):
        task = cases[start].request.task
        end = start + 1
        while end < len(cases) and cases[end].request.task is task:
            end += 1
        compatibility = cases[start].compatibility
        if require_single_compatibility and any(
            case.compatibility != compatibility for case in cases[start:end]
        ):
            raise QwenRequestCorpusError(f"batch compatibility drift within task {task.value}")
        buckets.append(
            QwenTaskBucket(
                task=task,
                start_case_ordinal=start,
                end_case_ordinal_exclusive=end,
                case_count=end - start,
                compatibility=compatibility,
            )
        )
        start = end
    if require_contiguous:
        tasks = tuple(bucket.task for bucket in buckets)
        if len(set(tasks)) != len(tasks):
            raise QwenRequestCorpusError("a task reappears after a later task bucket")
    if require_single_compatibility:
        by_task: dict[VisionTask, BatchCompatibilityProjection] = {}
        for case in cases:
            previous = by_task.setdefault(case.request.task, case.compatibility)
            if previous != case.compatibility:
                raise QwenRequestCorpusError(
                    f"batch compatibility drift across task {case.request.task.value}"
                )
    return tuple(buckets)


def _validate_expected_counts(
    expected: QwenRequestCorpusExpected,
    *,
    cases: tuple[QwenRequestCase, ...],
    task_buckets: tuple[QwenTaskBucket, ...],
    reference_count: int,
    unique_image_count: int,
) -> None:
    observed_task_counts = tuple((bucket.task, bucket.case_count) for bucket in task_buckets)
    checks: tuple[tuple[str, int | None, int], ...] = (
        ("intent_count", expected.intent_count, len(cases)),
        ("reference_count", expected.reference_count, reference_count),
        ("unique_image_count", expected.unique_image_count, unique_image_count),
    )
    for label, wanted, observed in checks:
        if wanted is not None and observed != wanted:
            raise QwenRequestCorpusError(f"{label} is {observed}; expected {wanted}")
    if expected.task_counts and observed_task_counts != expected.task_counts:
        raise QwenRequestCorpusError(
            f"task buckets are {observed_task_counts!r}; expected {expected.task_counts!r}"
        )
