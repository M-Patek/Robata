"""Serve one local quantized Hugging Face vision model on a loopback-only port."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.inference.local_hf_endpoint import (  # noqa: E402
    LocalHfEndpointService,
    build_local_hf_checkpoint_identity,
    create_local_hf_endpoint_app,
    load_local_hf_checkpoint_identity,
    write_local_hf_checkpoint_identity,
)
from robata.inference.local_hf_runtime import (  # noqa: E402
    LocalHuggingFaceRuntimeError,
    LocalHuggingFaceVisionRuntime,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return parsed


def _loopback_host(value: str) -> str:
    if value not in {"127.0.0.1", "localhost", "::1"}:
        raise argparse.ArgumentTypeError(
            "the local model endpoint may bind only to 127.0.0.1, localhost, or ::1"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "local-hf-endpoints" / "qwen",
        help="local cache/offload root for this one resident model",
    )
    parser.add_argument("--model-identifier", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-version", default="local")
    parser.add_argument(
        "--idempotency-state-path",
        type=Path,
        default=None,
        help=(
            "durable SQLite idempotency state; defaults to <state-dir>/endpoint-idempotency.sqlite3"
        ),
    )
    parser.add_argument(
        "--checkpoint-manifest-path",
        type=Path,
        default=None,
        help=(
            "canonical checkpoint identity manifest; when supplied, it is reused on startup "
            "instead of rehashing the selected model files"
        ),
    )
    parser.add_argument(
        "--refresh-checkpoint-manifest",
        action="store_true",
        help="rehash the model directory and overwrite --checkpoint-manifest-path",
    )
    parser.add_argument("--host", type=_loopback_host, default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=_port,
        default=8101,
        help="loopback port; 8101 is the local control/Qwen convention",
    )
    parser.add_argument("--max-image-side", type=_positive_int, default=448)
    parser.add_argument("--gpu-weight-memory-gib", type=_positive_int, default=4)
    parser.add_argument("--cpu-weight-memory-gib", type=_positive_int, default=1)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    state_root = arguments.state_dir.resolve()
    idempotency_state_path = (
        arguments.idempotency_state_path.resolve()
        if arguments.idempotency_state_path is not None
        else state_root / "endpoint-idempotency.sqlite3"
    )
    cache_root = state_root / "model-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_root / "modules"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf-home"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        import uvicorn

        if arguments.refresh_checkpoint_manifest and arguments.checkpoint_manifest_path is None:
            raise ValueError("--refresh-checkpoint-manifest requires --checkpoint-manifest-path")
        if (
            arguments.checkpoint_manifest_path is not None
            and arguments.checkpoint_manifest_path.is_file()
            and not arguments.refresh_checkpoint_manifest
        ):
            checkpoint_identity = load_local_hf_checkpoint_identity(
                manifest_path=arguments.checkpoint_manifest_path
            )
        else:
            checkpoint_identity = build_local_hf_checkpoint_identity(
                model_directory=arguments.model_dir
            )
            if arguments.checkpoint_manifest_path is not None:
                write_local_hf_checkpoint_identity(
                    identity=checkpoint_identity,
                    manifest_path=arguments.checkpoint_manifest_path,
                )

        runtime = LocalHuggingFaceVisionRuntime(
            model_directory=arguments.model_dir,
            offload_directory=state_root / "model-offload",
            max_image_side=arguments.max_image_side,
            gpu_weight_memory_gib=arguments.gpu_weight_memory_gib,
            cpu_weight_memory_gib=arguments.cpu_weight_memory_gib,
        )
        service = LocalHfEndpointService(
            runtime=runtime,
            model_identifier=arguments.model_identifier,
            model_version=arguments.model_version,
            checkpoint_identity=checkpoint_identity,
            idempotency_state_path=idempotency_state_path,
        )
        application = create_local_hf_endpoint_app(service)
        uvicorn.run(
            application,
            host=arguments.host,
            port=arguments.port,
            workers=1,
            log_level=arguments.log_level,
            access_log=True,
        )
    except (ImportError, LocalHuggingFaceRuntimeError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "LOCAL_HF_ENDPOINT_FAILED",
                    "detail": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
