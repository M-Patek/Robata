"""Run the local committed-run explorer ASGI server."""

from __future__ import annotations

import argparse
from pathlib import Path

from robata.web_api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve committed Robata runs read-only.")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--poll-interval-seconds", default=1.0, type=float)
    arguments = parser.parse_args()

    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit(
            "Web server dependencies are missing. Run `uv sync --extra web`."
        ) from error

    application = create_app(
        state_dir=arguments.state_dir,
        poll_interval_seconds=arguments.poll_interval_seconds,
    )
    uvicorn.run(application, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
