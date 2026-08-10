from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_bounded_mage_benchmark.py"


_ENDPOINT_SOURCE = r"""from __future__ import annotations
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--log-level")
parser.add_argument("--mode", choices=("ready", "no-listen"), default="ready")
args = parser.parse_args()
if args.mode == "no-listen":
    time.sleep(60)
    raise SystemExit(0)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"status": "READY", "loaded": True, "concurrency": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
"""

_BENCHMARK_SOURCE = r"""from __future__ import annotations
import argparse
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint-url", required=True)
parser.add_argument("--sleep-seconds", type=float, default=0.0)
parser.add_argument("--exit-code", type=int, default=0)
parser.add_argument("--marker", type=Path, default=None)
args = parser.parse_args()
with urllib.request.urlopen(args.endpoint_url + "/healthz", timeout=2.0) as response:
    health = json.loads(response.read())
assert health["status"] == "READY"
if args.marker is not None:
    args.marker.write_text("ran", encoding="utf-8")
time.sleep(args.sleep_seconds)
raise SystemExit(args.exit_code)
"""


def _script_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "run_bounded_mage_benchmark_test", SCRIPT_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture_scripts(tmp_path: Path) -> tuple[Path, Path]:
    endpoint = tmp_path / "fake_endpoint.py"
    benchmark = tmp_path / "fake_benchmark.py"
    endpoint.write_text(_ENDPOINT_SOURCE, encoding="utf-8")
    benchmark.write_text(_BENCHMARK_SOURCE, encoding="utf-8")
    return endpoint, benchmark


def _argv(
    tmp_path: Path,
    *,
    endpoint_args: list[str],
    benchmark_args: list[str],
    port: int,
    startup_timeout: float = 2.0,
    benchmark_timeout: float = 2.0,
    overall_timeout: float = 5.0,
) -> tuple[list[str], Path]:
    endpoint, benchmark = _fixture_scripts(tmp_path)
    endpoint_json = _write_json(tmp_path / "endpoint-args.json", endpoint_args)
    benchmark_json = _write_json(tmp_path / "benchmark-args.json", benchmark_args)
    report = tmp_path / "lifecycle.json"
    return (
        [
            "--endpoint-args-json",
            str(endpoint_json),
            "--benchmark-args-json",
            str(benchmark_json),
            "--report-json",
            str(report),
            "--endpoint-script",
            str(endpoint),
            "--benchmark-script",
            str(benchmark),
            "--python-executable",
            sys.executable,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--startup-timeout-seconds",
            str(startup_timeout),
            "--benchmark-timeout-seconds",
            str(benchmark_timeout),
            "--overall-timeout-seconds",
            str(overall_timeout),
            "--shutdown-timeout-seconds",
            "0.25",
            "--health-poll-seconds",
            "0.05",
            "--health-request-timeout-seconds",
            "0.1",
        ],
        report,
    )


def test_accepts_utf8_bom_argument_vectors(tmp_path: Path) -> None:
    module = _script_module()
    path = tmp_path / "args.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(["--flag", "value"]).encode("utf-8"))

    assert module._load_argv(path, forbidden_flags=frozenset()) == ["--flag", "value"]


def test_success_owns_routes_and_reaps_endpoint(tmp_path: Path) -> None:
    module = _script_module()
    port = _free_port()
    marker = tmp_path / "benchmark-ran.txt"
    argv, report_path = _argv(
        tmp_path,
        endpoint_args=[],
        benchmark_args=["--marker", str(marker)],
        port=port,
    )

    assert module.main(argv) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_version"] == module.REPORT_VERSION
    assert report["outcome"] == "SUCCEEDED"
    assert marker.read_text(encoding="utf-8") == "ran"
    assert report["process_ownership"]["direct_popen_ownership"] is True
    assert report["process_ownership"]["shell_used"] is False
    assert report["process_ownership"]["endpoint_alive_at_readiness"] is True
    assert report["process_ownership"]["post_shutdown_port_closed"] is True
    assert report["endpoint"]["cleanup"]["reaped"] is True
    assert report["benchmark"]["cleanup"]["reaped"] is True
    assert report["endpoint"]["health_response"]["status"] == "READY"
    assert module._port_is_available("127.0.0.1", port) is True
    assert Path(report["configuration"]["endpoint_log"]).is_file()
    assert Path(report["configuration"]["benchmark_log"]).is_file()


def test_startup_timeout_writes_report_and_reaps_endpoint(tmp_path: Path) -> None:
    module = _script_module()
    port = _free_port()
    argv, report_path = _argv(
        tmp_path,
        endpoint_args=["--mode", "no-listen"],
        benchmark_args=[],
        port=port,
        startup_timeout=0.25,
        overall_timeout=2.0,
    )

    assert module.main(argv) == module.EXIT_TIMEOUT

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "STARTUP_TIMEOUT"
    assert report["benchmark"]["cleanup"]["action"] == "NOT_STARTED"
    assert report["endpoint"]["cleanup"]["reaped"] is True
    assert report["process_ownership"]["post_shutdown_port_closed"] is True
    assert module._port_is_available("127.0.0.1", port) is True


def test_benchmark_timeout_reaps_both_children(tmp_path: Path) -> None:
    module = _script_module()
    port = _free_port()
    argv, report_path = _argv(
        tmp_path,
        endpoint_args=[],
        benchmark_args=["--sleep-seconds", "60"],
        port=port,
        benchmark_timeout=0.25,
        # The intended assertion is the benchmark deadline, not a hosted-Windows
        # process-startup race against the hard lifecycle deadline.
        overall_timeout=10.0,
    )

    assert module.main(argv) == module.EXIT_TIMEOUT

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "BENCHMARK_TIMEOUT"
    assert report["benchmark"]["cleanup"]["reaped"] is True
    assert report["endpoint"]["cleanup"]["reaped"] is True
    assert report["process_ownership"]["post_shutdown_port_closed"] is True
    assert module._port_is_available("127.0.0.1", port) is True


def test_orchestrator_owned_flags_fail_before_spawn(tmp_path: Path) -> None:
    module = _script_module()
    port = _free_port()
    argv, report_path = _argv(
        tmp_path,
        endpoint_args=["--port", "9999"],
        benchmark_args=[],
        port=port,
    )

    assert module.main(argv) == module.EXIT_FAILURE

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ORCHESTRATION_ERROR"
    assert "orchestrator-owned option" in report["error"]["message"]
    assert report["process_ownership"]["endpoint_pid"] is None
    assert report["endpoint"]["cleanup"]["action"] == "NOT_STARTED"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "localhost"])
def test_non_numeric_or_non_loopback_host_is_rejected(tmp_path: Path, host: str) -> None:
    module = _script_module()
    endpoint, benchmark = _fixture_scripts(tmp_path)
    endpoint_json = _write_json(tmp_path / "endpoint-args.json", [])
    benchmark_json = _write_json(tmp_path / "benchmark-args.json", [])
    report = tmp_path / "lifecycle.json"
    arguments = module._parser().parse_args(
        [
            "--endpoint-args-json",
            str(endpoint_json),
            "--benchmark-args-json",
            str(benchmark_json),
            "--report-json",
            str(report),
            "--endpoint-script",
            str(endpoint),
            "--benchmark-script",
            str(benchmark),
            "--host",
            host,
            "--port",
            str(_free_port()),
        ]
    )

    exit_code, payload = module.run(arguments)

    assert exit_code == module.EXIT_FAILURE
    assert payload["outcome"] == "ORCHESTRATION_ERROR"
    assert payload["process_ownership"]["endpoint_pid"] is None
