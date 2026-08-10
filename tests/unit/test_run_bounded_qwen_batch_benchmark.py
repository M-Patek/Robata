from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_bounded_qwen_batch_benchmark.py"


def _module() -> ModuleType:
    name = f"run_bounded_qwen_batch_benchmark_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_script(path: Path) -> Path:
    script = path / "child.py"
    script.write_text(
        """
import argparse
import pathlib
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--marker', type=pathlib.Path)
p.add_argument('--sleep-seconds', type=float, default=0.0)
p.add_argument('--exit-code', type=int, default=0)
p.add_argument('--token')
a=p.parse_args()
print('child-start', flush=True)
if a.marker is not None:
    a.marker.write_text('ran', encoding='utf-8')
if a.sleep_seconds:
    time.sleep(a.sleep_seconds)
print('child-end', flush=True)
raise SystemExit(a.exit_code)
""".lstrip(),
        encoding="utf-8",
    )
    return script


def _write_args(path: Path, values: list[str], *, bom: bool = False) -> Path:
    body = json.dumps(values).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + body)
    return path


def _argv(
    tmp_path: Path,
    *,
    child_args: list[str],
    benchmark_timeout: float = 2.0,
    overall_timeout: float = 3.0,
) -> tuple[list[str], Path]:
    report = tmp_path / "lifecycle.json"
    return (
        [
            "--benchmark-args-json",
            str(_write_args(tmp_path / "args.json", child_args)),
            "--report-json",
            str(report),
            "--benchmark-script",
            str(_fixture_script(tmp_path)),
            "--python-executable",
            sys.executable,
            "--benchmark-timeout-seconds",
            str(benchmark_timeout),
            "--overall-timeout-seconds",
            str(overall_timeout),
            "--shutdown-timeout-seconds",
            "0.5",
            "--poll-seconds",
            "0.02",
        ],
        report,
    )


def test_accepts_utf8_bom_argument_vector(tmp_path: Path) -> None:
    module = _module()
    path = _write_args(tmp_path / "args.json", ["--flag", "value"], bom=True)

    assert module._load_argv(path) == ["--flag", "value"]


def test_success_owns_and_reaps_finite_child(tmp_path: Path) -> None:
    module = _module()
    marker = tmp_path / "marker.txt"
    argv, report_path = _argv(tmp_path, child_args=["--marker", str(marker)])

    assert module.main(argv) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_version"] == module.REPORT_VERSION
    assert report["outcome"] == "SUCCEEDED"
    assert report["production_eligible"] is False
    assert marker.read_text(encoding="utf-8") == "ran"
    assert report["process_ownership"]["direct_popen_ownership"] is True
    assert report["process_ownership"]["shell_used"] is False
    assert report["benchmark"]["cleanup"]["reaped"] is True
    assert report["benchmark"]["log"]["exists"] is True
    assert report["benchmark"]["log"]["byte_count"] > 0
    assert len(report["benchmark"]["log"]["exact_sha256"]) == 64


def test_timeout_is_explicit_and_reaps_child(tmp_path: Path) -> None:
    module = _module()
    argv, report_path = _argv(
        tmp_path,
        child_args=["--sleep-seconds", "60"],
        benchmark_timeout=0.2,
        overall_timeout=1.0,
    )

    assert module.main(argv) == module.EXIT_TIMEOUT

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "BENCHMARK_TIMEOUT"
    assert report["benchmark"]["cleanup"]["reaped"] is True
    assert report["benchmark"]["cleanup"]["action"] in {"TERMINATED", "KILLED"}


def test_nonzero_child_is_a_benchmark_failure_not_success(tmp_path: Path) -> None:
    module = _module()
    argv, report_path = _argv(tmp_path, child_args=["--exit-code", "7"])

    assert module.main(argv) == module.EXIT_FAILURE

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "BENCHMARK_FAILED"
    assert report["benchmark"]["exit_code"] == 7
    assert report["benchmark"]["cleanup"]["reaped"] is True


def test_invalid_argument_vector_fails_before_spawn(tmp_path: Path) -> None:
    module = _module()
    report_path = tmp_path / "lifecycle.json"
    args_path = tmp_path / "args.json"
    args_path.write_text('{"not":"a-list"}', encoding="utf-8")

    exit_code = module.main(
        [
            "--benchmark-args-json",
            str(args_path),
            "--report-json",
            str(report_path),
            "--benchmark-script",
            str(_fixture_script(tmp_path)),
            "--python-executable",
            sys.executable,
            "--benchmark-timeout-seconds",
            "1",
            "--overall-timeout-seconds",
            "2",
            "--shutdown-timeout-seconds",
            "0.5",
        ]
    )

    assert exit_code == module.EXIT_FAILURE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ORCHESTRATION_ERROR"
    assert report["process_ownership"]["benchmark_pid"] is None


def test_command_report_redacts_secret_values(tmp_path: Path) -> None:
    module = _module()
    argv, report_path = _argv(
        tmp_path,
        child_args=["--token", "super-secret", "--exit-code", "7"],
    )

    assert module.main(argv) == module.EXIT_FAILURE

    report_text = report_path.read_text(encoding="utf-8")
    assert "super-secret" not in report_text
    report = json.loads(report_text)
    assert "<redacted>" in report["benchmark"]["command"]


def test_deadline_configuration_must_fit_shutdown_budget(tmp_path: Path) -> None:
    module = _module()
    argv, report_path = _argv(
        tmp_path,
        child_args=[],
        benchmark_timeout=1.0,
        overall_timeout=1.2,
    )
    shutdown_index = argv.index("--shutdown-timeout-seconds") + 1
    argv[shutdown_index] = "0.5"

    assert module.main(argv) == module.EXIT_FAILURE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "ORCHESTRATION_ERROR"
    assert "fit inside" in report["error"]["message"]


@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_positive_float_rejects_invalid_values(value: str) -> None:
    module = _module()
    with pytest.raises((ValueError, SystemExit)):
        module._parser().parse_args(
            [
                "--benchmark-args-json",
                "args.json",
                "--report-json",
                "report.json",
                "--benchmark-timeout-seconds",
                value,
            ]
        )
