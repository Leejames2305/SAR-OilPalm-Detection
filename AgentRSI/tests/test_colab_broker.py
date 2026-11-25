"""Unit and end-to-end tests for M8 Colab GPU Offload MCP Broker and stdio relay."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colab_broker.backend import CliColabBackend, MockColabBackend  # noqa: E402
from colab_broker.config import CONTAINER_IO_ROOT, ColabBrokerConfig  # noqa: E402
from colab_broker.guardrails import (  # noqa: E402
    resolve_staged_inputs,
    sanitize_cli_output,
    sanitized_colab_env,
    scan_script,
    validate_gpu,
    validate_output_paths,
    validate_pip_packages,
    verify_downloaded_outputs,
)
from colab_broker.server import ColabBrokerServer  # noqa: E402

RELAY_SCRIPT = AGENT_ROOT / "templates" / "agentrsi_mcp_relay.py"
BROKER_CLI = AGENT_ROOT / "scripts" / "colab_broker.py"


def _rpc_over_socket(sock_path: Path, requests: list[dict[str, object]]) -> list[dict[str, object]]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(sock_path))
        payload = "".join(json.dumps(r) + "\n" for r in requests).encode("utf-8")
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        out: list[dict[str, object]] = []
        with sock.makefile("r", encoding="utf-8") as reader:
            for line in reader:
                if line.strip():
                    out.append(json.loads(line))
        return out


class TestGuardrails(unittest.TestCase):
    def test_scan_script_rejects_forbidden_patterns(self) -> None:
        bad_snippets = [
            "from google.colab import auth\nauth.authenticate_user()",
            "import google.colab.auth",
            "authenticate_user()",
            "from google.colab import drive\ndrive.mount('/content/drive')",
            "import os; os.system('gcloud auth list')",
            "os.system('gsutil cp gs://bucket/file .')",
            "print('SAR-OilPalm-Detection')",
            "repo = 'sar-oilpalm'",
            "owner = 'Leejames2305'",
            "url = 'http://169.254.169.254/computeMetadata/v1/'",
            "host = 'metadata.google.internal'",
            "os.system('colab auth')",
            "open('/run/secrets/gcp-sa.json')",
            "os.environ['GOOGLE_APPLICATION_CREDENTIALS']",
            "   \n\t ",
        ]
        for snippet in bad_snippets:
            reasons = scan_script(snippet)
            self.assertTrue(reasons, f"Expected violation for snippet: {snippet!r}")

        self.assertEqual(scan_script("import torch\nprint(torch.cuda.is_available())"), [])

    def test_validate_gpu_allowlist_and_a100_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg_default = ColabBrokerConfig(
                socket_path=base / "s.sock",
                io_dir=base / "io",
                session_id="test-gpu",
            )
            self.assertEqual(validate_gpu("t4", cfg_default), "T4")
            self.assertEqual(validate_gpu("L4", cfg_default), "L4")
            with self.assertRaises(ValueError):
                validate_gpu("A100", cfg_default)
            with self.assertRaises(ValueError):
                validate_gpu("H100", cfg_default)

            cfg_a100 = ColabBrokerConfig(
                socket_path=base / "s.sock",
                io_dir=base / "io",
                session_id="test-gpu-a100",
                allow_a100=True,
            )
            self.assertEqual(validate_gpu("a100", cfg_a100), "A100")

    def test_validate_pip_packages(self) -> None:
        self.assertEqual(validate_pip_packages(None), [])
        self.assertEqual(
            validate_pip_packages(["torch==2.5.1", "timm[all]>=1.0,<2.0", "albumentations~=1.4"]),
            ["torch==2.5.1", "timm[all]>=1.0,<2.0", "albumentations~=1.4"],
        )
        bad = [
            "-e .",
            "-r requirements.txt",
            "--index-url=https://evil.example/simple",
            "https://evil.example/pkg.whl",
            "git+https://github.com/evil/repo.git",
            "file:///tmp/evil.whl",
            "../localpkg",
            "/tmp/local.whl",
            "pkg @ https://evil.example/pkg.whl",
            "pkg; python_version < '3.12'",
            "pkg==1.0/../../bad",
        ]
        for entry in bad:
            with self.assertRaises(ValueError, msg=f"Should reject pip spec: {entry!r}"):
                validate_pip_packages([entry])

    def test_staged_inputs_and_output_paths_block_traversal_symlinks_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            io_dir = Path(td) / "io"
            in_dir = io_dir / "in"
            in_dir.mkdir(parents=True)
            (in_dir / "sub").mkdir()
            good = in_dir / "sub" / "data.bin"
            good.write_bytes(b"hello" * 100)

            resolved = resolve_staged_inputs(io_dir, ["sub/data.bin"], max_upload_mb=1.0)
            self.assertEqual(resolved, [good.resolve()])

            for bad_rel in ["../secret.txt", "sub/../../etc/passwd", "/etc/passwd", "sub/"]:
                with self.assertRaises(ValueError):
                    resolve_staged_inputs(io_dir, [bad_rel], max_upload_mb=1.0)
                with self.assertRaises(ValueError):
                    validate_output_paths([bad_rel])

            secret_target = Path(td) / "outside.txt"
            secret_target.write_text("secret")
            link_path = in_dir / "escape.txt"
            link_path.symlink_to(secret_target)
            with self.assertRaises(ValueError):
                resolve_staged_inputs(io_dir, ["escape.txt"], max_upload_mb=1.0)

            # Size limit enforcement
            with self.assertRaises(ValueError):
                resolve_staged_inputs(io_dir, ["sub/data.bin"], max_upload_mb=0.0001)

    def test_sanitized_colab_env_strips_sensitive_vars(self) -> None:
        sensitive = {
            "API_KEY": "secret-api",
            "OPENAI_API_KEY": "secret-openai",
            "ANTHROPIC_AUTH_TOKEN": "secret-anth",
            "GOOGLE_APPLICATION_CREDENTIALS": "/run/secrets/gcp-sa.json",
            "AGENTRSI_VERIFIER_MCP_TOKEN": "secret-mcp",
            "COLAB_KEEP_ME": "ok-value",
        }
        with mock.patch.dict(os.environ, sensitive, clear=False):
            env = sanitized_colab_env()
        for key in (
            "API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "AGENTRSI_VERIFIER_MCP_TOKEN",
        ):
            self.assertNotIn(key, env)
        self.assertEqual(env.get("COLAB_KEEP_ME"), "ok-value")


class TestColabBrokerServerAndRelay(unittest.TestCase):
    def _make_config(self, td: str, **overrides: object) -> ColabBrokerConfig:
        base = Path(td)
        kwargs: dict[str, object] = {
            "socket_path": base / "stage" / "colab.sock",
            "io_dir": base / "stage" / "colab-io",
            "session_id": "261124-s001-open",
            "events_path": base / "events.jsonl",
        }
        kwargs.update(overrides)
        return ColabBrokerConfig(**kwargs)  # type: ignore[arg-type]

    def test_end_to_end_relay_mcp_handshake_run_fetch_stop_and_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._make_config(td)
            backend = MockColabBackend()
            with ColabBrokerServer(cfg, backend=backend) as srv:
                self.assertTrue(cfg.socket_path.exists())
                self.assertEqual(os.stat(cfg.socket_path.parent).st_mode & 0o777, 0o700)
                self.assertEqual(os.stat(cfg.socket_path).st_mode & 0o777, 0o600)

                staged = cfg.io_dir / "in" / "features" / "batch.json"
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_text(json.dumps({"values": [2, 4, 6]}))

                script = (
                    "import json\n"
                    "from pathlib import Path\n"
                    "data = json.loads(Path('features/batch.json').read_text())\n"
                    "out = {'sum': sum(data['values'])}\n"
                    "Path('preds').mkdir(exist_ok=True)\n"
                    "Path('preds/out.json').write_text(json.dumps(out))\n"
                    "print('OK_SUM=' + str(out['sum']))\n"
                )

                reqs = [
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "colab_run",
                            "arguments": {
                                "script": script,
                                "gpu": "T4",
                                "staged_inputs": ["features/batch.json"],
                                "output_paths": ["preds/out.json"],
                                "pip_packages": ["numpy>=2.0"],
                            },
                        },
                    },
                    {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "colab_status", "arguments": {}}},
                ]
                proc = subprocess.run(
                    [sys.executable, str(RELAY_SCRIPT), "--socket", str(cfg.socket_path)],
                    input="".join(json.dumps(r) + "\n" for r in reqs),
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
                self.assertEqual([r["id"] for r in responses], [1, 2, 3, 4, 5])

                # 1: initialize
                init_res = responses[0]["result"]
                self.assertEqual(init_res["protocolVersion"], "2025-03-26")
                self.assertEqual(init_res["serverInfo"]["name"], "agentrsi-colab")
                self.assertEqual(init_res["capabilities"], {"tools": {}})

                # 3: tools/list
                tool_names = [t["name"] for t in responses[2]["result"]["tools"]]
                self.assertEqual(tool_names, ["colab_run", "colab_status", "colab_fetch", "colab_stop"])

                # 4: colab_run
                run_call = responses[3]["result"]
                self.assertFalse(run_call["isError"])
                run_payload = json.loads(run_call["content"][0]["text"])
                self.assertTrue(run_payload["ok"])
                self.assertIn("OK_SUM=12", run_payload["stdout"])
                job_id = run_payload["job_id"]
                self.assertEqual(len(run_payload["outputs"]), 1)
                out_meta = run_payload["outputs"][0]
                self.assertEqual(out_meta["rel_path"], "preds/out.json")
                self.assertEqual(
                    out_meta["container_path"],
                    f"{CONTAINER_IO_ROOT}/out/{job_id}/preds/out.json",
                )
                downloaded_file = cfg.io_dir / "out" / job_id / "preds" / "out.json"
                self.assertTrue(downloaded_file.exists())
                expected_sha = hashlib.sha256(downloaded_file.read_bytes()).hexdigest()
                self.assertEqual(out_meta["sha256"], expected_sha)
                self.assertEqual(out_meta["bytes"], downloaded_file.stat().st_size)
                self.assertEqual(json.loads(downloaded_file.read_text()), {"sum": 12})

                # 5: colab_status
                status_payload = json.loads(responses[4]["result"]["content"][0]["text"])
                self.assertEqual(status_payload["active_gpu"], "T4")
                self.assertEqual(len(status_payload["jobs"]), 1)

                # colab_fetch (with and without logs) & colab_stop
                follow_up = _rpc_over_socket(
                    cfg.socket_path,
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 6,
                            "method": "tools/call",
                            "params": {"name": "colab_fetch", "arguments": {"job_id": job_id, "include_logs": False}},
                        },
                        {
                            "jsonrpc": "2.0",
                            "id": 7,
                            "method": "tools/call",
                            "params": {"name": "colab_stop", "arguments": {"job_id": job_id}},
                        },
                    ],
                )
                fetch_payload = json.loads(follow_up[0]["result"]["content"][0]["text"])
                self.assertNotIn("stdout", fetch_payload)
                self.assertEqual(fetch_payload["job_id"], job_id)

                stop_payload = json.loads(follow_up[1]["result"]["content"][0]["text"])
                self.assertTrue(stop_payload["stopped"])
                self.assertEqual(len(backend.started_vms), 1)
                self.assertEqual(len(backend.stopped_vms), 1)
                self.assertIsNotNone(srv)

            # After context exit (shutdown), socket & io_dir are cleaned up
            self.assertFalse(cfg.socket_path.exists())
            self.assertFalse(cfg.io_dir.exists())

            events = [json.loads(line) for line in cfg.events_path.read_text().splitlines() if line.strip()]  # type: ignore[union-attr]
            event_types = [ev["type"] for ev in events]
            self.assertIn("colab_vm_start", event_types)
            self.assertIn("colab_job_run", event_types)
            self.assertIn("colab_vm_stop", event_types)

    def test_max_one_vm_busy_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._make_config(td)
            backend = MockColabBackend()
            with ColabBrokerServer(cfg, backend=backend):
                first_result: list[dict[str, object]] = []

                def run_slow() -> None:
                    res = _rpc_over_socket(
                        cfg.socket_path,
                        [{
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "colab_run",
                                "arguments": {"script": "import time\ntime.sleep(0.5)\nprint('done')"},
                            },
                        }],
                    )
                    first_result.extend(res)

                t = threading.Thread(target=run_slow)
                t.start()
                time.sleep(0.12)
                second = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "colab_run",
                            "arguments": {"script": "print('concurrent')"},
                        },
                    }],
                )
                t.join(timeout=5)
                self.assertEqual(len(first_result), 1)
                self.assertFalse(first_result[0]["result"]["isError"])  # type: ignore[index]
                self.assertTrue(second[0]["result"]["isError"])  # type: ignore[index]
                err_payload = json.loads(second[0]["result"]["content"][0]["text"])  # type: ignore[index]
                self.assertEqual(err_payload["error"], "VM_BUSY")

    def test_gpu_wallclock_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # 0.004 min == 0.24 sec wall-clock budget
            cfg = self._make_config(td, max_session_gpu_min=0.004)
            backend = MockColabBackend()
            with ColabBrokerServer(cfg, backend=backend):
                r1 = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "colab_run",
                            "arguments": {"script": "import time\ntime.sleep(0.28)\nprint('ran')"},
                        },
                    }],
                )
                # First run may time out on clamped budget or exhaust budget right after
                self.assertEqual(len(backend.stopped_vms), 1)

                r2 = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "colab_run",
                            "arguments": {"script": "print('second')"},
                        },
                    }],
                )
                self.assertTrue(r2[0]["result"]["isError"])  # type: ignore[index]
                payload = json.loads(r2[0]["result"]["content"][0]["text"])  # type: ignore[index]
                self.assertEqual(payload["error"], "GPU_BUDGET_EXHAUSTED")
                self.assertTrue(r1)

    def test_idle_teardown_and_shutdown_stops_active_vm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._make_config(td, idle_teardown_sec=0.15)
            backend = MockColabBackend()
            with ColabBrokerServer(cfg, backend=backend):
                res = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "print('hi')"}},
                    }],
                )
                self.assertFalse(res[0]["result"]["isError"])  # type: ignore[index]
                self.assertEqual(len(backend.started_vms), 1)
                self.assertEqual(len(backend.stopped_vms), 0)

                # Wait for idle watchdog to tear down VM
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline and not backend.stopped_vms:
                    time.sleep(0.05)
                self.assertEqual(len(backend.stopped_vms), 1)

                # Start another job and exit context without explicit colab_stop -> shutdown() stops it
                res2 = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "print('hi2')"}},
                    }],
                )
                self.assertFalse(res2[0]["result"]["isError"])  # type: ignore[index]
            self.assertEqual(len(backend.stopped_vms), 2)

    def test_download_symlink_and_size_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out_job"
            out_dir.mkdir()
            secret = Path(td) / "host_secret.txt"
            secret.write_text("top-secret")
            (out_dir / "leak.txt").symlink_to(secret)
            with self.assertRaises(ValueError):
                verify_downloaded_outputs(out_dir, ["leak.txt"], 10.0, CONTAINER_IO_ROOT, "job-1")

            big = out_dir / "big.bin"
            big.write_bytes(b"x" * 2048)
            with self.assertRaises(ValueError):
                verify_downloaded_outputs(out_dir, ["big.bin"], 0.0005, CONTAINER_IO_ROOT, "job-1")
            self.assertFalse(big.exists())

    def test_check_auth_cli_and_backend(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            ok, msg = CliColabBackend().check_auth_ready()
            self.assertFalse(ok)
            self.assertIn("google-colab-cli", msg)

        with mock.patch("shutil.which", return_value="/usr/local/bin/colab"):
            with mock.patch(
                "subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess(["colab", "--help"], 0, "usage", ""),
                    subprocess.CompletedProcess(["colab", "ls"], 1, "", "Error: 401 unauthorized; run colab auth"),
                ],
            ):
                ok, msg = CliColabBackend().check_auth_ready()
                self.assertFalse(ok)
                self.assertIn("colab auth", msg)

            with mock.patch(
                "subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess(["colab", "--help"], 0, "usage", ""),
                    subprocess.CompletedProcess(["colab", "ls"], 0, "file1\n", ""),
                ],
            ):
                ok, _ = CliColabBackend().check_auth_ready()
                self.assertTrue(ok)

        # CLI --check-auth --mock
        cli_ok = subprocess.run(
            [sys.executable, str(BROKER_CLI), "--check-auth", "--mock"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(cli_ok.returncode, 0)

    def test_broker_cli_serve_and_sigterm_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            sock_path = base / "stage" / "broker.sock"
            io_dir = base / "stage" / "io"
            ready_file = base / "ready.json"

            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(BROKER_CLI),
                    "--mock",
                    "--socket",
                    str(sock_path),
                    "--io-dir",
                    str(io_dir),
                    "--session-id",
                    "cli-smoke",
                    "--write-ready-file",
                    str(ready_file),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not ready_file.exists():
                    time.sleep(0.05)
                self.assertTrue(ready_file.exists(), "Broker CLI did not write ready file")

                relay_res = subprocess.run(
                    [sys.executable, str(RELAY_SCRIPT), "--socket", str(sock_path)],
                    input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n",
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(relay_res.returncode, 0, msg=relay_res.stderr)
                resp = json.loads(relay_res.stdout.strip())
                self.assertEqual(len(resp["result"]["tools"]), 4)
            finally:
                proc.send_signal(signal.SIGTERM)
                proc.communicate(timeout=5)
                rc = proc.returncode
            self.assertEqual(rc, 0)
            self.assertFalse(sock_path.exists())
            self.assertFalse(io_dir.exists())

    def test_gpu_switch_ephemeral_vm_and_blocked_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._make_config(td, ephemeral_vm=False)
            backend = MockColabBackend()
            with ColabBrokerServer(cfg, backend=backend):
                # Blocked A100 and blocked script emit colab_blocked
                r_gpu = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "print(1)", "gpu": "A100"}},
                    }],
                )
                self.assertTrue(r_gpu[0]["result"]["isError"])  # type: ignore[index]
                self.assertEqual(
                    json.loads(r_gpu[0]["result"]["content"][0]["text"])["error"],  # type: ignore[index]
                    "GPU_NOT_ALLOWED",
                )

                r_bad = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "import google.colab.auth"}},
                    }],
                )
                self.assertTrue(r_bad[0]["result"]["isError"])  # type: ignore[index]
                self.assertEqual(
                    json.loads(r_bad[0]["result"]["content"][0]["text"])["error"],  # type: ignore[index]
                    "FORBIDDEN_SCRIPT_PATTERN",
                )

                # Run on T4 then switch to L4 -> stops T4 VM first
                _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "print('t4')", "gpu": "T4"}},
                    }],
                )
                _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "print('l4')", "gpu": "L4"}},
                    }],
                )
                self.assertEqual(len(backend.started_vms), 2)
                self.assertEqual(len(backend.stopped_vms), 1)
                self.assertEqual(backend.started_vm_gpus[backend.started_vms[-1]], "L4")

            events = [json.loads(line) for line in cfg.events_path.read_text().splitlines() if line.strip()]  # type: ignore[union-attr]
            self.assertGreaterEqual(sum(1 for e in events if e["type"] == "colab_blocked"), 2)

        with tempfile.TemporaryDirectory() as td:
            cfg_eph = self._make_config(td, ephemeral_vm=True)
            backend_eph = MockColabBackend()
            with ColabBrokerServer(cfg_eph, backend=backend_eph):
                _rpc_over_socket(
                    cfg_eph.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "colab_run", "arguments": {"script": "print('eph')"}},
                    }],
                )
                self.assertEqual(len(backend_eph.stopped_vms), 1)

    def test_cli_backend_command_shapes_and_sanitized_env(self) -> None:
        calls: list[tuple[list[str], dict[str, str]]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append((cmd, dict(kwargs.get("env") or {})))  # type: ignore[arg-type]
            return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

        class _FakePopen:
            def __init__(self, cmd: list[str], **kwargs: object) -> None:
                calls.append((cmd, dict(kwargs.get("env") or {})))  # type: ignore[arg-type]
                self.pid = 99999
                self.returncode = 0

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return ("ok\n", "")

            def poll(self) -> int | None:
                return self.returncode

        with tempfile.TemporaryDirectory() as td:
            local_in = Path(td) / "in.bin"
            local_in.write_bytes(b"123")
            local_out = Path(td) / "out.bin"

            with mock.patch.dict(os.environ, {"API_KEY": "secret", "AGENTRSI_VERIFIER_MCP_TOKEN": "tok"}, clear=False):
                with mock.patch("shutil.which", return_value="/usr/bin/colab"):
                    with (
                        mock.patch("subprocess.run", side_effect=fake_run),
                        mock.patch("subprocess.Popen", side_effect=_FakePopen),
                    ):
                        be = CliColabBackend()
                        be.start_vm("agentrsi-s1", "T4")
                        be.upload_file("agentrsi-s1", local_in, "/content/in.bin")
                        be.upload_file("agentrsi-s1", local_in, "/content/nested/in.bin")
                        res = be.exec_script("agentrsi-s1", "print(1)", 30.0, pip_packages=["scipy>=1.10"])
                        be.download_file("agentrsi-s1", "/content/out.bin", local_out)
                        be.stop_vm("agentrsi-s1")

        self.assertEqual(res.returncode, 0)
        self.assertEqual(calls[0][0], ["/usr/bin/colab", "new", "-s", "agentrsi-s1", "--gpu", "T4"])
        self.assertEqual(calls[1][0][:4], ["/usr/bin/colab", "upload", "-s", "agentrsi-s1"])
        # Nested upload creates parent directory first via `colab exec -s ... -f ...` then uploads
        self.assertEqual(calls[2][0][:4], ["/usr/bin/colab", "exec", "-s", "agentrsi-s1"])
        self.assertEqual(calls[3][0][:4], ["/usr/bin/colab", "upload", "-s", "agentrsi-s1"])
        self.assertEqual(calls[4][0][:4], ["/usr/bin/colab", "exec", "-s", "agentrsi-s1"])
        self.assertEqual(calls[5][0][:4], ["/usr/bin/colab", "download", "-s", "agentrsi-s1"])
        self.assertEqual(calls[6][0], ["/usr/bin/colab", "stop", "-s", "agentrsi-s1"])
        for _, env in calls:
            self.assertNotIn("API_KEY", env)
            self.assertNotIn("AGENTRSI_VERIFIER_MCP_TOKEN", env)

    def test_sanitize_cli_output_redacts_proxy_urls_and_tokens(self) -> None:
        raw = (
            "[colab] Upload failed: 500 Server Error for url: "
            "https://8080-gpu-l4.prod.r.colab.dev/api/contents/content/s009/chips.npz"
            "?authuser=0&token=SUPER_SECRET_TOKEN_123 "
            "Authorization: Bearer SECRET_BEARER_XYZ"
        )
        cleaned = sanitize_cli_output(raw)
        self.assertNotIn("SUPER_SECRET_TOKEN_123", cleaned)
        self.assertNotIn("SECRET_BEARER_XYZ", cleaned)
        self.assertIn("https://8080-gpu-l4.prod.r.colab.dev/api/contents/content/s009/chips.npz?<REDACTED>", cleaned)

        # CliColabBackend upload error scrubs leaked URL query strings
        def fail_upload(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if cmd[1] == "upload":
                return subprocess.CompletedProcess(cmd, 1, "", raw)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.bin"
            p.write_bytes(b"x")
            with mock.patch("shutil.which", return_value="/usr/bin/colab"):
                with mock.patch("subprocess.run", side_effect=fail_upload):
                    with self.assertRaises(RuntimeError) as ctx:
                        CliColabBackend().upload_file("agentrsi-s1", p, "/content/a.bin")
        self.assertNotIn("SUPER_SECRET_TOKEN_123", str(ctx.exception))
        self.assertIn("<REDACTED>", str(ctx.exception))

    def test_async_bounded_wait_status_fetch_and_preemptive_stop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._make_config(td, sync_wait_sec=0.15)
            backend = MockColabBackend()
            with ColabBrokerServer(cfg, backend=backend):
                # 1. Job exceeding sync_wait_sec returns status='running', then completes via status/fetch
                script = (
                    "import time\n"
                    "from pathlib import Path\n"
                    "time.sleep(0.45)\n"
                    "Path('async_out.txt').write_text('hello-async')\n"
                    "print('ASYNC_DONE')\n"
                )
                r1 = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "colab_run",
                            "arguments": {"script": script, "output_paths": ["async_out.txt"]},
                        },
                    }],
                )
                self.assertFalse(r1[0]["result"]["isError"])  # type: ignore[index]
                p1 = json.loads(r1[0]["result"]["content"][0]["text"])  # type: ignore[index]
                self.assertEqual(p1["status"], "running")
                job_id = p1["job_id"]

                # Poll colab_status(job_id) until completed
                deadline = time.monotonic() + 5.0
                final_status = "running"
                while time.monotonic() < deadline:
                    st = _rpc_over_socket(
                        cfg.socket_path,
                        [{
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {"name": "colab_status", "arguments": {"job_id": job_id}},
                        }],
                    )
                    st_p = json.loads(st[0]["result"]["content"][0]["text"])  # type: ignore[index]
                    final_status = st_p["status"]
                    if final_status != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(final_status, "completed")

                fetched = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "colab_fetch", "arguments": {"job_id": job_id}},
                    }],
                )
                fp = json.loads(fetched[0]["result"]["content"][0]["text"])  # type: ignore[index]
                self.assertTrue(fp["ok"])
                self.assertIn("ASYNC_DONE", fp["stdout"])
                self.assertEqual(len(fp["outputs"]), 1)

                # 2. Long stuck job can be preemptively cancelled & VM stopped immediately by colab_stop
                t0 = time.monotonic()
                r2 = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "colab_run",
                            "arguments": {"script": "import time\ntime.sleep(30)\n", "wait_sec": 0.1},
                        },
                    }],
                )
                p2 = json.loads(r2[0]["result"]["content"][0]["text"])  # type: ignore[index]
                self.assertEqual(p2["status"], "running")
                job2 = p2["job_id"]

                r_stop = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {"name": "colab_stop", "arguments": {"job_id": job2}},
                    }],
                )
                stop_p = json.loads(r_stop[0]["result"]["content"][0]["text"])  # type: ignore[index]
                elapsed = time.monotonic() - t0
                self.assertTrue(stop_p["ok"])
                self.assertTrue(stop_p["stopped"])
                self.assertEqual(stop_p["cancelled_job_id"], job2)
                self.assertLess(elapsed, 4.0)

                st2 = _rpc_over_socket(
                    cfg.socket_path,
                    [{
                        "jsonrpc": "2.0",
                        "id": 6,
                        "method": "tools/call",
                        "params": {"name": "colab_status", "arguments": {"job_id": job2}},
                    }],
                )
                st2_p = json.loads(st2[0]["result"]["content"][0]["text"])  # type: ignore[index]
                self.assertEqual(st2_p["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
