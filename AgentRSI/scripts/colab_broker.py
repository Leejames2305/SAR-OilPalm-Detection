"""CLI entrypoint for the host-side AgentRSI Colab GPU offload MCP broker."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import threading
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colab_broker.backend import CliColabBackend, MockColabBackend  # noqa: E402
from colab_broker.config import ColabBrokerConfig  # noqa: E402
from colab_broker.server import ColabBrokerServer  # noqa: E402


def _write_ready_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".colab_ready_", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload))
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run or check the AgentRSI host Colab MCP broker.")
    ap.add_argument("--check-auth", action="store_true", help="Check host google-colab-cli auth readiness and exit")
    ap.add_argument("--socket", type=Path, default=None, help="AF_UNIX socket path for the broker")
    ap.add_argument("--io-dir", type=Path, default=None, help="Host staging I/O directory (contains in/ and out/)")
    ap.add_argument("--session-id", default=None, help="Session identifier")
    ap.add_argument("--mock", action="store_true", help="Use offline MockColabBackend")
    ap.add_argument("--events-path", type=Path, default=None, help="Optional JSONL events ledger path")
    ap.add_argument("--epoch", type=int, default=1, help="Current epoch number for event records")
    ap.add_argument("--allow-a100", action="store_true", help="Allow A100 GPU requests")
    ap.add_argument("--max-job-timeout-min", type=float, default=30.0, help="Per-job timeout cap (minutes)")
    ap.add_argument("--max-session-gpu-min", type=float, default=60.0, help="Session GPU wall-clock cap (minutes)")
    ap.add_argument("--idle-teardown-sec", type=float, default=300.0, help="Idle VM teardown timeout (seconds)")
    ap.add_argument("--ephemeral-vm", action="store_true", help="Tear down VM after every colab_run job")
    ap.add_argument("--max-upload-mb", type=float, default=256.0, help="Max staged upload size per job (MB)")
    ap.add_argument("--max-download-mb", type=float, default=128.0, help="Max downloaded output size per job (MB)")
    ap.add_argument("--colab-bin", default="colab", help="Path/name of google-colab-cli binary")
    ap.add_argument("--write-ready-file", type=Path, default=None, help="Write JSON readiness marker when listening")
    args = ap.parse_args(argv)

    if args.check_auth:
        backend = MockColabBackend() if args.mock else CliColabBackend(colab_bin=args.colab_bin)
        ok, message = backend.check_auth_ready()
        stream = sys.stdout if ok else sys.stderr
        print(message, file=stream)
        return 0 if ok else 1

    if args.socket is None or args.io_dir is None or not args.session_id:
        ap.error("--socket, --io-dir, and --session-id are required when serving the broker")

    cfg = ColabBrokerConfig(
        socket_path=args.socket,
        io_dir=args.io_dir,
        session_id=args.session_id,
        allow_a100=args.allow_a100,
        max_job_timeout_min=args.max_job_timeout_min,
        max_session_gpu_min=args.max_session_gpu_min,
        idle_teardown_sec=args.idle_teardown_sec,
        ephemeral_vm=args.ephemeral_vm,
        max_upload_mb=args.max_upload_mb,
        max_download_mb=args.max_download_mb,
        colab_bin=args.colab_bin,
        events_path=args.events_path,
        epoch=args.epoch,
    )
    backend = MockColabBackend() if args.mock else CliColabBackend(colab_bin=args.colab_bin)
    server = ColabBrokerServer(cfg, backend=backend)

    stop_event = threading.Event()

    def _handle_sig(signum: int, frame: object) -> None:
        stop_event.set()

    prev_int = signal.signal(signal.SIGINT, _handle_sig)
    prev_term = signal.signal(signal.SIGTERM, _handle_sig)
    try:
        server.start_background()
        server.wait_ready(timeout=5.0)
        if args.write_ready_file is not None:
            _write_ready_file(
                args.write_ready_file,
                {
                    "ok": True,
                    "session_id": cfg.session_id,
                    "socket_path": str(cfg.socket_path),
                    "io_dir": str(cfg.io_dir),
                    "pid": os.getpid(),
                },
            )
        while not stop_event.wait(timeout=0.5):
            pass
    finally:
        server.shutdown()
        if args.write_ready_file is not None:
            args.write_ready_file.unlink(missing_ok=True)
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
