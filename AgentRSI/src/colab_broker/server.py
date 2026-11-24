"""Unix-socket JSON-RPC 2.0 MCP server for host-side Colab GPU offloading."""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from colab_broker.backend import CliColabBackend, ColabBackend
from colab_broker.config import ColabBrokerConfig
from colab_broker.guardrails import (
    resolve_staged_inputs,
    scan_script,
    validate_gpu,
    validate_output_paths,
    validate_pip_packages,
    verify_downloaded_outputs,
)

MCP_PROTOCOL_DEFAULT = "2025-03-26"
SERVER_INFO = {"name": "agentrsi-colab", "version": "0.1.0"}

_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "colab_run",
        "description": (
            "Run a Python script on an ephemeral/session Colab GPU VM. "
            "Inputs must be staged under /run/agentrsi/colab-io/in and requested outputs "
            "are downloaded to /run/agentrsi/colab-io/out/<job_id>/."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Python script to execute on Colab."},
                "gpu": {"type": "string", "default": "T4", "description": "GPU type (e.g. T4, L4)."},
                "timeout_min": {
                    "type": "number",
                    "default": 15.0,
                    "description": "Execution timeout in minutes (clamped by broker policy).",
                },
                "staged_inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relative paths under /run/agentrsi/colab-io/in to upload to /content/.",
                },
                "output_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relative paths under /content/ on Colab to download after execution.",
                },
                "pip_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional PyPI package specifications to install before running.",
                },
            },
            "required": ["script"],
        },
    },
    {
        "name": "colab_status",
        "description": "Inspect active Colab VM state, GPU wall-clock budget, and job status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": ["string", "null"],
                    "description": "Optional job_id to inspect; omit for overall broker status.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "colab_fetch",
        "description": "Fetch metadata, container output paths, and logs for a completed Colab job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID returned by colab_run."},
                "include_logs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include stdout/stderr in the response.",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "colab_stop",
        "description": "Terminate the active Colab VM runtime immediately and release GPU resources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": ["string", "null"],
                    "description": "Optional associated job_id (active VM is stopped regardless).",
                },
            },
            "required": [],
        },
    },
]


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structuredContent": payload,
        "isError": bool(is_error),
    }


def _sanitize_vm_name(session_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9-]+", "-", session_id).strip("-").lower()
    return f"agentrsi-{cleaned or 'session'}"[:48]


class ColabBrokerServer:
    """Host-side Unix-socket MCP broker managing Colab VM lifecycle and guardrails."""

    def __init__(self, cfg: ColabBrokerConfig, backend: ColabBackend | None = None) -> None:
        self.cfg = cfg
        self.backend: ColabBackend = backend if backend is not None else CliColabBackend(colab_bin=cfg.colab_bin)

        # Lock ordering invariant: _vm_lock -> _state_lock -> _event_lock
        self._vm_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._clients_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._client_socks: set[socket.socket] = set()
        self._client_threads: list[threading.Thread] = []

        self._shutting_down = False
        self._shutdown_done = False

        self._job_running = False
        self._active_vm_name: str | None = None
        self._active_vm_gpu: str | None = None
        self._vm_started_at: float | None = None
        self._accumulated_vm_sec = 0.0
        self._last_activity_at = time.monotonic()
        self._job_seq = 0
        self._jobs: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "ColabBrokerServer":
        self.start_background()
        self.wait_ready()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown()

    def _emit_event(
        self,
        event_type: str,
        summary: str,
        *,
        values: dict[str, Any] | None = None,
        dedup_id: str | None = None,
    ) -> None:
        if self.cfg.events_path is None:
            return
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": self.cfg.session_id,
            "epoch": int(self.cfg.epoch),
            "type": event_type,
            "summary": summary,
            "ref": f"sessions/{self.cfg.session_id}/meta.json",
        }
        if values is not None:
            record["values"] = values
        if dedup_id is not None:
            record["dedup_id"] = dedup_id
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._event_lock:
            self.cfg.events_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.events_path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def _gpu_seconds_used_locked(self) -> float:
        used = self._accumulated_vm_sec
        if self._active_vm_name is not None and self._vm_started_at is not None:
            used += max(0.0, time.monotonic() - self._vm_started_at)
        return used

    def _gpu_seconds_used(self) -> float:
        with self._state_lock:
            return self._gpu_seconds_used_locked()

    def _stop_active_vm_under_vm_lock(self, *, reason: str) -> tuple[bool, str | None]:
        with self._state_lock:
            vm_name = self._active_vm_name
            gpu = self._active_vm_gpu
            started_at = self._vm_started_at
            if vm_name is None:
                return False, None
            if started_at is not None:
                self._accumulated_vm_sec += max(0.0, time.monotonic() - started_at)
            self._active_vm_name = None
            self._active_vm_gpu = None
            self._vm_started_at = None

        err_msg: str | None = None
        try:
            self.backend.stop_vm(vm_name)
        except Exception as exc:
            err_msg = str(exc)

        self._emit_event(
            "colab_vm_stop",
            f"Stopped Colab VM {vm_name} ({reason})",
            values={"vm_name": vm_name, "gpu": gpu, "reason": reason, "error": err_msg},
        )
        return True, vm_name

    def _prepare_io_and_socket(self) -> socket.socket:
        sock_parent = self.cfg.socket_path.parent
        sock_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(sock_parent, 0o700)

        io_dir = self.cfg.io_dir
        io_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        os.chmod(io_dir, 0o755)
        for sub in ("in", "out"):
            sub_dir = io_dir / sub
            sub_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            os.chmod(sub_dir, 0o755)

        if self.cfg.socket_path.exists() or self.cfg.socket_path.is_symlink():
            self.cfg.socket_path.unlink()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.cfg.socket_path))
        os.chmod(self.cfg.socket_path, 0o600)
        srv.listen(16)
        srv.settimeout(0.5)
        return srv

    def start_background(self) -> None:
        """Bind the AF_UNIX socket and start accept + idle-watchdog daemon threads."""
        with self._state_lock:
            if self._server_sock is not None:
                return
            self._shutting_down = False
            self._shutdown_done = False
            self._stop_event.clear()
            self._ready_event.clear()
            self._server_sock = self._prepare_io_and_socket()

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name=f"colab-broker-accept-{self.cfg.session_id}",
            daemon=True,
        )
        self._watchdog_thread = threading.Thread(
            target=self._idle_watchdog_loop,
            name=f"colab-broker-watchdog-{self.cfg.session_id}",
            daemon=True,
        )
        self._accept_thread.start()
        self._watchdog_thread.start()
        self._ready_event.set()

    def wait_ready(self, timeout: float = 5.0) -> None:
        """Block until the broker socket is listening and present on disk."""
        deadline = time.monotonic() + float(timeout)
        if not self._ready_event.wait(timeout=max(0.0, deadline - time.monotonic())):
            raise TimeoutError("ColabBrokerServer did not become ready in time")
        while time.monotonic() < deadline:
            if self.cfg.socket_path.exists():
                return
            time.sleep(0.02)
        if not self.cfg.socket_path.exists():
            raise TimeoutError(f"Broker socket missing: {self.cfg.socket_path}")

    def shutdown(self) -> None:
        """Idempotently stop threads, tear down active Colab VM, and unlink socket/io_dir."""
        with self._state_lock:
            if self._shutdown_done:
                return
            self._shutting_down = True
            srv = self._server_sock
            self._server_sock = None
        self._stop_event.set()

        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass

        with self._clients_lock:
            clients = list(self._client_socks)
        for cs in clients:
            try:
                cs.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                cs.close()
            except OSError:
                pass

        cur = threading.current_thread()
        for t in (self._accept_thread, self._watchdog_thread):
            if t is not None and t.is_alive() and t is not cur:
                t.join(timeout=2.0)

        with self._clients_lock:
            threads = list(self._client_threads)
        for t in threads:
            if t.is_alive() and t is not cur:
                t.join(timeout=1.0)

        with self._vm_lock:
            self._stop_active_vm_under_vm_lock(reason="broker_shutdown")

        self.cfg.socket_path.unlink(missing_ok=True)
        shutil.rmtree(self.cfg.io_dir, ignore_errors=True)

        with self._state_lock:
            self._shutdown_done = True

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._state_lock:
                srv = self._server_sock
            if srv is None:
                break
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with self._clients_lock:
                self._client_socks.add(conn)
                t = threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    name="colab-broker-client",
                    daemon=True,
                )
                self._client_threads = [th for th in self._client_threads if th.is_alive()]
                self._client_threads.append(t)
                t.start()

    def _idle_watchdog_loop(self) -> None:
        poll_interval = min(1.0, max(0.05, float(self.cfg.idle_teardown_sec) / 4.0))
        while not self._stop_event.wait(timeout=poll_interval):
            should_stop = False
            with self._state_lock:
                if (
                    not self._shutting_down
                    and not self._job_running
                    and self._active_vm_name is not None
                    and (time.monotonic() - self._last_activity_at) >= float(self.cfg.idle_teardown_sec)
                ):
                    should_stop = True
            if should_stop:
                with self._vm_lock:
                    recheck = False
                    with self._state_lock:
                        if (
                            not self._shutting_down
                            and not self._job_running
                            and self._active_vm_name is not None
                            and (time.monotonic() - self._last_activity_at) >= float(self.cfg.idle_teardown_sec)
                        ):
                            recheck = True
                    if recheck:
                        self._stop_active_vm_under_vm_lock(reason="idle_teardown")

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(None)
            reader = conn.makefile("r", encoding="utf-8", errors="replace")
            for raw_line in reader:
                if self._stop_event.is_set():
                    break
                line = raw_line.strip()
                if not line:
                    continue
                response = self._dispatch_jsonrpc(line)
                if response is not None:
                    encoded = (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
                    conn.sendall(encoded)
        except OSError:
            pass
        finally:
            with self._clients_lock:
                self._client_socks.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch_jsonrpc(self, raw_line: str) -> dict[str, Any] | None:
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
        if not isinstance(req, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected JSON object"},
            }

        has_id = "id" in req
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if not isinstance(method, str):
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
            }

        # Notifications (no id) get no JSON-RPC response.
        if method == "notifications/initialized" or not has_id:
            return None

        if method == "initialize":
            client_proto = params.get("protocolVersion") if isinstance(params, dict) else None
            proto = client_proto if isinstance(client_proto, str) and client_proto else MCP_PROTOCOL_DEFAULT
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {}},
                    "serverInfo": dict(SERVER_INFO),
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOLS_SPEC}}

        if method == "tools/call":
            if not isinstance(params, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Invalid params: expected object"},
                }
            tool_name = params.get("name")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Invalid params: arguments must be an object"},
                }
            result = self._call_tool(str(tool_name or ""), arguments)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "colab_run":
            return self._tool_colab_run(args)
        if name == "colab_status":
            return self._tool_colab_status(args)
        if name == "colab_fetch":
            return self._tool_colab_fetch(args)
        if name == "colab_stop":
            return self._tool_colab_stop(args)
        return _tool_result({"ok": False, "error": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}, is_error=True)

    def _tool_colab_run(self, args: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            if self._shutting_down:
                return _tool_result(
                    {"ok": False, "error": "BROKER_SHUTTING_DOWN", "message": "Broker is shutting down"},
                    is_error=True,
                )
            if self._job_running:
                return _tool_result(
                    {
                        "ok": False,
                        "error": "VM_BUSY",
                        "message": "Another Colab job is currently active (max 1 active VM/job at a time)",
                    },
                    is_error=True,
                )
            self._job_running = True
            self._last_activity_at = time.monotonic()

        try:
            script = args.get("script")
            gpu_raw = args.get("gpu", "T4")
            timeout_min_raw = args.get("timeout_min", 15.0)
            staged_inputs_raw = args.get("staged_inputs")
            output_paths_raw = args.get("output_paths")
            pip_packages_raw = args.get("pip_packages")

            try:
                gpu = validate_gpu(str(gpu_raw), self.cfg)
            except ValueError as exc:
                self._emit_event("colab_blocked", f"Blocked GPU request: {exc}", values={"reason": "gpu"})
                return _tool_result({"ok": False, "error": "GPU_NOT_ALLOWED", "message": str(exc)}, is_error=True)

            violations = scan_script(script if isinstance(script, str) else "")
            if violations:
                self._emit_event(
                    "colab_blocked",
                    f"Blocked Colab script ({len(violations)} guardrail violation(s))",
                    values={"reason": "script_guardrail", "violations": violations},
                )
                return _tool_result(
                    {
                        "ok": False,
                        "error": "FORBIDDEN_SCRIPT_PATTERN",
                        "violations": violations,
                        "message": "; ".join(violations),
                    },
                    is_error=True,
                )

            try:
                pip_packages = validate_pip_packages(pip_packages_raw)
            except ValueError as exc:
                self._emit_event("colab_blocked", f"Blocked pip_packages: {exc}", values={"reason": "pip_packages"})
                return _tool_result({"ok": False, "error": "INVALID_PIP_PACKAGE", "message": str(exc)}, is_error=True)

            try:
                resolved_inputs = resolve_staged_inputs(self.cfg.io_dir, staged_inputs_raw, self.cfg.max_upload_mb)
            except (ValueError, FileNotFoundError) as exc:
                self._emit_event("colab_blocked", f"Blocked staged_inputs: {exc}", values={"reason": "staged_inputs"})
                return _tool_result({"ok": False, "error": "INVALID_STAGED_INPUT", "message": str(exc)}, is_error=True)

            try:
                output_paths = validate_output_paths(output_paths_raw)
            except ValueError as exc:
                self._emit_event("colab_blocked", f"Blocked output_paths: {exc}", values={"reason": "output_paths"})
                return _tool_result({"ok": False, "error": "INVALID_OUTPUT_PATH", "message": str(exc)}, is_error=True)

            try:
                req_timeout_min = float(timeout_min_raw)
            except (TypeError, ValueError):
                return _tool_result(
                    {"ok": False, "error": "INVALID_ARGUMENT", "message": "timeout_min must be a positive number"},
                    is_error=True,
                )
            if req_timeout_min <= 0:
                return _tool_result(
                    {"ok": False, "error": "INVALID_ARGUMENT", "message": "timeout_min must be > 0"},
                    is_error=True,
                )

            max_budget_sec = float(self.cfg.max_session_gpu_min) * 60.0
            with self._vm_lock:
                used_sec = self._gpu_seconds_used()
                remaining_sec = max_budget_sec - used_sec
                if remaining_sec <= 0:
                    self._stop_active_vm_under_vm_lock(reason="gpu_budget_exhausted")
                    self._emit_event(
                        "colab_blocked",
                        "Blocked colab_run: session GPU wall-clock budget exhausted",
                        values={"reason": "gpu_budget_exhausted", "gpu_seconds_used": round(used_sec, 3)},
                    )
                    return _tool_result(
                        {
                            "ok": False,
                            "error": "GPU_BUDGET_EXHAUSTED",
                            "gpu_seconds_used": round(used_sec, 3),
                            "max_session_gpu_min": self.cfg.max_session_gpu_min,
                            "message": "Session GPU wall-clock budget exhausted",
                        },
                        is_error=True,
                    )

                effective_timeout_sec = min(
                    req_timeout_min * 60.0,
                    float(self.cfg.max_job_timeout_min) * 60.0,
                    remaining_sec,
                )

                with self._state_lock:
                    cur_vm = self._active_vm_name
                    cur_gpu = self._active_vm_gpu
                if cur_vm is not None and cur_gpu != gpu:
                    self._stop_active_vm_under_vm_lock(reason="gpu_switch")
                    cur_vm = None

                if cur_vm is None:
                    vm_name = _sanitize_vm_name(self.cfg.session_id)
                    try:
                        self.backend.start_vm(vm_name, gpu)
                    except Exception as exc:
                        return _tool_result(
                            {"ok": False, "error": "VM_START_FAILED", "message": str(exc)},
                            is_error=True,
                        )
                    with self._state_lock:
                        self._active_vm_name = vm_name
                        self._active_vm_gpu = gpu
                        self._vm_started_at = time.monotonic()
                        self._last_activity_at = self._vm_started_at
                    self._emit_event(
                        "colab_vm_start",
                        f"Started Colab VM {vm_name} ({gpu})",
                        values={"vm_name": vm_name, "gpu": gpu},
                    )
                else:
                    vm_name = cur_vm

                with self._state_lock:
                    self._job_seq += 1
                    job_id = f"job-{self._job_seq:04d}-{uuid.uuid4().hex[:6]}"
                created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

                # Upload staged inputs
                staged_rel_list = list(staged_inputs_raw or [])
                try:
                    for rel_path, host_path in zip(staged_rel_list, resolved_inputs):
                        remote_dst = str(PurePosixPath("/content") / PurePosixPath(rel_path))
                        self.backend.upload_file(vm_name, host_path, remote_dst)
                except Exception as exc:
                    return _tool_result(
                        {"ok": False, "error": "UPLOAD_FAILED", "job_id": job_id, "message": str(exc)},
                        is_error=True,
                    )

                run_res = self.backend.exec_script(
                    vm_name,
                    script,
                    effective_timeout_sec,
                    pip_packages=pip_packages,
                )

                outputs_meta: list[dict[str, Any]] = []
                download_err: str | None = None
                if run_res.returncode == 0 and not run_res.timed_out and output_paths:
                    out_job_dir = self.cfg.io_dir / "out" / job_id
                    out_job_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
                    os.chmod(out_job_dir, 0o755)
                    try:
                        for rel_out in output_paths:
                            remote_src = str(PurePosixPath("/content") / PurePosixPath(rel_out))
                            local_dst = out_job_dir / rel_out
                            self.backend.download_file(vm_name, remote_src, local_dst)
                        outputs_meta = verify_downloaded_outputs(
                            out_job_dir,
                            output_paths,
                            self.cfg.max_download_mb,
                            self.cfg.container_io_root,
                            job_id,
                        )
                    except Exception as exc:
                        download_err = str(exc)

                finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if run_res.timed_out:
                    status = "timed_out"
                elif run_res.returncode != 0:
                    status = "failed"
                elif download_err is not None:
                    status = "output_error"
                else:
                    status = "completed"

                ok = status == "completed"
                used_after = self._gpu_seconds_used()
                if self.cfg.ephemeral_vm or used_after >= max_budget_sec:
                    stop_reason = "ephemeral_vm" if self.cfg.ephemeral_vm else "gpu_budget_exhausted"
                    self._stop_active_vm_under_vm_lock(reason=stop_reason)
                    used_after = self._gpu_seconds_used()

                job_record: dict[str, Any] = {
                    "ok": ok,
                    "job_id": job_id,
                    "vm_name": vm_name,
                    "gpu": gpu,
                    "status": status,
                    "returncode": run_res.returncode,
                    "timed_out": run_res.timed_out,
                    "duration_sec": round(run_res.duration_sec, 3),
                    "effective_timeout_sec": round(effective_timeout_sec, 3),
                    "stdout": run_res.stdout,
                    "stderr": run_res.stderr,
                    "outputs": outputs_meta,
                    "created_at": created_at,
                    "finished_at": finished_at,
                    "gpu_seconds_used": round(used_after, 3),
                    "gpu_seconds_remaining": round(max(0.0, max_budget_sec - used_after), 3),
                }
                if download_err is not None:
                    job_record["error"] = "OUTPUT_DOWNLOAD_FAILED"
                    job_record["message"] = download_err

                with self._state_lock:
                    self._jobs[job_id] = job_record

                self._emit_event(
                    "colab_job_run",
                    f"Colab job {job_id} {status} on {gpu} (rc={run_res.returncode})",
                    values={
                        "job_id": job_id,
                        "gpu": gpu,
                        "status": status,
                        "returncode": run_res.returncode,
                        "timed_out": run_res.timed_out,
                        "duration_sec": round(run_res.duration_sec, 3),
                        "outputs": len(outputs_meta),
                    },
                )
                return _tool_result(job_record, is_error=not ok)
        finally:
            with self._state_lock:
                self._job_running = False
                self._last_activity_at = time.monotonic()

    def _tool_colab_status(self, args: dict[str, Any]) -> dict[str, Any]:
        job_id = args.get("job_id")
        max_budget_sec = float(self.cfg.max_session_gpu_min) * 60.0
        with self._state_lock:
            self._last_activity_at = time.monotonic()
            used_sec = self._gpu_seconds_used_locked()
            if job_id is not None:
                job = self._jobs.get(str(job_id))
                if job is None:
                    return _tool_result(
                        {"ok": False, "error": "JOB_NOT_FOUND", "message": f"Unknown job_id: {job_id}"},
                        is_error=True,
                    )
                summary = {k: v for k, v in job.items() if k not in ("stdout", "stderr")}
                summary["gpu_seconds_used"] = round(used_sec, 3)
                summary["gpu_seconds_remaining"] = round(max(0.0, max_budget_sec - used_sec), 3)
                return _tool_result(summary)

            jobs_summary = [
                {
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "gpu": j["gpu"],
                    "returncode": j["returncode"],
                    "duration_sec": j["duration_sec"],
                }
                for j in self._jobs.values()
            ]
            return _tool_result({
                "ok": True,
                "session_id": self.cfg.session_id,
                "active_vm": self._active_vm_name,
                "active_gpu": self._active_vm_gpu,
                "job_running": self._job_running,
                "ephemeral_vm": self.cfg.ephemeral_vm,
                "idle_teardown_sec": self.cfg.idle_teardown_sec,
                "max_session_gpu_min": self.cfg.max_session_gpu_min,
                "gpu_seconds_used": round(used_sec, 3),
                "gpu_seconds_remaining": round(max(0.0, max_budget_sec - used_sec), 3),
                "jobs": jobs_summary,
            })

    def _tool_colab_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        job_id = args.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            return _tool_result(
                {"ok": False, "error": "INVALID_ARGUMENT", "message": "job_id is required"},
                is_error=True,
            )
        include_logs = bool(args.get("include_logs", True))
        with self._state_lock:
            self._last_activity_at = time.monotonic()
            job = self._jobs.get(job_id)
            if job is None:
                return _tool_result(
                    {"ok": False, "error": "JOB_NOT_FOUND", "message": f"Unknown job_id: {job_id}"},
                    is_error=True,
                )
            payload = dict(job)
        if not include_logs:
            payload.pop("stdout", None)
            payload.pop("stderr", None)
        return _tool_result(payload, is_error=not payload.get("ok", False))

    def _tool_colab_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            if self._job_running:
                return _tool_result(
                    {"ok": False, "error": "VM_BUSY", "message": "Cannot stop VM while a job is actively running"},
                    is_error=True,
                )
        with self._vm_lock:
            with self._state_lock:
                if self._job_running:
                    return _tool_result(
                        {"ok": False, "error": "VM_BUSY", "message": "Cannot stop VM while a job is actively running"},
                        is_error=True,
                    )
            stopped, vm_name = self._stop_active_vm_under_vm_lock(reason="colab_stop")
            used_sec = self._gpu_seconds_used()
        return _tool_result({
            "ok": True,
            "stopped": stopped,
            "vm_name": vm_name,
            "requested_job_id": args.get("job_id"),
            "gpu_seconds_used": round(used_sec, 3),
        })
