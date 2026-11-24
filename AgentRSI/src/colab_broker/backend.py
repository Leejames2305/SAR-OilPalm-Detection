"""Colab execution backends: `CliColabBackend` (google-colab-cli) and `MockColabBackend`."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from colab_broker.guardrails import sanitized_colab_env


@dataclass
class ColabRunResult:
    """Result of executing a script on a Colab VM (real or mocked)."""

    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False


class ColabBackend(ABC):
    """Abstract interface for provisioning and running jobs on Colab VMs."""

    @abstractmethod
    def check_auth_ready(self) -> tuple[bool, str]:
        """Verify Colab CLI availability and host authentication readiness."""

    @abstractmethod
    def start_vm(self, vm_name: str, gpu: str) -> None:
        """Provision a Colab VM session with the requested GPU."""

    @abstractmethod
    def upload_file(self, vm_name: str, local_path: Path, remote_path: str) -> None:
        """Upload a local file to `remote_path` on the Colab VM."""

    @abstractmethod
    def exec_script(
        self,
        vm_name: str,
        script_text: str,
        timeout_sec: float,
        pip_packages: list[str] | None = None,
    ) -> ColabRunResult:
        """Execute a Python script on the Colab VM with optional pip installs."""

    @abstractmethod
    def download_file(self, vm_name: str, remote_path: str, local_path: Path) -> None:
        """Download `remote_path` from the Colab VM to `local_path`."""

    @abstractmethod
    def stop_vm(self, vm_name: str) -> None:
        """Terminate and release the Colab VM session."""


class CliColabBackend(ColabBackend):
    """Host-side wrapper around `google-colab-cli` (`colab`)."""

    def __init__(
        self,
        colab_bin: str = "colab",
        *,
        check_timeout_sec: float = 15.0,
        vm_timeout_sec: float = 180.0,
        transfer_timeout_sec: float = 120.0,
    ) -> None:
        self.colab_bin = colab_bin
        self.check_timeout_sec = float(check_timeout_sec)
        self.vm_timeout_sec = float(vm_timeout_sec)
        self.transfer_timeout_sec = float(transfer_timeout_sec)

    def _resolve_bin(self) -> str | None:
        if "/" in self.colab_bin or "\\" in self.colab_bin:
            p = Path(self.colab_bin)
            return str(p) if p.is_file() else None
        search_path = os.environ.get("PATH", "")
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in search_path.split(os.pathsep):
            search_path = f"{search_path}{os.pathsep}{local_bin}" if search_path else local_bin
        return shutil.which(self.colab_bin, path=search_path)

    def _run_cli(
        self,
        args: list[str],
        *,
        timeout_sec: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        bin_path = self._resolve_bin()
        if not bin_path:
            raise FileNotFoundError(
                f"Colab CLI '{self.colab_bin}' not found. Install google-colab-cli on host WSL."
            )
        return subprocess.run(
            [bin_path, *args],
            env=sanitized_colab_env(),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )

    def check_auth_ready(self) -> tuple[bool, str]:
        bin_path = self._resolve_bin()
        if not bin_path:
            return (
                False,
                f"Colab CLI '{self.colab_bin}' not found on PATH. "
                "Install `google-colab-cli` on host WSL and run `colab auth`.",
            )
        try:
            help_res = self._run_cli(["--help"], timeout_sec=self.check_timeout_sec)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Failed to invoke `{self.colab_bin} --help`: {type(exc).__name__}"
        if help_res.returncode != 0:
            return (
                False,
                f"`{self.colab_bin} --help` exited with code {help_res.returncode}. "
                "Reinstall `google-colab-cli` on host WSL.",
            )

        try:
            ls_res = self._run_cli(["ls"], timeout_sec=self.check_timeout_sec)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return (
                False,
                f"Colab CLI probe `{self.colab_bin} ls` failed ({type(exc).__name__}). "
                "Verify network and run `colab auth` on host WSL.",
            )

        if ls_res.returncode == 0:
            return True, "google-colab-cli is installed and authenticated."

        combined = f"{ls_res.stdout}\n{ls_res.stderr}".lower()
        auth_hints = ("auth", "login", "credential", "token", "401", "403", "unauthenticated", "unauthorized")
        if any(hint in combined for hint in auth_hints):
            return (
                False,
                "google-colab-cli is not authenticated. Run `colab auth` on host WSL "
                "(keep host OAuth/ADC outside containers).",
            )

        if any(hint in combined for hint in ("no active", "no session", "no runtime", "not running")):
            # Control-plane / runtime-not-started error still means CLI is usable; check whoami if supported.
            try:
                who_res = self._run_cli(["whoami"], timeout_sec=self.check_timeout_sec)
                if who_res.returncode == 0:
                    return True, "google-colab-cli is installed and authenticated (no active runtime)."
                who_text = f"{who_res.stdout}\n{who_res.stderr}".lower()
                if any(hint in who_text for hint in auth_hints):
                    return (
                        False,
                        "google-colab-cli is not authenticated. Run `colab auth` on host WSL.",
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass
            return True, "google-colab-cli is installed (no active runtime)."

        return (
            False,
            f"Colab CLI probe `{self.colab_bin} ls` exited {ls_res.returncode}. "
            "Run `colab auth` on host WSL to authenticate.",
        )

    def start_vm(self, vm_name: str, gpu: str) -> None:
        res = self._run_cli(
            ["new", "-s", vm_name, "--gpu", gpu],
            timeout_sec=self.vm_timeout_sec,
        )
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"colab new failed (rc={res.returncode}): {msg[:500]}")

    def upload_file(self, vm_name: str, local_path: Path, remote_path: str) -> None:
        res = self._run_cli(
            ["upload", "-s", vm_name, str(local_path), remote_path],
            timeout_sec=self.transfer_timeout_sec,
        )
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"colab upload failed (rc={res.returncode}): {msg[:500]}")

    def exec_script(
        self,
        vm_name: str,
        script_text: str,
        timeout_sec: float,
        pip_packages: list[str] | None = None,
    ) -> ColabRunResult:
        full_script = script_text
        if pip_packages:
            pkg_literal = repr(list(pip_packages))
            preamble = (
                "import subprocess as _agentrsi_sp, sys as _agentrsi_sys\n"
                f"_agentrsi_sp.run([_agentrsi_sys.executable, '-m', 'pip', 'install', '-q', *{pkg_literal}], check=True)\n"
            )
            full_script = preamble + "\n" + script_text

        fd, tmp_name = tempfile.mkstemp(prefix="agentrsi_colab_", suffix=".py")
        tmp_path = Path(tmp_name)
        t0 = time.monotonic()
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(full_script)
            try:
                res = self._run_cli(
                    ["exec", "-s", vm_name, "-f", str(tmp_path), "--timeout", str(float(timeout_sec))],
                    timeout_sec=timeout_sec,
                )
                return ColabRunResult(
                    returncode=int(res.returncode),
                    stdout=res.stdout or "",
                    stderr=res.stderr or "",
                    duration_sec=time.monotonic() - t0,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                return ColabRunResult(
                    returncode=124,
                    stdout=stdout,
                    stderr=f"{stderr}\nExecution timed out after {timeout_sec:.1f}s".strip(),
                    duration_sec=time.monotonic() - t0,
                    timed_out=True,
                )
        finally:
            tmp_path.unlink(missing_ok=True)

    def download_file(self, vm_name: str, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        res = self._run_cli(
            ["download", "-s", vm_name, remote_path, str(local_path)],
            timeout_sec=self.transfer_timeout_sec,
        )
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"colab download failed (rc={res.returncode}): {msg[:500]}")

    def stop_vm(self, vm_name: str) -> None:
        res = self._run_cli(
            ["stop", "-s", vm_name],
            timeout_sec=self.vm_timeout_sec,
        )
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"colab stop failed (rc={res.returncode}): {msg[:500]}")


class MockColabBackend(ColabBackend):
    """Offline mock backend executing scripts in per-VM temp directories."""

    def __init__(self, *, auth_ready: bool = True) -> None:
        self.auth_ready = auth_ready
        self.started_vms: list[str] = []
        self.started_vm_gpus: dict[str, str] = {}
        self.stopped_vms: list[str] = []
        self.active_vms: dict[str, Path] = {}
        self.exec_history: list[dict[str, Any]] = []

    def check_auth_ready(self) -> tuple[bool, str]:
        if self.auth_ready:
            return True, "MockColabBackend authenticated and ready."
        return False, "MockColabBackend configured unauthenticated. Run `colab auth`."

    def _vm_workspace(self, vm_name: str) -> Path:
        ws = self.active_vms.get(vm_name)
        if ws is None:
            raise RuntimeError(f"Mock VM {vm_name!r} is not active")
        return ws

    def _resolve_vm_path(self, vm_name: str, remote_path: str) -> Path:
        ws = self._vm_workspace(vm_name)
        raw = str(remote_path).strip()
        if raw.startswith("/content/"):
            rel = raw[len("/content/"):]
        elif raw == "/content":
            rel = ""
        else:
            rel = raw.lstrip("/")
        posix = PurePosixPath(rel) if rel else PurePosixPath(".")
        if any(part == ".." for part in posix.parts):
            raise ValueError(f"Path traversal forbidden in mock remote path: {remote_path!r}")
        target = (ws / posix.as_posix()).resolve()
        ws_root = ws.resolve()
        if not target.is_relative_to(ws_root):
            raise ValueError(f"Mock remote path escapes VM workspace: {remote_path!r}")
        return target

    def start_vm(self, vm_name: str, gpu: str) -> None:
        if vm_name not in self.active_vms:
            self.active_vms[vm_name] = Path(tempfile.mkdtemp(prefix=f"mock_colab_{vm_name}_"))
        self.started_vms.append(vm_name)
        self.started_vm_gpus[vm_name] = gpu

    def upload_file(self, vm_name: str, local_path: Path, remote_path: str) -> None:
        dst = self._resolve_vm_path(vm_name, remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dst)

    def exec_script(
        self,
        vm_name: str,
        script_text: str,
        timeout_sec: float,
        pip_packages: list[str] | None = None,
    ) -> ColabRunResult:
        ws = self._vm_workspace(vm_name)
        script_path = ws / "_agentrsi_job.py"
        script_path.write_text(script_text, encoding="utf-8")
        env = sanitized_colab_env()
        env["COLAB_MOCK_VM"] = vm_name
        env["COLAB_MOCK_GPU"] = self.started_vm_gpus.get(vm_name, "T4")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ws),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            res = ColabRunResult(
                returncode=int(proc.returncode),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration_sec=time.monotonic() - t0,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            res = ColabRunResult(
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\nExecution timed out after {timeout_sec:.1f}s".strip(),
                duration_sec=time.monotonic() - t0,
                timed_out=True,
            )
        finally:
            script_path.unlink(missing_ok=True)

        self.exec_history.append({
            "vm_name": vm_name,
            "pip_packages": list(pip_packages or []),
            "returncode": res.returncode,
            "timed_out": res.timed_out,
        })
        return res

    def download_file(self, vm_name: str, remote_path: str, local_path: Path) -> None:
        src = self._resolve_vm_path(vm_name, remote_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Mock remote file not found: {remote_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)

    def stop_vm(self, vm_name: str) -> None:
        ws = self.active_vms.pop(vm_name, None)
        if ws is not None:
            shutil.rmtree(ws, ignore_errors=True)
        self.stopped_vms.append(vm_name)
