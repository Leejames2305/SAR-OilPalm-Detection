"""Configuration for the host-side AgentRSI Colab GPU offload MCP broker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONTAINER_IO_ROOT: str = "/run/agentrsi/colab-io"


@dataclass
class ColabBrokerConfig:
    """Policy and runtime settings for `ColabBrokerServer`."""

    socket_path: Path
    io_dir: Path
    session_id: str
    allowed_gpus: tuple[str, ...] = ("T4", "L4")
    allow_a100: bool = False
    max_job_timeout_min: float = 30.0
    max_session_gpu_min: float = 60.0
    idle_teardown_sec: float = 300.0
    ephemeral_vm: bool = False
    max_upload_mb: float = 256.0
    max_download_mb: float = 128.0
    colab_bin: str = "colab"
    events_path: Path | None = None
    epoch: int = 1
    container_io_root: str = CONTAINER_IO_ROOT

    def __post_init__(self) -> None:
        self.socket_path = Path(self.socket_path)
        self.io_dir = Path(self.io_dir)
        if self.events_path is not None:
            self.events_path = Path(self.events_path)
        self.allowed_gpus = tuple(str(g).strip().upper() for g in self.allowed_gpus if str(g).strip())
        if not self.session_id or not str(self.session_id).strip():
            raise ValueError("session_id must be a non-empty string")
        if self.max_job_timeout_min <= 0:
            raise ValueError("max_job_timeout_min must be > 0")
        if self.max_session_gpu_min <= 0:
            raise ValueError("max_session_gpu_min must be > 0")
        if self.idle_teardown_sec <= 0:
            raise ValueError("idle_teardown_sec must be > 0")
        if self.max_upload_mb <= 0 or self.max_download_mb <= 0:
            raise ValueError("max_upload_mb and max_download_mb must be > 0")
