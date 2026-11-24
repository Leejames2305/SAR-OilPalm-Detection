"""Stage GCP service-account credentials in /mnt/wsl/agentrsi for read-only container mount."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_STAGE_DIR = Path("/mnt/wsl/agentrsi")
CONTAINER_SECRET_PATH = "/run/secrets/gcp-sa.json"
CONTAINER_COLAB_SOCKET_PATH = "/run/agentrsi/colab-mcp.sock"
CONTAINER_COLAB_IO_DIR = "/run/agentrsi/colab-io"


def build_mounts_json(
    gcp_sa_dst: Path,
    *,
    colab_socket_path: Path | None = None,
    colab_io_dir: Path | None = None,
) -> str:
    """Build Harbor `--mounts` JSON array for GCP credentials and optional Colab broker paths."""
    mounts: list[dict[str, object]] = [
        {
            "type": "bind",
            "source": str(gcp_sa_dst),
            "target": CONTAINER_SECRET_PATH,
            "read_only": True,
        }
    ]
    if colab_io_dir is not None:
        mounts.append(
            {
                "type": "bind",
                "source": str(colab_io_dir),
                "target": CONTAINER_COLAB_IO_DIR,
            }
        )
    if colab_socket_path is not None:
        mounts.append(
            {
                "type": "bind",
                "source": str(colab_socket_path),
                "target": CONTAINER_COLAB_SOCKET_PATH,
            }
        )
    return json.dumps(mounts)


def stage_gcp_sa(sa_key_path: Path, stage_dir: Path = DEFAULT_STAGE_DIR) -> tuple[Path, str]:
    """Copy SA key to a mode-0600 file under mode-0700 stage_dir and return `(path, mounts_json)`."""
    if not sa_key_path.exists():
        raise FileNotFoundError(f"GCP service-account key not found: {sa_key_path}")
    stage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(stage_dir, 0o700)
    dst = stage_dir / "gcp-sa.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".gcp-sa.", dir=str(stage_dir))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as out_fh, open(sa_key_path, "rb") as in_fh:
            shutil.copyfileobj(in_fh, out_fh)
        os.replace(tmp_name, dst)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    return dst, build_mounts_json(dst)
