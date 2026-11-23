"""Stage GCP service-account credentials in /mnt/wsl/agentrsi for read-only container mount."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

DEFAULT_STAGE_DIR = Path("/mnt/wsl/agentrsi")
CONTAINER_SECRET_PATH = "/run/secrets/gcp-sa.json"


def stage_gcp_sa(sa_key_path: Path, stage_dir: Path = DEFAULT_STAGE_DIR) -> tuple[Path, str]:
    """Copy SA key to a mode-0600 file under mode-0700 stage_dir and return `(path, mounts_json)`."""
    if not sa_key_path.exists():
        raise FileNotFoundError(f"GCP service-account key not found: {sa_key_path}")
    stage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(stage_dir, 0o700)
    dst = stage_dir / "gcp-sa.json"
    shutil.copyfile(sa_key_path, dst)
    os.chmod(dst, 0o600)

    mounts = json.dumps([
        {
            "type": "bind",
            "source": str(dst),
            "target": CONTAINER_SECRET_PATH,
            "read_only": True,
        }
    ])
    return dst, mounts
