"""CLI wrapper to regrade a recorded Harbor job directory under rootful Podman."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.env_utils import podman_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Regrade a recorded Harbor job with `-e podman`.")
    ap.add_argument("source_job", type=Path, help="Recorded Harbor timestamped job directory")
    ap.add_argument("--task-dir", "-p", type=Path, required=True, help="Task directory providing the verifier")
    ap.add_argument("--jobs-dir", "-o", type=Path, default=AGENT_ROOT / "jobs" / "regrade")
    args = ap.parse_args()

    harbor_bin = Path(sys.executable).parent / "harbor"
    exe = str(harbor_bin) if harbor_bin.exists() else (shutil.which("harbor") or "harbor")
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "job",
        "regrade",
        str(args.source_job),
        "-p",
        str(args.task_dir),
        "-e",
        "podman",
        "-o",
        str(args.jobs_dir),
    ]
    res = subprocess.run(cmd, env=podman_env(AGENT_ROOT), check=False)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
