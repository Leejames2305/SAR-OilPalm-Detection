"""Build the prebuilt agent sandbox base image (localhost/agentrsi-agent-base:latest)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = AGENT_ROOT / "config"
TEMPLATES_DIR = AGENT_ROOT / "templates"

AGENT_DOCKERFILE = """FROM docker.io/library/python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl \\
        git \\
        ca-certificates \\
        libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
    numpy==2.5.5 \\
    pandas==3.1.1 \\
    scipy==1.18.1 \\
    scikit-learn==1.9.1 \\
    pyyaml==6.0.3 \\
    pyarrow==26.0.1 \\
    rasterio==1.5.2 \\
    earthengine-api==1.7.53 \\
    xgboost \\
    lightgbm \\
    geopandas

COPY agentrsi_cli.py /usr/local/bin/agentrsi
RUN chmod +x /usr/local/bin/agentrsi && \\
    mkdir -p /app/data /app/rasters /app/prior /app/incumbent /app/submission /app/report /app/datasets

WORKDIR /app
"""


def podman_env() -> dict[str, str]:
    env = os.environ.copy()
    tools_bin = str(AGENT_ROOT / ".tools" / "bin")
    env["PATH"] = f"{tools_bin}:{env.get('PATH', '')}"
    xdg = Path("/tmp/xdg-agentrsi")
    xdg.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_RUNTIME_DIR", str(xdg))
    sock = "unix:///mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock"
    env.setdefault("CONTAINER_HOST", sock)
    env.setdefault("DOCKER_HOST", sock)
    env["PODMAN_COMPOSE_WARNING_LOGS"] = "false"
    return env


def build_agent_base() -> str:
    cfg = yaml.safe_load((CONFIG_DIR / "session.yaml").read_text())
    image_tag = str(cfg["agent_base_image"])
    cli_src = TEMPLATES_DIR / "agentrsi_cli.py"
    if not cli_src.exists():
        raise FileNotFoundError(cli_src)

    with tempfile.TemporaryDirectory(prefix="agentrsi_agent_base_") as td:
        ctx = Path(td)
        shutil.copyfile(cli_src, ctx / "agentrsi_cli.py")
        (ctx / "Dockerfile").write_text(AGENT_DOCKERFILE)
        cmd = ["podman", "build", "-t", image_tag, str(ctx)]
        print("Running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, env=podman_env(), check=False)
        if res.returncode != 0:
            raise SystemExit(res.returncode)

    print("Built agent base image:", image_tag)
    return image_tag


def main() -> None:
    build_agent_base()


if __name__ == "__main__":
    main()
