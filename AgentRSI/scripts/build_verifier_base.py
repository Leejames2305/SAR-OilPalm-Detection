"""Build the local sealed verifier base image (localhost/agentrsi-sealed:epoch1).

Stages all sealed label and split files inside a temporary mode-0700 directory under
AgentRSI/.sealed/_img_build, verifies labels_master.csv against verifier_context/manifest.json,
builds the Podman image with root-only /sealed (0700/0600) and unprivileged user `runner`,
and removes the temporary build context immediately on exit.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parents[1]
SEALED_DIR = AGENT_ROOT / ".sealed"
CONFIG_DIR = AGENT_ROOT / "config"
VIEW_DIR = AGENT_ROOT / ".work" / "agent_view"

BASE_DOCKERFILE = """FROM docker.io/library/python:3.12-slim-bookworm

RUN pip install --no-cache-dir \\
    numpy==2.5.5 \\
    pandas==3.1.1 \\
    scipy==1.18.1 \\
    scikit-learn==1.9.1 \\
    pyyaml==6.0.3 \\
    pyarrow==26.0.1

RUN useradd --uid 10001 --create-home --shell /bin/bash runner

COPY sealed/ /sealed/
RUN chown -R root:root /sealed && chmod 700 /sealed && chmod 600 /sealed/*

COPY verifier/ /opt/agentrsi/verifier/
COPY starter/ /opt/agentrsi/starter/
RUN chown -R root:root /opt/agentrsi && chmod -R a+rX /opt/agentrsi

ENV PYTHONPATH=/opt/agentrsi
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


def verify_sealed_manifest() -> None:
    manifest = json.loads((SEALED_DIR / "verifier_context" / "manifest.json").read_text())
    vc_master = SEALED_DIR / "verifier_context" / "labels_master.csv"
    digest = hashlib.sha256(vc_master.read_bytes()).hexdigest()
    if digest != manifest["labels_master_sha256"]:
        raise RuntimeError(
            f"verifier_context/labels_master.csv sha256 {digest} != manifest {manifest['labels_master_sha256']}"
        )


def build_image() -> str:
    verifier_cfg = yaml.safe_load((CONFIG_DIR / "verifier.yaml").read_text())
    image_tag = str(verifier_cfg["sealed_base_image"])
    verify_sealed_manifest()

    ctx = SEALED_DIR / "_img_build"
    if ctx.exists():
        shutil.rmtree(ctx)
    ctx.mkdir(mode=0o700, parents=True)
    try:
        os.chmod(ctx, 0o700)
        sealed_stage = ctx / "sealed"
        sealed_stage.mkdir(mode=0o700)

        sealed_sources = [
            SEALED_DIR / "verifier_context" / "labels_master.csv",
            CONFIG_DIR / "split_assignment.csv",
            CONFIG_DIR / "split.yaml",
            CONFIG_DIR / "verifier.yaml",
            VIEW_DIR / "L_labelled.csv",
            VIEW_DIR / "unlabelled.csv",
        ]
        for src in sealed_sources:
            if not src.exists():
                raise FileNotFoundError(f"Required verifier source missing: {src}")
            dst = sealed_stage / src.name
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o600)

        shutil.copytree(
            AGENT_ROOT / "verifier",
            ctx / "verifier",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            AGENT_ROOT / "starter",
            ctx / "starter",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (ctx / "Dockerfile").write_text(BASE_DOCKERFILE)

        cmd = ["podman", "build", "-t", image_tag, str(ctx)]
        print("Running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, env=podman_env(), check=False)
        if res.returncode != 0:
            raise SystemExit(res.returncode)
    finally:
        shutil.rmtree(ctx, ignore_errors=True)

    print("Built sealed verifier base image:", image_tag)
    return image_tag


def main() -> None:
    build_image()


if __name__ == "__main__":
    main()
