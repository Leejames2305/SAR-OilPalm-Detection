"""Session runner: budget guard, secret staging, gateway config, and Harbor invocation."""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

from libbuilder.env_utils import AGENT_ROOT, load_dotenv, podman_env, resolve_gateway
from libbuilder.events import append_event
from libbuilder.lineage import check_budget_guard, load_lineage
from libbuilder.stage_secrets import CONTAINER_SECRET_PATH, build_mounts_json, stage_gcp_sa


def _harbor_bin() -> str:
    candidate = Path(sys.executable).parent / "harbor"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("harbor")
    if found:
        return found
    raise FileNotFoundError("harbor executable not found in active venv or PATH")


def build_harbor_command(
    *,
    task_dir: Path,
    jobs_out_dir: Path,
    agent: str,
    model: str,
    gateway: dict[str, str],
    env_type: str,
    mounts_json: str,
    gcp_project: str,
    agent_cpus: int,
    reasoning_effort: str,
    max_turns: int | None,
    max_budget_usd: float | None,
    dry_run: bool = False,
) -> list[str]:
    """Assemble the exact Harbor CLI invocation without embedding any secret values."""
    cmd = [
        _harbor_bin(),
        "run",
        "--yes",
        "-p",
        str(task_dir),
        "-a",
        agent,
        "-m",
        model,
        "-e",
        env_type,
        "--base-url",
        gateway["base_url"],
        "--api-format",
        gateway["api_format"],
        "--api-key-name",
        gateway["api_key_name"],
        "--mounts",
        mounts_json,
        "--ae",
        f"GOOGLE_APPLICATION_CREDENTIALS={CONTAINER_SECRET_PATH}",
        "--ae",
        f"GCP_PROJECT={gcp_project}",
        "--ae",
        f"AGENTRSI_CPUS={agent_cpus}",
        "-o",
        str(jobs_out_dir),
    ]
    if reasoning_effort:
        cmd.extend(["--ak", f"reasoning_effort={reasoning_effort}"])
    if agent == "claude-code":
        if max_turns is not None:
            cmd.extend(["--ak", f"max_turns={int(max_turns)}"])
        if max_budget_usd is not None:
            cmd.extend(["--ak", f"max_budget_usd={float(max_budget_usd)}"])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_session(
    session_id: str,
    *,
    agent: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    dry_run: bool = False,
    colab_mock: bool = False,
    colab_allow_a100: bool = False,
    tasks_root: Path | None = None,
    jobs_root: Path | None = None,
    env_path: Path | None = None,
    lineage_path: Path | None = None,
    events_path: Path | None = None,
    stage_dir: Path = Path("/mnt/wsl/agentrsi"),
    agent_root: Path = AGENT_ROOT,
) -> int:
    """Validate budget/kill-switch, stage secrets, and run Harbor for `session_id`."""
    task_dir = (tasks_root or (agent_root / "tasks")) / session_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory missing: {task_dir} (run build_task.py first)")

    cfg_dir = agent_root / "config"
    session_cfg = yaml.safe_load((cfg_dir / "session.yaml").read_text())
    verifier_cfg = yaml.safe_load((cfg_dir / "verifier.yaml").read_text())
    split_cfg = yaml.safe_load((cfg_dir / "split.yaml").read_text())

    ep = events_path or (agent_root / "state" / "events.jsonl")
    lp = lineage_path or (agent_root / "state" / "lineage.json")
    lineage = load_lineage(lp, split_cfg, verifier_cfg)
    epoch = int(lineage["epoch"])

    file_env = load_dotenv(env_path or (agent_root / ".env"))
    env_map = dict(file_env)
    for k, v in os.environ.items():
        if v and k in ("API_KEY", "BASE_URL", "GCP_PROJECT", "GCP_SA_KEY_PATH", "DAILY_SPEND_CAP_USD", "ENV"):
            env_map[k] = v

    daily_cap = float(env_map.get("DAILY_SPEND_CAP_USD") or 10.0)
    allowed, reason = check_budget_guard(lineage, daily_cap)
    if not allowed:
        if not dry_run:
            append_event(
                ep,
                session=session_id,
                epoch=epoch,
                event_type="spend_cap",
                summary=f"Blocked session start: {reason}",
                ref=f"sessions/{session_id}/meta.json",
                dedup_id=f"{session_id}:spend_cap",
            )
        print(f"BLOCKED: {reason}", file=sys.stderr)
        return 2

    use_agent = agent or str(session_cfg["agent"])
    use_model = model or str(session_cfg["model"])
    use_turns = max_turns if max_turns is not None else int(session_cfg.get("max_turns", 120))
    use_budget = max_budget_usd if max_budget_usd is not None else float(session_cfg.get("max_budget_usd", 10.0))
    use_effort = str(session_cfg.get("reasoning_effort", "medium"))
    use_env = env_map.get("ENV") or "podman"

    api_key = env_map.get("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_KEY is missing in AgentRSI/.env or environment")
    gcp_project = env_map.get("GCP_PROJECT", "").strip()
    if not gcp_project:
        raise RuntimeError("GCP_PROJECT is missing in AgentRSI/.env or environment")
    sa_key_raw = env_map.get("GCP_SA_KEY_PATH", "").strip()
    if not sa_key_raw:
        raise RuntimeError("GCP_SA_KEY_PATH is missing in AgentRSI/.env or environment")
    sa_key_path = Path(sa_key_raw)
    if not sa_key_path.is_absolute():
        sa_key_path = (agent_root / sa_key_path).resolve()

    gateway = resolve_gateway(use_agent, env_map)
    gcp_sa_dst, mounts_json = stage_gcp_sa(sa_key_path, stage_dir=stage_dir)

    jobs_out_dir = (jobs_root or (agent_root / "jobs")) / session_id
    jobs_out_dir.mkdir(parents=True, exist_ok=True)

    task_agent_cpus = int(session_cfg["agent_cpus"])
    mcp_server_names: set[str] = set()
    task_toml_path = task_dir / "task.toml"
    if task_toml_path.exists():
        try:
            task_meta = tomllib.loads(task_toml_path.read_text())
            env_tbl = task_meta.get("environment") or {}
            env_cpus = env_tbl.get("cpus")
            if env_cpus is not None:
                task_agent_cpus = int(env_cpus)
            for srv in env_tbl.get("mcp_servers") or []:
                if isinstance(srv, dict) and srv.get("name"):
                    mcp_server_names.add(str(srv["name"]))
        except Exception:
            pass

    extra_child_env: dict[str, str] = {gateway["api_key_name"]: api_key}
    if "agentrsi-verifier" in mcp_server_names:
        # Ephemeral per-session token: held in child_env only, never written to disk or logged.
        extra_child_env["AGENTRSI_VERIFIER_MCP_TOKEN"] = secrets.token_hex(32)

    broker: Any = None
    if "agentrsi-colab" in mcp_server_names:
        from colab_broker.backend import CliColabBackend, MockColabBackend
        from colab_broker.config import ColabBrokerConfig
        from colab_broker.server import ColabBrokerServer

        if colab_mock:
            backend: Any = MockColabBackend()
        else:
            backend = CliColabBackend()
            auth_ok, auth_msg = backend.check_auth_ready()
            if not auth_ok:
                raise RuntimeError(f"Colab broker auth check failed: {auth_msg}")

        colab_cfg = session_cfg.get("colab_broker") or {}
        colab_socket_path = stage_dir / f"colab-{session_id}.sock"
        colab_io_dir = stage_dir / f"colab-io-{session_id}"
        broker_cfg = ColabBrokerConfig(
            socket_path=colab_socket_path,
            io_dir=colab_io_dir,
            session_id=session_id,
            allowed_gpus=tuple(str(g) for g in (colab_cfg.get("allowed_gpus") or ["T4", "L4"])),
            allow_a100=bool(colab_allow_a100 or colab_cfg.get("allow_a100", False)),
            max_job_timeout_min=float(colab_cfg.get("max_job_timeout_min", 30.0)),
            max_session_gpu_min=float(colab_cfg.get("max_session_gpu_min", 60.0)),
            idle_teardown_sec=float(colab_cfg.get("idle_teardown_sec", 300.0)),
            max_upload_mb=float(colab_cfg.get("max_upload_mb", 256.0)),
            max_download_mb=float(colab_cfg.get("max_download_mb", 128.0)),
            events_path=None if dry_run else ep,
            epoch=epoch,
        )
        mounts_json = build_mounts_json(
            gcp_sa_dst,
            colab_socket_path=colab_socket_path,
            colab_io_dir=colab_io_dir,
        )
        broker = ColabBrokerServer(broker_cfg, backend=backend)

    cmd = build_harbor_command(
        task_dir=task_dir,
        jobs_out_dir=jobs_out_dir,
        agent=use_agent,
        model=use_model,
        gateway=gateway,
        env_type=use_env,
        mounts_json=mounts_json,
        gcp_project=gcp_project,
        agent_cpus=task_agent_cpus,
        reasoning_effort=use_effort,
        max_turns=use_turns,
        max_budget_usd=use_budget,
        dry_run=dry_run,
    )

    child_env = podman_env(agent_root, extra=extra_child_env)
    if not dry_run:
        append_event(
            ep,
            session=session_id,
            epoch=epoch,
            event_type="session_start",
            summary=f"Started {use_agent}/{use_model} for {session_id}",
            ref=f"sessions/{session_id}/meta.json",
            values={"agent": use_agent, "model": use_model},
            dedup_id=f"{session_id}:start",
        )

    try:
        if broker is not None:
            broker.start_background()
            broker.wait_ready(timeout=5.0)
        res = subprocess.run(cmd, env=child_env, check=False)
    finally:
        if broker is not None:
            broker.shutdown()

    if not dry_run:
        if res.returncode == 0:
            append_event(
                ep,
                session=session_id,
                epoch=epoch,
                event_type="session_end",
                summary=f"Harbor run finished for {session_id} (exit 0)",
                ref=f"sessions/{session_id}/meta.json",
                dedup_id=f"{session_id}:end",
            )
        else:
            append_event(
                ep,
                session=session_id,
                epoch=epoch,
                event_type="run_error",
                summary=f"Harbor run exited {res.returncode} for {session_id}",
                ref=f"sessions/{session_id}/meta.json",
                dedup_id=f"{session_id}:err:{res.returncode}",
            )
    return int(res.returncode)
