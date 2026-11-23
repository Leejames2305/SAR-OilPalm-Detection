"""Shared paths, .env parsing, gateway URL normalization, and Podman env helpers."""
from __future__ import annotations

import os
import re
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AGENT_ROOT.parent


def load_dotenv(env_path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines, stripping inline `# ...` comments and matching quotes."""
    if not env_path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if val and val[0] in ("'", '"') and len(val) >= 2:
            q = val[0]
            end_q = val.find(q, 1)
            if end_q != -1:
                val = val[1:end_q]
            else:
                val = re.sub(r"(^|\s)#.*$", "", val).strip()
        else:
            val = re.sub(r"(^|\s)#.*$", "", val).strip()
        out[key] = val
    return out


def resolve_gateway(agent: str, env_map: dict[str, str]) -> dict[str, str]:
    """Normalize gateway base_url, api_format, and api_key_name for the chosen agent."""
    raw_url = env_map.get("BASE_URL", "").strip().rstrip("/")
    if not raw_url:
        raise ValueError("BASE_URL is empty in .env / environment")

    if agent == "codex":
        base_url = raw_url if raw_url.endswith("/v1") else f"{raw_url}/v1"
        api_format = "openai_responses"
        api_key_name = "OPENAI_API_KEY"
    elif agent == "claude-code":
        base_url = raw_url[:-3] if raw_url.endswith("/v1") else raw_url
        api_format = "anthropic_messages"
        api_key_name = "ANTHROPIC_AUTH_TOKEN"
    else:
        raise ValueError(f"Unsupported agent {agent!r}; expected 'codex' or 'claude-code'")

    return {
        "base_url": base_url,
        "api_format": api_format,
        "api_key_name": api_key_name,
    }


def podman_env(agent_root: Path = AGENT_ROOT, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return environment configured for rootful Podman + Linux docker-compose."""
    env = os.environ.copy()
    tools_bin = str(agent_root / ".tools" / "bin")
    env["PATH"] = f"{tools_bin}:{env.get('PATH', '')}"
    xdg = Path("/tmp/xdg-agentrsi")
    xdg.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_RUNTIME_DIR", str(xdg))
    sock = "unix:///mnt/wsl/podman-sockets/podman-machine-default/podman-root.sock"
    env.setdefault("CONTAINER_HOST", sock)
    env.setdefault("DOCKER_HOST", sock)
    env["PODMAN_COMPOSE_WARNING_LOGS"] = "false"
    if extra:
        env.update(extra)
    return env
