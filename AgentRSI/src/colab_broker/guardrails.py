"""Security and policy guardrails for Colab GPU offload requests."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from colab_broker.config import ColabBrokerConfig

_FORBIDDEN_SCRIPT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"google\.colab\.auth", re.IGNORECASE), "Forbidden Colab auth module reference: google.colab.auth"),
    (re.compile(r"\bauthenticate_user\b"), "Forbidden Colab user auth call: authenticate_user"),
    (re.compile(r"google\.colab\.drive", re.IGNORECASE), "Forbidden Colab Drive module reference: google.colab.drive"),
    (re.compile(r"\bdrive\.mount\b"), "Forbidden Google Drive mount call: drive.mount"),
    (re.compile(r"\bgcloud\b", re.IGNORECASE), "Forbidden GCP CLI invocation/reference: gcloud"),
    (re.compile(r"\bgsutil\b", re.IGNORECASE), "Forbidden GCS CLI invocation/reference: gsutil"),
    (re.compile(r"sar[-_]?oilpalm", re.IGNORECASE), "Forbidden repository identifier: sar-oilpalm"),
    (re.compile(r"leejames2305", re.IGNORECASE), "Forbidden repository owner identifier: Leejames2305"),
    (re.compile(r"169\.254\.169\.254"), "Forbidden cloud metadata IP: 169.254.169.254"),
    (re.compile(r"metadata\.google\.internal", re.IGNORECASE), "Forbidden cloud metadata hostname: metadata.google.internal"),
    (re.compile(r"\bcolab\s+auth\b", re.IGNORECASE), "Forbidden Colab CLI auth command: colab auth"),
    (re.compile(r"/run/secrets"), "Forbidden secret mount path reference: /run/secrets"),
    (re.compile(r"\bGOOGLE_APPLICATION_CREDENTIALS\b"), "Forbidden credential env reference: GOOGLE_APPLICATION_CREDENTIALS"),
)

_PIP_PACKAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9.,_-]+\])?([<>=!~]=?[^;\s#]+)?$"
)

_SENSITIVE_ENV_VARS: frozenset[str] = frozenset({
    "API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AGENTRSI_VERIFIER_MCP_TOKEN",
})


def scan_script(script: str) -> list[str]:
    """Scan a Colab Python script for forbidden auth, metadata, or leak patterns."""
    if not isinstance(script, str) or not script.strip():
        return ["Script must be a non-empty string"]
    violations: list[str] = []
    for pattern, reason in _FORBIDDEN_SCRIPT_PATTERNS:
        if pattern.search(script):
            violations.append(reason)
    return violations


def validate_gpu(gpu: str, cfg: ColabBrokerConfig) -> str:
    """Validate requested GPU type against broker allowlist and A100 policy."""
    if not isinstance(gpu, str) or not gpu.strip():
        raise ValueError("GPU must be a non-empty string")
    norm = gpu.strip().upper()
    allowed = {g.strip().upper() for g in cfg.allowed_gpus if g.strip()}
    if cfg.allow_a100:
        allowed.add("A100")
    else:
        allowed.discard("A100")
    if norm not in allowed:
        raise ValueError(
            f"GPU {norm!r} is not allowed (allowed: {sorted(allowed)}; allow_a100={cfg.allow_a100})"
        )
    return norm


def validate_pip_packages(packages: list[str] | None) -> list[str]:
    """Validate PyPI package specs, rejecting pip flags, URLs, files, and paths."""
    if packages is None:
        return []
    if not isinstance(packages, (list, tuple)):
        raise ValueError("pip_packages must be a list of package specification strings")
    validated: list[str] = []
    for raw in packages:
        if not isinstance(raw, str):
            raise ValueError(f"Invalid pip package entry (must be str): {raw!r}")
        pkg = raw.strip()
        if not pkg or pkg != raw:
            raise ValueError(f"Invalid pip package (empty or surrounding whitespace): {raw!r}")
        if pkg.startswith("-"):
            raise ValueError(f"Pip flags are forbidden: {pkg!r}")
        lowered = pkg.lower()
        if "://" in pkg or "file:" in lowered or "/" in pkg or "\\" in pkg or ".." in pkg or "@" in pkg:
            raise ValueError(f"URLs, file references, and paths are forbidden in pip_packages: {pkg!r}")
        if not _PIP_PACKAGE_RE.fullmatch(pkg):
            raise ValueError(f"Invalid PyPI package specification: {pkg!r}")
        validated.append(pkg)
    return validated


def _validate_relative_posix_path(raw: str, *, kind: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"Invalid {kind} path: {raw!r}")
    if raw.startswith("/") or raw.startswith("\\") or "\\" in raw:
        raise ValueError(f"{kind} path must be a relative POSIX path: {raw!r}")
    if raw.endswith("/"):
        raise ValueError(f"{kind} path must reference a file, not a directory: {raw!r}")
    posix = PurePosixPath(raw)
    if posix.is_absolute() or not posix.parts:
        raise ValueError(f"Invalid {kind} path: {raw!r}")
    for part in posix.parts:
        if part in ("", ".", ".."):
            raise ValueError(f"Path traversal or dot segment forbidden in {kind} path: {raw!r}")
    return posix


def resolve_staged_inputs(
    io_dir: Path,
    rel_paths: list[str] | None,
    max_upload_mb: float,
) -> list[Path]:
    """Resolve and validate staged input files under `io_dir / 'in'` without symlinks."""
    if rel_paths is None:
        return []
    if not isinstance(rel_paths, (list, tuple)):
        raise ValueError("staged_inputs must be a list of relative file paths")

    in_dir = io_dir / "in"
    if in_dir.is_symlink():
        raise ValueError("Symlinks are forbidden in staged input root directory")
    in_root = in_dir.resolve()

    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()
    total_bytes = 0
    max_bytes = int(float(max_upload_mb) * 1024 * 1024)

    for raw in rel_paths:
        posix = _validate_relative_posix_path(raw, kind="staged_inputs")
        cur = in_dir
        for part in posix.parts:
            cur = cur / part
            if cur.is_symlink():
                raise ValueError(f"Symlink forbidden in staged input path: {raw!r}")
        resolved = cur.resolve()
        if not resolved.is_relative_to(in_root):
            raise ValueError(f"Staged input escapes input directory: {raw!r}")
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Staged input file not found under io_dir/in: {raw!r}")
        if resolved not in seen_paths:
            total_bytes += resolved.stat().st_size
            seen_paths.add(resolved)
        resolved_paths.append(resolved)

    if total_bytes > max_bytes:
        raise ValueError(
            f"Total staged upload size {total_bytes / (1024 * 1024):.2f} MB exceeds max_upload_mb={max_upload_mb}"
        )
    return resolved_paths


def validate_output_paths(rel_paths: list[str] | None) -> list[str]:
    """Validate requested Colab output relative paths (no traversal or absolute paths)."""
    if rel_paths is None:
        return []
    if not isinstance(rel_paths, (list, tuple)):
        raise ValueError("output_paths must be a list of relative file paths")
    out: list[str] = []
    seen: set[str] = set()
    for raw in rel_paths:
        posix = _validate_relative_posix_path(raw, kind="output_paths")
        norm = posix.as_posix()
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def verify_downloaded_outputs(
    out_job_dir: Path,
    rel_paths: list[str],
    max_download_mb: float,
    container_io_root: str,
    job_id: str,
) -> list[dict[str, Any]]:
    """Verify downloaded outputs contain no symlinks, stay within job out dir, and obey size cap."""
    if out_job_dir.is_symlink():
        raise ValueError("Symlink forbidden in job output directory")
    out_root = out_job_dir.resolve()
    max_bytes = int(float(max_download_mb) * 1024 * 1024)
    total_bytes = 0
    metadata: list[dict[str, Any]] = []

    for rel in rel_paths:
        posix = _validate_relative_posix_path(rel, kind="output_paths")
        cur = out_job_dir
        for part in posix.parts:
            cur = cur / part
            if cur.is_symlink():
                cur.unlink(missing_ok=True)
                raise ValueError(f"Downloaded output contains forbidden symlink: {rel!r}")
        resolved = cur.resolve()
        if not resolved.is_relative_to(out_root):
            raise ValueError(f"Downloaded output escapes job output directory: {rel!r}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Expected downloaded output missing: {rel!r}")

        size = resolved.stat().st_size
        total_bytes += size
        if total_bytes > max_bytes:
            for item_rel in rel_paths:
                (out_job_dir / item_rel).unlink(missing_ok=True)
            raise ValueError(
                f"Total downloaded output size {total_bytes / (1024 * 1024):.2f} MB "
                f"exceeds max_download_mb={max_download_mb}"
            )

        digest = hashlib.sha256()
        with open(resolved, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)

        metadata.append({
            "rel_path": posix.as_posix(),
            "container_path": f"{container_io_root.rstrip('/')}/out/{job_id}/{posix.as_posix()}",
            "bytes": size,
            "sha256": digest.hexdigest(),
        })
    return metadata


_URL_QUERY_RE = re.compile(r"(https?://[^\s'\"`?]+)\?[^\s'\"`<>)}\]]+")
_AUTH_HEADER_RE = re.compile(r"(?i)\b(Authorization)(\s*[:=]\s*)(?:Bearer\s+|Basic\s+)?[^\s,'\"`<>)}\]]+")
_SECRET_KV_RE = re.compile(
    r"(?i)\b(Bearer|access_token|id_token|refresh_token|authuser|X-Goog-[A-Za-z-]+)(\s*[:=]\s*|\s+)[^\s&,'\"`<>)}\]]+"
)


def sanitize_cli_output(text: str) -> str:
    """Scrub URL query strings (e.g., Colab proxy tokens) and auth tokens from CLI output."""
    if not text:
        return ""
    cleaned = _URL_QUERY_RE.sub(r"\1?<REDACTED>", text)
    cleaned = _AUTH_HEADER_RE.sub(r"\1\2<REDACTED>", cleaned)
    cleaned = _SECRET_KV_RE.sub(r"\1\2<REDACTED>", cleaned)
    return cleaned


def sanitized_colab_env() -> dict[str, str]:
    """Return a copy of `os.environ` with LLM/GCP/MCP secret variables stripped."""
    env = os.environ.copy()
    for key in _SENSITIVE_ENV_VARS:
        env.pop(key, None)
    return env
