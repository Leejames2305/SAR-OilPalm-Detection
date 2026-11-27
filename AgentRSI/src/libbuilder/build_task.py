"""Allowlist task builder: renders tasks/<session_id>/ from templates, agent_view, rasters, and prior."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from libbuilder.env_utils import AGENT_ROOT, REPO_ROOT
from libbuilder.lineage import load_lineage
from libbuilder.render_handoff import find_latest_handoff_path, render_handoff_section

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SESSION_SEQ_RE = re.compile(r"(?:^|-)s(\d+)(?:-|$)")


def infer_lens_from_session_id(session_id: str, lenses_dir: Path) -> str | None:
    """Return matching lens stem if `session_id` ends with `-<lens>` (longest match first)."""
    if not lenses_dir.exists():
        return None
    stems = sorted((p.stem for p in lenses_dir.glob("*.md")), key=len, reverse=True)
    for stem in stems:
        if session_id.endswith(f"-{stem}"):
            return stem
    return None


def next_session_seq(lineage: dict[str, Any], sessions_root: Path | None = None) -> int:
    """Return next sequential session number (min 1) across lineage and sessions/ directories."""
    max_seq = 0
    names: set[str] = set((lineage.get("sessions") or {}).keys())
    if sessions_root is not None and sessions_root.exists():
        names.update(p.name for p in sessions_root.iterdir() if p.is_dir())
    for name in names:
        m = SESSION_SEQ_RE.search(name)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1


def make_session_id(seq: int, lens: str, *, date_str: str | None = None) -> str:
    """Format a `YYMMDD-sXXX-<lens>` session identifier."""
    day = date_str or datetime.now().strftime("%y%m%d")
    sid = f"{day}-s{int(seq):03d}-{lens}"
    if not SESSION_ID_RE.match(sid):
        raise ValueError(f"Generated invalid session_id {sid!r}")
    return sid


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def _render_template(text: str, mapping: dict[str, str]) -> str:
    out = text
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


def _copy_incumbent_bundle(
    incumbent_dir: Path,
    dst_submission: Path,
    dst_datasets: Path,
    *,
    copy_solution_to: Path | None = None,
    touch_empty_datasets_keep: bool = False,
) -> None:
    """Copy incumbent submission files and optional prior datasets to target dirs."""
    dst_submission.mkdir(parents=True, exist_ok=True)
    dst_datasets.mkdir(parents=True, exist_ok=True)
    for item in sorted(incumbent_dir.iterdir()):
        if item.is_file() and item.name != "__pycache__" and not item.name.endswith(".pyc"):
            shutil.copyfile(item, dst_submission / item.name)
            if copy_solution_to is not None and (
                item.name in ("pipeline.py", "requirements.txt") or item.suffix == ".py"
            ):
                shutil.copyfile(item, copy_solution_to / item.name)
    if not (dst_submission / "requirements.txt").exists():
        (dst_submission / "requirements.txt").write_text("")
    if copy_solution_to is not None and not (copy_solution_to / "requirements.txt").exists():
        (copy_solution_to / "requirements.txt").write_text("")

    inc_session_ds = incumbent_dir.parent / "datasets"
    if incumbent_dir.name == "submission" and inc_session_ds.exists():
        shutil.copytree(
            inc_session_ds,
            dst_datasets,
            dirs_exist_ok=True,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    if touch_empty_datasets_keep and not any(dst_datasets.iterdir()):
        (dst_datasets / ".keep").write_text("")


def _render_mcp_servers_toml(*, inrun_enabled: bool, inrun_port: int, colab_enabled: bool) -> str:
    """Render `[[environment.mcp_servers]]` TOML blocks for enabled M7/M8 MCP servers."""
    blocks: list[str] = []
    if inrun_enabled:
        blocks.append(
            "\n".join([
                "[[environment.mcp_servers]]",
                'name = "agentrsi-verifier"',
                'transport = "streamable-http"',
                f'url = "http://verifier-sidecar:{inrun_port}/mcp"',
                "",
                "[environment.mcp_servers.headers]",
                'Authorization = "Bearer ${AGENTRSI_VERIFIER_MCP_TOKEN}"',
            ])
        )
    if colab_enabled:
        blocks.append(
            "\n".join([
                "[[environment.mcp_servers]]",
                'name = "agentrsi-colab"',
                'transport = "stdio"',
                'command = "python3"',
                'args = ["/usr/local/bin/agentrsi_mcp_relay.py", "--socket", "/run/agentrsi/colab-mcp.sock"]',
            ])
        )
    if not blocks:
        return ""
    return "\n" + "\n\n".join(blocks) + "\n"


def _render_mcp_tools_section(
    *,
    inrun_enabled: bool,
    inrun_cfg: dict[str, Any],
    colab_enabled: bool,
    colab_cfg: dict[str, Any],
    allow_a100: bool,
) -> str:
    """Render conditional `### MCP Tools` documentation for `instruction.md`."""
    if not inrun_enabled and not colab_enabled:
        return ""
    parts: list[str] = ["\n### MCP Tools\n"]
    if inrun_enabled:
        max_q = int(inrun_cfg.get("max_queries", 15))
        sigma = float(inrun_cfg.get("noise_sigma", 0.005))
        parts.append(
            "\n".join([
                "#### `agentrsi-verifier` (In-Run V-Split Scorer)",
                f"- Sidecar MCP server exposing `score_v`, `verifier_budget`, and `verifier_health` (query budget: `{max_q}` evaluations; additive Gaussian noise $\\sigma = {sigma}$ cached per submission content hash).",
                "- Returns noisy validation-split (`V`) composite and per-estate PR-AUC lift plus deltas vs incumbent and coords-only baseline.",
                "- Stage larger submissions or derived datasets under `/shared/<subdir>/` (e.g., `/shared/sub_01/pipeline.py`, `/shared/sub_01/requirements.txt`, and optional `/shared/ds_01/`) and pass `shared_submission_dir` / `shared_datasets_dir` to `score_v(...)`, or pass inline `pipeline_code` + `requirements_txt` for small candidates.",
                "- `verifier_budget` and `verifier_health` do not consume query budget.",
            ])
        )
    if colab_enabled:
        gpus = [str(g) for g in (colab_cfg.get("allowed_gpus") or ["T4", "L4"])]
        if allow_a100 and "A100" not in gpus:
            gpus.append("A100")
        gpus_text = ", ".join(f"`{g}`" for g in gpus)
        max_job_min = int(colab_cfg.get("max_job_timeout_min", 30))
        max_sess_min = int(colab_cfg.get("max_session_gpu_min", 60))
        parts.append(
            "\n".join([
                "#### `agentrsi-colab` (Remote GPU Offload Broker)",
                "- Host-managed MCP server exposing `colab_run`, `colab_status`, `colab_fetch`, and `colab_stop` for GPU-accelerated training or feature extraction.",
                f"- Allowed GPUs: {gpus_text}; per-job timeout cap: `{max_job_min}` min; session GPU cap: `{max_sess_min}` min. At most **1 active Colab VM/job** may run at a time.",
                "- Stage inputs under `/run/agentrsi/colab-io/in/` and pass relative paths in `staged_inputs`; outputs are downloaded to `/run/agentrsi/colab-io/out/<job_id>/` (copy reusable features into `/app/datasets/<name>/`).",
                "- Final `/app/submission/pipeline.py` still executes **CPU-only and offline** during grading; call `colab_stop` when GPU work finishes.",
            ])
        )
    return "\n".join(parts) + "\n"


def build_task(
    session_id: str,
    *,
    focus_lens: str | None = None,
    parent_session: str | None = None,
    ref_sessions: list[str] | None = None,
    datasets_from: str | None = None,
    timeout_override: int | None = None,
    agent_cpus_override: int | None = None,
    agent_memory_mb_override: int | None = None,
    verifier_cpus_override: int | None = None,
    verifier_memory_mb_override: int | None = None,
    enable_inrun_verifier: bool | None = None,
    enable_colab: bool | None = None,
    colab_allow_a100: bool | None = None,
    output_root: Path | None = None,
    agent_root: Path = AGENT_ROOT,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Build a Harbor task directory for `session_id` using a strict allowlist copy."""
    if not SESSION_ID_RE.match(session_id):
        raise ValueError(f"Invalid session_id {session_id!r}")

    cfg_dir = agent_root / "config"
    session_cfg = yaml.safe_load((cfg_dir / "session.yaml").read_text())
    verifier_cfg = yaml.safe_load((cfg_dir / "verifier.yaml").read_text())
    split_cfg = yaml.safe_load((cfg_dir / "split.yaml").read_text())
    lineage = load_lineage(agent_root / "state" / "lineage.json", split_cfg, verifier_cfg)

    templates_dir = agent_root / "templates"
    inferred_lens = infer_lens_from_session_id(session_id, templates_dir / "lenses")
    lens = focus_lens or inferred_lens or str(session_cfg.get("focus_lens", "open"))
    lens_path = templates_dir / "lenses" / f"{lens}.md"
    if not lens_path.exists():
        raise FileNotFoundError(f"Unknown focus lens {lens!r}: {lens_path} not found")

    if parent_session is not None:
        prev_session_dir: Path | None = agent_root / "sessions" / parent_session
        if not prev_session_dir.exists():
            raise FileNotFoundError(f"Parent session directory missing: {prev_session_dir}")
    else:
        prev_session_dir = None

    ref_session_dirs: list[Path] = []
    seen_refs: set[str] = set()
    for ref_id in ref_sessions or []:
        if not ref_id or ref_id in seen_refs or ref_id == parent_session:
            continue
        seen_refs.add(ref_id)
        rdir = agent_root / "sessions" / ref_id
        if not rdir.exists():
            raise FileNotFoundError(f"Reference session directory missing: {rdir}")
        ref_session_dirs.append(rdir)

    inc_rel = str(lineage.get("incumbent", {}).get("source_dir", "starter"))
    incumbent_dir = (agent_root / inc_rel).resolve()
    if not (incumbent_dir / "pipeline.py").exists():
        raise FileNotFoundError(f"Incumbent pipeline.py missing under {incumbent_dir}")

    task_dir = (output_root or (agent_root / "tasks")) / session_id
    if task_dir.exists():
        shutil.rmtree(task_dir)

    env_dir = task_dir / "environment"
    data_dst = env_dir / "data"
    rasters_dst = env_dir / "rasters"
    prior_dst = env_dir / "prior"
    poc_dst = prior_dst / "POCReport"
    inc_dst = env_dir / "incumbent"
    tests_dst = task_dir / "tests"
    sol_dst = task_dir / "solution"
    for d in (data_dst, rasters_dst, poc_dst, inc_dst, tests_dst, sol_dst):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Allowlisted agent_view CSVs (L + merged unlabelled V+T)
    view_dir = agent_root / ".work" / "agent_view"
    for fname in ("L_labelled.csv", "unlabelled.csv"):
        src = view_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing agent view file: {src}")
        shutil.copyfile(src, data_dst / fname)

    # 2. Allowlisted ROI-masked SAR rasters
    sar_dir = repo_root / "data" / "SAR-scenes"
    tifs = sorted(sar_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif scenes found under {sar_dir}")
    for tif in tifs:
        shutil.copyfile(tif, rasters_dst / tif.name)

    # 3. Allowlisted prior docs (never Plan-AgentRSI.md or raw labels)
    poc_dir = repo_root / "docs" / "POCReport"
    for md in sorted(poc_dir.glob("*.md")):
        shutil.copyfile(md, poc_dst / md.name)
    for doc_name in ("MainChecklist.md", "Plan-GEECatalogSurvey.md"):
        doc_src = repo_root / "docs" / "AgentPlan" / doc_name
        if doc_src.exists():
            shutil.copyfile(doc_src, prior_dst / doc_name)
    handoff_fallback = find_latest_handoff_path(prev_session_dir, lineage)
    if handoff_fallback is not None:
        shutil.copyfile(handoff_fallback[1], prior_dst / "prev_HANDOFF.md")
    if prev_session_dir is not None:
        if (prev_session_dir / "REPORT.md").is_file():
            shutil.copyfile(prev_session_dir / "REPORT.md", prior_dst / "prev_REPORT.md")
        prev_sub_src = prev_session_dir / "submission"
        if prev_sub_src.is_dir() and any(prev_sub_src.iterdir()):
            shutil.copytree(
                prev_sub_src,
                prior_dst / "prev_submission",
                dirs_exist_ok=True,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    for rdir in ref_session_dirs:
        ref_dst = prior_dst / "ref_sessions" / rdir.name
        ref_dst.mkdir(parents=True, exist_ok=True)
        for doc_name in ("HANDOFF.md", "REPORT.md"):
            doc_src = rdir / doc_name
            if doc_src.is_file():
                shutil.copyfile(doc_src, ref_dst / doc_name)
        ref_sub_src = rdir / "submission"
        if ref_sub_src.is_dir() and any(ref_sub_src.iterdir()):
            shutil.copytree(
                ref_sub_src,
                ref_dst / "submission",
                dirs_exist_ok=True,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

    # 4. Current incumbent pipeline (in /app/incumbent, solution/ for oracle, and tests/incumbent for verifier)
    for item in sorted(incumbent_dir.iterdir()):
        if item.is_file() and item.name != "__pycache__" and not item.name.endswith(".pyc"):
            shutil.copyfile(item, inc_dst / item.name)
    _copy_incumbent_bundle(
        incumbent_dir,
        tests_dst / "incumbent" / "submission",
        tests_dst / "incumbent" / "datasets",
        copy_solution_to=sol_dst,
    )

    # Resolve M7/M8 toggles against session.yaml defaults
    inrun_cfg = verifier_cfg.get("inrun") or {}
    colab_cfg = session_cfg.get("colab_broker") or {}
    inrun_enabled = (
        bool(session_cfg.get("inrun_verifier", False))
        if enable_inrun_verifier is None
        else bool(enable_inrun_verifier)
    )
    colab_enabled = (
        bool(colab_cfg.get("enabled", False))
        if enable_colab is None
        else bool(enable_colab)
    )
    allow_a100 = (
        bool(colab_cfg.get("allow_a100", False))
        if colab_allow_a100 is None
        else bool(colab_allow_a100)
    )
    inrun_port = int(inrun_cfg.get("port", 8765))

    if inrun_enabled:
        sidecar_dir = env_dir / "sidecar"
        _copy_incumbent_bundle(
            incumbent_dir,
            sidecar_dir / "incumbent" / "submission",
            sidecar_dir / "incumbent" / "datasets",
            touch_empty_datasets_keep=True,
        )
        (sidecar_dir / "Dockerfile").write_text(
            _render_template(
                (templates_dir / "Dockerfile.sidecar.tmpl").read_text(),
                {"SEALED_BASE_IMAGE": str(verifier_cfg["sealed_base_image"])},
            )
        )
        (env_dir / "docker-compose.yaml").write_text(
            _render_template(
                (templates_dir / "docker-compose.inrun.yaml.tmpl").read_text(),
                {
                    "INRUN_PORT": str(inrun_port),
                    "SIDECAR_CPUS": str(inrun_cfg.get("sidecar_cpus", 2)),
                    "SIDECAR_MEMORY_MB": str(int(inrun_cfg.get("sidecar_memory_mb", 2048))),
                },
            )
        )

    extra_artifacts_toml = (
        ', { source = "/var/log/agentrsi-verifier/queries.jsonl", destination = "sidecar/queries.jsonl", service = "verifier-sidecar" }'
        if inrun_enabled
        else ""
    )
    mcp_servers_toml = _render_mcp_servers_toml(
        inrun_enabled=inrun_enabled,
        inrun_port=inrun_port,
        colab_enabled=colab_enabled,
    )
    mcp_tools_section = _render_mcp_tools_section(
        inrun_enabled=inrun_enabled,
        inrun_cfg=inrun_cfg,
        colab_enabled=colab_enabled,
        colab_cfg=colab_cfg,
        allow_a100=allow_a100,
    )

    # 5. Optional prior derived datasets (size-capped; --datasets-from overrides default carry-over)
    extra_copy_lines = ""
    max_ds_mb = float(session_cfg.get("prior_datasets_max_mb", 128))
    ds_sources: list[Path] = []
    if datasets_from is not None:
        explicit_ds = agent_root / "sessions" / datasets_from / "datasets"
        if not explicit_ds.exists():
            raise FileNotFoundError(f"Requested --datasets-from directory missing: {explicit_ds}")
        ds_sources.append(explicit_ds)
    else:
        if incumbent_dir.name == "submission" and (incumbent_dir.parent / "datasets").exists():
            ds_sources.append(incumbent_dir.parent / "datasets")
        if prev_session_dir and (prev_session_dir / "datasets").exists():
            pds = prev_session_dir / "datasets"
            if pds not in ds_sources:
                ds_sources.append(pds)

    if ds_sources:
        ds_dst = env_dir / "datasets"
        staged_bytes = 0
        for src_ds in ds_sources:
            src_bytes = _dir_size_bytes(src_ds)
            if staged_bytes + src_bytes <= max_ds_mb * 1024 * 1024:
                shutil.copytree(
                    src_ds,
                    ds_dst,
                    dirs_exist_ok=True,
                    symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                staged_bytes = _dir_size_bytes(ds_dst)
        if ds_dst.exists() and any(ds_dst.rglob("*")):
            extra_copy_lines = "COPY datasets/ /app/datasets/"

    agent_timeout = int(timeout_override or session_cfg["agent_timeout_sec"])
    agent_cpus = int(agent_cpus_override or session_cfg["agent_cpus"])
    agent_memory_mb = int(agent_memory_mb_override or session_cfg["agent_memory_mb"])
    verifier_cpus = int(verifier_cpus_override or session_cfg["verifier_cpus"])
    verifier_memory_mb = int(verifier_memory_mb_override or session_cfg["verifier_memory_mb"])
    handoff_md = render_handoff_section(prev_session_dir, lineage, ref_session_dirs=ref_session_dirs)

    # Render templates (avoid doubling -<lens> when session_id already ends with -<lens>)
    task_name = session_id if session_id.endswith(f"-{lens}") else f"{session_id}-{lens}"
    (task_dir / "task.toml").write_text(
        _render_template(
            (templates_dir / "task.toml.tmpl").read_text(),
            {
                "TASK_NAME": task_name,
                "EXTRA_ARTIFACTS_TOML": extra_artifacts_toml,
                "AGENT_TIMEOUT_SEC": str(float(agent_timeout)),
                "AGENT_CPUS": str(agent_cpus),
                "AGENT_MEMORY_MB": str(agent_memory_mb),
                "AGENT_STORAGE_MB": str(int(session_cfg["agent_storage_mb"])),
                "MCP_SERVERS_TOML": mcp_servers_toml,
                "VERIFIER_TIMEOUT_SEC": str(int(session_cfg["verifier_timeout_sec"])),
                "VERIFIER_CPUS": str(verifier_cpus),
                "VERIFIER_MEMORY_MB": str(verifier_memory_mb),
            },
        )
    )
    (task_dir / "instruction.md").write_text(
        _render_template(
            (templates_dir / "instruction_base.md").read_text(),
            {
                "SESSION_ID": session_id,
                "AGENT_TIMEOUT_SEC": str(agent_timeout),
                "AGENT_CPUS": str(agent_cpus),
                "AGENT_MEMORY_MB": str(agent_memory_mb),
                "VERIFIER_CPUS": str(verifier_cpus),
                "VERIFIER_MEMORY_MB": str(verifier_memory_mb),
                "PER_ESTATE_TIMEOUT_SEC": str(int(verifier_cfg["runtime"]["per_estate_timeout_sec"])),
                "PIP_INSTALL_TIMEOUT_SEC": str(int(verifier_cfg["runtime"]["pip_install_timeout_sec"])),
                "MAX_DATASETS_MB": str(int(verifier_cfg["runtime"]["max_datasets_mb"])),
                "PRIOR_DATASETS_MAX_MB": str(int(max_ds_mb)),
                "MCP_TOOLS_SECTION": mcp_tools_section,
                "FOCUS_LENS_SECTION": lens_path.read_text().strip(),
                "HANDOFF_SECTION": handoff_md.strip(),
            },
        )
    )
    (env_dir / "Dockerfile").write_text(
        _render_template(
            (templates_dir / "Dockerfile.agent.tmpl").read_text(),
            {
                "AGENT_BASE_IMAGE": str(session_cfg["agent_base_image"]),
                "EXTRA_COPY_LINES": extra_copy_lines,
            },
        )
    )
    (tests_dst / "Dockerfile").write_text(
        _render_template(
            (templates_dir / "Dockerfile.verifier.tmpl").read_text(),
            {"SEALED_BASE_IMAGE": str(verifier_cfg["sealed_base_image"])},
        )
    )
    test_sh = tests_dst / "test.sh"
    test_sh.write_text((templates_dir / "test.sh.tmpl").read_text())
    os.chmod(test_sh, 0o755)

    solve_sh = sol_dst / "solve.sh"
    solve_sh.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "mkdir -p /app/submission /app/report /app/datasets\n"
        "cp /solution/*.py /app/submission/\n"
        "cp /solution/requirements.txt /app/submission/requirements.txt\n"
    )
    os.chmod(solve_sh, 0o755)

    # Canary verification: zero unreplaced placeholders, zero T markers, and 002-CheckLeaks PASS
    unreplaced_re = re.compile(r"\{\{[A-Z0-9_]+\}\}")
    for doc in (task_dir / "instruction.md", task_dir / "task.toml"):
        raw_txt = doc.read_text()
        leftovers = unreplaced_re.findall(raw_txt)
        if leftovers:
            raise RuntimeError(f"Unreplaced template placeholder(s) {leftovers} found in {doc}")
        txt_lc = raw_txt.lower()
        for forbidden in ("gate_passed", "certification"):
            if forbidden in txt_lc:
                raise RuntimeError(f"T-leak canary triggered: {forbidden!r} found in {doc}")

    leak_script = agent_root / "src" / "DataPipeline" / "002-CheckLeaks.py"
    res = subprocess.run(
        [sys.executable, str(leak_script), "--no-view", "--scan", str(task_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Leak check failed on built task {task_dir}:\n{res.stdout}\n{res.stderr}")

    return task_dir
