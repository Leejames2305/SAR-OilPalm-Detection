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
from libbuilder.render_handoff import render_handoff_section

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


def build_task(
    session_id: str,
    *,
    focus_lens: str | None = None,
    parent_session: str | None = None,
    timeout_override: int | None = None,
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
    if prev_session_dir and (prev_session_dir / "HANDOFF.md").exists():
        shutil.copyfile(prev_session_dir / "HANDOFF.md", prior_dst / "prev_HANDOFF.md")

    # 4. Current incumbent pipeline (both in /app/incumbent and solution/ for oracle/check)
    for item in sorted(incumbent_dir.iterdir()):
        if item.is_file() and item.name != "__pycache__" and not item.name.endswith(".pyc"):
            shutil.copyfile(item, inc_dst / item.name)
            if item.name in ("pipeline.py", "requirements.txt"):
                shutil.copyfile(item, sol_dst / item.name)
    if not (sol_dst / "requirements.txt").exists():
        (sol_dst / "requirements.txt").write_text("")

    # 5. Optional prior derived datasets (size-capped)
    extra_copy_lines = ""
    max_ds_mb = float(session_cfg.get("prior_datasets_max_mb", 128))
    if prev_session_dir and (prev_session_dir / "datasets").exists():
        prev_ds = prev_session_dir / "datasets"
        if _dir_size_bytes(prev_ds) <= max_ds_mb * 1024 * 1024:
            ds_dst = env_dir / "datasets"
            shutil.copytree(
                prev_ds,
                ds_dst,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            if any(ds_dst.rglob("*")):
                extra_copy_lines = "COPY datasets/ /app/datasets/"

    agent_timeout = int(timeout_override or session_cfg["agent_timeout_sec"])
    agent_cpus = int(session_cfg["agent_cpus"])
    handoff_md = render_handoff_section(prev_session_dir, lineage)

    # Render templates (avoid doubling -<lens> when session_id already ends with -<lens>)
    task_name = session_id if session_id.endswith(f"-{lens}") else f"{session_id}-{lens}"
    (task_dir / "task.toml").write_text(
        _render_template(
            (templates_dir / "task.toml.tmpl").read_text(),
            {
                "TASK_NAME": task_name,
                "AGENT_TIMEOUT_SEC": str(float(agent_timeout)),
                "AGENT_CPUS": str(agent_cpus),
                "AGENT_MEMORY_MB": str(int(session_cfg["agent_memory_mb"])),
                "AGENT_STORAGE_MB": str(int(session_cfg["agent_storage_mb"])),
                "VERIFIER_TIMEOUT_SEC": str(int(session_cfg["verifier_timeout_sec"])),
                "VERIFIER_CPUS": str(int(session_cfg["verifier_cpus"])),
                "VERIFIER_MEMORY_MB": str(int(session_cfg["verifier_memory_mb"])),
            },
        )
    )
    (task_dir / "instruction.md").write_text(
        _render_template(
            (templates_dir / "instruction_base.md").read_text(),
            {
                "SESSION_ID": session_id,
                "AGENT_CPUS": str(agent_cpus),
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
        "cp /solution/pipeline.py /app/submission/pipeline.py\n"
        "cp /solution/requirements.txt /app/submission/requirements.txt\n"
    )
    os.chmod(solve_sh, 0o755)

    # Canary verification: zero T markers in instruction/task.toml, and 002-CheckLeaks PASS
    for doc in (task_dir / "instruction.md", task_dir / "task.toml"):
        txt_lc = doc.read_text().lower()
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
