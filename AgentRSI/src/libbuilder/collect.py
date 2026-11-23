"""Host-only session collector: secret scan, smell checks, artifact copy, lineage & events update."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from libbuilder.dashboard import generate_dashboard
from libbuilder.env_utils import AGENT_ROOT, load_dotenv
from libbuilder.events import append_event
from libbuilder.lineage import load_lineage, record_session_result, save_lineage
from libbuilder.render_handoff import validate_handoff

BINARY_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z",
    ".parquet", ".feather", ".npy", ".npz", ".pkl", ".joblib",
    ".pyc", ".pyo", ".whl", ".so", ".dll", ".dylib", ".bin",
}
URL_RE = re.compile(r"https?://[^\s'\"`>]+", re.IGNORECASE)
NET_CALL_RE = re.compile(
    r"\b(urllib|requests|httpx|aiohttp|curl|wget|earthengine|socket)\b|\bimport\s+ee\b|\bfrom\s+ee\b"
)
BLOCKED_REPO_RE = re.compile(r"SAR-OilPalm-Detection|Leejames2305", re.IGNORECASE)


def _is_binary(p: Path) -> bool:
    if p.suffix.lower() in BINARY_SUFFIXES:
        return True
    try:
        with open(p, "rb") as fh:
            return b"\x00" in fh.read(4096)
    except OSError:
        return True


def _secret_literals(agent_root: Path, env_path: Path | None = None) -> list[str]:
    literals: list[str] = ["BEGIN PRIVATE KEY", '"private_key"']
    env_map = load_dotenv(env_path or (agent_root / ".env"))
    api_key = env_map.get("API_KEY", "").strip()
    if len(api_key) >= 12:
        literals.append(api_key)
    secrets_dir = agent_root / ".secrets"
    if secrets_dir.exists():
        for jp in sorted(secrets_dir.glob("*.json")):
            try:
                data = json.loads(jp.read_text())
                if key_id := str(data.get("private_key_id", "")).strip():
                    literals.append(key_id)
            except Exception:
                pass
    return literals


def scan_for_secrets(roots: list[Path], secret_literals: list[str]) -> list[str]:
    """Return findings if any text file under `roots` contains a credential marker or literal."""
    findings: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in sorted(root.rglob("*")) if p.is_file()]
        for f in files:
            if _is_binary(f):
                continue
            text = f.read_text(errors="replace")
            for tok in secret_literals:
                if tok and tok in text:
                    findings.append(f"{f.name}: matched secret token")
                    break
    return findings


PROMPT_ECHO_RE = re.compile(
    r"access or clone [`'\"]?https://github\.com/Leejames2305/SAR-OilPalm-Detection[`'\"]?",
    re.IGNORECASE,
)


def scan_smells(submission_dir: Path, artifact_files: list[Path], agent_log_files: list[Path] | None = None) -> list[str]:
    """Detect grading-time network calls in submission code and blocked-repo references."""
    smells: list[str] = []
    if submission_dir.exists():
        for f in sorted(submission_dir.rglob("*")):
            if not f.is_file() or _is_binary(f):
                continue
            txt = f.read_text(errors="replace")
            for m in URL_RE.finditer(txt):
                smells.append(f"submission/{f.name}: URL {m.group(0)}")
            for m in NET_CALL_RE.finditer(txt):
                smells.append(f"submission/{f.name}: network/GEE ref '{m.group(0)}'")
    for f in artifact_files:
        if not f.is_file() or _is_binary(f):
            continue
        txt = f.read_text(errors="replace")
        if BLOCKED_REPO_RE.search(txt):
            smells.append(f"{f.name}: references blocked research repo")
    for f in (agent_log_files or []):
        if not f.is_file() or _is_binary(f):
            continue
        txt = PROMPT_ECHO_RE.sub("", f.read_text(errors="replace"))
        if BLOCKED_REPO_RE.search(txt):
            smells.append(f"{f.name}: references blocked research repo")
    return sorted(set(smells))


def _resolve_job_and_trial(session_id: str, job_dir: Path | None, jobs_root: Path) -> tuple[Path, Path]:
    if job_dir is None:
        base = jobs_root / session_id
        if not base.exists():
            raise FileNotFoundError(f"No jobs found under {base}")
        candidates = sorted(p for p in base.iterdir() if p.is_dir() and (p / "result.json").exists())
        if not candidates:
            raise FileNotFoundError(f"No completed job directories under {base}")
        resolved_job = candidates[-1]
    else:
        resolved_job = job_dir

    trials = [
        p
        for p in sorted(resolved_job.iterdir())
        if p.is_dir() and (p / "result.json").exists() and (p / "verifier").is_dir()
    ]
    if not trials:
        raise FileNotFoundError(f"No trial directory with verifier outputs found under {resolved_job}")
    return resolved_job, trials[-1]


def _find_artifact_subdir(artifacts_root: Path, name: str) -> Path | None:
    for cand in (artifacts_root / "app" / name, artifacts_root / name):
        if cand.exists():
            return cand
    return None


def collect_session(
    session_id: str,
    *,
    job_dir: Path | None = None,
    parent_id: str = "starter",
    dry_session: bool = False,
    extra_secrets: list[str] | None = None,
    agent_root: Path = AGENT_ROOT,
    jobs_root: Path | None = None,
    sessions_root: Path | None = None,
    lineage_path: Path | None = None,
    events_path: Path | None = None,
    env_path: Path | None = None,
    dashboard_path: Path | None = None,
    update_dashboard: bool = True,
) -> dict[str, Any]:
    """Collect a finished Harbor job strictly on host: secret scan, copy, lineage & events update."""
    cfg_dir = agent_root / "config"
    verifier_cfg = yaml.safe_load((cfg_dir / "verifier.yaml").read_text())
    split_cfg = yaml.safe_load((cfg_dir / "split.yaml").read_text())
    ep = events_path or (agent_root / "state" / "events.jsonl")
    lp = lineage_path or (agent_root / "state" / "lineage.json")
    lineage = load_lineage(lp, split_cfg, verifier_cfg)
    epoch = int(lineage["epoch"])

    resolved_job, trial_dir = _resolve_job_and_trial(
        session_id, job_dir, jobs_root or (agent_root / "jobs")
    )
    job_id = resolved_job.name

    # Secret scan runs BEFORE copying anything to sessions/
    secrets = _secret_literals(agent_root, env_path=env_path)
    if extra_secrets:
        secrets.extend(s for s in extra_secrets if s)
    secret_hits = scan_for_secrets([resolved_job], secrets)
    if secret_hits:
        append_event(
            ep,
            session=session_id,
            epoch=epoch,
            event_type="run_error",
            summary=f"Secret scan FAILED collection for {session_id}: {secret_hits[0]}",
            ref=f"jobs/{session_id}/{job_id}",
            dedup_id=f"{job_id}:secret_fail",
        )
        raise RuntimeError(f"Secret scan failed on {resolved_job}: {secret_hits}")

    verifier_dir = trial_dir / "verifier"
    reward = json.loads((verifier_dir / "reward.json").read_text())
    v_report = json.loads((verifier_dir / "v_report.json").read_text())
    certification = json.loads((verifier_dir / "certification.json").read_text())

    trial_result = json.loads((trial_dir / "result.json").read_text())
    job_result = json.loads((resolved_job / "result.json").read_text())
    agent_res = trial_result.get("agent_result") or {}
    agent_info = trial_result.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    cost_usd = agent_res.get("cost_usd")
    if cost_usd is None:
        cost_usd = job_result.get("stats", {}).get("cost_usd", 0.0)

    session_dir = (sessions_root or (agent_root / "sessions")) / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Copy verifier reports
    for fn in ("reward.json", "v_report.json", "certification.json"):
        shutil.copyfile(verifier_dir / fn, session_dir / fn)

    artifacts_root = trial_dir / "artifacts"
    sub_src = _find_artifact_subdir(artifacts_root, "submission")
    if sub_src and any(sub_src.iterdir()):
        shutil.copytree(sub_src, session_dir / "submission", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    rep_src = _find_artifact_subdir(artifacts_root, "report")
    handoff_missing: list[str] = []
    if rep_src:
        for doc_name in ("REPORT.md", "HANDOFF.md"):
            p = rep_src / doc_name
            if p.exists():
                shutil.copyfile(p, session_dir / doc_name)
                if doc_name == "HANDOFF.md":
                    handoff_missing = validate_handoff(p.read_text())

    ds_src = _find_artifact_subdir(artifacts_root, "datasets")
    if ds_src and any(ds_src.iterdir()):
        shutil.copytree(ds_src, session_dir / "datasets", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Smell scan (submission network/URLs + blocked repo in artifacts/agent logs)
    artifact_files = [p for p in sorted(artifacts_root.rglob("*")) if p.is_file()] if artifacts_root.exists() else []
    agent_dir = trial_dir / "agent"
    agent_logs = [p for p in sorted(agent_dir.rglob("*")) if p.is_file()] if agent_dir.exists() else []
    smells = scan_smells(session_dir / "submission", artifact_files, agent_logs)

    meta = {
        "session_id": session_id,
        "job_id": job_id,
        "trial_name": trial_dir.name,
        "parent_id": parent_id,
        "dry_session": bool(dry_session),
        "agent": agent_info.get("name"),
        "model": model_info.get("name"),
        "started_at": trial_result.get("started_at"),
        "finished_at": trial_result.get("finished_at"),
        "cost_usd": float(cost_usd or 0.0),
        "n_input_tokens": agent_res.get("n_input_tokens"),
        "n_cache_tokens": agent_res.get("n_cache_tokens"),
        "n_output_tokens": agent_res.get("n_output_tokens"),
        "exception_info": trial_result.get("exception_info"),
        "smells": smells,
        "handoff_missing_sections": handoff_missing,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    sub_rel = f"sessions/{session_id}/submission"
    rec = record_session_result(
        lineage,
        session_id=session_id,
        job_id=job_id,
        parent_id=parent_id,
        meta=meta,
        reward=reward,
        v_report=v_report,
        certification=certification,
        submission_rel_dir=sub_rel,
        dry_session=dry_session,
    )
    save_lineage(lp, lineage)

    # Idempotent event updates (deduped on job_id)
    if v_report.get("submission_valid"):
        append_event(
            ep,
            session=session_id,
            epoch=epoch,
            event_type="nominee_declared",
            summary=f"Nominee graded for {session_id}: V lift {reward['score']:+.6f}",
            ref=f"sessions/{session_id}/v_report.json",
            values={"v_score": float(reward["score"]), "v_delta_vs_incumbent": float(reward["v_delta_vs_incumbent"])},
            dedup_id=f"{job_id}:nominee",
        )
    if not dry_session:
        cert_ok = bool(certification.get("gate_passed"))
        append_event(
            ep,
            session=session_id,
            epoch=epoch,
            event_type="certification_pass" if cert_ok else "certification_fail",
            summary=(
                f"Certification {'PASS' if cert_ok else 'FAIL'} for {session_id} "
                f"(T dPR={certification['composite']['comp_delta']:+.6f})"
            ),
            ref=f"sessions/{session_id}/certification.json",
            values={
                "comp_delta": float(certification["composite"]["comp_delta"]),
                "ci_low": float(certification["composite"]["ci_low"]),
            },
            dedup_id=f"{job_id}:cert",
        )
        if rec["promoted"]:
            append_event(
                ep,
                session=session_id,
                epoch=epoch,
                event_type="promoted",
                summary=f"Promoted {session_id} as new incumbent (V={reward['score']:+.6f})",
                ref=f"sessions/{session_id}/REPORT.md",
                values={"v_score": float(reward["score"])},
                dedup_id=f"{job_id}:promoted",
            )
        budget = int(lineage.get("t_attempt_budget", 20))
        used = int(lineage.get("t_attempts_used", 0))
        if used >= int(0.8 * budget):
            append_event(
                ep,
                session=session_id,
                epoch=epoch,
                event_type="budget_alert",
                summary=f"T-attempt budget alert: {used}/{budget} used",
                ref=f"sessions/{session_id}/meta.json",
                values={"t_attempts_used": used, "t_attempt_budget": budget},
                dedup_id=f"{job_id}:budget_{used}",
            )

    if smells or handoff_missing:
        notes = []
        if smells:
            notes.append(f"smells={smells}")
        if handoff_missing:
            notes.append(f"handoff_missing={handoff_missing}")
        append_event(
            ep,
            session=session_id,
            epoch=epoch,
            event_type="review_flag",
            summary=f"Review flag for {session_id}: " + "; ".join(notes),
            ref=f"sessions/{session_id}/meta.json",
            dedup_id=f"{job_id}:review",
        )

    # Final leak check on collected session folder
    leak_script = agent_root / "src" / "DataPipeline" / "002-CheckLeaks.py"
    res = subprocess.run(
        [sys.executable, str(leak_script), "--no-view", "--scan", str(session_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Leak check failed on collected session {session_dir}:\n{res.stdout}\n{res.stderr}")

    out: dict[str, Any] = {"session_dir": str(session_dir), "meta": meta, "lineage_update": rec}
    if update_dashboard:
        sr = sessions_root or (agent_root / "sessions")
        dash_out = dashboard_path or (
            (sessions_root.parent / "dashboard" / "index.html")
            if sessions_root is not None
            else (agent_root / "dashboard" / "index.html")
        )
        generate_dashboard(
            agent_root,
            output_path=dash_out,
            sessions_root=sr,
            lineage_path=lp,
            events_path=ep,
        )
        out["dashboard_path"] = str(dash_out)

    return out
