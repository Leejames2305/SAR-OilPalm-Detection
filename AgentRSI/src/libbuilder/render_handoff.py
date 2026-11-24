"""Handoff validation, ancestor fallback, and verified-V rendering for the next session's instruction.md."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = (
    "State of play",
    "What I tried",
    "Dead ends",
    "Open leads, ranked",
    "How to resume",
    "Artifact index",
)
SCORE_CLAIM_RE = re.compile(
    r"(?:v[_\s-]*score|composite\s+lift|pr[_\s-]*auc[_\s-]*lift)\s*[:=]\s*([+-]?\d+\.\d+)",
    re.IGNORECASE,
)


def validate_handoff(text: str) -> list[str]:
    """Return missing required H2 section titles from HANDOFF.md text."""
    headings = {m.strip().lower() for m in re.findall(r"^##\s+(.+?)\s*$", text, re.MULTILINE)}
    return [sec for sec in REQUIRED_SECTIONS if sec.lower() not in headings]


def find_latest_handoff_path(
    prev_session_dir: Path | None,
    lineage: dict[str, Any],
) -> tuple[str, Path] | None:
    """Return `(session_id, HANDOFF.md path)` for `prev_session_dir`, or walk ancestors on crash/missing."""
    if prev_session_dir is None or not prev_session_dir.exists():
        return None

    direct = prev_session_dir / "HANDOFF.md"
    if direct.exists():
        return prev_session_dir.name, direct

    sessions_root = prev_session_dir.parent
    sessions_meta = lineage.get("sessions") or {}
    visited = {prev_session_dir.name}
    cur_id = str(sessions_meta.get(prev_session_dir.name, {}).get("parent_id") or "")

    while cur_id and cur_id != "starter" and cur_id not in visited:
        visited.add(cur_id)
        cand = sessions_root / cur_id / "HANDOFF.md"
        if cand.exists():
            return cur_id, cand
        cur_id = str(sessions_meta.get(cur_id, {}).get("parent_id") or "")

    inc_id = str((lineage.get("incumbent") or {}).get("id") or "")
    if inc_id and inc_id != "starter" and inc_id not in visited:
        cand = sessions_root / inc_id / "HANDOFF.md"
        if cand.exists():
            return inc_id, cand

    return None


def _trim_handoff(text: str, limit: int = 12000) -> str:
    trimmed = text.strip()
    if len(trimmed) > limit:
        return trimmed[:limit] + "\n\n... [truncated]"
    return trimmed


def render_handoff_section(prev_session_dir: Path | None, lineage: dict[str, Any]) -> str:
    """Render verifier-backed V summary + previous/ancestor HANDOFF.md (never reads certification.json!)."""
    inc = lineage.get("incumbent", {})
    inc_id = inc.get("id", "starter")
    inc_v = float(inc.get("v_score", 0.0))
    lines = [
        "## Handoff From Previous Session",
        f"- **Current incumbent:** `{inc_id}` (verified composite V lift `{inc_v:+.6f}`)",
        f"- **T attempts used:** `{lineage.get('t_attempts_used', 0)} / {lineage.get('t_attempt_budget', 20)}` (Epoch `{lineage.get('epoch', 1)}`)",
    ]

    if prev_session_dir is None or not prev_session_dir.exists():
        lines.append("- **Previous session:** None (initial session from Session-0 NDVI-LogReg starter).")
        return "\n".join(lines) + "\n"

    reward_path = prev_session_dir / "reward.json"
    v_rep_path = prev_session_dir / "v_report.json"
    meta_path = prev_session_dir / "meta.json"
    handoff_path = prev_session_dir / "HANDOFF.md"
    prev_id = prev_session_dir.name
    prev_meta = lineage.get("sessions", {}).get(prev_id, {})
    was_promoted = bool(prev_meta.get("promoted"))

    lines.append(f"- **Previous session (`{prev_id}`):** promoted = `{was_promoted}`")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            exc_info = meta.get("exception_info")
            if isinstance(exc_info, dict) and exc_info.get("exception_type"):
                lines.append(
                    f"- **Previous session interruption:** exited early with `{exc_info['exception_type']}` "
                    "(scores below may reflect an unfinished mid-experiment checkpoint rather than a concluded result)."
                )
        except Exception:
            pass

    reward: dict[str, Any] | None = None
    if reward_path.exists() and v_rep_path.exists():
        reward = json.loads(reward_path.read_text())
        v_rep = json.loads(v_rep_path.read_text())
        lines.extend([
            f"- **Verifier-verified V score (`reward.json`):** `{reward['score']:+.6f}` "
            f"(Δ vs incumbent `{reward['v_delta_vs_incumbent']:+.6f}`, "
            f"Δ vs coords `{reward['v_delta_vs_coords']:+.6f}`, "
            f"valid=`{bool(reward['submission_valid'])}`, status=`{v_rep.get('status')}`)",
            "  - Per-estate V lift: "
            + ", ".join(
                f"{e} `{reward[f'v_lift_{e}']:+.6f}`"
                for e in ("AirHitam", "Palong", "Serting")
                if f"v_lift_{e}" in reward
            ),
        ])
        if v_rep.get("error"):
            lines.append(f"  - Verifier error: `{v_rep['error']}`")
        ds_info = (v_rep.get("datasets_audit") or {}).get("datasets") or {}
        if ds_info:
            ds_desc = ", ".join(
                f"`/app/datasets/{name}/{info.get('file', 'features.parquet')}` "
                f"({info.get('rows', '?')} rows × {info.get('cols', '?')} cols)"
                for name, info in sorted(ds_info.items())
            )
            lines.append(f"- **Derived datasets carried over from `{prev_id}`:** {ds_desc}")
    else:
        lines.append("- **Verifier-verified V score:** missing `reward.json`/`v_report.json` in previous session.")

    if handoff_path.exists():
        handoff_text = handoff_path.read_text()
        missing = validate_handoff(handoff_text)
        if missing:
            lines.append(f"- **Handoff schema note:** missing sections `{missing}`")
        if reward is not None:
            disagreements = []
            for m in SCORE_CLAIM_RE.finditer(handoff_text):
                claimed = float(m.group(1))
                if abs(claimed - float(reward["score"])) > 0.002:
                    disagreements.append(f"{claimed:+.6f} vs verifier {float(reward['score']):+.6f}")
            if disagreements:
                lines.append(
                    "- **WARNING (verifier vs agent HANDOFF disagreement):** "
                    + "; ".join(disagreements)
                    + " — always trust verifier numbers above."
                )
        lines.extend(["", "### Previous HANDOFF.md", "", _trim_handoff(handoff_text)])
    else:
        lines.append(
            f"- **Handoff note:** `{prev_id}` did not produce `HANDOFF.md` before exiting."
        )
        fallback = find_latest_handoff_path(prev_session_dir, lineage)
        if fallback is not None:
            anc_id, anc_path = fallback
            lines.extend([
                f"- **Fallback handoff:** loaded most recent surviving handoff from ancestor `{anc_id}`.",
                "",
                f"### Last Available HANDOFF.md (from `{anc_id}`)",
                "",
                _trim_handoff(anc_path.read_text()),
            ])

    return "\n".join(lines) + "\n"
