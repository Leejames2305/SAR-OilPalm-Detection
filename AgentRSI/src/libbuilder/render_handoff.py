"""Handoff validation and verified-V rendering for the next session's instruction.md."""
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


def render_handoff_section(prev_session_dir: Path | None, lineage: dict[str, Any]) -> str:
    """Render verifier-backed V summary + previous HANDOFF.md (never reads certification.json!)."""
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
    handoff_path = prev_session_dir / "HANDOFF.md"
    prev_id = prev_session_dir.name
    prev_meta = lineage.get("sessions", {}).get(prev_id, {})
    was_promoted = bool(prev_meta.get("promoted"))

    lines.append(f"- **Previous session (`{prev_id}`):** promoted = `{was_promoted}`")
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

        if handoff_path.exists():
            handoff_text = handoff_path.read_text()
            missing = validate_handoff(handoff_text)
            if missing:
                lines.append(f"- **Handoff schema note:** missing sections `{missing}`")
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
            trimmed = handoff_text.strip()
            if len(trimmed) > 12000:
                trimmed = trimmed[:12000] + "\n\n... [truncated]"
            lines.extend(["", "### Previous HANDOFF.md", "", trimmed])
    else:
        lines.append("- **Verifier-verified V score:** missing `reward.json`/`v_report.json` in previous session.")

    return "\n".join(lines) + "\n"
