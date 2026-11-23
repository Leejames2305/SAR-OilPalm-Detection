"""Lineage, incumbent tracking, T-attempt counter, and daily spend ledger."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Verified M2 starter incumbent V scores (never headline T scores here)
STARTER_V_METRICS = {
    "score": 0.05446945305942965,
    "per_estate_v_lift": {
        "AirHitam": 0.046744654198282894,
        "Palong": 0.04264048377897063,
        "Serting": 0.07402322120103541,
    },
}


def default_lineage(split_cfg: dict[str, Any], verifier_cfg: dict[str, Any]) -> dict[str, Any]:
    """Initialize epoch-1 lineage with the Session-0 starter incumbent (V metrics only)."""
    return {
        "schema_version": 1,
        "epoch": int(split_cfg["epoch"]),
        "split_sha256": str(split_cfg["sha256_of_assignment"]),
        "t_attempts_used": 0,
        "t_attempt_budget": int(verifier_cfg["gate"]["t_attempt_budget"]),
        "kill_switch": False,
        "daily_spend_usd": {},
        "incumbent": {
            "id": "starter",
            "source_dir": "starter",
            "epoch": int(split_cfg["epoch"]),
            "v_score": STARTER_V_METRICS["score"],
            "per_estate_v_lift": dict(STARTER_V_METRICS["per_estate_v_lift"]),
        },
        "epochs": [
            {
                "epoch": int(split_cfg["epoch"]),
                "split_sha256": str(split_cfg["sha256_of_assignment"]),
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ],
        "sessions": {},
    }


def load_lineage(lineage_path: Path, split_cfg: dict[str, Any], verifier_cfg: dict[str, Any]) -> dict[str, Any]:
    if not lineage_path.exists():
        return default_lineage(split_cfg, verifier_cfg)
    return json.loads(lineage_path.read_text())


def save_lineage(lineage_path: Path, data: dict[str, Any]) -> None:
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(json.dumps(data, indent=2) + "\n")


def check_budget_guard(
    lineage: dict[str, Any],
    daily_cap_usd: float,
    day_key: str | None = None,
) -> tuple[bool, str | None]:
    """Return (allowed, reason) against kill_switch, T-attempt budget, and daily spend cap."""
    if lineage.get("kill_switch"):
        return False, "kill_switch is active in lineage.json"
    used = int(lineage.get("t_attempts_used", 0))
    budget = int(lineage.get("t_attempt_budget", 20))
    if used >= budget:
        return False, f"T-attempt budget exhausted ({used}/{budget})"
    day = day_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    spent = float(lineage.get("daily_spend_usd", {}).get(day, 0.0))
    if spent >= daily_cap_usd:
        return False, f"Daily spend cap reached for {day}: ${spent:.4f} >= ${daily_cap_usd:.2f}"
    return True, None


def record_session_result(
    lineage: dict[str, Any],
    *,
    session_id: str,
    job_id: str,
    parent_id: str,
    meta: dict[str, Any],
    reward: dict[str, Any],
    v_report: dict[str, Any],
    certification: dict[str, Any],
    submission_rel_dir: str,
    dry_session: bool = False,
) -> dict[str, Any]:
    """Update lineage idempotently for `(session_id, job_id)` and promote on gate pass."""
    sessions = lineage.setdefault("sessions", {})
    existing = sessions.get(session_id)
    already_recorded = existing is not None and existing.get("job_id") == job_id

    cost_usd = float(meta.get("cost_usd") or 0.0)
    day = str(meta.get("started_at") or datetime.now(timezone.utc).isoformat())[:10]
    counts_toward_t = (not dry_session) and bool(v_report.get("submission_valid"))

    if not already_recorded:
        spend_map = lineage.setdefault("daily_spend_usd", {})
        spend_map[day] = round(float(spend_map.get(day, 0.0)) + cost_usd, 6)
        if counts_toward_t:
            lineage["t_attempts_used"] = int(lineage.get("t_attempts_used", 0)) + 1

    gate_passed = bool(certification.get("gate_passed")) and not dry_session
    promoted = False
    if gate_passed and not already_recorded:
        lineage["incumbent"] = {
            "id": session_id,
            "source_dir": submission_rel_dir,
            "epoch": int(lineage["epoch"]),
            "v_score": float(reward["score"]),
            "per_estate_v_lift": {
                e: float(reward[f"v_lift_{e}"])
                for e in ("AirHitam", "Palong", "Serting")
                if f"v_lift_{e}" in reward
            },
        }
        promoted = True
    elif existing is not None and already_recorded:
        promoted = bool(existing.get("promoted"))

    sessions[session_id] = {
        "session_id": session_id,
        "job_id": job_id,
        "parent_id": parent_id,
        "epoch": int(lineage["epoch"]),
        "dry_session": bool(dry_session),
        "status": str(v_report.get("status", "unknown")),
        "submission_valid": bool(v_report.get("submission_valid")),
        "v_score": float(reward.get("score", 0.0)),
        "v_delta_vs_incumbent": float(reward.get("v_delta_vs_incumbent", 0.0)),
        "gate_passed": bool(certification.get("gate_passed")),
        "t_comp_delta": float(certification.get("composite", {}).get("comp_delta", 0.0)),
        "t_ci_low": float(certification.get("composite", {}).get("ci_low", 0.0)),
        "promoted": promoted,
        "cost_usd": cost_usd,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {
        "already_recorded": already_recorded,
        "promoted": promoted,
        "t_attempts_used": int(lineage["t_attempts_used"]),
        "day": day,
        "cost_usd": cost_usd,
    }
