"""Append-only, idempotent JSONL event ledger (`AgentRSI/state/events.jsonl`)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_TYPES = frozenset({
    "session_start",
    "session_end",
    "nominee_declared",
    "certification_pass",
    "certification_fail",
    "promoted",
    "review_flag",
    "split_refresh",
    "budget_alert",
    "spend_cap",
    "run_error",
})


def load_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in events_path.read_text().splitlines():
        s = line.strip()
        if s:
            out.append(json.loads(s))
    return out


def append_event(
    events_path: Path,
    *,
    session: str,
    epoch: int,
    event_type: str,
    summary: str,
    ref: str = "",
    values: dict[str, Any] | None = None,
    dedup_id: str | None = None,
) -> bool:
    """Append an event if an identical event has not already been recorded. Returns True if written."""
    if event_type not in VALID_TYPES:
        raise ValueError(f"Unknown event type {event_type!r}")

    existing = load_events(events_path)
    for ev in existing:
        if ev.get("session") != session or ev.get("type") != event_type:
            continue
        if dedup_id is not None:
            if ev.get("dedup_id") == dedup_id:
                return False
        elif ev.get("summary") == summary:
            return False

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session,
        "epoch": int(epoch),
        "type": event_type,
        "summary": summary,
        "ref": ref,
    }
    if values is not None:
        record["values"] = values
    if dedup_id is not None:
        record["dedup_id"] = dedup_id

    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    return True
