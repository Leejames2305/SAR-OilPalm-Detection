"""CLI wrapper for host-only session collection (`sessions/<session_id>/`, lineage, events)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.collect import collect_session  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect and record an AgentRSI session.")
    ap.add_argument("--session-id", required=True, help="Session ID (e.g. s000-dry, s001)")
    ap.add_argument("--job-dir", type=Path, default=None, help="Explicit Harbor timestamped job dir")
    ap.add_argument("--parent", default="starter", help="Parent session ID or 'starter'")
    ap.add_argument("--dry-session", action="store_true", help="Do not spend T attempts or promote")
    args = ap.parse_args()

    res = collect_session(
        args.session_id,
        job_dir=args.job_dir,
        parent_id=args.parent,
        dry_session=args.dry_session,
    )
    print(json.dumps(res["lineage_update"], indent=2))


if __name__ == "__main__":
    main()
