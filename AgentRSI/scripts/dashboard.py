"""CLI wrapper to generate and optionally verify AgentRSI/dashboard/index.html."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.dashboard import generate_dashboard, verify_dashboard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate AgentRSI self-contained HTML dashboard.")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=AGENT_ROOT / "dashboard" / "index.html",
        help="Output HTML path (default: AgentRSI/dashboard/index.html)",
    )
    ap.add_argument(
        "--sessions-dir",
        type=Path,
        default=AGENT_ROOT / "sessions",
        help="Sessions directory (default: AgentRSI/sessions)",
    )
    ap.add_argument(
        "--lineage",
        type=Path,
        default=AGENT_ROOT / "state" / "lineage.json",
        help="Path to lineage.json",
    )
    ap.add_argument(
        "--events",
        type=Path,
        default=AGENT_ROOT / "state" / "events.jsonl",
        help="Path to events.jsonl",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Diff generated HTML numbers, charts, and events against source files.",
    )
    args = ap.parse_args()

    out = generate_dashboard(
        AGENT_ROOT,
        output_path=args.output,
        sessions_root=args.sessions_dir,
        lineage_path=args.lineage,
        events_path=args.events,
    )
    print(f"Wrote dashboard: {out}")

    if args.verify:
        errors = verify_dashboard(
            out,
            sessions_root=args.sessions_dir,
            lineage_path=args.lineage,
            events_path=args.events,
        )
        if errors:
            for err in errors:
                print(f"VERIFY ERROR: {err}", file=sys.stderr)
            sys.exit(1)
        print("Verified dashboard: 0 mismatches")


if __name__ == "__main__":
    main()
