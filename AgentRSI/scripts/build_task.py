"""CLI wrapper to build an AgentRSI Harbor task (`tasks/<session_id>/`)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.build_task import build_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an AgentRSI Harbor task directory.")
    ap.add_argument("--session-id", required=True, help="Session ID (e.g. s000-dry, s001)")
    ap.add_argument("--focus", default=None, help="Focus lens name under templates/lenses/")
    ap.add_argument("--parent", default=None, help="Previous session ID whose HANDOFF.md to inject")
    ap.add_argument("--timeout", type=int, default=None, help="Override agent timeout_sec")
    ap.add_argument("--output-root", type=Path, default=None, help="Override tasks output root")
    args = ap.parse_args()

    task_dir = build_task(
        args.session_id,
        focus_lens=args.focus,
        parent_session=args.parent,
        timeout_override=args.timeout,
        output_root=args.output_root,
    )
    print(f"Built task: {task_dir}")


if __name__ == "__main__":
    main()
