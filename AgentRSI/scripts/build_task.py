"""CLI wrapper to build an AgentRSI Harbor task (`tasks/<session_id>/`)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.build_task import build_task, make_session_id, next_session_seq  # noqa: E402
from libbuilder.lineage import load_lineage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an AgentRSI Harbor task directory.")
    ap.add_argument("--session-id", default=None, help="Session ID (e.g. 261123-s001-open)")
    ap.add_argument("--next", action="store_true", help="Auto-generate next YYMMDD-sXXX-<lens> session ID")
    ap.add_argument("--date", default=None, help="Optional YYMMDD date override when using --next")
    ap.add_argument("--focus", default=None, help="Focus lens name under templates/lenses/")
    ap.add_argument("--parent", default=None, help="Previous session ID whose HANDOFF.md to inject")
    ap.add_argument(
        "--datasets-from",
        default=None,
        help="Optional session ID whose datasets/ directory to carry into /app/datasets/",
    )
    ap.add_argument("--timeout", type=int, default=None, help="Override agent timeout_sec")
    ap.add_argument("--output-root", type=Path, default=None, help="Override tasks output root")
    args = ap.parse_args()

    if args.session_id and args.next:
        ap.error("Specify either --session-id or --next, not both")
    if not args.session_id and not args.next:
        ap.error("Provide --session-id (e.g. 261123-s001-open) or --next")

    session_id = args.session_id
    if args.next:
        cfg_dir = AGENT_ROOT / "config"
        session_cfg = yaml.safe_load((cfg_dir / "session.yaml").read_text())
        verifier_cfg = yaml.safe_load((cfg_dir / "verifier.yaml").read_text())
        split_cfg = yaml.safe_load((cfg_dir / "split.yaml").read_text())
        lineage = load_lineage(AGENT_ROOT / "state" / "lineage.json", split_cfg, verifier_cfg)
        lens = args.focus or str(session_cfg.get("focus_lens", "open"))
        seq = next_session_seq(lineage, AGENT_ROOT / "sessions")
        session_id = make_session_id(seq, lens, date_str=args.date)

    task_dir = build_task(
        session_id,
        focus_lens=args.focus,
        parent_session=args.parent,
        datasets_from=args.datasets_from,
        timeout_override=args.timeout,
        output_root=args.output_root,
    )
    print(f"Built task: {task_dir}")


if __name__ == "__main__":
    main()
