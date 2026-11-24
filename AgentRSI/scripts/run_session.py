"""CLI wrapper to stage secrets, check budget guards, and run a Harbor session."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = AGENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libbuilder.run_session import run_session  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an AgentRSI session via Harbor.")
    ap.add_argument("--session-id", required=True, help="Task/session ID under AgentRSI/tasks/")
    ap.add_argument("--agent", default=None, help="Override agent (codex | claude-code)")
    ap.add_argument("--model", default=None, help="Override model name")
    ap.add_argument("--max-turns", type=int, default=None, help="Override max_turns (claude-code)")
    ap.add_argument("--max-budget-usd", type=float, default=None, help="Override max_budget_usd (claude-code)")
    ap.add_argument("--dry-run", action="store_true", help="Pass --dry-run to harbor run")
    ap.add_argument("--colab-mock", action="store_true", help="Use MockColabBackend for M8 Colab broker")
    ap.add_argument("--colab-allow-a100", action="store_true", help="Allow A100 GPU selection in M8 Colab broker")
    args = ap.parse_args()

    rc = run_session(
        args.session_id,
        agent=args.agent,
        model=args.model,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        dry_run=args.dry_run,
        colab_mock=args.colab_mock,
        colab_allow_a100=args.colab_allow_a100,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
