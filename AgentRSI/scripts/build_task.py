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
        "--ref",
        action="append",
        default=[],
        dest="ref_sessions",
        help="Optional reference session ID(s) whose HANDOFF.md, REPORT.md, and submission/ to stage under /app/prior/ref_sessions/ (repeatable)",
    )
    ap.add_argument(
        "--datasets-from",
        default=None,
        help="Optional session ID whose datasets/ directory to carry into /app/datasets/",
    )
    ap.add_argument(
        "--user-prompt",
        type=Path,
        default=None,
        help="Optional Markdown file with researcher guidance injected into instruction.md and /app/prior/user_prompt.md (must remain strictly T-free)",
    )
    ap.add_argument("--timeout", type=int, default=None, help="Override agent timeout_sec")
    ap.add_argument("--agent-cpus", type=int, default=None, help="Override agent container CPUs")
    ap.add_argument("--agent-memory-mb", type=int, default=None, help="Override agent container memory (MB)")
    ap.add_argument("--verifier-cpus", type=int, default=None, help="Override verifier container CPUs")
    ap.add_argument("--verifier-memory-mb", type=int, default=None, help="Override verifier container memory (MB)")
    inrun_grp = ap.add_mutually_exclusive_group()
    inrun_grp.add_argument(
        "--enable-inrun-verifier",
        dest="enable_inrun_verifier",
        action="store_true",
        default=None,
        help="Enable M7 in-run verifier v2 sidecar MCP server",
    )
    inrun_grp.add_argument(
        "--disable-inrun-verifier",
        dest="enable_inrun_verifier",
        action="store_false",
        help="Disable M7 in-run verifier v2 sidecar MCP server",
    )
    colab_grp = ap.add_mutually_exclusive_group()
    colab_grp.add_argument(
        "--enable-colab",
        dest="enable_colab",
        action="store_true",
        default=None,
        help="Enable M8 Colab GPU offload MCP broker",
    )
    colab_grp.add_argument(
        "--disable-colab",
        dest="enable_colab",
        action="store_false",
        help="Disable M8 Colab GPU offload MCP broker",
    )
    ap.add_argument(
        "--colab-allow-a100",
        dest="colab_allow_a100",
        action="store_true",
        default=None,
        help="Allow A100 GPU selection in M8 Colab broker",
    )
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
        ref_sessions=args.ref_sessions,
        datasets_from=args.datasets_from,
        user_prompt=args.user_prompt,
        timeout_override=args.timeout,
        agent_cpus_override=args.agent_cpus,
        agent_memory_mb_override=args.agent_memory_mb,
        verifier_cpus_override=args.verifier_cpus,
        verifier_memory_mb_override=args.verifier_memory_mb,
        enable_inrun_verifier=args.enable_inrun_verifier,
        enable_colab=args.enable_colab,
        colab_allow_a100=args.colab_allow_a100,
        output_root=args.output_root,
    )
    print(f"Built task: {task_dir}")


if __name__ == "__main__":
    main()
