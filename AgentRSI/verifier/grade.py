"""CLI entry point invoked by the Harbor verifier container (`tests/test.sh`)."""
from __future__ import annotations

import argparse
from pathlib import Path

from verifier.runner import grade_submission


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade an AgentRSI submission in isolation.")
    ap.add_argument("--sealed-dir", type=Path, default=Path("/sealed"))
    ap.add_argument("--submission-dir", type=Path, default=Path("/app/submission"))
    ap.add_argument("--datasets-dir", type=Path, default=Path("/app/datasets"))
    ap.add_argument("--incumbent-dir", type=Path, default=None)
    ap.add_argument("--incumbent-datasets-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("/logs/verifier"))
    ap.add_argument("--run-user", type=str, default="runner")
    args = ap.parse_args()

    reward, v_report, cert = grade_submission(
        sealed_dir=args.sealed_dir,
        submission_dir=args.submission_dir,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        incumbent_dir=args.incumbent_dir,
        incumbent_datasets_dir=args.incumbent_datasets_dir,
        run_user=args.run_user,
    )
    print(
        f"graded status={v_report['status']} valid={reward['submission_valid']} "
        f"v_score={reward['score']:.6f} v_dInc={reward['v_delta_vs_incumbent']:+.6f} "
        f"gate_passed={cert['gate_passed']} runtime={reward['runtime_sec']}s"
    )


if __name__ == "__main__":
    main()
