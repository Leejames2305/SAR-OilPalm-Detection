"""Unprivileged per-estate submission worker invoked by runner.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute submission.pipeline.fit_predict on one estate.")
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--labeled", type=Path, required=True)
    ap.add_argument("--unlabeled", type=Path, required=True)
    ap.add_argument("--other-labeled-dir", type=Path, required=True)
    ap.add_argument("--datasets-dir", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--estate", type=str, required=True)
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labeled = pd.read_csv(args.labeled)
    unlabeled = pd.read_csv(args.unlabeled)
    other_labeled = {
        p.stem: pd.read_csv(p)
        for p in sorted(args.other_labeled_dir.glob("*.csv"))
    }
    ctx = {
        "estate": args.estate,
        "datasets_dir": args.datasets_dir,
        "other_labeled": other_labeled,
        "scratch_dir": args.scratch_dir,
        "cpus": int(args.cpus),
    }

    sys.path.insert(0, str(args.submission))
    import pipeline  # type: ignore[import-not-found]

    preds = pipeline.fit_predict(labeled, unlabeled, ctx)
    arr = np.asarray(preds, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != len(unlabeled):
        raise ValueError(
            f"{args.estate}: fit_predict returned shape {arr.shape}, expected ({len(unlabeled)},)"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{args.estate}: fit_predict returned NaN or Inf scores")

    np.save(args.out, arr)


if __name__ == "__main__":
    main()
