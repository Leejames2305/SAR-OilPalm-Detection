#!/usr/bin/env python3
"""Helper CLI inside the AgentRSI agent container (`agentrsi selfcheck|cv|gcp-init`)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as sk_ap
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/app/data")
DATASETS_DIR = Path("/app/datasets")
BLOCK_M = 80
PINNED_REQ_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9.,_-]+\])?==[^\s;#]+(\s*;.*)?$"
)


def _block_keys(lon: np.ndarray, lat: np.ndarray, bs: int = BLOCK_M) -> np.ndarray:
    mx = (lon - lon.min()) * 111320 * np.cos(np.radians(lat.mean()))
    my = (lat - lat.min()) * 110540
    return np.floor(mx / bs).astype(np.int64) * 100000 + np.floor(my / bs).astype(np.int64)


def _validate_requirements(req_path: Path) -> list[str]:
    if not req_path.exists():
        return []
    lines = []
    for raw in req_path.read_text().splitlines():
        cleaned = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if cleaned.startswith("-") or cleaned.startswith("/") or cleaned.startswith("."):
            raise ValueError(f"Forbidden requirements.txt option or path: {cleaned!r}")
        if any(frag in lower for frag in ("file:", "://", "/sealed", "..", "\\")):
            raise ValueError(f"Forbidden local/URL reference in requirements.txt: {cleaned!r}")
        if not PINNED_REQ_RE.match(cleaned):
            raise ValueError(f"Requirements must use exact 'package==version' pins (got {cleaned!r})")
        lines.append(cleaned)
    return lines


def _load_pipeline(submission_dir: Path):
    if not (submission_dir / "pipeline.py").exists():
        raise FileNotFoundError(f"Missing {submission_dir / 'pipeline.py'}")
    sys.path.insert(0, str(submission_dir))
    import pipeline  # type: ignore[import-not-found]

    return pipeline


def cmd_selfcheck(args: argparse.Namespace) -> None:
    sub = args.submission
    reqs = _validate_requirements(sub / "requirements.txt")
    print(f"requirements.txt: {len(reqs)} pinned package(s)")
    pipeline = _load_pipeline(sub)

    l_all = pd.read_csv(DATA_DIR / "L_labelled.csv")
    u_all = pd.read_csv(DATA_DIR / "unlabelled.csv")
    estates = sorted(l_all["estate"].unique())
    cpus = int(os.environ.get("AGENTRSI_CPUS", "6"))

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        for e in estates:
            le = l_all[l_all["estate"] == e].reset_index(drop=True)
            ue = u_all[u_all["estate"] == e].reset_index(drop=True)
            other = {o: l_all[l_all["estate"] == o].reset_index(drop=True) for o in estates if o != e}
            scratch = Path(td) / e
            scratch.mkdir()
            ctx = {
                "estate": e,
                "datasets_dir": DATASETS_DIR,
                "other_labeled": other,
                "scratch_dir": scratch,
                "cpus": cpus,
            }
            te0 = time.monotonic()
            preds = pipeline.fit_predict(le, ue, ctx)
            arr = np.asarray(preds, dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != len(ue):
                raise SystemExit(f"FAIL {e}: returned shape {arr.shape}, expected ({len(ue)},)")
            if not np.isfinite(arr).all():
                raise SystemExit(f"FAIL {e}: returned NaN or Inf values")
            dt = time.monotonic() - te0
            print(
                f"  {e:9s}: n_L={len(le)} n_U={len(ue)} dt={dt:.2f}s "
                f"min={arr.min():.4f} med={np.median(arr):.4f} max={arr.max():.4f}"
            )
    print(f"SELFCHECK PASS ({time.monotonic() - t0:.2f}s)")


def _ndvi_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    tr = train_df[train_df["label"].isin(["Healthy", "Unhealthy"])]
    y = (tr["label"] == "Unhealthy").astype(int).to_numpy()
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))
    m.fit(tr[["NDVI"]].to_numpy(float), y)
    return m.decision_function(val_df[["NDVI"]].to_numpy(float))


def cmd_cv(args: argparse.Namespace) -> None:
    sub = args.submission
    pipeline = _load_pipeline(sub)
    l_all = pd.read_csv(DATA_DIR / "L_labelled.csv")
    estates = sorted(l_all["estate"].unique())
    cpus = int(os.environ.get("AGENTRSI_CPUS", "6"))

    cand_lifts = []
    base_lifts = []
    print(f"Spatial 80 m 5-fold CV on L_labelled.csv ({len(l_all)} rows):")
    with tempfile.TemporaryDirectory() as td:
        for e in estates:
            le = l_all[l_all["estate"] == e].reset_index(drop=True)
            other = {o: l_all[l_all["estate"] == o].reset_index(drop=True) for o in estates if o != e}
            groups = _block_keys(le["Long"].to_numpy(float), le["Lat"].to_numpy(float))
            oof_cand = np.full(len(le), np.nan)
            oof_base = np.full(len(le), np.nan)
            for fold, (tr_idx, va_idx) in enumerate(GroupKFold(5).split(le, groups=groups)):
                tr_df = le.iloc[tr_idx].reset_index(drop=True)
                va_all = le.iloc[va_idx].reset_index(drop=True)
                va_unlabeled = va_all.drop(columns=["label"])
                scratch = Path(td) / f"{e}_{fold}"
                scratch.mkdir()
                ctx = {
                    "estate": e,
                    "datasets_dir": DATASETS_DIR,
                    "other_labeled": other,
                    "scratch_dir": scratch,
                    "cpus": cpus,
                }
                preds = np.asarray(pipeline.fit_predict(tr_df, va_unlabeled, ctx), dtype=np.float64)
                if preds.shape != (len(va_unlabeled),) or not np.isfinite(preds).all():
                    raise SystemExit(f"FAIL {e} fold {fold}: invalid predictions")
                oof_cand[va_idx] = preds
                oof_base[va_idx] = _ndvi_baseline(tr_df, va_unlabeled)

            scored = le["label"].isin(["Healthy", "Unhealthy"]).to_numpy()
            y_sc = (le.loc[scored, "label"] == "Unhealthy").astype(int).to_numpy()
            prev = float(y_sc.mean())
            pr_c = float(sk_ap(y_sc, oof_cand[scored]))
            pr_b = float(sk_ap(y_sc, oof_base[scored]))
            lift_c, lift_b = pr_c - prev, pr_b - prev
            cand_lifts.append(lift_c)
            base_lifts.append(lift_b)
            print(
                f"  {e:9s}: prev={prev:.4f} | cand PR={pr_c:.4f} (lift={lift_c:+.4f}) "
                f"| NDVI base PR={pr_b:.4f} (lift={lift_b:+.4f}) | dPR={lift_c - lift_b:+.4f}"
            )
    mc, mb = float(np.mean(cand_lifts)), float(np.mean(base_lifts))
    print(f"Composite L-CV mean lift: cand={mc:+.4f} vs NDVI={mb:+.4f} (delta={mc - mb:+.4f})")


def cmd_gcp_init(_: argparse.Namespace) -> None:
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    project = os.environ.get("GCP_PROJECT", "")
    if not cred_path or not Path(cred_path).exists():
        raise SystemExit(f"GOOGLE_APPLICATION_CREDENTIALS missing or unreadable: {cred_path!r}")
    if not project:
        raise SystemExit("GCP_PROJECT environment variable is empty")

    info = json.loads(Path(cred_path).read_text())
    client_email = info["client_email"]
    import ee

    creds = ee.ServiceAccountCredentials(client_email, cred_path)
    ee.Initialize(creds, project=project)
    ok = ee.Number(1).add(1).getInfo()
    print(f"GEE init OK (project={project}, sa={client_email}, probe={ok})")


def main() -> None:
    ap = argparse.ArgumentParser(prog="agentrsi", description="In-container helpers for AgentRSI.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sc = sub.add_parser("selfcheck", help="Validate requirements.txt and run fit_predict offline on all estates")
    p_sc.add_argument("--submission", type=Path, default=Path("/app/submission"))
    p_cv = sub.add_parser("cv", help="Run 5-fold 80 m spatial-block CV on L_labelled.csv")
    p_cv.add_argument("--submission", type=Path, default=Path("/app/submission"))
    sub.add_parser("gcp-init", help="Verify Earth Engine initialization using mounted SA key")
    args = ap.parse_args()
    if args.cmd == "selfcheck":
        cmd_selfcheck(args)
    elif args.cmd == "cv":
        cmd_cv(args)
    else:
        cmd_gcp_init(args)


if __name__ == "__main__":
    main()
