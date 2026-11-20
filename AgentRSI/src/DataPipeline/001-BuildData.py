"""Step 1: build the sealed master, the frozen split, the agent view and the verifier staging.

  master    .sealed/labels_master.csv from data/Labels (rewritten each run)
  features  .work/features.csv is frozen. --extract rebuilds it from Earth Engine and the SAR scenes
  split     config/split_assignment.csv: whole 80 m blocks to L/V/T, no labels. Writes nothing unless the sha256 is frozen
  view      .work/agent_view/*.csv: allowlisted columns, rows identified by estate and coordinates (labels only on L)
  verifier  .sealed/verifier_context/: master copy for the M2 image build

Nothing under .sealed/ may be copied into tasks/ or jobs/. Run 002-CheckLeaks.py after this.

Usage: python 001-BuildData.py [--extract]
"""
import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from common import (AGENT_VIEW, BLOCK_M, CLASSES, ESTATES, FEATURE_COLS, FEATURES, KEY_FILE, LABEL_DIR,
                    LABEL_FILES, MASTER, SEALED, SEALED_REPORT, SPLIT, SPLIT_FRAC, SPLIT_MIN, SPLIT_SEED,
                    SPLIT_SHA256, SPLIT_TOL, VERIFIER_CONTEXT, block_keys)
from features import build_feature_table


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_master():
    SEALED.mkdir(mode=0o700, exist_ok=True)
    os.chmod(SEALED, 0o700)

    frames = []
    for e in ESTATES:
        raw = pd.read_csv(LABEL_DIR / LABEL_FILES[e])
        raw.insert(0, "estate", e)
        frames.append(raw.rename(columns={"Class": "class"})[["estate", "id", "Long", "Lat", "class"]])
    master = pd.concat(frames, ignore_index=True)

    if master.duplicated(["estate", "id"]).any():
        raise SystemExit("duplicate (estate, id) in the raw labels")
    unknown = set(master["class"]) - set(CLASSES)
    if unknown:
        raise SystemExit(f"unexpected class values: {unknown}")

    master.to_csv(MASTER, index=False)
    os.chmod(MASTER, 0o600)

    print(master.groupby(["estate", "class"]).size().unstack(fill_value=0).to_string())
    print("master rows", len(master), "->", MASTER)


def build_features(extract):
    if extract:
        with open(KEY_FILE) as fh:
            project = json.load(fh)["project_id"]
        t0 = time.time()
        df = build_feature_table(project)
        FEATURES.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(FEATURES, index=False)
        print("features rebuilt:", len(df), "rows,", len(df.columns), "cols,", f"{time.time() - t0:.0f}s ->", FEATURES)
        print("missing values:", int(df.isna().sum().sum()))
    elif not FEATURES.exists():
        raise SystemExit(f"{FEATURES} is missing: run with --extract (calls Earth Engine)")
    else:
        print("features: frozen table", FEATURES)


def build_split():
    master = pd.read_csv(MASTER)
    rng = np.random.default_rng(SPLIT_SEED)
    parts_out, report = [], {}

    for e in ESTATES:
        sub = master[master.estate == e].reset_index(drop=True).copy()
        keys = block_keys(sub.Long.to_numpy(float), sub.Lat.to_numpy(float), BLOCK_M)
        ub, inv = np.unique(keys, return_inverse=True)
        pos = np.bincount(inv, weights=(sub["class"] == "Unhealthy").to_numpy(float), minlength=len(ub))
        nt = np.bincount(inv, minlength=len(ub)).astype(float)
        draws = 0
        while True:
            draws += 1
            u = rng.random(len(ub))
            part = np.where(u < SPLIT_FRAC["L"], 0, np.where(u < SPLIT_FRAC["L"] + SPLIT_FRAC["V"], 1, 2))
            pL, pV, pT = (pos[part == i].sum() for i in (0, 1, 2))
            # tree fractions per part, counting every tree (Middle included) in its block
            tL, tV, tT = (nt[part == i].sum() / nt.sum() for i in (0, 1, 2))
            ok_pos = pL >= SPLIT_MIN["L"] and pV >= SPLIT_MIN["V"] and pT >= SPLIT_MIN["T"]
            ok_frac = all(abs(f - SPLIT_FRAC[k]) <= SPLIT_TOL for f, k in ((tL, "L"), (tV, "V"), (tT, "T")))
            if ok_pos and ok_frac:
                break

        blk_part = dict(zip(ub, part))
        sub["block"] = keys
        sub["part"] = [("L", "V", "T")[blk_part[k]] for k in keys]
        parts_out.append(sub[["estate", "id", "block", "part"]])

        per = {}
        for p, g in sub.groupby("part"):
            per[p] = {"trees": int(len(g)), **{c: int((g["class"] == c).sum()) for c in CLASSES}}
        report[e] = {"blocks": int(len(ub)), "redraws": draws, "counts": per}
        print(e, "blocks", len(ub), "redraws", draws, json.dumps(per))

    df = pd.concat(parts_out).sort_values(["estate", "id"])
    digest = hashlib.sha256(df[["estate", "id", "part"]].to_csv(index=False).encode()).hexdigest()
    print("sha256", digest, "| frozen", SPLIT_SHA256)
    if digest != SPLIT_SHA256:
        raise SystemExit("split does not reproduce the frozen sha256; nothing written")

    df.to_csv(SPLIT, index=False)
    SEALED_REPORT.write_text(json.dumps({"block_m": BLOCK_M, "seed": SPLIT_SEED, "sha256": digest,
                                         "report": report}, indent=2))
    os.chmod(SEALED_REPORT, 0o600)
    print("wrote", SPLIT, "and", SEALED_REPORT)


def build_agent_view():
    feat = pd.read_csv(FEATURES)
    split = pd.read_csv(SPLIT)
    master = pd.read_csv(MASTER, dtype={"Long": str, "Lat": str})  # keep coordinate text exactly as in the master

    view = feat.merge(split[["estate", "id", "part"]], on=["estate", "id"], how="inner", validate="one_to_one")
    view = view.merge(master, on=["estate", "id"], how="left", validate="one_to_one")
    if len(view) != len(feat) or view["class"].isna().any():
        raise SystemExit("features, split and master do not line up")

    if view.duplicated(["estate", "Long", "Lat"]).any():
        raise SystemExit("two trees share estate and coordinates")
    view["label"] = view["class"].where(view["part"] == "L")

    AGENT_VIEW.mkdir(parents=True, exist_ok=True)
    cols = ["estate", "Long", "Lat", *FEATURE_COLS]
    for part, name in (("L", "L_labelled.csv"), ("V", "V_unlabelled.csv"), ("T", "T_unlabelled.csv")):
        sub = view[view["part"] == part].sort_values(["estate", "Long", "Lat"])
        keep = cols + (["label"] if part == "L" else [])
        sub[keep].to_csv(AGENT_VIEW / name, index=False)
        print(name, len(sub), "rows")
    print("wrote", AGENT_VIEW)


def stage_verifier():
    assert VERIFIER_CONTEXT.resolve().is_relative_to(SEALED.resolve()), "verifier context must stay inside .sealed/"
    VERIFIER_CONTEXT.mkdir(parents=True, exist_ok=True)
    os.chmod(VERIFIER_CONTEXT, 0o700)
    dst = VERIFIER_CONTEXT / MASTER.name
    shutil.copyfile(MASTER, dst)
    os.chmod(dst, 0o600)

    manifest = {
        "labels_master_sha256": sha256(MASTER),
        "labels_master_rows": int(len(pd.read_csv(MASTER))),
        "files": ["labels_master.csv"],
    }
    manifest_path = VERIFIER_CONTEXT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    os.chmod(manifest_path, 0o600)
    print("staged", ", ".join(manifest["files"]), "+ manifest.json in", VERIFIER_CONTEXT)
    print("labels_master sha256", manifest["labels_master_sha256"], "rows", manifest["labels_master_rows"])


def main():
    ap = argparse.ArgumentParser(description="Build the sealed master, split, agent view and verifier staging.")
    ap.add_argument("--extract", action="store_true",
                    help="rebuild .work/features.csv from Earth Engine and SAR scenes (replaces the frozen table)")
    args = ap.parse_args()

    build_master()
    build_features(args.extract)
    build_split()
    build_agent_view()
    stage_verifier()


if __name__ == "__main__":
    main()
