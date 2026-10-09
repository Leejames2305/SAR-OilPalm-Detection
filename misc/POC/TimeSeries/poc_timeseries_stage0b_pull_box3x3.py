"""Stage 0b of POC time-series (AirHitam only): 3x3 box re-pull for S1 and S2.

Why: stage 0 sampled the single 10 m pixel at each tree. S1 speckle at one
pixel is noisy, so this pull averages a 30 m box around each tree (about 3x3
pixels at 10 m). Window is extended to 15 July 2026 so it matches the stage-2
NDVI composite window (15 May - 15 Jul). The 12-month analysis is cut at the
survey date inside stage 1/2, not here.

Reuses helpers from poc_timeseries_stage0_pull.py.

Outputs (misc/POC_Results/TimeSeries/):
  ts_s1_box3x3.csv, ts_s2_box3x3.csv   long format: id, Class, date, band[, orbit]
  stage0b_summary.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
import ee

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import poc_timeseries_stage0_pull as s0  # noqa: E402

OUT = s0.OUT
PULL_START, PULL_END = "2025-06-10", "2026-07-16"  # end is exclusive in GEE
BOX_HALF_M = 15
TREE_CHUNK = 500


def box_fc(labels):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.Long), float(r.Lat)])
                   .buffer(BOX_HALF_M).bounds(),
                   {"id": int(r.id), "Class": str(r.Class)})
        for r in labels.itertuples()])


def sample_box(stack, labels, scale):
    frames = []
    n_chunks = -(-len(labels) // TREE_CHUNK)
    for i in range(0, len(labels), TREE_CHUNK):
        fc = box_fc(labels.iloc[i:i + TREE_CHUNK])
        s = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                scale=scale)
        feats = s.getInfo()["features"]
        frames.append(pd.DataFrame([f["properties"] for f in feats]))
        print(f"    chunk {i // TREE_CHUNK + 1}/{n_chunks}")
    return pd.concat(frames, ignore_index=True)


def main():
    account = s0.init()
    print("EE init OK as", account)
    labels = s0.load_labels()
    print("trees:", len(labels))
    summary = {"window": [PULL_START, PULL_END], "box_half_m": BOX_HALF_M,
               "account": account, "n_trees": len(labels)}

    print("[S2] NDVI/NDRE box")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(ee.Geometry.Point([float(labels.Long.mean()),
                                           float(labels.Lat.mean())]).buffer(3000))
          .filterDate(PULL_START, PULL_END))
    summary["s2_scenes"] = s2.size().getInfo()
    s2_stack = s0.date_prefixed_stack(s2.map(s0.s2_masked_indices),
                                      ["NDVI", "NDRE"])
    s2_wide = sample_box(s2_stack, labels, 10)
    s0.wide_to_long(s2_wide, ["NDVI", "NDRE"]).to_csv(
        OUT / "ts_s2_box3x3.csv", index=False)
    print("  S2 scenes:", summary["s2_scenes"])

    print("[S1] VV/VH by orbit, box")
    s1_base = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(ee.Geometry.Point([float(labels.Long.mean()),
                                                float(labels.Lat.mean())]).buffer(3000))
               .filterDate(PULL_START, PULL_END)
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains(
                   "transmitterReceiverPolarisation", "VH")))
    frames = []
    for orbit_no, tag in s0.S1_ORBITS.items():
        col = s1_base.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit_no))
        n = col.size().getInfo()
        summary[f"s1_{tag}_scenes"] = n
        print(f"  orbit {orbit_no} ({tag}): {n} scenes")
        stack = s0.date_prefixed_stack(col.map(s0.s1_bands),
                                       ["VV", "VH", "VH_VV_dB"])
        long = s0.wide_to_long(sample_box(stack, labels, 10),
                               ["VV", "VH", "VH_VV_dB"])
        long["orbit"] = tag
        frames.append(long)
    pd.concat(frames, ignore_index=True).to_csv(
        OUT / "ts_s1_box3x3.csv", index=False)

    (OUT / "stage0b_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
