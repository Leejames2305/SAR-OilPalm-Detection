"""Stage 0 of POC time-series (AirHitam only): pull per-tree GEE time series.

Window: 12 months ending at the AirHitam survey date (2025-06-10..2026-06-11).
Reference: docs/AgentPlan/GEECatalogSurvey.md.

Method: each dataset's dated images are renamed to "<YYYYMMDD>_<band>" and
stacked with toBands(); one reduceRegions call per tree chunk returns one
feature per tree (keeps every getInfo under the 5000-element cap).

Per tree (~10 m mean at tree coordinates):
  S1  COPERNICUS/S1_GRD  VV, VH, VH_VV_dB, per relative orbit (172 asc, 91 desc)
  S2  COPERNICUS/S2_SR_HARMONIZED  NDVI, NDRE (B8A-B5), SCL-masked
  L8  LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI (30 m)
  DW  GOOGLE/DYNAMICWORLD/V1 mode label + fractions over window (mask)
Estate-level per-date covariates (AOI mean):
  SMAP SPL3SMP_E/006 soil moisture am/pm (QA-filtered, 9 km)
  ERA5-Land DAILY_AGGR soil water layer 1, precipitation
  CHIRPS daily precipitation

Outputs (misc/POC_Results/TimeSeries/):
  ts_s1.csv, ts_s2.csv, ts_l8ndvi.csv   long format: id, Class, date, [values]
  ts_dw_mask.csv                         per-tree Dynamic World
  ts_covariates.csv                      date-level estate covariates
  stage0_summary.json
"""
import json
from pathlib import Path

import pandas as pd
import ee

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "TimeSeries"
OUT.mkdir(parents=True, exist_ok=True)

KEY = str(ROOT / "data" / "ZIPs" / "Sources" / "Keys_SABucketRead.json")
PROJECT = "project-33f6cb27-9693-4728-8d9"

WIN_START, WIN_END = "2025-06-10", "2026-06-11"
S2_SCL_KEEP = [4, 5, 6, 7]
S1_ORBITS = {172: "asc", 91: "desc"}
TREE_CHUNK = 500


def init():
    d = json.load(open(KEY))
    ee.Initialize(ee.ServiceAccountCredentials(d["client_email"], KEY),
                  project=PROJECT)
    return d["client_email"]


def load_labels():
    return pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")


def point_fc(labels):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.Long), float(r.Lat)]),
                   {"id": int(r.id), "Class": str(r.Class)})
        for r in labels.itertuples()])


def date_prefixed_stack(col, bands):
    """Stack images into one image; band names are 'YYYYMMDD__<band>__<k>'.

    The index k is the image's position in the date-sorted collection, so
    same-date scenes (e.g. two S2 tiles over the estate) are all kept.
    """
    col = col.sort("system:time_start")
    indexed = col.toList(col.size())

    def rename(i):
        img = ee.Image(indexed.get(i))
        tag = ee.Date(img.get("system:time_start")).format("YYYYMMdd")
        names = [ee.String(tag).cat("__").cat(b).cat("__").cat(
            ee.Number(i).format("%03d")) for b in bands]
        return img.select(bands).rename(names)

    stack = ee.ImageCollection.fromImages(
        ee.List.sequence(0, col.size().subtract(1)).map(rename))
    return stack.toBands()



def sample_stack(stack, labels, scale):
    """Sample a date-stacked image at every tree, in chunks; returns wide frame."""
    frames = []
    for i in range(0, len(labels), TREE_CHUNK):
        fc = point_fc(labels.iloc[i:i + TREE_CHUNK])
        s = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                scale=scale)
        feats = s.getInfo()["features"]
        frames.append(pd.DataFrame([f["properties"] for f in feats]))
        print(f"    chunk {i // TREE_CHUNK + 1}/{-(-len(labels) // TREE_CHUNK)}")
    return pd.concat(frames, ignore_index=True)


def wide_to_long(wide, value_bands):
    """Long frame (id, Class, date, band, value); same-date scenes averaged per tree."""
    meta = ["id", "Class"]
    cols = [c for c in wide.columns if c not in meta]
    rows = []
    for c in cols:
        head, band, _ = c.rsplit("__", 2)
        if band not in value_bands:
            continue
        date = head.rsplit("_", 1)[-1]
        sub = wide[meta + [c]].rename(columns={c: "value"})
        sub["date"] = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        sub["band"] = band
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    return (long.groupby(["id", "Class", "date", "band"], as_index=False)
                ["value"].mean())



def s2_masked_indices(img):
    scl = img.select("SCL")
    keep = scl.eq(S2_SCL_KEEP[0])
    for c in S2_SCL_KEEP[1:]:
        keep = keep.Or(scl.eq(c))
    img = img.updateMask(keep)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = img.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    return ee.Image.cat([ndvi, ndre]).set("system:time_start",
                                          img.get("system:time_start"))


def s1_bands(img):
    vv = img.select("VV")
    vh = img.select("VH")
    ratio = vh.subtract(vv).rename("VH_VV_dB")
    return ee.Image.cat([vv, vh, ratio]).set("system:time_start",
                                             img.get("system:time_start"))


def main():
    account = init()
    print("EE init OK as", account)
    labels = load_labels()
    print("trees:", len(labels))
    aoi = ee.Geometry.Rectangle([float(labels.Long.min()) - 0.001,
                                 float(labels.Lat.min()) - 0.001,
                                 float(labels.Long.max()) + 0.001,
                                 float(labels.Lat.max()) + 0.001])
    summary = {"window": [WIN_START, WIN_END], "account": account,
               "n_trees": len(labels),
               "smap_note": "retrieval_qual_flag == 1 on all retrieved dates; "
                            "catalogue says 0 = pass, so flag mask was not applied"}

    print("[DW] mask")
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(aoi).filterDate(WIN_START, WIN_END))
    dw_bands = ["built", "bare", "water", "trees", "crops", "grass"]
    dw_img = ee.Image.cat([dw.select("label").mode().rename("dw_label"),
                           dw.select(dw_bands).mean()])
    dw_wide = sample_stack(dw_img, labels, 10)
    dw_wide.drop(columns=[c for c in dw_wide.columns
                          if c.startswith("system:")], errors="ignore") \
        .to_csv(OUT / "ts_dw_mask.csv", index=False)
    print("  DW rows:", len(dw_wide))

    print("[S2] NDVI/NDRE")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(WIN_START, WIN_END))
    summary["s2_scenes"] = s2.size().getInfo()
    s2_stack = date_prefixed_stack(s2.map(s2_masked_indices),
                                   ["NDVI", "NDRE"])
    s2_wide = sample_stack(s2_stack, labels, 10)
    wide_to_long(s2_wide, ["NDVI", "NDRE"]).to_csv(
        OUT / "ts_s2.csv", index=False)
    print("  S2 scenes:", summary["s2_scenes"], "cols:", len(s2_wide.columns))

    print("[L8] 8-day NDVI")
    l8 = (ee.ImageCollection("LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI")
          .filterBounds(aoi).filterDate(WIN_START, WIN_END)
          .map(lambda i: i.select(["NDVI"]).set(
              "system:time_start", i.get("system:time_start"))))
    summary["landsat_composites"] = l8.size().getInfo()
    l8_wide = sample_stack(date_prefixed_stack(l8, ["NDVI"]), labels, 30)
    wide_to_long(l8_wide, ["NDVI"]).to_csv(OUT / "ts_l8ndvi.csv", index=False)
    print("  L8 composites:", summary["landsat_composites"])

    print("[S1] VV/VH by orbit")
    s1_base = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(aoi).filterDate(WIN_START, WIN_END)
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains(
                   "transmitterReceiverPolarisation", "VH")))
    frames = []
    for orbit_no, tag in S1_ORBITS.items():
        col = s1_base.filter(ee.Filter.eq("relativeOrbitNumber_start",
                                          orbit_no))
        n = col.size().getInfo()
        summary[f"s1_{tag}_scenes"] = n
        print(f"  orbit {orbit_no} ({tag}): {n} scenes")
        stack = date_prefixed_stack(col.map(s1_bands),
                                    ["VV", "VH", "VH_VV_dB"])
        long = wide_to_long(sample_stack(stack, labels, 10),
                            ["VV", "VH", "VH_VV_dB"])
        long["orbit"] = tag
        frames.append(long)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "ts_s1.csv", index=False)

    print("[COV] estate covariates")
    aoi_fc = ee.FeatureCollection([ee.Feature(aoi)])

    def estate_series(col, band, scale, label, mult=1.0):
        def per(img):
            v = img.select(band).multiply(mult).reduceRegions(
                collection=aoi_fc, reducer=ee.Reducer.mean(), scale=scale)
            return v.map(lambda f: ee.Feature(None, {
                "date": ee.Date(img.get("system:time_start")).format(
                    "YYYY-MM-dd"),
                "variable": label,
                "value": f.get("mean")}))
        feats = ee.FeatureCollection(col.map(per)).flatten().getInfo()["features"]
        return pd.DataFrame([f["properties"] for f in feats])

    smap = (ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
            .filterBounds(aoi).filterDate(WIN_START, WIN_END))
    smap_am = estate_series(smap, "soil_moisture_am", 9000, "smap_sm_am")
    smap_pm = estate_series(smap, "soil_moisture_pm", 9000, "smap_sm_pm")
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(aoi).filterDate(WIN_START, WIN_END))
    era5_sw = estate_series(era5, "volumetric_soil_water_layer_1", 11132,
                            "era5_swl1")
    era5_pr = estate_series(era5, "total_precipitation_sum", 11132,
                            "era5_precip_m")
    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterBounds(aoi).filterDate(WIN_START, WIN_END))
    chirps_pr = estate_series(chirps, "precipitation", 5566,
                              "chirps_precip_mm")
    pd.concat([smap_am, smap_pm, era5_sw, era5_pr, chirps_pr]).to_csv(
        OUT / "ts_covariates.csv", index=False)

    (OUT / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
