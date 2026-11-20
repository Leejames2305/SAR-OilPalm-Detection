"""Per-tree features from the raw sources only: SAR window means and Sentinel-2 NDVI.

Sources of truth (decided 19 Nov 2026):
  - data/Labels/*.csv        coordinates and classes, all three estates
  - data/SAR-scenes/*.tif    ALOS-2 scenes for the 260610 date, one set per estate

No file under data/Processed or misc/ is read here. NDVI comes from Google Earth Engine
(Sentinel-2 SR, the POC's window and cloud mask), not from any CSV.

Conventions follow the POC so numbers stay comparable:
  - NDVI: S2_SR_HARMONIZED, 2026-05-15..2026-07-15, SCL classes 4-7 kept, per-pixel median,
    normalizedDifference(B8, B4), reduceRegions(mean) at 10 m.
  - SAR: 3x3 window mean of linear power per band. Non-finite and <= 0 pixels are masked,
    the full window must fit, at least 5 valid pixels are needed.

SAR band meaning, unverified (file metadata has no band names):
  - Spk (4 bands): HH, HV, VH, VV. Inferred from the statistics: cross-pol ratio ~1,
    co-pol HH/VV ~1.3-1.45. The POC used the same order.
  - HAlpha (3 bands, linear fractions): HAlpha_1..3. Band meaning unverified.
  - Yamaguchi is excluded: dB-like values (median -9 to -18), so the mask <= 0 rule empties
    every tree. No POC used it and no conversion is specified, so it is deferred.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import ESTATES, KEY_FILE, LABEL_DIR, LABEL_FILES, REPO_ROOT

SAR_DATE = "260610"
# product -> (file suffix after estate and date, output column names)
SAR_PRODUCTS = {
    "Spk": ("Cal_Spk_TC.tif", ["HH", "HV", "VH", "VV"]),
    "HAlpha": ("Cal_mat_HAlpha_Spk_TC.tif", ["HAlpha_1", "HAlpha_2", "HAlpha_3"]),
}
S2_START, S2_END = "2026-05-15", "2026-07-15"
SCL_KEEP = (4, 5, 6, 7)
NDVI_COLS = ["B4", "B8", "NDVI", "clear_count"]


@dataclass(frozen=True)
class Tree:
    estate: str
    id: int
    lon: float
    lat: float
    cls: str  # Healthy | Unhealthy | Middle


def load_labels() -> list[Tree]:
    """Read the raw label CSVs. Returns every tree, Middle included."""
    trees: list[Tree] = []
    for estate, fname in LABEL_FILES.items():
        with open(LABEL_DIR / fname, newline="") as fh:
            for r in csv.DictReader(fh):
                trees.append(Tree(estate, int(r["id"]), float(r["Long"]), float(r["Lat"]), r["Class"]))
    return trees


def scene_path(estate: str, product: str):
    suffix = SAR_PRODUCTS[product][0]
    p = REPO_ROOT / "data" / "SAR-scenes" / f"ALOS2-Subset_{estate}_{SAR_DATE}_{suffix}"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def sample_window_means(tif, lon: np.ndarray, lat: np.ndarray, nbands: int) -> np.ndarray:
    """3x3 linear-power mean per tree, per band. Returns (n_trees, nbands), NaN where unusable."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window

    out = np.full((len(lon), nbands), np.nan)
    with rasterio.open(tif) as src:
        if src.count != nbands:
            raise ValueError(f"{tif.name}: expected {nbands} bands, found {src.count}")
        if src.crs and src.crs.to_epsg() != 4326:
            xs, ys = transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
            xs, ys = np.asarray(xs), np.asarray(ys)
        else:
            xs, ys = lon, lat
        for i, (x, y) in enumerate(zip(xs, ys)):
            row, col = src.index(x, y)
            r0, c0 = row - 1, col - 1
            if r0 < 0 or c0 < 0 or r0 + 3 > src.height or c0 + 3 > src.width:
                continue
            win = src.read(window=Window(c0, r0, 3, 3)).astype(float)
            win[~np.isfinite(win)] = np.nan
            win[win <= 0.0] = np.nan
            for b in range(nbands):
                v = win[b][np.isfinite(win[b])]
                if v.size >= 5:
                    out[i, b] = float(v.mean())
    return out


def sample_ndvi_earth_engine(trees: list[Tree], project: str) -> dict[tuple[str, int], dict]:
    """Per-tree S2 NDVI, B4, B8 and clear-observation count, sampled server-side in Earth Engine.

    Each estate uses its own label extent as the AOI, with the same window and cloud mask.
    """
    import ee

    with open(KEY_FILE) as fh:
        client_email = json.load(fh)["client_email"]
    creds = ee.ServiceAccountCredentials(client_email, str(KEY_FILE))
    ee.Initialize(creds, project=project)

    result: dict[tuple[str, int], dict] = {}
    by_estate: dict[str, list[Tree]] = {}
    for t in trees:
        by_estate.setdefault(t.estate, []).append(t)

    for estate, ts in by_estate.items():
        lons = [t.lon for t in ts]
        lats = [t.lat for t in ts]
        aoi = ee.Geometry.Rectangle([min(lons) - 0.001, min(lats) - 0.001,
                                     max(lons) + 0.001, max(lats) + 0.001])
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(aoi).filterDate(S2_START, S2_END))

        def mask(img):
            scl = img.select("SCL")
            keep = scl.eq(SCL_KEEP[0])
            for c in SCL_KEEP[1:]:
                keep = keep.Or(scl.eq(c))
            return img.updateMask(keep)

        masked = col.map(mask)
        comp = masked.median()
        ndvi = comp.normalizedDifference(["B8", "B4"]).rename("NDVI")
        clear = masked.select("B4").count().rename("clear_count")
        stack = comp.select(["B4", "B8"]).addBands([ndvi, clear])

        for start in range(0, len(ts), 500):
            chunk = ts[start:start + 500]
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([t.lon, t.lat]), {"id": t.id}) for t in chunk
            ])
            feats = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10).getInfo()["features"]
            for f in feats:
                p = f["properties"]
                result[(estate, int(p["id"]))] = {k: p.get(k) for k in NDVI_COLS}
    return result


def build_feature_table(project: str) -> pd.DataFrame:
    """One row per tree (all three estates, Middle included). Feature columns only, no labels."""
    trees = load_labels()
    sar_cols = [c for _, cols in SAR_PRODUCTS.values() for c in cols]
    rows = {(t.estate, t.id): {"estate": t.estate, "id": t.id} for t in trees}

    for estate in ESTATES:
        sub = [t for t in trees if t.estate == estate]
        lon = np.array([t.lon for t in sub])
        lat = np.array([t.lat for t in sub])
        for product, (_, cols) in SAR_PRODUCTS.items():
            arr = sample_window_means(scene_path(estate, product), lon, lat, len(cols))
            for t, vals in zip(sub, arr):
                for c, v in zip(cols, vals):
                    rows[(t.estate, t.id)][c] = float(v) if np.isfinite(v) else np.nan

    ndvi = sample_ndvi_earth_engine(trees, project)
    for key, rec in rows.items():
        rec.update(ndvi.get(key, {k: np.nan for k in NDVI_COLS}))

    df = pd.DataFrame(list(rows.values()))
    return df[["estate", "id"] + sar_cols + NDVI_COLS]
