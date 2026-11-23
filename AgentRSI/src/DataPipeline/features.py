"""Per-tree features from the raw sources only: multi-date SAR window means and Sentinel-2 NDVI.

Sources of truth (decided 19 Nov 2026; multi-date SAR added 23 Nov 2026):
  - data/Labels/*.csv        coordinates and classes, all three estates
  - data/SAR-scenes/*.tif    ALOS-2 scenes across all complete dates (auto-discovered: 260610, 261014, ...)

No file under data/Processed or misc/ is read here. NDVI comes from Google Earth Engine
(Sentinel-2 SR, the POC's window and cloud mask), not from any CSV.

Conventions follow the POC and main.py so numbers stay comparable:
  - NDVI: S2_SR_HARMONIZED, 2026-05-15..2026-07-15, SCL classes 4-7 kept, per-pixel median,
    normalizedDifference(B8, B4), reduceRegions(mean) at 10 m.
  - SAR: 3x3 window mean per band across every complete scene date (date-suffixed columns).
    Non-finite and <= 0 pixels are masked, the full window must fit, at least 5 valid pixels
    are needed.

SAR band order (authoritative per main.py / SNAP pipeline; GeoTIFFs carry no band names):
  - Spk (4 bands, linear power): HH, HV, VH, VV.
  - HAlpha (3 bands, linear): HAlpha_H (entropy, 0..1), HAlpha_A (anisotropy, 0..1),
    HAlpha_alpha (alpha angle, degrees).
  - Yamaguchi is excluded from the base table: dB-like values (median -9 to -18), so the
    mask <= 0 rule empties every tree. Deferred until a conversion rule is chosen; the raw
    GeoTIFFs remain exposed to agents in environment/rasters/.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import (
    ESTATES,
    KEY_FILE,
    LABEL_DIR,
    LABEL_FILES,
    NDVI_COLS,
    SAR_DIR,
    SAR_PRODUCTS,
    discover_sar_dates,
    feature_cols,
    sar_feature_cols,
)

S2_START, S2_END = "2026-05-15", "2026-07-15"
SCL_KEEP = (4, 5, 6, 7)


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


def scene_path(estate: str, date: str, product: str):
    suffix = SAR_PRODUCTS[product][0]
    p = SAR_DIR / f"ALOS2-Subset_{estate}_{date}_{suffix}"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def sample_window_means(tif, lon: np.ndarray, lat: np.ndarray, nbands: int) -> np.ndarray:
    """3x3 window mean per tree, per band. Returns (n_trees, nbands), NaN where unusable."""
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


def extract_sar_table(trees: list[Tree] | None = None, dates: list[str] | None = None) -> pd.DataFrame:
    """Extract multi-date SAR features locally for every tree. No labels."""
    trees = load_labels() if trees is None else trees
    use_dates = discover_sar_dates() if dates is None else dates
    if not use_dates:
        raise FileNotFoundError(f"No complete SAR scene date sets found in {SAR_DIR}")

    rows = {(t.estate, t.id): {"estate": t.estate, "id": t.id} for t in trees}
    for estate in ESTATES:
        sub = [t for t in trees if t.estate == estate]
        lon = np.array([t.lon for t in sub])
        lat = np.array([t.lat for t in sub])
        for dt in use_dates:
            for product, (_, cols) in SAR_PRODUCTS.items():
                arr = sample_window_means(scene_path(estate, dt, product), lon, lat, len(cols))
                for t, vals in zip(sub, arr):
                    for c, v in zip(cols, vals):
                        rows[(t.estate, t.id)][f"{c}_{dt}"] = float(v) if np.isfinite(v) else np.nan

    df = pd.DataFrame(list(rows.values()))
    return df[["estate", "id"] + sar_feature_cols(use_dates)]


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


def build_feature_table(project: str, dates: list[str] | None = None) -> pd.DataFrame:
    """One row per tree (all three estates, Middle included). Feature columns only, no labels."""
    trees = load_labels()
    use_dates = discover_sar_dates() if dates is None else dates
    sar_df = extract_sar_table(trees, use_dates)
    ndvi = sample_ndvi_earth_engine(trees, project)
    ndvi_rows = [
        {"estate": e, "id": tid, **vals}
        for (e, tid), vals in ndvi.items()
    ]
    ndvi_df = pd.DataFrame(ndvi_rows)
    df = sar_df.merge(ndvi_df, on=["estate", "id"], how="left", validate="one_to_one")
    return df[["estate", "id"] + feature_cols(use_dates)]
