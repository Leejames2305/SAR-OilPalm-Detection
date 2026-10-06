"""Stage 4 of POC RVI+NDVI (AirHitam only). Two data-side shots:

(a) Richer spectra: S2 red-edge/SWIR indices over the same composite window
    (2026-05-15..07-15): NDVI, NDRE (B5/B6/B7), EVI, NDMI + raw bands.
(b) Temporal NDVI over +/-90 d of SAR (2026-03-12..09-08): per-pixel median,
    min, std, linear-trend slope, late-minus-early delta; sampled per tree.

Then the same univariate battery + logreg fusion check (random ceiling +
honest spatial-block CV) asking: does anything beat single-date NDVI?

Outputs (misc/POC_Results/RVI_NDVI/):
  airhitam_spectral.csv, stage4_univariate.csv, stage4_summary.json,
  fig_spectral_kde.png, fig_spectral_d.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ee

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
OUT.mkdir(parents=True, exist_ok=True)

KEY = str(ROOT / "data" / "ZIPs" / "Sources" / "Keys_SABucketRead.json")
PROJECT = "project-33f6cb27-9693-4728-8d9"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"
WIN_A = ("2026-05-15", "2026-07-15")     # spectral composite window
WIN_B = ("2026-03-12", "2026-09-08")     # temporal window (+/-90 d of SAR)
SCL_KEEP = [4, 5, 6, 7]
B10 = ["B2", "B3", "B4", "B8"]
B20 = ["B5", "B6", "B7", "B8A", "B11", "B12"]


def s2_masked(image):
    scl = image.select("SCL")
    keep = scl.eq(SCL_KEEP[0])
    for c in SCL_KEEP[1:]:
        keep = keep.Or(scl.eq(c))
    return image.updateMask(keep)


def sample_stack(stack, pts, scale):
    res = stack.reduceRegions(collection=pts, reducer=ee.Reducer.mean(),
                              scale=scale).getInfo()["features"]
    return {int(f["properties"]["id"]): f["properties"] for f in res}


def univariate(h, u, pos_rate):
    h = pd.to_numeric(h, errors="coerce").dropna()
    u = pd.to_numeric(u, errors="coerce").dropna()
    sd_pool = float(np.sqrt((float(h.var(ddof=1)) +
                             float(u.var(ddof=1))) / 2.0))
    dd = (float(u.mean()) - float(h.mean())) / sd_pool if sd_pool > 0 \
        else float("nan")
    try:
        p = float(mannwhitneyu(h, u, method="asymptotic").pvalue)
    except Exception:
        p = float("nan")
    yt = np.array([0] * len(h) + [1] * len(u))
    ys = np.concatenate([h.to_numpy(), u.to_numpy()])
    roc = float(roc_auc_score(yt, ys)) if len(np.unique(ys)) > 1 \
        else float("nan")
    ap = float(average_precision_score(yt, ys if roc >= 0.5 else -ys))
    return dict(n_h=len(h), n_u=len(u),
                mean_h=round(float(h.mean()), 5),
                mean_u=round(float(u.mean()), 5),
                median_h=round(float(h.median()), 5),
                median_u=round(float(u.median()), 5),
                cohen_d=round(dd, 4), mwu_p=p,
                roc_auc=round(roc, 4), pr_auc=round(ap, 4),
                pr_lift=round(ap - pos_rate, 4))


def main():
    d = json.load(open(KEY))
    ee.Initialize(ee.ServiceAccountCredentials(d["client_email"], KEY),
                  project=PROJECT)
    print("EE init OK")
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    aoi = ee.Geometry.Rectangle([float(labels.Long.min()),
                                 float(labels.Lat.min()),
                                 float(labels.Long.max()),
                                 float(labels.Lat.max())])
    pts = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([float(r.Long), float(r.Lat)]),
                    {"id": int(r.id)})
         for r in labels.itertuples()])

    # ---- (a) spectral composite ----
    col_a = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterBounds(aoi).filterDate(*WIN_A))
    print("window-A scenes:", col_a.size().getInfo())
    comp = col_a.map(s2_masked).median()
    ndvi = comp.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre5 = comp.normalizedDifference(["B8A", "B5"]).rename("NDRE5")
    ndre6 = comp.normalizedDifference(["B8A", "B6"]).rename("NDRE6")
    ndre7 = comp.normalizedDifference(["B8A", "B7"]).rename("NDRE7")
    evi = comp.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": comp.select("B8"), "RED": comp.select("B4"),
         "BLUE": comp.select("B2")}).rename("EVI")
    ndmi = comp.normalizedDifference(["B8", "B11"]).rename("NDMI")
    idx10 = ndvi.addBands([evi])
    idx20 = ndre5.addBands([ndre6, ndre7, ndmi])
    s10 = sample_stack(comp.select(B10).addBands(idx10), pts, 10)
    s20 = sample_stack(comp.select(B20).addBands(idx20), pts, 20)
    print("sampled spectral stacks")

    # ---- (b) temporal NDVI ----
    col_b = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterBounds(aoi).filterDate(*WIN_B).map(s2_masked))
    print("window-B scenes:", col_b.size().getInfo())
    t0 = ee.Date(WIN_B[0]).millis()

    def add_ndvi_t(img):
        n = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        t = ee.Image.constant(
            ee.Date(img.get("system:time_start")).difference(
                ee.Date(WIN_B[0]), "day")).rename("t").float()
        return n.addBands(t).copyProperties(img, ["system:time_start"])

    ncol = col_b.map(add_ndvi_t)
    ndvi_only = ncol.select("NDVI")
    t_med = ndvi_only.reduce(ee.Reducer.median()).rename("ndvi_med")
    t_min = ndvi_only.reduce(ee.Reducer.min()).rename("ndvi_min")
    t_std = ndvi_only.reduce(ee.Reducer.stdDev()).rename("ndvi_std")
    t_cnt = ndvi_only.reduce(ee.Reducer.count()).rename("ndvi_cnt")
    slope = ncol.select(["t", "NDVI"]).reduce(
        ee.Reducer.linearFit()).select("scale").rename("ndvi_slope")
    early = ncol.filterDate(WIN_B[0], "2026-04-11").select("NDVI").reduce(
        ee.Reducer.median()).rename("e")
    late = ncol.filterDate("2026-08-09", WIN_B[1]).select("NDVI").reduce(
        ee.Reducer.median()).rename("l")
    delta = late.subtract(early).rename("ndvi_delta")
    tstack = t_med.addBands([t_min, t_std, t_cnt, slope, delta])
    st = sample_stack(tstack, pts, 10)
    print("sampled temporal stack")

    rows = []
    for r in labels.itertuples():
        i = int(r.id)
        rec = {"id": i, "Class": str(r.Class), "NDVI_a": s10[i].get("NDVI"),
               "EVI": s10[i].get("EVI")}
        for b in B10:
            rec[b] = s10[i].get(b)
        for k in ["NDRE5", "NDRE6", "NDRE7", "NDMI"]:
            rec[k] = s20[i].get(k)
        for b in B20:
            rec["spec_" + b] = s20[i].get(b)
        for k in ["ndvi_med", "ndvi_min", "ndvi_std", "ndvi_cnt",
                  "ndvi_slope", "ndvi_delta"]:
            rec[k] = st[i].get(k)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "airhitam_spectral.csv", index=False)
    print("saved spectral rows:", len(df),
          "missing NDVI_a:", int(df["NDVI_a"].isna().sum()))

    y = (df["Class"] == UNHEALTHY).to_numpy()
    pos_rate = float(y.mean())
    feats = ["NDVI_a", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI",
             "ndvi_med", "ndvi_min", "ndvi_std", "ndvi_slope", "ndvi_delta"]
    uni = {}
    for f in feats:
        sub = df[[f, "Class"]].dropna()
        uni[f] = univariate(sub.loc[sub.Class == HEALTHY, f],
                            sub.loc[sub.Class == UNHEALTHY, f], pos_rate)
    tab = pd.DataFrame(uni).T
    print(tab.to_string())
    print(tab.to_csv(OUT / "stage4_univariate.csv"))

    # ---- fusion check ----
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    sets = {"NDVI_only": ["NDVI_a"],
            "+rededge_SWIR": ["NDVI_a", "NDRE5", "NDRE6", "NDRE7",
                              "EVI", "NDMI"],
            "+temporal": ["NDVI_a", "ndvi_med", "ndvi_min", "ndvi_std",
                          "ndvi_slope", "ndvi_delta"],
            "all_spectral": ["NDVI_a", "NDRE5", "NDRE6", "NDRE7", "EVI",
                             "NDMI", "ndvi_med", "ndvi_min", "ndvi_std",
                             "ndvi_slope", "ndvi_delta"]}
    blocks = KMeans(n_clusters=15, n_init=10, random_state=42).fit_predict(
        labels[["Long", "Lat"]].to_numpy())
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    sgkf = StratifiedGroupKFold(n_splits=5)
    fusion = {}
    dd = df.copy()
    for name, cols in sets.items():
        sub = dd[cols + ["Class"]].dropna()
        yy = (sub["Class"] == UNHEALTHY).to_numpy()
        X = sub[cols].to_numpy(dtype=float)
        pr_r, roc_r, pr_s, roc_s = [], [], [], []
        for tr, te in rskf.split(X, yy):
            clf.fit(X[tr], yy[tr])
            s = clf.decision_function(X[te])
            pr_r.append(average_precision_score(yy[te], s))
            roc_r.append(roc_auc_score(yy[te], s))
        bl = KMeans(n_clusters=15, n_init=10,
                    random_state=42).fit_predict(
                        labels.loc[sub.index, ["Long", "Lat"]].to_numpy()) \
            if False else blocks[sub.index.to_numpy()]
        for tr, te in sgkf.split(X, yy, bl):
            clf.fit(X[tr], yy[tr])
            s = clf.decision_function(X[te])
            pr_s.append(average_precision_score(yy[te], s))
            roc_s.append(roc_auc_score(yy[te], s))
        fusion[name] = dict(n=len(sub),
                            rand_pr=round(float(np.mean(pr_r)), 4),
                            rand_roc=round(float(np.mean(roc_r)), 4),
                            spat_pr=round(float(np.mean(pr_s)), 4),
                            spat_roc=round(float(np.mean(roc_s)), 4))
    print(pd.DataFrame(fusion).T.to_string())
    (OUT / "stage4_summary.json").write_text(json.dumps(
        {"win_a": list(WIN_A), "win_b": list(WIN_B), "pos_rate": pos_rate,
         "univariate": uni, "fusion": fusion}, indent=2))

    # ---- plots ----
    pal = {HEALTHY: "#2ca02c", UNHEALTHY: "#d62728"}
    top = tab.reindex(
        tab["cohen_d"].abs().sort_values(ascending=False).index)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.flat, top.index):
        for cls, c in pal.items():
            v = df.loc[df.Class == cls, f].dropna()
            if v.nunique() > 1:
                sns.kdeplot(v, ax=ax, label=cls, color=c, fill=True,
                            alpha=0.3)
        ax.set_title(f"{f} (d={top.loc[f, 'cohen_d']:.2f})")
        ax.legend(fontsize=8)
    fig.suptitle("AirHitam: top spectral/temporal features by |Cohen d|",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectral_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dd2 = tab.sort_values("cohen_d")
    ax.barh(dd2.index, dd2["cohen_d"].astype(float))
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Cohen d (Unhealthy - Healthy)")
    ax.set_title("AirHitam: effect sizes across spectral/temporal features")
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectral_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()

