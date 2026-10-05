"""Stage 2 of POC RVI+NDVI (AirHitam only, real ground truth).

Pulls Sentinel-2 L2A around the SAR date (2026-06-10) from Google Earth
Engine, samples per-tree NDVI, and tests:
  Q1: does NDVI alone separate Healthy/Unhealthy?
  Q2: does RVI + NDVI fusion beat RVI alone (and raw-band reference)?

Requires: earthengine-api in .venv + a service account with
  roles/serviceusage.serviceUsageConsumer on the GCP project
  (Earth Engine Resources Viewer alone is NOT enough - init fails).

S2 design: COPERNICUS/S2_SR_HARMONIZED, 2026-05-15..2026-07-15
  (~ +/-30 d around SAR; oil palm near-static), SCL cloud/shadow mask
  (keep classes 4,5,6,7), per-pixel median composite.
  NDVI = (B8 - B4) / (B8 + B4); B4/B8 medians + clear-obs count kept for QA.
Sampling: reduceRegions(mean, 10 m) at the 2511 census coordinates.

Outputs (misc/POC_Results/RVI_NDVI/):
  airhitam_ndvi.csv, stage2_summary.json,
  fig_ndvi_kde.png, fig_rvi_ndvi_scatter.png, fig_fusion_roc.png
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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, cross_val_predict)
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

# Window around the SAR date 2026-06-10 (oil palm near-static over weeks).
S2_START, S2_END = "2026-05-15", "2026-07-15"
SCL_KEEP = [4, 5, 6, 7]  # veg, bare soil, water, unclassified


def s2_masked(image):
    scl = image.select("SCL")
    keep = scl.eq(SCL_KEEP[0])
    for c in SCL_KEEP[1:]:
        keep = keep.Or(scl.eq(c))
    return image.updateMask(keep)


def main():
    d = json.load(open(KEY))
    creds = ee.ServiceAccountCredentials(d["client_email"], KEY)
    ee.Initialize(creds, project=PROJECT)
    print("EE init OK as", d["client_email"])

    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    print("trees:", len(labels))
    aoi = ee.Geometry.Rectangle([float(labels.Long.min()),
                                 float(labels.Lat.min()),
                                 float(labels.Long.max()),
                                 float(labels.Lat.max())])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(S2_START, S2_END))
    n_scenes = col.size().getInfo()
    print(f"S2 scenes {S2_START}..{S2_END}: {n_scenes}")
    assert n_scenes > 0, "no S2 scenes in window - widen dates"

    masked = col.map(s2_masked)
    comp = masked.median()
    ndvi = comp.normalizedDifference(["B8", "B4"]).rename("NDVI")
    clear = masked.select("B4").count().rename("clear_count")
    stack = comp.select(["B4", "B8"]).addBands([ndvi, clear])

    pts = [ee.Feature(ee.Geometry.Point([float(r.Long), float(r.Lat)]),
                      {"id": int(r.id), "Class": str(r.Class)})
           for r in labels.itertuples()]
    fc = ee.FeatureCollection(pts)
    sampled = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                  scale=10)
    feats = sampled.getInfo()["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({"id": p["id"], "Class": p["Class"],
                     "B4": p.get("B4"), "B8": p.get("B8"),
                     "NDVI": p.get("NDVI"),
                     "clear_count": p.get("clear_count")})
    ndvi_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    ndvi_df.to_csv(OUT / "airhitam_ndvi.csv", index=False)
    n_missing = int(ndvi_df["NDVI"].isna().sum())
    print(f"sampled NDVI for {len(ndvi_df)} trees, missing={n_missing}")
    print(ndvi_df.groupby("Class")["clear_count"].describe().to_string())

    # ---- merge with Stage-1 RVI ----
    rvi = pd.read_csv(OUT / "airhitam_rvi_features.csv")
    m = rvi.merge(ndvi_df, on=["id", "Class"], how="inner")
    m = m.dropna(subset=["NDVI"]).reset_index(drop=True)
    print("merged rows:", len(m), "| class balance:",
          m["Class"].value_counts().to_dict())
    y = (m["Class"] == UNHEALTHY).to_numpy()
    pos_rate = float(y.mean())

    def univariate(h, u):
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
                    median_h=round(float(h.median()), 5),
                    median_u=round(float(u.median()), 5),
                    cohen_d=round(dd, 4), mwu_p=p,
                    roc_auc=round(roc, 4), pr_auc=round(ap, 4),
                    pr_lift=round(ap - pos_rate, 4))

    uni = {}
    for f in ["NDVI", "RVI_classic", "RVI_xavg", "HV", "HH"]:
        sub = m[[f, "Class"]].dropna()
        uni[f] = univariate(sub.loc[sub.Class == HEALTHY, f],
                            sub.loc[sub.Class == UNHEALTHY, f])
    print(pd.DataFrame(uni).T.to_string())

    # ---- fusion probe: LR on feature sets ----
    sets = {"RVI_only": ["RVI_classic"],
            "NDVI_only": ["NDVI"],
            "RVI+NDVI": ["RVI_classic", "NDVI"],
            "RVI+NDVI+HV": ["RVI_classic", "NDVI", "HV"]}
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    fusion = {}
    # (a) leaky-reference random CV (ceiling only, per repo convention)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                   random_state=42)
    # (b) honest spatial-block CV: KMeans blocks on coords
    coords = m[["Long", "Lat"]].to_numpy()
    blocks = KMeans(n_clusters=15, n_init=10,
                    random_state=42).fit_predict(coords)
    sgkf = StratifiedGroupKFold(n_splits=5)
    for name, cols in sets.items():
        X = m[cols].to_numpy()
        pr_r, roc_r, pr_s, roc_s = [], [], [], []
        for tr, te in rskf.split(X, y):
            clf.fit(X[tr], y[tr])
            s = clf.decision_function(X[te])
            pr_r.append(average_precision_score(y[te], s))
            roc_r.append(roc_auc_score(y[te], s))
        for tr, te in sgkf.split(X, y, blocks):
            clf.fit(X[tr], y[tr])
            s = clf.decision_function(X[te])
            pr_s.append(average_precision_score(y[te], s))
            roc_s.append(roc_auc_score(y[te], s))
        fusion[name] = dict(
            random_pr=round(float(np.mean(pr_r)), 4),
            random_roc=round(float(np.mean(roc_r)), 4),
            spatial_pr=round(float(np.mean(pr_s)), 4),
            spatial_roc=round(float(np.mean(roc_s)), 4))
    print("no-skill PR:", round(pos_rate, 4))
    print(pd.DataFrame(fusion).T.to_string())

    (OUT / "stage2_summary.json").write_text(json.dumps(
        {"s2_window": [S2_START, S2_END], "n_scenes": n_scenes,
         "n_missing_ndvi": n_missing, "pos_rate": pos_rate,
         "univariate": uni, "fusion": fusion}, indent=2))

    # ---- plots ----
    pal = {HEALTHY: "#2ca02c", UNHEALTHY: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, c in pal.items():
        v = m.loc[m.Class == cls, "NDVI"].dropna()
        sns.kdeplot(v, ax=ax, label=f"{cls} (n={(m.Class == cls).sum()})",
                    color=c, fill=True, alpha=0.3)
    ax.set_title("AirHitam: Sentinel-2 NDVI by class "
                 f"(median composite {S2_START}..{S2_END})")
    ax.set_xlabel("NDVI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_ndvi_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for cls, c in pal.items():
        sub = m[m.Class == cls]
        ax.scatter(sub["RVI_classic"], sub["NDVI"], s=10, alpha=0.5,
                   color=c, label=cls)
    ax.set_xlabel("RVI_classic (SAR)")
    ax.set_ylabel("NDVI (Sentinel-2)")
    ax.set_title("AirHitam: RVI vs NDVI by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_rvi_ndvi_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, cols in sets.items():
        X = m[cols].to_numpy()
        s = cross_val_predict(clf, X, y, cv=5, method="decision_function")
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr,
                label=f"{name} (AUC={roc_auc_score(y, s):.3f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("AirHitam: fusion ROC (5-fold CV scores)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fusion_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()
