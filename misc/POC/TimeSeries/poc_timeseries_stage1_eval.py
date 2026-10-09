"""Stage 1 of POC time-series (AirHitam): per-tree temporal features + spatial eval.

Input: misc/POC_Results/TimeSeries/ts_*.csv (from poc_timeseries_stage0_pull.py).

Per-tree temporal features (per band / orbit):
  mean, std, cv, slope (dB or NDVI per day), delta90 (last 90 d - first 90 d),
  anom_mean / anom_std: value minus the estate median on the same date.
  Anomalies remove shared seasonal and weather signal without using labels.

Evaluation follows poc_rvi_ndvi_stage2_gee.py: balanced LogReg, StandardScaler,
15 KMeans spatial blocks, StratifiedGroupKFold(5). Reports fold-mean PR-AUC
(comparable with stage 2) and pooled out-of-fold PR, ROC and recall@25%.
Windows: 12 months (primary) and last 6 months (robustness).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TS = ROOT / "misc" / "POC_Results" / "TimeSeries"
MIN_OBS = 10
SEED = 42
UNHEALTHY = "Unhealthy"
WINDOWS = {"12m": "2025-06-10", "6m": "2025-12-10"}


def wide(long, band, orbit=None):
    d = long[long.band == band]
    if orbit is not None:
        d = d[d.orbit == orbit]
    return d.pivot_table(index="id", columns="date", values="value",
                         aggfunc="mean")


def temporal_stats(M, prefix):
    cols = sorted(M.columns)
    M = M[cols]
    t_dt = pd.to_datetime(cols)
    t = (t_dt - t_dt.min()).days.astype(float)
    first = t <= t.min() + 90
    last = t >= t.max() - 90
    V = M.to_numpy()
    A = (M - M.median(axis=0)).to_numpy()
    n = M.notna().sum(axis=1).to_numpy()
    slope = np.full(len(M), np.nan)
    delta = np.full(len(M), np.nan)
    for i in range(len(M)):
        y = V[i]
        ok = ~np.isnan(y)
        if ok.sum() >= MIN_OBS:
            slope[i] = np.polyfit(t[ok], y[ok], 1)[0]
        f_ok, l_ok = ok & first, ok & last
        if f_ok.sum() >= 3 and l_ok.sum() >= 3:
            delta[i] = y[l_ok].mean() - y[f_ok].mean()
    mean = np.nanmean(V, axis=1)
    std = np.nanstd(V, axis=1, ddof=1)
    feats = pd.DataFrame({
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": std / np.abs(mean),
        f"{prefix}_slope": slope,
        f"{prefix}_delta90": delta,
        f"{prefix}_anom_mean": np.nanmean(A, axis=1),
        f"{prefix}_anom_std": np.nanstd(A, axis=1, ddof=1),
    }, index=M.index)
    feats.loc[n < MIN_OBS] = np.nan
    return feats


def build(labels, s2, s1, start, end="9999-12-31"):
    ids = labels["id"].to_numpy()
    cols_ok = lambda M: [c for c in M.columns if start <= c <= end]
    parts = []
    for band, prefix in [("NDVI", "s2ndvi"), ("NDRE", "s2ndre")]:
        M = wide(s2, band).reindex(ids)
        M = M[cols_ok(M)]
        parts.append(temporal_stats(M, prefix))
    for orbit in ["asc", "desc"]:
        for band in ["VV", "VH", "VH_VV_dB"]:
            M = wide(s1, band, orbit).reindex(ids)
            M = M[cols_ok(M)]
            parts.append(temporal_stats(M, f"s1{orbit}_{band}"))
    feats = pd.concat(parts, axis=1)
    feats.index = ids
    return feats


def groups(cols):
    s2 = [c for c in cols if c.startswith("s2")]
    s1 = [c for c in cols if c.startswith("s1")]
    return {
        "NDVI_mean": ["s2ndvi_mean"],
        "NDVI+NDRE_mean": ["s2ndvi_mean", "s2ndre_mean"],
        "REF_stage2_NDVI": ["ref_ndvi"],
        "REF_stage2_NDVI+S2_temporal": ["ref_ndvi"] + s2,
        "REF_stage2_NDVI+S2_temporal+S1_temporal": ["ref_ndvi"] + s2 + s1,
        "S2_temporal": s2,
        "S1_temporal": s1,
        "S1_anom_only": [c for c in s1 if "anom" in c],
        "S1_static_only": [c for c in s1 if "anom" not in c],
        "NDVI_mean+S1_temporal": ["s2ndvi_mean"] + s1,
        "NDVI_mean+S1_anom": ["s2ndvi_mean"] + [c for c in s1 if "anom" in c],
        "S2_temporal+S1_temporal": s2 + s1,
    }


def spatial_eval(X, y, blocks):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000,
                                           class_weight="balanced"))
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    fold_pr = []
    for tr, te in sgkf.split(X, y, blocks):
        clf.fit(X[tr], y[tr])
        s = clf.decision_function(X[te])
        oof[te] = s
        fold_pr.append(average_precision_score(y[te], s))
    top = np.argsort(-oof)[: int(round(0.25 * len(y)))]
    recall25 = float(y[top].sum() / y.sum())
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rand_pr = []
    for tr, te in rskf.split(X, y):
        clf.fit(X[tr], y[tr])
        rand_pr.append(average_precision_score(y[te], clf.decision_function(X[te])))
    return dict(
        random_pr=round(float(np.mean(rand_pr)), 4),
        spatial_pr_foldmean=round(float(np.mean(fold_pr)), 4),
        spatial_pr_pooled=round(float(average_precision_score(y, oof)), 4),
        spatial_roc_pooled=round(float(roc_auc_score(y, oof)), 4),
        recall_at_25=round(recall25, 4),
        enrich_at_25=round(recall25 / float(y.mean()), 3),
    )


def univariate(feat, y):
    h = feat[y == 0].dropna()
    u = feat[y == 1].dropna()
    sd = np.sqrt((h.var(ddof=1) + u.var(ddof=1)) / 2.0)
    d = (u.mean() - h.mean()) / sd if sd > 0 else np.nan
    p = mannwhitneyu(h, u, method="asymptotic").pvalue
    s = feat.dropna().to_numpy() * (1 if d < 0 else -1)
    yy = y[feat.notna().to_numpy()]
    roc = roc_auc_score(yy, -s if d < 0 else s)
    pr = average_precision_score(yy, -s if d < 0 else s)
    return dict(n=len(feat.dropna()), cohen_d=round(float(d), 4),
                mwu_p=float(p), roc_auc_dir=round(float(roc), 4),
                pr_auc_dir=round(float(pr), 4))


def main():
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    s2 = pd.read_csv(TS / "ts_s2.csv")
    s1 = pd.read_csv(TS / "ts_s1.csv")
    y = (labels["Class"] == UNHEALTHY).astype(int).to_numpy()
    pos_rate = float(y.mean())
    coords = labels[["Long", "Lat"]].to_numpy()
    blocks = KMeans(n_clusters=15, n_init=10,
                    random_state=SEED).fit_predict(coords)
    ref = pd.read_csv(ROOT / "misc/POC_Results/RVI_NDVI/airhitam_ndvi.csv")
    ref = ref.set_index("id")["NDVI"]

    summary = {"no_skill_pr": round(pos_rate, 4), "windows": {}}
    for wname, start in WINDOWS.items():
        feats = build(labels, s2, s1, start)
        too_sparse = feats.columns[feats.isna().mean() > 0.5].tolist()
        feats = feats.drop(columns=too_sparse)
        print(f"\n[{wname}] dropped {len(too_sparse)} sparse features: {too_sparse}")
        n_before = int(feats.isna().any(axis=1).sum())
        feats_imp = feats.fillna(feats.median())
        feats_imp["ref_ndvi"] = ref.reindex(feats_imp.index).to_numpy()
        feats_imp.to_csv(TS / f"stage1_features_{wname}.csv")

        uni = pd.DataFrame({c: univariate(feats[c], y) for c in feats.columns}).T
        uni.to_csv(TS / f"stage1_univariate_{wname}.csv")

        rows = {}
        for gname, cols in groups(list(feats.columns)).items():
            X = feats_imp[cols].to_numpy()
            rows[gname] = {"n_feats": len(cols), **spatial_eval(X, y, blocks)}
        ev = pd.DataFrame(rows).T
        ev.to_csv(TS / f"stage1_eval_{wname}.csv")
        print(f"\n== window {wname} (from {start}), trees with any NaN feature: {n_before}")
        print(ev.to_string())
        print("\ntop univariate by |cohen_d|:")
        print(uni.assign(abs_d=uni.cohen_d.abs()).sort_values(
            "abs_d", ascending=False).head(10).to_string())
        summary["windows"][wname] = {"start": start, "n_feats": feats.shape[1],
                                     "trees_with_nan": n_before}

    (TS / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nno-skill PR:", round(pos_rate, 4))


if __name__ == "__main__":
    main()
