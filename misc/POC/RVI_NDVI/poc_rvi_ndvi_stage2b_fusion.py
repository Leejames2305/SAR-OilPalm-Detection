"""Stage 2b of POC RVI+NDVI (AirHitam only). Offline analysis half of
Stage 2: reads airhitam_ndvi.csv (pulled from GEE by stage2_gee.py) +
Stage-1 RVI features and runs:
  Q1: NDVI-alone univariate separability,
  Q2: RVI+NDVI fusion probe (logistic regression, random CV ceiling +
      honest spatial-block CV), with plots.
Outputs (misc/POC_Results/RVI_NDVI/):
  stage2_summary.json, fig_ndvi_kde.png,
  fig_rvi_ndvi_scatter.png, fig_fusion_roc.png
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

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"


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
    rvi = pd.read_csv(OUT / "airhitam_rvi_features.csv")
    ndvi = pd.read_csv(OUT / "airhitam_ndvi.csv")
    qp = pd.read_csv(OUT / "airhitam_rvi_quadpol.csv")
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    m = (rvi.merge(ndvi, on=["id", "Class"], how="inner")
            .merge(qp[["id", "RVI_qp", "span"]], on="id", how="left")
            .merge(labels[["id", "Long", "Lat"]], on="id", how="left"))
    m = m.dropna(subset=["NDVI"]).reset_index(drop=True)
    print("merged rows:", len(m), m["Class"].value_counts().to_dict())
    y = (m["Class"] == UNHEALTHY).to_numpy()
    pos_rate = float(y.mean())
    print("pos_rate =", round(pos_rate, 4))

    uni = {}
    for f in ["NDVI", "RVI_classic", "RVI_xavg", "RVI_qp",
              "HV", "HH", "span"]:
        sub = m[[f, "Class"]].dropna()
        uni[f] = univariate(sub.loc[sub.Class == HEALTHY, f],
                            sub.loc[sub.Class == UNHEALTHY, f],
                            pos_rate)
    print(pd.DataFrame(uni).T.to_string())

    sets = {"RVI_only": ["RVI_classic"],
            "RVIqp_only": ["RVI_qp"],
            "NDVI_only": ["NDVI"],
            "RVI+NDVI": ["RVI_classic", "NDVI"],
            "RVIqp+NDVI": ["RVI_qp", "NDVI"],
            "RVI+NDVI+HV": ["RVI_classic", "NDVI", "HV"]}
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    blocks = KMeans(n_clusters=15, n_init=10, random_state=42).fit_predict(
        m[["Long", "Lat"]].to_numpy())
    sgkf = StratifiedGroupKFold(n_splits=5)
    fusion = {}
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
        {"s2_window": ["2026-05-15", "2026-07-15"], "pos_rate": pos_rate,
         "univariate": uni, "fusion": fusion}, indent=2))

    pal = {HEALTHY: "#2ca02c", UNHEALTHY: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, c in pal.items():
        v = m.loc[m.Class == cls, "NDVI"].dropna()
        sns.kdeplot(v, ax=ax, label=f"{cls} (n={(m.Class == cls).sum()})",
                    color=c, fill=True, alpha=0.3)
    ax.set_title("AirHitam: Sentinel-2 NDVI by class "
                 "(median composite 2026-05-15..07-15)")
    ax.set_xlabel("NDVI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_ndvi_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, rx, t in zip(axes, ["RVI_classic", "RVI_qp"],
                         ["dual-pol-style RVI", "quad-pol eigenvalue RVI"]):
        for cls, c in pal.items():
            sub = m[m.Class == cls]
            ax.scatter(sub[rx], sub["NDVI"], s=10, alpha=0.5, color=c,
                       label=cls)
        ax.set_xlabel(f"{rx} (SAR)")
        ax.set_ylabel("NDVI (Sentinel-2)")
        ax.set_title(f"AirHitam: {t} vs NDVI")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_rvi_ndvi_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, cols in sets.items():
        X = m[cols].to_numpy()
        s = cross_val_predict(clf, X, y, cv=5, method="decision_function")
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, s):.3f})")
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
