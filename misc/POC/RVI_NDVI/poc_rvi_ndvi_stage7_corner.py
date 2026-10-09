"""Stage 7: corner-deployment simulation (AirHitam).

Deployment: owner labels one corner of the estate, model predicts rest.
Design: 4 quadrants via median lon/lat split. For each corner rotation:
  (a) train on corner (~25% data), test on rest (spatial gap, the product);
  (b) size-matched RANDOM train subset, test on rest (controls for sample
      size: isolates what the spatial GAP costs vs small data alone).
Model: LogReg. Sets: NDVI_only, +rededge_SWIR, ALL_SAR reference.
Metric: PR/ROC-AUC + recall@25% on the held-out rest.

Outputs: stage7_table.csv, stage7_summary.json, fig_corner.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"

SETS = {
    "NDVI_only": ["NDVI"],
    "+rededge_SWIR": ["NDVI", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI"],
    "ALL_SAR": ["HH", "HV", "VH", "VV", "span",
                "RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI",
                "RVI_qp"],
}


def metrics(y_true, score):
    return (round(float(average_precision_score(y_true, score)), 4),
            round(float(roc_auc_score(y_true, score)), 4))


def main():
    rvi = pd.read_csv(OUT / "airhitam_rvi_features.csv")
    ndvi = pd.read_csv(OUT / "airhitam_ndvi.csv")
    qp = pd.read_csv(OUT / "airhitam_rvi_quadpol.csv")
    spec = pd.read_csv(OUT / "airhitam_spectral.csv")
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    m = (rvi.merge(ndvi, on=["id", "Class"], how="inner")
            .merge(qp[["id", "RVI_qp", "span"]], on="id", how="left")
            .merge(spec[["id", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI"]],
                   on="id", how="left")
            .merge(labels[["id", "Long", "Lat"]], on="id", how="left"))
    m = m.dropna().reset_index(drop=True)
    y = (m["Class"] == UNHEALTHY).to_numpy()
    print("rows:", len(m), "pos:", int(y.sum()))

    qlon = m["Long"].median()
    qlat = m["Lat"].median()
    corner = np.where(m["Long"] < qlon,
                      np.where(m["Lat"] < qlat, "SW", "NW"),
                      np.where(m["Lat"] < qlat, "SE", "NE"))
    m["corner"] = corner
    print(m.groupby("corner")["Class"].agg(
        n="size", pos=lambda s: int((s == UNHEALTHY).sum())).to_string())

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    rng = np.random.default_rng(3)
    rows = []
    for c in ["SW", "NW", "SE", "NE"]:
        te = np.where(m["corner"] != c)[0]   # rest of field
        tr_c = np.where(m["corner"] == c)[0]  # this corner
        for sname, cols in SETS.items():
            X = m[cols].to_numpy(dtype=float)
            clf.fit(X[tr_c], y[tr_c])
            pr, roc = metrics(y[te], clf.decision_function(X[te]))
            rows.append(dict(train=f"corner-{c}", set=sname, pr=pr,
                             roc=roc, n_train=len(tr_c),
                             pos_train=int(y[tr_c].sum())))
            # size-matched random controls (3 seeds)
            prs, rocs = [], []
            for _ in range(3):
                perm = rng.permutation(len(m))
                tr_r = perm[:len(tr_c)]
                te_r = perm[len(tr_c):]
                clf.fit(X[tr_r], y[tr_r])
                p, r = metrics(y[te_r], clf.decision_function(X[te_r]))
                prs.append(p)
                rocs.append(r)
            rows.append(dict(train=f"random~{c}", set=sname,
                             pr=round(float(np.mean(prs)), 4),
                             roc=round(float(np.mean(rocs)), 4),
                             n_train=len(tr_c), pos_train=-1))
    tab = pd.DataFrame(rows)
    print(tab.to_string())
    tab.to_csv(OUT / "stage7_table.csv", index=False)
    (OUT / "stage7_summary.json").write_text(json.dumps(
        {"rows": rows,
         "pos_rate": round(float(y.mean()), 4)}, indent=2))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(4)
    w = 0.35
    corners = ["SW", "NW", "SE", "NE"]
    c_pr = [tab[(tab.train == f"corner-{c}") &
                (tab.set == "+rededge_SWIR")]["pr"].iloc[0]
            for c in corners]
    r_pr = [tab[(tab.train == f"random~{c}") &
                (tab.set == "+rededge_SWIR")]["pr"].iloc[0]
            for c in corners]
    ax.bar(x - w / 2, c_pr, w, label="train: one corner")
    ax.bar(x + w / 2, r_pr, w, label="train: random same-size")
    ax.axhline(float(y.mean()), color="k", ls="--", lw=1,
               label="no-skill")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}" for c in corners])
    ax.set_ylabel("PR-AUC on rest of field")
    ax.set_title("Corner deployment (+rededge_SWIR LogReg): spatial gap "
                 "vs small-data control")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_corner.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()
