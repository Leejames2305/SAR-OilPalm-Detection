"""Stage 5 of POC RVI+NDVI (AirHitam only). Block-size sensitivity sweep.

Same LogReg, same feature sets, five evaluation granularities:
  random (RepeatedStratified 5x3, leakage ceiling) | K100 (~25 trees/block)
  | K50 (~50) | K15 (~167, current default) | K5 (~500).
Asks: does the NDVI/red-edge edge survive fine blocks? Does SAR-only
inflate as blocks shrink (leakage signature)?

Outputs (misc/POC_Results/RVI_NDVI/):
  stage5_table.csv, stage5_summary.json, fig_block_gradient.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)
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
    "+SARlvl": ["NDVI", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI",
                "HH", "HV", "VH", "VV", "span"],
    "+RVI": ["NDVI", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI",
             "HH", "HV", "VH", "VV", "span",
             "RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI",
             "RVI_qp"],
    "ALL_SAR": ["HH", "HV", "VH", "VV", "span",
                "RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI",
                "RVI_qp"],
}
BLOCKS = [("random", None), ("K100", 100), ("K50", 50), ("K15", 15),
          ("K5", 5)]
MX = 111320 * np.cos(np.radians(2.947))  # metres per deg lon at AirHitam
MY = 111320  # metres per deg lat


def block_info(coords, y, grp, k):
    diams, pos = [], []
    for g in range(k):
        pts = coords[grp == g]
        c0 = pts.mean(axis=0)
        dd = np.sqrt(((pts[:, 0] - c0[0]) * MX) ** 2 +
                     ((pts[:, 1] - c0[1]) * MY) ** 2)
        diams.append(2 * float(dd.mean()))
        pos.append(int(y[grp == g].sum()))
    return (f"{len(coords) / k:.0f} trees/block, "
            f"~{float(np.median(diams)):.0f} m diameter, "
            f"pos/block median {float(np.median(pos)):.0f}")


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
    pos_rate = float(y.mean())
    coords = m[["Long", "Lat"]].to_numpy(dtype=float)
    print("rows:", len(m), "pos:", int(y.sum()))

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    rows = []
    for bname, k in BLOCKS:
        if k is None:
            splits = list(rskf.split(coords, y))
            info = "leakage ceiling (no spatial constraint)"
        else:
            grp = KMeans(n_clusters=k, n_init=10,
                         random_state=42).fit_predict(coords)
            info = block_info(coords, y, grp, k)
            splits = list(StratifiedGroupKFold(n_splits=5).split(
                coords, y, grp))
        print(bname, "-", info)
        for sname, cols in SETS.items():
            X = m[cols].to_numpy(dtype=float)
            pr, roc = [], []
            for tr, te in splits:
                clf.fit(X[tr], y[tr])
                s = clf.decision_function(X[te])
                pr.append(average_precision_score(y[te], s))
                roc.append(roc_auc_score(y[te], s))
            rows.append(dict(protocol=bname, info=info, set=sname,
                             pr=round(float(np.mean(pr)), 4),
                             roc=round(float(np.mean(roc)), 4)))
    tab = pd.DataFrame(rows)
    print(tab.to_string())
    tab.to_csv(OUT / "stage5_table.csv", index=False)
    (OUT / "stage5_summary.json").write_text(json.dumps(
        {"pos_rate": round(pos_rate, 4), "rows": rows}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    order = ["random", "K100", "K50", "K15", "K5"]
    for ax, met in zip(axes, ["pr", "roc"]):
        for sname in SETS:
            sub = tab[tab.set == sname].set_index("protocol").loc[order]
            ax.plot(order, sub[met], marker="o", label=sname)
        ax.set_title("LogReg " + ("PR-AUC" if met == "pr" else "ROC-AUC") +
                     " vs block granularity")
        ax.set_xlabel("protocol (left = finer blocks, closer to random)")
        ax.legend(fontsize=8)
        if met == "pr":
            ax.axhline(pos_rate, color="k", ls="--", lw=1)
    fig.suptitle("AirHitam: leakage-gradient sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_block_gradient.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()
