"""Stage 6: pocket anatomy + memorization-vs-characteristics tests (AirHitam).

Settles the block-size argument empirically:
 A. Pocket anatomy: DBSCAN pocket sizes for Unhealthy trees; nearest-
    Unhealthy distances. Is disease in tiny scattered pockets (map claim)?
 B. Distance-stratified recall: random 80/20 LogReg, test-Unhealthy recall
    binned by distance to nearest TRAIN Unhealthy. Flat = learned
    characteristics; concentrated near train positives = neighbourhood
    memorization.
 C. Pocket-holdout: hold out whole DBSCAN pockets (never-seen pockets) vs
    random-split recall. Can the model detect a pocket it never saw?

Outputs: stage6_summary.json, fig_pockets.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"
MX = 111320 * np.cos(np.radians(2.947))
MY = 111320
FEATS = ["NDVI", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI",
         "HH", "HV", "VH", "VV", "span",
         "RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI", "RVI_qp"]


def load_frame():
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
    return m.dropna().reset_index(drop=True)


def main():
    m = load_frame()
    y = (m["Class"] == UNHEALTHY).to_numpy()
    xy = np.column_stack([m["Long"].to_numpy() * MX,
                          m["Lat"].to_numpy() * MY])
    X = m[FEATS].to_numpy(dtype=float)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced"))
    res = {}

    # ---- A. pocket anatomy (Unhealthy only, eps=30 m) ----
    upts = xy[y]
    duu = cdist(upts, upts)
    np.fill_diagonal(duu, np.inf)
    nn_u = duu.min(axis=1)
    lab = DBSCAN(eps=30, min_samples=2).fit(upts).labels_
    sizes = pd.Series(lab).value_counts()
    n_single = int((lab == -1).sum())
    n_clust = len(set(lab.tolist()) - {-1})
    res["A_anatomy"] = dict(
        n_unhealthy=int(y.sum()),
        n_singles_30m=n_single,
        n_clusters=n_clust,
        cluster_sizes=sorted([int(s) for s in
                              sizes[sizes.index != -1]], reverse=True),
        nn_unhealthy_median=round(float(np.median(nn_u)), 1),
        nn_unhealthy_mean=round(float(nn_u.mean()), 1),
        frac_nn_within_30m=round(float((nn_u < 30).mean()), 3))
    print(json.dumps(res["A_anatomy"], indent=1))

    # pocket id per tree: cluster label of unhealthy, else -1; spread to
    # nearby healthy (<30 m of a clustered unhealthy) for pocket-holdout
    pocket = np.full(len(m), -1)
    pocket[y] = lab
    d_healthy_to_clust = cdist(xy[~y], upts[lab != -1]).min(axis=1) \
        if n_clust else np.full((~y).sum(), np.inf)
    # map each healthy to nearest clustered-unhealthy pocket
    if n_clust:
        cu = upts[lab != -1]
        cl = lab[lab != -1]
        nearest = cdist(xy[~y], cu).argmin(axis=1)
        near = cdist(xy[~y], cu).min(axis=1) < 30
        pocket[np.where(~y)[0][near]] = cl[nearest[near]] + 1000

    # ---- B. distance-stratified recall (10x random 80/20) ----
    rng = np.random.default_rng(7)
    bins = [(0, 25), (25, 50), (50, 100), (100, 1e9)]
    hits = {b: [0, 0] for b in bins}  # [found, total]
    for _ in range(10):
        perm = rng.permutation(len(m))
        te = perm[:len(m) // 5]
        tr = perm[len(m) // 5:]
        clf.fit(X[tr], y[tr])
        s = clf.decision_function(X[te])
        thr = np.quantile(s, 0.75)  # top-25% review budget rule
        pred = s >= thr
        d_to_pos = cdist(xy[te], xy[tr[y[tr]]]).min(axis=1)
        for i, t in enumerate(te):
            if y[t]:
                for b in bins:
                    if b[0] <= d_to_pos[i] < b[1]:
                        hits[b][1] += 1
                        hits[b][0] += int(pred[i])
    res["B_dist_recall"] = {
        f"{b[0]}-{b[1] if b[1] < 1e9 else 'inf'}m":
        dict(n=int(v[1]), recall=round(v[0] / max(1, v[1]), 3))
        for b, v in hits.items()}
    print(json.dumps(res["B_dist_recall"], indent=1))

    # ---- C. pocket-holdout vs random recall ----
    # groups = pockets (clustered unhealthy + their <30 m halo share an id;
    # isolated trees each form a singleton group via their index)
    grp = pocket.copy()
    grp[(grp == -1)] = np.arange(100000,
                                 100000 + int((grp == -1).sum()))
    uniq = np.unique(grp[y])
    rng2 = np.random.default_rng(11)
    rec_hold, rec_rand = [], []
    for _ in range(10):
        # pocket holdout: whole pockets as test
        order = rng2.permutation(uniq)
        te_g = order[:len(uniq) // 5]
        te = np.where(np.isin(grp, te_g))[0]
        tr = np.where(~np.isin(grp, te_g))[0]
        clf.fit(X[tr], y[tr])
        s = clf.decision_function(X[te])
        pred = s >= np.quantile(s, 0.75)
        if y[te].sum():
            rec_hold.append(float(pred[y[te]].mean()))
        # matched random split of same test size
        perm = rng2.permutation(len(m))
        te2 = perm[:len(te)]
        tr2 = perm[len(te):]
        clf.fit(X[tr2], y[tr2])
        s2 = clf.decision_function(X[te2])
        pred2 = s2 >= np.quantile(s2, 0.75)
        if y[te2].sum():
            rec_rand.append(float(pred2[y[te2]].mean()))
    res["C_pocket_holdout"] = dict(
        n_pockets=int(len(uniq)),
        recall_holdout=round(float(np.mean(rec_hold)), 3),
        recall_random=round(float(np.mean(rec_rand)), 3))
    print(json.dumps(res["C_pocket_holdout"], indent=1))

    (OUT / "stage6_summary.json").write_text(json.dumps(res, indent=2))

    # ---- figure: estate map coloured by pocket + recall bars ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(m["Long"], m["Lat"], s=6, c="#bbbbbb", label="Healthy")
    cm = plt.get_cmap("tab20")
    for i, g in enumerate(sorted(set(lab) - {-1})):
        pts = upts[lab == g]
        ax.scatter(pts[:, 0] / MX, pts[:, 1] / MY, s=28, color=cm(i),
                   edgecolor="k", lw=0.5)
    sg = upts[lab == -1]
    if len(sg):
        ax.scatter(sg[:, 0] / MX, sg[:, 1] / MY, s=28, c="red",
                   marker="x", label="isolated Unhealthy")
    ax.set_title(f"AirHitam pockets: {n_clust} clusters + "
                 f"{n_single} isolated (30 m rule)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax = axes[1]
    bnames = list(res["B_dist_recall"])
    ax.bar(bnames, [res["B_dist_recall"][b]["recall"] for b in bnames],
           color="#d62728", alpha=0.7)
    ax.axhline(res["C_pocket_holdout"]["recall_random"], color="k",
               ls="--", label="random-split recall")
    ax.set_title("Test-Unhealthy recall by dist. to nearest TRAIN "
                 "Unhealthy")
    ax.set_ylabel("recall @ top-25% rule")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pockets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()


