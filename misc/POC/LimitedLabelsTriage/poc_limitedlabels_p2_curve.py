"""P2 LimitedLabelsTriage: learning-curve contextualisation + ceilings.
Loads P1 summary; adds (i) oracle 80/20 random (upper bound), (ii) spatial
KMeans-15 group 5-fold OOF (honest full-coverage), (iii) quadrant-train
continuity (train quadrant -> rest, NDVI-only LogReg). Emits p2_table.csv +
fig_p2_curve.png. No GEE, no resampling.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "LimitedLabelsTriage"

def recall_at_k(y_true, score, frac):
    order = np.argsort(-score)
    k = max(1, int(round(frac * len(y_true))))
    flagged = order[:k]
    pt = int(y_true.sum())
    return float(y_true[flagged].sum() / pt) if pt else float("nan")

def main():
    df = pd.read_csv(OUT / "tables" / "frame_frozen.csv")
    y = (df["Class"] == "Unhealthy").to_numpy()
    X = df[["NDVI"]].to_numpy()
    p1 = pd.read_csv(OUT / "tables" / "p1_summary.csv")
    rows = []
    # (i) oracle: 80% random train -> 20% test, 10 reps
    for r in range(10):
        rng = np.random.default_rng(9000 + r)
        idx = rng.permutation(len(df))
        ntr = int(0.8 * len(df))
        tr, te = idx[:ntr], idx[ntr:]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        rows.append({"protocol": "oracle_80rand", "rep": r, "n_train": ntr,
                     "pos_train": int(y[tr].sum()),
                     "pr": round(float(average_precision_score(y[te], s)), 4),
                     "roc": round(float(roc_auc_score(y[te], s)), 4),
                     "r25": round(recall_at_k(y[te], s, 0.25), 4)})
    # (ii) spatial KMeans-15 groups, 5-fold OOF x3 seeds (honest full coverage)
    for s in range(3):
        km = KMeans(n_clusters=15, n_init=10, random_state=400 + s).fit(
            np.column_stack([df["Long"].to_numpy(), df["Lat"].to_numpy()]))
        groups = km.labels_
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=500 + s)
        oof = np.full(len(df), np.nan)
        for tri, tei in skf.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tri], y[tri])
            oof[tei] = clf.predict_proba(X[tei])[:, 1]
        rows.append({"protocol": "spatial_K15_OOF", "rep": s, "n_train": int((~np.isnan(oof)).sum() * 0.8),
                     "pos_train": -1,
                     "pr": round(float(average_precision_score(y, oof)), 4),
                     "roc": round(float(roc_auc_score(y, oof)), 4),
                     "r25": round(recall_at_k(y, oof, 0.25), 4)})
    # (iii) quadrant continuity: train quadrant -> rest
    mlo, mla = float(df["Long"].median()), float(df["Lat"].median())
    for q, mk in {"SW": (df["Long"] <= mlo) & (df["Lat"] <= mla),
                  "NW": (df["Long"] <= mlo) & (df["Lat"] > mla),
                  "SE": (df["Long"] > mlo) & (df["Lat"] <= mla),
                  "NE": (df["Long"] > mlo) & (df["Lat"] > mla)}.items():
        tri = mk.to_numpy(); tei = ~tri
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tri], y[tri])
        s = clf.predict_proba(X[tei])[:, 1]
        rows.append({"protocol": f"quadrant_{q}", "rep": q, "n_train": int(tri.sum()),
                     "pos_train": int(y[tri].sum()),
                     "pr": round(float(average_precision_score(y[tei], s)), 4),
                     "roc": round(float(roc_auc_score(y[tei], s)), 4),
                     "r25": round(recall_at_k(y[tei], s, 0.25), 4)})
    p2 = pd.DataFrame(rows)
    p2.to_csv(OUT / "tables" / "p2_table.csv", index=False)
    print(p2.to_string())
    print(p2.groupby("protocol")[["pr", "roc", "r25"]].mean().to_string())
    # figure: P1 E_random curve + ceilings
    import seaborn as sns
    sns.set_style("whitegrid")
    e = p1[p1.design == "E_random"].sort_values("B")
    a = p1[p1.design == "A_block"].sort_values("B")
    orc = p2[p2.protocol == "oracle_80rand"]["pr"].mean()
    oof = p2[p2.protocol == "spatial_K15_OOF"]["pr"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(e["B"], e["pr_mean"], yerr=e["pr_ci95"], marker="o", capsize=3, label="E_random (control)")
    ax.errorbar(a["B"], a["pr_mean"], yerr=a["pr_ci95"], marker="s", capsize=3, label="A_block (product)")
    ax.axhline(orc, color="green", ls="--", label=f"oracle 80/20 mean PR {orc:.3f}")
    ax.axhline(oof, color="purple", ls=":", label=f"spatial-K15 OOF mean PR {oof:.3f}")
    ax.axhline(160 / 2511, color="k", ls="-", lw=1, label="no-skill 0.064")
    ax.set_xlabel("B (labelled trees)"); ax.set_ylabel("PR-AUC")
    ax.set_title("Learning curve is flat: budget 1000->1600 buys nothing (NDVI-LogReg)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_p2_curve.png", dpi=130)
    print("wrote p2_table.csv + fig_p2_curve.png")

if __name__ == "__main__":
    main()
