"""P3 LimitedLabelsTriage: model check at deployment budgets.
Question: with B in {1000,1600} under product (A_block) + control (E_random)
masks, does anything factually reasonable beat NDVI-only balanced LogReg?
Configs: logreg_NDVI | logreg_spec(6 feats) | pca2_logreg | pca3_logreg |
qda_ndvi(1D) | qda_2d(NDVI+NDRE5). PCA motivated by correlated indices;
QDA restricted to <=2 feats (covariance needs positives; 6D QDA would collapse
at ~64-103 positives). Outputs: tables/p3_table.csv, figs/fig_p3_model.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "LimitedLabelsTriage"
SPEC6 = ["NDVI", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI"]

def recall_at_k(y_true, score, frac):
    order = np.argsort(-score)
    k = max(1, int(round(frac * len(y_true))))
    flagged = order[:k]
    pt = int(y_true.sum())
    return float(y_true[flagged].sum() / pt) if pt else float("nan")

def make_clf(name):
    if name == "logreg_NDVI":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), ["NDVI"]
    if name == "logreg_spec":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), SPEC6
    if name == "pca2_logreg":
        return make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0),
                             LogisticRegression(max_iter=2000, class_weight="balanced")), SPEC6
    if name == "pca3_logreg":
        return make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0),
                             LogisticRegression(max_iter=2000, class_weight="balanced")), SPEC6
    if name == "qda_ndvi":
        return QuadraticDiscriminantAnalysis(reg_param=0.05), ["NDVI"]
    if name == "qda_2d":
        return QuadraticDiscriminantAnalysis(reg_param=0.1), ["NDVI", "NDRE5"]
    raise ValueError(name)

def main():
    df = pd.read_csv(OUT / "tables" / "frame_frozen.csv")
    y = (df["Class"] == "Unhealthy").to_numpy()
    MX = 111320 * np.cos(np.radians(df["Lat"].mean()))
    xm = df["Long"].to_numpy() * MX
    ym = df["Lat"].to_numpy() * 111320
    N = len(df)
    names = ["logreg_NDVI", "logreg_spec", "pca2_logreg", "pca3_logreg", "qda_ndvi", "qda_2d"]
    rows = []
    for B in [1000, 1600]:
        # product masks: 4 corner blocks (B nearest to extreme corners)
        corners = {"SW": (xm.min(), ym.min()), "NW": (xm.min(), ym.max()),
                   "SE": (xm.max(), ym.min()), "NE": (xm.max(), ym.max())}
        masks = {}
        for c, (sx, sy) in corners.items():
            d = np.sqrt((xm - sx) ** 2 + (ym - sy) ** 2)
            mk = np.zeros(N, bool); mk[np.argsort(d)[:B]] = True
            masks[f"A_{c}"] = ("A_block", mk)
        for r in range(12):
            rng = np.random.default_rng(7000 + (B * 100) + r)
            mk = np.zeros(N, bool); mk[rng.choice(N, B, replace=False)] = True
            masks[f"E_{r}"] = ("E_random", mk)
        for mkey, (design, mk) in masks.items():
            tr, te = mk, ~mk
            for name in names:
                clf, feats = make_clf(name)
                try:
                    clf.fit(df.loc[tr, feats].to_numpy(), y[tr])
                    s = clf.predict_proba(df.loc[te, feats].to_numpy())[:, 1]
                    pr = float(average_precision_score(y[te], s))
                    roc = float(roc_auc_score(y[te], s))
                    r25 = recall_at_k(y[te], s, 0.25)
                except Exception as e:
                    pr, roc, r25 = float("nan"), float("nan"), float("nan")
                rows.append({"B": B, "design": design, "mask": mkey, "model": name,
                             "pos_train": int(y[tr].sum()),
                             "pr": round(pr, 4) if pr == pr else None,
                             "roc": round(roc, 4) if roc == roc else None,
                             "r25": round(r25, 4) if r25 == r25 else None})
        print(f"B={B} done")
    p3 = pd.DataFrame(rows)
    p3.to_csv(OUT / "tables" / "p3_table.csv", index=False)
    s = p3.groupby(["B", "design", "model"]).agg(n=("pr", "size"), pr_mean=("pr", "mean"),
        pr_sd=("pr", "std"), r25_mean=("r25", "mean")).reset_index()
    s.to_csv(OUT / "tables" / "p3_summary.csv", index=False)
    print(s.to_string())
    import seaborn as sns
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, B in enumerate([1000, 1600]):
        sub = s[s.B == B]
        x = np.arange(len(names))
        w = 0.35
        for j, d in enumerate(["A_block", "E_random"]):
            v = sub[sub.design == d].set_index("model").reindex(names)
            ax[i].bar(x + (j - 0.5) * w, v["pr_mean"], w, yerr=v["pr_sd"], capsize=3, label=d)
        ax[i].set_xticks(x); ax[i].set_xticklabels([n.replace("logreg_", "LR-").replace("_logreg", "+LR") for n in names], rotation=20, ha="right", fontsize=8)
        ax[i].axhline(160 / 2511, color="k", ls="--", lw=1)
        ax[i].set_title(f"B={B} spatial PR by model"); ax[i].legend(fontsize=8)
    ax[0].set_ylabel("PR-AUC (test rest)")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_p3_model.png", dpi=130)
    print("wrote p3_table.csv, p3_summary.csv, fig_p3_model.png")

if __name__ == "__main__":
    main()
