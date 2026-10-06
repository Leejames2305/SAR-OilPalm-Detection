"""P4 LimitedLabelsTriage: confusion matrices for the deployment baseline.
Model: balanced LogReg, NDVI-only. Rule: top-25% of test-rest scores flagged
for review (triage operating point) + p=0.5 reference to show why the raw
threshold misleads at 6.4% prevalence. Masks: A_block corners + E_random reps
at B=1200. Outputs: tables/p4_confusion.csv, figs/fig_p4_confusion.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "LimitedLabelsTriage"
B = 1200

def cm_at_rule(y_true, score, frac=0.25):
    order = np.argsort(-score)
    k = max(1, int(round(frac * len(y_true))))
    pred = np.zeros_like(y_true); pred[order[:k]] = 1
    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn, k

def main():
    df = pd.read_csv(OUT / "tables" / "frame_frozen.csv")
    y = (df["Class"] == "Unhealthy").to_numpy()
    X = df[["NDVI"]].to_numpy()
    MX = 111320 * np.cos(np.radians(df["Lat"].mean()))
    xm = df["Long"].to_numpy() * MX
    ym = df["Lat"].to_numpy() * 111320
    N = len(df)
    corners = {"SW": (xm.min(), ym.min()), "NW": (xm.min(), ym.max()),
               "SE": (xm.max(), ym.min()), "NE": (xm.max(), ym.max())}
    masks = {}
    for c, (sx, sy) in corners.items():
        d = np.sqrt((xm - sx) ** 2 + (ym - sy) ** 2)
        mk = np.zeros(N, bool); mk[np.argsort(d)[:B]] = True
        masks[f"A_{c}"] = mk
    for r in range(30):
        rng = np.random.default_rng(7000 + B * 100 + 100 + r)
        mk = np.zeros(N, bool); mk[rng.choice(N, B, replace=False)] = True
        masks[f"E_{r}"] = mk
    rows = []
    for key, mk in masks.items():
        tr, te = mk, ~mk
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        tp, fp, fn, tn, k = cm_at_rule(y[te], s, 0.25)
        # p=0.5 reference
        p = (s >= 0.5).astype(int)
        tp5 = int(((p == 1) & (y[te] == 1)).sum()); fp5 = int(((p == 1) & (y[te] == 0)).sum())
        fn5 = int(((p == 0) & (y[te] == 1)).sum()); tn5 = int(((p == 0) & (y[te] == 0)).sum())
        rows.append({"mask": key, "family": key[0], "n_test": int(te.sum()), "pos_test": int(y[te].sum()),
                     "reviewed": k, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
                     "TP05": tp5, "FP05": fp5, "FN05": fn5, "TN05": tn5,
                     "prec05": round(tp5 / max(tp5 + fp5, 1), 4), "rec05": round(tp5 / max(tp5 + fn5, 1), 4)})
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "tables" / "p4_confusion.csv", index=False)
    agg = c.groupby("family")[["TP", "FP", "FN", "TN", "precision", "recall", "TP05", "FP05", "FN05", "TN05", "prec05", "rec05"]].mean().round(2)
    print("=== mean confusion @top-25% + @p0.5 ===")
    print(agg.to_string())
    print("\n=== per-corner @top-25% ===")
    print(c[c.family == "A"][["mask", "n_test", "pos_test", "reviewed", "TP", "FP", "FN", "TN", "precision", "recall"]].to_string(index=False))
    # figure: mean matrices as heatmaps (A corners mean vs E random mean)
    import seaborn as sns
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for i, fam in enumerate(["A", "E"]):
        sub = c[c.family == fam]
        M = np.array([[sub.TN.mean(), sub.FP.mean()], [sub.FN.mean(), sub.TP.mean()]])
        sns.heatmap(M, annot=True, fmt=".0f", cmap="Blues", ax=ax[i],
                    xticklabels=["Pred Healthy", "Pred Review"], yticklabels=["True Healthy", "True Unhealthy"])
        ax[i].set_title(f"{'Corner-block (4)' if fam == 'A' else 'Random (30)'} mean CM @top-25%, B=1200\n"
                        f"recall {sub.recall.mean():.2f} | precision {sub.precision.mean():.2f}")
    fig.suptitle("Deployment baseline (NDVI-only LogReg): confusion on never-visited rest")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_p4_confusion.png", dpi=130)
    print("wrote p4_confusion.csv + fig_p4_confusion.png")

if __name__ == "__main__":
    main()
