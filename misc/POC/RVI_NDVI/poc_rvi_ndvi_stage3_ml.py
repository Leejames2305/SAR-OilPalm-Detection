"""Stage 3 of POC RVI+NDVI (AirHitam only). ML probe beyond univariate stats:
can any classifier push S2-NDVI (base) + SAR/RVI (supplement) past the
univariate NDVI operating point (~0.13 spatial PR)?

Feature sets (strictly spectral - no coordinates, per transferability requirement):
Models (fixed configs, no tuning - probe, not benchmark):
  LogReg(balanced) | RF(400) | XGB(shallow, scale_pos_weight) |
  MLP(32, early-stop)
Protocols: repeated stratified 5-fold x3 (leakage ceiling) + KMeans-15
  spatial-block 5-fold (honest). Metrics: PR-AUC, ROC-AUC, recall@top10/25%
  review budgets + enrichment. No-skill PR = 0.0637.

Outputs (misc/POC_Results/RVI_NDVI/):
  stage3_summary.json, stage3_table.csv,
  fig_ml_spatial_pr.png, fig_ml_recall_budget.png
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"
POS_RATE = 160 / 2511

NDVI_F = ["NDVI", "B4", "B8"]
SARLVL_F = ["HH", "HV", "VH", "VV", "span", "lam1", "lam2", "lam3"]
RVI_F = ["RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI", "RVI_qp"]
SETS = {
    "NDVI_only": ["NDVI"],
    "NDVI+B": NDVI_F,
    "+SARlvl": NDVI_F + SARLVL_F,
    "+RVI": NDVI_F + SARLVL_F + RVI_F,
    "ALL_SAR": SARLVL_F + RVI_F,
}
MODELS = {
    "logreg": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight="balanced")),
    "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=42),
    "xgb": XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=(2351 / 160), n_jobs=-1,
                         random_state=42, eval_metric="logloss"),
    "mlp": make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(32,), alpha=0.01, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, max_iter=2000,
        random_state=42)),
}


def recall_at_budget(y_true, score, frac):
    k = max(1, int(len(y_true) * frac))
    top = np.argsort(score)[-k:]
    return float(y_true[top].sum() / max(1, y_true.sum()))


def main():
    rvi = pd.read_csv(OUT / "airhitam_rvi_features.csv")
    ndvi = pd.read_csv(OUT / "airhitam_ndvi.csv")
    qp = pd.read_csv(OUT / "airhitam_rvi_quadpol.csv")
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    m = (rvi.merge(ndvi, on=["id", "Class"], how="inner")
            .merge(qp[["id", "RVI_qp", "span", "lam1", "lam2", "lam3"]],
                   on="id", how="left")
            .merge(labels[["id", "Long", "Lat"]], on="id", how="left"))
    m = m.dropna(subset=["NDVI"]).reset_index(drop=True)
    y = (m["Class"] == UNHEALTHY).to_numpy()
    print("rows:", len(m), "pos:", int(y.sum()))
    blocks = KMeans(n_clusters=15, n_init=10,
                    random_state=42).fit_predict(
                        m[["Long", "Lat"]].to_numpy())
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    sgkf = StratifiedGroupKFold(n_splits=5)

    rows = []
    oof = {}  # (model, set) -> spatial-OOF scores for budget curves
    for (sname, cols), (mname, clf) in itertools.product(SETS.items(),
                                                         MODELS.items()):
        X = m[cols].to_numpy(dtype=float)
        pr_r, roc_r, pr_s, roc_s, rc10_s, rc25_s = [], [], [], [], [], []
        oof_s = np.zeros(len(y))
        for tr, te in rskf.split(X, y):
            clf.fit(X[tr], y[tr])
            s = clf.decision_function(X[te]) if hasattr(clf, "decision_function") \
                else clf.predict_proba(X[te])[:, 1]
            pr_r.append(average_precision_score(y[te], s))
            roc_r.append(roc_auc_score(y[te], s))
        for tr, te in sgkf.split(X, y, blocks):
            clf.fit(X[tr], y[tr])
            s = clf.decision_function(X[te]) if hasattr(clf, "decision_function") \
                else clf.predict_proba(X[te])[:, 1]
            oof_s[te] = s
            pr_s.append(average_precision_score(y[te], s))
            roc_s.append(roc_auc_score(y[te], s))
            rc10_s.append(recall_at_budget(y[te], s, 0.10))
            rc25_s.append(recall_at_budget(y[te], s, 0.25))
        rows.append(dict(set=sname, model=mname,
                         rand_pr=round(float(np.mean(pr_r)), 4),
                         rand_roc=round(float(np.mean(roc_r)), 4),
                         spat_pr=round(float(np.mean(pr_s)), 4),
                         spat_roc=round(float(np.mean(roc_s)), 4),
                         spat_r10=round(float(np.mean(rc10_s)), 3),
                         spat_r25=round(float(np.mean(rc25_s)), 3)))
        oof[(mname, sname)] = oof_s
        print(rows[-1])
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "stage3_table.csv", index=False)
    (OUT / "stage3_summary.json").write_text(json.dumps(
        {"pos_rate": round(POS_RATE, 4), "rows": rows}, indent=2))

    # Fig 1: honest spatial PR by set/model
    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.barplot(data=tab, x="set", y="spat_pr", hue="model", ax=ax)
    ax.axhline(POS_RATE, color="k", ls="--", lw=1, label="no-skill")
    ax.set_title("AirHitam: honest spatial-block PR-AUC (5-fold, KMeans-15)")
    ax.set_ylabel("PR-AUC")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_spatial_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: recall-vs-review-budget curves (spatial OOF, best set per model)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    budgets = np.linspace(0.05, 0.60, 12)
    for mname in MODELS:
        best = tab[tab.model == mname].sort_values("spat_pr").iloc[-1]
        s = oof[(mname, best["set"])]
        rec = [recall_at_budget(y, s, b) for b in budgets]
        ax.plot(budgets * 100, rec, marker="o", ms=3,
                label=f"{mname} [{best['set']}]")
    ax.plot(budgets * 100, budgets, "k--", lw=1, label="random")
    ax.set_xlabel("review budget (% of trees)")
    ax.set_ylabel("recall (Unhealthy found)")
    ax.set_title("AirHitam: triage curves (spatial-OOF scores)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_recall_budget.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()

