"""Stage 3 checks for the POC time-series (AirHitam only).

Check A - anomaly-only features with regularisation (spatial-block CV).
  Fixed feature families (chosen from stage 2, so this is a robustness check
  of those families, not a new search):
    S1_anom_all   Sentinel-1 anomaly features, both orbits
    S1_anom_desc  Sentinel-1 anomaly features, descending only
    S2_anom       Sentinel-2 NDVI/NDRE anomaly features
    S1_desc_all   all descending Sentinel-1 temporal features
  Each family is scored alone and with ref_ndvi, at LogReg C in {0.01, 0.1, 1}.
  Same folds as stage 2: StratifiedGroupKFold(5) on KMeans(15) blocks, 10 seeds.

Check B - corner deployment with temporal features (stage-7 protocol).
  Train on one estate quadrant (median split), predict the rest, and compare
  with 3 size-matched random training sets. Feature sets are the same fixed
  families as check A, with and without ref_ndvi. C in {1.0, 0.1}. C = 1.0
  matches stage 7.

Both checks use train-fold median imputation (no test data in imputation).

Outputs (misc/POC_Results/TimeSeries/):
  stage3_spatial_checks.csv   one row per source/window/set/C
  stage3_corner.csv           one row per source/window/set/C/corner
  stage3_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TS = ROOT / "misc" / "POC_Results" / "TimeSeries"
sys.path.insert(0, str(HERE))
import poc_timeseries_stage1_eval as s1e  # noqa: E402

SEEDS = list(range(10))
SOURCES = {
    "pt10m": ("ts_s2.csv", "ts_s1.csv"),
    "box3x3": ("ts_s2_box3x3.csv", "ts_s1_box3x3.csv"),
}
WINDOWS = {
    "survey_12m": ("2025-06-10", "2026-06-11"),
    "survey_6m": ("2025-12-10", "2026-06-11"),
    "matched_12m": ("2025-07-16", "2026-07-15"),
}
SPATIAL_C = [0.01, 0.1, 1.0]
CORNER_C = [1.0, 0.1]
CORNER_WINDOWS = ["survey_12m", "matched_12m"]
REF = "ref_ndvi"


def families(cols):
    fam = {
        "S1_anom_all": [c for c in cols if c.startswith("s1") and "anom" in c],
        "S1_anom_desc": [c for c in cols if c.startswith("s1desc") and "anom" in c],
        "S2_anom": [c for c in cols if c.startswith("s2") and "anom" in c],
        "S1_desc_all": [c for c in cols if c.startswith("s1desc")],
    }
    return {k: v for k, v in fam.items() if v}


def fit_score(Xtr, ytr, Xte, C):
    med = Xtr.median()
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=5000,
                                           class_weight="balanced"))
    clf.fit(Xtr.fillna(med).fillna(0).to_numpy(), ytr)
    return clf.decision_function(Xte.fillna(med).fillna(0).to_numpy())


def load(src, window):
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    s2f, s1f = SOURCES[src]
    s2 = pd.read_csv(TS / s2f)
    s1 = pd.read_csv(TS / s1f)
    feats = s1e.build(labels, s2, s1, *window)
    feats = feats.drop(columns=feats.columns[feats.isna().mean() > 0.5])
    feats = feats.reset_index(drop=True)
    ref = pd.read_csv(ROOT / "misc/POC_Results/RVI_NDVI/airhitam_ndvi.csv")
    feats[REF] = ref.set_index("id")["NDVI"].reindex(labels["id"]).to_numpy()
    assert not feats[REF].isna().any()
    y = (labels["Class"] == "Unhealthy").astype(int).to_numpy()
    return labels, feats, y


def spatial_cells(X, y, blocks_by_seed, C):
    """Seed-fold PR cells plus per-seed pooled ROC for one configuration."""
    cells, rocs = [], []
    for seed, blocks in blocks_by_seed.items():
        oof = np.zeros(len(y))
        sgkf = StratifiedGroupKFold(n_splits=5)
        for k, (tr, te) in enumerate(sgkf.split(X, y, blocks)):
            s = fit_score(X.iloc[tr], y[tr], X.iloc[te], C)
            oof[te] = s
            cells.append(dict(seed=seed, fold=k,
                              pr=average_precision_score(y[te], s)))
        rocs.append(roc_auc_score(y, oof))
    return pd.DataFrame(cells), float(np.mean(rocs))


def check_a(src, wname, labels, feats, y, blocks_by_seed, rows):
    fam = families(list(feats.columns))
    configs = [("NDVI_only", [REF])]
    for name, cols in fam.items():
        configs.append((name, cols))
        configs.append((name + "+NDVI", cols + [REF]))
    for C in SPATIAL_C:
        ref_cells, ref_roc = spatial_cells(feats[[REF]], y, blocks_by_seed, C)
        for name, cols in configs:
            cells, roc = spatial_cells(feats[cols], y, blocks_by_seed, C)
            diff = cells["pr"].to_numpy() - ref_cells["pr"].to_numpy()
            rows.append(dict(
                source=src, window=wname, C=C, set=name, n_feats=len(cols),
                fold_pr=round(float(cells["pr"].mean()), 4),
                ref_fold_pr=round(float(ref_cells["pr"].mean()), 4),
                diff_fold_pr=round(float(diff.mean()), 4),
                share_cells_better=round(float((diff > 0).mean()), 3),
                pooled_roc=round(roc, 4),
                ref_pooled_roc=round(ref_roc, 4),
            ))
            print(f"A {src:7s} {wname:12s} C={C:<5} {name:18s} "
                  f"PR {rows[-1]['fold_pr']:.4f} "
                  f"(diff {rows[-1]['diff_fold_pr']:+.4f}, "
                  f"better {rows[-1]['share_cells_better']:.0%}) "
                  f"ROC {rows[-1]['pooled_roc']:.4f}")


def corner_labels(labels):
    qlon, qlat = labels["Long"].median(), labels["Lat"].median()
    return np.where(labels["Long"] < qlon,
                    np.where(labels["Lat"] < qlat, "SW", "NW"),
                    np.where(labels["Lat"] < qlat, "SE", "NE"))


def metrics(y_true, score):
    return (round(float(average_precision_score(y_true, score)), 4),
            round(float(roc_auc_score(y_true, score)), 4))


def check_b(src, wname, labels, feats, y, rows):
    corner = corner_labels(labels)
    fam = families(list(feats.columns))
    configs = [("NDVI_only", [REF])]
    for name, cols in fam.items():
        configs.append((name, cols))
        configs.append((name + "+NDVI", cols + [REF]))
    rng = np.random.default_rng(3)
    for C in CORNER_C:
        for name, cols in configs:
            X = feats[cols]
            for c in ["SW", "NW", "SE", "NE"]:
                tr = np.where(corner == c)[0]
                te = np.where(corner != c)[0]
                pr, roc = metrics(y[te], fit_score(X.iloc[tr], y[tr],
                                                   X.iloc[te], C))
                rp, rr = [], []
                for _ in range(3):
                    perm = rng.permutation(len(y))
                    tr_r, te_r = perm[:len(tr)], perm[len(tr):]
                    p, r = metrics(y[te_r], fit_score(X.iloc[tr_r], y[tr_r],
                                                      X.iloc[te_r], C))
                    rp.append(p)
                    rr.append(r)
                rows.append(dict(
                    source=src, window=wname, C=C, set=name, corner=c,
                    n_train=len(tr), pos_train=int(y[tr].sum()),
                    corner_pr=pr, corner_roc=roc,
                    random_pr=round(float(np.mean(rp)), 4),
                    random_roc=round(float(np.mean(rr)), 4)))
            print(f"B {src:7s} {wname:12s} C={C:<5} {name:18s} "
                  + "  ".join(f"{r['corner']}:{r['corner_pr']:.3f}"
                              f"/{r['random_pr']:.3f}"
                              for r in rows[-4:]))


def main():
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    coords = labels[["Long", "Lat"]].to_numpy()
    blocks_by_seed = {s: KMeans(n_clusters=15, n_init=10,
                                random_state=s).fit_predict(coords)
                      for s in SEEDS}
    rows_a, rows_b = [], []
    for src in SOURCES:
        for wname, window in WINDOWS.items():
            labels, feats, y = load(src, window)
            check_a(src, wname, labels, feats, y, blocks_by_seed, rows_a)
            if wname in CORNER_WINDOWS:
                check_b(src, wname, labels, feats, y, rows_b)
    A = pd.DataFrame(rows_a)
    B = pd.DataFrame(rows_b)
    A.to_csv(TS / "stage3_spatial_checks.csv", index=False)
    B.to_csv(TS / "stage3_corner.csv", index=False)
    summary = dict(
        spatial_best_by_source_window=(
            A.sort_values("fold_pr", ascending=False)
             .groupby(["source", "window"]).head(1)
             .to_dict(orient="records")),
        corner_mean=(B.groupby(["source", "window", "C", "set"])
                     [["corner_pr", "random_pr"]].mean().round(4)
                     .reset_index().to_dict(orient="records")),
    )
    (TS / "stage3_summary.json").write_text(json.dumps(summary, indent=2,
                                                       default=str))
    print("\n== A: spatial ==\n" + A.to_string(index=False))
    print("\n== B: corner (mean over 4 corners) ==")
    print(B.groupby(["source", "window", "C", "set"])
          [["corner_pr", "random_pr", "corner_roc"]].mean().round(4)
          .to_string())


if __name__ == "__main__":
    main()
