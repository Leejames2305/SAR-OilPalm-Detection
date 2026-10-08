"""Stage 2 of POC time-series (AirHitam only): nested test of SAR/S2 temporal
features on top of the stage-2 NDVI reference.

Protocol per outer fold (StratifiedGroupKFold(5) on KMeans(15) spatial blocks):
  1. On the TRAINING fold only: compute |Cohen's d| of every candidate temporal
     feature (train-median imputed) and pick the top one.
  2. Fit balanced LogReg on [ref_ndvi] and on [ref_ndvi, picked feature].
  3. Score the TEST fold. Fold PR-AUC is comparable with stage 2 (REF = 0.133).

Robustness:
  - Repeated over 10 KMeans block seeds (block geometry changes; the reference
    NDVI stays fixed), so stability is measured, not one lucky split.
  - Two sources: 10 m point sample (stage 0) and 3x3 box (stage 0b).
  - Three windows: 12m to survey date (primary), 6m to survey date, and 12m
    matched to the stage-2 composite end (15 Jul 2026).
  - Candidate groups: S1 all, S1 descending, S1 ascending, S2 NDVI/NDRE.

Outputs (misc/POC_Results/TimeSeries/):
  stage2_nested_cells.csv     one row per source/window/group/seed/fold
  stage2_nested_summary.csv   one row per source/window/group
  stage2_nested_summary.json
"""
import json
import sys
from collections import Counter
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
GROUPS = ["S1_all", "S1_desc", "S1_asc", "S2_all"]


def group_cols(cols, name):
    prefix = {"S1_all": ("s1",), "S1_desc": ("s1desc",),
              "S1_asc": ("s1asc",), "S2_all": ("s2",)}[name]
    return [c for c in cols if c.startswith(prefix)]


def cohen_d(x, y):
    h, u = x[y == 0], x[y == 1]
    sd = np.sqrt((h.var(ddof=1) + u.var(ddof=1)) / 2.0)
    return (u.mean() - h.mean()) / sd if sd > 0 else 0.0


def fit_score(Xtr, ytr, Xte):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf.decision_function(Xte)


def run_seed(F, ref, cand, y, coords, seed):
    """One block seed: outer folds, in-fold selection, scores for ref and nested."""
    blocks = KMeans(n_clusters=15, n_init=10, random_state=seed).fit_predict(coords)
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_ref = np.zeros(len(y))
    oof_nes = np.zeros(len(y))
    rows = []
    for k, (tr, te) in enumerate(sgkf.split(ref.reshape(-1, 1), y, blocks)):
        s_ref = fit_score(ref[tr, None], y[tr], ref[te, None])
        oof_ref[te] = s_ref
        med = F[cand].iloc[tr].median()
        Ftr = F[cand].iloc[tr].fillna(med)
        Fte = F[cand].iloc[te].fillna(med)
        d_abs = {c: abs(cohen_d(Ftr[c].to_numpy(), y[tr])) for c in cand}
        pick = max(d_abs, key=d_abs.get)
        s_nes = fit_score(np.c_[ref[tr], Ftr[pick]], y[tr],
                          np.c_[ref[te], Fte[pick]])
        oof_nes[te] = s_nes
        rows.append(dict(seed=seed, fold=k,
                         pr_ref=average_precision_score(y[te], s_ref),
                         pr_nested=average_precision_score(y[te], s_nes),
                         pick=pick, pick_abs_d=round(d_abs[pick], 4)))
    return rows, oof_ref, oof_nes


def evaluate(F, ref, cand, y, coords):
    """All seeds for one source/window/group. Returns (summary stats, cell rows)."""
    ref_fold, nes_fold, ref_pool, nes_pool, ref_roc, nes_roc = ([] for _ in range(6))
    cells = []
    for seed in SEEDS:
        rows, o_r, o_n = run_seed(F, ref, cand, y, coords, seed)
        cells += rows
        ref_fold.append(np.mean([r["pr_ref"] for r in rows]))
        nes_fold.append(np.mean([r["pr_nested"] for r in rows]))
        ref_pool.append(average_precision_score(y, o_r))
        nes_pool.append(average_precision_score(y, o_n))
        ref_roc.append(roc_auc_score(y, o_r))
        nes_roc.append(roc_auc_score(y, o_n))
    diff = np.array([r["pr_nested"] - r["pr_ref"] for r in cells])
    stats = dict(
        ref_fold_pr=round(float(np.mean(ref_fold)), 4),
        nested_fold_pr=round(float(np.mean(nes_fold)), 4),
        diff_fold_pr=round(float(np.mean(nes_fold) - np.mean(ref_fold)), 4),
        diff_cell_sd=round(float(diff.std(ddof=1)), 4),
        share_cells_better=round(float((diff > 0).mean()), 3),
        ref_pooled_pr=round(float(np.mean(ref_pool)), 4),
        nested_pooled_pr=round(float(np.mean(nes_pool)), 4),
        ref_pooled_roc=round(float(np.mean(ref_roc)), 4),
        nested_pooled_roc=round(float(np.mean(nes_roc)), 4),
    )
    return stats, cells


def main():
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    y = (labels["Class"] == "Unhealthy").astype(int).to_numpy()
    coords = labels[["Long", "Lat"]].to_numpy()
    ref = pd.read_csv(ROOT / "misc/POC_Results/RVI_NDVI/airhitam_ndvi.csv")
    ref = ref.set_index("id")["NDVI"].reindex(labels["id"]).to_numpy()
    assert not np.isnan(ref).any(), "reference NDVI has missing trees"

    cells, summary = [], []
    for src, (s2f, s1f) in SOURCES.items():
        s2 = pd.read_csv(TS / s2f)
        s1 = pd.read_csv(TS / s1f)
        for wname, (start, end) in WINDOWS.items():
            feats = s1e.build(labels, s2, s1, start, end)
            too_sparse = feats.columns[feats.isna().mean() > 0.5].tolist()
            F = feats.drop(columns=too_sparse).reset_index(drop=True)
            for gname in GROUPS:
                cand = group_cols(list(F.columns), gname)
                if not cand:
                    continue
                group_cells, picks = [], Counter()
                stats, group_cells = evaluate(F, ref, cand, y, coords)
                cells += [dict(source=src, window=wname, group=gname, **r)
                          for r in group_cells]
                picks = Counter(r["pick"] for r in group_cells)
                row = dict(source=src, window=wname, group=gname,
                           n_cand=len(cand), n_sparse_dropped=len(too_sparse),
                           top_picks="; ".join(f"{k}:{v}" for k, v in
                                               picks.most_common(3)), **stats)
                summary.append(row)
                print(f"{src:7s} {wname:12s} {gname:8s} "
                      f"ref {row['ref_fold_pr']:.4f} -> nested "
                      f"{row['nested_fold_pr']:.4f} "
                      f"(diff {row['diff_fold_pr']:+.4f}, "
                      f"better {row['share_cells_better']:.0%})")

    S = pd.DataFrame(summary)
    pd.DataFrame(cells).to_csv(TS / "stage2_nested_cells.csv", index=False)
    S.to_csv(TS / "stage2_nested_summary.csv", index=False)
    (TS / "stage2_nested_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\n" + S.to_string(index=False))


if __name__ == "__main__":
    main()
