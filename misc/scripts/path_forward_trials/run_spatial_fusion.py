"""Targeted test: does SAR add anything on top of in-estate spatial context?"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402
from run_supervised import make_supervised_models  # noqa: E402

COORDS = ["estate_x_m", "estate_y_m"]
FEATURE_SETS = {
    "coords_only": COORDS,
    "log_raw4_plus_coords": tc.LOG_RAW_FEATURES + COORDS,
    "all14_plus_coords": tc.BASE_FEATURES + COORDS,
    "all14_plus_local_plus_coords": tc.BASE_FEATURES + tc.LOCAL_BASE_FEATURES + COORDS,
    "raw4_plus_local_plus_coords": tc.RAW_BANDS + tc.LOCAL_RAW_FEATURES + COORDS,
}
MODELS = ["logreg_l2", "rf_balanced", "xgboost_balanced"]


def broad_folds(g: pd.DataFrame) -> list[tc.Fold]:
    blocks = g.groupby("block", as_index=False).agg(Long=("Long", "mean"), Lat=("Lat", "mean"))
    labels = KMeans(n_clusters=min(4, len(blocks)), n_init=50, random_state=tc.SEED).fit(
        blocks[["Long", "Lat"]]
    ).labels_
    region = g.block.astype(str).map(dict(zip(blocks.block.astype(str), labels))).to_numpy(int)
    return [tc.Fold(0, int(r), np.where(region != r)[0], np.where(region == r)[0]) for r in sorted(np.unique(region))]


def main() -> None:
    start = time.time()
    df = tc.load_prepared_data()
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []
    for estate, group in df.groupby("Location", sort=True):
        g = group.reset_index(drop=False).rename(columns={"index": "source_row"})
        y = g.y_true.to_numpy(int)
        fold_sets = {
            "spatial_5fold_repeated": tc.make_folds(g),
            "leave_one_region_out_4": broad_folds(g),
        }
        for cv_name, folds in fold_sets.items():
            for fs, cols in FEATURE_SETS.items():
                x = g[cols].to_numpy(float)
                for repeat in sorted(set(f.repeat for f in folds)):
                    rfolds = [f for f in folds if f.repeat == repeat]
                    oof = {name: np.full(len(g), np.nan) for name in MODELS}
                    for fold in rfolds:
                        n_pos = int(y[fold.train].sum())
                        n_neg = int(len(fold.train) - n_pos)
                        models = make_supervised_models(tc.SEED + repeat * 101 + fold.fold, n_pos, n_neg)
                        for name in MODELS:
                            models[name].fit(x[fold.train], y[fold.train])
                            oof[name][fold.test] = tc.estimator_scores(models[name], x[fold.test])
                    for name in MODELS:
                        m = tc.evaluate_scores(y, oof[name])
                        rows.append(
                            tc.attach_metric_identity(
                                m,
                                stage="spatial_fusion",
                                cv=cv_name,
                                estate=estate,
                                model=name,
                                feature_set=fs,
                                repeat=repeat,
                            )
                        )
                        preds.append(
                            pd.DataFrame(
                                {
                                    "stage": "spatial_fusion",
                                    "cv": cv_name,
                                    "estate": estate,
                                    "model": name,
                                    "feature_set": fs,
                                    "repeat": repeat,
                                    "source_row": g.source_row.to_numpy(int),
                                    "id": g.id.to_numpy(int),
                                    "Long": g.Long.to_numpy(float),
                                    "Lat": g.Lat.to_numpy(float),
                                    "block": g.block.astype(str).to_numpy(),
                                    "fold": -1,
                                    "y_true": y,
                                    "score": oof[name],
                                }
                            )
                        )
    pd.DataFrame(rows).to_csv(tc.RESULT_DIR / "spatial_fusion_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(tc.RESULT_DIR / "spatial_fusion_oof_scores.csv", index=False)
    print(f"[spatial-fusion] wrote {len(rows)} rows in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
