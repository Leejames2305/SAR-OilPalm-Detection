"""Spatial label-context baselines for in-estate triage only.

These methods intentionally use confirmed training labels plus tree geometry.
They are not transferable to a new estate and are evaluated separately from the
SAR-feature classifiers.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402


def coords_m(g: pd.DataFrame) -> np.ndarray:
    lat0 = float(g.Lat.mean())
    x = (g.Long.to_numpy(float) - float(g.Long.mean())) * 111_320.0 * np.cos(np.deg2rad(lat0))
    y = (g.Lat.to_numpy(float) - float(g.Lat.mean())) * 111_320.0
    return np.column_stack([x, y])


def score_spatial_context(train_xy: np.ndarray, train_y: np.ndarray, test_xy: np.ndarray) -> dict[str, np.ndarray]:
    tree = cKDTree(train_xy)
    k = min(25, len(train_xy))
    dist, inds = tree.query(test_xy, k=k)
    y_n = train_y[inds]
    weights = 1.0 / np.maximum(dist, 1.0)
    out: dict[str, np.ndarray] = {}
    for kk in (10, 25):
        kk = min(kk, k)
        out[f"knn_label_fraction_{kk}"] = np.average(y_n[:, :kk], axis=1, weights=weights[:, :kk])
    pos = np.where(train_y == 1)[0]
    neg = np.where(train_y == 0)[0]
    if len(pos) and len(neg):
        du = cKDTree(train_xy[pos]).query(test_xy, k=1)[0]
        dh = cKDTree(train_xy[neg]).query(test_xy, k=1)[0]
        out["nearest_u_relative_distance"] = np.log((dh + 1.0) / (du + 1.0))
        out["nearest_u_minus_h"] = dh - du
    return out


def region_folds(g: pd.DataFrame, n_regions: int = 4) -> np.ndarray:
    blocks = g.groupby("block", as_index=False).agg(Long=("Long", "mean"), Lat=("Lat", "mean"))
    labels = KMeans(n_clusters=min(n_regions, len(blocks)), n_init=50, random_state=tc.SEED).fit(
        blocks[["Long", "Lat"]]
    ).labels_
    return g.block.astype(str).map(dict(zip(blocks.block.astype(str), labels))).to_numpy(int)


def main() -> None:
    start = time.time()
    df = tc.load_prepared_data()
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []
    for estate, group in df.groupby("Location", sort=True):
        g = group.reset_index(drop=False).rename(columns={"index": "source_row"})
        xy = coords_m(g)
        y = g.y_true.to_numpy(int)

        for cv_name, folds in [
            ("spatial_5fold_repeated", tc.make_folds(g)),
            ("leave_one_region_out_4", [
                tc.Fold(0, int(r), np.where(region_folds(g) != r)[0], np.where(region_folds(g) == r)[0])
                for r in sorted(np.unique(region_folds(g)))
            ]),
        ]:
            max_repeat = max(f.repeat for f in folds)
            for repeat in range(max_repeat + 1):
                rfolds = [f for f in folds if f.repeat == repeat]
                scores: dict[str, np.ndarray] = {}
                fold_ids = np.full(len(g), -1, int)
                for fold in rfolds:
                    local = score_spatial_context(xy[fold.train], y[fold.train], xy[fold.test])
                    for name, score in local.items():
                        if name not in scores:
                            scores[name] = np.full(len(g), np.nan)
                        scores[name][fold.test] = score
                    fold_ids[fold.test] = fold.fold
                for name, score in scores.items():
                    m = tc.evaluate_scores(y, score)
                    rows.append(
                        tc.attach_metric_identity(
                            m,
                            stage="spatial_context",
                            cv=cv_name,
                            estate=estate,
                            model=name,
                            feature_set="coords_plus_training_labels",
                            repeat=repeat,
                        )
                    )
                    preds.append(
                        pd.DataFrame(
                            {
                                "stage": "spatial_context",
                                "cv": cv_name,
                                "estate": estate,
                                "model": name,
                                "feature_set": "coords_plus_training_labels",
                                "repeat": repeat,
                                "source_row": g.source_row.to_numpy(int),
                                "id": g.id.to_numpy(int),
                                "Long": g.Long.to_numpy(float),
                                "Lat": g.Lat.to_numpy(float),
                                "block": g.block.astype(str).to_numpy(),
                                "fold": fold_ids,
                                "y_true": y,
                                "score": score,
                            }
                        )
                    )
    pd.DataFrame(rows).to_csv(tc.RESULT_DIR / "spatial_context_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(tc.RESULT_DIR / "spatial_context_oof_scores.csv", index=False)
    print(f"[spatial-context] wrote {len(rows)} rows in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
