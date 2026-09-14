"""Broader spatial holdout: cluster adjacent 220 m blocks into four regions."""
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
from run_healthy_anomaly import fit_healthy_detector  # noqa: E402


def region_map(g: pd.DataFrame) -> dict[str, int]:
    blocks = (
        g.groupby("block", as_index=False)
        .agg(Long=("Long", "mean"), Lat=("Lat", "mean"))
    )
    n_regions = min(4, len(blocks))
    model = KMeans(n_clusters=n_regions, n_init=50, random_state=tc.SEED).fit(blocks[["Long", "Lat"]])
    return dict(zip(blocks.block.astype(str), model.labels_))


def main() -> None:
    start = time.time()
    df = tc.load_prepared_data()
    tc.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    supervised_sets = ["all14", "log_raw4", "local14", "all14_plus_local", "coords_only"]
    supervised_models = ["logreg_l2", "rf_balanced", "xgboost_balanced"]
    anomaly_sets = ["all14", "local14", "raw4_plus_local"]
    anomaly_models = ["ocsvm_nu10", "isolation_forest", "pca_recon_10", "gmm_4"]

    for estate, group in df.groupby("Location", sort=True):
        estate_df = group.reset_index(drop=False).rename(columns={"index": "source_row"})
        mapping = region_map(estate_df)
        region = estate_df.block.astype(str).map(mapping).to_numpy(int)
        y = estate_df.y_true.to_numpy(int)
        regions = sorted(np.unique(region))
        print(f"[broad-spatial] {estate}: regions={np.bincount(region).tolist()}", flush=True)

        for fs in supervised_sets:
            x = estate_df[tc.FEATURE_SETS[fs]].to_numpy(float)
            oof = {name: np.full(len(estate_df), np.nan) for name in supervised_models}
            for test_region in regions:
                tr = np.where(region != test_region)[0]
                te = np.where(region == test_region)[0]
                n_pos = int(y[tr].sum())
                n_neg = int(len(tr) - n_pos)
                models = make_supervised_models(tc.SEED + test_region, n_pos, n_neg)
                for name in supervised_models:
                    model = models[name]
                    model.fit(x[tr], y[tr])
                    oof[name][te] = tc.estimator_scores(model, x[te])
            for name in supervised_models:
                m = tc.evaluate_scores(y, oof[name])
                metric_rows.append(
                    tc.attach_metric_identity(
                        m,
                        stage="supervised_broad_spatial",
                        cv="leave_one_region_out_4",
                        estate=estate,
                        model=name,
                        feature_set=fs,
                        repeat=0,
                    )
                )
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "stage": "supervised_broad_spatial",
                            "cv": "leave_one_region_out_4",
                            "estate": estate,
                            "model": name,
                            "feature_set": fs,
                            "repeat": 0,
                            "source_row": estate_df.source_row.to_numpy(int),
                            "id": estate_df.id.to_numpy(int),
                            "Long": estate_df.Long.to_numpy(float),
                            "Lat": estate_df.Lat.to_numpy(float),
                            "block": estate_df.block.astype(str).to_numpy(),
                            "fold": region,
                            "y_true": y,
                            "score": oof[name],
                        }
                    )
                )

        for fs in anomaly_sets:
            x = estate_df[tc.FEATURE_SETS[fs]].to_numpy(float)
            oof = {name: np.full(len(estate_df), np.nan) for name in anomaly_models}
            for test_region in regions:
                tr = np.where(region != test_region)[0]
                te = np.where(region == test_region)[0]
                healthy_train = tr[y[tr] == 0]
                for name in anomaly_models:
                    oof[name][te] = fit_healthy_detector(
                        name, x[healthy_train], x[te], tc.SEED + test_region
                    )
            for name in anomaly_models:
                m = tc.evaluate_scores(y, oof[name])
                metric_rows.append(
                    tc.attach_metric_identity(
                        m,
                        stage="healthy_only_broad_spatial",
                        cv="leave_one_region_out_4",
                        estate=estate,
                        model=name,
                        feature_set=fs,
                        repeat=0,
                    )
                )
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "stage": "healthy_only_broad_spatial",
                            "cv": "leave_one_region_out_4",
                            "estate": estate,
                            "model": name,
                            "feature_set": fs,
                            "repeat": 0,
                            "source_row": estate_df.source_row.to_numpy(int),
                            "id": estate_df.id.to_numpy(int),
                            "Long": estate_df.Long.to_numpy(float),
                            "Lat": estate_df.Lat.to_numpy(float),
                            "block": estate_df.block.astype(str).to_numpy(),
                            "fold": region,
                            "y_true": y,
                            "score": oof[name],
                        }
                    )
                )

    pd.DataFrame(metric_rows).to_csv(tc.RESULT_DIR / "broad_spatial_metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        tc.RESULT_DIR / "broad_spatial_oof_scores.csv", index=False
    )
    print(f"[broad-spatial] wrote {len(metric_rows)} rows in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
