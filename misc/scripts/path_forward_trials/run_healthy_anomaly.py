"""Healthy-only anomaly-detection trials.

Every detector is fitted on Healthy training trees only. Held-out Healthy and
Unhealthy trees receive a score where larger = more anomalous.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def fit_healthy_detector(name: str, x_train_h: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    scaler = RobustScaler().fit(x_train_h)
    xh = scaler.transform(x_train_h)
    xt = scaler.transform(x_test)

    if name == "ocsvm_nu05":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        model.fit(xh)
        return -model.decision_function(xt)
    if name == "ocsvm_nu10":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
        model.fit(xh)
        return -model.decision_function(xt)
    if name == "ocsvm_nu20":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.20)
        model.fit(xh)
        return -model.decision_function(xt)
    if name == "isolation_forest":
        model = IsolationForest(
            n_estimators=500,
            max_samples=min(256, len(xh)),
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(xh)
        return -model.score_samples(xt)
    if name == "lof_25":
        model = LocalOutlierFactor(n_neighbors=min(25, len(xh) - 1), novelty=True)
        model.fit(xh)
        return -model.score_samples(xt)
    if name in {"gmm_4", "gmm_8"}:
        n_components = 4 if name == "gmm_4" else 8
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-3,
            random_state=seed,
            max_iter=300,
        )
        model.fit(xh)
        return -model.score_samples(xt)
    if name == "pca_recon_10":
        n_components = min(10, xh.shape[1], max(1, len(xh) - 1))
        model = PCA(n_components=n_components, random_state=seed).fit(xh)
        recon = model.inverse_transform(model.transform(xt))
        return np.mean((xt - recon) ** 2, axis=1)
    if name == "knn_dist_20":
        k = min(20, len(xh))
        model = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(xh)
        distances, _ = model.kneighbors(xt)
        return np.mean(distances, axis=1)
    if name == "mahalanobis":
        model = LedoitWolf().fit(xh)
        return model.mahalanobis(xt)
    if name == "kmeans_dist_8":
        model = KMeans(n_clusters=8, n_init=20, random_state=seed).fit(xh)
        return np.min(model.transform(xt), axis=1)
    if name == "autoencoder":
        model = MLPRegressor(
            hidden_layer_sizes=(32, 8, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            max_iter=600,
            random_state=seed,
        )
        model.fit(xh, xh)
        recon = model.predict(xt)
        return np.mean((xt - recon) ** 2, axis=1)
    raise ValueError(f"unknown detector: {name}")


def _summarize(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    metric_cols = [
        c for c in metrics.columns
        if c not in {"stage", "cv", "estate", "model", "feature_set", "repeat"}
    ]
    rows = []
    for keys, g in metrics.groupby(["stage", "cv", "estate", "model", "feature_set"], sort=False):
        row = dict(zip(["stage", "cv", "estate", "model", "feature_set"], keys))
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def run(df: pd.DataFrame, *, quick: bool = False) -> None:
    primary_models = [
        "ocsvm_nu05",
        "ocsvm_nu10",
        "ocsvm_nu20",
        "isolation_forest",
        "lof_25",
        "gmm_4",
        "gmm_8",
        "pca_recon_10",
        "knn_dist_20",
        "mahalanobis",
        "kmeans_dist_8",
        "autoencoder",
    ]
    secondary_models = ["ocsvm_nu10", "isolation_forest", "pca_recon_10", "knn_dist_20", "gmm_4"]
    feature_sets = ["all14", "all14_plus_ext", "local14", "all14_plus_local", "raw4_plus_local", "raw4"]

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    start = time.time()

    for estate, group in df.groupby("Location", sort=True):
        estate_df = group.reset_index(drop=False).rename(columns={"index": "source_row"})
        folds = tc.make_folds(estate_df, n_repeats=1 if quick else tc.N_REPEATS)
        y = estate_df["y_true"].to_numpy(int)
        for fs in feature_sets:
            cols = tc.FEATURE_SETS[fs]
            squat = estate_df[cols].to_numpy(float)
            model_names = primary_models if fs == "all14" else secondary_models
            print(f"[healthy-only] {estate} / {fs}: {len(model_names)} detectors", flush=True)
            for repeat in sorted(set(f.repeat for f in folds)):
                repeat_folds = [f for f in folds if f.repeat == repeat]
                oof = {name: np.full(len(estate_df), np.nan) for name in model_names}
                fold_id = np.full(len(estate_df), -1, int)
                for fold in repeat_folds:
                    tr, te = fold.train, fold.test
                    healthy_train = tr[y[tr] == 0]
                    if len(healthy_train) < 50:
                        raise RuntimeError(f"too few healthy training trees: {estate}/{fs}/{fold.fold}")
                    for name in model_names:
                        score = fit_healthy_detector(
                            name,
                            squat[healthy_train],
                            squat[te],
                            tc.SEED + repeat * 101 + fold.fold,
                        )
                        if not np.all(np.isfinite(score)):
                            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
                        oof[name][te] = score
                        fold_id[te] = fold.fold
                for name in model_names:
                    score = oof[name]
                    if not np.all(np.isfinite(score)):
                        raise RuntimeError(f"non-finite OOF scores: {estate}/{fs}/{name}")
                    metrics = tc.evaluate_scores(y, score)
                    metric_rows.append(
                        tc.attach_metric_identity(
                            metrics,
                            stage="healthy_only",
                            cv="spatial_5fold_repeated",
                            estate=estate,
                            model=name,
                            feature_set=fs,
                            repeat=repeat,
                        )
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "stage": "healthy_only",
                                "cv": "spatial_5fold_repeated",
                                "estate": estate,
                                "model": name,
                                "feature_set": fs,
                                "repeat": repeat,
                                "source_row": estate_df["source_row"].to_numpy(int),
                                "id": estate_df["id"].to_numpy(int),
                                "Long": estate_df["Long"].to_numpy(float),
                                "Lat": estate_df["Lat"].to_numpy(float),
                                "block": estate_df["block"].astype(str).to_numpy(),
                                "fold": fold_id,
                                "y_true": y,
                                "score": score,
                            }
                        )
                    )

    metrics = tc.save_table(metric_rows, tc.RESULT_DIR / "healthy_only_repeat_metrics.csv")
    _summarize(metrics, tc.RESULT_DIR / "healthy_only_summary.csv")
    preds = pd.concat(prediction_frames, ignore_index=True)
    preds.to_csv(tc.RESULT_DIR / "healthy_only_oof_scores.csv", index=False)
    print(f"[healthy-only] wrote {len(metrics)} metric rows and {len(preds)} OOF rows in {time.time()-start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="one repeat; for smoke tests only")
    args = parser.parse_args()
    tc.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run(tc.load_prepared_data(), quick=args.quick)


if __name__ == "__main__":
    main()
