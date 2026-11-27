"""Offline oil-palm triage with partially pooled optical and patch context.

Higher scores mean higher screening priority, not calibrated disease odds.
Only Healthy/Unhealthy labels train the model; every target row is returned.
All feature tables are keyed by exact estate/longitude/latitude coordinates.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

_KEY = ["estate", "Long", "Lat"]


@lru_cache(maxsize=8)
def _table(root: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / name / "features.parquet").set_index(_KEY)


def _covariates(frame: pd.DataFrame, ctx: dict, name: str) -> pd.DataFrame:
    table = _table(str(ctx["datasets_dir"]), name)
    return table.loc[pd.MultiIndex.from_frame(frame[_KEY])].reset_index(drop=True)


def _optical(frame: pd.DataFrame, ctx: dict) -> np.ndarray:
    quality = _covariates(frame, ctx, "gee_quality")
    canopy = _covariates(frame, ctx, "gee_canopy")
    texture = _covariates(frame, ctx, "optical_jitter")
    return np.column_stack([
        np.log(np.maximum(quality.B4 + quality.B8, 1)),
        quality.B3 / (quality.B4 + quality.B3 + 1),
        canopy.NDVI_contrast35,
        canopy.NDRE1_contrast35,
        *[texture[column] for column in texture],
    ])


def _base(train: pd.DataFrame, target: pd.DataFrame, ctx: dict):
    lon0, lat0 = train.Long.mean(), train.Lat.mean()

    def matrix(frame):
        quality = _covariates(frame, ctx, "gee_quality")
        ndvi = .5 * frame.NDVI.to_numpy(float) + .5 * quality.qNDVI.to_numpy()
        x = (frame.Long.to_numpy() - lon0) * 111320 * np.cos(np.deg2rad(lat0))
        y = (frame.Lat.to_numpy() - lat0) * 110540
        return np.column_stack([ndvi, x, y])

    scaler = StandardScaler()
    x = scaler.fit_transform(matrix(train))
    z = scaler.transform(matrix(target))
    # Inherited incumbent complexity choice, retained after unsuccessful
    # inner block selectors. This is an explicit limitation for new estates.
    if True:
        polynomial = PolynomialFeatures(2, include_bias=False)
        x, z = polynomial.fit_transform(x), polynomial.transform(z)
        x[:, 3:] *= .3
        z[:, 3:] *= .3
    return x, z


def _context(train: pd.DataFrame, target: pd.DataFrame, ctx: dict, radar: bool = False):
    optical_train, optical_target = _optical(train, ctx), _optical(target, ctx)
    alpha_train = _covariates(train, ctx, "alphaearth").filter(regex="^c2025_")
    alpha_target = _covariates(target, ctx, "alphaearth").filter(regex="^c2025_")
    alpha_pca = PCA(n_components=8, svd_solver="full")
    x = np.column_stack([optical_train, alpha_pca.fit_transform(alpha_train.to_numpy())])
    z = np.column_stack([optical_target, alpha_pca.transform(alpha_target.to_numpy())])
    scaler = StandardScaler()
    x, z = scaler.fit_transform(x), scaler.transform(z)

    # Saved frozen-encoder PCs contain no labels. Further PCA/scaling sees
    # only this estate's supplied training covariates.
    dino_train = _covariates(train, ctx, "dino_optical").filter(regex="^nrg160_")
    dino_target = _covariates(target, ctx, "dino_optical").filter(regex="^nrg160_")
    dino_pca = PCA(n_components=8, svd_solver="full")
    dino_x = dino_pca.fit_transform(dino_train)
    dino_z = dino_pca.transform(dino_target)
    dino_scaler = StandardScaler()
    dino_x, dino_z = dino_scaler.fit_transform(dino_x), dino_scaler.transform(dino_z)
    x, z = np.column_stack([x, dino_x]), np.column_stack([z, dino_z])
    if radar:
        radar_train = _covariates(train, ctx, "dino_sar").filter(regex="^oct103_")
        radar_target = _covariates(target, ctx, "dino_sar").filter(regex="^oct103_")
        radar_pca = PCA(n_components=8, svd_solver="full")
        radar_x, radar_z = radar_pca.fit_transform(radar_train), radar_pca.transform(radar_target)
        radar_scaler = StandardScaler()
        radar_x, radar_z = radar_scaler.fit_transform(radar_x), radar_scaler.transform(radar_z)
        x, z = np.column_stack([x, radar_x]), np.column_stack([z, radar_z])
    return x, z


def _fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict, radar: bool = False) -> np.ndarray:
    frames = [labeled] + [d for _, d in sorted(ctx["other_labeled"].items())]
    frames = [d[d.label.isin(["Healthy", "Unhealthy"])].reset_index(drop=True) for d in frames]
    shared, nuisance, outcomes, weights = [], [], [], []
    for j, train in enumerate(frames):
        estate_ctx = dict(ctx, estate=str(train.estate.iloc[0]))
        target = unlabeled if j == 0 else train
        base_x, base_z = _base(train, target, estate_ctx)
        local_x, local_z = _context(train, target, estate_ctx, radar=radar)
        scaler = StandardScaler()
        shared_x = scaler.fit_transform(_optical(train, estate_ctx))
        shared_z = scaler.transform(_optical(target, estate_ctx))
        shared.append(shared_x)
        nuisance.append(np.column_stack([base_x, .3 * local_x]))
        y = (train.label == "Unhealthy").to_numpy(int)
        outcomes.append(y)
        # Equal total weight per estate and equal class weight within estate.
        weights.append(np.where(y == 1, .5 / y.mean(), .5 / (1 - y.mean())) * len(frames[0]) / len(train))
        if j == 0:
            target_local = np.column_stack([base_z, .3 * local_z])
            target_shared = shared_z

    counts = [len(v) for v in shared]
    widths = [v.shape[1] for v in nuisance]
    k = shared[0].shape[1]
    x = np.zeros((sum(counts), k + sum(widths) + 3))
    z = np.zeros((len(unlabeled), x.shape[1]))
    x[:, :k] = .3 * np.vstack(shared)
    z[:, :k] = .3 * target_shared
    row, column = 0, k
    for j, (n, width) in enumerate(zip(counts, widths)):
        x[row:row+n, column:column+width] = nuisance[j]
        x[row:row+n, k+sum(widths)+j] = 10
        if j == 0:
            z[:, column:column+width] = target_local
            z[:, k+sum(widths)+j] = 10
        row, column = row+n, column+width

    model = LogisticRegression(C=.1, max_iter=2000, fit_intercept=False)
    model.fit(x, np.concatenate(outcomes), sample_weight=np.concatenate(weights))
    return np.asarray(model.decision_function(z), dtype=np.float64)


def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    if len(unlabeled) == 0:
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=max(1, min(2, int(ctx["cpus"])))):
        return _fit_predict(labeled, unlabeled, ctx, radar=True)
