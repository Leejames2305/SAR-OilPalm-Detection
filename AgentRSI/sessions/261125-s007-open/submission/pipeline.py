"""Offline oil-palm triage using spatially validated optical model averaging.

Unhealthy is positive. Middle is excluded from every fitted loss and metric.
Model complexity and spectral mechanism are chosen from training labels by
block-disjoint CV and Bayesian block-bootstrap model averaging. Output scores
are ranking scores, not calibrated disease probabilities.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

_KEY = ['estate', 'Long', 'Lat']
_CONFIGS = (('base', 1), ('base', 2), ('prior', 1), ('prior', 2),
            ('rededge', 1), ('rededge', 2))
_PRIOR_BANDS = ['prev_' + b for b in
                ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')]


def _blocks(frame):
    x = ((frame.Long.to_numpy(float) - frame.Long.min()) * 111320
         * np.cos(np.deg2rad(frame.Lat.mean())))
    y = (frame.Lat.to_numpy(float) - frame.Lat.min()) * 110540
    return (np.floor(x / 80).astype(np.int64) * 100000
            + np.floor(y / 80).astype(np.int64))


def _xy(frame, reference):
    lat0 = float(reference.Lat.mean())
    lon0 = float(reference.Long.mean())
    return np.column_stack((
        (frame.Long.to_numpy(float) - lon0) * 111320 * np.cos(np.deg2rad(lat0)),
        (frame.Lat.to_numpy(float) - lat0) * 110540,
    ))


def _covariates(frame, features):
    # A left merge preserves prediction row order and exact floating-point keys.
    return frame[_KEY].merge(features, on=_KEY, how='left', sort=False,
                             validate='one_to_one').drop(columns=_KEY)


def _predict(train, target, features, config):
    kind, degree = config
    train = train[train.label.isin(('Healthy', 'Unhealthy'))]
    y = train.label.eq('Unhealthy').to_numpy(np.int64)
    if len(np.unique(y)) < 2:
        return np.zeros(len(target), dtype=np.float64)
    nt = train.NDVI.to_numpy(float)[:, None]
    nv = target.NDVI.to_numpy(float)[:, None]
    if kind != 'base':
        ft = _covariates(train, features)
        fv = _covariates(target, features)
    if kind == 'rededge':
        nt = np.column_stack((nt, ft.NDRE1))
        nv = np.column_stack((nv, fv.NDRE1))
    x = np.column_stack((nt, _xy(train, train)))
    z = np.column_stack((nv, _xy(target, train)))
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    z = scaler.transform(z)
    if degree == 2:
        poly = PolynomialFeatures(2, include_bias=False)
        x = poly.fit_transform(x)
        z = poly.transform(z)
    if kind == 'prior':
        # Prior-season bands enter linearly, avoiding a large interaction basis.
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        a = imputer.fit_transform(ft[_PRIOR_BANDS])
        b = imputer.transform(fv[_PRIOR_BANDS])
        scaler_aux = StandardScaler().fit(a)
        x = np.column_stack((x, scaler_aux.transform(a)))
        z = np.column_stack((z, scaler_aux.transform(b)))
    model = LogisticRegression(C=.03 if kind == 'rededge' else .1,
                               class_weight='balanced', max_iter=2000)
    model.fit(x, y)
    return model.decision_function(z)


def _weighted_ap(y, scores, weights):
    """Tie-aware step AP for many block-weight bootstrap replicates."""
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    w = weights[:, order]
    total = np.cumsum(w, axis=1)
    positive = np.cumsum(w * y[None, :], axis=1)
    increments = positive[:, ends] - np.column_stack((
        np.zeros(len(w)), positive[:, ends[:-1]],
    ))
    precision = positive[:, ends] / np.maximum(total[:, ends], 1e-12)
    return (np.sum(increments * precision, axis=1)
            / np.maximum(positive[:, -1], 1e-12))


def _fit_predict(labeled, unlabeled, ctx):
    if len(unlabeled) == 0:
        return np.empty(0, dtype=np.float64)
    features = pd.read_parquet(
        Path(ctx['datasets_dir']) / 'gee_s2_stress' / 'features.parquet'
    )
    train = labeled.reset_index(drop=True)
    scored = train.label.isin(('Healthy', 'Unhealthy')).to_numpy()
    y = train.label.eq('Unhealthy').to_numpy(float)[scored]
    groups = _blocks(train)
    unique, gi = np.unique(groups[scored], return_inverse=True)
    anchor = _predict(train, unlabeled, features, ('base', 2))
    if len(np.unique(groups)) < 5 or y.sum() < 5 or y.sum() == len(y):
        return np.asarray(anchor, dtype=np.float64)

    folds = (
        list(GroupKFold(5).split(train, groups=groups)),
        list(GroupKFold(5, shuffle=True, random_state=29).split(train, groups=groups)),
    )
    # Exponential weights are a Bayesian bootstrap over whole spatial blocks.
    rng = np.random.default_rng(54137)
    weights = rng.exponential(size=(128, len(unique)))[:, gi]
    utility = np.zeros((len(weights), len(_CONFIGS)))
    for j, config in enumerate(_CONFIGS):
        for split in folds:
            oof = np.empty(len(train), dtype=np.float64)
            for it, iv in split:
                oof[iv] = _predict(train.iloc[it], train.iloc[iv].drop(columns='label'),
                                   features, config)
            utility[:, j] += _weighted_ap(y, oof[scored], weights) / len(folds)
        # A small, shared complexity prior favours the compact reference family.
        utility[:, j] -= .005 * (config[0] != 'base')
    selected = np.argmax(utility, axis=1)
    model_weights = np.bincount(selected, minlength=len(_CONFIGS)) / len(weights)
    averaged = np.zeros(len(unlabeled), dtype=np.float64)
    for weight, config in zip(model_weights, _CONFIGS):
        if weight > 0:
            averaged += weight * _predict(train, unlabeled, features, config)
    # Uniform conservative anchor; no estate-name-specific model selection.
    return np.asarray(.75 * averaged + .25 * anchor, dtype=np.float64)


def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame,
                ctx: dict) -> np.ndarray:
    with threadpool_limits(limits=max(1, min(4, int(ctx['cpus'])))):
        return _fit_predict(labeled, unlabeled, ctx)
