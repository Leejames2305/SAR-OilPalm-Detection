"""Offline oil-palm inspection ranking with a uniform, regularized estate recipe.

Fit binary labels only. Add compact optical condition/context and eight training-
only PCs of prior-year AlphaEarth local contrast to NDVI and local coordinates.
All quadratic terms receive extra shrinkage. Scores are ranking values, not
calibrated disease probabilities. No label-dependent spatial neighbor terms.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

_KEYS = ['estate', 'Long', 'Lat']

@lru_cache(maxsize=1)
def _table(root, name):
    return pd.read_parquet(Path(root) / name / 'features.parquet').set_index(_KEYS)

def _read(frame, root, name, columns=None):
    data = _table(str(root), name).reindex(pd.MultiIndex.from_frame(frame[_KEYS]))
    if columns is not None:
        data = data[columns]
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite covariates in ' + name)
    return values

_OPTICAL = [
    'qNDVI', 'qNDRE', 'qNDMI', 'qNIRv', 'log_green_red',
    'NDVI_contrast35', 'NDRE1_contrast35', 'NDMI_contrast35',
    'ndvi_dip', 'ndvi_sd20', 'ndvi_edge', 'ndre_contrast',
]

def _optical(frame, root):
    return _read(frame, root, 'compact_triage', _OPTICAL)

def _scaled(train, target):
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(target)

def _fit_predict(labeled, unlabeled, ctx):
    if len(unlabeled) == 0:
        return np.empty(0, dtype=np.float64)
    train = labeled[labeled['label'].isin(['Healthy', 'Unhealthy'])].copy()
    y = (train['label'] == 'Unhealthy').to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return np.full(len(unlabeled), y.mean() if len(y) else 0., dtype=np.float64)
    root = ctx['datasets_dir']
    lat0 = float(train['Lat'].mean())
    lon0 = float(train['Long'].mean())
    def base(frame):
        return np.column_stack([
            frame['NDVI'].to_numpy(dtype=float),
            (frame['Long'].to_numpy(dtype=float) - lon0) * 111320 * np.cos(np.deg2rad(lat0)),
            (frame['Lat'].to_numpy(dtype=float) - lat0) * 110540,
        ])
    a, b = _scaled(base(train), base(unlabeled))
    poly = PolynomialFeatures(degree=2, include_bias=False)
    a, b = poly.fit_transform(a), poly.transform(b)
    # Differential ridge penalty: all degree-two terms have 1/.3**2 times
    # the linear-term penalty, with identical behavior for every estate.
    a[:, 3:] *= .3
    b[:, 3:] *= .3
    oa, ob = _scaled(_optical(train, root), _optical(unlabeled, root))
    cols = [f'c2025_A{i:02d}' for i in range(64)]
    ea, eb = _scaled(_read(train, root, 'compact_triage', cols), _read(unlabeled, root, 'compact_triage', cols))
    pca = PCA(n_components=min(8, len(train) - 1), svd_solver='full').fit(ea)
    ea, eb = _scaled(pca.transform(ea), pca.transform(eb))
    a = np.column_stack([a, .5 * oa, .5 * ea])
    b = np.column_stack([b, .5 * ob, .5 * eb])
    weights = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
    clf = LogisticRegression(C=.1, max_iter=2000, tol=1e-7, solver='lbfgs')
    clf.fit(a, y, sample_weight=weights)
    return np.asarray(clf.decision_function(b), dtype=np.float64)

def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    with threadpool_limits(limits=max(1, int(ctx['cpus']))):
        return _fit_predict(labeled, unlabeled, ctx)
