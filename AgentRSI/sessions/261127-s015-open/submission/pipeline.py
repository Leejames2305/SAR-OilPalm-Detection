"""Offline oil-palm triage with a source-only optical coefficient prior.

Each estate retains its own spatial, spectral and contextual response. Equally
trusted source estates supply the centre of a ridge prior on twelve optical
coefficients. Predictions are inspection priorities, not disease probabilities.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

_KEYS = ['estate', 'Long', 'Lat']
_OPTICAL = [
    'qNDVI', 'qNDRE', 'qNDMI', 'qNIRv', 'log_green_red',
    'NDVI_contrast35', 'NDRE1_contrast35', 'NDMI_contrast35',
    'ndvi_dip', 'ndvi_sd20', 'ndvi_edge', 'ndre_contrast',
]
_HEIGHT = ['height_mean6', 'height_sd6', 'height_contrast6']
_MIXTURE = ['deblur_' + name for name in [
    'NDVI', 'NDRE', 'NDMI', 'NBR', 'NIRv', 'GRVI',
    'NDVI_contrast', 'NDRE_contrast', 'NDMI_contrast',
]]
_C = .15


@lru_cache(maxsize=2)
def _table(root, name):
    return pd.read_parquet(Path(root) / name / 'features.parquet').set_index(_KEYS)


def _read(frame, root, columns, name='triage_v12'):
    table = _table(str(root), name)
    if isinstance(columns, str):
        columns = [column for column in table.columns if column.startswith(columns)]
    values = table.reindex(pd.MultiIndex.from_frame(frame[_KEYS]))[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite prepared covariates')
    return values


def _scaled(train, target):
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(target)


def _pcs(train, target, standardize=True):
    if standardize:
        train, target = _scaled(train, target)
    pca = PCA(n_components=min(8, len(train) - 1, train.shape[1]), svd_solver='full').fit(train)
    return _scaled(pca.transform(train), pca.transform(target))


def _design(train, target, root):
    latitude, longitude = float(train.Lat.mean()), float(train.Long.mean())

    def base(frame):
        return np.column_stack([
            frame.NDVI.to_numpy(float),
            (frame.Long.to_numpy() - longitude) * 111320 * np.cos(np.deg2rad(latitude)),
            (frame.Lat.to_numpy() - latitude) * 110540,
        ])

    a, b = _scaled(base(train), base(target))
    polynomial = PolynomialFeatures(2, include_bias=False)
    x, z = polynomial.fit_transform(a), polynomial.transform(b)
    x[:, 3:] *= .3
    z[:, 3:] *= .3
    oa, ob = _scaled(_read(train, root, _OPTICAL), _read(target, root, _OPTICAL))
    x, z = np.c_[x, .5 * oa], np.c_[z, .5 * ob]

    context_train, context_target = [], []
    columns = [f'c2025_A{i:02d}' for i in range(64)]
    a, b = _pcs(_read(train, root, columns), _read(target, root, columns))
    context_train.append(.5 * a)
    context_target.append(.5 * b)
    a, b = _read(train, root, _HEIGHT), _read(target, root, _HEIGHT)
    mean = a.mean(0)
    denominator = np.sqrt(a.var(0) + np.array([3., 1., 1.]) ** 2)
    context_train.append(.35 * (a - mean) / denominator)
    context_target.append(.35 * (b - mean) / denominator)
    a, b = _scaled(_read(train, root, _MIXTURE), _read(target, root, _MIXTURE))
    context_train.append(.3 * a)
    context_target.append(.3 * b)
    a, b = _pcs(_read(train, root, 'oct103_', 'dino_sar'),
                _read(target, root, 'oct103_', 'dino_sar'), False)
    context_train.append(.3 * a)
    context_target.append(.3 * b)
    a, b = np.c_[tuple(context_train)], np.c_[tuple(context_target)]
    x, z = np.c_[x, a], np.c_[z, b]
    # Estate-specific intercept with weak ridge shrinkage, identical in each fit.
    return np.c_[x, np.full(len(train), 10.)], np.c_[z, np.full(len(target), 10.)]


def _weights(y):
    return np.where(y == 1, len(y) / (2 * max(1, y.sum())),
                    len(y) / (2 * max(1, (1 - y).sum())))


def _predict(labeled, unlabeled, ctx):
    if len(unlabeled) == 0:
        return np.empty(0, dtype=np.float64)
    train = labeled[labeled.label.isin(['Healthy', 'Unhealthy'])].reset_index(drop=True)
    y = (train.label == 'Unhealthy').to_numpy(int)
    if len(np.unique(y)) < 2:
        return np.full(len(unlabeled), y.mean() if len(y) else 0., dtype=np.float64)
    root = ctx['datasets_dir']
    source_coefficients = []
    optical_indices = np.arange(9, 9 + len(_OPTICAL))
    for _, frame in sorted(ctx['other_labeled'].items()):
        source = frame[frame.label.isin(['Healthy', 'Unhealthy'])].reset_index(drop=True)
        outcome = (source.label == 'Unhealthy').to_numpy(int)
        if len(np.unique(outcome)) < 2:
            continue
        a, _ = _design(source, source, root)
        model = LogisticRegression(C=_C, fit_intercept=False, tol=1e-7, max_iter=2000)
        model.fit(a, outcome, sample_weight=_weights(outcome))
        source_coefficients.append(model.coef_[0, optical_indices])
    # Mean source coefficient: equal estate trust, without averaging risk scores.
    centre = (np.vstack(source_coefficients).mean(0) if source_coefficients
              else np.zeros(len(_OPTICAL)))
    precision = np.full(len(_OPTICAL), 1 / _C)
    x, z = _design(train, unlabeled, root)
    weights = _weights(y)

    def objective(beta):
        scores = x @ beta
        difference = beta[optical_indices] - centre
        # Replace only the optical zero-centred penalty by the source prior.
        value = (np.sum(weights * (np.logaddexp(0, scores) - y * scores))
                 + .5 * (beta @ beta) / _C
                 - .5 * (beta[optical_indices] @ beta[optical_indices]) / _C
                 + .5 * np.sum(precision * difference * difference))
        gradient = x.T @ (weights * (expit(scores) - y)) + beta / _C
        gradient[optical_indices] += precision * difference - beta[optical_indices] / _C
        return value, gradient

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-6})
    return np.asarray(z @ result.x, dtype=np.float64)


def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    with threadpool_limits(limits=max(1, int(ctx['cpus']))):
        return _predict(labeled, unlabeled, ctx)
