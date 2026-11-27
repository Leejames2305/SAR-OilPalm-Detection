"""Offline oil-palm inspection ranking with partially pooled condition features.

Estates retain independent spatial/context responses; twelve physical optical
measurements additionally share a regularized coefficient across equally trusted
source estates. Scores are priorities, not calibrated disease probabilities.
Middle never trains the binary target; prediction preserves every input row.
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


def _pcs(train, target, standardize):
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

    x, z = _scaled(base(train), base(target))
    polynomial = PolynomialFeatures(2, include_bias=False)
    x, z = polynomial.fit_transform(x), polynomial.transform(z)
    # Degree-two terms have 1/.3**2 times the linear-term ridge penalty.
    x[:, 3:] *= .3
    z[:, 3:] *= .3
    a, b = _scaled(_read(train, root, _OPTICAL), _read(target, root, _OPTICAL))
    x, z = np.c_[x, .5 * a], np.c_[z, .5 * b]
    columns = [f'c2025_A{i:02d}' for i in range(64)]
    a, b = _pcs(_read(train, root, columns), _read(target, root, columns), True)
    x, z = np.c_[x, .5 * a], np.c_[z, .5 * b]
    a, b = _read(train, root, _HEIGHT), _read(target, root, _HEIGHT)
    mean = a.mean(0)
    denominator = np.sqrt(a.var(0) + np.array([3., 1., 1.]) ** 2)
    x = np.c_[x, .35 * (a - mean) / denominator]
    z = np.c_[z, .35 * (b - mean) / denominator]
    a, b = _scaled(_read(train, root, _MIXTURE), _read(target, root, _MIXTURE))
    x, z = np.c_[x, .3 * a], np.c_[z, .3 * b]
    # Frozen RGB-pretrained October SAR context; further PCA is training-only.
    a, b = _pcs(_read(train, root, 'oct103_', 'dino_sar'),
                _read(target, root, 'oct103_', 'dino_sar'), False)
    return np.c_[x, .3 * a], np.c_[z, .3 * b]


def _predict(labeled, unlabeled, ctx):
    if len(unlabeled) == 0:
        return np.empty(0, dtype=np.float64)
    train = labeled[labeled.label.isin(['Healthy', 'Unhealthy'])].reset_index(drop=True)
    y = (train.label == 'Unhealthy').to_numpy(int)
    if len(np.unique(y)) < 2:
        return np.full(len(unlabeled), y.mean() if len(y) else 0., dtype=np.float64)
    frames = [train] + [
        frame[frame.label.isin(['Healthy', 'Unhealthy'])].reset_index(drop=True)
        for _, frame in sorted(ctx['other_labeled'].items())
    ]
    local, shared, outcomes, sample_weights = [], [], [], []
    root = ctx['datasets_dir']
    for j, frame in enumerate(frames):
        target = unlabeled if j == 0 else frame
        a, b = _design(frame, target, root)
        sa, sb = _scaled(_read(frame, root, _OPTICAL), _read(target, root, _OPTICAL))
        local.append(a)
        shared.append(sa)
        outcome = (frame.label == 'Unhealthy').to_numpy(int)
        outcomes.append(outcome)
        # Equal total estate mass and balanced class mass within each estate.
        weight = np.where(outcome == 1, len(outcome) / (2 * outcome.sum()),
                          len(outcome) / (2 * (len(outcome) - outcome.sum())))
        sample_weights.append(weight * len(train) / len(frame))
        if j == 0:
            target_local, target_shared = b, sb
    widths = [a.shape[1] for a in local]
    lengths = [len(a) for a in local]
    k, n = len(_OPTICAL), len(local)
    x = np.zeros((sum(lengths), k + sum(widths) + n))
    z = np.zeros((len(unlabeled), x.shape[1]))
    x[:, :k], z[:, :k] = .3 * np.vstack(shared), .3 * target_shared
    row, column = 0, k
    for j, (width, length) in enumerate(zip(widths, lengths)):
        x[row:row + length, column:column + width] = local[j]
        x[row:row + length, k + sum(widths) + j] = 10
        if j == 0:
            z[:, column:column + width] = target_local
            z[:, k + sum(widths) + j] = 10
        row += length
        column += width
    model = LogisticRegression(C=.15, max_iter=2000, tol=1e-7, fit_intercept=False)
    model.fit(x, np.concatenate(outcomes), sample_weight=np.concatenate(sample_weights))
    return np.asarray(model.decision_function(z), dtype=np.float64)


def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    with threadpool_limits(limits=max(1, int(ctx['cpus']))):
        return _predict(labeled, unlabeled, ctx)
