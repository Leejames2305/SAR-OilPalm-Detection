from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 10

def _knn_prior(tree, y, xy_q, k, self_excl):
    kk = k + 1 if self_excl else k
    d, idx = tree.query(xy_q, k=kk)
    if self_excl:
        d, idx = d[:, 1:], idx[:, 1:]
    w = 1.0 / (d + 10.0)
    return (y[idx] * w).sum(1) / w.sum(1)

def fit_predict(labeled, unlabeled, ctx):
    tr = labeled[labeled["label"].isin(["Healthy", "Unhealthy"])].reset_index(drop=True)
    y = (tr["label"] == "Unhealthy").astype(int).to_numpy()
    xy_tr = tr[["Long", "Lat"]].to_numpy(float)
    xy_va = unlabeled[["Long", "Lat"]].to_numpy(float)
    tree = cKDTree(xy_tr)
    p_tr = _knn_prior(tree, y, xy_tr, K, True)
    p_va = _knn_prior(tree, y, xy_va, K, False)
    prev = y.mean()
    def feats(ndvi, p):
        return np.column_stack([ndvi, np.log(p + prev * 0.5)])
    X_tr = feats(tr[["NDVI"]].to_numpy(float)[:, 0], p_tr)
    X_va = feats(unlabeled[["NDVI"]].to_numpy(float)[:, 0], p_va)
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    m.fit(X_tr, y)
    return np.asarray(m.decision_function(X_va), dtype=np.float64)
