"""Trusted reference baselines evaluated alongside every submission."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _scored_train(labeled: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    train = labeled[labeled["label"].isin(["Healthy", "Unhealthy"])]
    y = (train["label"] == "Unhealthy").astype(int).to_numpy()
    return train, y


def incumbent_fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Session-0 incumbent: NDVI-only balanced logistic regression."""
    train, y = _scored_train(labeled)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))
    model.fit(train[["NDVI"]].to_numpy(dtype=float), y)
    return np.asarray(model.decision_function(unlabeled[["NDVI"]].to_numpy(dtype=float)), dtype=np.float64)


def coords_only_fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Spatial coordinates-only reference baseline (distance-weighted 15-NN on Long/Lat)."""
    train, y = _scored_train(labeled)
    k = min(15, len(train))
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    model.fit(train[["Long", "Lat"]].to_numpy(dtype=float), y)
    probs = model.predict_proba(unlabeled[["Long", "Lat"]].to_numpy(dtype=float))
    if probs.shape[1] == 1:
        val = float(y[0]) if len(y) else 0.0
        return np.full(len(unlabeled), val, dtype=np.float64)
    return np.asarray(probs[:, 1], dtype=np.float64)
