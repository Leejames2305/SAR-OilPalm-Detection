"""Session-0 incumbent pipeline: NDVI-only balanced logistic regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Train on labeled Healthy/Unhealthy rows using NDVI only and score unlabeled rows."""
    train = labeled[labeled["label"].isin(["Healthy", "Unhealthy"])]
    X_train = train[["NDVI"]].to_numpy(dtype=float)
    y_train = (train["label"] == "Unhealthy").astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))
    model.fit(X_train, y_train)
    X_unlabeled = unlabeled[["NDVI"]].to_numpy(dtype=float)
    return np.asarray(model.decision_function(X_unlabeled), dtype=np.float64)
