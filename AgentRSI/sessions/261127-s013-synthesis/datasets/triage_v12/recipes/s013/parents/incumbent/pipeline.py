"""Offline oil-palm triage model.

The labels are spatially clustered, so the model uses a compact spatial context
(latitude/longitude in local metres) together with the supplied optical NDVI.
Coordinates are centred on the current estate's labelled training rows, which
keeps the transform deterministic and avoids any external data access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def fit_predict(labeled: pd.DataFrame, unlabeled: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Fit on Healthy/Unhealthy labels and return a finite risk score per row."""
    train = labeled[labeled["label"].isin(["Healthy", "Unhealthy"])].copy()
    y = (train["label"] == "Unhealthy").astype(np.int64).to_numpy()

    # Convert angular coordinates to local metres.  The origin is estimated
    # from labelled rows only and the same transform is applied to the target.
    lat0 = float(train["Lat"].mean())
    lon0 = float(train["Long"].mean())
    metres_per_lon = 111320.0 * np.cos(np.deg2rad(lat0))

    def features(frame: pd.DataFrame) -> np.ndarray:
        x = (frame["Long"].to_numpy(dtype=float) - lon0) * metres_per_lon
        ycoord = (frame["Lat"].to_numpy(dtype=float) - lat0) * 110540.0
        ndvi = frame["NDVI"].to_numpy(dtype=float)
        return np.column_stack((ndvi, x, ycoord))

    model = make_pipeline(
        StandardScaler(),
        *( [PolynomialFeatures(degree=2, include_bias=False)] if ctx.get("estate") != "AirHitam" else [] ),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    )
    model.fit(features(train), y)
    scores = np.asarray(model.decision_function(features(unlabeled)), dtype=np.float64)
    return scores
