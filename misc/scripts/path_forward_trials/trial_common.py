"""Shared utilities for the path-forward trials.

The primary evaluation unit is the tree. Honest within-estate CV is
StratifiedGroupKFold over the existing ~220 m spatial blocks. All models in a
comparison use identical folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "Processed" / "dataset_all_estates.csv"
RESULT_DIR = ROOT / "misc" / "POC_Results" / "PathForwardTrials"
SEED = 42
N_SPLITS = 5
N_REPEATS = 3

RAW_BANDS = ["HH_meanW3", "HV_meanW3", "VV_meanW3", "VH_meanW3"]
HA_FEATURES = [
    "HA_Entropy_meanW3",
    "HA_Anisotropy_meanW3",
    "HA_Alpha_meanW3",
]
YA_FEATURES = [
    "YA_Yamaguchi_dbl_r_meanW3",
    "YA_Yamaguchi_vol_g_meanW3",
    "YA_Yamaguchi_surf_b_meanW3",
    "YA_Yamaguchi_hlx_meanW3",
]
EXISTING_INDICES = ["RVI_meanW3", "RFDI_meanW3", "RFDI_VH_meanW3"]
BASE_FEATURES = RAW_BANDS + HA_FEATURES + YA_FEATURES + EXISTING_INDICES
EXTENDED_INDICES = [
    "RVI_ext_CS",
    "RVI_ext_BMI",
    "RVI_ext_Rc",
    "RVI_ext_Rp",
    "RVI_ext_Pt",
    "RVI_ext_VSI",
    "RVI_ext_CSI",
    "RVI_ext_RVIHH",
    "RVI_ext_RVIVV",
    "RVI_ext_RNDVI",
]
LOG_RAW_FEATURES = [f"log_{c}" for c in RAW_BANDS]
LOCAL_BASE_FEATURES = [f"local_{c}" for c in BASE_FEATURES]
LOCAL_RAW_FEATURES = [f"local_{c}" for c in RAW_BANDS]
COORD_FEATURES = ["estate_x_m", "estate_y_m"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically interpretable radar indices not already in the CSV.

    Equations follow the RVI review Table 1. Reciprocity is not imposed because
    the supplied HV and VH channels are not numerically equal.
    """
    out = df.copy()
    hh = out["HH_meanW3"].to_numpy(float)
    hv = out["HV_meanW3"].to_numpy(float)
    vv = out["VV_meanW3"].to_numpy(float)
    vh = out["VH_meanW3"].to_numpy(float)
    eps = np.finfo(float).eps

    cs = (hv + vh) / 2.0
    bmi = (hh + vv) / 2.0
    out["RVI_ext_CS"] = cs
    out["RVI_ext_BMI"] = bmi
    out["RVI_ext_Rc"] = cs / np.maximum(hh + vv, eps)
    out["RVI_ext_Rp"] = vv / np.maximum(hh, eps)
    out["RVI_ext_Pt"] = hh + hv + vv + vh
    out["RVI_ext_VSI"] = cs / np.maximum(cs + bmi, eps)
    out["RVI_ext_CSI"] = vv / np.maximum(hh + vv, eps)
    out["RVI_ext_RVIHH"] = 4.0 * hv / np.maximum(hh + hv, eps)
    out["RVI_ext_RVIVV"] = 4.0 * vh / np.maximum(vv + vh, eps)
    out["RVI_ext_RNDVI"] = (hh - vv) / np.maximum(hh + vv, eps)
    for col in RAW_BANDS:
        out[f"log_{col}"] = np.log(np.maximum(out[col].to_numpy(float), eps))
    return out


def add_local_residuals(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    k: int = 10,
    prefix: str = "local_",
) -> pd.DataFrame:
    """Add label-free residual features versus each tree's nearest peers.

    Coordinates are projected to approximate metres. For each estate, every
    tree is compared with its k nearest neighbours (self excluded). Divisions
    use the estate-level IQR so residuals are comparable across features.
    This is transductive but label-free: it can be computed at inference from
    the estate's SAR feature map.
    """
    out = df.copy()
    feature_cols = list(feature_cols)
    new_cols = [prefix + c for c in feature_cols]
    for col in new_cols:
        out[col] = np.nan

    for _, idx in out.groupby("Location", sort=False).groups.items():
        idx = np.asarray(idx)
        g = out.loc[idx]
        lat0 = float(g["Lat"].mean())
        x = (g["Long"].to_numpy(float) - float(g["Long"].mean())) * 111_320.0 * np.cos(np.deg2rad(lat0))
        y = (g["Lat"].to_numpy(float) - float(g["Lat"].mean())) * 111_320.0
        coords = np.column_stack([x, y])
        n = len(g)
        kk = min(k, n - 1)
        if kk < 1:
            continue
        _, neighbours = cKDTree(coords).query(coords, k=kk + 1)
        neighbours = neighbours[:, 1:]
        values = g[feature_cols].to_numpy(float)
        local_med = np.median(values[neighbours], axis=1)
        q25 = np.percentile(values, 25, axis=0)
        q75 = np.percentile(values, 75, axis=0)
        scale = q75 - q25
        fallback = np.std(values, axis=0)
        scale = np.where(np.abs(scale) > 1e-12, scale, fallback)
        scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
        residuals = (values - local_med) / scale
        out.loc[idx, new_cols] = residuals
        out.loc[idx, "estate_x_m"] = x
        out.loc[idx, "estate_y_m"] = y
    return out


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def make_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> list[Fold]:
    folds: list[Fold] = []
    groups = df["block"].astype(str).to_numpy()
    y = (df["Class"].to_numpy() == "Unhealthy").astype(int)
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            folds.append(Fold(repeat, fold, train, test))
    return folds


def estimator_scores(model, x: np.ndarray) -> np.ndarray:
    """Return a score where a larger value means more Unhealthy-like."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x)).ravel()
    return np.asarray(model.predict(x)).ravel()


def _safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(score)) < 2:
        return 0.5
    return float(roc_auc_score(y, score))


def top_budget_metrics(
    y: np.ndarray,
    score: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> dict[str, float]:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    n = len(y)
    n_pos = int(y.sum())
    prevalence = n_pos / n if n else np.nan
    order = np.argsort(-score, kind="mergesort")
    out: dict[str, float] = {}
    for frac in fractions:
        k = max(1, int(np.ceil(frac * n)))
        selected = order[:k]
        tp = int(y[selected].sum())
        precision = tp / k
        recall = tp / n_pos if n_pos else np.nan
        out[f"precision_at_{int(frac * 100)}pct"] = precision
        out[f"recall_at_{int(frac * 100)}pct"] = recall
        out[f"enrichment_at_{int(frac * 100)}pct"] = precision / prevalence if prevalence else np.nan
    return out


def evaluate_scores(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    prevalence = n_pos / len(y) if len(y) else np.nan
    pr_auc = float(average_precision_score(y, score)) if n_pos and n_neg else np.nan
    metrics = {
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc - prevalence if np.isfinite(pr_auc) else np.nan,
        "roc_auc": _safe_roc_auc(y, score) if n_pos and n_neg else np.nan,
    }
    metrics.update(top_budget_metrics(y, score))
    return metrics


def percentile_rank_by_repeat(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    """Convert each model/estate/repeat score vector to average percentile rank."""
    def convert(s: pd.Series) -> pd.Series:
        r = s.rank(method="average", pct=True)
        return r
    return df.groupby(["stage", "estate", "model", "feature_set", "repeat"], sort=False)[score_col].transform(convert)


def smooth_scores_spatial(
    df: pd.DataFrame,
    score_col: str,
    *,
    sigma_m: float = 30.0,
    max_radius_m: float = 100.0,
) -> np.ndarray:
    """Gaussian-smooth scores within an estate without using labels."""
    out = np.full(len(df), np.nan, dtype=float)
    for _, idx in df.groupby("estate", sort=False).groups.items():
        idx = np.asarray(idx)
        g = df.loc[idx]
        lat0 = float(g["Lat"].mean())
        x = (g["Long"].to_numpy(float) - float(g["Long"].mean())) * 111_320.0 * np.cos(np.deg2rad(lat0))
        y = (g["Lat"].to_numpy(float) - float(g["Lat"].mean())) * 111_320.0
        coords = np.column_stack([x, y])
        tree = cKDTree(coords)
        values = g[score_col].to_numpy(float)
        for j, point in enumerate(coords):
            neigh = tree.query_ball_point(point, max_radius_m)
            dist = np.linalg.norm(coords[neigh] - point, axis=1)
            w = np.exp(-0.5 * (dist / sigma_m) ** 2)
            vals = values[neigh]
            out[idx[j]] = float(np.average(vals, weights=w))
    return out


def attach_metric_identity(
    metrics: dict[str, float],
    *,
    stage: str,
    cv: str,
    estate: str,
    model: str,
    feature_set: str,
    repeat: int,
) -> dict[str, float | str | int]:
    return {
        "stage": stage,
        "cv": cv,
        "estate": estate,
        "model": model,
        "feature_set": feature_set,
        "repeat": repeat,
        **metrics,
    }


def save_table(rows: list[dict], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def load_prepared_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = add_derived_features(df)
    df = add_local_residuals(df, BASE_FEATURES, k=10)
    df["y_true"] = (df["Class"] == "Unhealthy").astype(int)
    return df


FEATURE_SETS: dict[str, list[str]] = {
    "raw4": RAW_BANDS,
    "existing_indices": EXISTING_INDICES,
    "extended_indices": EXTENDED_INDICES,
    "all14": BASE_FEATURES,
    "all14_plus_ext": BASE_FEATURES + EXTENDED_INDICES,
    "log_raw4": LOG_RAW_FEATURES,
    "local14": LOCAL_BASE_FEATURES,
    "all14_plus_local": BASE_FEATURES + LOCAL_BASE_FEATURES,
    "raw4_plus_local": RAW_BANDS + LOCAL_RAW_FEATURES,
    "raw4_plus_ext": RAW_BANDS + EXTENDED_INDICES,
    "coords_only": COORD_FEATURES,
}
