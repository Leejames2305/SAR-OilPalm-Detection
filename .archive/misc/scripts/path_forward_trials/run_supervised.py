"""Run supervised within-estate and cross-estate path trials."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _scale_pipe(clf, scaler=None):
    return Pipeline([("scale", scaler or RobustScaler()), ("clf", clf)])


def make_supervised_models(seed: int, n_pos: int, n_neg: int) -> dict[str, object]:
    pos_weight = max(n_neg / max(n_pos, 1), 1.0)
    return {
        "logreg_l2": _scale_pipe(
            LogisticRegression(
                C=0.5,
                l1_ratio=0.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            )
        ),
        "logreg_l1": _scale_pipe(
            LogisticRegression(
                C=0.2,
                l1_ratio=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=seed,
            )
        ),
        "svm_rbf": _scale_pipe(
            SVC(
                C=3.0,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                cache_size=1000,
            )
        ),
        "knn_25": _scale_pipe(
            KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
        ),
        "lda_shrink": _scale_pipe(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            StandardScaler(),
        ),
        "gaussian_nb": _scale_pipe(GaussianNB(), StandardScaler()),
        "rf_balanced": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "extratrees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gbdt_balanced": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        ),
        "xgboost_balanced": XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
        "mlp_64_16": _scale_pipe(
            MLPClassifier(
                hidden_layer_sizes=(64, 16),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                max_iter=500,
                random_state=seed,
            ),
            StandardScaler(),
        ),
    }


def _summarize(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    metric_cols = [
        c for c in metrics.columns
        if c not in {"stage", "cv", "estate", "model", "feature_set", "repeat"}
    ]
    rows = []
    for keys, g in metrics.groupby(["stage", "cv", "estate", "model", "feature_set"], sort=False):
        row = dict(zip(["stage", "cv", "estate", "model", "feature_set"], keys))
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def run_within_estate(df: pd.DataFrame, *, feature_sets: list[str], quick: bool = False) -> None:
    all_models = make_supervised_models(tc.SEED, 1, 1)
    primary_models = list(all_models)
    secondary_models = ["logreg_l2", "xgboost_balanced"]
    local_sets = {"local14", "all14_plus_local", "raw4_plus_local"}

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    start = time.time()

    for estate, group in df.groupby("Location", sort=True):
        estate_df = group.reset_index(drop=False).rename(columns={"index": "source_row"})
        folds = tc.make_folds(estate_df, n_repeats=1 if quick else tc.N_REPEATS)
        for fs in feature_sets:
            cols = tc.FEATURE_SETS[fs]
            if fs == "all14":
                model_names = primary_models
            else:
                model_names = list(secondary_models)
                if fs in local_sets:
                    model_names.append("rf_balanced")
            y = estate_df["y_true"].to_numpy(int)
            x = estate_df[cols].to_numpy(float)
            print(f"[supervised] {estate} / {fs}: {len(model_names)} models, {len(set(f.repeat for f in folds))} repeats", flush=True)

            for repeat in sorted(set(f.repeat for f in folds)):
                repeat_folds = [f for f in folds if f.repeat == repeat]
                oof = {name: np.full(len(estate_df), np.nan) for name in model_names}
                fold_id = np.full(len(estate_df), -1, int)
                for fold in repeat_folds:
                    tr, te = fold.train, fold.test
                    n_pos = int(y[tr].sum())
                    n_neg = int(len(tr) - n_pos)
                    models = make_supervised_models(tc.SEED + repeat * 101 + fold.fold, n_pos, n_neg)
                    for name in model_names:
                        model = models[name]
                        model.fit(x[tr], y[tr])
                        score = tc.estimator_scores(model, x[te])
                        if not np.all(np.isfinite(score)):
                            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
                        oof[name][te] = score
                        fold_id[te] = fold.fold
                for name in model_names:
                    score = oof[name]
                    if not np.all(np.isfinite(score)):
                        raise RuntimeError(f"non-finite OOF scores: {estate}/{fs}/{name}")
                    metrics = tc.evaluate_scores(y, score)
                    metric_rows.append(
                        tc.attach_metric_identity(
                            metrics,
                            stage="supervised",
                            cv="spatial_5fold_repeated",
                            estate=estate,
                            model=name,
                            feature_set=fs,
                            repeat=repeat,
                        )
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "stage": "supervised",
                                "cv": "spatial_5fold_repeated",
                                "estate": estate,
                                "model": name,
                                "feature_set": fs,
                                "repeat": repeat,
                                "source_row": estate_df["source_row"].to_numpy(int),
                                "id": estate_df["id"].to_numpy(int),
                                "Long": estate_df["Long"].to_numpy(float),
                                "Lat": estate_df["Lat"].to_numpy(float),
                                "block": estate_df["block"].astype(str).to_numpy(),
                                "fold": fold_id,
                                "y_true": y,
                                "score": score,
                            }
                        )
                    )

    metrics = tc.save_table(metric_rows, tc.RESULT_DIR / "supervised_repeat_metrics.csv")
    _summarize(metrics, tc.RESULT_DIR / "supervised_summary.csv")
    preds = pd.concat(prediction_frames, ignore_index=True)
    preds.to_csv(tc.RESULT_DIR / "supervised_oof_scores.csv", index=False)
    print(f"[supervised] wrote {len(metrics)} metric rows and {len(preds)} OOF rows in {time.time()-start:.1f}s", flush=True)


def _estate_standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for _, idx in out.groupby("Location", sort=False).groups.items():
        x = out.loc[idx, cols].to_numpy(float)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        out.loc[idx, cols] = (x - mu) / sd
    return out


def run_transfer(df: pd.DataFrame) -> None:
    estates = sorted(df["Location"].unique())
    model_names = ["logreg_l2", "rf_balanced", "xgboost_balanced"]
    feature_sets = ["all14", "all14_plus_ext", "all14_plus_local", "local14"]
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    start = time.time()

    for test_estate in estates:
        train_base = df[df["Location"] != test_estate]
        test_base = df[df["Location"] == test_estate]
        for fs in feature_sets:
            variants = [(fs, train_base, test_base)]
            # Estate-wise z-scoring is a domain-normalisation alternative for
            # applying a model to an unseen acquisition/estate.
            z_train = _estate_standardize(train_base, tc.FEATURE_SETS[fs])
            z_test = _estate_standardize(test_base, tc.FEATURE_SETS[fs])
            variants.append((fs + "_estate_z", z_train, z_test))
            for variant, tr_df, te_df in variants:
                x_tr = tr_df[tc.FEATURE_SETS[fs]].to_numpy(float)
                x_te = te_df[tc.FEATURE_SETS[fs]].to_numpy(float)
                y_tr = tr_df["y_true"].to_numpy(int)
                y_te = te_df["y_true"].to_numpy(int)
                n_pos = int(y_tr.sum())
                n_neg = int(len(y_tr) - n_pos)
                models = make_supervised_models(tc.SEED, n_pos, n_neg)
                print(f"[transfer] holdout={test_estate} / {variant}", flush=True)
                for name in model_names:
                    model = models[name]
                    model.fit(x_tr, y_tr)
                    score = tc.estimator_scores(model, x_te)
                    metrics = tc.evaluate_scores(y_te, score)
                    metric_rows.append(
                        tc.attach_metric_identity(
                            metrics,
                            stage="cross_estate",
                            cv="leave_one_estate_out",
                            estate=test_estate,
                            model=name,
                            feature_set=variant,
                            repeat=0,
                        )
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "stage": "cross_estate",
                                "cv": "leave_one_estate_out",
                                "estate": test_estate,
                                "model": name,
                                "feature_set": variant,
                                "repeat": 0,
                                "source_row": te_df.index.to_numpy(int),
                                "id": te_df["id"].to_numpy(int),
                                "Long": te_df["Long"].to_numpy(float),
                                "Lat": te_df["Lat"].to_numpy(float),
                                "block": te_df["block"].astype(str).to_numpy(),
                                "fold": -1,
                                "y_true": y_te,
                                "score": score,
                            }
                        )
                    )
    metrics = tc.save_table(metric_rows, tc.RESULT_DIR / "cross_estate_metrics.csv")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        tc.RESULT_DIR / "cross_estate_oof_scores.csv", index=False
    )
    print(f"[transfer] wrote {len(metrics)} rows in {time.time()-start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["within", "transfer", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="one repeat; for smoke tests only")
    args = parser.parse_args()

    tc.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = tc.load_prepared_data()
    if args.stage in {"within", "all"}:
        feature_sets = [
            "all14",
            "raw4",
            "existing_indices",
            "extended_indices",
            "all14_plus_ext",
            "log_raw4",
            "local14",
            "all14_plus_local",
            "raw4_plus_local",
            "raw4_plus_ext",
            "coords_only",
        ]
        run_within_estate(df, feature_sets=feature_sets, quick=args.quick)
    if args.stage in {"transfer", "all"}:
        run_transfer(df)


if __name__ == "__main__":
    main()
