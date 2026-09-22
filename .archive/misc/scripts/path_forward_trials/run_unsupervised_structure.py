"""Unsupervised structure diagnostics: clustering and label autocorrelation."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def neighbor_label_autocorrelation(g: pd.DataFrame, k: int = 10, permutations: int = 500, seed: int = 42) -> dict:
    lat0 = float(g["Lat"].mean())
    x = (g["Long"].to_numpy(float) - float(g["Long"].mean())) * 111_320.0 * np.cos(np.deg2rad(lat0))
    y = (g["Lat"].to_numpy(float) - float(g["Lat"].mean())) * 111_320.0
    tree = cKDTree(np.column_stack([x, y]))
    _, inds = tree.query(np.column_stack([x, y]), k=min(k + 1, len(g)))
    inds = inds[:, 1:]
    y_true = g["y_true"].to_numpy(int)
    neigh_rate = y_true[inds].mean(axis=1)

    observed = float(neigh_rate[y_true == 1].mean() - neigh_rate[y_true == 0].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        shuffled = rng.permutation(y_true)
        nr = shuffled[inds].mean(axis=1)
        null[i] = nr[shuffled == 1].mean() - nr[shuffled == 0].mean()
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return {
        "estate": g["Location"].iloc[0],
        "k": k,
        "pos_rate": float(y_true.mean()),
        "neighbor_u_rate_if_u": float(neigh_rate[y_true == 1].mean()),
        "neighbor_u_rate_if_h": float(neigh_rate[y_true == 0].mean()),
        "difference": observed,
        "permutation_p_one_sided": p_value,
    }


def cluster_diagnostics(g: pd.DataFrame, feature_set: str, cols: list[str]) -> tuple[list[dict], list[dict]]:
    x = StandardScaler().fit_transform(g[cols].to_numpy(float))
    y = g["y_true"].to_numpy(int)
    kmeans_rows: list[dict] = []
    gmm_rows: list[dict] = []
    sample_n = min(1500, len(g))
    rng = np.random.default_rng(tc.SEED)
    sample_idx = rng.choice(len(g), size=sample_n, replace=False)

    for k in range(2, 13):
        model = KMeans(n_clusters=k, n_init=20, random_state=tc.SEED + k).fit(x)
        labels = model.labels_
        sizes = np.bincount(labels, minlength=k)
        rates = np.array([y[labels == j].mean() if sizes[j] else 0.0 for j in range(k)])
        eligible = sizes >= 5
        best = int(np.argmax(np.where(eligible, rates, -1)))
        kmeans_rows.append(
            {
                "estate": g["Location"].iloc[0],
                "feature_set": feature_set,
                "method": "kmeans",
                "n_clusters": k,
                "ari": adjusted_rand_score(y, labels),
                "ami": adjusted_mutual_info_score(y, labels),
                "silhouette_sample": silhouette_score(x[sample_idx], labels[sample_idx]),
                "max_cluster_size": int(sizes[best]),
                "max_cluster_unhealthy_rate": float(rates[best]),
                "max_cluster_enrichment": float(rates[best] / y.mean()) if y.mean() else np.nan,
                "unhealthy_in_max_cluster": int((y[labels == best] == 1).sum()),
                "total_unhealthy": int(y.sum()),
            }
        )

    for k in range(2, 9):
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=5,
            max_iter=300,
            random_state=tc.SEED + k,
        ).fit(x)
        labels = model.predict(x)
        sizes = np.bincount(labels, minlength=k)
        rates = np.array([y[labels == j].mean() if sizes[j] else 0.0 for j in range(k)])
        eligible = sizes >= 5
        best = int(np.argmax(np.where(eligible, rates, -1)))
        gmm_rows.append(
            {
                "estate": g["Location"].iloc[0],
                "feature_set": feature_set,
                "method": "gmm_diag",
                "n_clusters": k,
                "bic": model.bic(x),
                "aic": model.aic(x),
                "ari": adjusted_rand_score(y, labels),
                "ami": adjusted_mutual_info_score(y, labels),
                "max_cluster_size": int(sizes[best]),
                "max_cluster_unhealthy_rate": float(rates[best]),
                "max_cluster_enrichment": float(rates[best] / y.mean()) if y.mean() else np.nan,
                "unhealthy_in_max_cluster": int((y[labels == best] == 1).sum()),
                "total_unhealthy": int(y.sum()),
            }
        )
    return kmeans_rows, gmm_rows


def main() -> None:
    start = time.time()
    df = tc.load_prepared_data()
    tc.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    autocorr = [neighbor_label_autocorrelation(g) for _, g in df.groupby("Location", sort=True)]
    pd.DataFrame(autocorr).to_csv(tc.RESULT_DIR / "label_spatial_autocorrelation.csv", index=False)

    cluster_rows: list[dict] = []
    pca_rows: list[dict] = []
    for estate, g in df.groupby("Location", sort=True):
        print(f"[unsupervised] {estate}", flush=True)
        for fs in ["all14", "extended_indices", "local14", "all14_plus_local"]:
            km, gm = cluster_diagnostics(g, fs, tc.FEATURE_SETS[fs])
            cluster_rows.extend(km)
            cluster_rows.extend(gm)
        x = StandardScaler().fit_transform(g[tc.FEATURE_SETS["all14"]].to_numpy(float))
        pca = PCA(n_components=min(10, x.shape[1], len(x) - 1), random_state=tc.SEED).fit(x)
        pca_rows.append(
            {
                "estate": estate,
                "n": len(g),
                "explained_variance_pc1": pca.explained_variance_ratio_[0],
                "explained_variance_pc2": pca.explained_variance_ratio_[1],
                "explained_variance_pc1_5": pca.explained_variance_ratio_[:5].sum(),
                "explained_variance_all_10": pca.explained_variance_ratio_.sum(),
            }
        )
        scores = pca.transform(x)[:, :2]
        pd.DataFrame(
            {
                "estate": estate,
                "pc1": scores[:, 0],
                "pc2": scores[:, 1],
                "y_true": g["y_true"].to_numpy(int),
                "block": g["block"].astype(str).to_numpy(),
            }
        ).to_csv(tc.RESULT_DIR / f"pca_scores_{estate}.csv", index=False)

    pd.DataFrame(cluster_rows).to_csv(tc.RESULT_DIR / "unsupervised_cluster_metrics.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(tc.RESULT_DIR / "pca_summary.csv", index=False)
    print(f"[unsupervised] wrote diagnostics in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
