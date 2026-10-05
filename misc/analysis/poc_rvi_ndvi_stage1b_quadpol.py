"""Stage 1b of POC RVI+NDVI (AirHitam only, real ground truth).

True QUAD-pol eigenvalue RVI from the T3 coherency-matrix subset
(ALOS2-Subset_AirHitam_260610_Cal_mat_Spk_TC.tif, EPSG:4326, 9 bands):
  T3 = [[T11, T12, T13], [T12*, T22, T23], [T13*, T23*, T33]]
  band order (SNAP export): T11, T12re, T12im, T13re, T13im,
                            T22, T23re, T23im, T33
  RVI_qp = 4*lam3 / (lam1 + lam2 + lam3), lam1 >= lam2 >= lam3
  eigenvalues of the per-tree 3x3-mean coherency matrix.

Same 3x3 window philosophy as Stage 1; same univariate test battery
(Cohen d, Mann-Whitney U, ROC-AUC, PR-AUC vs 0.0637 no-skill).

Outputs (misc/POC_Results/RVI_NDVI/):
  airhitam_rvi_quadpol.csv, stage1b_summary.json,
  fig_rvi_qp_kde.png, fig_rvi_qp_box.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.windows import Window
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
OUT.mkdir(parents=True, exist_ok=True)

T3_PATH = ROOT / "data/SAR-scenes" / \
    "ALOS2-Subset_AirHitam_260610_Cal_mat_Spk_TC.tif"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"
WIN = 3


def sample_t3_means(tif_path, lon, lat, win=3):
    """Mean of each T3 band over a win x win window. Off-diagonals stay
    signed (only non-finite masked); full window must fit in raster."""
    out = np.full((len(lon), 9), np.nan)
    hw = win // 2
    with rasterio.open(tif_path) as src:
        assert src.count == 9, f"T3 band count {src.count}, expected 9"
        for i, (x, y) in enumerate(zip(lon, lat)):
            try:
                row, col = src.index(x, y)  # scene already EPSG:4326
            except Exception:
                continue
            r0, c0 = row - hw, col - hw
            if r0 < 0 or c0 < 0 or r0 + win > src.height \
                    or c0 + win > src.width:
                continue
            data = src.read(window=Window(c0, r0, win, win)).astype(float)
            for b in range(9):
                v = data[b][np.isfinite(data[b])]
                if v.size >= 5:
                    out[i, b] = float(v.mean())
    return out


def quadpol_rvi(t11, t12re, t12im, t13re, t13im, t22, t23re, t23im, t33):
    """Eigenvalue RVI from mean-T3 elements. Returns (rvi, lam1..3, span)."""
    t12 = t12re + 1j * t12im
    t13 = t13re + 1j * t13im
    t23 = t23re + 1j * t23im
    T = np.array([[t11, t12, t13],
                  [np.conj(t12), t22, t23],
                  [np.conj(t13), np.conj(t23), t33]], dtype=complex)
    lam = np.linalg.eigvalsh(T)[::-1]  # descending
    lam = np.clip(lam.real, 0.0, None)  # clamp numerical negatives
    span = lam.sum()
    if span <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return 4 * lam[2] / span, lam[0], lam[1], lam[2], span


def main():
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    print("trees:", len(labels))
    m = sample_t3_means(str(T3_PATH), labels["Long"].tolist(),
                        labels["Lat"].tolist(), WIN)
    n_ok = int(np.isfinite(m).all(axis=1).sum())
    print(f"T3 sampled: {n_ok}/{len(labels)} full-window trees")

    recs = []
    for row in m:
        if not np.isfinite(row).all():
            recs.append((np.nan,) * 5)
        else:
            recs.append(quadpol_rvi(*row))
    recs = np.array(recs)
    df = pd.DataFrame({"id": labels["id"], "Class": labels["Class"],
                       "y": (labels["Class"] == UNHEALTHY).astype(int),
                       "RVI_qp": recs[:, 0], "lam1": recs[:, 1],
                       "lam2": recs[:, 2], "lam3": recs[:, 3],
                       "span": recs[:, 4],
                       "T11": m[:, 0], "T22": m[:, 5], "T33": m[:, 8]})
    df.to_csv(OUT / "airhitam_rvi_quadpol.csv", index=False)
    pos_rate = float((df["Class"] == UNHEALTHY).mean())

    def univariate(h, u):
        h = pd.to_numeric(h, errors="coerce").dropna()
        u = pd.to_numeric(u, errors="coerce").dropna()
        sd_pool = float(np.sqrt((float(h.var(ddof=1)) +
                                 float(u.var(ddof=1))) / 2.0))
        dd = (float(u.mean()) - float(h.mean())) / sd_pool if sd_pool > 0 \
            else float("nan")
        try:
            p = float(mannwhitneyu(h, u, method="asymptotic").pvalue)
        except Exception:
            p = float("nan")
        yt = np.array([0] * len(h) + [1] * len(u))
        ys = np.concatenate([h.to_numpy(), u.to_numpy()])
        roc = float(roc_auc_score(yt, ys)) if len(np.unique(ys)) > 1 \
            else float("nan")
        ap = float(average_precision_score(yt, ys if roc >= 0.5 else -ys))
        return dict(n_h=len(h), n_u=len(u),
                    mean_h=round(float(h.mean()), 5),
                    mean_u=round(float(u.mean()), 5),
                    median_h=round(float(h.median()), 5),
                    median_u=round(float(u.median()), 5),
                    cohen_d=round(dd, 4), mwu_p=p,
                    roc_auc=round(roc, 4), pr_auc=round(ap, 4),
                    pr_lift=round(ap - pos_rate, 4))

    rows = {}
    for f in ["RVI_qp", "span", "lam1", "lam2", "lam3",
              "T11", "T22", "T33"]:
        sub = df[[f, "Class"]].dropna()
        rows[f] = univariate(sub.loc[sub.Class == HEALTHY, f],
                             sub.loc[sub.Class == UNHEALTHY, f])
    tab = pd.DataFrame(rows).T
    print(tab.to_string())
    print(tab.to_csv(OUT / "stage1b_univariate.csv"))
    (OUT / "stage1b_summary.json").write_text(json.dumps(
        {"t3_scene": T3_PATH.name, "window": WIN, "n_ok": n_ok,
         "n_total": len(labels), "pos_rate": pos_rate,
         "features": rows}, indent=2))

    pal = {HEALTHY: "#2ca02c", UNHEALTHY: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, c in pal.items():
        v = df.loc[df.Class == cls, "RVI_qp"].dropna()
        if v.nunique() > 1:
            sns.kdeplot(v, ax=ax,
                        label=f"{cls} (n={(df.Class == cls).sum()})",
                        color=c, fill=True, alpha=0.3)
    ax.set_title("AirHitam: quad-pol eigenvalue RVI by class "
                 f"(T3, 3x3 mean, n={n_ok})")
    ax.set_xlabel("RVI_qp = 4*lam3 / span")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_rvi_qp_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    melt = df.melt(id_vars="Class", value_vars=["RVI_qp", "span"],
                   var_name="feat", value_name="value").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=melt, x="feat", y="value", hue="Class", palette=pal,
                ax=ax, showfliers=False)
    ax.set_title("AirHitam: quad-pol RVI / span spread by class")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rvi_qp_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()
