"""P0 LimitedLabelsTriage: data freeze & audit (AirHitam only).
Joins reused RVI_NDVI CSVs + labels, verifies N/quadrants/missingness,
emits p0_audit.json + fig_p0_qa.png. No modelling, no GEE pull.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
OUT = ROOT / "misc" / "POC_Results" / "LimitedLabelsTriage"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(parents=True, exist_ok=True)

FEATS_SPEC = ["NDVI", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI"]

def main():
    rvi = pd.read_csv(SRC / "airhitam_rvi_features.csv")
    ndvi = pd.read_csv(SRC / "airhitam_ndvi.csv")
    qp = pd.read_csv(SRC / "airhitam_rvi_quadpol.csv")
    spec = pd.read_csv(SRC / "airhitam_spectral.csv")
    labels = pd.read_csv(ROOT / "data" / "Labels" / "AirHitam-Classification_2026.csv")
    m = (rvi.merge(ndvi, on=["id", "Class"], how="inner")
            .merge(qp[["id", "RVI_qp", "span"]], on="id", how="left")
            .merge(spec[["id", "NDRE5", "NDRE6", "NDRE7", "EVI", "NDMI"]], on="id", how="left")
            .merge(labels[["id", "Long", "Lat"]], on="id", how="left"))
    n_raw = len(m)
    m = m.dropna().reset_index(drop=True)
    y = (m["Class"] == "Unhealthy").to_numpy()
    # estate extent in meters
    MX = 111320 * np.cos(np.radians(m["Lat"].mean()))
    x_m = m["Long"].to_numpy() * MX
    y_m = m["Lat"].to_numpy() * 111320
    extent = {"lon_min": float(m["Long"].min()), "lon_max": float(m["Long"].max()),
              "lat_min": float(m["Lat"].min()), "lat_max": float(m["Lat"].max()),
              "span_x_m": float(x_m.max() - x_m.min()), "span_y_m": float(y_m.max() - y_m.min())}
    # quadrants via median split
    mlo, mla = float(m["Long"].median()), float(m["Lat"].median())
    quads = {}
    for q, mask in {"SW": (m["Long"] <= mlo) & (m["Lat"] <= mla),
                    "NW": (m["Long"] <= mlo) & (m["Lat"] > mla),
                    "SE": (m["Long"] > mlo) & (m["Lat"] <= mla),
                    "NE": (m["Long"] > mlo) & (m["Lat"] > mla)}.items():
        quads[q] = {"n": int(mask.sum()), "pos": int(y[mask.to_numpy()].sum())}
    # missingness / ranges
    audit = {"n_labels": int(len(labels)), "n_joined_raw": int(n_raw), "n_final": int(len(m)),
             "n_healthy": int((y == 0).sum()), "n_unhealthy": int(y.sum()),
             "pos_rate": round(float(y.mean()), 5), "extent_m": extent,
             "median_split": {"lon": mlo, "lat": mla}, "quadrants": quads,
             "ndvi_missing": int(m["NDVI"].isna().sum()),
             "ndvi_range": [float(m["NDVI"].min()), float(m["NDVI"].max())],
             "clear_count_range": [float(m["clear_count"].min()), float(m["clear_count"].max())],
             "spec_missing": {f: int(m[f].isna().sum()) for f in FEATS_SPEC},
             "frozen_sources": ["airhitam_rvi_features.csv", "airhitam_ndvi.csv",
                                "airhitam_rvi_quadpol.csv", "airhitam_spectral.csv"]}
    with open(OUT / "tables" / "p0_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))
    # QA figure: map coloured by class + NDVI hist
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(m["Long"][y == 0], m["Lat"][y == 0], s=4, alpha=0.4, label="Healthy")
    ax[0].scatter(m["Long"][y == 1], m["Lat"][y == 1], s=10, alpha=0.9, label="Unhealthy")
    ax[0].axvline(mlo, color="k", lw=0.8, ls="--"); ax[0].axhline(mla, color="k", lw=0.8, ls="--")
    ax[0].set_title(f"AirHitam N={len(m)} (160 Unhealthy) + median quadrants")
    ax[0].set_xlabel("Long"); ax[0].set_ylabel("Lat"); ax[0].legend(markerscale=3)
    ax[1].hist(m["NDVI"][y == 0], bins=40, alpha=0.6, label="Healthy")
    ax[1].hist(m["NDVI"][y == 1], bins=20, alpha=0.8, label="Unhealthy")
    ax[1].set_title("NDVI distribution (reused S2 composite)")
    ax[1].set_xlabel("NDVI"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_p0_qa.png", dpi=130)
    print("wrote tables/p0_audit.json + figs/fig_p0_qa.png")
    # frozen analysis frame for downstream stages
    m.to_csv(OUT / "tables" / "frame_frozen.csv", index=False)
    print("wrote tables/frame_frozen.csv", m.shape)

if __name__ == "__main__":
    main()
