"""Stage 0 of POC RVI+NDVI (AirHitam only). Samples per-tree 3x3 means from
the RE-EXPORTED intensity scene (no aggressive multilooking):
  data/SAR-scenes/ALOS2-Subset_AirHitam_260610_Cal_Spk_TC.tif
  4 bands [HH, HV, VH, VV] (positional, same SNAP chain as before),
  EPSG:4326, ~5.1 m grid, 444x578 (coregistered with the T3 subset).

Linear-power conventions mirror Stage 1: non-finite and <= 0 masked,
full 3x3 window must fit, mean of valid pixels (require >= 5).

Output (misc/POC_Results/RVI_NDVI/): airhitam_intensity_new.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"
OUT.mkdir(parents=True, exist_ok=True)

SCENE = ROOT / "data/SAR-scenes" / \
    "ALOS2-Subset_AirHitam_260610_Cal_Spk_TC.tif"
ORDER = ["HH", "HV", "VH", "VV"]
WIN = 3


def main():
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    lon, lat = labels["Long"].tolist(), labels["Lat"].tolist()
    out = np.full((len(lon), 4), np.nan)
    hw = WIN // 2
    with rasterio.open(SCENE) as src:
        assert src.count == 4, f"band count {src.count}, expected 4"
        print("scene:", SCENE.name, "crs:", src.crs, "grid:",
              src.width, "x", src.height)
        for i, (x, y) in enumerate(zip(lon, lat)):
            row, col = src.index(x, y)  # scene already EPSG:4326
            r0, c0 = row - hw, col - hw
            if r0 < 0 or c0 < 0 or r0 + WIN > src.height \
                    or c0 + WIN > src.width:
                continue
            data = src.read(window=Window(c0, r0, WIN, WIN)).astype(float)
            data[~np.isfinite(data)] = np.nan
            data[data <= 0.0] = np.nan  # linear power: no <= 0 backscatter
            for b in range(4):
                v = data[b][np.isfinite(data[b])]
                if v.size >= 5:
                    out[i, b] = float(v.mean())
    df = pd.DataFrame({"id": labels["id"], "Long": labels["Long"],
                       "Lat": labels["Lat"], "Class": labels["Class"], "y": (labels["Class"] == "Unhealthy").astype(int)})
    for b, pol in enumerate(ORDER):
        df[f"{pol}_mean"] = out[:, b]
    n_ok = int(np.isfinite(out).all(axis=1).sum())
    print(f"sampled {n_ok}/{len(df)} full-window trees")
    df.to_csv(OUT / "airhitam_intensity_new.csv", index=False)
    print("wrote", OUT / "airhitam_intensity_new.csv")


if __name__ == "__main__":
    main()

