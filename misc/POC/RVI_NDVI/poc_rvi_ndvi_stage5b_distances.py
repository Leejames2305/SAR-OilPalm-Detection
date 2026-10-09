"""Stage 5b: nearest-train-distance diagnostic for the block-size debate.

For each protocol (random / K100 / K50 / K15 / K5), over its actual CV
splits, compute per-test-tree distance (metres) to the nearest train tree.
Reports median + % of test trees within 20 m / 50 m of a train tree.
If fine blocks still leave most test trees metres from train trees,
their leakage control is ~random CV, whatever the block diameter.

Output: stage5b_distances.json (printed + saved).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "RVI_NDVI"

MX = 111320 * np.cos(np.radians(2.947))
MY = 111320


def main():
    labels = pd.read_csv(ROOT / "data/Labels/AirHitam-Classification_2026.csv")
    xy = np.column_stack([labels["Long"].to_numpy() * MX,
                          labels["Lat"].to_numpy() * MY])
    y = (labels["Class"] == "Unhealthy").to_numpy()
    out = {}
    cfgs = [("random", None), ("K100", 100), ("K50", 50), ("K15", 15),
            ("K5", 5)]
    for name, k in cfgs:
        if k is None:
            splits = list(RepeatedStratifiedKFold(
                n_splits=5, n_repeats=3,
                random_state=7).split(xy, y))
        else:
            grp = KMeans(n_clusters=k, n_init=10,
                         random_state=42).fit_predict(
                             np.column_stack([labels["Long"], labels["Lat"]]))
            splits = list(StratifiedGroupKFold(n_splits=5).split(xy, y, grp))
        dmins = []
        for tr, te in splits:
            dmins.append(cdist(xy[te], xy[tr]).min(axis=1))
        dmins = np.concatenate(dmins)
        out[name] = dict(median_m=round(float(np.median(dmins)), 1),
                         mean_m=round(float(dmins.mean()), 1),
                         pct_within_20m=round(float((dmins < 20).mean()),
                                              3),
                         pct_within_50m=round(float((dmins < 50).mean()),
                                              3))
        print(name, out[name])
    (OUT / "stage5b_distances.json").write_text(json.dumps(out, indent=2))
    print("saved to", OUT / "stage5b_distances.json")


if __name__ == "__main__":
    main()
