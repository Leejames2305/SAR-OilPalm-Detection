"""Stage 1 of POC RVI+NDVI (AirHitam only, real ground truth).

Computes quad/dual-pol RVI variants from ALOS (PALSAR) quad-pol band means
(HH, HV, VH, VV linear power, 3x3 mean per tree) and tests Healthy vs
Unhealthy separability with univariate statistics + a fixed simple
classifier probe. No spatial CV here: the question is purely whether the
index distributions differ by class.

RVI variants (all scale-invariant ratios of linear power):
  RVI_classic = 8*HV / (HH + VV + 2*HV)      # prior-POC convention (MeanSampling)
  RVI_xavg    = 8*X  / (HH + VV + 2*X), X=(HV+VH)/2
  RVI_hh      = 4*HV / (HH + HV)             # dual-pol HH-HV form
  RVI_vv      = 4*VH / (VV + VH)             # dual-pol VV-VH form
  RFDI        = (HH - HV) / (HH + HV)        # continuity w/ prior POC
Raw bands (HH/HV/VH/VV means) included as reference.

Outputs (misc/POC_RVI_NDVI/):
  airhitam_rvi_features.csv, stage1_summary.json,
  fig_rvi_kde.png, fig_rvi_box.png, fig_rvi_roc.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

sns.set_style("whitegrid")

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'misc' / 'POC_Results' / 'RVI_NDVI'
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("data/Processed/sampled_AirHitam_w3_mean_std_min_max_p25_p50.csv")
assert SRC.exists(), f"missing sampled dataset: {SRC}"

HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"

df = pd.read_csv(SRC)
print(f"loaded {SRC}: shape={df.shape}")
print(df["Class"].value_counts().to_string())

hh = pd.to_numeric(df["HH_mean"], errors="coerce")
hv = pd.to_numeric(df["HV_mean"], errors="coerce")
vh = pd.to_numeric(df["VH_mean"], errors="coerce")
vv = pd.to_numeric(df["VV_mean"], errors="coerce")
x = (hv + vh) / 2.0

feats = pd.DataFrame({
    "id": df["id"], "Class": df["Class"], "y": df["y"],
    "HH": hh, "HV": hv, "VH": vh, "VV": vv,
    "RVI_classic": 8 * hv / (hh + vv + 2 * hv),
    "RVI_xavg": 8 * x / (hh + vv + 2 * x),
    "RVI_hh": 4 * hv / (hh + hv),
    "RVI_vv": 4 * vh / (vv + vh),
    "RFDI": (hh - hv) / (hh + hv),
})
feats.to_csv(OUT / "airhitam_rvi_features.csv", index=False)

pos_rate = float((df["Class"] == UNHEALTHY).mean())
print(f"no-skill PR-AUC (prevalence) = {pos_rate:.4f}")

FEATURES = ["RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI",
            "HH", "HV", "VH", "VV"]

def univariate(h, u):
    h = pd.to_numeric(h, errors="coerce").dropna()
    u = pd.to_numeric(u, errors="coerce").dropna()
    mh, mu = float(h.mean()), float(u.mean())
    medh, medu = float(h.median()), float(u.median())
    sd_pool = float(np.sqrt((float(h.var(ddof=1)) + float(u.var(ddof=1))) / 2.0))
    d = (mu - mh) / sd_pool if sd_pool > 0 else float("nan")
    try:
        p = float(mannwhitneyu(h, u, method="asymptotic").pvalue)
    except Exception:
        p = float("nan")
    y_true = np.array([0] * len(h) + [1] * len(u))
    y_s = np.concatenate([h.to_numpy(), u.to_numpy()])
    roc = float(roc_auc_score(y_true, y_s)) if len(np.unique(y_s)) > 1 else float("nan")
    # best-direction PR-AUC (flip sign if AUC<0.5 so AP reflects ranking power)
    ap = float(average_precision_score(y_true, y_s if roc >= 0.5 else -y_s))
    return dict(n_h=len(h), n_u=len(u), mean_h=round(mh, 5), mean_u=round(mu, 5),
                median_h=round(medh, 5), median_u=round(medu, 5),
                cohen_d=round(d, 4), mwu_p=p,
                roc_auc=round(roc, 4), pr_auc=round(ap, 4),
                pr_lift=round(ap - pos_rate, 4))

rows = {}
for f in FEATURES:
    sub = feats[[f, "Class"]].dropna()
    rows[f] = univariate(sub.loc[sub.Class == HEALTHY, f],
                         sub.loc[sub.Class == UNHEALTHY, f])
tab = pd.DataFrame(rows).T
print(tab.to_string())
print(tab.to_csv(OUT / "stage1_univariate.csv"))

summary = {"n_healthy": int((df.Class == HEALTHY).sum()),
           "n_unhealthy": int((df.Class == UNHEALTHY).sum()),
           "pos_rate": pos_rate,
           "features": {k: v for k, v in rows.items()}}
(OUT / "stage1_summary.json").write_text(json.dumps(summary, indent=2))

# ---- Fig 1: KDE overlap per index (common x-scale per panel) ----
idx_feats = ["RVI_classic", "RVI_xavg", "RVI_hh", "RVI_vv", "RFDI"]
fig, axes = plt.subplots(1, len(idx_feats), figsize=(4 * len(idx_feats), 3.6))
pal = {HEALTHY: "#2ca02c", UNHEALTHY: "#d62728"}
for ax, f in zip(axes, idx_feats):
    for cls, c in pal.items():
        v = feats.loc[feats.Class == cls, f].dropna()
        if v.nunique() > 1:
            sns.kdeplot(v, ax=ax, label=f"{cls} (n={(feats.Class == cls).sum()})",
                        color=c, fill=True, alpha=0.3)
    ax.set_title(f)
    ax.set_xlabel(f)
    ax.legend(fontsize=8)
fig.suptitle("AirHitam: RVI/RFDI distributions Healthy vs Unhealthy (3x3 mean, n=2511)",
             y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_rvi_kde.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: boxplots ----
melt = feats.melt(id_vars="Class", value_vars=idx_feats,
                  var_name="index", value_name="value")
fig, ax = plt.subplots(figsize=(10, 4.2))
sns.boxplot(data=melt, x="index", y="value", hue="Class", palette=pal, ax=ax,
            showfliers=False)
ax.set_title("AirHitam: index spread by class (outliers hidden)")
fig.tight_layout()
fig.savefig(OUT / "fig_rvi_box.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: ROC curves (single-feature ranking power) ----
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot([0, 1], [0, 1], "k--", lw=1)
for f in idx_feats:
    sub = feats[[f, "y"]].dropna()
    auc = roc_auc_score(sub["y"], sub[f])
    s = sub[f] if auc >= 0.5 else -sub[f]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(sub["y"], s)
    ax.plot(fpr, tpr, label=f"{f} (AUC={max(auc, 1 - auc):.3f})")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("AirHitam: single-index ROC (direction-optimised)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_rvi_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("artifacts written to", OUT)


