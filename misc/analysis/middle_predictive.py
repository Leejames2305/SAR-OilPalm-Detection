"""
Can the Middle class improve H-vs-U detection? Two exploitations:
  E1) Target relabelling: is (Unhealthy+Middle) vs Healthy more separable than U vs H?
  E2) Training augmentation: does adding Middle (as weak positives) to training improve
      held-out H-vs-U detection under honest spatial-block CV?
Run: ./.venv/bin/python misc/analysis/middle_predictive.py
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

R = "/d/User_Stuff/Coding/MasterProject/SAR-OilPalm-Detection/misc/POC_Results/MeanSampling"
df = pd.read_csv(f"{R}/rvi_rfdi_means.csv")
for w in (3, 5):
    df[f"RFDI_VH_meanW{w}"] = (df[f"HH_meanW{w}"] - df[f"VH_meanW{w}"]) / (df[f"HH_meanW{w}"] + df[f"VH_meanW{w}"])
df["HV_VH_meanW3"] = df.HV_meanW3 / df.VH_meanW3

FEATS = (["RVI_meanW3", "RVI_meanW5", "RFDI_meanW3", "RFDI_meanW5",
          "RFDI_VH_meanW3", "RFDI_VH_meanW5"] +
         [f"{p}_meanW{w}" for p in ("HH", "HV", "VV", "VH") for w in (3, 5)] +
         ["HV_VH_meanW3"])

def auc(x_pos, x_neg):
    x_pos, x_neg = np.asarray(x_pos), np.asarray(x_neg)
    return float(mannwhitneyu(x_pos, x_neg, alternative="two-sided").statistic / (len(x_pos) * len(x_neg)))

print("=" * 96)
print("E1) Effect sizes: (U+M) vs H  compared with  U vs H   (per feature, per estate)")
print("=" * 96)
rows = []
for loc, sub in df.groupby("Location"):
    H = sub[sub.Class == "Healthy"]; U = sub[sub.Class == "Unhealthy"]; M = sub[sub.Class == "Middle"]
    for f in FEATS:
        h = H[f].dropna(); u = U[f].dropna(); m = M[f].dropna()
        um = pd.concat([u, m])
        sd_uh = np.sqrt((h.var(ddof=1) + u.var(ddof=1)) / 2)
        sd_umh = np.sqrt((h.var(ddof=1) + um.var(ddof=1)) / 2)
        rows.append({"loc": loc.replace("_Basemap", ""), "feature": f,
                     "d_U_vs_H": round((u.mean()-h.mean())/sd_uh, 3),
                     "d_UM_vs_H": round((um.mean()-h.mean())/sd_umh, 3),
                     "auc_U_vs_H": round(auc(u, h), 3),
                     "auc_UM_vs_H": round(auc(um, h), 3),
                     "auc_M_vs_H": round(auc(m, h), 3)})
t = pd.DataFrame(rows)
t["gain"] = (t.auc_UM_vs_H - 0.5).abs() - (t.auc_U_vs_H - 0.5).abs()
print(t.to_string(index=False))
print("\nfeatures where |AUC-0.5| improved with M folded in:",
      int((t.gain > 0).sum()), "/", len(t), "| median gain:", round(t.gain.median(), 4))

print("\n" + "=" * 96)
print("E2) Training augmentation under spatial-block CV (task on held-out: H vs U)")
print("=" * 96)
def blocks(lon, lat, size_deg):
    return pd.Series(np.floor(lon/size_deg).astype(int).astype(str) + "_" +
                     np.floor(lat/size_deg).astype(int).astype(str))

res = []
for loc, sub0 in df.groupby("Location"):
    sub = sub0.reset_index(drop=True)
    grp = blocks(sub.Long.values, sub.Lat.values, 0.002)
    gsz = grp.value_counts(); keep = grp.isin(gsz[gsz >= 20].index).to_numpy()
    sub = sub[keep].reset_index(drop=True); grp = grp[keep]
    X = sub[FEATS].to_numpy()
    isU = (sub.Class == "Unhealthy").to_numpy()
    isM = (sub.Class == "Middle").to_numpy()
    y_hu = isU.astype(int)
    y_hum = (isU | isM).astype(int)
    gk = GroupKFold(5)
    for mname, mk, sw_key in (("LogReg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), "logisticregression__sample_weight"),
                              ("HGB", lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.1, random_state=0), "sample_weight")):
        variants = {
            "baseline (train H vs U)":      (y_hu, None),
            "aug: M as positive w=0.5":      (np.where(isM, 1, y_hu), np.where(isM, 0.5, 1.0)),
            "aug: M as positive w=1.0":      (np.where(isM, 1, y_hu), None),
            "target: H vs (U+M) [relabel]":  (y_hum, None),
        }
        for vname, (ytr, wtr) in variants.items():
            aucs, prs = [], []
            eval_y = sub.Class.isin(["Healthy", "Unhealthy"]).to_numpy()
            for tr, te_all in gk.split(X, ytr, grp):
                te = te_all[eval_y[te_all]]          # score ONLY H/U trees on held-out blocks
                if len(np.unique(y_hu[te])) < 2: continue
                m = mk()
                if wtr is None:
                    m.fit(X[tr], ytr[tr])
                else:
                    m.fit(X[tr], ytr[tr], **{sw_key: wtr[tr]})
                p = m.predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(y_hu[te], p)); prs.append(average_precision_score(y_hu[te], p))
            res.append({"loc": loc.replace("_Basemap", ""), "model": mname, "strategy": vname,
                        "roc": round(float(np.mean(aucs)), 3), "pr": round(float(np.mean(prs)), 3)})
r = pd.DataFrame(res)
print("class prevalences: U-only 9.2% | U+M ~20.3% (=> no-skill PR baselines 0.092 / 0.203 for relabelled target)")
for loc in r["loc"].unique():
    print(f"\n--- {loc} ---")
    print(r[r["loc"] == loc].drop(columns="loc").to_string(index=False))
r.to_csv("/tmp/middle_predictive_results.csv", index=False)
