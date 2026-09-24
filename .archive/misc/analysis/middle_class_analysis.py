"""
Middle-class investigation for the Mean-Sampling POC.

Question: can the 'Middle' health class (871 trees: Palong 493, Serting 378) be leveraged
to strengthen the statistical case for detecting Unhealthy trees?

Three hypotheses about what 'Middle' is, each with a falsifiable test:
  H1 intermediate-health-state : class medians order H -> M -> U on health-informative
                                 features, and Middle trees sit spatially closer to
                                 Unhealthy patches than Healthy trees do (disease front).
  H2 label noise               : Middle behaves like a ~50/50 random mixture of H and U
                                 distributions, with no spatial structure.
  H3 independent structural class : Middle is spatially clustered AWAY from Unhealthy
                                 and shows no ordered relationship to H/U.

Run with the repo venv:  ./.venv/bin/python misc/analysis/middle_class_analysis.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, norm
from scipy.spatial import cKDTree

R = "/d/User_Stuff/Coding/MasterProject/SAR-OilPalm-Detection/misc/POC_Results/MeanSampling"

df = pd.read_csv(f"{R}/rvi_rfdi_means.csv")

# extra features of interest (VH-based ratio family - best-performing in prior POCs)
for w in (3, 5):
    df[f"RFDI_VH_meanW{w}"] = (df[f"HH_meanW{w}"] - df[f"VH_meanW{w}"]) / (df[f"HH_meanW{w}"] + df[f"VH_meanW{w}"])
df["HV_VH_meanW3"] = df.HV_meanW3 / df.VH_meanW3

FEATURES = ["RVI_meanW3", "RVI_meanW5", "RFDI_meanW3", "RFDI_meanW5",
            "RFDI_VH_meanW3", "RFDI_VH_meanW5",
            "HH_meanW3", "HV_meanW3", "VV_meanW3", "VH_meanW3",
            "HH_meanW5", "HV_meanW5", "VV_meanW5", "VH_meanW5", "HV_VH_meanW3"]

def auc(x_pos, x_neg):
    x_pos, x_neg = np.asarray(x_pos), np.asarray(x_neg)
    n1, n0 = len(x_pos), len(x_neg)
    return float(mannwhitneyu(x_pos, x_neg, alternative="two-sided").statistic / (n1 * n0))

def jonckheere(x_h, x_m, x_u):
    """Jonckheere-Terpstra ordered-alternative test for H < M < U (two-sided p)."""
    groups = [np.asarray(x_h), np.asarray(x_m), np.asarray(x_u)]
    JT = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = groups[i], groups[j]
            JT += mannwhitneyu(b, a, alternative="greater").statistic
    n = sum(len(g) for g in groups)
    E = (n**2 - sum(len(g)**2 for g in groups)) / 4.0
    var = (n**3 - sum(len(g)**3 for g in groups)) / 36.0
    z = (JT - E - 0.5) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return JT, z, p

print("=" * 100)
print("A) Is Middle ORDINALLY between Healthy and Unhealthy?  (H1 expects: yes, consistently)")
print("=" * 100)
for loc, sub in df.groupby("Location"):
    H = sub[sub.Class == "Healthy"]; M = sub[sub.Class == "Middle"]; U = sub[sub.Class == "Unhealthy"]
    print(f"\n--- {loc}  (nH={len(H)}, nM={len(M)}, nU={len(U)}) ---")
    print(f"{'feature':<18}{'med_H':>9}{'med_M':>9}{'med_U':>9} | {'AUC(M,H)':>8} {'AUC(U,H)':>8} {'AUC(U,M)':>8} | {'JT z':>6} {'JT p':>8} {'orderOK':>8}")
    for f in FEATURES:
        xh, xm, xu = H[f].dropna(), M[f].dropna(), U[f].dropna()
        a_mh, a_uh, a_um = auc(xm, xh), auc(xu, xh), auc(xu, xm)
        _, z, p = jonckheere(xh, xm, xu)
        d_uh = 1 if a_uh > 0.5 else -1
        ok = (0.5 < a_mh < a_uh and a_um > 0.5) if d_uh > 0 else (0.5 > a_mh > a_uh and a_um < 0.5)
        print(f"{f:<18}{xh.median():>9.4f}{xm.median():>9.4f}{xu.median():>9.4f} | "
              f"{a_mh:>8.3f} {a_uh:>8.3f} {a_um:>8.3f} | {z:>6.2f} {p:>8.3g} {str(ok):>8}")

# ------------------------------------------------------------- B) mixture test
print("\n" + "=" * 100)
print("B) Is Middle a random MIXTURE of H and U?  (H2 expects mixing weight ~0.5 and much better")
print("   fit of the mixture than either pure class)")
print("=" * 100)
rng = np.random.default_rng(0)
for loc, sub in df.groupby("Location"):
    H = sub[sub.Class == "Healthy"]; M = sub[sub.Class == "Middle"]; U = sub[sub.Class == "Unhealthy"]
    print(f"\n--- {loc} ---")
    print(f"{'feature':<18}{'KS(M,H)':>9}{'KS(M,U)':>9}{'KS*(mix)':>10}{'pi*':>7}{'verdict':>24}")
    for f in FEATURES:
        xh = H[f].dropna().to_numpy(); xm = M[f].dropna().to_numpy(); xu = U[f].dropna().to_numpy()
        ks_h = ks_2samp(xm, xh).statistic
        ks_u = ks_2samp(xm, xu).statistic
        best = (9.0, -1.0)
        for pi in np.arange(0.0, 1.001, 0.05):
            ks_sum = 0.0
            for _ in range(30):
                take_u = rng.random(len(xm)) < pi
                mix = np.where(take_u, rng.choice(xu, len(xm)), rng.choice(xh, len(xm)))
                ks_sum += ks_2samp(xm, mix).statistic
            ks_avg = ks_sum / 30
            if ks_avg < best[0]:
                best = (ks_avg, pi)
        verdict = "noise-mixture ~50/50" if (abs(best[1]-0.5) <= 0.15 and best[0] < min(ks_h, ks_u)) else \
                  ("looks Healthy-like" if best[1] <= 0.15 else
                   ("looks Unhealthy-like" if best[1] >= 0.85 else "partial/unresolved"))
        print(f"{f:<18}{ks_h:>9.3f}{ks_u:>9.3f}{best[0]:>10.3f}{best[1]:>7.2f}{verdict:>24}")

# ------------------------------------------------------- C) spatial transition test
print("\n" + "=" * 100)
print("C) Spatial transition-zone test  (H1 expects Middle closer to Unhealthy than Healthy is)")
print("=" * 100)
KM_PER_DEG = 111.32
for loc, sub in df.groupby("Location"):
    sub = sub.reset_index(drop=True)
    x = sub.Long.to_numpy() * KM_PER_DEG * np.cos(np.radians(sub.Lat.to_numpy()))
    y = sub.Lat.to_numpy() * KM_PER_DEG
    pts = np.column_stack([x, y])
    isU = (sub.Class == "Unhealthy").to_numpy()
    isM = (sub.Class == "Middle").to_numpy()
    isH = (sub.Class == "Healthy").to_numpy()
    treeU = cKDTree(pts[isU])
    d_M_to_U = treeU.query(pts[isM])[0]
    d_H_to_U = treeU.query(pts[isH])[0]
    treeH = cKDTree(pts[isH])
    d_M_to_H = treeH.query(pts[isM])[0]
    a_mh_dist = auc(d_M_to_U, d_H_to_U)   # <0.5 => Middle nearer to U than Healthy is
    print(f"\n--- {loc} ---")
    print(f"  median dist Middle->nearest U : {np.median(d_M_to_U)*1000:7.1f} m")
    print(f"  median dist Healthy->nearest U: {np.median(d_H_to_U)*1000:7.1f} m")
    print(f"  median dist Middle->nearest H : {np.median(d_M_to_H)*1000:7.1f} m")
    print(f"  AUC( dist-to-nearest-U | M vs H ) = {a_mh_dist:.3f}   (<0.5 => Middle is a U-side transition zone)")
    for label, mask_from in (("Middle", isM), ("Healthy", isH)):
        cnt = treeU.query_ball_point(pts[mask_from], r=0.030)
        dens = np.mean([len(c) for c in cnt])
        print(f"  mean #U within 30 m of a {label:7s} tree: {dens:6.2f}")

# ------------------------------------------------------- D) Middle spatial clustering
print("\n" + "=" * 100)
print("D) Is Middle itself spatially clustered?  (H2 expects no; H3 expects yes, away from U)")
print("=" * 100)
for loc, sub in df.groupby("Location"):
    for cls in ("Healthy", "Middle", "Unhealthy"):
        s = sub[sub.Class == cls].reset_index(drop=True)
        blk = pd.Series(np.floor(s.Long.values / 0.002).astype(int).astype(str) + "_"
                        + np.floor(s.Lat.values / 0.002).astype(int).astype(str))
        g = s.assign(blk=blk).groupby("blk").size()
        g = g[g >= 10]
        if len(g) < 2:
            print(f"  {loc:16s} {cls:10s}: too few dense blocks"); continue
        lam = g.mean()
        disp = g.var() / lam
        print(f"  {loc:16s} {cls:10s}: {len(g):3d} blocks(>=10), mean/block={lam:6.1f}, var/mean={disp:6.2f} (>1 = clustered)")
