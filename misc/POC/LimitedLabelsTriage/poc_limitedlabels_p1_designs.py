"""P1 LimitedLabelsTriage: sampling-design simulation at fixed budgets.
Train mask of exactly B trees -> predict rest. Model FIXED (balanced LogReg,
NDVI-only) to isolate the labelling-design effect. B in {1000,1200,1400,1600}.
Designs: A corner-block | B strip | C dispersed | D ndvi-stratified | E random.
Outputs: tables/p1_runs.csv, tables/p1_summary.csv, figs/fig_p1_designs.png, figs/fig_p1_maps.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "misc" / "POC_Results" / "LimitedLabelsTriage"
B_LEVELS = [1000, 1200, 1400, 1600]
POS_RATE = 160 / 2511

def recall_at_k(y_true, score, frac):
    order = np.argsort(-score)
    k = max(1, int(round(frac * len(y_true))))
    flagged = order[:k]
    pt = int(y_true.sum())
    if pt == 0:
        return float("nan")
    return float(y_true[flagged].sum() / pt)

def build_masks(df, xm, ym, ndvi, B, seed_base=0):
    N = len(df)
    masks = {}  # key -> (design, rep_id, boolean mask)
    # A. corner blocks: 4 extreme seeds, B nearest in meters
    corners = {"SW": (xm.min(), ym.min()), "NW": (xm.min(), ym.max()),
               "SE": (xm.max(), ym.min()), "NE": (xm.max(), ym.max())}
    for c, (sx, sy) in corners.items():
        d = np.sqrt((xm - sx) ** 2 + (ym - sy) ** 2)
        idx = np.argsort(d)[:B]
        mk = np.zeros(N, bool); mk[idx] = True
        masks[f"A_block_{c}"] = ("A_block", c, mk)
    # B. strips: contiguous windows in x (N-S strips) and y (E-W strips)
    for axis, vals in (("lon", xm), ("lat", ym)):
        order = np.argsort(vals)
        for qi, q in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            c = int(round(q * N))
            lo, hi = max(0, c - B // 2), max(0, c - B // 2) + B
            if hi > N:
                hi, lo = N, N - B
            idx = order[lo:hi]
            mk = np.zeros(N, bool); mk[idx] = True
            masks[f"B_strip_{axis}_{qi}"] = ("B_strip", f"{axis}{qi}", mk)
    # C. dispersed mini-blocks: KMeans k=5, 10 seeds, equal share per cluster
    for r in range(10):
        km = KMeans(n_clusters=5, n_init=10, random_state=1000 + r).fit(np.column_stack([xm, ym]))
        lab = km.labels_
        mk = np.zeros(N, bool)
        shares = [B // 5] * 5
        for i in range(B % 5):
            shares[i] += 1
        for k in range(5):
            members = np.where(lab == k)[0]
            if len(members) == 0:
                continue
            ck = km.cluster_centers_[k]
            d = np.sqrt((xm[members] - ck[0]) ** 2 + (ym[members] - ck[1]) ** 2)
            take = min(shares[k], len(members))
            mk[members[np.argsort(d)[:take]]] = True
        # exact-size fix (rounding/cluster imbalance)
        cur = mk.sum()
        if cur < B:
            rest = np.where(~mk)[0]
            ck_all = np.column_stack([xm, ym]).mean(axis=0)
            d = np.sqrt((xm[rest] - ck_all[0]) ** 2 + (ym[rest] - ck_all[1]) ** 2)
            mk[rest[np.argsort(d)[:B - cur]]] = True
        elif cur > B:
            on = np.where(mk)[0]
            ck_all = np.column_stack([xm, ym]).mean(axis=0)
            d = np.sqrt((xm[on] - ck_all[0]) ** 2 + (ym[on] - ck_all[1]) ** 2)
            mk[on[np.argsort(-d)[:cur - B]]] = False
        masks[f"C_disp_{r}"] = ("C_dispersed", str(r), mk)
    # D. NDVI-stratified proportional draw, 30 reps
    strata = pd.qcut(ndvi, 5, labels=False, duplicates="drop")
    for r in range(30):
        rng = np.random.default_rng(2000 + r)
        mk = np.zeros(N, bool)
        for s in np.unique(strata):
            members = np.where(strata == s)[0]
            take = int(round(len(members) / N * B))
            take = min(max(take, 0), len(members))
            mk[rng.choice(members, take, replace=False)] = True
        cur = mk.sum()
        if cur < B:
            rest = np.where(~mk)[0]
            mk[rng.choice(rest, B - cur, replace=False)] = True
        elif cur > B:
            on = np.where(mk)[0]
            mk[rng.choice(on, cur - B, replace=False)] = False
        masks[f"D_strat_{r}"] = ("D_stratified", str(r), mk)
    # E. random scattered, 30 reps
    for r in range(30):
        rng = np.random.default_rng(3000 + r)
        mk = np.zeros(N, bool)
        mk[rng.choice(N, B, replace=False)] = True
        masks[f"E_rand_{r}"] = ("E_random", str(r), mk)
    return masks

def main():
    df = pd.read_csv(OUT / "tables" / "frame_frozen.csv")
    y = (df["Class"] == "Unhealthy").to_numpy()
    MX = 111320 * np.cos(np.radians(df["Lat"].mean()))
    xm = df["Long"].to_numpy() * MX
    ym = df["Lat"].to_numpy() * 111320
    ndvi = df["NDVI"].to_numpy()
    X = ndvi.reshape(-1, 1)
    # pockets on full unhealthy coords (frozen 30 m rule)
    db = DBSCAN(eps=30, min_samples=2).fit(np.column_stack([xm[y == 1], ym[y == 1]]))
    plab = db.labels_
    pocket_ids = sorted([p for p in set(plab) if p != -1])
    n_pockets = len(pocket_ids)
    n_singles = int((plab == -1).sum())
    print(f"pockets={n_pockets} singles={n_singles}")
    rows = []
    for B in B_LEVELS:
        masks = build_masks(df, xm, ym, ndvi, B)
        for key, (design, rep, mk) in masks.items():
            tr, te = mk, ~mk
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
            yt = y[te]
            pr = float(average_precision_score(yt, s))
            try:
                roc = float(roc_auc_score(yt, s))
            except Exception:
                roc = float("nan")
            r10 = recall_at_k(yt, s, 0.10); r25 = recall_at_k(yt, s, 0.25)
            # pocket hit: pocket counted hit if >=1 of its trees in train AND unhealthy (train always labelled)
            hit_p, hit_s = 0, 0
            uh_idx = np.where(y == 1)[0]
            tr_set = set(np.where(tr)[0].tolist())
            for p in pocket_ids:
                members = uh_idx[plab == p]
                if any(i in tr_set for i in members):
                    hit_p += 1
            s_members = uh_idx[plab == -1]
            if len(s_members):
                hit_s = sum(1 for i in s_members if i in tr_set)
            rows.append({"B": B, "design": design, "rep": rep, "key": key,
                         "n_train": int(tr.sum()), "pos_train": int(y[tr].sum()),
                         "n_test": int(te.sum()), "pos_test": int(yt.sum()),
                         "pr": round(pr, 4), "roc": round(roc, 4),
                         "r10": round(r10, 4), "r25": round(r25, 4),
                         "e25": round(r25 / 0.25, 3),
                         "pocket_hit_frac": round(hit_p / max(n_pockets, 1), 4),
                         "single_hit_frac": round(hit_s / max(n_singles, 1), 4)})
        print(f"B={B} done ({len(masks)} runs)")
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "tables" / "p1_runs.csv", index=False)
    g = runs.groupby(["B", "design"]).agg(n=("pr", "size"), pr_mean=("pr", "mean"), pr_sd=("pr", "std"),
        r25_mean=("r25", "mean"), r25_sd=("r25", "std"), roc_mean=("roc", "mean"),
        pos_train_mean=("pos_train", "mean"), pos_train_min=("pos_train", "min"),
        pocket_hit_mean=("pocket_hit_frac", "mean")).reset_index()
    # 95% CI via normal approx
    g["pr_ci95"] = 1.96 * g["pr_sd"] / np.sqrt(g["n"])
    g["r25_ci95"] = 1.96 * g["r25_sd"] / np.sqrt(g["n"])
    g.to_csv(OUT / "tables" / "p1_summary.csv", index=False)
    print(g.to_string())
    # figures
    import seaborn as sns
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for d in ["A_block", "B_strip", "C_dispersed", "D_stratified", "E_random"]:
        s = g[g.design == d].sort_values("B")
        ax[0].errorbar(s["B"], s["pr_mean"], yerr=s["pr_ci95"], marker="o", capsize=3, label=d)
        ax[1].errorbar(s["B"], s["r25_mean"], yerr=s["r25_ci95"], marker="o", capsize=3, label=d)
    ax[0].axhline(POS_RATE, color="k", ls="--", lw=1, label="no-skill")
    ax[0].set_title("PR-AUC vs labelling budget (NDVI-only LogReg)"); ax[0].set_ylabel("PR-AUC (test rest)")
    ax[1].axhline(0.25, color="k", ls="--", lw=1, label="chance@25%")
    ax[1].set_title("Recall@25% review vs budget"); ax[1].set_ylabel("recall")
    for a in ax:
        a.set_xlabel("B (labelled trees)"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_p1_designs.png", dpi=130)
    # example maps at B=1200 (first rep of each design)
    fig2, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharex=True, sharey=True)
    df1200 = build_masks(df, xm, ym, ndvi, 1200)
    picks = [("A_block", "A_block_SW"), ("B_strip", "B_strip_lon_2"), ("C_dispersed", "C_disp_0"),
             ("D_stratified", "D_strat_0"), ("E_random", "E_rand_0")]
    for axi, (d, k) in zip(axes, picks):
        _, _, mk = df1200[k]
        axi.scatter(df["Long"][~mk], df["Lat"][~mk], s=3, alpha=0.3, color="grey")
        axi.scatter(df["Long"][mk], df["Lat"][mk], s=3, alpha=0.6, color="tab:blue")
        uh = (df["Class"] == "Unhealthy").to_numpy() & mk
        axi.scatter(df["Long"][uh], df["Lat"][uh], s=8, color="red", alpha=0.9)
        axi.set_title(f"{d} B=1200\ntrain red=sick-in-train")
    fig2.tight_layout(); fig2.savefig(OUT / "figs" / "fig_p1_maps.png", dpi=130)
    print("wrote p1_runs.csv, p1_summary.csv, fig_p1_designs.png, fig_p1_maps.png")

if __name__ == "__main__":
    main()
