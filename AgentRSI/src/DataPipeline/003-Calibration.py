"""Step 3: block-size and gate calibration on the frozen 80 m split. Verifier-side; reads the data, writes only gate results.

  feasibility  per block size and estate: P(every part meets the Unhealthy minimums), P(tree shares within tolerance)
  leakage      NDVI incumbent: random 5-fold CV vs spatial-block CV (evidence for the block size)
  gate         composite T gate: NDVI incumbent vs candidates on T. Null arms are V-winners of random feature
               subsets; power arms add gamma * y. Each non-degradation threshold is scored on the same draws.

Gate rule (pre-registered, plan Decisions): incumbent = NDVI-only balanced LogReg trained on L, per estate.
Composite = equal-weight mean of per-estate dPR-AUC on T. Adopt iff the paired block-bootstrap 5th percentile > 0,
composite >= 0.03, and no estate's dPR is below the non-degradation threshold. The first --nondeg value is the
pre-registered -0.03; the others are scored on the same draws.

Only gate writes a file (.work/gate_results.json, or --out). The other two print to the terminal.

Usage:
  python 003-Calibration.py feasibility [--draws 100000]
  python 003-Calibration.py leakage
  python 003-Calibration.py gate [--features PATH] [--nb 200] [--reps 120] [--nondeg -0.03 -0.02 -0.04] [--out PATH]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (BLOCK_M, ESTATES, FEATURES, GATE_RESULTS, MASTER, SPLIT, SPLIT_FRAC, SPLIT_MIN, SPLIT_TOL,
                    block_keys)
from features import load_labels

FEAS_SIZES = [220, 165, 110, 80, 55]
LEAK_SIZES = [220, 165, 110]
COMP_FLOOR = 0.03
GAINS = (0.02, 0.04, 0.06)
ALL_FEATS = ["HH", "HV", "VH", "VV", "HAlpha_1", "HAlpha_2", "HAlpha_3", "B4", "B8", "clear_count"]
NDVI = ["NDVI"]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))


def fit_score(df, cols, tr, te):
    m = model()
    m.fit(df.loc[tr, cols].to_numpy(float), df.loc[tr, "y"].to_numpy())
    return m.decision_function(df.loc[te, cols].to_numpy(float))


def verdict(res, t):
    """Adopt iff lo > 0, composite >= floor, and no estate's dPR is below the threshold t."""
    degrade = any(v < t for v in res["per"].values())
    return bool(res["lo"] > 0 and res["comp"] >= COMP_FLOOR and not degrade), degrade


def draw_candidate(rng):
    k = rng.integers(2, 11)
    cols = list(rng.choice(ALL_FEATS, k, replace=False))
    if rng.random() < .5 and "NDVI" not in cols:
        cols[0] = "NDVI"
    return cols


def feasibility(draws):
    trees = load_labels()
    rng = np.random.default_rng(2026)
    print(f"{'size':>5} {'estate':9s} {'blocks':>6} {'P(pos)':>7} {'P(tree)':>8} {'P(both)':>8} {'exp draws':>10} {'top block':>10}")
    for bs in FEAS_SIZES:
        for e in ESTATES:
            sub = [t for t in trees if t.estate == e]
            y = np.array([t.cls == "Unhealthy" for t in sub], float)
            lon = np.array([t.lon for t in sub])
            lat = np.array([t.lat for t in sub])
            ub, inv = np.unique(block_keys(lon, lat, bs), return_inverse=True)
            P = np.bincount(inv, weights=y, minlength=len(ub))           # Unhealthy trees per block
            nt = np.bincount(inv, minlength=len(ub)).astype(float)       # all trees per block (Middle included)
            u = rng.random((draws, len(ub)))
            cat = np.where(u < SPLIT_FRAC["L"], 0, np.where(u < SPLIT_FRAC["L"] + SPLIT_FRAC["V"], 1, 2))
            Lp, Vp, Tp = (((cat == i) * P).sum(1) for i in (0, 1, 2))
            Ln, Vn, Tn = (((cat == i) * nt).sum(1) / nt.sum() for i in (0, 1, 2))
            pos = (Lp >= SPLIT_MIN["L"]) & (Vp >= SPLIT_MIN["V"]) & (Tp >= SPLIT_MIN["T"])
            frac = ((np.abs(Ln - SPLIT_FRAC["L"]) <= SPLIT_TOL) & (np.abs(Vn - SPLIT_FRAC["V"]) <= SPLIT_TOL)
                    & (np.abs(Tn - SPLIT_FRAC["T"]) <= SPLIT_TOL))
            both = pos & frac
            exp = "inf" if both.mean() == 0 else f"{1 / both.mean():,.1f}"
            print(f"{bs:>5} {e:9s} {len(ub):>6} {pos.mean():7.3f} {frac.mean():8.3f} {both.mean():8.4f} "
                  f"{exp:>10} {P.max() / y.sum():10.2f}")
    print("P(pos): every part meets the Unhealthy minimums. P(tree): tree shares within tolerance.")
    print("top block: largest block's share of Unhealthy trees.")


def leakage():
    F = pd.read_csv(FEATURES).merge(
        pd.read_csv(MASTER)[["estate", "id", "Long", "Lat", "class"]], on=["estate", "id"], validate="one_to_one"
    ).rename(columns={"class": "Class"})
    S = F[F.Class != "Middle"].reset_index(drop=True)

    rows = []
    for e in ESTATES:
        sub = S[S.estate == e].reset_index(drop=True)
        X = sub[["NDVI"]].to_numpy()
        y = (sub.Class == "Unhealthy").astype(int).to_numpy()
        prev = y.mean()
        rnd_pr, rnd_roc = [], []
        for seed in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                s = model().fit(X[tr], y[tr]).decision_function(X[te])
                rnd_pr.append(AP(y[te], s))
                rnd_roc.append(roc_auc_score(y[te], s))
        rows.append(dict(estate=e, scheme="random 5-fold", block_m="-", pr=np.mean(rnd_pr),
                         roc=np.mean(rnd_roc), lift=np.mean(rnd_pr) - prev))
        for bs in LEAK_SIZES:
            g = block_keys(sub.Long.to_numpy(), sub.Lat.to_numpy(), bs)
            if len(np.unique(g)) < 5:
                rows.append(dict(estate=e, scheme="spatial block", block_m=bs, pr=np.nan, roc=np.nan, lift=np.nan))
                continue
            pr, roc = [], []
            for seed in range(5):
                # shuffle group labels so the fold assignment changes per seed
                rng = np.random.default_rng(seed)
                uniq = np.unique(g)
                perm = dict(zip(uniq, rng.permutation(len(uniq))))
                gg = np.array([perm[v] for v in g])
                for tr, te in GroupKFold(5).split(X, y, gg):
                    if y[te].sum() == 0 or y[tr].sum() == 0:
                        continue
                    s = model().fit(X[tr], y[tr]).decision_function(X[te])
                    pr.append(AP(y[te], s))
                    roc.append(roc_auc_score(y[te], s))
            rows.append(dict(estate=e, scheme="spatial block", block_m=bs, pr=np.mean(pr),
                             roc=np.mean(roc), lift=np.mean(pr) - prev))

    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    print("PR-AUC change vs random 5-fold, by block size (nan = fewer than 5 blocks, not testable):")
    for e in ESTATES:
        r = out[(out.estate == e) & (out.scheme == "random 5-fold")].pr.iloc[0]
        print(f"  {e}: " + "  ".join(
            f"{bs}m {out[(out.estate == e) & (out.block_m == bs)].pr.iloc[0] - r:+.4f}" for bs in LEAK_SIZES))


def gate(args):
    NB, R, THRESH = args.nb, args.reps, args.nondeg
    rng = np.random.default_rng(11)

    FEAT = pd.read_csv(args.features)
    SPLIT_DF = pd.read_csv(SPLIT)
    LAB = pd.read_csv(MASTER)[["estate", "id", "class"]]
    D = FEAT.merge(SPLIT_DF, on=["estate", "id"], how="inner", validate="one_to_one")
    D = D.merge(LAB, on=["estate", "id"], how="left", validate="one_to_one")
    D = D[D["class"] != "Middle"].reset_index(drop=True)      # Middle never scored
    D["y"] = (D["class"] == "Unhealthy").astype(int)

    def evaluate(sc_by_estate, inc_by_estate, T_by_estate):
        """Per-estate dPR-AUC on T, composite and its block-bootstrap 5th percentile."""
        dpr = {}
        for e in ESTATES:
            sub = D.loc[T_by_estate[e]]
            dpr[e] = AP(sub.y, sc_by_estate[e]) - AP(sub.y, inc_by_estate[e])
        comp = np.mean(list(dpr.values()))
        # composite bootstrap: resample blocks within each estate, pair candidate with incumbent
        boots = []
        for _ in range(NB):
            vals = []
            for e in ESTATES:
                sub = D.loc[T_by_estate[e]]
                bb = sub.block.to_numpy()
                y = sub.y.to_numpy()
                ub = np.unique(bb)
                idx = {b: np.where(bb == b)[0] for b in ub}
                pick = rng.choice(ub, len(ub))
                ii = np.concatenate([idx[b] for b in pick])
                if y[ii].sum() < 3:
                    vals.append(np.nan)
                    continue
                vals.append(AP(y[ii], sc_by_estate[e][ii]) - AP(y[ii], inc_by_estate[e][ii]))
            if not np.isnan(vals).any():
                boots.append(np.mean(vals))
        return dict(comp=comp, lo=np.percentile(boots, 5), worst=min(dpr.values()), per=dpr)

    # incumbent and split sets per estate
    parts = {}
    for e in ESTATES:
        sub = D[D.estate == e]
        parts[e] = {p: np.isin(D.index, sub.index[sub.part == p].to_numpy()) for p in ["L", "V", "T"]}

    inc = {e: fit_score(D, NDVI, parts[e]["L"], parts[e]["T"]) for e in ESTATES}
    incV = {e: fit_score(D, NDVI, parts[e]["L"], parts[e]["V"]) for e in ESTATES}
    T_MASK = {e: parts[e]["T"] for e in ESTATES}
    print(f"split: {len(D)} scored trees; blocks at {BLOCK_M} m; NB={NB} R={R}; thresholds={THRESH}")
    base = evaluate(inc, inc, T_MASK)
    print("incumbent vs itself (should be ~0): comp", round(base["comp"], 4), "lo", round(base["lo"], 4))

    # calibrate gamma so the mean composite dPR on T hits each gain target (same gamma per estate)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    gm = []
    for g in grid:
        vals = []
        for _ in range(40):
            sT = {}
            for e in ESTATES:
                s_ = inc[e] / inc[e].std()
                sT[e] = s_ + g * D.y[parts[e]["T"]].to_numpy()
            vals.append(np.mean([AP(D.y[parts[e]["T"]], sT[e]) - AP(D.y[parts[e]["T"]], inc[e]) for e in ESTATES]))
        gm.append(np.mean(vals))
    gamma = {t: float(np.interp(t, gm, grid)) for t in GAINS}
    print("gamma calibration (grid -> composite dPR):", [(g, round(float(m), 4)) for g, m in zip(grid, gm)])
    print("gamma for targets:", {t: round(v, 3) for t, v in gamma.items()})

    arms = {f"null V-winner k={k}": [] for k in (1, 10, 30)}
    arms.update({f"true gain +{t:.2f}": [] for t in GAINS})

    for r in range(R):
        # null candidates: per candidate, fit on L, score V and T, per estate
        cands = []
        for _ in range(30):
            cols = draw_candidate(rng)
            sV = {e: fit_score(D, cols, parts[e]["L"], parts[e]["V"]) for e in ESTATES}
            sT = {e: fit_score(D, cols, parts[e]["L"], parts[e]["T"]) for e in ESTATES}
            vcomp = np.mean([AP(D.y[parts[e]["V"]], sV[e]) - AP(D.y[parts[e]["V"]], incV[e]) for e in ESTATES])
            cands.append((vcomp, sT))
        for k in (1, 10, 30):
            w = int(np.argmax([c[0] for c in cands[:k]]))
            arms[f"null V-winner k={k}"].append(evaluate(cands[w][1], inc, T_MASK))
        for t in GAINS:
            sT = {e: inc[e] / inc[e].std() + gamma[t] * D.y[parts[e]["T"]].to_numpy() for e in ESTATES}
            arms[f"true gain +{t:.2f}"].append(evaluate(sT, inc, T_MASK))
        if (r + 1) % 20 == 0:
            print("rep", r + 1, "/", R, flush=True)

    summary = {"nb": NB, "reps": R, "features": str(args.features), "thresholds": THRESH,
               "gamma": gamma, "base_comp": float(base["comp"]), "arms": {}}
    for name, rs in arms.items():
        row = {"comp": float(np.mean([x["comp"] for x in rs])), "worst": float(np.mean([x["worst"] for x in rs])),
               "p_adopt": {}, "p_degrade": {}}
        for t in THRESH:
            v = [verdict(x, t) for x in rs]
            row["p_adopt"][f"{t:+.2f}"] = float(np.mean([a for a, _ in v]))
            row["p_degrade"][f"{t:+.2f}"] = float(np.mean([d for _, d in v]))
        summary["arms"][name] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print("wrote", args.out)

    heads = " ".join(f"adopt@{t:+.2f}" for t in THRESH)
    dheads = " ".join(f"degr@{t:+.2f}" for t in THRESH)
    print(f"\n{'arm':22s} {heads}   comp     worst   {dheads}")
    for name, row in summary["arms"].items():
        a = " ".join(f"{row['p_adopt'][f'{t:+.2f}']:9.3f}" for t in THRESH)
        d = " ".join(f"{row['p_degrade'][f'{t:+.2f}']:9.3f}" for t in THRESH)
        print(f"{name:22s} {a}  {row['comp']:+.4f}  {row['worst']:+.4f}   {d}")


def main():
    ap = argparse.ArgumentParser(description="Block-size and gate calibration on the frozen split (verifier-side).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feasibility", help="expected split feasibility per block size")
    f.add_argument("--draws", type=int, default=100000, help="random splits per block size and estate")
    sub.add_parser("leakage", help="random vs spatial-block CV for the NDVI incumbent")
    g = sub.add_parser("gate", help="composite T gate calibration")
    g.add_argument("--features", type=Path, default=FEATURES)
    g.add_argument("--nb", type=int, default=200, help="bootstrap resamples")
    g.add_argument("--reps", type=int, default=120, help="repetitions")
    g.add_argument("--nondeg", type=float, nargs="+", default=[-0.03, -0.02, -0.04],
                   help="non-degradation thresholds; the first is pre-registered")
    g.add_argument("--out", type=Path, default=GATE_RESULTS, help="where to write the JSON summary")
    args = ap.parse_args()

    if args.cmd == "feasibility":
        feasibility(args.draws)
    elif args.cmd == "leakage":
        leakage()
    else:
        gate(args)


if __name__ == "__main__":
    main()
