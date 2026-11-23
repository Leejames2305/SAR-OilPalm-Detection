"""Step 3: block-size and gate calibration at the configured block size. Verifier-side; never touches frozen T.

  feasibility  per block size and estate: P(every part meets the Unhealthy minimums), P(tree shares within tolerance)
  leakage      NDVI incumbent: random 5-fold CV vs spatial-block CV (pooled OOF PR-AUC/ROC/lift, includes 80 m)
  gate         composite T gate over RESAMPLED spatial block splits (using split.yaml rules, never frozen T):
               NDVI incumbent vs candidates on T. Null arms are V-winners of random feature subsets; power arms
               add gamma * y. Each non-degradation threshold and composite-only adoption are scored on the same draws.

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
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    BLOCK_M,
    ESTATES,
    FEATURE_COLS,
    FEATURES,
    GATE_RESULTS,
    MASTER,
    SPLIT_FRAC,
    SPLIT_MIN,
    SPLIT_TOL,
    block_keys,
)
from features import load_labels

FEAS_SIZES = [220, 165, 110, 80, 55]
LEAK_SIZES = [220, 165, 110, 80]
COMP_FLOOR = 0.03
GAINS = (0.02, 0.04, 0.06)
NDVI = ["NDVI"]


def exact_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Tie-aware step-function average precision matching sklearn.metrics.average_precision_score."""
    order = np.argsort(-s, kind="mergesort")
    ss = s[order]
    ys = y[order]
    n_pos = int(ys.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(ys)
    distinct = np.where(np.diff(ss) != 0)[0]
    thr = np.r_[distinct, len(ss) - 1]
    tp_t = tp[thr]
    prec_t = tp_t / (thr + 1)
    dtp = np.diff(np.r_[0, tp_t])
    return float((prec_t * dtp).sum() / n_pos)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))


def fit_score_both(X_mat: np.ndarray, y_all: np.ndarray, col_ids: list[int],
                   tr: np.ndarray, v_mask: np.ndarray, t_mask: np.ndarray):
    Xtr = X_mat[tr][:, col_ids]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(class_weight="balanced", max_iter=500)
    clf.fit((Xtr - mu) / sd, y_all[tr])
    sV = clf.decision_function((X_mat[v_mask][:, col_ids] - mu) / sd)
    sT = clf.decision_function((X_mat[t_mask][:, col_ids] - mu) / sd)
    return sV, sT


def feasibility(draws: int):
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

    rows = []
    for e in ESTATES:
        # Compute block keys on ALL trees (matching split.yaml), then score on Healthy + Unhealthy
        sub_all = F[F.estate == e].reset_index(drop=True)
        scored = (sub_all.Class != "Middle").to_numpy()
        sub = sub_all[scored].reset_index(drop=True)
        X = sub[["NDVI"]].to_numpy(float)
        y = (sub.Class == "Unhealthy").astype(int).to_numpy()
        prev = float(y.mean())

        rnd_pr, rnd_roc = [], []
        for seed in range(5):
            oof = np.empty(len(y), float)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                oof[te] = model().fit(X[tr], y[tr]).decision_function(X[te])
            rnd_pr.append(AP(y, oof))
            rnd_roc.append(roc_auc_score(y, oof))
        rows.append(dict(estate=e, scheme="random 5-fold", block_m="-", pr=np.mean(rnd_pr),
                         roc=np.mean(rnd_roc), lift=np.mean(rnd_pr) - prev))

        for bs in LEAK_SIZES:
            g_all = block_keys(sub_all.Long.to_numpy(float), sub_all.Lat.to_numpy(float), bs)
            g = g_all[scored]
            if len(np.unique(g)) < 5:
                rows.append(dict(estate=e, scheme="spatial block", block_m=bs, pr=np.nan, roc=np.nan, lift=np.nan))
                continue
            pr, roc = [], []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                uniq = np.unique(g)
                perm = dict(zip(uniq, rng.permutation(len(uniq))))
                gg = np.array([perm[v] for v in g])
                oof = np.full(len(y), np.nan)
                for tr, te in GroupKFold(5).split(X, y, gg):
                    if y[tr].sum() == 0:
                        continue
                    oof[te] = model().fit(X[tr], y[tr]).decision_function(X[te])
                ok = np.isfinite(oof)
                pr.append(AP(y[ok], oof[ok]))
                roc.append(roc_auc_score(y[ok], oof[ok]))
            rows.append(dict(estate=e, scheme="spatial block", block_m=bs, pr=np.mean(pr),
                             roc=np.mean(roc), lift=np.mean(pr) - prev))

    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    print("Pooled OOF PR-AUC change vs random 5-fold (nan = fewer than 5 blocks, not testable):")
    for e in ESTATES:
        r = out[(out.estate == e) & (out.scheme == "random 5-fold")].pr.iloc[0]
        print(f"  {e}: " + "  ".join(
            f"{bs}m {out[(out.estate == e) & (out.block_m == bs)].pr.iloc[0] - r:+.4f}" for bs in LEAK_SIZES))


def gate(args):
    t0 = time.time()
    NB, R, THRESH = args.nb, args.reps, args.nondeg
    rng = np.random.default_rng(11)

    feat = pd.read_csv(args.features)
    master = pd.read_csv(MASTER)[["estate", "id", "Long", "Lat", "class"]]
    merged = master.merge(feat, on=["estate", "id"], how="inner", validate="one_to_one")

    avail_feats = [c for c in FEATURE_COLS if c in merged.columns and c != "NDVI"]
    all_cols = avail_feats + NDVI
    col_idx = {c: i for i, c in enumerate(all_cols)}
    nd_id = [col_idx["NDVI"]]
    max_k = min(10, len(avail_feats))

    # Precompute spatial blocks on ALL trees (Middle travels with its block, excluded from metrics)
    est_data = {}
    n_scored = 0
    for e in ESTATES:
        sub = merged[merged.estate == e].reset_index(drop=True)
        keys = block_keys(sub.Long.to_numpy(float), sub.Lat.to_numpy(float), BLOCK_M)
        ub, inv = np.unique(keys, return_inverse=True)
        pos_blk = np.bincount(inv, weights=(sub["class"] == "Unhealthy").to_numpy(float), minlength=len(ub))
        nt_blk = np.bincount(inv, minlength=len(ub)).astype(float)
        scored_mask = (sub["class"] != "Middle").to_numpy()
        y_all = (sub["class"] == "Unhealthy").astype(int).to_numpy()
        X_mat = sub[all_cols].to_numpy(float)
        n_scored += int(scored_mask.sum())
        est_data[e] = (keys, ub, inv, pos_blk, nt_blk, scored_mask, y_all, X_mat)

    def draw_split():
        sp = {}
        for e in ESTATES:
            _, ub, inv, pos_blk, nt_blk, _, _, _ = est_data[e]
            while True:
                u = rng.random(len(ub))
                p = np.where(u < SPLIT_FRAC["L"], 0, np.where(u < SPLIT_FRAC["L"] + SPLIT_FRAC["V"], 1, 2))
                pL, pV, pT = (pos_blk[p == i].sum() for i in (0, 1, 2))
                tL, tV, tT = (nt_blk[p == i].sum() / nt_blk.sum() for i in (0, 1, 2))
                ok_pos = pL >= SPLIT_MIN["L"] and pV >= SPLIT_MIN["V"] and pT >= SPLIT_MIN["T"]
                ok_frac = all(abs(f - SPLIT_FRAC[k]) <= SPLIT_TOL for f, k in ((tL, "L"), (tV, "V"), (tT, "T")))
                if ok_pos and ok_frac:
                    sp[e] = p[inv]
                    break
        return sp

    print(f"calibration: {n_scored} scored trees; resampled {BLOCK_M} m splits (never frozen T); "
          f"{len(avail_feats)} candidate features; NB={NB} R={R}; thresholds={THRESH}", flush=True)

    # Calibrate gamma over 80 fresh resampled splits so mean composite dPR hits each target
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    cal = {g: [] for g in grid}
    for _ in range(80):
        sp = draw_split()
        inc_s, yT_s = {}, {}
        for e in ESTATES:
            _, _, _, _, _, sc, y_all, X_mat = est_data[e]
            tp = sp[e]
            L, V, T = (tp == 0) & sc, (tp == 1) & sc, (tp == 2) & sc
            _, incT = fit_score_both(X_mat, y_all, nd_id, L, V, T)
            inc_s[e] = (incT, incT / incT.std())
            yT_s[e] = y_all[T]
        for g in grid:
            cal[g].append(np.mean([
                exact_ap(yT_s[e], inc_s[e][1] + g * yT_s[e]) - exact_ap(yT_s[e], inc_s[e][0])
                for e in ESTATES
            ]))
    gm = [float(np.mean(cal[g])) for g in grid]
    gamma = {t: float(np.interp(t, gm, grid)) for t in GAINS}
    print("gamma calibration (80 resampled splits -> composite dPR):",
          [(g, round(m, 4)) for g, m in zip(grid, gm)], flush=True)
    print("gamma for targets:", {t: round(v, 3) for t, v in gamma.items()}, flush=True)

    arms: dict[str, list[dict]] = {f"null V-winner k={k}": [] for k in (1, 10, 30)}
    arms.update({f"true gain +{t:.2f}": [] for t in GAINS})
    base_comps = []

    for r in range(R):
        sp = draw_split()
        L_m, V_m, T_m, yV_e, yT_e, incV_ap, incT_e, incT_ap = {}, {}, {}, {}, {}, {}, {}, {}
        boot_ii = {e: [] for e in ESTATES}
        boot_inc_ap = {e: [] for e in ESTATES}

        for e in ESTATES:
            keys, _, _, _, _, sc, y_all, X_mat = est_data[e]
            tp = sp[e]
            L_m[e], V_m[e], T_m[e] = (tp == 0) & sc, (tp == 1) & sc, (tp == 2) & sc
            yV_e[e], yT_e[e] = y_all[V_m[e]], y_all[T_m[e]]
            incV, incT = fit_score_both(X_mat, y_all, nd_id, L_m[e], V_m[e], T_m[e])
            incV_ap[e] = exact_ap(yV_e[e], incV)
            incT_e[e] = incT
            incT_ap[e] = exact_ap(yT_e[e], incT)

            blkT = keys[T_m[e]]
            ubT = np.unique(blkT)
            idx_list = [np.where(blkT == b)[0] for b in ubT]
            picks = rng.integers(0, len(ubT), size=(NB, len(ubT)))
            for b in range(NB):
                ii = np.concatenate([idx_list[j] for j in picks[b]])
                if yT_e[e][ii].sum() >= 3:
                    boot_ii[e].append(ii)
                    boot_inc_ap[e].append(exact_ap(yT_e[e][ii], incT[ii]))
                else:
                    boot_ii[e].append(None)
                    boot_inc_ap[e].append(np.nan)

        def evaluate(scT: dict[str, np.ndarray]) -> dict:
            dpr = {e: exact_ap(yT_e[e], scT[e]) - incT_ap[e] for e in ESTATES}
            comp = float(np.mean(list(dpr.values())))
            boots = []
            for b in range(NB):
                vals = []
                ok = True
                for e in ESTATES:
                    ii = boot_ii[e][b]
                    if ii is None:
                        ok = False
                        break
                    vals.append(exact_ap(yT_e[e][ii], scT[e][ii]) - boot_inc_ap[e][b])
                if ok:
                    boots.append(np.mean(vals))
            lo = float(np.percentile(boots, 5))
            return dict(comp=comp, lo=lo, worst=min(dpr.values()), per=dpr)

        if r == 0:
            base_res = evaluate(incT_e)
            base_comps.append(base_res["comp"])
            print("incumbent vs itself (should be 0): comp", round(base_res["comp"], 4),
                  "lo", round(base_res["lo"], 4), flush=True)

        cands = []
        for _ in range(30):
            k_f = int(rng.integers(2, max_k + 1))
            cols = list(rng.choice(avail_feats, k_f, replace=False))
            if rng.random() < 0.5 and "NDVI" not in cols:
                cols[0] = "NDVI"
            cids = [col_idx[c] for c in cols]
            sV_ap_diffs, sT = [], {}
            for e in ESTATES:
                _, _, _, _, _, _, y_all, X_mat = est_data[e]
                sv, st = fit_score_both(X_mat, y_all, cids, L_m[e], V_m[e], T_m[e])
                sV_ap_diffs.append(exact_ap(yV_e[e], sv) - incV_ap[e])
                sT[e] = st
            cands.append((float(np.mean(sV_ap_diffs)), sT))

        for k in (1, 10, 30):
            w = int(np.argmax([c[0] for c in cands[:k]]))
            arms[f"null V-winner k={k}"].append(evaluate(cands[w][1]))
        for t in GAINS:
            sT = {e: incT_e[e] / incT_e[e].std() + gamma[t] * yT_e[e] for e in ESTATES}
            arms[f"true gain +{t:.2f}"].append(evaluate(sT))

        if (r + 1) % 20 == 0:
            print(f"rep {r + 1} / {R} ({time.time() - t0:.1f}s)", flush=True)

    summary = {
        "block_m": BLOCK_M,
        "resampled_splits": True,
        "nb": NB,
        "reps": R,
        "n_candidate_features": len(avail_feats),
        "features": str(args.features),
        "thresholds": THRESH,
        "gamma": gamma,
        "base_comp": float(base_comps[0]),
        "arms": {},
    }
    for name, rs in arms.items():
        comp_only = [bool(x["lo"] > 0 and x["comp"] >= COMP_FLOOR) for x in rs]
        row = {
            "comp": float(np.mean([x["comp"] for x in rs])),
            "comp_sd": float(np.std([x["comp"] for x in rs])),
            "worst": float(np.mean([x["worst"] for x in rs])),
            "p_adopt_comp_only": float(np.mean(comp_only)),
            "p_adopt": {},
            "p_degrade": {},
        }
        for t in THRESH:
            degr = [any(v < t for v in x["per"].values()) for x in rs]
            adopt = [c and not d for c, d in zip(comp_only, degr)]
            row["p_adopt"][f"{t:+.2f}"] = float(np.mean(adopt))
            row["p_degrade"][f"{t:+.2f}"] = float(np.mean(degr))
        summary["arms"][name] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out} ({time.time() - t0:.1f}s)")

    heads = " ".join(f"adopt@{t:+.2f}" for t in THRESH)
    dheads = " ".join(f"degr@{t:+.2f}" for t in THRESH)
    print(f"\n{'arm':22s} {'comp_only':>9s} {heads}   comp (sd)       worst   {dheads}")
    for name, row in summary["arms"].items():
        a = " ".join(f"{row['p_adopt'][f'{t:+.2f}']:9.3f}" for t in THRESH)
        d = " ".join(f"{row['p_degrade'][f'{t:+.2f}']:9.3f}" for t in THRESH)
        print(f"{name:22s} {row['p_adopt_comp_only']:9.3f} {a}  "
              f"{row['comp']:+.4f} ({row['comp_sd']:.4f}) {row['worst']:+.4f}   {d}")


def main():
    ap = argparse.ArgumentParser(description="Block-size and gate calibration (verifier-side, resampled splits).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feasibility", help="expected split feasibility per block size")
    f.add_argument("--draws", type=int, default=100000, help="random splits per block size and estate")
    sub.add_parser("leakage", help="random vs spatial-block CV (pooled OOF) for the NDVI incumbent")
    g = sub.add_parser("gate", help="composite T gate calibration over resampled splits")
    g.add_argument("--features", type=Path, default=FEATURES)
    g.add_argument("--nb", type=int, default=200, help="bootstrap resamples")
    g.add_argument("--reps", type=int, default=120, help="resampled-split repetitions")
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
