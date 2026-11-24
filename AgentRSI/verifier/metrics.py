"""Scoring metrics, paired block-bootstrap intervals, and certification gate checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def exact_ap(y: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware step-function average precision matching sklearn.metrics.average_precision_score."""
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    n = len(y_arr)
    if n == 0:
        return 0.0
    n_pos = int(y_arr.sum())
    if n_pos == 0:
        return 0.0
    if np.ptp(s_arr) == 0.0:
        return float(n_pos / n)
    order = np.argsort(-s_arr, kind="mergesort")
    ss = s_arr[order]
    ys = y_arr[order]
    tp = np.cumsum(ys)
    distinct = np.where(np.diff(ss) != 0)[0]
    thr = np.r_[distinct, n - 1]
    tp_t = tp[thr]
    prec_t = tp_t / (thr + 1)
    dtp = np.diff(np.r_[0, tp_t])
    return float((prec_t * dtp).sum() / n_pos)


def recall_at_top_frac(y: np.ndarray, scores: np.ndarray, frac: float = 0.25) -> float:
    """Tie-averaged recall within the top `frac` fraction of ranked predictions.

    Any tied score group straddling the `frac * n` cutoff contributes proportionally
    to its overlap, so constant scores yield exact `frac` regardless of row order.
    """
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    n = len(y_arr)
    n_pos = int(y_arr.sum())
    if n == 0 or n_pos == 0 or frac <= 0.0:
        return 0.0
    if frac >= 1.0:
        return 1.0
    if np.ptp(s_arr) == 0.0:
        return float(frac)

    target = frac * n
    order = np.argsort(-s_arr, kind="mergesort")
    ss = s_arr[order]
    ys = y_arr[order]
    # Walk score-tie groups in strictly descending score order
    boundaries = np.r_[0, np.where(np.diff(ss) != 0)[0] + 1, n]
    found = 0.0
    remaining = target
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        m = end - start
        pos_in_group = float(ys[start:end].sum())
        if m <= remaining:
            found += pos_in_group
            remaining -= m
            if remaining <= 0.0:
                break
        else:
            found += pos_in_group * (remaining / m)
            break
    return float(found / n_pos)


def enrichment_at_top_frac(y: np.ndarray, scores: np.ndarray, frac: float = 0.10) -> float:
    """Enrichment factor (recall_at_top_frac / frac) over random triage."""
    if frac <= 0.0:
        return 0.0
    return float(recall_at_top_frac(y, scores, frac=frac) / frac)


def estate_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    """Compute all per-estate triage metrics for binary labels y and continuous risk scores."""
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    n = int(len(y_arr))
    n_pos = int(y_arr.sum())
    prev = float(n_pos / n) if n > 0 else 0.0
    ap = exact_ap(y_arr, s_arr)
    rec25 = recall_at_top_frac(y_arr, s_arr, frac=0.25)
    enr10 = enrichment_at_top_frac(y_arr, s_arr, frac=0.10)
    if n_pos == 0 or n_pos == n or np.ptp(s_arr) == 0.0:
        roc = 0.5
    else:
        roc = float(roc_auc_score(y_arr, s_arr))
    return {
        "pr_auc": float(ap),
        "prevalence": float(prev),
        "pr_auc_lift": float(ap - prev),
        "recall_at_25": float(rec25),
        "enrichment_top10": float(enr10),
        "roc_auc": float(roc),
        "n_scored": n,
        "n_pos": n_pos,
    }


def weighted_mean(values_by_estate: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted mean across estates present in `values_by_estate`."""
    total_w = sum(float(weights[e]) for e in values_by_estate)
    if total_w <= 0.0:
        raise ValueError("Total estate reward weight must be positive")
    return float(sum(float(weights[e]) * float(v) for e, v in values_by_estate.items()) / total_w)


def paired_block_bootstrap(
    y_by_estate: dict[str, np.ndarray],
    cand_by_estate: dict[str, np.ndarray],
    ref_by_estate: dict[str, np.ndarray],
    blocks_by_estate: dict[str, np.ndarray],
    weights: dict[str, float],
    B: int = 1000,
    seed: int = 20261120,
    ci_pct: float = 5.0,
    min_pos: int = 3,
) -> dict:
    """Paired spatial-block bootstrap of PR-AUC delta (`cand - ref`) across estates."""
    estates = list(y_by_estate)
    if not estates:
        raise ValueError("y_by_estate cannot be empty")
    total_w = sum(float(weights[e]) for e in estates)
    if total_w <= 0.0:
        raise ValueError("Total estate weight must be positive")

    point_deltas: dict[str, float] = {}
    block_indices: dict[str, list[np.ndarray]] = {}
    y_clean: dict[str, np.ndarray] = {}
    cand_clean: dict[str, np.ndarray] = {}
    ref_clean: dict[str, np.ndarray] = {}

    for e in estates:
        ye = np.asarray(y_by_estate[e], dtype=int)
        ce = np.asarray(cand_by_estate[e], dtype=float)
        re = np.asarray(ref_by_estate[e], dtype=float)
        be = np.asarray(blocks_by_estate[e])
        if not (len(ye) == len(ce) == len(re) == len(be)):
            raise ValueError(f"Mismatched array lengths for estate {e}")
        point_deltas[e] = exact_ap(ye, ce) - exact_ap(ye, re)
        y_clean[e], cand_clean[e], ref_clean[e] = ye, ce, re
        ub = np.unique(be)
        block_indices[e] = [np.where(be == b)[0] for b in ub]

    comp_delta = weighted_mean(point_deltas, weights)

    # Fast exact-zero path when candidate and reference predictions are identical
    if all(np.array_equal(cand_clean[e], ref_clean[e]) for e in estates):
        per = {e: {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0} for e in estates}
        return {
            "comp_delta": 0.0,
            "comp_ci_low": 0.0,
            "comp_ci_high": 0.0,
            "per_estate": per,
            "n_valid_draws": int(B),
            "n_requested_draws": int(B),
        }

    rng = np.random.default_rng(seed)
    boot_per: dict[str, list[float]] = {e: [] for e in estates}
    boot_comp: list[float] = []

    for _ in range(B):
        draw_deltas: dict[str, float] = {}
        ok = True
        for e in estates:
            idx_list = block_indices[e]
            picks = rng.integers(0, len(idx_list), size=len(idx_list))
            ii = np.concatenate([idx_list[j] for j in picks])
            y_sub = y_clean[e][ii]
            if int(y_sub.sum()) < min_pos:
                ok = False
                break
            draw_deltas[e] = exact_ap(y_sub, cand_clean[e][ii]) - exact_ap(y_sub, ref_clean[e][ii])
        if not ok:
            continue
        for e in estates:
            boot_per[e].append(draw_deltas[e])
        boot_comp.append(weighted_mean(draw_deltas, weights))

    if not boot_comp:
        raise RuntimeError(f"No valid bootstrap draws (all {B} draws had < {min_pos} positives)")

    per_estate = {
        e: {
            "delta": float(point_deltas[e]),
            "ci_low": float(np.percentile(boot_per[e], ci_pct)),
            "ci_high": float(np.percentile(boot_per[e], 100.0 - ci_pct)),
        }
        for e in estates
    }
    return {
        "comp_delta": float(comp_delta),
        "comp_ci_low": float(np.percentile(boot_comp, ci_pct)),
        "comp_ci_high": float(np.percentile(boot_comp, 100.0 - ci_pct)),
        "per_estate": per_estate,
        "n_valid_draws": int(len(boot_comp)),
        "n_requested_draws": int(B),
    }


def evaluate_gate(
    comp_delta: float,
    comp_ci_low: float,
    per_estate_deltas: dict[str, float],
    effect_floor: float = 0.03,
    non_degradation_floor: float = -0.03,
    min_improved_estates: int = 2,
) -> dict:
    """Apply the pre-registered certification gate to T deltas."""
    if not per_estate_deltas:
        raise ValueError("per_estate_deltas cannot be empty")
    worst_estate = min(per_estate_deltas, key=lambda e: per_estate_deltas[e])
    worst_delta = float(per_estate_deltas[worst_estate])
    n_improved = int(sum(1 for delta in per_estate_deltas.values() if float(delta) > 0.0))

    failed_reasons: list[str] = []
    if not (comp_ci_low > 0.0):
        failed_reasons.append(f"composite CI lower bound {comp_ci_low:+.6f} <= 0.0")
    if not (comp_delta >= effect_floor):
        failed_reasons.append(f"composite delta {comp_delta:+.6f} < effect floor {effect_floor:+.4f}")
    if n_improved < min_improved_estates:
        failed_reasons.append(
            f"improved estates {n_improved} < min_improved_estates {min_improved_estates}"
        )
    for estate, delta in per_estate_deltas.items():
        if delta < non_degradation_floor:
            failed_reasons.append(
                f"{estate} delta {delta:+.6f} < non-degradation floor {non_degradation_floor:+.4f}"
            )

    return {
        "passed": bool(len(failed_reasons) == 0),
        "comp_delta": float(comp_delta),
        "comp_ci_low": float(comp_ci_low),
        "effect_floor": float(effect_floor),
        "non_degradation_floor": float(non_degradation_floor),
        "min_improved_estates": int(min_improved_estates),
        "n_improved_estates": n_improved,
        "worst_estate": worst_estate,
        "worst_estate_delta": worst_delta,
        "failed_reasons": failed_reasons,
    }
