"""Summarise, spatially smooth, bootstrap, and plot path-trial results."""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402

warnings.filterwarnings("ignore")


ROLE_ORDER = [
    "RF baseline",
    "Best SAR supervised",
    "Coordinates diagnostic",
    "Best Healthy-only",
    "Best Healthy + local",
]


def load_summary() -> pd.DataFrame:
    p = tc.RESULT_DIR / "supervised_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def pick(group: pd.DataFrame, predicate) -> pd.Series | None:
    sub = group[predicate(group)]
    if sub.empty:
        return None
    return sub.sort_values(["pr_auc_mean", "enrichment_at_10pct_mean"], ascending=False).iloc[0]


def build_leaderboard() -> pd.DataFrame:
    sup = pd.read_csv(tc.RESULT_DIR / "supervised_summary.csv")
    hea = pd.read_csv(tc.RESULT_DIR / "healthy_only_summary.csv")
    all_summary = pd.concat([sup, hea], ignore_index=True)
    rows: list[dict] = []
    for estate, g in all_summary.groupby("estate", sort=True):
        choices: list[tuple[str, pd.Series | None]] = []
        choices.append(
            (
                "RF baseline",
                pick(g, lambda x: (x.stage == "supervised") & (x.model == "rf_balanced") & (x.feature_set == "all14")),
            )
        )
        choices.append(
            (
                "Best SAR supervised",
                pick(g, lambda x: (x.stage == "supervised") & (x.feature_set != "coords_only")),
            )
        )
        choices.append(
            (
                "Coordinates diagnostic",
                pick(g, lambda x: (x.stage == "supervised") & (x.feature_set == "coords_only")),
            )
        )
        choices.append(("Best Healthy-only", pick(g, lambda x: x.stage == "healthy_only")))
        choices.append(
            (
                "Best Healthy + local",
                pick(g, lambda x: (x.stage == "healthy_only") & x.feature_set.str.contains("local")),
            )
        )
        for role, choice in choices:
            if choice is None:
                continue
            row = choice.to_dict()
            row["role"] = role
            row["rank_score"] = row.get("pr_auc_mean", np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(tc.RESULT_DIR / "leaderboard.csv", index=False)
    return out


def averaged_rank_scores(stage: str, estate: str, model: str, feature_set: str) -> pd.DataFrame:
    path = tc.RESULT_DIR / ("supervised_oof_scores.csv" if stage == "supervised" else "healthy_only_oof_scores.csv")
    df = pd.read_csv(path)
    g = df[
        (df.estate == estate)
        & (df.model == model)
        & (df.feature_set == feature_set)
    ].copy()
    if g.empty:
        raise ValueError(f"missing predictions: {stage}/{estate}/{model}/{feature_set}")
    g["rank_score"] = g.groupby("repeat", sort=False)["score"].rank(method="average", pct=True)
    agg = (
        g.groupby("source_row", as_index=False)
        .agg(
            estate=("estate", "first"),
            Long=("Long", "first"),
            Lat=("Lat", "first"),
            block=("block", "first"),
            y_true=("y_true", "first"),
            score=("rank_score", "mean"),
        )
    )
    return agg


def smoothing_analysis(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    roles = ["Best SAR supervised", "Coordinates diagnostic", "Best Healthy-only", "Best Healthy + local"]
    for _, r in leaderboard[leaderboard.role.isin(roles)].iterrows():
        d = averaged_rank_scores(r.stage, r.estate, r.model, r.feature_set)
        for sigma in [0.0, 15.0, 30.0, 60.0, 120.0]:
            if sigma == 0:
                score = d["score"].to_numpy(float)
            else:
                score = tc.smooth_scores_spatial(d, "score", sigma_m=sigma, max_radius_m=3 * sigma)
            m = tc.evaluate_scores(d.y_true.to_numpy(int), score)
            rows.append(
                {
                    "estate": r.estate,
                    "role": r.role,
                    "stage": r.stage,
                    "model": r.model,
                    "feature_set": r.feature_set,
                    "sigma_m": sigma,
                    **m,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tc.RESULT_DIR / "spatial_smoothing_metrics.csv", index=False)
    return out


def bootstrap_role_deltas(leaderboard: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(tc.SEED)
    rows: list[dict] = []
    for estate, g in leaderboard.groupby("estate", sort=True):
        base_row = g[g.role == "RF baseline"]
        if base_row.empty:
            continue
        br = base_row.iloc[0]
        base = averaged_rank_scores(br.stage, br.estate, br.model, br.feature_set)
        base["score"] = base["score"].rank(pct=True)
        for _, r in g[g.role != "RF baseline"].iterrows():
            alt = averaged_rank_scores(r.stage, r.estate, r.model, r.feature_set)
            merged = base[["source_row", "block", "y_true", "score"]].merge(
                alt[["source_row", "score"]], on="source_row", suffixes=("_base", "_alt")
            )
            observed = float(
                average_precision_score(merged.y_true, merged.score_alt)
                - average_precision_score(merged.y_true, merged.score_base)
            )
            blocks = merged.block.unique()
            deltas: list[float] = []
            for _ in range(n_boot):
                sampled = rng.choice(blocks, size=len(blocks), replace=True)
                parts = []
                for j, block in enumerate(sampled):
                    part = merged[merged.block == block].copy()
                    part["boot_block"] = j
                    parts.append(part)
                b = pd.concat(parts, ignore_index=True)
                if b.y_true.nunique() < 2:
                    continue
                delta = (
                    average_precision_score(b.y_true, b.score_alt)
                    - average_precision_score(b.y_true, b.score_base)
                )
                deltas.append(float(delta))
            arr = np.asarray(deltas)
            rows.append(
                {
                    "estate": estate,
                    "role": r.role,
                    "delta_pr_auc": observed,
                    "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
                    "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
                    "bootstrap_n": len(arr),
                    "prob_delta_gt_0": float((arr > 0).mean()) if len(arr) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tc.RESULT_DIR / "paired_block_bootstrap.csv", index=False)
    return out


def block_level(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in leaderboard.iterrows():
        d = averaged_rank_scores(r.stage, r.estate, r.model, r.feature_set)
        g = (
            d.groupby("block")
            .agg(n=("y_true", "size"), n_pos=("y_true", "sum"), prevalence=("y_true", "mean"), mean_score=("score", "mean"))
            .reset_index()
        )
        g["n_neg"] = g.n - g.n_pos
        both = g[(g.n_pos > 0) & (g.n_neg > 0)]
        rows.append(
            {
                "estate": r.estate,
                "role": r.role,
                "n_blocks": len(g),
                "n_blocks_both_classes": len(both),
                "block_roc_auc": roc_auc_score((both.prevalence > both.prevalence.median()).astype(int), both.mean_score)
                if len(both) >= 2 and both.mean_score.nunique() > 1
                else np.nan,
                "block_spearman": spearmanr(g.mean_score, g.prevalence).statistic if len(g) >= 3 else np.nan,
                "within_estate_pr_auc": average_precision_score(d.y_true, d.score),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(tc.RESULT_DIR / "block_level_diagnostic.csv", index=False)
    return out


def make_plots(leaderboard: pd.DataFrame, smoothing: pd.DataFrame) -> None:
    plots = tc.RESULT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

    pivot = leaderboard.pivot(index="estate", columns="role", values="pr_auc_mean").reindex(columns=ROLE_ORDER)
    ax = pivot.plot(kind="bar", figsize=(10, 4.6), width=0.78)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("PR-AUC (positive rate shown in labels)")
    ax.set_title("Path trial leaderboard - mean repeated spatial CV")
    rates = leaderboard.groupby("estate").pos_rate_mean.first()
    labels = [f"{e}\n({rates.get(e, np.nan):.3f})" for e in pivot.index]
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "leaderboard_pr_auc.png")
    plt.close()

    enr = leaderboard.pivot(index="estate", columns="role", values="enrichment_at_10pct_mean").reindex(columns=ROLE_ORDER)
    ax = enr.plot(kind="bar", figsize=(10, 4.6), width=0.78)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Enrichment at top 10%")
    ax.set_title("Triage budget: unhealthy enrichment over random")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "leaderboard_enrichment_10pct.png")
    plt.close()

    estates = sorted(leaderboard.estate.unique())
    fig, axes = plt.subplots(1, len(estates), figsize=(15, 4.2))
    if len(estates) == 1:
        axes = [axes]
    for ax, estate in zip(axes, estates):
        for _, r in leaderboard[leaderboard.estate == estate].iterrows():
            d = averaged_rank_scores(r.stage, estate, r.model, r.feature_set)
            precision, recall, _ = precision_recall_curve(d.y_true, d.score)
            ax.plot(recall, precision, label=r.role, linewidth=1.3)
        base = float(leaderboard[(leaderboard.estate == estate) & (leaderboard.role == "RF baseline")].pos_rate_mean.iloc[0])
        ax.axhline(base, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(0.45, base * 4))
        ax.set_xlabel("Recall (Unhealthy)")
        ax.set_ylabel("Precision (Unhealthy)")
        ax.set_title(estate.replace("_Basemap", ""))
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Precision-recall curves - averaged OOF percentile scores")
    plt.tight_layout()
    plt.savefig(plots / "selected_pr_curves.png")
    plt.close()

    fig, axes = plt.subplots(1, len(estates), figsize=(15, 4.0))
    if len(estates) == 1:
        axes = [axes]
    for ax, estate in zip(axes, estates):
        s = smoothing[smoothing.estate == estate]
        for role, g in s.groupby("role", sort=False):
            ax.plot(g.sigma_m, g.pr_auc, marker="o", label=role, linewidth=1.2)
        ax.set_xlabel("Spatial smoothing sigma (m)")
        ax.set_ylabel("PR-AUC")
        ax.set_title(estate.replace("_Basemap", ""))
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=6, loc="best")
    fig.suptitle("Effect of label-free spatial smoothing of model scores")
    plt.tight_layout()
    plt.savefig(plots / "spatial_smoothing_effect.png")
    plt.close()


def main() -> None:
    tc.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard()
    smoothing = smoothing_analysis(leaderboard)
    bootstrap_role_deltas(leaderboard)
    block_level(leaderboard)
    make_plots(leaderboard, smoothing)
    print(f"wrote analysis outputs to {tc.RESULT_DIR}", flush=True)


if __name__ == "__main__":
    main()
