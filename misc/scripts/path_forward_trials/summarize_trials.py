"""Build final decision tables and a compact report from all trial artifacts."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_common as tc  # noqa: E402

OUT = tc.RESULT_DIR
METRIC_COLS = [
    "pr_auc", "pr_auc_lift", "roc_auc",
    "precision_at_10pct", "recall_at_10pct", "enrichment_at_10pct",
    "precision_at_5pct", "recall_at_5pct", "enrichment_at_5pct",
]


def summarize_repeats(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    agg = {c: ["mean", "std"] for c in METRIC_COLS if c in df.columns}
    out = df.groupby(keys, as_index=False).agg(agg)
    out.columns = [
        "_".join([str(x) for x in col if x]).rstrip("_") if isinstance(col, tuple) else col
        for col in out.columns
    ]
    for c in ["pr_auc", "pr_auc_lift", "roc_auc", "enrichment_at_5pct", "enrichment_at_10pct"]:
        std = f"{c}_std"
        if std not in out:
            out[std] = 0.0
    return out


def _load_grouped(path: Path, *, cv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if cv is not None and "cv" in df.columns:
        df = df[df.cv == cv].copy()
    keys = [c for c in ["stage", "cv", "estate", "model", "feature_set"] if c in df.columns]
    return summarize_repeats(df, keys)


def choose(g: pd.DataFrame, predicate) -> pd.Series | None:
    sub = g[predicate(g)]
    if sub.empty:
        return None
    return sub.sort_values(["pr_auc_mean", "enrichment_at_10pct_mean"], ascending=False).iloc[0]


def decision_table() -> pd.DataFrame:
    sup = _load_grouped(OUT / "supervised_repeat_metrics.csv", cv="spatial_5fold_repeated")
    healthy = _load_grouped(OUT / "healthy_only_repeat_metrics.csv", cv="spatial_5fold_repeated")
    fusion = _load_grouped(OUT / "spatial_fusion_metrics.csv", cv="spatial_5fold_repeated")
    context = _load_grouped(OUT / "spatial_context_metrics.csv", cv="spatial_5fold_repeated")
    broad_sup = _load_grouped(OUT / "broad_spatial_metrics.csv", cv="leave_one_region_out_4")
    broad_fusion = _load_grouped(OUT / "spatial_fusion_metrics.csv", cv="leave_one_region_out_4")
    broad_context = _load_grouped(OUT / "spatial_context_metrics.csv", cv="leave_one_region_out_4")

    rows: list[dict] = []
    for estate in sorted(sup.estate.unique()):
        block_rows = pd.concat(
            [
                sup[sup.estate == estate],
                healthy[healthy.estate == estate],
                fusion[fusion.estate == estate],
                context[context.estate == estate],
            ],
            ignore_index=True,
        )
        broad_rows = pd.concat(
            [
                broad_sup[broad_sup.estate == estate],
                broad_fusion[broad_fusion.estate == estate],
                broad_context[broad_context.estate == estate],
            ],
            ignore_index=True,
        )
        for protocol, table in [("block_5fold", block_rows), ("broad_region_4fold", broad_rows)]:
            choices: list[tuple[str, pd.Series | None]] = []
            if protocol == "block_5fold":
                choices.append(("RF baseline", choose(table, lambda x: (x.stage == "supervised") & (x.model == "rf_balanced") & (x.feature_set == "all14"))))
                choices.append(("Best supervised SAR", choose(table, lambda x: (x.stage == "supervised") & (~x.feature_set.str.contains("coord|local", regex=True)))))
                choices.append(("Best Healthy-only", choose(table, lambda x: x.stage == "healthy_only")))
                choices.append(("Best SAR + spatial", choose(table, lambda x: (x.stage == "spatial_fusion") & (x.feature_set != "coords_only"))))
                choices.append(("Spatial label context", choose(table, lambda x: x.stage == "spatial_context")))
            else:
                choices.append(("RF baseline", choose(table, lambda x: (x.stage == "supervised_broad_spatial") & (x.model == "rf_balanced") & (x.feature_set == "all14"))))
                choices.append(("Best supervised SAR", choose(table, lambda x: (x.stage == "supervised_broad_spatial") & (~x.feature_set.str.contains("coord|local", regex=True)))))
                choices.append(("Best Healthy-only", choose(table, lambda x: x.stage == "healthy_only_broad_spatial")))
                choices.append(("Best SAR + spatial", choose(table, lambda x: (x.stage == "spatial_fusion") & (x.feature_set != "coords_only"))))
                choices.append(("Spatial label context", choose(table, lambda x: x.stage == "spatial_context")))
            for role, row in choices:
                if row is None:
                    continue
                rows.append(
                    {
                        "protocol": protocol,
                        "estate": estate,
                        "role": role,
                        **{k: row.get(k) for k in [
                            "stage", "model", "feature_set", "pos_rate_mean", "pr_auc_mean", "pr_auc_std",
                            "pr_auc_lift_mean", "roc_auc_mean", "enrichment_at_5pct_mean",
                            "enrichment_at_10pct_mean", "recall_at_10pct_mean",
                        ]},
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "decision_rules.csv", index=False)
    return out


def model_race() -> pd.DataFrame:
    sup = pd.read_csv(OUT / "supervised_summary.csv")
    out = sup[(sup.feature_set == "all14")].copy()
    out = out.sort_values(["estate", "pr_auc_mean"], ascending=[True, False])
    out.to_csv(OUT / "best_all14_models.csv", index=False)
    return out


def feature_ablation() -> pd.DataFrame:
    sup = pd.read_csv(OUT / "supervised_summary.csv")
    ordered = ["raw4", "log_raw4", "existing_indices", "extended_indices", "local14", "all14", "raw4_plus_ext", "raw4_plus_local", "all14_plus_ext", "all14_plus_local", "coords_only"]
    out = sup[(sup.model == "xgboost_balanced") & (sup.feature_set.isin(ordered))].copy()
    out["feature_order"] = out.feature_set.map({k: i for i, k in enumerate(ordered)})
    out = out.sort_values(["estate", "feature_order"])
    out.to_csv(OUT / "feature_ablation_xgboost.csv", index=False)
    return out


def spatial_structure() -> tuple[pd.DataFrame, pd.DataFrame]:
    autocorr = pd.read_csv(OUT / "label_spatial_autocorrelation.csv")
    clusters = pd.read_csv(OUT / "unsupervised_cluster_metrics.csv")
    centered = clusters.copy()
    centered["abs_ari"] = centered.ari.abs()
    best = centered.sort_values(["estate", "abs_ari"], ascending=[True, False]).groupby("estate").head(1)
    return autocorr, best


def transfer_table() -> pd.DataFrame:
    d = pd.read_csv(OUT / "cross_estate_metrics.csv")
    out = d.sort_values(["estate", "pr_auc"], ascending=[True, False])
    out.to_csv(OUT / "best_cross_estate.csv", index=False)
    return out



def fusion_delta() -> pd.DataFrame:
    """Measure the incremental PR-AUC of SAR over coordinates in the fused stage."""
    d = pd.read_csv(OUT / "spatial_fusion_oof_scores.csv")
    pairs = {
        "AHCensus_Basemap": ("logreg_l2", "raw4_plus_local_plus_coords"),
        "Palong_Basemap": ("logreg_l2", "all14_plus_coords"),
        "Serting_Basemap": ("xgboost_balanced", "log_raw4_plus_coords"),
    }
    rows: list[dict] = []
    rng = np.random.default_rng(tc.SEED)
    for estate, (model, feature_set) in pairs.items():
        for cv, g in d[d.estate == estate].groupby("cv"):
            def avg_scores(set_name: str) -> pd.DataFrame:
                x = g[(g.model == model) & (g.feature_set == set_name)].copy()
                x["rank"] = x.groupby("repeat").score.rank(method="average", pct=True)
                return x.groupby("source_row", as_index=False).agg(
                    y=("y_true", "first"), block=("block", "first"), score=("rank", "mean")
                )
            fused = avg_scores(feature_set).rename(columns={"score": "fusion"})
            coords = avg_scores("coords_only")[["source_row", "score"]].rename(columns={"score": "coords"})
            m = fused.merge(coords, on="source_row")
            delta_ap = average_precision_score(m.y, m.fusion) - average_precision_score(m.y, m.coords)
            delta_auc = roc_auc_score(m.y, m.fusion) - roc_auc_score(m.y, m.coords)
            blocks = m.block.unique()
            boot: list[float] = []
            for _ in range(1000):
                sampled = rng.choice(blocks, size=len(blocks), replace=True)
                mm = pd.concat([m[m.block == b] for b in sampled], ignore_index=True)
                if mm.y.nunique() < 2:
                    continue
                boot.append(
                    average_precision_score(mm.y, mm.fusion)
                    - average_precision_score(mm.y, mm.coords)
                )
            arr = np.asarray(boot)
            rows.append(
                {
                    "estate": estate,
                    "cv": cv,
                    "model": model,
                    "feature_set": feature_set,
                    "delta_pr_auc": delta_ap,
                    "delta_roc_auc": delta_auc,
                    "ci_low": float(np.quantile(arr, 0.025)),
                    "ci_high": float(np.quantile(arr, 0.975)),
                    "prob_positive": float((arr > 0).mean()),
                    "boot_n": len(arr),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fusion_vs_coords.csv", index=False)
    return out



def budget_curve() -> pd.DataFrame:
    """Recall/enrichment curve for the best spatial-fusion configuration per estate."""
    d = pd.read_csv(OUT / "spatial_fusion_oof_scores.csv")
    pairs = {
        "AHCensus_Basemap": ("logreg_l2", "raw4_plus_local_plus_coords"),
        "Palong_Basemap": ("logreg_l2", "all14_plus_coords"),
        "Serting_Basemap": ("xgboost_balanced", "log_raw4_plus_coords"),
    }
    rows: list[dict] = []
    for estate, (model, feature_set) in pairs.items():
        for cv, g in d[d.estate == estate].groupby("cv"):
            x = g[(g.model == model) & (g.feature_set == feature_set)].copy()
            x["rank"] = x.groupby("repeat").score.rank(method="average", pct=True)
            x = x.groupby("source_row", as_index=False).agg(
                y=("y_true", "first"), score=("rank", "mean")
            )
            order = np.argsort(-x.score.to_numpy())
            y = x.y.to_numpy(int)
            prevalence = y.mean()
            n_pos = y.sum()
            for fraction in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                k = max(1, int(np.ceil(fraction * len(y))))
                tp = int(y[order[:k]].sum())
                precision = tp / k
                rows.append(
                    {
                        "estate": estate,
                        "cv": cv,
                        "model": model,
                        "feature_set": feature_set,
                        "budget": fraction,
                        "k": k,
                        "precision": precision,
                        "recall": tp / n_pos,
                        "enrichment": precision / prevalence,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "budget_curve_best_fusion.csv", index=False)
    return out


def make_plots(decision: pd.DataFrame, race: pd.DataFrame, ablation: pd.DataFrame) -> None:
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

    roles = ["RF baseline", "Best supervised SAR", "Best Healthy-only", "Best SAR + spatial", "Spatial label context"]
    for metric, label in [("pr_auc_mean", "PR-AUC"), ("enrichment_at_10pct_mean", "Enrichment at top 10%")]:
        for protocol in decision.protocol.unique():
            d = decision[decision.protocol == protocol]
            pivot = d.pivot(index="estate", columns="role", values=metric).reindex(columns=roles)
            ax = pivot.plot(kind="bar", figsize=(10, 4.4), width=0.78)
            if metric.startswith("enrichment"):
                ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            ax.set_ylabel(label)
            ax.set_title(f"{label} by path - {protocol}")
            ax.set_xticklabels(pivot.index, rotation=0)
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(plots / f"paths_{protocol}_{metric}.png")
            plt.close()

    for estate, g in race.groupby("estate"):
        g = g.sort_values("pr_auc_mean", ascending=True)
        ax = g.plot(kind="barh", x="model", y="pr_auc_mean", figsize=(7.5, 5), legend=False)
        ax.axvline(g.pos_rate_mean.iloc[0], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("PR-AUC (dashed = prevalence)")
        ax.set_title(f"All14 model race - {estate}")
        plt.tight_layout()
        plt.savefig(plots / f"model_race_{estate}.png")
        plt.close()

    for estate, g in ablation.groupby("estate"):
        g = g.sort_values("feature_order")
        ax = g.plot(kind="bar", x="feature_set", y="pr_auc_mean", figsize=(10, 4.4), legend=False)
        ax.axhline(g.pos_rate_mean.iloc[0], color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"Feature-family ablation (XGBoost) - {estate}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(plots / f"feature_ablation_{estate}.png")
        plt.close()


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    d = df[cols].copy()
    if max_rows is not None:
        d = d.head(max_rows)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def write_report(decision: pd.DataFrame, race: pd.DataFrame, ablation: pd.DataFrame, transfer: pd.DataFrame, auto: pd.DataFrame, clusters: pd.DataFrame, fusion: pd.DataFrame, budget: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Path-Forward Trials: Supervised, Unsupervised, and Healthy-First SAR Learning")
    lines.append("")
    lines.append("Generated from `misc/scripts/path_forward_trials/` using the sampled dataset at `data/Processed/dataset_all_estates.csv`.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Primary metric: PR-AUC, compared with estate prevalence; secondary metric: enrichment in the top 10% ranked trees.")
    lines.append("- Honest local CV: StratifiedGroupKFold on the existing ~220 m blocks, repeated with three seeds.")
    lines.append("- Broader holdout: adjacent blocks assigned to four spatial regions, then leave-one-region-out.")
    lines.append("- Cross-estate: leave-one-estate-out. Estate-wise z-score is reported as a domain-normalisation variant.")
    lines.append("- All labels are used only in supervised stages; Healthy-only detectors are fitted to Healthy training trees only.")
    lines.append("")
    lines.append("## Decision comparison")
    lines.append("")
    lines.append(markdown_table(decision, ["protocol", "estate", "role", "model", "feature_set", "pr_auc_mean", "pr_auc_lift_mean", "roc_auc_mean", "enrichment_at_10pct_mean"]))
    lines.append("")
    lines.append("## Model race on all 14 SAR features")
    lines.append("")
    lines.append(markdown_table(race, ["estate", "model", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "enrichment_at_10pct_mean"], max_rows=45))
    lines.append("")
    lines.append("## Feature-family ablation (XGBoost)")
    lines.append("")
    lines.append(markdown_table(ablation, ["estate", "feature_set", "pr_auc_mean", "pr_auc_lift_mean", "enrichment_at_10pct_mean"]))
    lines.append("")
    lines.append("## Cross-estate transfer")
    lines.append("")
    lines.append(markdown_table(transfer, ["estate", "model", "feature_set", "pr_auc", "pr_auc_lift", "roc_auc", "enrichment_at_10pct"], max_rows=24))
    lines.append("")
    lines.append("## Unsupervised structure")
    lines.append("")
    lines.append(markdown_table(auto, ["estate", "neighbor_u_rate_if_u", "neighbor_u_rate_if_h", "difference", "permutation_p_one_sided"]))
    lines.append("")
    lines.append(markdown_table(clusters, ["estate", "feature_set", "method", "n_clusters", "ari", "ami", "max_cluster_unhealthy_rate", "max_cluster_enrichment"]))
    lines.append("")
    lines.append("## Review-budget curve for best spatial fusion")
    lines.append("")
    lines.append(markdown_table(budget[budget.cv == "leave_one_region_out_4"], ["estate", "budget", "k", "precision", "recall", "enrichment"]))
    lines.append("")
    lines.append("## Incremental value of SAR on top of coordinates")
    lines.append("")
    lines.append(markdown_table(fusion, ["estate", "cv", "model", "feature_set", "delta_pr_auc", "delta_roc_auc", "ci_low", "ci_high", "prob_positive"]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. Model substitution is not the main lever. Several learners beat RF slightly (kNN on AHCensus, logistic regression on Palong, SVM-RBF on Serting), but the winner changes by estate and none reaches a useful SAR-only operating point under spatial holdout.")
    lines.append("2. Unsupervised clustering does not recover health classes: ARI values are near zero. The labels are spatially clustered, but the absolute SAR feature space does not form analogous clusters.")
    lines.append("3. Learning Healthy and flagging deviations does not produce a robust improvement. PCA reconstruction, isolation forest, GMM, OCSVM, kNN distance, and an autoencoder score at or near no-skill on most estate/protocol combinations, with the best detector changing by estate.")
    lines.append("4. Explicit RVI-adjacent indices do not rescue the model. VSI/CSI/Rc/Rp/RVIHH/RVIVV are deterministic functions of the supplied polarimetric channels and are mostly redundant with them.")
    lines.append("5. Adding coordinates is the only large and repeatable effect in these trials. It is a spatial-prior result, not a radar classification result, and is relevant only to in-estate triage where nearby confirmed labels are available. It does not establish transfer to a new estate.")
    lines.append("6. Cross-estate transfer remains weak. Estate-wise z-scoring helps some cases but does not create a robust disease signature.")
    lines.append("7. Recommended path: deprioritise further algorithm shopping and Healthy-only anomaly detection. If the intended product is in-estate triage, develop a spatial decision-support model that combines confirmed labels, tree geometry, and SAR features, evaluated with broader regional holdouts. If the product must generalise to new estates, change the data: multi-temporal/change SAR, optical fusion, and more estates are higher-value than another classifier.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in [
        "decision_rules.csv", "best_all14_models.csv", "feature_ablation_xgboost.csv",
        "best_cross_estate.csv", "label_spatial_autocorrelation.csv", "unsupervised_cluster_metrics.csv",
        "supervised_repeat_metrics.csv", "healthy_only_repeat_metrics.csv", "broad_spatial_metrics.csv",
        "spatial_fusion_metrics.csv", "spatial_context_metrics.csv",
        "fusion_vs_coords.csv", "budget_curve_best_fusion.csv",
    ]:
        lines.append(f"- `{OUT / name}`")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    decision = decision_table()
    race = model_race()
    ablation = feature_ablation()
    transfer = transfer_table()
    auto, clusters = spatial_structure()
    fusion = fusion_delta()
    budget = budget_curve()
    make_plots(decision, race, ablation)
    write_report(decision, race, ablation, transfer, auto, clusters, fusion, budget)
    print(f"wrote final tables, plots, and report to {OUT}", flush=True)


if __name__ == "__main__":
    main()
