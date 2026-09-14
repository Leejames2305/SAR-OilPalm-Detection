# Path-Forward Trials: Algorithm Choice versus Problem Reformulation

**Date:** 14 September 2026  
**Scope:** Compare non-RF supervised models, unsupervised learning, Healthy-first anomaly detection, RVI-adjacent features, local residuals, and spatial-context models on `data/Processed/dataset_all_estates.csv`.

## Verdict

The disappointing POC_AHCensus result is **not primarily an algorithm-selection failure**. Under spatial holdout, changing the learner, adding RVI-adjacent indices, or fitting only Healthy trees all leave performance close to chance. Unsupervised clustering also does not align with the health labels.

The only repeatable lever found was **spatial context**. Models using coordinates plus SAR features were substantially better than SAR-only models, both under 220 m block holdout and under a broader four-region holdout. This is a promising route only for **in-estate triage** with confirmed labels or coordinates. It is not evidence of a transferable new-estate disease classifier.

## What was tested

| Path | Examples | Main result |
|---|---|---|
| Supervised alternatives | Logistic L1/L2, SVM-RBF, kNN, LDA, Gaussian NB, ExtraTrees, HistGradientBoosting, XGBoost, MLP | Several models beat RF by a small amount, but the winner changed by estate: kNN on AHCensus, logistic regression on Palong, SVM-RBF on Serting. No alternative reached a useful SAR-only operating point. |
| Healthy-first anomaly detection | OCSVM, Isolation Forest, LOF, GMM, PCA reconstruction, kNN distance, Mahalanobis distance, KMeans distance, autoencoder | No robust improvement. Best PR-AUC lifts were only about +0.008 AHCensus, +0.019 Palong, +0.024 Serting; the best method changed per estate and ROC-AUC stayed near or below 0.5 in several cases. |
| Unsupervised clustering | KMeans and diagonal GMM, 2-12 clusters, with and without local residual features | Median ARI was -0.001 to +0.000. Clusters did not represent health status. |
| RVI-family additions | VSI, CSI, Rc, Rp, total power, RVIHH, RVIVV, RNDVI, plus existing RVI/RFDI | Redundant with the supplied SAR channels; no consistent lift. |
| Local residuals | Tree value minus median of its 10 nearest neighbours | Small and inconsistent effect when used alone or with raw SAR features. |
| Spatial fusion | Coordinates plus SAR, local residuals, or both | Consistent improvement; the strongest practical signal in these trials. |
| Cross-estate transfer | Train on two estates, test on the third, with and without estate-wise z-scoring | Weak and inconsistent. Estate-wise z-scoring helped Palong somewhat but did not create a robust signature. |

## 220 m spatial-block results

Prevalence is the no-skill PR-AUC. Enrichment is precision in the top 10% divided by prevalence.

| Estate | Prevalence | RF baseline | Best SAR-only | Best Healthy-only | Best SAR + spatial | Top-10% enrichment of best fusion |
|---|---:|---:|---:|---:|---:|---:|
| AHCensus | 0.064 | 0.072 | 0.092 (logistic, log raw4) | 0.072 (PCA reconstruction) | 0.121 (logistic, raw4 + local + coords) | 2.18x |
| Palong | 0.092 | 0.087 | 0.114 (logistic, log raw4) | 0.111 (PCA reconstruction, local14) | 0.239 (logistic, all14 + coords) | 2.39x |
| Serting | 0.092 | 0.083 | 0.098 (SVM-RBF, all14) | 0.116 (Isolation Forest, raw4 + local) | 0.179 (XGBoost, log raw4 + coords) | 2.46x |

The best SAR+spatial configurations also held up under the broader four-region holdout:

| Estate | Broad-holdout fusion PR-AUC | Lift over prevalence | Top-10% enrichment |
|---|---:|---:|---:|
| AHCensus | 0.125 | +0.062 | 2.49x |
| Palong | 0.242 | +0.150 | 2.44x |
| Serting | 0.164 | +0.071 | 2.46x |

The practical review-budget picture is less encouraging than PR-AUC alone. To capture at least half of the Unhealthy trees under the broad holdout, Palong required reviewing the top 25% of trees with 2.28x enrichment, while AHCensus and Serting required the top 40% at only 1.47x and 1.38x enrichment respectively. Only Palong approaches the stated "majority recall with reasonable false positives" objective. This is another reason to treat spatial fusion as a promising triage prototype, not yet as a deployable estate-wide classifier.

Coordinates alone were already strong on Palong and Serting. In paired block-bootstrap comparisons, adding SAR to coordinates increased PR-AUC by about 0.016-0.053, with 85-99% of bootstrap replicates positive but confidence intervals crossing zero in several estate/protocol combinations. The correct conclusion is therefore that SAR adds a modest, not yet definitive, increment to a spatial model.

The labels themselves are spatially clustered at all three estates. For example, the probability that a Unhealthy tree's ten nearest neighbours are Unhealthy was 0.119/0.336/0.255, versus 0.060/0.060/0.070 for Healthy trees. This spatial structure explains why the coordinates effect exists and why treating random CV as evidence of a spectral classifier was misleading.

## Transfer results

Leave-one-estate-out accuracy remained weak. The best PR-AUC per held-out estate was:

- AHCensus: 0.084 with Random Forest on all14 (prevalence 0.064).
- Palong: 0.124 with estate-z-scored logistic regression (prevalence 0.092).
- Serting: 0.104 with Random Forest on local14 (prevalence 0.092).

These are not strong enough for a transferable disease classifier.

## RVI paper implication

The attached RVI review supports the observed null rather than offering an easy feature rescue. Table 1 shows that the common RVI variants are simple functions of the same backscattering/decomposition channels already supplied. The review also states that when stress is mild or early, vegetation-water and structural changes may be minimal, so RVIs may not reflect the change accurately, and recommends optical fusion or spatiotemporal filtering/fusion for difficult cases. That matches the empirical result here.

## Recommendation

1. **Do not switch the project wholesale to unsupervised learning.** Clustering and Healthy-only anomaly detection did not reveal a usable health structure.
2. **Do not spend the next cycle on another supervised algorithm sweep.** The ranking differences are small, estate-specific, and dominated by spatial structure.
3. **If the intended use is in-estate triage,** pivot to a spatial decision-support model: confirmed labels + tree geometry + SAR features, evaluated by leave-region-out and scored with enrichment/recall at a fixed review budget. Treat coordinates as a contextual input, not as evidence that SAR can classify disease.
4. **If the intended use is transfer to a new estate,** change the data rather than the model. The highest-value next experiments are multi-temporal/change SAR, optical fusion, more estates for validation, and canopy-centred sampling. The current single-date, per-tree backscatter representation is the limiting factor.
5. **Keep RVI work as an ablation, not the main path.** It is useful for interpretability and reporting, but it does not add independent information to the existing quad-pol channels.

## Artifacts

- Detailed report and tables: `misc/POC_Results/PathForwardTrials/REPORT.md`
- Reproducible scripts: `misc/scripts/path_forward_trials/`
- Main decision table: `misc/POC_Results/PathForwardTrials/decision_rules.csv`
- Model race: `misc/POC_Results/PathForwardTrials/best_all14_models.csv`
- Feature ablation: `misc/POC_Results/PathForwardTrials/feature_ablation_xgboost.csv`
- Cross-estate: `misc/POC_Results/PathForwardTrials/best_cross_estate.csv`
- Spatial fusion comparison: `misc/POC_Results/PathForwardTrials/fusion_vs_coords.csv`
- Review-budget curve: `misc/POC_Results/PathForwardTrials/budget_curve_best_fusion.csv`
- Cluster diagnostics: `misc/POC_Results/PathForwardTrials/unsupervised_cluster_metrics.csv`
