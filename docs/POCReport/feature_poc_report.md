# RF Performance POC — Findings (Palong_Basemap_w3_v4)

Run locally with `feature_poc/poc_*.py` (venv `/tmp/sarrenv`). Notebook NOT modified.
Dataset: 2171 trees (Middle dropped), 1971 Healthy / 200 Unhealthy (9.9:1), 78 features.
RF hyperparams held FIXED (400 trees, n_jobs=-1, class_weight ~10:1, random_state 42);
only the feature-handling step varies. Headline metric = PR-AUC (average_precision).
No-skill PR-AUC baseline = 0.092.

## 1) Feature-selection / PCA comparison — stratified 5x2 CV (LEAKY reference)

| config            | #feat | PR-AUC      | ROC-AUC | F1(U) | Rec(U) |
|-------------------|------:|------------:|--------:|------:|-------:|
| all_raw           | 78    | 0.384 ± 0.070 | 0.820 | 0.294 | 0.205 |
| dedup90 (CUR)     | 58    | **0.391 ± 0.075** | 0.819 | 0.280 | 0.190 |
| dedup80           | 37    | 0.365 ± 0.078 | 0.800 | 0.202 | 0.128 |
| pca95             | 24    | 0.261 ± 0.054 | 0.743 | 0.078 | 0.045 |
| pca20             | 20    | 0.258 ± 0.054 | 0.739 | 0.073 | 0.043 |
| kbest_f15 (ANOVA) | 15    | 0.278 ± 0.066 | 0.732 | 0.201 | 0.135 |
| kbest_f25         | 25    | 0.318 ± 0.054 | 0.775 | 0.222 | 0.155 |
| kbest_mi15        | 15    | 0.340 ± 0.077 | 0.804 | 0.315 | 0.265 |
| kbest_mi25        | 25    | 0.351 ± 0.072 | 0.808 | 0.334 | 0.265 |
| rfe15             | 15    | 0.334 ± 0.065 | 0.798 | 0.285 | 0.228 |
| imp15 (RF imp)    | 15    | 0.330 ± 0.065 | 0.788 | 0.281 | 0.223 |
| imp25             | 25    | 0.353 ± 0.059 | 0.808 | 0.291 | 0.223 |
| imp35             | 35    | 0.379 ± 0.061 | 0.819 | 0.294 | 0.215 |

Conclusion: the notebook's dedup90 is already the best (tied with all-raw within noise).
Every MORE aggressive reduction (PCA / univariate / RFE / RF-importance top-k) LOWERS PR-AUC.
The more you cut, the worse it gets. PCA is catastrophically bad for RF.

## 2) Feature-family audit (which features actually carry signal)

Family-only RF (stratified 5x2 CV):

| family    | #feat | PR-AUC |
|-----------|------:|-------:|
| winstats  | 24    | 0.341  |   <- dominant family (min/max/percentiles of 3x3 window)
| derived   | 7     | 0.208  |
| yama      | 8     | 0.181  |
| glcm      | 20    | 0.179  |
| halp      | 6     | 0.178  |
| anom      | 4     | 0.143  |
| ratio     | 4     | 0.132  |
| core      | 4     | 0.120  |   <- raw HH/HV/VV/VH nearly useless alone
| rvi       | 1     | 0.095  |   <- ~ no-skill (0.092)

Remove-one-family deltas vs ALL-78 (0.388):  no-winstats **-0.073** (essential),
no-halp -0.012, no-derived -0.009, no-glcm -0.007, others neutral (±0.002).
So: raw backscatter, anomalies, ratios and RVI can be dropped with NO loss.

MI screening: top features are window _max/_min/_p* (HV_max, HH_max, HV_min, ...).
The H-alpha/Yamaguchi decomposition features (H_std, alpha_std, Ps_std, Pv_std,
Pc_ratio, ...) have mutual information = 0.0 with the target.

Effect sizes Healthy-vs-Unhealthy: all |Cohen's d| <= 0.35 (weak signal everywhere).

## 3) SPATIAL BLOCK CV — the HONEST numbers (notebook's own protocol)

KMeans(5) blocks -> merged to 4 -> 3 folds, StratifiedGroupKFold.

| config          | PR-AUC       | ROC-AUC | F1(U) | Rec(U) |
|-----------------|-------------:|--------:|------:|-------:|
| all_raw         | 0.158 ± 0.058 | 0.621 | 0.025 | 0.062 |
| dedup90 (CUR)   | 0.150 ± 0.051 | 0.621 | 0.034 | 0.073 |
| kbest_f25       | 0.164 ± 0.037 | 0.602 | 0.029 | 0.094 |
| kbest_mi25      | **0.170 ± 0.065** | 0.639 | 0.060 | 0.155 |
| pca95           | 0.124 ± 0.041 | 0.539 | 0.032 | 0.042 |

KEY FINDING: random splits inflate PR-AUC ~2.4x (0.39 -> 0.16). Under honest
spatial CV the model barely beats no-skill (0.092). Unhealthy trees are spatially
clustered -> the RF mostly memorises local neighbourhoods, not a transferable
spectral signature.

## Bottom line

1. Feature noise is NOT the main failure mode. The current dedup90 is already the
   best reducer; all stronger feature selection (PCA especially) HURTS RandomForest.
2. The biggest lever is spatial leakage, not feature engineering: reporting only
   spatial-block-CV numbers is mandatory.
3. Real room for improvement: window statistics (not means) are the only signal;
   the newest H-alpha/Yamaguchi features are ~0-MI noise; raw backscatter/RVI are
   droppable. Aggregate labels per plot/row (disease spreads spatially) rather
   than per-tree would be the highest-value experiment.

Artifacts: `poc_results.csv` (stratified CV table), logs in /tmp/poc_run3.log,
/tmp/poc_fam.log, /tmp/poc_sp.log.