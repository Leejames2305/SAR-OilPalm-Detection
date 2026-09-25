# Agent Plan - Main Checklist
**Last updated: 14th September 2026 - After PathForwardTrials POC**

Read the POC reports @ `docs/POCReport/...` first, before planning anything:
1. PathForwardTrials.md - latest cross-path trial + verdict (start here)
2. AirHitamRF.md - RF / G-SMOTE baseline and single-date data ceiling
3. PipelineAudit.md - root-cause analysis (spatial leakage, transfer)
4. MeanSampling.md - univariate / window-size / Middle-class findings
5. FeatureTests.md - feature-family + feature-selection evidence
...

This file is a **terse index only**. It shows what is already settled, what must never be re-run, and what is still open. Do not write analysis prose here - put findings in `docs/POCReport/`. Only update this list when given permission to.


## Legend

| Mark | Meaning |
|---|---|
| `[x]` | tried - conclusion recorded, safe to cite |
| `[!]` | tried - Dead End |
| `[ ]` | not tried |
| `[~]` | partially done / blocked |


## A. Settled facts - quick reference

- **Task:** per-tree binary `Unhealthy` (positive) vs `Healthy`.
  - `Middle` / `Vacant` currently dropped.
- **Locations:** `AirHitam_Basemap` 2511 trees @ 6.37% positive (real field ground truth);
  `Palong_Basemap` 2171 @ 9.21%; `Serting_Basemap` 1667 @ 9.24%.
- **Honest CV:** primary = `StratifiedGroupKFold` on ~220 m spatial blocks, 3 repeated seeds.
  Broader = four adjacent-block regions, leave-one-region-out. Transfer = leave-one-estate-out.
- **Leaky reference CV:** `StratifiedKFold` | `random`. Report as a leakage ceiling only - never as a result.
- **No-skill PR-AUC = Raw dataset positive rate:** 0.0637 / 0.0921 / 0.0924. Always compare against it.
- **Headline metric:** `pr_auc_lift` (= `pr_auc - pos_rate`). For triage, also report recall / enrichment at a fixed review budget.
- **SAR-only remains weak:** best block-holdout PR-AUC was 0.092 / 0.114 / 0.098 versus
  prevalence 0.064 / 0.092 / 0.092. RF was 0.072 / 0.087 / 0.083. The best learner changes by
  estate, so RF is not uniquely at fault, but no SAR-only model reaches a useful operating point.
- **Healthy-only anomaly detection is not a robust fix:** best PR-AUC was 0.072 / 0.111 / 0.116,
  but the winner changes by estate and ROC-AUC is near or below 0.5 in several cases.
- **Unsupervised health structure is absent:** KMeans / diagonal-GMM ARI was approximately
  -0.001 to 0.000 with and without local-residual features.
- **RVI-adjacent additions are redundant:** VSI, CSI, Rc, Rp, total power, RVIHH, RVIVV, RNDVI
  are functions of the supplied polarimetric channels and add no consistent lift.
- **Spatial context is the only repeatable lever found:** block-holdout fusion PR-AUC was
  0.121 / 0.239 / 0.179, with top-10% enrichment 2.18x / 2.39x / 2.46x. Four-region holdout gave
  0.125 / 0.242 / 0.164. This is an in-estate triage signal, not a transferable disease signature.
- **Spatial fusion is not yet deployable:** to reach roughly 50% recall, Palong required the top
  25% reviewed at 2.28x enrichment; AirHitam and Serting required the top 40% at 1.47x / 1.38x.
- **Coordinates carry most of the spatial gain.** Adding SAR to coordinates raised PR-AUC by about
  0.016-0.053; bootstrap intervals crossed zero in several estate/protocol combinations.
- **New-estate transfer is still weak:** best leave-one-estate-out PR-AUC was 0.084 / 0.124 / 0.104
  for AirHitam / Palong / Serting. `estate_z` normalisation did not create a robust signature.
- **Labels are strongly spatially clustered:** the probability that a Unhealthy tree's 10 nearest
  neighbours are Unhealthy was 0.119 / 0.336 / 0.255, versus 0.060 / 0.060 / 0.070 for Healthy trees.
- **Location independent:** each location is currently modelled fully independently.


## B. Dead ends so far

- [!] **Class-imbalance handling as a fix.** `none` / `class_weighted_rf` / `random_oversample` /
  `smote` / `gsmote` all land within **0.003 PR-AUC** of each other. Not the failure mode.
- [!] **G-SMOTE hyperparameter tuning.** 60 distinct configs/estate swept (`tau`, `delta`, `k`,
  `selection_strategy`). Whole surface only **0.013-0.023 PR-AUC wide** - flat. Do not re-grid finer.
- [!] **SMOTE / G-SMOTE implementation bug hunting.** Latest POC matches `imbalanced-learn-extra`. The implementation is correct.
- [!] **RF feature selection / reduction.** PCA, univariate, RFE, RF-importance top-k - all *hurt*
  or tie. Best reducer (MI top-k) only matches all-features within noise. [`FeatureTests.md`]
- [!] **Broad non-RF supervised sweep as the solution.** Logistic L1/L2, SVM-RBF, kNN, LDA,
  Gaussian NB, ExtraTrees, HistGradientBoosting, XGBoost and MLP were tested. Small estate-specific
  wins over RF exist, but the ranking does not transfer and none yields a useful SAR-only result.
- [!] **Healthy-only anomaly detection as a drop-in fix.** OCSVM, Isolation Forest, LOF, GMM,
  PCA reconstruction, kNN distance, Mahalanobis distance, KMeans distance and autoencoder were
  tested. No robust improvement over the supervised baselines. [`PathForwardTrials.md`]
- [!] **Unsupervised clustering as a replacement target.** KMeans and diagonal GMM over 2-12
  clusters did not align with health labels (ARI ~ 0). Do not pivot the project wholesale to this.
- [!] **RVI / RFDI indices as features.** ~0-MI, |d| <= 0.145, PR-AUC ~ baseline.
  [`MeanSampling.md` - Sec 7a]: both collapse to the HH/HV ratio and cancel the brightness signal.
  Extended RVI-family additions were also redundant. [`PathForwardTrials.md`]
- [!] **Raw 4-band backscatter window means as the whole feature set.** `core` family is
  "nearly useless alone" [`FeatureTests.md` Sec 2].
- [!] **Widening the sampling window** to reveal the class gap. 3 -> 21 px *dilutes* it.
  [`MeanSampling.md`]
- [!] **Local 10-nearest-neighbour residuals as a standalone fix.** Small and inconsistent when
  used alone or added to raw SAR; only spatial fusion produced a repeatable gain. [`PathForwardTrials.md`]
- [!] **Middle class as weak positives / auxiliary class.** Dilutes the signal or lowers ROC.
  [`MeanSampling.md` Sec 7b]
- [!] **Cross-estate transfer / pooling without a strong domain adaptation.** ROC-AUC 0.49-0.53 in
  the audit; best newer LOEO PR-AUC 0.084-0.124. Estate-wise z-scoring did not fix it.
  [`PipelineAudit.md` Sec 2C, `PathForwardTrials.md`]
- [!] **Threshold tuning to rescue recall.** Enrichment over random is **0.87x-1.25x** at every
  F1-optimal point. Lowering the threshold is not a model improvement.
- [!] **Reporting `accuracy`.** 0.93 is achievable by predicting "Healthy" for every tree.


## C. Open backlog - ranked by expected value

- [ ] **1. In-estate spatial decision-support prototype (only if triage is the product).**
  Use confirmed labels + tree geometry + SAR features; evaluate with broad leave-region-out and
  report recall / enrichment at fixed review budgets. Test explicitly whether SAR adds value over
  coordinates and label context, and define the maximum acceptable review budget first.

- [ ] **2. Multi-temporal / change SAR (2+ dates per estate).**
  Plant-stress signature is a *change* response, not an absolute level. This is the highest-value
  data change if a transferable or new-estate classifier is still required.

- [ ] **3. Window statistics (`winstats`) on the multi-estate dataset.**
  The older 78-feature Palong run found `_max` / `_min` / percentile window statistics were the
  dominant SAR family, but the current `dataset_all_estates.csv` is mean-only. Re-run them under
  the same repeated block and broad-region protocols. Do not confuse them with the local residual
  features already tested in `PathForwardTrials.md`.

- [ ] **4. Optical complement (Sentinel-2 NDVI / NDWI), per tree or per block.**
  SAR alone is a weak disease proxy; SAR + optical fusion is a standard fix. Use only if suitable
  dates and spatially aligned canopies are available.

- [ ] **5. Aggregate the target to plot / row / cell level or severity.**
  Cell-level hotspot aggregation reached ROC 0.62-0.80 within-estate where per-tree spatial CV
  fails [`PipelineAudit.md` Sec 2E, n=15 cells] - treat as headroom. The spatial-fusion result
  points in the same broad direction.

- [ ] **6. More estates as LOCO validation folds** (target >= 4-5, varied age / soil).
  More *validation*, not more features.

- [ ] **7. Resolution / geometry.** Finer SAR (~1-3 m) and snap samples to canopy centroid rather
  than trunk coordinate so the window sits on the crown.

- [ ] **8. Re-examine `Middle` as a severity / transition class for a regression or ordinal target**
  (not as a classifier class - see Section B).


## D. Open decisions (block planning until resolved)

- [ ] **What is the intended use?** This determines the CV protocol and the metric.
  - *Cross-estate transfer* -> current data is not sufficient; prioritise multi-temporal, optical,
    more estates, and canopy-centred sampling rather than another classifier.
  - *In-estate triage aid* -> spatial context is legitimate and currently the only promising route,
    but it is not yet deployable: review budgets for 50% recall are high, and SAR's increment over
    coordinates is modest.
- [ ] **What is the target unit?** per-tree binary vs severity regression vs hotspot / plot map.
- [ ] **Is a precision floor or a recall floor acceptable?** Needed to define the acceptance bar
  for a triage tool. Express it as top-k / top-x% review budget, not only as a probability threshold.
- [ ] **Keep or drop `Middle` / `Vacant`?** Currently dropped; `Middle` is spatially meaningful
  (median 14.6 m from nearest Unhealthy) even though it is spectrally inconsistent.


## E. Known gotchas

- **Cohen's d sign flips across locations** (AirHitam/Palong negative, Serting positive). A sign
  flip is not explicable by label noise - check this on any new feature before trusting it.
- **Spatial coordinates can look like model skill.** Report coordinate-only and coordinate+SAR
  results side by side; do not describe a spatial-prior gain as a SAR classification result.
- **Model rankings are unstable across estates and folds.** Prefer a fixed simple baseline and
  paired comparisons; do not choose an algorithm from a single estate or one seed.
- **G-SMOTE `delta=1` collapses to a line (degenerate SMOTE).** A historical bug lived here; the
  invariant battery in the notebook exists to catch it. Keep it if the G-SMOTE code is touched.
- **Large OOF score dumps are diagnostics, not headline evidence.** The summary/metrics tables and
  the report are the primary artifacts; regenerate row-level scores from the committed scripts if needed.


## F. Where things live

| Path | What |
|---|---|
| `misc/notebook` | All POCs' Notebook |
| `misc/scripts` | Includes POCs .py such as `path_forward_trials` |
| `misc/POC_Results` | Artifacts from POCs |
| `main.ipynb` | Original pipeline (outdated - random CV) |
| `docs/POCReport/` | All findings so far |
