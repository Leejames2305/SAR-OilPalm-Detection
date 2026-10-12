# Agent Plan - Main Checklist
**Last updated: 12th October 2026 - After TimeSeries + LimitedLabelsTriage (RVI_NDVI Stages 0-7 closed)**

Read the POC reports @ `docs/POCReport/...` first, before planning anything:
1. PathForwardTrials.md - latest cross-path trial + verdict (start here)
2. AirHitamRF.md - RF / G-SMOTE baseline and single-date data ceiling
3. PipelineAudit.md - root-cause analysis (spatial leakage, transfer)
4. MeanSampling.md - univariate / window-size / Middle-class findings
5. FeatureTests.md - feature-family + feature-selection evidence
6. RVI_NDVI.md - RVI null, NDVI baseline, rededge lift, block gradient, pockets, corner test @ AirHitam
7. LimitedLabelsTriage.md - label-walk x budget x model for NDVI-LogReg deployment @ AirHitam
8. TimeSeries.md - 12-month S1/S2 time series on top of NDVI @ AirHitam - negative

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
- **Locations:** 
  `AirHitam_Basemap` 2511 trees @ 6.37% positive (real field ground truth); `Palong_Basemap` 2171 @ 9.21%; `Serting_Basemap` 1667 @ 9.24%.
- **Honest CV:** 
  primary = `StratifiedGroupKFold` on spatial blocks, 3 repeated seeds.Broader = four adjacent-block regions, leave-one-region-out. Transfer = leave-one-estate-out.
- **Leaky reference CV:** 
  `StratifiedKFold` | `random`. Report as a leakage ceiling only - never as a result.
- **No-skill PR-AUC = Raw dataset positive rate:** 
  0.0637 / 0.0921 / 0.0924. Always compare against it.
- **Headline metric:** 
  `pr_auc_lift` (= `pr_auc - pos_rate`). For triage, also report recall / enrichment at a fixed review budget.
- **SAR-only remains weak:** 
  best block-holdout PR-AUC was 0.092 / 0.114 / 0.098 versus prevalence 0.064 / 0.092 / 0.092. RF was 0.072 / 0.087 / 0.083. The best learner changes byestate, so RF is not uniquely at fault, but no SAR-only model reaches a useful operating point.
- **Healthy-only anomaly detection is not a robust fix:** 
  best PR-AUC was 0.072 / 0.111 / 0.116,but the winner changes by estate and ROC-AUC is near or below 0.5 in several cases.
- **Unsupervised health structure is absent:** 
  KMeans / diagonal-GMM ARI was approximately -0.001 to 0.000 with and without local-residual features.
- **RVI-adjacent additions are redundant:** 
  VSI, CSI, Rc, Rp, total power, RVIHH, RVIVV, RNDVI are functions of the supplied polarimetric channels and add no consistent lift.
- **Spatial context is the only repeatable SAR-era lever found:**
  block-holdout fusion PR-AUC was 0.121 / 0.239 / 0.179, with top-10% enrichment 2.18x / 2.39x / 2.46x. Four-region holdout gave 0.125 / 0.242 / 0.164. This is an in-estate triage signal, not a transferable disease signature.
- **Spatial fusion is not yet deployable:** 
  to reach roughly 50% recall, Palong required the top 25% reviewed at 2.28x enrichment; AirHitam and Serting required the top 40% at 1.47x / 1.38x.
- **Coordinates carry most of the spatial gain.** 
  Adding SAR to coordinates raised PR-AUC by about 0.016-0.053; bootstrap intervals crossed zero in several estate/protocol combinations.
- **New-estate transfer is still weak:** 
  best leave-one-estate-out PR-AUC was 0.084 / 0.124 / 0.104 for AirHitam / Palong / Serting. `estate_z` normalisation did not create a robust signature.
- **Labels are strongly spatially clustered:** 
  the probability that a Unhealthy tree's 10 nearest neighbours are Unhealthy was 0.119 / 0.336 / 0.255, versus 0.060 / 0.060 / 0.070 for Healthy trees.
- **Location independent:** 
  each location is currently modelled fully independently.
- **Optical NDVI is the first robust non-spatial signal (AirHitam):** 
  d=-0.51,p=3.6e-11, PR-AUC 0.130 vs 0.064 prevalence, ROC 0.65; survives spatial-block CV (0.133/0.653) and the full random->K5 leakage gradient (0.142->0.129). +rededge/SWIR (NDRE5/6, NDMI) gives a small agreed lift (spatial PR 0.145). [`RVI_NDVI.md`]
- **RVI is closed (both families, smoothed and unsmoothed scenes):** 
  dual-pol ratios and eigenvalue quad-pol RVI are coin flips (|d|<=0.09, ROC~0.50);ratio normalisation cancels the proportional brightness shift. Temporal NDVI (+/-90 d slope/delta) is also null.
- **No ML pushes past univariate NDVI (AirHitam, n=160 pos):** 
  best honest spatial PR 0.14 (+RVI/LogReg) vs 0.133 NDVI-only is noise; RF/XGB trail linear LogReg everywhere, MLP collapses below no-skill. Capacity hurts at this sample size. [`RVI_NDVI.md`]
- **NDVI-LogReg triages at ~2x enrichment, flat across designs and budgets (AirHitam):**
  PR ~0.124-0.152 vs 0.064 no-skill, ROC ~0.64-0.67, recall@25% ~0.43-0.47 (~1.8-1.9x). Example B=1200: rest ~1311 trees (~84 sick), review top-25% (~328) -> ~40 found (~47%, ~1.9x). [`LimitedLabelsTriage.md`]
- **NDVI-LogReg saturates by ~1000 labels:** 
  B 1000->1600 buys nothing (E_random 0.133->0.130;  A_block 0.129->0.124); oracle 80% train (2008 trees) PR 0.123; spatial-K15 OOF 0.127.The ceiling is the floor. [`LimitedLabelsTriage.md`]
- **Labelling design barely matters for ranking, matters for variance:** 
  corner-block / strip / dispersed / stratified / random all land within ~0.01 PR at fixed B; corner/strip SD is 10-40x dispersed SD (corner lottery). Dispersed converts to only +0.01 PR at B=1600,
  nothing at B=1000. Pocket coverage != ranking power (stratified/random hit 79-96% of DBSCAN pockets, corners 40-66%, yet recall@25% differs ~0.02 - NDVI carries a global level shift, not a pocket fingerprint). [`LimitedLabelsTriage.md`]
- **Corner training (deployment proxy):**
  NDVI-only corner-train mean PR 0.127 vs 0.131 size-matched random-train - the spatial gap costs ~nothing. Corner lottery is real: SW 0.083 / NW 0.148 / SE 0.137 / NE 0.138. At corner sizes (24-63 pos) NDVI-only beats +rededge/SWIR (extra features overfit);
  SAR-only corner is 0.06-0.09 (~no-skill). Reproduced exactly by two independent POCs (`RVI_NDVI.md` stage 7, `TimeSeries.md` check B, `LimitedLabelsTriage.md`).
- **Disease pockets are chance-held-out:** 
  160 Unhealthy = 33 pockets (2-10 trees, 30 m rule) +35 isolated singles; median nearest-sick-neighbour 17 m. Pocket-holdout recall 0.26 vs 0.42 random-split (0.25 is coin-flip under top-25% rule); recall by distance to nearest train sick tree 0.55 (<25 m) -> 0.36 (25-100 m). Near labelled pockets it works, on unseen pockets it is chance. [`RVI_NDVI.md`]
- **12-month S1/S2 time series adds nothing on top of single-composite NDVI (AirHitam):**
  +S2 temporal lowers fold PR (0.132 vs 0.133 seed-42; -0.003 to -0.005 over 10 block seeds, ref mean 0.135). S1-temporal-only PR 0.084 vs 0.064 no-skill (ROC 0.55). Best nested +1 descending-VH-feature gain is +0.011 (3x3 box, 82% cells) but shrinks to +0.004 in the
  matched window and +0.003 at 6 months - it is the max of 24 source/window/group combos, do not report as confirmed. Regularised descending-anomaly+NDVI peaks at +0.006 (80% cells)
  on spatial CV but scores -0.018 vs random-train in the corner deployment test (0.113 vs 0.131; every temporal set < NDVI-only 0.127). [`TimeSeries.md`]
- **Deployment baseline is NDVI-only balanced-LogReg (AirHitam):** 
  agreed by all three optical POCs. No second sensor needed for this estate; revisit only with a second SAR scene or an independent estate. [`RVI_NDVI.md`, `LimitedLabelsTriage.md`, `TimeSeries.md`]


## B. Dead ends so far

- [!] **Class-imbalance handling as a fix.** 
  `none` / `class_weighted_rf` / `random_oversample` / `smote` / `gsmote` all land within **0.003 PR-AUC** of each other. Not the failure mode.
- [!] **G-SMOTE hyperparameter tuning.** 
  60 distinct configs/estate swept (`tau`, `delta`, `k`,`selection_strategy`). Whole surface only **0.013-0.023 PR-AUC wide** - flat. Do not re-grid finer.
- [!] **SMOTE / G-SMOTE implementation bug hunting.** 
  Latest POC matches `imbalanced-learn-extra`. The implementation is correct.
- [!] **RF feature selection / reduction.** 
  PCA, univariate, RFE, RF-importance top-k - all *hurt* or tie. Best reducer (MI top-k) only matches all-features within noise. [`FeatureTests.md`]
- [!] **PCA + LogReg / QDA as upgrades over NDVI-LogReg.** 
  PCA2/3+LogReg drops PR ~0.02 everywhere (first PCs keep healthy-canopy variance, not the disease direction); QDA-1D == LogReg-1D exactly, QDA-2D trails LogReg-spec. At ~64-103 train positives there is nothing quadratic to estimate.[`LimitedLabelsTriage.md` ,`FeatureTests.md`]
- [!] **Broad non-RF supervised sweep as the solution.** 
  Logistic L1/L2, SVM-RBF, kNN, LDA, Gaussian NB, ExtraTrees, HistGradientBoosting, XGBoost and MLP were tested. Small estate-specific wins over RF exist, but the ranking does not transfer and none yields a useful SAR-only result.
- [!] **MLP / deep capacity at n=160 positives.** 
  Collapses below no-skill on NDVI-only (0.048); nonlinear models trail linear everywhere. Do not re-run without an order of magnitude more positives. [`RVI_NDVI.md`]
- [!] **Healthy-only anomaly detection as a drop-in fix.** 
  OCSVM, Isolation Forest, LOF, GMM, PCA reconstruction, kNN distance, Mahalanobis distance, KMeans distance and autoencoder were tested. No robust improvement over the supervised baselines. [`PathForwardTrials.md`]
- [!] **Unsupervised clustering as a replacement target.** 
  KMeans and diagonal GMM over 2-12 clusters did not align with health labels (ARI ~ 0). Do not pivot the project wholesale to this.
- [!] **RVI / RFDI indices as features.** 
  ~0-MI, |d| <= 0.145, PR-AUC ~ baseline. [`MeanSampling.md` - Sec 7a]: both collapse to the HH/HV ratio and cancel the brightness signal.
  Extended RVI-family additions were also redundant. [`PathForwardTrials.md`]
- [!] **Raw 4-band backscatter window means as the whole feature set.** 
  `core` family is "nearly useless alone" [`FeatureTests.md` Sec 2].
- [!] **Widening the sampling window**
  to reveal the class gap. 3 -> 21 px *dilutes* it. [`MeanSampling.md`]
- [!] **Local 10-nearest-neighbour residuals as a standalone fix.** 
  Small and inconsistent when used alone or added to raw SAR; only spatial fusion produced a repeatable gain. [`PathForwardTrials.md`]
- [!] **Middle class as weak positives / auxiliary class.** 
  Dilutes the signal or lowers ROC. [`MeanSampling.md` Sec 7b]
- [!] **Cross-estate transfer / pooling without a strong domain adaptation.** 
  ROC-AUC 0.49-0.53 in the audit; best newer LOEO PR-AUC 0.084-0.124. Estate-wise z-scoring did not fix it. [`PipelineAudit.md` Sec 2C, `PathForwardTrials.md`]
- [!] **Threshold tuning to rescue recall.** 
  Enrichment over random is **0.87x-1.25x** at every F1-optimal point. Lowering the threshold is not a model improvement.
- [!] **Reporting `accuracy`.** 
  0.93 is achievable by predicting "Healthy" for every tree.
- [!] **RVI family incl. eigenvalue quad-pol, smoothed or unsmoothed scenes.**
  Null in every formulation; unsmoothed re-export changes nothing.
- [!] **Temporal NDVI (+/-90 d slope/delta) as the change signal.** 
  All null; oil palm is stable over this window.
- [!] **12-month Sentinel-2 temporal statistics (mean/std/CV/slope/delta90/anomaly) on top of single-composite NDVI.** 
  Lowers fold PR by 0.003-0.005 over 10 block seeds; anomaly+NDVI -0.007 spatial, worse in corner test. Strongest temporal-optical feature (NDRE anom d=-0.40) correlates ~0.67 with NDVI - no independent signal. [`TimeSeries.md`]
- [!] **12-month Sentinel-1 (C-band) temporal statistics as a deployment add-on to NDVI.**
  S1-only PR 0.084 (weak); nested +1-feature gain +0.004 to +0.011 is the max of 24 configs (unconfirmed); regularised descending-anomaly+NDVI +0.006 spatial does not survive the corner product test (all temporal sets below NDVI-only). Small corner sets (24-63 pos)
  cannot support the extra features. Do not deploy; reconsider only with a second ALOS (L-band) scene or an independent estate. [`TimeSeries.md`]
- [!] **Labelling budget >1000 or labelling-walk redesign as a ranking lift.** 
  B 1000->1600 buys nothing; all five walk designs within ~0.01 PR at fixed B. Design only controls variance (corner lottery), not the PR ceiling. [`LimitedLabelsTriage.md`]
- [!] **+rededge/SWIR on small or corner training sets.** 
  Helps only slightly, only at B=1600 off-corner (+0.009-0.011 on random masks); zero or negative on B=1000 corners - overfits. Deployment stays NDVI-only unless the labelling strategy changes first. [`LimitedLabelsTriage.md` ,`RVI_NDVI.md`, `TimeSeries.md`]

## C. Open backlog - ranked by expected value

- [~] **1. Multi-temporal / change SAR (2+ dates per estate).**
  Sentinel-1 12-month time series tried on AirHitam and closed as negative on top of NDVI (see Section B). What remains open is a second commercial ALOS-2 (L-band) scene: same-sensor change response, not C-band proxy. Reconsider temporal-SAR only with that or an independent estate. Note descending-only sparsity (18 scenes/12 mo, ~none in last 6 mo) limits the current S1 verdict to "no add-on signal", not "change is useless".
- [ ] **2. Window statistics (`winstats`) on the multi-estate dataset.**
  The older 78-feature Palong run found `_max` / `_min` / percentile window statistics were the dominant SAR family, but the current `dataset_all_estates.csv` is mean-only. Re-run them under the same repeated block and broad-region protocols. Do not confuse them with the local residual features already tested in `PathForwardTrials.md`.
- [x] **3. Optical complement (Sentinel-2 NDVI / NDWI), per tree or per block.**
  Done for AirHitam: single-composite NDVI is the deployment baseline (d=-0.51, spatial PR 0.133-0.135); +rededge/SWIR small agreed lift (0.145) but fragile (see Section B); 12-month S2-temporal closed as negative. Remaining: S2 pull for Palong/Serting (labels are prediction-derived there - treat as supporting, not validation).
- [ ] **4. Aggregate the target to plot / row / cell level or severity.**
  Cell-level hotspot aggregation reached ROC 0.62-0.80 within-estate where per-tree spatial CV fails [`PipelineAudit.md` Sec 2E, n=15 cells] - treat as headroom. The spatial-fusion result points in the same broad direction.
- [ ] **5. More estates as LOCO validation folds** 
  (target >= 4-5, varied age / soil). More *validation*, not more features.
- [ ] **6. Resolution / geometry.** 
  Finer SAR (~1-3 m) and snap samples to canopy centroid rather than trunk coordinate so the window sits on the crown.
- [ ] **7. Re-examine `Middle` as a severity / transition class for a regression or ordinal target**
  (not as a classifier class - see Section B).


## D. Open decisions (block planning until resolved)

- [x] **What is the intended use?** 
  Decided product shape so-far being modelling in a small portion of the entire field (~1000s of Ground Truth), and then the trained model predicts the rest (>90% area left) of the unlabeled field. 
- [x] **Deployment baseline (AirHitam)?** 
  NDVI-only balanced-LogReg. Agreed by `RVI_NDVI.md`,`LimitedLabelsTriage.md`, `TimeSeries.md`. Top-25% review -> ~45-47% recall at ~1.8-1.9x enrichment; saturates by ~1000 labels; every SAR/temporal/spectral addition is at best fragile and at worst negative in the corner product test.
- [ ] **What is the target unit?** 
  per-tree binary vs severity regression vs hotspot / plot map.
- [ ] **Is a precision floor or a recall floor acceptable?** 
  Needed to define the acceptance bar for a triage tool. Express it as top-k / top-x% review budget, not only as a probability threshold.
- [ ] **Keep or drop `Middle` / `Vacant`?** 
  Currently dropped; `Middle` is spatially meaningful (median 14.6 m from nearest Unhealthy) even though it is spectrally inconsistent.


## E. Known gotchas

- **Cohen's d sign flips across locations** 
  (AirHitam/Palong negative, Serting positive). A sign flip is not explicable by label noise - check this on any new feature before trusting it.
- **Spatial coordinates can look like model skill.** 
  Report coordinate-only and coordinate+SAR results side by side; do not describe a spatial-prior gain as a SAR classification result.
- **Model rankings are unstable across estates and folds.** 
  Prefer a fixed simple baseline and  paired comparisons; do not choose an algorithm from a single estate or one seed.
- **G-SMOTE `delta=1` collapses to a line (degenerate SMOTE).** 
  A historical bug lived here; the invariant battery in the notebook exists to catch it. Keep it if the G-SMOTE code is touched.
- **Large OOF score dumps are diagnostics, not headline evidence.** 
  The summary/metrics tables and the report are the primary artifacts; regenerate row-level scores from the committed scripts if needed.
- **The best temporal cell is the most optimistic one.** 
  The +0.011 nested gain is the max of 24 source/window/group combos (10 seeds); true gain is likely smaller. Corner product test takes priority over spatial-CV gain. [`TimeSeries.md`]
- **Small corner sets cannot support extra features.** 
  24-63 positives per quadrant; +rededge/SWIR, +SAR, +temporal all overfit there even when spatial CV looks flat/positive. [`RVI_NDVI.md`, `LimitedLabelsTriage.md`, `TimeSeries.md`]
- **Corner lottery persists across all optical POCs.** 
  SW 0.083 vs NW 0.148 (quadrants); pos_train lottery at B=1000: min 39 (NE-side) vs mean 61. Representativeness beats quantity.
- **Pocket coverage != ranking power.** 
  Hitting more DBSCAN pockets (dispersed/stratified 79-96% vs corners 40-66%) converts to ~0.02 recall, because NDVI carries a global level shift and pocket-holdout recall is chance (0.26). Do not use pocket-hit as a model-selection metric.
- **S2 10 m mixes crowns (~9 m spacing).** 
  Reported NDVI effect is conservative (mixing dilutes, not creates, signal). Same bias applies to S1/S2 time-series pulls (point vs 3x3 box compared head-to-head; box slightly better for SAR, neither changes verdicts).
- **Sparse descending S1.** 
  Only 18 descending scenes/12 mo, almost none in last 6 mo; ascending vs descending not fully comparable; 6-mo window has no usable descending features.


## F. Where things live

| Path | What |
|---|---|
| `./misc/POC` | All POCs' Runs (scripts & results)|
| `./misc/scripts` | Useful tools & automated scripts that help process datasets |
| `./main.py` | Marimo Notebook, update according to promissing POCs  |
| `./docs/POCReport/` | All POCs findings documented  |
