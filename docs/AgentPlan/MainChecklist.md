# Agent Plan - Main Checklist
**Last updated: 11th September - After POC_AhCencus_RF_GSMOTE's Run**

Read the POC reports @ `docs/POCReport/...` first, before planning anything:
1. RF_GSMOTE_Report.md - latest run + verdict (start here)
2. PipelineAudit.md - root-cause analysis (spatial leakage, transfer)
3. MeanSampling.md - univariate / window-size / Middle-class findings
4. FeatureTests.md - feature-family + feature-selection evidence
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
- **Locations:** `AHCensus_Basemap` 2511 trees @ 6.37% positive (real field ground truth);
  `Palong_Basemap` 2171 @ 9.21%; `Serting_Basemap` 1667 @ 9.24%.
- **Honest CV:** `GroupKFold` | `spatial` on 0.002 deg (~220 m) spatial blocks, 5 folds. Blocks: 16 / 11 / 8.
- **Leaky reference CV:** `StratifiedKFold` | `random`. Report as a leakage ceiling only - never as a result.
- **No-skill PR-AUC = Raw dataset positive rate:** 0.0637 / 0.0921 / 0.0924. Always compare against it.
- **Headline metric:** `pr_auc_lift` (= `pr_auc - pos_rate`). Not accuracy, not raw ROC.
- **Current best honest result:** PR-AUC lift **-0.013 to +0.007**; ROC-AUC **0.448-0.513**. No-skill.
- **Location independent:** Each location is currently modelled fully independently.


## B. Dead ends so far

- [!] **Class-imbalance handling as a fix.** `none` / `class_weighted_rf` / `random_oversample` /
  `smote` / `gsmote` all land within **0.003 PR-AUC** of each other. Not the failure mode.
- [!] **G-SMOTE hyperparameter tuning.** 60 distinct configs/estate swept (`tau`, `delta`, `k`,
  `selection_strategy`). Whole surface only **0.013-0.023 PR-AUC wide** - flat. Do not re-grid finer.
- [!] **SMOTE / G-SMOTE implementation bug hunting.** Latest POC matches `imbalanced-learn-extra`. The implementation is correct.
- [!] **RF feature selection / reduction.** PCA, univariate, RFE, RF-importance top-k - all *hurt*
  or tie. Best reducer (MI top-k) only matches all-features within noise. [`FeatureTests.md`]
- [!] **RVI / RFDI indices as features.** ~0-MI, |d| <= 0.145, PR-AUC ~ baseline.
  [`MeanSampling.md` - Sec 7a]: both collapse to the HH/HV ratio and cancel the brightness signal.
- [!] **Raw 4-band backscatter window means as the whole feature set.** `core` family is
  "nearly useless alone" [`FeatureTests.md` Sec 2].
- [!] **Widening the sampling window** to reveal the class gap. 3 -> 21 px *dilutes* it.
  [`MeanSampling.md`]
- [!] **Middle class as weak positives / auxiliary class.** Dilutes the signal or lowers ROC.
  [`MeanSampling.md` Sec 7b]
- [!] **Cross-estate transfer / pooling without an estate term.** ROC-AUC 0.49-0.53 - effectively
  random on a new estate. [`PipelineAudit.md` Sec 2C]
- [!] **Threshold tuning to rescue recall.** Enrichment over random is **0.87x-1.25x** at every
  F1-optimal point. Lowering the threshold is not a model improvement.
- [!] **Reporting `accuracy`.** 0.93 is achievable by predicting "Healthy" for every tree.


## C. Open backlog - ranked by expected value

- [ ] **1. Persist OOF predictions, then score the triage metrics.**
  `save_oof()` writes `oof_predictions.csv` +
  `rf_importances.csv`; precision@k,
  recall@k and enrichment@budget. Build the enrichment curve.

- [ ] **2. Replace window means with window statistics.**
  The `winstats` family (`_max`, `_min`, `_p25`/`_p50`/`_p75`, std) carries essentially all the
  signal that exists. **The G-SMOTE POC never varied the feature set** - it is the largest untested
  lever. Re-run the existing harness with `FEATURES` swapped.

- [ ] **3. Aggregate the target to plot / row / cell level.**
  Cell-level hotspot aggregation reached **ROC 0.62-0.80** within-estate where per-tree fails
  [`PipelineAudit.md` Sec 2E, n=15 cells] - treat as headroom. Disease is spatially contiguous;
  per-tree may be an ill-posed target.

- [ ] **4. Multi-temporal / change SAR (2+ dates per estate).**
  Plant-stress signature is a *change* response, not an absolute level. Named by all three prior reports as the single biggest lever. Requires additional acquisition.

- [ ] **5. Explicit spatial model for in-estate triage.**
  Only if the intended use is in-estate ranking (see Section D). Add coordinates or
  distance-to-nearest-confirmed-infected-tree as features; evaluate leave-one-cluster-out.

- [ ] **6. Optical complement (Sentinel-2 NDVI / NDWI), per tree or per block.**
  SAR alone is a weak disease proxy; SAR+optical fusion is the standard fix.

- [ ] **7. More estates as LOCO validation folds** (target >= 4-5, varied age/soil).
  More *validation*, not more features.

- [ ] **8. Resolution / geometry.** Finer SAR (~1-3 m) and snap samples to canopy centroid rather
  than trunk coordinate so the window sits on the crown.

- [ ] **9. Re-examine `Middle` as a severity/transition class for a regression or ordinal target**
  (not as a classifier class - see Section B).


## D. Open decisions (block planning until resolved)

- [ ] **What is the intended use?** This determines the CV protocol and the metric.
  - *Cross-estate transfer* -> keep spatial CV / LOCO. Current answer: impossible with this data.
  - *In-estate triage aid* -> spatial autocorrelation is a legitimate predictor and random-CV
    numbers (ROC ~0.57) become deployment-relevant. Then the metric is enrichment@k, not PR-AUC.
- [ ] **What is the target unit?** per-tree binary vs severity regression vs hotspot/plot map.
- [ ] **Is a precision floor or a recall floor acceptable?** Needed to define the acceptance bar
  for a triage tool. The prior bar (enrichment >= 2x) is not yet met.
- [ ] **Keep or drop `Middle` / `Vacant`?** Currently dropped; `Middle` is spatially meaningful
  (median 14.6 m from nearest Unhealthy) even though it is spectrally inconsistent.


## E. Known gotchas

- **Cohen's d sign flips across locations** (AHCensus/Palong negative, Serting positive). A sign
  flip is not explicable by label noise - check this on any new feature before trusting it.
- **G-SMOTE `delta=1` collapses to a line (degenerate SMOTE).** A historical bug lived here; the
  invariant battery in the notebook exists to catch it. Keep it if the G-SMOTE code is touched.


## F. Where things live

| Path | What |
|---|---|
| `misc/notebook` | All POCs' Notebook |
| `misc/POC_Results` | Artifacts from POCs |
| `main.ipynb` | Original pipeline (outdated - random CV) |
| `docs/POCReport/` | All findings so far |
