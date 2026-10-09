# POC: Multi-temporal SAR and Sentinel-2 time series on the NDVI baseline - AirHitam

**Date:** 8 October 2026

**Scope:** AirHitam only (2511 trees: 2351 Healthy / 160 Unhealthy @ 6.37%) -
  the sole estate with one-to-one field ground truth. Follows
  `RVI_NDVI.md` (stage-2 NDVI reference, PR 0.133) and
  `LimitedLabelsTriage.md` (corner deployment).

**Question:** With only commercial ALOS-2 scene is single-date (2026-06-10).
  Does a 12-month Google Earth Engine time series (Sentinel-1 SAR, Sentinel-2
  optical) add Healthy/Unhealthy signal beyond the single-composite NDVI
  reference?

## Verdict

1. **Temporal Sentinel-2 adds nothing.** Yearly NDVI/NDRE statistics on top of
   the stage-2 NDVI reference lower fold PR (0.132 vs 0.133 at seed 42, and
   -0.003 to -0.005 across 10 block seeds). NDRE anomaly is the strongest
   temporal optical feature (d = -0.40) but correlates ~0.67 with NDVI.
2. **Temporal SAR alone is weak.** 12-month Sentinel-1 VV/VH statistics give
   fold PR 0.084 against 0.064 no-skill (about 1.3x), ROC 0.55.
3. **Adding one SAR feature to NDVI gives a small, unconfirmed gain.** A
   descending-orbit feature chosen inside each training fold lifts fold PR
   by +0.004 (10 m point sample) and +0.011 (3x3 box, 12 months to survey
   date). The gain does not hold across windows: it falls to +0.004 for the
   12 months matched to the stage-2 composite and +0.003 for the 6-month window.
4. **The best cell is the most optimistic one.** Across 24 source / window /
   group combinations, the +0.011 result is the maximum. It should not be
   reported as a confirmed improvement.
5. **Regularised anomaly features give at most +0.006 on spatial CV.** The
   best family is descending-orbit anomaly features added to NDVI (C = 0.01,
   3x3 box, 12 months: +0.006 over 80% of seed-fold cells). Other families are
   within +/-0.003 or negative.
6. **The product test (corner deployment) is negative.** Training on one
   estate quadrant and predicting the rest, NDVI-only scores 0.127 (random
   control 0.131, as in `RVI_NDVI.md`). Every temporal set is lower: the best
   NDVI + SAR set reaches 0.117 (descending anomaly features, 3x3 box, 12
   months to 15 July), against 0.125-0.131 for same-size random training. At
   the survey date the same set reaches 0.113 against 0.131 random.
   Small corner training sets (24-63 positives) cannot support the extra features.
7. **Deployment baseline stays NDVI-only LogReg.** The temporal SAR signal is
   not stable on spatial CV and is negative in the corner test. The on-site
   pipeline does not need a second sensor for this estate. A second SAR scene
   or an independent estate would be needed before reconsidering.

## Data & Method

- **Sentinel-1 GRD** (`COPERNICUS/S1_GRD`, IW, VH present): VV, VH and VH-VV
  (dB), split by relative orbit (172 ascending; 91 descending). Point pull
  covers 2025-06-10 to 2026-06-11 (30 ascending, 17 descending scenes); box
  pull covers 2025-06-10 to 2026-07-15 (33 ascending, 18 descending scenes).
- **Sentinel-2 SR Harmonized** (`COPERNICUS/S2_SR_HARMONIZED`): SCL classes
  4-7 kept; NDVI = (B8-B4)/(B8+B4); NDRE = (B8A-B5)/(B8A+B5). 345 scenes in the
  point pull (to 11 June 2026); 381 in the box pull (to 15 July 2026).
- **Sampling:** two versions, compared head to head.
  - *Point:* single 10 m pixel at each tree (stage 0).
  - *Box:* 30 m square around each tree, about 3x3 pixels at 10 m, mean over
    valid pixels (stage 0b). Chosen to reduce speckle at tree scale.
- **Temporal features per tree:** mean, std, CV, slope (per day), delta90
  (last 90 days minus first 90 days), and anomaly versus the estate median on
  each date (anom_mean, anom_std). Features with more than 50% missing trees
  are dropped (for example, descending-orbit features in the 6-month window).
- **Reference:** `ref_ndvi` = stage-2 NDVI composite (15 May to 15 Jul 2026),
  reproduced exactly at 0.1330 on seed 42.
- **Dynamic World** mode label used as a mask. All 2511 trees are labelled
  `trees` over the window
- **Covariates** pulled but not used in any model: SMAP soil moisture (134 am
  dates), ERA5-Land soil water and precipitation, CHIRPS precipitation. The
  SMAP quality-flag discrepancy is documented in the Limitations.
- **Evaluation:** balanced LogReg with StandardScaler, StratifiedGroupKFold(5)
  on KMeans(15) spatial blocks, same as stage 2. Fold PR is the primary metric;
  pooled PR and ROC are secondary.
- **Nested selection:** from each training fold, the candidate temporal
  feature with the highest |Cohen's d| is added to `ref_ndvi`, then scored on
  the test fold. No test data is used for selection.
- **Block seeds:** 10 KMeans seeds (0-9). The reference averages 0.135 across
  seeds, against 0.133 for the single split in stage 2.

## Results

### Stage 1: feature groups (seed 42, 12 months to survey date, point sample)

| Feature set | n features | Fold-mean PR | Pooled ROC | Recall @25% |
|---|---:|---:|---:|---:|
| Stage-2 NDVI reference | 1 | 0.133 | 0.648 | 0.456 |
| + S2 temporal | 15 | 0.132 | 0.592 | 0.419 |
| + S2 and S1 temporal | 54 | 0.121 | 0.599 | 0.431 |
| S2 temporal only | 14 | 0.118 | 0.586 | 0.413 |
| S1 temporal only | 39 | 0.084 | 0.548 | 0.306 |

No-skill PR is 0.064. Feature counts include `ref_ndvi` where it is part of the
set. Adding all temporal features dilutes the NDVI ranking (160 positives
cannot support 54 features).

### Stage 2: nested test (10 block seeds, fold-mean PR)

Reference = 0.135 (10 seeds averaged). Diff = nested minus reference.
"Cells better" = share of seed-fold cells where the nested model beats the
reference on the same test fold.

| Source | Window | Candidate group | Fold PR diff | Cells better | Pooled ROC diff | Most-picked features |
|---|---|---|---:|---:|---:|---|
| 3x3 box | 12m to survey | S1 descending | **+0.011** | 82% | +0.016 | VH mean; VH anom |
| 3x3 box | 12m to survey | S1 all | +0.009 | 68% | +0.006 | VH mean; VH anom |
| 3x3 box | 12m to survey | S1 ascending | +0.003 | 60% | -0.001 | VV/VH dB mean; VV/VH dB anom |
| 3x3 box | 12m matched to 15 Jul | S1 descending | +0.004 | 72% | +0.004 | VH mean; VH anom |
| 3x3 box | 12m matched to 15 Jul | S1 all | +0.002 | 60% | -0.004 | VH mean; VH anom |
| 3x3 box | 6m to survey | S1 ascending | +0.003 | 66% | -0.011 | VH anom; VH mean |
| 10 m point | 12m to survey | S1 all (descending picked) | +0.004 | 60% | +0.007 | VH anom; VV mean |
| 10 m point | 12m matched to 15 Jul | S1 descending | +0.004 | 60% | +0.007 | VH anom; VV mean |
| Any | Any | S2 NDVI/NDRE temporal | -0.003 to -0.005 | 16-44% | -0.013 to -0.021 | NDRE anom |

Note: the 6-month window has no usable descending-orbit features, so
"S1 all" and "S1 ascending" give identical results there. For point-sample
S1 all and S1 descending give identical results because every pick was
descending.

### Interpretation

- The positive cells are largely the same descending-orbit VH features, so the
  signal is consistent in direction but weak in magnitude.
- The gain is about 8% relative to the reference (0.011 / 0.135) and shrinks
  to 2% in the matched window.
- The picked feature changes between folds, so no single SAR feature is
  stable.
- Descending SAR is more informative than ascending SAR across windows.
- Pooled ROC gains are +0.004 to +0.016 for the descending-orbit rows and
  negative for the 6-month and most ascending rows.

### A. Anomaly-only features with regularisation (spatial-block CV)

Fold-mean PR over 10 block seeds, 3x3 box, 12 months to survey date. Diff vs
NDVI-only at the same C; "cells better" = share of seed-fold cells where the
set beats NDVI-only on the same test fold. Full table: `stage3_spatial_checks.csv`.

| Feature set | C = 0.01 PR (diff, cells better) | C = 1.0 PR (diff, cells better) |
|---|---|---|
| NDVI only (reference) | 0.135 | 0.135 |
| S1 anomaly, both orbits + NDVI | 0.136 (+0.001, 60%) | 0.135 (-0.001, 60%) |
| **S1 anomaly, descending + NDVI** | **0.141 (+0.006, 80%)** | **0.141 (+0.005, 70%)** |
| S1 all descending + NDVI | 0.138 (+0.002, 48%) | 0.135 (-0.001, 32%) |
| S2 anomaly + NDVI | 0.128 (-0.007, 28%) | 0.129 (-0.006, 50%) |
| S1 anomaly alone (no NDVI) | 0.095 | 0.093 |
| S2 anomaly alone (no NDVI) | 0.107 | 0.102 |

Descending-orbit anomaly + NDVI is the only set with a consistent positive
sign: +0.003 to +0.006 in every window and source where descending features
exist (the 6-month window has none), over 60-80% of seed-fold cells. Its
pooled ROC is 0.652 against 0.643 for NDVI-only. Anomaly features alone reach
0.09-0.11 PR, well above no-skill (0.064) but below NDVI-only (0.135).
Regularisation does not change the ranking of sets. The families were chosen
after stage 2, so this is a robustness check, not a free search.

### B. Corner deployment (stage-7 protocol)

Train on one quadrant, predict the rest, 3 size-matched random controls.
Mean over the four corners, 3x3 box, 12 months to survey date, C = 1.0:

| Feature set | Corner-train PR | Random-train PR (same size) | Gap |
|---|---:|---:|---:|
| **NDVI only** | **0.127** | 0.131 | -0.004 |
| S1 anomaly, both orbits + NDVI | 0.103 | 0.121 | -0.018 |
| **S1 anomaly, descending + NDVI** | 0.113 | 0.131 | -0.018 |
| S1 all descending + NDVI | 0.109 | 0.119 | -0.009 |
| S2 anomaly + NDVI | 0.100 | 0.129 | -0.029 |
| S1 all descending alone | 0.081 | 0.080 | +0.001 |

Corner by corner, NDVI-only scores 0.148 (NW, 33 positives), 0.137 (SE, 40),
0.138 (NE, 24) and 0.083 (SW, 63), which reproduces the corner lottery in
`RVI_NDVI.md`. Results at the matched 12-month window and C = 0.1 are the same
in direction. The best NDVI + SAR set in any configuration reaches 0.117,
still below NDVI-only.

### Interpretation of the checks

- The spatial gain of descending anomaly features (+0.005) does not survive
  the product test. In the corner test it is -0.018 against random training.
- The corner test is the one that matches deployment, so it takes priority.
- Adding SAR features costs more in corner training than in random training
  because the corner sets have few positives.
- The check confirms `RVI_NDVI.md`: NDVI-only corner training is about as good
  as random training (gap -0.004), while every SAR or temporal addition is
  worse.

## Limitations

- **Single estate and single survey date.** No multi-year data, so
  seasonality cannot be separated from disease with more than one year.
- **Few positives.** 160 Unhealthy trees cannot support more than a few
  features. The nested test limits selection bias but not variance.
- **Multiple comparisons.** The reported best cell is the maximum of 24
  configurations. The true gain is likely smaller.
- **Sparse descending SAR.** Only 18 descending scenes over 12 months, and
  almost none in the last 6 months. Ascending and descending features are
  not fully comparable.
- **SMAP quality flag.** `retrieval_qual_flag` equals 1 on every retrieved
  date, which contradicts the catalogue convention (0 = pass). The mask was
  not applied. SMAP is not used in any model, so this does not affect results,
  but it must be resolved before SMAP is used.
- **Landsat 8 composites** are only 9% valid after cloud fill and were
  excluded.
- **Sentinel-2 at 10 m** mixes crowns, as in the stage-2 report, so NDVI
  signal is conservative.

## Artifacts

- `poc_timeseries_stage0_pull.py` - point-sample GEE pull (S1,
  S2, L8, DW, covariates)
- `poc_timeseries_stage0b_pull_box3x3.py` - 3x3 box pull
  (S1, S2), window to 15 Jul 2026
- `poc_timeseries_stage1_eval.py` - temporal features and
  feature-group evaluation
- `poc_timeseries_stage2_nested.py` - nested selection test over
  sources, windows, groups, and block seeds
- `poc_timeseries_stage3_checks.py` - anomaly-only regularised
  spatial CV (check A) and corner deployment (check B)
- Outputs: `.../TimeSeries/Results`
- Regenerable data: the `ts_*.csv` pulls, which are raw per-tree values from Google Earth Engine.
  Regenerate with
  `poc_timeseries_stage0_pull.py`, 
  `poc_timeseries_stage0b_pull_box3x3.py`
  Check the regenerated scene counts against `stage0_summary.json` and
  `stage0b_summary.json`. Stage 1, 2 and 3 scripts and their outputs depend on
  these files and must be rerun after a re-pull.

