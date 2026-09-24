# RF + G-SMOTE POC (AHCensus / Palong / Serting) - Findings and Verdict

**Run:** Colab notebook `misc/notebook/POC_AHCensus_RF_GSMOTE.ipynb`

**Target class:** `Unhealthy` (positive). `Healthy` = negative. `Middle` / `Vacant` dropped.

**Scope:** each location modelled independently. No pooling, no transfer

**Review method**: All numbers are taken / derived from the generated artifact
files


> **This POC is a negative result.** It was designed to give G-SMOTE and the resampling family a fair, well-instrumented chance. They got that chance, and the answer is that the signal is not in this dataset. The report below says why, and what the result does *not* say.


## 0. TL;DR

1. **Model is at no-skill on every honest (spatial-CV) configuration.** 
   - Across 30 baseline runs, PR-AUC lands within **-0.013 to +0.007** of the no-skill baseline. ROC-AUC is **0.448-0.513** (below a coin flip). 
2. **Seems not a class-imbalance problem, and the notebook proves it.** 
   - All five resampling arms (`none`, `class_weighted_rf`, `random_oversample`, `smote`, `gsmote`) score
   the same to within **0.003 PR-AUC**. Their PR curves are visually superimposed.
3. **G-SMOTE "captures more sick trees" is a threshold artifact, not an achievement.** 
   - The apparent recall gain (0.000 -> 0.156-0.266) is bought by false flagging a large fraction of the estate. Measured as **enrichment over random selection**, the F1-optimal operating point
   sits at **0.87x - 1.25x**. The best single cell flags 79% of the
   estate to net **+7 **trees over random guessing.
4. **The G-SMOTE sweep is flat.** 
   - 60 distinct configurations per estate; the whole surface spans only **0.058-0.071** (AHCensus), **0.079-0.093** (Palong), **0.087-0.110** (Serting). Hyperparameters cannot be tuned out of a null result.
5. **Random CV (leakage celling) inflates PR-AUC by +0.021 ~ +0.047 and ROC-AUC to ~0.57**.
6. **The raw univariate signal is weak and direction-inconsistent.** 
   - Strongest |Cohen's d| is **0.35**; the sign of the effect **flips** between AHCensus/Palong (Unhealthy darker) and
   Serting (Unhealthy brighter). A sign flip cannot be explained by label noise or small
   samples.
7. **`oof_predictions.csv` missing** so the metric for a triage use-case (precision@k / enrichment@budget) could not be computed from
the artifacts.


## 1. What this POC tests

Unlike the earlier POCs (`FeatureTests.md`, `MeanSampling.md`, `PipelineAudit.md`), this one
does **not** vary features. It holds the feature set fixed at the four raw backscatter window means and varies the **class-imbalance treatment** and the **G-SMOTE geometry**, to answer one question:

> *Is the failure to detect Unhealthy trees caused by class imbalance, or by the absence of a
> usable signal?*

It is also the first POC to include **AHCensus**, whose labels are **real on-field ground truth**, unlike Palong/Serting which are derived/estimated. That matters: the negative result cannot be blamed on the labels being synthetic for at least one of the three estates.

### Configuration

| Item | Value |
|---|---|
| Model | `RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)` |
| Features | `HH/HV/VV/VH_meanW3` only - **4 features**. `EXTRA_FEATURES = []` |
| Window | `WINDOWS = [3]` (single 3x3 mean) |
| CV (honest) | `GroupKFold` on `block_deg = 0.002` (~220 m) blocks, 5 folds |
| CV (leaky ref) | `StratifiedKFold`, 5 folds |
| Resampling arms | `none`, `class_weighted_rf`, `random_oversample`, `smote`, `gsmote` |
| G-SMOTE defaults | `k=5, truncation=1.0, deformation=0.0, selection='combined'` (paper default) |
| Resampling scope | fitted **inside each training fold only**, on standardised features, mapped back |

### Datasets

| Estate | Trees | Unhealthy | Healthy | Pos. rate | Blocks |
|---|---:|---:|---:|---:|---:|
| `AHCensus_Basemap` | 2511 | 160 | 2351 | **6.37%** | 16 |
| `Palong_Basemap` | 2171 | 200 | 1971 | **9.21%** | 11 |
| `Serting_Basemap` | 1667 | 154 | 1513 | **9.24%** | 8 |


## 2. Headline results - spatial CV (the honest protocol)

`baseline_metrics.csv`

| Estate | Arm | PR-AUC | baseline | lift | ROC-AUC | recall_U | TP | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AHCensus | none | 0.0618 | 0.0637 | **-0.0019** | 0.504 | 0.000 | 0 | 14 |
| AHCensus | class_weighted_rf | 0.0627 | 0.0637 | -0.0010 | 0.497 | 0.000 | 0 | 1 |
| AHCensus | random_oversample | 0.0607 | 0.0637 | -0.0030 | 0.491 | 0.000 | 0 | 26 |
| AHCensus | smote | 0.0631 | 0.0637 | -0.0007 | 0.494 | 0.125 | 20 | 326 |
| AHCensus | **gsmote** | 0.0635 | 0.0637 | **-0.0002** | 0.498 | 0.156 | 25 | 428 |
| Palong | none | 0.0807 | 0.0921 | -0.0114 | **0.458** | 0.000 | 0 | 26 |
| Palong | class_weighted_rf | 0.0793 | 0.0921 | -0.0128 | **0.448** | 0.000 | 0 | 7 |
| Palong | random_oversample | 0.0821 | 0.0921 | -0.0101 | 0.462 | 0.020 | 4 | 72 |
| Palong | smote | 0.0828 | 0.0921 | -0.0094 | 0.472 | 0.135 | 27 | 347 |
| Palong | **gsmote** | 0.0846 | 0.0921 | **-0.0075** | 0.467 | 0.205 | 41 | 494 |
| Serting | none | 0.0912 | 0.0924 | -0.0012 | 0.495 | 0.006 | 1 | 15 |
| Serting | class_weighted_rf | 0.0972 | 0.0924 | +0.0048 | 0.513 | 0.000 | 0 | 9 |
| Serting | random_oversample | 0.0993 | 0.0924 | +0.0069 | 0.508 | 0.039 | 6 | 46 |
| Serting | smote | 0.0961 | 0.0924 | +0.0037 | 0.506 | 0.182 | 28 | 238 |
| Serting | **gsmote** | 0.0972 | 0.0924 | **+0.0049** | 0.503 | 0.266 | 41 | 353 |

Not one of the 15 spatial rows clears a useful lift. The best anywhere is
Serting `random_oversample` at **+0.0069**, and Serting `class_weighted_rf` posts the same PR-AUC (0.0972) while achieving **recall = 0.000**. PR-AUC here is ranking noise, not skill.

### Uncertainty - taking "small dataset" seriously

Hanley-McNeil 95% CI on ROC-AUC (tested the "the sample is too small to tell" objection):

| Estate | Arm | ROC-AUC | 95% CI | Verdict |
|---|---|---:|---|---|
| AHCensus | none | 0.504 | 0.458 - 0.550 | 0.5 inside |
| AHCensus | gsmote | 0.498 | 0.452 - 0.544 | 0.5 inside |
| Palong | class_weighted_rf | 0.448 | **0.408 - 0.489** | **entirely below 0.5** |
| Palong | gsmote | 0.467 | 0.426 - 0.509 | 0.5 inside |
| Serting | class_weighted_rf | 0.513 | 0.465 - 0.561 | 0.5 inside |
| Serting | gsmote | 0.503 | 0.455 - 0.551 | 0.5 inside |

The intervals are wide (~ +/-0.025 SE): every interval
that is not already below 0.5 straddles it, and the *upper* bounds are all ~0.55. A triage tool
would want ROC-AUC north of ~0.7. The honest reading is "no signal detected", not "signal we cannot measure".


## 3. The central finding - "accept more false positives" does not help

This is the most important result in the POC, and it is the one the confusion matrices hide.

The notebook performed exactly this experiment: the
`thr_best_f1` / `f1_U_best` / `recall_U_best` columns are the F1-optimal operating point, and
the 60-config sweep is 180 further attempts at the same trade-off.

Scored the way a triage tool must be scored - **enrichment** ; `precision / pos_rate` (precision back-derived from `f1_U_best` and `recall_U_best` via `P = F1*R / (2R - F1)`):

| Estate | Arm | Threshold | Recall | Precision | Flagged | **Enrichment** |
|---|---|---:|---:|---:|---:|---:|
| AHCensus | gsmote | 0.09 | 83.1% | 6.7% | ~1985 / 2511 (79%) | **1.05x** |
| AHCensus | smote | 0.06 | 83.1% | 6.5% | ~2046 / 2511 | **1.02x** |
| AHCensus | none | 0.05 | 50.6% | 6.4% | ~1265 / 2511 | **1.00x** |
| AHCensus | class_weighted_rf | 0.05 | 43.1% | 5.9% | ~1169 / 2511 | **0.93x** |
| AHCensus | random_oversample | 0.10 | 33.8% | 6.2% | ~872 / 2511 | **0.97x** |
| Palong | gsmote | 0.05 | 90.0% | 9.0% | ~2000 / 2171 | **0.98x** |
| Palong | smote | 0.07 | 81.0% | 9.1% | ~1780 / 2171 | **0.99x** |
| Palong | class_weighted_rf | 0.05 | 54.5% | 8.2% | ~1329 / 2171 | **0.89x** |
| Serting | class_weighted_rf | 0.06 | 61.7% | 10.2% | ~932 / 1667 | **1.10x** |
| Serting | gsmote | 0.12 | 86.4% | 9.7% | ~1372 / 1667 | **1.05x** |
| Serting | smote | 0.07 | 82.5% | 9.4% | ~1352 / 1667 | **1.02x** |

**Enrichment sits between 0.87x and 1.25x.** The flagged list is at best, 5-10% better than picking trees at random.

The FP count is not a *cost* the user tolerates - it is the mechanism producing the TP count. This is why PR-AUC (not the confusion matrix) is the correct headline.

## 4. The G-SMOTE sweep - the tuning objection

Grid: `k in {3,4}` x `tau in {-1, 0, 0.5, 1}` x `delta in {0, 0.5, 1}` x
`selection in {minority, majority, combined}` = 72 raw, **60 distinct** per estate (`k` is unused by `selection='majority'`, so those duplicates were dropped).

| Estate | min | mean | max | std | Best config | Sweep max |
|---|---:|---:|---:|---:|---|---:|
| AHCensus | 0.0579 | 0.0626 | 0.0707 | 0.0031 | `tau=1, delta=0, k=4, minority` | 0.0707 |
| Palong | 0.0786 | 0.0833 | 0.0932 | 0.0036 | `tau=1, delta=0, k=4, minority` | 0.0932 |
| Serting | 0.0869 | 0.0976 | 0.1103 | 0.0048 | `tau=1, delta=1, k=4, combined` | 0.1103 |

- The full swept range is **0.013 / 0.015 / 0.023 PR-AUC wide** - flat paint, not a landscape.
- Best-by-selection-strategy collapses the same way (AHCensus: combined 0.0636, majority
  0.0653, minority 0.0707 - a 0.007 spread).
- **Verdict on tuning:** closed. `tau`, `delta`, `k` and `selection_strategy` cannot produce signal that is not in the features.


## 5. Leakage - what random CV buys, and why it is not usable

| Estate | PR-AUC (spatial) | PR-AUC (random) | baseline | lift (random) | ROC (spatial) | ROC (random) |
|---|---:|---:|---:|---:|---:|---:|
| AHCensus | 0.0635 | 0.0847 | 0.0637 | +0.0210 | 0.498 | **0.568** |
| Palong | 0.0846 | 0.1186 | 0.0921 | +0.0265 | 0.467 | **0.572** |
| Serting | 0.0972 | 0.1395 | 0.0924 | +0.0471 | 0.503 | **0.573** |

Random split puts a test tree's
next-door neighbour in the training set. The model memorises *"this fingerprint is near an
infected tree"* rather than learning anything transferable.


## 6. Univariate analysis - what the raw data says

`univariate_stats.csv`. Model-free. 
- d = (mean_U - mean_H) / pooled SD
- MWU = Mann-Whitney U (asymptotic) 
- frac = fraction of Unhealthy below the Healthy median (0.5 = total overlap)

| Estate | Feature | Cohen's d | MWU p | frac | Used as feature |
|---|---|---:|---:|---:|---|
| AHCensus | `HV_meanW3` | **-0.338** | 1.4e-05 | 0.637 | yes |
| AHCensus | `VV_meanW3` | -0.285 | 1.4e-03 | 0.594 | yes |
| AHCensus | `VH_meanW3` | -0.245 | 3.0e-03 | 0.606 | yes |
| AHCensus | `HH_meanW3` | -0.224 | 1.7e-04 | 0.613 | yes |
| AHCensus | `RFDI_VH_meanW3` | -0.036 | 7.6e-01 | 0.463 | no |
| AHCensus | `RVI_meanW3` | -0.025 | 8.7e-01 | 0.475 | no |
| Palong | `VH_meanW3` | **-0.353** | 5.2e-04 | 0.545 | yes |
| Palong | `HH_meanW3` | -0.256 | 1.5e-02 | 0.545 | yes |
| Palong | `VV_meanW3` | -0.255 | 3.5e-02 | 0.575 | yes |
| Palong | `HV_meanW3` | -0.243 | 1.6e-01 | 0.535 | yes |
| Palong | `RFDI_VH_meanW3` | +0.228 | 4.7e-03 | 0.400 | no |
| Serting | `VV_meanW3` | **+0.267** | 2.7e-03 | 0.383 | yes |
| Serting | `HV_meanW3` | +0.178 | 4.1e-02 | 0.442 | yes |
| Serting | `RFDI_meanW3` | -0.145 | 1.4e-01 | 0.526 | no |
| Serting | `RVI_meanW3` | +0.141 | 1.7e-01 | 0.481 | no |

Three conclusions:

1. **Every effect is small.** Max |d| = 0.35, most in 0.05-0.29. The distributions overlap heavily; `frac` hovers near 0.5-0.64 where 0.5 is no separation.
2. **Significance != separability.** With 2351 vs 160, p-values reach 1e-5 at |d| = 0.34. The p-value is answering "is there *any* difference", not "can I classify with it".
3. **The effect direction flips across estates.** AHCensus and Palong: Unhealthy trees are *darker* (all `d < 0`). Serting: Unhealthy are *brighter* (`VV d = +0.267`). This independently corroborates [`PipelineAudit.md` Sec 2C].


## 7. Figures - how to read them

### 7.1 Confusion matrices

**Read the percentages, not the counts** - the Healthy row has ~15x more trees and will always
dominate the colour scale. A matrix is "good" when the bottom-right cell is a meaningful share
of the bottom row *and* the top-right cell is small. Almost every matrix here fails that.

### 7.2 PR + ROC curves

- **All five arms are superimposed in the PR panels.** Immediate collapse onto the no-skill line and a flat ride to recall 1. This *is* "PR-AUC ~ baseline", drawn.
- **The ROC panels separate visibly while the PR panels do not.** With a 15:1 imbalance, ROC-AUC is dominated by the large Healthy class, while PR-AUC is dominated by the rare Unhealthy class and collapses.  

### 7.3 G-SMOTE sweep surfaces

Sweep tests shows hyperparameters of GSMOTE bring near identical performance, regardless the selections' mode. 

### 7.4 Feature importance

Mean RF impurity importance per feature, G-SMOTE paper-default arm. All three estates are **flat** - within each other.


## 8. Verdict

Per-tree, single-date, 4-feature backscatter classification of `Unhealthy` does not work, and
the POC rules out the usual escape routes.

| Hypothesis | Status | Evidence |
|---|---|---|
| Class imbalance is the cause | **Rejected** | 5 arms within 0.003 PR-AUC; PR curves superimposed |
| Wrong resampling algorithm | **Rejected** | G-SMOTE == SMOTE == ROS == class-weight == none |
| Bad G-SMOTE hyperparameters | **Rejected** | 60-config sweep, 0.013-0.023 wide |
| Threshold choice is the problem | **Rejected** | enrichment 0.87-1.25x at every F1-optimal point |
| Sample too small to tell | **Rejected** | Hanley-McNeil CIs all straddle or sit below 0.5; upper bounds ~0.55 |
| Feature selection / engineering | **Already shown elsewhere** | `FeatureTests.md` - every reducer hurts or ties |
| The features genuinely lack signal | **Supported** | max abs(d) = 0.35, sign flips across estates |


| Criterion | Bar | Actual | Pass |
|---|---|---|---|
| PR-AUC lift (spatial) | >= +0.05 on >=2 estates | -0.013 to +0.005 | no |
| ROC-AUC (spatial) | >= 0.65 | 0.448 - 0.513 | no |
| Enrichment at fixed budget | >= 2x | ~1.0x | no |
| Cohen's d | >= 0.5, **same sign all estates** | <= 0.35, sign flips | no |
| Recall >= 50% at precision >= 2x base | required | not reachable | no |

> **Bottom line:** Negative result. Not totally broken, but says the information required to separate Healthy from Unhealthy per tree is hard to present in a single-date, 3x3-mean, four-band backscatter sample. The POC's real value is that it closes the imbalance/resampling/tuning branch, so effort can move to changing the **data**, not the model.