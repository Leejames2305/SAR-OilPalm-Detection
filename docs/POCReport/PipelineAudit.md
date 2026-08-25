# Workflow Audit - Why the RF is Still Not Good, and What to Actually Do

**Branch audited:** dev (HEAD d840a2f "feat: Top 20 features selection, RF still less than ideal").
**Target class:** Unhealthy (the only true positive we care about).
**Method:** static code audit + numpy-only POCs (audit/02..05) + the committed datasets
(data/processed/dataset_{Palong,Serting}_Basemap_w3_v4.csv). A scikit-learn install was
attempted but this machine has no PyPI access, so the modeling POCs reproduce the *mechanism*
with a logistic-regression proxy and a nearest-neighbour classifier; the exact RandomForest
numbers are in docs/POCReport/feature_poc_report.md.


## 0. TL;DR

1. The RF apparent performance is almost entirely a **spatial-leakage artifact**, not real
   signal. Under an *honest* spatial-block CV the model collapses to ~no-skill (PR-AUC ~0.15 vs
   0.09 baseline; ROC-AUC ~0.5).
2. **"Main looks better" is a measurement illusion.** main reports random holdout / random CV
   (~PR-AUC 0.39) that leaks spatially-clustered neighbours into the test. dev added
   spatial-block CV and reports ~0.15 - it *got honest*, not *got worse*.
3. The label is **massively spatially autocorrelated**: ~90-92% of trees have a nearest   neighbour
   with the *same* health label at ~8 m spacing, while only ~9% are Unhealthy.
4. **Cross-estate transfer is ~zero**: train on Palong -> test Serting ROC-AUC 0.53; reverse
   0.49. Best discriminative features differ per estate (Palong -> VV-family; Serting -> VH /
   H-alpha / Yamaguchi family). A single-scene, single-estate model does not generalise.
5. **Productive fixes**: (a) reframe to local-health / hotspot maps instead of per-tree binary;
   (b) add multi-temporal SAR (the disease signature is a *change* response); (c) use each
   estate as a CV fold (LOCO). Feature selection / SMOTE / tuning cannot fix the absent
   transferable signal.



## 1. How the current state (dev + top-20) was reached

    main  ->  per-location split -> random 80/20 + random RepeatedStratifiedKFold
              no feature selection, ~20 raw/GLCM/ratio features, RANDOM-CV reported (~PR-AUC 0.39)
       |
       \-- dev: + H/alpha/Yamaguchi decomposition (78 feats)
                + dedup -> impute -> mutual-info top-20 selection (SelectKBest k=20)
                + SPATIAL-BLOCK CV introduced and reported as the HONEST headline (~PR-AUC 0.15)

## 2. What the data actually says (measured)

### A. Positive class is spatially autocorrelated (root cause #1)

    estate   | NN same-label rate | pos-rate | NN median dist
    ---------|-------------------:|---------:|---------------:
    Palong   |       92%          |   9.2%   |     8.3 m
    Serting  |       90%          |   9.2%   |     8.0 m

Trees sit on a tight ~8 m grid; health status is strongly contiguous (disease spreads
locally). Script: audit/02_spatial_corr_nn.py.

### B. Random-split vs spatial-split - why main looked better (leakage)

Nearest-neighbour classifier on coordinates (extreme case - reads only adjacency):

    estate   | random 80/20 acc / F1(U) | spatially-held-out strip acc / F1(U)
    ---------|--------------------------:|-----------------------------------:
    Palong   |      0.90 / 0.49          |          0.82 / 0.28
    Serting  |      0.90 / 0.48          |          0.91 / 0.40

Logistic regression on the 78 SAR features (feature-based proxy; audit/03_...):

    estate   | random ROC-AUC / PR-AUC | spatial ROC-AUC / PR-AUC
    ---------|------------------------:|------------------------:
    Palong   |      0.664 / 0.219      |     0.472 / 0.190
    Serting  |      0.670 / 0.217      |     0.515 / 0.133

Random split looks usable (ROC ~0.66); the spatial split collapses to ~chance (ROC 0.47-0.52,
PR ~0.13-0.19 vs 0.09 baseline). This is the same >2x inflation measured on RandomForest in
feature_poc_report.md (0.39 random -> 0.15 spatial).

### C. Cross-estate transfer is no-skill (root cause #2)

Leave-one-estate-out, logistic on the 78 features (audit/04_...):

    train -> test  | ROC-AUC | PR-AUC | precision@rec0.5
    ---------------|--------:|-------:|-----------------:
    Palong->Serting|  0.527  | 0.098  |     0.103
    Serting->Palong|  0.491  | 0.098  |     0.095

Baseline no-skill precision = pos-rate ~0.09, so the model is effectively random on a new
estate. The univariate analysis shows why: the feature families that separate Healthy/Unhealthy
are different at each estate.

    estate   | top |Cohen's d| features
    ---------|--------
    Palong   | VV, VV_anom, HVVVRatio, VV_p50, RVI, HH_anom (VV / ratio family)
    Serting  | A_std (Yamaguchi aniso), VH_p25, VH_min, HV_p25, VH_anom (VH + decomposition)

### D. Feature engineering is a minor contributor, not the cause

- 26 (Palong) / 43 (Serting) of the 78 features have |Cohen's d| < 0.10 -> tiny effect.
- The newest interferometry features (H_*, alpha_*, P*_std, P*_ratio) have ~0 mutual
  information with the label (also in feature_poc_report.md).
- feature_poc_report.md showed every aggressive reduction (PCA / univariate / RFE / RF-imp
  top-k) *hurts* RF, and the best reducer (MI top-k) only matches 'all features' within noise.
  Pruning cannot fix an absent signal.

### E. The one direction with headroom: aggregate the task spatially (audit/05_...)

Cell/hotspot aggregation (5x3 grid per estate; leave-one-cell-out logistic on cell means;
label = top-tercile unhealthy-fraction):

    estate   | cells | hotspot cells | ROC-AUC | corr(pred, severity)
    ---------|------:|--------------:|--------:|---------------------:
    Palong   |  15   |      5        |  0.80   |        0.60
    Serting  |  15   |      5        |  0.62   |        0.28

Because disease is spatially contiguous, a regional/hotspot framing carries real within-estate
signal even though the strict per-tree spatial split fails. (15 cells is small - treat as
headroom, not a final number.)


## 3. Why dev "looks worse than main" (direct answer)

1. **main never measured transfer**: it reported random holdout / random CV, where
   neighbouring Unhealthy trees leak between train and test, inflating PR-AUC ~2.5x
   (0.39 -> honest ~0.15).
2. **dev added spatial-block CV and made it the headline** -> ~0.15. The drop is the
   measurement becoming honest, not the top-20 selection breaking the model (feature_poc_report
   shows all-raw/dedup90 beat every aggressive reducer under spatial CV too).
3. Minor second-order effects in dev that are real but small:
   - the appended H-alpha / Yamaguchi features are ~0-MI noise that dilutes the informative
     window-stats family;
   - dev trains each estate fully independently (no pooling / no shared baselines), so it can
     never borrow across estates - and there is little to borrow anyway (Sec 2C).

> **Honest bottom line: per-tree, single-scene, single-estate SAR classification of
> 'Unhealthy' is at the information ceiling of this dataset.** The pipeline and feature
> engineering are not the bottleneck.


## 4. Recommended solutions (ranked by expected impact)

### A. Reframe the target, not the trees (do this first - cheap)

- Predict **local-health / disease-hotspot maps** (severity = fraction of unhealthy trees in a
  ~25-30 m block / row) instead of a hard per-tree 0/1. Rationale: decline is spatially
  contiguous; single-tree labels are noisy and spatially duplicated. Cell aggregation is the
  only POC with clear within-estate signal (Sec 2E).
- Keep 'Middle' as a mild/transitional severity (or regress severity 0->1) instead of deleting
  it - deleting it throws away the early-infection gradient, the most useful class for ranking.
- Make the elected metric the **spatial-block / leave-one-estate-out** metric and mark random
  splits as "leakage ceiling". dev already does this for spatial CV - extend to LOCO when
  pooling estates.

### B. Acquire the signal the data lacks (highest biophysical impact)

1. **Multi-temporal SAR (2+ dates per estate)** - the dominant signature of plant disease/stress
   is a *change* response (backscatter / RVI / alpha trajectory, seasonal deltas, CV over time).
   Single-date intensity cannot separate metabolically-stressed from healthy canopies well. This
   is the single biggest lever.
2. **Optical complement (Sentinel-2)**: NDVI / NDWI / chlorophyll change per tree or block.
   SAR alone is a weak disease proxy; a SAR + optical fusion lifts unhealthy recognition far more
   than SAR-only feature engineering.
3. **More estates as validation folds, not more features** - each new acquisition is a LOCO fold;
   aim for >=4-5 estates covering different age / soil to actually test generalisation.
4. **Sharper canopies**: ALOS-2 3 m / stripmap or ~1 m commercial SAR, and snap each sample to a
   canopy-centroid (not the raw tree centroid) so the small 3x3 window sits on the crown.

### C. Pragmatic engineering (cheap, marginal)

- Pool estates and add estate as a categorical / per-estate baseline so location offsets are
  captured; validate with LOCO.
- Use a change / neighbourhood feature set over stacked raw stats; drop RVI / ratios / decomp
  families if spatial-CV PR-AUC does not rise (per the POC they do not help).
- Operational use (we care about **Unhealthy TP**): pick the threshold maximizing recall of
  Unhealthy at acceptable precision on the *spatial* folds, and serve a **probability / heat
  map** (top-k risk regions), not a coarse 0/1 sweep.

### D. Not recommended (target the wrong failure mode)

- More feature selection / SMOTE / class balancing / deeper hyper-parameter tuning: the POC shows
  these re-rank within noise while spatial-CV PR-AUC stays ~0.15. They do not create transferable
  signal that does not exist in a single-date acquisition.

---
