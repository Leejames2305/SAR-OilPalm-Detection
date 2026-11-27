# Oil-palm triage: controlled ablation and simplification

Session **261127-s011-ablation**, 2026-11-27. Final nominee: `submission/pipeline.py`.

## Outcome and scope

The final compact recipe has official 80 m spatial-block composite lift **+0.252266**, versus incumbent **+0.140519** and distance-weighted coordinates15-NN **+0.089593**. Paired gains are **+0.111747** and **+0.162673**. The noisy V composite is **+0.182332**, delta incumbent **+0.063581**, delta coordinates **+0.115870**. Two of 15 V queries used. These are screening rankings, not diagnosis or calibrated disease probabilities.

This session removes the incumbent's estate-name capacity branch and rejects patch-encoder/source-pooling complexity. A controlled single-feature deletion (GRVI_crown) improves fresh repeated-block results and far-field Palong behavior. The final runtime needs one 3.89 MiB covariate table,29 model inputs, base scientific Python and threadpoolctl. No GPU, raw rasters, network, nearest-labelled-tree prior, pseudo-labels, source labels or Middle target is used at inference/training.

All three estates receive equal evaluation weight and their verified labels receive equal trust. Older statements questioning Palong/Serting labels are superseded. The current incumbent is the provided compact model, not the previous session's unavailable nominee. The previous session's user-provided V reward was about+.216825; this session does **not** claim to beat that stronger unavailable code. Its narrative and covariates informed hypotheses, not exact reconstruction. No sealed labels or split recovery was attempted.

## Data, literature and scientific hypotheses

Labelled rows:2747, of which2398 are Healthy/Unhealthy and 349 Middle. Binary counts are964/770/664, positives 56/66/67, prevalence.058091/.085714/.100904 for AirHitam/Palong/Serting. Deployment output has 4473 rows in original order. Exact covariate coverage is7220 unique coordinate keys. No new sensor acquisition occurred. Existing covariate extraction scripts/manifests were read and audited, and a label-free exact subset was consolidated for runtime.

1. **Feature redundancy:** small positive counts favor a few regularized condition/context blocks over stacked foundation representations. Remove a block at a time under identical folds; also test single-feature removals to distinguish useful suppression directions from duplicate variance.
2. **Capacity:** replace the hardcoded AirHitam branch with a uniform differential ridge penalty on quadratic terms. Uniform linear/full-quadratic and spatial-only quadratic controls expose the capacity tradeoff. This is a fixed generic recipe, not an estate gate selected on V. A nested selector was already closed in the previous handoff and was not repeated.
3. **Spatial confounding:** compare to required 15-NN, polynomial coordinates and a richer 32-radial-basis coordinate-only control. Remove optical and coordinate blocks, and widen spatial separation. This tests predictive information beyond simple smoothness; it cannot establish pathogen specificity.
4. **Pooling and Middle:** source-specific intercept/feature augmentation, a source-only optical axis with a paired-label permutation control, and covariate-only PCA inclusion of Middle/all-estate target covariates test whether their added complexity is warranted. No weak-positive Middle experiment was repeated.

Spatial dependence can make random CV optimistic and blocking can induce extrapolation; the validation choices follow [Roberts et al.(2017)](https://doi.org/10.1111/ecog.02881) and the spatial-block rationale of [Valavi et al.(2019)](https://doi.org/10.1111/2041-210X.13107). Spectral stress has a remote-sensing basis, including [Liaghat et al.(2014), hyperspectral basal-stem-rot detection](https://doi.org/10.1080/01431161.2014.903353), but those results do not validate this aerial mixed-pixel model. The annual embeddings are contextual representations from [Brown et al., AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) and its [official Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL). Carried patch covariates used frozen [DINOv2, Oquab et al.](https://arxiv.org/abs/2304.07193), not an encoder trained for oil-palm pathology. Primary pages/metadata are saved as literature_* files. The web search/open endpoint returned HTTP500; primary arXiv/catalogue pages and Crossref publisher metadata were fetched directly. Publisher full pages returned403; no unavailable paper results were inferred.

## Final model and reproducibility

Filter training labels to Healthy/Unhealthy; never drop target rows. Estimate coordinate origin and latitude correction from binary training rows. Standardize baseline NDVI/east/north, construct all 9 degree<=2 nonconstant monomials, and multiply six degree2 terms by.3. This increases their ridge penalty relative to linear terms by 1/.3²≈11.11. Add12 standardized optical variables multiplied by.5: qNDVI,qNDRE,qNDMI,qNIRv,log(B3/B4),three focal-median10 minus focal-median35 index contrasts,NDVI ring-minus-centre dip,SD20,edge and NDRE contrast. Drop the highest-NDVI-pixel GRVI_crown proxy. Fit a training-only StandardScaler/PCA8 on all 64 prior-year AlphaEarth local-contrast components; standardize the 8 PCs and multiply by.5. Optical/latent input scaling corresponds to 4x the linear-base ridge penalty. Fit balanced-sample-weight logistic regression,C=.1,lbfgs,tol1e-7,max_iter2000. Return its decision function as float64. The 29 input coordinates are9 base+12 optical+8 latent. Scores are not probabilities.

All preprocessing that learns a distribution or direction uses only current binary training rows. The selected model does not use `ctx['estate']` or `other_labeled` to select behavior. It uses `ctx['cpus']` through threadpoolctl, and `ctx['datasets_dir']` for every runtime covariate. New estates would need the same covariate extraction/coverage; current saved tables cover these 7220 keys only. Empty targets and single-class training have finite fallback behavior.

`datasets/compact_triage/MANIFEST.md`, `consolidate.py` and `source_checksums.json` document exact sources, windows, masking, geometry and transformations. Source S2 composites have harmonized DN (no extra1000 offset); native red-edge/SWIR20m is not sharpened by 10m sampling. AlphaEarth2025 is previous-year latent habitat/canopy context. No physical disease interpretation is assigned to a PC. Four-offset texture registration integration is fixed and label-free. No Yamaguchi dB or signed-T3 masking was performed because no raw raster feature was extracted this session.

## Validation design and selection history

`research/evaluate.py` reproduces the official `GroupKFold(5)` splits, including Middle rows in fold allocation, and then excludes Middle before AP/ROC/recall scoring. Local metres use (longitude-min)*111320*cos(mean latitude) and (latitude-min)*110540; block key=floor(east/80)*100000+floor(north/80). Every fold asserts no shared block. AP is sklearn's step-function tie-aware average precision; lift subtracts the estate's binary prevalence and the composite is an unweighted mean. Inspection-boundary ties receive fractional expected credit. No random tree split or same-block neighbor stacking is used.

Exploratory assignments:official,repeat1,repeat2. Repeats shuffle whole group assignments with GroupKFold, preserving the same80m grid. Wider checks use120/160m, shifted80m grid,40/80m nearest-held-out-tree buffers, and contiguous quadrants assigned by whole80m block position. Coarse blocks alone still permit close trees along boundaries; the buffer checks enforce separation explicitly. Quadrants are approximate halves of block-coordinate extents, not stratified by labels. The 80m buffer can leave as few as2 training positives in a Palong fold; its lower skill is a stress test, not a directly comparable sample-size estimate.

The two frozen initial candidates (strong_shrunk2 and alpha_stronger) were evaluated on repeat100..109 before any V query. Both beat incumbent and 15-NN in 30/30 estate-assignment comparisons. Source-axis follow-up used repeat200..209; small gains were inconsistent and rejected. Single-feature pruning was evaluated on fresh repeat300..309 after freezing the GRVI deletion. All 10 composite differences favored pruning,25/30 estate cells favored pruning, and all 30 cells beat both incumbent and 15-NN. The model was frozen after V2. `validation_plan.json` records declared assignments. Repeated folds reuse the same labels and spatial landscape; they are robustness checks, not30 independent trials. The feature/capacity search was adaptive (56 named configurations), and all uncertainty is selection-conditioned. The 10 seeds were not used to tune parameters further.

## Controlled experimental table

All values below are **official 80m AP lift**, with paired deltas computed from the same held-out assignments. Model definitions are in `research/model.py`; all per-estate/protocol metrics and paired deltas are in `EXPERIMENTS.csv`. `OFFICIAL_EXPERIMENTS.csv` includes every official candidate. These ablations are exploratory except the declared confirmation stages.

| Recipe | AirHitam | Palong | Serting | Composite | Delta incumbent | Delta coords |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | +0.03097 | +0.25564 | +0.13496 | +0.14052 | +0.00000 | +0.05093 |
| coords | +0.01079 | +0.17976 | +0.07822 | +0.08959 | -0.05093 | +0.00000 |
| coords_poly | +0.01164 | +0.14273 | +0.05890 | +0.07109 | -0.06943 | -0.01850 |
| coords_rbf | +0.00763 | +0.18290 | +0.10393 | +0.09816 | -0.04236 | +0.00856 |
| optical | +0.09543 | +0.30998 | +0.18424 | +0.19655 | +0.05603 | +0.10696 |
| quality | +0.06907 | +0.29985 | +0.15361 | +0.17418 | +0.03366 | +0.08458 |
| local | +0.07355 | +0.27942 | +0.18322 | +0.17873 | +0.03821 | +0.08914 |
| alpha | +0.08285 | +0.35068 | +0.22686 | +0.22013 | +0.07961 | +0.13054 |
| wide | +0.06468 | +0.33782 | +0.22782 | +0.21010 | +0.06958 | +0.12051 |
| wide_pool | +0.06639 | +0.32666 | +0.22310 | +0.20538 | +0.06487 | +0.11579 |
| core_alpha | +0.05785 | +0.32344 | +0.21861 | +0.19997 | +0.05945 | +0.11038 |
| alpha_stronger | +0.06880 | +0.36788 | +0.27826 | +0.23832 | +0.09780 | +0.14872 |
| strong_middle_cov | +0.06880 | +0.36660 | +0.28110 | +0.23883 | +0.09831 | +0.14924 |
| strong_trans_cov | +0.06728 | +0.37014 | +0.28204 | +0.23982 | +0.09930 | +0.15023 |
| strong_linear | +0.06880 | +0.29743 | +0.25476 | +0.20700 | +0.06648 | +0.11740 |
| strong_full2 | +0.04736 | +0.36788 | +0.27826 | +0.23117 | +0.09065 | +0.14158 |
| strong_shrunk2 | +0.06174 | +0.35166 | +0.33572 | +0.24971 | +0.10919 | +0.16012 |
| uniform_noalpha | +0.07739 | +0.32058 | +0.24930 | +0.21576 | +0.07524 | +0.12617 |
| uniform_no_optical | +0.01433 | +0.30284 | +0.18896 | +0.16871 | +0.02819 | +0.07912 |
| uniform_core | +0.04374 | +0.31814 | +0.31884 | +0.22691 | +0.08639 | +0.13732 |
| uniform_nojitter | +0.05577 | +0.29632 | +0.24028 | +0.19746 | +0.05694 | +0.10786 |
| uniform_noquality | +0.03729 | +0.31167 | +0.31209 | +0.22035 | +0.07983 | +0.13075 |
| uniform_no_sde | +0.04626 | +0.29564 | +0.28772 | +0.20987 | +0.06935 | +0.12028 |
| uniform_plus_dino | +0.04691 | +0.36323 | +0.32638 | +0.24551 | +0.10499 | +0.15591 |
| uniform_axis | +0.06159 | +0.37017 | +0.33630 | +0.25602 | +0.11550 | +0.16643 |
| uniform_axis_only | +0.03240 | +0.33495 | +0.24352 | +0.20362 | +0.06311 | +0.11403 |
| uniform_perm_axis | +0.00960 | +0.30052 | +0.16327 | +0.15780 | +0.01728 | +0.06820 |
| uniform_noedge | +0.05387 | +0.34972 | +0.32973 | +0.24444 | +0.10392 | +0.15485 |
| uniform_nosd | +0.05667 | +0.32763 | +0.31808 | +0.23413 | +0.09361 | +0.14454 |
| uniform_nonirv | +0.06552 | +0.36125 | +0.33733 | +0.25470 | +0.11418 | +0.16511 |
| uniform_nogrvi | +0.06765 | +0.35157 | +0.33758 | +0.25227 | +0.11175 | +0.16267 |

Interpretation: broad patch addition (`wide`, `uniform_plus_dino`) is not a consistent improvement. Source feature augmentation and a source-only axis do not justify final complexity. AlphaEarth adds genuine predictive headroom in this selected in-estate recipe, but it is contextual information, not established disease physiology. The broad 13-feature physical block improves over quality-only or local-only, and aggressive five-feature pruning loses lift. Removing the correlated SD/edge pair hurts, even though their marginal signs flip. Their incremental utility is conditional and estate-specific; do not call them universal stress markers. The successful crown-GRVI removal is a narrower, validated simplification. Dropping NIRv gives a small official win but fails broader tradeoffs, so it is retained. No high-margin feature family remained safe to remove after these checks.

## Frozen confirmation and spatial robustness

| estate | mean_lift | min_lift | max_lift | mean_delta_incumbent | min_delta_incumbent | mean_delta_coords | min_delta_coords | mean_recall25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | +0.09962 | +0.07472 | +0.12028 | +0.07508 | +0.04995 | +0.07381 | +0.05072 | +0.49821 |
| Palong | +0.34072 | +0.28980 | +0.37454 | +0.09511 | +0.04310 | +0.18854 | +0.12133 | +0.71667 |
| Serting | +0.26764 | +0.20144 | +0.30539 | +0.13483 | +0.08710 | +0.21257 | +0.18046 | +0.66866 |

Mean pruning deltas over repeat300..309, AirHitam/Palong/Serting: +0.007981/+0.015257/+0.006355. The final mean confirmation composite is+0.235993; it beats both comparators in 30/30 estate cells. Its repeated AirHitam Recall25 averages49.8%, which is below a dependable majority-capture claim.

| protocol | AirHitam | Palong | Serting | Composite | delta_incumbent | delta_coords |
| --- | --- | --- | --- | --- | --- | --- |
| official | +0.06765 | +0.35157 | +0.33758 | +0.25227 | +0.11175 | +0.16267 |
| shift40 | +0.09334 | +0.32230 | +0.26486 | +0.22683 | +0.08593 | +0.15885 |
| block120 | +0.07307 | +0.38220 | +0.26563 | +0.24030 | +0.11145 | +0.14875 |
| block160 | +0.04028 | +0.30522 | +0.26214 | +0.20255 | +0.08796 | +0.13674 |
| buffer40 | +0.06005 | +0.27709 | +0.23885 | +0.19200 | +0.08145 | +0.17660 |
| buffer80 | +0.07263 | +0.16087 | +0.11604 | +0.11651 | +0.07388 | +0.08991 |
| quadrant | +0.09449 | +0.28771 | +0.26950 | +0.21723 | +0.07290 | +0.16558 |

Final improves every estate over incumbent on every listed protocol. It improves every estate over15-NN except AirHitam120m: final+.07307 versus coordinates+.08592 (about-.01285). That exception and the V tie prevent claiming universal separation from a spatial prior in AirHitam. Polynomial and richer RBF coordinate controls remain substantially below final composite. `spatial_split_audit.csv` records row/positive counts and minimum distances; minimum official distances can still be6–8m across block edges. Buffered80m composite remains+.11651 versus incumbent+.04262 and coordinates+.02662, but finite-sample instability is substantial.

A paired 4000-replicate bootstrap resamples80 m blocks within each estate, using fixed OOF scores and identical resampling weights for comparators. Composite95% delta incumbent: **[+0.05320,+0.17716]**; delta coordinates: **[+0.08742,+0.21849]**. Individual AirHitam and Palong intervals versus incumbent cross0; Serting's is positive. All three intervals versus 15-NN are positive in this conditional calculation. Resampling does not refit the model, correct the adaptive search, or remove adjacent-block dependence. `bootstrap.json` includes all individual intervals and caveats. This supports a consistent composite improvement, not independent per-estate significance.

## Verifier evidence

| estate | noisy_pr_auc_lift | noisy_delta_vs_incumbent | noisy_delta_vs_coords |
| --- | --- | --- | --- |
| AirHitam | +0.03914 | +0.00211 | -0.00066 |
| Palong | +0.33490 | +0.08159 | +0.20709 |
| Serting | +0.17296 | +0.10704 | +0.14117 |

V1(unpruned) composite+.180935; V2(pruned)+.182332. Their difference is below the Gaussian noise scale and does not independently prove the small pruning gain. Selection favors the simpler model because of the predeclared local paired result, not this tiny V difference. The verifier's noisy AirHitam delta incumbent is+.002109 and delta coordinates-.000655: effectively tied. Palong and Serting improve incumbent by+.081594/+.107040. Consequently V clearly corroborates a two-estate improvement with no material reported degradation, while AirHitam improvement remains uncertain. Noise sigma=.005; no cosmetic-hash resubmissions or V weight sweeps occurred.

V2 used `research/candidates/pruned/pipeline.py` with four original datasets. Final packaging reads one exact consolidated table. `package_audit.json` proves bit-identical76 input covariates for all 7220 keys, exact full-fit predictions for all 3 estates and exact predictions in all 15 official OOF folds. V2 content hash:32b74d9621404f79e306404bc3ff56132ca3b2818a1f26dc9c63f57dc744c27f. Final pipeline file SHA256:a0641192bd99c08e063899900ad81f785681dc0475f8457106369e2d2a0ce5c8. Different packaging is not presented as a new V observation.

## Inspection budgets, errors and sign consistency

Official binary Recall25 is55.4%/75.8%/74.6%; ROC-AUC is0.670/0.848/0.852. Top10% enrichment is1.79x/5.61x/4.48x. These are OOF estimates, not the outcomes of 4473 unlabelled predictions.

When Middle consumes inspection capacity (without being scored as Healthy):

| estate | recall | precision | other_inspections_per_unhealthy |
| --- | --- | --- | --- |
| AirHitam | +0.55357 | +0.12863 | +6.77419 |
| Palong | +0.68182 | +0.18595 | +4.37778 |
| Serting | +0.70149 | +0.23067 | +3.33511 |

Middle inspection overhead lowers Palong/Serting recall and precision. Here precision means verified-Unhealthy fraction among all reviewed trees; other inspections include Middle, whose inspection value is unspecified. AirHitam's roughly6.8 other inspections per detected Unhealthy at25% capacity remains costly. `inspection_budget_diagnostics.csv` supplies10–50% curves with/without Middle. `unlabelled_ranking.csv/.parquet` preserves the original4473 target rows and provides an illustrative within-estate top25% allocation; it is not an optimized cross-estate resource policy.

`feature_effects.csv` and `feature_sign_consistency.csv` contain all raw-covariate Cohen's d. qNDRE,qNDVI,qNDMI,qNIRv are lower in Unhealthy across estates; NDRE contrast is negative and NDVI dip positive in all three. SD20 and edge reverse signs (negative AirHitam, positive Palong/Serting). Marginal signs do not identify causal mechanisms, and combined fitted coefficients can reverse under correlated predictors. `model_coefficients.csv` documents conditional optical coefficients and differently penalized basis terms. AlphaEarth/DINO component axes are latent, may change with PCA sign/estate, and their signs are not given physiological meanings.

`error_by_distance.csv` shows recovery versus nearest training tree/positive. Distance is used for diagnostics only, never as a leaked stacking feature. `covariate_support.csv` reports target values outside binary training ranges. Some Palong trees have extreme low polynomial scores (minimum about-22.9); outside-range contrasts/texture and NDVI tails mean confidence is unjustified. Outcomes are unknown, so these are extrapolation flags, not proven false negatives. Clipping/tail fitting was a previous closed trial and was not retuned on V. Prospective inspection should include an audit sample outside the high-ranked fraction and in weak-support regions.

## Runtime, data and reproduction audit

Final selfcheck and officialCV pass (`research/final_selfcheck.log`, `final_cv.log`). Fresh-process4-CPU contract checks pass for float64/finite/exact length, input immutability, determinism, target shuffle/duplicate invariance, Middle ignored, estate-name/source-label independence and empty/single-class fallback. All-check process peak is below 237 MiB. Shape-changing BLAS products can vary at floating-point roundoff: row-order/duplicate comparisons use1e-10 tolerance; an initial exact-bit duplicate assertion was corrected in the research test, not by changing predictions. Same-input determinism remains exact. No runtime writes to submission/datasets were observed.

Only explicit non-base import is `threadpoolctl==3.8.0`, pinned. Matplotlib3.11.0 is research-only. Final runtime is CPU/offline, with no source-data/prior/raster path and no network API. All 12 derived tables have7220 exact keys, zero duplicates/NaNs/forbidden columns/symlinks; all datasets total42.06 MiB, below 128 MiB carry-forward target and 512 MiB cap. Unused carried datasets are retained for provenance/reuse but are not runtime dependencies. No GPU jobs or GEE calls were launched. Incumbent and prior reports remain unchanged.

Reproduce:

```bash
python /app/datasets/compact_triage/consolidate.py
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords uniform_nogrvi --protocols official repeat300 repeat301 repeat302 repeat303 repeat304 repeat305 repeat306 repeat307 repeat308 repeat309
OPENBLAS_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords uniform_nogrvi --protocols block120 block160 shift40 buffer40 buffer80 quadrant
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python /app/research/diagnostics.py
OPENBLAS_NUM_THREADS=1 python /app/research/coefficients.py
OPENBLAS_NUM_THREADS=1 python /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python /app/research/package_audit.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
python /app/research/audit.py
python /app/research/plots.py
python /app/research/write_report.py
```

## Limits and next evidence

This is an in-estate condition/context triage recipe selected on a small spatial sample. Disease causality, new-estate/year transfer, probability calibration and guaranteed majority recall are not established. Nearby canopy imagery may be shared across fold boundaries; wider/buffered checks reduce that concern but cannot replace new data. V improvements are smaller than localCV and concentrate in two estates. The statistically positive composite conditional interval does not establish every individual estate gain. The earlier stronger nominee is not available for paired comparison. No on-site polarized-imagery system was built because it is out of scope.

The highest-value next step is prospective whole-block inspection in all 3 estates with uniform trusted protocols, including unranked and unsupported-covariate audit trees. More independent estates/years and registration ground control have greater value than another feature or hyperparameter search on these same56–67 positives. Preserve this compact nominee and its evidence; do not retune it to a third V response.
