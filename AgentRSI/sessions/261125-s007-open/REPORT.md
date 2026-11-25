# Oil-palm disease triage — session 261125-s007-open

The submitted model improves public-L spatial-block composite PR-AUC lift from **0.14052 to 0.16608**, with positive estate-level gains on all eight tested 80 m block assignments. The official paired composite gain is **+0.02556**, with a conditional 95% block-bootstrap interval **[+0.00678, +0.05156]**. The offline model, derived features, extraction scripts and validation evidence are preserved.

**The final model has no V score.** The verifier's derived-dataset staging fails before execution, including for the unchanged incumbent. This report claims an L-validation improvement, not a verified V improvement or promotion. The original incumbent is unchanged. AirHitam also retains an operational limitation: improved AP and top-10% enrichment come with lower Recall@25% on the official split.

## Data, endpoint and label trust

All three estates' labels are verified and equally trusted. No estate is discounted as derived or noisy. L contains 2,209 Healthy, 189 Unhealthy and 349 Middle rows. The scored binary subset has 964/770/664 rows and 56/66/67 positives in AirHitam/Palong/Serting. Their prevalences are 0.05809/0.08571/0.10090. There are 50/39/13 scored 80 m blocks respectively; Serting's spatial effective sample size is especially small.

Middle is excluded from all final fitted losses and metric targets. Its coordinates remain in fold construction, matching the CLI. All 4,473 unlabelled rows, including unmarked Middle palms, receive aligned scores. The endpoint is ranking Unhealthy above Healthy; no accuracy target or threshold-tuning claim is used. AP is tie-aware step-function average precision, not trapezoidal PR area; lift is AP minus estate prevalence. The composite weights estates equally. PR and ROC answer different questions under imbalance; see [Saito and Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432).

![Public spatial coverage](public_spatial_coverage.png)

Median distances from unlabelled palms to a labelled palm are 31.6/31.8/30.9 m; 90th percentiles are 67.8/85.0/67.9 m. Block interpolation is relevant, while region holdout checks a more difficult extrapolation regime.

## Final training recipe

`/app/submission/pipeline.py` fits each estate independently with the same recipe. It uses supplied NDVI and coordinates plus the carried `gee_s2_stress` covariates. There are no estate-name branches, embedded supervised coefficients, held-out labels or identifier features. `ctx['other_labeled']` is available but unused by the final model.

Six candidates are available in every invocation:

1. **Base, linear or quadratic:** standardized supplied NDVI and centred local-metre coordinates; optional full degree-2 interactions; balanced logistic regression, C=0.1.
2. **Prior-season raw optical, linear or quadratic base:** the base plus standardized 2025-05-15 to 2025-07-15 B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12. Auxiliary bands enter linearly, avoiding a large quadratic band-interaction basis. Balanced logistic regression, C=0.1.
3. **Current red-edge, linear or quadratic:** supplied NDVI, NDRE1 and coordinates standardized and optionally expanded jointly to degree 2. NDRE1=(B8A−B5)/(B8A+B5), from 2026-05-15 to 2026-07-15. Balanced logistic regression, C=0.03.

Training-only inner CV uses five-fold 80 m GroupKFold and a second five-fold block assignment with seed 29. Every scaler, imputer and classifier is fitted only on the relevant inner training rows. Label-free composites may cover all public tree coordinates; this does not expose validation labels.

For model averaging, 128 deterministic Bayesian bootstrap replicates place independent exponential weights on whole training spatial blocks. Each replicate computes exact weighted AP on both inner assignments, averages them, and subtracts a shared 0.005 AP complexity prior from the optical families. Winning configurations vote; vote frequencies become model weights. The weighted AP implementation matches sklearn on random weighted tied scores to 2.2×10⁻¹⁶.

The final score is **75% Bayesian model average plus 25% uniform quadratic base model**. The anchor reduces the unanchored ensemble's Serting region-holdout loss and is identical in every estate. The recipe performs at most 60 inner classifier fits plus six final fits and an anchor fit. Model vote fractions describe selection uncertainty, not clinical risk. Returned blended logits are ranking scores, not calibrated disease probabilities.

Full-L unanchored votes illustrate the adaptation: AirHitam selects linear prior bands; Palong assigns about 89% to quadratic prior bands; Serting assigns about 90% to quadratic NDVI+NDRE1. These are learned decisions rather than hardcoded estate choices (`../research/bayes_full_choices.json`).

## Validation discipline and uncertainty

`../research/common.py` exactly reproduces the CLI's local geometry: x=(Long−min Long)×111320×cos(mean Lat), y=(Lat−min Lat)×110540; group=floor(x/80)×100000+floor(y/80). The official split is deterministic five-fold GroupKFold over all estate L rows, with Middle removed only before fitting/scoring. Comparators use identical folds and scored rows.

Seven additional assignments use shuffled GroupKFold seeds 17, 83, 101, 203, 307, 409 and 503. The latter five seeds were declared before evaluating the first frozen spectral selector. They test partition robustness, not independent new labels: later recipe revisions and exploratory comparisons still share the public L sample. A four-region test holds out quadrants defined by median east/north position. Random row K-fold and same-block leave-one-out spatial priors are not used. Block validation and its extrapolation caveats are motivated by [Roberts et al., 2017](https://doi.org/10.1111/ecog.02881).

The primary paired bootstrap resamples whole scored spatial blocks with replacement within each estate, calculates weighted AP differences from saved OOF predictions, and combines estates with equal weight. There are 5,000 replicates versus the incumbent and 3,000 versus coordinates. These percentile intervals are conditional on OOF predictions: they do not refit the models, fully account for overlapping CV training sets, or correct for exploratory recipe selection. Individual AirHitam and Serting intervals cross zero. Independent V/prospective labels remain necessary for a deployment-level superiority claim.

## Primary results

Official five-fold 80 m CV lift and paired differences:

| estate | incumbent_lift | coords15_lift | lift | delta_incumbent | delta_coords15 |
| --- | --- | --- | --- | --- | --- |
| AirHitam | 0.03097 | 0.01079 | 0.03834 | 0.00737 | 0.02754 |
| Palong | 0.25564 | 0.17976 | 0.30596 | 0.05032 | 0.12619 |
| Serting | 0.13496 | 0.07822 | 0.15393 | 0.01898 | 0.07571 |
| Equal-estate mean | 0.14052 | 0.08959 | 0.16608 | 0.02556 | 0.07648 |

Conditional paired block-bootstrap uncertainty:

| comparison | delta | 95% interval | fraction positive |
| --- | --- | --- | --- |
| AirHitam vs incumbent | 0.00737 | [-0.01249, +0.04697] | 0.78620 |
| Palong vs incumbent | 0.05032 | [+0.01419, +0.09777] | 0.99680 |
| Serting vs incumbent | 0.01898 | [-0.01338, +0.07043] | 0.80100 |
| composite vs incumbent | 0.02556 | [+0.00678, +0.05156] | 0.99800 |
| composite vs coords15 | 0.07648 | [+0.01300, +0.12385] | 0.99100 |

Fractions of positive bootstrap replicates are descriptive uncertainty summaries, not adjusted hypothesis-test p-values.

| protocol | lift | delta_incumbent | delta_coords15 |
| --- | --- | --- | --- |
| official | 0.16608 | 0.02556 | 0.07648 |
| seed17 | 0.18567 | 0.02978 | 0.10674 |
| seed83 | 0.18618 | 0.03366 | 0.11021 |
| region | 0.14504 | 0.01512 | 0.08958 |
| seed101 | 0.16131 | 0.02032 | 0.06578 |
| seed203 | 0.12734 | 0.02811 | 0.05871 |
| seed307 | 0.17563 | 0.03359 | 0.09071 |
| seed409 | 0.15772 | 0.02236 | 0.06672 |
| seed503 | 0.17558 | 0.02719 | 0.07490 |

All **24 estate × block-assignment** primary comparisons against the incumbent are positive. Mean results over the eight assignments are:

| estate | lift | delta_incumbent | delta_coords15 | r25 | e10 |
| --- | --- | --- | --- | --- | --- |
| AirHitam | 0.06046 | 0.03442 | 0.02751 | 0.45982 | 2.40625 |
| Palong | 0.28381 | 0.02438 | 0.13733 | 0.75000 | 4.45076 |
| Serting | 0.15654 | 0.02391 | 0.07901 | 0.60075 | 3.19403 |

Four-region lift is 0.03372/0.29489/0.10653, composite 0.14504. AirHitam and Palong improve; Serting loses 0.00472 versus its incumbent 0.11125. This is a small observed loss, with uncertainty amplified by only 13 scored Serting blocks; no universal non-degradation guarantee is implied.

![Spatial gains and uncertainty](spatial_gain_uncertainty.png)

## Inspection budgets and error analysis

Official operating diagnostics (`r25` is tie-averaged Recall@25%; `e10` is Recall@10%/0.10):

| estate | ap | roc | r25 | e10 | incumbent r25 | incumbent e10 |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | 0.0964 | 0.6585 | 0.4107 | 2.1429 | 0.4821 | 1.2500 |
| Palong | 0.3917 | 0.8332 | 0.7424 | 4.8485 | 0.7424 | 4.5455 |
| Serting | 0.2548 | 0.7401 | 0.6119 | 3.4328 | 0.6119 | 2.7463 |

| estate | model | budget_for_recall50 | precision25 | inspections_per_positive25 |
| --- | --- | --- | --- | --- |
| AirHitam | final | 0.3143 | 0.0954 | 10.4783 |
| AirHitam | incumbent | 0.2739 | 0.1120 | 8.9259 |
| Palong | final | 0.1052 | 0.2545 | 3.9286 |
| Palong | incumbent | 0.1260 | 0.2545 | 3.9286 |
| Serting | final | 0.1604 | 0.2470 | 4.0488 |
| Serting | incumbent | 0.1694 | 0.2470 | 4.0488 |

Official AirHitam top-10% inspection finds 12/56 positives versus 7 for the incumbent, but top-25% finds 23 versus 27. Its budget for 50% recall rises from 27.4% to 31.4%. Palong's corresponding budget falls from 12.6% to 10.5%, and Serting's from 16.9% to 16.0%. These are retrospective diagnostics, not selected deployment thresholds. At top-25%, AirHitam still requires about 10.5 inspections per positive; its practical overhead remains high.

Across all eight assignments, mean Recall@25% is 0.4598/0.7500/0.6007 versus incumbent 0.3996/0.7633/0.5728. Thus the official AirHitam recall loss is a real fold-specific tradeoff rather than a consistent recall decline. Hard/soft training-only recall guards failed to repair that loss and reduced AP, so they were rejected.

![Inspection budget curves](inspection_budget_curves.png)

Distance-dependent errors remain. Among held-out positives more than 80 m from a training Unhealthy palm, the top-25% list finds 2/11 AirHitam, 1/7 Palong and 0/10 Serting. Within 40 m, it finds 10/15, 29/35 and 29/32. These small descriptive strata were not selection criteria. Unseen disease pockets remain difficult (`error_by_distance.csv`).

Middle is excluded before every reported budget calculation. Actual unlabelled lists include unmarked Middle, so their real workload may be diluted relative to binary CV metrics. The downstream polarized-imagery inspection system remains out of scope.

## Iterative research campaign

Directions covered spatial smoothness, honest Middle geometry, corrected multi-date L-band power/decomposition/texture, compact optical physiology, new October optical covariates, and ranking-oriented losses. Representative official results follow. `EXPERIMENTS.csv` holds all 220 named variants/checkpoints, including aliases and algebraic blends; these are exploratory comparisons, not 220 independent hypotheses.

| name | AirHitam | Palong | Serting | composite | delta incumbent | delta coords15 |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | 0.03097 | 0.25564 | 0.13496 | 0.14052 | 0.00000 | 0.05093 |
| coords15 | 0.01079 | 0.17976 | 0.07822 | 0.08959 | -0.05093 | 0.00000 |
| rbf_k4_w0.6_C0.01 | 0.01805 | 0.22757 | 0.08803 | 0.11122 | -0.02930 | 0.02162 |
| middle_honest_b40_C0.03 | 0.03097 | 0.23511 | 0.15423 | 0.14010 | -0.00042 | 0.05051 |
| fusion_tab_change_a0.1 | 0.03117 | 0.24863 | 0.13634 | 0.13872 | -0.00180 | 0.04912 |
| fusion_yam_fraction_a0.3 | 0.01756 | 0.25150 | 0.13731 | 0.13546 | -0.00506 | 0.04586 |
| fusion_texture_a0.3 | 0.03148 | 0.22757 | 0.11129 | 0.12344 | -0.01707 | 0.03385 |
| fusion_s2_indices_a0.3 | 0.03596 | 0.26337 | 0.13593 | 0.14509 | 0.00457 | 0.05549 |
| fusion_s2_prior_a1.0 | 0.07861 | 0.31266 | 0.11393 | 0.16840 | 0.02788 | 0.07880 |
| optical_n_r_C0.03 | 0.03258 | 0.25387 | 0.18197 | 0.15614 | 0.01562 | 0.06655 |
| selector_soft_ap_i2_m0.005 | 0.04323 | 0.29952 | 0.15643 | 0.16639 | 0.02587 | 0.07680 |
| bayes_p0 | 0.04547 | 0.30180 | 0.16845 | 0.17191 | 0.03139 | 0.08232 |
| final | 0.03834 | 0.30596 | 0.15393 | 0.16608 | 0.02556 | 0.07648 |
| autumn_change_a0.3 | 0.02659 | 0.24703 | 0.13820 | 0.13727 | -0.00325 | 0.04768 |
| autumn_contrast_a0.3 | 0.03461 | 0.25270 | 0.12551 | 0.13760 | -0.00291 | 0.04801 |
| focal_g1.0_C0.1 | 0.03084 | 0.25238 | 0.13633 | 0.13985 | -0.00067 | 0.05026 |
| recallguard_t0.02_s0 | 0.03732 | 0.28522 | 0.14577 | 0.15610 | 0.01558 | 0.06651 |

- **Spatial smoothing:** RBFs, splines, additive versus interacting polynomials generally lose. A weak Student-density blend had tiny local gains but negative noisy V; rejected (`spatial.py`, `context.py`).
- **Middle:** direct KDE and block-honest auxiliary Middle-frequency features were tested. Training context removes whole blocks, avoiding same-block self-neighbour leakage. Gains are mostly Serting-specific and Palong is unstable; excluded (`context.log`).
- **L-band:** June/October log-power levels and changes, HAlpha changes, correctly masked Yamaguchi dB/fractions, percentile spread and crown-background contrasts were strongly shrunk additive hypotheses. None supplies repeatable multiple-estate gain (`fusion_tab.log`, `fusion_sar.log`).
- **Optical:** uniformly applying prior-season raw bands materially hurts Serting; uniformly applying NDVI+NDRE1 loses some Palong skill. Single-assignment nested gates remain unstable. Repeated inner block CV with uncertainty-aware model averaging combines the complementary mechanisms (`adaptive.py`, `selector.py`, `bayes_selector.py`).
- **New October S2:** physiological changes, canopy maximum/median and local contrasts add no consistent gain across estates (`autumn.log`).
- **Loss/recall:** focal-loss emphasis within the incumbent's tiny basis has no consistent benefit; recall-guarded selection lowers AP without repairing official AirHitam recall (`focal.log`, `recall_guard*.log`).

The anchor sacrifices some official mean AP (unanchored 0.17191 versus final 0.16608) for a smaller Serting region loss (0.01822 reduced to 0.00472). It also removes the incumbent's estate-name complexity branch. At this checkpoint, further directions showed diminishing returns.

## Sensor versus spatial signal and feature signs

Ablations rerun the same nested selection/anchor with inputs or model families removed:

| name | AirHitam | Palong | Serting | composite |
| --- | --- | --- | --- | --- |
| final | 0.03834 | 0.30596 | 0.15393 | 0.16608 |
| ablation_sensor_only | 0.03689 | 0.18617 | 0.12161 | 0.11489 |
| ablation_coords_only | 0.02527 | 0.15133 | 0.06240 | 0.07967 |
| ablation_no_prior | 0.03012 | 0.25639 | 0.16467 | 0.15039 |
| ablation_no_rededge | 0.04650 | 0.30616 | 0.10659 | 0.15308 |

The prior-band family supplies much of AirHitam/Palong's gain; the red-edge family supplies Serting's gain. Optical-only skill and superiority over two coordinate-only comparators show that this is not solely a spatial smoothing upgrade. Raw bands may encode stand age, soil, canopy structure or management as well as disease. These ablations do not establish a causal disease signature or new-estate transfer.

Cohen's d is Unhealthy minus Healthy divided by pooled within-class standard deviation:

| feature | AirHitam | Palong | Serting |
| --- | --- | --- | --- |
| NDVI | -0.195 | -0.311 | -0.530 |
| s2_NDRE1 | -0.274 | -0.133 | -0.702 |
| s2_prev_B8 | 0.300 | 0.368 | -0.088 |
| power_HV_db_260610_r1_mean | -0.107 | 0.067 | 0.223 |
| power_HV_db_change_r1_mean | -0.035 | 0.123 | -0.368 |
| ha_H_change_r1_mean | 0.174 | 0.006 | -0.217 |
| yam_Pv_frac_260610_r1_mean | 0.089 | 0.128 | 0.145 |
| yam_Pv_frac_change_r1_mean | -0.011 | 0.220 | -0.265 |

Current NDRE1 remains negative after within-block centering: d=−0.192/−0.464/−0.541. Prior B8's within-block association is approximately zero in AirHitam despite a positive marginal d. Several SAR/change signs flip across estates despite equally trusted labels. The small sign-consistent volume-fraction effect did not translate into repeatable additive ranking gain. Full descriptive effects, not used to select the frozen model, are in `feature_effects.csv`.

Red-edge exploration has physiological motivation: a UAV hyperspectral BSR study found useful discrimination in selected crown segments and a few red-edge wavelengths. This is not validation of satellite-scale performance: [Early Detection of Basal Stem Rot Disease in Oil Palm Tree Using UAV-Based Hyperspectral Imaging, 2022](https://doi.org/10.3390/rs14030799). Yamaguchi features follow surface/double-bounce/volume/helix power in the [four-component scattering model](https://doi.org/10.1109/TGRS.2005.852084), which alone does not establish disease specificity. The binary Unhealthy label should not be interpreted as pathogen-specific diagnosis.

## Derived data and masking

The final model requires only `gee_s2_stress`: 7,220 exact unique public-tree keys, 54 columns, zero NaNs. Both seasonal windows use `COPERNICUS/S2_SR_HARMONIZED`, median composites and SCL classes 4–7, with extra bands sampled at 20 m. Supplied NDVI keeps its original 10 m sampling. Native red-edge/SWIR resolution is 20 m, coarser than approximately 9 m palm spacing; no super-resolution or crown independence is claimed.

The harmonized collection already adjusts post-baseline-04.00 DN changes. **No additional +1000 offset is applied**; the older catalogue survey must not be interpreted as instructing a second offset. Source and band resolutions: [official Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED).

New reproducible datasets, both excluded from the final model:

- **alos_physics:** 7,220 × 611, zero NaNs, approximately 36.4 MiB. WGS84 points are reprojected to each raster CRS. Backscatter uses finite positive support and per-pixel 10·log10(power). Yamaguchi retains signed dB and is cross-masked by positive finite backscatter support, reprojected with nearest-neighbour resampling. Component fractions use 10^(dB/10), normalized per pixel before averaging. Radius-1/2/4 windows supply mean, standard deviation, median and p90−p10, requiring at least five valid pixels. Changes are October minus June; contrasts are radius-1 minus radius-4. The AirHitam-only T3 raster was not newly extracted or used.
- **gee_s2_autumn:** 7,220 × 28, zero NaNs, approximately 0.8 MiB. Sentinel-2 window 2026-09-14 through 2026-11-14, explicit 10 m UTM grid with native 20 m band caveats. Includes raw bands/indices, radius-10 m maximum/median and centre-minus-radius-30 m median contrasts.

The carried `alos_winstats` has 593,334 NaNs, entirely in 126 T3 columns absent outside AirHitam. Its finite-only Yamaguchi mask and signed-dB ratios were flagged; it is unused. Carried topography is unused; its temporary `row` sequence must not be modeled. Manifests were amended with these caveats; the prior MainChecklist was not changed.

All five feature tables have exact 7,220-tree key coverage, zero duplicate keys, zero forbidden columns and no symlinks. Total derived data is about 53.2 MiB, below both the 128 MiB carry-forward threshold and 512 MiB cap. See `dataset_audit.json`; each dataset retains its manifest and extraction script, with masking/NaN audits for new data. No labels or supervised transformations enter extraction.

## Verifier limitation

Budget used: **3/15**, with noise sigma 0.005 cached per content hash.

1. A rejected 10% Student spatial-density blend returned valid noisy V lift **0.110036**, Δ incumbent **−0.008715**, Δ coordinates **+0.043574**. This is not the final nominee's score.
2. A small optical candidate with valid staged derived data passed the complete dataset audit, then failed before execution: `OSError: [Errno 30] Read-only file system: '/shared/s007_datasets'`.
3. The **unchanged incumbent** with the same valid data at `/shared/datasets` failed identically. The error is independent of candidate model logic. A canonical `/app/datasets` argument was rejected by the path guard without consuming a query.

No identical scoring repetitions or cosmetic hash variants were used to average noise. `/shared/s007_nominee` contains the final code/pin; `/shared/s007_datasets` contains its required compact dataset. Retry after host staging is repaired. The provided incumbent V composite remains +0.118751; the new nominee has no V score. No sealed labels/split assignments, forbidden repository or test files were accessed.

## Runtime checks and reproduction

`agentrsi selfcheck` and official `agentrsi cv` pass. The explicit non-base import is pinned as `threadpoolctl==3.8.0`; all other imported packages are supplied by the runtime. The PyPI release was verified available and supports Python >=3.10. Native threads are capped at min(4, ctx['cpus']).

Fresh scratch-directory subprocesses passed with networking disabled and writes guarded to scratch. Two complete fits took 1.7–2.1 seconds per estate, peak resident memory approximately 234 MiB, below 300 seconds/4 GiB. Target-row permutation and inverse restoration changed predictions by at most 4.5×10⁻¹⁶. Returned arrays are finite float64 with exact target lengths and ordering (`contract_verification.json`). Prediction loads covariates only via `ctx['datasets_dir']`; it needs no `/app/data`, `/app/rasters` or `/app/prior` access.

```bash
python -m pip install -r /app/submission/requirements.txt
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/validate_final.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/diagnostics.py final --reference incumbent_all --reps 5000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/diagnostics.py final --reference coords15_all --reps 3000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/summarize.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/plots.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/write_report.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/verify_contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/predict.py
```

Extraction/development versions are in `environment.json` and `../research/requirements-dev.txt`; they are not prediction-runtime requirements. Earth Engine needs authentication only during extraction. Credentials are never saved or printed. The final code performs no network calls or file writes during fitting.

## Decision and remaining limits

The nominee meets the local primary-metric evidence criteria: positive paired composite interval, improvements in all three estates on all eight block assignments, no estate-level primary loss on those assignments, and advantages over coordinate comparators with informative optical ablations. The small Serting region loss and official AirHitam Recall@25% decline remain explicit exceptions to broader operational non-degradation. Shared-label repeated CV and conditional bootstrap do not substitute for independent validation.

This is an improved **research screening recipe**, not confirmed diagnosis or a promise of majority detection in every estate at 25% inspection. New-estate transfer, pathogen specificity, prospective calibration and inspection costs with unmarked Middle remain unvalidated. The immediate next step is a repaired blind V evaluation, followed by independent spatial/prospective labels rather than another unconstrained feature sweep.

Aligned predictions for all 4,473 unlabelled rows are in `unlabelled_ranking.parquet` and `unlabelled_ranking.csv`, with within-estate ranks and an illustrative top-25% flag. They are predictions, not scored outcomes. The original incumbent is retained in `/app/incumbent`.
