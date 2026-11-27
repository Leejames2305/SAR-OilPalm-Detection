# Oil-palm disease triage — session 261127-s012-open

Generated 2026-11-27 05:13 UTC. All estate results are ordered AirHitam / Palong / Serting. All three estates' ground truth is equally verified and trusted. Older reports' contrary label-quality assumptions are not used.

## Result and scope

The submitted model is **`fusion_control`**, a uniform, regularized logistic ranking recipe combining current optical condition, spatial context, prior-year AlphaEarth contrast, historical fine-scale canopy structure, and label-free optical crown-mixture covariates. It has no estate-name branch, cross-estate label pooling, disease-label neighbor term, pseudo-labels, or supervised Middle process.

Official 80 m L-CV composite lift is **+0.269549**, versus incumbent +0.140519, coordinates-only +0.089593, and carried compact +0.252266. New noisy V lift is **+0.205755**, delta incumbent **+0.087005** and coordinates **+0.139293**. Both standard L-CV and V improve all three estates relative to the supplied incumbent. The largest and most secure gains are Palong and Serting.

The higher-noisy-V bootstrap alternative (+0.213993) is fully disclosed and preserved, but was not selected under a promotion rule written before its V query: its incremental V gain was Serting-only, with AirHitam/Palong essentially unchanged. This is a conservative selection decision, not evidence that the ensemble is intrinsically worse. The selected package is deliberately not described as the maximum observed V score.

This is an in-estate **screening/ranking layer**, not a pathogen diagnosis or a calibrated disease probability. An extreme buffered-region test exposes a Palong failure. Reliable majority capture in AirHitam is still not established. No downstream polarized on-site imaging was implemented.

## Data, integrity and validation design

L contains 2,747 trees: 2,398 scored binary rows (189 Unhealthy, 2,209 Healthy) and 349 Middle. Estate binary counts are 56/964, 66/770, and 67/664 positives. U contains 4,473 trees, including unmarked Middle. Runtime returns every U row in its original order. All 18 derived tables cover the exact 7,220 original `(estate, Long, Lat)` keys; no coordinate rounding, duplicate keys, NaNs, infinities, forbidden columns, or symlinks were found. Total dataset footprint is 83.00 MiB, below the 128 MiB carry-forward threshold. See [dataset audit](dataset_audit.json).

The primary protocol exactly reproduces `agentrsi cv`: five-fold GroupKFold on 80 m blocks computed from all L rows per estate, including Middle when assigning folds. Coordinates in local metres are east = `(Long−minLong)*111320*cos(meanLat)`, north = `(Lat−minLat)*110540`; key = `floor(east/80)*100000+floor(north/80)`. Only Healthy/Unhealthy rows train the model and contribute to binary AP. Middle rows never train a supervised health target and never contribute to binary AP. Base scalers, origin, PCA, and classifiers are fitted on the current fold's binary training rows only.

AP is tie-aware, step-function `average_precision_score`; lift is AP minus scored prevalence. Composite is the unweighted mean of estate lifts. Recall at a budget uses fractional credit for a boundary tie; enrichment at 10% is recall/0.10. We report both binary-scored budgets and budgets where all Middle rows consume capacity. No classification accuracy or threshold-tuned F1 is used. The AP convention follows [scikit-learn's primary documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Additional protocols are 120/160 m blocks, 30°/60° rotated 80 m grids, a 40 m grid-origin shift, 40 m point-buffered official folds, four coordinate-median quadrants, and those quadrants with an 80 m point buffer. The last protocol is named `regions80`. Every fold is disjoint in tree keys; minimum distances, positive counts and training sizes are in [spatial_split_audit.csv](spatial_split_audit.csv). These protocols address the autocorrelation concerns in [Roberts et al., spatial/temporal cross-validation](https://doi.org/10.1111/ecog.02881).

Three frozen assignment series (1010–1019, 1110–1119, 1210–1219) confirm the final recipe against the carried compact model and both baselines. Earlier exploration used official folds, repeat701/702 and wider holdouts. Earlier series810–819 tested the ranking loss;910–919 tested feature fusion. They are not all independent confirmation of the final C=.15 model. Repeated assignments reuse the same labels and imagery, and do not create independent estates or prospective observations. Validation/selection plans and timings are in [validation_plan.json](validation_plan.json), [selection.json](selection.json), and the verifier history.

## Exact submitted recipe

There are 41 encoded predictors, fitted independently per estate with the same recipe:

1. Standardized supplied NDVI and centred local east/north coordinates, expanded to degree two. Nine terms result; all six degree-two terms are multiplied by .3 for additional ridge shrinkage.
2. Twelve compact optical condition/context features, standardized on binary training rows and multiplied by .5: qNDVI, qNDRE, qNDMI, qNIRv, log green/red, three canopy contrasts, NDVI dip, NDVI SD20, NDVI edge, NDRE contrast.
3. All64 prior-year AlphaEarth local-contrast components are standardized on training rows, reduced by deterministic full-SVD PCA to8 training PCs, standardized again, and multiplied by .5. They encode 2025 context, not a current disease measurement. See [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291).
4. Three historical Meta/WRI height covariates: mean within6 m, SD within6 m, and mean6−mean24. Centre each on training rows and divide by `sqrt(training_variance + noise_floor**2)` with floors `[3m,1m,1m]`, then multiply by .35. These floors are conservative regularization heuristics informed by map accuracy, not site-calibrated measurement errors.
5. Nine native-optical covariates from the fixed deblur representation: NDVI, NDRE, NDMI, NBR, NIRv, GRVI and local NDVI/NDRE/NDMI contrasts. Standardize on training rows and multiply by .3.

The classifier is `LogisticRegression(C=.15, solver='lbfgs', max_iter=2000, tol=1e-7)`. Each training class receives total weight N/2. The returned `decision_function` is a risk ranking score. A single-class fallback returns a constant; empty target frames return a finite empty float64 array. No target rows affect scaling/PCA/model fitting. Prepared optical inversion/embeddings use all available covariates label-free, as permitted by the transductive preparation contract.

The sole runtime table is `datasets/triage_v12/features.parquet` (5.80 MiB,109 columns). The final recipe uses88 covariates plus3 keys;18 unused point/stable optical columns remain for exact family ablations. Training PCA reduces64 components to8. [Runtime consolidation script](../datasets/triage_v12/consolidate_runtime.py) and source checksums preserve exact source values. No raw raster, GEE, GPU, /app/data, /app/prior, or network access occurs at runtime. Only `threadpoolctl==3.8.0` is imported beyond the six preinstalled base distributions.

## New covariates and scientific hypotheses

**Fine-scale historical canopy.** A new independent structural source was accessible: Meta/WRI's public canopy map, band `cover_code`, native EPSG:3857 with1.1943 m pixels. Source Maxar imagery spans2009–2020 (mostly2018–2020), so height can encode planting age or management and is not2026 palm height. Circular3/6/12/24 m mean/SD/p90, cover fractions and contrasts were extracted without labels. Zero is valid low/ground height. The model uses only3 compact summaries. Full unit-variance PCA of24 summaries amplified sub-metre variation in Serting and failed wider holdouts; physical uncertainty scaling largely removed that problem. Serting mean6 is about.75 m withSD.84 m, below the3 m height floor. Data/model provenance: [Tolan et al.,2024](https://doi.org/10.1016/j.rse.2023.113888), [official model repository](https://github.com/facebookresearch/HighResCanopyHeight), and `highres_canopy/MANIFEST.md`. NICFI was not accessible to this account; no access bypass was attempted.

**Native optical crown-mixture inversion.** A new CloudScore+ controlled S2 mosaic uses Apr15–Aug15 2026, SCL4–7 and cs_cdf≥.65; fallback uses SCL4–7 over the same window. Keeping bare/soil/water categories avoids conditioning exclusively on vegetation. B2/B3/B4/B8/B5/B8A/B11/B12 median DN are divided by10,000. Harmonization is already applied by [S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); no extra +1000 correction is added. Native10/20 m information remains10/20 m information.

The Earth Engine pixel-coordinate probe showed pixel centres as(.5,.5), so sampleRectangle centre coordinates were explicitly converted to GeoTIFF upper-left corners (−5 m east,+5 m north correction). Coordinates are reprojected to EPSG:32648 before sampling. A fixed5 m Gaussian spectral mixture over up to8 supplied crowns within15 m maps latent crown spectra to observed10 m pixels. Ridge inversion solves `(W.T W+lambda I)(f−f0)=W.T(p−Wf0)` with bilinear point spectra f0 and fixed lambda .25; lambda1 and uninverted point controls were also retained. Reconstructed reflectance is clipped .0001..1, then standard index formulas and nearest9-tree contrasts are computed. Pixel RMSE declines as expected for this fitted inverse problem; that is not independent validation of physical super-resolution. The underdetermined reconstruction does not create measured per-crown resolution, assumes catalogue completeness/Gaussian mixing, can amplify registration noise, and has a weak global influence tail through the inverse. See `crown_mixture/MANIFEST.md`, scripts, [inversion diagnostics](../datasets/crown_mixture/inversion.json), and the coordinate audit.

**SAR-native structure/change.** New3/7/11-pixel statistics for both dates and same-pixel change covered power, HAlpha and Yamaguchi channels. Exact WGS84 keys were reprojected before indexing. Within-date raster geometries agree; October differs slightly in origin/width from June. October was therefore bilinearly reprojected to June in physical linear power before dB differences/stacking, including conversion of Yamaguchi dB to power and back. Finite negative Yamaguchi values were retained and cross-masked with positive power ROI support. All77 m extracted patches had100% support. These features helped AirHitam in some splits but hurt the other estates, especially on wider holdouts, and were rejected.

**Native self-supervision.** A T4 trained two64-dimensional masked denoising convolutional autoencoders (seeds21/42,100epochs each) on22-channel12×12 native SAR patches, plus an11-channel shared paired-date VICReg encoder (100epochs). Inputs were label-free; no labels, coordinates, IDs, or split assignments were uploaded. Frozen June/October/difference encodings and weights are retained in `sar_native`. Downstream8-PC reduction was training-only. This is a small native-SAR reconstruction experiment inspired by [masked autoencoders](https://arxiv.org/abs/2111.06377) and [VICReg](https://arxiv.org/abs/2105.04906), not a pretrained SAR foundation model. Loss declined but disease ranking did not improve consistently. GPU job `job-0001-14ba2d` completed, downloaded its outputs, and the runtime was explicitly stopped. Approximately183 GPU-seconds were charged including overhead; no active GPU remains. Existing RGB DINO assets were retained but not rerun or used by the final model.

**Other orthogonal directions.** Planting regularity/density from all coordinates helped mainly Palong and did not earn inclusion. Bounded feature responses, optical-by-context interactions, RBF manifold bases, older terrain and carried autumn features were mostly weak/inconsistent. A smooth-AP training surrogate lost; a pairwise objective improved slightly, but a matched-curvature C=.15 logistic control was competitive, so the custom ranking optimizer was not retained. Phenotype LabelSpreading used the full covariate catalogue with only training labels clamped and no coordinates in its graph metric; it added no consistent gain. Covariate-distance gates calibrated on training references more than80 m away, bounded residual updates, and paired bootstrap-coefficient variance gates did not cure the distant-region failure. The32-block probability ensemble is the disclosed unselected alternative.

Hyperspectral evidence motivates testing optical pigment/red-edge responses, but is not a resolution-matched satellite performance guarantee. See [MPOB's2018 airborne hyperspectral study](https://doi.org/10.21894/jopr.2018.0037). The supplied Unhealthy target is not assumed to be laboratory-confirmed Ganoderma; no pathogen-specific diagnosis is claimed.

## Official80 m performance and all experiments

| Estate | Prevalence | AP | Lift | Δincumbent | Δcoords | Δcarried | ROC-AUC | Recall25 | Enrichment10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | 5.8091% | 0.124557 | +0.066466 | +0.035501 | +0.055673 | -0.001187 | 0.6781 | 51.79% | 2.321 |
| Palong | 8.5714% | 0.468571 | +0.382857 | +0.127220 | +0.203093 | +0.031291 | 0.8586 | 75.76% | 5.758 |
| Serting | 10.0904% | 0.460229 | +0.359326 | +0.224370 | +0.281101 | +0.021745 | 0.8630 | 76.12% | 4.925 |
| Composite | — | — | +0.269549 | +0.129030 | +0.179956 | +0.017283 | — | — | — |

There were 49 experimental configurations and 61 protocol/assignment identifiers. The complete prediction-derived table is [EXPERIMENTS.csv](EXPERIMENTS.csv), with per-estate lift, AP, ROC, both recall budgets, and paired deltas. [EXPERIMENT_SUMMARY.csv](EXPERIMENT_SUMMARY.csv) contains protocol composites. OOF predictions are saved under `research/results/MODEL_PROTOCOL_ESTATE.npz`. No sealed labels or split assignments were read.

All official80 m configurations follow; names are research recipe identifiers, not additional deployment modes. `base` means the carried compact recipe, `curvature_control` is that recipe with C=.15, `fusion` adds both new families at C=.1, and `fusion_control` is selected C=.15. `point/deblur/stable` use fixed optical lambda0-prior/.25/1 variants; `highres_noise` is the3-feature physical-floor height block. The underlying functions and all other config semantics are in `research/model.py`.

| Recipe | AirHitam | Palong | Serting | Composite | Δincumbent | Δcoords | Δcarried |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ae_mae21 | +0.067587 | +0.354928 | +0.302425 | +0.241647 | +0.101127 | +0.152053 | -0.010619 |
| ae_mae42 | +0.068507 | +0.350018 | +0.337808 | +0.252111 | +0.111592 | +0.162518 | -0.000155 |
| ae_vicdelta | +0.057148 | +0.355132 | +0.304164 | +0.238815 | +0.098295 | +0.149221 | -0.013452 |
| ae_vicjun | +0.071982 | +0.341187 | +0.321046 | +0.244738 | +0.104219 | +0.155145 | -0.007528 |
| ae_vicoct | +0.069197 | +0.322432 | +0.304532 | +0.232054 | +0.091534 | +0.142460 | -0.020213 |
| autumn | +0.066818 | +0.366536 | +0.335912 | +0.256422 | +0.115903 | +0.166829 | +0.004156 |
| base | +0.067653 | +0.351565 | +0.337580 | +0.252266 | +0.111747 | +0.162673 | +0.000000 |
| block_bag | +0.087399 | +0.386420 | +0.328326 | +0.267382 | +0.126862 | +0.177788 | +0.015115 |
| block_uncertainty | +0.070083 | +0.380076 | +0.360215 | +0.270125 | +0.129606 | +0.180531 | +0.017859 |
| bounded | +0.060843 | +0.358353 | +0.291488 | +0.236895 | +0.096376 | +0.147301 | -0.015371 |
| bounded2 | +0.050989 | +0.327261 | +0.244445 | +0.207565 | +0.067046 | +0.117972 | -0.044701 |
| bounded_refine | +0.066449 | +0.383228 | +0.359721 | +0.269800 | +0.129280 | +0.180206 | +0.017533 |
| coords | +0.010793 | +0.179763 | +0.078224 | +0.089593 | -0.050926 | +0.000000 | -0.162673 |
| curvature_control | +0.060842 | +0.368382 | +0.350406 | +0.259877 | +0.119358 | +0.170284 | +0.007611 |
| fusion | +0.074703 | +0.370757 | +0.345683 | +0.263714 | +0.123195 | +0.174121 | +0.011448 |
| fusion_control | +0.066466 | +0.382857 | +0.359326 | +0.269549 | +0.129030 | +0.179956 | +0.017283 |
| fusion_pairwise | +0.069398 | +0.378435 | +0.354697 | +0.267510 | +0.126991 | +0.177917 | +0.015244 |
| fusion_point | +0.073022 | +0.361954 | +0.346154 | +0.260376 | +0.119857 | +0.170783 | +0.008110 |
| fusion_point_control | +0.066700 | +0.376622 | +0.358030 | +0.267118 | +0.126598 | +0.177524 | +0.014851 |
| fusion_point_pairwise | +0.068719 | +0.378556 | +0.353956 | +0.267077 | +0.126558 | +0.177483 | +0.014811 |
| geometry | +0.058682 | +0.398680 | +0.328026 | +0.261796 | +0.121277 | +0.172203 | +0.009530 |
| graph | +0.059564 | +0.344547 | +0.344847 | +0.249653 | +0.109134 | +0.160060 | -0.002613 |
| height | +0.064257 | +0.370750 | +0.325556 | +0.253521 | +0.113002 | +0.163928 | +0.001255 |
| height_pairwise | +0.069260 | +0.378045 | +0.342448 | +0.263251 | +0.122732 | +0.173658 | +0.010985 |
| highres | +0.067017 | +0.357947 | +0.321461 | +0.248809 | +0.108289 | +0.159215 | -0.003458 |
| highres_compact | +0.072736 | +0.363062 | +0.327881 | +0.254560 | +0.114040 | +0.164966 | +0.002293 |
| highres_interaction | +0.061157 | +0.351746 | +0.318025 | +0.243643 | +0.103124 | +0.154049 | -0.008623 |
| highres_noise | +0.071007 | +0.360538 | +0.337581 | +0.256376 | +0.115856 | +0.166782 | +0.004109 |
| highres_noise_control | +0.067379 | +0.380543 | +0.349364 | +0.265762 | +0.125243 | +0.176169 | +0.013496 |
| incumbent | +0.030965 | +0.255637 | +0.134956 | +0.140519 | +0.000000 | +0.050926 | -0.111747 |
| interaction | +0.062427 | +0.369822 | +0.316402 | +0.249550 | +0.109031 | +0.159957 | -0.002716 |
| interaction4 | +0.052620 | +0.361828 | +0.305654 | +0.240034 | +0.099515 | +0.150441 | -0.012232 |
| mixture_deblur | +0.071173 | +0.360654 | +0.346523 | +0.259450 | +0.118931 | +0.169857 | +0.007184 |
| mixture_deblur_control | +0.063725 | +0.374416 | +0.359328 | +0.265823 | +0.125304 | +0.176230 | +0.013557 |
| mixture_pairwise | +0.065325 | +0.368844 | +0.355189 | +0.263119 | +0.122600 | +0.173526 | +0.010853 |
| mixture_point | +0.072792 | +0.353487 | +0.346609 | +0.257629 | +0.117110 | +0.168036 | +0.005363 |
| mixture_stable | +0.071474 | +0.354145 | +0.345331 | +0.256983 | +0.116464 | +0.167390 | +0.004717 |
| pairwise | +0.062937 | +0.361965 | +0.343616 | +0.256173 | +0.115654 | +0.166580 | +0.003907 |
| pairwise_matched | +0.069565 | +0.344972 | +0.332282 | +0.248940 | +0.108420 | +0.159346 | -0.003327 |
| phenotype_graph | +0.061814 | +0.381954 | +0.343814 | +0.262527 | +0.122008 | +0.172934 | +0.010261 |
| phenotype_optical | +0.070843 | +0.379772 | +0.350809 | +0.267141 | +0.126622 | +0.177548 | +0.014875 |
| sar_all | +0.095451 | +0.358672 | +0.317165 | +0.257096 | +0.116577 | +0.167503 | +0.004830 |
| sar_change | +0.076342 | +0.331269 | +0.326458 | +0.244690 | +0.104170 | +0.155096 | -0.007577 |
| sar_contrast | +0.059721 | +0.316128 | +0.332531 | +0.236126 | +0.095607 | +0.146533 | -0.016140 |
| sar_near | +0.074717 | +0.302013 | +0.309877 | +0.228869 | +0.088350 | +0.139276 | -0.023397 |
| sar_yama | +0.063811 | +0.336901 | +0.325621 | +0.242111 | +0.101591 | +0.152517 | -0.010156 |
| softap | +0.054799 | +0.325581 | +0.304095 | +0.228158 | +0.087639 | +0.138565 | -0.024108 |
| support_bounded | +0.080567 | +0.374774 | +0.349536 | +0.268292 | +0.127773 | +0.178699 | +0.016026 |
| support_refine | +0.080635 | +0.375343 | +0.349717 | +0.268565 | +0.128046 | +0.178972 | +0.016299 |

## Confirmation, uncertainty and far-field limits

Frozen-series changes below compare selected fusion against the carried compact model, not the weaker supplied incumbent:

| Frozen series | AirHitam Δ | Palong Δ | Serting Δ | Composite Δ | Estate wins | Composite wins |
| --- | --- | --- | --- | --- | --- | --- |
| objective_confirmation | +0.010217 | +0.005642 | +0.015559 | +0.010473 | 25/30 | 10/10 |
| feature_confirmation | +0.015382 | +0.010751 | +0.016771 | +0.014301 | 29/30 | 10/10 |
| uncertainty_confirmation | +0.018109 | +0.005103 | +0.008415 | +0.010542 | 26/30 | 9/10 |

Across these30assignments, selected fusion beats the incumbent in90/90estate cells, beats carried compact in80/90, and wins29/30composites against carried compact. The mean extra gain over carried compact is about+.0118. These are repeated partitions of the same small sample, not30independent studies. The model/covariates were selected after an adaptive campaign, and neither those repeats nor the bootstrap fully accounts for selection uncertainty.

The following4,000-resample paired80 m block bootstrap conditions on the fixed official OOF predictions. It resamples complete spatial blocks within each estate and reports percentile intervals for AP differences; prevalence cancels within the paired comparison. It does not refit all models, resample independent estates, or remove adaptive-search optimism.

| Reference | Estate | 2.5% delta | 97.5% delta | Fraction delta>0 |
| --- | --- | --- | --- | --- |
| incumbent | AirHitam | -0.001388 | +0.115696 | 0.9688 |
| incumbent | Palong | -0.019282 | +0.292560 | 0.9493 |
| incumbent | Serting | +0.133969 | +0.283241 | 1.0000 |
| incumbent | composite | +0.069260 | +0.190203 | 1.0000 |
| coords | AirHitam | +0.015119 | +0.134158 | 0.9975 |
| coords | Palong | +0.037375 | +0.350855 | 0.9892 |
| coords | Serting | +0.126243 | +0.345622 | 0.9992 |
| coords | composite | +0.100776 | +0.233377 | 1.0000 |
| base | AirHitam | -0.031956 | +0.017683 | 0.4682 |
| base | Palong | -0.013557 | +0.077924 | 0.9167 |
| base | Serting | +0.001567 | +0.040878 | 0.9838 |
| base | composite | -0.003010 | +0.034037 | 0.9487 |

The composite interval versus incumbent is clearly positive. AirHitam and Palong individual incumbent intervals cross zero; Serting is strongest. The extra composite gain over the carried model has a95% interval crossing zero despite positive frozen-assignment means. Do not claim an independent95% proof of this incremental improvement.

| Protocol | AirHitam | Palong | Serting | Composite | Δincumbent | Δcoords | Δcarried |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | +0.066466 | +0.382857 | +0.359326 | +0.269549 | +0.129030 | +0.179956 | +0.017283 |
| 120 | +0.101414 | +0.368093 | +0.269264 | +0.246257 | +0.117402 | +0.154707 | +0.005956 |
| 160 | +0.053981 | +0.300420 | +0.286539 | +0.213647 | +0.099060 | +0.147836 | +0.011096 |
| rotate30 | +0.107547 | +0.349983 | +0.367852 | +0.275128 | +0.133516 | +0.162595 | +0.002743 |
| rotate60 | +0.144521 | +0.375725 | +0.324470 | +0.281572 | +0.129024 | +0.160538 | +0.019293 |
| offset40 | +0.111133 | +0.317971 | +0.286259 | +0.238454 | +0.097547 | +0.170469 | +0.011621 |
| buffer40 | +0.066113 | +0.302368 | +0.275538 | +0.214673 | +0.104129 | +0.199272 | +0.022676 |
| regions | +0.104669 | +0.301842 | +0.315516 | +0.240676 | +0.110749 | +0.185216 | +0.016088 |
| regions80 | +0.061843 | +0.179059 | +0.063091 | +0.101331 | +0.028282 | +0.100571 | -0.005835 |

The selected model beats incumbent and coordinates in all estates in all these protocols except the incumbent comparison for Palong under `regions80`: selected+.179059 versus incumbent+.211215 and carried+.224011. Removing optical inversion but keeping the uninverted point representation still gives Palong+.185651; the core C=.15 control gives+.225295. This implicates contextual/conditional extrapolation of the added covariates rather than attributing the whole failure to global inversion. Distance and bootstrap gates did not resolve it. The model should be used for spatially sampled in-estate screening with local audit coverage, and should not be assumed reliable in a distant unlabeled estate region.

Most final focal features have radii20–40 m and can share imagery across adjacent80 m blocks; the inverse has a global influence tail. Coarser/buffered checks reduce direct neighborhood overlap but do not prove independent imagery or causal disease signal. Performance beyond a15-NN labelled-coordinate prior is evidence of additional ranking information, not proof of an invariant physiological signature.

![Spatial robustness](spatial_robustness.png)

## Noisy V and selection discipline

Four genuinely distinct submissions were scored;11of15queries remain. No formatting/hash reruns or sealed-label recovery were attempted. Noise sigma is.005 and cached by content; differences of a few thousandths are not decisive. Source-family ablations are directionally consistent with retaining both families, but do not establish each family's independent statistical significance.

| V recipe | AirHitam lift | Palong lift | Serting lift | Composite | Δincumbent | Δcoords |
| --- | --- | --- | --- | --- | --- | --- |
| Selected fusion | +0.050384 | +0.368128 | +0.198754 | +0.205755 | +0.087005 | +0.139293 |
| Height family only | +0.040218 | +0.371722 | +0.184752 | +0.198897 | +0.080147 | +0.132435 |
| Mixture family only | +0.044861 | +0.347662 | +0.203323 | +0.198615 | +0.079865 | +0.132154 |
| 32-block ensemble (alternative) | +0.049482 | +0.366903 | +0.225594 | +0.213993 | +0.095242 | +0.147531 |

The supplied incumbent composite V reference is+.118751. The previous session reported+.182472 for the carried compact submission; selected fusion is approximately+.02328 above that independently noisy historical observation. That comparison was not re-queried with an identical/cosmetic package.

Before V4, promotion required a composite increment≥.005, at least2nonnegative estate increments over V1, and no estate loss>.015. V4 composite increment+.008238 passes, but estate changes are−.000902/−.001225/+.026839: only Serting is nonnegative. We therefore retain V1 without changing the rule after observing V. This does not label the tiny AirHitam/Palong changes meaningful degradations; it records insufficient prespecified incremental multi-estate evidence. The ensemble's source is preserved in `datasets/triage_v12/recipes/block_bag.py` so the next session will not face another unavailable stronger-source narrative.

## Inspection budgets, error analysis and feature signs

The actual deployment budget includes unmarked Middle. L-CV diagnostic25% budgets are:

| Estate | Budget population | Inspections | Unhealthy found | Healthy inspected | Middle inspected | Recall | Healthy/found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | binary_scored | 241.00 | 29.00 | 212.00 | 0.00 | 51.79% | 7.31 |
| AirHitam | all_trees_capacity | 241.00 | 29.00 | 212.00 | 0.00 | 51.79% | 7.31 |
| Palong | binary_scored | 192.50 | 50.00 | 142.50 | 0.00 | 75.76% | 2.85 |
| Palong | all_trees_capacity | 242.00 | 44.00 | 101.00 | 97.00 | 66.67% | 2.30 |
| Serting | binary_scored | 166.00 | 51.00 | 115.00 | 0.00 | 76.12% | 2.25 |
| Serting | all_trees_capacity | 203.75 | 51.00 | 92.00 | 60.75 | 76.12% | 1.80 |

AirHitam's51.8% point recall has little margin above a majority and costs about7.31Healthy inspections per positive found. Palong's binary75.8% becomes66.7% when Middle consumes capacity;97of242inspections are Middle. Serting retains76.1% in this particular capacity calculation, with additional Middle costs. These are L-OOF diagnostics, not measured U or V recall. Label-free U scores do not establish true positives, false negatives, or disease prevalence in uninspected trees.

[Inspection-budget diagnostics](inspection_budget_diagnostics.csv) cover5–50% budgets. [Error by nearest training-positive distance](error_by_positive_distance.csv) describes how performance varies with pocket proximity. [Covariate support](covariate_support.csv) flags weak-support score tails without pretending their unknown outcomes are false negatives. Palong still has a low-score unsupported tail; no V-driven tail clipping was performed.

Pocket proximity remains consequential. Palong recall25 is90% for20positives within25m of a training positive,74% for39positives25–80m away,50% for6positives80–150m away, and0/1 beyond150m. Serting falls from86% (22positives below25m) and86% (35positives25–80m) to20% (10positives80–150m). AirHitam is weaker throughout:50% (4),54% (41),45% (11) in those first three bins. These small, spatially dependent subgroups are descriptive, not independent estimates. The distant Serting pocket result reinforces the need for unranked field audits.

The label-free support heuristic flags249/1547AirHitam,258/1696Palong and149/1230Serting U trees. Palong's weak-support median score is−2.71 compared with−1.64 overall, with a minimum−24.41. Outcomes are unknown. Serting's weak-support median is higher rather than lower than its overall median, illustrating that feature-space novelty is not a reliable disease or model-error label. No target labels were used to construct these groups.

![Inspection budgets](inspection_curves.png)

Marginal Cohen's d is Unhealthy minus Healthy, using pooled binary-class SD:

| Feature | AirHitam d | Palong d | Serting d | Same sign |
| --- | --- | --- | --- | --- |
| NDVI | -0.195 | -0.311 | -0.530 | True |
| qNDVI | -0.195 | -0.356 | -0.687 | True |
| qNDMI | -0.137 | -0.218 | -0.718 | True |
| ndvi_dip | +0.082 | +0.549 | +0.698 | True |
| deblur_NDVI | -0.104 | -0.422 | -0.732 | True |
| deblur_NDRE | -0.348 | -0.398 | -0.834 | True |
| deblur_NDMI | -0.051 | -0.219 | -0.657 | True |
| deblur_NBR | -0.082 | -0.212 | -0.633 | True |
| height_mean6 | +0.307 | -0.558 | -0.256 | False |
| height_sd6 | -0.303 | +0.081 | -0.337 | False |
| height_contrast6 | +0.277 | -0.470 | -0.128 | False |
| HH_260610 | -0.147 | -0.176 | -0.084 | True |
| HH_261014 | -0.217 | -0.344 | +0.027 | False |

The optical deficits broadly agree in direction. Historical height changes sign across estates and must be treated as conditional stand context. October HH also flips; no label-quality explanation is invoked. [Full marginal effects](feature_effects.csv), [sign consistency](feature_sign_consistency.csv), [high correlations](feature_correlations.csv), and [conditional coefficients](model_coefficients.csv) are supplied.

Conditional coefficients are not marginal disease effects. For example qNDVI's full-fit1SD conditional effect is about−.236/+.032/−.284 despite negative marginal d in all estates; supplied NDVI remains negative (about−.275/−.730/−1.036), and deblurred NDVI also remains negative. Collinearity among current optical measurements, neighborhood contrasts and spatial interactions redistributes coefficients. Historical mean-height effects are about+.102/−.173/−.009 after uncertainty scaling. Do not delete a feature solely because its conditional sign differs from its marginal sign, or interpret latent PCA signs across estates; PCA axes are estate-specific. Joint regularization and ablations, not isolated coefficient stories, support the recipe.

![Feature signs](feature_signs.png)

## Reproducibility, packaging and runtime

Submission SHA256: `c835ed59e3db34f38cf8485a5b832ee051b49241656081f8369347821ffbd5c8`. This is the pipeline-file hash, distinct from the verifier's whole-package content hash. The packaged pipeline matches the V1 source exactly.18paired source/packaged fits (3full U fits and15official folds) preserve all rankings, with only sub-nanolevel numerical score differences; see `research/candidates/fusion_control/equivalence.json`. Coefficient/scaler summation order can change the last floating-point bits; equality tolerances are disclosed in the scripts.

`agentrsi selfcheck --submission /app/submission` and `agentrsi cv --submission /app/submission` both pass; raw outputs are [selfcheck.txt](selfcheck.txt) and [official_cv.txt](official_cv.txt). Import pins were audited. Fresh per-estate subprocess checks used scratchcwd,4CPUs, blocked network calls, guarded Python writes, and guards against forbidden runtime directory reads; these are invariant checks, not a replacement for kernel-level grading isolation. Runtime is≤0.102s per estate in that audit and peak processRSS≤259.8MiB, well within300s/4096MiB. Input frames remain unmodified; shuffled/duplicated/sharded targets, training-row permutation, repeated runs, CPU1vs4, dropped Middle, empty targets and a single-class fallback pass. See [contract_verification.json](contract_verification.json).

All4,473rankings are saved in [unlabelled_ranking.parquet](unlabelled_ranking.parquet) and [CSV](unlabelled_ranking.csv), preserving original U keys/order. Scores are logit-scale ranking values; `inspection_top25` is an illustrative per-estate ceiling-rounded budget, not a diagnosis or calibrated decision threshold.

Core rerun commands:

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords base fusion_control --protocols 80 120 160 regions regions80
OPENBLAS_NUM_THREADS=1 python /app/research/confirm.py --models incumbent coords base fusion_control --start 1010 --count 10 --name rerun_confirmation
OPENBLAS_NUM_THREADS=1 python /app/research/build_candidate.py --name fusion_control
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/audit.py
OPENBLAS_NUM_THREADS=1 python /app/research/aggregate.py
OPENBLAS_NUM_THREADS=1 python /app/research/diagnostics.py
OPENBLAS_NUM_THREADS=1 python /app/research/plots.py
python /app/research/write_report.py
```

OOF artifacts are cached by recipe/protocol identifier. Delete the corresponding `research/results/MODEL_PROTOCOL_ESTATE.npz` before intentionally changing a recipe and reusing its name. Per-run CSV files avoid parallel append races; `aggregate.py` reconstructs the canonical tables from prediction artifacts. Development dependency versions are in [environment.json](environment.json); only threadpoolctl is required beyond the grading base environment. Plotting uses matplotlib3.11.0 only in research.

To regenerate new covariates, run `extract_highres.py`, `extract_optical_native.py`, `deconvolve_optical.py`, then `consolidate_runtime.py`; unchanged compact source tables have their own extraction/consolidation scripts and manifests. To reproduce the rejected SAR representations, run `extract_sar.py`, upload only `native_sar.npz` to the broker, execute `native_sar_gpu.py`, fetch weights/features/logs, run `compress_native.py` (updating the job-output path), and explicitly stop the GPU. RawGEE/raster acquisition is preparation-only; frozen feature files and their checksums make grading fully offline. Source-provider reprocessing can alter a fresh extraction, so record new checksums rather than silently equating it to this snapshot.

## Deployment limits and next evidence

The large composite gain over the supplied incumbent is supported by spatial OOF, positive conditional composite bootstrap intervals, repeated assignments, and noisy V in all estates. The smaller improvement over the carried compact model is plausible but not independently proven at95%. This campaign evaluated many correlated alternatives on the same189positives; adaptive selection remains a limitation.

Prospective80m-block inspections should cover all estates equally, include weak-support and distant regions, record Middle inspection cost, and reserve a random unranked audit sample. AirHitam is the main majority-capture uncertainty; Palong's distant buffered-region failure needs a targeted audit. Accurate crown registration and genuinely finer current imagery are higher-value next evidence than another means-only SAR/model sweep. Independent estates/years are required before claiming transferability or a disease-specific spectral marker. Source labels and collection methodology are trusted equally throughout.
