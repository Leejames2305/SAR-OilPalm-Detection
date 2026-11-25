# Oil-palm disease triage — session261125-s009-open

The final offline model improves ranking over the incumbent in all three estates on official80m CV, all11 repeated block assignments per estate, and the held-out noisy verifier. It combines cloud-controlled optical measurements, prior-year foundation embeddings, and registration-integrated canopy-neighborhood texture. It is a screening/inspection ranking, not a pathogen diagnosis or calibrated disease probability. All estates' verified labels receive equal scientific trust.

## Data and starting state

L has2747 trees:964/968/815 by AirHitam/Palong/Serting, including198/151 Middle in the latter two. Binary scored totals964/770/664, positives56/66/67. U has4473 rows and is spatially disjoint on the supplied80m design. Middle remains in all output rankings and is never included in binary metric scoring. The official CLI includes Middle while assigning folds, then excludes it from scoring; our evaluator reproduces that procedure exactly. No sealed labels or split assignments were accessed.

The incumbent is unchanged. Its official estate lifts are .03097/.25564/.13496, composite .14052. Only gee_quality, gee_canopy, gee_s2_stress actually carried into this container; previous pipeline/source and SAR tables named in s008's handoff were absent. We reconstructed the optical mechanism; it is not a byte-identical recovery of s008. Previous reports' claims that Palong/Serting labels were less trustworthy are explicitly rejected.

## Final reproducible recipe

1. Keep Healthy/Unhealthy for target binary fitting. Average provided NDVI and gee_quality qNDVI_p75 equally. Fit standardized NDVI+local-metre coordinates using balanced logistic regression C=.1. Preserve the incumbent's degree1 AirHitam / degree2 other-estate branch. This inherited name branch is a limitation; generic additive splines/RBF replacements failed badly in Palong. The new residual components below use the same recipe in every estate.
2. Learn a source-estate local-vigor correction using the other two estates' labels. Each source receives5-fold80m block-crossfitted baseline-logit offsets. Standardize NDRE1_contrast35 within each source, clip±3, fit balanced logistic offset with ridge coefficient .1 (objective .05*a²). Standardize target contrasts from target training rows. Add min(coefficient*contrast,0), a one-sided vigorous-canopy penalty. No target-held-out labels enter source fitting.
3. Derive three cloud-controlled radiometric contrasts from DN/10000: normalized excess green=(2G−R−B)/(2G+R+B+1e−6); soil-adjusted ND=(NIR−R)/(NIR+R+.16); visible-darkness proxy=sqrt(max((1−B)(1−G)(1−R),0)). Earlier notes called the first feature VARI; that name was inaccurate. The code/formula above is the actual normalized excess-green feature. These are empirical condition/illumination covariates, not validated illumination invariants.
4. Load AlphaEarth2025 A00..A63 minus circular40m local mean. Standardize on target binary training rows; fit8-component PCA on training rows only. Concatenate components with the3 radiometric features, standardize, clip±3, and fit a balanced ridge-logistic offset on block-crossfitted optical-model logits, ridge .5. Add .3 times the learned correction.
5. Four optical texture measurements describe NDVI ring10–30m minus centre, NDVI SD20m, central4×4 finite-difference edge magnitude, and NDRE median10m−median40m. Compute each at four fixed(±5m,±5m) index offsets about correctly registered trunk pixels; average feature values uniformly before supervised fitting. Standardize from target binary training, clip±3. Fit ridge .5 residual offset using5-fold block-crossfitted predictions of step4, then add .3 times the correction. All supervised fits are block-disjoint; no same-block pseudo-label features or random CV.
6. Return aligned finite float64 scores for every input row. A threadpool limit honors ctx['cpus']; no raster, original-table, prior, network, or write access occurs in inference. Runtime tables are gee_quality, gee_canopy, alphaearth, optical_jitter. Exact coordinates remain join keys.

Inner offsets respect block exclusion but use a fold-local coordinate origin, so boundaries can shift relative to the outer official grid. They remain group-disjoint within each inner fit. Covariate scaling/PCA uses only each supervised training sample. Source normalization can include unscored Middle covariates but never their labels as binary outcomes. Hyperparameters8 PCs/.3 shrinkage/.5 ridge were fixed for these new residual trials; no V-estate gates or V-tuned thresholds were introduced.

## Experiments and paired comparators

Table entries are step-function AP minus scored prevalence. Composite is the unweighted mean of the3 estates. Deltas are paired on the same official folds, with tie-aware sklearn AP. Full long table also includes ROC, recall25, enrichment10, repeated assignment results, and elapsed fitting time in EXPERIMENTS.csv. The repeated comparator means use only matching assignments for each model.

| model | AirHitam | Palong | Serting | composite | delta_incumbent | delta_coords | assignments | repeat_composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | +0.03097 | +0.25564 | +0.13496 | +0.14052 | +0.00000 | +0.05093 | 11 | +0.13216 |
| coords | +0.01079 | +0.17976 | +0.07822 | +0.08959 | -0.05093 | +0.00000 | 11 | +0.07098 |
| optical | +0.04106 | +0.27108 | +0.16891 | +0.16035 | +0.01983 | +0.07076 | 11 | +0.14892 |
| spline_additive | +0.02199 | +0.16525 | +0.17549 | +0.12091 | -0.01961 | +0.03132 | 4 | +0.10222 |
| spline_rbf | +0.02069 | +0.16777 | +0.12216 | +0.10354 | -0.03698 | +0.01394 | 4 | +0.10152 |
| radiometry_chlorophyll | +0.04714 | +0.26584 | +0.17258 | +0.16185 | +0.02133 | +0.07226 | 4 | +0.16205 |
| radiometry_water | +0.04215 | +0.26372 | +0.17354 | +0.15980 | +0.01928 | +0.07021 | 4 | +0.16004 |
| radiometry_shadow | +0.05059 | +0.27522 | +0.17638 | +0.16740 | +0.02688 | +0.07780 | 11 | +0.15428 |
| middle_gate | +0.04106 | +0.26329 | +0.17411 | +0.15949 | +0.01897 | +0.06989 | 4 | +0.15871 |
| alpha_state | +0.04144 | +0.27702 | +0.17762 | +0.16536 | +0.02484 | +0.07577 | 4 | +0.16367 |
| alpha_contrast | +0.04054 | +0.27727 | +0.18360 | +0.16714 | +0.02662 | +0.07754 | 11 | +0.15773 |
| alpha_change | +0.04005 | +0.27807 | +0.17418 | +0.16410 | +0.02358 | +0.07451 | 4 | +0.16235 |
| joint_foundation | +0.04925 | +0.28347 | +0.18840 | +0.17371 | +0.03319 | +0.08412 | 11 | +0.16247 |
| source_radiometry | +0.05596 | +0.29098 | +0.19331 | +0.18008 | +0.03956 | +0.09049 | 11 | +0.16747 |
| texture_all | +0.05019 | +0.29517 | +0.20808 | +0.18448 | +0.04396 | +0.09489 | 4 | +0.18238 |
| texture_small | +0.05285 | +0.29613 | +0.19791 | +0.18230 | +0.04178 | +0.09270 | 11 | +0.17158 |
| texture_aligned | +0.05100 | +0.28789 | +0.19828 | +0.17906 | +0.03854 | +0.08947 | 11 | +0.16703 |
| texture_jitter | +0.05056 | +0.28872 | +0.19847 | +0.17925 | +0.03873 | +0.08966 | 11 | +0.16731 |
| registration_ensemble | +0.05143 | +0.29146 | +0.19796 | +0.18028 | +0.03976 | +0.09069 | 11 | +0.16941 |
| jitter_no_quality | +0.04839 | +0.27739 | +0.17835 | +0.16804 | +0.02752 | +0.07845 | 4 | +0.17130 |
| jitter_no_vigor | +0.04296 | +0.28733 | +0.19193 | +0.17407 | +0.03355 | +0.08448 | 4 | +0.17511 |
| jitter_no_spatial | +0.04079 | +0.05989 | +0.22100 | +0.10722 | -0.03329 | +0.01763 | 4 | +0.09176 |

Mechanisms tested span measurement-quality/radiometry, independent prior-year foundation context, spatial-function shape, Middle negative-control gating, local canopy texture, source-only transfer, and registration uncertainty. All gains are conditional on this research campaign, not a preregistered trial.

The final registration-integrated model sacrifices about.00305 official composite versus the accidental NW view. Its geometry is explicit and its V performance is essentially indistinguishable at the scorer noise scale. Equal blending of the two views gave little additional local benefit and was not sent to V. Source radiometric transfer was weaker than canopy texture; no new transfer term is used. SAR patch SSL was already rejected in s008, and no old SAR sweep was rerun without a new mechanism.

## Held-out noisy V checkpoints

| model | noisy_pr_auc_lift | noisy_delta_vs_incumbent | noisy_delta_vs_coords | AirHitam | Palong | Serting |
| --- | --- | --- | --- | --- | --- | --- |
| v_optical | +0.14275 | +0.02400 | +0.07629 | +0.00540 | +0.00172 | +0.06489 |
| v_shadow | +0.15270 | +0.03395 | +0.08624 | +0.01901 | +0.00803 | +0.07480 |
| v_joint | +0.15981 | +0.04106 | +0.09335 | +0.02228 | +0.01445 | +0.08644 |
| v_texture | +0.17716 | +0.05841 | +0.11070 | +0.02926 | +0.03234 | +0.11362 |
| v_jitter | +0.17767 | +0.05891 | +0.11120 | +0.02936 | +0.01939 | +0.12799 |

Per-estate columns are deltas versus incumbent; composite columns are the verifier's returned statistics. Gaussian noise σ=.005 is cached per content hash. Five genuine milestones were scored; no cosmetic hashes were resubmitted to redraw noise. Final selected V: .177665 lift, delta incumbent+.058915, delta coordinates+.111203. Estate deltas+.029357/+.019394/+.127993. Similar V scores for the two registration recipes do not establish equivalence. Selection used L evidence, corrected geometry, and V non-degradation; all statements remain subject to adaptive validation.

Final packaging removes unused research functions and limits threads; verified_prediction_equivalence.json confirms unchanged OOF predictions against the scored jitter recipe (maximum discrepancy below1e−7). No new V query was spent on packaging changes.

## Spatial robustness and uncertainty

| estate | mean_delta_incumbent | worst_delta_incumbent | wins | mean_delta_coords |
| --- | --- | --- | --- | --- |
| AirHitam | +0.02097 | +0.01541 | 11 | +0.01538 |
| Palong | +0.02365 | +0.00698 | 11 | +0.13178 |
| Serting | +0.06082 | +0.03373 | 11 | +0.14184 |

| protocol | estate | lift | delta_incumbent | delta_coords |
| --- | --- | --- | --- | --- |
| 120m blocks | AirHitam | +0.04612 | +0.02068 | -0.02484 |
| 120m blocks | Palong | +0.28960 | +0.03530 | +0.07368 |
| 120m blocks | Serting | +0.17418 | +0.04874 | +0.17059 |
| 4 contiguous quadrants | AirHitam | +0.04939 | +0.01824 | +0.05726 |
| 4 contiguous quadrants | Palong | +0.26451 | +0.01713 | +0.12495 |
| 4 contiguous quadrants | Serting | +0.19511 | +0.08386 | +0.16042 |
| 80m + 20m buffer | AirHitam | +0.05199 | +0.02475 | +0.02558 |
| 80m + 20m buffer | Palong | +0.26599 | +0.05218 | +0.22649 |
| 80m + 20m buffer | Serting | +0.18734 | +0.05666 | +0.18586 |
| Shifted40m grid | AirHitam | +0.04577 | +0.02155 | +0.01182 |
| Shifted40m grid | Palong | +0.26776 | +0.03650 | +0.18691 |
| Shifted40m grid | Serting | +0.20687 | +0.04099 | +0.09562 |

The final model does not dominate the coordinates comparator on every stress test: AirHitam coordinates-only performs better in2/3 wider120m assignments, despite positive composite differences. The model therefore remains a context-assisted triage model, with no proof of a spatially invariant disease signature.

The final model improves every estate in all11 assignments(seed0 official plus seeds1–10 GroupKFold shuffle) versus incumbent. Randomness only reallocates complete80m blocks. Contiguous quadrants,120m groups, and20m nearest-training exclusion challenge extrapolation. The earlier joint checkpoint also improves across9 shifted/buffered/wider-block comparisons; full results are robust_validation.csv. Final shifted-grid results, if present, are retained separately.

Paired bootstrap resamples original80m blocks with common multiplicities across all repeated assignments. Composite mean delta incumbent=+0.03514, descriptive95% interval[+0.01416,+0.07225]. Against coordinates: delta=+0.09633, interval[+0.02837,+0.15608]. Palong's individual interval against incumbent crosses zero. Repeated folds share training data; the bootstrap is exploration-conditional, does not account for all model selection, and does not establish prospective significance. Blocks themselves can still be spatially correlated.

After freezing, additional block seeds100–109 give mean composite delta +0.03307. Composite gains occur in 10/10 assignments. Estate means: {'AirHitam': 0.02151, 'Palong': 0.02347, 'Serting': 0.05422}. These still reuse the same labels; they are not independent prospective validation.

## Inspection budget and error analysis

![Official80m spatial-CV inspection curves](inspection_curves.png)

| estate | prevalence | ap | lift | roc | recall25 | enrich10 |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | +0.05809 | +0.10865 | +0.05056 | +0.68717 | +0.51786 | +2.67857 |
| Palong | +0.08571 | +0.37443 | +0.28872 | +0.84577 | +0.81818 | +4.24242 |
| Serting | +0.10090 | +0.29938 | +0.19847 | +0.78054 | +0.68657 | +3.79104 |

At a25% budget, report binary-only and visible-Middle-inclusive capacity separately:

| estate | include_middle | recall | precision | enrichment | non_unhealthy_per_found |
| --- | --- | --- | --- | --- | --- |
| AirHitam | False | +0.51786 | +0.12033 | +2.07143 | +7.31034 |
| AirHitam | True | +0.51786 | +0.12033 | +2.07143 | +7.31034 |
| Palong | False | +0.81818 | +0.28052 | +3.27273 | +2.56481 |
| Palong | True | +0.74242 | +0.20248 | +2.96970 | +3.93878 |
| Serting | False | +0.68657 | +0.27711 | +2.74627 | +2.60870 |
| Serting | True | +0.67164 | +0.22086 | +2.68657 | +3.52778 |

The inclusive calculation counts Middle trees against inspection capacity but only Unhealthy trees as finds. Unknown U labels prevent estimating actual field recall or false-positive overhead. The4473-row output ranking is not evaluated on U. No threshold or majority-recall guarantee is made, especially for AirHitam: corrected-registration official recall25 is near50%, and region holdout is lower. inspection_budget_diagnostics.csv provides budgets5–50%; error_by_distance.csv reports misses by nearest labelled training-tree distance using outer OOF predictions. Sparse far-distance bins should not be interpreted as stable distance effects.

![Descriptive feature sign consistency](feature_signs.png)

Feature effects are descriptive using all available L binary labels, not additional feature-selection evidence. NDVI/qNDVI, red-edge local contrast and canopy-dip directions agree across estates. Normalized excess green and NDVI texture SD/edge change sign across estates: this is not label-quality evidence. The model learns their residual coefficients separately within estates. AlphaEarth PC signs have no inherent physiology and cannot be compared across independent fold PCA bases. Prior-year foundation context can encode stand/environment/location as well as canopy condition. Removing explicit coordinates lowers composite substantially, particularly in Palong, despite residual optical/embedding context remaining. Therefore this is not a pure transferable spectral disease detector. feature_effects.csv and feature_sign_consistency.csv quantify these effects.

Ablations removing the quality-NDVI blend or one-sided vigor correction each reduce composite. These ablations refit downstream residual stages, and 'no quality' removes only the NDVI measurement blend, not the independent radiometric covariates from the quality table. Differences isolate a recipe component rather than prove a biological cause.

## Extraction, registration, and reproducibility

New AlphaEarth embeddings cover2023/24/25, including local contrasts, with7220 exact keys and zero missing values. GEE actually returned2025 in its queried collection; the public catalogue text still described2017–2024. Only2025 local contrasts enter the final model; other years are retained for rejected state/change experiments. This is prior-year context, not a current disease observation.

Cloud-controlled optical source mosaics use April15–August15 2026, SCL4 and CloudScore+ cs_cdf≥.65, fallback March15–September15 SCL4; reflectance is already harmonized. Source10m grid uses EPSG32648. B5/B8A remain native20m information. sampleRectangle was used after thumbnails.create download permission failed. Audit confirmed pixelCoordinates values at centers(.5,1.5,...). The first derived view accidentally treated these as corners, displacing affine+5m east/−5m south. That table is preserved transparently as a reproducible NW view, not claimed as exact canopy centering. The final optical_jitter script explicitly corrects this convention and averages the fixed4 offsets. No feature table was silently replaced after verification. Native pixels are comparable to9m palm spacing, so crown-level attribution remains uncertain.

Remote frozen DINOv2 extraction was attempted through the GPU broker, but VM startup/upload/status failures prevented verified features. The attempted script and label-free chips are retained; no successful GPU result is claimed, and no torch dependency enters final inference. Initial stop calls timed out. A final broker stop response confirmed vm_name=null (no active VM), with1541.845 GPU seconds charged; no completed GPU features are used. The learned AlphaEarth features were obtained through GEE and require no GPU during model execution.

Some unlabelled covariates fall outside L support; covariate_support.csv records counts. For example, five Palong NDVI values exceed the labelled range and an extreme low-NDVI tree receives a very negative score from the inherited quadratic model. Labels there are unknown; this is an extrapolation limitation, not a demonstrated false negative. Inspect unsupported covariates when deploying.

Every derived table has source/manifests/extraction code,7220 exact coordinate keys, no forbidden identifiers or label columns, and no symlinks. dataset_audit.json gives hashes, sizes and imports. All dataset assets remain below128MiB. Only4 runtime tables are required. Dataset extraction is reproducible online; submitted fitting/prediction is offline. Source mosaics/chips are research artifacts outside runtime datasets.

## Execution verification and limitations

See final_selfcheck.log, final_cv.log, contract_verification.json. Fresh isolated subprocess tests reject source-table/raster/prior reads, network connections, and writes outside scratch; test row shuffles, duplicate rows, exact dtype/length/finiteness and input immutability. The selected recipe fits within CPU/RAM/time caps. All non-base imports are pinned: threadpoolctl==3.8.0; others are standard library or runtime-base dependencies. No symlinks. Scores are logits plus shrunken corrections, not calibrated probabilities. Strong within-estate gains do not establish new-estate or new-season validity. The inherited estate-name polynomial branch should be replaced only with validation evidence. Independent inspection remains necessary, and the downstream polarized-imagery system is out of scope.

## Primary literature and documentation

- Izzuddin et al.(2018), [airborne hyperspectral Ganoderma analysis](https://doi.org/10.21894/jopr.2018.0037): common vegetation/red-edge indices had limited early-stage separation; this supports cautious condition-screening interpretation, not disease specificity.
- Brown et al.(2025), [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291), and [official Earth Engine embedding catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL):64-dimensional annual10m learned geospatial representations. Temporal coverage was checked directly in the API for this extraction.
- [Sentinel-2 harmonized SR catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) and [Cloud Score+ catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED): provenance/masking; no second+1000 offset applied.
- [Earth Engine sampleRectangle](https://developers.google.com/earth-engine/apidocs/ee-image-samplerectangle) and [pixelCoordinates](https://developers.google.com/earth-engine/apidocs/ee-image-pixelcoordinates): array-extraction and grid audit.
- Oquab et al.(2023), [DINOv2](https://arxiv.org/abs/2304.07193): frozen visual-feature hypothesis attempted; no completed experiment or scientific verdict.
- [scikit-learn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): tie-aware step-function AP, not trapezoidal PR area.
