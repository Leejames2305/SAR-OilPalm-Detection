# Oil-palm triage: cross-lineage synthesis, 261127-s014

## Result and scope

The final submission is **source_prior**: a single offline CPU pipeline using the compact s012 design and s013 October SAR context, with **independent source fits supplying a frozen optical coefficient prior** for the target estate. It keeps all spatial, historical-height, optical-mixture and latent responses local. This follows s009's separation of source information from target refinement, without importing its nested residual implementation or estate-name branch. All three estates' labels are verified and trusted equally.

Correct official80m composite AP lift is **+0.287409**, versus incumbent **+0.122984**, coordinates **+0.043863**, s009 **+0.160430**, s012 **+0.272519** and s013 **+0.280623**. The selected noisy V lift is **+0.226709**, delta incumbent **+0.107959** and coordinates **+0.160247**; V estates are **+0.059286/+0.344207/+0.276634**. These are inspection-ranking results, not calibrated disease probabilities or a pathogen diagnosis.

The model passes the selection rule written before V: composite increment at least.005 over the supplied s013 reward, at least2 nonnegative estate increments and no estate loss greater than.015. Its increments are **+.004022/+.005070/+.019014**, composite **+.009369**. No rule was amended after V. This is still adaptive, noisy validation: the incremental block-bootstrap interval versus s013 crosses zero and the small AirHitam/Palong V increments are near scorer noise. It does **not** dominate every parent in V: Palong remains below supplied s012 by.018970, and AirHitam below s009 by.006227.

The recipe was frozen before20 reserved assignments701–720. Composite beats incumbent, coords, s009, s012 and s013 in20/20 assignments. Estate wins versus incumbent are59/60, with Palong's one loss−.009656; means improve in all estates. The binary top25% recall is48.2/65.2/74.6%; including Middle against capacity gives48.2/63.6/73.1%. **AirHitam majority capture at25% is not established.**

## Data and a consequential validation correction

Authorized L has2747 palms:2398 Healthy/Unhealthy scoring rows,189 positives and349 Middle. Binary counts/positives are964/56,770/66,664/67. U has4473 rows; all7220 exact original coordinate keys occur in every carried feature table. Middle never trains the binary target and never enters binary AP. It remains in output rankings and capacity accounting. No sealed labels/splits, banned repository or credential contents were accessed. No new GEE, raster extraction or GPU session was needed.

The prior research evaluator used estate origins and mean latitude from **L alone**. The current CLI and task require **L union U**. These origins differ enough to split original deployment blocks in historical AirHitam/Palong folds. We detected this in source, corrected the evaluator before experiments and ran all four staged submissions with the actual CLI. A separate historical-origin reproduction recovers the published numbers exactly; those values are valid for that geometry but cannot substitute for the required current score.

| estate | legacy_Lonly_blocks | correct_union_blocks | key_changes | original_blocks_split_across_legacy_folds | legacy_validation_rows_sharing_original_training_block | legacy_validation_share |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | 50 | 26 | 453 | 19 | 799 | +0.828838 |
| Palong | 39 | 18 | 353 | 18 | 968 | +1.000000 |
| Serting | 13 | 13 | 0 | 0 | 0 | +0.000000 |

Thus82.9% of legacy AirHitam validation rows and100% of legacy Palong validation rows share an original80m block with training. Serting's origin is unchanged. This does not mean every legacy score must be larger: AirHitam's required split is numerically easier in this assignment. It means the validation question differs. Do not describe historical L-only folds as compliant current deployment-block validation.

Historical-origin diagnostic only (never used as the primary metric):

| model | AirHitam | Palong | Serting | composite |
| --- | --- | --- | --- | --- |
| coords | +0.010793 | +0.179763 | +0.078224 | +0.089593 |
| incumbent | +0.030965 | +0.255637 | +0.134956 | +0.140519 |
| s009 | +0.050556 | +0.288720 | +0.198473 | +0.179250 |
| s012 | +0.066466 | +0.382857 | +0.359326 | +0.269549 |
| s013 | +0.075808 | +0.409951 | +0.378503 | +0.288087 |

All primary folds now use east=(Long−union_min_Long)*111320*cos(union_mean_Lat), north=(Lat−union_min_Lat)*110540 and floor(east/80)*100000+floor(north/80). There are26/18/13 occupied L blocks. GroupKFold5 assigns the entire estate table including Middle; binary filtering happens after predictions. Correct folds match the CLI exactly. Repeats reallocate complete original blocks, not rows. Our120m stress groups assign original80m block centres to120m cells, preserving original blocks; their support can be80–160m.160m groups are exact2x2 aggregations. Whole-block quadrants use population-weighted medians of original block indices. Buffers remove training palms within40/80/160m of any held-out palm. Every spatial audit row reports zero shared original blocks.

The required coordinates comparator is15-NN with distance weights on training-standardized Long/Lat, fitted to binary labels. AP uses sklearn's step-function, tie-aware definition; lift subtracts scored prevalence and estates are averaged equally. Inspection ties receive fractional boundary allocation. Source labels enter only through authorized ctx.other_labeled; estates are spatially separate. All supervised transforms use the relevant training sample. Final fitting uses no nested target pseudo-labels or OOF predictors, so it avoids legacy s009's fold-local inner-origin problem.

## Unified empirical lineage comparison

L entries are newly rerun under the corrected common protocol. Parent V entries use the supplied verifier-verified rewards, not more favorable observations from their reports; unchanged parents were not re-queried. The selected model and signs use this session's two observations. Incumbent/coords V estate values are the scorer-returned references.

| model | L AirHitam | V AirHitam | L Palong | V Palong | L Serting | V Serting | L composite | V composite | L delta incumbent | L delta coords |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | +0.014687 | +0.037028 | +0.219310 | +0.253306 | +0.134956 | +0.065918 | +0.122984 | +0.118751 | +0.000000 | +0.079121 |
| coords | +0.014041 | +0.039792 | +0.039323 | +0.127807 | +0.078224 | +0.031787 | +0.043863 | +0.066462 | -0.079121 | +0.000000 |
| s009 | +0.030146 | +0.065513 | +0.252671 | +0.276497 | +0.198473 | +0.188360 | +0.160430 | +0.176790 | +0.037446 | +0.116568 |
| s012 | +0.151551 | +0.049552 | +0.306681 | +0.363177 | +0.359326 | +0.197506 | +0.272519 | +0.203412 | +0.149535 | +0.228657 |
| s013 | +0.151070 | +0.055264 | +0.312298 | +0.339137 | +0.378503 | +0.257620 | +0.280623 | +0.217340 | +0.157639 | +0.236761 |
| signs | +0.159112 | +0.054965 | +0.310892 | +0.340953 | +0.377704 | +0.259519 | +0.282569 | +0.218479 | +0.159585 | +0.238707 |
| source_prior | +0.159644 | +0.059286 | +0.311518 | +0.344207 | +0.391065 | +0.276634 | +0.287409 | +0.226709 | +0.164425 | +0.243547 |

![Corrected CV versus verified V](lineage_comparison.png)

s012's Palong V advantage and s013's Serting advantage persist despite similar L rankings. Their current official OOF correlations are high and only about5% of binary palms change top25% membership, yet AP remains sensitive to which few positives move at the ranking head. s009 differs more substantially. Pairwise correlation and fractional top25% disagreement are saved for every parent pair and final comparator:

| estate | first | second | spearman | top25_disagreement | top25_jaccard |
| --- | --- | --- | --- | --- | --- |
| AirHitam | s009 | s013 | +0.56195 | +0.20539 | +0.41765 |
| AirHitam | s012 | s013 | +0.95007 | +0.05187 | +0.81203 |
| Palong | s009 | s013 | +0.70611 | +0.18182 | +0.46667 |
| Palong | s012 | s013 | +0.98570 | +0.04935 | +0.82033 |
| Serting | s009 | s013 | +0.68610 | +0.15060 | +0.53704 |
| Serting | s012 | s013 | +0.97410 | +0.05120 | +0.81421 |
| AirHitam | source_prior | s012 | +0.94377 | +0.06224 | +0.77860 |
| AirHitam | source_prior | s013 | +0.99652 | +0.01452 | +0.94355 |
| Palong | source_prior | s012 | +0.98306 | +0.05455 | +0.80328 |
| Palong | source_prior | s013 | +0.99892 | +0.00519 | +0.97943 |
| Serting | source_prior | s012 | +0.97059 | +0.05422 | +0.80435 |
| Serting | source_prior | s013 | +0.99793 | +0.01506 | +0.94152 |

This supports testing mechanisms, not fitting V-based ensemble weights. The origin discrepancy explains the historical/current L mismatch. It does **not** explain all L–V estate ordering: after correction, local repeats still fail to predict which parent is best in each V estate. Small positive counts, clustered pockets, conditional relationships and spatial extrapolation remain plausible contributors; their causal contributions are not identifiable here. No estate's label quality is questioned.

## Controlled crossover campaign

The campaign has35 recipes/comparators and1334 completed estate/protocol records. Initial exploration uses official and repeat11–15; most exploratory candidates use the common11–13 subset plus160m,40m-buffer and whole-block region protocols. All four staged lineages also have120m/80m buffers and20 frozen assignments. Missing/failed estate runs are NA, never zero or partial-estate composites. Full long evidence is EXPERIMENTS.csv and OOF NPZ files. No broad SAR-mean/model grid, SMOTE, random-row CV or Middle weak-positive route was reopened.

| model | AirHitam | Palong | Serting | composite | delta_incumbent | delta_coords | delta_s012 | delta_s013 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner_context | +0.160767 | +0.308636 | +0.395182 | +0.288195 | +0.165211 | +0.244333 | +0.015676 | +0.007572 |
| source_prior | +0.159644 | +0.311518 | +0.391065 | +0.287409 | +0.164425 | +0.243547 | +0.014890 | +0.006786 |
| crossfit_prior | +0.159048 | +0.312832 | +0.389042 | +0.286974 | +0.163990 | +0.243111 | +0.014455 | +0.006351 |
| heterogeneous_prior | +0.158920 | +0.306571 | +0.390882 | +0.285458 | +0.162474 | +0.241595 | +0.012939 | +0.004834 |
| prior_no_sar | +0.167114 | +0.304006 | +0.380700 | +0.283940 | +0.160956 | +0.240077 | +0.011421 | +0.003317 |
| stable_prior | +0.158713 | +0.313386 | +0.377881 | +0.283327 | +0.160343 | +0.239464 | +0.010808 | +0.002703 |
| signs | +0.159112 | +0.310892 | +0.377704 | +0.282569 | +0.159585 | +0.238707 | +0.010050 | +0.001946 |
| spectral_protect | +0.152234 | +0.308826 | +0.381969 | +0.281010 | +0.158026 | +0.237147 | +0.008490 | +0.000386 |
| control | +0.151070 | +0.312298 | +0.378503 | +0.280623 | +0.157639 | +0.236761 | +0.008104 | +0.000000 |
| s013 | +0.151070 | +0.312298 | +0.378503 | +0.280623 | +0.157639 | +0.236761 | +0.008104 | +0.000000 |
| orthogonal | +0.152310 | +0.308279 | +0.377649 | +0.279413 | +0.156429 | +0.235550 | +0.006894 | -0.001211 |
| grouped_signs | +0.152540 | +0.323297 | +0.359222 | +0.278353 | +0.155369 | +0.234490 | +0.005834 | -0.002270 |
| domain_optical | +0.139737 | +0.325675 | +0.369093 | +0.278168 | +0.155184 | +0.234306 | +0.005649 | -0.002455 |
| no_sar | +0.161193 | +0.300185 | +0.370768 | +0.277382 | +0.154398 | +0.233519 | +0.004863 | -0.003241 |
| prior_zero | +0.145824 | +0.312446 | +0.370647 | +0.276306 | +0.153322 | +0.232443 | +0.003787 | -0.004318 |
| grouped | +0.145857 | +0.322567 | +0.360088 | +0.276171 | +0.153187 | +0.232308 | +0.003651 | -0.004453 |
| domain_context | +0.124952 | +0.310022 | +0.384097 | +0.273024 | +0.150040 | +0.229161 | +0.000505 | -0.007600 |
| prior_no_structure | +0.135893 | +0.298390 | +0.383966 | +0.272750 | +0.149766 | +0.228887 | +0.000231 | -0.007874 |
| s012 | +0.151551 | +0.306681 | +0.359326 | +0.272519 | +0.149535 | +0.228657 | +0.000000 | -0.008104 |
| inner_C | +0.153676 | +0.306717 | +0.352655 | +0.271016 | +0.148032 | +0.227153 | -0.001503 | -0.009607 |
| prior_permuted | +0.142408 | +0.305240 | +0.360578 | +0.269408 | +0.146424 | +0.225546 | -0.003111 | -0.011215 |
| anchor_background | +0.158663 | +0.299014 | +0.346518 | +0.268065 | +0.145081 | +0.224202 | -0.004454 | -0.012558 |
| context_regularized | +0.152436 | +0.308636 | +0.339322 | +0.266798 | +0.143814 | +0.222936 | -0.005721 | -0.013825 |
| context_regularized_signs | +0.151127 | +0.309304 | +0.338893 | +0.266442 | +0.143458 | +0.222579 | -0.006078 | -0.014182 |
| block_balanced | +0.130903 | +0.288963 | +0.369069 | +0.262978 | +0.139994 | +0.219116 | -0.009541 | -0.017645 |
| prior_no_alpha | +0.148053 | +0.318873 | +0.315921 | +0.260949 | +0.137965 | +0.217086 | -0.011570 | -0.019675 |
| additive_signs | +0.164416 | +0.262130 | +0.354943 | +0.260496 | +0.137512 | +0.216634 | -0.012023 | -0.020127 |
| date_context | +0.110472 | +0.304730 | +0.362917 | +0.259373 | +0.136389 | +0.215510 | -0.013146 | -0.021250 |
| additive_shape | +0.158100 | +0.260905 | +0.357780 | +0.258928 | +0.135944 | +0.215066 | -0.013591 | -0.021695 |
| change_context | +0.098280 | +0.281232 | +0.377356 | +0.252289 | +0.129305 | +0.208427 | -0.020230 | -0.028334 |
| additive_grouped | +0.149039 | +0.273845 | +0.316852 | +0.246578 | +0.123594 | +0.202716 | -0.025941 | -0.034045 |
| prior_no_coords | +0.132838 | +0.200483 | +0.394334 | +0.242552 | +0.119568 | +0.198689 | -0.029967 | -0.038072 |
| s009 | +0.030146 | +0.252671 | +0.198473 | +0.160430 | +0.037446 | +0.116568 | -0.112089 | -0.120193 |
| incumbent | +0.014687 | +0.219310 | +0.134956 | +0.122984 | +0.000000 | +0.079121 | -0.149535 | -0.157639 |
| coords | +0.014041 | +0.039323 | +0.078224 | +0.043863 | -0.079121 | +0.000000 | -0.228657 | -0.236761 |

Mechanisms and decisions:

1. **Correlated optical measurements and sign discipline.** Grouped optical axes average prespecified standardized vigor, moisture, contrast and roughness families, reducing redundant SD/edge and mixture dimensions. Grouping helps some Palong fits but loses Serting. Coefficient signs constrain core NDVI/optical deficit coefficients and positive gap response, while leaving brittle height/context unrestricted. Signs is competitive locally and V1+.218479, but does not earn the required V increment. These constraints are coefficient restrictions, not a global monotonic guarantee with NDVI interactions.
2. **Protect the optical backbone.** Training-only ridge projections remove predictable optical or optical+coordinate components from secondary context. orthogonal and spectral_protect isolate changed penalty geometry without changing covariates. Gains are small/mixed; neither justifies more V queries. Stronger context shrinkage also loses Serting.
3. **Spatial stability.** Equalized block-size/class weights and a healthy-background between-region logit penalty are fixed controls for geographical nuisance. They reduce performance. Removing NDVI quadratic/location interactions improves AirHitam but loses Palong/Serting substantially; therefore no unvalidated tail repair is shipped.
4. **Temporal representation.** Existing June+October PCs or encoded change replace/add to October alone. They lose composite and repeated robustness. This tests a representation crossover, not a new extraction, and does not prove that physiological L-band change is absent.
5. **Covariate shift.** Label-free target-versus-training domain classifiers provide clipped[.5,2] training ratios on optical or full context, with class mass restored. Optical adaptation helps Palong but harms AirHitam and some buffered Serting fits; full-context adaptation similarly trades estates. Neither earns V. Conditional shift cannot generally be repaired by marginal density weighting.
6. **Generalizable model selection.** Inner4-fold, original80m-block CV selects C in[.05,.15,.45] or context scale in[.5,1,1.5], preferring the simpler option within.005 AP. Both avoid estate-name branches and use training labels only. Their outer gains are inconsistent; the context selector loses distant Serting sharply. Fixed uniform capacity is retained based on paired evidence rather than hardcoded estate names.
7. **Modular source coefficient transfer, selected.** Fit sources independently with the same49-variable local design and intercept. Freeze their equally weighted12-optical-coefficient mean as the target prior centre. This separates source evidence from target nuisance adaptation, unlike s013's simultaneous shared/local fit. source_prior improves all three means in reserved assignments and every completed broad stress cell versus s013. The official Palong increment is−.000778, while repeated mean and V increments are positive. This is not score averaging.
8. **Extra source-prior complexity.** Per-feature source-disagreement attenuation, leave-block-fold coefficient means and jackknife stability attenuation do not improve the simpler prior reliably. They use block-disjoint training and align optical coefficients to full-source feature scales. No V query was spent on them. Zero-centre and fixed permuted-source controls reduce performance, supporting a contribution from real source outcomes rather than merely a new solver or ridge strength. One fixed permutation is a negative control, not a permutation-test p-value.

Adopted attribution: s012 supplies uniform shrunk quadratic geometry,C=.15, compact optical/AlphaEarth, physical-floor historical height and mixture covariates. s013 supplies October frozen SAR context and the physical-response-sharing hypothesis. s009 supplies the architectural lesson of keeping source information separate from target refinement. New source_prior implements that separation as a frozen Gaussian coefficient prior. Retired: incumbent/s009 estate-name capacity branch, s009 nested OOF offsets, s013 simultaneous shared/local duplication, score ensembles, V-tuned gates, new extraction, grouping, sign restrictions and unsupported inner selectors.

## Exact submitted recipe

For each estate independently, retain Healthy/Unhealthy. Standardize raw NDVI/east/north on its current binary training rows, form9 degree1/2 terms and multiply all degree2 terms by.3. Append12 standardized optical columns*.5,8 training PCs of standardized64-dimensional AlphaEarth2025 local contrasts*.5,3 historical height columns*.35 scaled by sqrt(training_variance+[3,1,1]^2),9 standardized deblur optical columns*.3, and8 training PCs of saved October103m SAR covariates*.3 (without standardizing the saved input PCs). Both PCA fits use deterministic full SVD and standardize their resulting8 scores. Add one intercept indicator of10: **50 classifier columns** in each fit, versus s013's162 simultaneous columns.

For each source, fit balanced LogisticRegression(C=.15,fit_intercept=False,tol=1e-7,max_iter=2000). Class weights give half the estate's training mass to each class. Average the source optical coefficients equally. They are coefficients in the standardized*.5 optical basis; this transfers standardized response, not identical raw-unit effects. Spatial/latent bases are never shared across estates.

Let mu contain this12-dimensional source mean and zeros elsewhere. The target minimizes summed balanced logistic loss plus (1/(2C))*[sum_nonopt beta_j^2 + sum_opt(beta_j−mu_j)^2],C=.15. This replaces only the centre of the optical ridge penalty; it does not increase target model width or average predictions. L-BFGS-B starts at zero, maxiter2000,ftol1e-12,gtol1e-6. Return the target design's linear predictor, finite float64 in exact requested order. Empty and single-class cases have deterministic finite fallbacks.

The carried optical measurements use harmonized Sentinel-2 SR. Quality medians use April15–August15 2026,SCL4 and CloudScore+cs_cdf≥.65, with the documented March–September fallback; no extra+1000 DN offset is applied. qNDVI/qNDRE/qNDMI/qNIRv here are medians, not the older s009p75 blend. Local optical contrasts and the four fixed registration-jitter summaries come from the carried canopy/texture tables. Crown-mixture spectra use the separately documented SCL4–7 composite and fixed5m Gaussian inverse withlambda.25;20m red-edge/SWIR information is not physically sharpened. Historical Meta/WRI/Maxar height uses2009–2020 imagery, predominantly2018–2020 per its manifest, so it is stand context rather than current disease height. October SAR features originate from2026-10-14 ALOS HH/HV/VV(bands1/2/4), reprojected to native CRS, finite-positive linear-power support then10log10 display preprocessing and a frozen DINOv2 encoder. Existing extraction scripts/manifests provide the full formulas, georeferencing, masks and immutable hashes. These are carried covariates; this session did not re-extract them.

Runtime reads only triage_v12 and dino_sar through ctx.datasets_dir, honors ctx.cpus, writes nothing and uses no network/GPU/raster access. No estate-name branch, target label-neighbor prior, binary Middle target, target pseudo-label or fitted score artifact is used. Source transforms/coefficients depend solely on other estates' L. The target's held-out labels never contribute.

## Frozen confirmation and uncertainty

Selection was frozen in selection.json before seeds701–720. No model changes or further V queries followed. All six required comparisons were rerun on those same assignments.

| model | AirHitam | Palong | Serting | composite | delta_incumbent | delta_coords | delta_s012 | delta_s013 | composite_wins_s013 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coords | +0.014555 | +0.049718 | +0.062938 | +0.042404 | -0.083753 | +0.000000 | -0.187991 | -0.205387 | 0 |
| incumbent | +0.022462 | +0.228689 | +0.127318 | +0.126156 | +0.000000 | +0.083753 | -0.104239 | -0.121635 | 0 |
| s009 | +0.041618 | +0.260560 | +0.184055 | +0.162078 | +0.035922 | +0.119674 | -0.068317 | -0.085713 | 0 |
| s012 | +0.104333 | +0.293024 | +0.293828 | +0.230395 | +0.104239 | +0.187991 | +0.000000 | -0.017396 | 0 |
| s013 | +0.115513 | +0.297071 | +0.330789 | +0.247791 | +0.121635 | +0.205387 | +0.017396 | +0.000000 | 0 |
| source_prior | +0.124267 | +0.301499 | +0.336904 | +0.254223 | +0.128067 | +0.211819 | +0.023828 | +0.006432 | 20 |

| estate | mean_lift | mean_delta_incumbent | worst_delta_incumbent | wins_incumbent | mean_delta_coords | mean_delta_s012 | mean_delta_s013 | wins_s013 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | +0.124267 | +0.101805 | +0.064097 | 20 | +0.109712 | +0.019933 | +0.008753 | 20 |
| Palong | +0.301499 | +0.072809 | -0.009656 | 19 | +0.251780 | +0.008474 | +0.004428 | 17 |
| Serting | +0.336904 | +0.209586 | +0.151375 | 20 | +0.273965 | +0.043076 | +0.006115 | 20 |

The source-prior composite exceeds every comparator in20/20 assignments; it beats s013 in57/60 estate cells and incumbent in59/60. Repeated assignments reuse the same labels and imagery; they are stability evidence, not20 independent prospective trials.

The paired4000-resample bootstrap resamples original80m blocks with common multiplicities across models, independently within estates, and recomputes weighted tie-aware AP from fixed official OOF scores. Composite point deltas versus incumbent/coords are+.164425/+.243547. Conditional95% intervals are[+.081272,+.232117] and[+.161696,+.303367]. The incremental interval versus s013 is[−.001351,+.014052]. Bootstrap means differ from point estimates because AP is nonlinear. Palong's individual incumbent interval crosses zero. These intervals do not refit models, correct adaptive selection, remove adjacent-block dependence or establish disease causality/new-estate validity.

| reference | bootstrap_mean | lower | upper | fraction_positive |
| --- | --- | --- | --- | --- |
| incumbent | +0.158956 | +0.081272 | +0.232117 | +1.000000 |
| coords | +0.234353 | +0.161696 | +0.303367 | +1.000000 |
| s009 | +0.114192 | +0.038593 | +0.189383 | +0.998250 |
| s012 | +0.013729 | -0.005426 | +0.032612 | +0.924750 |
| s013 | +0.006325 | -0.001351 | +0.014052 | +0.943250 |

## Coarser, buffered and contiguous-region stress tests

| protocol | AirHitam | Palong | Serting | composite | delta_incumbent | delta_coords | delta_s009 | delta_s012 | delta_s013 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official | +0.159644 | +0.311518 | +0.391065 | +0.287409 | +0.164425 | +0.243547 | +0.126979 | +0.014890 | +0.006786 |
| block120 | +0.148884 | +0.287515 | +0.343768 | +0.260056 | +0.167731 | +0.237605 | +0.134945 | +0.033055 | +0.007518 |
| block160 | +0.120163 | +0.332607 | +0.363554 | +0.272108 | +0.135432 | +0.249541 | +0.097894 | +0.043480 | +0.015212 |
| buffer40 | +0.153309 | +0.287348 | +0.329094 | +0.256584 | +0.129942 | +0.238019 | +0.095824 | +0.033453 | +0.009656 |
| buffer80 | +0.106046 | +0.261029 | +0.179757 | +0.182277 | +0.114588 | +0.176504 | +0.096480 | +0.036113 | +0.014080 |
| blockquad | +0.111843 | +0.331240 | +0.357541 | +0.266875 | +0.142359 | +0.242271 | +0.103552 | +0.035522 | +0.014237 |
| blockquad80 | +0.106479 | +0.184292 | +0.245475 | +0.178749 | +0.106893 | +0.165645 | +0.085266 | +0.043527 | +0.006983 |
| blockquad160 | +0.052606 | -0.014195 | +0.050425 | +0.029612 | +0.028751 | +0.017743 | NA | +0.000741 | +0.004853 |

Source-prior improves s013 in all21 estate cells of the seven broader protocols. It also improves incumbent in every reported cell, including the previously problematic80m-buffered Palong quadrant. However **Palong160m-buffered region lift is−.014195**, below coordinates+.016275; it is effectively failed extrapolation even though incumbent is slightly worse. The smallest such Palong training sample has44 total rows, seven positives, and only three inner groups for legacy s009; that parent fails there. Its AirHitam/Serting attempts are retained and the composite is NA. No failure was replaced by a successful nearby protocol.

SAR crops span about103m, so ordinary80m holdouts can share raw imagery.160m region buffers exceed the SAR crop diameter and test stronger separation; Serting lift drops to+.05042 and Palong remains weak. Optical inversion has a small global influence tail, so even these checks do not prove fully independent image information. Geography, management and historical stand condition may contribute alongside disease condition.

![Paired spatial stress gains](spatial_robustness.png)

## Noisy V and selection record

Only two distinct, L-verified architectures were sent to V. No unchanged parent/hash cosmetic rerun, source-label recovery or V-fitted score/estate weights was used.13of15 queries remain. The service reports sigma=.005 noise cached by submission content hash.

| query | model | AirHitam | Palong | Serting | composite | delta_incumbent | delta_coords | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | signs | +0.054965 | +0.340953 | +0.259519 | +0.218479 | +0.099729 | +0.152017 | False |
| 2 | source_prior | +0.059286 | +0.344207 | +0.276634 | +0.226709 | +0.107959 | +0.160247 | True |

V1 signs does not reach the fixed+.005 increment guard over supplied s013. V2 passes every part of that guard. Its content hash is9c21e7eb0f9e38aa44cfdf7e74360b7e7580a12eec6934e532b4799aeaf6a994. Small V increments are not confirmed independent improvements; no more queries followed the selection. All scalar/estate V deltas versus incumbent and coordinates are positive. Packaging was verified against the immutable V2 package, not re-scored for new noise.

## Feature effects, suppression and inspection errors

Marginal effects are Unhealthy−Healthy pooled-SD Cohen's d, descriptive on authorized L:

| feature | AirHitam | Palong | Serting | same_sign |
| --- | --- | --- | --- | --- |
| NDVI | -0.19502 | -0.31057 | -0.53012 | True |
| deblur_NDMI | -0.05101 | -0.21908 | -0.65704 | True |
| height_contrast6 | +0.27687 | -0.46990 | -0.12794 | False |
| height_mean6 | +0.30667 | -0.55824 | -0.25639 | False |
| ndvi_dip | +0.08194 | +0.54894 | +0.69822 | True |
| ndvi_edge | -0.24991 | +0.05790 | +0.11729 | False |
| ndvi_sd20 | -0.25346 | +0.06498 | +0.12258 | False |
| qNDMI | -0.13657 | -0.21829 | -0.71814 | True |
| qNDRE | -0.43171 | -0.60424 | -0.80618 | True |
| qNDVI | -0.19503 | -0.35568 | -0.68702 | True |

Height signs flip across equally trusted estates and are not interpreted as a common disease marker. NDVI/quality/red-edge deficits agree marginally, but correlated conditional coefficients can differ. SD/edge correlation is about.947/.970/.983 across estates. s012's Palong qNDVI coefficient is+.064 encoded (+.032 standardized), while s013 and source-prior make it negative. That does not establish a corrected biological mechanism: both SD/edge remain negative conditional coefficients in Palong/Serting despite positive marginal d; deblur_NDMI is positive conditional despite negative marginal effects. qNIRv and AirHitam dip retain suppressor behavior. PCA axes are estate-specific and their signs have no common physical interpretation.

Final encoded coefficients below multiply their encoded feature scales (.5 optical,.3 mixture), and cannot be compared to marginal d as if univariate effects. Quadratic interactions further modify total NDVI response. Full source means and score-contribution SDs are saved in final_coefficients.csv/source_prior_coefficients.csv.

| feature | AirHitam | Palong | Serting |
| --- | --- | --- | --- |
| deblur_NDMI | +0.22335 | +0.15112 | +0.03871 |
| ndvi_dip | -0.17510 | +0.48629 | +0.58968 |
| ndvi_edge | -0.35586 | -0.62892 | -0.69138 |
| ndvi_sd20 | -0.65726 | -0.91065 | -1.10082 |
| qNDMI | -0.20170 | -0.19549 | -0.35009 |
| qNDRE | -0.44194 | -0.49863 | -0.68233 |
| qNDVI | -0.57178 | -0.23943 | -0.74256 |
| qNIRv | +0.07972 | -0.05181 | -0.29847 |

| estate | prevalence | ap | lift | roc | recall25 | enrich10 | inclusive_recall25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | +0.05809 | +0.21774 | +0.15964 | +0.70341 | +0.48214 | +3.03571 | +0.48214 |
| Palong | +0.08571 | +0.39723 | +0.31152 | +0.80893 | +0.65152 | +4.84848 | +0.63636 |
| Serting | +0.10090 | +0.49197 | +0.39107 | +0.86547 | +0.74627 | +5.37313 | +0.73134 |

At25% capacity, counting Middle separately:

| estate | include_middle | inspections | found | healthy | middle | recall | other_per_found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | False | +241.00000 | +27.00000 | +214.00000 | +0.00000 | +0.48214 | +7.92593 |
| AirHitam | True | +241.00000 | +27.00000 | +214.00000 | +0.00000 | +0.48214 | +7.92593 |
| Palong | False | +192.50000 | +43.00000 | +149.50000 | +0.00000 | +0.65152 | +3.47674 |
| Palong | True | +242.00000 | +42.00000 | +111.00000 | +89.00000 | +0.63636 | +4.76190 |
| Serting | False | +166.00000 | +50.00000 | +116.00000 | +0.00000 | +0.74627 | +2.32000 |
| Serting | True | +203.75000 | +49.00000 | +96.75000 | +58.00000 | +0.73134 | +3.15816 |

AirHitam finds27/56 positives and needs about7.93 other inspections per find; Palong all-tree capacity finds42/66 with89 Middle inspections and4.76 other inspections per find; Serting finds49/67 with58 Middle inspections and3.16 other inspections per find. Middle is not declared Healthy. Unknown U outcomes prevent measuring deployment recall, false-positive overhead or prevalence. The4473 saved rankings contain no U label claims.

Error-by-distance uses each estate's single global top25% binary allocation. Serting finds0/10 held-out positives80–150m from the nearest training positive, versus30/35 at25–80m. AirHitam finds9/24 at80–150m. These are small correlated subgroups, not stable independent distance effects. Prospective random audits and distant-block inspections remain necessary.

Some U covariates lie outside L support. Palong's minimum score is about−25.449; quadratic NDVI/location extrapolation can yield low scores for extreme low-NDVI palms. Outcomes are unknown, so these are support warnings rather than identified false negatives. Removing those interactions hurt paired validation; no V-driven score clipping or threshold repair was made.

![Inspection curves including Middle capacity](inspection_curves.png)

## Integrity, runtime and reproducibility

The clean single-file submission matches immutable V2 in **57 fits exactly**, maximum absolute difference0.0: full U, all official folds, reserved repeat717 and80/160m-buffered whole-block regions. It was not re-queried. Pipeline SHA256: `2d8a95005148f61f7d1317784c786499159a7eac891eaf5d8ee1dd4fcfe9cafe`. All4473 original U keys/order are preserved in unlabelled_ranking.csv/.parquet.

Final agentrsi selfcheck and cv both pass; see selfcheck.txt and official_cv.txt. Fresh estate subprocesses use scratch cwd/4CPUs and Python-level guards against network, source/prior/raster reads and writes outside scratch. Shape/dtype/finiteness, immutability, determinism, shuffled/duplicate rows, Middle exclusion, renamed ctx estate, empty targets and single-class fallback pass. All57 target convex optimizations in the convergence audit terminate successfully (optimizer_audit.json). Observed full-fit runtime is at most0.344s/estate and peak RSS273.2MiB. Guards are not a replacement for kernel isolation.

Only non-base runtime pin is **threadpoolctl==3.8.0**; imports otherwise use the declared base libraries/stdlib. Research-only matplotlib==3.10.7 creates standalone PNG/PDF figures and is not imported by the submission. All10 feature tables pass exact7220-key coverage, duplicates/NaN/nonfinite/forbidden-column and symlink checks. Existing tables were unchanged; total datasets before preservation are49.57MiB, below128MiB carry-forward/512MiB caps. Hashes and environment are recorded. No new extraction provenance is claimed.

Essential runtime datasets are **triage_v12,dino_sar**. Keep gee_quality,gee_canopy,optical_jitter,alphaearth for exact s009 auditing; dino_optical supports preserved older recipes. compact_triage,highres_canopy,crown_mixture duplicate consolidated runtime values and can be archived after provenance needs are met. Do not mutate dino_sar in place: alternative recipes use its other saved dates/views. Reproduction code and evidence are preserved under datasets/triage_v12/recipes/s014 for future sessions; no fitted disease weights or scores are used as feature inputs.

Reproduce:

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords s009 s012 s013 source_prior --protocols official block120 block160 buffer40 buffer80 blockquad blockquad80 blockquad160
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords s009 s012 s013 source_prior --protocols repeat701 repeat702 repeat703 repeat704 repeat705 repeat706 repeat707 repeat708 repeat709 repeat710 repeat711 repeat712 repeat713 repeat714 repeat715 repeat716 repeat717 repeat718 repeat719 repeat720
python /app/research/aggregate.py
python /app/research/confirmation_summary.py
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python /app/research/final_diagnostics.py
python /app/research/audit.py
python /app/research/plots.py
python /app/research/write_report.py
```

Use legacy_origin_probe.py only to reproduce the historical geometry diagnostic; never use legacyL80 as the official metric. Per-protocol/estate cache files contain source hashes. The old s009 far-buffer failure can stop its multi-protocol invocation; run remaining estate/protocol attempts separately as documented in failure JSONs. No cached partial composite is accepted.

## Primary literature and next evidence

The literature supports representation provenance and cautious condition screening, not a claim that these features diagnose a pathogen:

- Izzuddin et al.(2018), [airborne hyperspectral vegetation/red-edge Ganoderma analysis](https://doi.org/10.21894/jopr.2018.0037). Different sensor/study conditions; it motivates investigating canopy condition while avoiding disease-specific attribution from coarse indices.
- Brown et al.(2025), [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291). Annual geospatial embedding context; carried table provenance records2025, while the paper's released dataset description covers earlier years. Embeddings can encode environment/management as well as canopy history.
- Oquab et al.(2023), [DINOv2](https://arxiv.org/abs/2304.07193). Frozen visual encoder provenance; RGB pretraining is not evidence of SAR-specific physiological representation.
- Sagawa et al.(2019/2020), [group-shift robustness and regularization](https://arxiv.org/abs/1911.08731). Motivation for testing spatial stability controls, not evidence that our rejected background penalty is group DRO or robust in deployment.
- [scikit-learn AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): non-interpolated step AP and weighted/tie handling. Primary pages were retrieved by HTTPS into report/literature; the web search service returned errors, so no unverifiable search result is relied upon.

Highest-value next evidence is prospective original80m block inspection in all estates, including AirHitam misses, distant Serting/Palong pockets, Middle costs, unsupported covariates and a random unranked audit sample. Independent estates/years and accurately registered current crown imagery are needed before claiming transportability. Same-label CV/V micro-tuning has diminishing returns after35 recipes. Do not reopen closed mean-only SAR/imbalance/Middle routes or refit ensemble weights to noisy V. Further source-prior work should test independent domains and genuine conditional shift, with source-only estimation and the corrected union block origin preserved.
