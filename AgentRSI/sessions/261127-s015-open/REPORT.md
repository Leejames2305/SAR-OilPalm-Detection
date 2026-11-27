# Oil-palm spatial triage research, session261127-s015-open

## Decision and scope

The final submission **retains the previous s014 source-prior recipe byte-for-byte**. This session found **no verified improvement over s014**. It reproduced a substantial improvement over the designated s001 incumbent and investigated the researcher's AirHitam concerns across 86 unique recipes/comparators, plus immutable V-snapshot aliases. The complete evidence comprises 1095 estate/protocol prediction records at report generation. Eleven distinct new milestones and one s014 reference used12 of15 V queries;3 remain. No hashes were cosmetically changed to redraw noise. There is no new V-selected estate branch, ensemble weight, or tail threshold in the final package.

Selected official union-anchored80m composite lift **+.287409**, versus incumbent **+.122984**, coordinates **+.043863**, and s009 **+.160430**. Selected noisy V composite **+.224729**, delta incumbent **+.105979**, delta coordinates **+.158268**, with estate lifts **+.058767/+.344529/+.270893**. Those are this session's genuine returned values; historical noisy scores should not be substituted or treated as exact paired values. No claim is made that s014 improves historical s009 AirHitam V+.065513.

At25% binary inspection capacity the retained model captures48.2%/65.2%/74.6% of unhealthy palms. **Majority capture is still unmet in AirHitam.** Its high L AP does not resolve its smaller V lift. Other estates are equally trusted, not lower-quality labels.

## Authorized data, geometry and metrics

L contains2747 rows,2398 binary-scored trees and189 unhealthy:964/770/664 binary observations and56/66/67 positives. Palong198 and Serting151 Middle rows are excluded from every disease target and binary metric, but remain in held-out prediction rows and inclusive inspection capacity. U has4473 rows:1547/1696/1230. All7220 original coordinate keys are used as exact floating-point joins. No sealed labels, split assignments, prohibited repository or /tests were accessed. Credential contents were never printed or copied.

Every primary outer split is sklearn GroupKFold5 on the estate's full L table including Middle, with east=(Long−union_min_Long)*111320*cos(union_mean_Lat),north=(Lat−union_min_Lat)*110540. Origins use authorized L∪U; group=floor(east/80)*100000+floor(north/80). There are26/18/13 occupied L blocks. Binary filtering happens only after fold assignment. Current CLI and local predictions agree. Earlier L-only origin results are not used as current-official evidence.

The160m protocol aggregates original cells2x2; quadrants assign complete original blocks using population-weighted block-index medians.80m buffers remove any training point within80m of validation trees, including label-free image overlap checks. Repeated seeds901–910 shuffle original groups, not rows; they were frozen before their results and were not used to change recipes. Inner folds for all new residuals/selectors recover their origin from the exact-key covariate table. The historical s009 reference retains its original L-only inner origin and inherited estate branch solely for faithful comparison, which limits it as an implementation template.

AP is sklearn tie-aware step-function average precision, minus scored prevalence. Composite is the unweighted estate mean. Recall at a fixed fractional budget averages tied boundary scores; enrichment10=recall10/.10. ROC and inclusive Middle recall are in EXPERIMENTS.csv. Constant scores have zero lift. Random-row CV, accuracy, trapezoidal PR area and training-set AP are not selection measures. The CLI's fitted NDVI baseline differs from the untrained raw ranking −NDVI, which is also reported separately.

## Baselines and exact verified milestones

Values are pooled OOF AP minus prevalence, rather than a mean of fold APs. All model comparisons use paired held-out rows.

| model | AirHitam | Palong | Serting | composite | composite_delta_incumbent | composite_delta_coords |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | 0.014687 | 0.219310 | 0.134956 | 0.122984 | 0.000000 | 0.079121 |
| coords | 0.014041 | 0.039323 | 0.078224 | 0.043863 | -0.079121 | 0.000000 |
| raw_ndvi | 0.020798 | 0.039424 | 0.092299 | 0.050841 | -0.072143 | 0.006978 |
| s009 | 0.030146 | 0.252671 | 0.198473 | 0.160430 | 0.037446 | 0.116568 |
| s012 | 0.151551 | 0.306681 | 0.359326 | 0.272519 | 0.149535 | 0.228657 |
| s013 | 0.151070 | 0.312298 | 0.378503 | 0.280623 | 0.157639 | 0.236761 |
| s014 | 0.159644 | 0.311518 | 0.391065 | 0.287409 | 0.164425 | 0.243547 |

The next table is rerun from **the exact immutable helper version staged for each V call**, with a unique helper module name to prevent Python import-cache contamination. Initial exploratory code evolved; these frozen aliases, not later helpers, are authoritative for the milestones. The other exploratory local recipes, per-estate deltas versus incumbent/coords/s009/s014, source hashes, and timings are in OFFICIAL_SUMMARY.csv and EXPERIMENT_SUMMARY.csv. NA means incomplete, never zero or an average over fewer estates.

| model | AirHitam | Palong | Serting | composite | composite_delta_incumbent | composite_delta_coords |
| --- | --- | --- | --- | --- | --- | --- |
| clip3_C02 | 0.134924 | 0.288375 | 0.263817 | 0.229039 | 0.106055 | 0.185176 |
| guard25 | 0.113318 | 0.284625 | 0.269016 | 0.222320 | 0.099336 | 0.178457 |
| sign_consistent | 0.156222 | 0.282498 | 0.334368 | 0.257696 | 0.134712 | 0.213834 |
| stress_project_physical | 0.173838 | 0.321183 | 0.363791 | 0.286271 | 0.163287 | 0.242408 |
| regime_ridge | 0.072578 | 0.305941 | 0.368307 | 0.248942 | 0.125958 | 0.205079 |
| moment_train | 0.184351 | 0.311518 | 0.388982 | 0.294951 | 0.171967 | 0.251088 |
| resid_protect_full | 0.115492 | 0.317229 | 0.265672 | 0.232797 | 0.109813 | 0.188935 |
| dro02 | 0.165307 | 0.313053 | 0.380211 | 0.286190 | 0.163206 | 0.242328 |
| late_psri | 0.160852 | 0.316114 | 0.386532 | 0.287832 | 0.164848 | 0.243970 |
| ae_sar_add | 0.159451 | 0.321607 | 0.384691 | 0.288583 | 0.165599 | 0.244720 |
| inner_region | 0.134924 | 0.291261 | 0.391065 | 0.272417 | 0.149433 | 0.228554 |

| model | AirHitam | Palong | Serting | composite | delta_incumbent | delta_coords |
| --- | --- | --- | --- | --- | --- | --- |
| s014 | 0.058767 | 0.344529 | 0.270893 | 0.224729 | 0.105979 | 0.158268 |
| clip3_C02 | 0.065385 | 0.302157 | 0.197431 | 0.188324 | 0.069574 | 0.121863 |
| guard25 | 0.065055 | 0.262536 | 0.256421 | 0.194671 | 0.075920 | 0.128209 |
| sign_consistent | 0.049955 | 0.308517 | 0.222146 | 0.193539 | 0.074789 | 0.127077 |
| stress_project_physical | 0.056110 | 0.331560 | 0.261213 | 0.216294 | 0.097544 | 0.149832 |
| regime_ridge | 0.041441 | 0.332665 | 0.255962 | 0.210023 | 0.091272 | 0.143561 |
| moment_train | 0.058980 | 0.333420 | 0.262109 | 0.218169 | 0.099419 | 0.151708 |
| resid_protect_full | 0.058285 | 0.270039 | 0.202027 | 0.176784 | 0.058033 | 0.110322 |
| dro02 | 0.048047 | 0.346752 | 0.277336 | 0.224045 | 0.105295 | 0.157583 |
| late_psri | 0.058020 | 0.335741 | 0.275979 | 0.223247 | 0.104496 | 0.156785 |
| ae_sar_add | 0.059238 | 0.332732 | 0.277187 | 0.223052 | 0.104302 | 0.156590 |
| inner_region | 0.057286 | 0.339720 | 0.275248 | 0.224085 | 0.105334 | 0.157623 |

V has cached additive Gaussian noiseσ=.005. Tiny differences among s014,DRO,PSRI,GPU and inner-region are not established improvements. The prospective rule in selection_plan.json was written before milestoneF, after six exploratory queries: require V composite≥s014+.005, no estate loss>.015, at least two estate gains over incumbent, plus local spatial support. **No new candidate passed.** This rule is an operational selection rule, not statistical control for the entire adaptive campaign. Later genuine milestones cover new loss/data/representation/selection mechanisms; the remaining3 queries were conserved after diminishing returns.

## Hypotheses and outcomes

1. **Tail-safe shape and shrinkage.** Standardized clips±3; C=.002/.005/.01/.02; removal of NDVI² and NDVI×coordinate terms; polynomial-only clipping and hinges; stronger differential nuisance penalties. Clipping+C=.02 preserves near-s009 AirHitam V+.065385, but loses Palong/Serting to+.302157/+.197431. Uniform strong shrinkage does not meet the equal-estate objective. Merely clipping or imposing coefficient signs is not a guarantee of monotonic total response.
2. **Protected physical core and honest residual.** Freeze a compact quality-NDVI/local-coordinate logit and fit bounded contextual corrections to original-block OOF offsets. Compare spatial/linear/full core shapes, optical grouping, stress/full-design projection, residual damping, and red-edge instead of NDVI core. Residuals are clipped, never trained on in-sample core predictions. The strongest full-core orthogonal residual milestone has L+.115492/.317229/.265672 but V+.058285/.270039/.202027. It protects extrapolation at a large price in other-estate ranking. The exact s009 reference remains useful evidence, but its name branch is not adopted.
3. **Source-sign discipline and parameter restraint.** No pooling; source prior only on physical optical features; select source-target marginal sign agreement; nuisance projection against physical NDVI; empirical clustering-based ridge with bounded C; inner80m, buffered and quadrant architecture selection. Sign-consistent transfer/additive shape V+.193539; projected physical source-prior V+.216294; clustering ridge V+.210023. Some improve local AirHitam but not V. Inner-region chooses between clipped C=.02 and s014 using actual training-only whole-block quadrant AP with a.01 simplicity margin; its V+.224085 is essentially tied with the simpler retained recipe, and coarse-holdout selection is unstable.
4. **Convex physical aggregate/derivative restrictions.** Constrain Cov(NDVI,score)≤0 using only training covariates; separately test target-only, train+target, lower-tail ordering, vigor axis and a raw-NDVI derivative box. These are convex linear score constraints solved with SLSQP; the raw derivative box is checked atNDVI−6/−3/0/+3 and spatial±3 vertices. Training moment restriction raises AirHitam L to+.184351 and80m-buffer lift to+.143590, but V remains+.058980 with composite+.218169. The strongest local AirHitam improvement still does not establish deployment transfer. Constraints involving other optical variables do not prove causal monotonicity.
5. **Distributionally robust blocks.** Blend global balanced logistic loss with a temperature-controlled log-sum-exp of original-block balanced losses. This is a different loss, not another label-neighbor model. DRO02 improves all three buffer80 and unbuffered-quadrant estate cells versus s014, but AirHitam V falls to+.048047; composite+.224045 is effectively tied. Temperature variants.2/.5/1 do not justify further V micro-tuning.
6. **Independent later-season physical observations.** New GEE table late_optical has7220×93 exact-key columns,0NaNs. Strict SCL4/CloudScore+≥.65; mid Apr15–Aug15 and later Sep1–Nov25, native20m sampling, harmonized SR. Per-image red-edge/moisture/senescence ratios, temporal medians/p25/p75, reflectance and changes. Later NDRE2 marginal d=−.502/−.490/−.686 and PSRI=+.228/+.339/+.425 agree across estates. Most compact combinations do not improve local composite; one-feature PSRI gives L+.287832 and V+.223247. Correlation is insufficient evidence of incremental signal or disease progression. Denominator floors, masks and imputation are explicit in MANIFEST/extract.py. The later window had44–86 intersecting scenes; strict per-pixel support is recorded separately.
7. **New GPU multitemporal polarimetric representation.** T4 jobjob-0001-a350d2 extracts22-channel7×7~45m SAR patches from both dates: four log-power channels, H/A/alpha90 and signed-dB Yamaguchi components with authoritative band order/native reprojection/ROI masks. Train a CNN denoising autoencoder with optical-covariate prediction as a label-free auxiliary task, then compress64latent to32 covariate-only PCs. No disease label enters this stage; latent provenance is transductive and optical-informed, so it is not independent SAR disease evidence. Add8 training-only PCs, replace DINO, strengthen embedding, and fit protected residual variants. The additive milestone gives L+.288583 but V+.223052 and loses Serting coarse skill. No improvement was adopted. GPU outputs were fetched, encoder/PCA saved, and VM stopped; total billed GPU about91seconds. The final pipeline imports neither torch nor rasterio.

## AirHitam tail audit and limits of the physical interpretation

The supplied observation is **confirmed as a ranking/distribution phenomenon**. For U raw-NDVI z<−2.5 relative to binary L, counts92/80/41. AirHitam's92 trees average risk percentile85.1 under incumbent,82.8 s009,25.2 s012,22.2 s013,21.0 s014. Only1/92 is in s014's top quarter; s009 includes63/92. At z<−3,57 U trees average16.0th percentile under s014 versus86.1st under s009. Overall AirHitam U Spearman correlation(score,−NDVI)=−.133 under s014 versus+.529 s009. The high-edge subset also ranks much lower. tail_predictions.csv,parent_u_predictions.parquet and tail_ranking figures preserve the facts.

The support shift is real but **not all low-tail values are beyond the labelled min/max**: AirHitam LminNDVI=.6838,Umin=.6430, but only2 U rows are below the absolute L minimum. U has57/1547 below−3SD versus16/964 in binary L; a density/support problem is broader than a simple out-of-range flag. NDVI is not a verified disease diagnosis. In fact, **all16 labelled AirHitam trees below−3SD are Healthy**;19 below−2.5SD contain2 positives. Thus the fitted tail curvature has evidence in L, and forcing all low-NDVI U trees upward cannot be assumed correct. Empty/young crowns, soil exposure, stand structure or differing disease manifestations are possible explanations requiring inspection, not inferred diagnoses. No U labels were available or sought.

The counterfactual audit holds derived covariates fixed and perturbs only raw NDVI, so it isolates a partial model derivative and does not simulate a physical canopy change. Full derived-index counterfactuals would require a radiative/observation model. Exact fitted weights and contributions show both mechanisms: AirHitam NDVIβ−.173 and qNDVIβ−.572 raise tail risk, while quadraticβ−.172,log_green_redβ+.828,edge/SD negative weights and deblur_GRVIβ+1.097 lower it. In the92-tree tail, raw-NDVI term contributes+.662,raw quadratic−.875,qNDVI+.917,log_green_red−1.079,SD−1.033,edge−.531,deblurGRVI−.906 average logits. Individual coefficients are collinear conditional responses, not biological effect estimates. Model-captured exact coefficients reproduce every exported U score withzero numerical discrepancy.

| feature | AirHitam | Palong | Serting |
| --- | --- | --- | --- |
| NDVI | -0.202 | -0.320 | -0.575 |
| deblur_GRVI | 0.717 | -0.207 | -0.347 |
| deblur_NDVI | -0.110 | -0.412 | -0.692 |
| height_mean6 | 0.317 | -0.517 | -0.269 |
| log_green_red | 0.633 | -0.129 | -0.359 |
| ndre_contrast | -0.292 | -0.659 | -0.711 |
| ndvi_dip | 0.096 | 0.562 | 0.633 |
| ndvi_edge | -0.289 | 0.053 | 0.120 |
| ndvi_sd20 | -0.294 | 0.062 | 0.125 |
| qNDMI | -0.128 | -0.232 | -0.741 |
| qNDRE | -0.371 | -0.625 | -0.828 |
| qNDVI | -0.195 | -0.355 | -0.653 |
| qNIRv | -0.057 | -0.319 | -0.641 |

ndvi_sd20 and ndvi_edge marginal signs are **negative in AirHitam and positive in Palong/Serting**, while height_mean6 reverses in the other direction; log_green_red and deblur_GRVI also flip. All three core vegetation indices remain negative marginally. Strong negative conditional texture coefficients in every estate therefore cannot be called sign-correct physiology. Source-sign restriction is tested explicitly and not retained. Exact raw-NDVI partial derivative is wrong-signed for about81.5% of AirHitam low-tail U trees and100% in the other two tails; these partial inversions do not establish wrong health rankings.

## Paired spatial robustness and uncertainty

| model | protocol | AirHitam | Palong | Serting | composite | composite_delta_incumbent | composite_delta_coords |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ae_sar_add | block160 | 0.120445 | 0.326109 | 0.327136 | 0.257897 | 0.121221 | 0.235329 |
| ae_sar_add | blockquad | 0.108709 | 0.332609 | 0.320790 | 0.254036 | 0.129520 | 0.229432 |
| ae_sar_add | blockquad80 | 0.106291 | 0.193570 | 0.218997 | 0.172953 | 0.101097 | 0.159849 |
| ae_sar_add | buffer80 | 0.137072 | 0.265765 | 0.156650 | 0.186496 | 0.118806 | 0.180722 |
| coords | block160 | -0.004782 | -0.002829 | 0.075313 | 0.022567 | -0.114108 | 0.000000 |
| coords | blockquad | -0.007086 | 0.002919 | 0.077979 | 0.024604 | -0.099912 | 0.000000 |
| coords | blockquad80 | 0.013504 | 0.000243 | 0.025563 | 0.013103 | -0.058752 | 0.000000 |
| coords | buffer80 | -0.008211 | -0.001858 | 0.027389 | 0.005773 | -0.061916 | 0.000000 |
| dro02 | block160 | 0.124167 | 0.331158 | 0.368318 | 0.274548 | 0.137872 | 0.251980 |
| dro02 | blockquad | 0.118181 | 0.344610 | 0.362472 | 0.275088 | 0.150572 | 0.250484 |
| dro02 | blockquad80 | 0.114901 | 0.183600 | 0.288079 | 0.195527 | 0.123671 | 0.182424 |
| dro02 | buffer80 | 0.110190 | 0.265009 | 0.198843 | 0.191347 | 0.123658 | 0.185574 |
| incumbent | block160 | 0.027631 | 0.281752 | 0.100645 | 0.136676 | 0.000000 | 0.114108 |
| incumbent | blockquad | 0.033334 | 0.226245 | 0.113968 | 0.124516 | 0.000000 | 0.099912 |
| incumbent | blockquad80 | 0.025393 | 0.177811 | 0.012362 | 0.071855 | 0.000000 | 0.058752 |
| incumbent | buffer80 | 0.006083 | 0.145824 | 0.051160 | 0.067689 | 0.000000 | 0.061916 |
| s009 | block160 | 0.046910 | 0.307538 | 0.168194 | 0.174214 | 0.037538 | 0.151646 |
| s009 | blockquad | 0.051420 | 0.250093 | 0.188456 | 0.163323 | 0.038807 | 0.138719 |
| s009 | blockquad80 | 0.049532 | 0.191683 | 0.039234 | 0.093483 | 0.021627 | 0.080379 |
| s009 | buffer80 | 0.019129 | 0.160184 | 0.078079 | 0.085797 | 0.018108 | 0.080024 |
| s014 | block160 | 0.120163 | 0.332607 | 0.363554 | 0.272108 | 0.135432 | 0.249541 |
| s014 | blockquad | 0.111843 | 0.331240 | 0.357541 | 0.266875 | 0.142359 | 0.242271 |
| s014 | blockquad80 | 0.106479 | 0.184292 | 0.245475 | 0.178749 | 0.106893 | 0.165645 |
| s014 | buffer80 | 0.106046 | 0.261029 | 0.179757 | 0.182277 | 0.114588 | 0.176504 |

The103m incumbent SAR encoder footprint can overlap adjacent80m blocks. The new45m autoencoder reduces support but still overlaps near block borders.80m buffers and contiguous-region tests materially reduce that concern; they do not prove pathogen causality, new-estate transportability, or full independence of global optical inversions. Coarse holdouts are stress tests, not interchangeable versions of official AP. The previous s014 report records very weak Palong160m-buffered-quadrant extrapolation; that limitation is not resolved by this session.

Frozen901–910 comparisons reuse the same L labels. They support conditional fold stability, not independent replication:

| baseline | metric | repeats | mean_lift | mean_delta | min_delta | wins |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent | AirHitam | 10 | 0.126513 | 0.105182 | 0.081862 | 10 |
| incumbent | Palong | 10 | 0.294703 | 0.052538 | -0.017065 | 9 |
| incumbent | Serting | 10 | 0.330373 | 0.188036 | 0.130993 | 10 |
| incumbent | composite | 10 | 0.250529 | 0.115252 | 0.084411 | 10 |
| coords | AirHitam | 10 | 0.126513 | 0.111274 | 0.088115 | 10 |
| coords | Palong | 10 | 0.294703 | 0.241461 | 0.206202 | 10 |
| coords | Serting | 10 | 0.330373 | 0.254582 | 0.167222 | 10 |
| coords | composite | 10 | 0.250529 | 0.202439 | 0.168325 | 10 |
| s009 | AirHitam | 10 | 0.126513 | 0.085860 | 0.065897 | 10 |
| s009 | Palong | 10 | 0.294703 | 0.022238 | -0.033186 | 7 |
| s009 | Serting | 10 | 0.330373 | 0.139023 | 0.081316 | 10 |
| s009 | composite | 10 | 0.250529 | 0.082374 | 0.056399 | 10 |

Paired4000-replicate original80m block bootstrap conditions on fixed OOF fits. Weighted sklearn AP is equivalent to resampling complete blocks. It ignores training/refitting and adaptive selection uncertainty. The composite s014-minus-incumbent95% interval is approximately[+.08335,+.23425], coordinates[+.15989,+.30441]. Palong's individual incumbent interval crosses0; do not claim every estate improvement is individually significant.

| baseline | estate | point_delta | lower95 | upper95 | probability_positive |
| --- | --- | --- | --- | --- | --- |
| incumbent | AirHitam | 0.144957 | 0.030429 | 0.268760 | 0.998250 |
| incumbent | Palong | 0.092209 | -0.081168 | 0.257739 | 0.837250 |
| incumbent | Serting | 0.256110 | 0.151826 | 0.314796 | 1.000000 |
| incumbent | composite | 0.164425 | 0.083350 | 0.234250 | 1.000000 |
| coords | AirHitam | 0.145603 | 0.020548 | 0.278981 | 0.996500 |
| coords | Palong | 0.272195 | 0.141785 | 0.396438 | 1.000000 |
| coords | Serting | 0.312841 | 0.149388 | 0.367752 | 0.999750 |
| coords | composite | 0.243547 | 0.159893 | 0.304409 | 1.000000 |

## Operational ranking and errors

| estate | recall | precision | fp_per_detected | inclusive_middle_recall |
| --- | --- | --- | --- | --- |
| AirHitam | 0.4821 | 0.1120 | 7.9259 | 0.4821 |
| Palong | 0.6515 | 0.2234 | 3.4767 | 0.6364 |
| Serting | 0.7463 | 0.3012 | 2.3200 | 0.7313 |

At25% inspection, estimated false-positive inspections per unhealthy tree detected are7.93/3.48/2.32 on binary OOF rows. With Middle consuming inspection capacity, recall is48.2%/63.6%/73.1%. These are validation diagnostics, not field yield guarantees. V has no disclosed operational recall statistics and no U outcomes; no U precision is reported.

Raw−NDVI ranking has L AP lifts+.02080/+.03942/+.09230 and recall25≈33.9%/37.9%/54.5%, distinct from the CLI fitted baseline. Retained model improvement over coordinates and raw stress supports added measured-condition information, while spatial/error analysis still shows unseen-pocket weakness. error_positive_distance.csv scores each held-out positive's distance to training positives using only that fold's training labels; wider-distance groups remain small. Serting has0/10 positives at80–150m found at25% capacity; this is an unresolved distant-pocket failure. Ranks for all4473 U palms are exported in original estate row order, including every unmarked Middle. top25_inspection selects ceil(.25*n) operational rows with deterministic input-order tie handling; reported metrics use fractional tie averaging.

## Final offline recipe and execution checks

Selected code is the unchanged s014 source-prior pipeline, not robust.py or any new candidate. Per estate: training-only standardization of rawNDVI/local coordinates; degree2 terms shrunk by.3;12optical columns*.5;8training PCs of64AlphaEarth contrasts*.5;3historical height columns*.35 using3m/1m/1m uncertainty floors;9deblur indices*.3;8training PCs of frozen October103m DINO SAR*.3;intercept10. Fit each source independently with C=.15,class-balanced ridge; average its12optical coefficients equally to centre the target ridge prior. All contextual/spatial weights remain target-local. All estates' labels have equal trust. No estate-name branch, target label-neighbor score, Middle target, pseudo-label, network or inference-time extraction.

Only runtime datasets triage_v12 and dino_sar are required; other tables are retained for reproducibility.12feature tables total about55.83MiB including scripts/weights and recipe archive; each covers7220 exact keys with0NaNs/duplicates/forbidden names, and no symlinks. New late_optical and temporal_sar have manifests and reproducible extraction/training scripts. Runtime pins only threadpoolctl==3.8.0 beyond the base environment. Required selfcheck and CLI CV pass. Fresh subprocess with cwd=scratch and cpus4 takes≤0.289s/estate and peakRSS≤275.0MiB. All outputs are aligned finitefloat64; permutation,empty U,Middle removal,and exact exported-score checks pass. Byte hashes match V query1. contract_verification.json and fresh_subprocess_contract.json distinguish the larger research-process memory from actual fresh-runtime memory.

## Literature and provenance

- Roberts etal.(2017),[Cross-validation strategies for data with temporal,spatial,hierarchical,or phylogenetic structure](https://doi.org/10.1111/ecog.02881): structured dependence warrants corresponding validation designs. Publisher full text was access-restricted; Crossref's publisher-supplied abstract/metadata was retrieved and archived. No unsupported detailed quotation is used.
- Liaghat etal.(2014),[Early detection of basal stem rot disease in oil palms based on hyperspectral reflectance data using pattern recognition algorithms](https://doi.org/10.1080/01431161.2014.903353),and[Analysis of airborne hyperspectral image using vegetation indices,red edge position and continuum removal for detection of Ganoderma disease in oil palm](https://doi.org/10.21894/jopr.2018.0037): motivate optical/red-edge measurements. Their sensor/design/results are not treated as expected Sentinel-2 per-tree performance. Publisher metadata and accessible primary pages are archived.
- Sagawa etal.(2019),[Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731): motivates group-risk robustness. This session's finite-block softmax logistic objective is an explicit adaptation, not a reproduction of that neural training procedure.
- [Sentinel-2 SR Harmonized Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED),[Cloud Score+ catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED): source/quality definitions. Harmonized data already handle baseline04.00 offsets; no second+1000DN correction.
- Oquab etal.(2023),[DINOv2](https://arxiv.org/abs/2304.07193): provenance of carried frozen RGB encoder, not evidence that RGB features identify SAR disease physiology. The new CNN has no pretrained weights.
- [sklearn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): precise AP/tie semantics. Base requirements are supplied by the task; research plotting uses matplotlib3.10.7 separately.

The web search tool returned service500 errors. Primary webpages and publisher metadata were subsequently fetched directly with requests; literature files are in report/literature. No biological claim depends on inaccessible full text.

## Artifacts and next decision

REPORT.md,HANDOFF.md,selection_plan.json,selection.json;EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,OFFICIAL_SUMMARY.csv,VERIFIER_SUMMARY.csv;verifier/full_history.json and per_estate_results.json;frozen_repeat_plan.json,repeat_comparison.csv,bootstrap.csv;tail_predictions.csv,covariate_shift.csv,L_tail_labels.csv,marginal_signs.csv,final_coefficients.csv,partial_derivatives.csv;inspection_diagnostics.csv,error_positive_distance.csv,unlabelled_ranking.parquet/.csv;selfcheck.txt,official_cv.txt,contract_verification.json,fresh_subprocess_contract.json,dataset_audit.json,import_audit.json;threePNG/PDF figure pairs;research scripts,immutable verified snapshots and cached actual OOF predictions. A label-free recipes/s015 archive under triage_v12 preserves source/config/reproduction code only; metrics CSVs remain under report.

The highest-value next evidence is a prospective original-block inspection sample separating low-NDVI crowns from disease, including s009/s014 disagreements and unranked controls in all three estates, followed by genuinely independent estate/year validation. Further tuning on the same L and noisy V has diminishing value. None of this session's changed models should be described as an improvement over the retained recipe.
