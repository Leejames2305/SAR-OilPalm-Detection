# Oil-palm disease triage — cross-session synthesis 261127-s013

## Outcome and scope

The final submission is **pooled_sar**, a single offline CPU recipe combining s011/s012's uniform compact condition/context design with s010's shared-optical/local-deviation architecture and October SAR patch representation. All three estates' labels are equally trusted and estates have equal evaluation weight. No estate-name capacity branch, disease-label neighbor, pseudo-label, or supervised Middle target is used.

Official 80 m composite AP lift is **+0.288087**, versus incumbent +0.140519, coordinates +0.089593, s010 +0.255805, and s012 +0.269549. Noisy V composite is **+.220382**, delta incumbent **+.101632**, delta coordinates **+.153921**. All three V estate deltas versus both required comparators are positive. These are ranking results, not pathogen diagnosis, calibrated disease probabilities, or prospective field recall.

The best reference V composite is s010 +.216825: the new +.003557 difference is below the .005 single-observation noise scale and is **not evidence of a confirmed V win over s010**. The synthesis trades Palong V against s012 (−.024857) and Serting V against s010 (−.046352). It does not dominate every parent in every estate.

**Selection-policy disclosure:** the provisional exploration plan included a stricter .015 maximum V estate loss relative to the s012 checkpoint. V8 fails that guard in Palong. The final choice explicitly changes that provisional rule after V, choosing the task's incumbent-relative scientific criteria and positive mean L increments in every estate over s012. This is a V-informed, exploratory decision, not a passed preregistered incremental-parent gate. The conservative s012 package remains byte-preserved under datasets/triage_v12/recipes/s013/parents/s012. No ensemble weights or estate gates were fitted to V, and no further V calls followed V8.

At a top-25% binary budget, recall is 48.2% / 74.2% / 76.1%. Counting Middle against capacity gives 48.2% / 71.2% / 74.6%. AirHitam's majority-capture objective remains unmet at this budget. A distant buffered Palong stress test remains weaker than incumbent. These limitations constrain deployment claims despite the large composite gain.

## Data, provenance and validation

L has 2,747 rows: 2,398 binary-scored palms (189 positives) and 349 Middle. Estate binary counts/positives are 964/56, 770/66, 664/67. U has 4,473 rows; every row is returned in exact input order. Middle is excluded from binary training and AP, and included only in capacity diagnostics. Other estates' L labels may enter through ctx.other_labeled, as authorized; target held-out labels never enter training.

Only **10** of the 18 advertised carried feature tables were actually staged. All dependencies of every staged submission are present. Missing advertised assets were autumn, gee_s2_stress, optical_aligned, optical_texture, planting_geometry, sar_native, sar_structure and terrain_canopy. They were not silently reconstructed or re-extracted. No new raster/GEE/GPU feature acquisition was necessary. The original source tables, raster data, prior files and credentials are never accessed by fit_predict.

Official folds exactly match agentrsi cv: local east=(Long−minLong)×111320×cos(meanLat), north=(Lat−minLat)×110540; block=floor(east/80)×100000+floor(north/80). There are50/39/13 occupied L blocks in AirHitam/Palong/Serting, respectively, and each contains binary-scored rows. GroupKFold(5) allocates the entire L table including Middle, followed by binary metric filtering. All supervised scaling, PCA and fitting use current training rows only. The required coordinates baseline is distance-weighted 15NN on training-standardized Long/Lat, fitted to binary labels. AP is sklearn's tie-aware step-function average_precision_score; lift subtracts scored prevalence, then estates receive an unweighted mean. Fractional boundary inclusion handles inspection-score ties.

Exploration reallocates complete 80 m blocks with seeds21–25. First confirmation uses seeds101–110. **Frozen final confirmation uses seeds401–420**, after V8 selection and without further model changes. Repeats reuse the same labels/imagery; they do not create independent estates or trials. The full table contains 45 models and 2415 completed estate/protocol evaluations. Failed runs are not replaced with optimistic scores or partial-estate composites.

Wider protocols: 120/160 m blocks; shifted80 m grid (+40m); 40/80 m nearest-held-out-tree exclusion buffers; population-weighted whole-block quadrants (blockquad), with80/160 m buffers. The older median-coordinate quadrants reproduce prior results but split some original80 m blocks and are secondary stress diagnostics. The unweighted occupied-block-median quadrant (quadrant/regions80) emphasizes sparse tails and can hold out590/815 Serting rows, leaving only8 training positives (7 with80 m buffer). It is a harsh sample-size/extrapolation test, not a balanced region estimate. All fold sizes, positive counts and distances are in spatial_split_audit.csv.

The final SAR view is approximately103 m square, so80 m blocking or buffering does not guarantee independent raw patches. Whole-block160 m region buffers provide a stronger direct-overlap check for the final SAR crops; scores are positive but lower. The optical inverse has a weak global influence tail, so even that buffer does not establish complete image independence or disease causality. No random-row validation or same-block label stacking was used.

## Unified cross-lineage audit

L values below are new paired reruns. V values for reference sessions are the **supplied verifier-verified reward table**, not substituted report maxima. Their reports contain slightly different noisy observations (s010 +.219732, s012 +.205755, etc.); those are historical observations, not repeated evidence here. Incumbent/coords V references are the scorer-returned baseline values. The preserved s012 bag V is the previous handoff's observation. We did not spend queries re-scoring unchanged parents.

| Model | L AirHitam | V AirHitam | L Palong | V Palong | L Serting | V Serting | L composite | V composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | +0.030965 | +0.037028 | +0.255637 | +0.253306 | +0.134956 | +0.065918 | +0.140519 | +0.118751 |
| coords | +0.010793 | +0.039792 | +0.179763 | +0.127807 | +0.078224 | +0.031787 | +0.089593 | +0.066462 |
| s009 | +0.050556 | +0.065513 | +0.288720 | +0.276497 | +0.198473 | +0.188360 | +0.179250 | +0.176790 |
| s010 | +0.105714 | +0.054553 | +0.381314 | +0.289959 | +0.280387 | +0.305963 | +0.255805 | +0.216825 |
| s011 | +0.067653 | +0.037031 | +0.351565 | +0.338494 | +0.337580 | +0.171891 | +0.252266 | +0.182472 |
| s012 | +0.066466 | +0.049552 | +0.382857 | +0.363177 | +0.359326 | +0.197506 | +0.269549 | +0.203411 |
| s012_bag | +0.087399 | +0.049482 | +0.386420 | +0.366903 | +0.328326 | +0.225594 | +0.267382 | +0.213993 |
| pooled_sar | +0.075808 | +0.063217 | +0.409951 | +0.338320 | +0.378503 | +0.259611 | +0.288087 | +0.220382 |

Mechanistic divergence is reproducible in L, but L does not reliably predict V estate ordering. s010's Serting L lift is lower than s012, including repeated assignments, while its V lift is about.108 higher. Palong L is nearly tied between those parents, yet s012 V is about.073 higher. AirHitam s010 L advantage shrinks sharply in V. These patterns support conditional/geographical instability and small-sample uncertainty; they do not prove a specific cause or justify questioning any estate's labels. Even stronger contiguous-region performance does not guarantee the V ordering.

Parent prediction disagreement is substantial enough to motivate controlled crossover, but does not itself establish an ensemble gain:

| estate | first | second | spearman | top25_jaccard | top25_disagreement |
| --- | --- | --- | --- | --- | --- |
| AirHitam | s009 | s012 | +0.626191 | +0.397101 | +0.215768 |
| AirHitam | s010 | s012 | +0.796773 | +0.530159 | +0.153527 |
| Palong | s009 | s012 | +0.816173 | +0.600832 | +0.124675 |
| Palong | s010 | s012 | +0.896121 | +0.688596 | +0.092208 |
| Serting | s009 | s012 | +0.683721 | +0.566038 | +0.138554 |
| Serting | s010 | s012 | +0.806113 | +0.693878 | +0.090361 |

top25_disagreement is the fraction of binary rows with different fractional selection; Jaccard uses fractional tie membership. These are official OOF predictions from the same folds.

## Controlled mechanisms and attribution

1. **s011/s012 uniform capacity:** replace s010's inherited estate branch with quadratic terms shrunk by.3, holding all other s010 features/pooling fixed (s010_curvature). L composite rises from+.255805 to+.269130; V+.214764 remains close to the parent reward. This removes a name branch without explaining the parent V divergence.
2. **s010 patch families into s012:** October SAR, optical NRG160, and both were added jointly or as s009-style block-OOF bounded residuals. Joint SAR improves repeated Serting most, while bounded offsets change rankings less. V1 SAR+.208825 and V2 offsets+.201727 show that L gains alone did not preserve Palong V. The final retains only October SAR; optical NRG160 did not earn its extra size/footprint.
3. **s010 pooling into s012:** shared physical optical coefficients plus independent estate deviations give pool_fullopt+.279816 official. Shared-only4-feature controls were weaker. Self-duplicate and source-label-permutation controls show that part of the gain is changed regularization, not demonstrated transferred physiology. The final pooled_sar combines that architecture with local SAR PCs; a common aligned SAR basis shared across estates was rejected without V because fresh repeat Palong deteriorated.
4. **Stability/bagging:** the preserved s01232-block probability ensemble and new SAR/both-patch bags were tested under the same protocols. SAR bag gives+.283051 official and positive repeat means in all estates, but V+.206329 does not improve the protected checkpoint balance. The final is one joint logistic fit, with no score averaging.
5. **Feature-gap crossover:** s010_curvature omits the compact moisture/quality, physical-floor height and mixture measurements. Adding them locally, jointly or bounded around the frozen parent, raises L Palong (legacy_complement+.282713 composite) but V6/V7 remain about+.214 and preserve the older Palong weakness. This is direct evidence that simply adding the missing family does not reproduce s012's V gain.
6. **Collinearity and pruning:** raw-vs-standardized AlphaEarth PCA, quality-NDVI blending, quality-family deletion and deblur redundancy pruning were controlled. Small local gains were mixed; no dimension, sensor-weight or V-feature sweep followed. The final uses unchanged s012 optical/height/mixture families and PCA counts, with the s010 architectural crossover.

Final ablations are descriptive after freezing, not a new selection step:

| Recipe | AirHitam | Palong | Serting | Composite | Delta incumbent | Delta coords |
| --- | --- | --- | --- | --- | --- | --- |
| pooled_sar | +0.075808 | +0.409951 | +0.378503 | +0.288087 | +0.147568 | +0.198494 |
| uniform_sar | +0.072954 | +0.399129 | +0.370655 | +0.280913 | +0.140393 | +0.191319 |
| pool_fullopt | +0.076162 | +0.392518 | +0.370768 | +0.279816 | +0.139297 | +0.190223 |
| pooled_sar_self | +0.058947 | +0.406681 | +0.375817 | +0.280481 | +0.139962 | +0.190888 |
| pooled_sar_permuted | +0.072433 | +0.394216 | +0.374179 | +0.280276 | +0.139757 | +0.190682 |
| pooled_sar_nocoords | +0.067696 | +0.228800 | +0.386518 | +0.227671 | +0.087152 | +0.138078 |
| pooled_sar_noalpha | +0.095660 | +0.373262 | +0.317053 | +0.261992 | +0.121473 | +0.172398 |
| pooled_sar_no_compactopt | +0.065827 | +0.338520 | +0.274256 | +0.226201 | +0.085682 | +0.136608 |
| pooled_sar_nostructure | +0.066457 | +0.381690 | +0.375754 | +0.274634 | +0.134114 | +0.185040 |

No-coordinates still retains imagery that can encode geography, so it is not a pure disease-signature test. Its large Palong loss establishes dependence on spatial context. Real source labels, self-duplication and one fixed permutation distinguish source information from effective penalty changes; one permutation is not a permutation-test p-value.

**Adopted:** uniform differential ridge and compact optical/AlphaEarth from s011; physical-floor historical height and label-free crown-mixture covariates/C=.15 from s012; shared-optical/local-deviation design and frozen October SAR view from s010. **Retired from final:** s009 nested offsets/one-sided vigor and reconstruction branch, s010 estate-name branch and optical-patch group, s012 probability bagging, common latent source pooling, new feature extraction and unvalidated pruning. Sources for every parent and V candidate are preserved; this session does not repeat the older unavailable-source narrative.

## Exact final training recipe

For the target estate and each source estate independently, keep Healthy/Unhealthy and construct49 local predictors:9 standardized NDVI/east/north degree≤2 terms (six degree2 terms multiplied.3),12 standardized compact optical features×.5,8 training PCs of standardized64-dimensional2025 AlphaEarth local contrast×.5,3 historical height features×.35 with sqrt(training_variance+[3m,1m,1m]^2) scaling,9 standardized deblur optical features×.3, and8 training PCs of frozen October SAR64-view PCs×.3. SAR PCA uses saved compressed components directly, then standardizes its8 outputs; AlphaEarth inputs are standardized before PCA. Both use deterministic full SVD.

The12 optical measurements additionally form a shared standardized block×.3. Each estate has its own49-column local block and an intercept indicator×10. With three estates there are162 classifier columns:12+3×49+3. Set each estate's total sample weight to the target binary training size, half on each class. Fit LogisticRegression(C=.15,max_iter=2000,tol=1e-7,fit_intercept=False). Return the target block decision function as finite float64. This shares optical coefficients while retaining local deviations and independent spatial/latent/structural responses. No physical sign constraint is imposed on estate-brittle context.

The two runtime tables are triage_v12/features.parquet and dino_sar/features.parquet. The former consolidates compact optical, prior-year embeddings, historical height and optical inversion; the latter supplies only oct103_ components. Runtime reads them through ctx.datasets_dir, uses ctx.cpus and writes nothing. No network/GPU/raster/library outside the declared pins is needed. Historical height predates2026; annual embeddings encode2025 context; frozen SAR RGB features are out-of-domain representations. None is a measured pathogen biomarker. Existing manifests/scripts specify native raster CRS reprojection, HH/HV/VV bands1/2/4, positive linear-power support and10log10 display preprocessing. No Yamaguchi/T3 extraction was performed this session.

## Repeated blocks and conditional uncertainty

| estate | Lift | Delta_incumbent | Worst_delta_incumbent | Delta_coords | Delta_s010 | Delta_s012 | s012_wins | assignments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | +0.122462 | +0.097969 | +0.060505 | +0.089082 | +0.001312 | +0.008926 | 17 | 20 |
| Palong | +0.325169 | +0.087521 | +0.020172 | +0.190493 | +0.000937 | +0.004100 | 13 | 20 |
| Serting | +0.318218 | +0.184883 | +0.121879 | +0.252685 | +0.077634 | +0.036063 | 20 | 20 |

Frozen20 assignments: incumbent wins 60/60 estate cells, coords wins 60/60. Composite beats incumbent 20/20 and s012 20/20. Means in every estate improve s012, but several individual AirHitam/Palong assignments do not. All parents were also run on these exact assignments:

| model | Composite | Delta_incumbent | Delta_coords | Assignments |
| --- | --- | --- | --- | --- |
| coords | +0.077863 | -0.053962 | +0.000000 | 20 |
| incumbent | +0.131825 | +0.000000 | +0.053962 | 20 |
| pooled_sar | +0.255283 | +0.123458 | +0.177420 | 20 |
| s009 | +0.163771 | +0.031946 | +0.085908 | 20 |
| s010 | +0.228655 | +0.096830 | +0.150793 | 20 |
| s011 | +0.224740 | +0.092915 | +0.146877 | 20 |
| s012 | +0.238920 | +0.107095 | +0.161057 | 20 |
| s012_bag | +0.246748 | +0.114923 | +0.168885 | 20 |

Paired4,000-resample bootstrap uses original80 m blocks within each estate, fixed official OOF predictions and common multiplicities for each comparator. Reported intervals condition on the adaptive campaign and fixed fits; they do not refit, correct model selection, remove adjacent-block dependence, or establish new-estate/year significance.

| reference | estate | mean | lower | upper | fraction_positive |
| --- | --- | --- | --- | --- | --- |
| incumbent | composite | +0.142574 | +0.081769 | +0.206256 | +1.000000 |
| coords | composite | +0.185595 | +0.119152 | +0.246979 | +1.000000 |
| s009 | composite | +0.098671 | +0.032949 | +0.163827 | +0.997500 |
| s010 | composite | +0.025759 | -0.013445 | +0.061788 | +0.896750 |
| s011 | composite | +0.030927 | +0.008620 | +0.051823 | +0.995250 |
| s012 | composite | +0.015486 | -0.002567 | +0.032180 | +0.956750 |
| s012_bag | composite | +0.018361 | -0.003699 | +0.039522 | +0.954000 |

The point official composite delta versus incumbent is+.147568, with95% interval[+.081769,+.206256]; coordinates point delta+.198494, interval[+.119152,+.246979]. Bootstrap means differ from original point estimates because AP is nonlinear. Incremental intervals versus s010 and s012 cross zero. AirHitam's individual incumbent interval also crosses zero; no universal individual-estate significance claim is made.

## Wider spatial holdouts and failures

| protocol | AirHitam | Palong | Serting | Composite | delta_incumbent | delta_coords | delta_s010 | delta_s012 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official | +0.075808 | +0.409951 | +0.378503 | +0.288087 | +0.147568 | +0.198494 | +0.032283 | +0.018538 |
| shift40 | +0.139074 | +0.318286 | +0.317914 | +0.258425 | +0.117518 | +0.190439 | +0.031787 | +0.019971 |
| block120 | +0.119031 | +0.388881 | +0.306073 | +0.271328 | +0.142473 | +0.179778 | +0.028877 | +0.025071 |
| block160 | +0.066008 | +0.310291 | +0.339488 | +0.238596 | +0.124009 | +0.172785 | +0.034578 | +0.024949 |
| buffer40 | +0.080518 | +0.307001 | +0.317880 | +0.235133 | +0.124589 | +0.219732 | +0.046756 | +0.020460 |
| buffer80 | +0.074840 | +0.146184 | +0.158784 | +0.126603 | +0.083973 | +0.100003 | +0.044102 | +0.007845 |
| blockquad | +0.114559 | +0.288594 | +0.332983 | +0.245379 | +0.111605 | +0.197116 | +0.036737 | +0.024120 |
| blockquad80 | +0.064352 | +0.130233 | +0.242172 | +0.145586 | +0.082881 | +0.133347 | +0.028935 | +0.032315 |
| blockquad160 | +0.028348 | +0.123984 | +0.040305 | +0.064212 | +0.047207 | +0.037080 | NA | +0.000738 |
| quadrant | +0.114559 | +0.284215 | +0.008666 | +0.135813 | +0.057745 | +0.123538 | +0.005907 | +0.004062 |
| regions80 | +0.064352 | +0.169224 | +0.030676 | +0.088084 | +0.044529 | +0.081487 | +0.028489 | +0.004597 |
| medianquad | +0.115765 | +0.308222 | +0.384155 | +0.269380 | +0.139453 | +0.213921 | -0.001681 | +0.028705 |
| median80 | +0.060490 | +0.233609 | +0.104785 | +0.132961 | +0.059913 | +0.132202 | -0.009668 | +0.031630 |

The final exceeds incumbent in all estates on120/160 m blocks, shifted grid and40/80 m random-block buffers. It trails incumbent in **Palong blockquad80** by−.016145; at160 m region buffer training may be as small as3 binary rows in one Palong fold, and the recipe's reduced-PC/fallback behavior matters. Non-monotonic results between80/160 m buffers reflect changed training support and must not be interpreted as a smooth robustness curve. Sparse-tail quadrant Serting lift is close to zero even though its official score is strong. S012 shares this weakness. This is additional information beyond15NN, not proof of spatially invariant disease physiology.

Unmodified s009 fails Serting's legacy median80 inner validation (<5 groups) and Palong blockquad160 (fixed8 PCs with only3 samples). s010 also fails that Palong160 m stress for fixed8 PCs; AirHitam and Serting were attempted separately and their available results are retained. Early generic offset prototypes fail a one-class inner fit under buffer80 and have NA for the unfinished estate/protocol. Those failures are documented in research logs/JSON, not filled with scores. An empty-shared-block harness bug was fixed and pool_no_share rerun; a pre-build invocation of nocoords was rerun after its file existed. No primary parent or final official/frozen-assignment result failed.

![Spatial holdout sensitivity](spatial_robustness.png)

## V observations and final decision

Eight distinct mechanisms/crossovers were queried, all valid;7/15 remain. Gaussian noiseσ=.005 is cached per content hash. V observations are adaptive and correlated in model provenance. No cosmetic re-query, tail clipping, source-label recovery, or V-fitted ensemble weight was used.

| Query | Recipe | AirHitam | Palong | Serting | Composite | Delta_incumbent | Delta_coords |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | uniform_sar | +0.057696 | +0.335938 | +0.232841 | +0.208825 | +0.090075 | +0.142363 |
| v2 | both_offset | +0.046851 | +0.349354 | +0.208978 | +0.201727 | +0.082977 | +0.135265 |
| v3 | pool_fullopt | +0.042317 | +0.341307 | +0.225932 | +0.203185 | +0.084435 | +0.136723 |
| v4 | s010_curvature | +0.059791 | +0.279899 | +0.304603 | +0.214764 | +0.096014 | +0.148302 |
| v5 | sar_bag | +0.052732 | +0.328260 | +0.237995 | +0.206329 | +0.087578 | +0.139867 |
| v6 | legacy_complement | +0.057933 | +0.278343 | +0.306204 | +0.214160 | +0.095409 | +0.147698 |
| v7 | legacy_protected | +0.055311 | +0.280931 | +0.304323 | +0.213522 | +0.094771 | +0.147060 |
| v8 | pooled_sar | +0.063217 | +0.338320 | +0.259611 | +0.220382 | +0.101632 | +0.153921 |

V8 versus V1 improves composite by+.011557, with increments+.005521/+.002382/+.026769. This is consistent with the combined architecture helping Serting, but small AirHitam/Palong differences are not precise mechanism estimates. The final decision and its explicit post-V checkpoint-guard amendment are in selection.json. The final has all-positive V estate deltas versus incumbent/coords, but cannot be described as all-parent non-degrading or independently V-confirmed superior to s010. s012 remains the conservative Palong-oriented alternative; s010 remains the stronger Serting-V reference.

## Feature signs, conditional suppression and operational errors

| feature | AirHitam | Palong | Serting | same_sign |
| --- | --- | --- | --- | --- |
| NDVI | -0.195021 | -0.310569 | -0.530124 | True |
| deblur_NDRE | -0.347821 | -0.397533 | -0.833507 | True |
| height_contrast6 | +0.276866 | -0.469901 | -0.127941 | False |
| height_mean6 | +0.306670 | -0.558240 | -0.256390 | False |
| height_sd6 | -0.302956 | +0.080652 | -0.337460 | False |
| ndvi_dip | +0.081944 | +0.548937 | +0.698218 | True |
| ndvi_edge | -0.249907 | +0.057896 | +0.117293 | False |
| ndvi_sd20 | -0.253461 | +0.064982 | +0.122584 | False |
| qNDMI | -0.136573 | -0.218290 | -0.718142 | True |
| qNDRE | -0.431710 | -0.604237 | -0.806178 | True |
| qNDVI | -0.195032 | -0.355679 | -0.687019 | True |

ndvi_sd20 and ndvi_edge correlate .947/.970/.983 approximately across estates, yet their marginal signs differ from their negative joint coefficients in Palong/Serting. qNDVI/qNDRE/qNDMI, raw/quality/deblur spectra and spatial interactions are also correlated; deblur moisture/NBR correlation reaches.975. s012's Palong qNDVI encoded coefficient is+.064 despite negative marginal d, while the final combined shared/local standardized qNDVI coefficient is−.055. This is a conditional coefficient redistribution, not proof of a corrected biological pathway. Supplied NDVI remains a core predictor. qNIRv/NDMI contrasts/dip still show conditional suppressions in some estates, and pruning controls do not justify blindly dropping them.

Historical mean-height marginal signs are+.307/−.558/−.256; October SAR/latent axes must not be pooled as universal disease signs. Independent PCA axes are not comparable across estates/folds. feature_effects.csv, feature_correlations.csv, model_coefficients.csv and final_optical_coefficients.csv record the marginal and conditional evidence.

| estate | prevalence | ap | lift | roc | recall25 | enrich10 |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | +0.058091 | +0.133900 | +0.075808 | +0.653929 | +0.482143 | +1.964286 |
| Palong | +0.085714 | +0.495665 | +0.409951 | +0.848765 | +0.742424 | +5.151515 |
| Serting | +0.100904 | +0.479406 | +0.378503 | +0.864772 | +0.761194 | +5.223881 |

| estate | include_middle | budget | inspections | found | healthy | middle | recall | precision | other_per_found |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AirHitam | False | +0.250000 | +241.000000 | +27.000000 | +214.000000 | +0.000000 | +0.482143 | +0.112033 | +7.925926 |
| AirHitam | True | +0.250000 | +241.000000 | +27.000000 | +214.000000 | +0.000000 | +0.482143 | +0.112033 | +7.925926 |
| Palong | False | +0.250000 | +192.500000 | +49.000000 | +143.500000 | +0.000000 | +0.742424 | +0.254545 | +2.928571 |
| Palong | True | +0.250000 | +242.000000 | +47.000000 | +101.000000 | +94.000000 | +0.712121 | +0.194215 | +4.148936 |
| Serting | False | +0.250000 | +166.000000 | +51.000000 | +115.000000 | +0.000000 | +0.761194 | +0.307229 | +2.254902 |
| Serting | True | +0.250000 | +203.750000 | +50.000000 | +95.750000 | +58.000000 | +0.746269 | +0.245399 | +3.075000 |

Middle consumes inspections without being declared Healthy. At25% budget Palong uses94 inspections on Middle; AirHitam costs about7.93 other inspections per detected Unhealthy. Unknown U outcomes preclude measuring actual field precision/recall or labeling low-score unsupported rows false negatives. Palong minimum U score remains about−24.97 under quadratic extrapolation; covariate_support.csv records outside-range features. error_by_positive_distance.csv reports small, descriptive distance bins using one estate-wide top25% allocation, never reranking within each bin.

![Paired lineage performance and inspection capacity](lineage_and_triage.png)

## Integrity, runtime and reproducibility

Final selfcheck and official CLI CV pass (selfcheck.txt, official_cv.txt). The clean single-file package equals the immutable V8 package in **45 paired fits**, maximum absolute difference **0.0**: full U, every official fold, buffered whole-block regions and one reserved repeat. Packaging was not re-scored for fresh V noise. File SHA256 is `71f416f1b15e6649b746f48a1f80e141817f8f01d4efab5403fbb4a2bc12843a`; V8 package content hash is `0ebba5e7a34162954e60a5d6d521693f8681df2376f07cd6960cbf3c9623849f`.

Fresh workers use scratch cwd/4 CPUs and guard network, source-directory reads and writes outside scratch. Shape/dtype/finiteness, immutability, deterministic repetition, shuffled/duplicate targets, dropped Middle, renamed ctx estate, empty target and single-class fallback pass. Full-fit runtimes are at most0.362s, peak observed RSS275.0MiB. Guards are Python-level checks, not a replacement for grading isolation. Only non-base pin is threadpoolctl==3.8.0. Matplotlib3.10.7 is research-only.

All10 feature tables retain7,220 exact keys, zero duplicate keys/NaNs/nonfinite covariates/forbidden feature names/symlinks. Total datasets including preserved sources are about48.92MiB, below128MiB carry-forward and512MiB caps. All4,473 exact-order U scores/ranks are in unlabelled_ranking.csv/.parquet. No predictions are copied into feature tables.

Essential runtime datasets: **triage_v12 and dino_sar**. Preserve their manifests/extraction scripts and recipes/s013. For continued parent auditing, also retain gee_quality, gee_canopy, optical_jitter, alphaearth and dino_optical. compact_triage, highres_canopy and crown_mixture duplicate values already consolidated into triage_v12 and are safe to archive/prune only after confirming no further extraction/provenance audit is needed. Unused SAR views in dino_sar could be compacted into a new exact-key subset in a future packaging step; do not silently mutate the original table because parent recipes use other views. Nothing was deleted here.

Core reproduction (all label use is authorized L):

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords s009 s010 s011 s012 s012_bag pooled_sar --protocols official block120 block160 buffer40 buffer80 blockquad blockquad80 blockquad160
OPENBLAS_NUM_THREADS=1 python /app/research/aggregate.py
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/final_diagnostics.py
OPENBLAS_NUM_THREADS=1 python /app/research/audit.py
python /app/research/write_report.py
```

Every research candidate is preserved under datasets/triage_v12/recipes/s013/candidates; every scored one has its exact portable source. Parents and reproduction scripts are also carried there, avoiding dependence on ephemeral research directories. Restore reproduction into /app/research; evaluate.py falls back to carried parents/candidates if original staging is absent. Cache NPZs include implementation hashes and only authorized OOF predictions/fold indices. Use --force for deliberate reruns; do not mix partial-estate composites. Raw GEE/raster extraction scripts remain in their existing dataset directories; provider reprocessing can alter fresh covariates, so saved table hashes define this reproducible snapshot.

## Next scientific work

1. Prospective80 m-block inspection in all estates, with weak-support/distant Palong, AirHitam misses, Middle costs and a random unranked audit sample. More local model tuning cannot resolve unseen conditional relationships.
2. Independent estates/years, accurately registered finer current crown imagery, and measured outcomes. Native S2 pixels mix palms spaced about9m; frozen embeddings/optical inversion do not create measured crown resolution.
3. Separate source-label information from regularization using additional independent domains. Existing controls establish only limited conditional transfer evidence. Protect the preserved parents and compare future models under the same folds, with footprint-aware buffers.
4. If revisiting the alternative with higher Palong V (s012) or higher Serting V (s010), use new outcomes or a genuinely new mechanism. Do not fit estate-specific ensemble weights to these noisy V observations or rerun hashes for noise.

## Complete official experiment table

Every row uses the same official folds. All per-estate/protocol paired deltas and secondary metrics are in EXPERIMENTS.csv; missing stress cells remain NA. Values below are exploratory, including final ablations after freeze.

| Recipe | AirHitam | Palong | Serting | Composite | Delta incumbent | Delta coords | V composite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_sar_permuted | +0.067683 | +0.398578 | +0.369994 | +0.278752 | +0.138232 | +0.189158 | NA |
| aligned_sar_pool | +0.071524 | +0.379452 | +0.373118 | +0.274698 | +0.134179 | +0.185105 | NA |
| aligned_sar_self | +0.046550 | +0.382620 | +0.373734 | +0.267635 | +0.127115 | +0.178041 | NA |
| both_bag | +0.085463 | +0.428224 | +0.341775 | +0.285154 | +0.144635 | +0.195560 | NA |
| both_offset | +0.070818 | +0.394637 | +0.369257 | +0.278237 | +0.137718 | +0.188644 | +0.201727 |
| compact_mix_pruned | +0.057949 | +0.398750 | +0.369257 | +0.275319 | +0.134800 | +0.185725 | NA |
| compact_quality_drop | +0.077253 | +0.407379 | +0.359427 | +0.281353 | +0.140834 | +0.191760 | NA |
| coords | +0.010793 | +0.179763 | +0.078224 | +0.089593 | -0.050926 | +0.000000 | +0.066462 |
| core_control | +0.060842 | +0.368382 | +0.350406 | +0.259877 | +0.119358 | +0.170284 | NA |
| fusion_offset | +0.065303 | +0.374352 | +0.347506 | +0.262387 | +0.121868 | +0.172794 | NA |
| incumbent | +0.030965 | +0.255637 | +0.134956 | +0.140519 | +0.000000 | +0.050926 | +0.118751 |
| legacy_complement | +0.103290 | +0.426680 | +0.318171 | +0.282713 | +0.142194 | +0.193120 | +0.214160 |
| legacy_protected | +0.098492 | +0.402958 | +0.315759 | +0.272403 | +0.131884 | +0.182810 | +0.213522 |
| legacy_quality | +0.095546 | +0.395927 | +0.314877 | +0.268783 | +0.128264 | +0.179190 | NA |
| legacy_quality_offset | +0.095182 | +0.390715 | +0.317375 | +0.267757 | +0.127238 | +0.178164 | NA |
| legacy_structure | +0.109420 | +0.421588 | +0.321157 | +0.284055 | +0.143536 | +0.194462 | NA |
| pool_fullopt | +0.076162 | +0.392518 | +0.370768 | +0.279816 | +0.139297 | +0.190223 | +0.203185 |
| pool_no_share | +0.066467 | +0.382862 | +0.359326 | +0.269552 | +0.129032 | +0.179958 | NA |
| pool_permuted | +0.068298 | +0.389241 | +0.364268 | +0.273936 | +0.133416 | +0.184342 | NA |
| pool_self | +0.065690 | +0.385990 | +0.367914 | +0.273198 | +0.132679 | +0.183605 | NA |
| pooled_sar | +0.075808 | +0.409951 | +0.378503 | +0.288087 | +0.147568 | +0.198494 | +0.220382 |
| pooled_sar_no_compactopt | +0.065827 | +0.338520 | +0.274256 | +0.226201 | +0.085682 | +0.136608 | NA |
| pooled_sar_noalpha | +0.095660 | +0.373262 | +0.317053 | +0.261992 | +0.121473 | +0.172398 | NA |
| pooled_sar_nocoords | +0.067696 | +0.228800 | +0.386518 | +0.227671 | +0.087152 | +0.138078 | NA |
| pooled_sar_nostructure | +0.066457 | +0.381690 | +0.375754 | +0.274634 | +0.134114 | +0.185040 | NA |
| pooled_sar_permuted | +0.072433 | +0.394216 | +0.374179 | +0.280276 | +0.139757 | +0.190682 | NA |
| pooled_sar_self | +0.058947 | +0.406681 | +0.375817 | +0.280481 | +0.139962 | +0.190888 | NA |
| protected_patch | +0.064458 | +0.383179 | +0.353997 | +0.267211 | +0.126692 | +0.177618 | NA |
| s009 | +0.050556 | +0.288720 | +0.198473 | +0.179250 | +0.038730 | +0.089656 | +0.176790 |
| s010 | +0.105714 | +0.381314 | +0.280387 | +0.255805 | +0.115286 | +0.166211 | +0.216825 |
| s010_curvature | +0.095185 | +0.391621 | +0.320584 | +0.269130 | +0.128611 | +0.179536 | +0.214764 |
| s010_spatial_control | +0.040899 | +0.362128 | +0.276970 | +0.226666 | +0.086147 | +0.137072 | NA |
| s010_uniform | +0.080781 | +0.378694 | +0.319857 | +0.259777 | +0.119258 | +0.170184 | NA |
| s011 | +0.067653 | +0.351565 | +0.337580 | +0.252266 | +0.111747 | +0.162673 | +0.182472 |
| s012 | +0.066466 | +0.382857 | +0.359326 | +0.269549 | +0.129030 | +0.179956 | +0.203411 |
| s012_bag | +0.087399 | +0.386420 | +0.328326 | +0.267382 | +0.126862 | +0.177788 | +0.213993 |
| sar_bag | +0.091858 | +0.417466 | +0.339830 | +0.283051 | +0.142532 | +0.193458 | +0.206329 |
| sar_offset | +0.073393 | +0.389249 | +0.366369 | +0.276337 | +0.135818 | +0.186744 | NA |
| uniform_blend | +0.065728 | +0.364086 | +0.354499 | +0.261438 | +0.120919 | +0.171844 | NA |
| uniform_bothpatch | +0.056786 | +0.421758 | +0.362139 | +0.280228 | +0.139708 | +0.190634 | NA |
| uniform_optpatch | +0.064958 | +0.407007 | +0.359133 | +0.277033 | +0.136513 | +0.187439 | NA |
| uniform_pool | +0.067289 | +0.377692 | +0.362690 | +0.269224 | +0.128705 | +0.179630 | NA |
| uniform_pool_patch | +0.056062 | +0.423076 | +0.364843 | +0.281327 | +0.140808 | +0.191734 | NA |
| uniform_rawalpha | +0.072812 | +0.386095 | +0.362091 | +0.273666 | +0.133147 | +0.184073 | NA |
| uniform_sar | +0.072954 | +0.399129 | +0.370655 | +0.280913 | +0.140393 | +0.191319 | +0.208825 |

## Primary literature and documentation

- [Oquab et al., DINOv2](https://arxiv.org/abs/2304.07193): frozen general-purpose visual representations motivate the carried patch mechanism; no oil-palm pathology validation is inferred.
- [Brown et al., AlphaEarth Foundations](https://arxiv.org/abs/2507.22291): learned geospatial embeddings supply contextual measurements, not physically signed disease indices.
- [Sentinel-2 SR Harmonized catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED): authoritative band/resolution/harmonization provenance; no second+1000 offset is applied.
- [Meta/WRI HighResCanopyHeight](https://github.com/facebookresearch/HighResCanopyHeight): source/method provenance for historical height; historical structure is not current disease confirmation.
- [Izzuddin et al., airborne hyperspectral Ganoderma analysis](https://doi.org/10.21894/jopr.2018.0037): context for canopy-condition screening; no disease-specific accuracy from that paper is transferred to this dataset.
- [Roberts et al., spatial/temporal cross-validation](https://doi.org/10.1111/ecog.02881): blocking/extrapolation rationale. Publisher text was403; Crossref metadata was checked, not an unavailable full-paper analysis.

The web endpoint failed; accessible primary arXiv/catalogue/publisher/repository pages were fetched directly and saved under literature/, with HTTP status provenance. All empirical claims in this report come from the authorized local experiments, not imported paper accuracies.
