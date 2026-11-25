# Oil-palm spatial triage research — session261125-s010-open

The final offline CPU submission is `submission/pipeline.py`: a regularized, partially pooled model combining optical condition and canopy contrast, prior-year AlphaEarth local context, frozen DINOv2 optical patch context, and frozen DINOv2 October ALOS-2 patch context. Official80m CV composite PR-AUC lift is **+0.255805**, versus incumbent **+.140524** and coordinates-only **+.089593**. Noisy validation lift is **+.219732**, with deltas **+.100982** versus incumbent and **+.153270** versus coordinates. All three estates improve; they are equally trusted and equally weighted. This is an in-estate screening model, not a pathogen diagnosis or a demonstrated new-estate transfer system.

Final artifacts include4473 exact aligned unlabelled risk scores and within-estate ranks. Scores are decision values, not calibrated disease probabilities. Incumbent files were not changed. The previous nominee's source was absent, so no byte-identical reconstruction or direct paired comparison to that nominee is claimed. Its carried feature tables were reused with provenance preserved.

## Data and validation

L contains2747 palms,2398 binary-scored and349 Middle. Binary counts are964/770/664, with56/66/67 positives and prevalence.05809/.08571/.10090 for AirHitam/Palong/Serting. U contains4473 rows. All7220 coordinate keys are exact, without rounding. No unlabelled outcomes or sealed splits were accessed. Middle is excluded only from supervised training and metric scoring; inference retains every row. The official folds are formed on the complete per-estate L table, including Middle, before binary scoring.

Our evaluator reproduces CLI groups: local metres relative to the full estate L minimum, floor(x/80)*100000+floor(y/80), GroupKFold(5). Pooled held-out scores within each estate are scored with sklearn step-function, tie-aware average_precision_score, then prevalence is subtracted. Estates are averaged without sample-size weighting. Secondary metrics use fractional inclusion at the boundary score tie. No random-row CV result is used as scientific evidence. The coordinates comparator is distance-weighted15NN on training-standardized Long/Lat, trained only on binary classes.

All supervised transforms and PCA fits in a fold use that fold's training rows. Other estates' full L tables may enter only through the supplied other_labeled context. They are independent estates, not held-out target-estate labels. Two pretrained feature compression PCAs are transductive and label-free: all7220 covariate vectors are used, as allowed by the task. Further8-component PCA and standardization are training-only for every estate/fold. No label-based spatial neighbors, pseudo-labels or same-block out-of-fold features are used. A previous source-only vigor penalty was investigated but not retained.

## Hypotheses and campaign

1. Independent optical quality, illumination, canopy-neighborhood and prior-year embedding context should carry condition information beyond the smooth coordinate prior. Small offset corrections were weaker than jointly estimating the features with the spatial basis.
2. Frozen patch encoders may recover morphology discarded by pixel means without training a high-capacity classifier on189 positives. We successfully completed two GPU extractions: multiscale optical patches and multitemporal calibrated SAR display patches. Only fixed frozen representations enter a small CPU classifier.
3. Partial pooling of physically aligned optical features can share weak signal while preserving estate-specific spatial/environmental responses and sign reversals. Unlike unadapted pooling, all estates retain independent nuisance coefficients and target optical deviations. Controls separate changed regularization from source-label information.
4. Independent terrain and historical canopy structure, late-season optical change, and Middle-as-separate-process multitask learning might offer extra information. These yielded smaller or inconsistent gains and were not retained.
5. Generic spatial-only bases, honest inner complexity selection, extrapolation clipping and a fixed sensor ensemble might improve robustness. None beat the selected primary recipe sufficiently to justify replacement. No finer feature-count, registration-offset, shrinkage-weight or V-gate search followed the final milestone.

## Reproducible final recipe

The baseline block uses equal-weight supplied NDVI and strict-quality qNDVI, plus locally centred longitude/latitude. Each is training-standardized. The inherited incumbent uses a linear basis for AirHitam and degree2 PolynomialFeatures (including spectral/spatial interactions) for the other estates. This explicit name branch remains a limitation: single inner GroupKFold selectors, conservative bootstrap selection, smoothed selection and generic spatial degree2/3 alternatives underperformed. No new estate-specific gate was fitted from V.

Eight physical context columns are log(B4+B8), B3/(B4+B3+1), current canopy NDVI and NDRE local contrast35, and four registration-integrated optical texture columns (NDVI ring dip, SD20, edge magnitude, NDRE med10-minus-med40). Prior-year AlphaEarth2025 local40m contrast vectors are reduced to8 training PCs. Frozen optical NIR-red-green160m and October SAR HH/HV/VV103m patch vectors are each reduced to8 training PCs. Physical/AlphaEarth context and each patch group are independently standardized. All context columns are multiplied by.3; base spatial/spectral columns retain scale1. This implements stronger shrinkage for additional features without a supervised feature-selection sweep.

For each of three estates, the design contains an independent baseline/context block and an estate intercept (indicator scale10). Shared optical columns receive scale.3 and appear for all estates, while independent optical deviations remain in each estate block. Within each estate, Healthy and Unhealthy each receive half its weight; all three estates receive the same total weight, equal to the target training binary count. This treats all labels equally while accommodating small different sample sizes. Fit LogisticRegression(C=.1,max_iter=2000,fit_intercept=False), return the target block's decision_function as float64. Model dimensions are128 columns after SAR augmentation; no GPU head is trained. The final model reads six tables via ctx.datasets_dir and performs no network/raster access or writes during fit_predict. Threads are capped at min(2,ctx.cpus).

## Features, geometry and GPU provenance

Carried optical manifests describe S2_SR_HARMONIZED and CloudScore+ strict vegetation observations, quality composites and fallback windows. No extra +1000 offset is applied to the harmonized collection. Red-edge/SWIR native20m information is not sharpened by sampling at10m. The carried optical_jitter table corrects the documented sampleRectangle centre/corner convention and uniformly averages four±5m registration hypotheses. It was not regenerated or tuned to labels.

Optical chips were reproduced from label-free mosaics. The saved mosaic origin includes pixel-centre coordinates; prepare_gpu.py subtracts5m from the east origin and adds5m to the north origin to recover corners. Patches anchor on the trunk-containing10m pixel. RGB80/160/320m and NIR-red-green160m views were extracted; the final uses NRG160m, fixed reflectance scaling/.5, clipping0..1, bilinear224 resize and ImageNet normalization. A320m patch is not a32cm crown image; upsampling adds no resolution. The encoder primarily supplies canopy-neighborhood/context information.

SAR sampling reprojects original WGS84 coordinates to each EPSG32648 raster. HH/HV/VV bands1/2/4 are converted from linear power to10log10, mapped through fixed[-25,0]/[-35,-5]/[-25,0]dB display ranges and uint8. All crops had100% positive finite source support. June/October51m and103m crops and a fixed display-difference103m view were encoded; the final uses October103m. No Yamaguchi scale assumption enters these features. The RGB-pretrained encoder is out of domain for SAR, so a weak/null view cannot rule out SAR-native pretraining.

Both jobs used official frozen DINOv2 ViT-S/14 CLS384 vectors without labels or fine-tuning. Each view was compressed to64 PCs using all7220 label-free vectors; compression retains roughly94–96% optical variance and has saved transforms. Official weights SHA256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9. Source commit7764ea0f912e53c92e82eb78a2a1631e92725fc8 was main at extraction and is explicitly pinned in reproduction scripts. The final fit sees saved compact features only. GPU jobs completed in50.406s and63.361s script runtime; broker teardown confirms no activeVM and242.558s total charged GPU time. Research raw outputs are available under /run/agentrsi/colab-io/out; dataset reproduction scripts/manifests/transforms carry forward.

Additional feature assets retain independent terrain (Copernicus30m surface model, not bare-earth DEM),2020 ETH10m canopy height, and autumn2026 optical condition/contrasts. Terrain/height and autumn features cover all7220 exact keys with zero NaNs but are not runtime dependencies.

## Official80m experiments

Lift entries are AirHitam/Palong/Serting. All paired per-estate deltas and all repeated-protocol rows are in EXPERIMENT_SUMMARY.csv; the table below reports paired composite deltas. Metrics are exploratory after numerous hypotheses, not prospective estimates.

| Model | Lift A/P/S | Composite | Δ incumbent | Δ coords |
| --- | --- | --- | --- | --- |
| incumbent | +0.03097/+0.25564/+0.13496 | +0.14052 | +0.00000 | +0.05093 |
| coords | +0.01079/+0.17976/+0.07822 | +0.08959 | -0.05093 | +0.00000 |
| ref_quality | +0.02993/+0.23688/+0.14479 | +0.13720 | -0.00332 | +0.04760 |
| ref_texture | +0.03769/+0.25294/+0.18041 | +0.15701 | +0.01649 | +0.06742 |
| joint_radiometry | +0.06907/+0.26029/+0.12850 | +0.15262 | +0.01210 | +0.06303 |
| joint_foundation | +0.02053/+0.28780/+0.18451 | +0.16428 | +0.02376 | +0.07468 |
| joint_joint | +0.09575/+0.33167/+0.22860 | +0.21867 | +0.07815 | +0.12908 |
| joint_raw_joint | +0.11475/+0.34966/+0.21161 | +0.22534 | +0.08482 | +0.13575 |
| dino80 | +0.07870/+0.33411/+0.25710 | +0.22330 | +0.08279 | +0.13371 |
| dino160 | +0.06416/+0.31874/+0.25729 | +0.21340 | +0.07288 | +0.12380 |
| dino320 | +0.09877/+0.30852/+0.25001 | +0.21910 | +0.07858 | +0.12951 |
| dino_nrg | +0.09208/+0.34621/+0.23592 | +0.22474 | +0.08422 | +0.13514 |
| dino_multi | +0.05666/+0.31501/+0.27063 | +0.21410 | +0.07358 | +0.12451 |
| terrain | +0.10033/+0.32284/+0.22031 | +0.21449 | +0.07397 | +0.12490 |
| height | +0.08837/+0.33957/+0.21898 | +0.21564 | +0.07512 | +0.12605 |
| structure | +0.09298/+0.33458/+0.21586 | +0.21448 | +0.07396 | +0.12488 |
| complexity_linear | +0.09575/+0.29429/+0.20631 | +0.19878 | +0.05826 | +0.10919 |
| complexity_quadratic | +0.03782/+0.33167/+0.22860 | +0.19936 | +0.05884 | +0.10977 |
| complexity_bounded | +0.03810/+0.30382/+0.22909 | +0.19034 | +0.04982 | +0.10074 |
| complexity_selector | +0.09575/+0.29528/+0.19188 | +0.19430 | +0.05378 | +0.10471 |
| complexity_conservative | +0.09575/+0.30007/+0.20631 | +0.20071 | +0.06019 | +0.11111 |
| complexity_mixture | +0.07843/+0.32614/+0.21718 | +0.20725 | +0.06673 | +0.11766 |
| transfer_tight | +0.10966/+0.35160/+0.24268 | +0.23464 | +0.09413 | +0.14505 |
| transfer_moderate | +0.12739/+0.37016/+0.25346 | +0.25034 | +0.10982 | +0.16074 |
| transfer_strong | +0.12886/+0.36153/+0.26366 | +0.25135 | +0.11083 | +0.16176 |
| spatial2 | +0.09005/+0.32091/+0.22515 | +0.21204 | +0.07152 | +0.12244 |
| spatial3 | +0.04755/+0.30444/+0.28173 | +0.21124 | +0.07072 | +0.12165 |
| spatial_interact | +0.05888/+0.31435/+0.21361 | +0.19562 | +0.05510 | +0.10602 |
| linear | +0.12955/+0.30054/+0.23311 | +0.22106 | +0.08054 | +0.13147 |
| transfer_dino | +0.12095/+0.37737/+0.25535 | +0.25122 | +0.11070 | +0.16163 |
| transfer_dino80 | +0.08744/+0.35215/+0.28553 | +0.24171 | +0.10119 | +0.15211 |
| transfer_terrain | +0.12997/+0.34031/+0.25341 | +0.24123 | +0.10071 | +0.15164 |
| transfer_height | +0.12206/+0.37248/+0.24535 | +0.24663 | +0.10611 | +0.15704 |
| vigor_source | +0.09575/+0.33167/+0.22808 | +0.21850 | +0.07798 | +0.12891 |
| vigor_target | +0.09569/+0.33121/+0.22861 | +0.21850 | +0.07798 | +0.12891 |
| ordinal_weak | +0.09575/+0.33739/+0.23380 | +0.22231 | +0.08179 | +0.13272 |
| ordinal_moderate | +0.09575/+0.33921/+0.24363 | +0.22620 | +0.08568 | +0.13661 |
| autumn_level | +0.11587/+0.36535/+0.23917 | +0.24013 | +0.09961 | +0.15054 |
| autumn_change | +0.09807/+0.37155/+0.25901 | +0.24288 | +0.10236 | +0.15328 |
| no_coords | +0.06454/+0.26406/+0.23612 | +0.18824 | +0.04772 | +0.09864 |
| no_foundation | +0.12824/+0.32933/+0.20652 | +0.22136 | +0.08084 | +0.13177 |
| pool_no_share | +0.09561/+0.33157/+0.22852 | +0.21857 | +0.07805 | +0.12897 |
| pool_self_duplicate | +0.10052/+0.34012/+0.24208 | +0.22757 | +0.08705 | +0.13798 |
| pool_permuted_source | +0.11521/+0.32894/+0.24370 | +0.22928 | +0.08876 | +0.13969 |
| sar_june | +0.06872/+0.34644/+0.24575 | +0.22030 | +0.07978 | +0.13071 |
| sar_october | +0.10571/+0.38131/+0.28039 | +0.25580 | +0.11529 | +0.16621 |
| sar_change | +0.07807/+0.36686/+0.27345 | +0.23946 | +0.09894 | +0.14987 |
| sar_pair | +0.12716/+0.38230/+0.25776 | +0.25574 | +0.11522 | +0.16615 |
| sensor_ensemble | +0.11022/+0.37343/+0.27002 | +0.25122 | +0.11070 | +0.16163 |
| final_no_coords | +0.06636/+0.27235/+0.25640 | +0.19837 | +0.05785 | +0.10878 |
| final_no_foundation | +0.10558/+0.33061/+0.22996 | +0.22205 | +0.08153 | +0.13246 |
| final_no_optical_patch | +0.10583/+0.36880/+0.28407 | +0.25290 | +0.11238 | +0.16331 |
| final_no_quality | +0.10762/+0.36430/+0.25803 | +0.24331 | +0.10280 | +0.15372 |

The joint reference improves markedly over simple offset corrections. DINO optical NRG alone adds most reliably in Serting. Partial pooling improves all three estates on the official split; tight/moderate/strong variants saturate, so no finer prior-weight search was run. At the optical-only checkpoint, removing coordinates lowers composite lift to+.18824 and removing AlphaEarth to+.22136. Final-recipe ablations give+.19837 without coordinates,+.22205 without AlphaEarth,+.25290 without the optical patch group,and+.24331 with raw NDVI replacing the quality blend. Thus the final recipe exceeds its coordinates-only comparator and retains substantial skill without explicit coordinates, but imagery can still encode spatial/environmental context. The optical patch group's final incremental L gain is small (+.00290 composite); it should not be called independently established disease morphology. No extra V query was spent tuning ablation dimensions. Annual context and physical optical condition contribute more clearly; association may reflect management/environmental confounding.

Controls use the same optical-only pooling design: no shared columns+.21857, a target-only duplicated optical block+.22757, permuted source labels with correspondingly permuted class weights+0.22928, real source labels+.25034. Some gain is changed regularization, so the whole gain must not be labelled transferable disease physiology. Across11 assignments, real pooling especially helps Palong; AirHitam gains over the matched controls are small. One fixed source-label permutation is a negative control, not a permutation-test p-value.

Middle multitask sharing yields+.22620 versus joint+.21867, but trails proper transfer. It treats Middle as a separate observation process, not weak disease positives; the final ignores Middle labels. Autumn optical level and change yield+.24013/+.24288 versus combined optical+.25122 and are rejected. June, change-display and paired SAR views are inconsistent and lose on broader checks. October SAR has small mean repeat gains in AirHitam/Palong and larger Serting gain; this adds new patch structure rather than rerunning raw SAR-means sweeps. Equal sensor logit averaging gives+.25122 and V+.21581, below October SAR's primary lift; no ensemble-weight tuning was performed.

## Noisy held-out V milestones

All six submissions were valid. Gaussian noiseσ=.005 is cached per submission hash; do not interpret tiny differences as precise measurements or resubmit cosmetic hashes. The final has three positive estate deltas versus both comparators. AirHitam's final noisy incumbent delta+.01729 is much smaller than its L-CV improvement, underscoring transport uncertainty. V informed milestone selection; it is not an untouched confirmatory test. Nine of15 queries remain.

| Model | Composite | Δ incumbent | Δ coords | Lift A/P/S |
| --- | --- | --- | --- | --- |
| joint | +0.17172 | +0.05297 | +0.10526 | +0.04215/+0.30553/+0.16749 |
| dino | +0.18344 | +0.06469 | +0.11698 | +0.03513/+0.27925/+0.23595 |
| transfer | +0.18840 | +0.06964 | +0.12193 | +0.04850/+0.31757/+0.19912 |
| transfer_dino | +0.20971 | +0.09096 | +0.14325 | +0.05296/+0.30098/+0.27520 |
| sar_october | +0.21973 | +0.10098 | +0.15327 | +0.05432/+0.29121/+0.31367 |
| sensor_ensemble | +0.21581 | +0.09706 | +0.14935 | +0.06280/+0.28836/+0.29628 |

## Spatial consistency and uncertainty

The selected model improves all three estates over incumbent and coordinates in all21 assignments (official0, exploration1–10, frozen100–109). Repeated assignments share labels, so they are not21 independent trials. Frozen assignments were declared after the optical feature checkpoint, before inspecting their outcomes; SAR confirmation used the same declared seeds. They support stability but do not remove exploration/selection bias. Final means/minima are:

| Assignments | Estate | Mean lift | Mean Δ incumbent | Minimum Δ incumbent | Minimum Δ coords | Positive Δ incumbent |
| --- | --- | --- | --- | --- | --- | --- |
| exploration seeds0–10 | AirHitam | +0.11233 | +0.08751 | +0.05078 | +0.03132 | 11/11 |
| exploration seeds0–10 | Palong | +0.33244 | +0.08317 | +0.05318 | +0.13754 | 11/11 |
| exploration seeds0–10 | Serting | +0.24040 | +0.11800 | +0.07701 | +0.15610 | 11/11 |
| frozen seeds100–109 | AirHitam | +0.13057 | +0.10627 | +0.08734 | +0.06892 | 10/10 |
| frozen seeds100–109 | Palong | +0.33380 | +0.08678 | +0.06372 | +0.06409 | 10/10 |
| frozen seeds100–109 | Serting | +0.25460 | +0.10241 | +0.05789 | +0.11980 | 10/10 |

Farther-field tests use120/160m blocks, a40m shifted80m grid, a20m exclusion buffer to validation rows, and four contiguous median-coordinate quadrants. All improve all estates against the incumbent. Comparator exceptions, if any, are visible in the exact table:

| Protocol | Lift A/P/S | Δ incumbent A/P/S | Δ coords A/P/S |
| --- | --- | --- | --- |
| block, 80m, shift0, buffer20 | +0.12319/+0.30304/+0.20876 | +0.08477/+0.04274/+0.06912 | +0.08961/+0.25446/+0.21489 |
| block, 80m, shift40, buffer0 | +0.12019/+0.32184/+0.23789 | +0.09190/+0.07602/+0.08928 | +0.09869/+0.27019/+0.10708 |
| block, 120m, shift0, buffer0 | +0.13077/+0.38853/+0.20805 | +0.11412/+0.13640/+0.09027 | +0.04487/+0.19891/+0.20892 |
| block, 160m, shift0, buffer0 | +0.08637/+0.29541/+0.23028 | +0.06738/+0.07128/+0.12963 | +0.08450/+0.17516/+0.15497 |
| region, 80m, shift0, buffer0 | +0.08948/+0.39484/+0.32887 | +0.05833/+0.14745/+0.21762 | +0.09735/+0.25528/+0.29418 |

Paired2000-resample block bootstrap reuses the same held-out80m blocks for model/comparators, independently resamples blocks inside each estate, and averages the three estate deltas. These intervals condition on the selected recipe and observed L data; they are not formal post-selection or future-estate coverage guarantees.

| Estate | Δ incumbent 95% interval | Δ coords 95% interval |
| --- | --- | --- |
| AirHitam | +0.00736 / +0.16371 | +0.02838 / +0.18068 |
| Palong | +0.00138 / +0.28070 | +0.02517 / +0.30810 |
| Serting | +0.07199 / +0.28423 | +0.09013 / +0.34584 |
| composite | +0.05898 / +0.18388 | +0.09321 / +0.23003 |

## Inspection utility and errors

| estate | prevalence | ap | lift | recall25 | enrichment10 | roc |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | 0.0581 | 0.1638 | 0.1057 | 0.5000 | 2.6786 | 0.6822 |
| Palong | 0.0857 | 0.4670 | 0.3813 | 0.7197 | 5.6061 | 0.8555 |
| Serting | 0.1009 | 0.3813 | 0.2804 | 0.6716 | 4.7761 | 0.8092 |

Binary official Recall25 is50.0%/72.0%/67.2%. AirHitam therefore does not establish majority recall at25% even on the official point estimate. Top30% is a separate operating budget, not a tuned score threshold or a claim that field performance will match CV. Middle trees consume inspection capacity even when unscored; the diagnostic below includes them as non-Unhealthy capacity consumers rather than declaring them verified negatives.

| estate | capacity | fraction | inspection_count | recall | precision | Other inspections / Unhealthy found |
| --- | --- | --- | --- | --- | --- | --- |
| AirHitam | binary | 0.25 | 241.000 | 0.500 | 0.116 | 7.607 |
| AirHitam | including_Middle | 0.25 | 241.000 | 0.500 | 0.116 | 7.607 |
| AirHitam | binary | 0.3 | 289.200 | 0.554 | 0.107 | 8.329 |
| AirHitam | including_Middle | 0.3 | 289.200 | 0.554 | 0.107 | 8.329 |
| Palong | binary | 0.25 | 192.500 | 0.720 | 0.247 | 3.053 |
| Palong | including_Middle | 0.25 | 242.000 | 0.636 | 0.174 | 4.762 |
| Palong | binary | 0.3 | 231.000 | 0.788 | 0.225 | 3.442 |
| Palong | including_Middle | 0.3 | 290.400 | 0.688 | 0.156 | 5.396 |
| Serting | binary | 0.25 | 166.000 | 0.672 | 0.271 | 2.689 |
| Serting | including_Middle | 0.25 | 203.750 | 0.672 | 0.221 | 3.528 |
| Serting | binary | 0.3 | 199.200 | 0.746 | 0.251 | 2.984 |
| Serting | including_Middle | 0.3 | 244.500 | 0.701 | 0.192 | 4.202 |

At6–10% prevalence, even useful enrichment leaves substantial inspection overhead, especially AirHitam. Consult inspection_curves.png/.pdf for full fixed-budget curves and the table for expected overhead; no accuracy metric is used. At30% review, including Middle capacity, recall is55.4%/68.8%/70.1% on this labelled sample. This is an observed operating point, not a field guarantee or an optimized threshold. error_by_distance.csv stratifies held-out errors by nearest training-positive distance; its generic recall25 reranks within each stratum, while global_top25_recall uses one estate-wide25% allocation. Far-distance bins contain few positives and have wide sampling uncertainty. covariate_support.csv records U values outside binary training NDVI ranges. Unsupported Palong spectral tails can still produce large negative polynomial scores. Those rows' labels are unknown; they must not be called false negatives. Rank scores are not calibrated probabilities and should not set a universal disease threshold.

feature_effects.csv and feature_sign_consistency.csv report signed Unhealthy-minus-Healthy Cohen's d and coordinate correlations. NDVI/NDRE local contrasts and NDRE texture contrast have consistently negative effects across estates, and the ring dip consistently positive. Brightness, green/red fraction, texture SD/edges, height and terrain derivatives often reverse sign. Independent estate deviations are therefore retained; no universal pooled disease sign is imposed. PCA-axis signs depend on orientation and are not biological directions. Correlations and feature ablations identify predictive association, not pathogen-specific mechanisms. Labels supply screening status only; no pathogen identity, infection time or causal treatment response is available.

## Packaging, reproduction and limits

The final single-file submission is prediction-equivalent to /shared/s010_sar_october: full U and additional held-out-row tests in all estates have maximum absolute score difference0. Clean packaging was not re-queried for new noise. Final officialCV, selfcheck and fresh contract checks are recorded in research/final_cv.log, final_selfcheck.log, contract.log and report/contract_verification.json. Shape,float64,finiteness,input immutability,determinism,shuffled-row alignment and duplicate-row behavior pass. AirHitam peak 327MiB; Palong peak 326MiB; Serting peak 327MiB. These measurements are development-container evidence; isolated-runtime constraints remain300s/4GiB per estate. Only non-base runtime import is threadpoolctl==3.8.0, explicitly pinned. Six runtime datasets are alphaearth,gee_quality,gee_canopy,optical_jitter,dino_optical,dino_sar. Total carried datasets=38.17MiB, below128MiB; every table has7220 unique exact keys, no forbidden columns and no symlinks. Final source imports no raster/GEE/GPU library and reads no development data paths.

Reproduction commands:

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/common.py /app/submission/pipeline.py packaged
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/final_diagnostics.py sar_october
OPENBLAS_NUM_THREADS=1 python /app/research/audit.py
OPENBLAS_NUM_THREADS=1 python /app/research/summarize.py
python /app/research/write_report.py
python /app/research/plots.py
```

Research scripts use source coordinate tables for feature extraction and environment-provided GEE authentication without writing credentials. Each new dataset directory contains its exact extraction/reproduction scripts and manifest. GPU output job paths in compression scripts are session-specific; replace only those paths when reproducing broker jobs. Public source imagery/catalogues may change; saved feature hashes, model source revision, weight hash and PCA transforms pin this submission's inputs. S2 quality/window differences are confounded; no gain is attributed uniquely to a mask. Repeated estate/year or prospectively labelled spatial blocks, particularly AirHitam and Palong, remain the strongest next scientific check. New-season/finer crown imagery, optical registration ground control, and SAR-native encoders are ranked research leads; another minor tabular/registration/PCA-dimension sweep is not justified.

## Literature and primary documentation

- Oquab et al., DINOv2: Learning Robust Visual Features without Supervision. Frozen all-purpose representations motivate the patch experiment; the paper does not validate oil-palm disease detection. https://arxiv.org/abs/2304.07193
- Brown et al., AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. Embeddings encode multisource context, not physical spectral bands. https://arxiv.org/abs/2507.22291 ; official annual dataset https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Lang et al., A high-resolution canopy height model of the Earth. Historical canopy height is an estimated structural covariate, not direct disease evidence. https://arxiv.org/abs/2204.08322 ; source/methods https://langnico.github.io/globalcanopyheight/
- Official Sentinel-2 SR Harmonized catalogue documents baseline harmonization, band resolutions and SCL. https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- Official Copernicus DEM catalogue describes a digital surface model and dataset version caveats. The extracted source is the carried GLO30 version, now superseded; it is not runtime-required. https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30
- Scikit-learn AP documentation defines the non-interpolated metric, and grouped CV documentation explains group separation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html ; https://scikit-learn.org/stable/modules/cross_validation.html

Primary pages were fetched directly and saved as literature_*.html after the provided web search endpoint failed. No unverified quotation or pathogen-causality claim is used.
