# Oil-palm triage research — 261125-s008-open

## Nominee and evidence

`/app/submission/pipeline.py` nominates **quality-controlled NDVI plus a conservative local-vigor penalty**. Official 80m GroupKFold lift is **+0.04228/+0.27044/+0.16971; composite +0.16081**, versus incumbent +0.14052 and distance-weighted15-NN +0.08959. Paired official composite gains are +0.02029 and +0.07121 respectively. All estate labels are equally verified/trusted; no estate is discounted.

The frozen candidate's noisy V composite is **+0.151519**, delta **+0.032769** vs incumbent and **+0.085058** vs coordinates. V estate deltas are **−0.001335/+0.015262/+0.084380**. AirHitam is approximately unchanged at noise scale (sigma .005), not proven equivalent. This is the first session candidate combining substantial V composite improvement, two estate improvements and no observed material third-estate loss. Hash `845096bf8402b7ddf7c955f746606f7e16f2e1d6a980f38b31a3ea66873df9d2` includes staged submission/data content; final code is the frozen quality_veto.py. V noise is cached per content hash, and no cosmetic reruns were used.

Mean composite improvement across official plus 10 repeated80m assignments is +0.02032. Whole-block bootstrap of that *mean of assignment-specific AP differences* gives95% **[+0.00224,+0.04629]**, P(positive)=.9842; the same sampled blocks are used across repeats, so repeats are not extra independent samples. Official-fold-only interval is **[−0.00472,+0.05794]**, crossing zero. These intervals condition on explored, fitted models; they exclude model-selection uncertainty and are not prospective proof. Per-estate intervals remain wide, especially Palong; criterion1 has supportive repeated-assignment evidence, not unconditional statistical confirmation.

## Data and validation discipline

L has 2,747 labelled trees, including 349 visible Middle rows. Binary scoring uses 2,398 Healthy/Unhealthy rows: AirHitam 964/56 positive; Palong 770/66; Serting 664/67. Target has 4,473 rows, all retained. Source-derived covariates cover all 7,220 exact `(estate,Long,Lat)` keys without rounding or duplicate keys. No target labels or sealed splits were accessed. `/tests` and the prohibited repository were not accessed.

Official folds reproduce CLI GroupKFold5 on 80 m local blocks, origin at estate L min coordinates, x scaling111320cos(mean latitude), y110540. Folds are formed **before excluding Middle**, matching CLI. AP is sklearn tie-aware step-function average precision, prevalence subtracted, macro estate mean. Random tree folds are never used. Repeated assignments shuffle whole blocks with seeds 0–9; region splits use four contiguous median-x/y quadrants. Extra shifted-grid, buffered and120m checks are below. Source-estate baseline residuals are also whole-block cross-fitted. Train/target row coordinates may share image pixels; label-free overlapping imagery is not label leakage, but effective sample sizes are spatial.

## Final algorithm

1. Load `gee_quality/features.parquet` and `gee_canopy/features.parquet` through ctx datasets_dir. Join on exact keys in incoming row order.
2. Replace NDVI by the equal blend of supplied NDVI and strict-quality per-image NDVI75th percentile. Quality data use S2 SR HARMONIZED, April 15–August 15 2026, SCL4 and CloudScore+ cs_cdf>=.65. A wider SCL4 fallback is available. This changes temporal window, cloud/shadow filtering and index aggregation simultaneously; no single cause is isolated. Native red-edge/SWIR remain20m. No extra +1000 offset is applied to harmonised DN.
3. Fit balanced logistic regression C=.1 on standardized blended NDVI, longitude and latitude. Retain incumbent complexity: linear for AirHitam, degree2 for Palong/Serting. **This inherited name branch is a limitation**: multiple data-driven inner block selectors were tested but lowered repeated/region performance; no new estate-specific gates were added from V. Unseen estates default to quadratic and require validation.
4. Local NDRE contrast is10m circular focal median minus35m focal median, from May 15–July 15 S2 composite, standardized separately using each estate's binary labelled training rows and clipped[-3,3]. It measures condition relative to local stand background.
5. On the **other two estates**, fit their quality NDVI baselines out of fold on 4 block folds. Keep these cross-fitted logits as offsets, and learn one shared standardized NDRE contrast coefficient plus one source intercept each by weighted logistic likelihood. Each source has balanced class mass; coefficient penalty 25, intercept penalty 1. No target labels enter the transferred coefficient, but target labels fit its baseline. All three source coefficients are negative when each estate is held out (−.5313/−.4755/−.5012). Quality-p75 NDVI class-effect sizes are−.294/−.357/−.697 versus original NDVI−.195/−.311/−.530; these marginal associations are descriptive and not causal.
6. Final risk = target baseline logit + min(coefficient × clipped standardized contrast,0). Locally vigorous crowns can be downweighted. Local contrast alone cannot create a new high-risk entry. Scores are uncalibrated ranking values, not disease probabilities. No Middle fitting, pseudo-labels or online network calls.

All source features and parameters are deterministic. CPU-only fitting is about 0.1–0.2s per estate; fresh guarded workers peak around225MiB. `threadpoolctl==3.8.0` is the sole non-base direct import and is pinned. GPU/EE/rasterio are extraction-only and absent from runtime imports.

## Multi-hypothesis campaign

Prior MainChecklist and handoff were read first. Original incumbent and empty report checkpoint were installed immediately. The previous nominee's code was not carried over, so it could only be reconstructed approximately; do not claim exact reproduction of its reported .16608. The new optical-average baseline scored .14861 locally and failed V. Derived-dataset V staging now works; previous read-only errors did not recur.

Four new label-free feature tables were built: (i)22-channel two-date SAR15x15 crop VICReg and masked-denoising embeddings; (ii)current/prior canopy-pixel and local optical background contrasts; (iii)independent2020 canopy height/uncertainty; (iv)CloudScore+-filtered optical quantiles. T4 trained both encoders without labels,121/101 epochs;32 PCs per128-dimensional representation retained99.16%/98.38% variance. GPU VM was stopped;173.3 GPU-accounted seconds. Features are reusable and fit_predict remains CPU/offline.

Additional orthogonal directions included partial pooling with estate-specific spatial nuisance terms, source-only spectral offsets, transductive graph residual diffusion, target density ratios, block balancing/bagging, cluster-sandwich uncertainty, asymmetric condition safeguards and block-honest measurement/complexity selection. Local model selection never used random-fold neighbours. Broad SAR means/change sweeps from prior sessions were not repeated.

| Experiment | AirHitam / Palong / Serting | Composite | Δ incumbent | Δ coords15 | Verdict |
|---|---|---|---|---|---|
| Incumbent | +0.03097/+0.25564/+0.13496 | +0.14052 | +0.00000 | +0.05093 | Reference |
| Coordinates-only 15-NN | +0.01079/+0.17976/+0.07822 | +0.08959 | -0.05093 | +0.00000 | Required spatial comparator |
| Reconstructed Bayesian optical averaging | +0.03929/+0.25919/+0.14734 | +0.14861 | +0.00809 | +0.05901 | V lost AirHitam |
| VICReg SAR patch, 8 PCs × .3 | +0.04108/+0.29775/+0.13411 | +0.15765 | +0.01713 | +0.06805 | Serting repeat/region loss |
| Masked-AE SAR patch,32 PCs × .3 | +0.04065/+0.23578/+0.16232 | +0.14625 | +0.00573 | +0.05665 | Palong repeat loss |
| Shared NDVI slopes, estate spatial terms | +0.03243/+0.25026/+0.14523 | +0.14264 | +0.00212 | +0.05305 | Small/unstable |
| Hierarchical red-edge physiology | +0.03157/+0.24578/+0.16096 | +0.14610 | +0.00558 | +0.05651 | Palong losses |
| Independent 2020 canopy structure | +0.02306/+0.27837/+0.13611 | +0.14585 | +0.00533 | +0.05625 | Repeat degradation |
| Inverse block sample-size weights | +0.02664/+0.24822/+0.13140 | +0.13542 | -0.00510 | +0.04583 | Reject |
| Target/train spatial density ratio | +0.02815/+0.22876/+0.14321 | +0.13337 | -0.00715 | +0.04378 | Reject |
| Transductive canopy graph residuals | +0.02979/+0.26007/+0.13422 | +0.14136 | +0.00084 | +0.05177 | Small/unstable |
| Cluster-sandwich predictive marginalisation | +0.03239/+0.25547/+0.13312 | +0.14033 | -0.00019 | +0.05073 | No consistent gain |
| Highest-NDVI canopy-pixel proxy | +0.05382/+0.26829/+0.15620 | +0.15944 | +0.01892 | +0.06985 | Palong/Serting repeat losses |
| Weak local-background contrasts | +0.03671/+0.26077/+0.13697 | +0.14482 | +0.00430 | +0.05522 | Local consistent; V AirHitam loss |
| Source-only local red-edge offset | +0.05420/+0.25800/+0.16219 | +0.15813 | +0.01761 | +0.06854 | V composite +.02472, AirHitam loss |
| Clipped source-only red-edge offset | +0.05538/+0.26388/+0.16315 | +0.16080 | +0.02028 | +0.07121 | Positive local CI; V AirHitam loss |
| Source-only VICReg SAR offset | +0.03585/+0.27377/+0.13725 | +0.14896 | +0.00844 | +0.05936 | V lost all estates |
| Joint source optical + SAR offsets | +0.05021/+0.28344/+0.15576 | +0.16314 | +0.02262 | +0.07354 | Local robust; V AirHitam loss |
| Strict-quality median NDVI replacement | +0.02689/+0.20734/+0.12367 | +0.11930 | -0.02122 | +0.02971 | Reject |
| Half original / half quality-p75 NDVI | +0.03408/+0.26717/+0.16230 | +0.15452 | +0.01400 | +0.06492 | V all three improve |
| Inner block Bayesian measurement/degree selection | +0.03552/+0.25650/+0.14791 | +0.14664 | +0.00613 | +0.05705 | Instability; no V |
| Inner block hard degree selection | +0.03443/+0.26717/+0.15848 | +0.15336 | +0.01284 | +0.06377 | Region Palong/Serting loss |
| Quality NDVI + symmetric .3 red-edge offset | +0.04571/+0.26052/+0.17037 | +0.15887 | +0.01835 | +0.06927 | V AirHitam loss |
| Final: quality NDVI + local-vigor penalty | +0.04228/+0.27044/+0.16971 | +0.16081 | +0.02029 | +0.07121 | Nominee |

Values are official OOF lifts. Full per-assignment values, operating metrics and paired deltas vs *both* references are in EXPERIMENTS.csv; means/worst deltas/wins in EXPERIMENT_SUMMARY.csv. Experimental tuning and multiple comparisons are substantial. V queries were spent on distinct L-validated milestones rather than each local configuration.

## Blind V milestones

| Candidate | V composite | Δ incumbent | Estate Δ A/P/S |
|---|---|---|---|
| contrast_v | +0.114378 | -0.004372 | -0.02364/+0.00570/+0.00483 |
| transfer_v | +0.125795 | +0.007044 | -0.00816/+0.00743/+0.02187 |
| transfer_full_v | +0.143472 | +0.024722 | -0.01227/+0.02343/+0.06300 |
| patch_transfer_v | +0.105299 | -0.013452 | -0.01961/-0.00968/-0.01107 |
| joint_v | +0.131473 | +0.012723 | -0.01120/+0.01553/+0.03383 |
| robust_v | +0.142614 | +0.023864 | -0.01267/+0.02886/+0.05540 |
| quality_v | +0.141113 | +0.022363 | +0.00682/+0.00344/+0.05683 |
| quality_local_v | +0.140780 | +0.022029 | -0.00977/+0.00276/+0.07310 |
| quality_veto_v | +0.151519 | +0.032769 | -0.00133/+0.01526/+0.08438 |

The unchanged incumbent audit also returned+.122473, apparent+.003723 from **noise only**; it is not a model gain. Original incumbent reference is+.118751. The first optical-average V result was+.116106 (delta−.002645), rejected. Eleven evaluations were used; four remained at checkpoint. Do not optimize cosmetic hashes to sample noise. Full structured results are in verifier_results.json; optical_average's first result is in VERIFICATION_NOTES.md and initial logs.

## Spatial robustness and ablations

| Protocol | Final lifts A/P/S | Δ incumbent A/P/S | Composite Δ |
|---|---|---|---|
| Four adjacent regions | +0.04184/+0.25181/+0.15770 | +0.01069/+0.00442/+0.04646 | +0.02052 |
| 80m grid origin shifted40m | +0.04313/+0.25825/+0.16457 | +0.01483/+0.01243/+0.01596 | +0.01441 |
| 80m folds +20m train buffer | +0.05167/+0.26966/+0.16558 | +0.01325/+0.00935/+0.02594 | +0.01618 |
| 120m blocks | +0.02648/+0.27829/+0.14112 | +0.00983/+0.02615/+0.02334 | +0.01977 |

All 12 estate×extra-protocol comparisons improve against incumbent. After freezing, ten additional block assignments (seeds 100–109) also improved composite lift10/10: mean+.01755, range+.00502 to+.03169. Estate mean deltas were+.01200/+.00436/+.03629, with 10/10,6/10,10/10 wins. Their same-block bootstrap composite95%[−.00188,+.04535] crosses zero, reinforcing uncertainty; AirHitam/Serting individual intervals were positive. These reuse L labels and are not an independent dataset. AirHitam mean gain over15-NN across these confirmation assignments was approximately zero (−.00025), limiting claims of disease-specific improvement for that estate. The candidate has a spatial component, but gains exceed pure spatial smoothing: required15-NN official lifts are+.01079/+.17976/+.07822, and same-recipe coordinates-only composite is+.07723. Removing coordinates leaves+.06236 composite, confirming spectral information but also a large location dependence. Removing new contrast yields+.15452; removing quality restores the older-NDVI plus vigor penalty. See ablations.log for full values. These are in-estate screening gains, not a transferable pathogen detector. A spectral cue can encode stand condition/environment and disease jointly.

Across11 assignments, mean estate deltas vs incumbent are+.01278/+.01045/+.03774. AirHitam and Serting improve11/11; Palong9/11 with worst−.01103. Far-region comparisons improve each estate by+.01069/+.00442/+.04646. Final remains uncertain in new disease pockets and beyond sampled geometry. error_by_distance.csv uses only each outer fold's training positives for distance diagnostics.

## Features, error patterns and cross-estate signs

Local NDRE contrast Cohen d is **−.485/−.778/−.701**, NDMI contrast−.228/−.668/−.531, NDVI contrast−.182/−.543/−.477. Their consistent stress direction motivated partial pooling. GRVI contrast flips +.261/−.158/−.159 and was not used in the final offset. Historical canopy height flips −.111/−.517/+.263; it is a stand-age/confounding signal, not a universal health marker. Patch latent dimensions also flip signs and their local Palong gains did not transfer to V. Separate coefficient/effect CSVs and feature_signs.png document this.

Final source-only local stress slopes agree in sign, but their Hessian standard errors assume independent weighted samples and **must not** be used as spatial significance tests. We use block-resampled OOF differences instead. Class labels refer to verified health screening, not lab-confirmed Ganoderma specifically; no causal interpretation is claimed.

## Inspection budget and practical limits

| Estate | PR lift | Recall@25% | Enrichment@10% | ROC AUC |
|---|---|---|---|---|
| AirHitam | +0.04228 | 50.0% | 2.32× | 0.675 |
| Palong | +0.27044 | 81.8% | 4.09× | 0.839 |
| Serting | +0.16971 | 65.7% | 3.49× | 0.767 |

At25% binary-scored inspection, estimated false inspections per Unhealthy found are7.61/2.56/2.77. Palong's top10% enrichment falls4.55→4.09 despite better AP and Recall25%; metrics must be considered together. AirHitam exactly50% recall is **not majority capture** and corresponds to just28/56 positives; one tree changes recall1.8 percentage points.

When visible Middle consumes inspection capacity but remains unscored as a disease class, Recall25 becomes50.0%/74.2%/64.2%, precision11.6%/20.2%/21.1%. These are L-only diagnostics; unknown deployment Middle can dilute lists differently. No fixed25% majority-recall guarantee is justified. All4473 target scores, original row order and estate-relative ranks are in unlabelled_ranking.csv/.parquet; top25 flags are illustrative, no hard disease decisions. A larger budget improves capture but is an operational choice, not model progress.

![Inspection budget](inspection_budget.png)
![Repeated paired deltas](repeated_deltas.png)
![Cross-estate feature signs](feature_signs.png)

## Reproduction and execution audit

```bash
agentrsi gcp-init
# Only needed to regenerate covariates, with authorised GEE credentials:
python /app/datasets/gee_canopy/extract.py
python /app/datasets/gee_quality/extract.py
# CPU/offline final validation:
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/common.py /app/submission/pipeline.py quality_veto --repeat 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/robust_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/bootstrap.py quality_veto incumbent 5000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/bootstrap_repeated.py quality_veto incumbent 5000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/verify_contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/predict.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/diagnostics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/plots.py
python /app/research/write_report.py
```

selfcheck and official CV pass. Fresh per-estate workers with network disabled, scratch-only write guards, unavailable-source read guards, shuffled target rows, duplicate target rows and input immutability checks pass. These are Python guards, not an OS sandbox. Source artifacts are hash-unchanged. Runtime imports use only base packages plus the exact threadpoolctl pin; evaluation versions match development for base packages. Peak worker memory~225MiB; three fit calls per estate<.31s. Frozen source SHA256: `1ec85cf1acaacbe23301855fd489d7013b3a16412fbbeeff7986a932d7a320d5`.

All 9 derived tables meet exact key/forbidden-column/no-symlink audits, total~68.1MiB below 128MiB carry-forward threshold. Carried alos_winstats T3 NaNs remain documented and unused; carried gee_topo has harmless temporary `row` covariate never modelled. New four tables have zero NaNs. Requirements and extraction versions are documented separately; torch training was remote and not a runtime dependency. No secrets are stored in artifacts.

## Literature and interpretation

- [Google S2 SR HARMONIZED documentation](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) specifies harmonised reflectance and native band resolutions. It supports the offset/resolution handling, not disease prediction claims.
- [Google Cloud Score+ documentation](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) describes clear-sky usability scores and linking to Sentinel-2. The threshold and quantile used here are local research choices.
- [Bardes et al., VICReg](https://arxiv.org/abs/2105.04906) motivates label-free invariance, variance and covariance regularisation. Its method does not imply that learned SAR features identify disease; our V test rejected their addition.
- [Lang et al., canopy height](https://doi.org/10.1038/s41559-023-02206-6) supplies independent historical structure. Six-year-old learned height is not current per-palm ground truth.
- [Saito and Rehmsmeier, PR evaluation](https://doi.org/10.1371/journal.pone.0118432) supports precision-recall reporting for imbalanced detection. Our primary AP and budget diagnostics directly reflect inspection costs.
- [Izzuddin et al., drone hyperspectral OBIA](https://doi.org/10.21894/jopr.2024.0025) explores canopy object segmentation for Ganoderma severity. Its drone/hyperspectral setting differs from native10–20m mixed pixels here; its classification results cannot establish our spatial generalisation. It motivates genuinely finer canopy geometry as a future data direction.

Public primary pages were retrieved using requests after web tool returned HTTP500; source snapshots/metadata are under research. No literature accuracy values are used as benchmark claims.

## Caveats and next evidence

Only189 binary positives and102 L blocks underpin the estimates (50/39/13 blocks including binary support). Repeated folds recycle labels, V was inspected11 times, and tuning was exploratory. The final significant repeated-bootstrap interval should be treated cautiously. AirHitam V equivalence is not established. Native pixels mix adjacent9m palms; median composites and canopy proxies are not isolated crowns. All estates share trusted methodology, but environmental effects still differ. A prospective, spatially independent inspection sample with Middle recorded explicitly is the highest-value next test. The downstream polarized inspection system is outside scope.
