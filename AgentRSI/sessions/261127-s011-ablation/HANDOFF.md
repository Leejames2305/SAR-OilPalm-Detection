# Session261127-s011-ablation

## State of play

Final nominee `/app/submission/pipeline.py`: uniform differential-ridge quadratic NDVI/coordinates +12 physical optical features +8 training-only AlphaEarth2025 local-contrast PCs. No estate-name branch, patch embeddings, source pooling, Middle labels, pseudo-labels or label-neighbor term. All estates equally trusted. Runtime one 3.89 MiB table `/app/datasets/compact_triage/features.parquet`, only non-base pin threadpoolctl==3.8.0.

Official 80m lift **+0.067653/+0.351565/+0.337580, composite+0.252266** versus incumbent+0.140519, coordinates+0.089593. Final noisyV **+0.182332**, delta incumbent+0.063581, coords+0.115870; estateV+.039137/+.334900/+.172958. AirHitam V effectively ties incumbent/coords; clear gains are Palong/Serting. Two V queries used,13 remain. Do not resubmit cosmetic hashes.

Frozen repeat300..309 final beats incumbent and coords in 30/30 estate cells. Paired4000-block conditional bootstrap composite95% delta incumbent[+0.05320,+0.17716], coords[+0.08742,+0.21849]. Individual AirHitam/Palong incumbent intervals cross0; repeated assignments are not independent new data. All broader protocols beat incumbent in all estates; AirHitam120m loses about.01285 to 15-NN. Binary Recall25=55.4/75.8/74.6%; counting Middle capacity=55.4/68.2/70.1%. AirHitam confirmation Recall25 averages49.8%, so dependable majority capture is not established.

Final selfcheck/CV/contract pass; runtime peak<237 MiB.4473 aligned rankings saved. All 12 datasets~42 MiB, exact7220 keys and zero NaNs/duplicates/forbidden columns/symlinks. No GPU or GEE extraction. Previous stronger nominee code was absent: do not claim to beat its user-reportedV~+.216825.

## What I tried

Values below are official 80m A/P/S, composite; all paired deltas vs incumbent and coords in report/EXPERIMENTS.csv. Code research/model.py; OOF research/results/MODEL_PROTOCOL.npz.

| Recipe | AirHitam | Palong | Serting | Composite |
| --- | --- | --- | --- | --- |
| incumbent | +0.03097 | +0.25564 | +0.13496 | +0.14052 |
| coords | +0.01079 | +0.17976 | +0.07822 | +0.08959 |
| coords_poly | +0.01164 | +0.14273 | +0.05890 | +0.07109 |
| coords_rbf | +0.00763 | +0.18290 | +0.10393 | +0.09816 |
| optical | +0.09543 | +0.30998 | +0.18424 | +0.19655 |
| quality | +0.06907 | +0.29985 | +0.15361 | +0.17418 |
| local | +0.07355 | +0.27942 | +0.18322 | +0.17873 |
| alpha | +0.08285 | +0.35068 | +0.22686 | +0.22013 |
| wide | +0.06468 | +0.33782 | +0.22782 | +0.21010 |
| wide_pool | +0.06639 | +0.32666 | +0.22310 | +0.20538 |
| core_alpha | +0.05785 | +0.32344 | +0.21861 | +0.19997 |
| alpha_stronger | +0.06880 | +0.36788 | +0.27826 | +0.23832 |
| strong_middle_cov | +0.06880 | +0.36660 | +0.28110 | +0.23883 |
| strong_trans_cov | +0.06728 | +0.37014 | +0.28204 | +0.23982 |
| strong_linear | +0.06880 | +0.29743 | +0.25476 | +0.20700 |
| strong_full2 | +0.04736 | +0.36788 | +0.27826 | +0.23117 |
| strong_shrunk2 | +0.06174 | +0.35166 | +0.33572 | +0.24971 |
| uniform_noalpha | +0.07739 | +0.32058 | +0.24930 | +0.21576 |
| uniform_no_optical | +0.01433 | +0.30284 | +0.18896 | +0.16871 |
| uniform_core | +0.04374 | +0.31814 | +0.31884 | +0.22691 |
| uniform_nojitter | +0.05577 | +0.29632 | +0.24028 | +0.19746 |
| uniform_noquality | +0.03729 | +0.31167 | +0.31209 | +0.22035 |
| uniform_no_sde | +0.04626 | +0.29564 | +0.28772 | +0.20987 |
| uniform_plus_dino | +0.04691 | +0.36323 | +0.32638 | +0.24551 |
| uniform_axis | +0.06159 | +0.37017 | +0.33630 | +0.25602 |
| uniform_axis_only | +0.03240 | +0.33495 | +0.24352 | +0.20362 |
| uniform_perm_axis | +0.00960 | +0.30052 | +0.16327 | +0.15780 |
| uniform_noedge | +0.05387 | +0.34972 | +0.32973 | +0.24444 |
| uniform_nosd | +0.05667 | +0.32763 | +0.31808 | +0.23413 |
| uniform_nonirv | +0.06552 | +0.36125 | +0.33733 | +0.25470 |
| uniform_nogrvi | +0.06765 | +0.35157 | +0.33758 | +0.25227 |

Exploratory repeat1/2 and far-field checks selected uniform quadratic shrinkage. Initial frozen repeat100..109 verified compact vs inherited branch; repeat200..209 rejected source-axis complexity; fresh repeat300..309 verified the GRVI crown deletion (all 10 composite wins,25/30 estate wins, mean estate deltas+.00798/+.01526/+.00635). Two V queries only, report/v_compact.json and v_pruned.json. Final packaging is exactly equivalent to V2 on all 76 covariates/all 7220 keys,3 full fits and 15 official folds; report/package_audit.json. Final pipeline file SHA256:a0641192bd99c08e063899900ad81f785681dc0475f8457106369e2d2a0ce5c8.

## Dead ends

- Keep prior/MainChecklist and prev_HANDOFF closed leads closed: means-only SAR/model sweeps, leaky spatial stacking, Middle weak positives, unadapted pooling and local nested selector searches.
- Wider DINO patch blocks do not consistently improve this compact recipe; retain tables for provenance but no runtime dependency.
- Five-feature physiology pruning, removing local contrast/registration texture, or removing SD/edge loses lift. Sign flips do not justify automatic deletion from a joint contextual model.
- Source pooled augmentation and source-only/permuted axes do not earn added complexity; replacing target physical block with source axis is worse. All source labels equally trusted; failures are domain/regularization effects.
- Including Middle/all-target covariates in PCA changes results only slightly and is not required. No supervised Middle process was reopened.
- Uniform linear loses Palong signal, full unshrunk quadratics lose AirHitam robustness. Uniform.3 quadratic scaling removes the estate branch with stronger overall margin.
- Dropping NIRv or one edge/SD member has inconsistent broader effects. Crown-GRVI is the one confirmed prune.
- Palong unsupported-covariate negative score tail persists. Unknown outcomes; do not claim detected false negatives or retune tail clipping on V.

## Open leads, ranked

1. Prospective block inspections all 3 estates; include Middle costs, weak-support trees and a random unranked audit sample. AirHitam is the main uncertainty.
2. Independent estates/years and accurate crown registration. Native10/20m S2 mixes9m-spaced crowns; upsampling adds no physical detail.
3. Recover prior stronger nominee source in a future authorized carryover and compare paired, rather than treating the narrative as exact code. This session improved the supplied incumbent, not that unavailable nominee.
4. Only with independent confirmation: SAR-native representation or covariate-support/extrapolation control. Additional localCV/V tuning has diminishing value.

## How to resume

Read report/REPORT.md, selection.json, validation_plan.json and prior/MainChecklist.md. Final model is frozen. Do not spend V on formatting or a cosmetic hash. No active GPU job. Scientific implementation research/model.py CONFIGS['uniform_nogrvi']; final uses only compact_triage via ctx. Sources remain in four original datasets, and compact_triage/consolidate.py reproduces the exact runtime subset without labels.

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 python /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python /app/research/package_audit.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
python /app/research/audit.py
```

Report contains full rerun commands. Research plotting only requires matplotlib==3.11.0 (not runtime). Boundary ties use fractional credit; shuffled target/duplicate comparisons use1e-10 tolerance for BLAS roundoff. Same-input runs are deterministic. Do not touch prior reports/checklist without authorization.

## Artifact index

- submission/pipeline.py,requirements.txt — final offline CPU nominee.
- datasets/compact_triage/features.parquet,MANIFEST.md,consolidate.py,source_checksums.json — single runtime table and reproduction/provenance.
- datasets/gee_quality,gee_canopy,optical_jitter,alphaearth — source features/manifests/scripts; other carried tables retained unused, all assets~42 MiB.
- report/REPORT.md,HANDOFF.md,selection.json,validation_plan.json — complete findings and decisions.
- report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,OFFICIAL_EXPERIMENTS.csv — all configurations/protocols/paired deltas.
- report/final_robust_validation.csv,frozen_confirmation.csv,repeat_summary.csv,pruning_confirmation.csv,bootstrap.json — confirmation and uncertainty.
- report/v_compact.json,v_pruned.json,verifier_history.json — both noisy V observations and query usage.
- report/final_metrics.csv,inspection_budget_diagnostics.csv,error_by_distance.csv,covariate_support.csv — triage costs/errors/support.
- report/unlabelled_ranking.csv/.parquet —4473 exact-order scores and illustrative estate top25% allocation.
- report/feature_effects.csv,feature_sign_consistency.csv,model_coefficients.csv,spatial_split_audit.csv — feature sign/conditional coefficients and fold separation.
- report/dataset_audit.json,contract_verification.json,packaging_equivalence.json,package_audit.json,environment.json — integrity/runtime provenance.
- report/inspection_curves,feature_signs,spatial_robustness .png/.pdf — standalone figures. literature_* — primary pages/publisher metadata.
- research/model.py,evaluate.py,bootstrap.py,diagnostics.py,coefficients.py,contract.py,equivalence.py,package_audit.py,audit.py,plots.py,write_report.py — reproducible experiment/evaluation tooling.
- research/results/*_*.npz,metrics.csv and research/*.log — saved predictions and run evidence.
- research/candidates/compact and pruned — both V-scored source packages (second uses4 source tables; exact one-table final equivalence proved).
