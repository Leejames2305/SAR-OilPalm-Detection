# Session 261127-s012-open

## State of play

Final selected model: `/app/submission/pipeline.py`, research recipe `fusion_control`. Uniform shrunk quadratic NDVI/coordinates +12 compact optical features +8 training-only AlphaEarth2025 local-contrast PCs +3 historical fine-canopy features with physical uncertainty scaling +9 label-free optical crown-mixture features. Balanced logistic C=.15, 41 encoded predictors. All three estates' labels are equally trusted. No estate-name branch, label-neighbor prior, source pooling, pseudo-labels or supervised Middle process.

One runtime table: `datasets/triage_v12/features.parquet` (about5.8MiB, 7,220×109, 88 covariates used). Only non-base pin: `threadpoolctl==3.8.0`. Official80m lift A/P/S **+.066466/+.382857/+.359326**, composite **+.269549**, versus incumbent+.140519, coordinates+.089593 and carried+.252266. Selected noisyV **+.205755**, delta incumbent+.087005 / coordinates+.139293; estateV+.050384/+.368128/+.198754.

Higher-noisy-V alternative is available, not missing: 32-block probability bagging reached **+.213993**, estateV+.049482/+.366903/+.225594. Its increments versus selected V1 were−.000902/−.001225/+.026839, so it failed the preregistered requirement for at least two nonnegative estate increments. Selected V1 is not the maximum observed noisy V. Full ready-to-run alternative source is carried in `datasets/triage_v12/recipes/block_bag.py`, with requirements adjacent. This does not prove bagging is worse; future independent confirmation can revisit it. Four V queries used, eleven remain. No more model/V tuning after V4. No active GPU.

Thirty frozen assignment repeats: 90/90 estate wins against incumbent; 80/90 against carried compact and29/30 composite wins. Conditional4,000-block bootstrap composite95% delta versus incumbent[+.06926,+.19020], coordinates[+.10078,+.23338], carried[−.00301,+.03404]. Individual AirHitam/Palong incumbent intervals cross zero. Repeats reuse the same labels.

Broader holdouts are strong except **Palong under an80m-buffered quadrant**: selected+.179059 versus incumbent+.211215 and carried+.224011. No support/uncertainty guard cured it. Recall25 binary is51.8/75.8/76.1%; counting Middle inspection capacity gives51.8/66.7/76.1%. AirHitam's majority capture is not dependable. Unknown U tail outcomes remain unknown.

Final selfcheck, officialCV, pins, fresh-estate contract and integrity audit pass. Runtime roughly.08–.10s per estate, peak processRSS below260MiB. All4,473 original-order rankings saved. Eighteen derived tables total about83MiB, exact7,220 keys, zero missing/nonfinite values, forbidden feature columns or symlinks. Source/hash evidence is in report/dataset_audit.json; pipeline hash is listed below.

## What I tried

Values are official80m A/P/S and composite. Full49-recipe table, paired deltas and wider protocols: `report/EXPERIMENTS.csv` and `REPORT.md`. OOF evidence: `research/results/MODEL_PROTOCOL_ESTATE.npz`.

| Recipe / mechanism | AirHitam | Palong | Serting | Composite | Verdict |
| --- | --- | --- | --- | --- | --- |
| base | +0.067653 | +0.351565 | +0.337580 | +0.252266 | Reproduced carried checkpoint |
| sar_all | +0.095451 | +0.358672 | +0.317165 | +0.257096 | AirHitam gain; wider Palong/Serting losses |
| ae_mae42 | +0.068507 | +0.350018 | +0.337808 | +0.252111 | Native GPU representation; no broad gain |
| ae_vicjun | +0.071982 | +0.341187 | +0.321046 | +0.244738 | Paired-date SSL; no broad gain |
| highres | +0.067017 | +0.357947 | +0.321461 | +0.248809 | Full standardized height PCs hurt distant Serting |
| highres_noise | +0.071007 | +0.360538 | +0.337581 | +0.256376 | Compact physical uncertainty floor useful |
| geometry | +0.058682 | +0.398680 | +0.328026 | +0.261796 | Mostly a Palong gain |
| mixture_deblur | +0.071173 | +0.360654 | +0.346523 | +0.259450 | Small gains from new optical preparation |
| pairwise | +0.062937 | +0.361965 | +0.343616 | +0.256173 | Gain largely explained by curvature control |
| curvature_control | +0.060842 | +0.368382 | +0.350406 | +0.259877 | Simple C=.15 head control |
| fusion_control | +0.066466 | +0.382857 | +0.359326 | +0.269549 | SELECTED; V+.205755 |
| phenotype_optical | +0.070843 | +0.379772 | +0.350809 | +0.267141 | No consistent added gain |
| support_refine | +0.080635 | +0.375343 | +0.349717 | +0.268565 | Does not cure distant-region failure |
| block_uncertainty | +0.070083 | +0.380076 | +0.360215 | +0.270125 | Complexity not earned |
| block_bag | +0.087399 | +0.386420 | +0.328326 | +0.267382 | V+.213993; preserved alternative, not selected |

New SAR extraction correctly reprojects WGS84 keys and October-to-June grids. Finite negative Yamaguchi dB retained; reprojection performed in physical power. All77m patches had100% valid support. Two masked denoising autoencoders and VICReg trained label-free on T4; no labels/coordinates/IDs/split assignments uploaded. Features, weights, logs and reproduction scripts are in datasets/sar_native. GPU stopped after about183 charged seconds.

New Meta/WRI historical fine-canopy data were accessible; NICFI was unavailable. Native S2 pixel-centre versus GeoTIFF-corner registration was independently probed and corrected. Gaussian spectral mixture inversion is label-free and does not create measured per-crown resolution. All new derived tables preserve exact keys. Neither credentials nor sealed information were exposed.

## Dead ends

- Keep prior/MainChecklist.md closed leads closed: no means-only SAR/SMOTE/algorithm-grid rescue without a new mechanism.
- Native77m,22-channel masked SAR reconstruction and paired-date VICReg do not consistently add disease ranking information to compact optical/AlphaEarth. This does not rule out every SAR foundation model.
- Unit-variance PCA of the full fine-canopy block amplified sub-accuracy Serting variation. Compact mean6/SD6/contrast6 with3/1/1m physical floors was useful. Height signs flip across estates; it is stand/management context, not a common biomarker.
- Extra optical-by-context interactions, bounded raw features, RBF basis, planting geometry and phenotype LabelSpreading did not consistently earn complexity.
- Smooth-AP loss lost. Pairwise loss partly changes effective regularization; C=.15 logistic control was competitive. Do not claim a confirmed ranking-objective mechanism.
- Feature-distance gates calibrated on training references over80m away, a smooth2-logit bound, and paired block-bootstrap coefficient-variance gates did not fix Palong's distant-region failure.
- Probability bagging improved all three mean frozen-repeat results, but its V increment was Serting-only and broader protocols were mixed. Preserve the source; keep the frozen conservative selection decision explicit.
- Inversion has a weak global influence tail. Uninverted point features also fail buffered Palong, so inverse overlap is not the sole explanation. No V-driven tail clipping or Middle weak-positive reopening occurred.

## Open leads, ranked

1. Prospective80m block inspections in all estates: record Middle costs, sample weak-support/distant Palong, examine AirHitam majority uncertainty, and retain a random unranked audit sample.
2. Independently confirm the preserved32-block ensemble before replacing selected V1. Do not rerun cosmetic hashes to seek fresh noise.
3. Independent estates/years, current finer imagery and accurate crown registration. Historical height predates labels; native S2 mixes9m-spaced palms.
4. Genuine protection against changed conditional relationships in distant regions, evaluated with new labels. Feature-space novelty alone did not detect this reliably. Further localCV/V micro-tuning has diminishing value.

## How to resume

Read report/REPORT.md, selection.json, validation_plan.json and prior/MainChecklist.md. The final selected recipe is frozen. No active GPU. Runtime uses only triage_v12 via ctx. The higher-V ensemble source is explicitly retained in the carried dataset; do not repeat the older unavailable-source narrative.

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 python /app/research/build_candidate.py --name fusion_control
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/audit.py
OPENBLAS_NUM_THREADS=1 python /app/research/aggregate.py
```

Research/model.py implements experiments. pipeline_template.py/build_candidate.py reproduce V1; build_bag.py reproduces the alternative. Cached NPZs are keyed by recipe/protocol; delete matching artifacts if intentionally changing a recipe under the same name. Each run writes its own CSV; aggregate after independent runs finish. All source extraction scripts/manifests and runtime consolidation are carried under datasets. Research requirements are pinned separately; matplotlib3.11.0 is plotting-only. Do not edit prior reports/checklist without authorization or access the banned repo, /tests, sealed labels/splits, or credentials.

## Artifact index

- `submission/pipeline.py`, `requirements.txt`: selected offline CPU model.
- `datasets/triage_v12/features.parquet`, `MANIFEST.md`, `source_checksums.json`, `consolidate_runtime.py`: one runtime table; `recipes/block_bag.py` and requirements preserve the higher-V alternative.
- `datasets/highres_canopy`, `crown_mixture`: new historical fine-canopy and native optical preparation, exact keys, manifests/scripts/audits.
- `datasets/sar_structure`, `sar_native`, `planting_geometry`: rejected new representations plus GPU weights/logs/scripts; twelve older assets retained. Eighteen tables about83MiB total.
- `report/REPORT.md`, `HANDOFF.md`, `selection.json`, `validation_plan.json`: final conclusions and frozen decisions.
- `report/EXPERIMENTS.csv`, `EXPERIMENT_SUMMARY.csv`, `OFFICIAL_EXPERIMENTS.csv`: all49 configurations, estate/composite paired deltas.
- `report/objective_confirmation.csv`, `feature_confirmation.csv`, `uncertainty_confirmation.csv`, `final_frozen_confirmation.csv`, `bootstrap_fusion_control.json`: final confirmation and conditional uncertainty. Earlier rank/fusion confirmation files explore candidate heads.
- `report/verifier/v1_fusion.json`, `v2_height.json`, `v3_optical.json`, `v4_block_bag.json`, `verifier_history.json`: all four V calls.
- `report/inspection_budget_diagnostics.csv`, `error_by_positive_distance.csv`, `covariate_support.csv`, `feature_effects.csv`, `feature_sign_consistency.csv`, `model_coefficients.csv`, `feature_correlations.csv`, `spatial_split_audit.csv`: inspection costs, errors, support, signs/weights and folds.
- `report/unlabelled_ranking.csv` / `.parquet`: all4,473 exact-original-order scores and illustrative top25% flags.
- `report/inspection_curves`, `feature_signs`, `spatial_robustness` `.png`/`.pdf`: standalone figures; `literature/`: cached primary pages.
- `report/dataset_audit.json`, `contract_verification.json`, `environment.json`, `selfcheck.txt`, `official_cv.txt`: integrity, pins, fresh-process/runtime and final CLI evidence.
- `research/model.py`, `evaluate.py`, `confirm.py`, `aggregate.py`, `diagnostics.py`, `contract.py`, `audit.py`, `plots.py`, `write_report.py`, `write_handoff.py`, `requirements.txt`: reproducible authorized-L research.
- `research/extract_sar.py`, `native_sar_gpu.py`, `compress_native.py`, `extract_highres.py`, `extract_geometry.py`, `extract_optical_native.py`, `deconvolve_optical.py`, `consolidate_runtime.py`: preparation. Native optical mosaics/metadata remain under research/native_optical; GPU staging is outside /app.
- `research/candidates/fusion_control`, `highres_noise_control`, `mixture_deblur_control`, `block_bag`: V-scored portable packages and18-fit equivalence proofs. `research/results/*`: immutable OOF predictions and per-run metrics.

Pipeline SHA256: `c835ed59e3db34f38cf8485a5b832ee051b49241656081f8369347821ffbd5c8`.
