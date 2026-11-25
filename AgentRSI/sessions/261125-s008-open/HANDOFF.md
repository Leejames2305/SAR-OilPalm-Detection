## State of play

Session 261125-s008-open nominates `/app/submission/pipeline.py`, quality NDVI plus source-estate local-vigor penalty. Official 80 m CV lift **+.04228/+.27044/+.16971, composite+.16081**, incumbent+.14052; coordinates15-NN+.08959. V **+.151519**, delta incumbent **+.032769**, delta coords **+.085058**; estate V deltas **−.001335/+.015262/+.084380**. Two estates improve, AirHitam difference below noise scale. Runtime datasets only gee_quality and gee_canopy; `threadpoolctl==3.8.0` pinned. All three estates equally trusted. Incumbent untouched.

Repeated 11-assignment mean composite delta+.020325; same-block bootstrap95%[+.00224,+.04629]. Official-only interval[−.00472,+.05794] crosses zero. Both are conditional on exploration; do not overclaim prospective significance. AirHitam/Serting11/11 assignment gains, Palong9/11 (worst−.01103). All three regions and all 9 shifted/buffered/120m comparisons improve over incumbent.

Selfcheck/CV pass; guarded fresh workers pass network/source-read/write guards, shuffle/duplicates and input immutability. Peak~225MiB. Full target 4,473 rankings generated. Top25 binary recall50.0%/81.8%/65.7%; including visible Middle50.0%/74.2%/64.2%. AirHitam50% is not majority. Frozen additional seeds 100–109 improve composite10/10 (mean+.01755), but bootstrap95%[−.00188,+.04535] crosses zero. AirHitam/Serting10/10, Palong6/10 estate wins. Four V queries remain; never resample cosmetic hashes.

## What I tried

Values official lifts AirHitam/Palong/Serting; detailed per-protocol results in report/EXPERIMENTS.csv and summary; paired baselines included. Evidence scripts/logs under research.

| Idea | Estate lifts | Composite | Verdict |
|---|---|---|---|
| Incumbent | +0.03097/+0.25564/+0.13496 | +0.14052 | Reference |
| Coordinates-only 15-NN | +0.01079/+0.17976/+0.07822 | +0.08959 | Required spatial comparator |
| Reconstructed Bayesian optical averaging | +0.03929/+0.25919/+0.14734 | +0.14861 | V lost AirHitam |
| VICReg SAR patch, 8 PCs × .3 | +0.04108/+0.29775/+0.13411 | +0.15765 | Serting repeat/region loss |
| Masked-AE SAR patch,32 PCs × .3 | +0.04065/+0.23578/+0.16232 | +0.14625 | Palong repeat loss |
| Shared NDVI slopes, estate spatial terms | +0.03243/+0.25026/+0.14523 | +0.14264 | Small/unstable |
| Hierarchical red-edge physiology | +0.03157/+0.24578/+0.16096 | +0.14610 | Palong losses |
| Independent 2020 canopy structure | +0.02306/+0.27837/+0.13611 | +0.14585 | Repeat degradation |
| Inverse block sample-size weights | +0.02664/+0.24822/+0.13140 | +0.13542 | Reject |
| Target/train spatial density ratio | +0.02815/+0.22876/+0.14321 | +0.13337 | Reject |
| Transductive canopy graph residuals | +0.02979/+0.26007/+0.13422 | +0.14136 | Small/unstable |
| Cluster-sandwich predictive marginalisation | +0.03239/+0.25547/+0.13312 | +0.14033 | No consistent gain |
| Highest-NDVI canopy-pixel proxy | +0.05382/+0.26829/+0.15620 | +0.15944 | Palong/Serting repeat losses |
| Weak local-background contrasts | +0.03671/+0.26077/+0.13697 | +0.14482 | Local consistent; V AirHitam loss |
| Source-only local red-edge offset | +0.05420/+0.25800/+0.16219 | +0.15813 | V composite +.02472, AirHitam loss |
| Clipped source-only red-edge offset | +0.05538/+0.26388/+0.16315 | +0.16080 | Positive local CI; V AirHitam loss |
| Source-only VICReg SAR offset | +0.03585/+0.27377/+0.13725 | +0.14896 | V lost all estates |
| Joint source optical + SAR offsets | +0.05021/+0.28344/+0.15576 | +0.16314 | Local robust; V AirHitam loss |
| Strict-quality median NDVI replacement | +0.02689/+0.20734/+0.12367 | +0.11930 | Reject |
| Half original / half quality-p75 NDVI | +0.03408/+0.26717/+0.16230 | +0.15452 | V all three improve |
| Inner block Bayesian measurement/degree selection | +0.03552/+0.25650/+0.14791 | +0.14664 | Instability; no V |
| Inner block hard degree selection | +0.03443/+0.26717/+0.15848 | +0.15336 | Region Palong/Serting loss |
| Quality NDVI + symmetric .3 red-edge offset | +0.04571/+0.26052/+0.17037 | +0.15887 | V AirHitam loss |
| Final: quality NDVI + local-vigor penalty | +0.04228/+0.27044/+0.16971 | +0.16081 | Nominee |

Key new mechanism: baseline NDVI=.5provided+.5quality-p75. Source-only local NDRE contrast coefficient is learned with block-crossfitted baseline logit offsets; clip standardized contrasts±3. Final correction min(coef*z,0) penalises locally vigorous crowns only. Incumbent degree branch retained (AirHitam linear, other quadratic); all attempts to replace with inner-CV degree selection degraded robustness. No new estate gate from V. See REPORT.md for full recipe and caveats.

## Dead ends

- Label-free CNN SAR patch embeddings improve some local folds but V source-transfer+.105299 loses all estates. Adding them jointly to optics also loses AirHitam. Do not repeat the same two encoders without new imagery/mechanism.
- Uniform symmetric red-edge contrast offsets create strong Palong/Serting V gains but AirHitam material loss. Clipping±3 alone does not fix it. Final one-sided vigor penalty does.
- Strict-quality median NDVI replacement harms Palong; upper quantile plus original measurement is complementary. Rank/robust/reliability blending variants do not robustly beat the half raw blend.
-2020 canopy height has sign flips and repeat losses; not a disease signature. Block balance, density-ratio weighting, graph diffusion and uncertainty marginalisation have weak/inconsistent benefit.
- Inner block Bayesian degree/measurement averaging and hard complexity gates repeatedly degrade Palong/Serting or regions. Incumbent complexity remains a limitation.
- Do not use random-fold spatial priors, same-block residual labels, harmonisation+1000 a second time, or raw Yamaguchi dB as positive linear power. No need to re-extract prior window means.

## Open leads, ranked

1. Prospective independent block inspection, especially AirHitam and Palong, including Middle capacity. Repeated CV/bootstrap is exploration-conditional. AirHitam V difference is noisy; no equivalence claim.
2. Explain quality-p75 mechanism: disaggregate cloud/shadow mask versus window versus per-image percentile with independent imagery/labels; do not V-tune thresholds.
3. Generalizable complexity control using more estates/labels. Current inherited name branch is explicit, inner80m selectors failed. Do not substitute a weaker generic rule merely for aesthetics.
4. Genuinely finer crown imagery/geometry; native S2 pixels mix adjacent palms. Local-vigor penalty is condition triage, not pathogen causality.
5. New-season/far-field calibration and review-budget uncertainty; no25% majority guarantee for AirHitam.

## How to resume

Read report/REPORT.md and prior/MainChecklist.md; final code is frozen. Commands:

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/verify_contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/common.py /app/submission/pipeline.py quality_veto --repeat 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/robust_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/bootstrap_repeated.py quality_veto incumbent 5000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/predict.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/diagnostics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/plots.py
python /app/research/write_report.py
```

Stage only required gee_quality/gee_canopy data. Frozen V submission `/shared/s008_quality_veto`, datasets `/shared/s008_quality_local_data`. Do not edit/requery only to draw fresh noise. Source-only transfer uses all other estates L, always block-crossfitted offsets; target uses its own baseline labels. GPU already stopped. Label-free extraction scripts/manifests retained. No final fitting needs data/rasters/prior/network.

## Artifact index

- submission/pipeline.py, requirements.txt — frozen CPU/offline nominee; research/quality_veto.py exact copy; incumbent untouched.
- datasets/gee_quality, gee_canopy — required features/manifests/extract scripts; all 7,220 exact keys/noNaNs.
- datasets/sar_ssl — rejected but reusable VICReg/AE32PC features, extraction/training/assembly scripts, raw embeddings, weights/transforms; no torch runtime dependency.
- datasets/gee_structure — rejected2020 independent canopy structure. All datasets~68.1MiB, no symlinks; prior5 tables retained.
- report/REPORT.md; EXPERIMENTS.csv/EXPERIMENT_SUMMARY.csv; final_metrics.csv — methods and full paired comparisons.
- report/verifier_results.json, VERIFICATION_NOTES.md — noisy V chronology;11queries at freeze. Optical-average first result noted separately.
- research/*_oof.npz, *_bootstrap.json — fixed predictions, conditional paired bootstrap; bootstrap_repeated.py uses common resamples across assignments.
- report/inspection_budget_diagnostics.csv, error_by_distance.csv, feature effects and final_coefficients.csv; three PNG figures.
- report/unlabelled_ranking.csv/.parquet —4473 aligned scores, ranks, illustrative inspection flag.
- report/dataset_audit.json, contract_verification.json, environment.json — audits; research/final_selfcheck.log, final_cv.log.
- research/confirm_frozen.py/log — extra seeds 100–109 after freezing; report/frozen_confirmation.csv and research/final_confirm_vs_incumbent_confirm_repeated_bootstrap.json contain paired evidence; no changes made after this check.
