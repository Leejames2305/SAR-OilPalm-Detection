## State of play

Session261125-s009-open final nominee is `/app/submission/pipeline.py`: quality NDVI + source-only local-vigor penalty + illumination radiometry + AlphaEarth2025 local context + registration-integrated optical texture. Incumbent untouched. Official80m CV lift **+.05056/+.28872/+.19847, composite+.17925**, incumbent+.14052; coordinates+.08959. Selected noisy V lift **+.177665**, delta incumbent **+.058915**, delta coords **+.111203**; estate deltas **+.029357/+.019394/+.127993**. All3 estates equally trusted.

All3 estates improve in all11 initial block assignments; mean composite delta incumbent+.035145; paired same-block bootstrap95%[+.01416,+.07225]. Palong individual interval crosses zero. Frozen seeds100–109 also improve all3 estates in all10 assignments, composite mean delta+.03307. These intervals are exploration-conditional; no prospective significance claim. Final wider blocks, buffer, shifted grid and contiguous quadrant checks improve all3 estates versus incumbent. AirHitam coordinates-only beats final in2/3 wider120m assignments (composite still improves). AirHitam region recall25 remains41.1%, so no field majority-recall guarantee.

Binary Recall25=51.8%/81.8%/68.7%; including visible Middle against inspection capacity=51.8%/74.2%/67.2%. 4473 aligned U rankings exported. Fresh guards/shape/dtype/finiteness/shuffle/duplicate/immutability checks pass; final selfcheck and officialCV pass. Runtime~1–2s/estate in selfcheck, peak fresh-worker memory<310MiB. Only threadpoolctl==3.8.0 is a non-base runtime import. Five V queries used,10 remain; do not resample cosmetic hashes.

The previous source pipeline and most advertised datasets were absent; only3 optical tables carried over. Reconstructed s008 is not byte-identical. Only4 tables needed at runtime: gee_quality,gee_canopy,alphaearth,optical_jitter. All assets below128MiB; no symlinks. Final packaging is prediction-equivalent to staged/scored jitter recipe; see verified_prediction_equivalence.json.

## What I tried

Per-estate lifts AirHitam/Palong/Serting, official80m GroupKFold. Evidence for every row: report/EXPERIMENTS.csv and research/<model>_block_80_0_metrics.csv plus OOFnpz. Paired incumbent/coords deltas are in EXPERIMENT_SUMMARY.csv.

| Idea | Official lifts A/P/S | Composite | Verdict |
|---|---|---|---|
| incumbent | +0.03097/+0.25564/+0.13496 | +0.14052 | reference |
| coords | +0.01079/+0.17976/+0.07822 | +0.08959 | required spatial comparator |
| optical | +0.04106/+0.27108/+0.16891 | +0.16035 | reconstructed prior; V positive |
| radiometry_shadow | +0.05059/+0.27522/+0.17638 | +0.16740 | V positive; checkpoint |
| alpha_contrast | +0.04054/+0.27727/+0.18360 | +0.16714 | contextual imagery useful |
| joint_foundation | +0.04925/+0.28347/+0.18840 | +0.17371 | V positive; checkpoint |
| spline_additive | +0.02199/+0.16525/+0.17549 | +0.12091 | reject: Palong loss |
| spline_rbf | +0.02069/+0.16777/+0.12216 | +0.10354 | reject: large loss |
| middle_gate | +0.04106/+0.26329/+0.17411 | +0.15949 | reject: Palong degradation |
| radiometry_chlorophyll | +0.04714/+0.26584/+0.17258 | +0.16185 | mixed/smaller benefit |
| radiometry_water | +0.04215/+0.26372/+0.17354 | +0.15980 | mixed/smaller benefit |
| source_radiometry | +0.05596/+0.29098/+0.19331 | +0.18008 | weaker than local texture |
| texture_all | +0.05019/+0.29517/+0.20808 | +0.18448 | extra dimensions unnecessary |
| texture_small | +0.05285/+0.29613/+0.19791 | +0.18230 | V positive, original displaced view |
| texture_aligned | +0.05100/+0.28789/+0.19828 | +0.17906 | corrected single view; smaller gain |
| texture_jitter | +0.05056/+0.28872/+0.19847 | +0.17925 | FINAL: corrected uniform registration integration |
| registration_ensemble | +0.05143/+0.29146/+0.19796 | +0.18028 | little gain; no V query |
| jitter_no_quality | +0.04839/+0.27739/+0.17835 | +0.16804 | quality-NDVI blend contributes |
| jitter_no_vigor | +0.04296/+0.28733/+0.19193 | +0.17407 | one-sided vigor contributes |
| jitter_no_spatial | +0.04079/+0.05989/+0.22100 | +0.10722 | Palong depends on coordinates |

New high-leverage mechanisms: independent annual foundation context and native10m optical canopy neighborhood. Model hyperparameters8PCs/.3 shrinkage/.5 ridge fixed for these directions. Tested7 scientific families including spatial shape, Middle gating and source transfer. Frozen DINOv2 GPU extraction attempted but did not complete due infrastructure failures; no GPU features or scientific result claimed.

Registration audit found GEE pixelCoordinates outputs centres(.5,1.5,...). First texture extraction treated centres as corners, yielding effective NW half-pixel view. Preserve that table and describe it honestly; final optical_jitter corrects origin and uniformly averages four(±5m,±5m) index shifts. Do not silently regenerate the old table as if scores were unchanged. Corrected integration has similar V but slightly lower L lift than NW view; selected for explicit geometry. Final code cleans unused research functions and caps threads; no new V query spent on packaging.

## Dead ends

- Prior checklist dead ends remain: SAR means/model sweeps, label-leaky spatial neighbors, Middle weak positives, generic high-capacity training on tiny positive samples. No reason to rerun s008 SAR encoders unchanged.
- Generic additive splines/RBF fail Palong strongly. Incumbent estate-name polynomial branch remains an explicit limitation. Prior inner selectors had also failed; no new V estate gate introduced.
- Middle negative-control gate and simple water/chlorophyll offsets do not improve robustly over stronger checkpoint. Source radiometry less valuable than canopy-neighborhood texture; SD/edge and normalized-excess-green directions flip across estates, so do not impose pooled disease signs.
- More texture dimensions and blending two registration recipes add little. Do not V-tune registration weights, percentile, masks, feature counts or score thresholds.
- GPU T4 startup and L4 upload/status failed; initial stop requests also timed out. Final retry confirms vm_name=null, no active VM;1541.845 GPU seconds charged. No confirmed GPU feature output. No credential-containing tool error URLs were copied into workspace. Do not treat infrastructure failure as a negative DINO result.
- Polynomial tail extrapolation can produce very negative scores for extreme NDVI outside L support (e.g.5 Palong U points beyond raw NDVI range). Labels unknown, so do not call these false negatives. covariate_support.csv documents support.

## Open leads, ranked

1. Independent prospective block inspection in all3 estates, especially AirHitam and Palong. Include Middle capacity and inspect unsupported-covariate trees. Conditional CV/bootstrap does not replace new labels.
2. New-season/finer crown imagery and registration validation.10m S2 /20m red-edge mix9m-spaced palms. Current dip/contrast signals describe condition/context, not pathogen causality. Test consistent canopy geometry before any registration-weight tuning.
3. Foundation-context interpretation: prior-year AlphaEarth may encode environment, planting/management or canopy history. Need genuinely independent estate/year validation and spatial counterfactuals, not another local embedding-dimension sweep.
4. Generalizable complexity/extrapolation control with more estates. Explicit name branch inherited; inner selectors and generic splines failed. Constrain or safeguard spectral tails only with validation evidence.
5. Retry frozen DINO/remote-sensing patch encoders if broker healthy. Existing image chips contain no labels. GPU script staged but not completed; finish label-free extraction, PCA all-covariates allowed, supervised fits remain block-honest. Stop runtime and verify teardown.

## How to resume

Read report/REPORT.md and prior/MainChecklist.md first. Final scientific recipe frozen; additional seeds did not change model selection. Commands:

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/common.py /app/submission/pipeline.py packaged
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/final_diagnostics.py
python /app/research/audit_datasets.py
python /app/research/write_report.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/plots.py
```

Runtime package only threadpoolctl==3.8.0; matplotlib3.11.2 is research-only. Base libraries pinned by environment. GEE extraction scripts read service-account path from environment and never print secrets. Runtime pipeline reads features only via ctx datasets_dir, never data/rasters/prior/network. Source-only transfer sees only ctx other_labeled and block-crossfits offsets; target residuals use nested block-disjoint predictions.

Frozen V stage `/shared/s009_jitter` + `/shared/s009_jitter_data`, hash50f7ba4e03033ec4d33ece71feaa8f6d8156c07846283b483a025f7c1e6d145f. Final CPU/thread/cleanup packaging gives same predictions; do not requery merely for new noise. Use new V queries only for genuinely distinct L-verified mechanisms. For extraction reproduce source mosaics via datasets/optical_jitter/extract_optical_chips.py then registration_texture.py. Those scripts recreate research/optical_chips as needed. AlphaEarth extract.py downloads2023/24/25; final uses2025 local contrast only.

## Artifact index

- submission/pipeline.py,requirements.txt,optical.py,directions.py,joint_foundation.py,texture_jitter_helper.py — cleaned offline CPU nominee.
- datasets/gee_quality,gee_canopy,alphaearth,optical_jitter — required exact-key feature tables, manifests, extraction scripts. gee_s2_stress,optical_texture,optical_aligned retained for reuse/audit; no symlinks or forbidden columns.
- report/REPORT.md,EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,VERIFIER_SUMMARY.csv,verifier_history.json,v_*.json — full methods, outcomes and5-query chronology.
- report/final_metrics.csv,inspection_budget_diagnostics.csv,error_by_distance.csv,covariate_support.csv,feature_effects.csv,feature_sign_consistency.csv — diagnostics and limitations.
- report/unlabelled_ranking.csv/.parquet —4473 exact aligned scores with within-estate rank and illustrative top25 inspection flag.
- report/final_robust_validation.csv,robust_validation.csv,repeat_summary.csv,frozen_confirmation.csv — broader spatial/confirmation evidence.
- report/dataset_audit.json,contract_verification.json,verified_prediction_equivalence.json,environment.json,literature_fetch.json — audit hashes/imports/runtime/provenance.
- report/inspection_curves.png/.pdf,feature_signs.png — standalone research figures; no new label information.
- research/common.py,bootstrap.py,robust.py,make_ablations.py,contract.py,final_diagnostics.py,write_report.py,plots.py — evaluators/report reproduction; saved metrics/OOF/bootstrap JSON/logs alongside.
- research/optical_chips,extract_alphaearth.py,extract_optical_chips.py,extract_texture.py,registration_texture.py,dino_gpu.py — label-free imagery assets/extraction and attempted frozen GPU script. No verified DINO output.
- research/final_selfcheck.log,final_cv.log,contract.log,packaged.log — pre-exit verification.
