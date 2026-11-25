## State of play

Session261125-s010-open final nominee is `/app/submission/pipeline.py`: partially pooled optical condition, AlphaEarth2025 local contrast, frozen DINOv2 optical NRG160m and October SAR HH/HV/VV103m context. Single offline CPU file; no runtime GPU/network/raster dependency. Incumbent untouched. All estates equally trusted.

Official80m CV lift **+.105714/+.381314/+.280387, composite+.255805**, incumbent+.140524, coordinates+.089593. Selected noisy V **+.219732**, delta incumbent **+.100982**, delta coords **+.153270**; V estate lifts+.054320/+.291206/+.313670 and incumbent deltas+.017293/+.037900/+.247752. Six valid V queries used,nine remain; do not resubmit cosmetic hashes. Selection is exploratory and V-informed, not prospective significance.

All3 estates beat incumbent and coords in all21 assignments (official0, exploration1–10, declared100–109). Composite paired80m block-bootstrap95% delta incumbent **[+.05898,+.18388]**, coords **[+.09321,+.23003]**; all individual intervals positive but Palong lower bound is only+.00138 and selection-conditioned. All3 estates improve both comparators in120/160m blocks,40m shifted grid,20m buffer,and contiguous quadrant holdouts.

Binary Recall25 **50.0%/72.0%/67.2%**; including Middle as capacity consumers **50.0%/63.6%/67.2%**. At30% capacity including Middle,55.4%/68.8%/70.1%. AirHitam majority recall at25% is not established. Rankings4473 rows aligned exactly, under report/unlabelled_ranking.csv/.parquet. Scores are uncalibrated priority values.

Final selfcheck and officialCV pass. Shape,float64,finiteness,immutability,shuffle,duplicate,determinism checks pass in fresh workers; peak<328MiB. Only non-base import threadpoolctl==3.8.0, pinned. Six runtime tables: gee_quality,gee_canopy,alphaearth,optical_jitter,dino_optical,dino_sar. All datasets~39MiB,exact7220 unique keys,zeroNaNs/forbidden columns/symlinks. Two GPU jobs completed and stopped; broker active_vm=null,242.558s charged. Clean packaging is prediction-equivalent to scored SAR stage in6 full/holdout comparisons (max difference0).

## What I tried

L lifts below are AirHitam/Palong/Serting. Full experiments and paired incumbent/coords deltas: report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv; each research/<model>_<protocol>_metrics.csv and OOFnpz is reproducible evidence. No prior nominee code was carried over; reconstruction was explicitly nonidentical.

| Idea | Official L lift A/P/S | Composite | Verdict / evidence |
| --- | --- | --- | --- |
| incumbent | +.03097/+.25564/+.13496 | +.14052 | reference; research/incumbent.log |
| coords15NN | +.01079/+.17976/+.07822 | +.08959 | required comparator; research/coords.log |
| reconstructed optical offsets | +.03769/+.25294/+.18041 | +.15701 | weaker than joint; references.log |
| joint physical+foundation+texture | +.09575/+.33167/+.22860 | +.21867 | V+.17172; joint.log,v_joint.json |
| frozen DINO NRG160 | +.09208/+.34621/+.23592 | +.22474 | V+.18344,AirHitam V−.0019 vs incumbent; dino.log,v_dino.json |
| partial pooling,moderate prior | +.12739/+.37016/+.25346 | +.25034 | V+.18840,all estates improve; transfer.log,v_transfer.json |
| pooling+DINO optical | +.12095/+.37737/+.25535 | +.25122 | V+.20971; recipe.log,v_transfer_dino.json |
| terrain / historical height | varies; tables in structure.log | +.21449/+.21564 | no robust gain; not runtime-required |
| generic spatial2 / spatial3 | varies; recipe.log | +.21204/+.21124 | capacity/extrapolation alternatives inferior |
| nested selector / conservative / mixture | varies; complexity.log | +.19430/+.20071/+.20725 | failed to replace inherited branch |
| Middle separate-process multitask | +.09575/+.33921/+.24363 | +.22620 | small vs joint,below transfer; ordinal.log |
| source-only / target vigor | varies; vigor.log | ~+.2185 | no gain |
| autumn optical level / change | varies; autumn_ablation.log | +.24013/+.24288 | below combined optical; reject |
| frozen June / display-change SAR | varies; sar.log | +.22030/+.23946 | inconsistent; reject |
| frozen paired51m SAR | +.12716/+.38230/+.25776 | +.25574 | official small gain but broader/repeat AirHitam loss; reject |
| frozen October103m SAR | +.10571/+.38131/+.28039 | +.25580 | FINAL,V+.21973; sar.log,v_sar_october.json |
| fixed equal sensor ensemble | +.11022/+.37343/+.27002 | +.25122 | V+.21581; no primary improvement; ensemble.log |
| final no coords / no AlphaEarth | tables in final_ablations.log | +.19837/+.22205 | contextual and spatial contributions quantified |
| final no optical patch / no quality blend | tables in final_ablations.log | +.25290/+.24331 | optical-patch incremental gain small; no dimension tuning |

Partial-pooling controls: no shared optical block+.21857,target-only duplicated block+.22757,one fixed permuted-source-label control+.22928,real pooling+.25034. Permuted labels and class weights are permuted together (initial research control corrected before reporting). Across11 assignments most additional real-source benefit is Palong; do not attribute all pooling gain to transferable disease physiology.

Two real frozen DINOv2 GPU extractions succeeded: optical4views and SAR5views. No labels/fine-tuning/GPU head. Official ViT-S/14 weights,source commit7764ea0f912e53c92e82eb78a2a1631e92725fc8; weights SHA256b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9. All7220 covariates PCA64/view,label-free; downstream8PC fits training-only. Source manifests/scripts/transforms saved. SAR raw patch support100%; no Yamaguchi dB mishandling. Optical corrected pixel-corner geometry preserved.

## Dead ends

- Prior MainChecklist and s009 closed means-only SAR/model sweeps,leaky spatial neighbors,Middle weak positives,unadapted pooling remain closed. This session's new mechanism was frozen patch representations plus domain-specific partial pooling.
- More patch views do not consistently help; four-view optical hurts AirHitam. June/change-display/paired SAR fail broader tradeoffs. Do not tune crop sizes/PCA counts/display ranges on V.
- Terrain and2020 canopy height add little on the stronger model. Autumn optical level/change underperform; no evidence that a generic optical time-series sweep would rescue them.
- Single nested degree selector,bootstrap-conservative selector,soft selector,spatial-only quadratic/cubic and clipping failed to outperform the inherited branch. Final still uses estate-name complexity inherited from incumbent; no new V estate gate.
- Middle separate-process auxiliary learning produces only small gains. One-sided source vigor contributes essentially nothing under the stronger joint recipe.
- Fixed equal sensor ensemble trails final primary lift; no V-weight sweep. Optical patch contribution in final L is only+.0029 composite,so do not overstate its independent evidence.
- NDVI polynomial tail extrapolation remains: Palong U outside training support can receive extreme negative scores. Their outcomes are unknown. See covariate_support.csv; no unsupported false-negative claim.
- Web tool search/open endpoint failed; primary papers/catalogue pages fetched directly with requests and saved. Initial ETH official-looking asset name was unavailable; actual author asset users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1 used. GLO30 is now deprecated; extracted terrain kept for reproducibility,not runtime.

## Open leads, ranked

1. Independent prospective spatial-block inspections across all3 estates,especially AirHitam and Palong; include Middle inspection cost and unsupported-covariate trees. Current adaptive CV/V evidence does not replace new outcomes.
2. Finer crown imagery and surveyed registration ground control. S2 native10m mixes9m-spaced palms; upsampled encoder inputs do not create crown detail. Separate condition/context from pathogen causality.
3. SAR-native pretrained encoders or a fixed radiometrically normalized encoder with independent estate/year confirmation. October RGB-pretrained SAR context helped most in Serting; June/change were inconsistent. Test a genuinely new sensor representation,not crop/display/PC-count tuning.
4. Generalizable capacity/extrapolation control with more estates and labels. The inherited AirHitam name branch remains; previous small-sample nested selectors failed. More validation domains are higher value than another local selector sweep.
5. Interpret shared optical physiology versus environmental/management effects using external holdout estates/years. Physical signs are consistent for NDRE contrasts/dip but flip for brightness,green/red,SD/edge,height and terrain. Source negative controls show some gain is regularization.

## How to resume

Read report/REPORT.md,report/selection.json and prior/MainChecklist.md. Final model is frozen; use V only for a genuinely new,L-verified mechanism. Do not query just for clean packaging/noise. The final script is byte-different but prediction-equivalent to verified /shared/s010_sar_october (data /shared/s010_sar_data),hashd74c38fcc62651eba70d5b76471c7151aa3130eb68f89d4f9ff60deca50bbf2c. V6/15 used. No active GPU VM.

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

Runtime pin only threadpoolctl==3.8.0. Research matplotlib==3.11.2 is not a runtime dependency. fit_predict reads only ctx datasets_dir;temp writes not needed. cpus respected via threadpool_limits. For GPU feature reproduction use scripts inside dino_optical/dino_sar; compression paths refer to completed job directories and need updating for a new broker job. Full source weights/commit and feature checksums pinned. Optical extract script writes the documented centre-origin raster; prepare_gpu.py corrects its corner before cropping. Do not silently substitute original NW-displaced texture features.

## Artifact index

- submission/pipeline.py,requirements.txt — clean final offline CPU nominee.
- datasets/gee_quality,gee_canopy,alphaearth,optical_jitter,dino_optical,dino_sar —6 runtime exact-key feature tables,manifests,reproduction scripts and PCA transforms where relevant.
- datasets/terrain_canopy,autumn —new independent covariates retained for reuse;gee_s2_stress,optical_texture,optical_aligned carried for provenance. All assets below128MiB.
- report/REPORT.md,EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,VERIFIER_SUMMARY.csv,verifier_history.json,v_*.json,selection.json —full evidence/selection history.
- report/final_metrics.csv,inspection_budget_diagnostics.csv,error_by_distance.csv,covariate_support.csv,feature_effects.csv,feature_sign_consistency.csv —triage,spatial errors,sign reversals/support.
- report/unlabelled_ranking.csv/.parquet —4473 aligned U scores,ranks,and illustrative top25 inspection allocation.
- report/final_robust_validation.csv,repeat_summary.csv,frozen_confirmation.csv,sar_october_bootstrap.json —broad spatial/paired uncertainty evidence.
- report/dataset_audit.json,contract_verification.json,verified_prediction_equivalence_radar.json,environment.json —data/import/runtime/prediction provenance. literature_*.html —primary sources.
- report/inspection_curves.png/.pdf,feature_signs.png/.pdf —standalone research figures.
- research/common.py,validate_milestone.py,frozen_confirmation.py,bootstrap.py,summarize.py,final_diagnostics.py,feature_analysis.py,contract.py,audit.py,write_report.py,plots.py —reproduction/evaluation.
- research/reference.py,joint.py,transfer.py,expanded.py,recipe.py,complexity.py,ordinal.py,vigor.py,sensor_ensemble.py,final_pipeline.py,final_ablation.py —scientific implementations;per-model metrics/OOFnpz saved alongside.
- research/prepare_gpu.py,dino_gpu.py,compress_dino.py,prepare_sar_gpu.py,dino_sar_gpu.py,compress_sar.py,extract_structure.py,extract_autumn.py,optical_chips —label-free source features and extraction logs.
- research/final_selfcheck.log,final_cv.log,contract.log,packaged.log,equivalence_radar.log —pre-exit verification. GPU raw outputs in /run/agentrsi/colab-io/out/job-0001-8dd9be and job-0002-451135 may not carry forward;derived tables and scripts do.
