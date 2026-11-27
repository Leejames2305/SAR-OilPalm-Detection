# Session 261127-s013-synthesis

## State of play

Final frozen submission: **pooled_sar**, one CPU/offline pipeline.py plus threadpoolctl==3.8.0. Uniform s012 compact optical/AlphaEarth/height/mixture design + s010 shared-optical/local-deviation architecture and October SAR8PCs.49 local predictors per estate,12 shared optical,162 total classifier columns, C=.15. No estate-name branch, label-neighbor feature, Middle target or score averaging. All three estates equally trusted/weighted.

Official lift A/P/S **+.075808/+.409951/+.378503**, composite **+.288087**; incumbent+.140519,coords+.089593,s010+.255805,s012+.269549. V **+.220382**, delta incumbent+.101632/coords+.153921; estates+.063217/+.338320/+.259611. Eight queries used,7 remain; no more V tuning after V8. No GPU/GEE work launched.

Selection is exploratory and V-informed. It loses Palong V vs s012 by.024857 and Serting V vs s010 by.046352; +.003557 composite vs s010 is below scorer noise. **It fails the provisional .015 checkpoint-loss guard; selection.json explicitly discloses the post-V policy amendment to the task's incumbent-relative criteria. Do not call this a passed preregistered all-parent non-degradation test.** Conservative s012 and all stronger per-estate references are preserved.

Reserved20 block assignments401–420:60/60 estate wins versus incumbent and coords; mean extra lift over s012 A/P/S+.008926/+.004100/+.036063; s012 estate wins17/20,13/20,20/20. Composite s012 wins20/20. Conditional4,000-block bootstrap composite95% delta incumbent[+.081769,+.206256],coords[+.119152,+.246979]; intervals vs s010/s012 cross0. Repeats reuse labels; AirHitam individual incumbent interval crosses0.

Binary Recall25 **48.2/74.2/76.1%**; counting Middle capacity **48.2/71.2/74.6%**. AirHitam majority capture remains unmet at25%. Palong whole-block quadrant+80m buffer loses.016145 to incumbent.160m buffered regions remain positive but can leave3 binary training rows. Sparse-tail Serting region skill approaches0. Unknown U outcomes remain unknown; no field recall/precision guarantee.

Final officialCV/selfcheck/pins/fresh-process contract pass. Clean package matches V8 in45 fits exactly. Runtime≤0.362s/estate,RSS≤275.0MiB.4473 exact-order rankings saved.10 tables with exact7220 keys, no NaNs/nonfinite/forbidden feature columns/symlinks;~49MiB including preserved sources. Only runtime tables triage_v12 and dino_sar. Pipeline SHA256 `71f416f1b15e6649b746f48a1f80e141817f8f01d4efab5403fbb4a2bc12843a`.

## What I tried

| Recipe | AirHitam | Palong | Serting | Composite | Delta incumbent | Delta coords | V composite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | +0.030965 | +0.255637 | +0.134956 | +0.140519 | +0.000000 | +0.050926 | +0.118751 |
| coords | +0.010793 | +0.179763 | +0.078224 | +0.089593 | -0.050926 | +0.000000 | +0.066462 |
| s009 | +0.050556 | +0.288720 | +0.198473 | +0.179250 | +0.038730 | +0.089656 | +0.176790 |
| s010 | +0.105714 | +0.381314 | +0.280387 | +0.255805 | +0.115286 | +0.166211 | +0.216825 |
| s011 | +0.067653 | +0.351565 | +0.337580 | +0.252266 | +0.111747 | +0.162673 | +0.182472 |
| s012 | +0.066466 | +0.382857 | +0.359326 | +0.269549 | +0.129030 | +0.179956 | +0.203411 |
| s012_bag | +0.087399 | +0.386420 | +0.328326 | +0.267382 | +0.126862 | +0.177788 | +0.213993 |
| uniform_sar | +0.072954 | +0.399129 | +0.370655 | +0.280913 | +0.140393 | +0.191319 | +0.208825 |
| both_offset | +0.070818 | +0.394637 | +0.369257 | +0.278237 | +0.137718 | +0.188644 | +0.201727 |
| pool_fullopt | +0.076162 | +0.392518 | +0.370768 | +0.279816 | +0.139297 | +0.190223 | +0.203185 |
| s010_curvature | +0.095185 | +0.391621 | +0.320584 | +0.269130 | +0.128611 | +0.179536 | +0.214764 |
| sar_bag | +0.091858 | +0.417466 | +0.339830 | +0.283051 | +0.142532 | +0.193458 | +0.206329 |
| legacy_complement | +0.103290 | +0.426680 | +0.318171 | +0.282713 | +0.142194 | +0.193120 | +0.214160 |
| legacy_protected | +0.098492 | +0.402958 | +0.315759 | +0.272403 | +0.131884 | +0.182810 | +0.213522 |
| pooled_sar | +0.075808 | +0.409951 | +0.378503 | +0.288087 | +0.147568 | +0.198494 | +0.220382 |

All values official80m A/P/S and paired deltas; V column uses supplied parent rewards/new observations. Evidence: report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,VERIFIER_SUMMARY.csv,REPORT.md; immutable authorized OOF predictions research/results/MODEL_PROTOCOL_ESTATE.npz. Main campaign has45 models. Staged parent code was available and actually rerun; no unavailable-source reconstruction claim.

- Capacity crossover: exact s010 except uniform.3 quadratic shrinkage; s010_curvature removes name branch, improves L, retains V tradeoff.
- Joint versus s009-style bounded block-OOF patch offsets: small stable local gains, no balanced parent V dominance.
- Compact optical partial pooling with self/permutation/no-share controls; combine with October SAR to form final. Some gains are changed regularization, not established transferred physiology.
- s01232-block probability bag crossed with SAR/optical patches: positive repeat means; V not balanced enough. No model score averaging in final.
- Missing quality/moisture/height/mixture measurements around uniform s010, joint or bounded: L Palong improves, V does not. This does not explain away parent divergence.
- Common aligned SAR basis transfer fails fresh Palong repeat mean and receives no V query. Raw-alpha scaling, quality blend and targeted redundant-feature pruning give mixed small gains.

Parent rank correlations s010/s012 A/P/S=.797/.896/.806; top25 disagreement15.35/9.22/9.04% of binary rows. s012 Palong qNDVI conditional sign is positive despite negative marginal d; final shared+local coefficient is negative. SD/edge correlate.947–.983 and have sign reversals. Coefficients are conditional predictive associations, not pathogen biomarkers; see signs/correlations/weights CSVs.

## Dead ends

- MainChecklist.md closed leads remain closed; do not repeat SAR means/algorithm grids, SMOTE, leaky label neighbors or Middle weak positives.
- No naive V-weighted ensemble/estate gates. Large L gains do not predict the estate V ordering: s010 stronger Serting V, s012 stronger Palong V.
- Optical NRG patch group, common latent SAR pooling, new bags and missing-family offsets do not earn final complexity. All eight exact V packages remain available.
- s009 fails legacy median80 Serting nested5-fold with<5 groups and Palong160m buffer PCA8 with3samples; s010 fails the latter. Attempted other estates are retained. Some generic offset prototypes fail one-class inner folds atbuffer80; these are NA, not zero/perfect scores. Initial empty-shared-block harness failure was corrected and rerun.
- Sparse-tail whole-block quadrant leaves only8 training positives for590/815 held-out Serting rows; near-zero skill reflects serious extrapolation/small-sample limits. Legacy median quadrants cut some80m blocks; do not call them block-disjoint.

## Open leads, ranked

1. Prospective whole-block inspections in all estates; include weak-support/distant Palong, AirHitam misses, Middle costs and random unranked audits. More same-label CV tuning cannot validate U outcomes.
2. Independent estates/years and accurate finer crown registration/current imagery.103m SAR context overlaps80m boundaries;160m region buffers help but do not remove the optical inverse's global tail.
3. Separate source-label information from regularization using independent domains. Do not interpret partial pooling as a universal disease signature.
4. Preserve conservative s012 and high-Serting-V s010. Confirm new mechanisms with new outcomes; no cosmetic V re-queries or estate-specific mixture-weight fit.

Essential datasets: **triage_v12,dino_sar**. Keep gee_quality,gee_canopy,optical_jitter,alphaearth,dino_optical for exact parent audits. compact_triage/highres_canopy/crown_mixture are redundant with consolidated values and safe to archive only after provenance needs are met. Do not mutate dino_sar in place; parent recipes use other saved views. Eight advertised datasets were absent at startup; no unnecessary re-extraction occurred.

## How to resume

Read REPORT.md,selection.json,validation_plan.json and MainChecklist.md. Final recipe is frozen after V8;7 queries remain but another micro-tuning round is not justified. No GPU session was started. Source-preserved portable variants and parents are **carried in datasets/triage_v12/recipes/s013/**, avoiding dependence on research paths surviving handoff. Restore reproduction/ into /app/research if needed; evaluate.py falls back to carried parents/candidates. Source feature files remain unchanged and source hashes audited.

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

Use --force for deliberate evaluator reruns; caches include implementation hashes. Full composites require3 estates; partial failures are NA. Runtime pin only threadpoolctl3.8.0; research plotting matplotlib3.10.7. Do not touch prior reports/checklist, /tests, sealed splits/labels, banned repo or credentials.

## Artifact index

- submission/pipeline.py,requirements.txt: clean selected offline model.
- datasets/triage_v12/recipes/s013/selected,parents,candidates,reproduction,evidence,README.md,checksums.json: all preserved sources/alternatives and reconstruction instructions. Each V-scored candidate has portable/ source and dependency metadata; no fitted labels or disease weights embedded.
- report/REPORT.md,HANDOFF.md,selection.json,validation_plan.json: conclusions, policy amendment, frozen decisions.
- report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,OFFICIAL_EXPERIMENTS.csv,VERIFIER_SUMMARY.csv,final_frozen_confirmation.csv,repeat_summary.csv: unified paired evidence.
- report/verifier/v1..v8 JSON;bootstrap.csv/.json;spatial_split_audit.csv: query results,conditional uncertainty and spatial support.
- report/model_coefficients.csv,final_optical_coefficients.csv,feature_effects.csv,feature_sign_consistency.csv,feature_correlations.csv,prediction_disagreement.csv: mechanisms/suppressors/signs/rank overlap.
- report/inspection_budget_diagnostics.csv,error_by_positive_distance.csv,covariate_support.csv,unlabelled_ranking.csv/.parquet: inspection costs,errors,support,4473 aligned U scores.
- report/lineage_and_triage.png/.pdf,spatial_robustness.png/.pdf,literature/: standalone figures/primary sources.
- report/selfcheck.txt,official_cv.txt,package_equivalence_final.json,contract_verification.json,dataset_audit.json,environment.json: verification,exact45-fit equality,48.9MiB data audit.
- research/evaluate.py,synthesis.py,legacy_extra.py,build.py,package.py,aggregate.py,diagnose.py,bootstrap.py,equivalence.py,contract.py,final_diagnostics.py,audit.py,plots.py,preserve.py,write_report.py: reproducible research; results/*.npz and logs remain local evidence.
