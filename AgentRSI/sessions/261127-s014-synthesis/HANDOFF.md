# Session 261127-s014-synthesis

## State of play

Final frozen submission: **source_prior**, single offline CPU pipeline.py and threadpoolctl==3.8.0. Independent source-estate full-design fits supply the equally weighted mean of12 standardized optical coefficients as the centre of the target ridge prior.49 local features plus intercept; all spatial/latent/height/mixture/SAR responses remain target-local. No estate-name branch, target label neighbors, Middle target, pseudo-label or score ensemble. All estates equally trusted.

Correct official80m lift A/P/S **+.159644/+.311518/+.391065**, composite **+.287409**, incumbent+.122984,coords+.043863,s009+.160430,s012+.272519,s013+.280623. V2 **+.226709**, delta incumbent+.107959/coords+.160247; estates+.059286/+.344207/+.276634. Passes the pre-V s013-relative rule: composite increment≥.005,≥2nonnegative estate increments, no estate loss>.015. No policy amendment. It still trails s012 Palong V by.018970 and s009 AirHitam by.006227. Incremental parent differences remain noisy/adaptive.

**Critical validation correction:** prior reports used L-only block origins. Current CLI/task require L∪U. Old folds share original80m blocks with training for82.9% AirHitam/100% Palong validation rows; Serting unchanged. New CLI reruns and all primary research protocols use union origins. Historical numbers exactly reproduced separately as legacyL80. Never mix these metrics or call legacy folds current-official. Correct L block counts26/18/13.120m stress aggregates whole original80m blocks by centres,160m groups are2x2 original cells; quadrants remain block-disjoint.

Frozen701–720: source_prior composite wins20/20 vs every parent, incumbent and coords. Estate means improve all comparators; wins incumbent59/60,s01357/60. Palong one incumbent loss−.009656; mean delta incumbent A/P/S+.101805/+.072809/+.209586. Conditional4000block-bootstrap composite95% delta incumbent[+.081272,+.232117],coords[+.161696,+.303367],s013[−.001351,+.014052]. Palong individual incumbent interval crosses0. Repeats reuse labels.

Binary Recall25 **48.2/65.2/74.6%**; including Middle capacity **48.2/63.6/73.1%**. AirHitam majority capture remains unmet. Serting0/10 positives80–150m from training positives found at25%; distant-pocket weakness. Source_prior beats s013 in every21broad-protocol estate cell, but Palong160m-buffered quadrant lift−.014195 below coords+.016275: no robust extrapolation guarantee.

Final selfcheck/CV/import pins/fresh subprocess contract pass. Clean package matches immutable V2 exactly in57fits(maxdelta0). Runtime≤.344s/estate,peakRSS≤273.2MiB.4473 exact-order U rankings exported.10tables exact7220keys/noNaNs/duplicates/forbidden columns/symlinks. Runtime datasets only **triage_v12,dino_sar**. No new extraction/GPU/GEE. Two V queries used,13remain; no more V tuning warranted.

## What I tried

All values below are corrected union-anchored official80m A/P/S and composite. Full35recipe table,paired incumbent/coords/parent deltas and1334completed estate/protocol records: report/EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,REPORT.md. OOF artifacts research/results/MODEL_PROTOCOL_ESTATE.npz. Staged sources actually rerun; no unavailable-parent reconstruction.

| Recipe / mechanism | AirHitam | Palong | Serting | Composite |
| --- | --- | --- | --- | --- |
| incumbent | 0.014687 | 0.219310 | 0.134956 | 0.122984 |
| coords | 0.014041 | 0.039323 | 0.078224 | 0.043863 |
| s009 | 0.030146 | 0.252671 | 0.198473 | 0.160430 |
| s012 | 0.151551 | 0.306681 | 0.359326 | 0.272519 |
| s013 | 0.151070 | 0.312298 | 0.378503 | 0.280623 |
| grouped | 0.145857 | 0.322567 | 0.360088 | 0.276171 |
| signs | 0.159112 | 0.310892 | 0.377704 | 0.282569 |
| spectral_protect | 0.152234 | 0.308826 | 0.381969 | 0.281010 |
| anchor_background | 0.158663 | 0.299014 | 0.346518 | 0.268065 |
| change_context | 0.098280 | 0.281232 | 0.377356 | 0.252289 |
| date_context | 0.110472 | 0.304730 | 0.362917 | 0.259373 |
| domain_optical | 0.139737 | 0.325675 | 0.369093 | 0.278168 |
| inner_C | 0.153676 | 0.306717 | 0.352655 | 0.271016 |
| source_prior | 0.159644 | 0.311518 | 0.391065 | 0.287409 |

Other controlled families: grouped+signs,full/spectral context projections,strong context regularization,block-balanced weights,additive spatial/spectral shape,full-context importance weighting,inner context-complexity selection,heterogeneity/crossfit/jackknife-stable source priors. Extra complexity fails consistent paired gains. Real-source versus zero/permuted controls supports transferred source information; one permutation is not a p-value. MainChecklist dead ends stayed closed.

Attribution: adopt s012 uniform shrunk quadratic optical/AlphaEarth/physical-floor height/mixture design,C=.15; s013 October SAR context/response-sharing hypothesis; s009's architectural source/target separation lesson. New modular prior freezes source response instead of simultaneously fitting shared/local blocks. Retire s009 nested offsets/name branch,s013 simultaneous duplication,ensemble scoring,V gates and unearned extra features. Collinearity remains: SD/edge correlations≈.947/.970/.983; final qNDVI negative all estates but deblur_NDMI positive conditional despite negative marginal d. Do not claim biological sign correction.

## Dead ends

- MainChecklist closed means-only SAR/SMOTE/model grids,label-neighbor leakage,Middle weak positives remain closed.
- Legacy L-only validation geometry is unsuitable as current primary. Repeating legacy metrics without origin audit is a known failure mode. Do not compare old reported numbers against new CLI as if same folds.
- Grouped optical/sign constraints/projection/healthy-background penalty/block weighting/additive shape/date-change context/domain weighting do not surpass final robustly. Small local improvements are not a reason for additional noisyV micro-tuning.
- Inner4fold original-block C/context selection avoids name branches but gives unstable outer gains; context selection particularly loses buffered Serting. No generic inner-CV superiority claim.
- Source disagreement attenuation,leave-block-fold means,jackknife attenuation add complexity without consistent gains. Preserve simpler coefficient-prior checkpoint.
- Palong160m buffered-region extrapolation remains effectively no skill. s009 fails Palong there with3inner groups<5; other estates attempted,compositeNA. Never replace failed results with0 or average only successful estates.
- Optical inverse has a weak global footprint;103m SAR crops overlap adjacent80m holdouts. Coarser/buffered checks help but do not establish disease causality. Historical height is stand context.
- AirHitam still misses majority at25%; Serting distant-pocket positives remain problematic. Unsupported U covariates and quadratic low-score tails have unknown outcomes. No V-driven score clipping.

## Open leads, ranked

1. Prospective original80m-block inspections in all3estates. Include AirHitam misses,distant Serting/Palong pockets,Middle inspection costs,unsupported covariates and a random unranked audit sample.
2. Independent estate/year validation and accurate current crown imagery/registration. More same-label CV/V does not confirm field outcomes or invariant disease signal.
3. Source-only transfer under genuine domain shifts,with separate source estimation and union-origin geometry. Test conditional shift with new domains rather than marginal-density tuning.
4. Preserve conservative s012 and s013 plus final prior. Source-prior parent bootstrap intervals still cross0; independent confirmation has higher value than optimizing V deltas near sigma.005.

Essential runtime datasets **triage_v12,dino_sar**. Keep gee_quality,gee_canopy,optical_jitter,alphaearth for exact s009 auditing; dino_optical supports preserved older recipes. compact_triage,highres_canopy,crown_mixture duplicate consolidated values and are safe to archive after provenance needs. Do not mutate original dino_sar: alternatives use its other date/views. No datasets were deleted.

## How to resume

Read REPORT.md,validation_origin_audit.csv,selection.json,validation_plan.json and prior/MainChecklist.md. Final recipe frozen afterV2; no more model changes selected on701–720. No active GPU was started. Clean submission is prediction-exact to V2 in57fits. Only threadpoolctl==3.8.0 beyond base runtime. Scientific source and evidence carried under datasets/triage_v12/recipes/s014; restore reproduction/ to research/ and parents/ as needed. Dataset covariates themselves never contain risk scores or fitted disease weights.

```bash
agentrsi selfcheck --submission /app/submission
agentrsi cv --submission /app/submission
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /app/research/evaluate.py --models incumbent coords s009 s012 s013 source_prior --protocols official block120 block160 buffer40 buffer80 blockquad blockquad80 blockquad160
python /app/research/aggregate.py
python /app/research/confirmation_summary.py
OPENBLAS_NUM_THREADS=1 python /app/research/bootstrap.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/equivalence.py
OPENBLAS_NUM_THREADS=1 python -B /app/research/contract.py
OPENBLAS_NUM_THREADS=1 python /app/research/final_diagnostics.py
python /app/research/audit.py
python /app/research/plots.py
python /app/research/write_report.py
```

legacy_origin_probe.py exists solely for historical reproduction,never current primary. evaluator caches include source hashes; use --force for deliberate reruns. Sparse s009 failure may stop remaining protocols; separately attempt other estate/protocols. AP is step-function sklearn,not trapezoidal PR area. Research-only plotting dependency matplotlib3.10.7. No secrets/sealed data/banned repo accessed.

## Artifact index

- submission/pipeline.py,requirements.txt — clean single-file offline CPU nominee.
- datasets/triage_v12/recipes/s014/ — selected package,all audited parents,candidates,reproduction,evidence,README/checksums. No runtime dependency on research/prior.
- report/REPORT.md,HANDOFF.md,validation_plan.json,selection.json — conclusions,pre-V rule and frozen decision.
- report/LINEAGE_COMPARISON.csv,EXPERIMENTS.csv,EXPERIMENT_SUMMARY.csv,VERIFIER_SUMMARY.csv — unified L/V/paired evidence.
- report/validation_origin_audit.csv,spatial_split_audit.csv — consequential origin correction and original-block integrity.
- report/frozen_confirmation.csv,frozen_confirmation_long.csv,frozen_model_comparison.csv,bootstrap.csv,bootstrap_method.json,final_robust_validation.csv — stability and conditional uncertainty.
- report/feature_effects.csv,feature_sign_consistency.csv,feature_correlations.csv,model_coefficients.csv,final_coefficients.csv,source_prior_coefficients.csv,prediction_disagreement.csv,final_prediction_disagreement.csv — marginal/conditional/suppression/rank mechanisms.
- report/final_metrics.csv,inspection_budget_diagnostics.csv,error_by_positive_distance.csv,covariate_support.csv,unlabelled_ranking.csv/.parquet — triage/error/support/aligned4473rankings.
- report/verifier/v1_signs.json,v2_source_prior.json,verifier_history.json — two genuine V calls and budget.
- report/selfcheck.txt,official_cv.txt,contract_verification.json,package_equivalence.json,import_audit.json,dataset_audit.json,environment.json — final verification.
- report/lineage_comparison,inspection_curves,spatial_robustness.png/.pdf;literature/,literature_fetch.json — standalone figures and retrieved primary sources.
- research/evaluate.py,engine.py,make_candidates.py,aggregate.py,origin_audit.py,legacy_origin_probe.py,diagnose.py,final_diagnostics.py,bootstrap.py,confirmation_summary.py,clean_pipeline.py,equivalence.py,contract.py,audit.py,plots.py,write_report.py — reproduction; results/*.npz/logs and immutable checkpoint_v2 adjacent.
